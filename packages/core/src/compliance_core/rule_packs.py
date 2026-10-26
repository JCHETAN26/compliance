"""Rule-pack schema, validation and YAML import/export.

A rule pack is a versioned list of rules (see ``rule_packs/*.yaml``). Validation is
strict (unknown fields are errors) because packs are also user-imported YAML.

Trained-model support is not a user-settable flag: ``model_supported: true`` is accepted
only for a per-document rule whose ``contractnli_key`` is one of the 17 ContractNLI
hypotheses the model was trained on **and** whose ``hypothesis_text`` equals that
hypothesis verbatim (the built-in ``nda_standard`` pack is the authoritative mapping and
is itself checked against the official dataset in ``ml/tests/test_data.py``). Any other
rule must use the LLM path (handoff section 7.6).
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    model_validator,
)

from compliance_core.services.errors import ServiceError

MAX_YAML_BYTES = 256 * 1024
MAX_RULES = 200
DecidedStatus = Literal["satisfied", "violated", "not_mentioned"]
DocTypeName = Literal["contract", "spreadsheet", "policy", "other"]
CrossKind = Literal["contract_on_file", "required_protections", "term_covers_date"]
RegisterRole = Literal["vendor", "data_access", "contract_file", "renewal_date"]
Key = Annotated[str, StringConstraints(pattern=r"^[a-z0-9][a-z0-9_]{0,63}$")]
Title = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)]
LongText = Annotated[str, StringConstraints(strip_whitespace=True, max_length=4000)]
Hypothesis = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]
Version = Annotated[str, StringConstraints(pattern=r"^\d+\.\d+\.\d+$")]
Alias = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=120)]


class RulePackValidationError(ServiceError):
    code = "invalid_rule_pack"

    def __init__(self, message: str, errors: list[str] | None = None) -> None:
        super().__init__(message)
        self.errors = errors or [message]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class SubjectFilter(_Strict):
    column: RegisterRole
    in_: Annotated[list[Alias], Field(alias="in", min_length=1, max_length=20)]


class CrossDocumentSpec(_Strict):
    kind: CrossKind
    subject_filter: SubjectFilter | None = None
    required_rules: Annotated[list[Key], Field(max_length=20)] = Field(default_factory=list)
    date_column: RegisterRole | None = None


class RuleSpec(_Strict):
    key: Key
    title: Title
    description: LongText = ""
    hypothesis_text: Hypothesis
    scope: Literal["per_document", "cross_document"]
    applies_to_doc_types: Annotated[list[DocTypeName], Field(min_length=1, max_length=4)]
    severity: Literal["low", "medium", "high"]
    model_supported: bool = False
    contractnli_key: Annotated[str, StringConstraints(pattern=r"^nda-\d{1,2}$")] | None = None
    # Statuses that count as compliant. None = the pack does not define compliance for
    # this rule (findings report the status only).
    compliant_statuses: Annotated[list[DecidedStatus], Field(max_length=3)] | None = None
    cross_document: CrossDocumentSpec | None = None
    remediation: LongText | None = None

    @model_validator(mode="after")
    def _scope_consistency(self) -> RuleSpec:
        if self.scope == "cross_document":
            if self.cross_document is None:
                raise ValueError("cross_document rules need a cross_document specification")
            if self.model_supported:
                raise ValueError("cross_document rules cannot be model_supported")
        elif self.cross_document is not None:
            raise ValueError("only cross_document rules may have a cross_document specification")
        return self


class RegisterSpec(_Strict):
    columns: dict[RegisterRole, Annotated[list[Alias], Field(min_length=1, max_length=20)]]

    @model_validator(mode="after")
    def _required_roles(self) -> RegisterSpec:
        missing = {"vendor", "contract_file"} - set(self.columns)
        if missing:
            raise ValueError(f"register.columns needs {sorted(missing)}")
        return self


class RulePackSpec(_Strict):
    key: Key
    name: Title
    version: Version
    description: LongText = ""
    # Free-form provenance (dataset, URL, license); informational only.
    source: dict[str, str] | None = None
    # Field named ``register_`` (BaseModel already has ``register``); YAML key ``register``.
    register_: RegisterSpec | None = Field(default=None, alias="register")
    rules: Annotated[list[RuleSpec], Field(min_length=1, max_length=MAX_RULES)]

    @model_validator(mode="after")
    def _references(self) -> RulePackSpec:
        keys = [r.key for r in self.rules]
        if len(set(keys)) != len(keys):
            raise ValueError("rule keys must be unique within a pack")
        per_document = {r.key for r in self.rules if r.scope == "per_document"}
        for rule in self.rules:
            spec = rule.cross_document
            if spec is None:
                continue
            if self.register_ is None:
                raise ValueError(f"rule {rule.key!r}: cross_document rules need a register section")
            roles = set(self.register_.columns)
            unknown = [k for k in spec.required_rules if k not in per_document]
            if unknown:
                raise ValueError(
                    f"rule {rule.key!r}: required_rules {unknown} are not per-document rules here"
                )
            if spec.kind == "required_protections" and not spec.required_rules:
                raise ValueError(f"rule {rule.key!r}: required_protections needs required_rules")
            if spec.kind == "term_covers_date" and spec.date_column is None:
                raise ValueError(f"rule {rule.key!r}: term_covers_date needs date_column")
            for role in (
                spec.date_column,
                spec.subject_filter.column if spec.subject_filter else None,
            ):
                if role is not None and role not in roles:
                    raise ValueError(f"rule {rule.key!r}: register has no {role!r} column")
        return self


# --------------------------------------------------------------------------- built-ins


def builtin_pack_dir() -> Path:
    """Installed wheels carry the packs as package data; a checkout reads the repo copy."""
    packaged = resources.files("compliance_core") / "builtin_rule_packs"
    if packaged.is_dir():
        return Path(str(packaged))
    return Path(__file__).resolve().parents[4] / "rule_packs"


def builtin_pack_files() -> list[Path]:
    return sorted(builtin_pack_dir().glob("*.yaml"))


@lru_cache(maxsize=1)
def known_model_hypotheses() -> dict[str, str]:
    """ContractNLI key -> verbatim training hypothesis (from the built-in nda_standard)."""
    raw = yaml.safe_load((builtin_pack_dir() / "nda_standard.yaml").read_text("utf-8"))
    return {r["contractnli_key"]: r["hypothesis_text"] for r in raw["rules"]}


def model_support_errors(spec: RulePackSpec) -> list[str]:
    known = known_model_hypotheses()
    errors = []
    for rule in spec.rules:
        if not rule.model_supported:
            continue
        if rule.contractnli_key is None or rule.contractnli_key not in known:
            errors.append(
                f"rule {rule.key!r}: model_supported requires a contractnli_key of a trained "
                "hypothesis; custom rules are checked by the LLM (set model_supported: false)"
            )
        elif rule.hypothesis_text != known[rule.contractnli_key]:
            errors.append(
                f"rule {rule.key!r}: model_supported requires hypothesis_text to equal the "
                f"trained {rule.contractnli_key} hypothesis verbatim; edited text must use the "
                "LLM (set model_supported: false)"
            )
    return errors


# --------------------------------------------------------------------------- parsing


def _format_errors(exc: ValidationError) -> list[str]:
    out = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err.get("loc", ()))
        out.append(f"{loc}: {err.get('msg')}" if loc else str(err.get("msg")))
    return out


def validate_spec(data: Any) -> RulePackSpec:
    try:
        spec = RulePackSpec.model_validate(data)
    except ValidationError as exc:
        errors = _format_errors(exc)
        raise RulePackValidationError("The rule pack is invalid.", errors) from None
    errors = model_support_errors(spec)
    if errors:
        raise RulePackValidationError("The rule pack is invalid.", errors)
    return spec


def parse_yaml(text: str) -> RulePackSpec:
    if len(text.encode("utf-8")) > MAX_YAML_BYTES:
        raise RulePackValidationError(f"Rule pack YAML is larger than {MAX_YAML_BYTES} bytes.")
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        where = f" (line {mark.line + 1})" if mark is not None else ""
        raise RulePackValidationError(f"The YAML could not be parsed{where}.") from None
    if not isinstance(data, dict):
        raise RulePackValidationError("A rule pack must be a YAML mapping.")
    return validate_spec(data)


def spec_document(spec: RulePackSpec) -> dict[str, Any]:
    return spec.model_dump(mode="json", by_alias=True, exclude_none=True)


def dump_yaml(spec: RulePackSpec) -> str:
    return str(yaml.safe_dump(spec_document(spec), sort_keys=False, allow_unicode=True, width=100))


def content_sha256(spec: RulePackSpec) -> str:
    canonical = json.dumps(spec_document(spec), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_builtin_specs() -> list[RulePackSpec]:
    return [parse_yaml(path.read_text("utf-8")) for path in builtin_pack_files()]


# --------------------------------------------------------------------------- ad-hoc checks

ADHOC_PACK_KEY = "adhoc_check"
ADHOC_RULE_KEY = "adhoc_rule"
MAX_HYPOTHESIS_CHARS = 2000


def trained_hypothesis_key(hypothesis: str) -> str | None:
    """The ContractNLI key whose training hypothesis equals ``hypothesis`` verbatim
    (surrounding whitespace aside), else None. Anything else, even a paraphrase, is
    checked by the LLM: the model was never trained on it."""
    text = hypothesis.strip()
    return next((key for key, known in known_model_hypotheses().items() if known == text), None)


def adhoc_rule_spec(hypothesis: str, doc_type: str) -> RulePackSpec:
    """A one-rule pack checking ``hypothesis`` against one document of ``doc_type``
    (``check_rule``). Validated like any imported pack."""
    text = hypothesis.strip()
    key = trained_hypothesis_key(text)
    title = text if len(text) <= 200 else text[:197].rstrip() + "..."
    return validate_spec(
        {
            "key": ADHOC_PACK_KEY,
            "name": "Ad-hoc rule check",
            "version": "1.0.0",
            "description": "A single hypothesis checked against a single document.",
            "rules": [
                {
                    "key": ADHOC_RULE_KEY,
                    "title": title,
                    "hypothesis_text": text,
                    "scope": "per_document",
                    "applies_to_doc_types": [doc_type],
                    "severity": "medium",
                    "model_supported": key is not None,
                    **({"contractnli_key": key} if key else {}),
                }
            ],
        }
    )
