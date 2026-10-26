"""Rule-pack schema, trained-model support validation and YAML round trips."""

from __future__ import annotations

from typing import Any

import pytest
import yaml

from compliance_core.rule_packs import (
    RulePackValidationError,
    builtin_pack_files,
    content_sha256,
    dump_yaml,
    known_model_hypotheses,
    load_builtin_specs,
    parse_yaml,
    validate_spec,
)


def _vendor_doc() -> dict[str, Any]:
    path = next(p for p in builtin_pack_files() if p.name == "vendor_confidentiality_review.yaml")
    data: dict[str, Any] = yaml.safe_load(path.read_text("utf-8"))
    return data


def test_builtin_packs_load_and_round_trip() -> None:
    specs = {s.key: s for s in load_builtin_specs()}
    assert set(specs) == {"nda_standard", "vendor_confidentiality_review"}
    assert len(specs["nda_standard"].rules) == 17
    for spec in specs.values():
        assert content_sha256(parse_yaml(dump_yaml(spec))) == content_sha256(spec)


def test_vendor_pack_model_rules_are_verbatim_training_hypotheses() -> None:
    known = known_model_hypotheses()
    vendor = {s.key: s for s in load_builtin_specs()}["vendor_confidentiality_review"]
    model_rules = [r for r in vendor.rules if r.model_supported]
    assert {r.contractnli_key for r in model_rules} == {"nda-16", "nda-8", "nda-19"}
    for rule in model_rules:
        assert rule.contractnli_key is not None
        assert rule.hypothesis_text == known[rule.contractnli_key]
    custom = [r for r in vendor.rules if r.scope == "per_document" and not r.model_supported]
    assert [r.key for r in custom] == ["third_party_sharing_restricted"]
    cross = [r for r in vendor.rules if r.scope == "cross_document"]
    assert {r.cross_document.kind for r in cross if r.cross_document} == {
        "contract_on_file",
        "required_protections",
        "term_covers_date",
    }


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda r: r.update(model_supported=True), "contractnli_key of a trained hypothesis"),
        (
            lambda r: r.update(model_supported=True, contractnli_key="nda-16"),
            "verbatim",
        ),
        (
            lambda r: r.update(model_supported=True, contractnli_key="nda-99"),
            "contractnli_key of a trained hypothesis",
        ),
    ],
)
def test_custom_rules_cannot_claim_trained_model_support(mutate: Any, message: str) -> None:
    doc = _vendor_doc()
    rule = next(r for r in doc["rules"] if r["key"] == "third_party_sharing_restricted")
    mutate(rule)
    with pytest.raises(RulePackValidationError) as info:
        validate_spec(doc)
    assert any(message in e for e in info.value.errors)


def test_editing_a_model_rule_hypothesis_requires_the_llm_path() -> None:
    doc = _vendor_doc()
    rule = next(r for r in doc["rules"] if r["key"] == "return_or_destruction")
    rule["hypothesis_text"] += " Within ten days."
    with pytest.raises(RulePackValidationError, match="invalid"):
        validate_spec(doc)
    rule["model_supported"] = False
    spec = validate_spec(doc)
    assert not next(r for r in spec.rules if r.key == "return_or_destruction").model_supported


def test_structural_validation_errors() -> None:
    doc = _vendor_doc()
    doc["rules"].append(dict(doc["rules"][0]))
    with pytest.raises(RulePackValidationError):
        validate_spec(doc)  # duplicate key
    doc = _vendor_doc()
    del doc["register"]
    with pytest.raises(RulePackValidationError):
        validate_spec(doc)  # cross-document rules need a register
    doc = _vendor_doc()
    cross = next(r for r in doc["rules"] if r["key"] == "high_access_required_protections")
    cross["cross_document"]["required_rules"] = ["not_a_rule"]
    with pytest.raises(RulePackValidationError):
        validate_spec(doc)
    doc = _vendor_doc()
    doc["rules"][0]["unexpected"] = 1
    with pytest.raises(RulePackValidationError):
        validate_spec(doc)


def test_yaml_parsing_limits() -> None:
    with pytest.raises(RulePackValidationError, match="could not be parsed"):
        parse_yaml("key: [unclosed")
    with pytest.raises(RulePackValidationError, match="mapping"):
        parse_yaml("- a\n- b\n")
    with pytest.raises(RulePackValidationError, match="larger"):
        parse_yaml("x: " + "a" * 300_000)


def test_adhoc_spec_uses_the_model_only_for_verbatim_training_hypotheses() -> None:
    from compliance_core.rule_packs import (
        adhoc_rule_spec,
        known_model_hypotheses,
        trained_hypothesis_key,
    )

    key, text = next(iter(known_model_hypotheses().items()))
    spec = adhoc_rule_spec(f"  {text}\n", "contract")
    [rule] = spec.rules
    assert rule.model_supported
    assert rule.contractnli_key == key
    assert rule.hypothesis_text == text
    assert rule.applies_to_doc_types == ["contract"]
    for variant in (text.lower(), text.rstrip(".") + "!", text + " Always.", "Sharing is fine."):
        assert trained_hypothesis_key(variant) is None
        [custom] = adhoc_rule_spec(variant, "policy").rules
        assert not custom.model_supported
        assert custom.contractnli_key is None
    long = adhoc_rule_spec("word " * 300, "other").rules[0]
    assert len(long.title) <= 200
    assert long.title.endswith("...")
