"""Extraction agent: structured entities with chunk citations, and register linking.

Entities are *proposed* by the LLM from numbered chunks and *accepted* only if:
- the chunk id was presented;
- the quote occurs exactly once in that chunk (whitespace-insensitive) -> exact offsets;
- the value is re-read deterministically from the quote (dates and durations are parsed
  from the quote, names must appear in it). Anything else is rejected and counted.

Register linking (spreadsheet row -> contract document) is deterministic and
conservative: exact file name, else the same name up to case/separators/extension, else a
unique party-name match when the row lists no file. Merely similar names (versions,
typos) are never linked automatically. A listed file that is not found, several
candidates, or a party match that contradicts the listed file are never linked: they
become ``missing`` or ``ambiguous`` and the dependent findings need review.
"""

from __future__ import annotations

import re
import uuid
from typing import Any

from compliance_agents.context import RunContext
from compliance_agents.facts import (
    dates_in,
    durations_in,
    filename_stem,
    normalize_filename,
    normalize_name,
    parse_duration_value,
    parse_iso,
    similarity,
    text_contains,
)
from compliance_agents.gateway import LLMStepError
from compliance_agents.nodes.common import digest, error_record
from compliance_agents.outputs import ENTITY_TYPES, EXTRACT_SCHEMA, ExtractOut
from compliance_agents.prompting import load_prompt, locate_quote, nonce, numbered
from compliance_agents.store import BudgetExceededError, EntityDraft
from compliance_agents.text import ChunkView, DocView, load_doc
from compliance_llm import LLMRequest

AGENT = "extraction"
MAX_CHUNK_CHARS = 20_000
_PRIORITY = re.compile(
    r"\b(?:between|party|parties|effective|dated|term|period|expire|terminat|governed|laws? of)\b",
    re.IGNORECASE,
)
SIMILAR_FILENAME = 0.8
PARTY_MATCH = 0.9


def _choose_chunks(doc: DocView) -> list[tuple[str, ChunkView]]:
    """Chunks to present: those with entity cues first, then document order, up to a cap;
    presented in document order with ids C1..Cn."""
    ranked = sorted(
        doc.chunks, key=lambda c: (0 if _PRIORITY.search(doc.slice(c.start, c.end)) else 1, c.index)
    )
    chosen: list[ChunkView] = []
    total = 0
    for chunk in ranked:
        size = chunk.end - chunk.start
        if total + size > MAX_CHUNK_CHARS and chosen:
            continue
        chosen.append(chunk)
        total += size
    chosen.sort(key=lambda c: c.index)
    return [(f"C{i}", c) for i, c in enumerate(chosen, start=1)]


_EFFECTIVE_CUE = re.compile(
    r"\b(?:effective|dated|entered\s+into|commenc\w*|made\s+(?:on|as\s+of)|signed\s+on)\b",
    re.IGNORECASE,
)
_EXPIRY_CUE = re.compile(
    r"\b(?:expir\w*|terminat\w*\s+on|end\w*\s+on|until|through|lapse\w*)\b", re.IGNORECASE
)
_TERM_CUE = re.compile(
    r"\b(?:term|period|duration|remain\s+in\s+(?:full\s+)?(?:force|effect))\b", re.IGNORECASE
)
# A duration of surviving or post-termination obligations is not the agreement's term.
_NOT_TERM = re.compile(
    r"\b(?:surviv\w*|after\s+(?:the\s+)?(?:termination|expiration|expiry)|following\s+"
    r"(?:the\s+)?(?:termination|expiration)|obligations?\s+(?:\w+\s+){0,3}"
    r"(?:continu\w*|remain\w*|last\w*|surviv\w*))\b",
    re.IGNORECASE,
)
# The date or duration must belong to the agreement itself ("This Agreement is effective
# ...", "the term of this NDA", "the Effective Date"), not to an offer, an obligation, a
# notice period or anything else that happens to have a date.
_AGREEMENT = re.compile(
    r"\b(?:this|the)\s+(?:mutual\s+)?(?:non-disclosure\s+)?(?:agreement|nda|contract)\b"
    r"|\beffective\s+date\b",
    re.IGNORECASE,
)


def _role_supported(kind: str, quote: str) -> bool:
    """The quote must say what the value *is* for the agreement, not only contain it."""
    if kind in ("effective_date", "expiration_date", "term") and not _AGREEMENT.search(quote):
        return False
    if kind == "effective_date":
        return bool(_EFFECTIVE_CUE.search(quote)) and not _EXPIRY_CUE.search(quote)
    if kind == "expiration_date":
        return bool(_EXPIRY_CUE.search(quote)) and not _EFFECTIVE_CUE.search(quote)
    if kind == "term":
        return bool(_TERM_CUE.search(quote)) and not _NOT_TERM.search(quote)
    return True


def _check_value(kind: str, quote: str, value: str) -> tuple[bool, str, dict[str, Any]]:
    if not _role_supported(kind, quote):
        return False, "role_not_supported_by_quote", {}
    if kind in ("effective_date", "expiration_date"):
        wanted = parse_iso(value)
        found = dates_in(quote)
        if wanted is None:
            return False, "value_not_iso_date", {}
        if found != [wanted]:
            return False, "date_not_in_quote" if wanted not in found else "ambiguous_quote", {}
        return True, "", {"date": wanted.isoformat()}
    if kind == "term":
        wanted_d = parse_duration_value(value)
        found_d = durations_in(quote)
        if wanted_d is None or found_d != [wanted_d]:
            return False, "duration_not_in_quote", {}
        return True, "", {"months": wanted_d.months, "label": wanted_d.label()}
    if not text_contains(quote, value):
        return False, "value_not_in_quote", {}
    return True, "", {"text": " ".join(value.split())}


async def extract(ctx: RunContext, payload: dict[str, Any]) -> dict[str, Any]:
    document_id = uuid.UUID(payload["document_id"])
    async with ctx.sessions() as session:
        doc = await load_doc(session, document_id, ctx.workspace_id)
    if doc is None:
        err = error_record(
            AGENT, "document_unavailable", "document deleted or not ready", str(document_id)
        )
        await ctx.store.emit(AGENT, "error", err)
        return {"errors": [err], "extracted_entities": {str(document_id): []}}
    await ctx.store.emit(
        AGENT, "agent_start", {"document_id": str(document_id), "filename": doc.filename}
    )
    prompt = load_prompt("extract_entities")
    presented = _choose_chunks(doc)
    by_ref = dict(presented)
    request_nonce = nonce(str(ctx.run_id), "extract", str(document_id))
    request = LLMRequest(
        task="extract_entities",
        system=prompt.render_system(entity_types=", ".join(ENTITY_TYPES)),
        user=prompt.render(
            chunks=numbered([(ref, doc.slice(c.start, c.end)) for ref, c in presented]),
            nonce=request_nonce,
        ),
        schema_name="extracted_entities",
        schema=EXTRACT_SCHEMA,
        max_output_tokens=ctx.settings.llm_max_output_tokens,
        context={
            "chunks": [{"ref": ref, "text": doc.slice(c.start, c.end)} for ref, c in presented],
            "entity_types": list(ENTITY_TYPES),
        },
    )
    step_key = (
        f"extract|{document_id}|{doc.text_sha256[:16]}|{prompt.version_id}|{ctx.provider.model}"
    )
    try:
        out, meta = await ctx.llm.run(
            step_key=step_key,
            agent=AGENT,
            prompt_version=prompt.version_id,
            request=request,
            output_model=ExtractOut,
            event_context={"document_id": str(document_id)},
        )
    except BudgetExceededError as exc:
        err = error_record(AGENT, "budget_exhausted", str(exc), str(document_id))
        await ctx.store.emit(AGENT, "error", err)
        return {
            "errors": [err],
            "budget": {"exhausted": True},
            "extracted_entities": {str(document_id): []},
        }
    except LLMStepError as exc:
        err = error_record(AGENT, exc.code, exc.message, str(document_id))
        await ctx.store.emit(AGENT, "error", err)
        return {"errors": [err], "extracted_entities": {str(document_id): []}}

    drafts: list[EntityDraft] = []
    rejected: dict[str, int] = {}
    seen: set[tuple[str, str]] = set()
    method = "mock_llm" if ctx.provider.is_mock else "llm"
    for entity in out.entities:
        chunk = by_ref.get(entity.chunk_ref)
        if chunk is None:
            rejected["unknown_chunk_ref"] = rejected.get("unknown_chunk_ref", 0) + 1
            continue
        located = locate_quote(doc.slice(chunk.start, chunk.end), entity.quote)
        if located is None:
            rejected["quote_not_found"] = rejected.get("quote_not_found", 0) + 1
            continue
        start, end = chunk.start + located[0], chunk.start + located[1]
        quote = doc.slice(start, end)
        ok, reason, parsed = _check_value(entity.type, quote, entity.value)
        if not ok:
            rejected[reason] = rejected.get(reason, 0) + 1
            continue
        key = (entity.type, str(sorted(parsed.items())))
        if key in seen:
            continue
        seen.add(key)
        _, _, page = doc.geometry(start, end)
        drafts.append(
            EntityDraft(
                document_id=document_id,
                type=entity.type,
                value={**parsed, "proposed_value": entity.value},
                chunk_id=chunk.id,
                quote=quote,
                char_start=start,
                char_end=end,
                page_number=page,
                method=method,
                prompt_version=prompt.version_id,
            )
        )
    ids = await ctx.store.replace_entities(document_id, list(ENTITY_TYPES), drafts)
    await ctx.store.emit(
        AGENT,
        "result",
        {
            "document_id": str(document_id),
            "accepted": len(drafts),
            "rejected": rejected,
            "types": sorted({d.type for d in drafts}),
            "cached": meta.cached,
        },
    )
    return {"extracted_entities": {str(document_id): [str(i) for i in ids]}}


def _name_score(vendor: str, name: str) -> float:
    if vendor and name and (vendor in name or name in vendor) and min(len(vendor), len(name)) >= 4:
        return 1.0
    return similarity(vendor, name)


def _names_match(vendor: str, name: str) -> bool:
    return _name_score(vendor, name) >= PARTY_MATCH


def link_subject(
    subject: dict[str, Any],
    documents: list[dict[str, Any]],
    parties: dict[str, list[str]],
) -> dict[str, Any]:
    listed = subject["values"].get("contract_file", "")
    vendor = normalize_name(subject["values"].get("vendor", ""))
    candidates = [d for d in documents if d["file_format"] not in ("xlsx", "csv")]

    def party_hits() -> list[tuple[float, dict[str, Any]]]:
        if not vendor:
            return []
        hits = []
        for d in candidates:
            names = [normalize_name(p) for p in parties.get(d["id"], [])]
            score = max((_name_score(vendor, n) for n in names if n), default=0.0)
            if score >= PARTY_MATCH:
                hits.append((score, d))
        return sorted(hits, key=lambda h: -h[0])

    base = {
        "subject_key": subject["subject_key"],
        "vendor": subject["label"],
        "contract_file": listed,
    }

    def contradicted(doc: dict[str, Any]) -> bool:
        """The listed file names parties, none of them this vendor, while another
        document does name the vendor: the register and the documents disagree."""
        names = [normalize_name(p) for p in parties.get(doc["id"], []) if p]
        if not names or not vendor or any(_names_match(vendor, n) for n in names):
            return False
        return any(h[1]["id"] != doc["id"] for h in party_hits())

    if listed:
        exact = [
            d for d in candidates if normalize_filename(d["filename"]) == normalize_filename(listed)
        ]
        if len(exact) == 1 and contradicted(exact[0]):
            return {
                **base,
                "outcome": "ambiguous",
                "method": "filename_exact",
                "confidence": None,
                "document_id": None,
                "filename": None,
                "candidates": [exact[0]["filename"]] + [h[1]["filename"] for h in party_hits()[:2]],
                "reason": "the listed file names other parties, and another document names "
                "this vendor",
            }
        if len(exact) == 1:
            return {
                **base,
                "outcome": "linked",
                "method": "filename_exact",
                "confidence": 1.0,
                "document_id": exact[0]["id"],
                "filename": exact[0]["filename"],
                "candidates": [],
            }
        if len(exact) > 1:
            return {
                **base,
                "outcome": "ambiguous",
                "method": "filename_exact",
                "confidence": None,
                "document_id": None,
                "filename": None,
                "candidates": [d["filename"] for d in exact],
                "reason": "several documents have the listed file name",
            }
        # Same name up to case, separators and extension (acme-nda.pdf ~ ACME_NDA.docx).
        same_stem = [
            d
            for d in candidates
            if filename_stem(d["filename"]).replace(" ", "")
            == filename_stem(listed).replace(" ", "")
        ]
        if len(same_stem) == 1:
            d = same_stem[0]
            return {
                **base,
                "outcome": "linked",
                "method": "filename_stem",
                "confidence": 1.0,
                "document_id": d["id"],
                "filename": d["filename"],
                "candidates": [],
            }
        similar = sorted(
            (
                (similarity(filename_stem(listed), filename_stem(d["filename"])), d)
                for d in candidates
            ),
            key=lambda s: -s[0],
        )
        similar = [s for s in similar if s[0] >= SIMILAR_FILENAME]
        if len(same_stem) > 1 or similar:
            return {
                **base,
                "outcome": "ambiguous",
                "method": "filename_similar",
                "confidence": None,
                "document_id": None,
                "filename": None,
                "candidates": [d["filename"] for d in same_stem]
                or [s[1]["filename"] for s in similar[:3]],
                "reason": "only similar file names were found (versions or typos are never "
                "matched automatically)",
            }
        hits = party_hits()
        if hits:
            return {
                **base,
                "outcome": "ambiguous",
                "method": "party_name",
                "confidence": None,
                "document_id": None,
                "filename": None,
                "candidates": [h[1]["filename"] for h in hits[:3]],
                "reason": "the listed file is not in scope, but a document names this vendor",
            }
        return {
            **base,
            "outcome": "missing",
            "method": "filename_exact",
            "confidence": None,
            "document_id": None,
            "filename": None,
            "candidates": [],
            "reason": "no document in the review scope has the listed file name",
        }
    hits = party_hits()
    if len(hits) == 1:
        d = hits[0][1]
        return {
            **base,
            "outcome": "linked",
            "method": "party_name",
            "confidence": round(hits[0][0], 4),
            "document_id": d["id"],
            "filename": d["filename"],
            "candidates": [],
        }
    if len(hits) > 1:
        return {
            **base,
            "outcome": "ambiguous",
            "method": "party_name",
            "confidence": None,
            "document_id": None,
            "filename": None,
            "candidates": [h[1]["filename"] for h in hits[:3]],
            "reason": "several documents name this vendor",
        }
    return {
        **base,
        "outcome": "no_contract_listed",
        "method": None,
        "confidence": None,
        "document_id": None,
        "filename": None,
        "candidates": [],
        "reason": "the register lists no contract and no document names this vendor",
    }


async def link_register(ctx: RunContext, state: dict[str, Any]) -> dict[str, Any]:
    subjects: list[dict[str, Any]] = state.get("subjects", [])
    if not subjects:
        return {"links": {}}
    await ctx.store.emit(AGENT, "tool_call", {"tool": "register.link", "rows": len(subjects)})
    parties: dict[str, list[str]] = {}
    for entity in await ctx.store.load_entities(["party"]):
        parties.setdefault(str(entity.document_id), []).append(
            str(entity.value_json.get("text", ""))
        )
    links = {
        s["subject_key"]: link_subject(s, state.get("documents", []), parties) for s in subjects
    }

    by_register: dict[str, list[EntityDraft]] = {}
    for s in subjects:
        link = links[s["subject_key"]]
        by_register.setdefault(s["document_id"], []).append(
            EntityDraft(
                document_id=uuid.UUID(s["document_id"]),
                type="register_link",
                value={k: v for k, v in link.items() if k != "subject_key"}
                | {"row_number": s["row_number"]},
                chunk_id=uuid.UUID(s["chunk_id"]) if s.get("chunk_id") else None,
                quote=None,
                char_start=s["char_start"],
                char_end=s["char_end"],
                page_number=None,
                method="deterministic",
                prompt_version=None,
                confidence=link.get("confidence"),
            )
        )
    for register_id, drafts in by_register.items():
        await ctx.store.replace_entities(uuid.UUID(register_id), ["register_link"], drafts)
    outcomes: dict[str, int] = {}
    for link in links.values():
        outcomes[link["outcome"]] = outcomes.get(link["outcome"], 0) + 1
    await ctx.store.emit(
        AGENT,
        "result",
        {"tool": "register.link", "outcomes": outcomes, "digest": digest(*sorted(links))},
    )
    return {"links": links}
