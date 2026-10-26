"""Versioned prompt files and safe rendering of untrusted document text.

Each prompt lives in ``prompts/<name>.md`` with a front-matter ``version``; the recorded
version is ``<name>@<version>+<sha256[:10]>`` so any edit to a prompt file is visible on
findings and LLM ledger rows even if someone forgets to bump the version.

Untrusted text (document spans, chunks, quotes) is rendered as data:
- whitespace is collapsed so each item stays on its own numbered line and cannot forge
  another item's ``[S3]`` prefix on a new line;
- anything that looks like one of our delimiter tags is defanged (``<`` becomes U+2039);
- the delimiters carry a per-request id unknown to the document's author.
Grounding is enforced after the call: only ids that were presented are accepted.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources

_TAG = re.compile(r"<(\s*/?\s*(?:document_excerpts|document_chunks|quotes|rule)\b)", re.IGNORECASE)
_WS = re.compile(r"\s+")
DEFANGED_LT = "\u2039"  # single left-pointing angle quotation mark


@dataclass(frozen=True)
class Prompt:
    name: str
    version: str
    system: str
    template: str

    @property
    def sha256(self) -> str:
        return hashlib.sha256(f"{self.system}\n===\n{self.template}".encode()).hexdigest()

    @property
    def version_id(self) -> str:
        return f"{self.name}@{self.version}+{self.sha256[:10]}"

    def render(self, **values: str) -> str:
        return _fill(self.template, values)

    def render_system(self, **values: str) -> str:
        """The system prompt with its (trusted) placeholders filled in."""
        return _fill(self.system, values)


def _fill(text: str, values: dict[str, str]) -> str:
    """Substitute placeholders. Values are inserted verbatim (never re-scanned), so
    braces inside document text are harmless."""
    names = set(re.findall(r"\{([a-z_]+)\}", text))
    if names - set(values):
        raise ValueError(f"unfilled prompt placeholders: {sorted(names - set(values))}")
    return re.sub(r"\{([a-z_]+)\}", lambda m: values.get(m.group(1), m.group(0)), text)


@lru_cache(maxsize=16)
def load_prompt(name: str) -> Prompt:
    raw = (resources.files("compliance_agents") / "prompts" / f"{name}.md").read_text("utf-8")
    if not raw.startswith("---\n"):
        raise ValueError(f"prompt {name} has no front matter")
    header, body = raw[4:].split("\n---\n", 1)
    meta = dict(line.split(":", 1) for line in header.strip().splitlines())
    meta = {k.strip(): v.strip() for k, v in meta.items()}
    if meta.get("name") != name or not meta.get("version"):
        raise ValueError(f"prompt {name} front matter must declare name and version")
    system, template = body.split("\n===\n", 1)
    return Prompt(name=name, version=meta["version"], system=system.strip(), template=template)


def prompt_versions() -> dict[str, str]:
    return {
        n: load_prompt(n).version_id for n in ("rule_verdict", "support_check", "extract_entities")
    }


def as_data(text: str) -> str:
    """One untrusted item as a single defanged line."""
    return _TAG.sub(lambda m: DEFANGED_LT + m.group(1), _WS.sub(" ", text).strip())


def numbered(items: Sequence[tuple[str, str]]) -> str:
    return "\n".join(f"[{item_id}] {as_data(text)}" for item_id, text in items)


def nonce(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:12]


def normalized_with_map(text: str) -> tuple[str, list[int]]:
    """Whitespace-collapsed text and, for each of its characters, the original index."""
    out: list[str] = []
    index: list[int] = []
    in_space = False
    for i, ch in enumerate(text):
        if ch.isspace():
            if not in_space and out:
                out.append(" ")
                index.append(i)
            in_space = True
            continue
        in_space = False
        out.append(ch)
        index.append(i)
    while out and out[-1] == " ":
        out.pop()
        index.pop()
    return "".join(out), index


def locate_quote(chunk_text: str, quote: str) -> tuple[int, int] | None:
    """Offsets of ``quote`` in ``chunk_text`` (exact, else whitespace-insensitive);
    None if absent or ambiguous (found more than once)."""
    quote = quote.strip()
    if not quote:
        return None
    first = chunk_text.find(quote)
    if first != -1:
        if chunk_text.find(quote, first + 1) != -1:
            return None
        return first, first + len(quote)
    norm_text, index = normalized_with_map(chunk_text)
    norm_quote = _WS.sub(" ", quote)
    pos = norm_text.find(norm_quote)
    if pos == -1 or norm_text.find(norm_quote, pos + 1) != -1:
        return None
    return index[pos], index[pos + len(norm_quote) - 1] + 1
