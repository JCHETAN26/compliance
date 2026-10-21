"""Rule-based English sentence segmentation tuned for contracts.

Returns spans (offsets into the given text) rather than strings so callers keep exact
source positions. The same segmentation is reused by the chunker and, in Phase 3, by the
rule-check model's span builder.

A boundary is sentence-final punctuation (``.``, ``!``, ``?``, optionally followed by
closing quotes/brackets) followed by whitespace, *unless*:

- the word before the period is a known abbreviation (``Inc.``, ``e.g.``, ``No.``) or a
  single capital initial (``J. Smith``);
- the sentence so far is only an enumerator (``1.``, ``1.2.``, ``(a)``, ``Section 4.``),
  as at the start of a numbered clause;
- the next character is lowercase (``approx. five``) or a digit continuing a list
  (``Sections 1. 2.`` is not split mid-reference).

Paragraph structure is handled by callers: spans never cross the ``[start, end)``
range they are given.
"""

from __future__ import annotations

import re

ABBREVIATIONS = frozenset(
    {
        "mr",
        "mrs",
        "ms",
        "dr",
        "prof",
        "sr",
        "jr",
        "st",
        "inc",
        "ltd",
        "co",
        "corp",
        "llc",
        "llp",
        "plc",
        "gmbh",
        "ag",
        "bv",
        "nv",
        "sa",
        "pty",
        "pte",
        "no",
        "nos",
        "sec",
        "secs",
        "art",
        "arts",
        "para",
        "paras",
        "cl",
        "sch",
        "vol",
        "pp",
        "p",
        "e.g",
        "i.e",
        "etc",
        "vs",
        "v",
        "cf",
        "viz",
        "al",
        "approx",
        "dept",
        "est",
        "u.s",
        "u.k",
        "u.s.a",
        "e.u",
        "n.a",
        "fig",
        "figs",
        "jan",
        "feb",
        "mar",
        "apr",
        "jun",
        "jul",
        "aug",
        "sep",
        "sept",
        "oct",
        "nov",
        "dec",
        "ref",
        "tel",
        "fax",
        "reg",
        "regs",
        "incl",
        "min",
        "max",
        "ext",
        "op",
        "ed",
        "eds",
        "rev",
    }
)

_CANDIDATE = re.compile(r"[.!?]+[\"'”’)\]]*(?=\s)")  # noqa: RUF001 - intentional Unicode
_WORD_BEFORE = re.compile(r"([A-Za-z][A-Za-z.]*)$")
_ENUMERATOR_ONLY = re.compile(
    r"""^\s*(?:
        (?:article|section|clause|schedule|annex|exhibit|part)\s+[0-9ivxlcdm]+[a-z]?(?:\.\d+)*
        |\d+(?:\.\d+)*
        |\(?[a-z]{1,4}\)
        |[ivxlcdm]{1,6}
        |[a-z]
    )[.)]?\s*$""",
    re.IGNORECASE | re.VERBOSE,
)


def _is_boundary(text: str, sent_start: int, punct_start: int, punct_end: int) -> bool:
    if text[punct_start] == ".":
        before = text[sent_start:punct_start]
        word = _WORD_BEFORE.search(before)
        if word:
            token = word.group(1).lower().rstrip(".")
            if token in ABBREVIATIONS:
                return False
            if len(word.group(1)) == 1 and word.group(1).isupper():
                return False
        if _ENUMERATOR_ONLY.match(text[sent_start:punct_end]):
            return False
    following = text[punct_end : punct_end + 40].lstrip()
    return not (following and following[0].islower())


def sentence_spans(text: str, start: int = 0, end: int | None = None) -> list[tuple[int, int]]:
    """Split ``text[start:end]`` into sentence spans with surrounding whitespace removed."""
    stop = len(text) if end is None else end
    spans: list[tuple[int, int]] = []
    cursor = start
    while cursor < stop and text[cursor].isspace():
        cursor += 1
    sent_start = cursor
    for match in _CANDIDATE.finditer(text, start, stop):
        if match.start() < sent_start:
            continue
        if not _is_boundary(text, sent_start, match.start(), match.end()):
            continue
        spans.append((sent_start, match.end()))
        sent_start = match.end()
        while sent_start < stop and text[sent_start].isspace():
            sent_start += 1
    tail_end = stop
    while tail_end > sent_start and text[tail_end - 1].isspace():
        tail_end -= 1
    if tail_end > sent_start:
        spans.append((sent_start, tail_end))
    return spans
