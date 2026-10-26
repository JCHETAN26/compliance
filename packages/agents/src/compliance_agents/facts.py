"""Deterministic parsing and matching used to verify extracted facts.

An LLM may *propose* an entity, but its value is accepted only if this module can read
the same value from the verbatim quote. Dates and durations are therefore never taken on
trust, and nothing is computed from text that is not in the document.
"""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import date
from difflib import SequenceMatcher
from pathlib import PurePosixPath

_MONTHS = {
    name: i
    for i, name in enumerate(
        [
            "january",
            "february",
            "march",
            "april",
            "may",
            "june",
            "july",
            "august",
            "september",
            "october",
            "november",
            "december",
        ],
        start=1,
    )
}
_MONTH = "(" + "|".join(_MONTHS) + ")"
_DATE_PATTERNS = (
    re.compile(rf"\b{_MONTH}\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})\b", re.IGNORECASE),
    re.compile(
        rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:day\s+of\s+)?{_MONTH},?\s+(\d{{4}})\b", re.IGNORECASE
    ),
    re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b"),
)
_NUMBERS = {
    w: i
    for i, w in enumerate(
        [
            "zero",
            "one",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
            "ten",
            "eleven",
            "twelve",
        ]
    )
}
_DURATION = re.compile(
    r"\b(?:(?P<word>" + "|".join(_NUMBERS) + r")\s*(?:\((?P<paren>\d{1,3})\)\s*)?"
    r"|(?P<num>\d{1,3})\s*)(?P<unit>years?|months?)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Duration:
    months: int

    def label(self) -> str:
        if self.months % 12 == 0:
            years = self.months // 12
            return f"{years} year{'s' if years != 1 else ''}"
        return f"{self.months} month{'s' if self.months != 1 else ''}"


def dates_in(text: str) -> list[date]:
    found: list[tuple[int, date]] = []
    for pattern in _DATE_PATTERNS:
        for m in pattern.finditer(text):
            groups = m.groups()
            try:
                if pattern is _DATE_PATTERNS[0]:
                    value = date(int(groups[2]), _MONTHS[groups[0].lower()], int(groups[1]))
                elif pattern is _DATE_PATTERNS[1]:
                    value = date(int(groups[2]), _MONTHS[groups[1].lower()], int(groups[0]))
                else:
                    value = date(int(groups[0]), int(groups[1]), int(groups[2]))
            except ValueError:
                continue
            found.append((m.start(), value))
    found.sort(key=lambda item: item[0])
    unique: list[date] = []
    for _, value in found:
        if value not in unique:
            unique.append(value)
    return unique


def durations_in(text: str) -> list[Duration]:
    out: list[Duration] = []
    for m in _DURATION.finditer(text):
        if m.group("word"):
            amount = _NUMBERS[m.group("word").lower()]
            paren = m.group("paren")
            if paren is not None and int(paren) != amount:
                continue  # "two (3) years" is contradictory: not readable
        else:
            amount = int(m.group("num"))
        if amount <= 0:
            continue
        months = amount * 12 if m.group("unit").lower().startswith("year") else amount
        if Duration(months) not in out:
            out.append(Duration(months))
    return out


def parse_iso(value: str) -> date | None:
    m = re.fullmatch(r"\s*(\d{4})-(\d{2})-(\d{2})\s*", value)
    if not m:
        return None
    try:
        return date(int(m[1]), int(m[2]), int(m[3]))
    except ValueError:
        return None


def parse_duration_value(value: str) -> Duration | None:
    found = durations_in(value)
    return found[0] if len(found) == 1 else None


def add_months(start: date, months: int) -> date:
    year = start.year + (start.month - 1 + months) // 12
    month = (start.month - 1 + months) % 12 + 1
    day = min(start.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


_NAME_NOISE = re.compile(
    r"\b(?:the|ltd|limited|gmbh|inc|llc|llp|corp|corporation|co|ag|plc|sa|pty|company)\b\.?",
    re.IGNORECASE,
)


def normalize_name(text: str) -> str:
    text = _NAME_NOISE.sub(" ", text.lower())
    return " ".join(re.findall(r"[a-z0-9]+", text))


def normalize_filename(name: str) -> str:
    return PurePosixPath(name.strip().replace("\\", "/")).name.lower()


def filename_stem(name: str) -> str:
    stem = normalize_filename(name)
    stem = stem.rsplit(".", 1)[0] if "." in stem else stem
    return " ".join(re.findall(r"[a-z0-9]+", stem))


def similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def text_contains(haystack: str, needle: str) -> bool:
    return " ".join(needle.lower().split()) in " ".join(haystack.lower().split())


def parse_register_date(value: object) -> date | None:
    if isinstance(value, str):
        iso = parse_iso(value[:10]) if re.match(r"\d{4}-\d{2}-\d{2}", value) else None
        if iso:
            return iso
        found = dates_in(value)
        return found[0] if len(found) == 1 else None
    return None
