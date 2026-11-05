"""
Object and triple normalization helpers for analysis and post-processing.

The functions here are intentionally conservative: they generate candidate
forms and stable matching keys, but do not mutate pipeline outputs by default.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable, List, Sequence, Tuple


Triple = Tuple[str, str, str]

_MONTHS = {
    "january": "01",
    "jan": "01",
    "february": "02",
    "feb": "02",
    "march": "03",
    "mar": "03",
    "april": "04",
    "apr": "04",
    "may": "05",
    "june": "06",
    "jun": "06",
    "july": "07",
    "jul": "07",
    "august": "08",
    "aug": "08",
    "september": "09",
    "sep": "09",
    "sept": "09",
    "october": "10",
    "oct": "10",
    "november": "11",
    "nov": "11",
    "december": "12",
    "dec": "12",
}


def _dedupe_preserve_order(values: Iterable[str]) -> List[str]:
    seen = set()
    result = []
    for value in values:
        if value is None:
            continue
        value = str(value).strip()
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def strip_outer_quotes(value: str) -> str:
    value = str(value).strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1].strip()
    return value


def normalize_unicode(value: str) -> str:
    value = str(value)
    value = value.replace("\u2018", "'").replace("\u2019", "'")
    value = value.replace("\u201c", '"').replace("\u201d", '"')
    value = value.replace("\u2013", "-").replace("\u2014", "-")
    value = value.replace("\u00a0", " ")
    return unicodedata.normalize("NFKC", value)


def strip_accents(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(value))
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def is_numeric_like(value: str) -> bool:
    value = strip_outer_quotes(normalize_unicode(value)).strip()
    value = re.sub(r"^[\$\u00a3\u20ac]\s*", "", value)
    return bool(re.fullmatch(r"[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?|[-+]?\d+(?:\.\d+)?", value))


def normalize_number(value: str) -> str | None:
    value = strip_outer_quotes(normalize_unicode(value)).strip()
    value = re.sub(r"^[\$\u00a3\u20ac]\s*", "", value)
    if not is_numeric_like(value):
        return None
    value = value.replace(",", "")
    if re.fullmatch(r"[-+]?\d+\.0+", value):
        value = value[: value.index(".")]
    return value


def normalize_date(value: str) -> str | None:
    value = strip_outer_quotes(normalize_unicode(value)).strip()
    value = value.replace("_", " ")
    value = re.sub(r",", " ", value)
    value = re.sub(r"\s+", " ", value).strip()

    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        return value

    # 24th February 1989 / 24 February 1989
    match = re.fullmatch(
        r"(?P<day>\d{1,2})(?:st|nd|rd|th)? (?P<month>[A-Za-z]+) (?P<year>\d{4})",
        value,
        flags=re.IGNORECASE,
    )
    if match:
        month = _MONTHS.get(match.group("month").lower())
        if month:
            return f"{match.group('year')}-{month}-{int(match.group('day')):02d}"

    # February 24th 1989 / February 24 1989
    match = re.fullmatch(
        r"(?P<month>[A-Za-z]+) (?P<day>\d{1,2})(?:st|nd|rd|th)? (?P<year>\d{4})",
        value,
        flags=re.IGNORECASE,
    )
    if match:
        month = _MONTHS.get(match.group("month").lower())
        if month:
            return f"{match.group('year')}-{month}-{int(match.group('day')):02d}"

    return None


def normalize_separators(value: str) -> str:
    value = normalize_unicode(value).strip()
    value = value.replace(",_", ", ")
    value = value.replace("_,", ",")
    value = re.sub(r"\s*,\s*", ", ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def normalize_object_candidates(value: str) -> List[str]:
    """Return plausible object forms, ordered from least to more normalized."""
    original = str(value).strip()
    unicode_value = normalize_unicode(original).strip()
    unquoted = strip_outer_quotes(unicode_value)
    candidates = [original, unicode_value, unquoted]

    separator_norm = normalize_separators(unquoted)
    candidates.append(separator_norm)
    candidates.append(separator_norm.replace(" ", "_"))
    candidates.append(separator_norm.replace("_", " "))

    number = normalize_number(unquoted)
    if number is not None:
        candidates.append(number)

    # 610 (metres) -> 610; 418476000000.0 (AU) -> 418476000000
    parenthetical = re.fullmatch(r"(.+?)\s*\(([^)]*)\)", unquoted)
    if parenthetical:
        head = parenthetical.group(1).strip()
        candidates.append(head)
        normalized_head_number = normalize_number(head)
        if normalized_head_number is not None:
            candidates.append(normalized_head_number)

    date = normalize_date(unquoted)
    if date is not None:
        candidates.append(date)

    # Accent-insensitive candidate for analysis; keep accented original earlier.
    accentless = strip_accents(separator_norm)
    if accentless != separator_norm:
        candidates.append(accentless)
        candidates.append(accentless.replace(" ", "_"))

    return _dedupe_preserve_order(candidates)


def canonical_object_key(value: str) -> str:
    """A stable key for comparing objects across common formatting variants."""
    candidates = normalize_object_candidates(value)
    value = candidates[-1] if candidates else str(value)

    date = normalize_date(value)
    if date is not None:
        return f"date:{date}"

    number = normalize_number(value)
    if number is not None:
        return f"num:{number}"

    value = strip_accents(strip_outer_quotes(normalize_unicode(value)))
    value = value.lower()
    value = value.replace("_", " ")
    value = re.sub(r"\s*,\s*", ", ", value)
    value = re.sub(r"\s+", " ", value).strip()
    value = re.sub(r"^[\"']|[\"']$", "", value)
    return value


def canonical_relation_key(value: str) -> str:
    value = strip_outer_quotes(normalize_unicode(value)).strip()
    value = re.sub(r"([a-z])([A-Z])", r"\1 \2", value)
    value = value.replace("_", " ").replace("-", " ")
    value = re.sub(r"\s+", " ", value)
    return value.lower().strip()


def canonical_subject_key(value: str) -> str:
    value = strip_accents(strip_outer_quotes(normalize_unicode(value)))
    value = value.replace("_", " ")
    value = re.sub(r"\s*,\s*", ", ", value)
    value = re.sub(r"\s+", " ", value)
    return value.lower().strip()


def canonical_triple_key(triple: Sequence[str]) -> Triple:
    if len(triple) != 3:
        raise ValueError(f"Expected a triple with 3 fields, got {triple!r}")
    subject, relation, obj = triple
    return (
        canonical_subject_key(subject),
        canonical_relation_key(relation),
        canonical_object_key(obj),
    )


def normalize_triple_candidates(triple: Sequence[str]) -> List[Triple]:
    """Return candidate triples produced by object normalization only."""
    if len(triple) != 3:
        raise ValueError(f"Expected a triple with 3 fields, got {triple!r}")
    subject, relation, obj = [str(part).strip() for part in triple]
    return [
        (subject, relation, candidate)
        for candidate in normalize_object_candidates(obj)
    ]
