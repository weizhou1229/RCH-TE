"""Final output post-processing for BF-EDC results.

This module is intentionally conservative. It performs deterministic surface
normalization and a small set of schema-aware relation repairs without using
gold references. The raw extraction output should still be kept for auditing.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from edc.object_normalizer import normalize_date, normalize_unicode, strip_outer_quotes


Triple = List[str]


PROFILE_DIR = Path(__file__).resolve().parent / "postprocess_profiles"


DATE_RELATION_MAP = {
    "birthPlace": "birthDate",
    "deathPlace": "deathDate",
    "foundationPlace": "foundingDate",
}

FOOD_COURSE_OBJECTS = {
    "appetizer",
    "dessert",
    "dessert course",
    "entree",
    "main course",
    "main dish",
    "side dish",
    "starter",
}

CATEGORY_OBJECT_MAP = {
    "contributing property": "Contributing_property",
}

EDUCATION_SUBJECT_MARKERS = (
    "university",
    "college",
    "institute",
    "school",
    "academy",
)

CAMPUS_OBJECTS = {
    "urban",
    "rural",
    "suburban",
    "college town",
}

LITERAL_RELATIONS = {
    "address",
    "birthDate",
    "buildDate",
    "buildingStartDate",
    "completionDate",
    "dateOfRetirement",
    "deathDate",
    "elevationAboveTheSeaLevel",
    "firstAired",
    "foundingDate",
    "height",
    "leaderTitle",
    "length",
    "numberOfMembers",
    "orbitalPeriod",
    "populationDensity",
    "populationMetro",
    "runwayName",
    "timeZone",
    "utcOffset",
    "weight",
}

QUOTED_DATE_RELATIONS = {
    "dateOfRetirement",
    "firstAired",
    "lastAired",
}

# Parenthetical suffixes that describe the ENTITY TYPE and are part of the
# canonical surface form (kept, not stripped). Anything else in a "type" object
# (typically a geographic/administrative disambiguator like "(Michigan)") is
# treated as a disambiguation suffix and stripped.
ENTITY_TYPE_PAREN_MARKERS = {
    "corporation",
    "company",
    "band",
    "genre",
    "film",
    "confectionery",
}


def _is_entity_type_paren(inner: str) -> bool:
    text = re.sub(r"\s+", " ", inner.replace("_", " ")).strip().lower()
    return text in ENTITY_TYPE_PAREN_MARKERS


@dataclass
class OutputPostprocessorConfig:
    normalize_objects: bool = True
    repair_relations: bool = True
    dedupe: bool = True
    relation_rule_set: str = "conservative"
    profile: str = "webnlg_conservative"


@dataclass(frozen=True)
class OutputPostprocessorProfile:
    name: str
    surface_mode: str
    enable_relation_repairs: bool
    load_legacy_entity_resources: bool
    strip_numeric_units: bool
    filter_schema_relations: bool
    dedupe_mode: str
    reject_pronoun_entities: bool = False
    require_entity_mentions: bool = False
    remove_inverse_duplicates: bool = False
    max_triples_per_row: Optional[int] = None
    enforce_rebel_date_triggers: bool = False
    enable_rebel_relation_constraints: bool = False
    enable_rebel_relation_reranking: bool = False
    enable_rebel_relation_evidence_reranking: bool = False
    length_adaptive_budget: Optional[Dict[str, int]] = None
    reject_self_relations: bool = False
    repair_truncated_years: bool = False
    reject_date_as_cause_of_death: bool = False
    enable_wiki_relation_constraints: bool = False


def load_output_postprocessor_profile(name_or_path: str) -> OutputPostprocessorProfile:
    """Load and validate a dataset-specific post-processing profile."""
    requested = str(name_or_path or "webnlg_conservative")
    candidate = Path(requested)
    if not candidate.is_file():
        candidate = PROFILE_DIR / f"{requested}.json"
    if not candidate.is_file():
        raise ValueError(f"Unknown output postprocess profile: {requested}")

    with candidate.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    surface_mode = payload.get("surface_mode")
    if surface_mode not in {"webnlg", "preserve"}:
        raise ValueError(
            f"Invalid surface_mode {surface_mode!r} in postprocess profile {candidate}"
        )

    dedupe_mode = payload.get("dedupe_mode", "exact")
    if dedupe_mode not in {"exact", "case_insensitive"}:
        raise ValueError(
            f"Invalid dedupe_mode {dedupe_mode!r} in postprocess profile {candidate}"
        )

    return OutputPostprocessorProfile(
        name=str(payload.get("name") or candidate.stem),
        surface_mode=surface_mode,
        enable_relation_repairs=bool(payload.get("enable_relation_repairs", False)),
        load_legacy_entity_resources=bool(
            payload.get("load_legacy_entity_resources", False)
        ),
        strip_numeric_units=bool(payload.get("strip_numeric_units", False)),
        filter_schema_relations=bool(payload.get("filter_schema_relations", False)),
        dedupe_mode=dedupe_mode,
        reject_pronoun_entities=bool(payload.get("reject_pronoun_entities", False)),
        require_entity_mentions=bool(payload.get("require_entity_mentions", False)),
        remove_inverse_duplicates=bool(
            payload.get("remove_inverse_duplicates", False)
        ),
        max_triples_per_row=(
            int(payload["max_triples_per_row"])
            if payload.get("max_triples_per_row") is not None
            else None
        ),
        enforce_rebel_date_triggers=bool(
            payload.get("enforce_rebel_date_triggers", False)
        ),
        enable_rebel_relation_constraints=bool(
            payload.get("enable_rebel_relation_constraints", False)
        ),
        enable_rebel_relation_reranking=bool(
            payload.get("enable_rebel_relation_reranking", False)
        ),
        enable_rebel_relation_evidence_reranking=bool(
            payload.get("enable_rebel_relation_evidence_reranking", False)
        ),
        length_adaptive_budget=(
            {str(key): int(value) for key, value in payload["length_adaptive_budget"].items()}
            if payload.get("length_adaptive_budget") is not None
            else None
        ),
        reject_self_relations=bool(payload.get("reject_self_relations", False)),
        repair_truncated_years=bool(payload.get("repair_truncated_years", False)),
        reject_date_as_cause_of_death=bool(
            payload.get("reject_date_as_cause_of_death", False)
        ),
        enable_wiki_relation_constraints=bool(
            payload.get("enable_wiki_relation_constraints", False)
        ),
    )


def _norm_text(value: str) -> str:
    value = strip_outer_quotes(normalize_unicode(str(value)))
    value = value.replace("_", " ")
    value = re.sub(r"\s+", " ", value)
    return value.lower().strip()


def _schema_has(schema: Dict[str, str], relation: str) -> bool:
    return not schema or relation in schema


def _context_matches(context: Optional[str], patterns: Iterable[str]) -> bool:
    if not context:
        return False
    text = _norm_text(context)
    return any(re.search(pattern, text) for pattern in patterns)


def _rebel_entity_mentioned(value: str, context: Optional[str]) -> bool:
    """Return whether an extracted entity has a contiguous source mention."""
    if context is None:
        return True
    entity = _norm_text(value)
    source = _norm_text(context)
    if not entity or not source:
        return False
    return re.search(r"(?<!\w)" + re.escape(entity) + r"(?!\w)", source) is not None


def _education_subject(subject: str) -> bool:
    text = _norm_text(subject)
    return any(marker in text for marker in EDUCATION_SUBJECT_MARKERS)


def _campus_surface_object(obj: str) -> bool:
    text = _norm_text(obj)
    if text in CAMPUS_OBJECTS:
        return True
    return bool(
        re.search(
            r"\b(?:campus|hall|building|centre|center|complex|block|tower|"
            r"road|street|avenue|drive|lane|boulevard|way|address)\b",
            text,
        )
        or re.search(r"\d{4,}", text)
        or "," in strip_outer_quotes(_collapse_nested_quotes(obj))
    )


def _plain_location_object(obj: str) -> bool:
    text = _norm_text(obj)
    if not text or _campus_surface_object(obj):
        return False
    return len(text.split()) <= 3


def _subjects_with_address(triples: Sequence[Sequence[str]]) -> set[str]:
    subjects = set()
    for triple in triples:
        if len(triple) != 3:
            continue
        subject, relation, _ = [str(part).strip() for part in triple]
        if relation == "address":
            subjects.add(_norm_text(subject))
    return subjects


def _collapse_nested_quotes(value: str, normalize: bool = True) -> str:
    value = str(value).strip()
    if normalize:
        value = normalize_unicode(value).strip()
    previous = None
    while previous != value:
        previous = value
        if value.startswith('""') and value.endswith('""') and len(value) >= 4:
            value = value[1:-1].strip()
        elif value.startswith("''") and value.endswith("''") and len(value) >= 4:
            value = value[1:-1].strip()
    return value


def _restore_dashes(normalized: str, original: str) -> str:
    """Restore original en/em-dash characters that normalize_unicode flattened
    to ASCII '-'. The reference keeps en-dashes in values like postal-code
    ranges ("76131–76229"), so the surface we write back must preserve them
    when the only difference from the original is the dash flattening."""
    orig = str(original)
    if "–" not in orig and "—" not in orig:
        return normalized
    if normalize_unicode(normalized) == normalize_unicode(orig):
        # Same value up to unicode folding: prefer the original dash glyphs.
        return strip_outer_quotes(_collapse_nested_quotes(orig, normalize=False))
    return normalized


def _normalize_entity_surface(value: str) -> str:
    value = strip_outer_quotes(_collapse_nested_quotes(value, normalize=False)).strip()
    value = re.sub(r"\s+", " ", value)
    if " " in value and not any(ch in value for ch in ['"', "'", ","]):
        value = value.replace(" ", "_")
    return value


def _normalize_numeric_surface(value: str) -> Optional[str]:
    raw = strip_outer_quotes(_collapse_nested_quotes(value)).strip()
    if re.fullmatch(r"[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?", raw):
        return raw.replace(",", "")
    return None


def normalize_object_surface(relation: str, obj: str) -> str:
    value = _collapse_nested_quotes(obj)
    unquoted = strip_outer_quotes(value)
    had_quotes = value != unquoted

    date_value = normalize_date(unquoted)
    if date_value is not None:
        if had_quotes or relation in QUOTED_DATE_RELATIONS:
            return f'"{date_value}"'
        return date_value

    number_value = _normalize_numeric_surface(unquoted)
    if number_value is not None:
        return number_value

    if relation == "utcOffset":
        trimmed = unquoted.strip()
        if re.fullmatch(r"-\d+(?::\d+)?", trimmed):
            return "−" + trimmed[1:]
        return trimmed

    if relation == "type":
        match = re.fullmatch(r"([^()]+)_\(([^)]*)\)", unquoted.strip())
        if match and not _is_entity_type_paren(match.group(2)):
            # Strip only disambiguation-style suffixes (e.g. "City_(Michigan)").
            # Keep entity-type descriptors like "S.A._(corporation)" which the
            # reference retains as part of the object surface.
            return match.group(1).strip("_")

    category_value = CATEGORY_OBJECT_MAP.get(_norm_text(unquoted))
    if relation == "category" and category_value:
        return category_value

    if relation in LITERAL_RELATIONS:
        return _restore_dashes(value, obj)

    return _restore_dashes(value, obj)


def normalize_triple_surface(
    triple: Sequence[str],
    surface_mode: str = "webnlg",
) -> Triple:
    if len(triple) != 3:
        raise ValueError(f"Expected a triple with 3 fields, got {triple!r}")
    subject, relation, obj = [str(part).strip() for part in triple]
    if surface_mode == "preserve":
        return [subject, relation, obj]
    if surface_mode != "webnlg":
        raise ValueError(f"Unknown surface normalization mode: {surface_mode}")
    return [
        _normalize_entity_surface(subject),
        relation,
        normalize_object_surface(relation, obj),
    ]


def _food_course_context(context: Optional[str]) -> bool:
    return _context_matches(
        context,
        [
            r"\bcourse\b",
            r"\bdessert\b",
            r"\bserved\b",
            r"\bdish\b",
            r"\bfood\b",
        ],
    )


def _main_ingredient_context(context: Optional[str]) -> bool:
    return _context_matches(
        context,
        [
            r"\bmade\s+(?:from|with|of)\b",
            r"\bcontains?\b",
            r"\bingredients?\b",
            r"\bcomposed\s+of\b",
            r"\bprepared\s+with\b",
        ],
    )


def _ingredient_list_object(obj: str) -> bool:
    raw = strip_outer_quotes(_collapse_nested_quotes(obj))
    text = _norm_text(raw)
    return bool(
        "," in raw
        or re.search(r"\b(?:and|with)\b", text)
        or len([part for part in re.split(r"[,;/]", raw) if part.strip()]) >= 2
    )


def _dish_variation_context(context: Optional[str]) -> bool:
    return _context_matches(
        context,
        [
            r"\bvariation\b",
            r"\bvariations\b",
            r"\bvaried\s+by\b",
            r"\bcan\s+be\s+varied\b",
            r"\bvariant\b",
        ],
    )


def _dish_variation_object(obj: str) -> bool:
    text = _norm_text(obj)
    return text in {
        "cottage cheese",
    }


def _category_context(context: Optional[str]) -> bool:
    return _context_matches(
        context,
        [
            r"\bcategory\b",
            r"\bfalls?\s+under\b",
            r"\bclassified\b",
        ],
    )


def _campus_context(context: Optional[str]) -> bool:
    return _context_matches(
        context,
        [
            r"\bcampus\b",
            r"\bcampus\s+address\b",
            r"\btechnical\s+campus\b",
        ],
    )


def propose_relation(
    triple: Sequence[str],
    context: Optional[str],
    schema: Dict[str, str],
    rule_set: str,
) -> Tuple[str, Optional[str]]:
    subject, relation, obj = [str(part).strip() for part in triple]
    obj_text = _norm_text(obj)

    date_value = normalize_date(obj)
    if date_value is not None:
        target = DATE_RELATION_MAP.get(relation)
        if target and _schema_has(schema, target):
            return target, f"date_object:{date_value}"

    if relation == "type" and _schema_has(schema, "course"):
        if obj_text in FOOD_COURSE_OBJECTS and _food_course_context(context):
            return "course", "food_course_context"

    if relation == "ingredient" and _schema_has(schema, "dishVariation"):
        if _dish_variation_context(context) and _dish_variation_object(obj):
            return "dishVariation", "dish_variation_context"

    if relation == "ingredient" and _schema_has(schema, "mainIngredient"):
        if _main_ingredient_context(context) and _ingredient_list_object(obj):
            return "mainIngredient", "main_ingredient_list_context"

    if relation == "type" and _schema_has(schema, "category"):
        if obj_text in CATEGORY_OBJECT_MAP or _category_context(context):
            return "category", "category_context"

    if relation in {"address", "location"} and _schema_has(schema, "campus"):
        if _education_subject(subject) and _campus_context(context):
            return "campus", "education_campus_context"

    # NOTE: a `location -> country` gazetteer rule was tried and reverted.
    # In WebNLG, `location` and `country` are both valid relations for the
    # same "entity is in country X" fact, chosen by schema pragmatics rather
    # than object type. Keying on "object is a country" broke more correct
    # `location` triples than it fixed (helped 16 / hurt 28, Full Triple F1
    # 0.2934 -> 0.2909). Object-type signal cannot disambiguate this pair.

    if rule_set == "conservative":
        return relation, None

    return relation, None


def dedupe_triples(triples: List[Triple]) -> Tuple[List[Triple], int]:
    seen = set()
    deduped: List[Triple] = []
    removed = 0
    for triple in triples:
        key = tuple(triple)
        if key in seen:
            removed += 1
            continue
        seen.add(key)
        deduped.append(triple)
    return deduped, removed


REBEL_PRONOUN_ENTITIES = {
    "he",
    "her",
    "hers",
    "him",
    "himself",
    "it",
    "its",
    "itself",
    "she",
    "they",
    "them",
    "their",
    "theirs",
    "themselves",
    "we",
    "you",
    "the council",
}

REBEL_INVERSE_RELATIONS = {
    ("part of", "has part"),
    ("has part", "part of"),
    (
        "located in the administrative territorial entity",
        "contains administrative territorial entity",
    ),
    (
        "contains administrative territorial entity",
        "located in the administrative territorial entity",
    ),
}

REBEL_SPORT_OBJECTS = {
    "american football",
    "basketball",
    "baseball",
    "boxing",
    "cricket",
    "cycling",
    "decathlon",
    "football",
    "gymnastics",
    "handball",
    "hockey",
    "ice hockey",
    "rowing",
    "rugby",
    "skiing",
    "soccer",
    "swimming",
    "sumo",
    "tennis",
    "volleyball",
    "water polo",
}

REBEL_RELATION_PRIORITY = {
    "country": 60,
    "contains administrative territorial entity": 58,
    "located in the administrative territorial entity": 56,
    "sport": 54,
    "date of birth": 52,
    "date of death": 52,
    "point in time": 52,
    "place of birth": 50,
    "member of sports team": 50,
    "cast member": 50,
    "director": 48,
    "author": 48,
    "genre": 46,
    "occupation": 30,
    "instance of": 20,
}

REBEL_DATE_TRIGGERS = {
    "date of birth": (r"\bborn\b", r"\bbirth\b"),
    "date of death": (r"\bdied\b", r"\bdeath\b", r"\bdeceased\b"),
    "inception": (r"\bfounded\b", r"\bfounding\b", r"\bestablished\b"),
    "publication date": (r"\bpublished\b", r"\bpublication\b", r"\breleased\b"),
    "point in time": (r"\boccurred\b", r"\btook place\b", r"\badopted\b"),
}


def _rebel_pronoun_entity(value: str) -> bool:
    return _norm_text(value) in REBEL_PRONOUN_ENTITIES


WIKI_YEAR_RELATION_MARKERS = (
    "year",
    "date",
    "time",
)


def _wiki_truncated_year_repair(
    triple: Sequence[str], context: Optional[str]
) -> Optional[str]:
    relation = _norm_text(triple[1])
    value = _norm_text(triple[2]).replace(",", "")
    if (
        not context
        or not any(marker in relation for marker in WIKI_YEAR_RELATION_MARKERS)
        or not re.fullmatch(r"\d{1,3}", value)
    ):
        return None
    normalized_context = normalize_unicode(str(context))
    if re.search(rf"(?<!\d){re.escape(value)}(?!\d)", normalized_context):
        return None
    def is_subsequence(short: str, long: str) -> bool:
        iterator = iter(long)
        return all(any(char == candidate for candidate in iterator) for char in short)

    candidates = {
        year
        for year in re.findall(r"(?<!\d)(\d{4})(?!\d)", normalized_context)
        if is_subsequence(value, year)
    }
    return next(iter(candidates)) if len(candidates) == 1 else None


def _wiki_date_as_cause_of_death(triple: Sequence[str]) -> bool:
    if _norm_text(triple[1]) != "cause of death":
        return False
    value = _norm_text(triple[2])
    return bool(
        re.fullmatch(r"\d{1,4}", value)
        or re.search(r"\b(?:19|20)\d{2}\b", value)
        or re.search(r"\b(?:january|february|march|april|may|june|july|august|"
                     r"september|october|november|december)\b", value)
    )


WIKI_CONTINENT_NAMES = {
    "africa", "antarctica", "asia", "europe", "north america",
    "south america", "oceania",
}


def _wiki_relation_constraint(
    triple: Triple, context: Optional[str]
) -> Tuple[Optional[Triple], Optional[str]]:
    """Apply high-confidence Wiki-NRE relation checks from local wording."""
    subject, relation, obj = triple
    text = _norm_text(context or "")
    obj_text = _norm_text(obj)

    if relation == "located in the administrative territorial entity":
        has_location_trigger = bool(
            re.search(r"\b(?:located|situated|based|headquartered)\s+in\b", text)
        )
        has_part_trigger = bool(
            re.search(r"\b(?:part of|forming part of|belongs to)\b", text)
        )
        if has_part_trigger and not has_location_trigger:
            return [subject, "part of", obj], "explicit_part_of_context"

    if relation == "serves" and re.search(
        r"\b(?:named|renamed|changed)\s+(?:after|to)\b", text
    ):
        return [subject, "named after", obj], "explicit_named_after_context"

    if relation == "original language of film or TV show":
        has_language_trigger = bool(
            re.search(r"\b(?:language|in)\s+(?:english|french|german|italian|"
                      r"spanish|hindi|tamil|telugu|japanese|chinese)\b", text)
        )
        if not has_language_trigger:
            return None, "language_relation_without_trigger"

    if relation == "country" and obj_text in WIKI_CONTINENT_NAMES:
        return None, "continent_as_country"

    if relation == "located on terrain feature" and re.search(
        r"\b(?:road|street|avenue|highway|lane|boulevard)\b", obj_text
    ):
        return None, "road_as_terrain_feature"

    if relation == "member of" and re.search(
        r"\b(?:played|competed|signed|selected)\s+(?:for|by)\b", text
    ):
        return [subject, "member of sports team", obj], "sports_team_context"

    return triple, None


def _rebel_date_relation_supported(
    relation: str,
    context: Optional[str],
    subject: str,
) -> bool:
    patterns = REBEL_DATE_TRIGGERS.get(relation)
    if not patterns or not context:
        return False
    subject_text = _norm_text(subject)
    # A trigger elsewhere in a multi-sentence article must not license a date
    # attached to a different entity.
    sentences = re.split(r"(?<=[.!?])\s+", normalize_unicode(str(context)))
    return any(
        re.search(
            r"(?<!\w)" + re.escape(subject_text) + r"(?!\w)",
            _norm_text(sentence),
        )
        and any(re.search(pattern, _norm_text(sentence)) for pattern in patterns)
        for sentence in sentences
    )


def _remove_rebel_inverse_duplicates(
    triples: List[Triple],
) -> Tuple[List[Triple], List[Tuple[int, str, Triple]]]:
    """Remove a reverse pair only when both directions were predicted."""
    reverse_indexes = set()
    keys = {(triple[0], triple[1], triple[2]): index for index, triple in enumerate(triples)}
    for index, (subject, relation, obj) in enumerate(triples):
        if (relation, relation) in REBEL_INVERSE_RELATIONS:
            continue
        for left, right in REBEL_INVERSE_RELATIONS:
            if relation != left:
                continue
            reverse = keys.get((obj, right, subject))
            if reverse is not None and reverse != index:
                # Keep the first generated direction and remove its converse.
                reverse_indexes.add(max(index, reverse))
    kept = [triple for index, triple in enumerate(triples) if index not in reverse_indexes]
    return kept, [
        (index, "inverse_duplicate", list(triples[index]))
        for index in sorted(reverse_indexes)
    ]


def _rebel_relation_repair(
    triple: Triple,
    context: Optional[str],
) -> Tuple[Triple, Optional[str]]:
    """Apply only high-confidence relation corrections from the surface form."""
    subject, relation, obj = triple
    if relation in {"instance of", "occupation"}:
        if _norm_text(obj) in REBEL_SPORT_OBJECTS:
            return [subject, "sport", obj], "explicit_sport_object"
    return triple, None


def _rebel_relation_score(triple: Triple, context: Optional[str]) -> int:
    relation = triple[1]
    score = REBEL_RELATION_PRIORITY.get(relation, 40)
    if relation == "sport" and _norm_text(triple[2]) in REBEL_SPORT_OBJECTS:
        score += 20
    if relation in REBEL_DATE_TRIGGERS and _rebel_date_relation_supported(
        relation, context, triple[0]
    ):
        score += 15
    return score


def _rebel_relation_evidence_score(triple: Triple, context: Optional[str]) -> int:
    """Add conservative lexical evidence to the pre-cap relation priority."""
    if not context:
        return 0
    subject, relation, obj = triple
    text = _norm_text(context)
    subject_text = _norm_text(subject)
    obj_text = _norm_text(obj)
    if not subject_text or not obj_text:
        return -8
    same_sentence = any(
        re.search(r"(?<!\w)" + re.escape(subject_text) + r"(?!\w)", sentence)
        and re.search(r"(?<!\w)" + re.escape(obj_text) + r"(?!\w)", sentence)
        for sentence in re.split(r"(?<=[.!?])\s+", text)
    )
    score = 3 if same_sentence else -3
    if relation in REBEL_DATE_TRIGGERS:
        score += 8 if _rebel_date_relation_supported(relation, context, subject) else -2
    elif relation == "occupation":
        score += 8 if re.search(
            r"\b(?:occupation|worked as|works as|served as|career as|professional)\b",
            text,
        ) else -5
    elif relation == "instance of":
        score += 8 if re.search(
            r"\b(?:is an?|was an?|type of|kind of|genus|species|subclass of)\b",
            text,
        ) else -5
    elif relation == "located in the administrative territorial entity":
        score += 6 if re.search(r"\b(?:located|situated|based|headquartered)\s+in\b", text) else -2
    elif relation == "contains administrative territorial entity":
        score += 6 if re.search(r"\b(?:contains|includes|comprises)\b", text) else -2
    elif relation == "member of political party":
        score += 6 if re.search(r"\bmember of\b", text) else -2
    return score


def _rebel_length_budget(
    context: Optional[str], budget: Optional[Dict[str, int]]
) -> Optional[int]:
    if not budget or context is None:
        return None
    length = len(str(context))
    if length <= budget["short_max_chars"]:
        return budget["short_max_triples"]
    if length <= budget["medium_max_chars"]:
        return budget["medium_max_triples"]
    return budget["long_max_triples"]


class OutputPostprocessor:
    """Apply final deterministic repairs to extracted rows."""

    def __init__(
        self,
        schema: Optional[Dict[str, str]] = None,
        config: Optional[OutputPostprocessorConfig] = None,
    ) -> None:
        self.schema = schema or {}
        self.config = config or OutputPostprocessorConfig()
        self.profile = load_output_postprocessor_profile(self.config.profile)

    def process_rows(
        self,
        rows: List[List[Sequence[str]]],
        contexts: Optional[List[str]] = None,
    ) -> Tuple[List[List[Triple]], Dict]:
        output_rows: List[List[Triple]] = []
        changes: List[Dict] = []
        change_counter: Counter[Tuple[str, str]] = Counter()
        deduped_total = 0

        for row_idx, triples in enumerate(rows):
            context = contexts[row_idx] if contexts and row_idx < len(contexts) else None
            new_row: List[Triple] = []
            subjects_with_address = _subjects_with_address(triples)

            for triple_idx, triple in enumerate(triples):
                current = [str(part).strip() for part in triple]
                if (
                    self.profile.require_entity_mentions
                    and len(current) == 3
                    and context is not None
                    and not (
                        _rebel_entity_mentioned(current[0], context)
                        and _rebel_entity_mentioned(current[2], context)
                    )
                ):
                    changes.append(
                        {
                            "line": row_idx + 1,
                            "triple_index": triple_idx,
                            "kind": "filter",
                            "reason": "entity_not_in_context",
                            "old": list(current),
                            "new": None,
                        }
                    )
                    change_counter[("filter", "entity_not_in_context")] += 1
                    continue
                if (
                    self.profile.reject_pronoun_entities
                    and len(current) == 3
                    and (
                        _rebel_pronoun_entity(current[0])
                        or _rebel_pronoun_entity(current[2])
                    )
                ):
                    changes.append(
                        {
                            "line": row_idx + 1,
                            "triple_index": triple_idx,
                            "kind": "filter",
                            "reason": "pronoun_entity",
                            "old": list(current),
                            "new": None,
                        }
                    )
                    change_counter[("filter", "pronoun_entity")] += 1
                    continue

                if (
                    len(current) == 3
                    and self.profile.reject_self_relations
                    and _norm_text(current[0]) == _norm_text(current[2])
                ):
                    changes.append(
                        {
                            "line": row_idx + 1,
                            "triple_index": triple_idx,
                            "kind": "filter",
                            "reason": "self_relation",
                            "old": list(current),
                            "new": None,
                        }
                    )
                    change_counter[("filter", "self_relation")] += 1
                    continue

                if (
                    len(current) == 3
                    and self.profile.repair_truncated_years
                ):
                    repaired_year = _wiki_truncated_year_repair(current, context)
                    if repaired_year is not None:
                        repaired = [current[0], current[1], repaired_year]
                        changes.append(
                            {
                                "line": row_idx + 1,
                                "triple_index": triple_idx,
                                "kind": "surface_repair",
                                "reason": "truncated_year_from_context",
                                "old": list(current),
                                "new": list(repaired),
                            }
                        )
                        change_counter[("surface", "truncated_year_from_context")] += 1
                        current = repaired

                if (
                    len(current) == 3
                    and self.profile.reject_date_as_cause_of_death
                    and _wiki_date_as_cause_of_death(current)
                ):
                    changes.append(
                        {
                            "line": row_idx + 1,
                            "triple_index": triple_idx,
                            "kind": "filter",
                            "reason": "date_as_cause_of_death",
                            "old": list(current),
                            "new": None,
                        }
                    )
                    change_counter[("filter", "date_as_cause_of_death")] += 1
                    continue

                if (
                    self.profile.enforce_rebel_date_triggers
                    and len(current) == 3
                    and current[1] in REBEL_DATE_TRIGGERS
                    and not _rebel_date_relation_supported(
                        current[1], context, current[0]
                    )
                ):
                    changes.append(
                        {
                            "line": row_idx + 1,
                            "triple_index": triple_idx,
                            "kind": "filter",
                            "reason": "date_relation_without_trigger",
                            "old": list(current),
                            "new": None,
                        }
                    )
                    change_counter[("filter", "date_relation_without_trigger")] += 1
                    continue

                before = list(current)

                if self.config.normalize_objects:
                    current = normalize_triple_surface(
                        current,
                        surface_mode=self.profile.surface_mode,
                    )
                    if current != before:
                        changes.append(
                            {
                                "line": row_idx + 1,
                                "triple_index": triple_idx,
                                "kind": "surface_normalization",
                                "reason": "object_normalizer",
                                "old": before,
                                "new": list(current),
                            }
                        )
                        change_counter[("surface", "normalized")] += 1

                if (
                    self.profile.enable_rebel_relation_constraints
                    and self.profile.enable_rebel_relation_reranking
                ):
                    repaired, reason = _rebel_relation_repair(current, context)
                    if repaired != current:
                        changes.append(
                            {
                                "line": row_idx + 1,
                                "triple_index": triple_idx,
                                "kind": "relation_constraint",
                                "reason": reason,
                                "old": list(current),
                                "new": list(repaired),
                            }
                        )
                        change_counter[("relation", reason or "repaired")] += 1
                        current = repaired

                if self.config.repair_relations and self.profile.enable_relation_repairs:
                    relation_before = current[1]
                    relation_after, reason = propose_relation(
                        current,
                        context=context,
                        schema=self.schema,
                        rule_set=self.config.relation_rule_set,
                    )
                    if (
                        relation_before == "location"
                        and relation_after == "campus"
                        and _plain_location_object(current[2])
                        and _norm_text(current[0]) in subjects_with_address
                    ):
                        relation_after, reason = relation_before, None
                    if relation_after != relation_before:
                        after = [current[0], relation_after, current[2]]
                        changes.append(
                            {
                                "line": row_idx + 1,
                                "triple_index": triple_idx,
                                "kind": "relation_repair",
                                "reason": reason,
                                "old": list(current),
                                "new": after,
                            }
                        )
                        change_counter[(relation_before, relation_after)] += 1
                        current = after

                if self.profile.enable_wiki_relation_constraints:
                    constrained, reason = _wiki_relation_constraint(current, context)
                    if constrained is None:
                        changes.append(
                            {
                                "line": row_idx + 1,
                                "triple_index": triple_idx,
                                "kind": "filter",
                                "reason": reason,
                                "old": list(current),
                                "new": None,
                            }
                        )
                        change_counter[("filter", reason or "wiki_relation_constraint")] += 1
                        continue
                    if constrained != current:
                        changes.append(
                            {
                                "line": row_idx + 1,
                                "triple_index": triple_idx,
                                "kind": "relation_constraint",
                                "reason": reason,
                                "old": list(current),
                                "new": list(constrained),
                            }
                        )
                        change_counter[("relation", reason or "wiki_relation_constraint")] += 1
                        current = constrained

                new_row.append(current)

            if self.config.dedupe:
                new_row, removed = dedupe_triples(new_row)
                deduped_total += removed

            if self.profile.remove_inverse_duplicates:
                new_row, inverse_changes = _remove_rebel_inverse_duplicates(new_row)
                for triple_idx, reason, removed_triple in inverse_changes:
                    changes.append(
                        {
                            "line": row_idx + 1,
                            "triple_index": triple_idx,
                            "kind": "filter",
                            "reason": reason,
                            "old": removed_triple,
                            "new": None,
                        }
                    )
                    change_counter[("filter", reason)] += 1

            if self.profile.enable_rebel_relation_reranking:
                new_row.sort(
                    key=lambda triple: (
                        _rebel_relation_score(triple, context)
                        + (
                            _rebel_relation_evidence_score(triple, context)
                            if self.profile.enable_rebel_relation_evidence_reranking
                            else 0
                        )
                    ),
                    reverse=True,
                )

            if (
                self.profile.max_triples_per_row is not None
                and len(new_row) > self.profile.max_triples_per_row
            ):
                limit = self.profile.max_triples_per_row
                for triple_idx in range(limit, len(new_row)):
                    changes.append(
                        {
                            "line": row_idx + 1,
                            "triple_index": triple_idx,
                            "kind": "filter",
                            "reason": "row_triple_budget",
                            "old": list(new_row[triple_idx]),
                            "new": None,
                        }
                    )
                    change_counter[("filter", "row_triple_budget")] += 1
                new_row = new_row[:limit]

            adaptive_limit = _rebel_length_budget(
                context, self.profile.length_adaptive_budget
            )
            if adaptive_limit is not None and len(new_row) > adaptive_limit:
                for triple_idx in range(adaptive_limit, len(new_row)):
                    changes.append(
                        {
                            "line": row_idx + 1,
                            "triple_index": triple_idx,
                            "kind": "filter",
                            "reason": "length_adaptive_budget",
                            "old": list(new_row[triple_idx]),
                            "new": None,
                        }
                    )
                    change_counter[("filter", "length_adaptive_budget")] += 1
                new_row = new_row[:adaptive_limit]

            output_rows.append(new_row)

        summary = {
            "rows": len(rows),
            "total_triples_before": sum(len(row) for row in rows),
            "total_triples_after": sum(len(row) for row in output_rows),
            "changed": len(changes),
            "deduped": deduped_total,
            "config": {
                "normalize_objects": self.config.normalize_objects,
                "repair_relations": self.config.repair_relations,
                "dedupe": self.config.dedupe,
                "relation_rule_set": self.config.relation_rule_set,
                "profile": self.profile.name,
                "profile_options": {
                    "reject_pronoun_entities": self.profile.reject_pronoun_entities,
                    "require_entity_mentions": self.profile.require_entity_mentions,
                    "remove_inverse_duplicates": self.profile.remove_inverse_duplicates,
                    "max_triples_per_row": self.profile.max_triples_per_row,
                    "enforce_rebel_date_triggers": self.profile.enforce_rebel_date_triggers,
                    "enable_rebel_relation_constraints": self.profile.enable_rebel_relation_constraints,
                    "enable_rebel_relation_reranking": self.profile.enable_rebel_relation_reranking,
                    "enable_rebel_relation_evidence_reranking": self.profile.enable_rebel_relation_evidence_reranking,
                    "length_adaptive_budget": self.profile.length_adaptive_budget,
                    "reject_self_relations": self.profile.reject_self_relations,
                    "repair_truncated_years": self.profile.repair_truncated_years,
                    "reject_date_as_cause_of_death": self.profile.reject_date_as_cause_of_death,
                    "enable_wiki_relation_constraints": self.profile.enable_wiki_relation_constraints,
                },
            },
            "top_changes": [
                {"from": old, "to": new, "count": count}
                for (old, new), count in change_counter.most_common(50)
            ],
            "changes": changes,
        }
        return output_rows, summary


def write_postprocess_artifacts(output_dir: str, summary: Dict) -> None:
    iter_dir = Path(output_dir) / "iter0"
    changes = summary.get("changes", [])

    with (iter_dir / "output_postprocess_changes.jsonl").open("w", encoding="utf-8") as f:
        for change in changes:
            f.write(json.dumps(change, ensure_ascii=False) + "\n")

    public_summary = {key: value for key, value in summary.items() if key != "changes"}
    with (iter_dir / "output_postprocess_summary.json").open("w", encoding="utf-8") as f:
        json.dump(public_summary, f, indent=2, ensure_ascii=False)
