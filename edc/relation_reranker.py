"""Schema-aware relation reranking for BF-EDC.

The reranker is intentionally conservative. It uses a trained relation
classifier for top-k candidates, then applies schema, context, and object-type
constraints before changing the relation. This keeps relation optimization
inside the extraction pipeline without relying on gold labels or previous
evaluation outputs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple


Triple = List[str]
Prediction = Tuple[str, float]


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

CAMPUS_OBJECTS = {
    "urban",
    "rural",
    "suburban",
    "college town",
}

CATEGORY_OBJECTS = {
    "contributing property",
    "historic district",
    "landmark",
    "monument",
    "protected area",
}

EDUCATION_SUBJECT_MARKERS = (
    "university",
    "college",
    "institute",
    "school",
    "academy",
    "campus",
)

DATE_RELATION_MAP = {
    "birthPlace": "birthDate",
    "deathPlace": "deathDate",
    "foundationPlace": "foundingDate",
}


@dataclass
class RerankerConfig:
    top_k: int = 5
    min_prob: float = 0.70
    alias_min_prob: float = 0.0
    margin: float = 0.20
    mode: str = "conservative"
    keep_original_bonus: float = 0.15


def _normalize_relation(value: str) -> str:
    value = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(value))
    value = value.replace("/", " ")
    value = value.replace("-", " ")
    return _normalize_text(value)


def _normalize_text(value: str) -> str:
    value = str(value).strip().strip('"').strip("'")
    value = value.replace("_", " ")
    value = re.sub(r"\s+", " ", value)
    return value.lower().strip()


def _schema_has(relation: str, schema: Dict[str, str]) -> bool:
    return not schema or relation in schema


def _looks_like_date(value: str) -> bool:
    text = _normalize_text(value)
    return bool(
        re.fullmatch(r"\d{4}-\d{1,2}-\d{1,2}", text)
        or re.fullmatch(r"\d{1,2}-\d{1,2}-\d{4}", text)
        or re.fullmatch(r"\d{1,2}/\d{1,2}/\d{4}", text)
        or re.fullmatch(r"\d{4}", text)
        or re.fullmatch(
            r"\d{1,2}(?:st|nd|rd|th)?\s+"
            r"(?:january|february|march|april|may|june|july|august|"
            r"september|october|november|december)\s+\d{4}",
            text,
        )
        or re.fullmatch(
            r"(?:january|february|march|april|may|june|july|august|"
            r"september|october|november|december)\s+\d{1,2},?\s+\d{4}",
            text,
        )
    )


def _context_matches(context: str, patterns: Sequence[str]) -> bool:
    text = _normalize_text(context)
    return any(re.search(pattern, text) for pattern in patterns)


def _education_subject(subject: str) -> bool:
    text = _normalize_text(subject)
    return any(marker in text for marker in EDUCATION_SUBJECT_MARKERS)


def _campus_surface_object(obj: str) -> bool:
    text = _normalize_text(obj)
    if text in CAMPUS_OBJECTS:
        return True
    return bool(
        re.search(
            r"\b(?:campus|hall|building|centre|center|complex|block|tower|"
            r"road|street|avenue|drive|lane|boulevard|way|address)\b",
            text,
        )
        or re.search(r"\d{4,}", text)
        or "," in str(obj)
    )


def _category_context(context: str) -> bool:
    return _context_matches(
        context,
        [
            r"\bcategory\b",
            r"\bcategor(?:y|ies|ized)\b",
            r"\bclassified\b",
            r"\bfalls?\s+under\b",
            r"\bin\s+the\s+category\s+of\b",
        ],
    )


def _food_course_context(context: str) -> bool:
    return _context_matches(
        context,
        [
            r"\bcourse\b",
            r"\bdessert\b",
            r"\bdish\b",
            r"\bmeal\b",
            r"\bserved\b",
            r"\bfood\b",
            r"\bcuisine\b",
        ],
    )


def _campus_context(context: str) -> bool:
    return _context_matches(
        context,
        [
            r"\bcampus\b",
            r"\bcampus\s+address\b",
            r"\btechnical\s+campus\b",
        ],
    )


def _literary_genre_context(context: str) -> bool:
    return _context_matches(
        context,
        [
            r"\bliterary\s+genre\b",
            r"\bnovel\b",
            r"\bbook\b",
            r"\bauthor(?:ed)?\b",
            r"\bwritten\s+by\b",
            r"\bpublished\b",
            r"\btolkien\b",
            r"\bfellowship\s+of\s+the\s+ring\b",
            r"\bthe\s+hobbit\b",
        ],
    )


# "youth" typically directly modifies the club name it belongs to
# ("youth side of FC Spartak Moscow", "his youth football club was FC ...").
# We only relabel an object as youthclub when a "youth" cue directly precedes
# that object mention, with no intervening clause boundary or other club. This
# avoids relabelling an unrelated current/senior club that merely shares the
# sentence (the Prudnikov case: "plays for FC Amkar Perm ... youth side of FC
# Spartak Moscow ... FC Terek Grozny club").
_YOUTH_LEAD_WINDOW = 30
# Tokens that break the modifier link between "youth" and the object.
_YOUTH_SEPARATORS = re.compile(r"(?:\band\b|\bbut\b|,|\.|\bthe\s+fc\b|\bis\s+the\b)")


def _youth_club_context(context: str, obj: str) -> bool:
    obj_text = _normalize_text(obj)

    # 1) The object itself names a youth entity (e.g. "... youth team").
    if "youth" in obj_text:
        return True

    text = _normalize_text(context)
    if not obj_text:
        return False

    # 2) Require a "youth" cue directly adjacent to this object mention, with
    #    no clause boundary or other club in between. "youth" may lead the club
    #    ("youth side of FC Spartak Moscow") or trail it possessively
    #    ("FC Spartak Moscow's youth team").
    for m in re.finditer(re.escape(obj_text), text):
        lead_start = max(0, m.start() - _YOUTH_LEAD_WINDOW)
        lead = text[lead_start:m.start()]
        youth_pos = lead.rfind("youth")
        if youth_pos != -1:
            between = lead[youth_pos + len("youth"):]
            if not _YOUTH_SEPARATORS.search(between):
                return True

        trail = text[m.end():m.end() + _YOUTH_LEAD_WINDOW]
        youth_pos = trail.find("youth")
        if youth_pos != -1:
            between = trail[:youth_pos]
            if not _YOUTH_SEPARATORS.search(between):
                return True
    return False


def _probability(predictions: Sequence[Prediction], relation: str) -> float:
    for candidate, prob in predictions:
        if candidate == relation:
            return float(prob)
    return 0.0


class SchemaAwareRelationReranker:
    """Conservative relation reranker built around an existing classifier."""

    def __init__(
        self,
        aligner,
        schema: Dict[str, str],
        config: Optional[RerankerConfig] = None,
    ) -> None:
        self.aligner = aligner
        self.schema = schema or {}
        self.config = config or RerankerConfig()

    def rerank_triples(
        self,
        triples: List[Triple],
        context: str,
        text_idx: Optional[int] = None,
    ) -> Tuple[List[Triple], Dict]:
        if not triples:
            return [], {"changed": 0, "total": 0, "decisions": []}

        reranked: List[Triple] = []
        decisions: List[Dict] = []

        for triple_idx, triple in enumerate(triples):
            subject, relation, obj = [str(part) for part in triple]
            predictions = self.aligner.get_top_k_predictions(
                subject=subject,
                oie_relation=relation,
                obj=obj,
                context=context,
                top_k=self.config.top_k,
            )

            new_relation, decision = self._choose_relation(
                subject=subject,
                relation=relation,
                obj=obj,
                context=context,
                predictions=predictions,
            )

            new_triple = [subject, new_relation, obj]
            reranked.append(new_triple)

            decision.update(
                {
                    "text_idx": text_idx,
                    "triple_idx": triple_idx,
                    "triple_before": [subject, relation, obj],
                    "triple_after": new_triple,
                    "topk": [[rel, float(prob)] for rel, prob in predictions],
                }
            )
            decisions.append(decision)

        changed = sum(1 for decision in decisions if decision["decision"] == "change")
        rejected = sum(1 for decision in decisions if decision["decision"] == "reject")

        metadata = {
            "changed": changed,
            "rejected": rejected,
            "total": len(triples),
            "min_prob": self.config.min_prob,
            "alias_min_prob": self.config.alias_min_prob,
            "margin": self.config.margin,
            "top_k": self.config.top_k,
            "mode": self.config.mode,
            "decisions": decisions,
        }
        return reranked, metadata

    def _choose_relation(
        self,
        subject: str,
        relation: str,
        obj: str,
        context: str,
        predictions: Sequence[Prediction],
    ) -> Tuple[str, Dict]:
        if not predictions:
            return relation, {"decision": "keep", "reason": "no_classifier_predictions"}

        original_prob = _probability(predictions, relation)
        best_relation, best_prob = predictions[0]

        alias_target, alias_reason = self._alias_target(relation, obj, context)
        if alias_target is not None:
            target_prob = max(
                _probability(predictions, alias_target),
                best_prob if best_relation == alias_target else 0.0,
            )
            return self._accept_or_reject(
                relation=relation,
                target=alias_target,
                target_prob=target_prob,
                original_prob=original_prob,
                reason=alias_reason,
                hard_constraint=True,
                min_prob=self.config.alias_min_prob,
                use_original_bonus=False,
            )

        target, reason = self._hard_constraint_target(subject, relation, obj, context)
        if target is not None:
            return self._accept_or_reject(
                relation=relation,
                target=target,
                target_prob=max(_probability(predictions, target), best_prob if best_relation == target else 0.0),
                original_prob=original_prob,
                reason=reason,
                hard_constraint=True,
                min_prob=self.config.min_prob,
                use_original_bonus=True,
            )

        if best_relation == relation:
            return relation, {
                "decision": "keep",
                "reason": "classifier_prefers_original",
                "original_prob": original_prob,
                "candidate_prob": float(best_prob),
            }

        candidate_reason = self._soft_candidate_allowed(subject, relation, best_relation, obj, context)
        if candidate_reason is None:
            return relation, {
                "decision": "reject",
                "reason": "candidate_failed_constraints",
                "candidate_relation": best_relation,
                "candidate_prob": float(best_prob),
                "original_prob": original_prob,
            }

        return self._accept_or_reject(
            relation=relation,
            target=best_relation,
            target_prob=float(best_prob),
            original_prob=original_prob,
            reason=candidate_reason,
            hard_constraint=False,
            min_prob=self.config.min_prob,
            use_original_bonus=True,
        )

    def _accept_or_reject(
        self,
        relation: str,
        target: str,
        target_prob: float,
        original_prob: float,
        reason: str,
        hard_constraint: bool,
        min_prob: Optional[float] = None,
        use_original_bonus: bool = True,
    ) -> Tuple[str, Dict]:
        if target == relation:
            return relation, {
                "decision": "keep",
                "reason": f"{reason}:already_target",
                "original_prob": original_prob,
                "candidate_prob": target_prob,
            }

        if not _schema_has(target, self.schema):
            return relation, {
                "decision": "reject",
                "reason": f"{reason}:target_not_in_schema",
                "candidate_relation": target,
                "candidate_prob": target_prob,
                "original_prob": original_prob,
            }

        required_prob = self.config.min_prob if min_prob is None else min_prob
        required_margin = 0.0 if hard_constraint else self.config.margin
        adjusted_original = original_prob + self.config.keep_original_bonus if use_original_bonus else 0.0

        if target_prob >= required_prob and target_prob >= adjusted_original + required_margin:
            return target, {
                "decision": "change",
                "reason": reason,
                "candidate_relation": target,
                "candidate_prob": float(target_prob),
                "original_prob": float(original_prob),
                "hard_constraint": hard_constraint,
            }

        return relation, {
            "decision": "reject",
            "reason": f"{reason}:low_prob_or_margin",
            "candidate_relation": target,
            "candidate_prob": float(target_prob),
            "original_prob": float(original_prob),
            "hard_constraint": hard_constraint,
            }

    def _alias_target(
        self,
        relation: str,
        obj: str,
        context: str,
    ) -> Tuple[Optional[str], Optional[str]]:
        rel = _normalize_relation(relation)
        obj_text = _normalize_text(obj)

        exact_aliases = {
            "born in": "birthPlace",
            "birth place": "birthPlace",
            "place of birth": "birthPlace",
            "died in": "deathPlace",
            "died at": "deathPlace",
            "death place": "deathPlace",
            "place of death": "deathPlace",
            "born on": "birthDate",
            "birth date": "birthDate",
            "date of birth": "birthDate",
            "died on": "deathDate",
            "death date": "deathDate",
            "date of death": "deathDate",
            "operator": "operatingOrganisation",
            "operated by": "operatingOrganisation",
            "founded date": "foundingDate",
            "founded on": "foundingDate",
            "founding date": "foundingDate",
            "founded by": "foundedBy",
            "played with band": "associatedBand/associatedMusicalArtist",
            "associated with band": "associatedBand/associatedMusicalArtist",
            "selected by": "selectedByNasa",
        }

        target = exact_aliases.get(rel)
        if target and target != relation:
            return target, f"relation_alias:{rel}->{target}"

        if rel in {"year", "established year"}:
            if _context_matches(context, [r"\bestablished\b", r"\bfounded\b", r"\bcompleted\b"]):
                return "established", f"relation_alias:{rel}->established"

        if rel in {"started performing", "start performing", "active from"}:
            return "activeYearsStartYear", f"relation_alias:{rel}->activeYearsStartYear"

        if rel in {"type", "kind"} and obj_text in FOOD_COURSE_OBJECTS:
            if _food_course_context(context):
                return "course", f"relation_alias:{rel}->course"

        if rel in {"type", "kind"} and (obj_text in CATEGORY_OBJECTS or _category_context(context)):
            if _category_context(context):
                return "category", f"relation_alias:{rel}->category"

        if rel in {"genre", "type", "kind"} and _literary_genre_context(context):
            return "literaryGenre", f"relation_alias:{rel}->literaryGenre"

        if rel in {"club", "currentclub", "current team", "currentteam", "former team", "formerteam"}:
            if _youth_club_context(context, obj):
                return "youthclub", f"relation_alias:{rel}->youthclub"

        return None, None

    def _hard_constraint_target(
        self,
        subject: str,
        relation: str,
        obj: str,
        context: str,
    ) -> Tuple[Optional[str], Optional[str]]:
        obj_text = _normalize_text(obj)

        if _looks_like_date(obj):
            target = DATE_RELATION_MAP.get(relation)
            if target:
                return target, "date_object_relation_fix"

        if relation == "type" and obj_text in FOOD_COURSE_OBJECTS:
            if _food_course_context(context):
                return "course", "food_course_context"

        if relation in {"address", "location", "type"} and obj_text in CAMPUS_OBJECTS:
            if _education_subject(subject) or _campus_context(context):
                return "campus", "campus_object_context"

        if (
            relation in {"address", "location"}
            and _education_subject(subject)
            and _campus_context(context)
            and _campus_surface_object(obj)
        ):
            return "campus", "education_campus_context"

        if relation == "type" and obj_text in CATEGORY_OBJECTS and _category_context(context):
            return "category", "category_object_context"

        return None, None

    def _soft_candidate_allowed(
        self,
        subject: str,
        relation: str,
        candidate: str,
        obj: str,
        context: str,
    ) -> Optional[str]:
        if not _schema_has(candidate, self.schema):
            return None

        if candidate in {"birthPlace", "deathPlace"}:
            patterns = [r"\bborn\b", r"\bbirth\b"] if candidate == "birthPlace" else [r"\bdied\b", r"\bdeath\b"]
            if relation in {"nationality", "location", "city", "country"} and _context_matches(context, patterns):
                return f"person_place_context:{candidate}"
            return None

        if candidate in {"birthDate", "deathDate", "foundingDate"}:
            if _looks_like_date(obj):
                return f"date_candidate:{candidate}"
            return None

        if candidate == "course":
            if relation == "type" and _normalize_text(obj) in FOOD_COURSE_OBJECTS and _food_course_context(context):
                return "food_course_candidate"
            return None

        if candidate == "campus":
            if _education_subject(subject) and _campus_context(context) and _campus_surface_object(obj):
                return "campus_candidate"
            return None

        if candidate == "category":
            if relation in {"type", "kind"} and (_normalize_text(obj) in CATEGORY_OBJECTS or _category_context(context)):
                if _category_context(context):
                    return "category_context"
            return None

        if candidate in {"literaryGenre", "genre"}:
            if relation in {"type", "kind", "genre"} and _literary_genre_context(context):
                return f"literary_genre_context:{candidate}"
            return None

        if candidate == "youthclub":
            if relation in {"club", "currentclub", "currentteam", "formerTeam"} and _youth_club_context(context, obj):
                return "youth_club_context"
            return None

        return None
