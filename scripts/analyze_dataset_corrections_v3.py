#!/usr/bin/env python3
"""Analyze REBEL/Wiki-NRE correction opportunities, then optionally run P -> V.

Default: CPU-only analysis and EXPERIMENTAL correction previews, no model calls.
Optional --run-verifier: freshly score BOTH raw and corrected candidates and
evaluate a matched four-way ablation. Never postprocess after verification.
Reference triples are used only by diagnostics/evaluation, never by correction.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import unicodedata

VERSION = "source_guarded_experimental_v3"
DISABLED_REPAIR_RULES = ["explicit_part_of"]
SCHEMAS = {"REBEL": "schemas/rebel_official_top200_schema.csv",
           "Wiki-NRE": "schemas/wiki_nre_official_train_schema.csv"}
REPAIRS = {
    "place of birth": ("date of birth", r"born"),
    "place of death": ("date of death", r"(?:died|passed away)"),
    "cause of death": ("date of death", r"(?:died|passed away)"),
}
COORDINATED_BY = r"(?:\s*,?\s*(?:and\s+|&\s+)?(?:written|directed|produced|authored|created|scripted)){0,3}\s+by\b"
ASYMMETRIC_ROLES = {
    "director": (r"\b(?:co[- ]*)?directed\b", r"\b(?:co[- ]*)?directed\b"+COORDINATED_BY),
    "producer": (r"\b(?:co[- ]*)?produced\b", r"\b(?:co[- ]*)?produced\b"+COORDINATED_BY),
    "author": (r"\b(?:wrote|authored)\b", r"\b(?:written|authored)\b"+COORDINATED_BY),
    "screenwriter": (r"\b(?:wrote|scripted)\b", r"\b(?:written|scripted)\b"+COORDINATED_BY),
    "cast member": (r"\b(?:starred|acted|appeared)\s+in\b", r"\b(?:stars|starring)\b"),
    "creator": (r"\bcreated\b", r"\bcreated\s+by\b"),
}
ROLE_PREFIX = set("also co who which and later then first finally subsequently has had have wrote written directed produced composed authored created acted starred appeared".split())
COPULA_PREFIX = ROLE_PREFIX | set("is are was were been being a an the movie film novel book part member currently formerly originally officially".split())
CONTINENTS = {"africa", "antarctica", "asia", "europe", "north america", "south america", "oceania", "australia"}
RULE_ORDER = ["source_attested_surface_v2", "source_defined_entity_representation_v3",
              "source_supported_relation_or_direction_v3", "explicit_event_date_relation", "exact_deduplication"]

# These are lexical guards, not reference-derived entity or relation mappings.
NAME_WORD = r"[A-ZÀ-ÖØ-Þ][^\W\d_]*(?:[-’'][^\W\d_]+)*"
NAME_PHRASE = NAME_WORD + r"(?:\s+(?:" + NAME_WORD + r"|of|the|and|for|de|van|von)){1,7}"
PERSON_SUBJECT_RELATIONS = {
    "date of birth", "date of death", "place of birth", "place of death",
    "country of citizenship", "educated at", "employer", "occupation",
    "award received", "member of", "member of political party", "member of sports team",
}
UNSAFE_CLAUSE = r"\b(?:not|never|without|denied|claims?|claimed|allegedly|reportedly|rumou?red|might|may|could|would|should|if|unless|instead|rather)\b"
KINSHIP = r"\b(?:father|mother|son|daughter|brother|sister|husband|wife|cousin|uncle|aunt|nephew|niece)\b"
SPORT_PERSON = r"(?:footballer|football\s+player|basketball\s+player|baseball\s+player|hockey\s+player|rugby\s+player|cricketer|volleyball\s+player)"
ENTITY_RULES = {"explicit_acronym_wrapper_v3", "explicit_acronym_expansion_v3", "explicit_alias_expansion_v3", "unique_leading_person_name_v3"}
FEATURES = {"entity_representation": True, "expanded_relation_patterns": True}


def acronym_matches(long_name, short):
    letters = re.sub(r"[^A-Za-z]", "", short).upper()
    words = re.findall(r"[^\W\d_]+", long_name, re.UNICODE)
    full = "".join(w[0] for w in words).upper()
    content = "".join(w[0] for w in words if w.lower() not in {"a", "an", "the", "of", "for", "and", "de", "van", "von"}).upper()
    return 2 <= len(letters) <= 12 and letters in {full, content}


def entity_representation(value, relation, field, text):
    """Return a uniquely source-licensed representation, or preserve it.

    Full names are the declared representation policy for explicit aliases.
    No reference, external dictionary, arbitrary qualifier stripping or fuzzy
    entity similarity is used. Surname expansion needs an earlier leading
    full-name introduction and rejects competing names and family narratives.
    """
    if not FEATURES["entity_representation"]:
        return None
    choices = {}
    wrapper = re.fullmatch(r"(.+?)\s*\(([A-Z][A-Z.&-]{1,15})\)", value)
    if wrapper and acronym_matches(*wrapper.groups()):
        ev = source_surface(value, text)
        if ev:
            long_name = wrapper.group(1).strip()
            choices[long_name] = (long_name, "explicit_acronym_wrapper_v3", ev[1])
    if re.fullmatch(r"[A-Z][A-Z.&-]{1,15}", value):
        pattern = (r"(?P<long>" + NAME_PHRASE + r")\s*\(\s*(?P<short>" + re.escape(value) + r")\s*\)")
        for m in re.finditer(pattern, text):
            full = m.group("long")
            if acronym_matches(full, value):
                choices[full] = (full, "explicit_acronym_expansion_v3", {"start": m.start(), "end": m.end(), "span": m.group()})
    # Explicit aliases can contain a surname or an acronym; unlike a shared
    # surname alone, 'also known as' establishes the name association.
    for m in re.finditer(r"(?P<long>" + NAME_PHRASE + r")\s*\(\s*(?:also\s+)?known\s+as\s+[\"']?" + re.escape(value) + r"[\"']?\s*\)", text):
        full = m.group("long")
        if full != value:
            choices[full] = (full, "explicit_alias_expansion_v3", {"start": m.start(), "end": m.end(), "span": m.group()})
    if field == 0 and relation in PERSON_SUBJECT_RELATIONS and re.fullmatch(NAME_WORD, value) and len(value) >= 3:
        pattern = NAME_WORD + r"(?:\s+" + NAME_WORD + r"){0,2}\s+" + re.escape(value) + r"(?!\w)"
        full_names = {m.group() for m in re.finditer(pattern, text)}
        if len(full_names) == 1 and not re.search(KINSHIP, text, re.I):
            full = next(iter(full_names))
            first = re.match(r"\s*" + literal_pattern(full) + r"(?!\w)", text)
            if first and re.match(r"\s*(?:\([^)]{0,120}\)\s*)?,?\s*(?:is|was|born|graduated|served|attended)\b", text[first.end():], re.I):
                short_mentions = [m for m in mentions(value, text) if m.start() > first.end()]
                # The shortened mention must itself be a later clause subject.
                for m in short_mentions:
                    before = text[max(first.end(), m.start()-4):m.start()]
                    after = text[m.end():m.end()+55]
                    if re.search(r"[.!?]\s*$", before) and re.match(r"\s+(?:was|is|graduated|served|attended|received|won|joined|played)\b", after, re.I):
                        choices[full] = (full, "unique_leading_person_name_v3", {"start": first.start(), "end": m.end(), "span": text[first.start():m.end()]})
    return next(iter(choices.values())) if len(choices) == 1 else None


def corrected_fields(original, text):
    t, changes = list(original), []
    for field in (0, 2):
        if field == 2 and date_key(t[field]) is not None:
            continue
        found = source_surface(t[field], text)
        if found and found[0] != t[field]:
            before = list(t); t[field] = found[0]
            changes.append({"rule": "source_attested_surface_v2", "field": field, "old": before, "new": list(t), "evidence": found[1]})
        found = entity_representation(t[field], t[1], field, text)
        if found and found[0] != t[field]:
            before = list(t); t[field] = found[0]
            changes.append({"rule": found[1], "field": field, "old": before, "new": list(t), "evidence": found[2]})
    return t, changes


def clause_breaks(value):
    # Recognise periods in initialisms without permitting sentence crossings.
    value = re.sub(r"(?:[A-Z]\.){2,}", "ABBREVIATION", value)
    return bool(re.search(r"[.;!?\n]", value))


def descriptive_role_prefix(prefix, kind):
    if re.search(UNSAFE_CLAUSE + "|" + KINSHIP + r"|\b(?:by|with|who|whose|of)\b", prefix, re.I):
        return False
    if clause_breaks(prefix) or len(prefix) > 150:
        return False
    clean = re.sub(r"\b(?:18|19|20)\d{2}\b", "", prefix)
    words = re.findall(r"[^\W\d_]+", clean, re.UNICODE)
    if any(w[0].isupper() and w.lower() not in {"american", "british", "german", "french", "italian", "indian", "hindi", "english", "japanese", "canadian", "australian", "chinese", "spanish"} for w in words):
        return False
    if re.search(r"\d", clean) or len(words) > 18:
        return False
    type_pattern = r"\b(?:film|movie|series|sitcom|screenplay)\b" if kind == "film" else r"\b(?:book|novel|biography|essay|poem)\b"
    return bool(re.search(type_pattern, prefix, re.I))


def work_role_prefix(prefix, kind):
    if descriptive_role_prefix(prefix, kind):
        return True
    # A later role can follow another explicitly scoped role in the same
    # introduction: 'a film directed by A and starring B'. Parse the earlier
    # role rather than letting directed-by borrow B from the starring clause.
    first = re.search(r"\b(?:directed|produced|written|created|authored|scripted)\b", prefix, re.I)
    if first is None or not descriptive_role_prefix(prefix[:first.start()], kind):
        return False
    rest = prefix[first.start():]
    role = r"(?:directed|produced|written|created|authored|scripted)"
    agent = NAME_WORD + r"(?:\s+"+NAME_WORD+r"){0,3}"
    role_chain = role + r"(?:\s+(?:and\s+)?" + role + r"){0,3}"
    piece = role_chain+r"\s+by\s+"+agent+r"(?:\s*(?:,|and|&)\s*"+agent+r")*\s*(?:,|and|&)\s*"
    # Name case is deliberate; all supported role verbs above are lowercase.
    return not re.search(UNSAFE_CLAUSE, rest, re.I) and bool(re.fullmatch(r"(?:"+piece+r"){1,3}", rest))


def expanded_role_evidence(work, relation, agent, text):
    if not FEATURES["expanded_relation_patterns"]:
        return []
    active = {"director": r"\b(?:directs|directed)\b", "producer": r"\b(?:produces|produced)\b",
              "creator": r"\b(?:creates|created)\b", "author": r"\b(?:writes|wrote|authored)\b",
              "screenwriter": r"\b(?:writes|wrote|scripted)\b"}
    hits = explicit_event(agent, work, active[relation], text) if relation in active else []
    if relation not in ASYMMETRIC_ROLES:
        return hits
    # Only typed work descriptions are relaxed. A cast member or another agent
    # between the work and cue is not treated as part of a nominal prefix.
    kind = "book" if relation == "author" else "film"
    cue = ASYMMETRIC_ROLES[relation][1]
    for a in mentions(work, text):
        for c in re.finditer(cue, text, re.I):
            prefix = text[a.end():c.start()]
            if c.start() < a.end() or not work_role_prefix(prefix, kind):
                continue
            for b in mentions(agent, text):
                gap = text[c.end():b.start()]
                if not c.end() <= b.start() <= c.end()+120 or clause_breaks(gap):
                    continue
                if re.search(UNSAFE_CLAUSE + r"|\b(?:by|who|which|but|while|after|before|for|with|starring|stars|featuring|features|based|screenplay|score|scored|directed|produced|written|created|authored|scripted)\b", gap, re.I):
                    continue
                hits.append({"start": a.start(), "end": b.end(), "span": text[a.start():b.end()], "predicate": c.group(), "template": "typed_work_description_v3"})
    return hits


def sports_membership_evidence(subject, obj, text):
    """Typed athlete plus local played-for/joins event; no generic affiliation."""
    if not FEATURES["expanded_relation_patterns"]:
        return []
    actors = []
    for a in mentions(subject, text):
        tail = text[a.end():a.end()+220]
        intro = re.match(r"\s*(?:\([^)]{0,110}\)\s*)?,?\s*(?:is|was)\s+[^.;!?\n]{0,90}?\b" + SPORT_PERSON + r"\b", tail, re.I)
        # A birth-date month such as May inside the introduction's parenthesis
        # is not a modal verb governing the membership assertion.
        if intro and not re.search(KINSHIP + "|" + UNSAFE_CLAUSE, re.sub(r"\([^)]*\)", "", intro.group()), re.I):
            actors.append((a, intro))
    # A typed team endpoint is sufficient for a directly attached action.
    typed_team = bool(re.search(r"\b(?:team|F\.?C\.?|football club|basketball club)\b", obj, re.I))
    anchors = [(a, a.end(), "named_actor") for a in mentions(subject, text)] if typed_team else []
    for a, intro in actors:
        anchors.append((a, a.end(), "typed_athlete"))
        # Resolve only the first immediately following subject pronoun in a
        # leading biography; any named intervening clause or kinship blocks it.
        if not text[:a.start()].strip() and not re.search(KINSHIP, text, re.I):
            intro_end = a.end()+intro.end()
            after = text[intro_end:intro_end+100]
            next_sentence = re.match(r"[^.;!?\n]{0,65}[.!?]\s+(He|She|They)\b", after)
            if next_sentence and not re.search(NAME_WORD, after[:next_sentence.start(1)]):
                start = intro_end+next_sentence.start(1)
                end = intro_end+next_sentence.end(1)
                proxy = re.match(r"(?:He|She|They)\b", text[start:])
                anchors.append((proxy, end, "leading_biography_pronoun"))
    event = (r"\b(?:plays?|played|playing)\b"
             r"(?:\s+during\s+(?:his|her|their)\s+career)?"
             r"(?:\s+as\s+(?:an?\s+)?(?:forward|midfielder|defender|goalkeeper|striker|winger|guard|pitcher))?"
             r"(?:\s+for\s+\d+\s+seasons?\s*(?:\([^)]{0,55}\))?\s+in\s+(?:the\s+)?[A-Za-z0-9 -]{1,45})?"
             r"\s+for\b|\b(?:joined|joins)\b")
    hits = []
    for a, end, kind in anchors:
        for c in re.finditer(event, text, re.I):
            if not end <= c.start() <= end+230:
                continue
            prefix = text[end:c.start()]
            allowed = not prefix.strip() or bool(re.fullmatch(r"\s*(?:also|later|then|subsequently|first)\s+", prefix, re.I))
            if kind == "typed_athlete":
                allowed = allowed or bool(re.fullmatch(r"\s*(?:\([^)]{0,110}\)\s*)?,?\s*(?:is|was)\s+[^.;!?\n]{0,90}?\b"+SPORT_PERSON+r"\s+(?:who|and)\s+", prefix, re.I))
            if not allowed or clause_breaks(prefix) or re.search(UNSAFE_CLAUSE + "|" + KINSHIP, re.sub(r"\([^)]*\)", "", prefix), re.I):
                continue
            for b in mentions(obj, text):
                gap = text[c.end():b.start()]
                if not c.end() <= b.start() <= c.end()+150 or clause_breaks(gap):
                    continue
                if re.search(UNSAFE_CLAUSE+r"|\b(?:against|but|who|which|while|before|after|for|in|on|at|by|with|coached|managed|visited|played|joined)\b", gap, re.I):
                    continue
                start = end-len(a.group()) if kind == "leading_biography_pronoun" else a.start()
                hits.append({"start": start, "end": b.end(), "span": text[start:b.end()], "predicate": c.group(), "template": kind+"_played_for_v3"})
    return hits


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")


def jsonl(path, records):
    Path(path).write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in records), encoding="utf-8")


def read_rows(path):
    rows = []
    for i, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        try:
            row = ast.literal_eval(line)
        except (ValueError, SyntaxError) as e:
            raise ValueError(f"{path}:{i}: invalid triple list") from e
        if not isinstance(row, list) or any(not isinstance(t, (list, tuple)) or len(t)!=3 or
                                           any(not isinstance(v,str) for v in t) for t in row):
            raise ValueError(f"{path}:{i}: expected list of string triples")
        rows.append([list(t) for t in row])
    return rows


def write_rows(path, rows):
    jsonl(path, rows)


def label(t):
    return " | ".join(t).lower()


def date_key(value):
    """Diagnostic equivalence only; never renders dates into predicted output."""
    value = value.strip().strip('\"').strip("'")
    value = re.sub(r"(\d)(?:st|nd|rd|th)\b", r"\1", value, flags=re.I)
    value = re.sub(r"\s+", " ", value)
    if re.fullmatch(r"\d{4}", value):
        return ("year", int(value)) if 1000 <= int(value) <= 2100 else None
    for fmt in ("%Y-%m-%d", "%B %d, %Y", "%b %d, %Y", "%d %B %Y", "%d %b %Y",
                "%B %d %Y", "%b %d %Y"):
        try:
            d = datetime.strptime(value, fmt)
            return ("day", d.year, d.month, d.day)
        except ValueError:
            pass
    return None


def diagnostic_key(value):
    value = unicodedata.normalize("NFKC", value)
    value = value.translate(str.maketrans({"’":"'", "‘":"'", "“":'"', "”":'"', "–":"-", "—":"-"}))
    return re.sub(r"\s+", " ", value.replace("_", " ").strip().strip('\"').strip("'")).lower()


def literal_pattern(value):
    return r"\s+".join(re.escape(w) for w in value.split())


def source_surface(value, text):
    """Only restore observed punctuation/spacing; never invent an entity alias."""
    value = value.strip()
    # Strip a balanced OUTER serialization wrapper, never internal apostrophes.
    # The unwrapped form must still be attested in the source below.
    pairs = {'"':'"', "'":"'", '“':'”', '‘':'’'}
    while len(value)>2 and value[0] in pairs and value[-1]==pairs[value[0]]:
        value=value[1:-1].strip()
    value = re.sub(r"\s+", " ", value.replace("_", " "))
    chars = []
    for c in value:
        if c in "-–—": chars.append("[-–—]")
        elif c in "'’‘": chars.append("['’‘]")
        elif c in '\"“”': chars.append('[\"“”]')
        elif c.isspace(): chars.append(r"\s+")
        else: chars.append(re.escape(c))
    if not chars: return None
    pattern = r"(?<!\w)"+"".join(chars)+r"(?!\w)"
    spans = list(re.finditer(pattern, text, flags=re.I))
    values = {m.group() for m in spans}
    # Conflicting source spellings are ambiguous; do not guess between them.
    if len(values) != 1: return None
    m = spans[0]
    if diagnostic_key(m.group()) != diagnostic_key(value): return None
    # Case-only changes do not help the lowercase Full metric; preserve them.
    if value.lower() == m.group().lower():
        restored = value
    else:
        restored = m.group()
    # Use ordinary whitespace rather than injecting invisible benchmark-specific
    # characters. We never infer a nonbreaking space from the reference answer.
    restored=re.sub(r"\s+", " ", restored)
    return restored, {"start":m.start(), "end":m.end(), "span":m.group()}


def mentions(value, text):
    return list(re.finditer(r"(?<!\w)"+literal_pattern(value)+r"(?!\w)",text,re.I))


def explicit_event(first, second, cue, text, copula=False):
    """Return a clause-local first -> predicate -> second evidence span.

    Prefixes cannot introduce another named person or a relative's action.
    Negation, hedging, by-agents and sentence crossings block this template.
    Lists after one predicate are allowed; unrelated clauses are not.
    """
    allowed=COPULA_PREFIX if copula else ROLE_PREFIX
    hits=[]
    for a in mentions(first,text):
        for c in re.finditer(cue,text,re.I):
            if not a.end()<=c.start()<=a.end()+100:continue
            prefix=text[a.end():c.start()]
            if re.search(r"[.;!?\n]",prefix):continue
            words=re.findall(r"[A-Za-z]+",prefix.lower())
            if len(words)>12 or any(w not in allowed for w in words):continue
            if re.search(r"\d|['’]s\b",prefix):continue
            for b in mentions(second,text):
                if not c.end()<=b.start()<=c.end()+180:continue
                gap=text[c.end():b.start()]
                if re.search(r"[.;!?\n]|\b(?:by|not|never|without|rather|instead|denied|claimed|allegedly)\b",gap,re.I):continue
                hits.append({"start":a.start(),"end":b.end(),"span":text[a.start():b.end()],"predicate":c.group()})
    unique={(h["start"],h["end"],h["predicate"]):h for h in hits}
    return list(unique.values())


def work_kind(work, text):
    """Type evidence stays near the same work mention, not elsewhere in text."""
    kinds=set()
    for m in mentions(work,text):
        before=text[max(0,m.start()-50):m.start()]
        after=text[m.end():m.end()+90]
        for kind,terms in [("film",r"(?:film|movie|screenplay|television series|sitcom)"),
                           ("book",r"(?:book|novel|biography|essay|poem)")]:
            if re.search(r"\b"+terms+r"\s*[\"'“‘]?\s*$",before,re.I):kinds.add(kind)
            if re.match(r"\s*[\"'”’]?\s*(?:,\s*)?(?:is|was)\s+(?:an?\s+)?[^.;!?\n]{0,60}\b"+terms+r"\b",after,re.I):kinds.add(kind)
    return kinds


def base_role_evidence(work, relation, agent, text):
    if relation not in ASYMMETRIC_ROLES:return []
    if relation=="screenwriter" and "film" not in work_kind(work,text):return []
    if relation=="author" and "book" not in work_kind(work,text):return []
    if relation=="cast member" and "film" not in work_kind(work,text):return []
    active,passive=ASYMMETRIC_ROLES[relation]
    ev=explicit_event(agent,work,active,text)
    # Cast "Film starring Actor" is canonical too, but avoid treating "Actor
    # features Film" as reversed membership just from shared words.
    ev+=explicit_event(work,agent,passive,text,copula=True)
    return ev


def role_evidence(work, relation, agent, text):
    ev = base_role_evidence(work, relation, agent, text)
    if relation == "screenwriter" and "film" not in work_kind(work, text):
        # The typed-description recogniser below supplies its own work type.
        return ev + [h for h in expanded_role_evidence(work, relation, agent, text)
                     if h.get("template") == "typed_work_description_v3"]
    if relation == "author" and "book" not in work_kind(work, text):
        return ev + [h for h in expanded_role_evidence(work, relation, agent, text)
                     if h.get("template") == "typed_work_description_v3"]
    if relation == "cast member" and "film" not in work_kind(work, text):
        return ev + [h for h in expanded_role_evidence(work, relation, agent, text)
                     if h.get("template") == "typed_work_description_v3"]
    return ev + expanded_role_evidence(work, relation, agent, text)


def possible_current_work_role(work, relation, agent, text):
    """Abstention guard, never evidence for a replacement.

    If the source mentions the current role and agent in the same local work
    description but this parser cannot confidently recover its scope, do not
    replace it with another recognised role. This protects coordinated roles,
    screenplay credits, and cast introduced with 'featuring'.
    """
    cues = {"director": r"\bdirect(?:ed|s|or)\b", "producer": r"\bproduc(?:ed|es|er)\b",
            "screenwriter": r"\b(?:writ(?:ten|es|er)|wrote|screenplay|script(?:ed|writer))\b",
            "author": r"\b(?:author(?:ed)?|writ(?:ten|es|er)|wrote)\b",
            "creator": r"\bcreat(?:ed|es|or)\b",
            "cast member": r"\b(?:star(?:ring|s|red)?|featur(?:ing|es|ed)|cast|actor|actress)\b"}
    if not FEATURES["expanded_relation_patterns"] or relation not in cues:
        return False
    for m in mentions(work, text):
        local = text[m.end():m.end()+300]
        # Only abstain within this sentence; abbreviation handling is not
        # needed for safety because truncation here merely reduces abstention.
        local = re.split(r"[.!?\n]", local, maxsplit=1)[0]
        if re.search(cues[relation], local, re.I) and mentions(agent, local):
            return True
    return False


def literal_relation_evidence(s,r,o,text):
    cues={
        "named after":r"\bnamed\s+(?:after|for|in\s+honou?r\s+of)\b",
        "part of":r"\bpart\s+of\b",
        "serves":r"\bserves?\b",
        "country of citizenship":r"\b(?:citizen\s+of|citizenship\s+(?:of|in))\b",
        "member of":r"\b(?:member\s+of|part\s+of|joined|represents?)\b",
        "member of political party":r"\b(?:member\s+of|represents?|MP\s+for|leader\s+of)\b",
        "member of sports team":r"\b(?:played\s+for|plays\s+for|part\s+of|joined|represents?)\b",
    }
    if r in {"location","located in the administrative territorial entity","located on terrain feature"}:
        return explicit_event(s,o,r"\b(?:located|situated)\s+(?:in|on|at)\b",text,copula=True)
    return explicit_event(s,o,cues[r],text,copula=True) if r in cues else []


def semantic_proposals(t,text,relations):
    """Return source-licensed alternatives, never reference-derived labels."""
    s,r,o=t;proposals=[]
    if r in ASYMMETRIC_ROLES:
        # Preserve a text-supported existing role even if another role co-occurs.
        if role_evidence(s,r,o,text):return []
        # Prefer correcting the direction of the SAME label over choosing
        # arbitrarily among other co-occurring roles of the same person.
        reversed_support=role_evidence(o,r,s,text) if r in relations else []
        if reversed_support:return [([o,r,s],"explicit_same_relation_direction",reversed_support[0])]
        if possible_current_work_role(s,r,o,text):return []
        for target in ASYMMETRIC_ROLES:
            if target not in relations:continue
            for ns,no in [(s,o),(o,s)]:
                for evidence in role_evidence(ns,target,no,text):
                    new=[ns,target,no]
                    if new!=t:proposals.append((new,"explicit_work_role",evidence))
    else:
        # A generic membership relation can be refined to an explicitly typed
        # team/party relation. Other independently supported facts are preserved.
        if r not in {"member of","affiliation","position held","country"} and literal_relation_evidence(s,r,o,text):return []
        target_specs=[]
        if r in {"serves","location","located in the administrative territorial entity","located on terrain feature","part of"}:
            target_specs.append(("named after","explicit_named_after"))
        # v2.1: part-of replacement is disabled. Its previous broad span could
        # jump across nested geographic phrases. Part-of wording may still
        # support an explicitly typed team membership, a separate rule.
        if r in {"member of","affiliation","position held"} and re.search(r"\b(?:party|democrats|republicans)\b",o,re.I):
            target_specs.append(("member of political party","explicit_party_membership"))
        if r in {"member of","affiliation"} and re.search(r"\b(?:team|F\.?C\.?|football club|basketball club)\b",o,re.I):
            target_specs.append(("member of sports team","explicit_team_membership"))
        if r in {"member of", "affiliation"} and "member of sports team" in relations:
            for evidence in sports_membership_evidence(s, o, text):
                proposals.append(([s, "member of sports team", o], "explicit_athlete_membership_v3", evidence))
        if r=="country":target_specs.append(("country of citizenship","explicit_citizenship"))
        for target,rule in target_specs:
            if target not in relations:continue
            for evidence in literal_relation_evidence(s,target,o,text):
                proposals.append(([s,target,o],rule,evidence))
        if r=="country" and o.lower() in CONTINENTS and "continent" in relations:
            for evidence in explicit_event(s,o,r"\b(?:in|on)\b",text,copula=True):
                proposals.append(([s,"continent",o],"continent_object_and_source",evidence))
    # Multiple licensed replacements are ambiguous; propose() will preserve.
    unique={tuple(new):(new,rule,ev) for new,rule,ev in proposals}
    return list(unique.values())


def relation_evidence(subject, obj, cue, text):
    # Require subject immediately attached to an explicit event predicate.
    # Thus "A's father died ..." does not license repairing a fact about A.
    pattern = (r"(?<!\w)"+literal_pattern(subject)+
               r"\s*(?:\(\s*)?(?:(?:was|is|had)\s+)?"+cue+
               r"\b[^.;!?\n]{0,60}?"+literal_pattern(obj.strip('\"').strip("'"))+r"(?!\w)")
    matches = list(re.finditer(pattern, text, flags=re.I))
    if len(matches) != 1: return None
    m = matches[0]
    return {"start":m.start(), "end":m.end(), "span":m.group()}


def propose(rows, texts, relations):
    """Reference-free experimental P. Fixed order: surfaces -> relation -> dedupe."""
    output, edits = [], []
    for i, (row, text) in enumerate(zip(rows, texts), 1):
        newrow, seen = [], set()
        for j, original in enumerate(row):
            t, field_changes = corrected_fields(original, text)
            edits.extend(dict(e, line=i, triple_index=j) for e in field_changes)
            alternatives=semantic_proposals(t,text,relations)
            if len(alternatives)==1:
                before=list(t);t,rule,evidence=alternatives[0];t=list(t)
                edits.append({"line":i,"triple_index":j,"rule":rule,"old":before,"new":list(t),
                              "argument_swap":before[0]==t[2] and before[2]==t[0],"evidence":evidence})
            repair = REPAIRS.get(t[1].lower())
            if repair and date_key(t[2]) is not None and repair[0] in relations:
                evidence = relation_evidence(t[0], t[2], repair[1], text)
                if evidence:
                    before = list(t); t[1] = repair[0]
                    edits.append({"line":i,"triple_index":j,"rule":"explicit_event_date_relation",
                                  "old":before,"new":list(t),"evidence":evidence})
            key = tuple(t)
            if key in seen:
                edits.append({"line":i,"triple_index":j,"rule":"exact_deduplication",
                              "old":list(t),"new":None,"evidence":None})
                continue
            seen.add(key); newrow.append(t)
        output.append(newrow)
    return output, edits


def guard_diagnostics(rows,texts,relations):
    blocked=[]
    for i,(row,text) in enumerate(zip(rows,texts),1):
        for j,original in enumerate(row):
            t, _ = corrected_fields(original, text)
            options=semantic_proposals(t,text,relations)
            if len(options)>1:
                blocked.append({"line":i,"triple_index":j,"reason":"multiple_source_supported_replacements",
                                "triple":t,"options":[{"triple":n,"rule":r,"evidence":e} for n,r,e in options],
                                "source_text":text,"uses_reference":False})
            if (t[1] in {"location","located in the administrative territorial entity","located on terrain feature"}
                and "part of" in relations and literal_relation_evidence(t[0],"part of",t[2],text)
                and not literal_relation_evidence(t[0],t[1],t[2],text)):
                blocked.append({"line":i,"triple_index":j,"reason":"part_of_repair_disabled_v2_1",
                                "triple":t,"source_text":text,"uses_reference":False})
            repair=REPAIRS.get(t[1].lower())
            if repair and date_key(t[2]) is not None and repair[0] not in relations:
                blocked.append({"line":i,"triple_index":j,"reason":"target_relation_not_in_schema",
                                "triple":t,"blocked_target":repair[0],"source_text":text,"uses_reference":False})
    return blocked


def error_category(t, refs):
    """Reference-aware diagnostic labels. Never called by propose()."""
    lower = [x.lower() for x in t]
    rs = [[x.lower() for x in r] for r in refs]
    if any(tuple(map(diagnostic_key,t)) == tuple(map(diagnostic_key,r)) for r in refs):
        return "surface_equivalence_candidate"
    if date_key(t[2]) and any(lower[:2]==r[:2] and date_key(t[2])==date_key(r[2]) for r in rs):
        return "date_rendering_difference"
    if any(lower[0]==r[0] and lower[2]==r[2] for r in rs):
        return "relation_label_difference"
    if any(lower==[r[2],r[1],r[0]] for r in rs):
        return "argument_direction_difference"
    if any(lower[:2]==r[:2] for r in rs):
        return "object_difference"
    if any(lower[1:]==r[1:] for r in rs):
        return "subject_difference"
    return "other_reference_nonmatch"


def metric_report(rows, refs):
    counts = defaultdict(lambda:[0,0,0]); tp=fp=fn=0
    for row, ref in zip(rows, refs):
        p,g = {label(t) for t in row},{label(t) for t in ref}
        tp+=len(p&g);fp+=len(p-g);fn+=len(g-p)
        for t in p|g:
            c=counts[t];c[0]+=t in p;c[1]+=t in g;c[2]+=t in p and t in g
    n=len(counts)
    return {"full_p":sum(c/a if a else 0 for a,b,c in counts.values())/n if n else 0.,
            "full_r":sum(c/b if b else 0 for a,b,c in counts.values())/n if n else 0.,
            "full_f1":sum(2*c/(a+b) for a,b,c in counts.values())/n if n else 0.,
            "label_count":n,"micro_tp":tp,"micro_fp":fp,"micro_fn":fn,
            "micro_f1":2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.}


def resolve(root, path):
    p=Path(path)
    return p.resolve() if p.is_absolute() else (root/p).resolve()


def load_inputs(args, name, root, base, manifest):
    custom = args.raw is not None
    cfg = manifest["datasets"][name]
    paths = {"raw":resolve(root,args.raw) if custom else base/name/"inputs/raw.txt",
             "texts":resolve(root,args.texts) if custom else base/name/"inputs/dataset.txt",
             "reference":resolve(root,args.reference) if args.reference else None if custom or args.without_reference else base/name/"inputs/reference.txt",
             "schema":resolve(root,args.schema) if args.schema else root/SCHEMAS[name]}
    if not custom:
        for k,old in [("raw","raw"),("texts","dataset"),("reference","reference")]:
            if paths[k] is not None and sha(paths[k])!=cfg["files"][cfg[old]]["sha256"]:
                raise ValueError(f"Saved input snapshot hash mismatch: {paths[k]}")
    rows=read_rows(paths["raw"]);texts=paths["texts"].read_text(encoding="utf-8-sig").splitlines()
    if len(rows)!=len(texts):raise ValueError("Candidate/text row mismatch")
    refs=read_rows(paths["reference"]) if paths["reference"] is not None else None
    if refs is not None and len(refs)!=len(rows):raise ValueError("Reference row mismatch")
    relations=set()
    if paths["schema"].is_file():
        with paths["schema"].open(encoding="utf-8-sig",newline="") as f:
            relations={r[0].strip() for r in csv.reader(f) if r and r[0].strip()}
    elif args.schema or args.run_verifier:
        raise FileNotFoundError(f"Required target schema missing: {paths['schema']}")
    if not relations and (args.schema or args.run_verifier):
        raise ValueError(f"Target relation schema is empty: {paths['schema']}")
    return cfg,paths,rows,texts,refs,relations


def run_command(command, cwd, env, log):
    with log.open("w",encoding="utf-8") as f:
        subprocess.run(command,cwd=cwd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)


def read_evaluation_metrics(path):
    text=Path(path).read_text(encoding="utf-8")
    top=text.split("Scores per tag")[0];metrics={}
    for title,key in [("Ent_type","ent_type"),("Partial","partial"),("Strict","strict"),("Exact","exact")]:
        m=re.search(r"(?:^|\n)"+title+r"\n.*?Precision:\s*([\d.eE+-]+)\s+Recall:\s*([\d.eE+-]+)\s+F1:\s*([\d.eE+-]+)",top,re.S)
        if not m:raise ValueError(f"Missing {title}: {path}")
        metrics.update({key+"_"+k:float(v) for k,v in zip(("p","r","f1"),m.groups())})
    m=re.search(r"Full triple scores.*?Precision:\s*([\d.eE+-]+)\s+Recall:\s*([\d.eE+-]+)\s+F1:\s*([\d.eE+-]+)",text,re.S)
    if not m:raise ValueError(f"Missing Full scores: {path}")
    metrics.update({"full_"+k:float(v) for k,v in zip(("p","r","f1"),m.groups())})
    return metrics


def audit_corrections_after_verifier(d,refs):
    """Track every new P label, including which reference gains survive V."""
    raw=read_rows(d/"A4_upstream/iter0/canon_kg.txt")
    corrected=read_rows(d/"A2_proposed_local/iter0/canon_kg.txt")
    final=read_rows(d/"A1_proposed_full/iter0/canon_kg.txt")
    verifier_only=read_rows(d/"A3_verifier_only/iter0/canon_kg.txt")
    score_records=[json.loads(s) for s in (d/"A1_proposed_full/fresh_scores.jsonl").read_text(encoding="utf-8").splitlines()]
    scores={(int(s["line"]),int(s["triple_index"])):float(s["score"]) for s in score_records}
    edit_path=d/"proposed_edits.jsonl"
    edits=[json.loads(s) for s in edit_path.read_text(encoding="utf-8").splitlines()] if edit_path.exists() else []
    audit=[];pair_gained=pair_lost=0
    for i,(a,p,b,v) in enumerate(zip(raw,corrected,final,verifier_only),1):
        sa,sp,sb,sv=({label(t) for t in rr} for rr in (a,p,b,v))
        gold={label(t) for t in refs[i-1]} if refs is not None else None
        if gold is not None:
            pair_gained+=len((sb-sv)&gold);pair_lost+=len((sv-sb)&gold)
        for new in sorted(sp-sa):
            indices=[j for j,t in enumerate(p) if label(t)==new]
            relevant=[e for e in edits if e["line"]==i and e.get("new") is not None and label(e["new"])==new]
            audit.append({"line":i,"triple":p[indices[0]],"corrected_indices":indices,
                          "rules":sorted({e["rule"] for e in relevant}),"source_edits":relevant,
                          "scores":[scores[(i,j)] for j in indices],"kept_by_verifier":new in sb,
                          "matches_reference":new in gold if gold is not None else None,
                          "reference_match_is_not_manual_factuality":True})
    jsonl(d/"correction_verifier_audit.jsonl",audit)
    matched=[r for r in audit if r["matches_reference"] is True]
    stats={"new_unique_labels_from_correction":len(audit),"pre_verifier_new_reference_matches":len(matched),
           "new_reference_matches_retained":sum(r["kept_by_verifier"] for r in matched),
           "new_reference_matches_rejected":sum(not r["kept_by_verifier"] for r in matched),
           "A1_vs_A3_unique_reference_matches_gained":pair_gained if refs is not None else None,
           "A1_vs_A3_unique_reference_matches_lost":pair_lost if refs is not None else None}
    dump(d/"correction_verifier_summary.json",stats)
    return stats


def fresh_four_way(root, d, cfg, paths, args):
    """A1=P->fresh V; A3=raw->fresh V. Filtering has no postprocessing branch."""
    env=dict(os.environ);env["PYTHONPATH"]=str(root)
    raw=d/"A4_upstream/iter0/canon_kg.txt";p=d/"A2_proposed_local/iter0/canon_kg.txt"
    model=resolve(root,args.model_dir or cfg["checkpoint"])
    if not model.is_dir():raise FileNotFoundError(f"Verifier checkpoint missing: {model}")
    if cfg["policy"]:
        policy=d/"threshold_policy.json"
        if args.threshold_policy:
            shutil.copyfile(resolve(root,args.threshold_policy),policy)
        else:
            # Copy policy from fixed ablation input, not a newly selected threshold.
            shutil.copyfile(resolve(root,args.ablation_dir)/cfg["dataset_name"]/"threshold_policy.json",policy)
        selection=["--threshold_policy",str(policy)]
        policy_value=json.loads(policy.read_text(encoding="utf-8"))
    else:
        if args.threshold_policy:
            raise ValueError("Wiki-NRE fixed comparison uses its global threshold; no relation policy override")
        selection=["--threshold",str(cfg["threshold"])]
        policy_value=None
    commands=[]
    dump(d/"verifier_configuration.json",{
        "checkpoint":str(model),"checkpoint_files":{p.name:sha(p) for p in sorted(model.iterdir()) if p.is_file()},
        "threshold":cfg["threshold"],"threshold_policy":policy_value,
        "max_length":cfg["scoring_max_length"],"context_window_chars":cfg["context_window_chars"],
        "batch_size":args.batch_size,"fresh_scores_required":True,"post_verifier_processing":False,
        "score_script_sha256":sha(root/"scripts/rebel/score_rebel_entailment.py") if (root/"scripts/rebel/score_rebel_entailment.py").is_file() else None,
        "filter_script_sha256":sha(root/"scripts/rebel/apply_rebel_entailment_scores.py") if (root/"scripts/rebel/apply_rebel_entailment_scores.py").is_file() else None})
    for branch,input_path in [("A1_proposed_full",p),("A3_verifier_only",raw)]:
        branch_dir=d/branch;branch_dir.mkdir()
        scores=branch_dir/"fresh_scores.jsonl"
        score_cmd=[sys.executable,str(root/"scripts/rebel/score_rebel_entailment.py"),
                   "--raw",str(input_path),"--dataset",str(paths["texts"]),"--model_dir",str(model),
                   "--output",str(scores),"--max_length",str(cfg["scoring_max_length"]),
                   "--batch_size",str(args.batch_size),"--structured_input",
                   "--context_window_chars",str(cfg["context_window_chars"])]
        print("FRESH SCORE",branch,flush=True)
        run_command(score_cmd,root,env,branch_dir/"scoring.log")
        rows=read_rows(input_path);sc={}
        for line in scores.read_text(encoding="utf-8").splitlines():
            r=json.loads(line);i,j=int(r["line"]),int(r["triple_index"])
            if (i,j) in sc or rows[i-1][j]!=r["triple"] or not 0<=float(r["score"])<=1:
                raise ValueError("Fresh score alignment/value failure")
            sc[(i,j)]=r["score"]
        if set(sc)!={(i,j) for i,row in enumerate(rows,1) for j in range(len(row))}:
            raise ValueError("Fresh scores incomplete: refusing to reuse old scores")
        apply_cmd=[sys.executable,str(root/"scripts/rebel/apply_rebel_entailment_scores.py"),
                   "--raw",str(input_path),"--dataset",str(paths["texts"]),"--scores",str(scores),
                   "--output_dir",str(branch_dir),"--missing_policy","keep"]+selection
        if "--profile" in apply_cmd:raise AssertionError("Post-verifier processing is forbidden")
        run_command(apply_cmd,root,env,branch_dir/"filtering.log")
        expected=[]
        for i,row in enumerate(rows,1):
            keep=[]
            for j,t in enumerate(row):
                threshold=(policy_value.get("thresholds",{}).get(t[1],policy_value.get("default_threshold",cfg["threshold"]))
                           if policy_value else cfg["threshold"])
                if float(sc[(i,j)])>=threshold:keep.append(t)
            expected.append(keep)
        if expected!=read_rows(branch_dir/"iter0/canon_kg.txt"):
            raise AssertionError("Verifier filtering altered candidates beyond score-based keep/drop")
        commands.extend([score_cmd,apply_cmd])
    results=[]
    for branch in ["A1_proposed_full","A2_proposed_local","A3_verifier_only","A4_upstream"]:
        pred=d/branch/"iter0/canon_kg.txt";rows=read_rows(pred)
        record={"configuration":branch,"candidates":sum(map(len,rows)),"prediction_sha256":sha(pred)}
        if paths["reference"] is not None:
            cmd=[sys.executable,str(root/"evaluate/evaluation_script1.py"),"--edc_output",str(pred),
                 "--reference",str(paths["reference"]),"--evaluation_output",str(pred.parent/"evaluation_scores_script1.txt")]
            run_command(cmd,d/branch,env,d/branch/"evaluation.log");commands.append(cmd)
            record["metrics"]=metric_report(rows,read_rows(paths["reference"]))
            record["evaluation_metrics"]=read_evaluation_metrics(pred.parent/"evaluation_scores_script1.txt")
            for key in ("full_p","full_r","full_f1"):
                if abs(record["metrics"][key]-record["evaluation_metrics"][key])>1e-12:
                    raise AssertionError("Independent Full metrics disagree with evaluator")
        results.append(record)
        if "evaluation_metrics" in record:
            print("EVALUATED",cfg.get("dataset_name","test"),branch,record["evaluation_metrics"]["full_f1"],flush=True)
    audit_corrections_after_verifier(d,read_rows(paths["reference"]) if paths["reference"] is not None else None)
    dump(d/"fresh_four_way_results.json",results);dump(d/"commands.json",commands)
    return results


def self_test():
    import unittest
    class Guards(unittest.TestCase):
        def test_source_surface(self):
            out,_=propose([[['New_York','country','United_States']]],['New York is in the United States.'],set())
            self.assertEqual(out,[[['New York','country','United States']]])
        def test_single_quotes_and_nbsp(self):
            row=[[['"A Title"','publication date','2001'],['King\u00a0X','spouse','Queen Y']]]
            out,_=propose(row,['The book "A Title" was published in 2001. King X married Queen Y.'],set())
            self.assertEqual(out[0][0][0],'A Title')
            self.assertEqual(out[0][1][0],'King X')
            self.assertEqual(propose([[['Ghost','r','B']]],['A r B.'],set())[0],[[['Ghost','r','B']]])
        def test_explicit_director_direction(self):
            row=[[['Person','director','A Title']]]
            out,_=propose(row,['Person directed the film "A Title".'],{'director','screenwriter','cast member','creator'})
            self.assertEqual(out,[[['A Title','director','Person']]])
        def test_role_relation_repair(self):
            row=[[['A Title','producer','Person']]]
            out,_=propose(row,['A Title was directed by Person.'],{'director','producer'})
            self.assertEqual(out,[[['A Title','director','Person']]])
        def test_keep_cooccurring_writer_and_director(self):
            text='Person wrote and directed the film A Title.'
            row=[[['A Title','screenwriter','Person'],['A Title','director','Person']]]
            self.assertEqual(propose(row,[text],{'screenwriter','director','producer'})[0],row)
            ambiguous=[[['A Title','producer','Person']]]
            self.assertEqual(propose(ambiguous,[text],{'screenwriter','director','producer'})[0],ambiguous)
            reversed_row=[[['Person','director','A Title']]]
            self.assertEqual(propose(reversed_row,[text],{'screenwriter','director','producer'})[0],[[['A Title','director','Person']]])
            coordinated=[[['A Title','producer','Person'],['A Title','director','Person']]]
            self.assertEqual(propose(coordinated,['The film A Title was produced and directed by Person.'],{'producer','director'})[0],coordinated)
        def test_no_symmetric_flip_or_negative_inference(self):
            row=[[['Person','spouse','Partner']]]
            self.assertEqual(propose(row,['Partner was married to Person.'],{'spouse'})[0],row)
            row=[[['Person','director','A Title']]]
            for text in ['Person never directed the film A Title.',"Person's father directed the film A Title.",'Person allegedly directed the film A Title.']:
                self.assertEqual(propose(row,[text],{'director'})[0],row)
        def test_named_after_and_old_fact_preservation(self):
            row=[[['Bridge','location','Person']]]
            self.assertEqual(propose(row,['Bridge was named after Person.'],{'named after','location'})[0],[[['Bridge','named after','Person']]])
            row=[[['Station','serves','Town']]]
            self.assertEqual(propose(row,['Station serves Town and was named after Town.'],{'serves','named after'})[0],row)
        def test_specific_membership(self):
            row=[[['Person','member of','Example national football team']]]
            out,_=propose(row,['Person was part of the Example national football team.'],{'member of','member of sports team'})
            self.assertEqual(out,[[['Person','member of sports team','Example national football team']]])
            row=[[['Person','member of','Example Party']]]
            out,_=propose(row,['Person represents the Example Party.'],{'member of','member of political party'})
            self.assertEqual(out,[[['Person','member of political party','Example Party']]])
        def test_part_of_repair_is_disabled(self):
            row=[[['Hall','location','University']]]
            self.assertEqual(propose(row,['Hall of University is open.'],{'location','part of'})[0],row)
            self.assertEqual(propose(row,['Hall is part of University.'],{'location','part of'})[0],row)
            nested=[[['Peak','located on terrain feature','Ranges']]]
            self.assertEqual(propose(nested,['Peak is part of a smaller range of Ranges.'],{'part of','located on terrain feature'})[0],nested)
        def test_schema_blocks_semantics(self):
            row=[[['Person','director','A Title']]]
            # An endpoint swap is still allowed for an existing in-schema label.
            self.assertEqual(propose(row,['Person directed A Title.'],set())[0],row)
            row=[[['A Title','producer','Person']]]
            self.assertEqual(propose(row,['A Title was directed by Person.'],{'producer'})[0],row)
        def test_no_false_cast_from_general_appearance(self):
            row=[[['Person','creator','Museum']]]
            self.assertEqual(propose(row,['Person appeared in Museum.'],{'creator','cast member'})[0],row)
            out,_=propose([[['New  York','country','United States']]],['New York is in the United States.'],set())
            self.assertEqual(out,[[['New York','country','United States']]])
        def test_relation_needs_both_schema_and_event(self):
            row=[[['Person','cause of death','March 27, 1973']]]
            self.assertEqual(propose(row,['Person died on March 27, 1973.'],{'date of death'})[0][0][0][1],'date of death')
            self.assertEqual(propose(row,['Person visited a city on March 27, 1973.'],{'date of death'})[0],row)
            self.assertEqual(propose(row,['Person died on March 27, 1973.'],set())[0],row)
        def test_no_event_transfer_from_relative(self):
            row=[[['Person','cause of death','March 27, 1973']]]
            self.assertEqual(propose(row,["Person's father died on March 27, 1973."],{'date of death'})[0],row)
        def test_date_form_preserved(self):
            row=[[['Person','date of birth','March 27, 1973']]]
            self.assertEqual(propose(row,['Person was born on March 27, 1973.'],{'date of birth'})[0],row)
        def test_dedupe_and_determinism(self):
            rows=[[['A','r','B'],['A','r','B']]]
            a=propose(rows,['A r B.'],{'r'});b=propose(rows,['A r B.'],{'r'})
            self.assertEqual(a,b);self.assertEqual(len(a[0][0]),1)
        def test_fresh_pipeline_order_without_models(self):
            # Exercise real command construction with local fake subprocesses.
            # This tests stage wiring, not transformer inference or its quality.
            import tempfile
            from types import SimpleNamespace
            from unittest.mock import patch
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);d=root/'job';d.mkdir();model=root/'model';model.mkdir()
                texts=root/'texts.txt';texts.write_text('A was born in 2000.\n')
                ref=root/'reference.txt';write_rows(ref,[[['A','date of birth','2000']]])
                a4=d/'A4_upstream/iter0';a4.mkdir(parents=True)
                a2=d/'A2_proposed_local/iter0';a2.mkdir(parents=True)
                write_rows(a4/'canon_kg.txt',[[['A','place of birth','2000']]])
                write_rows(a2/'canon_kg.txt',[[['A','date of birth','2000']]])
                cfg={'checkpoint':'model','policy':None,'threshold':.95,
                     'scoring_max_length':192,'context_window_chars':0}
                args=SimpleNamespace(model_dir=str(model),threshold_policy=None,
                                     batch_size=2,ablation_dir='unused')
                calls=[]
                def fake(command,cwd,env,log):
                    calls.append(command);log.write_text('offline wiring test\n')
                    def arg(k):return Path(command[command.index(k)+1])
                    if 'score_rebel_entailment.py' in command[1]:
                        rr=read_rows(arg('--raw'))
                        jsonl(arg('--output'),[{'line':i,'triple_index':j,'triple':t,'score':1.}
                                               for i,row in enumerate(rr,1) for j,t in enumerate(row)])
                    elif 'apply_rebel_entailment_scores.py' in command[1]:
                        p=arg('--output_dir')/'iter0';p.mkdir()
                        shutil.copyfile(arg('--raw'),p/'canon_kg.txt')
                    else:
                        mm=metric_report(read_rows(arg('--edc_output')),read_rows(arg('--reference')))
                        text='Total scores\n'
                        for title in ['Ent_type','Partial','Strict','Exact']:
                            text+=title+'\nPrecision: 1.0 Recall: 1.0\nF1: 1.0\n'
                        text+='Scores per tag\nFull triple scores\n'
                        text+=f"Precision: {mm['full_p']} Recall: {mm['full_r']}\nF1: {mm['full_f1']}\n"
                        arg('--evaluation_output').write_text(text)
                with patch.dict(globals(),{'run_command':fake}):
                    fresh_four_way(root,d,cfg,{'texts':texts,'reference':ref},args)
                scoring=[c for c in calls if 'score_rebel_entailment.py' in c[1]]
                filtering=[c for c in calls if 'apply_rebel_entailment_scores.py' in c[1]]
                self.assertEqual(len(scoring),2)
                self.assertEqual(scoring[0][scoring[0].index('--raw')+1],str(a2/'canon_kg.txt'))
                self.assertEqual(scoring[1][scoring[1].index('--raw')+1],str(a4/'canon_kg.txt'))
                self.assertTrue(all('--profile' not in c for c in filtering))
                self.assertEqual(read_rows(d/'A1_proposed_full/iter0/canon_kg.txt'),read_rows(a2/'canon_kg.txt'))
                audit=json.loads((d/'correction_verifier_summary.json').read_text(encoding="utf-8"))
                self.assertEqual(audit['pre_verifier_new_reference_matches'],1)
                self.assertEqual(audit['new_reference_matches_retained'],1)
                self.assertEqual(audit['new_reference_matches_rejected'],0)
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Guards)
    if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():raise SystemExit(1)
    class ExpandedGuards(unittest.TestCase):
        def run_one(self, triple, text, schema):
            return propose([[triple]], [text], set(schema))[0][0][0]
        def test_explicit_acronym_wrapper(self):
            t=["International Bird Society (IBS)","publisher","Journal"]
            self.assertEqual(self.run_one(t,"International Bird Society (IBS) publishes Journal.",[])[0],"International Bird Society")
        def test_qualifier_is_not_an_acronym(self):
            t=["Oxford (Mississippi)","country","United States"]
            self.assertEqual(self.run_one(t,"Oxford (Mississippi) is in the United States.",[]),t)
        def test_acronym_expansion_for_object(self):
            t=["Journal","publisher","IBS"]
            self.assertEqual(self.run_one(t,"International Bird Society (IBS) publishes Journal.",[])[2],"International Bird Society")
        def test_acronym_collision_preserves(self):
            t=["IBS","country","France"]
            text="International Bird Society (IBS) and Island Botany Society (IBS) met in France."
            self.assertEqual(self.run_one(t,text,[]),t)
        def test_explicit_alias(self):
            t=["Smith","employer","College"]
            self.assertEqual(self.run_one(t,"Alex Smith (also known as Smith) works at College.",[])[0],"Alex Smith")
        def test_later_surname_after_full_introduction(self):
            t=["Smith","educated at","College"]
            self.assertEqual(self.run_one(t,"Alex Smith was a physicist. Smith attended College.",[])[0],"Alex Smith")
        def test_surname_before_full_introduction_preserves(self):
            t=["Smith","educated at","College"]
            self.assertEqual(self.run_one(t,"Smith attended College and met Alex Smith.",[]),t)
        def test_competing_names_and_kinship_preserve(self):
            t=["Smith","educated at","College"]
            for text in ["Alex Smith was a physicist. Bob Smith met him. Smith attended College.","Alex Smith was a physicist. His son Smith attended College."]:
                self.assertEqual(self.run_one(t,text,[]),t)
        def test_no_external_alias_mapping(self):
            t=["UK","country","Britain"]
            self.assertEqual(self.run_one(t,"UK is in Britain.",[]),t)
        def test_typed_work_description_and_present_tense(self):
            t=["Film X","producer","Alex Smith"]
            self.assertEqual(self.run_one(t,"Film X is a 2013 Hindi comedy film directed by Alex Smith.",["director","producer"])[1],"director")
            self.assertEqual(self.run_one(t,"Alex Smith directs Film X.",["director","producer"])[1],"director")
        def test_preserve_multiple_supported_roles(self):
            text="Film X is a 2013 Hindi comedy film written and directed by Alex Smith."
            for r in ["director","screenwriter"]:
                t=["Film X",r,"Alex Smith"]
                self.assertEqual(self.run_one(t,text,["director","screenwriter","producer"]),t)
            t=["Film X","producer","Alex Smith"]
            self.assertEqual(self.run_one(t,text,["director","screenwriter","producer"]),t)
        def test_description_does_not_transfer_other_agents(self):
            t=["Film X","producer","Alex Smith"]
            for text in ["Film X starring Bob Jones was directed by Alex Smith.","Film X is a movie whose producer said it was directed by Alex Smith.","Film X is a film reportedly directed by Alex Smith."]:
                self.assertEqual(self.run_one(t,text,["director","producer"]),t)
        def test_director_and_cast_have_separate_agent_scopes(self):
            text="Film X is a 1934 German film directed by Arthur Robison and starring Albrecht Schoenhals, Hansi Knoteck and Willy Birgel."
            for person in ["Albrecht Schoenhals","Hansi Knoteck","Willy Birgel"]:
                t=["Film X","cast member",person]
                self.assertEqual(self.run_one(t,text,["director","cast member","producer"]),t)
                self.assertEqual(expanded_role_evidence("Film X","director",person,text),[])
        def test_unparsed_existing_role_abstains(self):
            cases = [
                (["Film X","cast member","Pauline Bush"],"Film X is a silent drama film directed by Joe Smith and featuring Pauline Bush."),
                (["Film X","screenwriter","Will Smith"],"Film X is a British film directed by Alex Smith and based on a screenplay from Alex and Will Smith."),
                (["Film X","creator","David Smith"],"Film X is an American television series created and primarily written by former reporter David Smith."),
                (["Film X","screenwriter","Alex Smith"],"Film X is a film written, co - scored and directed by Alex Smith."),
                (["Film X","director","Alex Smith"],"Film X is a film directed, co - written and co - produced by Alex Smith."),
                (["Film X","cast member","Alex Smith"],"Film X is a film starring and directed by Alex Smith."),
            ]
            for t,text in cases:
                self.assertEqual(self.run_one(t,text,["director","producer","screenwriter","creator","cast member"]),t)
        def test_athlete_relative_clause_and_list(self):
            text="Alex Smith is a retired footballer who played during his career as forward for Red Stars, Blue Stars and Green Stars."
            for team in ["Red Stars","Blue Stars","Green Stars"]:
                t=["Alex Smith","member of",team]
                self.assertEqual(self.run_one(t,text,["member of","member of sports team"])[1],"member of sports team")
        def test_leading_biography_pronoun_with_seasons(self):
            t=["Alex Smith","member of","Red Stars"]
            text="Alex Smith (born 12 May 1970) is a retired professional football player. He played for 7 seasons (168 games, 2 goals) in the Premier League for Red Stars."
            self.assertEqual(self.run_one(t,text,["member of","member of sports team"])[1],"member of sports team")
        def test_musician_and_coach_not_players(self):
            t=["Alex Smith","member of","Red Stars"]
            for text in ["Alex Smith is a musician who played for Red Stars.","Alex Smith is a footballer who coached Red Stars.","Alex Smith is a footballer. His brother played for Red Stars.","Alex Smith is a footballer who never played for Red Stars.","Alex Smith is a footballer who claimed he played for Red Stars."]:
                self.assertEqual(self.run_one(t,text,["member of","member of sports team"]),t)
        def test_competing_clause_and_schema(self):
            t=["Alex Smith","member of","Red Stars"]
            self.assertEqual(self.run_one(t,"Alex Smith is a footballer who played for Blue Stars against Red Stars.",["member of","member of sports team"]),t)
            self.assertEqual(self.run_one(t,"Alex Smith is a footballer who played for Red Stars.",["member of"]),t)
        def test_no_owner_or_geographic_rewrite(self):
            t=["Plant","operator","Company"]
            self.assertEqual(self.run_one(t,"Plant is operated by Company.",["operator","owned by"]),t)
            t=["Lake","located in the administrative territorial entity","Town"]
            self.assertEqual(self.run_one(t,"Lake is in Province and located in Town.",[t[1],"part of"]),t)
        def test_features_disabled_equal_legacy(self):
            from unittest.mock import patch
            t=["IBS","publisher","Journal"]
            with patch.dict(FEATURES,entity_representation=False,expanded_relation_patterns=False):
                self.assertEqual(self.run_one(t,"International Bird Society (IBS) publishes Journal.",[]),t)
        def test_reference_free_signature_and_determinism(self):
            import inspect
            self.assertEqual(list(inspect.signature(propose).parameters),["rows","texts","relations"])
            row=[[["IBS","publisher","Journal"]]];text=["International Bird Society (IBS) publishes Journal."]
            self.assertEqual(propose(row,text,set()),propose(row,text,set()))
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(ExpandedGuards)
    if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():raise SystemExit(1)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root",type=Path,default=Path.cwd())
    ap.add_argument("--dataset",choices=["both","rebel","wiki-nre"],default="both")
    ap.add_argument("--ablation-dir",default="output/cross_dataset_ablation_20261016")
    ap.add_argument("--out",type=Path)
    ap.add_argument("--raw");ap.add_argument("--texts");ap.add_argument("--reference");ap.add_argument("--schema")
    ap.add_argument("--without-reference",action="store_true")
    ap.add_argument("--run-verifier",action="store_true",help="Freshly score raw AND proposed candidates, then evaluate four configurations")
    ap.add_argument("--model-dir");ap.add_argument("--threshold-policy")
    ap.add_argument("--batch-size",type=int,default=32)
    ap.add_argument("--self-test",action="store_true")
    ap.add_argument("--disable-entity-correction",action="store_true")
    ap.add_argument("--disable-expanded-relations",action="store_true")
    args=ap.parse_args()
    if args.self_test:self_test();return
    FEATURES.update(entity_representation=not args.disable_entity_correction,
                    expanded_relation_patterns=not args.disable_expanded_relations)
    if args.out is None:ap.error("--out is required")
    if bool(args.raw)!=bool(args.texts):ap.error("--raw and --texts must be supplied together")
    if args.without_reference and args.reference:ap.error("--reference conflicts with --without-reference")
    if args.dataset=="both" and any([args.raw,args.texts,args.reference,args.schema,args.model_dir,args.threshold_policy]):
        ap.error("Custom input/model/policy overrides require one selected dataset")
    if args.batch_size<=0:ap.error("--batch-size must be positive")
    root=args.root.resolve();base=resolve(root,args.ablation_dir);out=resolve(root,args.out)
    if out.exists():raise FileExistsError(f"Refusing to overwrite: {out}")
    manifest=json.loads((base/"manifest.json").read_text(encoding="utf-8"))
    names=["REBEL","Wiki-NRE"] if args.dataset=="both" else ["REBEL" if args.dataset=="rebel" else "Wiki-NRE"]
    # Preflight all input/schema files before creating output or loading models.
    loaded={n:load_inputs(args,n,root,base,manifest) for n in names}
    if args.run_verifier:
        for required in ["scripts/rebel/score_rebel_entailment.py","scripts/rebel/apply_rebel_entailment_scores.py","evaluate/evaluation_script1.py"]:
            if not (root/required).is_file():raise FileNotFoundError(root/required)
        for name,(cfg,paths,raw,texts,refs,relations) in loaded.items():
            model=resolve(root,args.model_dir or cfg["checkpoint"])
            if not (model/"config.json").is_file() or not any(model.glob("*.safetensors")) and not any(model.glob("pytorch_model*.bin")):
                raise FileNotFoundError(f"Verifier config/weights incomplete: {model}")
            if cfg["policy"]:
                policy=resolve(root,args.threshold_policy) if args.threshold_policy else base/name/"threshold_policy.json"
                if not policy.is_file():raise FileNotFoundError(policy)
    out.mkdir(parents=True)
    summary={"version":VERSION,"script_sha256":sha(Path(__file__)),
             "stage_order":"reference-free proposed local correction -> fresh verifier -> evaluation",
             "experimental_preview":True,"run_verifier":args.run_verifier,"rule_order":RULE_ORDER,
             "disabled_repair_rules":DISABLED_REPAIR_RULES,"features":dict(FEATURES),
             "entity_policy":"source-defined full names; unique leading person surname expansion only", "datasets":{}}
    dump(out/"frozen_run_configuration.json",{k:v for k,v in summary.items() if k!="datasets"})
    for name in names:
        cfg,paths,raw,texts,refs,relations=loaded[name];cfg=dict(cfg,dataset_name=name)
        d=out/name;d.mkdir()
        # Rules receive no references. Freeze candidates before diagnostics.
        corrected,edits=propose(raw,texts,relations)
        blocked=guard_diagnostics(raw,texts,relations)
        jsonl(d/"blocked_proposals.jsonl",blocked)
        for branch,rows in [("A4_upstream",raw),("A2_proposed_local",corrected)]:
            p=d/branch/"iter0";p.mkdir(parents=True);write_rows(p/"canon_kg.txt",rows)
        frozen=sha(d/"A2_proposed_local/iter0/canon_kg.txt")
        before=[{label(t) for t in r} for r in raw];after=[{label(t) for t in r} for r in corrected]
        categories=Counter();diagnostics=[];confusions=Counter()
        if refs is not None:
            for i,(row,gold,text) in enumerate(zip(raw,refs,texts),1):
                gs={label(t) for t in gold};seen=set()
                for j,t in enumerate(row):
                    if label(t) in gs or label(t) in seen:continue
                    seen.add(label(t));cat=error_category(t,gold);categories[cat]+=1
                    diagnostics.append({"line":i,"triple_index":j,"category":cat,"triple":t,
                                        "source_text":text,"reference":gold,"diagnostic_only":True})
                    if cat=="relation_label_difference":
                        for g in gold:
                            if t[0].lower()==g[0].lower() and t[2].lower()==g[2].lower():
                                confusions[(t[1],g[1])]+=1
            for e in edits:
                gs={label(t) for t in refs[e["line"]-1]}
                e["old_reference_match"]=label(e["old"]) in gs
                e["new_reference_match"]=(label(e["new"]) in gs) if e["new"] else None
                e["reference_labels_are_diagnostic_only"]=True
        if frozen!=sha(d/"A2_proposed_local/iter0/canon_kg.txt"):
            raise AssertionError("Reference diagnostics mutated correction output")
        jsonl(d/"proposed_edits.jsonl",edits);jsonl(d/"candidate_diagnostics.jsonl",diagnostics)
        dump(d/"relation_confusions_diagnostic_only.json",[{"from":a,"reference_relation":b,"count":v} for (a,b),v in confusions.most_common()])
        counts=Counter(e["rule"] for e in edits)
        details={"rows":len(raw),"raw_candidates":sum(map(len,raw)),"preview_candidates":sum(map(len,corrected)),
                 "schema_available":bool(relations),"relation_count":len(relations),"rule_counts":dict(counts),
                 "blocked_rule_counts":dict(Counter(e["reason"] for e in blocked)),
                 "diagnostic_categories":dict(categories),"gold_used_for_corrections":False,
                 "input_files":{k:{"path":str(v),"sha256":sha(v)} for k,v in paths.items() if v is not None and v.is_file()},
                 "preview_sha256_frozen_before_reference_diagnostics":frozen,
                 "policy_note":"No pronoun hard-filter, inverse hard-filter, relation sorting, or per-row cap in this experimental correction preview."}
        if refs is not None:
            details["raw_metrics"]=metric_report(raw,refs);details["preview_metrics"]=metric_report(corrected,refs)
            details["unique_reference_matches_added"]=sum(len((a-b)&{label(t) for t in g}) for a,b,g in zip(after,before,refs))
            details["unique_reference_matches_lost"]=sum(len((b-a)&{label(t) for t in g}) for a,b,g in zip(after,before,refs))
        if args.run_verifier:
            details["fresh_four_way"]=fresh_four_way(root,d,cfg,paths,args)
            details["correction_verifier_summary"]=json.loads((d/"correction_verifier_summary.json").read_text(encoding="utf-8"))
            if refs is not None and details["correction_verifier_summary"]["pre_verifier_new_reference_matches"]!=details["unique_reference_matches_added"]:
                raise AssertionError("Correction gain audit disagrees with pre-verifier set counts")
        summary["datasets"][name]=details;dump(d/"analysis.json",details)
        print(name,"rows",len(raw),"proposed edits",dict(counts),"schema available",bool(relations),flush=True)
    dump(out/"summary.json",summary)
    if args.run_verifier:
        fields=["dataset","configuration","candidates","full_p","full_r","full_f1","delta_full_f1",
                "ent_type_f1","partial_f1","strict_f1","exact_f1"]
        with (out/"four_way_results.csv").open("w",newline="",encoding="utf-8-sig") as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
            for name,details in summary["datasets"].items():
                records=details["fresh_four_way"]
                if not all("evaluation_metrics" in r for r in records):continue
                anchor=records[0]["evaluation_metrics"]["full_f1"]
                for record in records:
                    metrics=record["evaluation_metrics"]
                    values={key:metrics.get(key) for key in fields}
                    values.update(dataset=name,configuration=record["configuration"],candidates=record["candidates"],
                                  delta_full_f1=metrics["full_f1"]-anchor)
                    writer.writerow(values)
    report=["# 数据集特定纠正分析 v3（实体表示与显式句式扩展）", "",
            "在 v2.1 基础上增加源文本明确给出的缩写/别名、受限的后续姓氏回指，以及运动员履历和作品角色句式。explicit_part_of 仍关闭；不改写运营与所有权或地理层级。", "",
            "本脚本借鉴 WebNLG 的受限规则纠正思路，但不把其下划线或 ISO 日期输出格式直接移植到其他数据集。外层引号及异常空白仅在源文本支持下清理，不根据参考答案注入不换行空格。", "",
            "固定顺序：表面修复 → 有限实体表示纠正 → 显式关系/方向纠正 → 模式允许的日期关系纠正 → 精确去重 → 可选的全新 Verifier 评分 → 评估。不存在验证后的后处理。", "",
            "参考答案只用于问题分类、匹配诊断和评估；不会参与生成修复后的候选。默认不运行模型。", "",
            "| 数据集 | 候选数 | 预览候选数 | 表面修复 | 实体表示纠正 | 有限关系/方向纠正 | 日期关系纠正 | 去重 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for n,v in summary["datasets"].items():
        c=v["rule_counts"]
        entity_count=sum(c.get(rule,0) for rule in ENTITY_RULES)
        semantic_count=sum(num for rule,num in c.items() if rule not in ENTITY_RULES | {"source_attested_surface_v2","explicit_event_date_relation","exact_deduplication"})
        report.append(f"| {n} | {v['raw_candidates']} | {v['preview_candidates']} | {c.get('source_attested_surface_v2',0)} | {entity_count} | {semantic_count} | {c.get('explicit_event_date_relation',0)} | {c.get('exact_deduplication',0)} |")
    report += ["", "查看 proposed_edits.jsonl 中每条修复的 old/new/evidence，再看 candidate_diagnostics.jsonl 的问题分布。blocked_proposals.jsonl 记录因目标不在模式中或存在多个可选关系而保持原样的候选。relation_confusions_diagnostic_only.json 只是参考对照，不能直接转换成无条件替换规则。", "",
               "方向纠正仅适用于明确作品—作者/导演等非对称角色关系；不会交换 spouse 等对称关系。若源文本已支持当前关系，则保留它，即使文本还支持其他关系。member of 只在明确团队/政党词汇下细化；不会把无类型证据的普通组织归为团队或政党。", "",
               "若 schema_available=false，脚本没有找到原运行目标模式，关系纠正会停用；不会用参考答案推断关系集合。正式对照需提供正确 --schema，开启 --run-verifier 时缺少模式会直接报错。", "",
               "这是待核查的实验性修复方案，不保证提高分数。已有历史评估范围只用于诊断；规则定稿和选择应使用训练/开发候选，并冻结后再报告独立评估结果。默认输入是已完成上游关系规范化的候选，不是最初 OIE 回复。", "",
               "开启 --run-verifier 会对 A4 原候选与 A2 修复后候选分别重新评分，不读取历史分数。A1 只过滤 A2，A3 只过滤 A4，过滤命令没有 --profile。"]
    if args.run_verifier:
        report += ["", "## 四组对照与修复保留", "",
                   "完整精度见 four_way_results.csv。新增匹配的验证结果见各数据集 correction_verifier_summary.json 与 correction_verifier_audit.jsonl。", "",
                   "| 数据集 | 后处理新增参考匹配 | 经 Verifier 保留 | 经 Verifier 删除 | A1相对A3新增匹配 | A1相对A3损失匹配 |", "|---|---:|---:|---:|---:|---:|"]
        for name,details in summary["datasets"].items():
            stats=details["correction_verifier_summary"]
            report.append(f"| {name} | {stats['pre_verifier_new_reference_matches']} | {stats['new_reference_matches_retained']} | {stats['new_reference_matches_rejected']} | {stats['A1_vs_A3_unique_reference_matches_gained']} | {stats['A1_vs_A3_unique_reference_matches_lost']} |")
    (out/"REPORT.zh-CN.md").write_text("\n".join(report)+"\n",encoding="utf-8")
    print("COMPLETE",out,flush=True)


if __name__=="__main__":
    main()
