"""Input formatting helpers shared by REBEL verifier training and scoring."""

from __future__ import annotations

import re


def entity_centered_context(context: str, subject: str, obj: str, window_chars: int) -> str:
    text = str(context or "")
    if window_chars <= 0 or not text:
        return text
    positions = []
    for value in (subject, obj):
        value = str(value or "").strip()
        if not value:
            continue
        match = re.search(re.escape(value), text, flags=re.IGNORECASE)
        if match:
            positions.extend((match.start(), match.end()))
    if not positions:
        return text[:window_chars]
    start = max(0, min(positions) - window_chars // 3)
    end = min(len(text), max(positions) + window_chars * 2 // 3)
    if end - start > window_chars:
        end = start + window_chars
    return text[start:end].strip()


def format_verifier_input(
    context: str,
    subject: str,
    relation: str,
    obj: str,
    structured: bool = False,
    context_window_chars: int = 0,
):
    context = entity_centered_context(context, subject, obj, context_window_chars)
    if structured:
        triple = (
            f"[SUBJECT] {subject} [RELATION] {relation} "
            f"[OBJECT] {obj}"
        )
    else:
        triple = f"{subject} [SEP] {relation} [SEP] {obj}"
    return context, triple
