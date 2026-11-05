#!/usr/bin/env python3
"""Build a deterministic sentence-level REBEL validation or test protocol."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import nltk.data


Triple = List[str]


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def stable_priority(seed: int, docid: str, sentence_id: int) -> int:
    value = f"{seed}\0{docid}\0{sentence_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(value).digest(), "big")


def load_schema_mapping(path: Path) -> Dict[str, str]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return {item["uri"]: item["label"] for item in payload["relations"]}


def normalize_sentence(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def mention_within_span(mention: Dict, start: int, end: int) -> bool:
    boundaries = mention.get("boundaries")
    return bool(
        isinstance(boundaries, list)
        and len(boundaries) == 2
        and isinstance(boundaries[0], int)
        and isinstance(boundaries[1], int)
        and start <= boundaries[0] <= boundaries[1] <= end
    )


def mention_boundaries(mention: Dict) -> Tuple[int, int] | None:
    boundaries = mention.get("boundaries")
    if not (
        isinstance(boundaries, list)
        and len(boundaries) == 2
        and isinstance(boundaries[0], int)
        and isinstance(boundaries[1], int)
        and 0 <= boundaries[0] <= boundaries[1]
    ):
        return None
    return boundaries[0], boundaries[1]


def resolve_sentence_span(
    sentence_id: int,
    sentence_triples: List[Dict],
    spans: List[Tuple[int, int]],
    counters: Counter,
) -> Tuple[int, int] | None:
    mention_spans = []
    for triple in sentence_triples:
        subject = triple.get("subject")
        obj = triple.get("object")
        if not isinstance(subject, dict) or not isinstance(obj, dict):
            counters["invalid_triple_parts"] += 1
            continue
        for mention in (subject, obj):
            boundaries = mention_boundaries(mention)
            if boundaries is None:
                counters["invalid_mention_boundaries"] += 1
                continue
            mention_spans.append(boundaries)

    if not mention_spans:
        counters["unresolved_sentence_spans"] += 1
        return None

    overlapping_indices = set()
    for mention_start, mention_end in mention_spans:
        indices = [
            index
            for index, (span_start, span_end) in enumerate(spans)
            if span_start < mention_end and mention_start < span_end
        ]
        if not indices:
            counters["mentions_outside_nltk_spans"] += 1
            return None
        overlapping_indices.update(indices)

    first_index = min(overlapping_indices)
    last_index = max(overlapping_indices)
    if first_index != sentence_id or last_index != sentence_id:
        counters["sentence_id_index_disagreement"] += 1
    if last_index > first_index:
        counters["merged_nltk_spans"] += 1

    start = spans[first_index][0]
    end = spans[last_index][1]
    if any(not (start <= mention_start <= mention_end <= end) for mention_start, mention_end in mention_spans):
        counters["unresolved_sentence_spans"] += 1
        return None
    return start, end


def candidate_sentences(
    record: Dict,
    line_number: int,
    split: str,
    tokenizer,
    schema_mapping: Dict[str, str],
    min_triples: int,
    max_triples: int,
    counters: Counter,
) -> List[Dict]:
    text = str(record.get("text", ""))
    docid = str(record.get("docid", ""))
    triples = record.get("triples", [])
    if not text or not docid or not isinstance(triples, list):
        counters["invalid_records"] += 1
        return []

    spans = list(tokenizer.span_tokenize(text))
    if not spans:
        counters["records_without_sentence_spans"] += 1
        return []

    grouped: Dict[int, List[Dict]] = defaultdict(list)
    for triple in triples:
        if not isinstance(triple, dict):
            counters["invalid_triples"] += 1
            continue
        sentence_id = triple.get("sentence_id")
        if not isinstance(sentence_id, int):
            counters["triples_without_integer_sentence_id"] += 1
            continue
        grouped[sentence_id].append(triple)

    resolved_groups: Dict[Tuple[int, int], Dict] = {}
    for sentence_id, sentence_triples in grouped.items():
        resolved_span = resolve_sentence_span(
            sentence_id=sentence_id,
            sentence_triples=sentence_triples,
            spans=spans,
            counters=counters,
        )
        if resolved_span is None:
            continue
        group = resolved_groups.setdefault(
            resolved_span,
            {"sentence_ids": [], "triples": []},
        )
        group["sentence_ids"].append(sentence_id)
        group["triples"].extend(sentence_triples)

    candidates = []
    for (start, end), resolved_group in resolved_groups.items():
        sentence_ids = sorted(set(resolved_group["sentence_ids"]))
        sentence_triples = resolved_group["triples"]
        triple_records = []
        invalid_boundary = False

        for triple in sentence_triples:
            subject = triple.get("subject")
            predicate = triple.get("predicate")
            obj = triple.get("object")
            if not all(isinstance(part, dict) for part in (subject, predicate, obj)):
                counters["invalid_triple_parts"] += 1
                continue
            if not mention_within_span(subject, start, end) or not mention_within_span(
                obj, start, end
            ):
                invalid_boundary = True
                counters["triple_sentence_boundary_mismatch"] += 1
                continue

            relation_uri = str(predicate.get("uri", "")).strip()
            original_relation = str(predicate.get("surfaceform", "")).strip()
            canonical_relation = schema_mapping.get(relation_uri, original_relation)
            subject_surface = str(subject.get("surfaceform", "")).strip()
            object_surface = str(obj.get("surfaceform", "")).strip()
            if not subject_surface or not canonical_relation or not object_surface:
                counters["empty_triple_surfaces"] += 1
                continue

            triple_records.append(
                {
                    "triple": [subject_surface, canonical_relation, object_surface],
                    "relation_uri": relation_uri,
                    "original_relation": original_relation,
                }
            )

        deduped = []
        relation_uris = []
        original_relations = []
        seen = set()
        for record_item in triple_records:
            triple = record_item["triple"]
            key = tuple(triple)
            if key not in seen:
                seen.add(key)
                deduped.append(triple)
                relation_uris.append(record_item["relation_uri"])
                original_relations.append(record_item["original_relation"])
            else:
                counters["duplicate_gold_triples_removed"] += 1

        if invalid_boundary:
            counters["sentences_with_boundary_mismatch"] += 1
            continue
        if not (min_triples <= len(deduped) <= max_triples):
            counters["sentences_outside_triple_limits"] += 1
            continue

        sentence = normalize_sentence(text[start:end])
        if not sentence:
            counters["empty_sentences"] += 1
            continue

        in_schema = [uri in schema_mapping for uri in relation_uris]
        candidates.append(
            {
                "source_split": split,
                "source_line": line_number,
                "docid": docid,
                "sentence_id": sentence_ids[0],
                "source_sentence_ids": sentence_ids,
                "sentence_span": [start, end],
                "text": sentence,
                "triples": deduped,
                "relation_uris": relation_uris,
                "original_relation_surfaces": original_relations,
                "in_schema_triples": sum(in_schema),
                "outside_schema_triples": len(in_schema) - sum(in_schema),
            }
        )
    return candidates


def ensure_outputs_available(paths: List[Path], force: bool) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing and not force:
        raise FileExistsError(
            "Refusing to overwrite existing outputs; use --force: " + ", ".join(existing)
        )
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw_file", required=True, type=Path)
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    parser.add_argument("--schema_mapping", required=True, type=Path)
    parser.add_argument("--size", required=True, type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_triples", type=int, default=1)
    parser.add_argument("--max_triples", type=int, default=7)
    parser.add_argument("--max_records", type=int)
    parser.add_argument("--progress_every", type=int, default=25000)
    parser.add_argument("--nltk_data", type=Path)
    parser.add_argument("--dataset_output", required=True, type=Path)
    parser.add_argument("--reference_output", required=True, type=Path)
    parser.add_argument("--manifest_output", required=True, type=Path)
    parser.add_argument("--summary_output", required=True, type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.size <= 0 or args.min_triples <= 0 or args.max_triples < args.min_triples:
        raise SystemExit("Invalid size or triple limits")
    if args.nltk_data:
        nltk.data.path.insert(0, str(args.nltk_data))
    try:
        tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
    except LookupError as exc:
        raise SystemExit(
            "NLTK English punkt tokenizer is required; set --nltk_data to its parent directory"
        ) from exc

    outputs = [
        args.dataset_output,
        args.reference_output,
        args.manifest_output,
        args.summary_output,
    ]
    ensure_outputs_available(outputs, args.force)
    schema_mapping = load_schema_mapping(args.schema_mapping)
    counters = Counter()
    selected_heap: List[Tuple[int, str, Dict]] = []
    seen_docids = set()
    seen_text_hashes = set()

    with args.raw_file.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            if args.max_records is not None and counters["records_scanned"] >= args.max_records:
                break
            counters["records_scanned"] += 1
            if args.progress_every and counters["records_scanned"] % args.progress_every == 0:
                print(f"scanned {counters['records_scanned']:,} records", flush=True)
            try:
                record = json.loads(line)
            except Exception:
                counters["invalid_json"] += 1
                continue

            docid = str(record.get("docid", ""))
            if docid in seen_docids:
                counters["duplicate_docids"] += 1
                continue
            seen_docids.add(docid)
            candidates = candidate_sentences(
                record=record,
                line_number=line_number,
                split=args.split,
                tokenizer=tokenizer,
                schema_mapping=schema_mapping,
                min_triples=args.min_triples,
                max_triples=args.max_triples,
                counters=counters,
            )
            if not candidates:
                continue

            candidate = min(
                candidates,
                key=lambda item: stable_priority(
                    args.seed, item["docid"], item["sentence_id"]
                ),
            )
            counters["documents_with_candidates"] += 1
            text_hash = sha256_text(candidate["text"])
            if text_hash in seen_text_hashes:
                counters["duplicate_sentence_texts"] += 1
                continue
            seen_text_hashes.add(text_hash)
            priority = stable_priority(args.seed, candidate["docid"], candidate["sentence_id"])
            tie_breaker = f"{candidate['docid']}:{candidate['sentence_id']}"
            candidate["selection_priority_sha256"] = f"{priority:064x}"
            candidate["text_sha256"] = text_hash
            candidate["triple_sha256"] = sha256_text(
                json.dumps(candidate["triples"], ensure_ascii=False, separators=(",", ":"))
            )

            entry = (-priority, tie_breaker, candidate)
            if len(selected_heap) < args.size:
                heapq.heappush(selected_heap, entry)
            elif priority < -selected_heap[0][0]:
                heapq.heapreplace(selected_heap, entry)

    if len(selected_heap) < args.size:
        raise RuntimeError(
            f"Only {len(selected_heap)} eligible documents found for requested size {args.size}"
        )

    selected = [entry[2] for entry in selected_heap]
    selected.sort(key=lambda item: item["selection_priority_sha256"])
    selected_relation_counts = Counter()
    selected_in_schema = 0
    selected_outside_schema = 0
    selected_triples = 0

    with args.dataset_output.open("w", encoding="utf-8") as dataset_file, \
         args.reference_output.open("w", encoding="utf-8") as reference_file, \
         args.manifest_output.open("w", encoding="utf-8") as manifest_file:
        for index, item in enumerate(selected):
            dataset_file.write(item["text"] + "\n")
            reference_file.write(repr(item["triples"]) + "\n")
            manifest_record = {key: value for key, value in item.items() if key != "text"}
            manifest_record["index"] = index
            manifest_file.write(json.dumps(manifest_record, ensure_ascii=False) + "\n")
            selected_triples += len(item["triples"])
            selected_in_schema += item["in_schema_triples"]
            selected_outside_schema += item["outside_schema_triples"]
            selected_relation_counts.update(item["relation_uris"])

    summary = {
        "protocol": f"rebel_official_{args.split}_{args.size}",
        "source": {
            "raw_file": str(args.raw_file),
            "split": args.split,
            "file_size": args.raw_file.stat().st_size,
            "max_records": args.max_records,
        },
        "selection": {
            "method": "smallest_sha256_priority_one_sentence_per_doc",
            "seed": args.seed,
            "size": args.size,
            "min_triples": args.min_triples,
            "max_triples": args.max_triples,
            "schema_mapping": str(args.schema_mapping),
        },
        "scan_counts": dict(sorted(counters.items())),
        "selected": {
            "records": len(selected),
            "triples": selected_triples,
            "unique_relation_uris": len(selected_relation_counts),
            "in_schema_triples": selected_in_schema,
            "outside_schema_triples": selected_outside_schema,
            "schema_coverage": selected_in_schema / selected_triples
            if selected_triples
            else 0.0,
        },
        "outputs": {
            "dataset": str(args.dataset_output),
            "reference": str(args.reference_output),
            "manifest": str(args.manifest_output),
        },
    }
    with args.summary_output.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(
        f"selected={len(selected):,}, triples={selected_triples:,}, "
        f"schema_coverage={summary['selected']['schema_coverage']:.4%}"
    )
    print(f"Summary written to {args.summary_output}")


if __name__ == "__main__":
    main()
