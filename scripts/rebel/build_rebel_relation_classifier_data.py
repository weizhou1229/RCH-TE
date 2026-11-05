#!/usr/bin/env python3
"""Build split-safe relation-classifier data from official REBEL train JSONL."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


Sample = Dict[str, object]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(*parts: object) -> int:
    payload = "\0".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest(), "big")


def assign_split(
    docid: str,
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> str:
    point = stable_hash(seed, docid) / float(1 << 256)
    if point < train_ratio:
        return "train"
    if point < train_ratio + val_ratio:
        return "val"
    return "test"


def normalize_space(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def load_schema_mapping(path: Path) -> Tuple[List[str], Dict[str, Dict]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    relations = sorted(payload["relations"], key=lambda item: int(item["rank"]))
    labels = [normalize_space(item["label"]) for item in relations]
    uris = [normalize_space(item["uri"]) for item in relations]
    if not labels or len(labels) != len(set(labels)):
        raise ValueError("Schema labels must be non-empty and unique")
    if len(uris) != len(set(uris)):
        raise ValueError("Schema relation URIs must be unique")
    return labels, {uri: item for uri, item in zip(uris, relations)}


def relation_variants(
    predicate_surface: str,
    schema_item: Dict,
    aliases_per_triple: int,
) -> List[Tuple[str, str]]:
    variants: List[Tuple[str, str]] = []
    seen = set()

    def add(value: object, source: str) -> None:
        cleaned = normalize_space(value)
        key = cleaned.casefold()
        if cleaned and key not in seen:
            seen.add(key)
            variants.append((cleaned, source))

    add(predicate_surface, "official_predicate_surface")
    add(schema_item["label"], "canonical_schema_label")

    alias_rows = schema_item.get("train_surface_variants", [])
    alias_count = 0
    for row in alias_rows:
        surface = row.get("surface") if isinstance(row, dict) else row
        before = len(variants)
        add(surface, "train_relation_alias")
        if len(variants) > before:
            alias_count += 1
        if alias_count >= aliases_per_triple:
            break

    for value, _source in list(variants):
        separated = normalize_space(value.replace("_", " ").replace("-", " "))
        add(separated, "separator_normalization")

    return variants


def keep_smallest(
    heap: List[Tuple[int, str, Sample]],
    sample: Sample,
    priority: int,
    capacity: int,
) -> None:
    tie_breaker = str(sample["sample_id"])
    entry = (-priority, tie_breaker, sample)
    if len(heap) < capacity:
        heapq.heappush(heap, entry)
    elif priority < -heap[0][0]:
        heapq.heapreplace(heap, entry)


def selected_samples(
    heaps: Dict[Tuple[str, str], List[Tuple[int, str, Sample]]],
    split: str,
    schema_labels: Iterable[str],
) -> List[Sample]:
    samples = []
    seen = set()
    for label in schema_labels:
        entries = sorted(
            heaps.get((split, label), []),
            key=lambda entry: (-entry[0], entry[1]),
        )
        for _negative_priority, _tie_breaker, sample in entries:
            key = (
                sample["subject"],
                sample["oie_relation"],
                sample["object"],
                sample["context"],
                sample["label"],
            )
            if key not in seen:
                seen.add(key)
                samples.append(sample)
    samples.sort(key=lambda sample: str(sample["sample_id"]))
    return samples


def ensure_outputs_available(paths: Iterable[Path], force: bool) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing and not force:
        raise FileExistsError(
            "Refusing to overwrite existing outputs; use --force: "
            + ", ".join(existing)
        )
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)


def build(args: argparse.Namespace) -> Dict:
    if not 0 < args.train_ratio < 1:
        raise ValueError("--train_ratio must be between 0 and 1")
    if not 0 < args.val_ratio < 1 - args.train_ratio:
        raise ValueError("--val_ratio must leave a positive test partition")
    if min(args.train_per_relation, args.val_per_relation, args.test_per_relation) <= 0:
        raise ValueError("Per-relation sample limits must be positive")

    output_paths = [
        args.output_dir / "train.json",
        args.output_dir / "val.json",
        args.output_dir / "test.json",
        args.output_dir / "schema_relations.json",
        args.output_dir / "stats.json",
    ]
    ensure_outputs_available(output_paths, args.force)
    schema_labels, schema_by_uri = load_schema_mapping(args.schema_mapping)
    capacities = {
        "train": args.train_per_relation,
        "val": args.val_per_relation,
        "test": args.test_per_relation,
    }
    heaps: Dict[Tuple[str, str], List[Tuple[int, str, Sample]]] = defaultdict(list)
    counters = Counter()
    candidate_counts = Counter()
    raw_digest = hashlib.sha256()

    with args.raw_file.open("rb") as source:
        for line_number, raw_line in enumerate(source, 1):
            if args.max_records is not None and counters["records_scanned"] >= args.max_records:
                break
            raw_digest.update(raw_line)
            counters["records_scanned"] += 1
            if args.progress_every and counters["records_scanned"] % args.progress_every == 0:
                print(f"scanned {counters['records_scanned']:,} records", flush=True)
            try:
                record = json.loads(raw_line)
            except Exception:
                counters["invalid_json"] += 1
                continue

            docid = normalize_space(record.get("docid"))
            text = normalize_space(record.get("text"))[: args.max_context_chars]
            triples = record.get("triples")
            if not docid or not text or not isinstance(triples, list):
                counters["invalid_records"] += 1
                continue

            split = assign_split(docid, args.seed, args.train_ratio, args.val_ratio)
            counters[f"documents_{split}"] += 1

            for triple_index, triple in enumerate(triples):
                if not isinstance(triple, dict):
                    counters["invalid_triples"] += 1
                    continue
                subject = triple.get("subject")
                predicate = triple.get("predicate")
                obj = triple.get("object")
                if not all(isinstance(part, dict) for part in (subject, predicate, obj)):
                    counters["invalid_triple_parts"] += 1
                    continue

                relation_uri = normalize_space(predicate.get("uri"))
                schema_item = schema_by_uri.get(relation_uri)
                if schema_item is None:
                    counters["triples_outside_schema"] += 1
                    continue
                subject_surface = normalize_space(subject.get("surfaceform"))
                object_surface = normalize_space(obj.get("surfaceform"))
                predicate_surface = normalize_space(predicate.get("surfaceform"))
                if not subject_surface or not object_surface:
                    counters["triples_missing_entity_surface"] += 1
                    continue

                label = normalize_space(schema_item["label"])
                variants = relation_variants(
                    predicate_surface=predicate_surface,
                    schema_item=schema_item,
                    aliases_per_triple=args.aliases_per_triple,
                )
                if not variants:
                    counters["triples_without_relation_variant"] += 1
                    continue
                counters["source_triples_in_schema"] += 1

                for variant_index, (variant, variant_source) in enumerate(variants):
                    sample_id = hashlib.sha256(
                        (
                            f"{args.seed}\0{docid}\0{line_number}\0{triple_index}"
                            f"\0{relation_uri}\0{variant_index}\0{variant}"
                        ).encode("utf-8")
                    ).hexdigest()
                    sample: Sample = {
                        "subject": subject_surface,
                        "oie_relation": variant,
                        "object": object_surface,
                        "context": text,
                        "label": label,
                        "sample_type": "positive_clean",
                        "variant_source": variant_source,
                        "source_split": "official_train",
                        "internal_split": split,
                        "docid": docid,
                        "source_line": line_number,
                        "triple_index": triple_index,
                        "relation_uri": relation_uri,
                        "official_predicate_surface": predicate_surface,
                        "sample_id": sample_id,
                    }
                    priority = stable_hash(args.seed, sample_id)
                    candidate_counts[(split, label)] += 1
                    keep_smallest(
                        heap=heaps[(split, label)],
                        sample=sample,
                        priority=priority,
                        capacity=capacities[split],
                    )

    splits = {
        split: selected_samples(heaps, split, schema_labels)
        for split in ("train", "val", "test")
    }
    split_docids = {
        split: {str(sample["docid"]) for sample in samples}
        for split, samples in splits.items()
    }
    if split_docids["train"] & split_docids["val"]:
        raise AssertionError("Document leakage between train and val")
    if split_docids["train"] & split_docids["test"]:
        raise AssertionError("Document leakage between train and test")
    if split_docids["val"] & split_docids["test"]:
        raise AssertionError("Document leakage between val and test")

    per_relation = {}
    sparse_relations = {}
    minimums = {
        "train": args.min_train_per_relation,
        "val": args.min_val_per_relation,
        "test": args.min_test_per_relation,
    }
    for label in schema_labels:
        counts = {
            split: sum(1 for sample in splits[split] if sample["label"] == label)
            for split in ("train", "val", "test")
        }
        per_relation[label] = counts
        missing = {
            split: count
            for split, count in counts.items()
            if count < minimums[split]
        }
        if missing:
            sparse_relations[label] = missing

    if sparse_relations and args.fail_on_sparse:
        preview = dict(list(sparse_relations.items())[:10])
        raise RuntimeError(f"Sparse schema relations detected: {preview}")

    for split, samples in splits.items():
        (args.output_dir / f"{split}.json").write_text(
            json.dumps(samples, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    (args.output_dir / "schema_relations.json").write_text(
        json.dumps(schema_labels, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    stats = {
        "protocol": "rebel_official_train_relation_classifier_v1",
        "source": {
            "raw_file": str(args.raw_file),
            "raw_sha256_scanned_prefix": raw_digest.hexdigest(),
            "scan_complete": args.max_records is None
            or counters["records_scanned"] < args.max_records,
            "max_records": args.max_records,
            "source_split": "official_train",
        },
        "schema": {
            "mapping": str(args.schema_mapping),
            "mapping_sha256": sha256_file(args.schema_mapping),
            "relation_count": len(schema_labels),
            "relations": schema_labels,
        },
        "selection": {
            "method": "smallest_sha256_priority_per_internal_split_and_relation",
            "seed": args.seed,
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "test_ratio": 1 - args.train_ratio - args.val_ratio,
            "per_relation_limits": capacities,
            "aliases_per_triple": args.aliases_per_triple,
            "max_context_chars": args.max_context_chars,
        },
        "scan_counts": dict(sorted(counters.items())),
        "selected": {
            split: {
                "samples": len(samples),
                "documents": len(split_docids[split]),
                "relations": len({str(sample["label"]) for sample in samples}),
            }
            for split, samples in splits.items()
        },
        "candidate_counts": {
            f"{split}:{label}": count
            for (split, label), count in sorted(candidate_counts.items())
        },
        "per_relation_selected": per_relation,
        "sparse_relations": sparse_relations,
        "leakage_checks": {
            "train_val_docid_overlap": 0,
            "train_test_docid_overlap": 0,
            "val_test_docid_overlap": 0,
        },
    }
    (args.output_dir / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw_file", required=True, type=Path)
    parser.add_argument("--schema_mapping", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--train_per_relation", type=int, default=500)
    parser.add_argument("--val_per_relation", type=int, default=100)
    parser.add_argument("--test_per_relation", type=int, default=100)
    parser.add_argument("--min_train_per_relation", type=int, default=50)
    parser.add_argument("--min_val_per_relation", type=int, default=10)
    parser.add_argument("--min_test_per_relation", type=int, default=10)
    parser.add_argument("--aliases_per_triple", type=int, default=2)
    parser.add_argument("--max_context_chars", type=int, default=600)
    parser.add_argument("--max_records", type=int)
    parser.add_argument("--progress_every", type=int, default=100000)
    parser.add_argument("--fail_on_sparse", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = build(args)
    print(
        "selected: "
        + ", ".join(
            f"{split}={stats['selected'][split]['samples']:,}"
            for split in ("train", "val", "test")
        )
    )
    print(f"sparse relations: {len(stats['sparse_relations'])}")
    print(f"Statistics written to {args.output_dir / 'stats.json'}")


if __name__ == "__main__":
    main()
