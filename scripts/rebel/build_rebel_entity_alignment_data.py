#!/usr/bin/env python3
"""Build URI-split entity-alignment data from official REBEL train JSONL."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple


Sample = Dict[str, object]
HeapEntry = Tuple[int, str, Tuple[str, str, str], Sample]


def stable_hash(*parts: object) -> int:
    payload = "\0".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest(), "big")


def assign_split(
    entity_uri: str,
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> str:
    point = stable_hash(seed, entity_uri) / float(1 << 256)
    if point < train_ratio:
        return "train"
    if point < train_ratio + val_ratio:
        return "val"
    return "test"


def normalize_space(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def entity_variants(surface: str) -> List[Tuple[str, str]]:
    variants: List[Tuple[str, str]] = []
    seen = set()

    def add(value: str, source: str) -> None:
        cleaned = normalize_space(value)
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            variants.append((cleaned, source))

    add(surface, "official_train_identity")
    if " " in surface:
        add(surface.replace(" ", "_"), "official_train_space_to_underscore")
    if "_" in surface:
        add(surface.replace("_", " "), "official_train_underscore_to_space")
    stripped = surface.strip('"\'')
    if stripped != surface:
        add(stripped, "official_train_outer_quote_removal")
    lower = surface.lower()
    if lower != surface and any(character.isalpha() for character in surface):
        add(lower, "official_train_case_variant")
    return variants


def keep_unique_smallest(
    heap: List[HeapEntry],
    keys: Set[Tuple[str, str, str]],
    sample: Sample,
    key: Tuple[str, str, str],
    priority: int,
    capacity: int,
) -> None:
    if key in keys:
        return
    entry: HeapEntry = (-priority, str(sample["sample_id"]), key, sample)
    if len(heap) < capacity:
        heapq.heappush(heap, entry)
        keys.add(key)
        return
    if priority < -heap[0][0]:
        removed = heapq.heapreplace(heap, entry)
        keys.remove(removed[2])
        keys.add(key)


def heap_samples(heap: List[HeapEntry], limit: int) -> List[Sample]:
    entries = sorted(heap, key=lambda entry: (-entry[0], entry[1]))
    return [entry[3] for entry in entries[:limit]]


def clean_entity_key(value: str) -> str:
    value = value.replace("_", " ").casefold()
    return normalize_space(value)


def negative_sample(
    left: Sample,
    right: Sample,
    source: str,
    seed: int,
) -> Sample:
    entity1 = str(left["entity2"])
    entity2 = str(right["entity2"])
    pair_key = (str(left["entity_uri"]), str(right["entity_uri"]), entity1, entity2)
    sample_id = hashlib.sha256(
        ("\0".join((str(seed),) + pair_key)).encode("utf-8")
    ).hexdigest()
    return {
        "entity1": entity1,
        "entity2": entity2,
        "label": 0,
        "source": source,
        "context": str(left.get("context", "")),
        "source_split": "official_train",
        "internal_split": left["internal_split"],
        "entity1_uri": left["entity_uri"],
        "entity2_uri": right["entity_uri"],
        "sample_id": sample_id,
    }


def build_negatives(
    positives: List[Sample],
    target: int,
    hard_ratio: float,
    hard_min_similarity: float,
    hard_window: int,
    seed: int,
) -> List[Sample]:
    canonical_by_key: Dict[Tuple[str, str], Sample] = {}
    for sample in positives:
        canonical_by_key.setdefault(
            (str(sample["entity_uri"]), str(sample["entity2"])), sample
        )
    candidates = sorted(
        canonical_by_key.values(),
        key=lambda sample: (
            clean_entity_key(str(sample["entity2"])),
            str(sample["entity_uri"]),
        ),
    )
    if len(candidates) < 2:
        raise RuntimeError("Not enough distinct entities to generate negatives")

    hard_target = int(target * hard_ratio)
    hard_heap: List[Tuple[int, str, Sample]] = []
    hard_keys = set()
    for index, left in enumerate(candidates):
        left_key = clean_entity_key(str(left["entity2"]))
        for right in candidates[index + 1 : index + 1 + hard_window]:
            if left["entity_uri"] == right["entity_uri"]:
                continue
            right_key = clean_entity_key(str(right["entity2"]))
            similarity = SequenceMatcher(None, left_key, right_key).ratio()
            if similarity < hard_min_similarity:
                continue
            for first, second in ((left, right), (right, left)):
                key = (str(first["entity_uri"]), str(second["entity_uri"]))
                if key in hard_keys:
                    continue
                sample = negative_sample(
                    first, second, "official_train_hard_negative", seed
                )
                priority = stable_hash(seed, sample["sample_id"])
                entry = (-priority, str(sample["sample_id"]), sample)
                if len(hard_heap) < hard_target:
                    heapq.heappush(hard_heap, entry)
                    hard_keys.add(key)
                elif priority < -hard_heap[0][0]:
                    removed = heapq.heapreplace(hard_heap, entry)
                    removed_sample = removed[2]
                    hard_keys.remove(
                        (
                            str(removed_sample["entity1_uri"]),
                            str(removed_sample["entity2_uri"]),
                        )
                    )
                    hard_keys.add(key)

    hard_samples = [
        entry[2] for entry in sorted(hard_heap, key=lambda entry: (-entry[0], entry[1]))
    ]
    negatives = list(hard_samples)
    pair_keys = {
        (str(sample["entity1_uri"]), str(sample["entity2_uri"]))
        for sample in negatives
    }
    random_target = target - len(negatives)
    random_heap: List[Tuple[int, str, Sample]] = []
    candidate_count = len(candidates)
    rounds = max(8, (random_target // candidate_count) + 8)

    for round_index in range(rounds):
        for index, left in enumerate(candidates):
            offset = 1 + stable_hash(
                seed, left["entity_uri"], round_index
            ) % (candidate_count - 1)
            right = candidates[(index + offset) % candidate_count]
            if left["entity_uri"] == right["entity_uri"]:
                continue
            key = (str(left["entity_uri"]), str(right["entity_uri"]))
            if key in pair_keys:
                continue
            sample = negative_sample(
                left, right, "official_train_random_negative", seed
            )
            priority = stable_hash(seed, sample["sample_id"])
            entry = (-priority, str(sample["sample_id"]), sample)
            if len(random_heap) < random_target:
                heapq.heappush(random_heap, entry)
                pair_keys.add(key)
            elif priority < -random_heap[0][0]:
                removed = heapq.heapreplace(random_heap, entry)
                removed_sample = removed[2]
                pair_keys.remove(
                    (
                        str(removed_sample["entity1_uri"]),
                        str(removed_sample["entity2_uri"]),
                    )
                )
                pair_keys.add(key)

    negatives.extend(
        entry[2]
        for entry in sorted(random_heap, key=lambda entry: (-entry[0], entry[1]))
    )
    if len(negatives) < target:
        raise RuntimeError(
            f"Only generated {len(negatives):,} of {target:,} requested negatives"
        )
    negatives.sort(key=lambda sample: str(sample["sample_id"]))
    return negatives[:target]


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
    if args.negative_ratio <= 0 or not 0 <= args.hard_negative_ratio <= 1:
        raise ValueError("Invalid negative-sampling ratios")

    outputs = [
        args.output_dir / "train.json",
        args.output_dir / "val.json",
        args.output_dir / "test.json",
        args.output_dir / "dataset_stats.json",
    ]
    ensure_outputs_available(outputs, args.force)
    targets = {
        "train": args.train_positive_samples,
        "val": args.val_positive_samples,
        "test": args.test_positive_samples,
    }
    heaps: Dict[str, List[HeapEntry]] = {split: [] for split in targets}
    heap_keys: Dict[str, Set[Tuple[str, str, str]]] = {
        split: set() for split in targets
    }
    counters = Counter()
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
            context = normalize_space(record.get("text"))[: args.max_context_chars]
            triples = record.get("triples")
            if not docid or not context or not isinstance(triples, list):
                counters["invalid_records"] += 1
                continue

            for triple_index, triple in enumerate(triples):
                if not isinstance(triple, dict):
                    counters["invalid_triples"] += 1
                    continue
                for role in ("subject", "object"):
                    mention = triple.get(role)
                    if not isinstance(mention, dict):
                        counters["invalid_entity_mentions"] += 1
                        continue
                    entity_uri = normalize_space(mention.get("uri"))
                    surface = normalize_space(mention.get("surfaceform"))
                    if not entity_uri or not surface:
                        counters["entity_mentions_missing_uri_or_surface"] += 1
                        continue
                    split = assign_split(
                        entity_uri, args.seed, args.train_ratio, args.val_ratio
                    )
                    counters[f"entity_mentions_{split}"] += 1
                    for variant, variant_source in entity_variants(surface):
                        key = (
                            entity_uri,
                            clean_entity_key(variant),
                            clean_entity_key(surface),
                        )
                        sample_id = hashlib.sha256(
                            ("\0".join((str(args.seed),) + key)).encode("utf-8")
                        ).hexdigest()
                        sample: Sample = {
                            "entity1": variant,
                            "entity2": surface,
                            "label": 1,
                            "source": variant_source,
                            "context": context,
                            "source_split": "official_train",
                            "internal_split": split,
                            "entity_uri": entity_uri,
                            "docid": docid,
                            "source_line": line_number,
                            "triple_index": triple_index,
                            "entity_role": role,
                            "sample_id": sample_id,
                        }
                        priority = stable_hash(args.seed, sample_id)
                        keep_unique_smallest(
                            heap=heaps[split],
                            keys=heap_keys[split],
                            sample=sample,
                            key=key,
                            priority=priority,
                            capacity=targets[split],
                        )

    positives = {
        split: heap_samples(heaps[split], targets[split]) for split in targets
    }
    for split in targets:
        if len(positives[split]) < targets[split]:
            raise RuntimeError(
                f"Only selected {len(positives[split]):,} of "
                f"{targets[split]:,} requested {split} positives"
            )

    datasets = {}
    uri_sets = {}
    for split in ("train", "val", "test"):
        negative_target = int(round(len(positives[split]) * args.negative_ratio))
        negatives = build_negatives(
            positives=positives[split],
            target=negative_target,
            hard_ratio=args.hard_negative_ratio,
            hard_min_similarity=args.hard_min_similarity,
            hard_window=args.hard_window,
            seed=args.seed,
        )
        rows = positives[split] + negatives
        rows.sort(key=lambda sample: str(sample["sample_id"]))
        datasets[split] = rows
        uri_sets[split] = {
            str(sample["entity_uri"])
            for sample in positives[split]
        }

    overlaps = {
        "train_val_uri_overlap": len(uri_sets["train"] & uri_sets["val"]),
        "train_test_uri_overlap": len(uri_sets["train"] & uri_sets["test"]),
        "val_test_uri_overlap": len(uri_sets["val"] & uri_sets["test"]),
    }
    if any(overlaps.values()):
        raise AssertionError(f"Entity URI leakage detected: {overlaps}")

    for split, rows in datasets.items():
        (args.output_dir / f"{split}.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    stats = {
        "protocol": "rebel_official_train_entity_alignment_v1",
        "source": {
            "raw_file": str(args.raw_file),
            "raw_sha256_scanned_prefix": raw_digest.hexdigest(),
            "scan_complete": args.max_records is None
            or counters["records_scanned"] < args.max_records,
            "max_records": args.max_records,
            "source_split": "official_train",
        },
        "selection": {
            "method": "smallest_sha256_priority_by_entity_uri_split",
            "seed": args.seed,
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "test_ratio": 1 - args.train_ratio - args.val_ratio,
            "positive_targets": targets,
            "negative_ratio": args.negative_ratio,
            "hard_negative_ratio": args.hard_negative_ratio,
            "hard_min_similarity": args.hard_min_similarity,
            "max_context_chars": args.max_context_chars,
        },
        "scan_counts": dict(sorted(counters.items())),
        "splits": {
            split: {
                "samples": len(rows),
                "positive": sum(row["label"] == 1 for row in rows),
                "negative": sum(row["label"] == 0 for row in rows),
                "entity_uris": len(uri_sets[split]),
                "source_distribution": dict(
                    Counter(str(row["source"]) for row in rows)
                ),
            }
            for split, rows in datasets.items()
        },
        "leakage_checks": overlaps,
    }
    (args.output_dir / "dataset_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw_file", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--train_positive_samples", type=int, default=20000)
    parser.add_argument("--val_positive_samples", type=int, default=3000)
    parser.add_argument("--test_positive_samples", type=int, default=3000)
    parser.add_argument("--negative_ratio", type=float, default=3.0)
    parser.add_argument("--hard_negative_ratio", type=float, default=0.3)
    parser.add_argument("--hard_min_similarity", type=float, default=0.55)
    parser.add_argument("--hard_window", type=int, default=12)
    parser.add_argument("--max_context_chars", type=int, default=500)
    parser.add_argument("--max_records", type=int)
    parser.add_argument("--progress_every", type=int, default=100000)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = build(args)
    for split in ("train", "val", "test"):
        details = stats["splits"][split]
        print(
            f"{split}: samples={details['samples']:,}, "
            f"positive={details['positive']:,}, "
            f"negative={details['negative']:,}, "
            f"entity_uris={details['entity_uris']:,}"
        )
    print(f"Leakage checks: {stats['leakage_checks']}")
    print(f"Statistics written to {args.output_dir / 'dataset_stats.json'}")


if __name__ == "__main__":
    main()
