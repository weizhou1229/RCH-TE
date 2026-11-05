#!/usr/bin/env python3
"""Build URI-disjoint entity-alignment data from official Wiki-NRE train."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterator, List


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.rebel.build_rebel_entity_alignment_data import (  # noqa: E402
    HeapEntry,
    Sample,
    assign_split,
    build_negatives,
    ensure_outputs_available,
    heap_samples,
    keep_unique_smallest,
    normalize_space,
    stable_hash,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_parquet_rows(path: Path, batch_size: int) -> Iterator[Dict]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit("pyarrow is required to read Wiki-NRE Parquet") from exc

    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(
        batch_size=batch_size,
        columns=["text", "id", "triplets"],
    ):
        yield from batch.to_pylist()


def canonical_surface(surface_counts: Counter) -> str:
    return min(surface_counts, key=lambda value: (-surface_counts[value], value))


def wiki_nre_entity_variants(surface: str) -> List[tuple[str, str]]:
    variants = []
    seen = set()

    def add(value: str, source: str) -> None:
        cleaned = normalize_space(value)
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            variants.append((cleaned, source))

    add(surface, "official_train_identity")
    if " " in surface:
        add(surface.replace(" ", "_"), "official_train_synthetic_space_to_underscore")
    if "_" in surface:
        add(surface.replace("_", " "), "official_train_synthetic_underscore_to_space")
    if "-" in surface:
        add(
            re.sub(r"\s*-\s*", " ", surface),
            "official_train_synthetic_hyphen_to_space",
        )
    if "," in surface:
        add(surface.replace(",", ""), "official_train_synthetic_comma_removal")
    stripped = surface.strip('"\'')
    if stripped != surface:
        add(stripped, "official_train_synthetic_outer_quote_removal")
    ascii_surface = unicodedata.normalize("NFKD", surface).encode(
        "ascii", "ignore"
    ).decode("ascii")
    if ascii_surface and ascii_surface != surface:
        add(ascii_surface, "official_train_synthetic_ascii_fold")
    if re.fullmatch(r"(?:[A-Za-z]\.){2,}", surface):
        add(surface.replace(".", ""), "official_train_synthetic_acronym_period_removal")
    return variants


def positive_source(surface: str, canonical: str, variant_source: str) -> str:
    if surface == canonical:
        return variant_source
    if variant_source == "official_train_identity":
        return "official_train_uri_alias"
    return f"official_train_uri_alias_{variant_source}"


def build(args: argparse.Namespace) -> Dict:
    if not 0 < args.train_ratio < 1:
        raise ValueError("--train_ratio must be between 0 and 1")
    if not 0 < args.val_ratio < 1 - args.train_ratio:
        raise ValueError("--val_ratio must leave a positive test partition")
    if min(
        args.train_positive_samples,
        args.val_positive_samples,
        args.test_positive_samples,
    ) <= 0:
        raise ValueError("Positive sample targets must be positive")
    if args.negative_ratio <= 0 or not 0 <= args.hard_negative_ratio <= 1:
        raise ValueError("Invalid negative-sampling ratios")
    if args.max_surfaces_per_entity <= 0:
        raise ValueError("--max_surfaces_per_entity must be positive")
    if not args.parquet_file.is_file() or args.parquet_file.stat().st_size == 0:
        raise FileNotFoundError(args.parquet_file)

    outputs = [
        args.output_dir / "train.json",
        args.output_dir / "val.json",
        args.output_dir / "test.json",
        args.output_dir / "dataset_stats.json",
    ]
    ensure_outputs_available(outputs, args.force)

    entity_surfaces: Dict[str, Counter] = defaultdict(Counter)
    entity_metadata: Dict[str, Dict] = {}
    counters = Counter()
    for source_row, row in enumerate(
        iter_parquet_rows(args.parquet_file, args.batch_size)
    ):
        if args.max_records is not None and counters["records_scanned"] >= args.max_records:
            break
        counters["records_scanned"] += 1
        if args.progress_every and counters["records_scanned"] % args.progress_every == 0:
            print(f"scanned {counters['records_scanned']:,} records", flush=True)

        text = normalize_space(row.get("text"))
        document_id = row.get("id")
        triples = row.get("triplets")
        if not text or document_id is None or not isinstance(triples, list):
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
                counters["entity_mentions"] += 1
                entity_surfaces[entity_uri][surface] += 1
                entity_metadata.setdefault(
                    entity_uri,
                    {
                        "context": text[: args.max_context_chars],
                        "document_id": str(document_id),
                        "source_row": source_row,
                        "triple_index": triple_index,
                        "entity_role": role,
                    },
                )

    counters["unique_entity_uris"] = len(entity_surfaces)
    counters["entity_uris_with_multiple_surfaces"] = sum(
        len(surfaces) > 1 for surfaces in entity_surfaces.values()
    )
    targets = {
        "train": args.train_positive_samples,
        "val": args.val_positive_samples,
        "test": args.test_positive_samples,
    }
    heaps: Dict[str, List[HeapEntry]] = {split: [] for split in targets}
    heap_keys = {split: set() for split in targets}

    for entity_uri in sorted(entity_surfaces):
        surfaces = entity_surfaces[entity_uri]
        canonical = canonical_surface(surfaces)
        split = assign_split(entity_uri, args.seed, args.train_ratio, args.val_ratio)
        metadata = entity_metadata[entity_uri]
        selected_surfaces = [
            surface
            for surface, _count in sorted(
                surfaces.items(), key=lambda item: (-item[1], item[0])
            )[: args.max_surfaces_per_entity]
        ]
        counters[f"entity_uris_{split}"] += 1

        for surface in selected_surfaces:
            for variant, variant_source in wiki_nre_entity_variants(surface):
                key = (
                    entity_uri,
                    normalize_space(variant),
                    normalize_space(canonical),
                )
                if key in heap_keys[split]:
                    continue
                source = positive_source(surface, canonical, variant_source)
                if source.startswith("official_train_uri_alias"):
                    counters["uri_alias_positive_candidates"] += 1
                if "_synthetic_" in source:
                    counters["synthetic_positive_candidates"] += 1
                sample_id = hashlib.sha256(
                    ("\0".join((str(args.seed),) + key)).encode("utf-8")
                ).hexdigest()
                sample: Sample = {
                    "entity1": variant,
                    "entity2": canonical,
                    "label": 1,
                    "source": source,
                    "context": metadata["context"],
                    "source_split": "official_train",
                    "internal_split": split,
                    "entity_uri": entity_uri,
                    "document_id": metadata["document_id"],
                    "source_row": metadata["source_row"],
                    "triple_index": metadata["triple_index"],
                    "entity_role": metadata["entity_role"],
                    "surface_frequency": surfaces[surface],
                    "canonical_frequency": surfaces[canonical],
                    "sample_id": sample_id,
                }
                keep_unique_smallest(
                    heap=heaps[split],
                    keys=heap_keys[split],
                    sample=sample,
                    key=key,
                    priority=stable_hash(args.seed, sample_id),
                    capacity=targets[split],
                )

    positives = {
        split: heap_samples(heaps[split], targets[split]) for split in targets
    }
    for split, target in targets.items():
        if len(positives[split]) < target:
            raise RuntimeError(
                f"Only selected {len(positives[split]):,} of "
                f"{target:,} requested {split} positives"
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
            str(sample["entity_uri"]) for sample in positives[split]
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
        "protocol": "wiki_nre_official_train_entity_alignment_v2",
        "source": {
            "parquet_file": str(args.parquet_file),
            "parquet_sha256": sha256_file(args.parquet_file),
            "source_split": "official_train",
            "records_scanned": counters["records_scanned"],
            "max_records": args.max_records,
        },
        "selection": {
            "method": "uri_disjoint_hash_split_with_train_uri_surface_aliases",
            "seed": args.seed,
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "test_ratio": 1 - args.train_ratio - args.val_ratio,
            "positive_targets": targets,
            "negative_ratio": args.negative_ratio,
            "hard_negative_ratio": args.hard_negative_ratio,
            "hard_min_similarity": args.hard_min_similarity,
            "hard_window": args.hard_window,
            "max_surfaces_per_entity": args.max_surfaces_per_entity,
            "max_context_chars": args.max_context_chars,
        },
        "scan_counts": dict(sorted(counters.items())),
        "splits": {
            split: {
                "samples": len(rows),
                "positive": sum(row["label"] == 1 for row in rows),
                "negative": sum(row["label"] == 0 for row in rows),
                "entity_uris": len(uri_sets[split]),
                "uri_alias_positives": sum(
                    row["label"] == 1
                    and str(row["source"]).startswith("official_train_uri_alias")
                    for row in rows
                ),
                "synthetic_positives": sum(
                    row["label"] == 1 and "_synthetic_" in str(row["source"])
                    for row in rows
                ),
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
    parser.add_argument("--parquet_file", required=True, type=Path)
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
    parser.add_argument("--max_surfaces_per_entity", type=int, default=4)
    parser.add_argument("--max_context_chars", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--max_records", type=int)
    parser.add_argument("--progress_every", type=int, default=25000)
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
            f"entity_uris={details['entity_uris']:,}, "
            f"uri_alias_positives={details['uri_alias_positives']:,}, "
            f"synthetic_positives={details['synthetic_positives']:,}"
        )
    print(f"Leakage checks: {stats['leakage_checks']}")
    print(f"Statistics written to {args.output_dir / 'dataset_stats.json'}")


if __name__ == "__main__":
    main()
