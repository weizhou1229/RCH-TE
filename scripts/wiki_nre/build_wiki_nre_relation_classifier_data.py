#!/usr/bin/env python3
"""Build split-safe relation-classifier data from official Wiki-NRE train files."""

from __future__ import annotations

import argparse
import ast
import hashlib
import heapq
import json
import re
from collections import Counter, defaultdict
from itertools import zip_longest
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


Sample = Dict[str, object]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def stable_hash(*parts: object) -> int:
    payload = "\0".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest(), "big")


def assign_split(
    document_id: str,
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> str:
    point = stable_hash(seed, document_id) / float(1 << 256)
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
    if not uris or len(uris) != len(set(uris)):
        raise ValueError("Schema relation URIs must be non-empty and unique")
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

    alias_count = 0
    for row in schema_item.get("train_surface_variants", []):
        surface = row.get("surface") if isinstance(row, dict) else row
        before = len(variants)
        add(surface, "train_relation_alias")
        if len(variants) > before:
            alias_count += 1
        if alias_count >= aliases_per_triple:
            break

    for value, _source in list(variants):
        add(
            normalize_space(value.replace("_", " ").replace("-", " ")),
            "separator_normalization",
        )
    return variants


def keep_smallest(
    heap: List[Tuple[int, str, Sample]],
    sample: Sample,
    priority: int,
    capacity: int,
) -> None:
    entry = (-priority, str(sample["sample_id"]), sample)
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
        for _priority, _tie_breaker, sample in entries:
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


def select_forced_train_documents(
    manifest_path: Path,
    schema_uris: Iterable[str],
    seed: int,
    documents_per_relation: int,
    max_records: int | None,
) -> Tuple[set[str], Dict[str, int]]:
    if documents_per_relation <= 0:
        return set(), {uri: 0 for uri in schema_uris}

    selected_uris = set(schema_uris)
    heaps: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
    heap_documents: Dict[str, set[str]] = defaultdict(set)
    with manifest_path.open("r", encoding="utf-8") as manifests:
        for line_number, line in enumerate(manifests, 1):
            if max_records is not None and line_number > max_records:
                break
            manifest = json.loads(line)
            if manifest.get("source_split") != "train":
                raise ValueError(
                    f"Manifest row {line_number} does not come from official train"
                )
            if "id" not in manifest:
                raise ValueError(f"Manifest row {line_number} is missing document id")
            document_id = str(manifest["id"])
            for uri_value in set(manifest.get("relation_uris", [])):
                uri = normalize_space(uri_value)
                if uri not in selected_uris or document_id in heap_documents[uri]:
                    continue
                priority = stable_hash(seed, "forced_train", uri, document_id)
                entry = (-priority, document_id)
                if len(heaps[uri]) < documents_per_relation:
                    heapq.heappush(heaps[uri], entry)
                    heap_documents[uri].add(document_id)
                elif priority < -heaps[uri][0][0]:
                    removed = heapq.heapreplace(heaps[uri], entry)
                    heap_documents[uri].remove(removed[1])
                    heap_documents[uri].add(document_id)

    forced_documents = {
        document_id
        for documents in heap_documents.values()
        for document_id in documents
    }
    counts = {uri: len(heap_documents[uri]) for uri in selected_uris}
    return forced_documents, counts


def parse_reference(line: str, line_number: int) -> List[List[str]]:
    try:
        value = ast.literal_eval(line.strip())
    except Exception as exc:
        raise ValueError(f"Reference row {line_number} is invalid: {exc}") from exc
    if not isinstance(value, list):
        raise ValueError(f"Reference row {line_number} is not a list")
    triples = []
    for triple in value:
        if not isinstance(triple, list) or len(triple) != 3:
            raise ValueError(f"Reference row {line_number} has an invalid triple")
        parts = [normalize_space(part) for part in triple]
        if not all(parts):
            raise ValueError(f"Reference row {line_number} has an empty triple part")
        triples.append(parts)
    return triples


def validate_manifest_row(
    manifest: Dict,
    text: str,
    triples: List[List[str]],
    line_number: int,
) -> None:
    if manifest.get("source_split") != "train":
        raise ValueError(f"Row {line_number} does not come from official train")
    if int(manifest.get("triple_count", -1)) != len(triples):
        raise ValueError(f"Row {line_number}: manifest/reference triple count mismatch")
    if manifest.get("text_sha256") and manifest["text_sha256"] != sha256_text(text):
        raise ValueError(f"Row {line_number}: text SHA-256 mismatch")
    triple_payload = json.dumps(triples, ensure_ascii=False, separators=(",", ":"))
    if (
        manifest.get("triples_sha256")
        and manifest["triples_sha256"] != sha256_text(triple_payload)
    ):
        raise ValueError(f"Row {line_number}: triple SHA-256 mismatch")


def build(args: argparse.Namespace) -> Dict:
    if not 0 < args.train_ratio < 1:
        raise ValueError("--train_ratio must be between 0 and 1")
    if not 0 < args.val_ratio < 1 - args.train_ratio:
        raise ValueError("--val_ratio must leave a positive test partition")
    if min(args.train_per_relation, args.val_per_relation, args.test_per_relation) <= 0:
        raise ValueError("Per-relation sample limits must be positive")
    if min(
        args.min_train_per_relation,
        args.min_val_per_relation,
        args.min_test_per_relation,
        args.min_train_documents_per_relation,
    ) < 0:
        raise ValueError("Per-relation minimums cannot be negative")
    if args.aliases_per_triple < 0:
        raise ValueError("--aliases_per_triple cannot be negative")

    output_paths = [
        args.output_dir / "train.json",
        args.output_dir / "val.json",
        args.output_dir / "test.json",
        args.output_dir / "schema_relations.json",
        args.output_dir / "stats.json",
    ]
    ensure_outputs_available(output_paths, args.force)
    schema_labels, schema_by_uri = load_schema_mapping(args.schema_mapping)
    forced_train_documents, forced_train_counts = select_forced_train_documents(
        manifest_path=args.manifest,
        schema_uris=schema_by_uri,
        seed=args.seed,
        documents_per_relation=args.min_train_documents_per_relation,
        max_records=args.max_records,
    )
    capacities = {
        "train": args.train_per_relation,
        "val": args.val_per_relation,
        "test": args.test_per_relation,
    }
    heaps: Dict[Tuple[str, str], List[Tuple[int, str, Sample]]] = defaultdict(list)
    counters = Counter()
    candidate_counts = Counter()

    with args.dataset.open("r", encoding="utf-8") as datasets, \
         args.reference.open("r", encoding="utf-8") as references, \
         args.manifest.open("r", encoding="utf-8") as manifests:
        rows = zip_longest(datasets, references, manifests)
        for line_number, row in enumerate(rows, 1):
            dataset_line, reference_line, manifest_line = row
            if dataset_line is None or reference_line is None or manifest_line is None:
                raise ValueError("Dataset, reference, and manifest row counts differ")
            if args.max_records is not None and counters["records_scanned"] >= args.max_records:
                break

            text = normalize_space(dataset_line)
            triples = parse_reference(reference_line, line_number)
            manifest = json.loads(manifest_line)
            validate_manifest_row(manifest, text, triples, line_number)
            relation_uris = manifest.get("relation_uris", [])
            if len(relation_uris) != len(triples):
                raise ValueError(f"Row {line_number}: relation URI/triple count mismatch")
            if "id" not in manifest:
                raise ValueError(f"Row {line_number}: missing official document id")

            document_id = str(manifest["id"])
            split = (
                "train"
                if document_id in forced_train_documents
                else assign_split(document_id, args.seed, args.train_ratio, args.val_ratio)
            )
            counters["records_scanned"] += 1
            counters[f"documents_{split}"] += 1
            counters["documents_forced_to_train"] += int(
                document_id in forced_train_documents
            )
            if args.progress_every and counters["records_scanned"] % args.progress_every == 0:
                print(f"scanned {counters['records_scanned']:,} records", flush=True)

            for triple_index, (triple, relation_uri_value) in enumerate(
                zip(triples, relation_uris)
            ):
                relation_uri = normalize_space(relation_uri_value)
                schema_item = schema_by_uri.get(relation_uri)
                if schema_item is None:
                    counters["triples_outside_schema"] += 1
                    continue
                subject, predicate_surface, obj = triple
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
                            f"{args.seed}\0{document_id}\0{line_number}\0{triple_index}"
                            f"\0{relation_uri}\0{variant_index}\0{variant}"
                        ).encode("utf-8")
                    ).hexdigest()
                    sample: Sample = {
                        "subject": subject,
                        "oie_relation": variant,
                        "object": obj,
                        "context": text[: args.max_context_chars],
                        "label": label,
                        "sample_type": "positive_clean",
                        "variant_source": variant_source,
                        "source_split": "official_train",
                        "internal_split": split,
                        "document_id": document_id,
                        "source_row": manifest.get("source_row"),
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
    split_documents = {
        split: {str(sample["document_id"]) for sample in samples}
        for split, samples in splits.items()
    }
    overlaps = {
        "train_val_document_overlap": len(
            split_documents["train"] & split_documents["val"]
        ),
        "train_test_document_overlap": len(
            split_documents["train"] & split_documents["test"]
        ),
        "val_test_document_overlap": len(
            split_documents["val"] & split_documents["test"]
        ),
    }
    if any(overlaps.values()):
        raise AssertionError(f"Document leakage detected: {overlaps}")

    minimums = {
        "train": args.min_train_per_relation,
        "val": args.min_val_per_relation,
        "test": args.min_test_per_relation,
    }
    per_relation = {}
    sparse_relations = {}
    for label in schema_labels:
        counts = {
            split: sum(sample["label"] == label for sample in splits[split])
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
        "protocol": "wiki_nre_official_train_relation_classifier_v1",
        "source": {
            "dataset": str(args.dataset),
            "dataset_sha256": sha256_file(args.dataset),
            "reference": str(args.reference),
            "reference_sha256": sha256_file(args.reference),
            "manifest": str(args.manifest),
            "manifest_sha256": sha256_file(args.manifest),
            "source_split": "official_train",
            "records_scanned": counters["records_scanned"],
            "max_records": args.max_records,
        },
        "schema": {
            "mapping": str(args.schema_mapping),
            "mapping_sha256": sha256_file(args.schema_mapping),
            "relation_count": len(schema_labels),
            "relations": schema_labels,
        },
        "selection": {
            "method": "document_hash_split_then_smallest_sample_hash_per_relation",
            "seed": args.seed,
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "test_ratio": 1 - args.train_ratio - args.val_ratio,
            "per_relation_limits": capacities,
            "min_train_documents_per_relation": (
                args.min_train_documents_per_relation
            ),
            "forced_train_documents": len(forced_train_documents),
            "forced_train_documents_per_relation": forced_train_counts,
            "aliases_per_triple": args.aliases_per_triple,
            "max_context_chars": args.max_context_chars,
        },
        "scan_counts": dict(sorted(counters.items())),
        "selected": {
            split: {
                "samples": len(samples),
                "documents": len(split_documents[split]),
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
        "leakage_checks": overlaps,
    }
    (args.output_dir / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--schema_mapping", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--train_per_relation", type=int, default=500)
    parser.add_argument("--val_per_relation", type=int, default=100)
    parser.add_argument("--test_per_relation", type=int, default=100)
    parser.add_argument("--min_train_per_relation", type=int, default=20)
    parser.add_argument("--min_val_per_relation", type=int, default=5)
    parser.add_argument("--min_test_per_relation", type=int, default=5)
    parser.add_argument("--min_train_documents_per_relation", type=int, default=0)
    parser.add_argument("--aliases_per_triple", type=int, default=2)
    parser.add_argument("--max_context_chars", type=int, default=600)
    parser.add_argument("--max_records", type=int)
    parser.add_argument("--progress_every", type=int, default=25000)
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
