#!/usr/bin/env python3
"""Build a train-only Wiki-NRE schema and auditable relation URI mapping."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
from collections import Counter, defaultdict
from itertools import zip_longest
from pathlib import Path
from typing import Dict, Iterable, Tuple


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_legacy_descriptions(path: Path | None) -> Dict[str, str]:
    if path is None:
        return {}
    descriptions = {}
    with path.open("r", encoding="utf-8") as source:
        for row in csv.reader(source):
            if row and row[0].strip():
                descriptions[row[0].strip()] = row[1].strip() if len(row) > 1 else ""
    return descriptions


def load_train_relations(manifest_path: Path, reference_path: Path) -> Dict[str, Counter]:
    surfaces: Dict[str, Counter] = defaultdict(Counter)
    with manifest_path.open("r", encoding="utf-8") as manifests, \
         reference_path.open("r", encoding="utf-8") as references:
        for line_number, (manifest_line, reference_line) in enumerate(
            zip_longest(manifests, references), 1
        ):
            if manifest_line is None or reference_line is None:
                raise ValueError("Train manifest and reference row counts differ")
            manifest = json.loads(manifest_line)
            triples = ast.literal_eval(reference_line.strip())
            relation_uris = manifest.get("relation_uris", [])
            if len(relation_uris) != len(triples):
                raise ValueError(
                    f"Row {line_number}: relation URI/triple count mismatch: "
                    f"{len(relation_uris)} != {len(triples)}"
                )
            for relation_uri, triple in zip(relation_uris, triples):
                if not relation_uri or not isinstance(triple, list) or len(triple) != 3:
                    raise ValueError(f"Row {line_number}: invalid relation record")
                relation_surface = str(triple[1]).strip()
                if not relation_surface:
                    raise ValueError(f"Row {line_number}: empty relation surface")
                surfaces[str(relation_uri)][relation_surface] += 1
    return surfaces


def canonical_surface(surface_counts: Counter) -> str:
    return min(surface_counts, key=lambda surface: (-surface_counts[surface], surface))


def load_summary_relations(path: Path) -> Counter:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return Counter(
        {str(uri): int(count) for uri, count in payload["relation_uri_counts"].items()}
    )


def coverage(selected: set[str], counts: Counter) -> Dict:
    total = sum(counts.values())
    covered = sum(count for uri, count in counts.items() if uri in selected)
    unseen = [
        {"uri": uri, "count": count}
        for uri, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        if uri not in selected
    ]
    return {
        "covered_relation_uris": sum(uri in selected for uri in counts),
        "total_relation_uris": len(counts),
        "covered_triples": covered,
        "total_triples": total,
        "triple_coverage": covered / total if total else 0.0,
        "unseen_relations": unseen,
    }


def ensure_available(paths: Iterable[Path], force: bool) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing and not force:
        raise FileExistsError(
            "Refusing to overwrite existing outputs; use --force: " + ", ".join(existing)
        )
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)


def build(args: argparse.Namespace) -> Dict:
    outputs = [args.schema_output, args.mapping_output, args.manifest_output]
    ensure_available(outputs, args.force)
    train_surfaces = load_train_relations(args.train_manifest, args.train_reference)
    legacy_descriptions = load_legacy_descriptions(args.legacy_schema)
    ranked = sorted(
        train_surfaces.items(),
        key=lambda item: (-sum(item[1].values()), item[0]),
    )
    selected_uris = {uri for uri, _counts in ranked}

    labels = {}
    relation_rows = []
    collisions = []
    for rank, (uri, surface_counts) in enumerate(ranked, 1):
        label = canonical_surface(surface_counts)
        if label in labels and labels[label] != uri:
            collisions.append({"label": label, "uris": [labels[label], uri]})
        labels[label] = uri
        legacy_description = legacy_descriptions.get(label, "").strip()
        description = legacy_description or (
            f"The subject has the '{label}' relationship with the object ({uri})."
        )
        relation_rows.append(
            {
                "rank": rank,
                "uri": uri,
                "label": label,
                "description": description,
                "description_source": "legacy_schema"
                if legacy_description
                else "generated_generic",
                "train_count": sum(surface_counts.values()),
                "train_surface_variants": [
                    {"surface": surface, "count": count}
                    for surface, count in sorted(
                        surface_counts.items(), key=lambda item: (-item[1], item[0])
                    )
                ],
            }
        )
    if collisions:
        raise ValueError(f"Canonical relation label collisions: {collisions}")

    with args.schema_output.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        for item in relation_rows:
            writer.writerow([item["label"], item["description"]])

    mapping = {
        "schema_name": "wiki_nre_official_train_all_relations",
        "selection": {
            "method": "all_relation_uris_observed_in_official_train",
            "uses_validation_for_selection": False,
            "uses_test_for_selection": False,
            "relation_count": len(relation_rows),
        },
        "relations": relation_rows,
    }
    args.mapping_output.write_text(
        json.dumps(mapping, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    split_summaries = {
        "train": Counter({uri: sum(counts.values()) for uri, counts in train_surfaces.items()}),
    }
    for name, path in (
        ("val", args.val_summary),
        ("test_full", args.test_full_summary),
        ("test_1000", args.test_1000_summary),
    ):
        if path is not None:
            split_summaries[name] = load_summary_relations(path)

    manifest = {
        "schema_name": mapping["schema_name"],
        "selection": mapping["selection"],
        "inputs": {
            "train_manifest": {
                "path": str(args.train_manifest),
                "sha256": sha256_file(args.train_manifest),
            },
            "train_reference": {
                "path": str(args.train_reference),
                "sha256": sha256_file(args.train_reference),
            },
            "legacy_schema": {
                "path": str(args.legacy_schema),
                "sha256": sha256_file(args.legacy_schema),
            }
            if args.legacy_schema
            else None,
        },
        "description_sources": dict(
            Counter(item["description_source"] for item in relation_rows)
        ),
        "canonical_label_collisions": collisions,
        "coverage": {
            name: coverage(selected_uris, counts)
            for name, counts in split_summaries.items()
        },
    }
    args.manifest_output.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_manifest", required=True, type=Path)
    parser.add_argument("--train_reference", required=True, type=Path)
    parser.add_argument("--legacy_schema", type=Path)
    parser.add_argument("--val_summary", type=Path)
    parser.add_argument("--test_full_summary", type=Path)
    parser.add_argument("--test_1000_summary", type=Path)
    parser.add_argument("--schema_output", required=True, type=Path)
    parser.add_argument("--mapping_output", required=True, type=Path)
    parser.add_argument("--manifest_output", required=True, type=Path)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build(args)
    print(f"Schema relations: {manifest['selection']['relation_count']}")
    for split, details in manifest["coverage"].items():
        print(
            f"{split}: relation_uris={details['total_relation_uris']}, "
            f"triple_coverage={details['triple_coverage']:.4%}, "
            f"unseen_relations={len(details['unseen_relations'])}"
        )
    print(f"Schema written to {args.schema_output}")


if __name__ == "__main__":
    main()
