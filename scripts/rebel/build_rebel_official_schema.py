#!/usr/bin/env python3
"""Build a train-selected official REBEL schema and auditable URI mapping."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Dict


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_relations(path: Path) -> Dict[str, Dict]:
    payload = load_json(path)
    return {item["uri"]: item for item in payload["relations"]}


def load_legacy_descriptions(path: Path | None) -> Dict[str, str]:
    if path is None:
        return {}
    descriptions = {}
    with path.open("r", encoding="utf-8") as f:
        for row in csv.reader(f):
            if row and row[0].strip():
                descriptions[row[0].strip()] = row[1].strip() if len(row) > 1 else ""
    return descriptions


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generic_description(label: str, uri: str) -> str:
    return f"The subject has the '{label}' relationship with the object ({uri})."


def coverage(selected: set[str], relations: Dict[str, Dict]) -> Dict:
    total = sum(item["total"] for item in relations.values())
    covered = sum(item["total"] for uri, item in relations.items() if uri in selected)
    return {
        "covered_relations": sum(1 for uri in relations if uri in selected),
        "total_relations": len(relations),
        "covered_triples": covered,
        "total_triples": total,
        "triple_coverage": covered / total if total else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_aliases", required=True, type=Path)
    parser.add_argument("--val_aliases", required=True, type=Path)
    parser.add_argument("--legacy_schema", type=Path)
    parser.add_argument("--top_k", type=int, default=200)
    parser.add_argument("--schema_output", required=True, type=Path)
    parser.add_argument("--mapping_output", required=True, type=Path)
    parser.add_argument("--manifest_output", required=True, type=Path)
    args = parser.parse_args()

    if args.top_k <= 0:
        raise SystemExit("--top_k must be positive")

    train = load_relations(args.train_aliases)
    val = load_relations(args.val_aliases)
    legacy_descriptions = load_legacy_descriptions(args.legacy_schema)
    ranked = sorted(train.values(), key=lambda item: (-item["total"], item["uri"]))
    selected_items = ranked[: min(args.top_k, len(ranked))]
    selected_uris = {item["uri"] for item in selected_items}

    labels: Dict[str, str] = {}
    collisions = []
    for item in selected_items:
        label = item["canonical_surface"]
        previous_uri = labels.get(label)
        if previous_uri is not None and previous_uri != item["uri"]:
            collisions.append({"surface": label, "uris": [previous_uri, item["uri"]]})
        labels[label] = item["uri"]
    if collisions:
        raise ValueError(f"Canonical surface collisions prevent schema generation: {collisions}")

    mapping = {
        "schema_name": f"rebel_official_top{len(selected_items)}",
        "selection": {
            "method": "top_k_train_frequency",
            "requested_top_k": args.top_k,
            "actual_top_k": len(selected_items),
            "uses_validation_for_selection": False,
            "uses_test": False,
        },
        "relations": [],
    }

    args.schema_output.parent.mkdir(parents=True, exist_ok=True)
    with args.schema_output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for rank, item in enumerate(selected_items, 1):
            uri = item["uri"]
            label = item["canonical_surface"]
            legacy_description = legacy_descriptions.get(label, "").strip()
            description = legacy_description or generic_description(label, uri)
            description_source = "legacy_schema" if legacy_description else "generated_generic"
            writer.writerow([label, description])
            mapping["relations"].append(
                {
                    "rank": rank,
                    "uri": uri,
                    "label": label,
                    "description": description,
                    "description_source": description_source,
                    "train_count": item["total"],
                    "val_count": val.get(uri, {}).get("total", 0),
                    "train_surface_variants": item["surfaces"],
                }
            )

    args.mapping_output.parent.mkdir(parents=True, exist_ok=True)
    with args.mapping_output.open("w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)

    train_coverage = coverage(selected_uris, train)
    val_coverage = coverage(selected_uris, val)
    val_outside = [
        {
            "uri": uri,
            "label": item["canonical_surface"],
            "count": item["total"],
            "seen_in_train": uri in train,
        }
        for uri, item in sorted(
            val.items(),
            key=lambda pair: (-pair[1]["total"], pair[0]),
        )
        if uri not in selected_uris
    ]
    manifest = {
        "schema_name": mapping["schema_name"],
        "selection": mapping["selection"],
        "inputs": {
            "train_aliases": {
                "path": str(args.train_aliases),
                "sha256": sha256_file(args.train_aliases),
            },
            "val_aliases": {
                "path": str(args.val_aliases),
                "sha256": sha256_file(args.val_aliases),
            },
            "legacy_schema": {
                "path": str(args.legacy_schema),
                "sha256": sha256_file(args.legacy_schema),
            }
            if args.legacy_schema
            else None,
        },
        "coverage": {
            "train": train_coverage,
            "val": val_coverage,
        },
        "description_sources": {
            "legacy_schema": sum(
                1 for item in mapping["relations"] if item["description_source"] == "legacy_schema"
            ),
            "generated_generic": sum(
                1
                for item in mapping["relations"]
                if item["description_source"] == "generated_generic"
            ),
        },
        "canonical_surface_collisions": collisions,
        "val_relations_outside_schema": val_outside,
    }

    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest_output.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"Schema relations: {len(selected_items)}")
    print(f"Train triple coverage: {train_coverage['triple_coverage']:.4%}")
    print(f"Val triple coverage: {val_coverage['triple_coverage']:.4%}")
    print(
        "Descriptions: "
        f"legacy={manifest['description_sources']['legacy_schema']}, "
        f"generated={manifest['description_sources']['generated_generic']}"
    )
    print(f"Schema written to {args.schema_output}")


if __name__ == "__main__":
    main()
