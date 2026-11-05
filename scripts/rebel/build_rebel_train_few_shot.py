#!/usr/bin/env python3
"""Select diverse train-only REBEL examples and write all BF-EDC few-shot files."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path
from typing import Dict, List


def load_references(path: Path) -> List[List[List[str]]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            try:
                payload = ast.literal_eval(line.strip())
            except Exception as exc:
                raise ValueError(f"{path}:{line_number}: invalid reference row: {exc}") from exc
            rows.append([[str(part) for part in triple] for triple in payload])
    return rows


def load_manifest(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_schema(path: Path) -> Dict[str, str]:
    schema = {}
    with path.open("r", encoding="utf-8") as f:
        for row in csv.reader(f):
            if row and row[0].strip():
                schema[row[0].strip()] = row[1].strip() if len(row) > 1 else ""
    return schema


def load_relation_ranks(path: Path) -> Dict[str, int]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return {item["uri"]: int(item["rank"]) for item in payload["relations"]}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unique_entities(triples: List[List[str]]) -> List[str]:
    entities = []
    seen = set()
    for subject, _, obj in triples:
        for entity in (subject, obj):
            if entity not in seen:
                seen.add(entity)
                entities.append(entity)
    return entities


def unique_relations(triples: List[List[str]]) -> List[str]:
    relations = []
    seen = set()
    for _, relation, _ in triples:
        if relation not in seen:
            seen.add(relation)
            relations.append(relation)
    return relations


def select_examples(
    texts: List[str],
    references: List[List[List[str]]],
    manifest: List[Dict],
    relation_ranks: Dict[str, int],
    count: int,
    min_triples: int,
    max_triples: int,
    max_text_chars: int,
) -> List[int]:
    eligible = []
    for index, (text, triples, item) in enumerate(zip(texts, references, manifest)):
        relation_uris = item.get("relation_uris", [])
        if not (min_triples <= len(triples) <= max_triples):
            continue
        if len(text) > max_text_chars:
            continue
        if item.get("outside_schema_triples", 0):
            continue
        if len(relation_uris) != len(triples):
            continue
        if any(uri not in relation_ranks for uri in relation_uris):
            continue
        eligible.append(index)

    selected = []
    covered_uris = set()
    remaining = set(eligible)
    while remaining and len(selected) < count:
        def score(index: int):
            relation_uris = set(manifest[index]["relation_uris"])
            new_uris = relation_uris - covered_uris
            commonness = sum(1000 - relation_ranks[uri] for uri in new_uris)
            return (
                len(new_uris),
                commonness,
                len(relation_uris),
                -len(texts[index]),
                -index,
            )

        best = max(remaining, key=score)
        selected.append(best)
        covered_uris.update(manifest[best]["relation_uris"])
        remaining.remove(best)

    if len(selected) != count:
        raise RuntimeError(f"Only {len(selected)} eligible few-shot examples found")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--schema", required=True, type=Path)
    parser.add_argument("--schema_mapping", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--examples", type=int, default=6)
    parser.add_argument("--min_triples", type=int, default=2)
    parser.add_argument("--max_triples", type=int, default=4)
    parser.add_argument("--max_text_chars", type=int, default=500)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    texts = args.dataset.read_text(encoding="utf-8").splitlines()
    references = load_references(args.reference)
    manifest = load_manifest(args.manifest)
    if len({len(texts), len(references), len(manifest)}) != 1:
        raise ValueError("Dataset, reference, and manifest row counts differ")

    schema = load_schema(args.schema)
    relation_ranks = load_relation_ranks(args.schema_mapping)
    selected = select_examples(
        texts=texts,
        references=references,
        manifest=manifest,
        relation_ranks=relation_ranks,
        count=args.examples,
        min_triples=args.min_triples,
        max_triples=args.max_triples,
        max_text_chars=args.max_text_chars,
    )

    output_paths = {
        "oie": args.output_dir / "oie_few_shot_examples.txt",
        "sd": args.output_dir / "sd_few_shot_examples.txt",
        "ee": args.output_dir / "ee_few_shot_examples.txt",
        "refine": args.output_dir / "oie_few_shot_refine_examples.txt",
        "manifest": args.output_dir / "few_shot_manifest.json",
    }
    existing = [str(path) for path in output_paths.values() if path.exists()]
    if existing and not args.force:
        raise FileExistsError("Refusing to overwrite: " + ", ".join(existing))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    oie_blocks = []
    sd_blocks = []
    ee_blocks = []
    refine_blocks = []
    selected_manifest = []
    for example_number, index in enumerate(selected, 1):
        text = texts[index]
        triples = references[index]
        relations = unique_relations(triples)
        entities = unique_entities(triples)
        answers = "\n".join(
            f"{relation}: {schema[relation]}" for relation in relations
        )

        oie_blocks.append(
            f"Example {example_number}:\nText: {text}\nTriplets: {triples}"
        )
        sd_blocks.append(
            f"Example {example_number}:\nText: {text}\nTriples: {triples}\n"
            f"Relations: {relations}\nAnswer:\n{answers}"
        )
        ee_blocks.append(
            f"Example {example_number}:\nText: {text}\nEntities: {entities}"
        )
        refine_blocks.append(
            f"Example {example_number}:\nText: {text}\n"
            f"Candidate entities: {entities}\nTriplets: {triples}"
        )
        selected_manifest.append(
            {
                "example": example_number,
                "protocol_index": index,
                "docid": manifest[index].get("docid"),
                "sentence_id": manifest[index].get("sentence_id"),
                "relation_uris": manifest[index].get("relation_uris", []),
                "relations": relations,
                "triple_count": len(triples),
            }
        )

    output_paths["oie"].write_text("\n\n".join(oie_blocks) + "\n", encoding="utf-8")
    output_paths["sd"].write_text("\n\n".join(sd_blocks) + "\n", encoding="utf-8")
    output_paths["ee"].write_text("\n\n".join(ee_blocks) + "\n", encoding="utf-8")
    output_paths["refine"].write_text(
        "\n\n".join(refine_blocks) + "\n", encoding="utf-8"
    )
    output_paths["manifest"].write_text(
        json.dumps(
            {
                "source_split": "train",
                "selection_method": "greedy_uncovered_relation_coverage",
                "inputs": {
                    "dataset": {"path": str(args.dataset), "sha256": sha256_file(args.dataset)},
                    "reference": {
                        "path": str(args.reference),
                        "sha256": sha256_file(args.reference),
                    },
                    "manifest": {
                        "path": str(args.manifest),
                        "sha256": sha256_file(args.manifest),
                    },
                    "schema": {"path": str(args.schema), "sha256": sha256_file(args.schema)},
                },
                "selected_examples": selected_manifest,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    covered_relations = {relation for item in selected_manifest for relation in item["relations"]}
    print(f"Selected examples: {len(selected_manifest)}")
    print(f"Covered relation labels: {len(covered_relations)}")
    print(f"Few-shot files written to {args.output_dir}")


if __name__ == "__main__":
    main()
