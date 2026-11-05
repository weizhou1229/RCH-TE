#!/usr/bin/env python3
"""Build document-split binary entailment data for WebNLG."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


Triple = Tuple[str, str, str]


def clean(value: object) -> str:
    return " ".join(str(value or "").strip().split())


def stable_hash(*parts: object) -> int:
    payload = "\0".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest(), "big")


def split_for(index: int, seed: int, train_ratio: float, val_ratio: float) -> str:
    point = stable_hash(seed, index) / float(1 << 256)
    if point < train_ratio:
        return "train"
    if point < train_ratio + val_ratio:
        return "val"
    return "test"


def load_relations(path: Path) -> List[str]:
    values = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = clean(line)
        if line and not line.lower().startswith("relation"):
            values.append(line.split(",", 1)[0].strip())
    return sorted(set(values))


def read_triples(path: Path) -> List[List[Triple]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        payload = ast.literal_eval(line)
        triples = []
        for item in payload:
            if isinstance(item, (list, tuple)) and len(item) == 3:
                triple = tuple(clean(part) for part in item)
                if all(triple):
                    triples.append(triple)
        rows.append(triples)
    return rows


def sample(context: str, triple: Triple, label: int, negative_type: str,
           row: int, sample_id: str) -> Dict:
    return {
        "context": context,
        "subject": triple[0],
        "relation": triple[1],
        "object": triple[2],
        "label": label,
        "negative_type": negative_type,
        "docid": f"webnlg-{row}",
        "source_line": row,
        "sample_id": sample_id,
    }


def build(args: argparse.Namespace) -> Dict:
    contexts = [clean(line) for line in args.context.read_text(encoding="utf-8").splitlines()]
    references = read_triples(args.reference)
    if len(contexts) != len(references):
        raise ValueError(f"row mismatch context={len(contexts)} reference={len(references)}")
    relations = load_relations(args.schema)
    relation_set = set(relations)
    outputs = {name: [] for name in ("train", "val", "test")}
    counts = Counter()
    rng = random.Random(args.seed)

    for row_number, (context, triples) in enumerate(zip(contexts, references), 1):
        triples = [item for item in triples if item[1] in relation_set]
        if not context or not triples:
            continue
        split = split_for(row_number, args.seed, args.train_ratio, args.val_ratio)
        gold = set(triples)
        for index, triple in enumerate(triples):
            base = f"{args.seed}:{row_number}:{index}"
            positive_id = hashlib.sha256((base + ":positive").encode()).hexdigest()
            outputs[split].append(sample(context, triple, 1, "positive", row_number, positive_id))
            counts["positive"] += 1

            alternatives = [relation for relation in relations if relation != triple[1]]
            if alternatives:
                wrong_relation = alternatives[stable_hash(base, "relation") % len(alternatives)]
                candidate = (triple[0], wrong_relation, triple[2])
                if candidate not in gold:
                    sid = hashlib.sha256((base + ":relation_swap").encode()).hexdigest()
                    outputs[split].append(sample(context, candidate, 0, "relation_swap", row_number, sid))
                    counts["relation_swap"] += 1

            other = triples[(index + 1) % len(triples)]
            negatives = [
                ("object_swap", (triple[0], triple[1], other[2])),
                ("subject_swap", (other[0], triple[1], triple[2])),
                ("reverse_direction", (triple[2], triple[1], triple[0])),
            ]
            for negative_type, candidate in negatives:
                if candidate in gold:
                    continue
                sid = hashlib.sha256((base + ":" + negative_type).encode()).hexdigest()
                outputs[split].append(sample(context, candidate, 0, negative_type, row_number, sid))
                counts[negative_type] += 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in outputs.items():
        rng.shuffle(rows)
        if args.max_samples_per_split is not None:
            rows = rows[:args.max_samples_per_split]
        outputs[split] = rows
        (args.output_dir / f"{split}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8")

    (args.output_dir / "schema_relations.json").write_text(
        json.dumps(relations, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    stats = {
        "protocol": "webnlg_triple_entailment_v1",
        "context": str(args.context),
        "reference": str(args.reference),
        "schema": str(args.schema),
        "seed": args.seed,
        "splits": {name: {"samples": len(rows), "positives": sum(row["label"] == 1 for row in rows)}
                   for name, rows in outputs.items()},
        "counts_before_limits": dict(counts),
    }
    (args.output_dir / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--schema", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--max_samples_per_split", type=int)
    print(json.dumps(build(parser.parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
