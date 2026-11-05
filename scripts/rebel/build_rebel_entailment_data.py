#!/usr/bin/env python3
"""Build document-safe entailment data for REBEL triple verification."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


def clean(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def stable_hash(*parts: object) -> int:
    payload = "\0".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest(), "big")


def split_for(docid: str, seed: int, train_ratio: float, val_ratio: float) -> str:
    point = stable_hash(seed, docid) / float(1 << 256)
    if point < train_ratio:
        return "train"
    if point < train_ratio + val_ratio:
        return "val"
    return "test"


def load_relations(path: Optional[Path]) -> List[str]:
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8", newline="") as source:
            return [clean(row[0]) for row in csv.reader(source) if row and clean(row[0])]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [clean(item.get("label") if isinstance(item, dict) else item) for item in payload]
    return [clean(item.get("label")) for item in payload.get("relations", []) if isinstance(item, dict)]


def triple_parts(triple: object) -> Optional[Tuple[str, str, str]]:
    if not isinstance(triple, dict):
        return None
    values = []
    for key in ("subject", "predicate", "object"):
        part = triple.get(key)
        if isinstance(part, dict):
            values.append(clean(part.get("surfaceform") or part.get("surface") or part.get("label") or part.get("uri")))
        else:
            values.append(clean(part))
    return tuple(values) if all(values) else None


CONFUSIONS = {
    "country": ("located in the administrative territorial entity", "location", "instance of"),
    "located in the administrative territorial entity": ("country", "location", "contains administrative territorial entity"),
    "contains administrative territorial entity": ("located in the administrative territorial entity", "country", "contains settlement"),
    "instance of": ("subclass of", "occupation", "genre"),
    "occupation": ("instance of", "position held", "field of work"),
    "member of": ("member of political party", "member of sports team", "part of"),
    "point in time": ("participant in", "season of club or team", "publication date"),
    "genre": ("instance of", "part of", "media franchise"),
}


def choose_wrong_relation(gold: str, relations: Sequence[str], seed: int) -> Optional[str]:
    preferred = [item for item in CONFUSIONS.get(gold, ()) if item in relations and item != gold]
    candidates = preferred + [item for item in relations if item != gold and item not in preferred]
    if not candidates:
        return None
    return candidates[stable_hash(seed, gold) % len(candidates)]


def make_sample(context: str, subject: str, relation: str, obj: str, label: int,
                negative_type: str, docid: str, source_line: int, sample_id: str) -> Dict:
    return {"context": context, "subject": subject, "relation": relation, "object": obj,
            "label": label, "negative_type": negative_type, "docid": docid,
            "source_line": source_line, "sample_id": sample_id}


def build(args: argparse.Namespace) -> Dict:
    if not 0 < args.train_ratio < 1 or not 0 < args.val_ratio < 1 - args.train_ratio:
        raise ValueError("train/val ratios must leave a non-empty test split")
    relations = load_relations(args.schema_mapping)
    if not relations:
        raise ValueError("--schema_mapping must contain at least one relation label")
    outputs = {split: [] for split in ("train", "val", "test")}
    counts = Counter()
    rng = random.Random(args.seed)

    with args.raw_file.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if args.max_records is not None and line_number > args.max_records:
                break
            try:
                record = json.loads(line.lstrip("\ufeff"))
            except json.JSONDecodeError:
                counts["invalid_json"] += 1
                continue
            docid = clean(record.get("docid") or record.get("id") or line_number)
            context = clean(record.get("text") or record.get("context"))[:args.max_context_chars]
            triples = [triple_parts(item) for item in record.get("triples", [])]
            triples = [item for item in triples if item and item[1] in relations]
            if not context or not triples:
                counts["invalid_records"] += 1
                continue
            split = split_for(docid, args.seed, args.train_ratio, args.val_ratio)
            gold_set = set(triples)
            for index, (subject, relation, obj) in enumerate(triples):
                base = f"{args.seed}:{docid}:{line_number}:{index}"
                sample_id = hashlib.sha256((base + ":positive").encode()).hexdigest()
                outputs[split].append(make_sample(context, subject, relation, obj, 1, "positive", docid, line_number, sample_id))
                counts["positive"] += 1

                wrong = choose_wrong_relation(relation, relations, args.seed + index)
                if wrong and (subject, wrong, obj) not in gold_set:
                    sample_id = hashlib.sha256((base + ":relation_swap:" + wrong).encode()).hexdigest()
                    outputs[split].append(make_sample(context, subject, wrong, obj, 0, "relation_swap", docid, line_number, sample_id))
                    counts["relation_swap"] += 1

                if len(triples) > 1:
                    other = triples[(index + 1) % len(triples)]
                    variants = [
                        ("object_swap", subject, relation, other[2]),
                        ("subject_swap", other[0], relation, obj),
                        ("reverse_direction", obj, relation, subject),
                    ]
                    for negative_type, neg_subject, neg_relation, neg_object in variants:
                        if (neg_subject, neg_relation, neg_object) in gold_set:
                            continue
                        sample_id = hashlib.sha256((base + ":" + negative_type).encode()).hexdigest()
                        outputs[split].append(make_sample(context, neg_subject, neg_relation, neg_object, 0, negative_type, docid, line_number, sample_id))
                        counts[negative_type] += 1

    for split, samples in outputs.items():
        rng.shuffle(samples)
        if args.max_samples_per_split is not None:
            outputs[split] = samples[:args.max_samples_per_split]
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / f"{split}.jsonl").write_text(
            "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in outputs[split]),
            encoding="utf-8")
    (args.output_dir / "schema_relations.json").write_text(
        json.dumps(relations, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    stats = {
        "protocol": "rebel_triple_entailment_v1",
        "source": str(args.raw_file),
        "schema_mapping": str(args.schema_mapping),
        "seed": args.seed,
        "splits": {key: {"samples": len(value), "documents": len({item["docid"] for item in value}),
                         "positives": sum(item["label"] == 1 for item in value)}
                   for key, value in outputs.items()},
        "counts_before_limits": dict(counts),
        "negative_types": sorted({item["negative_type"] for values in outputs.values()
                                  for item in values if item["label"] == 0}),
    }
    (args.output_dir / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw_file", required=True, type=Path)
    parser.add_argument("--schema_mapping", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--max_context_chars", type=int, default=1200)
    parser.add_argument("--max_records", type=int)
    parser.add_argument("--max_samples_per_split", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(build(parse_args()), ensure_ascii=False, indent=2))
