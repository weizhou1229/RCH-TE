#!/usr/bin/env python3
"""Apply verifier scores to an existing REBEL raw extraction.

Scores are JSONL records with line (1-based), triple_index (0-based) and
score in [0, 1]. Missing scores are kept by default so an incomplete
verifier run cannot silently erase predictions.
"""

from __future__ import annotations

import argparse
import ast
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

from edc.output_postprocessor import (
    OutputPostprocessor,
    OutputPostprocessorConfig,
    write_postprocess_artifacts,
)


def read_rows(path: Path) -> List[list]:
    return [ast.literal_eval(line) for line in path.read_text(encoding="utf-8").splitlines()]


def read_scores(path: Path) -> Dict[tuple, float]:
    scores: Dict[tuple, float] = {}
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: score record must be an object")
            try:
                key = (int(record["line"]), int(record["triple_index"]))
                raw_score = record.get("score")
                if raw_score is None:
                    raw_score = record["entailment_probability"]
                score = float(raw_score)
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_number}: expected line, triple_index, score") from exc
            if not 0.0 <= score <= 1.0:
                raise ValueError(f"{path}:{line_number}: score must be in [0, 1]")
            scores[key] = score
    return scores


def apply(rows: List[list], scores: Dict[tuple, float], threshold: float,
          missing_policy: str, max_triples_per_row: Optional[int] = None,
          threshold_policy: Optional[Dict] = None):
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("--threshold must be in [0, 1]")
    output = []
    changes = []
    counters = Counter()
    for row_index, row in enumerate(rows, 1):
        kept = []
        for triple_index, triple in enumerate(row):
            score = scores.get((row_index, triple_index))
            if score is None:
                if missing_policy == "drop":
                    counters["missing_score_dropped"] += 1
                    changes.append({"line": row_index, "triple_index": triple_index,
                                    "reason": "missing_score", "score": None,
                                    "triple": list(triple)})
                    continue
                kept.append(triple)
                counters["missing_score_kept"] += 1
                continue
            relation_threshold = threshold
            if threshold_policy is not None and len(triple) == 3:
                relation_threshold = float(
                    threshold_policy.get("thresholds", {}).get(
                        str(triple[1]), threshold_policy.get("default_threshold", threshold)
                    )
                )
            if score < relation_threshold:
                counters["below_threshold"] += 1
                changes.append({"line": row_index, "triple_index": triple_index,
                                "reason": "below_threshold", "score": score,
                                "threshold": relation_threshold, "triple": list(triple)})
                continue
            kept.append(triple)
            counters["kept"] += 1
        if max_triples_per_row is not None and len(kept) > max_triples_per_row:
            for triple_index in range(max_triples_per_row, len(kept)):
                counters["row_budget_dropped"] += 1
                changes.append({"line": row_index, "triple_index": triple_index,
                                "reason": "row_triple_budget", "score": None,
                                "threshold": threshold, "triple": list(kept[triple_index])})
            kept = kept[:max_triples_per_row]
        output.append(kept)
    return output, changes, counters


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--scores", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    threshold_group = parser.add_mutually_exclusive_group(required=True)
    threshold_group.add_argument("--threshold", type=float)
    threshold_group.add_argument("--threshold_policy", type=Path)
    parser.add_argument("--missing_policy", choices=("keep", "drop"), default="keep")
    parser.add_argument("--max_triples_per_row", type=int)
    parser.add_argument("--profile", help="Optional postprocess profile applied after verification")
    args = parser.parse_args()

    rows = read_rows(args.raw)
    contexts = args.dataset.read_text(encoding="utf-8").splitlines()
    if len(rows) != len(contexts):
        raise SystemExit(f"row mismatch: raw={len(rows)} dataset={len(contexts)}")
    scores = read_scores(args.scores)
    policy = None
    threshold = args.threshold
    if args.threshold_policy:
        policy = json.loads(args.threshold_policy.read_text(encoding="utf-8"))
        threshold = float(policy.get("default_threshold", 0.5))
    output, changes, counters = apply(
        rows, scores, threshold, args.missing_policy, args.max_triples_per_row, policy)
    postprocess_summary = None
    if args.profile:
        processor = OutputPostprocessor(
            config=OutputPostprocessorConfig(profile=args.profile)
        )
        output, postprocess_summary = processor.process_rows(output, contexts=contexts)

    iter_dir = args.output_dir / "iter0"
    iter_dir.mkdir(parents=True, exist_ok=True)
    (iter_dir / "canon_kg.txt").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output),
        encoding="utf-8")
    with (iter_dir / "entailment_verification_changes.jsonl").open("w", encoding="utf-8") as target:
        for change in changes:
            target.write(json.dumps(change, ensure_ascii=False) + "\n")
    summary = {
        "rows": len(rows),
        "total_triples_before": sum(len(row) for row in rows),
        "total_triples_after": sum(len(row) for row in output),
        "threshold": args.threshold,
        "threshold_policy": str(args.threshold_policy) if args.threshold_policy else None,
        "missing_policy": args.missing_policy,
        "scores_available": len(scores),
        "changes": len(changes),
        "counts": dict(counters),
        "postprocess_profile": args.profile,
    }
    (iter_dir / "entailment_verification_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if postprocess_summary is not None:
        write_postprocess_artifacts(str(args.output_dir), postprocess_summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
