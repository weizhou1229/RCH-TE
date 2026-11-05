#!/usr/bin/env python3
"""Score every triple in a REBEL raw extraction with a trained verifier."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from scripts.rebel.rebel_verifier_text import format_verifier_input


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--model_dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--structured_input", action="store_true")
    parser.add_argument("--context_window_chars", type=int, default=0)
    args = parser.parse_args()

    rows = [ast.literal_eval(line) for line in args.raw.read_text(encoding="utf-8").splitlines()]
    contexts = args.dataset.read_text(encoding="utf-8").splitlines()
    if len(rows) != len(contexts):
        raise SystemExit(f"row mismatch: raw={len(rows)} dataset={len(contexts)}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    pending = []
    with args.output.open("w", encoding="utf-8") as target, torch.no_grad():
        for line_index, (row, context) in enumerate(zip(rows, contexts), 1):
            for triple_index, triple in enumerate(row):
                if len(triple) != 3:
                    continue
                pending.append((line_index, triple_index, context, triple))
                if len(pending) < args.batch_size:
                    continue
                _write_batch(pending, tokenizer, model, device, args.max_length, target,
                             args.structured_input, args.context_window_chars)
                pending.clear()
        if pending:
            _write_batch(pending, tokenizer, model, device, args.max_length, target,
                         args.structured_input, args.context_window_chars)


def _write_batch(pending, tokenizer, model, device, max_length, target,
                 structured_input=False, context_window_chars=0):
    contexts, pairs = [], []
    for _, _, context, triple in pending:
        formatted_context, formatted_triple = format_verifier_input(
            context, triple[0], triple[1], triple[2],
            structured_input, context_window_chars)
        contexts.append(formatted_context)
        pairs.append(formatted_triple)
    encoded = tokenizer(contexts, text_pair=pairs,
                        max_length=max_length, padding=True, truncation=True,
                        return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}
    probabilities = torch.softmax(model(**encoded).logits, dim=-1)[:, 1].tolist()
    for (line_index, triple_index, _context, triple), score in zip(pending, probabilities):
        target.write(json.dumps({"line": line_index, "triple_index": triple_index,
                                 "score": score, "triple": list(triple)},
                                ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
