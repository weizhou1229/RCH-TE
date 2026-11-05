#!/usr/bin/env python3
"""Train a binary REBEL triple-entailment verifier."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from scripts.rebel.rebel_verifier_text import format_verifier_input


class EntailmentDataset(Dataset):
    def __init__(self, path: Path, tokenizer, max_length: int,
                 structured_input: bool = False, context_window_chars: int = 0):
        self.rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.structured_input = structured_input
        self.context_window_chars = context_window_chars

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        context, triple = format_verifier_input(
            row["context"], row["subject"], row["relation"], row["object"],
            self.structured_input, self.context_window_chars)
        encoded = self.tokenizer(
            context, text_pair=triple, max_length=self.max_length,
            padding="max_length", truncation=True, return_tensors="pt")
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(int(row["label"]), dtype=torch.long),
        }


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    tp = fp = fn = 0
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(**batch).logits
            predicted = logits.argmax(dim=-1)
            labels = batch["labels"]
            correct += int((predicted == labels).sum())
            total += labels.numel()
            tp += int(((predicted == 1) & (labels == 1)).sum())
            fp += int(((predicted == 1) & (labels == 0)).sum())
            fn += int(((predicted == 0) & (labels == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"accuracy": correct / total if total else 0.0, "precision": precision, "recall": recall, "f1": f1}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--model_name", default="bert-base-uncased")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--structured_input", action="store_true")
    parser.add_argument("--context_window_chars", type=int, default=0)
    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2).to(device)
    train_set = EntailmentDataset(args.data_dir / "train.jsonl", tokenizer, args.max_length,
                                  args.structured_input, args.context_window_chars)
    val_set = EntailmentDataset(args.data_dir / "val.jsonl", tokenizer, args.max_length,
                                args.structured_input, args.context_window_chars)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    best_f1 = -1.0
    history = []
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad()
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach())
        metrics = evaluate(model, val_loader, device)
        metrics.update({"epoch": epoch, "train_loss": loss_sum / max(1, len(train_loader))})
        history.append(metrics)
        print(json.dumps(metrics), flush=True)
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)

    (args.output_dir / "training_summary.json").write_text(
        json.dumps({"model_name": args.model_name, "device": str(device),
                    "train_samples": len(train_set), "val_samples": len(val_set),
                    "history": history, "best_val_f1": best_f1,
                    "structured_input": args.structured_input,
                    "context_window_chars": args.context_window_chars},
                   ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
