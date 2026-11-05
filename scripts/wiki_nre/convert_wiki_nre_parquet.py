#!/usr/bin/env python3
"""Convert Hugging Face Wiki-NRE Parquet splits into auditable EDC files."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple


SPLIT_LAYOUT = {
    "train": {
        "source": "train-00000-of-00001.parquet",
        "stem": "wiki-nre_official_train",
        "analysis": "train",
    },
    "val": {
        "source": "val-00000-of-00001.parquet",
        "stem": "wiki-nre_official_val",
        "analysis": "val",
    },
    "test": {
        "source": "test-00000-of-00001.parquet",
        "stem": "wiki-nre_official_test_full",
        "analysis": "test_full",
    },
    "stratified_test_1K": {
        "source": "stratified_test_1K-00000-of-00001.parquet",
        "stem": "wiki-nre_official_test_1000",
        "analysis": "test_1000",
    },
}


def normalize_line(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def convert_row(row: Dict, source_split: str, source_row: int) -> Tuple[str, List[List[str]], Dict]:
    text = normalize_line(row.get("text"))
    if not text:
        raise ValueError("empty text")
    row_id = row.get("id")
    if row_id is None:
        raise ValueError("missing id")
    source_triplets = row.get("triplets")
    if not isinstance(source_triplets, list):
        raise ValueError("triplets is not a list")

    triples: List[List[str]] = []
    relation_uris = []
    entity_uris = []
    for triple_index, triple in enumerate(source_triplets):
        if not isinstance(triple, dict):
            raise ValueError(f"triplet {triple_index} is not a struct")
        subject = triple.get("subject")
        predicate = triple.get("predicate")
        obj = triple.get("object")
        if not all(isinstance(part, dict) for part in (subject, predicate, obj)):
            raise ValueError(f"triplet {triple_index} has invalid parts")
        subject_surface = normalize_line(subject.get("surfaceform"))
        relation_surface = normalize_line(predicate.get("surfaceform"))
        object_surface = normalize_line(obj.get("surfaceform"))
        if not subject_surface or not relation_surface or not object_surface:
            raise ValueError(f"triplet {triple_index} has an empty surface")
        triples.append([subject_surface, relation_surface, object_surface])
        relation_uri = normalize_line(predicate.get("uri"))
        if relation_uri:
            relation_uris.append(relation_uri)
        for entity in (subject, obj):
            entity_uri = normalize_line(entity.get("uri"))
            if entity_uri:
                entity_uris.append(entity_uri)

    triple_payload = json.dumps(triples, ensure_ascii=False, separators=(",", ":"))
    manifest = {
        "index": source_row,
        "source_split": source_split,
        "source_row": source_row,
        "id": int(row_id),
        "text_sha256": sha256_text(text),
        "triples_sha256": sha256_text(triple_payload),
        "triple_count": len(triples),
        "relation_uris": relation_uris,
        "entity_uris": entity_uris,
    }
    return text, triples, manifest


def iter_parquet_rows(path: Path, batch_size: int) -> Iterator[Dict]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit(
            "pyarrow is required to read Parquet; install it in the selected Python environment"
        ) from exc

    parquet = pq.ParquetFile(path)
    columns = ["text", "id", "triplets"]
    for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
        yield from batch.to_pylist()


def output_paths(root: Path, split: str) -> Dict[str, Path]:
    layout = SPLIT_LAYOUT[split]
    analysis = root / "analysis" / "wiki_nre_official" / layout["analysis"]
    return {
        "dataset": root / "datasets" / f"{layout['stem']}.txt",
        "reference": root / "evaluate" / "references" / f"{layout['stem']}.txt",
        "manifest": analysis / "manifest.jsonl",
        "summary": analysis / "summary.json",
    }


def ensure_available(paths: Iterable[Path], force: bool) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing and not force:
        raise FileExistsError(
            "Refusing to overwrite existing outputs; use --force: " + ", ".join(existing)
        )
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)


def temp_path(path: Path) -> Path:
    return path.with_name(path.name + f".tmp.{os.getpid()}")


def convert_split(
    parquet_dir: Path,
    root: Path,
    split: str,
    batch_size: int,
    progress_every: int,
    force: bool,
) -> Dict:
    source = parquet_dir / SPLIT_LAYOUT[split]["source"]
    if not source.is_file() or source.stat().st_size == 0:
        raise FileNotFoundError(source)
    paths = output_paths(root, split)
    ensure_available(paths.values(), force)
    temporary = {name: temp_path(path) for name, path in paths.items()}
    for path in temporary.values():
        if path.exists():
            path.unlink()

    counters = Counter()
    relation_counts = Counter()
    relation_surface_counts = Counter()
    entity_uris = set()
    try:
        with temporary["dataset"].open("w", encoding="utf-8", newline="\n") as dataset_file, \
             temporary["reference"].open("w", encoding="utf-8", newline="\n") as reference_file, \
             temporary["manifest"].open("w", encoding="utf-8", newline="\n") as manifest_file:
            for source_row, row in enumerate(iter_parquet_rows(source, batch_size)):
                text, triples, manifest = convert_row(row, split, source_row)
                dataset_file.write(text + "\n")
                reference_file.write(repr(triples) + "\n")
                manifest_file.write(json.dumps(manifest, ensure_ascii=False) + "\n")
                counters["records"] += 1
                counters["triples"] += len(triples)
                counters["records_without_triples"] += int(not triples)
                relation_counts.update(manifest["relation_uris"])
                relation_surface_counts.update(triple[1] for triple in triples)
                entity_uris.update(manifest["entity_uris"])
                if progress_every and counters["records"] % progress_every == 0:
                    print(f"{split}: converted {counters['records']:,} rows", flush=True)

        summary = {
            "protocol": f"wiki_nre_hf_{split}",
            "source": {
                "split": split,
                "parquet": str(source),
                "file_size": source.stat().st_size,
                "sha256": sha256_file(source),
            },
            "counts": {
                "records": counters["records"],
                "triples": counters["triples"],
                "records_without_triples": counters["records_without_triples"],
                "unique_relation_uris": len(relation_counts),
                "unique_relation_surfaces": len(relation_surface_counts),
                "unique_entity_uris": len(entity_uris),
            },
            "relation_uri_counts": dict(relation_counts.most_common()),
            "relation_surface_counts": dict(relation_surface_counts.most_common()),
            "outputs": {
                name: {
                    "path": str(paths[name]),
                    "sha256": sha256_file(temporary[name]),
                }
                for name in ("dataset", "reference", "manifest")
            },
        }
        temporary["summary"].write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        for name in ("dataset", "reference", "manifest", "summary"):
            temporary[name].replace(paths[name])
    except Exception:
        for path in temporary.values():
            if path.exists():
                path.unlink()
        raise

    print(
        f"{split}: records={counters['records']:,}, triples={counters['triples']:,}, "
        f"relations={len(relation_counts):,}"
    )
    print(f"Summary written to {paths['summary']}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet_dir", required=True, type=Path)
    parser.add_argument("--output_root", type=Path, default=Path("."))
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=list(SPLIT_LAYOUT),
        default=list(SPLIT_LAYOUT),
    )
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--progress_every", type=int, default=25000)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise SystemExit("--batch_size must be positive")
    for split in args.splits:
        convert_split(
            parquet_dir=args.parquet_dir,
            root=args.output_root,
            split=split,
            batch_size=args.batch_size,
            progress_every=args.progress_every,
            force=args.force,
        )


if __name__ == "__main__":
    main()
