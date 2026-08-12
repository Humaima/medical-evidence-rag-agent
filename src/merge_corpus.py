"""
Merge all converted per-topic JSON files in data/converted/ into a single
corpus file usable by ingest.py.

Usage:
    python src/merge_corpus.py [--input-dir data/converted] [--out data/my_corpus.json]
"""

from __future__ import annotations
import argparse
import glob
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge converted corpus JSON files into one.")
    parser.add_argument("--input-dir", default="data/converted", help="Directory of *.json files to merge.")
    parser.add_argument("--out", default="data/my_corpus.json", help="Output path for merged corpus JSON.")
    args = parser.parse_args()

    docs = []
    seen_ids = set()
    for f in sorted(glob.glob(str(Path(args.input_dir) / "*.json"))):
        with open(f, "r", encoding="utf-8") as fh:
            file_docs = json.load(fh)
        for doc in file_docs:
            if doc["id"] in seen_ids:
                print(f"Skipping duplicate id '{doc['id']}' from {f}")
                continue
            seen_ids.add(doc["id"])
            docs.append(doc)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(docs, fh, indent=2, ensure_ascii=False)

    print(f"Merged {len(docs)} total documents into {out_path}")


if __name__ == "__main__":
    main()
