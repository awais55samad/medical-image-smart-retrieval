"""Command-line interface: ``med-retrieval build|search``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .features import extract_features
from .index import RetrievalIndex
from .io import load_index_csv, load_pgm


def _build_index(csv_path: Path) -> RetrievalIndex:
    index = RetrievalIndex()
    for row in load_index_csv(csv_path):
        image = load_pgm(row["file"])
        index.add(row["id"], row["label"], extract_features(image))
    return index


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="med-retrieval",
        description="Content-based medical image retrieval (similarity search only).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="Build an index from an index CSV")
    build.add_argument("--csv", required=True)
    build.add_argument("--out", required=True)

    search = sub.add_parser("search", help="Search an index with a query image")
    search.add_argument("--index", required=True)
    search.add_argument("--query", required=True)
    search.add_argument("--top", type=int, default=3)

    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            index = _build_index(Path(args.csv))
            Path(args.out).write_text(index.to_json() + "\n", encoding="utf-8")
            print(f"built index with {len(index.entries)} image(s) -> {args.out}")
            return 0
        if args.command == "search":
            index = RetrievalIndex.from_json(Path(args.index).read_text(encoding="utf-8"))
            query = extract_features(load_pgm(args.query))
            for rank, (id_, label, score) in enumerate(index.search(query, top_k=args.top), start=1):
                print(f"{rank}. {id_} ({label}) — similarity {score:.3f}")
            return 0
    except (ValueError, OSError, json.JSONDecodeError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
