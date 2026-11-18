"""Demo: index the synthetic example images and search with a query."""

from pathlib import Path

from med_retrieval import RetrievalIndex, extract_features, load_index_csv, load_pgm

ROOT = Path(__file__).resolve().parent


def main() -> None:
    index = RetrievalIndex()
    for row in load_index_csv(ROOT / "index.csv"):
        index.add(row["id"], row["label"], extract_features(load_pgm(row["file"])))
    print(f"indexed {len(index.entries)} image(s)")
    query = extract_features(load_pgm(ROOT / "images" / "query.pgm"))
    for rank, (id_, label, score) in enumerate(index.search(query, top_k=3), start=1):
        print(f"{rank}. {id_} ({label}) — similarity {score:.3f}")


if __name__ == "__main__":
    main()
