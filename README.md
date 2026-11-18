# Medical Image Smart Retrieval

A small content-based image retrieval prototype for medical-style
images, using hybrid feature mining, statistical modeling, and
similarity search.

This is an **original, independent implementation inspired by** the 2025
paper *“Unified computational intelligent framework for medical image
smart retrieval using hybrid feature mining, statistical modeling and
deep neural networks”*. It is **not** the official code from that
paper, and it does not claim to reproduce its results.

## What it does — and does not do

It does retrieval / similarity search only:

- Extracts features from a grayscale image: a normalised intensity
  histogram, statistical moments (mean, standard deviation, skewness,
  minimum, maximum, median), and gradient features.
- Builds a small searchable index of those feature vectors.
- Finds the most similar images to a query image by cosine similarity.

It does **not** diagnose, classify disease, or make any medical
decision. It is a teaching and prototyping implementation for
retrieval, not a clinical tool, and it must not be used for diagnosis
or treatment decisions.

The prototype uses only the Python standard library — no deep-learning
framework, no image library, and no network service are required.
Images are simple ASCII PGM (`P2`) files, which can be read and
written without extra dependencies.

## Quick start

```bash
python3 -m pip install -e .
python3 examples/demo.py
```

That builds an index from the six synthetic example images in
`examples/images/` and searches for the images most similar to
`query.pgm`.

### Command line

```bash
# Build an index from a CSV list of images
med-retrieval build --csv examples/index.csv --out index.json

# Search the index with a query image
med-retrieval search --index index.json --query examples/images/query.pgm --top 3
```

The index CSV has columns: `id,file,label`
- `id`: a short identifier for the image
- `file`: path to a PGM (`P2`) image file, relative to the CSV file
- `label`: a descriptive pattern label (for example `bright-center`);
  labels are descriptive only and are not diagnoses

## Project layout

```
src/med_retrieval/
  features.py  # histogram, statistical and gradient features
  index.py     # cosine similarity and the retrieval index
  io.py        # PGM loading and index-CSV loading
  cli.py       # Command-line interface
examples/
  images/      # six synthetic 8x8 grayscale PGM images + a query image
  index.csv
  demo.py
tests/
  test_retrieval.py
```

## Tests

```bash
python3 -m pytest -q
# or, without pytest:
python3 -m unittest discover -s tests -v
```

## Author

**Engr. Muhammad Awais Khan** — Bahauddin Zakariya University, Multan (Pakistan)

Google Scholar profile: https://scholar.google.com/citations?hl=en&user=c4UawvgAAAAJ

## Status

Experimental research code, version 0.1.0. It is a teaching and
prototyping implementation of content-based image retrieval, not a
production or clinical system, and it has not been validated on real
medical images.

## License

MIT — see `LICENSE`.
