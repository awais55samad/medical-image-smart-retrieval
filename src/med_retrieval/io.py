"""Loading for ASCII PGM (P2) images and index CSV files."""

from __future__ import annotations

import csv
from pathlib import Path


def load_pgm(path: str | Path) -> list[list[float]]:
    """Load an ASCII PGM (``P2``) grayscale image as a 2D pixel list."""
    tokens: list[str] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.split("#", 1)[0].strip()
            if line:
                tokens.extend(line.split())
    if len(tokens) < 4 or tokens[0] != "P2":
        raise ValueError("only ASCII PGM (P2) files are supported")
    try:
        width = int(tokens[1])
        height = int(tokens[2])
        max_value = int(tokens[3])
        values = [float(t) for t in tokens[4:]]
    except ValueError as exc:
        raise ValueError(f"invalid PGM file {path}: {exc}") from exc
    if max_value <= 0:
        raise ValueError("PGM max value must be positive")
    if len(values) != width * height:
        raise ValueError(
            f"PGM {path}: expected {width * height} pixels, got {len(values)}"
        )
    scale = 255.0 / max_value
    rows: list[list[float]] = []
    for r in range(height):
        rows.append([values[r * width + c] * scale for c in range(width)])
    return rows


def write_pgm(path: str | Path, image: list[list[float]]) -> None:
    """Write a 2D pixel list as an ASCII PGM (``P2``) file."""
    height = len(image)
    width = len(image[0]) if height else 0
    lines = ["P2", f"{width} {height}", "255"]
    for row in image:
        lines.append(" ".join(str(int(min(255, max(0, round(v))))) for v in row))
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_index_csv(path: str | Path) -> list[dict]:
    """Load an index CSV with columns ``id,file,label``.

    File paths are resolved relative to the CSV file's directory.
    """
    csv_path = Path(path)
    rows: list[dict] = []
    with open(csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"id", "file", "label"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"CSV must have columns: {sorted(required)}")
        for line_no, row in enumerate(reader, start=2):
            try:
                rows.append(
                    {
                        "id": row["id"].strip(),
                        "file": str(csv_path.parent / row["file"].strip()),
                        "label": row["label"].strip(),
                    }
                )
            except (KeyError, AttributeError) as exc:
                raise ValueError(f"CSV line {line_no}: {exc}") from exc
    if not rows:
        raise ValueError("CSV contains no images")
    return rows
