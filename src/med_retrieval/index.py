"""Cosine-similarity retrieval index over feature vectors."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity in [-1, 1]; 1 means identical direction."""
    if len(a) != len(b):
        raise ValueError("vectors must have the same length")
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return max(-1.0, min(1.0, dot / (norm_a * norm_b)))


@dataclass
class Entry:
    id: str
    label: str
    features: list[float]


class RetrievalIndex:
    """A small in-memory index of feature vectors."""

    def __init__(self) -> None:
        self.entries: list[Entry] = []

    def add(self, id: str, label: str, features: list[float]) -> None:
        if not id.strip():
            raise ValueError("id must not be empty")
        if any(e.id == id for e in self.entries):
            raise ValueError(f"duplicate id: {id}")
        self.entries.append(Entry(id=id, label=label, features=list(features)))

    def search(self, query: list[float], top_k: int = 3) -> list[tuple[str, str, float]]:
        """Return ``(id, label, score)`` for the top_k most similar entries."""
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if not self.entries:
            return []
        ranked = sorted(
            ((e.id, e.label, cosine_similarity(query, e.features)) for e in self.entries),
            key=lambda item: item[2],
            reverse=True,
        )
        return ranked[:top_k]

    def to_dict(self) -> dict:
        return {
            "version": 1,
            "entries": [
                {"id": e.id, "label": e.label, "features": e.features}
                for e in self.entries
            ],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    @classmethod
    def from_dict(cls, data: dict) -> "RetrievalIndex":
        index = cls()
        for item in data["entries"]:
            index.add(item["id"], item["label"], item["features"])
        return index

    @classmethod
    def from_json(cls, text: str) -> "RetrievalIndex":
        return cls.from_dict(json.loads(text))
