"""Hybrid feature extraction for grayscale images.

An image is a 2D list of pixel values (0-255). The feature vector
combines three kinds of features, hence "hybrid feature mining":

- histogram features: a normalised intensity histogram (8 bins);
- statistical features: mean, standard deviation, skewness, minimum,
  maximum and median of the pixel values;
- gradient features: mean absolute horizontal and vertical differences.

The vector has 8 + 6 + 2 = 16 components. It is a teaching prototype:
it captures brightness, contrast and texture at a coarse level, and
is not a deep neural network.
"""

from __future__ import annotations

import math


def _flatten(image: list[list[float]]) -> list[float]:
    if not image or not image[0]:
        raise ValueError("image must not be empty")
    width = len(image[0])
    pixels: list[float] = []
    for row in image:
        if len(row) != width:
            raise ValueError("image rows must all have the same length")
        for value in row:
            if value != value:  # NaN check
                raise ValueError("pixel value must not be NaN")
            pixels.append(float(value))
    return pixels


def histogram_features(pixels: list[float], bins: int = 8) -> list[float]:
    """Normalised intensity histogram; components sum to 1."""
    counts = [0.0] * bins
    for value in pixels:
        clamped = min(255.0, max(0.0, value))
        index = min(bins - 1, int(clamped / 256.0 * bins))
        counts[index] += 1.0
    total = float(len(pixels))
    return [c / total for c in counts]


def statistical_features(pixels: list[float]) -> list[float]:
    """Mean, std, skewness, min, max, median — each scaled to ~0-1."""
    n = len(pixels)
    mean = sum(pixels) / n
    variance = sum((v - mean) ** 2 for v in pixels) / n
    std = math.sqrt(variance)
    if std > 0:
        skewness = sum(((v - mean) / std) ** 3 for v in pixels) / n
    else:
        skewness = 0.0
    ordered = sorted(pixels)
    median = ordered[n // 2] if n % 2 == 1 else (ordered[n // 2 - 1] + ordered[n // 2]) / 2.0
    return [
        mean / 255.0,
        std / 255.0,
        max(-1.0, min(1.0, skewness / 2.0)),
        min(pixels) / 255.0,
        max(pixels) / 255.0,
        median / 255.0,
    ]


def gradient_features(image: list[list[float]]) -> list[float]:
    """Mean absolute horizontal and vertical differences, scaled 0-1."""
    rows = len(image)
    cols = len(image[0])
    h_diffs: list[float] = []
    v_diffs: list[float] = []
    for r in range(rows):
        for c in range(cols):
            if c + 1 < cols:
                h_diffs.append(abs(image[r][c + 1] - image[r][c]))
            if r + 1 < rows:
                v_diffs.append(abs(image[r + 1][c] - image[r][c]))
    h_mean = sum(h_diffs) / len(h_diffs) if h_diffs else 0.0
    v_mean = sum(v_diffs) / len(v_diffs) if v_diffs else 0.0
    return [h_mean / 255.0, v_mean / 255.0]


def extract_features(image: list[list[float]]) -> list[float]:
    """Extract the 16-component hybrid feature vector for an image."""
    pixels = _flatten(image)
    return (
        histogram_features(pixels)
        + statistical_features(pixels)
        + gradient_features(image)
    )
