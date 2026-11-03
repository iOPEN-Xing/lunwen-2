"""Metrics that are independent of any detector or dataset implementation."""
from __future__ import annotations

from collections.abc import Iterable, Sequence


def box_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def proposal_recall(
    candidates: Iterable[Sequence[float]],
    target: Sequence[float],
    top_k: int,
    threshold: float,
) -> float:
    selected = list(candidates)[:top_k]
    return float(any(box_iou(candidate, target) >= threshold for candidate in selected))

