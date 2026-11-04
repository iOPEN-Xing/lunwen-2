"""Shared data, features, and metrics for the paper reranker baselines."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import torch

from .metrics import box_iou


BASE_FEATURES = ("det_score", "cx", "cy", "w", "h", "area", "aspect")
RELATION_FEATURES = ("dx", "dy", "distance", "area_ratio", "width_ratio", "height_ratio", "overlap")


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def candidate_vector(candidate: dict, image_width: float, image_height: float) -> list[float]:
    x1, y1, x2, y2 = candidate["bbox_xyxy"]
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    image_area = max(1.0, image_width * image_height)
    return [
        float(candidate.get("det_score", 0.0)),
        (x1 + x2) / (2.0 * max(1.0, image_width)),
        (y1 + y2) / (2.0 * max(1.0, image_height)),
        width / max(1.0, image_width),
        height / max(1.0, image_height),
        width * height / image_area,
        width / max(height, 1e-6),
    ]


def relation_matrix(boxes: list[list[float]], image_width: float, image_height: float) -> torch.Tensor:
    rows = []
    centers = []
    areas = []
    widths = []
    heights = []
    for x1, y1, x2, y2 in boxes:
        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)
        centers.append(((x1 + x2) / (2.0 * max(1.0, image_width)), (y1 + y2) / (2.0 * max(1.0, image_height))))
        areas.append(width * height / max(1.0, image_width * image_height))
        widths.append(width / max(1.0, image_width))
        heights.append(height / max(1.0, image_height))
    for i, (cx_i, cy_i) in enumerate(centers):
        for j, (cx_j, cy_j) in enumerate(centers):
            if i == j:
                rows.append([0.0] * len(RELATION_FEATURES))
                continue
            dx, dy = cx_i - cx_j, cy_i - cy_j
            ratio = areas[i] / max(areas[j], 1e-6)
            rows.append([
                dx,
                dy,
                math.sqrt(dx * dx + dy * dy),
                math.log(max(ratio, 1e-6)),
                math.log(max(widths[i] / max(widths[j], 1e-6), 1e-6)),
                math.log(max(heights[i] / max(heights[j], 1e-6), 1e-6)),
                box_iou(boxes[i], boxes[j]),
            ])
    n = len(boxes)
    return torch.tensor(rows, dtype=torch.float32).reshape(n, n, len(RELATION_FEATURES))


def row_to_example(row: dict, positive_iou: float = 0.5) -> dict:
    candidates = row["candidates"]
    boxes = [c["bbox_xyxy"] for c in candidates]
    features = torch.tensor(
        [candidate_vector(c, row["width"], row["height"]) for c in candidates],
        dtype=torch.float32,
    )
    relations = relation_matrix(boxes, row["width"], row["height"])
    target = row["bbox_xyxy"]
    ious = torch.tensor([box_iou(box, target) for box in boxes], dtype=torch.float32)
    positives = ious >= positive_iou
    return {
        "sample_id": row.get("sample_id", row.get("qid", "")),
        "features": features,
        "relations": relations,
        "boxes": boxes,
        "ious": ious,
        "positive_mask": positives,
        "target": target,
        "metadata": row,
    }


def build_examples(rows: Iterable[dict], positive_iou: float = 0.5) -> list[dict]:
    examples = []
    for row in rows:
        example = row_to_example(row, positive_iou)
        if example["features"].numel() and example["positive_mask"].any():
            examples.append(example)
    return examples


def evaluate_scores(examples: list[dict], score_fn) -> dict:
    counts = {"n": 0, "acc50": 0, "acc75": 0, "mean_iou": 0.0, "proposal_recall": {}}
    for example in examples:
        with torch.no_grad():
            scores = score_fn(example)
        pick = int(scores.argmax().item())
        selected_iou = float(example["ious"][pick])
        counts["n"] += 1
        counts["acc50"] += selected_iou >= 0.5
        counts["acc75"] += selected_iou >= 0.75
        counts["mean_iou"] += selected_iou
        for k in (1, 5, 10, 20):
            counts["proposal_recall"].setdefault(str(k), 0)
            counts["proposal_recall"][str(k)] += bool((example["ious"][:k] >= 0.5).any())
    n = max(1, counts["n"])
    counts["acc50"] /= n
    counts["acc75"] /= n
    counts["mean_iou"] /= n
    counts["proposal_recall"] = {k: v / n for k, v in counts["proposal_recall"].items()}
    return counts

