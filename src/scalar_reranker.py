"""A compact candidate-independent baseline for frozen proposal sets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .metrics import box_iou


def candidate_features(candidate: dict, image_width: float, image_height: float) -> list[float]:
    x1, y1, x2, y2 = candidate["bbox_xyxy"]
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    area = width * height
    image_area = max(1.0, image_width * image_height)
    return [
        float(candidate.get("det_score", 0.0)),
        (x1 + x2) / (2.0 * max(1.0, image_width)),
        (y1 + y2) / (2.0 * max(1.0, image_height)),
        width / max(1.0, image_width),
        height / max(1.0, image_height),
        area / image_area,
        width / max(height, 1e-6),
    ]


class CandidateDataset(Dataset):
    def __init__(self, rows: list[dict], positive_iou: float = 0.5):
        self.examples = []
        for row in rows:
            target = row["bbox_xyxy"]
            candidates = row["candidates"]
            features = torch.tensor(
                [candidate_features(c, row["width"], row["height"]) for c in candidates],
                dtype=torch.float32,
            )
            labels = torch.tensor(
                [box_iou(c["bbox_xyxy"], target) >= positive_iou for c in candidates],
                dtype=torch.bool,
            )
            if features.numel() and labels.any():
                self.examples.append((features, labels))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int):
        return self.examples[index]


class ScalarReranker(nn.Module):
    def __init__(self, input_dim: int = 7, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def train(rows: list[dict], epochs: int, learning_rate: float, seed: int) -> ScalarReranker:
    torch.manual_seed(seed)
    dataset = CandidateDataset(rows)
    model = ScalarReranker()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fn = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        for features, labels in DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=lambda batch: batch[0]):
            scores = model(features)
            loss = loss_fn(scores, labels.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True, help="JSONL query/candidate manifest")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    model = train(load_rows(args.train), args.epochs, args.learning_rate, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "seed": args.seed}, args.output)


if __name__ == "__main__":
    main()

