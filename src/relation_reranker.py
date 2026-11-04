"""Candidate-relational extension of the paper's scalar ListwiseMLP baseline."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from .reranker_common import build_examples, evaluate_scores, load_jsonl


class CandidateRelationalRanker(nn.Module):
    def __init__(self, node_dim: int = 7, relation_dim: int = 7, hidden_dim: int = 64):
        super().__init__()
        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 2 + relation_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU()
        )
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, features: torch.Tensor, relations: torch.Tensor) -> torch.Tensor:
        nodes = self.node_encoder(features)
        n = nodes.shape[0]
        left = nodes[:, None, :].expand(n, n, -1)
        right = nodes[None, :, :].expand(n, n, -1)
        edges = self.edge_encoder(torch.cat([left, right, relations], dim=-1))
        neighbor = edges.sum(dim=1) / max(1, n - 1)
        return self.score_head(torch.cat([nodes, neighbor], dim=-1)).squeeze(-1)


def ranking_loss(
    scores: torch.Tensor,
    ious: torch.Tensor,
    pair_weight: float = 0.0,
    counterfactual_pair: tuple[int, int] | None = None,
) -> torch.Tensor:
    positive = int(ious.argmax().item())
    target = torch.tensor([positive], device=scores.device)
    listwise = F.cross_entropy(scores.unsqueeze(0), target)
    soft_target = (ious.clamp_min(0.0) ** 2) + 1e-6
    soft_target = soft_target / soft_target.sum()
    soft_iou = -(soft_target * F.log_softmax(scores, dim=0)).sum()
    pairwise = scores.new_zeros(())
    if pair_weight > 0:
        if counterfactual_pair is not None:
            positive_index, negative_index = counterfactual_pair
            pairwise = F.softplus(scores[negative_index] - scores[positive_index])
        else:
            negative = ious < 0.5
            if negative.any():
                pairwise = F.softplus(scores[negative] - scores[positive]).mean()
    return listwise + 0.3 * soft_iou + pair_weight * pairwise


def fit(train_examples: list[dict], dev_examples: list[dict], seed: int, epochs: int, lr: float, pair_weight: float):
    random.seed(seed)
    torch.manual_seed(seed)
    model = CandidateRelationalRanker()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best_state = None
    best_value = -1.0
    for _ in range(epochs):
        model.train()
        random.shuffle(train_examples)
        for example in train_examples:
            scores = model(example["features"], example["relations"])
            loss = ranking_loss(
                scores,
                example["ious"],
                pair_weight,
                example.get("counterfactual_pair"),
            )
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        model.eval()
        metrics = evaluate_scores(dev_examples, lambda item: model(item["features"], item["relations"]))
        if metrics["acc50"] > best_value:
            best_value = metrics["acc50"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--test", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--pair-weight", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    train_examples = build_examples(load_jsonl(args.train))
    dev_examples = build_examples(load_jsonl(args.dev))
    model = fit(train_examples, dev_examples, args.seed, args.epochs, args.learning_rate, args.pair_weight)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "seed": args.seed}, args.output)
    metrics = {"dev": evaluate_scores(dev_examples, lambda item: model(item["features"], item["relations"]))}
    if args.test:
        test_examples = build_examples(load_jsonl(args.test))
        metrics["test"] = evaluate_scores(test_examples, lambda item: model(item["features"], item["relations"]))
    args.output.with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

