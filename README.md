# Candidate Selection in Dense Remote-Sensing Visual Grounding

Research code and experiment records for the paper project:

> **Candidate-Relational Visual Grounding for Dense Remote Sensing Images**

The project studies a narrow problem: after a grounding detector produces a candidate set that often contains the target, how can a model select the correct instance when several same-class objects are present and the text uses attributes, direction, relative position, or ordinal expressions?

## Scope

The repository focuses on public benchmark experiments:

- DIOR-RSVG
- OPT-RSVG
- optional cross-sensor evaluation on OptSAR-RSVG

The experimental pipeline is deliberately two-stage:

```text
image + text
    -> fixed Top-K proposal generator
    -> candidate and context features
    -> candidate-independent / pairwise / listwise scorer
    -> final box
```

The paper contribution is the second stage. Proposal recall and final grounding accuracy are always reported separately.

## Repository layout

```text
docs/       research questions, data protocol, experiment log, release notes
src/        dataset adapters and candidate-ranking implementations
configs/    benchmark-specific experiment configurations
scripts/    reproducible experiment entry points
tests/      lightweight protocol and metric tests
```

## Data and checkpoints

Datasets and model checkpoints are not committed. Download them from their official sources or use local paths configured by each experiment. Candidate caches and generated outputs should remain outside Git.

The expected query record is:

```json
{
  "sample_id": "...",
  "image_id": "...",
  "image_path": "...",
  "text": "...",
  "bbox_xyxy": [0, 0, 1, 1],
  "category": "...",
  "split": "train|val|test",
  "dataset": "DIOR-RSVG"
}
```

The expected candidate record is:

```json
{
  "sample_id": "...",
  "candidate_id": 0,
  "bbox_xyxy": [0, 0, 1, 1],
  "det_score": 0.0,
  "category": "..."
}
```

Ground-truth IoU and positive labels may be generated for training or offline diagnosis, but must never be read by the model at inference time.

## Research order

1. Verify DIOR-RSVG and OPT-RSVG splits, coordinates, and metrics.
2. Export one immutable Top-K candidate cache per detector.
3. Measure candidate recall before training any reranker.
4. Reproduce a candidate-independent scalar scorer.
5. Add pairwise candidate geometry and hard-negative sampling.
6. Add listwise candidate relations and text-conditioned ranking.
7. Report failure types, crowded-scene subsets, and cross-dataset transfer.

See `docs/research_plan.md` for the current execution plan and stopping criteria.

The baseline is defined in `docs/baseline_reproduction.md`: public DIOR-RSVG/OPT-RSVG OTA-Det-L results followed by the fixed-candidate scalar reranker.

## Reproducibility

Every reported experiment must record the dataset split, candidate generator, K, feature manifest, random seed, checkpoint identifier, code revision, and evaluation command. Test sets are evaluated only after the configuration is frozen.

