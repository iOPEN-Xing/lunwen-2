# Baseline Reproduction Protocol

## Baseline hierarchy

The paper uses two related baselines and reports them separately:

```text
B0: OTA-Det-L detector / grounding model, top-1 output
B1: B0 Top-K candidate cache + candidate-independent scalar ListwiseMLP
```

`B0` measures proposal generation and the detector's native ranking. `B1` is the main reranking baseline for the paper. The proposed method must use the same candidate cache and feature manifest as `B1`; only the candidate relation module and training supervision may change.

The historical challenge score and challenge-only checkpoints are provenance evidence only. They are not the primary paper result.

## Public benchmark matrix

Run the following after the detector configuration and candidate cache are frozen:

| ID | Train | Validation | Test | Purpose |
|---|---|---|---|---|
| B0-D | DIOR-RSVG train | DIOR-RSVG val | DIOR-RSVG test | detector in-domain |
| B0-O | OPT-RSVG train | OPT-RSVG val | OPT-RSVG test | detector in-domain |
| B1-D | frozen DIOR candidates | DIOR val | DIOR test | scalar reranker |
| B1-O | frozen OPT candidates | OPT val | OPT test | scalar reranker |
| B1-DO | DIOR candidates | DIOR val | OPT test | cross-dataset transfer |
| B1-OD | OPT candidates | OPT val | DIOR test | cross-dataset transfer |

Each configuration uses at least three seeds. The test split is evaluated only after model selection on validation data.

## Required outputs

Every run writes a self-contained directory containing:

```text
config.yaml
command.txt
environment.txt
candidate_manifest.json
feature_manifest.json
metrics.json
checkpoint.pt
```

The metrics must include proposal recall@1/5/10/20, P@0.5, P@0.75, mIoU, and the available difficulty buckets.

## Current server status

The local workspace contains DIOR-RSVG, OPT-RSVG, the OTA-Det-L checkpoint, and historical reranking outputs. At the time this protocol was prepared, GPUs 0, 1, 3, and 4 were occupied by an unrelated four-GPU UNIPR job. Do not terminate that job; start the reproduction after those devices are released.

The existing OTA-Mix YAML references processed data under `../datasets/OTA-Det/`. The local public data under `dataset/` must first be converted or symlinked into the exact HF dataset layout expected by the config. The conversion manifest and source checksums must be saved with the run.

