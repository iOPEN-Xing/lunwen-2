# DIOR Counterfactual Training Protocol

The relational ranker is an extension of the existing scalar ListwiseMLP baseline. It must be trained in two clearly separated stages.

The synthetic file `dataset/DIOR-RSVG/data_new/data.jsonl` is a counterfactual text-pair file, not a candidate manifest. It has 300 rows, lacks candidate boxes and split labels, and contains positive-only rows as well as negative pairs. It must be joined with the public XML annotations by image id and target index. This join is for diagnosis and supervision metadata; it does not replace detector-generated candidates.

`src/prepare_dior_manifest.py` accepts a detector candidate JSONL through `--detector-candidates`. Without it, the command fails unless `--allow-gt-candidates` is explicitly set for a diagnostic-only manifest.

## Stage A: real-candidate baseline

Use a frozen candidate manifest exported by the detector. Each query contains a target box only for offline training/evaluation and a list of detector candidates. The model sees candidate features and pairwise geometry; inference never reads IoU or target boxes.

Do not use all ground-truth objects as detector candidates in the main grounding result. Such a manifest may be generated only with an explicit diagnostic flag and must be reported as `GT-object diagnostic`, never as detector reranking.

Example preparation command:

```bash
python -m src.prepare_dior_manifest \
  --annotation-dir /path/to/DIOR-RSVG/Annotations \
  --split-dir /path/to/DIOR-RSVG/split \
  --image-dir /path/to/DIOR-RSVG/JPEGImages \
  --synthetic /path/to/DIOR-RSVG/data_new/data.jsonl \
  --detector-candidates /path/to/dior_detector_candidates.jsonl \
  --output-dir /path/to/dior_manifests
```

The detector candidate rows must use `image_id`, `target` or `target_index`, and a `candidates` list containing `bbox_xyxy` and `det_score`. Candidate generation is a separate frozen stage and must be evaluated before reranker training.

The default objective is:

```text
L_base = CE(s, argmax_i IoU(c_i, target))
       + 0.3 * CE_soft(s, normalize(IoU(c_i, target)^2 + eps))
```

This is the objective inherited from the scalar baseline. The candidate-relational model changes the scorer, not the supervision protocol.

## Stage B: audited counterfactual supervision

DIOR counterfactual samples are added only after Stage A is reproducible. An audited T1/T2 record must identify a positive text, a changed component, and an explicit negative text. Ambiguous records are excluded from strict single-target training.

For a valid counterfactual pair, add:

```text
L_cf = softplus(s_negative - s_positive)
L = L_base + lambda_cf * L_cf
```

Run four controlled conditions:

```text
real only
real + T1 attribute
real + T2 relation
real + T1 + T2
```

T3 is a diagnostic/stress condition before it is used for main training because it changes two semantic components at once.

## Evaluation separation

Report three independent result blocks:

1. Candidate recall@1/5/10/20.
2. Real DIOR-RSVG grounding: P@0.5, P@0.75, mIoU and difficulty subsets.
3. Counterfactual discrimination: T1, T2, T3, hard, ambiguous and strict-unique subsets.

Counterfactual accuracy must never replace real benchmark grounding accuracy.

