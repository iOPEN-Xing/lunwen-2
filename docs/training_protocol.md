# DIOR Counterfactual Training Protocol

The relational ranker is an extension of the existing scalar ListwiseMLP baseline. It must be trained in two clearly separated stages.

## Stage A: real-candidate baseline

Use a frozen candidate manifest exported by the detector. Each query contains a target box only for offline training/evaluation and a list of detector candidates. The model sees candidate features and pairwise geometry; inference never reads IoU or target boxes.

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

