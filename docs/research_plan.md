# Research Plan

## Working title

**Candidate-Relational Visual Grounding for Dense Remote Sensing Images**

## Problem statement

Dense remote-sensing images often contain multiple instances of the same category. A detector can produce a candidate set with high recall while the final grounding system still selects the wrong instance. The difficult cases contain compositional language such as direction, relative position, size comparison, reference-object relations, and ordinal expressions.

The paper studies candidate selection as a separate problem from proposal generation.

## Research questions

### RQ1: Is candidate recall already sufficient?

Report proposal recall@1/5/10/20 and compare it with final P@0.5, P@0.75, and mIoU.

### RQ2: Do candidate relations improve selection?

Compare candidate-independent scalar scoring, pairwise scoring, and relational listwise scoring under the same frozen candidates.

### RQ3: Which compositional failures remain?

Break down results by attribute, directional, relational, ordinal, long-expression, repeated-expression, crowding, and object scale.

### RQ4: Does the method transfer?

Evaluate DIOR-RSVG and OPT-RSVG in-domain and in both cross-dataset directions.

## Method boundary

The proposal generator is an experimental control, not the main contribution. The research contribution is limited to candidate representation, neighborhood/full-image context, candidate geometry, text-conditioned relations, and listwise selection.

The method progression is:

```text
fixed Top-K candidates
    -> scalar scorer
    -> pairwise scorer
    -> candidate graph / relational scorer
    -> text-conditioned listwise ranker
```

Only one new source of information or one new loss is introduced per ablation.

## Required reporting

- P@0.5 / Acc@0.5
- P@0.75 / Acc@0.75
- mIoU
- proposal recall@1/5/10/20
- unique versus repeated expressions
- same-class crowded scenes
- small/medium/large targets
- category-only, attribute, directional, relational, ordinal, and long-expression subsets
- in-domain and cross-dataset transfer

Candidate recall, oracle diagnostics, and final grounding accuracy must never be merged into one score.

## Failure taxonomy

Each failed query should be assigned one primary failure type:

1. Candidate miss.
2. Category confusion.
3. Attribute failure.
4. Direction or relative-position failure.
5. Reference-object failure.
6. Ordinal failure.
7. Repeated-expression ambiguity.
8. Candidate-score calibration failure.
9. Localization failure after correct instance selection.

The first analysis should use only queries for which the fixed candidate set contains a valid target. This separates selection errors from proposal errors.

## Execution order

1. Build a unified DIOR/OPT adapter and verify split and bbox conventions.
2. Export immutable Top-K candidate manifests.
3. Reproduce scalar reranking.
4. Construct pairwise hard negatives from same-image candidates.
5. Add geometry and neighborhood relations.
6. Add text relation features and listwise loss.
7. Run cross-dataset transfer and failure analysis.
8. Freeze the final configuration before test evaluation.

## Stopping criteria

Do not keep adding modules if the relation model only improves one validation split, fails to improve repeated/crowded subsets, does not transfer across DIOR/OPT, or is not better than a simpler pairwise model.

