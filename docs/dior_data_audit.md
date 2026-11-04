# DIOR Data Audit

The current file `dataset/DIOR-RSVG/data_new/data.jsonl` contains 300 counterfactual records:

- 241 records have `negative_text` and 59 records are positive-only;
- records do not contain bounding boxes, detector candidates, or split labels;
- the target is represented by `image` and zero-based `target` index;
- `unique_status` must be filtered before strict pairwise training;
- image splits must be joined from the official DIOR object-index split.

After joining the current file with the local public XML annotations and official split, 124 strict records are available: 16 train, 2 validation, and 106 test. This distribution is not suitable as a standalone train/validation/test dataset. The current file should therefore be used as auxiliary supervision or a diagnostic subset until a larger image-balanced DIOR counterfactual release is prepared.

The 124-row join also reveals why all annotated objects cannot be silently treated as proposals: that would be a GT-object oracle diagnostic, not detector candidate reranking. Main experiments must use frozen detector candidate manifests with real detector scores.

