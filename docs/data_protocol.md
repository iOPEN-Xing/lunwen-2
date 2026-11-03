# Public Benchmark Data Protocol

## DIOR-RSVG

Use the verified public train/validation/test split. Convert annotations to the internal `bbox_xyxy` format and preserve original image dimensions. Use the benchmark's own evaluation definition for final results.

## OPT-RSVG

Use the official split and keep the text version explicit in every manifest. Do not mix reconstructed annotations with the official split without recording the conversion. Report text length and target scale separately.

## OptSAR-RSVG

Treat this as an optional extension. Before using it, record image availability, license, optical/SAR partition, annotation format, leakage checks, and the evaluation protocol.

## Internal records

All datasets should be converted to:

```json
{
  "sample_id": "...",
  "image_id": "...",
  "image_path": "...",
  "text": "...",
  "bbox_xyxy": [left, top, right, bottom],
  "category": "...",
  "split": "train|val|test",
  "dataset": "..."
}
```

Candidate exports should contain detector outputs only. Training-only fields such as IoU and `is_positive` must be generated in a separate diagnostic view.

