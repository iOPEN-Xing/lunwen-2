"""Build strict DIOR-RSVG query/candidate manifests from public annotations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .reranker_common import image_split_map, load_dior_rsvg_objects, load_jsonl, synthetic_pair_records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation-dir", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--synthetic", type=Path, required=True)
    parser.add_argument("--detector-candidates", type=Path, help="JSONL with image_id, target/index, and detector candidates")
    parser.add_argument("--allow-gt-candidates", action="store_true", help="Mark output as diagnostic only; uses all annotated objects as candidates")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    objects = load_dior_rsvg_objects(args.annotation_dir)
    split_map = image_split_map(args.split_dir)
    records = synthetic_pair_records(args.synthetic, split_map, {"train", "val", "test"})
    detector_rows = {}
    if args.detector_candidates:
        for row in load_jsonl(args.detector_candidates):
            key = (row.get("image_id", row.get("image")), row.get("target", row.get("target_index", 0)))
            detector_rows[key] = row
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {"records": len(records), "splits": {}}
    handles = {}
    try:
        for split in ("train", "val", "test"):
            handles[split] = (args.output_dir / f"dior_cf_{split}.jsonl").open("w", encoding="utf-8")
        for record in records:
            info = objects.get(record["image"])
            if info is None or not (0 <= record["target"] < len(info["objects"])):
                continue
            target = info["objects"][record["target"]]
            candidate_source = "ground_truth_objects"
            diagnostic_only = True
            candidates = []
            detector_row = detector_rows.get((record["image"], record["target"]))
            if detector_row is not None:
                candidates = detector_row.get("candidates", [])
                candidate_source = "detector"
                diagnostic_only = False
            elif args.allow_gt_candidates:
                candidates = [
                    {"candidate_id": obj["target_index"], "bbox_xyxy": obj["bbox_xyxy"], "category": obj["category"], "det_score": 0.0}
                    for obj in info["objects"]
                ]
            else:
                raise SystemExit(
                    "No detector candidate row found. Provide --detector-candidates, or pass "
                    "--allow-gt-candidates only for GT-object diagnostics."
                )
            output = {
                "sample_id": f"dior-cf:{record['image']}:{record['target']}:{record['kind']}:{record['positive_text']}:{record['negative_text']}",
                "image_id": record["image"],
                "image_path": str(args.image_dir / record["image"]),
                "width": info["width"],
                "height": info["height"],
                "text": record["positive_text"],
                "bbox_xyxy": target["bbox_xyxy"],
                "category": target["category"],
                "candidates": candidates,
                "counterfactual": {
                    "negative_text": record["negative_text"],
                    "error_type": record["error_type"],
                    "kind": record["kind"],
                    "changed_component": record.get("changed_component"),
                    "hardness": "hard" if record.get("hits_candidate") is not None else "soft",
                },
                "split": record["split"],
                "candidate_source": candidate_source,
                "diagnostic_only": diagnostic_only,
            }
            handles[record["split"]].write(json.dumps(output, ensure_ascii=False) + "\n")
            summary["splits"][record["split"]] = summary["splits"].get(record["split"], 0) + 1
    finally:
        for handle in handles.values():
            handle.close()
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

