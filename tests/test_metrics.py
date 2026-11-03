from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.metrics import box_iou, proposal_recall


def test_box_iou_for_identical_boxes():
    assert box_iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0


def test_proposal_recall_uses_top_k():
    candidates = [[0, 0, 1, 1], [10, 10, 20, 20]]
    target = [10, 10, 20, 20]
    assert proposal_recall(candidates, target, top_k=1, threshold=0.5) == 0.0
    assert proposal_recall(candidates, target, top_k=2, threshold=0.5) == 1.0

