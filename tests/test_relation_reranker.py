import torch

from src.relation_reranker import CandidateRelationalRanker, ranking_loss
from src.reranker_common import relation_matrix


def test_relation_matrix_is_pairwise_and_finite():
    boxes = [[0, 0, 10, 10], [20, 0, 30, 10]]
    matrix = relation_matrix(boxes, 40, 40)
    assert matrix.shape == (2, 2, 7)
    assert torch.isfinite(matrix).all()
    assert torch.equal(matrix[0, 0], torch.zeros(7))
    assert matrix[0, 1, 0] < 0


def test_ranker_and_loss_have_gradients():
    model = CandidateRelationalRanker()
    features = torch.randn(3, 7)
    relations = torch.randn(3, 3, 7)
    ious = torch.tensor([0.8, 0.2, 0.0])
    scores = model(features, relations)
    loss = ranking_loss(scores, ious, pair_weight=0.25)
    loss.backward()
    assert scores.shape == (3,)
    assert torch.isfinite(loss)

