"""Check the candidate losses and rank computation without any models or real data."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from team.train_rank import TextBank, candidate_loss, correct_ranks


def test_correct_ranks_match_a_stable_descending_sort():
    scores = torch.tensor([[0.1, 0.9, 0.5, 0.2, 0.0],
                           [0.5, 0.5, 0.5, 0.5, 0.5],
                           [0.5, 0.5, 0.5, 0.5, 0.5]])
    correct = torch.tensor([2, 0, 3])
    expected = []
    for row, column in zip(scores.tolist(), correct.tolist()):
        order = sorted(range(5), key=lambda index: -row[index])
        expected.append(order.index(column) + 1)
    assert correct_ranks(scores, correct).tolist() == expected == [2, 1, 4]


@pytest.mark.parametrize("loss", ["listwise", "bce"])
def test_losses_fall_when_the_correct_column_wins(loss):
    winning = torch.tensor([[5.0, -5.0, -5.0, -5.0, -5.0]])
    losing = torch.tensor([[-5.0, 5.0, -5.0, -5.0, -5.0]])
    assert candidate_loss(winning, loss) < candidate_loss(losing, loss)


def test_listwise_loss_is_cross_entropy_on_column_zero():
    logits = torch.tensor([[2.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    expected = -torch.log_softmax(logits, dim=1)[:, 0].mean()
    assert torch.isclose(candidate_loss(logits, "listwise"), expected)


def test_bce_weights_the_positive_like_all_negatives_together():
    # All logits 0 give the same per-pair loss, so the weighting must not change the mean.
    logits = torch.zeros(3, 5)
    assert torch.isclose(candidate_loss(logits, "bce"), torch.log(torch.tensor(2.0)))
    # A wrong positive costs as much as all k wrong negatives together.
    bad_positive = torch.tensor([[-10.0, -10.0, -10.0, -10.0, -10.0]])
    bad_negatives = torch.tensor([[10.0, 10.0, 10.0, 10.0, 10.0]])
    assert torch.isclose(candidate_loss(bad_positive, "bce"), candidate_loss(bad_negatives, "bce"),
                         rtol=1e-3)


def test_text_bank_reuses_known_texts_and_encodes_new_ones_once():
    class Encoder:
        calls = []

        def encode_texts(self, texts):
            self.calls.append(list(texts))
            return torch.ones(len(texts), 2)

    bank = TextBank(["a", "b"], torch.zeros(2, 2))
    encoder = Encoder()
    bank.add(["a", "c", "c", "d"], encoder)
    assert encoder.calls == [["c", "d"]]
    assert bank.ids(["b", "d", "c"]).tolist() == [1, 3, 2]
    bank.add(["a", "b"], None)
    with pytest.raises(ValueError):
        bank.add(["new"], None)
