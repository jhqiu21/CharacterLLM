"""Checkpoint selection by validation accuracy, ties broken by the lower validation loss (training.checkpoint_selection)."""

import pytest

import train
from helpers import make_config
from utils.logger import Logger


def _metrics(acc, loss):
    return {"loss_all": loss, "loss_last": loss, "acc": acc, "acc_last": acc, "perplexity": 1.0, "bpc": loss,
            "pred_entropy": 1.0, "ece": [0.0, None]}


def test_best_acc_bpc_prefers_higher_accuracy_then_lower_loss():
    logger = Logger(10)
    # (acc, loss, is the newest evaluation the best so far?)
    sequence = [
        (0.50, 1.00, True),   # first evaluation
        (0.60, 1.10, True),   # higher accuracy wins although the loss is higher
        (0.60, 1.05, True),   # same accuracy, lower loss
        (0.60, 1.20, False),  # same accuracy, higher loss
        (0.55, 0.90, False),  # lower accuracy loses although the loss is lower
        (0.60, 1.05, False),  # exact tie with an earlier evaluation: the earlier one is kept
        (0.61, 1.30, True),
    ]
    for it, (acc, loss, expected) in enumerate(sequence):
        logger.log_eval(it, _metrics(acc, loss))
        assert logger.last_is_best_acc_bpc() is expected, (it, acc, loss)


@pytest.mark.parametrize("selection,prefix", [("loss", "best_loss_all_"), ("acc", "best_acc_bpc_")])
def test_selected_checkpoint(selection, prefix):
    config = make_config()
    config.training.checkpoint_selection = selection
    assert train.selected_checkpoint(config)[0] == prefix


def test_default_selection_is_validation_loss():
    assert make_config().training.checkpoint_selection == "loss"
    config = train.resolve_config(train.load_config(train.REPO_ROOT / "configs" / "final" / "TR.yaml"))
    assert train.selected_checkpoint(config)[0] == "best_loss_all_"


def test_unknown_selection_is_rejected():
    config = make_config()
    config.training.checkpoint_selection = "bpc"
    with pytest.raises(ValueError, match="checkpoint_selection"):
        train.selected_checkpoint(config)
