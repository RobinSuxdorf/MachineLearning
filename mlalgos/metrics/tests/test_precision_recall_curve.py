import pytest
from contextlib import nullcontext
from typing import ContextManager, Optional
import numpy as np
from mlalgos.metrics import precision_recall_curve

@pytest.mark.parametrize(
    "y_true, pos_label, expectation",
    [
        (np.array([0, 0, 1, 1]), None, nullcontext()),
        (np.array([0, 0, 1, 1]), 1, nullcontext()),
        (np.array([0, 0, 2, 2]), 2, nullcontext()),
        (np.array([1, 1, 0, 0]), 0, nullcontext()),
        (np.array([0, 0, 1, 2]), None, pytest.raises(ValueError, match=r"y_true contains more than 2 different values.")),
        (np.array([0, 0, 1, 1]), 2, pytest.raises(ValueError, match=r"pos_label is not contained in y_true.")),
    ]
)
def test_precision_recall_curve(y_true: np.ndarray, pos_label: Optional[int], expectation: ContextManager) -> None:
    y_scores = np.array([0.1, 0.4, 0.35, 0.8])

    expected_precision = np.array([0.5, 0.6666666666666666, 0.5, 1, 1])
    expected_recall = np.array([1, 1, 0.5, 0.5, 0])
    expected_thresholds = np.array([0.1 , 0.35, 0.4 , 0.8])

    with expectation:
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores, pos_label=pos_label)

        assert np.array_equal(precision, expected_precision)
        assert np.array_equal(recall, expected_recall)
        assert np.array_equal(thresholds, expected_thresholds)
