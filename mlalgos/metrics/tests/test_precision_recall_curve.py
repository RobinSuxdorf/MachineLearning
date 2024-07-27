import numpy as np
from mlalgos.metrics import precision_recall_curve


def test_precision_recall_curve() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_scores = np.array([0.1, 0.4, 0.35, 0.8])

    expected_precision = np.array([0.5, 0.6666666666666666, 0.5, 1, 1])
    expected_recall = np.array([1, 1, 0.5, 0.5, 0])
    expected_thresholds = np.array([0.1 , 0.35, 0.4 , 0.8])

    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)

    assert np.array_equal(precision, expected_precision)
    assert np.array_equal(recall, expected_recall)
    assert np.array_equal(thresholds, expected_thresholds)
