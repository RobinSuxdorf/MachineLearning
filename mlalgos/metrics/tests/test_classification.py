import pytest
from typing import Any, Optional
import numpy as np

from mlalgos.metrics import accuracy_score, confusion_matrix, ClassificationReport


def test_accuracy_sccore() -> None:
    y_true = [1, 0, 1, 1, 0]
    y_pred = [1, 1, 0, 1, 0]

    accuracy = accuracy_score(y_true, y_pred)

    expected_accuracy = 0.6

    assert accuracy == expected_accuracy

@pytest.mark.parametrize(
    "y_true, y_pred, class_labels, expected_cm",
    [
        ([1, 0, 1, 1, 0, 0], [1, 1, 0, 1, 0, 1], None, np.array([[1, 2], [1, 2]])),
        ([0, 1, 2, 1, 2, 0, 1, 2], [0, 2, 1, 0, 2, 1, 0, 1], None, np.array([[1, 1, 0], [2, 0, 1], [0, 2, 1]])),
        ([1, 0, 1, 1, 0, 0], [1, 1, 0, 1, 0, 1], [1, 0], np.array([[2, 1], [2, 1]]))
    ]
)
def test_confusion_matrix(y_true: list[Any], y_pred: list[Any], class_labels: Optional[list[Any]], expected_cm: np.ndarray) -> None:
    cm = confusion_matrix(y_true, y_pred, class_labels)
    assert np.array_equal(cm, expected_cm)

def test_classification_report() -> None:
    y_true = [0, 2, 1, 2, 1, 1, 0, 1]
    y_pred = [2, 1, 0, 2, 2, 2, 0, 1]

    report = ClassificationReport(y_true, y_pred)

    expected_report = {
        0: {"precision": 0.5, "recall": 0.5, "f1_score": 0.5, "support": 2},
        1: {
            "precision": 0.5,
            "recall": 0.25,
            "f1_score": 0.3333333333333333,
            "support": 4,
        },
        2: {
            "precision": 0.25,
            "recall": 0.5,
            "f1_score": 0.3333333333333333,
            "support": 2,
        },
    }

    report._report == expected_report
