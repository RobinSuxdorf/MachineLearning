import numpy as np

from metrics import (
    accuracy_score,
    confusion_matrix
)

def test_accuracy_sccore():
    y_true = [1, 0, 1, 1, 0]
    y_pred = [1, 1, 0, 1, 0]

    accuracy = accuracy_score(y_true, y_pred)

    expected_accuracy = 0.6

    assert accuracy == expected_accuracy

def test_confusion_matrix_binary_classification():
    y_true = [1, 0, 1, 1, 0, 0]
    y_pred = [1, 1, 0, 1, 0, 1]

    cm = confusion_matrix(y_true, y_pred)

    expected_cm = np.array([
        [1, 2], 
        [1, 2]
    ])

    assert np.array_equal(cm, expected_cm)

def test_confusion_matrix_multiclass_classification():
    y_true = [0, 1, 2, 1, 2, 0, 1, 2]
    y_pred = [0, 2, 1, 0, 2, 1, 0, 1]

    cm = confusion_matrix(y_true, y_pred)

    expected_cm = np.array([
        [1, 1, 0],
        [2, 0, 1],
        [0, 2, 1]
    ])

    assert np.array_equal(cm, expected_cm)