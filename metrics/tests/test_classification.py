import numpy as np

from metrics import (
    accuracy_score,
    confusion_matrix,
    ClassificationReport
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

def test_classification_report():
    y_true = [0, 2, 1, 2, 1, 1, 0, 1]
    y_pred = [2, 1, 0, 2, 2, 2, 0, 1]

    report = ClassificationReport(y_true, y_pred)

    expected_report = {
        0: {
            'precision': 0.5, 
            'recall': 0.5, 
            'f1_score': 0.5, 
            'support': 2
        },
        1: {
            'precision': 0.5,
            'recall': 0.25,
            'f1_score': 0.3333333333333333,
            'support': 4
        },
        2: {
            'precision': 0.25,
            'recall': 0.5,
            'f1_score': 0.3333333333333333,
            'support': 2
        }
    }

    report._report == expected_report