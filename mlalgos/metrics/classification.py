from typing import Any, Optional
import numpy as np

def _check_length(list1: list[Any], list2: list[Any]) -> None:
    """
    Checks if the two lists have the same length.

    Args:
        list1 (list[Any]): The first list to check.
        list2 (list[Any]): The second list to check.

    Raises:
        ValueError: If the lists do not have the same length.
    """
    if len(list1) != len(list2):
        raise ValueError("The lists do not have the same length.")

def _check_type(list1: list[Any], list2: list[Any]) -> None:
    """
    Checks if all elements in both lists are of the same type.

    Args:
        list1 (list[Any]): The first list to check.
        lst2 (list[Any]): The second list to check.

    Raises:
        ValueError: If not all elements of list1 and list2 have the same type.
    """
    if not list1:
        raise ValueError("The first list is empty.")

    list_type = type(list1[0])

    if not all(isinstance(entry, list_type) for entry in list1):
        raise ValueError("Not all elements in the first list are of the same type.")

    if not all(isinstance(entry, list_type) for entry in list2):
        raise ValueError("Not all elements in the second list are of the same type as the first element of the first list.")

def accuracy_score(y_true: list[Any], y_pred: list[Any]) -> float:
    """
    Calculates the accuracy score, which is the proportion of correct predictions.

    Args:
        y_true (list[Any]): The list of true labels.
        y_pred (list[Any]): The list of predicted labels.

    Returns:
        float: The accuracy score as float between 0 and 1.
    """
    _check_length(y_true, y_pred)
    _check_type(y_true, y_pred)

    correct_predictions = sum(1 for (true_label, pred_label) in zip(y_true, y_pred) if true_label == pred_label)

    return correct_predictions / len(y_true)

def confusion_matrix(y_true: list[Any], y_pred: list[Any], class_labels: Optional[list[Any]] = None) -> np.ndarray:
    """
    Calculates the confusion matrix.

    Args:
        y_true (list[Any]): The list of true labels.
        y_pred (list[Any]): The list of predicted labels.

    Returns:
        np.ndarray: The confusion matrix as numpy array of arrays.
    """
    _check_length(y_true, y_pred)
    _check_type(y_true, y_pred)

    if class_labels is None:
        class_labels = np.union1d(y_true, y_pred)

    num_classes = len(class_labels)

    cf_matrix = np.zeros((num_classes, num_classes), dtype=int)

    class_to_idx = {label: idx for idx, label in enumerate(class_labels)}

    for (true_label, pred_label) in zip(y_true, y_pred):
        true_idx = class_to_idx[true_label]
        pred_idx = class_to_idx[pred_label]

        cf_matrix[true_idx][pred_idx] += 1

    return cf_matrix

class ClassificationReport:
    """
    A class to generate a classification report including precision, recall, F1-score, and support for each class,
    as well as overall accuracy.
    """
    def __init__(self, y_true: list[Any], y_pred: list[Any]) -> None:
        """
        Initializes the ClassificationReport with the true and predicted labels, computes the confusion matrix,
        and calculates precision, recall, F1-score, and support for each class.

        Args:
            y_true (list[Any]): The true class labels.
            y_pred (list[Any]): The predicted class labels.
        """
        _check_length(y_true, y_pred)
        _check_type(y_true, y_pred)

        self._report: dict[str, dict[str, Any]] = {}

        class_labels = np.union1d(y_true, y_pred)
        cf_matrix = confusion_matrix(y_true, y_pred, class_labels)

        col_sum = cf_matrix.sum(axis=0)
        row_sum = cf_matrix.sum(axis=1)

        diagonal_sum = 0

        for idx, class_label in enumerate(class_labels):
            precision = cf_matrix[idx][idx] / col_sum[idx] if col_sum[idx] != 0 else 0
            recall = cf_matrix[idx][idx] / row_sum[idx] if row_sum[idx] != 0 else 0

            f1_score = 0
            if precision != 0 and recall != 0:
                f1_score = 2 * precision * recall / (precision + recall)

            support = row_sum[idx]

            self._report[class_label] = {
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "support": support
            }

            diagonal_sum += cf_matrix[idx][idx]

        self._accuracy = diagonal_sum / sum(col_sum)
        self._total_support = sum(row_sum)

    def _report_to_str(self) -> str:
        """
        Converts the classification report to a formatted string.

        Returns:
            str: A string representation of the classification report.
        """
        header = "{: >10} {: >10} {: >10} {: >10} {: >10}".format("Class", "Precision", "Recall", "F1-Score", "Support")
        report = [header]
        for class_label, values in self._report.items():
            report.append("{: >10} {precision: >10.2f} {recall: >10.2f} {f1_score: >10.2f} {support: >10}".format(class_label, **values))

        report.append("{: >10} {: >10} {: >10} {: >10.2f} {: >10}".format("Accuracy", "", "", self._accuracy, self._total_support))

        return "\n".join(report)

    def __str__(self) -> str:
        """
        Returns the formatted classification report as a string.

        Returns:
            str: A string representation of the classification report.
        """
        return self._report_to_str()

    def __repr__(self) -> str:
        """
        Returns the formatted classification report as a string.

        Returns:
            str: A string representation of the classification report.
        """
        return self._report_to_str()
