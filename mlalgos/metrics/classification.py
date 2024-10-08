from typing import Any, Optional
import numpy as np
from mlalgos import ArrayLike
from mlalgos.helpers import check_array, check_length, check_type


def accuracy_score(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """
    Calculates the accuracy score, which is the proportion of correct predictions.

    Args:
        y_true (ArrayLike): The list of true labels.
        y_pred (ArrayLike): The list of predicted labels.

    Returns:
        float: The accuracy score as float between 0 and 1.
    """
    y_true, y_pred = check_array(y_true, y_pred)
    check_length(y_true, y_pred)
    check_type(y_true, y_pred)

    correct_predictions = sum(
        1
        for (true_label, pred_label) in zip(y_true, y_pred)
        if true_label == pred_label
    )

    return correct_predictions / len(y_true)


def confusion_matrix(
    y_true: ArrayLike, y_pred: ArrayLike, class_labels: Optional[ArrayLike] = None
) -> np.ndarray:
    """
    Calculates the confusion matrix.

    Args:
        y_true (ArrayLike): The list of true labels.
        y_pred (ArrayLike): The list of predicted labels.
        class_labels (Optional[ArrayLike]): List of labels to index the matrix.

    Returns:
        np.ndarray: The confusion matrix as numpy array of arrays.
    """
    y_true, y_pred = check_array(y_true, y_pred)
    check_length(y_true, y_pred)
    check_type(y_true, y_pred)

    if class_labels is None:
        class_labels = np.union1d(y_true, y_pred)

    num_classes = len(class_labels)

    cf_matrix = np.zeros((num_classes, num_classes), dtype=int)

    class_to_idx = {label: idx for idx, label in enumerate(class_labels)}

    for true_label, pred_label in zip(y_true, y_pred):
        true_idx = class_to_idx[true_label]
        pred_idx = class_to_idx[pred_label]

        cf_matrix[true_idx][pred_idx] += 1

    return cf_matrix


class ClassificationReport:
    """
    A class to generate a classification report including precision, recall, F1-score, and support for each class,
    as well as overall accuracy.
    """

    def __init__(self, y_true: ArrayLike, y_pred: ArrayLike) -> None:
        """
        Initializes the ClassificationReport with the true and predicted labels, computes the confusion matrix,
        and calculates precision, recall, F1-score, and support for each class.

        Args:
            y_true (ArrayLike): The true class labels.
            y_pred (ArrayLike): The predicted class labels.
        """
        y_true, y_pred = check_array(y_true, y_pred)
        check_length(y_true, y_pred)
        check_type(y_true, y_pred)

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
                "support": support,
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
        header = "{: >10} {: >10} {: >10} {: >10} {: >10}".format(
            "Class", "Precision", "Recall", "F1-Score", "Support"
        )
        report = [header]
        for class_label, values in self._report.items():
            report.append(
                "{: >10} {precision: >10.2f} {recall: >10.2f} {f1_score: >10.2f} {support: >10}".format(
                    class_label, **values
                )
            )

        report.append(
            "{: >10} {: >10} {: >10} {: >10.2f} {: >10}".format(
                "Accuracy", "", "", self._accuracy, self._total_support
            )
        )

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
