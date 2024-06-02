from typing import Any, Optional
import numpy as np

def confusion_matrix(y_true: list[Any], y_pred: list[Any], class_labels: Optional[list[Any]] = None) -> np.array:
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
    def __init__(self, y_true: list[Any], y_pred: list[Any]) -> None:
        self._report_rows: list[tuple] = []

        class_labels = np.union1d(y_true, y_pred)
        cf_matrix = confusion_matrix(y_true, y_pred, class_labels)

        col_sum = cf_matrix.sum(axis=0)
        row_sum = cf_matrix.sum(axis=1)

        self._diagonal_sum = 0

        for idx, class_label in enumerate(class_labels):
            precision = cf_matrix[idx][idx] / col_sum[idx] if col_sum[idx] != 0 else 0
            recall = cf_matrix[idx][idx] / row_sum[idx] if row_sum[idx] != 0 else 0

            f1_score = 0
            if precision != 0 and recall != 0:
                f1_score = 2 * precision * recall / (precision + recall)

            support = row_sum[idx]

            self._report_rows.append((
                class_label,
                precision,
                recall,
                f1_score,
                support
            ))

            self._diagonal_sum += cf_matrix[idx][idx]

        self._accuracy = self._diagonal_sum / sum(col_sum)

    def _report_to_str(self) -> str:
        header = "{: >10} {: >10} {: >10} {: >10} {: >10}".format("Class", "Precision", "Recall", "F1-Score", "Support")
        report = [header]
        for row in self._report_rows:
            report.append("{: >10} {: >10.2f} {: >10.2f} {: >10.2f} {: >10}".format(*row))

        report.append("{: >10} {: >10.2f} {: >10}".format("Accuracy", self._accuracy, self._diagonal_sum))

        return "\n".join(report)

    def __str__(self) -> str:
        return self._report_to_str()

    def __repr__(self) -> str:
        return self._report_to_str()
