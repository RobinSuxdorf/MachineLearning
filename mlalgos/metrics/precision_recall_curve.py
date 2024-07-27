from typing import Optional
import numpy as np
from mlalgos.helpers import check_length
from mlalgos.metrics import confusion_matrix


def precision_recall_curve(y_true: np.ndarray, y_scores: np.ndarray, pos_label: Optional[int] = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculates precision-recall pairs for different thresholds.

    Args:
        y_true (np.ndarray): Array of true binary labels.
        y_scores (np.ndarray): Target values.
        pos_label (Optional[int]): The label of the positive class.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: The precision and recall values for certain thresholds.
    """
    check_length(y_true, y_scores)

    unique_elements = np.unique(y_true)
    if len(unique_elements) > 2:
        raise ValueError("y_true contains more than 2 different values.")

    if pos_label is not None and pos_label not in unique_elements:
        raise ValueError("pos_label is not contained in y_true.")

    if pos_label is None:
        pos_label = unique_elements[1]

    neg_label = unique_elements[0] if pos_label != unique_elements[0] else unique_elements[1]

    precision: list[float] = []
    recall: list[float] = []
    thresholds = np.sort(y_scores)

    for threshold in thresholds:
        predictions = np.where(y_scores >= threshold, pos_label, neg_label)

        cf_matrix = confusion_matrix(y_true, predictions, class_labels=[pos_label, neg_label])

        tp = cf_matrix[0][0]
        fp = cf_matrix[1][0]
        fn = cf_matrix[0][1]

        precision_value = tp / (tp + fp) if (tp + fp) != 0 else 0
        recall_value = tp / (tp + fn) if (tp + fn) != 0 else 0

        precision.append(precision_value)
        recall.append(recall_value)

    precision.append(1)
    recall.append(0)

    return np.array(precision), np.array(recall), thresholds