from typing import Optional
import numpy as np
from mlalgos.metrics import confusion_matrix


def precision_recall_curve(y_true: np.ndarray, y_scores: np.ndarray, pos_label: Optional[int] = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # check for type and length?

    if not pos_label:
        # check whether y_true is in {-1, 1} or {0, 1}, otherwise raise a value error
        pass

    precision: list[float] = []
    recall: list[float] = []
    thresholds = np.sort(y_scores)

    for threshold in thresholds:
        predictions = np.array([1 if score >= threshold else 0 for score in y_scores]) # use pos_label

        cf_matrix = confusion_matrix(y_true, predictions, class_labels=[1, 0]) # use pos_label

        tp = cf_matrix[0][0]
        fp = cf_matrix[1][0]
        fn = cf_matrix[0][1]

        precision_value = tp / (tp + fp)
        recall_value = tp / (tp + fn)

        precision.append(precision_value)
        recall.append(recall_value)

    precision.append(1)
    recall.append(0)

    precision = np.asarray(precision)
    recall = np.asarray(recall)

    return precision, recall, thresholds