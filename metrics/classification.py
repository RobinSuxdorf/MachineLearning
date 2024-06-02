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
    