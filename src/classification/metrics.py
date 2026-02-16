"""
Classification metrics: accuracy, precision, recall, F1, AUC-ROC.
"""

import torch
import numpy as np
from typing import Dict, Optional, List
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


class ClassificationMetrics:
    """Accumulates predictions and computes classification metrics."""

    def __init__(self, num_classes: int = 4, class_names: Optional[List[str]] = None):
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]
        self.reset()

    def reset(self):
        self.all_preds = []
        self.all_targets = []
        self.all_probs = []

    def update(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ):
        """
        Update with batch predictions.

        Args:
            logits: Model logits (B, num_classes).
            targets: Ground truth labels (B,).
        """
        probs = torch.softmax(logits.detach(), dim=1).cpu().numpy()
        preds = logits.detach().argmax(dim=1).cpu().numpy()
        targets = targets.detach().cpu().numpy()

        self.all_preds.extend(preds)
        self.all_targets.extend(targets)
        self.all_probs.extend(probs)

    def compute(self) -> Dict[str, float]:
        """Compute all classification metrics."""
        preds = np.array(self.all_preds)
        targets = np.array(self.all_targets)
        probs = np.array(self.all_probs)

        results = {}

        # Accuracy
        results["accuracy"] = float(accuracy_score(targets, preds))

        # Precision, Recall, F1 (macro average)
        precision, recall, f1, _ = precision_recall_fscore_support(
            targets, preds, average="macro", zero_division=0
        )
        results["precision"] = float(precision)
        results["recall"] = float(recall)
        results["f1"] = float(f1)

        # Per-class metrics
        precision_pc, recall_pc, f1_pc, support = precision_recall_fscore_support(
            targets, preds, average=None, zero_division=0
        )
        for i, name in enumerate(self.class_names[:len(np.unique(targets))]):
            if i < len(precision_pc):
                results[f"precision_{name}"] = float(precision_pc[i])
                results[f"recall_{name}"] = float(recall_pc[i])
                results[f"f1_{name}"] = float(f1_pc[i])

        # AUC-ROC (one-vs-rest)
        try:
            if probs.shape[1] == 2:
                results["auc_roc"] = float(roc_auc_score(targets, probs[:, 1]))
            else:
                results["auc_roc"] = float(
                    roc_auc_score(targets, probs, multi_class="ovr", average="macro")
                )
        except ValueError:
            results["auc_roc"] = 0.0

        return results

    def confusion_matrix(self) -> np.ndarray:
        """Compute confusion matrix."""
        return confusion_matrix(self.all_targets, self.all_preds)

    def report(self) -> str:
        """Generate classification report string."""
        return classification_report(
            self.all_targets,
            self.all_preds,
            target_names=self.class_names,
            zero_division=0,
        )


if __name__ == "__main__":
    metrics = ClassificationMetrics(
        num_classes=4,
        class_names=["Normal", "Mild", "Moderate", "Severe"],
    )

    for _ in range(10):
        logits = torch.randn(8, 4)
        targets = torch.randint(0, 4, (8,))
        metrics.update(logits, targets)

    results = metrics.compute()
    for k, v in results.items():
        print(f"{k}: {v:.4f}")

    print("\nConfusion Matrix:")
    print(metrics.confusion_matrix())
    print("\nClassification Report:")
    print(metrics.report())
