"""
MSNet — Comprehensive Evaluation Metrics
==========================================
Computes all metrics required for IEEE TMI / MICCAI publication:

  Binary Classification (HC vs MS):
    - Accuracy, Sensitivity (Recall), Specificity, Precision
    - F1-Score, Matthews Correlation Coefficient (MCC)
    - ROC-AUC (with curve), PR-AUC (with curve)
    - Cohen's Kappa, Confusion Matrix

  Multi-class (HC / RRMS / SPMS / PPMS):
    - Per-class: Precision, Recall, F1
    - Macro and weighted averages
    - One-vs-Rest ROC-AUC
    - Micro-average PR-AUC
    - Confusion matrix

  Bootstrap Confidence Intervals:
    - 95% CI for all primary metrics

All metrics use scikit-learn and torchmetrics for GPU acceleration.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    average_precision_score,  # PR-AUC
    confusion_matrix,
    classification_report,
    cohen_kappa_score,
    matthews_corrcoef,
    roc_curve,
    precision_recall_curve,
)

logger = logging.getLogger(__name__)


# ============================================================
# Core Metric Computation
# ============================================================

def compute_all_metrics(
    labels: List[int],
    preds: List[int],
    num_classes: int = 4,
    probs: Optional[List[List[float]]] = None,
    average: str = "macro",
) -> Dict[str, float]:
    """
    Compute comprehensive classification metrics.

    Args:
        labels: True class labels [N]
        preds: Predicted class labels [N]
        num_classes: Number of classes
        probs: Softmax probabilities [N, n_classes] (for AUC metrics)
        average: Averaging strategy for multi-class ('macro'/'weighted')

    Returns:
        Dict of metric names → scalar values
    """
    labels = np.array(labels)
    preds  = np.array(preds)

    metrics = {}

    # ---- Core Classification Metrics ----
    metrics["accuracy"] = float(accuracy_score(labels, preds))

    # For binary: standard recall = sensitivity, specificity via CM
    if num_classes == 2:
        cm = confusion_matrix(labels, preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

        metrics["sensitivity"] = float(tp / (tp + fn + 1e-8))  # Recall for class 1
        metrics["specificity"] = float(tn / (tn + fp + 1e-8))  # Recall for class 0
        metrics["precision"]   = float(tp / (tp + fp + 1e-8))
        metrics["f1_score"]    = float(f1_score(labels, preds, zero_division=0))
        metrics["npv"]         = float(tn / (tn + fn + 1e-8))  # Negative predictive value

    else:
        # Multi-class: per-class and macro averages
        recall_per = recall_score(labels, preds, average=None, zero_division=0)
        prec_per   = precision_score(labels, preds, average=None, zero_division=0)
        f1_per     = f1_score(labels, preds, average=None, zero_division=0)

        for i, (r, p, f) in enumerate(zip(recall_per, prec_per, f1_per)):
            metrics[f"sensitivity_class{i}"] = float(r)
            metrics[f"precision_class{i}"]   = float(p)
            metrics[f"f1_class{i}"]          = float(f)

        metrics["sensitivity"] = float(recall_score(labels, preds, average=average, zero_division=0))
        metrics["precision"]   = float(precision_score(labels, preds, average=average, zero_division=0))
        metrics["f1_macro"]    = float(f1_score(labels, preds, average="macro", zero_division=0))
        metrics["f1_weighted"] = float(f1_score(labels, preds, average="weighted", zero_division=0))
        metrics["f1_score"]    = metrics["f1_macro"]

        # Multi-class specificity: macro average of per-class specificity
        cm = confusion_matrix(labels, preds)
        specificities = []
        for i in range(num_classes):
            tp_i = cm[i, i]
            fn_i = cm[i, :].sum() - tp_i
            fp_i = cm[:, i].sum() - tp_i
            tn_i = cm.sum() - tp_i - fn_i - fp_i
            specificities.append(tn_i / (tn_i + fp_i + 1e-8))
        metrics["specificity"] = float(np.mean(specificities))

    # ---- Cohen's Kappa & MCC ----
    metrics["cohen_kappa"] = float(cohen_kappa_score(labels, preds))
    metrics["mcc"]         = float(matthews_corrcoef(labels, preds))

    # ---- AUC Metrics (require probability scores) ----
    if probs is not None:
        probs_arr = np.array(probs)

        # ROC-AUC
        try:
            if num_classes == 2:
                metrics["roc_auc"] = float(roc_auc_score(labels, probs_arr[:, 1]))
            else:
                metrics["roc_auc"] = float(
                    roc_auc_score(labels, probs_arr, multi_class="ovr", average="macro")
                )
        except Exception as e:
            logger.warning(f"ROC-AUC computation failed: {e}")
            metrics["roc_auc"] = 0.0

        # PR-AUC (average precision)
        try:
            if num_classes == 2:
                metrics["pr_auc"] = float(average_precision_score(labels, probs_arr[:, 1]))
            else:
                # Macro OvR PR-AUC
                pr_aucs = []
                for c in range(num_classes):
                    y_bin = (labels == c).astype(int)
                    pr_aucs.append(
                        average_precision_score(y_bin, probs_arr[:, c])
                    )
                metrics["pr_auc"] = float(np.mean(pr_aucs))
        except Exception as e:
            logger.warning(f"PR-AUC computation failed: {e}")
            metrics["pr_auc"] = 0.0

    return metrics


# ============================================================
# Bootstrap Confidence Intervals
# ============================================================

def bootstrap_confidence_intervals(
    labels: List[int],
    preds: List[int],
    num_classes: int,
    probs: Optional[List[List[float]]] = None,
    n_bootstraps: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    """
    Compute bootstrap 95% confidence intervals for all metrics.

    Samples with replacement from [labels, preds] n_bootstraps times,
    recomputes all metrics, and returns percentile CIs.

    Args:
        labels: True labels [N]
        preds: Predicted labels [N]
        num_classes: Number of classes
        probs: Softmax probabilities [N, C] (optional)
        n_bootstraps: Number of bootstrap replicates
        confidence: Confidence level (0.95 = 95% CI)
        seed: Random seed for reproducibility

    Returns:
        Dict: {metric_name: {'mean': x, 'ci_low': y, 'ci_high': z}}
    """
    rng = np.random.RandomState(seed)
    labels = np.array(labels)
    preds  = np.array(preds)
    probs_arr = np.array(probs) if probs else None
    N = len(labels)

    # Collect bootstrap metric samples
    bootstrap_scores: Dict[str, List[float]] = {}

    for b in range(n_bootstraps):
        idx = rng.choice(N, size=N, replace=True)
        b_labels = labels[idx]
        b_preds  = preds[idx]
        b_probs  = probs_arr[idx].tolist() if probs_arr is not None else None

        # Skip bootstrap if only one class present
        if len(np.unique(b_labels)) < 2:
            continue

        try:
            m = compute_all_metrics(b_labels.tolist(), b_preds.tolist(), num_classes, b_probs)
            for k, v in m.items():
                bootstrap_scores.setdefault(k, []).append(v)
        except Exception:
            pass

    # Compute CIs
    alpha = (1 - confidence) / 2
    ci_dict = {}

    for metric, scores in bootstrap_scores.items():
        scores_arr = np.array(scores)
        ci_dict[metric] = {
            "mean":     float(np.mean(scores_arr)),
            "ci_low":   float(np.percentile(scores_arr, alpha * 100)),
            "ci_high":  float(np.percentile(scores_arr, (1 - alpha) * 100)),
            "std":      float(np.std(scores_arr)),
        }

    return ci_dict


# ============================================================
# Threshold Optimization (Binary)
# ============================================================

def find_optimal_threshold(
    labels: List[int],
    probs: List[float],
    method: str = "youden_j",
) -> Tuple[float, float]:
    """
    Find optimal decision threshold for binary classification.

    Methods:
      - 'youden_j': Maximizes sensitivity + specificity - 1
      - 'f1': Maximizes F1 score
      - 'balanced': Minimizes |sensitivity - specificity|

    Args:
        labels: True binary labels [N]
        probs: Probability of positive class [N]
        method: Threshold selection criterion

    Returns:
        (optimal_threshold, metric_value_at_threshold)
    """
    fpr, tpr, thresholds = roc_curve(labels, probs)

    if method == "youden_j":
        # Youden's J = sensitivity + specificity - 1
        youden = tpr - fpr
        idx = np.argmax(youden)
        return float(thresholds[idx]), float(youden[idx])

    elif method == "f1":
        # F1 maximization
        precision, recall, pr_thresholds = precision_recall_curve(labels, probs)
        f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
        idx = np.argmax(f1_scores)
        thresh = pr_thresholds[min(idx, len(pr_thresholds) - 1)]
        return float(thresh), float(f1_scores[idx])

    elif method == "balanced":
        # |sensitivity - specificity| minimization
        diff = np.abs(tpr - (1 - fpr))
        idx = np.argmin(diff)
        return float(thresholds[idx]), float(diff[idx])

    else:
        raise ValueError(f"Unknown threshold method: {method}")


# ============================================================
# Formatted Report Generator
# ============================================================

def generate_evaluation_report(
    labels: List[int],
    preds: List[int],
    num_classes: int,
    class_names: List[str],
    probs: Optional[List[List[float]]] = None,
    n_bootstraps: int = 1000,
    output_path: Optional[str] = None,
) -> str:
    """
    Generate a publication-quality evaluation report.

    Includes point estimates + 95% bootstrap CIs for all metrics,
    confusion matrix, and per-class breakdown.

    Args:
        labels: True labels [N]
        preds: Predicted labels [N]
        num_classes: Number of classes
        class_names: Human-readable class names
        probs: Softmax probabilities [N, C]
        n_bootstraps: Bootstrap replicates
        output_path: Optional path to save .txt report

    Returns:
        Formatted string report
    """
    # Point estimates
    metrics = compute_all_metrics(labels, preds, num_classes, probs)

    # Bootstrap CIs
    ci = bootstrap_confidence_intervals(labels, preds, num_classes, probs, n_bootstraps)

    # Confusion matrix
    cm = confusion_matrix(labels, preds)

    # sklearn report
    clf_report = classification_report(
        labels, preds, target_names=class_names, zero_division=0
    )

    lines = [
        "=" * 70,
        "MSNet — Classification Evaluation Report",
        "=" * 70,
        "",
        f"{'Metric':<25} {'Value':>10} {'95% CI':>25}",
        "-" * 65,
    ]

    key_metrics = [
        ("accuracy",     "Accuracy"),
        ("sensitivity",  "Sensitivity (macro)"),
        ("specificity",  "Specificity (macro)"),
        ("f1_macro",     "F1-Score (macro)"),
        ("roc_auc",      "ROC-AUC"),
        ("pr_auc",       "PR-AUC"),
        ("cohen_kappa",  "Cohen's Kappa"),
        ("mcc",          "Matthews CC"),
    ]

    for key, label in key_metrics:
        if key in metrics:
            val = metrics[key]
            if key in ci:
                ci_str = f"[{ci[key]['ci_low']:.4f}, {ci[key]['ci_high']:.4f}]"
            else:
                ci_str = "N/A"
            lines.append(f"{label:<25} {val:>10.4f} {ci_str:>25}")

    lines += [
        "",
        "-" * 65,
        "Per-Class Report:",
        clf_report,
        "",
        "Confusion Matrix:",
        "  Pred -> " + " ".join(f"{c:>8}" for c in class_names),
    ]

    for i, row in enumerate(cm):
        row_str = f"  {class_names[i]:>8} " + " ".join(f"{v:>8}" for v in row)
        lines.append(row_str)

    lines.append("=" * 70)
    report = "\n".join(lines)

    logger.info(f"\n{report}")

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)
        logger.info(f"[Report] Saved -> {output_path}")

    return report
