"""Generate gradient-saliency and SHAP explanations for the tabular CNN-Attention model."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

from models.cnn_attention import CNNAttentionHybrid


CLASS_NAMES = ["Healthy Control", "RRMS", "Progressive MS"]


def _load_checkpoint_model(
    checkpoint_path: Path, device: torch.device
) -> Tuple[Dict[str, Any], Any, CNNAttentionHybrid]:
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    preprocessor = checkpoint["preprocessor"]
    config = checkpoint.get("model_config", {})
    model = CNNAttentionHybrid(
        in_features=config.get("in_features", preprocessor.num_features),
        token_dim=config.get("token_dim", 256),
        cnn_channels=(128, 256, 256, 256),
        cnn_seq_len=32,
        cnn_dropout=0.2,
        cnn_drop_path_rate=0.10,
        cnn_blocks_per_stage=(2, 3, 3, 2),
        num_attn_layers=config.get("num_layers", 6),
        num_heads=config.get("num_heads", 8),
        ffn_expansion=4,
        attn_dropout=0.1,
        attn_drop_path_rate=0.10,
        attn_dim=config.get("attn_dim", 256),
        head_dropout=0.3,
        num_phenotype_classes=3,
        num_group_classes=2,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return checkpoint, preprocessor, model


def _normalize_shap_values(
    raw_values: Any, sample_count: int, feature_count: int, class_count: int
) -> np.ndarray:
    if isinstance(raw_values, list):
        values = np.stack(raw_values, axis=-1)
    else:
        values = np.asarray(raw_values)

    if values.ndim != 3:
        raise ValueError(f"Expected 3D multi-class SHAP values, got {values.shape}")
    if values.shape[0] != sample_count:
        raise ValueError(f"SHAP sample dimension mismatch: {values.shape}")
    if values.shape[1] != feature_count and values.shape[2] == feature_count:
        values = np.transpose(values, (0, 2, 1))
    if values.shape[1:] != (feature_count, class_count):
        raise ValueError(f"SHAP feature/class dimensions mismatch: {values.shape}")
    return values


def generate_explanations(
    checkpoint_path: Path,
    train_data_path: Path,
    test_data_path: Path,
    output_dir: Path,
    max_explain_samples: int = 32,
    background_samples: int = 32,
    shap_samples: int = 16,
    seed: int = 42,
) -> Tuple[Path, Path]:
    """Explain the checkpoint on a deterministic sample of its held-out test set."""
    if max_explain_samples < 1 or background_samples < 1 or shap_samples < 1:
        raise ValueError("Explanation sample counts must be positive")

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint, preprocessor, model = _load_checkpoint_model(
        checkpoint_path, device
    )

    train_frame = preprocessor.load_data(train_data_path)
    test_frame = preprocessor.load_data(test_data_path)
    train_features, _ = preprocessor.transform(train_frame)
    test_features, test_labels = preprocessor.transform(test_frame)
    if test_labels is None:
        raise ValueError("Test data must include the configured target column")

    test_tensor = torch.as_tensor(test_features, dtype=torch.float32, device=device)
    with torch.no_grad():
        test_logits = torch.cat(
            [
                model(batch, task="phenotype")
                for batch in test_tensor.split(64)
            ],
            dim=0,
        )
        test_probabilities = torch.softmax(test_logits, dim=1).cpu().numpy()
    test_predictions = test_probabilities.argmax(axis=1)
    test_metrics = {
        "accuracy": float(accuracy_score(test_labels, test_predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(test_labels, test_predictions)
        ),
        "macro_f1": float(
            f1_score(test_labels, test_predictions, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(
                test_labels, test_predictions, average="weighted", zero_division=0
            )
        ),
        "classification_report": classification_report(
            test_labels,
            test_predictions,
            labels=np.arange(test_probabilities.shape[1]),
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            test_labels,
            test_predictions,
            labels=np.arange(test_probabilities.shape[1]),
        ).tolist(),
        "macro_ovr_roc_auc": float(
            roc_auc_score(
                test_labels,
                test_probabilities,
                labels=np.arange(test_probabilities.shape[1]),
                multi_class="ovr",
                average="macro",
            )
        ),
    }

    rng = np.random.default_rng(seed)
    n_explain = min(max_explain_samples, len(test_features))
    n_background = min(background_samples, len(train_features))
    explain_indices = rng.choice(
        len(test_features), size=n_explain, replace=False
    )
    background_indices = rng.choice(
        len(train_features), size=n_background, replace=False
    )
    features_to_explain = torch.as_tensor(
        test_features[explain_indices], dtype=torch.float32, device=device
    ).requires_grad_(True)
    background = torch.as_tensor(
        train_features[background_indices], dtype=torch.float32, device=device
    )

    with torch.enable_grad():
        logits = model(features_to_explain, task="phenotype")
        predicted_classes = logits.argmax(dim=1)
        selected_logits = logits.gather(1, predicted_classes[:, None]).sum()
        input_gradients = torch.autograd.grad(
            selected_logits, features_to_explain
        )[0]
        gradient_importance = (
            input_gradients.abs().mean(dim=0).detach().cpu().numpy()
        )

    explainer = shap.GradientExplainer(model, background)
    raw_shap = explainer.shap_values(
        features_to_explain.detach(), nsamples=shap_samples
    )
    class_count = logits.shape[1]
    shap_values = _normalize_shap_values(
        raw_shap,
        sample_count=n_explain,
        feature_count=test_features.shape[1],
        class_count=class_count,
    )
    explained_classes = predicted_classes.detach().cpu().numpy()
    selected_shap = np.stack(
        [
            shap_values[row, :, class_id]
            for row, class_id in enumerate(explained_classes)
        ]
    )
    shap_importance = np.abs(selected_shap).mean(axis=0)
    feature_names = np.asarray(preprocessor.feature_names)

    output_dir.mkdir(parents=True, exist_ok=True)
    chart_path = output_dir / "msnet_test_gradient_saliency_shap.png"
    fig_height = max(10, len(feature_names) * 0.32)
    fig, axes = plt.subplots(
        2, 1, figsize=(12, fig_height), constrained_layout=True
    )
    panels = (
        (
            axes[0],
            gradient_importance,
            "Input-Gradient Saliency (MSNet)",
            "#2878B5",
            "Mean absolute gradient of predicted-class logit",
        ),
        (
            axes[1],
            shap_importance,
            "SHAP GradientExplainer (MSNet)",
            "#D97706",
            "Mean absolute SHAP value for predicted class (logit scale)",
        ),
    )
    for axis, values, title, color, xlabel in panels:
        order = np.argsort(values)
        axis.barh(feature_names[order], values[order], color=color)
        axis.set_title(title)
        axis.set_xlabel(xlabel)
        axis.grid(axis="x", alpha=0.25)
        axis.tick_params(axis="y", labelsize=8)
    fig.suptitle(
        "MSNet Feature Attribution on Held-Out Test Data\n"
        "Tabular input attribution; not spatial MRI Grad-CAM",
        fontsize=14,
    )
    fig.savefig(str(chart_path), dpi=180, bbox_inches="tight")
    plt.close(fig)

    report_path = output_dir / "msnet_test_explainability_report.json"
    report = {
        "model": "MSNet",
        "architecture": "CNNAttentionHybrid (multi-branch tabular CNN-Attention)",
        "target": checkpoint.get("model_config", {}).get(
            "target", preprocessor.target_col
        ),
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_validation_metrics": checkpoint.get("val_metrics", {}),
        "training_data": str(train_data_path),
        "test_data": str(test_data_path),
        "test_samples": int(len(test_features)),
        "test_metrics": test_metrics,
        "checkpoint_selection_data": "data/val_data.csv",
        "test_data_note": (
            "The training script selected the checkpoint using val_data.csv. "
            "The default train, validation, and test files have no overlapping "
            "Patient: Hash values in this workspace."
        ),
        "explained_samples": int(n_explain),
        "background_training_samples": int(n_background),
        "feature_count": int(len(feature_names)),
        "seed": int(seed),
        "methods": {
            "gradient_saliency": (
                "Mean absolute input gradient of each sample's predicted-class "
                "phenotype logit."
            ),
            "shap": (
                "SHAP GradientExplainer on phenotype logits; mean absolute "
                "attribution for each sample's predicted class."
            ),
        },
        "plot": str(chart_path),
        "top_gradient_features": [
            {
                "feature": str(feature_names[index]),
                "mean_absolute_gradient": float(gradient_importance[index]),
            }
            for index in np.argsort(gradient_importance)[::-1][:10]
        ],
        "top_shap_features": [
            {
                "feature": str(feature_names[index]),
                "mean_absolute_shap": float(shap_importance[index]),
            }
            for index in np.argsort(shap_importance)[::-1][:10]
        ],
        "explained_sample_predicted_classes": {
            CLASS_NAMES[int(class_id)]: int(count)
            for class_id, count in zip(
                *np.unique(explained_classes, return_counts=True)
            )
        },
        "interpretation_caveat": (
            "Feature attribution describes model sensitivity, not causation. "
            "Explainability does not itself improve predictive performance; "
            "use it to inspect potential leakage, site effects, and unstable "
            "features before changing or retraining the model."
        ),
        "feature_audit_notes": [
            (
                "feat_T2LV_x_EDSS and feat_EDSS_per_year are derived using EDSS "
                "even though the raw EDSS column is excluded. Verify EDSS is "
                "available at intended prediction time and is not part of the "
                "phenotype label definition."
            ),
            (
                "Treatment-code_-1.0 represents a missing treatment value; "
                "its strong SHAP attribution warrants checking whether missingness "
                "patterns differ by class or site."
            ),
            (
                "Institution-code features appear among the influential "
                "features; assess site-stratified performance and external-site "
                "generalization."
            ),
        ],
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return chart_path, report_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Explain the trained tabular MSNet checkpoint"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/cnn_attention/best_cnn_attention.pt"),
    )
    parser.add_argument("--train-data", type=Path, default=Path("data/train_split.csv"))
    parser.add_argument("--test-data", type=Path, default=Path("data/test_data_clean.csv"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/cnn_attention")
    )
    parser.add_argument("--max-explain-samples", type=int, default=32)
    parser.add_argument("--background-samples", type=int, default=32)
    parser.add_argument("--shap-samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    for input_path in (
        args.checkpoint,
        args.train_data,
        args.test_data,
    ):
        if not input_path.is_file():
            raise FileNotFoundError(f"Required input file not found: {input_path}")

    chart_path, report_path = generate_explanations(
        checkpoint_path=args.checkpoint,
        train_data_path=args.train_data,
        test_data_path=args.test_data,
        output_dir=args.output_dir,
        max_explain_samples=args.max_explain_samples,
        background_samples=args.background_samples,
        shap_samples=args.shap_samples,
        seed=args.seed,
    )
    print(f"Attribution graph: {chart_path}")
    print(f"Explanation report: {report_path}")


if __name__ == "__main__":
    main()
