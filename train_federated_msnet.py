"""Train the tabular CNN-Attention MSNet with site-partitioned FedAvg."""

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, TensorDataset

from data.preprocessing.tabular_preprocessor import TabularPreprocessor
from models.cnn_attention import CNNAttentionHybrid


LOGGER = logging.getLogger("MSNet.Federated")
CLASS_NAMES = ["Healthy Control", "RRMS", "Progressive MS"]


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def partition_by_site(site_codes: np.ndarray) -> Dict[Any, np.ndarray]:
    """Return disjoint client row indices, one client per observed institution."""
    site_codes = np.asarray(site_codes)
    if site_codes.ndim != 1 or len(site_codes) == 0:
        raise ValueError("Site codes must be a non-empty one-dimensional array")
    if pd.isna(site_codes).any():
        raise ValueError("Site codes cannot be missing when assigning FL clients")

    return {
        site: np.flatnonzero(site_codes == site)
        for site in np.sort(np.unique(site_codes))
    }


def fedavg_state_dict(
    client_states: List[Dict[str, torch.Tensor]],
    client_sizes: List[int],
) -> Dict[str, torch.Tensor]:
    """Average client model states weighted by each client's training rows."""
    if not client_states or len(client_states) != len(client_sizes):
        raise ValueError("Client states and sizes must be non-empty and aligned")
    if any(size <= 0 for size in client_sizes):
        raise ValueError("Every client must contribute a positive sample count")

    keys = client_states[0].keys()
    if any(state.keys() != client_states[0].keys() for state in client_states[1:]):
        raise ValueError("All client model states must have identical keys")

    weights = np.asarray(client_sizes, dtype=np.float64)
    weights /= weights.sum()
    averaged: Dict[str, torch.Tensor] = {}
    for key in keys:
        tensors = [state[key] for state in client_states]
        if any(tensor.shape != tensors[0].shape for tensor in tensors[1:]):
            raise ValueError(f"Client model tensor shape mismatch for {key}")
        if tensors[0].is_floating_point() or tensors[0].is_complex():
            accumulator = torch.zeros_like(tensors[0])
            for weight, tensor in zip(weights, tensors):
                accumulator.add_(tensor, alpha=float(weight))
            averaged[key] = accumulator
        else:
            averaged[key] = tensors[int(np.argmax(weights))].clone()
    return averaged


def build_model(model_config: Dict[str, Any], device: torch.device) -> CNNAttentionHybrid:
    model = CNNAttentionHybrid(
        in_features=int(model_config["in_features"]),
        token_dim=int(model_config["token_dim"]),
        cnn_channels=(128, 256, 256, 256),
        cnn_seq_len=32,
        cnn_dropout=0.2,
        cnn_drop_path_rate=0.10,
        cnn_blocks_per_stage=(2, 3, 3, 2),
        num_attn_layers=int(model_config["num_layers"]),
        num_heads=int(model_config["num_heads"]),
        ffn_expansion=4,
        attn_dropout=0.1,
        attn_drop_path_rate=0.10,
        attn_dim=int(model_config["attn_dim"]),
        head_dropout=0.3,
        num_phenotype_classes=3,
        num_group_classes=2,
    )
    return model.to(device)


def _optimizer(
    model: CNNAttentionHybrid, learning_rate: float, weight_decay: float
) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        [
            {"params": model.branch1_cnn.parameters(), "lr": learning_rate * 0.05},
            {"params": model.branch2_ft.parameters(), "lr": learning_rate * 0.2},
            {"params": model.branch3_tab.parameters(), "lr": learning_rate * 0.2},
            {"params": model.recurrent_fusion.parameters(), "lr": learning_rate * 0.5},
            {"params": model.transformer_blocks.parameters(), "lr": learning_rate * 0.5},
            {"params": [model.cls_token], "lr": learning_rate * 0.5},
            {"params": [model.pos_embed], "lr": learning_rate * 0.5},
            {"params": model.latent_fusion.parameters(), "lr": learning_rate},
            {"params": model.phenotype_head.parameters(), "lr": learning_rate},
            {"params": model.group_head.parameters(), "lr": learning_rate},
            {"params": model.edss_head.parameters(), "lr": learning_rate},
        ],
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
    )


def _evaluate(
    model: CNNAttentionHybrid,
    features: np.ndarray,
    labels: np.ndarray,
    device: torch.device,
) -> Dict[str, Any]:
    model.eval()
    logits_list = []
    with torch.no_grad():
        for start in range(0, len(features), 128):
            batch = torch.as_tensor(
                features[start : start + 128], dtype=torch.float32, device=device
            )
            output = model(batch, task="phenotype")
            logits_list.append(output.cpu())

    logits = torch.cat(logits_list, dim=0)
    probabilities = torch.softmax(logits, dim=1).numpy()
    predictions = probabilities.argmax(axis=1)
    label_ids = np.arange(len(CLASS_NAMES))
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(labels, predictions)
        ),
        "macro_f1": float(
            f1_score(labels, predictions, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(labels, predictions, average="weighted", zero_division=0)
        ),
        "macro_ovr_roc_auc": float(
            roc_auc_score(
                labels,
                probabilities,
                labels=label_ids,
                multi_class="ovr",
                average="macro",
            )
        ),
        "classification_report": classification_report(
            labels,
            predictions,
            labels=label_ids,
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            labels, predictions, labels=label_ids
        ).tolist(),
        "predictions": predictions,
        "probabilities": probabilities,
    }


def _train_one_client(
    global_state: Dict[str, torch.Tensor],
    model_config: Dict[str, Any],
    features: np.ndarray,
    labels: np.ndarray,
    edss_values: np.ndarray,
    class_weights: torch.Tensor,
    device: torch.device,
    local_epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
) -> Dict[str, torch.Tensor]:
    set_reproducible_seed(seed)
    model = build_model(model_config, device)
    model.load_state_dict(global_state)
    optimizer = _optimizer(model, learning_rate, weight_decay)
    dataset = TensorDataset(
        torch.as_tensor(features, dtype=torch.float32),
        torch.as_tensor(labels, dtype=torch.long),
        torch.as_tensor(edss_values, dtype=torch.float32),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    class_weights = class_weights.to(device)

    for _ in range(local_epochs):
        model.train()
        for batch_features, batch_labels, batch_edss in loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            batch_edss = batch_edss.to(device)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch_features, return_all=True)
            logits = outputs["phenotype"]
            ce = F.cross_entropy(
                logits,
                batch_labels,
                weight=class_weights,
                reduction="none",
            )
            target_prob = torch.exp(-ce.detach())
            focal = ((1.0 - target_prob) ** 2 * ce).mean()
            edss_loss = F.smooth_l1_loss(
                outputs["edss"].squeeze(-1), batch_edss
            )
            loss = focal + 0.2 * edss_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

    local_state = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }
    del model, optimizer, loader, dataset
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return local_state


def _assert_disjoint_patient_splits(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
) -> Dict[str, int]:
    hash_column = "Patient: Hash"
    if any(hash_column not in frame.columns for frame in (
        train_frame,
        validation_frame,
        test_frame,
    )):
        raise KeyError(
            "Training, validation, and test data must include 'Patient: Hash'"
        )
    hashes = {
        name: set(frame[hash_column].dropna().astype(str))
        for name, frame in (
            ("train", train_frame),
            ("validation", validation_frame),
            ("test", test_frame),
        )
    }
    overlaps = {
        f"{first}_{second}": len(hashes[first] & hashes[second])
        for first, second in (
            ("train", "validation"),
            ("train", "test"),
            ("validation", "test"),
        )
    }
    if any(overlaps.values()):
        raise ValueError(f"Patient leakage across splits detected: {overlaps}")
    return overlaps


def train_federated_msnet(
    train_path: Path,
    validation_path: Path,
    test_path: Path,
    output_dir: Path,
    rounds: int,
    local_epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    seed: int,
) -> Tuple[Path, Path, Path]:
    if min(rounds, local_epochs, batch_size, patience) < 1:
        raise ValueError("Rounds, local epochs, batch size, and patience must be positive")
    if learning_rate <= 0 or weight_decay < 0:
        raise ValueError("Learning rate must be positive and weight decay non-negative")

    set_reproducible_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preprocessor = TabularPreprocessor(
        target_name="phenotype",
        random_state=seed,
        exclude_cols=["EDSS", "Disease duration"],
        use_feature_engineering=True,
    )
    train_frame = preprocessor.load_data(train_path)
    validation_frame = preprocessor.load_data(validation_path)
    test_frame = preprocessor.load_data(test_path)
    split_overlaps = _assert_disjoint_patient_splits(
        train_frame, validation_frame, test_frame
    )

    train_x, train_y, _ = preprocessor.fit_transform(train_frame)
    validation_x, validation_y = preprocessor.transform(validation_frame)
    test_x, test_y = preprocessor.transform(test_frame)
    if validation_y is None or test_y is None:
        raise ValueError("Validation and test data must include phenotype labels")
    if not np.isfinite(train_x).all() or not np.isfinite(validation_x).all() or not np.isfinite(test_x).all():
        raise ValueError("Preprocessed features contain non-finite values")

    edss_values = np.clip(
        pd.to_numeric(train_frame["EDSS"], errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=np.float32),
        0.0,
        10.0,
    )
    if "INSTITUTION-CODE" not in train_frame:
        raise KeyError("Training data must include INSTITUTION-CODE for site clients")
    client_indices = partition_by_site(
        pd.to_numeric(train_frame["INSTITUTION-CODE"], errors="coerce").to_numpy()
    )
    if len(client_indices) < 2:
        raise ValueError("At least two institution clients are required")
    if set(np.unique(train_y)) != {0, 1, 2}:
        raise ValueError("Training data must contain all three phenotype classes")
    class_counts = np.bincount(train_y, minlength=3)
    class_weights = torch.tensor(
        len(train_y) / (3.0 * class_counts), dtype=torch.float32
    )

    model_config = {
        "in_features": int(train_x.shape[1]),
        "token_dim": 256,
        "attn_dim": 256,
        "num_layers": 6,
        "num_heads": 8,
        "target": "phenotype",
        "exclude_cols": ["EDSS", "Disease duration"],
    }
    global_model = build_model(model_config, device)
    global_state = {
        key: value.detach().cpu().clone()
        for key, value in global_model.state_dict().items()
    }
    del global_model

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best_federated_msnet.pt"
    report_path = output_dir / "federated_msnet_report.json"
    graph_path = output_dir / "federated_msnet_training.png"
    best_validation_f1 = -1.0
    best_round = 0
    rounds_without_improvement = 0
    validation_history = []
    client_order = list(client_indices)
    client_sizes = [len(client_indices[site]) for site in client_order]
    client_summaries = {
        str(site): {
            "site_code": int(site),
            "train_samples": int(len(client_indices[site])),
            "class_counts": {
                CLASS_NAMES[class_id]: int(
                    np.count_nonzero(train_y[client_indices[site]] == class_id)
                )
                for class_id in range(3)
            },
        }
        for site in client_order
    }

    for round_number in range(1, rounds + 1):
        client_states = []
        for client_number, site in enumerate(client_order):
            indices = client_indices[site]
            local_state = _train_one_client(
                global_state=global_state,
                model_config=model_config,
                features=train_x[indices],
                labels=train_y[indices],
                edss_values=edss_values[indices],
                class_weights=class_weights,
                device=device,
                local_epochs=local_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                seed=seed + round_number * 1009 + client_number,
            )
            client_states.append(local_state)

        global_state = fedavg_state_dict(client_states, client_sizes)
        global_model = build_model(model_config, device)
        global_model.load_state_dict(global_state)
        validation_metrics = _evaluate(
            global_model, validation_x, validation_y, device
        )
        validation_history.append(
            {
                "round": round_number,
                "accuracy": validation_metrics["accuracy"],
                "macro_f1": validation_metrics["macro_f1"],
                "balanced_accuracy": validation_metrics["balanced_accuracy"],
            }
        )
        LOGGER.info(
            "Round %d/%d | validation accuracy %.4f | macro-F1 %.4f",
            round_number,
            rounds,
            validation_metrics["accuracy"],
            validation_metrics["macro_f1"],
        )

        if validation_metrics["macro_f1"] > best_validation_f1:
            best_validation_f1 = validation_metrics["macro_f1"]
            best_round = round_number
            rounds_without_improvement = 0
            torch.save(
                {
                    "epoch": round_number,
                    "round": round_number,
                    "model_state_dict": global_state,
                    "val_metrics": {
                        key: value
                        for key, value in validation_metrics.items()
                        if key not in {"predictions", "probabilities"}
                    },
                    "preprocessor": preprocessor,
                    "model_config": model_config,
                    "federated_config": {
                        "protocol": "FedAvg",
                        "clients": len(client_order),
                        "client_partition": "one client per INSTITUTION-CODE",
                        "rounds_requested": rounds,
                        "local_epochs": local_epochs,
                        "batch_size": batch_size,
                        "learning_rate": learning_rate,
                        "weight_decay": weight_decay,
                        "seed": seed,
                    },
                    "client_sizes": {
                        str(site): int(len(client_indices[site]))
                        for site in client_order
                    },
                },
                checkpoint_path,
            )
        else:
            rounds_without_improvement += 1

        del global_model, client_states
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if rounds_without_improvement >= patience:
            LOGGER.info("Early stopping after round %d", round_number)
            break

    if best_round == 0:
        raise RuntimeError("Federated training did not produce a validation checkpoint")

    best_checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    final_model = build_model(model_config, device)
    final_model.load_state_dict(best_checkpoint["model_state_dict"])
    test_metrics = _evaluate(final_model, test_x, test_y, device)
    predictions = test_metrics.pop("predictions")
    probabilities = test_metrics.pop("probabilities")

    site_codes_test = pd.to_numeric(
        test_frame["INSTITUTION-CODE"], errors="coerce"
    ).to_numpy()
    site_test_metrics = {}
    for site in np.sort(np.unique(site_codes_test[~pd.isna(site_codes_test)])):
        indices = np.flatnonzero(site_codes_test == site)
        site_test_metrics[str(site)] = {
            "site_code": int(site),
            "test_samples": int(len(indices)),
            "accuracy": float(accuracy_score(test_y[indices], predictions[indices])),
            "balanced_accuracy": float(
                balanced_accuracy_score(test_y[indices], predictions[indices])
            ),
            "macro_f1": float(
                f1_score(
                    test_y[indices],
                    predictions[indices],
                    labels=np.arange(3),
                    average="macro",
                    zero_division=0,
                )
            ),
            "class_counts": {
                CLASS_NAMES[class_id]: int(
                    np.count_nonzero(test_y[indices] == class_id)
                )
                for class_id in range(3)
            },
        }
        if not np.isfinite(probabilities[indices]).all():
            raise RuntimeError(f"Non-finite probabilities found for client site {site}")

    for site, summary in client_summaries.items():
        summary["test_metrics"] = site_test_metrics.get(site)

    report = {
        "model": "MSNet",
        "architecture": "CNNAttentionHybrid (multi-branch tabular CNN-Attention)",
        "protocol": "sample-count-weighted FedAvg",
        "deployment_status": (
            "Central-process simulation; clients are partitioned by institution "
            "but data is loaded in one process. This is not a privacy guarantee."
        ),
        "training_data": str(train_path),
        "validation_data": str(validation_path),
        "test_data": str(test_path),
        "patient_hash_overlap": split_overlaps,
        "client_partition": "one institution per client, using INSTITUTION-CODE",
        "clients": client_summaries,
        "class_weights": {
            CLASS_NAMES[class_id]: float(class_weights[class_id])
            for class_id in range(3)
        },
        "training": {
            "rounds_completed": len(validation_history),
            "best_round": best_round,
            "best_validation_macro_f1": float(best_validation_f1),
            "local_epochs": local_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "seed": seed,
            "device": str(device),
        },
        "validation_history": validation_history,
        "test_metrics": {
            key: value
            for key, value in test_metrics.items()
        },
        "checkpoint": str(checkpoint_path),
        "graph": str(graph_path),
        "interpretation_notes": [
            (
                "The client split reflects observed institution-level label "
                "heterogeneity; local client class distributions are not IID."
            ),
            (
                "INSTITUTION-CODE is also a model input. Review site-stratified "
                "results for site dependence and test external-site generalization."
            ),
            (
                "Feature preprocessing is fit on the pooled training CSV in this "
                "simulation. Production FL should fit/share approved statistics "
                "without centralizing patient rows."
            ),
            (
                "The model is selected using validation macro-F1. Test metrics "
                "are evaluated once after the best round is restored."
            ),
        ],
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    axes[0].plot(
        [row["round"] for row in validation_history],
        [row["macro_f1"] for row in validation_history],
        marker="o",
        color="#2878B5",
    )
    axes[0].axvline(
        best_round,
        color="#D97706",
        linestyle="--",
        label=f"Best round {best_round}",
    )
    axes[0].set_xlabel("Federated round")
    axes[0].set_ylabel("Validation macro-F1")
    axes[0].set_title("MSNet FedAvg Validation Progress")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    sorted_sites = sorted(site_test_metrics, key=lambda value: int(value))
    site_labels = [f"Site {site}" for site in sorted_sites]
    site_scores = [site_test_metrics[site]["macro_f1"] for site in sorted_sites]
    axes[1].bar(site_labels, site_scores, color="#38A169")
    axes[1].axhline(
        test_metrics["macro_f1"],
        color="#D97706",
        linestyle="--",
        label=f"Overall test macro-F1 {test_metrics['macro_f1']:.3f}",
    )
    axes[1].set_xlabel("Institution client")
    axes[1].set_ylabel("Test macro-F1")
    axes[1].set_title("Held-Out Test Performance by Client Site")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend()
    fig.suptitle("MSNet Federated Training (FedAvg)")
    fig.tight_layout()
    fig.savefig(str(graph_path), dpi=180, bbox_inches="tight")
    plt.close(fig)
    return checkpoint_path, report_path, graph_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train CNN-Attention MSNet with institution-partitioned FedAvg"
    )
    parser.add_argument("--train-data", type=Path, default=Path("data/train_split.csv"))
    parser.add_argument(
        "--validation-data", type=Path, default=Path("data/val_data.csv")
    )
    parser.add_argument(
        "--test-data", type=Path, default=Path("data/test_data_clean.csv")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/federated_msnet")
    )
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING"], default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    for path in (args.train_data, args.validation_data, args.test_data):
        if not path.is_file():
            raise FileNotFoundError(f"Dataset file not found: {path}")

    paths = train_federated_msnet(
        train_path=args.train_data,
        validation_path=args.validation_data,
        test_path=args.test_data,
        output_dir=args.output_dir,
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        seed=args.seed,
    )
    print(f"Federated checkpoint: {paths[0]}")
    print(f"Federated report: {paths[1]}")
    print(f"Federated graph: {paths[2]}")


if __name__ == "__main__":
    main()
