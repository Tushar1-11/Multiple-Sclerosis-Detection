"""Generate offline scikit-learn FedAvg and permutation-importance graphs."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data.preprocessing.tabular_preprocessor import CATEGORICAL_COLS, NUMERICAL_COLS


BASE_DIR = Path(__file__).resolve().parent
TARGET_COLUMN = "PHENOTYPE-CODE"
EXCLUDED_NUMERICAL_COLUMNS = {"EDSS", "Disease duration"}
CLASS_LABELS = {
    0: "Healthy Control",
    1: "RRMS",
    2: "Progressive MS",
}


def _model_from_parameters(
    classes: np.ndarray,
    coefficients: np.ndarray,
    intercepts: np.ndarray,
    n_features: int,
) -> LogisticRegression:
    model = LogisticRegression(solver="lbfgs", max_iter=1000)
    model.classes_ = classes
    if len(classes) == 2:
        model.coef_ = (coefficients[1] - coefficients[0]).reshape(1, -1)
        model.intercept_ = np.array([intercepts[1] - intercepts[0]])
    else:
        model.coef_ = coefficients
        model.intercept_ = intercepts
    model.n_features_in_ = n_features
    model.n_iter_ = np.array([1], dtype=np.int32)
    return model


@dataclass
class FederatedResult:
    global_model: LogisticRegression
    local_models: List[LogisticRegression]
    client_sizes: List[int]
    client_accuracies: List[float]
    global_accuracy: float
    centralized_model: LogisticRegression
    majority_model: DummyClassifier
    metrics: dict[str, Any]
    training_config: dict[str, Any]


def select_fedavg_hyperparameters(
    x_train: np.ndarray,
    y_train: np.ndarray,
    clients: int = 4,
    random_state: int = 42,
) -> tuple[dict[str, Any], float]:
    """Select FedAvg settings by stratified CV without using the final test set."""
    candidates = [
        {"rounds": 30, "local_epochs": 3, "learning_rate": 0.1, "batch_size": 64},
        {"rounds": 60, "local_epochs": 1, "learning_rate": 0.05, "batch_size": 64},
        {"rounds": 100, "local_epochs": 1, "learning_rate": 0.05, "batch_size": 64},
        {"rounds": 60, "local_epochs": 2, "learning_rate": 0.03, "batch_size": 64},
        {"rounds": 100, "local_epochs": 3, "learning_rate": 0.01, "batch_size": 64},
        {
            "rounds": 30,
            "local_epochs": 3,
            "learning_rate": 0.1,
            "batch_size": 64,
            "balanced_classes": True,
        },
        {
            "rounds": 60,
            "local_epochs": 1,
            "learning_rate": 0.05,
            "batch_size": 64,
            "balanced_classes": True,
        },
    ]
    splitter = StratifiedKFold(
        n_splits=3, shuffle=True, random_state=random_state
    )
    candidate_results = []

    for candidate in candidates:
        fold_metrics = []
        for fit_indices, validation_indices in splitter.split(x_train, y_train):
            result = simulate_fedavg(
                x_train[fit_indices],
                y_train[fit_indices],
                x_train[validation_indices],
                y_train[validation_indices],
                clients=clients,
                random_state=random_state,
                **candidate,
            )
            fold_metrics.append(result.metrics["fedavg"])
        mean_macro_f1 = float(
            np.mean([metrics["macro_f1"] for metrics in fold_metrics])
        )
        mean_balanced_accuracy = float(
            np.mean([metrics["balanced_accuracy"] for metrics in fold_metrics])
        )
        mean_accuracy = float(
            np.mean([metrics["accuracy"] for metrics in fold_metrics])
        )
        candidate_results.append(
            (
                (mean_macro_f1, mean_balanced_accuracy, mean_accuracy),
                candidate,
            )
        )

    (best_macro_f1, _, _), best_config = max(
        candidate_results, key=lambda item: item[0]
    )
    return {**best_config, "cv_macro_f1": best_macro_f1}, best_macro_f1


def simulate_fedavg(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    clients: int = 4,
    random_state: int = 42,
    rounds: int = 30,
    local_epochs: int = 3,
    learning_rate: float = 0.1,
    batch_size: int = 64,
    l2: float = 1e-4,
    balanced_classes: bool = False,
) -> FederatedResult:
    """Train a multinomial linear model with iterative sample-weighted FedAvg."""
    x_train = np.asarray(x_train, dtype=np.float64)
    y_train = np.asarray(y_train)
    x_test = np.asarray(x_test, dtype=np.float64)
    y_test = np.asarray(y_test)
    if (
        x_train.ndim != 2
        or x_test.ndim != 2
        or y_train.ndim != 1
        or y_test.ndim != 1
        or x_train.shape[0] != y_train.shape[0]
        or x_test.shape[0] != y_test.shape[0]
        or x_train.shape[1] == 0
        or x_train.shape[1] != x_test.shape[1]
        or x_train.shape[0] == 0
        or x_test.shape[0] == 0
    ):
        raise ValueError("Training and test features/labels must have matching non-empty shapes")
    if not np.isfinite(x_train).all() or not np.isfinite(x_test).all():
        raise ValueError("Federated features must contain only finite values")
    if clients < 2:
        raise ValueError("At least two clients are required")
    if rounds < 1 or local_epochs < 1 or batch_size < 1:
        raise ValueError("Rounds, local epochs, and batch size must be positive")
    if learning_rate <= 0 or l2 < 0:
        raise ValueError("Learning rate must be positive and L2 strength non-negative")

    classes, class_counts = np.unique(y_train, return_counts=True)
    if len(classes) < 2 or int(class_counts.min()) < clients:
        raise ValueError("Each class must have at least one training record per client")
    if not np.isin(y_test, classes).all():
        raise ValueError("Test labels must all be present in the training labels")

    splitter = StratifiedKFold(
        n_splits=clients, shuffle=True, random_state=random_state
    )
    client_indices_list = [
        client_indices
        for _, client_indices in splitter.split(x_train, y_train)
    ]
    client_sizes = [len(client_indices) for client_indices in client_indices_list]
    class_indices = {label: index for index, label in enumerate(classes)}
    global_coefficients = np.zeros((len(classes), x_train.shape[1]))
    global_intercepts = np.zeros(len(classes))
    rng = np.random.default_rng(random_state)
    local_models: List[LogisticRegression] = []

    for _ in range(rounds):
        local_parameters = []
        for client_indices in client_indices_list:
            coefficients = global_coefficients.copy()
            intercepts = global_intercepts.copy()
            client_x = x_train[client_indices]
            client_y = np.asarray(
                [class_indices[label] for label in y_train[client_indices]],
                dtype=np.int64,
            )
            class_weights = np.ones(len(classes))
            if balanced_classes:
                class_counts = np.bincount(client_y, minlength=len(classes))
                class_weights = len(client_y) / (
                    len(classes) * np.maximum(class_counts, 1)
                )

            for _ in range(local_epochs):
                shuffled = rng.permutation(len(client_indices))
                for start in range(0, len(shuffled), batch_size):
                    batch_indices = shuffled[start : start + batch_size]
                    batch_x = client_x[batch_indices]
                    batch_y = client_y[batch_indices]
                    scores = batch_x @ coefficients.T + intercepts
                    scores -= scores.max(axis=1, keepdims=True)
                    probabilities = np.exp(scores)
                    probabilities /= probabilities.sum(axis=1, keepdims=True)
                    probabilities[np.arange(len(batch_y)), batch_y] -= 1
                    sample_weights = class_weights[batch_y]
                    probabilities *= (
                        sample_weights / sample_weights.sum()
                    )[:, np.newaxis]
                    coefficients -= learning_rate * (
                        probabilities.T @ batch_x + l2 * coefficients
                    )
                    intercepts -= learning_rate * probabilities.sum(axis=0)

            local_parameters.append((coefficients, intercepts))

        global_coefficients = np.average(
            [parameters[0] for parameters in local_parameters],
            axis=0,
            weights=client_sizes,
        )
        global_intercepts = np.average(
            [parameters[1] for parameters in local_parameters],
            axis=0,
            weights=client_sizes,
        )
        local_models = [
            _model_from_parameters(
                classes, coefficients, intercepts, x_train.shape[1]
            )
            for coefficients, intercepts in local_parameters
        ]

    global_model = _model_from_parameters(
        classes, global_coefficients, global_intercepts, x_train.shape[1]
    )
    centralized_model = LogisticRegression(
        solver="lbfgs", max_iter=2000, random_state=random_state
    ).fit(x_train, y_train)
    balanced_centralized_model = LogisticRegression(
        solver="lbfgs",
        max_iter=2000,
        class_weight="balanced",
        random_state=random_state,
    ).fit(x_train, y_train)
    majority_model = DummyClassifier(strategy="most_frequent").fit(
        x_train, y_train
    )
    models = {
        "fedavg": global_model,
        "centralized_logistic_regression": centralized_model,
        "centralized_balanced_logistic_regression": balanced_centralized_model,
        "majority_baseline": majority_model,
    }
    target_names = [CLASS_LABELS.get(int(label), str(label)) for label in classes]
    metrics = {
        name: {
            "accuracy": float(accuracy_score(y_test, model.predict(x_test))),
            "balanced_accuracy": float(
                balanced_accuracy_score(y_test, model.predict(x_test))
            ),
            "macro_f1": float(
                f1_score(y_test, model.predict(x_test), average="macro")
            ),
            "classification_report": classification_report(
                y_test,
                model.predict(x_test),
                labels=classes,
                target_names=target_names,
                output_dict=True,
                zero_division=0,
            ),
            "confusion_matrix": confusion_matrix(
                y_test, model.predict(x_test), labels=classes
            ).tolist(),
        }
        for name, model in models.items()
    }
    client_accuracies = [
        float(accuracy_score(y_test, model.predict(x_test)))
        for model in local_models
    ]
    global_accuracy = metrics["fedavg"]["accuracy"]

    return FederatedResult(
        global_model=global_model,
        local_models=local_models,
        client_sizes=client_sizes,
        client_accuracies=client_accuracies,
        global_accuracy=global_accuracy,
        centralized_model=centralized_model,
        majority_model=majority_model,
        metrics=metrics,
        training_config={
            "rounds": rounds,
            "local_epochs": local_epochs,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "l2": l2,
            "balanced_classes": balanced_classes,
        },
    )


def _load_features(data_path: Path):
    from data.preprocessing.tabular_preprocessor import TabularPreprocessor

    frame = TabularPreprocessor(use_feature_engineering=False).load_data(data_path)
    if TARGET_COLUMN not in frame.columns:
        raise KeyError(f"Required target column {TARGET_COLUMN!r} was not found")

    numerical_columns = [
        column
        for column in NUMERICAL_COLS
        if column in frame.columns and column not in EXCLUDED_NUMERICAL_COLUMNS
    ]
    categorical_columns = [
        column for column in CATEGORICAL_COLS if column in frame.columns
    ]
    if not numerical_columns and not categorical_columns:
        raise ValueError("The dataset has no supported clinical or volumetric features")

    labels = np.asarray(
        pd.to_numeric(frame.loc[:, TARGET_COLUMN], errors="coerce"),
        dtype=np.float64,
    )
    if not np.isfinite(labels).all() or not np.equal(labels, np.floor(labels)).all():
        raise ValueError(f"Target column {TARGET_COLUMN!r} contains missing or invalid labels")
    y = labels.astype(np.int64)
    if not set(np.unique(y)).issubset(CLASS_LABELS):
        raise ValueError("Phenotype labels must be 0 (HC), 1 (RRMS), or 2 (Progressive MS)")

    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    transformers = []
    if numerical_columns:
        transformers.append(("numeric", numeric_pipeline, numerical_columns))
    if categorical_columns:
        transformers.append(("categorical", categorical_pipeline, categorical_columns))
    preprocessor = ColumnTransformer(transformers, verbose_feature_names_out=False)

    x_train_frame, x_test_frame, y_train, y_test = train_test_split(
        frame,
        y,
        test_size=0.25,
        stratify=y,
        random_state=42,
    )
    x_train = preprocessor.fit_transform(x_train_frame)
    x_test = preprocessor.transform(x_test_frame)
    feature_names = preprocessor.get_feature_names_out().tolist()
    return x_train, x_test, y_train, y_test, feature_names


def generate_graphs(
    data_path: Path = BASE_DIR / "data" / "train_data.csv",
    output_dir: Path = BASE_DIR / "outputs",
) -> tuple[Path, Path]:
    """Train the offline scikit-learn demo and save its two graph outputs."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x_train, x_test, y_train, y_test, feature_names = _load_features(data_path)
    selected_config, cv_macro_f1 = select_fedavg_hyperparameters(
        x_train, y_train
    )
    selected_config.pop("cv_macro_f1")
    result = simulate_fedavg(
        x_train, y_train, x_test, y_test, **selected_config
    )
    output_dir = Path(output_dir)
    federated_dir = output_dir / "federated"
    explainability_dir = output_dir / "explainability"
    federated_dir.mkdir(parents=True, exist_ok=True)
    explainability_dir.mkdir(parents=True, exist_ok=True)

    federated_path = federated_dir / "fedavg_client_accuracy.png"
    metrics_path = federated_dir / "fedavg_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        json.dump(
            {
                "training_config": result.training_config,
                "cross_validation_macro_f1": cv_macro_f1,
                "metrics": result.metrics,
            },
            metrics_file,
            indent=2,
        )

    fig, ax = plt.subplots(figsize=(10, 5.5))
    positions = np.arange(len(result.local_models))
    bars = ax.bar(
        positions,
        result.client_accuracies,
        color=["#06b6d4", "#38bdf8", "#818cf8", "#a78bfa"][: len(positions)],
        edgecolor="#e2e8f0",
        linewidth=0.6,
    )
    ax.axhline(
        result.global_accuracy,
        color="#fbbf24",
        linestyle="--",
        linewidth=2,
        label=f"FedAvg global model ({result.global_accuracy:.1%})",
    )
    centralized_accuracy = result.metrics["centralized_logistic_regression"][
        "accuracy"
    ]
    balanced_accuracy = result.metrics[
        "centralized_balanced_logistic_regression"
    ]["accuracy"]
    majority_accuracy = result.metrics["majority_baseline"]["accuracy"]
    ax.axhline(
        centralized_accuracy,
        color="#22c55e",
        linestyle=":",
        linewidth=2,
        label=f"Centralized logistic regression ({centralized_accuracy:.1%})",
    )
    ax.axhline(
        balanced_accuracy,
        color="#a78bfa",
        linestyle="-.",
        linewidth=1.5,
        label=f"Balanced centralized logistic ({balanced_accuracy:.1%})",
    )
    ax.axhline(
        majority_accuracy,
        color="#f97316",
        linestyle="-.",
        linewidth=1.5,
        label=f"Majority baseline ({majority_accuracy:.1%})",
    )
    ax.set_xticks(
        positions,
        [
            f"Client {index + 1}\n{size:,} train rows"
            for index, size in enumerate(result.client_sizes)
        ],
    )
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy on shared held-out test set")
    ax.set_title("Iterative FedAvg: Client vs Centralized Baselines")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    for bar, accuracy in zip(bars, result.client_accuracies):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.015,
            f"{accuracy:.1%}",
            ha="center",
            va="bottom",
        )
    fig.text(
        0.5,
        0.01,
        "Central CSV simulation; disjoint stratified shards; iterative weighted softmax updates. Not a privacy guarantee.",
        ha="center",
        fontsize=8,
        color="#475569",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(str(federated_path), dpi=180, bbox_inches="tight")
    plt.close(fig)

    importance = permutation_importance(
        result.global_model,
        x_test,
        y_test,
        scoring="accuracy",
        n_repeats=8,
        random_state=42,
        n_jobs=1,
    )
    importance_means = importance["importances_mean"]
    importance_stds = importance["importances_std"]
    top_indices = np.argsort(importance_means)[-15:]
    explainability_path = explainability_dir / "sklearn_permutation_importance.png"
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(
        np.asarray(feature_names)[top_indices],
        importance_means[top_indices],
        xerr=importance_stds[top_indices],
        color="#06b6d4",
        edgecolor="#0e7490",
        capsize=3,
    )
    ax.axvline(0, color="#475569", linewidth=0.8)
    ax.set_xlabel("Decrease in held-out accuracy after feature shuffling")
    ax.set_title("Scikit-learn Permutation Importance for the FedAvg Model")
    ax.grid(axis="x", alpha=0.25)
    fig.text(
        0.5,
        0.01,
        "Mean decrease over 8 shuffles; larger positive values indicate greater predictive contribution, not causation.",
        ha="center",
        fontsize=9,
        color="#475569",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(str(explainability_path), dpi=180, bbox_inches="tight")
    plt.close(fig)

    for model_name, model_metrics in result.metrics.items():
        print(
            f"{model_name}: accuracy={model_metrics['accuracy']:.3f}, "
            f"balanced_accuracy={model_metrics['balanced_accuracy']:.3f}, "
            f"macro_f1={model_metrics['macro_f1']:.3f}"
        )
    print(f"Selected FedAvg settings: {result.training_config}")
    print(f"Training-only CV macro-F1: {cv_macro_f1:.3f}")
    print(f"Metrics report: {metrics_path}")
    print(f"Federated graph: {federated_path}")
    print(f"Explainability graph: {explainability_path}")
    print("This pooled-CSV simulation is offline and does not provide data privacy.")
    return federated_path, explainability_path


if __name__ == "__main__":
    generate_graphs()
