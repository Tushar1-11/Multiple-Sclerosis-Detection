import numpy as np
from sklearn.svm import SVC

from train_federated_svm import federated_linear_svc


def test_federated_linear_svc_matches_central_svc() -> None:
    rng = np.random.default_rng(12)
    labels = np.repeat(np.array([0, 1, 2]), 30)
    features = rng.normal(size=(len(labels), 4))
    features[:, 0] += labels * 2.0
    sites = np.tile(np.array([1, 2, 3]), 30)

    federated_model, details = federated_linear_svc(
        features, labels, sites, c_value=3.0
    )
    centralized_model = SVC(C=3.0, kernel="linear").fit(features, labels)

    np.testing.assert_array_equal(
        federated_model.predict(features), centralized_model.predict(features)
    )
    assert details["converged"]
    assert details["examples_shared"] <= details["total_training_examples"]
