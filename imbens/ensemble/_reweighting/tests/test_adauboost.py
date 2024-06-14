"""Test AdaUBoostClassifier."""

# Authors: Guillaume Lemaitre
#          Christos Aridas
#          Zhining Liu <zhining.liu@outlook.com>
# License: MIT

import numbers

import numpy as np
import pytest
import sklearn
from sklearn.datasets import load_iris, make_classification
from sklearn.model_selection import train_test_split
from sklearn.utils._testing import assert_array_equal
from sklearn.utils.fixes import parse_version

from imbens.ensemble import AdaUBoostClassifier

sklearn_version = parse_version(sklearn.__version__)


@pytest.fixture
def imbalanced_dataset():
    return make_classification(
        n_samples=10000,
        n_features=3,
        n_informative=2,
        n_redundant=0,
        n_repeated=0,
        n_classes=3,
        n_clusters_per_class=1,
        weights=[0.01, 0.05, 0.94],
        class_sep=0.8,
        random_state=0,
    )


@pytest.mark.parametrize("algorithm", ["SAMME"])
def test_algorithm(imbalanced_dataset, algorithm):
    X, y = imbalanced_dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, random_state=1
    )
    classes = np.unique(y)

    n_estimators = 50
    auboost = AdaUBoostClassifier(
        n_estimators=n_estimators, algorithm=algorithm, random_state=0
    )
    auboost.fit(X_train, y_train, beta="uniform")
    assert_array_equal(classes, auboost.classes_)

    # check that we have an ensemble of estimators with a
    # consistent size
    assert len(auboost.estimators_) > 1

    # each estimator in the ensemble should have different random state
    assert len({est.random_state for est in auboost.estimators_}) == len(
        auboost.estimators_
    )

    # check the consistency of the feature importances
    assert len(auboost.feature_importances_) == imbalanced_dataset[0].shape[1]

    # check the consistency of the prediction outpus
    y_pred = auboost.predict_proba(X_test)
    assert y_pred.shape[1] == len(classes)
    assert auboost.decision_function(X_test).shape[1] == len(classes)

    score = auboost.score(X_test, y_test)
    assert score > 0.6, f"Failed with algorithm {algorithm} and score {score}"

    y_pred = auboost.predict(X_test)
    assert y_pred.shape == y_test.shape


@pytest.mark.parametrize("algorithm", ["SAMME"])
def test_sample_weight(imbalanced_dataset, algorithm):
    X, y = imbalanced_dataset
    sample_weight = np.ones_like(y)
    auboost = AdaUBoostClassifier(algorithm=algorithm, random_state=0)

    # Predictions should be the same when sample_weight are all ones
    y_pred_sample_weight = auboost.fit(
        X,
        y,
        sample_weight=sample_weight,
        beta="uniform",
    ).predict(X)
    y_pred_no_sample_weight = auboost.fit(
        X,
        y,
        beta="uniform",
    ).predict(X)

    assert_array_equal(y_pred_sample_weight, y_pred_no_sample_weight)

    rng = np.random.RandomState(42)
    sample_weight = rng.rand(y.shape[0])
    y_pred_sample_weight = auboost.fit(
        X,
        y,
        sample_weight=sample_weight,
        beta="uniform",
    ).predict(X)

    with pytest.raises(AssertionError):
        assert_array_equal(y_pred_no_sample_weight, y_pred_sample_weight)


@pytest.mark.parametrize("algorithm", ["SAMME"])
@pytest.mark.parametrize(
    "beta", [None, "uniform", "inverse", "log1p-inverse", "random"]
)
def test_beta(imbalanced_dataset, algorithm, beta):
    expected_betas = {
        "uniform": {0: 1.0, 1: 1.0, 2: 1.0},
        "inverse": {0: 72.19587628865979, 1: 17.5075, 2: 1.0},
        "log1p-inverse": {
            0: 4.2931390845260236,
            1: 2.9181760553351164,
            2: 0.6931471805599453,
        },
    }

    X, y = imbalanced_dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, random_state=1
    )
    classes = np.unique(y)

    n_estimators = 50
    auboost = AdaUBoostClassifier(
        n_estimators=n_estimators, algorithm=algorithm, random_state=0
    )

    if beta in [None, "uniform", "inverse", "log1p-inverse"]:
        auboost.fit(X_train, y_train, beta=beta)
        assert_array_equal(classes, auboost.classes_)
        if beta is None:
            expected_beta = expected_betas["inverse"]
        else:
            expected_beta = expected_betas[beta]
        assert auboost.beta_ == expected_beta
    else:
        with pytest.raises(ValueError, match="When 'beta' is string"):
            auboost.fit(X_train, y_train, beta=beta)


@pytest.mark.parametrize(
    "beta",
    [
        {},
        {1: 1.0, 2: 1.0},
        {0: 1, 1: 1.0, 2: 1.0},
        {0: "a", 1: 1.0, 2: 1.0},
        10,
    ],
)
def test_beta_customize(imbalanced_dataset, beta):
    X, y = imbalanced_dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, random_state=1
    )

    auboost = AdaUBoostClassifier(random_state=0)
    if type(beta) == dict:
        if len(beta) < 3:
            with pytest.raises(
                ValueError, match="should specify beta values for all classes"
            ):
                auboost.fit(X_train, y_train, beta=beta)
        elif not all([isinstance(value, numbers.Number) for value in beta.values()]):
            with pytest.raises(
                ValueError, match="all values should be Integer or Real number"
            ):
                auboost.fit(X_train, y_train, beta=beta)
        else:
            auboost.fit(X_train, y_train, beta=beta)
    else:
        with pytest.raises(TypeError, match="'beta' should be one of"):
            auboost.fit(X_train, y_train, beta=beta)
