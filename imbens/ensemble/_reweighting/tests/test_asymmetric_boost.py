"""Test AsymBoostClassifier."""

# Authors: Guillaume Lemaitre
#          Christos Aridas
#          Zhining Liu <zhining.liu@outlook.com>
# License: MIT

import numpy as np
import pytest
import sklearn
from sklearn.datasets import load_iris, make_classification
from sklearn.model_selection import train_test_split
from sklearn.utils._testing import assert_array_equal
from sklearn.utils.fixes import parse_version

from imbens.ensemble import AsymBoostClassifier

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
    asymboost = AsymBoostClassifier(
        n_estimators=n_estimators, algorithm=algorithm, random_state=0
    )
    asymboost.fit(X_train, y_train, cost_matrix="uniform")
    assert_array_equal(classes, asymboost.classes_)

    # check that we have an ensemble of estimators with a
    # consistent size
    assert len(asymboost.estimators_) > 1

    # each estimator in the ensemble should have different random state
    assert len({est.random_state for est in asymboost.estimators_}) == len(
        asymboost.estimators_
    )

    # check the consistency of the feature importances
    assert len(asymboost.feature_importances_) == imbalanced_dataset[0].shape[1]

    # check the consistency of the prediction outpus
    y_pred = asymboost.predict_proba(X_test)
    assert y_pred.shape[1] == len(classes)
    assert asymboost.decision_function(X_test).shape[1] == len(classes)

    score = asymboost.score(X_test, y_test)
    assert score > 0.6, f"Failed with algorithm {algorithm} and score {score}"

    y_pred = asymboost.predict(X_test)
    assert y_pred.shape == y_test.shape


@pytest.mark.parametrize("algorithm", ["SAMME"])
def test_sample_weight(imbalanced_dataset, algorithm):
    X, y = imbalanced_dataset
    sample_weight = np.ones_like(y)
    asymboost = AsymBoostClassifier(algorithm=algorithm, random_state=0)

    # Predictions should be the same when sample_weight are all ones
    y_pred_sample_weight = asymboost.fit(
        X,
        y,
        sample_weight=sample_weight,
        cost_matrix="uniform",
    ).predict(X)
    y_pred_no_sample_weight = asymboost.fit(
        X,
        y,
        cost_matrix="uniform",
    ).predict(X)

    assert_array_equal(y_pred_sample_weight, y_pred_no_sample_weight)

    rng = np.random.RandomState(42)
    sample_weight = rng.rand(y.shape[0])
    y_pred_sample_weight = asymboost.fit(
        X,
        y,
        sample_weight=sample_weight,
        cost_matrix="uniform",
    ).predict(X)

    with pytest.raises(AssertionError):
        assert_array_equal(y_pred_no_sample_weight, y_pred_sample_weight)


@pytest.mark.parametrize("algorithm", ["SAMME"])
@pytest.mark.parametrize(
    "cost_matrix", [None, "uniform", "inverse", "log1p-inverse", "random"]
)
def test_cost_matrix(imbalanced_dataset, algorithm, cost_matrix):
    expected_cost_matrixs = {
        "uniform": np.array(
            [
                [1.00, 1.00, 1.00],
                [1.00, 1.00, 1.00],
                [1.00, 1.00, 1.00],
            ]
        ),
        "inverse": np.array(
            [
                [1.00, 0.24, 0.01],
                [4.12, 1.00, 0.06],
                [72.20, 17.51, 1.00],
            ]
        ),
        "log1p-inverse": np.array(
            [
                [0.69, 0.22, 0.01],
                [1.63, 0.69, 0.06],
                [4.29, 2.92, 0.69],
            ]
        ),
    }

    X, y = imbalanced_dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, random_state=1
    )
    classes = np.unique(y)

    n_estimators = 50
    asymboost = AsymBoostClassifier(
        n_estimators=n_estimators, algorithm=algorithm, random_state=0
    )

    if cost_matrix == "log1p-inverse":
        with pytest.raises(ValueError, match="'log1p-inverse' option is not available"):
            asymboost.fit(X_train, y_train, cost_matrix=cost_matrix)
    elif cost_matrix in [None, "uniform", "inverse"]:
        asymboost.fit(X_train, y_train, cost_matrix=cost_matrix)
        assert_array_equal(classes, asymboost.classes_)

        if cost_matrix is None:
            expected_cost_matrix = expected_cost_matrixs["inverse"]
        else:
            expected_cost_matrix = expected_cost_matrixs[cost_matrix]
        assert (asymboost.cost_matrix_.round(2) == expected_cost_matrix).all()
    else:
        with pytest.raises(ValueError, match="When 'cost_matrix' is string"):
            asymboost.fit(X_train, y_train, cost_matrix=cost_matrix)


@pytest.mark.parametrize(
    "cost_matrix",
    [
        np.array(
            [
                [1.00, 0.24, 0.01],
                [4.12, 1.00, 0.06],
                [72.20, 17.51, 1.00],
            ]
        ),
        np.array(
            [
                [1.00, 0.24, 0.01],
            ]
        ),
        np.array(
            [
                [1.00, 0.24, "a"],
                [4.12, 1.00, 0.06],
                [72.20, 17.51, 1.00],
            ]
        ),
        10,
    ],
)
def test_cost_matrix_customize(imbalanced_dataset, cost_matrix):
    X, y = imbalanced_dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, random_state=1
    )

    asymboost = AsymBoostClassifier(random_state=0)
    if isinstance(cost_matrix, np.ndarray):
        if cost_matrix.dtype != float:
            with pytest.raises(ValueError, match="dtype='numeric' is not compatible"):
                asymboost.fit(X_train, y_train, cost_matrix=cost_matrix)
        elif cost_matrix.shape != (3, 3):
            with pytest.raises(ValueError, match="it should be of shape"):
                asymboost.fit(X_train, y_train, cost_matrix=cost_matrix)
        else:
            asymboost.fit(X_train, y_train, cost_matrix=cost_matrix)
    else:
        with pytest.raises(ValueError, match="Expected 2D array, got scalar array"):
            asymboost.fit(X_train, y_train, cost_matrix=cost_matrix)
