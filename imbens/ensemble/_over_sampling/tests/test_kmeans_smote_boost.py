"""Test KmeansSMOTEBoostClassifier."""

# Authors: Zhining Liu <zhining.liu@outlook.com>
# License: MIT

import numpy as np
import pytest
import sklearn
from sklearn.datasets import load_iris, make_classification
from sklearn.model_selection import train_test_split
from sklearn.utils._testing import assert_array_equal
from sklearn.utils.fixes import parse_version

from imbens.ensemble import KmeansSMOTEBoostClassifier

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
        random_state=3,
    )


@pytest.mark.parametrize("algorithm", ["SAMME"])
def test_ksmoteboost(imbalanced_dataset, algorithm):
    X, y = imbalanced_dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, random_state=1
    )
    classes = np.unique(y)

    n_estimators = 100
    ksmoteboost = KmeansSMOTEBoostClassifier(
        n_estimators=n_estimators, algorithm=algorithm, random_state=0
    )
    ksmoteboost.fit(X_train, y_train)
    assert_array_equal(classes, ksmoteboost.classes_)

    # check that we have an ensemble of samplers and estimators with a
    # consistent size
    assert len(ksmoteboost.estimators_) > 1
    assert len(ksmoteboost.estimators_) == len(ksmoteboost.samplers_)

    # each sampler in the ensemble should have different random state
    assert len({sampler.random_state for sampler in ksmoteboost.samplers_}) == len(
        ksmoteboost.samplers_
    )
    # each estimator in the ensemble should have different random state
    assert len({est.random_state for est in ksmoteboost.estimators_}) == len(
        ksmoteboost.estimators_
    )

    # check the consistency of the feature importances
    assert len(ksmoteboost.feature_importances_) == imbalanced_dataset[0].shape[1]

    # check the consistency of the prediction outpus
    y_pred = ksmoteboost.predict_proba(X_test)
    assert y_pred.shape[1] == len(classes)
    assert ksmoteboost.decision_function(X_test).shape[1] == len(classes)

    score = ksmoteboost.score(X_test, y_test)
    assert score > 0.6, f"Failed with algorithm {algorithm} and score {score}"

    y_pred = ksmoteboost.predict(X_test)
    assert y_pred.shape == y_test.shape


@pytest.mark.parametrize("algorithm", ["SAMME"])
def test_smoteboost_sample_weight(imbalanced_dataset, algorithm):
    X, y = imbalanced_dataset
    sample_weight = np.ones_like(y)
    ksmoteboost = KmeansSMOTEBoostClassifier(algorithm=algorithm, random_state=0)

    # Predictions should be the same when sample_weight are all ones
    y_pred_sample_weight = ksmoteboost.fit(X, y, sample_weight=sample_weight).predict(X)
    y_pred_no_sample_weight = ksmoteboost.fit(X, y).predict(X)

    assert_array_equal(y_pred_sample_weight, y_pred_no_sample_weight)

    rng = np.random.RandomState(42)
    sample_weight = rng.rand(y.shape[0])
    y_pred_sample_weight = ksmoteboost.fit(X, y, sample_weight=sample_weight).predict(X)

    with pytest.raises(AssertionError):
        assert_array_equal(y_pred_no_sample_weight, y_pred_sample_weight)


@pytest.mark.parametrize("algorithm", ["SAMME"])
@pytest.mark.parametrize("k_neighbors", [-5, 0, 5, 500, "string"])
def test_smoteboost_k_neighbors(imbalanced_dataset, algorithm, k_neighbors):
    X, y = imbalanced_dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, random_state=1
    )
    if type(k_neighbors) != int:
        with pytest.raises(TypeError, match="'k_neighbors' should be of type"):
            ksmoteboost = KmeansSMOTEBoostClassifier(
                k_neighbors=k_neighbors, algorithm=algorithm, random_state=0
            ).fit(X_train, y_train)
    else:
        if k_neighbors < 1:
            with pytest.raises(ValueError, match="must be an int in the range"):
                ksmoteboost = KmeansSMOTEBoostClassifier(
                    k_neighbors=k_neighbors, algorithm=algorithm, random_state=0
                ).fit(X_train, y_train)
        elif k_neighbors > 100:
            with pytest.raises(RuntimeError, match="No clusters found with sufficient"):
                ksmoteboost = KmeansSMOTEBoostClassifier(
                    k_neighbors=k_neighbors, algorithm=algorithm, random_state=0
                ).fit(X_train, y_train)
        else:
            ksmoteboost = KmeansSMOTEBoostClassifier(
                k_neighbors=k_neighbors, algorithm=algorithm, random_state=0
            ).fit(X_train, y_train)
