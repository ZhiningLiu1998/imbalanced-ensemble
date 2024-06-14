"""Test BalanceCascadeClassifier."""

# Authors: Zhining Liu <zhining.liu@outlook.com>
# License: MIT

import numpy as np
import pytest
import sklearn
from scipy import sparse
from sklearn.datasets import load_iris
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.fixes import parse_version

from imbens.datasets import make_imbalance
from imbens.ensemble import BalanceCascadeClassifier

sklearn_version = parse_version(sklearn.__version__)
iris = load_iris()

X, y = make_imbalance(
    iris.data,
    iris.target,
    sampling_strategy={0: 20, 1: 25, 2: 50},
    random_state=0,
)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


@pytest.mark.parametrize(
    "estimator",
    [
        None,
        DummyClassifier(strategy="prior"),
        DecisionTreeClassifier(),
        KNeighborsClassifier(),
        SVC(gamma="scale", probability=True),
    ],
)
@pytest.mark.parametrize(
    "params",
    ParameterGrid(
        {
            "n_jobs": [1, 5],
            "replacement": [False, True],
        }
    ),
)
def test_estimator_init(estimator, params):
    """Check classification for various parameter settings."""
    bcc = BalanceCascadeClassifier(estimator=estimator, random_state=0, **params).fit(
        X_train, y_train
    )
    bcc.predict(X_test)
    bcc.predict_proba(X_test)
    bcc.score(X_test, y_test)
    if hasattr(estimator, "decision_function"):
        bcc.decision_function(X_test)


@pytest.mark.parametrize(
    "fit_params",
    ParameterGrid(
        {
            "sample_weight": [None, np.ones(71)],
            "target_label": [0, 1, 2, -1, 42],
            "balancing_schedule": ["uniform", "progressive"],
        }
    ),
)
def test_fit_target_label(fit_params):
    """Check classification for various fit parameter settings."""
    expect_target_distrs = {
        0: {1: 14, 2: 14, 0: 14},
        1: {1: 19, 2: 19, 0: 14},
        2: {1: 19, 2: 38, 0: 14},
    }
    if fit_params["target_label"] in [0, 1, 2]:
        if (
            fit_params["balancing_schedule"] == "uniform"
            or fit_params["target_label"] == 2
        ):
            bcc = BalanceCascadeClassifier(random_state=0).fit(
                X_train, y_train, **fit_params
            )
            expect_target_distr = expect_target_distrs[fit_params["target_label"]]
            assert bcc.target_distr_ == expect_target_distr
            bcc.predict(X_test)
            bcc.predict_proba(X_test)
            bcc.score(X_test, y_test)
        else:
            with pytest.raises(
                ValueError, match="only support static target sample distribution"
            ):
                bcc = BalanceCascadeClassifier(random_state=0).fit(
                    X_train, y_train, **fit_params
                )
    else:
        with pytest.raises(ValueError, match="is not present in the data"):
            bcc = BalanceCascadeClassifier(random_state=0).fit(
                X_train, y_train, **fit_params
            )


@pytest.mark.parametrize(
    "fit_params",
    ParameterGrid(
        {
            "sample_weight": [None, np.ones(71)],
            "n_target_samples": [10, 20, 50, -5],
            "balancing_schedule": ["uniform", "progressive"],
        }
    ),
)
def test_fit_target_samples_int(fit_params):
    """Check classification for various fit parameter settings."""
    expect_target_distrs = {
        10: {1: 10, 2: 10, 0: 10},
        20: {1: 19, 2: 20, 0: 14},
    }

    if fit_params["balancing_schedule"] == "uniform":
        if fit_params["n_target_samples"] > 38:
            with pytest.raises(
                ValueError, match="'n_target_samples' > the number of samples"
            ):
                bcc = BalanceCascadeClassifier(random_state=0).fit(
                    X_train, y_train, **fit_params
                )
        elif fit_params["n_target_samples"] <= 0:
            with pytest.raises(ValueError, match="'n_target_samples' must be positive"):
                bcc = BalanceCascadeClassifier(random_state=0).fit(
                    X_train, y_train, **fit_params
                )
        else:
            bcc = BalanceCascadeClassifier(random_state=0).fit(
                X_train, y_train, **fit_params
            )
            expect_target_distr = expect_target_distrs[fit_params["n_target_samples"]]
            assert bcc.target_distr_ == expect_target_distr
            bcc.predict(X_test)
            bcc.predict_proba(X_test)
            bcc.score(X_test, y_test)
    else:
        if fit_params["n_target_samples"] > 38:
            with pytest.raises(
                ValueError, match="'n_target_samples' > the number of samples"
            ):
                bcc = BalanceCascadeClassifier(random_state=0).fit(
                    X_train, y_train, **fit_params
                )
        elif fit_params["n_target_samples"] <= 0:
            with pytest.raises(ValueError, match="'n_target_samples' must be positive"):
                bcc = BalanceCascadeClassifier(random_state=0).fit(
                    X_train, y_train, **fit_params
                )
        else:
            with pytest.raises(
                ValueError, match="only support static target sample distribution"
            ):
                bcc = BalanceCascadeClassifier(random_state=0).fit(
                    X_train, y_train, **fit_params
                )


@pytest.mark.parametrize("n_target_samples_idx", [0, 1, 2, 3, 4])
@pytest.mark.parametrize(
    "fit_params",
    ParameterGrid(
        {
            "sample_weight": [None, np.ones(71)],
            "balancing_schedule": ["uniform", "progressive"],
        }
    ),
)
def test_fit_target_samples_dict(n_target_samples_idx, fit_params):
    """Check classification for various fit parameter settings."""
    input_n_target_samples_dict = {
        0: {1: 5, 2: 10},
        1: {0: 5, 1: 5, 2: 10},
        2: {0: 5, 1: 5, 2: 0},
        3: {0: 5, 1: 5, 2: -10},
        4: {0: 5, 1: 5, 2: 50},
    }
    expect_target_distrs = {
        0: {1: 5, 2: 10, 0: 14},
        1: {0: 5, 1: 5, 2: 10},
    }
    fit_params["n_target_samples"] = input_n_target_samples_dict[n_target_samples_idx]

    if fit_params["balancing_schedule"] == "uniform":
        if n_target_samples_idx in [0, 1]:
            bcc = BalanceCascadeClassifier(random_state=0).fit(
                X_train, y_train, **fit_params
            )
            expect_target_distr = expect_target_distrs[n_target_samples_idx]
            assert bcc.target_distr_ == expect_target_distr
            bcc.predict(X_test)
            bcc.predict_proba(X_test)
            bcc.score(X_test, y_test)
        elif n_target_samples_idx in [2, 3]:
            with pytest.raises(
                ValueError, match="The number of samples in a class must > 0"
            ):
                bcc = BalanceCascadeClassifier(random_state=0).fit(
                    X_train, y_train, **fit_params
                )
        elif n_target_samples_idx in [4]:
            with pytest.raises(
                ValueError, match="The target number of samples of class"
            ):
                bcc = BalanceCascadeClassifier(random_state=0).fit(
                    X_train, y_train, **fit_params
                )
    else:
        if n_target_samples_idx in [0, 1]:
            with pytest.raises(
                ValueError, match="only support static target sample distribution"
            ):
                bcc = BalanceCascadeClassifier(random_state=0).fit(
                    X_train, y_train, **fit_params
                )
        elif n_target_samples_idx in [2, 3]:
            with pytest.raises(
                ValueError, match="The number of samples in a class must > 0"
            ):
                bcc = BalanceCascadeClassifier(random_state=0).fit(
                    X_train, y_train, **fit_params
                )
        elif n_target_samples_idx in [4]:
            with pytest.raises(
                ValueError, match="The target number of samples of class"
            ):
                bcc = BalanceCascadeClassifier(random_state=0).fit(
                    X_train, y_train, **fit_params
                )


@pytest.mark.parametrize("n_target_samples", [10, {2: 10}], ids=["int", "dict"])
def test_fit_target_label_target_samples_error(n_target_samples):
    """Ensure we raise an error when set both target_label and n_target_samples"""
    with pytest.raises(ValueError, match="cannot be specified at the same time"):
        bcc = BalanceCascadeClassifier(random_state=0).fit(
            X_train,
            y_train,
            target_label=0,
            n_target_samples=n_target_samples,
        )
