"""Test SelfPacedEnsembleClassifier."""

# Authors: Zhining Liu <zhining.liu@outlook.com>
# License: MIT

import pytest
import sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.utils.fixes import parse_version

from imbens.datasets import make_imbalance
from imbens.utils.testing import all_estimators

sklearn_version = parse_version(sklearn.__version__)
iris = load_iris()
all_ensembles = all_estimators(type_filter='ensemble')

X, y = make_imbalance(
    iris.data,
    iris.target,
    sampling_strategy={0: 20, 1: 25, 2: 50},
    random_state=0,
)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=0)
init_param = {'random_state': 0, 'n_estimators': 20}


@pytest.mark.parametrize(
    "ensemble",
    all_ensembles,
)
def test_evaluate(ensemble):
    """Check classification with dynamic logging."""
    (ensemble_name, EnsembleCLass) = ensemble
    clf = EnsembleCLass(**init_param)
    clf.fit(
        X_train,
        y_train,
        train_verbose=True,
    )
    clf._evaluate('train', return_value_dict=True)


@pytest.mark.parametrize(
    "ensemble",
    all_ensembles,
)
def test_evaluate_verbose(ensemble):
    """Check classification with dynamic logging."""
    (ensemble_name, EnsembleCLass) = ensemble
    clf = EnsembleCLass(**init_param)
    if clf._properties['training_type'] == 'parallel':
        with pytest.raises(TypeError, match="can only be of type `bool`"):
            clf.fit(
                X_train,
                y_train,
                train_verbose={
                    'granularity': 10,
                },
            )
    else:
        clf.fit(
            X_train,
            y_train,
            train_verbose={
                'granularity': 10,
            },
        )
        clf.fit(
            X_train,
            y_train,
            train_verbose={
                'granularity': 10,
                'print_distribution': False,
                'print_metrics': True,
            },
        )


@pytest.mark.parametrize(
    "ensemble",
    all_ensembles,
)
def test_evaluate_eval_datasets(ensemble):
    """Check classification with dynamic logging."""
    (ensemble_name, EnsembleCLass) = ensemble
    clf = EnsembleCLass(**init_param)
    if clf._properties['training_type'] == 'parallel':
        with pytest.raises(TypeError, match="can only be of type `bool`"):
            clf.fit(
                X_train,
                y_train,
                eval_datasets={
                    'valid': (X_valid, y_valid),  # add validation data
                },
                train_verbose={
                    'granularity': 10,
                },
            )
    else:
        clf.fit(
            X_train,
            y_train,
            eval_datasets={
                'valid': (X_valid, y_valid),  # add validation data
            },
            train_verbose={
                'granularity': 10,
            },
        )


@pytest.mark.parametrize(
    "ensemble",
    all_ensembles,
)
def test_evaluate_eval_metrics(ensemble):
    """Check classification with dynamic logging."""
    (ensemble_name, EnsembleCLass) = ensemble
    clf = EnsembleCLass(**init_param)
    clf.fit(
        X_train,
        y_train,
        eval_datasets={
            'valid': (X_valid, y_valid),
        },
        eval_metrics={
            'weighted_f1': (
                sklearn.metrics.f1_score,
                {'average': 'weighted'},
            ),  # use weighted_f1
            'roc': (
                sklearn.metrics.roc_auc_score,
                {'multi_class': 'ovr', 'average': 'macro'},
            ),  # use roc_auc score
        },
        train_verbose=True,
    )
