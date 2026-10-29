"""Random-state contracts for iterative under-sampling ensembles."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier

from imbens.ensemble import BalanceCascadeClassifier, SelfPacedEnsembleClassifier


ENSEMBLES = [SelfPacedEnsembleClassifier, BalanceCascadeClassifier]


@pytest.mark.parametrize("ensemble_class", ENSEMBLES)
@pytest.mark.parametrize("labels", [[0, 1], [10, 20], ["majority", "minority"]])
def test_integer_random_state_refit_and_clone(ensemble_class, labels):
    X = np.arange(24, dtype=float).reshape(12, 2)
    y = np.asarray(labels)[[0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1]]
    estimator = DecisionTreeClassifier(max_depth=1)
    ensemble = ensemble_class(
        estimator=estimator, n_estimators=2, random_state=7, n_jobs=1
    )
    reference = clone(ensemble).fit(X, y)
    ensemble.fit(X, y)
    cloned = clone(ensemble).fit(X, y)
    ensemble.fit(X, y)

    # An integer seed must survive fitting so refits and clones reproduce
    # the fresh estimator's training draws and fitted predictions.
    for fitted in [ensemble, cloned]:
        assert fitted.get_params()["random_state"] == 7
        assert [e.random_state for e in fitted.estimators_] == [
            e.random_state for e in reference.estimators_
        ]
        for actual, expected in zip(fitted.samplers_, reference.samplers_):
            assert_array_equal(actual.sample_indices_, expected.sample_indices_)
        assert_allclose(fitted.predict_proba(X), reference.predict_proba(X))
        assert_array_equal(fitted.predict(X), reference.predict(X))
        assert_array_equal(fitted.classes_, np.unique(y))
    assert not hasattr(estimator, "tree_")


@pytest.mark.parametrize("ensemble_class", ENSEMBLES)
def test_random_state_instance_remains_stateful(ensemble_class):
    X = np.arange(24, dtype=float).reshape(12, 2)
    y = np.array([0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1])
    rng = np.random.RandomState(7)
    ensemble = ensemble_class(n_estimators=2, random_state=rng, n_jobs=1)
    ensemble.fit(X, y)
    first_seeds = [e.random_state for e in ensemble.estimators_]
    ensemble.fit(X, y)
    assert ensemble.random_state is rng
    assert first_seeds != [e.random_state for e in ensemble.estimators_]
