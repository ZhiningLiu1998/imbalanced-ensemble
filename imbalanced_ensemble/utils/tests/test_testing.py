"""Test for the testing module"""
# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Christos Aridas
# License: MIT

import pytest

from imbalanced_ensemble.sampler.base import SamplerMixin
from imbalanced_ensemble.utils.testing import all_estimators


def test_all_estimators():
    # check if the filtering is working with a list or a single string
    type_filter = "sampler"
    all_estimators(type_filter=type_filter)
    type_filter = ["sampler"]
    estimators = all_estimators(type_filter=type_filter)
    for estimator in estimators:
        # check that all estimators are sampler
        assert issubclass(estimator[1], SamplerMixin)

    # check that an error is raised when the type is unknown
    type_filter = "rnd"
    with pytest.raises(ValueError, match="Parameter type_filter must be 'sampler'"):
        all_estimators(type_filter=type_filter)