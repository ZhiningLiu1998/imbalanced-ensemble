"""Test validation_param module."""

# Authors: Zhining Liu <zhining.liu@outlook.com>
# License: MIT

import numpy as np
import pytest
import sklearn
from sklearn.datasets import load_iris
from sklearn.utils.fixes import parse_version

from imbens.datasets import make_imbalance
from imbens.utils._validation_param import (
    BALANCING_KIND,
    SAMPLING_KIND,
    _check_n_target_samples_dict,
    _check_n_target_samples_int,
    check_balancing_schedule,
)

sklearn_version = parse_version(sklearn.__version__)
iris = load_iris()

X, y = make_imbalance(
    iris.data,
    iris.target,
    sampling_strategy={0: 20, 1: 25, 2: 50},
    random_state=0,
)

classes_, _ = np.unique(y, return_inverse=True)
encode_map = {c: np.where(classes_ == c)[0][0] for c in classes_}
dropped_index = np.zeros_like(y).astype(bool)
dropped_index_empty = np.ones_like(y).astype(bool)
sample_weight = np.full_like(y, fill_value=1 / y.shape[0], dtype=float)
sampling_strategy_org = {0: 20, 1: 25, 2: 50}
# sampling_strategy_default = {2: 2, 1: 2, 0: 2}
# sampling_strategy_normal = {2: 6, 1: 4, 0: 2}
# sampling_strategy_error = {2: 200, 1: 100, 0: 90}

n_min, n_max = 20, 50


@pytest.mark.parametrize("n_target_samples", [-10, 0, 5, 25, 100])
@pytest.mark.parametrize(
    "sampling_type",
    ["over-sampling", "under-sampling", "multi-class-hybrid-sampling", "NOTEXIST"],
)
def test_check_n_target_samples_int(n_target_samples, sampling_type):
    if n_target_samples <= 0:
        with pytest.raises(ValueError, match="must be positive"):
            _check_n_target_samples_int(y, n_target_samples, sampling_type)
    elif sampling_type not in SAMPLING_KIND:
        with pytest.raises(NotImplementedError, match="must be one of"):
            _check_n_target_samples_int(y, n_target_samples, sampling_type)
    elif sampling_type == "under-sampling":
        if n_target_samples > n_max:
            with pytest.raises(ValueError, match="perform under-sampling properly"):
                _check_n_target_samples_int(y, n_target_samples, sampling_type)
        else:
            _check_n_target_samples_int(y, n_target_samples, sampling_type)
    elif sampling_type == "over-sampling":
        if n_target_samples < n_min:
            with pytest.raises(ValueError, match="perform over-sampling properly"):
                _check_n_target_samples_int(y, n_target_samples, sampling_type)
        else:
            _check_n_target_samples_int(y, n_target_samples, sampling_type)
    elif sampling_type == "multi-class-hybrid-sampling":
        if n_target_samples > n_min and n_target_samples < n_max:
            _check_n_target_samples_int(y, n_target_samples, sampling_type)
        else:
            with pytest.raises(Warning, match="n_target_samples"):
                _check_n_target_samples_int(y, n_target_samples, sampling_type)


target_under = {2: 2, 1: 2, 0: 2}
target_over = {2: 200, 1: 100, 0: 90}
target_hybrid = {2: 25, 1: 25, 0: 25}
target_neg = {2: -20, 1: 2, 0: 2}
target_zero = {2: 0, 1: 2, 0: 2}
target_wrongkey = {2: 20, 1: 2, 0: 2, 5: 2}


@pytest.mark.parametrize(
    "n_target_samples",
    [
        target_under,
        target_over,
        target_hybrid,
        target_neg,
        target_zero,
        target_wrongkey,
    ],
)
@pytest.mark.parametrize(
    "sampling_type",
    ["over-sampling", "under-sampling", "multi-class-hybrid-sampling", "NOTEXIST"],
)
def test_check_n_target_samples_dict(n_target_samples, sampling_type):
    if n_target_samples == target_wrongkey:
        with pytest.raises(ValueError, match="not present in the data"):
            _check_n_target_samples_dict(y, n_target_samples, sampling_type)
    elif n_target_samples == target_neg or n_target_samples == target_zero:
        with pytest.raises(ValueError, match="class must > 0"):
            _check_n_target_samples_dict(y, n_target_samples, sampling_type)
    elif sampling_type not in SAMPLING_KIND:
        with pytest.raises(NotImplementedError, match="must be one of"):
            _check_n_target_samples_dict(y, n_target_samples, sampling_type)
    elif sampling_type == "under-sampling":
        if n_target_samples != target_under:
            with pytest.raises(ValueError, match="perform under-sampling"):
                _check_n_target_samples_dict(y, n_target_samples, sampling_type)
        else:
            _check_n_target_samples_dict(y, n_target_samples, sampling_type)
    elif sampling_type == "over-sampling":
        if n_target_samples != target_over:
            with pytest.raises(ValueError, match="perform over-sampling"):
                _check_n_target_samples_dict(y, n_target_samples, sampling_type)
        else:
            _check_n_target_samples_dict(y, n_target_samples, sampling_type)
    elif sampling_type == "multi-class-hybrid-sampling":
        if n_target_samples != target_hybrid:
            with pytest.raises(Warning, match="-sampling will be carried out."):
                _check_n_target_samples_dict(y, n_target_samples, sampling_type)
        else:
            _check_n_target_samples_dict(y, n_target_samples, sampling_type)


def dummy_schedule(
    origin_distr: dict, target_distr: dict, i_estimator: int, total_estimator: int
):
    '''A dummy resampling schedule'''
    return origin_distr


def error_schedule(
    origin_distr: dict, target_distr: dict, i_estimator: int, total_estimator: int
):
    '''A dummy resampling schedule'''
    raise RuntimeError('error in self defined schedule')


@pytest.mark.parametrize(
    "balancing_schedule",
    [
        "uniform",
        "progressive",
        dummy_schedule,
        error_schedule,
        "unmatch_string",
        ('wrong_type'),
    ],
)
def test_check_balancing_schedule(balancing_schedule):
    if callable(balancing_schedule):
        if balancing_schedule is error_schedule:
            with pytest.raises(RuntimeError, match="self-defined `balancing_schedule`"):
                check_balancing_schedule(balancing_schedule)
        else:
            check_balancing_schedule(balancing_schedule)
    elif balancing_schedule in BALANCING_KIND:
        check_balancing_schedule(balancing_schedule)
    else:
        with pytest.raises(TypeError, match="'balancing_schedule' should be one of"):
            check_balancing_schedule(balancing_schedule)
