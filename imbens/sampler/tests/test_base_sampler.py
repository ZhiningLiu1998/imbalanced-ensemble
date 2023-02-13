"""Test the module base sampler."""

# Authors: Zhining Liu <zhining.liu@outlook.com>
# License: MIT

from collections import Counter

import numpy as np
import pytest
from sklearn.utils._testing import assert_array_equal

from imbens.sampler._under_sampling import RandomUnderSampler
from imbens.sampler.base import FunctionSampler, SamplerMixin

RND_SEED = 0
X = np.array(
    [
        [2.45166, 1.86760],
        [1.34450, -1.30331],
        [1.02989, 2.89408],
        [-1.94577, -1.75057],
        [1.21726, 1.90146],
        [2.00194, 1.25316],
        [2.31968, 2.33574],
        [1.14769, 1.41303],
        [1.32018, 2.17595],
        [-1.74686, -1.66665],
        [-2.17373, -1.91466],
        [2.41436, 1.83542],
        [1.97295, 2.55534],
        [-2.12126, -2.43786],
        [1.20494, 3.20696],
        [-2.30158, -2.39903],
        [1.76006, 1.94323],
        [2.35825, 1.77962],
        [-2.06578, -2.07671],
        [0.00245, -0.99528],
    ]
)
y = np.array([2, 0, 2, 1, 2, 2, 2, 2, 2, 1, 1, 2, 2, 1, 2, 1, 2, 2, 1, 0])

classes_, _ = np.unique(y, return_inverse=True)
encode_map = {c: np.where(classes_ == c)[0][0] for c in classes_}
sample_weight = np.full_like(y, fill_value=1 / y.shape[0], dtype=float)
sampling_strategy_default = {2: 2, 1: 2, 0: 2}
sampling_strategy_normal = {2: 6, 1: 4, 0: 2}
sampling_strategy_error = {2: 200, 1: 100, 0: 90}


sample_weight_normal = np.ones_like(y)
sample_weight_size = np.ones(2)
sample_weight_type = 'string error type'


@pytest.mark.parametrize("as_frame", [True, False], ids=["dataframe", "array"])
@pytest.mark.parametrize(
    "sample_weight",
    [None, sample_weight_normal, sample_weight_size, sample_weight_type],
    ids=["None", "normal", "size", "type"],
)
def test_sampler_mixin(as_frame, sample_weight):
    if as_frame:
        pd = pytest.importorskip("pandas")
        X_ = pd.DataFrame(X)
    else:
        X_ = X

    # init resampler
    sampler = RandomUnderSampler(
        random_state=RND_SEED,
    )

    # check fit method
    sampler = sampler.fit(X_, y)

    # check fit_resample method
    if sample_weight is None:
        X_resampled, y_resampled = sampler.fit_resample(
            X_,
            y,
            sample_weight=sample_weight,
        )
        if as_frame:
            assert hasattr(X_resampled, "loc")
            X_resampled = X_resampled.to_numpy()
        assert X_resampled.shape == (6, 2)
        assert y_resampled.shape == (6,)
        assert dict(Counter(y_resampled)) == sampling_strategy_default

    elif not isinstance(sample_weight, np.ndarray):
        with pytest.raises(ValueError, match="should be an array-like"):
            X_resampled, y_resampled = sampler.fit_resample(
                X_,
                y,
                sample_weight=sample_weight,
            )

    elif sample_weight.shape != y.shape:
        with pytest.raises(ValueError, match="sample_weight.shape"):
            X_resampled, y_resampled = sampler.fit_resample(
                X_,
                y,
                sample_weight=sample_weight,
            )

    else:
        X_resampled, y_resampled, w_resampled = sampler.fit_resample(
            X_,
            y,
            sample_weight=sample_weight,
        )
        if as_frame:
            if hasattr(X_resampled, "loc"):
                X_resampled = X_resampled.to_numpy()
        assert X_resampled.shape == (6, 2)
        assert y_resampled.shape == (6,)
        assert w_resampled.shape == (6,)
        assert dict(Counter(y_resampled)) == sampling_strategy_default


def dummy_sampling_func(X, y, sample_weight):
    return X, y, sample_weight


@pytest.mark.parametrize(
    "sample_weight",
    [None, sample_weight_normal, sample_weight_size, sample_weight_type],
    ids=["None", "normal", "size", "type"],
)
def test_func_sampler(sample_weight):
    func_sampler = FunctionSampler(
        func=dummy_sampling_func,
        kw_args={'sample_weight': sample_weight},
        accept_sparse=True,
        validate=True,
    )
    func_sampler.fit(X, y)
    X_resampled, y_resampled, w_resampled = func_sampler.fit_resample(X, y)
    assert X_resampled.shape == X.shape
    assert y_resampled.shape == y.shape
    if sample_weight is None:
        assert w_resampled is sample_weight
    else:
        assert_array_equal(w_resampled, sample_weight)
    assert dict(Counter(y_resampled)) == dict(Counter(y))
