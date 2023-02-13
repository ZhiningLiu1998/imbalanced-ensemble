"""Test the module self-paced under sampler."""

# Authors: Zhining Liu <zhining.liu@outlook.com>
# License: MIT

from collections import Counter

import numpy as np
import pytest
from sklearn.utils._testing import assert_array_equal

from imbens.sampler._under_sampling import SelfPacedUnderSampler

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

pred_proba_normal = np.array(
    [
        [0.29399155, 0.38311672, 0.32289173],
        [0.33750765, 0.26241723, 0.40007512],
        [0.1908342, 0.38890714, 0.42025866],
        [0.22501625, 0.46461061, 0.31037314],
        [0.36304264, 0.59155755, 0.04539982],
        [0.09269394, 0.02150968, 0.88579638],
        [0.29623897, 0.3312077, 0.37255333],
        [0.3915204, 0.22608603, 0.38239357],
        [0.13119027, 0.70980192, 0.15900781],
        [0.5021685, 0.2774049, 0.22042661],
        [0.17696742, 0.51790298, 0.3051296],
        [0.47178453, 0.01559502, 0.51262046],
        [0.28171115, 0.28393791, 0.43435094],
        [0.4612004, 0.24318019, 0.29561941],
        [0.48969517, 0.04227466, 0.46803016],
        [0.66403291, 0.20831055, 0.12765653],
        [0.25247682, 0.29112329, 0.4563999],
        [0.28685136, 0.64640994, 0.0667387],
        [0.20412182, 0.15763742, 0.63824076],
        [0.262743, 0.48371083, 0.25354616],
    ]
)
pred_proba_det = np.array(
    [
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 1],
    ]
)
classes_, _ = np.unique(y, return_inverse=True)
encode_map = {c: np.where(classes_ == c)[0][0] for c in classes_}
sample_weight = np.full_like(y, fill_value=1 / y.shape[0], dtype=float)
sampling_strategy_default = {2: 2, 1: 2, 0: 2}
sampling_strategy_normal = {2: 6, 1: 4, 0: 2}
sampling_strategy_error = {2: 200, 1: 100, 0: 90}


@pytest.mark.parametrize("as_frame", [True, False], ids=["dataframe", "array"])
@pytest.mark.parametrize("proba_det", [True, False], ids=["det", "normal"])
def test_spus_fit_resample(as_frame, proba_det):
    if as_frame:
        pd = pytest.importorskip("pandas")
        X_ = pd.DataFrame(X)
    else:
        X_ = X

    # prepare y_pred_proba
    if proba_det:
        y_pred_proba = pred_proba_det
    else:
        y_pred_proba = pred_proba_normal

    # init resampler
    spus = SelfPacedUnderSampler(
        k_bins=5,
        soft_resample_flag=True,
        replacement=False,
        random_state=RND_SEED,
    )

    # resampling
    X_resampled, y_resampled = spus.fit_resample(
        X_,
        y,
        alpha=0,
        y_pred_proba=y_pred_proba,
        classes_=classes_,
        encode_map=encode_map,
    )

    if as_frame:
        assert hasattr(X_resampled, "loc")
        X_resampled = X_resampled.to_numpy()

    assert X_resampled.shape == (6, 2)
    assert y_resampled.shape == (6,)
    assert dict(Counter(y_resampled)) == sampling_strategy_default


def test_spus_negative_alpha():
    spus = SelfPacedUnderSampler(
        k_bins=5,
        soft_resample_flag=True,
        replacement=False,
        random_state=RND_SEED,
    )
    with pytest.raises(ValueError, match="must not be negative"):
        X_resampled, y_resampled = spus.fit_resample(
            X,
            y,
            alpha=-1,
            y_pred_proba=pred_proba_normal,
            classes_=classes_,
            encode_map=encode_map,
        )


def test_spus_negative_k_bins():
    with pytest.raises(ValueError, match="'k_bins' should be > 0"):
        spus = SelfPacedUnderSampler(
            k_bins=-5,
            soft_resample_flag=True,
            replacement=False,
            random_state=RND_SEED,
        )


@pytest.mark.parametrize(
    "invalid_target", [True, False], ids=["invalid_target", "valid_target"]
)
def test_spus_no_sufficient_data(invalid_target):
    if invalid_target:
        sampling_strategy_ = sampling_strategy_error
    else:
        sampling_strategy_ = sampling_strategy_normal

    spus = SelfPacedUnderSampler(
        sampling_strategy=sampling_strategy_,
        k_bins=5,
        soft_resample_flag=True,
        replacement=False,
        random_state=RND_SEED,
    )
    if invalid_target:
        with pytest.raises(ValueError, match="With under-sampling methods"):
            X_resampled, y_resampled = spus.fit_resample(
                X,
                y,
                alpha=0,
                y_pred_proba=pred_proba_normal,
                classes_=classes_,
                encode_map=encode_map,
            )
    else:
        X_resampled, y_resampled = spus.fit_resample(
            X,
            y,
            alpha=0,
            y_pred_proba=pred_proba_normal,
            classes_=classes_,
            encode_map=encode_map,
        )
        assert y_resampled.shape[0] == 12


def test_spus_with_sample_weight():
    spus = SelfPacedUnderSampler(
        k_bins=5,
        soft_resample_flag=True,
        replacement=False,
        random_state=RND_SEED,
    )
    X_resampled, y_resampled, weight_resampled = spus.fit_resample(
        X,
        y,
        alpha=0,
        y_pred_proba=pred_proba_normal,
        sample_weight=sample_weight,
        classes_=classes_,
        encode_map=encode_map,
    )
    assert X_resampled.shape == (6, 2)
    assert y_resampled.shape == (6,)
    assert weight_resampled.shape == (6,)
    assert dict(Counter(y_resampled)) == sampling_strategy_default


@pytest.mark.parametrize(
    "invalid_target", [True, False], ids=["invalid_target", "valid_target"]
)
@pytest.mark.parametrize(
    "replacement", [True, False], ids=["replacement", "non_replacement"]
)
def test_spus_hard_resample(invalid_target, replacement):
    if invalid_target:
        sampling_strategy_ = sampling_strategy_normal
    else:
        sampling_strategy_ = "auto"
    spus = SelfPacedUnderSampler(
        sampling_strategy=sampling_strategy_,
        k_bins=5,
        soft_resample_flag=False,
        replacement=replacement,
        random_state=RND_SEED,
    )
    if invalid_target and not replacement:
        with pytest.raises(
            RuntimeError, match="with insufficient number of data samples"
        ):
            X_resampled, y_resampled = spus.fit_resample(
                X,
                y,
                alpha=0,
                y_pred_proba=pred_proba_normal,
                classes_=classes_,
                encode_map=encode_map,
            )
    else:
        X_resampled, y_resampled = spus.fit_resample(
            X,
            y,
            alpha=0,
            y_pred_proba=pred_proba_normal,
            classes_=classes_,
            encode_map=encode_map,
        )
        n_res = 12 if invalid_target else 6
        assert X_resampled.shape == (n_res, 2)
        assert y_resampled.shape == (n_res,)
        if invalid_target:
            assert dict(Counter(y_resampled)) == sampling_strategy_normal
        else:
            assert dict(Counter(y_resampled)) == sampling_strategy_default
