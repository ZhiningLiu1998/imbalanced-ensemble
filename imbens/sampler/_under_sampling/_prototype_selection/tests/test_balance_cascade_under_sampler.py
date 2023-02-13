"""Test the module balance cascade under sampler."""

# Authors: Zhining Liu <zhining.liu@outlook.com>
# License: MIT

from collections import Counter

import numpy as np
import pytest

from imbens.sampler._under_sampling import BalanceCascadeUnderSampler

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
dropped_index = np.zeros_like(y).astype(bool)
dropped_index_empty = np.ones_like(y).astype(bool)
sample_weight = np.full_like(y, fill_value=1 / y.shape[0], dtype=float)
sampling_strategy_org = {2: 12, 1: 6, 0: 2}
sampling_strategy_default = {2: 2, 1: 2, 0: 2}
sampling_strategy_normal = {2: 6, 1: 4, 0: 2}
sampling_strategy_error = {2: 200, 1: 100, 0: 90}


@pytest.mark.parametrize("as_frame", [True, False], ids=["dataframe", "array"])
@pytest.mark.parametrize("proba_det", [True, False], ids=["det", "normal"])
@pytest.mark.parametrize(
    "sampling",
    ['org', 'auto', 'default', 'normal', 'error'],
    ids=['org', 'auto', 'default', 'normal', 'error'],
)
@pytest.mark.parametrize(
    "keep_pop", ['org', 'default', 'normal'], ids=['org', 'default', 'normal']
)
def test_spus_fit_resample(as_frame, proba_det, sampling, keep_pop):
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

    if sampling == 'org':
        sampling_strategy = sampling_strategy_org
    elif sampling == 'auto':
        sampling_strategy = "auto"
    elif sampling == 'default':
        sampling_strategy = sampling_strategy_default
    elif sampling == 'normal':
        sampling_strategy = sampling_strategy_normal
    elif sampling == 'error':
        sampling_strategy = sampling_strategy_error

    if keep_pop == 'org':
        keep_populations = sampling_strategy_org
    elif keep_pop == 'default':
        keep_populations = sampling_strategy_default
    elif keep_pop == 'normal':
        keep_populations = sampling_strategy_normal

    # init resampler
    bcus = BalanceCascadeUnderSampler(
        sampling_strategy=sampling_strategy,
        replacement=False,
        random_state=RND_SEED,
    )

    # resampling
    if sampling == 'error':
        with pytest.raises(ValueError, match="With under-sampling methods"):
            X_resampled, y_resampled, new_dropped_index = bcus.fit_resample(
                X_,
                y,
                y_pred_proba=y_pred_proba,
                classes_=classes_,
                encode_map=encode_map,
                dropped_index=dropped_index,
                keep_populations=keep_populations,
            )
    else:
        X_resampled, y_resampled, new_dropped_index = bcus.fit_resample(
            X_,
            y,
            y_pred_proba=y_pred_proba,
            classes_=classes_,
            encode_map=encode_map,
            dropped_index=dropped_index,
            keep_populations=keep_populations,
        )

        if as_frame:
            if hasattr(X_resampled, "loc"):
                X_resampled = X_resampled.to_numpy()

        if sampling == 'auto':
            sampling_strategy = sampling_strategy_default
        assert X_resampled.shape == (sum(sampling_strategy.values()), 2)
        assert y_resampled.shape == (sum(sampling_strategy.values()),)
        assert dict(Counter(y_resampled)) == sampling_strategy
        assert dict(Counter(y[~new_dropped_index])) == keep_populations


def test_bcus_no_sufficient_data():
    # init resampler
    bcus = BalanceCascadeUnderSampler(
        sampling_strategy="auto",
        replacement=False,
        random_state=RND_SEED,
    )
    with pytest.raises(ValueError, match="Got n_target_samples_c"):
        X_resampled, y_resampled, new_dropped_index = bcus.fit_resample(
            X,
            y,
            y_pred_proba=pred_proba_normal,
            classes_=classes_,
            encode_map=encode_map,
            dropped_index=dropped_index_empty,
            keep_populations=sampling_strategy_normal,
        )


@pytest.mark.parametrize(
    "sampling",
    ['org', 'auto', 'default', 'normal'],
    ids=['org', 'auto', 'default', 'normal'],
)
def test_bcus_with_sample_weight(sampling):

    if sampling == 'org':
        sampling_strategy = sampling_strategy_org
    elif sampling == 'auto':
        sampling_strategy = "auto"
    elif sampling == 'default':
        sampling_strategy = sampling_strategy_default
    elif sampling == 'normal':
        sampling_strategy = sampling_strategy_normal

    # init resampler
    bcus = BalanceCascadeUnderSampler(
        sampling_strategy=sampling_strategy,
        replacement=False,
        random_state=RND_SEED,
    )
    (
        X_resampled,
        y_resampled,
        sample_weight_under,
        new_dropped_index,
    ) = bcus.fit_resample(
        X,
        y,
        y_pred_proba=pred_proba_normal,
        sample_weight=sample_weight,
        classes_=classes_,
        encode_map=encode_map,
        dropped_index=dropped_index,
        keep_populations=sampling_strategy_normal,
    )
    if sampling == 'auto':
        sampling_strategy = sampling_strategy_default
    assert X_resampled.shape == (sum(sampling_strategy.values()), 2)
    assert y_resampled.shape == (sum(sampling_strategy.values()),)
    assert sample_weight_under.shape == (sum(sampling_strategy.values()),)
    assert dict(Counter(y_resampled)) == sampling_strategy
    assert dict(Counter(y[~new_dropped_index])) == sampling_strategy_normal
