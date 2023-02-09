# """Test the module self-paced under sampler."""

# # Authors: Zhining Liu <zhining.liu@outlook.com>
# # License: MIT

# import numpy as np
# from collections import Counter
# import pytest

# from sklearn.utils._testing import assert_array_equal
# from sklearn.ensemble import BaggingClassifier

# from imbalanced_ensemble.sampler.under_sampling import SelfPacedUnderSampler

# RND_SEED = 0
# X = np.array([
#     [2.45166, 1.86760],
#     [1.34450, -1.30331],
#     [1.02989, 2.89408],
#     [-1.94577, -1.75057],
#     [1.21726, 1.90146],
#     [2.00194, 1.25316],
#     [2.31968, 2.33574],
#     [1.14769, 1.41303],
#     [1.32018, 2.17595],
#     [-1.74686, -1.66665],
#     [-2.17373, -1.91466],
#     [2.41436, 1.83542],
#     [1.97295, 2.55534],
#     [-2.12126, -2.43786],
#     [1.20494, 3.20696],
#     [-2.30158, -2.39903],
#     [1.76006, 1.94323],
#     [2.35825, 1.77962],
#     [-2.06578, -2.07671],
#     [0.00245, -0.99528],
# ])
# y = np.array([2, 0, 2, 1, 2, 2, 2, 2, 2, 1, 1, 2, 2, 1, 2, 1, 2, 2, 1, 0])


# @pytest.mark.parametrize("as_frame", [True, False], ids=["dataframe", "array"])
# def test_rus_fit_resample(as_frame):
#     if as_frame:
#         pd = pytest.importorskip("pandas")
#         X_ = pd.DataFrame(X)
#     else:
#         X_ = X

#     clf = BaggingClassifier(
#         n_estimators=5,
#         random_state=RND_SEED,
#     ).fit(X_, y)
#     y_pred_proba = clf.predict_proba(X)

#     spus = SelfPacedUnderSampler(random_state=RND_SEED, replacement=True)

#     X_resampled, y_resampled = spus.fit_resample(X_, y)

#     X_gt = np.array([
#         [0.09125, -0.85410],
#         [0.12373, 0.65362],
#         [0.20793, 1.49408],
#     ])
#     y_gt = np.array([1, 1, 1])

#     if as_frame:
#         assert hasattr(X_resampled, "loc")
#         X_resampled = X_resampled.to_numpy()

#     assert_array_equal(X_resampled, X_gt)
#     assert_array_equal(y_resampled, y_gt) 