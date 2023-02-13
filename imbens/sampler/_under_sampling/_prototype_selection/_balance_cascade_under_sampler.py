"""Class to perform random under-sampling for BalanceCascade."""

# Authors: Zhining Liu <zhining.liu@outlook.com>
# License: MIT

# %%
LOCAL_DEBUG = False

if not LOCAL_DEBUG:
    from ....utils._docstring import Substitution, _random_state_docstring
    from ....utils._validation import _deprecate_positional_args, check_target_type
    from ....utils._validation_param import check_pred_proba, check_type
    from ..base import BaseUnderSampler
else:  # pragma: no cover
    import sys  # For local test

    sys.path.append("../../..")
    from sampler._under_sampling.base import BaseUnderSampler
    from utils._docstring import _random_state_docstring, Substitution
    from utils._validation_param import check_pred_proba, check_type
    from utils._validation import _deprecate_positional_args, check_target_type

import numpy as np
from sklearn.utils import _safe_indexing, check_random_state


@Substitution(
    sampling_strategy=BaseUnderSampler._sampling_strategy_docstring,
    random_state=_random_state_docstring,
)
class BalanceCascadeUnderSampler(BaseUnderSampler):
    """Class to perform under-sampling for BalanceCascade in [1]_.

    Parameters
    ----------
    {sampling_strategy}

    replacement : bool, default=False
        Whether samples are drawn with replacement. If ``False``
        and ``soft_resample_flag = False``, may raise an error when
        a bin has insufficient number of data samples for resampling.

    {random_state}

    Attributes
    ----------
    sample_indices_ : ndarray of shape (n_new_samples,)
        Indices of the samples selected.

    See Also
    --------
    SelfPacedUnderSampler : Dynamic under-sampling for SelfPacedEnsemble.

    Notes
    -----
    Supports multi-class resampling by sampling each class independently.
    Supports heterogeneous data as object array containing string and numeric
    data.

    References
    ----------
    .. [1] Liu, X. Y., Wu, J., & Zhou, Z. H. "Exploratory undersampling for
       class-imbalance learning." IEEE Transactions on Systems, Man, and
       Cybernetics, Part B (Cybernetics) 39.2 (2008): 539-550.
    """

    @_deprecate_positional_args
    def __init__(
        self,
        *,
        sampling_strategy="auto",
        replacement=False,
        random_state=None,
    ):
        super().__init__(sampling_strategy=sampling_strategy)

        self.replacement = replacement
        self.random_state = random_state

        # Check parameters
        self.replacement = check_type(replacement, 'replacement', bool)

    def _check_X_y(self, X, y):
        y, binarize_y = check_target_type(y, indicate_one_vs_all=True)
        X, y = self._validate_data(
            X,
            y,
            reset=True,
            accept_sparse=["csr", "csc"],
            dtype=None,
            force_all_finite=False,
        )
        return X, y, binarize_y

    def _get_new_dropped_index_single_class(
        self, error_kept_c, n_samples, n_drop_samples_c, absolute_index_kept_c
    ):
        """Return the new dropped index of a single class"""

        sorted_index = np.argsort(error_kept_c)
        sorted_absolute_index_kept_c = absolute_index_kept_c[sorted_index]
        dropped_absolute_index_c = sorted_absolute_index_kept_c[:n_drop_samples_c]
        new_dropped_index_c = np.full(n_samples, fill_value=False, dtype=bool)
        new_dropped_index_c[dropped_absolute_index_c] = True

        return new_dropped_index_c

    def _undersample_single_class(
        self, n_target_samples_c, absolute_index_kept_c, random_state
    ):
        """Return the absolute index of samples after balance-cascade
        under-sampling of a single class"""

        replacement = self.replacement

        return random_state.choice(
            absolute_index_kept_c, size=n_target_samples_c, replace=replacement
        )

    def fit_resample(self, X, y, *, sample_weight=None, **kwargs):
        """Resample the dataset.

        Parameters
        ----------
        X : {array-like, dataframe, sparse matrix} of shape \
                (n_samples, n_features)
            Matrix containing the data which have to be sampled.

        y : array-like of shape (n_samples,)
            Corresponding label for each sample in X.

        y_pred_proba : array-like of shape (n_samples, n_classes)
            The predicted class probabilities of the input samples
            by the current SPE ensemble classifier. The order of the
            classes corresponds to that in the parameter `classes_`.
        
        dropped_index : array-like of shape (n_samples,)
            The mask corresponding to X, `True` indicates that the
            corresponding sample is already discarded.
        
        keep_populations : dict
            Specify how many samples will be kept for each class 
            after this resampling. The keys (``int``) correspond 
            to the targeted classes. The values (``int``) correspond 
            to the number of samples to keep.
        
        classes_ : ndarray of shape (n_classes,)
            The classes labels.

        sample_weight : array-like of shape (n_samples,), default=None
            Corresponding weight for each sample in X.
        
        Returns
        -------
        X_resampled : {array-like, dataframe, sparse matrix} of shape \
                (n_samples_new, n_features)
            The array containing the resampled data.

        y_resampled : array-like of shape (n_samples_new,)
            The corresponding label of `X_resampled`.
        
        sample_weight : array-like of shape (n_samples_new,), default=None
            The corresponding weight of `X_resampled`.
            Only will be returned if input sample_weight is not None.
        """
        return super().fit_resample(X, y, sample_weight=sample_weight, **kwargs)

    @_deprecate_positional_args
    def _fit_resample(
        self,
        X,
        y,
        *,
        y_pred_proba,
        dropped_index,
        keep_populations: dict,
        classes_,
        encode_map,
        sample_weight=None,
    ):

        n_samples, n_classes = X.shape[0], classes_.shape[0]

        # Check random_state and predict probabilities
        replacement = self.replacement
        random_state = check_random_state(self.random_state)
        y_pred_proba = check_pred_proba(
            y_pred_proba, n_samples, n_classes, dtype=np.float64
        )

        # Only consider samples that have not been discarded
        y_pred_proba_kept = y_pred_proba[~dropped_index].copy()
        y_kept = y[~dropped_index].copy()

        # Only consider samples that have not been discarded
        absolute_index = np.arange(n_samples)
        absolute_index_kept = absolute_index[~dropped_index]
        absolute_index_resample_list = []
        new_dropped_index = dropped_index.copy()

        # For each class C
        for target_class in classes_:
            if target_class in self.sampling_strategy_.keys():

                # Get the desired & actual number of samples of class C
                # and the index mask of class C (all on kept samples)
                n_target_samples_c = self.sampling_strategy_[target_class]
                class_index_mask_kept = y_kept == target_class
                n_samples_c = np.count_nonzero(class_index_mask_kept)

                # absolute_index_kept_c: absolute indexes of class C kept samples
                absolute_index_kept_c = absolute_index_kept[class_index_mask_kept]

                if n_target_samples_c <= n_samples_c or replacement == True:

                    # Get the absolute indexes of resampled class C samples
                    absolute_index_resample_c = self._undersample_single_class(
                        n_target_samples_c=n_target_samples_c,
                        absolute_index_kept_c=absolute_index_kept_c,
                        random_state=random_state,
                    )

                # If no sufficient samples in class C and `replacement` is False
                # raise an RuntimeError
                else:
                    raise ValueError(
                        f"Got n_target_samples_c ({n_target_samples_c})"
                        f" > n_samples_c ({n_samples_c} for class {target_class}.)"
                        f" during BalanceCascade Under-sampling."
                        f" Set 'balancing_schedule' to 'uniform' when calling `fit`"
                        f" to avoid this issue."
                    )
                absolute_index_resample_list.append(absolute_index_resample_c)

                # Compute how many samples should be discarded in class C
                n_drop_samples_c = n_samples_c - keep_populations[target_class]

                # Compute prediction error and
                # collect new dropped index of the class C
                error_kept_c = np.abs(
                    np.ones(n_samples_c)
                    - y_pred_proba_kept[class_index_mask_kept, encode_map[target_class]]
                )
                new_dropped_index_c = self._get_new_dropped_index_single_class(
                    error_kept_c=error_kept_c,
                    n_samples=n_samples,
                    n_drop_samples_c=n_drop_samples_c,
                    absolute_index_kept_c=absolute_index_kept_c,
                )
                new_dropped_index = new_dropped_index | new_dropped_index_c
            else:
                class_index_mask_kept = y_kept == target_class
                absolute_index_kept_c = absolute_index_kept[class_index_mask_kept]
                absolute_index_resample_list.append(absolute_index_kept_c)

        # Concatenate the result
        index_bcu = np.hstack(absolute_index_resample_list)

        # Store the final undersample indexes
        self.sample_indices_ = index_bcu

        # Return the resampled X, y
        # also return resampled sample_weight (if sample_weight is not None)
        if sample_weight is not None:
            # sample_weight is already validated in super().fit_resample()
            sample_weight_under = _safe_indexing(sample_weight, index_bcu)
            return (
                _safe_indexing(X, index_bcu),
                _safe_indexing(y, index_bcu),
                sample_weight_under,
                new_dropped_index,
            )
        else:
            return (
                _safe_indexing(X, index_bcu),
                _safe_indexing(y, index_bcu),
                new_dropped_index,
            )

    def _more_tags(self):  # pragma: no cover
        return {
            "X_types": ["2darray", "string", "sparse", "dataframe"],
            "sample_indices": True,
            "allow_nan": True,
        }


# %%

if __name__ == "__main__":  # pragma: no cover

    from collections import Counter

    import pandas as pd
    from sklearn.datasets import make_classification

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

    X = pd.DataFrame(X)

    # X, y = make_classification(n_classes=3, class_sep=2,
    #     weights=[0.1, 0.3, 0.6], n_informative=3, n_redundant=1, flip_y=0,
    #     n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
    print('Original dataset shape %s' % Counter(y))

    bcus = BalanceCascadeUnderSampler(
        # sampling_strategy=sampling_strategy_org,
        replacement=False,
        random_state=0,
    )
    X_res, y_res, new_dropped_index, weights_res = bcus.fit_resample(
        # X_res, y_res, new_dropped_index = bcus.fit_resample(
        X,
        y,
        # y_pred_proba=pred_proba_normal,
        y_pred_proba=pred_proba_det,
        sample_weight=sample_weight,
        classes_=classes_,
        encode_map=encode_map,
        dropped_index=dropped_index,
        keep_populations=sampling_strategy_org,
    )
    print('Resampled dataset shape %s' % Counter(y_res))
    print('Kept dataset shape %s' % Counter(y[~new_dropped_index]))

# %%
