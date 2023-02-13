"""Class to perform self-paced under-sampling."""

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
    from utils._docstring import Substitution
    from utils._docstring import _random_state_docstring
    from utils._validation_param import check_pred_proba, check_type
    from utils._validation import _deprecate_positional_args, check_target_type

import numbers

import numpy as np
from sklearn.utils import _safe_indexing, check_random_state


@Substitution(
    sampling_strategy=BaseUnderSampler._sampling_strategy_docstring,
    random_state=_random_state_docstring,
)
class SelfPacedUnderSampler(BaseUnderSampler):
    """Class to perform self-paced under-sampling in [1]_.

    Parameters
    ----------
    {sampling_strategy}

    k_bins : int, default=5
        The number of hardness bins that were used to approximate
        hardness distribution. It is recommended to set it to 5.
        One can try a larger value when the smallest class in the
        data set has a sufficient number (say, > 1000) of samples.

    soft_resample_flag : bool, default=False
        Whether to use weighted sampling to perform soft self-paced
        under-sampling, rather than explicitly cut samples into
        ``k``-bins and perform hard sampling.

    replacement : bool, default=True
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
    BalanceCascadeUnderSampler :  Dynamic under-sampling for BalanceCascade.

    Notes
    -----
    Supports multi-class resampling by sampling each class independently.
    Supports heterogeneous data as object array containing string and numeric
    data.

    References
    ----------
    .. [1] Liu, Z., Cao, W., Gao, Z., Bian, J., Chen, H., Chang, Y., & Liu, T. Y.
       "Self-paced ensemble for highly imbalanced massive data classification."
       2020 IEEE 36th International Conference on Data Engineering (ICDE).
       IEEE, 2010: 841-852.
    """

    @_deprecate_positional_args
    def __init__(
        self,
        *,
        sampling_strategy="auto",
        k_bins=5,
        soft_resample_flag=True,
        replacement=False,
        random_state=None,
    ):
        super().__init__(sampling_strategy=sampling_strategy)

        self.k_bins = k_bins
        self.soft_resample_flag = soft_resample_flag
        self.replacement = replacement
        self.random_state = random_state

        # Check parameters
        self.k_bins = check_type(k_bins, 'k_bins', numbers.Integral)
        self.replacement = check_type(replacement, 'replacement', bool)
        self.soft_resample_flag = check_type(
            soft_resample_flag, 'soft_resample_flag', bool
        )
        if self.k_bins <= 0:
            raise ValueError(f"'k_bins' should be > 0, got k_bins={self.k_bins}.")

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
        
        alpha : float
            The self-paced factor that controls SPE under-sampling.
        
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
        alpha: float,
        classes_,
        encode_map,
        sample_weight=None,
    ):

        n_samples, n_classes = X.shape[0], classes_.shape[0]

        # Check random_state and predict probabilities
        random_state = check_random_state(self.random_state)
        y_pred_proba = check_pred_proba(
            y_pred_proba, n_samples, n_classes, dtype=np.float64
        )
        if self.k_bins <= 0:
            raise ValueError(f"'k_bins' should be > 0, got k_bins={self.k_bins}.")

        # Check the self-paced factor alpha
        alpha = check_type(alpha, 'alpha', numbers.Number)
        if alpha < 0:
            raise ValueError("'alpha' must not be negative.")

        indexes = np.arange(n_samples)
        index_list = []

        # For each class C
        for target_class in np.unique(y):
            if target_class in self.sampling_strategy_.keys():

                # Get the desired & actual number of samples of class C
                # and the index mask of class C
                n_target_samples_c = self.sampling_strategy_[target_class]
                class_index_mask = y == target_class
                n_samples_c = np.count_nonzero(class_index_mask)

                # Compute the hardness array
                hardness_c = np.abs(
                    np.ones(n_samples_c)
                    - y_pred_proba[class_index_mask, encode_map[target_class]]
                )

                # index_c: absolute indexes of class C samples
                index_c = indexes[class_index_mask]

                if n_target_samples_c <= n_samples_c:

                    # Get the absolute indexes of resampled class C samples
                    index_c_result = self._undersample_single_class(
                        hardness_c=hardness_c,
                        n_target_samples_c=n_target_samples_c,
                        index_c=index_c,
                        alpha=alpha,
                        random_state=random_state,
                    )

                # If no sufficient samples in class C, raise an RuntimeError
                else:
                    raise RuntimeError(
                        f"Got n_target_samples_c ({n_target_samples_c})"
                        f" > n_samples_c ({n_samples_c} for class {target_class}.)"
                    )
                index_list.append(index_c_result)
            else:
                class_index_mask = y == target_class
                index_c = indexes[class_index_mask]
                index_list.append(index_c)

        # Concatenate the result
        index_spu = np.hstack(index_list)

        # Store the final undersample indexes
        self.sample_indices_ = index_spu

        # Return the resampled X, y
        # also return resampled sample_weight (if sample_weight is not None)
        if sample_weight is not None:
            # sample_weight is already validated in super().fit_resample()
            weights_under = _safe_indexing(sample_weight, index_spu)
            return (
                _safe_indexing(X, index_spu),
                _safe_indexing(y, index_spu),
                weights_under,
            )
        else:
            return _safe_indexing(X, index_spu), _safe_indexing(y, index_spu)

    def _undersample_single_class(
        self, hardness_c, n_target_samples_c, index_c, alpha, random_state
    ):
        """Perform self-paced under-sampling in a single class"""
        k_bins = self.k_bins
        soft_resample_flag = self.soft_resample_flag
        replacement = self.replacement
        n_samples_c = hardness_c.shape[0]

        # if hardness is not distinguishable or no sample will be dropped
        if hardness_c.max() == hardness_c.min() or n_target_samples_c == n_samples_c:
            # perform random under-sampling
            return random_state.choice(
                index_c, size=n_target_samples_c, replace=replacement
            )

        with np.errstate(divide='ignore', invalid='ignore'):
            # compute population & hardness contribution of each bin
            populations, edges = np.histogram(hardness_c, bins=k_bins)
            contributions = np.zeros(k_bins)
            index_bins = []
            for i_bin in range(k_bins):
                index_bin = (hardness_c >= edges[i_bin]) & (
                    hardness_c < edges[i_bin + 1]
                )
                if i_bin == (k_bins - 1):
                    index_bin = index_bin | (hardness_c == edges[i_bin + 1])
                index_bins.append(index_bin)
                if populations[i_bin] > 0:
                    contributions[i_bin] = hardness_c[index_bin].mean()

            # compute the expected number of samples to be sampled from each bin
            bin_weights = 1.0 / (contributions + alpha)
            bin_weights[np.isnan(bin_weights) | np.isinf(bin_weights)] = 0
            n_target_samples_bins = n_target_samples_c * bin_weights / bin_weights.sum()
            # check whether exists empty bins
            n_invalid_samples = sum(n_target_samples_bins[populations == 0])
            if n_invalid_samples > 0:
                n_valid_samples = n_target_samples_c - n_invalid_samples
                n_target_samples_bins *= n_target_samples_c / n_valid_samples
                n_target_samples_bins[populations == 0] = 0
            n_target_samples_bins = n_target_samples_bins.astype(int)
            while True:
                for i in np.flip(np.argsort(populations)):
                    n_target_diff = n_target_samples_c - n_target_samples_bins.sum()
                    if n_target_diff == 0:
                        break
                    elif populations[i] > 0:
                        n_target_samples_bins[i] += 1
                if n_target_diff == 0:
                    break
            assert n_target_samples_c == n_target_samples_bins.sum()

        if soft_resample_flag:
            with np.errstate(divide='ignore', invalid='ignore'):
                # perform soft (weighted) self-paced under-sampling
                soft_spu_bin_weights = n_target_samples_bins / populations
                soft_spu_bin_weights[~np.isfinite(soft_spu_bin_weights)] = 0
                # print ('soft_spu_bin_weights: ', soft_spu_bin_weights)
            # compute sampling probabilities
            soft_spu_sample_proba = np.zeros_like(hardness_c)
            for i_bin in range(k_bins):
                soft_spu_sample_proba[index_bins[i_bin]] = soft_spu_bin_weights[i_bin]
            soft_spu_sample_proba /= soft_spu_sample_proba.sum()
            # sample with respect to the sampling probabilities
            return random_state.choice(
                index_c,
                size=n_target_samples_c,
                replace=replacement,
                p=soft_spu_sample_proba,
            )
        else:
            # perform hard self-paced under-sampling
            index_c_results = []
            for i_bin in range(k_bins):
                # if no sufficient data in bin for resampling, raise an RuntimeError
                if (
                    populations[i_bin] < n_target_samples_bins[i_bin]
                    and not replacement
                ):
                    raise RuntimeError(
                        f"Met {i_bin}-th bin with insufficient number of data samples"
                        f" ({populations[i_bin]}, expected"
                        f" >= {n_target_samples_bins[i_bin]})."
                        f" Set 'soft_resample_flag' or 'replacement' to `True` to."
                        f" avoid this issue."
                    )
                index_c_bin = index_c[index_bins[i_bin]]
                # random sample from each bin
                if len(index_c_bin) > 0:
                    index_c_results.append(
                        random_state.choice(
                            index_c_bin,
                            size=n_target_samples_bins[i_bin],
                            replace=replacement,
                        )
                    )
            # concatenate and return the result
            index_c_result = np.hstack(index_c_results)
            return index_c_result

    def _more_tags(self):  # pragma: no cover
        return {
            "X_types": ["2darray", "string", "sparse", "dataframe"],
            "sample_indices": True,
            "allow_nan": True,
        }


# %%

if __name__ == "__main__":  # pragma: no cover

    from collections import Counter

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
    sample_weight = np.full_like(y, fill_value=1 / y.shape[0], dtype=float)
    sampling_strategy_normal = {2: 6, 1: 4, 0: 2}
    sampling_strategy_error = {2: 200, 1: 100, 0: 90}

    # X, y = make_classification(n_classes=3, class_sep=2,
    #     weights=[0.1, 0.3, 0.6], n_informative=3, n_redundant=1, flip_y=0,
    #     n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
    print('Original dataset shape %s' % Counter(y))

    spu = SelfPacedUnderSampler(
        sampling_strategy=sampling_strategy_normal,
        k_bins=5,
        soft_resample_flag=False,
        replacement=False,
        random_state=0,
    )
    # X_res, y_res, weights_res = spu.fit_resample(
    X_res, y_res = spu.fit_resample(
        X,
        y,
        y_pred_proba=pred_proba_normal,
        alpha=0,
        # sample_weight=sample_weight,
        classes_=classes_,
        encode_map=encode_map,
    )
    print('Resampled dataset shape %s' % Counter(y_res))

# %%
