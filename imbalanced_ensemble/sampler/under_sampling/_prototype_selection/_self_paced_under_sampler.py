"""Class to perform self-paced under-sampling."""

# Authors: Zhining Liu <zhining.liu@outlook.com>
# License: MIT

# %%


import numbers
import numpy as np

from sklearn.utils import check_random_state
from sklearn.utils import _safe_indexing


from ..base import BaseUnderSampler
from ....utils._docstring import Substitution
from ....utils._docstring import _random_state_docstring
from ....utils._validation_param import check_pred_proba, check_type
from ....utils._validation import _deprecate_positional_args, check_target_type

# # For local test
# import sys
# sys.path.append("../../..")
# from sampler.under_sampling.base import BaseUnderSampler
# from utils._docstring import Substitution
# from utils._docstring import _random_state_docstring
# from utils._validation_param import check_pred_proba, check_type
# from utils._validation import _deprecate_positional_args, check_target_type



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
        self, *, 
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
        self.k_bins_ = check_type(k_bins, 'k_bins', numbers.Integral)
        self.replacement_ = check_type(replacement, 'replacement', bool)
        self.soft_resample_flag_ = check_type(soft_resample_flag, 
            'soft_resample_flag', bool)


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


    def fit_resample(self, X, y, *, sample_weight, **kwargs):
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
    def _fit_resample(self, X, y, *, 
                      y_pred_proba, alpha:float, 
                      classes_, sample_weight=None):
        
        n_samples, n_classes = X.shape[0], classes_.shape[0]

        # Check random_state and predict probabilities
        random_state = check_random_state(self.random_state)
        y_pred_proba = check_pred_proba(y_pred_proba, n_samples, n_classes, dtype=np.float64)

        # Check the self-paced factor alpha
        alpha = check_type(alpha, 'alpha', numbers.Number)
        if alpha < 0:
            raise ValueError("'alpha' must not be negative.")

        indexes = np.arange(n_samples)
        index_list = []

        # For each class C
        for target_class in classes_:
            if target_class in self.sampling_strategy_.keys():

                # Get the desired & actual number of samples of class C
                # and the index mask of class C
                n_target_samples_c = self.sampling_strategy_[target_class]
                class_index_mask = y == target_class
                n_samples_c = np.count_nonzero(class_index_mask)

                # Compute the hardness array
                hardness_c=np.abs(
                    np.ones(n_samples_c) - \
                    y_pred_proba[class_index_mask, target_class])

                # index_c: absolute indexes of class C samples
                index_c = indexes[class_index_mask]

                if n_target_samples_c <= n_samples_c:
                    
                    # Get the absolute indexes of resampled class C samples
                    index_c_result = self._undersample_single_class(
                        hardness_c=hardness_c,
                        n_target_samples_c=n_target_samples_c,
                        index_c=index_c,
                        alpha=alpha,
                        random_state=random_state)

                # If no sufficient samples in class C, raise an RuntimeError
                else: raise RuntimeError(
                    f"Got n_target_samples_c ({n_target_samples_c})"
                    f" > n_samples_c ({n_samples_c} for class {target_class}.)"
                )
                index_list.append(index_c_result)
        
        # Concatenate the result
        index_spu = np.hstack(index_list)

        # Store the final undersample indexes
        self.sample_indices_ = index_spu

        # Return the resampled X, y
        # also return resampled sample_weight (if sample_weight is not None)
        if sample_weight is not None:
            # sample_weight is already validated in super().fit_resample()
            weights_under = _safe_indexing(sample_weight, index_spu)
            return _safe_indexing(X, index_spu), _safe_indexing(y, index_spu), weights_under
        else: return _safe_indexing(X, index_spu), _safe_indexing(y, index_spu)


    def _undersample_single_class(self, hardness_c, n_target_samples_c, 
                                  index_c, alpha, random_state):
        """Perform self-paced under-sampling in a single class"""
        k_bins = self.k_bins_
        soft_resample_flag = self.soft_resample_flag_
        replacement = self.replacement_
        n_samples_c = hardness_c.shape[0]

        # if hardness is not distinguishable or no sample will be dropped
        if hardness_c.max() == hardness_c.min() or n_target_samples_c == n_samples_c:
            # perform random under-sampling
            return random_state.choice(
                index_c,
                size=n_target_samples_c,
                replace=replacement)

        with np.errstate(divide='ignore', invalid='ignore'):
            # compute population & hardness contribution of each bin
            populations, edges = np.histogram(hardness_c, bins=k_bins)
            contributions = np.zeros(k_bins)
            index_bins = []
            for i_bin in range(k_bins):
                index_bin = ((hardness_c >= edges[i_bin]) & (hardness_c < edges[i_bin+1]))
                if i_bin == (k_bins-1):
                    index_bin = index_bin | (hardness_c==edges[i_bin+1])
                index_bins.append(index_bin)
                if populations[i_bin] > 0:
                    contributions[i_bin] = hardness_c[index_bin].mean()

            # compute the expected number of samples to be sampled from each bin
            bin_weights = 1. / (contributions + alpha)
            bin_weights[np.isnan(bin_weights)] = 0
            n_target_samples_bins = n_target_samples_c * bin_weights / bin_weights.sum()
            n_target_samples_bins = n_target_samples_bins.astype(int)+1

        if soft_resample_flag:
            with np.errstate(divide='ignore', invalid='ignore'):
                # perform soft (weighted) self-paced under-sampling
                soft_spu_bin_weights = n_target_samples_bins / populations
                soft_spu_bin_weights[~np.isfinite(soft_spu_bin_weights)] = 0
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
                p=soft_spu_sample_proba,)
        else:
            # perform hard self-paced under-sampling
            index_c_results = []
            for i_bin in range(k_bins):
                # if no sufficient data in bin for resampling, raise an RuntimeError
                if populations[i_bin] < n_target_samples_bins[i_bin] and not replacement:
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
                            replace=replacement,)
                    )
            # concatenate and return the result
            index_c_result = np.hstack(index_c_results)
            return index_c_result
        

    def _more_tags(self):
        return {
            "X_types": ["2darray", "string", "sparse", "dataframe"],
            "sample_indices": True,
            "allow_nan": True,
        }

# %%

if __name__ == "__main__":

    import sys
    sys.path.append("../../..")
    import numpy as np

    from sklearn.utils import check_random_state
    from sklearn.utils import _safe_indexing
    from sklearn.utils.validation import _check_sample_weight

    from sampler.under_sampling.base import BaseUnderSampler
    from utils._docstring import _random_state_docstring, Substitution
    from utils._validation import _deprecate_positional_args, check_target_type
    from utils._validation_param import check_pred_proba, check_type

    from collections import Counter
    from sklearn.datasets import make_classification
    from sklearn.ensemble import BaggingClassifier

    X, y = make_classification(n_classes=3, class_sep=2,
        weights=[0.1, 0.3, 0.6], n_informative=3, n_redundant=1, flip_y=0,
        n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
    print('Original dataset shape %s' % Counter(y))

    sampling_strategy_ = {2: 200, 1: 100, 0: 90}
    print('Target dataset shape %s' % sampling_strategy_)

    sample_weight = np.full_like(y, fill_value=1/y.shape[0], dtype=float)
    clf = BaggingClassifier(
        n_estimators=50,
    ).fit(X, y)
    
    y_pred_proba = clf.predict_proba(X)

    alpha = 0

    spu = SelfPacedUnderSampler(
        sampling_strategy=sampling_strategy_,
        k_bins=5,
        soft_resample_flag=True,
        replacement=False, 
        random_state=0,
    )
    X_res, y_res, weights_res = spu.fit_resample(X, y, y_pred_proba=y_pred_proba, alpha=0, sample_weight=sample_weight)
    print('Resampled dataset shape %s' % Counter(y_res))
# %%
