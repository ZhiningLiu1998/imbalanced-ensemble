"""Class to perform random under-sampling for BalanceCascade."""

# Authors: Zhining Liu <zhining.liu@outlook.com>
# License: MIT


# %%
LOCAL_DEBUG = False

if not LOCAL_DEBUG:
    from ..base import BaseUnderSampler
    from ....utils._validation_param import check_pred_proba, check_type
    from ....utils._validation import _deprecate_positional_args, check_target_type
    from ....utils._docstring import _random_state_docstring, Substitution
else:
    # For local test
    import sys
    sys.path.append("../../..")
    from sampler.under_sampling.base import BaseUnderSampler
    from utils._docstring import _random_state_docstring, Substitution
    from utils._validation_param import check_pred_proba, check_type
    from utils._validation import _deprecate_positional_args, check_target_type

import numpy as np

from sklearn.utils import check_random_state
from sklearn.utils import _safe_indexing


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
        self, *, 
        sampling_strategy="auto", 
        replacement=False, 
        random_state=None, 
    ):
        super().__init__(sampling_strategy=sampling_strategy)

        self.replacement = replacement
        self.random_state = random_state

        # Check parameters
        self.replacement_ = check_type(replacement, 'replacement', bool)


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


    def _get_new_dropped_index_single_class(self, error_kept_c, n_samples, 
                                            n_drop_samples_c, absolute_index_kept_c):
        """Return the new dropped index of a single class"""

        sorted_index = np.argsort(error_kept_c)
        sorted_absolute_index_kept_c = absolute_index_kept_c[sorted_index]
        dropped_absolute_index_c = sorted_absolute_index_kept_c[: n_drop_samples_c]
        new_dropped_index_c = np.full(n_samples, fill_value=False, dtype=bool)
        new_dropped_index_c[dropped_absolute_index_c] = True
        
        return new_dropped_index_c
    

    def _undersample_single_class(self, n_target_samples_c, 
                                  absolute_index_kept_c, random_state):
        """Return the absolute index of samples after balance-cascade 
        under-sampling of a single class"""

        replacement = self.replacement_

        return random_state.choice(
            absolute_index_kept_c,
            size=n_target_samples_c,
            replace=replacement)
            
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
    def _fit_resample(self, X, y, *, y_pred_proba, 
                      dropped_index, keep_populations:dict, 
                      classes_, encode_map, sample_weight=None):
        
        n_samples, n_classes = X.shape[0], classes_.shape[0]

        # Check random_state and predict probabilities
        replacement = self.replacement_
        random_state = check_random_state(self.random_state)
        y_pred_proba = check_pred_proba(y_pred_proba, n_samples, n_classes, dtype=np.float64)

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
                        random_state=random_state)

                # If no sufficient samples in class C and `replacement` is False
                # raise an RuntimeError
                else: raise ValueError(
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
                error_kept_c=np.abs(
                    np.ones(n_samples_c) - \
                    y_pred_proba_kept[class_index_mask_kept, encode_map[target_class]])
                new_dropped_index_c = self._get_new_dropped_index_single_class(
                    error_kept_c=error_kept_c,
                    n_samples=n_samples,
                    n_drop_samples_c=n_drop_samples_c,
                    absolute_index_kept_c=absolute_index_kept_c,
                )
                new_dropped_index = new_dropped_index | new_dropped_index_c
        
        # Concatenate the result
        index_bcu = np.hstack(absolute_index_resample_list)

        # Store the final undersample indexes
        self.sample_indices_ = index_bcu

        # Return the resampled X, y
        # also return resampled sample_weight (if sample_weight is not None)
        if sample_weight is not None:
            # sample_weight is already validated in super().fit_resample()
            sample_weight_under = _safe_indexing(sample_weight, index_bcu)
            return _safe_indexing(X, index_bcu), _safe_indexing(y, index_bcu), \
                sample_weight_under, new_dropped_index
        else:
            return _safe_indexing(X, index_bcu), _safe_indexing(y, index_bcu), \
                new_dropped_index
        

    def _more_tags(self):
        return {
            "X_types": ["2darray", "string", "sparse", "dataframe"],
            "sample_indices": True,
            "allow_nan": True,
        }

# %%

if __name__ == "__main__":
    from collections import Counter
    from sklearn.datasets import make_classification
    from sklearn.ensemble import AdaBoostClassifier

    X, y = make_classification(n_classes=3, class_sep=2,
        weights=[0.1, 0.3, 0.6], n_informative=3, n_redundant=1, flip_y=0,
        n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
    
    origin_distr = dict(Counter(y))
    print('Original dataset shape %s' % origin_distr)
    target_distr = {2: 200, 1: 100, 0: 90}

    sampling_strategy_ = {2: 200, 1: 100, 0: 90}
    print('Target dataset shape %s' % sampling_strategy_)

    sample_weight = np.full_like(y, fill_value=1/y.shape[0], dtype=float)
    clf = AdaBoostClassifier(
        n_estimators=50,
    ).fit(X, y)

    y_pred_proba = clf.predict_proba(X)
    dropped_index = np.full_like(y, fill_value=False, dtype=bool)

    n_estimators = 10
    i_iter = 0

    keep_ratios = {
        label: np.power(
            (target_distr[label]/origin_distr[label]), 1/(n_estimators)
        ) for label in origin_distr.keys()
    }
    keep_populations = {
        label: int(
            origin_distr[label]*np.power(keep_ratios[label], i_iter+1) + 1e-5
            )
        for label in origin_distr.keys()
    }

    bcu = BalanceCascadeSampler(
        sampling_strategy=sampling_strategy_, 
        replacement=False, 
        random_state=None, 
    )
    X_res, y_res, sample_weight_res, new_dropped_index = bcu.fit_resample(
        X, y, y_pred_proba=y_pred_proba, 
        dropped_index=dropped_index,
        keep_populations=keep_populations,
        sample_weight=sample_weight)
    print('Resampled dataset shape %s' % Counter(y_res))
