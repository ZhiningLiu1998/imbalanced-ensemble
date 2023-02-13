"""Class to perform random under-sampling."""
# Adapted from imbalanced-learn

# Authors: Guillaume Lemaitre
#          Christos Aridas
#          Zhining Liu <zhining.liu@outlook.com>
# License: MIT

# %%
LOCAL_DEBUG = False

if not LOCAL_DEBUG:
    from ....utils._docstring import Substitution, _random_state_docstring
    from ....utils._validation import _deprecate_positional_args, check_target_type
    from ..base import BaseUnderSampler
else:  # pragma: no cover
    import sys  # For local test

    sys.path.append("../../..")
    from sampler._under_sampling.base import BaseUnderSampler
    from utils._docstring import Substitution
    from utils._docstring import _random_state_docstring
    from utils._validation import _deprecate_positional_args, check_target_type

import numpy as np
from sklearn.utils import _safe_indexing, check_random_state


@Substitution(
    sampling_strategy=BaseUnderSampler._sampling_strategy_docstring,
    random_state=_random_state_docstring,
)
class RandomUnderSampler(BaseUnderSampler):
    """Class to perform random under-sampling.

    Under-sample the majority class(es) by randomly picking samples
    with or without replacement.

    Read more in the `User Guide <https://imbalanced-learn.org/stable/under_sampling.html#controlled-under-sampling>`_.

    Parameters
    ----------
    {sampling_strategy}

    {random_state}

    replacement : bool, default=False
        Whether the sample is with or without replacement.

    Attributes
    ----------
    sample_indices_ : ndarray of shape (n_new_samples,)
        Indices of the samples selected.

    See Also
    --------
    NearMiss : Undersample using near-miss samples.

    Notes
    -----
    Supports multi-class resampling by sampling each class independently.
    Supports heterogeneous data as object array containing string and numeric
    data.

    Examples
    --------
    >>> from collections import Counter
    >>> from sklearn.datasets import make_classification
    >>> from imbens.sampler._under_sampling import \
RandomUnderSampler # doctest: +NORMALIZE_WHITESPACE
    >>> X, y = make_classification(n_classes=2, class_sep=2,
    ...  weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
    ... n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
    >>> print('Original dataset shape %s' % Counter(y))
    Original dataset shape Counter({{1: 900, 0: 100}})
    >>> rus = RandomUnderSampler(random_state=42)
    >>> X_res, y_res = rus.fit_resample(X, y)
    >>> print('Resampled dataset shape %s' % Counter(y_res))
    Resampled dataset shape Counter({{0: 100, 1: 100}})
    """

    @_deprecate_positional_args
    def __init__(
        self, *, sampling_strategy="auto", random_state=None, replacement=False
    ):
        super().__init__(sampling_strategy=sampling_strategy)
        self.random_state = random_state
        self.replacement = replacement

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

    def _fit_resample(self, X, y, sample_weight=None, sample_proba=None):
        random_state = check_random_state(self.random_state)

        if sample_proba is None:
            pass
        elif not isinstance(sample_proba, (np.ndarray, list)):
            raise TypeError(
                f"`sample_proba` should be an array-like of shape (n_samples,),"
                f" got {type(sample_proba)} instead."
            )
        else:
            sample_proba = np.asarray(sample_proba)
            if sample_proba.shape != y.shape:
                raise ValueError(
                    f"`sample_proba` should be of shape {y.shape}, got {sample_proba.shape}."
                )
            else:
                try:
                    sample_proba = sample_proba.astype(float)
                except Exception as e:
                    e_args = list(e.args)
                    e_args[0] += (
                        f"\n`sample_proba` should be an array-like with dtype == float,"
                        + f" please check your usage."
                    )
                    e.args = tuple(e_args)
                    raise e

        idx_under = np.empty((0,), dtype=int)

        for target_class in np.unique(y):
            class_idx = y == target_class
            if target_class in self.sampling_strategy_.keys():
                if sample_proba is not None:
                    probabilities = np.array(sample_proba[class_idx]).astype(float)
                    probabilities /= probabilities.sum()
                else:
                    probabilities = None
                n_samples = self.sampling_strategy_[target_class]
                index_target_class = random_state.choice(
                    range(np.count_nonzero(class_idx)),
                    size=n_samples,
                    replace=self.replacement,
                    p=probabilities,
                )
            else:
                index_target_class = slice(None)

            idx_under = np.concatenate(
                (
                    idx_under,
                    np.flatnonzero(class_idx)[index_target_class],
                ),
                axis=0,
            )

        self.sample_indices_ = idx_under

        if sample_weight is not None:
            # sample_weight is already validated in self.fit_resample()
            sample_weight_under = _safe_indexing(sample_weight, idx_under)
            return (
                _safe_indexing(X, idx_under),
                _safe_indexing(y, idx_under),
                sample_weight_under,
            )
        else:
            return _safe_indexing(X, idx_under), _safe_indexing(y, idx_under)

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

    X, y = make_classification(
        n_classes=3,
        class_sep=2,
        weights=[0.1, 0.3, 0.6],
        n_informative=3,
        n_redundant=1,
        flip_y=0,
        n_features=20,
        n_clusters_per_class=1,
        n_samples=1000,
        random_state=10,
    )
    print('Original dataset shape %s' % Counter(y))

    origin_distr = Counter(y)
    target_distr = {2: 200, 1: 100, 0: 100}

    undersampler = RandomUnderSampler(random_state=42, sampling_strategy=target_distr)
    X_res, y_res, weight_res = undersampler.fit_resample(X, y, sample_weight=y)

    print('Resampled dataset shape %s' % Counter(y_res))
    print('Test resampled weight shape %s' % Counter(weight_res))

# %%
