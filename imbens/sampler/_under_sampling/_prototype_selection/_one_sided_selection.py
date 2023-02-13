"""Class to perform under-sampling based on one-sided selection method."""
# Adapted from imbalanced-learn

# Authors: Guillaume Lemaitre
#          Christos Aridas
#          Zhining Liu <zhining.liu@outlook.com>
# License: MIT

# %%
LOCAL_DEBUG = False

if not LOCAL_DEBUG:
    from ....utils._docstring import (
        Substitution,
        _n_jobs_docstring,
        _random_state_docstring,
    )
    from ....utils._validation import _deprecate_positional_args
    from ..base import BaseCleaningSampler
    from ._tomek_links import TomekLinks
else:  # pragma: no cover
    import sys  # For local test

    sys.path.append("../../..")
    from sampler._under_sampling.base import BaseCleaningSampler
    from sampler._under_sampling._prototype_selection._tomek_links import TomekLinks
    from utils._docstring import _n_jobs_docstring, Substitution
    from utils._docstring import _random_state_docstring
    from utils._validation import _deprecate_positional_args

from collections import Counter

import numpy as np
from sklearn.base import clone
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import _safe_indexing, check_random_state


@Substitution(
    sampling_strategy=BaseCleaningSampler._sampling_strategy_docstring,
    n_jobs=_n_jobs_docstring,
    random_state=_random_state_docstring,
)
class OneSidedSelection(BaseCleaningSampler):
    """Class to perform under-sampling based on one-sided selection method.

    Read more in the `User Guide <https://imbalanced-learn.org/stable/under_sampling.html#condensed-nearest-neighbors>`_.

    Parameters
    ----------
    {sampling_strategy}

    {random_state}

    n_neighbors : int or estimator object, default=None
        If ``int``, size of the neighbourhood to consider to compute the
        nearest neighbors. If object, an estimator that inherits from
        :class:`~sklearn.neighbors.base.KNeighborsMixin` that will be used to
        find the nearest-neighbors. If `None`, a
        :class:`~sklearn.neighbors.KNeighborsClassifier` with a 1-NN rules will
        be used.

    n_seeds_S : int, default=1
        Number of samples to extract in order to build the set S.

    {n_jobs}

    Attributes
    ----------
    sample_indices_ : ndarray of shape (n_new_samples,)
        Indices of the samples selected.

    See Also
    --------
    EditedNearestNeighbours : Undersample by editing noisy samples.

    Notes
    -----
    The method is based on [1]_.

    Supports multi-class resampling. A one-vs.-one scheme is used when sampling
    a class as proposed in [1]_. For each class to be sampled, all samples of
    this class and the minority class are used during the sampling procedure.

    References
    ----------
    .. [1] M. Kubat, S. Matwin, "Addressing the curse of imbalanced training
       sets: one-sided selection," In ICML, vol. 97, pp. 179-186, 1997.

    Examples
    --------

    >>> from collections import Counter
    >>> from sklearn.datasets import make_classification
    >>> from imbens.sampler._under_sampling import \
    OneSidedSelection # doctest: +NORMALIZE_WHITESPACE
    >>> X, y = make_classification(n_classes=2, class_sep=2,
    ... weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
    ... n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
    >>> print('Original dataset shape %s' % Counter(y))
    Original dataset shape Counter({{1: 900, 0: 100}})
    >>> oss = OneSidedSelection(random_state=42)
    >>> X_res, y_res = oss.fit_resample(X, y)
    >>> print('Resampled dataset shape %s' % Counter(y_res))
    Resampled dataset shape Counter({{1: 496, 0: 100}})
    """

    @_deprecate_positional_args
    def __init__(
        self,
        *,
        sampling_strategy="auto",
        random_state=None,
        n_neighbors=None,
        n_seeds_S=1,
        n_jobs=None,
    ):
        super().__init__(sampling_strategy=sampling_strategy)
        self.random_state = random_state
        self.n_neighbors = n_neighbors
        self.n_seeds_S = n_seeds_S
        self.n_jobs = n_jobs

    def _validate_estimator(self):
        """Private function to create the NN estimator"""
        if self.n_neighbors is None:
            self.estimator_ = KNeighborsClassifier(n_neighbors=1, n_jobs=self.n_jobs)
        elif isinstance(self.n_neighbors, int):
            self.estimator_ = KNeighborsClassifier(
                n_neighbors=self.n_neighbors, n_jobs=self.n_jobs
            )
        elif isinstance(self.n_neighbors, KNeighborsClassifier):
            self.estimator_ = clone(self.n_neighbors)
        else:
            raise ValueError(
                f"`n_neighbors` has to be a int or an object"
                f" inherited from KNeighborsClassifier."
                f" Got {type(self.n_neighbors)} instead."
            )

    def _fit_resample(self, X, y, sample_weight=None):
        self._validate_estimator()

        random_state = check_random_state(self.random_state)
        target_stats = Counter(y)
        class_minority = min(target_stats, key=target_stats.get)

        idx_under = np.empty((0,), dtype=int)

        for target_class in np.unique(y):
            if target_class in self.sampling_strategy_.keys():
                # select a sample from the current class
                idx_maj = np.flatnonzero(y == target_class)
                sel_idx_maj = random_state.randint(
                    low=0, high=target_stats[target_class], size=self.n_seeds_S
                )
                idx_maj_sample = idx_maj[sel_idx_maj]

                minority_class_indices = np.flatnonzero(y == class_minority)
                C_indices = np.append(minority_class_indices, idx_maj_sample)

                # create the set composed of all minority samples and one
                # sample from the current class.
                C_x = _safe_indexing(X, C_indices)
                C_y = _safe_indexing(y, C_indices)

                # create the set S with removing the seed from S
                # since that it will be added anyway
                idx_maj_extracted = np.delete(idx_maj, sel_idx_maj, axis=0)
                S_x = _safe_indexing(X, idx_maj_extracted)
                S_y = _safe_indexing(y, idx_maj_extracted)
                self.estimator_.fit(C_x, C_y)
                pred_S_y = self.estimator_.predict(S_x)

                S_misclassified_indices = np.flatnonzero(pred_S_y != S_y)
                idx_tmp = idx_maj_extracted[S_misclassified_indices]
                idx_under = np.concatenate((idx_under, idx_maj_sample, idx_tmp), axis=0)
            else:
                idx_under = np.concatenate(
                    (idx_under, np.flatnonzero(y == target_class)), axis=0
                )

        X_resampled = _safe_indexing(X, idx_under)
        y_resampled = _safe_indexing(y, idx_under)

        # apply Tomek cleaning
        tl = TomekLinks(sampling_strategy=list(self.sampling_strategy_.keys()))
        X_cleaned, y_cleaned = tl.fit_resample(X_resampled, y_resampled)

        self.sample_indices_ = _safe_indexing(idx_under, tl.sample_indices_)

        idx_under = self.sample_indices_
        if sample_weight is not None:
            # sample_weight is already validated in self.fit_resample()
            sample_weight_under = _safe_indexing(sample_weight, idx_under)
            return X_cleaned, y_cleaned, sample_weight_under
        else:
            return X_cleaned, y_cleaned

    def _more_tags(self):  # pragma: no cover
        return {"sample_indices": True}


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
    target_distr = [1, 2]
    # target_distr = {2: 200, 1: 100, 0: 100}

    undersampler = OneSidedSelection(random_state=42, sampling_strategy=target_distr)
    X_res, y_res, weight_res = undersampler.fit_resample(X, y, sample_weight=y)

    print('Resampled dataset shape %s' % Counter(y_res))
    print('Test resampled weight shape %s' % Counter(weight_res))

# %%
