"""Class performing under-sampling based on the neighbourhood cleaning rule."""
# Adapted from imbalanced-learn

# Authors: Guillaume Lemaitre
#          Christos Aridas
#          Zhining Liu <zhining.liu@outlook.com>
# License: MIT

# %%
LOCAL_DEBUG = False

if not LOCAL_DEBUG:
    from ....utils._docstring import Substitution, _n_jobs_docstring
    from ....utils._validation import _deprecate_positional_args, check_neighbors_object
    from ..base import BaseCleaningSampler
    from ._edited_nearest_neighbours import EditedNearestNeighbours
else:  # pragma: no cover
    import sys  # For local test

    sys.path.append("../../..")
    from sampler._under_sampling.base import BaseCleaningSampler
    from sampler._under_sampling._prototype_selection._edited_nearest_neighbours import (
        EditedNearestNeighbours,
    )
    from utils._docstring import _n_jobs_docstring, Substitution
    from utils._validation import _deprecate_positional_args, check_neighbors_object

from collections import Counter

import numpy as np
from scipy.stats import mode
from sklearn.utils import _safe_indexing

SEL_KIND = ("all", "mode")


@Substitution(
    sampling_strategy=BaseCleaningSampler._sampling_strategy_docstring,
    n_jobs=_n_jobs_docstring,
)
class NeighbourhoodCleaningRule(BaseCleaningSampler):
    """Undersample based on the neighbourhood cleaning rule.

    This class uses ENN and a k-NN to remove noisy samples from the datasets.

    Read more in the `User Guide <https://imbalanced-learn.org/stable/under_sampling.html#condensed-nearest-neighbors>`_.

    Parameters
    ----------
    {sampling_strategy}

    n_neighbors : int or estimator object, default=3
        If ``int``, size of the neighbourhood to consider to compute the
        nearest neighbors. If object, an estimator that inherits from
        :class:`~sklearn.neighbors.base.KNeighborsMixin` that will be used to
        find the nearest-neighbors. By default, it will be a 3-NN.

    kind_sel : {{"all", "mode"}}, default='all'
        Strategy to use in order to exclude samples in the ENN sampling.

        - If ``'all'``, all neighbours will have to agree with the samples of
          interest to not be excluded.
        - If ``'mode'``, the majority vote of the neighbours will be used in
          order to exclude a sample.

        The strategy `"all"` will be less conservative than `'mode'`. Thus,
        more samples will be removed when `kind_sel="all"` generally.

    threshold_cleaning : float, default=0.5
        Threshold used to whether consider a class or not during the cleaning
        after applying ENN. A class will be considered during cleaning when:

        Ci > C x T ,

        where Ci and C is the number of samples in the class and the data set,
        respectively and theta is the threshold.

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
    See the original paper: [1]_.

    Supports multi-class resampling. A one-vs.-rest scheme is used when
    sampling a class as proposed in [1]_.

    References
    ----------
    .. [1] J. Laurikkala, "Improving identification of difficult small classes
       by balancing class distribution," Springer Berlin Heidelberg, 2001.

    Examples
    --------
    >>> from collections import Counter
    >>> from sklearn.datasets import make_classification
    >>> from imbens.sampler._under_sampling import \
NeighbourhoodCleaningRule # doctest: +NORMALIZE_WHITESPACE
    >>> X, y = make_classification(n_classes=2, class_sep=2,
    ... weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
    ... n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
    >>> print('Original dataset shape %s' % Counter(y))
    Original dataset shape Counter({{1: 900, 0: 100}})
    >>> ncr = NeighbourhoodCleaningRule()
    >>> X_res, y_res = ncr.fit_resample(X, y)
    >>> print('Resampled dataset shape %s' % Counter(y_res))
    Resampled dataset shape Counter({{1: 877, 0: 100}})
    """

    @_deprecate_positional_args
    def __init__(
        self,
        *,
        sampling_strategy="auto",
        n_neighbors=3,
        kind_sel="all",
        threshold_cleaning=0.5,
        n_jobs=None,
    ):
        super().__init__(sampling_strategy=sampling_strategy)
        self.n_neighbors = n_neighbors
        self.kind_sel = kind_sel
        self.threshold_cleaning = threshold_cleaning
        self.n_jobs = n_jobs

    def _validate_estimator(self):
        """Create the objects required by NCR."""
        self.nn_ = check_neighbors_object(
            "n_neighbors", self.n_neighbors, additional_neighbor=1
        )
        self.nn_.set_params(**{"n_jobs": self.n_jobs})

        if self.kind_sel not in SEL_KIND:
            raise NotImplementedError

        if self.threshold_cleaning > 1 or self.threshold_cleaning < 0:
            raise ValueError(
                f"'threshold_cleaning' is a value between 0 and 1."
                f" Got {self.threshold_cleaning} instead."
            )

    def _fit_resample(self, X, y, sample_weight=None):
        self._validate_estimator()
        enn = EditedNearestNeighbours(
            sampling_strategy=self.sampling_strategy,
            n_neighbors=self.n_neighbors,
            kind_sel="mode",
            n_jobs=self.n_jobs,
        )
        enn.fit_resample(X, y)
        index_not_a1 = enn.sample_indices_
        index_a1 = np.ones(y.shape, dtype=bool)
        index_a1[index_not_a1] = False
        index_a1 = np.flatnonzero(index_a1)

        # clean the neighborhood
        target_stats = Counter(y)
        class_minority = min(target_stats, key=target_stats.get)
        # compute which classes to consider for cleaning for the A2 group
        classes_under_sample = [
            c
            for c, n_samples in target_stats.items()
            if (
                c in self.sampling_strategy_.keys()
                and (n_samples > X.shape[0] * self.threshold_cleaning)
            )
        ]
        self.nn_.fit(X)
        class_minority_indices = np.flatnonzero(y == class_minority)
        X_class = _safe_indexing(X, class_minority_indices)
        y_class = _safe_indexing(y, class_minority_indices)
        nnhood_idx = self.nn_.kneighbors(X_class, return_distance=False)[:, 1:]
        nnhood_label = y[nnhood_idx]
        if self.kind_sel == "mode":
            nnhood_label_majority, _ = mode(nnhood_label, axis=1, keepdims=True)
            nnhood_bool = np.ravel(nnhood_label_majority) == y_class
        elif self.kind_sel == "all":
            nnhood_label_majority = nnhood_label == class_minority
            nnhood_bool = np.all(nnhood_label, axis=1)
        else:
            raise NotImplementedError
        # compute a2 group
        index_a2 = np.ravel(nnhood_idx[~nnhood_bool])
        index_a2 = np.unique(
            [index for index in index_a2 if y[index] in classes_under_sample]
        )

        union_a1_a2 = np.union1d(index_a1, index_a2).astype(int)
        selected_samples = np.ones(y.shape, dtype=bool)
        selected_samples[union_a1_a2] = False
        self.sample_indices_ = np.flatnonzero(selected_samples)

        idx_under = self.sample_indices_
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

    undersampler = NeighbourhoodCleaningRule(sampling_strategy=target_distr)
    X_res, y_res, weight_res = undersampler.fit_resample(X, y, sample_weight=y)

    print('Resampled dataset shape %s' % Counter(y_res))
    print('Test resampled weight shape %s' % Counter(weight_res))

# %%
