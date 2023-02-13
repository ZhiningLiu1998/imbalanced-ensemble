"""Class to perform under-sampling by generating centroids based on
clustering.
"""
# Adapted from imbalanced-learn

# Authors: Guillaume Lemaitre
#          Fernando Nogueira
#          Christos Aridas
#          Zhining Liu <zhining.liu@outlook.com>
# License: MIT

# %%
LOCAL_DEBUG = False

if not LOCAL_DEBUG:
    from ....utils._docstring import Substitution, _random_state_docstring
    from ....utils._validation import _deprecate_positional_args
    from ..base import BaseUnderSampler
else:  # pragma: no cover
    import sys  # For local test

    sys.path.append("../../..")
    from sampler._under_sampling.base import BaseUnderSampler
    from utils._docstring import Substitution
    from utils._docstring import _random_state_docstring
    from utils._validation import _deprecate_positional_args

import numpy as np
from scipy import sparse
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import _safe_indexing

VOTING_KIND = ("auto", "hard", "soft")


@Substitution(
    sampling_strategy=BaseUnderSampler._sampling_strategy_docstring,
    random_state=_random_state_docstring,
)
class ClusterCentroids(BaseUnderSampler):
    """Undersample by generating centroids based on clustering methods.

    Method that under samples the majority class by replacing a
    cluster of majority samples by the cluster centroid of a KMeans
    algorithm.  This algorithm keeps N majority samples by fitting the
    KMeans algorithm with N cluster to the majority class and using
    the coordinates of the N cluster centroids as the new majority
    samples.

    Read more in the `User Guide <https://imbalanced-learn.org/stable/under_sampling.html#cluster-centroids>`_.

    Parameters
    ----------
    {sampling_strategy}

    {random_state}

    estimator : estimator object, default=None
        Pass a :class:`~sklearn.cluster.KMeans` estimator. By default, it will
        be a default :class:`~sklearn.cluster.KMeans` estimator.

    voting : {{"hard", "soft", "auto"}}, default='auto'
        Voting strategy to generate the new samples:

        - If ``'hard'``, the nearest-neighbors of the centroids found using the
          clustering algorithm will be used.
        - If ``'soft'``, the centroids found by the clustering algorithm will
          be used.
        - If ``'auto'``, if the input is sparse, it will default on ``'hard'``
          otherwise, ``'soft'`` will be used.

    See Also
    --------
    EditedNearestNeighbours : Under-sampling by editing samples.

    CondensedNearestNeighbour: Under-sampling by condensing samples.

    Notes
    -----
    Supports multi-class resampling by sampling each class independently.

    Examples
    --------

    >>> from collections import Counter
    >>> from sklearn.datasets import make_classification
    >>> from imbens.sampler._under_sampling import \
ClusterCentroids # doctest: +NORMALIZE_WHITESPACE
    >>> X, y = make_classification(n_classes=2, class_sep=2,
    ... weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
    ... n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
    >>> print('Original dataset shape %s' % Counter(y))
    Original dataset shape Counter({{1: 900, 0: 100}})
    >>> cc = ClusterCentroids(random_state=42)
    >>> X_res, y_res = cc.fit_resample(X, y)
    >>> print('Resampled dataset shape %s' % Counter(y_res))
    ... # doctest: +ELLIPSIS
    Resampled dataset shape Counter({{...}})
    """

    @_deprecate_positional_args
    def __init__(
        self,
        *,
        sampling_strategy="auto",
        random_state=None,
        estimator=None,
        voting="auto",
    ):
        super().__init__(sampling_strategy=sampling_strategy)
        self.random_state = random_state
        self.estimator = estimator
        self.voting = voting

    def _validate_estimator(self):
        """Private function to create the KMeans estimator"""
        if self.estimator is None:
            self.estimator_ = KMeans(n_init='auto', random_state=self.random_state)
        elif isinstance(self.estimator, KMeans):
            self.estimator_ = clone(self.estimator)
        else:
            raise ValueError(
                f"`estimator` has to be a KMeans clustering."
                f" Got {type(self.estimator)} instead."
            )

    def _generate_sample(self, X, y, centroids, target_class):
        if self.voting_ == "hard":
            nearest_neighbors = NearestNeighbors(n_neighbors=1)
            nearest_neighbors.fit(X, y)
            indices = nearest_neighbors.kneighbors(centroids, return_distance=False)
            X_new = _safe_indexing(X, np.squeeze(indices))
        else:
            if sparse.issparse(X):
                X_new = sparse.csr_matrix(centroids, dtype=X.dtype)
            else:
                X_new = centroids

        y_new = np.array([target_class] * centroids.shape[0], dtype=y.dtype)

        return X_new, y_new

    def _fit_resample(self, X, y, sample_weight=None):

        if sample_weight is not None:
            raise NotImplementedError(
                f"'ClusterCentroids' under-sampler does not support"
                f" resampling data that comes with 'sample_weight'."
            )

        self._validate_estimator()

        if self.voting == "auto":
            if sparse.issparse(X):
                self.voting_ = "hard"
            else:
                self.voting_ = "soft"
        else:
            if self.voting in VOTING_KIND:
                self.voting_ = self.voting
            else:
                raise ValueError(
                    f"'voting' needs to be one of {VOTING_KIND}. "
                    f"Got {self.voting} instead."
                )

        X_resampled, y_resampled = [], []
        for target_class in np.unique(y):
            target_class_indices = np.flatnonzero(y == target_class)
            if target_class in self.sampling_strategy_.keys():
                n_samples = self.sampling_strategy_[target_class]
                self.estimator_.set_params(**{"n_clusters": n_samples})
                self.estimator_.fit(_safe_indexing(X, target_class_indices))
                X_new, y_new = self._generate_sample(
                    _safe_indexing(X, target_class_indices),
                    _safe_indexing(y, target_class_indices),
                    self.estimator_.cluster_centers_,
                    target_class,
                )
                X_resampled.append(X_new)
                y_resampled.append(y_new)
            else:
                X_resampled.append(_safe_indexing(X, target_class_indices))
                y_resampled.append(_safe_indexing(y, target_class_indices))

        if sparse.issparse(X):
            X_resampled = sparse.vstack(X_resampled)
        else:
            X_resampled = np.vstack(X_resampled)
        y_resampled = np.hstack(y_resampled)

        return X_resampled, np.array(y_resampled, dtype=y.dtype)

        # if sample_weight is not None:
        #     # sample_weight is already validated in self.fit_resample()
        #     sample_weight_under = _safe_indexing(sample_weight, idx_under)
        #     return _safe_indexing(X, idx_under), _safe_indexing(y, idx_under), sample_weight_under
        # else: return _safe_indexing(X, idx_under), _safe_indexing(y, idx_under)

    def _more_tags(self):  # pragma: no cover
        return {"sample_indices": False}


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

    rus = ClusterCentroids(
        random_state=42,
        sampling_strategy=target_distr,
        voting="hard",
    )
    X_res, y_res = rus.fit_resample(X, y)

    print('Resampled dataset shape %s' % Counter(y_res))

# %%
