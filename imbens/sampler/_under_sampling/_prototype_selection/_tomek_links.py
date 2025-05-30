"""Class to perform under-sampling by removing Tomek's links."""

# Adapted from imbalanced-learn

# Authors: Guillaume Lemaitre
#          Fernando Nogueira
#          Christos Aridas
# License: MIT

# %%
LOCAL_DEBUG = False

if not LOCAL_DEBUG:
    from ....utils._docstring import Substitution, _n_jobs_docstring
    from ....utils._validation import _deprecate_positional_args
    from ..base import BaseCleaningSampler
else:  # pragma: no cover
    import sys  # For local test

    sys.path.append("../../..")
    from sampler._under_sampling.base import BaseCleaningSampler
    from utils._docstring import _n_jobs_docstring, Substitution
    from utils._validation import _deprecate_positional_args

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import _safe_indexing


@Substitution(
    sampling_strategy=BaseCleaningSampler._sampling_strategy_docstring,
    n_jobs=_n_jobs_docstring,
)
class TomekLinks(BaseCleaningSampler):
    """Under-sampling by removing Tomek's links.

    Read more in the `User Guide <https://imbalanced-learn.org/stable/under_sampling.html#tomek-s-links>`_.

    Parameters
    ----------
    {sampling_strategy}

    {n_jobs}

    Attributes
    ----------
    sample_indices_ : ndarray of shape (n_new_samples,)
        Indices of the samples selected.

    See Also
    --------
    EditedNearestNeighbours : Undersample by samples edition.

    CondensedNearestNeighbour : Undersample by samples condensation.

    RandomUnderSampling : Randomly under-sample the dataset.

    Notes
    -----
    This method is based on [1]_.

    Supports multi-class resampling. A one-vs.-rest scheme is used as
    originally proposed in [1]_.

    References
    ----------
    .. [1] I. Tomek, "Two modifications of CNN," In Systems, Man, and
       Cybernetics, IEEE Transactions on, vol. 6, pp 769-772, 1976.

    Examples
    --------
    >>> from collections import Counter
    >>> from sklearn.datasets import make_classification
    >>> from imbens.sampler._under_sampling import \
TomekLinks # doctest: +NORMALIZE_WHITESPACE
    >>> X, y = make_classification(n_classes=2, class_sep=2,
    ... weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
    ... n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
    >>> print('Original dataset shape %s' % Counter(y))
    Original dataset shape Counter({{1: 900, 0: 100}})
    >>> tl = TomekLinks()
    >>> X_res, y_res = tl.fit_resample(X, y)
    >>> print('Resampled dataset shape %s' % Counter(y_res))
    Resampled dataset shape Counter({{1: 897, 0: 100}})
    """

    @_deprecate_positional_args
    def __init__(self, *, sampling_strategy="auto", n_jobs=None):
        super().__init__(sampling_strategy=sampling_strategy)
        self.n_jobs = n_jobs

    @staticmethod
    def is_tomek(y, nn_index, class_type):
        """Detect if samples are Tomek's link.

        More precisely, it uses the target vector and the first neighbour of
        every sample point and looks for Tomek pairs. Returning a boolean
        vector with True for majority Tomek links.

        Parameters
        ----------
        y : ndarray of shape (n_samples,)
            Target vector of the data set, necessary to keep track of whether a
            sample belongs to minority or not.

        nn_index : ndarray of shape (len(y),)
            The index of the closes nearest neighbour to a sample point.

        class_type : int or str
            The label of the minority class.

        Returns
        -------
        is_tomek : ndarray of shape (len(y), )
            Boolean vector on len( # samples ), with True for majority samples
            that are Tomek links.
        """
        links = np.zeros(len(y), dtype=bool)

        # find which class to not consider
        class_excluded = [c for c in np.unique(y) if c not in class_type]

        # there is a Tomek link between two samples if they are both nearest
        # neighbors of each others.
        for index_sample, target_sample in enumerate(y):
            if target_sample in class_excluded:
                continue

            if y[nn_index[index_sample]] != target_sample:
                if nn_index[nn_index[index_sample]] == index_sample:
                    links[index_sample] = True

        return links

    def _fit_resample(self, X, y, sample_weight=None):
        # Find the nearest neighbour of every point
        nn = NearestNeighbors(n_neighbors=2, n_jobs=self.n_jobs)
        nn.fit(X)
        nns = nn.kneighbors(X, return_distance=False)[:, 1]

        links = self.is_tomek(y, nns, self.sampling_strategy_)
        self.sample_indices_ = np.flatnonzero(np.logical_not(links))

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

    def __sklearn_tags__(self):  # pragma: no cover
        tags = super().__sklearn_tags__()
        # tags.sample_indices = True
        return tags


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
    print("Original dataset shape %s" % Counter(y))

    origin_distr = Counter(y)
    target_distr = [1, 2]
    # target_distr = {2: 200, 1: 100, 0: 100}

    undersampler = TomekLinks(sampling_strategy=target_distr)
    X_res, y_res, weight_res = undersampler.fit_resample(X, y, sample_weight=y)

    print("Resampled dataset shape %s" % Counter(y_res))
    print("Test resampled weight shape %s" % Counter(weight_res))

# %%
