"""Base class and original SMOTE methods for over-sampling"""
# Adapted from imbalanced-learn

# Authors: Guillaume Lemaitre
#          Fernando Nogueira
#          Christos Aridas
#          Dzianis Dudnik
#          Zhining Liu <zhining.liu@outlook.com>
# License: MIT

# %%
LOCAL_DEBUG = False

if not LOCAL_DEBUG:
    from ..base import BaseOverSampler
    from ....utils._docstring import _n_jobs_docstring, Substitution
    from ....utils._docstring import _random_state_docstring
    from ....utils._validation import (_deprecate_positional_args,
                                       check_neighbors_object)
else:           # pragma: no cover
    import sys  # For local test
    sys.path.append("../../..")
    from sampler._over_sampling.base import BaseOverSampler
    from utils._docstring import _n_jobs_docstring, Substitution
    from utils._docstring import _random_state_docstring
    from utils._validation import (_deprecate_positional_args,
                                   check_neighbors_object)

import numpy as np
from collections import Counter
from scipy import sparse

from sklearn.preprocessing import normalize
from sklearn.utils import check_random_state
from sklearn.utils import _safe_indexing


# %%

class BaseSMOTE(BaseOverSampler):
    """Base class for the different SMOTE algorithms."""

    def __init__(
        self,
        sampling_strategy="auto",
        random_state=None,
        k_neighbors=5,
        n_jobs=None,
    ):
        super().__init__(sampling_strategy=sampling_strategy)
        self.random_state = random_state
        self.k_neighbors = k_neighbors
        self.n_jobs = n_jobs

    def _validate_estimator(self):
        """Check the NN estimators shared across the different SMOTE
        algorithms.
        """
        self.nn_k_ = check_neighbors_object(
            "k_neighbors", self.k_neighbors, additional_neighbor=1
        )

    def _make_samples(
        self, X, y_dtype, y_type, nn_data, nn_num, n_samples, step_size=1.0
    ):
        """A support function that returns artificial samples constructed along
        the line connecting nearest neighbours.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Points from which the points will be created.

        y_dtype : dtype
            The data type of the targets.

        y_type : str or int
            The minority target value, just so the function can return the
            target values for the synthetic variables with correct length in
            a clear format.

        nn_data : ndarray of shape (n_samples_all, n_features)
            Data set carrying all the neighbours to be used

        nn_num : ndarray of shape (n_samples_all, k_nearest_neighbours)
            The nearest neighbours of each sample in `nn_data`.

        n_samples : int
            The number of samples to generate.

        step_size : float, default=1.0
            The step size to create samples.

        Returns
        -------
        X_new : {ndarray, sparse matrix} of shape (n_samples_new, n_features)
            Synthetically generated samples.

        y_new : ndarray of shape (n_samples_new,)
            Target values for synthetic samples.
        """
        random_state = check_random_state(self.random_state)
        samples_indices = random_state.randint(low=0, high=nn_num.size, size=n_samples)

        # np.newaxis for backwards compatability with random_state
        steps = step_size * random_state.uniform(size=n_samples)[:, np.newaxis]
        rows = np.floor_divide(samples_indices, nn_num.shape[1])
        cols = np.mod(samples_indices, nn_num.shape[1])

        X_new = self._generate_samples(X, nn_data, nn_num, rows, cols, steps)
        y_new = np.full(n_samples, fill_value=y_type, dtype=y_dtype)
        return X_new, y_new

    def _generate_samples(self, X, nn_data, nn_num, rows, cols, steps):
        r"""Generate a synthetic sample.

        The rule for the generation is:

        .. math::
           \mathbf{s_{s}} = \mathbf{s_{i}} + \mathcal{u}(0, 1) \times
           (\mathbf{s_{i}} - \mathbf{s_{nn}}) \,

        where \mathbf{s_{s}} is the new synthetic samples, \mathbf{s_{i}} is
        the current sample, \mathbf{s_{nn}} is a randomly selected neighbors of
        \mathbf{s_{i}} and \mathcal{u}(0, 1) is a random number between [0, 1).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Points from which the points will be created.

        nn_data : ndarray of shape (n_samples_all, n_features)
            Data set carrying all the neighbours to be used.

        nn_num : ndarray of shape (n_samples_all, k_nearest_neighbours)
            The nearest neighbours of each sample in `nn_data`.

        rows : ndarray of shape (n_samples,), dtype=int
            Indices pointing at feature vector in X which will be used
            as a base for creating new samples.

        cols : ndarray of shape (n_samples,), dtype=int
            Indices pointing at which nearest neighbor of base feature vector
            will be used when creating new samples.

        steps : ndarray of shape (n_samples,), dtype=float
            Step sizes for new samples.

        Returns
        -------
        X_new : {ndarray, sparse matrix} of shape (n_samples, n_features)
            Synthetically generated samples.
        """
        diffs = nn_data[nn_num[rows, cols]] - X[rows]

        if sparse.issparse(X):
            sparse_func = type(X).__name__
            steps = getattr(sparse, sparse_func)(steps)
            X_new = X[rows] + steps.multiply(diffs)
        else:
            X_new = X[rows] + steps * diffs

        return X_new.astype(X.dtype)

    def _in_danger_noise(self, nn_estimator, samples, target_class, y, kind="danger"):
        """Estimate if a set of sample are in danger or noise.

        Used by BorderlineSMOTE and SVMSMOTE.

        Parameters
        ----------
        nn_estimator : estimator object
            An estimator that inherits from
            :class:`~sklearn.neighbors.base.KNeighborsMixin` use to determine
            if a sample is in danger/noise.

        samples : {array-like, sparse matrix} of shape (n_samples, n_features)
            The samples to check if either they are in danger or not.

        target_class : int or str
            The target corresponding class being over-sampled.

        y : array-like of shape (n_samples,)
            The true label in order to check the neighbour labels.

        kind : {'danger', 'noise'}, default='danger'
            The type of classification to use. Can be either:

            - If 'danger', check if samples are in danger,
            - If 'noise', check if samples are noise.

        Returns
        -------
        output : ndarray of shape (n_samples,)
            A boolean array where True refer to samples in danger or noise.
        """
        x = nn_estimator.kneighbors(samples, return_distance=False)[:, 1:]
        nn_label = (y[x] != target_class).astype(int)
        n_maj = np.sum(nn_label, axis=1)

        if kind == "danger":
            # Samples are in danger for m/2 <= m' < m
            return np.bitwise_and(
                n_maj >= (nn_estimator.n_neighbors - 1) / 2,
                n_maj < nn_estimator.n_neighbors - 1,
            )
        elif kind == "noise":
            # Samples are noise for m = m'
            return n_maj == nn_estimator.n_neighbors - 1
        else:
            raise NotImplementedError


@Substitution(
    sampling_strategy=BaseOverSampler._sampling_strategy_docstring,
    n_jobs=_n_jobs_docstring,
    random_state=_random_state_docstring,
)
class SMOTE(BaseSMOTE):
    """Class to perform over-sampling using SMOTE.

    This object is an implementation of SMOTE - Synthetic Minority
    Over-sampling Technique as presented in [1]_.

    Read more in the `User Guide <https://imbalanced-learn.org/stable/over_sampling.html#smote-adasyn>`_.

    Parameters
    ----------
    {sampling_strategy}

    {random_state}

    k_neighbors : int or object, default=5
        If ``int``, number of nearest neighbours to used to construct synthetic
        samples.  If object, an estimator that inherits from
        :class:`~sklearn.neighbors.base.KNeighborsMixin` that will be used to
        find the k_neighbors.

    {n_jobs}

    See Also
    --------
    BorderlineSMOTE : Over-sample using the borderline-SMOTE variant.

    SVMSMOTE : Over-sample using the SVM-SMOTE variant.

    ADASYN : Over-sample using ADASYN.

    KMeansSMOTE : Over-sample applying a clustering before to oversample using
        SMOTE.

    Notes
    -----
    See the original papers: [1]_ for more details.

    Supports multi-class resampling. A one-vs.-rest scheme is used as
    originally proposed in [1]_.

    References
    ----------
    .. [1] N. V. Chawla, K. W. Bowyer, L. O.Hall, W. P. Kegelmeyer, "SMOTE:
       synthetic minority over-sampling technique," Journal of artificial
       intelligence research, 321-357, 2002.

    Examples
    --------
    >>> from collections import Counter
    >>> from sklearn.datasets import make_classification
    >>> from imbens.sampler._over_sampling import \
    SMOTE # doctest: +NORMALIZE_WHITESPACE
    >>> X, y = make_classification(n_classes=2, class_sep=2,
    ... weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
    ... n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
    >>> print('Original dataset shape %s' % Counter(y))
    Original dataset shape Counter({{1: 900, 0: 100}})
    >>> sm = SMOTE(random_state=42)
    >>> X_res, y_res = sm.fit_resample(X, y)
    >>> print('Resampled dataset shape %s' % Counter(y_res))
    Resampled dataset shape Counter({{0: 900, 1: 900}})
    """

    @_deprecate_positional_args
    def __init__(
        self,
        *,
        sampling_strategy="auto",
        random_state=None,
        k_neighbors=5,
        n_jobs=None,
    ):
        super().__init__(
            sampling_strategy=sampling_strategy,
            random_state=random_state,
            k_neighbors=k_neighbors,
            n_jobs=n_jobs,
        )

    def _fit_resample(self, X, y, sample_weight=None):
        self._validate_estimator()

        X_resampled = [X.copy()]
        y_resampled = [y.copy()]

        for class_sample, n_samples in self.sampling_strategy_.items():
            if n_samples == 0:
                continue
            target_class_indices = np.flatnonzero(y == class_sample)
            X_class = _safe_indexing(X, target_class_indices)

            self.nn_k_.fit(X_class)
            nns = self.nn_k_.kneighbors(X_class, return_distance=False)[:, 1:]
            X_new, y_new = self._make_samples(
                X_class, y.dtype, class_sample, X_class, nns, n_samples, 1.0
            )
            X_resampled.append(X_new)
            y_resampled.append(y_new)

        if sparse.issparse(X):
            X_resampled = sparse.vstack(X_resampled, format=X.format)
        else:
            X_resampled = np.vstack(X_resampled)
        y_resampled = np.hstack(y_resampled)
        
        # If given sample_weight
        if sample_weight is not None:
            # sample_weight is already validated in self.fit_resample()
            sample_weight_new = \
                np.empty(y_resampled.shape[0] - y.shape[0], dtype=np.float64)
            sample_weight_new[:] = np.mean(sample_weight)
            sample_weight_resampled = np.hstack([sample_weight, sample_weight_new]).reshape(-1, 1)
            sample_weight_resampled = \
                np.squeeze(normalize(sample_weight_resampled, axis=0, norm='l1'))
            return X_resampled, y_resampled, sample_weight_resampled
        else: return X_resampled, y_resampled


# %%

if __name__ == "__main__":  # pragma: no cover
    from collections import Counter
    from sklearn.datasets import make_classification

    X, y = make_classification(n_classes=3, class_sep=2,
        weights=[0.1, 0.3, 0.6], n_informative=3, n_redundant=1, flip_y=0,
        n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
    print('Original dataset shape %s' % Counter(y))

    origin_distr = Counter(y)
    target_distr = {2: 600, 1: 400, 0: 300}
    
    print ('SMOTE Resampling ...')
    sample_weight = np.full_like(y, 1/y.shape[0], dtype=np.float64)
    smote = SMOTE(random_state=42, sampling_strategy=target_distr)
    X_res, y_res, sample_weight_res = smote.fit_resample(X, y, sample_weight=sample_weight)

    print('Resampled dataset shape %s' % Counter(y_res))

# %%
