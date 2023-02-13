"""Base class for sampling
"""
# Adapted from imbalanced-learn

# Authors: Guillaume Lemaitre
#          Christos Aridas
#          Zhining Liu <zhining.liu@outlook.com>
# License: MIT

# %%
LOCAL_DEBUG = False

if not LOCAL_DEBUG:
    from ..utils._validation import (
        ArraysTransformer,
        _deprecate_positional_args,
        check_sampling_strategy,
        check_target_type,
    )
else:  # pragma: no cover
    import sys  # For local test

    sys.path.append("..")
    from utils._validation import (
        ArraysTransformer,
        _deprecate_positional_args,
        check_sampling_strategy,
        check_target_type,
    )

from abc import ABCMeta, abstractmethod

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.preprocessing import label_binarize
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import _check_sample_weight


class SamplerMixin(BaseEstimator, metaclass=ABCMeta):
    """Mixin class for samplers with abstract method.

    Warning: This class should not be used directly. Use the derive classes
    instead.
    """

    _estimator_type = "sampler"

    def fit(self, X, y):
        """Check inputs and statistics of the sampler.

        You should use ``fit_resample`` in all cases.

        Parameters
        ----------
        X : {array-like, dataframe, sparse matrix} of shape \
                (n_samples, n_features)
            Data array.

        y : array-like of shape (n_samples,)
            Target array.

        Returns
        -------
        self : object
            Return the instance itself.
        """
        X, y, _ = self._check_X_y(X, y)
        self.sampling_strategy_ = check_sampling_strategy(
            self.sampling_strategy, y, self._sampling_type
        )
        return self

    @_deprecate_positional_args
    def fit_resample(self, X, y, *, sample_weight=None, **kwargs):
        """Resample the dataset.

        Parameters
        ----------
        X : {array-like, dataframe, sparse matrix} of shape \
                (n_samples, n_features)
            Matrix containing the data which have to be sampled.

        y : array-like of shape (n_samples,)
            Corresponding label for each sample in X.

        sample_weight : array-like of shape (n_samples,), default=None
            Corresponding weight for each sample in X. 

            - If ``None``, perform normal resampling and return 
              ``(X_resampled, y_resampled)``.
            - If array-like, the given ``sample_weight`` will be resampled 
              along with ``X`` and ``y``, and the *resampled* sample weights 
              will be added to returns. The function will return
              ``(X_resampled, y_resampled, sample_weight_resampled)``.

        Returns
        -------
        X_resampled : {array-like, dataframe, sparse matrix} of shape \
                (n_samples_new, n_features)
            The array containing the resampled data.

        y_resampled : array-like of shape (n_samples_new,)
            The corresponding label of `X_resampled`.
        
        sample_weight_resampled : array-like of shape (n_samples_new,), default=None
            The corresponding weight of `X_resampled`.
            *Only will be returned if input sample_weight is not* ``None``.
        """
        check_classification_targets(y)
        arrays_transformer = ArraysTransformer(X, y)
        X, y, binarize_y = self._check_X_y(X, y)

        self.sampling_strategy_ = check_sampling_strategy(
            self.sampling_strategy, y, self._sampling_type
        )

        if sample_weight is None:
            output = self._fit_resample(X, y, **kwargs)
        else:
            try:
                sample_weight = _check_sample_weight(sample_weight, X, dtype=np.float64)
            except Exception as e:
                e_args = list(e.args)
                e_args[0] += (
                    f"\n'sample_weight' should be an array-like of shape (n_samples,),"
                    + f" got {type(sample_weight)}, please check your usage."
                )
                e.args = tuple(e_args)
                raise e
            else:
                output = self._fit_resample(X, y, sample_weight=sample_weight, **kwargs)

        y_ = (
            label_binarize(output[1], classes=np.unique(y)) if binarize_y else output[1]
        )

        X_, y_ = arrays_transformer.transform(output[0], y_)
        if len(output) == 2:
            output = (X_, y_)
        else:  # with sample_weight
            output = tuple([X_, y_] + [output[i] for i in range(2, len(output))])
        return output

    @abstractmethod
    def _fit_resample(self, X, y, **kwargs):  # pragma: no cover
        """Base method defined in each sampler to defined the sampling
        strategy.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Matrix containing the data which have to be sampled.

        y : array-like of shape (n_samples,)
            Corresponding label for each sample in X.

        Returns
        -------
        X_resampled : {ndarray, sparse matrix} of shape \
                (n_samples_new, n_features)
            The array containing the resampled data.

        y_resampled : ndarray of shape (n_samples_new,)
            The corresponding label of `X_resampled`.

        """
        pass


class BaseSampler(SamplerMixin):
    """Base class for sampling algorithms.

    Warning: This class should not be used directly. Use the derive classes
    instead.
    """

    def __init__(self, sampling_strategy="auto"):
        self.sampling_strategy = sampling_strategy

    def _check_X_y(self, X, y, accept_sparse=None):
        if accept_sparse is None:
            accept_sparse = ["csr", "csc"]
        y, binarize_y = check_target_type(y, indicate_one_vs_all=True)
        X, y = self._validate_data(X, y, reset=True, accept_sparse=accept_sparse)
        return X, y, binarize_y

    def _more_tags(self):  # pragma: no cover
        return {"X_types": ["2darray", "sparse", "dataframe"]}


def _identity(X, y):  # pragma: no cover
    return X, y


class FunctionSampler(BaseSampler):
    """Construct a sampler from calling an arbitrary callable.

    Parameters
    ----------
    func : callable, default=None
        The callable to use for the transformation. This will be passed the
        same arguments as transform, with args and kwargs forwarded. If func is
        None, then func will be the identity function.

    accept_sparse : bool, default=True
        Whether sparse input are supported. By default, sparse inputs are
        supported.

    kw_args : dict, default=None
        The keyword argument expected by ``func``.

    validate : bool, default=True
        Whether or not to bypass the validation of ``X`` and ``y``. Turning-off
        validation allows to use the ``FunctionSampler`` with any type of
        data.

    See Also
    --------

    sklearn.preprocessing.FunctionTransfomer : Stateless transformer.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.datasets import make_classification
    >>> from imbens import FunctionSampler
    >>> X, y = make_classification(n_classes=2, class_sep=2,
    ... weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
    ... n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)

    We can create to select only the first ten samples for instance.

    >>> def func(X, y):
    ...   return X[:10], y[:10]
    >>> sampler = FunctionSampler(func=func)
    >>> X_res, y_res = sampler.fit_resample(X, y)
    >>> np.all(X_res == X[:10])
    True
    >>> np.all(y_res == y[:10])
    True

    We can also create a specific function which take some arguments.

    >>> from collections import Counter
    >>> from imbens.sampler._under_sampling import RandomUnderSampler
    >>> def func(X, y, sampling_strategy, random_state):
    ...   return RandomUnderSampler(
    ...       sampling_strategy=sampling_strategy,
    ...       random_state=random_state).fit_resample(X, y)
    >>> sampler = FunctionSampler(func=func,
    ...                           kw_args={'sampling_strategy': 'auto',
    ...                                    'random_state': 0})
    >>> X_res, y_res = sampler.fit_resample(X, y)
    >>> print(f'Resampled dataset shape {sorted(Counter(y_res).items())}')
    Resampled dataset shape [(0, 100), (1, 100)]
    """

    _sampling_type = "bypass"

    @_deprecate_positional_args
    def __init__(self, *, func=None, accept_sparse=True, kw_args=None, validate=True):
        super().__init__()
        self.func = func
        self.accept_sparse = accept_sparse
        self.kw_args = kw_args
        self.validate = validate

    def fit(self, X, y):
        """Check inputs and statistics of the sampler.

        You should use ``fit_resample`` in all cases.

        Parameters
        ----------
        X : {array-like, dataframe, sparse matrix} of shape \
                (n_samples, n_features)
            Data array.

        y : array-like of shape (n_samples,)
            Target array.

        Returns
        -------
        self : object
            Return the instance itself.
        """
        # we need to overwrite SamplerMixin.fit to bypass the validation
        if self.validate:
            check_classification_targets(y)
            X, y, _ = self._check_X_y(X, y, accept_sparse=self.accept_sparse)

        self.sampling_strategy_ = check_sampling_strategy(
            self.sampling_strategy, y, self._sampling_type
        )

        return self

    def fit_resample(self, X, y):
        """Resample the dataset.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Matrix containing the data which have to be sampled.

        y : array-like of shape (n_samples,)
            Corresponding label for each sample in X.

        Returns
        -------
        X_resampled : {array-like, sparse matrix} of shape \
                (n_samples_new, n_features)
            The array containing the resampled data.

        y_resampled : array-like of shape (n_samples_new,)
            The corresponding label of `X_resampled`.
        """
        arrays_transformer = ArraysTransformer(X, y)

        if self.validate:
            check_classification_targets(y)
            X, y, binarize_y = self._check_X_y(X, y, accept_sparse=self.accept_sparse)

        self.sampling_strategy_ = check_sampling_strategy(
            self.sampling_strategy, y, self._sampling_type
        )

        output = self._fit_resample(X, y)

        if self.validate:

            y_ = (
                label_binarize(output[1], classes=np.unique(y))
                if binarize_y
                else output[1]
            )
            X_, y_ = arrays_transformer.transform(output[0], y_)
            return (X_, y_) if len(output) == 2 else (X_, y_, output[2])

        return output

    def _fit_resample(self, X, y):
        func = _identity if self.func is None else self.func
        output = func(X, y, **(self.kw_args if self.kw_args else {}))
        return output


# %%
