"""Base classes for all bagging-like methods in imbens.

ResampleBaggingClassifier Base class for all resampling + 
bagging imbalanced ensemble classifier.
"""

# Authors: Zhining Liu <zhining.liu@outlook.com>
# License: MIT

# %%
LOCAL_DEBUG = False

if not LOCAL_DEBUG:
    from ..pipeline import Pipeline
    from ..utils._docstring import (
        FuncGlossarySubstitution,
        FuncSubstitution,
        _get_parameter_docstring,
    )
    from ..utils._validation import (
        _deprecate_positional_args,
        check_sampling_strategy,
        check_target_type,
    )
    from ..utils._validation_data import check_eval_datasets
    from ..utils._validation_param import (
        check_eval_metrics,
        check_train_verbose,
        check_type,
    )
    from .base import MAX_INT, ImbalancedEnsembleClassifierMixin
else:  # pragma: no cover
    import sys  # For local test

    sys.path.append("..")
    from ensemble.base import ImbalancedEnsembleClassifierMixin, MAX_INT
    from pipeline import Pipeline
    from utils._validation_data import check_eval_datasets
    from utils._validation_param import (
        check_train_verbose,
        check_eval_metrics,
        check_type,
    )
    from utils._validation import (
        _deprecate_positional_args,
        check_sampling_strategy,
        check_target_type,
    )
    from utils._docstring import (
        FuncSubstitution,
        FuncGlossarySubstitution,
        _get_parameter_docstring,
    )

import itertools
import numbers
from abc import ABCMeta, abstractmethod
from warnings import warn

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble._base import _partition_estimators
from sklearn.pipeline import Pipeline as skPipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import check_random_state
from sklearn.utils.parallel import Parallel, delayed
from sklearn.utils.random import sample_without_replacement
from sklearn.utils.validation import _check_sample_weight, has_fit_parameter


def _generate_indices(random_state, bootstrap, n_population, n_samples):
    """Draw randomly sampled indices."""
    # Draw sample indices
    if bootstrap:
        indices = random_state.randint(0, n_population, n_samples)
    else:
        indices = sample_without_replacement(
            n_population, n_samples, random_state=random_state
        )

    return indices


def _generate_bagging_indices(
    random_state,
    bootstrap_features,
    bootstrap_samples,
    n_features,
    n_samples,
    max_features,
    max_samples,
):
    """Randomly draw feature and sample indices."""
    # Get valid random state
    random_state = check_random_state(random_state)

    # Draw indices
    feature_indices = _generate_indices(
        random_state, bootstrap_features, n_features, max_features
    )
    sample_indices = _generate_indices(
        random_state, bootstrap_samples, n_samples, max_samples
    )

    return feature_indices, sample_indices


def _parallel_build_estimators(
    n_estimators, ensemble, X, y, sample_weight, seeds, total_n_estimators, verbose
):
    """Private function used to build a batch of estimators within a job."""
    # Retrieve settings
    n_samples, n_features = X.shape
    max_features = ensemble._max_features
    max_samples = ensemble._max_samples
    bootstrap = ensemble.bootstrap
    bootstrap_features = ensemble.bootstrap_features

    # Check if the estimator supports sample_weight
    estimator_ = ensemble.estimator_
    while isinstance(estimator_, skPipeline):  # for Pipelines
        estimator_ = estimator_._final_estimator
    support_sample_weight = has_fit_parameter(estimator_, "sample_weight")
    if not support_sample_weight and sample_weight is not None:
        raise ValueError("The base estimator doesn't support sample weight")

    # Build estimators
    estimators = []
    estimators_features = []
    estimators_n_training_samples = []

    for i in range(n_estimators):
        if verbose > 1:
            print(
                "Building estimator %d of %d for this parallel run "
                "(total %d)..." % (i + 1, n_estimators, total_n_estimators)
            )

        random_state = seeds[i]
        estimator = ensemble._make_estimator(append=False, random_state=random_state)

        # Draw random feature, sample indices
        features, indices = _generate_bagging_indices(
            random_state,
            bootstrap_features,
            bootstrap,
            n_features,
            n_samples,
            max_features,
            max_samples,
        )

        # Draw samples, using sample weights, and then fit
        if support_sample_weight and sample_weight is not None:
            curr_sample_weight = sample_weight.copy()
            estimator.fit(
                (X[indices])[:, features],
                y[indices],
                sample_weight=curr_sample_weight[indices],
            )
        else:
            estimator.fit((X[indices])[:, features], y[indices])

        if hasattr(estimator, 'n_training_samples_'):
            n_training_samples = getattr(estimator, 'n_training_samples_')
        else:
            n_training_samples = len(indices)

        estimators.append(estimator)
        estimators_features.append(features)
        estimators_n_training_samples.append(n_training_samples)

    return estimators, estimators_features, estimators_n_training_samples


_super = BaggingClassifier


class ResampleBaggingClassifier(
    ImbalancedEnsembleClassifierMixin, BaggingClassifier, metaclass=ABCMeta
):
    """Base class for all resampling + bagging imbalanced ensemble classifier.

    Warning: This class should not be used directly. Use the derive classes
    instead.
    """

    _ensemble_type = 'bagging'
    _solution_type = 'resampling'
    _training_type = 'parallel'

    _properties = {
        'ensemble_type': _ensemble_type,
        'solution_type': _solution_type,
        'training_type': _training_type,
    }

    @_deprecate_positional_args
    def __init__(
        self,
        estimator=None,
        n_estimators=10,
        *,
        sampler,
        sampling_type,
        sampling_strategy="auto",
        max_samples=1.0,
        max_features=1.0,
        bootstrap=True,
        bootstrap_features=False,
        oob_score=False,
        warm_start=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
    ):

        self.sampling_strategy = sampling_strategy
        self._sampling_type = sampling_type
        self.sampler = sampler

        super().__init__(
            estimator=estimator,
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=max_features,
            bootstrap=bootstrap,
            bootstrap_features=bootstrap_features,
            oob_score=oob_score,
            warm_start=warm_start,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
        )

    def _validate_y(self, y):
        """Validate the label vector."""
        y_encoded = super()._validate_y(y)
        if (
            isinstance(self.sampling_strategy, dict)
            and self.sampler_._sampling_type != "bypass"
        ):
            self._sampling_strategy = {
                np.where(self.classes_ == key)[0][0]: value
                for key, value in check_sampling_strategy(
                    self.sampling_strategy,
                    y,
                    self.sampler_._sampling_type,
                ).items()
            }
        else:
            self._sampling_strategy = self.sampling_strategy
        return y_encoded

    def _validate_estimator(self, default=DecisionTreeClassifier()):
        """Check the estimator and the n_estimator attribute, set the
        `estimator_` attribute."""
        if not isinstance(self.n_estimators, (numbers.Integral, np.integer)):
            raise ValueError(
                f"n_estimators must be an integer, " f"got {type(self.n_estimators)}."
            )

        if self.n_estimators <= 0:
            raise ValueError(
                f"n_estimators must be greater than zero, " f"got {self.n_estimators}."
            )

        if self.estimator is not None:
            estimator = clone(self.estimator)
        else:
            estimator = clone(default)

        # validate sampler and sampler_kwargs
        # validated sampler stored in self.sampler_
        try:
            self.sampler_ = clone(self.sampler)
        except Exception as e:
            e_args = list(e.args)
            e_args[0] = (
                "Exception occurs when trying to validate" + " sampler: " + e_args[0]
            )
            e.args = tuple(e_args)
            raise e

        if self.sampler_._sampling_type != "bypass":
            self.sampler_.set_params(sampling_strategy=self._sampling_strategy)
            self.sampler_.set_params(**self.sampler_kwargs_)

        self._estimator = Pipeline(
            [("sampler", self.sampler_), ("classifier", estimator)]
        )
        try:
            # scikit-learn < 1.2
            self.estimator_ = self._estimator
        except AttributeError:
            pass

    def _more_tags(self):  # pragma: no cover
        tags = super()._more_tags()
        tags_key = "_xfail_checks"
        failing_test = "check_estimators_nan_inf"
        reason = "Fails because the sampler removed infinity and NaN values"
        if tags_key in tags:
            tags[tags_key][failing_test] = reason
        else:
            tags[tags_key] = {failing_test: reason}
        return tags

    @_deprecate_positional_args
    @FuncSubstitution(
        eval_datasets=_get_parameter_docstring('eval_datasets'),
        eval_metrics=_get_parameter_docstring('eval_metrics'),
        train_verbose=_get_parameter_docstring('train_verbose', **_properties),
    )
    def _fit(
        self,
        X,
        y,
        *,
        sample_weight=None,
        sampler_kwargs: dict = {},
        max_samples=None,
        eval_datasets: dict = None,
        eval_metrics: dict = None,
        train_verbose: bool or int or dict,
    ):
        """Build a Bagging ensemble of estimators from the training set (X, y).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.

        y : array-like of shape (n_samples,)
            The target values (class labels in classification, real numbers in
            regression).

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted.
            Note that this is supported only if the base estimator supports
            sample weighting.

        sampler_kwargs : dict, default={}
            The kwargs to use as additional parameters when instantiating a
            new sampler. If none are given, default parameters are used.

        max_samples : int or float, default=None
            Argument to use instead of self.max_samples.

        %(eval_datasets)s

        %(eval_metrics)s

        %(train_verbose)s

        Returns
        -------
        self : object
        """

        # Check data, sampler_kwargs and random_state
        check_target_type(y)

        self.sampler_kwargs_ = check_type(sampler_kwargs, 'sampler_kwargs', dict)

        random_state = check_random_state(self.random_state)

        # Convert data (X is required to be 2d and indexable)
        check_x_y_args = {
            'accept_sparse': ['csr', 'csc'],
            'dtype': None,
            'force_all_finite': False,
            'multi_output': True,
        }
        X, y = self._validate_data(X, y, **check_x_y_args)

        # Check evaluation data
        self.eval_datasets_ = check_eval_datasets(eval_datasets, X, y, **check_x_y_args)

        # Check evaluation metrics
        self.eval_metrics_ = check_eval_metrics(eval_metrics)

        # Check verbose
        self.train_verbose_ = check_train_verbose(
            train_verbose, self.n_estimators, **self._properties
        )
        self._init_training_log_format()

        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X, dtype=None)

        # Remap output
        n_samples, self.n_features_in_ = X.shape
        self._n_samples = n_samples
        y = self._validate_y(y)

        # Check parameters
        self._validate_estimator()

        # Validate max_samples
        if max_samples is None:
            max_samples = self.max_samples
        if not isinstance(max_samples, numbers.Integral):
            max_samples = int(max_samples * X.shape[0])

        if not (0 < max_samples <= X.shape[0]):
            raise ValueError("max_samples must be in (0, n_samples]")

        # Store validated integer row sampling value
        self._max_samples = max_samples

        # Validate max_features
        if isinstance(self.max_features, numbers.Integral):
            max_features = self.max_features
        elif isinstance(self.max_features, float):
            max_features = self.max_features * self.n_features_in_
        else:
            raise ValueError("max_features must be int or float")

        if not (0 < max_features <= self.n_features_in_):
            raise ValueError("max_features must be in (0, n_features]")

        max_features = max(1, int(max_features))

        # Store validated integer feature sampling value
        self._max_features = max_features

        # Other checks
        if not self.bootstrap and self.oob_score:
            raise ValueError(
                "Out of bag estimation only available" " if bootstrap=True"
            )

        if self.warm_start and self.oob_score:
            raise ValueError(
                "Out of bag estimate only available" " if warm_start=False"
            )

        if hasattr(self, "oob_score_") and self.warm_start:
            del self.oob_score_

        if not self.warm_start or not hasattr(self, 'estimators_'):
            # Free allocated memory, if any
            self.estimators_ = []
            self.estimators_features_ = []
            self.estimators_n_training_samples_ = []

        n_more_estimators = self.n_estimators - len(self.estimators_)

        if n_more_estimators < 0:
            raise ValueError(
                'n_estimators=%d must be larger or equal to '
                'len(estimators_)=%d when warm_start==True'
                % (self.n_estimators, len(self.estimators_))
            )

        elif n_more_estimators == 0:
            warn(
                "Warm-start fitting without increasing n_estimators does not "
                "fit new trees."
            )
            return self

        # Parallel loop
        n_jobs, n_estimators, starts = _partition_estimators(
            n_more_estimators, self.n_jobs
        )
        total_n_estimators = sum(n_estimators)

        # Advance random state to state after training
        # the first n_estimators
        if self.warm_start and len(self.estimators_) > 0:
            random_state.randint(MAX_INT, size=len(self.estimators_))

        seeds = random_state.randint(MAX_INT, size=n_more_estimators)
        self._seeds = seeds

        all_results = Parallel(
            n_jobs=n_jobs, verbose=self.verbose, **self._parallel_args()
        )(
            delayed(_parallel_build_estimators)(
                n_estimators[i],
                self,
                X,
                y,
                sample_weight,
                seeds[starts[i] : starts[i + 1]],
                total_n_estimators,
                verbose=self.verbose,
            )
            for i in range(n_jobs)
        )

        # Reduce
        self.estimators_ += list(
            itertools.chain.from_iterable(t[0] for t in all_results)
        )
        self.estimators_features_ += list(
            itertools.chain.from_iterable(t[1] for t in all_results)
        )
        self.estimators_n_training_samples_ += list(
            itertools.chain.from_iterable(t[2] for t in all_results)
        )

        if self.oob_score:
            self._set_oob_score(X, y)

        # Print training infomation to console.
        self._training_log_to_console()

        return self

    @abstractmethod
    def fit(self, X, y, sample_weight, **kwargs):
        """Needs to be implemented in the derived class"""
        pass

    @FuncGlossarySubstitution(_super.predict_proba, 'classes_')
    def predict_proba(self, X):
        return super().predict_proba(X)

    @FuncGlossarySubstitution(_super.predict_log_proba, 'classes_')
    def predict_log_proba(self, X):
        return super().predict_log_proba(X)

    def set_params(self, **params):
        return super().set_params(**params)
