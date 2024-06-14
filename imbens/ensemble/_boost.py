"""Base classes for all boosting-like methods in imbens.

ResampleBoostClassifier: Base class for all resampling + boosting 
imbalanced ensemble classifier

ReweightBoostClassifier: Base class for all reweighting + boosting 
imbalanced ensemble classifier.
"""

# Authors: Zhining Liu <zhining.liu@outlook.com>
# License: MIT

# %%
LOCAL_DEBUG = False

if not LOCAL_DEBUG:
    from ..utils._docstring import FuncGlossarySubstitution
    from ..utils._validation import _deprecate_positional_args
    from ..utils._validation_data import check_eval_datasets
    from ..utils._validation_param import (
        check_balancing_schedule,
        check_eval_metrics,
        check_target_label_and_n_target_samples,
        check_train_verbose,
        check_type,
    )
    from .base import MAX_INT, ImbalancedEnsembleClassifierMixin
else:  # pragma: no cover
    import sys  # For local test

    sys.path.append("..")
    from ensemble.base import ImbalancedEnsembleClassifierMixin, MAX_INT
    from utils._docstring import FuncGlossarySubstitution
    from utils._validation import _deprecate_positional_args
    from utils._validation_data import check_eval_datasets
    from utils._validation_param import (
        check_target_label_and_n_target_samples,
        check_balancing_schedule,
        check_train_verbose,
        check_eval_metrics,
        check_type,
    )

from abc import ABCMeta, abstractmethod
from collections import Counter
from copy import copy

import numpy as np
from scipy.special import xlogy
from sklearn.base import clone
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble._base import _set_random_states
from sklearn.ensemble._forest import BaseForest
from sklearn.tree import BaseDecisionTree
from sklearn.utils import check_array, check_random_state
from sklearn.utils.validation import _check_sample_weight

_super = AdaBoostClassifier


class ResampleBoostClassifier(
    ImbalancedEnsembleClassifierMixin, AdaBoostClassifier, metaclass=ABCMeta
):
    """Base class for all resampling + boosting imbalanced ensemble classifier.

    Warning: This class should not be used directly. Use the derive classes
    instead.
    """

    _ensemble_type = "boosting"
    _solution_type = "resampling"
    _training_type = "iterative"

    _properties = {
        "solution_type": _solution_type,
        "ensemble_type": _ensemble_type,
        "training_type": _training_type,
    }

    def __init__(
        self,
        estimator,
        n_estimators: int,
        sampler,
        sampling_type: str,
        learning_rate: float = 1.0,
        algorithm: str = "SAMME",
        early_termination: bool = False,
        random_state=None,
    ):

        self._sampling_type = sampling_type
        self.sampler = sampler
        self.early_termination = early_termination

        super(ResampleBoostClassifier, self).__init__(
            estimator=estimator,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            algorithm=algorithm,
            random_state=random_state,
        )

    def _validate_estimator(self):
        """Check the estimator, sampler and the n_estimator attribute.

        Sets the estimator_` and sampler_` attributes.
        """

        # validate estimator using
        # sklearn.ensemble.AdaBoostClassifier._validate_estimator
        super()._validate_estimator()

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

    def _make_sampler(self, append=True, random_state=None, **overwrite_kwargs):
        """Make and configure a copy of the `sampler_` attribute.

        Warning: This method should be used to properly instantiate new
        sub-samplers.
        """

        sampler = clone(self.sampler_)
        sampler.set_params(**self.sampler_kwargs_)

        # Arguments passed to _make_sampler function have higher priority,
        # they will overwrite the self.sampler_kwargs_
        sampler.set_params(**overwrite_kwargs)

        if random_state is not None:
            _set_random_states(sampler, random_state)

        if append:
            self.samplers_.append(sampler)

        return sampler

    def _boost(
        self,
        iboost,
        X_resampled,
        y_resampled,
        sample_weight_resampled,
        X,
        y,
        sample_weight,
        random_state,
    ):
        """Implement a single boost.

        Perform a single boost according to the real multi-class SAMME.R
        algorithm or to the discrete SAMME algorithm and return the updated
        sample weights.

        Parameters
        ----------
        iboost : int
            The index of the current boost iteration.

        X_resampled : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples of the resampled data.

        y_resampled : array-like of shape (n_samples,)
            The target values (class labels) of the resampled data.

        sample_weight_resampled : array-like of shape (n_samples,)
            The current sample weights of the resampled data.

        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples.

        y : array-like of shape (n_samples,)
            The target values (class labels).

        sample_weight : array-like of shape (n_samples,)
            The current sample weights.

        random_state : RandomState instance
            The RandomState instance used if the base estimator accepts a
            `random_state` attribute.

        Returns
        -------
        sample_weight : array-like of shape (n_samples,) or None
            The reweighted sample weights.
            If None then boosting has terminated early.

        estimator_weight : float
            The weight for the current boost.
            If None then boosting has terminated early.

        estimator_error : float
            The classification error for the current boost.
            If None then boosting has terminated early.
        """
        if self.algorithm == "SAMME.R":
            return self._boost_real(
                iboost,
                X_resampled,
                y_resampled,
                sample_weight_resampled,
                X,
                y,
                sample_weight,
                random_state,
            )

        else:  # elif self.algorithm == "SAMME":
            return self._boost_discrete(
                iboost,
                X_resampled,
                y_resampled,
                sample_weight_resampled,
                X,
                y,
                sample_weight,
                random_state,
            )

    def _boost_real(
        self,
        iboost,
        X_resampled,
        y_resampled,
        sample_weight_resampled,
        X,
        y,
        sample_weight,
        random_state,
    ):
        """Implement a single boost using the SAMME.R real algorithm."""
        estimator = self._make_estimator(random_state=random_state)

        estimator.fit(X_resampled, y_resampled, sample_weight=sample_weight_resampled)

        y_predict_proba = estimator.predict_proba(X)

        y_predict = self.classes_.take(np.argmax(y_predict_proba, axis=1), axis=0)

        # Instances incorrectly classified
        incorrect = y_predict != y

        # Error fraction
        estimator_error = np.mean(np.average(incorrect, weights=sample_weight, axis=0))

        # Stop if classification is perfect
        if estimator_error <= 0:
            return sample_weight, 1.0, 0.0

        # Construct y coding as described in Zhu et al [2]:
        #
        #    y_k = 1 if c == k else -1 / (K - 1)
        #
        # where K == n_classes_ and c, k in [0, K) are indices along the second
        # axis of the y coding with c being the index corresponding to the true
        # class label.
        n_classes = self.n_classes_
        classes = self.classes_
        y_codes = np.array([-1.0 / (n_classes - 1), 1.0])
        y_coding = y_codes.take(classes == y[:, np.newaxis])

        # Displace zero probabilities so the log is defined.
        # Also fix negative elements which may occur with
        # negative sample weights.
        proba = y_predict_proba  # alias for readability
        np.clip(proba, np.finfo(proba.dtype).eps, None, out=proba)

        # Boost weight using multi-class AdaBoost SAMME.R alg
        estimator_weight = (
            -1.0
            * self.learning_rate
            * ((n_classes - 1.0) / n_classes)
            * xlogy(y_coding, y_predict_proba).sum(axis=1)
        )

        # Only boost the weights if it will fit again
        if not iboost == self.n_estimators - 1:
            # Only boost positive weights
            sample_weight *= np.exp(
                estimator_weight * ((sample_weight > 0) | (estimator_weight < 0))
            )

        return sample_weight, 1.0, estimator_error

    def _boost_discrete(
        self,
        iboost,
        X_resampled,
        y_resampled,
        sample_weight_resampled,
        X,
        y,
        sample_weight,
        random_state,
    ):
        """Implement a single boost using the SAMME discrete algorithm."""
        estimator = self._make_estimator(random_state=random_state)

        estimator.fit(X_resampled, y_resampled, sample_weight=sample_weight_resampled)

        y_predict = estimator.predict(X)

        # Instances incorrectly classified
        incorrect = y_predict != y

        # Error fraction
        estimator_error = np.mean(np.average(incorrect, weights=sample_weight, axis=0))

        # Stop if classification is perfect
        if estimator_error <= 0:
            return sample_weight, 1.0, 0.0

        n_classes = self.n_classes_

        # # Stop if the error is at least as bad as random guessing
        # if estimator_error >= 1. - (1. / n_classes):
        #     self.estimators_.pop(-1)
        #     self.samplers_.pop(-1)
        #     if len(self.estimators_) == 0:
        #         raise ValueError(
        #             'BaseClassifier in AdaBoostClassifier '
        #             'ensemble is worse than random, ensemble '
        #             'can not be fit.'
        #         )
        #     return None, None, None

        # Boost weight using multi-class AdaBoost SAMME alg
        estimator_weight = self.learning_rate * (
            np.log((1.0 - estimator_error) / estimator_error) + np.log(n_classes - 1.0)
        )

        # Only boost the weights if I will fit again
        if not iboost == self.n_estimators - 1:
            # Only boost positive weights
            sample_weight *= np.exp(estimator_weight * incorrect * (sample_weight > 0))

        return sample_weight, estimator_weight, estimator_error

    @_deprecate_positional_args
    def _fit(
        self,
        X,
        y,
        *,
        sample_weight,
        sampler_kwargs: dict,
        target_label: int,
        n_target_samples: int or dict,
        balancing_schedule: str or function = "uniform",
        update_x_y_after_resample: bool = False,
        eval_datasets: dict,
        eval_metrics: dict,
        train_verbose: bool or int or dict,
    ):

        update_x_y_after_resample = check_type(
            update_x_y_after_resample, "update_x_y_after_resample", bool
        )

        self.sampler_kwargs_ = check_type(sampler_kwargs, "sampler_kwargs", dict)

        early_termination_ = check_type(
            self.early_termination, "early_termination", bool
        )

        # Check that algorithm is supported.
        if self.algorithm not in ("SAMME", "SAMME.R"):
            raise ValueError("algorithm %s is not supported" % self.algorithm)

        # Check parameters.
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero")

        if self.estimator == None or isinstance(
            self.estimator, (BaseDecisionTree, BaseForest)
        ):
            DTYPE = np.float64  # from fast_dict.pxd
            dtype = DTYPE
            accept_sparse = "csc"
        else:
            dtype = None
            accept_sparse = ["csr", "csc"]

        check_x_y_args = {
            "accept_sparse": accept_sparse,
            "ensure_2d": True,
            "allow_nd": True,
            "dtype": dtype,
            "y_numeric": False,
        }
        X, y = self._validate_data(X, y, **check_x_y_args)

        # Check evaluation data
        self.eval_datasets_ = check_eval_datasets(eval_datasets, X, y, **check_x_y_args)

        self.classes_, self._y_encoded = np.unique(y, return_inverse=True)
        self.n_classes_ = len(self.classes_)

        # Store original class distribution
        self.origin_distr_ = dict(Counter(y))
        (
            self.target_label_,
            self.target_distr_,
        ) = check_target_label_and_n_target_samples(
            y, target_label, n_target_samples, self._sampling_type
        )

        self.balancing_schedule_ = check_balancing_schedule(balancing_schedule)

        self.eval_metrics_ = check_eval_metrics(eval_metrics)

        self.train_verbose_ = check_train_verbose(
            train_verbose, self.n_estimators, **self._properties
        )

        self._init_training_log_format()

        # Check sample weight
        sample_weight = _check_sample_weight(sample_weight, X, np.float64)
        sample_weight /= sample_weight.sum()
        if np.any(sample_weight < 0):
            raise ValueError("sample_weight cannot contain negative weights")

        self.raw_sample_weight_ = sample_weight

        sample_weight = copy(self.raw_sample_weight_)

        self._validate_estimator()

        # Check random state
        random_state = check_random_state(self.random_state)

        # Clear any previous fit results.
        self.estimators_ = []
        self.estimator_weights_ = np.zeros(self.n_estimators, dtype=np.float64)
        self.estimator_errors_ = np.ones(self.n_estimators, dtype=np.float64)
        self.estimators_n_training_samples_ = np.zeros(self.n_estimators, dtype=int)

        self.samplers_ = []

        # Genrate random seeds array
        seeds = random_state.randint(MAX_INT, size=self.n_estimators)
        self._seeds = seeds

        epsilon = np.finfo(sample_weight.dtype).eps
        zero_weight_mask = sample_weight == 0.0
        sampler_ = self.sampler_

        for iboost in range(self.n_estimators):

            current_iter_distr = self.balancing_schedule_(
                origin_distr=self.origin_distr_,
                target_distr=self.target_distr_,
                i_estimator=iboost,
                total_estimator=self.n_estimators,
            )

            sampler = self._make_sampler(
                append=True,
                random_state=seeds[iboost],
                sampling_strategy=current_iter_distr,
            )

            # Perform re-sampling
            X_resampled, y_resampled, sample_weight_resampled = sampler.fit_resample(
                X, y, sample_weight=sample_weight
            )

            # Update X, y, sample_weight if update_x_y_after_resample is True
            if update_x_y_after_resample:
                X, y = X_resampled, y_resampled
                sample_weight = sample_weight_resampled

            # Boosting step.
            sample_weight, estimator_weight, estimator_error = self._boost(
                iboost,
                X_resampled,
                y_resampled,
                sample_weight_resampled,
                X,
                y,
                sample_weight,
                random_state,
            )

            self.estimator_weights_[iboost] = estimator_weight
            self.estimator_errors_[iboost] = estimator_error
            self.estimators_n_training_samples_[iboost] = y_resampled.shape[0]

            # Print training infomation to console.
            self._training_log_to_console(iboost, y_resampled)

            # Early termination.
            if sample_weight is None and early_termination_:
                print(
                    f"Training early-stop at iteration"
                    f" {iboost+1}/{self.n_estimators}"
                    f" (sample_weight is None)."
                )
                break

            # Stop if error is zero.
            if estimator_error == 0 and early_termination_:
                print(
                    f"Training early-stop at iteration"
                    f" {iboost+1}/{self.n_estimators}"
                    f" (training error is 0)."
                )
                break

            sample_weight_sum = np.sum(sample_weight)

            # Stop if the sum of sample weights has become non-positive.
            if (
                sample_weight is not None
                and sample_weight_sum <= 0
                and early_termination_
            ):
                print(
                    f"Training early-stop at iteration"
                    f" {iboost+1}/{self.n_estimators}"
                    f" (sample_weight_sum <= 0)."
                )
                break

            if iboost < self.n_estimators - 1:
                # Normalize.
                sample_weight /= sample_weight_sum

        return self

    @abstractmethod
    def fit(self, X, y, *, sample_weight=None, **kwargs):
        """Needs to be implemented in the derived class"""
        pass

    @property
    def feature_importances_(self):
        """The impurity-based feature importances.
        The higher, the more important the feature.
        The importance of a feature is computed as the (normalized)
        total reduction of the criterion brought by that feature.  It is also
        known as the Gini importance.
        Warning: impurity-based feature importances can be misleading for
        high cardinality features (many unique values). See
        :func:`sklearn.inspection.permutation_importance` as an alternative.

        Returns
        -------
        feature_importances_ : ndarray of shape (n_features,)
            The feature importances.
        """
        if self.estimators_ is None or len(self.estimators_) == 0:
            raise ValueError(
                "Estimator not fitted, call `fit` before `feature_importances_`."
            )

        try:
            if hasattr(self, "estimator_weights_"):
                norm = self.estimator_weights_.sum()
                return (
                    sum(
                        weight * clf.feature_importances_
                        for weight, clf in zip(
                            self.estimator_weights_, self.estimators_
                        )
                    )
                    / norm
                )
            else:
                return sum(clf.feature_importances_ for clf in self.estimators_) / len(
                    self.estimators_
                )

        except AttributeError as e:
            raise AttributeError(
                "Unable to compute feature importances "
                "since estimator does not have a "
                "feature_importances_ attribute"
            ) from e

    @FuncGlossarySubstitution(_super.decision_function, "classes_")
    def decision_function(self, X):
        return super().decision_function(X)

    @FuncGlossarySubstitution(_super.predict_log_proba, "classes_")
    def predict_log_proba(self, X):
        return super().predict_log_proba(X)

    @FuncGlossarySubstitution(_super.predict_proba, "classes_")
    def predict_proba(self, X):
        return super().predict_proba(X)

    @FuncGlossarySubstitution(_super.staged_decision_function, "classes_")
    def staged_decision_function(self, X):
        return super().staged_decision_function(X)

    @FuncGlossarySubstitution(_super.staged_predict_proba, "classes_")
    def staged_predict_proba(self, X):
        return super().staged_predict_proba(X)


SET_COST_MATRIX_HOW = ("uniform", "inverse", "log1p-inverse")


class ReweightBoostClassifier(
    ImbalancedEnsembleClassifierMixin, AdaBoostClassifier, metaclass=ABCMeta
):
    """Base class for all reweighting + boosting imbalanced ensemble classifier.

    Warning: This class should not be used directly. Use the derive classes
    instead.
    """

    _ensemble_type = "boosting"
    _solution_type = "reweighting"
    _training_type = "iterative"

    _properties = {
        "solution_type": _solution_type,
        "ensemble_type": _ensemble_type,
        "training_type": _training_type,
    }

    def __init__(
        self,
        estimator,
        n_estimators: int,
        learning_rate: float = 1.0,
        algorithm: str = "SAMME",
        early_termination: bool = False,
        random_state=None,
    ):

        self.early_termination = early_termination

        super(ReweightBoostClassifier, self).__init__(
            estimator=estimator,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            algorithm=algorithm,
            random_state=random_state,
        )

    def _compute_mult_in_exp_weights_array(self, y_true, y_pred):
        """
        Compute the additional weights that need to be multiplied
        INSIDE of exp while boosting.
        Return an array of shape = (n_samples,).

        This function can be re-implemented (if needed) in the drived class.
        """
        return np.ones_like(y_true, dtype=np.float64)

    def _compute_mult_out_exp_weights_array(self, y_true, y_pred):
        """
        Compute the additional weights that need to be multiplied
        OUTSIDE of exp while boosting.
        Return an array of shape = (n_samples,).

        This function can be re-implemented (if needed) in the derived class.
        """
        return np.ones_like(y_true, dtype=np.float64)

    def _preprocess_sample_weight(self, sample_weight, y):
        """
        Preprocessing the sample_weight before start boost training.
        Return an array of shape = (n_samples,).

        This function can be re-implemented (if needed) in the derived class.
        """
        return sample_weight

    def _set_cost_matrix(self, how: str = "inverse"):
        """Set the cost matrix according to the 'how' parameter."""
        classes, origin_distr = self._encode_map.values(), self.origin_distr_
        cost_matrix = []
        for c_pred in classes:
            cost_c = [
                origin_distr[c_pred] / origin_distr[c_actual] for c_actual in classes
            ]
            cost_c[c_pred] = 1
            cost_matrix.append(cost_c)
        if how == "uniform":
            return np.ones_like(cost_matrix)
        elif how == "inverse":
            return cost_matrix
        elif how == "log1p-inverse":
            return np.log1p(cost_matrix)
        else:
            raise ValueError(
                f"When 'cost_matrix' is string, it should be"
                f" in {SET_COST_MATRIX_HOW}, got {how}."
            )

    @staticmethod
    def _validate_cost_matrix(cost_matrix, n_classes):
        """validate the cost matrix."""
        cost_matrix = check_array(
            cost_matrix, ensure_2d=True, allow_nd=False, force_all_finite=True
        )
        if cost_matrix.shape != (n_classes, n_classes):
            raise ValueError(
                "When 'cost_matrix' is array-like, it should"
                " be of shape = [n_classes, n_classes],"
                " got shape = {0}".format(cost_matrix.shape)
            )
        return cost_matrix

    def _boost_real(self, iboost, X, y, sample_weight, random_state):
        """Implement a single boost using the SAMME.R real algorithm."""
        estimator = self._make_estimator(random_state=random_state)

        estimator.fit(X, y, sample_weight=sample_weight)

        y_predict_proba = estimator.predict_proba(X)

        y_predict = self.classes_.take(np.argmax(y_predict_proba, axis=1), axis=0)
        y_predict = np.array(list(map(lambda x: self._encode_map[x], y_predict)))

        # Instances incorrectly classified
        incorrect = y_predict != y

        # Error fraction
        estimator_error = np.mean(np.average(incorrect, weights=sample_weight, axis=0))

        # Stop if classification is perfect
        if estimator_error <= 0:
            return sample_weight, 1.0, 0.0

        # Construct y coding as described in Zhu et al [2]:
        #
        #    y_k = 1 if c == k else -1 / (K - 1)
        #
        # where K == n_classes_ and c, k in [0, K) are indices along the second
        # axis of the y coding with c being the index corresponding to the true
        # class label.
        n_classes = self.n_classes_
        classes = self.classes_
        y_codes = np.array([-1.0 / (n_classes - 1), 1.0])
        y_coding = y_codes.take(classes == y[:, np.newaxis])

        # Displace zero probabilities so the log is defined.
        # Also fix negative elements which may occur with
        # negative sample weights.
        proba = y_predict_proba  # alias for readability
        np.clip(proba, np.finfo(proba.dtype).eps, None, out=proba)

        # Boost weight using multi-class AdaBoost SAMME.R alg
        estimator_weight = (
            -1.0
            * self.learning_rate
            * ((n_classes - 1.0) / n_classes)
            * xlogy(y_coding, y_predict_proba).sum(axis=1)
        )

        # Compute additional weights for multiplication
        mult_in_exp_weight = self._compute_mult_in_exp_weights_array(
            y_true=self._y_encoded, y_pred=y_predict
        )
        mult_in_exp_weight /= mult_in_exp_weight.max()
        mult_out_exp_weight = self._compute_mult_out_exp_weights_array(
            y_true=self._y_encoded, y_pred=y_predict
        )

        # Only boost the weights if it will fit again
        if not iboost == self.n_estimators - 1:
            # Only boost positive weights
            sample_weight *= (
                np.exp(
                    estimator_weight
                    * mult_in_exp_weight
                    * ((sample_weight > 0) | (estimator_weight < 0))
                )
                * mult_out_exp_weight
            )

        # print (f'y {np.unique(y)}')
        # print (f'_y_encoded {np.unique(self._y_encoded)}')
        # print (f'y_predict {np.unique(y_predict)}')
        # print (f'estimator_weight {estimator_weight.max()}')
        # print (f'mult_in_exp_weight {mult_in_exp_weight.max()}')
        # print (f'mult_out_exp_weight {mult_out_exp_weight.max()}')
        # print (f'sample_weight {sample_weight.max()}')

        return sample_weight, 1.0, estimator_error

    def _boost_discrete(self, iboost, X, y, sample_weight, random_state):
        """Implement a single boost using the SAMME discrete algorithm."""
        estimator = self._make_estimator(random_state=random_state)

        estimator.fit(X, y, sample_weight=sample_weight)

        y_predict = estimator.predict(X)

        # Instances incorrectly classified
        incorrect = y_predict != y

        y_predict = np.array(list(map(lambda x: self._encode_map[x], y_predict)))

        # Error fraction
        estimator_error = np.mean(np.average(incorrect, weights=sample_weight, axis=0))

        # Stop if classification is perfect
        if estimator_error <= 0:
            return sample_weight, 1.0, 0.0

        n_classes = self.n_classes_

        # Stop if the error is at least as bad as random guessing
        # if estimator_error >= 1. - (1. / n_classes):
        #     self.estimators_.pop(-1)
        #     if len(self.estimators_) == 0:
        #         raise ValueError('BaseClassifier in AdaBoostClassifier '
        #                          'ensemble is worse than random, ensemble '
        #                          'can not be fit.')
        #     return None, None, None

        # Boost weight using multi-class AdaBoost SAMME alg
        estimator_weight = self.learning_rate * (
            np.log((1.0 - estimator_error) / estimator_error) + np.log(n_classes - 1.0)
        )

        # Compute additional weights for multiplication
        mult_in_exp_weight = self._compute_mult_in_exp_weights_array(
            y_true=self._y_encoded, y_pred=y_predict
        )
        mult_out_exp_weight = self._compute_mult_out_exp_weights_array(
            y_true=self._y_encoded, y_pred=y_predict
        )

        # Only boost the weights if I will fit again
        if not iboost == self.n_estimators - 1:
            # Only boost positive weights
            sample_weight *= (
                np.exp(
                    estimator_weight
                    * incorrect
                    * mult_in_exp_weight
                    * (sample_weight > 0)
                )
                * mult_out_exp_weight
            )

        return sample_weight, estimator_weight, estimator_error

    @_deprecate_positional_args
    def _fit(
        self,
        X,
        y,
        *,
        sample_weight,
        cost_matrix,
        eval_datasets: dict,
        eval_metrics: dict,
        train_verbose: bool or int or dict,
    ):

        early_termination_ = check_type(
            self.early_termination, "early_termination", bool
        )

        # Check that algorithm is supported.
        if self.algorithm not in ("SAMME", "SAMME.R"):
            raise ValueError("algorithm %s is not supported" % self.algorithm)

        # Check parameters.
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero")

        if self.estimator == None or isinstance(
            self.estimator, (BaseDecisionTree, BaseForest)
        ):
            DTYPE = np.float64  # from fast_dict.pxd
            dtype = DTYPE
            accept_sparse = "csc"
        else:
            dtype = None
            accept_sparse = ["csr", "csc"]

        check_x_y_args = {
            "accept_sparse": accept_sparse,
            "ensure_2d": True,
            "allow_nd": True,
            "dtype": dtype,
            "y_numeric": False,
        }
        X, y = self._validate_data(X, y, **check_x_y_args)

        # Check evaluation data
        self.eval_datasets_ = check_eval_datasets(eval_datasets, X, y, **check_x_y_args)

        raw_y = y.copy()
        self.classes_, self._y_encoded = np.unique(y, return_inverse=True)
        self._encode_map = {
            c: np.where(self.classes_ == c)[0][0] for c in self.classes_
        }
        self.n_classes_ = len(self.classes_)

        # Store original class distribution
        self.origin_distr_ = dict(Counter(self._y_encoded))
        self.target_distr_ = dict(Counter(self._y_encoded))

        self.eval_metrics_ = check_eval_metrics(eval_metrics)

        self.train_verbose_ = check_train_verbose(
            train_verbose, self.n_estimators, **self._properties
        )

        self._init_training_log_format()

        # Check sample weight
        sample_weight = _check_sample_weight(sample_weight, X, np.float64)
        sample_weight = self._preprocess_sample_weight(sample_weight, self._y_encoded)
        sample_weight /= sample_weight.sum()
        if np.any(sample_weight < 0):
            raise ValueError("sample_weight cannot contain negative weights")

        self.raw_sample_weight_ = sample_weight

        sample_weight = copy(self.raw_sample_weight_)

        # Initialize & validate cost matrix
        if cost_matrix is None:
            cost_matrix = self._set_cost_matrix()
        elif isinstance(cost_matrix, str):
            cost_matrix = self._set_cost_matrix(how=cost_matrix)
        cost_matrix = self._validate_cost_matrix(cost_matrix, self.n_classes_)
        self.cost_matrix_ = cost_matrix

        self._validate_estimator()

        # Check random state
        random_state = check_random_state(self.random_state)

        # Clear any previous fit results.
        self.estimators_ = []
        self.estimator_weights_ = np.zeros(self.n_estimators, dtype=np.float64)
        self.estimator_errors_ = np.ones(self.n_estimators, dtype=np.float64)
        self.estimators_n_training_samples_ = np.zeros(self.n_estimators, dtype=int)

        # Genrate random seeds array
        seeds = random_state.randint(MAX_INT, size=self.n_estimators)
        self._seeds = seeds

        for iboost in range(self.n_estimators):
            # Boosting step
            sample_weight, estimator_weight, estimator_error = self._boost(
                iboost, X, raw_y, sample_weight, random_state
            )

            self.estimator_weights_[iboost] = estimator_weight
            self.estimator_errors_[iboost] = estimator_error
            self.estimators_n_training_samples_[iboost] = y.shape[0]

            # Print training infomation to console.
            self._training_log_to_console(iboost, y)

            # Early termination.
            if sample_weight is None and early_termination_:
                print(
                    f"Training early-stop at iteration"
                    f" {iboost+1}/{self.n_estimators}"
                    f" (sample_weight is None)."
                )
                break

            # Stop if error is zero.
            if estimator_error == 0 and early_termination_:
                print(
                    f"Training early-stop at iteration"
                    f" {iboost+1}/{self.n_estimators}"
                    f" (training error is 0)."
                )
                break

            sample_weight_sum = np.sum(sample_weight)

            # Stop if the sum of sample weights has become non-positive.
            if sample_weight_sum <= 0 and early_termination_:
                print(
                    f"Training early-stop at iteration"
                    f" {iboost+1}/{self.n_estimators}"
                    f" (sample_weight_sum <= 0)."
                )
                break

            if iboost < self.n_estimators - 1:
                # Normalize.
                sample_weight /= sample_weight_sum

        return self

    @abstractmethod
    def fit(self, X, y, *, sample_weight=None, **kwargs):
        """Needs to be implemented in the derived class"""
        pass

    @property
    def feature_importances_(self):
        """The impurity-based feature importances.
        The higher, the more important the feature.
        The importance of a feature is computed as the (normalized)
        total reduction of the criterion brought by that feature.  It is also
        known as the Gini importance.
        Warning: impurity-based feature importances can be misleading for
        high cardinality features (many unique values). See
        :func:`sklearn.inspection.permutation_importance` as an alternative.

        Returns
        -------
        feature_importances_ : ndarray of shape (n_features,)
            The feature importances.
        """
        if self.estimators_ is None or len(self.estimators_) == 0:
            raise ValueError(
                "Estimator not fitted, call `fit` before `feature_importances_`."
            )

        try:
            if hasattr(self, "estimator_weights_"):
                norm = self.estimator_weights_.sum()
                return (
                    sum(
                        weight * clf.feature_importances_
                        for weight, clf in zip(
                            self.estimator_weights_, self.estimators_
                        )
                    )
                    / norm
                )
            else:
                return sum(clf.feature_importances_ for clf in self.estimators_) / len(
                    self.estimators_
                )

        except AttributeError as e:
            raise AttributeError(
                "Unable to compute feature importances "
                "since estimator does not have a "
                "feature_importances_ attribute"
            ) from e

    @FuncGlossarySubstitution(_super.decision_function, "classes_")
    def decision_function(self, X):
        return super().decision_function(X)

    @FuncGlossarySubstitution(_super.predict_log_proba, "classes_")
    def predict_log_proba(self, X):
        return super().predict_log_proba(X)

    @FuncGlossarySubstitution(_super.predict_proba, "classes_")
    def predict_proba(self, X):
        return super().predict_proba(X)

    @FuncGlossarySubstitution(_super.staged_decision_function, "classes_")
    def staged_decision_function(self, X):
        return super().staged_decision_function(X)

    @FuncGlossarySubstitution(_super.staged_predict_proba, "classes_")
    def staged_predict_proba(self, X):
        return super().staged_predict_proba(X)
