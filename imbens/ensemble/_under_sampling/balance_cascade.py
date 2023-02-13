"""BalanceCascadeClassifier: A balance-cascade Classifier for 
class-imbalanced learning.
"""

# Authors: Zhining Liu <zhining.liu@outlook.com>
# License: MIT

# %%
LOCAL_DEBUG = False

if not LOCAL_DEBUG:
    from ...sampler._under_sampling import BalanceCascadeUnderSampler
    from ...utils._docstring import (
        FuncSubstitution,
        Substitution,
        _get_example_docstring,
        _get_parameter_docstring,
    )
    from ...utils._validation import _deprecate_positional_args
    from ...utils._validation_data import check_eval_datasets
    from ...utils._validation_param import (
        check_balancing_schedule,
        check_eval_metrics,
        check_target_label_and_n_target_samples,
        check_train_verbose,
    )
    from ..base import MAX_INT, BaseImbalancedEnsemble
else:  # pragma: no cover
    import sys  # For local test

    sys.path.append("../..")
    from ensemble.base import BaseImbalancedEnsemble, MAX_INT
    from sampler._under_sampling import BalanceCascadeUnderSampler
    from utils._validation_data import check_eval_datasets
    from utils._validation_param import (
        check_target_label_and_n_target_samples,
        check_balancing_schedule,
        check_train_verbose,
        check_eval_metrics,
    )
    from utils._validation import _deprecate_positional_args
    from utils._docstring import (
        Substitution,
        FuncSubstitution,
        _get_parameter_docstring,
        _get_example_docstring,
    )

from collections import Counter

import numpy as np

# Properties
_method_name = 'BalanceCascadeClassifier'
_sampler_class = BalanceCascadeUnderSampler

_solution_type = 'resampling'
_sampling_type = 'under-sampling'
_ensemble_type = 'general'
_training_type = 'iterative'

_properties = {
    'solution_type': _solution_type,
    'sampling_type': _sampling_type,
    'ensemble_type': _ensemble_type,
    'training_type': _training_type,
}


@Substitution(
    random_state=_get_parameter_docstring('random_state', **_properties),
    n_jobs=_get_parameter_docstring('n_jobs', **_properties),
    example=_get_example_docstring(_method_name),
)
class BalanceCascadeClassifier(BaseImbalancedEnsemble):
    """A balance-cascade Classifier for class-imbalanced learning.

    BalanceCascade [1]_ iteratively drops majority class samples
    that were already well-classified by the current ensemble.
    After that, it performs random under-sampling on the remaining
    majority class samples and train a new base estimator.

    This implementation extends BalanceCascade to support multi-class
    classification.

    Parameters
    ----------
    estimator : estimator object, default=None
        The base estimator to fit on self-paced under-sampled subsets
        of the dataset. Support for sample weighting is NOT required,
        but need proper ``classes_`` and ``n_classes_`` attributes.
        If ``None``, then the base estimator is ``DecisionTreeClassifier()``.

    n_estimators : int, default=50
        The number of base estimators in the ensemble.

    replacement : bool, default=True
        Whether samples are drawn with replacement. If ``False``
        and ``soft_resample_flag = False``, may raise an error when
        a bin has insufficient number of data samples for resampling.

    estimator_params : list of str, default=tuple()
        The list of attributes to use as parameters when instantiating a
        new base estimator. If none are given, default parameters are used.

    {n_jobs}

    {random_state}

    verbose : int, default=0
        Controls the verbosity when predicting.

    Attributes
    ----------
    estimator : estimator
        The base estimator from which the ensemble is grown.

    sampler_ : BalanceCascadeUnderSampler
        The base sampler.

    estimators_ : list of estimator
        The collection of fitted base estimators.

    samplers_ : list of BalanceCascadeUnderSampler
        The collection of fitted samplers.

    classes_ : ndarray of shape (n_classes,)
        The classes labels.

    n_classes_ : int
        The number of classes.

    feature_importances_ : ndarray of shape (n_features,)
        The feature importances if supported by the ``estimator``.

    estimators_n_training_samples_ : list of ints
        The number of training samples for each fitted
        base estimators.

    See Also
    --------
    SelfPacedEnsembleClassifier : Ensemble with self-paced dynamic under-sampling.

    EasyEnsembleClassifier : Bag of balanced boosted learners.

    RUSBoostClassifier : Random under-sampling integrated in AdaBoost.

    References
    ----------
    .. [1] Liu, X. Y., Wu, J., & Zhou, Z. H. "Exploratory undersampling for
       class-imbalance learning." IEEE Transactions on Systems, Man, and
       Cybernetics, Part B (Cybernetics) 39.2 (2008): 539-550.

    Examples
    --------
    {example}
    """

    @_deprecate_positional_args
    def __init__(
        self,
        estimator=None,
        n_estimators: int = 50,
        *,
        replacement: bool = True,
        estimator_params=tuple(),
        n_jobs=None,
        random_state=None,
        verbose=0,
    ):

        super(BalanceCascadeClassifier, self).__init__(
            estimator=estimator,
            n_estimators=n_estimators,
            estimator_params=estimator_params,
            random_state=random_state,
            n_jobs=n_jobs,
            verbose=verbose,
        )

        self.__name__ = _method_name
        self.sampler = _sampler_class()
        self._sampling_type = _sampling_type
        self._sampler_class = _sampler_class
        self._properties = _properties

        self.replacement = replacement

    @_deprecate_positional_args
    @FuncSubstitution(
        target_label=_get_parameter_docstring('target_label', **_properties),
        n_target_samples=_get_parameter_docstring('n_target_samples', **_properties),
        balancing_schedule=_get_parameter_docstring('balancing_schedule'),
        eval_datasets=_get_parameter_docstring('eval_datasets'),
        eval_metrics=_get_parameter_docstring('eval_metrics'),
        train_verbose=_get_parameter_docstring('train_verbose', **_properties),
    )
    def fit(self, X, y, *, sample_weight=None, **kwargs):
        """Build a BalanceCascade classifier from the training set (X, y).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. DOK and LIL are converted to CSR.

        y : array-like of shape (n_samples,)
            The target values (class labels).

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, the sample weights are initialized to
            ``1 / n_samples``.

        %(target_label)s

        %(n_target_samples)s

        %(balancing_schedule)s

        %(eval_datasets)s

        %(eval_metrics)s

        %(train_verbose)s

        Returns
        -------
        self : object
            Returns self.
        """
        return super().fit(X, y, sample_weight=sample_weight, **kwargs)

    @_deprecate_positional_args
    def _fit(
        self,
        X,
        y,
        *,
        sample_weight=None,
        target_label: int = None,
        n_target_samples: int or dict = None,
        balancing_schedule: str or function = 'uniform',
        eval_datasets: dict = None,
        eval_metrics: dict = None,
        train_verbose: bool or int or dict = False,
    ):

        # X, y, sample_weight, base_estimators_ (default=DecisionTreeClassifier),
        # n_estimators, random_state, sample_weight are already validated in super.fit()
        random_state, n_estimators, classes_, replacement = (
            self.random_state,
            self.n_estimators,
            self.classes_,
            self.replacement,
        )

        # Check evaluation data
        check_x_y_args = self.check_x_y_args
        self.eval_datasets_ = check_eval_datasets(eval_datasets, X, y, **check_x_y_args)

        # Check target sample strategy
        origin_distr_ = dict(Counter(y))
        target_label_, target_distr_ = check_target_label_and_n_target_samples(
            y, target_label, n_target_samples, self._sampling_type
        )
        self.origin_distr_, self.target_label_, self.target_distr_ = (
            origin_distr_,
            target_label_,
            target_distr_,
        )

        # Check balancing schedule
        balancing_schedule_ = check_balancing_schedule(balancing_schedule)
        self.balancing_schedule_ = balancing_schedule_

        # Check evaluation metrics
        self.eval_metrics_ = check_eval_metrics(eval_metrics)

        # Check training train_verbose format
        self.train_verbose_ = check_train_verbose(
            train_verbose, self.n_estimators, **self._properties
        )

        # Set training verbose format
        self._init_training_log_format()

        # Clear any previous fit results.
        self.estimators_ = []
        self.estimators_features_ = []
        self.estimators_n_training_samples_ = np.zeros(n_estimators, dtype=int)
        self.samplers_ = []
        self.sample_weights_ = []

        # Genrate random seeds array
        seeds = random_state.randint(MAX_INT, size=n_estimators)
        self._seeds = seeds

        # Initialize the keep_ratios and dropped_index
        keep_ratios = {
            label: np.power(
                (target_distr_[label] / origin_distr_[label]), 1 / (n_estimators)
            )
            for label in classes_
        }
        self.keep_ratios_ = keep_ratios
        dropped_index = np.full_like(y, fill_value=False, dtype=bool)

        # Check if sample_weight is specified
        specified_sample_weight = sample_weight is not None

        for i_iter in range(self.n_estimators):

            current_iter_distr = balancing_schedule_(
                origin_distr=origin_distr_,
                target_distr=target_distr_,
                i_estimator=i_iter,
                total_estimator=n_estimators,
            )

            if current_iter_distr != target_distr_:
                raise ValueError(
                    f"`BalanceCascadeClassifier` only support static target "
                    f"sample distribution, please set `balancing_schedule='uniform'` "
                    f"or pass your own callable `balancing_schedule` that returns a "
                    f"same target across different iterations to avoid this issue."
                )

            sampler = self._make_sampler(
                append=True,
                random_state=seeds[i_iter],
                sampling_strategy=current_iter_distr,
                replacement=replacement,
            )

            # compute keep_populations
            keep_populations = {
                label: int(
                    origin_distr_[label] * np.power(keep_ratios[label], i_iter + 1)
                    + 1e-5
                )
                for label in classes_
            }

            # update self.y_pred_proba_latest
            self._update_cached_prediction_probabilities(i_iter, X)

            # Perform self-paced under-sampling
            resample_out = sampler.fit_resample(
                X,
                y,
                y_pred_proba=self.y_pred_proba_latest,
                dropped_index=dropped_index,
                keep_populations=keep_populations,
                classes_=classes_,
                encode_map=self._encode_map,
                sample_weight=sample_weight,
            )

            # Train a new base estimator on resampled data
            # and add it into self.estimators_
            estimator = self._make_estimator(append=True, random_state=seeds[i_iter])
            if specified_sample_weight:
                (
                    X_resampled,
                    y_resampled,
                    sample_weight_resampled,
                    dropped_index,
                ) = resample_out
                estimator.fit(
                    X_resampled, y_resampled, sample_weight=sample_weight_resampled
                )
            else:
                (X_resampled, y_resampled, dropped_index) = resample_out
                estimator.fit(X_resampled, y_resampled)

            self.estimators_features_.append(self.features_)
            self.estimators_n_training_samples_[i_iter] = y_resampled.shape[0]

            # Print training infomation to console.
            self._training_log_to_console(i_iter, y_resampled)

        return self

    def _update_cached_prediction_probabilities(self, i_iter, X):
        """Private function that maintains a latest prediction probabilities of the training
        data during ensemble training. Must be called in each iteration before fit the
        estimator."""

        if i_iter == 0:
            self.y_pred_proba_latest = np.zeros(
                (self._n_samples, self.n_classes_), dtype=np.float64
            )
        else:
            y_pred_proba_latest = self.y_pred_proba_latest
            y_pred_proba_new = self.estimators_[-1].predict_proba(X)
            self.y_pred_proba_latest = (
                y_pred_proba_latest * i_iter + y_pred_proba_new
            ) / (i_iter + 1)
        return


# %%

if __name__ == "__main__":  # pragma: no cover
    from collections import Counter
    from copy import copy

    from sklearn.datasets import make_classification
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
    from sklearn.model_selection import train_test_split

    # X, y = make_classification(n_classes=2, class_sep=2, # 2-class
    #     weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
    #     n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
    X, y = make_classification(
        n_classes=3,
        class_sep=2,  # 3-class
        weights=[0.1, 0.3, 0.6],
        n_informative=3,
        n_redundant=1,
        flip_y=0,
        n_features=20,
        n_clusters_per_class=1,
        n_samples=2000,
        random_state=10,
    )

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.5, random_state=42
    )

    origin_distr = dict(Counter(y_train))  # {2: 600, 1: 300, 0: 100}
    print('Original training dataset shape %s' % origin_distr)

    target_distr = {2: 200, 1: 100, 0: 100}

    init_kwargs_default = {
        'estimator': None,
        'n_estimators': 100,
        'replacement': False,
        'estimator_params': tuple(),
        'n_jobs': None,
        'random_state': 10,
        # 'random_state': None,
        'verbose': 0,
    }

    fit_kwargs_default = {
        'X': X_train,
        'y': y_train,
        'sample_weight': None,
        'target_label': None,
        'n_target_samples': None,
        # 'n_target_samples': target_distr,
        'balancing_schedule': 'uniform',
        'eval_datasets': {'valid': (X_valid, y_valid)},
        'eval_metrics': {
            'acc': (accuracy_score, {}),
            'balanced_acc': (balanced_accuracy_score, {}),
            'weighted_f1': (f1_score, {'average': 'weighted'}),
        },
        'train_verbose': {
            'granularity': 10,
            'print_distribution': True,
            'print_metrics': True,
        },
    }

    ensembles = {}

    init_kwargs, fit_kwargs = copy(init_kwargs_default), copy(fit_kwargs_default)
    bcc = BalanceCascadeClassifier(**init_kwargs).fit(**fit_kwargs)
    ensembles['bcc'] = bcc

    # %%
    from imbens.visualizer import ImbalancedEnsembleVisualizer

    visualizer = ImbalancedEnsembleVisualizer(
        eval_datasets=None,
        eval_metrics=None,
    ).fit(
        ensembles=ensembles,
        granularity=5,
    )
    fig, axes = visualizer.performance_lineplot(
        on_ensembles=None,
        on_datasets=None,
        split_by=[],
        n_samples_as_x_axis=False,
        sub_figsize=(4, 3.3),
        sup_title=True,
        alpha=0.8,
    )
    fig, axes = visualizer.confusion_matrix_heatmap(
        on_ensembles=None,
        on_datasets=None,
        sub_figsize=(4, 3.3),
    )

    # %%
