"""SMOTEBaggingClassifier: A Bagging classifier with 
intergrated SMOTE over-sampling.
"""

# Authors: Zhining Liu <zhining.liu@outlook.com>
# License: MIT

# %%
LOCAL_DEBUG = False

if not LOCAL_DEBUG:
    from ...sampler._over_sampling import SMOTE
    from ...utils._docstring import (
        FuncSubstitution,
        Substitution,
        _get_example_docstring,
        _get_parameter_docstring,
    )
    from ...utils._validation import _deprecate_positional_args
    from ...utils._validation_param import check_type
    from .._bagging import ResampleBaggingClassifier
else:  # pragma: no cover
    import sys  # For local test

    sys.path.append("../..")
    from ensemble._bagging import ResampleBaggingClassifier
    from sampler._over_sampling import SMOTE
    from utils._validation_param import check_type
    from utils._validation import _deprecate_positional_args
    from utils._docstring import (
        Substitution,
        FuncSubstitution,
        _get_parameter_docstring,
        _get_example_docstring,
    )

import numbers

# Properties
_method_name = 'SMOTEBaggingClassifier'
_sampler_class = SMOTE

_solution_type = ResampleBaggingClassifier._solution_type
_sampling_type = 'over-sampling'
_ensemble_type = ResampleBaggingClassifier._ensemble_type
_training_type = ResampleBaggingClassifier._training_type

_properties = {
    'solution_type': _solution_type,
    'sampling_type': _sampling_type,
    'ensemble_type': _ensemble_type,
    'training_type': _training_type,
}


@Substitution(
    random_state=_get_parameter_docstring('random_state', **_properties),
    n_jobs=_get_parameter_docstring('n_jobs', **_properties),
    warm_start=_get_parameter_docstring('warm_start', **_properties),
    example=_get_example_docstring(_method_name),
)
class SMOTEBaggingClassifier(ResampleBaggingClassifier):
    """A Bagging classifier with intergrated SMOTE over-sampling.

    This implementation of SMOTEBagging [1]_ is similar to the scikit-learn
    implementation. It includes an additional step to balance the training set
    at fit time using a SMOTE over-sampler.

    This implementation extends SMOTEBagging to support multi-class classification.

    Parameters
    ----------
    estimator : object, default=None
        The base estimator to fit on oversampled dataset.
        If None, then the base estimator is a
        ``Pipeline([SMOTE(), DecisionTreeClassifier()])``.

    n_estimators : int, default=50
        The number of base estimators in the ensemble.

    k_neighbors : int or object, default=1
        If ``int``, number of nearest neighbours to used to construct synthetic
        samples in SMOTE.  If object, an estimator that inherits from
        :class:`~sklearn.neighbors.base.KNeighborsMixin` that will be used to
        find the k_neighbors.

    max_samples : int or float, default=1.0
        The number of samples to draw from X to train each base estimator (with
        replacement by default, see `bootstrap` for more details).

        - If ``int``, then draw `max_samples` samples.
        - If ``float``, then draw `max_samples * X.shape[0]` samples.

    max_features : int or float, default=1.0
        The number of features to draw from X to train each base estimator (
        without replacement by default, see `bootstrap_features` for more
        details).

        - If ``int``, then draw `max_features` features.
        - If ``float``, then draw `max_features * X.shape[1]` features.

    bootstrap : bool, default=True
        Whether samples are drawn with replacement. If False, sampling
        without replacement is performed.

    bootstrap_features : bool, default=False
        Whether features are drawn with replacement.

    oob_score : bool, default=False
        Whether to use out-of-bag samples to estimate
        the generalization error.

    {warm_start}

    {n_jobs}

    {random_state}

    verbose : int, default=0
        Controls the verbosity when fitting and predicting.

    Attributes
    ----------
    estimator_ : pipeline estimator
        The base estimator from which the ensemble is grown.

    sampler_ : SMOTE
        The base sampler.

    estimators_ : list of classifiers
        The collection of fitted sub-estimators.

    n_features_in_ : int
        The number of features when :meth:`fit` is performed.

    estimators_ : list of estimators
        The collection of fitted base estimators.

    estimators_samples_ : list of arrays
        The subset of drawn samples (i.e., the in-bag samples) for each base
        estimator. Each subset is defined by an array of the indices selected.

    estimators_features_ : list of arrays
        The subset of drawn features for each base estimator.

    estimators_n_training_samples_ : list of ints
        The number of training samples for each fitted
        base estimators.

    classes_ : ndarray of shape (n_classes,)
        The classes labels.

    n_classes_ : int or list
        The number of classes.

    oob_score_ : float
        Score of the training dataset obtained using an out-of-bag estimate.
        This attribute exists only when ``oob_score`` is True.

    oob_decision_function_ : ndarray of shape (n_samples, n_classes)
        Decision function computed with out-of-bag estimate on the training
        set. If n_estimators is small it might be possible that a data point
        was never left out during the bootstrap. In this case,
        `oob_decision_function_` might contain NaN. This attribute exists
        only when ``oob_score`` is True.

    See Also
    --------
    OverBaggingClassifier : Bagging with intergrated random over-sampling.

    SMOTEBoostClassifier : SMOTE over-sampling integrated in AdaBoost.

    UnderBaggingClassifier : Bagging with intergrated random under-sampling.

    References
    ----------
    .. [1] S. Wang, and X. Yao. "Diversity analysis on imbalanced data sets by
           using ensemble models." 2009 IEEE symposium on computational
           intelligence and data mining. IEEE, 2009.

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
        k_neighbors: int = 1,
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

        sampling_strategy = 'auto'
        sampler = _sampler_class()
        sampling_type = _sampling_type

        super().__init__(
            estimator=estimator,
            n_estimators=n_estimators,
            sampler=sampler,
            sampling_type=sampling_type,
            sampling_strategy=sampling_strategy,
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

        self.__name__ = _method_name
        self._sampling_type = _sampling_type
        self._sampler_class = _sampler_class
        self._properties = _properties

        self.k_neighbors = k_neighbors
        self.k_neighbors_ = check_type(k_neighbors, 'k_neighbors', numbers.Integral)
        if self.k_neighbors_ < 1:
            raise ValueError(
                f"The 'k_neighbors' parameter of NearestNeighbors must be"
                f" an int in the range [1, inf) or None. Got {self.k_neighbors_} instead."
            )

    @_deprecate_positional_args
    @FuncSubstitution(
        eval_datasets=_get_parameter_docstring('eval_datasets'),
        eval_metrics=_get_parameter_docstring('eval_metrics'),
        train_verbose=_get_parameter_docstring('train_verbose', **_properties),
    )
    def fit(
        self,
        X,
        y,
        *,
        sample_weight=None,
        max_samples=None,
        eval_datasets: dict = None,
        eval_metrics: dict = None,
        train_verbose: bool or int or dict = False,
    ):
        """Build a SMOTEBagging ensemble of estimators from the training set (X, y).

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

        max_samples : int or float, default=None
            Argument to use instead of self.max_samples.

        %(eval_datasets)s

        %(eval_metrics)s

        %(train_verbose)s

        Returns
        -------
        self : object
        """

        smote_sampler_kwargs = {'k_neighbors': self.k_neighbors_}

        return self._fit(
            X,
            y,
            sample_weight=sample_weight,
            sampler_kwargs=smote_sampler_kwargs,
            max_samples=max_samples,
            eval_datasets=eval_datasets,
            eval_metrics=eval_metrics,
            train_verbose=train_verbose,
        )


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
        'k_neighbors': 5,
        'max_samples': 1.0,
        'max_features': 1.0,
        'bootstrap': True,
        'bootstrap_features': False,
        'oob_score': False,
        'warm_start': False,
        'n_jobs': None,
        'random_state': 42,
        # 'random_state': None,
        'verbose': 0,
    }
    fit_kwargs_default = {
        'X': X_train,
        'y': y_train,
        'sample_weight': None,
        'max_samples': None,
        'eval_datasets': {'valid': (X_valid, y_valid)},
        'eval_metrics': {
            'acc': (accuracy_score, {}),
            'balanced_acc': (balanced_accuracy_score, {}),
            'weighted_f1': (f1_score, {'average': 'weighted'}),
        },
        'train_verbose': True,
    }

    ensembles = {}

    init_kwargs, fit_kwargs = copy(init_kwargs_default), copy(fit_kwargs_default)
    smotebagging = SMOTEBaggingClassifier(**init_kwargs).fit(**fit_kwargs)
    ensembles['smotebagging'] = smotebagging

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
