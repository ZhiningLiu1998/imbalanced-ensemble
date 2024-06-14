"""KmeansSMOTEBoostClassifier: Kmeans-SMOTE over-sampling 
integrated in the learning of AdaBoost.
"""

# Authors: Zhining Liu <zhining.liu@outlook.com>
# License: MIT

# %%
LOCAL_DEBUG = False

if not LOCAL_DEBUG:
    from ...sampler._over_sampling import KMeansSMOTE
    from ...utils._docstring import (
        FuncSubstitution,
        Substitution,
        _get_example_docstring,
        _get_parameter_docstring,
    )
    from ...utils._validation import _deprecate_positional_args
    from ...utils._validation_param import check_type
    from .._boost import ResampleBoostClassifier
else:  # pragma: no cover
    import sys  # For local test

    sys.path.append("../..")
    from ensemble._boost import ResampleBoostClassifier
    from sampler._over_sampling import KMeansSMOTE
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
_method_name = "KmeansSMOTEBoostClassifier"
_sampler_class = KMeansSMOTE

_solution_type = ResampleBoostClassifier._solution_type
_sampling_type = "over-sampling"
_ensemble_type = ResampleBoostClassifier._ensemble_type
_training_type = ResampleBoostClassifier._training_type

_properties = {
    "solution_type": _solution_type,
    "sampling_type": _sampling_type,
    "ensemble_type": _ensemble_type,
    "training_type": _training_type,
}


@Substitution(
    n_jobs_sampler=_get_parameter_docstring("n_jobs_sampler", **_properties),
    early_termination=_get_parameter_docstring("early_termination", **_properties),
    random_state=_get_parameter_docstring("random_state", **_properties),
    example=_get_example_docstring(_method_name),
)
class KmeansSMOTEBoostClassifier(ResampleBoostClassifier):
    """Kmeans-SMOTE over-sampling integrated in the learning of AdaBoost.

    KmeansSMOTEBoost is similar to SMOTEBoost [1]_, but use KmeansSMOTE
    instead of SMOTE. It alleviates the problem of class balancing by
    KmeansSMOTE over-sampling the sample at each iteration of the boosting algorithm.

    This KmeansSMOTEBoost implementation supports multi-class classification.

    Parameters
    ----------
    estimator : estimator object, default=None
        The base estimator from which the boosted ensemble is built.
        Support for sample weighting is required, as well as proper
        ``classes_`` and ``n_classes_`` attributes. If ``None``, then
        the base estimator is ``DecisionTreeClassifier(max_depth=1)``.

    n_estimators : int, default=50
        The maximum number of estimators at which boosting is terminated.
        In case of perfect fit, the learning procedure is stopped early.

    k_neighbors : int or object, default=2
        If ``int``, number of nearest neighbors to used to construct synthetic
        samples in KmeansSMOTE. If object, an estimator that inherits from
        :class:`~sklearn.neighbors.base.KNeighborsMixin` that will be used to
        find the k_neighbors.

    {n_jobs_sampler}

    kmeans_estimator : int or object, default=None
        A KMeans instance or the number of clusters to be used. By default,
        we used a :class:`~sklearn.cluster.MiniBatchKMeans` which tend to be
        better with large number of samples.

    cluster_balance_threshold : "auto" or float, default=0.1
        The threshold at which a cluster is called balanced and where samples
        of the class selected for SMOTE will be oversampled. If "auto", this
        will be determined by the ratio for each class, or it can be set
        manually.

    density_exponent : "auto" or float, default="auto"
        This exponent is used to determine the density of a cluster. Leaving
        this to "auto" will use a feature-length based exponent.

    learning_rate : float, default=1.0
        Learning rate shrinks the contribution of each classifier by
        ``learning_rate``. There is a trade-off between ``learning_rate`` and
        ``n_estimators``.

    algorithm : {{'SAMME', 'SAMME.R'}}, default='SAMME'
        If 'SAMME.R' then use the SAMME.R real boosting algorithm.
        ``estimator`` must support calculation of class probabilities.
        If 'SAMME' then use the SAMME discrete boosting algorithm.
        The SAMME.R algorithm typically converges faster than SAMME,
        achieving a lower test error with fewer boosting iterations.

    {early_termination}

    {random_state}

    Attributes
    ----------
    estimator_ : estimator
        The base estimator from which the ensemble is grown.

    sampler_ : KMeansSMOTE
        The base sampler.

    estimators_ : list of classifiers
        The collection of fitted sub-estimators.

    samplers_ : list of KMeansSMOTE
        The collection of used samplers.

    classes_ : ndarray of shape (n_classes,)
        The classes labels.

    n_classes_ : int
        The number of classes.

    estimator_weights_ : ndarray of shape (n_estimator,)
        Weights for each estimator in the boosted ensemble.

    estimator_errors_ : ndarray of shape (n_estimator,)
        Classification error for each estimator in the boosted
        ensemble.

    estimators_n_training_samples_ : list of ints
        The number of training samples for each fitted
        base estimators.

    feature_importances_ : ndarray of shape (n_features,)
        The feature importances if supported by the ``estimator``.

    See Also
    --------
    SMOTEBoostClassifier : SMOTE over-sampling integrated in AdaBoost.

    OverBoostClassifier : Random over-sampling integrated in AdaBoost.

    SMOTEBaggingClassifier : Bagging with intergrated SMOTE over-sampling.

    References
    ----------
    .. [1] Chawla, N. V., Lazarevic, A., Hall, L. O., & Bowyer, K. W.
       "SMOTEBoost: Improving prediction of the minority class in boosting."
       European conference on principles of data mining and knowledge discovery.
       Springer, Berlin, Heidelberg, (2003): 107-119.

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
        k_neighbors: int = 2,
        n_jobs_sampler=None,
        kmeans_estimator=None,
        cluster_balance_threshold=0.1,
        density_exponent="auto",
        learning_rate: float = 1.0,
        algorithm: str = "SAMME",
        early_termination: bool = False,
        random_state=None,
    ):

        sampler = _sampler_class()
        sampling_type = _sampling_type

        super(KmeansSMOTEBoostClassifier, self).__init__(
            estimator=estimator,
            n_estimators=n_estimators,
            sampler=sampler,
            sampling_type=sampling_type,
            learning_rate=learning_rate,
            algorithm=algorithm,
            early_termination=early_termination,
            random_state=random_state,
        )

        self.__name__ = _method_name
        self._sampling_type = _sampling_type
        self._sampler_class = _sampler_class
        self._properties = _properties

        self.k_neighbors = k_neighbors
        self.k_neighbors_ = check_type(k_neighbors, "k_neighbors", numbers.Integral)
        self.n_jobs_sampler = n_jobs_sampler
        self.kmeans_estimator = kmeans_estimator
        self.cluster_balance_threshold = cluster_balance_threshold
        self.density_exponent = density_exponent
        if self.k_neighbors_ < 1:
            raise ValueError(
                f"The 'k_neighbors' parameter of NearestNeighbors must be"
                f" an int in the range [1, inf) or None. Got {self.k_neighbors_} instead."
            )

    @_deprecate_positional_args
    @FuncSubstitution(
        target_label=_get_parameter_docstring("target_label", **_properties),
        n_target_samples=_get_parameter_docstring("n_target_samples", **_properties),
        balancing_schedule=_get_parameter_docstring("balancing_schedule"),
        eval_datasets=_get_parameter_docstring("eval_datasets"),
        eval_metrics=_get_parameter_docstring("eval_metrics"),
        train_verbose=_get_parameter_docstring("train_verbose", **_properties),
    )
    def fit(
        self,
        X,
        y,
        *,
        sample_weight=None,
        target_label: int = None,
        n_target_samples: int or dict = None,
        balancing_schedule: str or function = "uniform",
        eval_datasets: dict = None,
        eval_metrics: dict = None,
        train_verbose: bool or int or dict = False,
    ):
        """Build a KmeansSMOTEBoost classifier from the training set (X, y).

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
        """

        kmeans_smote_sampler_kwargs = {
            "k_neighbors": self.k_neighbors_,
            "n_jobs": self.n_jobs_sampler,
            "kmeans_estimator": self.kmeans_estimator,
            "cluster_balance_threshold": self.cluster_balance_threshold,
            "density_exponent": self.density_exponent,
        }
        update_x_y_after_resample = True

        return self._fit(
            X,
            y,
            sample_weight=sample_weight,
            sampler_kwargs=kmeans_smote_sampler_kwargs,
            update_x_y_after_resample=update_x_y_after_resample,
            target_label=target_label,
            n_target_samples=n_target_samples,
            balancing_schedule=balancing_schedule,
            eval_datasets=eval_datasets,
            eval_metrics=eval_metrics,
            train_verbose=train_verbose,
        )


# %%

if __name__ == "__main__":  # pragma: no cover
    import warnings

    warnings.filterwarnings("ignore")
    from collections import Counter
    from copy import copy

    from sklearn.datasets import make_classification
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier

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
    print("Original training dataset shape %s" % origin_distr)

    init_kwargs_default = {
        "estimator": None,
        # 'estimator': DecisionTreeClassifier(max_depth=2),
        "n_estimators": 100,
        "k_neighbors": 5,
        "learning_rate": 1.0,
        "algorithm": "SAMME.R",
        "random_state": 10,
        # 'random_state': None,
    }
    fit_kwargs_default = {
        "X": X_train,
        "y": y_train,
        "sample_weight": None,
        "target_label": None,
        "n_target_samples": None,
        # 'n_target_samples': target_distr,
        "balancing_schedule": "uniform",
        "eval_datasets": {"valid": (X_valid, y_valid)},
        "eval_metrics": {
            "acc": (accuracy_score, {}),
            "balanced_acc": (balanced_accuracy_score, {}),
            "weighted_f1": (f1_score, {"average": "weighted"}),
        },
        "train_verbose": {
            "granularity": 10,
            "print_distribution": True,
            "print_metrics": True,
        },
    }

    ensembles = {}

    init_kwargs, fit_kwargs = copy(init_kwargs_default), copy(fit_kwargs_default)
    kmsmoteboost = KmeansSMOTEBoostClassifier(**init_kwargs).fit(**fit_kwargs)
    ensembles["kmsmoteboost"] = kmsmoteboost

    init_kwargs, fit_kwargs = copy(init_kwargs_default), copy(fit_kwargs_default)
    fit_kwargs.update({"balancing_schedule": "progressive"})
    kmsmoteboost_prog = KmeansSMOTEBoostClassifier(**init_kwargs).fit(**fit_kwargs)
    ensembles["kmsmoteboost_prog"] = kmsmoteboost_prog

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
