"""AdaUBoostClassifier: An AdaUBoost classifier."""

# Authors: Zhining Liu <zhining.liu@outlook.com>
# License: MIT

# %%
LOCAL_DEBUG = False

if not LOCAL_DEBUG:
    from ...utils._docstring import (
        FuncSubstitution,
        Substitution,
        _get_example_docstring,
        _get_parameter_docstring,
    )
    from ...utils._validation import _deprecate_positional_args
    from .._boost import ReweightBoostClassifier
else:  # pragma: no cover
    import sys  # For local test

    sys.path.append("../..")
    from ensemble._boost import ReweightBoostClassifier
    from utils._validation import _deprecate_positional_args
    from utils._docstring import (
        Substitution,
        FuncSubstitution,
        _get_parameter_docstring,
        _get_example_docstring,
    )

import numbers

import numpy as np
import pandas as pd

# Properties
_method_name = "AdaUBoostClassifier"

_solution_type = ReweightBoostClassifier._solution_type
_ensemble_type = ReweightBoostClassifier._ensemble_type
_training_type = ReweightBoostClassifier._training_type

_properties = {
    "solution_type": _solution_type,
    "ensemble_type": _ensemble_type,
    "training_type": _training_type,
}


# All possible string values for beta
SET_BETA_HOW = ("uniform", "inverse", "log1p-inverse")


@Substitution(
    early_termination=_get_parameter_docstring("early_termination", **_properties),
    random_state=_get_parameter_docstring("random_state", **_properties),
    example=_get_example_docstring(_method_name),
)
class AdaUBoostClassifier(ReweightBoostClassifier):
    """An AdaUBoost cost-sensitive classifier.

    AdaUBoost [1]_, a variant of AdaBoost, is designed to be optimize
    an unequal loss on imbalanced training set by preprocess, and also
    manipulate the training distribution within successive boosting rounds.
    The purpose is to reduce the cumulative misclassification cost.

    This AdaUBoost implementation supports multi-class classification.

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

    learning_rate : float, default=1.
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
    estimators_ : list of classifiers
        The collection of fitted sub-estimators.

    beta_ : dict
        The beta values of each class.

    classes_ : array of shape = [n_classes]
        The classes labels.

    n_classes_ : int
        The number of classes.

    estimator_weights_ : array of floats
        Weights for each estimator in the boosted ensemble.

    estimator_errors_ : array of floats
        Classification error for each estimator in the boosted
        ensemble.

    estimators_n_training_samples_ : list of ints
        The number of training samples for each fitted
        base estimators.

    feature_importances_ : array of shape = [n_features]
        The feature importances if supported by the ``estimator``.

    See also
    --------
    AsymBoostClassifier : An Asymmetric Boosting classifier.

    AdaCostClassifier : An AdaCost cost-sensitive boosting classifier.

    References
    ----------
    .. [1] Shawe-Taylor, G. K. J., & Karakoulas, G. "Optimizing classifiers
       for imbalanced training sets." Advances in neural information
       processing systems 11.11 (1999): 253.

    Examples
    --------
    >>> from imbens.ensemble import AdaUBoostClassifier
    >>> from sklearn.datasets import make_classification
    >>>
    >>> X, y = make_classification(n_samples=1000, n_classes=3,
    ...                            n_informative=4, weights=[0.2, 0.3, 0.5],
    ...                            random_state=0)
    >>> clf = AdaUBoostClassifier(random_state=0)
    >>> clf.fit(X, y)  # doctest: +ELLIPSIS
    AdaUBoostClassifier(...)
    >>> clf.predict(X)  # doctest: +ELLIPSIS
    array([...])
    """

    @_deprecate_positional_args
    def __init__(
        self,
        estimator=None,
        n_estimators: int = 50,
        *,
        learning_rate: float = 1.0,
        algorithm: str = "SAMME",
        early_termination: bool = False,
        random_state=None,
    ):

        self.__name__ = "AdaUBoostClassifier"

        super(AdaUBoostClassifier, self).__init__(
            estimator=estimator,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            algorithm=algorithm,
            early_termination=early_termination,
            random_state=random_state,
        )

    def _compute_mult_in_exp_weights_array(self, y_true, y_pred=None, init_mult=False):
        """Return the adauboost weights of shape = (n_samples,).

        Parameters
        ----------
        y_true : array-like of shape = [n_samples, 1]
                 True class values.

        y_pred : array-like of shape = [n_samples, 1]
                 Predicted class values.
        """

        df = pd.DataFrame({"y_true": y_true})
        if init_mult:
            mult_weight_table = self._init_mult_weight_table
        else:
            mult_weight_table = self._iter_mult_weight_table
        df = df.merge(mult_weight_table, how="left", on=["y_true"])

        return df["mult_weight"].values

    def _check_beta(self, beta: str or dict) -> dict:
        """Private function for checking the parameter 'beta'."""

        if beta is None:
            beta = self._set_beta()
        elif isinstance(beta, str):
            beta = self._set_beta(how=beta)
        elif isinstance(beta, dict):
            all_classes = self.classes_
            if set(beta.keys()) != set(all_classes):
                raise ValueError(
                    f"When 'beta' is a dict, it should specify"
                    f" beta values for all classes, keys should"
                    f" be {all_classes}, got {beta.keys()}."
                )
            if not all([isinstance(value, numbers.Number) for value in beta.values()]):
                not_number_idx = [
                    not isinstance(value, numbers.Number) for value in beta.values()
                ]
                not_number_types = set(
                    [
                        type(element)
                        for element in np.array(list(beta.values()))[not_number_idx]
                    ]
                )
                raise ValueError(
                    f"When 'beta' is a dict, all values should be"
                    f" Integer or Real number, got type {not_number_types}"
                    f" in values."
                )
        else:
            raise TypeError(
                f"'beta' should be one of {SET_BETA_HOW} or a `dict` that specifies"
                f" the beta value of each class."
            )

        return beta

    def _set_beta(self, how: str = "inverse") -> dict:
        """Set the self.beta_ by 'how'."""

        classes, origin_distr = self._encode_map.values(), self.origin_distr_
        c_maj = max(origin_distr.keys(), key=(lambda x: origin_distr[x]))
        beta = [origin_distr[c_maj] / origin_distr[c_min] for c_min in classes]
        if how == "uniform":
            return dict(zip(classes, np.ones_like(beta)))
        elif how == "inverse":
            return dict(zip(classes, beta))
        elif how == "log1p-inverse":
            return dict(zip(classes, np.log1p(beta)))
        else:
            raise ValueError(
                f"When 'beta' is string, it should be" f" in {SET_BETA_HOW}, got {how}."
            )

    def _set_class_weights(self):
        """Set cost map table for preprocessing and within-iter update."""

        beta = self.beta_
        beta_keys = np.array(list(beta.keys()))
        beta_values = np.array(list(beta.values()))
        # Algorithm 1, step 1 of [1]_
        self._init_mult_weight_table = pd.DataFrame(
            {
                "y_true": beta_keys,
                "mult_weight": beta_values,
            }
        )
        # Algorithm 1, step 2 of [1]_
        self._iter_mult_weight_table = pd.DataFrame(
            {
                "y_true": beta_keys,
                "mult_weight": np.power(1 / beta_values, 1 / self.n_estimators),
            }
        )

    def _preprocess_sample_weight(self, sample_weight, y):
        """Preprocessing the initial data distribution."""

        # Validate and store the beta parameter
        self.beta_ = self._check_beta(self.beta)
        self._set_class_weights()
        # Preprocess the sample_weight according to
        # Algorithm 1, step 1 in reference [1].
        sample_weight *= self._compute_mult_in_exp_weights_array(
            y,
            init_mult=True,
        )
        return sample_weight

    @_deprecate_positional_args
    @FuncSubstitution(
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
        beta=None,
        eval_datasets: dict = None,
        eval_metrics: dict = None,
        train_verbose: bool or int or dict = False,
    ):
        """Build a AdaUBoost classifier from the training set (X, y).

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

        beta : str or dict, default=None
            Beta values for each class.

            - If ``None``, equivalent to ``'inverse'``.
            - If ``'uniform'``, set beta value to 1 for all classes.
            - If ``'inverse'``, set class :math:`c`'s beta value to
              :math:`N_{max} / N_{c}`, where :math:`N_{c}` is the number of samples
              of class :math:`c` and :math:`N_{max}` is the number of samples of the
              largest class.
            - If ``'log1p-inverse'``, apply ``numpy.log1p`` on the result beta values
              of 'inverse'.
            - If ``dict``, the keys of type ``int`` correspond to the classes, and
              the values of type ``int`` or ``float`` correspond to the beta value
              for each class.

        %(eval_datasets)s

        %(eval_metrics)s

        %(train_verbose)s

        Returns
        -------
        self : object
        """

        self.beta = beta

        return self._fit(
            X,
            y,
            sample_weight=sample_weight,
            cost_matrix=None,
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
        "n_estimators": 100,
        "learning_rate": 0.5,
        "algorithm": "SAMME",
        # 'random_state': 42,
        "random_state": None,
    }
    fit_kwargs_default = {
        "X": X_train,
        "y": y_train,
        "beta": "inverse",
        "sample_weight": None,
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
    adauboost = AdaUBoostClassifier(**init_kwargs).fit(**fit_kwargs)
    ensembles["adauboost"] = adauboost

    init_kwargs, fit_kwargs = copy(init_kwargs_default), copy(fit_kwargs_default)
    fit_kwargs.update(
        {
            "beta": "log1p-inverse",
        }
    )
    adauboost_log = AdaUBoostClassifier(**init_kwargs).fit(**fit_kwargs)
    ensembles["adauboost_log"] = adauboost_log

    init_kwargs, fit_kwargs = copy(init_kwargs_default), copy(fit_kwargs_default)
    fit_kwargs.update(
        {
            "beta": "uniform",
        }
    )
    adauboost_uniform = AdaUBoostClassifier(**init_kwargs).fit(**fit_kwargs)
    ensembles["adauboost_uniform"] = adauboost_uniform

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
