"""AsymBoostClassifier: An Asymmetric Boosting classifier.
"""

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

import numpy as np
import pandas as pd

# Properties
_method_name = "AsymBoostClassifier"

_solution_type = ReweightBoostClassifier._solution_type
_ensemble_type = ReweightBoostClassifier._ensemble_type
_training_type = ReweightBoostClassifier._training_type

_properties = {
    "solution_type": _solution_type,
    "ensemble_type": _ensemble_type,
    "training_type": _training_type,
}


@Substitution(
    early_termination=_get_parameter_docstring("early_termination", **_properties),
    random_state=_get_parameter_docstring("random_state", **_properties),
    example=_get_example_docstring(_method_name),
)
class AsymBoostClassifier(ReweightBoostClassifier):
    """An Asymmetric Boosting classifier.

    Asymmetric Boosting (AsymBoost) [1]_, a variant of AdaBoost, is a
    cost-sensitive boosting method. It uses the asymmetric misclassication
    cost to update the training distribution on successive boosting rounds.

    This AsymBoost implementation supports multi-class classification.

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

    cost_matrix_ : array of shape = [n_classes, n_classes]
        The used cost matrix. The rows represent the predicted class and
        columns represent the actual class. The order of the classes
        corresponds to that in the attribute ``classes_``.

    cost_table_asymboost_ : DataFrame of shape = [n_classes*n_classes, 3]
        The used cost map table.

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
    AdaUBoostClassifier : An AdaUBoost cost-sensitive classifier.

    AdaCostClassifier : An AdaCost cost-sensitive boosting classifier.

    References
    ----------
    .. [1] Viola, P., & Jones, M. "Fast and robust classification using
       asymmetric adaboost and a detector cascade." Advances in Neural
       Information Processing System 14 (2001).

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
        learning_rate: float = 1.0,
        algorithm: str = "SAMME",
        early_termination: bool = False,
        random_state=None,
    ):

        super(AsymBoostClassifier, self).__init__(
            estimator=estimator,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            algorithm=algorithm,
            early_termination=early_termination,
            random_state=random_state,
        )

        self.__name__ = _method_name
        self._properties = _properties

    def _compute_mult_out_exp_weights_array(self, y_true, y_pred):
        """Return the asymmetric weights of shape = (n_samples,).

        Parameters
        ----------
        y_true : array-like of shape = [n_samples, 1]
                 True class values.

        y_pred : array-like of shape = [n_samples, 1]
                 Predicted class values.
        """
        df = pd.DataFrame({"y_pred": y_pred, "y_true": y_true})
        df = df.merge(self.cost_table_asymboost_, how="left", on=["y_pred", "y_true"])

        return df["asym_weight"].values

    def _cost_matrix_to_asymmetric_weights(self, cost_matrix):
        """Creates a table of asymmetric weight map table from the cost matrix.

        Parameters
        ----------
        cost_matrix : array-like of shape = [n_classes, n_classes]

        Returns
        -------
        df : pd.DataFrame of shape = [n_classes, 2]

        """
        table = np.empty((0, 3))

        for (x, y), value in np.ndenumerate(cost_matrix):
            # Section 4 of [1]
            table = np.vstack(
                (
                    table,
                    np.array(
                        [x, y, np.exp(1 / self.n_estimators * np.log(np.sqrt(value)))]
                    ),
                )
            )

        return pd.DataFrame(table, columns=["y_pred", "y_true", "asym_weight"])

    def _validate_cost_matrix(self, cost_matrix, n_classes):
        """validate the cost matrix & set the cost map table."""

        cost_matrix = super()._validate_cost_matrix(cost_matrix, n_classes)
        self.cost_table_asymboost_ = self._cost_matrix_to_asymmetric_weights(
            cost_matrix
        )

        return cost_matrix

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
        cost_matrix="inverse",
        eval_datasets: dict = None,
        eval_metrics: dict = None,
        train_verbose: bool or int or dict = False,
    ):
        """Build a AsymBoost classifier from the training set (X, y).

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

        cost_matrix : str or numpy.ndarray, default=None
            A matrix representing the cost of misclassification.

            - If ``None``, equivalent to ``'inverse'``.
            - If ``'uniform'``, set misclassification cost to be equal.
            - If ``'inverse'``, set misclassification cost by inverse class frequency.
            - If ``numpy.ndarray`` of shape (n_classes, n_classes), the rows
              represent the predicted class and columns represent the actual class.
              Thus the value at :math:`i`-th row :math:`j`-th column (:math:`C_{ij}`)
              represents the cost of classifying a sample from class :math:`j` to
              class :math:`i`.
              Note: the cost matrix must satisfy that :math:`C_{ij} * C_{ji} = 1`.

        %(eval_datasets)s

        %(eval_metrics)s

        %(train_verbose)s

        Returns
        -------
        self : object
        """

        # Note that AsymBoost requires the cost matrix to satisfy
        # C_{ij} * C_{ji} = 1, thus the 'log1p-inverse' option is not
        # available in this situation.
        if type(cost_matrix) == str and cost_matrix == "log1p-inverse":
            raise ValueError(
                "'log1p-inverse' option is not available for "
                "AsymBoostClassifier. Please use 'inverse' instead."
            )

        return self._fit(
            X,
            y,
            sample_weight=sample_weight,
            cost_matrix=cost_matrix,
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
        "learning_rate": 1.0,
        "algorithm": "SAMME.R",
        "random_state": 42,
        # 'random_state': None,
    }
    fit_kwargs_default = {
        "X": X_train,
        "y": y_train,
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
    asymboost = AsymBoostClassifier(**init_kwargs).fit(**fit_kwargs)
    ensembles["asymboost"] = asymboost

    init_kwargs, fit_kwargs = copy(init_kwargs_default), copy(fit_kwargs_default)
    fit_kwargs.update(
        {
            "cost_matrix": "uniform",
        }
    )
    asymboost_uniform = AsymBoostClassifier(**init_kwargs).fit(**fit_kwargs)
    ensembles["asymboost_uniform"] = asymboost_uniform

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
