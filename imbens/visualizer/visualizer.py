"""ImbalancedEnsembleVisualizer is built for visualizing 
ensemble estimators.
"""

# Authors: Zhining Liu <zhining.liu@outlook.com>
# License: MIT

# %%
LOCAL_DEBUG = False

if not LOCAL_DEBUG:
    from ..utils._validation_data import check_eval_datasets
    from ..utils._validation_param import (
        check_eval_metrics,
        check_has_diff_elements,
        check_plot_figsize,
        check_type,
        check_visualizer_ensembles,
    )
else:  # pragma: no cover
    import sys  # For local test

    sys.path.append("..")
    from utils._validation_data import check_eval_datasets
    from utils._validation_param import (
        check_eval_metrics,
        check_visualizer_ensembles,
        check_has_diff_elements,
        check_plot_figsize,
        check_type,
    )

import numbers
from copy import copy
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import font_manager, rcParams
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    BaseEnsemble,
    RandomForestClassifier,
)
from sklearn.exceptions import NotFittedError
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

DATAFRAME_COLUMNS = [
    "n_estimators",
    "method",
    "dataset",
    "metric",
    "score",
    "n_samples",
]

SPLIT_BY = ["method", "dataset"]

LINEPLOT_KWARGS_DEFAULT = {
    "markers": True,
    "alpha": 0.8,
}

HEATMAP_KWARGS_DEFAULT = {
    "alpha": 0.8,
}

FONT_DEFAULT = "Consolas"

RESERVED_SUPTITLE_INCHES = 0.4


"""
fontsize or size	    float or {'xx-small', 'x-small', 'small', 'medium', 
                        'large', 'x-large', 'xx-large'}
fontstretch or stretch	{a numeric value in range 0-1000, 'ultra-condensed', 
                        'extra-condensed', 'condensed', 'semi-condensed', 'normal', 
                        'semi-expanded', 'expanded', 'extra-expanded', 'ultra-expanded'}
fontstyle or style	    {'normal', 'italic', 'oblique'}
fontvariant or variant	{'normal', 'small-caps'}
fontweight or weight	{a numeric value in range 0-1000, 'ultralight', 'light', 
                        'normal', 'regular', 'book', 'medium', 'roman', 'semibold', 
                        'demibold', 'demi', 'bold', 'heavy', 'extra bold', 'black'}
"""


def set_ax_border(ax, border_color="black", border_width=2):
    """Set the border color and width."""

    for _, spine in ax.spines.items():
        spine.set_color(border_color)
        spine.set_linewidth(border_width)

    return ax


class ImbalancedEnsembleVisualizer:
    """A visualizer, providing several utilities to visualize:

        - the model performance curve with respect to the number of
          base estimators / training samples, could be grouped by
          method, evaluation dataset, or both;
        - the confusion matrix of the model prediction.

    This visualization tool can be used to:

        - provide further information about the training process (for
          iteratively trained ensemble) of a single ensemble model;
        - or to compare the performance of multiple different ensemble
          models in an intuitive way.


    Parameters
    ----------
    eval_datasets : dict, default=None
        Dataset(s) used for evaluation and visualization.
        The keys should be strings corresponding to evaluation datasets' names.
        The values should be tuples corresponding to the input samples and target
        values.

        Example: ``eval_datasets = {'valid' : (X_valid, y_valid)}``

    eval_metrics : dict, default=None
        Metric(s) used for evaluation and visualization.

        - If ``None``, use 3 default metrics:
            ``'acc'``: sklearn.metrics.accuracy_score();
            ``'balanced_acc'``: sklearn.metrics.balanced_accuracy_score();
            ``'weighted_f1'``: sklearn.metrics.f1_score(acerage='weighted');
        - If ``dict``, the keys should be strings corresponding to evaluation
            metrics' names. The values should be tuples corresponding to the metric
            function (``callable``) and additional kwargs (``dict``).
            - The metric function should at least take 2 positional arguments
            ``y_true``, ``y_pred``, and returns a ``float`` as its score.
            - The metric additional kwargs should specify the additional arguments
            that need to be passed into the metric function.

        Example:
        ``{'weighted_f1': (sklearn.metrics.f1_score, {'average': 'weighted'})}``

    Attributes
    ----------
    perf_dataframe_ : DataFrame
        The performance scores of all ensemble methods on given evaluation
        datasets and metrics.

    conf_matrices_ : dict
        The confusion matrices of all ensemble methods' predictions on given
        evaluation datasets. The keys are the ensemble names, the values are
        dicts with dataset names as keys and corresponding confusion matrices
        as values. Each confusion matrix is a ndarray of shape (n_classes,
        n_classes), The order of the classes corresponds to that in the
        ensemble classifier's attribute ``classes_``.

    Examples
    --------
    >>> from imbens.visualizer import ImbalancedEnsembleVisualizer
    >>> from imbens.ensemble import (
    >>>    SelfPacedEnsembleClassifier,
    >>>    RUSBoostClassifier,
    >>>    SMOTEBoostClassifier,
    >>> )
    >>> from sklearn.datasets import make_classification
    >>>
    >>> X, y = make_classification(n_samples=1000, n_classes=3,
    ...                            n_informative=4, weights=[0.2, 0.3, 0.5],
    ...                            random_state=0)
    >>> ensembles = {
    >>>     'spe': SelfPacedEnsembleClassifier().fit(X, y),
    >>>     'rusboost': RUSBoostClassifier().fit(X, y),
    >>>     'smoteboost': SMOTEBoostClassifier().fit(X, y),
    >>> }
    >>> visualizer = ImbalancedEnsembleVisualizer().fit(
    >>>     ensembles = ensembles,
    >>>     granularity = 5,
    >>> )
    >>> visualizer.performance_lineplot()
    >>> visualizer.confusion_matrix_heatmap()
    """

    def __init__(self, eval_metrics: dict = None, eval_datasets: dict = None):

        self.__name__ = "ImbalancedEnsembleVisualizer"
        self._fitted = False

        # Check evaluation metrics
        self.eval_metrics_ = check_eval_metrics(eval_metrics)

        # Check evaluation datasets
        check_x_y_args = {
            "accept_sparse": ["csr", "csc"],
            "force_all_finite": False,
            "dtype": None,
        }
        self.eval_datasets_ = check_eval_datasets(eval_datasets, **check_x_y_args)

        # Set default rcParams
        rcParams.update(
            {
                "font.weight": "roman",
            }
        )
        try:
            font_manager.findfont(FONT_DEFAULT, fallback_to_default=False)
            rcParams.update(
                {
                    "font.family": FONT_DEFAULT,
                }
            )
        except:
            print(
                f"\nFont family ['{FONT_DEFAULT}'] not found in your system."
                f" Falling back to DejaVu Sans."
            )

        # Default styles of different titles
        self.suptitle_style = {
            "size": "x-large",
            "stretch": "expanded",
            "style": "italic",
            "variant": "small-caps",
            "weight": "black",
        }
        self.row_col_title_style = {
            "size": "large",
            "weight": "bold",
            "bbox": {
                "boxstyle": "round",
                "pad": 0.25,
                "fc": "white",
                "ec": "black",
                "lw": 1,
                "alpha": 0.5,
            },
        }
        self.axis_title_style = {
            "size": "medium",
            "weight": "bold",
        }

    def fit(
        self,
        ensembles: dict,
        granularity: int = None,
    ):
        """Fit visualizer to the given ensemble models.
        Collect data for visualization with the given granularity.

        Parameters
        ----------
        ensembles : dict
            The ensemble models and their names for visualization.
            The keys should be strings corresponding to ensemble names.
            The values should be fitted imbalance_ensemble.ensemble or
            sklearn.ensemble estimator objects.

            Note: all training/evaluation datasets (if any) of all ensemble
            estimators should be sampled from the same task/distribution for
            comparable visualization.

        granularity : int, default=None
            The granularity of performance evaluation.
            For each (ensemble, eval_dataset) pair, the performance evaluation
            is conducted by starting with empty ensemble, and add ``granularity``
            fitted base estimators per round. If ``None``, it will be set to
            ``max_n_estimators/5``, where ``max_n_estimators`` is the maximum
            number of base estimators among all models given in ``ensembles``.

            .. warning::
                Setting a small ``granularity`` value can be costly when the
                evaluation data is large or the model predictions/metric scores
                are hard to compute. If you find that ``fit()`` is running slow,
                try using a larger ``granularity``.

        Returns
        -------
        self : object
        """

        eval_datasets_, eval_metrics_ = self.eval_datasets_, self.eval_metrics_
        # eval_datasets_ needs to be further validated in check_visualizer_ensembles
        (
            self.ensembles_,
            self.eval_datasets_,
            self.vis_format_,
        ) = check_visualizer_ensembles(ensembles, eval_datasets_, eval_metrics_)
        (
            self.ensembles_n_training_samples_,
            self.ensembles_has_n_training_samples_,
        ) = self._check_ensembles_stored_n_training_samples()

        # Check granularity
        if granularity is not None:
            granularity_ = check_type(granularity, "granularity", numbers.Integral)
        else:
            max_n_estimators = max(
                [len(ens.estimators_) for ens in self.ensembles_.values()]
            )
            granularity_ = max(int(max_n_estimators / 5), 1)
        self.granularity_ = granularity_

        # Collect data for visualization
        self.perf_dataframe_ = self._collect_all_ensemble_performance_data()
        self.conf_matrices_ = self._collect_all_ensemble_confusion_matrix()

        self._fitted = True

        return self

    def _check_ensembles_stored_n_training_samples(self):
        """Check whether the ensembles stored the number of training
        samples of each base estimator.

        Returns
        -------
        ensembles_n_training_samples : dict
            The keys are strings corresponding to ensemble names.
            The values are the fetched arrays of number of training
            samples of each base estimator (None if fetch fails).

        ensembles_has_n_training_samples : dict
            The keys are strings corresponding to ensemble names.
            The values are ``bool`` that indicates whether the ensemble
            stored the number of training samples.
        """

        ensembles_has_n_training_samples = {}
        ensembles_n_training_samples = {}
        for name, ensemble in self.ensembles_.items():
            (
                estimators_n_training_samples_,
                flag,
            ) = self._get_ensemble_stored_n_training_samples(ensemble)
            ensembles_n_training_samples[name] = estimators_n_training_samples_
            ensembles_has_n_training_samples[name] = flag

        return ensembles_n_training_samples, ensembles_has_n_training_samples

    @staticmethod
    def _get_ensemble_stored_n_training_samples(ensemble):
        """Try to get the number of training samples of each base estimator.

        Parameters
        ----------
        ensembles : dict
            The ensemble models and their names for visualization.
            The keys should be strings corresponding to ensemble names.
            The values should be fitted imbalance_ensemble.ensemble or
            sklearn.ensemble estimator objects.

        Returns
        -------
        ensembles_n_training_samples : array of shape (n_estimators, )
            The fetched arrays of number of training samples of each
            base estimator (None if fetch fails).

        success_flag : bool
            Whether the fetch is successful.
        """
        max_n_estimators = len(ensemble.estimators_)
        # If not imbalanced-ensemble classifier
        if not hasattr(ensemble, "estimators_n_training_samples_"):
            # If Bagging classifier
            if isinstance(ensemble, BaggingClassifier):
                out = (
                    np.full(max_n_estimators, fill_value=ensemble._max_samples),
                    True,
                )
            else:
                # If other sklearn ensemble classifier
                out = (None, False)
        else:
            out = (np.array(ensemble.estimators_n_training_samples_), True)
        return out

    @staticmethod
    def _get_boost_predict_proba(X, ensemble, n_estimators: int = None):
        """Get n-estimators predicted probabilities from a boosting-like ensemble."""

        # Need to consider the estimator_weights_ in boosting
        tmp_ensemble_estimators, tmp_ensemble_estimator_weights_ = copy(
            ensemble.estimators_
        ), copy(ensemble.estimator_weights_)
        ensemble.estimators_ = tmp_ensemble_estimators[: n_estimators + 1]
        ensemble.estimator_weights_ = tmp_ensemble_estimator_weights_[
            : n_estimators + 1
        ]
        y_pred_proba = ensemble.predict_proba(X)
        ensemble.estimators_, ensemble.estimator_weights_ = (
            tmp_ensemble_estimators,
            tmp_ensemble_estimator_weights_,
        )

        return y_pred_proba

    @staticmethod
    def _get_parallel_predict_proba(X, ensemble, n_estimators: int = None):
        """Get n-estimators predicted probabilities from a bagging-like ensemble."""

        # Check whether the ensemble has verbose attribute
        has_verbose = hasattr(ensemble, "verbose")
        # Temporarily disable built-in verbose
        if has_verbose:
            tmp_verbose, ensemble.verbose = ensemble.verbose, 0
        tmp_ensemble_estimators = copy(ensemble.estimators_)
        ensemble.estimators_ = tmp_ensemble_estimators[: n_estimators + 1]
        y_pred_proba = ensemble.predict_proba(X)
        ensemble.estimators_ = tmp_ensemble_estimators
        # Recover verbose attribute
        if has_verbose:
            ensemble.verbose = tmp_verbose

        return y_pred_proba

    @staticmethod
    def _get_ensemble_predict_proba(X, ensemble, n_estimators: int = None):
        """Get n-estimators predicted probabilities from a general ensemble."""

        # Check whether the ensemble has verbose attribute
        has_verbose = hasattr(ensemble, "verbose")
        # Temporarily disable built-in verbose
        if has_verbose:
            tmp_verbose, ensemble.verbose = ensemble.verbose, 0
        tmp_ensemble_estimators = copy(ensemble.estimators_)
        ensemble.estimators_ = tmp_ensemble_estimators[: n_estimators + 1]
        y_pred_proba = ensemble.predict_proba(X)
        ensemble.estimators_ = tmp_ensemble_estimators
        # Recover verbose attribute
        if has_verbose:
            ensemble.verbose = tmp_verbose

        return y_pred_proba

    def _get_predict_proba(self, X, ensemble, n_estimators: int = None):
        """Get n-estimators predicted probabilities from a ensemble."""

        # If n_estimators is not specified
        if n_estimators is None:
            return ensemble.predict_proba(X)
        elif isinstance(n_estimators, numbers.Integral):
            # If n_estimators equal to the number of base estimators
            if n_estimators == len(ensemble.estimators_):
                return ensemble.predict_proba(X)
            # If n_estimators less than the number of base estimators
            elif n_estimators < len(ensemble.estimators_):
                # Check ensemble type
                if isinstance(ensemble, AdaBoostClassifier):
                    return self._get_boost_predict_proba(X, ensemble, n_estimators)
                elif isinstance(ensemble, (BaggingClassifier, RandomForestClassifier)):
                    return self._get_parallel_predict_proba(X, ensemble, n_estimators)
                elif isinstance(ensemble, BaseEnsemble):
                    return self._get_ensemble_predict_proba(X, ensemble, n_estimators)
                # If ensemble is not a ensemble estimator instance
                else:
                    raise TypeError("'ensemble' must be an ensemble estimator object")
            # If n_estimators value is not valid
            else:
                raise ValueError(
                    "'n_estimators' must less or equal to 'len(ensemble.estimators_)'"
                )
        # If n_estimators value is not int
        else:
            raise TypeError("'n_estimators' must be of type `int`")

    def _collect_ensemble_performance_data(
        self, ensemble_name, ensemble, max_len_ensemble_name, max_len_dataset_name
    ):
        """Private function for collecting performance data of a single ensemble model."""

        # Set local variables
        granularity = self.granularity_
        eval_metrics = self.eval_metrics_
        classes_ = ensemble.classes_
        max_n_estimators = len(ensemble.estimators_)
        estimators_n_training_samples_ = self.ensembles_n_training_samples_[
            ensemble_name
        ]
        has_n_training_samples_ = self.ensembles_has_n_training_samples_[ensemble_name]

        results = []
        # For each evaluation dataset
        for dataset_name, (X_eval, y_eval) in self.eval_datasets_.items():
            # Visualizer fitting verbose
            iterations = tqdm(range(len(ensemble.estimators_)))
            # Format the verbose with max_len_ensemble_name & max_len_dataset_name
            iterations.set_description(
                "Visualizer evaluating model {:^{l_e}s} on dataset {:^{l_d}s} :".format(
                    ensemble_name,
                    dataset_name,
                    l_e=max_len_ensemble_name,
                    l_d=max_len_dataset_name,
                )
            )
            # Collect performance data per ``granularity`` base estimators
            for i in iterations:
                if (i + 1) % granularity == 0 or i == 0 or (i + 1) == max_n_estimators:
                    curr_y_pred_proba = self._get_predict_proba(X_eval, ensemble, i)
                    # Set number of training samples
                    if has_n_training_samples_:
                        n_samples = sum(estimators_n_training_samples_[: i + 1])
                    # If cannot fetch n_training_samples_ array from the ensemble
                    else:
                        n_samples = None
                    for metric_name, (
                        metric_func,
                        kwargs,
                        ac_proba,
                        ac_labels,
                    ) in eval_metrics.items():
                        if ac_labels:
                            kwargs["labels"] = classes_
                        if ac_proba:  # If the metric take predict probabilities
                            score = metric_func(y_eval, curr_y_pred_proba, **kwargs)
                        else:  # If the metric do not take predict probabilities
                            curr_y_pred = classes_.take(
                                np.argmax(curr_y_pred_proba, axis=1), axis=0
                            )
                            score = metric_func(y_eval, curr_y_pred, **kwargs)
                        results.append(
                            [
                                i + 1,
                                ensemble_name,
                                dataset_name,
                                metric_name,
                                score,
                                n_samples,
                            ]
                        )

        return pd.DataFrame(results, columns=DATAFRAME_COLUMNS)

    def _collect_all_ensemble_performance_data(self):
        """Private function for collecting performance data of all ensemble models."""

        # max_len_xxx_name was used to format the verbose
        max_len_ensemble_name = max([len(_) for _ in self.ensembles_.keys()])
        max_len_dataset_name = max([len(_) for _ in self.vis_format_["dataset_names"]])

        return pd.concat(
            [
                self._collect_ensemble_performance_data(
                    ensemble_name, ensemble, max_len_ensemble_name, max_len_dataset_name
                )
                for ensemble_name, ensemble in self.ensembles_.items()
            ]
        )

    def _split_ensembles_by_has_n_training_samples_(self, on_ensembles_):
        """Private function spliting ensemble names by have/not have
        the array of numbers of training samples of each base estimator.
        """

        ensembles_has_n_training_samples = self.ensembles_has_n_training_samples_
        positives, negatives = [], []
        for ensemble_name in on_ensembles_:
            if ensembles_has_n_training_samples[ensemble_name]:
                positives.append(ensemble_name)
            else:
                negatives.append(ensemble_name)

        return tuple(positives), tuple(negatives)

    def performance_lineplot(
        self,
        on_ensembles: list = None,
        on_datasets: list = None,
        on_metrics: list = None,
        split_by: list = [],
        n_samples_as_x_axis: bool = False,
        sub_figsize: tuple = (4.0, 3.3),
        sup_title: bool or str = True,
        **lineplot_kwargs,
    ):
        """Draw a performance line plot.

        Parameters
        ----------
        on_ensembles : list of strings, default=None
            The names of ensembles to include in the plot. It should be a
            subset of ``self.ensembles_.keys()``. if ``None``, all ensembles
            fitted by the visualizer will be included.

        on_datasets : list of strings, default=None
            The names of evaluation datasets to include in the plot. It
            should be a subset of ``self.eval_datasets_.keys()``. if ``None``,
            all evaluation datasets will be included.

        on_metrics : list of strings, default=None
            The names of evaluation metrics to include in the plot. It
            should be a subset of ``self.eval_metrics_.keys()``. if ``None``,
            all evaluation metrics will be included.

        split_by : list of {'method', 'dataset'}, default=[]
            How to group the results for visualization.

            - if contains ``'method'``, the performance results of different
              ensemble methods will be displayed in independent sub-figures.
            - if contains ``'dataset'``, the performance results on different
              evaluation datasets will be displayed in independent sub-figures.

        n_samples_as_x_axis : bool, default=False
            Whether to use the number of training samples as the x-axis.

        sub_figsize: (float, float), default=(4.0, 3.3)
            The size of an subfigure (width, height in inches).
            The overall figure size will be automatically determined by
            (sub_figsize[0] * num_columns, sub_figsize[1] * num_rows).

        sup_title: bool or str, default=True
            The super title of the figure.

            - if ``True``, automatically determines the super title.
            - if ``False``, no super title will be displayed.
            - if ``string``, super title will be ``sup_title``.

        **lineplot_kwargs : key, value mappings
            Other keyword arguments are passed down to
            :meth:`seaborn.lineplot`.

        Returns
        -------
        self : object
        """

        vis_perf_dataframe = self.perf_dataframe_.copy()

        # Check if the visualizer is fitted
        if not self._fitted:
            raise NotFittedError(
                f"This visualizer is not fitted yet."
                f" Call 'fit' with appropriate arguments before calling"
                f" 'performance_lineplot'."
            )

        # Check parameters
        on_ensembles = self._check_is_subset(
            on_ensembles, "on_ensembles", self.vis_format_["ensemble_names"]
        )
        on_datasets = self._check_is_subset(
            on_datasets, "on_datasets", self.vis_format_["dataset_names"]
        )
        on_metrics = self._check_is_subset(
            on_metrics, "on_metrics", self.vis_format_["metric_names"]
        )
        n_ensembles, n_datasets, n_metrics = (
            len(on_ensembles),
            len(on_datasets),
            len(on_metrics),
        )

        if not isinstance(split_by, list):
            raise TypeError(
                f"'split_by' should be a `list` of `string`, got {type(split_by)}."
                f" Elements should be one of {SPLIT_BY}."
            )
        check_has_diff_elements(
            split_by,
            SPLIT_BY,
            msg=f"Got unsupported value %(diff_set)s in 'split_by'."
            f" Elements should be one of {SPLIT_BY}.",
        )

        n_samples_as_x_axis_ = check_type(
            n_samples_as_x_axis, "n_samples_as_x_axis_", bool
        )

        (sub_fig_width, sub_fig_height) = check_plot_figsize(sub_figsize)

        sup_title_ = check_type(sup_title, "sup_title", (bool, str))

        lineplot_kwargs_ = copy(LINEPLOT_KWARGS_DEFAULT)
        lineplot_kwargs_.update(lineplot_kwargs)
        for kw in ["ax", "data", "x", "y", "hue", "style"]:
            if kw in lineplot_kwargs_.keys():
                raise ValueError(
                    f"Cannot set parameter '{kw}' for"
                    f" performance_lineplot function."
                )

        # Select data for visualization
        on_ensembles_mask = vis_perf_dataframe["method"].map(
            {k: k in on_ensembles for k in self.vis_format_["ensemble_names"]}
        )
        on_datasets_mask = vis_perf_dataframe["dataset"].map(
            {k: k in on_datasets for k in self.vis_format_["dataset_names"]}
        )
        on_metrics_mask = vis_perf_dataframe["metric"].map(
            {k: k in on_metrics for k in self.vis_format_["metric_names"]}
        )
        final_mask = on_ensembles_mask & on_datasets_mask & on_metrics_mask
        vis_perf_dataframe = vis_perf_dataframe.loc[final_mask]

        # Set subfigure x-axis (# estimators or # training samples)
        if n_samples_as_x_axis_:
            (
                include_ensembles,
                exclude_ensembles,
            ) = self._split_ensembles_by_has_n_training_samples_(on_ensembles)
            # If any estimator does not have `estimators_n_training_samples_` attribute
            if len(exclude_ensembles) > 0:
                # Raise a warning of excluding some methods from visualization
                warn(
                    f"scikit-learn ensemble estimator(s) with name"
                    + f" {exclude_ensembles} does not record the number"
                    + f" training samples, they will be excluded from the"
                    + f" figure when `n_samples_as_x_axis` = `True`."
                )
                # Update the number of included ensemble methods
                n_ensembles = len(include_ensembles)
                # Select data
                has_n_samples_mask = vis_perf_dataframe["method"].map(
                    self.ensembles_has_n_training_samples_
                )
                vis_perf_dataframe = vis_perf_dataframe.loc[has_n_samples_mask]
            # Set x-axis to number of samples
            x_column = "n_samples"
            x_label = "# Training Samples"
        else:
            # Set x-axis to number of base estimators
            x_column = "n_estimators"
            x_label = "# Base Estimators"

        # Set figure size and layout
        n_rows_fig, n_columns_fig = 1, n_metrics
        if "method" in split_by:
            n_rows_fig *= n_ensembles
        if "dataset" in split_by:
            n_rows_fig *= n_datasets

        total_width, total_height = (
            sub_fig_width * n_columns_fig,
            sub_fig_height * n_rows_fig,
        )
        # If has sup_title, add reserved space
        total_height += RESERVED_SUPTITLE_INCHES if sup_title_ else 0
        figsize = (total_width, total_height)
        fig, axes = plt.subplots(n_rows_fig, n_columns_fig, figsize=figsize)
        axes = np.array(axes).reshape(n_rows_fig, n_columns_fig)

        # Set column titles
        pad = 10
        col_titles = ["Metric: <{}>".format(metric) for metric in on_metrics]
        for ax, col_title in zip(axes[0], col_titles):
            ax.annotate(
                col_title,
                xy=(0.5, 1),
                xytext=(0, pad),
                xycoords="axes fraction",
                textcoords="offset points",
                ha="center",
                va="baseline",
                **self.row_col_title_style,
            )

        # Plot performances on each ax
        vis_df_grp = vis_perf_dataframe.groupby(by=split_by + ["metric"])
        for (key, _), ax in zip(vis_df_grp.groups.items(), axes.flatten()):
            metric_name = key if len(split_by) == 0 else key[-1]
            # Use seaborn.lineplot for visualization
            if isinstance(key, str):
                key = (key,)
            kwargs = {
                "data": vis_df_grp.get_group(key).reset_index(drop=True),
                "x": x_column,
                "y": "score",
                "hue": "method",
                "style": "dataset",
                "ax": ax,
            }
            kwargs.update(lineplot_kwargs_)
            ax = sns.lineplot(**kwargs)
            # Set legend, border, x_label, y_label, and grid properties
            ax.legend(
                columnspacing=0.3,
                borderaxespad=0.3,
                handletextpad=0.3,
                labelspacing=0.1,
                handlelength=None,
                borderpad=0.4,
            )
            ax = set_ax_border(ax, border_color="black", border_width=2)
            ax.set_xlabel(x_label, **self.axis_title_style)
            ax.set_ylabel(f"{metric_name}", **self.axis_title_style)
            ax.grid(color="black", linestyle="-.", alpha=0.3)

        # Use tight layout
        height_rect = (total_height - RESERVED_SUPTITLE_INCHES) / total_height
        plt.tight_layout(rect=(0, 0, 1, height_rect))

        # Set super title
        if sup_title_ == True:
            if n_rows_fig == n_columns_fig == 1:
                fig.suptitle("Performance Curve", **self.suptitle_style)
            else:
                fig.suptitle("Performance Curves", **self.suptitle_style)
        elif sup_title_ == False:
            pass
        else:
            fig.suptitle(sup_title_, **self.suptitle_style)

        return fig, axes

    def _collect_all_ensemble_confusion_matrix(
        self,
    ):
        """Private function for collecting confusion matrices of all ensemble models."""

        print("Visualizer computing confusion matrices", end="")
        conf_matrices = {}
        for ensemble_name, ensemble in self.ensembles_.items():
            conf_matrices_ensemble = {}
            # Check whether the ensemble has verbose attribute
            has_verbose = hasattr(ensemble, "verbose")
            # Temporarily disable built-in verbose
            if has_verbose:
                tmp_verbose, ensemble.verbose = ensemble.verbose, 0
            for dataset_name, (X_eval, y_eval) in self.eval_datasets_.items():
                print(".", end="")
                classes = ensemble.classes_
                y_pred = ensemble.predict(X_eval)
                conf_matrices_ensemble[dataset_name] = pd.DataFrame(
                    confusion_matrix(y_eval, y_pred, labels=classes),
                    columns=classes,
                    index=classes,
                )
            # Recover verbose attribute
            if has_verbose:
                ensemble.verbose = tmp_verbose
            conf_matrices[ensemble_name] = conf_matrices_ensemble
        print(" Finished!")

        return conf_matrices

    @staticmethod
    def _check_is_subset(param: list, param_name: str, universal_set: list):
        """Private function to check whether param is a subset of universal_set."""

        if param is None:
            return copy(universal_set)
        elif isinstance(param, list):
            check_has_diff_elements(
                param,
                universal_set,
                msg=f"Got unsupported value %(diff_set)s in '{param_name}'."
                f" The possible values are {universal_set} or a list of them.",
            )
            return param
        else:
            raise TypeError(
                f"'{param_name}' should be a `list` of `string`,"
                f" got {type(param)}."
                f" The possible values are {universal_set}."
            )

    def confusion_matrix_heatmap(
        self,
        on_ensembles: list = None,
        on_datasets: list = None,
        false_pred_only: bool = False,
        sub_figsize: tuple = (4.0, 3.3),
        sup_title: bool or str = True,
        **heatmap_kwargs,
    ):
        """Draw a confusion matrix heatmap.

        Parameters
        ----------
        on_ensembles : list of strings, default=None
            The names of ensembles to include in the plot. It should be a
            subset of ``self.ensembles_.keys()``. if ``None``, all ensembles
            fitted by the visualizer will be included.

        on_datasets : list of strings, default=None
            The names of evaluation datasets to include in the plot. It
            should be a subset of ``self.eval_datasets_.keys()``. if ``None``,
            all evaluation datasets will be included.

        false_pred_only : bool, default=False
            Whether to plot only the false predictions in the confusion matrix.
            if ``True``, only the numbers of false predictions will be shown
            in the plot.

        sub_figsize: (float, float), default=(4.0, 3.3)
            The size of an subfigure (width, height in inches).
            The overall figure size will be automatically determined by
            (sub_figsize[0] * num_columns, sub_figsize[1] * num_rows).

        sup_title: bool or str, default=True
            The super title of the figure.

            - if ``True``, automatically determines the super title.
            - if ``False``, no super title will be displayed.
            - if ``string``, super title will be ``sup_title``.

        **heatmap_kwargs : key, value mappings
            Other keyword arguments are passed down to
            :meth:`seaborn.heatmap`.

        Returns
        -------
        self : object
        """
        # Check parameters
        if not self._fitted:
            raise NotFittedError(
                f"This visualizer is not fitted yet."
                f" Call 'fit' with appropriate arguments before calling"
                f" 'confusion_matrix_heatmap'."
            )

        on_ensembles = self._check_is_subset(
            on_ensembles, "on_ensembles", self.vis_format_["ensemble_names"]
        )
        on_datasets = self._check_is_subset(
            on_datasets, "on_datasets", self.vis_format_["dataset_names"]
        )
        n_ensembles, n_datasets = len(on_ensembles), len(on_datasets)

        false_pred_only = check_type(false_pred_only, "false_pred_only", bool)

        (sub_fig_width, sub_fig_height) = check_plot_figsize(sub_figsize)

        sup_title_ = check_type(sup_title, "sup_title", (bool, str))

        heatmap_kwargs_ = copy(HEATMAP_KWARGS_DEFAULT)
        heatmap_kwargs_.update(heatmap_kwargs)
        for kw in ["ax", "data"]:
            if kw in heatmap_kwargs_.keys():
                raise ValueError(
                    f"Cannot set parameter '{kw}' for"
                    f" confusion_matrix_heatmap function."
                )

        # Set figure size and layout
        n_rows_fig, n_columns_fig = n_datasets, n_ensembles
        total_width, total_height = (
            sub_fig_width * n_columns_fig,
            sub_fig_height * n_rows_fig,
        )
        # If has sup_title, add reserved space
        total_height += RESERVED_SUPTITLE_INCHES if sup_title_ else 0
        figsize = (total_width, total_height)
        fig, axes = plt.subplots(n_rows_fig, n_columns_fig, figsize=figsize)
        axes = np.array(axes).reshape(n_rows_fig, n_columns_fig)

        # Set titles for each column and row
        pad = 10
        row_titles = ["On dataset: <{}>".format(col) for col in on_datasets]
        col_titles = ["Method: <{}>".format(row) for row in on_ensembles]
        # Set column titles
        for ax, col_title in zip(axes[0], col_titles):
            ax.annotate(
                col_title,
                xy=(0.5, 1),
                xytext=(0, pad),
                xycoords="axes fraction",
                textcoords="offset points",
                ha="center",
                va="baseline",
                **self.row_col_title_style,
            )

        # Set row titles
        for ax, row_title in zip(axes[:, 0], row_titles):
            ax.annotate(
                row_title,
                xy=(0, 0.5),
                xytext=(-ax.yaxis.labelpad - pad, 0),
                xycoords=ax.yaxis.label,
                textcoords="offset points",
                ha="right",
                va="center",
                rotation=90,
                **self.row_col_title_style,
            )

        # Plot confusion matrix heatmap on each ax
        conf_matrices = self.conf_matrices_
        for dataset_name, i_row in zip(on_datasets, range(n_rows_fig)):
            for ensemble_name, i_col in zip(on_ensembles, range(n_columns_fig)):
                ax = axes[i_row, i_col]
                conf_matrix_df = conf_matrices[ensemble_name][dataset_name]
                # Use seaborn.heatmap for visualization
                kwargs = {
                    "data": conf_matrix_df,
                    "annot": True,
                    "fmt": "d",
                    "linewidths": 0.5,
                    "ax": ax,
                }
                kwargs.update(heatmap_kwargs_)
                # if false_pred_only, set mask to the heatmap
                if false_pred_only:
                    kwargs["mask"] = np.identity(conf_matrix_df.shape[0])
                ax = sns.heatmap(**kwargs)
                # Set x_label and y_label properties
                ax.set_xlabel("Predicted Label", **self.axis_title_style)
                ax.set_ylabel("Ground Truth", **self.axis_title_style)

        # Use tight layout
        height_rect = (total_height - RESERVED_SUPTITLE_INCHES * 1.3) / total_height
        plt.tight_layout(rect=(0, 0, 1, height_rect))

        # Set super title
        if sup_title_ == True:
            if n_rows_fig == n_columns_fig == 1:
                fig.suptitle("Confusion Matrix", **self.suptitle_style)
            else:
                fig.suptitle("Confusion Matrices", **self.suptitle_style)
        elif sup_title_ == False:
            pass
        else:
            fig.suptitle(sup_title_, **self.suptitle_style)

        return fig, axes
