# coding: utf-8
"""Metrics to assess performance on a classification task given class
predictions. The available metrics are complementary from the metrics available
in scikit-learn.

Functions named as ``*_score`` return a scalar value to maximize: the higher
the better

Function named as ``*_error`` or ``*_loss`` return a scalar value to minimize:
the lower the better
"""
# Adapted from imbalanced-learn

# Authors: Guillaume Lemaitre
#          Dariusz Brzezinski
# License: MIT

LOCAL_DEBUG = False

if not LOCAL_DEBUG:
    from ..utils._validation import _deprecate_positional_args
else:  # pragma: no cover
    import sys  # For local test

    sys.path.append("..")
    from utils._validation import _deprecate_positional_args

import functools
import warnings
from inspect import signature

import numpy as np
import scipy as sp
from sklearn.metrics import mean_absolute_error, precision_recall_fscore_support
from sklearn.metrics._classification import _check_targets, _prf_divide
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_consistent_length, column_or_1d

_AVERAGE_OPTIONS = (None, "micro", "macro", "weighted", "samples")


def _check_average_validity(average, average_options):
    if average not in average_options and average != "binary":
        raise ValueError("average has to be one of " + str(average_options))


def _handle_binary_average(average, y_type, pos_label, present_labels, labels):
    if average == "binary":
        if y_type == "binary":
            if pos_label not in present_labels:
                if len(present_labels) < 2:
                    return True, (0.0, 0.0, 0), None
                raise ValueError(
                    "pos_label=%r is not a valid label: %r"
                    % (pos_label, present_labels)
                )
            return False, None, [pos_label]
        raise ValueError(
            "Target is %s but average='binary'. Please "
            "choose another average setting." % y_type
        )
    elif pos_label not in (None, 1):
        warnings.warn(
            "Note that pos_label (set to %r) is ignored when "
            "average != 'binary' (got %r). You may use "
            "labels=[pos_label] to specify a single positive class."
            % (pos_label, average),
            UserWarning,
        )
    return False, None, labels


def _resolve_labels(labels, present_labels):
    if labels is None:
        return present_labels, None
    n_labels = len(labels)
    labels = np.hstack(
        [labels, np.setdiff1d(present_labels, labels, assume_unique=True)]
    )
    return labels, n_labels


def _compute_confusion_statistics(
    y_true, y_pred, labels, n_labels, y_type, average, sample_weight
):
    if y_type.startswith("multilabel"):
        raise ValueError("imblearn does not support multilabel")
    elif average == "samples":
        raise ValueError(
            "Sample-based precision, recall, fscore is "
            "not meaningful outside multilabel "
            "classification. See the accuracy_score instead."
        )

    le = LabelEncoder()
    le.fit(labels)
    y_true = le.transform(y_true)
    y_pred = le.transform(y_pred)
    sorted_labels = le.classes_

    tp = y_true == y_pred
    tp_bins = y_true[tp]
    if sample_weight is not None:
        tp_bins_weights = np.asarray(sample_weight)[tp]
    else:
        tp_bins_weights = None

    if len(tp_bins):
        tp_sum = np.bincount(
            tp_bins, weights=tp_bins_weights, minlength=len(labels)
        )
    else:
        true_sum = pred_sum = tp_sum = np.zeros(len(labels))
    if len(y_pred):
        pred_sum = np.bincount(y_pred, weights=sample_weight, minlength=len(labels))
    if len(y_true):
        true_sum = np.bincount(y_true, weights=sample_weight, minlength=len(labels))

    tn_sum = y_true.size - (pred_sum + true_sum - tp_sum)

    indices = np.searchsorted(sorted_labels, labels[:n_labels])
    tp_sum = tp_sum[indices]
    true_sum = true_sum[indices]
    pred_sum = pred_sum[indices]
    tn_sum = tn_sum[indices]

    return tp_sum, pred_sum, true_sum, tn_sum


def _apply_micro_average(tp_sum, pred_sum, true_sum, tn_sum, average):
    if average == "micro":
        tp_sum = np.array([tp_sum.sum()])
        pred_sum = np.array([pred_sum.sum()])
        true_sum = np.array([true_sum.sum()])
        tn_sum = np.array([tn_sum.sum()])
    return tp_sum, pred_sum, true_sum, tn_sum


def _compute_final_average(
    sensitivity, specificity, average, true_sum, sample_weight
):
    if average == "weighted":
        weights = true_sum
        if weights.sum() == 0:
            return 0, 0, None
    elif average == "samples":
        weights = sample_weight
    else:
        weights = None

    if average is not None:
        assert average != "binary" or len(specificity) == 1
        specificity = np.average(specificity, weights=weights)
        sensitivity = np.average(sensitivity, weights=weights)
        true_sum = None

    return sensitivity, specificity, true_sum


def _compute_sensitivity_specificity(tn_sum, pred_sum, tp_sum, true_sum, average, warn_for):
    with np.errstate(divide="ignore", invalid="ignore"):
        specificity = _prf_divide(
            tn_sum,
            tn_sum + pred_sum - tp_sum,
            "specificity",
            "predicted",
            average,
            warn_for,
        )
        sensitivity = _prf_divide(
            tp_sum, true_sum, "sensitivity", "true", average, warn_for
        )
    return sensitivity, specificity


@_deprecate_positional_args
def sensitivity_specificity_support(
    y_true,
    y_pred,
    *,
    labels=None,
    pos_label=1,
    average=None,
    warn_for=("sensitivity", "specificity"),
    sample_weight=None,
):
    """Compute sensitivity, specificity, and support for each class.

    The sensitivity is the ratio ``tp / (tp + fn)`` and the specificity
    is the ratio ``tn / (tn + fp)``.

    Parameters
    ----------
    y_true : ndarray of shape (n_samples,)
        Ground truth target values.
    y_pred : ndarray of shape (n_samples,)
        Estimated targets as returned by a classifier.
    labels : list, default=None
        The set of labels to include when ``average != 'binary'``.
    pos_label : str or int, default=1
        The class to report if ``average='binary'`` and the data is binary.
    average : str, default=None
        Averaging method: ``'binary'``, ``'micro'``, ``'macro'``,
        ``'weighted'``, ``'samples'``, or ``None``.
    warn_for : tuple, default=("sensitivity", "specificity")
        Determines which warnings will be made.
    sample_weight : ndarray of shape (n_samples,), default=None
        Sample weights.

    Returns
    -------
    sensitivity : float or ndarray
    specificity : float or ndarray
    support : int or ndarray

    References
    ----------
    .. [1] Wikipedia entry for Sensitivity and specificity.
    .. [2] https://imbalanced-learn.org/stable/metrics.html
    """
    _check_average_validity(average, _AVERAGE_OPTIONS)

    y_type, y_true, y_pred = _check_targets(y_true, y_pred)
    present_labels = unique_labels(y_true, y_pred)

    early_return, result, labels = _handle_binary_average(
        average, y_type, pos_label, present_labels, labels
    )
    if early_return:
        return result

    labels, n_labels = _resolve_labels(labels, present_labels)

    tp_sum, pred_sum, true_sum, tn_sum = _compute_confusion_statistics(
        y_true, y_pred, labels, n_labels, y_type, average, sample_weight
    )

    tp_sum, pred_sum, true_sum, tn_sum = _apply_micro_average(
        tp_sum, pred_sum, true_sum, tn_sum, average
    )

    sensitivity, specificity = _compute_sensitivity_specificity(
        tn_sum, pred_sum, tp_sum, true_sum, average, warn_for
    )

    sensitivity, specificity, true_sum = _compute_final_average(
        sensitivity, specificity, average, true_sum, sample_weight
    )

    return sensitivity, specificity, true_sum


@_deprecate_positional_args
def sensitivity_score(
    y_true,
    y_pred,
    *,
    labels=None,
    pos_label=1,
    average="binary",
    sample_weight=None,
):
    """Compute the sensitivity

    The sensitivity is the ratio ``tp / (tp + fn)`` where ``tp`` is the number
    of true positives and ``fn`` the number of false negatives. The sensitivity
    quantifies the ability to avoid false negatives.

    The best value is 1 and the worst value is 0.

    Read more in the `User Guide <https://imbalanced-learn.org/stable/metrics.html#sensitivity-specificity>`_.

    Parameters
    ----------
    y_true : ndarray of shape (n_samples,)
        Ground truth (correct) target values.

    y_pred : ndarray of shape (n_samples,)
        Estimated targets as returned by a classifier.

    labels : list, default=None
        The set of labels to include when ``average != 'binary'``, and their
        order if ``average is None``. Labels present in the data can be
        excluded, for example to calculate a multiclass average ignoring a
        majority negative class, while labels not present in the data will
        result in 0 components in a macro average.

    pos_label : str or int, default=1
        The class to report if ``average='binary'`` and the data is binary.
        If the data are multiclass, this will be ignored;
        setting ``labels=[pos_label]`` and ``average != 'binary'`` will report
        scores for that label only.

    average : str, default=None
        If ``None``, the scores for each class are returned. Otherwise, this
        determines the type of averaging performed on the data:

        ``'binary'``:
            Only report results for the class specified by ``pos_label``.
            This is applicable only if targets (``y_{true,pred}``) are binary.
        ``'micro'``:
            Calculate metrics globally by counting the total true positives,
            false negatives and false positives.
        ``'macro'``:
            Calculate metrics for each label, and find their unweighted
            mean.  This does not take label imbalance into account.
        ``'weighted'``:
            Calculate metrics for each label, and find their average, weighted
            by support (the number of true instances for each label). This
            alters 'macro' to account for label imbalance; it can result in an
            F-score that is not between precision and recall.
        ``'samples'``:
            Calculate metrics for each instance, and find their average (only
            meaningful for multilabel classification where this differs from
            :func:`accuracy_score`).

    sample_weight : ndarray of shape (n_samples,), default=None
        Sample weights.

    Returns
    -------
    specificity : float (if `average is None`) or ndarray of \
            shape (n_unique_labels,)
        The specifcity metric.

    Examples
    --------
    >>> import numpy as np
    >>> from imbens.metrics import sensitivity_score
    >>> y_true = [0, 1, 2, 0, 1, 2]
    >>> y_pred = [0, 2, 1, 0, 0, 1]
    >>> sensitivity_score(y_true, y_pred, average='macro')
    0.33333333333333331
    >>> sensitivity_score(y_true, y_pred, average='micro')
    0.33333333333333331
    >>> sensitivity_score(y_true, y_pred, average='weighted')
    0.33333333333333331
    >>> sensitivity_score(y_true, y_pred, average=None)
    array([ 1.,  0.,  0.])
    """
    s, _, _ = sensitivity_specificity_support(
        y_true,
        y_pred,
        labels=labels,
        pos_label=pos_label,
        average=average,
        warn_for=("sensitivity",),
        sample_weight=sample_weight,
    )

    return s


@_deprecate_positional_args
def specificity_score(
    y_true,
    y_pred,
    *,
    labels=None,
    pos_label=1,
    average="binary",
    sample_weight=None,
):
    """Compute the specificity

    The specificity is the ratio ``tn / (tn + fp)`` where ``tn`` is the number
    of true negatives and ``fp`` the number of false positives. The specificity
    quantifies the ability to avoid false positives.

    The best value is 1 and the worst value is 0.

    Read more in the `User Guide <https://imbalanced-learn.org/stable/metrics.html#sensitivity-specificity>`_.

    Parameters
    ----------
    y_true : ndarray of shape (n_samples,)
        Ground truth (correct) target values.

    y_pred : ndarray of shape (n_samples,)
        Estimated targets as returned by a classifier.

    labels : list, default=None
        The set of labels to include when ``average != 'binary'``, and their
        order if ``average is None``. Labels present in the data can be
        excluded, for example to calculate a multiclass average ignoring a
        majority negative class, while labels not present in the data will
        result in 0 components in a macro average.

    pos_label : str or int, default=1
        The class to report if ``average='binary'`` and the data is binary.
        If the data are multiclass, this will be ignored;
        setting ``labels=[pos_label]`` and ``average != 'binary'`` will report
        scores for that label only.

    average : str, default=None
        If ``None``, the scores for each class are returned. Otherwise, this
        determines the type of averaging performed on the data:

        ``'binary'``:
            Only report results for the class specified by ``pos_label``.
            This is applicable only if targets (``y_{true,pred}``) are binary.
        ``'micro'``:
            Calculate metrics globally by counting the total true positives,
            false negatives and false positives.
        ``'macro'``:
            Calculate metrics for each label, and find their unweighted
            mean.  This does not take label imbalance into account.
        ``'weighted'``:
            Calculate metrics for each label, and find their average, weighted
            by support (the number of true instances for each label). This
            alters 'macro' to account for label imbalance; it can result in an
            F-score that is not between precision and recall.
        ``'samples'``:
            Calculate metrics for each instance, and find their average (only
            meaningful for multilabel classification where this differs from
            :func:`accuracy_score`).

    sample_weight : ndarray of shape (n_samples,), default=None
        Sample weights.

    Returns
    -------
    specificity : float (if `average is None`) or ndarray of \
            shape (n_unique_labels,)
        The specificity metric.

    Examples
    --------
    >>> import numpy as np
    >>> from imbens.metrics import specificity_score
    >>> y_true = [0, 1, 2, 0, 1, 2]
    >>> y_pred = [0, 2, 1, 0, 0, 1]
    >>> specificity_score(y_true, y_pred, average='macro')
    0.66666666666666663
    >>> specificity_score(y_true, y_pred, average='micro')
    0.66666666666666663
    >>> specificity_score(y_true, y_pred, average='weighted')
    0.66666666666666663
    >>> specificity_score(y_true, y_pred, average=None)
    array([ 0.75,  0.5 ,  0.75])
    """
    _, s, _ = sensitivity_specificity_support(
        y_true,
        y_pred,
        labels=labels,
        pos_label=pos_label,
        average=average,
        warn_for=("specificity",),
        sample_weight=sample_weight,
    )

    return s


@_deprecate_positional_args
def geometric_mean_score(
    y_true,
    y_pred,
    *,
    labels=None,
    pos_label=1,
    average="multiclass",
    sample_weight=None,
    correction=0.0,
):
    """Compute the geometric mean.

    The geometric mean (G-mean) is the root of the product of class-wise
    sensitivity. This measure tries to maximize the accuracy on each of the
    classes while keeping these accuracies balanced. For binary classification
    G-mean is the squared root of the product of the sensitivity
    and specificity. For multi-class problems it is a higher root of the
    product of sensitivity for each class.

    For compatibility with other imbalance performance measures, G-mean can be
    calculated for each class separately on a one-vs-rest basis when
    ``average != 'multiclass'``.

    The best value is 1 and the worst value is 0. Traditionally if at least one
    class is unrecognized by the classifier, G-mean resolves to zero. To
    alleviate this property, for highly multi-class the sensitivity of
    unrecognized classes can be "corrected" to be a user specified value
    (instead of zero). This option works only if ``average == 'multiclass'``.

    Read more in the `User Guide <https://imbalanced-learn.org/stable/metrics.html#imbalanced-metrics>`_.

    Parameters
    ----------
    y_true : ndarray of shape (n_samples,)
        Ground truth (correct) target values.

    y_pred : ndarray of shape (n_samples,)
        Estimated targets as returned by a classifier.

    labels : list, default=None
        The set of labels to include when ``average != 'binary'``, and their
        order if ``average is None``. Labels present in the data can be
        excluded, for example to calculate a multiclass average ignoring a
        majority negative class, while labels not present in the data will
        result in 0 components in a macro average.

    pos_label : str or int, default=1
        The class to report if ``average='binary'`` and the data is binary.
        If the data are multiclass, this will be ignored;
        setting ``labels=[pos_label]`` and ``average != 'binary'`` will report
        scores for that label only.

    average : str or None, default='multiclass'
        If ``None``, the scores for each class are returned. Otherwise, this
        determines the type of averaging performed on the data:

        ``'binary'``:
            Only report results for the class specified by ``pos_label``.
            This is applicable only if targets (``y_{true,pred}``) are binary.
        ``'micro'``:
            Calculate metrics globally by counting the total true positives,
            false negatives and false positives.
        ``'macro'``:
            Calculate metrics for each label, and find their unweighted
            mean.  This does not take label imbalance into account.
        ``'weighted'``:
            Calculate metrics for each label, and find their average, weighted
            by support (the number of true instances for each label). This
            alters 'macro' to account for label imbalance; it can result in an
            F-score that is not between precision and recall.
        ``'samples'``:
            Calculate metrics for each instance, and find their average (only
            meaningful for multilabel classification where this differs from
            :func:`accuracy_score`).

    sample_weight : ndarray of shape (n_samples,), default=None
        Sample weights.

    correction: float, default=0.0
        Substitutes sensitivity of unrecognized classes from zero to a given
        value.

    Returns
    -------
    geometric_mean : float

    Notes
    -----
    See :ref:`sphx_glr_auto_examples_evaluation_plot_metrics.py` for an example.

    References
    ----------
    .. [1] Kubat, M. and Matwin, S. "Addressing the curse of
       imbalanced training sets: one-sided selection" ICML (1997)

    .. [2] Barandela, R., Sánchez, J. S., Garcıa, V., & Rangel, E. "Strategies
       for learning in class imbalance problems", Pattern Recognition,
       36(3), (2003), pp 849-851.

    Examples
    --------
    >>> from imbens.metrics import geometric_mean_score
    >>> y_true = [0, 1, 2, 0, 1, 2]
    >>> y_pred = [0, 2, 1, 0, 0, 1]
    >>> geometric_mean_score(y_true, y_pred)
    0.0
    >>> geometric_mean_score(y_true, y_pred, correction=0.001)
    0.010000000000000004
    >>> geometric_mean_score(y_true, y_pred, average='macro')
    0.47140452079103168
    >>> geometric_mean_score(y_true, y_pred, average='micro')
    0.47140452079103168
    >>> geometric_mean_score(y_true, y_pred, average='weighted')
    0.47140452079103168
    >>> geometric_mean_score(y_true, y_pred, average=None)
    array([ 0.8660254,  0.       ,  0.       ])
    """
    if average is None or average != "multiclass":
        sen, spe, _ = sensitivity_specificity_support(
            y_true,
            y_pred,
            labels=labels,
            pos_label=pos_label,
            average=average,
            warn_for=("specificity", "specificity"),
            sample_weight=sample_weight,
        )

        return np.sqrt(sen * spe)
    else:
        present_labels = unique_labels(y_true, y_pred)

        if labels is None:
            labels = present_labels
            n_labels = None
        else:
            n_labels = len(labels)
            labels = np.hstack(
                [labels, np.setdiff1d(present_labels, labels, assume_unique=True)]
            )

        le = LabelEncoder()
        le.fit(labels)
        y_true = le.transform(y_true)
        y_pred = le.transform(y_pred)
        sorted_labels = le.classes_

        # labels are now from 0 to len(labels) - 1 -> use bincount
        tp = y_true == y_pred
        tp_bins = y_true[tp]

        if sample_weight is not None:
            tp_bins_weights = np.asarray(sample_weight)[tp]
        else:
            tp_bins_weights = None

        if len(tp_bins):
            tp_sum = np.bincount(
                tp_bins, weights=tp_bins_weights, minlength=len(labels)
            )
        else:
            # Pathological case
            true_sum = tp_sum = np.zeros(len(labels))
        if len(y_true):
            true_sum = np.bincount(y_true, weights=sample_weight, minlength=len(labels))

        # Retain only selected labels
        indices = np.searchsorted(sorted_labels, labels[:n_labels])
        tp_sum = tp_sum[indices]
        true_sum = true_sum[indices]

        with np.errstate(divide="ignore", invalid="ignore"):
            recall = _prf_divide(tp_sum, true_sum, "recall", "true", None, "recall")
        recall[recall == 0] = correction

        with np.errstate(divide="ignore", invalid="ignore"):
            gmean = sp.stats.gmean(recall)
        # old version of scipy return MaskedConstant instead of 0.0
        if isinstance(gmean, np.ma.core.MaskedConstant):
            return 0.0
        return gmean


@_deprecate_positional_args
def make_index_balanced_accuracy(*, alpha=0.1, squared=True):
    """Balance any scoring function using the index balanced accuracy

    This factory function wraps scoring function to express it as the
    index balanced accuracy (IBA). You need to use this function to
    decorate any scoring function.

    Only metrics requiring ``y_pred`` can be corrected with the index
    balanced accuracy. ``y_score`` cannot be used since the dominance
    cannot be computed.

    Read more in the `User Guide <https://imbalanced-learn.org/stable/metrics.html#imbalanced-metrics>`_.

    Parameters
    ----------
    alpha : float, default=0.1
        Weighting factor.

    squared : bool, default=True
        If ``squared`` is True, then the metric computed will be squared
        before to be weighted.

    Returns
    -------
    iba_scoring_func : callable,
        Returns the scoring metric decorated which will automatically compute
        the index balanced accuracy.

    Notes
    -----
    See :ref:`sphx_glr_auto_examples_evaluation_plot_metrics.py` for an example.

    References
    ----------
    .. [1] García, Vicente, Javier Salvador Sánchez, and Ramón Alberto
       Mollineda. "On the effectiveness of preprocessing methods when dealing
       with different levels of class imbalance." Knowledge-Based Systems 25.1
       (2012): 13-21.

    Examples
    --------
    >>> from imbens.metrics import geometric_mean_score as gmean
    >>> from imbens.metrics import make_index_balanced_accuracy as iba
    >>> gmean = iba(alpha=0.1, squared=True)(gmean)
    >>> y_true = [1, 0, 0, 1, 0, 1]
    >>> y_pred = [0, 0, 1, 1, 0, 1]
    >>> print(gmean(y_true, y_pred, average=None))
    [ 0.44444444  0.44444444]
    """

    def decorate(scoring_func):
        @functools.wraps(scoring_func)
        def compute_score(*args, **kwargs):
            signature_scoring_func = signature(scoring_func)
            params_scoring_func = set(signature_scoring_func.parameters.keys())

            # check that the scoring function does not need a score
            # and only a prediction
            prohibitied_y_pred = set(["y_score", "y_prob", "y2"])
            if prohibitied_y_pred.intersection(params_scoring_func):
                raise AttributeError(
                    f"The function {scoring_func.__name__} has an unsupported"
                    f" attribute. Metric with`y_pred` are the"
                    f" only supported metrics is the only"
                    f" supported."
                )

            args_scoring_func = signature_scoring_func.bind(*args, **kwargs)
            args_scoring_func.apply_defaults()
            _score = scoring_func(*args_scoring_func.args, **args_scoring_func.kwargs)
            if squared:
                _score = np.power(_score, 2)

            signature_sens_spec = signature(sensitivity_specificity_support)
            params_sens_spec = set(signature_sens_spec.parameters.keys())
            common_params = params_sens_spec.intersection(
                set(args_scoring_func.arguments.keys())
            )

            args_sens_spec = {k: args_scoring_func.arguments[k] for k in common_params}

            if scoring_func.__name__ == "geometric_mean_score":
                if "average" in args_sens_spec:
                    if args_sens_spec["average"] == "multiclass":
                        args_sens_spec["average"] = "macro"
            elif (
                scoring_func.__name__ == "accuracy_score"
                or scoring_func.__name__ == "jaccard_score"
            ):
                # We do not support multilabel so the only average supported
                # is binary
                args_sens_spec["average"] = "binary"

            sensitivity, specificity, _ = sensitivity_specificity_support(
                **args_sens_spec
            )

            dominance = sensitivity - specificity
            return (1.0 + alpha * dominance) * _score

        return compute_score

    return decorate


@_deprecate_positional_args
def classification_report_imbalanced(
    y_true,
    y_pred,
    *,
    labels=None,
    target_names=None,
    sample_weight=None,
    digits=2,
    alpha=0.1,
    output_dict=False,
    zero_division="warn",
):
    """Build a classification report based on metrics used with imbalanced
    dataset

    Specific metrics have been proposed to evaluate the classification
    performed on imbalanced dataset. This report compiles the
    state-of-the-art metrics: precision/recall/specificity, geometric
    mean, and index balanced accuracy of the
    geometric mean.

    Read more in the `User Guide <https://imbalanced-learn.org/stable/metrics.html#classification-report>`_.

    Parameters
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) target values.

    y_pred : 1d array-like, or label indicator array / sparse matrix
        Estimated targets as returned by a classifier.

    labels : array-like of shape (n_labels,), default=None
        Optional list of label indices to include in the report.

    target_names : list of str of shape (n_labels,), default=None
        Optional display names matching the labels (same order).

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    digits : int, default=2
        Number of digits for formatting output floating point values.
        When ``output_dict`` is ``True``, this will be ignored and the
        returned values will not be rounded.

    alpha : float, default=0.1
        Weighting factor.

    output_dict : bool, default=False
        If True, return output as dict.

    zero_division : "warn" or {0, 1}, default="warn"
        Sets the value to return when there is a zero division. If set to
        "warn", this acts as 0, but warnings are also raised.

    Returns
    -------
    report : string / dict
        Text summary of the precision, recall, specificity, geometric mean,
        and index balanced accuracy.
        Dictionary returned if output_dict is True. Dictionary has the
        following structure::

            {'label 1': {'pre':0.5,
                         'rec':1.0,
                         ...
                        },
             'label 2': { ... },
              ...
            }

    Examples
    --------
    >>> import numpy as np
    >>> from imbens.metrics import classification_report_imbalanced
    >>> y_true = [0, 1, 2, 2, 2]
    >>> y_pred = [0, 0, 2, 2, 1] # doctest : +NORMALIZE_WHITESPACE
    >>> target_names = ['class 0', 'class 1', \
    'class 2'] # doctest : +NORMALIZE_WHITESPACE
    >>> print(classification_report_imbalanced(y_true, y_pred, \
    target_names=target_names))
                       pre       rec       spe        f1       geo       iba\
       sup
    <BLANKLINE>
        class 0       0.50      1.00      0.75      0.67      0.87      0.77\
         1
        class 1       0.00      0.00      0.75      0.00      0.00      0.00\
         1
        class 2       1.00      0.67      1.00      0.80      0.82      0.64\
         3
    <BLANKLINE>
    avg / total       0.70      0.60      0.90      0.61      0.66      0.54\
         5
    <BLANKLINE>

    """

    if labels is None:
        labels = unique_labels(y_true, y_pred)
    else:
        labels = np.asarray(labels)

    last_line_heading = "avg / total"

    if target_names is None:
        target_names = [f"{label}" for label in labels]
    name_width = max(len(cn) for cn in target_names)
    width = max(name_width, len(last_line_heading), digits)

    headers = ["pre", "rec", "spe", "f1", "geo", "iba", "sup"]
    fmt = "%% %ds" % width  # first column: class name
    fmt += "  "
    fmt += " ".join(["% 9s" for _ in headers])
    fmt += "\n"

    headers = [""] + headers
    report = fmt % tuple(headers)
    report += "\n"

    # Compute the different metrics
    # Precision/recall/f1
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        sample_weight=sample_weight,
        zero_division=zero_division,
    )
    # Specificity
    specificity = specificity_score(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        sample_weight=sample_weight,
    )
    # Geometric mean
    geo_mean = geometric_mean_score(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        sample_weight=sample_weight,
    )
    # Index balanced accuracy
    iba_gmean = make_index_balanced_accuracy(alpha=alpha, squared=True)(
        geometric_mean_score
    )
    iba = iba_gmean(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        sample_weight=sample_weight,
    )

    report_dict = {}
    for i, label in enumerate(labels):
        report_dict_label = {}
        values = [target_names[i]]
        for score_name, score_value in zip(
            headers[1:-1],
            [
                precision[i],
                recall[i],
                specificity[i],
                f1[i],
                geo_mean[i],
                iba[i],
            ],
        ):
            values += ["{0:0.{1}f}".format(score_value, digits)]
            report_dict_label[score_name] = score_value
        values += [f"{support[i]}"]
        report_dict_label[headers[-1]] = support[i]
        report += fmt % tuple(values)

        report_dict[label] = report_dict_label

    report += "\n"

    # compute averages
    values = [last_line_heading]
    for score_name, score_value in zip(
        headers[1:-1],
        [
            np.average(precision, weights=support),
            np.average(recall, weights=support),
            np.average(specificity, weights=support),
            np.average(f1, weights=support),
            np.average(geo_mean, weights=support),
            np.average(iba, weights=support),
        ],
    ):
        values += ["{0:0.{1}f}".format(score_value, digits)]
        report_dict[f"avg_{score_name}"] = score_value
    values += [f"{np.sum(support)}"]
    report += fmt % tuple(values)
    report_dict["total_support"] = np.sum(support)

    if output_dict:
        return report_dict
    return report


def macro_averaged_mean_absolute_error(y_true, y_pred, *, sample_weight=None):
    """Compute Macro-Averaged Mean Absolute Error (MA-MAE)
    for imbalanced ordinal classification.

    This function computes each MAE for each class and average them,
    giving an equal weight to each class.

    Read more in the `User Guide <https://imbalanced-learn.org/stable/metrics.html#macro-averaged-mean-absolute-error>`_.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.

    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated targets as returned by a classifier.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    Returns
    -------
    loss : float or ndarray of floats
        Macro-Averaged MAE output is non-negative floating point.
        The best value is 0.0.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.metrics import mean_absolute_error
    >>> from imbens.metrics import macro_averaged_mean_absolute_error
    >>> y_true_balanced = [1, 1, 2, 2]
    >>> y_true_imbalanced = [1, 2, 2, 2]
    >>> y_pred = [1, 2, 1, 2]
    >>> mean_absolute_error(y_true_balanced, y_pred)
    0.5
    >>> mean_absolute_error(y_true_imbalanced, y_pred)
    0.25
    >>> macro_averaged_mean_absolute_error(y_true_balanced, y_pred)
    0.5
    >>> macro_averaged_mean_absolute_error(y_true_imbalanced, y_pred)
    0.16666666666666666
    """
    _, y_true, y_pred = _check_targets(y_true, y_pred)
    if sample_weight is not None:
        sample_weight = column_or_1d(sample_weight)
    else:
        sample_weight = np.ones(y_true.shape)
    check_consistent_length(y_true, y_pred, sample_weight)
    labels = unique_labels(y_true, y_pred)
    mae = []
    for possible_class in labels:
        indices = np.flatnonzero(y_true == possible_class)

        mae.append(
            mean_absolute_error(
                y_true[indices],
                y_pred[indices],
                sample_weight=sample_weight[indices],
            )
        )

    return np.sum(mae) / len(mae)
