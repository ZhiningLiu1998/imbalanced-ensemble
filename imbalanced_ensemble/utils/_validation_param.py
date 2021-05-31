"""Utilities for parameter validation."""

# Authors: Zhining Liu <zhining.liu@outlook.com>
# License: MIT


from copy import copy
from warnings import warn
from collections import Counter
from inspect import signature

import numbers
import numpy as np
from math import ceil
from sklearn.ensemble import BaseEnsemble
from sklearn.utils import check_array, check_X_y
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


SAMPLING_KIND = (
    "over-sampling",
    "under-sampling",
    "multi-class-hybrid-sampling",
)
SamplingKindError = NotImplementedError(
    f"'sampling_type' must be one of {SAMPLING_KIND}."
)


def _target_samples_int(y, n_target_samples, sampling_type):
    target_stats = dict(Counter(y))
    max_class_ = max(target_stats, key=target_stats.get)
    min_class_ = min(target_stats, key=target_stats.get)
    n_max_class_samples_ = target_stats[max_class_]
    n_min_class_samples_ = target_stats[min_class_]
    if sampling_type == 'under-sampling':
        if n_target_samples >= n_max_class_samples_:
            raise ValueError(
                f"'n_target_samples' >= the number of samples"
                f" of the largest class ({n_max_class_samples_})."
                f" Set 'n_target_samples' < {n_max_class_samples_}"
                f" to perform under-sampling properly."
            )
        target_distr = dict([
            (label, min(n_target_samples, target_stats[label]))
            for label in target_stats.keys()
        ])
        return target_distr
    elif sampling_type == 'over-sampling':
        if n_target_samples <= n_min_class_samples_:
            raise ValueError(
                f"'n_target_samples' <= the number of samples"
                f" of the largest class ({n_min_class_samples_})."
                f" Set 'n_target_samples' > {n_min_class_samples_}"
                f" to perform over-sampling properly."
            )
        target_distr = dict([
            (label, max(n_target_samples, target_stats[label]))
            for label in target_stats.keys()
        ])
        return target_distr
    elif sampling_type == "multi-class-hybrid-sampling":
        warning_info =  f" Set 'n_target_samples' between [{n_min_class_samples_}" + \
                f" , {n_max_class_samples_}] if you want to perform" + \
                f" multi-class hybrid-sampling (under-sample the minority" + \
                f" classes, over-sample the majority classes) properly."
        if n_target_samples >= n_max_class_samples_:
            raise Warning(
                f"'n_target_samples' >= the number of samples" + \
                f" of the largest class ({n_max_class_samples_})." + \
                f" ONLY over-sampling will be applied to all classes." + warning_info
            )
        elif n_target_samples <= n_min_class_samples_:
            raise Warning(
                f"'n_target_samples' <= the number of samples" + \
                f" of the largest class ({n_min_class_samples_})." + \
                f" ONLY under-sampling will be applied to all classes." + warning_info
            )
        target_distr = dict([
            (label, n_target_samples)
            for label in target_stats.keys()
        ])
        return target_distr
    else: raise SamplingKindError


def _target_samples_dict(y, n_target_samples, sampling_type):
    target_stats = dict(Counter(y))
    # check that all keys in n_target_samples are also in y
    set_diff_sampling_strategy_target = set(n_target_samples.keys()) - set(
        target_stats.keys()
    )
    if len(set_diff_sampling_strategy_target) > 0:
        raise ValueError(
            f"The {set_diff_sampling_strategy_target} target class is/are not "
            f"present in the data."
        )
    # check that there is no negative number
    if any(n_samples < 0 for n_samples in n_target_samples.values()):
        raise ValueError(
            f"The number of samples in a class cannot be negative."
            f"'n_target_samples' contains some negative value: {n_target_samples}"
        )

    if sampling_type == 'under-sampling':
        target_distr = copy(target_stats)
        for class_label, n_target_sample in n_target_samples.items():
            n_origin_sample = target_stats[class_label]
            if n_target_sample > n_origin_sample:
                raise ValueError(
                    f" The target number of samples of class {class_label}"
                    f" should be < {n_origin_sample} (number of samples"
                    f" in class {class_label}) to perform under-sampling,"
                    f" got {n_target_sample}."
                )
            else:
                target_distr[class_label] = n_target_sample
        return target_distr

    elif sampling_type == 'over-sampling':
        target_distr = copy(target_stats)
        for class_label, n_target_sample in n_target_samples.items():
            n_origin_sample = target_stats[class_label]
            if n_target_sample < n_origin_sample:
                raise ValueError(
                    f" The target number of samples of class {class_label}"
                    f" should be > {n_origin_sample} (number of samples"
                    f" in class {class_label}) to perform over-sampling,"
                    f" got {n_target_sample}."
                )
            else:
                target_distr[class_label] = n_target_sample
        return target_distr

    elif sampling_type == "multi-class-hybrid-sampling":
        target_distr = copy(target_stats)
        if all(n_target_samples[label] <= target_stats[label] for label in n_target_samples.keys()):
            raise Warning(
                f"The target number of samples is smaller than the number"
                f" of original samples for all classes. ONLY under-sampling"
                f" will be carried out."
            )
        elif all(n_target_samples[label] >= target_stats[label] for label in n_target_samples.keys()):
            raise Warning(
                f"The target number of samples is greater than the number"
                f" of original samples for all classes. ONLY over-sampling"
                f" will be carried out."
            )
        target_distr.update(n_target_samples)
        return target_distr
    
    else: raise SamplingKindError


def check_n_target_samples(y, n_target_samples, sampling_type):
    if isinstance(n_target_samples, numbers.Integral):
        return _target_samples_int(y, n_target_samples, sampling_type)
    elif isinstance(n_target_samples, dict):
        return _target_samples_dict(y, n_target_samples, sampling_type)
    else: raise ValueError(
        f"'n_target_samples' should be of type `int` or `dict`,"
        f" got {type(n_target_samples)}."
    )


def check_target_label(y, target_label, sampling_type):
    """check parameter `target_label`."""
    
    target_stats = dict(Counter(y))
    if isinstance(target_label, numbers.Integral):
        if target_label in target_stats.keys():
            return target_label
        else: raise ValueError(
            f"The target class {target_label} is not present in the data."
        )
    else: raise TypeError(
        f"'target_label' should be of type `int`,"
        f" got {type(target_label)}."
    )


def check_target_label_and_n_target_samples(y, target_label, n_target_samples, sampling_type):
    """Jointly check `target_label` and `n_target_samples` parameters."""

    # Store the original target class distribution
    target_stats = dict(Counter(y))
    min_class = min(target_stats, key=target_stats.get)
    maj_class = max(target_stats, key=target_stats.get)
    
    # if n_target_samples is NOT specified
    if n_target_samples == None:
        # Set target_label if NOT specified
        if target_label == None:
            if sampling_type == "under-sampling":
                target_label_ = min_class
            elif sampling_type == "over-sampling":
                target_label_ = maj_class
            elif sampling_type == "multi-class-hybrid-sampling":
                raise ValueError(
                    f"For \"multi-class-hybrid-sampling\", must specify"
                    f" 'n_target_samples' or 'target_label'."
                )    
            else: raise SamplingKindError
        # Check target_label
        else: target_label_ = check_target_label(y, target_label, sampling_type)
        # Set n_target_samples
        n_target_samples = target_stats[target_label_]
    
    # if n_target_samples is specified
    else:
        if target_label == None:
            target_label_ = target_label
        # n_target_samples and target_label CANNOT both be both specified
        else:
            raise ValueError(
                f"'n_target_samples' and 'target_label' cannot"
                f" be specified at the same time."
            )
    
    # Check n_target_samples
    target_distr_ = check_n_target_samples(y, n_target_samples, sampling_type)

    return target_label_, target_distr_


BALANCING_SCHEDULE_PARAMS_TYPE = {
    'origin_distr': dict,
    'target_distr': dict,
    'i_estimator': numbers.Integral,
    'total_estimator': numbers.Integral,
}


def _uniform_schedule(origin_distr, target_distr, i_estimator, total_estimator):
    """Return target distribution"""
    for param, (param_name, param_type) in zip(
            [origin_distr, target_distr, i_estimator, total_estimator], 
            list(BALANCING_SCHEDULE_PARAMS_TYPE.items())):
        if not isinstance(param, param_type):
            raise TypeError(
                f"'{param_name}' must be `{param_type}`, got {type(param)}."
            )
    if i_estimator >= total_estimator:
        raise ValueError(
            f"'i_estimator' should < 'total_estimator',"
            f" got 'i_estimator' = {i_estimator} >= 'total_estimator' = {total_estimator}."
        )
    return target_distr


def _progressive_schedule(origin_distr, target_distr, i_estimator, total_estimator):
    """Progressively interpolate between original and target distribution"""
    for param, (param_name, param_type) in zip(
            [origin_distr, target_distr, i_estimator, total_estimator], 
            list(BALANCING_SCHEDULE_PARAMS_TYPE.items())):
        if not isinstance(param, param_type):
            raise TypeError(
                f"'{param_name}' must be `{param_type}`, got {type(param)}."
            )
    if i_estimator >= total_estimator:
        raise ValueError(
            f"'i_estimator' should < 'total_estimator',"
            f" got 'i_estimator' = {i_estimator} >= 'total_estimator' = {total_estimator}."
        )
    result_distr = {}
    if total_estimator == 1:
        progress_ = 1
    else: progress_ = i_estimator / (total_estimator-1)
    for label in origin_distr.keys():
        result_distr[label] = ceil(
            origin_distr[label]*(1.-progress_) + \
            target_distr[label]*progress_ - 1e-10
        )
    return result_distr


BALANCING_KIND_MAPPING = {
    "uniform": _uniform_schedule,
    "progressive": _progressive_schedule,
}
BALANCING_KIND = list(BALANCING_KIND_MAPPING.keys())
BALANCING_SCHEDULE_INFO = \
    "\nNote: self-defined `balancing_schedule` should take 4 positional" + \
    " arguments with order ('origin_distr': `dict`, 'target_distr':" + \
    " `dict`, 'i_estimator': `int`, 'total_estimator': `int`), and" + \
    " return a 'result_distr': `dict`. For all `dict`, the keys" + \
    " correspond to the targeted classes, and the values correspond to the" + \
    " (desired) number of samples for each class."


def check_balancing_schedule(balancing_schedule):
    """Check the `balancing_schedule` parameter."""
    if callable(balancing_schedule):
        try:
            return_value = balancing_schedule({}, {}, 0, 0)
        except Exception as e:
            e_args = list(e.args)
            e_args[0] += BALANCING_SCHEDULE_INFO
            e.args = tuple(e_args)
            raise e
        else:
            if not isinstance(return_value, dict):
                raise TypeError(
                    f" The self-defined `balancing_schedule` must return a `dict`," + \
                    f" got {type(return_value)}" + \
                    BALANCING_SCHEDULE_INFO
                )
        return balancing_schedule

    if balancing_schedule in BALANCING_KIND:
        return BALANCING_KIND_MAPPING[balancing_schedule]
    else:
        balancing_schedule_info = balancing_schedule if isinstance(balancing_schedule, str) \
            else type(balancing_schedule)
        raise TypeError(
            f"'balancing_schedule' should be one of {BALANCING_KIND} or `callable`,"
            f" got {balancing_schedule_info}."
        )


EVAL_METRICS_DEFAULT = {
    'acc': (accuracy_score, {}),
    'balanced_acc': (balanced_accuracy_score, {}),
    'weighted_f1': (f1_score, {'average':'weighted'}),
}
EVAL_METRICS_INFO = \
            " Example 'eval_metrics': {..., 'metric_name': ('metric_func', 'metric_kwargs'), ...}."
            # " where `metric_name` is string, `metric_func` is `callable`," + \
            # " and `metric_arguments` is a dict of arguments" + \
            # " that needs to be passed to the metric function," + \
            # " e.g., {..., `argument_name`: `value`}."
EVAL_METRICS_TUPLE_TYPE = {
    'metric_func': callable,
    'metric_kwargs': dict,
}
EVAL_METRICS_TUPLE_LEN = len(EVAL_METRICS_TUPLE_TYPE)


def _check_eval_metric_func(metric_func):
    if not callable(metric_func):
        raise TypeError(
            f" The 'metric_func' should be `callable`, got {type(metric_func)},"
            f" please check your usage."
            + EVAL_METRICS_INFO
        )
    if 'y_true' not in signature(metric_func).parameters:
        raise RuntimeError(
            f"The metric function must have the keyword argument 'y_true'"
            f" (true labels or binary label indicators, 1d-array of shape (n_samples,))."
        )
    if 'y_pred' not in signature(metric_func).parameters and \
        'y_score' not in signature(metric_func).parameters:
        raise RuntimeError(
            f"The metric function must have the keyword argument 'y_pred' or 'y_score'."
            f" When use 'y_pred': it corresponds to predicted labels, 1d-array of shape (n_samples,)."
            f" When use 'y_score': it corresponds to predicted labels, or an array of shape"
            f" (n_samples, n_classes) of probability estimates provided by the predict_proba method.)"
        )
    accept_proba = 'y_score' in signature(metric_func).parameters
    accept_labels = 'labels' in signature(metric_func).parameters
    return metric_func, accept_proba, accept_labels


def _check_eval_metric_args(metric_kwargs):
    if not isinstance(metric_kwargs, dict):
        raise TypeError(
            f" The 'metric_kwargs' should be a `dict` of arguments"
            f" that needs to be passed to the metric function,"
            f" got {type(metric_kwargs)}, "
            f" please check your usage."
            + EVAL_METRICS_INFO
        )
    return metric_kwargs


def _check_eval_metric_name(metric_name):
    if not isinstance(metric_name, str):
        raise TypeError(
            f" The keys must be `string`, got {type(metric_name)}, "
            f" please check your usage."
            + EVAL_METRICS_INFO
        )
    return metric_name


def _check_eval_metric_tuple(metric_tuple, metric_name):
    if not isinstance(metric_tuple, tuple):
        raise TypeError(
            f" The value of '{metric_name}' is {type(metric_tuple)} (should be tuple)," + \
            f" please check your usage."
            + EVAL_METRICS_INFO
        )
    elif len(metric_tuple) != EVAL_METRICS_TUPLE_LEN:
        raise ValueError(
            f" The data tuple of '{metric_name}' has {len(metric_tuple)} element(s)" + \
            f" (should be {EVAL_METRICS_TUPLE_LEN}), please check your usage."
            + EVAL_METRICS_INFO
        )
    else:
        metric_func_, accept_proba, accept_labels = _check_eval_metric_func(metric_tuple[0])
        metric_kwargs_ = _check_eval_metric_args(metric_tuple[1])
        return (
            metric_func_,
            metric_kwargs_,
            accept_proba,
            accept_labels,
        )


def _check_eval_metrics_dict(eval_metrics_dict):
    """check 'eval_metrics' dict."""
    eval_metrics_dict_ = {}
    for metric_name, metric_tuple in eval_metrics_dict.items():
        
        metric_name_ = _check_eval_metric_name(metric_name)
        metric_tuple_ = _check_eval_metric_tuple(metric_tuple, metric_name_)
        eval_metrics_dict_[metric_name_] = metric_tuple_
    
    return eval_metrics_dict_

def check_eval_metrics(eval_metrics):
    """Check parameter `eval_metrics`."""
    if eval_metrics is None:
        return _check_eval_metrics_dict(EVAL_METRICS_DEFAULT)
    elif isinstance(eval_metrics, dict):
        return _check_eval_metrics_dict(eval_metrics)
    else: 
        raise TypeError(
            f"'eval_metrics' must be of type `dict`, got {type(eval_metrics)}, please check your usage."
            + EVAL_METRICS_INFO
        )


TRAIN_VERBOSE_TYPE = {
    'granularity': numbers.Integral,
    'print_distribution': bool,
    'print_metrics': bool,
}

TRAIN_VERBOSE_DEFAULT = {
    # 'granularity' will be set to int(n_estimators_ensemble/10)
    #  when check_train_verbose() is called
    'print_distribution': True,
    'print_metrics': True,
}

TRAIN_VERBOSE_DICT_INFO = \
        " When 'train_verbose' is `dict`, at least one of the following" + \
        " terms should be specified: " + \
        " {'granularity': `int` (default=1)," + \
        " 'print_distribution': `bool` (default=True)," + \
        " 'print_metrics': `bool` (default=True)}."


def check_train_verbose(train_verbose:bool or numbers.Integral or dict,
                        n_estimators_ensemble:int, training_type:str, 
                        **ignored_properties):
                        # n_estimators_ensemble:int,):

    train_verbose_ = copy(TRAIN_VERBOSE_DEFAULT)
    train_verbose_.update({
        'granularity': max(1, int(n_estimators_ensemble/10))
    })

    if training_type == 'parallel':
        # For ensemble classifiers trained in parallel
        # train_verbose can only be of type bool
        if isinstance(train_verbose, bool):
            if train_verbose == True:
                train_verbose_['print_distribution'] = False 
                return train_verbose_
            if train_verbose == False:
                return False
        else: raise TypeError(
            f"'train_verbose' can only be of type `bool`"
            f" for ensemble classifiers trained in parallel,"
            f" gor {type(train_verbose)}."
        )
    
    elif training_type == 'iterative':
        # For ensemble classifiers trained in iterative manner
        # train_verbose can be of type bool / int / dict
        if isinstance(train_verbose, bool):
            if train_verbose == True:
                return train_verbose_
            if train_verbose == False:
                return False

        if isinstance(train_verbose, numbers.Integral):
            train_verbose_.update({'granularity': train_verbose})
            return train_verbose_
            
        if isinstance(train_verbose, dict):
            # check key value type
            set_diff_verbose_keys = set(train_verbose.keys()) - set(TRAIN_VERBOSE_TYPE.keys())
            if len(set_diff_verbose_keys) > 0:
                raise ValueError(
                    f"'train_verbose' keys {set_diff_verbose_keys} are not supported." + \
                    TRAIN_VERBOSE_DICT_INFO
                )
            for key, value in train_verbose.items():
                if not isinstance(value, TRAIN_VERBOSE_TYPE[key]):
                    raise TypeError(
                        f"train_verbose['{key}'] has wrong data type, should be {TRAIN_VERBOSE_TYPE[key]}." + \
                        TRAIN_VERBOSE_DICT_INFO
                    )
            train_verbose_.update(train_verbose)
            return train_verbose_
            
        else: raise TypeError(
            f"'train_verbose' should be of type `bool`, `int`, or `dict`, got {type(train_verbose)} instead." + \
            TRAIN_VERBOSE_DICT_INFO
        )
    
    else: raise NotImplementedError(
        f"'check_train_verbose' for 'training_type' = {training_type}"
        f" needs to be implemented."
    )


VISUALIZER_ENSEMBLES_EXAMPLE_INFO = " Example: {..., ensemble_name: ensemble, ...}"

VISUALIZER_ENSEMBLES_USAGE_INFO = \
    f" All imbalanced ensemble estimators should use the same training & validation" + \
    f" datasets and dataset names for comparable visualizations." + \
    f" Call `fit` with same 'X', 'y', 'eval_datasets'."


def _check_visualizer_ensemble_item(name, estimator) -> bool:
    if not isinstance(name, str):
        raise TypeError(
            f"Ensemble name must be `string`, got {type(name)}."
        )
    
    # Ensure estimator is an fitted sklearn/imbalanced-ensemble estimator
    # and is already fitted.
    check_is_fitted(estimator)

    if not isinstance(estimator, BaseEnsemble):
        raise TypeError(
            f"Value with name '{name}' is not an ensemble classifier instance."
        )

    if getattr(estimator, "_estimator_ensemble_type", None) == \
            "imbalanced_ensemble_classifier":
        is_imbalanced_ensemble_clf = True
    else: is_imbalanced_ensemble_clf = False

    return is_imbalanced_ensemble_clf


def get_dict_subset_by_key(dictionary:dict, subset_keys:list, exclude:bool=False):
    if exclude:
        return {k: v for k, v in dictionary.items() if k not in subset_keys}
    else: return {k: v for k, v in dictionary.items() if k in subset_keys}


def check_visualizer_ensembles(ensembles:dict, eval_datasets_:dict, eval_metrics_:dict) -> dict:

    # Check 'ensembles' parameter
    if not isinstance(ensembles, dict):
        raise TypeError(
            f"'ensembles' must be a `dict`, got {type(ensembles)}." + \
            VISUALIZER_ENSEMBLES_EXAMPLE_INFO
        )
    if len(ensembles) == 0:
        raise ValueError(
            f"'ensembles' must not be empty." + VISUALIZER_ENSEMBLES_EXAMPLE_INFO
        )
    
    # Check all key-value pairs of 'ensembles' and 
    # record names of those are not imbalanced ensemble classifier
    names_imbalanced_ensemble = []
    for name, estimator in ensembles.items():
        if _check_visualizer_ensemble_item(name, estimator):
            names_imbalanced_ensemble.append(name)
    names_sklearn_ensemble = list(set(ensembles.keys()) - set(names_imbalanced_ensemble))

    # Raise error if not all ensembles have the same n_features_
    n_features_fitted = _check_all_estimators_have_same_attribute(ensembles,
        attr_alias = ('n_features_', 'n_features_in_'))
    
    sklearn_ensembles = get_dict_subset_by_key(ensembles, names_sklearn_ensemble)
    imb_ensembles = get_dict_subset_by_key(ensembles, names_sklearn_ensemble, exclude=True)

    # Raise error if not all imbalanced ensembles have the same eval_datasets names
    if not _all_elements_equal([list(estimator.eval_datasets_.keys())
                                for estimator in imb_ensembles.values()]):
        raise ValueError(
            f"Got ensemble estimators that used inconsistent dataset names." + \
            VISUALIZER_ENSEMBLES_USAGE_INFO
        )
        
    # If eval_datasets_ is not given
    if len(eval_datasets_) == 0:
        # If all are sklearn ensemble classifier
        if len(imb_ensembles) == 0:
            raise ValueError(
                f"The 'eval_datasets' must not be empty when all "
                f" input 'ensembles' are sklearn.ensemble classifiers."
            )
        else:
            # Use imbalanced-ensemble estimators' evaluation datasets by default
            return_eval_datasets_ = copy(list(imb_ensembles.values())[0].eval_datasets_)

            # If got mixed types of ensemble classifier and eval_datasets_ is not given
            if len(sklearn_ensembles) > 0:
                warn(
                    f"the 'eval_datasets' is not specified and the input 'ensembles'"
                    f" contains sklearn.ensemble classifier, using evaluation datasets"
                    f" of other imbalanced-ensemble classifiers by default."
                )
    # If eval_datasets_ is given
    else:
        # eval_datasets_ is already validated, 
        # all data should have the same number of features
        n_features_given = list(eval_datasets_.values())[0][0].shape[1]

        # If the given data is inconsistent with the training data
        if n_features_given != n_features_fitted:
            raise ValueError(
                f"Given data in 'eval_datasets' has {n_features_given} features,"
                f" but the ensemble estimators are trained on data with"
                f" {n_features_fitted} features."
            )
        
        # Use the given evaluation datasets
        return_eval_datasets_ = copy(eval_datasets_)

    ensemble_names = list(ensembles.keys())
    dataset_names = list(return_eval_datasets_.keys())
    metric_names = list(eval_metrics_.keys())
    vis_format = {
        'n_ensembles': len(ensemble_names),
        'ensemble_names': tuple(ensemble_names),
        'n_datasets': len(dataset_names),
        'dataset_names': tuple(dataset_names),
        'n_metrics': len(metric_names),
        'metric_names': tuple(metric_names),
    }

    return ensembles, return_eval_datasets_, vis_format


def _check_all_estimators_have_same_attribute(
        ensembles:dict, attr_alias:tuple):

    has_attrs, values, not_has_attr_names = [], [], []
    for name, estimator in ensembles.items():
        recorded_flag = False
        for alias in attr_alias:
            if hasattr(estimator, alias):
                recorded_flag = True
                has_attrs.append(True)
                values.append(getattr(estimator, alias))
                break
        if not recorded_flag:
            has_attrs.append(False)
            values.append(None)
            not_has_attr_names.append(name)
    
    if not all(has_attrs):
        raise ValueError(
            f"Estimators with name {not_has_attr_names} has no"
            f" attribute {attr_alias}, check your usage." 
        )
    
    if not _all_elements_equal(values):
        raise ValueError(
            f"Got ensemble estimators that has inconsistent {attr_alias}."
            f" Make sure that the training data for all estimators"
            f" (also the evaluation data for imbalanced-ensemble estimators)"
            f" are sampled from the same task/distribution."
            )

    return values[0]


def _all_elements_equal(list_to_check:list) -> bool:
    """Private function to check whether all elements of
    list_to_check are equal."""

    # set() is not used here as some times the list 
    # elements are not hashable, e.g., strings.
    if len(list_to_check) == 1:
        return True
    return all([
        (list_to_check[i] == list_to_check[i+1])
        for i in range(len(list_to_check)-1)
        ])


PLOT_FIGSIZE_INFO = " Example: (width, height)."

def check_plot_figsize(figsize):
    if not isinstance(figsize, tuple):
        raise TypeError(
            f"'figsize' must be a tuple with 2 elements,"
            f" got {type(figsize)}." + PLOT_FIGSIZE_INFO
        )
    if len(figsize) != 2:
        raise ValueError(
            f"'figsize' must be a tuple with 2 elements,"
            f" got {len(figsize)} elements." + PLOT_FIGSIZE_INFO
        )
    for value in figsize:
        if not isinstance(value, numbers.Number):
            raise ValueError(
                f"Elements of 'figsize' must be a `int` or `float`,"
                f" got {type(value)}." + PLOT_FIGSIZE_INFO
            )
    return figsize


def check_has_diff_elements(given_set:list or set, 
                            universal_set:list or set, 
                            msg:str=""):
    diff_set = set(given_set) - set(universal_set)
    if len(diff_set) > 0:
        raise ValueError(
            msg % {"diff_set": diff_set}
        )


def check_type(param, param_name:str, typ, typ_name:str=None):
    if not isinstance(param, typ):
        typ_name = str(typ) if typ_name is None else typ_name
        raise ValueError(
            f"'{param_name}' should be of type `{typ_name}`,"
            f" got {type(param)}."
        )
    return param

    
def check_pred_proba(y_pred_proba, n_samples, n_classes, dtype=None):
    """Private function for validating y_pred_proba"""
    if dtype is not None and dtype not in [np.float32, np.float64]:
        dtype = np.float64

    if dtype is None:
        dtype = [np.float64, np.float32]
    y_pred_proba = check_array(
        y_pred_proba, accept_sparse=False, ensure_2d=False, dtype=dtype,
        order="C"
    )
    if y_pred_proba.ndim != 2:
        raise ValueError("Predicted probabilites must be 2D array")

    if y_pred_proba.shape != (n_samples, n_classes):
        raise ValueError("y_pred_proba.shape == {}, expected {}!"
                        .format(y_pred_proba.shape, (n_samples, n_classes)))
    return y_pred_proba
