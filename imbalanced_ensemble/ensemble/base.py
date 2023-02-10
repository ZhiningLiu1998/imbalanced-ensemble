"""ImbalancedEnsembleClassifierMixin: mixin class for all 
imbalanced ensemble estimators.
BaseImbalancedEnsemble: a general base class for imbalanced ensemble.
"""

# Authors: Zhining Liu <zhining.liu@outlook.com>
# License: MIT


# %%
LOCAL_DEBUG = False

if not LOCAL_DEBUG:
    from ..base import TRAINING_TYPES
    from ..utils._validation import _deprecate_positional_args
    from ..utils._docstring import Substitution, _get_parameter_docstring
else:           # pragma: no cover
    import sys  # For local test
    sys.path.append("..")
    from base import TRAINING_TYPES
    from utils._validation import _deprecate_positional_args
    from utils._docstring import Substitution, _get_parameter_docstring

from abc import ABCMeta, abstractmethod

import numpy as np
from collections import Counter
from joblib import Parallel

import numpy as np
from sklearn.base import ClassifierMixin, clone
from sklearn.ensemble import BaseEnsemble
from sklearn.ensemble._base import _set_random_states
from sklearn.ensemble._base import _partition_estimators
from sklearn.ensemble._bagging import _parallel_predict_proba
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import check_random_state
from sklearn.utils import check_array
from sklearn.utils.fixes import delayed
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import (_check_sample_weight, 
                                      check_random_state, 
                                      check_is_fitted, 
                                      column_or_1d, 
                                      check_array,
                                      has_fit_parameter,)


TRAINING_LOG_HEAD_TITLES = {
    'iter': '#Estimators',
    'class_distr': 'Class Distribution',
    'datasets': 'Datasets',
    'metrics': 'Metrics',
}

MAX_INT = np.iinfo(np.int32).max


def sort_dict_by_key(d):
    """Sort a dict by key, return sorted dict."""
    return dict(sorted(d.items(), key=lambda k: k[0]))


class ImbalancedEnsembleClassifierMixin(ClassifierMixin):
    """Mixin class for all ensemble classifiers in imbalanced-ensemble.

    This class is essential for a derived class to be identified by the
    sklearn and imbalanced-ensemble package. Additionally, it provides 
    several utilities for formatting training logs of imbalanced-ensemble 
    classifiers.
    
    Attributes
    ----------
    _estimator_type : ``'classifier'``
        scikit-learn use this attribute to identify a classifier.

    _estimator_ensemble_type : ``'imbalanced_ensemble_classifier'``
        imbalanced-ensemble use this attribute to identify a classifier.
    """
    
    _estimator_type = "classifier"

    _estimator_ensemble_type = "imbalanced_ensemble_classifier"

    
    def _evaluate(self, 
                dataset_name:str, 
                eval_metrics:dict=None,
                return_header:bool=False,
                return_value_dict:bool=False,) -> str or dict:
        """Private function for performance evaluation during the 
        ensemble training process.
        """
        
        eval_datasets_ = self.eval_datasets_
        classes_ = self.classes_
        verbose_format_ = self.train_verbose_format_

        # Temporarily disable verbose
        support_verbose = hasattr(self, 'verbose')
        if support_verbose:
            verbose, self.verbose = self.verbose, 0

        # If no eval_metrics is given, use self.eval_metrics_
        if eval_metrics == None:
            eval_metrics = self.eval_metrics_

        # If return numerical results
        if return_value_dict == True:
            value_dict = {}
            for data_name, (X_eval, y_eval) in eval_datasets_.items():
                y_predict_proba = self.predict_proba(X_eval)
                data_value_dict = {}
                for metric_name, (metric_func, kwargs, ac_proba, ac_labels) \
                                                    in eval_metrics.items():
                    if ac_labels: kwargs['labels'] = classes_
                    if ac_proba: # If the metric take predict probabilities
                        score = metric_func(y_eval, y_predict_proba, **kwargs)
                    else: # If the metric do not take predict probabilities
                        y_predict = classes_.take(np.argmax(
                            y_predict_proba, axis=1), axis=0)
                        score = metric_func(y_eval, y_predict, **kwargs)
                    data_value_dict[metric_name] = score
                value_dict[data_name] = data_value_dict
            out = value_dict
        
        # If return string
        else:
            eval_info = ""
            if return_header == True:
                for metric_name in eval_metrics.keys():
                    eval_info = self._training_log_add_block(
                        eval_info, metric_name, "", "", " ", 
                        verbose_format_['len_metrics'][metric_name], strip=False)
            else:
                (X_eval, y_eval) = eval_datasets_[dataset_name]
                y_predict_proba = self.predict_proba(X_eval)
                for metric_name, (metric_func, kwargs, ac_proba, ac_labels) \
                                                    in eval_metrics.items():
                    if ac_labels: kwargs['labels'] = classes_
                    if ac_proba: # If the metric take predict probabilities
                        score = metric_func(y_eval, y_predict_proba, **kwargs)
                    else: # If the metric do not take predict probabilities
                        y_predict = classes_.take(np.argmax(
                            y_predict_proba, axis=1), axis=0)
                        score = metric_func(y_eval, y_predict, **kwargs)
                    eval_info = self._training_log_add_block(
                        eval_info, "{:.3f}".format(score), "", "", " ", 
                        verbose_format_['len_metrics'][metric_name], strip=False)
            out = eval_info[:-1]

        # Recover verbose state
        if support_verbose:
            self.verbose = verbose

        return out


    def _init_training_log_format(self):
        """Private function for initialization of the training verbose format"""

        if self.train_verbose_:
            len_iter = max(
                len(str(self.n_estimators)),
                len(TRAINING_LOG_HEAD_TITLES['iter'])) + 2
            if self.train_verbose_['print_distribution']:
                len_class_distr = max(
                    len(str(self.target_distr_)),
                    len(str(self.origin_distr_)),
                    len(TRAINING_LOG_HEAD_TITLES['class_distr'])) + 2
            else: len_class_distr = 0
            len_metrics = {
                metric_name: max(len(metric_name), 5) + 2
                for metric_name in self.eval_metrics_.keys()
            }
            metrics_total_length = sum(len_metrics.values()) + len(len_metrics) - 1
            len_datasets = {
                dataset_name: max(metrics_total_length, len("Data: "+dataset_name)+2)
                for dataset_name in self.eval_datasets_.keys()
            }
            self.train_verbose_format_ = {
                'len_iter': len_iter,
                'len_class_distr': len_class_distr,
                'len_metrics': len_metrics,
                'len_datasets': len_datasets,}

        return


    def _training_log_add_block(self, info, text, sta_char, fill_char, 
                                end_char, width, strip=True):
        """Private function for adding a block to training log."""

        info = info.rstrip(end_char) if strip else info
        info += "{}{:{fill}^{width}s}{}".format(
            sta_char, text, end_char,
            fill=fill_char, width=width)

        return info


    def _training_log_add_line(self, info="", texts=None, tabs=None, 
                               widths=None, flags=None):
        """Private function for adding a line to training log."""

        if texts == None:
            texts = ("", "", tuple("" for _ in self.eval_datasets_.keys()))
        if tabs == None:
            tabs = ("┃", "┃", "┃", " ")
        if widths == None:
            widths = (
                self.train_verbose_format_['len_iter'],
                self.train_verbose_format_['len_class_distr'],
                tuple(self.train_verbose_format_['len_datasets'].values())
            )
        if flags == None:
            flags = (True, self.train_verbose_['print_distribution'], self.train_verbose_['print_metrics'])
        (sta_char, mid_char, end_char, fill_char) = tabs
        (flag_iter, flag_distr, flag_metric) = flags
        (text_iter, text_distr, text_metrics) = texts
        (width_iter, width_distr, width_metrics) = widths
        if flag_iter:
            info = self._training_log_add_block(
                info, text_iter, sta_char, fill_char, end_char, width_iter)
        if flag_distr:
            info = self._training_log_add_block(
                info, text_distr, mid_char, fill_char, end_char, width_distr)
        if flag_metric:
            for text_metric, width_metric in zip(text_metrics, width_metrics):
                info = self._training_log_add_block(
                    info, text_metric, mid_char, fill_char, end_char, width_metric)

        return info
        

    def _training_log_to_console_head(self):
        """Private function for printing a table header."""

        # line 1
        info = self._training_log_add_line(
            tabs=("┏", "┳", "┓", "━"),
            )+"\n"
        # line 2
        info = self._training_log_add_line(info,
            texts=("", "", tuple("Data: "+data_name 
                for data_name in self.eval_datasets_.keys()))
            )+"\n"
        # line 3
        info = self._training_log_add_line(info,
            texts=(
                TRAINING_LOG_HEAD_TITLES['iter'], 
                TRAINING_LOG_HEAD_TITLES['class_distr'], 
                tuple("Metric" for data_name in self.eval_datasets_.keys())
                )
            )+"\n"
        # line 4
        info = self._training_log_add_line(info,
            texts=("", "", tuple(
                self._evaluate('', return_header=True)
                for data_name in self.eval_datasets_.keys()))
            )+"\n"
        # line 5
        info = self._training_log_add_line(info,
            tabs=("┣", "╋", "┫", "━"))

        return info


    def _training_log_to_console(self, i_iter=None, y=None):
        """Private function for printing training log to sys.stdout."""

        if self.train_verbose_:
            
            if not hasattr(self, '_properties'):
                raise AttributeError(
                    f"All imbalanced-ensemble estimators should" + \
                    f" have a `_properties` attribute to specify" + \
                    f" the method family they belong to."
                )

            try:
                training_type = self._properties['training_type']
            except Exception as e:
                e_args = list(e.args)
                e_args[0] += \
                    f" The key 'training_type' does not exist in" + \
                    f" the `_properties` attribute, please check" + \
                    f" your usage."
                e.args = tuple(e_args)
                raise e

            if training_type not in TRAINING_TYPES:
                raise ValueError(f"'training_type' should be in {TRAINING_TYPES}")
            if training_type == 'iterative':
                self._training_log_to_console_iterative(i_iter, y)
            elif training_type == 'parallel':
                self._training_log_to_console_parallel()
            else: raise NotImplementedError(
                f"'_training_log_to_console' for 'training_type' = {training_type}"
                f" needs to be implemented."
            )


    def _training_log_to_console_iterative(self, i_iter, y_resampled):
        """Private function for printing training log to sys.stdout.
        (for ensemble classifiers that train in an iterative manner)"""

        if i_iter == 0:
            print(self._training_log_to_console_head())
        
        eval_data_names = self.eval_datasets_.keys()

        if (i_iter+1) % self.train_verbose_['granularity'] == 0 or i_iter == 0:
            print(self._training_log_add_line(texts=(
                    f"{i_iter+1}", f"{sort_dict_by_key(Counter(y_resampled))}", 
                    tuple(self._evaluate(data_name) for data_name in eval_data_names)
                )))
        
        if (i_iter+1) == self.n_estimators:
            print(self._training_log_add_line(tabs=("┣", "╋", "┫", "━")))
            print(self._training_log_add_line(texts=(
                    "final", f"{sort_dict_by_key(Counter(y_resampled))}", 
                    tuple(self._evaluate(data_name) for data_name in eval_data_names)
                )))
            print(self._training_log_add_line(tabs=("┗", "┻", "┛", "━")))


    def _training_log_to_console_parallel(self):
        """Private function for printing training log to sys.stdout.
        (for ensemble classifiers that train in a parallel manner)"""

        eval_data_names = self.eval_datasets_.keys()
        print(self._training_log_to_console_head())
        print(self._training_log_add_line(texts=(
                str(self.n_estimators), "", 
                tuple(self._evaluate(data_name) for data_name in eval_data_names)
            )))
        print(self._training_log_add_line(tabs=("┗", "┻", "┛", "━")))


_properties = {
    'ensemble_type': 'general',
}

@Substitution(
    random_state=_get_parameter_docstring('random_state'),
    n_jobs=_get_parameter_docstring('n_jobs', **_properties),
)
class BaseImbalancedEnsemble(ImbalancedEnsembleClassifierMixin, 
                             BaseEnsemble, metaclass=ABCMeta):
    """Base class for all imbalanced-ensemble classes that are 
    NOT based an existing ensemble learning framework like Boosting,
    Bagging or RandomForest.

    Warning: This class should not be used directly. Use derived classes
    instead.

    Parameters
    ----------
    estimator : object
        The base estimator from which the ensemble is built.

    n_estimators : int, default=50
        The number of estimators in the ensemble.

    estimator_params : list of str, default=tuple()
        The list of attributes to use as parameters when instantiating a
        new base estimator. If none are given, default parameters are used.

    {n_jobs}
    
    {random_state}

    verbose : int, default=0
        Controls the verbosity when predicting.

    Attributes
    ----------
    estimator_ : estimator
        The base estimator from which the ensemble is grown.
        
    estimators_ : list of estimators
        The collection of fitted base estimators.
    """
    
    def __init__(self, 
                 estimator, 
                 n_estimators:int=50,
                 estimator_params=tuple(), 
                 random_state=None,
                 n_jobs=None, 
                 verbose=0,):

        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.check_x_y_args = {
            'accept_sparse': ['csr', 'csc'],
            'force_all_finite': False,
            'dtype': None,
        }

        super(BaseImbalancedEnsemble, self).__init__(
            estimator=estimator,
            n_estimators=n_estimators,
            estimator_params=estimator_params,
        )

        self._properties = _properties
    

    def _validate_y(self, y):
        """Validate the label vector."""
        y = column_or_1d(y, warn=True)
        check_classification_targets(y)
        self.classes_, y_encoded = np.unique(y, return_inverse=True)
        self.n_classes_ = len(self.classes_)
        return y


    def _validate_estimator(self, default):
        """Check the estimator, sampler and the n_estimator attribute.

        Sets the estimator_` and sampler_` attributes.
        """

        # validate estimator using 
        # sklearn.ensemble.BaseEnsemble._validate_estimator
        super()._validate_estimator(default=default)

        if hasattr(self, 'sampler'):
            # validate sampler and sampler_kwargs
            # validated sampler stored in self.sampler_
            try:
                self.sampler_ = clone(self.sampler)
            except Exception as e:
                e_args = list(e.args)
                e_args[0] = "Exception occurs when trying to validate" + \
                            " sampler: " + e_args[0]
                e.args = tuple(e_args)
                raise e
    

    def _make_sampler(self, append=True, random_state=None, **overwrite_kwargs):
        """Make and configure a copy of the `sampler_` attribute.

        Warning: This method should be used to properly instantiate new
        sub-samplers.
        """

        sampler = clone(self.sampler_)
        if hasattr(self, 'sampler_kwargs_'):
            sampler.set_params(**self.sampler_kwargs_)

        # Arguments passed to _make_sampler function have higher priority,
        # they will overwrite the self.sampler_kwargs_
        sampler.set_params(**overwrite_kwargs)
        
        if random_state is not None:
            _set_random_states(sampler, random_state)

        if append:
            self.samplers_.append(sampler)

        return sampler
    

    @_deprecate_positional_args
    def fit(self, X, y, *, sample_weight=None, **kwargs):
        """Build the ensemble classifier from the training set (X, y)."""

        # Check random state
        self.random_state = check_random_state(self.random_state)

        # Convert data (X is required to be 2d and indexable)
        X, y = self._validate_data(X, y, **self.check_x_y_args)
        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X, dtype=np.float64)
            sample_weight /= sample_weight.sum()
            if np.any(sample_weight < 0):
                raise ValueError("sample_weight cannot contain negative weights")

        # Remap output
        n_samples, self.n_features_in_ = X.shape
        self.features_ = np.arange(self.n_features_in_)
        self._n_samples = n_samples
        y = self._validate_y(y)
        self._encode_map = {c: np.where(self.classes_==c)[0][0] for c in self.classes_}

        # Check parameters
        self._validate_estimator(default=DecisionTreeClassifier())
        
        # If the base estimator do not support sample weight and sample weight
        # is not None, raise an ValueError
        support_sample_weight = has_fit_parameter(self.estimator_,
                                                "sample_weight")
        if not support_sample_weight and sample_weight is not None:
            raise ValueError("The base estimator doesn't support sample weight")

        self.estimators_, self.estimators_features_ = [], []

        return self._fit(X, y, sample_weight=sample_weight, **kwargs)
    

    @abstractmethod
    def _fit(self, X, y, sample_weight, **kwargs):
        """Needs to be implemented in the derived class"""
        pass
        
    
    def predict_proba(self, X):
        """Predict class probabilities for X.

        The predicted class probabilities of an input sample is computed as
        the mean predicted class probabilities of the base estimators in the
        ensemble. If base estimators do not implement a ``predict_proba``
        method, then it resorts to voting and the predicted class probabilities
        of an input sample represents the proportion of estimators predicting
        each class.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.

        Returns
        -------
        p : array of shape = [n_samples, n_classes]
            The class probabilities of the input samples. 
        """

        check_is_fitted(self)
        # Check data
        X = check_array(
            X, accept_sparse=['csr', 'csc'], dtype=None,
            force_all_finite=False
        )
        if self.n_features_in_ != X.shape[1]:
            raise ValueError("Number of features of the model must "
                             "match the input. Model n_features is {0} and "
                             "input n_features is {1}."
                             "".format(self.n_features_in_, X.shape[1]))
        
        # Parallel loop
        n_jobs, _, starts = _partition_estimators(self.n_estimators,
                                                             self.n_jobs)

        all_proba = Parallel(n_jobs=n_jobs, verbose=self.verbose,
                             **self._parallel_args())(
            delayed(_parallel_predict_proba)(
                self.estimators_[starts[i]:starts[i + 1]],
                self.estimators_features_[starts[i]:starts[i + 1]],
                X,
                self.n_classes_)
            for i in range(n_jobs))

        # Reduce
        proba = sum(all_proba) / len(self.estimators_)

        return proba
        

    def predict(self, X):
        """Predict class for X.

        The predicted class of an input sample is computed as the class with
        the highest mean predicted probability. If base estimators do not
        implement a ``predict_proba`` method, then it resorts to voting.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.

        Returns
        -------
        y : ndarray of shape (n_samples,)
            The predicted classes.
        """
        
        predicted_probabilitiy = self.predict_proba(X)
        return self.classes_.take((np.argmax(predicted_probabilitiy, axis=1)),
                                  axis=0)


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
            if hasattr(self, 'estimator_weights_'):
                norm = self.estimator_weights_.sum()
                return (
                    sum(
                        weight * clf.feature_importances_
                        for weight, clf in zip(self.estimator_weights_, self.estimators_)
                    )
                    / norm
                )
            else:
                return (
                    sum(
                        clf.feature_importances_ for clf in self.estimators_
                    )
                    / len(self.estimators_)
                )

        except AttributeError as e:
            raise AttributeError(
                "Unable to compute feature importances "
                "since estimator does not have a "
                "feature_importances_ attribute"
            ) from e


    def _parallel_args(self):
        return {}
