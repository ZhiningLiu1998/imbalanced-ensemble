"""Utilities for docstring in imbens.
"""
# Adapted from imbalanced-learn

# Authors: Zhining Liu <zhining.liu@outlook.com>
#          Guillaume Lemaitre
# License: MIT

# %%

ENSEMBLE_TYPES = ('boosting', 'bagging', 'random-forest', 'general')

TRAINING_TYPES = ('iterative', 'parallel')

SOLUTION_TYPES = ('resampling', 'reweighting')

SAMPLING_TYPES = ('under-sampling', 'over-sampling')


class Substitution:
    """Decorate a class' docstring to perform string substitution on it.

    This decorator should be robust even if obj.__doc__ is None
    (for example, if -OO was passed to the interpreter)
    """

    def __init__(self, *args, **kwargs):
        if args and kwargs:
            raise AssertionError("Only positional or keyword args are allowed")

        self.params = args or kwargs

    def __call__(self, obj):
        obj.__doc__ = obj.__doc__.format(**self.params)
        return obj


# class SamplerUserGuideSubstitution:
#     """Decorate a sampler class' docstring to replace imblearn User Guide by link.

#     This decorator should be robust even if obj.__doc__ is None
#     (for example, if -OO was passed to the interpreter)
#     """
#     def __init__(self):
#         base_link = 'https://imbalanced-learn.org/stable/'
#         if isinstance(self, (BaseUnderSampler, BaseCleaningSampler)):
#             self.base_link = base_link + 'under_sampling.html'
#         elif isinstance(self, BaseOverSampler):
#             self.base_link = base_link + 'over_sampling.html'

#     def user_guide_sub(self, match: re.Match):
#         string = match.group()
#         term = re.findall(r"<(.*)>", string)[0]
#         if term == 'tomek_links':
#             surfix = 'tomek-s-links'
#         else:
#             surfix = term.replace('_', '-')
#         string = f'`User Guide <{self.base_link}#{surfix}>`_'
#         return string

#     def __call__(self, obj):
#         obj.__doc__ = re.sub(r":ref:`User Guide <.*>`",
#                              self.user_guide_sub,
#                              obj.__doc__)
#         return obj


class FuncSubstitution:
    """Decorate a function's docstring to perform string substitution on it.

    This decorator should be robust even if obj.__doc__ is None
    (for example, if -OO was passed to the interpreter)
    """

    def __init__(self, *args, **kwargs):
        if args:
            raise AssertionError("Only keyword args are allowed")

        self.params = kwargs

    def __call__(self, obj):
        obj.__doc__ = obj.__doc__ % (self.params)
        return obj


class FuncGlossarySubstitution:
    """Decorate a function's docstring to replace sklearn glossary term by link.

    This decorator should be robust even if obj.__doc__ is None
    (for example, if -OO was passed to the interpreter)
    """

    def __init__(self, super_func, *args):
        self.super_doc = super_func.__doc__
        self.args = args
        self.base_link = 'https://scikit-learn.org/stable/glossary.html#term-'

    def __call__(self, obj):
        base_link = self.base_link
        for keyword in self.args:
            surfix = '__' if keyword in base_link else '__'
            obj.__doc__ = self.super_doc.replace(
                ':term:`{kw}`'.format(kw=keyword),
                '`{kw} <{bl}{kw}>`{sf}'.format(kw=keyword, bl=base_link, sf=surfix),
            )
        return obj


_early_termination_docstring = """early_termination : bool, default=False
        Whether to enable early termination for AdaBoost training.
        If True, AdaBoost training can be terminated early when the error 
        is zero or the sum of the sample weights is non-positive.
    """.rstrip()

_random_state_docstring = """random_state : int, RandomState instance, default=None
        Control the randomization of the algorithm.

        - If ``int``, ``random_state`` is the seed used by the random number
          generator;
        - If ``RandomState`` instance, random_state is the random number
          generator;
        - If ``None``, the random number generator is the ``RandomState``
          instance used by ``np.random``.
    """.rstrip()

_random_state_ensemble_docstring = """random_state : int, RandomState instance or None, default=None
        Control the randomization of the algorithm.
        If the base estimator accepts a `random_state` attribute, a different
        seed is generated for each instance in the ensemble.
        Pass an ``int`` for reproducible output across multiple function calls.
        
        - If ``int``, ``random_state`` is the seed used by the random number
          generator;
        - If ``RandomState`` instance, random_state is the random number
          generator;
        - If ``None``, the random number generator is the ``RandomState``
          instance used by ``np.random``.
    """.rstrip()

_random_state_ensemble_resampling_docstring = """random_state : int, RandomState instance or None, default=None
        Control the randomization of the algorithm.
        Within each iteration, a different seed is generated for each sampler.
        If the base estimator accepts a `random_state` attribute, a different
        seed is generated for each instance in the ensemble.
        Pass an ``int`` for reproducible output across multiple function calls.

        - If ``int``, ``random_state`` is the seed used by the random number
          generator;
        - If ``RandomState`` instance, random_state is the random number
          generator;
        - If ``None``, the random number generator is the ``RandomState``
          instance used by ``np.random``.
    """.rstrip()

_n_jobs_docstring = """n_jobs : int, default=None
        Number of CPU cores used during the cross-validation loop.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See
        `Glossary <https://scikit-learn.org/stable/glossary.html#term-n_jobs>`_
        for more details.
    """.rstrip()

_n_jobs_sampler_docstring = """n_jobs_sampler : int, default=None
        Number of CPU cores used during the resampling.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See
        `Glossary <https://scikit-learn.org/stable/glossary.html#term-n_jobs>`_
        for more details.
    """.rstrip()

_n_jobs_fit_pred_docstring = """n_jobs : int, default=None
        The number of jobs to run in parallel for both :meth:`fit` and
        :meth:`predict`. ``None`` means 1 unless in a
        :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors. See `Glossary <https://scikit-learn.org/stable/glossary.html#term-n_jobs>`_
        for more details.
    """.rstrip()

_n_jobs_pred_docstring = """n_jobs : int, default=None
        The number of jobs to run in parallel for :meth:`predict`. 
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` 
        context. ``-1`` means using all processors. See `Glossary <https://scikit-learn.org/stable/glossary.html#term-n_jobs>`_
        for more details.
    """.rstrip()

_warm_start_docstring = """warm_start : bool, default=False
        When set to True, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just fit
        a whole new ensemble. See
        `Glossary <https://scikit-learn.org/stable/glossary.html#term-warm_start>`_
        for more details.
    """.rstrip()

_balancing_schedule_docstring = """balancing_schedule : str, or callable, default='uniform'
            Scheduler that controls how to sample the data set during the ensemble 
            training process.

            - If ``str``, using the predefined balancing schedule.
              Possible choices are:

                - ``'uniform'``: resample to target distribution for all base estimators;
                - ``'progressive'``: The resample class distributions are progressive 
                  interpolation between the original and the target class distribution.
                  Example: For a class :math:`c`, say the number of samples is :math:`N_{c}` 
                  and the target number of samples is :math:`N'_{c}`. Suppose that we are 
                  training the :math:`t`-th base estimator of a :math:`T`-estimator ensemble, then 
                  we expect to get :math:`(1-\\frac{t}{T}) \\cdot N_{c} + \\frac{t}{T} \\cdot N'_{c}` 
                  samples after resampling;

            - If callable, function takes 4 positional arguments with order (``'origin_distr'``: 
              ``dict``, ``'target_distr'``: ``dict``, ``'i_estimator'``: ``int``, ``'total_estimator'``: 
              ``int``), and returns a ``'result_distr'``: ``dict``. For all parameters of type ``dict``, 
              the keys of type ``int`` correspond to the targeted classes, and the values of type ``str`` 
              correspond to the (desired) number of samples for each class.
            """.rstrip()

_eval_datasets_docstring = """eval_datasets : dict, default=None
            Dataset(s) used for evaluation during the ensemble training process.
            The keys should be strings corresponding to evaluation datasets' names. 
            The values should be tuples corresponding to the input samples and target
            values. 
            
            Example: ``eval_datasets = {'valid' : (X_valid, y_valid)}``
            """.rstrip()

_eval_metrics_docstring = """eval_metrics : dict, default=None
            Metric(s) used for evaluation during the ensemble training process.

            - If ``None``, use 3 default metrics:

                - ``'acc'``: 
                  ``sklearn.metrics.accuracy_score()``
                - ``'balanced_acc'``: 
                  ``sklearn.metrics.balanced_accuracy_score()``
                - ``'weighted_f1'``: 
                  ``sklearn.metrics.f1_score(average='weighted')``

            - If ``dict``, the keys should be strings corresponding to evaluation 
              metrics' names. The values should be tuples corresponding to the metric 
              function (``callable``) and additional kwargs (``dict``).

                - The metric function should at least take 2 named/keyword arguments, 
                  ``y_true`` and one of [``y_pred``, ``y_score``], and returns a float
                  as the evaluation score. Keyword arguments:

                  - ``y_true``, 1d-array of shape (n_samples,), true labels or binary 
                    label indicators corresponds to ground truth (correct) labels.
                  - When using ``y_pred``, input will be 1d-array of shape (n_samples,) 
                    corresponds to predicted labels, as returned by a classifier.
                  - When using ``y_score``, input will be 2d-array of shape (n_samples, 
                    n_classes,) corresponds to probability estimates provided by the 
                    predict_proba method. In addition, the order of the class scores 
                    must correspond to the order of ``labels``, if provided in the metric 
                    function, or else to the numerical or lexicographical order of the 
                    labels in ``y_true``.
                
                - The metric additional kwargs should be a dictionary that specifies 
                  the additional arguments that need to be passed into the metric function. 
            
            Example: ``{'weighted_f1': (sklearn.metrics.f1_score, {'average': 'weighted'})}``
            """.rstrip()

_train_verbose_iterative_docstring = """train_verbose : bool, int or dict, default=False
            Controls the verbosity during ensemble training/fitting.

            - If ``bool``: ``False`` means disable training verbose. ``True`` means 
              print training information to sys.stdout use default setting:
              
                - ``'granularity'``        : ``int(n_estimators/10)``
                - ``'print_distribution'`` : ``True``
                - ``'print_metrics'``      : ``True``

            - If ``int``, print information per ``train_verbose`` rounds.

            - If ``dict``, control the detailed training verbose settings. They are:

                - ``'granularity'``: corresponding value should be ``int``, the training
                  information will be printed per ``granularity`` rounds.
                - ``'print_distribution'``: corresponding value should be ``bool``, 
                  whether to print the data class distribution 
                  after resampling. Will be ignored if the 
                  ensemble training does not perform resampling.
                - ``'print_metrics'``: corresponding value should be ``bool``, 
                  whether to print the latest performance score.
                  The performance will be evaluated on the training 
                  data and all given evaluation datasets with the 
                  specified metrics.
              
            .. warning::
                Setting a small ``'granularity'`` value with ``'print_metrics'`` enabled 
                can be costly when the training/evaluation data is large or the metric 
                scores are hard to compute. Normally, one can set ``'granularity'`` to 
                ``n_estimators/10`` (this is used by default).
            """.rstrip()

_train_verbose_parallel_docstring = """train_verbose : bool, default=False
            Controls the verbosity during ensemble training/fitting.
            
            - ``False``: disable training verbose. 
            - ``True``: print the performance score to sys.stdout after the parallel 
              training finished.
            """.rstrip()

_target_label_under_sampling_docstring = """target_label : int, default=None
            Specify the class targeted by the under-sampling. 
            All other classes that have more samples than the target class will 
            be considered as majority classes. They will be under-sampled until 
            the number of samples is equalized. The remaining minority classes 
            (if any) will stay unchanged.
            """.rstrip()

_target_label_over_sampling_docstring = """target_label : int, default=None
            Specify the class targeted by the over-sampling. 
            All other classes that have less samples than the target class will 
            be considered as minority classes. They will be over-sampled until 
            the number of samples is equalized. The remaining majority classes 
            (if any) will stay unchanged.
            """.rstrip()

_n_target_samples_under_sampling_docstring = """n_target_samples : int or dict, default=None
            Specify the desired number of samples (of each class) after the 
            under-sampling. 

            - If ``int``, all classes that have more than the ``n_target_samples`` 
              samples will be under-sampled until the number of samples is equalized.
            - If ``dict``, the keys correspond to the targeted classes. The values 
              correspond to the desired number of samples for each targeted class.
            """.rstrip()

_n_target_samples_over_sampling_docstring = """n_target_samples : int or dict, default=None
            Specify the desired number of samples (of each class) after the 
            over-sampling. 

            - If ``int``, all classes that have less than the ``n_target_samples`` 
              samples will be over-sampled until the number of samples is equalized.
            - If ``dict``, the keys correspond to the targeted classes. The values 
              correspond to the desired number of samples for each targeted class.
            """.rstrip()

_docstring_example = """>>> from imbens.ensemble import %(method_name)s
    >>> from sklearn.datasets import make_classification
    >>>
    >>> X, y = make_classification(n_samples=1000, n_classes=3,
    ...                            n_informative=4, weights=[0.2, 0.3, 0.5],
    ...                            random_state=0)
    >>> clf = %(method_name)s(random_state=0)
    >>> clf.fit(X, y)  # doctest: +ELLIPSIS
    %(method_name)s(...)
    >>> clf.predict(X)  # doctest: +ELLIPSIS
    array([...])
    """.rstrip()


def _get_example_docstring(method_name: str):
    return _docstring_example % ({'method_name': method_name})


def _check_type_parameter(typ_str, typ_name, all_types):
    if not isinstance(typ_str, str):
        raise TypeError(f"'{typ_name}' should be string, got {type(typ_str)}.")
    if typ_str not in all_types:
        raise ValueError(
            f"'{typ_name}' should be one of {all_types}," f" got '{typ_str}'."
        )


def _get_random_state_docstring(sampling_type: str = None, **ignored_kwargs) -> str:
    if sampling_type is None:
        return _random_state_ensemble_docstring
    _check_type_parameter(sampling_type, 'sampling_type', SAMPLING_TYPES)
    return _random_state_ensemble_resampling_docstring


def _get_n_jobs_docstring(ensemble_type: str, **ignored_kwargs) -> str:
    _check_type_parameter(ensemble_type, 'ensemble_type', ENSEMBLE_TYPES)
    if ensemble_type == 'bagging' or ensemble_type == 'random-forest':
        return _n_jobs_fit_pred_docstring
    elif ensemble_type == 'general':
        return _n_jobs_pred_docstring
    elif ensemble_type == 'boosting':
        raise ValueError(
            f"boosting-like ensemble should not have 'n_jobs' parameter."
            f" Check your usage."
        )
    else:
        raise NotImplementedError(
            f"_get_n_jobs_docstring for 'ensemble_type' = {ensemble_type}"
            f" needs to be implemented."
        )


def _get_train_verbose_docstring(training_type: str, **ignored_kwargs) -> str:
    _check_type_parameter(training_type, 'training_type', TRAINING_TYPES)
    if training_type == 'iterative':
        return _train_verbose_iterative_docstring
    elif training_type == 'parallel':
        return _train_verbose_parallel_docstring
    else:
        raise NotImplementedError(
            f"_get_train_verbose_docstring for 'training_type' = {training_type}"
            f" needs to be implemented."
        )


def _get_target_label_docstring(sampling_type: str, **ignored_kwargs) -> str:
    _check_type_parameter(sampling_type, 'sampling_type', SAMPLING_TYPES)
    if sampling_type == 'under-sampling':
        return _target_label_under_sampling_docstring
    elif sampling_type == 'over-sampling':
        return _target_label_over_sampling_docstring
    else:
        raise NotImplementedError(
            f"_get_target_label_docstring for 'sampling_type' = {sampling_type}"
            f" needs to be implemented."
        )


def _get_n_target_samples_docstring(sampling_type: str, **ignored_kwargs) -> str:
    _check_type_parameter(sampling_type, 'sampling_type', SAMPLING_TYPES)
    if sampling_type == 'under-sampling':
        return _n_target_samples_under_sampling_docstring
    elif sampling_type == 'over-sampling':
        return _n_target_samples_over_sampling_docstring
    else:
        raise NotImplementedError(
            f"_get_n_target_samples_docstring for 'sampling_type' = {sampling_type}"
            f" needs to be implemented."
        )


PARAM_DOCSTRING_TYPE = (
    'n_jobs_sampler',
    'balancing_schedule',
    'eval_datasets',
    'eval_metrics',
    'random_state',
    'n_jobs',
    'warm_start',
    'train_verbose',
    'target_label',
    'n_target_samples',
    'early_termination',
)


def _get_parameter_docstring(param: str, **properties):
    _check_type_parameter(param, 'param', PARAM_DOCSTRING_TYPE)
    if param == 'n_jobs_sampler':
        return _n_jobs_sampler_docstring
    elif param == 'balancing_schedule':
        return _balancing_schedule_docstring
    elif param == 'eval_datasets':
        return _eval_datasets_docstring
    elif param == 'eval_metrics':
        return _eval_metrics_docstring
    elif param == 'warm_start':
        return _warm_start_docstring
    elif param == 'early_termination':
        return _early_termination_docstring
    elif param == 'random_state':
        return _get_random_state_docstring(**properties)
    elif param == 'n_jobs':
        return _get_n_jobs_docstring(**properties)
    elif param == 'train_verbose':
        return _get_train_verbose_docstring(**properties)
    elif param == 'target_label':
        return _get_target_label_docstring(**properties)
    elif param == 'n_target_samples':
        return _get_n_target_samples_docstring(**properties)
    else:
        raise NotImplementedError(
            f"_get_parameter_docstring for 'param' = {param}"
            f" needs to be implemented."
        )


# %%

if __name__ == "__main__":  # pragma: no cover

    for param in PARAM_DOCSTRING_TYPE:
        if param == 'random_state':
            for sampling_type in SAMPLING_TYPES:
                print(f"\n_get_random_state_docstring('{sampling_type}')\n")
                print(_get_parameter_docstring(param, sampling_type=sampling_type))
        elif param == 'n_jobs':
            for ensemble_type in ENSEMBLE_TYPES:
                print(f"\n_get_n_jobs_docstring('{ensemble_type}')\n")
                print(_get_parameter_docstring(param, ensemble_type=ensemble_type))
        elif param == 'train_verbose':
            for training_type in TRAINING_TYPES:
                print(f"\n_get_train_verbose_docstring('{training_type}')\n")
                print(_get_parameter_docstring(param, training_type=training_type))
        elif param == 'target_label':
            for sampling_type in SAMPLING_TYPES:
                print(f"\n_get_target_label_docstring('{sampling_type}')\n")
                print(_get_parameter_docstring(param, sampling_type=sampling_type))
        elif param == 'n_target_samples':
            for sampling_type in SAMPLING_TYPES:
                print(f"\n_get_n_target_samples_docstring('{sampling_type}')\n")
                print(_get_parameter_docstring(param, sampling_type=sampling_type))
        else:
            print(f"\n_get_parameter_docstring('{param}')\n")
            print(_get_parameter_docstring(param))

# %%
