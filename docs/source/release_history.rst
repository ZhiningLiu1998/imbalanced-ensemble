Release History
***************


Version 0.2.2 (2024.06)
=========================

Maintenance:

- Bump dependency version: scikit-learn to ``1.5.0``, pandas to ``2.1.1``, seaborn to ``0.13.2``.
- Fix numerous Errors and Warnings appeared in the CI.

Version 0.2.1 (2023.07)
=========================

Maintenance:

- Bump supported scikit-learn version to ``1.3.0``.
- Update requirements for building documentation with ``sphinx``.
- Remove redundant doc (``auto_examples`` and ``back_references``) files in the source distribution.

Bug Fixes:

- Fix ``AttributeError`` in :class:`imbens.ensemble.BalancedRandomForestClassifier`.
- Fix several bugs encountered in CI.

Version 0.2.0 (2023.02)
=========================

Enhancement:

- Enable CircleCI with CodeCov report.
- Easier usage:
  
  - the package is now imported as ``imbens``
  - all samplers can be directly accessed in ``imbens.sampler``

Maintenance:

- Complement unit tests (59% -> 96% coverage).
- Set default ``k_neighbors=1`` for SMOTEBagging to prevent error in few-shot cases.
- Set default ``cluster_balance_threshold=0.1`` for KmeansSMOTEBoost to prevent error in few-shot cases.
- Add ``decision_function()`` for supported ensemble classifiers.
- The parameter ``base_sampler`` is renamed to ``sampler``.
- The attribute ``base_sampler_`` is renamed to ``sampler_``.
- Bump supported Python version to ``3.8, 3.9, 3.10, 3.11``.
- Following sklearn version >1.2, for all ensemble classifiers, 

  - the parameter ``base_estimator`` is renamed to ``estimator``.
  - the attribute ``base_estimator_`` is renamed to ``estimator_``.

Bug Fixes:

- Add missing comma in the INSTALL_REQUIRES list which breaks ``conda env export``.
- Fix ``BalanceCascade`` and ``SelfPacedEnsemble``'s ``_make_sampler()`` behaviour.
- Fix ``BalanceCascade`` and ``SelfPacedEnsemble`` parameter check.
- Fix cost_matrix type check for cost-sensitive methods
- Fix ``SVMSMOTE`` with ``sample_weight``
- Fix ``CompatibleAdaBoost`` with ``train_verbose``
- Fix samplers in/output type consistency


Version 0.1.7 (2022.01)
=========================

Enhancement: 

- Add ``feature_importances_`` attribute for supported methods:

  - :class:`imbens.ensemble.AdaCostClassifier`
  - :class:`imbens.ensemble.AdaUBoostClassifier`
  - :class:`imbens.ensemble.AsymBoostClassifier`
  - :class:`imbens.ensemble.BalanceCascadeClassifier`
  - :class:`imbens.ensemble.BalancedRandomForestClassifier`
  - :class:`imbens.ensemble.CompatibleAdaBoostClassifier`
  - :class:`imbens.ensemble.KmeansSMOTEBoostClassifier`
  - :class:`imbens.ensemble.OverBoostClassifier`
  - :class:`imbens.ensemble.RUSBoostClassifier`
  - :class:`imbens.ensemble.SMOTEBoostClassifier`
  - :class:`imbens.ensemble.SelfPacedEnsembleClassifier`

Documentation:

- Paper describing this package "`IMBENS: Ensemble Class-imbalanced Learning in Python <https://arxiv.org/abs/2111.12776>`_".


Version 0.1.6 (2021.11)
=========================

Enhancement: 

- All boosting-based methods now support ``early_termination``, which can be used to enable/disable strict early termination for Adaboost training.
- Add utility functions :func:`imbens.datasets.generate_imbalance_data` and :func:`imbens.utils.evaluate_print` to ease the test and evaluation.

Bug Fixes:

- Fixed Resampling + Bagging models (e.g., `OverBagging`) raise error when used with base estimators that do not support `sample_weight` (e.g., `sklearn.KNeighborsClassifier`). 
- Fixed AttributeError occurs when initializing bagging-based models.


Version 0.1.5 (2021.08)
=========================

Enhancement: 

- :class:`imbens.sampler.RandomUnderSampler` now support ``sample_proba`` (the probability of each instance being sampled, not ``sample_weight``).

Bug Fixes:

- Fixed ValueError when using :class:`imbens.visualizer.ImbalancedEnsembleVisualizer` with ``seaborn`` v0.11.2.
- Fixed all ensemble algorithms (error or performance issue) when the classification targets do not begin with 0.


Version 0.1.4 (2021.06)
=========================

Enhancement: 

- :func:`imbens.visualizer.ImbalancedEnsembleVisualizer.performance_lineplot`: add option ``on_metrics`` to select evaluation metrics to include in the plot. 
- :func:`imbens.visualizer.ImbalancedEnsembleVisualizer.confusion_matrix_heatmap`: add option ``false_pred_only`` to control whether to plot only the false predictions in the confusion matrix.
- Add some utilities for data visualization in :mod:`imbens.utils._plot`.


Documentation:

- Add more comprehensive examples in the `examples gallery <https://imbalanced-ensemble.readthedocs.io/en/latest/auto_examples/index.html#>`_ (11 new, 16 in total).
- Add a `Chinese README <https://github.com/ZhiningLiu1998/imbalanced-ensemble/blob/main/docs/README_CN.md>`_.

Maintenance:

- :func:`imbens.utils.testing.all_estimators` now support ``'ensemble'`` type_filter.
- Renamed some functions in :mod:`imbens.utils._validation_param` to improve readability

Bug Fixes:

- Fixed typo bugs in:
  
  - :class:`imbens.ensemble.KmeansSMOTEBoostClassifier`
  - :class:`imbens.ensemble.SMOTEBoostClassifier`
  - :class:`imbens.ensemble.SMOTEBaggingClassifier`


Version 0.1.3 (2021.06)
=========================

Bug Fixes:

- Fixed a typo bug in :class:`imbens.ensemble.BalanceCascadeClassifier`.
- Fixed an import Error in :class:`imbens.ensembleCompatibleAdaBoostClassifier`.


Version 0.1.2 (2021.05)
=========================

Enhancement: 

- Add support for metric functions that take probability as input.
- Boosting-based classifiers now will print a message when the training is early terminated.
- :func:`imbens.visualizer.ImbalancedEnsembleVisualizer.performance_lineplot`: ``granularity`` now can be automatically set.

Maintenance:

- All ensemble classifiers now can be directly imported from the :mod:`imbens.ensemble` module.
- The default value of ``train_verbose`` of ``Classifier.fit()``: ``True`` -> ``False``.
- The default value of ``n_estimators`` of ``Classifier.__init__()``: 50 for all ensemble classifiers.
- The default value of ``granularity`` of ``Visualizer.fit()``: 5 -> ``None`` (automatically determined).
- :func:`imbens.visualizer.ImbalancedEnsembleVisualizer.confusion_matrix_heatmap`: swap rows and columns, now rows/columns correspond to datasets/methods.

Bug Fixes:

- Fixed ``ZeroDivisionError`` when using :class:`imbens.sampler.SelfPacedUnderSampler`.


Version 0.1.1 (2021.05)
=========================

Bug Fixes:

- Unexpected print messages when using the :mod:`imbens.pipeline` module.


Version 0.1.0 (2021.05)
=========================

Initial release.