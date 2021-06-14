Release History
***************

Version 0.1.4 (2021.06)
=========================

Enhancement: 

- :func:`imbalanced_ensemble.visualizer.ImbalancedEnsembleVisualizer.performance_lineplot`: add option ``on_metrics`` to select evaluation metrics to include in the plot. 
- :func:`imbalanced_ensemble.visualizer.ImbalancedEnsembleVisualizer.confusion_matrix_heatmap`: add option ``false_pred_only`` to control whether to plot only the false predictions in the confusion matrix.
- Add some utilities for data visualization in :mod:`imbalanced_ensemble.utils._plot`.


Documentation:

- Add more comprehensive examples in the `examples gallery <https://imbalanced-ensemble.readthedocs.io/en/latest/auto_examples/index.html#>`_ (11 new, 16 in total).
- Add a `Chinese README <https://github.com/ZhiningLiu1998/imbalanced-ensemble/blob/main/docs/README_CN.md>`_.

Maintenance:

- :func:`imbalanced_ensemble.utils.testing.all_estimators` now support ``'ensemble'`` type_filter.
- Renamed some functions in :mod:`imbalanced_ensemble.utils._validation_param` to improve readability

Bug Fixes:

- Fixed typo bugs in:
  
  - :class:`imbalanced_ensemble.ensemble.KmeansSMOTEBoostClassifier`
  - :class:`imbalanced_ensemble.ensemble.SMOTEBoostClassifier`
  - :class:`imbalanced_ensemble.ensemble.SMOTEBaggingClassifier`

Version 0.1.3 (2021.06)
=========================

Bug Fixes:

- Fixed a typo bug in :class:`imbalanced_ensemble.ensemble.BalanceCascadeClassifier`.
- Fixed an import Error in :class:`imbalanced_ensemble.ensembleCompatibleAdaBoostClassifier`.

Version 0.1.2 (2021.05)
=========================


Enhancement: 

- Add support for metric functions that take probability as input.
- Boosting-based classifiers now will print a message when the training is early terminated.
- :func:`imbalanced_ensemble.visualizer.ImbalancedEnsembleVisualizer.performance_lineplot`: ``granularity`` now can be automatically set.

Maintenance:

- All ensemble classifiers now can be directly imported from the :mod:`imbalanced_ensemble.ensemble` module.
- The default value of ``train_verbose`` of ``Classifier.fit()``: ``True`` -> ``False``.
- The default value of ``n_estimators`` of ``Classifier.__init__()``: 50 for all ensemble classifiers.
- The default value of ``granularity`` of ``Visualizer.fit()``: 5 -> ``None`` (automatically determined).
- :func:`imbalanced_ensemble.visualizer.ImbalancedEnsembleVisualizer.confusion_matrix_heatmap`: swap rows and columns, now rows/columns correspond to datasets/methods.

Bug Fixes:

- Fixed ``ZeroDivisionError`` when using :class:`imbalanced_ensemble.sampler.under_sampling.SelfPacedUnderSampler`.


Version 0.1.1 (2021.05)
=========================

Bug Fixes:

- Unexpected print messages when using the :mod:`imbalanced_ensemble.pipeline` module.

Version 0.1.0 (2021.05)
=========================

Initial release.