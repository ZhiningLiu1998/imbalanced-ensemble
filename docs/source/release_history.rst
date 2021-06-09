Release History
***************

Version 0.1.3 (2021.06)
=========================

Fix:

- Fixed a typo bug in ``BalanceCascadeClassifier``
- Fixed an import Error in ``CompatibleAdaBoostClassifier``

Version 0.1.2 (2021.05)
=========================

Fix:

- Fixed ``ZeroDivisionError`` when using ``SelfPacedUnderSampler``
- All ensemble classifiers now can be directly imported from the ``imbalanced_ensemble.ensemble`` module
- Boosting-based classifiers now will print a message when the training is early terminated

Enhancement: 

- Add support for metric functions that take probability as input
- ``granularity`` used in the ``Visualizer.performance_lineplot()`` method now can be automatically set.

Change:

- The default value of ``train_verbose`` of ``Classifier.fit()``: ``True`` -> ``False``
- The default value of ``n_estimators`` of ``Classifier.__init__()``: 50 for all ensemble classifiers
- The default value of ``granularity`` of ``Visualizer.fit()``: 5 -> ``None`` (automatically determined)
- ``Visualizer.confusion_matrix_heatmap()``: swap rows and columns, now rows/columns correspond to datasets/methods.

Version 0.1.1 (2021.05)
=========================

Fix:

- Unexpected print messages when using the ``imbalanced_ensemble.pipeline`` module.

Version 0.1.0 (2021.05)
=========================

Initial release.