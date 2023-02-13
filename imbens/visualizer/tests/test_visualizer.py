"""Test visualizer."""

# Authors: Zhining Liu <zhining.liu@outlook.com>
# License: MIT


import pytest
import numpy as np
import matplotlib.pyplot as plt
from time import time
import sklearn
from sklearn.datasets import make_classification
from sklearn.model_selection import ParameterGrid, train_test_split

import imbens


RANDOM_STATE = 42
# make dataset
X, y = make_classification(
    n_classes=3,
    class_sep=2,
    weights=[0.1, 0.3, 0.6],
    n_informative=3,
    n_redundant=1,
    flip_y=0,
    n_features=20,
    n_clusters_per_class=2,
    n_samples=2000,
    random_state=0,
)
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.5, stratify=y, random_state=RANDOM_STATE
)


# Train all ensemble classifiers, store the results in fitted_ensembles
init_kwargs = {'n_estimators': 50, 'random_state': RANDOM_STATE}
fit_kwargs = {'X': X_train, 'y': y_train}
ensemble_dict = {
    'SPE': imbens.ensemble.SelfPacedEnsembleClassifier(**init_kwargs),
    'EasyEns': imbens.ensemble.EasyEnsembleClassifier(**init_kwargs),
    'BalanceForest': imbens.ensemble.BalancedRandomForestClassifier(**init_kwargs),
    'SMOTEBagging': imbens.ensemble.SMOTEBaggingClassifier(**init_kwargs),
}
fitted_ensembles = {}
for clf_name, clf in ensemble_dict.items():
    start_time = time()
    clf.fit(**fit_kwargs)
    fit_time = time() - start_time
    fitted_ensembles[clf_name] = clf
    print('Training {:^30s} | Time used: {:.3f}s'.format(clf.__name__, fit_time))

visualizer = imbens.visualizer.ImbalancedEnsembleVisualizer(
    eval_datasets={
        'training': (X_train, y_train),
        'validation': (X_valid, y_valid),
    },
    eval_metrics={
        'acc': (sklearn.metrics.accuracy_score, {}),
        'balanced_acc': (sklearn.metrics.balanced_accuracy_score, {}),
        'weighted_f1': (sklearn.metrics.f1_score, {'average': 'weighted'}),
    },
)
# Fit visualizer
visualizer.fit(fitted_ensembles)


def test_performance_curve():
    # Performance w.r.t. number of base estimators
    fig, axes = visualizer.performance_lineplot()
    # Performance w.r.t. number of training samples
    fig, axes = visualizer.performance_lineplot(
        n_samples_as_x_axis=True,
    )
    # Select results for visualization
    fig, axes = visualizer.performance_lineplot(
        on_ensembles=['SPE', 'EasyEns', 'BalanceForest'],
        on_datasets=['validation'],
        on_metrics=['balanced_acc', 'weighted_f1'],
        n_samples_as_x_axis=True,
    )
    # Custom visualization
    fig, axes = visualizer.performance_lineplot(
        on_ensembles=['SPE', 'EasyEns', 'BalanceForest'],
        on_datasets=['training', 'validation'],
        on_metrics=['balanced_acc', 'weighted_f1'],
        n_samples_as_x_axis=True,
        # Customize visual appearance
        sub_figsize=(3, 4),
        sup_title='My Suptitle',
        # arguments pass down to seaborn.lineplot()
        linewidth=3,
        markers=True,
        alpha=0.8,
    )
    # Group results by dataset
    fig, axes = visualizer.performance_lineplot(
        on_ensembles=['SPE', 'EasyEns', 'BalanceForest'],
        on_datasets=['training', 'validation'],
        on_metrics=['balanced_acc', 'weighted_f1'],
        n_samples_as_x_axis=True,
        sub_figsize=(3, 2.3),
        split_by=['dataset'],  # Group results by dataset
    )
    # Group results by method
    fig, axes = visualizer.performance_lineplot(
        on_ensembles=['SPE', 'EasyEns', 'BalanceForest'],
        on_datasets=['training', 'validation'],
        on_metrics=['balanced_acc', 'weighted_f1'],
        n_samples_as_x_axis=True,
        sub_figsize=(3, 2.3),
        split_by=['method'],  # Group results by method
    )


def test_confusion_matrix_heatmap():
    fig, axes = visualizer.confusion_matrix_heatmap()
    # False predictions only
    fig, axes = visualizer.confusion_matrix_heatmap(
        false_pred_only=True,
    )
    # Select: method ('SPE', 'BalanceForest'), data ('validation')
    fig, axes = visualizer.confusion_matrix_heatmap(
        on_ensembles=['SPE', 'BalanceForest'],
        on_datasets=['validation'],
    )
    # Customize visual appearance
    fig, axes = visualizer.confusion_matrix_heatmap(
        on_ensembles=['SPE', 'BalanceForest'],
        on_datasets=['training', 'validation'],
        # Customize visual appearance
        sub_figsize=(4, 3.3),
        sup_title='My Suptitle',
        # arguments pass down to seaborn.heatmap()
        cmap='YlOrRd',
        cbar=True,
        linewidths=10,
        vmax=20,
    )


@pytest.mark.parametrize(
    "vis_params",
    ParameterGrid(
        {
            "on_ensembles": [
                ['SPE'],
            ],
            "on_datasets": [
                ['validation'],
            ],
            "on_metrics": [
                ['acc'],
            ],
            "n_samples_as_x_axis": [False, True],
            "split_by": [
                ['method'],
                ['method', 'dataset'],
            ],
        }
    ),
)
def test_performance_curve_grid(vis_params):
    fig, axes = visualizer.performance_lineplot(**vis_params)


@pytest.mark.parametrize(
    "vis_params",
    ParameterGrid(
        {
            "on_ensembles": [
                ['SPE'],
            ],
            "on_datasets": [
                ['validation'],
            ],
            "false_pred_only": [False, True],
        }
    ),
)
def test_confusion_matrix_heatmap_grid(vis_params):
    fig, axes = visualizer.confusion_matrix_heatmap(**vis_params)
