"""
=========================================================
Basic usage example of ``imbalanced_ensemble``
=========================================================

This example shows the basic usage of the ensemble estimators 
(:class:`imbalanced_ensemble.ensemble.under_sampling.SelfPacedEnsembleClassifier`,
:class:`imbalanced_ensemble.ensemble.under_sampling.RUSBoostClassifier`,
:class:`imbalanced_ensemble.ensemble.under_sampling.EasyEnsembleClassifier`,
:class:`imbalanced_ensemble.ensemble.under_sampling.BalancedRandomForestClassifier`,
:class:`imbalanced_ensemble.ensemble.over_sampling.SMOTEBoostClassifier`,
:class:`imbalanced_ensemble.ensemble.over_sampling.OverBaggingClassifier`,
) in :mod:`imbalanced_ensemble.ensemble` module. 

We also show how to use the :class:`imbalanced_ensemble.visualizer.ImbalancedEnsembleVisualizer` 
to visualize and compare different ensemble classifiers.

"""

# Authors: Zhining Liu <zhining.liu@outlook.com>
# License: MIT

# %% [markdown]
# Import imbalanced_ensemble
# ----------------------------
#
# First, we will import necessary packages and implement 
# some utilities for data visualization

import imbalanced_ensemble as imbens
from imbalanced_ensemble.ensemble.under_sampling import (
    SelfPacedEnsembleClassifier,
    RUSBoostClassifier,
    EasyEnsembleClassifier,
    BalancedRandomForestClassifier,
)
from imbalanced_ensemble.ensemble.over_sampling import (
    SMOTEBoostClassifier,
    OverBaggingClassifier,
)

import pandas as pd
import time
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import KernelPCA
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

# implement some utilities for data visualization

vis_params = {
    'palette': plt.cm.rainbow,
    'cmap': plt.cm.rainbow,
    'edgecolor': 'black',
    'alpha': 0.6,
}

def set_ax_border(ax, border_color='black', border_width=2):
    for _, spine in ax.spines.items():
        spine.set_color(border_color)
        spine.set_linewidth(border_width)
        
    return ax

def plot_scatter(X, y, ax=None, weights=None, title='',
                 projection=None, vis_params=vis_params):
    if ax is None:
        ax = plt.axes()
    X_vis = projection.transform(X) if X.shape[1] > 2 else X
    title += ' (2D projection by {})'.format(
        str(projection.__class__).split('.')[-1][:-2]
    )
    size = 50 if weights is None else weights
    sns.scatterplot(x=X_vis[:, 0], y=X_vis[:, 1], 
        hue=y, style=y, s=size, **vis_params, legend='full', ax=ax)
    
    ax.set_title(title)
    ax = set_ax_border(ax, border_color='black', border_width=2)
    ax.grid(color='black', linestyle='-.', alpha=0.5)
    
    return ax

def plot_class_distribution(y, ax=None, title='', 
                            sort_values=False, plot_average=True):
    count = pd.DataFrame(list(Counter(y).items()), 
                         columns=['Class', 'Frequency'])
    if sort_values:
        count = count.sort_values(by='Frequency', ascending=False)
    if ax is None:
        ax = plt.axes()
    count.plot.bar(x='Class', y='Frequency', title=title, ax=ax)
    
    ax.set_title(title)
    ax = set_ax_border(ax, border_color='black', border_width=2)
    ax.grid(color='black', linestyle='-.', alpha=0.5, axis='y')

    if plot_average:
        ax.axhline(y=count['Frequency'].mean(),ls="dashdot",c="red")
        xlim_min, xlim_max, ylim_min, ylim_max = ax.axis()
        ax.text(
            x=xlim_min+(xlim_max-xlim_min)*0.82,
            y=count['Frequency'].mean()+(ylim_max-ylim_min)*0.03,
            c="red",s='Average')
    
    return ax

def plot_2Dprojection_and_cardinality(X, y, figsize=(10, 4), vis_params=vis_params,
                                     projection=None, weights=None, plot_average=True,
                                     title1='Dataset', title2='Class Distribution'):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    if projection == None:
        projection = KernelPCA(n_components=2).fit(X, y)
    ax1 = plot_scatter(X, y, ax=ax1, weights=weights, title=title1, 
                    projection=projection, vis_params=vis_params)
    ax2 = plot_class_distribution(y, ax=ax2, title=title2, 
                    sort_values=True, plot_average=plot_average)
    plt.tight_layout()
    return fig

# %% [markdown]
# Make a toy 3-class imbalanced classification task
# --------------------------------------------------

X, y = make_classification(n_classes=3, class_sep=2, # 3-class
    weights=[0.1, 0.3, 0.6], n_informative=3, n_redundant=1, flip_y=0,
    n_features=20, n_clusters_per_class=1, n_samples=2000, random_state=10)

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.5, random_state=42)

origin_distr = dict(Counter(y_train)) # {2: 600, 1: 300, 0: 100}
print('Original training dataset shape %s' % origin_distr)

# Visualize the dataset
projection = KernelPCA(n_components=2).fit(X, y)
fig = plot_2Dprojection_and_cardinality(X, y, projection=projection)
plt.show()

# %% [markdown]
# Train some imbalanced_ensemble classifiers
# --------------------------------------------------

# Set training parameters
init_kwargs = {
    'n_estimators': 50,
    'random_state': 10,
}
fit_kwargs = {
    'X': X_train,
    'y': y_train,
    'eval_datasets': {'valid': (X_valid, y_valid)},
    'eval_metrics': {
        'acc': (accuracy_score, {}),
        'balanced_acc': (balanced_accuracy_score, {}),
        'weighted_f1': (f1_score, {'average':'weighted'}),
    },
    'train_verbose': True,
}

# Train ensemble estimators
ensembles = {}

ensembles['spe'] = spe = SelfPacedEnsembleClassifier(**init_kwargs)
print ('Training {} ...'.format(spe.__name__))
start_time = time.time()
spe.fit(**fit_kwargs)
print ('Running time of {}.fit(): {:.4f}s\n'.format(
    spe.__name__, time.time() - start_time,
))

ensembles['rusboost'] = rusboost = RUSBoostClassifier(**init_kwargs)
print ('Training {} ...'.format(rusboost.__name__))
start_time = time.time()
rusboost.fit(**fit_kwargs)
print ('Running time of {}.fit(): {:.4f}s\n'.format(
    rusboost.__name__, time.time() - start_time,
))

ensembles['easyens'] = easyens = EasyEnsembleClassifier(**init_kwargs)
print ('Training {} ...'.format(easyens.__name__))
start_time = time.time()
easyens.fit(**fit_kwargs)
print ('Running time of {}.fit(): {:.4f}s\n'.format(
    easyens.__name__, time.time() - start_time,
))

ensembles['balanced_rf'] = balanced_rf = BalancedRandomForestClassifier(**init_kwargs)
print ('Training {} ...'.format(balanced_rf.__name__))
start_time = time.time()
balanced_rf.fit(**fit_kwargs)
print ('Running time of {}.fit(): {:.4f}s\n'.format(
    balanced_rf.__name__, time.time() - start_time,
))

ensembles['smoteboost'] = smoteboost = SMOTEBoostClassifier(**init_kwargs)
print ('Training {} ...'.format(smoteboost.__name__))
start_time = time.time()
smoteboost.fit(**fit_kwargs)
print ('Running time of {}.fit(): {:.4f}s\n'.format(
    smoteboost.__name__, time.time() - start_time,
))

ensembles['overbagging'] = overbagging = OverBaggingClassifier(**init_kwargs)
print ('Training {} ...'.format(overbagging.__name__))
start_time = time.time()
overbagging.fit(**fit_kwargs)
print ('Running time of {}.fit(): {:.4f}s\n'.format(
    overbagging.__name__, time.time() - start_time,
))


# %% [markdown]
# Visualize the results with ImbalancedEnsembleVisualizer
# -----------------------------------------------------------

from imbalanced_ensemble.visualizer import ImbalancedEnsembleVisualizer

# Fit visualizer
visualizer = ImbalancedEnsembleVisualizer().fit(
    ensembles = ensembles,
    granularity = 5,
)

# %% [markdown]
# plot performance curves w.r.t. number of base estimators

fig, axes = visualizer.performance_lineplot(
    n_samples_as_x_axis=False,
    alpha=0.6,
)
plt.show()

# %% [markdown]
# plot performance curves w.r.t. number of training samples
# split subfigures by datasets

fig, axes = visualizer.performance_lineplot(
    split_by=['dataset'],
    n_samples_as_x_axis=True,
    alpha=0.6,
)
plt.show()

# %% [markdown]
# plot performance curves w.r.t. number of training samples
# split subfigures by datasets

fig, axes = visualizer.performance_lineplot(
    split_by=['dataset'],
    n_samples_as_x_axis=False,
    alpha=0.6,
)
plt.show()

# %% [markdown]
# plot confusion matrices for selected methods/datasets

fig, axes = visualizer.confusion_matrix_heatmap(
    on_ensembles=['spe', 'smoteboost'],
    on_datasets=['valid'],
    sub_figsize=(4, 3.3),
)
plt.show()

# %% [markdown]
# plot confusion matrices for all methods/datasets

fig, axes = visualizer.confusion_matrix_heatmap(
    sub_figsize=(4, 3.3),
)
plt.show()

# %%