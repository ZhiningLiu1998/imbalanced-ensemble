"""
=========================================================
Visualize an ensemble classifier
=========================================================

This example illustrates how to quickly visualize an 
:mod:`imbens.ensemble` classifier with
the :mod:`imbens.visualizer` module.

This example uses:

    - :class:`imbens.ensemble.SelfPacedEnsembleClassifier`
    - :class:`imbens.visualizer.ImbalancedEnsembleVisualizer`
"""

# Authors: Zhining Liu <zhining.liu@outlook.com>
# License: MIT


# %%
print(__doc__)

# Import imbalanced-ensemble
import imbens

# Import utilities
import sklearn
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42

# sphinx_gallery_thumbnail_number = 2

# %% [markdown]
# Prepare data
# ------------
# Make a toy 3-class imbalanced classification task.

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

# train valid split
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.5, stratify=y, random_state=RANDOM_STATE
)


# %% [markdown]
# Train an ensemble classifier
# ----------------------------
# Take ``SelfPacedEnsembleClassifier`` as example

# Initialize and train an SPE classifier
clf = imbens.ensemble.SelfPacedEnsembleClassifier(random_state=RANDOM_STATE).fit(
    X_train, y_train
)

# Store the fitted SelfPacedEnsembleClassifier
fitted_ensembles = {'SPE': clf}


# %% [markdown]
# Fit an ImbalancedEnsembleVisualizer
# -----------------------------------------------------

# Initialize visualizer
visualizer = imbens.visualizer.ImbalancedEnsembleVisualizer(
    eval_datasets={
        'training': (X_train, y_train),
        'validation': (X_valid, y_valid),
    },
)

# Fit visualizer
visualizer.fit(fitted_ensembles)


# %% [markdown]
# Plot performance curve
# ----------------------
# **performance w.r.t. number of base estimators**

fig, axes = visualizer.performance_lineplot()

# %% [markdown]
# Plot confusion matrix
# ---------------------

fig, axes = visualizer.confusion_matrix_heatmap(
    on_datasets=['validation'],  # only on validation set
    sup_title=False,
)
