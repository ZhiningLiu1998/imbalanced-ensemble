"""
=========================================================
Customize ensemble training log
=========================================================

This example illustrates how to enable and customize the training 
log when training an :mod:`imbens.ensemble` classifier.

This example uses:

    - :class:`imbens.ensemble.SelfPacedEnsembleClassifier`
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

# sphinx_gallery_thumbnail_path = '../../docs/source/_static/training_log_thumbnail.png'

# %% [markdown]
# Prepare data
# ----------------------------
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
# Customize training log
# ---------------------------------------------------------------------------
# Take ``SelfPacedEnsembleClassifier`` as example, training log is controlled by 3 parameters of the ``fit()`` method:
#
#   - ``eval_datasets``: Dataset(s) used for evaluation during the ensemble training.
#   - ``eval_metrics``: Metric(s) used for evaluation during the ensemble training.
#   - ``train_verbose``: Controls the granularity and content of the training log.

clf = imbens.ensemble.SelfPacedEnsembleClassifier(random_state=RANDOM_STATE)

# %% [markdown]
# Set training log format
# -----------------------
# (``fit()`` parameter: ``train_verbose``: bool, int or dict)

# %% [markdown]
# **Enable auto training log**

clf.fit(
    X_train,
    y_train,
    train_verbose=True,
)


# %% [markdown]
# **Customize training log granularity**

clf.fit(
    X_train,
    y_train,
    train_verbose={
        'granularity': 10,
    },
)


# %% [markdown]
# **Customize training log content column**

clf.fit(
    X_train,
    y_train,
    train_verbose={
        'granularity': 10,
        'print_distribution': False,
        'print_metrics': True,
    },
)


# %% [markdown]
# Add additional evaluation dataset(s)
# ------------------------------------
# (``fit()`` parameter: ``eval_datasets``: dict)

clf.fit(
    X_train,
    y_train,
    eval_datasets={
        'valid': (X_valid, y_valid),  # add validation data
    },
    train_verbose={
        'granularity': 10,
    },
)


# %% [markdown]
# Specify evaluation metric(s)
# ----------------------------
# (``fit()`` parameter: ``eval_metrics``: dict)

clf.fit(
    X_train,
    y_train,
    eval_datasets={
        'valid': (X_valid, y_valid),
    },
    eval_metrics={
        'weighted_f1': (
            sklearn.metrics.f1_score,
            {'average': 'weighted'},
        ),  # use weighted_f1
    },
    train_verbose={
        'granularity': 10,
    },
)

# %%
