"""
=========================================================
Train and predict with an ensemble classifier
=========================================================

This example shows the basic usage of an 
:mod:`imbens.ensemble` classifier.

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
from collections import Counter
import sklearn
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from imbens.ensemble.base import sort_dict_by_key

# Import plot utilities
import matplotlib.pyplot as plt
from imbens.utils._plot import plot_2Dprojection_and_cardinality

RANDOM_STATE = 42

# %% [markdown]
# Prepare & visualize the data
# ----------------------------
# Make a toy 3-class imbalanced classification task.

# Generate and split a synthetic dataset
X, y = make_classification(
    n_classes=3,
    n_samples=2000,
    class_sep=2,
    weights=[0.1, 0.3, 0.6],
    n_informative=3,
    n_redundant=1,
    flip_y=0,
    n_features=20,
    n_clusters_per_class=2,
    random_state=RANDOM_STATE,
)
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.5, stratify=y, random_state=RANDOM_STATE
)

# Visualize the training dataset
fig = plot_2Dprojection_and_cardinality(X_train, y_train, figsize=(8, 4))
plt.show()

# Print class distribution
print('Training dataset distribution    %s' % sort_dict_by_key(Counter(y_train)))
print('Validation dataset distribution  %s' % sort_dict_by_key(Counter(y_valid)))

# %% [markdown]
# Using ensemble classifiers in ``imbens``
# -----------------------------------------------------
# Take ``SelfPacedEnsembleClassifier`` as example

# Initialize an SelfPacedEnsembleClassifier
clf = imbens.ensemble.SelfPacedEnsembleClassifier(random_state=RANDOM_STATE)

# Train an SelfPacedEnsembleClassifier
clf.fit(X_train, y_train)

# Make predictions
y_pred_proba = clf.predict_proba(X_valid)
y_pred = clf.predict(X_valid)

# Evaluate
balanced_acc_score = sklearn.metrics.balanced_accuracy_score(y_valid, y_pred)
print(f'SPE: ensemble of {clf.n_estimators} {clf.estimator_}')
print('Validation Balanced Accuracy: {:.3f}'.format(balanced_acc_score))


# %% [markdown]
# Set the ensemble size
# ---------------------
# (parameter ``n_estimators``: int)

from imbens.ensemble import SelfPacedEnsembleClassifier as SPE
from sklearn.metrics import balanced_accuracy_score

clf = SPE(
    n_estimators=5,  # Set ensemble size to 5
    random_state=RANDOM_STATE,
).fit(X_train, y_train)

# Evaluate
balanced_acc_score = balanced_accuracy_score(y_valid, clf.predict(X_valid))
print(f'SPE: ensemble of {clf.n_estimators} {clf.estimator_}')
print('Validation Balanced Accuracy: {:.3f}'.format(balanced_acc_score))


# %% [markdown]
# Use different base estimator
# ----------------------------
# (parameter ``estimator``: estimator object)

from sklearn.svm import SVC

clf = SPE(
    n_estimators=5,
    estimator=SVC(probability=True),  # Use SVM as the base estimator
    random_state=RANDOM_STATE,
).fit(X_train, y_train)

# Evaluate
balanced_acc_score = balanced_accuracy_score(y_valid, clf.predict(X_valid))
print(f'SPE: ensemble of {clf.n_estimators} {clf.estimator_}')
print('Validation Balanced Accuracy: {:.3f}'.format(balanced_acc_score))


# %% [markdown]
# Enable training log
# -------------------
# (``fit()`` parameter ``train_verbose``: bool, int or dict)

clf = SPE(random_state=RANDOM_STATE).fit(
    X_train,
    y_train,
    train_verbose=True,  # Enable training log
)
