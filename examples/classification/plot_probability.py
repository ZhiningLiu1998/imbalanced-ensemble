"""
=================================================================
Plot probabilities with different base classifiers
=================================================================

Plot the classification probability for ensemble models with different base classifiers. 

We use a 3-class imbalanced dataset, and we classify it with a ``SelfPacedEnsembleClassifier`` (ensemble size = 5).
We use Decision Tree, Support Vector Machine (rbf kernel), and Gaussian process classifier as the base classifier.

This example uses:

    - :class:`imbens.ensemble.SelfPacedEnsembleClassifier`
"""

# Adapted from sklearn
# Author: Zhining Liu <zhining.liu@outlook.com>
#         Alexandre Gramfort <alexandre.gramfort@inria.fr>
# License: BSD 3 clause

# %%
print(__doc__)

# Import imbalanced-ensemble
import imbens

# Import utilities
import numpy as np
from collections import Counter
import sklearn
from imbens.datasets import make_imbalance
from imbens.ensemble.base import sort_dict_by_key

RANDOM_STATE = 42

# %% [markdown]
# Preparation
# -----------
# **Make 3 imbalanced iris classification tasks.**

iris = sklearn.datasets.load_iris()
X = iris.data[:, 0:2]  # we only take the first two features for visualization
y = iris.target

X, y = make_imbalance(
    X, y, sampling_strategy={0: 50, 1: 30, 2: 10}, random_state=RANDOM_STATE
)
print(
    'Class distribution of imbalanced iris dataset: \n%s' % sort_dict_by_key(Counter(y))
)


# %% [markdown]
# **Create SPE (ensemble size = 5) with different base classifiers.**

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

classifiers = {
    'SPE-DT': imbens.ensemble.SelfPacedEnsembleClassifier(
        n_estimators=5,
        estimator=DecisionTreeClassifier(),
    ),
    'SPE-SVM-rbf': imbens.ensemble.SelfPacedEnsembleClassifier(
        n_estimators=5,
        estimator=SVC(kernel='rbf', probability=True),
    ),
    'SPE-GPC': imbens.ensemble.SelfPacedEnsembleClassifier(
        n_estimators=5,
        estimator=GaussianProcessClassifier(1.0 * RBF([1.0, 1.0])),
    ),
}

n_classifiers = len(classifiers)


# %% [markdown]
# Plot classification probabilities
# ---------------------------------

import matplotlib.pyplot as plt

n_features = X.shape[1]

plt.figure(figsize=(3 * 2, n_classifiers * 2))
plt.subplots_adjust(bottom=0.2, top=0.95)

xx = np.linspace(3, 9, 100)
yy = np.linspace(1, 5, 100).T
xx, yy = np.meshgrid(xx, yy)
Xfull = np.c_[xx.ravel(), yy.ravel()]

for index, (name, classifier) in enumerate(classifiers.items()):
    classifier.fit(X, y)

    y_pred = classifier.predict(X)
    accuracy = sklearn.metrics.balanced_accuracy_score(y, y_pred)
    print("Balanced Accuracy (train) for %s: %0.1f%% " % (name, accuracy * 100))

    # View probabilities:
    probas = classifier.predict_proba(Xfull)
    n_classes = np.unique(y_pred).size
    for k in range(n_classes):
        plt.subplot(n_classifiers, n_classes, index * n_classes + k + 1)
        plt.title("Class %d" % k)
        if k == 0:
            plt.ylabel(name)
        imshow_handle = plt.imshow(
            probas[:, k].reshape((100, 100)), extent=(3, 9, 1, 5), origin='lower'
        )
        plt.xticks(())
        plt.yticks(())
        idx = y_pred == k
        if idx.any():
            plt.scatter(X[idx, 0], X[idx, 1], marker='o', c='w', edgecolor='k')

ax = plt.axes([0.15, 0.04, 0.7, 0.05])
plt.title("Probability")
plt.colorbar(imshow_handle, cax=ax, orientation='horizontal')
plt.show()
