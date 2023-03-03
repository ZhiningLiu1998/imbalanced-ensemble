"""
=============================================
Evaluate classification by compiling a report
=============================================

Specific metrics have been developed to evaluate classifier which has been
trained using imbalanced data. "mod:`imbens` provides a classification report
(:func:`imbens.metrics.classification_report_imbalanced`) 
similar to :mod:`sklearn`, with additional metrics specific to imbalanced
learning problem.
"""

# Adapted from imbalanced-learn
# Authors: Guillaume Lemaitre
# License: MIT

from sklearn import datasets
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split

from imbens.sampler import SMOTE
from imbens import pipeline as pl
from imbens.metrics import classification_report_imbalanced

print(__doc__)

RANDOM_STATE = 42

# sphinx_gallery_thumbnail_path = '../../docs/source/_static/thumbnail.png'

# Generate a dataset
X, y = datasets.make_classification(
    n_classes=2,
    class_sep=2,
    weights=[0.1, 0.9],
    n_informative=10,
    n_redundant=1,
    flip_y=0,
    n_features=20,
    n_clusters_per_class=4,
    n_samples=5000,
    random_state=RANDOM_STATE,
)

pipeline = pl.make_pipeline(
    SMOTE(random_state=RANDOM_STATE), LinearSVC(random_state=RANDOM_STATE)
)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=RANDOM_STATE)

# Train the classifier with balancing
pipeline.fit(X_train, y_train)

# Test the classifier and get the prediction
y_pred_bal = pipeline.predict(X_test)

# Show the classification report
print(classification_report_imbalanced(y_test, y_pred_bal))
