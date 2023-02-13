"""
The :mod:`imbens.ensemble._over_sampling` submodule contains 
a set of over-sampling-based ensemble imbalanced learning methods.
"""

from .over_boost import OverBoostClassifier
from .smote_boost import SMOTEBoostClassifier
from .kmeans_smote_boost import KmeansSMOTEBoostClassifier
from .smote_bagging import SMOTEBaggingClassifier
from .over_bagging import OverBaggingClassifier

__all__ = [
    "OverBoostClassifier",
    "SMOTEBoostClassifier",
    "KmeansSMOTEBoostClassifier",
    "OverBaggingClassifier",
    "SMOTEBaggingClassifier",
]
