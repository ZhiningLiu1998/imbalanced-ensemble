"""
The :mod:`imbens.ensemble.compatible` submodule contains 
a set of `sklearn.ensemble` learning methods that were re-implemented in 
`imbens` style.
"""

from .adaboost_compatible import CompatibleAdaBoostClassifier
from .bagging_compatible import CompatibleBaggingClassifier

__all__ = [
    "CompatibleAdaBoostClassifier",
    "CompatibleBaggingClassifier",
]
