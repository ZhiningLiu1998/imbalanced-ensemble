"""
The :mod:`imbens.ensemble._under_sampling` submodule contains 
a set of under-sampling-based ensemble imbalanced learning methods.
"""

from .self_paced_ensemble import SelfPacedEnsembleClassifier
from .balance_cascade import BalanceCascadeClassifier
from .balanced_random_forest import BalancedRandomForestClassifier
from .easy_ensemble import EasyEnsembleClassifier
from .rus_boost import RUSBoostClassifier
from .under_bagging import UnderBaggingClassifier

__all__ = [
    "SelfPacedEnsembleClassifier",
    "BalanceCascadeClassifier",
    "BalancedRandomForestClassifier",
    "EasyEnsembleClassifier",
    "RUSBoostClassifier",
    "UnderBaggingClassifier",
]
