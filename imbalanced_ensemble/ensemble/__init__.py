"""
The :mod:`imbalanced_ensemble.ensemble` module contains a set of 
ensemble imbalanced learning methods.
"""

from . import under_sampling
from . import over_sampling
from . import reweighting
from . import compatible

from .under_sampling import SelfPacedEnsembleClassifier
from .under_sampling import BalanceCascadeClassifier
from .under_sampling import BalancedRandomForestClassifier
from .under_sampling import EasyEnsembleClassifier
from .under_sampling import RUSBoostClassifier
from .under_sampling import UnderBaggingClassifier

from .over_sampling import OverBoostClassifier
from .over_sampling import SMOTEBoostClassifier
from .over_sampling import KmeansSMOTEBoostClassifier
from .over_sampling import SMOTEBaggingClassifier
from .over_sampling import OverBaggingClassifier

from .reweighting import AdaCostClassifier
from .reweighting import AdaUBoostClassifier
from .reweighting import AsymBoostClassifier

from .compatible import CompatibleAdaBoostClassifier
from .compatible import CompatibleBaggingClassifier

__all__ = [
    "under_sampling",
    "over_sampling",
    "reweighting",
    "compatible",

    "SelfPacedEnsembleClassifier",
    "BalanceCascadeClassifier",
    "BalancedRandomForestClassifier",
    "EasyEnsembleClassifier",
    "RUSBoostClassifier",
    "UnderBaggingClassifier",
    
    "OverBoostClassifier",
    "SMOTEBoostClassifier",
    "KmeansSMOTEBoostClassifier",
    "OverBaggingClassifier",
    "SMOTEBaggingClassifier",
    
    "AdaCostClassifier",
    "AdaUBoostClassifier",
    "AsymBoostClassifier",
    
    "CompatibleAdaBoostClassifier",
    "CompatibleBaggingClassifier",
]