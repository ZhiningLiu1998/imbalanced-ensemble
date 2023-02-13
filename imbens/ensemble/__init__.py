"""
The :mod:`imbens.ensemble` module contains a set of 
ensemble imbalanced learning methods.
"""

from . import _under_sampling
from . import _over_sampling
from . import _reweighting
from . import _compatible

from ._under_sampling import SelfPacedEnsembleClassifier
from ._under_sampling import BalanceCascadeClassifier
from ._under_sampling import BalancedRandomForestClassifier
from ._under_sampling import EasyEnsembleClassifier
from ._under_sampling import RUSBoostClassifier
from ._under_sampling import UnderBaggingClassifier

from ._over_sampling import OverBoostClassifier
from ._over_sampling import SMOTEBoostClassifier
from ._over_sampling import KmeansSMOTEBoostClassifier
from ._over_sampling import SMOTEBaggingClassifier
from ._over_sampling import OverBaggingClassifier

from ._reweighting import AdaCostClassifier
from ._reweighting import AdaUBoostClassifier
from ._reweighting import AsymBoostClassifier

from ._compatible import CompatibleAdaBoostClassifier
from ._compatible import CompatibleBaggingClassifier

__all__ = [
    "_under_sampling",
    "_over_sampling",
    "_reweighting",
    "_compatible",

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