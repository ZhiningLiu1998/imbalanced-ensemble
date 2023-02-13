"""Test for the deprecation helper"""

# Authors: Guillaume Lemaitre
# License: MIT

import pytest

from imbens.utils.deprecation import deprecate_parameter


class Sampler:
    def __init__(self):
        self.a = "something"
        self.b = "something"


def test_deprecate_parameter():
    with pytest.warns(DeprecationWarning, match="is deprecated from"):
        deprecate_parameter(Sampler(), "0.2", "a")
    with pytest.warns(DeprecationWarning, match="Use 'b' instead."):
        deprecate_parameter(Sampler(), "0.2", "a", "b")
