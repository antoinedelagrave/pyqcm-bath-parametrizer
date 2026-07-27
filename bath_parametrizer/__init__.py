"""Parametrizing the bath automatically using character
table and generators of an abelian point group (Pyqcm formalism).
"""

from .bath_parametrization import BathParametrizer
from .point_groups import all_point_groups

# Defines what modules/submodules are imported
# when the wildcard import * is used!
__all__  = ["BathParametrizer", "all_point_groups"]
