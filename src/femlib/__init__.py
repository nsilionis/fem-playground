"""
femlib - Educational Finite Element Analysis Library.

A modular, object-oriented finite element library for educational
purposes, focusing on 2D isoparametric elements.
"""

__version__ = '0.1.0'

# Import main subpackages for convenience
from . import elements
from . import materials
from . import mesh
from . import postprocess

__all__ = [
    'elements',
    'materials',
    'mesh',
    'postprocess',
]
