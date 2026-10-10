"""
Material constitutive models.

Provides material classes for linear elastic behaviour. Cross-sectional
properties are held separately, in femlib.sections.
"""

from .elastic import LinearElastic1D, LinearElastic2D

__all__ = [
    'LinearElastic1D',
    'LinearElastic2D',
]
