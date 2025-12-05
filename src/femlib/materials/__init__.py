"""
Material constitutive models.

Provides material classes for linear elastic and other constitutive
behaviours.
"""

from .elastic import LinearElastic

__all__ = [
    'LinearElastic',
]
