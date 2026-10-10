"""
Cross-sectional properties of structural elements.

Provides section classes holding the geometric properties of an
element's cross-section, separately from its material.
"""

from .section import PlaneSection, TrussSection

__all__ = [
    'PlaneSection',
    'TrussSection',
]
