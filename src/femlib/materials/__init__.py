"""
Material constitutive models.

Provides material classes for linear elastic and other constitutive
behaviours.
"""

from .elastic import LinearElastic2D, LinearElasticTruss
import warnings


# Backward compatibility with deprecation warning
class LinearElastic(LinearElastic2D):
    """
    Deprecated: Use LinearElastic2D instead.

    This alias is maintained for backward compatibility and will be
    removed in a future version.
    """
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "LinearElastic is deprecated, use LinearElastic2D instead",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__(*args, **kwargs)


__all__ = [
    'LinearElastic2D',
    'LinearElasticTruss',
    'LinearElastic',
]
