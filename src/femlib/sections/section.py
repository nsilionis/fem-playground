"""
Cross-sectional properties of structural elements.

This module holds the geometric properties of an element's
cross-section. Constitutive properties are held separately, in
femlib.materials.
"""


class PlaneSection:
    """
    Out-of-plane thickness of a 2D continuum element.

    Parameters
    ----------
    thickness : float
        Thickness in the out-of-plane direction [m].

    Raises
    ------
    ValueError
        If thickness is not positive.

    Examples
    --------
    >>> plate = PlaneSection(thickness=0.01)
    """

    def __init__(self, thickness):
        """Initialise plane section."""
        if thickness <= 0:
            raise ValueError(
                f"Thickness must be positive, got thickness={thickness}"
            )
        self.thickness = float(thickness)

    def __repr__(self):
        """String representation of section."""
        return f"PlaneSection(thickness={self.thickness:.3e} m)"


class TrussSection:
    """
    Cross-sectional area of a truss (bar) element.

    Parameters
    ----------
    A : float
        Cross-sectional area [m²].

    Raises
    ------
    ValueError
        If A is not positive.

    Examples
    --------
    >>> bar = TrussSection(A=0.001)
    """

    def __init__(self, A):
        """Initialise truss section."""
        if A <= 0:
            raise ValueError(
                f"Cross-sectional area must be positive, got A={A}"
            )
        self.A = float(A)

    def __repr__(self):
        """String representation of section."""
        return f"TrussSection(A={self.A:.6f} m²)"


class BeamSection:
    """
    Second moment of area of a beam cross-section.

    Parameters
    ----------
    I : float or callable
        Second moment of area about the bending axis [m^4]. A function
        must accept a numpy.ndarray of positions x and return an array
        of the same shape.

    Raises
    ------
    ValueError
        If a constant I is not positive.

    Examples
    --------
    >>> section = BeamSection(I=8.0e-6)
    """

    def __init__(self, I):  # noqa: E741
        """Initialise beam section."""
        if not callable(I) and I <= 0:
            raise ValueError(
                f"Second moment of area must be positive, got I={I}"
            )
        self.I = I if callable(I) else float(I)  # noqa: E741

    def __repr__(self):
        """String representation of section."""
        if callable(self.I):
            return f"BeamSection(I={self.I!r})"
        return f"BeamSection(I={self.I:.3e} m^4)"
