"""
Linear elastic material models for 2D analysis.

This module provides constitutive models for linear elastic materials
under plane stress and plane strain conditions.
"""

import numpy as np

"""
Linear elastic material model for truss elements.

This module provides a material model for 1D truss (bar) elements
requiring only Young's modulus and cross-sectional area.
"""


class LinearElasticTruss:
    """
    Linear elastic material for truss elements.

    This material model is used for 1D bar/truss elements in 2D or 3D
    space. Unlike plane stress/strain materials, it only requires
    axial stiffness properties.

    Parameters
    ----------
    E : float
        Young's modulus [Pa].
    A : float
        Cross-sectional area [m²].

    Attributes
    ----------
    E : float
        Young's modulus.
    A : float
        Cross-sectional area.
    axial_stiffness : float
        Product EA, computed on initialization for efficiency.

    Examples
    --------
    >>> steel_truss = LinearElasticTruss(E=200e9, A=0.001)
    >>> steel_truss.axial_stiffness
    2e8

    Notes
    -----
    This material model does not provide a constitutive matrix (D)
    since truss elements use direct axial stiffness formulation rather
    than strain-displacement-stress relationships.
    """

    def __init__(self, E, A):
        """
        Initialise linear elastic truss material.

        Parameters
        ----------
        E : float
            Young's modulus [Pa]. Must be positive.
        A : float
            Cross-sectional area [m²]. Must be positive.

        Raises
        ------
        ValueError
            If E or A is non-positive.
        """
        if E <= 0:
            raise ValueError(
                f"Young's modulus must be positive, got E={E}"
            )
        if A <= 0:
            raise ValueError(
                f"Cross-sectional area must be positive, got A={A}"
            )

        self.E = float(E)
        self.A = float(A)
        self.axial_stiffness = self.E * self.A

    def __repr__(self):
        """String representation of material."""
        return (
            f"LinearElasticTruss(E={self.E:.3e} Pa, "
            f"A={self.A:.6f} m²)"
        )

    def __str__(self):
        """Human-readable description."""
        return (
            f"Linear Elastic Truss Material:\n"
            f"  Young's modulus: {self.E/1e9:.1f} GPa\n"
            f"  Cross-section:   {self.A*1e4:.2f} cm²\n"
            f"  Axial stiffness: {self.axial_stiffness:.3e} N"
        )


class LinearElastic2D:
    """
    Linear elastic material for 2D analysis.

    Supports both plane stress and plane strain formulations with
    isotropic material properties.

    Parameters
    ----------
    E : float
        Young's modulus (Pa or consistent units).
    nu : float
        Poisson's ratio (dimensionless, typically 0.0 to 0.5).
    thickness : float
        Thickness in the out-of-plane direction (m or consistent units).
    plane_type : str, optional
        Analysis type: 'stress' or 'strain'. Default is 'stress'.

    Attributes
    ----------
    E : float
        Young's modulus.
    nu : float
        Poisson's ratio.
    thickness : float
        Material thickness.
    plane_type : str
        Plane stress or plane strain formulation.

    Raises
    ------
    ValueError
        If material properties are physically invalid or plane_type
        is not recognised.

    Examples
    --------
    >>> steel = LinearElastic2D(E=210e9, nu=0.3, thickness=0.01)
    >>> D = steel.constitutive_matrix()
    >>> D.shape
    (3, 3)
    """

    def __init__(self, E, nu, thickness, plane_type='stress'):
        """Initialise linear elastic material."""
        # Validate material properties
        if E <= 0:
            raise ValueError(
                f"Young's modulus must be positive, got E={E}"
            )
        if not (0.0 <= nu < 0.5):
            raise ValueError(
                f"Poisson's ratio must be in [0, 0.5), got nu={nu}"
            )
        if thickness <= 0:
            raise ValueError(
                f"Thickness must be positive, got thickness={thickness}"
            )
        if plane_type not in ['stress', 'strain']:
            raise ValueError(
                f"plane_type must be 'stress' or 'strain', "
                f"got '{plane_type}'"
            )

        self.E = float(E)
        self.nu = float(nu)
        self.thickness = float(thickness)
        self.plane_type = plane_type

    def constitutive_matrix(self):
        """
        Compute constitutive (elasticity) matrix D.

        The constitutive matrix relates stresses to strains:
            {σ} = [D]{ε}

        For plane stress:
            σ = [σ_xx, σ_yy, τ_xy]^T
            ε = [ε_xx, ε_yy, γ_xy]^T

        For plane strain:
            Similar form but with modified elastic constants.

        Returns
        -------
        D : ndarray, shape (3, 3)
            Constitutive matrix for the specified plane type.

        Notes
        -----
        Plane stress: Assumes σ_zz = 0 (thin plates).
        Plane strain: Assumes ε_zz = 0 (long structures).
        """
        if self.plane_type == 'stress':
            # Plane stress formulation
            factor = self.E / (1.0 - self.nu**2)
            D = factor * np.array([
                [1.0,       self.nu,   0.0],
                [self.nu,   1.0,       0.0],
                [0.0,       0.0,       (1.0 - self.nu) / 2.0]
            ])

        elif self.plane_type == 'strain':
            # Plane strain formulation
            factor = self.E / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))
            D = factor * np.array([
                [1.0 - self.nu,  self.nu,        0.0],
                [self.nu,        1.0 - self.nu,  0.0],
                [0.0,            0.0,            (1.0 - 2.0 * self.nu) / 2.0]
            ])

        return D

    def __repr__(self):
        """String representation of material."""
        return (
            f"LinearElastic(E={self.E:.3e}, nu={self.nu:.3f}, "
            f"thickness={self.thickness:.3e}, "
            f"plane_type='{self.plane_type}')"
        )
