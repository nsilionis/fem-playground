"""
Linear elastic material models for 2D analysis.

This module provides constitutive models for linear elastic materials
under plane stress and plane strain conditions.
"""

import numpy as np


class LinearElastic:
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
    >>> steel = LinearElastic(E=210e9, nu=0.3, thickness=0.01)
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
