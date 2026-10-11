"""
Linear elastic material models.

This module provides constitutive models for linear elastic materials:
a uniaxial model for 1D elements, and plane stress and plane strain
models for 2D continuum elements.
"""

import numpy as np


class LinearElastic1D:
    """
    Linear elastic material under uniaxial stress.

    Used by 1D elements, the truss and the beam. Cross-sectional
    properties are supplied separately by a section object.

    Parameters
    ----------
    E : float or callable
        Young's modulus [Pa]. A function must accept a numpy.ndarray of
        positions x and return an array of the same shape. The truss
        element requires a float.
    rho : float or callable, optional
        Density [kg/m^3], in the same forms as E. Needed only for mass
        matrices. Default is None.

    Attributes
    ----------
    E : float or callable
        Young's modulus.
    rho : float, callable or None
        Density.

    Raises
    ------
    ValueError
        If a constant E or rho is not positive.

    Examples
    --------
    >>> steel = LinearElastic1D(E=200e9, rho=7850.0)
    """

    def __init__(self, E, rho=None):
        """Initialise uniaxial linear elastic material."""
        if not callable(E) and E <= 0:
            raise ValueError(
                f"Young's modulus must be positive, got E={E}"
            )
        if rho is not None and not callable(rho) and rho <= 0:
            raise ValueError(
                f"Density must be positive, got rho={rho}"
            )
        self.E = E if callable(E) else float(E)
        if rho is None or callable(rho):
            self.rho = rho
        else:
            self.rho = float(rho)

    def __repr__(self):
        """String representation of material."""
        return f"LinearElastic1D(E={self.E!r}, rho={self.rho!r})"


class LinearElastic2D:
    """
    Linear elastic material for 2D analysis.

    Supports both plane stress and plane strain formulations with
    isotropic material properties. The out-of-plane thickness is
    supplied separately by a PlaneSection.

    Parameters
    ----------
    E : float
        Young's modulus (Pa or consistent units).
    nu : float
        Poisson's ratio (dimensionless, typically 0.0 to 0.5).
    plane_type : str, optional
        Analysis type: 'stress' or 'strain'. Default is 'stress'.

    Attributes
    ----------
    E : float
        Young's modulus.
    nu : float
        Poisson's ratio.
    plane_type : str
        Plane stress or plane strain formulation.

    Raises
    ------
    ValueError
        If material properties are physically invalid or plane_type
        is not recognised.

    Examples
    --------
    >>> steel = LinearElastic2D(E=210e9, nu=0.3)
    >>> D = steel.constitutive_matrix()
    >>> D.shape
    (3, 3)
    """

    def __init__(self, E, nu, plane_type='stress'):
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
        if plane_type not in ['stress', 'strain']:
            raise ValueError(
                f"plane_type must be 'stress' or 'strain', "
                f"got '{plane_type}'"
            )

        self.E = float(E)
        self.nu = float(nu)
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
            f"LinearElastic2D(E={self.E:.3e}, nu={self.nu:.3f}, "
            f"plane_type='{self.plane_type}')"
        )
