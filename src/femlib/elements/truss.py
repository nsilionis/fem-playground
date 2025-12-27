"""
2-node truss (bar) element implementation.

This module provides a linear truss element for 2D structural analysis
using isoparametric formulation. The parent element is purely 1D with
axial behavior only; transformation to 2D global coordinates is handled
separately via rotation matrix.
"""

import numpy as np
from .base import Element1D


class Truss(Element1D):
    """
    2-node linear truss element in 2D space using isoparametric formulation.

    The truss element is fundamentally a 1D element with axial deformation
    only. The parent element is formulated in local coordinates (along the
    element axis) and then transformed to global 2D coordinates using a
    rotation matrix based on element orientation.

    Parent Element (Local Coordinates):
    ------------------------------------
    - 2 nodes with 1 DOF each (axial displacement)
    - Natural coordinate: ξ ∈ [-1, 1]
    - Local DOFs: [u1', u2'] (displacements along element axis)
    - Local stiffness: 2×2 matrix

    Transformation to Global Coordinates:
    --------------------------------------
    - Rotation based on element orientation angle θ
    - Global DOFs: [u1, v1, u2, v2] (2D displacements at each node)
    - Transformation matrix T: maps global → local
    - Global stiffness: K_global = T^T * K_local * T (4×4 matrix)

    Node numbering:
        1 =========== 2
        (node i)    (node j)

    Natural coordinate: ξ ∈ [-1, 1]
        Node 1 at ξ = -1
        Node 2 at ξ = +1

    Parameters
    ----------
    node_coords : ndarray, shape (2, 2)
        Nodal coordinates [[x1, y1], [x2, y2]] in global system.
    validate : bool, optional
        If True, validate element geometry on initialisation.
        Default is True.

    Attributes
    ----------
    length : float
        Element length in physical space.
    angle : float
        Element orientation angle (radians), measured counter-clockwise
        from positive x-axis.
    cos_theta : float
        Cosine of orientation angle.
    sin_theta : float
        Sine of orientation angle.

    Examples
    --------
    >>> coords = np.array([[0.0, 0.0], [3.0, 4.0]])
    >>> element = Truss(coords)
    >>> element.length
    5.0
    >>> element.n_dof_local
    2
    >>> K_local = element.compute_local_stiffness(material)
    >>> K_local.shape
    (2, 2)
    >>> K_global = element.compute_stiffness_matrix(material)
    >>> K_global.shape
    (4, 4)

    Notes
    -----
    The local element stiffness is computed using isoparametric formulation:
        K_local = ∫ B^T (EA) B |J| dξ

    where B relates local nodal displacements to axial strain, and the
    integration is performed over ξ ∈ [-1, 1] using Gauss quadrature.
    """

    def __init__(self, node_coords, validate=True):
        """Initialise truss element with nodal coordinates."""
        node_coords = np.asarray(node_coords, dtype=float)

        if node_coords.shape != (2, 2):
            raise ValueError(
                f"Truss element requires 2 nodes in 2D space, "
                f"got shape {node_coords.shape}"
            )

        # Call parent constructor (computes length)
        super().__init__(node_coords, validate=validate)

        # Compute orientation angle and direction cosines
        self.angle = self._compute_angle()
        self.cos_theta = np.cos(self.angle)
        self.sin_theta = np.sin(self.angle)

    def _compute_angle(self):
        """
        Compute element orientation angle.

        Returns
        -------
        float
            Angle in radians, measured counter-clockwise from positive
            x-axis to element axis (node 1 → node 2).
        """
        delta = self.node_coords[1] - self.node_coords[0]
        return np.arctan2(delta[1], delta[0])

    @property
    def element_type(self):
        """Return element type identifier."""
        return 'Truss'

    @property
    def n_nodes(self):
        """Return number of nodes in element."""
        return 2

    @property
    def n_dof_local(self):
        """
        Return number of DOFs for parent element in local coordinates.

        Returns
        -------
        int
            Number of local DOFs (1 per node for truss).
        """
        return 2

    @property
    def n_dof(self):
        """
        Return total degrees of freedom in global coordinate system.

        Returns
        -------
        int
            Number of global DOFs (2 per node in 2D space).
        """
        return 4

    def shape_functions(self, xi):
        """
        Evaluate linear shape functions at natural coordinate ξ.

        The linear shape functions are:
            N_1(ξ) = (1 - ξ) / 2
            N_2(ξ) = (1 + ξ) / 2

        Parameters
        ----------
        xi : float
            Natural coordinate ξ ∈ [-1, 1].

        Returns
        -------
        N : ndarray, shape (2,)
            Shape function values [N_1, N_2].
        """
        N = np.array([
            0.5 * (1.0 - xi),  # N_1
            0.5 * (1.0 + xi)   # N_2
        ])
        return N

    def shape_function_derivatives(self, xi):
        """
        Evaluate derivatives of shape functions with respect to ξ.

        For linear elements, derivatives are constant:
            dN_1/dξ = -1/2
            dN_2/dξ = +1/2

        Parameters
        ----------
        xi : float
            Natural coordinate ξ (unused for linear elements, but
            included for interface consistency).

        Returns
        -------
        dN_dxi : ndarray, shape (2,)
            Derivatives [dN_1/dξ, dN_2/dξ].
        """
        dN_dxi = np.array([-0.5, 0.5])
        return dN_dxi

    def get_integration_points(self, order):
        """
        Return Gauss integration points and weights for 1D integration.

        Parameters
        ----------
        order : str or int
            Integration order:
            - 'full' or 2: 2-point Gauss quadrature
            - 'reduced' or 1: 1-point Gauss quadrature (exact for linear truss)

        Returns
        -------
        points : ndarray, shape (n_points,)
            Integration point coordinates ξ in natural space.
        weights : ndarray, shape (n_points,)
            Integration weights.

        Notes
        -----
        For linear truss elements with constant EA, 1-point integration
        is exact for the stiffness matrix.
        """
        if order in ('reduced', 1):
            # 1-point Gauss quadrature (centre)
            points = np.array([0.0])
            weights = np.array([2.0])

        elif order in ('full', 2):
            # 2-point Gauss quadrature
            gp = 1.0 / np.sqrt(3.0)  # ±1/√3
            points = np.array([-gp, gp])
            weights = np.array([1.0, 1.0])

        else:
            raise ValueError(
                f"Invalid integration order '{order}'. "
                f"Must be 'full', 'reduced', 1, or 2."
            )

        return points, weights

    def compute_B_matrix_local(self, xi):
        """
        Compute strain-displacement matrix B in local coordinates.

        For the parent 1D element, B relates local nodal displacements
        [u1', u2'] to axial strain:
            ε = B_local * u_local

        Parameters
        ----------
        xi : float
            Natural coordinate ξ.

        Returns
        -------
        B : ndarray, shape (1, 2)
            Strain-displacement matrix in local coordinates.
            Maps [u1', u2'] → ε_axial
        det_J : float
            Jacobian determinant (= L/2 for linear element).

        Notes
        -----
        The B matrix relates derivatives of shape functions to strain:
            B = dN/dx = (dN/dξ) / J

        For a truss element:
            ε_axial = du'/dx = Σ (dN_i/dx) * u_i'

        where x is the local axial coordinate along the element.
        """
        # Get shape function derivatives w.r.t. ξ
        dN_dxi = self.shape_function_derivatives(xi)

        # Compute Jacobian: J = dx/dξ = L/2 for linear mapping
        J = self.compute_jacobian_1d(xi)

        # Transform derivatives to physical coordinate
        # dN/dx = (dN/dξ) * (dξ/dx) = (dN/dξ) / J
        dN_dx = dN_dxi / J

        # B matrix for axial strain
        # ε = [dN_1/dx, dN_2/dx] * [u1', u2']^T
        B = dN_dx.reshape(1, -1)

        return B, J

    def compute_local_stiffness(self, material, integration_order='reduced'):
        """
        Compute element stiffness matrix in local coordinates.

        The local stiffness matrix relates local nodal forces to local
        nodal displacements:
            f_local = K_local * u_local

        where:
            f_local = [f1', f2']^T (axial forces at nodes)
            u_local = [u1', u2']^T (axial displacements at nodes)

        The stiffness is computed using isoparametric formulation:
            K_local = ∫ B^T (EA) B |J| dξ

        Parameters
        ----------
        material : LinearElasticTruss
            Material object providing E and A.
        integration_order : str, optional
            Integration scheme: 'full' or 'reduced'.
            Default is 'reduced' (exact for linear truss).

        Returns
        -------
        K_local : ndarray, shape (2, 2)
            Element stiffness matrix in local coordinates.

        Notes
        -----
        For a linear truss element with constant EA, the exact result is:
            K_local = (EA/L) * [[ 1, -1],
                                [-1,  1]]

        The isoparametric formulation reproduces this result and
        demonstrates the unified framework used throughout the library.
        """
        # Get material properties
        try:
            EA = material.E * material.A
        except AttributeError:
            raise TypeError(
                "Material must have E and A attributes \
                    (use LinearElasticTruss)"
            )

        # Get integration points and weights
        points, weights = self.get_integration_points(integration_order)

        # Initialise local stiffness matrix (2×2)
        K_local = np.zeros((self.n_dof_local, self.n_dof_local))

        # Numerical integration loop
        for xi, w in zip(points, weights):
            # Compute B matrix and Jacobian in local coordinates
            B, J = self.compute_B_matrix_local(xi)

            # Add contribution to stiffness matrix
            # K_local += w * B^T * (EA) * B * |J|
            K_local += w * (B.T @ (EA * B)) * J

        return K_local

    def get_transformation_matrix(self):
        """
        Compute transformation matrix from global to local coordinates.

        The transformation matrix T relates global nodal displacements
        to local nodal displacements:
            u_local = T * u_global

        where:
            u_global = [u1, v1, u2, v2]^T (2D displacements)
            u_local = [u1', u2']^T (axial displacements)

        Returns
        -------
        T : ndarray, shape (2, 4)
            Transformation matrix.

        Notes
        -----
        The transformation uses direction cosines to project 2D
        displacements onto the element axis:
            u_i' = cos(θ) * u_i + sin(θ) * v_i

        The transformation matrix has the structure:
            T = [[c, s, 0, 0],
                 [0, 0, c, s]]

        where c = cos(θ), s = sin(θ).
        """
        c = self.cos_theta
        s = self.sin_theta

        T = np.array([
            [c, s, 0, 0],  # u1' = c*u1 + s*v1
            [0, 0, c, s]   # u2' = c*u2 + s*v2
        ])

        return T

    def compute_stiffness_matrix(self, material, integration_order='reduced'):
        """
        Compute element stiffness matrix in global coordinates.

        The global stiffness matrix is obtained by transforming the local
        stiffness matrix using the rotation matrix:
            K_global = T^T * K_local * T

        This transformation ensures that the element stiffness properly
        relates global nodal forces to global nodal displacements:
            f_global = K_global * u_global

        Parameters
        ----------
        material : LinearElasticTruss
            Material object providing E and A.
        integration_order : str, optional
            Integration scheme: 'full' or 'reduced'.
            Default is 'reduced'.

        Returns
        -------
        K_global : ndarray, shape (4, 4)
            Element stiffness matrix in global coordinates.
            DOF ordering: [u1, v1, u2, v2]

        Notes
        -----
        This implementation properly separates:
        1. Parent element formulation (local, 1D, axial only)
        2. Coordinate transformation (rotation to global 2D system)

        The resulting global stiffness has the form:
            K_global = (EA/L) * [[ c²,  cs, -c², -cs],
                                  [ cs,  s², -cs, -s²],
                                  [-c², -cs,  c²,  cs],
                                  [-cs, -s², cs,  s²]]

        where c = cos(θ), s = sin(θ).
        """
        # Compute local stiffness matrix (2×2)
        K_local = self.compute_local_stiffness(material, integration_order)

        # Get transformation matrix (2×4)
        T = self.get_transformation_matrix()

        # Transform to global coordinates: K_global = T^T * K_local * T
        K_global = T.T @ K_local @ T

        return K_global

    def __repr__(self):
        """String representation."""
        return (
            f"Truss(length={self.length:.4f} m, "
            f"angle={np.degrees(self.angle):.2f}°)"
        )

    def __str__(self):
        """Human-readable description."""
        return (
            f"Truss Element:\n"
            f"  Node 1: ({self.node_coords[0, 0]:.3f}, "
            f"{self.node_coords[0, 1]:.3f}) m\n"
            f"  Node 2: ({self.node_coords[1, 0]:.3f}, "
            f"{self.node_coords[1, 1]:.3f}) m\n"
            f"  Length: {self.length:.4f} m\n"
            f"  Angle:  {np.degrees(self.angle):.2f}° from x-axis\n"
            f"  Local DOFs: {self.n_dof_local}, Global DOFs: {self.n_dof}"
        )
