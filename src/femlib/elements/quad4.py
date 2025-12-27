"""
4-node quadrilateral (Q4) element implementation.

This module provides the Q4 isoparametric element for 2D plane stress
and plane strain analysis.
"""

import numpy as np
from .base import Element2D


class Q4Element(Element2D):
    """
    4-node quadrilateral isoparametric element (Q4).

    The Q4 element uses bilinear shape functions and is suitable for
    plane stress and plane strain analysis. Nodes are numbered
    counter-clockwise starting from the bottom-left corner in the
    natural coordinate system.

    Natural coordinates (ξ, η) ∈ [-1, 1] × [-1, 1]

    Node numbering:
        4 ----------- 3
        |             |
        |      η      |
        |      ↑      |
        |      |      |
        |      o--→ ξ |
        |             |
        1 ----------- 2

    Parameters
    ----------
    node_coords : ndarray, shape (4, 2)
        Nodal coordinates [x, y] for each of the 4 nodes,
        ordered counter-clockwise.
    validate : bool, optional
        If True, validate element geometry on initialisation. Default is True.

    Examples
    --------
    >>> coords = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    >>> element = Q4Element(coords)
    >>> N = element.shape_functions(0.0, 0.0)  # Centre of element
    """

    def __init__(self, node_coords, validate=True):
        """Initialise Q4 element with nodal coordinates."""
        if node_coords.shape[0] != 4:
            raise ValueError(
                f"Q4 element requires 4 nodes, got {node_coords.shape[0]}"
            )
        super().__init__(node_coords, validate=validate)

    @property
    def element_type(self):
        """Return element type identifier."""
        return 'Q4'

    @property
    def n_nodes(self):
        """Return number of nodes in element."""
        return 4

    @property
    def n_dof(self):
        """Return total degrees of freedom."""
        return 8

    def shape_functions(self, xi, eta):
        """
        Evaluate Q4 shape functions at (ξ, η).

        The bilinear shape functions are:
            N_i = (1 + ξ_i*ξ)(1 + η_i*η) / 4

        where (ξ_i, η_i) are the natural coordinates of node i.

        Parameters
        ----------
        xi : float
            Natural coordinate ξ ∈ [-1, 1].
        eta : float
            Natural coordinate η ∈ [-1, 1].

        Returns
        -------
        N : ndarray, shape (4,)
            Shape function values [N_1, N_2, N_3, N_4].
        """
        N = np.array([
            0.25 * (1 - xi) * (1 - eta),  # N_1
            0.25 * (1 + xi) * (1 - eta),  # N_2
            0.25 * (1 + xi) * (1 + eta),  # N_3
            0.25 * (1 - xi) * (1 + eta)   # N_4
        ])
        return N

    def shape_function_derivatives(self, xi, eta):
        """
        Evaluate derivatives of Q4 shape functions w.r.t. (ξ, η).

        Parameters
        ----------
        xi : float
            Natural coordinate ξ.
        eta : float
            Natural coordinate η.

        Returns
        -------
        dN_dxi : ndarray, shape (4,)
            Derivatives ∂N_i/∂ξ for i=1,2,3,4.
        dN_deta : ndarray, shape (4,)
            Derivatives ∂N_i/∂η for i=1,2,3,4.
        """
        dN_dxi = np.array([
            -0.25 * (1 - eta),  # ∂N_1/∂ξ
            0.25 * (1 - eta),  # ∂N_2/∂ξ
            0.25 * (1 + eta),  # ∂N_3/∂ξ
            -0.25 * (1 + eta)   # ∂N_4/∂ξ
        ])

        dN_deta = np.array([
            -0.25 * (1 - xi),   # ∂N_1/∂η
            -0.25 * (1 + xi),   # ∂N_2/∂η
            0.25 * (1 + xi),   # ∂N_3/∂η
            0.25 * (1 - xi)    # ∂N_4/∂η
        ])

        return dN_dxi, dN_deta

    def get_integration_points(self, order):
        """
        Return Gauss integration points and weights for Q4 element.

        Parameters
        ----------
        order : str
            Integration order:
            - 'full': 2×2 Gauss quadrature (4 points)
            - 'reduced': 1×1 Gauss quadrature (1 point)

        Returns
        -------
        points : ndarray, shape (n_points, 2)
            Integration point coordinates (ξ, η).
        weights : ndarray, shape (n_points,)
            Integration weights.

        Raises
        ------
        ValueError
            If order is not 'full' or 'reduced'.
        """
        if order == 'full':
            # 2×2 Gauss quadrature
            gp = 1.0 / np.sqrt(3.0)  # ±1/√3
            points = np.array([
                [-gp, -gp],
                [gp, -gp],
                [gp,  gp],
                [-gp,  gp]
            ])
            weights = np.array([1.0, 1.0, 1.0, 1.0])

        elif order == 'reduced':
            # 1×1 Gauss quadrature (centre point)
            points = np.array([[0.0, 0.0]])
            weights = np.array([4.0])

        else:
            raise ValueError(
                f"Invalid integration order '{order}'. "
                f"Must be 'full' or 'reduced'."
            )

        return points, weights

    def compute_stiffness_matrix(
        self,
        material,
        integration_order='full'
    ):
        """
        Compute Q4 element stiffness matrix using Gauss quadrature.

        The stiffness matrix is computed as:
            K_e = ∫∫ B^T D B |J| t dξ dη

        where:
            B: strain-displacement matrix
            D: constitutive matrix (from material)
            |J|: Jacobian determinant
            t: thickness (from material)

        Parameters
        ----------
        material : Material
            Material object providing constitutive_matrix() and
            thickness attribute.
        integration_order : str, optional
            Integration scheme: 'full' (2×2) or 'reduced' (1×1).
            Default is 'full'.

        Returns
        -------
        K_e : ndarray, shape (8, 8)
            Element stiffness matrix.

        Notes
        -----
        Full integration (2×2) integrates the stiffness exactly but
        may exhibit shear locking for thin structures.

        Reduced integration (1×1) can alleviate locking but may
        introduce spurious zero-energy modes (hourglassing).
        """
        # Get integration points and weights
        points, weights = self.get_integration_points(integration_order)

        # Get constitutive matrix and thickness from material
        D = material.constitutive_matrix()
        t = material.thickness

        # Initialise element stiffness matrix
        K_e = np.zeros((self.n_dof, self.n_dof))

        # Numerical integration loop
        for (xi, eta), w in zip(points, weights):
            # Compute B matrix and Jacobian determinant at this point
            B, det_J = self.compute_B_matrix(xi, eta)

            # Add contribution to stiffness matrix
            # K_e += w * B^T * D * B * det(J) * t
            K_e += w * (B.T @ D @ B) * det_J * t

        return K_e
