"""
Base classes for finite elements.

This module provides abstract base classes that define the interface
for all finite element types in the library.
"""

from abc import ABC, abstractmethod
import numpy as np


class Element2D(ABC):
    """
    Abstract base class for 2D finite elements.

    All 2D element types (plane stress, plane strain) must inherit from
    this class and implement the abstract methods to define element-specific
    behaviour.

    Attributes
    ----------
    node_coords : ndarray, shape (n_nodes, 2)
        Nodal coordinates in global coordinate system [x, y] for each node.
    n_nodes : int
        Number of nodes in the element.
    n_dof : int
        Total number of degrees of freedom (2 per node for 2D elements).
    """

    def __init__(self, node_coords, validate=True):
        """
        Initialise the 2D element.

        Parameters
        ----------
        node_coords : ndarray, shape (n_nodes, 2)
            Nodal coordinates [x, y] for each element node.
            Nodes should be ordered counter-clockwise.
        validate : bool, optional
            If True, validate element geometry on initialisation.
            Default is True.
        """
        self.node_coords = np.asarray(node_coords, dtype=float)
        self.n_nodes = self.node_coords.shape[0]
        self.n_dof = 2 * self.n_nodes

        # Validate input
        if self.node_coords.shape[1] != 2:
            raise ValueError(
                f"node_coords must have shape (n_nodes, 2), "
                f"got {self.node_coords.shape}"
            )

        # Optional geometry validation
        if validate:
            quality = self.check_element_quality()
            if not quality['valid']:
                raise ValueError(
                    "Invalid element geometry detected:\n"
                    + "\n".join(quality['warnings'])
                )

    @property
    @abstractmethod
    def element_type(self):
        """
        Return element type identifier (e.g., 'Q4', 'Q8', 'T3').

        Returns
        -------
        str
            Element type string.
        """
        pass

    @abstractmethod
    def shape_functions(self, xi, eta):
        """
        Evaluate shape functions at natural coordinates (xi, eta).

        Parameters
        ----------
        xi : float
            Natural coordinate xi (typically in [-1, 1]).
        eta : float
            Natural coordinate eta (typically in [-1, 1]).

        Returns
        -------
        N : ndarray, shape (n_nodes,)
            Shape function values at (xi, eta).
        """
        pass

    @abstractmethod
    def shape_function_derivatives(self, xi, eta):
        """
        Evaluate derivatives of shape functions with
        respect to natural coordinates.

        Parameters
        ----------
        xi : float
            Natural coordinate xi.
        eta : float
            Natural coordinate eta.

        Returns
        -------
        dN_dxi : ndarray, shape (n_nodes,)
            Derivatives of shape functions with respect to xi.
        dN_deta : ndarray, shape (n_nodes,)
            Derivatives of shape functions with respect to eta.
        """
        pass

    @abstractmethod
    def get_integration_points(self, order):
        """
        Return Gauss integration points and weights for specified order.

        Parameters
        ----------
        order : str
            Integration order: 'full' or 'reduced'.

        Returns
        -------
        points : ndarray, shape (n_points, 2)
            Integration point coordinates (xi, eta) in natural space.
        weights : ndarray, shape (n_points,)
            Integration weights corresponding to each point.
        """
        pass

    def compute_jacobian(self, xi, eta):
        """
        Compute Jacobian matrix of isoparametric transformation at (xi, eta).

        The Jacobian relates derivatives in natural coordinates to derivatives
        in global coordinates:
            J = [dx/dxi   dy/dxi  ]
                [dx/deta  dy/deta ]

        Parameters
        ----------
        xi : float
            Natural coordinate xi.
        eta : float
            Natural coordinate eta.

        Returns
        -------
        J : ndarray, shape (2, 2)
            Jacobian matrix.
        """
        dN_dxi, dN_deta = self.shape_function_derivatives(xi, eta)

        # J_ij = sum_k (dN_k/dxi_i * x_k_j)
        J = np.zeros((2, 2))
        J[0, 0] = np.sum(dN_dxi * self.node_coords[:, 0])   # dx/dxi
        J[0, 1] = np.sum(dN_dxi * self.node_coords[:, 1])   # dy/dxi
        J[1, 0] = np.sum(dN_deta * self.node_coords[:, 0])  # dx/deta
        J[1, 1] = np.sum(dN_deta * self.node_coords[:, 1])  # dy/deta

        return J

    def compute_B_matrix(self, xi, eta):
        """
        Compute strain-displacement matrix B at natural coordinates (xi, eta).

        The B-matrix relates nodal displacements to strains:
            {ε} = [B]{u}

        For 2D plane stress/strain:
            ε = [ε_xx, ε_yy, γ_xy]^T

        Parameters
        ----------
        xi : float
            Natural coordinate xi.
        eta : float
            Natural coordinate eta.

        Returns
        -------
        B : ndarray, shape (3, n_dof)
            Strain-displacement matrix.
        det_J : float
            Determinant of Jacobian (for integration).

        Raises
        ------
        ValueError
            If Jacobian determinant is non-positive (invalid element geometry).
        """
        dN_dxi, dN_deta = self.shape_function_derivatives(xi, eta)
        J = self.compute_jacobian(xi, eta)
        det_J = np.linalg.det(J)

        # Check for invalid element geometry
        if det_J <= 0:
            raise ValueError(
                f"Non-positive Jacobian determinant: {det_J:.6e}. "
                f"Element geometry is invalid "
                f"(possibly inverted or degenerate)."
            )

        # Compute inverse Jacobian
        J_inv = np.linalg.inv(J)

        # Transform derivatives to global coordinates
        # [dN/dx]   [dxi/dx   deta/dx ] [dN/dxi ]
        # [dN/dy] = [dxi/dy   deta/dy ] [dN/deta]
        dN_dx = J_inv[0, 0] * dN_dxi + J_inv[0, 1] * dN_deta
        dN_dy = J_inv[1, 0] * dN_dxi + J_inv[1, 1] * dN_deta

        # Assemble B matrix
        # For each node i: contributes to columns [2*i, 2*i+1]
        B = np.zeros((3, self.n_dof))
        for i in range(self.n_nodes):
            B[0, 2*i] = dN_dx[i]      # ε_xx = ∂u/∂x
            B[1, 2*i + 1] = dN_dy[i]      # ε_yy = ∂v/∂y
            B[2, 2*i] = dN_dy[i]      # γ_xy = ∂u/∂y + ∂v/∂x
            B[2, 2*i + 1] = dN_dx[i]

        return B, det_J

    @abstractmethod
    def compute_stiffness_matrix(self, material, integration_order='full'):
        """
        Compute element stiffness matrix.

        The element stiffness matrix is computed via numerical integration:
            K_e = ∫∫ B^T D B |J| dξ dη

        Parameters
        ----------
        material : Material
            Material object providing constitutive matrix and thickness.
        integration_order : str, optional
            Integration scheme: 'full' or 'reduced'. Default is 'full'.

        Returns
        -------
        K_e : ndarray, shape (n_dof, n_dof)
            Element stiffness matrix in global coordinates.
        """
        pass

    def check_element_quality(self):
        """
        Perform quality checks on element geometry.

        Checks Jacobian determinant at integration points to detect
        distorted or inverted elements.

        Returns
        -------
        quality_info : dict
            Dictionary containing:
            - 'valid': bool indicating if element passes quality checks
            - 'min_det_J': minimum Jacobian determinant
            - 'max_det_J': maximum Jacobian determinant
            - 'warnings': list of warning messages
        """
        warnings = []

        # Check at full integration points
        points, _ = self.get_integration_points('full')
        det_J_values = []

        for xi, eta in points:
            try:
                J = self.compute_jacobian(xi, eta)
                det_J = np.linalg.det(J)
                det_J_values.append(det_J)

                if det_J <= 0:
                    warnings.append(
                        f"Non-positive Jacobian at (ξ={xi:.3f}, η={eta:.3f}): "
                        f"det(J)={det_J:.6e}"
                    )
            except Exception as e:
                warnings.append(f"Error computing Jacobian: {str(e)}")

        min_det_J = min(det_J_values) if det_J_values else None
        max_det_J = max(det_J_values) if det_J_values else None

        # Element is valid if all Jacobian determinants are positive
        valid = len(warnings) == 0 and min_det_J is not None and min_det_J > 0

        return {
            'valid': valid,
            'min_det_J': min_det_J,
            'max_det_J': max_det_J,
            'warnings': warnings
        }
