"""Two-node Euler-Bernoulli beam element with cubic Hermite interpolation."""

import numpy as np

from .base import Element1D

FULL_GAUSS_POINTS = 4


def _at(value, x):
    """A property at the positions x, whether a float or a function."""
    if callable(value):
        return np.asarray(value(x), dtype=float)
    return np.full(x.shape, value)


class Beam(Element1D):
    """Two-node Euler-Bernoulli beam element along the global x-axis.

    The DOFs are the transverse displacement w and the rotation
    theta = dw/dx at each node, ordered [w1, theta1, w2, theta2].
    The bending stiffness may vary along the element and is integrated
    by Gauss quadrature at the global positions of the Gauss points.

    Parameters
    ----------
    node_coords : numpy.ndarray, shape (2, 1) or (2, 2)
        Nodal coordinates. With two columns, both nodes must have the
        same y.
    validate : bool, optional
        If True, check that the element has positive length.

    Raises
    ------
    ValueError
        If node 2 does not lie at larger x than node 1, or the two
        nodes have different y.
    """

    dof_names = ('w', 'theta')

    def __init__(self, node_coords, validate=True):
        node_coords = np.asarray(node_coords, dtype=float)
        if node_coords.shape not in ((2, 1), (2, 2)):
            raise ValueError(
                f"Beam requires node_coords of shape (2, 1) or (2, 2), "
                f"got {node_coords.shape}"
            )
        if node_coords[1, 0] <= node_coords[0, 0]:
            raise ValueError("Node 2 must lie at larger x than node 1")
        if (node_coords.shape[1] == 2
                and node_coords[0, 1] != node_coords[1, 1]):
            raise ValueError("Beam must lie along the x-axis")
        super().__init__(node_coords, validate=validate)

    @property
    def element_type(self):
        """Return element type identifier."""
        return 'Beam'

    @property
    def n_nodes(self):
        """Return number of nodes in element."""
        return 2

    @property
    def n_dof(self):
        """Return total degrees of freedom."""
        return 4

    def shape_functions(self, xi):
        """Cubic Hermite shape functions at xi.

        Parameters
        ----------
        xi : float
            Natural coordinate in [-1, 1].

        Returns
        -------
        numpy.ndarray, shape (4,)
            Values for [w1, theta1, w2, theta2].
        """
        L = self.length
        return np.array([
            0.25 * (1 - xi) ** 2 * (2 + xi),
            L / 8 * (1 - xi) ** 2 * (1 + xi),
            0.25 * (1 + xi) ** 2 * (2 - xi),
            -L / 8 * (1 + xi) ** 2 * (1 - xi),
        ])

    def shape_function_derivatives(self, xi):
        """First derivatives of the shape functions with respect to xi.

        Parameters
        ----------
        xi : float
            Natural coordinate in [-1, 1].

        Returns
        -------
        numpy.ndarray, shape (4,)
            dN/dxi for [w1, theta1, w2, theta2].
        """
        L = self.length
        return np.array([
            -0.75 * (1 - xi ** 2),
            L / 8 * (3 * xi + 1) * (xi - 1),
            0.75 * (1 - xi ** 2),
            L / 8 * (3 * xi - 1) * (xi + 1),
        ])

    def shape_function_second_derivatives(self, xi):
        """Second derivatives of the shape functions with respect to xi.

        Parameters
        ----------
        xi : float
            Natural coordinate in [-1, 1].

        Returns
        -------
        numpy.ndarray, shape (4,)
            d2N/dxi2 for [w1, theta1, w2, theta2].
        """
        L = self.length
        return np.array([
            1.5 * xi,
            L / 4 * (3 * xi - 1),
            -1.5 * xi,
            L / 4 * (3 * xi + 1),
        ])

    def get_integration_points(self, order):
        """Gauss-Legendre points and weights on [-1, 1].

        Parameters
        ----------
        order : str or int
            'full' for four points, or an integer number of points of
            at least 2.

        Returns
        -------
        points : numpy.ndarray, shape (n,)
            Natural coordinates of the points.
        weights : numpy.ndarray, shape (n,)
            Weights of the points.

        Raises
        ------
        ValueError
            If order is not 'full' or an integer of at least 2.
        """
        if order == 'full':
            n = FULL_GAUSS_POINTS
        elif (isinstance(order, (int, np.integer))
              and not isinstance(order, bool) and order >= 2):
            n = int(order)
        else:
            raise ValueError(
                f"Invalid integration order {order!r}. Must be 'full' "
                f"or an integer of at least 2."
            )
        return np.polynomial.legendre.leggauss(n)

    def compute_jacobian_1d(self, xi):
        """Return dx/dxi, which is L/2 for the linear geometric map."""
        return self.length / 2.0

    def compute_B_matrix(self, xi):
        """Curvature-displacement row at xi.

        Parameters
        ----------
        xi : float
            Natural coordinate in [-1, 1].

        Returns
        -------
        B : numpy.ndarray, shape (1, 4)
            Maps [w1, theta1, w2, theta2] to the curvature d2w/dx2.
        J : float
            dx/dxi.
        """
        J = self.compute_jacobian_1d(xi)
        B = self.shape_function_second_derivatives(xi) / J ** 2
        return B.reshape(1, -1), J

    def compute_stiffness_matrix(self, material, section,
                                 integration_order='full'):
        """Element stiffness matrix by Gauss quadrature.

            K_e = sum_g w_g EI(x_g) B_g^T B_g J

        Parameters
        ----------
        material : LinearElastic1D
            Supplies E, as a float or a function of x.
        section : BeamSection
            Supplies I, as a float or a function of x.
        integration_order : str or int, optional
            See get_integration_points. Default is 'full'.

        Returns
        -------
        numpy.ndarray, shape (4, 4)
            Element stiffness matrix.

        Raises
        ------
        ValueError
            If EI is not positive at every Gauss point.
        """
        points, weights = self.get_integration_points(integration_order)
        J = self.length / 2.0
        x = self.node_coords[0, 0] + (1.0 + points) * J
        EI = _at(material.E, x) * _at(section.I, x)
        if np.any(EI <= 0):
            raise ValueError("EI must be positive at every Gauss point")

        K_e = np.zeros((4, 4))
        for xi, w, ei in zip(points, weights, EI):
            B, _ = self.compute_B_matrix(xi)
            K_e += w * ei * (B.T @ B) * J
        return K_e

    def compute_mass_matrix(self, material, section,
                            integration_order='full'):
        """Consistent element mass matrix by Gauss quadrature.

            M_e = sum_g w_g rhoA(x_g) N_g^T N_g J

        Parameters
        ----------
        material : LinearElastic1D
            Supplies rho, as a float or a function of x.
        section : BeamSection
            Supplies A, as a float or a function of x.
        integration_order : str or int, optional
            See get_integration_points. Default is 'full', which is
            exact for rho * A up to linear in x.

        Returns
        -------
        numpy.ndarray, shape (4, 4)
            Element mass matrix.

        Raises
        ------
        ValueError
            If material.rho or section.A is not set, or rho * A is not
            positive at every Gauss point.
        """
        if material.rho is None:
            raise ValueError("Mass matrix needs a density: material.rho "
                             "is not set")
        if section.A is None:
            raise ValueError("Mass matrix needs an area: section.A "
                             "is not set")
        points, weights = self.get_integration_points(integration_order)
        J = self.length / 2.0
        x = self.node_coords[0, 0] + (1.0 + points) * J
        rhoA = _at(material.rho, x) * _at(section.A, x)
        if np.any(rhoA <= 0):
            raise ValueError("rho * A must be positive at every Gauss point")

        M_e = np.zeros((4, 4))
        for xi, w, ra in zip(points, weights, rhoA):
            N = self.shape_functions(xi)
            M_e += w * ra * np.outer(N, N) * J
        return M_e
