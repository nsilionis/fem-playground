"""
Response recovery along beam meshes.

Displacement, curvature and surface strain at arbitrary positions,
from nodal DOF vectors such as mode shapes, using the Hermite shape
functions of the element that contains each position.
"""

import numpy as np

from ..assembly import get_element_dofs
from ..elements import Beam


def beam_displacement(mesh, u, x):
    """
    Transverse displacement w at positions x.

    Parameters
    ----------
    mesh : dict
        Beam mesh, as from create_beam_mesh, with elements ordered
        along x.
    u : numpy.ndarray, shape (n_dofs,) or (n_dofs, n_modes)
        Nodal DOF vector, or one column per mode.
    x : float or numpy.ndarray, shape (n_points,)
        Positions along the beam [m].

    Returns
    -------
    numpy.ndarray, shape (n_points,) or (n_points, n_modes)
        Displacement at each position.

    Raises
    ------
    ValueError
        If the mesh is not a beam mesh, or a position lies outside it.
    """
    return _evaluate(mesh, u, x, 'displacement')


def beam_curvature(mesh, u, x):
    """
    Curvature d2w/dx2 at positions x.

    Parameters and returns as for beam_displacement.

    Notes
    -----
    Curvature is linear within each element and generally differs
    between the two elements at a node. At an interior node, the mean
    of the two element values is returned.
    """
    return _evaluate(mesh, u, x, 'curvature')


def surface_strain(mesh, u, x, z):
    """
    Bending strain -z * curvature at a fibre a distance z from the
    neutral axis, positive in the direction of positive w.

    Parameters
    ----------
    mesh, u, x
        As for beam_displacement.
    z : float
        Distance of the gauge from the neutral axis [m]; +h/2 is the
        top surface.

    Returns
    -------
    numpy.ndarray, shape (n_points,) or (n_points, n_modes)
        Strain at each position.
    """
    return -z * beam_curvature(mesh, u, x)


def _evaluate(mesh, u, x, kind):
    """Average of the left and right element values at each position."""
    if mesh.get('element_type') != 'Beam':
        raise ValueError("Response recovery needs a beam mesh")
    nodes_x = mesh['nodes'][:, 0]
    elements = mesh['elements']
    starts = nodes_x[elements[:, 0]]
    ends = nodes_x[elements[:, 1]]

    x = np.atleast_1d(np.asarray(x, dtype=float))
    if np.any(x < starts[0]) or np.any(x > ends[-1]):
        raise ValueError(
            f"Positions must lie within [{starts[0]}, {ends[-1]}]"
        )
    n_el = len(elements)
    # Element whose end is the first at or beyond x, and element whose
    # start is the last at or before x; they differ only at nodes.
    left = np.clip(np.searchsorted(ends, x, side='left'), 0, n_el - 1)
    right = np.clip(np.searchsorted(starts, x, side='right') - 1,
                    0, n_el - 1)

    u = np.asarray(u, dtype=float)
    U = u.reshape(u.shape[0], -1)
    values = 0.5 * (_element_values(mesh, U, x, left, kind)
                    + _element_values(mesh, U, x, right, kind))
    return values[:, 0] if u.ndim == 1 else values


def _element_values(mesh, U, x, elem, kind):
    """Values from the given element at each position."""
    out = np.empty((len(x), U.shape[1]))
    for e in np.unique(elem):
        idx = np.flatnonzero(elem == e)
        element = Beam(mesh['nodes'][mesh['elements'][e]])
        d = U[get_element_dofs(mesh['elements'][e], len(Beam.dof_names))]
        xi = 2.0 * (x[idx] - element.node_coords[0, 0]) / element.length - 1
        for k, xi_k in zip(idx, xi):
            if kind == 'displacement':
                row = element.shape_functions(xi_k)
            else:
                row = element.compute_B_matrix(xi_k)[0][0]
            out[k] = row @ d
    return out
