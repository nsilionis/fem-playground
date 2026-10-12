"""
Assembly routines for global finite element matrices.

This module provides functions for assembling global stiffness and
mass matrices from element-level contributions, as dense arrays or as
sparse matrices.
"""

import numpy as np
import scipy.sparse as sp


def get_element_dofs(element_nodes, dofs_per_node=2):
    """
    Convert element node indices to global DOF indices.

    Uses interleaved DOF numbering: node i has DOFs [2*i, 2*i+1]
    corresponding to [u_i, v_i] for 2D problems.

    Parameters
    ----------
    element_nodes : array_like
        Global node indices for the element.
    dofs_per_node : int, optional
        Number of DOFs per node. Default is 2 (for 2D problems).

    Returns
    -------
    element_dofs : ndarray
        Global DOF indices for the element in order:
        [u_0, v_0, u_1, v_1, ..., u_n, v_n]

    Examples
    --------
    >>> get_element_dofs([5, 6, 12, 11])
    array([10, 11, 12, 13, 24, 25, 22, 23])
    """
    element_nodes = np.asarray(element_nodes, dtype=int)
    n_nodes = len(element_nodes)
    element_dofs = np.zeros(n_nodes * dofs_per_node, dtype=int)

    for i, node in enumerate(element_nodes):
        for j in range(dofs_per_node):
            element_dofs[i * dofs_per_node + j] = (
                node * dofs_per_node + j
            )

    return element_dofs


def _element_class(mesh):
    """Element class named by mesh['element_type'] (default 'Q4')."""
    # Lazy import to avoid circular dependencies
    from ..elements import Beam, Q4Element, Truss

    element_types = {'Q4': Q4Element, 'Truss': Truss, 'Beam': Beam}
    name = mesh.get('element_type', 'Q4')
    if name not in element_types:
        raise ValueError(
            f"Unknown element type '{name}'. "
            f"Available types: {', '.join(element_types)}"
        )
    return element_types[name]


def _assemble(mesh, material, section, method, integration_order, sparse):
    """Assemble the global matrix given by an element method."""
    ElementClass = _element_class(mesh)
    if not hasattr(ElementClass, method):
        raise ValueError(
            f"Element type '{mesh.get('element_type', 'Q4')}' does not "
            f"provide {method}"
        )
    nodes = mesh['nodes']
    elements = mesh['elements']
    dofs_per_node = len(ElementClass.dof_names)
    n_dofs = dofs_per_node * nodes.shape[0]

    if sparse:
        rows, cols, values = [], [], []
    else:
        A_global = np.zeros((n_dofs, n_dofs))

    for elem_nodes in elements:
        element = ElementClass(nodes[elem_nodes])
        A_e = getattr(element, method)(
            material, section, integration_order=integration_order
        )
        elem_dofs = get_element_dofs(elem_nodes, dofs_per_node)

        if sparse:
            n_e = len(elem_dofs)
            rows.append(np.repeat(elem_dofs, n_e))
            cols.append(np.tile(elem_dofs, n_e))
            values.append(A_e.ravel())
        else:
            for i, dof_i in enumerate(elem_dofs):
                for j, dof_j in enumerate(elem_dofs):
                    A_global[dof_i, dof_j] += A_e[i, j]

    if sparse:
        # Entries at repeated positions are summed by the conversion
        triplets = (np.concatenate(values),
                    (np.concatenate(rows), np.concatenate(cols)))
        return sp.coo_array(triplets, shape=(n_dofs, n_dofs)).tocsr()
    return A_global


def assemble_stiffness(mesh, material, section, integration_order='full',
                       sparse=False):
    """
    Assemble global stiffness matrix from element contributions.

    Loops over all elements in the mesh, computes element stiffness
    matrices, and adds them into the global matrix at the element DOF
    indices. The element type, and with it the number of DOFs per
    node, is taken from the mesh.

    Parameters
    ----------
    mesh : dict
        Mesh dictionary containing:
        - 'nodes': (n_nodes, spatial_dim) nodal coordinates
        - 'elements': (n_elements, n_nodes_per_elem) connectivity
        - 'element_type': 'Q4', 'Truss' or 'Beam'. Defaults to 'Q4'.
    material : Material
        Material object (e.g., LinearElastic2D, LinearElastic1D)
        providing constitutive properties.
    section : Section
        Section object (e.g., PlaneSection, TrussSection, BeamSection)
        providing cross-sectional properties.
    integration_order : str or int, optional
        Integration scheme passed to each element. Default is 'full'.
    sparse : bool, optional
        If True, return a scipy.sparse.csr_array. Default is False.

    Returns
    -------
    K_global : numpy.ndarray or scipy.sparse.csr_array,
               shape (n_dofs, n_dofs)
        Global stiffness matrix.

    Raises
    ------
    ValueError
        If element_type is not recognised.

    Examples
    --------
    >>> from femlib.mesh import create_rectangular_mesh
    >>> from femlib.materials import LinearElastic2D
    >>> from femlib.sections import PlaneSection
    >>> mesh = create_rectangular_mesh(10.0, 1.0, 4, 2)
    >>> material = LinearElastic2D(E=210e9, nu=0.3)
    >>> section = PlaneSection(thickness=0.1)
    >>> K = assemble_stiffness(mesh, material, section, 'full')
    >>> K.shape
    (30, 30)

    >>> # Truss example
    >>> from femlib.materials import LinearElastic1D
    >>> from femlib.sections import TrussSection
    >>> truss_mesh = {
    ...     'nodes': np.array([[0, 0], [1, 0], [0.5, 1]]),
    ...     'elements': np.array([[0, 1], [1, 2], [2, 0]]),
    ...     'element_type': 'Truss'
    ... }
    >>> material = LinearElastic1D(E=200e9)
    >>> section = TrussSection(A=0.001)
    >>> K = assemble_stiffness(truss_mesh, material, section, 'reduced')
    """
    return _assemble(mesh, material, section, 'compute_stiffness_matrix',
                     integration_order, sparse)


def assemble_mass(mesh, material, section, integration_order='full',
                  sparse=False):
    """
    Assemble global consistent mass matrix from element contributions.

    Parameters
    ----------
    mesh : dict
        Mesh dictionary, as for assemble_stiffness.
    material : Material
        Material object providing the density (e.g., LinearElastic1D
        with rho set).
    section : Section
        Section object providing the area (e.g., BeamSection with A
        set).
    integration_order : str or int, optional
        Integration scheme passed to each element. Default is 'full'.
    sparse : bool, optional
        If True, return a scipy.sparse.csr_array. Default is False.

    Returns
    -------
    M_global : numpy.ndarray or scipy.sparse.csr_array,
               shape (n_dofs, n_dofs)
        Global mass matrix.

    Raises
    ------
    ValueError
        If element_type is not recognised, or the element type has no
        mass matrix.

    Examples
    --------
    >>> from femlib.mesh import create_beam_mesh
    >>> from femlib.materials import LinearElastic1D
    >>> from femlib.sections import BeamSection
    >>> mesh = create_beam_mesh(2.0, 10)
    >>> M = assemble_mass(mesh, LinearElastic1D(E=210e9, rho=7850.0),
    ...                   BeamSection(I=8.0e-6, A=0.01))
    """
    return _assemble(mesh, material, section, 'compute_mass_matrix',
                     integration_order, sparse)
