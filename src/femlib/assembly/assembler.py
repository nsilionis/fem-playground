"""
Assembly routines for global finite element matrices.

This module provides functions for assembling global stiffness matrices
and load vectors from element-level contributions.
"""

import numpy as np


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


def assemble_stiffness(mesh, material, integration_order='full'):
    """
    Assemble global stiffness matrix from element contributions.

    Loops over all elements in the mesh, computes element stiffness
    matrices, and scatters them to the global stiffness matrix using
    the element DOF connectivity.

    The element type is determined from the mesh specification, and
    the appropriate element class is instantiated using a factory pattern.

    Parameters
    ----------
    mesh : dict
        Mesh dictionary containing:
        - 'nodes': (n_nodes, 2) nodal coordinates
        - 'elements': (n_elements, n_nodes_per_elem) connectivity
        - 'element_type': str, element type identifier (e.g., 'Q4', 'Truss')
          If not provided, defaults to 'Q4' for backward compatibility.
    material : Material
        Material object (e.g., LinearElastic2D, LinearElasticTruss)
        providing constitutive properties.
    integration_order : str, optional
        Integration scheme: 'full' or 'reduced'. Default is 'full'.

    Returns
    -------
    K_global : ndarray, shape (n_dofs, n_dofs)
        Global stiffness matrix (symmetric, sparse in practice).

    Examples
    --------
    >>> from femlib.mesh import create_rectangular_mesh
    >>> from femlib.materials import LinearElastic2D
    >>> mesh = create_rectangular_mesh(10.0, 1.0, 4, 2)
    >>> material = LinearElastic2D(E=210e9, nu=0.3, thickness=0.1)
    >>> K = assemble_stiffness(mesh, material, 'full')
    >>> K.shape
    (78, 78)

    >>> # Truss example
    >>> from femlib.materials import LinearElasticTruss
    >>> truss_mesh = {
    ...     'nodes': np.array([[0, 0], [1, 0], [0.5, 1]]),
    ...     'elements': np.array([[0, 1], [1, 2], [2, 0]]),
    ...     'element_type': 'Truss'
    ... }
    >>> material = LinearElasticTruss(E=200e9, A=0.001)
    >>> K = assemble_stiffness(truss_mesh, material, 'reduced')

    Notes
    -----
    The global stiffness matrix is assembled using direct indexing
    rather than explicit construction of connectivity matrices,
    which is more efficient.

    For large problems, consider using sparse matrix formats
    (scipy.sparse) to reduce memory usage.

    Raises
    ------
    ValueError
        If element_type is not recognized or not available.
    """
    # Import element classes (lazy import to avoid circular dependencies)
    from ..elements import Q4Element, Truss

    # Element type registry
    ELEMENT_TYPES = {
        'Q4': Q4Element,
        'Truss': Truss,
    }

    nodes = mesh['nodes']
    elements = mesh['elements']

    # Determine element type from mesh specification
    element_type_str = mesh.get('element_type', 'Q4')  # Default to Q4

    if element_type_str not in ELEMENT_TYPES:
        available = ', '.join(ELEMENT_TYPES.keys())
        raise ValueError(
            f"Unknown element type '{element_type_str}'. "
            f"Available types: {available}"
        )

    ElementClass = ELEMENT_TYPES[element_type_str]

    n_nodes = nodes.shape[0]
    n_dofs = 2 * n_nodes  # 2 DOFs per node (u, v) for 2D problems

    # Initialise global stiffness matrix (dense for now)
    K_global = np.zeros((n_dofs, n_dofs))

    # Loop over elements
    for elem_id, elem_nodes in enumerate(elements):
        # Extract nodal coordinates for this element
        elem_coords = nodes[elem_nodes]

        # Create element object using factory pattern
        element = ElementClass(elem_coords)

        # Compute element stiffness matrix
        K_e = element.compute_stiffness_matrix(
            material,
            integration_order=integration_order
        )

        # Get global DOF indices for this element
        elem_dofs = get_element_dofs(elem_nodes)

        # Scatter element matrix to global matrix
        for i, dof_i in enumerate(elem_dofs):
            for j, dof_j in enumerate(elem_dofs):
                K_global[dof_i, dof_j] += K_e[i, j]

    return K_global
