"""
Boundary condition application for finite element models.

This module provides utilities for identifying boundary nodes
geometrically and applying Dirichlet (essential) boundary conditions
to the global system.
"""

import numpy as np
import scipy.sparse as sp


def identify_boundary_nodes(mesh, boundary_spec, tolerance=1e-6):
    """
    Identify nodes based on geometric boundary specification.

    Parameters
    ----------
    mesh : dict
        Mesh dictionary containing 'nodes' array.
    boundary_spec : dict or str
        Geometric specification for boundary identification.
        Can be:
        - str: 'left', 'right', 'top', 'bottom' (uses mesh edges)
        - dict: {'x': value}, {'y': value}, or {'x': val, 'y': val}
    tolerance : float, optional
        Geometric tolerance for coordinate comparisons.
        Default is 1e-6.

    Returns
    -------
    node_indices : ndarray
        Array of global node indices on the specified boundary.

    Examples
    --------
    >>> nodes = identify_boundary_nodes(mesh, 'left')
    >>> nodes = identify_boundary_nodes(mesh, {'x': 0.0})
    >>> nodes = identify_boundary_nodes(mesh, {'y': 1.0})
    """
    # If string, use pre-identified boundary nodes
    if isinstance(boundary_spec, str):
        if boundary_spec not in mesh['boundary_nodes']:
            raise ValueError(
                f"Unknown boundary '{boundary_spec}'. "
                f"Available: {list(mesh['boundary_nodes'].keys())}"
            )
        return mesh['boundary_nodes'][boundary_spec]

    # Otherwise, use geometric criteria
    nodes = mesh['nodes']
    mask = np.ones(nodes.shape[0], dtype=bool)

    if 'x' in boundary_spec:
        x_target = boundary_spec['x']
        mask &= np.abs(nodes[:, 0] - x_target) < tolerance

    if 'y' in boundary_spec:
        y_target = boundary_spec['y']
        mask &= np.abs(nodes[:, 1] - y_target) < tolerance

    node_indices = np.where(mask)[0]

    if len(node_indices) == 0:
        raise ValueError(
            f"No nodes found matching boundary specification "
            f"{boundary_spec} with tolerance {tolerance}"
        )

    return node_indices


def _dof_indices(node, dof, dof_names):
    """Global DOF indices for a node and a DOF name, or 'all'."""
    n = len(dof_names)
    if dof == 'all':
        return [n * node + k for k in range(n)]
    if dof not in dof_names:
        valid = ", ".join(repr(d) for d in (*dof_names, 'all'))
        raise ValueError(f"Unknown DOF '{dof}'. Must be one of {valid}.")
    return [n * node + dof_names.index(dof)]


def apply_dirichlet_bc(K, F, bc_specification, method='elimination',
                       dof_names=('ux', 'uy')):
    """
    Apply Dirichlet (essential) boundary conditions to the system.

    Modifies the global stiffness matrix K and force vector F to
    enforce prescribed displacements at specified DOFs. Node i owns
    the global DOFs n*i to n*i + n - 1, with n = len(dof_names), in the
    order of dof_names.

    Parameters
    ----------
    K : numpy.ndarray or scipy.sparse array, shape (n_dofs, n_dofs)
        Global stiffness matrix.
    F : numpy.ndarray, shape (n_dofs,)
        Global force vector.
    bc_specification : dict
        Boundary condition specification. Two formats supported:

        Format 1 (same constraint for all nodes):
        {
            'nodes': array of node indices,
            'dofs': list with a single DOF name,
            'values': list with a single prescribed value
        }

        Format 2 (individual constraints per node):
        {
            'constraints': [
                {'node': int, 'dof': str, 'value': float},
                ...
            ]
        }

        A DOF name is one of dof_names, or 'all' for every DOF at the
        node.
    method : str, optional
        'elimination' removes the constrained DOFs; 'penalty' keeps the
        system size. Default is 'elimination'.
    dof_names : tuple of str, optional
        DOF names per node, as declared by the element class (e.g.
        Beam.dof_names). Default is ('ux', 'uy').

    Returns
    -------
    K_bc : numpy.ndarray or scipy.sparse.csr_array
        Modified stiffness matrix, of the same kind as K.
    F_bc : numpy.ndarray
        Modified force vector.
    constrained_dofs : numpy.ndarray
        Constrained DOF indices (for post-processing).
    prescribed_values : numpy.ndarray
        Corresponding prescribed values.

    Raises
    ------
    ValueError
        If a DOF name is unknown, the format 1 lists are not single
        entries, or the method is unknown.

    Examples
    --------
    >>> # Format 1: fix all DOFs at three nodes
    >>> bc = {
    ...     'nodes': np.array([0, 5, 10]),
    ...     'dofs': ['all'],
    ...     'values': [0.0]
    ... }
    >>> K_bc, F_bc, c_dofs, c_vals = apply_dirichlet_bc(K, F, bc)

    >>> # Format 2: different constraints per node
    >>> bc = {
    ...     'constraints': [
    ...         {'node': 0, 'dof': 'all', 'value': 0.0},
    ...         {'node': 1, 'dof': 'ux', 'value': 0.0},
    ...         {'node': 2, 'dof': 'uy', 'value': 0.5}
    ...     ]
    ... }
    >>> K_bc, F_bc, c_dofs, c_vals = apply_dirichlet_bc(K, F, bc)

    >>> # Beam: clamp node 0, simple support at node 4
    >>> bc = {
    ...     'constraints': [
    ...         {'node': 0, 'dof': 'all', 'value': 0.0},
    ...         {'node': 4, 'dof': 'w', 'value': 0.0}
    ...     ]
    ... }
    >>> K_bc, F_bc, c_dofs, c_vals = apply_dirichlet_bc(
    ...     K, F, bc, dof_names=Beam.dof_names)
    """
    constrained_dofs = []
    prescribed_values = []

    if 'constraints' in bc_specification:
        # Format 2: individual constraints
        for constraint in bc_specification['constraints']:
            indices = _dof_indices(
                constraint['node'], constraint['dof'], dof_names
            )
            constrained_dofs.extend(indices)
            prescribed_values.extend([constraint['value']] * len(indices))

    else:
        # Format 1: same constraint for all nodes
        nodes = bc_specification['nodes']
        dof_types = bc_specification['dofs']
        values = bc_specification['values']

        if len(dof_types) != 1 or len(values) != 1:
            raise ValueError(
                "Format 1 requires exactly one DOF name and one value. "
                "For different constraints per node, use format 2:\n"
                "bc = {'constraints': [{'node': i, 'dof': 'ux', "
                "'value': v}, ...]}"
            )

        for node in nodes:
            indices = _dof_indices(node, dof_types[0], dof_names)
            constrained_dofs.extend(indices)
            prescribed_values.extend([values[0]] * len(indices))

    constrained_dofs = np.array(constrained_dofs, dtype=int)
    prescribed_values = np.array(prescribed_values)

    if method == 'elimination':
        K_bc, F_bc = _apply_bc_elimination(
            K, F, constrained_dofs, prescribed_values
        )
    elif method == 'penalty':
        K_bc, F_bc = _apply_bc_penalty(
            K, F, constrained_dofs, prescribed_values
        )
    else:
        raise ValueError(
            f"Unknown BC method '{method}'. "
            f"Must be 'elimination' or 'penalty'."
        )

    return K_bc, F_bc, constrained_dofs, prescribed_values


def _apply_bc_elimination(K, F, constrained_dofs, prescribed_values):
    """
    Apply BCs using elimination method (row/column removal).

    Moves K_fc * u_c to the right-hand side and removes the constrained
    rows and columns. Returns the reduced system.
    """
    n_dofs = K.shape[0]
    free_dofs = np.setdiff1d(np.arange(n_dofs), constrained_dofs)

    # F_f = F_f - K_fc u_c, with u_c zero except at the constrained DOFs
    u_c = np.zeros(n_dofs)
    u_c[constrained_dofs] = prescribed_values
    F_modified = F - K @ u_c

    if sp.issparse(K):
        K_reduced = K[free_dofs][:, free_dofs]
    else:
        K_reduced = K[np.ix_(free_dofs, free_dofs)]
    return K_reduced, F_modified[free_dofs]


def _apply_bc_penalty(K, F, constrained_dofs, prescribed_values,
                      penalty=1e10):
    """
    Apply BCs using penalty method.

    Adds a large spring stiffness (penalty) to the diagonal of the
    constrained DOFs and modifies the right-hand side accordingly.
    Keeps the system size unchanged.
    """
    if sp.issparse(K):
        P = sp.coo_array(
            (np.full(len(constrained_dofs), penalty),
             (constrained_dofs, constrained_dofs)),
            shape=K.shape,
        )
        K_bc = (K + P).tocsr()
    else:
        K_bc = K.copy()
        for c_dof in constrained_dofs:
            K_bc[c_dof, c_dof] += penalty

    F_bc = F.copy()
    for c_dof, c_val in zip(constrained_dofs, prescribed_values):
        F_bc[c_dof] += penalty * c_val

    return K_bc, F_bc
