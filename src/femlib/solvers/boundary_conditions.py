"""
Boundary condition application for finite element models.

This module provides utilities for identifying boundary nodes
geometrically and applying Dirichlet (essential) boundary conditions
to the global system.
"""

import numpy as np


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


def apply_dirichlet_bc(K, F, bc_specification, method='elimination'):
    """
    Apply Dirichlet (essential) boundary conditions to the system.

    Modifies the global stiffness matrix K and force vector F to
    enforce prescribed displacements at specified DOFs.

    Parameters
    ----------
    K : ndarray, shape (n_dofs, n_dofs)
        Global stiffness matrix.
    F : ndarray, shape (n_dofs,)
        Global force vector.
    bc_specification : dict
        Boundary condition specification. Two formats supported:

        **Format 1 (Legacy - same constraint for all nodes):**
        {
            'nodes': array of node indices,
            'dofs': list with single DOF type to constrain,
            'values': list with single prescribed value
        }
        DOF type: 'x', 'y', or 'both' (applied to ALL nodes)
        Value: prescribed displacement (applied to ALL constrained DOFs)

        **Format 2 (Flexible - individual constraints per node):**
        {
            'constraints': [
                {'node': int, 'dof': str, 'value': float},
                {'node': int, 'dof': str, 'value': float},
                ...
            ]
        }
        Each constraint dict specifies one node with its DOF type and value.

    method : str, optional
        Method for applying BCs:
        - 'elimination': Remove constrained DOFs (reduces system size)
        - 'penalty': Use penalty method (keeps system size)
        Default is 'elimination'.

    Returns
    -------
    K_bc : ndarray
        Modified stiffness matrix.
    F_bc : ndarray
        Modified force vector.
    constrained_dofs : ndarray
        Array of constrained DOF indices (for post-processing).
    prescribed_values : ndarray
        Corresponding prescribed values.

    Examples
    --------
    >>> # Legacy format: Fix all nodes in both directions
    >>> bc = {
    ...     'nodes': np.array([0, 5, 10]),
    ...     'dofs': ['both'],
    ...     'values': [0.0]
    ... }
    >>> K_bc, F_bc, c_dofs, c_vals = apply_dirichlet_bc(K, F, bc)

    >>> # Flexible format: Different constraints per node
    >>> bc = {
    ...     'constraints': [
    ...         {'node': 0, 'dof': 'both', 'value': 0.0},
    ...         {'node': 1, 'dof': 'x', 'value': 0.0},
    ...         {'node': 2, 'dof': 'y', 'value': 0.5}
    ...     ]
    ... }
    >>> K_bc, F_bc, c_dofs, c_vals = apply_dirichlet_bc(K, F, bc)

    Notes
    -----
    The elimination method is more accurate but requires solving
    a reduced system and then recovering the full displacement vector.

    For fixed supports (zero displacement), values should be [0.0].
    """
    # Build list of constrained DOFs and their prescribed values
    constrained_dofs = []
    prescribed_values = []

    # Check which format is being used
    if 'constraints' in bc_specification:
        # Format 2: Flexible individual constraints
        for constraint in bc_specification['constraints']:
            node = constraint['node']
            dof_type = constraint['dof']
            value = constraint['value']

            if dof_type == 'x':
                constrained_dofs.append(2 * node)
                prescribed_values.append(value)
            elif dof_type == 'y':
                constrained_dofs.append(2 * node + 1)
                prescribed_values.append(value)
            elif dof_type == 'both':
                constrained_dofs.extend([2 * node, 2 * node + 1])
                prescribed_values.extend([value, value])
            else:
                raise ValueError(
                    f"Unknown DOF type '{dof_type}'. "
                    f"Must be 'x', 'y', or 'both'."
                )

    else:
        # Format 1: Legacy format (same constraint for all nodes)
        nodes = bc_specification['nodes']
        dof_types = bc_specification['dofs']
        values = bc_specification['values']

        # Validate legacy format: should have single dof type and value
        if len(dof_types) != 1 or len(values) != 1:
            raise ValueError(
                "Legacy format requires exactly one DOF type and one value. "
                "For multiple different constraints, use the flexible format:\n"
                "bc = {'constraints': [{'node': i, 'dof': 'x', 'value': v}, ...]}"
            )

        dof_type = dof_types[0]
        value = values[0]

        # Apply same constraint to all nodes
        for node in nodes:
            if dof_type == 'x':
                constrained_dofs.append(2 * node)
                prescribed_values.append(value)
            elif dof_type == 'y':
                constrained_dofs.append(2 * node + 1)
                prescribed_values.append(value)
            elif dof_type == 'both':
                constrained_dofs.extend([2 * node, 2 * node + 1])
                prescribed_values.extend([value, value])
            else:
                raise ValueError(
                    f"Unknown DOF type '{dof_type}'. "
                    f"Must be 'x', 'y', or 'both'."
                )

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

    For prescribed values u_c at constrained DOFs:
    - Moves K_fc * u_c to RHS
    - Removes constrained rows/columns

    Returns reduced system.
    """
    n_dofs = K.shape[0]

    # Identify free DOFs (not constrained)
    all_dofs = np.arange(n_dofs)
    free_dofs = np.setdiff1d(all_dofs, constrained_dofs)

    # Modify force vector: F_f = F_f - K_fc * u_c
    F_modified = F.copy()
    for i, (c_dof, c_val) in enumerate(
        zip(constrained_dofs, prescribed_values)
    ):
        if np.abs(c_val) > 1e-14:  # Non-zero prescribed displacement
            # Subtract contribution from constrained DOF
            F_modified -= K[:, c_dof] * c_val

    # Extract reduced system (free DOFs only)
    K_reduced = K[np.ix_(free_dofs, free_dofs)]
    F_reduced = F_modified[free_dofs]

    return K_reduced, F_reduced


def _apply_bc_penalty(K, F, constrained_dofs, prescribed_values,
                      penalty=1e10):
    """
    Apply BCs using penalty method.

    Adds large spring stiffness (penalty) to diagonal of constrained
    DOFs and modifies RHS accordingly.

    Keeps system size unchanged.
    """
    K_bc = K.copy()
    F_bc = F.copy()

    for c_dof, c_val in zip(constrained_dofs, prescribed_values):
        # Add penalty to diagonal
        K_bc[c_dof, c_dof] += penalty

        # Modify RHS
        F_bc[c_dof] += penalty * c_val

    return K_bc, F_bc
