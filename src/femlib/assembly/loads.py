"""
Load vector assembly with geometric specification.

This module handles various load types (concentrated, distributed edge
loads, body forces) using proper FEM methodology with shape function
integration.
"""

import numpy as np
from ..elements import Q4Element


def identify_nodes_by_geometry(mesh, location_spec, tolerance=1e-6):
    """
    Identify nodes based on geometric criteria.

    Parameters
    ----------
    mesh : dict
        Mesh dictionary containing 'nodes' array.
    location_spec : dict
        Geometric specification, e.g.:
        - {'x': 0.0} - nodes where x = 0
        - {'y': 1.0} - nodes where y = 1
        - {'x': 10.0, 'y': 0.5} - specific point
    tolerance : float, optional
        Geometric tolerance for comparisons. Default is 1e-6.

    Returns
    -------
    node_indices : ndarray
        Array of node indices satisfying the criteria.

    Examples
    --------
    >>> nodes = identify_nodes_by_geometry(mesh, {'x': 0.0})
    >>> nodes = identify_nodes_by_geometry(mesh, {'y': 1.0})
    """
    nodes = mesh['nodes']
    mask = np.ones(nodes.shape[0], dtype=bool)

    if 'x' in location_spec:
        x_target = location_spec['x']
        mask &= np.abs(nodes[:, 0] - x_target) < tolerance

    if 'y' in location_spec:
        y_target = location_spec['y']
        mask &= np.abs(nodes[:, 1] - y_target) < tolerance

    return np.where(mask)[0]


def find_nearest_node(mesh, location):
    """
    Find the node nearest to a specified location.

    Parameters
    ----------
    mesh : dict
        Mesh dictionary containing 'nodes' array.
    location : tuple or array_like
        Target coordinates (x, y).

    Returns
    -------
    node_id : int
        Index of the nearest node.

    Examples
    --------
    >>> node = find_nearest_node(mesh, (10.0, 0.5))
    """
    nodes = mesh['nodes']
    location = np.asarray(location)

    distances = np.linalg.norm(nodes - location, axis=1)
    return np.argmin(distances)


def identify_edge_elements(mesh, edge):
    """
    Identify elements that lie on a specified mesh edge.

    Parameters
    ----------
    mesh : dict
        Mesh dictionary with 'elements' and 'boundary_nodes'.
    edge : str
        Edge identifier: 'left', 'right', 'top', 'bottom'.

    Returns
    -------
    edge_elements : list of tuples
        Each tuple contains (element_id, local_edge_nodes).
        local_edge_nodes are the two node indices (0-3) within
        the element that form the edge.

    Examples
    --------
    >>> edge_elems = identify_edge_elements(mesh, 'right')
    """
    elements = mesh['elements']
    boundary_nodes = mesh['boundary_nodes'][edge]
    boundary_set = set(boundary_nodes)

    edge_elements = []

    for elem_id, elem_nodes in enumerate(elements):
        # Find which nodes of this element are on the boundary
        local_boundary_nodes = []
        for local_id, global_node in enumerate(elem_nodes):
            if global_node in boundary_set:
                local_boundary_nodes.append(local_id)

        # If exactly 2 nodes on boundary and they're adjacent,
        # this element has an edge on the boundary
        if len(local_boundary_nodes) == 2:
            # Check if nodes are adjacent in Q4 numbering
            # Q4 edges: (0,1), (1,2), (2,3), (3,0)
            n0, n1 = sorted(local_boundary_nodes)

            # Adjacent if differ by 1, or are 0 and 3
            is_adjacent = (n1 - n0 == 1) or (n0 == 0 and n1 == 3)

            if is_adjacent:
                edge_elements.append((elem_id, local_boundary_nodes))

    return edge_elements


def compute_edge_load_vector(element, local_edge_nodes,
                             load_density, n_gauss=2):
    """
    Compute element load vector for distributed edge load.

    Uses Gauss quadrature along the edge with element shape
    functions to create consistent load vector.

    Parameters
    ----------
    element : Q4Element
        Element object with nodal coordinates.
    local_edge_nodes : list
        Local node indices (0-3) forming the loaded edge.
    load_density : array_like
        Load per unit length [qx, qy] in global coordinates.
    n_gauss : int, optional
        Number of Gauss points for edge integration.
        Default is 2.

    Returns
    -------
    f_e : ndarray, shape (8,)
        Element load vector.

    Notes
    -----
    The load vector is computed as:
        f_e = ∫_edge N^T q |J_edge| dξ

    where |J_edge| is the Jacobian of the edge parametrisation.
    """
    load_density = np.asarray(load_density)
    f_e = np.zeros(element.n_dof)

    # Map local edge nodes to Q4 edge parametrisation
    # Q4 edges in natural coordinates:
    # Edge 0-1: η = -1, ξ ∈ [-1, 1]
    # Edge 1-2: ξ = +1, η ∈ [-1, 1]
    # Edge 2-3: η = +1, ξ ∈ [-1, 1]
    # Edge 3-0: ξ = -1, η ∈ [-1, 1]

    n0, n1 = sorted(local_edge_nodes)

    # Determine which edge and parametrisation
    if n0 == 0 and n1 == 1:
        # Bottom edge: η = -1, ξ varies
        def xi_eta(xi_1d):
            return xi_1d, -1.0
    elif n0 == 1 and n1 == 2:
        # Right edge: ξ = +1, η varies
        def xi_eta(xi_1d):
            return 1.0, xi_1d
    elif n0 == 2 and n1 == 3:
        # Top edge: η = +1, ξ varies
        def xi_eta(xi_1d):
            return xi_1d, 1.0
    elif n0 == 0 and n1 == 3:
        # Left edge: ξ = -1, η varies
        def xi_eta(xi_1d):
            return -1.0, xi_1d
    else:
        raise ValueError(
            f"Invalid edge nodes: {local_edge_nodes}"
        )

    # Gauss quadrature points and weights for 1D integration
    if n_gauss == 1:
        gauss_points = np.array([0.0])
        gauss_weights = np.array([2.0])
    elif n_gauss == 2:
        gp = 1.0 / np.sqrt(3.0)
        gauss_points = np.array([-gp, gp])
        gauss_weights = np.array([1.0, 1.0])
    elif n_gauss == 3:
        gauss_points = np.array([-np.sqrt(3.0/5.0), 0.0,
                                 np.sqrt(3.0/5.0)])
        gauss_weights = np.array([5.0/9.0, 8.0/9.0, 5.0/9.0])
    else:
        raise ValueError(
            f"n_gauss must be 1, 2, or 3, got {n_gauss}"
        )

    # Integrate along edge
    for xi_1d, weight in zip(gauss_points, gauss_weights):
        xi, eta = xi_eta(xi_1d)

        # Evaluate shape functions at this point
        N = element.shape_functions(xi, eta)

        # Compute edge Jacobian (length scaling factor)
        # For edge parametrised by ξ_1d, we need |dx/dξ_1d|
        dN_dxi, dN_deta = element.shape_function_derivatives(xi, eta)

        # Tangent vector along edge
        if n0 == 0 and n1 == 1:  # Bottom edge (ξ varies)
            dx_dxi = np.sum(dN_dxi * element.node_coords[:, 0])
            dy_dxi = np.sum(dN_dxi * element.node_coords[:, 1])
            tangent = np.array([dx_dxi, dy_dxi])
        elif n0 == 1 and n1 == 2:  # Right edge (η varies)
            dx_deta = np.sum(dN_deta * element.node_coords[:, 0])
            dy_deta = np.sum(dN_deta * element.node_coords[:, 1])
            tangent = np.array([dx_deta, dy_deta])
        elif n0 == 2 and n1 == 3:  # Top edge (ξ varies)
            dx_dxi = np.sum(dN_dxi * element.node_coords[:, 0])
            dy_dxi = np.sum(dN_dxi * element.node_coords[:, 1])
            tangent = np.array([dx_dxi, dy_dxi])
        else:  # Left edge (η varies)
            dx_deta = np.sum(dN_deta * element.node_coords[:, 0])
            dy_deta = np.sum(dN_deta * element.node_coords[:, 1])
            tangent = np.array([dx_deta, dy_deta])

        # Edge Jacobian (differential length)
        J_edge = np.linalg.norm(tangent)

        # Assemble element load vector: f = ∫ N^T q |J| dξ
        for i in range(element.n_nodes):
            f_e[2*i] += weight * N[i] * load_density[0] * J_edge
            f_e[2*i + 1] += weight * N[i] * load_density[1] * J_edge

    return f_e


def assemble_load_vector(mesh, load_specification):
    """
    Assemble global load vector from geometric load specification.

    Supports concentrated loads and distributed edge loads using
    proper FEM methodology with shape function integration.

    Parameters
    ----------
    mesh : dict
        Mesh dictionary containing 'nodes', 'elements',
        'boundary_nodes'.
    load_specification : dict
        Load specification with geometric information. Format:
        {
            'concentrated': [
                {
                    'location': (x, y) or {'x': val, 'y': val},
                    'force': [Fx, Fy]
                },
                ...
            ],
            'distributed': [
                {
                    'edge': 'left'|'right'|'top'|'bottom',
                    'load_density': [qx, qy]  # Force per length
                },
                ...
            ]
        }

    Returns
    -------
    F_global : ndarray, shape (n_dofs,)
        Global load vector.

    Examples
    --------
    >>> loads = {
    ...     'concentrated': [
    ...         {'location': (10.0, 0.5), 'force': [0.0, -5000.0]}
    ...     ],
    ...     'distributed': [
    ...         {'edge': 'right', 'load_density': [0.0, -1000.0]}
    ...     ]
    ... }
    >>> F = assemble_load_vector(mesh, loads)

    Notes
    -----
    Distributed loads use consistent load vectors computed via
    numerical integration with element shape functions, not
    simple nodal averaging.
    """
    nodes = mesh['nodes']
    elements = mesh['elements']
    n_nodes = nodes.shape[0]
    n_dofs = 2 * n_nodes

    F_global = np.zeros(n_dofs)

    # Process concentrated loads
    if 'concentrated' in load_specification:
        for load_spec in load_specification['concentrated']:
            location = load_spec['location']
            force = np.asarray(load_spec['force'])

            if force.shape[0] != 2:
                raise ValueError(
                    f"Force must be 2D [Fx, Fy], got {force}"
                )

            # Find node at or nearest to location
            if isinstance(location, dict):
                # Geometric specification like {'x': 10.0}
                node_candidates = identify_nodes_by_geometry(
                    mesh, location
                )
                if len(node_candidates) == 0:
                    raise ValueError(
                        f"No nodes found matching {location}"
                    )
                # If multiple nodes, use first (or could average)
                node_id = node_candidates[0]
            else:
                # Point specification like (10.0, 0.5)
                node_id = find_nearest_node(mesh, location)

            # Apply force to node DOFs
            F_global[2*node_id] += force[0]
            F_global[2*node_id + 1] += force[1]

    # Process distributed edge loads
    if 'distributed' in load_specification:
        for load_spec in load_specification['distributed']:
            edge = load_spec['edge']
            load_density = np.asarray(load_spec['load_density'])

            if load_density.shape[0] != 2:
                raise ValueError(
                    f"Load density must be 2D [qx, qy], "
                    f"got {load_density}"
                )

            # Identify elements on this edge
            edge_elements = identify_edge_elements(mesh, edge)

            # For each edge element, compute and assemble load vector
            for elem_id, local_edge_nodes in edge_elements:
                elem_nodes = elements[elem_id]
                elem_coords = nodes[elem_nodes]

                # Create element
                element = Q4Element(elem_coords)

                # Compute element load vector for this edge
                f_e = compute_edge_load_vector(
                    element,
                    local_edge_nodes,
                    load_density,
                    n_gauss=2
                )

                # Get global DOF indices
                from .assembler import get_element_dofs
                elem_dofs = get_element_dofs(elem_nodes)

                # Scatter to global load vector
                for i, dof in enumerate(elem_dofs):
                    F_global[dof] += f_e[i]

    return F_global
