"""
Structured mesh generation for rectangular domains.

This module provides utilities for generating structured quadrilateral
meshes suitable for Q4 finite element analysis.
"""

import numpy as np


def create_rectangular_mesh(Lx, Ly, nx, ny):
    """
    Create a structured Q4 mesh for a rectangular domain.

    Generates a uniform mesh with counter-clockwise element node
    ordering. Nodes are numbered row-by-row, left-to-right,
    bottom-to-top. Elements are numbered similarly.

    Parameters
    ----------
    Lx : float
        Length of domain in x-direction.
    Ly : float
        Length of domain in y-direction.
    nx : int
        Number of elements in x-direction.
    ny : int
        Number of elements in y-direction.

    Returns
    -------
    mesh : dict
        Dictionary containing:
        - 'nodes': ndarray, shape (n_nodes, 2)
            Nodal coordinates [x, y].
        - 'elements': ndarray, shape (n_elements, 4)
            Element connectivity with counter-clockwise node ordering.
            Each row contains global node indices [n0, n1, n2, n3].
        - 'boundary_nodes': dict
            Dictionary with keys 'left', 'right', 'bottom', 'top',
            each containing arrays of boundary node indices.

    Raises
    ------
    ValueError
        If dimensions or mesh density parameters are invalid.

    Examples
    --------
    >>> mesh = create_rectangular_mesh(Lx=10.0, Ly=1.0, nx=4, ny=1)
    >>> mesh['nodes'].shape
    (10, 2)
    >>> mesh['elements'].shape
    (4, 4)

    Notes
    -----
    Node numbering (example: nx=3, ny=2):
        6---7---8---9
        |   |   |   |
        |   |   |   |
        2---3---4---5
        |   |   |   |
        |   |   |   |
        0---1---2---3

    Element local node ordering (counter-clockwise):
        3-------2
        |       |
        |       |
        0-------1
    """
    # Validate inputs
    if Lx <= 0 or Ly <= 0:
        raise ValueError(
            f"Domain dimensions must be positive: Lx={Lx}, Ly={Ly}"
        )
    if nx < 1 or ny < 1:
        raise ValueError(
            f"Mesh density must be at least 1: nx={nx}, ny={ny}"
        )

    # Calculate number of nodes
    n_nodes_x = nx + 1
    n_nodes_y = ny + 1
    n_nodes = n_nodes_x * n_nodes_y
    n_elements = nx * ny

    # Generate nodal coordinates
    x_coords = np.linspace(0.0, Lx, n_nodes_x)
    y_coords = np.linspace(0.0, Ly, n_nodes_y)

    nodes = np.zeros((n_nodes, 2))
    for j in range(n_nodes_y):
        for i in range(n_nodes_x):
            node_id = j * n_nodes_x + i
            nodes[node_id, 0] = x_coords[i]
            nodes[node_id, 1] = y_coords[j]

    # Generate element connectivity (counter-clockwise)
    elements = np.zeros((n_elements, 4), dtype=int)
    for j in range(ny):
        for i in range(nx):
            element_id = j * nx + i

            # Bottom-left node of element
            n0 = j * n_nodes_x + i

            # Counter-clockwise node ordering:
            # 0: bottom-left, 1: bottom-right
            # 2: top-right, 3: top-left
            elements[element_id, 0] = n0
            elements[element_id, 1] = n0 + 1
            elements[element_id, 2] = n0 + n_nodes_x + 1
            elements[element_id, 3] = n0 + n_nodes_x

    # Identify boundary nodes
    boundary_nodes = {}

    # Left edge (x = 0)
    boundary_nodes['left'] = np.arange(
        0, n_nodes, n_nodes_x, dtype=int
    )

    # Right edge (x = Lx)
    boundary_nodes['right'] = np.arange(
        n_nodes_x - 1, n_nodes, n_nodes_x, dtype=int
    )

    # Bottom edge (y = 0)
    boundary_nodes['bottom'] = np.arange(
        0, n_nodes_x, dtype=int
    )

    # Top edge (y = Ly)
    boundary_nodes['top'] = np.arange(
        n_nodes - n_nodes_x, n_nodes, dtype=int
    )

    # Assemble mesh dictionary
    mesh = {
        'nodes': nodes,
        'elements': elements,
        'boundary_nodes': boundary_nodes
    }

    return mesh


def mesh_statistics(mesh):
    """
    Compute and return basic mesh statistics.

    Parameters
    ----------
    mesh : dict
        Mesh dictionary from create_rectangular_mesh().

    Returns
    -------
    stats : dict
        Dictionary containing:
        - 'n_nodes': Number of nodes
        - 'n_elements': Number of elements
        - 'domain_x': [x_min, x_max]
        - 'domain_y': [y_min, y_max]
    """
    nodes = mesh['nodes']
    elements = mesh['elements']

    stats = {
        'n_nodes': nodes.shape[0],
        'n_elements': elements.shape[0],
        'domain_x': [nodes[:, 0].min(), nodes[:, 0].max()],
        'domain_y': [nodes[:, 1].min(), nodes[:, 1].max()]
    }

    return stats


def visualise_mesh(mesh, show_node_numbers=False,
                   show_element_numbers=False):
    """
    Visualise the mesh using matplotlib.

    Parameters
    ----------
    mesh : dict
        Mesh dictionary from create_rectangular_mesh().
    show_node_numbers : bool, optional
        If True, display node numbers. Default is False.
    show_element_numbers : bool, optional
        If True, display element numbers. Default is False.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object.
    ax : matplotlib.axes.Axes
        Axes object.

    Notes
    -----
    Requires matplotlib to be installed.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "Matplotlib is required for mesh visualisation. "
            "Install with: pip install matplotlib"
        )

    nodes = mesh['nodes']
    elements = mesh['elements']

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot elements
    for elem_id, elem_nodes in enumerate(elements):
        # Get coordinates of element nodes (close the loop)
        elem_coords = nodes[elem_nodes]
        elem_x = np.append(elem_coords[:, 0], elem_coords[0, 0])
        elem_y = np.append(elem_coords[:, 1], elem_coords[0, 1])

        ax.plot(elem_x, elem_y, 'b-', linewidth=1)

        # Optionally show element numbers
        if show_element_numbers:
            centroid_x = elem_coords[:, 0].mean()
            centroid_y = elem_coords[:, 1].mean()
            ax.text(
                centroid_x, centroid_y, str(elem_id),
                ha='center', va='center',
                fontsize=8, color='red'
            )

    # Plot nodes
    ax.plot(nodes[:, 0], nodes[:, 1], 'ko', markersize=4)

    # Optionally show node numbers
    if show_node_numbers:
        for node_id, (x, y) in enumerate(nodes):
            ax.text(
                x, y, f' {node_id}',
                fontsize=7, color='blue',
                ha='left', va='bottom'
            )

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Structured Q4 Mesh')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    return fig, ax
