"""
Stress and strain recovery for post-processing.

This module provides utilities for computing stress and strain fields
from displacement solutions, including evaluation at integration points
and extrapolation to nodes for visualisation.
"""

import numpy as np
from ..elements import Q4Element


def compute_von_mises_stress(stress):
    """
    Compute von Mises equivalent stress from stress components.

    For plane stress/strain, von Mises stress is:
        σ_vm = √(σ_xx² - σ_xx·σ_yy + σ_yy² + 3·τ_xy²)

    Parameters
    ----------
    stress : ndarray, shape (..., 3)
        Stress components [σ_xx, σ_yy, τ_xy] in last dimension.
        Can be single point (..., 3) or multiple points (..., n, 3).

    Returns
    -------
    von_mises : ndarray, shape (...)
        Von Mises stress values.

    Examples
    --------
    >>> stress = np.array([100e6, 50e6, 25e6])  # Pa
    >>> vm = compute_von_mises_stress(stress)
    """
    stress = np.asarray(stress)

    # Extract components
    sigma_xx = stress[..., 0]
    sigma_yy = stress[..., 1]
    tau_xy = stress[..., 2]

    # Von Mises formula
    von_mises = np.sqrt(
        sigma_xx**2 - sigma_xx * sigma_yy + sigma_yy**2 + 3 * tau_xy**2
    )

    return von_mises


def compute_element_stress_strain(element, u_element, material,
                                  integration_order='full'):
    """
    Compute stress and strain at element integration points.

    Parameters
    ----------
    element : Q4Element
        Element object with nodal coordinates.
    u_element : ndarray, shape (8,)
        Element displacement vector [u1, v1, u2, v2, ..., u4, v4].
    material : Material
        Material object providing constitutive matrix.
    integration_order : str, optional
        Integration order: 'full' or 'reduced'. Default is 'full'.

    Returns
    -------
    results : dict
        Dictionary containing:
        - 'gauss_points': (n_points, 2) natural coordinates (ξ, η)
        - 'global_coords': (n_points, 2) global coordinates (x, y)
        - 'strain': (n_points, 3) strain [ε_xx, ε_yy, γ_xy]
        - 'stress': (n_points, 3) stress [σ_xx, σ_yy, τ_xy]
        - 'von_mises': (n_points,) von Mises stress

    Examples
    --------
    >>> elem = Q4Element(coords)
    >>> results = compute_element_stress_strain(elem, u_e, material)
    >>> stress_at_gauss_points = results['stress']
    """
    # Get integration points
    gauss_points, _ = element.get_integration_points(integration_order)
    n_points = gauss_points.shape[0]

    # Constitutive matrix
    D = material.constitutive_matrix()

    # Initialize arrays
    global_coords = np.zeros((n_points, 2))
    strain = np.zeros((n_points, 3))
    stress = np.zeros((n_points, 3))

    # Compute at each integration point
    for i, (xi, eta) in enumerate(gauss_points):
        # Evaluate shape functions for coordinate interpolation
        N = element.shape_functions(xi, eta)

        # Global coordinates at this point
        x = np.sum(N * element.node_coords[:, 0])
        y = np.sum(N * element.node_coords[:, 1])
        global_coords[i] = [x, y]

        # Compute B matrix
        B, _ = element.compute_B_matrix(xi, eta)

        # Strain: ε = B * u_e
        strain[i] = B @ u_element

        # Stress: σ = D * ε
        stress[i] = D @ strain[i]

    # Von Mises stress
    von_mises = compute_von_mises_stress(stress)

    results = {
        'gauss_points': gauss_points,
        'global_coords': global_coords,
        'strain': strain,
        'stress': stress,
        'von_mises': von_mises
    }

    return results


def compute_stress_strain_field(mesh, u, material,
                                integration_order='full'):
    """
    Compute stress and strain field over entire mesh.

    Evaluates stress/strain at integration points for all elements
    and extrapolates to nodes using simple averaging.

    Parameters
    ----------
    mesh : dict
        Mesh dictionary with 'nodes' and 'elements'.
    u : ndarray, shape (n_dofs,)
        Global displacement vector.
    material : Material
        Material object.
    integration_order : str, optional
        Integration order: 'full' or 'reduced'. Default is 'full'.

    Returns
    -------
    field_data : dict
        Dictionary containing:
        - 'gauss': dict with integration point data for each element
        - 'nodal': dict with extrapolated nodal values
            - 'stress': (n_nodes, 3)
            - 'strain': (n_nodes, 3)
            - 'von_mises': (n_nodes,)

    Examples
    --------
    >>> field = compute_stress_strain_field(mesh, u, material, 'full')
    >>> nodal_vm = field['nodal']['von_mises']
    >>> element_stresses = field['gauss'][0]['stress']

    Notes
    -----
    Nodal values are computed by averaging contributions from all
    elements sharing that node. This is simple but effective for
    visualisation purposes.
    """
    nodes = mesh['nodes']
    elements = mesh['elements']
    n_nodes = nodes.shape[0]
    # n_elements = elements.shape[0]

    # Storage for integration point data
    gauss_data = {}

    # Storage for nodal averaging
    nodal_stress = np.zeros((n_nodes, 3))
    nodal_strain = np.zeros((n_nodes, 3))
    nodal_von_mises = np.zeros(n_nodes)
    nodal_count = np.zeros(n_nodes)  # For averaging

    # Process each element
    for elem_id, elem_nodes in enumerate(elements):
        # Extract element nodal coordinates and displacements
        elem_coords = nodes[elem_nodes]

        # Extract element DOFs
        elem_dofs = np.zeros(8)
        for i, node in enumerate(elem_nodes):
            elem_dofs[2*i] = u[2*node]
            elem_dofs[2*i + 1] = u[2*node + 1]

        # Create element object
        element = Q4Element(elem_coords)

        # Compute stress/strain at integration points
        elem_results = compute_element_stress_strain(
            element,
            elem_dofs,
            material,
            integration_order
        )

        # Store integration point data
        gauss_data[elem_id] = elem_results

        # Extrapolate to element nodes (simple averaging)
        # Average all integration point values for this element
        elem_stress_avg = np.mean(elem_results['stress'], axis=0)
        elem_strain_avg = np.mean(elem_results['strain'], axis=0)
        elem_vm_avg = np.mean(elem_results['von_mises'])

        # Accumulate at nodes
        for node in elem_nodes:
            nodal_stress[node] += elem_stress_avg
            nodal_strain[node] += elem_strain_avg
            nodal_von_mises[node] += elem_vm_avg
            nodal_count[node] += 1

    # Average nodal values (divide by count)
    for i in range(n_nodes):
        if nodal_count[i] > 0:
            nodal_stress[i] /= nodal_count[i]
            nodal_strain[i] /= nodal_count[i]
            nodal_von_mises[i] /= nodal_count[i]

    # Assemble results
    field_data = {
        'gauss': gauss_data,
        'nodal': {
            'stress': nodal_stress,
            'strain': nodal_strain,
            'von_mises': nodal_von_mises
        }
    }

    return field_data


def get_nodal_stress_component(field_data, component):
    """
    Extract specific stress component from field data.

    Parameters
    ----------
    field_data : dict
        Output from compute_stress_strain_field().
    component : str
        Component name: 'sigma_xx', 'sigma_yy', 'tau_xy', 'von_mises'

    Returns
    -------
    values : ndarray, shape (n_nodes,)
        Nodal values of requested component.

    Examples
    --------
    >>> sigma_xx = get_nodal_stress_component(field_data, 'sigma_xx')
    """
    component_map = {
        'sigma_xx': 0,
        'sigma_yy': 1,
        'tau_xy': 2
    }

    if component == 'von_mises':
        return field_data['nodal']['von_mises']
    elif component in component_map:
        idx = component_map[component]
        return field_data['nodal']['stress'][:, idx]
    else:
        raise ValueError(
            f"Unknown component '{component}'. "
            f"Valid: 'sigma_xx', 'sigma_yy', 'tau_xy', 'von_mises'"
        )


def get_nodal_strain_component(field_data, component):
    """
    Extract specific strain component from field data.

    Parameters
    ----------
    field_data : dict
        Output from compute_stress_strain_field().
    component : str
        Component name: 'epsilon_xx', 'epsilon_yy', 'gamma_xy'

    Returns
    -------
    values : ndarray, shape (n_nodes,)
        Nodal values of requested component.

    Examples
    --------
    >>> eps_xx = get_nodal_strain_component(field_data, 'epsilon_xx')
    """
    component_map = {
        'epsilon_xx': 0,
        'epsilon_yy': 1,
        'gamma_xy': 2
    }

    if component in component_map:
        idx = component_map[component]
        return field_data['nodal']['strain'][:, idx]
    else:
        raise ValueError(
            f"Unknown component '{component}'. "
            f"Valid: 'epsilon_xx', 'epsilon_yy', 'gamma_xy'"
        )
