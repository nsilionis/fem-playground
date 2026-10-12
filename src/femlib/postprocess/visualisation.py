"""
Visualisation utilities for FEM results.

This module provides contour plotting functions for displacement,
stress, and strain fields using matplotlib's triangulation-based
contour plotting, and mode-shape plots for beam meshes.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

from .beam_response import beam_displacement


def plot_contour_field(mesh, field_data, field_name, **kwargs):
    """
    Plot contour map of a scalar field over the mesh.

    Universal plotting function for any scalar field (displacement,
    stress, strain). Uses triangulation-based contouring for clean
    visualisation of quadrilateral mesh data.

    Parameters
    ----------
    mesh : dict
        Mesh dictionary containing 'nodes' and 'elements'.
    field_data : ndarray, shape (n_nodes,)
        Scalar field values at each node.
    field_name : str
        Field identifier for automatic labelling and colormap selection.
        Recognized names:
        - Displacement: 'u', 'v', 'displacement_magnitude'
        - Stress: 'sigma_xx', 'sigma_yy', 'tau_xy', 'von_mises'
        - Strain: 'epsilon_xx', 'epsilon_yy', 'gamma_xy'

    Keyword Arguments
    -----------------
    cmap : str, optional
        Matplotlib colormap name. If not provided, selected
        automatically based on field_name.
    levels : int, optional
        Number of contour levels. Default is 15.
    show_mesh : bool, optional
        If True, overlay mesh edges. Default is False.
    show_colorbar : bool, optional
        If True, display colorbar. Default is True.
    title : str, optional
        Plot title. If not provided, generated from field_name.
    figsize : tuple, optional
        Figure size (width, height). Default is (10, 6).
    vmin : float, optional
        Minimum value for colormap scaling.
    vmax : float, optional
        Maximum value for colormap scaling.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object.
    ax : matplotlib.axes.Axes
        Axes object.
    cbar : matplotlib.colorbar.Colorbar or None
        Colorbar object if show_colorbar=True, else None.

    Examples
    --------
    >>> # Plot von Mises stress
    >>> fig, ax, cbar = plot_contour_field(
    ...     mesh,
    ...     field_data['nodal']['von_mises'],
    ...     'von_mises',
    ...     levels=20,
    ...     show_mesh=True
    ... )

    >>> # Plot vertical displacement
    >>> v_nodal = u[1::2]
    >>> fig, ax, cbar = plot_contour_field(
    ...     mesh,
    ...     v_nodal,
    ...     'v'
    ... )

    Notes
    -----
    Q4 elements are automatically subdivided into triangles for
    matplotlib's tricontourf function, ensuring smooth contours.
    """
    # Extract mesh data
    nodes = mesh['nodes']
    elements = mesh['elements']

    # Default settings by field type
    field_settings = {
        # Displacement fields
        'u': {
            'cmap': 'viridis',
            'label': r'$u$ [mm]',
            'title': 'Horizontal Displacement'
        },
        'v': {
            'cmap': 'viridis',
            'label': r'$v$ [mm]',
            'title': 'Vertical Displacement'
        },
        'displacement_magnitude': {
            'cmap': 'plasma',
            'label': r'$|\mathbf{u}|$ [mm]',
            'title': 'Displacement Magnitude'
        },
        # Stress fields
        'sigma_xx': {
            'cmap': 'RdBu_r',
            'label': r'$\sigma_{xx}$ [MPa]',
            'title': 'Normal Stress (x-direction)'
        },
        'sigma_yy': {
            'cmap': 'RdBu_r',
            'label': r'$\sigma_{yy}$ [MPa]',
            'title': 'Normal Stress (y-direction)'
        },
        'tau_xy': {
            'cmap': 'PuOr_r',
            'label': r'$\tau_{xy}$ [MPa]',
            'title': 'Shear Stress'
        },
        'von_mises': {
            'cmap': 'hot',
            'label': r'$\sigma_{vm}$ [MPa]',
            'title': 'von Mises Stress'
        },
        # Strain fields
        'epsilon_xx': {
            'cmap': 'RdBu_r',
            'label': r'$\varepsilon_{xx}$',
            'title': 'Normal Strain (x-direction)'
        },
        'epsilon_yy': {
            'cmap': 'RdBu_r',
            'label': r'$\varepsilon_{yy}$',
            'title': 'Normal Strain (y-direction)'
        },
        'gamma_xy': {
            'cmap': 'PuOr_r',
            'label': r'$\gamma_{xy}$',
            'title': 'Shear Strain'
        }
    }

    # Get settings for this field
    if field_name in field_settings:
        settings = field_settings[field_name]
    else:
        # Default settings for unknown field
        settings = {
            'cmap': 'viridis',
            'label': field_name,
            'title': f'Field: {field_name}'
        }

    # Override with user-provided kwargs
    cmap = kwargs.get('cmap', settings['cmap'])
    levels = kwargs.get('levels', 15)
    show_mesh = kwargs.get('show_mesh', False)
    show_colorbar = kwargs.get('show_colorbar', True)
    title = kwargs.get('title', settings['title'])
    figsize = kwargs.get('figsize', (10, 4))
    vmin = kwargs.get('vmin', None)
    vmax = kwargs.get('vmax', None)

    # Create triangulation from Q4 mesh
    # Each Q4 element is split into 2 triangles
    triangles = []
    for elem_nodes in elements:
        # Triangle 1: nodes 0-1-2
        triangles.append([elem_nodes[0], elem_nodes[1], elem_nodes[2]])
        # Triangle 2: nodes 0-2-3
        triangles.append([elem_nodes[0], elem_nodes[2], elem_nodes[3]])

    triangles = np.array(triangles)

    # Create matplotlib Triangulation object
    triang = Triangulation(nodes[:, 0], nodes[:, 1], triangles)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Prepare data for plotting and apply unit conversions
    field_array = np.asarray(field_data)

    # Displacements -> plot in millimetres (m -> mm)
    if field_name in ('u', 'v', 'displacement_magnitude'):
        plot_data = field_array * 1e3
    # Stresses -> plot in MPa (Pa -> MPa)
    elif field_name in ('sigma_xx', 'sigma_yy', 'tau_xy', 'von_mises'):
        plot_data = field_array / 1e6
    else:
        plot_data = field_array

    # Plot filled contours
    contour_plot = ax.tricontourf(
        triang,
        plot_data,
        levels=levels,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    # Optionally show mesh edges
    if show_mesh:
        for elem_nodes in elements:
            # Create closed polygon for element
            elem_coords = nodes[elem_nodes]
            x_elem = np.append(elem_coords[:, 0], elem_coords[0, 0])
            y_elem = np.append(elem_coords[:, 1], elem_coords[0, 1])
            ax.plot(x_elem, y_elem, 'k-', linewidth=0.5, alpha=0.5)

    # Add colorbar
    cbar = None
    if show_colorbar:
        cbar = plt.colorbar(
            contour_plot,
            ax=ax,
            label=settings['label'],
            fraction=0.046,  # Controls colorbar width relative to axes
            pad=0.04,  # Space between axes and colorbar
        )

    # Labels and title
    ax.set_xlabel(r'$x$ [m]')
    ax.set_ylabel(r'$y$ [m]')
    ax.set_title(title)
    # ax.set_aspect('equal')

    plt.tight_layout()

    return fig, ax, cbar


def plot_displacement_field(mesh, u, component='magnitude', **kwargs):
    """
    Plot displacement field contours.

    Convenience wrapper for plot_contour_field specifically for
    displacement visualization.

    Parameters
    ----------
    mesh : dict
        Mesh dictionary.
    u : ndarray, shape (n_dofs,)
        Global displacement vector.
    component : str, optional
        Component to plot: 'u', 'v', or 'magnitude'. Default is
        'magnitude'.
    **kwargs
        Additional arguments passed to plot_contour_field.

    Returns
    -------
    fig, ax, cbar
        Figure, axes, and colorbar objects.

    Examples
    --------
    >>> fig, ax, cbar = plot_displacement_field(mesh, u, 'v')
    """
    # n_nodes = len(u) // 2

    if component == 'u':
        field_data = u[0::2]
        field_name = 'u'
    elif component == 'v':
        field_data = u[1::2]
        field_name = 'v'
    elif component == 'magnitude':
        u_x = u[0::2]
        u_y = u[1::2]
        field_data = np.sqrt(u_x**2 + u_y**2)
        field_name = 'displacement_magnitude'
    else:
        raise ValueError(
            f"Unknown component '{component}'. "
            f"Must be 'u', 'v', or 'magnitude'."
        )

    return plot_contour_field(mesh, field_data, field_name, **kwargs)


def plot_stress_field(mesh, field_data, component='von_mises',
                      **kwargs):
    """
    Plot stress field contours.

    Convenience wrapper for plot_contour_field specifically for
    stress visualization.

    Parameters
    ----------
    mesh : dict
        Mesh dictionary.
    field_data : dict
        Output from compute_stress_strain_field().
    component : str, optional
        Stress component: 'von_mises', 'sigma_xx', 'sigma_yy',
        'tau_xy'. Default is 'von_mises'.
    **kwargs
        Additional arguments passed to plot_contour_field.

    Returns
    -------
    fig, ax, cbar
        Figure, axes, and colorbar objects.

    Examples
    --------
    >>> fig, ax, cbar = plot_stress_field(
    ...     mesh, field_data, 'von_mises', show_mesh=True
    ... )
    """
    from .stress_recovery import get_nodal_stress_component

    stress_values = get_nodal_stress_component(field_data, component)

    return plot_contour_field(mesh, stress_values, component, **kwargs)


def plot_strain_field(mesh, field_data, component='epsilon_xx',
                      **kwargs):
    """
    Plot strain field contours.

    Convenience wrapper for plot_contour_field specifically for
    strain visualization.

    Parameters
    ----------
    mesh : dict
        Mesh dictionary.
    field_data : dict
        Output from compute_stress_strain_field().
    component : str, optional
        Strain component: 'epsilon_xx', 'epsilon_yy', 'gamma_xy'.
        Default is 'epsilon_xx'.
    **kwargs
        Additional arguments passed to plot_contour_field.

    Returns
    -------
    fig, ax, cbar
        Figure, axes, and colorbar objects.

    Examples
    --------
    >>> fig, ax, cbar = plot_strain_field(
    ...     mesh, field_data, 'epsilon_xx'
    ... )
    """
    from .stress_recovery import get_nodal_strain_component

    strain_values = get_nodal_strain_component(field_data, component)

    return plot_contour_field(mesh, strain_values, component, **kwargs)


def plot_beam_modes(mesh, modes, n_modes=4, n_points=401):
    """
    Plot beam mode shapes, one panel per mode, two panels per row.

    Each mode is drawn from its Hermite interpolation at n_points
    positions, scaled to unit maximum magnitude, with the nodal values
    marked.

    Parameters
    ----------
    mesh : dict
        Beam mesh, as from create_beam_mesh.
    modes : numpy.ndarray, shape (n_dofs, n_available)
        Mode shapes, one per column, as from solve_eigen.
    n_modes : int, optional
        Number of modes to plot. Default is 4.
    n_points : int, optional
        Number of positions for each curve. Default is 401.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : numpy.ndarray of matplotlib.axes.Axes
    """
    x_nodes = mesh['nodes'][:, 0]
    x = np.linspace(x_nodes[0], x_nodes[-1], n_points)
    w = beam_displacement(mesh, modes[:, :n_modes], x)
    w_nodes = beam_displacement(mesh, modes[:, :n_modes], x_nodes)

    n_rows = int(np.ceil(n_modes / 2))
    fig, axes = plt.subplots(n_rows, 2, figsize=(6.4, 2.4 * n_rows),
                             sharex=True, squeeze=False)
    for i, ax in enumerate(axes.flat):
        if i >= n_modes:
            ax.set_visible(False)
            continue
        scale = np.max(np.abs(w[:, i]))
        ax.axhline(0.0, color='#B0B0B0', linewidth=0.6)
        ax.plot(x, w[:, i] / scale, color='#4878D0', label='Mode shape')
        ax.plot(x_nodes, w_nodes[:, i] / scale, 'o', color='#4878D0',
                markerfacecolor='#D9D9D9', markersize=4.5, zorder=3,
                label='Nodes')
        ax.set_ylabel(rf'$\phi_{i + 1} / \max|\phi_{i + 1}|$')
    for ax in axes[-1]:
        ax.set_xlabel(r'$x\ [\mathrm{m}]$')
    axes[0, 0].legend()
    fig.tight_layout()
    return fig, axes
