"""
Post-processing module for FEM analysis results.

Provides utilities for stress recovery, result visualisation,
and plotting configuration.
"""

from .plot_config import (
    setup_plot_style,
    reset_plot_style,
    get_color_palette
)

from .stress_recovery import (
    compute_stress_strain_field,
    compute_von_mises_stress,
    get_nodal_stress_component,
    get_nodal_strain_component
)

from .visualisation import (
    plot_contour_field,
    plot_displacement_field,
    plot_stress_field,
    plot_strain_field
)

__all__ = [
    # Plotting
    'setup_plot_style',
    'reset_plot_style',
    'get_color_palette',
    # Stress/strain recovery
    'compute_stress_strain_field',
    'compute_von_mises_stress',
    'get_nodal_stress_component',
    'get_nodal_strain_component',
    # Visualization
    'plot_contour_field',
    'plot_displacement_field',
    'plot_stress_field',
    'plot_strain_field',
]
