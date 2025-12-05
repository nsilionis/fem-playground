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

__all__ = [
    'setup_plot_style',
    'reset_plot_style',
    'get_color_palette',
]
