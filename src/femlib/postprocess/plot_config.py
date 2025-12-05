"""
Matplotlib configuration for consistent plotting style.

This module provides utilities for setting up matplotlib with
custom styling preferences, ensuring consistent figure aesthetics
across all visualisations.
"""

import matplotlib.pyplot as plt
from matplotlib import rcParams


def setup_plot_style():
    """
    Configure matplotlib with preferred styling.

    Applies custom font settings using Libertinus fonts with LaTeX
    rendering and siunitx for proper unit formatting.

    Call this function at the start of any script that creates plots
    to ensure consistent styling across all figures.

    Examples
    --------
    >>> from femlib.postprocess.plot_config import setup_plot_style
    >>> setup_plot_style()
    >>> fig, ax = plt.subplots()
    >>> # All plots now use custom styling
    """
    rcParams.update({
        # LaTeX rendering
        "text.usetex": True,
        "font.family": "serif",
        "axes.unicode_minus": False,

        # LaTeX preamble with Libertinus fonts and siunitx
        "text.latex.preamble": r"""
            \usepackage{libertinus}
            \usepackage{libertinust1math}
            \usepackage{siunitx}
            \sisetup{
                detect-all,
                per-mode=symbol,
                retain-unity-mantissa = false
            }
        """,

        # Figure settings
        "figure.figsize": (10, 6),
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "savefig.format": "pdf",
        "savefig.bbox": "tight",

        # Axes settings
        "axes.linewidth": 1.0,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "axes.grid": True,

        # Grid settings
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,

        # Line settings
        "lines.linewidth": 1.5,
        "lines.markersize": 6,

        # Legend settings
        "legend.frameon": True,
        "legend.fontsize": 10,
        "legend.framealpha": 0.9,

        # Tick settings
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })


def reset_plot_style():
    """
    Reset matplotlib to default settings.

    Useful for restoring original matplotlib configuration after
    applying custom styling.

    Examples
    --------
    >>> from femlib.postprocess.plot_config import (
    ...     setup_plot_style, reset_plot_style
    ... )
    >>> setup_plot_style()
    >>> # ... create plots ...
    >>> reset_plot_style()  # Restore defaults
    """
    plt.rcdefaults()


def get_color_palette(name='default'):
    """
    Return a colour palette for consistent plot styling.

    Parameters
    ----------
    name : str, optional
        Palette name. Options: 'default', 'stress', 'displacement'.
        Default is 'default'.

    Returns
    -------
    colors : dict
        Dictionary mapping colour names to hex codes.

    Examples
    --------
    >>> colors = get_color_palette('default')
    >>> plt.plot(x, y, color=colors['primary'])
    """
    palettes = {
        'default': {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'accent': '#2ca02c',
            'neutral': '#7f7f7f',
            'highlight': '#d62728',
        },
        'stress': {
            'tension': '#d62728',
            'compression': '#1f77b4',
            'shear': '#ff7f0e',
            'vonmises': '#9467bd',
        },
        'displacement': {
            'horizontal': '#1f77b4',
            'vertical': '#2ca02c',
            'magnitude': '#9467bd',
        }
    }

    if name not in palettes:
        raise ValueError(
            f"Unknown palette '{name}'. "
            f"Available: {list(palettes.keys())}"
        )

    return palettes[name]
