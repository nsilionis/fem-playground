"""
Mesh generation and manipulation utilities.

Provides functions for creating structured meshes for rectangular
domains and beams.
"""

from .structured import (
    create_beam_mesh,
    create_rectangular_mesh,
    mesh_statistics,
    visualise_mesh
)

__all__ = [
    'create_beam_mesh',
    'create_rectangular_mesh',
    'mesh_statistics',
    'visualise_mesh',
]
