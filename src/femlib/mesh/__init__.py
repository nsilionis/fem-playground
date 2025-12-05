"""
Mesh generation and manipulation utilities.

Provides functions for creating structured meshes for rectangular
domains.
"""

from .structured import (
    create_rectangular_mesh,
    mesh_statistics,
    visualise_mesh
)

__all__ = [
    'create_rectangular_mesh',
    'mesh_statistics',
    'visualise_mesh',
]
