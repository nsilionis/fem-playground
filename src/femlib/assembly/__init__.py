"""
Assembly module for global FEM matrices and vectors.

Provides functions for assembling global stiffness matrices and
load vectors from element contributions with geometric load
specification.
"""

from .assembler import (
    assemble_stiffness,
    get_element_dofs
)
from .loads import (
    assemble_load_vector,
    identify_nodes_by_geometry,
    find_nearest_node
)

__all__ = [
    'assemble_stiffness',
    'get_element_dofs',
    'assemble_load_vector',
    'identify_nodes_by_geometry',
    'find_nearest_node',
]
