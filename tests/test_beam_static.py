"""Beam mesh, and the static cantilever against PL^3 / 3EI."""

import numpy as np
import pytest

from femlib.assembly import assemble_load_vector, assemble_stiffness
from femlib.elements import Beam
from femlib.materials import LinearElastic1D
from femlib.mesh import create_beam_mesh
from femlib.sections import BeamSection
from femlib.solvers import (
    apply_dirichlet_bc,
    identify_boundary_nodes,
    solve_system,
)

RTOL = 1e-12
EPS = np.finfo(float).eps


def test_beam_mesh_shapes_and_symmetry():
    L, n = 3.0, 7
    mesh = create_beam_mesh(L, n)
    x = mesh["nodes"][:, 0]
    assert mesh["nodes"].shape == (n + 1, 1)
    assert mesh["elements"].shape == (n, 2)
    assert mesh["element_type"] == "Beam"
    assert np.max(np.abs((L - x) - x[::-1])) <= RTOL * L


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("n", [1, 4, 25])
def test_cantilever_tip_deflection(n, sparse):
    L, E, I, P = 2.0, 210e9, 8.0e-6, -1.0e3  # noqa: E741
    mesh = create_beam_mesh(L, n)
    K = assemble_stiffness(mesh, LinearElastic1D(E), BeamSection(I),
                           sparse=sparse)
    F = assemble_load_vector(
        mesh, {"concentrated": [{"location": (L,), "force": [P, 0.0]}]},
        dof_names=Beam.dof_names,
    )
    bc = {"nodes": identify_boundary_nodes(mesh, "left"),
          "dofs": ["all"], "values": [0.0]}
    K_bc, F_bc, c_dofs, c_vals = apply_dirichlet_bc(
        K, F, bc, dof_names=Beam.dof_names
    )
    u = solve_system(K_bc, F_bc, c_dofs, c_vals, dof_names=Beam.dof_names)

    exact = P * L ** 3 / (3 * E * I)
    K_dense = K_bc.toarray() if sparse else K_bc
    bound = 10 * np.linalg.cond(K_dense) * EPS * abs(exact)
    assert abs(u[2 * n] - exact) <= bound
