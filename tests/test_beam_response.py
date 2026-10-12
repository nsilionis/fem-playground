"""Beam response recovery: exact cantilever, strain sign, first mode, nodes."""

import numpy as np
import pytest

from femlib.assembly import (
    assemble_load_vector,
    assemble_mass,
    assemble_stiffness,
)
from femlib.elements import Beam
from femlib.materials import LinearElastic1D
from femlib.mesh import create_beam_mesh
from femlib.postprocess import (
    beam_curvature,
    beam_displacement,
    surface_strain,
)
from femlib.sections import BeamSection
from femlib.solvers import (
    apply_dirichlet_bc,
    identify_boundary_nodes,
    solve_eigen,
    solve_system,
)

L, E, I, A, RHO = 2.0, 210e9, 8.0e-6, 0.01, 7850.0  # noqa: E741
EPS = np.finfo(float).eps


def cantilever(n_elements=5, P=-1.0e3):
    """Cantilever under a tip load: mesh, displacements, cond(K), load."""
    mesh = create_beam_mesh(L, n_elements)
    K = assemble_stiffness(mesh, LinearElastic1D(E), BeamSection(I))
    F = assemble_load_vector(
        mesh, {"concentrated": [{"location": (L,), "force": [P, 0.0]}]},
        dof_names=Beam.dof_names,
    )
    bc = {"nodes": identify_boundary_nodes(mesh, "left"),
          "dofs": ["all"], "values": [0.0]}
    K_bc, F_bc, c_dofs, c_vals = apply_dirichlet_bc(
        K, F, bc, dof_names=Beam.dof_names
    )
    u = solve_system(K_bc, F_bc, c_dofs, c_vals,
                     dof_names=Beam.dof_names)
    return mesh, u, np.linalg.cond(K_bc), P


def test_cantilever_recovered_exactly():
    mesh, u, cond, P = cantilever()
    rng = np.random.default_rng(0)
    x = np.sort(np.concatenate([rng.uniform(0.0, L, 50),
                                mesh["nodes"][:, 0]]))
    w_exact = P * x ** 2 * (3 * L - x) / (6 * E * I)
    kappa_exact = P * (L - x) / (E * I)
    bound = 10 * cond * EPS
    w = beam_displacement(mesh, u, x)
    kappa = beam_curvature(mesh, u, x)
    assert np.max(np.abs(w - w_exact)) <= bound * np.max(np.abs(w_exact))
    assert (np.max(np.abs(kappa - kappa_exact))
            <= bound * np.max(np.abs(kappa_exact)))


def test_downward_tip_load_puts_top_surface_in_tension():
    mesh, u, _, P = cantilever()
    assert P < 0
    x = np.linspace(0.0, 0.9 * L, 40)
    assert np.all(surface_strain(mesh, u, x, z=0.05) > 0)


def test_first_simply_supported_mode():
    mesh = create_beam_mesh(L, 100)
    material = LinearElastic1D(E=E, rho=RHO)
    section = BeamSection(I=I, A=A)
    K = assemble_stiffness(mesh, material, section)
    M = assemble_mass(mesh, material, section)
    bc = {"constraints": [{"node": 0, "dof": "w", "value": 0.0},
                          {"node": 100, "dof": "w", "value": 0.0}]}
    _, _, c_dofs, _ = apply_dirichlet_bc(K, np.zeros(K.shape[0]), bc,
                                         dof_names=Beam.dof_names)
    _, modes = solve_eigen(K, M, c_dofs, n_modes=1,
                           dof_names=Beam.dof_names)

    x = np.linspace(0.0, L, 201)
    phi = np.sqrt(2 / (RHO * A * L)) * np.sin(np.pi * x / L)
    kappa = -(np.pi / L) ** 2 * phi
    w_h = beam_displacement(mesh, modes[:, 0], x)
    kappa_h = beam_curvature(mesh, modes[:, 0], x)
    assert np.max(np.abs(w_h - phi)) <= 1e-6 * np.max(np.abs(phi))
    assert np.max(np.abs(kappa_h - kappa)) <= 1e-3 * np.max(np.abs(kappa))


def test_curvature_at_node_is_mean_of_both_elements():
    mesh = create_beam_mesh(L, 2)
    u = np.random.default_rng(1).normal(size=6)
    left = Beam(mesh["nodes"][[0, 1]])
    right = Beam(mesh["nodes"][[1, 2]])
    kappa_left = left.compute_B_matrix(1.0)[0][0] @ u[0:4]
    kappa_right = right.compute_B_matrix(-1.0)[0][0] @ u[2:6]
    kappa = beam_curvature(mesh, u, L / 2)
    assert np.isclose(kappa[0], 0.5 * (kappa_left + kappa_right),
                      rtol=1e-14, atol=0.0)


@pytest.mark.parametrize("x", [-1e-9, L + 1e-9])
def test_position_outside_beam_rejected(x):
    mesh = create_beam_mesh(L, 2)
    with pytest.raises(ValueError, match="within"):
        beam_displacement(mesh, np.zeros(6), x)


def test_mode_matrix_gives_one_column_per_mode():
    mesh = create_beam_mesh(L, 4)
    U = np.random.default_rng(2).normal(size=(10, 3))
    x = np.linspace(0.0, L, 17)
    together = beam_curvature(mesh, U, x)
    assert together.shape == (17, 3)
    scale = np.max(np.abs(together))
    for j in range(3):
        alone = beam_curvature(mesh, U[:, j], x)
        assert np.allclose(together[:, j], alone,
                           rtol=0.0, atol=1e-14 * scale)
