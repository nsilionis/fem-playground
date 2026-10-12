"""Dirichlet conditions: DOF names, 'all', sparse against dense."""

import numpy as np
import pytest

from femlib.assembly import assemble_stiffness
from femlib.elements import Beam
from femlib.materials import LinearElastic1D, LinearElastic2D
from femlib.mesh import create_beam_mesh, create_rectangular_mesh
from femlib.sections import BeamSection, PlaneSection
from femlib.solvers import apply_dirichlet_bc

RTOL = 1e-12


def close(value, reference):
    """Agreement to RTOL times the largest absolute reference value."""
    value, reference = np.asarray(value), np.asarray(reference)
    error = np.max(np.abs(value - reference))
    return error <= RTOL * np.max(np.abs(reference))


STEEL = LinearElastic1D(E=210e9, rho=7850.0)
SECTION = BeamSection(I=8.0e-6, A=0.01)


def test_beam_names_map_to_global_rows():
    mesh = create_beam_mesh(2.0, 2)  # 3 nodes, 6 DOFs
    K = assemble_stiffness(mesh, STEEL, SECTION)
    bc = {"constraints": [{"node": 0, "dof": "all", "value": 0.0},
                          {"node": 2, "dof": "w", "value": 0.0}]}
    _, _, c_dofs, _ = apply_dirichlet_bc(K, np.zeros(6), bc,
                                         dof_names=Beam.dof_names)
    assert list(c_dofs) == [0, 1, 4]


def test_unknown_dof_name_lists_valid_names():
    bc = {"constraints": [{"node": 0, "dof": "rotation", "value": 0.0}]}
    with pytest.raises(ValueError, match="'w', 'theta', 'all'"):
        apply_dirichlet_bc(np.eye(6), np.zeros(6), bc,
                           dof_names=Beam.dof_names)


@pytest.mark.parametrize("method", ["elimination", "penalty"])
def test_sparse_matches_dense(method):
    mesh = create_rectangular_mesh(4.0, 1.0, 8, 2)
    material, section = LinearElastic2D(E=210e9, nu=0.3), PlaneSection(0.1)
    K = assemble_stiffness(mesh, material, section)
    K_sp = assemble_stiffness(mesh, material, section, sparse=True)
    F = 1e3 * np.linspace(-1.0, 1.0, K.shape[0])
    left = mesh["boundary_nodes"]["left"]
    bc = {"constraints": [{"node": left[0], "dof": "all", "value": 0.0},
                          {"node": left[1], "dof": "all", "value": 0.0},
                          {"node": left[2], "dof": "ux", "value": 1e-3}]}
    Kd, Fd, cd, _ = apply_dirichlet_bc(K, F, bc, method=method)
    Ks, Fs, cs, _ = apply_dirichlet_bc(K_sp, F, bc, method=method)
    assert np.array_equal(cd, cs)
    assert close(Ks.toarray(), Kd)
    assert close(Fs, Fd)
