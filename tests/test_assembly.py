"""Global assembly: beam mass, sparse against dense, per-element sections."""

import numpy as np
import pytest
import scipy.sparse as sp

from femlib.assembly import assemble_mass, assemble_stiffness
from femlib.materials import LinearElastic1D, LinearElastic2D
from femlib.mesh import create_beam_mesh, create_rectangular_mesh
from femlib.sections import BeamSection, PlaneSection, TrussSection

RTOL = 1e-12


def close(value, reference):
    """Agreement to RTOL times the largest absolute reference value."""
    value, reference = np.asarray(value), np.asarray(reference)
    error = np.max(np.abs(value - reference))
    return error <= RTOL * np.max(np.abs(reference))


STEEL = LinearElastic1D(E=210e9, rho=7850.0)
SECTION = BeamSection(I=8.0e-6, A=0.01)


def test_beam_mass_rigid_translation_total_mass():
    L = 2.0
    M = assemble_mass(create_beam_mesh(L, 5), STEEL, SECTION)
    u = np.zeros(M.shape[0])
    u[0::2] = 1.0  # every w = 1, every theta = 0
    assert close(u @ M @ u, 7850.0 * 0.01 * L)


def test_mass_assembly_rejects_q4():
    mesh = create_rectangular_mesh(1.0, 1.0, 2, 2)
    with pytest.raises(ValueError, match="Q4"):
        assemble_mass(mesh, LinearElastic2D(E=1e9, nu=0.3),
                      PlaneSection(0.1))


def _q4():
    return (create_rectangular_mesh(4.0, 1.0, 4, 2),
            LinearElastic2D(E=210e9, nu=0.3), PlaneSection(0.01))


def _truss():
    mesh = {"nodes": np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]]),
            "elements": np.array([[0, 1], [1, 2], [2, 0]]),
            "element_type": "Truss"}
    return mesh, LinearElastic1D(E=200e9), TrussSection(A=1e-3)


def _beam():
    return create_beam_mesh(2.0, 5), STEEL, SECTION


@pytest.mark.parametrize("build, assemble", [
    (_q4, assemble_stiffness),
    (_truss, assemble_stiffness),
    (_beam, assemble_stiffness),
    (_beam, assemble_mass),
])
def test_sparse_matches_dense(build, assemble):
    mesh, material, section = build()
    dense = assemble(mesh, material, section)
    sparse = assemble(mesh, material, section, sparse=True)
    assert isinstance(sparse, sp.csr_array)
    assert close(sparse.toarray(), dense)


def test_section_list_of_one_repeated_section_matches_single():
    mesh, material, section = _truss()
    single = assemble_stiffness(mesh, material, section)
    listed = assemble_stiffness(mesh, material, [section] * 3)
    assert close(listed, single)


@pytest.mark.parametrize("sparse", [False, True])
def test_per_element_sections_sum_of_single_elements(sparse):
    mesh, material, _ = _truss()
    sections = [TrussSection(A=a) for a in (1e-3, 2e-3, 5e-4)]
    K = assemble_stiffness(mesh, material, sections, sparse=sparse)
    expected = sum(
        assemble_stiffness({**mesh, "elements": mesh["elements"][[e]]},
                           material, sections[e])
        for e in range(3)
    )
    assert close(K.toarray() if sparse else K, expected)


def test_per_element_sections_beam_mass():
    L = 2.0
    mesh = create_beam_mesh(L, 4)
    areas = [0.01, 0.02, 0.03, 0.04]
    sections = [BeamSection(I=8.0e-6, A=a) for a in areas]
    M = assemble_mass(mesh, STEEL, sections)
    u = np.zeros(M.shape[0])
    u[0::2] = 1.0
    assert close(u @ M @ u, 7850.0 * sum(areas) * L / 4)


def test_section_list_of_wrong_length_rejected():
    mesh, material, section = _truss()
    with pytest.raises(ValueError, match="one section per element"):
        assemble_stiffness(mesh, material, [section] * 2)
