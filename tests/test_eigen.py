"""Free vibration: convergence, symmetry, sparse against dense, zero modes."""

import warnings

import numpy as np
import pytest
from scipy.linalg import LinAlgWarning
from scipy.optimize import brentq

from femlib.assembly import assemble_mass, assemble_stiffness
from femlib.elements import Beam
from femlib.materials import LinearElastic1D
from femlib.mesh import create_beam_mesh
from femlib.sections import BeamSection
from femlib.solvers import apply_dirichlet_bc, solve_eigen

L, E, I, A, RHO = 10.0, 210e9, 8.0e-6, 0.01, 7850.0  # noqa: E741
EPS = np.finfo(float).eps


def clamped_roots(n):
    """beta_k L for k = 1..n, the roots of cos(x) cosh(x) = 1."""
    def g(x):
        return np.cos(x) * np.cosh(x) - 1.0
    return np.array([brentq(g, (k + 0.5) * np.pi - 0.3,
                            (k + 0.5) * np.pi + 0.3)
                     for k in range(1, n + 1)])


def analytical(kind, n):
    """First n analytical frequencies [Hz], 'ss' or 'cc'."""
    if kind == "ss":
        beta_L = np.pi * np.arange(1, n + 1)
    else:
        beta_L = clamped_roots(n)
    return beta_L ** 2 / (2 * np.pi * L ** 2) * np.sqrt(E * I / (RHO * A))


def beam(n_elements, E_of_x=E, sparse=False):
    """Mesh, stiffness and mass of a uniform beam."""
    mesh = create_beam_mesh(L, n_elements)
    material = LinearElastic1D(E=E_of_x, rho=RHO)
    section = BeamSection(I=I, A=A)
    return (mesh,
            assemble_stiffness(mesh, material, section, sparse=sparse),
            assemble_mass(mesh, material, section, sparse=sparse))


def supports(mesh, K, kind):
    """Constrained DOFs for simple supports or clamps at both ends."""
    dof = "w" if kind == "ss" else "all"
    ends = (mesh["boundary_nodes"]["left"][0],
            mesh["boundary_nodes"]["right"][0])
    bc = {"constraints": [{"node": n, "dof": dof, "value": 0.0}
                          for n in ends]}
    _, _, c_dofs, _ = apply_dirichlet_bc(K, np.zeros(K.shape[0]), bc,
                                         dof_names=Beam.dof_names)
    return c_dofs


def frequencies(n_elements, kind, n_modes=4, **kwargs):
    mesh, K, M = beam(n_elements, **kwargs)
    f, _ = solve_eigen(K, M, supports(mesh, K, kind), n_modes=n_modes)
    return f


def test_clamped_roots_round_to_quoted_values():
    quoted = np.array([4.7300, 7.8532, 10.9956])
    assert np.all(np.abs(clamped_roots(3) - quoted) <= 0.5e-4)


@pytest.mark.parametrize("kind", ["ss", "cc"])
def test_frequencies_converge_from_above(kind):
    f = np.array([frequencies(n, kind) for n in [4, 8, 16, 32]])
    assert np.all(np.diff(f, axis=0) < 0)
    assert np.all(f > analytical(kind, 4))


@pytest.mark.parametrize("kind", ["ss", "cc"])
def test_frequencies_settle_on_analytical(kind):
    f = frequencies(100, kind)
    assert np.all(np.abs(f / analytical(kind, 4) - 1) <= 1e-6)


def bump(c, d=0.3, ell=0.05 * L):
    """E(x) with a smooth stiffness loss of depth d centred at c."""
    return lambda x: E * (1 - d * np.exp(-(x - c) ** 2 / (2 * ell ** 2)))


def all_eigenvalues(E_of_x):
    mesh, K, M = beam(40, E_of_x=E_of_x)
    f, _ = solve_eigen(K, M, supports(mesh, K, "ss"))
    return (2 * np.pi * f) ** 2


def test_mirrored_damage_gives_same_frequencies():
    lam_a = all_eigenvalues(bump(0.3 * L))
    lam_b = all_eigenvalues(bump(0.7 * L))
    lam_c = all_eigenvalues(bump(0.31 * L))
    bound = 100 * EPS * lam_a.max()
    assert np.all(np.abs(lam_a[:5] - lam_b[:5]) <= bound)
    assert np.all(np.abs(lam_c[:5] - lam_b[:5]) > bound)


def test_sparse_matches_dense():
    mesh, K, M = beam(100)
    _, K_s, M_s = beam(100, sparse=True)
    c_dofs = supports(mesh, K, "ss")
    f_d, modes_d = solve_eigen(K, M, c_dofs)
    f_s, modes_s = solve_eigen(K_s, M_s, c_dofs, n_modes=5,
                               dof_names=Beam.dof_names)
    lam_d, lam_s = (2 * np.pi * f_d) ** 2, (2 * np.pi * f_s) ** 2
    bound = 100 * EPS * lam_d.max()
    assert np.all(np.abs(lam_s - lam_d[:5]) <= bound)
    gaps = np.diff(lam_d[:6])
    for i in range(5):
        gap = min(gaps[i], gaps[i - 1]) if i > 0 else gaps[0]
        scale = np.max(np.abs(modes_d[:, i]))
        error = np.max(np.abs(modes_s[:, i] - modes_d[:, i]))
        assert error <= bound / gap * scale


def test_free_free_rigid_modes_normalisation_and_signs():
    _, K, M = beam(20)
    f, modes = solve_eigen(K, M, n_modes=6)
    assert np.all(f[:2] <= 1e-3 * f[2])
    G = modes.T @ M @ modes
    assert np.max(np.abs(G - np.eye(6))) <= 100 * EPS * np.linalg.cond(M)
    for column in modes.T:
        significant = np.abs(column) > 1e-6 * np.max(np.abs(column))
        assert column[np.flatnonzero(significant)[0]] > 0


def test_sparse_needs_n_modes():
    _, K, M = beam(10, sparse=True)
    with pytest.raises(ValueError, match="n_modes"):
        solve_eigen(K, M)


def test_sparse_free_free_refused():
    _, K, M = beam(10, sparse=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error", LinAlgWarning)
        with pytest.raises((np.linalg.LinAlgError, LinAlgWarning)):
            solve_eigen(K, M, n_modes=3, dof_names=Beam.dof_names)
