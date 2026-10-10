"""Beam element stiffness: closed form, exact quadrature, cantilever."""

import numpy as np
import pytest
from numpy.polynomial import Polynomial

from femlib.elements import Beam
from femlib.materials import LinearElastic1D
from femlib.sections import BeamSection

RTOL = 1e-12


def close(value, reference):
    """Agreement to RTOL times the largest absolute reference value."""
    value, reference = np.asarray(value), np.asarray(reference)
    error = np.max(np.abs(value - reference))
    return error <= RTOL * np.max(np.abs(reference))


def hermite_curvatures(L):
    """Second s-derivatives of the Hermite functions, s = x - x1."""
    xi = Polynomial([-1.0, 2.0 / L])
    N = [
        0.25 * (1 - xi) ** 2 * (2 + xi),
        L / 8 * (1 - xi) ** 2 * (1 + xi),
        0.25 * (1 + xi) ** 2 * (2 - xi),
        -L / 8 * (1 + xi) ** 2 * (1 - xi),
    ]
    return [n.deriv(2) for n in N]


def exact_stiffness(EI, L):
    """Element stiffness by exact polynomial integration over s in [0, L]."""
    B = hermite_curvatures(L)
    K = np.empty((4, 4))
    for i, Bi in enumerate(B):
        for j, Bj in enumerate(B):
            K[i, j] = (EI * Bi * Bj).integ()(L)
    return K


def test_constant_EI_matches_closed_form():
    L, E, I = 0.7, 210e9, 8.0e-6  # noqa: E741
    K = Beam([[0.0, 0.0], [L, 0.0]]).compute_stiffness_matrix(
        LinearElastic1D(E), BeamSection(I)
    )
    K_ref = E * I / L ** 3 * np.array([
        [12, 6 * L, -12, 6 * L],
        [6 * L, 4 * L ** 2, -6 * L, 2 * L ** 2],
        [-12, -6 * L, 12, -6 * L],
        [6 * L, 2 * L ** 2, -6 * L, 4 * L ** 2],
    ])
    assert close(K, K_ref)


def test_single_element_cantilever_tip():
    L, E, I, P = 2.0, 210e9, 8.0e-6, -1.0e3  # noqa: E741
    K = Beam([[0.0, 0.0], [L, 0.0]]).compute_stiffness_matrix(
        LinearElastic1D(E), BeamSection(I)
    )
    w2, theta2 = np.linalg.solve(K[2:, 2:], [P, 0.0])
    assert close(w2, P * L ** 3 / (3 * E * I))
    assert close(theta2, P * L ** 2 / (2 * E * I))


def test_quintic_EI_exact_with_full_order():
    x1 = 2.3
    beam = Beam([[x1, 0.0], [3.0, 0.0]])
    s = Polynomial([0.0, 1.0])
    EI = 1.0e6 * (2 + s + s ** 2 - s ** 3 + s ** 4 + s ** 5)
    material = LinearElastic1D(E=lambda x: EI(x - x1))
    section = BeamSection(I=1.0)
    K_ref = exact_stiffness(EI, beam.length)
    K_full = beam.compute_stiffness_matrix(material, section, 'full')
    K_three = beam.compute_stiffness_matrix(material, section, 3)
    assert close(K_full, K_ref)
    assert not close(K_three, K_ref)


@pytest.mark.parametrize("order", ["reduced", 1, 0, True])
def test_invalid_integration_order_rejected(order):
    with pytest.raises(ValueError):
        Beam([[0.0, 0.0], [1.0, 0.0]]).get_integration_points(order)


def test_nonpositive_EI_rejected():
    material = LinearElastic1D(E=lambda x: 1.0 - x)
    with pytest.raises(ValueError):
        Beam([[0.0, 0.0], [2.0, 0.0]]).compute_stiffness_matrix(
            material, BeamSection(I=1.0)
        )
