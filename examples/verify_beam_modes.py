"""
Mode shapes and convergence of modal curvature under mesh refinement.

Simply supported and clamped-clamped Euler-Bernoulli beams. For each,
writes the first four mode shapes on a 16-element mesh, and the
relative energy-norm error of the modal curvature against a
512-element reference, to outputs/figures; --show also displays them.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from femlib.assembly import assemble_mass, assemble_stiffness
from femlib.elements import Beam
from femlib.materials import LinearElastic1D
from femlib.mesh import create_beam_mesh
from femlib.postprocess import (
    beam_curvature,
    plot_beam_modes,
    setup_plot_style,
)
from femlib.sections import BeamSection
from femlib.solvers import apply_dirichlet_bc, solve_eigen

L, E, I, A, RHO = 10.0, 210e9, 8.0e-6, 0.01, 7850.0  # noqa: E741
N_MODES = 4
ELEMENTS = [4, 8, 16, 32, 64, 128]
REFERENCE_ELEMENTS = 512
MODE_PLOT_ELEMENTS = 16
COLOUR = "#4878D0"
OUTPUT_DIR = Path("outputs/figures")
NAMES = {"ss": "simply_supported", "cc": "clamped"}


def modes(n_elements, kind):
    """Mesh and first N_MODES mass-normalised modes, 'ss' or 'cc'."""
    mesh = create_beam_mesh(L, n_elements)
    material = LinearElastic1D(E=E, rho=RHO)
    section = BeamSection(I=I, A=A)
    K = assemble_stiffness(mesh, material, section)
    M = assemble_mass(mesh, material, section)

    dof = "w" if kind == "ss" else "all"
    ends = (mesh["boundary_nodes"]["left"][0],
            mesh["boundary_nodes"]["right"][0])
    bc = {"constraints": [{"node": n, "dof": dof, "value": 0.0}
                          for n in ends]}
    _, _, c_dofs, _ = apply_dirichlet_bc(K, np.zeros(K.shape[0]), bc,
                                         dof_names=Beam.dof_names)
    _, phi = solve_eigen(K, M, c_dofs, n_modes=N_MODES,
                         dof_names=Beam.dof_names)
    return mesh, phi


def reference_gauss_points():
    """Two Gauss points and weights per element of the reference mesh."""
    x_nodes = np.linspace(0.0, L, REFERENCE_ELEMENTS + 1)
    xi, w = np.polynomial.legendre.leggauss(2)
    mid = 0.5 * (x_nodes[:-1] + x_nodes[1:])
    half = 0.5 * np.diff(x_nodes)
    points = (mid[:, None] + half[:, None] * xi[None, :]).ravel()
    weights = (half[:, None] * w[None, :]).ravel()
    return points, weights


def curvature_errors(kind):
    """Relative energy-norm curvature error, shape (len(ELEMENTS), N_MODES)."""
    x, weights = reference_gauss_points()
    mesh_ref, phi_ref = modes(REFERENCE_ELEMENTS, kind)
    kappa_ref = beam_curvature(mesh_ref, phi_ref, x)
    weight = E * I * weights[:, None]
    norm_ref = np.sqrt(np.sum(weight * kappa_ref ** 2, axis=0))

    errors = []
    for n in ELEMENTS:
        mesh, phi = modes(n, kind)
        kappa = beam_curvature(mesh, phi, x)
        # Align each mode's sign with the reference before differencing
        kappa *= np.sign(np.sum(weight * kappa * kappa_ref, axis=0))
        diff = np.sqrt(np.sum(weight * (kappa - kappa_ref) ** 2, axis=0))
        errors.append(diff / norm_ref)
    return np.array(errors)


def plot_curvature_convergence(kind, show):
    nodes = np.array(ELEMENTS) + 1
    errors = curvature_errors(kind)

    fig, axes = plt.subplots(2, 2, figsize=(6.4, 4.8), sharex=True)
    for i, ax in enumerate(axes.flat):
        ax.plot(nodes, errors[:, i], "o-", color=COLOUR)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(nodes, [str(n) for n in nodes])
        ax.minorticks_off()
        ax.set_ylabel(rf"$e_{i + 1}$")
    for ax in axes[1]:
        ax.set_xlabel("Number of nodes")
    fig.tight_layout()
    save(fig, f"beam_curvature_convergence_{NAMES[kind]}.pdf", show)

    print(f"{kind}: curvature error at {ELEMENTS[-1]} elements: "
          + ", ".join(f"{e:.1e}" for e in errors[-1]))


def plot_modes(kind, show):
    mesh, phi = modes(MODE_PLOT_ELEMENTS, kind)
    fig, _ = plot_beam_modes(mesh, phi, n_modes=N_MODES)
    save(fig, f"beam_modes_{NAMES[kind]}.pdf", show)


def save(fig, filename, show):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / filename)
    if not show:
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--show", action="store_true",
                        help="display the figures after saving them")
    args = parser.parse_args()

    setup_plot_style()
    for kind in ("ss", "cc"):
        plot_modes(kind, args.show)
        plot_curvature_convergence(kind, args.show)
    if args.show:
        plt.show()
