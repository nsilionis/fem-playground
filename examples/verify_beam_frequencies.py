"""
Convergence of beam natural frequencies under mesh refinement.

Simply supported and clamped-clamped Euler-Bernoulli beams, compared
with f_n = (beta_n L)^2 / (2 pi L^2) * sqrt(EI / rho A). Writes one
figure per support condition to outputs/figures; --show also displays
them.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

from femlib.assembly import assemble_mass, assemble_stiffness
from femlib.elements import Beam
from femlib.materials import LinearElastic1D
from femlib.mesh import create_beam_mesh
from femlib.postprocess import setup_plot_style
from femlib.sections import BeamSection
from femlib.solvers import apply_dirichlet_bc, solve_eigen

L, E, I, A, RHO = 10.0, 210e9, 8.0e-6, 0.01, 7850.0  # noqa: E741
N_MODES = 4
ELEMENTS = [4, 8, 16, 32, 64, 128]
COLOURS = {"fe": "#4878D0", "analytical": "#D65F5F"}
OUTPUT_DIR = Path("outputs/figures")


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


def frequencies(n_elements, kind):
    """First N_MODES finite element frequencies [Hz], 'ss' or 'cc'."""
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
    f, _ = solve_eigen(K, M, c_dofs, n_modes=N_MODES)
    return f


def plot_convergence(kind, filename, show=False):
    nodes = np.array(ELEMENTS) + 1
    f_fe = np.array([frequencies(n, kind) for n in ELEMENTS])
    f_ex = analytical(kind, N_MODES)

    fig, axes = plt.subplots(2, 2, figsize=(6.4, 4.8), sharex=True)
    for i, ax in enumerate(axes.flat):
        ax.plot(nodes, f_fe[:, i], "o-", color=COLOURS["fe"],
                label="Finite element")
        ax.axhline(f_ex[i], color=COLOURS["analytical"], linestyle="--",
                   label="Analytical")
        ax.set_xscale("log", base=2)
        ax.set_xticks(nodes, [str(n) for n in nodes])
        ax.minorticks_off()
        ax.set_ylabel(rf"$f_{i + 1}\ [\mathrm{{Hz}}]$")
    for ax in axes[1]:
        ax.set_xlabel("Number of nodes")
    axes[0, 0].legend()
    fig.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / filename)
    if not show:
        plt.close(fig)

    rel = f_fe[-1] / f_ex - 1
    print(f"{kind}: relative error at {ELEMENTS[-1]} elements: "
          + ", ".join(f"{r:.1e}" for r in rel))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--show", action="store_true",
                        help="display the figures after saving them")
    args = parser.parse_args()

    setup_plot_style()
    plot_convergence("ss", "beam_frequencies_simply_supported.pdf",
                     show=args.show)
    plot_convergence("cc", "beam_frequencies_clamped.pdf", show=args.show)
    if args.show:
        plt.show()
