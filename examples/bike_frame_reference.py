"""
Reference solution for the bicycle frame sizing assignment.

The frame is modelled as a plane pin-jointed truss of five aluminium
tubes. Point A is pinned and point D is held vertically. Vertical loads
act at B and C, and an optional horizontal load acts at D. By default
the static load at D is the weight of the rider and the frame together,
with the frame mass computed from the tube sections. Each static load
is multiplied by its own dynamic load factor. The script prints
the loads, the member properties, the nodal displacements, the member
stresses, a strength safety factor and a buckling utilisation. It
writes the deformed shape and the member stresses to
outputs/figures/bike_frame; --show also displays them.

Units are mm, N and MPa throughout. The tube diameters, the static
loads and the dynamic load factors may be set on the command line to
check a student's design. For example, for a design in which only the
load at D was amplified:

    uv run python examples/bike_frame_reference.py \\
        --diameters 30 30 32 20 16 --dlf 1 1 4

Members 1 to 3 are single tubes with D/t = 25. Members 4 and 5 are each
a pair of identical tubes with D/t = 10. The diameter given for them is
that of one tube, and their area and second moment of area are twice
those of one tube.

Stresses are positive in tension. The strength safety factor is
sigma_y / |sigma|. The buckling utilisation is |sigma| / sigma_cr for
members in compression, with the effective length equal to the member
length. sigma_cr is the Euler stress sigma_E = pi^2 E I / (A L^2), or
the Johnson parabola sigma_y (1 - sigma_y / (4 sigma_E)) where
sigma_E > sigma_y / 2.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from tabulate import tabulate

from femlib.assembly import assemble_load_vector, assemble_stiffness
from femlib.materials import LinearElastic1D
from femlib.postprocess import get_color_palette, setup_plot_style
from femlib.sections import TrussSection
from femlib.solvers import apply_dirichlet_bc, solve_system

# ── Geometry [mm], as dimensioned in the figure ───────────────────────
a, b, c, d = 250.0, 800.0, 550.0, 400.0

# Nodal coordinates, with the origin at C
NODE_NAMES = ["A", "B", "C", "D"]
NODES = np.array([
    [-a, c - d],    # A: rear axle
    [0.0, c],       # B: seat
    [0.0, 0.0],     # C: bottom bracket
    [b - a, c],     # D: head tube
])
NODE = {name: i for i, name in enumerate(NODE_NAMES)}

# Members 1 to 5, each named by its start and end nodes
MEMBER_NAMES = ["BD", "CD", "BC", "AB", "AC"]
ELEMENTS = np.array([[NODE[m[0]], NODE[m[1]]] for m in MEMBER_NAMES])

# ── Material [MPa, kg/mm^3] ───────────────────────────────────────────
E = 70e3
SIGMA_Y = 241.0
RHO = 2710e-9   # 2710 kg/m^3

# ── Cross-sections ────────────────────────────────────────────────────
D_OVER_T = np.array([25.0, 25.0, 25.0, 10.0, 10.0])
N_TUBES = np.array([1, 1, 1, 2, 2])
# Default diameters [mm]: with the wall thickness varied in 0.5 mm
# steps and D set by D/t, the smallest that give a strength safety
# factor of at least 1.25 and a buckling utilisation of at most 0.8
# under the default loads. Wall thicknesses: 1, 1, 0.5, 2 and 1 mm.
DEFAULT_DIAMETERS = [25.0, 25.0, 12.5, 20.0, 10.0]

# ── Default static loads [N] for a 125 kg rider ───────────────────────
# The default static load at D, rearward (-x), is (RIDER_MASS + frame
# mass) * G. It depends on the sections and is computed in main.
G = 9.81
RIDER_MASS = 125.0
DEFAULT_LOAD_B = 0.8 * RIDER_MASS * G   # downward, at B
DEFAULT_LOAD_C = 0.2 * RIDER_MASS * G   # downward, at C

# Default dynamic load factors for the loads at B, C and D
DEFAULT_DLF = [4.0, 4.0, 4.0]

OUTPUT_DIR = Path("outputs/figures/bike_frame")


def member_geometry(nodes, elements):
    """Length [mm] and angle from the x-axis [rad] of each member."""
    delta = nodes[elements[:, 1]] - nodes[elements[:, 0]]
    lengths = np.hypot(delta[:, 0], delta[:, 1])
    angles = np.arctan2(delta[:, 1], delta[:, 0])
    return lengths, angles


def tube_sections(diameters):
    """Wall thickness, area and second moment of area of each member.

    The area and second moment of area of members 4 and 5 are those of
    the pair of tubes.
    """
    D = np.asarray(diameters, dtype=float)
    t = D / D_OVER_T
    D_in = D - 2.0 * t
    area = N_TUBES * np.pi / 4.0 * (D ** 2 - D_in ** 2)
    second_moment = N_TUBES * np.pi / 64.0 * (D ** 4 - D_in ** 4)
    return t, area, second_moment


def frame_mass(areas, lengths):
    """Mass of the five members [kg]."""
    return RHO * np.sum(areas * lengths)


def solve_frame(areas, load_b, load_c, load_d):
    """Nodal displacements [mm], shape (n_nodes, 2)."""
    mesh = {"nodes": NODES, "elements": ELEMENTS, "element_type": "Truss"}
    material = LinearElastic1D(E=E)
    sections = [TrussSection(A=A) for A in areas]
    K = assemble_stiffness(mesh, material, sections)

    loads = {"concentrated": [
        {"location": tuple(NODES[NODE["B"]]), "force": [0.0, -load_b]},
        {"location": tuple(NODES[NODE["C"]]), "force": [0.0, -load_c]},
        {"location": tuple(NODES[NODE["D"]]), "force": [-load_d, 0.0]},
    ]}
    F = assemble_load_vector(mesh, loads)

    supports = {"constraints": [
        {"node": NODE["A"], "dof": "all", "value": 0.0},
        {"node": NODE["D"], "dof": "uy", "value": 0.0},
    ]}
    K_bc, F_bc, c_dofs, c_vals = apply_dirichlet_bc(K, F, supports)
    u = solve_system(K_bc, F_bc, c_dofs, c_vals)
    return u.reshape(-1, 2)


def member_stresses(u, lengths, angles):
    """Axial stress [MPa] of each member, positive in tension."""
    du = u[ELEMENTS[:, 1]] - u[ELEMENTS[:, 0]]
    elongation = np.cos(angles) * du[:, 0] + np.sin(angles) * du[:, 1]
    return E * elongation / lengths


def assess(sigma, areas, second_moments, lengths):
    """Strength safety factor and buckling utilisation of each member.

    Returns the safety factor, the Euler stress, the critical stress,
    and the utilisation, which is NaN for members not in compression.
    """
    magnitude = np.abs(sigma)
    with np.errstate(divide="ignore"):
        safety_factor = np.where(magnitude > 0.0, SIGMA_Y / magnitude,
                                 np.inf)
    sigma_euler = np.pi ** 2 * E * second_moments / (areas * lengths ** 2)
    sigma_cr = np.where(sigma_euler > SIGMA_Y / 2.0,
                        SIGMA_Y * (1.0 - SIGMA_Y / (4.0 * sigma_euler)),
                        sigma_euler)
    utilisation = np.where(sigma < 0.0, magnitude / sigma_cr, np.nan)
    return safety_factor, sigma_euler, sigma_cr, utilisation


def print_loads(static_loads, dlf):
    print("\nLoads")
    design = static_loads * dlf
    rows = zip(["B", "C", "D"], ["Downward", "Downward", "Rearward"],
               static_loads, dlf, design)
    print(tabulate(rows, tablefmt="grid",
                   floatfmt=("", "", ".1f", ".2f", ".1f"),
                   headers=["Point", "Direction", "Static [N]", "Factor",
                            "Design [N]"]))


def print_tables(diameters, t, areas, second_moments, lengths, angles,
                 u, sigma, safety_factor, sigma_euler, sigma_cr,
                 utilisation):
    members = [f"{i + 1} ({name})" for i, name in enumerate(MEMBER_NAMES)]

    print("\nMember properties")
    rows = zip(members, lengths, np.degrees(angles), diameters, t,
               N_TUBES, areas, second_moments)
    print(tabulate(rows, tablefmt="grid", floatfmt=".2f", headers=[
        "Member", "L [mm]", "Angle [deg]", "D [mm]", "t [mm]", "Tubes",
        "A [mm²]", "I [mm⁴]"]))

    print("\nNodal displacements")
    rows = zip(NODE_NAMES, NODES[:, 0], NODES[:, 1], u[:, 0], u[:, 1],
               np.hypot(u[:, 0], u[:, 1]))
    print(tabulate(rows, tablefmt="grid",
                   floatfmt=(".0f", ".0f", ".0f", ".4f", ".4f", ".4f"),
                   headers=["Node", "x [mm]", "y [mm]", "u_x [mm]",
                            "u_y [mm]", "|u| [mm]"]))

    print("\nMember stresses (positive in tension)")
    states = np.where(sigma > 0.0, "Tension",
                      np.where(sigma < 0.0, "Compression", "Unstressed"))
    rows = zip(members, sigma * areas, sigma, states)
    print(tabulate(rows, tablefmt="grid", floatfmt=".1f",
                   headers=["Member", "N [N]", "σ [MPa]", "State"]))

    print(f"\nStrength and buckling assessment (σ_y = {SIGMA_Y:.0f} MPa)")
    rows = []
    for i in range(len(members)):
        compressive = sigma[i] < 0.0
        curve = "Johnson" if sigma_euler[i] > SIGMA_Y / 2.0 else "Euler"
        rows.append([
            members[i],
            f"{sigma[i]:.1f}",
            f"{safety_factor[i]:.2f}" if np.isfinite(safety_factor[i])
            else "-",
            f"{sigma_euler[i]:.1f}" if compressive else "-",
            f"{sigma_cr[i]:.1f}" if compressive else "-",
            curve if compressive else "-",
            f"{utilisation[i]:.2f}" if compressive else "-",
        ])
    print(tabulate(rows, tablefmt="grid", disable_numparse=True,
                   colalign=("left", "right", "right", "right", "right",
                             "left", "right"),
                   headers=["Member", "σ [MPa]", "Safety factor",
                            "σ_E [MPa]", "σ_cr [MPa]", "Curve",
                            "Utilisation"]))


def nice_scale(target):
    """Largest number of the form 1, 2 or 5 x 10^k not above target."""
    power = 10.0 ** np.floor(np.log10(target))
    return max(m for m in (1.0, 2.0, 5.0) if m * power <= target) * power


def label_nodes(ax, nodes):
    offsets = {"A": (-10, -4), "B": (8, 10), "C": (10, -12), "D": (8, 8)}
    for name, (x, y) in zip(NODE_NAMES, nodes):
        ax.annotate(rf"${name}$", (x, y), textcoords="offset points",
                    xytext=offsets[name], ha="center", va="center")


def plot_deformed(u, scale, show):
    colours = get_color_palette()
    deformed = NODES + scale * u

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for i, (start, end) in enumerate(ELEMENTS):
        ax.plot(NODES[[start, end], 0], NODES[[start, end], 1], "--",
                color=colours["neutral"],
                label="Undeformed" if i == 0 else None)
        ax.plot(deformed[[start, end], 0], deformed[[start, end], 1], "-",
                color=colours["primary"], lw=2.0,
                label=rf"Deformed ($\times{scale:g}$)" if i == 0 else None)
    ax.plot(NODES[:, 0], NODES[:, 1], "o", color=colours["neutral"],
            mfc="white", ms=5)
    ax.plot(deformed[:, 0], deformed[:, 1], "o", color=colours["primary"],
            mfc="white", ms=5)
    label_nodes(ax, NODES)
    finish(ax)
    ax.legend(loc="lower right")
    save(fig, "bike_frame_deformed.pdf", show)


def plot_stresses(sigma, show):
    limit = np.max(np.abs(sigma))
    norm = Normalize(vmin=-limit, vmax=limit)
    cmap = plt.get_cmap("coolwarm")

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for i, (start, end) in enumerate(ELEMENTS):
        x, y = NODES[[start, end], 0], NODES[[start, end], 1]
        ax.plot(x, y, "-", color="black", lw=6.5, solid_capstyle="butt")
        ax.plot(x, y, "-", color=cmap(norm(sigma[i])), lw=5.0,
                solid_capstyle="butt")
        label_member(ax, i, rf"$\sigma_{{{i + 1}}} = {sigma[i]:+.1f}$")
    ax.plot(NODES[:, 0], NODES[:, 1], "o", color="black", mfc="white",
            ms=5, zorder=3)
    label_nodes(ax, NODES)
    finish(ax)
    fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                 label=r"$\sigma$ [MPa]", shrink=0.8)
    save(fig, "bike_frame_stress.pdf", show)


# Offsets [points] of the member labels from the member midpoints
MEMBER_LABEL_OFFSETS = [(0, -16), (30, -14), (34, 0), (-32, 10), (-14, -18)]


def label_member(ax, i, text):
    """Write text beside the midpoint of member i."""
    start, end = NODES[ELEMENTS[i]]
    ax.annotate(text, 0.5 * (start + end), textcoords="offset points",
                xytext=MEMBER_LABEL_OFFSETS[i], ha="center", va="center")


def finish(ax):
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x$ [mm]")
    ax.set_ylabel(r"$y$ [mm]")
    ax.margins(0.12)


def save(fig, filename, show):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / filename)
    print(f"Saved: {OUTPUT_DIR / filename}")
    if not show:
        plt.close(fig)


def main(diameters, static_loads, dlf, scale=None, show=False):
    """Solve and assess the frame.

    static_loads holds the static loads at B, C and D [N], and dlf the
    dynamic load factor applied to each. A load of None at D is
    replaced by the weight of the rider and the frame.
    """
    diameters = np.asarray(diameters, dtype=float)
    dlf = np.asarray(dlf, dtype=float)
    lengths, angles = member_geometry(NODES, ELEMENTS)
    t, areas, second_moments = tube_sections(diameters)
    m_frame = frame_mass(areas, lengths)
    load_b, load_c, load_d = static_loads
    if load_d is None:
        load_d = (RIDER_MASS + m_frame) * G
    static_loads = np.array([load_b, load_c, load_d], dtype=float)
    u = solve_frame(areas, *(static_loads * dlf))
    sigma = member_stresses(u, lengths, angles)
    safety_factor, sigma_euler, sigma_cr, utilisation = assess(
        sigma, areas, second_moments, lengths)

    print(f"Frame mass: {m_frame:.3f} kg")
    print_loads(static_loads, dlf)
    print_tables(diameters, t, areas, second_moments, lengths, angles, u,
                 sigma, safety_factor, sigma_euler, sigma_cr, utilisation)

    if scale is None:
        # Largest displacement drawn as about a tenth of the frame size
        size = np.ptp(NODES, axis=0).max()
        scale = nice_scale(0.1 * size / np.max(np.abs(u)))
    print(f"\nDisplacement scale factor: {scale:g}")

    setup_plot_style()
    plot_deformed(u, scale, show)
    plot_stresses(sigma, show)
    if show:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--diameters", type=float, nargs=5,
                        default=DEFAULT_DIAMETERS,
                        metavar=("D1", "D2", "D3", "D4", "D5"),
                        help="outer tube diameters of members 1 to 5 [mm];"
                             " for members 4 and 5, of one tube")
    parser.add_argument("--load-b", type=float, default=DEFAULT_LOAD_B,
                        help="static downward load at B [N] "
                             "(default: %(default).1f)")
    parser.add_argument("--load-c", type=float, default=DEFAULT_LOAD_C,
                        help="static downward load at C [N] "
                             "(default: %(default).1f)")
    parser.add_argument("--load-d", type=float, default=None,
                        help="static rearward load at D [N]; 0 omits it "
                             "(default: weight of the rider and the "
                             "frame)")
    parser.add_argument("--dlf", type=float, nargs=3, default=DEFAULT_DLF,
                        metavar=("DLF_B", "DLF_C", "DLF_D"),
                        help="dynamic load factors for the loads at B, C "
                             "and D (default: 4 4 4)")
    parser.add_argument("--scale", type=float, default=None,
                        help="displacement scale factor for the deformed "
                             "shape (default: chosen automatically)")
    parser.add_argument("--show", action="store_true",
                        help="display the figures after saving them")
    args = parser.parse_args()

    if min(args.diameters) <= 0.0:
        parser.error("diameters must be positive")
    if min(args.dlf) <= 0.0:
        parser.error("dynamic load factors must be positive")
    main(args.diameters, [args.load_b, args.load_c, args.load_d], args.dlf,
         scale=args.scale, show=args.show)
