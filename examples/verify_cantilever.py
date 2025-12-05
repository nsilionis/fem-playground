"""
Cantilever beam verification against Euler-Bernoulli theory.

This script verifies the FEM implementation by comparing displacement
solutions with analytical Euler-Bernoulli beam theory for a cantilever
subjected to combined point and distributed loading.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from femlib.mesh import create_rectangular_mesh
from femlib.materials import LinearElastic
from femlib.assembly import assemble_stiffness, assemble_load_vector
from femlib.solvers import (
    identify_boundary_nodes,
    apply_dirichlet_bc,
    solve_system,
)
from femlib.postprocess import setup_plot_style


def euler_bernoulli_deflection(x, L, P, q, E, I_area):
    """
    Analytical deflection for cantilever beam (Euler-Bernoulli).

    Parameters
    ----------
    x : array_like
        Position along beam from fixed end [m].
    L : float
        Beam length [m].
    P : float
        Point load at free end [N] (negative for downward).
    q : float
        Distributed load [N/m] (negative for downward).
    E : float
        Young's modulus [Pa].
    I_area : float
        Second moment of area [m^4].

    Returns
    -------
    v : ndarray
        Vertical deflection at position x [m].

    Notes
    -----
    Deflection formula combines point load and distributed load:
        v(x) = P/(6EI) * (3Lx² - x³) + q/(24EI) * (6L²x² - 4Lx³ + x⁴)
    """
    x = np.asarray(x)

    # Point load contribution
    v_point = (P / (6 * E * I_area)) * (3 * L * x**2 - x**3)

    # Distributed load contribution
    v_distributed = (q / (24 * E * I_area)) * (
        6 * L**2 * x**2 - 4 * L * x**3 + x**4
    )

    return v_point + v_distributed


def extract_neutral_axis_displacements(mesh, u, y_target=None, tolerance=1e-6):
    """
    Extract displacements along neutral axis (mid-height) of beam.

    If no nodes exist on the neutral axis (coarse mesh with ny=1),
    averages top and bottom surface displacements.

    Parameters
    ----------
    mesh : dict
        Mesh dictionary.
    u : ndarray
        Full displacement vector [u1, v1, u2, v2, ...].
    y_target : float, optional
        y-coordinate of neutral axis. If None, uses mid-height.
    tolerance : float, optional
        Tolerance for identifying neutral axis nodes.

    Returns
    -------
    x_coords : ndarray
        x-coordinates along beam.
    v_displacements : ndarray
        Vertical (v) displacements.
    method : str
        Method used: 'neutral_axis' or 'averaged'
    """
    nodes = mesh["nodes"]

    if y_target is None:
        y_min = nodes[:, 1].min()
        y_max = nodes[:, 1].max()
        y_target = 0.5 * (y_min + y_max)

    # Try finding nodes on neutral axis
    mask = np.abs(nodes[:, 1] - y_target) < tolerance
    neutral_nodes = np.where(mask)[0]

    if len(neutral_nodes) > 0:
        # Nodes exist on neutral axis - use directly
        x_coords = nodes[neutral_nodes, 0]
        v_displacements = u[2 * neutral_nodes + 1]

        sort_idx = np.argsort(x_coords)
        return x_coords[sort_idx], v_displacements[sort_idx], "neutral_axis"

    # No nodes on neutral axis - average top and bottom
    top_nodes = mesh["boundary_nodes"]["top"]
    bottom_nodes = mesh["boundary_nodes"]["bottom"]

    # Get coordinates
    x_top = nodes[top_nodes, 0]
    x_bottom = nodes[bottom_nodes, 0]

    # Get displacements
    v_top = u[2 * top_nodes + 1]
    v_bottom = u[2 * bottom_nodes + 1]

    # Sort both by x
    sort_top = np.argsort(x_top)
    sort_bottom = np.argsort(x_bottom)

    x_top = x_top[sort_top]
    v_top = v_top[sort_top]
    x_bottom = x_bottom[sort_bottom]
    v_bottom = v_bottom[sort_bottom]

    # Verify same x-coordinates
    if not np.allclose(x_top, x_bottom):
        raise ValueError("Top and bottom edges have different x-coordinates")

    # Average displacements
    v_avg = 0.5 * (v_top + v_bottom)

    return x_top, v_avg, "averaged"


def main():
    """Run cantilever verification analysis."""
    # Apply custom plotting style
    setup_plot_style()

    print("=" * 70)
    print("CANTILEVER BEAM VERIFICATION")
    print("Comparison with Euler-Bernoulli analytical solution")
    print("=" * 70)

    # =================================================================
    # PROBLEM DEFINITION
    # =================================================================

    # Geometry
    L = 10.0  # Length [m]
    h = 1.0  # Height [m]
    t = 0.1  # Thickness [m]

    # Material properties
    E = 210e9  # Young's modulus [Pa]
    nu = 0.3  # Poisson's ratio

    # Loading
    P = -10000.0  # Point load at tip [N] (negative = downward)
    q = -1000.0  # Distributed load [N/m] (negative = downward)

    # Mesh density
    nx = 40  # Elements along length
    ny = 10  # Elements along height

    print("\nProblem Parameters:")
    print(f"  Geometry: L={L:.1f} m, h={h:.1f} m, t={t:.3f} m")
    print(f"  Material: E={E:.2e} Pa, ν={nu:.2f}")
    print(f"  Loading: P={P:.1f} N, q={q:.1f} N/m")
    print(f"  Mesh: {nx}×{ny} = {nx * ny} elements")

    # =================================================================
    # ANALYTICAL SOLUTION
    # =================================================================

    # Second moment of area for rectangular cross-section
    I_area = (t * h**3) / 12.0

    print("\nAnalytical Solution (Euler-Bernoulli):")
    print(f"  Second moment of area: I={I_area:.6e} m⁴")

    # Tip deflection (analytical)
    delta_tip_point = (P * L**3) / (3 * E * I_area)
    delta_tip_dist = (q * L**4) / (8 * E * I_area)
    delta_tip_total = delta_tip_point + delta_tip_dist

    print(
        f"  Tip deflection (point load): {delta_tip_point*1000:.4f} mm"
    )
    print(
        f"  Tip deflection (distributed): {delta_tip_dist*1000:.4f} mm"
    )
    print(f"  Total tip deflection: {delta_tip_total*1000:.4f} mm")

    # =================================================================
    # FEM SOLUTION
    # =================================================================

    print("\nFEM Solution:")

    # Create mesh
    print("  Creating mesh...")
    mesh = create_rectangular_mesh(L, h, nx, ny)
    n_nodes = mesh["nodes"].shape[0]
    n_dofs = 2 * n_nodes
    print(f"    Nodes: {n_nodes}, DOFs: {n_dofs}")

    # Define material
    material = LinearElastic(
        E=E, nu=nu, thickness=t, plane_type="stress"
    )

    # Assemble stiffness matrix
    print("  Assembling stiffness matrix...")
    K = assemble_stiffness(mesh, material, integration_order="full")
    print(f"    K shape: {K.shape}")

    # Assemble load vector
    print("  Assembling load vector...")

    # Point load at tip (find node at x=L, y=h/2)
    tip_node = np.argmin(
        (mesh["nodes"][:, 0] - L) ** 2
        + (mesh["nodes"][:, 1] - h / 2) ** 2
    )

    load_spec = {
        "concentrated": [
            {"location": mesh["nodes"][tip_node], "force": [0.0, P]}
        ],
        "distributed": [{"edge": "top", "load_density": [0.0, q]}],
    }

    F = assemble_load_vector(mesh, load_spec)
    print(f"    F shape: {F.shape}")
    print(f"    Total applied force: {np.sum(F):.2f} N")

    # Apply boundary conditions (fixed left edge)
    print("  Applying boundary conditions...")
    fixed_nodes = identify_boundary_nodes(mesh, "left")
    print(f"    Fixed nodes: {len(fixed_nodes)}")

    bc = {"nodes": fixed_nodes, "dofs": ["both"], "values": [0.0]}

    K_bc, F_bc, c_dofs, c_vals = apply_dirichlet_bc(K, F, bc)
    print(f"    Reduced system: {K_bc.shape[0]} DOFs")

    # Solve system
    print("  Solving linear system...")
    u = solve_system(K_bc, F_bc, c_dofs, c_vals)
    print(f"    Solution vector: {u.shape}")

    # Extract tip displacement
    u_tip = u[2 * tip_node]  # Horizontal displacement
    v_tip = u[2 * tip_node + 1]  # Vertical displacement

    print(f"\n  FEM tip displacement: {v_tip*1000:.4f} mm")
    print(f"  FEM tip horizontal: {u_tip*1000:.6f} mm")

    # =================================================================
    # COMPARISON AND ERROR ANALYSIS
    # =================================================================

    print("\nComparison:")
    error_tip = np.abs(v_tip - delta_tip_total)
    rel_error_tip = 100 * error_tip / np.abs(delta_tip_total)

    print(f"  Analytical tip: {delta_tip_total*1000:.4f} mm")
    print(f"  FEM tip:        {v_tip*1000:.4f} mm")
    print(f"  Absolute error: {error_tip*1000:.4f} mm")
    print(f"  Relative error: {rel_error_tip:.3f}%")

    # Extract displacement profile
    x_fem, v_fem, extraction_method = extract_neutral_axis_displacements(
        mesh, u
    )

    if extraction_method == "averaged":
        print("\n  Note: Using averaged top/bottom displacements")
        print(f"        (no nodes on neutral axis with ny={ny})")

    # Compute analytical solution at same x-coordinates
    v_analytical = euler_bernoulli_deflection(x_fem, L, P, q, E, I_area)

    # Compute pointwise errors
    abs_errors = np.abs(v_fem - v_analytical)
    rel_errors = 100 * abs_errors / np.abs(v_analytical)

    print(
        f"\n  Max absolute error along beam: {np.max(abs_errors)*1000:.4f} mm"
    )
    print(f"  Max relative error along beam: {np.max(rel_errors):.3f}%")
    print(f"  Mean relative error: {np.mean(rel_errors):.3f}%")

    # =================================================================
    # VISUALISATION
    # =================================================================

    print("\nGenerating plots...")

    # Create output directory
    output_dir = Path("outputs/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: Deflection comparison
    fig, ax = plt.subplots(figsize=(10, 5))

    # Analytical solution (smooth curve)
    x_smooth = np.linspace(0, L, 200)
    v_smooth = euler_bernoulli_deflection(x_smooth, L, P, q, E, I_area)

    ax.plot(
        x_smooth,
        v_smooth * 1000,
        "k-",
        linewidth=1.5,
        label="Euler-Bernoulli (analytical)",
        zorder=1,
    )

    marker_label = "FEM solution"
    if extraction_method == "averaged":
        marker_label += " (averaged)"

    ax.plot(
        x_fem,
        v_fem * 1000,
        "o",
        color="#1f77b4",
        markersize=5,
        markeredgecolor="k",
        markeredgewidth=0.5,
        label=marker_label,
        zorder=2,
    )

    ax.set_xlabel(r"$x \ \mathrm{[m]}$")
    ax.set_ylabel(r"$v \ \mathrm{[mm]}$")
    ax.legend(framealpha=0.95, loc="lower left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.2, L + 0.2)

    plt.tight_layout()
    output_path = output_dir / "cantilever_verification_deflection.pdf"
    plt.savefig(output_path, bbox_inches="tight")
    print(f"  Saved: {output_path}")

    # Plot 2: Error distribution
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        x_fem,
        rel_errors,
        "o-",
        color="#d62728",
        markersize=5,
        markeredgecolor="k",
        markeredgewidth=0.5,
        linewidth=1.0,
    )

    ax.set_xlabel(r"$x \ \mathrm{[m]}$")
    ax.set_ylabel(r"Relative error [\%]")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.2, L + 0.2)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    output_path = output_dir / "cantilever_verification_error.pdf"
    plt.savefig(output_path, bbox_inches="tight")
    print(f"  Saved: {output_path}")

    plt.show()

    # =================================================================
    # SUMMARY
    # =================================================================

    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)

    # Pass/fail criteria
    tolerance = 2.0  # 2% tolerance

    if rel_error_tip < tolerance:
        status = "PASS ✓"
    else:
        status = "FAIL ✗"

    print(f"\nTip displacement error: {rel_error_tip:.3f}% [{status}]")
    print(f"Tolerance: {tolerance:.1f}%")

    if np.max(rel_errors) < tolerance:
        status = "PASS ✓"
    else:
        status = "FAIL ✗"

    print(f"\nMax error along beam: {np.max(rel_errors):.3f}% [{status}]")

    if ny == 1:
        print("\nNote: For ny=1, displacements are averaged from")
        print("      top/bottom surfaces. Use ny>=2 for neutral axis")
        print("      extraction and better accuracy.")

    print("\n" + "=" * 70)
    print("Verification complete!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
