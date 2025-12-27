"""
Reference solution for cantilever beam locking assignment.

This script generates all required results for the shear locking
investigation assignment, including displacement comparisons,
error plots, stress contours, and tabulated results.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

from femlib.mesh import create_rectangular_mesh
from femlib.materials import LinearElastic2D
from femlib.assembly import assemble_stiffness, assemble_load_vector
from femlib.solvers import (
    identify_boundary_nodes,
    apply_dirichlet_bc,
    solve_system
)
from femlib.postprocess import (
    setup_plot_style,
    compute_stress_strain_field,
    plot_stress_field
)


def euler_bernoulli_point_load(x, L, P, E, second_moment):
    """
    Analytical deflection for cantilever with point load at tip.

    Parameters
    ----------
    x : array_like
        Position along beam from fixed end [m].
    L : float
        Beam length [m].
    P : float
        Point load at tip [N] (negative for downward).
    E : float
        Young's modulus [Pa].
    second_moment : float
        Second moment of area [m^4].

    Returns
    -------
    v : ndarray
        Vertical deflection at position x [m].

    Notes
    -----
    Deflection formula: v(x) = P/(6EI) * (3Lx² - x³)
    """
    x = np.asarray(x)
    v = (P / (6 * E * second_moment)) * (3 * L * x**2 - x**3)
    return v


def extract_neutral_axis_displacements(mesh, u, y_target=None,
                                       tolerance=1e-6):
    """
    Extract displacements along neutral axis.

    Handles cases where no nodes exist on neutral axis by
    averaging top and bottom surfaces.
    """
    nodes = mesh['nodes']

    if y_target is None:
        y_min = nodes[:, 1].min()
        y_max = nodes[:, 1].max()
        y_target = 0.5 * (y_min + y_max)

    # Try finding nodes on neutral axis
    mask = np.abs(nodes[:, 1] - y_target) < tolerance
    neutral_nodes = np.where(mask)[0]

    if len(neutral_nodes) > 0:
        # Nodes exist on neutral axis
        x_coords = nodes[neutral_nodes, 0]
        v_displacements = u[2 * neutral_nodes + 1]

        sort_idx = np.argsort(x_coords)
        return x_coords[sort_idx], v_displacements[sort_idx]

    else:
        # Average top and bottom surfaces
        top_nodes = mesh['boundary_nodes']['top']
        bottom_nodes = mesh['boundary_nodes']['bottom']

        x_top = nodes[top_nodes, 0]
        x_bottom = nodes[bottom_nodes, 0]

        v_top = u[2 * top_nodes + 1]
        v_bottom = u[2 * bottom_nodes + 1]

        sort_top = np.argsort(x_top)
        sort_bottom = np.argsort(x_bottom)

        x_top = x_top[sort_top]
        v_top = v_top[sort_top]
        x_bottom = x_bottom[sort_bottom]
        v_bottom = v_bottom[sort_bottom]

        v_avg = 0.5 * (v_top + v_bottom)

        return x_top, v_avg


def solve_cantilever(L, h, t, E, nu, P, nx, ny, integration_order):
    """
    Solve cantilever beam problem with specified parameters.

    Returns
    -------
    results : dict
        Dictionary containing mesh, displacement, and stress data.
    """
    # Create mesh
    mesh = create_rectangular_mesh(L, h, nx, ny)

    # Define material
    material = LinearElastic2D(
        E=E,
        nu=nu,
        thickness=t,
        plane_type='stress'
    )

    # Assemble stiffness
    K = assemble_stiffness(mesh, material, integration_order)

    # Apply point load at tip (find node closest to (L, h/2))
    tip_node = np.argmin(
        (mesh['nodes'][:, 0] - L)**2 +
        (mesh['nodes'][:, 1] - h/2)**2
    )

    load_spec = {
        'concentrated': [
            {'location': mesh['nodes'][tip_node], 'force': [0.0, P]}
        ]
    }

    F = assemble_load_vector(mesh, load_spec)

    # Apply boundary conditions
    fixed_nodes = identify_boundary_nodes(mesh, 'left')
    bc = {
        'nodes': fixed_nodes,
        'dofs': ['both'],
        'values': [0.0]
    }

    K_bc, F_bc, c_dofs, c_vals = apply_dirichlet_bc(K, F, bc)

    # Solve
    u = solve_system(K_bc, F_bc, c_dofs, c_vals)

    # Compute stresses
    field_data = compute_stress_strain_field(
        mesh, u, material, integration_order
    )

    # Extract results
    x_fem, v_fem = extract_neutral_axis_displacements(mesh, u)
    v_tip = u[2 * tip_node + 1]

    max_sigma_xx = np.max(np.abs(field_data['nodal']['stress'][:, 0]))
    max_von_mises = np.max(field_data['nodal']['von_mises'])

    results = {
        'mesh': mesh,
        'u': u,
        'field_data': field_data,
        'x_fem': x_fem,
        'v_fem': v_fem,
        'v_tip': v_tip,
        'max_sigma_xx': max_sigma_xx,
        'max_von_mises': max_von_mises,
        'tip_node': tip_node
    }

    return results


def main():
    """Run complete assignment reference analysis."""
    # Apply plotting style
    setup_plot_style()

    print("=" * 70)
    print("CANTILEVER BEAM SHEAR LOCKING ASSIGNMENT - REFERENCE SOLUTION")
    print("=" * 70)

    # ================================================================
    # PROBLEM PARAMETERS
    # ================================================================

    L = 10.0        # Length [m]
    h = 0.5         # Height [m]
    t = 0.1         # Thickness [m]
    E = 210e9       # Young's modulus [Pa]
    nu = 0.3        # Poisson's ratio
    P = -20000.0    # Point load [N] (20 kN downward)

    # Mesh configurations
    mesh_configs = [
        (8, 1, 'o', '-', 8),      # (nx, ny, marker, linestyle, size)
        (16, 4, 's', '--', 6),
        (40, 20, '^', ':', 4)
    ]

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    print("\nProblem Parameters:")
    print(f"  Geometry: L={L} m, h={h} m, t={t} m (L/h = {L/h})")
    print(f"  Material: E={E/1e9:.0f} GPa, ν={nu}")
    print(f"  Loading: P={P/1000:.0f} kN (point load at tip)")
    print(f"  Mesh configurations: {len(mesh_configs)}")

    # ================================================================
    # ANALYTICAL SOLUTION
    # ================================================================

    second_moment = (t * h**3) / 12.0
    delta_tip_analytical = (P * L**3) / (3 * E * second_moment)

    print("\nAnalytical Solution (Euler-Bernoulli):")
    print(f"  Second moment of area: I={second_moment:.6e} m⁴")
    print(f"  Tip deflection: {delta_tip_analytical*1000:.4f} mm")

    # ================================================================
    # FEM SOLUTIONS
    # ================================================================

    print("\nSolving FEM problems...")

    results_full = []
    results_reduced = []

    for (nx, ny, _, _, _) in mesh_configs:
        print(f"  Mesh {nx}×{ny}...", end=" ")

        # Full integration
        res_full = solve_cantilever(
            L, h, t, E, nu, P, nx, ny, 'full'
        )
        results_full.append(res_full)

        # Reduced integration (skip 8×1 due to hourglassing)
        if nx == 8 and ny == 1:
            print("Full done, reduced skipped (hourglassing)")
            results_reduced.append(None)  # Placeholder
        else:
            res_reduced = solve_cantilever(
                L, h, t, E, nu, P, nx, ny, 'reduced'
            )
            results_reduced.append(res_reduced)
            print("Done (both)")

    # ================================================================
    # RESULTS TABLE
    # ================================================================

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    table_data = []

    for i, (nx, ny, _, _, _) in enumerate(mesh_configs):
        # Full integration
        v_tip_full = results_full[i]['v_tip'] * 1000  # mm
        error_full = 100 * (
            v_tip_full / (delta_tip_analytical * 1000) - 1
        )
        sigma_xx_full = results_full[i]['max_sigma_xx'] / 1e6
        vm_full = results_full[i]['max_von_mises'] / 1e6

        table_data.append([
            f"{nx}×{ny}",
            "Full",
            f"{v_tip_full:.4f}",
            f"{error_full:.2f}",
            f"{sigma_xx_full:.2f}",
            f"{vm_full:.2f}"
        ])

        # Reduced integration (skip if None)
        if results_reduced[i] is not None:
            v_tip_red = results_reduced[i]['v_tip'] * 1000
            error_red = 100 * (
                v_tip_red / (delta_tip_analytical * 1000) - 1
            )
            sigma_xx_red = (
                results_reduced[i]['max_sigma_xx'] / 1e6
            )
            vm_red = results_reduced[i]['max_von_mises'] / 1e6

            table_data.append([
                f"{nx}×{ny}",
                "Reduced",
                f"{v_tip_red:.4f}",
                f"{error_red:.2f}",
                f"{sigma_xx_red:.2f}",
                f"{vm_red:.2f}"
            ])
        else:
            # Hourglassing case - show as N/A
            table_data.append([
                f"{nx}×{ny}",
                "Reduced",
                "N/A*",
                "N/A*",
                "N/A*",
                "N/A*"
            ])

    headers = [
        "Mesh",
        "Integration",
        "Tip Disp. [mm]",
        "Error [%]",
        "Max σ_xx [MPa]",
        "Max σ_vm [MPa]"
    ]

    print("\n" + tabulate(table_data, headers=headers,
                          tablefmt="grid"))

    print(f"\nAnalytical tip displacement: "
          f"{delta_tip_analytical*1000:.4f} mm")
    print("\n* N/A: Reduced integration with ny=1 produces "
          "hourglassing")

    # ================================================================
    # CREATE OUTPUT DIRECTORY
    # ================================================================

    output_dir = Path('outputs/figures/assignment')
    output_dir.mkdir(parents=True, exist_ok=True)

    # ================================================================
    # DISPLACEMENT COMPARISON PLOTS
    # ================================================================

    print("\nGenerating displacement comparison plots...")

    # Analytical solution (smooth curve)
    x_smooth = np.linspace(0, L, 200)
    v_smooth_analytical = euler_bernoulli_point_load(
        x_smooth, L, P, E, second_moment
    )

    for integration_name, results_list in [
        ('full', results_full),
        ('reduced', results_reduced)
    ]:
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot analytical
        ax.plot(
            x_smooth,
            v_smooth_analytical * 1000,
            'k-',
            linewidth=2,
            label='Euler-Bernoulli (analytical)',
            zorder=1
        )

        # Plot FEM results
        for i, (nx, ny, marker, linestyle, markersize) in enumerate(
            mesh_configs
        ):
            # Skip if this result is None (hourglassing)
            if results_list[i] is None:
                continue

            x_fem = results_list[i]['x_fem']
            v_fem = results_list[i]['v_fem'] * 1000  # mm

            ax.plot(
                x_fem,
                v_fem,
                marker=marker,
                linestyle=linestyle,
                color=colors[i],
                markersize=markersize,
                markeredgecolor='k',
                markeredgewidth=0.5,
                linewidth=1.5,
                label=f'{nx}×{ny}',
                zorder=2 + i
            )

        ax.set_xlabel(r'$x$ [m]')
        ax.set_ylabel(r'$v$ [mm]')
        ax.set_title(
            f'Displacement Comparison - '
            f'{integration_name.capitalize()} Integration'
        )
        ax.legend(framealpha=0.95, loc='lower left')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.2, L + 0.2)

        plt.tight_layout()
        filename = f'displacement_{integration_name}_integration.pdf'
        plt.savefig(output_dir / filename, bbox_inches='tight')
        print(f"  Saved: {filename}")
        plt.close()

    # ================================================================
    # ERROR PLOTS
    # ================================================================

    print("\nGenerating error plots...")

    for integration_name, results_list in [
        ('full', results_full),
        ('reduced', results_reduced)
    ]:
        fig, ax = plt.subplots(figsize=(10, 6))

        for i, (nx, ny, marker, linestyle, markersize) in enumerate(
            mesh_configs
        ):
            # Skip if this result is None (hourglassing)
            if results_list[i] is None:
                continue

            x_fem = results_list[i]['x_fem']
            v_fem = results_list[i]['v_fem']

            # Analytical at same points
            v_analytical = euler_bernoulli_point_load(
                x_fem, L, P, E, second_moment
            )

            # Percentage error
            errors = 100 * np.abs(v_fem - v_analytical) / np.abs(
                v_analytical
            )

            ax.plot(
                x_fem,
                errors,
                marker=marker,
                linestyle=linestyle,
                color=colors[i],
                markersize=markersize,
                markeredgecolor='k',
                markeredgewidth=0.5,
                linewidth=1.5,
                label=f'{nx}×{ny}'
            )

        ax.set_xlabel(r'$x$ [m]')
        ax.set_ylabel(r'Relative Error [\%]')
        ax.set_title(
            f'Displacement Error - '
            f'{integration_name.capitalize()} Integration'
        )
        ax.legend(framealpha=0.95)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.2, L + 0.2)
        ax.set_ylim(bottom=0)

        plt.tight_layout()
        filename = f'error_{integration_name}_integration.pdf'
        plt.savefig(output_dir / filename, bbox_inches='tight')
        print(f"  Saved: {filename}")
        plt.close()

    # ================================================================
    # STRESS CONTOUR PLOTS
    # ================================================================

    print("\nGenerating stress contour plots...")

    stress_components = [
        ('sigma_xx', 'Normal Stress (x-direction)'),
        ('von_mises', 'von Mises Stress')
    ]

    for i, (nx, ny, _, _, _) in enumerate(mesh_configs):
        for integration_name, results_list in [
            ('full', results_full),
            ('reduced', results_reduced)
        ]:
            # Skip if result is None (hourglassing)
            if results_list[i] is None:
                continue

            mesh = results_list[i]['mesh']
            field_data = results_list[i]['field_data']

            for component, title_suffix in stress_components:
                fig, ax, cbar = plot_stress_field(
                    mesh,
                    field_data,
                    component,
                    levels=20,
                    show_mesh=True
                )

                # Update title
                ax.set_title(
                    f'{title_suffix} - {nx}×{ny} - '
                    f'{integration_name.capitalize()} Integration'
                )

                filename = (
                    f'stress_{component}_{nx}x{ny}_'
                    f'{integration_name}.pdf'
                )
                plt.savefig(output_dir / filename,
                            bbox_inches='tight')
                print(f"  Saved: {filename}")
                plt.close()

    # ================================================================
    # SUMMARY
    # ================================================================

    print("\n" + "=" * 70)
    print("REFERENCE SOLUTION COMPLETE")
    print("=" * 70)
    print(f"\nAll outputs saved to: {output_dir}")
    print("  Displacement plots: 2")
    print("  Error plots: 2")
    print("  Stress contour plots: 10 (8×1 reduced skipped)")
    print("  Total figures: 14")
    print("\nNote: 8×1 mesh with reduced integration skipped due to")
    print("      hourglassing (zero-energy modes with ny=1).")
    print("\n" + "=" * 70 + "\n")


if __name__ == '__main__':
    main()
