"""Test stress/strain visualisation."""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from femlib.mesh import create_rectangular_mesh
from femlib.materials import LinearElastic
from femlib.assembly import assemble_stiffness, assemble_load_vector
from femlib.solvers import (
    identify_boundary_nodes,
    apply_dirichlet_bc,
    solve_system
)
from femlib.postprocess import (
    setup_plot_style,
    compute_stress_strain_field,
    plot_stress_field,
    plot_displacement_field
)

# Setup
setup_plot_style()

# Geometry
L, h, t = 10.0, 0.5, 0.1
mesh = create_rectangular_mesh(L, h, nx=40, ny=10)
material = LinearElastic(E=210e9, nu=0.3, thickness=t,
                         plane_type='stress')

# Solve
K = assemble_stiffness(mesh, material, 'full')
# Point load at tip (find node at x=L, y=h/2)
tip_node = np.argmin(
        (mesh["nodes"][:, 0] - L) ** 2
        + (mesh["nodes"][:, 1] - h / 2) ** 2
    )

load_spec = {
        "concentrated": [
            {"location": mesh["nodes"][tip_node], "force": [0.0, -10000.0]}
        ],
}
F = assemble_load_vector(mesh, load_spec)

fixed_nodes = identify_boundary_nodes(mesh, 'left')
bc = {'nodes': fixed_nodes, 'dofs': ['both'], 'values': [0.0]}
K_bc, F_bc, c_dofs, c_vals = apply_dirichlet_bc(K, F, bc)
u = solve_system(K_bc, F_bc, c_dofs, c_vals)

# Compute stresses
field_data = compute_stress_strain_field(mesh, u, material, 'full')

# Create output directory
output_dir = Path('outputs/figures')
output_dir.mkdir(parents=True, exist_ok=True)

# Plot von Mises stress
fig, ax, cbar = plot_stress_field(
    mesh, field_data, 'von_mises',
    levels=40, show_mesh=True
)
plt.savefig(output_dir / 'stress_von_mises.pdf')
print(f"Saved: {output_dir / 'stress_von_mises.pdf'}")

# Plot sigma_xx
fig, ax, cbar = plot_stress_field(
    mesh, field_data, 'sigma_xx',
    levels=40, show_mesh=True
)
plt.savefig(output_dir / 'stress_sigma_xx.pdf')
print(f"Saved: {output_dir / 'stress_sigma_xx.pdf'}")

# Plot vertical displacement
fig, ax, cbar = plot_displacement_field(
    mesh, u, 'v',
    levels=40, show_mesh=True
)
plt.savefig(output_dir / 'displacement_v.pdf')
print(f"Saved: {output_dir / 'displacement_v.pdf'}")

plt.show()

print("\nVisualization complete!")
