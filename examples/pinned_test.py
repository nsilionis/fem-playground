"""
Simply-supported deep beam under uniform transverse load.

Geometry  : 4 m (L) x 2 m (H) x 0.2 m (t), plane stress
Supports  : pin at both bottom corner nodes (ux = uy = 0)
Load      : uniform downward load q = 20 kN/m along top edge
Mesh      : 40 x 20 Q4 elements (full integration)
Material  : E = 100 MPa, nu = 0.30 (structural steel)
"""

import matplotlib.pyplot as plt

from femlib.mesh import create_rectangular_mesh
from femlib.materials import LinearElastic2D
from femlib.assembly import assemble_stiffness, assemble_load_vector
from femlib.solvers import apply_dirichlet_bc, solve_system
from femlib.postprocess import (
    setup_plot_style,
    compute_stress_strain_field,
    plot_displacement_field,
    plot_stress_field,
    plot_strain_field,
)

# ── Problem parameters ─────────────────────────────────────────────────────
Lx = 4.0        # beam length  [m]
Ly = 2.0        # beam height  [m]
t = 0.2        # out-of-plane thickness [m]
nx, ny = 40, 20

E = 1e8     # Young's modulus [Pa]
nu = 0.3        # Poisson ratio
q = 20e3       # load intensity [N/m], downward

# ── Mesh ───────────────────────────────────────────────────────────────────
mesh = create_rectangular_mesh(Lx, Ly, nx, ny)

# Node numbering: row-by-row, left-to-right, bottom-to-top
# Bottom row: nodes 0 ... nx  at y = 0
n_nodes_x = nx + 1            # 41
node_bot_left = 0           # (x=0,   y=0)
node_bot_right = n_nodes_x - 1   # (x=4,   y=0)
node_bot_mid = nx // 2     # (x=2,   y=0)  — midspan

# ── Material ───────────────────────────────────────────────────────────────
setup_plot_style()
material = LinearElastic2D(E=E, nu=nu, thickness=t, plane_type='stress')

# ── Global stiffness ───────────────────────────────────────────────────────
print("Assembling stiffness matrix ...")
K = assemble_stiffness(mesh, material, integration_order='full')

# ── Load vector ────────────────────────────────────────────────────────────
load_spec = {
    'distributed': [
        {'edge': 'top', 'load_density': [0.0, -q]}
    ]
}
F = assemble_load_vector(mesh, load_spec)

# ── Boundary conditions ────────────────────────────────────────────────────
bc = {
    'constraints': [
        {'node': node_bot_left,  'dof': 'both', 'value': 0.0},
        {'node': node_bot_right, 'dof': 'both', 'value': 0.0},
    ]
}
K_bc, F_bc, c_dofs, c_vals = apply_dirichlet_bc(K, F, bc)

# ── Solve ──────────────────────────────────────────────────────────────────
print("Solving linear system ...")
u = solve_system(K_bc, F_bc, c_dofs, c_vals)

# ── Midspan bottom-node displacement ───────────────────────────────────────
dof_ux = 2 * node_bot_mid
dof_uy = 2 * node_bot_mid + 1

print(f"\nMidspan bottom node  (node {node_bot_mid},  x = 2.0 m,  y = 0.0 m)")
print(f"  ux = {u[dof_ux]*1e2:+.6f} cm")
print(f"  uy = {u[dof_uy]*1e2:+.6f} cm")

# ── Post-processing ────────────────────────────────────────────────────────
print("\nComputing stress / strain fields ...")
field_data = compute_stress_strain_field(
    mesh, u, material, integration_order='full')

# --- displacement contours ---
fig1, _, _ = plot_displacement_field(mesh, u, component='u',
                                     title='Horizontal Displacement  u')
fig2, _, _ = plot_displacement_field(mesh, u, component='v',
                                     title='Vertical Displacement  v')
fig3, _, _ = plot_displacement_field(mesh, u, component='magnitude',
                                     title='Displacement Magnitude')

# --- stress contours ---
fig4, _, _ = plot_stress_field(mesh, field_data, component='sigma_xx',
                               title=r'Normal Stress  $\sigma_{xx}$')
fig5, _, _ = plot_stress_field(mesh, field_data, component='sigma_yy',
                               title=r'Normal Stress  $\sigma_{yy}$')
fig6, _, _ = plot_stress_field(mesh, field_data, component='tau_xy',
                               title=r'Shear Stress  $\tau_{xy}$')
fig7, _, _ = plot_stress_field(mesh, field_data, component='von_mises',
                               title='von Mises Stress')

# --- strain contours ---
fig8,  _, _ = plot_strain_field(mesh, field_data, component='epsilon_xx',
                                title=r'Normal Strain  $\varepsilon_{xx}$')
fig9,  _, _ = plot_strain_field(mesh, field_data, component='epsilon_yy',
                                title=r'Normal Strain  $\varepsilon_{yy}$')
fig10, _, _ = plot_strain_field(mesh, field_data, component='gamma_xy',
                                title=r'Shear Strain  $\gamma_{xy}$')

plt.show()
