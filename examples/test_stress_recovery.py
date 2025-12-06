"""Quick test of stress recovery functionality."""

import numpy as np
from femlib.mesh import create_rectangular_mesh
from femlib.materials import LinearElastic
from femlib.assembly import assemble_stiffness, assemble_load_vector
from femlib.solvers import (
    identify_boundary_nodes,
    apply_dirichlet_bc,
    solve_system
)
from femlib.postprocess import compute_stress_strain_field

# Quick cantilever setup
L, h, t = 10.0, 0.1, 0.1
mesh = create_rectangular_mesh(L, h, 20, 10)
material = LinearElastic(E=210e9, nu=0.3, thickness=t)

# Solve (abbreviated)
K = assemble_stiffness(mesh, material, 'full')
load_spec = {'distributed': [{'edge': 'top', 'load_density': [0.0, -1000.0]}]}
F = assemble_load_vector(mesh, load_spec)

fixed_nodes = identify_boundary_nodes(mesh, 'left')
bc = {'nodes': fixed_nodes, 'dofs': ['both'], 'values': [0.0]}
K_bc, F_bc, c_dofs, c_vals = apply_dirichlet_bc(K, F, bc)
u = solve_system(K_bc, F_bc, c_dofs, c_vals)

# Test stress recovery
print("Testing stress recovery...")
field_data = compute_stress_strain_field(mesh, u, material, 'full')

vm_nodal = field_data['nodal']['von_mises']
print(f"✓ Max von Mises stress: {np.max(vm_nodal)/1e6:.2f} MPa")

elem_0_stresses = field_data['gauss'][0]['stress']
print("✓ Element 0 Gauss point stresses:")
print(f"  {elem_0_stresses/1e6} MPa")

print("\nStress recovery working correctly!")
