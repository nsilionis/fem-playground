"""
Verification script for structured mesh generator.

This script creates a simple rectangular mesh and visualises it
to verify correct node numbering and element connectivity.
"""

from femlib.mesh import (
    create_rectangular_mesh,
    mesh_statistics,
    visualise_mesh
)
from femlib.postprocess import setup_plot_style
import matplotlib.pyplot as plt
from pathlib import Path


def main():
    """Create and visualise a test mesh."""
    # Create a simple mesh for the cantilever problem
    print("Creating rectangular mesh...")
    mesh = create_rectangular_mesh(Lx=10.0, Ly=0.5, nx=10, ny=1)

    # Print statistics
    stats = mesh_statistics(mesh)
    print("\nMesh Statistics:")
    print(f"  Number of nodes: {stats['n_nodes']}")
    print(f"  Number of elements: {stats['n_elements']}")
    print(f"  Domain x: {stats['domain_x']}")
    print(f"  Domain y: {stats['domain_y']}")

    # Print boundary node information
    print("\nBoundary Nodes:")
    for edge, nodes in mesh['boundary_nodes'].items():
        print(f"  {edge}: {nodes}")

    # Visualise mesh
    # Create output directory if it doesn't exist
    output_dir = Path('outputs/figures')
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = "mesh_10x1.pdf"
    print("\nGenerating visualisation...")
    setup_plot_style()
    fig, ax = visualise_mesh(
        mesh,
        show_node_numbers=True,
        show_element_numbers=True
    )
    plt.savefig(
        output_dir / filename, dpi=300, bbox_inches='tight')
    print(
        f"Saved mesh visualisation to "
        f"'{output_dir / filename}'"
    )
    plt.show()


if __name__ == '__main__':

    main()
