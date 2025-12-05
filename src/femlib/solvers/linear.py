"""
Linear system solvers for finite element analysis.

This module provides solution routines for linear systems arising
from FEM discretisation.
"""

import numpy as np


def solve_system(K, F, constrained_dofs=None, prescribed_values=None):
    """
    Solve linear system K * u = F.

    If constrained DOFs are provided (from elimination method BC
    application), reconstructs the full displacement vector including
    prescribed values.

    Parameters
    ----------
    K : ndarray, shape (n_free_dofs, n_free_dofs)
        Stiffness matrix (possibly reduced after BC application).
    F : ndarray, shape (n_free_dofs,)
        Force vector (possibly reduced after BC application).
    constrained_dofs : ndarray, optional
        Indices of constrained DOFs (for reconstruction).
    prescribed_values : ndarray, optional
        Prescribed displacement values at constrained DOFs.

    Returns
    -------
    u : ndarray, shape (n_total_dofs,)
        Displacement vector (full size, including constrained DOFs).

    Examples
    --------
    >>> u = solve_system(K_reduced, F_reduced, c_dofs, c_vals)

    Notes
    -----
    Uses numpy.linalg.solve which employs LU decomposition.
    For large sparse systems, consider iterative solvers.
    """
    # Check for singular system
    if np.abs(np.linalg.det(K)) < 1e-12:
        raise np.linalg.LinAlgError(
            "Stiffness matrix is singular or nearly singular. "
            "Check boundary conditions - system may be unconstrained."
        )

    # Solve reduced system
    u_free = np.linalg.solve(K, F)

    # If no constrained DOFs, return directly
    if constrained_dofs is None or len(constrained_dofs) == 0:
        return u_free

    # Reconstruct full displacement vector
    n_free = len(u_free)
    n_constrained = len(constrained_dofs)
    n_total = n_free + n_constrained

    u_full = np.zeros(n_total)

    # Identify free DOF indices
    all_dofs = np.arange(n_total)
    free_dofs = np.setdiff1d(all_dofs, constrained_dofs)

    # Fill in free DOF values
    u_full[free_dofs] = u_free

    # Fill in prescribed values at constrained DOFs
    u_full[constrained_dofs] = prescribed_values

    return u_full


def compute_reaction_forces(K_original, u, constrained_dofs):
    """
    Compute reaction forces at constrained DOFs.

    Reaction forces are computed from the original (unreduced)
    stiffness matrix and the full displacement vector.

    Parameters
    ----------
    K_original : ndarray, shape (n_dofs, n_dofs)
        Original global stiffness matrix (before BC application).
    u : ndarray, shape (n_dofs,)
        Full displacement vector (including prescribed displacements).
    constrained_dofs : ndarray
        Indices of constrained DOFs.

    Returns
    -------
    reactions : dict
        Dictionary mapping constrained DOF indices to reaction forces.

    Examples
    --------
    >>> reactions = compute_reaction_forces(K, u, constrained_dofs)
    >>> print(f"Reaction at DOF 0: {reactions[0]:.2f} N")

    Notes
    -----
    Reaction forces represent the forces required to maintain the
    prescribed displacements (support reactions).
    """
    # Compute all internal forces: F_internal = K * u
    F_internal = K_original @ u

    # Reaction forces at constrained DOFs
    reactions = {}
    for dof in constrained_dofs:
        reactions[dof] = F_internal[dof]

    return reactions
