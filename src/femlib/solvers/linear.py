"""
Linear system solvers for finite element analysis.

This module provides solution routines for linear systems arising
from FEM discretisation.
"""

import numpy as np
import scipy.linalg


def solve_system(K, F, constrained_dofs=None, prescribed_values=None):
    """
    Solve linear system K * u = F.

    If constrained DOFs are provided (from elimination method BC
    application), reconstructs the full displacement vector including
    prescribed values.

    Parameters
    ----------
    K : ndarray, shape (n_free_dofs, n_free_dofs)
        Stiffness matrix (possibly reduced after BC application). Must
        be symmetric positive definite.
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

    Raises
    ------
    numpy.linalg.LinAlgError
        If K is not positive definite, which happens when the boundary
        conditions do not prevent rigid-body motion.

    Examples
    --------
    >>> u = solve_system(K_reduced, F_reduced, c_dofs, c_vals)

    Notes
    -----
    Solves by Cholesky factorisation, through scipy.linalg.solve with
    assume_a='pos'. A scipy.linalg.LinAlgWarning is issued when K is
    ill-conditioned and the solution may be inaccurate.
    """
    # Cholesky factorisation: fails if K is not positive definite
    try:
        u_free = scipy.linalg.solve(K, F, assume_a='pos')
    except np.linalg.LinAlgError as err:
        raise np.linalg.LinAlgError(
            "Stiffness matrix is not positive definite. "
            "Check boundary conditions - system may be unconstrained."
        ) from err

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
