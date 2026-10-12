"""
Linear system solvers for finite element analysis.

This module provides solution routines for linear systems arising
from FEM discretisation.
"""

import warnings

import numpy as np
import scipy.linalg
import scipy.sparse as sp
from scipy.sparse.linalg import splu


def solve_system(K, F, constrained_dofs=None, prescribed_values=None,
                 dof_names=('ux', 'uy'), max_ratio=1e13):
    """
    Solve linear system K * u = F.

    If constrained DOFs are provided (from elimination method BC
    application), reconstructs the full displacement vector including
    prescribed values.

    Parameters
    ----------
    K : numpy.ndarray or scipy.sparse array,
        shape (n_free_dofs, n_free_dofs)
        Stiffness matrix (possibly reduced after BC application). Must
        be symmetric positive definite.
    F : ndarray, shape (n_free_dofs,)
        Force vector (possibly reduced after BC application).
    constrained_dofs : ndarray, optional
        Indices of constrained DOFs (for reconstruction).
    prescribed_values : ndarray, optional
        Prescribed displacement values at constrained DOFs.
    dof_names : tuple of str, optional
        DOF names per node, used to name the location of a failed
        pivot. Default is ('ux', 'uy').
    max_ratio : float, optional
        Sparse K only: warn when a matrix diagonal term divided by the
        corresponding factor diagonal term exceeds this value. Default
        is 1e13.

    Returns
    -------
    u : ndarray, shape (n_total_dofs,)
        Displacement vector (full size, including constrained DOFs).

    Raises
    ------
    numpy.linalg.LinAlgError
        If K is not positive definite, which happens when the boundary
        conditions do not prevent rigid-body motion. For sparse K, the
        message names the node and DOF of each non-positive pivot.

    Examples
    --------
    >>> u = solve_system(K_reduced, F_reduced, c_dofs, c_vals)

    Notes
    -----
    Dense K is solved by Cholesky factorisation, through
    scipy.linalg.solve with assume_a='pos'; SciPy issues a
    scipy.linalg.LinAlgWarning when K is ill-conditioned.

    Sparse K is solved by SuperLU with a symmetric ordering and
    diagonal pivots, so that the U diagonal holds the elimination
    pivots; K is positive definite exactly when all are positive. A
    scipy.linalg.LinAlgWarning is issued when the largest diagonal
    ratio exceeds max_ratio, following Nastran's MAXRATIO. Close to
    singularity, double precision cannot tell a mechanism from a very
    ill-conditioned structure, so the warning is a guide and not a
    proof.
    """
    n_free = K.shape[0]
    if constrained_dofs is None or len(constrained_dofs) == 0:
        free_dofs = np.arange(n_free)
    else:
        n_total = n_free + len(constrained_dofs)
        free_dofs = np.setdiff1d(np.arange(n_total), constrained_dofs)

    if sp.issparse(K):
        u_free = _solve_sparse(K, F, free_dofs, dof_names, max_ratio)
    else:
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
    u_full = np.zeros(n_total)

    # Fill in free DOF values
    u_full[free_dofs] = u_free

    # Fill in prescribed values at constrained DOFs
    u_full[constrained_dofs] = prescribed_values

    return u_full


def _solve_sparse(K, F, free_dofs, dof_names, max_ratio):
    """Solve a sparse system through the checked factorisation."""
    lu = _factorise_sparse(K, free_dofs, dof_names, max_ratio)
    return lu.solve(np.asarray(F, dtype=float))


def _factorise_sparse(K, free_dofs, dof_names, max_ratio):
    """LU with symmetric ordering and diagonal pivots; pivots checked."""
    A = sp.csc_array(K)
    try:
        lu = splu(A, permc_spec='MMD_AT_PLUS_A', diag_pivot_thresh=0.0,
                  options={'SymmetricMode': True})
    except RuntimeError as err:
        raise np.linalg.LinAlgError(
            "Stiffness matrix is singular: a zero pivot was met. "
            "Check boundary conditions - system may be unconstrained."
        ) from err
    if not np.array_equal(lu.perm_r, lu.perm_c):
        raise RuntimeError(
            "SuperLU did not keep a symmetric ordering; "
            "the pivot check would not be valid"
        )

    order = np.argsort(lu.perm_c)          # pivot k belongs to row order[k]
    pivots = lu.U.diagonal()
    ratios = A.diagonal()[order] / pivots  # a_kk / d_k, as Nastran reports

    def where(k):
        dof = free_dofs[order[k]]
        n = len(dof_names)
        return f"node {dof // n}, {dof_names[dof % n]}"

    bad = np.flatnonzero(pivots <= 0)
    if bad.size:
        listed = "; ".join(where(k) for k in bad[:10])
        more = f" and {bad.size - 10} more" if bad.size > 10 else ""
        raise np.linalg.LinAlgError(
            f"Stiffness matrix is not positive definite: {bad.size} "
            f"non-positive pivot(s), at {listed}{more}. "
            "Check boundary conditions - system may be unconstrained."
        )

    worst = np.argmax(ratios)
    if ratios[worst] > max_ratio:
        warnings.warn(
            f"Stiffness matrix is nearly singular: diagonal ratio "
            f"{ratios[worst]:.3e} exceeds max_ratio={max_ratio:.1e} at "
            f"{where(worst)}. The solution may be inaccurate.",
            scipy.linalg.LinAlgWarning,
            stacklevel=4,
        )
    return lu


def compute_reaction_forces(K_original, u, constrained_dofs):
    """
    Compute reaction forces at constrained DOFs.

    Reaction forces are computed from the original (unreduced)
    stiffness matrix and the full displacement vector.

    Parameters
    ----------
    K_original : numpy.ndarray or scipy.sparse array,
        shape (n_dofs, n_dofs)
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
