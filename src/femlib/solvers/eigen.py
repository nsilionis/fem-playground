"""
Generalised eigenvalue solution for free vibration.

This module solves K phi = omega^2 M phi on the free DOFs and returns
natural frequencies and mass-normalised mode shapes.
"""

import numpy as np
import scipy.linalg
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, eigsh

from .linear import _factorise_sparse

ZERO_TOL = 1e-10
SIGNIFICANT = 1e-6


def solve_eigen(K, M, constrained_dofs=None, n_modes=None,
                dof_names=('ux', 'uy'), max_ratio=1e13):
    """
    Natural frequencies and mass-normalised mode shapes.

    Solves K phi = lambda M phi, with lambda = omega^2, on the DOFs not
    in constrained_dofs, which are held at zero.

    Parameters
    ----------
    K : numpy.ndarray or scipy.sparse array, shape (n_dofs, n_dofs)
        Global stiffness matrix, before boundary conditions.
    M : numpy.ndarray or scipy.sparse array, shape (n_dofs, n_dofs)
        Global mass matrix, of the same kind as K.
    constrained_dofs : array_like of int, optional
        DOFs held at zero, e.g. as returned by apply_dirichlet_bc.
    n_modes : int, optional
        Number of lowest modes to return. Required for sparse K; None
        returns all modes of a dense problem.
    dof_names : tuple of str, optional
        DOF names per node, used in sparse factorisation errors.
        Default is ('ux', 'uy').
    max_ratio : float, optional
        Sparse K only: diagonal ratio above which a warning is issued,
        as in solve_system. Default is 1e13.

    Returns
    -------
    frequencies : numpy.ndarray, shape (n_modes,)
        Natural frequencies [Hz], ascending.
    modes : numpy.ndarray, shape (n_dofs, n_modes)
        Mode shapes, one per column, with phi^T M phi = 1 and zeros at
        the constrained DOFs.

    Raises
    ------
    ValueError
        If K and M are not of the same kind, or n_modes is missing or
        too large.
    numpy.linalg.LinAlgError
        If an eigenvalue is clearly negative, or a sparse K has a zero
        or negative pivot.

    Notes
    -----
    The sign of each mode is fixed so that its first entry, in DOF
    order, larger in magnitude than 1e-6 times its largest entry is
    positive. Negative eigenvalues smaller in magnitude than 1e-10
    times max_k K_kk / M_kk are rounding of zero, as for rigid-body
    modes, and are set to zero.

    Free-free structures need dense matrices. On the sparse path they
    raise a numpy.linalg.LinAlgError or issue a
    scipy.linalg.LinAlgWarning, depending on rounding.
    """
    if sp.issparse(K) != sp.issparse(M):
        raise ValueError("K and M must both be dense or both be sparse")
    n_dofs = K.shape[0]
    constrained = np.asarray(
        [] if constrained_dofs is None else constrained_dofs, dtype=int
    )
    free = np.setdiff1d(np.arange(n_dofs), constrained)
    if n_modes is not None and not 1 <= n_modes <= len(free):
        raise ValueError(
            f"n_modes must be between 1 and the {len(free)} free DOFs, "
            f"got {n_modes}"
        )

    if sp.issparse(K):
        lam, phi, scale = _eigen_sparse(K, M, free, n_modes,
                                        dof_names, max_ratio)
    else:
        lam, phi, scale = _eigen_dense(K, M, free, n_modes)

    if np.any(lam < -ZERO_TOL * scale):
        raise np.linalg.LinAlgError(
            "Stiffness matrix has a clearly negative eigenvalue; it is "
            "not positive semi-definite"
        )
    lam = np.where(lam < 0.0, 0.0, lam)

    modes = np.zeros((n_dofs, phi.shape[1]))
    modes[free] = _fix_signs(phi)
    return np.sqrt(lam) / (2.0 * np.pi), modes


def _eigen_dense(K, M, free, n_modes):
    """All or the lowest n_modes eigenpairs of the dense free system."""
    K_f = K[np.ix_(free, free)]
    M_f = M[np.ix_(free, free)]
    subset = None if n_modes is None else [0, n_modes - 1]
    lam, phi = scipy.linalg.eigh(K_f, M_f, subset_by_index=subset)
    return lam, phi, np.max(K_f.diagonal() / M_f.diagonal())


def _eigen_sparse(K, M, free, n_modes, dof_names, max_ratio):
    """Lowest n_modes eigenpairs by shift-invert at zero."""
    if n_modes is None:
        raise ValueError("n_modes is required for sparse matrices")
    if n_modes >= len(free):
        raise ValueError(
            f"n_modes must be less than the {len(free)} free DOFs for "
            "sparse matrices; use dense matrices"
        )
    K_f = sp.csr_array(K)[free][:, free].tocsc()
    M_f = sp.csr_array(M)[free][:, free].tocsc()
    lu = _factorise_sparse(K_f, free, dof_names, max_ratio)
    OPinv = LinearOperator(K_f.shape, matvec=lu.solve, dtype=float)
    lam, phi = eigsh(K_f, k=n_modes, M=M_f, sigma=0.0, which='LM',
                     OPinv=OPinv)
    order = np.argsort(lam)
    lam, phi = lam[order], phi[:, order]
    phi = phi / np.sqrt(np.sum(phi * (M_f @ phi), axis=0))
    return lam, phi, np.max(K_f.diagonal() / M_f.diagonal())


def _fix_signs(phi):
    """Flip modes so their first significant entry is positive."""
    phi = phi.copy()
    for j in range(phi.shape[1]):
        column = phi[:, j]
        significant = np.abs(column) > SIGNIFICANT * np.max(np.abs(column))
        if column[np.flatnonzero(significant)[0]] < 0:
            phi[:, j] = -column
    return phi
