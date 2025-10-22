import sys, os
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import List, Tuple, Callable, Any, Dict

from methods.PolyCG.polycg.transforms.transform_marginals import send_to_back_permutation
from methods.midstep_composites import midstep_composition_transformation, midstep_groundstate, midstep_groundstate_se3
from methods.midstep_composites import midstep_composition_transformation_correction, midstep_composition_transformation_correction_old
from methods.read_nuc_data import read_nucleosome_triads, GenStiffness
from methods.free_energy import calculate_midstep_triads, midstep_excess_vals
from methods.PolyCG.polycg.SO3 import so3

def get_midstep_locations(left_open: int, right_open: int, base_locations = None, sort = True):
    if base_locations is None:
        MIDSTEP_LOCATIONS = [
            2, 6, 14, 17, 24, 29, 
            34, 38, 45, 49, 55, 59, 
            65, 69, 76, 80, 86, 90, 
            96, 100, 107, 111, 116, 121, 
            128, 131, 139, 143
        ]
    else:
        MIDSTEP_LOCATIONS = base_locations
    if left_open + right_open > len(MIDSTEP_LOCATIONS):
        return []
    
    locs = MIDSTEP_LOCATIONS[left_open:len(MIDSTEP_LOCATIONS)-right_open]
    if sort:
        locs = sorted(list(set(locs)))
    return locs

def Hinverse(Psi):
    psih = so3.hat_map(Psi)
    psihsq = psih @ psih
    Hinv = np.eye(3)
    Hinv += 0.5* psih
    Hinv += 1./12 * psihsq
    Hinv -= 1./720 * psihsq @ psihsq
    Hinv += 1./30240 * psihsq @ psihsq @ psihsq
    return Hinv
    
def coordinate_transformation(muk0s,sks):
    B = np.zeros((len(sks)*6,len(muk0s)*6))
    Pbar = np.zeros(len(sks)*6)
    for k in range(len(sks)):
        sig0 = np.linalg.inv(muk0s[k]) @ muk0s[k+1]
        Sig  = sig0[:3,:3]
        sig  = sig0[:3,3]
        Sk   = sks[k,:3,:3]
        sk   = sks[k,:3,3]
        
        Psi  = so3.rotmat2euler(Sk.T @ Sig)
        Hi   = Hinverse(Psi)
        Bkm = np.zeros((6,6))
        Bkp = np.zeros((6,6))
        Bkm[:3,:3] = -Hi @ Sig.T
        Bkm[3:,:3] = Sk.T @ so3.hat_map(sig)
        Bkm[3:,3:] = -Sk.T
        Bkp[:3,:3] = Hi
        Bkp[3:,3:] = Sk.T @ Sig
        
        B[6*k:6*(k+1),6*k:6*(k+1)]      = Bkm
        B[6*k:6*(k+1),6*(k+1):6*(k+2)]  = Bkp
        
        Pbar[k*6:k*6+3]   = Psi
        Pbar[k*6+3:k*6+6] = Sk.T @ (sig-sk)
    return B, Pbar

def coordinate_transformation_correction(muk0s,sks,Z_delta_ref):
    
    if len(Z_delta_ref.shape) < 2:
        Z_delta_ref = Z_delta_ref.reshape((len(Z_delta_ref)//6,6))
    
    B = np.zeros((len(sks)*6,len(muk0s)*6))
    Pbar = np.zeros(len(sks)*6)
    for k in range(len(sks)):
        sig0 = np.linalg.inv(muk0s[k]) @ muk0s[k+1]
        SIG  = sig0[:3,:3]
        sig  = sig0[:3,3]
        Sk   = sks[k,:3,:3]
        sk   = sks[k,:3,3]
        
        Psi  = so3.rotmat2euler(Sk.T @ SIG)
        Hi   = Hinverse(Psi)
        
        Z0k = so3.euler2rotmat(Z_delta_ref[k,:3])
        htheta0 = so3.hat_map(Z_delta_ref[k,:3])
        
        Bkm = np.zeros((6,6))
        Bkp = np.zeros((6,6))
        
        Bkm[:3,:3] = -Hi @ SIG.T
        Bkm[3:,:3] = Sk.T @ so3.hat_map(sig)
        Bkm[3:,3:] = -Sk.T @ Z0k.T
        
        Bkp[:3,:3] = Hi
        Bkp[3:,3:] = Sk.T @ Z0k.T @ SIG
        
        B[6*k:6*(k+1),6*k:6*(k+1)]      = Bkm
        B[6*k:6*(k+1),6*(k+1):6*(k+2)]  = Bkp
        
        Pbar[k*6:k*6+3]   = Psi
        Pbar[k*6+3:k*6+6] = Sk.T @ ( (Z0k.T + htheta0) @ sig-sk)
    return B, Pbar


###################################################################
###################################################################
###################################################################

import scipy.linalg   as la
import scipy.sparse   as sp
import scipy.sparse.linalg as spla

# def apply_Ainv_M_AinvT(A, M, is_sparse_A=False):
#     """
#     Return A⁻ᵀ · M · A⁻¹  without forming A⁻¹.

#     Parameters
#     ----------
#     A : (N,N) array or sparse
#     M : (N,N) array or sparse   (will be treated as dense if sparse)
#     is_sparse_A : bool
#         Pass True if `A` is sparse; we use SuperLU instead of LAPACK.

#     Notes
#     -----
#     *Works with your two cases*  
#     • `A` ~10 % dense  →  CSR / CSC  
#     • `M` dense **or** 6×6 block-diag BSR  
#     """
#     # ---------- ensure the right formats ----------
#     if sp.issparse(M):
#         M = M.toarray()        # block-diag BSR → dense is cheap (∼6 MB at 876²)

#     if is_sparse_A or sp.issparse(A):
#         # SuperLU factor (done once)
#         lu = spla.splu(A.tocsc() if sp.issparse(A) else sp.csc_matrix(A))

#         X  = lu.solve(M)       # A⁻¹ · M
#         Y  = lu.solve(X.T)     # A⁻¹ · (A⁻¹ M)ᵀ  =  A⁻ᵀ Mᵀ
#         return Y.T             # (A⁻ᵀ M A⁻¹)
#     else:
#         # Dense LU factor (also done once)
#         lu, piv = la.lu_factor(A, check_finite=False)

#         X  = la.lu_solve((lu, piv), M, trans=0, check_finite=False)  # A⁻¹ M
#         Y  = la.lu_solve((lu, piv), X.T, trans=0, check_finite=False)
#         return Y.T

# def apply_Ainv_M_AinvT(A, M, is_sparse_A=False):
#     """
#     Return A^{-T} · M · A^{-1}  without forming A^{-1}.

#     Works for dense or sparse A (CSR/CSC) and dense or sparse M.
#     """
#     if sp.issparse(M):
#         M = M.toarray()                    # cheap (876×876 ≈ 6 MB)

#     # --- sparse path -------------------------------------------------- #
#     if is_sparse_A or sp.issparse(A):
#         lu = spla.splu(A.tocsc() if sp.issparse(A) else sp.csc_matrix(A))

#         X  = lu.solve(M,               trans='T')   # A^{-T} · M
#         Y  = lu.solve(X,               trans='N')   # (A^{-T} M) · A^{-1}
#         return Y
#     # --- dense path --------------------------------------------------- #
#     else:
#         lu, piv = la.lu_factor(A, check_finite=False)

#         X  = la.lu_solve((lu, piv), M, trans=1, check_finite=False)  # A^{-T} · M
#         Y  = la.lu_solve((lu, piv), X, trans=0, check_finite=False)  # ... · A^{-1}
#         return Y


def apply_Ainv_M_AinvT(A, M, rcond=1e-12):
    """
    Compute A^{-T} · M · A^{-1} without forming A^{-1}.

    Robust order:
    1. sparse LU  (if A is sparse & factor succeeds)
    2. dense  LU  (if factor succeeds)
    3. SVD pseudo-inverse with cutoff `rcond`
    """
    # ---- ensure M is dense -------------------------------------- #
    if sp.issparse(M):
        M = M.toarray()

    # ---- 1) sparse LU ------------------------------------------ #
    if sp.issparse(A):
        try:
            lu = spla.splu(A.tocsc(), diag_pivot_thresh=0.0)
            X  = lu.solve(M, trans='T')        # A^{-T} · M
            Y  = lu.solve(X, trans='N')        # … · A^{-1}
            return Y
        except RuntimeError:
            A = A.toarray()                    # fall through to dense

    # ---- 2) dense LU ------------------------------------------- #
    try:
        lu, piv = la.lu_factor(A, check_finite=False)
        X       = la.lu_solve((lu, piv), M, trans=1, check_finite=False)
        Y       = la.lu_solve((lu, piv), X, trans=0, check_finite=False)
        return Y
    except la.LinAlgError:
        pass                                    # fall through to SVD

    # ---- 3) SVD pseudo-inverse --------------------------------- #
    # A = U Σ Vᵀ   ⇒   A^{-1} = V Σ^{-1} Uᵀ   (with cutoff)
    U, s, VT = la.svd(A, check_finite=False)
    tol      = rcond * s.max()
    s_inv    = np.where(s > tol, 1.0 / s, 0.0)
    Ainv     = (VT.T * s_inv) @ U.T



def spsolve_csc(A, b):
    if sp.issparse(A) and A.format != 'csc':
        A = A.tocsc()
    return spla.spsolve(A, b)

def spsolve_csc(A, b):
    if A.format != 'csc': A = A.tocsc()
    return spla.spsolve(A, b)

def as_dense(A):
    return A.toarray() if sp.issparse(A) else np.asarray(A)

def logdet_sym(A, jitters=(0.,1e-12,1e-10,1e-8,1e-6,1e-4)):
    Ad = A.toarray() if sp.issparse(A) else np.asarray(A)
    for eps in jitters:
        try:
            L = la.cholesky(Ad + eps*np.eye(Ad.shape[0])) if eps else la.cholesky(Ad)
            return 1., 2.*np.log(np.diag(L)).sum()
        except la.LinAlgError:
            pass
    return np.linalg.slogdet(Ad)                 # (sign, log|det|)


def binding_model_free_energy_optimized(
    free_gs: np.ndarray,
    free_M: np.ndarray,    
    nuc_mu0_full: np.ndarray,
    nuc_K_full: np.ndarray,
    left_open: int = 0,
    right_open: int = 0,
    use_correction: bool = True,
    is_block_diag: bool = False,
) -> np.ndarray:

    midstep_constraint_locations = get_midstep_locations(left_open, right_open)
    if len(midstep_constraint_locations) <= 1:
        n = len(free_M)
        F_pi = -0.5*n * np.log(2*np.pi)
        # matrix term
        logdet_sign, logdet = np.linalg.slogdet(free_M)
        F_mat = 0.5*logdet
        F = F_mat + F_pi  
        Fdict = {
            'F': F,
            'F_entropy' : F,
            'F_enthalpy': 0,
            'F_jacob'   : 0,
            'F_freedna'    : F
        }
        return Fdict
    
    sks = midstep_groundstate_se3(free_gs,midstep_constraint_locations)
    
    # select midstep triads and their stiffness 
    nuc_mu0 = nuc_mu0_full[left_open:len(nuc_mu0_full)-right_open]
    nuc_K = nuc_K_full[6*left_open:len(nuc_K_full)-6*right_open,6*left_open:len(nuc_K_full)-6*right_open]
    
    # find composite transformation
    transform, replaced_ids = midstep_composition_transformation(
        free_gs,
        midstep_constraint_locations
    )
        
    # # transform stiffness matrix
    # inv_transform = np.linalg.inv(transform)
    # M_transformed = inv_transform.T @ free_M @ inv_transform
        
    # OPTIONAL: convert to sparse on-the-fly if only ~10 % nnz
    if transform.size and np.count_nonzero(transform) / transform.size < 0.15:
        transform = sp.csr_matrix(transform)

    # M_transformed = apply_Ainv_M_AinvT(transform,
    #                                 free_M,
    #                                 is_sparse_A=sp.issparse(transform))
    
    M_transformed = apply_Ainv_M_AinvT(transform, free_M)
    
    # # rearrange stiffness matrix
    # full_replaced_ids = list()
    # for i in range(len(replaced_ids)):
    #     full_replaced_ids += [6*replaced_ids[i]+j for j in range(6)]
     
    # P = send_to_back_permutation(len(free_M),full_replaced_ids)
    # M_rearranged = P @ M_transformed @ P.T
    
    # # select M and R submatrices
    # N  = len(M_rearranged)
    # NC = len(full_replaced_ids)
    # NF = N-NC
    
    # M_R = M_rearranged[:NF,:NF]
    # M_M = M_rearranged[NF:,NF:]
    # M_RM = M_rearranged[:NF,NF:]
    
    # # Calculate M block marginal
    # M_Mp = M_M - M_RM.T @ np.linalg.inv(M_R) @ M_RM
    # M_Mp = 0.5*(M_Mp+M_Mp.T)
    
    # ---------------------------------------------------------------
    # ❶  Build index lists – no Python loop, no permutation matrix
    # ---------------------------------------------------------------
    nM = free_M.shape[0]

    full_ids  = (6*np.repeat(replaced_ids, 6) + np.tile(np.arange(6), len(replaced_ids))).astype(int)
    # idx_free  = np.setdiff1d(np.arange(nM), full_ids, assume_unique=True)   # unconstrained dofs
    mask      = np.ones(nM, dtype=bool)
    mask[full_ids] = False
    idx_free  = np.flatnonzero(mask)           # preserves 0…N-1 order

    # sizes
    NF = idx_free.size
    NC = full_ids.size

    # ---------------------------------------------------------------
    # ❷  Extract the four blocks directly by advanced indexing
    #     Works for both dense and sparse `M_transformed`
    # ---------------------------------------------------------------
    M_R  = M_transformed[np.ix_(idx_free,  idx_free )]      # (NF × NF)
    M_M  = M_transformed[np.ix_(full_ids, full_ids)]        # (NC × NC)
    M_RM = M_transformed[np.ix_(idx_free,  full_ids)]       # (NF × NC)

    # ---------------------------------------------------------------
    # ❸  Block marginal   M_M' = M_M − M_RMᵀ · M_R⁻¹ · M_RM
    #     • one symmetric solve instead of an explicit inverse
    #     • stays dense because we need the result in BLAS later
    # ---------------------------------------------------------------
    if sp.issparse(M_R):
        M_R  = M_R.toarray()
    if sp.issparse(M_RM):
        M_RM = M_RM.toarray()
    if sp.issparse(M_M):
        M_M  = M_M.toarray()

    sol      = la.solve(M_R, M_RM, assume_a='sym', check_finite=False)  # M_R⁻¹ · M_RM
    M_Mp     = M_M - M_RM.T @ sol
    M_Mp     = 0.5 * (M_Mp + M_Mp.T)     # force symmetry
    
    # ------------------------------------------------------------------
    # ❺  Binding–model core  (fast, inverse–free)
    # ------------------------------------------------------------------

    # ---------- helpers used several times ---------------------------------
    def sym_solve(A, B):
        """Solve A x = B for a symmetric (possibly SPD) A without forming A⁻¹."""
        try:                                 # fast path – Cholesky
            return la.cho_solve(la.cho_factor(A, check_finite=False), B)
        except la.LinAlgError:               # fall back – LU
            return la.solve(A, B, assume_a='sym', check_finite=False)

    def sparsify(mat, thresh=0.20):
        """CSR-convert if density < thresh."""
        if not sp.issparse(mat) and np.count_nonzero(mat) / mat.size < thresh:
            return sp.csr_matrix(mat)
        return mat

    # ---------- first incidence matrix / ground state ----------------------
    B, Pbar = coordinate_transformation(nuc_mu0, sks)
    B       = sparsify(B)

    Kcomb   = nuc_K + (B.T @ M_Mp @ B)
    alpha   = -sym_solve(Kcomb, B.T @ M_Mp @ Pbar)

    # ---------- corrected incidence matrix ---------------------------------
    B, Pbar = coordinate_transformation_correction(nuc_mu0, sks, alpha)
    B       = sparsify(B)

    Kcomb   = nuc_K + (B.T @ M_Mp @ B)
    alpha   = -sym_solve(Kcomb, B.T @ M_Mp @ Pbar)

    Y_C     = Pbar + B @ alpha

    # ---------- enthalpy ---------------------------------------------------
    inner        = sym_solve(Kcomb, B.T @ M_Mp)
    F_enthalpy   = 0.5 * Pbar.T @ (M_Mp - M_Mp @ B @ inner) @ Pbar

    # ---------- gamma ------------------------------------------------------
    gamma = -sym_solve(M_R, M_RM @ Y_C)

    # ------------------------------------------------------------------
    # ❻  Optional rigid-body-shift correction (kept)
    # ------------------------------------------------------------------
    if use_correction:
        # --- project (γ, Y_C) back without explicit permutation ----------
        order        = np.hstack((idx_free, full_ids))       # from earlier step
        gs_perm      = np.concatenate((gamma, Y_C))
        gs_transf    = np.empty_like(gs_perm)
        gs_transf[order] = gs_perm                           # Pᵀ · (γ, Y_C)

        # --- back-transform to SE(3) coordinates ------------------------
        if sp.issparse(transform):
            gs = spsolve_csc(transform, gs_transf)
        else:
            gs = la.solve(transform, gs_transf, assume_a='gen')
        gs = gs.reshape(-1, 6)

        # --- corrected composite transformation ------------------------
        transform, replaced_ids, shift = (
            midstep_composition_transformation_correction(
                free_gs, midstep_constraint_locations, gs)
        )
        transform = sparsify(transform)

        # --- compute  M̃ = A⁻ᵀ M A⁻¹  without explicit inverse ----------
        M_transformed = apply_Ainv_M_AinvT(transform, free_M)

        # --- rebuild block slices quickly ------------------------------
        full_ids  = (6*np.repeat(replaced_ids, 6) +
                    np.tile(np.arange(6), len(replaced_ids))).astype(int)
        idx_free  = np.setdiff1d(np.arange(nM), full_ids, assume_unique=True)

        M_R  = M_transformed[np.ix_(idx_free,  idx_free )]
        M_M  = M_transformed[np.ix_(full_ids,  full_ids)]
        M_RM = M_transformed[np.ix_(idx_free,  full_ids)]

        if sp.issparse(M_R):  M_R  = M_R.toarray()
        if sp.issparse(M_RM): M_RM = M_RM.toarray()
        if sp.issparse(M_M):  M_M  = M_M.toarray()

        sol      = sym_solve(M_R, M_RM)
        M_Mp     = M_M - M_RM.T @ sol
        M_Mp     = 0.5*(M_Mp + M_Mp.T)

        # --- binding model again with shift ----------------------------
        B, Pbar  = coordinate_transformation(nuc_mu0, sks)
        B        = sparsify(B)
        Kcomb    = nuc_K + (B.T @ M_Mp @ B)
        alpha    = -sym_solve(Kcomb, B.T @ M_Mp @ Pbar)

        B, Pbar  = coordinate_transformation_correction(nuc_mu0, sks, alpha)
        B        = sparsify(B)

        Kcomb    = nuc_K + (B.T @ M_Mp @ B)
        Pbar    -= shift

        alpha    = -sym_solve(Kcomb, B.T @ M_Mp @ Pbar)
        Y_C      = Pbar + B @ alpha
        gamma    = -sym_solve(M_R, M_RM @ Y_C)

        inner        = sym_solve(Kcomb, B.T @ M_Mp)
        F_enthalpy   = 0.5 * Pbar.T @ (M_Mp - M_Mp @ B @ inner) @ Pbar

    # ##############################################
    # # Binding Model
    # ##############################################
    
    # # Calculate Incidence Matrix
    # B, Pbar = coordinate_transformation(nuc_mu0,sks)  
    # Kcomb = nuc_K + B.T @ M_Mp @ B
    # # calculate ground state
    # alpha = -np.linalg.inv(Kcomb) @ B.T @ M_Mp @ Pbar
    
    # B, Pbar = coordinate_transformation_correction(nuc_mu0,sks,alpha)
    # Kcomb = nuc_K + B.T @ M_Mp @ B
    # # calculate ground state
    # alpha = -np.linalg.inv(Kcomb) @ B.T @ M_Mp @ Pbar
    
    # Y_C = Pbar + B @ alpha
    # F_enthalpy = 0.5* Pbar.T @ ( M_Mp - M_Mp @ B @ np.linalg.inv(Kcomb) @ B.T @ M_Mp ) @ Pbar
    # # print(f'F_enthalpy = {F_enthalpy}')
    
    # gamma = -np.linalg.inv(M_R) @ M_RM @ Y_C
    
    # if use_correction:
    
    #     gs_transf_perm = np.concatenate((gamma,Y_C))
    #     gs_transf = P.T @ gs_transf_perm
    #     gs = inv_transform @ gs_transf

    #     gs = gs.reshape((len(gs)//6,6))
    #     # find composite transformation
    #     transform, replaced_ids, shift = midstep_composition_transformation_correction(
    #         free_gs,
    #         midstep_constraint_locations,
    #         gs
    #     )
        
    #     # transform stiffness matrix
    #     inv_transform = np.linalg.inv(transform)
    #     M_transformed = inv_transform.T @ free_M @ inv_transform
        
    #     # rearrange stiffness matrix
    #     full_replaced_ids = list()
    #     for i in range(len(replaced_ids)):
    #         full_replaced_ids += [6*replaced_ids[i]+j for j in range(6)]
        
    #     P = send_to_back_permutation(len(free_M),full_replaced_ids)
    #     M_rearranged = P @ M_transformed @ P.T
        
    #     # select M and R submatrices
    #     N  = len(M_rearranged)
    #     NC = len(full_replaced_ids)
    #     NF = N-NC
        
    #     M_R = M_rearranged[:NF,:NF]
    #     M_M = M_rearranged[NF:,NF:]
    #     M_RM = M_rearranged[:NF,NF:]
        
    #     # Calculate M block marginal
    #     M_Mp = M_M - M_RM.T @ np.linalg.inv(M_R) @ M_RM
    #     M_Mp = 0.5*(M_Mp+M_Mp.T)
        
    #     ##############################################
    #     # Binding Model
    #     ##############################################
        
    #     # Calculate Incidence Matrix
    #     B, Pbar = coordinate_transformation(nuc_mu0,sks)  
        
    #     Kcomb = nuc_K + B.T @ M_Mp @ B
    #     # calculate ground state
    #     alpha = -np.linalg.inv(Kcomb) @ B.T @ M_Mp @ Pbar
        
    #     B, Pbar = coordinate_transformation_correction(nuc_mu0,sks,alpha)
    #     Kcomb = nuc_K + B.T @ M_Mp @ B 
        
    #     # b -> b - a
    #     Pbar -= shift
        
    #     # calculate ground state
    #     alpha = -np.linalg.inv(Kcomb) @ B.T @ M_Mp @ Pbar
    #     Y_C = Pbar + B @ alpha
    #     gamma = -np.linalg.inv(M_R) @ M_RM @ Y_C
        
    #     F_enthalpy = 0.5* Pbar.T @ ( M_Mp - M_Mp @ B @ np.linalg.inv(Kcomb) @ B.T @ M_Mp ) @ Pbar
    #     # print(f'F_enthalpy = {F_enthalpy}')
        
    # gs_transf_perm = np.concatenate((gamma,Y_C))
    # gs_transf = P.T @ gs_transf_perm
    # gs = inv_transform @ gs_transf
       
    # alphas = alpha.reshape((len(alpha)//6,6))
        
    # # Z entropy term
    # n = len(Kcomb)
    # logdet_sign, logdet_K = np.linalg.slogdet(Kcomb)
    # F_piK = -0.5*n * np.log(2*np.pi)
    # Z_entropy = 0.5*logdet_K + F_piK
    
    # # Z entropy term
    # n = len(M_R)
    # logdet_sign, logdet_R = np.linalg.slogdet(M_R)
    # F_piR = -0.5*n * np.log(2*np.pi)
    # R_entropy = 0.5*logdet_R + F_piR
    
    # # jacobian A
    # # F_jacob = np.log(np.linalg.det(transform))
    # signjacob, F_Ajacob = np.linalg.slogdet(transform)
        
    # # volume element B
    # signBlogdet, Blogdet = np.linalg.slogdet(B@B.T)
    # F_Bjacob = 0.5*Blogdet
    
    # # Full entropy term
    # F_entropy = Z_entropy + R_entropy + F_Ajacob #+ F_Bjacob
    
    
    # # free energy of unconstrained DNA
    # ff_logdet_sign, ff_logdet = np.linalg.slogdet(free_M)
    # ff_pi = -0.5*len(free_M) * np.log(2*np.pi)
    # F_free = 0.5*ff_logdet + ff_pi
    
    # # prepare output
    # Fdict = {
    #     'F': F_entropy + F_enthalpy,
    #     'F_entropy' : F_entropy,
    #     'F_enthalpy': F_enthalpy,
    #     'F_Ajacob'  : F_Ajacob,
    #     'F_Bjacob'  : F_Bjacob,
    #     'F_freedna' : F_free,
    #     'gs'        : gs,
    #     'alphas'    : alphas
    # }
    # return Fdict
    
    # ------------------------------------------------------------------
    # ❼  Recover ground-state vector  gs
    # ------------------------------------------------------------------
    # order = [unconstrained dofs | constrained dofs]  from the earlier slice step
    order        = np.hstack((idx_free, full_ids))

    gs_perm      = np.concatenate((gamma, Y_C))       # shape (N,)
    gs_transf    = np.empty_like(gs_perm)
    gs_transf[order] = gs_perm                        # Pᵀ · (γ, Y_C)

    if sp.issparse(transform):
        gs = spsolve_csc(transform, gs_transf)
    else:
        gs = la.solve(transform, gs_transf, assume_a='gen')
    # ------------------------------------------------------------------
    alphas   = alpha.reshape(-1, 6)                   # same as before
    # ------------------------------------------------------------------
    # ❽  Entropy / Jacobian pieces  (robust to sparse)
    # ------------------------------------------------------------------
    _, logdetK = logdet_sym(Kcomb)                    # Z–block
    Z_entropy  = 0.5*logdetK - 0.5*Kcomb.shape[0]*np.log(2*np.pi)

    _, logdetR = logdet_sym(M_R)                      # R–block
    R_entropy  = 0.5*logdetR - 0.5*M_R.shape[0]*np.log(2*np.pi)

    _, logdetA = logdet_sym(transform)                # Jacobian A
    F_Ajacob   = logdetA

    # volume element of B  (force dense → slogdet)
    BBt        = as_dense(B @ B.T)
    _, Blogdet = np.linalg.slogdet(BBt)
    F_Bjacob   = 0.5*Blogdet

    F_entropy  = Z_entropy + R_entropy + F_Ajacob     # (+ F_Bjacob if needed)

    # ------------------------------------------------------------------
    # ❾  Free DNA reference
    # ------------------------------------------------------------------
    _, logdetF = logdet_sym(free_M)
    F_free     = 0.5*logdetF - 0.5*free_M.shape[0]*np.log(2*np.pi)

    # ------------------------------------------------------------------
    # ❿  Bundle results
    # ------------------------------------------------------------------
    Fdict = {
        'F'         : F_entropy + F_enthalpy,
        'F_entropy' : F_entropy,
        'F_enthalpy': F_enthalpy,
        'F_Ajacob'  : F_Ajacob,
        'F_Bjacob'  : F_Bjacob,
        'F_freedna' : F_free,
        'gs'        : gs,
        'alphas'    : alphas
    }
    return Fdict
    

def binding_model_free_energy(
    free_gs: np.ndarray,
    free_M: np.ndarray,    
    nuc_mu0_full: np.ndarray,
    nuc_K_full: np.ndarray,
    left_open: int = 0,
    right_open: int = 0,
    use_correction: bool = True,
) -> np.ndarray:

    midstep_constraint_locations = get_midstep_locations(left_open, right_open)
    if len(midstep_constraint_locations) <= 1:
        n = len(free_M)
        F_pi = -0.5*n * np.log(2*np.pi)
        # matrix term
        logdet_sign, logdet = np.linalg.slogdet(free_M)
        F_mat = 0.5*logdet
        F = F_mat + F_pi  
        Fdict = {
            'F': F,
            'F_entropy' : F,
            'F_enthalpy': 0,
            'F_jacob'   : 0,
            'F_freedna'    : F
        }
        return Fdict
    
    sks = midstep_groundstate_se3(free_gs,midstep_constraint_locations)
    
    # select midstep triads and their stiffness 
    nuc_mu0 = nuc_mu0_full[left_open:len(nuc_mu0_full)-right_open]
    nuc_K = nuc_K_full[6*left_open:len(nuc_K_full)-6*right_open,6*left_open:len(nuc_K_full)-6*right_open]
    
    # find composite transformation
    transform, replaced_ids = midstep_composition_transformation(
        free_gs,
        midstep_constraint_locations
    )
        
    # transform stiffness matrix
    inv_transform = np.linalg.inv(transform)
    M_transformed = inv_transform.T @ free_M @ inv_transform
    
    # rearrange stiffness matrix
    full_replaced_ids = list()
    for i in range(len(replaced_ids)):
        full_replaced_ids += [6*replaced_ids[i]+j for j in range(6)]
    
    nM = free_M.shape[0]
    P = send_to_back_permutation(nM,full_replaced_ids)
    M_rearranged = P @ M_transformed @ P.T
    
    # select M and R submatrices
    N  = len(M_rearranged)
    NC = len(full_replaced_ids)
    NF = N-NC
    
    M_R = M_rearranged[:NF,:NF]
    M_M = M_rearranged[NF:,NF:]
    M_RM = M_rearranged[:NF,NF:]
    
    # Calculate M block marginal
    M_Mp = M_M - M_RM.T @ np.linalg.inv(M_R) @ M_RM
    M_Mp = 0.5*(M_Mp+M_Mp.T)
    

    ##############################################
    # Binding Model
    ##############################################
    
    # Calculate Incidence Matrix
    B, Pbar = coordinate_transformation(nuc_mu0,sks)  
    Kcomb = nuc_K + B.T @ M_Mp @ B
    # calculate ground state
    alpha = -np.linalg.inv(Kcomb) @ B.T @ M_Mp @ Pbar
    
    B, Pbar = coordinate_transformation_correction(nuc_mu0,sks,alpha)
    Kcomb = nuc_K + B.T @ M_Mp @ B
    # calculate ground state
    alpha = -np.linalg.inv(Kcomb) @ B.T @ M_Mp @ Pbar
    
    Y_C = Pbar + B @ alpha
    F_enthalpy = 0.5* Pbar.T @ ( M_Mp - M_Mp @ B @ np.linalg.inv(Kcomb) @ B.T @ M_Mp ) @ Pbar
    # print(f'F_enthalpy = {F_enthalpy}')
    
    gamma = -np.linalg.inv(M_R) @ M_RM @ Y_C
    
    if use_correction:
    
        gs_transf_perm = np.concatenate((gamma,Y_C))
        gs_transf = P.T @ gs_transf_perm
        gs = inv_transform @ gs_transf

        gs = gs.reshape((len(gs)//6,6))
        # find composite transformation
        transform, replaced_ids, shift = midstep_composition_transformation_correction(
            free_gs,
            midstep_constraint_locations,
            gs
        )
        
        # transform stiffness matrix
        inv_transform = np.linalg.inv(transform)
        M_transformed = inv_transform.T @ free_M @ inv_transform
        
        # rearrange stiffness matrix
        full_replaced_ids = list()
        for i in range(len(replaced_ids)):
            full_replaced_ids += [6*replaced_ids[i]+j for j in range(6)]
        
        P = send_to_back_permutation(len(free_M),full_replaced_ids)
        M_rearranged = P @ M_transformed @ P.T
        
        # select M and R submatrices
        N  = len(M_rearranged)
        NC = len(full_replaced_ids)
        NF = N-NC
        
        M_R = M_rearranged[:NF,:NF]
        M_M = M_rearranged[NF:,NF:]
        M_RM = M_rearranged[:NF,NF:]
        
        # Calculate M block marginal
        M_Mp = M_M - M_RM.T @ np.linalg.inv(M_R) @ M_RM
        M_Mp = 0.5*(M_Mp+M_Mp.T)
        
        ##############################################
        # Binding Model
        ##############################################
        
        # Calculate Incidence Matrix
        B, Pbar = coordinate_transformation(nuc_mu0,sks)  
        
        Kcomb = nuc_K + B.T @ M_Mp @ B
        # calculate ground state
        alpha = -np.linalg.inv(Kcomb) @ B.T @ M_Mp @ Pbar
        
        B, Pbar = coordinate_transformation_correction(nuc_mu0,sks,alpha)
        Kcomb = nuc_K + B.T @ M_Mp @ B 
        
        # b -> b - a
        Pbar -= shift
        
        # calculate ground state
        alpha = -np.linalg.inv(Kcomb) @ B.T @ M_Mp @ Pbar
        Y_C = Pbar + B @ alpha
        gamma = -np.linalg.inv(M_R) @ M_RM @ Y_C
        
        F_enthalpy = 0.5* Pbar.T @ ( M_Mp - M_Mp @ B @ np.linalg.inv(Kcomb) @ B.T @ M_Mp ) @ Pbar
        # print(f'F_enthalpy = {F_enthalpy}')
        
    gs_transf_perm = np.concatenate((gamma,Y_C))
    gs_transf = P.T @ gs_transf_perm
    gs = inv_transform @ gs_transf
       
    alphas = alpha.reshape((len(alpha)//6,6))
        
    # Z entropy term
    n = len(Kcomb)
    logdet_sign, logdet_K = np.linalg.slogdet(Kcomb)
    F_piK = -0.5*n * np.log(2*np.pi)
    Z_entropy = 0.5*logdet_K + F_piK
    
    # Z entropy term
    n = len(M_R)
    logdet_sign, logdet_R = np.linalg.slogdet(M_R)
    F_piR = -0.5*n * np.log(2*np.pi)
    R_entropy = 0.5*logdet_R + F_piR
    
    # jacobian A
    # F_jacob = np.log(np.linalg.det(transform))
    signjacob, F_Ajacob = np.linalg.slogdet(transform)
        
    # volume element B
    signBlogdet, Blogdet = np.linalg.slogdet(B@B.T)
    F_Bjacob = 0.5*Blogdet
    
    # Full entropy term
    F_entropy = Z_entropy + R_entropy + F_Ajacob #+ F_Bjacob
    
    
    # free energy of unconstrained DNA
    ff_logdet_sign, ff_logdet = np.linalg.slogdet(free_M)
    ff_pi = -0.5*len(free_M) * np.log(2*np.pi)
    F_free = 0.5*ff_logdet + ff_pi
    
    # prepare output
    Fdict = {
        'F': F_entropy + F_enthalpy,
        'F_entropy' : F_entropy,
        'F_enthalpy': F_enthalpy,
        'F_Ajacob'  : F_Ajacob,
        'F_Bjacob'  : F_Bjacob,
        'F_freedna' : F_free,
        'gs'        : gs,
        'alphas'    : alphas
    }
    return Fdict




if __name__ == '__main__':
    
    
    genstiff = GenStiffness(method='hybrid')   # alternatively you can use the 'crystal' method for the Olson data
    seq  = "CTGGAGAATCCCGGTGCCGAGGCCGCTCAATTGGTCGTAGACAGCTCTAGCACCGCTTAAACGCACGTACGCGCTGTCCCCCGCGTTTTAACCGCCAAGGGGATTACTCCCTAGTCTCCAGGCACGTGTCAGATATATACATCCTGT"

    stiffmat,groundstate = genstiff.gen_params(seq,use_group=True,sparse=True)

    triadfn = 'methods/State/Nucleosome.state'
    nuctriads = read_nucleosome_triads(triadfn)

    midstep_constraint_locations = [
        2, 6, 14, 17, 24, 29, 
        34, 38, 45, 49, 55, 59, 
        65, 69, 76, 80, 86, 90, 
        96, 100, 107, 111, 116, 121, 
        128, 131, 139, 143
    ]
    
    nuc_mu0 = calculate_midstep_triads(
        midstep_constraint_locations,
        nuctriads
    )
            
    left_open = 0
    right_open = 0
    nuc_K_pos_resc_sym    = np.load('MDParams/nuc_K_pos_resc_sym.npy')

    print('##################################')
    print('Calculate model')
    nucout = binding_model_free_energy_optimized(
        groundstate,
        stiffmat,    
        nuc_mu0,
        nuc_K_pos_resc_sym,
        left_open=left_open,
        right_open=right_open,
        use_correction=True,
        is_block_diag=True
    )
    
    print(nucout['F'])
    
    stiffmat,groundstate = genstiff.gen_params(seq,use_group=True,sparse=False)
    nucout = binding_model_free_energy(
        groundstate,
        stiffmat,    
        nuc_mu0,
        nuc_K_pos_resc_sym,
        left_open=left_open,
        right_open=right_open,
        use_correction=True,
    )
    print(nucout['F'])
    