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
from line_profiler import profile
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu
from scipy.linalg import solve
from methods.PolyCG.polycg.SO3.so3.pyConDec.pycondec import cond_jit


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


@cond_jit
def Hinverse(Psi):
    psih = so3.hat_map(Psi)
    psihsq = psih @ psih
    Hinv = np.eye(3)
    Hinv += 0.5* psih
    Hinv += 1./12 * psihsq
    Hinv -= 1./720 * psihsq @ psihsq
    Hinv += 1./30240 * psihsq @ psihsq @ psihsq
    return Hinv

@cond_jit
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

@cond_jit
def coordinate_transformation_correction(muk0s,sks,Z_delta_ref):
    
    # if len(Z_delta_ref.shape) < 2:
    #     Z_delta_ref = Z_delta_ref.reshape((len(Z_delta_ref)//6,6))
    
    if Z_delta_ref.ndim == 1:
        Z2 = Z_delta_ref.reshape((Z_delta_ref.shape[0] // 6, 6))
    else:
        Z2 = Z_delta_ref

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
        
        # Z0k = so3.euler2rotmat(Z_delta_ref[k,:3])
        # htheta0 = so3.hat_map(Z_delta_ref[k,:3])
        Z0k    = so3.euler2rotmat(Z2[k, :3])
        htheta0 = so3.hat_map(Z2[k, :3])

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



def rearange_matrix_move_back(M: np.ndarray,
                              move_back_ids,
                              ordered: bool = False) -> np.ndarray:
    
    move_back_ids = np.asarray(move_back_ids, dtype=int)
    if ordered:
        move_back_ids = np.sort(move_back_ids)

    N = M.shape[0]
    # Build the permutation **once** as an index array
    leading_mask      = np.ones(N,  dtype=bool)
    leading_mask[move_back_ids] = False
    leading          = np.nonzero(leading_mask)[0]          # elements to keep in front
    order            = np.concatenate([leading, move_back_ids])

    # Fancy-index twice – rows, then columns – in a single call
    return M[np.ix_(order, order)]



from scipy.linalg import cholesky, LinAlgError

def matrix_properties(A, name="A", tol=1e-8):
    props = {}
    # Basic
    props['shape']       = A.shape
    props['dtype']       = A.dtype
    
    # Square?
    props['is_square']   = (A.ndim == 2 and A.shape[0] == A.shape[1])
    if not props['is_square']:
        return props  # only square matrices get the rest
    
    n = A.shape[0]
    
    # Symmetry
    props['is_symmetric']    = bool(np.allclose(A, A.T, atol=tol))
    # Orthogonal (A⁻¹ == Aᵀ)
    if props['is_symmetric']:
        props['is_orthogonal'] = False
    else:
        I = np.eye(n, dtype=A.dtype)
        props['is_orthogonal'] = bool(np.allclose(A.T @ A, I, atol=tol))
    
    # Diagonal
    props['is_diagonal']     = bool(np.allclose(A, np.diag(np.diag(A)), atol=tol))
    
    # Triangular
    props['is_upper_tri']    = bool(np.allclose(A, np.triu(A), atol=tol))
    props['is_lower_tri']    = bool(np.allclose(A, np.tril(A), atol=tol))
    
    # Positive-definite?
    if props['is_symmetric']:
        try:
            _ = cholesky(A, lower=True)
            props['is_spd']     = True
        except LinAlgError:
            props['is_spd']     = False
    else:
        props['is_spd']         = False
    
    # Sparsity
    nnz = np.count_nonzero(A)
    props['nnz']              = int(nnz)
    props['density']          = nnz / (n*n)
    
    # Numeric rank & conditioning
    props['rank']             = np.linalg.matrix_rank(A, tol=tol)
    # Cond number is infinite if singular
    try:
        props['cond']         = np.linalg.cond(A)
    except LinAlgError:
        props['cond']         = np.inf
    
    return props



def _slice_nucleo_params(mu0_full, K_full, lo, ro):
    mu0 = mu0_full[lo:len(mu0_full)-ro]
    K = K_full[6*lo:len(K_full)-6*ro, 6*lo:len(K_full)-6*ro]
    return mu0, K





# @profile
def _compute_splu(transform):
    T_sp = csc_matrix(transform)
    return splu(T_sp, permc_spec='NATURAL', diag_pivot_thresh=1.0)
# @profile
def apply_inv(X, lu, transpose=False):
    flag = 'T' if transpose else 'N'
    return lu.solve(X, flag)
# @profile
def _apply_transform_blocks(M, lu):

    X = lu.solve(M.T, 'T')
    return lu.solve(X.T, 'T')

def _expand_replaced_ids(ids):
    full = []
    for i in ids:
        full.extend([6*i + j for j in range(6)])
    return full

def __compute_marginal_blocks(M_rearr, NC):
    N = len(M_rearr)
    NF = N - NC
    M_R = M_rearr[:NF, :NF]
    M_M = M_rearr[NF:, NF:]
    M_RM = M_rearr[:NF, NF:]
    X = solve(M_R, M_RM, assume_a='gen', check_finite=False)
    M_Mp = M_M - M_RM.T @ X
    M_Mp = 0.5 * (M_Mp + M_Mp.T)
    return M_R, M_Mp, X

# def _compute_entropy_terms(transform, M_R, Kcomb):
#     # Jacobian
#     F_A = np.linalg.slogdet(transform)[1]

#     # R entropy
#     signR, logdetR = np.linalg.slogdet(M_R)
#     F_R = 0.5 * logdetR - 0.5 * len(M_R) * np.log(2 * np.pi)
#     # K entropy
#     logdet_sign, logdet_K = np.linalg.slogdet(Kcomb)
#     F_K = 0.5 * logdet_K - 0.5 * len(Kcomb) * np.log(2 * np.pi)
#     return F_A + F_R + F_K, F_A


from scipy.linalg import cho_factor

def _compute_entropy_terms(lu, M_R, Kcomb):
    
    # F_A = np.linalg.slogdet(transform)[1]
    # 1) Jacobian: log|det(transform)| via sparse LU of T
    #    (transform is identity+few-blocks ⇒ very sparse)
    # T_sp = transform if isinstance(transform, csc_matrix) else csc_matrix(transform)
    # luT  = splu(T_sp, permc_spec='COLAMD', diag_pivot_thresh=0.0)
    # # log|det(T)| = sum_i log|U_ii|
    diagU = lu.U.diagonal()
    F_A    = float(np.sum(np.log(np.abs(diagU))))

    # 2) R entropy via Cholesky (M_R ⪰ 0)
    cR, lowerR = cho_factor(M_R, check_finite=False)
    logdetR   = 2.0 * np.sum(np.log(np.diag(cR)))
    nR        = M_R.shape[0]
    F_R       = 0.5 * logdetR - 0.5 * nR * np.log(2*np.pi)

    # 3) K entropy via Cholesky (Kcomb ⪰ 0)
    cK, lowerK = cho_factor(Kcomb, check_finite=False)
    logdetK    = 2.0 * np.sum(np.log(np.diag(cK)))
    nK         = Kcomb.shape[0]
    F_K        = 0.5 * logdetK - 0.5 * nK * np.log(2*np.pi)

    return (F_A + F_R + F_K), F_A


def _free_dna_term(M):
    n = M.shape[0]
    ff_logdet = sum(np.linalg.slogdet(M[6*i:6*(i+1), 6*i:6*(i+1)])[1] for i in range(n // 6))
    ff_pi = -0.5*len(M) * np.log(2*np.pi)
    return 0.5 * ff_logdet + ff_pi

from functools import wraps
import time
def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        retval = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__} took {end - start:.6f} seconds")
        return retval
    return wrapper


from scipy.linalg import lu_factor, lu_solve

def _compute_gamma_Yc(
    nuc_mu0, sks,
    lu_piv,    # (lu,piv) from __factor_and_marginal
    M_RM,      # off‐diag block
    M_Mp,
    nuc_K,
    get_enth=False
):

    Bn, Pbarn = coordinate_transformation(nuc_mu0, sks)
    Kcomb   = nuc_K + Bn.T @ M_Mp @ Bn
    alpha   = -solve(Kcomb, Bn.T @ M_Mp @ Pbarn, check_finite=False)

    B, Pbar = coordinate_transformation_correction(nuc_mu0, sks, alpha)
    Kcomb   = nuc_K + B.T @ M_Mp @ B
    alpha   = -solve(Kcomb, B.T @ M_Mp @ Pbar, check_finite=False)


    Y_C = Pbar + B @ alpha
    lu, piv = lu_piv
    gamma = - lu_solve((lu, piv),
                    M_RM @ Y_C,
                    check_finite=False)
    F_enthalpy = 0.0
    if get_enth:
        invKcomb   = solve(Kcomb, B.T @ M_Mp, check_finite=False)
        F_enthalpy = 0.5 * Pbar.T @ (M_Mp - M_Mp @ B @ invKcomb) @ Pbar
    return gamma, Y_C, Kcomb, F_enthalpy, Bn, Pbarn

def _compute_correction_enthalpy(nuc_mu0, sks, M_Mp, nuc_K, shift, Bn=None, Pbarn=None):
    """
    Compute the correction enthalpy term.
    """
    # 1) build incidence + GS
    if Pbarn is None or Bn is None:
        Bn, Pbarn = coordinate_transformation(nuc_mu0, sks)
    Kcomb   = nuc_K + Bn.T @ M_Mp @ Bn
    alpha   = -solve(Kcomb, Bn.T @ M_Mp @ Pbarn, check_finite=False)

    # 2) correction step
    B, Pbar = coordinate_transformation_correction(nuc_mu0, sks, alpha)
    Kcomb   = nuc_K + B.T @ M_Mp @ B
    alpha   = -solve(Kcomb, B.T @ M_Mp @ Pbar, check_finite=False)

    Pbar -= shift

    invKcomb   = solve(Kcomb, B.T @ M_Mp, check_finite=False)
    F_enthalpy = 0.5 * Pbar.T @ (M_Mp - M_Mp @ B @ invKcomb) @ Pbar

    return Kcomb, F_enthalpy


def __factor_and_marginal(M_rearr, NC, new_method=False):
    """
    Factor the top-left block M_R, return
      - lu,piv:   LU factorization of M_R
      - M_RM:     off-diagonal block
      - M_Mp:     Schur complement
    """
    N  = M_rearr.shape[0]
    NF = N - NC

    M_R  = M_rearr[:NF, :NF]
    M_RM = M_rearr[:NF, NF:]
    M_M  = M_rearr[NF:, NF:]

    if new_method:
        # 1) factor M_R once
        lu, piv = lu_factor(M_R, check_finite=False)
        # 2) solve for X = M_R^{-1} @ M_RM (multi‐RHS)
        X = lu_solve((lu, piv), M_RM, check_finite=False)
    else:
        X = solve(M_R, M_RM, assume_a='gen', check_finite=False)
        lu,piv = (None, None)
    # 3) schur‐complement
    M_Mp = M_M - M_RM.T @ X
    M_Mp = 0.5 * (M_Mp + M_Mp.T)

    return (lu, piv), M_RM, M_Mp, M_R


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

    midstep_constraint_locations = np.array(midstep_constraint_locations, dtype=int)
    indices = (midstep_constraint_locations[0],midstep_constraint_locations[-1]+1)
    midstep_constraint_locations = midstep_constraint_locations - midstep_constraint_locations[0]
    # print(f"midstep_constraint_locations: {midstep_constraint_locations}")

    free_gs = free_gs[indices[0]:indices[1]]
    free_M = free_M[indices[0]*6:indices[1]*6,indices[0]*6:indices[1]*6]

    # print("free_M properties:")
    # for k,v in matrix_properties(free_M, "free_M").items():
    #     print(f"  {k:15s}: {v}")
    if len(midstep_constraint_locations) <= 1:

        F = _free_dna_term(free_M)
        Fdict = {
            'F': F,
            'F_entropy' : F,
            'F_enthalpy': 0,
            'F_jacob'   : 0,
            'F_freedna'    : F
        }
        return Fdict
    
    sks = midstep_groundstate_se3(free_gs,midstep_constraint_locations)
    
    nuc_mu0, nuc_K = _slice_nucleo_params(nuc_mu0_full, nuc_K_full, left_open, right_open)

    # find composite transformation
    transform, replaced_ids = midstep_composition_transformation(free_gs, midstep_constraint_locations )

    ####################################################################################################333
    lu = _compute_splu(transform)
    M_transformed = _apply_transform_blocks(free_M, lu)
    full_replaced_ids = _expand_replaced_ids(replaced_ids)
    # print("free_M properties:")
    # for k,v in matrix_properties(free_M, "free_M").items():
    #     print(f"  {k:15s}: {v}")

    # print("\ntransform properties:")
    # for k,v in matrix_properties(transform, "transform").items():
    #     print(f"  {k:15s}: {v}")
    ######################################################################################################
     
    # P = send_to_back_permutation(len(free_M),full_replaced_ids)
    M_rearranged = rearange_matrix_move_back(
                                            M_transformed,
                                            move_back_ids=full_replaced_ids,   
                                            ordered=False                      
                                        )

    # M_R, M_Mp, X = __compute_marginal_blocks(M_rearr = M_rearranged, NC=len(full_replaced_ids))

    ################### Binding Model ###########################
    # Binding Model
    ##############################################
    # gamma, Y_C, Kcomb, F_enthalpy = _compute_gamma_and_enthalpy(nuc_mu0, sks, M_Mp, nuc_K,
                                                                #  X=X, shift=None, get_enth=False)
    lu_piv, M_RM, M_Mp, M_R = __factor_and_marginal(M_rearranged, NC=len(full_replaced_ids), new_method=True)

    gamma, Y_C, Kcomb, F_enthalpy, Bn, Pbarn = _compute_gamma_Yc(nuc_mu0, sks,
                                                        lu_piv, M_RM, M_Mp, nuc_K, 
                                                        get_enth=False)


    if use_correction:

        mask = np.ones(len(free_M), dtype=bool)
        mask[full_replaced_ids] = False
        leading = np.nonzero(mask)[0]
        order = np.concatenate([leading, full_replaced_ids])

        # Build the inverse permutation so that (P.T @ x)[i] == x[inv_order[i]]
        inv_order = np.empty_like(order)
        inv_order[order] = np.arange(len(free_M))

    
        gs_transf_perm = np.concatenate((gamma,Y_C))
        # gs_transf = P.T @ gs_transf_perm
        gs_transf = gs_transf_perm[inv_order]
        # print("gs_transf properties:")
        # for k,v in matrix_properties(gs_transf, "gs_transf").items():
        #     print(f"  {k:15s}: {v}")

        # gs = apply_inv(gs_transf, lu)
        gs = lu.solve(gs_transf, 'N')

        gs = gs.reshape((len(gs)//6,6))
        # find composite transformation      
        
        transform, replaced_ids, shift = midstep_composition_transformation_correction(
                                                                                    free_gs,
                                                                                    midstep_constraint_locations,
                                                                                    gs
                                                                                )
        lu = _compute_splu(transform)
        M_transformed = _apply_transform_blocks(free_M, lu)

        full_replaced_ids = _expand_replaced_ids(replaced_ids)

        M_rearranged = rearange_matrix_move_back(
                                        M_transformed,
                                        move_back_ids=full_replaced_ids,   
                                        ordered=False    
                                    )
        
        # M_R, M_Mp, X = __compute_marginal_blocks(M_rearr = M_rearranged, NC=len(full_replaced_ids))
        lu_piv, M_RM, M_Mp, M_R = __factor_and_marginal(M_rearranged, NC=len(full_replaced_ids), new_method=True)

        ##############################################
        # Binding Model
        ##############################################
        # gamma, Y_C, Kcomb, F_enthalpy = _compute_gamma_and_enthalpy(nuc_mu0, sks, M_Mp, 
        #                                                             nuc_K, X=X, 
        #                                                             shift=shift, get_enth=True)

        Kcomb, F_enthalpy = _compute_correction_enthalpy(nuc_mu0, sks, M_Mp, nuc_K, shift, Bn, Pbarn)

    # print("free_M properties:")
    # for k,v in matrix_properties(free_M, "free_M").items():
    #     print(f"  {k:15s}: {v}")
    
    # print("\ntransform properties:")
    # for k,v in matrix_properties(transform, "transform").items():
    #     print(f"  {k:15s}: {v}")

    # print("\nM_R properties:")
    # for k,v in matrix_properties(M_R, "M_R").items():
    #     print(f"  {k:15s}: {v}")

    # print("\nKcomb properties:")
    # for k,v in matrix_properties(Kcomb, "Kcomb").items():
    #     print(f"  {k:15s}: {v}")
######################################################################3
    

    # F_entropy, F_Ajacob = _compute_entropy_terms(transform, M_R, Kcomb)
    F_entropy, F_Ajacob = _compute_entropy_terms(lu, M_R, Kcomb)



    F_free = _free_dna_term(free_M)

    # prepare output
    Fdict = {
        'F': F_entropy + F_enthalpy,
        'F_entropy' : F_entropy,
        'F_enthalpy': F_enthalpy,
        'F_Ajacob'  : F_Ajacob,
        # 'F_Bjacob'  : F_Bjacob,
        'F_freedna' : F_free
        # 'gs'        : gs,
        # 'alphas'    : alphas
    }
    return Fdict


def binding_model_free_energy_old(
    free_gs: np.ndarray,
    free_M: np.ndarray,    
    nuc_mu0: np.ndarray,
    nuc_K: np.ndarray,
    left_open: int = 0,
    right_open: int = 0,
    # NUCLEOSOME_TRIADS: np.ndarray,
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
    
    # # FOR NOW WE USE THE FIXED MIDSTEP TRIADS AS MU_0
    # # Find midstep triads in fixed framework for comparison
    # FIXED_midstep_triads = calculate_midstep_triads(
    #     midstep_constraint_locations,
    #     NUCLEOSOME_TRIADS
    # )
    # nuc_mu0 = FIXED_midstep_triads
    
    sks = midstep_groundstate_se3(free_gs,midstep_constraint_locations)
    
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
    
    nuc_K *= 1
    
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
    C = 0.5* Pbar.T @ ( M_Mp - M_Mp @ B @ np.linalg.inv(Kcomb) @ B.T @ M_Mp ) @ Pbar
    # print(f'C = {C}')
    
    gamma = -np.linalg.inv(M_R) @ M_RM @ Y_C
    
    if use_correction:
    
        gs_transf_perm = np.concatenate((gamma,Y_C))
        gs_transf = P.T @ gs_transf_perm
        gs = inv_transform @ gs_transf

        gs = gs.reshape((len(gs)//6,6))
        # find composite transformation
        transform, replaced_ids = midstep_composition_transformation_correction_old(
            free_gs,
            midstep_constraint_locations,
            -gs
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
        
        # Kbare = nuc_K - B.T @ M_Mp @ B 
        # eigenvals, Q = np.linalg.eigh(Kbare)
        # print(eigenvals)
        
        # free_M = np.load('MDParams/free_midstep_Mm.npy')
        # Kbare = nuc_K - B.T @ free_M @ B 
        # eigenvals, Q = np.linalg.eigh(Kbare)
        # print(eigenvals)
        # sys.exit()
        
        # print(is_positive_definite(Kcomb))
        
        # calculate ground state
        alpha = -np.linalg.inv(Kcomb) @ B.T @ M_Mp @ Pbar
        
        Y_C = Pbar + B @ alpha
        gamma = -np.linalg.inv(M_R) @ M_RM @ Y_C
        
        C = 0.5* Pbar.T @ ( M_Mp - M_Mp @ B @ np.linalg.inv(Kcomb) @ B.T @ M_Mp ) @ Pbar
        # print(f'C = {C}')
        
    gs_transf_perm = np.concatenate((gamma,Y_C))
    gs_transf = P.T @ gs_transf_perm
    gs = inv_transform @ gs_transf
        
    F_enthalpy = C
    
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
    
    # print(f'shape M_R = {M_R.shape}')
    
    
    F_jacob = np.log(np.linalg.det(transform))
    
    
    F_entropy = Z_entropy + R_entropy + F_jacob
    
    # free energy of unconstrained DNA
    ff_logdet_sign, ff_logdet = np.linalg.slogdet(free_M)
    ff_pi = -0.5*len(free_M) * np.log(2*np.pi)
    F_free = 0.5*ff_logdet + ff_pi


    # _, logdet1 = np.linalg.slogdet(M_R)
    # _, logdet2 = np.linalg.slogdet(M_Mp)
    
    # Mcomb = np.linalg.inv(B @ np.linalg.inv(Kcomb) @ B.T)
    # logdet_sign, logdet_K = np.linalg.slogdet(Mcomb)
    # F_piK = -0.5*len(Mcomb) * np.log(2*np.pi)
    # Mcomb_entropy = 0.5*logdet_K + F_piK
    
    # print(0.5*logdet1+0.5*logdet2+F_jacob -0.5*len(free_M) * np.log(2*np.pi) )
    # print(f'{Z_entropy=}')
    # print(f'{Mcomb_entropy=}')
    # print(f'{R_entropy=}')
    # print(f'{F_free=}')
    
    # prepare output
    Fdict = {
        'F': F_entropy + F_enthalpy,
        'F_entropy' : F_entropy,
        'F_enthalpy': F_enthalpy,
        'F_jacob'   : F_jacob,
        'F_freedna'    : F_free,
        'gs'        : gs
    }
    return Fdict


