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
    
    # nuc_K *= 1
    
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
        
        # try:
        #     transform, replaced_ids, shift = midstep_composition_transformation_correction(
        #         free_gs,
        #         midstep_constraint_locations,
        #         gs
        #     )
        # except np.linalg.LinAlgError:
        #     for g in gs:
        #         print(g)        
        
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