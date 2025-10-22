import sys, os
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import List, Tuple, Callable, Any, Dict


from methods.PolyCG.polycg.transforms.transform_marginals import send_to_back_permutation
from methods.midstep_composites import midstep_composition_transformation, midstep_se3_groundstate
from methods.midstep_composites import midstep_composition_transformation_correction
from methods.read_nuc_data import read_nucleosome_triads, GenStiffness
from methods.free_energy import calculate_midstep_triads, midstep_excess_vals


def get_midstep_locations(left_open: int, right_open: int):
    MIDSTEP_LOCATIONS = [
        2, 6, 14, 17, 24, 29, 
        34, 38, 45, 49, 55, 59, 
        65, 69, 76, 80, 86, 90, 
        96, 100, 107, 111, 116, 121, 
        128, 131, 139, 143
    ]
    if left_open + right_open > len(MIDSTEP_LOCATIONS):
        return []
    return MIDSTEP_LOCATIONS[left_open:len(MIDSTEP_LOCATIONS)-right_open]

def get_block_diag(stiffs: np.ndarray, dim: int = 6, left_open: int = 0, right_open: int = 0):
    if left_open + right_open > len(stiffs):
        return []
    right_id = len(stiffs)- right_open
    mddim = (len(stiffs)-left_open-right_open)*dim
    M = np.zeros((mddim,mddim))
    for i in range(len(stiffs)-left_open-right_open):
        M[i*dim:(i+1)*dim,i*dim:(i+1)*dim] = stiffs[i+left_open]
    return M

def set_block_diag(M: np.ndarray,dim: int = 6):
    M_diag = np.zeros(M.shape)
    for i in range(len(M)//dim):
        M_diag[i*dim:(i+1)*dim,i*dim:(i+1)*dim] = M[i*dim:(i+1)*dim,i*dim:(i+1)*dim]
    return M_diag

def select_partial(M, dim: int = 6, left_open: int = 0, right_open: int = 0, marginalize=True):
    n = len(M)//dim
    if marginalize:
        return np.linalg.inv(np.linalg.inv(M)[left_open*dim:(n-right_open)*dim,left_open*dim:(n-right_open)*dim])
    return M[left_open*dim:(n-right_open)*dim,left_open*dim:(n-right_open)*dim]



def soft_free_energy(
    groundstate: np.ndarray,
    stiffmat: np.ndarray,    
    left_open: int,
    right_open: int,
    nucleosome_triads: np.ndarray,
    K: np.ndarray,
    use_correction: bool = True,
) -> np.ndarray:

    midstep_constraint_locations = get_midstep_locations(left_open, right_open)

    if len(midstep_constraint_locations) <= 1:
        n = len(stiffmat)
        F_pi = -0.5*n * np.log(2*np.pi)
        # matrix term
        logdet_sign, logdet = np.linalg.slogdet(stiffmat)
        F_mat = 0.5*logdet
        F = F_mat + F_pi  
        Fdict = {
            'F': F,
            'F_entropy' : F,
            'F_enthalpy': 0,
            'F_jacob'   : 0,
            'F_free'    : F,
            'D_diff'    : 0
        }
        return Fdict
    
    midstep_constraint_locations = sorted(list(set(midstep_constraint_locations)))

    midstep_triads = calculate_midstep_triads(
        midstep_constraint_locations,
        nucleosome_triads
    )
    
    # find contraint excess values
    excess_vals = midstep_excess_vals(
        groundstate,
        midstep_constraint_locations,
        midstep_triads
    )  
    C = excess_vals.flatten()
        
    # find composite transformation
    transform, replaced_ids = midstep_composition_transformation(
        groundstate,
        midstep_constraint_locations
    )
    
    # transform stiffness matrix
    inv_transform = np.linalg.inv(transform)
    stiffmat_transformed = inv_transform.T @ stiffmat @ inv_transform
    
    # rearrange stiffness matrix
    full_replaced_ids = list()
    for i in range(len(replaced_ids)):
        full_replaced_ids += [6*replaced_ids[i]+j for j in range(6)]
     
    P = send_to_back_permutation(len(stiffmat),full_replaced_ids)
    stiffmat_rearranged = P @ stiffmat_transformed @ P.T

    # select fluctuating, constraint and coupling part of matrix
    N  = len(stiffmat)
    NC = len(full_replaced_ids)
    NF = N-NC
    
    MF = stiffmat_rearranged[:NF,:NF]
    MC = stiffmat_rearranged[NF:,NF:]
    MM = stiffmat_rearranged[NF:,:NF]
    
    MFi = np.linalg.inv(MF)
    b = MM.T @ C
    
    ########################################
    ########################################
    if use_correction:
        alpha = -MFi @ b
        
        gs_transf_perm = np.concatenate((alpha,C))
        gs_transf = P.T @ gs_transf_perm
        gs = inv_transform @ gs_transf
    
        gs = gs.reshape((len(gs)//6,6))
        # find composite transformation
        transform, replaced_ids = midstep_composition_transformation_correction(
            groundstate,
            midstep_constraint_locations,
            -gs
        )
        
        # transform stiffness matrix
        inv_transform = np.linalg.inv(transform)
        stiffmat_transformed = inv_transform.T @ stiffmat @ inv_transform
        
        stiffmat_rearranged = P @ stiffmat_transformed @ P.T

        # select fluctuating, constraint and coupling part of matrix
        N  = len(stiffmat)
        NC = len(full_replaced_ids)
        NF = N-NC
        
        MF = stiffmat_rearranged[:NF,:NF]
        MC = stiffmat_rearranged[NF:,NF:]
        MM = stiffmat_rearranged[NF:,:NF]
        
        MFi = np.linalg.inv(MF)
        b = MM.T @ C
    
    ########################################
    ########################################
    
    # constant energies
    F_const_C =  0.5 * C.T @ MC @ C
    F_const_b = -0.5 * b.T @ MFi @ b
    
    F_enthalpy = F_const_C + F_const_b
    
    
    K_partial = select_partial(K,left_open=left_open,right_open=right_open)
    Mtot_k = np.copy(stiffmat_rearranged)
    Mtot_k[NF:,NF:] += K_partial
    
    # print(stiffmat_rearranged[NF:NF+6,NF:NF+6])
    # print(Mtot_k[NF:NF+6,NF:NF+6])
    
    
    # entropy term
    n = len(Mtot_k)
    logdet_sign, logdet = np.linalg.slogdet(Mtot_k)
    F_pi = -0.5*n * np.log(2*np.pi)
    # matrix term
    F_mat = 0.5*logdet
    F_entropy = F_pi + F_mat
    F_jacob = np.log(np.linalg.det(transform))
    
    # free energy of unconstrained DNA
    ff_logdet_sign, ff_logdet = np.linalg.slogdet(stiffmat)
    ff_pi = -0.5*len(stiffmat) * np.log(2*np.pi)
    F_free = 0.5*ff_logdet + ff_pi
     
    # prepare output
    Fdict = {
        'F': F_entropy + F_jacob + F_enthalpy,
        'F_entropy' : F_entropy + F_jacob,
        'F_enthalpy': F_enthalpy,
        'F_jacob'   : F_jacob,
        'F_free'    : F_free,
        'Fdiff'     : F_entropy + F_jacob + F_enthalpy - F_free
    }
    return Fdict
