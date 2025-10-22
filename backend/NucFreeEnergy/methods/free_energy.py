import sys, os
import numpy as np
import scipy as sp
from typing import List, Tuple, Callable, Any, Dict

from .PolyCG.polycg.SO3 import so3
from .PolyCG.polycg.transforms.transform_SO3 import euler2rotmat_so3
from .PolyCG.polycg.transforms.transform_marginals import send_to_back_permutation
from .midstep_composites import midstep_composition_transformation, midstep_se3_groundstate
from .midstep_composites import midstep_composition_transformation_correction
from .read_nuc_data import read_nucleosome_triads, GenStiffness
from .midstep_composites import midstep_composition_transformation_correction_old

from .PolyCG.polycg.cgnaplus import cgnaplus_bps_params


def nucleosome_free_energy(
    groundstate: np.ndarray,
    stiffmat: np.ndarray,
    midstep_constraint_locations: List[int],  # index of the lower (left-hand) triad neighboring the constraint midstep-triad
    nucleosome_triads: np.ndarray,
    use_correction: bool = False,
) -> np.ndarray:
    
    if len(midstep_constraint_locations) == 0:
        n = len(stiffmat)
        F_pi = -0.5*n * np.log(2*np.pi)
        # matrix term
        logdet_sign, logdet = np.linalg.slogdet(stiffmat)
        F_mat = 0.5*logdet
        F = F_mat + F_pi  
    
        Fdict = {
            'F': F,
            'F_entropy'  : F,
            'F_enthalpy' : 0,
            'F_jacob'    : 0,
            'F_freedna'  : F,
            'gs'         : np.zeros(n)
        }
        return Fdict
    
    
    midstep_constraint_locations = sorted(list(set(midstep_constraint_locations)))

    midstep_triads = calculate_midstep_triads(
        midstep_constraint_locations,
        nucleosome_triads
    )
    # mcl = [
    #     2, 6, 14, 17, 24, 29, 
    #     34, 38, 45, 49, 55, 59, 
    #     65, 69, 76, 80, 86, 90, 
    #     96, 100, 107, 111, 116, 121, 
    #     128, 131, 139, 143
    # ]
    # midstep_triads = calculate_midstep_triads(
    #     mcl,
    #     nucleosome_triads
    # )
    
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
        transform, replaced_ids, shift = midstep_composition_transformation_correction(
            groundstate,
            midstep_constraint_locations,
            gs
        )
        
        # transform stiffness matrix
        inv_transform = np.linalg.inv(transform)
        stiffmat_transformed = inv_transform.T @ stiffmat @ inv_transform
        
        # # rearrange stiffness matrix
        # full_replaced_ids = list()
        # for i in range(len(replaced_ids)):
        #     full_replaced_ids += [6*replaced_ids[i]+j for j in range(6)]
        
        # P = send_to_back_permutation(len(stiffmat),full_replaced_ids)
        stiffmat_rearranged = P @ stiffmat_transformed @ P.T

        # select fluctuating, constraint and coupling part of matrix
        N  = len(stiffmat)
        NC = len(full_replaced_ids)
        NF = N-NC
        
        MF = stiffmat_rearranged[:NF,:NF]
        MC = stiffmat_rearranged[NF:,NF:]
        MM = stiffmat_rearranged[NF:,:NF]
        
        # shift[3::6] = 0
        # shift[4::6] = 0
        
        C = C - shift
        
        MFi = np.linalg.inv(MF)
        b = MM.T @ C
        
    # Calculate ground state 
    alpha = -MFi @ b
    gs_transf_perm = np.concatenate((alpha,C))
    gs_transf = P.T @ gs_transf_perm
    gs = inv_transform @ gs_transf
    # # gs = gs.reshape((len(gs)//6,6))
    
    # constant energies
    F_const_C =  0.5 * C.T @ MC @ C
    F_const_b = -0.5 * b.T @ MFi @ b
    
    # entropy term
    n = len(MF)
    logdet_sign, logdet = np.linalg.slogdet(MF)
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
        'F': F_entropy + F_jacob + F_const_C + F_const_b,
        'F_entropy'  : F_entropy + F_jacob,
        'F_enthalpy' : F_const_C + F_const_b,
        'F_jacob'    : F_jacob,
        'F_freedna'  : F_free,
        'gs'         : gs
    }
    return Fdict


def nucleosome_free_energy_(
    groundstate: np.ndarray,
    stiffmat: np.ndarray,
    midstep_constraint_locations: List[int],  # index of the lower (left-hand) triad neighboring the constraint midstep-triad
    nucleosome_triads: np.ndarray,
    use_correction: bool = False,
) -> np.ndarray:
    
    if len(midstep_constraint_locations) == 0:
        n = len(stiffmat)
        F_pi = -0.5*n * np.log(2*np.pi)
        # matrix term
        logdet_sign, logdet = np.linalg.slogdet(stiffmat)
        F_mat = 0.5*logdet
        F = F_mat + F_pi  
    
        Fdict = {
            'F': F,
            'F_entropy' : F,
            'F_enthalpy'   : 0,
            'F_jacob'   : 0,
            'F_freedna'    : F
        }
        return Fdict
    
    
    midstep_constraint_locations = sorted(list(set(midstep_constraint_locations)))

    midstep_triads = calculate_midstep_triads(
        midstep_constraint_locations,
        nucleosome_triads
    )
    # mcl = [
    #     2, 6, 14, 17, 24, 29, 
    #     34, 38, 45, 49, 55, 59, 
    #     65, 69, 76, 80, 86, 90, 
    #     96, 100, 107, 111, 116, 121, 
    #     128, 131, 139, 143
    # ]
    # midstep_triads = calculate_midstep_triads(
    #     mcl,
    #     nucleosome_triads
    # )
    
    # find contraint excess values3::]
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
    
    # use_correction = False
    ########################################
    ########################################
    if use_correction:
        alpha = -MFi @ b
        
        gs_transf_perm = np.concatenate((alpha,C))
        gs_transf = P.T @ gs_transf_perm
        gs = inv_transform @ gs_transf
    
        gs = gs.reshape((len(gs)//6,6))
        # find composite transformation
        transform, replaced_ids, shift = midstep_composition_transformation_correction(
            groundstate,
            midstep_constraint_locations,
            gs
        )
        
        # transform stiffness matrix
        inv_transform = np.linalg.inv(transform)
        stiffmat_transformed = inv_transform.T @ stiffmat @ inv_transform
        
        stiffmat_rearranged = P @ stiffmat_transformed @ P.T

        # select fluctuating, constraint and coupling part of matrix
        N  = len(stiffmat)
        NM = len(full_replaced_ids)
        NR = N-NM
        
        M_R = stiffmat_rearranged[:NR,:NR]
        M_M = stiffmat_rearranged[NR:,NR:]
        M_K = stiffmat_rearranged[NR:,:NR]
        
        # shift[:] = 0
        
        M_Ri = np.linalg.inv(M_R)
        b = M_K.T @ (C-shift)
        
        # b = M_K.T @ C
        
        # from completing the square
        Fenth1 = -0.5 * b.T @ M_Ri @ b
        # terms 3,4, and 6 
        Fenth2 =  0.5 * (C-shift).T @ M_M @ (C-shift)
        
        # Fenth2 =  0.5 * (C).T @ M_M @ (C)
        
        # print(C)
        # print(shift)
        print(Fenth1)
        print(Fenth2)
        
        
        F_enth = Fenth1 + Fenth2
        
        # entropy term
        n = len(M_R)
        logdet_sign, logdet = np.linalg.slogdet(M_R)
        F_pi = -0.5*n * np.log(2*np.pi)
        # matrix term
        F_mat = 0.5*logdet
        F_entropy = F_pi + F_mat
        F_jacob = np.log(np.linalg.det(transform))
        
        # free energy of unconstrained DNA
        ff_logdet_sign, ff_logdet = np.linalg.slogdet(stiffmat)
        ff_pi = -0.5*len(stiffmat) * np.log(2*np.pi)
        F_free = 0.5*ff_logdet + ff_pi
        
        # alpha = -MFi @ b
        # gs_transf_perm = np.concatenate((alpha,C))
        # gs_transf = P.T @ gs_transf_perm
        # gs = inv_transform @ gs_transf
        # # gs = gs.reshape((len(gs)//6,6))
    
    else:
        # constant energies
        F_const_C =  0.5 * C.T @ MC @ C
        F_const_b = -0.5 * b.T @ MFi @ b
        F_enth = F_const_C + F_const_b
    
        print(F_const_C)
        print(F_const_b)
    
        # entropy term
        n = len(MF)
        logdet_sign, logdet = np.linalg.slogdet(MF)
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
        'F': F_entropy + F_jacob + F_enth,
        'F_entropy' : F_entropy + F_jacob,
        'F_enthalpy'   : F_enth,
        'F_jacob'   : F_jacob,
        'F_freedna'    : F_free
    }
    return Fdict

def nucleosome_free_energy_old(
    groundstate: np.ndarray,
    stiffmat: np.ndarray,
    midstep_constraint_locations: List[int],  # index of the lower (left-hand) triad neighboring the constraint midstep-triad
    nucleosome_triads: np.ndarray,
    use_correction: bool = False,
) -> np.ndarray:
    
    if len(midstep_constraint_locations) == 0:
        n = len(stiffmat)
        F_pi = -0.5*n * np.log(2*np.pi)
        # matrix term
        logdet_sign, logdet = np.linalg.slogdet(stiffmat)
        F_mat = 0.5*logdet
        F = F_mat + F_pi  
    
        Fdict = {
            'F': F,
            'F_entropy' : F,
            'F_enthalpy'   : 0,
            'F_jacob'   : 0,
            'F_freedna'    : F
        }
        return Fdict
    
    
    midstep_constraint_locations = sorted(list(set(midstep_constraint_locations)))

    midstep_triads = calculate_midstep_triads(
        midstep_constraint_locations,
        nucleosome_triads
    )
    # mcl = [
    #     2, 6, 14, 17, 24, 29, 
    #     34, 38, 45, 49, 55, 59, 
    #     65, 69, 76, 80, 86, 90, 
    #     96, 100, 107, 111, 116, 121, 
    #     128, 131, 139, 143
    # ]
    # midstep_triads = calculate_midstep_triads(
    #     mcl,
    #     nucleosome_triads
    # )
    
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
            gs
        )
        
        # transform stiffness matrix
        inv_transform = np.linalg.inv(transform)
        stiffmat_transformed = inv_transform.T @ stiffmat @ inv_transform
        
        # # rearrange stiffness matrix
        # full_replaced_ids = list()
        # for i in range(len(replaced_ids)):
        #     full_replaced_ids += [6*replaced_ids[i]+j for j in range(6)]
        
        # P = send_to_back_permutation(len(stiffmat),full_replaced_ids)
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
        
        # alpha = -MFi @ b
        # gs_transf_perm = np.concatenate((alpha,C))
        # gs_transf = P.T @ gs_transf_perm
        # gs = inv_transform @ gs_transf
        # # gs = gs.reshape((len(gs)//6,6))
    
    # constant energies
    F_const_C =  0.5 * C.T @ MC @ C
    F_const_b = -0.5 * b.T @ MFi @ b
    
    # entropy term
    n = len(MF)
    logdet_sign, logdet = np.linalg.slogdet(MF)
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
        'F': F_entropy + F_jacob + F_const_C + F_const_b,
        'F_entropy' : F_entropy + F_jacob,
        'F_enthalpy'   : F_const_C + F_const_b,
        'F_jacob'   : F_jacob,
        'F_freedna'    : F_free
    }
    return Fdict


def nucleosome_groundstate(
    groundstate: np.ndarray,
    stiffmat: np.ndarray,
    midstep_constraint_locations: List[int],  # index of the lower (left-hand) triad neighboring the constraint midstep-triad
    nucleosome_triads: np.ndarray,
    use_correction: bool = False
) -> np.ndarray:
    
    if len(midstep_constraint_locations) == 0:
        n = len(stiffmat)
        F_pi = -0.5*n * np.log(2*np.pi)
        # matrix term
        logdet_sign, logdet = np.linalg.slogdet(stiffmat)
        F_mat = 0.5*logdet
        F = F_mat + F_pi  
        return F, F, 0, 0
    
    
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
    # print(replaced_ids)
    # print(len(transform)//6)
    # sys.exit()
    
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
    
    alpha = -MFi @ b
    
    gs_transf_perm = np.concatenate((alpha,C))
    gs_transf = P.T @ gs_transf_perm
    gs = inv_transform @ gs_transf
    
    
    ####################################
    
    if use_correction:
        
        gs = gs.reshape((len(gs)//6,6))
        # find composite transformation
        transform, replaced_ids, shift = midstep_composition_transformation_correction(
            groundstate,
            midstep_constraint_locations,
            gs
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
        
        # shift[3::6] = 0
        # shift[4::6] = 0
        
        C = C - shift
        MFi = np.linalg.inv(MF)
        b = MM.T @ C
        
        alpha = -MFi @ b
        
        gs_transf_perm = np.concatenate((alpha,C))
        gs_transf = P.T @ gs_transf_perm
        gs = inv_transform @ gs_transf
        
        # gs = gs.reshape((len(gs)//6,6))
    return gs

def nucleosome_groundstate_old(
    groundstate: np.ndarray,
    stiffmat: np.ndarray,
    midstep_constraint_locations: List[int],  # index of the lower (left-hand) triad neighboring the constraint midstep-triad
    nucleosome_triads: np.ndarray,
    use_correction: bool = False
) -> np.ndarray:
    
    if len(midstep_constraint_locations) == 0:
        n = len(stiffmat)
        F_pi = -0.5*n * np.log(2*np.pi)
        # matrix term
        logdet_sign, logdet = np.linalg.slogdet(stiffmat)
        F_mat = 0.5*logdet
        F = F_mat + F_pi  
        return F, F, 0, 0
    
    
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
    # print(replaced_ids)
    # print(len(transform)//6)
    # sys.exit()
    
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
    
    alpha = -MFi @ b
    
    gs_transf_perm = np.concatenate((alpha,C))
    gs_transf = P.T @ gs_transf_perm
    gs = inv_transform @ gs_transf
    
    
    ####################################
    
    if use_correction:
        
        gs = gs.reshape((len(gs)//6,6))
        # find composite transformation
        transform, replaced_ids = midstep_composition_transformation_correction_old(
            groundstate,
            midstep_constraint_locations,
            -gs
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
        
        C = C
        MFi = np.linalg.inv(MF)
        b = MM.T @ C
        
        alpha = -MFi @ b
        
        gs_transf_perm = np.concatenate((alpha,C))
        gs_transf = P.T @ gs_transf_perm
        gs = inv_transform @ gs_transf
        
        # gs = gs.reshape((len(gs)//6,6))
    return gs


def calculate_midstep_triads(
    triad_ids: List[int],  # index of the lower (left-hand) triad neighboring the constraint midstep-triad
    nucleosome_triads: np.ndarray
) -> np.ndarray:
    midstep_triads = np.zeros((len(triad_ids),4,4))
    for i,id in enumerate(triad_ids):
        T1 = nucleosome_triads[id]
        T2 = nucleosome_triads[id+1]
        midstep_triads[i,:3,:3] = T1[:3,:3] @ so3.euler2rotmat(0.5*so3.rotmat2euler(T1[:3,:3].T @ T2[:3,:3]))
        midstep_triads[i,:3,3]  = 0.5* (T1[:3,3]+T2[:3,3])
        midstep_triads[i,3,3]   = 1
    return midstep_triads
    

def midstep_excess_vals(
    groundstate: np.ndarray,
    midstep_constraint_locations: List[int],
    midstep_triads: np.ndarray  
):
    
    num = len(midstep_constraint_locations)-1
    excess_vals = np.zeros((num,6))
    for i in range(num):
        id1 = midstep_constraint_locations[i]
        id2 = midstep_constraint_locations[i+1]
        triad1 = midstep_triads[i]
        triad2 = midstep_triads[i+1]
        partial_gs = groundstate[id1:id2+1] 
        excess_vals[i] = midstep_composition_excess(partial_gs,triad1,triad2) 
    return excess_vals
    
    
def midstep_composition_excess(
    groundstate: np.ndarray,
    triad1: np.ndarray,
    triad2: np.ndarray
) -> np.ndarray:
    g_ij = np.linalg.inv(triad1) @ triad2
    Smats = midstep_se3_groundstate(groundstate)
    s_ij = np.eye(4)
    for Smat in Smats:
        s_ij = s_ij @ Smat
    d_ij = np.linalg.inv(s_ij) @ g_ij
    # d_ij = np.linalg.inv(g_ij) @ s_ij
    X = so3.se3_rotmat2euler(d_ij)
    return X



if __name__ == '__main__':
    
    np.set_printoptions(linewidth=250,precision=3,suppress=True)
    
    genstiff = GenStiffness(method='MD')
    
    seq = ''.join(['ATCG'[np.random.randint(4)] for i in range(147)])
    seq601 = "ATCGAGAATCCCGGTGCCGAGGCCGCTCAATTGGTCGTAGACAGCTCTAGCACCGCTTAAACGCACGTACGCGCTGTCCCCCGCGTTTTAACCGCCAAGGGGATTACTCCCTAGTCTCCAGGCACGTGTCAGATATATACATCCGAT"
    randseq = 'TTCCACATGGATAATACAAGAGATTCATCGACGTGCTCATTTGGCATTAGGGCATCATCCTAATGAGATTCGGTGGCGCTAACAACTTCGCTGAAAGATCAGTGGAGCGAACTGCCCTACTGTTAATTGGGTACCAGACCTCCTCACATCGTTGGTAGCTCCGTTCCTCGCGGACCGCAAGGGCAAACGTCTTACGCGACATCTGTGAATCATAACTCAGTACTTTAAAGCTAGGGCGTATTATGCA'
    
    # seq = randseq
    seq = seq601
    
    beta = 1./4.114
    
    stiff,gs = genstiff.gen_params(seq)
    
    triadfn = os.path.join(os.path.dirname(__file__), 'State/Nucleosome.state')
    nuctriads = read_nucleosome_triads(triadfn)

    midstep_constraint_locations = [
        2, 6, 14, 17, 24, 29, 
        34, 38, 45, 49, 55, 59, 
        65, 69, 76, 80, 86, 90, 
        96, 100, 107, 111, 116, 121, 
        128, 131, 139, 143
    ]
        
    
    extended_601 = seq601 + seq601[:100]
    Fdicts = []

    sweepseq = randseq
    probs_filename = 'randseq.probs'
    sweepseq = extended_601
    probs_filename = '601.probs'

    for i in range(len(sweepseq)-146):
        
        print(i)
        seq = sweepseq[i:i+147]
        
        stiff, gs = genstiff.gen_params(seq)

        # gs,stiff = cgnaplus_bps_params(seq,euler_definition=True,group_split=True)
        
        Fdict  = nucleosome_free_energy(
            gs,
            stiff,
            midstep_constraint_locations, 
            nuctriads
        )
        Fdicts.append(Fdict)

    
    probs = np.loadtxt(probs_filename)
    betaE = -np.log(probs)
    betaE -= np.mean(betaE)
    pos   = np.arange(len(betaE))
        
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(8.6/2.54,10./2.54))
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)
    
    ax1.plot(pos,betaE,lw=1.4,color='black',zorder=0)
    
    Ftots = np.array([Fdict['F'] for Fdict in Fdicts])
    Fcnst = np.array([Fdict['F_const'] for Fdict in Fdicts])
    Fentr = np.array([Fdict['F_entropy'] for Fdict in Fdicts])
    
    Ftots_rel = Ftots - np.mean(Ftots)
    Fcnst_rel = Fcnst - np.mean(Fcnst)
    Fentr_rel = Fentr - np.mean(Fentr)
        
    Epos   = np.arange(len(Ftots_rel))
    ax1.plot(Epos,Ftots_rel,lw=1,color='blue',zorder=2)
    ax2.plot(Epos,Fentr_rel,lw=1,color='green')
    ax3.plot(Epos,Fcnst_rel,lw=1,color='red')
    
    ax1.plot(Epos,Fcnst_rel,lw=1,color='red',alpha=0.7,zorder=1)
    
    tick_pad            = 2
    axlinewidth         = 0.9
    axtick_major_width  = 0.6
    axtick_major_length = 1.6
    tick_labelsize      = 6
    label_fontsize      = 7
    
    ax1.set_xlabel('Nucleosome Position',size = label_fontsize,labelpad=1)
    ax1.set_ylabel(r'$\beta E$',size = label_fontsize,labelpad=1)
    ax2.set_xlabel('Nucleosome Position',size = label_fontsize,labelpad=1)
    ax2.set_ylabel(r'$\beta E_{\mathrm{entropic}}$',size = label_fontsize,labelpad=1)
    ax3.set_xlabel('Nucleosome Position',size = label_fontsize,labelpad=1)
    ax3.set_ylabel(r'$\beta E_{\mathrm{enthalpic}}$',size = label_fontsize,labelpad=1)
    
    ax1.tick_params(axis="both",which='major',direction="in",width=axtick_major_width,length=axtick_major_length,labelsize=tick_labelsize,pad=tick_pad)
    ax2.tick_params(axis="both",which='major',direction="in",width=axtick_major_width,length=axtick_major_length,labelsize=tick_labelsize,pad=tick_pad)
    ax3.tick_params(axis="both",which='major',direction="in",width=axtick_major_width,length=axtick_major_length,labelsize=tick_labelsize,pad=tick_pad)
        
    for axis in ['top','bottom','left','right']:
        ax1.spines[axis].set_linewidth(0.7)
        ax2.spines[axis].set_linewidth(0.7)
        ax3.spines[axis].set_linewidth(0.7)
        

    
    plt.subplots_adjust(
        left=0.09,
        right=0.98,
        bottom=0.06,
        top=0.98,
        wspace=0.2,
        hspace=0.26)
    
    plt.savefig(f'Figs/{probs_filename.split(".")[0]}.png',dpi=300,facecolor='white')
    plt.close()
    
    