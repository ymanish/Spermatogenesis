import sys, os
import numpy as np
from typing import List, Tuple, Callable, Any, Dict

from .PolyCG.polycg.SO3 import so3
from .PolyCG.polycg.transforms.transform_SO3 import euler2rotmat_so3
from .PolyCG.polycg.transforms.transform_marginals import send_to_back_permutation
from .PolyCG.polycg.SO3.so3.pyConDec.pycondec import cond_jit
from line_profiler import profile

@cond_jit
def midstep_groundstate_se3(gs,midstep_locs):
    num = len(midstep_locs)-1
    sks = np.zeros((num,4,4))
    for i in range(num):
        id1 = midstep_locs[i]
        id2 = midstep_locs[i+1]
        partial_gs = gs[id1:id2+1] 
        
        Smats = midstep_se3_groundstate(partial_gs)
        s_ij = np.eye(4)
        for Smat in Smats:
            s_ij = s_ij @ Smat
        sks[i] = s_ij
    return sks

@cond_jit
def midstep_groundstate(gs,midstep_locs):
    num = len(midstep_locs)-1
    mid_gs = np.zeros((num,6))
    for i in range(num):
        id1 = midstep_locs[i]
        id2 = midstep_locs[i+1]
        partial_gs = gs[id1:id2+1] 
        
        Smats = midstep_se3_groundstate(partial_gs)
        s_ij = np.eye(4)
        for Smat in Smats:
            s_ij = s_ij @ Smat
        mid_gs[i] = so3.se3_rotmat2euler(s_ij)
    return mid_gs

@cond_jit
def midstep_se3_groundstate(groundstate: np.ndarray):
    Phi0s = groundstate[:,:3]
    N = len(groundstate)
    # assign static rotation matrices
    srots = np.zeros((N,3,3))
    srots[0]  = so3.euler2rotmat(0.5*Phi0s[0])    
    srots[-1] = so3.euler2rotmat(0.5*Phi0s[-1])    
    for l in range(1,len(srots)-1):
        srots[l] = so3.euler2rotmat(Phi0s[l])   
    # assign translation vectors
    trans = np.copy(groundstate[:,3:])
    trans[0] = 0.5*trans[0]
    trans[-1] = 0.5* srots[-1].T @ trans[-1]
    
    Smats = np.zeros((N,4,4))
    for i in range(N):
        S = np.zeros((4,4))
        S[:3,:3] = srots[i]
        S[:3,3]  = trans[i]
        S[3,3]   = 1
        Smats[i] = S
    return Smats

@cond_jit
def midstep_composition_transformation(
    groundstate: np.ndarray,
    midstep_constraint_locations: List[int],
) -> np.ndarray:
    N = len(groundstate)
    mat = np.eye(N*6)
    # replaced_ids = []

    # pre-allocate
    nblocks = len(midstep_constraint_locations) - 1
    replaced_ids = np.empty(nblocks, dtype=np.int64)

    for i in range(len(midstep_constraint_locations)-1):
        id1 = midstep_constraint_locations[i]
        id2 = midstep_constraint_locations[i+1]
        # replace_id = id2
        partial_gs = groundstate[id1:id2+1]
        midstep_comp_block = midstep_composition_block(partial_gs)

        replaced_ids[i] = id2
        mat[id2*6:id2*6+6, id1*6:id2*6+6] = midstep_comp_block

        # mat[replace_id*6:replace_id*6+6,id1*6:id2*6+6] = midstep_comp_block
        # replaced_ids.append(replace_id)
    return mat, replaced_ids

@cond_jit
def midstep_composition_block(
    groundstate: np.ndarray
) -> np.ndarray:
    if len(groundstate) < 2:
        raise ValueError(f'midstep_composition_block: grounstate needs to contain at least two elements. {len(groundstate)} provided.')
    
    Phi0s = groundstate[:,:3]
    # ss    = groundstate[:,3:]
    
    N = len(groundstate)
    # assign static rotation matrices
    srots = np.zeros((N,3,3))
    srots[0]  = so3.euler2rotmat(0.5*Phi0s[0])    
    srots[-1] = so3.euler2rotmat(0.5*Phi0s[-1])    
    for l in range(1,len(srots)-1):
        srots[l] = so3.euler2rotmat(Phi0s[l])    
    
    # assign translation vectors
    trans = np.copy(groundstate[:,3:])
    trans[0] = 0.5*trans[0]
    trans[-1] = 0.5* srots[-1].T @ trans[-1]
    
    ndims = 6
    N = len(groundstate)
    i = 0
    j = N-1
    comp_block  = np.zeros((ndims,N*ndims))
    
    ################################  
    # set middle blocks (i < k < j)
    for k in range(i,j+1):
        Saccu = rot_accu(srots,k+1,j)
        comp_block[:3,k*6:k*6+3]   = Saccu.T
        comp_block[3:,k*6+3:k*6+6] = Saccu.T
        
        coup = np.zeros((3,3))
        for l in range(k+1,j+1):
            coup += so3.hat_map(-rot_accu(srots,l,j).T @ trans[l])
        coup = coup @ Saccu.T
        comp_block[3:,k*6:k*6+3] = coup
    
    ################################  
    # set first block (i)
    Saccu = rot_accu(srots,i+1,j)
    Phi_0 = Phi0s[0]
    H_half = so3.splittransform_algebra2group(0.5*Phi_0)
    Hinv   = so3.splittransform_group2algebra(Phi_0)
    Hprod  = H_half @ Hinv
    
    # assign diagonal blocks
    comp_block[:3,:3] = 0.5 * Saccu.T @ Hprod
    comp_block[3:,3:6] = 0.5 * Saccu.T
    
    coup = np.zeros((3,3))
    # first term
    for l in range(1,j+1):
        coup += so3.hat_map(-rot_accu(srots,l,j).T @ trans[l])
    coup = coup @ Saccu.T
    # second term
    coup += Saccu.T @ srots[i].T @ so3.hat_map(trans[i])
    # multiply everything with 0.5 * Hprod
    coup = 0.5 * coup @ Hprod
    # assign coupling term
    comp_block[3:,:3] = coup
    
    ################################  
    # set last block (j)
    Phi_0 = Phi0s[-1]
    H_half = so3.splittransform_algebra2group(0.5*Phi_0)
    Hinv   = so3.splittransform_group2algebra(Phi_0)
    Hprod  = H_half @ Hinv
    
    # assign diagonal blocks
    comp_block[:3,j*6:j*6+3]   = 0.5 * Hprod
    comp_block[3:,j*6+3:j*6+6] = 0.5 * srots[-1]
        
    return comp_block

@cond_jit
def midstep_composition_transformation_correction(
    groundstate: np.ndarray,
    midstep_constraint_locations: List[int],
    correction: np.ndarray
) -> np.ndarray:
    N = len(groundstate)
    mat = np.eye(N*6)
    # replaced_ids = []
    # shifts = []
    nblocks = len(midstep_constraint_locations) - 1
    replaced_ids = np.empty(nblocks, dtype=np.int64)
    shifts        = np.zeros((nblocks, 6))
    for i in range(len(midstep_constraint_locations)-1):
        id1 = midstep_constraint_locations[i]
        id2 = midstep_constraint_locations[i+1]
        # replace_id = id2
        partial_gs = groundstate[id1:id2+1]
        partial_correction = correction[id1:id2+1]
        midstep_comp_block,shift = midstep_composition_block_correction(partial_gs,partial_correction)
        # shifts.append(shift)        
        # mat[replace_id*6:replace_id*6+6,id1*6:id2*6+6] = midstep_comp_block
        # replaced_ids.append(replace_id)
        ## assign instead of append
        replaced_ids[i] = id2
        shifts[i, :]    = shift
        mat[id2*6:id2*6+6, id1*6:id2*6+6] = midstep_comp_block

    # shifts = np.array(shifts).flatten()
    shifts = shifts.reshape(nblocks * 6)
    return mat, replaced_ids, shifts

@cond_jit
def midstep_composition_block_correction(
    groundstate: np.ndarray,
    deformations: np.ndarray
) -> np.ndarray:
    if len(groundstate) < 2:
        raise ValueError(f'midstep_composition_block: groundstate needs to contain at least two elements. {len(groundstate)} provided.')
     
    if len(groundstate) != len(deformations):
        raise ValueError('Dimsional mismatch between groundstate and deformation')
    
    N = len(groundstate)
    ndims = 6
    N = len(groundstate)
    i = 0
    j = N-1
    
    ################################################
    ################################################
    # assign groundstate components
    
    # Euler vectors
    Phi0s = groundstate[:,:3]
    # static rotation matrices
    srots = np.zeros((N,3,3))
    # left half-step
    srots[0]  = so3.euler2rotmat(0.5*Phi0s[0])    
    # right half-step
    srots[-1] = so3.euler2rotmat(0.5*Phi0s[-1])    
    # bulf steps
    for l in range(1,len(srots)-1):
        srots[l] = so3.euler2rotmat(Phi0s[l])
    
    # assign translation vectors
    strans = np.copy(groundstate[:,3:])
    strans[0] = 0.5* srots[0].T @ strans[0] 
    strans[-1] = 0.5*strans[-1]
    
    ################################################
    ################################################
    # assign deformation components
    # D^0
    # Phi^0
    # R^0
    
    # Euler Vectors
    Phid0 = deformations[:,:3]
    
    # dynamic rotation matrices
    drots = np.zeros((N,3,3))
    # left half-step
    Phi_0 = Phi0s[0]
    H_half = so3.splittransform_algebra2group(0.5*Phi_0)
    Hinv   = so3.splittransform_group2algebra(Phi_0)
    Hprod  = H_half @ Hinv
    drots[0]  = so3.euler2rotmat(0.5*Hprod @ Phid0[0]) 
    # right half-step
    Phi_0 = Phi0s[-1]
    H_half = so3.splittransform_algebra2group(0.5*Phi_0)
    Hinv   = so3.splittransform_group2algebra(Phi_0)
    Hprod  = H_half @ Hinv
    drots[-1]  = so3.euler2rotmat(0.5*Hprod @ Phid0[-1]) 
    # bulk steps
    for l in range(1,len(drots)-1):
        drots[l] = so3.euler2rotmat(Phid0[l])
        
    ################################################
    ################################################
    # pre compute repeatedly occuring products
    
    Rrots = np.zeros(srots.shape)
    for l in range(len(drots)):
        Rrots[l] = srots[l] @ drots[l]
    
    ################################################
    # products of static rotation matrices
    # S_{[l,j]}
    S_lj = np.zeros((N+1,3,3))
    curr = np.eye(3)
    S_lj[N] = curr
    for k in range(N):
        curr = srots[N-1-k] @ curr
        S_lj[N-1-k] = curr
        
    ################################################
    # translational component of composites
    # s_{(l,j)}
    s_lj = np.zeros((N+1,3))
    for l in range(N):
        scomp = np.zeros(3)
        for k in range(l,N):
            scomp += rot_accu(srots,l,k-1) @ strans[k]
        s_lj[l] = scomp

    ################################################
    # lambda_k

    lambdak = np.zeros((N,3))
    for k in range(N):
        lambsum = np.zeros(3)
        # j+1 = N
        for l in range(k+1,N):
            lambsum += rot_accu(Rrots,k+1,l-1) @ srots[l] @ (drots[l] - np.eye(3)) @ s_lj[l+1]
        lambdak[k] = lambsum

    ################################################
    ################################################
    # compose composite block
    comp_block = np.zeros((ndims,N*ndims))
    const      = np.zeros(6)

    ################################  
    # set middle blocks (i < k < j)
    for l in range(i,j+1):
        prefac = S_lj[i].T @ rot_accu(Rrots,i,l-1) @ srots[l]

        if l == i:
            Phi_0 = Phi0s[0]
            H_half = so3.splittransform_algebra2group(0.5*Phi_0)
            Hinv   = so3.splittransform_group2algebra(Phi_0)
            Hprod  = H_half @ Hinv
            
            # rot
            comp_block[:3,l*6:l*6+3] = 0.5 * S_lj[l+1].T @ Hprod
            # trans
            comp_block[3:,l*6+3:l*6+6] = 0.5 * prefac
            # coupling and constant
            phid0_i = 0.5 * Hprod @ Phid0[0]            
            Hmat    = so3.splittransform_algebra2group(phid0_i)
            hspdlamHmat = so3.hat_map(s_lj[l+1]) + drots[l] @ so3.hat_map(lambdak[l]) @ Hmat
            # rot-trans coupling
            comp_block[3:,l*6:l*6+3]   = -0.5 * prefac @ hspdlamHmat @ Hprod
            # const
            const[3:] += prefac @ ( (drots[l] - np.eye(3)) @ s_lj[l+1] +  hspdlamHmat @ phid0_i ) 
                        
        elif l == j:
            Phi_0 = Phi0s[-1]
            H_half = so3.splittransform_algebra2group(0.5*Phi_0)
            Hinv   = so3.splittransform_group2algebra(Phi_0)
            Hprod  = H_half @ Hinv
            # rot
            comp_block[:3,l*6:l*6+3] = 0.5 * Hprod
            # trans
            comp_block[3:,l*6+3:l*6+6] = 0.5 * prefac
            # no constant
        else:
            # rot
            comp_block[:3,l*6:l*6+3]   = S_lj[l+1].T
            # trans
            comp_block[3:,l*6+3:l*6+6] = prefac
            # coupling and constant
            Hmat = so3.splittransform_algebra2group(Phid0[l])
            hspdlamHmat = so3.hat_map(s_lj[l+1]) + drots[l] @ so3.hat_map(lambdak[l]) @ Hmat
            # rot-trans coupling
            comp_block[3:,l*6:l*6+3] = -prefac @ hspdlamHmat
            # const
            const[3:] += prefac @ ( (drots[l] - np.eye(3)) @ s_lj[l+1] +  hspdlamHmat @ Phid0[l] ) 
    
    return comp_block,const


def midstep_composition_transformation_correction_prev(
    groundstate: np.ndarray,
    midstep_constraint_locations: List[int],
    correction: np.ndarray
) -> np.ndarray:
    N = len(groundstate)
    mat = np.eye(N*6)
    replaced_ids = []
    shifts = []
    for i in range(len(midstep_constraint_locations)-1):
        id1 = midstep_constraint_locations[i]
        id2 = midstep_constraint_locations[i+1]
        replace_id = id2
        partial_gs = groundstate[id1:id2+1]
        partial_correction = correction[id1:id2+1]
        midstep_comp_block,shift = midstep_composition_block_correction_prev(partial_gs,partial_correction)
        shifts.append(shift)        
        mat[replace_id*6:replace_id*6+6,id1*6:id2*6+6] = midstep_comp_block
        replaced_ids.append(replace_id)
    shifts = np.array(shifts).flatten()
    return mat, replaced_ids, shifts


def midstep_composition_block_correction_prev(
    groundstate: np.ndarray,
    deformations: np.ndarray
) -> np.ndarray:
    if len(groundstate) < 2:
        raise ValueError(f'midstep_composition_block: grounstate needs to contain at least two elements. {len(groundstate)} provided.')
     
    if len(groundstate) != len(deformations):
        raise ValueError('Dimsional mismatch between groundstate and deformation')
    
    N = len(groundstate)
    ndims = 6
    N = len(groundstate)
    i = 0
    j = N-1
    
    ################################################
    ################################################
    # assign groundstate components
    
    # Euler vectors
    Phi0s = groundstate[:,:3]
    # static rotation matrices
    srots = np.zeros((N,3,3))
    # left half-step
    srots[0]  = so3.euler2rotmat(0.5*Phi0s[0])    
    # right half-step
    srots[-1] = so3.euler2rotmat(0.5*Phi0s[-1])    
    # bulf steps
    for l in range(1,len(srots)-1):
        srots[l] = so3.euler2rotmat(Phi0s[l])
    
    # assign translation vectors
    strans = np.copy(groundstate[:,3:])
    strans[0] = 0.5* srots[0].T @ strans[0] 
    strans[-1] = 0.5*strans[-1]
    
    # strans[0] = 0.5*strans[0]
    # strans[-1] = 0.5* srots[-1].T @ strans[-1]
    
    ################################################
    ################################################
    # assign deformation components
     
    # Euler Vectors
    Phids = deformations[:,:3]
    
    # dynamic rotation matrices
    drots = np.zeros((N,3,3))
    # left half-step
    Phi_0 = Phi0s[0]
    H_half = so3.splittransform_algebra2group(0.5*Phi_0)
    Hinv   = so3.splittransform_group2algebra(Phi_0)
    Hprod  = H_half @ Hinv
    drots[0]  = so3.euler2rotmat(0.5*Hprod @ Phids[0]) 
    # right half-step
    Phi_0 = Phi0s[-1]
    H_half = so3.splittransform_algebra2group(0.5*Phi_0)
    Hinv   = so3.splittransform_group2algebra(Phi_0)
    Hprod  = H_half @ Hinv
    drots[-1]  = so3.euler2rotmat(0.5*Hprod @ Phids[-1]) 
    # bulk steps
    for l in range(1,len(drots)-1):
        drots[l] = so3.euler2rotmat(Phids[l])
        
    ################################################
    ################################################
    # pre compute repeatedly occuring products
    
    ################################################
    # products of static rotation matrices
    # S_{[l,j]}
    S_lj = np.zeros((N+1,3,3))
    curr = np.eye(3)
    S_lj[N] = curr
    for k in range(N):
        curr = srots[N-1-k] @ curr
        S_lj[N-1-k] = curr
        
    ################################################
    # translational component of composites
    # s_{(l,j)}
    s_lj = np.zeros((N+1,3))
    for l in range(N):
        scomp = np.zeros(3)
        for k in range(l,N):
            scomp += rot_accu(srots,l,k-1) @ strans[k]
        s_lj[l] = scomp

    # print(so3.rotmat2euler(S_ikm1[0])*180/np.pi)



    ################################################
    ################################################
    # TEST
    
    rrots = np.zeros((N,3,3))
    for l in range(len(drots)):
        rrots[l] = srots[l] @ drots[l].T

    # s_lj = np.zeros((N+1,3))
    # for l in range(N):
    #     scomp = np.zeros(3)
    #     for k in range(l,N):
    #         scomp += rot_accu(rrots,l,k-1) @ strans[k]
    #     s_lj[l] = scomp

    # l = 0
    # scomp = np.zeros(3)
    # for k in range(l,N):
    #     scomp += rot_accu(rrots,l,k-1) @ strans[k]
    # s_lj[l] = scomp


    ################################################
    ################################################
    # compose composite block
    comp_block  = np.zeros((ndims,N*ndims))
    
    # ################################  
    # # set middle blocks (i < k < j)
    # for k in range(i,j+1):
    #     Saccu = rot_accu(srots,k+1,j)
    #     Raccu = rot_accu(rrots,k+1,j)
    #     comp_block[:3,k*6:k*6+3]   = Saccu.T
    #     comp_block[3:,k*6+3:k*6+6] = Raccu.T
    #     # comp_block[3:,k*6+3:k*6+6] = Saccu.T
        
    #     coup = np.zeros((3,3))
    #     for l in range(k+1,j+1):
    #         coup += so3.hat_map(-rot_accu(srots,l,j).T @ strans[l])
    #     coup = coup @ Raccu.T
    #     comp_block[3:,k*6:k*6+3] = coup
    
    ################################  
    
    shift = np.zeros(6)
    
    # set middle blocks (i < k < j)
    for l in range(i,j+1):
        
        # rotational prefactor
        prefac = S_lj[i+1].T
        for k in range(i,l):
            prefac = prefac @ drots[k] @ srots[k+1] 
            # prefac = prefac @ np.eye(3) @ srots[k+1] 
        shat = so3.hat_map(s_lj[l+1])
        
        ho_rot = prefac @ (drots[l]-np.eye(3)-so3.hat_map(Phids[l]))
        
        # prefac = rot_accu(rrots,k+1,j).T
        # TEST: This should be the uncorrected version
        # prefac = S_lj[l+1].T
        
        if l == i:
            Phi_0 = Phi0s[0]
            H_half = so3.splittransform_algebra2group(0.5*Phi_0)
            Hinv   = so3.splittransform_group2algebra(Phi_0)
            Hprod  = H_half @ Hinv
            
            # rot
            comp_block[:3,l*6:l*6+3] = 0.5 * S_lj[l+1].T @ Hprod
            # trans
            comp_block[3:,l*6+3:l*6+6] = 0.5 * prefac
            # rot-trans coupling
            comp_block[3:,l*6:l*6+3]   = 0.5 * prefac @ ( srots[i].T @ so3.hat_map(strans[i]) - shat ) @ Hprod
            # comp_block[3:,l*6:l*6+3]   = 0.5 * prefac @ ( - shat ) @ Hprod
            
            # scomp = np.zeros(3)
            # for k in range(l+1,N):
            #     scomp += rot_accu(rrots,l+1,k-1) @ strans[k]
            # shat = so3.hat_map(scomp)
            
            # comp_block[3:,l*6:l*6+3]   = 0.5 * rot_accu(rrots,i+1,j).T @ ( srots[i].T @ so3.hat_map(strans[i]) - shat ) @ Hprod
            # print(prefac)
            # print(rot_accu(srots,i+1,j).T)
            # # sys.exit()
            
            # compare = np.copy(comp_block[:6,l*6:l*6+6])
            
            shift[3:] += 0.5 * ho_rot @ s_lj[l+1]
             
        elif l == j:
            Phi_0 = Phi0s[-1]
            H_half = so3.splittransform_algebra2group(0.5*Phi_0)
            Hinv   = so3.splittransform_group2algebra(Phi_0)
            Hprod  = H_half @ Hinv
            
            # print(S_lj[l+1].T)
            # print('here')
            
            # print(s_lj[l+1])
            # sys.exit()
            
            # rot
            # S_lj[l+1].T = np.eye(3)
            comp_block[:3,l*6:l*6+3] = 0.5 * Hprod
            # comp_block[:3,l*6:l*6+3] = 0.5 * S_lj[l+1].T @ Hprod
            # trans
            comp_block[3:,l*6+3:l*6+6] = 0.5 * prefac * srots[-1]
            # no rot-trans coupling
            
            # shift += 0.5 * ho_rot @ s_lj[l+1]

        else:
            # rot
            comp_block[:3,l*6:l*6+3]   = S_lj[l+1].T
            # trans
            comp_block[3:,l*6+3:l*6+6] = prefac
            # rot-trans coupling
            comp_block[3:,l*6:l*6+3]   = -prefac @ shat
            
            shift[3:] += ho_rot @ s_lj[l+1]
            
            # ########################################
            # # TEST
            
            # Raccu = rot_accu(rrots,i+1,j)
            # comp_block[3:,l*6+3:l*6+6] = Raccu.T
            
            # coup = np.zeros((3,3))
            # for k in range(l+1,j+1):
            #     coup += so3.hat_map(-rot_accu(rrots,k,j).T @ strans[k])
            # # coup = coup @ S_lj[l+1].T
            # coup = coup @ Raccu.T
            
            # print(-S_lj[l+1].T @ shat)
            # print(coup)
        
    ################################  
    # set first block (i)
    Saccu = rot_accu(srots,i+1,j)
    Raccu = rot_accu(rrots,i+1,j)
    Phi_0 = Phi0s[0]
    H_half = so3.splittransform_algebra2group(0.5*Phi_0)
    Hinv   = so3.splittransform_group2algebra(Phi_0)
    Hprod  = H_half @ Hinv
    
    # assign diagonal blocks
    comp_block[:3,:3] = 0.5 * Saccu.T @ Hprod
    comp_block[3:,3:6] = 0.5 * Raccu.T
    
    coup = np.zeros((3,3))
    # first term
    for l in range(1,j+1):
        coup += so3.hat_map(-rot_accu(rrots,l,j).T @ strans[l])
    coup = coup @ Raccu.T
    # second term
    coup += Saccu.T @ srots[i].T @ so3.hat_map(strans[i])
    # multiply everything with 0.5 * Hprod
    coup = 0.5 * coup @ Hprod
    # assign coupling term
    # comp_block[3:,:3] = coup
    
    # print('trans')
    # print(compare[3:,3:])
    # print(0.5 * Raccu.T)
    # print('coup')
    # print(compare[3:,:3])
    # print(coup)
    # sys.exit()
    
    # ################################  
    # # set last block (j)
    # Phi_0 = Phi0s[-1]
    # H_half = so3.splittransform_algebra2group(0.5*Phi_0)
    # Hinv   = so3.splittransform_group2algebra(Phi_0)
    # Hprod  = H_half @ Hinv
    
    # # assign diagonal blocks
    # comp_block[:3,j*6:j*6+3]   = 0.5 * Hprod
    # comp_block[3:,j*6+3:j*6+6] = 0.5 * rrots[-1]
        
    return comp_block,shift


def midstep_composition_transformation_correction_old(
    groundstate: np.ndarray,
    midstep_constraint_locations: List[int],
    correction: np.ndarray
) -> np.ndarray:
    N = len(groundstate)
    mat = np.eye(N*6)
    replaced_ids = []
    for i in range(len(midstep_constraint_locations)-1):
        id1 = midstep_constraint_locations[i]
        id2 = midstep_constraint_locations[i+1]
        replace_id = id2
        partial_gs = groundstate[id1:id2+1]
        partial_correction = correction[id1:id2+1]
        midstep_comp_block = midstep_composition_block_correction_old(partial_gs,partial_correction)
        mat[replace_id*6:replace_id*6+6,id1*6:id2*6+6] = midstep_comp_block
        replaced_ids.append(replace_id)
    return mat, replaced_ids

def midstep_composition_block_correction_old(
    groundstate: np.ndarray,
    deformations: np.ndarray
) -> np.ndarray:
    if len(groundstate) < 2:
        raise ValueError(f'midstep_composition_block: grounstate needs to contain at least two elements. {len(groundstate)} provided.')
    
    Phi0s = groundstate[:,:3]
    # ss    = groundstate[:,3:]
    Phids = deformations[:,:3]
    
    N = len(groundstate)
    # assign static rotation matrices
    srots = np.zeros((N,3,3))
    srots[0]  = so3.euler2rotmat(0.5*Phi0s[0])    
    srots[-1] = so3.euler2rotmat(0.5*Phi0s[-1])    
    for l in range(1,len(srots)-1):
        srots[l] = so3.euler2rotmat(Phi0s[l])
        
    drots = np.zeros((N,3,3))
    
    Phi_0 = Phi0s[0]
    H_half = so3.splittransform_algebra2group(0.5*Phi_0)
    Hinv   = so3.splittransform_group2algebra(Phi_0)
    Hprod  = H_half @ Hinv
    drots[0]  = so3.euler2rotmat(0.5*Hprod @ Phids[0]) 
    
    Phi_0 = Phi0s[-1]
    H_half = so3.splittransform_algebra2group(0.5*Phi_0)
    Hinv   = so3.splittransform_group2algebra(Phi_0)
    Hprod  = H_half @ Hinv
    drots[-1]  = so3.euler2rotmat(0.5*Hprod @ Phids[-1]) 
        
    # drots[0]  = so3.euler2rotmat(0.5*Phids[0])    
    # drots[-1] = so3.euler2rotmat(0.5*Phids[-1])    
    for l in range(1,len(drots)-1):
        drots[l] = so3.euler2rotmat(Phids[l])
        
    rrots = np.zeros((N,3,3))
    for l in range(len(drots)):
        rrots[l] = srots[l] @ drots[l]
    
    # assign translation vectors
    strans = np.copy(groundstate[:,3:])
    strans[0] = 0.5* srots[0].T @ strans[0] 
    strans[-1] = 0.5*strans[-1]
    
    # strans[0] = 0.5*strans[0]
    # strans[-1] = 0.5* srots[-1].T @ strans[-1]
    
    ndims = 6
    N = len(groundstate)
    i = 0
    j = N-1
    comp_block  = np.zeros((ndims,N*ndims))
    
    ################################  
    # set middle blocks (i < k < j)
    for k in range(i,j+1):
        Saccu = rot_accu(srots,k+1,j)
        Raccu = rot_accu(rrots,k+1,j)
        comp_block[:3,k*6:k*6+3]   = Saccu.T
        comp_block[3:,k*6+3:k*6+6] = Raccu.T
        # comp_block[3:,k*6+3:k*6+6] = Saccu.T
        
        coup = np.zeros((3,3))
        for l in range(k+1,j+1):
        #     coup += so3.hat_map(-rot_accu(srots,l,j).T @ trans[l])
        # coup = coup @ Saccu.T
            coup += so3.hat_map(-rot_accu(rrots,l,j).T @ strans[l])
            # coup += so3.hat_map(-rot_accu(srots,l,j).T @ strans[l])
        coup = coup @ Raccu.T
        comp_block[3:,k*6:k*6+3] = coup
            
    ################################  
    # set first block (i)
    Saccu = rot_accu(srots,i+1,j)
    Raccu = rot_accu(rrots,i+1,j)
    Phi_0 = Phi0s[0]
    H_half = so3.splittransform_algebra2group(0.5*Phi_0)
    Hinv   = so3.splittransform_group2algebra(Phi_0)
    Hprod  = H_half @ Hinv
    
    # assign diagonal blocks
    comp_block[:3,:3] = 0.5 * Saccu.T @ Hprod
    comp_block[3:,3:6] = 0.5 * Raccu.T
    
    coup = np.zeros((3,3))
    # first term
    for l in range(1,j+1):
        coup += so3.hat_map(-rot_accu(rrots,l,j).T @ strans[l])
    coup = coup @ Raccu.T
    # second term
    coup += Saccu.T @ srots[i].T @ so3.hat_map(strans[i])
    # multiply everything with 0.5 * Hprod
    coup = 0.5 * coup @ Hprod
    # assign coupling term
    comp_block[3:,:3] = coup
    
    ################################  
    # set last block (j)
    Phi_0 = Phi0s[-1]
    H_half = so3.splittransform_algebra2group(0.5*Phi_0)
    Hinv   = so3.splittransform_group2algebra(Phi_0)
    Hprod  = H_half @ Hinv
    
    # assign diagonal blocks
    comp_block[:3,j*6:j*6+3]   = 0.5 * Hprod
    comp_block[3:,j*6+3:j*6+6] = 0.5 * rrots[-1]
        
    return comp_block

@cond_jit
def rot_accu(rots: np.ndarray,i,j) -> np.ndarray:
    raccu = np.eye(3)
    for k in range(i,j+1):
        raccu = raccu @ rots[k]
    return raccu


# def midstep_srots_and_trans(groundstate: np.ndarray) -> np.ndarray:
#     Phi0s = groundstate[:,:3]
#     N = len(Phi0s)
#     srots = np.zeros((N,3,3))
#     srots[0]  = so3.euler2rotmat(0.5*Phi0s[0])    
#     srots[-1] = so3.euler2rotmat(0.5*Phi0s[-1])    
#     for l in range(1,len(srots)-1):
#         srots[l] = so3.euler2rotmat(Phi0s[l])  
    
#     trans = np.copy(groundstate[:,3:])
#     trans[0] = 0.5*trans[0]
#     trans[-1] = 0.5* srots[-1].T @ trans[-1]
#     return srots, trans

