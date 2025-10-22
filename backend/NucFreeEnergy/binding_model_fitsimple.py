import sys, os
num_threads = 3
os.environ["OMP_NUM_THREADS"] = f"{num_threads}"    # For libraries using OpenMP
os.environ["MKL_NUM_THREADS"] = f"{num_threads}"    # For Intel MKL
os.environ["OPENBLAS_NUM_THREADS"] = f"{num_threads}"  # For OpenBLAS
os.environ["BLIS_NUM_THREADS"] = f"{num_threads}"   # For BLIS

import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import List, Tuple, Callable, Any, Dict
from methods.PolyCG.polycg.SO3 import so3
from methods.read_nuc_data import read_nucleosome_triads, GenStiffness
from methods.free_energy import calculate_midstep_triads

from binding_model import *

np.set_printoptions(linewidth=250,precision=6,suppress=True)

def build_helicoidal(Ys,Yd):
    if Ys.shape[-1] == 4:
        ss = Ys
    else:
        ss = np.zeros((len(Ys),4,4))
        for i in range(len(ss)):
            ss[i] = so3.se3_euler2rotmat(Ys[i])
    
    if len(Yd.shape) == 1:
        Yd = np.copy(Yd).reshape((len(Yd)//6,6))
            
    ds = np.zeros((len(Yd),4,4))
    for i in range(len(Yd)):
        ds[i] = so3.se3_euler2rotmat(Yd[i])
    X = np.zeros(Yd.shape)
    for i in range(len(X)):
        X[i] = so3.se3_rotmat2euler(ss[i] @ ds[i])
    return X

def helicoidal_full2dynamic(X,Ys):
    if Ys.shape[-1] == 4:
        ss = Ys
    else:
        ss = np.zeros((len(Ys),4,4))
        for i in range(len(ss)):
            ss[i] = so3.se3_euler2rotmat(Ys[i])
    if X.shape[-1] == 4:
        gs = X
    else:
        gs = np.zeros((len(X),4,4))
        for i in range(len(gs)):
            gs[i] = so3.se3_euler2rotmat(X[i])
            
    Yd = np.zeros(X.shape)
    for i in range(len(gs)):
        Yd[i] = so3.se3_rotmat2euler(np.linalg.inv(ss[i]) @ gs[i])
    return Yd


######################################################################################################################################################
# Plot helicoidal parameters
######################################################################################################################################################

def plot_helicoidal(
    reference_values: np.ndarray, 
    theory: np.ndarray, 
    savefn: str=None,
    shift_midids: int=0,
    vectorfig: bool=True
    ):

    midstep_ids = [
        2, 6, 
        14, 17, 24, 29, 34, 38, 45, 49, 55,
        59, 65, 69, 76, 80, 86, 90, 96, 100, 107,
        111, 116, 121, 128, 131, 139, 143
    ]
    midstep_ids = np.array(midstep_ids) + shift_midids
    gs_thc = np.copy(theory)
    gs_num = np.copy(reference_values)

    ##################################################
    # general Figure Setup

    fig_width = 17.6
    fig_height = 9.5

    axlinewidth  = 0.8
    axtick_major_width  = 0.8
    axtick_major_length = 2.4
    axtick_minor_width  = 0.4
    axtick_minor_length = 1.6

    tick_pad        = 2
    tick_labelsize  = 5
    label_fontsize  = 6
    legend_fontsize = 6

    panel_label_fontsize = 8
    label_fontweight= 'bold'
    panel_label_fontweight= 'bold'

    ##################################################
    # colors 
    # base_colors  = ['#5694bf','#bf8156','#aa7fb7','#8cb77f']
    colors = ['#5694bf','#bf8156','#aa7fb7','#8cb77f','blue','green','red','purple']
    # colors = ['#2F0C1F','#432435','#6D5462','#97858F','#C0B6BB','#D2CBCF']
    colors = colors[::-1]

    markers = ['o','s','D','<']
    base_size = [18,13,13.2]

    ##################################################
    # Plot Specs

    filled_markers = True
    scatter_zorder = 3
    scatter_alpha  = 0.8
    plot_zorder    = 1

    theory_ls = '-'
    theory_alpha = 0.7

    marker_size     = 10
    marker_linewidth = 0.7
    plot_linewidth  = 1

    ###########################################################################################
    ### FIGURE SETUP
    ###########################################################################################
    def cm_to_inch(cm: float) -> float:
        return cm/2.54

    fig = plt.figure(figsize=(cm_to_inch(fig_width)*8.2/14, cm_to_inch(fig_height)), dpi=300,facecolor='w',edgecolor='k') 
    ax1 = plt.subplot2grid(shape=(18, 8), loc=(0, 0), colspan=8,rowspan=3)
    plt.minorticks_on() 
    ax2 = plt.subplot2grid(shape=(18, 8), loc=(3, 0), colspan=8,rowspan=3)
    plt.minorticks_on()
    ax3 = plt.subplot2grid(shape=(18, 8), loc=(6, 0), colspan=8,rowspan=3)
    plt.minorticks_on() 
    ax4 = plt.subplot2grid(shape=(18, 8), loc=(9, 0), colspan=8,rowspan=3)
    plt.minorticks_on() 
    ax5 = plt.subplot2grid(shape=(18, 8), loc=(12, 0), colspan=8,rowspan=3)
    plt.minorticks_on() 
    ax6 = plt.subplot2grid(shape=(18, 8), loc=(15, 0), colspan=8,rowspan=3)
    plt.minorticks_on() 
    axes = [ax1,ax2,ax3,ax4,ax5,ax6]

    ######################################################
    # Panel a

    marker = markers[0]
    labels = ['Tilt','Roll','Twist','Shift','Slide','Rise']
    size = 7

    bpids = np.arange(len(gs_num))
    for i in range(6):
        ax = axes[i]
        label = labels[i]

        if i < 3:
            ax.set_ylabel(f'{label} (rad)',fontsize=label_fontsize,fontweight=label_fontweight)
        else:
            ax.set_ylabel(f'{label} (nm)',fontsize=label_fontsize,fontweight=label_fontweight)

        color = colors[i]
        ax.scatter(bpids,gs_num[:,i],s=size,edgecolors='black',linewidth=marker_linewidth,color=color,label=label,marker=marker,zorder=2,alpha=0.4)
        ax.plot(bpids,gs_thc[:,i],linewidth=1.2,color='black',alpha=0.5,zorder=1)
            
        ax.set_xlim((-1,146))
        ax.yaxis.set_label_coords(-0.066,0.5)
        
        # ax.plot(bpids,gs_thc_rand[:,i],linewidth=0.6,color='blue',alpha=1,zorder=1)
        # ax.scatter(bpids,gs_thc_rand[:,i],s=4,edgecolors='black',linewidth=marker_linewidth,color='black',label=label,marker=marker,zorder=2,alpha=0.4)
    
        if i < 5:
            ax.set_xticklabels([])
            
        ########################
        # set ylim
        ymargin = 0.05
        vals = [gs_num[:,i],gs_thc[:,i]]
        vals = np.concatenate(vals)
        ymax = np.max(vals)
        ymin = np.min(vals)
        yrge = np.ptp(vals)
        ax.set_ylim((ymin-yrge*ymargin,ymax+yrge*ymargin))
        
        ########################
        # shaded regions of midsteps
        top = ymax+yrge*ymargin*2
        bot = ymin-yrge*ymargin*2
        for mid in midstep_ids:
            ax.fill_between([mid-0.5,mid+0.5],[bot,bot],[top,top],zorder=1,color='grey',alpha=0.2,edgecolor='none')
        
        mid_fontweight = 'normal'
        mid_fontsize   = 7
        if i == 0:
            for bid in range(len(midstep_ids)//2):
                ypos = top+0.005
                xpos = 0.5*(midstep_ids[2*bid]+midstep_ids[2*bid+1])
                ax.text(xpos, ypos, f'{bid+1}',
                fontsize=mid_fontsize,
                horizontalalignment='center',
                verticalalignment='center',
                # transform = ax.transAxes,
                fontweight=mid_fontweight)
    
    
        
    ax6.set_xlabel('Base Pair Step Position along Nucleosome',fontsize=label_fontsize,fontweight=label_fontweight)
    ax6.xaxis.set_label_coords(0.5,-0.18)

    ##############################################
    # Panel Labels
    ax1.text(-0.06, 1.02, '(a)',
        fontsize=panel_label_fontsize,
        horizontalalignment='center',
        verticalalignment='center',
        transform = ax1.transAxes,
        fontweight=panel_label_fontweight)
    
    for ax in axes:
        
        ###############################
        # set grid
        ax.set_axisbelow(True)
        ax.set_facecolor('#F2F2F2')
        # Remove border around plot.
        # [ax.spines[side].set_visible(False) for side in ax.spines]
        # Style the grid.
        ax.grid(which='major', color='#FEFEFE', linewidth=0.8)
        # ax.grid(which='minor', color='#FEFEFE', linewidth=0.2)
        
        ###############################
        # set major and minor ticks
        # ax.tick_params(axis="both",which='major',direction="in",width=axtick_major_width,length=axtick_major_length,labelsize=tick_labelsize,pad=tick_pad,)
        # ax.tick_params(axis='both',which='minor',direction="in",width=axtick_minor_width,length=axtick_minor_length)
        ax.tick_params(axis="both",which='major',direction="in",width=axtick_major_width,length=axtick_major_length,labelsize=tick_labelsize,pad=tick_pad,color='#cccccc')
        ax.tick_params(axis='both',which='minor',direction="in",width=axtick_minor_width,length=axtick_minor_length,color='#cccccc')


        ###############################
        ax.xaxis.set_ticks_position('both')
        # set ticks right and top
        ax.yaxis.set_ticks_position('both')
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(axlinewidth)
            ax.spines[axis].set_color('grey')
            ax.spines[axis].set_alpha(0.1)


    ax1.tick_params(axis="both",which='major',direction="in",width=axtick_major_width,length=axtick_major_length,labelsize=tick_labelsize,pad=0.4)

    plt.subplots_adjust(left=0.1,
                        right=0.98,
                        bottom=0.05,
                        top=0.97,
                        wspace=6,
                        hspace=0.25)

    if savefn is None:
        plt.show()
    else:
        fig.savefig(savefn+'.png',dpi=300,transparent=False)
        if vectorfig:
            fig.savefig(savefn+'.pdf',dpi=300,transparent=True)
            fig.savefig(savefn+'.svg',dpi=300,transparent=True)
        plt.close()

######################################################################################################################################################
# Fit binding model
######################################################################################################################################################

class SymmetricTriads:

    def __init__(
        self,
        mu0_init: np.ndarray,
        ):
        self.mu0_init = np.copy(mu0_init)
        self.mu0_base = self.symmetrize_mu0(self.mu0_init)
        
        self.num_independent = int(np.ceil(len(self.mu0_base) / 2))
        self.deform_shape = (self.num_independent,6)
        self.dof          = self.num_independent*6
        self.even = self.num_independent % 2 == 0
    
    def reverse_taus(self,taus):
        rtaus = np.copy(taus)
        rtaus[:,:,1:3] *= -1
        return rtaus
    
    def reverse_tau(self,tau):
        rtau = np.copy(tau)
        rtau[:,1:3] *= -1
        return rtau
    
    
    #######################################################
    # Symmetrize mu0
    
    def center_mu0(self,mu0):
        mu0[:,:3,3] = mu0[:,:3,3] - np.mean(mu0[:,:3,3],axis=0)
        return mu0
    
    def symmetrize_mu0(self,mu0):   
        # center mu0
        mu0 = self.center_mu0(mu0)
        # mirror mu0
        smu0 = self.mirror_taus(mu0)
        # average pairs
        avgmu0 = np.zeros(mu0.shape)
        for i in range(len(mu0)):
            # calculate half rotation
            Rh = so3.euler2rotmat(0.5*so3.rotmat2euler(mu0[i,:3,:3].T @ smu0[i,:3,:3]))
            # mid rotated triad
            avgmu0[i,:3,:3] = mu0[i,:3,:3] @ Rh
            # mid position
            avgmu0[i,:3,3] = 0.5*(mu0[i,:3,3] + smu0[i,:3,3])
            avgmu0[i,3,3] = 1
        return avgmu0
        
    def symmetry_axis(self,pts):
        pts = pts - np.mean(pts,axis=0)
        Npairs = len(pts)//2
        pairs = np.array([(i, len(pts)-1-i) for i in range(Npairs)]) 
                
        diff = pts[pairs[:, 0]] - pts[pairs[:, 1]]
        diff_centered = diff - np.mean(diff, axis=0)

        _, _, vh = np.linalg.svd(diff_centered)
        rotation_axis = vh[-1]
        rotation_axis /= np.linalg.norm(rotation_axis)
        return rotation_axis

    def mirror_taus(self,taus):
        pts = taus[:,:3,3]
        rotax = self.symmetry_axis(pts)
        rtaus = np.copy(taus)
        R = sp.spatial.transform.Rotation.from_rotvec(np.pi * rotax).as_matrix()
        rtaus[:,:3,3] = (R @ rtaus[:,:3,3].T).T
        rtaus[:,:3,:3] = (R @ rtaus[:,:3,:3].T).T
        rtaus = rtaus[::-1]
        rtaus[:,:3,1:3] *= -1
        return rtaus  
        
    def deform2mu0(self,deform):
        if deform.shape != self.deform_shape:
            raise ValueError(f'Dimensional mismatch of provided deforms')
                
        mu0 = np.zeros(self.mu0_base.shape)
        if self.even:
            for i in range(self.num_independent):
                gdeform = so3.se3_euler2rotmat(deform[i])
                mu0[i]            = self.mu0_base[i] @ gdeform
                mu0[len(mu0)-1-i] = self.reverse_tau(
                    self.reverse_tau(self.mu0_base[len(mu0)-1-i]) @ gdeform
                    )
        else:
            for i in range(self.num_independent-1):
                gdeform = so3.se3_euler2rotmat(deform[i])
                mu0[i]            = self.mu0_base[i] @ gdeform
                mu0[len(mu0)-1-i] = self.reverse_tau(
                    self.reverse_tau(self.mu0_base[len(mu0)-1-i]) @ gdeform
                    )
            mu0[self.num_independent-1] = self.mu0_base[self.num_independent-1] @ so3.se3_euler2rotmat(deform[self.num_independent-1])
        # self.check_symmetry(mu0,deform=deform)
        #######################################
        return mu0
            
    def check_symmetry(self,mu0,eps=1e-6,deform=None):
        ta = mu0
        tb = self.reverse_taus(ta)[::-1]
        
        ga = np.zeros((len(ta)-1,4,4))
        gb = np.zeros((len(ta)-1,4,4))
        for i in range(len(ga)):
            ga[i] = np.linalg.inv(ta[i]) @ ta[i+1]
            gb[i] = np.linalg.inv(tb[i]) @ tb[i+1]
        diff = np.abs((ga-gb)).sum()
        if diff > eps:            
            raise ValueError(f'mu0s not symmetric. Difference = {diff}')
     
######################################################################################################################################################
######################################################################################################################################################
######################################################################################################################################################
 
def reverse_taus(taus):
    rtaus = np.copy(taus)
    rtaus[:,:,1:3] *= -1
    return rtaus
    
def reverse_tau(tau):
    rtau = np.copy(tau)
    rtau[:,1:3] *= -1
    return rtau


def eval_datadicts(params,args):
    datadicts = args['datadicts']
    symtriads = args['symtriads']
    
    mparaps = np.zeros(84)
    mparaps[:12] = params[:12]
    kparams = params[12:]
    
    mu0_deform = mparaps[:symtriads.dof].reshape(symtriads.deform_shape)
    mu0 = symtriads.deform2mu0(mu0_deform)
    
    fkparams = np.ones(6)
    fkparams[:3] = 10
    fkparams[3:] = 100
    
    diags = [fkparams for i in range(len(mu0))] 
    diags[0] = kparams
    diags[1] = kparams
    diags[-1] = kparams
    diags[-2] = kparams
    diags = np.concatenate(diags)
    
    K = np.diag(diags)
    for datadict in datadicts:
        nucout = binding_model_free_energy(
            datadict['Ys'],
            datadict['M'],    
            mu0,
            K,
            left_open=0,
            right_open=0,
            use_correction=True,
        )
        datadict['nucout'] = nucout
    return datadicts

def eval_sqdiff(params,args):
    datadicts = eval_datadicts(params,args)
    sqdiff = 0
    for datadict in datadicts:
        helicoidal_theory = build_helicoidal(datadict['Ys'],datadict['nucout']['gs'])[2:-2]
        helicoidal_pdb = datadict['X_pdb']

        # plot_helicoidal(helicoidal_pdb,helicoidal_theory,shift_midids=-2)
        diff = helicoidal_theory - helicoidal_pdb
        for d in range(6):
            sqdiff += np.dot(diff[:,d],diff[:,d])*args['helicoidal_weights'][d]
    return sqdiff

def eval_total_free_energy(params,args):
    datadicts = eval_datadicts(params,args)
    Ftot = 0
    for datadict in datadicts:
        Ftot += datadict['nucout']['F']
    return Ftot


def helicoidal_residue(params,args):

    print(params)
    symtriads = args['symtriads']
    # mu0_deform = params[:symtriads.dof].reshape(symtriads.deform_shape)
    # mu0 = symtriads.deform2mu0(mu0_deform)
    diags = np.concatenate([params[12:] for i in range(len(args['mu0_init']))])
    
    # mu0 = args['mu0_init']
    # diags = np.concatenate([params for i in range(len(mu0))])
    # K = np.diag(diags)
    
    datadicts = eval_datadicts(params,args)
    
    sqdiff = 0
    for datadict in datadicts:        
        helicoidal_theory = build_helicoidal(datadict['Ys'],datadict['nucout']['gs'])[2:-2]
        helicoidal_pdb = datadict['X_pdb']
        # plot_helicoidal(helicoidal_pdb,helicoidal_theory,shift_midids=-2)
        diff = helicoidal_theory - helicoidal_pdb
        for d in range(6):
            sqdiff += np.dot(diff[:,d],diff[:,d])*args['helicoidal_weights'][d]
    
    Ftot = 0
    for datadict in datadicts:
        Ftot += datadict['nucout']['F']
    
    diff = args['weight_heldiff'] * sqdiff + args['weight_fe'] * Ftot
    
    
    outfn = 'PDBData/2triads_'
    if args['iter'] % 50 == 0:
        for datadict in datadicts:
            helicoidal_theory = build_helicoidal(datadict['Ys'],datadict['nucout']['gs'])[2:-2]
            helicoidal_pdb = datadict['X_pdb']
            savefn = outfn + datadict['pdb'] + f'_{args["iter"]}'
            plot_helicoidal(helicoidal_pdb,helicoidal_theory,savefn=savefn,shift_midids=-2)  
    
      
    args['iter'] += 1
    # if args['iter'] % args['print_every'] == 0:
    if True:
        print(f'Iteration {args["iter"]}, diff: {diff}, current sqdiff: {sqdiff}, Ftot: {Ftot}, Kdiags {diags[:6]}')
    return diff



def fit_binding_model(
    datadicts,
    mu0_init,
    K1_diag_init=np.ones(6),
    helicoidal_weights=np.ones(6),
    weight_heldiff:float=2,
    weight_fe:float=1,
    rbp_method='hybrid',
    tol = 1e-8):

    symtriads = SymmetricTriads(mu0_init)
    # mu0_base  = symtriads.mu0_base
    
    genstiff = GenStiffness(method=rbp_method)
    #############################
    # Prepare PDB data
    
    for datadict in datadicts:
    
        # extend sequence to 147 bp
        seq = 'CG' + datadict['bound_seq'] + 'CG'
        
        # generate RPB stiffness and groundstate
        M,Ys = genstiff.gen_params(seq,use_group=True)
        datadict['M']  = M
        datadict['Ys'] = Ys
        
        # set PDB full and dynamic parameters
        datadict['X_pdb']  = np.copy(datadict['bound_helicoidal'])
        datadict['Yd_pdb'] = helicoidal_full2dynamic(datadict['X_pdb'],Ys[2:-2])
        
    #############################
    # prepare fit parameters
    
    # initialize params
    params = np.zeros((symtriads.dof+6))
    params[-6:] = K1_diag_init
    
    params = np.zeros(18)
    params[-6:] = K1_diag_init
    
    
    # params = params[-6:]
    
    
    args = {
        'symtriads' : symtriads,
        'datadicts' : datadicts,
        'helicoidal_weights' : helicoidal_weights,
        'weight_heldiff' : weight_heldiff,
        'weight_fe' :      weight_fe,
        'print_every' : len(params),
        'iter'      : 0,
        'mu0_init'  : mu0_init
    }
    
    weight_heldiff /= eval_sqdiff(params,args)
    weight_fe /= eval_total_free_energy(params,args)
    
    print(f'{weight_heldiff=}')
    print(f'{weight_fe=}')
    args['weight_heldiff']  =  weight_heldiff
    args['weight_fe']       =  weight_fe
    
    helicoidal_residue(params,args)
    # res = sp.optimize.minimize(helicoidal_residue, params, method='L-BFGS-B',args=(args,),options={'maxiter':10000000},tol=tol)
    # res = sp.optimize.minimize(helicoidal_residue, params, method='Nelder-Mead',args=(args,),options={'maxiter':10000000},tol=tol)
    
    bounds = np.zeros((18,2))
    bounds[:12,0] = -np.pi/2
    bounds[:12,1] = np.pi/2
    
    bounds[12:,0] = 0.1
    bounds[12:,1] = 1000
    
    res = sp.optimize.minimize(helicoidal_residue, params, method='Powell',args=(args,),options={'maxiter':10000000},tol=tol,bounds=bounds)
    fitparams = res.x
    # fitparams = params
    
    mparams = np.zeros(84)
    mparams[:12] = fitparams[:12]
    
    mu0_deform = mparams.reshape(symtriads.deform_shape)
    mu0 = symtriads.deform2mu0(mu0_deform)
    K1diag = fitparams[-6:]
    
    
    datadicts = eval_datadicts(fitparams,args)
    outfn = 'PDBData/2triads_'
    
    for datadict in datadicts:
        helicoidal_theory = build_helicoidal(datadict['Ys'],datadict['nucout']['gs'])[2:-2]
        helicoidal_pdb = datadict['X_pdb']

        savefn = outfn + datadict['pdb']
        plot_helicoidal(helicoidal_pdb,helicoidal_theory,savefn=savefn,shift_midids=-2)
    
    
    
    return mu0,K1diag,fitparams


if __name__ == '__main__':
    
    datafn = 'PDBData/canonical_matches'
    df = pd.read_pickle(datafn)
    
    MIDSTEP_LOCATIONS = [
        2, 6, 14, 17, 24, 29, 
        34, 38, 45, 49, 55, 59, 
        65, 69, 76, 80, 86, 90, 
        96, 100, 107, 111, 116, 121, 
        128, 131, 139, 143
    ]
    
    genstiff = GenStiffness(method='hybrid')
    triadfn = 'methods/State/Nucleosome.state'
    nuctriads = read_nucleosome_triads(triadfn)
    
    initial_nuc_mu0 = calculate_midstep_triads(
        MIDSTEP_LOCATIONS,
        nuctriads
    )
    
    datadicts = [df.iloc[i].to_dict() for i in range(len(df))]

    # filter pdb
    selected_pdb = ['2nzd','3uta','3ut9','7ohc','1kx5','6wz5']
    selected_pdb = ['3uta','3ut9','7ohc','1kx5','6wz5']
    selected_pdb = ['6wz5','7ohc']
    
    seldatadicts = [datadict for datadict in datadicts if datadict['pdb'].lower() in selected_pdb]
    for datadict in seldatadicts:
        print(datadict['pdb'])
    
    K1_diag_init = np.ones(6)*10
    K1_diag_init[3:] = 100
    
    helicoidal_weights = np.ones(6)
    # weights[0] = 0.5
    # weights[1] = 0.5
    # weights[2] = 0.5
    # weights[3] = 0.2
    # weights[4] = 0.2
    # weights[5] = 0.2
    
    # weights[0] = 1
    # weights[1] = 2
    # weights[2] = 1
    # weights[3] = 0.2
    # weights[4] = 0.2
    # weights[5] = 0.2
    
    weight_heldiff = 1
    weight_fe      = 0
    
    mu0,K1diag,fitparams = fit_binding_model(seldatadicts,initial_nuc_mu0,K1_diag_init,rbp_method='hybrid',helicoidal_weights=helicoidal_weights,weight_heldiff=weight_heldiff,weight_fe=weight_fe)
    
    outfn = 'PDBData/2triads_fit'
    
    np.save(outfn+'_mu0',mu0)
    np.save(outfn+'_K1diag',K1diag)
    np.save(outfn+'_fitparams',fitparams)
    
    