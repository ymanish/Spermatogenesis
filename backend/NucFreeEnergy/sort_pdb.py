import sys, os
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

# np.set_printoptions(linewidth=250,precision=5,suppress=True)

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
    K1_diag_init = np.ones(6)
    K1_diag_init[3:] = 10
    
    nucsets = [df.iloc[i].to_dict() for i in range(len(df))]
    
    seqs = []
    for i in range(len(nucsets)):
        # nucdata = df.iloc[i].to_dict()
        nucdata = nucsets[i]
        # #############################################################################################
        # # fix nucdata
        # patternlen = np.max(nucdata['pattern']) - np.min(nucdata['pattern']) + 2
        # nucdata['bound_taus'] = nucdata['taus'][nucdata['entry_id']:nucdata['entry_id']+patternlen]
        # nucdata['bound_seq'] = nucdata['seq'][nucdata['entry_id']:nucdata['entry_id']+patternlen]
        # #############################################################################################
        bound_seq = nucdata['bound_seq']
        bound_seq = bound_seq[2:-2]
        seqs.append(bound_seq)
    
    seqs_set = list(set(seqs))
    seqs_set = sorted(seqs_set)
    for seq in seqs_set:
        print('####################')
        print(seq)
        setnd = [nd for nd in nucsets if nd['bound_seq'][2:-2] == seq]
        print(', '.join([nd['pdb'] for nd in setnd]))
        
    print(len(seqs))

    for i in range(len(df)):
        nucdata = df.iloc[i].to_dict()
        print(f'evaluating {nucdata["pdb"]}')
        print(nucdata["bound_seq"])
        
        # #############################################################################################
        # # fix nucdata
        # patternlen = np.max(nucdata['pattern']) - np.min(nucdata['pattern']) + 2
        # nucdata['bound_taus'] = nucdata['taus'][nucdata['entry_id']:nucdata['entry_id']+patternlen]
        # nucdata['bound_seq'] = nucdata['seq'][nucdata['entry_id']:nucdata['entry_id']+patternlen]
        # #############################################################################################
        
        
        ###################
        seq = 'CG' + nucdata['bound_seq'] + 'CG'
        M,gs = genstiff.gen_params(seq,use_group=True)
        
        diags = np.ones(len(MIDSTEP_LOCATIONS)*6)*10
        diags[0::6] = 1
        diags[1::6] = 1
        diags[2::6] = 1
        K = np.diag(diags)


        nucout = binding_model_free_energy(
            gs,
            M,    
            initial_nuc_mu0,
            K,
            left_open=0,
            right_open=0,
            use_correction=True,
        )
        
        gs_th   = build_helicoidal(gs,nucout['gs'])[2:-2]
        gs_ref  = nucdata['helicoidal'][nucdata['entry_id']:nucdata['entry_id']+len(gs_th)]
        
        gs_th = helicoidal_full2dynamic(gs_th,gs)
        gs_ref = helicoidal_full2dynamic(gs_ref,gs)


        ##################################################
        # Save Figure
        savefn = 'Figs/Helicoidal_compare_' + nucdata["pdb"] + '_sc'

        plot_helicoidal(gs_ref,gs_th,savefn,shift_midids=2,vectorfig=False)


