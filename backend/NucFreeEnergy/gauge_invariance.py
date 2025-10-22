import sys, os
num_threads = 4
os.environ["OMP_NUM_THREADS"] = f"{num_threads}"    # For libraries using OpenMP
os.environ["MKL_NUM_THREADS"] = f"{num_threads}"    # For Intel MKL
os.environ["OPENBLAS_NUM_THREADS"] = f"{num_threads}"  # For OpenBLAS
os.environ["BLIS_NUM_THREADS"] = f"{num_threads}"   # For BLIS

import numpy as np
import random
import matplotlib.pyplot as plt
from  methods import nucleosome_free_energy, nucleosome_groundstate, read_nucleosome_triads, GenStiffness, calculate_midstep_triads
from binding_model import binding_model_free_energy, binding_model_free_energy_old
from methods.PolyCG.polycg import cgnaplus_bps_params


def plot_dF(seqsdata,refs,gccont,seqs,savefn,labels=None,subtract_free=False):

    def cm_to_inch(cm: float) -> float:
        return cm/2.54

    fig_width = 8.6
    fig_height = 10

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

    fig = plt.figure(figsize=(cm_to_inch(fig_width), cm_to_inch(fig_height)), dpi=300,facecolor='w',edgecolor='k') 
    axes = []
    axes.append(fig.add_subplot(311))
    axes.append(fig.add_subplot(312))
    axes.append(fig.add_subplot(313))

    ##############################################
    ax1 = axes[0]
    ax2 = axes[1]
    ax3 = axes[2]

    markersize = 5
    linewidth  = 0.6

    # for i,K in enumerate(Kmats):
    for i in range(len(seqsdata[0])):
        
        if labels is None:
            label = f'K{i+1}'
        else:
            label = labels[i]
        
        marker = 'x'
        
        if not subtract_free:
            dFs = [seqdata[i]['F']-refs[i]['F'] for seqdata in seqsdata]
            dFentrs = [seqdata[i]['F_entropy']-refs[i]['F_entropy'] for seqdata in seqsdata]
            dFenths = [seqdata[i]['F_enthalpy']-refs[i]['F_enthalpy'] for seqdata in seqsdata]
        else:
            dFs = [seqdata[i]['F']-refs[i]['F'] - (seqdata[i]['F_freedna'] - refs[i]['F_freedna']) for seqdata in seqsdata]
            dFentrs = [seqdata[i]['F_entropy']-refs[i]['F_entropy']  - (seqdata[i]['F_freedna'] - refs[i]['F_freedna']) for seqdata in seqsdata]
            dFenths = [seqdata[i]['F_enthalpy']-refs[i]['F_enthalpy'] for seqdata in seqsdata]
        
        ax1.plot(gccont,dFs,lw=linewidth,zorder=1,alpha=0.5)
        if marker == 'x':
            ax1.scatter(gccont,dFs,s=markersize,marker=marker,zorder=2,alpha=0.75,label=label,linewidth=0.5)
        else:
            ax1.scatter(gccont,dFs,s=markersize,edgecolors='white',linewidth=0.5*linewidth,marker=marker,zorder=2,alpha=0.75,label=label)
        
        ax2.plot(gccont,dFentrs,lw=linewidth,zorder=1,alpha=0.5)
        if marker == 'x':
            ax2.scatter(gccont,dFentrs,s=markersize,marker=marker,zorder=2,alpha=0.75,label=label,linewidth=0.5)
        else:
            ax2.scatter(gccont,dFentrs,s=markersize,edgecolors='white',linewidth=0.5*linewidth,marker=marker,zorder=2,alpha=0.75,label=label)
        
        ax3.plot(gccont,dFenths,lw=linewidth,zorder=1,alpha=0.5)
        if marker == 'x':
            ax3.scatter(gccont,dFenths,s=markersize,marker=marker,zorder=2,alpha=0.75,label=label,linewidth=0.5)
        else:
            ax3.scatter(gccont,dFenths,s=markersize,edgecolors='white',linewidth=0.5*linewidth,marker=marker,zorder=2,alpha=0.75,label=label)


    ax = axes[0]
    ax.set_xlabel('GC Content',fontsize=label_fontsize,fontweight=label_fontweight)
    ax.set_ylabel(r'$\mathbf{\Delta F}$',fontsize=label_fontsize,fontweight=label_fontweight,rotation=90)
    ax = axes[1]
    ax.set_xlabel('GC Content',fontsize=label_fontsize,fontweight=label_fontweight)
    ax.set_ylabel(r'$\mathbf{\Delta F_{\mathrm{entropy}}}$',fontsize=label_fontsize,fontweight=label_fontweight,rotation=90)
    ax = axes[2]
    ax.set_xlabel('GC Content',fontsize=label_fontsize,fontweight=label_fontweight)
    ax.set_ylabel(r'$\mathbf{\Delta F_{\mathrm{enthalpy}}}$',fontsize=label_fontsize,fontweight=label_fontweight,rotation=90)

    for ax in axes:
        ax.xaxis.set_label_coords(0.5,-0.1)
        ax.yaxis.set_label_coords(-0.055,0.5)
        ax.set_xlim([0,1])

    ax1.legend(fontsize=legend_fontsize,borderpad=0.2,framealpha=0.5,fancybox=True,handletextpad=0.2,loc='lower left', bbox_to_anchor=(0.00,1.00),ncol=8,columnspacing=0.7)

    ##############################################
    # Axes configs
    for ax in axes:
        ###############################
        # set major and minor ticks
        ax.tick_params(axis="both",which='major',direction="in",width=axtick_major_width,length=axtick_major_length,labelsize=tick_labelsize,pad=tick_pad,color='#cccccc')
        ax.tick_params(axis='both',which='minor',direction="in",width=axtick_minor_width,length=axtick_minor_length,color='#cccccc')

        ###############################
        ax.xaxis.set_ticks_position('both')
        # set ticks right and top
        ax.yaxis.set_ticks_position('both')
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(axlinewidth)
            ax.spines[axis].set_color('grey')
            ax.spines[axis].set_alpha(0.7)
            
    plt.subplots_adjust(left=0.09,
                        right=0.98,
                        bottom=0.05,
                        top=0.95,
                        wspace=6,
                        hspace=0.25)
            
    fig.savefig(savefn+'.pdf',dpi=300,transparent=True)
    fig.savefig(savefn+'.svg',dpi=300,transparent=True)
    fig.savefig(savefn+'.png',dpi=300,transparent=False)



def random_seq(N,gc=0.5):
    NGC = int(N*gc)
    NAT = N-NGC
    seqlist = ['AT'[np.random.randint(2)] for i in range(NAT)] + ['CG'[np.random.randint(2)] for i in range(NGC)]
    random.shuffle(seqlist)
    seq = ''.join(seqlist)
    return seq


if __name__ == '__main__':

    method = 'crystal'
    # method = 'hybrid'
    method = 'cgNAplus'

    genstiffmethods = ['crystal','hybrid']

    if method.lower() in genstiffmethods:
        genstiff = GenStiffness(method=method)   # alternatively you can use the 'crystal' method for the Olson data
    
    
    triadfn = 'methods/State/Nucleosome.state'
    nuctriads = read_nucleosome_triads(triadfn)

    midstep_constraint_locations = [
        2, 6, 14, 17, 24, 29, 
        34, 38, 45, 49, 55, 59, 
        65, 69, 76, 80, 86, 90, 
        96, 100, 107, 111, 116, 121, 
        128, 131, 139, 143
    ]

    # FOR NOW WE USE THE FIXED MIDSTEP TRIADS AS MU_0
    # Find midstep triads in fixed framework for comparison
    nuc_mu0 = calculate_midstep_triads(
        midstep_constraint_locations,
        nuctriads
    )
    
    ######################
    # ref vals
    seq601  = "CTGGAGAATCCCGGTGCCGAGGCCGCTCAATTGGTCGTAGACAGCTCTAGCACCGCTTAAACGCACGTACGCGCTGTCCCCCGCGTTTTAACCGCCAAGGGGATTACTCCCTAGTCTCCAGGCACGTGTCAGATATATACATCCTGT"
    refseq = seq601

    # Nseqs = 100
    # seqs = [''.join(['ATCG'[np.random.randint(4)] for i in range(147)]) for j in range(Nseqs)]

    Nseqs = 101
    # Nseqs = 5
    seqs = []
    gccont = []
    for i in range(Nseqs):
        gc = i/(Nseqs-1)
        seq = random_seq(147,gc)
        seqs.append(seq)
        gccont.append(gc)
    

    #########################################################################
    factors = [1000,100,10,1,0.1,0.01,0.001]
    factors = [10,1,0.1]
    base_savefn = f'Figs/GaugeInvariance_{method}'
    
    wrstates = [0,4,8,12] 
    wrstates = [0,8] 
    #########################################################################



    #################
    # K1-10
    Kentries = np.array([1,1,1,10,10,10])
    diags = np.concatenate([Kentries]*len(nuc_mu0))
    Kbase = np.diag(diags)
    basename = 'K1-10'

    #################
    # K_pos_rescaled
    
    # Kmd_comb     = np.load('MDParams/nuc_K_comb.npy')
    # Kmd_raw     = np.load('MDParams/nuc_K.npy')
    # Kmd_pos = np.load('MDParams/nuc_K_pos.npy')
    # Kmd_pos_resc = np.load('MDParams/nuc_K_posresc.npy')
        
    # Kbase = Kmd_pos_resc
    # basename = 'Kposresc'


    ##########################

    Kmats = []
    for fac in factors:
        Kmats.append(Kbase*fac)
        

    #########################################################################

    labels = [f'{fac}' for fac in factors]



    for nopen in wrstates:
        
        left_open  = nopen
        right_open = nopen

        if method.lower() in genstiffmethods:
            stiffmat,groundstate = genstiff.gen_params(refseq,use_group=True)
        elif method.lower() in ['cgnaplus']:
            groundstate,stiffmat = cgnaplus_bps_params(refseq,group_split=True)
        else:
            raise ValueError(f'Unknown Method {method}')
        
        refs = []
        for K in Kmats:
            nucout = binding_model_free_energy(
                groundstate,
                stiffmat,    
                nuc_mu0,
                K,
                left_open=left_open,
                right_open=right_open,
                use_correction=True,
            )
            refs.append(nucout)
            
        seqsdata = []
        for i,seq in enumerate(seqs):
            print(i)
            if method.lower() in genstiffmethods:
                stiffmat,groundstate = genstiff.gen_params(seq,use_group=True)
            elif method.lower() in ['cgnaplus']:
                groundstate,stiffmat = cgnaplus_bps_params(seq,group_split=True)
            else:
                raise ValueError(f'Unknown Method {method}')
            seqdata = []
            for K in Kmats:
                nucout = binding_model_free_energy(
                    groundstate,
                    stiffmat,    
                    nuc_mu0,
                    K,
                    left_open=left_open,
                    right_open=right_open,
                    use_correction=True,
                )
                seqdata.append(nucout)
            seqsdata.append(seqdata)
        
        savefn = f'{base_savefn}_{basename}_l{right_open}_r{right_open}'
        plot_dF(seqsdata,refs,gccont,seqs,savefn,labels=labels)

    ########################################################################################################################################################
    ########################################################################################################################################################
    ########################################################################################################################################################

    #################
    # K_pos_rescaled
    
    Kmd_comb     = np.load('MDParams/nuc_K_comb.npy')
    Kmd_raw     = np.load('MDParams/nuc_K.npy')
    Kmd_pos = np.load('MDParams/nuc_K_pos.npy')
    Kmd_pos_resc = np.load('MDParams/nuc_K_posresc.npy')
        
    Kbase = Kmd_pos_resc
    basename = 'Kposresc'


    ##########################

    Kmats = []
    for fac in factors:
        Kmats.append(Kbase*fac)
        

    #########################################################################

    labels = [f'{fac}' for fac in factors]

    ######################
    # ref vals
    seq601  = "CTGGAGAATCCCGGTGCCGAGGCCGCTCAATTGGTCGTAGACAGCTCTAGCACCGCTTAAACGCACGTACGCGCTGTCCCCCGCGTTTTAACCGCCAAGGGGATTACTCCCTAGTCTCCAGGCACGTGTCAGATATATACATCCTGT"
    refseq = seq601


    for nopen in wrstates:
        
        left_open  = nopen
        right_open = nopen

        # stiffmat,groundstate = genstiff.gen_params(refseq,use_group=True)
        if method.lower() in genstiffmethods:
            stiffmat,groundstate = genstiff.gen_params(refseq,use_group=True)
        else:
            groundstate,stiffmat = cgnaplus_bps_params(refseq,group_split=True)
        
        refs = []
        for K in Kmats:
            nucout = binding_model_free_energy(
                groundstate,
                stiffmat,    
                nuc_mu0,
                K,
                left_open=left_open,
                right_open=right_open,
                use_correction=True,
            )
            refs.append(nucout)
            
        seqsdata = []
        for i,seq in enumerate(seqs):
            print(i)
            
            if method.lower() in genstiffmethods:
                stiffmat,groundstate = genstiff.gen_params(seq,use_group=True)
            else:
                groundstate,stiffmat = cgnaplus_bps_params(seq,group_split=True)
                
            seqdata = []
            for K in Kmats:
                nucout = binding_model_free_energy(
                    groundstate,
                    stiffmat,    
                    nuc_mu0,
                    K,
                    left_open=left_open,
                    right_open=right_open,
                    use_correction=True,
                )
                seqdata.append(nucout)
            seqsdata.append(seqdata)
        
        savefn = f'{base_savefn}_{basename}_l{right_open}_r{right_open}'
        plot_dF(seqsdata,refs,gccont,seqs,savefn,labels=labels)


