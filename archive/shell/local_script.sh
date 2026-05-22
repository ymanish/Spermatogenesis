
TASK_ID_PADDED=001

for PROM_TYPE in "unboundprom" "boundprom"; do
    HAMNUC_DATA_DIR="/home/pol_schiessel/maya620d/pol/Projects/Codebase/Spermatogensis/hamnucret_data/$PROM_TYPE/breath_energy"
    INFILE="$HAMNUC_DATA_DIR/$TASK_ID_PADDED.tsv"  
    STORAGE_DIR="/home/pol_schiessel/maya620d/pol/Projects/Codebase/Spermatogensis/output/$PROM_TYPE/GSim"

    BATCH_SIZE=10  ## int: id+subid

    K_WRAP=1.0     ## float
    BINDING_SITES=14   ## int
    INF_PROTAMINE="true"  ### true/false
    PROT_K_UNBIND=0.01   ## float
    PROT_K_BIND=10.0     ## float
    PROT_P_CONC=0.001      ## float
    PROT_COOPERATIVITY=10.0  ## float

    T_STOP=5000.0    ## float
    T_NUM=10000      ## int

    SAVE_TRAJECTORIES="false"  ### true/false

    if [ "$INF_PROTAMINE" = "true" ]; then
        inf_arg="--inf_protamine"
    else
        inf_arg=""
    fi

    if [ "$SAVE_TRAJECTORIES" = "true" ]; then
        traj_arg="--save_trajectories"
    else
        traj_arg=""
    fi

    echo "-> Launching main worker script for $PROM_TYPE....."
    singularity exec \
        --bind $PWD:/project \
        nucleosome.sif \
        python3 /project/src/scripts/exec_sim.py \
            --infile "$INFILE" \
            --storage_dir "$STORAGE_DIR" \
            --batch_size "$BATCH_SIZE" \
            --n_workers 20 \
            --k_wrap "$K_WRAP" \
            --binding_sites "$BINDING_SITES" \
            $inf_arg \
            --prot_k_unbind "$PROT_K_UNBIND" \
            --prot_k_bind "$PROT_K_BIND" \
            --prot_p_conc "$PROT_P_CONC" \
            --prot_cooperativity "$PROT_COOPERATIVITY" \
            --t_stop "$T_STOP" \
            --t_num "$T_NUM" \
            $traj_arg

done