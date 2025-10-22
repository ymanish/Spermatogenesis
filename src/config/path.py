# src/config/path.py
# Author: MY
# Last Updated: 2025-03-25


from pathlib import Path
import datetime

RUN_DATE = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")



######LOCAL DIR#######

SRC_DIR = Path(__file__).parent.parent.parent
OUT_DIR = Path(__file__).parent.parent.parent
HAMNUCRET_DATA_DIR = Path("/home/pol_schiessel/maya620d/pol/Projects/Codebase/Spermatogensis/hamnucret_data")


######CLUSTER DIR#######

# SRC_DIR = Path('/home/pol_schiessel/maya620d/Spermatogensis')
# OUT_DIR = Path('/group/pol_schiessel/Manish/Spermatogensis')
# HAMNUCRET_DATA_DIR = Path("/group/pol_schiessel/Manish/HAMNucRetSeq_pipeline/output")


print(f"Source Directory: {SRC_DIR}")
print(f"Output Directory: {OUT_DIR}")

PARAM_DIR = SRC_DIR / "parameters"
RESULTS_DIR = OUT_DIR / "output"
print(f"HAMNUCRET Data Directory: {HAMNUCRET_DATA_DIR}")


###### Parameters ######
MD_param_loc = PARAM_DIR/"MD_stiffness.csv"
MD_param_loc = PARAM_DIR/"MMC_MD_stiffness.csv"

Olson_prob_di_loc = PARAM_DIR/"Olson_RoomTemp_dinucdist.csv"
Olson_prob_tri_loc = PARAM_DIR/"Olson_RoomTemp_trinucdist.csv"
Olson_prob_mono_loc = PARAM_DIR/"Olson_RoomTemp_nucdist.csv"
SAXS_loc = PARAM_DIR/"SAXS_fraction.csv"


###### Results ######
STATE_RECORD_DIR = RESULTS_DIR / RUN_DATE
BREATHING_DIR = RESULTS_DIR / "Breathing_plots"
SIMULATION_PLOTS_DIR = RESULTS_DIR / "simulation_plots"
NUCLEOSOME_STATE_RECORD_DIR = RESULTS_DIR / "Derivate_state_record"

