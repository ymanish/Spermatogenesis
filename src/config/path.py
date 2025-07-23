# src/config/path.py
# Author: MY
# Last Updated: 2025-03-25


from pathlib import Path
import datetime

RUN_DATE = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

SRC_DIR = Path(__file__).parent.parent.parent
PARAM_DIR = SRC_DIR / "parameters"
RESULTS_DIR = SRC_DIR / "outputs"

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

