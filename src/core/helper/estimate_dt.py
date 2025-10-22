from src.core.nucleosomes import Nucleosomes
from src.core.protamine import protamines
from src.core.gillespie_simulator import GillespieSimulator
from src.utils.logger_util import get_logger
from src.config.custom_type import SimulationState
import numpy as np

def estimate_timescales(nuc_inst: Nucleosomes, 
                        prot_inst: protamines,
                          pilot_steps: int = 100, 
                            t_max: float = 10.0,
                          logger=None):
    """
    Run a pilot simulation to estimate characteristic timescales.
    Returns average tau (1/total_rate) and decorrelation time (time for n_closed to change by ~1 site on average).
    """
    if logger is None:
        logger = get_logger(__name__, log_file=None, level='INFO')

    pilot_sim = GillespieSimulator(nuc_inst=nuc_inst, prot_inst=prot_inst, t_points=None, max_steps=pilot_steps)

    total_rates = []
    n_closed_evol = []  # Average closed sites over nucleosomes
    times = []
    
    for state in pilot_sim.run_steps():
        rates = [pilot_sim.calculate_rates(i) for i in range(pilot_sim.num_nuc)]
        total_rate = sum(sum(r.total.values()) for r in rates)
        total_rates.append(total_rate)
        avg_closed = state.cs_total / pilot_sim.num_nuc
        n_closed_evol.append(avg_closed)
        times.append(state.time)
        logger.info(f"Time: {state.time:.2f}, Avg Closed Sites: {avg_closed:.2f}, Total Rate: {total_rate:.2e}")

    if not total_rates:
        raise ValueError("Pilot run had no reactions.")
    
    avg_tau = 1 / np.mean(total_rates)  # Average time between reactions
    
    # Estimate decorrelation time: Time for average closed sites to decrease by 1
    if len(n_closed_evol) > 1:
        delta_closed = np.diff(n_closed_evol)
        delta_t = np.diff(times)
        rate_change = np.mean(np.abs(delta_closed / delta_t))  # Avg rate of change in closed sites/s
        tau_decorr = 1 / rate_change  # Time for ~1 site change
    else:
        tau_decorr = avg_tau * 10  
    
    logger.info(f"Pilot: Avg tau={avg_tau:.2e} s, Decorr tau={tau_decorr:.2e} s")

    return avg_tau, tau_decorr
            

def reset_simulation(nuc_inst: Nucleosomes, prot_inst: protamines):
    """
    Reset nucleosome states to initial (fully wrapped).
    """
    prot_inst.N_bound = 0
    prot_inst.P_free = prot_inst.p_conc * 6e23 * 1e-15


    for nuc in nuc_inst.nucs:
        nuc.state.fill(0)
        nuc.n_closed = nuc.binding_sites