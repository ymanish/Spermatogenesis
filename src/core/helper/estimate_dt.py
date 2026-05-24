from src.core.nucleosomes import Nucleosome
from src.core.protamine import protamines
from src.core.gillespie_simulator import GillespieSimulator
from src.utils.logger_util import get_logger
import numpy as np


def estimate_timescales(nuc_inst: Nucleosome,
                        prot_inst: protamines,
                        pilot_steps: int = 100,
                        t_max: float = 10.0,
                        logger=None):
    """
    Run a pilot simulation to estimate characteristic timescales.
    Returns average tau (1/total_rate) and decorrelation time.
    """
    if logger is None:
        logger = get_logger(__name__, log_file=None, level='INFO')

    tau_points = np.linspace(0, t_max, pilot_steps)
    pilot_sim = GillespieSimulator(nuc_inst=nuc_inst, prot_inst=prot_inst,
                                   t_points=None, max_steps=None, tau_points=tau_points)

    n_closed_evol = []
    times = []

    for state in pilot_sim.run():
        n_closed_evol.append(state.cs_total)
        times.append(state.time)
        logger.info(f"Time: {state.time:.2f}, Closed Sites: {state.cs_total}")

    if not times:
        raise ValueError("Pilot run had no states.")

    n_closed_evol = np.array(n_closed_evol)
    times = np.array(times)

    # Estimate average rate from spacing
    avg_tau = t_max / pilot_steps

    # Estimate decorrelation time from rate of change in closed sites
    if len(n_closed_evol) > 1:
        delta_t = np.diff(times)
        delta_closed = np.abs(np.diff(n_closed_evol.astype(float)))
        nonzero = delta_t > 0
        if nonzero.any():
            rate_change = np.mean(delta_closed[nonzero] / delta_t[nonzero])
            tau_decorr = 1.0 / rate_change if rate_change > 0 else avg_tau * 10
        else:
            tau_decorr = avg_tau * 10
    else:
        tau_decorr = avg_tau * 10

    logger.info(f"Pilot: Avg tau={avg_tau:.2e} s, Decorr tau={tau_decorr:.2e} s")
    return avg_tau, tau_decorr
