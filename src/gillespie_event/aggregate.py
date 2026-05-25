"""Per-nucleosome aggregation of replicate results."""

from dataclasses import dataclass, field
from typing import List
import math
import numpy as np

from src.core.gillespie_event_simulator import ReplicateResult
from src.core.nucleosomes import Nucleosome

# numpy >= 2.0 renamed `trapz` to `trapezoid`. Pick whichever is available so
# the pipeline runs on both old (cluster image legacy) and new numpy.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz


@dataclass
class NucleosomeAggregate:
    # Identity
    id:                  str
    subid:               int
    n_replicates:        int

    # First-passage statistics
    mfpt_uncensored:     float
    rmst:                float
    half_life:           float
    censored_fraction:   float
    final_survival:      float
    n_events_total:      int

    # Ensemble means — mean-of-means
    mean_n_open_mom:     float
    mean_n_open_mom_std: float
    mean_bprot_mom:      float
    mean_bprot_mom_std:  float

    # Ensemble means — total time-weighted
    mean_n_open_tw:      float
    mean_bprot_tw:       float

    # Survival on common grid
    tau_grid:            np.ndarray
    survival:            np.ndarray

    # Raw detach times
    detach_times:        np.ndarray

    # Per-replicate trajectories
    traj_tau:            List[np.ndarray] = field(default_factory=list)
    traj_n_closed:       List[np.ndarray] = field(default_factory=list)


def _empirical_survival(detach_times: np.ndarray, tau_grid: np.ndarray) -> np.ndarray:
    """S(tau) = fraction of replicates whose effective lifetime exceeds tau.
    Censored replicates have detach_time = NaN; treat their lifetime as +inf
    (always alive).
    """
    # For each grid point, count replicates with detach_time > tau or NaN.
    # Vectorize over both axes.
    dt = detach_times[:, None]                          # shape (N_rep, 1)
    grid = tau_grid[None, :]                            # shape (1, N_grid)
    alive = np.isnan(dt) | (dt > grid)                  # shape (N_rep, N_grid)
    return alive.mean(axis=0)                           # shape (N_grid,)


def _half_life(survival: np.ndarray, tau_grid: np.ndarray) -> float:
    """Smallest tau with S(tau) <= 0.5; NaN if S never crosses."""
    below = np.where(survival <= 0.5)[0]
    if below.size == 0:
        return math.nan
    return float(tau_grid[below[0]])


def aggregate_replicates(
    nuc: Nucleosome,
    results: List[ReplicateResult],
    tau_grid: np.ndarray,
) -> NucleosomeAggregate:
    n_rep = len(results)
    if n_rep == 0:
        raise ValueError("Cannot aggregate empty replicate list")

    detach_times = np.array([r.detach_tau for r in results], dtype=np.float64)
    final_taus = np.array([r.final_tau for r in results], dtype=np.float64)
    mean_n_open = np.array([r.mean_n_open for r in results], dtype=np.float64)
    mean_bprot = np.array([r.mean_bprot for r in results], dtype=np.float64)
    censored = np.array([r.censored for r in results], dtype=bool)

    # First-passage stats
    survival = _empirical_survival(detach_times, tau_grid)
    rmst = float(_trapezoid(survival, tau_grid))
    half_life = _half_life(survival, tau_grid)
    censored_fraction = float(censored.mean())
    final_survival = float(survival[-1])

    if censored.all():
        mfpt_uncensored = math.nan
    else:
        mfpt_uncensored = float(np.nanmean(detach_times))

    # Ensemble means
    mean_n_open_mom = float(mean_n_open.mean())
    mean_n_open_mom_std = float(mean_n_open.std(ddof=0))
    mean_bprot_mom = float(mean_bprot.mean())
    mean_bprot_mom_std = float(mean_bprot.std(ddof=0))

    total_time = float(final_taus.sum())
    if total_time > 0:
        mean_n_open_tw = float((mean_n_open * final_taus).sum() / total_time)
        mean_bprot_tw = float((mean_bprot * final_taus).sum() / total_time)
    else:
        mean_n_open_tw = 0.0
        mean_bprot_tw = 0.0

    n_events_total = int(sum(
        sum(r.n_events_by_type.values()) for r in results
    ))

    return NucleosomeAggregate(
        id=str(nuc.id),
        subid=int(nuc.subid),
        n_replicates=n_rep,
        mfpt_uncensored=mfpt_uncensored,
        rmst=rmst,
        half_life=half_life,
        censored_fraction=censored_fraction,
        final_survival=final_survival,
        n_events_total=n_events_total,
        mean_n_open_mom=mean_n_open_mom,
        mean_n_open_mom_std=mean_n_open_mom_std,
        mean_bprot_mom=mean_bprot_mom,
        mean_bprot_mom_std=mean_bprot_mom_std,
        mean_n_open_tw=mean_n_open_tw,
        mean_bprot_tw=mean_bprot_tw,
        tau_grid=np.asarray(tau_grid, dtype=np.float64),
        survival=survival,
        detach_times=detach_times,
        traj_tau=[r.traj_tau for r in results],
        traj_n_closed=[r.traj_n_closed for r in results],
    )
