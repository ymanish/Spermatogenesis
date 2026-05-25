"""Run a single replicate of the event-driven Gillespie simulator."""

import copy
from typing import Optional

from src.core.gillespie_event_simulator import (
    GillespieEventSimulator,
    ReplicateResult,
)
from src.core.nucleosomes import Nucleosome
from src.core.protamine import protamines
from src.config.var import seed_for


def run_single_replicate(
    nuc: Nucleosome,
    replicate_num: int,
    prot_params: dict,
    tau_max: float,
    inf_protamine: bool,
) -> ReplicateResult:
    """Run one replicate. Deep-copies the nucleosome so the caller's
    instance is never mutated; builds a fresh `protamines` instance;
    derives a deterministic seed via `seed_for(nuc, replicate_num)`.
    """
    nuc_copy = copy.deepcopy(nuc)
    prot = protamines(**prot_params)
    seed = seed_for(nuc, replicate_num)

    sim = GillespieEventSimulator(
        nuc=nuc_copy,
        prot=prot,
        tau_max=tau_max,
        inf_protamine=inf_protamine,
        seed=seed,
    )
    return sim.run()
