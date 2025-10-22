
from enum import IntEnum
from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np
from typing import NamedTuple

class SiteState(IntEnum):
    CLOSED = 0   # fully wrapped
    OPEN   = 1   # unwrapped / unbound
    BOUND  = 2   # unwrapped + protamine

@dataclass
class SimulationState:
    time: float
    cs_total: int
    detached_total: int
    bprot: int
    cs: Optional[List[np.ndarray]] = None 
    t_blocked: Optional[float] = None
    nucs_snapshot: Optional[List[np.ndarray]] = None


class ReactionType(IntEnum):
    UNWRAPPING = 0
    REWRAPPING = 1
    BINDING    = 2
    UNBINDING  = 3

class Rates(NamedTuple):
    persite: Dict[ReactionType, Dict[int, float]]
    total:   Dict[ReactionType, float]

@dataclass
class ReactionChoice:
    nuc_idx: int
    reaction: ReactionType

REACTION_TARGET_STATE: Dict[ReactionType, SiteState] = {
    ReactionType.UNWRAPPING: SiteState.OPEN,
    ReactionType.REWRAPPING: SiteState.CLOSED,
    ReactionType.BINDING:    SiteState.BOUND,
    ReactionType.UNBINDING:  SiteState.OPEN,
}