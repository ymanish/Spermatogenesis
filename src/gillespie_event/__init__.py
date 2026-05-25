"""Event-driven Gillespie simulator package.

Runs Gillespie simulations without a fixed sampling grid, recording state
only at n_closed-change events. Produces MFPT, RMST, survival, and
event-only trajectory outputs per SPRM dataset.
"""

from src.gillespie_event.config import GillespieEventConfig
from src.gillespie_event.orchestrator import run_gillespie_event

__all__ = ["GillespieEventConfig", "run_gillespie_event"]
