# src/core/trajectory.py
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List

@dataclass
class TrajectoryRecorder:
    times:  np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=float))
    n_closed: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=int))
    n_bound:  np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=int))
    n_open:  np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=int))  # not used, but can be useful
    p_free: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=float))  # not used, but can be useful
    nuc_states: List[np.ndarray] = field(default_factory=list)
    nuc_fell_idx: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=int))  # flattened sites
    nuc_fall_time: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=float))  # flattened sites

    def add(self, t, nucleosomes, n_closed, n_bound, n_open, p_free):
        self.times = np.append(self.times, t)
        self.n_closed = np.append(self.n_closed, n_closed)
        self.n_bound = np.append(self.n_bound, n_bound)
        self.n_open = np.append(self.n_open, n_open)
        self.p_free = np.append(self.p_free, p_free)
        self.nuc_states.append(np.concatenate(nucleosomes).copy())

    # quick access as a pandas DataFrame
    def as_dataframe(self):
        return pd.DataFrame({
            "t": self.times,
            "n_closed": self.n_closed,
            "n_bound": self.n_bound,
            "n_open": self.n_open,
            "p_free": self.p_free
        })
    def add_nuc_fell_state(self, nuc_idx, nuc_fall_time):
        """Add the fell state of nucleosomes and their fall time."""
        self.nuc_fell_idx = np.append(self.nuc_fell_idx, nuc_idx)
        self.nuc_fall_time = np.append(self.nuc_fall_time, nuc_fall_time)

    def save_csv(self, path):
        self.as_dataframe().to_csv(path, index=False)

    def save_npy(self, path):
        # large arrays → binary is faster
        np.save(path, np.stack(self.nuc_states, axis=0))
