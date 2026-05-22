"""
TNP2-mediated protamine cooperativity disruption.

TNP2 binding is represented as an analytically averaged, CpG-dependent
occupancy profile over the 14 exposed DNA lattice sites.  It does not scale
nucleosome opening or rewrapping rates directly.  Instead, TNP2 reduces the
effective protamine-protamine cooperativity bond between adjacent sites:

    J_eff(j, j+1) = J_bare * (1 - p_T(j)) * (1 - p_T(j+1))

The Markov state space remains the original (l, r) nucleosome breathing chain.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np


PHOSPHATE_BIND_SITES = [
    2, 6, 14, 17, 24, 29, 34, 38,
    45, 49, 55, 59, 65, 69, 76,
    80, 86, 90, 96, 100, 107, 111,
    116, 121, 128, 131, 139, 143,
]
CONTACT_FIRST_PHOSPHATE = [PHOSPHATE_BIND_SITES[2 * j] for j in range(14)]
CONTACT_LAST_PHOSPHATE = [PHOSPHATE_BIND_SITES[2 * j + 1] for j in range(14)]


@dataclass
class TNP2Config:
    """Configuration for the TNP2 J_eff layer."""

    enabled: bool = False
    eps_cpg: float = 1.0
    mu_t0: float = -8.0


def count_cpg(seq: Optional[str]) -> int:
    """Count CpG dinucleotides in a DNA sequence."""
    if not seq:
        return 0
    s = seq.upper()
    return sum(1 for i in range(len(s) - 1) if s[i] == "C" and s[i + 1] == "G")


def get_site_ranges(
    n_sites: int = 14,
    seq_len: int = 147,
    side: str = "left",
) -> list[tuple[int, int]]:
    """Return [start, end) bp windows ordered from the opening side inward.

    ``side='left'`` uses first-phosphate boundaries and returns sites ordered
    from left entry DNA toward the dyad.  ``side='right'`` uses last-phosphate
    boundaries and returns sites ordered from right entry DNA toward the dyad.
    The two orientations are intentionally not the same partition reversed,
    because each arm exposes DNA relative to the contact phosphates on that side.
    """
    if n_sites != 14:
        raise ValueError(f"TNP2 site ranges are defined for 14 sites, got {n_sites}")
    if side not in ("left", "right"):
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")

    site_ranges = []
    if side == "left":
        for j in range(n_sites):
            start = 0 if j == 0 else CONTACT_FIRST_PHOSPHATE[j]
            end = seq_len if j == n_sites - 1 else CONTACT_FIRST_PHOSPHATE[j + 1]
            site_ranges.append((start, end))
    else:
        for j in range(n_sites):
            contact = n_sites - 1 - j
            start = 0 if contact == 0 else CONTACT_LAST_PHOSPHATE[contact - 1] + 1
            end = seq_len if contact == n_sites - 1 else CONTACT_LAST_PHOSPHATE[contact] + 1
            site_ranges.append((start, end))
    return site_ranges


def count_cpg_per_site(seq_147: str, n_sites: int = 14, side: str = "left") -> np.ndarray:
    """Count CpG dinucleotides in each oriented nucleosome lattice-site window."""
    if seq_147 is None:
        raise ValueError("seq_147 must not be None")
    seq = seq_147.upper()
    if len(seq) != 147:
        raise ValueError(f"Expected 147 bp sequence, got {len(seq)}")

    counts = []
    for start, end in get_site_ranges(n_sites=n_sites, seq_len=len(seq), side=side):
        count = 0
        for i in range(start, min(end, len(seq) - 1)):
            if seq[i] == "C" and seq[i + 1] == "G":
                count += 1
        counts.append(count)
    return np.asarray(counts, dtype=int)


def compute_tnp2_occupancy_profile(
    seq_147: str,
    eps_cpg: float,
    mu_t0: float,
    beta: float = 1.0,
    n_sites: int = 14,
    side: str = "left",
) -> tuple[np.ndarray, np.ndarray]:
    """Return site-wise TNP2 occupancy probabilities and CpG counts."""
    cpg_counts = count_cpg_per_site(seq_147, n_sites=n_sites, side=side)
    x = beta * (mu_t0 + eps_cpg * cpg_counts.astype(float))
    p_t = np.empty_like(x, dtype=float)
    pos = x >= 0.0
    p_t[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    p_t[~pos] = ex / (1.0 + ex)
    return p_t, cpg_counts


def compute_jeff_profile(
    seq_147: str,
    eps_cpg: float,
    mu_t0: float,
    j_bare: float,
    beta: float = 1.0,
    n_sites: int = 14,
    side: str = "left",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return effective protamine cooperativity bonds, TNP2 occupancy, and CpG counts."""
    p_t, cpg_counts = compute_tnp2_occupancy_profile(
        seq_147=seq_147,
        eps_cpg=eps_cpg,
        mu_t0=mu_t0,
        beta=beta,
        n_sites=n_sites,
        side=side,
    )
    jeff = j_bare * (1.0 - p_t[:-1]) * (1.0 - p_t[1:])
    return jeff, p_t, cpg_counts


def compute_oriented_jeff_profiles(
    seq_147: str,
    eps_cpg: float,
    mu_t0: float,
    j_bare: float,
    beta: float = 1.0,
    n_sites: int = 14,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Return left- and right-oriented J_eff profiles."""
    return {
        side: compute_jeff_profile(
            seq_147=seq_147,
            eps_cpg=eps_cpg,
            mu_t0=mu_t0,
            j_bare=j_bare,
            beta=beta,
            n_sites=n_sites,
            side=side,
        )
        for side in ("left", "right")
    }


def parse_fasta(
    fa_path: Union[str, Path],
    id_style: str = "name",
) -> Dict[str, str]:
    """Parse a FASTA file into ``{key: sequence}``."""
    if id_style not in ("coord", "name"):
        raise ValueError(f"id_style must be 'coord' or 'name', got {id_style!r}")

    records: Dict[str, str] = {}
    current_key: Optional[str] = None
    seq_parts: list[str] = []

    with open(fa_path) as fh:
        for line in fh:
            line = line.rstrip()
            if not line:
                continue
            if line.startswith(">"):
                if current_key is not None:
                    records[current_key] = "".join(seq_parts).upper()
                header = line[1:]
                parts = header.split("|")
                if id_style == "coord":
                    current_key = parts[0].strip()
                elif len(parts) >= 2:
                    current_key = parts[1].strip()
                else:
                    current_key = parts[0].strip()
                seq_parts = []
            else:
                seq_parts.append(line)

    if current_key is not None:
        records[current_key] = "".join(seq_parts).upper()

    return records
