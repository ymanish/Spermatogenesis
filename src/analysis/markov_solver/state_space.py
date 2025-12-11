"""
State space construction for Markov chain in (l,r) space.
"""
from typing import Dict, List, Tuple


def build_state_space(N_MAX: int) -> Tuple[List[Tuple[int, int]], 
                                            List[Tuple[int, int]], 
                                            Dict[Tuple[int, int], int]]:
    """
    Enumerate all (l,r) states and classify as transient or absorbing.
    
    A state (l,r) is:
    - Transient if l+r < N_MAX (nucleosome partially wrapped)
    - Absorbing if l+r == N_MAX (nucleosome fully detached)
    
    Args:
        N_MAX: Maximum number of contacts (typically 14 for nucleosomes)
        
    Returns:
        transient_states: List of (l,r) tuples with l+r < N_MAX
        absorbing_states: List of (l,r) tuples with l+r == N_MAX
        index_map: Dictionary mapping (l,r) -> index for transient states only
    """
    transient_states = []
    absorbing_states = []
    index_map = {}
    
    for l in range(N_MAX + 1):
        for r in range(N_MAX + 1 - l):
            if l + r < N_MAX:
                idx = len(transient_states)
                transient_states.append((l, r))
                index_map[(l, r)] = idx
            elif l + r == N_MAX:
                absorbing_states.append((l, r))
    
    return transient_states, absorbing_states, index_map
