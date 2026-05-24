#src/config/var.py
# Author: MY
from src.core.nucleosomes import Nucleosome

def seed_for(nuc: Nucleosome, rep: int, base: int = 17071) -> int:
    return base ^ (hash(nuc.id) & 0x7fffffff) ^ (int(nuc.subid) * 1_000_003) ^ (rep * 97_003)
