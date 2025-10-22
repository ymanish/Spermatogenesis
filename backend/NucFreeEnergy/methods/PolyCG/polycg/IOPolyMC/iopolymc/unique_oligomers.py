import sys
from typing import List

import numpy as np


def dna_oligomers(num_bp: int, omit_equiv=True) -> List[str]:
    uo = UniqueOligomers(omit_equiv=omit_equiv)
    return uo.get_oligomers(num_bp)


class UniqueOligomers:
    def __init__(self, omit_equiv: bool = True):
        self.bases = "atcg"
        self.omit_equiv = omit_equiv

    def get_oligomers(self, num_bp):
        self.seqlist = []
        self._seqloop("", 0, num_bp)
        return self.seqlist

    def _seqloop(self, seq: str, current: int, num_bp: int):
        current += 1
        for i in range(len(self.bases)):
            new_seq = seq + self.bases[i]
            if current < num_bp:
                self._seqloop(new_seq, current, num_bp)
            else:
                if not (self.omit_equiv and self.invert_seq(new_seq) in self.seqlist):
                    self.seqlist.append(new_seq)

    def invert_seq(self, seq: str):
        comp_dict = {"a": "t", "t": "a", "c": "g", "g": "c"}
        return "".join(comp_dict[base] for base in seq[::-1])

    def get_mid_dimer(self, seq: str):
        if len(seq) % 2 == 0:
            unique_dimers = sorted(self.get_oligomers(2))
            dimer = seq[len(seq) // 2 - 1 : len(seq) // 2 + 1]
            if dimer not in unique_dimers:
                dimer = self.invert_seq(dimer)
                seq = self.invert_seq(seq)
            return dimer, seq
        return "", seq


def complementary_sequence(sequence: str):
    comp_dict = {"a": "t", "t": "a", "c": "g", "g": "c"}
    return "".join(comp_dict[base] for base in sequence.lower()[::-1])


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: %s N" % sys.argv[0])
        sys.exit()

    N = int(sys.argv[1])
    bases = "atcg"
    if len(sys.argv) >= 3:
        bases = sys.argv[2]

    uo = UniqueOligomers(bases=bases)
    olis = uo.get_oligomers(N)

    print(len(olis))
    # ~ for oli in olis:
    # ~ print(oli)
