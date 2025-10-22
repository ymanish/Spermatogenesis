#!/bin/env python3
import os
import sys
from typing import Dict

import numpy as np

IOPOLYMC_SEQ_PATTERN_EXACT = "exact"

##############################################################################
##############################################################################


def read_seq(seqfn: str) -> Dict[str, str]:
    lines = list()
    with open(seqfn, "r") as f:
        lines = [line.strip() for line in f.readlines() if line.strip() != ""]
    if len(lines) == 0:
        raise ValueError(f'Sequence file "{seqfn}" is empty.')
    if len(lines) > 2:
        raise ValueError(f'More than 2 lines found in sequence file "{seqfn}".')
    if len(lines) == 1:
        seqdict = {"opt": IOPOLYMC_SEQ_PATTERN_EXACT, "seq": lines[0]}
    if len(lines) == 2:
        seqdict = {"opt": lines[0], "seq": lines[1]}
    return seqdict


##############################################################################
##############################################################################


def write_seq(seqfn: str, seq: str, opt: str = None) -> None:
    with open(seqfn, "w") as f:
        if opt is not None and opt != IOPOLYMC_SEQ_PATTERN_EXACT:
            f.write(opt + "\n")
        f.write(seq)


##############################################################################
##############################################################################

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python %s fin" % sys.argv[0])
        sys.exit(0)
    seqfn = sys.argv[1]
    seqdict = read_seq(seqfn)
    for key in seqdict.keys():
        print(f"{key}: {seqdict[key]}")

    write_seq(seqfn + "_copy", seq=seqdict["seq"], opt=seqdict["opt"])
