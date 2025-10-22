import glob
import os
import sys
from typing import Any, Dict, List

import numpy as np

from .simplest_type import simplest_type

"""
########################################################


   
########################################################
"""


def read_input(filename: str) -> Dict[str, Any]:
    """
    Reads PolyMC input file and returns dictionary of contained arguments
    """
    args = dict()
    with open(filename, "r") as f:
        all_lines = f.readlines()
        dlines = [
            line.strip()
            for line in all_lines
            if len(line.strip()) > 0 and line.strip()[0] != "#"
        ]
        for line in dlines:
            if "=" not in line:
                continue
            argname = line.split("=")[0].strip()
            argstr = "=".join(line.split("=")[1:]).strip()
            arglist = [arg.strip() for arg in argstr.split(" ")]
            if len(arglist) == 1:
                args[argname] = simplest_type(arglist[0])
            else:
                args[argname] = simplest_type(arglist)
    return args


def write_input(filename: str, args: Dict[str, Any]):
    """
    Writes PolyMC input file given a dictionary containing the arguments. Argument names are specified by
    the keys.
    """
    ml = np.max([len(key) for key in args.keys()])
    with open(filename, "w") as f:
        for key in args.keys():
            elems = args[key]
            wstr = key.ljust(ml + 1) + "="
            if type(elems) is list:
                for elem in elems:
                    wstr += f" {elem}"
            else:
                wstr += f" {elems}"
            f.write(wstr + "\n")


########################################################
########################################################
########################################################
# testing

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python %s filename" % sys.argv[0])
        sys.exit(0)
    fn  = sys.argv[1]
    args = read_input(fn)
    for key in args.keys():
        print(f'{key}: {args[key]} ({type(args[key])})')


