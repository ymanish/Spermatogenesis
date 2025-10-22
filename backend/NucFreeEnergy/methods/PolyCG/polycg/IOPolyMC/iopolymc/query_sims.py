import glob
import os
import sys
from typing import Any, Dict, List
import numpy as np
from .input import read_input

"""
########################################################


   
########################################################
"""

########################################################
########################################################
########################################################
# query simulations

def querysims(
    path: str,
    select: Dict[str, Any] = None,
    recursive=False,
    extension="in",
    sims: List[Dict[str, Any]] = None,
) -> List[dict]:
    """
    Queries directory and subdirectories (if recursive=True), for simulations and ready the input. Specified simulations
    can be selected by passing a parameter dictionary via the argument select.
    """
    if sims is None:
        sims = _init_querysims(path, recursive=recursive, extension=extension)
    if select is not None:
        if not isinstance(select, dict):
            raise TypeError(
                f"Error in querysims: argument select needs to be a dictionary"
            )
        selected = list()
        for sim in sims:
            match = True
            for key in select.keys():
                if key not in sim.keys():
                    # raise exception if specified argument is not contained in input file
                    raise ValueError(
                        f"The argument '{key}' is not contained in the inputfile '{sim['input']}'"
                    )
                if sim[key] != select[key]:
                    match = False
                    break
            if match:
                selected.append(sim)
        sims = selected
    return sims


def _init_querysims(path: str, recursive=False, extension="in") -> List[dict]:
    """
    finds and reads all PolyMC input files and identifies other files belonging to the corresponding simulation.
    returns list of dictionary, one dictionary for each simulation. The other corresponding simulation files are contained in
    the dictionary under the key 'files'
    """
    infiles = list()
    if recursive:
        subpaths = [path] + _fast_scandir(path)
        for subpath in subpaths:
            infiles += glob.glob(os.path.join(subpath, "*." + extension))
    else:
        infiles += glob.glob(os.path.join(path, "*." + extension))

    sims = list()
    for infile in infiles:
        siminput = read_input(infile)
        files = simfiles(infile)
        siminput["input"] = infile
        siminput["files"] = files
        sims.append(siminput)
    return sims


def simfiles(infile: str, extension="in") -> List[str]:
    basefn = infile.replace("." + extension, "")
    allfns = glob.glob(basefn + ".*")
    return allfns

########################################################
########################################################
########################################################
# extra funcs

def _fast_scandir(path: str) -> List[str]:
    subpaths = [f.path for f in os.scandir(path) if f.is_dir()]
    for path in list(subpaths):
        subpaths.extend(_fast_scandir(path))
    return subpaths


########################################################
########################################################
########################################################
# testing

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python %s filename" % sys.argv[0])
        sys.exit(0)
        
    path = sys.argv[1]
    print(path)
    sims = querysims(path, recursive=True, extension="in")
    # for sim in sims:
    #     print(sim)
