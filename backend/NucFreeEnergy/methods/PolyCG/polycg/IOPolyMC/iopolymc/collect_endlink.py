import glob
import os
import sys
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

from .query_sims import querysims

############################################################################################
############################################################################################
############################################################################################


def eval_endlink(
    path: str,
    forces: List[float] = None,
    select: Dict[str, Any] = None,
    fileext: str = ".endlink",
    disc_len: float = 0.34,
    recursive: bool = True,
    num_files: int = None,
    save: bool = True,
    load: bool = True,
    check_most_recent: bool = True,
    print_status: bool = True,
) -> np.ndarray:
    evals_path = path + "/evals"
    npyfn = evals_path + "/ceff.npy"

    if select is None:
        select = {}

    if forces is None:
        sims = querysims(path, select=select, recursive=recursive)
        forces = [sim["force"] for sim in sims]
    forces = sorted(list(set(forces)))

    nbps = list()
    for force in forces:
        select["force"] = force
        sims = querysims(path, select=select, recursive=recursive)
        nbps += [sim["num_bp"] for sim in sims]
    if len(list(set(nbps))) > 1:
        raise ValueError(f"Encountered different chain lengths")

    # load from file if binary is still most recent
    if load and os.path.isfile(npyfn):
        if print_status:
            print("attempting to load")
        if check_most_recent:
            allsims = []
            for force in forces:
                select["force"] = force
                allsims += querysims(path, select=select, recursive=recursive)
            latest = _find_latest_file(allsims, fileext=fileext)
            if os.path.getmtime(npyfn) >= latest:
                if print_status:
                    print("loading from binary")
                return np.load(npyfn)
        else:
            if print_status:
                print("loading from binary")
            return np.load(npyfn)

    L = disc_len * nbps[0]
    data = np.zeros([len(forces), 9])
    for i, force in enumerate(forces):
        select["force"] = force
        endlink = collect_endlink(
            path, select, fileext=fileext, recursive=recursive, num_files=num_files
        )
        lk = endlink[:, 0]
        tw = endlink[:, 1]
        data[i, 0] = force
        data[i, 1] = np.mean(lk)
        data[i, 2] = np.var(lk)
        data[i, 3] = np.mean(tw)
        data[i, 4] = np.var(tw)
        data[i, 5] = L
        data[i, 6] = L / (4 * np.pi**2 * data[i, 2])
        data[i, 7] = L / (4 * np.pi**2 * data[i, 4])
        data[i, 8] = L / (4 * np.pi**2 * np.var(lk - tw))

    if save:
        if not os.path.exists(evals_path):
            os.makedirs(evals_path)
        np.save(npyfn, data)
    return data


############################################################################################
############################################################################################
############################################################################################


def _find_latest_file(sims: Dict[str, Any], fileext: str = ".endlink"):
    latest = 0
    for sim in sims:
        for fn in sim["files"]:
            if os.path.splitext(fn)[-1].lower() == fileext:
                t = os.path.getmtime(fn)
                if t > latest:
                    latest = t
    return latest


############################################################################################
############################################################################################
############################################################################################


def collect_endlink(
    path: str,
    select: Dict[str, Any],
    fileext: str = "endlink",
    recursive: bool = True,
    num_files: int = None,
    save_binary: bool = True,
    print_status: bool = True
    # ) -> np.ndarray | None:
) -> np.ndarray:
    # querey sims
    sims = querysims(path, select=select, recursive=recursive)

    if fileext[0] != ".":
        fileext = "." + fileext

    # calculate topols for each simulation
    endlinks = []
    num = 0
    for sim in sims:
        if num_files is not None and num >= num_files:
            break
        for fn in sim["files"]:
            if os.path.splitext(fn)[-1].lower() != fileext:
                continue
            if print_status:
                print(f"loading {fn}")
            endlink = load_endlink(fn, save_binary=save_binary, fileext=fileext)
            endlinks.append(endlink)
            num += 1
    if len(endlinks) > 0:
        return np.concatenate(endlinks)
    return None


############################################################################################
############################################################################################
############################################################################################


def load_endlink(
    fn: str, save_binary: bool = True, fileext: str = ".endlink"
) -> np.ndarray:
    if fileext[0] != ".":
        fileext = "." + fileext
    npyfn = fn.replace(fileext, fileext.replace(".", "_")) + ".npy"
    if os.path.isfile(npyfn) and os.path.getmtime(npyfn) >= os.path.getmtime(fn):
        endlink = np.load(npyfn)
    else:
        endlink = np.loadtxt(fn)
        if save_binary:
            np.save(npyfn, endlink)
    return endlink


############################################################################################
############################################################################################
############################################################################################


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: python %s path force" % sys.argv[0])
        sys.exit(0)

    path = sys.argv[1]
    force = float(sys.argv[2])

    # print(path)
    # print(force)
    # sys.exit()

    select = {"force": force}
    data = collect_endlink(path, select, save_binary=True)

    print(data.shape)
