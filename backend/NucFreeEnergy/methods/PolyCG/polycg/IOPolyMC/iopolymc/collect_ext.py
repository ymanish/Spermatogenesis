import glob
import os
import sys
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

from .query_sims import querysims

############################################################################################
############################################################################################
############################################################################################


def eval_rotation_curve(
    path: str,
    force: float,
    # sigmas: List[float] | np.ndarray | None = None,
    sigmas: List[float] = None,
    fileext: str = ".zext",
    disc_len: float = 0.34,
    helical_repeat: float = 3.57,
    recursive: bool = True,
    num_files: int = None,
    save: bool = True,
    load: bool = True,
    mirror: bool = True,
    check_most_recent: bool = True,
    print_status: bool = True,
) -> np.ndarray:
    evals_path = path + "/evals"
    npyfn = evals_path + ("/rotcurve_f%.3f" % force).replace(".", "p") + ".npy"
    select = {"force": force}
    nbps = list()
    if sigmas is None:
        sims = querysims(path, select=select, recursive=recursive)
        sigmas = [sim["sigma"] for sim in sims]
        nbps += [sim["num_bp"] for sim in sims]
    sigmas = sorted(list(set(sigmas)))

    if len(list(set(nbps))) > 1:
        raise ValueError(f"Encountered different chain lengths")

    # load from file if binary is still most recent
    if load and os.path.isfile(npyfn):
        if print_status:
            print("attempting to load")
        if check_most_recent:
            allsims = []
            for sig in sigmas:
                allsims += querysims(
                    path, select={"force": force, "sigma": sig}, recursive=recursive
                )
            latest = _find_latest_file(allsims, fileext=fileext)
            if os.path.getmtime(npyfn) >= latest:
                if print_status:
                    print("loading from binary")
                data = np.load(npyfn)
                if mirror:
                    data = mirror_rotcurve_data(data)
                return data
        else:
            if print_status:
                print("loading from binary")
            data = np.load(npyfn)
            if mirror:
                data = mirror_rotcurve_data(data)
            return data

    L = disc_len * nbps[0]
    data = np.zeros([len(sigmas), 5])
    for i, sigma in enumerate(sigmas):
        select["sigma"] = sigma
        exts = collect_ext(
            path, select, fileext=fileext, recursive=recursive, num_files=num_files
        )
        dLk = L / helical_repeat * sigma

        mean = np.mean(exts)
        var = np.var(exts)
        data[i, 0] = sigma
        data[i, 1] = dLk
        data[i, 2] = mean
        data[i, 3] = var
        data[i, 4] = L

    if save:
        if not os.path.exists(evals_path):
            os.makedirs(evals_path)
        np.save(npyfn, data)
    if mirror:
        data = mirror_rotcurve_data(data)
    return data


############################################################################################
############################################################################################
############################################################################################


def mirror_rotcurve_data(data: np.ndarray) -> np.ndarray:
    haszero = 0 in data[:, 0]
    if haszero:
        ndata = np.zeros([1 + 2 * (len(data) - 1), len(data[0])])
        ndata[(len(data) - 1) :] = data
        ndata[: (len(data) - 1)] = data[1:][::-1]
        ndata[: (len(data) - 1), 0] = -ndata[: (len(data) - 1), 0]
        ndata[: (len(data) - 1), 1] = -ndata[: (len(data) - 1), 1]
    else:
        ndata = np.zeros([2 * len(data), len(data[0])])
        ndata[len(data) :] = data
        ndata[: len(data)] = data[::-1]
        ndata[: len(data), 0] = -ndata[: len(data), 0]
        ndata[: len(data), 1] = -ndata[: len(data), 1]
    return ndata


############################################################################################
############################################################################################
############################################################################################


def eval_force_extension(
    path: str,
    forces: List[float] = None,
    select: Dict[str, Any] = None,
    fileext: str = ".zext",
    disc_len: float = 0.34,
    recursive: bool = True,
    num_files: int = None,
    save: bool = True,
    load: bool = True,
    check_most_recent: bool = True,
    print_status: bool = True,
) -> np.ndarray:
    evals_path = path + "/evals"
    npyfn = evals_path + "/forceext.npy"

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
    data = np.zeros([len(forces), 4])
    for i, force in enumerate(forces):
        select["force"] = force
        exts = collect_ext(
            path, select, fileext=fileext, recursive=recursive, num_files=num_files
        )
        mean = np.mean(exts)
        var = np.var(exts)
        data[i, 0] = force
        data[i, 1] = mean
        data[i, 2] = var
        data[i, 3] = L

    if save:
        if not os.path.exists(evals_path):
            os.makedirs(evals_path)
        np.save(npyfn, data)
    return data


############################################################################################
############################################################################################
############################################################################################


def _find_latest_file(sims: Dict[str, Any], fileext: str = ".zext"):
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


def collect_ext(
    path: str,
    select: Dict[str, Any],
    fileext: str = "zext",
    recursive: bool = True,
    num_files: int = None,
    save_binary: bool = True
    # ) -> np.ndarray | None:
) -> np.ndarray:
    # querey sims
    sims = querysims(path, select=select, recursive=recursive)

    if fileext[0] != ".":
        fileext = "." + fileext

    # calculate topols for each simulation
    exts = []
    num = 0
    for sim in sims:
        if num_files is not None and num >= num_files:
            break
        for fn in sim["files"]:
            if os.path.splitext(fn)[-1].lower() != fileext:
                continue
            print(f"loading {fn}")
            ext = load_zext(fn, save_binary=save_binary, fileext=fileext)
            exts.append(ext)
            num += 1
    if len(exts) > 0:
        return np.concatenate(exts)
    return None


############################################################################################
############################################################################################
############################################################################################


def load_zext(fn: str, save_binary: bool = True, fileext: str = ".zext") -> np.ndarray:
    if fileext[0] != ".":
        fileext = "." + fileext
    npyfn = fn.replace(fileext, fileext.replace(".", "_")) + ".npy"
    if os.path.isfile(npyfn) and os.path.getmtime(npyfn) >= os.path.getmtime(fn):
        ext = np.load(npyfn)
    else:
        ext = np.loadtxt(fn)
        if save_binary:
            np.save(npyfn, ext)
    return ext


############################################################################################
############################################################################################
############################################################################################


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: python %s path force" % sys.argv[0])
        sys.exit(0)

    path = sys.argv[1]
    force = float(sys.argv[2])
    data = eval_rotation_curve(path, force)

    z = [dat[2] for dat in data]
    v = [dat[3] for dat in data]

    print(f"{z}")
    print(f"{v}")
