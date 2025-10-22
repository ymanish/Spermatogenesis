import glob
import os
import sys
from typing import List

import numpy as np

from .state import read_spec


def scan_path(path: str, ext: str, recursive=False) -> List[str]:
    """
    Returns all state files in path
    """
    entries = list()
    if not os.path.exists(path):
        print("Path '%s' does not exist" % path)
        return entries
    statefiles = np.sort(glob.glob(path + "/*.%s" % ext), recursive=recursive)
    for stfn in statefiles:
        specs = read_spec(stfn)
        raw_fn = stfn[:-6]
        entries.append([raw_fn, specs])
    return entries


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python %s path" % sys.argv[0])
        sys.exit(0)

    path = sys.argv[1]
    entries = scan_path(path)

    for entry in entries:
        if entry[1]["EVradius"] == 7.0:
            print(entry[0])
