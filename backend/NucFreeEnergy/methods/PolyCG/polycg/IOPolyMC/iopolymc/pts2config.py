import sys

import numpy as np

from .restart import write_restart
from .xyz import write_xyz

# import iopolymc


def pts2config(
    pts: np.ndarray,
    disc_len: float,
    closed=False,
    numbp: int = None,
    translate_first: bool = True,
) -> np.ndarray:
    """
    Interpolates 3D polymer config between points.
    Arguments:
        pts:        Points between which the configuration is interpolated
        disc_len:   Discretization length of generated trajectory
        closed:     Boolian controlling whether generated trajectory is closed. If True, the contour is
                    closed by connecting the first and the last point.
                    Default: False
                    NOT YET IMPLEMENTED
        numbp:      Number of points in generated trajectory. First generates a configuration by interpolating between the given points.
                    If the generated number of points doesn't match numbp, the generated configuration is either concatenated or extended
                    along the terminus tangents.
                    Default: None
        translate_first:
                    Translates the configuration such that first point conincides with origin
                    Default: True
    """

    def find_corner_pt(
        p1: np.ndarray, B: np.ndarray, C: np.ndarray, disc_len: float
    ) -> np.ndarray:
        """
        subsidiary function for pts2config
        """
        lamb = find_lambda(p1, B, C, disc_len)
        if lamb > 1:
            return False
        else:
            v = C - B
            return B + lamb * v

    def find_lambda(
        p1: np.ndarray, B: np.ndarray, C: np.ndarray, disc_len: float
    ) -> np.ndarray:
        """
        subsidiary function for pts2config
        """
        w = p1 - B
        v = C - B

        dot = np.dot(v, w)
        vsq = np.dot(v, v)
        wsq = np.dot(w, w)

        discrim = dot**2 - vsq * (wsq - disc_len**2)
        sqrtdis = np.sqrt(discrim)
        if dot > sqrtdis:
            return (dot - sqrtdis) / vsq
        else:
            return (dot + sqrtdis) / vsq

    if closed:
        raise ValueError(
            "The option to generate closed configurations is not yet implemented!"
        )

    config = list()
    config.append(pts[0])
    ptid = 1
    # ptA     = pts[0]
    while ptid < len(pts):
        ptA = pts[ptid - 1]
        ptB = pts[ptid]
        v = ptB - config[-1]
        dv = np.linalg.norm(v)
        uv = v / dv

        # num points in line segments
        num = int(np.floor(dv / disc_len))
        for i in range(num):
            npt = config[-1] + uv * disc_len
            config.append(npt)

        if ptid < len(pts) - 1:
            for refid in range(ptid, len(pts) - 1):
                p1 = config[-1]
                B = pts[refid]
                C = pts[refid + 1]
                lamb = find_lambda(p1, B, C, disc_len)
                if lamb < 0:
                    print("lambda is negative!")
                    sys.exit()
                if lamb <= 1:
                    npt = B + lamb * (C - B)
                    config.append(npt)
                    ptid = refid + 1
                    break
        else:
            ptid += 1
        print(ptid)

    config = np.array(config)
    if numbp is not None:
        if len(config) > numbp:
            diff = len(config) - numbp
            if diff % 2 == 0:
                lrm = int(diff / 2)
                rrm = lrm
            else:
                lrm = int(np.ceil(diff / 2))
                rrm = lrm - 1
            config = config[lrm:-rrm]
        elif len(config) < numbp:
            diff = numbp - len(config)
            if diff % 2 == 0:
                ladd = int(diff / 2)
                radd = ladd
            else:
                ladd = int(np.ceil(diff / 2))
                radd = ladd - 1
            configadd = np.zeros((numbp, 3))
            configadd[ladd:-radd] = config
            ltan = config[1] - config[0]
            rtan = config[-1] - config[-2]
            for i in range(ladd):
                configadd[ladd - 1 - i] = configadd[ladd - i] - ltan
            for i in range(radd):
                configadd[-radd + i] = configadd[-radd - 1 + i] + rtan
            config = configadd

    if translate_first:
        config -= config[0]

    return config


def config2triads(config: np.ndarray) -> np.ndarray:
    """
    generates triads for given configuration
    """

    se1 = np.array([1, 0, 0])
    se2 = np.array([0, 1, 0])

    triads = np.zeros((len(config), 3, 3))
    for i in range(len(config) - 1):
        e3 = config[i + 1] - config[i]
        e3 = e3 / np.linalg.norm(e3)
        if np.abs(np.abs(np.dot(e3, se1)) - 1) > 1e-6:
            e1 = se1 - np.dot(se1, e3) * e3
        else:
            e1 = se2 - np.dot(se2, e3) * e3
        e1 = e1 / np.linalg.norm(e1)
        e2 = np.cross(e3, e1)

        # test if invalid tangent vectors are assigned
        if (
            e1[0] != e1[0]
            or e1[1] != e1[1]
            or e1[2] != e1[2]
            or e2[0] != e2[0]
            or e2[1] != e2[1]
            or e2[2] != e2[2]
        ):
            print(e1)
            print(e2)
            print(e3)
            sys.exit()

        triads[i, :, 0] = e1
        triads[i, :, 1] = e2
        triads[i, :, 2] = e3
    triads[-1] = triads[-2]
    return triads


def pts2xyz(
    outfn: str,
    pts: np.ndarray,
    disc_len: float,
    numbp: int = None,
    sequence: str = None,
    default_type: str = "C",
):
    config = pts2config(pts, disc_len, numbp=numbp)
    dat = {"pos": np.array([config])}
    if sequence is None:
        dat["types"] = default_type * len(config)
    write_xyz(outfn, dat)


def pts2restart(
    outfn: str,
    pts: np.ndarray,
    disc_len: float,
    closed: bool = False,
    numbp: int = None,
    sequence: str = None,
    default_type: str = "a",
    snapshotid: int = 0,
):
    config = pts2config(pts, disc_len, numbp=numbp)
    triads = config2triads(config)

    if sequence is None:
        sequence = default_type * len(config)

    restart = dict()
    restart["snapshot"] = snapshotid
    print("snapshotid")
    print(snapshotid)
    if closed:
        restart["type"] = "circular"
    else:
        restart["type"] = "linear"
    restart["num_bp"] = len(config)
    restart["sequence"] = sequence
    restart["dLK"] = 0
    restart["pos"] = config
    restart["triads"] = triads

    write_restart(outfn, [restart])


if __name__ == "__main__":
    disc_len = 3.4
    numbp = 500

    # pts = list()
    # pts.append(np.array([0,0,0]))
    # pts.append(np.array([0,0,40]))
    # pts.append(np.array([0,10,40]))
    # pts.append(np.array([-5,10,30]))
    # pts.append(np.array([-5,-5,30]))
    # pts.append(np.array([5,-5,35]))
    # pts.append(np.array([5,5,35]))
    # pts.append(np.array([-5,5,35]))
    # pts.append(np.array([-5,5,45]))
    # pts.append(np.array([0,0,45]))
    # pts.append(np.array([0,0,80]))

    pts = list()
    pts.append(np.array([0, 0, 0]))
    pts.append(np.array([0, 0, 80]))
    pts.append(np.array([0, 40, 80]))
    pts.append(np.array([-20, 40, 40]))
    pts.append(np.array([-20, -20, 40]))
    pts.append(np.array([20, -20, 60]))
    pts.append(np.array([20, 20, 60]))
    pts.append(np.array([-20, 20, 60]))
    pts.append(np.array([-20, 20, 100]))
    pts.append(np.array([0, 0, 100]))
    pts.append(np.array([0, 0, 140]))

    pts = np.array(pts)
    pts *= 6
    # pts.append(np.array([2,2,82]))

    # config = pts2config(pts,disc_len,numbp=500)

    # print(config[0])

    # for p in config:
    #     print(p)

    # for i in range(len(config)-1):
    #     print(np.linalg.norm(config[i+1]-config[i]))
    # print(np.min([np.linalg.norm(config[i+1]-config[i]) for i in range(len(config)-1)]))

    # dat   = {'pos' : np.array([config]), 'types' : ['C' for i in range(len(config))]}
    xyzfn = "trefoil_large.xyz"
    restartfn = "trefoil_large.restart"

    numbp = 800
    pts2xyz(xyzfn, pts, disc_len, numbp=numbp)
    pts2restart(restartfn, pts, disc_len, numbp=numbp)
