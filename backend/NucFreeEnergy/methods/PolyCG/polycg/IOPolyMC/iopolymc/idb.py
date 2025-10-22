import sys
from typing import Any, Dict, List

import numpy as np


def read_idb(filename: str) -> dict:
    """
    returns dictionary
    keys:
    - interaction_range (0 = local)
    - monomer_types (what are the possible types used)
    - discretization
    - avg_inconsist (relevant for non-local couplings)
    - seq_params (dict with oligomer types (key) + 3 arguments: types, vec (groundstate), params (identifier model + list of model parameters))
    """

    def gen_seq_combinations(interaction_range: int, monomer_types: str) -> List[str]:
        num = 2 * (interaction_range + 1)
        seqs = list()

        def iterate(seq, pos):
            for i in range(len(monomer_types)):
                added_seq = seq + monomer_types[i]
                nextpos = pos + 1
                if nextpos < num:
                    iterate(added_seq, nextpos)
                else:
                    seqs.append(added_seq)

        iterate("", 0)
        return seqs

    def get_paramset_line_ids(lines: List[str], seqs: List[str]) -> List[List[List]]:
        seq_ids = list()
        missing = list()
        for seq in seqs:
            found = False
            for i in range(len(lines)):
                if seq == lines[i].strip():
                    seq_ids.append([seq, i])
                    found = True
                    break
            if not found:
                missing.append(seq)
        return seq_ids, missing

    def stripsplit(line, delimiter=" ", replaces=["\t", "\n"]):
        for repl in replaces:
            line = line.replace(repl, delimiter)
        line = line.strip()
        while delimiter + delimiter in line:
            line = line.replace(delimiter + delimiter, delimiter)
        return line.split(delimiter)

    with open(filename, "r") as f:
        lines = f.readlines()
        lines = [
            line for line in lines if len(line.strip()) > 0 and line.strip()[0] != "#"
        ]

        idb = dict()
        for line in lines:
            arg = line.split(" ")[0].split("=")[0]
            if arg.lower() == "interaction_range":
                interaction_range = int(line.split("=")[-1].strip())
            if arg.lower() == "monomer_types":
                monomer_types = line.split("=")[-1].strip()
            if arg.lower() in ["discretization", "disc_len"]:
                disc_len = float(line.split("=")[-1].strip())
            if arg.lower() == "avg_inconsist":
                avg_inconsist = bool(line.split("=")[-1].strip())

        successful = True
        try:
            idb["interaction_range"] = interaction_range
        except NameError:
            print('argument "interaction_range" not found in idb file')
            successful = False
        try:
            idb["monomer_types"] = monomer_types
        except NameError:
            print('argument "monomer_types" not found in idb file')
            successful = False
        try:
            idb["disc_len"] = disc_len
        except NameError:
            print('argument "disc_len" (or "discretization") not found in idb file')
            successful = False
        try:
            idb["avg_inconsist"] = avg_inconsist
        except NameError:
            print('argument "avg_inconsist" not found in idb file')
            successful = False

        seqs = gen_seq_combinations(interaction_range, monomer_types)
        seq_ids, missing = get_paramset_line_ids(lines, seqs)
        if len(missing) > 0:
            print(
                "The following sequence parameters have not been specified in the IDB files:"
            )
            for seq in missing:
                print(" - %s" % seq)
            raise Exception("Inconsistent IDB file")

        num_interactions = 1 + 2 * interaction_range
        seq_params = dict()
        nlines = num_interactions + 2
        for seq_id in seq_ids:
            fseq, lid = seq_id

            part_lines = list()
            while len(part_lines) < nlines:
                if lines[lid].strip() != "" and lines[lid].strip()[0] != "#":
                    part_lines.append(lines[lid])
                lid += 1

            seq = part_lines[0].strip()
            if seq not in seqs:
                raise Exception(f"Unexpected sequence '{seq}' encountered.")
            if seq != fseq:
                raise Exception(f"Error in line identification")
            params = list()
            for i in range(num_interactions):
                splitline = [
                    elem.strip()
                    for elem in part_lines[1 + i].strip().replace("\t", " ").split(" ")
                    if elem.strip() != ""
                ]
                param = [splitline[0]] + [float(v) for v in splitline[1:]]
                params.append(param)
            splitline = [
                elem.strip()
                for elem in part_lines[-1].strip().replace("\t", " ").split(" ")
                if elem.strip() != ""
            ]
            vec = [float(v) for v in splitline[1:]]

            seq_param = dict()
            seq_param["seq"] = seq
            seq_param["vec"] = vec
            seq_param["interaction"] = params
            seq_params[seq] = seq_param

        idb["params"] = seq_params
        return idb


def write_idb(filename: str, idbdict: Dict[str, Any], decimals=3) -> None:
    with open(filename, "w") as f:
        f.write(
            "################################################################################\n"
        )
        f.write(
            "############### SETUP ##########################################################\n"
        )
        f.write(
            "################################################################################\n"
        )
        f.write("\n")

        f.write("interaction_range  = %d \n" % idbdict["interaction_range"])
        f.write("monomer_types      = %s \n" % idbdict["monomer_types"])
        f.write("discretization     = %.2f \n" % idbdict["disc_len"])
        f.write("avg_inconsist      = %d \n" % idbdict["avg_inconsist"])

        f.write("\n")
        f.write(
            "################################################################################\n"
        )
        f.write(
            "############ INTERACTIONS ######################################################\n"
        )
        f.write(
            "################################################################################\n"
        )

        for seq in sorted([key for key in idbdict["params"].keys()]):
            seqparams = idbdict["params"][seq]
            seq = seqparams["seq"]
            vec = seqparams["vec"]
            coups = seqparams["interaction"]

            f.write("\n")
            f.write(seq + "\n")
            # write couplings
            for coup in coups:
                line = (
                    f"\t{coup[0]}\t"
                    + " ".join([f"{np.round(c, decimals)}" for c in coup[1:]])
                    + "\n"
                )
                f.write(line)
            # write vec
            f.write(
                "\tvec\t\t\t"
                + " ".join([str(np.round(val, decimals)) for val in vec])
                + "\n"
            )
        f.close()


if __name__ == "__main__":
    idb = read_idb("test/test.idb")

    for key in idb:
        print("########")
        print(key)

    for key in idb["params"].keys():
        print(key)
        print(idb["params"][key])

    write_idb("test/test_out.idb", idb)
