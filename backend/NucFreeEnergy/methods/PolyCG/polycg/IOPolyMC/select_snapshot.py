import sys, os
import numpy as np
import argparse
import iopolymc as iopmc

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Extract single snapshot from xyz or state file")
    parser.add_argument("-in", "--infile", type=str, required=True)
    parser.add_argument("-out", "--outfile", type=str, default=None)
    parser.add_argument("-t", "--type", type=str, default=None)
    parser.add_argument("-s", "--snapid", type=int, required=True)
    args = parser.parse_args()
    
    if args.outfile is None:
        basename = os.path.splitext(args.infile)[0]
        fout = basename + f'_#{args.snapid}.xyz'
    else:
        fout = args.outfile
        if os.path.splitext(fout)[-1].lower() != '.xyz':
            fout += '.xyz'
    
    t = args.type
    
    if iopmc.isxyz(args.infile):
        data = iopmc.load_xyz(args.infile,savenpy=False)
        pos = data['pos']
        if t is None:
            types = data['types']
        else:
            types = [t for i in range(pos.shape[1])]
    elif iopmc.isstate(args.infile):
        data = iopmc.load_state(args.infile,savenpy=False)
        pos = data['pos']
        if t is None:
            t = 'C'
        types = [t for i in range(pos.shape[1])]
    else:
        raise ValueError(f'Unknown filetype')
    
    xyz = {'pos': [pos[args.snapid]], 'types': types}
    iopmc.write_xyz(fout,xyz,append=False)
