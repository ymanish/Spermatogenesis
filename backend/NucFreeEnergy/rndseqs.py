import sys, os
import numpy as np

if __name__ == '__main__':
    
    seqsfn = sys.argv[1]
    num    = int(sys.argv[2])
    with open(seqsfn, 'w') as f:
        for i in range(num):
            seq = ''.join(['ATCG'[np.random.randint(4)] for i in range(147)])
            f.write(seq+'\n')