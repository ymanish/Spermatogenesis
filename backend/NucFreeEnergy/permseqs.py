import sys, os
import numpy as np

if __name__ == '__main__':
    
    outfn           = sys.argv[1]
    num             = int(sys.argv[2])
    if len(sys.argv) > 3:
        single_size     = int(sys.argv[3])
        overlap_size    = int(sys.argv[4])
    else:
        single_size = 0
        
    seqrnd = ['ATCG'[np.random.randint(4)] for i in range(147)]
    seq601 = 'ATCGAGAATCCCGGTGCCGAGGCCGCTCAATTGGTCGTAGACAGCTCTAGCACCGCTTAAACGCACGTACGCGCTGTCCCCCGCGTTTTAACCGCCAAGGGGATTACTCCCTAGTCTCCAGGCACGTGTCAGATATATACATCCGAT'
    seq = seq601
    
    switch_all = True
    
    if not switch_all:
        # generate
        seqs = [seq]
        print(seqs[-1])
        rids = []
        for i in range(num-1): 
            seql = [seqs[-1][i] for i in range(len(seqs[-1]))]
            rid = np.random.randint(len(seql))
            rids.append(rid)
            seql[rid] = 'ATCG'.replace(seql[rid],'')[np.random.randint(3)]
            seqs.append(''.join(seql))
            print(' '*rid + '|')
            print(seqs[-1])
    
    else:       
        while True:
            # generate
            seqs = [seq]
            rids = []
            for i in range(num-1): 
                seql = [seqs[-1][i] for i in range(len(seqs[-1]))]
                rid = np.random.randint(len(seql))
                rids.append(rid)
                seql[rid] = 'ATCG'.replace(seql[rid],'')[np.random.randint(3)]
                seqs.append(''.join(seql))
                
            print(len(seqs[0])-len(list(set(rids))))
            if len(list(set(rids))) == len(seqs[0]):
                break
        
    
    with open(outfn, 'w') as f:
        for seq in seqs:
            f.write(seq+'\n')
        
    if single_size > 0 and single_size < num:
        if overlap_size*2 >= single_size:
            raise ValueError('overlap_size needs to be smaller than half of single_size')
        
        basename = os.path.splitext(outfn)[0]
        incsize = single_size - overlap_size
        nsplit = int(np.ceil((num - single_size)/incsize))+1
        
        splits = []
        cid = 0
        while cid < num:
            eid = cid + single_size
            if eid >= num:
                eid = num
                print(cid,eid)
                splits.append(seqs[cid:eid])
                print(len(splits[-1]))
                break
            print(cid,eid)
            splits.append(seqs[cid:eid])
            print(len(splits[-1]))
            cid += incsize    
        
        print(len(splits))
        print(nsplit)
        
        for i,split in enumerate(splits):
            splitfn = basename + f'_split{i+1}.seqs'
            with open(splitfn, 'w') as f:
                for seq in split:
                    f.write(seq+'\n')
        
        