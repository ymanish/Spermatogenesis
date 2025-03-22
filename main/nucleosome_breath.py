import numpy as np
import math as math
import os
import sys

current_file_path = os.path.realpath(__file__)
parent_directory_path = os.path.dirname(current_file_path)
grandparent_directory_path = os.path.dirname(parent_directory_path)
sys.path.append(grandparent_directory_path)

from Backend.NucFreeEnergy.methods.free_energy import nucleosome_free_energy
from Backend.NucFreeEnergy.methods.read_nuc_data import GenStiffness
from Backend.NucFreeEnergy.methods.read_nuc_data import read_nucleosome_triads

class NucleosomeBreath:
    def __init__(self,  
                nuc_method='crystal', 
                hang_dna_method='md'):
        
        self.genstiff_nuc = GenStiffness(method=nuc_method, simple=False)
        self.genstiff_hang = GenStiffness(method=hang_dna_method, simple=True)
        self.triadfn = os.path.join(os.path.dirname(__file__), r"C:\Users\maya620d\PycharmProjects\Spermatogensis\Backend\NucFreeEnergy\methods\State\Nucleosome.state")
        #/home/pol_schiessel/maya620d/Nucleosome_Free_Energy/Backend/NucFreeEnergy/methods/State/Nucleosome.state
        self.nuctriads = read_nucleosome_triads(self.triadfn)

    def calculate_free_energy(self, seq601:str, site_loc:np.ndarray):
        stiff, gs = self.genstiff_nuc.gen_params(seq601)
        F_dict= nucleosome_free_energy(
            gs,
            stiff,
            site_loc, 
            self.nuctriads
        )
        F601 = F_dict['F']
        F_entrop = F_dict['F_entropy']
        F_entalap = F_dict['F_const']
        F_free = F_dict['F_free']

        return F601, F_entrop, F_entalap, F_free

        
    def select_phosphate_bind_sites(self, left=0, right=13):

        phosphate_bind_sites = [2, 6, 14, 17, 24, 29, 34, 38, 
                                    45, 49, 55, 59, 65, 69, 76, 
                                    80, 86, 90, 96, 100, 107, 111, 
                                    116, 121, 128, 131, 139, 143]
        
        return phosphate_bind_sites[left*2:(right*2)+2]
    

#  {3, 7, 15, 18, 25, 30, 35, 39, 46, 50, 56, 60, 66, 70, 77, 81, 87, 91, 97, 101, 108, 112, 117, 122, 129, 132, 140, 144, 5000};


if __name__ == '__main__':
    # seq601 = "ATCGAGAATCCCGGTGCCGAGGCCGCTCAATTGGTCGTAGACAGCTCTAGCACCGCTTAAACGCACGTACGCGCTGTCCCCCGCGTTTTAACCGCCAAGGGGATTACTCCCTAGTCTCCAGGCACGTGTCAGATATATACATCCGAT"
    # nucleosome_breath = NucleosomeBreath()

    # closed_sites = [2, 3, 4, 5, 6, 7, 8, 9, 10]


    

    # F601, F_entrop, F_entalap = nucleosome_breath.calculate_free_energy(seq601, site_loc=nucleosome_breath.select_phosphate_bind_sites(left=closed_sites[0]))
    # print(F601, F_entrop, F_entalap)
    # F601_, F_entrop_, F_entalap_ = nucleosome_breath.calculate_free_energy(seq601, site_loc=nucleosome_breath.select_phosphate_bind_sites(right=closed_sites[-1]))
    # print(F601_, F_entrop_, F_entalap_ )







    # Write the stiffness matrix data to the output file
    import os
    import numpy as np
    import pandas as pd

    # Define the directory path
    dir_path = r"C:\Users\maya620d\PycharmProjects\Spermatogensis\Backend\NucFreeEnergy\methods\Parametrization\MolecularDynamics"

    # Define the output file
    output_file = "MMC_MD_stiffness.csv"

    # Define the headers
    headers = "CG,CA,TA,AG,GG,AA,GA,AT,AC,GC,TG,CT,CC,TT,TC,GT"

    # Initialize an empty dictionary to hold the data
    data = {}
    
    # Loop through each file in the directory
    for filename in os.listdir(dir_path):
        # Check if the file is a Stiffness file
        if "Stiffness" in filename:
            # Get the di-nucleotide name from the filename
            di_nucleotide = os.path.splitext(filename)[0]
            print(di_nucleotide)
            # Read the file
            with open(os.path.join(dir_path, filename), 'r') as file:
                # Read the file content into a numpy array

                # content = np.loadtxt(file, delimiter=' ')
                content = np.fromstring(file.read(), sep=' ')
                # Reshape the content into a 6x6 matrix
                content = content.reshape(6, 6)
                temp_nuc = dict()

                eul = ['tilt', 'roll', 'twist', 'shift', 'slide', 'rise']
                exists_moves = []
                for i in range(6):
                    for j in range(6):
                        print(eul[i], eul[j], content[j][i])
                        s1 = eul[i] +'-'+ eul[j]
                        s2 = eul[j] +'-'+ eul[i]

                        if s1 not in exists_moves and s2 not in exists_moves:
                            temp_nuc[s1] = content[j][i]
                            exists_moves.append(s1)
                            exists_moves.append(s2)


                # # Add the mean to the data dictionary
                data[di_nucleotide.split('-')[2]] = temp_nuc
    
    df = pd.DataFrame(data)
    df.to_csv(output_file)
    sys.exit()

    # Write the data to the output file
    with open(output_file, 'w') as file:
        # Write the headers
        file.write(',' + headers + '\n')
        
        # Write the data
        for key, value in data.items():
            file.write(f"{key},{value}\n")