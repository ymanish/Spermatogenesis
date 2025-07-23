import itertools
import numpy as np


# ratio_range = np.arange(0.0, 1.1, 0.1, dtype=float)
# ratio_range = np.round(ratio_range, 1)


# cooperativity = np.arange(0.0, 1.1, 0.1, dtype=float)
# cooperativity = np.round(cooperativity, 1)
# ku = np.arange(0.1, 1.1, 0.1, dtype=float)
# ku = np.round(ku, 1)
ku = [210]

# kw = np.arange(330, 5000, 200, dtype=int)
# kw = np.round(kw, 1)
kw =  [330, 530, 730, 1330, 2330, 3330, 4330, 4930]
# ka = np.arange(0.1, 1.1, 0.1, dtype=float)
# ka = np.round(ka, 1)
# ka = [2113]
# kd = np.arange(0.1, 1.1, 0.1, dtype=float)
# kd = np.round(kd, 1)
# kd = [0.23, 23]
p_conc_2 = [0]
coop_2 = [0]

p_conc = np.arange(0.1, 5, 0.5, dtype=float)
coop=[0, 0.5, 1, 2]

# print(p_conc)

binding_sites=14

combinations = list(itertools.product(ku, kw, p_conc, coop))
combination_2 = list(itertools.product(ku, kw, p_conc_2, coop_2))
# combinations = list(itertools.product(ratio_range, cooperativity))

count = 1
with open("With_protamine_Breathing_Parameters.txt", "w") as file:
    for c in combinations:
        file.write(f'{count},{c[0]},{c[1]},{c[2]},{c[3]},\n')
        count = count+1

    # for k in combination_2:
    #     file.write(f'{count},{k[0]},{k[1]},{k[2]},{k[3]},\n')
    #     count = count+1

# with open("param_file_breath_cooperate.txt", "w") as file:
#     for c in combinations:
#         file.write(f'{count},{c[0]},{c[1]},\n')
#         count = count+1