import itertools
import numpy as np
import radialtreerc


ratio_range = np.arange(0.0, 1.1, 0.1, dtype=float)
ratio_range = np.round(ratio_range, 1)


cooperativity = np.arange(0.0, 1.1, 0.1, dtype=float)
cooperativity = np.round(cooperativity, 1)
# ku = np.arange(0.1, 1.1, 0.1, dtype=float)
# ku = np.round(ku, 1)
#
# kw = np.arange(0.1, 1.1, 0.1, dtype=float)
# kw = np.round(kw, 1)

# ka = np.arange(0.1, 1.1, 0.1, dtype=float)
# ka = np.round(ka, 1)
ka = [0]
# kd = np.arange(0.1, 1.1, 0.1, dtype=float)
# kd = np.round(kd, 1)
kd = [0]
binding_sites=14

# combinations = list(itertools.product(ku, kw, ka, kd))

combinations = list(itertools.product(ratio_range, cooperativity))

count = 1
# with open("param_file_breath_cooperate.txt", "w") as file:
#     for c in combinations:
#         file.write(f'{count},{c[0]},{c[1]},{c[2]},{c[3]},\n')
#         count = count+1
#

with open("param_file_breath_cooperate.txt", "w") as file:
    for c in combinations:
        file.write(f'{count},{c[0]},{c[1]},\n')
        count = count+1