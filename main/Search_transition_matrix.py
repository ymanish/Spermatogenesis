import numpy as np
from scipy.optimize import minimize
import pandas as pd

# Define the size of the lattice
lattice_size = 15 ## (0-14) 0 is complete wrapped and 14 is last binding site wrapped
num_states = 15*16/2
print(num_states)

binding_bp_right = [7, 18, 30, 39, 50, 60, 70, 81, 91, 101, 112, 122, 132, 144]
binding_bp_left = [2, 14, 24, 34, 45, 55, 65, 76, 86, 96, 107, 116, 128, 139]

SAXS_loc = r'C:\Users\maya620d\PycharmProjects\Spermatogensis\Parameters\SAXS_fraction.csv'
SAXS_BD_loc = r'C:\Users\maya620d\PycharmProjects\Spermatogensis\Parameters\SAXS_fraction_BD_site.csv'

SAXS_df = pd.read_csv(SAXS_loc)
SAXS_BD_df = pd.read_csv(SAXS_BD_loc)

print(SAXS_BD_df)


def decode_state(state):

    ##encoding scheme is L*15 + R, 
    L = state // 15
    R = state % 15
    if L +R <= 14:
        return L, R 
    else: 
        return None, None


def state_stationary_value(L, R, major_side=None):
    site = L + R
    print(L, R, site, major_side)
    if L == R:
        
        F, l_, r_ = SAXS_BD_df.loc[SAXS_BD_df['Site'] == site, ['Fraction', 'Left', 'Right']].values[0]
        # print(F, l_, r_, F- (l_ + r_))
        return F- (l_ + r_)
    elif (L + R) % 2 == 0:
        # print(SAXS_BD_df.loc[SAXS_BD_df['Site'] == site, major_side].values[0])
        # print(SAXS_BD_df.loc[SAXS_BD_df['Site'] == site, major_side].values[0] / ((L + R) // 2))
        return SAXS_BD_df.loc[SAXS_BD_df['Site'] == site, major_side].values[0] / ((L + R) // 2)
    else:
        F, l_, r_ = SAXS_BD_df.loc[SAXS_BD_df['Site'] == site, ['Fraction', 'Left', 'Right']].values[0]
        # print(F, l_, r_, SAXS_BD_df.loc[SAXS_BD_df['Site'] == site, major_side].values[0] )
        # print((F-l_-r_)/ (L + R + 1), (L + R + 1), (F-(l_+r_)))
        # print(SAXS_BD_df.loc[SAXS_BD_df['Site'] == site, major_side].values[0] / ((L + R + 1) // 2), (F-(l_+r_)/ (L + R + 1)))
        # print(SAXS_BD_df.loc[SAXS_BD_df['Site'] == site, major_side].values[0] / ((L + R + 1) // 2)) + (F-(l_+r_)/ (L + R + 1))
        return (SAXS_BD_df.loc[SAXS_BD_df['Site'] == site, major_side].values[0] / ((L + R + 1) // 2)) + (F-l_-r_)/(L + R + 1)
       



empty_array = np.zeros((lattice_size, lattice_size))

for s in range(0, int(lattice_size*lattice_size)):
    L, R = decode_state(s)
    # print(L, R)
    if L is None:
        continue
    elif L>R:
        
        empty_array[L, R] = state_stationary_value(L, R, 'Left')

    elif L<R:
        empty_array[L, R]=state_stationary_value(L, R, 'Right')

    else:
        empty_array[L, R] = state_stationary_value(L, R)

    print('-------------------')


np.savetxt('601_stationary_dist.txt', empty_array)


print(empty_array[7,8])







import sys
sys.exit()
























def find_closest_fraction(bp_unwrap):
    closest_bp = None
    closest_distance = float('inf') 


    for _, row in SAXS_df.iterrows():
        bp = row['Basepair']
        distance = abs(bp - bp_unwrap)

        if distance < closest_distance:
            closest_distance = distance
            closest_bp = bp

            fraction = row['Fraction']
            if closest_distance == 0:
                return fraction

    if closest_bp is not None:
        return fraction

    return None

# Define your known stationary distributions for combined states
# Example: stationary_distributions = {k: value, ...}
stationary_distributions = {}
count = 0
# sum=0
for i in binding_bp_left:
    # print(count, find_closest_fraction(i))
    stationary_distributions[count] = find_closest_fraction(i)
    # sum += find_closest_fraction(i)
    count += 1

# print(sum)
# print(stationary_distributions)
# Normalize the stationary_distributions
sum_values = sum(stationary_distributions.values())
stationary_distributions = {k: round(v / sum_values, 3) for k, v in stationary_distributions.items()}

print(stationary_distributions)



# Initialize the transition matrix P with equal probabilities
P = np.full((num_states, num_states), 1 / num_states)




def adjust_transition_matrix(P):
    # Adjust P for boundary conditions and lattice structure
    for i in range(lattice_size):
        for j in range(lattice_size):
            # Calculate the indices of the neighboring lattice points
            up = (i - 1, j)
            down = (i + 1, j)
            left = (i, j - 1)
            right = (i, j + 1)

            # Check if the neighboring lattice points are within the lattice boundaries
            if up[0] >= 0:
                P[i * lattice_size + j, up[0] * lattice_size + up[1]] = rate_up

            if down[0] < lattice_size:
                P[i * lattice_size + j, down[0] * lattice_size + down[1]] = rate_down

            if left[1] >= 0:
                P[i * lattice_size + j, left[0] * lattice_size + left[1]] = rate_left

            if right[1] < lattice_size:
                P[i * lattice_size + j, right[0] * lattice_size + right[1]] = rate_right

    return P

print(P.shape)

import sys 
sys.exit()


# Ensure P adheres to the lattice structure and boundary conditions
# You need to implement this function based on your lattice configuration
def adjust_transition_matrix(P):
    # Adjust P for boundary conditions and lattice structure
    return P

P = adjust_transition_matrix(P)

# Constraint function
def constraints(P_flat):
    P = P_flat.reshape((num_states, num_states))
    s = np.linalg.matrix_power(P, 100)[0]  # Example way to find stationary distribution
    constraint_values = []
    for k, value in stationary_distributions.items():
        # Calculate the sum of s_ij for i + j = k
        # Add the constraint (sum - value) to the constraint_values list
        pass
    return constraint_values

# Objective function (to be minimized)
def objective(P_flat):
    return np.sum(np.square(constraints(P_flat)))

# Flatten the matrix for the optimizer
P_flat = P.flatten()

# Run the optimization
result = minimize(objective, P_flat, method='SLSQP', constraints={'type': 'eq', 'fun': constraints})

# Reshape the result back into a matrix
P_optimized = result.x.reshape((num_states, num_states))

# Normalize P_optimized to ensure it's a valid transition matrix
# Implement normalization here

# Print or save the optimized matrix
print(P_optimized)
