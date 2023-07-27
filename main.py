# import pandas as pd
import numpy as np
# import random
# # random.seed(a = 1)
# from numba import njit, types
# import numba as nb
# import seaborn as sns
# import matplotlib.pyplot as plt
import time
import concurrent.futures
from initialise import *

def Open_Close_nucleosome(bstate, ku, kw, binding_points, nuc_off=False):
    opening = dict()
    closing = dict()
    left_length = len(bstate[0])
    right_length = len(bstate[1])
    total_sites = left_length + right_length

    if total_sites == binding_points:
        # print('nucleosome has fallen')
        nuc_off = True

    elif left_length > 0 and right_length > 0:

        if bstate[0][-1] == '1' and bstate[1][0] == '0':

            new_o_l = (bstate[0] + '0', bstate[1])
            new_o_r = (bstate[0], '0' + bstate[1])
            new_c_r = (bstate[0], bstate[1][1:])

            opening[new_o_l] = ku
            opening[new_o_r] = ku
            closing[new_c_r] = kw

        elif bstate[0][-1] == '0' and bstate[1][0] == '1':

            new_o_l = (bstate[0] + '0', bstate[1])
            new_c_l = (bstate[0][:-1], bstate[1])
            new_o_r = (bstate[0], '0' + bstate[1])

            opening[new_o_l] = ku
            opening[new_o_r] = ku
            closing[new_c_l] = kw


        elif bstate[0][-1] == '1' and bstate[1][0] == '1':
            new_o_l = (bstate[0] + '0', bstate[1])
            new_o_r = (bstate[0], '0' + bstate[1])

            opening[new_o_l] = ku
            opening[new_o_r] = ku

        else:
            new_o_l = (bstate[0] + '0', bstate[1])
            new_o_r = (bstate[0], '0' + bstate[1])
            new_c_r = (bstate[0], bstate[1][1:])
            new_c_l = (bstate[0][:-1], bstate[1])

            opening[new_o_l] = ku
            opening[new_o_r] = ku
            closing[new_c_l] = kw
            closing[new_c_r] = kw


    elif left_length == 0 and right_length > 0:
        new_o_l = (bstate[0] + '0', bstate[1])
        new_o_r = (bstate[0], '0' + bstate[1])
        opening[new_o_l] = ku
        opening[new_o_r] = ku

        if bstate[1][0] == '0':
            new_c_r = (bstate[0], bstate[1][1:])
            closing[new_c_r] = kw


    elif left_length > 0 and right_length == 0:

        new_o_l = (bstate[0] + '0', bstate[1])
        new_o_r = (bstate[0], '0' + bstate[1])
        opening[new_o_l] = ku
        opening[new_o_r] = ku

        if bstate[0][-1] == '0':
            new_c_l = (bstate[0][:-1], bstate[1])
            closing[new_c_l] = kw

    else:
        new_o_l = (bstate[0] + '0', bstate[1])
        new_o_r = (bstate[0], '0' + bstate[1])
        opening[new_o_l] = ku
        opening[new_o_r] = ku

    return opening, closing, nuc_off

def protamine_binding(state):
    one_states = []
    zero_states = []
    state_ = state[0] + '_' + state[1]

    for idx, x in enumerate(state_):
        state_list = list(state_)
        if x == '1':
            state_list[idx] = '0'
            dis_x = "".join(state_list)
            zero_states.append(dis_x)
        if x == '0':
            state_list[idx] = '1'
            ass_x = "".join(state_list)
            one_states.append(ass_x)

    return one_states, zero_states


def uniform_pos_arg():
    return np.random.uniform(0.0, 1.0)
# uniform_pos_arg_njit = nb.njit(uniform_pos_arg)


def A_D_state(bstate, ku, kw, ka, kd):
    diss_states = dict()
    ass_states = dict()
    left_length = len(bstate[0])
    right_length = len(bstate[1])

    A_states, D_states = protamine_binding(bstate)

    if ka == 0 and kd == 0:
        return diss_states, ass_states

    elif ka != 0 and kd == 0:
        for indi, i in enumerate(A_states):
            tm = i.split('_')
            ass_states[(tm[0], tm[1])] = ka
        return diss_states, ass_states


    elif ka == 0 and kd != 0:
        for indj, j in enumerate(D_states):
            tm = j.split('_')
            diss_states[(tm[0], tm[1])] = kd
        return diss_states, ass_states

    else:
        for indi, i in enumerate(A_states):
            tm = i.split('_')
            ass_states[(tm[0], tm[1])] = ka

        for indj, j in enumerate(D_states):
            tm = j.split('_')
            diss_states[(tm[0], tm[1])] = kd
        return diss_states, ass_states

def kmc(N, bs, ku, kw, ka, kd):
    t = 0
    state_tracker = []
    time_tracker = []
    energy_tracker = []

    #     open_sites = random.randrange(0, bs)
    #     st = random.randrange(0, 2**open_sites)

    ## the current state will start with nucleosome closed

    current_state = ('', '')

    #     count_0 = bn_state.count('0')
    #     count_1 = bn_state.count('1')
    E = 0
    energy_tracker.append(E)

    state_tracker.append(current_state)
    time_tracker.append(t)
    # off_iter = False
    for n in range(1, N):

        #         if n == 0:
        #             open_sites = current_state[0]
        #             st = current_state[1]
        #             bn_state = ('{0:'+'0'+str(open_sites)+'b'+'}').format(st)

        #         print(bn_state, current_state)
        Opening_state, Closing_state, off_iter = Open_Close_nucleosome(current_state, ku, kw, bs)
        if off_iter:
            return state_tracker, time_tracker, energy_tracker, off_iter

        Unbound_state, Bound_state = A_D_state(current_state, ku, kw, ka, kd)

        Total_states = list(Opening_state.keys()) + \
                       list(Closing_state.keys()) + \
                       list(Unbound_state.keys()) + \
                       list(Bound_state.keys())

        Total_rates = list(Opening_state.values()) + \
                      list(Closing_state.values()) + \
                      list(Unbound_state.values()) + \
                      list(Bound_state.values())
        Q = sum(Total_rates)
        u = uniform_pos_arg()

        # cum_R = np.cumsum(Total_rates)

        cum_R = 0
        for r_idx, r in enumerate(Total_rates):
            cum_R = cum_R + r
            if Q * u <= cum_R:
                # Update the current state
                #                 next_state = (len(Total_states[r_idx]), binary_str_to_int(Total_states[r_idx]))
                next_state = Total_states[r_idx]
                #                 int(Total_states[r_idx],2)
                break


        # ns = search(cum_R, Q*u)
        # next_random_state = (len(Total_states[ns[0]]), int(Total_states[ns[0]],2))

        u_ = uniform_pos_arg()
        delta_t = np.log(1 / u_) / Q
        t = t + delta_t

        current_state = next_state
        count_0 = current_state[0].count('0') + current_state[1].count('0')
        count_1 = current_state[0].count('1') + current_state[1].count('1')
        E = count_0 * 1 + count_1 * -1
        energy_tracker.append(E)
        state_tracker.append(current_state)
        time_tracker.append(t)
    return state_tracker, time_tracker, energy_tracker, off_iter

def main(n_steps, k_unwrap, k_wrap,  k_ads, k_des, binding_sites):


    state_T, time_T, energy_T, nucleosome_fall = kmc(n_steps, binding_sites, k_unwrap, k_wrap, k_ads, k_des)

    all_states = []
    open_sites_array = []
    protamine_sites_array = []
    nucleosome_sites_array = []

    for s in state_T:
        nuc_len = binding_sites - (len(s[0]) + len(s[1]))
        # print('sdfsfdfd')
        DNA = s[0] + nuc_len * 'N' + s[1]
        all_states.append(list(DNA))
        open_sites_array.append(DNA.count('0'))
        nucleosome_sites_array.append(DNA.count('N'))
        protamine_sites_array.append(DNA.count('1'))


    return nucleosome_fall, time_T, open_sites_array, nucleosome_sites_array, protamine_sites_array


if __name__ == '__main__':

    nuc_fall_count = 0
    Fall_time = []
    start = time.perf_counter()
    nuc_job_counter = 0
    with concurrent.futures.ProcessPoolExecutor() as executor:
        iter_seq = range(Nucleosomes)
        pool = [executor.submit(main, n_steps=Simulation_steps,
                                k_unwrap=K_UNWRAP, k_wrap=K_WRAP,  k_ads=K_ADS,  k_des=K_DES, binding_sites=NUC_BIND) for i in iter_seq]
        for j in concurrent.futures.as_completed(pool):

            nuc_off, time_hist, open_site_hist, nucl_site_hist, prot_site_hist = j.result()
            # print(j, nuc_off, time_hist[-1])
            if nuc_off:
                nuc_fall_count =  nuc_fall_count + 1
                Fall_time.append(time_hist[-1])
            nuc_job_counter = nuc_job_counter + 1
    print (nuc_job_counter)
    if nuc_job_counter != Nucleosomes:
        print(f'Did not run for all nucleosomes, Run again for this {Param_ind}')
    # nuc_frac = nuc_fall_count/Nucleosomes


    with open(RESULT_DIR + f"{Param_ind}.txt", "w") as file1:
        file1.write(f"P:{Param_ind} ku:{K_UNWRAP} kw:{K_WRAP} ka:{K_ADS} kd:{K_DES} N:{Nucleosomes} F:{nuc_fall_count} KMC:{Simulation_steps} \n")
        file1.write('\n'.join(map(str, Fall_time)))

    end = time.perf_counter()
    print(f'Finished in {round(end - start, 2)} second(s)')
    print(f'Parameter {Param_ind} is done>>>>>>>>>>>>>>>>>>>>')

