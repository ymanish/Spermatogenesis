#/src/core/protamine.py

import numpy as np
import math

class protamines:
    def __init__(self, k_unbind, k_bind, p_conc, cooperativity=1):

        """defines the unbinding rate of protamine for each bound site
        :param k_unbind: unbinding rate
        :param k_bind: binding rate
        :param p_conc: protamine concentration in microMolar
        :param cooperativity: cooperativity factor"""



        self.k_unbind = k_unbind
        self.k_bind = (k_bind* math.pow(10, -2))/6 # the binding rate is provided in microMolar^-1s^-1. To convert it to s^-1*(molecules/micrometer^3)^-1, multiply by 10^15 for conversion to micrometer^3 from liter, 
                                                    # then devide by 6*10^23*10^-6 for conversion to molecules from micromolar to molecules. OR simply mutiply by (1/6)*1e-2

        
        self.cooperativity = cooperativity
        self.p_conc = p_conc * math.pow(10, -6)  # convert from micro_molar to Molar which is also moles/liter
        self.P_free = self.p_conc * 6 * math.pow(10, 23) * math.pow(10, -15)  # initial concentration of protamines molecules/cubic_micrometer
        self.N_bound = 0
        print('Protamine binding rate and molecules' , self.k_bind, self.P_free)
        # self.k_ratio = 0.00244
        self.k_ratio = self.k_bind/self.k_unbind 

    def protein_binding(self, open_sites):
        bind_sites = dict()
        # neigh_dict=dict()
        sum_rate = 0
        if len(open_sites) != 0:
            for i in open_sites:
                bind_sites[i] = self.k_bind* self.P_free
                # print(self.P_free)
            
            sum_rate = sum(bind_sites.values())

        return bind_sites, sum_rate


    def protein_unbinding(self, bound_sites):
        """defines the unbinding rate of protamine for each bound site
        :param bound_sites: contains the indexes for the protamine bound site
         :return: dictionary with key as the index of the bound sites and value as its unbinding rate.
                    Also returns the sum of all the unbinding rates"""
    
        unbound_sites = dict()
        sum_rate = 0
    
        if len(bound_sites) != 0:
            for i in bound_sites:
                unbound_sites[i] = self.k_unbind

            sum_rate = sum(unbound_sites.values())
    
        return unbound_sites, sum_rate

    def delta_E(self, p, J, s_l=0, s_r=0):
        ### get the energy required to move from a bound state to unbound state
        ## s_l and s_r are the left and right neighbours to the bound site.
        ## refer to the task documentation for the derivation of the formulae. The energy here is in terms of Kb*T.

        # U = np.log(0.00244 * p)  ## in terms of KT
        U = np.log(self.k_ratio * p)  ## in terms of KT

        #     J=12.6/(1-s_l-s_r)

        #     return (U + J)-J*0.5*(s_l+s_r)
        return J * (s_l + s_r) + U

    def get_spin_value(self, v):
        if v == 2:
            return 1
        else:
            return 0

    ##### With Coooperativity, here the cooperativity will make the protamine unbinding difficult.
    def protein_unbinding_coop(self, nuce, pt_indexes, B=1):
        bd_sites = dict()
        sum_rate = 0
        beta = B
        J = self.cooperativity

        if len(pt_indexes) != 0:
            for i in pt_indexes:

                if i == 0:
                    s_right = self.get_spin_value(nuce[i + 1])

                    #                 bd_sites[i] = k_unbind
                    bd_sites[i] = (self.k_bind * self.P_free )/ math.exp(beta * self.delta_E(p=self.P_free, J=J, s_r=s_right))

                elif i == 13:
                    s_left = self.get_spin_value(nuce[i - 1])

                    #                 bd_sites[i] = k_unbind
                    bd_sites[i] = (self.k_bind * self.P_free) / math.exp(beta * self.delta_E(p=self.P_free, J=J, s_l=s_left))

                else:
                    s_left = self.get_spin_value(nuce[i - 1])
                    s_right = self.get_spin_value(nuce[i + 1])

                    #                 bd_sites[i] = k_unbind
                    bd_sites[i] = (self.k_bind * self.P_free) / math.exp(beta * self.delta_E(p=self.P_free, J=J, s_l=s_left, s_r=s_right))
            # print('Function Bpund Site Pfree', self.P_free)
            # print('Function Bpund Site Kbind' ,self.k_bind)
            # print('Function Bpund Site J', J)
        if len(bd_sites) > 0:
            sum_rate = sum(bd_sites.values())

        return bd_sites, sum_rate


