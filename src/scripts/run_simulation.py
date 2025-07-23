#/src/scripts/run_simulation.py


import concurrent.futures
import time

from src.core.nucleosome import nucleosome
from src.core.protamine import protamines
from src.core.gillespie import GillespieSimulator


class Simulation_Core():

    def __init__(self, K_UNWRAP, K_WRAP, K_ADS, K_DES, P_CONC, COOPERATIVITY, NNuc,
                  Simulation_steps, Indpt_runs, Bind_sites=14):
        self.K_UNWRAP = K_UNWRAP
        self.K_WRAP = K_WRAP
        self.K_ADS = K_ADS
        self.K_DES = K_DES
        self.P_CONC = P_CONC
        self.COOPERATIVITY = COOPERATIVITY
        self.NNuc = NNuc
        self.Simulation_steps = Simulation_steps
        self.Indpt_runs = Indpt_runs
        self.NUC_BIND = Bind_sites

    def simulation_fn(self, index):
        nuc_instance = nucleosome(k_unwrap=self.K_UNWRAP,
                                   k_wrap=self.K_WRAP,
                                   num_nucleosomes=self.NNuc,
                                   binding_sites=self.NUC_BIND)

        protamines_instance = protamines(k_unbind=self.K_DES,
                                         k_bind=self.K_ADS,
                                         p_conc=self.P_CONC,
                                         cooperativity=self.COOPERATIVITY)

        simulation = GillespieSimulator(nuc_instance,
                                        protamines_instance, 
                                        num_nucleosomes=self.NNuc, 
                                        STEPS=self.Simulation_steps)

        print('Simulation instance created for index:', index)
        return list(simulation.run())


    def parallel_execute_simulation(self):
        with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
            pool = []
            for i in range(self.Indpt_runs):
                print(i)
                pool.append(executor.submit(self.simulation_fn, i))

            for j in concurrent.futures.as_completed(pool):
                states = j.result()
                yield states



if __name__== '__main__':

    start = time.perf_counter()
    

    SC_instance = Simulation_Core(K_UNWRAP=250, 
                                K_WRAP=350, 
                                K_ADS=2113, 
                                K_DES=0.23,
                                P_CONC=0.1, 
                                COOPERATIVITY=0.5, 
                                NNuc=2, 
                                Simulation_steps=100, 
                                Indpt_runs=10, 
                                )
    



    for result in SC_instance.parallel_execute_simulation():
        print(result[-1])

       # Process the states as needed
        # for state in result:
        #     print(state)
