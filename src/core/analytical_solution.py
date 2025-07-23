import numpy as np
from scipy.optimize import brentq
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.interpolate import interp1d
import sys

class Nucl_Breathing_Sol:
    def __init__(self, v, c2, L, t_values, x_0, gamma=0, N_alpha=200):
        self.v = v
        self.c2 = c2
        self.L = L
        self.t_values = t_values
        self.x_0 = x_0
        # self.c = np.sqrt(D)
        self.a = self.v / (2 * self.c2)
        self.gamma = gamma
        self.N_alpha = N_alpha
        self.h = 2*c2/ ((self.v+2*self.gamma)*self.L)
        self.alpha_n_values = self.roots_alpha(rN=self.N_alpha, 
                                               h=self.h)
        print('h:', self.h)
        print('beta:', self.v)
        print('c2:', self.c2)   
        # sys.exit()
        if 1> self.h > 0:
            self.mu_root = self.roots_mu(h=self.h)
        else:
            self.mu_root = None
        
        print('alpha_n_values:', self.alpha_n_values)
        print('mu_root:', self.mu_root)

    def analytical_survival_probability(self):

        # c= np.sqrt(self.D)
        a = self.v / (2 * self.c2)  
        alpha_n_values = self.alpha_n_values
        mu = self.mu_root

        if a==0:
            F_values = [ self.F_diffusion(t=t, 
                        L=self.L, 
                        c2=self.c2, 
                        x_0 = self.x_0) for t in self.t_values]
        else:
            F_values = [ self.F(t=t, 
                        alpha_n_values=alpha_n_values,
                            mu=mu,
                            L=self.L, 
                            c2=self.c2, 
                            a=a, 
                            beta = self.v,
                            x_0 = self.x_0, 
                            h=self.h) for t in self.t_values]
        # F_values = F(t_values, alpha_n_values, L, c, a, x_0)

        return F_values

    def F_diffusion(self, t, L, c2, x_0):

        F_sum = 0
        for n in range(0, 200):
            term1 = (4/((2*n + 1)*np.pi))*np.sin(((2*n +1)*np.pi*x_0)/(2*L))*np.exp(-c2*((2*n+1)**2)*(np.pi**2)*t/(4*L**2))
            F_sum += term1
        return F_sum


    # Function to calculate F(x, t)
    def F(self, t, alpha_n_values, mu, L, c2, a, beta, x_0, h):
        F_sum = 0
        for alpha_n in alpha_n_values:
            b = alpha_n / L  # b definition
            term1 = np.sin(alpha_n * x_0 / L) * np.exp(-c2 * alpha_n**2 * t / L**2)
            term2 = 1 - (np.sin(2 * alpha_n) / (2 * alpha_n))
            # term3 = (b / (a**2 + b**2)) * (1 - np.exp(a * L) * np.cos(alpha_n) / 2)
            term3 = (b / (a**2 + b**2))
            term4 = (b*np.exp(a*L)*np.cos(alpha_n)*(a*h*L - 1))/(a**2 + b**2)


            F_sum += (term1 / term2) * (term3 + term4)
        
        if 1 >h>0 and mu is not None:
            b_ = mu / L
            term11 = np.sinh(mu * x_0 / L) * np.exp(c2 * mu**2 * t / L**2)
            term21 = (np.sinh(2 * mu) / (2 * mu))-1
            term31 = (b_ / (a**2 - b_**2))
            # extra = ((1.5*x_0)/(L**2))*(np.exp(a*L)*(a*L-1)+1)*(1/(a**2))
            # F_sum += ((term11 / term21) * term31) + extra
            term41 = (b_*np.exp(a*L)*np.cosh(mu)*(a*h*L - 1))/(a**2 - b_**2)
            F_sum += ((term11 / term21) * (term31+term41))



        if h == 1: 
            F_sum = F_sum + 1.5*x_0 ### This is for case when mu=0 or for the lamnda=0 case, which is possible when beta*L/2c**2 = 1.

        # if a*L < 1:
        #     extra = ((1.5*x_0)/(L**2))*(np.exp(a*L)*(a*L-1)+1)*(1/(a**2))
        #     F_sum = F_sum + extra

        F_result = (2 / L) * np.exp(-a * x_0) * np.exp(-beta**2 * t / (4 * c2)) * F_sum
        return F_result




    def MFPT(self):
        a = self.v / (2 * self.c2) 
        x_0 = self.x_0
        L = self.L
        alpha_n_values = self.alpha_n_values
        mu = self.mu_root
        F_sum_mean = 0

        for alpha_n in alpha_n_values:
            b = alpha_n / L  

            F_sum_mean_term1 = np.sin(alpha_n * x_0 / L)/(1 - (np.sin(2 * alpha_n) / (2 * alpha_n)))
            F_sum_mean_term2 = (b/((a**2 + b**2)**2))
            F_sum_mean += F_sum_mean_term1 * F_sum_mean_term2

        if 1 >self.h>0 and mu is not None:
            mu_root = self.mu_root
            b_ = mu_root / L
            F_sum_mean_term11 = np.sinh(mu_root * x_0 / L)/((np.sinh(2 * mu_root) / (2 * mu_root)-1))
            F_sum_mean_term21 = (b_/((a**2 - b_**2)**2))
            F_sum_mean += F_sum_mean_term11 * F_sum_mean_term21

        if self.h == 1:
            F_sum_mean = F_sum_mean + 1.5*x_0*L**2

        mean_first_time = (2 / (L*c2)) * np.exp(-a * x_0) * F_sum_mean
        
        return mean_first_time



    # Analytical probability distribution function
    def P_analytical(self, x, t):
        c2 = self.c2
        beta = self.v
        L = self.L
        x0 = self.x_0
        t = t
        alpha_n_values = self.alpha_n_values
        mu_root = self.mu_root
        sum_terms = np.zeros_like(x, dtype=np.float64)
        h =self.h

        ### Only Diffusion
        if beta ==0: 
            for n in range(0,200):
                term1 = np.sin(((2*n +1)*np.pi*x)/(2*L))*np.sin(((2*n+1)*np.pi*x0)/(2*L))*np.exp(-c2*((2*n+1)**2)*(np.pi**2)*t/(4*L**2))
                sum_terms += term1

            P = (2/L)*sum_terms
            return P

        else:

            for alpha_n in alpha_n_values:
                denominator = 1 - (np.sin(2 * alpha_n) / (2 * alpha_n))
                numerator = np.sin(alpha_n * x0 / L) * np.sin(alpha_n * x / L)
                exponential = np.exp(- (c2 * alpha_n**2 * t) / L**2)
                sum_terms += (numerator / denominator) * exponential


            if 1>h>0 and mu_root is not None:
                deno = (np.sinh(2 * mu_root) / (2 * mu_root)) -1 
                numo = np.sinh(mu_root * x0 / L) * np.sinh(mu_root * x / L)
                expo = np.exp((c2 * mu_root**2 * t) / L**2)
                sum_terms += (numo / deno) * expo

            if h == 1:
                sum_terms = sum_terms + (1.5*x*x0)/L**2  
        
            prefactor = (2 / L) * np.exp((beta * (x - x0)) / (2 * c2)) * np.exp(- (beta**2 * t) / (4 * c2))
            P = prefactor * sum_terms 

            return P


    def alpha_equation(self, alpha, kappa):
        return np.tan(alpha) - kappa * alpha



    def roots_alpha(self, rN, h):
        # Number of roots to find
        # N_roots = rN # Adjust as needed

        # h = 2*c**2 / ((beta+2*gamma)*L)  # Negative value
        # kappa = 2*c**2 / (beta * L)  # Negative value

        # h_dash  = 2*c**2 / ((beta+2*gamma)*L)

        alpha_n_values = []

        ## To find the negative x-axis roots the roots will lie in domain (n.pie , (n+0.5).pie) where n is negative integer.
        if h<0:
            for n in range(1, rN + 1):
                # Define the interval where the function may change sign
                lower_bound = (n - 0.5) * np.pi + 1e-5
                upper_bound = n * np.pi - 1e-5  # Avoid the asymptote at n * pi
            
                # Check if the function changes sign in the interval
                f_lower = self.alpha_equation(lower_bound, h)
                f_upper = self.alpha_equation(upper_bound, h)
                
                if f_lower * f_upper < 0:
                    # Find the root in the interval
                    root = brentq(self.alpha_equation, lower_bound, upper_bound, args=(h))
                    alpha_n_values.append(root)
                else:
                    print(f"No root found in interval [{lower_bound}, {upper_bound}] for n={n}")
            
            alpha_n_values = np.array(alpha_n_values)
            return alpha_n_values
            
        elif 1 >= h>0:
                print('h:', h)
                start_index = 1

        elif h>1:
                start_index = 0
        else:
            return np.array(alpha_n_values)


        for n in range(start_index, rN + 1):
            # Define the interval where the function may change sign
            lower_bound = n*np.pi + 1e-5
            upper_bound = (n +0.5) * np.pi  - 1e-5  # Avoid the asymptote at n * pi
            
            # Check if the function changes sign in the interval
            f_lower = self.alpha_equation(lower_bound, h)
            f_upper = self.alpha_equation(upper_bound, h)
            
            if f_lower * f_upper < 0:
                # Find the root in the interval
                root = brentq(self.alpha_equation, lower_bound, upper_bound, args=(h))
                alpha_n_values.append(root)
            else:
                print(f"No root found in interval [{lower_bound}, {upper_bound}] for n={n}")


        alpha_n_values = np.array(alpha_n_values)
        return alpha_n_values
    
    def alpha_equation_1(self, alpha, kappa):
        return np.tanh(alpha) - kappa * alpha


    def roots_mu(self, h):
        # mu_values = []
        # h = 2*c**2 / (beta * L)  # Positive value

        lower_bound = 0 + 1e-5
        upper_bound = 1/h - 1e-5 # the maximum value of tanh(x) is 1.
        
        # Check if the function changes sign in the interval
        f_lower = self.alpha_equation_1(lower_bound, h)
        f_upper = self.alpha_equation_1(upper_bound, h)
        
        if f_lower * f_upper < 0:
            # Find the root in the interval
            root = brentq(self.alpha_equation_1, lower_bound, upper_bound, args=(h))
            # mu_values.append(root)
            return root
        else:
            print(f"No root found in interval [{lower_bound}, {upper_bound}] for mu")
            return None




if __name__=='__main__':

    # Parameters
    gamma = 0       # Absorption rate
    L = 14.5          # Length of the domain
    x0 = 13.5         # Initial position
    k_r = 5        # Rate constant for right movement
    k_u = 6   # Rate constant for left movement
    dt = 0.01         # Time step
    T_max = 100.0       # Maximum simulation time
    N_particles = 1000  # Number of particles to simulate
    N_alpha = 200     # Number of alpha_n terms in the series
    t_points = int(T_max / dt)  # Number of time points
    t_array = np.arange(0, T_max, dt)
    
    x_positions = np.linspace(0, L, 300)  # Positions for plotting
    # Derived parameters
    beta = 2 * (k_r - k_u)       # Beta parameter
    c2 = k_u + k_r               # c^2 parameter
    # c = np.sqrt(c2)              # c parameter
    v = beta                     # Drift velocity

    nuc_breath = Nucl_Breathing_Sol(v=v, 
                                    c2=c2, 
                                    L=L,
                                     t_values=t_array, 
                                     x_0=x0, 
                                     gamma=gamma,
                                     N_alpha=N_alpha)


    # Time points at which to record positions
    record_times = [10.0, 50.0, 60, 80]  # Adjust as needed
    record_indices = [int(t / dt) for t in record_times]
    print(record_indices)
    recorded_positions = {}
    # Simulation of the particles
    print('Simulation started')
    print(beta, c2)

    def simulate_particles():
        positions = np.full(N_particles, x0, dtype=np.float64)
        alive = np.ones(N_particles, dtype=bool)
        survival_prob = np.zeros(t_points)

        for t in range(t_points):

            # Update positions for alive particles
            alive_indices = np.where(alive)[0]
            if alive_indices.size == 0:
                break  # All particles have been absorbed
            
            positions[alive] += beta * dt +np.sqrt(2 * c2 * dt) * np.random.randn(alive_indices.size)

            # Apply boundary conditions
            # Absorbing at x = 0
            absorbed = positions <= 0
            alive[absorbed] = False
            positions[positions <= 0] = 0  # For plotting purposes

            # 3) Partially absorbing (Robin) at x=L
            #    Find which are beyond L
            beyond_L = (positions >= L) & alive
            n_beyond = np.sum(beyond_L)
            # print(n_beyond)
            if n_beyond > 0:
                # Probability of absorption in this step
                p_absorb = 1.0 - np.exp(-gamma * dt)
                # print(p_absorb)
                # Draw random numbers to decide which among those beyond L get absorbed
                rand_vals = np.random.rand(n_beyond)
                # print(rand_vals)
                
                will_absorb = rand_vals < p_absorb

                # print(will_absorb)
                # sys.exit()


                # Mark those absorbed
                beyond_indices = np.where(beyond_L)[0]
                absorb_indices = beyond_indices[will_absorb]
                alive[absorb_indices] = False

                # The rest reflect
                reflect_indices = beyond_indices[~will_absorb]
                positions[reflect_indices] = 2*L - positions[reflect_indices]  # reflection


            # Record survival probability
            survival_prob[t] = np.sum(alive) / N_particles

             # Record positions at specified times
            if t in record_indices:
                recorded_time = t_array[t]
                print(recorded_time)
                recorded_positions[recorded_time] = positions.copy()
                print(recorded_positions)
        return survival_prob, recorded_positions

    survival_probability, recorded_positions = simulate_particles()


    # Compute Mean First Passage Time (MFPT)
    # MFPT_simulation = np.nanmean(absorption_times)


    # alpha_n_values = nuc_breath.roots_alpha(rN=200, c=c, beta=v, L=L)
    alpha_n_values = nuc_breath.alpha_n_values
    # alpha_n_values = np.append(alpha_n_values, 1e-3)  # Use np.append instead of list append
    # print(alpha_n_values)

    # Compute analytical survival probability
    analytical_S = nuc_breath.analytical_survival_probability()
    # Normalize analytical survival probability if necessary
    # analytical_S = np.clip(analytical_S, 0, 1)  # Ensure values are between 0 and 1

    # # Analytical Mean First Passage Time
    # MFPT_analytical = MFPT(x0, v, c2, L, alpha_n_values)

    # Plot survival probability
    plt.figure(figsize=(10, 6))
    plt.plot(t_array, survival_probability, label='Simulation')
    plt.plot(t_array, analytical_S, 'r--', label='Analytical')
    plt.xlabel('Time')
    plt.ylabel('Survival Probability')
    plt.title('Survival Probability vs Time')
    plt.legend()
    plt.grid()
    plt.show()


    bin_size = int(2*(t_points)**(1/3)) ##using scott's rule for bin size calculation
    # Plotting
    for t in recorded_positions.keys():
        pos = recorded_positions[t]

        # Compute histogram counts without density normalization
        hist_counts, bin_edges = np.histogram(pos, bins=bin_size, range=(0, L), density=False)

        bin_width = bin_edges[1] - bin_edges[0]
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        print(bin_centers)
        # Normalize histogram to get probability density over total number of particles
        hist_density = hist_counts / (N_particles * bin_width)
        
        P_x_t = nuc_breath.P_analytical(x_positions,t)

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.bar(bin_centers, hist_density, width=bin_width, alpha=0.5, label='Simulation')
        plt.plot(x_positions, P_x_t, 'r-', label='Analytical')
        plt.xlabel('Position x')
        plt.ylabel('Probability Density P(x, t)')
        plt.title(f'Probability Distribution at time t = {t:.2f}')
        # plt.ylim(0, 0.1)
        plt.legend()
        plt.grid(True)
        plt.show()












    # # Print Mean First Passage Times
    # print(f"Mean First Passage Time (Simulation): {MFPT_simulation:.4f}")
    # print(f"Mean First Passage Time (Analytical): {MFPT_analytical:.4f}")
