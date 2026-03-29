import numpy as np

'''

This Python script contains all the constants that will be used for running the simulations.

'''

# PHYSICAL CONSTANTS
G =4*np.pi**2
sb = 5.67*10**(-8)

# Solar luminosity (W)
L_sun = 3.828e26

# HZ constants (Kopparapu et al. 2013, empirical limits)
# S_eff_sun: effective stellar flux at the HZ boundary for the Sun
S_eff_sun_inner = 1.7763   # Recent Venus
S_eff_sun_outer = 0.3207   # Early Mars

# Kopparapu polynomial coefficients for S_eff(T_star)
a_inner = 1.4335e-4
b_inner = 3.3954e-9
c_inner = -7.6364e-12
d_inner = -1.1950e-15

a_outer = 5.4471e-5
b_outer = 1.5275e-9
c_outer = -2.1709e-12
d_outer = -3.8282e-16