'''

Tools is a helper Python script used for housing various functions for coordinate transformations, helper functions for integration, etc.

'''

import numpy as np

def orb_to_cartesian():
    return None


def aei(mp, Ms, pos, vel, G):
    mu = G * (mp + Ms)
    z_unit = np.array([0.0, 0.0, 1.0])

    r = np.linalg.norm(pos, axis=1)                 # (N,)
    v2 = np.sum(vel * vel, axis=1)                  # (N,)
    rv = np.sum(pos * vel, axis=1)                  # (N,)
    
    # specific energy
    E_esp = 0.5 * v2 - mu / r

    # semi major axis
    a = -(mu)/(2.0*E_esp)

    # eccentricity
    # e_vec = ((v^2 - mu/r) r_vec - (r·v) v_vec) / mu (slightly different from the standard formula but the same)
    e = np.linalg.norm(((v2 - mu/r) * pos - rv * vel) / mu, axis=0)

    # inclination
    h = np.cross(pos, vel)
    h_norm = np.linalg.norm(h, axis=1)
    cos_i = h[:, 2] / h_norm # only take the third component which signifies the perpendicular component
    cos_i = np.clip(cos_i, -1.0, 1.0)
    i = np.arccos(cos_i)

    return a, e, i

def jacobi2cart(x, masses, N, eta):
    cart = np.zeros(N*6)
    # Rcm and Vcm

    Rcm = x[0:3]*eta[-1]
    Vcm = x[N*3: (N+1)*3]*[eta[-1]]

    for i in range(N-1, 0, -1):
        Rcm = (Rcm - masses[i]*x[i*3 : (i+1)*3])/eta[i]
        Vcm = (Vcm - masses[i]*x[(N+i)*3 : (N+i+1)*3])/eta[i]

        R = Rcm + x[i*3 : (i+1)*3]
        V = Vcm + x[(N+i)*3 : (N+i+1)*3]

        cart[i*3:(i+1)*3] = R
        cart[(N+i)*3 : (N+i+1)*3] = V

        Rcm = Rcm*eta[i-1]
        Vcm = Vcm*eta[i-1]

    R0 = Rcm/masses[0]
    V0 = Vcm/masses[0]

    cart[0:3] = R0
    cart[N*3:(N+1)*3] = V0

    return cart

def cart2jacobi(x, masses, N, eta):
    jacobi = np.zeros(N*6)

    Rcm = x[0:3]*masses[0]
    Vcm = x[N*3:(N+1)*3]*masses[0]

    for i in range(1, N-1):
        ri = (x[i:i+1] - Rcm)/eta[i-1]
        vi = (x[(N+i)*3:(N+i+1)*3] - Vcm)/eta[i-1]

        jacobi[i*3:(i+1)*3] = ri
        jacobi[(N+i)*3:(N+i+1)*3] = vi

        Rcm = Rcm*(1+masses[i]/eta[i-1])+masses[i]*ri
        Vcm = Vcm*(1+masses[i]/eta[i-1])+masses[i]*vi

    r0 = Rcm/eta[N-1]
    v0 = Vcm/eta[N-1]
    jacobi[0:3] = r0
    jacobi[N*3:(N+1)*3] = v0

    return jacobi

def stumpff_functions(z):

    # reducing z until the solution will be precise
    n = 0
    while (abs(z) > 0.1):
        n += 1
        z /= 4.0
    
    # compute c3, c2, c1, c0
    c3 = (1.0 - z * (1.0 - z * (1.0 - z* (1.0 - z* (1.0 - z*(1.0 - z / 210.0)\
    / 156.0)/110.0)/72.0)/42.0)/20.0)/6.0

    c2 = (1.0 - z*(1.0 - z*(1.0 - z*(1.0 - z*(1.0 - z*(1.0 - z / 182.0)\
        /132.0)/90.0)/56.0)/30.0)/12.0)/2.0
    
    c1 = 1.0 - z*c3
    c0 = 1.0 - z*c2

    # recovering the actual argument
    while n>0:
        n -= 1
        c3 = (c2 + c0*c3)/4.0
        c2 = c1*c1/2.0
        c1 = c0 * c1
        c0 = 2.0 * c0 * c0 - 1.0

    return c0, c1, c2, c3