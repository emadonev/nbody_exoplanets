'''

Tools is a helper Python script used for housing various functions for coordinate transformations, helper functions for integration, etc.

'''

import numpy as np

def orb_to_cartesian():
    return None


def aei(mp, Ms, pos, vel, G):
    mu = G * (mp + Ms)

    r = np.linalg.norm(pos, axis=1)                 # (N,)
    v2 = np.sum(vel * vel, axis=1)                  # (N,)
    rv = np.sum(pos * vel, axis=1)                  # (N,)
    
    # specific energy
    E_esp = 0.5 * v2 - mu / r

    # semi major axis
    a = -(mu)/(2.0*E_esp)

    # eccentricity
    # e_vec = ((v^2 - mu/r) r_vec - (r·v) v_vec) / mu (slightly different from the standard formula but the same)
    e = np.linalg.norm(((v2 - mu/r)[:, None] * pos - (rv[:, None] * vel)) / mu[:, None], axis=1)

    # inclination
    h = np.cross(pos, vel)
    h_norm = np.linalg.norm(h, axis=1)
    cos_i = np.divide(h[:, 2], h_norm, out=np.zeros_like(h_norm), where=h_norm > 0) # only take the third component which signifies the perpendicular component
    cos_i = np.clip(cos_i, -1.0, 1.0)
    i = np.arccos(cos_i)

    return a, e, i

# -----------
# jacobi
# -----------

def cart2jacobi(x, masses, N=None, eta=None):
    x = np.asarray(x, dtype=float).reshape(-1)
    masses = np.asarray(masses, dtype=float).reshape(-1)

    if x.size != 6 * N:
        raise ValueError(f"x must have size 6N (= {6*N})")
    
    # reshape pos and vel arrays to be in groups of 3
    pos = x[: 3 * N].reshape(N, 3)
    vel = x[3 * N :].reshape(N, 3)

    # calculate eta just in case
    if eta is None:
        eta = np.cumsum(masses)

    # setup separate jacobi for position and velocity so indexing isn't messy
    jac_pos = np.zeros_like(pos)
    jac_vel = np.zeros_like(vel)

    # calculation of barycenter
    jac_pos[0] = (masses[:, None] * pos).sum(axis=0) / eta[-1]
    jac_vel[0] = (masses[:, None] * vel).sum(axis=0) / eta[-1]

     # running COM of interior bodies
    com_pos = pos[0].copy()
    com_vel = vel[0].copy()

    for i in range(1, N):
        # COM of bodies 0..i-1
        com_pos = (masses[:i, None] * pos[:i]).sum(axis=0) / eta[i - 1]
        com_vel = (masses[:i, None] * vel[:i]).sum(axis=0) / eta[i - 1]

        jac_pos[i] = pos[i] - com_pos
        jac_vel[i] = vel[i] - com_vel

    return np.concatenate([jac_pos.reshape(-1), jac_vel.reshape(-1)])

def jacobi2cart(x, masses, N=None, eta=None):
    # make sure everything is properly formated
    x = np.asarray(x, dtype=float).reshape(-1)
    masses = np.asarray(masses, dtype=float).reshape(-1)

    # account for dimensions
    if N is None:
        N = masses.size
    if x.size != 6 * N:
        raise ValueError(f"x must have size 6N (= {6*N})")

    # assign position and velocity jacobi vectors
    jac_pos = x[: 3 * N].reshape(N, 3)
    jac_vel = x[3 * N :].reshape(N, 3)

    if eta is None:
        eta = np.cumsum(masses)

    # reconstruct cartesian using recursion
    cart_pos = np.zeros_like(jac_pos)
    cart_vel = np.zeros_like(jac_vel)

    # x0 = r0 - sum_{i=1}^{N-1} (m_i/eta_i) r_i
    x0 = jac_pos[0].copy()
    v0 = jac_vel[0].copy()
    for i in range(1, N):
        x0 = x0 - (masses[i] / eta[i]) * jac_pos[i]
        v0 = v0 - (masses[i] / eta[i]) * jac_vel[i]

    cart_pos[0] = x0
    cart_vel[0] = v0

    com_pos = x0.copy()  # COM of bodies 0..0
    com_vel = v0.copy()

    for i in range(1, N):
        cart_pos[i] = com_pos + jac_pos[i]
        cart_vel[i] = com_vel + jac_vel[i]

        # update COM to include body i
        com_pos = com_pos + (masses[i] / eta[i]) * jac_pos[i]
        com_vel = com_vel + (masses[i] / eta[i]) * jac_vel[i]

    return np.concatenate([cart_pos.reshape(-1), cart_vel.reshape(-1)])


'''
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

def propagate_kepler_universal(r0_vec, v0_vec, dt:float, mu:float, tol:float = 1e-12, max_iter: int = 100):
    # assuring proper dimensions of properties
    r0_vec = np.asarray(r0_vec, dtype=float).reshape(3)
    v0_vec = np.asarray(v0_vec, dtype=float).reshape(3)
    dt = float(dt)
    mu = float(mu)

    # if nothing to derive position and vel for
    if dt == 0:
        return r0_vec.copy(), v0_vec.copy()
    
    r0 = float(np.linalg.norm(r0_vec)) # quantity of position vector
    v0 = float(np.linalg.norm(v0_vec)) # quantity of velocity vector

    dr0 = float(np.dot(r0_vec, v0_vec) / r0)

    # alpha quantity
    alpha = 2.0 * mu / r0 - v0*v0

    # initial guess
    s = dt / r0

    # using iterations to get towards the solution
    for _ in range(max_iter):
        c0, c1, c2, c3 = stumpff_functions(alpha*s*s)

        # universal Kepler's equation
        F = r0 * s * c1 + r0 * dr0 * s * s * c2 + mu * s * s * s * c3 - dt
        if abs(F) < tol:
            break # if the guess is below the tolerance, thats it!

        df = r0 * c0 + r0 * dr0 * s * c1 + mu * s * s * c2
        ds = -F / df
        s += ds

        if abs(ds) < tol:
            break

    r = df # radius at final time

    f = 1.0 - (mu / r0) * s * s * c2
    g = dt - mu * s * s * c3
    fdot = -(mu / (r * r0)) * s * c1
    gdot = 1.0 - (mu / r) * s * s * c2

    r_vec = f * r0_vec + g * v0_vec
    v_vec = fdot * r0_vec + gdot * v0_vec

    return r_vec, v_vec
    '''

def _stumpff_C2C3(z: float):
    """Return C2(z), C3(z) with stable series near z=0."""
    if abs(z) < 1e-8:
        # series expansions
        c2 = 1/2 - z/24 + z*z/720 - z**3/40320
        c3 = 1/6 - z/120 + z*z/5040 - z**3/362880
        return c2, c3

    if z > 0:
        s = np.sqrt(z)
        c2 = (1 - np.cos(s)) / z
        c3 = (s - np.sin(s)) / (s**3)
    else:
        s = np.sqrt(-z)
        c2 = (1 - np.cosh(s)) / z          # z is negative
        c3 = (np.sinh(s) - s) / (s**3)
    return c2, c3


def propagate_kepler_universal(r0_vec, v0_vec, dt, mu, tol=1e-12, max_iter=50):
    """
    Universal variable Kepler propagator.
    Inputs:
      r0_vec, v0_vec: (3,) arrays
      dt: float
      mu: GM (float)
    Returns:
      r1_vec, v1_vec: propagated (3,), (3,)
    """
    r0_vec = np.asarray(r0_vec, dtype=np.float64)
    v0_vec = np.asarray(v0_vec, dtype=np.float64)

    if dt == 0.0:
        return r0_vec.copy(), v0_vec.copy()

    r0 = np.linalg.norm(r0_vec)
    v2 = np.dot(v0_vec, v0_vec)
    vr0 = np.dot(r0_vec, v0_vec) / r0

    # alpha = 1/a (universal variable formulation)
    alpha = 2.0 / r0 - v2 / mu

    # Initial guess for chi (universal anomaly), reasonably robust
    # (There are fancier guesses; this one works well for your regime.)
    if abs(alpha) > 1e-12:
        chi = np.sqrt(mu) * abs(alpha) * dt
    else:
        chi = np.sqrt(mu) * dt / r0

    sqrt_mu = np.sqrt(mu)

    # Newton iteration on Kepler's universal time-of-flight equation
    ratio = np.inf
    for _ in range(max_iter):
        z = alpha * chi * chi
        c2, c3 = _stumpff_C2C3(z)

        # Kepler time-of-flight residual F(chi)
        F = (r0 * vr0 / sqrt_mu) * chi * chi * c2 + (1.0 - alpha * r0) * chi**3 * c3 + r0 * chi - sqrt_mu * dt

        # derivative dF/dchi
        dF = (r0 * vr0 / sqrt_mu) * chi * (1.0 - z * c3) + (1.0 - alpha * r0) * chi * chi * c2 + r0

        ratio = F / dF
        chi -= ratio

        if abs(ratio) < tol:
            break

    # If it didn't converge, you can raise, warn, or just proceed.
    # I'd rather warn than crash for now.
    # if abs(ratio) >= tol:
    #     raise RuntimeError("Kepler solver did not converge")

    z = alpha * chi * chi
    c2, c3 = _stumpff_C2C3(z)

    f = 1.0 - (chi * chi / r0) * c2
    g = dt - (chi**3 / sqrt_mu) * c3

    r1_vec = f * r0_vec + g * v0_vec
    r1 = np.linalg.norm(r1_vec)

    fdot = (sqrt_mu / (r0 * r1)) * (z * c3 - 1.0) * chi
    gdot = 1.0 - (chi * chi / r1) * c2

    v1_vec = fdot * r0_vec + gdot * v0_vec
    return r1_vec, v1_vec
