'''

Tools is a helper Python script used for housing various functions for coordinate transformations, helper functions for integration, etc.

'''

import numpy as np
import math

def orb_to_cartesian(a, e, i, Omega, omega, theta, mu, angles_in_degrees=False):
    """
    Convert Keplerian orbital elements to Cartesian state vectors.

    Inputs:
      a, e, i, Omega, omega, theta : classical elements
      mu                           : gravitational parameter G*(M_primary + m_body)
      angles_in_degrees            : set True if angular elements are in degrees
    Returns:
      r_vec, v_vec                 : relative position and velocity, shape (3,)
    """
    a = float(a)
    e = float(e)
    i = float(i)
    Omega = float(Omega)
    omega = float(omega)
    theta = float(theta)
    mu = float(mu)

    if angles_in_degrees:
        i = np.deg2rad(i)
        Omega = np.deg2rad(Omega)
        omega = np.deg2rad(omega)
        theta = np.deg2rad(theta)

    if a == 0.0:
        raise ValueError("semi-major axis a must be non-zero")
    if mu <= 0.0:
        raise ValueError(f"mu must be positive, got {mu}")

    p = a * (1.0 - e * e)
    if p <= 0.0:
        raise ValueError(f"invalid semilatus rectum p={p}; check a/e")

    cO = np.cos(Omega)
    sO = np.sin(Omega)
    co = np.cos(omega)
    so = np.sin(omega)
    ci = np.cos(i)
    si = np.sin(i)
    ct = np.cos(theta)
    st = np.sin(theta)

    # Perifocal basis vectors projected in inertial frame.
    p_hat = np.array(
        [cO * co - sO * so * ci, sO * co + cO * so * ci, so * si],
        dtype=float,
    )
    q_hat = np.array(
        [-cO * so - sO * co * ci, -sO * so + cO * co * ci, co * si],
        dtype=float,
    )

    r = p / (1.0 + e * ct)
    r_vec = r * (ct * p_hat + st * q_hat)
    v_vec = np.sqrt(mu / p) * (-st * p_hat + (e + ct) * q_hat)
    return r_vec, v_vec


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

def cart2jacobi(x, masses, N, eta):
    x = np.asarray(x, dtype=float).reshape(-1)
    masses = np.asarray(masses, dtype=float).reshape(-1)

    if masses.size != N:
        raise ValueError(f"masses must have size N (= {N})")
    if x.size != 6 * N:
        raise ValueError(f"x must have size 6N (= {6*N})")
    
    pos = x[: 3 * N].reshape(N, 3)
    vel = x[3 * N :].reshape(N, 3)

    if eta is None:
        eta = np.cumsum(masses)
    else:
        eta = np.asarray(eta, dtype=float).reshape(-1)
        if eta.size != N:
            raise ValueError(f"eta must have size N (= {N})")

    jac_pos = np.zeros_like(pos)
    jac_vel = np.zeros_like(vel)

    # Jacobi coordinate 0 stores the barycenter position/velocity.
    jac_pos[0] = np.sum(masses[:, None] * pos, axis=0) / eta[-1]
    jac_vel[0] = np.sum(masses[:, None] * vel, axis=0) / eta[-1]

    # Running COM of interior bodies 0..i-1.
    com_pos = pos[0].copy()
    com_vel = vel[0].copy()
    for i in range(1, N):
        jac_pos[i] = pos[i] - com_pos
        jac_vel[i] = vel[i] - com_vel

        com_pos = com_pos + (masses[i] / eta[i]) * jac_pos[i]
        com_vel = com_vel + (masses[i] / eta[i]) * jac_vel[i]

    return np.concatenate([jac_pos.reshape(-1), jac_vel.reshape(-1)])

def jacobi2cart(x, masses, N, eta):
    x = np.asarray(x, dtype=float).reshape(-1)
    masses = np.asarray(masses, dtype=float).reshape(-1)

    if masses.size != N:
        raise ValueError(f"masses must have size N (= {N})")
    if x.size != 6 * N:
        raise ValueError(f"x must have size 6N (= {6*N})")

    jac_pos = x[: 3 * N].reshape(N, 3)
    jac_vel = x[3 * N :].reshape(N, 3)

    if eta is None:
        eta = np.cumsum(masses)
    else:
        eta = np.asarray(eta, dtype=float).reshape(-1)
        if eta.size != N:
            raise ValueError(f"eta must have size N (= {N})")

    cart_pos = np.zeros_like(jac_pos)
    cart_vel = np.zeros_like(jac_vel)

    # Recover body 0 from barycenter and Jacobi offsets.
    cart_pos[0] = jac_pos[0].copy()
    cart_vel[0] = jac_vel[0].copy()
    for i in range(1, N):
        cart_pos[0] -= (masses[i] / eta[i]) * jac_pos[i]
        cart_vel[0] -= (masses[i] / eta[i]) * jac_vel[i]

    # Rebuild remaining bodies from running interior COM.
    com_pos = cart_pos[0].copy()
    com_vel = cart_vel[0].copy()
    for i in range(1, N):
        cart_pos[i] = jac_pos[i] + com_pos
        cart_vel[i] = jac_vel[i] + com_vel

        com_pos = com_pos + (masses[i] / eta[i]) * jac_pos[i]
        com_vel = com_vel + (masses[i] / eta[i]) * jac_vel[i]

    return np.concatenate([cart_pos.reshape(-1), cart_vel.reshape(-1)])


def stumpff_functions(z):
    # reducing z until the solution will be precise
    n = 0
    while (abs(z) > 0.1):
        n += 1
        z /= 4.0

    # c4 and c5
    c4 = 1/(math.factorial(4)) - z/(math.factorial(6))
    c5 = 1/(math.factorial(5)) - z/(math.factorial(7))

    z_conj = -z
    p = z_conj
    k = 8
    c4_prev = 0

    while c4 != c4_prev:
        c4_prev = c4
        p = p * z_conj
        c4 = c4 + p/(math.factorial(k))
        k += 1
        c5 = c5 + p/(math.factorial(k))
        k += 1
    
    c3 = 1/6 - z * c5
    c2 = 1/2 - z*c4
    c1 = 1 - z*c3

    while n > 0:
        z = 4*z
        c5 = 1/16 * (c5 + c4 + c3 + c2)
        c4 = 1/8 * c3 * (1-c1)
        c3 = 1/6 - z*c5
        c2 = 1/2 - z*c4
        c1 = 1 - z*c3
        n -= 1

    c0 = 1 - z*c2
    
    return c0, c1, c2, c3, c4, c5

def propagate_kepler_universal(r0_vec, v0_vec, dt, mu, tol=1e-13, max_iter=80):
    """
    Universal-variable Kepler drift with Stumpff functions and G-functions.

    Inputs:
      r0_vec, v0_vec : initial relative state vectors, shape (3,)
      dt             : timestep
      mu             : gravitational parameter, G*(M+m)
    Returns:
      r1_vec, v1_vec : propagated relative state vectors, shape (3,)
    """


    r0_vec = np.asarray(r0_vec, dtype=float).reshape(3)
    v0_vec = np.asarray(v0_vec, dtype=float).reshape(3)
    dt = float(dt)
    mu = float(mu)

    # if nothing to derive position and vel for
    if dt == 0:
        return r0_vec.copy(), v0_vec.copy()
    
    if mu <= 0.0:
        raise ValueError(f"mu must be positive, got {mu}")
    
    r0 = float(np.linalg.norm(r0_vec))
    if r0 == 0.0:
        raise ValueError("initial position norm must be non-zero")
    
    v0sq = float(np.dot(v0_vec, v0_vec))
    eta0 = float(np.dot(r0_vec, v0_vec))
    beta = 2.0 * mu / r0 - v0sq
    zeta0 = mu - beta * r0

    # initial guess
    X = dt / r0
    if beta > 0.0:
        X = np.sign(dt) * min(abs(X), 0.5 * np.pi / np.sqrt(beta))

    converged = False
    F = np.nan
    # using iterations to get towards the solution
    for _ in range(max_iter):
        
        z = beta * X * X
        c0, c1, c2, c3, _, _ = stumpff_functions(z)

        G1 = X * c1
        G2 = X * X * c2
        G3 = X * X * X * c3

        # Universal Kepler equation and derivative wrt X.
        F = r0 * G1 + eta0 * G2 + zeta0 * G3 - dt
        dF = r0 * c0 + eta0 * G1 + zeta0 * G2

        if dF == 0.0:
            break

        dX = -F / dF
        X += dX

        if abs(dX) < tol and abs(F) < tol:
            converged = True
            break

    if not converged:
        raise RuntimeError(
            "Kepler universal solve did not converge "
            f"(dt={dt}, mu={mu}, r0={r0}, X={X}, F={F}, max_iter={max_iter})"
        )

    z = beta * X * X
    c0, c1, c2, c3, _, _ = stumpff_functions(z)
    G1 = X * c1
    G2 = X * X * c2
    G3 = X * X * X * c3
    r = r0 * c0 + eta0 * G1 + zeta0 * G2

    if r <= 0.0:
        raise RuntimeError(f"invalid propagated radius r={r}")
    

    f = 1.0 - (mu / r0) * G2
    g = dt - mu * G3
    fdot = -(mu / (r * r0)) * G1
    gdot = 1.0 - (mu / r) * G2

    r_vec1 = f * r0_vec + g * v0_vec
    v_vec1 = fdot * r0_vec + gdot * v0_vec

    return r_vec1, v_vec1
