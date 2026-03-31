'''

Tools is a helper Python script used for housing various functions for coordinate transformations, helper functions for integration, etc.

'''

import numpy as np
import math
from numba import njit


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

# HIERARCHICAL JACOBI
# -----

def cart2HJS(x, M, N=None):
    x = np.asarray(x, dtype=float).reshape(-1)

    M = np.asarray(M, dtype=float)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError("M must be a square matrix")
    if N is None:
        N = M.shape[0]
    else:
        N = int(N)

    if x.size != 6 * N:
        raise ValueError(f"x must have size 6N (= {6*N})")

    if M.shape != (N, N):
        raise ValueError(f"M must have shape ({N}, {N})")

    pos = x[:3 * N].reshape(N, 3)
    vel = x[3 * N:].reshape(N, 3)

    hjs_pos = M @ pos
    hjs_vel = M @ vel

    return np.concatenate([hjs_pos.reshape(-1), hjs_vel.reshape(-1)])

def HJS2cart(x, M, N=None):
    x = np.asarray(x, dtype=float).reshape(-1)

    M = np.asarray(M, dtype=float)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError("M must be a square matrix")
    if N is None:
        N = M.shape[0]
    else:
        N = int(N)

    if x.size != 6 * N:
        raise ValueError(f"x must have size 6N (= {6*N})")

    if M.shape != (N, N):
        raise ValueError(f"M must have shape ({N}, {N})")

    hjs_pos = x[:3 * N].reshape(N, 3)
    hjs_vel = x[3 * N:].reshape(N, 3)

    M_inv = np.linalg.inv(M)

    pos = M_inv @ hjs_pos
    vel = M_inv @ hjs_vel

    return np.concatenate([pos.reshape(-1), vel.reshape(-1)])

# -------

@njit
def stumpff_functions(z):
    z = float(z)
    az = abs(z)

    if az < 1.0e-10:
        z2 = z * z
        z3 = z2 * z
        z4 = z3 * z
        c0 = 1.0 - z / 2.0 + z2 / 24.0 - z3 / 720.0 + z4 / 40320.0
        c1 = 1.0 - z / 6.0 + z2 / 120.0 - z3 / 5040.0 + z4 / 362880.0
        c2 = 0.5 - z / 24.0 + z2 / 720.0 - z3 / 40320.0 + z4 / 3628800.0
        c3 = 1.0 / 6.0 - z / 120.0 + z2 / 5040.0 - z3 / 362880.0 + z4 / 39916800.0
    elif z > 0.0:
        s = math.sqrt(z)
        c0 = math.cos(s)
        c1 = math.sin(s) / s
        c2 = (1.0 - c0) / z
        c3 = (1.0 - c1) / z
    else:
        s = math.sqrt(-z)
        c0 = math.cosh(s)
        c1 = math.sinh(s) / s
        c2 = (1.0 - c0) / z
        c3 = (1.0 - c1) / z

    if az < 1.0e-10:
        c4 = 1.0 / 24.0 - z / 720.0 + (z * z) / 40320.0
        c5 = 1.0 / 120.0 - z / 5040.0 + (z * z) / 362880.0
    else:
        c4 = (0.5 - c2) / z
        c5 = (1.0 / 6.0 - c3) / z

    return c0, c1, c2, c3, c4, c5

@njit
def propagate_kepler_universal(r0_vec, v0_vec, dt, mu, tol=1e-12, max_iter=80):
    """
    Universal-variable Kepler drift (book-style Stumpff solve).

    Inputs:
      r0_vec, v0_vec : initial relative state vectors, shape (3,)
      dt             : timestep
      mu             : gravitational parameter, G*(M+m)
    Returns:
      r1_vec, v1_vec : propagated relative state vectors, shape (3,)
    """
    dt = float(dt)
    mu = float(mu)

    if dt == 0.0:
        return r0_vec.copy(), v0_vec.copy()
    if mu <= 0.0:
        raise ValueError("mu must be positive")

    r0 = float(np.linalg.norm(r0_vec))
    if r0 == 0.0:
        raise ValueError("initial position norm must be non-zero")

    v0sq = float(np.dot(v0_vec, v0_vec))
    dr0 = float(np.dot(r0_vec, v0_vec)) / r0
    alpha = 2.0 * mu / r0 - v0sq

    s = dt / r0
    converged = False
    F = np.nan

    tol_F = tol * max(1.0, abs(dt))
    for _ in range(max_iter):
        z = alpha * s * s
        c0, c1, c2, c3, _, _ = stumpff_functions(z)

        s2 = s * s
        s3 = s2 * s
        F = r0 * s * c1 + r0 * dr0 * s2 * c2 + mu * s3 * c3 - dt
        dF = r0 * c0 + r0 * dr0 * s * c1 + mu * s2 * c2

        if dF == 0.0:
            break

        ds = -F / dF
        s += ds

        if abs(F) < tol_F or abs(ds) < tol:
            converged = True
            break

    if not converged:
        z = alpha * s * s
        c0, c1, c2, c3, _, _ = stumpff_functions(z)
        s2 = s * s
        s3 = s2 * s
        F = r0 * s * c1 + r0 * dr0 * s2 * c2 + mu * s3 * c3 - dt
        if abs(F) < 10.0 * tol_F:
            converged = True

    if not converged:
        raise RuntimeError("Kepler universal solve did not converge")

    z = alpha * s * s
    c0, c1, c2, c3, _, _ = stumpff_functions(z)
    s2 = s * s
    s3 = s2 * s
    r = r0 * c0 + r0 * dr0 * s * c1 + mu * s2 * c2
    if r <= 0.0:
        raise RuntimeError("invalid propagated radius")

    f = 1.0 - mu * s2 * c2 / r0
    g = dt - mu * s3 * c3
    fdot = -mu * s * c1 / (r * r0)
    gdot = 1.0 - mu * s2 * c2 / r

    r_vec1 = f * r0_vec + g * v0_vec
    v_vec1 = fdot * r0_vec + gdot * v0_vec
    return r_vec1, v_vec1


@njit
def accel_pairs(x, masses, G, n):
    """All-pairs gravitational acceleration (Numba-jitted)."""
    a = np.zeros((n, 3))
    for k in range(n):
        for j in range(n):
            if j == k:
                continue
            rx = x[j, 0] - x[k, 0]
            ry = x[j, 1] - x[k, 1]
            rz = x[j, 2] - x[k, 2]
            r2 = rx * rx + ry * ry + rz * rz
            inv_r3 = 1.0 / (r2 * math.sqrt(r2))
            fac = G * masses[j] * inv_r3
            a[k, 0] += fac * rx
            a[k, 1] += fac * ry
            a[k, 2] += fac * rz
    return a
