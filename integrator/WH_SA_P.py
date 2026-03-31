import numpy as np
from . import tools

class WisdomHolman_SA_P(object):
    '''
    Wisdom-Holman integrator for S(A)-P type triple hierarchical systems.

    Planet(s) orbit star A; star B is a companion to A; star C is an outer companion.

    Body order in HJS space: A, planet_0, ..., planet_{N-1}, B, C
    HJS coordinates (rows of transform matrix M):
      Row 0..N-1 : X_{Pj} = r_{Pj} - r_A
      Row N      : X_B    = r_B    - COM(A + planets)
      Row N+1    : X_C    = r_C    - COM(A + planets + B)
      Row N+2    : X_COM  = total barycenter
    '''
    def __init__(self):
        self.particles = None
        self.t0 = None
        self.tf = None
        self.dt = None

    def bind(self, particles, t0, tf, dt):
        self.particles = particles
        self.t0 = float(t0)
        self.tf = float(tf)
        self.dt = float(dt)
        self.G = particles.g

        # Resolve stars A, B, C by their index label.
        self._iA, self.pA = particles.get_by_label("A")
        self._iB, self.pB = particles.get_by_label("B")
        self._iC, self.pC = particles.get_by_label("C")

        self.mA = float(particles.masses[self._iA])
        self.mB = float(particles.masses[self._iB])
        self.mC = float(particles.masses[self._iC])

        # Collect planets (ptype == 1) — these orbit star A.
        planet_idx = particles.planet_indices
        self.planet_indices = planet_idx
        self.m_planets = particles.masses[planet_idx].copy()  # shape (N_planets,)
        self.N_planets = len(self.m_planets)

        # Composite masses for the S(A)-P hierarchy.
        self.m_inner = self.mA + float(np.sum(self.m_planets))  # A + planets
        self.m_mid   = self.m_inner + self.mB                   # A + planets + B
        self.M_total = self.m_mid + self.mC                     # everything

        # GM parameters for Kepler solvers.
        self.GM_planet = self.G * self.mA       # planets orbit A
        self.GM_B      = self.G * self.m_inner  # B orbits COM(A + planets)
        self.GM_C      = self.G * self.m_mid    # C orbits COM(A + planets + B)

        # HJS body order in the particles array: A, planet_0..N-1, B, C
        self.hjs_order = np.array(
            [self._iA] + list(self.planet_indices) + [self._iB, self._iC],
            dtype=int,
        )

        self._build_transform_matrices()

    def _build_transform_matrices(self):
        N = self.N_planets
        n = N + 3  # A, N planets, B, C

        M = np.zeros((n, n))

        # Rows 0..N-1: X_{Pj} = r_{Pj} - r_A
        for j in range(N):
            M[j, 0]   = -1.0   # -r_A
            M[j, 1+j] =  1.0   # +r_{Pj}

        # Row N: X_B = r_B - COM(A + planets)
        M[N, 0] = -self.mA / self.m_inner
        for j in range(N):
            M[N, 1+j] = -self.m_planets[j] / self.m_inner
        M[N, N+1] = 1.0  # +r_B

        # Row N+1: X_C = r_C - COM(A + planets + B)
        M[N+1, 0] = -self.mA / self.m_mid
        for j in range(N):
            M[N+1, 1+j] = -self.m_planets[j] / self.m_mid
        M[N+1, N+1] = -self.mB / self.m_mid
        M[N+1, N+2] = 1.0  # +r_C

        # Row N+2: X_COM = total barycenter
        M[N+2, 0] = self.mA / self.M_total
        for j in range(N):
            M[N+2, 1+j] = self.m_planets[j] / self.M_total
        M[N+2, N+1] = self.mB / self.M_total
        M[N+2, N+2] = self.mC / self.M_total

        self.M     = M
        self.M_inv = np.linalg.inv(M)

    # HJS <-> Cartesian transforms (positions and velocities use the same M)
    def to_hjs(self, x, v):
        return self.M @ x, self.M @ v

    def to_cart(self, X, V):
        return self.M_inv @ X, self.M_inv @ V

    def propagate(self):
        if self.particles is None:
            raise RuntimeError("integrator is not bound to particles")
        if self.hjs_order.size <= 1:
            return

        # Extract Cartesian state in HJS body order (A, planets..., B, C)
        x = self.particles.pos[self.hjs_order].copy()  # (n, 3)
        v = self.particles.vel[self.hjs_order].copy()  # (n, 3)

        # Transform to HJS coordinates
        X, V = self.to_hjs(x, v)

        # Drift-Kick-Drift
        X, V = self._kepler_drift(X, V, self.dt / 2.0)
        V    = self._kick(X, V, self.dt)
        X, V = self._kepler_drift(X, V, self.dt / 2.0)

        # Transform back and write results into the particles arrays
        x_new, v_new = self.to_cart(X, V)
        for local_i, global_i in enumerate(self.hjs_order):
            self.particles._pos[global_i] = x_new[local_i]
            self.particles._vel[global_i] = v_new[local_i]

    def _kepler_drift(self, X, V, dt):
        X = X.copy()
        V = V.copy()

        N = self.N_planets

        # Planets (rows 0..N-1): each orbits star A
        for j in range(N):
            X[j], V[j] = tools.propagate_kepler_universal(
                X[j], V[j], dt, self.GM_planet
            )

        # Star B (row N): orbits COM(A + planets)
        X[N], V[N] = tools.propagate_kepler_universal(
            X[N], V[N], dt, self.GM_B
        )

        # Star C (row N+1): orbits COM(A + planets + B)
        X[N+1], V[N+1] = tools.propagate_kepler_universal(
            X[N+1], V[N+1], dt, self.GM_C
        )

        # Total COM drifts linearly (row N+2)
        X[N+2] += dt * V[N+2]

        return X, V

    def _compute_accel(self, X, V):
        N = self.N_planets
        n = N + 3  # A, planets..., B, C

        # Recover Cartesian positions (body order: A, P_0..P_{N-1}, B, C)
        x, _ = self.to_cart(X, V)

        masses = np.concatenate([[self.mA], self.m_planets, [self.mB, self.mC]])
        a = tools.accel_pairs(x, masses, self.G, n)

        # Transform to HJS accelerations: A_full = M @ a_cart
        A = self.M @ a

        # Subtract Keplerian components so only the perturbation remains.
        # A_kep = -GM * X / |X|^3  =>  A_pert = A_full + GM * X / |X|^3

        # Planets (rows 0..N-1)
        for j in range(N):
            R_j = np.linalg.norm(X[j])
            A[j] += self.GM_planet * X[j] / R_j**3

        # Star B (row N)
        R_B = np.linalg.norm(X[N])
        A[N] += self.GM_B * X[N] / R_B**3

        # Star C (row N+1)
        R_C = np.linalg.norm(X[N+1])
        A[N+1] += self.GM_C * X[N+1] / R_C**3

        # COM coordinate: no perturbation
        A[N+2] = np.zeros(3)

        return A

    def _kick(self, X, V, dt):
        acc = self._compute_accel(X, V)
        V = V.copy()
        V += dt * acc
        return V
