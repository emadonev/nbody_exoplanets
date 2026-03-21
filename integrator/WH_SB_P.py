import numpy as np
from . import tools

class WisdomHolman_SB_P(object):
    '''
    Wisdom-Holman integrator for S(B)-P type triple hierarchical systems.

    Planet(s) orbit star B; star B orbits star A; star C is an outer companion.

    Body order in HJS space: A, B, planet_0, ..., planet_{N-1}, C
    HJS coordinates (rows of transform matrix M):
      Row 0      : X_B    = r_B - r_A
      Row 1..N   : X_{Pj} = r_{Pj} - r_B
      Row N+1    : X_C    = r_C - COM(A + B + planets)
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

        # Collect planets (ptype == 1) — these orbit star B.
        planet_idx = particles.planet_indices
        self.planet_indices = planet_idx
        self.m_planets = particles.masses[planet_idx].copy()  # shape (N_planets,)
        self.N_planets = len(self.m_planets)

        # Composite masses for the S(B)-P hierarchy.
        self.m_inner = self.mB + float(np.sum(self.m_planets))  # B + planets
        self.m_mid   = self.m_inner + self.mA                   # A + B + planets
        self.M_total = self.m_mid + self.mC                     # everything

        # GM parameters for Kepler solvers.
        self.GM_B      = self.G * self.mA       # B orbits A
        self.GM_planet = self.G * self.mB       # planets orbit B
        self.GM_C      = self.G * self.m_mid    # C orbits COM(A + B + planets)

        # HJS body order in the particles array: A, B, planet_0..N-1, C
        self.hjs_order = np.array(
            [self._iA, self._iB] + list(self.planet_indices) + [self._iC],
            dtype=int,
        )

        self._build_transform_matrices()

    def _build_transform_matrices(self):
        N = self.N_planets
        n = N + 3  # A, B, N planets, C

        M = np.zeros((n, n))

        # Row 0: X_B = r_B - r_A
        # Columns: 0=A, 1=B, 2..N+1=planets, N+2=C
        M[0, 0] = -1.0   # -r_A
        M[0, 1] =  1.0   # +r_B

        # Rows 1..N: X_{Pj} = r_{Pj} - r_B
        for j in range(N):
            M[j+1, 1]   = -1.0   # -r_B
            M[j+1, 2+j] =  1.0   # +r_{Pj}

        # Row N+1: X_C = r_C - COM(A + B + planets)
        M[N+1, 0] = -self.mA / self.m_mid          # A
        M[N+1, 1] = -self.mB / self.m_mid          # B
        for j in range(N):
            M[N+1, 2+j] = -self.m_planets[j] / self.m_mid  # planet j
        M[N+1, N+2] = 1.0                          # +r_C

        # Row N+2: X_COM = total barycenter
        M[N+2, 0] = self.mA / self.M_total
        M[N+2, 1] = self.mB / self.M_total
        for j in range(N):
            M[N+2, 2+j] = self.m_planets[j] / self.M_total
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

        # Extract Cartesian state in HJS body order (A, B, planets..., C)
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

        # Row 0: B orbits A
        X[0], V[0] = tools.propagate_kepler_universal(X[0], V[0], dt, self.GM_B)

        # Rows 1..N: planets orbit B
        for j in range(N):
            X[1+j], V[1+j] = tools.propagate_kepler_universal(
                X[1+j], V[1+j], dt, self.GM_planet
            )

        # Row N+1: C orbits COM(A + B + planets)
        X[N+1], V[N+1] = tools.propagate_kepler_universal(
            X[N+1], V[N+1], dt, self.GM_C
        )

        # Row N+2: total COM drifts linearly
        X[N+2] += dt * V[N+2]

        return X, V

    def _compute_accel(self, X, V):
        N = self.N_planets
        n = N + 3  # A, B, planets..., C

        # Recover Cartesian positions (body order matches hjs_order: A, B, planets..., C)
        x, _ = self.to_cart(X, V)

        masses = np.concatenate([[self.mA, self.mB], self.m_planets, [self.mC]])
        a = np.zeros((n, 3))
        for k in range(n):
            for j in range(n):
                if j == k:
                    continue
                r_vec = x[j] - x[k]
                r = np.linalg.norm(r_vec)
                a[k] += self.G * masses[j] * r_vec / r**3

        # Transform to HJS accelerations: A_full = M @ a_cart
        A = self.M @ a

        # Subtract Keplerian components so only the perturbation remains.
        # A_kep = -GM * X / |X|^3  =>  A_pert = A_full + GM * X / |X|^3

        # Row 0: B
        R_B = np.linalg.norm(X[0])
        A[0] += self.GM_B * X[0] / R_B**3

        # Rows 1..N: planets
        for j in range(N):
            R_j = np.linalg.norm(X[1+j])
            A[1+j] += self.GM_planet * X[1+j] / R_j**3

        # Row N+1: C
        R_C = np.linalg.norm(X[N+1])
        A[N+1] += self.GM_C * X[N+1] / R_C**3

        # Row N+2: COM coordinate — no perturbation
        A[N+2] = np.zeros(3)

        return A

    def _kick(self, X, V, dt):
        acc = self._compute_accel(X, V)
        V = V.copy()
        V += dt * acc
        return V
