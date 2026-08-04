class WisdomHolman_SC_P(object):
    '''
    Wisdom-Holman integrator for S(C)-P type triple hierarchical systems.

    Stars A & B form an inner binary; star C orbits their COM; planet(s) orbit star C.

    Body order in HJS space: A, B, C, planet_0, ..., planet_{N-1}
    HJS coordinates (rows of transform matrix M):
      Row 0      : X_B    = r_B    - r_A
      Row 1      : X_C    = r_C    - COM(A + B)
      Row 2..N+1 : X_{Pj} = r_{Pj} - r_C
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

        # Collect planets (ptype == 1) — these orbit star C.
        planet_idx = particles.planet_indices
        self.planet_indices = planet_idx
        self.m_planets = particles.masses[planet_idx].copy()
        self.N_planets = len(self.m_planets)

        # Composite masses for the S(C)-P hierarchy.
        self.m_inner = self.mA + self.mB
        self.m_mid   = self.m_inner + self.mC
        self.M_total = self.m_mid + float(np.sum(self.m_planets))

        # GM parameters for Kepler solvers.
        self.GM_B = self.G * self.m_inner
        self.GM_C = self.G * (self.m_inner + self.mC)
        self.GM_planets = np.asarray(self.G * (self.mC + self.m_planets), dtype=np.float64)

        # HJS body order in the particles array: A, B, C, planet_0..N-1
        self.hjs_order = np.array(
            [self._iA, self._iB, self._iC] + list(self.planet_indices),
            dtype=np.int64,
        )

        # Pre-compute masses array in HJS body order for accel_pairs
        self._masses = np.array(
            [self.mA, self.mB, self.mC] + list(self.m_planets),
            dtype=np.float64,
        )

        self._build_transform_matrices()

    def _build_transform_matrices(self):
        N = self.N_planets
        n = N + 3

        M = np.zeros((n, n))

        M[0, 0] = -1.0
        M[0, 1] =  1.0

        M[1, 0] = -self.mA / self.m_inner
        M[1, 1] = -self.mB / self.m_inner
        M[1, 2] =  1.0

        for j in range(N):
            M[j+2, 2]   = -1.0
            M[j+2, 3+j] =  1.0

        M[N+2, 0] = self.mA / self.M_total
        M[N+2, 1] = self.mB / self.M_total
        M[N+2, 2] = self.mC / self.M_total
        for j in range(N):
            M[N+2, 3+j] = self.m_planets[j] / self.M_total

        self.M     = np.ascontiguousarray(M)
        self.M_inv = np.ascontiguousarray(np.linalg.inv(M))

    def propagate(self):
        """Single DKD step — used when the simulation loop is in Python."""
        p = self.particles
        hjs = self.hjs_order
        n = self.N_planets + 3

        x = p._pos[hjs].copy()
        v = p._vel[hjs].copy()

        X = self.M @ x
        V = self.M @ v

        half_dt = self.dt * 0.5

        self._kepler_drift(X, V, half_dt)
        self._kick(X, V, self.dt)
        self._kepler_drift(X, V, half_dt)

        x_new = self.M_inv @ X
        v_new = self.M_inv @ V
        for li in range(n):
            gi = hjs[li]
            p._pos[gi] = x_new[li]
            p._vel[gi] = v_new[li]

    def _kepler_drift(self, X, V, dt):
        X[0], V[0] = tools.propagate_kepler_universal(X[0], V[0], dt, self.GM_B)
        X[1], V[1] = tools.propagate_kepler_universal(X[1], V[1], dt, self.GM_C)
        for j in range(self.N_planets):
            k = 2 + j
            X[k], V[k] = tools.propagate_kepler_universal(X[k], V[k], dt, self.GM_planets[j])
        X[self.N_planets + 2] += dt * V[self.N_planets + 2]

    def _compute_accel(self, X):
        n = self.N_planets + 3
        x = self.M_inv @ X
        a = tools.accel_pairs(x, self._masses, self.G, n)
        A = self.M @ a

        x0 = X[0]; r2 = x0[0]*x0[0] + x0[1]*x0[1] + x0[2]*x0[2]
        A[0] += self.GM_B * x0 / (r2 * r2**0.5)

        x1 = X[1]; r2 = x1[0]*x1[0] + x1[1]*x1[1] + x1[2]*x1[2]
        A[1] += self.GM_C * x1 / (r2 * r2**0.5)

        for j in range(self.N_planets):
            k = 2 + j
            xk = X[k]; r2 = xk[0]*xk[0] + xk[1]*xk[1] + xk[2]*xk[2]
            A[k] += self.GM_planets[j] * xk / (r2 * r2**0.5)

        A[self.N_planets + 2] = 0.0
        return A

    def _kick(self, X, V, dt):
        V += dt * self._compute_accel(X)
