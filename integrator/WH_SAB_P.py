import numpy as np
import tools

class WisdomHolman_SAB_P(object):
    '''
    Wisdom-Holman integrator for S(AB)-P type triple hierarchical systems.

    Body order in HJS space: A, B, planet_0, ..., planet_{N-1}, C
    HJS coordinates (rows of transform matrix M):
      Row 0      : X_B   = r_B - r_A
      Row 1..N   : X_Pj  = r_{Pj} - COM(AB)
      Row N+1    : X_C   = r_C  - COM(AB + planets)
      Row N+2    : X_COM = total barycenter
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

        # Collect circumbinary planets (ptype == 1).
        planet_idx = particles.planet_indices          # global array indices
        self.planet_indices = planet_idx
        self.m_planets = particles.masses[planet_idx].copy()  # shape (N_planets,)
        self.N_planets = len(self.m_planets)

        # Composite masses for the S(AB)-P hierarchy.
        self.mb      = self.mA + self.mB
        self.mt      = self.mb + float(np.sum(self.m_planets))
        self.M_total = self.mt + self.mC

        # GM parameters for Kepler solvers.
        self.GM_binary = self.G * self.mb   # B orbits A
        self.GM_planet = self.G * self.mb   # planets orbit binary COM
        self.GM_outer  = self.G * self.mt   # C orbits (AB + planets) COM

        # HJS body order in the particles array: A, B, planet_0..N-1, C
        self.hjs_order = np.array(
            [self._iA, self._iB] + list(self.planet_indices) + [self._iC],
            dtype=int,
        )

        self._build_transform_matrices()

    def _build_transform_matrices(self):
        N = self.N_planets
        n = N + 3           # A, B, N planets, C  (COM row also lives here)

        M = np.zeros((n, n))

        # Row 0: X_B = r_B - r_A
        M[0, 0] = -1.0
        M[0, 1] =  1.0

        # Rows 1..N: X_{Pj} = r_{Pj} - (mA*r_A + mB*r_B) / mb
        for j in range(N):
            M[j+1, 0]   = -self.mA / self.mb
            M[j+1, 1]   = -self.mB / self.mb
            M[j+1, 2+j] =  1.0

        # Row N+1: X_C = r_C - COM(AB + planets)
        M[N+1, 0] = -self.mA / self.mt
        M[N+1, 1] = -self.mB / self.mt
        for j in range(N):
            M[N+1, 2+j] = -self.m_planets[j] / self.mt
        M[N+1, 2+N] = 1.0

        # Row N+2: X_COM = total barycenter
        M[N+2, 0] = self.mA / self.M_total
        M[N+2, 1] = self.mB / self.M_total
        for j in range(N):
            M[N+2, 2+j] = self.m_planets[j] / self.M_total
        M[N+2, 2+N] = self.mC / self.M_total

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

        # Inner binary: B orbits A
        X[0], V[0] = tools.propagate_kepler_universal(X[0], V[0], dt, self.GM_binary)

        # Circumbinary planets: each orbits binary COM
        for j in range(self.N_planets):
            X[1+j], V[1+j] = tools.propagate_kepler_universal(
                X[1+j], V[1+j], dt, self.GM_planet
            )

        # Outer star C: orbits COM of (AB + planets)
        X[1+self.N_planets], V[1+self.N_planets] = tools.propagate_kepler_universal(
            X[1+self.N_planets], V[1+self.N_planets], dt, self.GM_outer
        )

        # Total COM drifts linearly
        X[2+self.N_planets] += dt * V[2+self.N_planets]

        return X, V

    def _compute_accel(self, X, V):
        N = self.N_planets
        n = N + 3   # A, B, planets..., C

        # Recover Cartesian positions and velocities
        x, _ = self.to_cart(X, V)  # (n, 3)

        # Pairwise Cartesian accelerations for all physical bodies
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
        A = self.M @ a  # (n, 3)

        # Subtract Keplerian components so only the perturbation remains.
        # A_kep = -GM * X / |X|^3  =>  A_pert = A_full - A_kep = A_full + GM * X / |X|^3

        # Binary
        R_B = np.linalg.norm(X[0])
        A[0] += self.GM_binary * X[0] / R_B**3

        # Each planet
        for j in range(N):
            R_j = np.linalg.norm(X[1+j])
            A[1+j] += self.GM_planet * X[1+j] / R_j**3

        # Outer star C
        R_C = np.linalg.norm(X[1+N])
        A[1+N] += self.GM_outer * X[1+N] / R_C**3

        # COM coordinate: no perturbation (COM drifts freely)
        A[2+N] = np.zeros(3)

        return A

    def _kick(self, X, V, dt):
        acc = self._compute_accel(X, V)
        V = V.copy()
        V += dt * acc
        return V
