import numpy as np
import tools


class WisdomHolman(object):
    def __init__(self):
        self.particles = None
        self.t0 = None
        self.tf = None
        self.dt = None
        self.central = 0

    def bind(self, particles, t0, tf, dt):
        self.particles = particles
        self.t0 = float(t0)
        self.tf = float(tf)
        self.dt = float(dt)
        self._refresh_active_order()

    def propagate(self):
        if self.particles is None:
            raise RuntimeError("integrator is not bound to particles")

        self._refresh_active_order()
        if self.order.size <= 1:
            return

        x_cart = self._get_active_cart()
        x_jac = tools.cart2jacobi(x_cart, self.masses_ord, self.order.size, self.eta)

        # Drift-Kick-Drift (DKD)
        x_jac = self._drift_jacobi(x_jac, 0.5 * self.dt)
        x_cart_mid = tools.jacobi2cart(x_jac, self.masses_ord, self.order.size, self.eta)

        jac_acc = self.compute_accel(x_cart_mid, x_jac)
        N = self.order.size
        jac_vel = x_jac[3 * N :].reshape(N, 3)
        jac_vel[1:] += self.dt * jac_acc[1:]
        x_jac[3 * N :] = jac_vel.reshape(-1)

        x_jac = self._drift_jacobi(x_jac, 0.5 * self.dt)
        x_cart_new = tools.jacobi2cart(x_jac, self.masses_ord, self.order.size, self.eta)
        self._set_active_cart(x_cart_new)
        #jac_mom[1:] += self.dt * dPdt[1:]
       # jac_vel[1:] = jac_mom[1:] / mu_red[:, None]
        #x_jac[3 * N :] = jac_vel.reshape(-1)

        #x_jac = self._drift_jacobi(x_jac, 0.5 * self.dt)
        #x_cart_new = tools.jacobi2cart(x_jac, self.masses_ord, self.order.size, self.eta)
        #self._set_active_cart(x_cart_new)

    def _refresh_active_order(self):
        p = self.particles
        active = p.active_indices
        if active.size == 0:
            self.order = np.zeros(0, dtype=int)
            self.inv_order = {}
            self.masses_ord = np.zeros(0, dtype=float)
            self.eta = np.zeros(0, dtype=float)
            return

        # Central body for Jacobi ordering comes from Particles resolver.
        self.central = int(p.resolve_primary_index(getattr(p, "primary", "#COM#")))

        others = active[active != self.central]
        self.order = np.concatenate(([self.central], others)).astype(int)
        self.inv_order = {int(idx): i for i, idx in enumerate(self.order)}
        self.masses_ord = p.masses[self.order].astype(float)
        self.eta = np.cumsum(self.masses_ord)

    def _get_active_cart(self):
        pos = self.particles.pos[self.order]
        vel = self.particles.vel[self.order]
        return np.concatenate([pos.reshape(-1), vel.reshape(-1)])

    def _set_active_cart(self, x_cart):
        N = self.order.size
        pos = x_cart[: 3 * N].reshape(N, 3)
        vel = x_cart[3 * N :].reshape(N, 3)
        self.particles.pos[self.order] = pos
        self.particles.vel[self.order] = vel

    def _drift_jacobi(self, x_jac, h):
        x_jac = np.asarray(x_jac, dtype=float).copy().reshape(-1)
        N = self.order.size

        jac_pos = x_jac[: 3 * N].reshape(N, 3)
        jac_vel = x_jac[3 * N :].reshape(N, 3)

        # Coordinate 0 is the system barycenter and drifts linearly.
        jac_pos[0] = jac_pos[0] + h * jac_vel[0]

        m0 = self.masses_ord[0]
        for i in range(1, N):
            mu_i = self.particles.g * m0 * self.eta[i] / self.eta[i - 1]
            r1, v1 = tools.propagate_kepler_universal(jac_pos[i], jac_vel[i], h, mu_i)
            jac_pos[i] = r1
            jac_vel[i] = v1

        x_jac[: 3 * N] = jac_pos.reshape(-1)
        x_jac[3 * N :] = jac_vel.reshape(-1)
        return x_jac

    def _inv_r3(self, vec):
        r = np.linalg.norm(vec)
        if r == 0.0:
            return 0.0
        return 1.0 / (r * r * r)

    def compute_accel(self, x_cart, x_jac):
        # Return dP/dt (canonical momentum derivative) for Jacobi kick.
        N = self.order.size
        g = self.particles.g
        m = self.masses_ord
        eta = self.eta

        cart = x_cart[: 3 * N].reshape(N, 3)
        jac = x_jac[: 3 * N].reshape(N, 3)
        accel = np.zeros((N, 3), dtype=float)

        m0 = m[0]
        for i in range(1, N):
            eta_i = eta[i]
            eta_im1 = eta[i - 1]

            r0i = cart[i] - cart[0]
            accel[i] = g * m0 * eta_i / eta_im1 * (jac[i] / (np.linalg.norm(jac[i])**3) - r0i / (np.linalg.norm(r0i)**3))

            aux = np.zeros(3, dtype=float)
            for j in range(1, i):
                rji = cart[i] - cart[j]
                aux += g * m[j] * rji / (np.linalg.norm(rji)**3)
            accel[i] += -(eta_i / eta_im1) * aux

            aux = np.zeros(3, dtype=float)
            for j in range(i+1, N):
                rij = cart[j] - cart[i]
                aux += g * m[j] * rij / (np.linalg.norm(rij)**3)
            accel[i] += aux

            aux = np.zeros(3, dtype=float)
            for j in range(0, i):
                for k in range(i + 1, N):
                    rjk = cart[k] - cart[j]
                    aux += g * m[j] * m[k] * rjk * self._inv_r3(rjk)
            accel[i] -= aux / eta_im1

        return accel
    