import numpy as np
import tools

class WisdomHolman_HJS(object):
    def __init__(self):
        self.particles = None
        self.t0 = None
        self.tf = None
        self.dt = None
        self.central = 0

        self.hierarchy = None
        self.active_indices = np.zeros(0, dtype=int)
        self.n_active = 0
        self.M_hjs = None
        self.M_hjs_inv = None
        self.orbit_nodes = []

    def bind(self, particles, t0, tf, dt):
        self.particles = particles
        self.t0 = float(t0)
        self.tf = float(tf)
        self.dt = float(dt)
        self._rebuild_hierarchy()


    def propagate(self):
        if self.particles is None:
            raise RuntimeError("integrator is not bound to particles")

        if self._needs_rebuild():
            self._rebuild_hierarchy()

        if self.n_active <= 1:
            return

        x_cart = self._get_active_cart()
        x_hjs = tools.cart2HJS(x_cart, self.M_hjs, self.n_active)

        # Drift-Kick-Drift (DKD)
        x_hjs = self._drift_hjs(x_hjs, 0.5 * self.dt)
        x_cart_mid = tools.HJS2cart(x_hjs, self.M_hjs, self.n_active)

        hjs_acc = self.compute_accel(x_cart_mid, x_hjs)
        hjs_vel = x_hjs[3 * self.n_active :].reshape(self.n_active, 3)
        hjs_vel[1:] += self.dt * hjs_acc[1:]
        x_hjs[3 * self.n_active :] = hjs_vel.reshape(-1)

        x_hjs = self._drift_hjs(x_hjs, 0.5 * self.dt)
        x_cart_new = tools.HJS2cart(x_hjs, self.M_hjs, self.n_active)
        self._set_active_cart(x_cart_new)


    def _needs_rebuild(self):
        current_active = self.particles.active_indices.astype(int)
        if current_active.size != self.active_indices.size:
            return True
        if current_active.size == 0:
            return False
        return np.any(current_active != self.active_indices)

    def _rebuild_hierarchy(self):
        self.active_indices = self.particles.active_indices.astype(int)
        self.n_active = int(self.active_indices.size)

        if self.n_active == 0:
            self.hierarchy = None
            self.M_hjs = np.zeros((0, 0), dtype=float)
            self.M_hjs_inv = np.zeros((0, 0), dtype=float)
            self.orbit_nodes = []
            return

        if self.n_active == 1:
            self.hierarchy = None
            self.M_hjs = np.eye(1, dtype=float)
            self.M_hjs_inv = np.eye(1, dtype=float)
            self.orbit_nodes = []
            return

        self.hierarchy = self.particles.tree_build()
        self.active_indices = self.particles.active_body_indices.copy()
        self.n_active = int(self.active_indices.size)
        self.M_hjs = np.asarray(self.particles.M_hjs, dtype=float)
        self.M_hjs_inv = np.linalg.inv(self.M_hjs)
        self.orbit_nodes = list(self.particles.orbit_nodes)

    def _get_active_cart(self):
        pos = self.particles.pos[self.active_indices]
        vel = self.particles.vel[self.active_indices]
        return np.concatenate([pos.reshape(-1), vel.reshape(-1)])

    def _set_active_cart(self, x_cart):
        pos = x_cart[: 3 * self.n_active].reshape(self.n_active, 3)
        vel = x_cart[3 * self.n_active :].reshape(self.n_active, 3)
        self.particles.pos[self.active_indices] = pos
        self.particles.vel[self.active_indices] = vel

    def _drift_hjs(self, x_hjs, h):
        x_hjs = np.asarray(x_hjs, dtype=float).copy().reshape(-1)
        hjs_pos = x_hjs[: 3 * self.n_active].reshape(self.n_active, 3)
        hjs_vel = x_hjs[3 * self.n_active :].reshape(self.n_active, 3)

        # Row 0 is the system barycenter.
        hjs_pos[0] = hjs_pos[0] + h * hjs_vel[0]

        # for every orbit_row and node in the tree, compute their paths
        for orbit_row, node in enumerate(self.orbit_nodes, start=1):
            mu = self.particles.g * node.submass
            r1, v1 = tools.propagate_kepler_universal(hjs_pos[orbit_row], hjs_vel[orbit_row], h, mu)
            hjs_pos[orbit_row] = r1
            hjs_vel[orbit_row] = v1

        x_hjs[: 3 * self.n_active] = hjs_pos.reshape(-1)
        x_hjs[3 * self.n_active :] = hjs_vel.reshape(-1)
        return x_hjs

    def _inv_r3(self, vec):
        r = np.linalg.norm(vec)
        if r == 0.0:
            return 0.0
        return 1.0 / (r * r * r)

    def compute_accel(self, x_cart, x_hjs):
        pos = x_cart[: 3 * self.n_active].reshape(self.n_active, 3)
        hjs_pos = x_hjs[: 3 * self.n_active].reshape(self.n_active, 3)
        masses = self.particles.masses[self.active_indices].astype(float)

        body_acc = np.zeros((self.n_active, 3), dtype=float)
        for i in range(self.n_active):
            for j in range(i + 1, self.n_active):
                rij = pos[j] - pos[i]
                inv_r3 = self._inv_r3(rij)
                if inv_r3 == 0.0:
                    continue
                fac = self.particles.g * inv_r3
                body_acc[i] += fac * masses[j] * rij
                body_acc[j] -= fac * masses[i] * rij

        hjs_acc = self.M_hjs @ body_acc
        hjs_acc[0] = 0.0

        # Remove the Kepler pieces already handled in the drift.
        for orbit_row, node in enumerate(self.orbit_nodes, start=1):
            r = hjs_pos[orbit_row]
            inv_r3 = self._inv_r3(r)
            if inv_r3 == 0.0:
                continue
            hjs_acc[orbit_row] -= self.particles.g * node.submass * r * inv_r3

        return hjs_acc
    
