import numpy as np
from .data_io import DataIO
from .physics import Physics
from .WH_SC_P import _dkd_loop

class Simulation(object):
    def __init__(self, particles, integrator, dataio: DataIO, physics: Physics):
        self.particles = particles
        self.integrator = integrator
        self.physics = physics or Physics()
        self.dataio = dataio or DataIO(const_g=particles.g)

    def run(self, t0, tf, dt, output_every_n=1, handle_collisions=False):
        self.integrator.bind(self.particles, t0, tf, dt)
        self.dataio.initialize_buffer(self.particles.N)

        t0 = float(t0)
        dt = float(dt)
        n_steps = int(np.ceil((float(tf) - t0) / dt))
        output_every_n = int(output_every_n)
        n_snaps = n_steps // output_every_n

        # store initial snapshot
        self._store_snapshot(t0)

        # allocate snapshot buffers for the JIT loop
        N = self.particles.N
        buf_t = np.empty(n_snaps, dtype=np.float64)
        buf_pos = np.empty((n_snaps, N * 3), dtype=np.float64)
        buf_vel = np.empty((n_snaps, N * 3), dtype=np.float64)

        try:
            integ = self.integrator

            # run the entire integration in compiled code
            actual_snaps = _dkd_loop(
                self.particles._pos, self.particles._vel,
                integ.hjs_order, integ.M, integ.M_inv, integ._masses,
                integ.GM_B, integ.GM_C, integ.GM_planets,
                integ.G, integ.N_planets, dt,
                n_steps, output_every_n,
                buf_pos, buf_vel, buf_t, t0,
            )

            # flush all snapshots to HDF5
            for s in range(actual_snaps):
                self.dataio.store_state(
                    t=buf_t[s],
                    pos=buf_pos[s],
                    vel=buf_vel[s],
                    masses=self.particles.masses,
                    radii=self.particles.radii,
                    hashes=self.particles.hashes,
                    ptypes=self.particles.ptypes,
                )
        finally:
            self.dataio.close()

        return self.dataio.output_name

    def _store_snapshot(self, t):
        self.dataio.store_state(
            t=t,
            pos=self.particles.positions,
            vel=self.particles.velocities,
            masses=self.particles.masses,
            radii=self.particles.radii,
            hashes=self.particles.hashes,
            ptypes=self.particles.ptypes,
        )
