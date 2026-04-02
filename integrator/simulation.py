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

        self._store_snapshot(t0)

        N = self.particles.N
        integ = self.integrator
        buf_len = self.dataio.buf_len

        # Run JIT loop in chunks of buf_len snapshots.
        # After each chunk, feed snapshots to DataIO which handles flushing to HDF5.
        # If the solver crashes, all previously fed chunks are already on disk.
        steps_done = 0
        chunk_snaps = buf_len
        buf_t = np.empty(chunk_snaps, dtype=np.float64)
        buf_pos = np.empty((chunk_snaps, N * 3), dtype=np.float64)
        buf_vel = np.empty((chunk_snaps, N * 3), dtype=np.float64)

        try:
            while steps_done < n_steps:
                remaining = n_steps - steps_done
                chunk_steps = min(remaining, chunk_snaps * output_every_n)

                actual_snaps = _dkd_loop(
                    self.particles._pos, self.particles._vel,
                    integ.hjs_order, integ.M, integ.M_inv, integ._masses,
                    integ.GM_B, integ.GM_C, integ.GM_planets,
                    integ.G, integ.N_planets, dt,
                    chunk_steps, output_every_n,
                    buf_pos, buf_vel, buf_t, t0 + steps_done * dt,
                )

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

                steps_done += chunk_steps
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
