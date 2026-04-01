import numpy as np
from .data_io import DataIO
from .physics import Physics

class Simulation(object):
    def __init__(self, particles, integrator, dataio: DataIO, physics: Physics):
        self.particles = particles
        self.integrator = integrator
        self.physics = physics or Physics()
        self.dataio = dataio or DataIO(const_g=particles.g)

    def run(self, t0, tf, dt, output_every_n=1, handle_collisions=False):
        # initialize the integrator
        self.integrator.bind(self.particles, t0, tf, dt)

        # initialize buffer
        self.dataio.initialize_buffer(self.particles.N)

        # total number of steps is fixed — independent of output_every_n
        n_steps = int(np.ceil((float(tf) - float(t0)) / float(dt)))

        # store initial snapshot
        self._store_snapshot(float(t0))

        # time loop driven by integer counter to avoid float accumulation
        for step in range(1, n_steps + 1):
            # perform step
            self.integrator.propagate()

            # synchronize objects
            if hasattr(self.particles, '_sync_objects'):
                self.particles._sync_objects()

            # handle collisions
            if handle_collisions:
                if self._handle_collisions():
                    break

            # compute time exactly from t0 to avoid drift
            t = float(t0) + step * float(dt)

            if step % int(output_every_n) == 0:
                self._store_snapshot(t)

        self.dataio.close()
        return self.dataio.output_name

    def _handle_collisions(self):
        active = self.particles.active_indices

        for a_i, i in enumerate(active):
            if self.particles.masses[i] <= 0: # if the mass is less than zero (deactivated)
                continue
            for j in active[a_i + 1 :]: # for every other particle to the end of the list
                if self.particles.masses[j] <= 0: # if this particle is deactivated
                    continue
                if self.physics.check_collision(self.particles, int(i), int(j)): # if they are all active particles, perform collision mechanics
                    return True

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
        return