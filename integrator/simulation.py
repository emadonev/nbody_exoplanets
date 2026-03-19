import numpy as np
from data_io import DataIO
from physics import Physics
import tools

class Simulation(object):
    def __init__(self, particles, integrator, dataio: DataIO, physics: Physics):
        self.particles = particles
        self.integrator = integrator
        self.physics = physics or Physics()
        self.dataio = dataio or DataIO(const_g=particles.g)

    def run(self, t0, tf, dt, output_every_n=1, handle_collisions=False):
        # orbital-element reporting frame — defaults to total COM
        self.Ms, self.rs, self.vs = self.particles.resolve_primary_state(self.particles.primary)

        # initialize the integrator
        self.integrator.bind(self.particles, t0, tf, dt)

        # initialize buffer
        self.dataio.initialize_buffer(self.particles.N)

        # track time and step
        t = float(t0)
        step = 0

        # store initial snapshot
        self._store_snapshot(t)

        # time loop
        while t < tf:
            # perform step
            self.integrator.propagate()

            # synchronize objects
            if hasattr(self.particles, '_sync_objects'):
                self.particles._sync_objects()

            # handle collisions
            if handle_collisions:
                self._handle_collisions()

            # update time and step
            t += dt
            step += 1

            self.Ms, self.rs, self.vs = self.particles.resolve_primary_state(self.particles.primary)

            if step % int(output_every_n) == 0:
                self._store_snapshot(t)
        
        self.dataio.close()

    def _handle_collisions(self):
        active = self.particles.active_indices

        for a_i, i in enumerate(active):
            if self.particles.masses[i] <= 0: # if the mass is less than zero (deactivated)
                continue
            for j in active[a_i + 1 :]: # for every other particle to the end of the list
                if self.particles.masses[j] <= 0: # if this particle is deactivated
                    continue
                self.physics.collision(self.particles, int(i), int(j)) # if they are all active particles, perform collision mechanics

    def _store_snapshot(self, t):
        E = self.physics.energy(self.particles) # compute energy at this step
        temp = self.physics.calculate_temp(self.particles) # compute temperatures at this step

        pos_rel = self.particles.pos - self.rs # position relative to the PRIMARY
        vel_rel = self.particles.vel - self.vs # velocity relative to the PRIMARY

        # allocate for a, e, i
        a = np.full(self.particles.N, np.nan, dtype=float)
        e = np.full(self.particles.N, np.nan, dtype=float)
        inc = np.full(self.particles.N, np.nan, dtype=float)

        # Orbital elements are only meaningful for orbiting bodies.
        # Exclude stars/primaries to avoid r=0 singularities in diagnostics.
        orbit_idx = self.particles.planet_indices
        if orbit_idx.size > 0:
            mp = self.particles.masses[orbit_idx]
            a_act, e_act, inc_act = tools.aei(
                mp=mp,
                Ms=self.Ms,
                pos=pos_rel[orbit_idx],
                vel=vel_rel[orbit_idx],
                G=self.particles.g,
            )
            a[orbit_idx] = a_act
            e[orbit_idx] = e_act
            inc[orbit_idx] = inc_act
        
        self.dataio.store_state(
            t=t,
            pos=self.particles.positions,
            vel=self.particles.velocities,
            masses=self.particles.masses,
            temperature=temp,
            radii=self.particles.radii,
            hashes=self.particles.hashes,
            ptypes=self.particles.ptypes,
            a=a,
            e=e,
            inc=inc,
            energy=E
        )
