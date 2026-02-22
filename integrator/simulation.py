from data_io import DataIO
from physics import Physics
import constants
import tools
import numpy as np

class Simulation(object):
    def __init__(self, particles, integrator):
        self.particles = particles
        self.integrator = integrator
        self.physics = Physics
        self.dataio = DataIO

    def run(self, t0, tf, dt, output_every_n=1):
        # resolve primary
        spec = getattr(self.particles, 'primary', None)
        if spec is None:
            self.Ms, self.rs, self.vs = self.resolve_primary(self.particles, '#COM#')        

        # initialize the integrator
        self.integrator.bind(self.particles, t0, tf, dt)

        # initialize buffer
        self.dataio.initialize_buffer(self.particles.N)

        # track time and step
        t = t0
        step = 0

        # store initial snapshot
        self._store_snapshot(t)

        # time loop
        while t < tf:
            # perform step
            self.integrator.propagate()
            # update time and step
            t += dt
            step += 1

            # handle collisions
            for i in range(self.particles.N):
                for j in range(i, self.particles.N):
                    self.physics.collision(self.particles, i, j)

            if step % output_every_n == 0:
                self._store_snapshot(t)

            # redo COM calculation in case of collision or the motion of the center of motion
            self.Ms, self.rs, self.vs = self.physics.center_of_mass(self.particles)
        
        self.dataio.close()

    def _store_snapshot(self, t):
        E = self.physics.energy(self.particles)
        temp = self.physics.calculate_temp(self.particles)

        a, e, i = tools.aei(
                mp = self.particles.masses,
                ms = self.Ms,
                pos=np.array([self.particles._pos - self.rs]),
                vel=np.array([self.particles._vel - self.vs]),
                G=constants.G
            )
        
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
            i=i,
            energy=E
        )
