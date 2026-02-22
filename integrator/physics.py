import numpy as np
from particle import Particle
from particles import Particles
import numbers
import tools
import constants

class Physics(object):
    def energy(self, particles):
        ke = 0.5 * np.sum(particles.masses * np.sum(particles._vel**2, axis=1))

        pe = 0.0
        for i in range(particles.N):
            for j in range(i+1, particles.N):
                d = np.linalg.norm(particles._pos[i] - particles._pos[j])
                pe -= constants.G * particles.masses[i] * particles.masses[j] / d
        return ke + pe
    
    def center_of_mass(self, particles, subset=None):
        # define the center of mass with position and velocity
        if subset is None:
            m = particles._mass
            pos = particles._pos
            vel = particles._vel
        else:
            idx = np.asarray(subset, dtype=int)
            m = particles._mass[idx]
            pos = particles._pos[idx]
            vel = particles._vel[idx]

        mtot = m.sum()
        com_pos = (m[:, None] * pos).sum(axis=0) / mtot
        com_vel = (m[:, None] * vel).sum(axis=0) / mtot

        return mtot, com_pos, com_vel
    
    def resolve_primary(self, particles, primary_spec=None):
        # how do you determine the primary body - multiple systems of using it

        if primary_spec is None:
            primary_spec = particles.primary

        # we define the center of mass as the primary body - most common
        if primary_spec == '#COM#':
            return self.center_of_mass(particles)
        
        # the primary body is the most massive body - save the same parameters as with COM
        if primary_spec == '#MAX#':
            i = int(np.argmax(particles._mass))
            return particles._mass[i], particles._pos[i].copy(), particles._vel[i].copy()

        # if primary_spec is a subset of indices - get the COM
        if isinstance(primary_spec, (list, tuple, np.ndarray)):
            return self.center_of_mass(particles, subset=primary_spec)
        
        # if the primary_spec is a name of what is our primary
        if isinstance(primary_spec, str):
            # if the name isn't found, raise an error
            if primary_spec not in particles.names:
                raise KeyError(f"No particle named {primary_spec}")
            
            # otherwise proceed as normal
            i = particles.names[primary_spec]
            return particles._mass[i], particles._pos[i].copy(), particles._vel[i].copy()

        # if the primary_spec is an index
        if isinstance(primary_spec, (int, np.integer)):
            i = int(primary_spec)
            return particles._mass[i], particles._pos[i].copy, particles.vel[i].copy()
        
        raise TypeError(f'Unsupported primary spec: {type(primary_spec)}')
    
    def collision(self, particles, id1, id2):
        # check if the 2 particles are in the system
        try:
            p1 = particles._particles[id1]
            p2 = particles._particles[id2]
        except ValueError:
            return -1

        if p1 is not None and p2 is not None:
            if p1.m > p2.m:
                # body 1 is more massive so merge 2 into 1
                p1.vel = (p1.m * p1.vel + p2.m * p2.vel)/(p1.m + p2.m) # the new velocity is gained from the conservation of momentum
                p1.m = p1.m + p2.m # add the masses together
                p1.r = np.power(np.power(p1.r, 3.0) + np.power(p2.r, 3.0), 1.0/3) # add the radii as cubes and then take the cube root (adding volumes)

                # if any orbital parameters were defined with respect to p2, now they are with respect to p1
                for p in particles._particles:
                    if (p.primary is not None) and (particles._particles[p.primary] == id2 or particles._particles[p.primary] == p2.name):
                        p.primary = id1
                particles.remove_particle(p2)
            else: # p2 is more massive
                p2.vel = (p1.m * p2.vel + p2.m * p2.vel)/(p1.m + p2.m)
                p2.m = p1.m + p2.m
                p2.r = np.power(np.power(p1.r, 3.0) + np.power(p2.r, 3.0), 1.0/3)

                for p in particles._particles:
                    if (p.primary is not None) and (particles._particles[p.primary] == id1 or particles._particles[p.primary] == p1.name):
                        p.primary = id2
                particles.remove_particle(p1)
            return 0 # if none of these conditions are satisfied
        else:
            return -1
    

    def calculate_temp(self, particles):
        # calculate flux from each star
        F = 0.0
        
        for i in particles.star_indices:
            Ls = 4 * np.pi * particles.radii[i]**2 * constants.sb * particles.temperatures[i]**4
            for j in particles.planet_indices:
                d = np.linalg.norm(particles._pos[j] - particles._pos[i])
                F += Ls / (4 * np.pi * d**2)
        
        for k in particles.planet_indices:
            particles._particles[k].T = ((1-particles._particles[k].albedo) * F / (4.0 * constants.sb))**(1/4)
            particles.temperatures[k] = particles._particles[k].T
        
        return particles.temperatures
