import numpy as np
import constants

class Physics(object):
    # ---------
    # energy
    # ---------
    def energy(self, particles)-> float:
        # not all particles have to be active so
        active = particles.active_indices
        m = particles.masses
        v = particles.vel

        ke = 0.5 * np.sum(m[active] * np.sum(v[active] ** 2, axis=1))

        pe = 0.0
        for ii, i in enumerate(active):
            for j in active[ii + 1 :]:
                d = np.linalg.norm(particles.pos[i] - particles.pos[j])
                if d == 0:
                    continue
                pe -= particles.g * particles.masses[i] * particles.masses[j] / d
        return float(ke + pe)
    
    # -----------
    # resolving primary object
    # -----------
    def center_of_mass(self, particles, subset=None):
        # define the center of mass with position and velocity
        if subset is None:
            idx = particles.active_indices # use all active particles
        else:
            idx = np.asarray(subset, dtype=int)
            idx = idx[particles.masses[idx] > 0.0] # take the ids of the subset of particles
        
        m = particles.masses[idx]
        pos = particles.pos[idx]
        vel = particles.vel[idx]

        mtot = float(m.sum())

        com_pos = (m[:, None] * pos).sum(axis=0) / mtot
        com_vel = (m[:, None] * vel).sum(axis=0) / mtot

        return mtot, com_pos, com_vel
    
    # ----------
    # collisions
    # ----------

    def check_collision(self, particles, i: int, j: int) -> bool:
        if i == j: # if we check the same object, no collision automatically
            return False
        # distance between both objects
        d = np.linalg.norm(particles.pos[i] - particles.pos[j])
        # check if the distance is less than the sum of their radii - they definitely collided
        return d <= (particles.radii[i] + particles.radii[j])
    
    def collision(self, particles, i:int, j:int)->bool:
        #check if a collision occured - especially if i == j
        if not self.check_collision(particles, i, j):
            return False
        
        # more massive particle wins the collision
        if particles.masses[i] >= particles.masses[j]:
            keep, drop = i, j
        else:
            keep, drop = j, i

        # define the properties of the 2 particles
        m1, m2 = float(particles.masses[keep]), float(particles.masses[drop])
        r1, r2 = float(particles.radii[keep]), float(particles.radii[drop])
        p1 = particles.pos[keep].copy()
        p2 = particles.pos[drop].copy()
        v1 = particles.vel[keep].copy()
        v2 = particles.vel[drop].copy()

        m_new = m1 + m2 # mass of the new body is the sum of both masses
        v_new = (m1*v1 + m2*v2) / m_new # conservation of linear momentum during collision
        p_new = (m1*p1 + m2*p2) / m_new # new position of body is "center of mass" calculation of both bodies
        r_new = (r1**3 + r2**3)**(1.0/3.0) # new radius

        # update the arrays with this new particle
        particles.pos[keep] = p_new
        particles.vel[keep] = v_new
        particles.masses[keep] = m_new
        particles.radii[keep] = r_new

        # deactivate the smaller particle to keep the arrays the same size
        particles.deactivate_particle(drop)
        return True # a collision happened
    
    # ------------
    # temperature
    # ------------

    def calculate_temp(self, particles):
        star_idx = particles.star_indices # set the star indices
        planet_idx = particles.planet_indices # set the planet indices

        if star_idx.size == 0 or planet_idx.size == 0:
            return particles.temperatures # no particles or stars in the system
        
        # calculate luminosities from all stars
        L = 4.0 * np.pi * particles.radii[star_idx] ** 2 * constants.sb * particles.temperatures[star_idx]**4 # this returns a list of luminosities


        # calculate flux from each star
        
        for k in planet_idx: # for each planet
            Fk = 0.0 # the starting flux is 0
            for s_i, Ls in zip(star_idx, L): # for every index and luminosity in the list of indices and luminosities
                d = np.linalg.norm(particles.pos[k] - particles.pos[s_i]) # calculate distance
                if d == 0:
                    continue
                Fk += Ls / (4.0 * np.pi * d**2) # calculate flux

            A = particles.albedos[k] # select albedo
            if A is None:
                A = 0.0 # if albedo is undefined select that it absorbs everything
            T_eq = ((1.0 - float(A))* Fk / (4.0 * constants.sb)) ** 0.25 # calculate temperature
            particles.temperatures[k] = T_eq # assign temperature to the list

        return particles.temperatures