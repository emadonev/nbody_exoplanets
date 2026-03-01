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
        self.t0 = t0
        self.tf = tf
        self.dt = dt

        # pick central body: first active star (ptype==0) else most massive active
        stars = particles.star_indices
        if stars.size > 0:
            self.central = int(stars[0])
        else:
            active = particles.active_indices
            self.central = int(active[np.argmax(particles.masses[active])])

    def propagate(self):
        # kick for dt/2 using acceleration at t
        self._kick(0.5 * self.dt)
        # drift for dt
        self._drift(self.dt)
        # second kick
        self._kick(0.5*self.dt)

    def _kick(self, h:float):
        p = self.particles
        c = self.central
        idx = self.particles.planet_indices
        r = self.particles.pos[idx] - self.particles.pos[self.central]
        v = self.particles.vel[idx] - self.particles.vel[self.central]

        #??
        # heliocentric positions/velocities
        r = p.pos[idx] - p.pos[c]
        v = p.vel[idx] - p.vel[c]

        # compute mutual accelerations among planets
        a = np.zeros_like(r)
        for ii in range(idx.size):
            ri = r[ii]
            for jj in range(idx.size):
                if jj == ii:
                    continue
                rj = r[jj]
                dr = rj - ri
                d3 = np.linalg.norm(dr) ** 3
                if d3 == 0:
                    continue
                a[ii] += p.g * p.masses[idx[jj]] * dr / d3

        # update heliocentric velocities
        v = v + a * h

        # write back to barycentric velocities (central v held fixed)
        p.vel[idx] = p.vel[c] + v
    
    def _drift(self, h:float):
        # propagate each body assuming Keplerian motion
        for i in range(1, self.particles.N):
            p = self.particles
            c = self.central

            idx = p.planet_indices
            if idx.size == 0:
                return

            r0 = p.pos[idx] - p.pos[c]
            v0 = p.vel[idx] - p.vel[c]

            # per-planet mu
            mu = p.g * (p.masses[c] + p.masses[idx])

            r1 = np.empty_like(r0)
            v1 = np.empty_like(v0)
            for k in range(idx.size):
                r1[k], v1[k] = tools.propagate_kepler_universal(r0[k], v0[k], h, mu[k])

            p.pos[idx] = p.pos[c] + r1
            p.vel[idx] = p.vel[c] + v1
    
    '''
    def compute_acc(self):
        # for each body
        for i in range(1, self.particles.N):
            # first part of formula
            r0i = self.cart[i*3: (i+1)*3] - self.cart[0:3] # this it the zero vector compared to the center

            self.accel[i*3: (i+1)*3] = self.particles.G*self.particles.masses[0]*self.eta[i]/self.eta[i-1]*\
            (self.jacobi[i*3:(i+1)*3]/np.linalg.norm(self.jacobi[i*3:(i+1)*3])**3-\
             r0i/np.linalg.norm(r0i)**3)
            
            # second part of formula
            for j in range(1, i):
                rji = self.cart[j*3:(j+1)*3] - self.cart[i*3:(i+1)*3]
                aux += self.particles.masses[j]*self.particles.G/(np.linalg.norm(rji)**3)*rji

            self.accel[i*3: (i+1)*3] += -(self.eta[i]/self.eta[i-1])*aux

            aux *= 0.0
            # third part of formula
            for j in range(self.particles.N)):
                rij = self.cart[i*3:(i+1)*3] - self.cart[j*3:(j+1)*3]
                aux += self.particles.G*self.particles.masses[j]*rij/(np.linalg.norm(rij)**3)

            self.accel[i*3: (i+1)*3] += aux

            aux *= 0.0
            for j in range(0, i):
                for k in range(i+1, self.particles.N):
                    rjk = self.cart[j*3:(j+1)*3] - self.cart[k*3:(k+1)*3]
                    aux += self.particles.G*self.particles.masses[j]*self.particles.masses[k]*rjk/(np.linalg.norm(rjk)**3)

            self.accel[i*3: (i+1)*3] += -aux/self.eta[i-1]
            '''