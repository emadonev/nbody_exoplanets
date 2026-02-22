import numpy as np
import sys
import constants
import tools

class WisdomHolman(object):
    def __init__(self):
        return
    
    def bind(self, particles, t0, tf, dt):
        self.particles = particles
        self.t0 = t0
        self.tf = tf
        self.dt = dt
        self.cart = self.setup()
        self.eta = np.zeros(self.particles.N)
        self.eta[0] = self.particles.masses[0]
        for i in range(1,self.particles.N):
            self.eta[i] = self.particles.masses[i] + self.eta[i-1]
        
        # compute jacobi coordinates from initial conditions
        self.jacobi = tools.cart2jacobi(self.cart, 
                                        self.particles.masses, 
                                        self.particles.N, 
                                        self.eta)
        # compute acceleration for first kick
        self.accel = self.compute_acc(self.cart, self.jacobi, self.particles.masses, self.particles.N, constants.G, self.eta)
        
    def setup(self):
        state_vec = np.concatenate(
            (self.particles.positions, self.particles.velocities)
        )
        # state_vec = [pos1, pos2, pos3,...,vel1,vel2,vel3,...]

        helio = self.move_to_helio(state_vec, self.particles.N)

        return helio

    def propagate(self):
        # kick for dt/2 using acceleration at t
        self.kick()

        # drift for dt
        self.drift()

        # convert back to cartesian coordinates for 
        # acceleration computation
        self.cart = tools.jacobi2cart(self.jacobi, 
                                        self.particles.masses, 
                                        self.particles.N, 
                                        self.eta)

        # compute acceleration
        self.compute_acc()

        # second kick
        self.kick()

        # store solution
        self.cart = tools.jacobi2cart(self.jacobi, 
                                        self.particles.masses, 
                                        self.particles.N, 
                                        self.eta)

        self.particles.positions = self.cart[0:self.particles.N*3+1]
        self.particles.velocities = self.cart[self.particles.N*3:]

    def kick(self):
        # apply velocity kick
        self.jacobi[self.particles.N*3:] += self.accel * 0.5*self.dt # simply v = a*t
    
    def drift(self):
        # propagate each body assuming Keplerian motion
        for i in range(1, self.particles.N):
            gm = self.particles.masses[0] * self.eta[i] / self.eta[i-1] * constants.G

            pos0 = self.jacobi[i*3: (i+1)*3]
            vel0 = self.jacobi[(self.particles.N+i)*3: (self.particles.N+i+1)*3]

            pos, vel = self.propagate_kepler(0.0, self.dt, pos0, vel0, gm)

            self.jacobi[i*3: (i+1)(3)] = pos
            self.jacobi[(self.particles.N+1)*3: (self.particles.N+1+i)*3] = vel
    
    def compute_acc(self):
        # for each body
        for i in range(1, self.particles.N):
            # first part of formula
            r0i = self.cart[i*3: (i+1)*3] - self.cart[0:3] # this it the zero vector compared to the center

            self.accel[i*3: (i+1)*3] = constants.G*self.particles.masses[0]*self.eta[i]/self.eta[i-1]*\
            (self.jacobi[i*3:(i+1)*3]/np.linalg.norm(self.jacobi[i*3:(i+1)*3])**3-\
             r0i/np.linalg.norm(r0i)**3)
            
            # second part of formula
            for j in range(1, i):
                rji = self.cart[j*3:(j+1)*3] - self.cart[i*3:(i+1)*3]
                aux += self.particles.masses[j]*constants.G/(np.linalg.norm(rji)**3)*rji

            self.accel[i*3: (i+1)*3] += -(self.eta[i]/self.eta[i-1])*aux

            aux *= 0.0
            # third part of formula
            for j in range(i+1, N):
                rij = self.cart[i*3:(i+1)*3] - self.cart[j*3:(j+1)*3]
                aux += constants.G*self.particles.masses[j]*rij/(np.linalg.norm(rij)**3)

            self.accel[i*3: (i+1)*3] += aux

            aux *= 0.0
            for j in range(0, i):
                for k in range(i+1, self.particles.N):
                    rjk = self.cart[j*3:(j+1)*3] - self.cart[k*3:(k+1)*3]
                    aux += constants.G*self.particles.masses[j]*self.particles.masses[k]*rjk/(np.linalg.norm(rjk)**3)

            self.accel[i*3: (i+1)*3] += -aux/self.eta[i-1]

    def propagate_kepler(t0, tf, vr0, vv0, gm):
        # analytic method for propagating the orbits of planets
        if t0 == tf:
            vrf=vr0
            vvf=vv0
            return 
        
        dt = tf - t0
        tol = 1e-12

        # magnitude of starting vectors
        r0 = np.linalg.norm(vr0)
        v0 = np.linalg.norm(vv0)

        # parameter alpha
        alpha = -(v0**2 - 2.0*gm / r0)

        # radial velocity
        dr0 = np.dot(vr0, vv0)/r0
        
        # solving Kepler's equation
        s = dt/r0
        for j in range(0, 50):
            c0, c1, c2, c3 = tools.stumpff_functions(alpha * s**2)

            # evaluate Keplers equation
            F = r0 * s * c1 + r0 * dr0 * s**2 * c2 + gm * s**3 * c3 - dt
            if abs(F)<tol:
                break

            # compute derivative
            dF = r0 * c0 + r0*dr0*s*c1 + gm * s**2 * c2

            ds = -F/dF

            if abs(ds) < tol:
                break

            # advance step:
            s += ds

        r = dF
        f = 1.0 - gm * s**2 * c2 / r0
        g = dt - gm * s**3 * c3
        df = -gm / (r*r0) * s * c1
        dg = 1.0 - gm / r * s**2 * c2

        vr = f * vr0 + g * vv0
        vv = df * vr0 + dg * vv0

        return vr, vv


    @staticmethod
    def move_to_helio(x, nbodies):
        '''
        This function moves all coordinates with respect to the first body.
        
        :param x: Description
        :param nbodies: Description
        '''
        helio = x.copy()
        for ibod in range(1, nbodies):
            # adjust all position vectors based on the first body
            helio[ibod * 3 : (ibod + 1) * 3] = (
                helio[ibod * 3 : (ibod + 1) * 3] - helio[0:3]
            )

            # adjust all the velocity vectors based on the first body
            helio[(nbodies + ibod) * 3 : (nbodies + ibod + 1) * 3] = (
                helio[(nbodies + ibod) * 3 : (nbodies + ibod + 1) * 3]
                - helio[nbodies * 3 : (nbodies + 1) * 3]
            )
        return helio