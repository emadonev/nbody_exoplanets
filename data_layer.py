'''

This layer of the N-body integrator handles the initialization and addition of particles or bodies, tracking properties of the 
particles system and storage of integrator data.

'''

import numpy as np
import numbers
import sys
import tools

# ====================

class Particle:
    '''
    Particle defines an object classified as a particle.

    ptype=0: by default a regular object; ptype=1 massless; ptype=2 low-mass
    pos=np.zeroes(3): position vector (x, y, z)
    vel=np.zeroes(3): velocity vector (x, y, z)
    mass=0.0: mass of particle 
    radius=0.0: radius of particle
    name=None: optional parameter, for plotting if one decides to name particle
    primary=None:
    '''
    def __init__(self, ptype=0, pos=np.zeroes(3), vel=np.zeroes(3), mass=0.0, radius=0.0, name=None, primary=None):
        self.ptype = ptype
        self.__pos = pos
        self.__vel = vel
        self.m = mass
        self.r = radius
        self.hash = hash(self)
        self.name = name

    @property
    def pos(self):
        return self.__pos
    
    @property
    def vel(self):
        return self.__vel
    
    @pos.setter
    def pos(self, pos_vec):
        if type(pos_vec).__module__ == np.__name__:
            if pos_vec.size == 3:
                self.__pos = pos_vec
            else:
                raise ValueError('Position vector must be a len=3 vector.')
        else:
            raise ValueError('Position must be a numpy vector with len=3.')
        
    @vel.setter
    def vel(self, vel_vec):
        if type(vel_vec).__module__ == np.__name__:
            if vel_vec.size == 3:
                self.__vel = vel_vec
            else:
                raise ValueError('Velocity must be a len=3 vector.')
        else:
            raise ValueError('Velocity must be a numpy vector with len=3.')


class Particles(object):
    def __init__(self, G):
        self.g = G
        self.__all_particles = []
        self.__N = 0
    
    @property
    def all_particles(self):
        return self.__all_particles
    
    @property
    def N(self):
        return self.__N
    
    def __getitem__(self, item):
        # this checks if our particle is still in the list and has an index
        if isinstance(item, numbers.Integral) and not isinstance(item, bool):
            if item < len(self.__all_particles):
                return self.__all_particles[item]
            else:
                raise ValueError('This particle does not exist!' % item)
        # this checks if the user inputted the name of the particle instead of the number
        elif isinstance(item, str):
            if item in self.__names:
                return self.__all_particles[self.__names[item]]
            else:
                raise ValueError('This particle does not exist!' % item)
        else:
            return None

    # adding a particle to the system
    def add_particle(self, particle):
        if isinstance(particle, Particle) and (particle not in self.all_particles):
            self.__all_particles.append(particle)
    
    # subtracting a particle from the system
    def remove_particle(self, particle):
        if isinstance(particle, Particle) and (particle in self.all_particles):
            self.__all_particles.remove(particle)
    
    # this is a shortcut method to be able to add a particle within the Particles class
    # we also want to include the option of adding orbital elements
    # as we will use them for analysis later
    def add(self, pos=None, vel=None, mass=0.0, radius=0.0, name=None, a=None, e=0.0, i=0.0, 
                            Omega=None, omega=None, f=None, primary=None):
        
        # convert from orbital to cartesian
        if a is not None:
            pos, vel = tools.orb_to_cartesian(mass, primary.mass, a, e, i, Omega, omega, f, self.g)
            particle = Particle(ptype=0, pos=pos, vel=vel, mass=mass, radius=radius)
            self.add_particle(particle)
        elif pos is not None and vel is not None:
            particle = Particle(ptype=0, pos=pos, vel=vel, mass=mass, radius=radius)
            self.add_particle(particle)
        else:
            raise ValueError('Either the pos/vel or the orbital elements should be defined!')


