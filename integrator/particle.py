import numpy as np
import constants

class Particle(object):
    '''
    Particle defines an object classified as a particle.

    ptype= 0: star, 1: planet
    pos=np.zeroes(3): position vector (x, y, z)
    vel=np.zeroes(3): velocity vector (x, y, z)
    mass=0.0: mass of particle 
    radius=0.0: radius of particle
    name=None: optional parameter, for plotting if one decides to name particle
    primary=None:
    '''
    def __init__(self, ptype=0, 
                 pos=None, 
                 vel=None, 
                 mass=0.0, 
                 radius=0.0, 
                 temperature=0.0,
                 albedo=None,
                 name=None):
        self.ptype = int(ptype)
        self._pos = np.zeros(3) if pos is None else np.asarray(pos, dtype=float)
        self._vel = np.zeros(3) if vel is None else np.asarray(vel, dtype=float)
        self.x = pos[0]  # x
        self.y = pos[1]  # y
        self.z = pos[2]  # z
        self.vx = vel[0]  # vx
        self.vy = vel[1]  # vy
        self.vz = vel[2]  # vz
        self.m = float(mass)
        self.r = float(radius)
        self.T = float(temperature)
        self.albedo = albedo
        self.hash = np.random.randint(100000000, 999999999)
        self.name = name

    def __repr__(self):
        return f"Particle: m={self.m}, pos={self._pos}, vel={self._vel}, r={self.r}, name={self.name}, hash={self.hash}"

    @property
    def pos(self):
        return self._pos
    
    @property
    def vel(self):
        return self._vel
        
    def temperature(self, particle):
        return self.T
        
    @pos.setter
    def pos(self, pos_vec):
        if type(pos_vec).__module__ == np.__name__:
            if pos_vec.size == 3:
                self.x = pos_vec[0]
                self.y = pos_vec[1]
                self.z = pos_vec[2]
                self._pos = pos_vec
            else:
                raise ValueError('Position vector must be a len=3 vector.')
        else:
            raise ValueError('Position must be a numpy vector with len=3.')
        
    @vel.setter
    def vel(self, vel_vec):
        if type(vel_vec).__module__ == np.__name__:
            if vel_vec.size == 3:
                self.vx = vel_vec[0]
                self.vy = vel_vec[1]
                self.vz = vel_vec[2]
                self._vel = vel_vec
            else:
                raise ValueError('Velocity must be a len=3 vector.')
        else:
            raise ValueError('Velocity must be a numpy vector with len=3.')