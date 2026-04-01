import numpy as np
from . import constants

class Particle(object):
    '''
    Particle defines an object classified as a particle.

    ptype= 0: star, 1: planet
    pos=np.zeroes(3): position vector (x, y, z)
    vel=np.zeroes(3): velocity vector (x, y, z)
    mass=0.0: mass of particle 
    radius=0.0: radius of particle
    temperature=0.0: temperature of particle
    name=None: optional parameter, for plotting if one decides to name particle
    index=None: the index of the particle
    primary=None: the primary body of reference
    hash=None: hash of the particle (unique number)
    a=None: semi-major axis of particle
    e=None: eccentricity of particle
    inc=None: inclination of particle relative to our plane
    Omega=None: longitude of the ascending node
    omega=None: argument of the pericentre
    theta=None: the current location of the particle on its orbit
    angle_in_degrees=False: are the previous orbital elements in degrees or radians

    Methods:
    pos() - get the position
    vel() - get the velocity
    '''
    def __init__(self, 
                 ptype: int =0, 
                 pos=None, 
                 vel=None, 
                 mass: float =0.0, 
                 radius: float =0.0, 
                 temperature: float =0.0,
                 name=None,
                 index=None,
                 primary=None,
                 hash=None,
                 a=None,
                 e=None,
                 inc=None,
                 Omega=None,
                 omega=None,
                 theta=None,
                 angles_in_degrees: bool = False):
        
        # setting the particle type
        self.ptype = int(ptype)

        # checking if the position and velocity are set
        if (pos is None) ^ (vel is None):
            raise ValueError("pos and vel must be both provided or both omitted")
        self._cartesian_input = (pos is not None and vel is not None)
        self._pos = np.zeros(3, dtype=float) if pos is None else np.asarray(pos, dtype=float).reshape(3)
        self._vel = np.zeros(3, dtype=float) if vel is None else np.asarray(vel, dtype=float).reshape(3)

        # setting up all the other auxiliary values
        self.m = float(mass)
        self.r = float(radius)
        self.T = float(temperature)
        self.hash = int(hash) if hash is not None else int(np.random.randint(100000000, 999999999))
        self.name = name
        self.index = None if index is None else str(index)
        self.primary = primary

        # setting the orbital parameters
        self.a = None if a is None else float(a)
        self.e = None if e is None else float(e)
        self.inc = None if inc is None else float(inc)
        self.Omega = None if Omega is None else float(Omega)
        self.omega = None if omega is None else float(omega)
        self.theta = None if theta is None else float(theta)
        self.angles_in_degrees = bool(angles_in_degrees)

    def __repr__(self) -> str:
        # print method
        return (
            f"Particle(name={self.name!r}, ptype={self.ptype}, m={self.m}, r={self.r}, "
            f"pos={self._pos.tolist()}, vel={self._vel.tolist()}, hash={self.hash})"
        )
    
    # getter and setter methods for position and velocity
    @property
    def pos(self)-> np.ndarray:
        return self._pos
    
    @property
    def vel(self)-> np.ndarray:
        return self._vel
        
    @pos.setter
    def pos(self, pos_vec)-> None:
        # setting the position + checking the shape
        pos_vec = np.asarray(pos_vec, dtype=float)
        if pos_vec.shape != (3,):
            raise ValueError("pos must be shape (3,)")
        self._pos = pos_vec
        
    @vel.setter
    def vel(self, vel_vec):
        # setting the velocity + checking the shape
        vel_vec = np.asarray(vel_vec, dtype=float)
        if vel_vec.shape != (3,):
            raise ValueError("vel must be shape (3,)")
        self._vel = vel_vec