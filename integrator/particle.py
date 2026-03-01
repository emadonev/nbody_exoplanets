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
    def __init__(self, 
                 ptype: int =0, 
                 pos=None, 
                 vel=None, 
                 mass: float =0.0, 
                 radius: float =0.0, 
                 temperature: float =0.0,
                 albedo=None,
                 name=None,
                 primary=None,
                 hash=None):
        
        self.ptype = int(ptype)
        self._pos = np.zeros(3, dtype=float) if pos is None else np.asarray(pos, dtype=float).reshape(3)
        self._vel = np.zeros(3, dtype=float) if vel is None else np.asarray(vel, dtype=float).reshape(3)

        self.m = float(mass)
        self.r = float(radius)
        self.T = float(temperature)
        self.albedo = albedo
        self.hash = int(hash) if hash is not None else int(np.random.randint(100000000, 999999999))
        self.name = name

    def __repr__(self) -> str:
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
        pos_vec = np.asarray(pos_vec, dtype=float)
        if pos_vec.shape != (3,):
            raise ValueError("pos must be shape (3,)")
        self._pos = pos_vec
        
    @vel.setter
    def vel(self, vel_vec):
        vel_vec = np.asarray(vel_vec, dtype=float)
        if vel_vec.shape != (3,):
            raise ValueError("vel must be shape (3,)")
        self._vel = vel_vec

    # just in case - convenience values if we need a specific coordinate direciton for pos and vel
    @property
    def x(self) -> float:
        return float(self._pos[0])

    @property
    def y(self) -> float:
        return float(self._pos[1])

    @property
    def z(self) -> float:
        return float(self._pos[2])

    @property
    def vx(self) -> float:
        return float(self._vel[0])

    @property
    def vy(self) -> float:
        return float(self._vel[1])

    @property
    def vz(self) -> float:
        return float(self._vel[2])
