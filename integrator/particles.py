import numpy as np
from particle import Particle
import numbers
import constants

class Particles(object):
    def __init__(self, G):
        self.g = float(G)

        self._particles: list[Particle] = []
        self._pos = np.zeros((0, 3), dtype=np.float64)
        self._vel = np.zeros((0, 3), dtype=np.float64)
        self._mass = np.zeros((0,), dtype=np.float64)
        self._rad = np.zeros((0,), dtype=np.float64)
        self._temp = np.zeros((0,), dtype=np.float64)
        self._ptype = np.zeros((0,), dtype=np.int32)
        self._hash = np.zeros((0,), dtype=np.int64)
        self._albedo = np.empty((0,), dtype=object)

        self._name_to_index: dict[str, int] = {}
        self.primary = "#COM#"
    
    # properties of the particles container

    # number of particles
    @property
    def N(self) -> int: return int(self._pos.shape[0])

    # returning regular pos vector
    @property
    def pos(self) -> np.ndarray:
        return self._pos
    
    # returning regular velocity vector
    @property
    def vel(self) -> np.ndarray:
        return self._vel
    
    # returning masses of particles
    @property
    def masses(self) -> np.ndarray: 
        return self._mass
    
    # returning all the radii of the particles
    @property
    def radii(self) -> np.ndarray:
        return self._rad

    # returning all the temperatures of the particles
    @property
    def temperatures(self) -> np.ndarray:
        return self._temp

    # returning the particle types (star, planet)
    @property
    def ptypes(self) -> np.ndarray:
        return self._ptype

    # returning the unique identifiers
    @property
    def hashes(self) -> np.ndarray:
        return self._hash

    # returning the albedos of particles
    @property
    def albedos(self) -> np.ndarray:
        return self._albedo

    # returning the names of the particles
    @property
    def names(self) -> dict[str, int]:
        return dict(self._name_to_index)

    @property
    def positions(self) -> np.ndarray:
        #Flattened positions (3N,).
        return self._pos.reshape(-1)
    
    @positions.setter
    def positions(self, x) -> None:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            if x.size != self.N * 3:
                raise ValueError(f"positions flat vector must have size {self.N*3}")
            self._pos[:] = x.reshape(self.N, 3)
        elif x.shape == (self.N, 3):
            self._pos[:] = x
        else:
            raise ValueError("positions must be shape (3N,) or (N,3)")
    
    @property
    def velocities(self) -> np.ndarray:
        #Flattened velocities (3N,).
        return self._vel.reshape(-1)

    @velocities.setter
    def velocities(self, v) -> None:
        v = np.asarray(v, dtype=np.float64)
        if v.ndim == 1:
            if v.size != self.N * 3:
                raise ValueError(f"velocities flat vector must have size {self.N*3}")
            self._vel[:] = v.reshape(self.N, 3)
        elif v.shape == (self.N, 3):
            self._vel[:] = v
        else:
            raise ValueError("velocities must be shape (3N,) or (N,3)")
    
    # additional properties

    @property
    def star_indices(self): 
        return np.where((self._ptype == 0) & (self._mass > 0.0))[0]

    @property
    def planet_indices(self): 
        return np.where((self._ptype == 1) & (self._mass > 0.0))[0]
    
    @property
    def active_indices(self) -> np.ndarray:
        return np.where(self._mass > 0.0)[0]
    
    def __repr__(self) -> str:
        return (
            f"Particles({self._particles}, {self.pos}, {self.vel})"
        )
    
    # getting an item + indexing
    def __getitem__(self, item):
        # this checks if our particle is still in the list and has an index
        if isinstance(item, numbers.Integral) and not isinstance(item, bool):
            i = int(item)
            if not (0 <= i < self.N): # if this number is not between 1 and N
                raise IndexError(i)
            return self._particles[i] # index by position eg. particle 2 (starting from particle 0)
        
        # this checks if the user inputted the name of the particle instead of the number
        if isinstance(item, str):
            if item not in self._name_to_index:
                raise KeyError(f'No particles named {item}')
            return self._particles[self._name_to_index[item]]
        
        # if none of these conditions are satisfied, user inputted incorrectly
        raise TypeError("index must be int or str")

    # adding a particle to the system
    def add_particle(self, particle):
        if not isinstance(particle, Particle):
            raise TypeError("add_particle expects a Particle")
        
        # initialize the properties of the added particle
        pos = np.asarray(particle.pos, dtype=np.float64).reshape(3)
        vel = np.asarray(particle.vel, dtype=np.float64).reshape(3)
        m = float(particle.m)
        r = float(particle.r)
        pt = int(getattr(particle, 'ptype', 0))
        temp = float(getattr(particle, 'T', 0.0))
        h = int(getattr(particle, 'hash', np.random.randint(100000000, 999999999)))
        alb = getattr(particle, 'albedo', None)

        # check that all values are accurate
        if not np.isfinite(pos).all() or not np.isfinite(vel).all():
            raise ValueError("pos/vel must be finite")
        
        if m <= 0:
            raise ValueError("mass must be > 0")
        
        if h in set(self._hash.tolist()):
            raise ValueError(f"Duplicate hash {h}")

        idx = self.N

        # update the container properties by adding each component to its list
        self._pos = np.vstack([self._pos, pos[None, :]])
        self._vel = np.vstack([self._vel, vel[None, :]])
        self._mass = np.append(self._mass, m)
        self._rad = np.append(self._rad, r)
        self._temp = np.append(self._temp, temp)
        self._ptype = np.append(self._ptype, pt)
        self._hash = np.append(self._hash, h)
        self._albedo = np.append(self._albedo, alb)
        
        # adding the object itself
        self._particles.append(particle)
        
        name = getattr(particle, "name", None)
        if name is not None:
            if name in self._name_to_index:
                raise ValueError(f"Duplicate name {name}")
            self._name_to_index[name] = idx

        return h
    
    # subtracting a particle from the system
    def remove_particle(self, h_or_idx) -> None:
        # check if the user inputted hash or index
        if isinstance(h_or_idx, numbers.Integral) and not isinstance(h_or_idx, bool):
            i = int(h_or_idx)
            # check if it is the index
            if 0 <= i <= self.N:
                idx = i
            else: # if not, hash inputted
                h = int(h_or_idx)
                w = np.where(self._hash == h)[0] # find the index of this hash and use to find the particle
                if w.size == 0:
                    raise KeyError(f"No particle with hash={h}")
                idx = int(w[0])
        else:
            raise TypeError('remove_particle expects int (index or hash)')
        
        # remove the name from the system
        name = getattr(self._particles[idx], "name", None)
        if name is not None and name in self._name_to_index:
            del self._name_to_index[name]

        # remove properties of this particle from all list and the object itself
        self._particles.pop(idx)
        self._pos = np.delete(self._pos, idx, axis=0)
        self._vel = np.delete(self._vel, idx, axis=0)
        self._mass = np.delete(self._mass, idx)
        self._rad = np.delete(self._rad, idx)
        self._temp = np.delete(self._temp, idx)
        self._ptype = np.delete(self._ptype, idx)
        self._hash = np.delete(self._hash, idx)
        self._albedo = np.delete(self._albedo, idx)

        # rebuild mapping according to the new order of particles
        self._name_to_index = {
            p.name: k
            for k, p in enumerate(self._particles)
            if getattr(p, "name", None) is not None
        }

    def deactivate_particle(self, idx:int)-> None:
        # this marks a particle as inactive without changing array sizes.

        idx = int(idx)
        if not(0<=idx<self.N):
            raise IndexError(idx)
        
        # remove this particle from mapping
        name = getattr(self._particles[idx], 'name', None)
        if name is not None and name in self._name_to_index:
            del self._name_to_index[name]

        # set the properties to null
        self._mass[idx] = 0.0
        self._rad[idx] = 0.0
        self._temp[idx] = np.nan
        self._ptype[idx] = -1
        self._pos[idx] = np.nan
        self._vel[idx] = np.nan

    def _sync_objects(self) -> None:
        # copy array state into the stored Particle objects to keep everything in sync!
        # once we update anything with an integrator we need to update each particle individually as well
        # not just the global container properties
        for i, p in enumerate(self._particles):
            p._pos = self._pos[i].copy()
            p._vel = self._vel[i].copy()
            p.m = float(self._mass[i])
            p.r = float(self._rad[i])
            p.T = float(self._temp[i])
            p.ptype = int(self._ptype[i])
            p.hash = int(self._hash[i])
            p.albedo = self._albedo[i]