import numpy as np
from particle import Particle
import numbers
import tools

class Particles(object):
    def __init__(self, G):
        self.g = G
        self._particles = []
        self._pos = np.zeros((0, 3), np.float64)  # (N,3)
        self._vel = np.zeros((0, 3), np.float64)  # (N,3)
        self._mass = np.zeros((0,), np.float64)   # (N,)
        self._rad = np.zeros((0,), np.float64) 
        self._temp = np.zeros((0,), np.float64)   # (N,)
        self._lum = np.zeros((0,), np.float64)
        self._ptype = np.zeros((0,), np.int32)    # (N,)
        self._hash = np.zeros((0,), np.int64)

        self._name_to_index = {}
        self.primary = '#COM#'
    
    #@property
    #def particles(self): return self.__particles
    
    @property
    def N(self): return len(self._particles)
    
    @property
    def positions(self): return self._pos.reshape(-1) # [3N] shape
    
    @property
    def velocities(self): return self._vel.reshape(-1)

    @property
    def names(self): return dict(self._name_to_index)

    @property
    def hashes(self): return self._hashes

    @property
    def radii(self): return self.__rad

    @property
    def ptypes(self): return self._ptype
    
    @property
    def masses(self): return self._mass

    @property
    def temperatures(self): return self._temp

    @property
    def star_indices(self): return np.where(self.ptypes == 0)[0]

    @property
    def planet_indices(self): return np.where(self.ptypes == 1)[0]
        
    
    def __getitem__(self, item):
        # this checks if our particle is still in the list and has an index
        if isinstance(item, numbers.Integral) and not isinstance(item, bool):
            if item < len(self._particles):
                return self._particles[item]
            else:
                raise ValueError('This particle does not exist!' % item)
        # this checks if the user inputted the name of the particle instead of the number
        if isinstance(item, str):
            if item in self._names:
                if item not in self._name_to_index:
                    raise KeyError(f'No particles named {item}')
                return self._particles[self._names[item]]
            else:
                raise ValueError('This particle does not exist!' % item)
        else:
            return None

    # adding a particle to the system
    def add_particle(self, particle):
        if isinstance(particle, Particle) and (particle not in self._particles):
            pos = np.asarray(particle.pos, dtype=np.float64).reshape(3)
            vel = np.asarray(particle.vel, dtype=np.float64).reshape(3)
            m = float(particle.m)
            r = float(getattr(particle, 'r', 0.0))
            pt = int(getattr(particle, 'ptype', 0))
            temp = float(getattr(particle, 'T', 0.0))
            h = int(getattr(particle, 'hash', np.random.randint(100000000, 999999999)))

            if not np.isfinite(pos).all() or not np.isfinite(vel).all():
                raise ValueError("pos/vel must be finite")
            if m <= 0:
                raise ValueError("mass must be > 0")
            if h in set(self._hash.tolist()):
                raise ValueError(f"Duplicate hash {h}")

            idx = self.N
            self._particles.append(particle)
            self._pos = np.vstack([self._pos, pos[None, :]])
            self._vel = np.vstack([self._vel, vel[None, :]])
            self._mass = np.append(self._mass, m)
            self._rad = np.append(self._rad, r)
            self._temp = np.append(self._temp, temp)
            self._ptype = np.append(self._ptype, pt)
            self._hashes = np.append(self._hashes, h)

            name = getattr(particle, "name", None)
            if name is not None:
                if name in self._name_to_index:
                    raise ValueError(f"Duplicate name {name}")
                self._name_to_index[name] = idx

            return h
        else:
            raise TypeError('Incompatible particle type.')
    
    # subtracting a particle from the system
    def remove_particle(self, h):
        h = int(h)
        idx = np.where(self._hash == h)[0]

        if idx.size == 0:
            raise KeyError(f"No particle with hash={h}")
        i = int(idx[0])

        name = getattr(self._particles[i], "name", None)
        if name in self._name_to_index:
            del self._name_to_index[name]


        self._list.pop(i)
        self._pos = np.delete(self._pos, i, axis=0)
        self._vel = np.delete(self._vel, i, axis=0)
        self._mass = np.delete(self._mass, i)
        self._radius = np.delete(self._radius, i)
        self._temp = np.delete(self._temp, i)
        self._ptype = np.delete(self._ptype, i)
        self._hash = np.delete(self._hash, i)

        self._name_to_index = {
            getattr(p, "name"): k
            for k, p in enumerate(self._particles)
            if getattr(p, "name", None) is not None
        }