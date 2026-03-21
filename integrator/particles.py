import numpy as np
import numbers
from . import tools
from .particle import Particle

class Particles(object):
    '''
    Particles - a container of all the particles within the N body system we wish to integrate.
    Contains a list of all the particles, as well as arrays of position, velocity, etc. of all the properties
    in the shape (N, 3). Manages the updated particle positions, velocities, etc. as well as the structural hierarchy of the system
    in the form of a binary tree.

    Methods:
    - _get_body_index() - get the index of a particular particle within the _particles list
    - _child_body_indices() - return all the leaf indices 
    - _child_submass() - return the submass of the children nodes of the parent node
    - _build_transform_row() - returns a row of the transformation matrix necessary for conversion between Cart and HJS coords
    - postorder() - recursive method for getting a structured orbit_list to iterate over in the integrator; compiler method
    - tree_build - builds a binary tree of individual 2-body orbits within the more complex hierarchical system and assigns values to the
    parent nodes - submass, transform rows, etc.; compiler method
    - resolve_primary_index() - method to resolve which body is the primary within a 2 body system or a multi body system with no hierarchy
    - resolve_primary_state() - method to get the primary state information of the center of mass
    - add_particle() - add a Particle object to the container
    - remove_particle() - remove a Particle object from the container
    - deactivate_particle() - in case of a collision, deactivate the body but don't remove it from the system
    - _sync_objects() - after every integration step, update the particle's positions, velocities, etc. 
    '''
    def __init__(self, G, system_type=None):
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

        self._a = np.zeros((0,), dtype=np.float64)
        self._e = np.zeros((0,), dtype=np.float64)
        self._i = np.zeros((0,), dtype=np.float64)
        self._Omega = np.zeros((0,), dtype=np.float64)
        self._omega = np.zeros((0,), dtype=np.float64)
        self._theta = np.zeros((0,), dtype=np.float64)

        self._name_to_index: dict[str, int] = {}
        self._index_to_array: dict[str, int] = {}  # maps particle.index label (A/B/C/P…) -> array position
        self.system = system_type  # can be 5 types: SA, SB, SC, SAB-P, P(ABC)
        self.primary = "#COM#"     # default reference frame: total centre of mass

    def _get_body_index(self, particle):
        return self._particles.index(particle)

    def get_by_label(self, label: str) -> tuple:
        """Return (array_index, Particle) for a particle with the given index label (e.g. 'A', 'B', 'C')."""
        key = str(label)
        if key not in self._index_to_array:
            raise KeyError(f"No particle with index label '{label}'")
        i = self._index_to_array[key]
        return i, self._particles[i]
        
    def resolve_primary_index(self, primary_spec=None):
        active = self.active_indices
        if active.size == 0:
            raise ValueError("cannot resolve primary without existing active bodies")

        if primary_spec is None:
            primary_spec = self.primary

        if isinstance(primary_spec, str):
            if primary_spec == "#COM#":
                stars = self.star_indices
                if stars.size > 0:
                    return int(stars[0])
                return int(active[np.argmax(self._mass[active])])
            if primary_spec == "#MAX#":
                return int(active[np.argmax(self._mass[active])])
            if primary_spec not in self._name_to_index:
                raise KeyError(f"No particles named {primary_spec}")
            return int(self._name_to_index[primary_spec])

        if isinstance(primary_spec, numbers.Integral):
            i = int(primary_spec)
            if not (0 <= i < self.N):
                raise IndexError(i)
            if self._mass[i] <= 0.0:
                raise ValueError(f"primary index {i} is inactive")
            return i

        raise TypeError(f"Unsupported primary spec type: {type(primary_spec)}")

    def resolve_primary_state(self, primary_spec=None):
        if primary_spec is None:
            primary_spec = self.primary

        if isinstance(primary_spec, str) and primary_spec == "#COM#":
            idx = self.active_indices
            m = self._mass[idx]
            pos = self._pos[idx]
            vel = self._vel[idx]
            mtot = float(m.sum())
            com_pos = (m[:, None] * pos).sum(axis=0) / mtot
            com_vel = (m[:, None] * vel).sum(axis=0) / mtot
            return mtot, com_pos, com_vel

        if isinstance(primary_spec, (list, tuple, np.ndarray)):
            idx = np.asarray(primary_spec, dtype=int)
            idx = idx[(0 <= idx) & (idx < self.N)]
            idx = idx[self._mass[idx] > 0.0]
            if idx.size == 0:
                raise ValueError("primary subset has no active bodies")
            m = self._mass[idx]
            pos = self._pos[idx]
            vel = self._vel[idx]
            mtot = float(m.sum())
            com_pos = (m[:, None] * pos).sum(axis=0) / mtot
            com_vel = (m[:, None] * vel).sum(axis=0) / mtot
            return mtot, com_pos, com_vel

        i = self.resolve_primary_index(primary_spec)
        return float(self._mass[i]), self._pos[i].copy(), self._vel[i].copy()
    
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

        # initialize scalar properties
        m = float(particle.m)
        r = float(particle.r)
        pt = int(getattr(particle, 'ptype', 0))
        temp = float(getattr(particle, 'T', 0.0))
        h = int(getattr(particle, 'hash', np.random.randint(100000000, 999999999)))
        alb = getattr(particle, 'albedo', None)

        if m <= 0:
            raise ValueError("mass must be > 0")
        if h in set(self._hash.tolist()):
            raise ValueError(f"Duplicate hash {h}")

        # Optional orbital elements.
        a = getattr(particle, "a", None)
        e = getattr(particle, "e", None)
        inc = getattr(particle, "inc", None)
        Omega = getattr(particle, "Omega", None)
        omega = getattr(particle, "omega", None)
        theta = getattr(particle, "theta", None)
        angles_in_degrees = bool(getattr(particle, "angles_in_degrees", False))
        orbital_values = [a, e, inc, Omega, omega, theta]
        has_orbital = all(v is not None for v in orbital_values)
        if any(v is not None for v in orbital_values) and not has_orbital:
            raise ValueError("a/e/inc/Omega/omega/theta must all be set for orbital conversion")

        # Determine Cartesian state:
        # 1) explicit pos/vel if provided by user
        # 2) else derive from orbital elements relative to resolved primary
        # 3) else default zeros (useful for first central star)
        if getattr(particle, "_cartesian_input", False):
            pos = np.asarray(particle.pos, dtype=np.float64).reshape(3)
            vel = np.asarray(particle.vel, dtype=np.float64).reshape(3)
        elif has_orbital:
            if self.N == 0:
                if pt == 0:
                    # First central body: no reference primary exists yet.
                    # Keep the default origin state for the initial star.
                    pos = np.zeros(3, dtype=np.float64)
                    vel = np.zeros(3, dtype=np.float64)
                else:
                    raise ValueError("cannot convert orbital elements without an existing primary body")
            else:
                primary_spec = getattr(particle, "primary", None)
                if isinstance(primary_spec, (list, tuple, np.ndarray)):
                    # Primary is a subset of bodies — orbit their centre of mass.
                    idx_list = []
                    for p in primary_spec:
                        if isinstance(p, str):
                            if p not in self._name_to_index:
                                raise KeyError(f"No particle named '{p}'")
                            idx_list.append(self._name_to_index[p])
                        else:
                            idx_list.append(int(p))
                    mp, primary_pos, primary_vel = self.resolve_primary_state(
                        np.array(idx_list, dtype=int)
                    )
                    mu = self.g * (mp + m)
                else:
                    primary_idx = self.resolve_primary_index(primary_spec)
                    mp = float(self._mass[primary_idx])
                    mu = self.g * (mp + m)
                    primary_pos = self._pos[primary_idx].copy()
                    primary_vel = self._vel[primary_idx].copy()
                r_rel, v_rel = tools.orb_to_cartesian(
                    a=a,
                    e=e,
                    i=inc,
                    Omega=Omega,
                    omega=omega,
                    theta=theta,
                    mu=mu,
                    angles_in_degrees=angles_in_degrees,
                )
                pos = primary_pos + r_rel
                vel = primary_vel + v_rel
        else:
            if pt != 0 and self.N > 0:
                raise ValueError("non-star particles require either pos/vel or full orbital elements")
            pos = np.asarray(particle.pos, dtype=np.float64).reshape(3)
            vel = np.asarray(particle.vel, dtype=np.float64).reshape(3)

        if not np.isfinite(pos).all() or not np.isfinite(vel).all():
            raise ValueError("pos/vel must be finite")

        if has_orbital and not np.isfinite(np.array([a, e, inc, Omega, omega, theta], dtype=float)).all():
            raise ValueError("orbital elements must be finite")

        # Store orbital elements in radians internally when available.
        if has_orbital and angles_in_degrees:
            inc_store = np.deg2rad(float(inc))
            Omega_store = np.deg2rad(float(Omega))
            omega_store = np.deg2rad(float(omega))
            theta_store = np.deg2rad(float(theta))
        elif has_orbital:
            inc_store = float(inc)
            Omega_store = float(Omega)
            omega_store = float(omega)
            theta_store = float(theta)
        else:
            inc_store = np.nan
            Omega_store = np.nan
            omega_store = np.nan
            theta_store = np.nan

        idx = self.N
        name = getattr(particle, "name", None)
        if name is not None and name in self._name_to_index:
            raise ValueError(f"Duplicate name {name}")

        # update the container properties by adding each component to its list
        self._pos = np.vstack([self._pos, pos[None, :]])
        self._vel = np.vstack([self._vel, vel[None, :]])
        self._mass = np.append(self._mass, m)
        self._rad = np.append(self._rad, r)
        self._temp = np.append(self._temp, temp)
        self._ptype = np.append(self._ptype, pt)
        self._hash = np.append(self._hash, h)
        self._albedo = np.append(self._albedo, alb)
        self._a = np.append(self._a, float(a) if has_orbital else np.nan)
        self._e = np.append(self._e, float(e) if has_orbital else np.nan)
        self._i = np.append(self._i, inc_store)
        self._Omega = np.append(self._Omega, Omega_store)
        self._omega = np.append(self._omega, omega_store)
        self._theta = np.append(self._theta, theta_store)
        
        # Keep object state consistent with container arrays at insertion time.
        particle._pos = pos.copy()
        particle._vel = vel.copy()
        particle.m = m
        particle.r = r
        particle.T = temp
        particle.ptype = pt
        particle.hash = h
        particle.albedo = alb
        particle.a = float(a) if has_orbital else None
        particle.e = float(e) if has_orbital else None
        particle.inc = inc_store if np.isfinite(inc_store) else None
        particle.Omega = Omega_store if np.isfinite(Omega_store) else None
        particle.omega = omega_store if np.isfinite(omega_store) else None
        particle.theta = theta_store if np.isfinite(theta_store) else None

        # adding the object itself
        self._particles.append(particle)
        if name is not None:
            self._name_to_index[name] = idx
        label = getattr(particle, "index", None)
        if label is not None:
            self._index_to_array[str(label)] = idx

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
        self._a = np.delete(self._a, idx)
        self._e = np.delete(self._e, idx)
        self._i = np.delete(self._i, idx)
        self._Omega = np.delete(self._Omega, idx)
        self._omega = np.delete(self._omega, idx)
        self._theta = np.delete(self._theta, idx)

        # rebuild mappings according to the new order of particles
        self._name_to_index = {
            p.name: k
            for k, p in enumerate(self._particles)
            if getattr(p, "name", None) is not None
        }
        self._index_to_array = {
            str(p.index): k
            for k, p in enumerate(self._particles)
            if getattr(p, "index", None) is not None
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
        self._a[idx] = np.nan
        self._e[idx] = np.nan
        self._i[idx] = np.nan
        self._Omega[idx] = np.nan
        self._omega[idx] = np.nan
        self._theta[idx] = np.nan

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
            p.a = float(self._a[i]) if np.isfinite(self._a[i]) else None
            p.e = float(self._e[i]) if np.isfinite(self._e[i]) else None
            p.inc = float(self._i[i]) if np.isfinite(self._i[i]) else None
            p.Omega = float(self._Omega[i]) if np.isfinite(self._Omega[i]) else None
            p.omega = float(self._omega[i]) if np.isfinite(self._omega[i]) else None
            p.theta = float(self._theta[i]) if np.isfinite(self._theta[i]) else None
