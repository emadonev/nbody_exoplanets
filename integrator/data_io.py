import numpy as np
import h5py
import os
from . import constants

class DataIO(object):
    '''

    DataIO is an object used to store the data of a desired simulation run in the form of a HDF5 file. It stores all relevant metadata,
    including but not limited to: position, velocity, mass, energy, radius, temperature, orbital parameters,...

    It uses a buffer to fill up storage and then flushes all the data to a data.hdf5 file.

    '''
    def __init__(
            self,
            buf_len: int =1024,
            output_file_name: str ='data.hdf5',
            const_g: float =constants.G
        ):

        # SETTING UP THE BUFFER NAMES
        # =============

        self.buf_len = int(buf_len)
        self.output_name = output_file_name
        self.const_g = float(const_g)

        self.buf_initialized=False
        self.n_particles = None

        self.buf_t = None # buffer for time vector
        self.buf_pos = None # position
        self.buf_vel = None # velocity
        self.buf_mass = None # mass
        self.buf_radius = None # radius
        self.buf_ptype = None # type of body: star or planet
        self.buf_hashes = None # specific number for each object

        self.buf_cursor = 0
        self.h5_file = None
        self.h5_step_id = 0

    def initialize_buffer(self, n_particles: int):
        '''
        Initialize the buffer - allocate space for all variables and ensure correct number of particles.
        '''
        if self.buf_initialized:
            return # if the buffer is already initialized, no need for initialization

        # setup the saving with the number of particles for proper shape
        n_particles = int(n_particles)
        if n_particles <= 0:
            raise ValueError('n_particles must be > 0')
        self.n_particles = n_particles

        buf_len = self.buf_len # length of buffer

        # core state — always saved
        self.buf_t = np.full(buf_len, np.nan, dtype=np.float64)
        self.buf_pos = np.full((buf_len, n_particles * 3), np.nan, dtype=np.float64)
        self.buf_vel = np.full((buf_len, n_particles * 3), np.nan, dtype=np.float64)
        self.buf_mass = np.full((buf_len, n_particles), np.nan, dtype=np.float64)
        self.buf_radius = np.full((buf_len, n_particles), np.nan, dtype=np.float64)
        self.buf_ptype = np.full((buf_len, n_particles), -1, dtype=np.int32)
        self.buf_hashes = np.full((buf_len, n_particles), -1, dtype=np.int64)

        self.buf_cursor = 0
        self.h5_step_id = 0

        # if we have leftover files with the same name, remove them beforehand
        if self.output_name and os.path.isfile(self.output_name):
            os.remove(self.output_name)
        self.buf_initialized = True # we have now initialized a buffer


    def flush(self):
        '''
        flush - push all the data once buffer is filled to the hdf5 file
        '''
        if self.buf_cursor == 0:
            # already flushed
            return

        if self.h5_file is None:
            out_dir = os.path.dirname(self.output_name)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            self.h5_file = h5py.File(self.output_name, "w") # if we haven't started writing a file, create one
            self.h5_file.attrs['G'] = self.const_g # assign the constant G
            self.h5_file.attrs['postprocessed'] = False

        h5_step_group = self.h5_file.create_group(f"Step_{self.h5_step_id}") # create a subgroup until a specified step
        sl = slice(0, self.buf_cursor)

        # core state — always saved
        state_dict = {
            "time": self.buf_t[sl],
            "pos": self.buf_pos[sl],
            "vel": self.buf_vel[sl],
            "mass": self.buf_mass[sl],
            "radius": self.buf_radius[sl],
            "ptype": self.buf_ptype[sl],
            "hash": self.buf_hashes[sl],
        }

        # save data to the hdf5 file (gzip level 4)
        for k, v in state_dict.items():
            h5_step_group.create_dataset(k, data=v, compression='gzip', compression_opts=4)

        # reset cursor
        self.h5_file.flush()
        self.h5_step_id += 1
        self.buf_cursor = 0

    def close(self):
        '''
        close - close the hdf5 file once everything is pushed and the simulation is finished.
        '''
        if self.buf_cursor > 0:
            self.flush() # if there happens to be more data to push

        if self.h5_file is not None:
            self.h5_file.close()
            self.h5_file = None

    def store_state(
            self,
            t: float,
            pos: np.ndarray,
            vel: np.ndarray,
            masses: np.array,
            radii: np.ndarray,
            hashes: np.ndarray,
            ptypes: np.ndarray
    ):
        '''
        store_state - store the desired state of every parameter within simulation, and then later
        when the buffer fills up it will go into the hdf5 file.
        '''
        if not self.buf_initialized:
            raise RuntimeError("Call initialize_buffer(n_particles) before store_state().")

        if self.buf_cursor == self.buf_len:
            # buffer is full
            self.flush()

        # core state — always saved
        self.buf_t[self.buf_cursor] = float(t)
        self.buf_pos[self.buf_cursor] = np.asarray(pos, dtype=np.float64).reshape(-1)
        self.buf_vel[self.buf_cursor] = np.asarray(vel, dtype=np.float64).reshape(-1)
        self.buf_mass[self.buf_cursor] = np.asarray(masses, dtype=np.float64)
        self.buf_radius[self.buf_cursor] = np.asarray(radii, dtype=np.float64).reshape(-1)
        self.buf_hashes[self.buf_cursor] = np.asarray(hashes, dtype=np.int64).reshape(-1)
        self.buf_ptype[self.buf_cursor] = np.asarray(ptypes, dtype=np.int32).reshape(-1)

        self.buf_cursor += 1