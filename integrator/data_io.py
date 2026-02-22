import numpy as np
import h5py
import os
import constants

class DataIO(object):
    # handling the data aspect of the project
    def __init__(
            self,
            buf_len=1024,
            output_file_name='data.hdf5',
            collision_file_name='collisions.txt',
            close_encounter_name='close_encounters.txt',
            const_g=constants.G
        ):

        self.buf_len = buf_len
        self.buf_initialized=False
        self.buf_t = None # buffer for time vector
        self.store_t = 0.0 # the current time of the snapshot data
        self.buf_energy = None # store the total energy as well
        self.buf_state = None # for pos and vel
        self.buf_pos = None
        self.buf_vel = None
        self.buf_mass = None
        self.buf_radius = None
        self.buf_temp = None
        self.buf_ptype = None
        # save orbital parameters as well
        self.buf_a = None
        self.buf_e = None
        self.buf_i = None
        self.buf_hashes = None

        self.buf_cursor = 0
        self.output_name = output_file_name
        self.collision_name = collision_file_name
        self.close_encounter = close_encounter_name
        self.h5_file = None
        self.h5_step_id = 0
        self.const_g = const_g

    # initialize buffer and allocate space for everything in advance
    def initialize_buffer(self, n_particles):
        # if we have not yet initialized the buffer, allocate space
        if self.buf_initialized is False:
            buf_len = self.buf_len
            self.buf_t = np.full(buf_len, np.nan, dtype=np.float64)
            self.buf_energy = np.full(buf_len, np.nan, dtype=np.float64)

            self.buf_pos = np.full((buf_len, n_particles * 3), np.nan, dtype=np.float64)
            self.buf_vel = np.full((buf_len, n_particles * 3), np.nan, dtype=np.float64)
            self.buf_mass = np.full(buf_len, np.nan, dtype=np.float64)
            self.buf_radius = np.full(buf_len, np.nan, dtype=np.float64)
            self.buf_temp = np.full(buf_len, np.nan, dtype=np.float64)

            self.buf_ptype = np.full(buf_len, -1, dtype=np.int32)
            self.buf_hashes = np.full(buf_len, -1, dtype=np.int64)


            self.buf_a = np.full(buf_len, np.nan, dtype=np.float64)
            self.buf_e = np.full(buf_len, np.nan, dtype=np.float64)
            self.buf_i = np.full(buf_len, np.nan, dtype=np.float64)

            self.buf_cursor = 0
            self.h5_step_id = 0

            # if we have leftover files with the same name, remove them beforehand
            if os.path.isfile(self.collision_name):
                os.remove(self.collision_name)
            if os.path.isfile(self.close_encounter):
                os.remove(self.close_encounter)

            self.buf_initialized = True # we have now initialized a buffer
        else:
            return
    
    def reset_buffer(self):
        self.buf_initialized = False # we want to go back to initialization after saving the files

    def flush(self):
        # write the data to the hdf5 file
        if self.buf_cursor == 0:
            # already flushed
            return
        
        if self.h5_file is None:
            self.h5_file = h5py.File(self.output_name, "w") # if we haven't started writing a file, create one
            self.h5_file.attrs['G'] = self.const_g # assign the constant G

        h5_step_group = self.h5_file.create_group(f"Step_{self.h5_step_id}") # create a subgroup until a specified step

        sl = slice(0, self.buf_cursor)
        state_dict = {
            'time': self.buf_t[sl],
            'mass': self.buf_mass[sl],
            'ptype': self.buf_ptype[sl],
            'hash': self.buf_hashes[sl],
            'radius': self.buf_radius[sl],
            'temperature': self.buf_temp[sl],
            'pos': self.buf_pos[sl],
            'vel': self.buf_vel[sl],
            'a': self.buf_a[sl],
            'e': self.buf_e[sl],
            'i': self.buf_i[sl],
            'energy': self.buf_energy[sl]
        }

        for k, v in state_dict.items():
            h5_step_group.create_dataset(k, data=v)

        # reset cursor
        self.h5_file.flush()
        self.h5_step_id += 1
        self.buf_cursor = 0
    
    def close(self):
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
            temperature: np.array,
            radii=None,
            hashes=None,
            ptypes=None,
            a=None,
            e=None,
            i=None,
            energy=None
    ):
        if not self.buf_initialized:
            raise RuntimeError("Call initialize_buffer(n_particles) before store_state().")
        
        if self.buf_cursor == self.buf_len:
            # buffer is full
            self.flush()
            #self.buf_cursor = 0
        
        self.buf_t[self.buf_cursor] = float(t)

        self.buf_pos[self.buf_cursor] = np.asarray(pos, dtype=np.float64).reshape(-1)
        self.buf_vel[self.buf_cursor] = np.asarray(vel, dtype=np.float64).reshape(-1)
        self.buf_mass[self.buf_cursor] = np.asarray(masses, dtype=np.float64)

        if radii is not None:
            self.buf_radius[self.buf_cursor] = np.asarray(radii, dtype=np.float64)

        if temperature is not None:
            self.buf_temp[self.buf_cursor] = np.asarray(temperature, dtype=np.float64)

        if hashes is not None:
            self.buf_hashes[self.buf_cursor] = np.asarray(hashes, dtype=np.int64)

        if ptypes is not None:
            self.buf_ptype[self.buf_cursor] = np.asarray(ptypes, dtype=np.int32)

        if a is not None:
            self.buf_a[self.buf_cursor] = np.asarray(a, dtype=np.float64)

        if e is not None:
            self.buf_e[self.buf_cursor] = np.asarray(e, dtype=np.float64)

        if i is not None:
            self.buf_i[self.buf_cursor] = np.asarray(i, dtype=np.float64)

        if energy is not None:
            self.buf_energy[self.buf_cursor] = float(energy)

        self.buf_cursor += 1

    
    def store_collisions(self, collision_buffer):
        if self.collision_name is not None:
            np.savetxt(
                self.collision_name,
                collision_buffer,
                fmt="%g, %d, %d, %g",
                header="Time, Particle 1, Particle 2, Distance"
            )