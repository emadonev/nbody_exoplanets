import numpy as np
from .particles import Particles
from .particle import Particle
from .physics import Physics
from .data_io import DataIO
from .simulation import Simulation
from .WH_SC_P import WisdomHolman_SC_P
from .WH_SC_P2 import WisdomHolman_SC_P2
from . import constants
import h5py
import json


# Maps system type strings to their integrator classes.
INTEGRATOR_MAP = {
    'LTT1445':   WisdomHolman_SC_P,
    'GJ667': WisdomHolman_SC_P2
}

role_map = {
    'inner_primary': 'A',
    'inner_secondary': 'B',
    'host': 'C',
    'perturber': 'A'
}

# choose role for index
def role_to_index(config, name, role_map):
    role = config['bodies'][name]['role']
    return role_map[role]

# select the shortest period in the system
def best_dt(config) -> float:
    periods = []

    for name in config['active_planets']:
        periods.append(config['bodies'][name]['orbit']['period_yr'])

    return min(periods) / float(20)

def planet_type(config, name):
    if config['bodies'][name]['role'] == 'planet':
        return 1
    else:
        return 0

def run_system(config_path: str) -> str:
    """
    Run a single N-body simulation from a serializable config dict.

    Returns the path to the output HDF5 file.
    """

    with open(config_path, 'r') as file:
        config = json.load(file)


    #stype = config['system_type']
    host = next(name for name, b in config['bodies'].items() if b['role'] == 'host')

    # calculation of dt
    dt = float(best_dt(config=config))


    # --- BUILD PARTICLES ---
    particles = Particles(constants.G)

    # go object by object
    for name in config['bodies']:
        # if the object doesn't have an orbit - set it at 0,0
        if name in config['active_planets']:
            particles.add_particle(Particle(
                ptype=planet_type(config, name), mass=config['bodies'][name]['mass_msun'], radius=0,
                temperature=0, name=name, index=name,
                primary= config['bodies'][name]['orbit']['primary'], angles_in_degrees=False,
                a=config['bodies'][name]['orbit']['a_au'],
                e=config['bodies'][name]['orbit']['e'],
                inc=config['bodies'][name]['orbit']['inc_rad'],
                Omega=config['bodies'][name]['orbit']['Omega_rad'],
                omega=config['bodies'][name]['orbit']['omega_rad'],
                theta=config['bodies'][name]['orbit']['theta0_rad'],
            ))
        else:
            if config['bodies'][name].get('orbit') == None:
                particles.add_particle(Particle(
                    ptype=planet_type(config, name), mass=config['bodies'][name]['mass_msun'], radius=0,
                    temperature=0, name=name,
                    pos=[0.0, 0.0, 0.0], vel=[0.0, 0.0, 0.0], index=role_to_index(config, name, role_map),
                ))
            else:
                particles.add_particle(Particle(
                ptype=planet_type(config, name), mass=config['bodies'][name]['mass_msun'], radius=0,
                temperature=0, name=name, index=role_to_index(config, name, role_map),
                primary= config['bodies'][name]['orbit']['primary'], angles_in_degrees=False,
                a=config['bodies'][name]['orbit']['a_au'],
                e=config['bodies'][name]['orbit']['e'],
                inc=config['bodies'][name]['orbit']['inc_rad'],
                Omega=config['bodies'][name]['orbit']['Omega_rad'],
                omega=config['bodies'][name]['orbit']['omega_rad'],
                theta=config['bodies'][name]['orbit']['theta0_rad'],
            ))

    # Shift to system COM frame
    m = particles.masses
    r_com = (m[:, None] * particles.pos).sum(axis=0) / m.sum()
    v_com = (m[:, None] * particles.vel).sum(axis=0) / m.sum()
    particles._pos -= r_com[None, :]
    particles._vel -= v_com[None, :]
    particles._sync_objects()

    # setting the correct primary for planets
    planets = config['active_planets']
    particles.primary = config['bodies'][planets[0]]['orbit']['primary']

    # --- create integrator, I/O, simulation ---
    integrator_cls = INTEGRATOR_MAP[config['system']]
    integrator = integrator_cls()

    dataio = DataIO(
        buf_len=int(config.get('buf_len', 1024)),
        output_file_name='/Users/emadonev/Documents/PROJECTS/nbody_exoplanets/output/' + config['run_id'] + '.h5',
        const_g=constants.G,
    )

    sim = Simulation(particles, integrator, dataio, Physics())

    star_temps = np.array([
        particles.temperatures[i]
        for i in range(particles.N)
        if particles.ptypes[i] == 0
    ])

    try:
        out_cadence_yr = config['integration']['out_cadence_yr']
        sim.run(
            t0=config.get('t0', 0.0),
            #tf=config['integration']['tspan_yr'],
            tf=10000,
            dt=dt,
            output_every_n=max(1, round(out_cadence_yr / dt)),
            handle_collisions=bool(config.get('handle_collisions', False)),
        )
    finally:
        # Always write metadata so post-processing can work on partial runs
        with h5py.File(dataio.output_name, "a") as hf:
            hf.attrs['star_temperatures'] = star_temps
            hf.attrs['host_star'] = host

    return dataio.output_name
