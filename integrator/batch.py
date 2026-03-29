import numpy as np
from .particles import Particles
from .particle import Particle
from .physics import Physics
from .data_io import DataIO
from .simulation import Simulation
from .WH_SA_P import WisdomHolman_SA_P
from .WH_SB_P import WisdomHolman_SB_P
from .WH_SC_P import WisdomHolman_SC_P
from .WH_SAB_P import WisdomHolman_SAB_P
from .WH_SABC_P import WisdomHolman_SABC_P
from . import constants

# Maps system type strings to their integrator classes.
INTEGRATOR_MAP = {
    'S(A)':   WisdomHolman_SA_P,
    'S(B)':   WisdomHolman_SB_P,
    'S(C)':   WisdomHolman_SC_P,
    'S(AB)':  WisdomHolman_SAB_P,
    'P(ABC)': WisdomHolman_SABC_P,
}

# Maps host_star label to the primary= argument for planet particles.
HOST_PRIMARY_MAP = {
    'A':   'star A',
    'B':   'star B',
    'C':   'star C',
    'AB':  ['star A', 'star B'],
    'ABC': ['star A', 'star B', 'star C'],
}


def _safe(val, default=0.0):
    """Replace NaN / None with a default value."""
    if val is None:
        return default
    try:
        if np.isnan(val):
            return default
    except (TypeError, ValueError):
        pass
    return val


def run_system(config: dict) -> str:
    """
    Run a single N-body simulation from a serializable config dict.
    Designed to be called by Spark workers for parallel execution.

    Returns the path to the output HDF5 file.

    Expected config keys
    --------------------
    system_type : str
        One of 'S(A)', 'S(B)', 'S(C)', 'S(AB)', 'P(ABC)'.
    host_star : str
        Which star the planets orbit: 'A', 'B', 'C', 'AB', or 'ABC'.

    Stars (all in solar units):
        mA, mB, mC        : masses  [M_sun]
        RA, RB, RC         : radii   [R_sun]
        TA, TB, TC         : effective temperatures [K]

    Inner binary orbital elements (B around A):
        inner_a, inner_e, inner_i, inner_Omega, inner_omega, inner_T0
        (angles in degrees)

    Outer orbit (C around COM(AB)):
        outer_a, outer_e, outer_i, outer_Omega, outer_omega, outer_T0
        (angles in degrees)

    planets : list[dict]
        Each dict contains:
            mass_earth, radius_earth  : physical properties
            a, e, inc, Omega, omega, theta  : orbital elements (radians)
        Planet orbital elements are relative to the host star/binary.

    Simulation control:
        t0, tf, dt           : time span and step size [yr]
        output_every_n       : snapshot cadence (default 1)
        output_file          : path to output HDF5 file
        buf_len              : DataIO buffer length (default 1024)
        handle_collisions    : bool (default False)
    """
    stype = config['system_type']
    host = config['host_star']

    # --- build particles ---
    particles = Particles(constants.G, system_type=stype)

    # Star A at origin
    particles.add_particle(Particle(
        ptype=0, mass=config['mA'], radius=config['RA'],
        temperature=config['TA'], name='star A',
        pos=[0.0, 0.0, 0.0], vel=[0.0, 0.0, 0.0], index='A',
    ))

    # Star B: inner binary companion orbiting star A
    particles.add_particle(Particle(
        ptype=0, mass=config['mB'], radius=config['RB'],
        temperature=config['TB'], name='star B', index='B',
        primary='star A', angles_in_degrees=True,
        a=_safe(config['inner_a']),
        e=_safe(config['inner_e']),
        inc=_safe(config['inner_i']),
        Omega=_safe(config['inner_Omega']),
        omega=_safe(config['inner_omega']),
        theta=_safe(config['inner_T0']),
    ))

    # Star C: outer companion orbiting COM(A,B)
    particles.add_particle(Particle(
        ptype=0, mass=config['mC'], radius=config['RC'],
        temperature=config['TC'], name='star C', index='C',
        primary=['star A', 'star B'], angles_in_degrees=True,
        a=_safe(config['outer_a']),
        e=_safe(config['outer_e']),
        inc=_safe(config['outer_i']),
        Omega=_safe(config['outer_Omega']),
        omega=_safe(config['outer_omega']),
        theta=_safe(config['outer_T0']),
    ))

    # Planets
    planet_primary = HOST_PRIMARY_MAP[host]
    for j, pl in enumerate(config.get('planets', [])):
        particles.add_particle(Particle(
            ptype=1,
            mass=pl['mass_earth'] * 3.00274e-6,       # M_earth -> M_sun
            radius=pl['radius_earth'] * 0.00916794,    # R_earth -> R_sun
            name=f'planet {j}', index='P',
            primary=planet_primary, angles_in_degrees=False,
            a=pl['a'], e=pl.get('e', 0.0),
            inc=pl.get('inc', 0.0), Omega=pl.get('Omega', 0.0),
            omega=pl.get('omega', 0.0), theta=pl.get('theta', 0.0),
        ))

    # Shift to system COM frame
    m = particles.masses
    r_com = (m[:, None] * particles.pos).sum(axis=0) / m.sum()
    v_com = (m[:, None] * particles.vel).sum(axis=0) / m.sum()
    particles._pos -= r_com[None, :]
    particles._vel -= v_com[None, :]
    particles._sync_objects()

    # --- create integrator, I/O, simulation ---
    integrator_cls = INTEGRATOR_MAP[stype]
    integrator = integrator_cls()

    dataio = DataIO(
        buf_len=int(config.get('buf_len', 1024)),
        output_file_name=config['output_file'],
        const_g=constants.G,
    )

    sim = Simulation(particles, integrator, dataio, Physics())

    output_path = sim.run(
        t0=config.get('t0', 0.0),
        tf=config['tf'],
        dt=config['dt'],
        output_every_n=int(config.get('output_every_n', 5)),
        handle_collisions=bool(config.get('handle_collisions', False)),
    )

    return output_path
