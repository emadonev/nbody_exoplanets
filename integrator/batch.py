import numpy as np
from .particles import Particles
from .particle import Particle
from .physics import Physics
from .data_io import DataIO
from .simulation import Simulation
from .WH_SC_P import WisdomHolman_SC_P
from . import constants
import h5py


# Maps system type strings to their integrator classes.
INTEGRATOR_MAP = {
    'S(C)':   WisdomHolman_SC_P,
}

# Maps host_star label to the primary= argument for planet particles.
HOST_PRIMARY_MAP = {
    'A':   'star A',
    'B':   'star B',
    'C':   'star C',
    'AB':  ['star A', 'star B'],
    'ABC': ['star A', 'star B', 'star C'],
}

def _safe(value, default=0.0):
    if value is None:
        return default
    try:
        if np.isnan(value):
            return default
    except (TypeError, ValueError):
        pass
    return float(value)


# check if semimajor axis is defined and if not set a default value
def _has_valid_orbit(config, prefix):
    a = config.get(f'{prefix}_a')
    if a is None:
        return False
    try:
        return np.isfinite(a) and a > 0
    except (TypeError, ValueError):
        return False

# calculate the keplerian period for best dt
def _orbital_period(a, mu):
    a = float(a)
    mu = float(mu)
    if not (np.isfinite(a) and a > 0.0):
        raise ValueError(f"semi-major axis must be finite and positive, got {a}")
    if not (np.isfinite(mu) and mu > 0.0):
        raise ValueError(f"mu must be finite and positive, got {mu}")
    return 2.0 * np.pi * np.sqrt(a**3 / mu)

# return mu for dt selection
def _planet_mu(config, host, mass_earth):
    mp = float(mass_earth) * 3.00274e-6  # M_earth -> M_sun
    mA = float(config['mA'])
    mB = float(config['mB'])
    mC = float(config['mC'])
    if host == 'A':
        return constants.G * (mA + mp)
    if host == 'B':
        return constants.G * (mB + mp)
    if host == 'C':
        return constants.G * (mC + mp)
    if host == 'AB':
        return constants.G * (mA + mB + mp)
    if host == 'ABC':
        return constants.G * (mA + mB + mC + mp)
    raise ValueError(f"Unsupported host_star {host!r}")

# select the shortest period in the system
def _shortest_period(config: dict) -> float:
    periods = []

    mu_inner = constants.G * (float(config['mA']) + float(config['mB']))
    periods.append(_orbital_period(config['inner_a'], mu_inner))

    mu_outer = constants.G * (float(config['mA']) + float(config['mB']) + float(config['mC']))
    periods.append(_orbital_period(config['outer_a'], mu_outer))

    host = config['host_star']
    for pl in config.get('planets', []):
        mu_planet = _planet_mu(config, host, pl['mass_earth'])
        periods.append(_orbital_period(pl['a'], mu_planet))

    return min(periods)

# the recommended time step based on the shortest period in the system - 1/10th of that is dt
def recommended_dt(config: dict, steps_per_shortest_orbit: int = 10) -> float:
    if steps_per_shortest_orbit <= 0:
        raise ValueError("steps_per_shortest_orbit must be positive")
    return _shortest_period(config) / float(steps_per_shortest_orbit)


def run_system(config: dict) -> str:
    """
    Run a single N-body simulation from a serializable config dict.

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

    # ensure semimajor axis is defined for the stellar orbits, otherwise integration is not possible
    if not _has_valid_orbit(config, 'inner'):
        raise ValueError(
            f"System '{config.get('output_file', '?')}': inner binary orbit has no valid "
            f"semi-major axis (inner_a={config.get('inner_a')}). Cannot integrate."
        )
    if not _has_valid_orbit(config, 'outer'):
        raise ValueError(
            f"System '{config.get('output_file', '?')}': outer orbit has no valid "
            f"semi-major axis (outer_a={config.get('outer_a')}). Cannot integrate."
        )

    dt = float(config['dt'])
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"dt must be positive and finite, got {config['dt']}")

    dt_recommended = recommended_dt(config)
    if dt > dt_recommended:
        raise ValueError(
            f"dt={dt:.6g} is too large for this system. "
            f"Recommended dt <= {dt_recommended:.6g} based on the shortest orbit."
        )

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

    # setting the correct primary for planets
    particles.primary = planet_primary

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
        dt=dt,
        output_every_n=int(config.get('output_every_n', 5)),
        handle_collisions=bool(config.get('handle_collisions', False)),
    )

    # Store stellar temperatures as file attribute for post-processing
    star_temps = np.array([
        particles.temperatures[i]
        for i in range(particles.N)
        if particles.ptypes[i] == 0
    ])
    with h5py.File(output_path, "a") as hf:
        hf.attrs['star_temperatures'] = star_temps
        hf.attrs['host_star'] = host

    return output_path
