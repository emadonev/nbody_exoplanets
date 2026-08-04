"""
Expand per-system YAML into fully-resolved run configs.

For each run it writes:
  configs/<run_id>.json   -- COMPLETE initial conditions (this feeds the simulator)
  <SYSTEM>_manifest.csv   -- one summary row per run (human index; join results here)

Run from anywhere:  python build_manifest.py [system.yaml ...]
Internal units: AU, year, M_sun, radian.  G = 4*pi^2.
"""
import sys, os, json, yaml, itertools, csv
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent          # folder this script lives in
G = 4.0 * np.pi**2
MSUN_PER_MEARTH = 1.0 / 332946.0
DAY_PER_YR = 365.25
TWO_PI = 2 * np.pi

# masses interior to each body's orbit (NOT counting the body itself)
INTERIOR = {
    'GJ667':   {'C': ['AB'], 'b': ['C'], 'c': ['C']},
    'LTT1445': {'B': ['C'], 'A': ['B', 'C'], 'b': ['A'], 'c': ['A'], 'd': ['A']},
}


# convert the mass into solar masses if in earth masses (planets)
def to_msun(m):
    return m['value'] * MSUN_PER_MEARTH if m.get('unit') == 'Mearth' else m['value']

# convert degrees to radians for later conversions
def to_rad(a):
    return np.deg2rad(a['value']) if a.get('unit', 'deg') == 'deg' else float(a['value'])


# accessing the mass of a body - in case of GJ 667 AB we get both of their masses
def body_mass(bodies, name):
    return to_msun(bodies[name]['mass'])

# calculate the mu of any chosen masses
def mu_of(system, bodies, name):
    """G * (interior mass + this body's mass) -- the Kepler mu for this orbit."""
    interior = sum(body_mass(bodies, nm) for nm in INTERIOR[system][name])
    return G * (interior + body_mass(bodies, name))

# Newton solver for kepler's equation to get the theta value if we have the mean anomaly
def kepler(M, e):
    M = (M + np.pi) % TWO_PI - np.pi
    E = M if e < 0.8 else np.pi
    for _ in range(60):
        E -= (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
    return 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2), np.sqrt(1 - e) * np.cos(E / 2))

# calculate the true anomaly based on how the T_0 or lambda_ref value is taken
def theta0(orbit, e, P_yr, t_ref_bjd):
    ep = orbit['epoch']; kind = ep['type']
    
    # just accessing the true anomaly
    if kind == 'fixed_anomaly':
        th = to_rad(ep)
        return th if str(ep.get('kind', 'true_anomaly')).startswith('true') else kepler(th, e)
    
    # this is for the case when we do not have any anomaly and we want similar starting conditions for all planets
    if kind == 'fiducial_phase':
        return float(np.arccos(-e))
    
    P_day = P_yr * DAY_PER_YR

    # if we have the periastron time, convert to M and then later complete Kepler
    if kind == 'periastron_bjd':
        M = TWO_PI * (t_ref_bjd - ep['T0']['value']) / P_day
    
    # if we have the conjunction time, first convert to periastron time and then later Kepler
    elif kind == 'inferior_conjunction_bjd':
        w = to_rad(orbit['omega']); Tc = ep['T0']['value']
        fc = np.pi / 2 - w
        Ec = 2 * np.arctan2(np.sqrt(1 - e) * np.sin(fc / 2), np.sqrt(1 + e) * np.cos(fc / 2))
        Tp = Tc - P_day * (Ec - e * np.sin(Ec)) / TWO_PI
        M = TWO_PI * (t_ref_bjd - Tp) / P_day
    
    # if we have the mean longitude, calculate the mean anomaly and repeat Kepler
    elif kind == 'mean_longitude':
        return float(kepler((to_rad(ep['lambda_ref']) - to_rad(orbit['omega'])) % TWO_PI, e))
    
    # in case of the besselian time, convert to regular T_0, mean anomaly and Kepler
    elif kind == 'periastron_besselian':
        T0_jd = 2415020.31352 + (ep['T0']['value'] - 1900.0) * 365.242198781
        M = TWO_PI * (t_ref_bjd - T0_jd) / P_day
    else:
        raise ValueError(f"unknown epoch type: {kind}")
    return float(kepler(M % TWO_PI, e))

# build an orbit
def resolve_orbit(system, bodies, name, orbit, ov, t_ref):
    """Turn one orbit block into fully-numeric elements (SI-free, radians)."""
    # eccentricity and inclination may be swept
    e = ov[f"{name}.e"] if f"{name}.e" in ov else orbit['e']['value']
    inc = np.deg2rad(ov[f"{name}.i"]) if f"{name}.i" in ov else to_rad(orbit['i'])

    # semi-major axis: from period (planets) or literal a (stars)
    if 'period' in orbit:
        P_yr = orbit['period']['value'] / DAY_PER_YR
        a = (P_yr**2 * (mu_of(system, bodies, name) / G))**(1 / 3)
    else:
        a = orbit['a']['value']
        P_yr = TWO_PI * np.sqrt(a**3 / mu_of(system, bodies, name))

    # all other orbital elements get saved
    return {
        'primary':    orbit.get('primary'),
        'a_au':       float(a),
        'e':          float(e),
        'inc_rad':    float(inc),
        'Omega_rad':  float(to_rad(orbit['Omega'])),
        'omega_rad':  float(to_rad(orbit['omega'])),
        'theta0_rad': theta0(orbit, e, P_yr, t_ref),
        'period_yr':  float(P_yr),
    }

# in the case of GJ 667 we need to sweep the values of eccentricity and inclination
def sweep_axes(bodies):
    return {f"{bn}.{k}": v['sweep']
            for bn, b in bodies.items()
            for k, v in b.get('orbit', {}).items()
            if isinstance(v, dict) and 'sweep' in v}

# build the JSON file for each system
def build(path):
    d = yaml.safe_load(open(path))
    system, bodies = d['system'], d['bodies']
    exp = d['experiment']; t_ref = exp['t_ref_bjd']
    frame = d.get('reference_frame')

    axes = sweep_axes(bodies); keys = list(axes)
    combos = list(itertools.product(*axes.values())) if axes else [()]
    sets = exp.get('sets', [{'name': 'default', 'active_planets': exp.get('active_planets')}])

    cfg_dir = HERE / 'configs'; cfg_dir.mkdir(exist_ok=True)
    summary = []

    for st in sets:
        active = st['active_planets']
        for vals in combos:
            ov = dict(zip(keys, vals))
            rid = f"{system}__{st['name']}" + "".join(f"__{k.replace('.', '')}{v}" for k, v in ov.items())

            # --- FULL resolved config (this is what the simulator will load) ---
            cfg = {
                'run_id': rid, 'system': system, 'set': st['name'],
                'reference_frame': frame, 't_ref_bjd': t_ref,
                'integration': dict(exp['integration']),
                'active_planets': active,
                'bodies': {},
            }
            for bn, b in bodies.items():
                if b.get('role') == 'planet' and bn not in active:
                    continue                                   # skip inactive planets
                entry = {'role': b.get('role'), 'mass_msun': float(body_mass(bodies, bn))}
                if 'orbit' in b:
                    entry['orbit'] = resolve_orbit(system, bodies, bn, b['orbit'], ov, t_ref)
                cfg['bodies'][bn] = entry

            with open(cfg_dir / f"{rid}.json", 'w') as fh:
                json.dump(cfg, fh, indent=2)

            # --- thin summary row for the CSV index ---
            row = {'run_id': rid, 'system': system, 'set': st['name'],
                   'n_planets': len(active),
                   'tspan_yr': float(exp['integration']['tspan_yr']),
                   'config_file': f"configs/{rid}.json"}
            row.update({k.replace('.', '_'): v for k, v in ov.items()})
            summary.append(row)

    cols = list(dict.fromkeys(k for r in summary for k in r))
    out = HERE / f"{system}_manifest.csv"
    with open(out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(summary)
    print(f"{system}: {len(summary)} runs -> {out.name}  (+ configs/*.json)")


if __name__ == '__main__':
    paths = sys.argv[1:] or [HERE / 'gj667.yaml', HERE / 'ltt1445.yaml']
    for p in paths:
        build(p)
