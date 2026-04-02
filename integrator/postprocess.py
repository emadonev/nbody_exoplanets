"""
Reads raw state (pos, vel, mass, ptype, radius) and computes all derived
quantities: orbital elements, energy, temperature, habitable zone diagnostics.
Results are written back into the same HDF5 file.
"""

import numpy as np
import h5py
from . import constants, tools
from .physics import Physics


class _Snapshot:
    """Lightweight wrapper so Physics.calculate_temp can work on HDF5 arrays."""
    def __init__(self, pos, radii, temperatures, star_indices, planet_indices):
        self.pos = pos
        self.radii = radii
        self.temperatures = temperatures.copy()
        self.star_indices = star_indices
        self.planet_indices = planet_indices


def _resolve_host_indices(host_star, star_idx):
    """
    Map a host_star label ('A', 'B', 'AB', 'ABC', etc.) to the array
    indices of the stars that form the orbital reference.
    """
    label_map = {'A': 0, 'B': 1, 'C': 2}
    if host_star in label_map:
        return np.array([star_idx[label_map[host_star]]])
    host_indices = []
    for ch in str(host_star):
        if ch in label_map and label_map[ch] < len(star_idx):
            host_indices.append(star_idx[label_map[ch]])
    if len(host_indices) == 0:
        return star_idx[:1]
    return np.array(host_indices, dtype=int)


def _primary_state(pos, vel, masses, host_indices):
    """
    Return (total_mass, com_pos, com_vel) for a set of host body indices.
    Single star: returns that star directly.
    Multiple stars: returns their centre of mass.
    """
    if host_indices.size == 1:
        i = host_indices[0]
        return float(masses[i]), pos[i], vel[i]
    m = masses[host_indices]
    mtot = float(m.sum())
    com_pos = (m[:, None] * pos[host_indices]).sum(axis=0) / mtot
    com_vel = (m[:, None] * vel[host_indices]).sum(axis=0) / mtot
    return mtot, com_pos, com_vel


def postprocess(h5_path: str, host_star: str = 'auto'):
    physics = Physics()
    G = constants.G

    with h5py.File(h5_path, "r+") as f:
        if f.attrs.get('postprocessed', False):
            print(f"  {h5_path}: already post-processed, skipping")
            return

        keys = sorted(
            [k for k in f.keys() if k.startswith("Step_")],
            key=lambda s: int(s.split("_")[1]),
        )

        # read first snapshot to determine system structure
        g0 = f[keys[0]]
        mass0 = g0["mass"][0]
        ptype0 = g0["ptype"][0]
        n_particles = mass0.shape[0]

        star_idx = np.where(ptype0 == 0)[0]
        planet_idx = np.where(ptype0 == 1)[0]

        # resolve host_star label (from file attr or argument)
        if host_star == 'auto':
            host_star = f.attrs.get('host_star', 'A')
            if isinstance(host_star, bytes):
                host_star = host_star.decode()
        host_indices = _resolve_host_indices(host_star, star_idx)

        # stellar temperatures
        if 'star_temperatures' in f.attrs:
            star_temps = np.array(f.attrs['star_temperatures'])
        else:
            star_temps = None

        # process each step group
        for step_key in keys:
            g = f[step_key]
            n_snap = g["time"].shape[0]

            pos_flat = g["pos"][:]
            vel_flat = g["vel"][:]
            mass_arr = g["mass"][:]
            radius_arr = g["radius"][:]

            pos = pos_flat.reshape(n_snap, n_particles, 3)
            vel = vel_flat.reshape(n_snap, n_particles, 3)

            # --- Energy ---
            energy = np.array([
                physics.energy(pos[s], vel[s], mass_arr[s], G)
                for s in range(n_snap)
            ])

            # --- Orbital elements for ALL bodies ---
            # Each body's elements are computed relative to its natural primary:
            #   - Star B (idx 1) relative to Star A (idx 0)  [inner binary]
            #   - Star C (idx 2) relative to COM(A+B)         [outer orbit]
            #   - Planets relative to their host COM           [from host_indices]
            a_arr = np.full((n_snap, n_particles), np.nan)
            e_arr = np.full((n_snap, n_particles), np.nan)
            inc_arr = np.full((n_snap, n_particles), np.nan)

            for s in range(n_snap):
                # Inner binary: star B relative to star A
                if len(star_idx) >= 2:
                    iA, iB = star_idx[0], star_idx[1]
                    mu_AB = mass_arr[s, iA]
                    pos_rel = (pos[s, iB] - pos[s, iA]).reshape(1, 3)
                    vel_rel = (vel[s, iB] - vel[s, iA]).reshape(1, 3)
                    a_s, e_s, i_s = tools.aei(
                        mp=mass_arr[s, iB:iB+1], Ms=mu_AB,
                        pos=pos_rel, vel=vel_rel, G=G,
                    )
                    a_arr[s, iB] = a_s[0]
                    e_arr[s, iB] = e_s[0]
                    inc_arr[s, iB] = i_s[0]

                # Outer orbit: star C relative to COM(A+B)
                if len(star_idx) >= 3:
                    iC = star_idx[2]
                    mAB = mass_arr[s, iA] + mass_arr[s, iB]
                    com_pos_AB = (mass_arr[s, iA] * pos[s, iA] + mass_arr[s, iB] * pos[s, iB]) / mAB
                    com_vel_AB = (mass_arr[s, iA] * vel[s, iA] + mass_arr[s, iB] * vel[s, iB]) / mAB
                    pos_rel = (pos[s, iC] - com_pos_AB).reshape(1, 3)
                    vel_rel = (vel[s, iC] - com_vel_AB).reshape(1, 3)
                    a_s, e_s, i_s = tools.aei(
                        mp=mass_arr[s, iC:iC+1], Ms=mAB,
                        pos=pos_rel, vel=vel_rel, G=G,
                    )
                    a_arr[s, iC] = a_s[0]
                    e_arr[s, iC] = e_s[0]
                    inc_arr[s, iC] = i_s[0]

                # Planets relative to host COM
                if len(planet_idx) > 0:
                    Ms, com_pos, com_vel = _primary_state(
                        pos[s], vel[s], mass_arr[s], host_indices,
                    )
                    pos_rel = pos[s, planet_idx] - com_pos[None, :]
                    vel_rel = vel[s, planet_idx] - com_vel[None, :]
                    mp = mass_arr[s, planet_idx]
                    a_s, e_s, i_s = tools.aei(
                        mp=mp, Ms=Ms, pos=pos_rel, vel=vel_rel, G=G,
                    )
                    a_arr[s, planet_idx] = a_s
                    e_arr[s, planet_idx] = e_s
                    inc_arr[s, planet_idx] = i_s

            # --- Temperature & HZ ---
            if star_temps is None:
                _write_datasets(g, {
                    "energy": energy,
                    "a": a_arr, "e": e_arr, "inc": inc_arr,
                })
                continue

            # build per-snapshot diagnostics via Physics.calculate_temp
            temperature = np.full((n_snap, n_particles), np.nan)
            F_bol_arr = np.full((n_snap, n_particles), np.nan)
            F_sw_inner_arr = np.full((n_snap, n_particles), np.nan)
            F_sw_outer_arr = np.full((n_snap, n_particles), np.nan)
            in_hz_arr = np.full((n_snap, n_particles), np.nan)
            S_eff_inner_arr = np.full((n_snap, n_particles), np.nan)
            S_eff_outer_arr = np.full((n_snap, n_particles), np.nan)
            hz_inner_r_arr = np.full((n_snap, n_particles), np.nan)
            hz_outer_r_arr = np.full((n_snap, n_particles), np.nan)

            for s in range(n_snap):
                temps = np.full(n_particles, np.nan)
                temps[star_idx] = star_temps

                snap = _Snapshot(
                    pos=pos[s],
                    radii=radius_arr[s],
                    temperatures=temps,
                    star_indices=star_idx,
                    planet_indices=planet_idx,
                )
                _, hz_data = physics.calculate_temp(snap)

                temperature[s] = snap.temperatures
                for k, hzd in hz_data.items():
                    F_bol_arr[s, k] = hzd['F_bol']
                    F_sw_inner_arr[s, k] = hzd['F_sw_inner']
                    F_sw_outer_arr[s, k] = hzd['F_sw_outer']
                    in_hz_arr[s, k] = float(hzd['in_hz'])
                    S_eff_inner_arr[s, k] = hzd['S_eff_inner']
                    S_eff_outer_arr[s, k] = hzd['S_eff_outer']
                    hz_inner_r_arr[s, k] = hzd['hz_inner_radius']
                    hz_outer_r_arr[s, k] = hzd['hz_outer_radius']

            for si in star_idx:
                temperature[:, si] = star_temps[np.where(star_idx == si)[0][0]]

            _write_datasets(g, {
                "energy": energy,
                "a": a_arr, "e": e_arr, "inc": inc_arr,
                "temperature": temperature,
                "F_bol": F_bol_arr,
                "F_sw_inner": F_sw_inner_arr,
                "F_sw_outer": F_sw_outer_arr,
                "in_hz": in_hz_arr,
                "S_eff_inner": S_eff_inner_arr,
                "S_eff_outer": S_eff_outer_arr,
                "hz_inner_radius": hz_inner_r_arr,
                "hz_outer_radius": hz_outer_r_arr,
            })

        f.attrs['postprocessed'] = True


def _write_datasets(group, data_dict):
    """Write datasets to an HDF5 group, overwriting if they already exist."""
    for k, v in data_dict.items():
        if k in group:
            del group[k]
        group.create_dataset(k, data=v, compression='gzip', compression_opts=4)
