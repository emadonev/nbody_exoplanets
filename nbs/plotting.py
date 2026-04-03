import matplotlib.pyplot as plt
import seaborn as sns
import h5py
import numpy as np
from scipy.ndimage import uniform_filter1d
import os, re

def load_hdf5_data(H5_PATH):
    with h5py.File(H5_PATH, "r") as f:
        keys = sorted(
            [k for k in f.keys() if k.startswith("Step_")],
            key=lambda s: int(s.split("_")[1])
        )
        t_list, pos_list, mass_list, ptype_list, energy_list, temperature_list, in_hz_list = [], [], [], [], [], [], []
        hz_inner_radius_list, hz_outer_radius_list = [], []
        a_list, e_list, inc_list = [], [], []
        s_eff_inner_list, s_eff_outer_list = [], []
        for k in keys:
            g = f[k]
            t_list.append(g["time"][:])
            pos_list.append(g["pos"][:])
            mass_list.append(g["mass"][:])
            ptype_list.append(g["ptype"][:])
            
            hz_inner_radius_list.append(g["hz_inner_radius"][:])
            hz_outer_radius_list.append(g["hz_outer_radius"][:])
            energy_list.append(g["energy"][:])
            temperature_list.append(g["temperature"][:])
            in_hz_list.append(g["in_hz"][:])
            a_list.append(g["a"][:])
            e_list.append(g["e"][:])
            inc_list.append(g["inc"][:])

            s_eff_inner_list.append(g['S_eff_inner'][:])
            s_eff_outer_list.append(g['S_eff_outer'][:])
            
    t = np.concatenate(t_list)
    pos = np.concatenate(pos_list)
    mass = np.concatenate(mass_list)
    ptype = np.concatenate(ptype_list)
    hz_inner_radius = np.concatenate(hz_inner_radius_list)
    hz_outer_radius = np.concatenate(hz_outer_radius_list)
    energy = np.concatenate(energy_list)
    temperature = np.concatenate(temperature_list)
    in_hz = np.concatenate(in_hz_list)
    a = np.concatenate(a_list)
    e = np.concatenate(e_list)
    inc = np.concatenate(inc_list)
    s_eff_inner = np.concatenate(s_eff_inner_list)
    s_eff_outer = np.concatenate(s_eff_outer_list)

    n_particles = mass.shape[1]
    if pos.ndim == 2:
        pos = pos.reshape(pos.shape[0], n_particles, 3)

    return t, pos, mass, ptype, hz_inner_radius, hz_outer_radius, energy, temperature, in_hz, a, e, inc, s_eff_inner, s_eff_outer

def determine_system(mass, pos, ptype):
    star_idx = np.where(ptype[0] == 0)[0]
    planet_idx = np.where(ptype[0] == 1)[0]
    m0 = mass[0]

    iA, iB = star_idx[0], star_idx[1]
    iC = star_idx[2] if len(star_idx) > 2 else None
    mA, mB = m0[iA], m0[iB]
    r_com_bin = (mA * pos[:, iA, :] + mB * pos[:, iB, :]) / (mA + mB)

    return star_idx, planet_idx, iA, iB, iC, mA, mB, r_com_bin

def plot_orbital_elements(path, star_smooth=1, planet_smooth=1000, save=None):
    t, pos, mass, ptype, hz_inner_radius, hz_outer_radius, energy, temperature, in_hz, a, e, inc, s_eff_inner, s_eff_outer = load_hdf5_data(path)
    star_idx, planet_idx, iA, iB, iC, mA, mB, r_com_bin = determine_system(mass, pos, ptype)

    scenario_mapping = {
        'hz inner': 'unutarnja granica habitabilnosti',
        'hz mid': 'sredina granice habitabilnosti',
        'hz outer': 'vanjska granica habitabilnosti',
        'observed': 'otkriveni planeti',
    }

    # Extract title from filename: e.g. "94_Ceti_hz_inner_ev0.3_iv104.hdf5"
    basename = os.path.splitext(os.path.basename(path))[0]
    # Parse system name, scenario, eccentricity, inclination
    m = re.match(r'(.+?)_(observed|hz_\w+)_ev([\d.]+)_iv(\d+)', basename)
    if m:
        system = m.group(1).replace('_', ' ')
        scenario_orig = m.group(2).replace('_', ' ')
        scenario = scenario_mapping[scenario_orig]
        outer_e = m.group(3)
        outer_i = m.group(4)
        title = f"{system} — {scenario} (e={outer_e}, i={outer_i}°)"
    else:
        title = basename.replace('_', ' ')

    star_colors = ['#ee9b00', '#bb3e03']
    planet_colors = ['#001219', '#005f73', '#0a9396', '#94d2bd', '#e9d8a6', '#bb3e03', '#ae2012']
    star_labels = {0: 'B', 1: 'C'}

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=14)

    # Left column: stars (skip A, it's the reference)
    for c, i in enumerate(star_idx[1:]):
        label = star_labels.get(c, f'Zvijezda {c+1}')
        a_s = uniform_filter1d(a[:, i], size=star_smooth, mode='reflect') if star_smooth > 1 else a[:, i]
        e_s = uniform_filter1d(e[:, i], size=star_smooth, mode='reflect') if star_smooth > 1 else e[:, i]
        i_s = uniform_filter1d(np.degrees(inc[:, i]), size=star_smooth, mode='reflect') if star_smooth > 1 else np.degrees(inc[:, i])

        axes[0, 0].plot(t, a_s, color=star_colors[c], label=label)
        axes[1, 0].plot(t, e_s, color=star_colors[c], label=label)
        axes[2, 0].plot(t, i_s, color=star_colors[c], label=label)

    axes[0, 0].set_ylabel('a [AU]')
    axes[0, 0].set_title('Zvijezde')
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(alpha=0.3)

    axes[1, 0].set_ylabel('e')
    axes[1, 0].grid(alpha=0.3)

    axes[2, 0].set_ylabel('i [°]')
    axes[2, 0].set_xlabel('Vrijeme (godina)')
    axes[2, 0].grid(alpha=0.3)

    # Right column: planets
    for c, j in enumerate(planet_idx):
        color = planet_colors[c % len(planet_colors)]
        label = f'Planet {c}'

        a_p = uniform_filter1d(a[:, j], size=planet_smooth, mode='reflect')
        e_p = uniform_filter1d(e[:, j], size=planet_smooth, mode='reflect')
        i_p = uniform_filter1d(np.degrees(inc[:, j]), size=planet_smooth, mode='reflect')

        axes[0, 1].plot(t, a_p, color=color, label=label)
        axes[1, 1].plot(t, e_p, color=color, label=label)
        axes[2, 1].plot(t, i_p, color=color, label=label)

    # Auto-scale planet axes nicely
    for row in range(3):
        ax = axes[row, 1]
        lines = ax.get_lines()
        if lines:
            all_y = np.concatenate([l.get_ydata() for l in lines])
            lo, hi = np.nanpercentile(all_y, [1, 99])
            pad = max((hi - lo) * 0.1, 1e-10)
            ax.set_ylim(lo - pad, hi + pad)
        ax.ticklabel_format(useOffset=False, style='plain')
        #ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=6))
        ax.grid(alpha=0.3)

    axes[0, 1].set_title('Planeti')
    axes[0, 1].legend(fontsize=8)
    axes[2, 1].set_xlabel('Vrijeme (godina)')

    plt.tight_layout()

    if save:
        plt.savefig('../plots/'+save, dpi=300, bbox_inches='tight')
    #plt.show()

def plot_temperature(path, planet_smooth=1000, save=None):
    t, pos, mass, ptype, hz_inner_radius, hz_outer_radius, energy, temperature, in_hz, a, e, inc, s_eff_inner, s_eff_outer = load_hdf5_data(path)
    star_idx, planet_idx, iA, iB, iC, mA, mB, r_com_bin = determine_system(mass, pos, ptype)

    print(temperature)
    planet_colors = ['#001219', '#005f73', '#0a9396', '#94d2bd', '#e9d8a6', '#bb3e03', '#ae2012']

    scenario_mapping = {
        'hz inner': 'unutarnja granica habitabilnosti',
        'hz mid': 'sredina granice habitabilnosti',
        'hz outer': 'vanjska granica habitabilnosti',
        'observed': 'otkriveni planeti',
    }

    # Extract title from filename: e.g. "94_Ceti_hz_inner_ev0.3_iv104.hdf5"
    basename = os.path.splitext(os.path.basename(path))[0]
    # Parse system name, scenario, eccentricity, inclination
    m = re.match(r'(.+?)_(observed|hz_\w+)_ev([\d.]+)_iv(\d+)', basename)
    if m:
        system = m.group(1).replace('_', ' ')
        scenario_orig = m.group(2).replace('_', ' ')
        scenario = scenario_mapping[scenario_orig]
        outer_e = m.group(3)
        outer_i = m.group(4)
        title = f"{system} — {scenario} (e={outer_e}, i={outer_i}°)"
    else:
        title = basename.replace('_', ' ')

    plt.figure(figsize=(14, 6))
    plt.title(title, fontsize=18)

    for c, j in enumerate(planet_idx):
        color = planet_colors[c % len(planet_colors)]
        label = f'Planet {c}'
        T_smooth = uniform_filter1d(temperature[:, j], size=planet_smooth, mode='reflect')
        plt.plot(t, T_smooth, color=color, label=label, lw=1.0)

    plt.gca().yaxis.set_major_locator(plt.MaxNLocator(nbins=8))

    plt.ticklabel_format(useOffset=False, style='plain')
    plt.xlabel('Vrijeme (godina)')
    plt.ylabel('T [K]')
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)

    s_inner = float(s_eff_inner[0, planet_idx[0]])
    s_outer = float(s_eff_outer[0, planet_idx[0]])

    T_inner = 278.0 * s_inner ** 0.25
    T_outer = 278.0 * s_outer ** 0.25

    plt.axhline(T_inner, ls='--', lw=2.0, color='#ae2012', label='Unutarnja HZ')
    plt.axhline(T_outer, ls='-.', lw=2.0, color='#ca6702', label='Vanjska HZ')

    plt.tight_layout()

    if save:
        plt.savefig('../plots/' + save, dpi=300, bbox_inches='tight')
    plt.show()