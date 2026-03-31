import copy
import re

import numpy as np

from .batch import recommended_dt, recommended_output_every_n
from .physics import Physics


DEFAULT_OUTER_E_VALUES = (0.0, 0.2, 0.4, 0.6, 0.8)
DEFAULT_OUTER_I_VALUES_DEG = (0.0, 30.0, 60.0, 90.0)
DEFAULT_INNER_E = 0.4
DEFAULT_INNER_I_DEG = 0.0
DETECTED_HABITABLE_SYSTEMS = frozenset({"Gliese 667", "LTT 1445"})

_PHYSICAL_LABELS = ("A", "B", "C")
_PAIR_RE = re.compile(r"^\s*([ABC]+)\s*-\s*([ABC]+)\s*$")


def _value(row, key):
    return row[key]


def _safe_float(value, default=0.0):
    if value is None:
        return float(default)
    try:
        if np.isnan(value):
            return float(default)
    except (TypeError, ValueError):
        pass
    return float(value)


def _is_missing(value):
    if value is None:
        return True
    try:
        return bool(np.isnan(value))
    except (TypeError, ValueError):
        return False


def _source_or_default(value, default):
    if _is_missing(value):
        return float(default)
    return float(value)


def _slugify(text):
    text = str(text).strip().replace(" ", "_")
    return re.sub(r"[^A-Za-z0-9_.-]", "_", text)


def _parse_pair_label(label):
    match = _PAIR_RE.match(str(label))
    if not match:
        raise ValueError(f"Invalid pair label {label!r}")
    left, right = match.groups()
    return left, right


def _expand_host_label(label):
    label = str(label).strip().replace(" ", "")
    parts = tuple(ch for ch in label if ch in _PHYSICAL_LABELS)
    if not parts:
        raise ValueError(f"Invalid host label {label!r}")
    return parts


def canonical_star_mapping(inner_pair, outer_pair):
    """
    Map physical star labels to the integrator's canonical hierarchy.

    The integrator expects:
    - internal A-B to be the inner binary
    - internal C to orbit COM(A,B)
    """
    inner_left, inner_right = _parse_pair_label(inner_pair)
    outer_left, outer_right = _parse_pair_label(outer_pair)

    inner_labels = tuple(inner_left + inner_right)
    if len(inner_labels) != 2 or len(set(inner_labels)) != 2:
        raise ValueError(f"Inner pair must contain exactly two distinct stars, got {inner_pair!r}")

    remaining = tuple(lbl for lbl in _PHYSICAL_LABELS if lbl not in inner_labels)
    if len(remaining) != 1:
        raise ValueError(f"Could not determine outer singleton from {inner_pair!r}")
    outer_singleton = remaining[0]

    outer_labels = {outer_left, outer_right}
    expected_outer = {outer_singleton, "".join(inner_labels)}
    expected_outer_reversed = {outer_singleton, "".join(inner_labels[::-1])}
    if outer_labels not in (expected_outer, expected_outer_reversed):
        raise ValueError(
            f"Outer pair {outer_pair!r} is inconsistent with inner pair {inner_pair!r}"
        )

    internal_to_physical = {
        "A": inner_labels[0],
        "B": inner_labels[1],
        "C": outer_singleton,
    }
    physical_to_internal = {physical: internal for internal, physical in internal_to_physical.items()}
    return internal_to_physical, physical_to_internal


def canonical_host_label(host_label, physical_to_internal):
    parts = _expand_host_label(host_label)
    internal = tuple(sorted(physical_to_internal[p] for p in parts))
    return "".join(internal)


def system_type_from_host(host_label):
    host_label = str(host_label)
    if host_label in ("A", "B", "C"):
        return f"S({host_label})"
    if host_label == "AB":
        return "S(AB)"
    if host_label == "ABC":
        return "P(ABC)"
    raise ValueError(f"Unsupported canonical host label {host_label!r}")


def base_config_from_row(row, tf=1e4, output_dir="output"):
    system_name = str(_value(row, "Sustav"))
    inner_pair = _value(row, "Oznake unutarnjeg para")
    outer_pair = _value(row, "Oznake vanjskog para")

    internal_to_physical, physical_to_internal = canonical_star_mapping(inner_pair, outer_pair)
    host_star = canonical_host_label(_value(row, "Oznaka matične zvijezde"), physical_to_internal)

    config = {
        "system_name": system_name,
        "source_system_type": _value(row, "Orbitalna konfiguracija"),
        "source_host_star": _value(row, "Oznaka matične zvijezde"),
        "source_inner_pair": inner_pair,
        "source_outer_pair": outer_pair,
        "source_inner_e": _value(row, "e_u"),
        "source_inner_i": _value(row, "i_u [º]"),
        "source_outer_e": _value(row, "e_v"),
        "source_outer_i": _value(row, "i_v [º]"),
        "source_has_inner_constraints": (
            not _is_missing(_value(row, "e_u")) or not _is_missing(_value(row, "i_u [º]"))
        ),
        "source_has_outer_constraints": (
            not _is_missing(_value(row, "e_v")) or not _is_missing(_value(row, "i_v [º]"))
        ),
        "internal_to_physical": dict(internal_to_physical),
        "physical_to_internal": dict(physical_to_internal),
        "system_type": system_type_from_host(host_star),
        "host_star": host_star,
        "mA": _value(row, f"Masa {internal_to_physical['A']} [m_s]"),
        "mB": _value(row, f"Masa {internal_to_physical['B']} [m_s]"),
        "mC": _value(row, f"Masa {internal_to_physical['C']} [m_s]"),
        "RA": _value(row, f"Radijus {internal_to_physical['A']} [R_s]"),
        "RB": _value(row, f"Radijus {internal_to_physical['B']} [R_s]"),
        "RC": _value(row, f"Radijus {internal_to_physical['C']} [R_s]"),
        "TA": _value(row, f"T_eff {internal_to_physical['A']} [K]"),
        "TB": _value(row, f"T_eff {internal_to_physical['B']} [K]"),
        "TC": _value(row, f"T_eff {internal_to_physical['C']} [K]"),
        "nameA": _value(row, f"Ime zvijezde {internal_to_physical['A']}"),
        "nameB": _value(row, f"Ime zvijezde {internal_to_physical['B']}"),
        "nameC": _value(row, f"Ime zvijezde {internal_to_physical['C']}"),
        "inner_a": _value(row, "a_u [AU]"),
        "inner_e": _safe_float(_value(row, "e_u")),
        "inner_i": _safe_float(_value(row, "i_u [º]")),
        "inner_Omega": _safe_float(_value(row, "Omega_u [º]")),
        "inner_omega": _safe_float(_value(row, "omega_u [º]")),
        "inner_T0": 0.0,
        "outer_a": _value(row, "a_v [AU]"),
        "outer_e": _safe_float(_value(row, "e_v")),
        "outer_i": _safe_float(_value(row, "i_v [º]")),
        "outer_Omega": _safe_float(_value(row, "Omega_v [º]")),
        "outer_omega": _safe_float(_value(row, "omega_v [º]")),
        "outer_T0": 0.0,
        "planets": [],
        "tf": float(tf),
        "output_file": f"{output_dir}/{_slugify(system_name)}.hdf5",
    }
    return config


def detected_planets_for_row(row, planets_df):
    system_name = str(_value(row, "Sustav"))
    subset = planets_df[planets_df["pl_name"].str.contains(system_name, na=False)]

    planets = []
    for _, planet in subset.iterrows():
        planets.append({
            "name": planet["pl_name"],
            "mass_earth": _safe_float(planet["pl_bmasse"], 1.0),
            "radius_earth": _safe_float(planet["pl_rade"], 1.0),
            "a": float(planet["pl_orbsmax"]),
            "e": _safe_float(planet["pl_orbeccen"], 0.0),
            "inc": 0.0,
            "Omega": 0.0,
            "omega": 0.0,
            "theta": 0.0,
        })
    return planets


def estimate_host_hz(config):
    physics = Physics()
    host_labels = tuple(str(config["host_star"]))
    radii = [config[f"R{label}"] for label in host_labels]
    temperatures = [config[f"T{label}"] for label in host_labels]
    return physics.static_habitable_zone(radii, temperatures)


def synthetic_hz_planets(config, mass_earth=1.0, radius_earth=1.0):
    hz = estimate_host_hz(config)
    inner = hz["inner_radius"]
    outer = hz["outer_radius"]
    middle = 0.5 * (inner + outer)

    def build_planet(label, a):
        return {
            "name": f"{config['system_name']} {label}",
            "mass_earth": float(mass_earth),
            "radius_earth": float(radius_earth),
            "a": float(a),
            "e": 0.0,
            "inc": 0.0,
            "Omega": 0.0,
            "omega": 0.0,
            "theta": 0.0,
        }

    return {
        "hz_inner": [build_planet("HZ inner", inner)],
        "hz_mid": [build_planet("HZ middle", middle)],
        "hz_outer": [build_planet("HZ outer", outer)],
    }


def finalize_config(config):
    config["dt"] = recommended_dt(config)
    config["output_every_n"] = recommended_output_every_n(config, dt=config["dt"])
    return config


def _outer_sweep_values(base, outer_e_values, outer_i_values_deg, force_full_sweep=False):
    if force_full_sweep:
        return tuple(float(v) for v in outer_e_values), tuple(float(v) for v in outer_i_values_deg)

    source_outer_e = base["source_outer_e"]
    source_outer_i = base["source_outer_i"]

    e_values = (
        tuple(float(v) for v in outer_e_values)
        if _is_missing(source_outer_e)
        else (float(source_outer_e),)
    )
    i_values = (
        tuple(float(v) for v in outer_i_values_deg)
        if _is_missing(source_outer_i)
        else (float(source_outer_i),)
    )
    return e_values, i_values


def generate_habitability_experiments(
    triples_df,
    planets_df,
    detected_systems=DETECTED_HABITABLE_SYSTEMS,
    outer_e_values=DEFAULT_OUTER_E_VALUES,
    outer_i_values_deg=DEFAULT_OUTER_I_VALUES_DEG,
    inner_e=DEFAULT_INNER_E,
    inner_i_deg=DEFAULT_INNER_I_DEG,
    tf=1e4,
    output_dir="output",
):
    """
    Build a sweep of canonicalized configs without editing the source CSV.

    Assumptions:
    - detected habitable systems keep observed planets and always run the full
      outer e/i sweep
    - synthetic-planet systems keep measured outer values when present and only
      sweep the missing outer dimensions
    - inner orbital elements keep source-table values when present and fall back
      to the provided defaults when missing
    - synthetic HZ planets are generated as one-planet scenarios at the inner
      edge, midpoint, and outer edge of a static host HZ estimate
    """
    configs = []

    for _, row in triples_df.iterrows():
        base = base_config_from_row(row, tf=tf, output_dir=output_dir)
        system_name = base["system_name"]

        if system_name in detected_systems:
            scenario_planets = {"observed": detected_planets_for_row(row, planets_df)}
            outer_e_iter, outer_i_iter = _outer_sweep_values(
                base,
                outer_e_values,
                outer_i_values_deg,
                force_full_sweep=True,
            )
        else:
            scenario_planets = synthetic_hz_planets(base)
            outer_e_iter, outer_i_iter = _outer_sweep_values(
                base,
                outer_e_values,
                outer_i_values_deg,
                force_full_sweep=False,
            )

        for scenario_name, planets in scenario_planets.items():
            for outer_e in outer_e_iter:
                for outer_i in outer_i_iter:
                    config = copy.deepcopy(base)
                    config["planets"] = copy.deepcopy(planets)
                    config["planet_scenario"] = scenario_name
                    config["inner_e"] = _source_or_default(base["source_inner_e"], inner_e)
                    config["inner_i"] = _source_or_default(base["source_inner_i"], inner_i_deg)
                    config["inner_Omega"] = _safe_float(config["inner_Omega"], 0.0)
                    config["inner_omega"] = _safe_float(config["inner_omega"], 0.0)
                    config["inner_T0"] = 0.0
                    config["outer_e"] = float(outer_e)
                    config["outer_i"] = float(outer_i)
                    config["outer_Omega"] = _safe_float(config["outer_Omega"], 0.0)
                    config["outer_omega"] = _safe_float(config["outer_omega"], 0.0)
                    config["experiment_name"] = (
                        f"{_slugify(system_name)}_{scenario_name}_"
                        f"ev{outer_e:.1f}_iv{int(outer_i):03d}"
                    )
                    config["output_file"] = f"{output_dir}/{config['experiment_name']}.hdf5"
                    finalize_config(config)
                    configs.append(config)

    return configs
