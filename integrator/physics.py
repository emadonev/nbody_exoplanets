import numpy as np
from . import constants

class Physics(object):
    # ----------
    # energy
    # ----------

    @staticmethod
    def energy(pos, vel, masses, G):
        """Total energy (kinetic + potential) for an array of bodies."""
        active = np.where(masses > 0)[0]
        ke = 0.5 * np.sum(masses[active] * np.sum(vel[active] ** 2, axis=1))
        pe = 0.0
        for ii, i in enumerate(active):
            for j in active[ii + 1:]:
                d = np.linalg.norm(pos[i] - pos[j])
                if d > 0:
                    pe -= G * masses[i] * masses[j] / d
        return ke + pe

    # ----------
    # collisions
    # ----------

    def check_collision(self, particles, i: int, j: int) -> bool:
        if i == j: # if we check the same object, no collision automatically
            return False
        # distance between both objects
        d = np.linalg.norm(particles.pos[i] - particles.pos[j])
        # check if the distance is less than the sum of their radii - they definitely collided
        return d <= (particles.radii[i] + particles.radii[j])
    
    # ------------
    # temperature & habitable zone (Müller & Haghighipour 2014)
    # ------------

    # Kopparapu et al. (2013) effective stellar flux at a HZ boundary
    def _s_eff(self, T_star, boundary='inner'):
        # T_S definition
        T_s = T_star - 5780.0

        # calculate for inner boundary
        if boundary == 'inner':
            S0 = constants.S_eff_sun_inner
            a, b, c, d = constants.a_inner, constants.b_inner, constants.c_inner, constants.d_inner
        
        # calculate for outer boundary
        else:
            S0 = constants.S_eff_sun_outer
            a, b, c, d = constants.a_outer, constants.b_outer, constants.c_outer, constants.d_outer
        # return the polynomial
        return S0 + a*T_s + b*T_s**2 + c*T_s**3 + d*T_s**4

    # Spectral weight W_i for a star at a given HZ boundary (Eq. 1, M&H 2014)
    def _spectral_weight(self, T_star, boundary='inner'):
        S0 = constants.S_eff_sun_inner if boundary == 'inner' else constants.S_eff_sun_outer
        return self._s_eff(T_star, boundary) / S0

    # calculate the luminosity of each star based on its radius and temperature
    def stellar_luminosity_ratio(self, radius_solar, temperature_kelvin):
        radius_solar = np.nan_to_num(np.asarray(radius_solar, dtype=float), nan=0.0)
        temperature_kelvin = np.nan_to_num(np.asarray(temperature_kelvin, dtype=float), nan=0.0)
        radii_m = radius_solar * constants.R_sun
        luminosity_w = 4.0 * np.pi * radii_m**2 * constants.sb * temperature_kelvin**4

        return luminosity_w / constants.L_sun

    def static_habitable_zone(self, radius_solar, temperature_kelvin):
        """
        Approximate a static habitable zone for a host star or co-located host stars.

        For multi-star hosts we collapse the host to a single effective source by
        summing luminosities and using a luminosity-weighted temperature. Only for the purpose of placing a synthetic
        planet in orbit.
        """
        radii = np.atleast_1d(np.asarray(radius_solar, dtype=float))
        temps = np.atleast_1d(np.asarray(temperature_kelvin, dtype=float))
        if radii.shape != temps.shape:
            raise ValueError("radius_solar and temperature_kelvin must have the same shape")

        valid = np.isfinite(radii) & np.isfinite(temps) & (radii > 0.0) & (temps > 0.0)
        if not np.any(valid):
            raise ValueError("host HZ requires at least one finite positive radius/temperature pair")

        radii = radii[valid]
        temps = temps[valid]
        luminosities = np.atleast_1d(self.stellar_luminosity_ratio(radii, temps))
        total_luminosity = float(np.sum(luminosities))
        effective_temperature = float(np.average(temps, weights=luminosities))

        s_eff_inner = float(self._s_eff(effective_temperature, 'inner'))
        s_eff_outer = float(self._s_eff(effective_temperature, 'outer'))

        return {
            'luminosity_ratio': total_luminosity,
            'effective_temperature': effective_temperature,
            'S_eff_inner': s_eff_inner,
            'S_eff_outer': s_eff_outer,
            'inner_radius': float(np.sqrt(total_luminosity / s_eff_inner)),
            'outer_radius': float(np.sqrt(total_luminosity / s_eff_outer)),
        }

    def calculate_temp(self, particles):
        star_idx = particles.star_indices
        planet_idx = particles.planet_indices

        if star_idx.size == 0 or planet_idx.size == 0:
            return particles.temperatures, {}

        L_ratio = np.atleast_1d(
            self.stellar_luminosity_ratio(
                particles.radii[star_idx],
                particles.temperatures[star_idx],
            )
        )

        hz_data = {}

        for k in planet_idx:
            # distances from this planet to every star (AU)
            dists = np.array([np.linalg.norm(particles.pos[k] - particles.pos[s])
                              for s in star_idx])

            # identify host star as the nearest star
            host_local = np.argmin(dists)
            T_host = particles.temperatures[star_idx[host_local]]

            # --- Step 1: spectrally-weighted flux for each HZ boundary ---
            F_sw_inner = 0.0
            F_sw_outer = 0.0
            for Lr, d_s, s in zip(L_ratio, dists, star_idx):
                if d_s == 0:
                    continue
                W_in = self._spectral_weight(particles.temperatures[s], 'inner')
                W_out = self._spectral_weight(particles.temperatures[s], 'outer')
                F_sw_inner += W_in * Lr / d_s**2
                F_sw_outer += W_out * Lr / d_s**2

            # --- Step 2: S_eff boundaries for the host star (Kopparapu) ---
            S_eff_inner = self._s_eff(T_host, 'inner')
            S_eff_outer = self._s_eff(T_host, 'outer')

            # Host-only circular HZ radii in AU, useful for host-centric plots.
            L_host = L_ratio[host_local]
            hz_inner_radius = np.sqrt(L_host / S_eff_inner)
            hz_outer_radius = np.sqrt(L_host / S_eff_outer)

            # --- Step 3: total bolometric flux (solar units) & HZ check ---
            F_bol = float(np.sum(L_ratio[dists > 0] / dists[dists > 0]**2))
            in_hz = (S_eff_outer < F_bol < S_eff_inner)

            # --- Step 4: equilibrium temperature ---
            T_eq = 278.0 * F_bol**0.25
            particles.temperatures[k] = T_eq

            hz_data[k] = {
                'in_hz': in_hz,
                'F_bol': F_bol,
                'F_sw_inner': F_sw_inner,
                'F_sw_outer': F_sw_outer,
                'S_eff_inner': S_eff_inner,
                'S_eff_outer': S_eff_outer,
                'hz_inner_radius': hz_inner_radius,
                'hz_outer_radius': hz_outer_radius,
                'T_eq': T_eq,
            }

        return particles.temperatures, hz_data
