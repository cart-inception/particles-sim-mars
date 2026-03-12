"""
Rankine vortex model for dust devil simulation.

A Rankine vortex has two regions:
  - Inside the core (r < r_core): solid-body rotation, v_tan = Gamma * r / (2*pi * r_core^2)
  - Outside the core (r >= r_core): irrotational flow, v_tan = Gamma / (2*pi * r)

Each vortex also produces:
  - An updraft (Gaussian bell, strongest on the axis)
  - A weak inward radial inflow to feed the column
  - A slow horizontal drift across the ground plane
"""

import numpy as np


class Vortex:
    def __init__(
        self,
        cx: float,
        cy: float,
        core_radius: float = 4.0,
        circulation: float = 60.0,
        updraft_strength: float = 8.0,
        inflow_strength: float = 2.5,
        influence_radius: float = 30.0,
        height: float = 25.0,
        drift_speed: float = 0.4,
    ):
        """
        Parameters
        ----------
        cx, cy          : horizontal center of the vortex column
        core_radius     : radius of the solid-body rotation core (meters)
        circulation     : Gamma — vortex strength (m^2/s)
        updraft_strength: peak vertical wind speed at the column axis (m/s)
        inflow_strength : peak radial inflow speed (m/s)
        influence_radius: distance at which the vortex wind decays to ~0
        height          : effective height of the vortex column; wind fades above this
        drift_speed     : how fast the vortex wanders across the ground (m/s)
        """
        self.cx = float(cx)
        self.cy = float(cy)
        self.core_radius = float(core_radius)
        self.circulation = float(circulation)
        self.updraft_strength = float(updraft_strength)
        self.inflow_strength = float(inflow_strength)
        self.influence_radius = float(influence_radius)
        self.height = float(height)
        self.drift_speed = float(drift_speed)

        # Random drift direction, changes slowly over time
        angle = np.random.uniform(0, 2 * np.pi)
        self._drift_vel = np.array([np.cos(angle), np.sin(angle)]) * drift_speed
        self._drift_timer = 0.0
        self._drift_change_interval = np.random.uniform(4.0, 10.0)

    # ------------------------------------------------------------------
    # wind_at — vectorized, no Python loops
    # ------------------------------------------------------------------

    def wind_at(self, positions: np.ndarray) -> np.ndarray:
        """
        Compute wind velocity at each particle position.

        Parameters
        ----------
        positions : (N, 3) float32 array of particle world positions

        Returns
        -------
        wind : (N, 3) float32 array of wind velocity vectors (m/s)
        """
        N = len(positions)
        wind = np.zeros((N, 3), dtype=np.float32)

        # Horizontal displacement from vortex column axis
        dx = positions[:, 0] - self.cx   # (N,)
        dy = positions[:, 1] - self.cy   # (N,)
        r = np.sqrt(dx * dx + dy * dy)   # horizontal distance (N,)
        r_safe = np.where(r < 1e-6, 1e-6, r)  # avoid divide-by-zero

        # Height values and normalised height fraction through the column [0..1]
        z = positions[:, 2]
        z_clamped = np.maximum(z, 0.0)
        z_frac = np.clip(z_clamped / self.height, 0.0, 1.0)

        # Wind weakens above the column top
        height_fade = np.exp(-z_clamped / self.height)  # (N,)

        # Horizontal influence envelope: fade beyond influence_radius
        sigma_h = self.influence_radius / 2.5
        horiz_fade = np.exp(-(r * r) / (2.0 * sigma_h * sigma_h))  # (N,)

        # ---- Height-dependent core radius --------------------------------
        # Core widens with altitude, giving the column its flared shape.
        # Near the ground the core is tight; at the top it is ~3.5x wider.
        r_c_eff = self.core_radius * (1.0 + 2.5 * z_frac)  # (N,)

        # ---- Tangential (swirling) velocity — Rankine with expanding core
        inside_core = r < r_c_eff
        v_tan = np.where(
            inside_core,
            self.circulation * r_safe / (2.0 * np.pi * r_c_eff ** 2),
            self.circulation / (2.0 * np.pi * r_safe),
        )
        v_tan *= height_fade * horiz_fade

        wind[:, 0] += (v_tan * (-dy / r_safe)).astype(np.float32)
        wind[:, 1] += (v_tan * (dx / r_safe)).astype(np.float32)

        # ---- Updraft: sigma tied to expanding core ----------------------
        # Column fans outward as it rises rather than staying pencil-thin.
        sigma_up = r_c_eff * 1.5
        v_up = self.updraft_strength * np.exp(-(r * r) / (2.0 * sigma_up * sigma_up))
        v_up *= height_fade
        wind[:, 2] += v_up.astype(np.float32)

        # ---- Outward exhaust near column top ----------------------------
        # As rising air approaches the ceiling it fans outward, creating the
        # characteristic spreading crown / flared top of a dust devil.
        exhaust_fade = np.exp(-((1.0 - z_frac) ** 2) / 0.06)   # peaks at z_frac ≈ 1
        exhaust_fade *= (z_frac > 0.35).astype(np.float32)      # only upper 65 %
        sigma_ex = self.core_radius * 2.5
        exhaust_env = np.exp(-(r * r) / (2.0 * sigma_ex * sigma_ex))
        v_ex = self.updraft_strength * 0.9 * exhaust_fade * exhaust_env
        wind[:, 0] += (v_ex * dx / r_safe).astype(np.float32)
        wind[:, 1] += (v_ex * dy / r_safe).astype(np.float32)

        # ---- Radial inflow: concentrated near the ground ----------------
        # Strong near z=0 so low-lying sand is pulled in; fades rapidly with
        # height so upper-column particles aren't constantly yanked inward.
        sigma_in = self.influence_radius / 2.0
        v_in = self.inflow_strength * np.exp(-(r * r) / (2.0 * sigma_in * sigma_in))
        ground_fade = np.exp(-(z_frac ** 2) / 0.20) * height_fade
        v_in *= ground_fade
        wind[:, 0] -= (v_in * dx / r_safe).astype(np.float32)
        wind[:, 1] -= (v_in * dy / r_safe).astype(np.float32)

        return wind

    # ------------------------------------------------------------------
    # update — drift the vortex center slowly
    # ------------------------------------------------------------------

    def update(self, dt: float, world_radius: float = 80.0) -> None:
        """Move vortex center by its drift velocity. Occasionally change direction."""
        self._drift_timer += dt

        if self._drift_timer >= self._drift_change_interval:
            self._drift_timer = 0.0
            self._drift_change_interval = np.random.uniform(4.0, 10.0)
            # Nudge the drift angle randomly
            angle = np.arctan2(self._drift_vel[1], self._drift_vel[0])
            angle += np.random.uniform(-np.pi / 3, np.pi / 3)
            self._drift_vel = np.array([np.cos(angle), np.sin(angle)]) * self.drift_speed

        self.cx += self._drift_vel[0] * dt
        self.cy += self._drift_vel[1] * dt

        # Soft boundary: steer back toward origin if too far out
        dist = np.sqrt(self.cx ** 2 + self.cy ** 2)
        if dist > world_radius:
            # Reflect drift velocity inward
            self._drift_vel[0] -= 2 * (self._drift_vel[0] * self.cx / dist)
            self._drift_vel[1] -= 2 * (self._drift_vel[1] * self.cy / dist)
            self.cx *= world_radius / dist
            self.cy *= world_radius / dist


# ------------------------------------------------------------------
# Vortex–vortex interactions (Fujiwhara effect)
# ------------------------------------------------------------------

def process_vortex_interactions(vortices: list, dt: float) -> list:
    """
    Apply mutual Fujiwhara orbital drift between co-rotating vortex pairs,
    and merge any pair whose centers fall within each other's combined core.

    The Fujiwhara effect (observed in real tropical cyclones and dust devils):
    two co-rotating vortices orbit their shared center of circulation.
    Angular velocity Ω ≈ (Γ_a + Γ_b) / (4π d²), where d is separation.

    When separation drops below (r_core_a + r_core_b) * 3 the vortices merge:
      - Conservation of circulation:  Γ_merged = Γ_a + Γ_b  (minus ~15% heat loss)
      - New core radius from area conservation: r_c = sqrt(r_a² + r_b²)
      - Strength-weighted average for other parameters

    Parameters
    ----------
    vortices : list of Vortex objects
    dt       : physics timestep (seconds)

    Returns
    -------
    vortices : list with merged vortices removed
    """
    if len(vortices) < 2:
        return vortices

    to_remove = set()

    for i in range(len(vortices)):
        if i in to_remove:
            continue
        for j in range(i + 1, len(vortices)):
            if j in to_remove:
                continue

            va = vortices[i]
            vb = vortices[j]

            dx   = vb.cx - va.cx
            dy   = vb.cy - va.cy
            dist = np.sqrt(dx * dx + dy * dy)

            if dist < 1e-3:
                continue

            total_circ = va.circulation + vb.circulation
            w_a = va.circulation / total_circ
            w_b = vb.circulation / total_circ

            # Circulation-weighted shared centroid
            cx_mid = va.cx * w_a + vb.cx * w_b
            cy_mid = va.cy * w_a + vb.cy * w_b

            # ---- Fujiwhara orbit ----------------------------------------
            # Angular velocity capped to prevent instability at very close range
            omega = np.clip(total_circ / (4.0 * np.pi * dist * dist), 0.0, 0.6)
            angle = omega * dt

            cos_a, sin_a = np.cos(angle), np.sin(angle)

            # Rotate va around shared centroid
            dxa = va.cx - cx_mid
            dya = va.cy - cy_mid
            va.cx = cx_mid + dxa * cos_a - dya * sin_a
            va.cy = cy_mid + dxa * sin_a + dya * cos_a

            # Rotate vb around shared centroid
            dxb = vb.cx - cx_mid
            dyb = vb.cy - cy_mid
            vb.cx = cx_mid + dxb * cos_a - dyb * sin_a
            vb.cy = cy_mid + dxb * sin_a + dyb * cos_a

            # ---- Merge check --------------------------------------------
            merge_threshold = (va.core_radius + vb.core_radius) * 3.0
            if dist < merge_threshold:
                # Move merged vortex to shared centroid
                va.cx = cx_mid
                va.cy = cy_mid

                # Conserve angular momentum (minus ~15% dissipated as heat/turbulence)
                va.circulation      = total_circ * 0.85
                # New core area = sum of areas (conservation of solid-body region)
                va.core_radius      = float(np.sqrt(va.core_radius ** 2 + vb.core_radius ** 2))
                # Strength-weighted blend for the rest, with a slight intensity boost
                va.updraft_strength  = (va.updraft_strength * w_a + vb.updraft_strength * w_b) * 1.25
                va.inflow_strength   = (va.inflow_strength  * w_a + vb.inflow_strength  * w_b) * 1.15
                va.influence_radius  = max(va.influence_radius, vb.influence_radius) * 1.2
                va.height            = max(va.height, vb.height) * 1.1
                # Blend drift velocities so the merged vortex moves naturally
                va._drift_vel        = va._drift_vel * w_a + vb._drift_vel * w_b

                to_remove.add(j)
                print(
                    f"[dust-devil] Vortex merge! "
                    f"Γ={va.circulation:.0f} m²/s  "
                    f"core={va.core_radius:.1f} m  "
                    f"height={va.height:.0f} m"
                )

    return [v for i, v in enumerate(vortices) if i not in to_remove]
