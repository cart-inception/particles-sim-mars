"""
ParticleSystem — vectorized NumPy physics for 10,000 sand particles.

Physics summary
---------------
- Mars gravity: 3.72 m/s^2 downward
- Aerodynamic drag: F = -drag_coeff * (v_particle - v_wind)
  Drag coefficient chosen so a particle at rest in still air falls ~terminal velocity
  of a ~0.3 mm sand grain in thin Martian atmosphere (~0.02 kg/m^3 density)
- Ground bounce: particles that hit z=0 lose horizontal momentum and get a tiny
  random upward kick — they then respawn naturally rather than teleporting
- Turbulence: small per-step Gaussian noise on velocity
- Color: tan at ground height, rust-orange higher up, fades slightly above column
"""

import numpy as np
from typing import List
from vortex import Vortex

# Mars surface gravity (m/s^2)
GRAVITY = 3.72

# Drag coefficient (1/s) — tuned so settling time looks right visually
DRAG_COEFF = 1.2

# Terminal velocity (just for reference): v_term = g / drag_coeff ≈ 3.1 m/s

# Turbulence amplitude (m/s per sqrt(s))
TURBULENCE = 0.25

# Ground restitution (0 = fully inelastic, 1 = fully elastic)
GROUND_RESTITUTION = 0.15

# Height at which particle color is fully rust-orange
COLOR_HEIGHT_MAX = 20.0

# Tan (low) to rust-orange (high) — Mars palette
COLOR_LOW  = np.array([0.76, 0.60, 0.42], dtype=np.float32)  # sandy tan
COLOR_HIGH = np.array([0.72, 0.27, 0.08], dtype=np.float32)  # rust orange


class ParticleSystem:
    def __init__(self, n_particles: int, spawn_radius: float = 90.0, rng_seed: int = 42):
        """
        Parameters
        ----------
        n_particles  : total number of sand particles
        spawn_radius : XY radius of the ground spawn area (meters)
        rng_seed     : numpy RNG seed for reproducibility
        """
        self.n = n_particles
        self.spawn_radius = spawn_radius
        self._rng = np.random.default_rng(rng_seed)

        # Core arrays — float32 for GL compatibility
        self.positions   = np.zeros((n_particles, 3), dtype=np.float32)
        self.velocities  = np.zeros((n_particles, 3), dtype=np.float32)
        self.colors      = np.zeros((n_particles, 3), dtype=np.float32)
        # Per-particle size variation (used by renderer for point size)
        self.sizes       = np.ones(n_particles, dtype=np.float32)
        # Per-particle drag — fine particles float much longer than coarse grains
        self._drag_coeffs = np.ones(n_particles, dtype=np.float32)

        self._init_particles()

    # ------------------------------------------------------------------

    def _init_particles(self) -> None:
        """Scatter particles randomly on the ground plane."""
        # Random positions in a disk on z=0
        angles = self._rng.uniform(0, 2 * np.pi, self.n)
        radii  = self._rng.uniform(0, self.spawn_radius, self.n)
        self.positions[:, 0] = (radii * np.cos(angles)).astype(np.float32)
        self.positions[:, 1] = (radii * np.sin(angles)).astype(np.float32)
        self.positions[:, 2] = 0.0

        # Tiny randomised initial velocities (mostly at rest on the ground)
        self.velocities[:] = (self._rng.standard_normal((self.n, 3)) * 0.1).astype(np.float32)
        self.velocities[:, 2] = np.maximum(self.velocities[:, 2], 0.0)

        # Size variation: most particles are small dust, a few are larger grains
        self.sizes[:] = (self._rng.exponential(1.0, self.n) * 1.5 + 0.5).clip(0.5, 5.0).astype(np.float32)

        # Drag scales inversely with size — mirrors the real physics of small grains
        # having a high surface-area-to-mass ratio in Mars' thin atmosphere.
        #   size 0.5 → drag 6.0 → terminal vel ≈ 0.6 m/s  (fine dust, floats)
        #   size 1.0 → drag 3.0 → terminal vel ≈ 1.2 m/s  (medium)
        #   size 3.0 → drag 1.0 → terminal vel ≈ 3.7 m/s  (coarse grain)
        #   size 5.0 → drag 0.6 → terminal vel ≈ 6.2 m/s  (largest grains)
        self._drag_coeffs = (3.0 / self.sizes).astype(np.float32)

        self._update_colors()

    # ------------------------------------------------------------------

    def step(self, dt: float, vortices: List[Vortex], ambient_wind=None) -> None:
        """
        Advance physics by dt seconds.

        Steps:
          1. Accumulate wind from all vortices + optional ambient wind
          2. Compute per-particle drag acceleration
          3. Apply gravity
          4. Add turbulence
          5. Integrate velocity and position
          6. Handle ground collisions
          7. Update colors
        """
        # 1. Wind field — sum contributions from all vortices
        wind = np.zeros((self.n, 3), dtype=np.float32)
        for v in vortices:
            wind += v.wind_at(self.positions)
        # Add background ambient wind if enabled
        if ambient_wind is not None:
            wind += ambient_wind.wind_at(self.positions)

        # 2. Per-particle drag: a_drag = -drag_coeff[i] * (v_particle - v_wind)
        #    drag_coeffs vary by particle size — fine dust floats, coarse grains fall fast
        rel_vel = self.velocities - wind
        accel = -(self._drag_coeffs[:, np.newaxis] * rel_vel)

        # 3. Gravity
        accel[:, 2] -= GRAVITY

        # 4. Turbulence — random Gaussian noise scaled by sqrt(dt)
        noise_scale = TURBULENCE * np.sqrt(dt)
        accel += (self._rng.standard_normal((self.n, 3)) * noise_scale).astype(np.float32)

        # 5. Euler integrate
        self.velocities += (accel * dt).astype(np.float32)
        self.positions  += (self.velocities * dt).astype(np.float32)

        # 6. Ground collision at z = 0
        on_ground = self.positions[:, 2] < 0.0
        if np.any(on_ground):
            self.positions[on_ground, 2] = 0.0
            # Reflect vertical velocity with restitution
            self.velocities[on_ground, 2] = np.abs(
                self.velocities[on_ground, 2]
            ) * GROUND_RESTITUTION
            # Damp horizontal velocity on ground contact (friction)
            self.velocities[on_ground, 0] *= 0.6
            self.velocities[on_ground, 1] *= 0.6

        # 7. Colors
        self._update_colors()

    # ------------------------------------------------------------------

    def _update_colors(self) -> None:
        """Map z-height to a tan → rust-orange gradient."""
        t = (self.positions[:, 2] / COLOR_HEIGHT_MAX).clip(0.0, 1.0)[:, np.newaxis]
        self.colors[:] = COLOR_LOW * (1.0 - t) + COLOR_HIGH * t
