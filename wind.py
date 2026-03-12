"""
AmbientWind — persistent background wind field with gust variation.

Martian surface winds average 2–7 m/s with episodic gusts exceeding 25 m/s.
This model uses a constant base direction with a sinusoidal gust cycle and a
slow random-walk noise component so the strength varies naturally over time.

The wind is spatially uniform (same vector everywhere) — no spatial gradient.
A future improvement would be a Perlin-noise-based spatial wind field.
"""

import numpy as np


class AmbientWind:
    def __init__(
        self,
        direction_deg: float = 225.0,   # bearing wind blows FROM (225 = SW → pushes NE)
        base_speed: float    = 4.0,      # m/s — average sustained speed
        gust_amplitude: float = 3.0,     # m/s — additional speed at gust peak
        gust_period: float   = 9.0,      # seconds per main gust cycle
    ):
        """
        Parameters
        ----------
        direction_deg  : compass bearing the wind blows FROM (0 = from north)
        base_speed     : average wind speed in m/s
        gust_amplitude : peak additional speed during a gust
        gust_period    : duration of one full gust cycle in seconds
        """
        self.base_speed     = float(base_speed)
        self.gust_amplitude = float(gust_amplitude)
        self.gust_period    = float(gust_period)

        self._time      = 0.0
        self._noise     = 0.0   # slow random component [-1, 1]
        self._noise_vel = 0.0   # rate of change of noise (random walk)

        # Unit vector in the direction the wind blows TOWARD
        az = np.radians(direction_deg + 180.0)   # flip: "from" → "toward"
        self._dir = np.array([np.sin(az), np.cos(az), 0.0], dtype=np.float32)

    # ------------------------------------------------------------------

    def update(self, dt: float) -> None:
        """Advance the gust timer and noise random walk."""
        self._time += dt
        # Random walk on noise value — gives irregular gust timing
        self._noise_vel += np.random.randn() * 0.6 * dt
        self._noise_vel *= 0.94  # dampen to prevent runaway
        self._noise = float(np.clip(self._noise + self._noise_vel * dt, -1.0, 1.0))

    def wind_at(self, positions: np.ndarray) -> np.ndarray:
        """
        Return the current wind velocity vector broadcast to all particle positions.

        Parameters
        ----------
        positions : (N, 3) float32 — particle positions (spatial variation ignored)

        Returns
        -------
        wind : (N, 3) float32 — identical wind vector for every particle
        """
        # Gust = sinusoidal base cycle blended with slow noise
        gust_wave = np.sin(self._time * 2.0 * np.pi / self.gust_period)
        gust      = gust_wave * 0.65 + self._noise * 0.35
        speed     = max(0.0, self.base_speed + self.gust_amplitude * gust)

        wind = np.zeros_like(positions)
        wind[:, 0] = self._dir[0] * speed
        wind[:, 1] = self._dir[1] * speed
        # Small upward component during strong gusts (thermal convection proxy)
        wind[:, 2] = max(0.0, (speed - self.base_speed) * 0.08)
        return wind

    @property
    def current_speed(self) -> float:
        """Instantaneous wind speed (m/s) — useful for HUD display."""
        gust_wave = np.sin(self._time * 2.0 * np.pi / self.gust_period)
        gust      = gust_wave * 0.65 + self._noise * 0.35
        return max(0.0, self.base_speed + self.gust_amplitude * gust)
