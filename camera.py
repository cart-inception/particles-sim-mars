"""
Orbit camera with mouse-driven rotation and scroll-wheel zoom.

Controls
--------
Left mouse button drag  : rotate azimuth / elevation
Scroll wheel            : zoom in / out
Middle mouse button drag: pan the look-at target (optional, if ENABLE_PAN=True)
"""

import math
import numpy as np
import pygame
from OpenGL.GL import *
from OpenGL.GLU import gluLookAt, gluPerspective

ENABLE_PAN = True


class OrbitCamera:
    def __init__(
        self,
        azimuth: float = 30.0,
        elevation: float = 20.0,
        radius: float = 120.0,
        target: tuple = (0.0, 0.0, 8.0),
        fov: float = 60.0,
        near: float = 0.5,
        far: float = 1000.0,
    ):
        """
        Parameters
        ----------
        azimuth   : initial horizontal angle around target (degrees)
        elevation : initial vertical angle above horizontal (degrees)
        radius    : distance from target
        target    : world-space look-at point (x, y, z)
        fov       : vertical field of view (degrees)
        """
        self.azimuth   = float(azimuth)
        self.elevation = float(elevation)
        self.radius    = float(radius)
        self.target    = list(target)
        self.fov       = float(fov)
        self.near      = float(near)
        self.far       = float(far)

        self._prev_mouse  = None
        self._mmb_prev    = None
        self._dragging    = False
        self._panning     = False

    # ------------------------------------------------------------------
    # Input event handling
    # ------------------------------------------------------------------

    def handle_event(self, event: pygame.event.Event) -> None:
        """Feed pygame events into the camera controller."""
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:   # LMB
                self._dragging = True
                self._prev_mouse = pygame.mouse.get_pos()
            elif event.button == 3 and ENABLE_PAN:  # RMB pan
                self._panning = True
                self._mmb_prev = pygame.mouse.get_pos()
            elif event.button == 4:  # scroll up → zoom in
                self.radius = max(10.0, self.radius * 0.92)
            elif event.button == 5:  # scroll down → zoom out
                self.radius = min(800.0, self.radius * 1.08)

        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                self._dragging = False
                self._prev_mouse = None
            elif event.button == 3:
                self._panning = False
                self._mmb_prev = None

        elif event.type == pygame.MOUSEMOTION:
            if self._dragging and self._prev_mouse is not None:
                mx, my = pygame.mouse.get_pos()
                dx = mx - self._prev_mouse[0]
                dy = my - self._prev_mouse[1]
                self.azimuth   += dx * 0.4
                self.elevation -= dy * 0.4
                self.elevation  = max(-89.0, min(89.0, self.elevation))
                self._prev_mouse = (mx, my)

            if self._panning and self._mmb_prev is not None and ENABLE_PAN:
                mx, my = pygame.mouse.get_pos()
                dx = mx - self._mmb_prev[0]
                dy = my - self._mmb_prev[1]
                # Move target in the camera's local horizontal plane
                az_rad = math.radians(self.azimuth)
                pan_speed = self.radius * 0.001
                self.target[0] -= (math.cos(az_rad) * dx + math.sin(az_rad) * dy) * pan_speed
                self.target[1] -= (math.sin(az_rad) * dx - math.cos(az_rad) * dy) * pan_speed
                self._mmb_prev = (mx, my)

    # ------------------------------------------------------------------
    # Projection + view matrix setup
    # ------------------------------------------------------------------

    def set_projection(self, width: int, height: int) -> None:
        """Set up the perspective projection matrix."""
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        aspect = width / max(height, 1)
        gluPerspective(self.fov, aspect, self.near, self.far)
        glMatrixMode(GL_MODELVIEW)

    def apply(self) -> None:
        """Load the view matrix corresponding to current orbit state."""
        az  = math.radians(self.azimuth)
        el  = math.radians(self.elevation)
        # Eye position in spherical coordinates offset from target
        ex = self.target[0] + self.radius * math.cos(el) * math.sin(az)
        ey = self.target[1] + self.radius * math.cos(el) * math.cos(az)
        ez = self.target[2] + self.radius * math.sin(el)

        glLoadIdentity()
        gluLookAt(
            ex, ey, ez,
            self.target[0], self.target[1], self.target[2],
            0.0, 0.0, 1.0,   # Z is up
        )

    @property
    def eye_position(self) -> np.ndarray:
        az = math.radians(self.azimuth)
        el = math.radians(self.elevation)
        return np.array([
            self.target[0] + self.radius * math.cos(el) * math.sin(az),
            self.target[1] + self.radius * math.cos(el) * math.cos(az),
            self.target[2] + self.radius * math.sin(el),
        ], dtype=np.float32)
