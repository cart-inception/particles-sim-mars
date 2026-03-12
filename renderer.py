"""
OpenGL renderer for the dust devil simulation.

Rendering strategy
------------------
- Ground plane  : a large flat quad textured in reddish-brown Martian regolith colour,
                  with a subtle grid overlay so the camera has visual reference.
- Sand particles: GL_POINTS batch with per-vertex colour and size
                  Additive blending (GL_ONE dest) gives a soft, dusty glow when many
                  particles overlap in the vortex column.
- Sky gradient  : a full-screen skybox quad rendered behind everything else, fading
                  from a deep rust-orange at the horizon to a dark pinkish sky at zenith.
- HUD           : FPS counter and particle count rendered with pygame onto a surface
                  then blitted to screen after the GL swap.

Coordinate system: X east, Y north, Z up (standard right-handed).
"""

import numpy as np
import pygame
from OpenGL.GL import *
from OpenGL.GLU import gluQuadricNormals, gluNewQuadric


# ── colours ─────────────────────────────────────────────────────────────────
GROUND_COLOR      = (0.55, 0.30, 0.12, 1.0)   # dark reddish-brown regolith
GROUND_GRID_COLOR = (0.45, 0.22, 0.08, 1.0)   # darker grid lines
SKY_TOP_COLOR     = (0.18, 0.12, 0.14, 1.0)   # dark pinkish-purple zenith
SKY_HORIZON_COLOR = (0.72, 0.35, 0.10, 1.0)   # rust-orange horizon

# Point size range mapped from per-particle size array
POINT_SIZE_MIN = 1.5
POINT_SIZE_MAX = 6.0

# Ground plane half-extent (matches spawn_radius in simulation)
GROUND_HALF = 120.0
GRID_STEP    = 10.0   # grid lines every N metres


class Renderer:
    def __init__(self, width: int, height: int):
        self.width  = width
        self.height = height
        self._hud_font = None
        self._hud_freetype = False

        self._gl_init()

    # ------------------------------------------------------------------
    # OpenGL state setup
    # ------------------------------------------------------------------

    def _gl_init(self) -> None:
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LEQUAL)

        # Smooth points
        glEnable(GL_POINT_SMOOTH)
        glHint(GL_POINT_SMOOTH_HINT, GL_NICEST)

        # Blending for particles: additive gives a glowing, dusty look
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE)

        glClearColor(*SKY_HORIZON_COLOR)

        # Allow per-vertex point sizes set via glPointSize
        # (GL_VERTEX_PROGRAM_POINT_SIZE is for shaders — not needed here)
        glLineWidth(1.0)

    # ------------------------------------------------------------------
    # Resize
    # ------------------------------------------------------------------

    def resize(self, width: int, height: int) -> None:
        self.width  = width
        self.height = height
        glViewport(0, 0, width, height)

    # ------------------------------------------------------------------
    # Frame begin / end
    # ------------------------------------------------------------------

    def begin_frame(self) -> None:
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self._draw_sky()

    def end_frame(self) -> None:
        pass  # pygame.display.flip() is called from main.py

    # ------------------------------------------------------------------
    # Sky gradient — drawn as a screen-aligned quad behind everything
    # ------------------------------------------------------------------

    def _draw_sky(self) -> None:
        """Render a top-to-bottom sky gradient before the 3-D scene."""
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, 1, 0, 1, -1, 1)

        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        glDisable(GL_DEPTH_TEST)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        glBegin(GL_QUADS)
        # Bottom (horizon)
        glColor4f(*SKY_HORIZON_COLOR)
        glVertex2f(0, 0)
        glVertex2f(1, 0)
        # Top (zenith)
        glColor4f(*SKY_TOP_COLOR)
        glVertex2f(1, 1)
        glVertex2f(0, 1)
        glEnd()

        glEnable(GL_DEPTH_TEST)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE)   # restore additive blend for particles

        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()

    # ------------------------------------------------------------------
    # Ground plane
    # ------------------------------------------------------------------

    def draw_ground(self) -> None:
        """Draw the Martian surface as a flat quad with a subtle grid."""
        H = GROUND_HALF

        # Solid ground fill
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glBegin(GL_QUADS)
        glColor4f(*GROUND_COLOR)
        glVertex3f(-H, -H, -0.01)
        glVertex3f( H, -H, -0.01)
        glVertex3f( H,  H, -0.01)
        glVertex3f(-H,  H, -0.01)
        glEnd()

        # Grid overlay (drawn slightly above ground to prevent z-fighting)
        glBegin(GL_LINES)
        glColor4f(*GROUND_GRID_COLOR)
        x = -H
        while x <= H + 0.01:
            glVertex3f(x, -H, 0.0)
            glVertex3f(x,  H, 0.0)
            x += GRID_STEP
        y = -H
        while y <= H + 0.01:
            glVertex3f(-H, y, 0.0)
            glVertex3f( H, y, 0.0)
            y += GRID_STEP
        glEnd()

        # Restore additive blending for particles drawn after this call
        glBlendFunc(GL_SRC_ALPHA, GL_ONE)

    # ------------------------------------------------------------------
    # Vortex axis markers (optional debug visual)
    # ------------------------------------------------------------------

    def draw_vortex_markers(self, vortices) -> None:
        """Draw a faint vertical line at each vortex column centre."""
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glLineWidth(1.0)
        glBegin(GL_LINES)
        glColor4f(1.0, 0.8, 0.5, 0.25)
        for v in vortices:
            glVertex3f(v.cx, v.cy, 0.0)
            glVertex3f(v.cx, v.cy, v.height)
        glEnd()
        glBlendFunc(GL_SRC_ALPHA, GL_ONE)

    # ------------------------------------------------------------------
    # Particles
    # ------------------------------------------------------------------

    def draw_particles(
        self,
        positions: np.ndarray,
        colors: np.ndarray,
        sizes: np.ndarray,
        camera_radius: float,
    ) -> None:
        """
        Render all particles as GL_POINTS.

        Point size is scaled by the per-particle size array but also shrinks
        slightly with camera distance to give a natural depth feel without
        requiring per-particle distance sorting.

        Parameters
        ----------
        positions     : (N, 3) float32
        colors        : (N, 3) float32  RGB in [0, 1]
        sizes         : (N,)   float32  per-particle base size
        camera_radius : current orbit radius (used to scale point sizes)
        """
        # Scale point rendering so it looks good across zoom levels
        distance_scale = np.clip(80.0 / camera_radius, 0.4, 2.5)

        # Rather than issuing one glPointSize call, sort and batch by size bucket
        # to keep draw calls low while still having variation (5 buckets).
        N = len(positions)
        buckets = 5
        bucket_size = (POINT_SIZE_MAX - POINT_SIZE_MIN) / buckets

        # Precompute scaled sizes → bucket indices
        scaled = sizes * distance_scale
        indices = ((scaled - POINT_SIZE_MIN) / bucket_size).astype(np.int32).clip(0, buckets - 1)

        # Alpha is 0.65 for all particles (additive blend means many = bright)
        glBegin(GL_POINTS)  # fallback: single pass, ignore per-size variation
        # We iterate buckets to allow per-bucket glPointSize calls
        glEnd()

        for b in range(buckets):
            mask = indices == b
            if not np.any(mask):
                continue
            pt_size = POINT_SIZE_MIN + (b + 0.5) * bucket_size
            glPointSize(float(pt_size * distance_scale))

            pos_b   = positions[mask]
            col_b   = colors[mask]

            glBegin(GL_POINTS)
            # Interleave colour + vertex calls in Python loop is slow for 10k;
            # use vertex arrays instead for acceptable performance.
            glEnd()

            # Use client-side vertex arrays (immediate + array combo)
            glEnableClientState(GL_VERTEX_ARRAY)
            glEnableClientState(GL_COLOR_ARRAY)

            # Build (N, 4) RGBA array — alpha 0.8 for all
            alpha_col = np.hstack([
                col_b,
                np.full((len(col_b), 1), 0.80, dtype=np.float32),
            ])

            glColorPointer(4, GL_FLOAT, 0, alpha_col)
            glVertexPointer(3, GL_FLOAT, 0, pos_b)
            glDrawArrays(GL_POINTS, 0, len(pos_b))

            glDisableClientState(GL_VERTEX_ARRAY)
            glDisableClientState(GL_COLOR_ARRAY)

    # ------------------------------------------------------------------
    # HUD overlay (rendered in 2D over the GL frame)
    # ------------------------------------------------------------------

    def init_hud(self) -> None:
        """Call once after pygame.display is set up."""
        try:
            import pygame.freetype as _ft
            _ft.init()
            self._hud_font = _ft.SysFont("monospace", 14)
            self._hud_freetype = True
        except Exception:
            try:
                pygame.font.init()
                self._hud_font = pygame.font.Font(None, 18)  # built-in font
                self._hud_freetype = False
            except Exception:
                self._hud_font = None  # HUD disabled

    def draw_hud(
        self,
        surface: pygame.Surface,
        fps: float,
        n_particles: int,
        n_vortices: int,
    ) -> None:
        """Blit HUD text onto the pygame surface (called after GL swap)."""
        if self._hud_font is None:
            return

        lines = [
            f"FPS: {fps:5.1f}",
            f"Particles: {n_particles:,}",
            f"Vortices:  {n_vortices}",
            "",
            "LMB drag: rotate",
            "RMB drag: pan",
            "Scroll:   zoom",
            "R: reset camera",
            "ESC: quit",
        ]
        y = 8
        for line in lines:
            if not line:
                y += 6
                continue
            if getattr(self, '_hud_freetype', False):
                surf, _ = self._hud_font.render(line, (220, 200, 180))
            else:
                surf = self._hud_font.render(line, True, (220, 200, 180))
            surface.blit(surf, (8, y))
            y += 18
