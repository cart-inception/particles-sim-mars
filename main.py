"""
Mars Dust Devil Particle Simulator
===================================
Entry point — initialises pygame + OpenGL, wires together the simulation,
vortices, camera and renderer, and runs the main loop.

Controls
--------
LMB drag   : orbit camera
RMB drag   : pan look-at target
Scroll     : zoom
R          : reset camera to default view
P          : pause / unpause physics
V          : toggle vortex axis markers
+/-        : add / remove a vortex (up to MAX_VORTICES)
ESC / Q    : quit
"""

import sys
import numpy as np
import pygame
from pygame.locals import DOUBLEBUF, OPENGL, RESIZABLE
from OpenGL.GL import *

from simulation import ParticleSystem
from vortex     import Vortex
from camera     import OrbitCamera
from renderer   import Renderer


# ── simulation constants ──────────────────────────────────────────────────────
N_PARTICLES    = 20_000
SPAWN_RADIUS   = 90.0
PHYSICS_DT     = 1.0 / 60.0   # fixed physics step (seconds)
MAX_VORTICES   = 6

# ── display ───────────────────────────────────────────────────────────────────
WINDOW_W = 1280
WINDOW_H = 720
TITLE    = "Mars Dust Devil Simulator"
TARGET_FPS = 60

# ── default vortex configurations ────────────────────────────────────────────
# Each entry: (cx, cy, core_radius, circulation, updraft, inflow, influence, height)
INITIAL_VORTICES = [
    dict(cx=-20.0, cy= 15.0, core_radius=4.5, circulation=65.0,
         updraft_strength=9.0,  inflow_strength=3.0, influence_radius=28.0,
         height=28.0, drift_speed=0.5),
    dict(cx= 25.0, cy=-10.0, core_radius=3.5, circulation=50.0,
         updraft_strength=7.5,  inflow_strength=2.5, influence_radius=22.0,
         height=22.0, drift_speed=0.6),
    dict(cx=  5.0, cy= 35.0, core_radius=5.0, circulation=75.0,
         updraft_strength=10.0, inflow_strength=3.5, influence_radius=32.0,
         height=32.0, drift_speed=0.35),
]

# Camera defaults (stored for reset)
CAM_DEFAULTS = dict(azimuth=35.0, elevation=18.0, radius=130.0, target=(0.0, 0.0, 8.0))


# ── helpers ───────────────────────────────────────────────────────────────────

def make_random_vortex() -> Vortex:
    """Spawn a vortex at a random ground position with varied parameters."""
    rng = np.random.default_rng()
    angle = rng.uniform(0, 2 * np.pi)
    r     = rng.uniform(20.0, 60.0)
    return Vortex(
        cx=float(r * np.cos(angle)),
        cy=float(r * np.sin(angle)),
        core_radius=float(rng.uniform(3.0, 6.0)),
        circulation=float(rng.uniform(45.0, 80.0)),
        updraft_strength=float(rng.uniform(6.0, 11.0)),
        inflow_strength=float(rng.uniform(2.0, 4.0)),
        influence_radius=float(rng.uniform(20.0, 35.0)),
        height=float(rng.uniform(20.0, 35.0)),
        drift_speed=float(rng.uniform(0.3, 0.7)),
    )


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── pygame + OpenGL init ──────────────────────────────────────────────────
    pygame.init()
    screen = pygame.display.set_mode(
        (WINDOW_W, WINDOW_H),
        DOUBLEBUF | OPENGL | RESIZABLE,
    )
    pygame.display.set_caption(TITLE)

    # ── subsystems ────────────────────────────────────────────────────────────
    camera   = OrbitCamera(**CAM_DEFAULTS)
    renderer = Renderer(WINDOW_W, WINDOW_H)
    renderer.init_hud()
    camera.set_projection(WINDOW_W, WINDOW_H)

    sim      = ParticleSystem(N_PARTICLES, spawn_radius=SPAWN_RADIUS)
    vortices = [Vortex(**cfg) for cfg in INITIAL_VORTICES]

    # ── state flags ───────────────────────────────────────────────────────────
    paused          = False
    show_markers    = True
    clock           = pygame.time.Clock()
    accum_time      = 0.0   # accumulated time for fixed-step physics

    # ── HUD overlay surface ───────────────────────────────────────────────────
    # We render GL into the display, then create a transparent overlay surface
    # for text and blit it via pygame's 2D blitter.
    hud_surface = pygame.Surface((WINDOW_W, WINDOW_H), pygame.SRCALPHA)

    print(f"[dust-devil] {N_PARTICLES:,} particles | {len(vortices)} vortices | "
          f"{WINDOW_W}x{WINDOW_H}")
    print("[dust-devil] LMB drag=rotate  RMB drag=pan  Scroll=zoom  "
          "R=reset  P=pause  V=markers  +/-=add/rm vortex  ESC=quit")

    # ── main loop ─────────────────────────────────────────────────────────────
    running = True
    while running:
        real_dt = clock.tick(TARGET_FPS) / 1000.0
        fps     = clock.get_fps()

        # ── events ────────────────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                key = event.key
                if key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif key == pygame.K_r:
                    camera.azimuth   = CAM_DEFAULTS["azimuth"]
                    camera.elevation = CAM_DEFAULTS["elevation"]
                    camera.radius    = CAM_DEFAULTS["radius"]
                    camera.target    = list(CAM_DEFAULTS["target"])
                elif key == pygame.K_p:
                    paused = not paused
                    print(f"[dust-devil] {'paused' if paused else 'resumed'}")
                elif key == pygame.K_v:
                    show_markers = not show_markers
                elif key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    if len(vortices) < MAX_VORTICES:
                        vortices.append(make_random_vortex())
                        print(f"[dust-devil] Added vortex → {len(vortices)} total")
                elif key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    if len(vortices) > 1:
                        vortices.pop()
                        print(f"[dust-devil] Removed vortex → {len(vortices)} total")

            elif event.type == pygame.VIDEORESIZE:
                renderer.resize(event.w, event.h)
                camera.set_projection(event.w, event.h)
                hud_surface = pygame.Surface((event.w, event.h), pygame.SRCALPHA)

            camera.handle_event(event)

        # ── fixed-step physics ────────────────────────────────────────────────
        if not paused:
            accum_time = min(accum_time + real_dt, 0.1)  # clamp to avoid spiral
            while accum_time >= PHYSICS_DT:
                for v in vortices:
                    v.update(PHYSICS_DT)
                sim.step(PHYSICS_DT, vortices)
                accum_time -= PHYSICS_DT

        # ── render ────────────────────────────────────────────────────────────
        renderer.begin_frame()

        # Apply camera view matrix
        camera.apply()

        # Ground
        renderer.draw_ground()

        # Optional vortex column markers
        if show_markers:
            renderer.draw_vortex_markers(vortices)

        # Particles
        renderer.draw_particles(
            sim.positions,
            sim.colors,
            sim.sizes,
            camera.radius,
        )

        renderer.end_frame()

        # ── HUD overlay ───────────────────────────────────────────────────────
        # Switch to 2D, blit HUD text, then switch back
        hud_surface.fill((0, 0, 0, 0))
        renderer.draw_hud(hud_surface, fps, N_PARTICLES, len(vortices))

        # Save / restore GL state, blit 2D surface over the GL frame
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, renderer.width, renderer.height, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        glDisable(GL_DEPTH_TEST)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        # Convert pygame surface to GL texture and draw as a full-screen quad
        text_data = pygame.image.tostring(hud_surface, "RGBA", True)
        tex_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex_id)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA,
                     hud_surface.get_width(), hud_surface.get_height(),
                     0, GL_RGBA, GL_UNSIGNED_BYTE, text_data)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        glEnable(GL_TEXTURE_2D)
        w, h = hud_surface.get_size()
        glBegin(GL_QUADS)
        glTexCoord2f(0, 0); glVertex2f(0, 0)
        glTexCoord2f(1, 0); glVertex2f(w, 0)
        glTexCoord2f(1, 1); glVertex2f(w, h)
        glTexCoord2f(0, 1); glVertex2f(0, h)
        glEnd()
        glDisable(GL_TEXTURE_2D)
        glDeleteTextures([tex_id])

        glEnable(GL_DEPTH_TEST)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE)  # restore particle blend

        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()

        pygame.display.flip()

    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()
