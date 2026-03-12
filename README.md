# Mars Dust Devil Particle Simulator

A real-time 3D particle simulation of Martian dust devils, inspired by the surface phenomena observed by NASA rovers and depicted in *The Martian*. 10,000–20,000 sand particles are simulated using a Rankine vortex wind field model with Mars-accurate gravity, aerodynamic drag, ground interaction, and multiple independent vortex columns.

![Simulation screenshot placeholder]

---

## Requirements

- Python **3.10 – 3.13** recommended (3.14 works but disables the HUD overlay due to a pygame font bug)
- pip

---

## Setup

### 1. Clone or download the project

```bash
git clone <your-repo-url>
cd particles-sim
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate it

**Linux / macOS:**
```bash
source .venv/bin/activate
```

**Windows:**
```bat
.venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

> **Tip:** If pygame takes a long time to install (it's compiling from source), use:
> ```bash
> pip install -r requirements.txt --prefer-binary
> ```

### 5. Run the simulation

```bash
python main.py
```

---

## Controls

| Input | Action |
|---|---|
| **Left mouse drag** | Orbit / rotate camera |
| **Right mouse drag** | Pan the look-at target |
| **Scroll wheel** | Zoom in / out |
| **R** | Reset camera to default position |
| **P** | Pause / unpause physics |
| **V** | Toggle vortex column axis markers |
| **+** | Add a new random vortex (max 6) |
| **−** | Remove the most recent vortex |
| **ESC / Q** | Quit |

---

## Customising the Simulation

All of the main tuning knobs are constants at the top of two files: `main.py` and `simulation.py`. No physics knowledge is required — just change a number and rerun.

### Particle count — `main.py`

```python
N_PARTICLES = 20_000
```

| Value | Effect |
|---|---|
| `5_000` | Very fast, sparse — good for testing on low-end hardware |
| `10_000` | Original default — lighter feel |
| `20_000` | Current default — good column density |
| `50_000` | Dense, cinematic — needs a decent GPU/CPU |
| `100_000`+ | Possible but will drop below 60 fps without GPU instancing |

### Spawn area — `main.py`

```python
SPAWN_RADIUS = 90.0   # metres
```

How wide an area particles initially scatter across. Increase this to spread the ground coverage; decrease it to pack particles closer to the vortices.

### Number of dust devils — `main.py`

```python
INITIAL_VORTICES = [
    dict(cx=-20.0, cy=15.0,  core_radius=4.5, circulation=65.0, ...),
    dict(cx= 25.0, cy=-10.0, core_radius=3.5, circulation=50.0, ...),
    dict(cx=  5.0, cy=35.0,  core_radius=5.0, circulation=75.0, ...),
]
```

Add, remove, or reposition vortex entries here. Each entry is a dictionary with these fields:

| Field | Unit | Description |
|---|---|---|
| `cx`, `cy` | metres | Starting position on the ground plane |
| `core_radius` | metres | Radius of the tight-spinning central core |
| `circulation` | m²/s | Vortex strength — higher = faster spin |
| `updraft_strength` | m/s | Peak vertical wind speed at the column axis |
| `inflow_strength` | m/s | How hard the vortex pulls sand off the ground |
| `influence_radius` | metres | How far out the vortex affects particles |
| `height` | metres | Effective height of the column |
| `drift_speed` | m/s | How fast the dust devil wanders across the ground |

**To make a dust devil taller and more violent:**
```python
dict(cx=0.0, cy=0.0, core_radius=5.0, circulation=120.0,
     updraft_strength=14.0, inflow_strength=5.0, influence_radius=40.0,
     height=50.0, drift_speed=0.3)
```

**To make a small, fast, tight twister:**
```python
dict(cx=10.0, cy=0.0, core_radius=2.0, circulation=90.0,
     updraft_strength=7.0, inflow_strength=2.0, influence_radius=15.0,
     height=18.0, drift_speed=1.2)
```

### Physics constants — `simulation.py`

```python
GRAVITY       = 3.72   # m/s²  — Mars (use 9.81 for Earth)
DRAG_COEFF    = 1.2    # 1/s   — aerodynamic drag on each particle
TURBULENCE    = 0.25   # m/s   — random jitter added each frame
GROUND_RESTITUTION = 0.15   # 0=clay, 1=rubber ball
```

| Constant | What changing it does |
|---|---|
| **`GRAVITY`** | Use `9.81` to simulate Earth — particles fall faster and columns are shorter |
| **`DRAG_COEFF`** | Lower (e.g. `0.5`) makes particles feel heavier and harder to lift; higher (e.g. `3.0`) makes them follow the wind almost perfectly |
| **`TURBULENCE`** | Higher values add chaotic jitter — good for a rougher, less perfect look |
| **`GROUND_RESTITUTION`** | Higher values make particles bounce more when they hit the ground |

### Particle colour gradient — `simulation.py`

```python
COLOR_LOW  = np.array([0.76, 0.60, 0.42], dtype=np.float32)  # ground level colour
COLOR_HIGH = np.array([0.72, 0.27, 0.08], dtype=np.float32)  # top of column colour
COLOR_HEIGHT_MAX = 20.0   # height (metres) at which colour is fully COLOR_HIGH
```

Colours are RGB in `[0.0, 1.0]`. For example, for a grey/ash volcanic look:
```python
COLOR_LOW  = np.array([0.55, 0.50, 0.48], dtype=np.float32)
COLOR_HIGH = np.array([0.25, 0.22, 0.20], dtype=np.float32)
```

### Window size — `main.py`

```python
WINDOW_W = 1280
WINDOW_H = 720
```

Change to `1920, 1080` for full-HD or `2560, 1440` for 1440p.

---

## Project Structure

```
particles-sim/
├── main.py          # Entry point — game loop, event handling, simulation wiring
├── simulation.py    # ParticleSystem class — vectorised NumPy physics
├── vortex.py        # Vortex class — Rankine vortex wind field model
├── renderer.py      # PyOpenGL rendering — sky, ground, particles, HUD
├── camera.py        # Orbit camera — mouse input, view/projection matrices
└── requirements.txt
```

### How the physics works (brief)

Each frame, every particle has three forces applied to it:

1. **Drag** — `F = -drag_coeff × (v_particle − v_wind)`. Particles are pulled toward the local wind velocity.
2. **Gravity** — constant downward acceleration (`GRAVITY` m/s²).
3. **Turbulence** — a small random Gaussian kick every frame to break up perfect symmetry.

The wind field at each position is computed by summing the contributions from all active vortices. Each vortex uses a **Rankine model** with height-dependent behaviour:
- **Below mid-column**: tight inflow pulls sand off the ground into the base
- **Core region**: solid-body rotation below the core radius, irrotational `1/r` falloff outside
- **Core expands with height**: the column flares from narrow at the ground to ~3.5× wider at the top
- **Upper column**: outward exhaust force fans particles back out, completing the circulation loop

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'numpy'`** — You're running the system Python instead of the venv. Use `python main.py` only after activating the venv (`source .venv/bin/activate`), or run it directly with `.venv/bin/python main.py`.

**Slow framerate with high particle counts** — Try reducing `N_PARTICLES` or lowering the number of vortices in `INITIAL_VORTICES`. The physics is CPU-bound (NumPy); a future optimisation would be to move the wind accumulation loop to a compute shader.

**Window opens but is black** — Usually an OpenGL driver issue. Make sure your graphics drivers are up to date. On Linux with a virtual machine, try setting `LIBGL_ALWAYS_SOFTWARE=1` before running.

**Particles collapse to a line** — The vortex `updraft_strength` is too high relative to `circulation`. Lower it or raise `circulation` to give particles enough tangential momentum to orbit rather than just rising straight up.
