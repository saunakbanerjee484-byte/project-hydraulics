# 🌊 Open Channel Flow Analysis Suite

An object-oriented Python engine for open-channel hydraulics — uniform
flow, critical flow, specific energy, gradually varied flow (GVF), and
hydraulic jumps — with three front ends (CLI, Tkinter desktop GUI,
Streamlit web app) built on one shared physics core.

---

## Features

- **4 cross-sections**: Rectangular, Triangular, Trapezoidal, Circular — all sharing one interface, so the entire solver stack works identically regardless of shape.
- **Normal & critical depth** solvers (Manning's equation / Froude = 1) with automatic bracketing and a `ConvergenceError` on failure.
- **Specific energy** diagram + alternate-depth solver.
- **Gradually Varied Flow (GVF)** profiles via fixed-step RK4, direction-aware (upstream vs. downstream control), with automatic M1–M3 / S1–S3 curve classification.
- **Hydraulic jump detection** for the two standard mixed-regime cases (steep channel + downstream control; mild channel + upstream sluice gate), using a general momentum-function solver cross-checked against the closed-form Belanger equation.
- **Control structures**: sluice gate and sharp-crested weir boundary-depth calculators.
- **Extension lab**: most-economical trapezoidal section design (SciPy SLSQP) and Manning's-n roughness sensitivity sweeps.
- **Three front ends** sharing the exact same core: CLI, Tkinter desktop dashboard, and a styled Streamlit web app.
- **18+ pytest unit tests** covering geometry, core hydraulics, energy, GVF, and jump modules.

---

## Project structure

```
open_channel_flow/
│── open_channel_flow/            <- core engine (import this from anywhere)
│   ├── __init__.py
│   ├── geometry.py                 4 cross-sections + validation guards
│   ├── core.py                     OpenChannelFlow class (normal/critical depth, Froude, regime)
│   ├── energy.py                   specific energy, alternate depth
│   ├── gvf.py                      RK4 GVF integrator, curve classification
│   ├── jump.py                     momentum function, sequent depth, jump location
│   ├── control_structures.py       sluice gate / weir boundary depths
│   ├── io_utils.py                 station tables, CSV/JSON/Excel export
│   ├── optimize.py                 economical section design, roughness sensitivity
│   └── visualization.py            Matplotlib dashboards
├── main.py                       <- CLI
├── gui_tkinter.py                <- Tkinter desktop GUI
├── app_streamlit.py              <- Streamlit web app
├── tests/
│   └── test_hydraulics.py        <- pytest suite
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Unzip and navigate into the project

Make sure `requirements.txt`, `main.py`, and `app_streamlit.py` are all
in the folder you're standing in — check with:

```bash
dir        # Windows PowerShell
ls         # macOS / Linux
```

If you see another `open_channel_flow` folder inside, `cd` into it
first.

### 2. Install dependencies

```bash
python -m pip install -r requirements.txt
```

> **Windows tip:** if `pip` or `streamlit` aren't recognized as
> commands, it's almost always a PATH issue, not a broken install.
> Always run tools through Python directly and it'll work regardless
> of PATH:
> ```powershell
> python -m pip install -r requirements.txt
> python -m streamlit run app_streamlit.py
> python -m pytest tests/ -v
> ```
> If `python` itself isn't recognized, try `py` instead.

---

## Running it

### CLI (interactive prompts)
```bash
python main.py
```
Exports `gvf_results.csv`, `gvf_summary.json`, `gvf_results.xlsx`, and
`gvf_dashboard.png` into the project folder.

### Desktop GUI (needs a display)
```bash
python gui_tkinter.py
```

### Web app
```bash
python -m streamlit run app_streamlit.py
```
Opens at `http://localhost:8501`. All inputs (channel shape, geometry,
discharge, roughness, boundary condition, reach length) live-update the
charts and station table as you move the sliders.

### Tests
```bash
python -m pytest tests/ -v
```

---

## How the physics is organized

- **One interface, four shapes.** `area()`, `top_width()`,
  `wetted_perimeter()`, `hydraulic_radius()`, `area_moment()` are
  defined identically for every `CrossSection`, so the Manning solver,
  critical-depth solver, GVF integrator, and jump momentum function all
  work unchanged when you swap in a different shape.
- **Validity guards live in the geometry/core layer**, not scattered
  through the app code: negative widths/slopes raise `GeometryError` at
  construction time; non-positive depth raises it on first use; solver
  non-convergence raises `ConvergenceError`.
- **The GVF integrator won't cross critical depth smoothly.** A real
  GVF curve can't pass through y = y꜀ — that's exactly where a
  hydraulic jump belongs instead. `solve_profile()` stops there rather
  than grinding through to a numerically meaningless blow-up, and
  `locate_jump()` + `stitch_uniform_flow()` fill in the rest of the
  reach with the correct uniform-flow segment.

### What hydraulic jump detection does and doesn't cover

`locate_jump()` handles the two standard "mixed regime" textbook cases:

- **Case A** (steep channel, downstream control): a dam/weir forces a
  subcritical depth downstream of a channel whose normal flow is
  naturally supercritical. The jump sits where the backwater curve's
  depth matches the sequent depth of the uniform upstream flow.
- **Case B** (mild channel, upstream control): a sluice gate releases
  supercritical flow into a channel whose normal flow is subcritical.
  The jump sits where the sequent depth of the curve first matches the
  downstream normal depth.

It does **not** currently handle multiple jumps in one reach, jumps
forced by a mid-channel slope break (compound-slope channels), or
submerged/drowned jumps.

---

## Recent fixes

- **Circular closed-form area moment.** `Circular.area_moment()` now
  uses a derived closed form (`R³·(sinφ − φcosφ − sin³φ/3)`, where
  φ = arccos(1 − y/R)) instead of falling back to numerical
  integration — verified against Simpson's-rule integration to 5+
  significant figures and locked in by
  `test_circular_closed_form_area_moment_matches_numerical_integration`.
- **Hydraulic jump detection reliability.** Found and fixed a real bug:
  `solve_profile()`'s adaptive step-refinement only validated the
  *final* RK4 result for a positive depth, not the intermediate
  `k2`/`k3`/`k4` stage evaluations. On larger steps those intermediate
  stages could evaluate the friction slope at a non-physical (negative)
  depth and throw an uncaught `GeometryError`, silently aborting the
  whole GVF solve before `locate_jump()` ever ran — exactly the
  "possible on paper, not found by the app" symptom. `_dydx()` now
  raises a catchable `ConvergenceError` on a non-positive trial depth,
  so the step-halving loop retries with a smaller step instead of
  crashing. Locked in with three new tests: a guaranteed-jump Case A
  scenario, a guaranteed-jump Case B scenario, and a negative-control
  case confirming the solver correctly reports "no jump" when the
  boundary depth is genuinely too low.
- **Sidebar contrast.** The Streamlit sidebar previously forced white
  text on a dark-navy background across every native widget, which
  broke contrast on some elements (e.g. open dropdown text). Replaced
  with a light, high-contrast theme that keeps Streamlit's native
  widget styling intact.

## Known limitations / next steps

- The RK4 step size uses adaptive halving only near critical depth —
  it is not a general adaptive-step (e.g. embedded RK45) integrator.
- Composite/multi-reach channels (different S0 or n along the same
  channel) aren't modeled — each run assumes one uniform reach.
- `locate_jump()` still only covers the two standard single-jump mixed
  regime cases (see below) — no multiple jumps, slope-break-forced
  jumps, or submerged/drowned jumps yet.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Could not open requirements file` | You're one folder above the project root — `cd` into the folder containing `requirements.txt`. |
| `streamlit` / `pytest` "not recognized" (Windows) | Run via `python -m streamlit ...` / `python -m pytest ...` instead of the bare command. |
| `python` "not recognized" (Windows) | Try `py` instead of `python`. |
| Excel export fails | `pip install pandas openpyxl` (already in `requirements.txt`). |
| Tkinter GUI won't launch | Requires a display; won't run in a headless/SSH environment. |
