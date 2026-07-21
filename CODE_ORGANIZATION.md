# Code organization — working on `cortical_rate_model`

This document is for **developers/maintainers** of the package (how the code is
structured and how to extend it). For *using* the package, see `README.md`.

---

## 1. Design principles

* **Self-contained.** Nothing imports from the three original project folders
  (`2D_network_StructConn-main`, `models_cortical_development-main`,
  `Influence_maping-main`). The package keeps working if they are deleted.
* **PyTorch for the dynamics, NumPy for setup.** Connectivity and inputs are
  built with NumPy/SciPy (readable, no autograd overhead); the time-stepping and
  linear algebra of the *dynamics* use PyTorch (batching + optional GPU).
* **Object-oriented with a shallow hierarchy.** One base class
  (`RateNetwork`) and two focused subclasses. Everything else is plain functions
  grouped by responsibility.
* **Registries, not `if/elif` chains.** Nonlinearities, integrators and
  connectivity kinds are dictionaries mapping a name → a function, each with a
  `get_*` / `available_*` helper. Adding a new option = adding one dict entry.
* **Naming.** `snake_case` functions/variables, `CamelCase` classes, clear
  NumPy-style docstrings — following the `Influence_maping` project's style.

---

## 2. Module map

The package files live flat in the repository folder (itself named
`cortical_rate_model`, which acts as the importable package):

```
cortical_rate_model/          <- repository folder == the package
├── __init__.py        Public API (re-exports the names below).
│
├── layout.py          Layout: ring/sheet layout (topology inferred from shape),
│                      coordinates, pairwise distance() and signed delta().
│
├── nonlinearities.py  phi functions (torch) + NONLINEARITIES registry,
│                      get_nonlinearity(), available_nonlinearities().
│
├── integrators.py     forward_euler / runge_kutta2 / runge_kutta step
│                      functions + INTEGRATORS registry, get_integrator().
│
├── connectivity.py    Weight-matrix generators (mexican_hat,
│                      elongated_mexican_hat, grf_mexican_hat,
│                      shuffled_mexican_hat, random_network, modular_network,
│                      and the ei_* E/I profiles via excitatory_inhibitory)
│                      + CONNECTIVITY_GENERATORS registry, make_connectivity(),
│                      scale_spectral_radius(), spectral_radius(),
│                      shuffle_connectivity(), DEFAULT_SEED.
│
├── network.py         RateNetwork (base), LinearRateNetwork, EIRateNetwork:
│                      the dynamics, steady_state / simulate / response, and
│                      get_params / save / load.
│
├── inputs.py          External-drive generators (uniform_input, noisy_background,
│                      local_stimulation, bandpass_stimulation, oriented_grating,
│                      traveling_wave [time-varying], to_ei_input).
│
├── presets.py         PRESETS dict + load_preset() / get_preset() /
│                      available_presets(): ready-made recipes.
│
├── paths.py           PathConfig + default_paths: THE ONE place that defines
│                      output directories (edit here when moving machines).
│
├── io_utils.py        save_network / load_network / save_activity /
│                      load_activity — all routed through paths.py.
│
├── plotting.py        Matplotlib helpers that understand the Layout.
│
├── requirements.txt   Environment / package versions.
├── README.md          User guide (parameters and their meaning).
└── CODE_ORGANIZATION.md   This file.
```

### Dependency direction (no cycles)

```
layout ─┐
        ├─> connectivity ─┐
nonlinearities ─> network ┤
integrators ────> network │
inputs  ──> (uses layout, connectivity._random_field, DEFAULT_SEED)
presets ──> layout + connectivity + network
io_utils <── network        (network imports io_utils lazily inside methods)
paths   <── io_utils
plotting ──> layout (only)
```

`network.py` imports `io_utils` **lazily** (inside `save`/`load`) to avoid a
circular import, since `io_utils` needs `RateNetwork` to reconstruct a network.

---

## 3. How the dynamics fit together

`RateNetwork.rate_derivative(r, inp)` returns
`(-r + phi(r @ W.T + inp)) / tau`. The chosen integrator step function takes
that derivative and advances one `dt`. `simulate`, `simulate_sequence`,
`steady_state` and `response` are thin loops over `step`. Because every
operation is written for a trailing unit axis, a leading batch axis
(`(n_patterns, num_units)`) works everywhere for free.

`LinearRateNetwork` overrides `steady_state` with the exact `(I − W)⁻¹ I`;
`EIRateNetwork` only customises construction (block matrix, per-population
`tau`) and adds `split_activity`.

---

## 4. Extending the package (recipes)

**Add a nonlinearity.** Write `def nl_myfunc(x): ...` in `nonlinearities.py`
(operate on a `torch.Tensor`) and add `"myfunc": nl_myfunc` to `NONLINEARITIES`.

**Add an integrator.** Write `def my_step(r, inp, dt, fprime): ...` in
`integrators.py` and register it in `INTEGRATORS`.

**Add a connectivity kind.** Write
`def my_conn(layout, **params) -> np.ndarray` in `connectivity.py` (give every
parameter a default; return a square matrix; use `layout.distance()` /
`layout.delta()`), then add it to `CONNECTIVITY_GENERATORS`. It is then usable by
name and in presets.

**Add an input type.** Write a function in `inputs.py` returning a
`(num_units,)` or `(n_patterns, num_units)` NumPy array (or `(n_steps,
num_units)` for a time-varying input).

**Add a preset.** Add an entry to `PRESETS` in `presets.py` (sections:
`layout`, `connectivity`, `network`, `spectral_radius`, `model`).

**Change where files are saved.** Edit `paths.py` (or set
`CORTICAL_RATE_MODEL_HOME`) — nothing else hard-codes a path.

---

## 5. Conventions & gotchas

* **Matrix convention:** `W[i, j]` = weight *from* `j` *onto* `i`. The dynamics
  use `r @ W.T`, so heterogeneity indexed by the *receiving* unit lives in the
  rows of `W`.
* **Reshape order:** `Layout.to_grid` / `to_flat` use C (row-major) order and
  `np.meshgrid(..., indexing="ij")`, so flat index and `(x, y)` grid position
  stay consistent across connectivity, inputs and plotting.
* **Periodic boundaries** are applied through the minimum-image convention in
  `Layout.delta()` / `Layout.distance()`.
* **Sheet-only kinds** (`elongated_mexican_hat`, `ei_elongated`, `modular`,
  `bandpass_stimulation`) raise a clear `ValueError` on a ring layout.
* **Reproducibility:** connectivity/input randomness is controlled by explicit
  `seed` arguments (NumPy `default_rng`), defaulting to `DEFAULT_SEED = 8473`.

---

## 6. Smoke test

A minimal end-to-end check (run in `myenv`):

```python
import cortical_rate_model as crm
for name in crm.available_presets():
    net = crm.load_preset(name)
    drive = crm.inputs.uniform_input(net.layout, 1.0) \
        if net.layout.num_units == net.num_units \
        else crm.inputs.to_ei_input(
            exc_input=crm.inputs.uniform_input(net.layout, 1.0),
            num_units=net.layout.num_units)
    net.steady_state(drive, n_steps=300)
    print(name, "ok")
```
