# cortical_rate_model

A **PyTorch**-based firing-rate models of
the cortex. It unifies the firing-rate models from previous projects into a single, well-documented library.
---

## 1. Installation

```bash
conda activate myenv           # or any Python >= 3.9 environment
pip install -r requirements.txt
```

The package files live directly in this repository folder (which is itself named
`cortical_rate_model`), so the package is importable whenever its **parent**
directory is on the Python path. Either:

* run Python from the parent directory of this repo, or
* add the parent directory to `PYTHONPATH`
  (`export PYTHONPATH=/path/to/parent:$PYTHONPATH`), or
* `pip install -e .` if you add a packaging file.

Quick check (from the parent directory):

```python
import cortical_rate_model as crm
print(crm.available_presets())
```

---

## 2. The model

Every network integrates the standard rate equation

```
tau * dr/dt = -r + phi(W @ r + I)
```

* `r`   – vector of unit firing rates, length `num_units`
* `W`   – recurrent weight matrix (`W[i, j]` = weight from unit `j` onto unit `i`)
* `I`   – external input
* `phi` – a point-wise nonlinearity
* `tau` – time constant (scalar, or one value per unit)

A **linear** network (`phi = identity`) additionally has a closed-form steady
state `r* = (I − W)⁻¹ I`.

---

## 3. Five-line example

```python
import cortical_rate_model as crm
from cortical_rate_model import Layout, RateNetwork, make_connectivity, inputs

lay  = Layout((30, 30), periodic=True)                         # 2D sheet
W    = make_connectivity("mexican_hat", lay, sigma=1.8, inh_factor=2.5)
W    = crm.scale_spectral_radius(W, 0.95)                      # set the spectral radius
net  = RateNetwork(W, layout=lay, nonlinearity="rectification")
resp = net.steady_state(inputs.local_stimulation(lay, radius=3, intensity=1.0))
```

Or skip the setup and load a **preset**:

```python
net = crm.load_preset("sheet_elongated_mh")
```

Every connectivity parameter has a default, so `make_connectivity(kind, lay)`
works on its own — you only set what you want to change.

---

## 4. All the choices you can make

### 4.1 Layout — `Layout(shape, periodic)`

The topology is inferred from `shape` (never specified twice).

| parameter  | meaning | options |
|------------|---------|---------|
| `shape`    | network size **and** topology | `int` → **1D ring**; `(N, M)` → **2D sheet** |
| `periodic` | boundary conditions | `True` = periodic (wrap-around), `False` = open |

### 4.2 Connectivity — `make_connectivity(kind, layout, **params)`

Single-population kinds:

| `kind` | meaning | works on | key parameters |
|--------|---------|----------|----------------|
| `"mexican_hat"` | homogeneous difference-of-Gaussians (short-range excitation, broad inhibition) | ring & sheet | `sigma`, `inh_factor`, `excitation_amplitude`, `inhibition_amplitude` |
| `"elongated_mexican_hat"` | anisotropic DoG whose width / eccentricity / orientation vary across the sheet (the "salt-and-pepper" model) | sheet | + `eccentricity`, `eccentricity_sd`, `sigma_sd`, `orientation`, `orientation_sd`, `smooth`, `seed` |
| `"grf_mexican_hat"` | homogeneous DoG modulated by a Gaussian random field | ring & sheet | + `grf_strength`, `grf_sigma1`, `grf_sigma2`, `grf_same_pattern`, `normalise`, `seed` |
| `"shuffled_mexican_hat"` | homogeneous DoG with weights partially shuffled (spatial scrambling) | ring & sheet | + `shuffle_type`, `fraction`, `local_radius`, `seed` |
| `"random"` | random Gaussian/uniform (optionally sparse) recurrent network | ring & sheet | `g`, `density`, `gaussian`, `normalise`, `spectral_radius`, `seed` |
| `"modular"` | long-range modular connectivity from low-dimensional band-pass patterns | sheet | `dist_spec`, `mod_high`, `mod_low`, `beta`, `normalise`, `seed` |

Two-population (E/I) kinds — return `(W, meta)`, a `2N×2N` matrix plus an E/I
split dict:

| `kind` | meaning | works on |
|--------|---------|----------|
| `"ei_homogeneous"` | isotropic Gaussian E/I blocks | ring & sheet |
| `"ei_elongated"` | elongated / heterogeneous E/I blocks | sheet |
| `"ei_grf"` | Gaussian-random-field-modulated E/I blocks | ring & sheet |
| `"ei_random"` | random E/I blocks (spectrally normalised) | ring & sheet |

All E/I kinds share: `sigma_ee`, `sigma_ei`, `sigma_ie`, `sigma_ii` (the four
block widths), `w_ee`, `w_ei`, `w_ie`, `w_ii` (block strengths),
`self_inhibition`, `alpha`, plus the profile-specific parameters (elongation /
GRF / random) and `seed`.

Meaning of the recurring connectivity parameters:

* **`sigma`** – width of the excitatory Gaussian (grid units).
* **`inh_factor`** – inhibitory width as a **ratio** to excitation
  (inhibitory sigma = `inh_factor * sigma`).
* **`excitation_amplitude` / `inhibition_amplitude`** – **absolute** amplitudes
  multiplying the unit-integral excitatory / inhibitory Gaussians
  (`inhibition_amplitude = 0` → purely excitatory).
* **`sigma_ee/ei/ie/ii`** – the four Gaussian widths of an E/I network.
* **`w_ee/ei/ie/ii`** – the four E/I block strengths (absolute).
* **`eccentricity`** – how elongated the anisotropic fields are (`0` isotropic,
  →`1` line-like); `*_sd` parameters set the spread of the heterogeneity.
* **`orientation` / `orientation_sd`** – mean and randomness of field orientation.
* **`smooth`** – whether heterogeneity varies smoothly (`True`) or per-unit
  ("salt and pepper", `False`).
* **`grf_strength`, `grf_sigma1`, `grf_sigma2`** – amplitude and band-pass
  spatial scales of the Gaussian-random-field modulation.
* **`shuffle_type`** – `"weight_swap"` (swap local↔non-local weights) or
  `"location"` (permute unit locations); `fraction` sets how much.
* **`g`, `density`, `gaussian`** – gain, connection probability and
  Gaussian-vs-uniform draw of the random network.
* **`seed`** – random seed (default `DEFAULT_SEED = 8473`).

**Spectral radius.** Use `scale_spectral_radius(W, spectral_radius)` to set the
largest real eigenvalue (close to but below 1 = strongly amplifying /
near-critical; ≥ 1 = unstable). Read it back with `spectral_radius(W)`. You can
also post-process any matrix with `shuffle_connectivity(W, layout, ...)`.

### 4.3 Nonlinearity — `nonlinearity=...`

`"linear"`, `"rectification"` (ReLU), `"rectification_threshold"`,
`"shallow_rectification"`, `"rectification_squared"`, `"power_law"` (rectified
`x**n`, SSN-style, default `n=2`), `"clipping"`, `"sigmoid"`, `"sigmoid_v2"`,
`"fermi"`, `"tanh"`, `"softplus"`.
You may also pass **any callable** operating on a `torch.Tensor`
(e.g. `functools.partial(nl_power_law, exponent=2.5)`).
List them with `crm.available_nonlinearities()`.

### 4.4 Integrator — `integrator=...`

`"forward_euler"` (1st order), `"runge_kutta2"` (2nd order),
`"runge_kutta"` (4th order). List with `crm.available_integrators()`.

### 4.5 Network classes

| class | use it for |
|-------|-----------|
| `RateNetwork` | general nonlinear network |
| `LinearRateNetwork` | linear network with **analytic** steady state `(I−W)⁻¹I` and input-covariance transform |
| `EIRateNetwork` | two-population E/I network; per-population `tau_e`, `tau_i`; `split_activity()` |

Common network arguments: `layout`, `dt` (time step), `tau` (scalar or per-unit
vector), `device` (`"cpu"` / `"cuda"`), `dtype`.

### 4.6 Inputs — `cortical_rate_model.inputs`

Static (constant-in-time) inputs:

| function | meaning | key parameters |
|----------|---------|----------------|
| `uniform_input` | spatially uniform drive | `level` |
| `noisy_background` | baseline + band-pass spatial noise (spontaneous) | `noise_level`, `n_patterns`, `baseline`, `sigma1`, `sigma2`, `seed` |
| `local_stimulation` | localised disc of drive (opto "spot") | `center`, `radius`, `intensity`, `baseline` |
| `bandpass_stimulation` | binary band-pass random pattern (opto band-pass) | `low_freq`, `high_freq`, `intensity`, `threshold_percentile`, `seed` |
| `oriented_grating` | smooth static sinusoidal stimulus map | `wavelength`, `orientation`, `phase`, `amplitude` |

Time-varying input (shape `(n_steps, num_units)`, for `simulate_sequence`):

| function | meaning | key parameters |
|----------|---------|----------------|
| `traveling_wave` | drifting sinusoidal grating (moving grating / retinal wave) | `n_steps`, `wavelength`, `orientation`, `n_cycles`, `amplitude`, `offset`, `rectify` |

| helper | meaning | key parameters |
|--------|---------|----------------|
| `to_ei_input` | wrap an input for the E/I network (target E, I, or both) | `exc_input`, `inh_input` |

---

## 5. Running the network

```python
traj  = net.simulate(drive, n_steps=200, record="all")   # (n_steps, num_units)
final = net.simulate(drive, n_steps=200, record="last")  # (num_units,)
r_ss  = net.steady_state(drive)                          # integrate to fixed point
seq_r = net.simulate_sequence(wave_inputs)               # inputs shape (T, num_units)
batch = net.response(input_patterns)                     # (n_patterns, num_units)
```

A moving-wave example:

```python
wave = inputs.traveling_wave(lay, n_steps=120, wavelength=8, n_cycles=2)
traj = net.simulate_sequence(wave, record="all")          # (120, num_units)
```

Inputs and initial states may carry a leading batch axis
(`(n_patterns, num_units)`), so many stimuli can be run at once.

For a linear network, `steady_state` is exact and instant:

```python
lin = crm.load_preset("sheet_linear_mh")
r   = lin.steady_state(drive)                     # r = (I - W)^{-1} @ drive
cov = lin.transform_input_covariance(sigma_in)    # response covariance
```

---

## 6. Getting parameters

```python
net.get_params()      # dict of scalar/string params (nonlinearity, dt, tau, ...)
net.get_weights()     # recurrent matrix as a NumPy array
net.eigenvalues()     # eigenvalues of W
```

---

## 7. Saving and loading

Output locations live in **one** place — `cortical_rate_model/paths.py` — not
scattered through the code. Set the base directory via the
`CORTICAL_RATE_MODEL_HOME` environment variable or a `PathConfig`:

```python
from cortical_rate_model import PathConfig
cfg = PathConfig(base_dir="~/my_runs")     # creates models/ activity/ figures/

net.save("run01", path_config=cfg)                 # -> run01.npz + run01.json
net2 = crm.RateNetwork.load("run01", path_config=cfg)

crm.save_activity(r_ss, "run01_resp", path_config=cfg)
r = crm.load_activity("run01_resp", path_config=cfg)["activity"]
```

Omitting `path_config` uses the default (`~/cortical_rate_model_output`).

---

## 8. Presets

`crm.available_presets()` →
`ring_mexican_hat`, `sheet_homogeneous_mh`, `sheet_elongated_mh`,
`sheet_grf_mh`, `sheet_shuffled_mh`, `sheet_random`, `sheet_modular`,
`sheet_ei_homogeneous`, `sheet_ei_elongated`, `ring_ei_homogeneous`,
`sheet_linear_mh`.

```python
net = crm.load_preset("sheet_random", nonlinearity="tanh")   # override any setting
params = crm.get_preset("sheet_random")                      # inspect / copy the recipe
```

---

## 9. Plotting

```python
from cortical_rate_model import plotting
plotting.plot_pattern(r_ss, net.layout)            # ring → line, sheet → image
plotting.plot_connectivity_field(net, net.layout)  # a unit's effective RF
plotting.plot_connectivity_matrix(net)
plotting.plot_activity_timeseries(traj)
plotting.plot_activity_snapshots(traj, net.layout)
plotting.plot_eigenspectrum(net)
```

---

## 10. Where things live

See **`CODE_ORGANIZATION.md`** for the module layout and guidance on extending
the package.
