"""
Default (preset) parameter sets.

Each preset is a self-contained recipe -- layout, connectivity and network
settings -- distilled from the configurations used in the three source
projects.  Load one with :func:`load_preset` to get a ready-to-run network, or
inspect :data:`PRESETS` to see / copy the parameters.

Example
-------
>>> from cortical_rate_model import load_preset
>>> net = load_preset("sheet_elongated_mh")
>>> net.num_units
900
"""

from __future__ import annotations

import copy

from .layout import Layout
from .connectivity import make_connectivity, scale_spectral_radius
from .network import RateNetwork, LinearRateNetwork, EIRateNetwork


#: Registry of named presets.  Each entry has the sections:
#: ``layout`` (kwargs for :class:`Layout`), ``connectivity`` (``kind`` plus its
#: kwargs), ``network`` (kwargs for the network class), ``spectral_radius``
#: (target largest eigenvalue, or ``None`` to skip rescaling) and ``model``
#: (which network class: ``"rate"``, ``"linear"`` or ``"ei"``).
PRESETS = {
    # 1D ring, homogeneous Mexican hat --------------------------------------
    "ring_mexican_hat": {
        "description": "1D ring of 100 units, homogeneous Mexican hat.",
        "layout": {"shape": 100, "periodic": True},
        "connectivity": {"kind": "mexican_hat", "sigma": 4.0, "inh_factor": 2.0},
        "network": {"nonlinearity": "rectification",
                    "integrator": "forward_euler", "dt": 0.1, "tau": 1.0},
        "spectral_radius": 0.95,
        "model": "rate",
    },
    # 2D homogeneous Mexican hat --------------------------------------------
    "sheet_homogeneous_mh": {
        "description": "30x30 sheet, homogeneous Mexican hat.",
        "layout": {"shape": (30, 30), "periodic": True},
        "connectivity": {"kind": "mexican_hat", "sigma": 1.8, "inh_factor": 2.5},
        "network": {"nonlinearity": "rectification",
                    "integrator": "forward_euler", "dt": 0.1, "tau": 1.0},
        "spectral_radius": 1.02,
        "model": "rate",
    },
    # 2D elongated (heterogeneous) Mexican hat ------------------------------
    "sheet_elongated_mh": {
        "description": "30x30 sheet, elongated anisotropic Mexican hat "
                       "(eccentricity 0.8), the salt-and-pepper model.",
        "layout": {"shape": (30, 30), "periodic": True},
        "connectivity": {"kind": "elongated_mexican_hat", "sigma": 1.8,
                         "inh_factor": 2.5, "eccentricity": 0.8,
                         "orientation_sd": 1.0, "smooth": False},
        "network": {"nonlinearity": "rectification",
                    "integrator": "forward_euler", "dt": 0.1, "tau": 1.0},
        "spectral_radius": 1.02,
        "model": "rate",
    },
    # 2D Mexican hat with a Gaussian-random-field modulation ----------------
    "sheet_grf_mh": {
        "description": "30x30 sheet, homogeneous Mexican hat modulated by a "
                       "Gaussian random field.",
        "layout": {"shape": (30, 30), "periodic": True},
        "connectivity": {"kind": "grf_mexican_hat", "sigma": 1.8,
                         "inh_factor": 2.0, "grf_strength": 0.4},
        "network": {"nonlinearity": "rectification",
                    "integrator": "forward_euler", "dt": 0.1, "tau": 1.0},
        "spectral_radius": 1.02,
        "model": "rate",
    },
    # 2D shuffled Mexican hat -----------------------------------------------
    "sheet_shuffled_mh": {
        "description": "30x30 sheet, homogeneous Mexican hat with 50% of local "
                       "weights swapped for long-range ones.",
        "layout": {"shape": (30, 30), "periodic": True},
        "connectivity": {"kind": "shuffled_mexican_hat", "sigma": 1.8,
                         "inh_factor": 2.5, "shuffle_type": "weight_swap",
                         "fraction": 0.5, "local_radius": 3.0},
        "network": {"nonlinearity": "rectification",
                    "integrator": "forward_euler", "dt": 0.1, "tau": 1.0},
        "spectral_radius": 0.95,
        "model": "rate",
    },
    # 2D random recurrent network -------------------------------------------
    "sheet_random": {
        "description": "30x30 random (Gaussian) recurrent network.",
        "layout": {"shape": (30, 30), "periodic": True},
        "connectivity": {"kind": "random", "g": 1.0, "density": 1.0,
                         "normalise": "spectral", "spectral_radius": 0.95},
        "network": {"nonlinearity": "tanh", "integrator": "runge_kutta",
                    "dt": 0.1, "tau": 1.0},
        "spectral_radius": None,   # already normalised inside the generator
        "model": "rate",
    },
    # 2D modular long-range network -----------------------------------------
    "sheet_modular": {
        "description": "30x30 sheet, long-range modular connectivity.",
        "layout": {"shape": (30, 30), "periodic": True},
        "connectivity": {"kind": "modular", "dist_spec": 5.0, "mod_high": 5.0,
                         "mod_low": 2.0, "beta": 1.0},
        "network": {"nonlinearity": "rectification",
                    "integrator": "forward_euler", "dt": 0.1, "tau": 1.0},
        "spectral_radius": 0.95,
        "model": "rate",
    },
    # 2D homogeneous E/I network --------------------------------------------
    "sheet_ei_homogeneous": {
        "description": "30x30 two-population homogeneous E/I network.",
        "layout": {"shape": (30, 30), "periodic": True},
        "connectivity": {"kind": "ei_homogeneous", "sigma_ee": 2.0,
                         "sigma_ei": 5.0, "sigma_ie": 2.0, "sigma_ii": 5.0,
                         "normalise": "spectral", "spectral_radius": 0.95},
        "network": {"nonlinearity": "rectification",
                    "integrator": "forward_euler", "dt": 0.1,
                    "tau_e": 1.0, "tau_i": 1.0},
        "spectral_radius": None,
        "model": "ei",
    },
    # 2D elongated E/I network ----------------------------------------------
    "sheet_ei_elongated": {
        "description": "30x30 two-population elongated (heterogeneous) E/I "
                       "network.",
        "layout": {"shape": (30, 30), "periodic": True},
        "connectivity": {"kind": "ei_elongated", "sigma_ee": 2.0,
                         "sigma_ei": 5.0, "sigma_ie": 2.0, "sigma_ii": 5.0,
                         "eccentricity": 0.8, "normalise": "spectral",
                         "spectral_radius": 0.95},
        "network": {"nonlinearity": "rectification",
                    "integrator": "forward_euler", "dt": 0.1,
                    "tau_e": 1.0, "tau_i": 1.0},
        "spectral_radius": None,
        "model": "ei",
    },
    # 1D ring E/I network ---------------------------------------------------
    "ring_ei_homogeneous": {
        "description": "1D ring of 100 units, two-population homogeneous E/I.",
        "layout": {"shape": 100, "periodic": True},
        "connectivity": {"kind": "ei_homogeneous", "sigma_ee": 4.0,
                         "sigma_ei": 10.0, "sigma_ie": 4.0, "sigma_ii": 10.0,
                         "normalise": "spectral", "spectral_radius": 0.95},
        "network": {"nonlinearity": "rectification",
                    "integrator": "forward_euler", "dt": 0.1,
                    "tau_e": 1.0, "tau_i": 1.0},
        "spectral_radius": None,
        "model": "ei",
    },
    # Linear homogeneous Mexican hat (analytic steady state) ----------------
    "sheet_linear_mh": {
        "description": "30x30 sheet, homogeneous Mexican hat, LINEAR network "
                       "with analytic steady state.",
        "layout": {"shape": (30, 30), "periodic": True},
        "connectivity": {"kind": "mexican_hat", "sigma": 1.8, "inh_factor": 2.5},
        "network": {"integrator": "forward_euler", "dt": 0.1, "tau": 1.0},
        "spectral_radius": 0.95,
        "model": "linear",
    },
}

_MODEL_CLASSES = {"rate": RateNetwork, "linear": LinearRateNetwork,
                  "ei": EIRateNetwork}


def available_presets():
    """Return the sorted list of preset names."""
    return sorted(PRESETS)


def get_preset(name):
    """Return a deep copy of the raw preset dictionary ``name``."""
    try:
        return copy.deepcopy(PRESETS[name])
    except KeyError as exc:
        raise ValueError(
            f"Unknown preset '{name}'. Available: {available_presets()}"
        ) from exc


def load_preset(name, device="cpu", **network_overrides):
    """Build and return the network described by preset ``name``.

    Parameters
    ----------
    name : str
        One of :func:`available_presets`.
    device : str
        PyTorch device for the network.
    **network_overrides
        Override any of the preset's ``network`` settings (e.g.
        ``nonlinearity="sigmoid"``, ``dt=0.05``).

    Returns
    -------
    RateNetwork
        A ready-to-simulate network (nonlinear, linear or E/I depending on the
        preset).  ``net.layout`` holds the associated :class:`Layout`.
    """
    spec = get_preset(name)
    layout = Layout(**spec["layout"])

    conn = dict(spec["connectivity"])
    kind = conn.pop("kind")
    built = make_connectivity(kind, layout, **conn)

    net_kwargs = dict(spec["network"])
    net_kwargs.update(network_overrides)
    net_kwargs["device"] = device
    model_cls = _MODEL_CLASSES[spec["model"]]

    if spec["model"] == "ei":
        w_rec, meta = built
        return model_cls(w_rec, meta=meta, layout=layout, **net_kwargs)

    w_rec = built[0] if isinstance(built, tuple) else built
    if spec.get("spectral_radius") is not None:
        w_rec = scale_spectral_radius(w_rec, spec["spectral_radius"])
    return model_cls(w_rec, layout=layout, **net_kwargs)
