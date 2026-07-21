"""
cortical_rate_model
====================

A small, easy-to-read PyTorch package for firing-rate models of the cortex,
unifying the models from three research projects (2D structured-connectivity
networks, cortical-development models, and influence mapping).

Quick start
-----------
>>> from cortical_rate_model import Layout, RateNetwork, inputs
>>> from cortical_rate_model import make_connectivity, scale_spectral_radius
>>> lay = Layout((30, 30), periodic=True)
>>> W = make_connectivity("mexican_hat", lay, sigma=1.8, inh_factor=2.5)
>>> W = scale_spectral_radius(W, 0.95)
>>> net = RateNetwork(W, layout=lay, nonlinearity="rectification")
>>> drive = inputs.local_stimulation(lay, radius=3, intensity=1.0)
>>> response = net.steady_state(drive)

Or start from a preset:

>>> from cortical_rate_model import load_preset
>>> net = load_preset("sheet_elongated_mh")

Main building blocks
--------------------
* :class:`Layout`                      -- ring (1D) or sheet (2D) layout.
* :mod:`connectivity`                  -- recurrent weight generators.
* :class:`RateNetwork` and subclasses  -- the dynamical models.
* :mod:`inputs`                        -- external drive generators.
* :mod:`presets`                       -- ready-made parameter sets.
* :mod:`plotting`                      -- visualisation helpers.
* :mod:`paths` / :mod:`io_utils`       -- configurable saving and loading.
"""

from __future__ import annotations

from .layout import Layout
from .network import RateNetwork, LinearRateNetwork, EIRateNetwork
from .connectivity import (
    make_connectivity,
    available_connectivities,
    scale_spectral_radius,
    spectral_radius,
    shuffle_connectivity,
    DEFAULT_SEED,
)
from .nonlinearities import get_nonlinearity, available_nonlinearities
from .integrators import get_integrator, available_integrators
from .presets import load_preset, get_preset, available_presets, PRESETS
from .paths import PathConfig, default_paths
from .io_utils import (
    save_network,
    load_network,
    save_activity,
    load_activity,
)

from . import connectivity
from . import inputs
from . import plotting
from . import presets
from . import layout
from . import network
from . import nonlinearities
from . import integrators
from . import io_utils
from . import paths

__version__ = "0.1.0"

__all__ = [
    # core classes
    "Layout",
    "RateNetwork",
    "LinearRateNetwork",
    "EIRateNetwork",
    # connectivity
    "make_connectivity",
    "available_connectivities",
    "scale_spectral_radius",
    "spectral_radius",
    "shuffle_connectivity",
    "DEFAULT_SEED",
    # functions / registries
    "get_nonlinearity",
    "available_nonlinearities",
    "get_integrator",
    "available_integrators",
    # presets
    "load_preset",
    "get_preset",
    "available_presets",
    "PRESETS",
    # io / paths
    "PathConfig",
    "default_paths",
    "save_network",
    "load_network",
    "save_activity",
    "load_activity",
    # submodules
    "connectivity",
    "inputs",
    "plotting",
    "presets",
    "layout",
    "network",
    "nonlinearities",
    "integrators",
    "io_utils",
    "paths",
    "__version__",
]
