"""
Saving and loading of networks and simulated activity.

A network is stored as two files under ``<base>/models/``:

* ``<name>.npz``   -- the recurrent weight matrix and the per-unit time constant;
* ``<name>.json``  -- the human-readable scalar/string parameters
  (nonlinearity, integrator, dt, layout, class, plus any ``extra`` metadata).

The output location is governed entirely by :mod:`.paths`, so nothing here
hard-codes a directory.
"""

from __future__ import annotations

import json

import numpy as np

from .paths import default_paths


def _resolve(path_config):
    """Return the given PathConfig or the package default, creating dirs."""
    cfg = path_config or default_paths
    cfg.make_dirs()
    return cfg


def save_network(network, name, path_config=None, extra=None):
    """Save ``network``'s weights and parameters under ``name``.

    Parameters
    ----------
    network : RateNetwork
        The network to save.
    name : str
        File stem (no extension).
    path_config : PathConfig, optional
        Where to save; defaults to :data:`cortical_rate_model.paths.default_paths`.
    extra : dict, optional
        Additional JSON-serialisable metadata to store alongside the params.

    Returns
    -------
    pathlib.Path
        The ``.npz`` weight-file path.
    """
    cfg = _resolve(path_config)
    stem = cfg.model_path(name)

    weights = network.get_weights()
    tau = network.tau.detach().cpu().numpy()
    np.savez(stem.with_suffix(".npz"), w_rec=weights, tau=tau)

    params = network.get_params()
    if extra:
        params["extra"] = extra
    with open(stem.with_suffix(".json"), "w") as fh:
        json.dump(params, fh, indent=2)
    return stem.with_suffix(".npz")


def load_network(name, path_config=None, cls=None, **network_kwargs):
    """Load a network previously saved with :func:`save_network`.

    Parameters
    ----------
    name : str
        File stem used when saving.
    path_config : PathConfig, optional
        Where to look; defaults to the package default.
    cls : type, optional
        The network class to instantiate (default
        :class:`~cortical_rate_model.network.RateNetwork`).
    **network_kwargs
        Overrides forwarded to the constructor (e.g. ``device``).  Any of
        ``nonlinearity``, ``integrator``, ``dt`` not supplied here are read back
        from the saved JSON.

    Returns
    -------
    RateNetwork
    """
    from .network import RateNetwork
    cls = cls or RateNetwork

    cfg = _resolve(path_config)
    stem = cfg.model_path(name)

    data = np.load(stem.with_suffix(".npz"))
    w_rec = data["w_rec"]
    tau = data["tau"]

    with open(stem.with_suffix(".json")) as fh:
        params = json.load(fh)

    kwargs = {
        "nonlinearity": params.get("nonlinearity", "rectification"),
        "integrator": params.get("integrator", "forward_euler"),
        "dt": params.get("dt", 0.1),
        "tau": tau,
    }
    kwargs.update(network_kwargs)   # explicit overrides win
    # A custom nonlinearity/integrator cannot be restored from a string.
    if kwargs["nonlinearity"] == "custom":
        kwargs["nonlinearity"] = "rectification"
    if kwargs["integrator"] == "custom":
        kwargs["integrator"] = "forward_euler"
    return cls(w_rec, **kwargs)


def save_activity(activity, name, path_config=None, **metadata):
    """Save an activity/response array (``.npz``) with optional metadata.

    Parameters
    ----------
    activity : array-like or torch.Tensor
        The activity to save.
    name : str
        File stem.
    path_config : PathConfig, optional
    **metadata
        Extra arrays saved in the same file.
    """
    cfg = _resolve(path_config)
    activity = np.asarray(activity.detach().cpu()) if hasattr(activity, "detach") \
        else np.asarray(activity)
    path = cfg.activity_path(name).with_suffix(".npz")
    np.savez(path, activity=activity, **metadata)
    return path


def load_activity(name, path_config=None):
    """Load an activity file saved with :func:`save_activity`.

    Returns the ``NpzFile``; access the main array via ``["activity"]``.
    """
    cfg = _resolve(path_config)
    return np.load(cfg.activity_path(name).with_suffix(".npz"))
