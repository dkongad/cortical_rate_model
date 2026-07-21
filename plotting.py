"""
Plotting helpers for visualising connectivity, inputs and activity.

Thin, dependency-light wrappers around Matplotlib that understand the network
:class:`~cortical_rate_model.layout.Layout`, so a flat ``(num_units,)``
vector is automatically reshaped to a ring (line plot) or a sheet (image).

All functions accept an optional ``ax`` and return the Matplotlib ``Axes`` (or
``Figure``) so they compose with user-built figures.  ``torch.Tensor`` inputs
are converted automatically.
"""

from __future__ import annotations

import numpy as np


def _to_numpy(x):
    """Convert a tensor / array-like to a detached NumPy array."""
    if hasattr(x, "detach"):
        x = x.detach().cpu()
    return np.asarray(x)


def _new_ax(ax, **fig_kw):
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(**fig_kw)
    return ax


def plot_pattern(pattern, layout, ax=None, cmap="RdBu_r", symmetric=True,
                 colorbar=True, title=None):
    """Plot a single spatial pattern (input or activity snapshot).

    For a ring the pattern is drawn as a line; for a sheet as an image.

    Parameters
    ----------
    pattern : array-like
        Length ``num_units`` vector.
    layout : Layout
    ax : matplotlib Axes, optional
    cmap : str
        Colormap for the sheet image.
    symmetric : bool
        Centre the colour scale on zero (useful for signed activity).
    colorbar : bool
        Draw a colour bar (sheet only).
    title : str, optional
    """
    pattern = _to_numpy(pattern)
    ax = _new_ax(ax)

    if layout.ndim == 1:
        ax.plot(pattern)
        ax.set_xlabel("unit")
        ax.set_ylabel("value")
    else:
        grid = layout.to_grid(pattern)
        vmax = np.nanmax(np.abs(grid)) if symmetric else None
        vmin = -vmax if symmetric else None
        im = ax.imshow(grid, cmap=cmap, vmin=vmin, vmax=vmax,
                       interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        if colorbar:
            ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if title:
        ax.set_title(title)
    return ax


def plot_connectivity_matrix(network, ax=None, cmap="RdBu_r", title="W_rec"):
    """Plot the full recurrent weight matrix as an image."""
    w = network.get_weights() if hasattr(network, "get_weights") \
        else _to_numpy(network)
    ax = _new_ax(ax)
    vmax = np.nanmax(np.abs(w))
    im = ax.imshow(w, cmap=cmap, vmin=-vmax, vmax=vmax, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xlabel("presynaptic unit")
    ax.set_ylabel("postsynaptic unit")
    if title:
        ax.set_title(title)
    return ax


def plot_connectivity_field(network, layout, unit=None, ax=None,
                            cmap="RdBu_r", title=None):
    """Plot the outgoing/incoming connectivity of a single unit in space.

    Shows row ``unit`` of ``W`` (the weights *onto* that unit) reshaped to the
    layout -- i.e. the effective receptive field of the unit.

    Parameters
    ----------
    network : RateNetwork or array-like
    layout : Layout
    unit : int, optional
        Unit index; defaults to the central unit.
    """
    w = network.get_weights() if hasattr(network, "get_weights") \
        else _to_numpy(network)
    if unit is None:
        unit = layout.num_units // 2
    row = w[unit]
    if title is None:
        title = f"connectivity of unit {unit}"
    return plot_pattern(row, layout, ax=ax, cmap=cmap, title=title)


def plot_activity_timeseries(trajectory, units=None, ax=None, title="activity"):
    """Plot activity traces over time for selected units.

    Parameters
    ----------
    trajectory : array-like
        Shape ``(T, num_units)`` (as returned by ``simulate(record="all")``).
    units : sequence of int, optional
        Which units to plot; defaults to 10 evenly spaced units.
    """
    traj = _to_numpy(trajectory)
    if traj.ndim != 2:
        raise ValueError("trajectory must have shape (T, num_units).")
    T, num = traj.shape
    if units is None:
        units = np.linspace(0, num - 1, min(num, 10)).astype(int)
    ax = _new_ax(ax)
    for u in units:
        ax.plot(traj[:, u], label=f"unit {u}", alpha=0.8)
    ax.set_xlabel("time step")
    ax.set_ylabel("rate")
    if title:
        ax.set_title(title)
    return ax


def plot_activity_snapshots(trajectory, layout, times=None, cmap="RdBu_r",
                            symmetric=True):
    """Plot a row of spatial snapshots of a sheet's activity over time.

    Parameters
    ----------
    trajectory : array-like
        Shape ``(T, num_units)``.
    layout : Layout (2D sheet)
    times : sequence of int, optional
        Time indices to show; defaults to 5 evenly spaced frames.
    """
    import matplotlib.pyplot as plt
    traj = _to_numpy(trajectory)
    if times is None:
        times = np.linspace(0, traj.shape[0] - 1, 5).astype(int)
    fig, axes = plt.subplots(1, len(times), figsize=(3 * len(times), 3))
    if len(times) == 1:
        axes = [axes]
    for ax, t in zip(axes, times):
        plot_pattern(traj[t], layout, ax=ax, cmap=cmap, symmetric=symmetric,
                     colorbar=False, title=f"t = {t}")
    fig.tight_layout()
    return fig


def plot_eigenspectrum(network, ax=None, title="eigenvalue spectrum"):
    """Scatter the eigenvalues of the recurrent matrix in the complex plane."""
    eig = network.eigenvalues() if hasattr(network, "eigenvalues") \
        else np.linalg.eigvals(_to_numpy(network))
    ax = _new_ax(ax)
    ax.scatter(eig.real, eig.imag, s=12, alpha=0.6)
    ax.axvline(1.0, color="r", ls="--", lw=1, label="Re = 1 (instability)")
    ax.axhline(0.0, color="k", lw=0.5)
    ax.set_xlabel("Re")
    ax.set_ylabel("Im")
    ax.legend()
    if title:
        ax.set_title(title)
    return ax
