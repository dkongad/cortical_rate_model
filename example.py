"""
Runnable example / demo for the cortical_rate_model package.

Run from the repository root with the project environment active:

    python -m cortical_rate_model.example

It walks through the main workflow -- build a network, drive it, read out the
steady state, and save a summary figure -- for both a 2D sheet and a 1D ring.
Figures are written to the configured figures directory (see paths.py).
"""

from __future__ import annotations

import numpy as np

import cortical_rate_model as crm
from cortical_rate_model import Layout, RateNetwork, inputs, plotting
from cortical_rate_model.paths import PathConfig


def demo_sheet():
    """Heterogeneous Mexican-hat sheet responding to a local stimulation."""
    print("\n=== 2D sheet: heterogeneous Mexican hat ===")
    lay = Layout((40, 40), periodic=True)
    W = crm.make_connectivity("elongated_mexican_hat", lay,
                              sigma=1.8, inh_factor=2.5, eccentricity=0.8,
                              seed=1804)
    W = crm.scale_spectral_radius(W, 1.02)
    net = RateNetwork(W, layout=lay, nonlinearity="rectification",
                      integrator="forward_euler", dt=0.1, tau=1.0)

    drive = inputs.local_stimulation(lay, radius=4, intensity=1.0)
    trajectory = net.simulate(drive, n_steps=300, record="all")
    steady = net.steady_state(drive)
    print(f"  num_units = {net.num_units}, "
          f"spectral radius = {crm.spectral_radius(net.get_weights()):.3f}")
    print(f"  steady-state rate: mean={float(steady.mean()):.3f}, "
          f"max={float(steady.max()):.3f}")
    return net, trajectory, steady


def demo_ring():
    """1D ring forming a localized 'bump' from uniform noisy input."""
    print("\n=== 1D ring: Mexican-hat bump ===")
    net = crm.load_preset("ring_mexican_hat")
    drive = inputs.uniform_input(net.layout, 1.0)
    steady = net.steady_state(drive, n_steps=2000)
    peak = int(np.argmax(steady.numpy()))
    print(f"  num_units = {net.num_units}, activity bump peak at unit {peak}")
    return net, steady


def demo_linear_vs_nonlinear():
    """Show the analytic linear steady state matches iterating the dynamics."""
    print("\n=== linear network: analytic vs iterative steady state ===")
    lay = Layout((20, 20), periodic=True)
    W = crm.scale_spectral_radius(crm.make_connectivity("mexican_hat", lay), 0.9)
    lin = crm.LinearRateNetwork(W, layout=lay)
    nonlin = RateNetwork(W, layout=lay, nonlinearity="linear")
    drive = inputs.uniform_input(lay, 1.0)
    diff = np.max(np.abs(lin.steady_state(drive).numpy()
                         - nonlin.steady_state(drive, n_steps=5000,
                                               tol=1e-10).numpy()))
    print(f"  max |analytic - iterative| = {diff:.2e}")


def main():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        have_mpl = True
    except ImportError:
        have_mpl = False

    net, trajectory, steady = demo_sheet()
    demo_ring()
    demo_linear_vs_nonlinear()

    if have_mpl:
        cfg = PathConfig()
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        plotting.plot_connectivity_field(net, net.layout, ax=axes[0])
        plotting.plot_pattern(steady, net.layout, ax=axes[1],
                              title="steady-state response")
        plotting.plot_activity_timeseries(trajectory, ax=axes[2])
        fig.tight_layout()
        out = cfg.figure_path("example_demo.png")
        fig.savefig(out, dpi=120)
        print(f"\nSaved demo figure to: {out}")
    else:
        print("\n(matplotlib not installed -- skipping the figure)")


if __name__ == "__main__":
    main()
