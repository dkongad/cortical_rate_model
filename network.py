"""
Firing-rate network models (PyTorch).

This module defines the model classes that tie together a recurrent weight
matrix, a nonlinearity and an integrator, and that simulate (or analytically
solve) the rate dynamics

    tau * dr/dt = -r + phi(W @ r + I) .

Class hierarchy (parent -> children)::

    RateNetwork               generic nonlinear rate network
    ├── LinearRateNetwork     phi = identity; adds the analytic steady state
    │                         r* = (I - W)^{-1} I  and input-covariance transform
    └── EIRateNetwork         two-population excitatory/inhibitory network with
                              per-population time constants and E/I splitting

The ``RateNetwork`` API deliberately mirrors the ``BrainNetwork`` of the source
projects (``run`` -> :meth:`simulate`, ``res_Input_mat`` -> :meth:`response`)
but is written in PyTorch and supports both 1D rings and 2D sheets, batched
inputs and GPU execution.
"""

from __future__ import annotations

import numpy as np
import torch

from .layout import Layout
from .nonlinearities import get_nonlinearity
from .integrators import get_integrator
from . import connectivity as _connectivity


class RateNetwork:
    """Generic nonlinear firing-rate network.

    Parameters
    ----------
    w_rec : array-like
        Recurrent weight matrix, shape ``(num_units, num_units)``.
        ``w_rec[i, j]`` is the weight from unit ``j`` onto unit ``i``.
    layout : Layout, optional
        Spatial layout (used only for plotting / reshaping; not required for
        the dynamics).
    nonlinearity : str or callable, optional
        Transfer function ``phi`` (see :mod:`.nonlinearities`).
    integrator : str or callable, optional
        Integration scheme (see :mod:`.integrators`).
    dt : float, optional
        Integration time step.
    tau : float or array-like, optional
        Membrane/adaptation time constant.  A scalar is shared by all units; an
        array of length ``num_units`` gives a per-unit time constant.
    device, dtype :
        PyTorch device and floating dtype.
    """

    def __init__(self, w_rec, layout=None, nonlinearity="rectification",
                 integrator="forward_euler", dt=0.1, tau=1.0,
                 device="cpu", dtype=torch.float32):
        self.device = torch.device(device)
        self.dtype = dtype

        w = torch.as_tensor(np.asarray(w_rec), dtype=dtype, device=self.device)
        if w.ndim != 2 or w.shape[0] != w.shape[1]:
            raise ValueError("w_rec must be a square 2D matrix.")
        self.w_rec = w
        self.num_units = w.shape[0]

        self.layout = layout
        self.nonlinearity_rule = nonlinearity
        self.integrator_name = integrator
        self._nonlinearity = get_nonlinearity(nonlinearity)
        self._integrator = get_integrator(integrator)

        self.dt = float(dt)
        self.tau = self._as_vector(tau, "tau")

    # ------------------------------------------------------------------ #
    # Construction helpers
    # ------------------------------------------------------------------ #
    def _as_vector(self, value, name):
        """Return ``value`` as a tensor broadcastable over the unit axis."""
        arr = np.asarray(value, dtype=np.float64)
        if arr.ndim == 0:
            return torch.as_tensor(float(arr), dtype=self.dtype, device=self.device)
        if arr.shape != (self.num_units,):
            raise ValueError(f"{name} must be a scalar or length-{self.num_units} "
                             f"vector, got shape {arr.shape}.")
        return torch.as_tensor(arr, dtype=self.dtype, device=self.device)

    @classmethod
    def from_connectivity(cls, kind, layout, connectivity_params=None,
                          **network_kwargs):
        """Build a network directly from a connectivity ``kind``.

        Convenience constructor that calls
        :func:`~cortical_rate_model.connectivity.make_connectivity` and forwards
        the resulting weight matrix.  For ``"excitatory_inhibitory"`` prefer
        :class:`EIRateNetwork`.
        """
        connectivity_params = connectivity_params or {}
        w_rec = _connectivity.make_connectivity(kind, layout,
                                                **connectivity_params)
        if isinstance(w_rec, tuple):          # e.g. EI returns (w, meta)
            w_rec = w_rec[0]
        return cls(w_rec, layout=layout, **network_kwargs)

    # ------------------------------------------------------------------ #
    # Parameter access
    # ------------------------------------------------------------------ #
    def get_params(self):
        """Return a dict of the scalar/string model parameters (no weights)."""
        return {
            "num_units": self.num_units,
            "nonlinearity": self.nonlinearity_rule
            if isinstance(self.nonlinearity_rule, str) else "custom",
            "integrator": self.integrator_name
            if isinstance(self.integrator_name, str) else "custom",
            "dt": self.dt,
            "tau": self.tau.detach().cpu().numpy().tolist(),
            "layout": repr(self.layout) if self.layout is not None else None,
            "class": type(self).__name__,
        }

    def get_weights(self):
        """Return the recurrent weight matrix as a NumPy array."""
        return self.w_rec.detach().cpu().numpy()

    def eigenvalues(self):
        """Eigenvalues of the recurrent weight matrix (NumPy, complex)."""
        return np.linalg.eigvals(self.get_weights())

    # ------------------------------------------------------------------ #
    # Dynamics
    # ------------------------------------------------------------------ #
    def _to_tensor(self, x):
        return torch.as_tensor(np.asarray(x), dtype=self.dtype, device=self.device)

    def rate_derivative(self, r, inp):
        """Right-hand side ``dr/dt = (-r + phi(W r + I)) / tau``.

        Works for a single state ``(num_units,)`` or a batch
        ``(batch, num_units)``.
        """
        total_input = torch.matmul(r, self.w_rec.T) + inp
        return (-r + self._nonlinearity(total_input)) / self.tau

    def step(self, r, inp):
        """Advance the state ``r`` by one time step under input ``inp``."""
        return self._integrator(r, inp, self.dt, self.rate_derivative)

    def _initial_state(self, inputs):
        """Zero initial state with the batch shape implied by ``inputs``."""
        return torch.zeros(inputs.shape, dtype=self.dtype, device=self.device)

    @torch.no_grad()
    def simulate(self, inputs, r0=None, n_steps=100, record="all"):
        """Integrate the dynamics under a *constant* input.

        Parameters
        ----------
        inputs : array-like
            Input pattern, shape ``(num_units,)`` or ``(batch, num_units)``.
            Held fixed for every step.
        r0 : array-like, optional
            Initial rate; defaults to zeros.
        n_steps : int
            Number of integration steps.
        record : {"all", "last"}
            Return the full trajectory or only the final state.

        Returns
        -------
        torch.Tensor
            ``(n_steps, *batch, num_units)`` if ``record == "all"`` else
            ``(*batch, num_units)``.
        """
        inputs = self._to_tensor(inputs)
        r = self._initial_state(inputs) if r0 is None else self._to_tensor(r0)

        if record == "all":
            traj = torch.empty((n_steps,) + tuple(r.shape),
                               dtype=self.dtype, device=self.device)
        for t in range(n_steps):
            r = self.step(r, inputs)
            if record == "all":
                traj[t] = r
        return traj if record == "all" else r

    @torch.no_grad()
    def simulate_sequence(self, input_sequence, r0=None, record="all"):
        """Integrate the dynamics under a *time-varying* input.

        Parameters
        ----------
        input_sequence : array-like
            Inputs with time on the first axis: shape
            ``(T, num_units)`` or ``(T, batch, num_units)``.
        r0 : array-like, optional
            Initial rate; defaults to zeros.
        record : {"all", "last"}
            Return the full trajectory or only the final state.
        """
        seq = self._to_tensor(input_sequence)
        n_steps = seq.shape[0]
        r = self._initial_state(seq[0]) if r0 is None else self._to_tensor(r0)

        if record == "all":
            traj = torch.empty((n_steps,) + tuple(r.shape),
                               dtype=self.dtype, device=self.device)
        for t in range(n_steps):
            r = self.step(r, seq[t])
            if record == "all":
                traj[t] = r
        return traj if record == "all" else r

    @torch.no_grad()
    def steady_state(self, inputs, r0=None, n_steps=1000, tol=1e-6,
                     check_every=10):
        """Integrate to the fixed point under a constant input.

        Runs until the relative change per step falls below ``tol`` or until
        ``n_steps`` is reached.

        Returns
        -------
        torch.Tensor
            The (approximate) steady-state rate, shape matching ``inputs``.
        """
        inputs = self._to_tensor(inputs)
        r = self._initial_state(inputs) if r0 is None else self._to_tensor(r0)
        for t in range(n_steps):
            r_new = self.step(r, inputs)
            if t % check_every == 0:
                change = torch.linalg.norm(r_new - r) / (
                    torch.linalg.norm(r_new) + 1e-12)
                if change < tol:
                    return r_new
            r = r_new
        return r

    def response(self, input_patterns, **kwargs):
        """Steady-state response to a batch of input patterns.

        Analogous to ``res_Input_mat`` in the source projects.

        Parameters
        ----------
        input_patterns : array-like
            Shape ``(n_patterns, num_units)``.
        **kwargs
            Passed to :meth:`steady_state`.

        Returns
        -------
        numpy.ndarray
            Steady-state responses, shape ``(n_patterns, num_units)``.
        """
        result = self.steady_state(input_patterns, **kwargs)
        return result.detach().cpu().numpy()

    # ------------------------------------------------------------------ #
    # Persistence (delegates to io_utils to keep paths configurable)
    # ------------------------------------------------------------------ #
    def save(self, name, path_config=None, extra=None):
        """Save this network. See :func:`cortical_rate_model.io_utils.save_network`."""
        from .io_utils import save_network
        return save_network(self, name, path_config=path_config, extra=extra)

    @classmethod
    def load(cls, name, path_config=None, **network_kwargs):
        """Load a network. See :func:`cortical_rate_model.io_utils.load_network`."""
        from .io_utils import load_network
        return load_network(name, path_config=path_config, cls=cls,
                            **network_kwargs)

    def __repr__(self):
        return (f"{type(self).__name__}(num_units={self.num_units}, "
                f"nonlinearity='{self.nonlinearity_rule}', "
                f"integrator='{self.integrator_name}', dt={self.dt})")


class LinearRateNetwork(RateNetwork):
    """Linear firing-rate network with an analytic steady state.

    Forces ``phi`` to be the identity so that the fixed point is available in
    closed form::

        r* = (I - W)^{-1} I .

    This reproduces ``BrainNetworkLinear`` from the 2D network project and is
    the fast, exact alternative to iterating the dynamics.
    """

    def __init__(self, w_rec, layout=None, integrator="forward_euler",
                 dt=0.1, tau=1.0, device="cpu", dtype=torch.float32):
        super().__init__(w_rec, layout=layout, nonlinearity="linear",
                         integrator=integrator, dt=dt, tau=tau,
                         device=device, dtype=dtype)
        # M = (I - W)^{-1}: r* = M @ I  (steady state), computed once.
        identity = torch.eye(self.num_units, dtype=self.dtype, device=self.device)
        self.interaction_matrix = torch.linalg.inv(identity - self.w_rec)

    @torch.no_grad()
    def steady_state(self, inputs, **_ignored):
        """Analytic fixed point ``r* = (I - W)^{-1} I``.

        Accepts inputs of shape ``(num_units,)`` or ``(n_patterns, num_units)``.
        Extra keyword arguments (``r0``, ``tol`` ...) are ignored, so the linear
        network is a drop-in replacement for the nonlinear one.
        """
        inp = self._to_tensor(inputs)
        return torch.matmul(inp, self.interaction_matrix.T)

    @torch.no_grad()
    def transform_input_covariance(self, sigma_in):
        """Map an input covariance to the response covariance.

        ``Sigma_out = M @ Sigma_in @ M^T`` with ``M = (I - W)^{-1}``.
        """
        sigma = self._to_tensor(sigma_in)
        M = self.interaction_matrix
        return (M @ sigma @ M.T).detach().cpu().numpy()


class EIRateNetwork(RateNetwork):
    """Two-population excitatory/inhibitory firing-rate network.

    The recurrent matrix is a ``(2N x 2N)`` block matrix (see
    :func:`~cortical_rate_model.connectivity.excitatory_inhibitory`); the first
    ``N`` units are excitatory and the last ``N`` inhibitory.  Per-population
    time constants can be given via ``tau_e`` / ``tau_i``.

    Parameters
    ----------
    w_rec : array-like
        ``(2N, 2N)`` weight matrix.
    meta : dict, optional
        E/I split metadata as returned by ``excitatory_inhibitory``.  If
        omitted the populations are assumed to be the two equal halves.
    tau_e, tau_i : float
        Excitatory / inhibitory time constants.
    """

    def __init__(self, w_rec, meta=None, layout=None,
                 nonlinearity="rectification", integrator="forward_euler",
                 dt=0.1, tau_e=1.0, tau_i=1.0, device="cpu",
                 dtype=torch.float32):
        num = np.asarray(w_rec).shape[0]
        if meta is None:
            half = num // 2
            meta = {"n_exc": half, "n_inh": num - half,
                    "exc_slice": slice(0, half), "inh_slice": slice(half, num)}
        self.meta = meta

        # Build the per-unit tau vector from the two population constants.
        tau_vec = np.empty(num, dtype=np.float64)
        tau_vec[meta["exc_slice"]] = tau_e
        tau_vec[meta["inh_slice"]] = tau_i

        super().__init__(w_rec, layout=layout, nonlinearity=nonlinearity,
                         integrator=integrator, dt=dt, tau=tau_vec,
                         device=device, dtype=dtype)
        self.tau_e = tau_e
        self.tau_i = tau_i

    @classmethod
    def from_layout(cls, layout, connectivity_params=None, **network_kwargs):
        """Build an E/I network from a layout and connectivity parameters."""
        connectivity_params = connectivity_params or {}
        w_rec, meta = _connectivity.excitatory_inhibitory(
            layout, **connectivity_params)
        return cls(w_rec, meta=meta, layout=layout, **network_kwargs)

    def split_activity(self, activity):
        """Split a ``(..., 2N)`` activity array into ``(exc, inh)`` parts."""
        activity = np.asarray(activity)
        return (activity[..., self.meta["exc_slice"]],
                activity[..., self.meta["inh_slice"]])
