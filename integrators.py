"""
Numerical integrators for the firing-rate dynamics (PyTorch).

The rate equation integrated by every network is

    tau * dr/dt = -r + phi(W @ r + I) ,

where ``r`` is the rate vector, ``W`` the recurrent weight matrix, ``I`` the
external input and ``phi`` the nonlinearity.  Each integrator advances the
state ``r`` by one time step ``dt`` given the *derivative function* ``fprime``.

The three schemes reproduce those in ``integration_methods_tf`` of the source
projects:

* ``forward_euler``  -- 1st order explicit Euler.
* ``runge_kutta2``   -- 2nd order (midpoint) Runge-Kutta.
* ``runge_kutta``    -- 4th order (classical) Runge-Kutta.

A derivative function has the signature ``fprime(r, inp) -> dr/dt`` and already
includes the ``1/tau`` factor (see :class:`~cortical_rate_model.network.RateNetwork`).
"""

from __future__ import annotations


def forward_euler(r, inp, dt, fprime):
    """One explicit forward-Euler step."""
    return r + dt * fprime(r, inp)


def runge_kutta2(r, inp, dt, fprime):
    """One 2nd-order (midpoint) Runge-Kutta step."""
    k1 = fprime(r, inp)
    return r + dt * fprime(r + 0.5 * dt * k1, inp)


def runge_kutta(r, inp, dt, fprime):
    """One classical 4th-order Runge-Kutta step."""
    k1 = fprime(r, inp)
    k2 = fprime(r + 0.5 * dt * k1, inp)
    k3 = fprime(r + 0.5 * dt * k2, inp)
    k4 = fprime(r + dt * k3, inp)
    return r + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


#: Mapping from the ``integrator`` string to the implementing step function.
INTEGRATORS = {
    "forward_euler": forward_euler,
    "runge_kutta2": runge_kutta2,
    "runge_kutta": runge_kutta,
}


def available_integrators():
    """Return the sorted list of registered integrator names."""
    return sorted(INTEGRATORS)


def get_integrator(name):
    """Return the integrator step function for ``name``.

    Parameters
    ----------
    name : str or callable
        Either a key of :data:`INTEGRATORS` or a callable with signature
        ``step(r, inp, dt, fprime)`` (returned unchanged).
    """
    if callable(name):
        return name
    try:
        return INTEGRATORS[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown integrator '{name}'. "
            f"Available: {available_integrators()}"
        ) from exc
