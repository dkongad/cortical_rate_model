"""
Activation (transfer) functions for the firing-rate model.

Every network applies a point-wise nonlinearity ``phi`` to its total input,

    dr/dt = (-r + phi(W @ r + I)) / tau .

The functions here are PyTorch implementations of the nonlinearities that
appeared across the three source projects (``bn_tools_tf`` etc.).  They are
looked up by name through :func:`get_nonlinearity`, mirroring the
``nonlinearity_rule`` string used in the original ``BrainNetwork``.

All functions accept and return a :class:`torch.Tensor`.
"""

from __future__ import annotations

import torch


# --------------------------------------------------------------------------- #
# Individual nonlinearities (PyTorch)
# --------------------------------------------------------------------------- #
def nl_linear(x):
    """Identity: ``phi(x) = x`` (linear network)."""
    return x


def nl_rectification(x):
    """Threshold-linear / ReLU: ``phi(x) = max(x, 0)``."""
    return torch.clamp(x, min=0.0)


def nl_rectification_threshold(x, threshold=0.1):
    """Rectification with an offset threshold: ``max(x - threshold, 0)``."""
    return torch.clamp(x - threshold, min=0.0)


def nl_shallow_rectification(x, gain=0.1):
    """Rectification with a shallow slope: ``max(gain * x, 0)``."""
    return torch.clamp(gain * x, min=0.0)


def nl_rectification_squared(x):
    """Squared rectification: ``max(x, 0) ** 2`` (expansive power law, n=2)."""
    return torch.clamp(x, min=0.0) ** 2


def nl_power_law(x, exponent=2.0):
    """Rectified power law: ``max(x, 0) ** exponent``.

    The general supralinear (SSN-style) transfer function.  The default
    ``exponent=2`` reproduces :func:`nl_rectification_squared`; pass a callable
    or ``functools.partial`` with another exponent for a different power.
    """
    return torch.clamp(x, min=0.0) ** exponent


def nl_clipping(x):
    """Hard saturation between 0 and 1: ``clip(x, 0, 1)``."""
    return torch.clamp(x, min=0.0, max=1.0)


def nl_sigmoid(x):
    """Logistic sigmoid: ``1 / (1 + exp(-x))``."""
    return torch.sigmoid(x)


def nl_sigmoid_v2(x):
    """Rectified, re-centred sigmoid: ``2 * relu(sigmoid(x) - 0.5)``.

    Zero for ``x <= 0`` and saturating towards 1, as used in
    ``nl_sigmoidcustom_tf`` of the 2D network project.
    """
    return torch.clamp(torch.sigmoid(x) - 0.5, min=0.0) * 2.0


def nl_fermi(x, beta=50.0):
    """Steep (near step) sigmoid: ``sigmoid(beta * x)``."""
    return torch.sigmoid(beta * x)


def nl_tanh(x):
    """Hyperbolic tangent."""
    return torch.tanh(x)


def nl_softplus(x):
    """Smooth rectification: ``log(1 + exp(x))``."""
    return torch.nn.functional.softplus(x)


# --------------------------------------------------------------------------- #
# Registry / lookup
# --------------------------------------------------------------------------- #
#: Mapping from the ``nonlinearity_rule`` string to the implementing function.
NONLINEARITIES = {
    "linear": nl_linear,
    "rectification": nl_rectification,
    "rectification_threshold": nl_rectification_threshold,
    "shallow_rectification": nl_shallow_rectification,
    "rectification_squared": nl_rectification_squared,
    "power_law": nl_power_law,
    "clipping": nl_clipping,
    "sigmoid": nl_sigmoid,
    "sigmoid_v2": nl_sigmoid_v2,
    "fermi": nl_fermi,
    "tanh": nl_tanh,
    "softplus": nl_softplus,
}


def available_nonlinearities():
    """Return the sorted list of registered nonlinearity names."""
    return sorted(NONLINEARITIES)


def get_nonlinearity(rule):
    """Return the nonlinearity callable for ``rule``.

    Parameters
    ----------
    rule : str or callable
        Either a key of :data:`NONLINEARITIES` or an arbitrary callable
        (which is returned unchanged, allowing custom transfer functions).
    """
    if callable(rule):
        return rule
    try:
        return NONLINEARITIES[rule]
    except KeyError as exc:
        raise ValueError(
            f"Unknown nonlinearity '{rule}'. "
            f"Available: {available_nonlinearities()}"
        ) from exc
