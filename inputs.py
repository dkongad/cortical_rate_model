"""
External input generators for the firing-rate model.

These functions build the input ``I`` (shape ``(num_units,)`` or
``(n_patterns, num_units)``) fed to
:meth:`~cortical_rate_model.network.RateNetwork.simulate` /
:meth:`~cortical_rate_model.network.RateNetwork.steady_state`, plus one
*time-varying* input (:func:`traveling_wave`) with time on the first axis,
suitable for :meth:`~cortical_rate_model.network.RateNetwork.simulate_sequence`.

Static (constant-in-time) inputs
--------------------------------
* :func:`uniform_input`        -- spatially uniform drive.
* :func:`noisy_background`     -- baseline plus band-pass spatial noise (spontaneous).
* :func:`local_stimulation`    -- a localised disc of drive (optogenetic "spot").
* :func:`bandpass_stimulation` -- a binary band-pass random pattern (opto band-pass).
* :func:`oriented_grating`     -- a smooth sinusoidal stimulus map.

Time-varying inputs
-------------------
* :func:`traveling_wave`       -- a drifting sinusoidal grating (retinal-wave /
                                  moving-grating stimulus), shape ``(n_steps, num_units)``.

For an E/I network, wrap a single-population input with :func:`to_ei_input`.
"""

from __future__ import annotations

import numpy as np

from .connectivity import _random_field, DEFAULT_SEED


def uniform_input(layout, level=1.0):
    """Spatially uniform input of value ``level`` (shape ``(num_units,)``)."""
    return np.full(layout.num_units, float(level), dtype=np.float64)


def noisy_background(layout, noise_level=0.3, n_patterns=1, baseline=1.0,
                     sigma1=0.5, sigma2=None, seed=DEFAULT_SEED):
    """Spontaneous input: a constant baseline plus band-pass spatial noise.

    White noise convolved with a difference-of-Gaussians kernel (to give it a
    spatial scale), z-scored, then added to a uniform baseline.  Ring or sheet.

    Parameters
    ----------
    layout : Layout
    noise_level : float
        Amplitude of the spatial noise relative to the baseline.
    n_patterns : int
        Number of independent input realisations to generate.
    baseline : float
        Uniform baseline drive.
    sigma1, sigma2 : float
        Band-pass kernel scales (``sigma2`` defaults to ``4 * sigma1``).
    seed : int
        Random seed.

    Returns
    -------
    numpy.ndarray
        ``(n_patterns, num_units)`` (or ``(num_units,)`` if ``n_patterns == 1``).
    """
    if sigma2 is None:
        sigma2 = 4.0 * sigma1
    field = _random_field(layout, n_patterns, sigma1, sigma2, seed)
    patterns = baseline + noise_level * field
    return patterns[0] if n_patterns == 1 else patterns


def local_stimulation(layout, center=None, radius=3.0, intensity=1.0,
                      baseline=0.0):
    """Localised disc of input ("optogenetic spot").

    All units within ``radius`` of ``center`` receive ``intensity``; the rest
    receive ``baseline``.  Distances respect the layout's boundary conditions.

    Parameters
    ----------
    layout : Layout
    center : int, tuple or None
        Grid coordinates of the stimulation centre (flat index for a ring,
        ``(x, y)`` for a sheet).  Defaults to the centre of the layout.
    radius : float
        Radius of the stimulated disc.
    intensity : float
        Drive inside the disc.
    baseline : float
        Drive outside the disc.
    """
    coords = layout.coordinates
    if center is None:
        center = np.asarray(layout.shape, dtype=np.float64) / 2.0
    else:
        center = np.atleast_1d(np.asarray(center, dtype=np.float64))
        if center.shape[0] != layout.ndim:
            raise ValueError("center must match the layout dimensionality.")

    d = coords - center[None, :]
    if layout.periodic:
        lengths = np.asarray(layout.shape, dtype=np.float64)
        d = d - lengths * np.round(d / lengths)
    dist = np.sqrt(np.sum(d ** 2, axis=-1))

    inp = np.full(layout.num_units, float(baseline), dtype=np.float64)
    inp[dist <= radius] = intensity
    return inp


def bandpass_stimulation(layout, low_freq=2, high_freq=6, intensity=1.0,
                         threshold_percentile=60.0, seed=DEFAULT_SEED):
    """Binary band-pass random stimulation pattern (sheet only).

    Draws white noise, keeps a band of spatial frequencies, thresholds it into a
    binary pattern and scales it by ``intensity``.

    Parameters
    ----------
    layout : Layout (2D sheet)
    low_freq, high_freq : int
        Frequency band (index range) kept in the Fourier domain.
    intensity : float
        Drive of the active (above-threshold) units.
    threshold_percentile : float
        Percentile used to binarise the pattern.
    seed : int
        Random seed.
    """
    if layout.ndim != 2:
        raise ValueError("bandpass_stimulation requires a 2D sheet.")
    N, M = layout.shape
    rng = np.random.default_rng(seed)
    noise = rng.random((N, M))

    ft = np.fft.fftshift(np.fft.fft2(noise))
    mask = np.zeros((N, M))
    mask[low_freq:high_freq, low_freq:high_freq] = 1
    pattern = np.abs(np.fft.ifft2(ft * mask))

    threshold = np.percentile(pattern, threshold_percentile)
    binary = (pattern >= threshold).astype(np.float64)
    return (intensity * binary).reshape(layout.num_units)


def oriented_grating(layout, wavelength=10.0, orientation=0.0, phase=0.0,
                     amplitude=1.0):
    """Smooth (static) sinusoidal input map -- a simple oriented stimulus.

    ``I(r) = amplitude * cos(k . r + phase)`` with wavevector set by
    ``wavelength`` and ``orientation``.  Works for ring (1D) and sheet (2D).
    """
    coords = layout.coordinates
    k = 2.0 * np.pi / wavelength
    if layout.ndim == 1:
        proj = k * coords[:, 0]
    else:
        proj = k * (np.cos(orientation) * coords[:, 0]
                    + np.sin(orientation) * coords[:, 1])
    return amplitude * np.cos(proj + phase)


def traveling_wave(layout, n_steps=100, wavelength=10.0, orientation=0.0,
                   n_cycles=1.0, amplitude=1.0, offset=0.0, rectify=True):
    """Drifting sinusoidal grating -- a time-varying (moving) stimulus.

    Reproduces the moving-grating / retinal-wave input of the 2D-network project:
    a sinusoidal grating whose phase advances linearly in time so the pattern
    sweeps across the sheet.  The phase completes ``n_cycles`` full turns over the
    ``n_steps`` frames.

    ``I(r, t) = offset + amplitude * cos(k . r - 2*pi*n_cycles*t/n_steps)``

    Parameters
    ----------
    layout : Layout
    n_steps : int
        Number of time frames (first axis of the returned array).
    wavelength : float
        Spatial period of the grating in grid units.
    orientation : float
        Direction of travel / grating orientation in radians (sheet only).
    n_cycles : float
        Number of temporal cycles swept over the whole sequence (sets the wave
        speed).
    amplitude : float
        Grating amplitude.
    offset : float
        Constant baseline added at every unit and time.
    rectify : bool
        If ``True`` clip negative values to zero (a non-negative drive).

    Returns
    -------
    numpy.ndarray
        Shape ``(n_steps, num_units)``, ready for
        :meth:`~cortical_rate_model.network.RateNetwork.simulate_sequence`.
    """
    coords = layout.coordinates
    k = 2.0 * np.pi / wavelength
    if layout.ndim == 1:
        proj = k * coords[:, 0]
    else:
        proj = k * (np.cos(orientation) * coords[:, 0]
                    + np.sin(orientation) * coords[:, 1])

    phases = 2.0 * np.pi * n_cycles * np.arange(n_steps) / n_steps
    wave = offset + amplitude * np.cos(proj[None, :] - phases[:, None])
    if rectify:
        wave = np.clip(wave, 0.0, None)
    return wave.astype(np.float64)


def to_ei_input(exc_input=None, inh_input=None, num_units=None):
    """Assemble single-population inputs into a 2-population E/I input.

    Parameters
    ----------
    exc_input, inh_input : array-like or None
        Input to the excitatory / inhibitory population, each shape
        ``(num_units,)`` (or ``(n_patterns, num_units)``).  A ``None`` population
        receives zeros.
    num_units : int, optional
        Units per population; inferred from the provided inputs when omitted.

    Returns
    -------
    numpy.ndarray
        Concatenated ``(..., 2 * num_units)`` input ``[exc, inh]``.
    """
    if exc_input is None and inh_input is None:
        raise ValueError("Provide at least one of exc_input / inh_input.")
    if num_units is None:
        ref = exc_input if exc_input is not None else inh_input
        num_units = np.asarray(ref).shape[-1]
    if exc_input is None:
        exc_input = np.zeros(num_units)
    if inh_input is None:
        inh_input = np.zeros(num_units)
    return np.concatenate([np.asarray(exc_input, dtype=np.float64),
                           np.asarray(inh_input, dtype=np.float64)], axis=-1)
