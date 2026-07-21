"""
Recurrent connectivity generators for the firing-rate model.

Every generator takes a :class:`~cortical_rate_model.layout.Layout` and a set of
parameters and returns a dense recurrent weight matrix ``W`` where ``W[i, j]`` is
the weight *from* unit ``j`` *onto* unit ``i``.  Single-population generators
return an ``(num_units, num_units)`` array; the excitatory/inhibitory (E/I)
generators return an ``(2*num_units, 2*num_units)`` array together with a
metadata dict describing the E/I split.

Every parameter has a sensible default (taken from the presets), so any kind can
be built with just a layout.  Randomness is controlled by an explicit ``seed``
(default :data:`DEFAULT_SEED`).

Connectivity kinds (``kind`` argument of :func:`make_connectivity`)
------------------------------------------------------------------
Single population:

* ``"mexican_hat"``           -- homogeneous difference-of-Gaussians (DoG).
* ``"elongated_mexican_hat"`` -- DoG whose width, eccentricity and orientation
                                 vary smoothly across the sheet (anisotropic /
                                 elongated fields). **Sheet only.**
* ``"grf_mexican_hat"``       -- homogeneous DoG modulated by a Gaussian random
                                 field (GRF).  Ring or sheet.
* ``"shuffled_mexican_hat"``  -- a homogeneous DoG whose weights are partially
                                 shuffled (spatial scrambling).  Ring or sheet.
* ``"random"``                -- random (Gaussian/uniform, optionally sparse)
                                 recurrent network.  Ring or sheet.
* ``"modular"``               -- long-range modular connectivity.  **Sheet only.**

Two populations (return ``(W, meta)``):

* ``"ei_homogeneous"`` -- Gaussian E/I blocks.  Ring or sheet.
* ``"ei_elongated"``   -- elongated (heterogeneous) E/I blocks.  **Sheet only.**
* ``"ei_grf"``         -- GRF-modulated E/I blocks.  Ring or sheet.
* ``"ei_random"``      -- random E/I blocks.  Ring or sheet.

Shared parameter conventions
----------------------------
* ``sigma``               -- width of the excitatory Gaussian (grid units).
* ``inh_factor``          -- inhibitory width as a *ratio* to the excitatory
                             width; inhibitory sigma = ``inh_factor * sigma``.
* ``excitation_amplitude`` / ``inhibition_amplitude`` -- **absolute** amplitudes
                             multiplying the (unit-integral) excitatory and
                             inhibitory Gaussians.
* ``sigma_ee/ei/ie/ii``   -- the four Gaussian widths of an E/I network.
* ``w_ee/ei/ie/ii``       -- the four E/I block strengths (absolute).
* ``grf_strength``, ``grf_sigma1``, ``grf_sigma2`` -- amplitude and band-pass
                             scales of the Gaussian-random-field modulation.
"""

from __future__ import annotations

import numpy as np
from scipy import linalg
from scipy.ndimage import gaussian_filter

#: Default random seed used throughout the package (chosen for fun).
DEFAULT_SEED = 8473


# --------------------------------------------------------------------------- #
# Low-level helpers
# --------------------------------------------------------------------------- #
def _normalised_gaussian(sq_dist, sigma, ndim):
    """Normalised isotropic Gaussian of a squared-distance array.

    Returns ``(2*pi*sigma**2)**(-ndim/2) * exp(-sq_dist / (2*sigma**2))`` so the
    kernel integrates to one in ``ndim`` dimensions.
    """
    norm = (2.0 * np.pi * sigma ** 2) ** (-ndim / 2.0)
    return norm * np.exp(-sq_dist / (2.0 * sigma ** 2))


def _isotropic_gaussian(layout, sigma):
    """Normalised isotropic Gaussian connectivity (of the pairwise distance)."""
    return _normalised_gaussian(layout.distance() ** 2, sigma, layout.ndim)


def spectral_radius(w_rec):
    """Return the largest real part of the eigenvalues of ``w_rec``."""
    return float(np.nanmax(np.real(linalg.eigvals(w_rec))))


def scale_spectral_radius(w_rec, radius=0.95):
    """Rescale ``w_rec`` so its largest real eigenvalue equals ``radius``.

    This is the standard way to set the recurrent strength of the network.
    ``radius`` close to (but below) 1 gives a strongly amplifying, near-critical
    network; ``radius >= 1`` makes the linearised dynamics unstable.

    Parameters
    ----------
    w_rec : numpy.ndarray
        Square recurrent weight matrix.
    radius : float
        Desired maximum real part of the eigenvalue spectrum.
    """
    max_real = spectral_radius(w_rec)
    if np.isclose(max_real, 0.0):
        raise ValueError("Largest real eigenvalue is ~0; cannot rescale.")
    return radius * w_rec / max_real


def _random_field(layout, n_fields, sigma1, sigma2, seed):
    """Gaussian random fields with a spatial scale (band-pass white noise).

    Draw white noise and convolve it with a difference-of-Gaussians kernel to
    impose a spatial scale, then z-score each field.  Works for both a 1D ring
    and a 2D sheet.  Returns an array of shape ``(n_fields, num_units)``.
    """
    rng = np.random.default_rng(seed)

    if layout.ndim == 1:
        N = layout.shape[0]
        noise = rng.normal(size=(n_fields, N))
        if np.isclose(sigma1, 0.0):
            field = noise.copy()
        else:
            x = np.linspace(-N // 2 + 1, N // 2, N)
            kern1 = np.exp(-x ** 2 / (2.0 * sigma1 ** 2)) / (np.sqrt(2 * np.pi) * sigma1)
            kern2 = np.exp(-x ** 2 / (2.0 * sigma2 ** 2)) / (np.sqrt(2 * np.pi) * sigma2)
            kernel_ft = np.fft.fft(kern1 - kern2)
            field = np.real(np.fft.ifft(kernel_ft[None] * np.fft.fft(noise, axis=1),
                                        axis=1))
        field -= field.mean(axis=1, keepdims=True)
        field /= field.std(axis=1, keepdims=True)
        return field

    N, M = layout.shape
    noise = rng.normal(size=(n_fields, N, M))
    if np.isclose(sigma1, 0.0):
        field = noise.copy()
    else:
        x, y = np.meshgrid(np.linspace(-N // 2 + 1, N // 2, N),
                           np.linspace(-M // 2 + 1, M // 2, M), indexing="ij")
        kern1 = np.exp(-(x ** 2 + y ** 2) / (2.0 * sigma1 ** 2))
        kern1 /= (np.sqrt(2 * np.pi) * sigma1) ** 2
        kern2 = np.exp(-(x ** 2 + y ** 2) / (2.0 * sigma2 ** 2))
        kern2 /= (np.sqrt(2 * np.pi) * sigma2) ** 2
        kernel_ft = np.fft.fft2(kern1 - kern2)
        field = np.real(np.fft.ifft2(kernel_ft[None] * np.fft.fft2(noise, axes=(1, 2)),
                                     axes=(1, 2)))
    field -= field.mean(axis=(1, 2), keepdims=True)
    field /= field.std(axis=(1, 2), keepdims=True)
    return field.reshape(n_fields, layout.num_units)


def _elongation_maps(layout, eccentricity, eccentricity_sd, sigma_sd,
                     orientation, orientation_sd, smooth, smooth_sigma1,
                     smooth_sigma2, seed):
    """Per-unit heterogeneity maps for elongated (anisotropic) connectivity.

    Returns three length-``num_units`` arrays:

    * ``sigma_scale`` -- multiplicative width factor around 1 (fractional jitter);
    * ``ecc``         -- per-unit eccentricity, clipped to ``[0, 0.95]``;
    * ``theta``       -- per-unit orientation (radians).

    Sheet only.  Shared by ``elongated_mexican_hat`` and ``ei_elongated`` so all
    blocks/populations use the *same* heterogeneity realisation.
    """
    if layout.ndim != 2:
        raise ValueError("Elongated connectivity requires a 2D sheet layout.")
    num = layout.num_units
    if smooth:
        raw = _random_field(layout, 3, smooth_sigma1, smooth_sigma2, seed)
        sig_f, ecc_f, ori_f = raw[0], raw[1], raw[2]
    else:
        rng = np.random.default_rng(seed)
        sig_f = rng.standard_normal(num)
        ecc_f = rng.standard_normal(num)
        ori_f = rng.standard_normal(num)

    sigma_scale = np.clip(1.0 + sigma_sd * sig_f, 1e-3, None)
    ecc = np.clip(eccentricity + eccentricity_sd * ecc_f, 0.0, 0.95)
    theta = orientation + orientation_sd * (np.pi * 0.5) * ori_f
    return sigma_scale, ecc, theta


def _anisotropic_gaussian(layout, base_sigma, maps):
    """Normalised anisotropic Gaussian connectivity given elongation ``maps``.

    ``maps`` is the ``(sigma_scale, ecc, theta)`` tuple from
    :func:`_elongation_maps`; each receiving unit ``i`` gets an elliptical
    Gaussian of width ``base_sigma * sigma_scale[i]``, eccentricity ``ecc[i]``
    and orientation ``theta[i]``.  Sheet only.
    """
    sigma_scale, ecc, theta = maps
    sigma_x = (base_sigma * sigma_scale)[:, None]        # per receiving unit (row)
    sigma_y = (sigma_x * np.sqrt(1.0 - ecc ** 2)[:, None])

    delta = layout.delta()                               # (i, j, 2)
    dx, dy = delta[:, :, 0], delta[:, :, 1]
    cos_t, sin_t = np.cos(theta)[:, None], np.sin(theta)[:, None]
    x_rot = dx * cos_t - dy * sin_t
    y_rot = dx * sin_t + dy * cos_t

    quad = (x_rot / sigma_x) ** 2 + (y_rot / sigma_y) ** 2
    norm = 1.0 / (2.0 * np.pi * sigma_x * sigma_y)
    return norm * np.exp(-quad / 2.0)


# --------------------------------------------------------------------------- #
# Single-population connectivity
# --------------------------------------------------------------------------- #
def mexican_hat(layout, sigma=1.8, inh_factor=2.5, excitation_amplitude=1.0,
                inhibition_amplitude=1.0):
    """Homogeneous Mexican-hat (difference-of-Gaussians) connectivity.

    Short-range excitation minus broader inhibition::

        W(d) = excitation_amplitude * G(d; sigma)
             - inhibition_amplitude * G(d; inh_factor * sigma)

    where ``G`` is a unit-integral Gaussian and ``d`` the (periodic) distance.
    Both amplitudes are *absolute*.  Setting ``inhibition_amplitude=0`` gives a
    purely excitatory Gaussian network.  Works for ring (1D) and sheet (2D).
    """
    exc = excitation_amplitude * _isotropic_gaussian(layout, sigma)
    inh = inhibition_amplitude * _isotropic_gaussian(layout, inh_factor * sigma)
    return (exc - inh).astype(np.float64)


def elongated_mexican_hat(layout, sigma=1.8, inh_factor=2.5,
                          excitation_amplitude=1.0, inhibition_amplitude=1.0,
                          eccentricity=0.8, eccentricity_sd=None, sigma_sd=0.08,
                          orientation=0.0, orientation_sd=1.0, smooth=False,
                          smooth_sigma1=1.8, smooth_sigma2=None,
                          seed=DEFAULT_SEED):
    """Elongated, heterogeneous Mexican-hat connectivity (sheet only).

    Each receiving unit has its own *elongated* (anisotropic) Mexican-hat field
    whose width, eccentricity and orientation vary across the sheet.  This is the
    salt-and-pepper / modular heterogeneity model of the 2D-network and
    cortical-development projects.

    Parameters
    ----------
    layout : Layout (2D sheet)
    sigma, inh_factor, excitation_amplitude, inhibition_amplitude : float
        Mexican-hat parameters (see :func:`mexican_hat`).
    eccentricity : float
        Mean eccentricity of the elongated fields (0 = isotropic, ->1 = line-like).
    eccentricity_sd : float, optional
        Spread of the per-unit eccentricity (default ``0.13 * eccentricity``).
    sigma_sd : float
        Fractional spread of the per-unit width.
    orientation, orientation_sd : float
        Mean and randomness of the per-unit orientation.
    smooth : bool
        If ``True`` the heterogeneity varies smoothly; otherwise it is
        independent per unit ("salt and pepper").
    smooth_sigma1, smooth_sigma2 : float
        Band-pass scales of the smoothing (default ``smooth_sigma2 =
        inh_factor * smooth_sigma1``).
    seed : int
        Random seed of the heterogeneity realisation.
    """
    if eccentricity_sd is None:
        eccentricity_sd = 0.13 * eccentricity
    if smooth_sigma2 is None:
        smooth_sigma2 = inh_factor * smooth_sigma1

    maps = _elongation_maps(layout, eccentricity, eccentricity_sd, sigma_sd,
                            orientation, orientation_sd, smooth,
                            smooth_sigma1, smooth_sigma2, seed)
    exc = excitation_amplitude * _anisotropic_gaussian(layout, sigma, maps)
    inh = inhibition_amplitude * _anisotropic_gaussian(layout, inh_factor * sigma,
                                                       maps)
    return (exc - inh).astype(np.float64)


def grf_mexican_hat(layout, sigma=1.8, inh_factor=2.0, excitation_amplitude=1.0,
                    inhibition_amplitude=1.0, grf_strength=0.4, grf_sigma1=2.0,
                    grf_sigma2=6.0, grf_same_pattern=True, normalise=True,
                    seed=DEFAULT_SEED):
    """Homogeneous Mexican hat modulated by a Gaussian random field (GRF).

    Starts from a homogeneous DoG and multiplies the outgoing weights of each
    unit by a positive band-pass Gaussian random field ``1 + grf_strength *
    GRF``.  The deformation is a smooth spatial random field (nothing to do with
    receptive fields).  Works for ring (1D) and sheet (2D).

    Parameters
    ----------
    layout : Layout
    sigma, inh_factor, excitation_amplitude, inhibition_amplitude : float
        Homogeneous Mexican-hat parameters (see :func:`mexican_hat`).
    grf_strength : float
        Amplitude of the Gaussian-random-field modulation.
    grf_sigma1, grf_sigma2 : float
        Band-pass kernel scales of the random field.
    grf_same_pattern : bool
        If ``True`` the same field modulates every unit (as in the published
        model); otherwise each unit gets an independent field.
    normalise : bool
        If ``True`` normalise each row to unit std and unit sum.
    seed : int
        Random seed.
    """
    base = mexican_hat(layout, sigma=sigma, inh_factor=inh_factor,
                       excitation_amplitude=excitation_amplitude,
                       inhibition_amplitude=inhibition_amplitude)
    num = layout.num_units
    if grf_same_pattern:
        field = _random_field(layout, 1, grf_sigma1, grf_sigma2, seed)[0]
        pert = np.clip(1.0 + grf_strength * field, 0.0, None)
        w_rec = base * pert[None, :]
    else:
        field = _random_field(layout, num, grf_sigma1, grf_sigma2, seed)
        pert = np.clip(1.0 + grf_strength * field, 0.0, None)
        w_rec = base * pert

    if normalise:
        w_rec = w_rec / w_rec.std(axis=1, keepdims=True)
        w_rec = w_rec / w_rec.sum(axis=1, keepdims=True)
    return w_rec.astype(np.float64)


def shuffle_connectivity(w_rec, layout, shuffle_type="weight_swap", fraction=0.5,
                         local_radius=None, n_pop=1, seed=DEFAULT_SEED):
    """Partially shuffle a connectivity matrix (spatial scrambling).

    Two schemes, following the influence-mapping ``ring`` model:

    * ``"location"``    -- permute a random ``fraction`` of unit *locations*
      (rows and columns together), scrambling the spatial arrangement while
      preserving every weight value.
    * ``"weight_swap"`` -- for each unit, swap a ``fraction`` of its *local*
      weights (within ``local_radius``) with randomly chosen *non-local* ones,
      injecting long-range connections while preserving the weight distribution.

    Parameters
    ----------
    w_rec : numpy.ndarray
        Connectivity to shuffle; ``(num, num)`` for one population or
        ``(2*num, 2*num)`` for an E/I network (set ``n_pop=2``).
    layout : Layout
        The layout the *single population* lives on.
    shuffle_type : {"weight_swap", "location"}
    fraction : float
        Fraction of locations / local weights to shuffle, in ``[0, 1]``.
    local_radius : float, optional
        Radius defining "local" for ``weight_swap`` (default ``2 * 1``-ish;
        falls back to ``3.0``).
    n_pop : {1, 2}
        Number of populations packed into ``w_rec``.
    seed : int
        Random seed.
    """
    rng = np.random.default_rng(seed)
    w = np.asarray(w_rec, dtype=np.float64).copy()
    num = layout.num_units
    dist = layout.distance()
    if local_radius is None:
        local_radius = 3.0

    if shuffle_type == "location":
        k = int(round(fraction * num))
        loc_perm = np.arange(num)
        if k > 1:
            idx = rng.choice(num, size=k, replace=False)
            loc_perm[idx] = idx[rng.permutation(k)]
        full = loc_perm if n_pop == 1 else np.concatenate(
            [loc_perm + p * num for p in range(n_pop)])
        return w[full][:, full]

    if shuffle_type == "weight_swap":
        within = dist <= local_radius
        outside = ~within
        np.fill_diagonal(within, False)
        np.fill_diagonal(outside, False)

        def swap_block(block):
            for i in range(num):
                local = np.where(within[i])[0]
                nonlocal_ = np.where(outside[i])[0]
                n_swaps = int(np.floor(fraction * local.size))
                if n_swaps == 0 or nonlocal_.size < n_swaps:
                    continue
                a = rng.choice(local, size=n_swaps, replace=False)
                b = rng.choice(nonlocal_, size=n_swaps, replace=False)
                block[i, a], block[i, b] = block[i, b].copy(), block[i, a].copy()
            return block

        if n_pop == 1:
            return swap_block(w)
        for pr in range(n_pop):
            for pc in range(n_pop):
                sub = w[pr * num:(pr + 1) * num, pc * num:(pc + 1) * num]
                w[pr * num:(pr + 1) * num, pc * num:(pc + 1) * num] = swap_block(sub)
        return w

    raise ValueError(f"Unknown shuffle_type '{shuffle_type}'.")


def shuffled_mexican_hat(layout, sigma=1.8, inh_factor=2.5,
                         excitation_amplitude=1.0, inhibition_amplitude=1.0,
                         shuffle_type="weight_swap", fraction=0.5,
                         local_radius=None, seed=DEFAULT_SEED):
    """Homogeneous Mexican hat with its weights partially shuffled.

    Convenience wrapper: build a homogeneous DoG (see :func:`mexican_hat`) and
    apply :func:`shuffle_connectivity`.  Ring or sheet.
    """
    base = mexican_hat(layout, sigma=sigma, inh_factor=inh_factor,
                       excitation_amplitude=excitation_amplitude,
                       inhibition_amplitude=inhibition_amplitude)
    return shuffle_connectivity(base, layout, shuffle_type=shuffle_type,
                                fraction=fraction, local_radius=local_radius,
                                seed=seed)


def random_network(layout, g=1.0, density=1.0, gaussian=True,
                   normalise="spectral", spectral_radius=0.95, seed=DEFAULT_SEED):
    """Random recurrent network (Gaussian or uniform, optionally sparse).

    Each weight is drawn independently and scaled by ``g / sqrt(num * density)``.
    With ``density < 1`` a fraction of the weights is set to zero.  Ring or sheet.

    Parameters
    ----------
    layout : Layout
    g : float
        Overall gain of the random weights.
    density : float
        Connection probability (fraction of non-zero weights), in ``(0, 1]``.
    gaussian : bool
        Draw from a normal distribution (``True``) or uniform ``[0, 1)`` (``False``).
    normalise : {"spectral", None}
        If ``"spectral"`` rescale so the largest real eigenvalue equals
        ``spectral_radius``.
    spectral_radius : float
        Target spectral radius when ``normalise == "spectral"``.
    seed : int
        Random seed.
    """
    num = layout.num_units
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal((num, num)) if gaussian else rng.random((num, num))
    w_rec = raw * (g / np.sqrt(num * density))
    if density < 1.0:
        w_rec = w_rec * (rng.random((num, num)) < density)
    if normalise == "spectral":
        w_rec = scale_spectral_radius(w_rec, spectral_radius)
    return w_rec.astype(np.float64)


def modular_network(layout, dist_spec=5.0, mod_high=5.0, mod_low=2.0, beta=1.0,
                    normalise=True, seed=DEFAULT_SEED):
    """Long-range modular connectivity (sheet only).

    Builds low-dimensional band-pass random patterns, forms a covariance matrix
    with exponentially-decaying eigenvalues, and multiplies it by a Gaussian
    distance kernel so that correlations remain spatially local.  Produces
    distributed, modular recurrent structure.

    Parameters
    ----------
    layout : Layout (2D sheet)
    dist_spec : float
        Spatial scale of the Gaussian distance kernel.
    mod_high, mod_low : float
        High/low Gaussian kernel scales of the band-pass pattern filter.
    beta : float
        Controls the eigenvalue decay ``exp(-2 k / beta)``.
    normalise : bool
        If ``True`` balance E/I per row and normalise to unit std.
    seed : int
        Random seed.
    """
    if layout.ndim != 2:
        raise ValueError("modular_network requires a 2D sheet.")
    N, M = layout.shape
    num = layout.num_units

    eigenvalues = np.exp(-2.0 * np.arange(num) / beta)
    rng = np.random.default_rng(seed)
    patterns = rng.normal(size=(num, N, M))
    patterns = (gaussian_filter(patterns, sigma=(0, mod_low, mod_low))
                - gaussian_filter(patterns, sigma=(0, mod_high, mod_high)))

    basis = patterns.reshape(num, num).T
    long_range = (basis * eigenvalues[None, :]) @ basis.T

    dist = layout.distance()
    dist_kernel = np.exp(-dist ** 2 / (2.0 * dist_spec ** 2)) / (
        np.sqrt(2 * np.pi) * dist_spec)
    w_rec = dist_kernel * long_range

    if normalise:
        pos = np.clip(w_rec, 0, np.inf)
        neg = np.clip(w_rec, -np.inf, 0)
        ratio = np.divide(-neg.sum(1), pos.sum(1),
                          out=np.ones(num), where=pos.sum(1) != 0)
        w_rec = neg / ratio[:, None] + pos
        w_rec = w_rec / w_rec.std(axis=0, keepdims=True)
        w_rec = w_rec.T
    return w_rec.astype(np.float64)


# --------------------------------------------------------------------------- #
# Excitatory / inhibitory (two-population) connectivity
# --------------------------------------------------------------------------- #
def _ei_meta(num):
    return {"n_exc": num, "n_inh": num,
            "exc_slice": slice(0, num), "inh_slice": slice(num, 2 * num)}


def excitatory_inhibitory(layout, profile="homogeneous", sigma_ee=2.0,
                          sigma_ei=5.0, sigma_ie=2.0, sigma_ii=5.0, w_ee=1.0,
                          w_ei=1.0, w_ie=1.0, w_ii=1.0, self_inhibition=False,
                          alpha=0.5, normalise=None, spectral_radius=0.95,
                          eccentricity=0.8, eccentricity_sd=None, sigma_sd=0.08,
                          orientation=0.0, orientation_sd=1.0, smooth=False,
                          grf_strength=0.4, grf_sigma1=2.0, grf_sigma2=6.0,
                          gaussian=True, seed=DEFAULT_SEED):
    """Two-population excitatory/inhibitory network with four Gaussian blocks.

    Returns a ``(2N, 2N)`` matrix with the block structure::

        W = [[  w_ee * G_ee ,  -w_ei * G_ei ],
             [  w_ie * G_ie ,  -w_ii * G_ii ]]

    where each block ``G_xy`` has its own width ``sigma_xy``.  The first ``N``
    rows/columns are the excitatory population, the last ``N`` the inhibitory.
    The ``profile`` selects how the blocks are built:

    * ``"homogeneous"`` -- isotropic Gaussians (ring or sheet);
    * ``"elongated"``   -- anisotropic, heterogeneous Gaussians (sheet only);
    * ``"grf"``         -- Gaussian-random-field-modulated Gaussians (ring/sheet);
    * ``"random"``      -- random Gaussian/uniform blocks (ring or sheet),
      spectrally normalised.

    Parameters
    ----------
    layout : Layout
    profile : {"homogeneous", "elongated", "grf", "random"}
    sigma_ee, sigma_ei, sigma_ie, sigma_ii : float
        Widths of the four Gaussian blocks (E and I reaches).
    w_ee, w_ei, w_ie, w_ii : float
        Strengths of the four blocks (absolute).
    self_inhibition : bool
        Mix a diagonal ``-I`` term into the I->I block (weighted by ``alpha``).
    alpha : float
        Self-inhibition fraction.
    normalise : {"spectral", None}
        Spectral normalisation of the full matrix (default ``"spectral"`` for the
        ``random`` profile, ``None`` otherwise).
    spectral_radius : float
        Target spectral radius when normalising.
    eccentricity, eccentricity_sd, sigma_sd, orientation, orientation_sd, smooth :
        Heterogeneity parameters for the ``"elongated"`` profile.
    grf_strength, grf_sigma1, grf_sigma2 : float
        GRF parameters for the ``"grf"`` profile.
    gaussian : bool
        Draw Gaussian (vs uniform) blocks for the ``"random"`` profile.
    seed : int
        Random seed.

    Returns
    -------
    (numpy.ndarray, dict)
        The ``2N x 2N`` matrix and E/I split metadata.
    """
    num = layout.num_units
    rng = np.random.default_rng(seed)

    if profile == "elongated":
        if eccentricity_sd is None:
            eccentricity_sd = 0.13 * eccentricity
        maps = _elongation_maps(layout, eccentricity, eccentricity_sd, sigma_sd,
                                orientation, orientation_sd, smooth,
                                smooth_sigma1=min(sigma_ee, sigma_ie),
                                smooth_sigma2=max(sigma_ei, sigma_ii), seed=seed)

    def block(sigma):
        """Build one (positive) connectivity block for the chosen profile."""
        if profile == "homogeneous":
            return _isotropic_gaussian(layout, sigma)
        if profile == "elongated":
            return _anisotropic_gaussian(layout, sigma, maps)
        if profile == "grf":
            g = _isotropic_gaussian(layout, sigma)
            field = _random_field(layout, 1, grf_sigma1, grf_sigma2, seed)[0]
            return g * np.clip(1.0 + grf_strength * field, 0.0, None)[None, :]
        if profile == "random":
            return (rng.standard_normal((num, num)) if gaussian
                    else rng.random((num, num)))
        raise ValueError(f"Unknown EI profile '{profile}'.")

    mee = w_ee * block(sigma_ee)
    mei = -w_ei * block(sigma_ei)
    mie = w_ie * block(sigma_ie)
    mii_block = block(sigma_ii)
    if self_inhibition:
        mii = -w_ii * ((1.0 - alpha) * mii_block + alpha * np.eye(num))
    else:
        mii = -w_ii * mii_block

    w_rec = np.block([[mee, mei], [mie, mii]]).astype(np.float64)

    if normalise is None:
        normalise = "spectral" if profile == "random" else None
    if normalise == "spectral":
        w_rec = scale_spectral_radius(w_rec, spectral_radius)
    return w_rec, _ei_meta(num)


def _ei_profile(profile):
    """Return an ``excitatory_inhibitory`` wrapper for a fixed ``profile``."""
    def wrapper(layout, **params):
        return excitatory_inhibitory(layout, profile=profile, **params)
    wrapper.__doc__ = (f"E/I network with the '{profile}' block profile. "
                       f"See :func:`excitatory_inhibitory`.")
    return wrapper


ei_homogeneous = _ei_profile("homogeneous")
ei_elongated = _ei_profile("elongated")
ei_grf = _ei_profile("grf")
ei_random = _ei_profile("random")


# --------------------------------------------------------------------------- #
# Dispatcher
# --------------------------------------------------------------------------- #
#: Mapping from ``kind`` string to generator function.
CONNECTIVITY_GENERATORS = {
    "mexican_hat": mexican_hat,
    "elongated_mexican_hat": elongated_mexican_hat,
    "grf_mexican_hat": grf_mexican_hat,
    "shuffled_mexican_hat": shuffled_mexican_hat,
    "random": random_network,
    "modular": modular_network,
    "ei_homogeneous": ei_homogeneous,
    "ei_elongated": ei_elongated,
    "ei_grf": ei_grf,
    "ei_random": ei_random,
}


def available_connectivities():
    """Return the sorted list of registered connectivity kinds."""
    return sorted(CONNECTIVITY_GENERATORS)


def make_connectivity(kind, layout, **params):
    """Build a recurrent weight matrix by name.

    Parameters
    ----------
    kind : str
        One of :func:`available_connectivities`.
    layout : Layout
        Spatial layout of the network.
    **params
        Passed through to the generator (see the individual functions); every
        parameter has a default, so ``make_connectivity(kind, layout)`` works.

    Returns
    -------
    numpy.ndarray or (numpy.ndarray, dict)
        The weight matrix.  The ``ei_*`` kinds additionally return E/I metadata.
    """
    try:
        generator = CONNECTIVITY_GENERATORS[kind]
    except KeyError as exc:
        raise ValueError(
            f"Unknown connectivity kind '{kind}'. "
            f"Available: {available_connectivities()}"
        ) from exc
    return generator(layout, **params)
