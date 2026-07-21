"""
Spatial layout of the neural sheet.

A :class:`Layout` describes *where* the units of a firing-rate network sit in
space and how far apart they are.  The topology is inferred directly from the
``shape`` (so it is never specified twice):

* a scalar ``N``      -> a one-dimensional **ring** of ``N`` units;
* a pair ``(N, M)``   -> a two-dimensional **sheet** of ``N`` x ``M`` units.

Either topology can use periodic boundary conditions (``periodic=True``) or open
boundaries.

The class exposes the two quantities every connectivity rule needs:

* :meth:`distance` -- the pairwise Euclidean distance matrix.
* :meth:`delta`    -- the signed, per-axis coordinate differences (needed for
  anisotropic / oriented connectivity such as the elongated Mexican hat).

Both respect the periodic boundary via the minimum-image convention.
"""

from __future__ import annotations

import numpy as np


class Layout:
    """Spatial layout of the units of a firing-rate network.

    Parameters
    ----------
    shape : int or tuple of int
        Size of the network.  A scalar (or 1-tuple) creates a 1D ring; a
        2-tuple ``(N, M)`` creates a 2D sheet.  The topology is inferred from
        this and exposed as :attr:`topology`.
    periodic : bool, optional
        Whether to use periodic boundary conditions (default ``True``).
    """

    def __init__(self, shape, periodic=True):
        # ---- normalise the shape into a tuple of ints -----------------------
        if np.isscalar(shape):
            shape = (int(shape),)
        else:
            shape = tuple(int(s) for s in shape)

        if len(shape) not in (1, 2):
            raise ValueError("shape must be 1D (ring) or 2D (sheet), "
                             f"got {shape}")

        self.shape = shape
        self.periodic = bool(periodic)
        self.ndim = len(shape)
        self.topology = "ring" if self.ndim == 1 else "sheet"
        self.num_units = int(np.prod(shape))

        # Cache the coordinate grid (num_units, ndim), row-major (C) order so
        # that reshaping activity to ``self.shape`` is always consistent.
        self._coordinates = self._build_coordinates()

    # ------------------------------------------------------------------ #
    # Construction helpers
    # ------------------------------------------------------------------ #
    def _build_coordinates(self):
        """Return integer coordinates of every unit, shape (num_units, ndim)."""
        axes = [np.arange(s) for s in self.shape]
        mesh = np.meshgrid(*axes, indexing="ij")
        coords = np.stack([m.ravel() for m in mesh], axis=-1)
        return coords.astype(np.float64)

    @property
    def coordinates(self):
        """Integer grid coordinates of every unit, shape ``(num_units, ndim)``."""
        return self._coordinates

    # ------------------------------------------------------------------ #
    # Layout queries
    # ------------------------------------------------------------------ #
    def delta(self):
        """Signed per-axis coordinate differences between all unit pairs.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(num_units, num_units, ndim)`` where
            ``delta[i, j, d] = coord_i[d] - coord_j[d]``.  With periodic
            boundaries the minimum-image convention is applied so that the
            difference lies in ``[-L_d / 2, L_d / 2]``.
        """
        coords = self._coordinates
        d = coords[:, None, :] - coords[None, :, :]
        if self.periodic:
            lengths = np.asarray(self.shape, dtype=np.float64)
            d = d - lengths * np.round(d / lengths)
        return d

    def distance(self):
        """Pairwise Euclidean distance matrix, shape ``(num_units, num_units)``.

        Periodic boundaries are respected via the minimum-image convention.
        """
        d = self.delta()
        return np.sqrt(np.sum(d ** 2, axis=-1))

    # ------------------------------------------------------------------ #
    # Reshaping helpers
    # ------------------------------------------------------------------ #
    def to_grid(self, activity):
        """Reshape a flat ``(..., num_units)`` array to ``(..., *shape)``."""
        activity = np.asarray(activity)
        lead = activity.shape[:-1]
        return activity.reshape(lead + self.shape)

    def to_flat(self, activity):
        """Reshape a grid ``(..., *shape)`` array back to ``(..., num_units)``."""
        activity = np.asarray(activity)
        lead = activity.shape[:-self.ndim]
        return activity.reshape(lead + (self.num_units,))

    # ------------------------------------------------------------------ #
    def __repr__(self):
        return (f"Layout(shape={self.shape}, topology='{self.topology}', "
                f"periodic={self.periodic}, num_units={self.num_units})")
