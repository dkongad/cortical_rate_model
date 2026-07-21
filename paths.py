"""
Centralised path configuration for saving and loading.

The source projects hard-coded absolute output paths in a ``global_params``
module.  Here the paths live in a single, easily-edited place: an environment
variable or the :class:`PathConfig` object.  **This is the one file to edit when
moving the package to a new machine or storage location.**

Resolution order for the base output directory:

1. the ``base_dir`` passed explicitly to :class:`PathConfig`;
2. the ``CORTICAL_RATE_MODEL_HOME`` environment variable;
3. the default ``~/cortical_rate_model_output``.

Under the base directory the following sub-folders are used:

    <base>/models/     saved networks (weights + parameters)
    <base>/activity/   simulated activity / responses
    <base>/figures/    saved plots
"""

from __future__ import annotations

import os
from pathlib import Path


#: Environment variable that overrides the default base directory.
ENV_VAR = "CORTICAL_RATE_MODEL_HOME"

#: Default base directory used when nothing else is configured.
DEFAULT_BASE_DIR = Path.home() / "cortical_rate_model_output"


class PathConfig:
    """Container for the input/output directory layout.

    Parameters
    ----------
    base_dir : str or pathlib.Path, optional
        Root directory for all outputs.  If ``None`` the environment variable
        :data:`ENV_VAR` or :data:`DEFAULT_BASE_DIR` is used.
    create : bool, optional
        Create the directories on construction (default ``True``).
    """

    def __init__(self, base_dir=None, create=True):
        if base_dir is None:
            base_dir = os.environ.get(ENV_VAR, DEFAULT_BASE_DIR)
        self.base_dir = Path(base_dir).expanduser().resolve()
        self.models_dir = self.base_dir / "models"
        self.activity_dir = self.base_dir / "activity"
        self.figures_dir = self.base_dir / "figures"
        if create:
            self.make_dirs()

    def make_dirs(self):
        """Create the base directory and all sub-folders if missing."""
        for d in (self.models_dir, self.activity_dir, self.figures_dir):
            d.mkdir(parents=True, exist_ok=True)

    # Convenience helpers returning full file paths ---------------------- #
    def model_path(self, name):
        """Path (without extension) for a saved model called ``name``."""
        return self.models_dir / name

    def activity_path(self, name):
        """Path for a saved activity file called ``name``."""
        return self.activity_dir / name

    def figure_path(self, name):
        """Path for a saved figure called ``name``."""
        return self.figures_dir / name

    def __repr__(self):
        return f"PathConfig(base_dir='{self.base_dir}')"


#: A module-level default configuration (directories are created lazily on use).
default_paths = PathConfig(create=False)
