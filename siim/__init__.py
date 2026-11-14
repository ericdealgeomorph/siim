"""siim — Sliding Ice Incision Model.

Coupled glacial-fluvial landscape evolution: a 1D profile model
(``siim.siim1d``), a standalone 2D raster model (``siim.siim2d``, with an
escarpment variant in ``siim.escarpment``), and the analytical steady-state
solution (``siim.analytical``) both numerical models embed as their reference.

Entry points::

    from siim.siim1d import siim                       # 1D profile model
    from siim.siim2d import siim                       # 2D landscape model
    from siim.fastscape import glacial_processes       # optional adapter for your own fastscape model
    from siim import analytical_steady_state_solution  # analytical reference
    from siim.analytical import GeneralProfile         # steepness-form theory
    from siim.analytical import MarginalCoulombProfile # exact d*phi=1 case
    from siim.forcing import ela_sawtooth, uplift_step, interp_forcing # time-varying forcing builders

The 1D and 2D models run directly on the package's NumPy/Numba numerical core;
the 2D driver uses in-house routing, flexure, and diffusion. xsimlab, fastscape,
and fastscapelib-fortran are needed only for the optional ``siim.fastscape``
adapter. ``siim.analytical`` itself needs only NumPy/SciPy. This module imports
nothing eagerly so the optional adapter never affects a standard import.
"""

import os as _os
import sys as _sys

# Numba threading layer, set BEFORE any submodule imports numba: TBB when numba
# can load it (conda-forge's tbb, or the optional `siim-lem[tbb]` extra), else
# numba's own workqueue pool; OpenMP is never reached (workqueue always loads).
# The level-scheduled eroder opens one parallel region per drainage level, far
# cheaper to start on TBB (docs/getting_started/install.md), and every result
# is bit-for-bit on either layer. TBB with the flexure FFT ran clean under a
# Jupyter kernel on 2026-11-12; the 2026-07-07 crash that pinned workqueue was
# numba 0.65. An explicit NUMBA_THREADING_LAYER, or the user's own priority
# (setdefault), always wins.
_os.environ.setdefault("NUMBA_THREADING_LAYER_PRIORITY", "tbb workqueue omp")
# A numba imported before siim has already read its environment, and launching
# its thread pool does not read it again (only a compile does): without this
# its own priority, OpenMP ahead of workqueue, would stand. numba re-reads its
# whole configuration here, as it would at its next compile anyway (the
# environment changed): doing it now keeps what the user sets on numba.config
# after this import, where writing the one setting lost it at that compile.
# What was set on numba.config BEFORE this import is reset either way, when
# the setdefault above added the variable.
if "numba" in _sys.modules:
    _sys.modules["numba"].config.reload_config()

__version__ = "0.10.0"


_LAZY = ("analytical_steady_state_solution", "GeneralProfile",
         "MarginalCoulombProfile", "SteadyStateProfile")


def __getattr__(name):
    # lazy: keeps `import siim` free of the heavy (and optional) 2D deps
    if name == "analytical_steady_state_solution":
        from .analytical import analytical_steady_state_solution
        return analytical_steady_state_solution
    if name in ("GeneralProfile", "MarginalCoulombProfile",
                "SteadyStateProfile"):
        from . import analytical
        return getattr(analytical, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    # surface the lazy attributes for dir(siim) / IDE completion (audit N18)
    return sorted(list(globals()) + list(_LAZY))
