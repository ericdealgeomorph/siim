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

# Numba threading layer: default to 'workqueue' (numba's own fork-safe thread
# pool) BEFORE any submodule imports numba. This avoids the macOS crash where
# the default TBB/OpenMP layer collides with Accelerate-backed numpy/scipy and
# the flexure FFT under a Jupyter kernel (Eric, 2026-07-07: reliably kills
# flexure runs, rare otherwise). setdefault so an explicit user
# NUMBA_THREADING_LAYER (e.g. 'tbb' for the last few % on parallel kernels)
# always wins; the choice is correctness-neutral (bit-identical results) and
# within process noise on siim's non-nested prange kernels.
_os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")

__version__ = "0.9.4"


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
