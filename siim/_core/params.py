"""Frozen per-run parameter record for the law_code skeleton kernels.

One ``GlacialParams`` bundles every per-run physical/law scalar the skeletons
consume, so a kernel call passes one record instead of ~16 positional floats
(and no per-law ``0.0`` padding). Built once per run by the model, unpacked at
the top of each skeleton; the per-law dispatchers still receive plain scalars.

IMPORTANT — this type MUST be defined here (a stable, importable module) and
imported everywhere; never reconstruct an equivalent namedtuple in ``__main__``
or per-call. numba keys a ``@njit(cache=True)`` function's on-disk cache on its
argument *types*, and a namedtuple's type identity comes from its defining
module + qualname. A ``__main__``- or locally-defined twin is a *different*
numba type, so the cache silently misses and recompiles every process (same
failure class as the closure/exec patterns the Step-0 spike ruled out;
verified directly with NUMBA_DEBUG_CACHE).
"""
from collections import namedtuple

#: Order is load-bearing: the skeletons tuple-unpack ``p`` positionally.
#: Every field defaults to 0.0 so the production builders keyword-construct only
#: their law's real fields (the inactive-law padding stays implicit; audit N19).
GlacialParams = namedtuple('GlacialParams', [
    'Ko', 'Co', 'ce',                 # erosion coefficients (fluvial Ko; glacial Co/ce)
    'n', 'nu', 'm', 'mu',             # stream-power / sliding exponents
    'cg', 'alpha_g', 'lambda_p',      # glacial flux + eff-exp/power length scale
    'lambda_c', 'tau_c', 'coulomb_clamp', 'rho_g_g',   # regularized-Coulomb set
    'hc_over_H', 'D_H',               # channel-floor datum ratio; H diffusivity
], defaults=[0.0] * 16)
