"""Math kernel for the analytical steady-state machinery.

Pure numpy/scipy: the analytically continued incomplete beta (including the
exact logarithmic limit at b = 0, i.e. the marginal-Coulomb case d*phi = 1),
the closure root-solver shared by the profile classes, and the shared result
containers. Nothing here imports the model stack.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.special as _sp
from scipy.optimize import brentq

_EULER_GAMMA = float(np.euler_gamma)
# Crossover between the small-x (hyp2f1 series) and near-one (boundary-term /
# digamma) evaluations of the incomplete beta.
_X_SPLIT = 0.5
# Below this eps the b = 0 branch (a <= 1) uses the digamma asymptotic
# -log(eps) - psi(a) - gamma + (a-1) eps  (error O(eps^2)); above it scipy's
# hyp2f1 at x = 1-eps is accurate for a <= 1.
_EPS_ASYM = 1e-6
# For a > 1, scipy's hyp2f1 is defective at the c-a-b = 0 logarithmic point
# for 1e-6 < eps <~ 1e-3 (audit m4). On eps <= _EPS_SERIES the exact
# logarithmic series (below) is used instead; hyp2f1 is reliable above it.
_EPS_SERIES = 1e-2
_B0_SERIES_TERMS = 24


def _ib_b0_log_series(eps, a):
    r"""``B(1-eps; a, 0)`` via the exact logarithmic expansion

    .. math::

       B(1-\epsilon;\,a,\,0) = -\log\epsilon - \psi(a) - \gamma_E
       + \sum_{n\ge 1} \frac{(-1)^{n+1}}{n}\binom{a-1}{n}\,\epsilon^{n},

    obtained by integrating ``dB/d\epsilon = -(1-\epsilon)^{a-1}/\epsilon``. It
    converges with no cancellation for small ``eps`` (unlike scipy's hyp2f1 at
    the ``c-a-b = 0`` logarithmic point, which is defective for ``a > 1``;
    audit m4). Intended for ``a > 1`` on ``eps <= _EPS_SERIES``; ``a`` scalar,
    ``eps`` array-like.
    """
    eps = np.asarray(eps, dtype=float)
    with np.errstate(divide='ignore'):            # eps = 0 -> -log -> +inf
        out = -np.log(eps) - _sp.digamma(a) - _EULER_GAMMA
    binom = 1.0                                   # C(a-1, 0)
    epsn = np.ones_like(eps)
    for n in range(1, _B0_SERIES_TERMS + 1):
        binom *= (a - n) / n                      # C(a-1, n) from C(a-1, n-1)
        epsn = epsn * eps
        out = out + ((-1.0) ** (n + 1) * binom / n) * epsn
    return out
# Cap on the b -> b+1 recurrence (guards the b = (1-d*phi)/k -> -inf abuse
# that previously hung or NaN-corrupted the evaluation).
_MAX_RECURRENCE = 10_000


def _arccosh_safe(v):
    return np.arccosh(np.maximum(v, 1.0))


def _validate_ab(a, b):
    if not (a > 0):
        raise ValueError(f'incomplete_beta requires a > 0, got a={a}')
    if b < -_MAX_RECURRENCE:
        raise ValueError(
            f'incomplete_beta: b={b} is too negative (recurrence would need '
            f'{-b:.0f} steps); for the glacial profile b = (1-d*phi)/k — '
            'check k and d*phi.')


def _ib_small_x(x, a, b):
    """B(x; a, b) for x in [0, ~0.5] via DLMF 8.17.7:
    B(x; a, b) = x^a / a * 2F1(a, 1-b; a+1; x). Valid for any real b."""
    return x ** a / a * _sp.hyp2f1(a, 1.0 - b, a + 1.0, x)


def _ib_near_one(eps, a, b):
    r"""B(1-eps; a, b) for eps in [0, ~0.5], exact for b <= 0.

    b < 0 is lifted to [0, 1) with the integration-by-parts recurrence

    .. math::

       B(x;\,a,\,b) = -\frac{x^a (1-x)^b}{b} + \frac{a+b}{b}\,B(x;\,a,\,b+1)

    whose boundary terms are computed directly in eps (= 1-x), so the
    divergence as eps -> 0 is represented exactly instead of saturating.
    The b = 0 remainder uses the exact logarithmic limit

    .. math::

       B(1-\epsilon;\,a,\,0) = -\log\epsilon - \psi(a) - \gamma_E
       + (a-1)\,\epsilon + O(\epsilon^2)

    for small eps and hyp2f1 otherwise; b > 0 uses the reflection
    B(1-eps; a, b) = B(a, b) - B(eps; b, a).
    """
    eps = np.asarray(eps, dtype=float)
    xa = np.exp(a * np.log1p(-eps))        # (1-eps)^a without cancellation

    boundary = np.zeros_like(eps)
    coeff = 1.0
    with np.errstate(over='ignore', divide='ignore'):
        while b < 0:
            boundary -= coeff * xa * eps ** b / b
            coeff *= (a + b) / b
            b += 1.0

        if b == 0.0:
            out = np.empty_like(eps)
            if a <= 1.0:
                # Exact for a <= 1 (the in-package regime, oracle-pinned):
                # digamma asymptotic for tiny eps, hyp2f1 otherwise. UNCHANGED.
                small = eps <= _EPS_ASYM
                if small.any():
                    e = eps[small]
                    with np.errstate(divide='ignore'):  # eps = 0 -> log -> -inf
                        out[small] = (-np.log(e) - _sp.digamma(a) - _EULER_GAMMA
                                      + (a - 1.0) * e)
                if (~small).any():
                    e = eps[~small]
                    out[~small] = _ib_small_x(1.0 - e, a, 0.0)
            else:
                # a > 1: the exact logarithmic series on the near-one window
                # (where hyp2f1 is defective; audit m4), hyp2f1 above it.
                near = eps <= _EPS_SERIES
                if near.any():
                    out[near] = _ib_b0_log_series(eps[near], a)
                if (~near).any():
                    out[~near] = _ib_small_x(1.0 - eps[~near], a, 0.0)
            core = out
        else:
            core = _sp.beta(a, b) * (1.0 - _sp.betainc(b, a, eps))

    return boundary + coeff * core


def incomplete_beta(x, a, b):
    r"""Non-regularized incomplete beta function, analytically continued in b.

    .. math::

       B(x;\,a,\,b) = \int_0^x t^{a-1} (1-t)^{b-1}\,dt

    Valid for a > 0 and any real b, including b = 0 (the logarithmically
    divergent marginal case d*phi = 1, handled exactly) and non-positive
    integers. x in [0, 1]; at x = 1 the value is the complete beta for
    b > 0 and +inf for b <= 0. Vectorizes over x.

    When x is close to 1 and you know eps = 1 - x exactly (e.g. (xo/Lt)^k),
    call :func:`incomplete_beta_compl` instead to avoid the 1 - x roundoff.

    Parameters
    ----------
    x : float or array_like
        Upper integration limit(s), in [0, 1].
    a : float
        First shape parameter; must be > 0.
    b : float
        Second shape parameter; any real value (analytic continuation).

    Returns
    -------
    float or ndarray
        B(x; a, b), scalar when ``x`` is scalar, else an array of x's shape.

    Raises
    ------
    ValueError
        If a <= 0, if x is outside [0, 1], or if b is so negative that the
        b -> b+1 recurrence would exceed the internal step cap.
    """
    _validate_ab(a, b)
    x = np.asarray(x, dtype=float)
    scalar = x.ndim == 0
    x = np.atleast_1d(x)
    if np.any((x < 0.0) | (x > 1.0)):
        raise ValueError('incomplete_beta requires x in [0, 1]')

    out = np.empty_like(x)
    lo = x <= _X_SPLIT
    if b > 0:
        out[:] = _sp.beta(a, b) * _sp.betainc(a, b, x)
    else:
        if lo.any():
            out[lo] = _ib_small_x(x[lo], a, b)
        if (~lo).any():
            out[~lo] = _ib_near_one(1.0 - x[~lo], a, b)
    return float(out[0]) if scalar else out


def incomplete_beta_compl(eps, a, b):
    """B(1 - eps; a, b) with eps supplied directly (exact for tiny eps).

    The profile shape function G evaluates the incomplete beta at
    1 - (x/Lt)^k; for x << Lt the argument is so close to 1 that forming it
    in floating point loses (or zeroes) the information. Passing
    eps = (x/Lt)^k here keeps the evaluation exact all the way into the
    divergent tail — this replaces the old upper-limit clip, which silently
    froze G once (x/Lt)^k dropped below 1e-12. Vectorizes over eps.

    Parameters
    ----------
    eps : float or array_like
        Complement of the upper integration limit, eps = 1 - x, in [0, 1].
    a, b : float
        Shape parameters, as in :func:`incomplete_beta`.

    Returns
    -------
    float or ndarray
        B(1 - eps; a, b), scalar when ``eps`` is scalar.

    Raises
    ------
    ValueError
        Same conditions as :func:`incomplete_beta`.
    """
    _validate_ab(a, b)
    eps = np.asarray(eps, dtype=float)
    scalar = eps.ndim == 0
    eps = np.atleast_1d(eps)
    if np.any((eps < 0.0) | (eps > 1.0)):
        raise ValueError('incomplete_beta_compl requires eps in [0, 1]')

    out = np.empty_like(eps)
    near = eps <= (1.0 - _X_SPLIT)
    if near.any():
        out[near] = _ib_near_one(eps[near], a, b)
    if (~near).any():
        x = 1.0 - eps[~near]
        if b > 0:
            out[~near] = _sp.beta(a, b) * _sp.betainc(a, b, x)
        else:
            out[~near] = _ib_small_x(x, a, b)
    return float(out[0]) if scalar else out


# ---------------------------------------------------------------------------
# Closure root-solver
# ---------------------------------------------------------------------------
# Both profile classes (and, in a later phase, the physical front end) reduce
# the steady state to roots of one continuous scalar closure zELA(Lt) = zELA.
# The closure is C0 but generally not C1 at the mixed/glacial seam Lt = L
# (the "kink"), and saddle-node fold pairs can sit closer together than a
# scan cell. The solver below owns those concerns once.

def find_closure_roots(f, lo, hi, *, kink=None, n=4000, hi_max=None,
                       n_per_decade_ext=400, refine=64):
    """All roots of a vectorized scalar function f on [lo, hi(, hi_max)].

    Log-spaced scan + brentq on every strict sign change. Grid nodes where
    f is exactly zero count as roots once (np.sign(0) double-bracketing is
    avoided). Local ``|f|`` minima without a sign change are re-scanned
    `refine` times finer to catch fold pairs narrower than one cell. If
    `hi_max` is given, the grid is extended decade-by-decade past `hi`
    (coarser, `n_per_decade_ext` points per decade) so far-out roots are not
    missed. `kink` (e.g. Lt = L) is inserted as an exact grid node so no
    bracket straddles the non-smooth point. brentq failures on a valid
    bracket propagate — a sign change that cannot be refined is a bug in f,
    not noise to be swallowed.

    Returns
    -------
    list of float
        Sorted roots (duplicates within rtol 1e-9 merged).
    """
    grid = np.geomspace(lo, hi, n)
    if hi_max is not None and hi_max > hi:
        decades = np.geomspace(hi, hi_max,
                               max(int(np.log10(hi_max / hi)
                                       * n_per_decade_ext), 2))
        grid = np.concatenate([grid, decades[1:]])
    if kink is not None and lo < kink < grid[-1]:
        grid = np.append(grid, kink)
    grid = np.unique(grid)

    vals = np.asarray(f(grid), dtype=float)
    if np.isnan(vals).any():
        # Truncate at the first NaN (e.g. inf - inf in a divergent tail);
        # everything before it is still scanned.
        first = int(np.argmax(np.isnan(vals)))
        if first == 0:
            raise FloatingPointError('closure returned NaN at the scan start')
        grid, vals = grid[:first], vals[:first]

    roots = list(grid[vals == 0.0])

    def _bracketed(g, v):
        found = []
        s = np.sign(v)
        for i in np.where(s[:-1] * s[1:] < 0)[0]:
            found.append(brentq(f, g[i], g[i + 1], xtol=1e-6, rtol=1e-12))
        return found

    roots += _bracketed(grid, vals)

    # Fold-pair refinement: a local minimum of |f| with no adjacent sign
    # change can hide a tangency / close root pair inside one cell.
    av = np.abs(vals)
    s = np.sign(vals)
    interior = np.arange(1, len(grid) - 1)
    is_min = (av[interior] < av[interior - 1]) & (av[interior] <= av[interior + 1])
    same_sign = (s[interior] == s[interior - 1]) & (s[interior] == s[interior + 1])
    nonzero = vals[interior] != 0.0
    for j in interior[is_min & same_sign & nonzero]:
        sub = np.geomspace(grid[j - 1], grid[j + 1], refine)
        subvals = np.asarray(f(sub), dtype=float)
        roots += list(sub[subvals == 0.0])
        roots += _bracketed(sub, subvals)

    roots.sort()
    merged = []
    for r in roots:
        if not merged or abs(r - merged[-1]) > 1e-9 * max(abs(r), 1.0):
            merged.append(r)
    return merged


def closure_slope(f, root, *, kink=None, lo=None, rel_h=1e-4):
    """One-sided secant estimate of df/dLt at a root.

    Sided away from `kink` when the root is within 2h of it, so the estimate
    never straddles a point where f is continuous but not differentiable
    (at the mixed/glacial seam the glacial side can have a divergent
    derivative, which would otherwise invert the stability flag).
    """
    h = abs(root) * rel_h
    a, b = root - h, root + h
    if kink is not None and abs(root - kink) <= 2.0 * h:
        if root <= kink:
            a, b = root - 2.0 * h, root - h
        else:
            a, b = root + h, root + 2.0 * h
    if lo is not None and a < lo:
        a = lo
    return (f(b) - f(a)) / (b - a)


# ---------------------------------------------------------------------------
# Shared result containers
# ---------------------------------------------------------------------------

@dataclass
class Solution:
    """A single steady-state solution.

    ``stable`` reflects the sign of d(zELA)/d(Lt) at the root: a glaciated
    state is stable when cooling (lowering zELA) grows the glacier, i.e.
    d(zELA)/d(Lt) < 0. The warm saddle of a fold has the opposite sign. The
    fluvial state, when viable, is always stable.
    """
    regime: str            # 'fluvial' | 'mixed' | 'glacial'
    Lt: float              # glacier terminus position [m]; NaN for fluvial
    zt: float              # terminus elevation [m]; NaN for fluvial
    zo: float              # divide elevation [m]
    stable: bool


@dataclass
class AARResult:
    """Steady-state accumulation-area ratio and its ingredients.

    Two surface closures fix the ELA crossing u_ELA = x_ELA / Lt and hence the
    catchment accumulation fraction fc = u_ELA^d (see ``GeneralProfile.aar``):

    - ``surface='powerlaw'``: u_ELA^k = lam            -> fc = lam^(d/k)
    - ``surface='gsurface'`` : G(x_ELA) = (1-lam) G(xo) solved on the consistent
      glacial shape -> fc = u_ELA^d

    For a fully glacial state the ice footprint is truncated at base level
    (x = L): the ablation integral stops there, and a crossing at or beyond
    L means the whole on-orogen glacier accumulates (aar = 1).

    ``eta_bar`` is the mean glacierized fraction of the below-ELA swath
    (A_a divided by the swath catchment area). Ice cannot be wider than its
    valley, so eta_bar <= 1 physically; a value above 1 means the ice-width
    closure (alpha_g * kH) and the Hack closure (kh) are mutually
    inconsistent for these parameters and the set should be rejected
    (model paper, ``sec:aar`` / ``eq:floor``) — ``aar`` warns when this
    happens.
    """
    surface: str           # 'powerlaw' | 'gsurface'
    aar: float             # A_c / (A_c + A_a)
    u_ela: float           # x_ELA / Lt
    fc: float              # catchment accumulation fraction u_ela^d
    N: float               # scale group (alpha_g kH / kh) Lt^(Lambda+1-d)
    Aa_over_Ac: float      # ablation / accumulation area ratio
    eta_bar: float         # glacierized fraction of below-ELA swath (<= 1)
