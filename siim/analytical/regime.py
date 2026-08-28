r"""Nondimensional regime map for the coupled glacial-fluvial steady state.

The steady state of the coupled profile is controlled by two dimensionless
numbers (theory paper, *Coupled glacial-fluvial steady state*): the fluvial
and glacial erosion numbers :math:`N_f` and :math:`N_g`. This module works in
the derived coordinates used for regime diagrams,

.. math::

   \kappa = N_g/N_f, \qquad Y = z_{\rm ELA}/z_{fo},

with all lengths in units of the orogen length L (so ``xo`` here is
:math:`x_o/L` and ``Lt`` is :math:`L_t/L`), and all elevations in units of
the reference fluvial relief :math:`z_{fo} = k_s L^{1-d\theta} F(x_o/L)`.

:class:`RegimeMap` evaluates, over arrays of :math:`(\kappa, Y)`, the same
continuous closure that :class:`siim.analytical.profiles.GeneralProfile`
solves per point in physical units:

.. math::

   Y(L_t';\,\kappa)\,F(x_o') =
   \begin{cases}
      F(L_t') + (\kappa/\kappa_c)\, L_t'^{\,r}\, G(x_o'/L_t')
          & L_t' \le 1 \text{ (mixed)} \\[2pt]
      \kappa\, L_t'^{\,r}\left[(1/\kappa_c)\, G(x_o'/L_t') - G(1/L_t')\right]
          & L_t' > 1 \text{ (glacial)}
   \end{cases}

with :math:`r = (1-d\phi)/(1+\phi)` and
:math:`\kappa_c = 1/(1-\lambda)`. For the default exponent-derived
:math:`\lambda`, :math:`\kappa_c = (k+d\sigma)/k`. The two branches join
continuously at :math:`L_t' = 1`. The shape functions are evaluated through
the analytically continued incomplete beta of
:mod:`siim.analytical.core`, so the marginal-Coulomb case
:math:`d\phi = 1` (and :math:`d\theta = 1`) needs no special casing — the
kernel's exact ``b = 0`` logarithmic branch reproduces the
:math:`\operatorname{arcosh}` closed forms.

Because the closure is *linear in* :math:`\kappa` at fixed :math:`L_t'`, the
shape functions are evaluated once on a shared :math:`L_t'` grid; per-
:math:`\kappa` closure traces are then cheap vector operations. Array solves
locate roots by monotone-segment interpolation on that trace (accuracy set by
``n_trace``); size-1 inputs are solved trace-free by damped Newton from a
closed-form initial guess (machine precision at the root, and cheap enough
that consumers may construct a fresh map per evaluation).

Everything here is numpy/scipy only (no model stack, no matplotlib).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import scipy.special as _sp

from .core import incomplete_beta_compl

__all__ = ['RegimeMap', 'SaddleNodeBoundary']

# Default extent of the fully-glaciated tail of the closure trace, in units
# of L. Cells whose cold root lies beyond this return NaN (deep-cold corner
# of a diagram; extend Lt_max if you need it).
_LT_MAX = 50.0


@dataclass
class SaddleNodeBoundary:
    """Parametric saddle-node (fold) trace of the partial-glacial closure.

    The locus in :math:`(\\kappa, Y)` where the warm and cold mixed roots
    merge — the warm boundary of the bistable strip on a regime diagram.
    Sampled parametrically in the terminus position ``Lt`` (in units of L);
    the first entry is the exact :math:`L_t' \\to x_o'` corner
    :math:`(\\kappa, Y, z_o/z_{fo}) = (0, 1, 1)`.
    """
    kappa: np.ndarray          # fold position kappa(Lt)
    Y: np.ndarray              # fold position Y(Lt) = zELA/zfo
    zo: np.ndarray             # divide elevation zo/zfo on the fold
    Lt: np.ndarray             # parameter: terminus position Lt/L


class RegimeMap:
    r"""Steady-state regime map in :math:`(\kappa, Y)` coordinates.

    Parameters
    ----------
    d : float, optional
        Hack-area exponent. Default 2.0.
    sigma : float, optional
        Hack-flux exponent for ice-flux integration; enters via
        :math:`\lambda = d\sigma/(d\sigma+k)`. Default 0.5.
    phi : float, optional
        Glacial concavity index, in [0, 1). Default 0.5 (Coulomb).
    theta : float, optional
        Fluvial concavity index. Default 0.5.
    k : float, optional
        Ice-accumulation shape exponent. Default 1.0.
    xo : float, optional
        Channel-head cutoff :math:`x_o/L`, in (0, 1). Default 1e-3.
    lam : float, optional
        AAR-like ratio override. Defaults to :math:`d\sigma/(d\sigma+k)`.
    Lt_max : float, optional
        Far end of the fully-glaciated closure trace, in units of L.
        Default 50.
    n_trace : int, optional
        Shared-grid resolution of the closure trace, built lazily on the
        first array solve (size-1 solves are trace-free Newton).
        Default 4096.

    Attributes
    ----------
    kappa_c : float
        Critical steepness ratio :math:`1/(1-\lambda)`. With the default
        exponent-derived :math:`\lambda`, this is
        :math:`(k + d\sigma)/k`.
    alpha : float
        Geometric cutoff factor :math:`G(x_o')/F(x_o')` — the slope of the
        cold (fully-glaciated) regime boundary :math:`Y = \alpha\kappa/\kappa_c`.
    r : float
        Glacial length exponent :math:`(1-d\phi)/(1+\phi)`.
    lam : float
        AAR-like elevation-interval ratio :math:`\lambda`.

    Notes
    -----
    The array methods (:meth:`Lt`, :meth:`zo`, :meth:`zt`, :meth:`masks`)
    return the **cold branch**: the largest-:math:`L_t'` root of the
    continuous closure — the most-glaciated stable state, the one a cooling
    history lands on. ``branch='warm'`` returns the smallest root instead
    (the small-glacier branch followed when warming out of a bistable
    state). Where no glaciated root exists the state is fluvial:
    ``Lt`` is NaN and ``zo`` is 1.
    """

    def __init__(self, *, d=2.0, sigma=0.5, phi=0.5, theta=0.5, k=1.0,
                 xo=1e-3, lam=None, Lt_max=_LT_MAX, n_trace=4096):
        if not 0.0 <= phi < 1.0:
            raise ValueError(
                f'phi must be in [0, 1) — the glacial shape integral '
                f'diverges at the channel head otherwise — got {phi}')
        if k <= 0:
            raise ValueError(f'k must be positive, got {k}')
        if d <= 0:
            raise ValueError(f'd must be positive, got {d}')
        if theta < 0:
            raise ValueError(f'theta must be >= 0, got {theta}')
        if not 0.0 < xo < 1.0:
            raise ValueError(f'xo (= x_o/L) must be in (0, 1), got {xo}')
        if Lt_max <= 1.0:
            raise ValueError(f'Lt_max must exceed 1, got {Lt_max}')

        self.d = float(d)
        self.sigma = float(sigma)
        self.phi = float(phi)
        self.theta = float(theta)
        self.k = float(k)
        self.xo = float(xo)
        self.Lt_max = float(Lt_max)

        self.dtheta = self.d * self.theta
        self.dphi = self.d * self.phi
        self.r = (1.0 - self.dphi) / (1.0 + self.phi)
        self.lam = (float(lam) if lam is not None
                    else self.d * self.sigma / (self.d * self.sigma + self.k))
        if not 0.0 <= self.lam < 1.0:
            raise ValueError(f'lam must be in [0, 1), got {self.lam}')
        # Preserve the exponent-derived default exactly, while allowing an
        # explicit AAR-like ratio to define the corresponding closure scale.
        self.kappa_c = ((self.k + self.d * self.sigma) / self.k
                        if lam is None else 1.0 / (1.0 - self.lam))

        # beta-kernel shape parameters of G
        self._a = 1.0 - self.phi
        self._b = (1.0 - self.dphi) / self.k

        self.F_xo = float(self.F(self.xo))
        self.alpha = self._G_scalar(self.xo) / self.F_xo

        # The shared closure trace is built lazily (first array solve /
        # extrema query): scalar solves use a trace-free Newton, so
        # constructing a RegimeMap is cheap — important for consumers that
        # rebuild the map per evaluation (e.g. a critical-wedge solve, where
        # xo/L varies continuously along the root-find).
        self._n_trace = int(n_trace)
        self._Lt_grid = None
        self._A = None
        self._B = None

    def _ensure_trace(self):
        """Build the shared closure-trace grid on first use.

        Log-concentrated toward both ends of the mixed interval (xo, 1) —
        the closure is steep near xo and folds can sit near either end —
        plus a geometric extension over the glacial tail (1, Lt_max], with
        Lt = 1 an exact node (the C0 kink).
        """
        if self._Lt_grid is not None:
            return
        half = max(self._n_trace // 3, 64)
        lo = self.xo + (1.0 - self.xo) * np.geomspace(1e-9, 0.5, half)
        hi = 1.0 - (1.0 - self.xo) * np.geomspace(0.5, 1e-9, half)
        glacial = np.geomspace(1.0, self.Lt_max, half)
        self._Lt_grid = np.unique(np.concatenate([lo, hi, [1.0], glacial]))
        # kappa-independent pieces: Y * F_xo = A(Lt) + kappa * B(Lt)
        self._A, self._B = self._closure_pieces(self._Lt_grid)

    # ------------------------------------------------------------------
    # Shape functions
    # ------------------------------------------------------------------

    def F(self, u):
        r"""Dimensionless fluvial profile shape :math:`F(u)`, u = x/L.

        .. math::

           F(u) = \mathcal{B}(1-u;\,1,\,1-d\theta)
                = \frac{1 - u^{1-d\theta}}{1-d\theta}
                \;\xrightarrow{\;d\theta\to 1\;}\; \ln(1/u)

        Evaluated through ``expm1`` so the :math:`d\theta = 1` logarithmic
        limit is reached continuously, with no branch tolerance. Vectorizes
        over u.
        """
        u = np.asarray(u, dtype=float)
        e = 1.0 - self.dtheta
        with np.errstate(divide='ignore'):
            logu = np.log(u)
        if e == 0.0:
            out = -logu
        else:
            out = -np.expm1(e * logu) / e
        return float(out) if np.ndim(u) == 0 else out

    def G(self, u):
        r"""Dimensionless glacial profile shape :math:`G(u)`, u = x/Lt.

        .. math::

           G(u) = \tfrac{1}{k}\,
                  \mathcal{B}\!\left(1-u^k;\,1-\phi,\,\tfrac{1-d\phi}{k}\right)

        evaluated through the eps-form of the analytically continued
        incomplete beta (exact for u << 1 and at the marginal case
        :math:`d\phi = 1`, where b = 0). Vectorizes over u.
        """
        u = np.asarray(u, dtype=float)
        eps = np.clip(u, 0.0, 1.0) ** self.k
        out = incomplete_beta_compl(eps, 1.0 - self.phi,
                                    (1.0 - self.dphi) / self.k) / self.k
        return float(out) if np.ndim(u) == 0 else out

    def _G_scalar(self, u):
        """Scalar G(u) without the array kernel's overhead (hot path of the
        scalar solves). Closed forms where they are exact: the marginal
        Coulomb case a = 1/2, b = 0 is (2/k) arcosh(u^(-k/2)) (pinned
        against the kernel by the test suite); b > 0 is scipy's scalar
        betainc. Anything else falls back to the kernel."""
        if u >= 1.0:
            return 0.0
        if self._b == 0.0 and self._a == 0.5:
            return (2.0 / self.k) * math.acosh(u ** (-self.k / 2.0))
        if self._b > 0.0:
            eps = u ** self.k
            return (_sp.beta(self._a, self._b)
                    * (1.0 - _sp.betainc(self._b, self._a, eps))) / self.k
        return float(self.G(u))

    # ------------------------------------------------------------------
    # Closure
    # ------------------------------------------------------------------

    def _closure_pieces(self, Lt):
        r"""kappa-independent decomposition Y(Lt; kappa) F(xo) = A + kappa B.

        Mixed branch (Lt <= 1): A = F(Lt), B = Lt^r G(xo/Lt) / kappa_c.
        Glacial branch (Lt > 1): A = 0,
        B = Lt^r [G(xo/Lt)/kappa_c - G(1/Lt)].
        Continuous at Lt = 1 (F(1) = 0 = G(1)).
        """
        Lt = np.asarray(Lt, dtype=float)
        G_head = self.G(self.xo / Lt)
        Ltr = Lt ** self.r
        A = np.where(Lt <= 1.0, self.F(np.minimum(Lt, 1.0)), 0.0)
        B = np.where(Lt <= 1.0,
                     Ltr * G_head / self.kappa_c,
                     Ltr * (G_head / self.kappa_c
                            - self.G(np.minimum(1.0 / Lt, 1.0))))
        return A, B

    def closure_Y(self, Lt, kappa):
        r"""The continuous closure :math:`Y(L_t'; \kappa)`.

        The climate :math:`Y = z_{\rm ELA}/z_{fo}` for which ``Lt`` is a
        steady-state terminus at steepness ratio ``kappa``. This is the
        dimensionless form of
        :meth:`siim.analytical.profiles.GeneralProfile._zELA_of_Lt`.
        Broadcasts over ``Lt`` and ``kappa``.
        """
        Lt = np.asarray(Lt, dtype=float)
        A, B = self._closure_pieces(Lt)
        out = (A + np.asarray(kappa, dtype=float) * B) / self.F_xo
        return float(out) if out.ndim == 0 else out

    def _dclosure_dLt(self, Lt, kappa):
        r"""Exact :math:`\partial Y/\partial L_t'` of the closure.

        Mixed: :math:`F_{xo}\,Y' = -L_t^{-d\theta}
        + (\kappa/\kappa_c) L_t^{r-1} [\,r G(u) + u^{1-d\phi}(1-u^k)^{-\phi}]`
        with u = xo/Lt. Glacial: same structure with the
        :math:`G(1/L_t)` term subtracted (u2 = 1/Lt).
        """
        Lt = np.asarray(Lt, dtype=float)
        kappa = np.asarray(kappa, dtype=float)
        u1 = self.xo / Lt
        # r*G(u) - u*G'(u), the bracket of d/dLt [Lt^r G(c/Lt)]
        D1 = (self.r * self.G(u1)
              + u1 ** (1.0 - self.dphi) * (1.0 - u1 ** self.k) ** (-self.phi))
        mixed = Lt <= 1.0
        with np.errstate(divide='ignore', invalid='ignore'):
            dmix = (-np.minimum(Lt, 1.0) ** (-self.dtheta)
                    + (kappa / self.kappa_c) * Lt ** (self.r - 1.0) * D1)
            u2 = np.minimum(1.0 / Lt, 1.0 - 1e-300)
            D2 = (self.r * self.G(u2)
                  + u2 ** (1.0 - self.dphi) * (1.0 - u2 ** self.k) ** (-self.phi))
            dglac = kappa * Lt ** (self.r - 1.0) * (D1 / self.kappa_c - D2)
        out = np.where(mixed, dmix, dglac) / self.F_xo
        return float(out) if out.ndim == 0 else out

    # ------------------------------------------------------------------
    # Root solves
    # ------------------------------------------------------------------

    def _F_scalar(self, u):
        """Scalar F(u) in plain math (hot path)."""
        e = 1.0 - self.dtheta
        if e == 0.0:
            return -math.log(u)
        return -math.expm1(e * math.log(u)) / e

    def _f_fp_scalar(self, Lt, kappa, Y):
        """Closure residual and its exact dLt-derivative at a scalar point,
        sharing the G evaluations between the two (the hot path of
        :meth:`_solve_scalar`)."""
        if Lt <= 1.0:
            u = self.xo / Lt
            Gu = self._G_scalar(u)
            pw = u ** (1.0 - self.dphi) * (1.0 - u ** self.k) ** (-self.phi)
            f = (self._F_scalar(Lt)
                 + (kappa / self.kappa_c) * Lt ** self.r * Gu) / self.F_xo - Y
            fp = (-Lt ** (-self.dtheta)
                  + (kappa / self.kappa_c) * Lt ** (self.r - 1.0)
                  * (self.r * Gu + pw)) / self.F_xo
        else:
            u1 = self.xo / Lt
            u2 = 1.0 / Lt
            G1 = self._G_scalar(u1)
            G2 = self._G_scalar(u2)
            pw1 = u1 ** (1.0 - self.dphi) * (1.0 - u1 ** self.k) ** (-self.phi)
            pw2 = u2 ** (1.0 - self.dphi) * (1.0 - u2 ** self.k) ** (-self.phi)
            bracket = G1 / self.kappa_c - G2
            f = kappa * Lt ** self.r * bracket / self.F_xo - Y
            fp = (kappa * Lt ** (self.r - 1.0)
                  * (self.r * bracket + pw1 / self.kappa_c - pw2)) / self.F_xo
        return f, fp

    def _newton_scalar(self, x, lo, hi, kappa, Y, max_iter=30):
        """Damped Newton on the scalar closure, clipped to [lo, hi]; returns
        NaN unless the residual converges (the no-root signal)."""
        for _ in range(max_iter):
            f, fp = self._f_fp_scalar(x, kappa, Y)
            if f == 0.0:
                return x
            step = -f / fp if fp != 0.0 else 0.0
            cap = 0.3 * x
            if step > cap:
                step = cap
            elif step < -cap:
                step = -cap
            x_new = x + step
            if not math.isfinite(x_new):
                return float('nan')
            if x_new < lo:
                x_new = lo
            elif x_new > hi:
                x_new = hi
            if abs(x_new - x) <= 1e-15 * max(abs(x), 1.0):
                x = x_new
                break
            x = x_new
        f, _ = self._f_fp_scalar(x, kappa, Y)
        if abs(f) > 1e-6 * (abs(Y) + 1.0):
            return float('nan')
        return x

    def _solve_scalar(self, kappa, Y, branch):
        """Trace-free cold-branch solve for one (kappa, Y).

        Regime dispatch + damped Newton from a closed-form initial guess —
        the same convention the figure code historically used (the guess
        lands in the cold root's basin); a failed residual means no root
        (fluvial). The warm branch needs the global trace and routes through
        the array machinery.
        """
        kappa = float(kappa)
        Y = float(Y)
        if branch == 'warm':
            self._ensure_trace()
            out = self._solve_grid(np.asarray([kappa]), np.asarray([Y]),
                                   'warm')
            return float(out[0])

        L2 = self.alpha * kappa / self.kappa_c
        if Y < L2 and kappa > 0:
            # fully-glaciated branch (Lt > 1)
            out = self._newton_scalar(1.5, 1.0 + 1e-9, self.Lt_max,
                                      kappa, Y)
            if math.isnan(out):
                out = self._scan_newton_scalar(1.0 + 1e-9, self.Lt_max,
                                               kappa, Y, branch)
            return out
        # NB: no `Y > 1 and kappa >= kappa_c -> NaN` early return here (audit m8):
        # the mixed trace rises just above Y = 1 off the Lt = xo corner, so a thin
        # warm sliver of glaciated roots exists there for kappa >= kappa_c. The
        # mixed-branch Newton + coarse-scan fallback below finds that near-corner
        # root (agreeing with the array path / GeneralProfile) and returns NaN
        # only where the scan genuinely finds none (true fluvial-only).
        # mixed branch: closed-form-style initial guess Lt = xo^tau
        denom = self.kappa_c - kappa
        if abs(denom) < 1e-9:
            x = 0.5 * (self.xo + 1.0)
        else:
            tau = (self.kappa_c * Y - self.alpha * kappa) / denom
            tau = min(max(tau, 1e-3), 1.0 - 1e-3)
            x = self.xo ** tau
        lo = self.xo * (1.0 + 1e-9)
        hi = 1.0 - 1e-12
        x = min(max(x, lo), hi)
        out = self._newton_scalar(x, lo, hi, kappa, Y)
        # The closed-form guess can land in the wrong fold basin (e.g. a small
        # near-corner glacier behind an interior maximum): Newton then either
        # fails (NaN) or converges onto the warm root, which sits on the
        # ASCENDING side of the fold (dY/dLt > 0). The cold root is the
        # largest-Lt root and always lies on a descending segment
        # (dY/dLt < 0) — so re-solve by a coarse log scan (trace-free) when
        # Newton failed or landed on an ascending root; the scan's last
        # bracket picks the largest root for the cold branch (audit B1).
        wrong_side = (not math.isnan(out)
                      and self._f_fp_scalar(out, kappa, Y)[1] > 0.0)
        if math.isnan(out) or wrong_side:
            out = self._scan_newton_scalar(lo, hi, kappa, Y, branch)
        return out

    def _scan_newton_scalar(self, lo, hi, kappa, Y, branch, n=96):
        """Trace-free fallback bracketing: scan the closure on [lo, hi] with a
        grid log-concentrated toward BOTH ends (the shared trace's own
        structure — folds pinch near the channel head xo and near the Lt = 1
        kink, so a uniform log scan misses a tightly-packed near-corner root
        pair, e.g. the warm/cold pair just above xo in the bistable strip),
        then guarded Newton inside the branch's bracketing cell."""
        half = max(n // 2, 32)
        frac = np.geomspace(1e-9, 0.5, half)
        span = hi - lo
        grid = np.unique(np.concatenate([lo + span * frac, hi - span * frac]))
        f = (self._A_B_eval(grid, kappa)) / self.F_xo - Y
        s = np.sign(f)
        idx = np.flatnonzero(s[:-1] * s[1:] < 0)
        if idx.size == 0:
            exact = grid[f == 0.0]
            if exact.size:
                return float(exact.max() if branch == 'cold' else exact.min())
            return float('nan')
        i = int(idx[-1] if branch == 'cold' else idx[0])
        return self._newton_scalar(
            math.sqrt(grid[i] * grid[i + 1]), grid[i], grid[i + 1], kappa, Y)

    def _A_B_eval(self, Lt, kappa):
        """A(Lt) + kappa*B(Lt) on an arbitrary Lt array (no shared trace)."""
        A, B = self._closure_pieces(np.asarray(Lt, dtype=float))
        return A + kappa * B

    def _solve_grid(self, kappa, Y, branch, n_refine=8):
        """Monotone-segment interpolation solve over broadcast arrays,
        polished by ``n_refine`` guarded vectorized Newton iterations on the
        exact closure (the interp guess is within a trace cell of the root,
        so Newton converges to machine precision)."""
        self._ensure_trace()
        K, YY = np.broadcast_arrays(np.asarray(kappa, dtype=float),
                                    np.asarray(Y, dtype=float))
        out = np.full(K.shape, np.nan)
        K_flat = K.ravel()
        Y_flat = YY.ravel()
        out_flat = out.ravel()

        uniq, inv = np.unique(K_flat, return_inverse=True)
        for ui, kap in enumerate(uniq):
            cells = inv == ui
            if not np.isfinite(kap):
                continue
            Y_tr = (self._A + kap * self._B) / self.F_xo
            # maximal monotone segments of the trace
            dY = np.diff(Y_tr)
            direction = np.sign(dY)
            direction[direction == 0] = 1
            breaks = np.where(np.diff(direction) != 0)[0] + 1
            starts = np.concatenate([[0], breaks])
            ends = np.concatenate([breaks, [len(Y_tr) - 1]])
            Yc = Y_flat[cells]
            best = np.full(Yc.shape, np.nan)
            for s0, s1 in zip(starts, ends):
                if s1 <= s0:
                    continue
                seg_Y = Y_tr[s0:s1 + 1]
                seg_L = self._Lt_grid[s0:s1 + 1]
                if seg_Y[1] < seg_Y[0]:        # descending -> flip
                    seg_Y = seg_Y[::-1]
                    seg_L = seg_L[::-1]
                inside = (Yc >= seg_Y[0]) & (Yc <= seg_Y[-1])
                if not inside.any():
                    continue
                cand = np.interp(Yc[inside], seg_Y, seg_L)
                cur = best[inside]
                if branch == 'cold':
                    take = np.isnan(cur) | (cand > cur)
                else:
                    take = np.isnan(cur) | (cand < cur)
                cur[take] = cand[take]
                best[inside] = cur
            out_flat[cells] = best

        # Polish on the exact closure, vectorized over all solved cells. The
        # interp guess sits within one trace cell of its root; use that cell as
        # a fixed BRACKET and run safeguarded Newton — bisect whenever a Newton
        # step would leave the (shrinking) bracket. Trace cells never straddle
        # the Lt = 1 kink (an exact grid node), so an iterate can never cross
        # the sqrt-singular seam and drift off down the wrong (mixed) branch:
        # a plain capped-Newton polish overshoots there (the near-vertical
        # glacial cusp) and walks away from the root (audit B2).
        ok = np.isfinite(out_flat)
        if ok.any() and n_refine > 0:
            x = out_flat[ok]
            kx = K_flat[ok]
            yx = Y_flat[ok]
            cell = np.clip(np.searchsorted(self._Lt_grid, x) - 1,
                           0, len(self._Lt_grid) - 2)
            a = self._Lt_grid[cell]              # fancy-index copies
            b = self._Lt_grid[cell + 1]
            fa = np.asarray(self.closure_Y(a, kx), dtype=float) - yx
            x = np.clip(x, a, b)
            for _ in range(n_refine):
                f = np.asarray(self.closure_Y(x, kx), dtype=float) - yx
                fp = np.asarray(self._dclosure_dLt(x, kx), dtype=float)
                # keep the root enclosed: replace the endpoint on x's side
                left = np.sign(f) == np.sign(fa)
                a = np.where(left, x, a)
                fa = np.where(left, f, fa)
                b = np.where(left, b, x)
                with np.errstate(divide='ignore', invalid='ignore'):
                    xn = x - f / fp
                bad = ~np.isfinite(xn) | (xn < a) | (xn > b)
                x = np.where(bad, 0.5 * (a + b), xn)
            out_flat[ok] = x
        return out

    def Lt(self, kappa, Y, *, branch='cold'):
        r"""Steady-state terminus position :math:`L_t'/L` over (kappa, Y).

        NaN where no glaciated steady state exists (fluvial), or where the
        cold root lies beyond ``Lt_max``. Size-1 inputs are solved to
        machine precision; arrays by trace interpolation (accuracy set by
        ``n_trace``).
        """
        if branch not in ('cold', 'warm'):
            raise ValueError(f"branch must be 'cold' or 'warm', got {branch!r}")
        if np.ndim(kappa) == 0 and np.size(Y) == 1:
            val = self._solve_scalar(kappa, np.asarray(Y).item(), branch)
            return np.asarray(val).reshape(np.shape(Y))
        return self._solve_grid(kappa, Y, branch)

    def zt(self, kappa, Y, *, branch='cold', Lt=None):
        r"""Terminus elevation :math:`z_t/z_{fo}`.

        :math:`F(L_t')/F(x_o')` on the mixed branch;
        :math:`-(\kappa/F(x_o'))\,L_t'^{\,r}\,G(1/L_t')` (below base level)
        on the glacial branch; NaN where fluvial. Pass ``Lt`` to reuse an
        already-computed terminus array.
        """
        if Lt is None:
            Lt = self.Lt(kappa, Y, branch=branch)
        if np.ndim(kappa) == 0 and np.size(Lt) == 1:      # scalar fast path
            Lt_s = float(np.asarray(Lt).reshape(()))
            return np.asarray(self._zt_scalar(float(kappa), Lt_s)
                              ).reshape(np.shape(Lt))
        shape = np.shape(Lt)
        Lt = np.atleast_1d(np.asarray(Lt, dtype=float))
        K = np.broadcast_to(np.asarray(kappa, dtype=float), Lt.shape)
        out = np.full(Lt.shape, np.nan)
        ok = np.isfinite(Lt)
        m = ok & (Lt <= 1.0)
        g = ok & (Lt > 1.0)
        if m.any():
            out[m] = self.F(Lt[m]) / self.F_xo
        if g.any():
            out[g] = -(K[g] / self.F_xo) * Lt[g] ** self.r * self.G(1.0 / Lt[g])
        return out.reshape(shape)

    def _zt_scalar(self, kappa, Lt):
        if not math.isfinite(Lt):
            return float('nan')
        if Lt <= 1.0:
            return self._F_scalar(Lt) / self.F_xo
        return -(kappa / self.F_xo) * Lt ** self.r * self._G_scalar(1.0 / Lt)

    def zo(self, kappa, Y, *, branch='cold', Lt=None):
        r"""Divide elevation :math:`z_o/z_{fo}`.

        :math:`z_o = z_t + (\kappa/F(x_o'))\,L_t'^{\,r}\,G(x_o'/L_t')` on a
        glaciated root, exactly as
        :meth:`~siim.analytical.profiles.GeneralProfile._make_solution`
        builds it; 1 where the state is fluvial.
        """
        if Lt is None:
            Lt = self.Lt(kappa, Y, branch=branch)
        if np.ndim(kappa) == 0 and np.size(Lt) == 1:      # scalar fast path
            Lt_s = float(np.asarray(Lt).reshape(()))
            if not math.isfinite(Lt_s):
                zo_s = 1.0
            else:
                zo_s = (self._zt_scalar(float(kappa), Lt_s)
                        + (float(kappa) / self.F_xo) * Lt_s ** self.r
                        * self._G_scalar(self.xo / Lt_s))
            return np.asarray(zo_s).reshape(np.shape(Lt))
        shape = np.shape(Lt)
        Lt = np.atleast_1d(np.asarray(Lt, dtype=float))
        K = np.broadcast_to(np.asarray(kappa, dtype=float), Lt.shape)
        zt = np.atleast_1d(self.zt(kappa, Y, branch=branch, Lt=Lt))
        out = np.ones(Lt.shape)
        ok = np.isfinite(Lt)
        if ok.any():
            out[ok] = (zt[ok] + (K[ok] / self.F_xo)
                       * Lt[ok] ** self.r * self.G(self.xo / Lt[ok]))
        return out.reshape(shape)

    def masks(self, kappa, Y, *, branch='cold'):
        """(fluvial, mixed, glacial) boolean masks from the cold-branch root.

        ``mixed`` where the root sits in (xo, 1]; ``glacial`` where it
        exceeds 1 (fully glaciated to base level); ``fluvial`` where no
        glaciated root exists. Note the bistable strip (kappa < kappa_c,
        Y > 1 below the saddle-node) reads as *mixed* under this convention
        — the diagram shows the glaciated attractor there; overlay
        :meth:`saddle_node` to mark the strip.
        """
        Lt = self.Lt(kappa, Y, branch=branch)
        Lt = np.asarray(Lt, dtype=float)
        glacial = np.isfinite(Lt) & (Lt > 1.0)
        mixed = np.isfinite(Lt) & ~glacial
        fluvial = ~np.isfinite(Lt)
        return fluvial, mixed, glacial

    # ------------------------------------------------------------------
    # Fold structure
    # ------------------------------------------------------------------

    def closure_extrema(self, kappa):
        """Interior (Y_max, Y_min) of the mixed-branch closure trace.

        The fold pair of the partial-glacial closure on (xo, 1): an interior
        maximum (fluvial/mixed bistability, the warm saddle) and/or an
        interior minimum (small/large-glacier bistability, the cooling
        spinodal). Either is NaN when the trace is monotone there.
        """
        self._ensure_trace()
        kappa = float(kappa)
        mix = self._Lt_grid <= 1.0
        Y_tr = (self._A[mix] + kappa * self._B[mix]) / self.F_xo
        n = Y_tr.size
        i_max = int(np.argmax(Y_tr))
        i_min = int(np.argmin(Y_tr))
        clip = max(3, n // 200)
        Y_max = Y_tr[i_max] if clip < i_max < n - clip else float('nan')
        Y_min = Y_tr[i_min] if clip < i_min < n - clip else float('nan')
        return float(Y_max), float(Y_min)

    def saddle_node(self, n_samples=400):
        r"""Parametric saddle-node (L1) boundary of the mixed closure.

        Setting :math:`\partial Y/\partial L_t' = 0` on the mixed branch
        gives, with :math:`u = x_o'/L_t'` and
        :math:`D(u) = r\,G(u) + u^{1-d\phi}(1-u^k)^{-\phi}`,

        .. math::

           \frac{\kappa}{\kappa_c} = \frac{L_t'^{\,1-r-d\theta}}{D(u)},

        and the fold's :math:`(Y, z_o/z_{fo})` follow from the closure and
        the profile geometry at that :math:`(L_t', \kappa)`. The trace is
        sampled with log concentration toward both endpoints and prepended
        with the exact corner :math:`(0, 1, 1)` at :math:`L_t' = x_o'`;
        in the marginal-Coulomb case it reduces to the
        :math:`\kappa/\kappa_c = \sqrt{1 - v^{-2}}`,
        :math:`v = (L_t'/x_o')^{k/2}` arcosh trace.

        Returns
        -------
        SaddleNodeBoundary
            Arrays (kappa, Y, zo, Lt) along the fold, ordered by Lt.
        """
        half = max(int(n_samples) // 2, 32)
        lo = self.xo + (1.0 - self.xo) * np.geomspace(1e-9, 0.5, half)
        hi = 1.0 - (1.0 - self.xo) * np.geomspace(0.5, 1e-6, half)
        Lts = np.unique(np.concatenate([lo, hi]))

        u = self.xo / Lts
        Gu = self.G(u)
        D = (self.r * Gu
             + u ** (1.0 - self.dphi) * (1.0 - u ** self.k) ** (-self.phi))
        with np.errstate(divide='ignore', invalid='ignore'):
            kr = Lts ** (1.0 - self.r - self.dtheta) / D      # kappa/kappa_c
        kappa = self.kappa_c * kr
        F_Lt = self.F(Lts)
        Y = (F_Lt + kr * Lts ** self.r * Gu) / self.F_xo
        zo = F_Lt / self.F_xo + self.kappa_c * kr * Lts ** self.r * Gu / self.F_xo

        return SaddleNodeBoundary(
            kappa=np.concatenate([[0.0], kappa]),
            Y=np.concatenate([[1.0], Y]),
            zo=np.concatenate([[1.0], zo]),
            Lt=np.concatenate([[self.xo], Lts]),
        )
