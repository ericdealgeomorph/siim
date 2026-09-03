"""Steady-state glacial-fluvial profile classes (steepness-index form).

Driven by ks, cs, zELA and L directly, not by the underlying erosion/sliding
parameters — uplift, erodibility, sliding law and mass balance are absorbed
into ks and cs.

``GeneralProfile`` solves the full coupled steady state for arbitrary
exponents (d, sigma, phi, theta, k). It computes the elevation
profile z(x), ice thickness H(x), reports the regime
(fluvial / mixed / glacial), and detects bistability by finding all roots of
the continuous zELA(Lt) closure.

``MarginalCoulombProfile`` solves the exact phi = theta = 1/2, d = 2 special
case in closed (arcosh) form. That marginal case (d*phi = 1) is
precisely where the general beta-function machinery is logarithmically
divergent, so this class doubles as the closed-form cross-check of
``GeneralProfile``: the two must agree in that limit.

Both classes share one ``Solution``/``AARResult`` vocabulary (siim.analytical
.core), the same constructor convention (``Class(params_dict)`` ==
``Class(**params_dict)``), the same argument order, and the package-wide
constants from ``siim.constants``. matplotlib is imported lazily inside
``plot``; importing this module needs numpy/scipy only.
"""
from __future__ import annotations

from typing import List, Optional
import math
import warnings

import numpy as np
import scipy.special as _sp
from scipy.optimize import brentq

from ..constants import (ALPHA_G, AC, BETA, HC_OVER_H, KH, KT, LAMBDA_C,
                         RHO_ICE_G, TAU_C)
from .core import (AARResult, Solution, _arccosh_safe, closure_slope,
                   find_closure_roots, incomplete_beta_compl)


def _require(cond, msg):
    if not cond:
        raise ValueError(msg)


def _validate_common(obj):
    """Shared parameter validation (both profile classes)."""
    _require(obj.ks > 0, f'ks must be > 0, got {obj.ks}')
    # cs <= 0 previously fabricated a 'stable glacial' state from garbage.
    _require(obj.cs > 0, f'cs must be > 0, got {obj.cs}')
    _require(math.isfinite(obj.zELA), f'zELA must be finite, got {obj.zELA}')
    _require(math.isfinite(obj.L) and obj.L > 0,
             f'L must be positive and finite, got {obj.L}')
    _require(obj.sigma >= 0, f'sigma must be >= 0, got {obj.sigma}')
    _require(obj.kh > 0, f'kh must be > 0, got {obj.kh}')
    _require(obj.alpha_g > 0, f'alpha_g must be > 0, got {obj.alpha_g}')


# ---------------------------------------------------------------------------
# Shared machinery: solution bookkeeping, profile sampling, plotting.
# Module-level functions assigned as methods on both classes, so each lives
# once. The class-specific physics enters through the `_z_fluvial(x)` /
# `_z_glacial(x, sol)` seam and the d/gamma/k attributes.
# ---------------------------------------------------------------------------

def _finalize_solutions(obj):
    """Set primary/bistable/regime/Lt/zt/zo from obj.solutions.

    primary = largest-Lt stable solution (fluvial Lt = NaN ranks below any
    glacier); bistable = more than one stable steady state.
    """
    stable = [s for s in obj.solutions if s.stable]
    obj.bistable = len(stable) > 1

    if stable:
        def _key(s):
            return -1.0 if math.isnan(s.Lt) else s.Lt
        obj.primary = max(stable, key=_key)
    elif obj.solutions:
        obj.primary = obj.solutions[0]
    else:
        obj.primary = None

    if obj.primary is not None:
        obj.regime = obj.primary.regime
        obj.Lt = obj.primary.Lt
        obj.zt = obj.primary.zt
        obj.zo = obj.primary.zo
    else:
        obj.regime = None
        obj.Lt = float('nan')
        obj.zt = float('nan')
        obj.zo = float('nan')


def _profile(self, n_points: int = 3000, *, solution: Optional[Solution] = None):
    """Surface elevation profile (x, z) for the chosen solution.

    For mixed solutions x is concentrated such that both the glacial and
    fluvial sections are well resolved.

    Parameters
    ----------
    n_points : int, optional
        Number of sample points (mixed profiles keep at least 1000 per
        section). Default 3000.
    solution : Solution, optional
        Steady state to sample. Defaults to ``primary``.

    Returns
    -------
    x : ndarray
        Distance from the divide [m], geometrically spaced on [xo, L].
    z : ndarray
        Surface elevation above base level [m], same shape as ``x``.
        Both arrays are empty when no steady state exists.
    """
    sol = solution if solution is not None else self.primary
    if sol is None:
        return np.array([]), np.array([])

    if sol.regime == 'fluvial':
        x = np.geomspace(self.xo, self.L, n_points)
        return x, self._z_fluvial(x)

    if sol.regime == 'mixed':
        n_g = max(int(n_points * sol.Lt / self.L), 1000)
        n_f = max(n_points - n_g, 1000)
        x_g = np.geomspace(self.xo, sol.Lt, n_g)
        x_f = np.geomspace(sol.Lt, self.L, n_f)
        z_g = self._z_glacial(x_g, sol)
        z_f = self._z_fluvial(x_f)
        x = np.concatenate([x_g[:-1], x_f])
        z = np.concatenate([z_g[:-1], z_f])
        return x, z

    # Fully glacial (Lt > L): glacial branch spans [xo, L].
    x = np.geomspace(self.xo, self.L, n_points)
    return x, self._z_glacial(x, sol)


def _thickness(self, x, *, solution: Optional[Solution] = None):
    r"""Ice thickness profile for the chosen solution.

    .. math::

       H(x) = z_{\rm ELA}\, N_H \left[\bigl(1 - (x/L_t)^k\bigr)\,
       (x/L_t)^d\right]^{\gamma}

    Parameters
    ----------
    x : array_like
        Distance from the divide [m].
    solution : Solution, optional
        Steady state to sample. Defaults to ``primary``.

    Returns
    -------
    ndarray
        Width-mean ice thickness [m], same shape as ``x``; zero outside
        the ice extent and identically zero for fluvial (or absent)
        solutions.
    """
    sol = solution if solution is not None else self.primary
    x = np.asarray(x, dtype=float)
    H = np.zeros_like(x)
    if sol is None or sol.regime == 'fluvial':
        return H
    Lt = sol.Lt
    NH_val = self.NH(Lt)
    if not math.isfinite(NH_val):
        return H
    if sol.regime == 'mixed':
        mask = (x > self.xo) & (x <= Lt)
    else:  # glacial: ice spans the whole orogen up to base level
        mask = (x > self.xo) & (x <= self.L)
    ratio = np.clip(x[mask] / Lt, 0.0, 1.0)
    shape = (1.0 - ratio ** self.k) * ratio ** self.d
    H[mask] = self.zELA * NH_val * np.power(np.maximum(shape, 0.0), self.gamma)
    return H


def _plot(self, ax=None, *, solution: Optional[Solution] = None,
          n_points: int = 3000, show_ela: bool = True,
          show_thickness: bool = True, show_bedrock: bool = True,
          topo_color='k', ice_color='#b0d6e2', ice_alpha: float = 0.55,
          bedrock_color='#888888', bedrock_fade='auto',
          bedrock_alpha: float = 0.6,
          label: Optional[str] = None):
    """Plot the steady-state surface and (optionally) the ice column.

    Parameters
    ----------
    ax : matplotlib.axes.Axes, optional
        Axis to draw into; a new 8x3-inch figure is created when None.
    solution : Solution, optional
        Steady state to draw. Defaults to ``primary``.
    n_points : int, optional
        Sample count for the surface profile. Default 3000.
    show_ela : bool, optional
        Draw the ELA as a dashed horizontal red line. Default True.
    show_thickness : bool, optional
        Fill the ice column (bed to surface) and draw the sub-ice bed
        line. Default True.
    show_bedrock : bool, optional
        Render the bedrock fill below the bed surface. Default True.
    topo_color : color, optional
        Line color for the topographic surface and sub-ice bed.
        Default ``'k'``.
    ice_color : color, optional
        Fill color of the ice column. Default ``'#b0d6e2'``.
    ice_alpha : float, optional
        Opacity of the ice fill. Default 0.55.
    bedrock_color : color, optional
        Fill color of the bedrock. Default ``'#888888'``.
    bedrock_fade : float | 'auto' | None
        Vertical e-folding scale [m] for the bedrock alpha fade below the
        bedrock surface. ``'auto'`` picks 0.25 * relief. ``None`` falls back
        to a flat fill from a baseline up to the surface.
    bedrock_alpha : float
        Maximum opacity of the bedrock right at the surface. Default 0.6.
    label : str, optional
        Legend label for the surface line.

    Returns
    -------
    matplotlib.axes.Axes
        The axis drawn into (unchanged when there is no solution to draw).

    Notes
    -----
    The bed under ice is reconstructed on the channel-floor datum,
    ``zb = z - HC_OVER_H * H``; H is the width-mean depth and is not
    rescaled. See ``docs/guides/concepts.md`` for the public datum convention.
    """
    import matplotlib.pyplot as plt  # lazy: keep the module headless-importable

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 3))

    sol = solution if solution is not None else self.primary
    x, z = self.profile(n_points=n_points, solution=sol)
    if x.size == 0:
        return ax

    H = self.thickness(x, solution=sol)
    # Channel-floor datum: the bed under ice sits a full centerline depth
    # hc = HC_OVER_H * H below the surface (H is the width-mean depth, the
    # driver — it is NOT rescaled; only the bed reconstruction is).
    zb = z - HC_OVER_H * H

    # H -> 0 at the shape's endpoints, so the sub-ice bed line is bounded by
    # the glacier's geometric extent rather than an H > 0 mask.
    has_glacier = sol.regime in ('mixed', 'glacial')
    ice_end = sol.Lt if sol.regime == 'mixed' else self.L
    draw_profile(ax, x, z, zb,
                 ice_x_range=(self.xo, ice_end) if has_glacier else None,
                 topo_color=topo_color, ice_color=ice_color,
                 ice_alpha=ice_alpha, bedrock_color=bedrock_color,
                 bedrock_fade=bedrock_fade, bedrock_alpha=bedrock_alpha,
                 show_bedrock=show_bedrock,
                 show_thickness=show_thickness and has_glacier, label=label)

    if show_ela:
        ax.axhline(self.zELA, color='r', lw=0.8, ls='--', zorder=-1)

    ax.set_xlabel('Distance from divide (km)')
    ax.set_ylabel('Elevation (m)')
    ax.set_xlim(0, self.L / 1e3)
    return ax


def _draw_bedrock(ax, x, zb, z, *, color, fade, alpha_max):
    """Render the bedrock under zb: solid fill if fade is None, else an imshow
    with alpha decaying as exp(-depth/fade) below the surface."""
    from matplotlib.colors import to_rgb

    if fade is None:
        ymin_fill = min(float(zb.min()), 0.0)
        ax.fill_between(x / 1e3, ymin_fill, zb, color=color,
                        alpha=alpha_max, zorder=0, linewidth=0)
        return

    if fade == 'auto':
        relief = float(z.max() - zb.min())
        fade = max(0.25 * relief, 50.0)

    # Resample bedrock onto a regular x grid for imshow.
    nx_img = 800
    x_reg = np.linspace(float(x.min()), float(x.max()), nx_img)
    zb_reg = np.interp(x_reg, x, zb)

    ny_img = 500
    ymax = float(zb_reg.max())
    # 4 * fade -> alpha ~ e^-4 ~ 0.018 at the bottom (barely visible).
    # Using 6 * fade pushes the y-axis autoscale unnecessarily low.
    ymin = float(zb_reg.min()) - 4.0 * fade
    y_grid = np.linspace(ymin, ymax, ny_img)

    depth = zb_reg[None, :] - y_grid[:, None]   # >0 below the surface
    # exp only where depth > 0 (a full-array exp overflows above the surface)
    alpha = alpha_max * np.where(depth > 0,
                                 np.exp(-np.maximum(depth, 0.0) / fade), 0.0)

    rgba = np.empty((ny_img, nx_img, 4))
    rgba[..., :3] = to_rgb(color)
    rgba[..., 3] = alpha

    ax.imshow(rgba,
              extent=(x_reg.min() / 1e3, x_reg.max() / 1e3, ymin, ymax),
              aspect='auto', origin='lower', zorder=0,
              interpolation='bilinear', rasterized=True)


def draw_profile(ax, x, surface, bed, *, ice_x_range=None,
                 topo_color='k', ice_color='#b0d6e2', ice_alpha=0.55,
                 bedrock_color='#888888', bedrock_fade='auto',
                 bedrock_alpha=0.6, show_bedrock=True, show_thickness=True,
                 label=None):
    """Render a steady-state profile in the paper's canonical visual language:
    a depth-faded bedrock fill below ``bed``, the ice column shaded between
    ``bed`` and ``surface``, the topographic surface line, and the sub-ice bed
    line. The single source of truth behind both ``GeneralProfile.plot`` and
    ``analytical_steady_state_solution.plot``.

    ``x``, ``surface``, ``bed`` are in metres (``x`` strictly increasing;
    plotted in km). ``bed`` is the channel-floor datum (``surface -
    HC_OVER_H*H`` under ice, ``surface`` where ice-free). ``ice_x_range`` =
    ``(x_lo, x_hi)`` in metres bounds the sub-ice bed line — H -> 0 at the
    glacier endpoints, so a geometric extent draws cleaner than an ``H > 0``
    mask; ``None`` draws it wherever ``surface > bed``. ``show_thickness`` is
    a no-op on a fully fluvial profile (no cells with ``surface > bed``).
    """
    if show_bedrock:
        _draw_bedrock(ax, x, bed, surface, color=bedrock_color,
                      fade=bedrock_fade, alpha_max=bedrock_alpha)

    # Topography surface (rock where exposed, ice where glaciated).
    ax.plot(x / 1e3, surface, '-', color=topo_color, lw=0.7, label=label)

    if show_thickness:
        has_ice = surface > bed + 1e-9
        if has_ice.any():
            ax.fill_between(x / 1e3, bed, surface, where=has_ice,
                            color=ice_color, alpha=ice_alpha, linewidth=0)
            if ice_x_range is not None:
                bed_mask = (x >= ice_x_range[0]) & (x <= ice_x_range[1])
            else:
                bed_mask = has_ice
            ax.plot(x[bed_mask] / 1e3, bed[bed_mask], '-',
                    color=topo_color, lw=0.8)
    return ax


def _repr(self):
    name = type(self).__name__
    bist = ' [bistable]' if self.bistable else ''
    if self.primary is None:
        return f'<{name} no_solution>'
    Lt_str = 'NaN' if math.isnan(self.Lt) else f'{self.Lt:.0f}m'
    return (f'<{name} regime={self.regime}{bist} '
            f'Lt={Lt_str} zo={self.zo:.0f}m '
            f'kappa={self.kappa:.3f} kappa_c={self.kappa_c:.3f} '
            f'zELA/zfo={self.zELA / self.zfo:.3f}>')


# ---------------------------------------------------------------------------
# General solution (arbitrary exponents)
# ---------------------------------------------------------------------------

class GeneralProfile:
    """General steady-state coupled glacial-fluvial profile (steepness-in).

    Parameters
    ----------
    ks, cs : float
        Fluvial and glacial steepness indices. ks has units m^(d*theta); cs has
        units m^(1-r). They roll up uplift, erodibility and climate.
    zELA : float
        Equilibrium-line altitude [m].
    L : float
        Orogen half-width / base-level distance [m].
    xo : float, optional
        Hillslope / channel-head cutoff [m]. Defaults to L/1000. Enters G(xo)
        in both closures and the reference relief zfo.
    d : float, optional
        Hack-area exponent (A = kh * x^d). Default 2.0.
    sigma : float, optional
        Hack-flux exponent for ice-flux integration. Enters only via
        lam = d*sigma/(d*sigma + k). Default 0.5.
    phi : float, optional
        Glacial concavity index (mu/nu). Coulomb -> 1/2, power-law -> 4/15.
        Default 0.5.
    theta : float, optional
        Fluvial concavity index (m/n). Default 0.5.
    k : float, optional
        Ice-accumulation profile shape exponent
        (Beff ~ 1 - (x/Lt)^k). Default 1.0 (the linear-ansatz baseline).
    lam : float, optional
        AAR-like ratio. Defaults to d*sigma/(d*sigma + k); override to decouple.
    kH : float, optional
        Ice-thickness prefactor in H = kH * Lt^(g(1+d)/(1+phi)) * shape^g.
        Model-agnostic; tune to taste. Default 0.05.
    gamma : float, optional
        Ice-thickness shape exponent. Defaults to phi (Coulomb closure);
        power-law uses (2/3)(1/3 + phi).
    alpha_g : float, optional
        Valley width-to-thickness ratio (W = alpha_g H). Sets the AAR ablation
        area; only the group alpha_g/kh matters. Default constants.ALPHA_G
        (currently 5).
    kh : float, optional
        Hack coefficient (catchment area A = kh x^d). Sets the AAR accumulation
        area. Default constants.KH (5).

    Attributes
    ----------
    solutions : list[Solution]
        All steady states found (stable branches + warm saddles).
    primary : Solution
        Largest-Lt stable solution (fluvial Lt = NaN ranks below any glacier).
        Drives ``profile``, ``thickness``, ``plot`` and the convenience attrs.
    bistable : bool
        True when more than one stable steady state exists.
    """

    def __init__(self, ks=None, cs=None, zELA=None, L=None, *,
                 xo=None, d=2.0, sigma=0.5, phi=0.5, theta=0.5, k=1.0,
                 lam=None, kH=0.05, gamma=None, alpha_g=ALPHA_G, kh=KH):
        # Allow a single dict as the first positional arg, matching the sibling
        # class: GeneralProfile(params) == GeneralProfile(**params).
        if isinstance(ks, dict):
            params = dict(ks)
            ks = params.pop('ks', None)
            cs = params.pop('cs', cs)
            zELA = params.pop('zELA', zELA)
            L = params.pop('L', L)
            xo = params.pop('xo', xo)
            d = params.pop('d', d)
            sigma = params.pop('sigma', sigma)
            phi = params.pop('phi', phi)
            theta = params.pop('theta', theta)
            k = params.pop('k', k)
            lam = params.pop('lam', lam)
            kH = params.pop('kH', kH)
            gamma = params.pop('gamma', gamma)
            alpha_g = params.pop('alpha_g', alpha_g)
            kh = params.pop('kh', kh)
            if params:
                raise TypeError(f'unknown parameter(s): {sorted(params)}')

        missing = [n for n, v in [('ks', ks), ('cs', cs),
                                   ('zELA', zELA), ('L', L)] if v is None]
        if missing:
            raise TypeError(f'missing required parameter(s): {missing}')

        if xo is None:
            xo = L / 1000.0
        if not 0 < xo < L:
            raise ValueError(f'xo ({xo}) must be in (0, L={L})')
        if k <= 0:
            raise ValueError(f'k ({k}) must be positive')

        self.ks = float(ks)
        self.cs = float(cs)
        self.zELA = float(zELA)
        self.L = float(L)
        self.xo = float(xo)
        self.d = float(d)
        self.sigma = float(sigma)
        self.phi = float(phi)
        self.theta = float(theta)
        self.k = float(k)
        self.kH = float(kH)
        # Geometry prefactors that set the AAR scale group N (see ``aar``).
        # alpha_g: valley width-to-thickness ratio (W = alpha_g H).
        # kh:      Hack coefficient (catchment area A = kh x^d).
        self.alpha_g = float(alpha_g)
        self.kh = float(kh)
        # Optional overrides retained so solve() can re-derive them (m27).
        self._lam_arg = lam
        self._gamma_arg = gamma

        self.solve()

    def solve(self):
        """Re-derive every downstream quantity from the current raw parameter
        attributes and re-run the steady-state solve. Called by ``__init__``;
        call it again after mutating a parameter attribute (e.g. ``self.zELA``)
        to refresh the object (construct-then-mutate-then-``solve()``;
        idempotent; audit m27)."""
        self.gamma = self.phi if self._gamma_arg is None else float(self._gamma_arg)
        self.dtheta = self.d * self.theta
        # d*phi = 1 (the marginal-Coulomb singularity) is handled exactly by
        # the kernel's b = 0 logarithmic branch — no nudge.
        self.dphi = self.d * self.phi
        self.r = (1.0 - self.dphi) / (1.0 + self.phi)

        self.lam = (float(self._lam_arg) if self._lam_arg is not None
                    else self.d * self.sigma / (self.d * self.sigma + self.k))

        _validate_common(self)
        _require(self.d > 0, f'd must be > 0, got {self.d}')
        _require(0.0 <= self.phi < 1.0,
                 f'phi must be in [0, 1) — the glacial shape integral '
                 f'diverges at the channel head otherwise — got {self.phi}')
        _require(self.theta >= 0, f'theta must be >= 0, got {self.theta}')
        _require(self.kH >= 0, f'kH must be >= 0, got {self.kH}')
        _require(self.gamma >= 0, f'gamma must be >= 0, got {self.gamma}')
        _require(0.0 <= self.lam < 1.0,
                 f'lam must be in [0, 1), got {self.lam}')

        self.solutions = self._find_solutions()
        _finalize_solutions(self)
        return self

    # ------------------------------------------------------------------
    # Dimensionless groups
    # ------------------------------------------------------------------

    @property
    def Nf(self):
        r"""Fluvial erosion number.

        .. math::

           N_f = \frac{k_s\, L^{1-d\theta}}{z_{\rm ELA}}

        +inf at zELA = 0 (ELA at base level).
        """
        if self.zELA == 0.0:
            return float('inf')
        return self.ks * self.L ** (1.0 - self.dtheta) / self.zELA

    @property
    def Ng(self):
        r"""Glacial erosion number.

        .. math::

           N_g = \frac{c_s\, L^{r}}{z_{\rm ELA}}, \qquad
           r = \frac{1 - d\phi}{1 + \phi}

        +inf at zELA = 0.
        """
        if self.zELA == 0.0:
            return float('inf')
        return self.cs * self.L ** self.r / self.zELA

    @property
    def kappa(self):
        """Master control ratio kappa = Ng / Nf = cs L^r / (ks L^(1-d*theta)).
        Equals cs/ks only in the marginal-Coulomb case (where r = 1-d*theta = 0).
        zELA cancels in the ratio, so kappa stays finite at zELA = 0."""
        return (self.cs * self.L ** self.r
                / (self.ks * self.L ** (1.0 - self.dtheta)))

    @property
    def kappa_c(self):
        r"""Critical steepness ratio
        :math:`\kappa_c = 1/(1-\lambda)`.

        The critical value of ``kappa`` in the zELA(Lt) closure (theory
        paper, *Coupled glacial-fluvial steady state*). With the default
        exponent-derived ``lam``, this is :math:`(k+d\sigma)/k`; an explicit
        ``lam`` override supplies the corresponding closure scale directly.
        """
        if self._lam_arg is None:
            return (self.k + self.d * self.sigma) / self.k
        return 1.0 / (1.0 - self.lam)

    @property
    def zfo(self):
        """Reference fluvial relief: divide elevation of the unglaciated orogen."""
        return float(self._z_fluvial(self.xo))

    # ------------------------------------------------------------------
    # Profile shape functions
    # ------------------------------------------------------------------

    def _z_fluvial(self, x):
        """Fluvial surface relief above base level, z(x) - z(L) = ks L^(1-dt) F(x).

        F(x) = B(1 - x/L; 1, 1 - d*theta) has the exact closed form
        (1 - (x/L)^(1-d*theta))/(1 - d*theta), or ln(L/x) when d*theta = 1.
        """
        x = np.asarray(x, dtype=float)
        if abs(1.0 - self.dtheta) < 1e-12:
            return self.ks * np.log(self.L / x)
        e = 1.0 - self.dtheta
        return self.ks * self.L ** e * (1.0 - (x / self.L) ** e) / e

    def _z_glacial(self, x, sol):
        """Glacial surface elevation z(x) = zt + cs * Lt^r * G(x; Lt)."""
        return sol.zt + self.cs * sol.Lt ** self.r * self._G(x, sol.Lt)

    def _G(self, x, Lt):
        """Glacial shape G(x; Lt) = (1/k) B(1 - (x/Lt)^k; 1 - phi, (1-d*phi)/k).

        The relief between x and the terminus is cs * Lt^r * G(x; Lt).
        Evaluated through the eps-form of the analytically continued
        incomplete beta (eps = (x/Lt)^k), exact arbitrarily deep into the
        x << Lt tail — including the marginal case d*phi = 1 (b = 0).
        Vectorizes over x or Lt.
        """
        x = np.asarray(x, dtype=float)
        if np.any(x > np.asarray(Lt) * (1.0 + 1e-12)):
            raise ValueError('G(x; Lt) evaluated beyond the terminus (x > Lt)')
        eps = np.minimum(x / Lt, 1.0) ** self.k
        out = incomplete_beta_compl(eps, 1.0 - self.phi,
                                    (1.0 - self.dphi) / self.k) / self.k
        if np.ndim(x) == 0 and np.ndim(Lt) == 0:
            return float(out)
        return out

    # ------------------------------------------------------------------
    # Closure: one continuous zELA(Lt) across mixed and glacial regimes
    # ------------------------------------------------------------------

    def _zELA_of_Lt(self, Lt):
        r"""Climate zELA for which Lt is a steady-state terminus.

        .. math::

           z_{\rm ELA}(L_t) =
           \begin{cases}
              z_{\rm fluvial}(L_t) + (1-\lambda)\, c_s L_t^{r}\, G(x_o; L_t)
                  & L_t \le L \text{ (mixed)} \\
              c_s L_t^{r} \left[(1-\lambda)\, G(x_o; L_t) - G(L; L_t)\right]
                  & L_t > L \text{ (glacial)}
           \end{cases}

        The two branches join continuously at Lt = L (both F(L) and G(L; L)
        vanish there), so this is a single continuous function of Lt.
        Vectorizes over Lt.
        """
        Lt = np.asarray(Lt, dtype=float)
        scalar = Lt.ndim == 0
        Lt = np.atleast_1d(Lt)
        G_xo = self._G(self.xo, Lt)
        cgrel = self.cs * Lt ** self.r
        out = np.empty_like(Lt)
        m = Lt <= self.L
        if m.any():
            out[m] = (np.asarray(self._z_fluvial(Lt[m]), dtype=float)
                      + (1.0 - self.lam) * cgrel[m] * G_xo[m])
        if (~m).any():
            out[~m] = cgrel[~m] * ((1.0 - self.lam) * G_xo[~m]
                                   - self._G(self.L, Lt[~m]))
        return float(out[0]) if scalar else out

    def _make_solution(self, Lt):
        """Build a Solution (regime, zt, zo, stability) for a glaciated root."""
        if Lt <= self.L:
            zt = float(self._z_fluvial(Lt))
            zo = zt + self.cs * Lt ** self.r * self._G(self.xo, Lt)
            regime = 'mixed'
        else:
            zt = -self.cs * Lt ** self.r * self._G(self.L, Lt)
            zo = zt + self.cs * Lt ** self.r * self._G(self.xo, Lt)
            regime = 'glacial'

        # Stability from the fold slope: stable when cooling grows the
        # glacier. One-sided away from the Lt = L kink, where the glacial
        # side's divergent derivative would otherwise invert the flag.
        slope = closure_slope(self._zELA_of_Lt, Lt, kink=self.L,
                              lo=self.xo * 1.0001)
        return Solution(regime, Lt, zt, zo, stable=slope < 0.0)

    def _find_solutions(self) -> List[Solution]:
        # Glaciated roots: all roots of the continuous closure
        # zELA(Lt) = zELA, dense on [xo, 10 L] (with Lt = L an exact grid
        # node) and decade-extended to 1e8 L for far-out glacial states.
        def f(Lts):
            return self._zELA_of_Lt(Lts) - self.zELA

        lo = self.xo * 1.0001
        roots = find_closure_roots(f, lo, self.L * 10.0, kink=self.L,
                                   n=4000, hi_max=self.L * 1e8)
        out = [self._make_solution(Lt) for Lt in roots if Lt > self.xo]

        # Fluvial state: viable when the unglaciated divide sits at/below the ELA.
        if self.zELA >= self.zfo:
            out.append(Solution('fluvial', float('nan'), float('nan'),
                                self.zfo, stable=True))

        out.sort(key=lambda s: (math.isnan(s.Lt), s.Lt))
        return out

    # ------------------------------------------------------------------
    # Ice-thickness scale
    # ------------------------------------------------------------------

    def NH(self, Lt: Optional[float] = None) -> float:
        r"""Dimensionless ice-thickness scale
        :math:`H/z_{\rm ELA} = (k_H/z_{\rm ELA})\, L_t^{\Lambda}`.

        Returns
        -------
        float
            The H/zELA scale; NaN when there is no glacier (Lt non-finite
            or Lt <= xo).
        """
        if Lt is None:
            Lt = self.Lt
        if not math.isfinite(Lt) or Lt <= self.xo:
            return float('nan')
        return (self.kH / self.zELA) * Lt ** self.Lambda

    # ------------------------------------------------------------------
    # Accumulation-area ratio
    # ------------------------------------------------------------------

    @property
    def Lambda(self):
        """Ice-thickness size exponent Lambda = gamma (1+d)/(1+phi).

        (Same exponent that drives ``NH``; named to match the AAR derivation.)
        """
        return self.gamma * (1.0 + self.d) / (1.0 + self.phi)

    def _u_ela_k(self, Lt, surface):
        """u_ELA^k = (x_ELA / Lt)^k for the requested surface closure.

        'powerlaw': closed form u_ELA^k = lam (z ~ zo - (zo-zt) u^k).
        'gsurface': solve the consistent shape G(x_ELA; Lt) = (1-lam) G(xo; Lt)
                    for x_ELA, then u_ELA^k = (x_ELA/Lt)^k. This is the *exact*
                    crossing of the modelled surface at self.zELA (it reproduces
                    where ``profile`` hits zELA), keeping the xo head offset via
                    G(xo). The xo-neglected quantile form
                    u_ELA^k = 1 - I^{-1}_{1-lam}(1-phi, (1-dphi)/k) is the same in
                    the xo -> 0 limit (and only when (1-dphi)/k > 0); the head
                    offset is not always negligible, so the root-find is preferred
                    and also covers the nudged marginal case.
        """
        if surface == 'powerlaw':
            return self.lam
        if surface != 'gsurface':
            raise ValueError(f"surface must be 'powerlaw' or 'gsurface', got {surface!r}")
        G0 = self._G(self.xo, Lt)
        target = (1.0 - self.lam) * G0
        lo = self.xo
        hi = Lt * (1.0 - 1e-9)
        x_ela = brentq(lambda x: self._G(x, Lt) - target, lo, hi,
                       xtol=1e-6, rtol=1e-12)
        return (x_ela / Lt) ** self.k

    def _aar_from_uk(self, Lt, uk, surface, alpha_g, kh):
        """Assemble an ``AARResult`` from u_ELA^k for one surface closure.

        The ablation integral runs over the on-orogen ice footprint only,
        u^k from uk to min(1, (L/Lt)^k): a fully glacial state is truncated
        at base level, and an ELA crossing at/beyond the footprint end means
        the whole on-orogen glacier accumulates (aar = 1).

        A_c is the *full* catchment at the crossing, kh * x_ELA^d. The
        head-referencing that is essential for the closure (G evaluated at
        xo — z diverges toward x = 0 in the marginal cases) does not apply
        to the area bookkeeping: area stays finite, and the above-ELA
        catchment delivers its accumulation to the network in full.
        """
        a = (self.d * self.gamma + 1.0) / self.k
        b = self.gamma + 1.0
        u_ela = uk ** (1.0 / self.k)
        fc = uk ** (self.d / self.k)                      # = u_ELA^d
        N = (alpha_g * self.kH / kh) * Lt ** (self.Lambda + 1.0 - self.d)
        uk_max = min(1.0, (self.L / Lt) ** self.k)        # base-level truncation
        if uk >= uk_max:
            return AARResult(surface=surface, aar=1.0, u_ela=u_ela,
                             fc=fc, N=N, Aa_over_Ac=0.0, eta_bar=0.0)
        # B(a,b)[I_c(uk) - I_c(uk_max)]: upper incomplete beta over the
        # footprint. betaincc keeps lam -> 1 (uk -> 1) from cancelling to 0.
        upper_inc = _sp.beta(a, b) * (_sp.betaincc(a, b, uk)
                                      - _sp.betaincc(a, b, uk_max))
        # Consistency check (model paper, AAR appendix): the glacierized
        # fraction of the below-ELA swath cannot exceed 1 — ice can't be
        # wider than its valley.
        swath = uk_max ** (self.d / self.k) - fc          # (u_max^d - u_ELA^d)
        eta_bar = (N * upper_inc / (self.k * swath)) if swath > 0 else float('inf')
        if eta_bar > 1.0:
            warnings.warn(
                f'aar ({surface}): eta_bar = {eta_bar:.3g} > 1 — the ice '
                'footprint exceeds its valley, i.e. the ice-width closure '
                '(alpha_g*kH) and the Hack closure (kh) are mutually '
                'inconsistent for these parameters; reject or retune.',
                UserWarning, stacklevel=3)
        if fc == 0.0:
            # sigma = 0 / lam = 0: the crossing collapses to the divide.
            return AARResult(surface=surface, aar=0.0, u_ela=u_ela,
                             fc=fc, N=N, Aa_over_Ac=float('inf'),
                             eta_bar=eta_bar)
        Aa_over_Ac = N * upper_inc / (self.k * fc)        # A_a / A_c
        aar = fc / (fc + N * upper_inc / self.k)
        return AARResult(surface=surface, aar=aar, u_ela=u_ela,
                         fc=fc, N=N, Aa_over_Ac=Aa_over_Ac, eta_bar=eta_bar)

    def aar(self, surface='gsurface', *, solution=None, alpha_g=None, kh=None):
        r"""Steady-state accumulation-area ratio AAR = A_c / (A_c + A_a).

        Accumulation area is the full catchment above the ELA,
        :math:`A_c = k_h x_{\rm ELA}^d`; ablation area is the ice footprint
        below the ELA,

        .. math::

           A_a = \frac{\alpha_g k_H}{k}\, L_t^{\Lambda+1}\, B(a, b)
           \left[1 - I_{u_{\rm ELA}^k}(a, b)\right], \qquad
           a = \frac{d\gamma + 1}{k},\quad b = \gamma + 1.

        Both reduce to a single number once u_ELA is fixed by the surface
        closure (see ``AARResult``).

        Parameters
        ----------
        surface : {'gsurface', 'powerlaw', 'both'}
            Which ELA-crossing closure to use. Default 'gsurface' (the exact
            in-model G(xo) crossing). 'powerlaw' reads the crossing from the
            power-law accumulation ansatz (u_ELA^k = lam); the ansatz serves
            only to close the accumulation integral and misplaces the
            crossing, so treat it as a comparison tool, not a result. 'both'
            returns a dict keyed by both labels.
        solution : Solution, optional
            Which steady state to use. Defaults to ``primary``.
        alpha_g, kh : float, optional
            Override the stored width-to-thickness ratio / Hack coefficient.

        Returns
        -------
        AARResult, or {'powerlaw': AARResult, 'gsurface': AARResult}.
        ``aar`` is NaN for a fluvial (unglaciated) state.
        """
        sol = solution if solution is not None else self.primary
        alpha_g = self.alpha_g if alpha_g is None else float(alpha_g)
        kh = self.kh if kh is None else float(kh)

        def _nan(label):
            return AARResult(label, float('nan'), float('nan'), float('nan'),
                             float('nan'), float('nan'), float('nan'))

        Lt = float('nan') if sol is None else sol.Lt
        viable = sol is not None and sol.regime in ('mixed', 'glacial') \
            and math.isfinite(Lt) and Lt > self.xo

        labels = ('powerlaw', 'gsurface') if surface == 'both' else (surface,)
        results = {}
        for label in labels:
            if not viable:
                results[label] = _nan(label)
            else:
                uk = self._u_ela_k(Lt, label)
                results[label] = self._aar_from_uk(Lt, uk, label, alpha_g, kh)

        return results if surface == 'both' else results[labels[0]]

    # ------------------------------------------------------------------
    # Shared presentation methods (module-level implementations above)
    # ------------------------------------------------------------------

    profile = _profile
    thickness = _thickness
    plot = _plot
    __repr__ = _repr


def sweep(ks, cs, zELA, L, *, cls=GeneralProfile, **fixed):
    """Solve a grid of steady states (regime diagrams, paper figures).

    Broadcasts ks, cs, zELA and L against each other (numpy rules) and
    constructs ``cls`` at every grid point; parameters held fixed across the
    sweep go in ``fixed``. Returns a dict of arrays in the broadcast shape:

    - ``'regime'``   : int8 code, -1 none / 0 fluvial / 1 mixed / 2 glacial
    - ``'Lt', 'zt', 'zo'`` : primary-solution values (NaN where absent)
    - ``'n_stable'`` : number of stable steady states at that point
    - ``'bistable'`` : bool

    plus ``'regime_codes'`` mapping the names to the codes. Example::

        out = sweep(ks=120.0, cs=np.linspace(50, 400, 80),
                    zELA=np.linspace(0, 2500, 120)[:, None], L=1e5)
        plt.pcolormesh(..., out['regime'])
    """
    codes = {None: -1, 'fluvial': 0, 'mixed': 1, 'glacial': 2}
    b = np.broadcast(*(np.asarray(v, dtype=float)
                       for v in (ks, cs, zELA, L)))
    shape = b.shape
    out = {
        'regime': np.full(shape, -1, dtype=np.int8),
        'Lt': np.full(shape, np.nan),
        'zt': np.full(shape, np.nan),
        'zo': np.full(shape, np.nan),
        'n_stable': np.zeros(shape, dtype=np.int16),
        'bistable': np.zeros(shape, dtype=bool),
    }
    for idx, (ks_i, cs_i, z_i, L_i) in zip(np.ndindex(shape), b):
        p = cls(ks_i, cs_i, z_i, L_i, **fixed)
        out['regime'][idx] = codes[p.regime]
        out['Lt'][idx], out['zt'][idx], out['zo'][idx] = p.Lt, p.zt, p.zo
        out['n_stable'][idx] = sum(s.stable for s in p.solutions)
        out['bistable'][idx] = p.bistable
    out['regime_codes'] = {name: c for name, c in codes.items() if name}
    return out


# ---------------------------------------------------------------------------
# Marginal-Coulomb solution (exact arcosh closed form)
# ---------------------------------------------------------------------------

class MarginalCoulombProfile:
    """Steady-state marginal-Coulomb profile (phi = theta = 1/2, d = 2).

    The exact closed (arcosh) form of the case d*phi = 1, where the general
    beta-function machinery is logarithmically divergent — which makes this
    class the closed-form cross-check of ``GeneralProfile`` in that limit.

    Parameters
    ----------
    ks, cs : float
        Fluvial and glacial steepness indices [m].
    zELA : float
        Equilibrium-line altitude [m].
    L : float
        Orogen half-width [m].
    sigma : float, optional
        Hack-flux exponent for ice-flux integration. Default 0.5.
    xo : float, optional
        Hillslope cutoff [m]. Defaults to L/1000.
    k : float, optional
        Ice-flux profile exponent (Beff(x) = beta*(zo-zELA)*[1-(x/Lt)^k]).
        Default 1.0 (the linear-ansatz baseline, matching GeneralProfile).
    Ac : float, optional
        Glen's flow-law coefficient A [Pa^-3 s^-1]; the deformation prefactor
        2A/5 is applied internally. Default constants.AC (2.5e-24).
    alpha_g : float, optional
        Valley width-to-thickness ratio. Default constants.ALPHA_G (currently 5).
    tau_c : float, optional
        Coulomb yield stress [Pa]. Default constants.TAU_C (1e5).
    lam_c : float, optional
        Coulomb sliding length [m]. Default constants.LAMBDA_C (1e3).
    beta : float, optional
        Mass-balance gradient [yr^-1]. Default constants.BETA (1e-2).
    kh : float, optional
        Hack's-law prefactor (A = kh * x^d). Default constants.KH (5).

    Attributes
    ----------
    solutions : list[Solution]
        All steady states found (stable + the warm saddle if it exists).
    primary : Solution
        The cold/largest-Lt stable solution. Used by ``profile``,
        ``thickness``, ``plot`` and the ``Lt``, ``zt``, ``zo``, ``regime``
        convenience attributes.
    bistable : bool
        True when more than one stable steady state exists.
    """

    def __init__(self, ks=None, cs=None, zELA=None, L=None, *, sigma=0.5,
                 xo=None, k=1.0,
                 Ac=AC, alpha_g=ALPHA_G, tau_c=TAU_C,
                 lam_c=LAMBDA_C, beta=BETA, kh=KH):
        # Allow a single dict of all parameters as the first positional arg:
        #   MarginalCoulombProfile(params) == MarginalCoulombProfile(**params).
        if isinstance(ks, dict):
            params = dict(ks)
            ks = params.pop('ks', None)
            cs = params.pop('cs', cs)
            zELA = params.pop('zELA', zELA)
            L = params.pop('L', L)
            sigma = params.pop('sigma', sigma)
            xo = params.pop('xo', xo)
            k = params.pop('k', k)
            Ac = params.pop('Ac', Ac)
            alpha_g = params.pop('alpha_g', alpha_g)
            tau_c = params.pop('tau_c', tau_c)
            lam_c = params.pop('lam_c', lam_c)
            beta = params.pop('beta', beta)
            kh = params.pop('kh', kh)
            if params:
                raise TypeError(f'unknown parameter(s): {sorted(params)}')

        missing = [n for n, v in [('ks', ks), ('cs', cs),
                                   ('zELA', zELA), ('L', L)] if v is None]
        if missing:
            raise TypeError(f'missing required parameter(s): {missing}')

        if xo is None:
            xo = L / 1000.0
        if not 0 < xo < L:
            raise ValueError(f'xo ({xo}) must be in (0, L={L})')
        if k <= 0:
            raise ValueError(f'k ({k}) must be positive')

        # Fixed exponents of the marginal-Coulomb special case.
        self.d = 2.0
        self.phi = 0.5
        self.theta = 0.5
        self.gamma = 0.5

        self.ks = float(ks)
        self.cs = float(cs)
        self.zELA = float(zELA)
        self.L = float(L)
        self.sigma = float(sigma)
        self.xo = float(xo)
        self.k = float(k)
        self.Ac = float(Ac)
        self.alpha_g = float(alpha_g)
        self.tau_c = float(tau_c)
        self.lam_c = float(lam_c)
        self.beta = float(beta)
        self.kh = float(kh)

        self.solve()

    def solve(self):
        """Re-run the marginal-Coulomb solve from the current raw parameter
        attributes. Called by ``__init__``; call it again after mutating a
        parameter attribute (e.g. ``self.zELA``) to refresh the object
        (construct-then-mutate-then-``solve()``; idempotent; audit m27)."""
        _validate_common(self)
        _require(self.beta > 0, f'beta must be > 0, got {self.beta}')
        _require(self.tau_c > 0, f'tau_c must be > 0, got {self.tau_c}')
        _require(self.lam_c >= 0, f'lam_c must be >= 0, got {self.lam_c}')
        _require(self.Ac > 0, f'Ac must be > 0, got {self.Ac}')

        self.solutions = self._find_solutions()
        _finalize_solutions(self)
        return self

    # ------------------------------------------------------------------
    # Dimensionless groups
    # ------------------------------------------------------------------

    @property
    def kappa(self):
        r"""Master control ratio :math:`\kappa = c_s/k_s`.

        In the marginal-Coulomb case both length exponents vanish
        (r = 1 - d*theta = 0), so kappa is the bare steepness ratio.
        """
        return self.cs / self.ks

    @property
    def kappa_c(self):
        r"""Critical steepness ratio
        :math:`\kappa_c = (k + d\sigma)/k = 1/(1-\lambda)`."""
        return (self.k + self.d * self.sigma) / self.k

    @property
    def lam(self):
        r"""Flux-partition ratio :math:`\lambda = d\sigma/(d\sigma + k)`,
        the fraction of accumulation-area ice flux already committed at the
        ELA crossing (fixed by the exponents in this class)."""
        return self.d * self.sigma / (self.d * self.sigma + self.k)

    @property
    def zfo(self):
        """Reference fluvial relief: ks * log(L/xo)."""
        return self.ks * math.log(self.L / self.xo)

    @property
    def zgo(self):
        """Reference glacial-to-base-level relief: cs * log(2^(2/k) * L/xo)."""
        return self.cs * math.log(2.0 ** (2.0 / self.k) * self.L / self.xo)

    # ------------------------------------------------------------------
    # Closure: one continuous zELA(Lt) across mixed and glacial regimes
    # ------------------------------------------------------------------

    def _zELA_of_Lt(self, Lt):
        r"""Climate zELA for which Lt is a steady-state terminus (theory
        paper, *Special solution: critical Coulomb*).

        .. math::

           z_{\rm ELA}(L_t) =
           \begin{cases}
              k_s \ln(L/L_t) + (c_s/\kappa_c)\, G(x_o; L_t) & L_t \le L \\
              c_s \left[G(x_o; L_t)/\kappa_c - G(L; L_t)\right] & L_t > L
           \end{cases}

        with the exact marginal-Coulomb shape
        :math:`G(x; L_t) = (2/k)\,\mathrm{arccosh}\bigl((L_t/x)^{k/2}\bigr)`
        on *both* branches, so the curve is continuous through the seam at
        Lt = L. (The old glacial closure used the asymptotic zgo form, which
        sat a few metres off the mixed branch at the seam and could demote
        the genuine root's stability.) Vectorizes over Lt.
        """
        Lt = np.asarray(Lt, dtype=float)
        scalar = Lt.ndim == 0
        Lt = np.atleast_1d(Lt)
        G_xo = (2.0 / self.k) * _arccosh_safe((Lt / self.xo) ** (self.k / 2.0))
        with np.errstate(invalid='ignore'):  # (Lt/L)^... only used where Lt > L
            G_L = (2.0 / self.k) * _arccosh_safe((Lt / self.L) ** (self.k / 2.0))
        out = np.where(Lt <= self.L,
                       self.ks * np.log(self.L / Lt)
                       + self.cs / self.kappa_c * G_xo,
                       self.cs * (G_xo / self.kappa_c - G_L))
        return float(out[0]) if scalar else out

    def _find_solutions(self) -> List[Solution]:
        # All roots of the one continuous closure (mixed and glacial are its
        # two halves), each with stability from the closure slope — replacing
        # the old "largest-Lt root is the stable cold branch" heuristic and
        # the separate asymptotic glacial Newton solve.
        def f(Lts):
            return self._zELA_of_Lt(Lts) - self.zELA

        lo = self.xo * 1.0001
        roots = find_closure_roots(f, lo, self.L * 10.0, kink=self.L,
                                   n=4000, hi_max=self.L * 1e8)

        branch: List[Solution] = []
        for Lt in roots:
            G_xo = (2.0 * self.cs / self.k) * math.acosh(
                max((Lt / self.xo) ** (self.k / 2.0), 1.0))
            if Lt <= self.L:
                regime = 'mixed'
                zt = self.ks * math.log(self.L / Lt)
            else:
                regime = 'glacial'
                zt = -(2.0 * self.cs / self.k) * math.acosh(
                    (Lt / self.L) ** (self.k / 2.0))
            zo = zt + G_xo
            slope = closure_slope(self._zELA_of_Lt, Lt, kink=self.L, lo=lo)
            branch.append(Solution(regime, Lt, zt, zo, stable=slope < 0.0))

        # Fluvial state exists when an unglaciated orogen's divide would
        # sit at or below the ELA.
        if self.zELA >= self.zfo:
            branch.append(Solution('fluvial', float('nan'), float('nan'),
                                   self.zfo, stable=True))

        branch.sort(key=lambda s: (math.isnan(s.Lt), s.Lt))
        return branch

    # ------------------------------------------------------------------
    # Profile shape (the _z_fluvial/_z_glacial seam used by ``profile``)
    # ------------------------------------------------------------------

    def _z_fluvial(self, x):
        """Fluvial surface elevation z(x) = ks * log(L/x)."""
        return self.ks * np.log(self.L / np.asarray(x, dtype=float))

    def _z_glacial(self, x, sol):
        """Glacial surface z(x) = zt + (2 cs / k) arccosh((Lt/x)^(k/2))."""
        return sol.zt + (2.0 * self.cs / self.k) * _arccosh_safe(
            (sol.Lt / np.asarray(x, dtype=float)) ** (self.k / 2.0))

    # ------------------------------------------------------------------
    # Ice-thickness scale (Coulomb closure)
    # ------------------------------------------------------------------

    def NH(self, Lt: Optional[float] = None) -> float:
        r"""Dimensionless ice-thickness scale H/zELA for given Lt.

        .. math::

           N_H = \frac{L_t}{z_{\rm ELA}} \left[
           \left(\frac{c_s}{\lambda_\tau}\right)^{3}
           + \frac{\lambda_c\, c_g\, \lambda_\tau^{2}}
                  {k_h\,\beta\,\lambda\,G_o} \right]^{-1/3}

        with :math:`c_g = k_t\,\alpha_g\,(2A_c/5)\,(\rho_i g)^3`,
        :math:`\lambda_\tau = \tau_c/(\rho_i g)`, and the marginal-Coulomb
        shape factor :math:`G_o = G(x_o; L_t) = B_o/k` where
        :math:`B_o = 2\,\mathrm{arccosh}\bigl((L_t/x_o)^{k/2}\bigr)` —
        matching the physical front end's coulomb NH (quadrature-verified;
        the old phi factor in place of the 1/k was a structural error,
        ~phi/k off).

        Returns
        -------
        float
            The H/zELA scale; NaN when there is no glacier (or when
            lam = 0, i.e. no ice flux).
        """
        if Lt is None:
            Lt = self.Lt
        if not math.isfinite(Lt) or Lt <= self.xo:
            return float('nan')
        Bo = 2.0 * math.acosh(max((Lt / self.xo) ** (self.k / 2.0), 1.0))
        if Bo == 0.0 or self.lam == 0.0:
            return float('nan')
        cg = KT * self.alpha_g * (2.0 * self.Ac / 5.0) * RHO_ICE_G ** 3
        lam_tau = self.tau_c / RHO_ICE_G
        Go = Bo / self.k
        denom = ((self.cs / lam_tau) ** 3
                 + self.lam_c * cg * lam_tau ** 2
                   / (self.kh * self.beta * self.lam * Go))
        return (Lt / self.zELA) * denom ** (-1.0 / 3.0)

    # ------------------------------------------------------------------
    # Shared presentation methods (module-level implementations above)
    # ------------------------------------------------------------------

    profile = _profile
    thickness = _thickness
    plot = _plot
    __repr__ = _repr
