"""Physical-parameter front end for the analytical steady state.

``SteadyStateProfile`` is the preferred public name; the class object is
``analytical_steady_state_solution`` (kept as the historical/primary name,
with ``SteadyStateProfile`` a module-level alias). It maps the model's
physical parameters — uplift, erodibility, climate, sliding law — onto the
steepness-form theory and solves the coupled glacial-fluvial steady state.

Unlike the steepness-form classes in ``siim.analytical.profiles``, the
glacial steepness index here is not a constant: cs depends on Lt through the
shape factor Go (C = Co (kh beta lam Go)^mu), so the closure zELA(Lt) is
assembled with cs(Lt) inside and handed to the same shared root solver
(``siim.analytical.core.find_closure_roots``) that GeneralProfile and
MarginalCoulombProfile use. One engine, three front ends.

The k = 1 linear mass-balance ansatz recovers the sliding-ice incision
model of :cite:t:`dealSlidingIceIncision2021`; the fluvial limb is the
stream-power model of :cite:t:`whippleDynamicsStreampowerRiver1999`.

Imports stay numpy/scipy-light: usable without the model stack.
"""
import warnings
from types import SimpleNamespace

import numpy as np

from ..constants import (AC, ALPHA_G, BETA, CE, D_HACK, DEFAULT_SLIDING_LAW,
                         GRAVITY, HC_OVER_H, K_ACCUM, KH, KO, KT, LAMBDA_C,
                         LAMBDA_P, N_FLUVIAL, NU, P, RHO_ICE, SIGMA, TAU_C, U,
                         XO, ZELA, Co_power, cg_prefactor, derive_coulomb,
                         derive_power)
from .core import (Solution, closure_slope, find_closure_roots,
                   incomplete_beta_compl)


class analytical_steady_state_solution:
    """Standalone steady-state analytical solution for the coupled
    glacial-fluvial profile.

    Constructed from a dict of user parameters; computes the full SS profile
    eagerly. Same parameter conventions as siim1d, restricted to the subset the
    analytical needs (no simulation-loop params). U and zELA must be scalars.

    Use::

        a = analytical_steady_state_solution({"zELA": 1000, "Ko": 1e-6, ...})
        a.surface, a.bed   # arrays on a.x  (None if no glacier-or-fluvial regime)
        a.zo, a.Lt         # divide elevation, terminus position
        a.analytical_z(xp), a.analytical_zb(xp), a.analytical_ice_thickness(xp)
        a.solutions        # all steady states found (incl. warm saddles)
    """

    def __init__(self, user_params=None):
        if user_params is None:
            user_params = {}
        elif not isinstance(user_params, dict):
            raise TypeError("params must be provided as a dictionary.")
        self.set_and_check_parameters(user_params)
        # eager solve. Sets self.surface/.bed (primary) and, when bistable,
        # self.surface_alt/.bed_alt (fluvial alt). glacier_flag enumerates:
        #   -1 : no SS (no stable glaciated root and fluvial inviable)
        #    1 : pure fluvial
        #    2 : mixed glacial-fluvial (Lt in [xo, Ld])
        #    3 : purely glacial (virtual Lt > Ld)
        #    4 : bistable mixed/fluvial (mixed primary, fluvial alt)
        #    5 : bistable glacial/fluvial (glacial primary, fluvial alt)
        self.surface, self.bed = self._analytical_profiles()

    # ---- Parameter setup --------------------------------------------------
    def set_and_check_parameters(self, user_params):
        defaults = {
            # domain
            "xo": XO,   # single-sourced (was 500; m54)
            "L": 5e4,
            "nx": 1001,
            "dx": None,
            # climate
            "P": P,
            "beta": BETA,
            "zELA": ZELA,
            "zT": None,
            "U": U,
            # fluvial
            "k_h": KH,
            "d": D_HACK,
            "sigma": SIGMA,
            "n": N_FLUVIAL,
            "m": None,
            "Ko": KO,
            # glacial
            "Ac": AC,
            "alpha_g": ALPHA_G,
            "lambda_p": LAMBDA_P,
            "lambda_c": None,
            "mu": None,
            "nu": NU,
            "ell": None,
            "lam": None,
            "k": K_ACCUM,         # accumulation-profile shape exponent (paper's p)
            "sliding_law": DEFAULT_SLIDING_LAW,
            "ce": CE,
            "tau_c": TAU_C,
            "left_bc": "base_level",
            "right_bc": "reflecting",
        }

        unexpected = set(user_params) - set(defaults)
        if unexpected:
            raise ValueError(f"Unexpected parameter(s): {sorted(unexpected)}")

        params = SimpleNamespace(**{**defaults, **user_params})

        if (params.left_bc != "base_level") & (params.right_bc != "base_level"):
            raise ValueError("At least one boundary condition must be 'base_level'")

        # U and zELA must be scalars in the analytical
        if not np.isscalar(params.U):
            raise TypeError("U must be a scalar for the analytical solution.")
        if params.zELA is not None and not np.isscalar(params.zELA):
            raise TypeError("zELA must be a scalar for the analytical solution.")

        self.left_bc = params.left_bc
        self.right_bc = params.right_bc

        # grid
        self.L = params.L
        self.xo = params.xo
        if params.dx is not None:
            self.nx = int(self.L / params.dx) + 1
        else:
            self.nx = params.nx
        self.x = self.L - np.linspace(0, self.L, self.nx)
        self.dx = np.abs(np.mean(np.diff(self.x)))

        # uplift
        self.U = params.U

        # zELA / zT
        if params.zELA is None and params.zT is None:
            self.zELA = ZELA
            self.zT = ZELA + params.P / params.beta
        elif params.zELA is None and params.zT is not None:
            self.zELA = params.zT - params.P / params.beta
            self.zT = params.zT
        else:
            self.zELA = params.zELA
            self.zT = self.zELA + params.P / params.beta
        self.P = params.P
        self.beta = params.beta

        # geometry / fluvial
        self.k_h = params.k_h
        self.d = params.d
        self.sigma = params.sigma
        self.kt = KT
        self.n = params.n
        self.m = self.n / 2 if params.m is None else params.m
        self.Ko = params.Ko

        # glacial constants
        self.g = GRAVITY
        self.rho_g = RHO_ICE
        self.Ac = params.Ac
        # lambda_p: plain user param, default constants.LAMBDA_P (a merge
        # default, so a user value trumps).
        self.lambda_p = float(params.lambda_p)
        self.lambda_c = params.lambda_c if params.lambda_c is not None else LAMBDA_C
        self.alpha_g = params.alpha_g
        self.ce = params.ce
        self.tau_c = params.tau_c
        self.sliding_law = params.sliding_law

        # cg (kt absorbed; 2/5 = depth-integrated Glen prefactor) and fluvial K
        self.cg = cg_prefactor(self.alpha_g, self.Ac, self.rho_g, self.g)
        self.K = self.Ko * (self.k_h * self.P) ** self.m
        self.q = self.d * self.m / self.n

        # per-law dispatch of ell/nu/mu/phi/Co
        if user_params.get('nu') is not None and user_params.get('ell') is not None:
            warnings.warn(
                "Both 'nu' and 'ell' were provided; 'ell' takes precedence and "
                "'nu' is ignored (re-derived from ell by the sliding-law relations).",
                UserWarning, stacklevel=2)
        if self.sliding_law == 'coulomb':
            c = derive_coulomb(self.ce, self.alpha_g, self.tau_c,
                               self.rho_g, self.g,
                               nu=params.nu, ell=params.ell)
        else:
            c = derive_power(self.ce, self.cg, self.lambda_p, self.alpha_g,
                             nu=params.nu, ell=params.ell)
        self.ell, self.nu, self.mu, self.phi, self.Co = c
        if params.mu is not None:
            self.mu = params.mu
            self.phi = self.mu / self.nu
            # Co carries mu in its exponent (paper Co(mu)); recompute it for the
            # eff-exp/power laws whose C = Co*(...)**mu consumes it, so an
            # explicit mu override doesn't pair a stale Co with the new mu
            # (audit B5). Coulomb's Co depends on ell (unchanged here).
            if self.sliding_law != 'coulomb':
                self.Co = Co_power(self.ce, self.cg, self.lambda_p,
                                   self.alpha_g, self.mu)

        # Hack-substituted concavity exponents
        self.p = self.d * self.phi              # paper d*phi (NOT the accumulation k)
        # d*phi = 1 is the marginal case where Go's beta argument
        # (1-d*phi)/k = 0; the kernel's exact b = 0 logarithmic branch
        # handles it.
        self.r = (1.0 - self.p) / (1.0 + self.phi)
        # k is the accumulation-profile shape exponent (paper's p).
        # k=1 recovers the linear ansatz (Deal & Prasicek 2021); k != 1 captures
        # divide concavity. Plumbed through Go, C, cs, the surface formula, and
        # the H formula.
        self.k = params.k
        # lam = d*sigma / (d*sigma + k).
        self.lam = (params.lam if params.lam is not None
                    else self.d * self.sigma / (self.d * self.sigma + self.k))

    # ---- Analytical solve -------------------------------------------------
    def solve(self):
        """Re-run the full SS solve (idempotent). Updates self.surface/.bed."""
        self.surface, self.bed = self._analytical_profiles()
        return self.surface, self.bed

    def _mirror_if_two_sided(self, arr):
        """When both BCs are base_level, the analytical was solved on the
        upstream half [0, Ld] and we mirror to fill the downstream half."""
        if (self.left_bc == "base_level") & (self.right_bc == "base_level"):
            if self.nx % 2 == 0:
                return np.concatenate((arr, arr[::-1]))
            return np.concatenate((arr, arr[-2::-1]))
        return arr

    def _analytical_profiles(self):
        """Return (surface, bed) for the primary SS profile and populate
        self.surface_alt/.bed_alt for the bistable cases (flags 4, 5).
        Solves the continuous zELA(Lt) closure for all steady states via the
        shared root solver, then classifies the regime. Returns (None, None)
        when no SS exists (flag == -1) or grid not yet defined."""
        self.ks = (self.U / self.K) ** (1 / self.n)
        if (self.left_bc == "base_level") & (self.right_bc == "base_level"):
            self.Ld = self.L / 2
        else:
            self.Ld = self.L

        self._solve_steady_state()
        self._set_glacier_flag()

        # default: no alt solution
        self.surface_alt = self.bed_alt = None

        if not hasattr(self, "x") or self.x is None:
            return None, None
        if self.glacier_flag == -1:
            return None, None

        # set xp based on boundary conditions
        if (self.left_bc == "base_level") & (self.right_bc == "reflecting"):
            self.xp = self.x / self.Ld
        elif (self.left_bc == "reflecting") & (self.right_bc == "base_level"):
            self.xp = (self.Ld - self.x) / self.Ld
        elif (self.left_bc == "base_level") & (self.right_bc == "base_level"):
            # Two-sided: solved on the upstream half [0, Ld] and mirrored. For
            # odd nx a node sits on the divide and x[nx//2:]/Ld runs the
            # divide->outlet half exactly (kept bit-for-bit). For even nx there
            # is no divide node, so x[nx//2]/Ld starts at (nx-2)/(nx-1) < 1 and
            # the mirror violates z=0 at the outlets; build the half-grid
            # coordinate from distance-to-divide directly instead so the mirror
            # is exact for any nx (audit B3).
            if self.nx % 2 == 1:
                self.xp = self.x[self.nx // 2:] / self.Ld
            else:
                self.xp = np.abs(self.x[:self.nx // 2] - self.L / 2.0) / self.Ld

        if self.glacier_flag == 1:
            # Pure fluvial — no glacier
            surface = self._mirror_if_two_sided(self.analytical_z_fluvial(self.xp))
            bed = surface
        else:
            # flags 2, 3, 4, 5 — primary has a glacier. analytical_z and
            # analytical_zb dispatch on xp <= Lt/Ld; for purely-glacial Lt > Ld
            # so xp <= Lt/Ld is True everywhere and the glacial branch covers
            # the full domain (model paper app:glacier, virtual terminus —
            # eq:z_purely_glaciated).
            surface = self._mirror_if_two_sided(self.analytical_z(self.xp))
            bed = self._mirror_if_two_sided(self.analytical_zb(self.xp))
            if self.glacier_flag in (4, 5):
                # bistable: also expose the pure-fluvial alternative SS.
                self.surface_alt = self._mirror_if_two_sided(self.analytical_z_fluvial(self.xp))
                self.bed_alt = self.surface_alt
        if surface is None or bed is None:
            return None, None
        if not (np.all(np.isfinite(surface)) and np.all(np.isfinite(bed))):
            return None, None
        return surface, bed

    def plot(self, ax=None, *, show_ela=True, show_thickness=True,
             show_bedrock=True, topo_color='k', ice_color='#b0d6e2',
             ice_alpha=0.55, bedrock_color='#888888', bedrock_fade='auto',
             bedrock_alpha=0.6, label=None):
        """Plot the steady-state profile in the paper's canonical profile
        style — the same renderer as ``GeneralProfile.plot``
        (``analytical.profiles.draw_profile``), so a basin-fitted analytical
        reference overlays in the identical visual language as the standalone
        profiles. ``bed`` is the channel-floor datum (``z - HC_OVER_H*H``).

        Plotted against distance from the divide (``x`` runs L -> 0). Returns
        the axis (a new 8x3-inch figure when ``ax`` is None); a no-op axis
        when there is no steady-state glacier to draw.
        """
        import matplotlib.pyplot as plt  # lazy: keep the module headless
        from .profiles import draw_profile

        surface, bed = self._analytical_profiles()
        if surface is None:
            return ax
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 3))

        # x runs L -> 0 (from the outlet); plot distance from the divide.
        x_from_divide = self.L - self.x
        order = np.argsort(x_from_divide)
        xd = x_from_divide[order]
        has_glacier = self.glacier_flag in (2, 3, 4, 5)
        ice_end = (self.Lt if (has_glacier and np.isfinite(self.Lt)
                               and self.Lt <= self.L) else self.L)
        draw_profile(ax, xd, surface[order], bed[order],
                     ice_x_range=(self.xo, ice_end) if has_glacier else None,
                     topo_color=topo_color, ice_color=ice_color,
                     ice_alpha=ice_alpha, bedrock_color=bedrock_color,
                     bedrock_fade=bedrock_fade, bedrock_alpha=bedrock_alpha,
                     show_bedrock=show_bedrock, show_thickness=show_thickness,
                     label=label)

        if show_ela:
            ax.axhline(self.zELA, color='r', lw=0.8, ls='--', zorder=-1)
        ax.set_xlabel('Distance from divide (km)')
        ax.set_ylabel('Elevation (m)')
        ax.set_xlim(0, self.L / 1e3)
        return ax

    # Field names mirror siim1d's ``plot.profile`` (siim/plotting/profiles.py).
    _PROFILE_FIELDS = ('elevation', 'ice_thickness', 'erosion_rate',
                       'ice_flux', 'water_flux')

    def plot_profile(self, fields='elevation', *, axes=None):
        """Stacked analytical-profile panels (filled-band notebook style), one
        per field. Mirrors the five fields of ``siim1d``'s ``plot.profile`` --
        'elevation', 'ice_thickness', 'erosion_rate', 'ice_flux', 'water_flux'
        -- but draws the analytical steady state only (no extracted 2D channel).

        Unlike ``plot`` (the canonical single-axis ``draw_profile`` renderer),
        this reproduces the two-panel bed/ice/thickness layout. ``bed`` is the
        channel-floor datum (``z - HC_OVER_H*H``); at steady state the erosion
        rate is uniformly ``U``. Plotted against ``x`` (distance, km).

        Parameters
        ----------
        fields : str or sequence of str
            One field name, or an ordered list -> one stacked panel each.
        axes : matplotlib.axes.Axes or sequence, optional
            Draw into existing axes (length must match ``fields``); otherwise a
            new figure is created.

        Returns
        -------
        (fig, axes), or ``(None, None)`` when there is no steady-state glacier.
        """
        import matplotlib.pyplot as plt  # lazy: keep the module headless

        if isinstance(fields, str):
            fields = [fields]
        fields = list(fields)
        bad = [f for f in fields if f not in self._PROFILE_FIELDS]
        if bad:
            raise ValueError(f"unknown profile field(s) {bad}; "
                             f"allowed: {list(self._PROFILE_FIELDS)}")
        if self.surface is None:
            return None, None

        x = self.x / 1e3
        H = self.surface - self.bed

        if axes is None:
            fig, axes = plt.subplots(len(fields), 1,
                                     figsize=(10, 2 * len(fields)),
                                     sharex=True, squeeze=False)
            axes = axes[:, 0]
        else:
            axes = np.atleast_1d(axes)
            fig = axes[0].figure
            if len(axes) != len(fields):
                raise ValueError("axes length must match the number of fields")

        for ax, field in zip(axes, fields):
            if field == 'elevation':
                ax.fill_between(x, 0, self.bed, color='gray', alpha=0.2)
                ax.fill_between(x, self.bed, self.surface, color='steelblue',
                                alpha=0.25, label='ice')
                ax.plot(x, self.bed, 'k-', lw=1, label='bedrock')
                ax.plot(x, self.surface, 'navy', lw=1, label='ice surface')
                ax.axhline(self.zELA, color='goldenrod', ls='--', lw=1,
                           label='ELA')
                ax.set_ylabel('Elevation (m)')
                ax.set_ylim(0, 2 * self.zELA)
                ax.legend(loc='upper right')
            elif field == 'ice_thickness':
                ax.fill_between(x, 0, H, color='steelblue', alpha=0.3)
                ax.plot(x, H, color='navy', lw=1)
                ax.set_ylabel('Ice thickness (m)')
                ax.set_ylim(0, None)
            elif field == 'erosion_rate':
                ax.axhline(self.U, color='firebrick', lw=1.5,
                           label='erosion rate = U (steady state)')
                ax.set_ylabel('Erosion rate (m/yr)')
                ax.set_ylim(0, 2 * self.U if self.U > 0 else None)
                ax.legend(loc='upper right')
            elif field == 'ice_flux':
                Qg = self._analytical_ice_flux()
                ax.fill_between(x, 0, Qg, color='dodgerblue', alpha=0.25)
                ax.plot(x, Qg, color='navy', lw=1)
                ax.set_ylabel('Ice flux (m$^3$/yr)')
                ax.set_ylim(0, None)
            elif field == 'water_flux':
                Qf = self._analytical_water_flux()
                ax.fill_between(x, 0, Qf, color='orangered', alpha=0.25)
                ax.plot(x, Qf, color='firebrick', lw=1)
                ax.set_ylabel('Water flux (m$^3$/yr)')
                ax.set_ylim(0, None)
            ax.grid(True, alpha=0.2)
            ax.set_xlim(0, self.L / 1e3)

        axes[-1].set_xlabel('Distance (km)')
        fig.tight_layout()
        return fig, axes

    # --- Steady-state solve (shared closure solver) -------------------------
    # Go = (1/k) * B(1 - (xo/Lt)^k; 1 - phi, (1 - d*phi)/k) is the paper's
    # glacial shape function evaluated at the channel head x = xo. cs depends
    # on Lt through Go (C = Co (kh beta lam Go)^mu), so the closure below
    # carries cs(Lt) inside and the (Go, cs, Lt) joint dependence is handled
    # by the root solver directly — no fixed-point iteration.

    def _cs_at(self, Go):
        """cs = (U/C)^(1/(nu+mu)) with C = Co (kh beta lam Go)^mu. Vectorizes
        over Go. U = 0 gives cs = 0 (flat steady state) without error."""
        C = self.Co * (self.k_h * self.beta * self.lam * Go) ** self.mu
        return (self.U / C) ** (1.0 / (self.nu + self.mu))

    def _zELA_of_Lt(self, Lt):
        r"""Climate zELA for which Lt is a steady-state terminus.

        .. math::

           z_{\rm ELA}(L_t) =
           \begin{cases}
              z_{\rm fluvial}(L_t)
              + (1-\lambda)\, c_s(L_t)\, L_t^{r}\, G_o(L_t)
                  & L_t \le L_d \text{ (mixed)} \\
              c_s(L_t)\, L_t^{r}
              \left[(1-\lambda)\, G_o(L_t) - G(L_d, L_t)\right]
                  & L_t > L_d \text{ (glacial)}
           \end{cases}

        Unlike the steepness-form classes, cs depends on Lt (through Go).
        Continuous through the seam at Lt = Ld (z_fluvial(Ld) = 0 and
        G(Ld; Ld) = 0). Vectorizes over Lt.
        """
        Lt = np.asarray(Lt, dtype=float)
        scalar = Lt.ndim == 0
        Lt = np.atleast_1d(Lt)
        Go = self._G_at(self.xo, Lt)
        csr = self._cs_at(Go) * Lt ** self.r
        out = np.empty_like(Lt)
        mref = Lt <= self.Ld
        if mref.any():
            out[mref] = (np.asarray(self.analytical_z_fluvial(Lt[mref] / self.Ld),
                                    dtype=float)
                         + (1.0 - self.lam) * csr[mref] * Go[mref])
        if (~mref).any():
            out[~mref] = csr[~mref] * ((1.0 - self.lam) * Go[~mref]
                                       - self._G_at(self.Ld, Lt[~mref]))
        return float(out[0]) if scalar else out

    def _zt_zo_at(self, Lt):
        """(zt, zo) for a glaciated root, without mutating solver state."""
        Go = float(self._Go_at(Lt))
        cs = float(self._cs_at(Go))
        if Lt <= self.Ld:
            zt = float(self.analytical_z_fluvial(Lt / self.Ld))
        else:
            zt = -cs * Lt ** self.r * self._G_at(self.Ld, Lt)
        return zt, zt + cs * Lt ** self.r * Go

    def _solve_steady_state(self):
        """All roots of the continuous closure zELA(Lt) = zELA on
        [xo, 1e8 Ld] (the mixed/glacial seam Lt = Ld is an exact grid node),
        with stability from the closure slope. The largest stable root is the
        steady state (cold branch); warm saddles are exposed via
        ``solutions`` but never selected. Replaces the old fixed-point
        (Go, Lt) iteration, single-bracket mixed brentq, and the separate
        purely-glacial solve. Sets Lt, zLt, Go, C, cs, kappa, solutions."""
        lo = self.xo * 1.0001

        def f(Lts):
            return self._zELA_of_Lt(Lts) - self.zELA

        roots = find_closure_roots(f, lo, self.Ld * 10.0, kink=self.Ld,
                                   n=4000, hi_max=self.Ld * 1e8)

        sols = []
        for Lt in roots:
            slope = closure_slope(self._zELA_of_Lt, Lt, kink=self.Ld, lo=lo)
            zt, zo = self._zt_zo_at(Lt)
            regime = 'mixed' if Lt <= self.Ld else 'glacial'
            sols.append(Solution(regime, Lt, zt, zo, stable=slope < 0.0))

        # Fluvial state, for the shared Solution vocabulary (flag logic below
        # makes its own equivalent viability test).
        z_fluv_divide = float(self.analytical_z_fluvial(self.xo / self.Ld))
        if self.zELA >= z_fluv_divide:
            sols.append(Solution('fluvial', float('nan'), float('nan'),
                                 z_fluv_divide, stable=True))
        self.solutions = sols

        stable_glaciated = [s for s in sols
                            if s.stable and np.isfinite(s.Lt)]
        if stable_glaciated:
            primary = max(stable_glaciated, key=lambda s: s.Lt)
            self.Lt = primary.Lt
            self.Go = self._Go_at(self.Lt)
            self._set_C_and_cs()
            self.zLt = primary.zt
        else:
            # No stable glaciated steady state. Evaluate the prefactors at the
            # old initial guess Lt = Ld/2 so cs/kappa stay queryable.
            self.Lt = np.nan
            self.Go = self._Go_at(0.5 * self.Ld)
            self._set_C_and_cs()
            self.zLt = np.nan

    def _set_C_and_cs(self):
        # C = Co * (k_h * beta * lam * Go)^mu.
        self.C = self.Co * (self.k_h * self.beta * self.lam * self.Go) ** self.mu
        self.cs = (self.U / self.C) ** (1.0 / (self.nu + self.mu))
        # Dimensionless glacial/fluvial ratio kappa = N_g / N_f. Stashed for
        # queryability. U = 0 makes both steepness indices 0 (kappa = 0/0):
        # report NaN instead of dividing by zero.
        num = self.cs * self.Ld ** self.r
        den = self.ks * self.Ld ** (1.0 - self.q)
        if den == 0.0:
            self.kappa = float('nan') if num == 0.0 else float('inf')
        else:
            self.kappa = num / den

    def _Go_at(self, Lt):
        """Go = G(xo, Lt). Convenience wrapper for evaluation at the head."""
        return self._G_at(self.xo, Lt)

    def _G_at(self, x, Lt):
        """G(x, Lt) = (1/k) * B(1 - (x/Lt)^k; 1-phi, (1-d*phi)/k).

        The canonical paper-G function. Vectorizes over x or Lt; returns a
        Python float when both are scalars. Evaluated through the eps-form of
        the analytically continued incomplete beta (eps = (x/Lt)^k), exact
        arbitrarily deep into the x << Lt tail — including b <= 0 and the
        marginal b = 0 case.
        """
        eps = np.minimum(np.asarray(x, dtype=float) / Lt, 1.0) ** self.k
        out = incomplete_beta_compl(eps, 1.0 - self.phi,
                                    (1.0 - self.p) / self.k) / self.k
        if np.ndim(x) == 0 and np.ndim(Lt) == 0:
            return float(out)
        return out

    def _set_glacier_flag(self):
        """Classify the SS based on terminus location and the fluvial-divide
        elevation. Bistability (glacier AND fluvial both valid) is detected
        by z_fluv(xo) < zELA — fluvial divide below ELA → no precip falls as
        snow at the head → fluvial-only is also a self-consistent SS.
            -1 : no SS in domain (no stable glaciated root, fluvial inviable)
             1 : pure fluvial (Lt < xo, or Lt non-finite with z_fluv(xo) <= zELA)
             2 : mixed glacial-fluvial unique (xo <= Lt <= Ld, z_fluv(xo) >= zELA)
             3 : purely glacial unique (virtual Lt > Ld, z_fluv(xo) >= zELA)
             4 : bistable mixed/fluvial (xo <= Lt <= Ld, z_fluv(xo) < zELA)
             5 : bistable glacial/fluvial (virtual Lt > Ld, z_fluv(xo) < zELA)
        Sets self.zo (primary divide elevation); for flags 4 and 5 also sets
        self.zo_alt to the fluvial-only divide elevation."""
        z_fluv_divide = self.analytical_z_fluvial(self.xo / self.Ld)
        Lt = self.Lt
        # Bistability: when a glacier solution exists AND the fluvial divide
        # sits below the ELA, the fluvial-only profile is also a self-
        # consistent SS (no precip falls as snow at the head → no glaciation
        # → fluvial is closed). Both regimes are then valid.
        bistable_alt = z_fluv_divide < self.zELA

        if np.isfinite(Lt) and self.xo <= Lt <= self.Ld:
            # Mixed glacial-fluvial.
            self.glacier_flag = 4 if bistable_alt else 2
        elif np.isfinite(Lt) and Lt > self.Ld:
            # Purely glacial (virtual Lt past domain edge — model paper
            # app:glacier, eq:Lt_purely_glaciated).
            self.glacier_flag = 5 if bistable_alt else 3
        elif np.isnan(Lt) and z_fluv_divide > self.zELA:
            # No stable glaciated root but z_fluv > zELA suggests a glacier
            # should physically exist (e.g. only a warm saddle was found).
            self.glacier_flag = -1
        else:
            # Lt = nan with z_fluv <= zELA, or Lt < xo edge — pure fluvial.
            self.glacier_flag = 1

        self.zo_alt = None
        if self.glacier_flag == 1:
            self.zo = z_fluv_divide
        elif self.glacier_flag in (2, 3, 4, 5):
            self.zo = self.analytical_z_glacial(self.xo / self.Ld)
            if self.glacier_flag in (4, 5):
                self.zo_alt = z_fluv_divide
        else:  # -1
            self.zo = np.nan

    # --- Profile evaluators --------------------------------------------------
    # All formulas in terms of the dimensionless coordinate v = x/Lt (profile
    # evaluators), per the steady-state profile derivations (model paper
    # app:glacier; theory paper, *Coupled glacial-fluvial steady state*).

    def analytical_z_fluvial(self, xp):
        r"""Fluvial profile (x > Lt branch).

        .. math::

           z(x) = k_s\, L^{1-d\theta}\,
           B\!\left(1 - x/L;\ 1,\ 1-d\theta\right)

        with xp = x/L the dimensionless along-flow coordinate. xp is clamped
        to >= xo/Ld (the channel-head cutoff) so the q=1 log singularity at
        xp=0 doesn't appear and x < xo reads as 'at the channel head'.

        Returns
        -------
        ndarray or float
            Surface elevation above base level [m], shape of ``xp``.
        """
        xp = np.maximum(xp, self.xo / self.Ld)
        if abs(1.0 - self.q) < 1e-12:
            return self.ks * np.log(1.0 / xp)
        return self.ks * (self.Ld ** (1.0 - self.q) / (1.0 - self.q)) * (1.0 - xp ** (1.0 - self.q))

    def analytical_z_glacial(self, xp):
        r"""Glacial profile (x <= Lt branch).

        .. math::

           z(x) - z(L_t) = c_s\, L_t^{r}\, G(x, L_t)

        with xp = x/Ld the dimensionless along-domain coordinate. xp is clamped
        to >= xo/Ld so x < xo reads as 'at the channel head'; the clamp
        guarantees upper = 1 - (x/Lt)^k <= 1 - (xo/Lt)^k, so no separate
        physical cap is needed inside _G_at.

        Returns
        -------
        ndarray or float
            Surface elevation above base level [m], shape of ``xp``.
        """
        xp = np.maximum(xp, self.xo / self.Ld)
        return self.zLt + self.cs * self.Lt ** self.r * self._G_at(xp * self.Ld, self.Lt)

    def analytical_ice_thickness(self, xp):
        r"""Ice thickness profile (dimensional, despite the internal H').

        .. math::

           H(x) = z_{\rm ELA}\, N_H \left[\bigl(1-(x/L_t)^k\bigr)\,
           (x/L_t)^d\right]^{\gamma}

        Dispatches on ``self.sliding_law``:

        - ``coulomb``: gamma = phi with the coulomb NH
        - ``eff-exp`` / ``power``: gamma = (2/3)(1/3 + phi) with the
          eff-exp NH

        The Lt-exponent is gamma*(1+d)/(1+phi) for both. Pre-factors (cs,
        Lt, Go, lam, r) carry the law-specific phi via the per-law dispatch.

        Returns
        -------
        ndarray or float
            Width-mean ice thickness [m] (the dimensionless H' = H/zELA is
            multiplied back by zELA before returning), shape of ``xp``. Zero
            outside the ice extent (x > L_t) and identically zero for a fluvial
            or absent solution (L_t is NaN) — matching the profiles._thickness
            family (audit m26); no NaN/RuntimeWarning past the terminus.
        """
        Lt = self.Lt
        phi = self.phi
        xp_arr = np.asarray(xp, dtype=float)
        if not np.isfinite(Lt):
            return np.zeros_like(xp_arr) if xp_arr.ndim else 0.0
        xp = np.maximum(xp_arr, self.xo / self.Ld)
        v = xp / (Lt / self.Ld)  # = x/Lt
        # Zero beyond the terminus (v > 1): clamp the spatial factor to 0 before
        # the fractional power so the negative base never yields NaN (m26).
        spatial = np.where(v <= 1.0, (1.0 - v ** self.k) * v ** self.d, 0.0)

        if self.sliding_law == 'coulomb':
            # Coulomb closure: gamma = phi,
            # NH = ((cs/lambda_tau)^3 + cg*lambda_c*lambda_tau^2/(kh*beta*lam*Go))^(-1/3)
            #      * Lt^(gamma(1+d)/(1+phi)) / zELA, lambda_tau = tau_c/(rho_g g).
            gamma = phi
            lambda_tau = self.tau_c / (self.rho_g * self.g)
            denom = ((self.cs / lambda_tau) ** 3
                     + self.cg * self.lambda_c * lambda_tau ** 2
                       / (self.k_h * self.beta * self.lam * self.Go))
            NH = denom ** (-1.0 / 3.0)
        else:
            # eff-exp / power closure: gamma = (2/3)(1/3 + phi),
            # NH = (kh*beta*lam*Go / (cg*lambda_p^(3/2)*cs^2))^(2/9)
            #      * Lt^(gamma(1+d)/(1+phi)) / zELA.
            gamma = (2.0 / 3.0) * (1.0 / 3.0 + phi)
            NH = (self.k_h * self.beta * self.lam * self.Go
                  / (self.cg * self.lambda_p ** 1.5 * self.cs ** 2)) ** (2.0 / 9.0)

        NH *= Lt ** (gamma * (1.0 + self.d) / (1.0 + phi)) / self.zELA
        return self.zELA * NH * spatial ** gamma

    def analytical_z(self, xp):
        """Surface elevation [m] at xp = x/Ld: glacial branch where
        xp <= Lt/Ld, fluvial elsewhere. Scalar in, scalar out."""
        xp = np.asarray(xp, dtype=float)
        scalar = xp.ndim == 0
        xp = np.atleast_1d(xp)
        z = np.zeros_like(xp)
        glacier = xp <= self.Lt / self.Ld
        z[glacier] = self.analytical_z_glacial(xp[glacier])
        z[~glacier] = self.analytical_z_fluvial(xp[~glacier])
        return float(z[0]) if scalar else z

    def analytical_zb(self, xp):
        """Bed elevation [m] at xp = x/Ld on the channel-floor datum:
        zb = z - HC_OVER_H * H under ice, zb = z elsewhere."""
        xp = np.asarray(xp, dtype=float)
        scalar = xp.ndim == 0
        xp = np.atleast_1d(xp)
        zb = np.zeros_like(xp)
        glacier = xp <= self.Lt / self.Ld
        # Channel-floor datum: bed = surface - hc with hc = HC_OVER_H * H
        # (H is the width-mean depth and is NOT rescaled — only the bed
        # reconstruction carries the parabola max/mean factor).
        zb[glacier] = (self.analytical_z_glacial(xp[glacier])
                       - HC_OVER_H * self.analytical_ice_thickness(xp[glacier]))
        zb[~glacier] = self.analytical_z_fluvial(xp[~glacier])
        return float(zb[0]) if scalar else zb

    # ---- On-grid convenience accessors ------------------------------------
    # These return the analytical quantity sampled on the model grid (mirrored
    # for the two-sided base-level case). Useful for plotting alongside the
    # numerical model output. Assumes _analytical_profiles() has been called
    # (eager solve in __init__ takes care of this).

    def _analytical_ice_thickness(self):
        """Ice thickness on the full grid: analytical_ice_thickness(xp) inside
        the glacier zone, 0 elsewhere. Returns None if flag==-1.
        For purely-glacial flags (3, 5) Lt > Ld so xp <= Lt/Ld is True
        throughout and the ice covers the full domain."""
        if self.glacier_flag == -1:
            return None
        H = np.zeros_like(self.xp)
        if self.glacier_flag in (2, 3, 4, 5):
            mask = self.xp <= self.Lt / self.Ld
            if mask.any():
                H[mask] = self.analytical_ice_thickness(self.xp[mask])
        return self._mirror_if_two_sided(H)

    def _analytical_ice_flux(self):
        """Ice flux Q_g(x) = kh*beta*(zo-zELA) * (1 - (x/Lt)^k) * x^d in the
        glacier zone, 0 elsewhere. Same form holds for purely-glacial cases
        where Lt is virtual. Returns None if flag==-1."""
        if self.glacier_flag == -1:
            return None
        Qg = np.zeros_like(self.xp)
        if self.glacier_flag in (2, 3, 4, 5):
            v = self.xp / (self.Lt / self.Ld)  # x_from_divide / Lt
            mask = v <= 1.0
            if mask.any():
                x_dim = self.xp[mask] * self.Ld
                Qg[mask] = (self.k_h * self.beta * (self.zo - self.zELA)
                            * (1.0 - v[mask] ** self.k)
                            * x_dim ** self.d)
        return self._mirror_if_two_sided(Qg)

    def _analytical_water_flux(self):
        """Water flux Q_f(x) = P*kh*x^d - Q_g(x). Returns None if flag==-1."""
        Qg = self._analytical_ice_flux()
        if Qg is None:
            return None
        # x_from_divide on the same (full or half) grid as self.xp/_mirror logic
        x_grid_from_divide = self._mirror_if_two_sided(self.xp * self.Ld)
        Qtot = self.P * self.k_h * x_grid_from_divide ** self.d
        return Qtot - Qg


# Canonical name; the lowercase historical name above remains a permanent
# alias (it IS the class — the alias direction keeps pickles and reprs
# stable for existing users).
SteadyStateProfile = analytical_steady_state_solution
