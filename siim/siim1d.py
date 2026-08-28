import warnings
import matplotlib.animation as anm, matplotlib.pyplot as plt, numpy as np, tqdm, numba
from types import SimpleNamespace

from .constants import (GRAVITY, KT, RHO_ICE, Co_power, derive_coulomb,
                        derive_power)
from . import constants as _constants
from .analytical import analytical_steady_state_solution
from ._output import output_path
from ._core.solvers import (  # shared scalar Newton solvers / H-closures (unified)
    _solve_ice_thickness_power_analytical, _solve_ice_thickness_coulomb,
    _solver_fluvial, _solver_glacial, _solver_glacial_power,
    _solver_glacial_coulomb,
    LAW_EFFEXP, LAW_POWER, LAW_COULOMB,
)
from ._core.skeleton import (  # 1D joint walk + fill + implicit border step
    _diag_walk, _implicit_border_step, _lake_fill_1d)
from ._core.eroders import _modeb_border_erosion  # border-bed glacial erosion rate
from ._core.params import GlacialParams


def _load_initial_topography(src):
    """Load 1D initial topography from an ndarray, pandas DataFrame, or CSV file.

    Long-format DataFrame / CSV columns: 'x', 'topography__elevation' (optional
    'time' — the latest time slice is used). x must be on a regular grid;
    coordinates are interpreted as extent and shifted to a 0-origin.

    Returns
    -------
    arr : (nx,) ndarray
        Elevation in CSV-natural order (x ascending, 0 → L). siim1d's self.x
        runs L → 0, so if the loaded profile lands the wrong way, reverse it
        yourself before passing. With base-level on both sides it usually
        doesn't matter.
    grid : dict or None
        {'nx', 'L'} when loaded from a file/DataFrame; None when the input was
        already an ndarray (caller keeps user nx/L).
    """
    if isinstance(src, np.ndarray):
        return np.asarray(src, dtype=float), None

    import pandas as pd
    from pathlib import Path

    if isinstance(src, (str, Path)):
        df = pd.read_csv(src)
    elif isinstance(src, pd.DataFrame):
        df = src
    else:
        raise TypeError(
            f"initial_topography must be an ndarray, pandas DataFrame, or "
            f"CSV filename; got {type(src).__name__}")

    needed = {'x', 'topography__elevation'}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(
            f"initial_topography source missing column(s): {sorted(missing)}")

    if 'time' in df.columns:
        df = df[df['time'] == df['time'].max()]

    df = df.sort_values('x')
    x_sorted = df['x'].to_numpy()
    z_sorted = df['topography__elevation'].to_numpy(dtype=float)

    diffs = np.diff(x_sorted)
    if not np.allclose(diffs, diffs[0]):
        raise ValueError("initial_topography x coordinates are not uniformly spaced")
    if not np.all(np.isfinite(z_sorted)):
        raise ValueError("initial_topography contains NaN")

    nx = len(x_sorted)
    L = float((nx - 1) * diffs[0])
    return z_sorted, dict(nx=nx, L=L)


# SIIM: a 1D model of coupled fluvial and glacial incision, with an analytical solution for the steady state profile.
class siim:
    """One-dimensional coupled glacial-fluvial profile model.

    Parameters
    ----------
    user_params : dict, optional
        Parameter overrides. Omitted keys use the documented model defaults;
        unknown keys raise ``ValueError``. See the public parameter reference
        for units, accepted values, and the distinction between numerical and
        analytical-only controls.

    Notes
    -----
    Construction validates parameters and prepares the embedded analytical
    steady-state reference. Call :meth:`run` to integrate the numerical model;
    results are exposed through the ``*_out`` arrays and ``plot`` helper.
    The 1D model does not implement the 2D model's pickle ``save``/``load``
    interface.
    """

    def __init__(self, user_params=None):
        # Check that params is a dictionary if it is provided, otherwise set it to an empty dictionary
        if user_params is None: user_params = {}
        elif not isinstance(user_params, dict): raise TypeError("params must be provided as a dictionary.")

        # set parameters, using defaults for any that are not provided, and check for unexpected parameters
        self.set_and_check_parameters(user_params)

        # composition: analytical solution (standalone class) and plotter.
        # Pass the analytical-relevant subset of user_params; coerce any
        # array-valued U / zELA to scalars (the analytical only handles scalars).
        self.analytical = analytical_steady_state_solution(
            self._analytical_user_params()
        )
        self.plot = siim_plotter(self)

    def _analytical_user_params(self):
        """Build the analytical reference from the model's RESOLVED parameter
        values, so it is an exact function of the simulation — not a mix of the
        user dict and the analytical's own defaults. (Those defaults used to
        agree only by coincidence of matching default dicts; if one ever drifted
        the analytical would silently use a different value — the fluvial-mismatch
        trap.) Every analytical input is forwarded from ``self.<attr>`` EXCEPT:
        ``ell`` and ``zT`` are omitted so the analytical re-derives them from the
        forwarded ``nu`` / ``zELA`` (avoiding the duplicate-exponent warning and
        any zELA/zT conflict). Array-valued U / zELA are already reduced to
        scalars (U: time-mean; zELA: minimum, the max-extent reference) by
        set_and_check_parameters. Base level ``bl`` is scalar-reduced (time-mean
        ``self.bl``) but deliberately NOT forwarded: the analytical reference is
        graded to the datum (z(outlet)=0), so a nonzero bl only shifts the whole
        profile by a constant — set_and_check_parameters warns rather than
        re-datum the analytical (per the array-forcing memory rule + the design
        study's §7 Q5 choice)."""
        keys = ("P", "beta", "k_h", "d", "sigma", "n", "m", "Ko",
                "Ac", "alpha_g", "lambda_p", "lambda_c", "mu", "nu", "lam",
                "k", "sliding_law", "ce", "tau_c", "left_bc", "right_bc",
                "xo", "L", "nx", "U", "zELA")
        return {k: getattr(self, k) for k in keys}

    def set_and_check_parameters(self, user_params):
        # default parameters
        defaults = {
            # domain params
            "xo": _constants.XO,   # single-sourced (was 100; m54)
            "L": 5e4,
            # model run params
            "nx": 501,
            "dx": None,
            "to": 0,
            "T": 3e6,
            "nt": 3000,
            "nt_out": 101,  # number of output frames (None = every step)
            # climate params
            "P": _constants.P,
            "beta": _constants.BETA,
            "zELA": _constants.ZELA,
            "zT": None,
            "U": _constants.U,
            # Base level: the water-line (Dirichlet) datum. Scalar (default
            # constants.BL = 0) or a length-nt time series like zELA/U/P; the
            # per-step value floors the erosion working view, the border-bed
            # budget, the flotation reference and the recovery threshold. The
            # analytical reference stays at the datum (bl scalar-reduced, not
            # forwarded — a nonzero mean warns).
            "bl": _constants.BL,
            # Global waterline-flotation gate (mode B; default
            # constants.FLOTATION_GATE = True): grounded <=> zs = zb + hc*H >= bl
            # (rho_i = rho_w). Interior: a sub-waterline icy cell does
            # reduced/no glacial erosion. Border: the ramp is the physical
            # bound inside the closed-form IMPLICIT border budget (the bed
            # digs on the arrival slope to the flotation-draft equilibrium).
            # False un-bounds the border — diagnostics only. See the boundary
            # condition notes in docs/guides/concepts.md.
            "flotation_gate": _constants.FLOTATION_GATE,
            # Flotation-ramp width gamma (default constants.FLOTATION_RAMP =
            # 0.1): the gate is the effective-pressure ramp — glacial erosion
            # is multiplied by f = clip((zs - bl)/(gamma*hc*H), 0, 1),
            # f = 0 exactly for zs <= bl. 0 = the hard binary gate (interior
            # bit-for-bit; at the border its implicit solution is the
            # flotation sliding mode). Safe ceiling 0.2 (wide ramps can dome
            # cap=False configs). Only active when flotation_gate is on.
            "flotation_ramp": _constants.FLOTATION_RAMP,
            # fluvial params
            "k_h": _constants.KH,
            "d": _constants.D_HACK,
            "sigma": _constants.SIGMA,
            "n": _constants.N_FLUVIAL,
            "m": None,
            "Ko": _constants.KO,
            # glacial params
            "Ac": _constants.AC,
            "alpha_g": _constants.ALPHA_G,
            "lambda_p": _constants.LAMBDA_P,  # eff-exp/power critical ice thickness [m]
            "lambda_c": None,     # regularized Coulomb sliding length; default constants.LAMBDA_C
            # SS flux exponent in U = Co Q^mu S^nu. For exact power/Coulomb
            # numerics an explicit override is analytical-only and warns.
            "mu": None,
            "nu": _constants.NU,  # SS slope exponent (primary user input)
            "ell": None,          # erosion-law exponent in Eg = ce (kt ub)^ell; default per-law-derived from nu
            "lam": None,          # AAR-like ratio (zo-zELA)/(zo-zLt); default d*sigma/(d*sigma+k)
            "k": _constants.K_ACCUM,  # accumulation-profile shape exponent (paper's k)
            "progress_bar": True,
            # Cap the accumulation rate at P (solid precip cannot exceed
            # total precip). Default True; False (uncapped linear balance)
            # is ONLY for comparing against the analytical solutions — at
            # high surfaces it permits unphysical accumulation rates that
            # can latch a permanent whole-domain ice dome.
            "cap_ice_accumulation": True,
            "sliding_law": _constants.DEFAULT_SLIDING_LAW,    # "eff-exp", "power", or "coulomb"
            "ce": _constants.CE,     # glacial erosion coefficient for power/coulomb laws
            "tau_c": _constants.TAU_C,  # maximum shear stress (Pa) for coulomb law
            "coulomb_clamp": _constants.COULOMB_CLAMP,  # minimum relative gap from the pole maintained by line-search bisection
            "border_bed_uplift": None,  # mode B: uplift rate (m/yr) applied to the bed at
                                        # base-level outlets (net U - E under ice; recovery
                                        # toward the datum when ice-free). None (default) =
                                        # use the node's tectonic U; set 0 to freeze.
            "left_bc": "base_level",
            "right_bc": "reflecting",
            "initial_topography": None,
            # Integration mode:
            #   'A' = z-tracking: z is the persistent state, slope is
            #         read off self.z, H solved explicitly; zb is bookkept via
            #         zb = z − hc_over_H·H at end of step.
            #   'B' = bedrock + ice-thickness tracking (default; all sliding
            #         laws): z_b is the
            #         persistent state, H solved jointly with z_s via the upstream
            #         walk on z_b, z_s = z_b + hc_over_H·H; preserves carved overdeepenings
            #         under ice retreat (bed memory) — the model's native regime.
            # User-facing names: 'ice_surface' (A) and 'bedrock+ice_thickness'
            # (B, default); 'A'/'B' accepted as aliases. Single scalar; applies
            # to the whole run. (constants.normalize_mode maps to the internal
            # 'A'/'B' the solvers use.)
            "mode": _constants.DEFAULT_MODE,
            # Mode-B erosion in closed surface basins (lakes). Default (True):
            # a terminus reaching a closed basin is treated as floating /
            # decoupled from the bed, so the lake-FILLED surface is eroded and a
            # lake suppresses erosion with or without ice. False: the ice is
            # grounded — erode the RAW ice surface in one pass, then zero erosion
            # only at cells that are in a lake AND ice-free, so the ice toe
            # carves its self-made overdeepening and only bare-rock lakes are
            # frozen. (Behavioural switch, not a true flotation criterion.) No
            # effect in mode A.
            "floating_termini": True,
            # Ice thickness diffusivity (m²/yr) for both modes. When > 0, H
            # is smoothed by an explicit FD diffusion right after H is solved
            # (formula in A, joint walk in B), before z_s/tau/ub are derived.
            # Sub-stepped for CFL. Smooths the ice profile especially at the
            # toe where H ∝ S^(-2/3) can produce sharp transitions. None or 0
            # disables.
            "H_diffusivity": None,
            # Centerline-to-mean depth ratio hc/H for surfaces built from
            # (zb, H) state: zs = zb + hc_over_H * H (the tracked bed is the
            # channel floor; H stays the mean depth driving flux/erosion).
            # None = use the package constant siim.constants.HC_OVER_H.
            # Override is mainly for A/B convention experiments.
            "hc_over_H": None}

        # check for unexpected parameters
        unexpected = set(user_params) - set(defaults)
        if unexpected:
            raise ValueError(f"Unexpected parameter(s): {sorted(unexpected)}")

        # update defaults with user-provided parameters
        params = SimpleNamespace(**{**defaults, **user_params})

        # initial_topography from ndarray, DataFrame, or CSV file. When loaded
        # from a file/DataFrame, the derived (nx, L) override any user-supplied
        # grid params; dx is reset so nx wins downstream.
        if params.initial_topography is not None:
            arr, grid = _load_initial_topography(params.initial_topography)
            params.initial_topography = arr
            if grid is not None:
                params.L = grid['L']
                params.nx = grid['nx']
                params.dx = None
                print(f"[siim1d] initial_topography loaded: "
                      f"nx={params.nx}, L={params.L/1e3:.1f} km, "
                      f"dx={params.L/(params.nx-1):.0f} m")

        # check for valid boundary conditions
        if (params.left_bc != "base_level") & (params.right_bc != "base_level"):
            raise ValueError("At least one boundary condition must be 'base_level'")

        # profile time-coordinates (yr)
        self.to = params.to  # initial time (yr)
        self.T = params.T  # total time (yr)
        self.nt = params.nt  # number of time steps
        self.t = np.linspace(self.to, self.T, self.nt)  # time (yr)
        self.dt = np.mean(np.diff(self.t))  # time step (yr)
        self.nt_out = params.nt_out  # number of output frames (None = every step)

        self.left_bc = params.left_bc
        self.right_bc = params.right_bc

        # profile x-coordinates (m)
        self.L = params.L  # profile length (m)
        self.xo = params.xo  # minimum catchment area in length (m)
        if params.dx is not None:
            self.nx = int(self.L / params.dx) + 1
        else:
            self.nx = params.nx  # number of nodes

        self.x = self.L - np.linspace(0, self.L, self.nx)
        self.dx = np.abs(np.mean(np.diff(self.x)))  # spatial step (m)

        # optional user-supplied initial topography, indexed like self.x (L → 0)
        if params.initial_topography is None:
            self.initial_topography = None
        else:
            z0 = np.asarray(params.initial_topography, dtype=float)
            if z0.shape != (self.nx,):
                raise ValueError(
                    f"initial_topography must be a 1D array of length nx={self.nx}, got shape {z0.shape}")
            self.initial_topography = z0

        # The uplift and zELA parameters can be either scalars or arrays, if they are arrays, I will use the mean value for the analytical model
        # tectonic params
        if np.isscalar(params.U):
            self.U = params.U
            self.U_matrix = params.U * np.ones((self.nx, self.nt))
        else:
            # asarray-coerce so a plain list works too (m21, parity with P/bl).
            U_arr = np.asarray(params.U, dtype=float)
            if U_arr.ndim == 2:
                # Only the documented (nx, nt) orientation; reject the transpose
                # (a silent mis-broadcast in the old code — audit m19).
                if U_arr.shape != (self.nx, self.nt):
                    raise ValueError(
                        f"a 2D U must have shape (nx, nt) = ({self.nx}, {self.nt}); "
                        f"got {U_arr.shape}.")
                self.U_matrix = U_arr
            elif U_arr.size == self.nt:
                self.U_matrix = U_arr.ravel() * np.ones((self.nx, 1))
            elif U_arr.size == self.nx:
                self.U_matrix = U_arr.ravel()[:, np.newaxis] * np.ones((1, self.nt))
            elif U_arr.size == self.nx * self.nt:
                # A flat nx*nt array is the documented (nx, nt) grid in row-major
                # order (old code stored it 1-D → later IndexError; audit m19).
                self.U_matrix = U_arr.reshape(self.nx, self.nt)
            else:
                raise ValueError(
                    f"U must be a scalar, a length-nt ({self.nt}) or length-nx "
                    f"({self.nx}) array, a flat nx*nt array, or a 2D (nx, nt) "
                    f"array; got shape {U_arr.shape}.")
            self.U = U_arr.mean()

        # precipitation P: scalar (default constants.P) or a length-nt time
        # series, exactly paralleling zELA below. P_run feeds the per-step ice
        # cap + water flux; the scalar self.P is the analytical-reference value
        # (time-MEAN for a series — the baseline the profile integrates toward,
        # cf. the U convention above).
        if np.isscalar(params.P) or params.P is None:
            self.P = _constants.P if params.P is None else params.P
            self.P_run = self.P * np.ones_like(self.t)
        else:
            P_arr = np.asarray(params.P, dtype=float).ravel()
            if P_arr.size != self.t.size:
                raise ValueError(
                    f"P must be a scalar or a time series with nt={self.t.size} "
                    f"entries; got size {P_arr.size}.")
            self.P_run = P_arr
            self.P = P_arr.mean()

        # base level bl: scalar (default constants.BL) or a length-nt series,
        # exactly paralleling P above. bl_run feeds the per-step waterline datum
        # (erosion-view floor, border-bed budget, flotation reference, recovery
        # threshold); the scalar self.bl is the analytical-reference value (time-
        # MEAN for a series) — but the analytical stays at the datum, so a
        # nonzero mean only drives a heads-up warning (see _analytical_user_params).
        if np.isscalar(params.bl) or params.bl is None:
            self.bl = _constants.BL if params.bl is None else float(params.bl)
            self.bl_run = self.bl * np.ones_like(self.t)
        else:
            bl_arr = np.asarray(params.bl, dtype=float).ravel()
            if bl_arr.size != self.t.size:
                raise ValueError(
                    f"bl must be a scalar or a time series with nt={self.t.size} "
                    f"entries; got size {bl_arr.size}.")
            self.bl_run = bl_arr
            self.bl = float(bl_arr.mean())
        # Global waterline-flotation gate (mode B). Off is for diagnostics only.
        self.flotation_gate = bool(params.flotation_gate)
        # Flotation-ramp width gamma (0 = hard binary gate, bit-for-bit).
        self.flotation_ramp = float(params.flotation_ramp)
        if self.flotation_ramp < 0.0:
            raise ValueError("flotation_ramp (gamma) must be >= 0.")
        if self.bl != 0.0:
            warnings.warn(
                "Nonzero base level bl: the analytical steady-state reference "
                "stays at the datum (bl=0), so comparisons (rms_vs_analytical, "
                "the analytical overlay) are offset by ~bl. Add bl to the "
                "profile yourself, or compare shapes.",
                UserWarning, stacklevel=2)

        # if neither zELA or zT is user defined
        if params.zELA is None and params.zT is None:
            self.zELA = _constants.ZELA
            self.zT = _constants.ZELA + self.P / params.beta
        # if zT is user defined, but not ELA
        elif params.zELA is None and params.zT is not None:
            self.zELA = params.zT - self.P / params.beta
            self.zT = params.zT
        # if zELA is user defined, then zT is ignored, whether it was user defined or not.
        else:
            if np.isscalar(params.zELA):
                self.zELA = params.zELA
            else:
                # For time-varying (e.g. cyclic) ELA forcing the analytical
                # reference deliberately uses the MINIMUM ELA — the most-
                # glaciated extreme of the cycle. (U is different: its time-
                # MEAN sets the baseline the profile integrates toward.)
                # asarray-coerce so a plain list works too (m21).
                self.zELA = np.asarray(params.zELA, dtype=float).min()
            self.zT = self.zELA + self.P / params.beta

        # zELA_run: time series for the simulation (always defined)
        if np.isscalar(params.zELA) or params.zELA is None:
            # zT given + P-series ⇒ time-varying zELA_run = zT − P(t)/β (paper
            # semantics: the ELA emerges from the climate forcing); the scalar
            # self.zELA keeps the time-MEAN as the analytical reference. Every
            # other scalar/None path (incl. scalar P, and the zELA-input path
            # below) stays a constant zELA_run, bit-for-bit unchanged. (audit F2)
            p_is_series = not (np.isscalar(params.P) or params.P is None)
            if params.zELA is None and params.zT is not None and p_is_series:
                self.zELA_run = params.zT - self.P_run / params.beta
            else:
                self.zELA_run = self.zELA * np.ones_like(self.t)
        else:
            zELA_arr = np.asarray(params.zELA, dtype=float).ravel()
            if zELA_arr.size != self.t.size:
                raise ValueError(
                    f"zELA must be a scalar or a time series with nt={self.t.size} "
                    f"entries; got size {zELA_arr.size}. (Spatially varying zELA "
                    f"is not supported.)")
            self.zELA_run = zELA_arr
        self.beta = params.beta

        # model parameters
        self.progress_bar = params.progress_bar
        self.sliding_law = params.sliding_law
        self.cap_ice_accumulation = params.cap_ice_accumulation

        # profile geometry parameters
        self.k_h = params.k_h  # hack's law coefficient
        self.d = params.d  # hack's law exponent
        self.sigma = params.sigma  # drainage contribution coefficient
        self.kt = KT  # time conversion factor (s/yr)

        # fluvial params
        self.n = params.n  # fluvial slope exponent
        if params.m is None:
            self.m = self.n / 2  # fluvial area exponent
        else:
            self.m = params.m
        self.Ko = params.Ko  # erodibility coefficient

        # glacial params
        self.g = GRAVITY  # gravity (m/s^2)
        self.rho_g = RHO_ICE  # ice density (kg/m^3)
        self.Ac = params.Ac
        # lambda_p: critical ice thickness for eff-exp/power. Plain user param,
        # default constants.LAMBDA_P (a merge default, so a user value trumps).
        # lambda_c (coulomb sliding length) is set separately below.
        self.lambda_p = float(params.lambda_p)
        self.lambda_c = params.lambda_c if params.lambda_c is not None else _constants.LAMBDA_C
        self.alpha_g = params.alpha_g  # glacial channel aspect ratio
        self.ce = params.ce  # glacial erosion coefficient
        self.tau_c = params.tau_c  # maximum shear stress (Pa) for coulomb law
        self.coulomb_clamp = params.coulomb_clamp
        self.border_bed_uplift = (None if params.border_bed_uplift is None
                                  else float(params.border_bed_uplift))

        # cg = alpha_g * kt * (2*Ac/5) * (rho_g g)^3 [m^-3 yr^-1] (model paper;
        # 2/5 = depth-integrated Glen prefactor 2A/(n_c+2), n_c = 3).
        # kt is absorbed into cg so the H-equations and prefactors carry no explicit kt.
        self.cg = _constants.cg_prefactor(self.alpha_g, self.Ac, self.rho_g, self.g)
        # Fluvial concavity (sliding-law-independent): q = d*theta = d*m/n.
        self.q = self.d * self.m / self.n

        # Slope/erosion exponents and Co — sliding-law-specific dispatch
        # (model paper: coulomb eq:glacial_streampower, power
        # app:power_law_sliding_law). Primary user input is `nu`; the law's
        # nu-ell-mu relationship sets the rest. `ell` set directly back-derives
        # nu/mu via the same per-law relations. `mu` set directly overrides.
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
            if self.sliding_law in {'power', 'coulomb'}:
                warnings.warn(
                    f"Explicit 'mu' with sliding_law={self.sliding_law!r} is "
                    "retained for the analytical steady-state interpretation "
                    "and reported phi, but the exact numerical sliding law "
                    "derives its flux exponent from nu/ell and does not use "
                    "the override.",
                    UserWarning, stacklevel=2)
            self.mu = params.mu
            self.phi = self.mu / self.nu
            # Co carries mu in its exponent (paper Co(mu)); recompute it for the
            # effective-exponent numerical law and analytical/reporting state.
            # The exact power kernel derives its own exponent and prefactor;
            # Coulomb's Co depends on ell and is unchanged by a mu-only override.
            if self.sliding_law != 'coulomb':
                self.Co = Co_power(self.ce, self.cg, self.lambda_p,
                                   self.alpha_g, self.mu)

        # k is the accumulation-profile shape exponent (paper's k).
        # k = 1 recovers the linear ansatz; only the analytical uses it here.
        self.k = params.k
        # lam = d*sigma / (d*sigma + k).
        self.lam = (params.lam if params.lam is not None
                    else self.d * self.sigma / (self.d * self.sigma + self.k))

        # Integration mode per step ('A' = z-tracking,
        # 'B' = z_b + hc_over_H*H tracking).
        # Surface-evolution mode: 'A' (z-tracking) or 'B' (bed + thickness
        # tracking with the channel-floor surface convention).
        # Mode 'B' is implemented for all three sliding laws (eff-exp, power, coulomb).
        self.mode = _constants.normalize_mode(params.mode)
        if self.mode == 'C':
            raise ValueError(
                "mode 'C' (sub-grid width carving) is 2D-only; the 1D model "
                "has no width machinery — use mode='B'.")
        self._mode_B_active = (self.mode == 'B')
        self.floating_termini = bool(params.floating_termini)

        # Centerline-to-mean depth ratio for surface construction:
        # zs = zb + hc * H (hc = 1 is the historical mean-bed datum).
        self.hc_over_H = (float(_constants.HC_OVER_H) if params.hc_over_H is None
                   else float(params.hc_over_H))
        if not self.hc_over_H > 0.0:
            raise ValueError(f"hc_over_H must be > 0, got {params.hc_over_H!r}")

        # Ice thickness diffusivity (both modes); None disables.
        self.H_diffusivity = (None if params.H_diffusivity is None
                                          else float(params.H_diffusivity))
        if self.H_diffusivity is not None and self.H_diffusivity < 0:
            raise ValueError("H_diffusivity must be >= 0 or None.")

    @property
    def run_id(self):
        ela = self.zELA_run[-1] if hasattr(self, 'zELA_run') else self.zELA
        return f"Ko{self.Ko:.0e}_ce{self.ce:.0e}_n{self.n:g}_nu{self.nu:g}_ELA{ela:g}"

    @property
    def _forcing_idx(self):
        """Per-step forcing index for the time-varying series (zELA, U, P, bl).

        Step-START (xsimlab-native) convention (audit m22): step ``tj``
        integrates the interval ``[t[tj-1], t[tj]]`` and consumes ``series[tj-1]``;
        the ``tj = 0`` initialisation flux consumes ``series[0]``. So ``series[0]``
        is consumed and ``series[nt-1]`` is unused — aligned with siim2d, where
        xsimlab feeds the step-start value. (Constant/scalar series are
        index-invariant, so every default run is bit-for-bit unchanged.)
        """
        return self.tj - 1 if self.tj > 0 else 0

    def locate_divide(self):
        # divide is located somewhere in the middle of the domain
        if (self.left_bc == "base_level") & (self.right_bc == "base_level"):

            # find peak
            didx = np.argmax(self.z)

            # guard flat / edge-peaked profiles (the flank slices below
            # would be empty and np.max would raise)
            if didx == 0:
                self.didx_l, self.didx_r = 0, 1
            elif didx == self.nx - 1:
                self.didx_l, self.didx_r = self.nx - 2, self.nx - 1
            else:
                # find next highest points on either side of peak
                left_max = np.max(self.z[:didx])
                right_max = np.max(self.z[didx+1:])

                # divide is between two highest points
                if left_max > right_max:
                    self.didx_l = didx - 1
                    self.didx_r = didx
                else:
                    self.didx_l = didx
                    self.didx_r = didx + 1

            self.xo_l = (self.xo + self.dx) / 2
            self.xo_r = (self.xo + self.dx) / 2

        # divide is at right boundary
        # there is no right side so we set didx_r to nx to index empty array
        elif self.right_bc == "reflecting":
            self.didx_l = self.nx - 1
            self.didx_r = self.nx
            # head node sits at xo, aligned with the analytical's head boundary.
            self.xo_l = self.xo
            self.xo_r = 0

        # divide is at left boundary
        # there is no left side, so we set didx_l to -1 to index empty array
        elif self.left_bc == "reflecting":
            self.didx_l = -1
            self.didx_r = 0
            self.xo_l = 0
            self.xo_r = self.xo

    # functions for ice flux calculation
    def calculate_ice_flux(self):
        fj = self._forcing_idx
        P_now = self.P_run[fj]
        B_cap = P_now if self.cap_ice_accumulation else np.inf
        self.B, self.Qg, self.Qf = _solve_ice_flux(
            self.x, self.z, self.zELA_run[fj], self.beta,
            self.d, self.sigma, self.k_h, P_now,
            self.xo_l, self.xo_r, self.didx_l, self.didx_r, self.nx, B_cap
        )

    def _glacial_params_and_code(self):
        """(law_code, GlacialParams) for the 1D mode-B walk. The walk does only
        the H-closure, so just the closure fields are real (per law; the rest
        0.0); the per-law dispatch reads only its own fields."""
        if self.sliding_law == "power":
            return LAW_POWER, GlacialParams(
                cg=self.cg, lambda_p=self.lambda_p, hc_over_H=self.hc_over_H)
        elif self.sliding_law == "coulomb":
            return LAW_COULOMB, GlacialParams(
                cg=self.cg, lambda_c=self.lambda_c, tau_c=self.tau_c,
                coulomb_clamp=self.coulomb_clamp, rho_g_g=self.rho_g * self.g,
                hc_over_H=self.hc_over_H)
        return LAW_EFFEXP, GlacialParams(
            cg=self.cg, lambda_p=self.lambda_p, hc_over_H=self.hc_over_H)

    def calculate_ice_thickness(self):
        """
        Calculate ice thickness with bidirectional flow from divide.

        Per-step mode (self._mode_B_active):
          * Mode A: slope is read off the current state self.z, then
            H is solved per-node from the sliding-law H equation using that
            explicit slope.
          * Mode B (default; all three laws): the upstream walk solves H and the
            new ice-surface slope jointly per node, using self.zb (the persistent
            bedrock state) and the receiver's just-solved
            z_s = zb_r + hc_over_H*H_r.
        """
        # Mode B path: the upstream walk solves H and
        # z_s = zb + hc_over_H*H jointly per node (tight closure), dispatched
        # by sliding law.
        if self._mode_B_active:
            if self.sliding_law not in ("eff-exp", "power", "coulomb"):
                raise ValueError(
                    f"Mode B not implemented for sliding_law='{self.sliding_law}'.")
            law_code, p = self._glacial_params_and_code()
            _diag_walk(self.zb, self.Qg, self.H, law_code, p,
                       self.dx, self.didx_l, self.didx_r, self.nx)
            self._diffuse_interior_1d(self.H, self.H_diffusivity)
            np.maximum(self.H, 0.0, out=self.H)
            zs = self.zb + self.hc_over_H * self.H
            ice_surface_slope = np.zeros_like(self.x)
            if self.didx_l >= 0:
                ice_surface_slope[1:self.didx_l+1] = np.diff(zs[:self.didx_l+1]) / self.dx
            if self.didx_r < self.nx:
                ice_surface_slope[self.didx_r:self.nx-1] = -np.diff(zs[self.didx_r:]) / self.dx
            np.clip(ice_surface_slope, 0.0, None, out=ice_surface_slope)
            self.H[self.Qg == 0] = 0.0
            self.H[self.H < 0] = 0.0
            self.H[~np.isfinite(self.H)] = 0.0
            # Outlet diagnostics: fixed-gradient outflow — the slope at a
            # through-flowing outlet is the reflected upwind surface slope.
            if self.didx_l >= 0 and self.H[0] > 0.0:
                ice_surface_slope[0] = max(0.0, (zs[1] - zs[0]) / self.dx)
            if self.didx_r < self.nx and self.H[self.nx - 1] > 0.0:
                ice_surface_slope[self.nx - 1] = max(
                    0.0, (zs[self.nx - 2] - zs[self.nx - 1]) / self.dx)
            self._sliding_velocity_from_tau(ice_surface_slope)
            return

        # Calculate ice surface slope for all nodes
        ice_surface_slope = np.zeros_like(self.x)
        ice_surface_slope[1:self.didx_l+1] = np.diff(self.z[:self.didx_l+1]) / self.dx
        ice_surface_slope[self.didx_r:self.nx-1] = -np.diff(self.z[self.didx_r:]) / self.dx
        # Base-level outlets: in mode A the *surface* is the anchored BC
        # (z is pinned at base level), so the outlet's slope is the upwind
        # surface slope into it, and its H comes from the same per-law
        # closure as every interior node.
        if self.didx_l >= 0:
            ice_surface_slope[0] = max(0.0, (self.z[1] - self.z[0]) / self.dx)
        if self.didx_r < self.nx:
            ice_surface_slope[self.nx - 1] = max(
                0.0, (self.z[self.nx - 2] - self.z[self.nx - 1]) / self.dx)

        # calculate D parameter and solve for H. With kt now inside cg, the H-equations
        # carry no explicit kt (Qg is m^3/yr, cg is m^-3/yr).
        if self.sliding_law == "power":
            # Power H eq: H*(1 + (lambda_p/H)^2)^(1/6) = (Qg/(cg*S^3))^(1/6)
            with np.errstate(divide='ignore', invalid='ignore'):
                self.D = (self.Qg / (self.cg * self.lambda_p**2 * ice_surface_slope**3)) ** (1/4)
            if self.didx_l >= 0:
                self.H[:self.didx_l+1] = _solve_ice_thickness_power_analytical_vec(self.D[:self.didx_l+1], self.lambda_p)
            if self.didx_r < self.nx:
                self.H[self.didx_r:] = _solve_ice_thickness_power_analytical_vec(self.D[self.didx_r:], self.lambda_p)

        elif self.sliding_law == "coulomb":
            rho_g_g = self.rho_g * self.g
            with np.errstate(divide='ignore', invalid='ignore'):
                # Coulomb H eq: H(H+Omega)^(1/5) = (Qg/(cg*S^3))^(1/5).
                D = (self.Qg / (self.cg * ice_surface_slope**3)) ** 0.2
                a = (rho_g_g * ice_surface_slope / self.tau_c) ** 3
            if self.didx_l >= 0:
                self.H[:self.didx_l+1] = _solve_ice_thickness_coulomb_vec(
                    D[:self.didx_l+1], a[:self.didx_l+1], self.lambda_c, self.coulomb_clamp)
            if self.didx_r < self.nx:
                self.H[self.didx_r:] = _solve_ice_thickness_coulomb_vec(
                    D[self.didx_r:], a[self.didx_r:], self.lambda_c, self.coulomb_clamp)

        elif self.sliding_law == "eff-exp":
            # eff-exp H closed-form: H = (Qg/(cg*lambda_p^(3/2)*S^3))^(2/9)
            with np.errstate(divide='ignore', invalid='ignore'):
                self.H = (self.Qg / (self.cg * self.lambda_p**1.5 * ice_surface_slope**3))**(2/9)

        self.H[self.Qg == 0] = 0.0
        self.H[self.H < 0] = 0.0
        # isfinite, not isnan: the eff-exp closed form gives H=inf where the
        # slope is 0 (e.g. the outlet node), and inf would diffuse inward.
        self.H[~np.isfinite(self.H)] = 0.0

        # optional H diffusion (smooths the ice profile before tau/ub).
        self._diffuse_interior_1d(self.H, self.H_diffusivity)
        np.maximum(self.H, 0.0, out=self.H)
        # re-zero ice-free cells AFTER diffusion (mode B's order): diffusion
        # smears ice past the terminus and nothing else removes it.
        self.H[self.Qg == 0] = 0.0

        self._sliding_velocity_from_tau(ice_surface_slope)

    def _sliding_velocity_from_tau(self, ice_surface_slope):
        """Set self.tau and basal sliding velocity self.ub (m/yr) from H and the
        given ice-surface slope. Shared by the mode-A and mode-B thickness paths."""
        self.tau = self.rho_g * self.g * self.H * ice_surface_slope
        if self.sliding_law == "coulomb":
            # Guard the tau = tau_c pole (audit m20): the closure keeps tau <
            # tau_c, but a post-solve H diffusion can push the diagnostic slope
            # past it, sending y -> 1 and ub negative. Clamp y just below 1
            # (mirroring the closure's coulomb_clamp) so the reported sliding
            # velocity stays finite and positive at/above the yield stress.
            y = np.minimum((self.tau / self.tau_c) ** 3, 1.0 - self.coulomb_clamp)
            self.ub = self.kt * (2.0 * self.Ac / 5.0) * self.lambda_c * self.tau_c**3 * y / (1.0 - y)
        else:  # power, eff-exp
            with np.errstate(divide='ignore', invalid='ignore'):
                self.ub = self.kt * (2.0 * self.Ac / 5.0) * self.tau**3 * self.lambda_p**2 / self.H
        self.ub[self.H <= 0] = 0.0
        self.ub[~np.isfinite(self.ub)] = 0.0

    def _erode_border_bed_1d(self):
        """Mode B: evolve the bed at base-level outlets under the OUTFLOW BC —
        the IMPLICIT BORDER BUDGET. An
        icy border bed keeps ERODING: dzb/dt = U − f·E, with E the glacial law
        on the interior ARRIVAL slope (zs[nb2] − zs[nb], floored at
        constants.S_FLOOR_BC — the ~0 local slope of zero-gradient-H would
        starve it into a sill) and f the flotation ramp evaluated at the NEW
        bed, integrated by the closed-form backward-Euler step
        (_core.skeleton._implicit_border_step). The bed approaches the
        flotation-draft equilibrium zb* = bl − hc·H + δ·U/E monotonically at
        any dt: no explicit overshoot (the gate-era failure), no sill (the
        zerograd failure — the digging border always hands the donor a real
        receiver drop). flotation_gate off → f ≡ 1, unbounded (diagnostics
        only). Ice-free, a bed below the datum is uplifted toward bl
        (post-glacial recovery, clamped at bl). The uplift rate is the node's
        tectonic U unless border_bed_uplift overrides it."""
        fj = self._forcing_idx
        bl = self.bl_run[fj]
        law_code, _ = self._glacial_params_and_code()
        zs = self.zb + self.hc_over_H * self.H
        for b, nb, nb2 in (((0, 1, 2),) if self.didx_l >= 0 else ()) + \
                          (((self.nx - 1, self.nx - 2, self.nx - 3),)
                           if self.didx_r < self.nx else ()):
            U_b = (self.U_matrix[b, fj] if self.border_bed_uplift is None
                   else self.border_bed_uplift)
            if self.Qg[b] <= 0.0:
                if self.zb[b] < bl:
                    self.zb[b] = min(self.zb[b] + U_b * self.dt, bl)
                continue
            S_arr = max(_constants.S_FLOOR_BC, (zs[nb2] - zs[nb]) / self.dx)
            E = _modeb_border_erosion(law_code, self.Qg[b], S_arr, self.H[b],
                                      self.Co, self.mu, self.nu, self.ce,
                                      self.cg, self.alpha_g, self.lambda_p)
            if self.flotation_gate:
                self.zb[b] = _implicit_border_step(
                    self.zb[b], U_b, E, self.dt, self.H[b], self.hc_over_H,
                    bl, self.flotation_ramp)
            else:                    # unbounded control (diagnostics only)
                self.zb[b] = self.zb[b] + (U_b - E) * self.dt

    # erosion functions
    def calculate_erosion(self):

        if self.sliding_law == "eff-exp":
            if (self.nu == 1) & (self.n == 1):
                _linear_erode(self.z, self.z_o, self.Qf, self.Qg,
                              (self.dt / self.dx) * self.Ko,
                              (self.dt / self.dx) * self.Co,
                              self.m, self.mu, self.nx, self.didx_l, self.didx_r)
            else:
                _nonlinear_erode(self.z, self.z_o, self.Qf, self.Qg,
                                 (self.dt / self.dx**self.n) * self.Ko,
                                 (self.dt / self.dx**self.nu) * self.Co,
                                 self.m, self.mu, self.nx, self.n, self.nu,
                                 self.didx_l, self.didx_r)

        elif self.sliding_law == "power":
            # E = G_o * S^(3*ell/2),
            # G_o = ce * (cg * lambda_p^2 * Qg / (alpha_g^2 * (1 + (H/lambda_p)^2)))^(ell/2)
            t = 3.0 * self.ell / 2.0
            Kf = (self.dt / self.dx**self.n) * self.Ko
            Kg_prefactor = ((self.dt / self.dx**t) * self.ce
                            * (self.cg * self.lambda_p**2
                               / (self.alpha_g**2.0)) ** (self.ell / 2.0))
            _power_erode(self.z, self.z_o, self.Qf, self.Qg, self.H,
                         Kf, Kg_prefactor, self.m, self.nx, self.n, t,
                         self.lambda_p, self.didx_l, self.didx_r)

        elif self.sliding_law == "coulomb":
            # Regularized Coulomb. H and R are re-solved inside the z-Newton;
            # A_const absorbs the S-, H-, R-, Qg-independent factors. With kt
            # inside cg, no explicit kt enters here.
            t = 6.0 * self.ell / 5.0
            Kf = (self.dt / self.dx**self.n) * self.Ko
            A_const = (self.ce
                       * (self.cg ** 0.4 / self.alpha_g) ** self.ell
                       * self.dt / (self.dx ** t))
            _coulomb_erode(self.z, self.z_o, self.Qf, self.Qg,
                              Kf, A_const, self.m, self.nx, self.n,
                              self.ell, t, self.cg, self.rho_g * self.g,
                              self.tau_c, self.lambda_c, self.dx,
                              self.coulomb_clamp,
                              self.didx_l, self.didx_r)

        else:
            raise ValueError(f"Unknown sliding_law: '{self.sliding_law}'. "
                             "Options: 'eff-exp', 'power', 'coulomb'")

        # calculate erosion rate in-place
        np.subtract(self.z_o, self.z, out=self.erosion_rate)
        self.erosion_rate /= self.dt

    def lake_fill_1d(self, zb):
        """Monotone-fill on a 1D bedrock profile.

        Walks each side of the divide from outlet up; any node lower than its
        receiver is raised to the receiver's elevation (the 1D analog of
        priority-flood depression filling — every closed basin spills at its
        downstream rim). Returns a copy; does not mutate `zb`. Use this to
        preview what a flow router would see after ice retreats, before
        committing to the structural switch.
        """
        zb = np.asarray(zb, dtype=float).copy()
        _lake_fill_1d(zb, self.didx_l, self.didx_r, self.nx)
        return zb

    def _diffuse_interior_1d(self, arr, D):
        """Explicit FD diffusion of `arr` on interior nodes [1:-1] (in-place).
        Boundary nodes are held fixed. Sub-steps to keep alpha/n_sub ≤ 0.4.
        No-op when D is None or 0."""
        if D is None or D <= 0.0:
            return
        alpha = D * self.dt / (self.dx * self.dx)
        n_sub = max(1, int(np.ceil(alpha / 0.4)))
        a_sub = alpha / n_sub
        for _ in range(n_sub):
            arr[1:-1] = arr[1:-1] + a_sub * (arr[:-2] - 2.0 * arr[1:-1] + arr[2:])

    def calculate_uplift(self):
        fj = self._forcing_idx
        if (self.right_bc == 'reflecting') & (self.left_bc == 'base_level'):
            self.z[1:] += self.U_matrix[1:, fj] * self.dt

        elif (self.left_bc == 'reflecting') & (self.right_bc == 'base_level'):
            self.z[:-1] += self.U_matrix[:-1, fj] * self.dt

        elif (self.right_bc == 'base_level') & (self.left_bc == 'base_level'):
            self.z[1:-1] += self.U_matrix[1:-1, fj] * self.dt

        else:
            raise ValueError("Boundary conditions not recognized for uplift calculation.")

        # print(max(self.z), self.U_matrix[1:-1, fj] * self.dt)

    def calculate_surface_evolution(self):

        # Mode B path (all three laws): z_b is the persistent state.
        #   1) uplift z_b on interior nodes.
        #   2) build z' = z_b + hc_over_H*H (H frozen from
        #      calculate_ice_thickness's
        #      upstream walk this step).
        #   3) erode z'. Two behaviours, set by floating_termini:
        #      - default (True): a terminus in a closed basin is floating /
        #        decoupled — lake-fill z' and BW-erode the FILLED surface, so a
        #        basin suppresses erosion in its interior with or without ice
        #        (delta=0 by construction at lake-interior cells).
        #      - False: the ice is grounded — BW-erode the RAW z' (one pass),
        #        then zero erosion only at cells that are in a lake AND ice-free,
        #        so the ice toe carves its overdeepening; only bare-rock lakes
        #        are frozen.
        #   4) erosion delta applied to z_b; carved depressions persist as state.
        #   5) z = z_b + hc_over_H*H for next step's routing (shows actual
        #      depressions).
        if self._mode_B_active:
            fj = self._forcing_idx
            bl = self.bl_run[fj]  # per-step water-line datum
            # uplift on z_b (state) using the same BC pattern as calculate_uplift
            if (self.right_bc == 'reflecting') & (self.left_bc == 'base_level'):
                self.zb[1:] += self.U_matrix[1:, fj] * self.dt
            elif (self.left_bc == 'reflecting') & (self.right_bc == 'base_level'):
                self.zb[:-1] += self.U_matrix[:-1, fj] * self.dt
            elif (self.right_bc == 'base_level') & (self.left_bc == 'base_level'):
                self.zb[1:-1] += self.U_matrix[1:-1, fj] * self.dt
            else:
                raise ValueError("Boundary conditions not recognized for uplift calculation.")

            # build z' = z_b + hc_over_H*H (the real ice surface). This is the ROCK
            # problem's working view: an ICE-FREE base-level outlet presents the
            # WATER LINE max(z', bl) — its Dirichlet datum — and the lake-fill
            # below propagates that water level over a submerged boundary trough
            # exactly as it floods any interior overdeepening (a submerged toe is
            # raised to the water level and decoupled: floating, no erosion). An
            # ICY (outflow) outlet keeps its true surface (a free outflow, not
            # still base water). The ice problem never sees this view.
            np.add(self.zb, self.hc_over_H * self.H, out=self.z)
            if self.didx_l >= 0 and self.H[0] <= 0.0:
                self.z[0] = max(self.z[0], bl)
            if self.didx_r < self.nx and self.H[self.nx - 1] <= 0.0:
                self.z[self.nx - 1] = max(self.z[self.nx - 1], bl)

            if self.floating_termini:
                # floating/decoupled: lake-fill z' as a slope view and BW-erode
                # the FILLED surface, so a lake suppresses erosion in its
                # interior with or without ice.
                self.z[:] = self.lake_fill_1d(self.z)
                np.copyto(self.z_o, self.z)
                self.calculate_erosion()
            else:
                # grounded: erode the RAW surface (one pass), then turn off
                # erosion only in ice-free lakes; ice-covered cells keep their
                # real-slope erosion and carve the overdeepening.
                z_filled = self.lake_fill_1d(self.z)
                lake_icefree = (z_filled > self.z + 1e-6) & (self.H <= 0.0)
                np.copyto(self.z_o, self.z)
                self.calculate_erosion()
                if lake_icefree.any():
                    self.z[lake_icefree] = self.z_o[lake_icefree]
                    self.erosion_rate[lake_icefree] = 0.0

            # INTERIOR flotation gate: scale glacial erosion at icy interior
            # cells by the effective-pressure ramp
            # f = clip((zs - bl)/(gamma*hc*H), 0, 1) (f = 0 exactly for
            # zs <= bl — no incision beneath the waterline; gamma =
            # flotation_ramp, 0 = the hard binary gate). The ramp removes the
            # binary gate's single-step E*dt overshoot at the toe. The border
            # is not gated (its bed is slaved below). Off for diagnostics only.
            if self.flotation_gate:
                zs_true = self.zb + self.hc_over_H * self.H
                icy = self.H > 0.0
                if self.flotation_ramp > 0.0:
                    d = self.flotation_ramp * self.hc_over_H * self.H
                    f = np.ones_like(d)
                    np.divide(zs_true - bl, d, out=f, where=d > 0.0)
                    np.clip(f, 0.0, 1.0, out=f)
                    m = icy & (f < 1.0)         # d <= 0 keeps f = 1 (excluded)
                    if m.any():
                        self.z[m] = self.z_o[m] - f[m] * (self.z_o[m] - self.z[m])
                        self.erosion_rate[m] *= f[m]
                else:                            # gamma = 0: hard binary gate
                    afloat = (zs_true < bl) & icy
                    if afloat.any():
                        self.z[afloat] = self.z_o[afloat]
                        self.erosion_rate[afloat] = 0.0

            # erosion delta applied to actual z_b state — carved depressions persist
            self.zb -= (self.z_o - self.z)

            # Border bed at base-level outlets (OUTFLOW BC): the erosion
            # kernels skip outlets; an icy outlet bed erodes by the IMPLICIT
            # arrival-slope budget (closed-form step, ramp-bounded at the
            # flotation draft), and recovers toward the datum at U when
            # ice-free.
            self._erode_border_bed_1d()

            # rebuild z for next step's routing, mass balance and output as
            # z_s = z_b + hc_over_H*H from current state — the TRUE state
            # everywhere (the public output convention):
            # a relict drowned border bed shows through below bl, while a
            # through-flowing icy (outflow) border stands at its true surface
            # (no floor). Interior trough nodes likewise present
            # their drowned bed, so the mass balance melts at the real (deep)
            # elevation and ice flux dies crossing an empty trough. The bl floor
            # + lake fill live ONLY in the erosion working view above; water is a
            # display layer at bl(t), not stored state.
            np.add(self.zb, self.hc_over_H * self.H, out=self.z)
            return

        # default path: ice-surface tracking (z is state, z_b is bookkep)
        self.calculate_uplift()

        # Anchor the base-level outlet(s) at the water-line datum bl (mode A's
        # Dirichlet BC: z[outlet] is otherwise held only by uplift+erosion both
        # skipping it, so it stays at its initial value). Applied before erosion
        # so interior slopes grade to bl; a step in bl(t) enters here as a
        # knickpoint source. bl = 0 default -> the historical z[outlet] = 0.
        bl = self.bl_run[self._forcing_idx]
        if self.left_bc == 'base_level':
            self.z[0] = bl
        if self.right_bc == 'base_level':
            self.z[self.nx - 1] = bl

        # copy landscape into pre-allocated buffer
        np.copyto(self.z_o, self.z)

        # calculate erosion
        self.calculate_erosion()

        # update bedrock surface in-place (channel floor: zb = z - hc*H)
        np.subtract(self.z, self.hc_over_H * self.H, out=self.zb)

    # main run functions
    def initialize_simulation(self):
        """
        Initializes the simulation state.
        """

        # initial topography (m)
        if self.initial_topography is not None:
            self.z = self.initial_topography.copy()
        elif self.left_bc == "reflecting":
            self.z = 1000 * (np.linspace(0, 1, self.nx)[::-1])**2
        elif self.right_bc == "reflecting":
            self.z = 1000 * (np.linspace(0, 1, self.nx))**2
        else:
            split = int(self.nx / 2)
            self.z = 1000 * np.hstack(((np.linspace(0, 1, split))**2, (np.linspace(0, 1, self.nx - split)[::-1])**2))

        # initial ice thickness (m)
        self.H = np.ones_like(self.x) * 1e-3

        # initial bedrock (m): channel floor under the (tiny) initial ice
        self.zb = self.z - self.hc_over_H * self.H

        # pre-allocate scratch buffers (reused every time step)
        self.z_o = np.empty_like(self.x)
        self.erosion_rate = np.zeros_like(self.x)
        self.tau = np.zeros_like(self.x)
        self.ub = np.zeros_like(self.x)

        # locate divide
        self.locate_divide()

        # initial ice accumulation rate (m/yr)
        self.B = np.zeros_like(self.x)

        # initial ice flux
        self.tj = 0
        self.calculate_ice_flux()

        # prepare output storage aligned with requested sampling interval
        self._configure_output_schedule()
        n_snapshots = self.output_steps.size
        self.zb_out = np.zeros((self.nx, n_snapshots))
        self.z_out = np.zeros((self.nx, n_snapshots))
        self.H_out = np.zeros((self.nx, n_snapshots))
        self.Qg_out = np.zeros((self.nx, n_snapshots))
        self.Qf_out = np.zeros((self.nx, n_snapshots))
        self.erosion_rate_out = np.zeros((self.nx, n_snapshots))
        self.tau_out = np.zeros((self.nx, n_snapshots))
        self.ub_out = np.zeros((self.nx, n_snapshots))
        self.B_out = np.zeros((self.nx, n_snapshots))
        self.zELA_out = np.zeros(n_snapshots)

        # divide + terminus tracking: length n_sides on the first axis.
        # n_sides = 2 for base_level/base_level (ordered left, right),
        # n_sides = 1 for a reflecting-boundary setup (only the active side).
        self.n_sides = int(self.didx_l >= 0) + int(self.didx_r < self.nx)
        self.zo_out = np.zeros((self.n_sides, n_snapshots))
        self.xd_out = np.zeros((self.n_sides, n_snapshots))
        self.Lt_out = np.zeros((self.n_sides, n_snapshots))
        self.zLt_out = np.zeros((self.n_sides, n_snapshots))

        # save initial state and run simulation
        self._next_output_idx = 0
        self._maybe_save_snapshot()

    def run(self):
        """
        Runs the simulation.
        """

        # initialize simulation
        self.initialize_simulation()

        if self.progress_bar:
            for jj, self.tj in enumerate(tqdm.tqdm(range(1, self.nt))):
                self.step()
        else:
            for jj, self.tj in enumerate(range(1, self.nt)):
                self.step()

    def _step_body(self):
        """Per-step physics for the current integration mode (self.mode)."""
        # NB (audit N30, accepted asymmetry): 1D solves H (calculate_ice_thickness)
        # on the PRE-uplift bed — uplift is applied inside calculate_surface_evolution
        # below — whereas the 2D kernels receive the post-uplift bed. This is an
        # O(U*dt) per-step ordering difference (WON'T-FIX; sub-resolution).
        self.locate_divide()
        self.calculate_ice_flux()
        self.calculate_ice_thickness()
        self.calculate_surface_evolution()

    def step(self):
        """Run one time step. INTERNAL — driven by :meth:`run`, which advances
        the step index ``self.tj`` around each call. Calling this standalone
        reuses the previous step's forcing index (no snapshot advance); use
        :meth:`run` to integrate (audit N15)."""
        self._step_body()
        self._maybe_save_snapshot()

    # output functions
    def _configure_output_schedule(self):
        # nt_out = number of output frames (matches siim2d); None saves every step.
        if self.nt_out is None:
            self.output_steps = np.arange(self.nt, dtype=int)
        else:
            self.output_steps = np.unique(
                np.round(np.linspace(0, self.nt - 1, int(self.nt_out))).astype(int))
        self.output_times = self.t[self.output_steps]

    def _save_snapshot(self, snapshot_idx, time_idx):
        self.zb_out[:, snapshot_idx] = self.zb
        self.z_out[:, snapshot_idx] = self.z
        self.H_out[:, snapshot_idx] = self.H
        self.B_out[:, snapshot_idx] = self.B
        self.Qg_out[:, snapshot_idx] = self.Qg
        self.Qf_out[:, snapshot_idx] = self.Qf
        self.erosion_rate_out[:, snapshot_idx] = self.erosion_rate
        self.tau_out[:, snapshot_idx] = self.tau
        self.ub_out[:, snapshot_idx] = self.ub
        self.zELA_out[snapshot_idx] = self.zELA_run[time_idx]
        zo, xd, Lt, zLt = self._compute_divide_and_termini()
        self.zo_out[:, snapshot_idx] = zo
        self.xd_out[:, snapshot_idx] = xd
        self.Lt_out[:, snapshot_idx] = Lt
        self.zLt_out[:, snapshot_idx] = zLt

    def _compute_divide_and_termini(self):
        """Per-side divide and terminus positions/elevations for the current
        state. A side is active per the boundary conditions: left when
        didx_l >= 0, right when didx_r < nx. Output order is (left, right)
        for two-sided runs; a one-sided run returns a length-1 array for the
        active side. Terminus is located via Qg > 0 (NaN if that side is
        ice-free)."""
        zo, xd, Lt, zLt = [], [], [], []
        if self.didx_l >= 0:
            zo.append(self.z[self.didx_l])
            xd.append(self.x[self.didx_l])
            mask = self.Qg[:self.didx_l+1] > 0
            if mask.any():
                idx = int(np.argmax(mask))  # leftmost glaciated node
                Lt.append(self.x[idx]); zLt.append(self.z[idx])
            else:
                Lt.append(np.nan); zLt.append(np.nan)
        if self.didx_r < self.nx:
            zo.append(self.z[self.didx_r])
            xd.append(self.x[self.didx_r])
            mask = self.Qg[self.didx_r:] > 0
            if mask.any():
                offset = len(mask) - 1 - int(np.argmax(mask[::-1]))
                idx = self.didx_r + offset  # rightmost glaciated node
                Lt.append(self.x[idx]); zLt.append(self.z[idx])
            else:
                Lt.append(np.nan); zLt.append(np.nan)
        return np.array(zo), np.array(xd), np.array(Lt), np.array(zLt)

    def _maybe_save_snapshot(self):
        if self._next_output_idx >= self.output_steps.size:
            return
        if self.tj == self.output_steps[self._next_output_idx]:
            self._save_snapshot(self._next_output_idx, self.tj)
            self._next_output_idx += 1

    def rms_vs_analytical(self, i=-1):
        """Compute RMS difference between the model's profile at output step
        ``i`` (default last) and its analytical SS solution.

        Both sides use the channel-floor datum so a perfect model gives
        ``bed_rms ~ 0``: the model bed is ``z - hc_over_H*H`` and the
        analytical bed already carries the ``HC_OVER_H`` factor
        (``a.bed = a.surface - HC_OVER_H*H``). Bedrock can be negative inside
        thick glaciers and must not be clipped.

        Returns
        -------
        (surface_rms, bed_rms) : tuple of float
            RMS values in metres. NaN if the analytical has no SS
            (``glacier_flag == -1``).
        """
        a = self.analytical
        if a.surface is None:
            return float('nan'), float('nan')
        z_num = self.z_out[:, i]
        H_num = self.H_out[:, i]
        zb_num = z_num - self.hc_over_H * H_num          # channel-floor datum (audit B4)
        a_H_grid = a.surface - a.bed                    # = HC_OVER_H * H (analytical bed carries the factor)
        if (a.x.shape == self.x.shape) and np.allclose(a.x, self.x):
            a_surf, a_H = a.surface, a_H_grid
        else:
            # x arrays decrease from L → 0; flip for np.interp's monotone-x rule.
            a_surf = np.interp(self.x[::-1], a.x[::-1], a.surface[::-1])[::-1]
            a_H    = np.interp(self.x[::-1], a.x[::-1], a_H_grid[::-1])[::-1]
        a_bed = a_surf - a_H                            # bedrock = surface − H
        surface_rms = float(np.sqrt(np.nanmean((z_num  - a_surf) ** 2)))
        bed_rms     = float(np.sqrt(np.nanmean((zb_num - a_bed ) ** 2)))
        return surface_rms, bed_rms


# Plotting class
class siim_plotter:
    """Handles all visualization for the siim model.
    Accesses model state via self.model reference."""

    def __init__(self, model):
        self.model = model

    def analytical(self, ax=None, alpha_ice=0.15, alpha_zb=0.2,
                   plot_analytical=True, bistable=True):
        """Plot the analytical SS profile. Handles all flags 1–5; for the
        bistable flags (4, 5), overlays the fluvial alt in light gray when
        ``bistable=True``."""
        m = self.model
        if not plot_analytical:
            return
        analytical_surface, analytical_bed = m.analytical._analytical_profiles()

        if analytical_surface is None or analytical_bed is None:
            print('No analytical solution available for the current model setup.')
            return

        flag = m.analytical.glacier_flag
        if flag == -1:
            return
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(15, 5))

        # Plot on the analytical's own grid (m.x and m.analytical.x can have
        # different nx when the user does not pass an explicit nx).
        x_km = m.analytical.x / 1e3
        ax.fill_between(x_km, np.zeros_like(x_km), analytical_bed,
                        color='gray', alpha=alpha_zb)
        ax.plot(x_km, analytical_bed, "k-", lw=0.5)
        if flag in (2, 3, 4, 5):
            ax.fill_between(x_km, analytical_bed, analytical_surface,
                            color='blue', alpha=alpha_ice, edgecolor=None)
            ax.plot(x_km, analytical_surface, "b--", lw=1)
        if bistable and flag in (4, 5) and m.analytical.surface_alt is not None:
            ax.plot(x_km, m.analytical.surface_alt, "--",
                    color="gray", alpha=0.5, lw=1, label="Analytical fluvial alt")
        ax.set_ylim(0, 1.3 * np.max(analytical_surface))
        ax.set_xlim(0, m.analytical.L / 1e3)

    def _water_display_1d(self, i):
        """Reconstruct the water DISPLAY layer for output frame ``i`` (the
        elevation panel's ponds + sea).

        Water is not stored state — ``z_out`` is the TRUE bed+ice surface
        (true-state output convention). This mirrors the model's erosion
        working view: an ICE-FREE base-level outlet presents the WATER LINE
        ``max(zs, bl)`` (its Dirichlet datum, so the MINIMUM waterline is
        ``bl`` — 0 by default — NOT the outlet's own bed sill), then
        :meth:`lake_fill_1d` propagates that level up over any submerged
        trough. An icy (outflow) outlet keeps its true surface (no floor).

        Returns ``(zs_true, z_fill, wet)``: the true presented surface, the
        lake-filled water surface, and the boolean wet mask
        (``z_fill > zs_true``). The caller paints ``zs_true..z_fill`` on the wet
        segments (skipping ice-dammed ones). ``lake_fill_1d`` walks the run's
        FINAL divide indices, so snapshots near a divide migration flood
        approximately."""
        m = self.model
        zs_true = m.zb_out[:, i] + m.hc_over_H * m.H_out[:, i]
        ice = m.H_out[:, i] > 0.0
        bl_i = m.bl_run[m.output_steps[i]]
        z_water = m.z_out[:, i].copy()
        if m.didx_l >= 0 and not ice[0]:
            z_water[0] = max(z_water[0], bl_i)
        if m.didx_r < m.nx and not ice[m.nx - 1]:
            z_water[m.nx - 1] = max(z_water[m.nx - 1], bl_i)
        z_fill = m.lake_fill_1d(z_water)
        wet = z_fill > zs_true + 1e-2
        return zs_true, z_fill, wet

    def _draw_field(self, ax, i, field="elevation", analytical=True,
                    bistable=True, legend=True):
        """Draw a single field snapshot onto ax — the per-panel renderer for
        profile / view_profile / animate_profile.

        ``analytical`` toggles the analytical overlay; ``bistable`` (default
        True) additionally overlays the fluvial alt SS in light gray when the
        analytical is in a bistable regime (flags 4, 5). ``legend=False``
        suppresses the per-panel legend."""
        import warnings
        m = self.model
        x_km = m.x / 1e3
        analytical_alt = None
        if analytical:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                analytical_surface, analytical_bed = m.analytical._analytical_profiles()
            show_analytical_bed = analytical_bed is not None and (
                analytical_surface is None
                or not np.allclose(analytical_bed, analytical_surface, equal_nan=True)
            )
            if bistable and m.analytical.glacier_flag in (4, 5):
                analytical_alt = m.analytical.surface_alt
        else:
            analytical_surface = analytical_bed = None
            show_analytical_bed = False

        if field == "elevation":
            zb_i = m.zb_out[:, i]
            z_i = m.z_out[:, i]
            ice = m.H_out[:, i] > 0.0
            ax.fill_between(x_km, zb_i, z_i, where=ice, color="blue", alpha=0.15)
            # Water DISPLAY layer (ponds + sea): reconstructed here because it is
            # not stored state — the floor at the base-level datum bl and the
            # lake fill live only in the model's working view (see
            # _water_display_1d). Paint zs_true..z_fill on wet segments, skipping
            # ICE-dammed ones (terrain ponding against a glacier surface).
            zs_true, z_fill, wet = self._water_display_1d(i)
            if wet.any():
                hw = 0.5 * abs(x_km[1] - x_km[0])
                labeled = False
                w = np.flatnonzero(wet)
                seg_starts = w[np.r_[True, np.diff(w) > 1]]
                seg_ends = w[np.r_[np.diff(w) > 1, True]]
                for a, b in zip(seg_starts, seg_ends):
                    # downstream (toward-outlet) dam cell; skip ice-dammed
                    dam = a - 1 if b <= m.didx_l else (b + 1 if a >= m.didx_r else -1)
                    if 0 <= dam < self.model.nx and ice[dam]:
                        continue
                    lbl = None if labeled else "Water"
                    if b > a:
                        ax.fill_between(x_km[a:b+1], zs_true[a:b+1],
                                        z_fill[a:b+1], color="skyblue",
                                        alpha=0.5, label=lbl)
                    else:   # single-cell pond: explicit half-cell rectangle
                        ax.fill_between([x_km[a] - hw, x_km[a] + hw],
                                        [zs_true[a], zs_true[a]],
                                        [z_fill[a], z_fill[a]],
                                        color="skyblue", alpha=0.5, label=lbl)
                    labeled = True
            if show_analytical_bed:
                ax.plot(x_km, analytical_bed, "--", color="dimgray", alpha=0.6, lw=1.0,
                        label="Analytical steady state soln (bedrock)", zorder=0)
            if analytical_surface is not None:
                ax.plot(x_km, analytical_surface, "--", color="teal", alpha=0.6, lw=1.1,
                        label="Analytical steady state soln (surface)", zorder=0)
                ax.axhline(m.zELA, color="goldenrod", linestyle="--", lw=1, label="Analytical ELA")
            if analytical_alt is not None:
                ax.plot(x_km, analytical_alt, "--", color="gray", alpha=0.4, lw=1.0,
                        label="Analytical fluvial alt", zorder=0)
            ax.axhline(m.zELA_out[i], color="goldenrod", linestyle="-", lw=1, label="ELA")
            ax.plot(x_km, zb_i, color="dimgray", label="Bedrock")
            # ice surface drawn only where there is ice — a bare (or water)
            # cell shows just its bed, so no spurious cliff connects the
            # waterline to a retreating toe
            ax.plot(x_km, np.where(ice, z_i, np.nan), color="navy", label="Ice surface")
            ax.set_ylabel("Elevation (m)")
        elif field == "ice_thickness":
            ax.fill_between(x_km, np.zeros_like(x_km), m.H_out[:, i], color="steelblue", alpha=0.2)
            ax.plot(x_km, m.H_out[:, i], color="navy", label="Ice thickness")
            ax.set_ylabel("Ice thickness (m)")
        elif field == "ice_flux":
            ax.plot(x_km, m.Qg_out[:, i], color="dodgerblue", label="Ice flux $Q_g$")
            ax.set_ylabel("Ice flux (m$^3$/yr)")
        elif field == "water_flux":
            ax.plot(x_km, m.Qf_out[:, i], color="orangered", label="Water flux $Q_f$")
            ax.set_ylabel("Water flux (m$^3$/yr)")
        elif field == "erosion_rate":
            ax.plot(x_km, m.erosion_rate_out[:, i], color="crimson", label="Erosion rate")
            ax.plot(x_km, m.U_matrix[:, m.output_steps[i]], color="black", linestyle="--", label="Uplift rate")
            ax.set_yscale("log")
            ax.set_ylabel("Erosion rate (m/yr)")
        elif field == "shear_stress":
            ax.plot(x_km, m.tau_out[:, i] / 1e3, color="purple",
                    label=r"Basal shear stress $\tau = \rho g H S$")
            if m.sliding_law == "coulomb":
                ax.axhline(m.tau_c / 1e3, color="black", linestyle="--", lw=1, label=r"$\tau_c$")
            ax.set_ylabel("Basal shear stress (kPa)")
        elif field == "sliding_velocity":
            ax.plot(x_km, m.ub_out[:, i], color="darkgreen", label=r"Sliding velocity $u_b$")
            ax.set_ylabel("Sliding velocity (m/yr)")

        ax.set_xlabel("Distance (km)")
        if legend:
            ax.legend(loc="upper left")
        ax.grid(True, alpha=0.2)

    _valid_fields = {"elevation", "ice_flux", "water_flux", "ice_thickness", "erosion_rate", "shear_stress", "sliding_velocity"}

    def _compute_field_ylims(self, fields):
        """Compute stable y-axis limits for each field across all output steps."""
        m = self.model
        fmap = {'elevation': ('zb_out', 'z_out'), 'ice_thickness': ('H_out', 'H_out'),
                'erosion_rate': ('erosion_rate_out', 'erosion_rate_out'),
                'ice_flux': ('Qg_out', 'Qg_out'), 'water_flux': ('Qf_out', 'Qf_out'),
                'shear_stress': ('tau_out', 'tau_out'),
                'sliding_velocity': ('ub_out', 'ub_out')}
        ylims = {}
        for field in fields:
            if field == 'ice_thickness':
                # Scale with the data like the 2D plotter (m31): a 500 m floor,
                # but grow for thick ice (dome/overdeepening regimes) instead of
                # silently clipping.
                hi = float(np.nanmax(m.H_out)) if m.H_out.size else 0.0
                ylims[field] = (0, max(500.0, hi * 1.05))
                continue
            lo_attr, hi_attr = fmap[field]
            lo = getattr(m, lo_attr)
            hi = getattr(m, hi_attr)
            ymin = float(np.nanmin(lo))
            ymax = float(np.nanmax(hi))
            if field == 'elevation':
                ymin = 0.0
            elif field == 'erosion_rate':
                ymin = ymax / 100
            elif ymin > 0:
                ymin = 0.0
            pad = (ymax - ymin) * 0.05 if ymax > ymin else 1.0
            ylims[field] = (ymin, ymax + pad)
        return ylims

    _PROFILE_FIELDS = ('elevation', 'ice_thickness', 'erosion_rate', 'ice_flux',
                       'water_flux', 'shear_stress', 'sliding_velocity')

    def _profile_fields(self, fields, field_min, field_max):
        """Normalize ``fields`` (str / list / {field: (min,max)|None}) into
        ``{field: (lo, hi)}`` with resolved, all-step-stable limits.
        ``field_min``/``field_max`` are the single-field shorthand."""
        if fields is None:
            fields = 'elevation'
        if isinstance(fields, str):
            norm = {fields.lower(): None}
        elif isinstance(fields, dict):
            norm = {k.lower(): v for k, v in fields.items()}
        else:
            norm = {f.lower(): None for f in fields}
        bad = [f for f in norm if f not in self._PROFILE_FIELDS]
        if bad:
            raise ValueError(f"unknown profile field(s) {bad}; allowed: "
                             f"{list(self._PROFILE_FIELDS)}")
        if len(norm) != 1 and (field_min is not None or field_max is not None):
            raise ValueError("field_min/field_max apply to a single field; use "
                             "the {field: (min, max)} dict form for per-field limits")
        auto = self._compute_field_ylims(list(norm))
        single = len(norm) == 1
        resolved = {}
        for field, override in norm.items():
            lo_a, hi_a = auto[field]
            omin = omax = None
            if override is not None:
                omin, omax = override
            if single:
                omin = field_min if field_min is not None else omin
                omax = field_max if field_max is not None else omax
            resolved[field] = (omin if omin is not None else lo_a,
                               omax if omax is not None else hi_a)
        return resolved

    def profile(self, fields='elevation', i=-1, field_min=None, field_max=None,
                basin_rank=0, ref=-1, analytical=True, bistable=True, ax=None):
        """1D profile at output step ``i`` vs the analytical SS. Same call shape
        as the 2D ``profile`` — ``basin_rank``/``ref`` are accepted but no-ops in
        1D (there is one profile). ``fields`` is a str, list, or
        ``{field: (min,max)|None}`` dict; 7 fields incl. shear_stress /
        sliding_velocity. Returns ``(fig, axes)``."""
        resolved = self._profile_fields(fields, field_min, field_max)
        m = self.model
        x_km = m.x / 1e3
        nf = len(resolved)
        if ax is not None:
            if nf != 1:
                raise ValueError("ax= is only valid for a single field")
            fig, axes = ax.figure, np.array([ax])
        else:
            fig, axes = plt.subplots(nf, 1, figsize=(15, 3 * nf), sharex=True,
                                     squeeze=False, facecolor="white")
            axes = axes[:, 0]
        for ax_k, (field, (lo, hi)) in zip(axes, resolved.items()):
            self._draw_field(ax_k, i, field=field, analytical=analytical,
                             bistable=bistable)
            ax_k.set_xlim(x_km[0], x_km[-1])
            ax_k.set_ylim(lo, hi)
        if ax is None:
            fig.tight_layout()
        return fig, axes

    def view_profile(self, fields='elevation', field_min=None, field_max=None,
                     basin_rank=0, ref=-1, analytical=True, bistable=True,
                     fig_width=12, aspect=0.27, legend=False):
        """Interactive slider over output steps (see ``profile``).

        Uses ipympl for a smooth live canvas (managed locally, so your other
        plots stay on the default backend). ``fig_width`` (inches) sets the
        on-screen size and ``aspect`` the per-panel height/width ratio;
        ``legend`` shows a static legend (off by default).
        """
        from .plotting._render import _profile_slider
        m = self.model
        if not hasattr(m, "z_out") or m.z_out.size == 0:
            raise ValueError("No simulation output; run the model first.")
        resolved = self._profile_fields(fields, field_min, field_max)
        x_km = m.x / 1e3
        times = np.asarray(m.output_times)

        def frame(axes, idx):
            for ax, (field, (lo, hi)) in zip(axes, resolved.items()):
                ax.clear()
                self._draw_field(ax, idx, field=field, analytical=analytical,
                                 bistable=bistable)
                ax.set_xlim(x_km[0], x_km[-1])
                ax.set_ylim(lo, hi)

        return _profile_slider(frame, len(times), len(resolved), times,
                               fig_width=fig_width, aspect=aspect, legend=legend)

    def animate_profile(self, fields='elevation', path=None, run_id=None,
                        field_min=None, field_max=None, basin_rank=0, ref=-1,
                        analytical=True, bistable=True, fps=20, interval=42):
        """MP4 over output steps (see ``profile``). Returns the path."""
        m = self.model
        if not hasattr(m, "z_out") or m.z_out.size == 0:
            raise ValueError("No simulation output; run the model first.")
        resolved = self._profile_fields(fields, field_min, field_max)
        nf = len(resolved)
        x_km = m.x / 1e3
        times = m.output_times
        nframes = m.z_out.shape[1]
        fig, axes = plt.subplots(nf, 1, figsize=(15, 3 * nf), sharex=True,
                                 squeeze=False, facecolor="white")
        axes = axes[:, 0]
        pbar = tqdm.tqdm(total=nframes, desc="Rendering frames")

        def update(idx):
            for ax, (field, (lo, hi)) in zip(axes, resolved.items()):
                ax.clear()
                self._draw_field(ax, idx, field=field, analytical=analytical,
                                 bistable=bistable)
                ax.set_xlim(x_km[0], x_km[-1])
                ax.set_ylim(lo, hi)
            fig.suptitle(f"t = {times[idx] / 1e3:.1f} kyr")
            pbar.update(1)
            return axes

        update(0)
        fig.tight_layout()
        name = f"{run_id}_1d" if run_id else (path or "profile_1d")
        out = output_path(name, 'movies')
        ani = anm.FuncAnimation(fig, update, frames=range(nframes), interval=interval)
        ani.save(filename=out + ".mp4", writer="ffmpeg", fps=fps, dpi=150)
        pbar.close()
        plt.close(fig)
        return out + ".mp4"

    def _pick_side(self, side=None):
        """Terminus side to analyse: the given index, or (None) the side with
        the largest terminus excursion. Ignores all-NaN (ice-free) sides."""
        if side is not None:
            return int(side)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            spreads = (np.nanmax(self.model.Lt_out, axis=1)
                       - np.nanmin(self.model.Lt_out, axis=1))
        return int(np.nanargmax(spreads)) if np.isfinite(spreads).any() else 0

    def _detect_limit_cycle(self, side=None, n_settle=None, prominence_frac=0.15):
        """Locate the settled limit cycle in the terminus-position series Lt(t).

        Returns a dict with the chosen ``side``, the (t, Lt) series, the
        detected terminus maxima, the last full cycle window ``(i0, i1)``
        bounded by the two most-settled successive maxima, the ``period``, and
        the peak-to-peak ``amplitude``. The *last* cycle is used because the
        MISI oscillation takes a few cycles to settle into a consistent period.
        """
        from scipy.signal import find_peaks
        m = self.model
        if not hasattr(m, "Lt_out") or m.Lt_out.size == 0:
            raise ValueError("No simulation output available; run the model first.")
        t = m.output_times
        side = self._pick_side(side)
        Lt = m.Lt_out[side].astype(float)
        valid = np.isfinite(Lt)
        if valid.sum() < 3:
            raise ValueError(f"Side {side} is glaciated for too few frames to find a cycle.")
        # interpolate over ice-free (NaN) gaps so peak-finding stays continuous.
        Lt_f = Lt.copy()
        if not valid.all():
            Lt_f[~valid] = np.interp(np.flatnonzero(~valid),
                                     np.flatnonzero(valid), Lt[valid])
        amp = float(np.nanmax(Lt_f) - np.nanmin(Lt_f))
        if amp <= 0:
            raise ValueError("Terminus position is constant — no limit cycle to identify.")
        # skip the leading transient; default to the second half of the run.
        if n_settle is None:
            n_settle = Lt_f.size // 2
        peaks, _ = find_peaks(Lt_f, prominence=prominence_frac * amp)
        settled = peaks[peaks >= n_settle]
        use = settled if settled.size >= 2 else peaks
        if use.size < 2:
            raise ValueError(
                "Could not identify two successive terminus maxima — the run may not "
                "contain a full settled cycle yet, or try lowering prominence_frac.")
        i0, i1 = int(use[-2]), int(use[-1])  # last (most settled) full cycle
        return dict(side=side, t=t, Lt=Lt, Lt_filled=Lt_f, peaks=peaks,
                    settled_peaks=use, i0=i0, i1=i1,
                    period=float(t[i1] - t[i0]), amplitude=amp, n_settle=n_settle)

    def _cycle_field_ylim(self, field, sample_idx, z_max=None):
        """Shared y-limits for a field across the cycle snapshot frames, so the
        n panels are directly comparable. Mirrors how ``plot`` draws each field
        (kPa for shear stress, log decades for erosion rate, fixed 0–500 for
        ice thickness); returns None to leave the axis auto-scaled."""
        m = self.model
        if field == "ice_thickness":
            return (0.0, 500.0)
        if field == "elevation":
            top = z_max if z_max is not None else 1.05 * float(np.nanmax(m.z_out[:, sample_idx]))
            return (0.0, top)
        if field == "erosion_rate":
            # include the uplift reference line in the log range.
            ero = m.erosion_rate_out[:, sample_idx].ravel()
            up = m.U_matrix[:, m.output_steps[sample_idx]].ravel()
            pos = np.concatenate([ero, up])
            pos = pos[pos > 0]
            if pos.size == 0:
                return None
            ymax = float(np.nanmax(pos))
            ymin = max(float(np.nanmin(pos)), ymax / 1e4)  # cap at 4 decades
            return (ymin, ymax * 1.5)
        attr = {"ice_flux": "Qg_out", "water_flux": "Qf_out",
                "shear_stress": "tau_out", "sliding_velocity": "ub_out"}.get(field)
        if attr is None:
            return None
        ymax = float(np.nanmax(getattr(m, attr)[:, sample_idx]))
        if field == "shear_stress":
            ymax /= 1e3  # plotted in kPa
        return (0.0, ymax * 1.05) if ymax > 0 else None

    def limit_cycle(self, side=None, n=10, field="elevation", n_settle=None,
                    prominence_frac=0.15, figsize=None, ncols=5, z_max=None,
                    analytical=None, bistable=True, mark_lt=True, legend=False,
                    save=False, path="limit_cycle", run_id=None):
        """Plot the terminus-position limit cycle.

        Top panel: terminus position ``Lt`` vs time, with detected maxima
        marked and the last (most settled) full cycle shaded. Below: a grid of
        ``n`` snapshots of ``field`` evenly spaced across that one cycle, from
        its starting maximum (most-advanced terminus) through to the next.

        Parameters
        ----------
        side : int or None
            Terminus side (0/1) to analyse. None auto-selects the side with the
            largest terminus excursion.
        n : int
            Number of snapshot panels spanning one cycle (default 10).
        field : str
            One field drawn in every snapshot panel (default 'elevation'). Any
            field ``plot`` supports: 'elevation', 'ice_thickness', 'ice_flux',
            'water_flux', 'erosion_rate' (log-scaled, with the uplift rate as a
            black dashed line), 'shear_stress', 'sliding_velocity'. All panels
            share a y-limit computed across the cycle so frames are comparable.
        n_settle : int or None
            Number of leading output frames to treat as transient and exclude
            from cycle detection. None uses the second half of the run.
        prominence_frac : float
            Maxima-detection prominence as a fraction of the Lt peak-to-peak
            amplitude (default 0.15).
        ncols : int
            Columns in the snapshot grid (default 5).
        mark_lt : bool
            Draw the terminus position Lt as a thin dashed vertical line on each
            snapshot panel (default True; skipped on ice-free frames).
        analytical : bool or None
            Overlay the analytical SS solution on each snapshot, as in
            ``view()``. None defaults to True for the 'eff-exp' sliding law
            (matching ``view``); ``bistable`` additionally overlays the fluvial
            alt SS in regimes with flags 4, 5.
        save : bool
            If True, save a PNG via output_path ({run_id}_limit_cycle or path).

        Returns
        -------
        (fig, info) : the Figure and the detection dict (side, period,
            amplitude, cycle window, ...) from ``_detect_limit_cycle``.
        """
        import matplotlib.gridspec as gridspec
        m = self.model
        if analytical is None:  # match view(): default on for eff-exp
            analytical = (m.sliding_law == 'eff-exp')
        info = self._detect_limit_cycle(side=side, n_settle=n_settle,
                                        prominence_frac=prominence_frac)
        i0, i1, t, Lt = info["i0"], info["i1"], info["t"], info["Lt"]
        # n snapshot indices evenly spaced across the cycle [i0, i1].
        sample_idx = np.unique(np.round(np.linspace(i0, i1, n)).astype(int))
        nrows = int(np.ceil(sample_idx.size / ncols))
        if figsize is None:
            figsize = (3.2 * ncols, 2.6 * (nrows + 1))
        fig = plt.figure(figsize=figsize, facecolor="white")
        gs = gridspec.GridSpec(nrows + 1, ncols, figure=fig)

        # --- top panel: terminus position vs time ---
        ax_t = fig.add_subplot(gs[0, :])
        ax_t.plot(t / 1e3, Lt / 1e3, color="navy", lw=1)
        ax_t.plot(t[info["peaks"]] / 1e3, Lt[info["peaks"]] / 1e3, "v",
                  color="darkorange", ms=6, label="terminus maxima")
        ax_t.axvspan(t[i0] / 1e3, t[i1] / 1e3, color="goldenrod", alpha=0.15,
                     label="cycle shown")
        ax_t.plot(t[sample_idx] / 1e3, Lt[sample_idx] / 1e3, "o", color="crimson",
                  ms=4, label="snapshots")
        ax_t.set_xlabel("Time (kyr)")
        ax_t.set_ylabel("Terminus $L_t$ (km)")
        ax_t.set_title(
            f"Limit cycle: side {info['side']}, period = {info['period'] / 1e3:.2f} kyr, "
            f"amplitude = {info['amplitude'] / 1e3:.2f} km")
        ax_t.legend(loc="best", fontsize=8)
        ax_t.grid(True, alpha=0.2)

        # --- snapshot grid spanning one cycle, max Lt -> next max Lt ---
        x_km = m.x / 1e3
        ylim = self._cycle_field_ylim(field, sample_idx, z_max)
        for k, idx in enumerate(sample_idx):
            r, c = divmod(k, ncols)
            ax = fig.add_subplot(gs[r + 1, c])
            self._draw_field(ax, int(idx), field=field, analytical=analytical,
                             bistable=bistable, legend=(legend and k == 0))
            ax.set_xlim(x_km[0], x_km[-1])
            if ylim is not None:
                ax.set_ylim(*ylim)
            if mark_lt and np.isfinite(Lt[idx]):  # terminus position reference
                ax.axvline(Lt[idx] / 1e3, color="dimgray", ls="--", lw=0.8,
                           zorder=5, label="$L_t$")
                if legend and k == 0:
                    ax.legend(loc="upper left")
            ax.set_title(f"t = {t[idx] / 1e3:.1f} kyr", fontsize=9)
            if c != 0:
                ax.set_ylabel("")
            if r + 1 != nrows:
                ax.set_xlabel("")
        fig.tight_layout()

        if save:
            fname = f"{run_id}_limit_cycle" if run_id else path
            fig.savefig(output_path(fname, "images") + ".png", dpi=150,
                        bbox_inches="tight")
        return fig, info

    @staticmethod
    def _x_ela_crossing(x, z, ela, ice, Lt_ref=np.nan):
        """Position where surface ``z`` crosses ``ela`` within ice (the
        snowline), or NaN if there is no crossing. When several exist, the one
        nearest ``Lt_ref`` is returned, isolating the active flank's snowline."""
        d = z - ela
        cross = np.flatnonzero((d[:-1] * d[1:] < 0) & ice[:-1] & ice[1:])
        if cross.size == 0:
            return np.nan
        xc = x[cross] + (x[cross + 1] - x[cross]) * (
            -d[cross] / (d[cross + 1] - d[cross]))
        if np.isfinite(Lt_ref):
            return float(xc[np.argmin(np.abs(xc - Lt_ref))])
        return float(xc[0])

    def _x_ela_at(self, k, Lt_k):
        """Snowline position at output frame ``k`` (ice surface ∩ ELA),
        restricted to glaciated nodes (H > 0), using the per-frame ELA."""
        m = self.model
        return self._x_ela_crossing(m.x, m.z_out[:, k], m.zELA_out[k],
                                    m.H_out[:, k] > 0, Lt_k)

    def _analytical_phase_ss(self):
        """Steady-state phase coordinates from the analytical solution:
        dict(lt=L_t, xela=snowline x), or None if there is no glacial SS."""
        import warnings
        a = self.model.analytical
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            a_surf, a_bed = a._analytical_profiles()
        if (a_surf is None or a_bed is None
                or a.glacier_flag not in (2, 3, 4, 5) or not np.isfinite(a.Lt)):
            return None
        xela = self._x_ela_crossing(a.x, a_surf, a.zELA, (a_surf - a_bed) > 0, a.Lt)
        return dict(lt=a.Lt, xela=xela, bed=float(a_bed.mean()) / a.zELA)

    # phase-portrait axis modes -> (label, can-be-negative)
    _PHASE_AXES = {
        "length": (r"Normalized glacier length, $L_t / L$", False),
        "ela": (r"Normalized ELA position, $x_{\mathrm{ELA}} / L$", False),
        "imbalance": (r"Domain-mean imbalance, $\langle \dot{E} - U \rangle$ (m/yr)", True),
        "bed": (r"Normalized mean bed, $\langle z_b \rangle / z_{\mathrm{ELA}}$", True),
    }

    def limit_cycle_phase(self, side=None, xaxis="length", yaxis="imbalance",
                          ax=None, figsize=(7, 6), cmap="viridis", lw=1.5,
                          alpha=0.8, n_settle=0, logx=False, analytical=True,
                          colorbar=True, save=False, path="limit_cycle_phase",
                          run_id=None):
        """Phase-portrait of the terminus limit cycle.

        Plots the glacier's trajectory through state space over the whole run,
        colored by time. The ``xaxis``/``yaxis`` quantities are each one of::

          'length'    : normalized glacier length, L_t / L
          'ela'       : normalized ELA position x_ELA / L, where x_ELA is the
                        point at which the ice surface z crosses the ELA
          'imbalance' : domain-mean erosion–uplift imbalance <E - U> (m/yr),
                        which is 0 at steady state
          'bed'       : domain-mean bedrock elevation normalized by the ELA,
                        <z_b> / z_ELA

        Default axes are x='length', y='imbalance'. The transient spirals in
        toward the closed limit-cycle loop; frames with no terminus or no
        surface/ELA crossing break the line.

        Parameters
        ----------
        side : int or None
            Terminus side; None auto-selects the most active side.
        xaxis, yaxis : {'length', 'ela', 'imbalance', 'bed'}
            State variables for the two axes (see above).
        n_settle : int
            Leading output frames to drop as transient (default 0 = whole run).
        logx : bool
            Log-scale the x axis (default False).
        analytical : bool
            Overlay the analytical steady state as a point, default True (the
            'imbalance' coordinate of the SS is 0 by definition). Skipped when
            the analytical has no glacial SS (glacier_flag not in 2–5).
        colorbar : bool
            Draw the time colorbar (default True).

        Returns
        -------
        (fig, data) : the Figure and a dict with the side, time ``t``, the phase
            coordinates ``x``/``y``, the ``xaxis``/``yaxis`` modes, and the
            analytical ``ss_point`` (x_ss, y_ss).
        """
        from matplotlib.collections import LineCollection
        m = self.model
        if not hasattr(m, "Lt_out") or m.Lt_out.size == 0:
            raise ValueError("No simulation output available; run the model first.")
        for axname in (xaxis, yaxis):
            if axname not in self._PHASE_AXES:
                raise ValueError(f"axis must be one of {sorted(self._PHASE_AXES)}")
        side = self._pick_side(side)
        sl = slice(int(n_settle), None)
        t = m.output_times[sl]
        Lt = m.Lt_out[side][sl].astype(float)
        frames = np.arange(m.output_times.size)[sl]

        # per-frame candidate coordinate series
        x_ela = np.array([self._x_ela_at(int(frames[j]), Lt[j])
                          for j in range(t.size)])
        imbalance = (m.erosion_rate_out[:, frames]
                     - m.U_matrix[:, m.output_steps[frames]]).mean(axis=0)
        bed = m.zb_out[:, frames].mean(axis=0) / m.zELA_out[frames]
        a_ss = (self._analytical_phase_ss()
                if (analytical and hasattr(m, "analytical")) else None)
        series = {
            "length": (Lt / m.L, (a_ss["lt"] / m.L) if a_ss else np.nan),
            "ela": (x_ela / m.L, (a_ss["xela"] / m.L) if a_ss else np.nan),
            "imbalance": (imbalance, 0.0 if a_ss else np.nan),
            "bed": (bed, a_ss["bed"] if a_ss else np.nan),
        }
        x, x_ss = series[xaxis]
        y, y_ss = series[yaxis]

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, facecolor="white")
        else:
            fig = ax.figure

        # time-colored trajectory (segments with a NaN endpoint don't render).
        pts = np.column_stack([x, y]).reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        norm = plt.Normalize(t[0] / 1e3, t[-1] / 1e3)
        lc = LineCollection(segs, cmap=cmap, norm=norm)
        lc.set_array(t[:-1] / 1e3)
        lc.set_linewidth(lw)
        lc.set_alpha(alpha)
        ax.add_collection(lc)

        # balance reference line on whichever axis is the imbalance.
        bal = r"$\langle \dot{E} - U \rangle = 0$"
        if xaxis == "imbalance":
            ax.axvline(0.0, color="dimgray", ls="--", lw=0.8, label=bal)
        if yaxis == "imbalance":
            ax.axhline(0.0, color="dimgray", ls="--", lw=0.8, label=bal)

        if np.isfinite(x_ss) and np.isfinite(y_ss):
            ax.plot(x_ss, y_ss, "*", color="crimson", ms=15, mec="black",
                    mew=0.6, zorder=6, label="analytical SS")

        finite = np.isfinite(x) & np.isfinite(y)
        if finite.any():
            k0 = int(np.flatnonzero(finite)[0])
            ax.plot(x[k0], y[k0], "o", color="black", ms=5, label="start")
            xs = np.append(x[finite], x_ss)  # include SS point in the view
            ys = np.append(y[finite], y_ss)
            xmin, xmax = np.nanmin(xs), np.nanmax(xs)
            ymin, ymax = np.nanmin(ys), np.nanmax(ys)
            xpad = 0.05 * (xmax - xmin) if xmax > xmin else 0.05
            ypad = 0.05 * (ymax - ymin) if ymax > ymin else 0.05
            # clamp at 0 only for non-negative coordinates.
            xlo = xmin - xpad if self._PHASE_AXES[xaxis][1] else max(0.0, xmin - xpad)
            ylo = ymin - ypad if self._PHASE_AXES[yaxis][1] else max(0.0, ymin - ypad)
            ax.set_xlim(xlo, xmax + xpad)
            ax.set_ylim(ylo, ymax + ypad)
        if logx:
            ax.set_xscale("log")
        ax.set_xlabel(self._PHASE_AXES[xaxis][0])
        ax.set_ylabel(self._PHASE_AXES[yaxis][0])
        ax.set_title(f"Limit-cycle phase portrait (side {side})")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(True, alpha=0.2)
        if colorbar:
            fig.colorbar(lc, ax=ax, label="Time (kyr)")
        fig.tight_layout()

        if save:
            fname = f"{run_id}_limit_cycle_phase" if run_id else path
            fig.savefig(output_path(fname, "images") + ".png", dpi=150,
                        bbox_inches="tight")
        return fig, dict(side=side, t=t, x=x, y=y, xaxis=xaxis, yaxis=yaxis,
                         ss_point=(x_ss, y_ss))


# --- Numerical solvers ---

# --- Ice flux solver ---

@numba.njit(cache=True)
def _cumtrapz(y, x):
    """Cumulative trapezoidal integration with initial value of 0."""
    result = np.zeros(len(y))
    for i in range(1, len(y)):
        result[i] = result[i-1] + 0.5 * (y[i-1] + y[i]) * (x[i] - x[i-1])
    return result


@numba.njit(cache=True)
def _reflect_nonneg(a, start, stop, step):
    """In-place Skorokhod reflection at 0 of the running flux integral a[], swept
    in flow order start->stop (exclusive, step +1 or -1). The running total is
    floored at 0 (a[j] -= min(0, min_{k<=j in flow order} a[k])) so an ablation
    deficit is NOT carried past a terminus: a downstream accumulation zone (a
    mode-B proglacial forebulge that rises back above the ELA) re-nucleates from
    zero rather than after repaying the dead trunk's melt debt (audit m2).
    This is the flux-integral analogue of the 2D accumulator's per-hop
    max(field_ice, 0) clip and of the paper's b=0-below-terminus branch. On a
    single monotone-terminating tongue (no sub-terminus re-ascent above the ELA)
    the running minimum stays 0 above the terminus and equals a[j] below it, so
    the result is bit-for-bit the old pointwise max(Qg, 0) clip."""
    run_min = 0.0
    for j in range(start, stop, step):
        v = a[j]
        if v < run_min:
            run_min = v
        a[j] = v - run_min


@numba.njit(cache=True)
def _solve_ice_flux(x, z, zELA, beta, d, sigma, k_h, P, xo_l, xo_r, didx_l, didx_r, nx, B_cap):
    """Hack-catchment ice flux Qg (and balance B, fluvial Qf) along the 1D
    profile. Qg on each flank is the Hack-weighted cumulative integral of the
    mass balance B = min(beta*(z - zELA), B_cap) from the divide outward, then
    reflected at 0 (`_reflect_nonneg`) so the running integral never carries a
    melt deficit past a terminus — matching the 2D per-hop clip and the paper's
    b=0-below-terminus branch (audit m2). The reflection makes Qg >= 0 by
    construction, so no pointwise clip is needed."""
    B = np.minimum(beta * (z - zELA), B_cap)
    dsigma = d * sigma

    # One-sided detection: a side is the only block when the other side is empty.
    only_right = (didx_l == -1)  # left_bc='reflecting' → right block is the only block
    only_left  = (didx_r == nx)  # right_bc='reflecting' → left block is the only block

    x_from_divide = np.zeros_like(x)
    integral = np.zeros_like(x)
    if didx_l >= 0:
        x_from_divide[:didx_l+1] = x[:didx_l+1] - x[didx_l] + xo_l
        cum = _cumtrapz(
            (x_from_divide[:didx_l+1]**(dsigma-1) * B[:didx_l+1])[::-1],
            x_from_divide[:didx_l+1][::-1])[::-1]
        if only_left:
            # One-sided: head integral over [0, xo_l] using linear z extrapolation
            # z(x') = z_first + S·(xo_l - x') for x' ∈ [0, xo_l]. Includes the
            # unresolved upstream catchment, matching the analytical's Beff·A form.
            z_first = z[didx_l]
            if didx_l - 1 >= 0:
                dx_grid = x[didx_l - 1] - x[didx_l]   # >0 since x decreases with index
                S = (z[didx_l] - z[didx_l - 1]) / dx_grid
            else:
                S = 0.0
            if S < 0.0:
                S = 0.0
            xol_p  = xo_l ** dsigma
            xol_p1 = xo_l ** (dsigma + 1.0)
            # Honor the P cap (B_cap) on the extrapolated head: b = beta*(z-zELA)
            # is linear and non-increasing in x' (S>=0), so the cap binds nearest
            # the divide (small x'). Split the integral at x* where b = B_cap.
            # When the cap never binds (b at the divide <= B_cap, incl.
            # B_cap=inf when cap_ice_accumulation=False) this is bit-for-bit the
            # uncapped closed form (audit F1).
            b_head = beta * (z_first + S * xo_l - zELA)   # b at x'=0 (divide), the max
            if b_head <= B_cap:
                I_head = (beta * (z_first + S * xo_l - zELA) * xol_p / dsigma
                          - beta * S * xol_p1 / (dsigma + 1.0))
            elif S <= 0.0:
                I_head = B_cap * xol_p / dsigma
            else:
                x_star = (z_first + S * xo_l - zELA - B_cap / beta) / S
                if x_star >= xo_l:
                    I_head = B_cap * xol_p / dsigma
                else:
                    A_head = z_first + S * xo_l - zELA
                    xs_p  = x_star ** dsigma
                    xs_p1 = x_star ** (dsigma + 1.0)
                    I_head = (B_cap * xs_p / dsigma
                              + beta * (A_head * (xol_p - xs_p) / dsigma
                                        - S * (xol_p1 - xs_p1) / (dsigma + 1.0)))
            integral[:didx_l+1] = cum + I_head
        else:
            # Two-sided: head-catchment integral over [0, xo_l] at the divide,
            # z(x') ~ z_divide (locally flat): int_0^xo x'^(dsigma-1) B dx'
            # = B_divide * xo^dsigma / dsigma — the S=0 case of the one-sided
            # head term, and it makes Qg(divide) = k_h * B * xo^d exactly
            # (the analytical's Beff*A form). The old code added the bare
            # integrand, a factor xo/dsigma too small.
            integral[:didx_l+1] = cum + B[didx_l] * xo_l ** dsigma / dsigma
        # Floor the running integral at 0 in flow order (divide didx_l -> outlet
        # 0): a melt deficit is not carried past a terminus (audit m2).
        _reflect_nonneg(integral, didx_l, -1, -1)
    if didx_r < nx:
        x_from_divide[didx_r:] = x[didx_r] - x[didx_r:] + xo_r
        cum = _cumtrapz(
            x_from_divide[didx_r:]**(dsigma-1) * B[didx_r:],
            x_from_divide[didx_r:])
        if only_right:
            z_first = z[didx_r]
            if didx_r + 1 < nx:
                dx_grid = x[didx_r] - x[didx_r + 1]
                S = (z[didx_r] - z[didx_r + 1]) / dx_grid
            else:
                S = 0.0
            if S < 0.0:
                S = 0.0
            xor_p  = xo_r ** dsigma
            xor_p1 = xo_r ** (dsigma + 1.0)
            # Capped head, mirroring the left flank (audit F1): bit-for-bit the
            # uncapped closed form when the cap never binds (incl. B_cap=inf).
            b_head = beta * (z_first + S * xo_r - zELA)   # b at x'=0 (divide), the max
            if b_head <= B_cap:
                I_head = (beta * (z_first + S * xo_r - zELA) * xor_p / dsigma
                          - beta * S * xor_p1 / (dsigma + 1.0))
            elif S <= 0.0:
                I_head = B_cap * xor_p / dsigma
            else:
                x_star = (z_first + S * xo_r - zELA - B_cap / beta) / S
                if x_star >= xo_r:
                    I_head = B_cap * xor_p / dsigma
                else:
                    A_head = z_first + S * xo_r - zELA
                    xs_p  = x_star ** dsigma
                    xs_p1 = x_star ** (dsigma + 1.0)
                    I_head = (B_cap * xs_p / dsigma
                              + beta * (A_head * (xor_p - xs_p) / dsigma
                                        - S * (xor_p1 - xs_p1) / (dsigma + 1.0)))
            integral[didx_r:] = cum + I_head
        else:
            # Two-sided right flank: same head integral as the left flank.
            integral[didx_r:] = cum + B[didx_r] * xo_r ** dsigma / dsigma
        # Flow order divide didx_r -> outlet nx-1 (audit m2).
        _reflect_nonneg(integral, didx_r, nx, 1)
    # The reflection guarantees integral >= 0 on every filled node, and the
    # prefactor is > 0, so Qg >= 0 — the old pointwise max(Qg, 0) clip is now
    # redundant and has been dropped.
    Qg = (k_h * sigma * d / x_from_divide**(d*(sigma-1))) * integral
    Qf = P * k_h * x_from_divide**d
    return B, Qg, Qf

# --- Ice thickness solver ---


@numba.njit(cache=True)
def _solve_ice_thickness_power_analytical_vec(D_arr, lambda_p):
    H = np.zeros_like(D_arr)
    for i in range(len(D_arr)):
        if D_arr[i] > 0.0:
            H[i] = _solve_ice_thickness_power_analytical(D_arr[i], lambda_p)
    return H


@numba.njit(cache=True)
def _solve_ice_thickness_coulomb_vec(D_arr, a_arr, lambda_c, clamp):
    H = np.zeros_like(D_arr)
    for i in range(len(D_arr)):
        if D_arr[i] > 0.0 and a_arr[i] > 0.0:
            H[i] = _solve_ice_thickness_coulomb(D_arr[i], a_arr[i], lambda_c, clamp)
    return H

# --- Fluvial erosion solver ---
# E = Kf * Qf^m * S^n  (stream-power law)
# Fi = (dt / dx^n) * Kf * Qf^m




# --- Glacial effective-exponent sliding law erosion solver ---
# E = Kg * Qg^mu * S^nu  (eff-exp sliding: a single-power approximation)
# Gi = (dt / dx^nu) * Kg * Qg^mu
# Falls back to fluvial stream power where ice flux is zero.




@numba.njit(cache=True)
def _linear_erode(z, zo, Qf, Qg, Kf, Kg, m, mu, nx, didx_l, didx_r):
    if didx_l >= 0:
        for i in range(1, didx_l + 1):
            Gi = Kg * Qg[i]**mu
            if Gi > 0:
                z[i] = (zo[i] + Gi * z[i-1]) / (1 + Gi)
            else:
                Fi = Kf * Qf[i]**m
                z[i] = (zo[i] + Fi * z[i-1]) / (1 + Fi)
    if didx_r < nx:
        for i in range(nx-2, didx_r - 1, -1):
            Gi = Kg * Qg[i]**mu
            if Gi > 0:
                z[i] = (zo[i] + Gi * z[i+1]) / (1 + Gi)
            else:
                Fi = Kf * Qf[i]**m
                z[i] = (zo[i] + Fi * z[i+1]) / (1 + Fi)

@numba.njit(cache=True)
def _nonlinear_erode(z, zo, Qf, Qg, Kf, Kg, m, mu, nx, n, nu, didx_l, didx_r):
    if didx_l >= 0:
        for i in range(1, didx_l + 1):
            Gi = Kg * Qg[i]**mu
            if Gi > 0:
                z[i] = _solver_glacial(zo[i], z[i-1], Gi, nu)
            else:
                Fi = Kf * Qf[i]**m
                z[i] = _solver_fluvial(zo[i], z[i-1], Fi, n)
    if didx_r < nx:
        for i in range(nx-2, didx_r - 1, -1):
            Gi = Kg * Qg[i]**mu
            if Gi > 0:
                z[i] = _solver_glacial(zo[i], z[i+1], Gi, nu)
            else:
                Fi = Kf * Qf[i]**m
                z[i] = _solver_fluvial(zo[i], z[i+1], Fi, n)

# --- Power sliding law erosion solver ---
# E = G_o * S^t, where t = 3*ell/2
# Same implicit structure as the eff-exp solver, just with different exponent.
# G_o = c_e * (cg * lambda_p^2 * Qg / (alpha_g^2 * (1 + (H/lambda_p)^2)))^(ell/2)
# (kt is absorbed into cg).




@numba.njit(cache=True)
def _power_erode(z, zo, Qf, Qg, H, Kf, Kg_prefactor, m, nx, n, t, lambda_p,
                 didx_l, didx_r):
    """
    Power sliding law erosion.
    Kg_prefactor = (dt / dx^t) * c_e * (kt * (2*Ac/5) * lambda_p^2 * (rho_g*g)^3 / alpha_g)^(ell/2)
    G_o,i = Kg_prefactor * (Qg_i / (1 + (H_i/lambda_p)^2))^(ell/2)
    where ell = 2*t/3
    """
    ell_half = t / 3.0  # ell/2 = t/3
    if didx_l >= 0:
        for i in range(1, didx_l + 1):
            if Qg[i] > 0.0:
                rheology_factor = 1.0 + (H[i] / lambda_p) ** 2
                Gi = Kg_prefactor * (Qg[i] / rheology_factor) ** ell_half
                z[i] = _solver_glacial_power(zo[i], z[i-1], Gi, t)
            else:
                Fi = Kf * Qf[i]**m
                z[i] = _solver_fluvial(zo[i], z[i-1], Fi, n)
    if didx_r < nx:
        for i in range(nx-2, didx_r - 1, -1):
            if Qg[i] > 0.0:
                rheology_factor = 1.0 + (H[i] / lambda_p) ** 2
                Gi = Kg_prefactor * (Qg[i] / rheology_factor) ** ell_half
                z[i] = _solver_glacial_power(zo[i], z[i+1], Gi, t)
            else:
                Fi = Kf * Qf[i]**m
                z[i] = _solver_fluvial(zo[i], z[i+1], Fi, n)

# --- Regularized Coulomb (coulomb) sliding law erosion solver ---
#   E_g = c_e * [cg^(2/5)/alpha_g * R * (H+R)^(-3/5)]^ell
#             * Qg^(3*ell/5) * S^(6*ell/5)
# with R = lambda_c / (1 - (rho_g*g*H*S/tau_c)^3), and H tied to S via mass
# conservation (same equation _solve_ice_thickness_coulomb solves).
# kt is absorbed into cg, so no explicit kt factor enters here.
#
# Unlike the power solver (which freezes H from calculate_ice_thickness and
# uses an analytic partial Jacobian), the Coulomb erosion has dE/dH ~ 1/(1-y)
# near the pole: the partial derivative is not just inaccurate, it's dominated
# by a term that's missing. So we refresh H inside every F evaluation and use
# an analytical *total* derivative dF/dz obtained by implicitly differentiating
# the H-eq. Cost: one H-solve per F-and-dF call (vs two in the earlier FD form).



@numba.njit(cache=True)
def _coulomb_erode(z, zo, Qf, Qg, Kf, A_const, m, nx, n, ell, t,
                      cg, rho_g_g, tau_c, lambda_c, dx, clamp,
                      didx_l, didx_r):
    """
    Regularized Coulomb sliding law erosion.
    A_const = ce * (cg^(2/5)/alpha_g)^ell * dt / dx^t   (node-independent)
    Per-node Qg^(3*ell/5) and the (H, R)-dependent mass factor are applied inside.
    With kt now inside cg, no explicit kt factor enters here.
    """
    exp_Q = 3.0 * ell / 5.0
    if didx_l >= 0:
        for i in range(1, didx_l + 1):
            if Qg[i] > 0.0:
                A_pre = A_const * Qg[i] ** exp_Q
                z[i] = _solver_glacial_coulomb(
                    zo[i], z[i-1], Qg[i], A_pre, ell, t,
                    cg, rho_g_g, tau_c, lambda_c, dx, clamp)
            else:
                Fi = Kf * Qf[i] ** m
                z[i] = _solver_fluvial(zo[i], z[i-1], Fi, n)
    if didx_r < nx:
        for i in range(nx - 2, didx_r - 1, -1):
            if Qg[i] > 0.0:
                A_pre = A_const * Qg[i] ** exp_Q
                z[i] = _solver_glacial_coulomb(
                    zo[i], z[i+1], Qg[i], A_pre, ell, t,
                    cg, rho_g_g, tau_c, lambda_c, dx, clamp)
            else:
                Fi = Kf * Qf[i] ** m
                z[i] = _solver_fluvial(zo[i], z[i+1], Fi, n)


# --- diagnostic: bedrock-state tracking H solve --------------------------------
# Solves each law's H equation self-consistently with the ice surface slope
# built from a tracked bedrock z_b plus the just-solved upstream H values.
# Per node, given the already-solved receiver z_s_r = z_b_r + hc*H_r and
# a = z_b_i - z_s_r (can be negative when z_b sits in an overdeepening),
# the slope S_i = (hc*H_i + a) / dx is substituted into each law's H equation
# (hc = HC_OVER_H, the centerline-to-mean depth ratio: the tracked bed is the
# channel floor and the column standing on it is hc*H, while the H in the
# flux closures stays the width-mean depth).
# Constraint H>0, hc*H+a>0. Single-variable Newton with bisection back into
# the valid region on bad steps.
#
# At hc = 1 the residuals are:
# eff-exp:  H * (H + a)^(2/3) = C            with C = (Q_g/(cg λ^1.5))^(2/9) dx^(2/3)
# power:    H^4 (H^2 + λ^2) (H + a)^3 = K    with K = Q_g · dx^3 / cg
# coulomb:  H^5 (H + a)^3 (H + λ_c/(1-φ^3)) = K  with φ = β·H·(H+a), β = ρg/(τ_c·dx),
#           pole at φ=1 (τ = τ_c). Newton + line-search keeps φ < 1-clamp.
#
# General hc is handled WITHOUT touching the solver bodies, via the column
# substitution G = hc*H (solve for the column height, then H = G/hc):
# eff-exp:  G (G + a)^(2/3) = hc*C
# power:    G^4 (G^2 + (hc λ)^2) (G + a)^3 = hc^6 K
# coulomb:  G^5 (G + a)^3 (G + hc λ_c/(1-φ^3)) = hc^6 K, φ = (β/hc)·G·(G+a)
#           (φ is τ/τ_c with τ = ρg·H·S unchanged — H mean, S the hc-surface
#           slope; the pole bound H_safe transforms consistently in G).
# The fixed-slope outlet closures H(Q, S_up) take the slope as data and need
# no substitution — only the slope itself is built from hc-surfaces.





# Mode-B 1D walk: see _glac_fast_solve / _diag_walk in siim._core.skeleton.
# The per-law wrappers were dropped in the API refactor — the 1D model maps
# sliding_law -> law_code and calls _diag_walk(law_code, p, ...) directly
# (see _glacial_params_and_code + calculate_ice_thickness).
