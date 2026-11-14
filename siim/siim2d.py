import copy
import numbers
import numpy as np
import os
import pickle
import sys
import tempfile
import warnings
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
# gate: persistence records the producing version from the package's single source.
from . import __version__ as _SIIM_VERSION
from . import constants as _constants
from .constants import check_range as _check_range
from .constants import (GRAVITY, KT, RHO_ICE, derive_coulomb,
                        derive_power)
from .analytical import analytical_steady_state_solution
from ._output import output_path
from .plotting import siim_plotter
from ._core.outputs import _donor_lists, _sanitized_receivers
from ._core import driver as _driver
from ._core import outputs as _outputs
from ._core.flexure import flexure as _inhouse_flexure
from ._core.router import (DONOR_FIRST_ROUTERS as _DONOR_FIRST_ROUTERS,
                           SLOPE_NORMALISED_ROUTERS as _SLOPE_NORMALISED_ROUTERS,
                           check_mfd_config as _check_mfd_config,
                           check_inner_loop_config as _check_inner_loop_config,
                           check_routing_filter_config
                           as _check_routing_filter_config,
                           resolve_trunk_dip_k as _resolve_trunk_dip_k)
from ._core.regime import check_regime_grid as _check_regime_grid
from ._core import threads as _threads
from ._core.solvers import ice_speeds
from ._core.step import (build_glacial_params, initial_topography,
                         uplift_mask, block_uplift)

try:  # numpy >= 1.25
    from numpy.exceptions import RankWarning as _RankWarning
except ImportError:  # numpy < 1.25 (np.RankWarning removed in numpy 2)
    _RankWarning = np.RankWarning  # noqa: NPY201 reached only on numpy < 1.25; the try branch serves numpy 2


_SAVE_FORMAT = "siim.model-state.pickle"
_SAVE_SCHEMA_VERSION = 1
_SAVE_REQUIRED_KEYS = frozenset({
    "save_format", "schema_version", "siim_version", "model_identity",
    "_user_params", "ds_out",
})
# Parameters this version removed, each with what to use instead: passing one
# raises (set_and_check_parameters); a saved model carrying one loads without
# it, with a warning (load).
_REMOVED_PARAMS = {
    'routing_relax': (
        "modes B, C and D damp the ice itself (inner_damping, default "
        f"{_constants.MODE_BCD_INNER_DAMPING!r}); drop the key, or pass "
        "inner_relax=1.0, inner_passes=1 for the undamped model"),
}


_BOUNDARY_KINDS = ('fixed_value', 'core', 'looped')


def _check_boundary_status(boundary_status):
    """The 4-list [left, right, bottom, top] of a scalar or 4-sequence
    boundary_status, validated at the user boundary (audit #1, #15, #16, #32):
    each side one of _BOUNDARY_KINDS (a typo used to close the edge silently),
    'looped' on both sides of an axis or neither (the wrap flags read one side
    only), and at least one 'fixed_value' side, the outlet (without one the
    priority flood has no seed and nothing drains)."""
    try:
        bs = [str(s) for s in np.broadcast_to(np.asarray(boundary_status, dtype=object), 4)]
    except ValueError:
        raise ValueError("boundary_status must be one value or four "
                         f"[left, right, bottom, top], got {boundary_status!r}") from None
    bad = sorted(set(bs) - set(_BOUNDARY_KINDS))
    if bad:
        raise ValueError(f"boundary_status sides must be one of {_BOUNDARY_KINDS}, "
                         f"got {bad} in {bs}")
    for (i, j), axis in (((0, 1), 'left/right'), ((2, 3), 'bottom/top')):
        if (bs[i] == 'looped') != (bs[j] == 'looped'):
            raise ValueError(f"boundary_status: 'looped' pairs the two sides of an "
                             f"axis, got {bs[i]!r}/{bs[j]!r} for {axis}")
    if 'fixed_value' not in bs:
        raise ValueError("boundary_status needs at least one 'fixed_value' side "
                         f"(the outlet), got {bs}")
    return bs


def _model_identity(model_cls):
    return f"{model_cls.__module__}.{model_cls.__qualname__}"


def _validate_saved_state(state, model_cls):
    """Validate the small versioned envelope around a pickled model state."""
    if not isinstance(state, dict):
        raise ValueError(
            "Invalid SIIM saved model: the top-level pickle payload must be a dict.")

    # Before schema version 1, save() wrote only these two keys. Refuse that
    # ambiguous payload explicitly instead of rebuilding it under today's
    # parameter defaults and output conventions.
    if ("schema_version" not in state
            and {"_user_params", "ds_out"}.issubset(state)):
        raise ValueError(
            "Unsupported legacy SIIM saved model: this file predates the "
            "versioned save format. Re-run and save it with the SIIM version "
            "that created it before loading it here.")

    missing = sorted(_SAVE_REQUIRED_KEYS - state.keys())
    if missing:
        raise ValueError(
            f"Invalid SIIM saved model: missing required key(s): {missing}.")

    if not isinstance(state["save_format"], str):
        raise ValueError(
            "Invalid SIIM saved model: 'save_format' must be a string.")
    if state["save_format"] != _SAVE_FORMAT:
        raise ValueError(
            f"Unsupported SIIM save format {state['save_format']!r}; "
            f"expected {_SAVE_FORMAT!r}.")

    schema_version = state["schema_version"]
    if type(schema_version) is not int:
        raise ValueError(
            "Invalid SIIM saved model: 'schema_version' must be an integer.")
    if schema_version != _SAVE_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported SIIM save schema version {schema_version}; this "
            f"SIIM release supports version {_SAVE_SCHEMA_VERSION}.")

    if not isinstance(state["siim_version"], str) or not state["siim_version"]:
        raise ValueError(
            "Invalid SIIM saved model: 'siim_version' must be a non-empty string.")

    if not isinstance(state["model_identity"], str):
        raise ValueError(
            "Invalid SIIM saved model: 'model_identity' must be a string.")
    expected_model = _model_identity(model_cls)
    if state["model_identity"] != expected_model:
        raise ValueError(
            f"Saved model is for {state['model_identity']!r}, not "
            f"{expected_model!r}; load it with the matching model class.")

    if not isinstance(state["_user_params"], dict):
        raise ValueError(
            "Invalid SIIM saved model: '_user_params' must be a dict.")

    import xarray as xr
    if not isinstance(state["ds_out"], xr.Dataset):
        raise ValueError(
            "Invalid SIIM saved model: 'ds_out' must be an xarray.Dataset.")

def _load_initial_topography(src):
    """Load initial topography from an ndarray, pandas DataFrame, or CSV file.

    Long-format DataFrame / CSV columns: 'x', 'y', 'topography__elevation'
    (optional 'time' — the latest time slice is used). x and y must be on a
    regular grid; coordinates are interpreted as extent and shifted to a
    (0, 0) origin.

    Returns
    -------
    arr : (ny, nx) ndarray
    grid : dict or None
        {'nx', 'ny', 'Lx', 'Ly'} when loaded from a file/DataFrame; None when
        the input was already an ndarray (caller keeps user nx/ny/Lx/Ly).
    """
    if isinstance(src, np.ndarray):
        return np.asarray(src, dtype=float), None

    import pandas as pd

    if isinstance(src, (str, Path)):
        df = pd.read_csv(src)
    elif isinstance(src, pd.DataFrame):
        df = src
    else:
        raise TypeError(
            f"initial_topography must be an ndarray, pandas DataFrame, or "
            f"CSV filename; got {type(src).__name__}")

    needed = {'x', 'y', 'topography__elevation'}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(
            f"initial_topography source missing column(s): {sorted(missing)}")

    if 'time' in df.columns:
        df = df[df['time'] == df['time'].max()]

    x_uniq = np.sort(df['x'].unique())
    y_uniq = np.sort(df['y'].unique())
    nx, ny = len(x_uniq), len(y_uniq)

    def _uniform_step(arr, name):
        diffs = np.diff(arr)
        if not np.allclose(diffs, diffs[0]):
            raise ValueError(
                f"initial_topography {name} coordinates are not uniformly spaced")
        return float(diffs[0])

    dx = _uniform_step(x_uniq, 'x')
    dy = _uniform_step(y_uniq, 'y')

    pivot = df.pivot_table(index='y', columns='x', values='topography__elevation')
    pivot = pivot.sort_index().sort_index(axis=1)
    arr = pivot.to_numpy()
    if arr.shape != (ny, nx):
        raise ValueError(
            f"initial_topography is not on a regular (y, x) grid: "
            f"pivot shape {arr.shape}, expected ({ny}, {nx})")
    if not np.all(np.isfinite(arr)):
        raise ValueError("initial_topography contains NaN — grid has gaps")

    return arr, dict(nx=nx, ny=ny, Lx=(nx - 1) * dx, Ly=(ny - 1) * dy)


def _ibc_from_border_status(border_status):
    """Encode ``border_status`` for the in-house flexure/diffusion kernels.

    The retained boundary code is ``left*1 + right*100 + bottom*1000 +
    top*10`` in siim's side names (``bottom`` is row 0), with a digit 1 per
    ``'fixed_value'`` edge. It matches the historical Fastscape convention
    used by these numerical kernels.
    """
    bs = list(np.broadcast_to(border_status, 4))   # [left, right, bottom, top]
    arr_bc = np.array([1 if s == 'fixed_value' else 0 for s in bs])
    return int(sum(arr_bc * np.array([1, 100, 1000, 10])))


def load(filename):
    """Module-level shortcut for :meth:`siim.load`.

    ``filename`` must be a trusted SIIM pickle; pickle can execute code before
    the saved-state envelope is validated.
    """
    return siim.load(filename)


# =============================================================================
# siim2d wrapper class — mirrors the siim1d.siim interface
# =============================================================================

class siim:
    """Standalone two-dimensional coupled glacial-fluvial landscape model.

    Parameters
    ----------
    user_params : dict, optional
        Parameter overrides. Omitted keys use the documented model defaults;
        unknown keys raise ``ValueError``. See the public parameter reference
        for units, accepted values, and mode-specific behavior.

    Notes
    -----
    The default driver is the in-house NumPy/Numba implementation and does not
    require Fastscape or xsimlab. Call :meth:`run` to integrate, use the
    ``*_out`` arrays or ``ds_out`` for scientific output, and use ``plot``
    for visualization. :meth:`save` and :meth:`load` use versioned pickle files;
    load only files from trusted sources.
    """

    def __init__(self, user_params=None):
        if user_params is None:
            user_params = {}
        elif not isinstance(user_params, dict):
            raise TypeError("params must be provided as a dictionary.")

        # Stash a copy so _set_analytical_grid can rebuild the analytical with
        # channel-fitted geometry overrides (called by extract_channel). A deep
        # copy, and the run reads it, so a caller mutating an input array later
        # cannot change the run (audit #35).
        user_params = self._user_params = _constants.unwrap_0d(copy.deepcopy(user_params))
        self.set_and_check_parameters(user_params)

        # Standalone analytical (siim.analytical.analytical_steady_state_solution).
        # Constructed from a curated subset of user_params; see _analytical_user_params.
        # None on a non-eroding bed (ce or Ko = 0; see _build_analytical).
        self.analytical = self._build_analytical(
            self._analytical_user_params(user_params)
        )
        if self.analytical is None:
            # No reference grid; the Hack geometry keeps the analytical's defaults
            # (siim2d takes no xo/k_h/d of its own; extract_channel re-fits them).
            self.x, self.L = None, self.Lx
            self.xo, self.k_h = _constants.XO, _constants.KH
            self.d, self.sigma = _constants.D_HACK, _constants.SIGMA
        else:
            # Plotter aliases — preserve the pre-strip API where the plotter reads
            # m.x / m.L for the 1D analytical reference grid.
            self.x = self.analytical.x
            self.L = self.analytical.L
            # Hack geometry mirrors the analytical reference (extract_channel re-fits
            # these per basin); siim2d has no k_h/d/sigma/xo params of its own.
            self.xo = self.analytical.xo
            self.k_h = self.analytical.k_h
            self.d = self.analytical.d
            self.sigma = self.analytical.sigma

        self.plot = siim_plotter(self)

    def _build_analytical(self, params):
        """The 1D analytical steady-state reference, or None when ``ce`` or
        ``Ko`` is 0: that steady state balances uplift against glacial and
        fluvial erosion and divides by both, so a non-eroding (frozen) bed has
        none to build. The run itself accepts 0 (no erosion)."""
        if self.ce == 0 or self.Ko == 0:
            return None
        return analytical_steady_state_solution(params)

    def _analytical_user_params(self, user_params):
        """Filter user_params down to keys the standalone analytical accepts and
        substitute siim2d's already-processed scalar values for U / zELA / P
        (each may be a per-step series in the run, but the analytical reference
        is scalar — see the time-MEAN/MIN reductions in set_and_check_parameters).
        Base level ``bl`` is scalar-reduced (``self.bl``) but deliberately NOT
        forwarded: the analytical is graded to the datum (z(outlet)=0), so a
        nonzero bl only shifts the profile by a constant (the overlay warns when
        drawn) — per the array-forcing memory rule + the design study's §7 Q5. The
        1D-reference grid is one-sided (reflecting at the head, base_level at
        the outlet) and uses L = Lx; per-basin geometry is overridden via
        _set_analytical_grid after extract_channel fits Hack's law. BCs and nx
        are forced — the user has no override path for them in 2D because the
        analytical is always a basin-extracted profile."""
        # NB: no 'sigma'/'lam' — siim2d rejects them as unexpected keys before
        # this filter runs, and _set_analytical_grid injects only xo/k_h/d, so
        # they could never match (audit m39).
        keys = {"xo", "P", "beta", "k_h", "d", "n", "m", "Ko", "Ac",
                "alpha_g", "lambda_p", "lambda_c",
                "mu", "nu", "ell", "k",
                "sliding_law", "ce", "tau_c"}
        out = {k: v for k, v in user_params.items() if k in keys}
        if out.get("nu") is not None and out.get("ell") is not None:
            # 'ell' takes precedence; drop 'nu' so the analytical does not
            # re-issue the duplicate-exponent warning already raised here.
            del out["nu"]
        out["U"] = self.U
        out["zELA"] = self.zELA
        out["P"] = self.P
        out["hc_over_H"] = self.hc_over_H       # the reference's bed on the run's datum (ED-183)
        out["L"] = self.Lx
        out["nx"] = _constants.ANALYTICAL_NX_2D
        out["left_bc"] = "reflecting"
        out["right_bc"] = "base_level"
        return out

    def _default_params(self):
        """Parameter-defaults dict. Factored out of set_and_check_parameters so
        subclasses can extend the accepted parameter set by overriding this and
        merging in their own keys; the same dict drives the
        unexpected-parameter check."""
        return {
            # 2D domain
            "Lx": 5e4,
            "Ly": 5e4,
            "nx": 201,
            "ny": 201,
            # time
            "to": 0,
            "T": 3e6,
            "nt": 2501,
            "nt_out": 101,
            # climate
            "P": _constants.P,
            "beta": _constants.BETA,
            "zELA": None,   # None: constants.ZELA, unless zT is given
            "zT": None,
            "U": _constants.U,
            # Base level: the water-line (Dirichlet) datum for mode B. Scalar
            # (default constants.BL = 0) or a length-nt series on the 'tstep'
            # clock like zELA/P; the per-step value floors the erosion working
            # view, the border-bed budget, the flotation reference and the
            # recovery threshold. Also accepts a PER-SIDE dict
            # {'left'|'right'|'bottom'|'top': scalar or series} — one datum per
            # fixed_value outlet, each interior node taking its basin outlet's
            # (in-house driver only). The analytical reference stays at the datum
            # (bl scalar-reduced, not forwarded — nonzero mean warns).
            "bl": _constants.BL,
            # Global waterline-flotation gate (mode B; default
            # constants.FLOTATION_GATE = True): grounded <=> zs = zb + hc*H >= bl
            # (rho_i = rho_w). Interior: a sub-waterline icy cell does
            # reduced/no glacial erosion (and so carves nothing). Border: the
            # ramp is the physical bound inside the closed-form IMPLICIT
            # border budget (the bed digs on the arrival slope to the
            # flotation-draft equilibrium). False un-bounds the border —
            # diagnostics only.
            "flotation_gate": _constants.FLOTATION_GATE,
            # Flotation-ramp width gamma (default constants.FLOTATION_RAMP =
            # 0.1): the gate is the effective-pressure ramp — glacial erosion
            # is multiplied by f = clip((zs - bl)/(gamma*hc*H), 0, 1),
            # f = 0 exactly for zs <= bl. 0 = the hard binary gate (interior
            # bit-for-bit; at the border its implicit solution is the
            # flotation sliding mode). Safe ceiling 0.2 (wide ramps can dome
            # cap=False configs). Only active when flotation_gate is on.
            "flotation_ramp": _constants.FLOTATION_RAMP,
            # Mode-B parallel-eroder toggle:
            # run the erosion step level-scheduled across threads (topological
            # levels of the flow graph). BIT-FOR-BIT identical to the serial
            # eroder at any thread count (pinned by test_parallel_erode);
            # default ON (under flow_routing='mfd': on under the TBB
            # threading layer only, constants.MFD_PARALLEL_ERODE_LAYERS).
            # False = the serial eroder. Its per-level loops run at the performance-core thread
            # count (_core.threads.level_threads).
            "parallel_erode": _constants.PARALLEL_ERODE,
            # fluvial
            "n": _constants.N_FLUVIAL,
            "m": None,
            "Ko": _constants.KO,
            # glacial
            "Ac": _constants.AC,
            "alpha_g": _constants.ALPHA_G,
            # Channel-floor datum: zs = zb + hc_over_H*H. None = constants.HC_OVER_H
            # (the parabola's max/mean depth ratio); 1 = the sheet datum.
            "hc_over_H": None,
            "lambda_p": _constants.LAMBDA_P,  # eff-exp/power critical ice thickness [m]
            # Default per-law-derived from nu. For exact power/Coulomb numerics
            # an explicit override is analytical-only and warns.
            "mu": None,
            "nu": _constants.NU,   # SS slope exponent (primary user input)
            "ell": None,           # erosion-law exponent in Eg = ce (kt ub)^ell; default per-law-derived from nu
            "k": _constants.K_ACCUM,  # accumulation-profile shape exponent (paper's p)
            "sliding_law": _constants.DEFAULT_SLIDING_LAW,    # "eff-exp", "power", or "coulomb"
            "ce": _constants.CE,    # glacial erosion coefficient (power / coulomb law)
            "lambda_c": None,       # coulomb sliding length; default constants.LAMBDA_C
            # Sub-grid glacier-width amplification of below-ELA ablation
            # (Hack-style: width = width_hack_k * A^width_hack_p, A = upstream area).
            # Always on; caps at cell_area for headwater cells. Defaults
            # constants.WIDTH_HACK_K / WIDTH_HACK_P.
            "width_hack_k": _constants.WIDTH_HACK_K,
            "width_hack_p": _constants.WIDTH_HACK_P,

            # Flow routing: 'single' (D8 single-flow, default), 'infinite'
            # (Tarboton 1997 D-infinity) or 'multiple' (multiple-flow-direction);
            # the router codes 'dinf' / 'mfd' are accepted as aliases
            # (constants.normalize_flow_routing).
            # The first two produce stable SS with the area-based
            # width-amplified mass balance.
            # - 'single' (D8): each cell drains to one steepest neighbor. Cheapest,
            #   fastest, recommended default. Use with coulomb or power for paper
            #   figures.
            # - 'infinite' / 'dinf' (Tarboton): flux split between the two neighbors that bracket
            #   the steepest-facet direction, with continuous angle-based weights.
            #   Useful for finer routing geometry (confluence resolution, drainage
            #   pattern visualizations). All three sliding laws (eff-exp, power,
            #   coulomb) work under either routing. Since 2026-09-05 the gamma
            #   effective-slope normalisation is ON by default here (see
            #   `slope_normalise`), so the closure/eroder slope is |grad zs|
            #   rather than its projection on the two-receiver direction;
            #   `slope_normalise=False` restores the older behaviour.
            # - 'multiple' / 'mfd': flux split over ALL strictly
            #   downhill 8-neighbours with weights (S_k/S_max)^mfd_exponent, and
            #   the gamma effective-slope normalisation turned on so the
            #   thickness closure and the eroders still see |grad zs|. Built for
            #   the ice-sheet regime, where D8/D-inf make the flow graph a step
            #   function of the surface; runs in valleys too (mode C, the carve
            #   and the trunk surface; dip per resolve_trunk_dip_k). REQUIRES
            #   mode 'B' or 'C', a transect closure_flux and dx == dy (each
            #   raises otherwise). An unset `parallel_erode` then follows the
            #   threading layer, on under TBB alone (MFD raises the
            #   topological level count 151 -> 400 at 201^2, so the
            #   level-scheduled eroder's fork/join cost roughly triples on
            #   workqueue; measured in mode D).
            "flow_routing": "single",

            # MFD spreading exponent p in w_k = (S_k/S_max)^p (flow_routing='mfd'
            # only; positive, finite up to constants.MFD_EXPONENT_MAX = 50, or
            # inf). Default constants.MFD_EXPONENT = 1.0 —
            # Budd & Warner's split and the only exponent whose weighted receiver
            # vector is exactly parallel to the surface gradient on a square
            # 8-stencil. p = 2 concentrates; p = inf is the steepest link alone,
            # and a finite p above 50 is routed as inf with a warning. Inert
            # under 'single' / 'dinf'.
            "mfd_exponent": _constants.MFD_EXPONENT,

            # gamma effective-slope normalisation. None (default) resolves per
            # router from _core.router.SLOPE_NORMALISED_ROUTERS: ON for every
            # MULTI-receiver router ('dinf', 'mfd'), OFF for 'single'. An
            # explicit True/False wins, so the pre-2026-09-05 D-inf behaviour is
            # reachable as slope_normalise=False. Under 'single' the flag is a
            # NO-OP (one receiver => gamma == 1 identically); under 'mfd' an
            # explicit False RAISES (MFD without it is 17-24 % low by
            # construction). See constants.MFD_GAMMA_FLOOR and
            # docs/dev/dinf_slope_normalisation.md.
            "slope_normalise": None,

            # Single-flow routing backend (see constants.ROUTER_DEFAULT).
            # 'inhouse_d8' = siim's numba D8 fill-then-route
            # (_core.step.route_d8) — the only accepted value since the 0.9.1
            # standalone flip; the param is the router-contract plug point
            # for future backends. Only affects
            # flow_routing='single' (the D-inf directions + mask + basin are
            # in-house on the same contract).
            "router_backend": _constants.ROUTER_DEFAULT,

            # Numerics backend for flexure + hillslope diffusion (see
            # constants.NUMERICS_BACKEND). 'inhouse' = siim's own scipy.fft
            # plate solve + numba ADI diffuser — the only accepted value since
            # the 0.9.1 standalone flip. Does not affect routing (a separate
            # axis).
            "numerics_backend": _constants.NUMERICS_BACKEND,

            # Surface-evolution mode (works under both routings).
            #   'A' — track the ice surface as state; H is solved from the local
            #     slope of the current surface each step. The historical siim2d
            #     behaviour.
            #   'B' (default) — track the bedrock as state. H is solved jointly
            #     with z_s = zb + hc*H (hc = constants.HC_OVER_H) via an upstream
            #     walk (single receiver under SFR; weighted effective receiver
            #     under D-inf); the surface view is lake-filled (SFR stack walk
            #     / D-inf priority flood) so closed depressions persist in zb
            #     (bed memory) — the model's native regime, where the carved
            #     overdeepenings and MISI-style autogenic cycling live.
            #     User-facing names: 'ice_surface' (A) and
            #     'bedrock+ice_thickness' (B, default); 'A'/'B' accepted as
            #     aliases (constants.normalize_mode maps to the internal code).
            #     One-breath taxonomy: A = ice-surface state; B = bedrock+ice
            #     state (bed memory, NO carving); C = B + sub-grid width carving
            #     (2D only; equivalent to mode='B', carve_width=True); D = C with
            #     the channel/sheet reading choosing valley or sheet rules per cell
            #     (2D, standalone driver; constants.MODE_D_*). The 2D
            #     DEFAULT is mode C (the flagship carved config); an unqualified
            #     mode='B' does NOT carve. Single scalar; applies to the run.
            "mode": _constants.DEFAULT_MODE_2D,
            # Optional explicit FD diffusion of H (m^2/yr), applied after the
            # H solve. Smooths the singular H ∝ S^(-2/3) toe transition. MODE B
            # ONLY in 2D — the 2D mode-A skeletons have no diffusion pass, so a
            # value is silently inert there (unlike 1D, which diffuses in both
            # modes; audit m17). No-op when None or 0.
            "H_diffusivity": None,
            "border_bed_uplift": None,  # mode B: uplift rate (m/yr) applied to the bed
                                        # at base-level borders (net U - E under ice;
                                        # recovery toward the datum when ice-free). None
                                        # (default) = the tectonic U, local and per-step
                                        # (scalar, (ny,nx), or (nt,ny,nx) like U); set 0
                                        # to freeze.
            # Sub-grid glacier-width carving (mode B only; OPT-IN). None (the
            # default) resolves per mode: ON for mode C (the 2D default), OFF for
            # an unqualified mode='B'. Set True explicitly to carve in a mode you
            # spelled 'B'. Carve the union-of-
            # discs footprint (width alpha_g*H around glaciated cells; power-
            # diagram attribution) into the bed. Footprint cells — bare AND
            # icy, including terrain above the ice (ridge-eating = channel
            # capture) — descend toward the parabola hung from the source's
            # ice surface (rim at zs = zb + hc*H, floor at the source bed;
            # hc = constants.HC_OVER_H, the parabola max/mean depth ratio),
            # at E_widening = (1 + widening_rate)*E_c — widening_rate = eta >= 0
            # is the excess over the centerline incision E_c as a fraction of
            # E_c (0 = no net widening; 1 = widen at the incision rate; None/
            # np.inf/'inf'/'infinity' = instant), measured from the pre-step
            # bed (never additive with the cell's own erosion).
            # 'looped' boundaries supported: the footprint transform
            # wrap-pads along looped axes (nearest periodic image; exact).
            "carve_width": None,
            "widening_rate": _constants.DEFAULT_WIDENING_RATE,
            # Fabricated trunk-surface routing (mode B/C). Route on
            # a fabricated ice surface with a linear cross-valley dip toward the
            # centerline, so flow CONVERGES onto the trunk chain and its raw flux
            # is the full cross-section. The V builds the router graph ONLY
            # (FIREWALL: see siim._core.driver). See
            # ``docs/dev/trunk_surface_routing.md``.
            # Knob: trunk_dip_k — None (default) is router-derived, see
            # _core.router.resolve_trunk_dip_k.
            # Default None = the mode-C STANDARD (resolved after mode
            # normalization): ON for mode C (mode B + carve), OFF for plain
            # mode B and mode A. An explicit True/False always wins.
            "trunk_surface": None,
            "trunk_dip_k": None,
            # EXPERIMENTAL (branch closure-transect-flux): which ice flux the
            # thickness closure + erosion laws see. 'cell' = the cell's own
            # routed flux; 'transect' = the routed flux crossing
            # a line one closure-width (alpha_g*H) long through the cell,
            # perpendicular to its flow (a trunk sees its own Q, a cell inside a
            # broad sheet sees the flux of the whole width the closure assumes);
            # 'transect_min' = same, but a neighbour contributes at most the
            # cell's own flux. Default None resolves per mode: constants.
            # CLOSURE_FLUX ('transect_min') for mode B/C, 'cell' for mode A
            # (which has no transect path); an explicit value always wins, and
            # an explicit 'cell' is the pre-branch closure bit-for-bit. See
            # constants.CLOSURE_FLUX and docs/dev/closure_transect_results.md.
            "closure_flux": None,
            # Within-step damped iteration (mode B/C sheet knobs). inner_passes
            # = n >= 1 evaluations of the diagnostic ice map per landscape step,
            # the first n-1 at dt = 0 (rehearsals); after EVERY pass the
            # thickness iterate moves x <- x + omega*(F(x) - x) with omega =
            # inner_relax in (0, 1]. omega < 1 retains 1 - omega of the previous
            # iterate (a numerical ice memory of ~1/(omega*n) steps). Modes
            # B/C/D resolve their own defaults (set_and_check_parameters). See
            # constants.INNER_PASSES.
            "inner_passes": _constants.INNER_PASSES,
            "inner_relax": _constants.INNER_RELAX,
            # How the passes commit: 'omega' (the uniform omega above) or an
            # adaptive preset damping only the flickering cells, down to the
            # preset's floor (constants.INNER_DAMPING_FLOORS). None = the
            # mode's default: constants.MODE_BCD_INNER_DAMPING in modes B/C/D,
            # except 'omega' when inner_relax is set; 'omega' in mode A.
            "inner_damping": None,
            # Routing-surface spatial filter (mode B/C sheet knob). f > 0 builds
            # the ROUTER GRAPH alone from a smoothed routing surface
            # (_core.step.filter_routing_surface), so a filtered run is a
            # different model, not a filtered view of one. f = 2 is the measured
            # setting; 0 = off, bit-for-bit today. See
            # constants.ROUTING_FILTER_FACTOR.
            "routing_filter_factor": _constants.ROUTING_FILTER_FACTOR,
            # Mode D's bed bar kB (constants.REGIME_BED_BAR): a cell reads
            # channel only where its bed lies below the bed around it by at
            # least (kB - 1) times the surrounding ice thickness. None = the
            # default (mode D only); 0 = no bar.
            "channel_bed_bar": None,

            # hillslope diffusivity
            "D": _constants.HILLSLOPE_D,
            # boundary conditions: [left, right, bottom, top]
            # each is 'fixed_value', 'core', or 'looped'.
            # Default = left/right fixed base level, top/bottom periodic. OQ-1(b)
            # (2026-07-13, ratified): the shipped default is the EXPLICIT looped
            # list — SFR and D-inf now agree on it, and 'core' means plain
            # non-periodic interior for every router (was: 'core' made the fortran
            # SFR silently Y-cyclic while siim's D-inf did not — a split brain).
            # Only two edge populations shift: explicit-'core' SFR users (now
            # non-periodic) and D-inf-on-default users (now periodic in y, to
            # match SFR). CHANGELOG-documented at S5.
            "boundary_status": ['fixed_value', 'fixed_value', 'looped', 'looped'],
            # initial topography: if `initial_topography` is None, build a
            # tent that ramps from 0 at fixed_value edges up to
            # initial_max_elevation at the farthest interior point.
            "initial_max_elevation": _constants.INITIAL_MAX_ELEVATION,
            "initial_topography": None,
            "noise_amplitude": _constants.NOISE_AMPLITUDE,
            "seed": None,
            # run options
            "progress_bar": True,
            # coulomb-law params (k_h/d/sigma/xo geometry lives only in the
            # analytical reference, which extract_channel fits from the 2D run)
            "tau_c": _constants.TAU_C,
            "coulomb_clamp": _constants.COULOMB_CLAMP,
            # Flexural isostasy (off by default). When flexure=True, the
            # GlacialFlexure process is added: fastscape's elastic-plate Flexure
            # re-sourced to load from the glacial erosion process's denudation (the
            # TRUE rock removed — delta-zb incl. sub-grid carve in mode B, delta-zs
            # in mode A) AND
            # (when ice_load=True, the default) the per-step glacial ice load — the
            # channel cross-section alpha_g*H^2/L (= Qg/V, hc-free; a slab H on
            # mode D's sheet cells) —
            # i.e. true glacial isostatic adjustment: ice loading +
            # erosional/tectonic unloading + elastic rebound. Hillslope diffusion
            # of the ice surface is deliberately left out of the load (an accepted
            # small approximation — a near-conservative redistribution of the ice
            # surface, not rock unloading).
            "flexure": False,
            # Include the ice load in flexure (true GIA). False = erosional/tectonic
            # unloading only (the pre-GIA behaviour; isolate the ice contribution or
            # reproduce runs saved before the ice load). Only consulted when flexure=True.
            "ice_load": _constants.ICE_LOAD,
            # Route per-step eroded volume down the flow graph and report it.
            # False (off — skips the accumulation pass), True / 'basin': the
            # per-node throughput (sediment_flux_out) + its running total
            # (eroded_volume_out); 'edge': the volume delivered across each
            # 'fixed_value' domain edge (sediment_edge_flux_out + its running
            # total), a (time, side) pair instead of rasters; 'both'.
            "track_sediment": False,
            # Store the closure's own (flux, thickness) rows, closure_Qg_out /
            # closure_H_out, where they differ from Qg_out / H_out: ice_speed
            # is exact with them, approximate (and warns) without.
            "save_closure_pair": _constants.SAVE_CLOSURE_PAIR,
            "lithos_density": _constants.LITHOS_DENSITY,   # lithospheric rock density (kg/m^3); scalar or (ny,nx)
            "asthen_density": _constants.ASTHEN_DENSITY,   # asthenospheric density (kg/m^3)
            "e_thickness": _constants.E_THICKNESS,      # effective elastic plate thickness Te (m)
        }

    #: ``bl`` dict keys, in boundary_status order (the ``side`` output coord).
    _BL_SIDES = _outputs.SIDES

    def _parse_bl_sides(self, bl, boundary_status):
        """Validate a PER-SIDE ``bl`` dict against ``boundary_status`` and return
        a 4-list [left, right, bottom, top] of ``(scalar, series)`` entries —
        one per ``fixed_value`` side (unspecified sides default to
        ``constants.BL``), ``None`` for the others. Only ``fixed_value`` sides
        are base-level outlets, so a datum on any other side is a contradiction
        rather than a no-op and raises. ``boundary_status`` is the checked
        4-list (``_check_boundary_status``, which also guarantees an outlet)."""
        bs = boundary_status
        unknown = sorted(set(bl) - set(self._BL_SIDES))
        if unknown:
            raise ValueError(
                f"unknown bl side(s) {unknown}: bl dict keys must be a subset "
                f"of {list(self._BL_SIDES)}")
        for side in bl:
            status = str(bs[self._BL_SIDES.index(side)])
            if status != 'fixed_value':
                raise ValueError(
                    f"bl[{side!r}] given but that side's boundary_status is "
                    f"{status!r}: only 'fixed_value' sides are base-level "
                    "outlets and carry a water datum.")
        sides = []
        for k, side in enumerate(self._BL_SIDES):
            if bs[k] != 'fixed_value':
                sides.append(None)
                continue
            value = bl.get(side, _constants.BL)
            if value is None:
                sides.append((_constants.BL, None))
                continue
            arr = np.asarray(value, dtype=float)
            if arr.ndim == 0:                     # scalar (incl. a 0-d array)
                sides.append((float(arr), None))
                continue
            if arr.ndim != 1 or arr.shape[0] != self.nt:
                raise ValueError(
                    f"bl[{side!r}] must be a scalar or a length-nt={self.nt} "
                    f"series, got shape {arr.shape}")
            sides.append((float(arr.mean()), arr))
        return sides

    def set_and_check_parameters(self, user_params):
        defaults = self._default_params()

        removed = sorted(set(user_params) & set(_REMOVED_PARAMS))
        if removed:
            raise ValueError("; ".join(f"{k} was removed: {_REMOVED_PARAMS[k]}"
                                       for k in removed) + ".")
        unexpected = set(user_params) - set(defaults)
        if unexpected:
            raise ValueError(f"Unexpected parameter(s): {sorted(unexpected)}")

        params = SimpleNamespace(**{**defaults, **user_params})
        params.boundary_status = _check_boundary_status(params.boundary_status)

        # Mode D (constants.MODE_D_*): mode C with the channel/sheet reading choosing
        # valley or sheet rules per cell. Its defaults go into params HERE,
        # before any knob below is bound, for every key the user left unset
        # (None counts as unset: the sentinels' own default); explicit values win.
        self.regime_switch = _constants.normalize_mode(params.mode) == 'D'
        if self.regime_switch:
            for key, value in (
                    ('width_hack_k', _constants.MODE_D_WIDTH_HACK_K),
                    ('routing_filter_factor', _constants.MODE_D_ROUTING_FILTER_FACTOR),
                    ('flow_routing', _constants.MODE_D_FLOW_ROUTING),
                    ('trunk_surface', _constants.MODE_D_TRUNK_SURFACE),
                    ('channel_bed_bar', _constants.REGIME_BED_BAR)):
                if user_params.get(key) is None:
                    setattr(params, key, value)
        # Under MFD routing (mode D's default) an unset parallel_erode is on
        # only where the layer's fork/join is cheap.
        if (user_params.get('parallel_erode') is None
                and _constants.normalize_flow_routing(params.flow_routing) == 'mfd'):
            params.parallel_erode = (_threads.threading_layer()
                                     in _constants.MFD_PARALLEL_ERODE_LAYERS)
        # Damping (constants.INNER_DAMPINGS). Modes B/C/D default to
        # constants.MODE_BCD_INNER_DAMPING, but an explicit inner_relax picks
        # 'omega' (so inner_relax=1, inner_passes=1 is the undamped model).
        # 'omega' fills each unset dial with the mode's (B/C:
        # constants.MODE_BC_*; D: MODE_D_INNER_RELAX over one pass); an
        # adaptive preset carries its floor in inner_relax, which is then not a
        # dial (an explicit value equal to the floor, a round-trip, is
        # accepted). Mode A has nothing to damp (the plain defaults).
        # lazy: the floor in inner_relax keeps every "the commit can differ from
        # F" gate (inner_relax < 1: the closure-thickness row, the ice_speed
        # and diffusion warnings, the xsimlab refusal) working unchanged, but
        # only while every preset's floor is < 1; upgrade path: one `blends`
        # predicate (damping != 'omega' or inner_relax < 1) at those sites.
        damped_mode = _constants.normalize_mode(params.mode) in ('B', 'C', 'D')
        relax_set = user_params.get('inner_relax') is not None
        damping = params.inner_damping
        if damping is None:
            damping = (_constants.MODE_BCD_INNER_DAMPING
                       if damped_mode and not relax_set else _constants.INNER_DAMPING)
        if damping not in _constants.INNER_DAMPINGS:
            raise ValueError(f"inner_damping must be one of "
                             f"{_constants.INNER_DAMPINGS}, got {damping!r}.")
        if damping in _constants.INNER_DAMPING_FLOORS:
            if not damped_mode:
                raise ValueError(f"inner_damping={damping!r} requires mode "
                                 f"'B', 'C' or 'D', got mode={params.mode!r}.")
            floor = _constants.INNER_DAMPING_FLOORS[damping]
            if relax_set and float(user_params['inner_relax']) != floor:
                raise ValueError(
                    f"inner_relax is the 'omega' preset's dial; "
                    f"inner_damping={damping!r} sets its own floor ({floor:.3g}).")
            params.inner_relax = floor
        elif damped_mode:
            omega_dials = ((_constants.MODE_D_INNER_RELAX, _constants.INNER_PASSES)
                           if self.regime_switch else
                           (_constants.MODE_BC_INNER_RELAX, _constants.MODE_BC_INNER_PASSES))
            for key, value in zip(('inner_relax', 'inner_passes'), omega_dials):
                if user_params.get(key) is None:
                    setattr(params, key, value)
        self.inner_damping = damping

        # initial_topography from ndarray, DataFrame, or CSV file path. When
        # loaded from a file/DataFrame, the derived (nx, ny, Lx, Ly) override
        # any user-supplied grid params.
        if params.initial_topography is not None:
            arr, grid = _load_initial_topography(params.initial_topography)
            params.initial_topography = arr
            if grid is not None:
                params.nx, params.ny = grid['nx'], grid['ny']
                params.Lx, params.Ly = grid['Lx'], grid['Ly']
                print(f"[siim2d] initial_topography loaded: "
                      f"nx={params.nx}, ny={params.ny}, "
                      f"Lx={params.Lx/1e3:.1f} km, Ly={params.Ly/1e3:.1f} km, "
                      f"dx={params.Lx/(params.nx-1):.0f} m, "
                      f"dy={params.Ly/(params.ny-1):.0f} m")

        # 2D grid
        self.grid_nx = params.nx
        self.grid_ny = params.ny
        self.Lx = params.Lx
        self.Ly = params.Ly
        # one node divided by zero in dx; three is the least a looped axis can
        # wrap (the hillslope's cyclic solve) and leaves a fixed axis an
        # interior node (audit #40)
        if not (self.grid_nx >= 3 and self.grid_ny >= 3):
            raise ValueError(f"nx and ny must be >= 3, got nx={self.grid_nx}, ny={self.grid_ny}")
        _check_range('Lx', self.Lx, 0.0)
        _check_range('Ly', self.Ly, 0.0)

        # time
        self.to = params.to
        self.T = params.T
        self.nt = params.nt
        self.nt_out = params.nt_out
        # The clock needs one step at least and must run forward (audit #3, #4,
        # #40): nt = 1 ran zero steps and died in run() on a KeyError, T == to
        # made every erosion rate 0/0, and T < to integrated on a negative dt.
        if not self.nt >= 2:
            raise ValueError(f"nt must be >= 2 (one step at least), got nt={self.nt}")
        if not (np.isfinite(self.to) and np.isfinite(self.T) and self.T > self.to):
            raise ValueError(f"T must be finite and later than to, got to={self.to}, T={self.T}")
        # nt_out > nt is a degenerate output cadence: out_idx would carry
        # duplicate master steps, which xsimlab silently left as NaN frames
        # (and the in-house driver would fill with duplicated data) — reject it
        # loudly instead of replicating either pathology. nt_out = 1 is legal:
        # a single frame-0 snapshot, identical under both drivers.
        if not (1 <= self.nt_out <= self.nt):
            raise ValueError(
                f"nt_out must satisfy 1 <= nt_out <= nt; got nt_out="
                f"{self.nt_out} with nt={self.nt} (more output frames than "
                f"master steps duplicates output indices).")
        self.t = np.linspace(self.to, self.T, self.nt)
        self.dt = np.mean(np.diff(self.t))
        out_idx = np.round(np.linspace(0, self.nt - 1, self.nt_out)).astype(int)
        self.t_out = self.t[out_idx]

        # climate — zELA is the primary input; zT (paper's z_T, the snow-line
        # elevation above which all precipitation falls as snow) is derived.
        # precipitation P: scalar (default constants.P) or a length-nt time
        # series, paralleling zELA below. A series is wired onto the 'tstep'
        # clock in run() (glacial_flow__runoff); the scalar self.P is the
        # analytical/zT-reference value (time-MEAN for a series).
        self._P_series = None
        if np.isscalar(params.P) or params.P is None:
            self.P = _constants.P if params.P is None else params.P
        else:
            P_arr = np.asarray(params.P, dtype=float)
            if P_arr.ndim == 1 and len(P_arr) == self.nt:
                self._P_series = P_arr
                self.P = P_arr.mean()  # scalar for the analytical/zT reference
            else:
                raise ValueError(
                    f"P array must have length nt={self.nt}, got {len(P_arr)}"
                )
        # base level bl: scalar (default constants.BL), a length-nt series
        # wired onto the 'tstep' clock in run() (glacial_spl__bl), paralleling P
        # above, or a PER-SIDE dict {'left'|'right'|'bottom'|'top': scalar or
        # series} giving each fixed_value outlet its own water datum (mode B,
        # in-house driver). The scalar self.bl is the analytical-reference value
        # (time-MEAN for a series; the mean over the fixed sides for a dict) —
        # but the analytical stays at the datum, so a nonzero mean only sets a
        # heads-up note, emitted when the analytical overlay is drawn
        # (plotting.profiles._analytical_overlay; see _analytical_user_params).
        self._bl_series = None
        self._bl_sides = None
        if isinstance(params.bl, dict):
            self._bl_sides = self._parse_bl_sides(params.bl,
                                                  params.boundary_status)
            self.bl = float(np.mean([e[0] for e in self._bl_sides
                                     if e is not None]))
        elif np.isscalar(params.bl) or params.bl is None:
            self.bl = _constants.BL if params.bl is None else float(params.bl)
        else:
            bl_arr = np.asarray(params.bl, dtype=float)
            if bl_arr.ndim == 1 and len(bl_arr) == self.nt:
                self._bl_series = bl_arr
                self.bl = float(bl_arr.mean())
            else:
                raise ValueError(
                    f"bl array must have length nt={self.nt}, got {len(bl_arr)}")
        # Global waterline-flotation gate (mode B). Off is for diagnostics only.
        self.flotation_gate = bool(params.flotation_gate)
        # Flotation-ramp width gamma (0 = hard binary gate, bit-for-bit).
        self.flotation_ramp = float(params.flotation_ramp)
        _check_range('flotation_ramp (gamma)', self.flotation_ramp, 0.0, inclusive=True)
        # Mode-B parallel-eroder toggle (bit-for-bit; default constants.PARALLEL_ERODE).
        self.parallel_erode = bool(params.parallel_erode)
        self._analytical_bl_note = None
        if self._bl_sides is not None:
            if any(e[0] != 0.0 for e in self._bl_sides if e is not None):
                self._analytical_bl_note = (
                    "Per-side base level bl: the analytical steady-state "
                    "reference stays at the datum (bl=0) and CANNOT be "
                    "offset-corrected, since the outlets sit at different "
                    "data — the analytical overlay is not meaningful across "
                    f"them (self.bl = {self.bl:g} is the fixed-side mean, a "
                    "label only).")
        elif self.bl != 0.0:
            self._analytical_bl_note = (
                "Nonzero base level bl: the analytical steady-state reference "
                "stays at the datum (bl=0), so the analytical overlay is "
                "offset by ~bl.")

        self.beta = params.beta
        _check_range('beta', self.beta, 0.0)   # every zELA/zT conversion divides by it (audit #41)
        if params.zELA is None and params.zT is None:
            self.zELA = _constants.ZELA
        elif params.zELA is None and params.zT is not None:
            self.zELA = params.zT - self.P / params.beta
        else:
            self.zELA = params.zELA

        # zELA: scalar, (nt,) series, (ny, nx) field or (nt, ny, nx) field series.
        # A series goes on the 'tstep' clock (_zELA_series), a static field to
        # the mass balance as is (_zELA_field). The scalar self.zELA is the
        # analytical / zT reference: a series' minimum (most glaciated), a
        # field's spatial mean (the minimum over time of it for a field series;
        # the zT + P-series path below keeps its time mean).
        self._zELA_series = None
        self._zELA_field = None
        grid = (self.grid_ny, self.grid_nx)
        if not np.isscalar(self.zELA):
            zELA_arr = np.asarray(self.zELA, dtype=float)     # the constructor's deep copy, never the caller's array
            if not np.isfinite(zELA_arr).all():
                raise ValueError("zELA must be finite everywhere (a field or series with NaN/inf would "
                                 "carry NaN into the mass balance and the analytical reference)")
            if zELA_arr.shape == (self.nt,):
                self._zELA_series = zELA_arr
                self.zELA = np.min(zELA_arr)  # scalar for analytical solution
            elif zELA_arr.shape == grid:
                self._zELA_field = zELA_arr
                self.zELA = float(zELA_arr.mean())
            elif zELA_arr.shape == (self.nt, *grid):
                self._zELA_series = zELA_arr
                self.zELA = float(zELA_arr.mean(axis=(1, 2)).min())
            else:
                raise ValueError(
                    f"zELA must be a scalar, (nt,) = ({self.nt},), (ny, nx) = "
                    f"{grid}, or (nt, ny, nx) = {(self.nt, *grid)}; got shape "
                    f"{zELA_arr.shape}")
        # zT given + P-series ⇒ time-varying zELA on the clock: zELA_run(t) =
        # zT − P(t)/β (paper semantics; the scalar self.zELA keeps the time-MEAN
        # for the analytical reference), per cell for an (ny, nx) zT. Only this
        # zT-input path changes; the zELA-input path stays bit-for-bit. (audit F2)
        # A zT series pairs with the P series step by step too (it used the P's
        # time mean, a constant zELA; audit #34).
        if (self._P_series is not None
                and params.zELA is None and params.zT is not None):
            zT_shape = np.shape(params.zT)
            if zT_shape not in ((), grid, (self.nt,), (self.nt, *grid)):
                raise ValueError(f"zT must be a scalar, an (ny, nx) = {grid} field or an (nt,) / "
                                 f"(nt, ny, nx) series, got shape {zT_shape}")
            timed = zT_shape[:1] == (self.nt,) and len(zT_shape) in (1, 3)
            # lazy: a field zT materialises the (nt, ny, nx) series (~200 MB at nt 2501 on 100^2);
            # upgrade path: keep zT and P apart and form zT - P[k]/beta per step in the driver.
            P_t = self._P_series.reshape((-1,) + (1,) * (len(zT_shape) - timed))
            self._zELA_series = params.zT - P_t / self.beta
            self._zELA_field = None
            if timed:   # the series rule for the analytical scalar: its most glaciated step
                self.zELA = float(self._zELA_series.reshape(self.nt, -1).mean(axis=1).min())
        self.zT = self.zELA + self.P / self.beta

        # uplift — scalar, (nt,), (ny, nx), or (nt, ny, nx) array.
        if np.isscalar(params.U):
            self.U = params.U
            self._U_user = None
        else:
            U_arr = np.asarray(params.U, dtype=float)
            if U_arr.shape == (self.nt,):
                # (nt,) = spatially-uniform, time-varying uplift; threaded as a
                # per-step scalar on the clock (m51), like zELA/P — no full
                # (nt, ny, nx) materialised.
                pass
            elif U_arr.shape == (params.ny, params.nx):
                pass
            elif U_arr.shape == (self.nt, params.ny, params.nx):
                pass
            else:
                raise ValueError(
                    f"U must be a scalar, (nt,) = ({self.nt},), (ny, nx) = "
                    f"({params.ny}, {params.nx}), or (nt, ny, nx) = "
                    f"({self.nt}, {params.ny}, {params.nx}); got shape {U_arr.shape}")
            self.U = float(U_arr.mean())  # scalar reduction for the analytical reference
            self._U_user = U_arr

        # initial topography — optional (ny, nx) array.
        if params.initial_topography is None:
            self.initial_topography = None
        else:
            z0 = np.asarray(params.initial_topography, dtype=float)
            if z0.shape != (params.ny, params.nx):
                raise ValueError(
                    f"initial_topography must be shape (ny, nx) = "
                    f"({params.ny}, {params.nx}); got {z0.shape}")
            self.initial_topography = z0

        # fluvial
        self.n = params.n
        self.m = params.m if params.m is not None else self.n / 2
        self.Ko = params.Ko

        # glacial
        self.Ac = params.Ac
        self.alpha_g = params.alpha_g
        # lambda_p: plain user param, default constants.LAMBDA_P (a merge
        # default, so a user value trumps).
        self.lambda_p = float(params.lambda_p)
        self.lambda_c = params.lambda_c if params.lambda_c is not None else _constants.LAMBDA_C
        self.sliding_law = params.sliding_law
        self.ce = params.ce
        self.tau_c = params.tau_c
        # 0 divided by zero at construction and < 0 ran silently (audit #8)
        _check_range('lambda_p', self.lambda_p, 0.0)
        _check_range('lambda_c', self.lambda_c, 0.0)
        _check_range('tau_c', self.tau_c, 0.0)
        self.width_hack_k = float(params.width_hack_k)
        self.width_hack_p = float(params.width_hack_p)
        self.flow_routing = _constants.normalize_flow_routing(params.flow_routing)
        # MFD spreading exponent (inert under 'single' / 'dinf'; validated
        # always, so a typo is loud rather than silently ignored). Mode D runs
        # on MFD alone, checked first so the refusal never follows the
        # resolver's warning, and refuses an exponent that resolves to inf
        # (single flow by another name).
        if self.regime_switch and self.flow_routing != 'mfd':
            raise ValueError(
                "mode='D' requires flow_routing='mfd': the channel/sheet reading grows "
                "false channels along single-flow and D-inf flux trees. Got "
                f"flow_routing={params.flow_routing!r}.")
        self.mfd_exponent = _constants.resolve_mfd_exponent(params.mfd_exponent,
                                                            finite=self.regime_switch)
        if params.router_backend != 'inhouse_d8':
            raise ValueError(
                f"router_backend must be 'inhouse_d8' (the retired 'fortran' "
                f"SFR was removed at the 0.9.1 standalone flip), "
                f"got {params.router_backend!r}"
            )
        self.router_backend = params.router_backend
        if params.numerics_backend != 'inhouse':
            raise ValueError(
                f"numerics_backend must be 'inhouse' (the retired 'fortran' "
                f"flexure/diffusion was removed at the 0.9.1 standalone flip), "
                f"got {params.numerics_backend!r}"
            )
        self.numerics_backend = params.numerics_backend
        self.k = params.k                          # accumulation shape exponent (analytical only)

        # Mode A / B (both routings).
        self.mode = _constants.normalize_mode(params.mode)
        # Mode 'C' is the user-facing alias for the mode-B + sub-grid-carve
        # configuration (selected by class: GlacialSPLModeC) — not a new
        # dynamical mode. Resolve it to internal 'B' with carving implied on;
        # an explicit carve_width=False contradicts the name (be loud). Mode
        # 'D' resolves the same way (its own defaults were written above).
        if self.mode in ('C', 'D'):
            if user_params.get('carve_width') is False:
                raise ValueError(
                    f"mode={self.mode!r} carves (mode B + sub-grid carve"
                    f"{' + the channel/sheet reading' if self.mode == 'D' else ''}); "
                    "carve_width=False contradicts it (use mode='B' for plain "
                    "bed memory without carving).")
            params.carve_width = True
            self.mode = 'B'

        # Optional H diffusivity (mode B only in 2D; None disables — see m17).
        if params.H_diffusivity is None:
            self.H_diffusivity = None
        else:
            D_H = float(params.H_diffusivity)
            _check_range('H_diffusivity', D_H, 0.0, inclusive=True)
            self.H_diffusivity = D_H
        # raw; resolved per-cell/per-step by _make_border_bed_uplift()
        self.border_bed_uplift = params.border_bed_uplift
        # carve_width default (None) resolves to OFF: carving is OPT-IN, reached
        # via mode='C' (handled above) or an explicit carve_width=True. So an
        # unqualified mode='B' does NOT carve (Eric, 2026-07-07). mode A can't.
        if params.carve_width is None:
            params.carve_width = False
        self.carve_width = bool(params.carve_width)
        if self.carve_width and self.mode != 'B':
            # The only way here is an EXPLICIT carve_width=True with mode='A'
            # (mode C already became 'B' above) — a genuine contradiction: mode A
            # re-derives the bed from the surface and heals carved troughs
            # instantly.
            raise ValueError("carve_width=True requires mode='B' (bed memory: "
                             "mode A heals carved troughs instantly).")
        self.widening_rate = params.widening_rate
        # validate now (fail-fast at construction); the fastscape process
        # converts widening_rate (eta) to the internal (1 + eta) factor.
        self.widening_factor = _constants.widening_factor_from_rate(self.widening_rate)
        # --- Mode-C standard defaults -------------------------------------------
        # Mode C (mode B + sub-grid carve — the flagship carved mode) resolves its
        # unset trunk_surface sentinel (None) to the mode-C standard
        # (constants.mode_c_standard, shared with the fastscape facade); plain
        # mode B (no carve) and mode A keep it OFF. An EXPLICIT
        # user value (not None) always wins. widening_rate has no sentinel — its
        # constants default (3.0) is the mode-C standard and is a no-op wherever
        # carve is off. Resolved here, after carve_width is finalized (line above).
        if params.trunk_surface is None:
            params.trunk_surface = _constants.mode_c_standard(
                self.mode == 'B' and self.carve_width)
        # Fabricated trunk-surface routing (mode B/C only). Routes on a
        # fabricated ice surface so trunk flow converges to the centerline
        # (raw flux there = full cross-section).
        self.trunk_surface = bool(params.trunk_surface)
        self.trunk_dip_k = _resolve_trunk_dip_k(params.trunk_dip_k,
                                                self.flow_routing)
        if self.trunk_surface and self.mode != 'B':
            raise ValueError("trunk_surface=True requires mode='B' "
                             "(mode A reconstructs the bed; no trunk to converge).")
        # Within-step damped iteration (mode B/C). Validated AFTER the mode
        # resolution above; single-sourced in _core.router so the fastscape
        # facade raises the same errors.
        self.inner_passes, self.inner_relax = _check_inner_loop_config(
            params.inner_passes, params.inner_relax, mode=self.mode)
        if self.inner_relax < 1.0 and self.H_diffusivity:
            warnings.warn(
                "inner_relax < 1 damping attenuates the diffusion correction: "
                "H diffusion runs on the final pass only and the committed "
                "state blends the diffused closure with the undiffused "
                f"iterate (inner_damping={self.inner_damping!r}, "
                f"inner_relax={self.inner_relax:.3g}, "
                f"H_diffusivity={self.H_diffusivity}).", stacklevel=2)
        # Routing-surface spatial filter (mode B/C), validated the same way.
        # Grid spacing (shared with the MFD guards below) is resolved as
        # _build_driver_config does.
        dx, dy = self.Lx / (self.grid_nx - 1), self.Ly / (self.grid_ny - 1)
        if params.channel_bed_bar is not None and not self.regime_switch:
            raise ValueError("channel_bed_bar is mode D's channel/sheet dial; "
                             f"it requires mode='D', got mode={params.mode!r}.")
        self.channel_bed_bar = params.channel_bed_bar   # mode D's default is set above
        if self.regime_switch:
            bar = self.channel_bed_bar
            # a real number, not a bool; the chained comparison is exact for any
            # int and fails for NaN and inf. The bound is Python's float: numpy 2
            # raises OverflowError comparing a huge int with np.finfo(float).max
            if not (isinstance(bar, numbers.Real) and not isinstance(bar, bool)
                    and 0.0 <= bar <= sys.float_info.max):
                raise ValueError("channel_bed_bar must be a finite number >= 0 (0 = no bar), "
                                 f"got {params.channel_bed_bar!r}.")
            self.channel_bed_bar = float(bar)
            # square cells, a finite carve widening factor, and rays that do
            # not wrap onto themselves (constants.REGIME_DX_WARN: the reading
            # was measured at 1 km - 250 m)
            bs = params.boundary_status
            _check_regime_grid(
                self.grid_ny, self.grid_nx, dx,
                wrap_y=bs[2] == 'looped', wrap_x=bs[0] == 'looped', dy=dy,
                widening_factor=self.widening_factor if self.carve_width else None)
            if dx > _constants.REGIME_DX_WARN:
                warnings.warn(
                    f"mode='D' at {dx / 1e3:g} km cells: the channel/sheet reading "
                    "needs the glacier's width resolved and was measured at "
                    f"1 km - 250 m; on grids coarser than "
                    f"{_constants.REGIME_DX_WARN / 1e3:g} km most outlets are sub-grid.",
                    stacklevel=2)
        self.routing_filter_factor = _check_routing_filter_config(
            params.routing_filter_factor, mode=self.mode,
            trunk_surface=self.trunk_surface, dx=dx, dy=dy,
            regime_switch=self.regime_switch)
        # closure_flux (EXPERIMENTAL): 'cell' | 'transect' | 'transect_min'; the
        # None sentinel resolves per mode (transect_min for B/C, cell for A).
        if params.closure_flux is None:
            params.closure_flux = _constants.CLOSURE_FLUX if self.mode == 'B' else _constants.MODE_A_CLOSURE_FLUX
        self.closure_flux = str(params.closure_flux)
        if self.closure_flux not in _constants.CLOSURE_FLUXES:
            raise ValueError("closure_flux must be 'cell', 'transect' or "
                             f"'transect_min', got {params.closure_flux!r}.")
        if self.closure_flux != 'cell' and self.mode != 'B':
            raise ValueError("closure_flux != 'cell' requires mode 'B' or 'C' "
                             "(the transect closure is a mode-B kernel input).")
        # gamma effective-slope normalisation. A multi-receiver reduction
        # returns the gradient PROJECTED on the weighted receiver vector, so any
        # spreading router needs the 1/gamma correction to feed |grad zs| to the
        # closure and the eroders. Default (None) = router-derived from
        # _core.router.SLOPE_NORMALISED_ROUTERS ('dinf' and 'mfd' on, 'single'
        # off); an explicit value wins, which is how the pre-2026-09-05 D-inf
        # behaviour stays reachable (slope_normalise=False).
        if params.slope_normalise is None:
            self.slope_normalise = self.flow_routing in _SLOPE_NORMALISED_ROUTERS
        else:
            self.slope_normalise = bool(params.slope_normalise)
        # MFD coherence guards. Single-sourced in
        # _core.router so siim2d, the fastscape facade and the adapter processes
        # raise the same errors.
        _check_mfd_config(
            self.flow_routing, mode=self.mode, closure_flux=self.closure_flux,
            dx=dx, dy=dy, slope_normalise=self.slope_normalise)
        # Citizen Mode B: the bed is tracked AS topography (topography__elevation
        # = zb), routing on the reconstructed ice surface (GlacialSurfaceToErode),
        # erosion = denudation. Both mode-B paths are citizens now — no carve
        # (GlacialSPLModeB) and carve (GlacialSPLModeC) alike — so bed-as-topography
        # holds whenever mode == 'B'.
        self._citizen_mode_b = (self.mode == 'B')
        # hc_over_H: the run's channel-floor datum (per mode: docs/guides/parameter_reference.md).
        self.hc_over_H = (float(_constants.HC_OVER_H) if params.hc_over_H is None
                          else float(params.hc_over_H))
        if not (np.isfinite(self.hc_over_H) and self.hc_over_H > 0.0):
            raise ValueError("hc_over_H must be finite and > 0, got "
                             f"{params.hc_over_H!r}.")

        # Physical constants needed for the per-law dispatch
        self.kt = KT
        self.g = GRAVITY
        self.rho_g = RHO_ICE
        # cg = alpha_g * kt * (2*Ac/5) * (rho_g g)^3  [m^-3 yr^-1]  (model paper;
        # 2/5 = depth-integrated Glen prefactor 2A/(n_c+2), n_c = 3. kt absorbed
        # into cg so the H- and erosion-formulas carry no explicit kt; matches
        # siim1d / siim.analytical).
        self.cg = _constants.cg_prefactor(self.alpha_g, self.Ac, self.rho_g, self.g)

        # Slope/erosion exponents and Co — per-law dispatch.
        # Coulomb: mu = nu/2. eff-exp / power: mu = 4*nu/15.
        if user_params.get('nu') is not None and user_params.get('ell') is not None:
            warnings.warn(
                "Both 'nu' and 'ell' were provided; 'ell' takes precedence and "
                "'nu' is ignored (re-derived from ell by the sliding-law relations).",
                UserWarning, stacklevel=2)
        # the exponent the law is derived from; nu and ell share its sign (audit #8)
        exponent = 'ell' if params.ell is not None else 'nu'
        _check_range(exponent, getattr(params, exponent), 0.0)
        if self.sliding_law == 'coulomb':
            c = derive_coulomb(self.ce, self.alpha_g, self.tau_c,
                               self.rho_g, self.g,
                               nu=params.nu, ell=params.ell)
        else:
            c = derive_power(self.ce, self.cg, self.lambda_p, self.alpha_g,
                             nu=params.nu, ell=params.ell)
        # Co is computed by derive_* but not stored on siim2d (GlacialLaw
        # recomputes its own _Co from ce/cg/lambda_p/alpha_g/mu).
        self.ell, self.nu, self.mu, self.phi, _ = c
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

        # hillslope + run config
        self.D = params.D
        _check_range('D', self.D, 0.0, inclusive=True)   # < 0 anti-diffused, NaN spread (audit #29, #38)
        self.boundary_status = params.boundary_status   # checked 4-list (_check_boundary_status)
        self.initial_max_elevation = params.initial_max_elevation
        self.noise_amplitude = params.noise_amplitude
        self.seed = params.seed
        self.progress_bar = params.progress_bar

        # flexural isostasy (true GIA: ice loading + erosional/tectonic unloading)
        self.flexure = bool(params.flexure)
        self.ice_load = bool(params.ice_load)
        # track_sediment: False | True/'basin' (per-node rasters) | 'edge'
        # (per-domain-edge totals) | 'both'. Anything not a string keeps the
        # old bool() reading first (1, np.True_, a legacy saved flag).
        track = params.track_sediment
        if not isinstance(track, str):
            track = 'basin' if track else False
        if track not in (False, 'basin', 'edge', 'both'):
            raise ValueError(
                "track_sediment must be False, True, 'basin', 'edge' or "
                f"'both', got {params.track_sediment!r}")
        self.track_sediment = track
        if not isinstance(params.save_closure_pair, (bool, np.bool_)):
            raise ValueError("save_closure_pair must be True or False, got "
                             f"{params.save_closure_pair!r}")
        self.save_closure_pair = bool(params.save_closure_pair)
        self.sediment_basin = track in ('basin', 'both')
        self.sediment_edge = track in ('edge', 'both')
        self.lithos_density = params.lithos_density
        self.asthen_density = params.asthen_density
        self.e_thickness = params.e_thickness
        if self.flexure:
            # rho_a = Te = 0 divided by zero in the plate transfer (audit #29);
            # Te = 0 alone is Airy isostasy. The plate takes scalar Te and rho_a.
            _check_range('lithos_density', self.lithos_density, 0.0)
            _check_range('asthen_density', self.asthen_density, 0.0)
            _check_range('e_thickness', self.e_thickness, 0.0, inclusive=True)
            if np.ndim(self.asthen_density) or np.ndim(self.e_thickness):
                raise ValueError("asthen_density and e_thickness must be scalars "
                                 "(only lithos_density may be an (ny, nx) field)")

        self.coulomb_clamp = params.coulomb_clamp
        # >= 1 zeroed the ice everywhere; < 0 let phi pass the yield pole (audit #8)
        _check_range('coulomb_clamp', self.coulomb_clamp, 0.0, hi=1.0)

    @property
    def run_id(self):
        # the last step's ELA (its spatial mean for a field series)
        ela = (np.mean(self._zELA_series[-1]) if self._zELA_series is not None
               else self.zELA)
        return f"Ko{self.Ko:.0e}_ce{self.ce:.0e}_n{self.n:g}_nu{self.nu:g}_ELA{ela:g}"

    def _set_analytical_grid(self, L, xo, k_h, d):
        """Rebuild the 1D analytical reference with channel-fitted (L, xo, k_h, d).
        Called by extract_channel after fitting Hack's law to a numerical channel."""
        self.L = L
        self.xo = xo
        self.k_h = k_h
        self.d = d
        new_user = {**self._user_params, "xo": xo, "k_h": k_h, "d": d}
        new_params = self._analytical_user_params(new_user)
        new_params["L"] = L  # use the fitted channel length, not Lx
        self.analytical = self._build_analytical(new_params)
        self.x = None if self.analytical is None else self.analytical.x

    def _make_initial_topo(self):
        """Build initial topography that slopes toward fixed_value boundaries.

        Returns a (ny, nx) array with elevation 0 at fixed_value sides and
        ``initial_max_elevation`` at the farthest interior point.  If no
        sides are 'core', the surface slopes from the centre toward all
        fixed_value edges.
        """
        ny, nx = self.grid_ny, self.grid_nx

        if self.initial_topography is not None:
            return self.initial_topography

        if self.initial_max_elevation == 0:
            return np.zeros((ny, nx))

        bs = self.boundary_status  # [left, right, bottom, top], normalised in __init__

        x = np.linspace(0, 1, nx)
        y = np.linspace(0, 1, ny)
        X, Y = np.meshgrid(x, y)

        fixed_sides = [i for i, s in enumerate(bs) if s == 'fixed_value']   # >= 1 (checked)

        distances = []
        for side in fixed_sides:
            if side == 0:    # left:   x = 0
                distances.append(X)
            elif side == 1:  # right:  x = Lx
                distances.append(1 - X)
            elif side == 2:  # bottom: y = 0
                distances.append(Y)
            elif side == 3:  # top:    y = Ly
                distances.append(1 - Y)

        min_dist = np.minimum.reduce(distances)
        max_val = min_dist.max()
        if max_val > 0:
            topo = min_dist / max_val * self.initial_max_elevation
        else:
            topo = np.zeros((ny, nx))

        return topo

    def _make_uplift_field(self):
        """Build the uplift rate field.

        Returns a `(ny, nx)` array (static) or an xsimlab-style
        `(dims, array)` tuple `(('tstep', 'y', 'x'), array)` for time-varying
        input. xsimlab slices the time axis per step before BlockUplift sees it.
        """
        ny, nx = self.grid_ny, self.grid_nx

        if self._U_user is None:
            return np.full((ny, nx), self.U)

        if self._U_user.ndim == 2:
            return self._U_user

        if self._U_user.ndim == 1:
            # (nt,) spatially-uniform series → per-step scalar on the clock
            # (m51). xsimlab slices it to a scalar each step; GlacialBlockUplift
            # (and the border-bed budget) broadcast a scalar rate to the grid,
            # so no (nt, ny, nx) array is built.
            return (('tstep',), self._U_user)

        # ndim == 3, (nt, ny, nx)
        return (('tstep', 'y', 'x'), self._U_user)

    def _make_border_bed_uplift(self):
        """Border-bed uplift for the glacial erosion process: the tectonic uplift
        field (local, per-step — same wrapping as _make_uplift_field) when
        border_bed_uplift is None; otherwise the user's value, scalar or
        (ny, nx) or (nt, ny, nx)."""
        if self.border_bed_uplift is None:
            return self._make_uplift_field()
        v = self.border_bed_uplift
        if np.isscalar(v):
            return float(v)
        arr = np.asarray(v, dtype=float)
        return arr if arr.ndim == 2 else (('tstep', 'y', 'x'), arr)

    def _process_overrides(self):
        """xsimlab process-class overrides applied to fastscape's basic_model
        (the stock 'spl' / 'drainage' slots are dropped in run() before these
        are applied). Built from :func:`siim.fastscape.glacial_processes` — the
        single source of truth shared with external fastscape users, so the two
        never diverge. Subclasses extend the returned dict to swap in alternative
        forcing processes (e.g. siim_escarpment replaces 'uplift' and
        'init_topography')."""
        # Adapter-only seam: lazy import so `import siim.siim2d` stays
        # stack-free (the guarded siim.fastscape raises a directed ImportError
        # in a fastscape-less env).
        from .fastscape import glacial_processes
        # mode A -> GlacialSPLModeA; mode B + carve -> GlacialSPLModeC citizen;
        # mode B + no-carve -> GlacialSPLModeB citizen (both add GlacialSurfaceToErode).
        return glacial_processes(
            mode='D' if self.regime_switch else self.mode,   # D is refused by name
            carve=self.carve_width,
            routing=self.flow_routing,
            router_backend=self.router_backend,
            flexure=self.flexure,
            # One tracker serves both reports; output_vars picks what is stored.
            sediment=self.sediment_basin or self.sediment_edge,
            trunk_surface=self.trunk_surface,
            numerics_backend=self.numerics_backend,
            # Forwarded so the adapter path REFUSES the standalone-only sheet
            # knobs instead of silently dropping them (the xsimlab process graph
            # has neither a second-routing seam nor a filtered-surface seam).
            inner_passes=self.inner_passes,
            inner_relax=self.inner_relax,
            routing_filter_factor=self.routing_filter_factor,
        )

    def _forcing_input_vars(self):
        """Input-vars for the initial-topography and uplift processes, split out
        so subclasses that override those processes (e.g. siim_escarpment's
        WaveUplift / PlateauSurface) can supply their own keys instead."""
        forcing = {
            'init_topography__elevation_init': self._make_initial_topo(),
            'uplift__rate': self._make_uplift_field(),
        }
        if self.seed is not None:
            forcing['init_topography__seed'] = self.seed
        return forcing

    def run(self, hooks=None, driver=None):
        """Run the model and unpack outputs. ``driver`` selects the time-loop
        backend: ``'xsimlab'`` (the fastscape/xsimlab adapter orchestration) or
        ``'inhouse'`` (siim's own framework-free loop, :mod:`siim._core.driver`);
        default ``constants.DRIVER_DEFAULT``. Both produce the identical
        ``ds_out`` (bit-for-bit on the same backend). ``hooks`` (an xsimlab
        ``RuntimeHook``) is honored ONLY by the xsimlab driver — the standalone
        driver drops it (OQ-2)."""
        # Invalidate the profile-channel cache the plotting layer actually reads
        # (plotting/profiles.py::_get_channel), so a re-run re-extracts instead
        # of re-plotting the previous run's channel.
        self._profile_channel = None
        self._profile_channel_key = None
        driver = driver if driver is not None else _constants.DRIVER_DEFAULT
        if driver == 'inhouse':
            self._run_inhouse(hooks)
        elif driver == 'xsimlab':
            self._run_xsimlab(hooks)
        else:
            raise ValueError(f"driver must be 'inhouse' or 'xsimlab', got {driver!r}")
        self._unpack_outputs()

    def _run_xsimlab(self, hooks):
        """The fastscape/xsimlab adapter time loop: assemble the xsimlab model
        from :func:`glacial_processes`, run it, and store ``self.ds_out``.
        Adapter-env only (conda; ``environment.yml``) — the stack imports are
        lazy so the standalone default path never touches them."""
        if self._bl_sides is not None:
            raise NotImplementedError(
                "per-side bl (a dict) is in-house-driver only: the xsimlab "
                "adapter's glacial_spl__bl is a single scalar per step. Run "
                "with driver='inhouse' (the default).")
        if self._zELA_series is not None and self._zELA_series.ndim == 3:
            raise NotImplementedError(
                "an (nt, ny, nx) zELA series is in-house-driver only: the "
                "xsimlab adapter takes a scalar, an (nt,) series or a static "
                "(ny, nx) field. Run with driver='inhouse' (the default).")
        import xsimlab as xs
        from fastscape.models import basic_model
        # Drop the stock spl/drainage slots; the renamed glacial_spl / glacial_flow
        # slots (+ law) are added by _process_overrides via update_processes.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="dropping variables using",
                category=FutureWarning, module="xsimlab")
            model = (basic_model
                     .drop_processes(['spl', 'drainage'])
                     .update_processes(self._process_overrides()))

        input_vars = {
            'grid__shape': [self.grid_ny, self.grid_nx],
            'grid__length': [self.Ly, self.Lx],
            'boundary__status': self.boundary_status,
            'init_topography__noise_amplitude': self.noise_amplitude,
            'glacial_flow__runoff': (
                ('tstep',), self._P_series
            ) if self._P_series is not None else self.P,
            'glacial_flow__beta': self.beta,
            'glacial_flow__zELA': (
                ('tstep',), self._zELA_series
            ) if self._zELA_series is not None else self._zELA_static,
            # Law params (raw inputs + constant algebra) now live on GlacialLaw.
            'law__n': self.n,
            'law__nu': self.nu,
            'law__Ko': self.Ko,
            'law__Ac': self.Ac,
            'law__lambda_p': self.lambda_p,
            'law__lambda_c': self.lambda_c,
            'law__tau_c': self.tau_c,
            'law__coulomb_clamp': self.coulomb_clamp,
            'law__m': self.m,
            'law__mu': self.mu,
            'law__alpha_g': self.alpha_g,
            'law__sliding_law': self.sliding_law,
            'law__ce': self.ce,
            'law__H_diffusivity': self.H_diffusivity,
            # self.hc_over_H captured the constant at construction (the
            # monkeypatch-before-setup override convention), single-sourced here
            # + reused by the output reconstruction and the plotters.
            'law__hc_over_H': self.hc_over_H,
            # alpha_g is single-sourced on GlacialLaw (law__alpha_g above):
            # there is no glacial_flow__alpha_g.
            'glacial_flow__width_hack_k': self.width_hack_k,
            'glacial_flow__width_hack_p': self.width_hack_p,
            'glacial_spl__border_bed_uplift': self._make_border_bed_uplift(),
            'glacial_spl__carve_width': self.carve_width,
            'glacial_spl__closure_flux': self.closure_flux,
            'glacial_spl__slope_normalise': self.slope_normalise,
            'glacial_spl__widening_rate': self.widening_rate,
            **({'flow__mfd_exponent': self.mfd_exponent}
               if self.flow_routing == 'mfd' else {}),
            # Base level: scalar, or a length-nt series on the 'tstep' clock
            # (per-step waterline datum, like glacial_flow__zELA above).
            'glacial_spl__bl': (
                ('tstep',), self._bl_series
            ) if self._bl_series is not None else self.bl,
            'glacial_spl__flotation_gate': self.flotation_gate,
            'glacial_spl__flotation_ramp': self.flotation_ramp,
            'glacial_spl__parallel_erode': self.parallel_erode,
            'diffusion__diffusivity': self.D,
            'glacial_spl__ice_thickness': np.zeros((self.grid_ny, self.grid_nx)),
        }
        input_vars.update(self._forcing_input_vars())
        if self.trunk_surface:
            # The trunk-surface provider (surf2erode slot) owns the dip knob.
            input_vars['surf2erode__trunk_dip_k'] = self.trunk_dip_k
        if self.sediment_basin or self.sediment_edge:
            # The tracker skips the per-node running integral nothing stores
            # (the driver gates the same add on cfg.sediment).
            input_vars['sediment__basin'] = self.sediment_basin
        if self.flexure:
            input_vars.update({
                'flexure__lithos_density': self.lithos_density,
                'flexure__asthen_density': self.asthen_density,
                'flexure__e_thickness': self.e_thickness,
                'flexure__ice_load': self.ice_load,
                'flexure__numerics_backend': self.numerics_backend,
            })

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*IndexVariable.*", category=FutureWarning)
            warnings.filterwarnings(
                "ignore", message="dropping variables using",
                category=FutureWarning, module="xsimlab")
            warnings.filterwarnings("ignore", message=".*squeeze.*", category=UserWarning)

            # Both orchestration paths consume one output-schema definition.
            # Dtypes matter only to the in-house buffers; xsimlab needs the
            # same active names, each sampled on its ``time`` clock.
            output_vars = {
                name: 'time'
                for name, _dtype, _dims in _outputs.output_spec(
                    self.mode, self.flexure, self.sediment_basin,
                    self.sediment_edge, regime_switch=self.regime_switch,
                    closure_flux=self.closure_flux,
                    inner_relax=self.inner_relax,
                    save_closure_pair=self.save_closure_pair)
            }

            ds_in = xs.create_setup(
                model=model,
                clocks={'time': self.t_out, 'tstep': self.t},
                master_clock='tstep',
                input_vars=input_vars,
                output_vars=output_vars,
            )

            # Disable zarr's default blosc compression on output variables.
            # Compression adds ~15-20 ms per variable per save, which dominates
            # save-step cost at 300x200+ resolutions.
            encoding = {var: {'compressor': None} for var in output_vars}

            if self.progress_bar:
                with model, xs.monitoring.ProgressBar():
                    self.ds_out = ds_in.xsimlab.run(hooks=hooks, encoding=encoding)
            else:
                with model:
                    self.ds_out = ds_in.xsimlab.run(hooks=hooks, encoding=encoding)

        if 'side' in self.ds_out.dims:      # xsimlab stores the dim, not the labels
            self.ds_out = self.ds_out.assign_coords(side=list(_outputs.SIDES))

    def _run_inhouse(self, hooks):
        """siim's own framework-free time loop (:func:`siim._core.driver.run_loop`)
        calling the SAME step functions the adapter shells call — the standalone
        default path (in-house routing, flexure and diffusion; no stack import).
        Packs the identical ``ds_out`` via :mod:`siim._core.outputs`. ``hooks``
        is rejected here (OQ-2): a RuntimeHook has no meaning off the xsimlab
        orchestrator."""
        if hooks is not None:
            raise ValueError(
                "hooks= is only supported by driver='xsimlab'; the standalone "
                "in-house driver does not support the xsimlab RuntimeHooks "
                "kwarg (OQ-2).")
        cfg = self._build_driver_config()
        # Fully framework-free numba D8 / D-inf producer (S4).
        from ._core.router import InhouseRouter
        router = InhouseRouter([self.grid_ny, self.grid_nx],
                               self.boundary_status, self.flow_routing,
                               cfg.dx, cfg.dy, self.mfd_exponent)
        with router as route:
            cfg.route = route
            buffers = _driver.run_loop(cfg)
        spec = _outputs.output_spec(self.mode, self.flexure, self.sediment_basin,
                                    self.sediment_edge,
                                    regime_switch=self.regime_switch,
                                    closure_flux=self.closure_flux,
                                    inner_relax=self.inner_relax,
                                    save_closure_pair=self.save_closure_pair)
        self.ds_out = _outputs.build_dataset(buffers, spec, cfg.t_out, cfg.x,
                                             cfg.y, self.t)

    def _driver_initial_surface(self):
        """The initial topography array for the in-house driver (mode-B bed /
        mode-A surface). Base = :func:`initial_topography` on
        ``_make_initial_topo()``; subclasses (escarpment) override for
        alternative initial-condition processes (PlateauSurface)."""
        shape = (self.grid_ny, self.grid_nx)
        return initial_topography(
            self._make_initial_topo(), shape, self.boundary_status,
            self.seed, self.noise_amplitude)

    def _driver_uplift_fn(self):
        """A per-step ``uplift_fn(k, dt) -> (ny, nx)`` for the in-house driver.
        Base = block uplift (``block_uplift`` on the resolved rate field, sliced
        directly by step); subclasses (escarpment) override for WaveUplift."""
        shape = (self.grid_ny, self.grid_nx)
        mask = uplift_mask(self.boundary_status, shape)
        spec = self._make_uplift_field()
        if isinstance(spec, tuple):                # clock-threaded series
            rate_arr = np.asarray(spec[1], dtype=float)
            return lambda k, dt: block_uplift(rate_arr[k], dt, mask, shape)
        return lambda k, dt: block_uplift(spec, dt, mask, shape)

    def _driver_bbu_fn(self):
        """A per-step ``bbu_fn(k, dt) -> (ny, nx)`` border-bed uplift for the in-house
        driver, or None for the static / clock-sliced ``_make_border_bed_uplift``
        (the base). The escarpment wave overrides it so the border follows the wave."""
        return None

    @property
    def _zELA_static(self):
        """The run's static zELA: the (ny, nx) field when one was given, else the scalar."""
        return self.zELA if self._zELA_field is None else self._zELA_field

    def _build_driver_config(self):
        """Assemble the resolved-parameter bundle the in-house driver reads
        (:func:`siim._core.driver.run_loop`). Every value flows from the
        already-validated ``self.*`` attributes (single-sourced; no re-literalled
        default) plus the injected flexure-solve callable + forcing seams."""
        ny, nx = self.grid_ny, self.grid_nx
        dx = self.Lx / (nx - 1)
        dy = self.Ly / (ny - 1)
        ibc = _ibc_from_border_status(self.boundary_status)
        bs = self.boundary_status
        out_idx = np.round(np.linspace(0, self.nt - 1, self.nt_out)).astype(int)
        law_code, gp = build_glacial_params(
            sliding_law=self.sliding_law, Ko=self.Ko, ce=self.ce,
            n=self.n, nu=self.nu, m=self.m, mu=self.mu, Ac=self.Ac,
            alpha_g=self.alpha_g, lambda_p=self.lambda_p,
            lambda_c=self.lambda_c, tau_c=self.tau_c,
            coulomb_clamp=self.coulomb_clamp, hc_over_H=self.hc_over_H,
            H_diffusivity=self.H_diffusivity)
        # border_bed_uplift: static (scalar/(ny,nx)) or a clock series (nt,ny,nx).
        bbu_spec = self._make_border_bed_uplift()
        if bbu_spec is None and self._driver_bbu_fn() is None:
            raise RuntimeError("border_bed_uplift resolved to None without a per-step "
                               "_driver_bbu_fn: a subclass must override both")
        if isinstance(bbu_spec, tuple):
            bbu_static, bbu_series = None, np.asarray(bbu_spec[1], dtype=float)
        else:
            bbu_static, bbu_series = bbu_spec, None
        # Flexure plate solve: the in-house scipy.fft solve (the injected seam).
        flexure_solve = _inhouse_flexure if self.flexure else None
        return SimpleNamespace(
            # grid
            ny=ny, nx=nx, dx=dx, dy=dy, cell_area=dx * dy,
            x=np.linspace(0, self.Lx, nx), y=np.linspace(0, self.Ly, ny),
            xl=self.Lx, yl=self.Ly, ibc=ibc, border_status=bs,
            wrap_x=(bs[0] == 'looped'), wrap_y=(bs[2] == 'looped'),
            # time / cadence
            nt=self.nt, nt_out=self.nt_out, t=self.t, out_idx=out_idx,
            t_out=self.t[out_idx], progress_bar=self.progress_bar,
            # mode / flags
            mode=self.mode, carve=self.carve_width,
            trunk_surface=self.trunk_surface, flexure=self.flexure,
            sediment=self.sediment_basin, sediment_edge=self.sediment_edge,
            save_closure_pair=self.save_closure_pair,
            # law record
            law_code=law_code, gp=gp, hc_over_H=float(gp.hc_over_H),
            alpha_g=float(gp.alpha_g), widening_factor=self.widening_factor,
            # physics knobs
            beta=self.beta, width_hack_k=self.width_hack_k,
            width_hack_p=self.width_hack_p, D=self.D,
            flotation_gate=self.flotation_gate, flotation_ramp=self.flotation_ramp,
            parallel_erode=self.parallel_erode,
            inner_passes=self.inner_passes, inner_relax=self.inner_relax,
            inner_damping=self.inner_damping,
            routing_filter_factor=self.routing_filter_factor,
            # mode D's valley <-> sheet devices: s from the channel/sheet reading
            regime_switch=self.regime_switch,
            hc_sheet=_constants.HC_SHEET,
            channel_bed_bar=self.channel_bed_bar,
            closure_flux=self.closure_flux,
            slope_normalise=self.slope_normalise,
            trunk_dip_k=self.trunk_dip_k,
            trunk_dip_floor=_constants.TRUNK_DIP_FLOOR,
            # flexure params
            lithos_density=self.lithos_density, asthen_density=self.asthen_density,
            e_thickness=self.e_thickness, ice_load=self.ice_load,
            flexure_solve=flexure_solve,
            # forcing series / scalars
            zELA_series=self._zELA_series,
            zELA=self._zELA_static,
            runoff_series=self._P_series, P=self.P,
            bl_series=self._bl_series, bl=self.bl, bl_sides=self._bl_sides,
            bbu_static=bbu_static, bbu_series=bbu_series,
            # injected seams
            uplift_fn=self._driver_uplift_fn(), bbu_fn=self._driver_bbu_fn(),
            initial_surface=self._driver_initial_surface(),
            route=None,   # set inside the InhouseRouter context in _run_inhouse
        )

    def _unpack_outputs(self):
        """Populate the familiar (z_out, H_out, …) attributes from self.ds_out.
        Shared by run() and load() so they produce identical post-run state.

        The (zb_out, z_out) reconstruction is mode-aware (keyed on whether the
        run stored a separate bedrock_surface output):

        - Citizen Mode B / C (GlacialSPLModeB, GlacialSPLModeC): topography__elevation
          IS the bed, so ``zb_out = topography__elevation`` and the ice surface is
          the reconstruction ``z_out = zb_out + hc_over_H*H_out`` — or, when the
          run carries mode D's per-cell datum
          (``hc_cell_out``, ``glacial_spl__hc_cell``), ``zb_out + hc_cell_out*H_out``:
          the surface the model balanced on. No flicker: zb
          is a clean tracked state; only the derived display surface can jump.
        - Mode A: topography is the ice surface (``z_out``); the bed is the
          kernel-committed ``bedrock_surface``. (Runs saved under the retired
          surface-replace mode-B+carve class also land here — they stored a
          bedrock_surface — so they still reload correctly.)
        """
        self.H_out = self.ds_out['glacial_spl__ice_thickness'].values  # (time, y, x)
        topo = self.ds_out['topography__elevation'].values             # (time, y, x)
        hc = float(self.hc_over_H)
        # The per-cell datum (mode D); None = the scalar hc.
        self.hc_cell_out = (self.ds_out['glacial_spl__hc_cell'].values
                            if 'glacial_spl__hc_cell' in self.ds_out else None)
        # Mode D's binary class (True = channel); None outside mode D and for
        # mode-D runs saved before the class was stored.
        self.channel_out = (self.ds_out['glacial_spl__channel'].values.astype(bool)
                            if 'glacial_spl__channel' in self.ds_out else None)
        if 'glacial_spl__bedrock_surface' in self.ds_out:
            # Mode A (or a legacy surface-replace run): topography is the surface,
            # bed is stored. Prefer the kernel-committed bedrock_surface —
            # topography and the ice_thickness out-variable aren't guaranteed to
            # pair within a frame, so a derived z - hc*H mixes steps and JUMPS by
            # ~hc*H at flicker cells while the stored bed stays physical.
            self.z_out = topo
            self.zb_out = self.ds_out['glacial_spl__bedrock_surface'].values
        elif self._citizen_mode_b:
            # Citizen Mode B / C: topography IS the bed; reconstruct the ice
            # surface on the datum the model balanced on (per cell in mode D).
            self.zb_out = topo
            hc_field = hc if self.hc_cell_out is None else self.hc_cell_out
            self.z_out = self.zb_out + hc_field * self.H_out
        else:
            # Fallback (runs saved before the bed was stored): derive the bed from
            # the surface under the CURRENT convention.
            self.z_out = topo
            self.zb_out = topo - hc * self.H_out
        self.area_out = self.ds_out['glacial_flow__area'].values           # (time, y, x)
        self.receivers_out = self.ds_out['glacial_flow__receivers_2d'].values  # (time, y, x)
        self.output_times = self.ds_out.time.values

        if self._zELA_series is not None:
            out_idx = np.round(np.linspace(0, self.nt - 1, len(self.output_times))).astype(int)
            self._zELA_output = self._zELA_series[out_idx]
            if self._zELA_output.ndim == 3:        # a field series: its spatial mean
                self._zELA_output = self._zELA_output.mean(axis=(1, 2))
        else:
            self._zELA_output = np.full(len(self.output_times), self.zELA)

        self.Qg_out = self.ds_out['glacial_flow__ice_flux'].values         # (time, y, x)
        # The closure's own (flux, thickness), stored only where it differs
        # from (Qg_out, H_out): a transect closure_flux / inner_relax < 1.
        # None otherwise (and on runs saved before the rows existed).
        self.closure_Qg_out = (self.ds_out['glacial_spl__closure_ice_flux'].values
                               if 'glacial_spl__closure_ice_flux' in self.ds_out
                               else None)
        self.closure_H_out = (self.ds_out['glacial_spl__closure_ice_thickness'].values
                              if 'glacial_spl__closure_ice_thickness' in self.ds_out
                              else None)
        self.Qf_out = self.ds_out['glacial_flow__water_flux'].values       # (time, y, x)
        self.basin_out = self.ds_out['glacial_flow__basin_ids'].values     # (time, y, x)
        self.stack_out = self.ds_out['glacial_flow__stack_2d'].values          # (time, y, x)
        self.erosion_rate_out = self.ds_out['glacial_spl__erosion_rate'].values # (time, y, x)
        # Per-step rock denudation (delta-zb in mode B incl. carve, delta-zs in
        # mode A) — the sediment / flexural-unloading source. Guarded for back-
        # compat with saved runs that predate the variable.
        if 'glacial_spl__denudation' in self.ds_out:
            self.denudation_out = self.ds_out['glacial_spl__denudation'].values
        self.lengths_out = self._reconstruct_lengths_out(self.receivers_out)

        if self.sediment_basin and 'sediment__cumulative' in self.ds_out:
            # (time, y, x): per-node upstream-eroded throughput per step, and its
            # running time-integral. diff eroded_volume_out along axis 0 for
            # per-interval volumes; outlet node = whole-basin cumulative yield.
            self.sediment_flux_out = self.ds_out['sediment__flux'].values
            self.eroded_volume_out = self.ds_out['sediment__cumulative'].values

        if self.sediment_edge and 'sediment__edge_cumulative' in self.ds_out:
            # (time, side) in _BL_SIDES order [left, right, bottom, top]: the
            # volume (m^3) delivered across each domain edge per step and its
            # running time-integral — the border ring of sediment__flux summed
            # per side, each corner counted once. NaN on a side whose
            # boundary_status is not 'fixed_value' (no outlet there, which is
            # not the same as zero export).
            self.sediment_edge_flux_out = self.ds_out['sediment__edge_flux'].values
            self.sediment_edge_cumulative_out = self.ds_out['sediment__edge_cumulative'].values

        if self.flexure and 'flexure__rebound' in self.ds_out:
            # (time, y, x): per-step flexural deflection applied to the column
            # (negative = subsidence under load, positive = isostatic rebound).
            self.rebound_out = self.ds_out['flexure__rebound'].values

    def save(self, filename=None):
        """Atomically pickle the current model state (user_params + ds_out) to
        ``./model_outputs/saved_models/`` under the current working directory.
        With no filename, an auto-name combining a UTC timestamp and ``run_id``
        is generated. Returns the path written."""
        if not hasattr(self, 'ds_out'):
            raise RuntimeError("Nothing to save — call model.run() first.")

        if filename is None:
            stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
            filename = f"{stamp}_{self.run_id}.pkl"
        elif not str(filename).endswith('.pkl'):
            filename = f"{filename}.pkl"

        path = Path(output_path(filename, 'saved_models'))
        state = {
            'save_format': _SAVE_FORMAT,
            'schema_version': _SAVE_SCHEMA_VERSION,
            'siim_version': _SIIM_VERSION,
            'model_identity': _model_identity(type(self)),
            '_user_params': self._user_params,
            'ds_out': self.ds_out,
        }

        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                    mode='wb', dir=path.parent, prefix=f'.{path.name}.',
                    suffix='.tmp', delete=False) as f:
                temp_path = Path(f.name)
                pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp_path, path)
        except BaseException:
            if temp_path is not None:
                try:
                    temp_path.unlink(missing_ok=True)
                except OSError:  # gate: cleanup failure must not mask the original save error.
                    pass
            raise
        print(f"Saved to {path}")
        return path

    @classmethod
    def load(cls, filename):
        """Rebuild a siim instance from a previously ``save()``-ed pickle.
        ``filename`` may be a name under ``./model_outputs/saved_models/``
        (with or without the ``.pkl`` suffix; a subfolder such as
        ``'batch/run003'`` round-trips with :meth:`save`) or an absolute /
        relative path.

        Pickle files can execute code while loading. Only load files from a
        trusted source. Unversioned legacy payloads are rejected rather than
        interpreted under current model conventions.
        """
        path = Path(filename)
        if not path.is_file():
            candidate = Path(output_path(str(path), 'saved_models'))
            if not candidate.is_file() and not str(candidate).endswith('.pkl'):
                # append, as save() does: with_suffix would turn 'sweep_nu1.5'
                # into 'sweep_nu1.pkl', another run
                candidate = candidate.with_name(candidate.name + '.pkl')
            path = candidate
        try:
            with open(path, 'rb') as f:
                state = pickle.load(f)
        except (pickle.UnpicklingError, EOFError, AttributeError, ImportError,
                IndexError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Could not load SIIM saved model {path}: the pickle is invalid "
                "or incompatible with this Python environment.") from exc
        _validate_saved_state(state, cls)
        if state['siim_version'] != _SIIM_VERSION:
            warnings.warn(
                f"{path} was saved by siim {state['siim_version']} and is "
                f"loaded by siim {_SIIM_VERSION}: its parameters are stored as "
                "passed, so defaults such as the router-derived trunk_dip_k / "
                "slope_normalise are re-resolved from this version's constants.",
                RuntimeWarning, stacklevel=2)
        params = dict(state['_user_params'])
        # a falsy value was the knob's off setting: dropping it changes nothing
        dropped = [k for k in _REMOVED_PARAMS if params.pop(k, None)]
        if dropped:
            warnings.warn(
                f"{path} was run with {', '.join(dropped)}, which this version "
                "removed: its stored outputs load unchanged, but the "
                "reconstructed model (and any re-run) takes this version's "
                "defaults in its place (modes B/C/D: inner_damping="
                f"{_constants.MODE_BCD_INNER_DAMPING!r}).", RuntimeWarning, stacklevel=2)
        model = cls(params)
        model.ds_out = state['ds_out']
        model._unpack_outputs()
        print(f"Loaded from {path}")
        return model

    def _reconstruct_lengths_out(self, receivers_out):
        """Reconstruct per-timestep node-to-receiver distances from the saved
        receivers array and grid geometry. Avoids saving a redundant (time, y, x)
        float array through xsimlab on every output step.

        Each node's length to its receiver is determined by the (Δy, Δx) offset
        between node and receiver in grid coordinates, scaled by dx, dy.
        Boundary / outlet nodes (receiver == self, or NaN after zarr round-trip)
        get length 0 by convention.
        """
        nt, ny, nx = receivers_out.shape
        dx = self.Lx / (nx - 1)
        dy = self.Ly / (ny - 1)
        flat_idx = np.arange(ny * nx)
        iy, ix = np.divmod(flat_idx, nx)

        recs_raw = receivers_out.reshape(nt, -1)
        # Zarr round-trip may return float64 with NaN for boundary nodes; clean before int cast
        nan_mask = np.isnan(recs_raw) if recs_raw.dtype.kind == 'f' else None
        recs_flat = np.nan_to_num(recs_raw, nan=0.0).astype(np.int64) if nan_mask is not None else recs_raw.astype(np.int64, copy=False)

        ry, rx = np.divmod(recs_flat, nx)
        # Flow always routes to an immediately adjacent cell, so the true
        # per-axis index offset is 0 or 1. A larger magnitude means the step
        # crossed a periodic ('looped') boundary (receiver on the opposite
        # edge); the true geometric step there is still one cell. Clamp the
        # absolute offset to <=1 so periodic crossings don't inject a ~domain-
        # width jump into the cumulative channel distance.
        dy_cells = np.minimum(np.abs(iy[None, :] - ry), 1)
        dx_cells = np.minimum(np.abs(ix[None, :] - rx), 1)
        dy_dist = dy_cells * dy
        dx_dist = dx_cells * dx
        lengths = np.sqrt(dx_dist * dx_dist + dy_dist * dy_dist)

        # Self-receiving and NaN-receiver nodes (boundary / outlets) have length 0
        lengths[recs_flat == flat_idx[None, :]] = 0.0
        if nan_mask is not None:
            lengths[nan_mask] = 0.0
        return lengths.reshape(nt, ny, nx)

    def ice_speed(self, i=-1):
        """Ice surface speed and cross-section mean speed (m/yr) at output i.

        Recomputed by :func:`siim._core.solvers.ice_speeds` from the flux and
        thickness the kernel's closure solved at output i: ``closure_Qg_out``
        / ``closure_H_out`` where the run stores them (``save_closure_pair=True``
        under a transect ``closure_flux``, the mode B/C/D default / ``inner_relax
        < 1``), ``Qg_out`` / ``H_out`` otherwise. No speed is stored.
        Returns ``(us, U)``, each shaped ``(grid_ny, grid_nx)``, zero where
        the closure carried no ice (under ``inner_relax < 1`` a blended
        ``H_out`` can still be positive there) and at outlet cells (whose H
        is copied from their donor, not solved). Under 'eff-exp' ``us = U``
        (no sliding split). Approximate, with a warning, in two cases: ``H_diffusivity`` smooths
        the kernel's H after the closure solve, and a run without the
        closure rows (``save_closure_pair=False``, the default) lacks the pair its
        settings need.

        Every icy cell is a channel of width ``W = alpha_g*H``, so thick
        interior ice that covers its cell as a sheet (``W > dx``) reads slow:
        the same flux through the cell width would move about ``W/dx`` times
        faster.
        Under 'coulomb', thin ice near the yield stress ``tau_c`` can show
        very large speeds at a few cells.
        """
        if self.H_diffusivity:
            warnings.warn("H_diffusivity smoothed H after the closure solve; "
                          "ice_speed is approximate for this run", stacklevel=2)
        _outputs.warn_if_closure_pair_missing(self)
        H, Q = _outputs.closure_pair(self, i)
        return ice_speeds(H, Q, self.sliding_law,
                          self.alpha_g, self.Ac, self.lambda_p,
                          self.lambda_c, self.tau_c,
                          receivers=self.receivers_out[i])

    def extract_channel(self, i=-1, basin_rank=0):
        """Extract the main channel from a basin at output time step i,
        then follow those same nodes through all output time steps.

        Parameters
        ----------
        i : int
            Reference time step for channel extraction (default -1, last step).
        basin_rank : int
            Which basin to extract from, ranked by node count.
            0 = largest (default), 1 = second largest, etc.

        Returns a SimpleNamespace with:
            nodes    — flat node indices along the channel (headwater → outlet)
            x_coord  — x coordinates (m) along the channel
            y_coord  — y coordinates (m) along the channel
            distance — cumulative distance from headwater (m)
            z, zb, H, Qg, Qf, area, erosion_rate — (nt_out, n_nodes) arrays
            k_h, d, xo, L — Hack's law fit from time step i
        """
        rec    = _sanitized_receivers(self.receivers_out[i])
        area   = self.area_out[i].flatten()
        basins = self.basin_out[i].flatten()

        # 1. Select basin by rank (0 = largest, 1 = second largest, ...)
        unique, counts = np.unique(basins, return_counts=True)
        rank_order = np.argsort(-counts)
        if basin_rank >= len(rank_order):
            raise ValueError(f"basin_rank={basin_rank} but only {len(rank_order)} basins exist")
        target_id = unique[rank_order[basin_rank]]
        basin_mask = basins == target_id
        basin_nodes = np.where(basin_mask)[0]

        # 2. Outlet = max area node in that basin
        outlet = basin_nodes[np.argmax(area[basin_nodes])]

        # 3. Build donor lookup (one pass over rec), then walk upstream
        donor_list = _donor_lists(rec)

        path = [outlet]
        node = outlet
        while True:
            donors = donor_list[node]
            if len(donors) == 0:
                break
            best = max(donors, key=lambda d: area[d])
            path.append(best)
            node = best

        # Reverse: headwater → outlet (matches 1D convention)
        nodes = np.array(path[::-1])

        # 4. Node coordinates on the grid
        ny, nx = self.grid_ny, self.grid_nx
        dx = self.Lx / (nx - 1)
        dy = self.Ly / (ny - 1)
        row, col = np.divmod(nodes, nx)
        xc = col * dx
        yc = row * dy

        # 5. Cumulative distance from headwater (from reference time step)
        lengths = self.lengths_out[i].flatten()
        dist = np.zeros(len(nodes))
        for j in range(1, len(nodes)):
            dist[j] = dist[j - 1] + lengths[nodes[j - 1]]

        nt_out = len(self.output_times)
        z  = np.array([self.z_out[t].flatten()[nodes] for t in range(nt_out)])
        zb = np.array([self.zb_out[t].flatten()[nodes] for t in range(nt_out)])
        H  = np.array([self.H_out[t].flatten()[nodes] for t in range(nt_out)])
        Qg = np.array([self.Qg_out[t].flatten()[nodes] for t in range(nt_out)])
        Qf = np.array([self.Qf_out[t].flatten()[nodes] for t in range(nt_out)])
        ar = np.array([self.area_out[t].flatten()[nodes] for t in range(nt_out)])
        er = np.array([self.erosion_rate_out[t].flatten()[nodes] for t in range(nt_out)])

        # 8. Fit Hack's law from reference time step i
        ar_ref = ar[i]
        valid = dist > 0
        log_x = np.log(dist[valid])
        log_a = np.log(ar_ref[valid])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", _RankWarning)
            d_fit, log_kh = np.polyfit(log_x, log_a, 1)
        k_h_fit = np.exp(log_kh)

        # 9. Update analytical model from reference time step
        channel_length = dist[-1]
        xo_fit = (ar_ref[0] / k_h_fit) ** (1.0 / d_fit)
        self._set_analytical_grid(channel_length, xo_fit, k_h_fit, d_fit)

        return SimpleNamespace(
            nodes=nodes, distance=dist,
            x_coord=xc, y_coord=yc,
            z=z, zb=zb, H=H,
            Qg=Qg, Qf=Qf, area=ar,
            erosion_rate=er,
            k_h=k_h_fit, d=d_fit, xo=xo_fit, L=channel_length,
        )

    def strahler_order(self, i=-1, channel_threshold=1e3):
        """Compute Strahler stream order over the full grid, threshold-gated.

        Parameters
        ----------
        i : int
            Output time step index (default -1, last step).
        channel_threshold : float
            Minimum upstream area in cell units to be considered a channel node.
            Nodes below this threshold get order 0.

        Returns
        -------
        SimpleNamespace with:
            order : ndarray, shape (grid_ny, grid_nx)
                Strahler order at each node (0 for sub-threshold non-channel nodes).
            area : ndarray, shape (grid_ny, grid_nx)
                Upstream catchment area (m²) at each node.
            area_by_order : dict
                Mean-area member arrays keyed by Strahler order (1..max_order).
            max_order : int
                Highest Strahler order present.
            sigma : ndarray
                Per-order area ratios log(2)/log(A_{k+1}/A_k) (Horton scaling).

        Notes
        -----
        Works correctly with flow_routing='single' and 'dinf' (SFR-equivalent
        donor topology in both cases — D-inf's primary-receiver collapse is
        meaningful because facet-weighted routing picks a clear dominant
        direction). Under ``'mfd'`` it is a PRIMARY-PATH diagnostic: the walk
        follows the largest-weight receiver only, which at p = 1 carries well
        under half of the cell's flux, so the orders/ratios describe the
        dominant network, not the full MFD graph (see
        ``docs/guides/outputs_and_io.md``).
        """
        ny, nx = self.grid_ny, self.grid_nx
        rec = _sanitized_receivers(self.receivers_out[i])
        stack = np.nan_to_num(self.stack_out[i].flatten(), nan=0).astype(int)
        area = self.area_out[i].flatten()
        nn = rec.size
        donor_list = _donor_lists(rec)

        # Walk upstream → downstream. Stack convention differs by router:
        #   SFR         stores stack outlet-first → walk stack[::-1] (head→outlet)
        #   D-inf / MFD store stack donor-first   → walk stack       (head→outlet)
        # Without this dispatch, a donor-first router walks outlet→head and every
        # donor is unvisited at lookup time, so every channel cell collapses to
        # order=1, max_order=1, A has length 1, sigma becomes an empty array,
        # and downstream np.mean(sigma) returns NaN. Router identity — not
        # arity — selects here, so EVERY donor-first router must be listed.
        walk_order = (stack if self.flow_routing in _DONOR_FIRST_ROUTERS
                      else stack[::-1])
        order = np.zeros(nn, dtype=int)
        for j in walk_order:
            if area[j] < channel_threshold:
                continue
            # Collect orders of channel donors
            donor_orders = [order[d] for d in donor_list[j] if order[d] > 0]
            if len(donor_orders) == 0:
                order[j] = 1
            else:
                max_ord = max(donor_orders)
                count_max = donor_orders.count(max_ord)
                order[j] = max_ord + 1 if count_max >= 2 else max_ord

        max_order = int(order.max())
        area_by_order = {k: area[order == k] for k in range(1, max_order + 1)}
        A = np.array([area_by_order[k].mean() for k in range(1, max_order + 1)])
        sigma = np.log(2) / np.log(A[1:] / A[:-1])
        return SimpleNamespace(order=order.reshape(ny, nx), area=area.reshape(ny, nx),
                               area_by_order=area_by_order, max_order=max_order,
                               sigma=sigma)
