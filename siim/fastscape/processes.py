"""
xsimlab process classes for siim2d.

Three processes plug into fastscape's ``basic_model``:

  - ``InitialTopography`` replaces ``FlatSurface`` for the initial elevation
    field (precomputed elevation + zeroed-on-fixed-edges noise).
  - ``GlacialFlowAccumulator`` extends ``FlowAccumulator`` to track water flux,
    ice flux, drainage area, basin IDs, and the per-step routing topology
    (receivers_2d, stack_2d) needed by downstream extract_channel /
    extract_basin / strahler_order code in siim2d.py.
  - ``GlacialSPLModeA`` / ``GlacialSPLModeB`` / ``GlacialSPLModeC`` (all
    ``GlacialSPLBase`` subclasses) plug into the model's erosion slot, one per
    surface-evolution mode (build-time selection): A tracks the ice surface, B
    the bed + ice thickness (citizen), C = B plus the sub-grid width carve. They
    dispatch on (sliding_law, routing) and call the corresponding ``law_code``
    step skeleton in :mod:`siim._core.skeleton`.

  - ``DinfFlowRouter`` overrides fastscape's ``FlowRouter`` with Tarboton
    (1997) D-infinity flow directions (defined at the bottom of this module).

The numba flow-accumulation + D-infinity routing primitives live in
:mod:`siim._core.routing`; the ice-thickness + erosion solvers in
:mod:`siim._core` (``solvers`` / ``eroders`` / ``skeleton``).
"""

import numpy as np
import xsimlab as xs
from fastscape.processes import (
    BlockUplift,
    FlowAccumulator, FlowRouter, RasterGrid2D, SurfaceToErode,
    SurfaceAfterTectonics, SurfaceTopography, BorderBoundary, Flexure,
    TectonicForcing,
)

from ..constants import GRAVITY, RHO_ICE
from .. import constants as _constants
from .._core.routing import (
    _flow_accumulate_sd, _flow_accumulate_sd_2,
    _flow_accumulate_dinf, _flow_accumulate_dinf_2,
    _priority_flood_eps, _dinf_route, _dinf_topo_stack, _dinf_pack,
    _DINF_E1_DJ, _DINF_E1_DI, _DINF_E2_DJ, _DINF_E2_DI,
)
from .._core.skeleton import (
    _glac_fast_solve_modeA_sfr, _glac_fast_solve_modeA_dinf,
    _glac_fast_solve_modeB_sfr, _glac_fast_solve_modeB_dinf,
)
from .._core.params import GlacialParams
from .._core.solvers import (
    LAW_EFFEXP, LAW_POWER, LAW_COULOMB,
    _modeb_closure,
)
from .._core.carve import (
    _power_dt_2d, _power_dt_2d_periodic, _carve_offsets, _carve_subgrid_width,
)


def _scalar_dt(dt):
    """Coerce xsimlab's ``step_delta`` to a plain float. The driver passes a
    shape-(1,) array (xarray stopped squeezing the size-1 clock dim), and
    ``float(shape-(1,) array)`` is a numpy DeprecationWarning today and a hard
    TypeError under numpy >= 2 (audit m1). Call once at the top of each
    ``run_step`` and pass the plain float down."""
    return float(np.asarray(dt).ravel()[0])


@xs.process
class InitialTopography:
    """Initialize surface topography from a pre-computed elevation array plus
    uniform tie-breaking noise. Noise is zeroed on 'fixed_value' edges so the
    boundary z stays exactly at the elevation_init value across the run
    (BlockUplift and the glacial erosion process both skip fixed cells; without
    this any IC noise there gets locked in and drifts the base level)."""

    seed = xs.variable(default=None, description='Random seed')
    elevation_init = xs.variable(dims=('y', 'x'), description='Pre-computed initial elevation (m)')
    noise_amplitude = xs.variable(
        default=None,
        description='Uniform-noise amplitude (m) added to elevation_init for D8 '
                    'tie-breaking. Default None -> 0.1 * max(elevation_init).'
    )
    shape = xs.foreign(RasterGrid2D, 'shape')
    elevation = xs.foreign(SurfaceTopography, 'elevation', intent='out')
    border_status = xs.foreign(BorderBoundary, 'border_status')

    def initialize(self):
        if self.seed is not None:
            seed = None if np.isnan(float(self.seed)) else int(self.seed)
        else:
            seed = None
        rs = np.random.RandomState(seed=seed)
        if self.noise_amplitude is None:
            noise_scale = 0.1 * np.max(self.elevation_init)
        else:
            noise_scale = float(self.noise_amplitude)
        noise = noise_scale * rs.rand(*self.shape)
        bs = list(self.border_status)
        if bs[0] == "fixed_value": noise[:,  0] = 0.0
        if bs[1] == "fixed_value": noise[:, -1] = 0.0
        if bs[2] == "fixed_value": noise[0,  :] = 0.0
        if bs[3] == "fixed_value": noise[-1, :] = 0.0
        self.elevation = self.elevation_init + noise


@xs.process
class GlacialBlockUplift(BlockUplift):
    """BlockUplift that tolerates a time-varying ``(nt, ny, nx)`` uplift rate.

    xsimlab slices the clock axis of a ``(('tstep', 'y', 'x'), array)`` input
    per step, but xarray no longer squeezes groupby, so the per-step ``rate``
    keeps a leading size-1 dim and stock ``BlockUplift``'s
    ``np.broadcast_to((1, ny, nx), (ny, nx))`` raises. Drop that leading dim
    before broadcasting; scalar / ``(y, x)`` rates are unaffected."""

    @xs.runtime(args="step_delta")
    def run_step(self, dt):
        rate = np.asarray(self.rate)
        if rate.ndim == 3:
            rate = rate[0]
        rate = np.broadcast_to(rate, self.shape) * self._mask
        self.uplift = rate * dt


@xs.process
class GlacialLaw:
    """Raw sliding-law inputs + the derived-constant algebra, hoisted out of
    the glacial erosion process (:class:`GlacialSPLBase`).

    Owns the per-run physical/law scalars (sliding law, erosion coefficients,
    ice-rheology parameters, the channel-floor ratio ``hc_over_H``, the channel
    aspect ratio ``alpha_g``) and turns them into the frozen
    ``(law_code, GlacialParams)`` record the ``law_code`` step skeletons consume
    (:mod:`siim._core.skeleton`). Exposed as ``params`` so the erosion process
    (and any sibling) reads one record by foreign instead of re-deriving the
    constants. ``hc_over_H`` is the centerline/mean channel-depth ratio (default
    ``constants.HC_OVER_H``); the surfaces built from ``(zb, H)`` state are
    ``zs = zb + hc_over_H * H`` (the tracked bed is the sub-grid channel floor,
    ``H`` the width-mean depth).
    """

    # --- Sliding law selection ---
    sliding_law = xs.variable(default=_constants.DEFAULT_SLIDING_LAW, description="'eff-exp', 'power', or 'coulomb'")

    # --- Erosion parameters ---
    Ko = xs.variable(default=_constants.KO,  description='Fluvial erodibility (m^(1-3m)/yr)')
    ce = xs.variable(default=_constants.CE,  description='Glacial erosion coefficient')
    n  = xs.variable(default=_constants.N_FLUVIAL,     description='Fluvial slope exponent')
    nu = xs.variable(default=_constants.NU,     description='Glacial slope exponent')
    m  = xs.variable(default=None,  description='Fluvial area exponent (default n/2)')
    mu = xs.variable(default=None,  description='Glacial area exponent (default derived from nu: coulomb nu/2, power/eff-exp 4*nu/15)')

    # --- Ice rheology parameters ---
    Ac       = xs.variable(default=_constants.AC, description="Glen's flow-law coefficient A (deformation prefactor 2A/5 applied internally)")
    alpha_g  = xs.variable(default=_constants.ALPHA_G, description='Glacial channel aspect ratio')
    lambda_p = xs.variable(default=_constants.LAMBDA_P, description='Critical ice thickness for eff-exp/power laws (m)')
    lambda_c = xs.variable(default=None,    description='Coulomb sliding length (default constants.LAMBDA_C)')
    tau_c    = xs.variable(default=_constants.TAU_C,     description='Upper-bound shear stress for coulomb law (Pa)')
    coulomb_clamp = xs.variable(default=_constants.COULOMB_CLAMP, description='Minimum relative gap from the pole for the coulomb H-solver')
    hc_over_H = xs.variable(default=_constants.HC_OVER_H, description='Centerline-to-mean channel-depth ratio: surfaces built from (zb, H) state are zs = zb + hc_over_H * H (the tracked bed is the channel floor, H the width-mean depth). Default constants.HC_OVER_H.')
    H_diffusivity = xs.variable(
        default=None,
        description="Explicit H diffusivity for mode B (m^2/yr); None disables")

    # The (law_code, GlacialParams) record the skeletons consume. any_object is
    # always intent=out (no intent kwarg); the erosion process reads it by foreign.
    params = xs.any_object(
        description="(law_code, GlacialParams) record for the active sliding law "
                    "— the frozen per-run scalars the law_code skeletons consume.")

    def initialize(self):
        # Validate the sliding law on the standalone fastscape surface (siim1d
        # already raises; an unrecognized string silently fell through to
        # eff-exp here — audit m14).
        if self.sliding_law not in ('eff-exp', 'power', 'coulomb'):
            raise ValueError(
                f"Unknown sliding_law: '{self.sliding_law}'. "
                f"Options: 'eff-exp', 'power', 'coulomb'")
        self._lambda_p = float(self.lambda_p)
        self._lambda_c = float(self.lambda_c) if self.lambda_c is not None else _constants.LAMBDA_C
        self._tau_c    = float(self.tau_c)
        self._coulomb_clamp = float(self.coulomb_clamp)
        self._rho_g = RHO_ICE
        self._g     = GRAVITY
        self._rho_g_g = self._rho_g * self._g
        # cg via the single-source rheology helper (kt absorbed; audit m36).
        self._cg = _constants.cg_prefactor(float(self.alpha_g), float(self.Ac),
                                           self._rho_g, self._g)
        # mu: the siim2d wrapper always passes a per-law derived value; standalone
        # use falls back to the SAME per-law relations via constants.derive_*
        # (single-sourced; audit m35). Co (eff-exp solver) via Co_power with the
        # effective mu — an explicit override wins, and Co tracks it (B5).
        if self.mu is not None:
            self._mu = float(self.mu)
        elif self.sliding_law == 'coulomb':
            self._mu = _constants.derive_coulomb(
                float(self.ce), float(self.alpha_g), self._tau_c,
                self._rho_g, self._g, nu=float(self.nu)).mu
        else:  # power, eff-exp
            self._mu = _constants.derive_power(
                float(self.ce), self._cg, self._lambda_p,
                float(self.alpha_g), nu=float(self.nu)).mu
        self._Co = _constants.Co_power(float(self.ce), self._cg,
                                       self._lambda_p, float(self.alpha_g),
                                       self._mu)
        self._D_H = (0.0 if self.H_diffusivity is None
                     else float(self.H_diffusivity))
        # Centerline-to-mean depth ratio: surfaces built from (zb, H) state are
        # zs = zb + hc_over_H * H.
        self._hc_over_H = float(self.hc_over_H)
        if not self._hc_over_H > 0.0:
            raise ValueError(
                f"hc_over_H must be > 0, got {self.hc_over_H!r}")
        self.params = self._glacial_params_and_code()

    def _glacial_params_and_code(self):
        """(law_code, GlacialParams) for the current sliding law. Mirrors the
        old per-law wrapper arg lists: each law fills its own constants and 0.0
        for the others (the inactive fields reach only the unused skeleton
        dispatch branch). hc_over_H/D_H are always real (mode A ignores them)."""
        Ko, n, nu = float(self.Ko), float(self.n), float(self.nu)
        m = float(self.m) if self.m is not None else n / 2.0   # default n/2 (matches siim1d/2d)
        cg, alpha_g = self._cg, float(self.alpha_g)
        hc_over_H, D_H = self._hc_over_H, self._D_H
        if self.sliding_law == 'power':
            return LAW_POWER, GlacialParams(
                Ko=Ko, ce=float(self.ce), n=n, nu=nu, m=m, cg=cg,
                alpha_g=alpha_g, lambda_p=self._lambda_p,
                hc_over_H=hc_over_H, D_H=D_H)
        elif self.sliding_law == 'coulomb':
            return LAW_COULOMB, GlacialParams(
                Ko=Ko, ce=float(self.ce), n=n, nu=nu, m=m, cg=cg,
                alpha_g=alpha_g, lambda_c=self._lambda_c, tau_c=self._tau_c,
                coulomb_clamp=self._coulomb_clamp, rho_g_g=self._rho_g_g,
                hc_over_H=hc_over_H, D_H=D_H)
        return LAW_EFFEXP, GlacialParams(
            Ko=Ko, Co=self._Co, n=n, nu=nu, m=m, mu=self._mu, cg=cg,
            alpha_g=alpha_g, lambda_p=self._lambda_p,
            hc_over_H=hc_over_H, D_H=D_H)


@xs.process
class GlacialFlowAccumulator(FlowAccumulator):
    """Accumulates water flux, ice flux, and drainage area through the flow graph.

    Sub-grid mass balance: ablation below the ELA scales with the glacier's
    plan-view area (width × flow length) wherever that exceeds cell_area.
    Width is set by Hack-style drainage-area scaling
    (:cite:t:`hackStudiesLongitudinalStream1957`):

        glacier_width = width_hack_k * upstream_area ** width_hack_p

    This decouples width from instantaneous ice thickness, so the kinematic-
    wave mode at the toe doesn't feed back through ablation. Matches the
    strictly-elevation-driven mass balance of the 1D analytical SS.
    """
    beta = xs.variable(description='Ice accumulation lapse rate (m/yr per m elevation)')
    zELA = xs.variable(dims=[(), ('y', 'x')], description='Equilibrium line altitude (m)')
    # alpha_g is single-sourced on GlacialLaw (the SSOT); read it by foreign so
    # the wrapper sets only law__alpha_g.
    alpha_g       = xs.foreign(GlacialLaw, 'alpha_g', intent='in')
    width_hack_k = xs.variable(
        default=_constants.WIDTH_HACK_K,
        description="Glacier-width prefactor: glacier_width = k * A^p [m], "
                    "with A in m^2 (default constants.WIDTH_HACK_K).")
    width_hack_p = xs.variable(
        default=_constants.WIDTH_HACK_P,
        description="Glacier-width Hack-law exponent: glacier_width = k * A^p "
                    "(default constants.WIDTH_HACK_P).")
    # (No ice_thickness global_ref: glacier width is Hack-area-based, so the
    # accumulator never reads ice thickness; the declaration was dead — audit m25.)
    # Post-tectonics surface (SurfaceToErode -> SurfaceAfterTectonics in the
    # model): the same elevation the flow router stacks on. Routing + the
    # per-cell glacier-area lengths come from this post-uplift surface; the
    # ELA-relative mass balance b(z) is evaluated one increment lower, on the
    # PRE-uplift climate surface z_clim = surface - surface_upward (below).
    surface     = xs.foreign(SurfaceToErode, 'elevation')
    # Per-step surface uplift (BlockUplift.uplift = rate*dt, zeroed at
    # 'fixed_value' borders by BlockUplift's mask). Subtracting it recovers the
    # surface the ice experienced at step start: mode A surf2erode = zs+uplift
    # -> z_clim = zs; citizen mode B surf2erode = (zb+uplift)+hc_over_H*H ->
    # z_clim = zb+hc_over_H*H (pre-uplift ice surface). Uniform across modes.
    # The mass balance belongs at z_clim so it carries no O(U*dt) bias from
    # climate-after-uplift; slopes (routing/erosion) are ~uplift-invariant and
    # stay on the post-uplift surface.
    surface_upward = xs.foreign(TectonicForcing, 'surface_upward', intent='in')
    ice_flux    = xs.variable(dims=('y', 'x'), intent='out', description='Accumulated ice flux (m^3/yr)')
    water_flux  = xs.variable(dims=('y', 'x'), intent='out', description='Accumulated water flux (m^3/yr)')
    area        = xs.variable(dims=('y', 'x'), intent='out', description='Upstream drainage area (m^2)')
    basin_ids    = xs.variable(dims=('y', 'x'), intent='out', description='Basin IDs')
    receivers_2d = xs.variable(dims=('y', 'x'), intent='out', description='Receiver indices (flat), reshaped to grid')
    stack_2d     = xs.variable(dims=('y', 'x'), intent='out', description='Stack order (flat), reshaped to grid')
    _basin_foreign = xs.foreign(FlowRouter, 'basin')
    _lengths_foreign = xs.foreign(FlowRouter, 'lengths')

    def run_step(self):
        zELA  = np.broadcast_to(self.zELA, self.shape)
        field = np.broadcast_to(self.runoff * self.cell_area, self.shape)
        # Climate (pre-uplift) surface: the ice surface at step start, before
        # this step's tectonic increment. The ELA-relative mass balance b(z) is
        # evaluated here (not on the post-uplift `surface`) so it carries no
        # O(U*dt) climate-after-uplift bias. At U=0 (or on fixed-value borders)
        # surface_upward == 0 -> z_clim == surface, bit-for-bit identical.
        z_clim = self.surface - np.broadcast_to(self.surface_upward, self.shape)

        # D-inf outputs are 2D — receivers, lengths, weights are all
        # (n_nodes, nb_rec_max). SFR outputs are 1D. Dispatch on ndim.
        is_dinf = self.receivers.ndim == 2
        if is_dinf:
            nb_rec = np.asarray(self.nb_receivers, dtype=np.int64)
            recs   = np.asarray(self.receivers,   dtype=np.int64)
            wts    = np.asarray(self.weights,     dtype=np.float64)

        # 1. Drainage area — accumulated once and reused for both glacier_width
        #    and self.area output.
        field_area = np.broadcast_to(self.cell_area, self.shape).flatten().copy()
        if is_dinf:
            _flow_accumulate_dinf(field_area, self.stack, nb_rec, recs, wts)
        else:
            _flow_accumulate_sd(field_area, self.stack, self.receivers)

        # 2. Sub-grid glacier plan-view area for the width-aware ablation.
        #    For D-inf, per-cell "length" is the weighted-mean across receivers
        #    (Σ_k w_k · L_k; weights sum to 1).
        if is_dinf:
            lengths_2d = (self._lengths_foreign * self.weights).sum(axis=1).reshape(self.shape)
        else:
            lengths_2d = self._lengths_foreign.reshape(self.shape)
        glacier_width = float(self.width_hack_k) * field_area.reshape(self.shape) ** float(self.width_hack_p)
        glacier_area  = glacier_width * lengths_2d
        wide_area     = np.maximum(glacier_area, self.cell_area)
        # Above ELA, snow falls on the cell, not the glacier's footprint.
        # (ELA test on the climate surface — see z_clim above.)
        accum_area = np.where(z_clim < zELA, wide_area, self.cell_area)

        # 3. Source field for ice accumulation, capped at the cell's total
        #    precipitation flux (no-op on the negative/melt branch). b(z) is the
        #    ELA-relative balance on the pre-uplift climate surface z_clim.
        field_ice = self.beta * (z_clim - zELA) * accum_area
        field_ice = np.where(field_ice > field, field, field_ice)

        field     = field.flatten()
        field_ice = field_ice.flatten()

        # 4. Accumulate water + ice through the flow graph.
        if is_dinf:
            _flow_accumulate_dinf_2(field, field_ice, self.stack, nb_rec, recs, wts)
        else:
            _flow_accumulate_sd_2(field, field_ice, self.stack, self.receivers)

        # Clamp ice flux (nodes below ELA with no upstream contribution can go negative).
        np.maximum(field_ice, 0.0, out=field_ice)

        self.ice_flux = field_ice.reshape(self.shape)
        # Water flux = total precip flux minus ice flux
        self.water_flux = field.reshape(self.shape) - self.ice_flux
        # flowacc is what fastscape's parent class expects (used by other processes)
        self.flowacc = self.water_flux
        self.area = field_area.reshape(self.shape)
        self.basin_ids = self._basin_foreign.astype(np.int32, copy=False)
        # For D-inf, output the "primary" receiver per cell (the largest-weight one)
        # so downstream code (extract_channel etc.) keeps a single-receiver view.
        # Cells with no active receivers (boundary / pits — padded as -1 with all-
        # zero weights) get receiver = self, matching the SFR self-receiving
        # convention used by donor-tree builders.
        if is_dinf:
            recs_2d = np.asarray(self.receivers)
            wts_2d  = np.asarray(self.weights)
            n_nodes = recs_2d.shape[0]
            primary_k = np.argmax(wts_2d, axis=1)
            primary_rec = recs_2d[np.arange(n_nodes), primary_k].astype(np.int32)
            self_idx = np.arange(n_nodes, dtype=np.int32)
            no_flow = (wts_2d.sum(axis=1) <= 0.0) | (primary_rec < 0)
            primary_rec = np.where(no_flow, self_idx, primary_rec)
            self.receivers_2d = primary_rec.reshape(self.shape)
        else:
            self.receivers_2d = self.receivers.reshape(self.shape).astype(np.int32, copy=False)
        self.stack_2d = self.stack.reshape(self.shape).astype(np.int32, copy=False)


@xs.process
class GlacialSPLBase:
    """Glacial + fluvial erosion — shared base for the per-mode erosion
    processes (``GlacialSPLModeA``, ``GlacialSPLModeB``, ``GlacialSPLModeC``).
    Abstract: it owns the
    shared declarations + setup but defines NO ``run_step`` (the concrete
    subclass that fills the ``glacial_spl`` slot supplies it; siim2d picks the
    subclass at build time by mode). It is ``@xs.process``-decorated only so its
    variables are discoverable as foreign targets (siblings + subclasses); it is
    never assigned to a model slot itself.

    The tracked bed is the sub-grid channel floor zb = z - hc_over_H*H (H the
    width-mean depth). The raw law params + constant algebra live in
    :class:`GlacialLaw`; this base reads the derived ``(law_code, GlacialParams)``
    record via the ``law`` foreign. Subclasses dispatch on the record's
    ``law_code`` and on routing dimensionality (SFR vs D-inf), calling the
    ``law_code`` step skeletons in :mod:`siim._core.skeleton`.

    Sibling processes foreign onto THIS base (xsimlab resolves ``foreign(Base)``
    via the MRO to whichever subclass fills the slot): ``SedimentTracker`` and
    ``GlacialFlexure`` read ``denudation`` / ``ice_thickness`` here.
    """

    # --- Outputs (referenced by siblings via foreign(GlacialSPLBase, ...)) ---
    # bedrock_surface is NOT here: it is the derived bed for mode A (which tracks
    # the SURFACE as state), so it lives on GlacialSPLModeA. The citizen mode-B
    # classes (GlacialSPLModeB, GlacialSPLModeC) track zb AS topography, so they
    # have no separate bed output.
    erosion         = xs.variable(dims=("y", "x"), groups="erosion", intent="out")
    erosion_rate    = xs.variable(dims=('y', 'x'), intent='out', description='Erosion rate (m/yr)')
    denudation      = xs.variable(dims=('y', 'x'), intent='out', description="Per-step rock denudation (m, +ve = eroded): the true material removed this step, consumed by SedimentTracker and GlacialFlexure as the sediment source / flexural-unloading signal. Mode B: the tracked bed lowering delta-zb (glacial/fluvial incision + sub-grid carve; bed memory). Mode A: the ice-surface lowering delta-zs (= `erosion`; mode A erodes the surface as its single hc-invariant state, so the derived bed change delta-zs - hc_over_H*delta-H would inject spurious hc-dependence). Distinct from `erosion`, which drives fastscape's elevation update and in mode B equals delta-zb - hc_over_H*delta-H.")

    # --- Law record (raw params + derived constants live in GlacialLaw) ---
    law = xs.foreign(GlacialLaw, 'params')

    # (No `mode` variable: the runtime is selected by class — siim2d picks
    # GlacialSPLModeA vs the mode-B citizens ModeB/ModeC — so an input `mode`
    # was an inert no-op on the public surface and was removed; audit m40.)
    border_bed_uplift = xs.variable(dims=[(), ('y', 'x')], default=0.0, description='Mode B: uplift rate (m/yr) applied to the bed at base-level borders — the U in the icy border budget (net U - f*E on the arrival slope, implicitly integrated) and the rate of the ice-free post-glacial recovery toward bl. Scalar or (y, x) field; time-varying via the clock like uplift__rate. The siim2d wrapper passes its tectonic U (local, per-step) by default; 0 freezes the ice-free bed')
    bl = xs.variable(dims=[()], default=_constants.BL, description="Mode B base level: the water-line (Dirichlet) datum (m). Scalar or a length-nt series on the clock (like zELA); per-step it floors the ice-free-border erosion working view + lake-fill seed, sets the ice-free recovery threshold, and is the flotation reference. Default constants.BL = 0 (bit-for-bit with the historical hard-coded datum).")
    flotation_gate = xs.variable(default=_constants.FLOTATION_GATE, description="Mode B global waterline-flotation gate (default constants.FLOTATION_GATE = True): grounded <=> zs = zb + hc*H >= bl (rho_i = rho_w). Interior: a sub-waterline icy cell does reduced/no glacial erosion (and, via E_c <= 0, carves nothing). Border: the ramp is the physical bound inside the closed-form IMPLICIT border budget (the bed digs on the arrival slope to the flotation-draft equilibrium zb* = bl - hc*H + delta*U/E). False un-bounds the border (f == 1, measured runaway) — diagnostics only. See docs/dev/boundary_conditions.md.")
    flotation_ramp = xs.variable(default=_constants.FLOTATION_RAMP, description="Flotation-ramp width gamma (default constants.FLOTATION_RAMP = 0.1): the flotation gate is the effective-pressure ramp — glacial erosion is multiplied by f = clip((zs - bl)/(gamma*hc*H), 0, 1), with f = 0 exactly for zs <= bl. At the border the ramp is solved implicitly (closed form) inside the border budget, giving the draft equilibrium with standoff gamma*hc*H*U/E. 0 = the hard binary gate (interior bit-for-bit; at the border the flotation sliding mode). Safe ceiling 0.2 (wide ramps can dome cap=False configs). Only active when flotation_gate is on. See docs/dev/soft_gate_probe.md + docs/dev/outflow_implicit_budget.md.")
    parallel_erode = xs.variable(default=_constants.PARALLEL_ERODE, description="Mode-B parallel-eroder toggle (default constants.PARALLEL_ERODE = True): run the erosion step level-scheduled across threads (topological levels of the flow graph; nodes within a level are independent). BIT-FOR-BIT identical to the serial eroder at any thread count (disjoint writes, identical per-node arithmetic; pinned by test_parallel_erode) — a pure scheduling change. False = the serial eroder. See docs/dev/perf_audit.md.")
    carve_width = xs.variable(default=True, description='Mode B only (ValueError otherwise; default True): carve the sub-grid glacier footprint (union of discs of width alpha_g*H around glaciated cells; power-diagram attribution) into the bed. Footprint cells — bare AND icy, INCLUDING terrain above the ice (ridge-eating = channel capture; no surface gate) — descend toward the parabola hung from the source ice surface: rim at zs = zb + HC_OVER_H*H, floor at the source bed. Self-attribution: target = own bed, a structural no-op')
    widening_rate = xs.variable(default=_constants.DEFAULT_WIDENING_RATE, description="Sub-grid widening rate eta = (E_widening - E_c)/E_c >= 0 (dimensionless): the footprint descends at E_widening = (1 + eta)*E_c, i.e. eta centerline-incision-rates of excess over the centerline incision E_c (0 = no net widening, footprint eaten at the channel rate; 1 = widen at the incision rate). None / inf / 'inf' / 'infinity' (any case) = instant U-imposition. Negative is rejected (use carve_width=False to disable).")

    # --- State ---
    ice_thickness = xs.variable(dims=('y', 'x'), intent='inout', global_name='ice_thickness',
                                description='Ice thickness (m)')

    # --- Foreign variables from fastscape processes ---
    # Under D-inf, receivers / lengths / weights are 2D (n_nodes, nb_rec_max);
    # nb_receivers (n_nodes,) tells us how many of them are active per cell.
    # Under SFR, receivers / lengths are 1D and nb_receivers / weights are
    # still provided by the FlowRouter base class.
    # Post-tectonics surface (fastscape's erosion-process convention; the
    # router uses the same source). Reading SurfaceTopography here would make
    # the solver act on a surface one uplift increment stale — slopes lag,
    # and the committed elevation would end up surface_out + U*dt.
    surface       = xs.foreign(SurfaceToErode,     'elevation')
    # (left, right, top, bottom) — the width carve wrap-pads its footprint
    # transform along 'looped' axes (see _power_dt_2d_periodic).
    border_status = xs.foreign(BorderBoundary,     'border_status')
    ice_flux      = xs.foreign(GlacialFlowAccumulator, 'ice_flux',     intent='in')
    water_flux    = xs.foreign(GlacialFlowAccumulator, 'water_flux',   intent='in')
    lengths       = xs.foreign(FlowRouter,         'lengths',      intent='in')
    receivers     = xs.foreign(FlowRouter,         'receivers',    intent='in')
    nb_receivers  = xs.foreign(FlowRouter,         'nb_receivers', intent='in')
    weights       = xs.foreign(FlowRouter,         'weights',      intent='in')
    stack         = xs.foreign(FlowRouter,         'stack',        intent='in')
    shape         = xs.foreign(RasterGrid2D,       'shape')
    _dx_cell      = xs.foreign(RasterGrid2D,       'dx')
    _dy_cell      = xs.foreign(RasterGrid2D,       'dy')

    def initialize(self):
        # The raw law params + constant algebra live in GlacialLaw; read the
        # derived (law_code, GlacialParams) record (xsimlab orders GlacialLaw
        # before this consumer). Unpack the pieces the per-node border closures
        # + carve footprint need; the skeletons take the whole record.
        self._law_code, self._gp = self.law
        # Centerline-to-mean depth ratio (zs = zb + hc_over_H*H) and the channel
        # aspect ratio for the carve footprint — both single-sourced on the
        # record so this process carries no law constants of its own.
        self._hc_over_H = float(self._gp.hc_over_H)
        # Sub-grid width carve: convert the user widening_rate (eta) to the
        # internal E_widening/E_c factor (= 1 + eta); validates loudly for
        # standalone xsimlab use too (the siim2d wrapper checks earlier).
        # inf => instant U-imposition. (Mode A ignores both — it never carves.)
        self._carve = bool(self.carve_width)
        self._widening_factor = _constants.widening_factor_from_rate(self.widening_rate)
        # Looped (periodic) axes — left/right loop the x axis, top/bottom
        # the y axis (fastscape enforces symmetric pairs). Consumed by the
        # carve's footprint transform and the D-inf mode-B lake flood.
        bs = list(self.border_status)
        self._wrap_x = bs[0] == 'looped'
        self._wrap_y = bs[2] == 'looped'


@xs.process
class GlacialSPLModeA(GlacialSPLBase):
    """Mode A erosion (ice-surface state). Fills the ``glacial_spl`` slot when
    ``mode == 'A'``. The ice surface is the single tracked state (pinned at base
    level by the BCs); H is solved from the local pre-erosion surface slope each
    step, the bed is the derived ``z - hc_over_H*H``. Historical siim2d
    behaviour; never carves (the bed is reconstructed, not stored)."""

    bedrock_surface = xs.variable(dims=('y', 'x'), intent='out',
        description='Bedrock (channel-floor) elevation = ice surface - hc_over_H * H')

    def _run_modeA_kernel(self, z_flat, H_flat, dt):
        """Dispatch to the right mode-A skeleton based on routing (law via
        law_code). Modifies z_flat (eroded ice surface) and H_flat (new ice
        thickness) in place. Note: mode A's H solver reads the local slope of
        z_flat *before* erosion, so the H written is consistent with the
        pre-erosion geometry."""
        law_code, p = self._law_code, self._gp
        if self.receivers.ndim == 2:   # D-inf
            nb_rec = np.asarray(self.nb_receivers, dtype=np.int64)
            recs   = np.asarray(self.receivers,   dtype=np.int64)
            wts    = np.asarray(self.weights,     dtype=np.float64)
            lens   = np.asarray(self.lengths,     dtype=np.float64)
            _glac_fast_solve_modeA_dinf(
                z_flat, self.ice_flux.ravel(), self.water_flux.ravel(), H_flat,
                law_code, p, dt, self.stack, nb_rec, recs, wts, lens)
        else:                          # SFR
            _glac_fast_solve_modeA_sfr(
                z_flat, self.ice_flux.ravel(), self.water_flux.ravel(), H_flat,
                law_code, p, dt, self.lengths, self.stack, self.receivers)

    def _H_from_QS(self, Qg, S):
        """Per-law point closure H(Qg, S): the from_slope branch of the shared
        :func:`_modeb_closure` (single-sourced; audit m34). Constants come from
        the GlacialLaw record (``self._gp``); dispatch on ``self._law_code``."""
        p = self._gp
        return float(_modeb_closure(
            self._law_code, True, S, Qg, 0.0, 1.0,
            p.cg, p.lambda_p, p.lambda_c, p.tau_c, p.rho_g_g, p.coulomb_clamp))

    def _solve_border_H_modeA(self, z_flat, H_flat):
        """Mode A: the surface is the anchored BC (z pinned at base level by
        the boundary conditions), so a self-receiving border node with
        through-flowing ice gets its thickness from the per-law H(Q, S)
        closure with S the steepest upwind (donor-side) surface slope —
        matching siim1d's outlet treatment. This feeds bedrock_surface and
        border diagnostics; the surface dynamics are unchanged (z is the
        state and the erosion kernels skip self-receiving nodes)."""
        nn = z_flat.shape[0]
        idx = np.arange(nn)
        ice = self.ice_flux.ravel()
        s_up = np.zeros(nn)
        if self.receivers.ndim == 2:  # D-inf
            rec = np.asarray(self.receivers, dtype=np.int64)
            lens = np.asarray(self.lengths, dtype=np.float64)
            nb = np.asarray(self.nb_receivers, dtype=np.int64)
            self_rec = (nb == 1) & (rec[:, 0] == idx)
            for k in range(rec.shape[1]):
                m = (~self_rec) & (lens[:, k] > 0.0) & (rec[:, k] != idx)
                if k == 1:
                    m &= nb == 2
                np.maximum.at(s_up, rec[m, k],
                              (z_flat[m] - z_flat[rec[m, k]]) / lens[m, k])
        else:  # SFR
            rec = np.asarray(self.receivers, dtype=np.int64)
            lens = np.asarray(self.lengths, dtype=np.float64)
            self_rec = rec == idx
            m = (~self_rec) & (lens > 0.0)
            np.maximum.at(s_up, rec[m], (z_flat[m] - z_flat[rec[m]]) / lens[m])
        for b in np.where(self_rec & (ice > 0.0) & (s_up > 0.0))[0]:
            H_flat[b] = self._H_from_QS(float(ice[b]), float(s_up[b]))

    def _commit_modeA(self, z_flat, H_flat, dt):
        self.ice_thickness = H_flat.reshape(self.shape)
        self.bedrock_surface = (z_flat.reshape(self.shape)
                                - self._hc_over_H * self.ice_thickness)
        self.erosion = self.surface - z_flat.reshape(self.shape)
        self.erosion_rate = self.erosion / dt
        # Mode A erodes the ice surface as its single, hc-invariant state, so the
        # surface lowering IS the denudation (the derived bed change
        # delta-zs - hc*delta-H is hc-dependent and would break that invariance).
        self.denudation = self.erosion

    @xs.runtime(args=("step_delta",))
    def run_step(self, dt):
        dt = _scalar_dt(dt)                      # numpy-2-safe (audit m1)
        z_flat = self.surface.flatten()          # flatten() already returns a fresh copy
        H_flat = self.ice_thickness.flatten()    # ditto
        self._run_modeA_kernel(z_flat, H_flat, dt)
        self._solve_border_H_modeA(z_flat, H_flat)
        self._commit_modeA(z_flat, H_flat, dt)


@xs.process
class GlacialSPLModeB(GlacialSPLBase):
    """Mode B erosion as a fastscape *citizen* (Fork B), no carve.

    The tracked state ``topography__elevation`` IS the pre-uplift bed ``zb`` (the
    sub-grid channel floor); ``ice_thickness`` carries H. Each step reads the
    bed from ``SurfaceTopography.elevation`` (the persisted state — pre-this-step
    uplift; fastscape composes uplift at finalize) and H from ``ice_thickness``,
    runs the UNCHANGED mode-B kernel (which reconstructs ``zs = zb + hc_over_H*H``
    internally), and reports a genuine erosion height:

        ``denudation = zb_in - zb_out``   (the bed lowering this step)
        ``erosion    = denudation``       (the erosion-group height)

    fastscape then finalizes ``zb_new = zb + uplift - erosion = zb + uplift -
    delta-zb``, so the bed evolves correctly. No ``bedrock_surface`` output —
    ``topography`` IS the bed (the derived display surface ``zb + hc_over_H*H``
    is reconstructed by the consumer / plotter). The sub-grid width carve rides
    on the :class:`GlacialSPLModeC` subclass; routing/erosion slopes come from
    the reconstructed ice surface (:class:`GlacialSurfaceToErode`).
    """

    # The persisted bedrock state, read pre-this-step uplift. self.surface
    # (= SurfaceToErode, the reconstructed ice surface here) drives the kernel's
    # routing/slopes; this foreign supplies the actual bed the kernel evolves.
    bed_state = xs.foreign(SurfaceTopography, 'elevation', intent='in')

    def _run_modeB_kernel_nocarve(self, zb_flat, H_flat, dt):
        """No-carve mode-B kernel dispatch (SFR or D-inf). Modifies zb_flat and
        H_flat in place; returns the kernel's surface_out (= zb + hc*H). Mode B
        discards surface_out (its erosion height is denudation, not a
        surface-replace); the carving :class:`GlacialSPLModeC` reuses this
        dispatch and then carves the returned bed."""
        ny, nx = int(self.shape[0]), int(self.shape[1])
        dx_cell = float(self._dx_cell)
        dy_cell = float(self._dy_cell)
        surface_out = np.empty_like(zb_flat)
        # Drop the leading size-1 tstep dim left by xsimlab on a (nt, y, x) slice
        # (xarray no longer squeezes groupby) before broadcasting.
        bbu = np.asarray(self.border_bed_uplift, dtype=np.float64)
        if bbu.ndim == 3:
            bbu = bbu[0]
        bbu_flat = np.broadcast_to(bbu, (ny, nx)).ravel()
        # Per-step water-line datum (the clock strips the 'tstep' dim, so self.bl
        # is a scalar each step) + the global flotation gate and its ramp width.
        bl = float(np.asarray(self.bl).ravel()[-1])
        gate = bool(self.flotation_gate)
        ramp = float(self.flotation_ramp)
        par = bool(self.parallel_erode)
        law_code, p = self._law_code, self._gp
        if self.receivers.ndim == 2:   # D-inf
            nb_rec = np.asarray(self.nb_receivers, dtype=np.int64)
            recs = np.asarray(self.receivers, dtype=np.int64)
            wts = np.asarray(self.weights, dtype=np.float64)
            lens = np.asarray(self.lengths, dtype=np.float64)
            _glac_fast_solve_modeB_dinf(
                zb_flat, self.ice_flux.ravel(), self.water_flux.ravel(),
                H_flat, surface_out, law_code, p,
                dt, self.stack, nb_rec, recs, wts, lens,
                ny, nx, dx_cell, dy_cell, bbu_flat, self._wrap_y, self._wrap_x,
                bl, gate, ramp, par)
        else:
            _glac_fast_solve_modeB_sfr(
                zb_flat, self.ice_flux.ravel(), self.water_flux.ravel(),
                H_flat, surface_out, law_code, p,
                dt, self.lengths, self.stack, self.receivers,
                ny, nx, dx_cell, dy_cell, bbu_flat, bl, gate, ramp,
                self._wrap_y, self._wrap_x, par)
        return surface_out

    @xs.runtime(args=("step_delta",))
    def run_step(self, dt):
        dt = _scalar_dt(dt)                   # numpy-2-safe (audit m1)
        # The bed is the persisted state (pre-this-step uplift). flatten()
        # returns a fresh copy, so zb_flat is safe to mutate in place.
        zb_flat = self.bed_state.flatten()
        H_flat = self.ice_thickness.flatten()
        zb_in = zb_flat.copy()                # pre-kernel bed -> denudation = delta-zb
        self._run_modeB_kernel_nocarve(zb_flat, H_flat, dt)
        self.ice_thickness = H_flat.reshape(self.shape)
        # denudation = the true bed lowering this step; erosion = the same height
        # (fastscape composes zb_new = zb + uplift - erosion = zb + uplift - delta-zb).
        # NB: no bedrock_surface output — topography IS the bed.
        self.denudation = (zb_in - zb_flat).reshape(self.shape)
        self.erosion = self.denudation
        self.erosion_rate = self.erosion / dt


@xs.process
class GlacialSPLModeC(GlacialSPLModeB):
    """Mode C erosion: the citizen mode-B bed+H class (:class:`GlacialSPLModeB`)
    plus the sub-grid glacier-width carve. Fills the ``glacial_spl`` slot for
    ``mode == 'B'`` with ``carve_width`` on (the default siim2d path).

    Everything the citizen does is inherited: ``topography`` IS the tracked bed
    ``zb`` (bed memory; no flicker, no ``bedrock_surface`` output), routing +
    erosion slopes come off the reconstructed ice surface
    (:class:`GlacialSurfaceToErode`), and ``erosion == denudation == delta-zb``
    so fastscape finalizes ``zb_new = zb + uplift - delta-zb``. This subclass
    runs the UNCHANGED mode-B kernel and then carves the sub-grid footprint into
    the bed (the topography state) — so the carve deepening flows straight into
    ``denudation`` (the sediment / flexural-unloading source), the same rock the
    retired surface-replace carve class removed but now on citizen semantics. The
    carve edits the BED after the kernel; ``surface_out`` (the kernel's
    reconstructed ice surface) is handed to the carve for its per-cell updates but
    discarded by the citizen.

    With ``carve_width=False`` this is :class:`GlacialSPLModeB` bit-for-bit (the
    carve is gated on ``self._carve``); the class exists so the carve rides the
    citizen path rather than the surface-replace one."""

    def _ensure_carve_buffers(self, ny, nx):
        """Persistent per-step workspaces for the width carve (the grid shape is
        fixed for the run). The pre-kernel bed snapshot is the run_step-local
        ``zb_in``, so only the transform + post-kernel scratch buffers live
        here."""
        if getattr(self, '_c_D', None) is None or self._c_D.shape != (ny, nx):
            nn = ny * nx
            self._c_zb_kern = np.empty(nn)
            self._c_offsets = np.empty((ny, nx))
            self._c_D = np.empty((ny, nx))
            self._c_SRC = np.empty((ny, nx), dtype=np.int64)

    def _carve_bed(self, zb_flat, H_flat, surface_out, zb_pre):
        """Apply the sub-grid width carve (see :mod:`siim._core.carve`) to the
        post-kernel bed ``zb_flat`` IN PLACE. ``zb_pre`` is the pre-kernel bed
        (the denudation datum + the descent-cap origin, so kernel erosion and
        the carve arbitrate — never add); ``surface_out`` is the kernel's
        reconstructed ice surface (updated for carved cells, discarded by the
        citizen). Routing-agnostic: receivers enter only through the border
        marker ``rec[i] == i`` (self-receiving = base-level border), handed to
        the carve as a 1-D marker array under D-inf. Every icy interior cell
        seeds a disc (all-ones seed mask)."""
        ny, nx = int(self.shape[0]), int(self.shape[1])
        dx_cell = float(self._dx_cell)
        dy_cell = float(self._dy_cell)
        self._ensure_carve_buffers(ny, nx)
        if self.receivers.ndim == 2:            # D-inf: 1-D self-at-border marker
            nn = ny * nx
            idx = np.arange(nn, dtype=np.int64)
            recs = np.asarray(self.receivers, dtype=np.int64)
            rec_marker = np.where(recs[:, 0] == idx, idx, -1)
        else:
            rec_marker = self.receivers
        seed_mask = np.ones(ny * nx, dtype=np.int8)   # seed every icy interior cell
        n_seed = _carve_offsets(H_flat, rec_marker, float(self._gp.alpha_g),
                                self._c_offsets.ravel(), seed_mask)
        if n_seed == 0:
            return
        if self._wrap_x or self._wrap_y:
            _power_dt_2d_periodic(self._c_offsets, dy_cell, dx_cell,
                                  self._c_D, self._c_SRC,
                                  self._wrap_y, self._wrap_x)
        else:
            _power_dt_2d(self._c_offsets, dy_cell, dx_cell,
                         self._c_D, self._c_SRC)
        np.copyto(self._c_zb_kern, zb_flat)     # post-kernel bed (source anchors)
        _carve_subgrid_width(zb_flat, self._c_zb_kern, zb_pre, H_flat,
                             surface_out, rec_marker, self._c_D, self._c_SRC,
                             self._c_offsets, self._widening_factor,
                             self._hc_over_H)

    @xs.runtime(args=("step_delta",))
    def run_step(self, dt):
        dt = _scalar_dt(dt)             # numpy-2-safe (audit m1)
        # Citizen mode-B run_step (bed = the persisted topography state) with the
        # sub-grid carve applied to the bed after the kernel. carve OFF ->
        # GlacialSPLModeB.run_step bit-for-bit.
        zb_flat = self.bed_state.flatten()
        H_flat = self.ice_thickness.flatten()
        zb_in = zb_flat.copy()          # pre-kernel bed -> denudation datum + carve zb_pre
        surface_out = self._run_modeB_kernel_nocarve(zb_flat, H_flat, dt)
        if self._carve:
            self._carve_bed(zb_flat, H_flat, surface_out, zb_in)
        self.ice_thickness = H_flat.reshape(self.shape)
        # denudation = the true bed lowering this step (kernel incision + carve);
        # erosion = the same height (fastscape composes zb_new = zb + uplift -
        # erosion). No bedrock_surface output — topography IS the bed.
        self.denudation = (zb_in - zb_flat).reshape(self.shape)
        self.erosion = self.denudation
        self.erosion_rate = self.erosion / dt



@xs.process
class GlacialSurfaceToErode(SurfaceAfterTectonics):
    """Surface used for routing + erosion slopes in citizen Mode B
    (:class:`GlacialSPLModeB`).

    Subclasses fastscape's :class:`~fastscape.processes.SurfaceAfterTectonics`,
    whose ``elevation = topo_elevation + forced_motion`` is the post-uplift
    surface (here ``topo`` IS the bed ``zb``, so this is the post-uplift bed).
    Adds the reconstructed ice column on top:

        ``elevation = (post-uplift bed zb) + hc_over_H * H_lag``  = ``zs``

    i.e. the ice surface the flow router stacks on and the erosion kernel takes
    its slopes from. The uplift add is inherited (no explicit ``+uplift``); H is
    the step-start (lagged) ``ice_thickness`` global, consistent with the order
    ``surf2erode -> glacial_flow -> glacial_spl``. ``hc_over_H`` is read from the
    law record via :class:`GlacialLaw`. (A pre-uplift climate variant
    ``topo + hc_over_H*H`` is deferred to Phase 3.)

    Anti-flicker relaxation (``routing_relax`` = r; design record
    ``docs/dev/step_flicker.md``): the once-per-step H -> zs -> D8-receivers ->
    flux -> closure-H loop lags by one step and D8 receiver choice is discrete,
    so near-tied surfaces flip whole subtrees of flux each step and H ~ Q^(1/4..
    1/5) turns those O(1) swaps into a period-2 ice-thickness slosh (a cosmetic
    planview flicker; integrated area/volume + attractor stats are unaffected).
    With r > 0 the routing thickness is the EMA
    ``H_eff(t) = r*H_eff(t-1) + (1-r)*H_lag(t)`` (seeded ``H_eff(0) = H_lag``)
    instead of the raw lagged H. r = 0 (default) uses the raw H, bit-for-bit.

    Firewall (the runaway lesson, ``docs/dev/flux_consolidation.md``): the
    relaxed H reaches ONLY this provider's elevation, which feeds ONLY the flow
    router graph and :class:`GlacialFlowAccumulator`'s mass-balance surface. The
    mode-B kernel reconstructs its own ``zs = zb + hc*H`` from the tracked bed +
    raw H for every closure and erosion slope (it never reads this elevation),
    so no relaxed/geometric value ever enters a physics closure, the carve, the
    flexure load or the outputs.
    """

    hc_over_H = xs.foreign(GlacialLaw, 'hc_over_H', intent='in')
    ice_thickness = xs.global_ref('ice_thickness', intent='in')
    routing_relax = xs.variable(
        default=_constants.ROUTING_RELAX,
        description="EMA relaxation r in [0, 1) of the lagged thickness feeding "
                    "the routing + mass-balance surface (anti-flicker): "
                    "H_eff = r*H_eff_prev + (1-r)*H_lag, seeded at H_lag. 0 = off "
                    "(raw lagged H, bit-for-bit). Routing/accumulation only — the "
                    "kernel/carve/outputs stay on raw state. Default "
                    "constants.ROUTING_RELAX.")

    def initialize(self):
        # EMA carry + current-step cache (one array per model step). None seeds
        # the first step to the raw lagged H (H_eff(0) = H_lag). A subclass
        # (TrunkSurfaceToErode) reuses self._H_eff — the SAME single update.
        self._H_eff = None

    def _routing_thickness(self):
        """The (optionally EMA-relaxed) lagged thickness for this step's routing
        + mass-balance surface. ONE update per model step; the result is cached
        on ``self._H_eff`` (both the EMA carry to next step AND the value a
        subclass reuses). r = 0 returns the raw ``ice_thickness`` unchanged
        (bit-for-bit)."""
        r = float(self.routing_relax)
        if r == 0.0:
            H_eff = self.ice_thickness           # raw lagged H (bit-for-bit)
        else:
            H_lag = np.asarray(self.ice_thickness, dtype=np.float64)
            prev = self._H_eff
            H_eff = H_lag if prev is None else r * prev + (1.0 - r) * H_lag
        self._H_eff = H_eff
        return H_eff

    def run_step(self):
        # SurfaceAfterTectonics.run_step sets elevation = topo + forced_motion
        # (= post-uplift bed). Add the reconstructed ice column hc_over_H*H_eff
        # (H_eff = the relaxed lagged H; raw H when routing_relax == 0).
        super().run_step()
        self.elevation = (self.elevation
                          + float(self.hc_over_H) * self._routing_thickness())


def _fabricate_trunk_surface(zs_dyn, zb, H_lag, border, alpha_g, dx, dy,
                             k_dip, floor, offsets, D, SRC, wrap_y, wrap_x):
    """Build the fabricated trunk routing surface (design record
    ``docs/dev/trunk_surface_routing.md``). Pure function (no process state) so
    the process and the channel-persistence test share one implementation.

    ``zs_dyn`` (ny, nx) is the dynamic ice surface ``zb + hc*H_lag``; ``zb`` the
    matching bed; ``H_lag`` the lagged thickness; ``border`` (ny, nx bool) the
    base-level edges (excluded as seeds + never fabricated). ``offsets/D/SRC``
    are (ny, nx) scratch buffers. Returns a fresh (ny, nx) elevation: the
    V-dipped trunk surface at footprint cells (``max(zs_geo, zb)``), ``zs_dyn``
    elsewhere.
    """
    ny, nx = zs_dyn.shape
    nn = ny * nx
    cell_scale = min(dx, dy)
    zs_flat = zs_dyn.ravel()
    zb_flat = zb.ravel()
    border_flat = border.ravel()
    idx = np.arange(nn)

    # Attribution on lagged H (thickest disc wins): seeds = icy ∧ non-border.
    rec_marker = np.where(border_flat, idx, -1)
    seed_mask = (~border_flat).astype(np.int8)
    n_seed = _carve_offsets(H_lag.ravel(), rec_marker, float(alpha_g),
                            offsets.ravel(), seed_mask)
    if n_seed == 0:
        return zs_dyn.copy()
    if wrap_x or wrap_y:
        _power_dt_2d_periodic(offsets, dy, dx, D, SRC, wrap_y, wrap_x)
    else:
        _power_dt_2d(offsets, dy, dx, D, SRC)
    Df = D.ravel()
    SRCf = SRC.ravel()

    # Cross-slope per source: S_c = k_dip * max(|grad zs_dyn|, floor). |grad| at a
    # trunk-bottom source ~= the down-valley slope (cross-valley ~0 there);
    # over-estimating S_c only aids convergence, so grad magnitude is the safe
    # estimator.
    gy, gx = np.gradient(zs_dyn, dy, dx)
    S_c = float(k_dip) * np.maximum(np.hypot(gy, gx).ravel(), float(floor))

    member = (Df < 0.0) & (~border_flat)
    mem_idx = np.nonzero(member)[0]
    elev = zs_flat.copy()
    if mem_idx.size == 0:
        return elev.reshape(ny, nx)
    s = SRCf[mem_idx]
    R2s = -offsets.ravel()[s]                     # R_s^2 at the source
    Rs = np.sqrt(R2s)
    gate = (s >= 0) & (Rs > cell_scale)           # sub-cell sources: no trunk
    mem_idx = mem_idx[gate]
    if mem_idx.size == 0:
        return elev.reshape(ny, nx)
    s = s[gate]
    d = np.sqrt(np.maximum(Df[mem_idx] + R2s[gate], 0.0))
    zs_geo = zs_flat[s] + S_c[s] * (d - Rs[gate])
    elev[mem_idx] = np.maximum(zs_geo, zb_flat[mem_idx])
    return elev.reshape(ny, nx)


@xs.process
class TrunkSurfaceToErode(GlacialSurfaceToErode):
    """Fabricated trunk-surface routing (mode B/C, opt-in ``trunk_surface``;
    design record ``docs/dev/trunk_surface_routing.md``).

    A trunk glacier of mean thickness H occupies width ``W = alpha_g*H`` —
    routinely several cells — but the flow graph carries its ice down 2-3
    *parallel* chains, each with its own diluted ``ice_flux`` and its own closure
    H, understating the trunk H. This provider replaces the plain reconstructed
    ice surface (``zb + hc*H``, :class:`GlacialSurfaceToErode`) with a
    **fabricated** surface that has a LINEAR cross-valley dip toward the trunk
    centerline. The flow router stacks on it and the accumulator reads it, so
    (a) flow converges onto the centerline chain — its raw accumulated flux then
    IS the full cross-section discharge (no accounting correction needed), and
    (b) mass balance is evaluated at the trunk-surface elevation.

    Fabrication (per-step, from LAGGED closure-H — this provider runs before the
    router). The power transform the carve uses gives footprint membership
    ``D < 0`` and per-cell source ``SRC`` (the thickest disc wins — the
    medial-axis-transform inversion). For a footprint cell ``i`` with source
    ``s`` (whose disc radius ``R_s = alpha_g*H_s/2 > cell``)::

        zs_geo(i) = zs_dyn(s) + S_c(s) * (d_i - R_s)

    with ``d_i`` the distance to the source (``d_i^2 = D_i + R_s^2``, from the
    transform), ``zs_dyn(s) = zb_s + hc*H_s`` the source's own dynamic surface,
    and ``S_c(s) = TRUNK_DIP_K * max(|grad zs_dyn|(s), TRUNK_DIP_FLOOR)`` the
    cross-slope (0 at the rim ``d = R_s``, ``-S_c*R_s`` at the axis). The routing
    value inside a footprint is ``max(zs_geo, zb)`` — bare cells / nunataks
    present their rock (routing goes around them); a footprint cell's own
    dynamic ice column is deliberately IGNORED (its ice is part of the trunk;
    taking a max with it would re-erect a spine). Outside footprints the surface
    is the dynamic ``zb + hc*H`` unchanged.

    Discipline (the runaway lesson, ``docs/dev/flux_consolidation.md``): the
    fabricated surface reaches ONLY the flow router (graph) and the accumulator's
    mass-balance surface. The mode-B kernel is UNTOUCHED — it uses the router's
    receiver graph but reconstructs its own ``zs = zb + hc*H`` for every closure
    and erosion slope, so no fabricated (geometric) elevation ever enters a
    closure. A flank cell routed to a taller centerline hits the well-posed
    negative-``a`` (from-a) branch, bounded by flux; the carve reads post-kernel
    closure state. The dip is a routing numerics device (like the priority-flood
    eps), not physics.
    """

    alpha_g       = xs.foreign(GlacialLaw, 'alpha_g', intent='in')
    shape         = xs.foreign(RasterGrid2D, 'shape')
    _dx_cell      = xs.foreign(RasterGrid2D, 'dx')
    _dy_cell      = xs.foreign(RasterGrid2D, 'dy')
    border_status = xs.foreign(BorderBoundary, 'border_status')
    trunk_dip_k = xs.variable(
        default=_constants.TRUNK_DIP_K,
        description="Cross-valley dip slope factor k_dip: S_c = k_dip * S_v(s), "
                    "S_v the local down-valley surface slope. D8 convergence "
                    "needs k_dip > 0.414; default constants.TRUNK_DIP_K.")
    trunk_dip_floor = xs.variable(
        default=_constants.TRUNK_DIP_FLOOR,
        description="Down-valley slope floor (m/m) for flat reaches, so the dip "
                    "still beats routing ties; default constants.TRUNK_DIP_FLOOR.")

    def initialize(self):
        super().initialize()                   # seeds the routing_relax EMA state
        bs = list(self.border_status)          # [left, right, top, bottom]
        self._wrap_x = bs[0] == 'looped'
        self._wrap_y = bs[2] == 'looped'
        ny, nx = int(self.shape[0]), int(self.shape[1])
        # Non-looped domain edges are base-level borders (self-receiving under the
        # router): excluded as seeds AND left at the dynamic surface (never
        # fabricated) — mirrors the carve's border exclusion.
        border = np.zeros((ny, nx), dtype=bool)
        if bs[0] != 'looped':
            border[:, 0] = True
        if bs[1] != 'looped':
            border[:, -1] = True
        if bs[2] != 'looped':
            border[0, :] = True
        if bs[3] != 'looped':
            border[-1, :] = True
        self._border = border.ravel()
        self._offsets = np.empty((ny, nx))
        self._D = np.empty((ny, nx))
        self._SRC = np.empty((ny, nx), dtype=np.int64)

    def run_step(self):
        # 1. Dynamic surface zs_dyn = post-uplift bed + hc*H_eff (the parent).
        #    The parent runs the SINGLE routing_relax EMA update this step and
        #    caches its result on self._H_eff; we reuse it (never re-update), so
        #    the fabrication seeds/radii and the zb re-derivation ride the same
        #    relaxed thickness the routing surface does.
        super().run_step()
        ny, nx = int(self.shape[0]), int(self.shape[1])
        hc = float(self.hc_over_H)
        zs_dyn = np.asarray(self.elevation, dtype=np.float64).reshape(ny, nx)
        H_eff = np.asarray(self._H_eff, dtype=np.float64).reshape(ny, nx)
        zb = zs_dyn - hc * H_eff                     # post-uplift bed (exact)
        # 2. Fabricate the V-dipped trunk routing surface (see the helper).
        self.elevation = _fabricate_trunk_surface(
            zs_dyn, zb, H_eff, self._border.reshape(ny, nx), float(self.alpha_g),
            float(self._dx_cell), float(self._dy_cell),
            self.trunk_dip_k, self.trunk_dip_floor,
            self._offsets, self._D, self._SRC, self._wrap_y, self._wrap_x)


@xs.process
class SedimentTracker:
    """Optional sediment-throughput tracker (added only when siim's
    ``track_sediment`` is True — zero cost otherwise).

    Routes each step's denuded rock volume — ``max(denudation, 0) * cell_area`` —
    down the flow graph in one accumulation pass, giving per node the total
    upstream-eroded volume passing through it that step (``flux``, m^3) and its
    running time-integral (``cumulative``, m^3). Detachment-limited bookkeeping
    only: no deposition, every node passes all upstream sediment through.
    Difference ``cumulative`` along the time axis to recover per-interval volumes;
    the outlet node's ``cumulative`` is the whole-basin yield.

    Reads the erosion process's ``denudation`` (via ``GlacialSPLBase``: the true
    rock removed — delta-zb incl. sub-grid carve in mode B, delta-zs in mode A;
    per-step, already includes dt) so it runs after the erosion step and reuses
    the same receiver/stack flow graph.
    """
    denudation   = xs.foreign(GlacialSPLBase, 'denudation',  intent='in')
    cell_area    = xs.foreign(RasterGrid2D, 'cell_area')
    shape        = xs.foreign(RasterGrid2D, 'shape')
    receivers    = xs.foreign(FlowRouter,   'receivers',    intent='in')
    nb_receivers = xs.foreign(FlowRouter,   'nb_receivers', intent='in')
    weights      = xs.foreign(FlowRouter,   'weights',      intent='in')
    stack        = xs.foreign(FlowRouter,   'stack',        intent='in')

    flux       = xs.variable(dims=('y', 'x'), intent='out',
                             description='Upstream-eroded volume routed through each node this step (m^3)')
    cumulative = xs.variable(dims=('y', 'x'), intent='out',
                             description='Running time-integral of flux per node (m^3)')

    def initialize(self):
        self._cum = np.zeros(self.shape, dtype=np.float64)

    def run_step(self):
        # Per-cell eroded volume (m^3), routed down the flow graph. Clamp to >= 0
        # so deposition / boundary noise can't subtract from the sediment budget.
        field = np.maximum(self.denudation, 0.0).astype(np.float64).flatten() * float(self.cell_area)
        if self.receivers.ndim == 2:   # D-inf
            nb_rec = np.asarray(self.nb_receivers, dtype=np.int64)
            recs   = np.asarray(self.receivers,   dtype=np.int64)
            wts    = np.asarray(self.weights,     dtype=np.float64)
            _flow_accumulate_dinf(field, self.stack, nb_rec, recs, wts)
        else:                          # SFR
            _flow_accumulate_sd(field, self.stack, self.receivers)

        self.flux = field.reshape(self.shape)
        # New array each step (not in-place) so prior snapshots stay valid.
        self._cum = self._cum + self.flux
        self.cumulative = self._cum


@xs.process
class GlacialFlexure(Flexure):
    """Flexural isostasy with true glacial isostatic adjustment: ice loading +
    erosional/tectonic unloading + elastic rebound.

    fastscape's :class:`~fastscape.processes.Flexure` loads the elastic plate from
    ``TotalErosion.height`` (the ice-SURFACE change in siim2d mode B — ice-thickness
    change leaks in). This override does two things:

    1. Re-sources the *rock* load from the erosion process's per-step ``denudation``
       (via :class:`GlacialSPLBase`; mode B: delta-zb incl. sub-grid carve; mode A:
       delta-zs), so the plate flexes in response to rock erosional + tectonic
       unloading, correct under transient ice.
    2. Adds the per-step ICE load. The glacial ice load is the channel cross-section
       carried over the cell, ``col = alpha_g * H**2 / L`` (L = sqrt(cell_area)) — the
       mass-conserving, hc-FREE ice volume per cell. alpha_g*H**2 = Qg/V (the ice flux
       over the channel velocity), and the centerline/mean hc factor cancels the
       parabola's 2/3, so it never appears. Its per-step change folds in as a
       rock-equivalent column, (rho_ice/lithos_density)*d(col), into the ``diff`` that
       drives fastscape's incremental plate solve (incremental: the full column would
       re-apply the whole load every step). Width-aware (∝ H**2): a glacier wider than
       the cell (alpha_g*H > L) piles proportionally more ice than the mean depth H.
       Gated by ``ice_load`` (default True; False = erosion-only, the pre-GIA behaviour).
       The mass-conserving Qg/V load is used in preference to a per-cell thickness
       because siim does not resolve 3D ice flow: sheet accumulation Qg = area*balance
       is trustworthy, the unresolved per-cell H is not (see DECISIONS 2026-06-16).

    The plate solve, densities, boundary handling and rebound feedback are inherited
    unchanged. The ice LOAD is applied everywhere ice is present, including afloat cells
    (zs = zb + hc*H < bl): the waterline-flotation gate (docs/dev/boundary_conditions.md)
    gates glacial EROSION, not the flexural load. This is an accepted secondary seam —
    the ice mass is physically present regardless of grounding, and a true floating-shelf
    load would be reduced by the water it displaces; siim's overdeepenings sit right at
    the flotation draft, so the residual is small. Hillslope diffusion of the ice surface
    is deliberately not in the load (an accepted small approximation: it is a near-
    conservative redistribution of the ice surface, not rock unloading).
    """
    erosion = xs.foreign(GlacialSPLBase, 'denudation')
    ice_thickness = xs.foreign(GlacialSPLBase, 'ice_thickness', intent='in')
    alpha_g = xs.foreign(GlacialLaw, 'alpha_g', intent='in')
    cell_area = xs.foreign(RasterGrid2D, 'cell_area')
    ice_load = xs.variable(default=True, description="Include the per-step glacial ice "
        "load (the channel cross-section alpha_g*H**2/L, = Qg/V) in the flexural plate "
        "load (true glacial isostatic adjustment). False = erosional/tectonic unloading "
        "only (the pre-GIA behaviour; isolate the ice contribution or reproduce runs "
        "saved before the ice load).")

    def initialize(self):
        self._col_prev = np.zeros(self.shape)

    def _ice_column(self):
        # Mass-conserving glacial ice load: the channel cross-section alpha_g*H**2
        # (= Qg/V, hc-free) carried over a cell-sized length L -> areal column
        # alpha_g*H**2 / L, with L = sqrt(cell_area). Width-aware (∝ H**2).
        L = float(self.cell_area) ** 0.5
        col = float(self.alpha_g) * self.ice_thickness ** 2 / L
        return col

    def run_step(self):
        import fastscapelib_fortran as fs   # lazy: fortran lib is not mocked in the docs build
        ny, nx = self.shape
        yl, xl = self.length

        lithos_density = np.broadcast_to(self.lithos_density, self.shape).flatten()
        elevation_eq = self.elevation.flatten()
        diff = (self.surface_upward - self.erosion).ravel()
        col = self._ice_column()
        if self.ice_load:
            # Incremental rock-equivalent ice load: rho_ice*g*d(col) == rho_lithos*g*
            # ((rho_ice/lithos_density)*d(col)); g cancels (density-ratio identity),
            # lithos_density is per-cell so divide element-wise. col is the channel
            # cross-section alpha_g*H**2/L (mass-conserving; see _ice_column).
            dcol = (col - self._col_prev).ravel()
            diff = diff + (RHO_ICE / lithos_density) * dcol

        elevation_pre = elevation_eq + diff
        elevation_post = elevation_pre.copy()
        fs.flexure(
            elevation_post,
            elevation_eq,
            nx,
            ny,
            xl,
            yl,
            lithos_density,
            self.asthen_density,
            self.e_thickness,
            self.ibc,
        )
        self.rebound = (elevation_post - elevation_pre).reshape(self.shape)
        self._col_prev = col.copy()


@xs.process
class DinfFlowRouter(FlowRouter):
    """D-infinity flow router :cite:p:`tarbotonNewMethodDetermination1997`.

    Subclasses fastscape's ``FlowRouter``. Calls the fortran routing first for
    boundary handling, donor lists, and basin IDs, then overrides receivers,
    weights, lengths and stack with the D-infinity computation. Outputs use the
    (n_nodes, 2) receiver/weight shape consumed by GlacialFlowAccumulator and the
    mode-B / mode-A D-inf kernels.

    Continuous facet-direction weights replace the discrete neighbor weighting
    jumps of Quinn-style multi-flow routing, removing per-step receiver-flip
    kicks (~10x reduction at 100x100 vs Quinn MFR).

    Depressions: directions are computed on the eps-filled surface
    (priority-flood from the outlet cells, _priority_flood_eps), so flux
    crosses closed basins toward their spills — the same depression
    semantics as the fortran SFR. The PHYSICS stays on the true surface:
    the mass balance samples real z (flux crossing a deep trough melts at
    the drowned elevation and dies) and the erosion/H kernels clip
    negative true-surface slopes to zero (lake interiors do not erode).
    Looped boundaries wrap (fill and facet scan both). The citable
    reference is docs/dev/dinf_routing.md.
    """

    dx = xs.foreign(RasterGrid2D, 'dx')
    dy = xs.foreign(RasterGrid2D, 'dy')
    border_status = xs.foreign(BorderBoundary, 'border_status')

    def route_flow(self):
        # Run fastscapelib_fortran's flowrouting to populate fs_context with
        # baseline SFR info (used for boundary detection + donors + basin).
        import fastscapelib_fortran as fs
        fs.flowrouting()

    def run_step(self):
        # Base class sets fs_context['h'], calls route_flow(), populates donors.
        super().run_step()

        ny, nx = int(self.shape[0]), int(self.shape[1])
        nn = ny * nx

        # Boundary detection: SFR rec[i]==i marks boundary/sink cells.
        sfr_rec = (np.asarray(self.fs_context['rec']).astype(np.int64) - 1)
        interior = (sfr_rec != np.arange(nn)).astype(np.int8)

        # D-inf routing on the eps-filled surface: every interior cell —
        # including closed-basin floors — gets a strictly downhill facet,
        # so flux crosses depressions toward their spills and the
        # topological sort is valid by construction. The kernels/physics
        # keep consuming the true elevation.
        bs = list(np.broadcast_to(self.border_status, 4))
        wrap_x = bs[0] == 'looped'
        wrap_y = bs[2] == 'looped'
        z_flat = np.ascontiguousarray(self.elevation.ravel().astype(np.float64))
        z_route = np.empty(nn, dtype=np.float64)
        _priority_flood_eps(z_flat, ny, nx, interior, 1e-6,
                            wrap_y, wrap_x, z_route)
        rec1 = np.zeros(nn, dtype=np.int64)
        rec2 = np.zeros(nn, dtype=np.int64)
        w1 = np.zeros(nn, dtype=np.float64)
        w2 = np.zeros(nn, dtype=np.float64)
        len1 = np.zeros(nn, dtype=np.float64)
        len2 = np.zeros(nn, dtype=np.float64)
        # slope_out = the steepest-facet slope on the FILLED surface; the router
        # itself does not consume it (audit N14), but it is a cheap routing
        # diagnostic exercised by the D-inf slope-correctness tests — kept, not
        # a live input to anything downstream.
        slope_out = np.zeros(nn, dtype=np.float64)
        _dinf_route(z_route, ny, nx, float(self.dx), float(self.dy), interior,
                    rec1, rec2, w1, w2, len1, len2, slope_out,
                    _DINF_E1_DJ, _DINF_E1_DI, _DINF_E2_DJ, _DINF_E2_DI,
                    wrap_y, wrap_x)

        # Pack into (n, 2) receiver/weight arrays
        receivers = np.zeros((nn, 2), dtype=np.int64)
        weights = np.zeros((nn, 2), dtype=np.float64)
        lengths = np.zeros((nn, 2), dtype=np.float64)
        nb_receivers = np.zeros(nn, dtype=np.int64)
        _dinf_pack(rec1, rec2, w1, w2, len1, len2, nn,
                   receivers, weights, lengths, nb_receivers)

        # Topological sort (receivers-first), then reverse for fastscape's
        # donor-first convention (_flow_accumulate_dinf / _flow_accumulate_dinf_2
        # iterate `for inode in stack`, walking from upstream down). Rebuilt each
        # step: the topology changes every step during transient evolution, so
        # the old topology cache never hit (measured 0/80 steps) and the rebuild
        # it would save is ~1% of the D-inf step — removed (audit N32).
        stack_rec_first = np.zeros(nn, dtype=np.int64)
        _dinf_topo_stack(rec1, rec2, w1, w2, nn, stack_rec_first)
        stack = stack_rec_first[::-1].copy()

        # Set xs.variable outputs
        self.stack = stack
        self.nb_receivers = nb_receivers
        self.receivers = receivers
        self.weights = weights
        self.lengths = lengths
