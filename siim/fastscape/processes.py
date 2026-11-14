"""
xsimlab process classes for siim2d.

The processes that plug into fastscape's ``basic_model`` include:

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
    (1997) D-infinity flow directions, and ``MfdFlowRouter`` with the
    multiple-flow-direction split (both defined at the bottom of this module).

The numba flow-accumulation + D-infinity routing primitives live in
:mod:`siim._core.routing`; the ice-thickness + erosion solvers in
:mod:`siim._core` (``solvers`` / ``eroders`` / ``skeleton``).
"""

import numpy as np

try:
    import xsimlab as xs
    from fastscape.processes import (
        BlockUplift,
        FlowAccumulator, FlowRouter, RasterGrid2D, SurfaceToErode,
        SurfaceAfterTectonics, SurfaceTopography, BorderBoundary, Flexure,
        TectonicForcing,
    )
except ImportError as e:
    raise ImportError(
        "siim.fastscape is the OPTIONAL fastscape/xsimlab adapter — the "
        "standalone 2D model (siim.siim2d) needs none of it. To use the "
        "adapter: `pip install siim-lem[fastscape]` provides fastscape + xsimlab "
        "from PyPI, but fastscapelib-fortran (the fortran backend fastscape's "
        "stock processes call at run time) has no PyPI wheel — use the conda "
        "env instead: `conda env create -f environment.yml`."
    ) from e

from .. import constants as _constants
from .._core.flexure import flexure as _inhouse_flexure
from .._core.hillslope import diffuse as _inhouse_diffuse
from .._core.step import (
    build_glacial_params, initial_topography, uplift_mask, block_rate, block_uplift,
    routing_surface, _fabricate_trunk_surface,
    accumulate_glacial_flow, run_modeA_step, _solve_border_H_modeA,
    run_modeB_kernel, carve_bed, route_dinf, route_d8, route_mfd,
    accumulate_sediment, edge_sediment, glacial_flexure_step, transect_flux,
)
from .._core.router import check_mfd_config as _check_mfd_config, resolve_trunk_dip_k


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
        self.elevation = initial_topography(
            self.elevation_init, self.shape, self.border_status,
            self.seed, self.noise_amplitude)


@xs.process
class GlacialBlockUplift(BlockUplift):
    """BlockUplift that tolerates a time-varying ``(nt, ny, nx)`` uplift rate.

    xsimlab slices the clock axis of a ``(('tstep', 'y', 'x'), array)`` input
    per step, but xarray no longer squeezes groupby, so the per-step ``rate``
    keeps a leading size-1 dim and stock ``BlockUplift``'s
    ``np.broadcast_to((1, ny, nx), (ny, nx))`` raises. Drop that leading dim
    before broadcasting; scalar / ``(y, x)`` rates are unaffected."""

    border_rate = xs.variable(
        dims=('y', 'x'), intent='out', groups=['border_bed_uplift'],
        description="The local uplift rate (m/yr), unmasked: the default border-bed uplift")

    def initialize(self):
        self._mask = uplift_mask(self.status, self.shape)

    @xs.runtime(args="step_delta")
    def run_step(self, dt):
        self.uplift = block_uplift(self.rate, dt, self._mask, self.shape)
        self.border_rate = block_rate(self.rate, self.shape)


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
    mu = xs.variable(
        default=None,
        description="Glacial flux exponent override. Used by the effective-"
                    "exponent law; the exact power and Coulomb kernels derive "
                    "their flux exponents from nu/ell and do not consume it.")

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
        self.params = build_glacial_params(
            sliding_law=self.sliding_law, Ko=self.Ko, ce=self.ce,
            n=self.n, nu=self.nu, m=self.m, mu=self.mu, Ac=self.Ac,
            alpha_g=self.alpha_g, lambda_p=self.lambda_p,
            lambda_c=self.lambda_c, tau_c=self.tau_c,
            coulomb_clamp=self.coulomb_clamp, hc_over_H=self.hc_over_H,
            H_diffusivity=self.H_diffusivity)


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
    # model): the dynamic surface, never the trunk V (FIREWALL: see
    # siim._core.driver). Routing + the per-cell glacier-area lengths come from
    # the router's graph; the ELA-relative mass balance b(z) is evaluated one
    # increment lower, on the PRE-uplift climate surface
    # z_clim = surface - surface_upward (below).
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
        (self.ice_flux, self.water_flux, self.area, self.basin_ids,
         self.receivers_2d, self.stack_2d) = accumulate_glacial_flow(
            self.surface, self.surface_upward, self.zELA, self.beta,
            self.runoff, self.cell_area, self.width_hack_k, self.width_hack_p,
            self.shape, self.stack, self.receivers, self.nb_receivers,
            self.weights, self._lengths_foreign, self._basin_foreign)
        # flowacc is what fastscape's parent class expects (used by other processes).
        self.flowacc = self.water_flux


_CARVE_WIDTH_DOC = ('GlacialSPLModeC only (refused on the other classes, which never carve): '
                    'carve the sub-grid glacier footprint (union of discs of width alpha_g*H around glaciated cells; power-diagram attribution) into the bed. Footprint cells — bare AND icy, INCLUDING terrain above the ice (ridge-eating = channel capture; no surface gate) — descend toward the parabola hung from the source ice surface: rim at zs = zb + HC_OVER_H*H, floor at the source bed. Self-attribution: target = own bed, a structural no-op')


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

    _carves = False                                   # GlacialSPLModeC carves
    _closures = _constants.CLOSURE_FLUXES             # mode A: 'cell' only

    # --- Outputs (referenced by siblings via foreign(GlacialSPLBase, ...)) ---
    # bedrock_surface is NOT here: it is the derived bed for mode A (which tracks
    # the SURFACE as state), so it lives on GlacialSPLModeA. The citizen mode-B
    # classes (GlacialSPLModeB, GlacialSPLModeC) track zb AS topography, so they
    # have no separate bed output.
    erosion         = xs.variable(dims=("y", "x"), groups="erosion", intent="out")
    erosion_rate    = xs.variable(dims=('y', 'x'), intent='out', description='Erosion rate (m/yr)')
    denudation      = xs.variable(dims=('y', 'x'), intent='out', description="Per-step rock denudation (m, +ve = eroded): the true material removed this step, consumed by SedimentTracker and GlacialFlexure as the sediment source / flexural-unloading signal. Mode B: the tracked bed lowering delta-zb (glacial/fluvial incision + sub-grid carve; bed memory). Mode A: the ice-surface lowering delta-zs (= `erosion`; mode A erodes the surface as its single hc-invariant state, so the derived bed change delta-zs - hc_over_H*delta-H would inject spurious hc-dependence). In both modes it equals `erosion`, the height that drives fastscape's elevation update.")

    # --- Law record (raw params + derived constants live in GlacialLaw) ---
    law = xs.foreign(GlacialLaw, 'params')

    # (No `mode` variable: the runtime is selected by class — siim2d picks
    # GlacialSPLModeA vs the mode-B citizens ModeB/ModeC — so an input `mode`
    # was an inert no-op on the public surface and was removed; audit m40.)
    border_bed_uplift = xs.variable(dims=[(), ('y', 'x')], default=None, description='Mode B: uplift rate (m/yr) applied to the bed at base-level borders — the U in the icy border budget (net U - f*E on the arrival slope, implicitly integrated) and the rate of the ice-free post-glacial recovery toward bl (the bed never rises above bl). None (default) takes the local, per-step rate the uplift process publishes in the border_bed_uplift group (GlacialBlockUplift, WaveUplift), as siim2d does; a value given is used alone. Scalar or (y, x) field; time-varying via the clock like uplift__rate; 0 freezes the recovery')
    # the local uplift rate the uplift process publishes (GlacialBlockUplift,
    # WaveUplift): the border-bed uplift when border_bed_uplift is None, as
    # siim2d defaults it (audit #25, #44); empty, it is 0
    border_bed_uplift_forcing = xs.group('border_bed_uplift')
    bl = xs.variable(dims=[()], default=_constants.BL, description="Mode B base level: the water-line (Dirichlet) datum (m). Scalar or a length-nt series on the clock (like zELA); per-step it floors the ice-free-border erosion working view + lake-fill seed, sets the ice-free recovery threshold, and is the flotation reference. Default constants.BL = 0 (bit-for-bit with the historical hard-coded datum).")
    flotation_gate = xs.variable(default=_constants.FLOTATION_GATE, description="Mode B global waterline-flotation gate (default constants.FLOTATION_GATE = True): grounded <=> zs = zb + hc*H >= bl (rho_i = rho_w). Interior: a sub-waterline icy cell does reduced/no glacial erosion (and, via E_c <= 0, carves nothing). Border: the ramp is the physical bound inside the closed-form IMPLICIT border budget (the bed digs on the arrival slope to the flotation-draft equilibrium zb* = bl - hc*H + delta*U/E). False un-bounds the border (f == 1, measured runaway) — diagnostics only. See docs/guides/concepts.md.")
    flotation_ramp = xs.variable(default=_constants.FLOTATION_RAMP, description="Flotation-ramp width gamma (default constants.FLOTATION_RAMP = 0.1): the flotation gate is the effective-pressure ramp — glacial erosion is multiplied by f = clip((zs - bl)/(gamma*hc*H), 0, 1), with f = 0 exactly for zs <= bl. At the border the ramp is solved implicitly (closed form) inside the border budget, giving the draft equilibrium with standoff gamma*hc*H*U/E. 0 = the hard binary gate (interior bit-for-bit; at the border the flotation sliding mode). Safe ceiling 0.2 (wide ramps can dome cap=False configs). Only active when flotation_gate is on. See docs/guides/concepts.md.")
    parallel_erode = xs.variable(default=_constants.PARALLEL_ERODE, description="Mode-B parallel-eroder toggle (default constants.PARALLEL_ERODE = True): run the erosion step level-scheduled across threads (topological levels of the flow graph; nodes within a level are independent). BIT-FOR-BIT identical to the serial eroder at any thread count (disjoint writes, identical per-node arithmetic; pinned by test_parallel_erode) — a pure scheduling change. False = the serial eroder.")
    carve_width = xs.variable(default=False, description=_CARVE_WIDTH_DOC)
    # EXPERIMENTAL closure flux (constants.CLOSURE_FLUX; branch
    # closure-transect-flux): 'cell' = the cell's own routed flux (the
    # pre-branch closure), 'transect' / 'transect_min' = the routed flux crossing
    # a line one closure-width (alpha_g*H_lag) long through the cell. Consumed by
    # the mode-B/C kernel dispatch only (mode A has only 'cell' and refuses the
    # others); mirrors the in-house driver: the kernel's closures +
    # eroders see it, routing/outputs/carve/flexure/sediment keep ice_flux.
    closure_flux = xs.variable(
        default=_constants.CLOSURE_FLUX,
        description="'cell', 'transect' or 'transect_min' (see constants.CLOSURE_FLUX)")
    # gamma effective-slope normalisation (constants.MFD_GAMMA_FLOOR). Derived
    # from the ROUTER, not a physics knob: the None default resolves from the
    # RECEIVER ARITY, which is this side of the adapter seam's faithful mirror
    # of _core.router.SLOPE_NORMALISED_ROUTERS (any multi-receiver bundle -> on,
    # single-flow -> off). siim2d passes an explicit bool.
    slope_normalise = xs.variable(
        default=None,
        description="Divide the multi-receiver effective slope by ``gamma = |sum_k w_k*e_k|``, applied as gamma-scaled link lengths so the thickness closure and all three eroders see the same slope (see constants.MFD_GAMMA_FLOOR). ``None`` (default) is router-derived: ON for any multi-receiver router (D-inf, MFD), OFF for single-flow, where gamma == 1 identically and the flag is a no-op. ``False`` restores the pre-2026-09-05 unnormalised D-inf reduction (7.6 % low at mid-facet); MFD without the normalisation is not supported.")
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
        # inf => instant U-imposition. (Only GlacialSPLModeC carves; the other
        # classes refuse carve_width=True below.)
        self._carve = bool(self.carve_width)
        self._widening_factor = _constants.widening_factor_from_rate(self.widening_rate)
        # Inputs a class ignores are refused, not run silently (audit #45, #49):
        # the carve is chosen by class, mode A has only the cell closure.
        if self._carve and not self._carves:
            raise ValueError(f"carve_width=True needs the carving process "
                             f"(GlacialSPLModeC, glacial_model(mode='C')); "
                             f"{type(self).__name__} never carves")
        if self.closure_flux not in self._closures:
            raise ValueError(f"closure_flux must be one of {self._closures} for "
                             f"{type(self).__name__}, got {self.closure_flux!r}")
        # Looped (periodic) axes — left/right loop the x axis, top/bottom
        # the y axis (fastscape enforces symmetric pairs). Consumed by the
        # carve's footprint transform and the D-inf mode-B lake flood.
        bs = list(self.border_status)
        self._wrap_x = bs[0] == 'looped'
        self._wrap_y = bs[2] == 'looped'

    def _resolve_slope_normalise(self):
        """The gamma effective-slope normalisation for this step.

        ``None`` (the default) is ROUTER-DERIVED, mirroring
        :data:`siim._core.router.SLOPE_NORMALISED_ROUTERS` across the adapter
        seam through the only router signal this process has: receiver arity.
        A ``(n, k)`` bundle is D-inf or MFD (normalise); a 1-D bundle is
        single-flow, where gamma == 1 identically and the flag is a no-op.
        An explicit ``True``/``False`` wins."""
        if self.slope_normalise is None:
            return np.ndim(self.receivers) == 2
        return bool(self.slope_normalise)

    def _border_bed_uplift(self):
        """The border-bed uplift this step: the explicit input, else the local
        rate the uplift process publishes (GlacialBlockUplift, WaveUplift). An
        uplift process that publishes none (stock BlockUplift) is refused rather
        than freezing the borders at 0 unasked (audit #44)."""
        if self.border_bed_uplift is not None:
            return self.border_bed_uplift
        published = list(self.border_bed_uplift_forcing)
        if not published:
            raise ValueError(
                "glacial_spl__border_bed_uplift is None (the local uplift rate), but no "
                "process publishes one: use GlacialBlockUplift or WaveUplift for the "
                "'uplift' slot, or set glacial_spl__border_bed_uplift (0 freezes the borders)")
        return sum(published)

    def _check_mfd_closure(self, closure_flux):
        """The closure half of the MFD guard set (this process owns the
        closure; the router owns dx == dy, the facade the mode). MFD is
        detected by receiver arity > 2 -- ``slope_normalise`` stopped being an
        MFD marker when D-inf joined SLOPE_NORMALISED_ROUTERS (2026-09-05).
        Mode A passes ``'cell'``, its only closure, so a hand-built MFD mode-A
        model raises here too."""
        if np.ndim(self.receivers) == 2 and self.receivers.shape[1] > 2:
            _check_mfd_config('mfd', closure_flux=closure_flux,
                              slope_normalise=self.slope_normalise)


@xs.process
class GlacialSPLModeA(GlacialSPLBase):
    """Mode A erosion (ice-surface state). Fills the ``glacial_spl`` slot when
    ``mode == 'A'``. The ice surface is the single tracked state (pinned at base
    level by the BCs); H is solved from the local pre-erosion surface slope each
    step, the bed is the derived ``z - hc_over_H*H``. Historical siim2d
    behaviour; never carves (the bed is reconstructed, not stored)."""

    _closures = (_constants.MODE_A_CLOSURE_FLUX,)
    closure_flux = xs.variable(default=_constants.MODE_A_CLOSURE_FLUX, description="'cell', mode A's only closure")

    bedrock_surface = xs.variable(dims=('y', 'x'), intent='out',
        description='Bedrock (channel-floor) elevation = ice surface - hc_over_H * H')

    def _solve_border_H_modeA(self, z_flat, H_flat):
        """Mode A border-H closure — thin delegator to the framework-free
        :func:`siim._core.step._solve_border_H_modeA`. Kept as a method for the
        direct-instance test (``test_solver_bcs.py``); the run_step path solves
        the border inside :func:`run_modeA_step`. Mutates ``H_flat`` in place;
        ``nb_receivers`` is read only on the D-inf branch (absent under SFR)."""
        _solve_border_H_modeA(
            z_flat, H_flat, self.ice_flux, self.receivers,
            getattr(self, 'nb_receivers', None), self.lengths,
            self._law_code, self._gp)

    @xs.runtime(args=("step_delta",))
    def run_step(self, dt):
        dt = _scalar_dt(dt)                      # numpy-2-safe (audit m1)
        self._check_mfd_closure('cell')
        (_z_eroded, self.ice_thickness, self.bedrock_surface,
         self.erosion, self.denudation) = run_modeA_step(
            self.surface, self.ice_thickness, self.ice_flux, self.water_flux,
            self._law_code, self._gp, dt, self.stack, self.receivers,
            self.nb_receivers, self.weights, self.lengths, self._hc_over_H,
            self.shape, self._resolve_slope_normalise(), self._dx_cell,
            self._dy_cell, self._wrap_y, self._wrap_x)
        self.erosion_rate = self.erosion / dt


@xs.process
class GlacialSPLModeB(GlacialSPLBase):
    """Mode B erosion as a fastscape *citizen* (Fork B), no carve.

    The tracked state ``topography__elevation`` IS the bed ``zb`` (the sub-grid
    channel floor); ``ice_thickness`` carries H. Each step hands the kernel the
    POST-uplift bed (persisted state + this step's ``TectonicForcing`` bed motion;
    fastscape composes the same uplift at finalize) and H from ``ice_thickness``,
    runs the UNCHANGED mode-B kernel (which reconstructs ``zs = zb + hc_over_H*H``
    internally), and reports a genuine erosion height:

        ``denudation = zb_in - zb_out``   (the bed lowering this step)
        ``erosion    = denudation``       (the erosion-group height)

    fastscape then finalizes ``zb_new = zb + uplift - erosion = zb + uplift -
    delta-zb`` = the kernel's eroded post-uplift bed, so the bed evolves correctly. No ``bedrock_surface`` output —
    ``topography`` IS the bed (the derived display surface ``zb + hc_over_H*H``
    is reconstructed by the consumer / plotter). The sub-grid width carve rides
    on the :class:`GlacialSPLModeC` subclass. The router routes on the
    reconstructed ice surface (:class:`GlacialSurfaceToErode`, or the fabricated
    trunk V of :class:`TrunkSurfaceToErode`); the kernel rebuilds its own
    ``zs = zb + hc_over_H*H`` for every closure and erosion slope.
    """

    # The persisted bedrock state (pre-this-step uplift; see bed_uplift): the
    # actual bed the kernel evolves. The kernel never reads self.surface (=
    # SurfaceToErode, the reconstructed ice surface here).
    bed_state = xs.foreign(SurfaceTopography, 'elevation', intent='in')
    # This step's imposed bed uplift (TectonicForcing sums the block/wave uplift
    # group; zero on fixed_value borders). The kernel contract is the POST-uplift
    # bed: eroding the pre-uplift bed against the pinned borders left a U*dt lip
    # on the first interior row at every base-level border.
    bed_uplift = xs.foreign(TectonicForcing, 'bedrock_upward', intent='in')
    # siim2d stores it only with save_closure_pair=True under a transect closure
    # (_core.outputs.output_spec)
    closure_ice_flux = xs.variable(
        dims=('y', 'x'), intent='out',
        description="Ice flux the thickness closure solved H from (m^3/yr): ice_flux under closure_flux='cell', the transect flux otherwise. With ice_thickness it is the pair the post-hoc ice speed inverts (siim._core.solvers.ice_speeds).")

    def _run_modeB_kernel_nocarve(self, zb_flat, H_flat, dt):
        """No-carve mode-B kernel dispatch — thin wrapper over the framework-free
        :func:`siim._core.step.run_modeB_kernel`. Mutates ``zb_flat`` and
        ``H_flat`` in place; returns the kernel's ``surface_out`` (= zb + hc*H).
        Mode B discards ``surface_out`` (its erosion height is denudation, not a
        surface-replace); the carving :class:`GlacialSPLModeC` reuses this
        dispatch and then carves the returned bed."""
        cf = str(self.closure_flux)
        self._check_mfd_closure(cf)
        if cf != 'cell':
            # H_flat is the RAW lagged H(t-1) at this point (pre-kernel copy).
            Q_close = transect_flux(
                self.ice_flux, H_flat.reshape(self.shape), self.receivers,
                self.weights, self.nb_receivers, self.shape, self._dx_cell,
                self._dy_cell, float(self._gp.alpha_g), self._wrap_y, self._wrap_x,
                min_rule=(cf == 'transect_min'))
        else:
            Q_close = self.ice_flux
        self.closure_ice_flux = Q_close
        return run_modeB_kernel(
            zb_flat, H_flat, Q_close, self.water_flux,
            self._law_code, self._gp, dt, self.stack, self.receivers,
            self.nb_receivers, self.weights, self.lengths, self.shape,
            self._dx_cell, self._dy_cell,
            self._border_bed_uplift(),
            self.bl, self.flotation_gate, self.flotation_ramp,
            self.parallel_erode, self._wrap_y, self._wrap_x,
            self._resolve_slope_normalise())

    @xs.runtime(args=("step_delta",))
    def run_step(self, dt):
        dt = _scalar_dt(dt)                   # numpy-2-safe (audit m1)
        # Post-uplift bed (kernel contract): persisted state + this step's uplift.
        # The sum is a fresh array, so zb_flat is safe to mutate in place.
        zb_flat = (self.bed_state + self.bed_uplift).flatten()
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
    """Mode C erosion: the citizen mode-B bed-and-thickness class
    (:class:`GlacialSPLModeB`)
    plus the sub-grid glacier-width carve. Fills the ``glacial_spl`` slot for
    ``mode == 'B'`` with ``carve_width`` on (the default siim2d path).

    Everything the citizen does is inherited: ``topography`` IS the tracked bed
    ``zb`` (bed memory; no flicker, no ``bedrock_surface`` output), the router
    routes on the reconstructed ice surface or the fabricated trunk V (mode C's
    default, :class:`TrunkSurfaceToErode`) while the kernel rebuilds its own
    ``zs`` for every slope, and ``erosion == denudation == delta-zb``
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

    _carves = True
    carve_width = xs.variable(default=True, description=_CARVE_WIDTH_DOC)

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
        """Apply the sub-grid width carve to the post-kernel bed ``zb_flat`` IN
        PLACE — thin wrapper over the framework-free
        :func:`siim._core.step.carve_bed`, passing the persistent per-step scratch
        buffers. ``zb_pre`` is the pre-kernel bed (the denudation datum + the
        descent-cap origin, so kernel erosion and the carve arbitrate — never
        add); ``surface_out`` the kernel's reconstructed ice surface (updated for
        carved cells, discarded by the citizen)."""
        ny, nx = int(self.shape[0]), int(self.shape[1])
        self._ensure_carve_buffers(ny, nx)
        carve_bed(
            zb_flat, H_flat, surface_out, zb_pre, self.receivers,
            self._gp.alpha_g, self._hc_over_H, self._widening_factor,
            self.shape, self._dx_cell, self._dy_cell,
            self._wrap_y, self._wrap_x,
            self._c_offsets, self._c_D, self._c_SRC, self._c_zb_kern)

    @xs.runtime(args=("step_delta",))
    def run_step(self, dt):
        dt = _scalar_dt(dt)             # numpy-2-safe (audit m1)
        # Citizen mode-B run_step (bed = the persisted topography state) with the
        # sub-grid carve applied to the bed after the kernel. carve OFF ->
        # GlacialSPLModeB.run_step bit-for-bit.
        zb_flat = (self.bed_state + self.bed_uplift).flatten()   # post-uplift bed (kernel contract)
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

    i.e. the ice surface the flow router stacks on (unless a ``router_surface``
    provider such as :class:`TrunkSurfaceToErode` replaces it), the mass balance
    reads and the hillslope diffusion acts on. The uplift add is inherited (no
    explicit ``+uplift``); H is
    the step-start (lagged) ``ice_thickness`` global, consistent with the order
    ``surf2erode -> glacial_flow -> glacial_spl``. ``hc_over_H`` is read from the
    law record via :class:`GlacialLaw`. (The pre-uplift climate surface
    ``topo + hc_over_H*H`` is recovered in :class:`GlacialFlowAccumulator` as
    ``z_clim = surface - surface_upward``.)

    The mode-B kernel reconstructs its own ``zs = zb + hc*H`` from the tracked
    bed + H and never reads this ``elevation`` (FIREWALL: see
    ``siim._core.driver``).
    """

    hc_over_H = xs.foreign(GlacialLaw, 'hc_over_H', intent='in')
    ice_thickness = xs.global_ref('ice_thickness', intent='in')

    def run_step(self):
        # SurfaceAfterTectonics.run_step sets elevation = topo + forced_motion
        # (= post-uplift bed). Add the reconstructed ice column hc_over_H*H.
        super().run_step()
        self.elevation = routing_surface(self.elevation, self.hc_over_H,
                                         self.ice_thickness)


# The dip input of both trunk-surface classes (they differ only in its default).
_TRUNK_DIP_K_DOC = (
    "Cross-valley dip slope factor k_dip: S_c = k_dip * S_v(s), S_v the local "
    "down-valley surface slope; D8 convergence needs k_dip > 0.414. The "
    "default is router-derived: 1.0 (constants.TRUNK_DIP_K_MFD) when "
    "glacial_processes builds the model for routing='mfd', 0.6 "
    "(constants.TRUNK_DIP_K) otherwise; an explicit value wins.")


@xs.process
class TrunkSurfaceToErode(GlacialSurfaceToErode):
    """Fabricated trunk-surface routing (mode B/C, ``trunk_surface``; on by
    default for mode C, as part of the mode-C standard).

    A trunk glacier of mean thickness H occupies width ``W = alpha_g*H`` —
    routinely several cells — but the flow graph carries its ice down 2-3
    *parallel* chains, each with its own diluted ``ice_flux`` and its own closure
    H, understating the trunk H. This provider replaces the plain reconstructed
    ice surface (``zb + hc*H``, :class:`GlacialSurfaceToErode`) with a
    **fabricated** surface that has a LINEAR cross-valley dip toward the trunk
    centerline, published as ``routing_elevation`` in the ``router_surface``
    group, which the in-house routers route on in place of ``elevation``. Flow
    converges onto the centerline chain — its raw accumulated flux then IS the
    full cross-section discharge (no accounting correction needed).
    ``elevation`` stays the dynamic ``zb + hc*H``.

    Fabrication (per-step, from LAGGED closure-H — this provider runs before the
    router). The power transform the carve uses gives footprint membership
    ``D < 0`` and per-cell source ``SRC`` (the thickest disc wins — the
    medial-axis-transform inversion). For a footprint cell ``i`` with source
    ``s`` (whose disc radius ``R_s = alpha_g*H_s/2 > cell``)::

        zs_geo(i) = zs_dyn(s) + S_c(s) * (d_i - R_s)

    with ``d_i`` the distance to the source (``d_i^2 = D_i + R_s^2``, from the
    transform), ``zs_dyn(s) = zb_s + hc*H_s`` the source's own dynamic surface,
    and ``S_c(s) = trunk_dip_k * max(|grad zs_dyn|(s), TRUNK_DIP_FLOOR)`` the
    cross-slope (0 at the rim ``d = R_s``, ``-S_c*R_s`` at the axis). The routing
    value inside a footprint is ``max(zs_geo, zb)`` — bare cells / nunataks
    present their rock (routing goes around them); a footprint cell's own
    dynamic ice column is deliberately IGNORED (its ice is part of the trunk;
    taking a max with it would re-erect a spine). Outside footprints the surface
    is the dynamic ``zb + hc*H`` unchanged.

    The fabricated surface builds the router graph alone (FIREWALL: see
    ``siim._core.driver``). A flank cell routed to a taller centerline hits the
    well-posed negative-``a`` (from-a) branch, bounded by flux. The dip is a
    routing numerics device (like the priority-flood eps), not physics.

    The dip is the ``trunk_dip_k`` input, resolved and validated at initialize
    by :func:`siim._core.router.resolve_trunk_dip_k` against the class's
    ``_dip_routing`` (never by reading the router, which would make the cycle
    ``surf2erode -> flow -> surf2erode``): ``None`` is the router's default, an
    explicit value (siim2d passes the resolved number) wins, and a NaN or
    negative dip is refused.
    """

    _dip_routing = 'single'    # the dip default's router (resolve_trunk_dip_k)

    alpha_g       = xs.foreign(GlacialLaw, 'alpha_g', intent='in')
    shape         = xs.foreign(RasterGrid2D, 'shape')
    _dx_cell      = xs.foreign(RasterGrid2D, 'dx')
    _dy_cell      = xs.foreign(RasterGrid2D, 'dy')
    border_status = xs.foreign(BorderBoundary, 'border_status')
    routing_elevation = xs.variable(
        dims=('y', 'x'), intent='out', groups='router_surface',
        description="The fabricated trunk V, the surface the in-house routers "
                    "route on (the router graph alone)")
    trunk_dip_k = xs.variable(default=_constants.TRUNK_DIP_K,
                              description=_TRUNK_DIP_K_DOC)
    trunk_dip_floor = xs.variable(
        default=_constants.TRUNK_DIP_FLOOR,
        description="Down-valley slope floor (m/m) for flat reaches, so the dip "
                    "still beats routing ties; default constants.TRUNK_DIP_FLOOR.")

    def initialize(self):
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
        # finite and >= 0, None = the router-derived default (audit #46)
        self._trunk_dip_k = resolve_trunk_dip_k(self.trunk_dip_k, self._dip_routing)
        self._D = np.empty((ny, nx))
        self._SRC = np.empty((ny, nx), dtype=np.int64)

    def run_step(self):
        # 1. Dynamic surface zs_dyn = post-uplift bed + hc*H (the parent).
        super().run_step()
        ny, nx = int(self.shape[0]), int(self.shape[1])
        hc = float(self.hc_over_H)
        zs_dyn = np.asarray(self.elevation, dtype=np.float64).reshape(ny, nx)
        H = np.asarray(self.ice_thickness, dtype=np.float64).reshape(ny, nx)
        zb = zs_dyn - hc * H                         # post-uplift bed (exact)
        # 2. Fabricate the V-dipped trunk surface (see the helper) as the
        #    router's input; ``elevation`` stays zs_dyn.
        self.routing_elevation = _fabricate_trunk_surface(
            zs_dyn, zb, H, self._border.reshape(ny, nx), float(self.alpha_g),
            float(self._dx_cell), float(self._dy_cell),
            self._trunk_dip_k, self.trunk_dip_floor,
            self._offsets, self._D, self._SRC, self._wrap_y, self._wrap_x)


@xs.process
class _MfdTrunkSurfaceToErode(TrunkSurfaceToErode):
    """:class:`TrunkSurfaceToErode` with the MFD dip default
    (``constants.TRUNK_DIP_K_MFD``): :func:`siim.fastscape.glacial_processes`
    fills the ``surf2erode`` slot with it under ``routing='mfd'``. The input
    keeps its name, ``surf2erode__trunk_dip_k``."""

    _dip_routing = 'mfd'

    trunk_dip_k = xs.variable(default=_constants.TRUNK_DIP_K_MFD,
                              description=_TRUNK_DIP_K_DOC)


@xs.process
class SedimentTracker:
    """Optional sediment-throughput tracker (added only when siim's
    ``track_sediment`` is on — zero cost otherwise).

    Routes each step's denuded rock volume — ``max(denudation, 0) * cell_area`` —
    down the flow graph in one accumulation pass, giving per node the total
    upstream-eroded volume passing through it that step (``flux``, m^3) and its
    running time-integral (``cumulative``, m^3). Detachment-limited bookkeeping
    only: no deposition, every node passes all upstream sediment through.
    Difference ``cumulative`` along the time axis to recover per-interval volumes;
    the outlet node's ``cumulative`` is the whole-basin yield.

    ``edge_flux`` / ``edge_cumulative`` report the same routed flux summed over
    each domain-edge outlet ring instead (dims ``('side',)``, the
    ``border_status`` order left/right/bottom/top, NaN off ``'fixed_value'``
    sides) — one shared sum (:func:`siim._core.step.edge_sediment`) with the
    in-house driver. Both reports ride the one accumulation pass; siim's
    ``track_sediment`` value picks which are stored, and clears ``basin`` when
    only the edge totals are wanted so the per-node running integral is skipped.

    Reads the erosion process's ``denudation`` (via ``GlacialSPLBase``: the true
    rock removed — delta-zb incl. sub-grid carve in mode B, delta-zs in mode A;
    per-step, already includes dt) so it runs after the erosion step and reuses
    the same receiver/stack flow graph.
    """
    denudation   = xs.foreign(GlacialSPLBase, 'denudation',  intent='in')
    cell_area    = xs.foreign(RasterGrid2D, 'cell_area')
    shape        = xs.foreign(RasterGrid2D, 'shape')
    border_status = xs.foreign(BorderBoundary, 'border_status')
    receivers    = xs.foreign(FlowRouter,   'receivers',    intent='in')
    nb_receivers = xs.foreign(FlowRouter,   'nb_receivers', intent='in')
    weights      = xs.foreign(FlowRouter,   'weights',      intent='in')
    stack        = xs.foreign(FlowRouter,   'stack',        intent='in')

    basin      = xs.variable(intent='in', default=True,
                             description='Keep the per-node running integral (cumulative); False reports only the per-edge totals')
    flux       = xs.variable(dims=('y', 'x'), intent='out',
                             description='Upstream-eroded volume routed through each node this step (m^3)')
    cumulative = xs.variable(dims=('y', 'x'), intent='out',
                             description='Running time-integral of flux per node (m^3)')
    edge_flux       = xs.variable(dims=('side',), intent='out',
                                  description='Volume leaving the domain across each edge this step (m^3); NaN off fixed_value sides')
    edge_cumulative = xs.variable(dims=('side',), intent='out',
                                  description='Running time-integral of edge_flux per domain edge (m^3)')

    def initialize(self):
        self._cum = np.zeros(self.shape, dtype=np.float64)
        self._edge_cum = np.zeros(4)

    def run_step(self):
        self.flux = accumulate_sediment(
            self.denudation, self.cell_area, self.stack, self.receivers,
            self.nb_receivers, self.weights, self.shape)
        if self.basin:
            # New array each step (not in-place) so prior snapshots stay valid.
            self._cum = self._cum + self.flux
        self.cumulative = self._cum
        self.edge_flux = edge_sediment(self.flux, self.border_status)
        self._edge_cum = self._edge_cum + self.edge_flux
        self.edge_cumulative = self._edge_cum


@xs.process
class GlacialFlexure(Flexure):
    """Flexural isostasy with true glacial isostatic adjustment: ice loading +
    erosional/tectonic unloading + elastic rebound.

    fastscape's :class:`~fastscape.processes.Flexure` loads the elastic plate from
    ``TotalErosion.height`` (the erosion group, which adds the hillslope diffusion
    of the ice surface ``zb + hc_over_H*H`` in mode B to the rock removed). This
    override does two things:

    1. Re-sources the *rock* load from the erosion process's per-step ``denudation``
       (via :class:`GlacialSPLBase`; mode B: delta-zb incl. sub-grid carve; mode A:
       delta-zs), so the plate flexes in response to rock erosional + tectonic
       unloading, correct under transient ice.
    2. Adds the per-step ICE load: each channel's mass-conserving, hc-FREE
       cross-section alpha_g*H**2 (= Qg/V, the ice flux over the channel velocity)
       spread over the cells it covers (:func:`siim._core.step.footprint_ice_column`,
       each cell counted once; ED-184). Its per-step change folds in as a
       rock-equivalent column, (rho_ice/lithos_density)*d(col), into the ``diff`` that
       drives fastscape's incremental plate solve (incremental: the full column would
       re-apply the whole load every step).
       Gated by ``ice_load`` (default True; False = erosion-only, the pre-GIA behaviour).
       The mass-conserving Qg/V load is used in preference to a per-cell thickness
       because siim does not resolve 3D ice flow: sheet accumulation Qg = area*balance
       is trustworthy, the unresolved per-cell H is not (see DECISIONS 2026-06-16).

    The plate solve, densities, boundary handling and rebound feedback are inherited
    unchanged. The ice LOAD is applied everywhere ice is present, including afloat cells
    (zs = zb + hc*H < bl): the waterline-flotation gate
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
    border_status = xs.foreign(BorderBoundary, 'border_status')
    ice_load = xs.variable(default=_constants.ICE_LOAD, description="Include the per-step glacial ice "
        "load (each channel's cross-section alpha_g*H**2 over its footprint) in the flexural plate "
        "load (true glacial isostatic adjustment). False = erosional/tectonic unloading "
        "only (the pre-GIA behaviour; isolate the ice contribution or reproduce runs "
        "saved before the ice load).")
    numerics_backend = xs.variable(
        default=_constants.NUMERICS_BACKEND,
        description="Plate-solve backend: 'inhouse' (siim scipy.fft native-grid "
        "solve, fixes the fortran pihy anisotropy bug) — the only accepted "
        "value since the 0.9.1 standalone flip. See constants.NUMERICS_BACKEND.")

    def initialize(self):
        self._col_prev = np.zeros(self.shape)
        bs = list(self.border_status)          # the footprint wraps looped axes
        self._wrap_x = bs[0] == 'looped'
        self._wrap_y = bs[2] == 'looped'

    def _flexure_solve(self):
        """Return the biharmonic plate solve callable, validating the backend
        value (a typo like 'fortan' must raise the same ValueError the siim2d /
        glacial_processes entry points give, not silently run in-house). The
        in-house solve keeps step.py framework-free; the retired 'fortran'
        fs.flexure arm was deleted at the 0.9.1 standalone flip."""
        if self.numerics_backend != 'inhouse':
            raise ValueError("numerics_backend must be 'inhouse' (the retired "
                             "'fortran' backend was removed at the 0.9.1 "
                             f"standalone flip), got {self.numerics_backend!r}")
        return _inhouse_flexure

    def run_step(self):
        # The biharmonic plate solve is the injected seam (fortran fs.flexure or
        # the in-house scipy.fft solve); everything else is framework-free.
        self.rebound, self._col_prev = glacial_flexure_step(
            self.elevation, self.erosion, self.surface_upward,
            self.ice_thickness, self.alpha_g, self.cell_area,
            self.lithos_density, self.asthen_density, self.e_thickness,
            self.ibc, self.shape, self.length, self._col_prev, self.ice_load,
            self._flexure_solve(), wrap_y=self._wrap_y, wrap_x=self._wrap_x)


@xs.process
class HillslopeDiffusion:
    """In-house linear hillslope diffusion (ADI), the standalone replacement for
    fastscape's stock ``LinearDiffusion`` (fortran ``fs.diffusion``).

    Same input/output contract as ``LinearDiffusion`` — reads
    ``SurfaceToErode.elevation`` (the dynamic surface; FIREWALL: see
    ``siim._core.driver``) + the ``diffusivity`` input, writes the per-step
    ``erosion`` into the ``"erosion"`` group — so it drops into the ``'diffusion'``
    model slot when ``numerics_backend='inhouse'``. The unconditionally-stable ADI
    solve (:func:`siim._core.hillslope.diffuse`) is bit-for-bit with the fortran
    for the uniform ``kd`` siim uses on grids with no looped axis; a looped axis
    is periodic there (the fortran made it a no-flux wall). Boundary handling
    keys off BorderBoundary's ``ibc`` code and ``border_status`` (the looped
    axes), read directly rather than via the fortran context.
    """

    diffusivity = xs.variable(
        dims=[(), ('y', 'x')], description="diffusivity (transport coefficient)")
    erosion = xs.variable(dims=('y', 'x'), intent='out', groups='erosion')

    shape = xs.foreign(RasterGrid2D, 'shape')
    length = xs.foreign(RasterGrid2D, 'length')
    ibc = xs.foreign(BorderBoundary, 'ibc')
    border_status = xs.foreign(BorderBoundary, 'border_status')
    elevation = xs.foreign(SurfaceToErode, 'elevation')

    def initialize(self):
        bs = list(self.border_status)          # [left, right, top, bottom]
        self._wrap_x = bs[0] == 'looped'
        self._wrap_y = bs[2] == 'looped'

    @xs.runtime(args='step_delta')
    def run_step(self, dt):
        ny, nx = self.shape
        yl, xl = self.length
        diffused = _inhouse_diffuse(
            self.elevation, self.diffusivity, _scalar_dt(dt), nx, ny, xl, yl, self.ibc,
            self._wrap_y, self._wrap_x)
        self.erosion = self.elevation.reshape(self.shape) - diffused


@xs.process
class _InhouseRouterShell(FlowRouter):
    """Shared shell for the two fortran-free in-house routers (:class:`D8FlowRouter`,
    :class:`DinfFlowRouter`): the ``@xs.process`` adapter over a framework-free
    ``_core.step`` producer. Sets the shared fortran-context surface (as the base
    ``FlowRouter`` does, for the co-resident stock diffusion/flexure) but SKIPS
    the fortran routing call — the interior mask (from ``border_status``), the
    directions, and ``basin`` are all in-house (S4, Map 3 §3-§4). Donors are
    declared by the base ``FlowRouter`` but unconsumed by siim (grep-verified),
    so they are zeroed to keep the (unused) foreign targets well-formed. Basin is
    overridden as an on_demand returning the in-house outlet labeling.

    The directions come off the ``router_surface`` group's member when a
    process publishes one (:class:`TrunkSurfaceToErode`'s V), else off
    ``elevation`` (FIREWALL: see ``siim._core.driver``)."""

    router_surface = xs.group('router_surface')
    dx = xs.foreign(RasterGrid2D, 'dx')
    dy = xs.foreign(RasterGrid2D, 'dy')
    border_status = xs.foreign(BorderBoundary, 'border_status')

    # Override the base FlowRouter's fortran-catch on_demand with the in-house
    # (deterministic, outlet-index) basin labeling produced by the router.
    basin = xs.on_demand(dims=('y', 'x'), description='river catchments')
    # the base FlowRouter's lake_depth reads a fortran field no in-house router
    # fills, so a requested output came back all zeros (audit #48)
    lake_depth = xs.on_demand(dims=('y', 'x'), description='not computed by the in-house routers')

    def _route(self):
        """Return ``(receivers, weights, lengths, nb_receivers, stack, basin)``.
        Implemented by the concrete SFR / D-inf subclass."""
        raise NotImplementedError

    def _router_input(self):
        """The surface the directions come off (see the class docstring)."""
        surfaces = list(self.router_surface)
        if len(surfaces) > 1:
            raise ValueError("at most one process may publish a "
                             f"'router_surface' variable, got {len(surfaces)}")
        return next(iter(surfaces), self.elevation)

    def run_step(self):
        # Keep the shared fortran-context surface in sync (the base FlowRouter
        # sets it too), but do NOT run the fortran router.
        # fs_context['h'] stays the dynamic ``elevation``: no consumer reads it.
        self.fs_context['h'] = self.elevation.ravel()
        (self.receivers, self.weights, self.lengths,
         self.nb_receivers, self.stack, self._basin_ids) = self._route()
        nn = self.shape[0] * self.shape[1]
        self.nb_donors = np.zeros(nn, dtype=int)
        self.donors = np.full((nn, 1), -1, dtype=int)

    @basin.compute
    def _basin(self):
        return self._basin_ids

    @lake_depth.compute
    def _lake_depth(self):
        raise NotImplementedError(
            "flow__lake_depth is not computed by siim's in-house routers (they route "
            "across closed basins on an eps-filled surface); drop it from output_vars")


@xs.process
class D8FlowRouter(_InhouseRouterShell):
    """In-house D8 single-flow router (S4) — the framework-free replacement for
    fastscape's fortran ``SingleFlowRouter`` (``fs.flowroutingsingleflowdirection``).
    Fills the ``flow`` slot when ``routing='single'`` and
    ``router_backend='inhouse_d8'``. Produces the SFR bundle (1D receivers /
    lengths, all-ones weights / nb_receivers, outlet-first stack) via
    :func:`siim._core.step.route_d8` on the eps-filled surface; basin labeled by
    outlet index. The routing delta vs the fortran SFR is confined to
    depression/tie cells (behavioral/attractor gate)."""

    def _route(self):
        return route_d8(self._router_input(), self.shape, self.dx, self.dy,
                        self.border_status)


@xs.process
class DinfFlowRouter(_InhouseRouterShell):
    """D-infinity flow router :cite:p:`tarbotonNewMethodDetermination1997`.

    Subclasses the in-house router shell — fully fortran-free (S4): the
    interior/boundary mask comes from ``border_status`` (Map 3 §4, provably
    identical to the old fortran ``sfr_rec != i`` mask) and ``basin`` from the
    in-house outlet labeling, so the last ``fs.flowrouting()`` call is gone.
    Outputs use the (n_nodes, 2) receiver/weight shape consumed by
    GlacialFlowAccumulator and the mode-B / mode-A D-inf kernels.

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
    Looped boundaries wrap (fill and facet scan both). See
    ``docs/guides/concepts.md`` for the public routing overview.
    """

    def _route(self):
        return route_dinf(self._router_input(), self.shape, self.dx, self.dy,
                          self.border_status)


@xs.process
class MfdFlowRouter(_InhouseRouterShell):
    """Multiple-flow-direction router — the mirror of
    :class:`DinfFlowRouter` over :func:`siim._core.step.route_mfd`.

    Splits a cell's flux over ALL strictly downhill 8-neighbours of the same
    eps-filled surface, with weights ``(S_k/S_max)**mfd_exponent`` normalised to
    1. Outputs use the ``(n_nodes, 8)`` receiver/weight shape; the base
    ``FlowRouter``'s dimensions accept any ``nb_rec_max``, and the siim consumers
    all loop ``nb_receivers``, so nothing downstream changes shape-wise.

    Built for the ice-sheet regime, where D8/D-inf make the flow graph a step
    function of the surface and the mode-B loop limit-cycles; it runs in
    valleys too (mode C, the carve and :class:`TrunkSurfaceToErode`, dip per
    :func:`siim._core.router.resolve_trunk_dip_k`). The coherence guards live in
    :func:`siim._core.router.check_mfd_config` and are enforced here (square
    cells), in :class:`siim.fastscape.GlacialSPLBase` (transect closure) and in
    :func:`siim.fastscape.glacial_processes` (mode).
    ``basin`` is the D8 chain on the same filled surface, as for D-inf, and may
    disagree with the MFD argmax graph; ``receivers_2d`` is the argmax-weight
    receiver, a PRIMARY-PATH diagnostic (see ``docs/guides/outputs_and_io.md``).
    """

    mfd_exponent = xs.variable(
        default=_constants.MFD_EXPONENT,
        description=f"MFD spreading exponent p in w_k = (S_k/S_max)^p; positive, finite up to {_constants.MFD_EXPONENT_MAX:g} or inf (see constants.MFD_EXPONENT_MAX; a larger finite value is routed as inf, the steepest link alone, with a warning); p = 1 (the default) is Budd & Warner's split and the only exponent whose weighted receiver vector is exactly parallel to the surface gradient on a square 8-stencil.")

    def _route(self):
        _check_mfd_config('mfd', dx=self.dx, dy=self.dy)
        return route_mfd(self._router_input(), self.shape, self.dx, self.dy,
                         self.border_status, self.mfd_exponent)
