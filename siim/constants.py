"""Shared physical constants and per-sliding-law parameter dispatch.

Single source of truth for the numerical models (siim1d, siim2d, the
standalone fastscape process module) and the analytical machinery
(siim.analytical). Imports nothing heavier than numpy, so the lightweight
analytical layer can depend on it without dragging in the model stack.
"""
from collections import namedtuple
import math
import warnings

# --- Physical constants ---------------------------------------------------
RHO_ICE = 920.0                  # ice density [kg m^-3]
GRAVITY = 9.8                    # gravitational acceleration [m s^-2]
RHO_ICE_G = RHO_ICE * GRAVITY    # rho_ice * g [Pa m^-1]
KT = 365.25 * 24 * 3600.0        # seconds per year [s yr^-1]

# Centerline-to-mean depth ratio hc/H for the sub-grid channel cross-section
# (parabola: max/mean = 3/2). Every surface the models BUILD from bed-and-
# thickness state uses zs = zb + HC_OVER_H * H, i.e. the tracked bed is the
# channel floor (thalweg) and H stays the width-MEAN depth that drives the
# flux closures, sliding laws and erosion laws (which are untouched by this
# constant). Setting the ratio to 1.0 recovers the historical mean-bed datum.
# See ``docs/guides/concepts.md`` for the public datum convention.
HC_OVER_H = 1.5

# --- Package-wide parameter defaults ---------------------------------------
# Single source of truth: the model default dicts (siim1d, siim2d, the fastscape
# process) and the analytical machinery all import these — never re-literal them.
KH = 5.0          # Hack coefficient (catchment area A = kh * x^d)
BETA = 1e-2       # mass-balance gradient [yr^-1]
LAMBDA_C = 1e3    # coulomb sliding length [m]
TAU_C = 1e5       # coulomb yield stress [Pa]
ALPHA_G = 5.0     # valley width-to-thickness ratio (W = alpha_g * H)
AC = 2.5e-24      # Glen's flow-law coefficient A [Pa^-3 s^-1] (canonical ~2.4e-24
                  # at 0 C; 2.5e-24 preserves the package's historical cg, which
                  # previously absorbed the deformation factor 2 into Ac=5e-24)
CE = 1e-5         # glacial erosion coefficient
WIDTH_HACK_K = 0.5  # glacier-width Hack coefficient (W = width_hack_k * A^width_hack_p)
WIDTH_HACK_P = 0.5  # glacier-width Hack exponent
KO = 1e-6         # fluvial erodibility [m^(1-3m) yr^(m-1)]
N_FLUVIAL = 1     # fluvial slope exponent (E_f ~ S^n)
NU = 2            # glacial slope exponent (primary glacial input)
D_HACK = 1.8      # Hack's-law exponent (catchment area A = KH * x^D_HACK)
XO = 300.0        # head-catchment reference length x_o [m] (single-sourced
                  # fixed default for siim1d + the analytical steady_state;
                  # GeneralProfile keeps its own relative L/1000 default)
SIGMA = 0.45      # tributary contribution parameter (0 = no local, 1 = no tributaries)
LAMBDA_P = 300.0  # eff-exp / power critical ice thickness [m]
K_ACCUM = 1.0     # accumulation-profile shape exponent (paper's k)
ZELA = 1500.0     # equilibrium-line altitude [m]
U = 1e-3          # rock uplift rate [m yr^-1]
P = 1             # precipitation / runoff rate [m yr^-1]
BL = 0.0          # base level: the water-line (Dirichlet) datum [m]. The
                  # mode-B kernels reference this as the waterline everywhere a
                  # literal 0 used to stand (erosion-view floor, lake-fill seed,
                  # border-bed clamp, flotation reference, recovery threshold).
                  # Scalar or a length-nt series (like zELA/U/P).
# Global waterline-flotation gate (mode B). grounded <=> zs = zb + hc*H >= bl
# (rho_i = rho_w); a sub-waterline icy cell does reduced/no glacial erosion
# (and, via E_c <= 0, carves nothing). INTERIOR: applied to the erosion delta
# (largely latent in 1D, mildly behavioral in 2D). BORDER: the ramp is the
# physical bound inside the closed-form IMPLICIT border budget (the bed digs
# on the arrival slope to the flotation-draft equilibrium
# zb* = bl - hc*H + delta*U/E) — turning
# the gate off un-bounds the border (f == 1: measured runaway). Off is for
# diagnostics only.
# See ``docs/guides/concepts.md`` for the boundary-condition contract.
FLOTATION_GATE = True
# Flotation-gate ramp width gamma (the effective-pressure softening of the
# gate). Instead of a hard on/off switch at the
# waterline, glacial erosion is multiplied by
#     f = clip((zs - bl) / (gamma*hc*H), 0, 1),   f = 0 exactly for zs <= bl
# — the freeboard-proportional factor an effective-pressure erosion law gives
# under rho_i = rho_w (N ∝ zs - bl). This removes the binary gate's
# single-step explicit overshoot at the toe (a grounded cell taking a full
# ~E*dt plunge below bl in one step); the toe instead settles to a standoff
# height ~ delta*U/E above bl (delta = gamma*hc*H). gamma = 0 is the hard
# binary gate, bit-for-bit. Recommended 0.05-0.1; gamma <= 0.2 is the SAFE
# CEILING — 0.2 already under-carves the coulomb flotation draft, and wide
# ramps can tip dome-prone (cap_ice_accumulation=False) configs into a
# permanent whole-domain dome. At the BORDER the ramp is solved implicitly
# (closed form) inside the border budget, where gamma = 0 degenerates to the
# binary gate's Filippov sliding mode (bed pinned at flotation) — well-defined
# at any dt. Only active when FLOTATION_GATE is on.
FLOTATION_RAMP = 0.1
# Arrival-slope floor [m/m] for the outflow border-bed erosion. The border bed
# erodes on the interior flow slope the through-flowing ice CARRIES (one cell
# inside the border), NOT the ~0 local border slope that zero-gradient-H
# produces (which would starve erosion and re-dam the border — the sill /
# reflected-slope failure family). Baked in, not user-facing.
S_FLOOR_BC = 1e-3
# Level-scheduled parallel mode-B erosion: the
# eroder dependency is a DAG depth, so topological levels run in order with
# prange inside each — BIT-FOR-BIT identical to the serial walk at any thread
# count (pinned by test_parallel_erode). Default ON (Eric, 2026-07-07; it is
# purely a scheduling change — SFR coulomb steps measured -43%). False = the
# serial eroder, for single-core environments or debugging thread issues.
PARALLEL_ERODE = True
COULOMB_CLAMP = 1e-12          # regularized-Coulomb H-solver: min relative gap from the pole
DEFAULT_SLIDING_LAW = 'power'  # 'eff-exp', 'power', or 'coulomb' (every front-end default)
SLIDING_LAWS = ('eff-exp', 'power', 'coulomb')   # the accepted sliding_law values
BOUNDARY_CONDITIONS_1D = ('base_level', 'reflecting')   # 1D left_bc / right_bc
# Numerics backend for the flexure + hillslope-diffusion operators (2D):
# 'inhouse' = siim's scipy.fft plate solve (_core.flexure) + numba ADI diffuser
# (_core.hillslope) — the ONLY accepted value since the S5 standalone flip
# (the fortran arms are deleted; the param survives as the numerics plug
# point). The ADI is bit-for-bit with the retired fs.diffusion for uniform kd;
# the FFT flexure is native-grid (fixes the fortran pihy anisotropy bug) and
# validated against the closed-form Kelvin point-load solution.
NUMERICS_BACKEND = 'inhouse'
# 2D time-loop driver. 'inhouse' (default since the S5 standalone flip) =
# siim's own framework-free time loop (_core.driver); 'xsimlab' = the OPTIONAL
# fastscape/xsimlab adapter orchestration (conda env only — the guarded
# siim.fastscape shells calling the SAME step functions, packing the identical
# ds_out). The two are bit-for-bit on the same backend, gated by
# test_driver_vs_xsimlab (adapter CI job).
DRIVER_DEFAULT = 'inhouse'
# 2D flow-routing backend: 'inhouse_d8' = siim's numba D8 fill-then-route
# producer (_core.step.route_d8 / _core.routing._d8_*) — the ONLY accepted
# value since the standalone-driver transition (the fortran SFR arm is deleted).
# The param survives as the ROUTER-CONTRACT plug point: a future backend
# (e.g. fastscapelib) re-widens the
# accepted set. The S4 router swap was the one NON-bit-for-bit numeric change
# of the standalone migration (tie-break/basin-carve paths differ from the
# retired fortran SFR; gated behaviorally by test_d8_receiver_parity vs the
# frozen raw fortran refs + the router contract battery).
ROUTER_DEFAULT = 'inhouse_d8'

# 2D initial surface (siim2d). With no initial_topography, a tent from 0 at the
# fixed_value edges up to INITIAL_MAX_ELEVATION; uniform tie-breaking noise of
# NOISE_AMPLITUDE goes on top of either surface (zeroed on fixed_value edges).
INITIAL_MAX_ELEVATION = 1000  # tent peak (m)
NOISE_AMPLITUDE = 100         # noise amplitude (m)
# Grid of siim2d's 1D analytical reference profile (fixed; the user has no override).
ANALYTICAL_NX_2D = 3000

# 2D hillslope and flexure defaults (siim2d; the adapter's flexure process).
HILLSLOPE_D = 1e-3           # hillslope diffusivity D (m^2/yr)
ICE_LOAD = True              # flexure: load the plate with the ice, not only the rock unloading
LITHOS_DENSITY = 2800        # lithospheric rock density (kg/m^3)
ASTHEN_DENSITY = 3200        # asthenospheric density (kg/m^3)
E_THICKNESS = 35e3           # effective elastic plate thickness Te (m)

# 2D outputs (siim2d). The closure's own (flux, thickness) rows make ice_speed exact;
# they add ~10 % to the saved output in modes B/C and ~19 % in mode D, so they are opt-in.
SAVE_CLOSURE_PAIR = False

# Escarpment defaults (siim.escarpment; the adapter's WaveUplift / PlateauSurface).
WAVE_WIDTH = 200e3           # 1/e half-width of the uplift wave (m)
WAVE_VELOCITY = 15e-3        # wave propagation velocity (m/yr)
X_ESCARPMENT = 0.0           # initial wave-centre position from the left edge (m)
WAVE_CALIBRATION = 1.0       # multiplier on the wave's peak rate (1 deposits delta_h per passage)
U_INF = 0.0                  # steady background uplift under the wave (m/yr)
PLATEAU_ZO = 1000.0          # plateau elevation of escarpment.plateau_topography (m)
PLATEAU_DZ = 1.0             # slope across the plateau (m); seeds the divides
PLATEAU_FRAC = 0.8           # fraction of x the plateau occupies
PLATEAU_W = 10e3             # escarpment transition width (m)


def cg_prefactor(alpha_g=ALPHA_G, Ac=AC, rho_ice=RHO_ICE, g=GRAVITY):
    """Glacial rheology group cg = alpha_g * kt * (2*Ac/5) * (rho_ice*g)^3
    [m^-3 yr^-1] (the s/yr conversion kt is absorbed). The 2/5 is the
    depth-integrated Glen deformation prefactor 2A/(n_c+2) with n_c = 3
    (model-paper convention; Ac is the physical Glen coefficient)."""
    return alpha_g * KT * (2.0 * Ac / 5.0) * (rho_ice * g) ** 3


def unwrap_0d(params):
    """``params`` with every 0-d array replaced by its Python scalar, so each
    scalar-or-array parameter reads one as the scalar it is (audit #41)."""
    import numpy as np                      # constants itself stays numpy-free at import
    return {k: v.item() if isinstance(v, np.ndarray) and v.ndim == 0 else v
            for k, v in params.items()}


def check_range(name, value, lo, *, inclusive=False, hi=None):
    """Raise ValueError unless ``value`` (a scalar or a field) is finite
    everywhere, above ``lo`` (or at it, if ``inclusive``) and, when ``hi`` is
    given, below ``hi``. NaN fails every comparison, so a bare ``x < 0`` guard
    let it through (audit #38)."""
    import numpy as np                      # constants itself stays numpy-free at import
    rule = f"{'>=' if inclusive else '>'} {lo:g}" + ("" if hi is None else f" and < {hi:g}")
    try:
        a = np.asarray(value, dtype=float)
    except OverflowError:                   # an int beyond float range is not finite: the same refusal
        raise ValueError(f"{name} must be finite and {rule}; got an int too large for a float") from None
    ok = np.isfinite(a) & ((a >= lo) if inclusive else (a > lo))
    if hi is not None:
        ok &= a < hi
    if not ok.all():
        got = (f"got {value!r}" if a.ndim == 0
               else f"got {np.count_nonzero(~ok)} of {a.size} cells outside that")
        raise ValueError(f"{name} must be finite and {rule}; {got}")


# --- Per-sliding-law parameter dispatch -------------------------------------
LawConstants = namedtuple('LawConstants', 'ell nu mu phi Co')


def derive_coulomb(ce, alpha_g, tau_c, rho_g, g, *, nu=None, ell=None):
    """Coulomb law: nu = 2*ell, mu = ell.
    Co = ce * ((rho_g g)^2 / (alpha_g tau_c^2))^ell."""
    if ell is not None:
        nu_out, mu = 2.0 * ell, ell
    else:
        ell, nu_out, mu = nu / 2.0, nu, nu / 2.0
    Co = ce * ((rho_g * g) ** 2 / (alpha_g * tau_c ** 2)) ** ell
    return LawConstants(ell, nu_out, mu, mu / nu_out, Co)


def Co_power(ce, cg, lambda_p, alpha_g, mu):
    """Power/eff-exp erosion prefactor Co = ce * (cg^(5/4) * lambda_p^3 /
    alpha_g^(9/4))^mu. Single-sourced so an explicit mu override can recompute
    Co without re-literalling the formula (siim1d / analytical.steady_state,
    audit B5)."""
    return ce * (cg ** (5 / 4) * lambda_p ** 3 / alpha_g ** (9 / 4)) ** mu


def derive_power(ce, cg, lambda_p, alpha_g, *, nu=None, ell=None):
    """Power and eff-exp laws: nu = 5*ell/3, mu = 4*ell/9.
    Co = ce * (cg^(5/4) * lambda_p^3 / alpha_g^(9/4))^mu."""
    if ell is not None:
        nu_out, mu = 5.0 * ell / 3.0, 4.0 * ell / 9.0
    else:
        ell, nu_out, mu = 3.0 * nu / 5.0, nu, 4.0 * nu / 15.0
    Co = Co_power(ce, cg, lambda_p, alpha_g, mu)
    return LawConstants(ell, nu_out, mu, mu / nu_out, Co)


# --- Surface-evolution mode (user-facing names <-> internal A/B codes) -------
# The numerical models and numba kernels run on the single-letter codes
# 'A'/'B' internally; the user-facing `mode` parameter takes the legible names
# below, with 'A'/'B' kept as permanent aliases. 'bedrock+ice_thickness' (B) is
# the default.
#   'ice_surface'           (A): one tracked state  — the ice surface.
#   'bedrock+ice_thickness' (B): two tracked states — bedrock + ice thickness
#                                (bed memory: carved overdeepenings persist on
#                                ice retreat).
#   'C': a front-door alias (2D only) for the mode-B + sub-grid width-carving
#        configuration — NOT a new dynamical mode. siim2d resolves it to
#        internal mode 'B' with carve_width on (there is no long-name sibling).
#   'D': mode C with the valley <-> sheet reading deciding, per cell and
#        per step, whether the valley devices or the sheet treatment act (2D,
#        standalone driver only; see the REGIME_* and MODE_D_* blocks below).
#        siim2d resolves it to internal 'B' + carve, like 'C'.
MODE_ICE_SURFACE = 'ice_surface'
MODE_BEDROCK_ICE = 'bedrock+ice_thickness'
DEFAULT_MODE = MODE_BEDROCK_ICE
# 2D default is mode C (the flagship carved configuration). Carving lives only
# in 2D, so the shared DEFAULT_MODE stays plain mode B for 1D; siim2d uses this
# instead. An unqualified mode='B' does NOT carve — carving is opt-in, via
# mode='C' or an explicit carve_width=True (Eric, 2026-07-07).
DEFAULT_MODE_2D = 'C'

# --- Mode-C standard defaults -----------------------------------------------
# Mode C (mode B + sub-grid width carving — siim's flagship carved mode)
# defaults trunk-surface routing ON, so a standard carved run converges trunk
# flow onto the centerline (honest cross-section flux). Plain mode B (no carve)
# and mode A keep it OFF. siim2d and the fastscape facade (glacial_processes)
# both resolve the sentinel with mode_c_standard() after mode normalization; an
# explicit user value always wins. siim2d damps modes B, C and D with
# MODE_BCD_INNER_DAMPING (below).
MODE_C_TRUNK_SURFACE = True


def mode_c_standard(is_mode_c):
    """The mode-C standard ``trunk_surface`` default.

    ``MODE_C_TRUNK_SURFACE`` for mode C (mode B with sub-grid carving),
    ``False`` otherwise. Callers apply it only to a sentinel left at ``None``;
    an explicit user value wins.
    """
    return MODE_C_TRUNK_SURFACE if is_mode_c else False


# --- Mode-D standard defaults -----------------------------------------------
# Mode D = mode C + the valley <-> sheet reading (REGIME_* below) choosing valley
# or sheet rules per cell; decisions/2026-11-02-mode-d.md, 2026-11-03-mode-d-defaults.md
# and 2026-11-13-channel-sheet-discharge-reading.md (private). siim2d writes these into every mode-D parameter the user did not
# set; an explicit value wins. The damping is MODE_BCD_INNER_DAMPING (below),
# with MODE_D_INNER_RELAX the 'omega' preset's dial (whole-domain omega 1/3); the Hack footprint coefficient is half the WIDTH_HACK_K default; the fabricated trunk V is off
# (on, it roughly triples the planview flicker on sheet-dominated beds); the
# routing filter acts on sheet cells only (its thickness fades with s); MFD
# routing, the only routing mode D accepts. Everything else resolves as in mode C. Measured against the shallow-
# ice reference (MinSIA) at 1 km with the volumes counted over the sub-grid width
# of the channel cells (assessment/suites/minsia.py), massif / notebook / flagship,
# under the channel/sheet reading of 2026-11-13 and the adaptive damping:
# +6.1 / -4.3 / +5.7 % at the bed bar's 1.5 default (2026-11-14; raw -11.8 / -18.1 /
# -3.9 %), +7.0 / -10.5 / +2.6 % at 2. (Under the retired rim-and-flank reading,
# 2026-11-05: +16.9 / -6.6 / -0.8 %.)
MODE_D_INNER_RELAX = 1.0 / 3.0
MODE_D_TRUNK_SURFACE = False
MODE_D_WIDTH_HACK_K = 0.25
MODE_D_ROUTING_FILTER_FACTOR = 2.0
MODE_D_FLOW_ROUTING = 'mfd'
_MODE_TO_CODE = {       # keyed in lower case: normalize_mode ignores case
    MODE_ICE_SURFACE: 'A',
    MODE_BEDROCK_ICE: 'B',
    'a': 'A',
    'b': 'B',
    'c': 'C',
    'd': 'D',
}


def normalize_mode(value):
    """Map a user-facing surface-evolution mode to its internal code.

    Accepts the legible names ``'ice_surface'`` (one tracked state, the ice
    surface) and ``'bedrock+ice_thickness'`` (two tracked states, bedrock + ice
    thickness), plus the permanent short aliases ``'A'`` / ``'B'``, ``'C'``
    (the mode-B + sub-grid-carve alias, resolved to ``'B'`` + carving by
    siim2d) and ``'D'`` (mode C with the channel/sheet reading choosing valley or sheet
    rules, also resolved by siim2d), all in any case. Returns ``'A'``, ``'B'``,
    ``'C'`` or ``'D'``.
    """
    code = _MODE_TO_CODE.get(value.lower()) if isinstance(value, str) else None
    if code is None:
        raise ValueError(
            f"mode must be {MODE_ICE_SURFACE!r} or {MODE_BEDROCK_ICE!r} "
            f"(or the aliases 'A' / 'B' / 'C' / 'D'); got {value!r}")
    return code


_FLOW_ROUTING_TO_CODE = {
    'single': 'single',
    'multiple': 'mfd',
    'infinite': 'dinf',
    'mfd': 'mfd',
    'dinf': 'dinf',
}


def normalize_flow_routing(value, name='flow_routing'):
    """Map a flow-routing name to its router code: ``'single'`` (D8 steepest
    descent), ``'multiple'`` (multiple-flow-direction, code ``'mfd'``) or
    ``'infinite'`` (Tarboton D-infinity, code ``'dinf'``); the codes ``'mfd'``
    and ``'dinf'`` are accepted as aliases. Returns ``'single'``, ``'mfd'`` or
    ``'dinf'``; ``name`` is the parameter the error message names."""
    code = _FLOW_ROUTING_TO_CODE.get(value) if isinstance(value, str) else None
    if code is None:
        raise ValueError(
            f"{name} must be 'single', 'multiple' or 'infinite' (or the aliases "
            f"'mfd' / 'dinf'); got {value!r}")
    return code


# --- Sub-grid carve widening rate (user-facing eta <-> internal factor) ------
# widening_rate is eta = (E_widening - E_c)/E_c: the *excess* of a footprint
# cell's carve-erosion (E_widening) over the centerline incision (E_c), as a
# fraction of E_c (dimensionless). 0 = no net widening (footprint eaten at the
# centerline rate); 1 = widen at the centerline rate (E_widening = 2*E_c);
# 3 (the default) = E_widening = 4*E_c; None / inf / 'inf' / 'infinity' =
# instant U-imposition. The carve kernel multiplies E_c by (1 + eta) =
# E_widening/E_c.
DEFAULT_WIDENING_RATE = 3.0


def widening_factor_from_rate(value):
    """User-facing ``widening_rate`` (eta >= 0) -> internal ``E_widening/E_c`` factor.

    Returns ``1 + eta``, the multiple of the centerline erosion rate the carve
    kernel applies to a footprint cell. ``None`` / ``inf`` / ``'inf'`` /
    ``'infinity'`` (any case) request instant U-imposition (factor = inf).
    Negative eta is rejected -- disable carving with ``carve_width=False``.
    """
    if value is None:
        return math.inf
    if isinstance(value, str):
        if value.strip().lower() in ('inf', 'infinity'):
            return math.inf
        raise ValueError(
            f"widening_rate string must be 'inf' or 'infinity'; got {value!r}")
    v = float(value)
    if math.isinf(v) and v > 0:
        return math.inf
    if math.isnan(v) or v < 0.0:
        raise ValueError(
            "widening_rate (eta) must be >= 0, or None / inf / 'inf' / "
            f"'infinity' for instant U-imposition; got {value!r}")
    return 1.0 + v


# --- Within-step damped iteration (modes B/C/D) ------------------------------
# A landscape step evaluates the diagnostic ice map F(H) once. inner_passes = n
# evaluates it n times instead, moving the thickness iterate
# x <- x + omega*(F(x) - x) after EVERY pass (inner_relax = omega in (0, 1]);
# the first n-1 passes are rehearsals run at dt = 0 (no bed motion, no thickness
# diffusion) and only the last carries the real dt. omega < 1 retains 1 - omega
# of the previous iterate, giving the sheet a numerical memory of about
# 1/(omega*n) landscape steps at n-fold cost — the sheet-regime smoother
# measured in docs/dev/sheet_knob_tradeoffs.md. Defaults (1, 1.0) = a single
# fresh evaluation, bit-for-bit. Modes B/C/D only (mode A has no closure path); the xsimlab adapter
# routes once per step and refuses non-defaults.
INNER_PASSES = 1
INNER_RELAX = 1.0
# The 'omega' preset's mode-B/C dials (decisions/2026-11-04-bc-damping-omega-passes.md,
# private): omega 1/2 over 2 passes. Two passes are two steps
# of the recurrence, so a step retains q = (1 - omega)**passes = 1/4 of the old
# iterate and a steady ramp trails by q/(1 - q) = 1/3 step, against 2 steps for
# one pass at omega 1/3; the twitch falls 3-8x on the standard fixtures at
# 1.4-1.8x the cost. Resolution rule: siim2d.set_and_check_parameters.
MODE_BC_INNER_RELAX = 0.5
MODE_BC_INNER_PASSES = 2
# Under omega < 1 a cell the kernel leaves bare (F = 0) only decays, by
# 1 - omega per pass, and never reaches 0: every cell a glacier ever reached would
# keep a vanishing film that the kernel's H > 0 tests (border base-level fill,
# flotation gate, carve seeding) and the renderer take for ice. The blend snaps
# such a cell to 0 once its iterate is below this thickness (m).
INNER_RELAX_BARE_H = 1.0

# --- Damping presets (inner_damping; siim2d modes B, C and D) ----------------
# 'omega' is the uniform move above: every cell commits x + omega*(F - x), with
# inner_relax / inner_passes the dials. The adaptive presets damp only the cells
# whose update keeps reversing (the period-2 flicker; Rprop-style), at every
# pass's commit, d = F - x and d_prev the cell's previous move (across passes and
# steps), per-cell w (start 1):
#     w <- clip(ADAPTIVE_SHRINK*w if d*d_prev < 0 else ADAPTIVE_GROWTH*w, floor, 1)
#     x <- x + w*d, with the bare-cell snap above.
# A cell that keeps flickering sits at the floor, so its lag is at most
# (1 - floor)/floor steps: 4 at 'adaptive' (0.2, the best constant-climate damper
# measured), 2 at 'adaptive_capped' (1/3). Each preset carries its floor in
# inner_relax, so inner_relax is not a dial there. Measured against omega 1/2 x 2
# (2026-11-04, prototype; design/adaptive-per-cell-omega.md, private): 1x cost,
# less twitch at constant climate, a jumpier ice front under a moving climate and
# weaker tracking at coarse dt. Resolution rule: siim2d.set_and_check_parameters.
INNER_DAMPING = 'omega'
INNER_DAMPING_FLOORS = {'adaptive': 0.2, 'adaptive_capped': 1.0 / 3.0}
INNER_DAMPINGS = (INNER_DAMPING,) + tuple(INNER_DAMPING_FLOORS)
ADAPTIVE_SHRINK = 0.5
ADAPTIVE_GROWTH = 1.25
# The siim2d defaults of modes B, C and D: 'adaptive_capped' (Eric, 2026-11-04),
# then 'adaptive' (Eric, 2026-11-05: dome central thickness within -2.6 to
# +1.2 % of the references, where the capped floor ran up to +10.3 % with up
# to 4.9 % tail oscillation; same cost; ED-166/167).
MODE_BCD_INNER_DAMPING = 'adaptive'

# --- Routing-surface spatial filter (mode B/C sheet knob, opt-in) -------------
# routing_filter_factor = f replaces the surface the ROUTER GRAPH is built from
# with a thickness-weighted TRIANGULAR (chapeau) mean of it — window and
# threshold are stated by _core.step.filter_routing_surface (FIREWALL: see
# siim._core.driver); f = 2 is the
# measured setting (docs/dev/sheet_smoother_choice.md). It is a topology
# smoother, not a display filter: filtering redistributes Q and hence the
# diagnosed H, and the graph can send flux along an edge that is uphill on the
# unfiltered surface. 0 = off (bit-for-bit). Mode B/C, the standalone driver
# and square cells only, and not with trunk_surface (the opposite-regime
# routing device) outside mode D, whose valley <-> sheet map fades each where
# the other applies. Windows wrap on looped axes and truncate at non-looped edges.
ROUTING_FILTER_FACTOR = 0.0

# --- Valley <-> sheet devices (mode D) ---------------------------------------
# Five valley devices assume the routing string carries the WHOLE glacier with
# EMPTY flanks within alpha_g*H: the hc = HC_OVER_H channel datum, the Hack
# ablation footprint, the sub-grid carve, the fabricated trunk V and (by
# exclusion) the routing filter. A sheet has FULL flanks and breaks them. In
# mode D every cell carries a confinedness s (1 = valley, mode C's code
# exactly; 0 = sheet; any s in [0, 1]) from the reading below: the binary class
# read once per step after the passes from the committed H and the eroded bed,
# relaxed in time (REGIME_SWAP_STEPS); the carve consumes it at once, the
# routing surface, the mass balance and the kernel on the next step (lagged
# like H itself); s = 1 on the first step; base-level border cells take their
# dominant donor's s (_core.step.border_inherits). Every device consumes s as
# the convex blend a*s + b*(1 - s), exact at both ends (x*1.0, x*0.0 and + 0.0
# are exact):
#   hc datum       hc_cell = HC_OVER_H*s + HC_SHEET*(1 - s), the per-cell datum
#                  every surface the kernel and the router build their columns
#                  on (threaded into the mode-B kernels; the scalar outside mode
#                  D), with s smoothed over one cell (REGIME_DATUM_SIGMA) and
#                  the border rule run again on it, so a border's datum is
#                  its donor's;
#   footprint      wide*s + cell*(1 - s) below the ELA (the sheet recipe's
#                  width_hack_k = 0 special case retires);
#   carve          widening factor * s(source): a sheet source carves nothing;
#   trunk V        router surface V*s + dynamic*(1 - s), then
#   filter         thickness H*(1 - s) for window and weights — the two router
#                  devices coexist only in mode D.
# Standalone driver only (the adapter refuses mode D).
HC_SHEET = 1.0

# --- Valley <-> sheet reading (mode D) --------------------------------------
# Mode D reads every icy cell (H > REGIME_ICY_H) as CHANNEL or SHEET once per
# step, after the passes, from the committed thickness H and the eroded bed zb
# (_core.regime; decisions/2026-11-13-channel-sheet-discharge-reading.md,
# private). Every length is relative to the glacier's own half-width
# r0 = alpha_g*H/2, with floors in cells (no absolute lengths).
#   discharge   q = Q/(alpha_g*H), Q from a SEPARATE routing + accumulation
#               pass over the plain slab zb + H (every cell treated as sheet:
#               filter thickness H, datum 1, footprint the cell), so the class
#               never sees its own datum. It feeds the reading ONLY.
#   ring        on the 8 half-axes, the samples from r0 + max(REGIME_RING_GAP_HW*r0,
#               REGIME_RING_GAP_CELLS*dx) over max(REGIME_RING_SPAN_HW*r0,
#               REGIME_RING_SPAN_CELLS*dx); bare ground counts as q = 0, H = 0.
#   span        qb = the largest q within r0 of the cell (the channel its
#               cross-section spans); zmin the lowest bed there.
#   weak        fewer than REGIME_RING_SHARE (half) of the ring samples reach
#               qb/REGIME_WEAK (roughly: qb > REGIME_WEAK x the ring median of q);
#               strong: fewer than that share reach the cell's own q/REGIME_STRONG.
#   bed bar     1 + (ring median zb - zmin)/max(ring median H, REGIME_BAR_MIN_H) > kB, the
#               user dial channel_bed_bar (default REGIME_BED_BAR; 0 = off),
#               on weak and strong alike: a channel sits in a trough.
#   hysteresis  8-connected weak regions (joined across looped edges) are
#               channel when their strong cells number at least
#               REGIME_CORE_CELLS and cover at least (alpha_g x their median
#               H)^2: the binary class c (1 channel, 0 sheet; off the ice 1).
#   swap        the devices' s moves 1/REGIME_SWAP_STEPS of the way to c each
#               step (s = c on the first reading and where the cell was not icy
#               on the last); border cells then inherit their dominant donor's.
#               Noted, not built: a time constant w = 1 - exp(-dt/tau), tau
#               about 10 kyr, is the form that does not depend on dt.
#   datum       hc_cell takes s Gaussian-smoothed over icy cells (sigma
#               REGIME_DATUM_SIGMA cells, normalised, wrapped on looped axes).
# Measured at 1 km - 250 m; mode D runs on multiple-flow routing only (single
# flow and D-inf grow false channels along their flux trees).
REGIME_ICY_H = 10.0
REGIME_RING_GAP_HW = 2.0
REGIME_RING_GAP_CELLS = 3
REGIME_RING_SPAN_HW = 2.0
REGIME_RING_SPAN_CELLS = 5
REGIME_WEAK = 3.0
REGIME_STRONG = 10.0
REGIME_RING_SHARE = 0.5
REGIME_CORE_CELLS = 5
REGIME_BED_BAR = 1.5
REGIME_BAR_MIN_H = 1.0
REGIME_SWAP_STEPS = 5
REGIME_DATUM_SIGMA = 1.0
REGIME_DX_WARN = 2.5e3

# --- Closure flux (mode B/C, EXPERIMENTAL; branch closure-transect-flux) ------
# What ice flux the thickness closure + erosion laws see at a cell. This is the
# mode-B/C default (Eric, 2026-09-05: 'transect_min'); mode A always resolves
# to 'cell' (siim2d's None sentinel), and an explicit 'cell' reproduces the
# pre-branch model bit-for-bit.
#   'cell'         the cell's own routed flux Q (the pre-branch closure).
#   'transect'     the routed flux crossing a line one closure-width
#                  (alpha_g*H_lag) long through the cell, perpendicular to
#                  its flow direction: a trunk with flux-free flanks sees its
#                  own Q; a cell inside a broad ice body sees the flux of the
#                  whole width the closure assumes (Q*alpha_g*H/w_eff). Fixes
#                  the sheet-regime under-closure diagnosed in
#                  docs/dev/ice_sheet_dome_probes; see
#                  docs/dev/closure_width_plan.md (v2: transect variant).
#   'transect_min' as 'transect' but a neighbour contributes at most the
#                  cell's own flux (thin flanks cannot borrow a trunk's Q).
# Routing, outputs, carve, flexure load and sediment keep the true flux.
CLOSURE_FLUX = 'transect_min'
CLOSURE_FLUXES = ('cell', 'transect', 'transect_min')   # the accepted values
MODE_A_CLOSURE_FLUX = 'cell'     # mode A's only closure (the one-cell closure)

# --- Effective-slope normalisation gamma (multi-receiver routers) -------------
# A multi-receiver reduction S_raw = sum_k w_k*(zs_i - zs_rk)/L_k is the surface
# gradient PROJECTED on the weighted receiver vector v = sum_k w_k*e_k, i.e.
# S_raw = g.v = |g|*gamma*cos(delta) with gamma = |v| and delta the angle
# between g and v. Dividing by gamma recovers |g|*cos(delta) -- exact when v is
# parallel to g, which holds for bare S^1 weights on a SQUARE 8-neighbour
# stencil and is a small approximation otherwise (measured max error over plane
# angles: 0.25 % at p = 0.5, 0.0017 % at p = 1.1, 0.0094 % at p = 2, 0.0014 %
# at p = 4; 4-8 % on RECTANGULAR cells at p = 1, which is why v1 of the MFD
# router requires dx == dy). Applied as per-cell scaled link lengths
# L_eff[i,k] = gamma_i*L[i,k], so the closure reduction gets L_bar_eff =
# gamma*L_bar (Z_bar is unchanged) and the eroders get the same effective slope.
#
# Floor: gamma -> 0 where a cell's receivers span more than a half-plane (a
# saddle or an eps-fill ridge cell), where |grad zs| is ill-defined and the ice
# flux is ~0 anyway. MFD_GAMMA_FLOOR bounds the slope amplification to
# 1/floor. PROVISIONAL at 0.5 -- sensitivity at 0.25 / 0.5 / 0.75 is measured in
# stage S3 of docs/dev/mfd_router_plan.md, and the value should be revisited
# with that measurement in hand. The same floor conditions the transect
# closure's flow direction: below it the transect kernel falls back to
# Q_transect = Q_i rather than normalising an ill-conditioned direction vector.
# A D-inf cell cannot reach it (its two receivers span 45 deg, so gamma >=
# cos(22.5 deg) = 0.9239), so the floor is inert under D-inf. MFD reaches it at
# ridge crests and peaks, whose receivers fan over both flanks (gamma ~ 0 at a
# symmetric crest), so under MFD it acts and amplifies the slope there by up to
# 1/floor.
MFD_GAMMA_FLOOR = 0.5

# --- MFD spreading exponent (flow_routing='mfd') ------------------------------
# The multiple-flow-direction router splits a cell's flux over ALL strictly
# downhill 8-neighbours of the eps-filled surface with weights
# w_k = (S_k/S_max)^p / sum_j (S_j/S_max)^p, p = MFD_EXPONENT (exposed as the
# `mfd_exponent` parameter; see resolve_mfd_exponent below). p -> inf recovers
# D8's receivers; p = 1 is Budd & Warner's split and the ONLY exponent whose weighted
# receiver vector is exactly parallel to the surface gradient on a square
# 8-stencil (sum_k |e_k| e_k (x) e_k = 2(1+sqrt2)*dx*I), which is what makes the
# gamma normalisation above exact there. PROVISIONAL default pending the stage-S4
# dome sweep (p = 1 vs 1.1); see docs/dev/mfd_router_plan.md section 2.2.
MFD_EXPONENT = 1.0
# The largest finite exponent the router takes. A raw weight (S/S_max)^p
# underflows to an exact zero once S/S_max < 2^(-1075/p), leaving a link that
# carries nothing in the pack: common above 50 (seen from p = 64), none at
# p <= 50 in the model runs tested (study of 2026-11-12, ED-180), though a
# near-flat link beside a steep one can still do it there. Above 50 each
# further step in p moved every mode-D statistic by about one twin-run spread.
# A larger finite exponent is taken as p = inf, with a warning. p = inf is an
# exact branch of the router: the steepest link alone (equal steepest links
# share). It is still the MFD code path, not flow_routing='single'; in the two
# 400-step mode-D runs of the study, p = 50 and inf differed by up to 1 % in
# ice extent, 3.3 % in ice volume and 2.5-6 % in the time-mean fields.
MFD_EXPONENT_MAX = 50.0
# Under flow_routing='mfd' (mode D's default routing) an unset parallel_erode
# follows numba's threading layer: on under the layers named here, off under
# the others (every other router keeps PARALLEL_ERODE). MFD's drainage levels
# are many and small (about 160 a step on a 101 x 101 grid, 1900 on 101 x
# 1001), and the eroder pays one fork/join per level: in mode D under TBB the
# parallel eroder is 1.44x (Coulomb) and 1.10x (power) per notebook step and
# 1.23x on the 1000 km escarpment, on workqueue 1.00x and 0.65x (ED-168,
# 2026-11-06). siim2d reads the layer the process runs
# (_core.threads.threading_layer).
MFD_PARALLEL_ERODE_LAYERS = ('tbb',)


def resolve_mfd_exponent(value, finite=False):
    """``mfd_exponent`` as the router takes it: a positive float, finite up to
    ``MFD_EXPONENT_MAX``, or ``inf``. A larger finite value becomes ``inf``
    with a warning that names it; zero, a negative and NaN raise. With
    ``finite`` (mode D) anything above ``MFD_EXPONENT_MAX``, inf included,
    raises instead: an infinite exponent is single-flow routing."""
    p = float(value)
    if not p > 0.0:                         # NaN fails the comparison too
        raise ValueError(f"mfd_exponent must be positive (inf included), got {value!r}.")
    if finite and p > MFD_EXPONENT_MAX:
        raise ValueError(
            f"mode='D' requires a finite mfd_exponent (at most {MFD_EXPONENT_MAX:g}): an "
            "infinite one routes down the steepest link alone, and the channel/sheet "
            f"reading reads single-flow trees as false channels. Got {value!r}.")
    if MFD_EXPONENT_MAX < p < math.inf:
        warnings.warn(
            f"mfd_exponent={p:g} is above {MFD_EXPONENT_MAX:g}, where the "
            "weights of all but the steepest link begin to underflow: taken "
            "as mfd_exponent=inf (flow_routing='mfd' then routes down the "
            "steepest link alone).", stacklevel=3)
        return math.inf
    return p

# --- Fabricated trunk-surface routing (mode B/C, opt-in `trunk_surface`) ------
# The trunk-surface provider routes (the router graph ONLY; FIREWALL: see
# siim._core.driver) on a fabricated ice surface
# with a LINEAR cross-valley dip toward the centerline, so flow converges onto
# the trunk chain and the centerline's raw flux is the full cross-section
# discharge. The dip is a
# routing NUMERICS device, not physics (real trunk surfaces are ~convex-up);
# nothing physical is computed from it. Cross-slope S_c = TRUNK_DIP_K * S_v(s),
# S_v the local down-valley surface slope at the source. D8 convergence needs
# S_c > (sqrt(2)-1)*S_v ~= 0.414*S_v; 0.6 sits safely above with margin.
# The `trunk_dip_k=None` default is ROUTER-DERIVED
# (_core.router.resolve_trunk_dip_k): TRUNK_DIP_K under D8 and D-inf,
# TRUNK_DIP_K_MFD under MFD.
TRUNK_DIP_K = 0.6
# MFD (p = 1) leaks off the trunk axis through the two down-valley diagonals:
# axis leak = 2s/(1+2s), s = max(1-k, 0)/sqrt(2), ZERO exactly at k >= 1 (the
# first-flank leak is 1/(2+sqrt2) ~= 0.293 AT k = 1 and decays slowly above it
# as 1/(1 + k + sqrt2*k), the inward-upstream diagonal turning downhill past
# k = 1: 0.242 at k = 1.3, 0.172 at k = 2). D8 and D-inf keep 0.6: D-inf's
# leak clamps to zero at k = 1 and collapses its cycle there.
# Closed form + measurement: docs/dev/mfd_valleys_e1e2.md section 9.1.
TRUNK_DIP_K_MFD = 1.0
# Slope floor for flat reaches (m/m): where the down-valley surface slope falls
# below this, S_v is floored here so the dip still beats routing ties/noise.
TRUNK_DIP_FLOOR = 1e-3
