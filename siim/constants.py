"""Shared physical constants and per-sliding-law parameter dispatch.

Single source of truth for the numerical models (siim1d, siim2d, the
standalone fastscape process module) and the analytical machinery
(siim.analytical). Imports nothing heavier than numpy, so the lightweight
analytical layer can depend on it without dragging in the model stack.
"""
from collections import namedtuple
import math

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
# constant). 1.0 reproduces the historical mean-bed datum (zs = zb + H).
# See docs/dev/hc_convention_notes.md for the datum-redefinition argument.
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
# zb* = bl - hc*H + delta*U/E; docs/dev/outflow_implicit_budget.md) — turning
# the gate off un-bounds the border (f == 1: measured runaway). Off is for
# diagnostics only.
# See docs/dev/boundary_conditions.md + docs/dev/outflow_bc_study.md.
FLOTATION_GATE = True
# Flotation-gate ramp width gamma (the effective-pressure softening of the
# gate; docs/dev/soft_gate_probe.md). Instead of a hard on/off switch at the
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
# Level-scheduled parallel mode-B erosion (docs/dev/perf_audit.md §8): the
# eroder dependency is a DAG depth, so topological levels run in order with
# prange inside each — BIT-FOR-BIT identical to the serial walk at any thread
# count (pinned by test_parallel_erode). Default ON (Eric, 2026-07-07; it is
# purely a scheduling change — SFR coulomb steps measured -43%). False = the
# serial eroder, for single-core environments or debugging thread issues.
PARALLEL_ERODE = True
COULOMB_CLAMP = 1e-12          # regularized-Coulomb H-solver: min relative gap from the pole
DEFAULT_SLIDING_LAW = 'power'  # 'eff-exp', 'power', or 'coulomb' (every front-end default)
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
# value since the S5 standalone flip (the fortran SFR arm is deleted). The
# param survives as the ROUTER-CONTRACT plug point (docs/dev/
# router_contract.md): a future backend (e.g. fastscapelib) re-widens the
# accepted set. The S4 router swap was the one NON-bit-for-bit numeric change
# of the standalone migration (tie-break/basin-carve paths differ from the
# retired fortran SFR; gated behaviorally by test_d8_receiver_parity vs the
# frozen raw fortran refs + the router contract battery).
ROUTER_DEFAULT = 'inhouse_d8'


def cg_prefactor(alpha_g=ALPHA_G, Ac=AC, rho_ice=RHO_ICE, g=GRAVITY):
    """Glacial rheology group cg = alpha_g * kt * (2*Ac/5) * (rho_ice*g)^3
    [m^-3 yr^-1] (the s/yr conversion kt is absorbed). The 2/5 is the
    depth-integrated Glen deformation prefactor 2A/(n_c+2) with n_c = 3
    (model-paper convention; Ac is the physical Glen coefficient)."""
    return alpha_g * KT * (2.0 * Ac / 5.0) * (rho_ice * g) ** 3


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
# defaults trunk-surface routing ON and the routing surface EMA-relaxed, so a
# standard carved run converges trunk flow onto the centerline (honest
# cross-section flux) and damps the discrete-D8 planview ice flicker. Plain
# mode B (no carve) and mode A keep both OFF. siim2d resolves the sentinels
# after mode normalization; an explicit user value always wins. (The fastscape
# facade stays explicit — no mode magic for standalone users.)
MODE_C_TRUNK_SURFACE = True
MODE_C_ROUTING_RELAX = 0.5
_MODE_TO_CODE = {
    MODE_ICE_SURFACE: 'A',
    MODE_BEDROCK_ICE: 'B',
    'A': 'A',
    'B': 'B',
    'C': 'C',
    'c': 'C',
}


def normalize_mode(value):
    """Map a user-facing surface-evolution mode to its internal code.

    Accepts the legible names ``'ice_surface'`` (one tracked state, the ice
    surface) and ``'bedrock+ice_thickness'`` (two tracked states, bedrock + ice
    thickness), plus the permanent short aliases ``'A'`` / ``'B'`` (exact
    match) and ``'C'`` / ``'c'`` (the mode-B + sub-grid-carve alias, resolved to
    ``'B'`` + carving by siim2d). Returns ``'A'``, ``'B'`` or ``'C'``.
    """
    code = _MODE_TO_CODE.get(value) if isinstance(value, str) else None
    if code is None:
        raise ValueError(
            f"mode must be {MODE_ICE_SURFACE!r} or {MODE_BEDROCK_ICE!r} "
            f"(or the aliases 'A' / 'B' / 'C'); got {value!r}")
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


# --- Routing-surface EMA relaxation (mode B/C anti-flicker, opt-in) -----------
# The once-per-step routing/mass-balance surface is built from the LAGGED
# closure H, and D8 receiver choice is discrete: near-tied surfaces flip whole
# subtrees of flux each step and H ~ Q^(1/4..1/5) turns those O(1) flux swaps
# into a step-scale ice-thickness slosh (a cosmetic period-2 planview flicker;
# integrated area/volume + attractor stats are unaffected). routing_relax = r
# replaces the raw lagged H feeding the surf2erode surface with an EMA
# H_eff(t) = r*H_eff(t-1) + (1-r)*H_lag(t) (routing + accumulation only — the
# kernel/carve/outputs stay on raw state, the runaway firewall). 0 = off
# (bit-for-bit); r in [0, 1). See docs/dev/step_flicker.md.
ROUTING_RELAX = 0.0

# --- Fabricated trunk-surface routing (mode B/C, opt-in `trunk_surface`) ------
# The trunk-surface provider routes + accumulates on a fabricated ice surface
# with a LINEAR cross-valley dip toward the centerline, so flow converges onto
# the trunk chain and the centerline's raw flux is the full cross-section
# discharge (design record: docs/dev/trunk_surface_routing.md). The dip is a
# routing NUMERICS device, not physics (real trunk surfaces are ~convex-up);
# nothing physical is computed from it. Cross-slope S_c = TRUNK_DIP_K * S_v(s),
# S_v the local down-valley surface slope at the source. D8 convergence needs
# S_c > (sqrt(2)-1)*S_v ~= 0.414*S_v; 0.6 sits safely above with margin.
TRUNK_DIP_K = 0.6
# Slope floor for flat reaches (m/m): where the down-valley surface slope falls
# below this, S_v is floored here so the dip still beats routing ties/noise.
TRUNK_DIP_FLOOR = 1e-3
