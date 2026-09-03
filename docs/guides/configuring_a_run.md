# Configuring a run

Both models take a parameter dict; a handful of keys set the regime. This page
explains the choices that interact. See {doc}`parameter_reference` for every
accepted key, default, unit, and model-specific restriction.

## Surface-evolution mode — `mode`

`'bedrock+ice_thickness'` tracks **two** state fields — the bedrock and the ice
thickness — and rebuilds the ice surface each step, so carved overdeepenings
persist on ice retreat (*bed memory*). It is the 1D default and the dynamical
state beneath the 2D mode-C default. The alternative, `'ice_surface'`, tracks a
**single** field — the ice surface — and heals troughs instantly; use it for
simpler steady-profile behaviour. (`'A'` / `'B'` are accepted as short aliases for these two modes;
`'C'` is a 2D-only alias for mode B with carving on.) The channel-floor datum
relating the tracked bed to the ice surface is the `hc_over_H` convention.

## Sub-grid glacier-width carving — `carve_width` (2D)

The 2D package default is `mode='C'`, which resolves to mode B with
`carve_width=True`. By contrast, spelling `mode='B'` leaves carving **off**
unless you explicitly pass `carve_width=True`; `carve_width=None` is the
mode-dependent sentinel that implements this distinction. A glacier of mean
thickness *H* fills a valley of width *α_g·H* and that footprint erodes the bed,
giving troughs real width, a hypsometric feedback, and channel capture. The
footprint widens at rate `widening_rate` (η, default `3.0`). Mode A cannot be
combined with carving and rejects `carve_width=True`.

## The mode-C standard (carved runs)

A carved bedrock run — `mode='bedrock+ice_thickness'` with carving on, i.e. the
2D default, also spelled `mode='C'` — turns on two routing-side helpers by
default: `trunk_surface=True` fabricates a converging ice surface so a wide
trunk's flux is routed onto its centerline (cross-section discharge), and
`routing_relax=0.5` EMA-relaxes the once-per-step routing surface to damp a
planview ice flicker. Both modify the routing/mass-balance surface rather than
post-processing stored output. They therefore can change the simulated routing,
accumulation, and resulting evolution, even though erosion/thickness kernels and
stored state remain raw. Both are opt-out (set them to `False` / `0.0`). Plain
`mode='B'` (no carve) and `mode='A'` keep them off. An explicit value always
wins over the mode-dependent default.

**Choosing a mode.** Use the mode-C default (`'C'`, or leave `mode` unset) for
carved troughs with width and channel capture. Use plain `mode='B'` for bed
memory without the width feedback — or on a grid too coarse to resolve the
carve footprint `R = α_g·H/2` (see {doc}`../getting_started/first_2d_run`),
where carving is a no-op anyway (`mode='B'` also turns off the mode-C routing
helpers, which are grid-independent). Use `mode='A'` for the simplest, fastest
steady-profile behaviour (troughs heal instantly, no bed memory).

⚠ **zELA re-tune caveat.** Trunk-surface routing evaluates the mass balance on
the fabricated converging surface, so a mode-C run accumulates over a different
area than the same config without `trunk_surface` — the ice extent and results
shift. Re-tune `zELA` before comparing mode-C runs against plain mode-B or
pre-trunk-surface runs.

## Sliding law — `sliding_law`

`'power'` (default), `'eff-exp'`, or `'coulomb'`; all three work under either
routing and either mode.

## Flow routing (2D) — `flow_routing`

`'single'` (default, steepest descent) or `'dinf'` (Tarboton D-infinity,
multiple-flow). Closed depressions are routed across by an eps-fill while the
physics stays on the true surface.

## Climate and uplift forcing

`zELA` (equilibrium-line altitude) and the uplift rate can be constant scalars
or time series. `siim.forcing` builds the common time-varying forcings:

```python
from siim.forcing import ela_sawtooth, uplift_step

_, zELA = ela_sawtooth(T, nt, ela_high=1500, ela_low=300, period=100e3)
_, U = uplift_step(T, nt, U_init=1e-3, U_final=2e-3, step_frac=0.5)
```

Pass the resulting series as the `zELA` / uplift parameters for a model run of
length `T` over `nt` steps — `ela_sawtooth` drives glacial cycles, `uplift_step`
a change in uplift rate partway through.

## Base level and the coastline — `bl`

Base level is a **water line** `bl` — a movable Dirichlet datum that nothing in
the model can erode. Like `zELA`, it can be a scalar (default `0`) or a length-`nt`
time series on the run clock, so transgression / regression can be driven the
same way `ela_sawtooth` drives ELA cycles. A step in `bl` launches a knickpoint
at the outlet that migrates upstream (the water/rock separation is a proper
time-dependent Dirichlet condition for the stream-power characteristics).

The border **rock** is ordinary erodible state that may sit far below `bl` (a
carved fjord, a relict overdeepening). An icy, through-flowing border is an
**outflow**: the domain edge is an arbitrary cut through a continuing glacier, so
the border column takes its upstream neighbour's thickness (zero-gradient) and
its bed keeps eroding on the interior flow slope, bounded at the ice column's
flotation draft (`zb → bl − hc·H`, a fjord-mouth trench) by the
**waterline-flotation ramp** (`flotation_gate`, default on; `flotation_ramp` =
γ, default 0.1) — the ρ_i = ρ_w effective-pressure law: glacial erosion scales
with `f = clip((zs − bl)/(γ·hc·H), 0, 1)`, so erosion shuts off smoothly at
flotation. The border budget is integrated **implicitly** (a closed-form step),
so it approaches the draft monotonically at any `dt`. In the interior the same
ramp stops a deep-carved cell whose ice surface sinks to the waterline. `γ = 0`
is a hard on/off gate, and keep `γ ≤ 0.2` (wide ramps can dome `cap=False`
runs). Turning the gate off un-bounds the border — diagnostics only. There is
no calving.

**Outputs are the true state.** State and outputs report the actual `(zb, H)`
everywhere: a drowned or relict border bed shows through *below* `bl` (bed
memory), and the ocean / lakes are a display layer only (rendered at `max(zs, bl)`,
never stored as a floor). Sub-datum beds under a nonzero `bl` are expected.
`border_bed_uplift` sets the border rock's uplift forcing — the `U` in the icy
border budget and the ice-free recovery rate (defaults to the tectonic `U`),
orthogonal to the water forcing `bl`.

The default (`bl = 0`) suits a single-outlet domain. There is no border
time-step restriction: the border budget is implicit (it cannot overshoot), so
the old `dt ≤ 500 yr` icy-border caveat stays retired.

**Per-side base level (2D).** A 2D domain can drain to outlets at genuinely
different water lines — a fjord coast on one side, an inland basin on the other.
Pass `bl` as a dict keyed by side, `{'left': …, 'right': …, 'bottom': …,
'top': …}`, each value a scalar or a length-`nt` series (mixing allowed);
unspecified sides stay at the default `0`. Only `'fixed_value'` sides are
base-level outlets, so a datum on a `'looped'` or `'core'` side raises, as does
an unknown key. Each border node then carries its own side's datum, and every
interior node uses the datum of **the outlet its basin drains to** — the whole
watershed of a coastal outlet is graded, floated and flooded against that
coast's water line. A corner shared by two fixed sides takes the x-side
(`left`/`right`) value. Per-side `bl` needs the in-house driver (the optional
xsimlab adapter takes a single scalar per step and raises on a dict); a dict
whose fixed sides all carry the same value is the scalar path bit-for-bit. In
2D, `bl` — scalar or per-side — is consumed by **modes B and C only**: 2D mode
A holds its borders at their initial elevation and ignores `bl` entirely
(a known gap, not a per-side limitation), so a dict passed to a mode-A run is
accepted and does nothing. The analytical steady-state reference stays graded
to zero and is not offset-corrected per side, so `rms_vs_analytical` and the
analytical overlay are not meaningful across outlets at different datums (the
constructor warns; `self.bl` is then only the fixed-side mean, a label).
