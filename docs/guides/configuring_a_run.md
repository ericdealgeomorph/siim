# Configuring a run

Both models take a parameter dict; a handful of keys set the regime, and the
defaults are the model's native, glacially-interesting configuration.

## Surface-evolution mode — `mode`

The default `'bedrock+ice_thickness'` tracks **two** state fields — the bedrock
and the ice thickness — and rebuilds the ice surface each step, so carved
overdeepenings persist on ice retreat (*bed memory*, the regime where the
MISI-style autogenic cycling lives). The alternative, `'ice_surface'`, tracks a
**single** field — the ice surface — and heals troughs instantly; use it for
simpler steady-profile behaviour. (`'A'` / `'B'` are accepted as short aliases
for `'ice_surface'` / `'bedrock+ice_thickness'`; `'C'` is a 2D-only alias for
mode B with carving on — see the mode-C standard below.) The channel-floor datum
relating the tracked bed to the ice surface is the `hc_over_H` convention.

## Sub-grid glacier-width carving — `carve_width` (2D)

Default `True` in `'bedrock+ice_thickness'` mode. A glacier of mean thickness
*H* fills a valley of width *α_g·H* — many grid cells wide — and that footprint
erodes the bed, giving troughs real width, a hypsometric feedback, and channel
capture. The footprint widens at rate `widening_rate` (η, default `3.0`).
Selecting `mode='ice_surface'` coerces carving off.

## The mode-C standard (carved runs)

A carved bedrock run — `mode='bedrock+ice_thickness'` with carving on, i.e. the
2D default, also spelled `mode='C'` — turns on two routing-side helpers by
default: `trunk_surface=True` fabricates a converging ice surface so a wide
trunk's flux is routed onto its centerline (honest cross-section discharge), and
`routing_relax=0.5` EMA-relaxes the once-per-step routing surface to damp a
cosmetic planview ice flicker. **Both act only on the routing graph and the
mass-balance surface — never on the erosion/thickness physics or the outputs** —
and both are opt-out (set them to `False` / `0.0`). Plain `mode='B'` (no carve)
and `mode='A'` keep them off. An explicit value always wins over the default.

*Which mode?* Reach for the mode-C default (`'C'`, or just leave `mode` unset)
for realistic carved troughs with width and channel capture. Drop to plain
`mode='B'` when you want bed memory without the width feedback — or on a grid too
coarse to resolve the carve footprint `R = α_g·H/2` (see
{doc}`../getting_started/first_2d_run`), where carving is a no-op anyway. Use
`mode='A'` for the simplest, fastest steady-profile behaviour (troughs heal
instantly, no bed memory).

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
time series on the run clock, so you can drive transgression / regression exactly
the way `ela_sawtooth` drives glacial cycles. A step in `bl` launches a knickpoint
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
never stored as a floor). If you set `bl` and see sub-datum beds, that is correct.
`border_bed_uplift` sets the border rock's uplift forcing — the `U` in the icy
border budget and the ice-free recovery rate (defaults to the tectonic `U`),
orthogonal to the water forcing `bl`.

The default (`bl = 0`) suits a single-outlet domain. There is no border
time-step restriction: the border budget is implicit (it cannot overshoot), so
the old `dt ≤ 500 yr` icy-border caveat stays retired.
