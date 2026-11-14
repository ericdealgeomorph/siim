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
`'C'` is a 2D-only alias for mode B with carving on, and `'D'` is mode C with
the valley ↔ sheet reading described below.) The channel-floor datum
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
2D default, also spelled `mode='C'` — turns on a routing-side helper by
default: `trunk_surface=True` fabricates a converging ice surface so a wide
trunk's flux is routed onto its centerline (cross-section discharge). It
modifies the router graph's input rather than post-processing stored output,
so it can change the simulated routing and the resulting evolution, even
though the erosion/thickness kernels and stored state remain raw. It is
opt-out (`trunk_surface=False`); plain `mode='B'` (no carve) and `mode='A'`
keep it off. An explicit value always wins over the mode-dependent default.
Modes B, C and D also damp the ice state by default
(`inner_damping='adaptive'`, next section).

**Choosing a mode.** Use the mode-C default (`'C'`, or leave `mode` unset) for
carved troughs with width and channel capture. Use plain `mode='B'` for bed
memory without the width feedback — or on a grid too coarse to resolve the
carve footprint `R = α_g·H/2` (see {doc}`../getting_started/first_2d_run`),
where carving is a no-op anyway (`mode='B'` also turns off the mode-C routing
helpers, which are grid-independent). Use `mode='A'` for the simplest, fastest
steady-profile behaviour (troughs heal instantly, no bed memory).

The fabricated trunk surface is a routing device only: the router graph is
built on it, while the mass balance and the hillslope diffusion are evaluated on
the dynamic ice surface.

## Damping the ice map within a step (sheet runs)

A landscape step evaluates the diagnostic ice map `F(H)` once. `inner_passes = n`
evaluates it n times instead, moving the thickness iterate
`H ← H + ω(F(H) − H)` after every pass with `ω = inner_relax`; the first `n − 1`
passes are rehearsals run at `dt = 0` (no bed motion, no thickness diffusion) and
only the last carries the real step. ω is the *update* fraction, so `1 − ω` is
what the iterate retains: below 1 the ice acquires a numerical memory of roughly
`1/(ω·n)` landscape steps, at n-fold cost, which damps the margin flicker of
ice-sheet runs and the period-2 planview flicker of valley runs. `(1, 1.0)` is
one fresh evaluation, the undamped model.

`inner_damping` chooses how each pass commits. `'omega'` is the uniform move
above, with `inner_relax` and `inner_passes` as its dials (modes B and C fill an
unset dial with `inner_relax=0.5` over `inner_passes=2`: a step retains a quarter
of the old iterate, so a steady change trails by about a third of a step, at
about 1.5× the cost of one pass; mode D with `1/3` over one pass). The two
adaptive presets damp only the cells whose update keeps reversing sign, the
period-2 flicker: each cell's weight halves on a reversal and grows by 1.25
otherwise, between the preset's floor and 1, so a cell that keeps flickering
lags at most `(1 − floor)/floor` steps and a cleanly growing or retreating cell
runs undamped. `'adaptive'` (floor 0.2, at most 4 steps) is the default in
modes B, C and D; `'adaptive_capped'` (floor 1/3, at most 2) damps more
lightly. Both cost one pass a step (`inner_passes` still composes), carry their floor in
`inner_relax` (so setting `inner_relax` with them is an error, and setting it
alone picks `'omega'`), and need the standalone driver. Measured against
`'omega'` at ω ½ over 2 passes: less twitch at a fixed climate, a jumpier ice
front under a moving one, and weaker tracking at coarse `dt` (a cell that truly
reverses within a few steps reads as flicker). `driver='xsimlab'` needs
`inner_passes=1, inner_relax=1.0`.

Under `inner_relax < 1` the final pass's
eroder still reads the kernel's own closure thickness while the state carries the
blend, and the committed state can hold ice where that pass's diagnostic flux is
zero (it decays, and snaps to bare below `constants.INNER_RELAX_BARE_H`, 1 m);
with `H_diffusivity > 0` the damping also attenuates the diffusion correction
(the model warns). These are properties of the damped model, not
display filters, so re-tune and re-measure before comparing a damped run with an
undamped one.

The recommended sheet recipe under `'omega'` (measured before the adaptive
presets; setting `inner_relax` picks `'omega'`) is one pass at
`inner_relax = 1/3` with `routing_filter_factor = 2`, at any spacing. ω alone sets what is physical —
volume, trough fill, how far the closure's one-cell margin threads reach, the
step-to-step flicker — and a second pass only shortens the thin skin of decaying
ice on the retreat, at twice the cost; `1/2` keeps more ice and some threads,
`1/5` removes the threads at the price of thin ice.

`routing_filter_factor` is the spatial companion of those knobs, and composes
with them. Above 0 the *router graph alone* is built from a thickness-weighted
triangular (chapeau) mean of the routing surface `zb + hc·H`, over a window
whose half-width is `max(f·H/dx − ½, 0)` cells — about f ice thicknesses, set by
the local thickness — while the mass-balance surface and the hillslope diffusion
keep the unfiltered surface; dry cells and cells thinner than `dx/(2f)` keep
their raw value. The window therefore opens wherever `H/dx > 1/(2f)` — a quarter
of a cell at `f = 2`, which is most of a sheet's wet area even at 1 km spacing
(87–88 % of wet cells on the study's L fixtures). What depends on resolution is
the measured benefit, not the activation, and it is empirical: on the
kilometre-thick L sheets the filtered and unfiltered recipes agree to a tenth of
a point of bulk error at 1 km, while at 500 m the filter is worth 0.7–4.5 points
of it and 2–4× less local range; on a dissected massif whose ice is mostly under
500 m it is inert at both spacings, because a window under a cell wide has
nothing to average.
`f = 2` is the measured setting; `0` (the default) is off, bit-for-bit. Like the
damping, this changes the model rather than the picture — filtering redistributes
`Q` and hence the diagnosed `H`, and the graph can send flux along an edge that
is uphill on the unfiltered mass-balance surface. Mode B/C and the standalone
driver only, on square cells (windows wrap on looped axes and truncate at the
other edges), and never together with
`trunk_surface` outside mode D (below) — they are opposite-regime routing-surface devices, so a filtered
mode-C run has to pass `trunk_surface=False` (mode C resolves it to `True`).

## Valley glaciers and ice sheets in one domain — `mode='D'`

Five mode-B/C devices assume a valley glacier, a trunk whose flanks are empty of
ice within its width `α_g·H`: the `1.5` channel-floor datum, the Hack-width
ablation footprint below the ELA, the sub-grid carve, the fabricated trunk
surface and, by exclusion, the routing filter. A sheet has full flanks and
breaks every one of them. `mode='D'` lets one run hold both: each cell carries a
confinedness `s`, `1` for a valley channel and `0` for part of an ice sheet, and
every device blends `a·s + b·(1 − s)` between its valley and sheet forms:

1. the datum `hc_cell = 1.5·s + 1.0·(1 − s)`: a valley surface is `zb + 1.5·H`,
   a sheet's `zb + H`;
2. the below-ELA ablation footprint: the Hack width where `s = 1`, the cell where
   `s = 0`;
3. the carve's widening factor is scaled by the *source* cell's `s`, so a sheet
   source carves nothing;
4. the trunk surface's V is scaled by its source's `s`;
5. the routing filter uses the thickness `H·(1 − s)` for its window and its
   weights, so a valley cell opens no window and weighs nothing in a sheet
   neighbour's.

`s` comes from the channel/sheet reading: does the cell carry markedly more ice
discharge than the ice around it, in a trough? Every step, after the thickness
solve, each icy cell (`H > 10 m`) is read from

- the discharge per unit width `q = Q/(α_g·H)` of a *separate* routing pass over
  the plain slab `zb + H`, every cell treated as sheet, so the reading never sees
  its own datum (this pass feeds the reading and nothing else);
- a ring around the cell on the eight half-axes, starting two half-widths
  (`α_g·H/2`) beyond its own half-width, at least 3 cells out, and spanning two
  half-widths, at least 5 cells; bare ground counts as `q = 0`.

A cell is *weak* evidence of a channel when fewer than half the ring's samples
reach a third of the largest `q` within its own half-width (roughly: that `q` is
more than 3 times the ring's median), and *strong* when fewer than half reach a
tenth of its own `q`. Both must also clear a bed bar: the lowest bed within the
cell's half-width must lie below the ring's median bed by more than `(k_B − 1)`
times the ring's median ice thickness, with `k_B = channel_bed_bar` (1.5 by
default; 0 turns the bar off).
Connected weak regions (joined across looped edges) are channel when their strong
cells number at least 5 and cover at least one channel width squared,
`(α_g·H)²`; every other icy cell is sheet.

Every length is relative to the glacier's own width, with floors in cells. The
class then reaches the devices slowly: `s` moves a fifth of the way to the new
class each step (a cell that was not icy on the last step takes the class at
once), base-level border cells take the `s` of their dominant donor, and the
datum takes `s` smoothed over one cell of the icy neighbourhood (the border
cells again taking their donor's value). The carve uses
`s` at once, the routing surface, the mass balance and the thickness closure on
the next step, and `s = 1` on the first step. Mode D adds no routed surface to
the closures — the V and the filtered surface still reach nothing but the router
graph, and the reading's slab nothing but its own routing pass — and with `s = 1` everywhere it is mode C with mode
D's settings, exactly. The per-cell datum is stored as `hc_cell_out` and the
class as `channel_out` ({doc}`outputs_and_io`).

Mode D also sets its own defaults wherever you leave them unset: the
`'adaptive'` damping (and, if you choose `inner_damping='omega'`,
`inner_relax = 1/3` over one pass), the Hack footprint
coefficient `width_hack_k = 0.25`, the trunk surface off, the routing filter
`routing_filter_factor = 2` on sheet cells and MFD routing, the only routing it
accepts (single-flow and D-infinity flux trees read as false channels).
It is the one mode in which `routing_filter_factor > 0`
and `trunk_surface=True` combine. It needs the standalone driver, square cells,
a finite `widening_rate` and every looped axis at least 17 cells long, so that
the ring's floor cannot wrap onto itself (a run warns when its thickest ice
widens the ring past half a looped axis), and it warns on cells coarser than
2.5 km: the reading was measured at 1 km to 250 m, and coarser grids leave most
outlets narrower than a cell.

## Sliding law — `sliding_law`

`'power'` (default), `'eff-exp'`, or `'coulomb'`; all three work under every
routing and every mode. What the laws' length scales `lambda_p` and `lambda_c` set,
and how they trade off against `ce`, is in {doc}`choosing_glacial_parameters`.

## Flow routing (2D) — `flow_routing`

`'single'` (default, steepest descent), `'multiple'` (multiple-flow-direction:
every strictly downhill neighbour, weights `(S_k/S_max)^mfd_exponent`), or
`'infinite'` (Tarboton D-infinity, two-receiver). The router codes `'mfd'`
(= `'multiple'`) and `'dinf'` (= `'infinite'`) are accepted as aliases, and the
rest of this guide uses them. Closed depressions are routed across by an
eps-fill while the physics stays on the true surface.

Both multi-receiver routers run with the gamma effective-slope normalisation on
by default (`slope_normalise`), so the thickness closure and all three eroders
read the surface-gradient magnitude rather than its projection on the weighted
receiver direction. For D-inf that is a behavioural change (see the changelog):
the unnormalised reduction returned 0.9239 of the true gradient on a plane tilted
pi/8 (mid-facet, its worst angle), inflating `H` there by up to about 5 %, and
was exact only on the grid axes. Pass `slope_normalise=False` to reproduce the
older behaviour.

`'mfd'` runs in modes B, C and D (it is mode D's default and only routing). It
requires a transect `closure_flux` and square cells (`dx == dy`), and refuses
mode A, which has only the one-cell closure.
Under it an unset `parallel_erode` follows Numba's threading layer: the
parallel eroder under TBB, the serial one on `workqueue`, where MFD's many
small drainage levels make the parallel schedule slower
({doc}`../getting_started/install`). See `mfd_exponent` in the parameter
reference.

**MFD in valleys (mode C).** Under `'mfd'` the carve and the trunk surface run as
they do under D8, with the dip of the fabricated trunk surface — the V — resolved
to `trunk_dip_k = 1.0` instead of `0.6`: a spreading router keeps sending part of
an axis cell's flux off the axis until the cross-valley slope matches the
down-valley slope. MFD in valleys needs the trunk resolved: the V and the carve
engage only where α_g·H/2 > dx, i.e. a trunk wider than two cells at its
thinnest. On the 500 m quickstart landscape the minimum-phase trunk is sub-cell
and MFD carries +17 % volume / +25 % mean thickness against D8 there; at 250 m
the two agree within D8's own cycle scatter at both phases and MFD's bed has
about 7 % less relief. (The quickstart is the README's 100 × 50 km mode-C example,
`nx = 201`.) Budget for the cost: MFD mode C runs 1.6–1.9× D8's wall time at
250 m.

## Climate and uplift forcing

`zELA` (equilibrium-line altitude), the precipitation `P` and the uplift rate
can be constant scalars or length-`nt` time series on the run clock.
`siim.forcing` builds the common time-varying forcings:

```python
from siim.forcing import ela_sawtooth, uplift_step, interp_forcing

_, zELA = ela_sawtooth(T, nt, ela_high=1500, ela_low=300, period=100e3)
_, U = uplift_step(T, nt, U_init=1e-3, U_final=2e-3, step_frac=0.5)
_, P = interp_forcing(T, nt, times=[0, T], values=[1.0, 0.3])   # drying ramp (m/yr)
```

Pass the resulting series as the `zELA` / `P` / uplift parameters for a model
run of length `T` over `nt` steps — `ela_sawtooth` drives glacial cycles,
`uplift_step` a change in uplift rate partway through, and `interp_forcing`
interpolates any piecewise-linear history (here precipitation falling from
1.0 to 0.3 m/yr). A `P` series drives both the runoff and the mass balance; a
constant series reproduces the scalar run exactly, and the analytical reference
uses its time mean. The builders lay each series on the model's clock,
`linspace(to, T, nt)`, so a run that starts at a nonzero `to` passes the same
`to=` to them.

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
2D, `bl` — scalar or per-side — is consumed by **modes B, C and D only**: 2D mode
A holds its borders at their initial elevation and ignores `bl` entirely
(a known gap, not a per-side limitation), so a dict passed to a mode-A run is
accepted and does nothing. The analytical steady-state reference stays graded
to zero and is not offset-corrected per side, so the analytical overlay is not
meaningful across outlets at different datums (drawing the overlay warns;
`self.bl` is then only the fixed-side mean, a label).
