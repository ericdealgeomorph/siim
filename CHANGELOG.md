# Changelog

Short release notes. Public behavior and configuration are documented in the
guides and API reference under `docs/`.

## 0.10.1 — November 2026 (current)

### Behavioural changes

- **siim1d corrects the glacial slope at the terminus, so every 1D run with a
  glacier changes (modes A and B; `toe_slope_correction=False` restores the
  uncorrected toe step bit for bit, not the 0.10.0 model: the erosion
  solvers changed too, see below).** siim1d builds each cell's rise from the
  slope at its upstream node. Near a terminus the steady surface is singular,
  `S ~ Q**-phi_t` with the flux `Q` falling linearly to zero (`phi_t` the
  sliding law's thin-ice limit: 1/3 power, 1/2 Coulomb, `mu/nu` effective
  exponent), and that step pinned the terminus to its cell. Now every icy
  node whose downstream neighbour carries less ice flux scales its upwind
  slope by a factor of the flux ratio `q`, `(1 - phi_t)(1 - q)/(1 - q**(1 -
  phi_t))`, which makes the step exact for that surface and tends to 1 where
  the flux changes little across a cell; the last icy node, the one beside
  the outlet included, takes its slope from the sub-grid terminus, placed no
  nearer the node than `constants.TOE_ZETA_MIN` (1e-6 of a cell, so the
  factor stays finite), and takes the uncorrected step where that slope is
  no steeper than the bare reach it was extended along (see below) or where
  the glacier draining to that terminus holds fewer
  than `constants.TOE_MIN_CELLS` (5) icy nodes, a toe the grid does not
  resolve (counted on the terminus's own side of the divide, so an ice cap
  centred on the divide is judged flank by flank). The correction applies
  in the mode-A slope, the four 1D erosion kernels and the mode-B walk;
  siim2d is unchanged by default (see `toe_slope_correction_2d` below). At
  the paper suite's 1D reference (L 60 km, mode A unless stated):
  - mode A's divide swing on fine cells is gone: 19-26 m on 50 and 25 m
    cells before for the power and effective-exponent laws (9 m for Coulomb
    on 25 m cells), 0.01-0.03 m now on every grid from 400 to 25 m, with the
    exception below;
  - the divide and terminus converge at first order in `dx` for every law
    (orders 1.04-1.10), where the power and effective-exponent divides had
    no order before; the power-law limit, zo 4103.0 m and Lt 26 910 m,
    matches an independent shooting solution of the continuum problem to
    0.5 m and 15 m;
  - mode B settles on mode A's state: at 50 m with the cap on they agree to
    1 mm in zo and 1 cm in Lt (power, Coulomb), where they rested 6.5 m and
    3.6 m apart in zo;
  - on coarse cells the divide rises, by 10-33 m on 400 m cells, and it
    approaches its converged limit from above at first order; on 25 m cells
    it moves by under 1 m for the power and effective-exponent laws and
    rises 4-7 m for Coulomb;
  - mode A's swing starts at a lower relief: its onset falls from
    `d*sigma` about 0.8 to about 0.38 for the power law and to 0.52-0.54
    for Coulomb with the cap off, and it moves a quarter to a third as far
    with `dx`;
  - on coarse cells mode B's limit cycle passes through glaciers 1-4 cells
    long at the divide, where the last-node term over-corrects, so those
    glaciers take the uncorrected step there. On 250 m cells of the paper
    suite's fixed 1D strip (power law, mode B; 4-8 Myr, 10 kyr frames) the
    term on them left the divide ice-free in 16 % of the frames and put the
    cycle's trough (the mean of its minima) at 1453 m; with the guard the
    divide stays icy and the trough is 1508 m (1518, 1521 and 1523 m on 125,
    62.5 and 31 m cells, where the unguarded model gives 1501, 1515 and
    1523 m). Runs whose glaciers stay 5 cells or longer are unchanged by the
    guard, bit for bit.

  The exception is Coulomb with the cap on, which keeps a slow swing in mode
  A: 0.5-0.9 m on 100 and 50 m cells, 15 m on 25 m cells (decaying at about
  0.04 /Myr), and 46 m on 12.5 m cells (not decaying), against 32-35 m before
  on every grid from 200 to 25 m. In mode B with the cap on, the power law
  at the suite's reference erodibility cycled on 25 m cells under a first
  form of the correction (zo range 35 m, period 1.35 Myr): that is mode B's
  own cycle, which the uncorrected model shows on 12.5 m cells.

  The last icy node takes the uncorrected step wherever its corrected slope
  is no steeper than the bare slope its toe was extended along (the toe's
  surface is no gentler than the bed under it), not only where that slope is
  not positive. A node just turned icy keeps its bare surface for a step, a
  hair above the extended toe, and read a slope near zero, which mode A's
  closure turned into a one-step spike in H: 18.7 km against about 410 m on
  200 m cells (effective exponent, 0.5 mm above the toe). The mode-B walk
  takes the same rule. At the suite's 1D reference erodibility, mode A's
  transient peak H, where it moves, falls from 0.6-6.6 km to 0.3-0.5 km on
  2500 to 200 m cells, and mode A's attractors from 1250 to 25 m cells move
  by under 4 mm in zo and 3 cm in Lt (up to 6 cm and 35 cm for Coulomb on
  50 m cells at lower erodibilities). Mode B's effective-exponent divide on
  50 m cells moves 2.8 m onto mode A's, and on 25 m cells falls 1.2 m, away
  from it; mode B's limit cycles on 50 m cells keep their onset brackets and
  their periods to within 7 kyr. On 5000 and 2500 m cells, where the
  Coulomb toe is gentler than the bare reach below it, the Coulomb divide
  falls 65-72 m and the terminus retreats 2.3-2.4 km (mode A on both, mode B
  on 5000 m).

- **The implicit erosion solvers stop on the residual, not the step, and
  return the root Newton found instead of a point up to 1 mm off it, so every
  run that erodes changes (not bit-for-bit with 0.10.0).** The four bracketed
  Newton solvers (fluvial, eff-exp and power glacial, which are now one
  solver, and the D-inf/MFD multi-receiver one) stopped once a step fell under
  their 1 mm tolerance, which failed two ways. Once Newton reached the root,
  the next step rounded to nothing and landed on the bracket end that iterate
  had just set; the strict bracket test rejected it, bisected away, and the
  solver halved its way back until a step fell under 1 mm. And under a
  sub-linear exponent the slope near a receiver is steep, so a small step did
  not mean a small error: a fluvial solve at n = 0.22 returned 3.3e-4 m
  against a root of 0.054 m. The four now share one step that stops when the
  residual is under half the tolerance (its slope is at least 1, so it bounds
  the error), when the Newton step rounds to nothing, or when Newton leaves a
  bracket already narrower than the tolerance; the result lies within the
  tolerance and is never the receiver elevation itself. Measured on 2000
  random solves per exponent with drops of 1 mm to 3 km and prefactors from
  1e-8 to 1e8: at n = 1, the default fluvial exponent, 58 % of single-receiver
  solves came back more than 1e-6 m off (median 0.3 mm, up to 1 mm), as much
  as a whole step's erosion at a small dt; on those drops the solvers now
  return the root to round-off in at most two iterations (the multi-receiver
  one, at p = 1, in at most five; up to 24 before). Over exponents 0.2 to 2.7,
  on these drops and on sub-mm ones, the worst error falls from 4.7 mm to 0.35
  mm; above 1, on drops of 1 mm or more, the share of solves more than 1e-6 m
  off falls from 17-31 % to 0.1-1.3 %, at about one more iteration per
  solve (6.2 against 5.0 at the power-law exponent 1.8), with no measurable
  cost on a 31 x 31 mode-B run at n = 2. The two Coulomb solvers (one receiver
  and several), which stopped on a relative step under 1e-8 after Armijo
  backtracking, take the same rule at their own tolerance, 1e-8 x (1 + |z|):
  on 750 random solves with drops of 1 mm or more the one-receiver solver now
  returns the root to round-off (11 % were more than 1e-6 m off before, up to
  2.6e-5 m) in 2.0 residual evaluations, each an ice-thickness solve, against
  8.9 (up to 54), and a 2000-step 31 x 31 Coulomb mode-B run takes 1.6 s
  against 3.8 s; the multi-receiver one stays within its tolerance (0.8 % more
  than 1e-6 m off, against 15 %). On its own the Coulomb change moves a
  Coulomb run by under 3e-7 m over 100 steps. On the reference battery's base
  configuration mode B moves by 1 mm in the first step and 2 cm over 300
  steps; the fully glaciated mode-D run (glacial slope exponent 1.8, where the
  defect seldom fired) moves by under 1e-7 m over 100 steps; modes C and D
  then diverge as autogenic runs do, with their attractor statistics within
  about one seed standard deviation (over steps 1000 to 2000; four seeds in
  mode C, eight in mode D, where two sets of four seeds of the old code differ
  by as much), and the 1D power-law limit cycle keeps its period (794 and 793
  kyr). The reference battery is regenerated.

- **The ablation footprint's Hack width coefficient `width_hack_k` defaults
  to 0.2 in every 2D mode (0.5 before; 0.25 in mode D), so every 2D run with
  ice below the ELA changes; siim1d has no footprint and is unchanged.** Mode
  D no longer has a default of its own (`constants.MODE_D_WIDTH_HACK_K` is
  gone). Calibrated against the shallow-ice reference (MinSIA) at 1 km, the
  best footprint is 0.37-0.39 of the old one for the ice volume and
  0.32-0.41 for the trunk length, and the flagship ice cap does not depend
  on it. On an eroding 100 x 50 km landscape in modes B, C and D at 1 km and
  500 m, 0.2 stays bounded with its termini inside the domain and is calmer
  than 0.5 (no sawtooth surges at 500 m); it holds 30-64 % more ice with
  trunks 8-23 km longer, and under D8 routing the terminus moves by 1-3
  cells more often. Pass `width_hack_k=0.5` (0.25 in mode D) for the old
  footprint. The reference battery is regenerated.

### Features

- **`toe_slope_correction_2d` (siim2d, EXPERIMENTAL, off by default, so every
  2D run is unchanged; standalone driver only).** It ports siim1d's toe slope
  correction to the 2D routing graph, in every mode: each icy edge's glacial
  slope is scaled by `c(q) = (1 - phi_t)(1 - q)/(1 - q**(1 - phi_t))`, with
  `q` the receiver's flux over `Q* = min(Q_node, the receiver's icy inflow)`,
  and an edge to a bare receiver takes the sub-grid terminus's slope (its
  distance `zeta` held at least `constants.TOE_ZETA_MIN`) where at least
  `constants.TOE_MIN_CELLS` icy cells drain through the node, and the
  uncorrected step where that slope is no steeper than the bare receiver's
  own downhill slope (the toe's surface is no gentler than the bed under it;
  a node barely clearing the bare profile extended up to it read a slope near
  zero, which spiked H for one step, to 13.6 km in a mode-A single-flow
  benchmark run on 1 km cells and 33.7 km on 500 m cells); a terminus
  whose bare receiver is, or drains straight to, a base-level node takes no
  receiver extension, all as in siim1d. It needs a thin-ice exponent below 1 (an
  effective-exponent law with `mu/nu >= 1` is refused). Three known
  limitations: the terminus's extension is read from the step's starting
  surface and applied as a length divisor, so it lags by one step where siim1d
  extends the receiver implicitly (it may add to the wander of 2D
  landscapes); termini converging on one bare cell share its ablation, so each
  one's `zeta` is biased large; and the factor reads the cell-accumulated ice
  flux (it needs the pre-clamp signal of a bare receiver) while under a
  transect `closure_flux` (mode B's default) the closures read the transect
  flux. It stays off because it showed no gain in 2D (measured before the
  rule that gives a toe no steeper than the bed below it the uncorrected
  step was added): on the paper suite's D8 strip the uncorrected mode-B cycle trough already sits within 3 m of the 1D limit
  (1534 m) on cells from 250 to 31 m, and the correction moves it by 0.3 to
  5 m; on the 100 x 50 km benchmark case in mode B with single flow at 1 km it
  raises the divide's swing from 89 m to 147 m (a replicate run: 91 m)
  instead of damping it.

- **`footprint_from_H` and `footprint_width_factor` (siim2d, EXPERIMENTAL,
  off by default (`False`, `1.0`), so every 2D run is unchanged; standalone
  driver only).** Below the ELA a cell ablates over its glacier's footprint,
  the width times the flow length (at least the cell). `footprint_from_H=True`
  takes that width from the ice, `alpha_g*H` on the lagged pass thickness, in
  place of the Hack width `width_hack_k*A**width_hack_p`, closing the
  H -> width -> ablation loop the Hack width cuts (the model paper's
  sec:ablation); `footprint_width_factor` f multiplies the footprint width,
  either form, and nothing else (`alpha_g` in the cross-section, the closures
  and the carve is unchanged). Calibrated against MinSIA on c6's three frozen
  beds at 1 km (massif, notebook, flagship; ice volume and trunk geometry, f
  swept from 1 to 4): the H form fits best near f = 3.2 in mode C (multiple
  flow: one shared f of 3.29, set by the notebook's volume; at it the volume
  errors are +7, 0 and -10 % and the trunk lengths -11, 0 and +2 %). Two
  caveats. Mode B with D8 does not calibrate on the massif: at the
  notebook's f of 3.65 the massif holds 56 % too much ice and its trunk runs
  45 % long. And the trunk-length spread across beds is wide, so one f is a
  compromise: the massif's volume wants f above 4 while its terminus fits at
  2.6, and the flagship's cap is nearly insensitive to f (-11 to -9 % over
  the sweep). Unscaled (f = 1), the H form holds about 2.5 to 5 times the Hack
  footprint's ice on the massif and the notebook. On the eroding 100 x 50 km
  benchmark case over 8 Myr, the H form at f = 3 holds 59 to 94 % more ice
  than the Hack width at `width_hack_k=0.5` (modes B, C and D, 1 km and
  500 m cells; `width_hack_k=0.2`: 29 to 64 %), and its trunk termini hop more
  often. The Hack form scaled by f is mathematically the Hack width at
  `f*width_hack_k`, and bit for bit when one of the two is a power of two
  (`width_hack_k=0.5` with f = 0.4 reproduced `width_hack_k=0.2` bit for bit
  over 8 Myr in modes B and D at 1 km).

- **A 2D run can start from another run's end state, ice included, and then
  continues that run exactly.** `m.restart_state()` returns the state a
  finished run ended on (the bed, or mode A's ice surface; the ice thickness;
  the plate's previous ice column; the adaptive damping's weights; mode D's
  channel/sheet state), and the new `state` parameter starts a run from it: T,
  then T more from the end state, equals 2T in one go, bit for bit, in modes A
  to D (the escarpment's uplift wave aside, which restarts from
  `x_escarpment`), when the two clocks take the same step sizes. Before, a
  restart could pass only the end bed as `initial_topography`, and the ice
  regrew from zero: on the 100 x 50 km benchmark case (mode C, multiple flow,
  power law) it sat 4 to 11 % below the continued run's 400 kyr after the
  restart, depending on the spin-up and the volume measure read. The new
  `initial_ice_thickness` parameter is the simpler start, an `(ny, nx)` ice
  thickness on the supplied bed (modes B, C and D, with `noise_amplitude=0`;
  the other carried state starts cold). Both need the in-house driver. With
  neither set, this change leaves every run unchanged, bit for bit (the
  reference battery and a mode-D, sediment and flexure set gated against
  0.10.0, before the toe and solver changes above).

## 0.10.0 — November 2026

### Behavioural changes

- **An icy border cell with no icy donor now carries no ice, so mode-B/C/D
  runs whose ice reaches a border change (every B/C battery case does).** The
  2D mode-B kernels give an outflow border the thickness of its dominant icy
  donor (zero gradient). When a glacier retreated off the border, the border
  kept last step's thickness instead, and that column's own surface kept its
  mass balance positive: on the reference
  battery's configs (six seeds, second half of each run), 8 to 35 border cells
  per output frame held up to about 300 m of phantom ice beside ice-free
  ground. The border now copies the ice-free ground, as the 1D model always
  did; that count drops to 1 to 9, the interior ice volume of mode-C runs rises
  13 to 20 %, and mode-B runs move within their seed scatter. The B and C
  reference battery is regenerated.
- **The elastic plate is periodic along a looped axis, so every flexure run on
  a looped grid changes (the default boundary loops y).** A looped axis stands
  for an infinitely long source, a slice from a longer chain, but the plate
  solve treated its edges as free and held the plate flat about half a domain
  beyond them: the deflection depended on where the seam fell (a quarter to a
  third of its maximum on a 20 km grid) and was about a third of a periodic
  plate's. The plate is now solved in a Fourier basis along looped axes (the
  sine basis, held flat beyond the edge, along the others; Airy for the mean
  when both loop) and commutes with a seam shift. On the reference battery's
  flexure runs the larger rebound lifts the mean bed by about 1 to 2 m; ice
  volume and peak bed stay within the seed scatter. Grids with no looped axis
  are bit-for-bit. The seven flexure-on reference cases are regenerated.
- **The flexure ice load counts each channel once, spread over its footprint,
  so every run with `flexure=True` changes.** Each icy cell loaded the plate
  with its own channel cross-section, `alpha_g*H**2` over one cell length:
  right for a valley narrower than a cell, but a wider one piled its section on
  one cell, and where neighbouring cells were icy too the overlapping sections
  counted the same ice again (about 9 times on a broad cap). Each section is
  now spread across the cells it covers (a parabola of mean depth `H` over the
  width `alpha_g*H`, the carve's footprint, each cell taken by one channel):
  an isolated valley keeps its section, a sub-cell one is unchanged, a broad
  cap loads about `1.5*H` a cell (a thinner channel inside a thicker one's width
  counts as part of it; the fixed border seeds none, as in the carve). In mode
  D each footprint is weighted by its source's s and sheet cells add their slab. On the battery's flexure runs the ice load
  falls to 45-65 % of its old value; their mean bed, ice volume and peak bed
  stay within the seed scatter (the plate on a 20 km domain is held mostly by
  its rigidity). The seven flexure-on reference cases are regenerated.
- **Mode D's sheet ice loads the plate as a slab, so mode-D runs with
  `flexure=True` change.** The flexure ice load counts each icy cell as a
  channel cross-section, `alpha_g*H**2` per cell length, which counts a sheet
  cell `alpha_g*H/L` times its slab. Mode D now gives each cell
  `s*alpha_g*H**2/L + (1 - s)*H` with s its valley <-> sheet reading: the
  section on valley cells, a slab of thickness `H` on sheet cells (a thick
  sheet cell loads less, one thinner than `L/alpha_g` more). On the mode-D test
  run the sheet cells (13 to 20 % of the ice) had counted 5.7 to 9.5 times
  their slab, and the total load drops to 88 to 96 % of its old value. s is 0
  or 1, so a cell that flips steps its load; there, flips made a median 4 % of
  each step's load change and the plate's per-step rebound stayed below 2.2 cm.
  Modes A, B and C are unchanged (bit-for-bit, 14/14 flexure-on battery
  arrays); their wide valleys still count the full section.
- **The linear eroders (`n = nu = 1`, 1D and 2D) no longer raise a cell toward
  a higher receiver.** Their closed form skipped the guard every nonlinear solver
  has: a pit below its receiver was raised (from 10 m to 15 m in a
  one-cell test), and a D-inf or MFD cell was under-eroded whenever a
  receiver lay above its new elevation (raised, when above the cell itself).
  The closed form now returns the same root as the `n = 1` Newton solver,
  and is bit-for-bit unchanged when every receiver lies at or below the
  result. Runs with `nu != 1` (the defaults) are unaffected.
- **Modes B, C and D now damp the ice state by default with a new adaptive
  preset, `inner_damping='adaptive'`, so every default mode-B/C/D run
  changes.** Undamped mode B settles into a period-2 planview flicker, and
  mode C's `routing_relax = 0.5` damped only the routing surface, not the ice:
  the flicker biased volumes up (about 3.5 % on a frozen bed, and on eroding
  beds under `flow_routing='multiple'` it compounded through erosion into a
  runaway). The new `inner_damping` chooses how each pass commits:
  `'adaptive_capped'` and `'adaptive'` damp only the cells whose update keeps
  reversing sign (each cell's weight halves on a reversal and grows by 1.25
  otherwise, between a floor of 1/3 or 0.2 and 1), at one pass a step;
  `'omega'` is the uniform `inner_relax`/`inner_passes` move, whose mode-B/C
  dials default to ω = ½ over 2 passes (a steady change trails by about a
  third of a step, at 1.4–1.8× the cost; the step-to-step thickness change
  falls 3–8× on the standard fixtures) and mode D's to ω = ⅓ over one. The
  default preset lags at most 4 steps on a flickering cell (the capped one 2)
  and not at all on a cleanly moving one, at the cost of one pass; against
  ω ½ × 2 the adaptive presets measured less twitch at a fixed climate and a
  jumpier ice front under a moving one, and they track coarse-`dt` forcing
  worse. On the ice-dome suite the default's central thickness lies within
  −2.6 to +1.2 % of the references (the capped preset's within −2.6 to
  +10.3 %, with a 1–5 % late oscillation). Mode D's MinSIA calibration moves
  (like-for-like volume, ω ⅓ → adaptive: notebook −0.5 → −6.6 %, massif
  +5.5 → +16.9 %, flagship +1.1 → −0.8 %). An explicit `inner_relax` picks
  `'omega'`, so `inner_relax=1, inner_passes=1` is the undamped model; an
  adaptive preset carries its floor in `inner_relax` and rejects an explicit
  one. The fastscape facade cannot iterate within a step, so it runs
  undamped; `siim2d`'s `driver='xsimlab'` asks for
  `inner_relax=1, inner_passes=1`. Mode A is unchanged. With
  `H_diffusivity > 0` any of these attenuates the diffusion correction, as
  `inner_relax < 1` already did (the model warns). The mode-B/C cases of the
  reference battery are regenerated.
- **`routing_relax` is removed, from `siim2d` and from the fastscape facade's
  `surf2erode` processes (`surf2erode__routing_relax`).** It relaxed the
  routing surface alone and left the ice flickering; the ice damping above
  replaces it. Passing it to `siim2d` raises a `ValueError` naming the
  replacement. Every run that had it at 0.5 changes: the facade's mode C (its
  `TrunkSurfaceToErode` defaulted it to 0.5) and `siim2d`'s mode C at
  `inner_relax=1, inner_passes=1` (including every mode-C run with
  `driver='xsimlab'`), which now run undamped; every run that had it at 0 is
  bit-for-bit unchanged (measured on the reference battery). A model saved
  with `routing_relax` still loads, with a warning: its outputs are as saved,
  and the rebuilt model runs this version's damping.
- **The fabricated trunk surface now feeds the router graph only, so every
  run with a trunk surface — the mode-C default — changes.** `trunk_surface` builds its V-shaped
  cross-valley dip for the flow router alone; the accumulator's mass balance and
  the ice-surface hillslope diffusion are now evaluated on the dynamic
  surface `zb + hc*H`, where before they read the fabricated one. The mass
  balance no longer depends on `trunk_dip_k`, and the advice to re-tune `zELA`
  before comparing mode-C runs with runs without a trunk surface is withdrawn.
  Measured over 7–10 autogenic cycles on the 250 m mode-C fixture and the 500 m
  quickstart landscape: relief +2–3 % under D8 (less under D-inf and MFD), the
  cycle 4–10 % longer, the ice inside the cycle scatter with no consistent sign.
  The mode-C cases of the reference battery are regenerated. Runs with
  `trunk_surface=False` — mode A, plain mode B, mode C with the trunk surface
  off — are unchanged bit-for-bit. Evidence: `docs/dev/mfd_valleys_e1e2.md` §10.
  - **`flow_routing='mfd'` supports mode C** — the carve and the trunk surface,
    through `siim.siim2d.siim`, `siim.fastscape.glacial_processes(routing='mfd')`
    and the adapter processes alike. It still requires mode B or C (mode A has
    only the one-cell closure), a transect `closure_flux` and square cells, and
    like every router it refuses `routing_filter_factor > 0` together with
    `trunk_surface=True`. MFD in valleys needs the trunk resolved: the V and the
    carve engage only where `alpha_g*H/2 > dx`, a trunk wider than two cells at
    its thinnest; on a grid that leaves the trunk sub-cell MFD carries more ice
    than D8 at the glacial minimum (+17 % volume / +25 % mean thickness on the
    500 m quickstart landscape), while at 250 m the two agree within D8's own
    cycle scatter and MFD's bed has about 7 % less relief. MFD mode C costs
    1.6–1.9× D8's wall time at 250 m.
  - **`trunk_dip_k` defaults to `None`, resolved per router:** `1.0`
    (`constants.TRUNK_DIP_K_MFD`) under `'mfd'`, `0.6` (`constants.TRUNK_DIP_K`,
    unchanged) under `'single'` and `'dinf'`. A spreading router leaks part of
    an axis cell's flux off the trunk until the cross-valley dip matches the
    down-valley slope (closed-form axis leak `2s/(1+2s)`,
    `s = max(1-k, 0)/sqrt(2)`, zero at k >= 1); D-infinity's leak clamps to zero
    at k = 1 and its cycle turns into a multi-Myr transient there, so it keeps
    0.6. An explicit value is honoured. Like every sentinel default, an omitted
    dip is saved as `None` and re-resolved from the current constants when the
    run is loaded; `load()` now warns when a file was saved by another siim
    version.
  - **`siim.fastscape`: `TrunkSurfaceToErode` publishes `routing_elevation` in
    the `router_surface` group; the in-house routers route on that surface when
    present and on `elevation` otherwise**, so the accumulator and the hillslope
    diffusion keep the dynamic `elevation`. The dip input
    `surf2erode__trunk_dip_k` defaults per router through both facades:
    `siim.siim2d.siim` passes the resolved number, and
    `glacial_processes(routing='mfd')` wires a trunk-surface variant whose
    default is 1.0; an explicit value wins. A composition built by hand without
    `glacial_processes` gets the process default 0.6 under MFD too. A hand-built
    MFD model in mode A now raises when it runs (the erosion process refuses the
    one-cell closure), not only through the facade.

- **D-inf routing now normalises the effective slope by default
  (`slope_normalise`), so D-inf thickness and erosion results change wherever
  flow is not cardinal.** The mode-A/B reduction
  `S = sum_k w_k (zs_i - zs_rk)/L_k` is the surface gradient PROJECTED on the
  weighted receiver direction, not its magnitude: on a plane tilted pi/8 —
  mid-facet, D-inf's worst angle, with both receivers 22.5 degrees off-axis —
  it returned 0.9239 of the true gradient, a 7.6 % slope deficit that inflated
  `H` there by up to about 5 % with a period-pi/4 azimuthal signature, and was
  exact only on the grid axes. Dividing by `gamma = |sum_k w_k e_k|` recovers
  the gradient magnitude exactly on a plane; it is applied as gamma-scaled link
  lengths, so the thickness closure and all three eroders read one slope.
  Expect thinner ice at mid-facet flow angles (up to about -5 % in `H`), and a
  correspondingly smaller azimuthal texture in radially symmetric runs.
  Cardinal-plane results, and every `flow_routing='single'` run, are unchanged
  bit-for-bit (D-inf collapses to one receiver there, `gamma == 1`).
  - The previous behaviour is reachable with the new `slope_normalise`
    parameter: `None` (default) resolves from the router
    (`siim._core.router.SLOPE_NORMALISED_ROUTERS`), `False` restores the
    unnormalised reduction, and under `'single'` the flag is a no-op. Under
    `'mfd'` an explicit `False` raises. The adapter mirrors this on
    `glacial_spl__slope_normalise` (`None` = derived from receiver arity).
  - **The committed CI reference battery
    (`siim/tests/data/reference/`) must be regenerated for the `dinf` cases.**
    The battery is platform-bound (frozen on linux/x86-64 in CI) and cannot be
    refrozen off-CI; the `single` cases are unaffected and still reproduce
    bit-for-bit.

- **`landscape`'s `z_min` defaults to -500 m, not 0 m.** The bed colours
  saturate at `z_min`, so every bed below sea level was drawn in the 0 m
  colour (an overdeepened trough became one black slab). Pass `z_min=0` for
  the old scale; the section and hypsometry panels follow it.
- **`landscape`: vertical exaggeration 4 for the hillshade and the section, and
  ice-bar arrows.** The hillshade's `ve` defaults to 4 (was 2). The section
  panel was a fixed share of the map width, so its exaggeration ran from about
  2x on a 20 km valley to 30x on an 800 km map; a new `section_ve` (default 4)
  now sizes its height for that exaggeration over `z_min`..`z_max`, clamped to
  0.8-2.6 in, and the panel still prints the exaggeration it got (large domains
  hit the lower clamp and read higher). The ice colorbar gets an arrow where the
  model's ice column (`hc_over_H*H`) runs past `H_min`/`H_max`, as the bed bar
  does (new `extend_ice`; a movie applies the same rule once over its frames);
  at the default run-global `H_max` it has none. The averaged section's IQR legend moves out of the panel into
  one row in the gap above it, which grows by one legend line to hold it, and
  its labels shorten to 'ice IQR' (the ice thickness's) and 'bed IQR'.
- **The analytical reference is built on the run's `hc_over_H`.** `siim1d` and
  `siim2d` forward `hc_over_H` to their `SteadyStateProfile`, which now takes it (default
  `constants.HC_OVER_H` = 1.5) and reports its bed as `z - hc_over_H*H`, so the
  profile overlay, `rms_vs_analytical` and the reference agree at any datum.
  The solved surface and thickness do not depend on it. At the default nothing
  changes; at another value the overlaid bed moves by `(1.5 - hc_over_H)*H`.
  `siim1d` now also refuses an infinite `hc_over_H`, as `siim2d` does.
- **`animate_landscape` is about 2.2x faster and takes a `quality` preset;
  `landscape`'s contours move by a few anti-aliased pixels.** On a 201 x 201
  run of 301 frames the default movie takes 29 s, against 65 s before. The
  parallel workers write raw frames that stream into ffmpeg as they render;
  compressing each frame to a PNG was 40 % of a movie, and the encode no
  longer waits for the last frame. On that parallel path the MP4 replaces
  `path` only once it is complete. `landscape` settles its axes without a second full render, and
  bins the hypsometry by a search among its few edges instead of sorting every
  value; both leave every frame pixel-identical. The terrain contours and the
  glacier margin use contourpy's `'serial'` algorithm, which traces the same
  levels 1.6-1.7x faster and changes at most a few pixels per frame (12 px, 5
  colour levels, in the frames checked). `quality='high'` (the default) keeps
  the still render's look; `quality='draft'`, for research movies that will
  soon be replaced, renders at most at `oversample=2` (`style='raw'` stays at
  1) with x264's `veryfast` preset: 17-18 s on the same run, 3.7-3.8x. Smooth's
  `ice_sigma_cells` now defaults to half the oversample, the same half-cell
  outline smoothing in model cells at any oversample (2 at the default 4, so
  default renders are unchanged; it was a fixed 2). A smooth render at an
  explicit `oversample` other than 4 does change (4 px at 8, 0.5 px at 1);
  pass `ice_sigma_cells=2` for the old outline. The automatic worker count
  times its probe frame with its savefig and goes parallel past a projected 10
  s (was the render alone and 30 s): short movies had taken the serial path,
  4.5x slower on a 101-frame 101 x 101 run.
- **`mfd_exponent` takes `inf`, and a finite exponent above 50 is taken as
  `inf` with a warning.** The MFD weights are `(S_k/S_max)^p`; at a large p
  those of all but the steepest link underflow to exact zeros, which stay in
  the receiver list as links carrying nothing (common from p = 64; at 50 or
  below none in the runs tested, though a near-flat link beside a steep one can
  still do it). `mfd_exponent=inf`, refused before, is now the exact limit: the
  steepest link alone, equal steepest links sharing. It is still the MFD code
  path, not `flow_routing='single'`. A finite exponent above
  `constants.MFD_EXPONENT_MAX = 50` becomes `inf` and says so: above 50 each
  further step in p moved every mode-D statistic by about one twin-run spread.
  In two 400-step mode-D test runs, 50 and `inf` differed by up to 1 % in ice
  extent, 3.3 % in ice volume and 2.5-6 % in the time-mean thickness and
  erosion fields. Runs at an exponent of 50 or below are bit-for-bit.
- **Faster steps, bit for bit: TBB when Numba can load it, the eroder's levels
  at the performance-core thread count, and a narrower transect scan.** Every
  output is unchanged on either threading layer and at any thread count; what
  changes is the layer a process runs and the eroder's default under MFD.
  - `import siim` now sets Numba's threading-layer priority
    (`NUMBA_THREADING_LAYER_PRIORITY="tbb workqueue omp"`) instead of pinning
    `NUMBA_THREADING_LAYER=workqueue`: Numba runs TBB when it can load the
    library and workqueue otherwise, never OpenMP. An explicit
    `NUMBA_THREADING_LAYER`, or your own priority, still wins. The workqueue
    pin dated from a flexure crash under Jupyter (Numba 0.65); TBB with flexure
    ran clean in a Jupyter kernel on Numba 0.68.
  - Under `flow_routing='mfd'`, mode D's default routing, an unset
    `parallel_erode` follows the layer: on under TBB, off on workqueue, where
    MFD's many small drainage levels make the parallel eroder slower. An
    explicit value wins. This replaces mode D's own default of `False`
    (`constants.MODE_D_PARALLEL_ERODE` gives way to
    `MFD_PARALLEL_ERODE_LAYERS = ('tbb',)`): a mode-B or -C run under MFD on
    workqueue now gets the serial eroder without being told.
  - On Apple silicon the parallel eroder's per-level loops use as many threads
    as the Mac has performance cores; every other kernel keeps Numba's full
    thread count. `NUMBA_NUM_THREADS` and `numba.set_num_threads` still cap
    everything.
  - The transect closure flux scans only the strip of each row that can hold a
    crossing link (the same terms, in the same order): 1.4-2.2x on that kernel.
  - TBB is optional: `environment.yml` installs it, and
    `pip install 'siim-lem[tbb]'` adds PyPI's `tbb` where a wheel exists
    (x86_64 Linux, Windows, Intel Macs) and nothing on Apple silicon. In a
    Linux virtual environment Numba finds that library only with the
    environment's `lib` directory on `LD_LIBRARY_PATH` (tested); without it
    SIIM runs on workqueue as before.
  - Per step against 0.9.10, on a 4 + 6 core laptop with TBB: on a 101 x 101
    grid, 1.27x (Coulomb law) and 1.85-1.87x (power law) in modes B and C,
    1.42x and 1.10x in mode D; on an icy 101 x 1001 grid, 1.18x in mode C and
    1.44x in mode D. On workqueue the gain is 1.02-1.20x.
  - Less bookkeeping per step, with every output unchanged: mode D's border
    cells take their inherited confinedness in compiled code, and in every
    mode the routing-topology outputs (`basin_out`, `receivers_out`,
    `stack_out`, with the D8 basin labelling the MFD and D-infinity routers
    run for them) are built only on the steps that save a frame: a further
    1.06x (Coulomb) and 1.12x (power) per step in mode D on a 101 x 101 grid,
    1.04x on a 101 x 1001 grid.
  - Mode D's channel/sheet reading and the MFD router cost less, with every
    output unchanged (checked on macOS and Linux): the routing filter runs as
    one compiled kernel, the reading takes its ring medians in place and its
    own routing pass accumulates ice alone, the pit fill keeps its priority
    queue to the depression rims, and the MFD router builds its links and its
    stack in one pass each. Per step in mode D on a quiet 4 + 6 core Mac with
    TBB: 1.54x on the 201 x 201 notebook grid, 1.29x on an icy 801 x 301
    escarpment. The pit fill and the MFD router serve every mode that uses
    them.
- **`landscape` draws no trunk ribbons by default, and the veil's opacity
  follows absolute ice depth.** Both presets now resolve `trunk_display` to
  `'none'`. The footprint fill already draws the carve's parabolic cross-section
  wherever a glacier spans more than about two cells. On sheet runs the
  ribbons drew rings, because their trunk class reads sheet ice as channel.
  `trunk_display='ribbons'` still draws them. The veil's opacity is now an
  S-curve in absolute depth, `t^k / (t^k + (1 - t)^k)` with
  `t = depth / ice_opaque_depth` (new knob, default 100 m) and
  `k = ice_alpha_sharpness` (new knob, default 2), with no floor, so a few
  metres of ice no longer tint the terrain (10 m: 1.2 % opaque, 20 m: 6 %,
  50 m: 50 %, 80 m: 94 %, 100 m and deeper: opaque). It used to be a floored ramp on depth / `H_max`, which left
  300 m trunks translucent on sheet runs, where `H_max` is kilometres. Its
  colours are unchanged. The margin line now follows the ice edge on a default
  map, where it used to follow the ribbon edge, and is drawn at 0.3 pt (`margin_lw`,
  was 1.0).
- **`ice_time_avg` removed.** `landscape` and `animate_landscape` no longer
  take it. It averaged trailing output frames, so at coarsely spaced saves it
  double-exposed two glacial epochs, and it had been off by default since
  2026-07-07. Passing it, with any value including 1, raises a `ValueError`
  saying it was removed in 0.10.0 and why; drop the argument. Every render
  shows frame `i` alone, and a render that left it at its default is
  unchanged bit for bit (every field and style, with and without a section,
  on a glaciated test run).
- **The default bed colormap is Crameri's turku with its darkest 20 %
  trimmed, not `gist_earth`.** It applies to `landscape` (`cmap_bed`, with its
  section and hypsometry) and to `map(field='bedrock')`. `gist_earth` is not
  perceptually uniform, is black at its low end and has blues that compete
  with the ice; turku has no blue, and the trim starts it at a dark olive-grey
  instead of black. `'bedrock+speed'` keeps its light grey ramp. The colour
  table is embedded in `siim/plotting/_colormaps.py` with its attribution
  (Crameri, *Scientific colour maps* 8.0, doi:10.5281/zenodo.8035877) and the
  MIT notice cmcrameri distributes it under, so siim does not depend on
  cmcrameri. Pass `cmap_bed='gist_earth'` (or `cmap='gist_earth'` to `map`)
  for the old look.
- **Plot text follows the active rcParams.** The plotters no longer hard-code
  font sizes (38 literals of 8 to 11 pt: titles, axis and colorbar labels, tick
  labels, legends and in-panel notes), so a house style such as dealfig's
  `paper` sets them. At matplotlib's defaults tick labels go from 9 to 10 pt,
  titles from 9-11 to 12 pt, legends from 8 to 10 pt and in-panel notes from
  8-9 to 10 pt; axis and colorbar labels stay at 10 pt. Figure sizes are
  unchanged, except that `landscape`'s text-holding margins (left, bottom,
  top, the bar side, the gap before the second bar and above the section) now
  scale with the active tick and label sizes, relative to the 9 pt / 10 pt they
  were tuned for: with fixed inches the labels overhung the canvas by up to
  0.27 in under dealfig's `paper` style (movie frames are not cropped). Now
  nothing overhangs at matplotlib's defaults, or at any width under `paper`
  but the speed map's rotated bar label at 5 in (0.09 in), and a `landscape`
  figure grows by the extra margin. Every `m.plot` figure changes (the analytical package's own
  `plot` methods do not), and nothing else does: with the fonts and the
  colormap put back, every render in a 71-image set (all four landscape
  fields in both styles, the sections, maps, profiles, viewers, basin
  summaries, the 1D plots and the limit cycle) is bit for bit the old one.
- **Plots give each quantity one colour, one label and the house widths, so
  nearly every figure changes** (66 of the 71-image set; `map(field='bedrock')`,
  its viewer and the limit-cycle phase portrait do not).
  - *Colours.* The colour roles moved to `siim._palette.COLORS`
    (matplotlib-free; `siim.plotting._style.COLORS` re-exports them), so the
    analytical package shares them. Its ELA was red in `GeneralProfile.plot`,
    `MarginalCoulombProfile.plot` and `SteadyStateProfile.plot` and goldenrod in
    `plot_profile`; it is now the model plotters' `COLORS['ela']`. The paper
    renderer `draw_profile` (and those `plot` methods) defaults its ice fill to
    `COLORS['ice_fill']` (was `'#b0d6e2'`) and its depth-faded bed to
    `COLORS['bed']` (was `'#888888'`); pass `ice_color='#b0d6e2',
    bedrock_color='#888888'` for the old look. `map(field='ice')` and
    `flux('ice')` take the landscape veil's glacier ramp (were `Blues`), and
    `flux('water')` a ramp from a pale tint to the profile panels' water-flux
    colour (was `YlOrRd`). `landscape`'s `lake_color` defaults to
    `COLORS['water']` (`'#56b4e9'`, the 1D ponds' colour; was `'#a8c8e8'`).
  - *Bed.* Every profile elevation panel (1D and 2D `profile`, `view_profile`,
    `animate_profile`, `view_xsection` and `analytical`) draws the bed as a
    `COLORS['bed']` line over one translucent fill down to the panel's floor.
    The 1D `analytical()` filled from zero, so a bed below sea level was
    filled upward to zero.
  - *Analytical references.* Both `analytical()` methods draw through the
    profile renderer: the reference's bed and ice fills, its dashed lines, its
    ELA (new), the bistable fluvial alternative dotted (was dashed grey) and
    `profile()`'s one-panel figure and limits; the 2D x-axis spans the
    reference (was 0 to 1.05 L). A fluvial (ice-free) reference, here and in
    `profile` overlays, is one dashed `Analytical bedrock` line, no longer an
    `Analytical ice surface`. `SteadyStateProfile.plot_profile`'s elevation
    panel is `plot()` itself (depth-faded bed, legend `Surface` and `ELA`); its
    other panels take the shared colours.
  - *Labels.* `Distance (km)` is retired: the 1D profile, `analytical()` and
    limit-cycle panels read `x (km)` (the grid coordinate), the 2D profile and
    `analytical()` `Distance from divide (km)`, and `plot_profile` as `plot`.
    `plot_profile`'s thickness panel, which plots the column depth
    `surface - bed = hc_over_H*H`, reads `Ice column depth (m)` (was
    `Ice thickness (m)`).
  - *Units.* `flux()` and the basin maps of `largest_basins` and
    `sediment_history` are drawn in km (they were laid out in metres under km
    tick labels), so an overlay in km lands where it should.
  - *Widths* (inches, house standard 12 / 6): `landscape` and
    `animate_landscape` 8 -> 12; `largest_basins`, `largest_basins_history`,
    `sediment_history` and `steady_state` 10 -> 12; `profile`,
    `view_profile`, `animate_profile` and `view_xsection` 8 -> 6 for one field
    (aspect 0.38) and 12 for a stack (aspect 0.25, so a panel stays about
    3 in tall; was 0.38); `analytical()` (1D and 2D) 8 x 3.04 -> 6 x 2.28;
    `hacks_law` 8 -> 6 (aspect 0.38 -> 0.5, the same 3 in height); `flux`
    6.5 x 5 -> 6 wide at the domain's aspect; the analytical package's
    `plot` 8 x 3 -> 6 x 2.25 and `plot_profile` 10 x 2n -> 6 x 2.28 for one
    field, 12 x 3n for a stack. `map` stays at 6.
  - *Layout.* `flux()`, the analytical package's `plot` and `plot_profile`
    use constrained layout (none, none and `tight_layout` before), as every
    figure but `landscape`, which keeps its own geometry.

### Fixes

- **A `zT` given without `zELA` now sets the ELA in the 1D and 2D models.**
  `zELA` defaulted to 1500 m, so `siim1d({'zT': 3000})` and
  `siim2d({'zT': 3000})` ran at zELA 1500 m, zT 1600 m and dropped the `zT`.
  `zELA` now defaults to `None`, as in `SteadyStateProfile`: with neither given
  it is `constants.ZELA`, with `zT` alone it is `zT - P/beta` (step by step for
  a `P` series), and an explicit `zELA` still wins. Runs that set neither, or
  set `zELA`, are bit-for-bit.
  A file saved by an earlier version from a run given `zT` alone holds outputs
  computed at zELA 1500 m; loaded now, the model reads the derived `zELA`.
- **Both models take a 0-d array as the scalar it is, and refuse a `beta` that
  is not > 0.** In siim2d a 0-d `P` or `bl` raised a `TypeError` from inside
  the error message meant for a bad series, and a 0-d `zELA` or `U` was refused
  for its shape, as a 0-d `P` was in siim1d. Every parameter given as a 0-d
  array now acts as its scalar. `beta = 0` raised a bare `ZeroDivisionError`
  in both models; a zero, negative or non-finite `beta` now raises a
  `ValueError` that names it.
- **`animate_landscape`: the user's ffmpeg settings, a failed frame, windowed
  movies.** Both the serial and the parallel path now pass
  `rcParams['animation.ffmpeg_args']` to ffmpeg, with a `quality='draft'`
  preset added after them (the serial draft replaced them, the parallel path
  ignored them), and the parallel path names its codec from
  `rcParams['animation.codec']` as the serial one does, asking for `yuv420p`
  only with h264. When a frame fails,
  the frames still queued for the workers are skipped instead of rendered. The
  workers are handed only the frames the movie encodes, each once, and the RAM
  cap on the worker count counts just those, so a short window of a long run
  can use more workers. At the default settings movies are byte-identical.
- **A still with `ice_cmap` and no `H_max` no longer shows a 'max' arrow on
  its ice colour bar.** The scale's top is then the frame's own thickest drawn
  column, so no colour on the map saturates; the arrow compared it with the model's
  unsmoothed column, which is a hair thicker, and appeared on some frames. A
  pinned `H_max`, and a movie's frozen scale, keep their arrows.
- **`animate_landscape` from a script with no `if __name__ == '__main__':`
  guard no longer re-runs the script in every worker.** The parallel renderer
  spawns its workers, and a spawned process re-imports the launching script:
  an unguarded one ran its model again in each worker and then failed with
  `BrokenProcessPool`. The script's path is now hidden from the workers, as it
  is absent in a notebook, and the workers take the calling process's
  `rcParams` instead: a style set in a script or a notebook now reaches a
  parallel movie as it always reached a serial one (the frames were drawn at
  matplotlib's defaults from a notebook). A colormap the script registers by
  name does not reach them; a keyword or `rcParams` object of a class the
  script defines keeps the script visible, so it still needs the guard. Also
  on the parallel path: the finished movie is moved beside `path` and swapped
  in atomically (an output directory on another filesystem than the temporary
  one was copied over the old movie in place), `yuv420p` is not forced when
  your `rcParams['animation.ffmpeg_args']` name a `-pix_fmt`, and
  `rcParams['animation.bitrate']` reaches ffmpeg as on the serial path. On
  both paths `quality='draft'` passes x264's `-preset veryfast` with the
  `h264` codec alone; another `rcParams['animation.codec']` was handed a
  preset it may refuse.
- **Mode names are case-insensitive in every front end, and the fastscape
  facade accepts the legible ones.** `glacial_processes` and `glacial_model`
  upper-cased `mode` before normalizing it, so `'ice_surface'` and
  `'bedrock+ice_thickness'` raised the very error that lists them as valid,
  while siim1d and siim2d refused `'a'` and `'b'`. `normalize_mode` now ignores
  case for every name, and all three front ends call it on the mode as given.
- **`incomplete_beta(1, a, b)` and `incomplete_beta_compl(0, a, b)` are +inf
  for every b <= 0.** At a >= 1 with b <= -1 (for example (1.5, -1) or
  (3, -2.5)) the recurrence formed inf - inf and returned NaN with a
  RuntimeWarning; x = 1 now returns +inf from both, as documented, through an
  exact branch. No profile changes: the profile classes call
  `incomplete_beta_compl` at a = 1 - phi < 1, where it already returned +inf.
- **The analytical profiles no longer go O(1) wrong when d·φ is one rounding
  step off 1.** For tiny nonzero b = (1 − d·φ)/k the incomplete beta subtracted
  two ~1/b terms (the reflection for b > 0, the recurrence for b < 0), so a φ
  from `np.arange` near 1/d gave Lt = 99 715 m where the marginal profile is
  32 548 m. Below |b| = 10⁻³·min(a, 1) it now uses an exact small-b form
  (relative error below 10⁻¹² against 50-digit quadrature); b = 0 is unchanged.
- **zELA = 0 no longer raises.** The analytical thickness was zELA·NH with
  NH = …/zELA; it is now formed without the division, so the 1D and 2D models
  and `SteadyStateProfile` build their reference at an ELA at base level, and
  `NH` is infinite there. Analytical thickness values move at the last bit.
- **A scalar `boundary_status` behaves like the 4-list everywhere.** The uplift
  mask and the driver's initial surface iterated the string's characters, so
  `boundary_status='fixed_value'` gave the fixed borders block uplift and noise
  (about 100 m off the 4-list run); `plot.steady_state` misread it too. Runs
  given a 4-list are bit-for-bit unchanged.
- **On a looped axis the trunk surface no longer depends on where the seam
  falls, so every default mode-C run on a looped grid changes.** Its
  cross-slope came from `np.gradient`, one-sided at the edge rows even across a
  periodic seam, so a trunk on the seam dug a V up to ~110 m deeper than the
  same trunk elsewhere. The edge rows/columns of a looped axis now take the
  central difference across the seam. Mode D with `trunk_surface=True` changes
  the same way; modes A and B are bit-for-bit unchanged. The mode-C cases of
  the reference battery are regenerated.
- **Hillslope diffusion wraps across a looped seam, so every default run on a
  looped grid changes.** The ADI treated a `'looped'` edge as a no-flux wall
  while the router, carve, filter and thickness diffusion wrap (relief on the
  seam row diffused into its neighbour on one side only). On a looped axis the
  edge rows now join the implicit solve with their neighbours across the seam
  (a cyclic tridiagonal solve); without a looped axis the solve is the
  fastscape twin, bit-for-bit. The whole reference battery is regenerated
  (mode A moves by at most 0.1 m on it; modes B and C diverge further over the
  run).
- **`boundary_status` is validated.** A misspelled side (`'fixed'`) used to act
  as a closed edge silently, a `'looped'` set on one side of an axis only gave a
  half-periodic seam into an outlet, and a domain without a `'fixed_value'` side
  had no outlet at all (nothing drained). Each now raises a `ValueError`.
- **`glacial_model`'s border-bed uplift defaults to the local uplift rate, as
  siim2d's does.** A direct composition that left
  `glacial_spl__border_bed_uplift` unset got 0, so a border bed the ice
  deepened below `bl` never recovered, and `glacial_model(mode='C')` did not
  match siim2d mode C as documented. The input now defaults to `None`, which
  takes the rate `GlacialBlockUplift` or `WaveUplift` publishes each step; a
  value given is used alone. siim2d's own runs are unchanged.
- **`SteadyStateProfile` checks its inputs and re-derives on `solve()`.** A
  misspelled boundary condition crashed later on a missing attribute, an
  unknown `sliding_law` silently ran the power law and `sigma = 0` divided by
  zero; each now raises a `ValueError`. After an input changed, `solve()` re-ran
  only the root solve and mixed the new input with stale derived values (`K`,
  `Co`, `lam`, the grid); it now rebuilds them first, and a `GeneralProfile`'s
  default `xo` follows a new `L`. A `zT` given alone was ignored for the
  default `zELA`; it now sets `zELA`.
- **The 1D model: `rms_vs_analytical` compares beds on one datum, and inputs
  are copied.** With `hc_over_H` overridden, the model bed used the override
  and the analytical bed 1.5 (226 m of rms for a model planted exactly on the
  analytical); the analytical bed is rebuilt on the model's `hc_over_H`. The 1D
  model also keeps its own copy of its inputs, and a NaN `flotation_ramp` or
  `H_diffusivity` is refused, as in 2D.
- **The fastscape adapter refuses what siim2d refuses.** A direct
  `glacial_model` composition ran these silently: an unknown
  `glacial_spl__closure_flux` string ran the transect closure; a carve setting
  (`carve_width=True`) on the non-carving mode-A/B classes was ignored, and
  mode A took any closure though it has only `'cell'`; a NaN
  `surf2erode__trunk_dip_k` poisoned the router surface and a negative one
  turned the V into a ridge; MFD ran with `slope_normalise=False`; and a
  requested `flow__lake_depth` came back all zeros. Each now raises. The
  mode-A/B classes default `carve_width` to False (they never carve), and
  `trunk_dip_k=None` resolves to the router's default as documented.
- **The 2D model keeps its own copy of every input.** `U`, `P`, `bl` and
  `initial_topography` arrays aliased the caller's, so mutating one after
  construction changed the run while `m.U` and the analytical reference kept
  the old value.
- **A `zT` series pairs with a `P` series step by step.** With both given as
  series, `zELA` used the time-mean `P` and came out constant; it is now
  `zT(t) - P(t)/beta`.
- **Under the escarpment wave, the border beds follow the wave.** With
  `uplift_type='wave'` and no `border_bed_uplift`, each base-level border bed
  rose every step at the run-average rate `U_inf + delta_h/T` meant for the
  analytical reference, about 8 times the wave's own peak at the escarpment
  defaults and wherever the wave was. It now rises at the wave's local rate,
  in both drivers, so with `U_inf = 0` a border the wave has not reached gets
  no uplift, as in a block run at `U = 0`. Ice-free border beds stay pinned
  near base level either way: on a 200 km mode-C test run the borders ended
  within 0.6 m of before.
- **The forcing builders and the escarpment wave follow the model clock when
  `to` is not 0.** `ela_sawtooth`, `uplift_step` and `interp_forcing` built
  `linspace(0, T, nt)` while the models run `linspace(to, T, nt)`, so a forcing
  for such a run was stretched; they take `to=` now (default 0).
  The wave sat at `x_escarpment + t*v` on the absolute clock, so it had
  already moved when the run began, and the escarpment's analytical uplift
  divided `delta_h` by `T` rather than the run length; both now count from
  the run's start. Runs with `to = 0` are unchanged.
- **The 2D model checks its time axis, grid and key physical inputs at
  construction.** Several values used to construct and then run wrong without a word: `nt = 1` died in `run()` on a
  `KeyError`, `T == to` made every erosion rate NaN, `T < to` ran backwards in
  time, `coulomb_clamp >= 1` removed all ice, a negative `D` anti-diffused the
  surface by kilometres in one step, and a NaN `D` or `flotation_ramp` turned
  the bed NaN. Others failed late on a `ZeroDivisionError` (`nx = 1`,
  `lambda_p = 0`, `tau_c = 0`). Each now raises a `ValueError` naming the
  parameter and its range (listed in the parameter reference). Valid inputs
  run exactly as before. The flexure inputs are checked only with
  `flexure=True`.
- **`siim2d.load` appends `.pkl` to a dotted name, as `save` does.**
  `load('sweep_nu1.5')` opened `sweep_nu1.pkl`, another run.
- **The landscape hillshade is lit from the stated azimuth.** The map draws
  row 0 at the bottom, but the shading took its slopes in matplotlib's
  row-0-at-the-top convention, so a light at `azdeg` came from
  `180 - azdeg`: the default NW light (315) shone from the SW, and relief
  could read inverted (bumps as pits). Every hillshaded render changes.
- **Landscape movies no longer pulse in their hillshade.** The shading is
  stretched to each render's own illumination range, so a steep feature
  appearing elsewhere re-lit the whole map (-69 % contrast on an unchanged
  hill). `animate_landscape` now freezes the stretch over the movie, estimated
  on the native surface; stills keep their own, and `shade_stretch` sets it by
  hand.
- **`field='bedrock+lakes'` draws lakes on ice-free ground only.** It
  flooded the bare bed whatever the ice, so on a glaciated run nearly every
  lake pixel lay under ice (5340 of 5341 on a mode-D crop). Lakes are now
  drawn only on ground the ice footprint leaves bare.
- **`lake_min_area` counts a lake across a looped seam whole.** Labelled
  without the seam, a lake split by a looped edge counted as two smaller
  lakes and could be dropped (24 km² at `lake_min_area` = 15 km²).
- **The landscape section images sit on the nodes.** They spanned the
  domain with one pixel per node, so they lay up to half a pixel off the
  section lines (0.44 km two nodes in from the edge of a 1 km raw grid).
- **A `cross_section` outside the domain warns.** It still draws the edge
  row, and the warning names the requested y, the domain's range and the row
  drawn.
- **`map(field='ice')` saves under usetex.** Its colorbar label wrote the mean
  with a combining macron, which LaTeX cannot typeset; it is now
  `$\bar{H}$`.
- **The 1D `m.plot.analytical()` is oriented as `m.plot.profile()`**, not
  mirrored.
- **The 2D `m.plot.analytical()` draws the reference of the basin `profile`
  draws.** Every channel extraction refits the reference to its own basin, so
  after `largest_basins()` it drew the last basin's. It takes `basin_rank`
  and `ref`, as `profile` does.
- **`SteadyStateProfile.plot_profile` puts the divide where `plot` does and
  fits its elevation axis to the profile.** With base level on the right its
  glacier sat at the far end, and the fixed `2*zELA` top clipped the surface
  (1260 m under a 1200 m top at `zELA = 600`).
- **`SteadyStateProfile.plot` with base level at both ends** draws the sub-ice
  bed line under the ice (measured from the wrong end, it lay at 0.3-9.5 km
  under ice at 15.6-34.4 km) and labels the axis `x (km)`, not distance from
  the divide.
- **`m.plot.flux()` labels a fractional-km tick as it is.** Its km tick
  formatter rounded to whole km, so ticks at 500 m steps read 0, 0, 1, 2, 2,
  2, 3; it now shares the basin map's formatter (2.5 reads 2.5). Whole-km
  ticks are unchanged.

### Features

- **siim2d constructor knobs for a frozen bed, the channel-floor datum and a
  spatial ELA.** Defaults are unchanged; each knob reproduces bit for bit the
  run that set the same value on the model after construction.
  - `ce=0` and `Ko=0` are accepted (a frozen bed); the 1D analytical reference
    is then not built (`m.analytical is None`, no profile overlay).
  - `hc_over_H` (default `constants.HC_OVER_H` = 1.5), the channel-floor datum.
  - `zELA` also takes an `(ny, nx)` field and an `(nt, ny, nx)` field series
    (the xsimlab adapter takes the field, not the series).
  Details in the parameter reference.
- **Ice surface speed.** `m.ice_speed(i)` returns the ice surface speed and
  the cross-section mean speed (m/yr) of a 2D output frame, recomputed from
  the flux and thickness the closure solved, so it is exact under every
  `closure_flux` and `inner_relax` (no speed is stored) when the run keeps
  that pair. `save_closure_pair=True` (opt-in: about +10 % saved output in
  modes B/C, +19 % in D) stores the closure's own rows where they differ
  from `Qg_out` / `H_out`: `closure_Qg_out` under a transect `closure_flux`
  (the mode B/C/D default; the xsimlab adapter computes it too) and
  `closure_H_out` under `inner_relax < 1` (see the outputs table). Without
  them, and under `H_diffusivity`, the speed is approximate and `ice_speed`
  warns. Existing outputs are unchanged.
  `landscape` and `animate_landscape` take `field='bedrock+speed'`, which
  draws that speed over grey hillshaded terrain on a log colour scale
  (`speed_min`, `speed_max`, `speed_cmap`; default 20–1000 m/yr, `magma_r`).
- **Flow-routing names `'single'`, `'multiple'` and `'infinite'`.**
  `flow_routing` (siim2d) and `routing` (`siim.fastscape.glacial_processes`,
  `glacial_model`) take `'multiple'` for multiple-flow-direction and
  `'infinite'` for D-infinity, alongside `'single'`. The older codes `'mfd'`
  and `'dinf'` stay accepted as aliases, and a run reports the code
  (`m.flow_routing == 'mfd'` after `flow_routing='multiple'`), as mode C reports
  `'B'`. Results are unchanged.
- **The landscape bed colorbar marks where its colours saturate.** The bed
  colours saturate at `z_min` (default -500 m) and `z_max` by design. A
  still's bed colorbar now gets an arrow at each end the drawn surface goes
  past (`extend='min'`, `'max'` or `'both'`); `animate_landscape` decides the
  arrows once over the encoded frames, so the bar keeps one shape through the
  movie. `extend_bed` sets them by hand. The map is unchanged, and a bed
  inside the range draws the bar as before.

### Deprecations

Each warns with a `DeprecationWarning` naming its removal in 1.0.

- The 1D `m.plot.analytical()` takes the 2D signature,
  `(ax=None, bistable=True, basin_rank=0, ref=-1)`, ignoring `basin_rank` and
  `ref`. Its `plot_analytical=` keyword is deprecated (skip the call instead;
  `False` still draws nothing), and so are `alpha_ice=` and `alpha_zb=`, which
  are now ignored: the fills follow `profile()`. A call in the old positional
  order, `(ax, alpha_ice, alpha_zb, plot_analytical, bistable)`, recognised by
  a non-bool second argument or a fifth, is read that way and warns.
- `view_xsection(figsize=)` still sizes the viewer; use `fig_width` and
  `aspect`, as `profile` does.
- The `landscape` aliases `show_trimline`, `trimline_color` and `trimline_lw`
  already warned; the warning now names the removal release.
- `i_ref` in `largest_basins_history` and `sediment_history` already warned;
  the warning now names the removal release.

### Other

- `import siim.siim1d` no longer imports `matplotlib.pyplot`; the plots import
  it when they draw, as the 2D model's do.
- `landscape`'s docstring says that `channel_threshold`, `smoothed_path_color`
  and `smoothed_path_lw` act only with `show_smoothed_paths=True`.

- The escarpment, hillslope and flexure defaults (`wave_*`, `plateau_*`, `U_inf`,
  `D`, the densities, `e_thickness`, `ice_load`) are written once, in
  `siim/constants.py`, instead of as literals in up to three places; the values
  are unchanged.

- **New `mode='D'` (2D, standalone driver): mode C with a valley ↔ sheet
  reading deciding, per cell and per step, whether the valley devices or the
  sheet treatment act.** Each icy cell (`H > 10 m`) is read channel or sheet
  from the discharge per unit width `q = Q/(alpha_g*H)` of a separate routing
  pass over the plain slab `zb + H` (so the reading never sees its own datum),
  by local contrast with a ring beyond the cell's own half-width `alpha_g*H/2`:
  a cell is weak evidence of a channel where fewer than half the ring's samples
  reach a third of the largest `q` within its half-width (roughly 3 times the
  ring's median), strong where fewer than half reach a tenth of its own `q`,
  and both only in a trough: the lowest bed within its
  half-width lies below the bed around it by more than `(kB - 1)` times the
  surrounding ice thickness, with the
  new dial `channel_bed_bar = kB` (1.5 by default, 0 off). A connected region of
  weak cells reads channel when it holds at least 5 strong cells covering one
  channel width squared, `(alpha_g*H)**2`; every other icy cell reads sheet. Every length is relative to the glacier's own width, with floors in
  cells. Each cell carries a confinedness `s` (1 channel, 0 sheet) that moves a
  fifth of the way to the newly read class each step, and the five valley
  devices blend `a*s + b*(1 - s)`, exact at both ends: the channel-floor datum
  `hc_cell = 1.5*s + 1.0*(1 - s)` (built on `s` smoothed over one cell, a
  base-level border taking its donor's smoothed value), the
  below-ELA Hack footprint (down to the cell), the carve's widening factor and
  the trunk V (each scaled by its source's `s`), and the routing filter
  (thickness `H*(1 - s)`), so channel cells take the 1.5 datum, the Hack
  footprint and the carve, and sheet cells the slab datum, the cell footprint
  and the routing filter. On a 1000 km escarpment run (2.7 to 3.0 Myr, at the
  default bar) the high-flux trunks read channel 95 % of the time on average, a
  cell changes class on about 1.4 % of steps, and the deep single-cell pits
  under the ice fall from 77 to 1 against the first, rim-and-flank reading this
  entry replaces before release; the reading costs one extra routing pass a
  step (about +30 % per step on that 801 x 301 grid). In mode D
  `routing_filter_factor > 0` and `trunk_surface=True` may be combined. New
  outputs `hc_cell_out` / `glacial_spl__hc_cell` and `channel_out` /
  `glacial_spl__channel` (the class; mode D only), and
  `z_out = zb_out + hc_cell_out*H_out` there; internally, the mode-B kernels
  take an optional per-cell datum. Mode D
  defaults to the `'adaptive'` damping (see above),
  `width_hack_k = 0.25`, `trunk_surface = False`, `routing_filter_factor = 2`
  and `flow_routing = 'mfd'`, the only routing it accepts (single-flow and
  D-infinity flux trees read as false channels), and otherwise resolves as
  mode C; explicit values win. It
  needs square cells, a finite `widening_rate` and every looped axis at least
  17 cells long, and warns on cells coarser than 2.5 km; 1D and the xsimlab
  adapter refuse it. Modes A, B and C are unchanged bit-for-bit.
- **`routing_filter_factor > 0` now runs on looped grids.** Windows wrap on a
  looped axis (half-width capped at `(n-1)/2` cells there) and truncate at the
  other edges, so filtering commutes with a seam shift; the looped-grid refusal
  is removed. Non-looped grids are unchanged bit-for-bit.
- **New sheet knobs: `inner_passes` and `inner_relax` (mode B/C, standalone
  driver).** A landscape step evaluates the diagnostic ice map `F(H)` once;
  `inner_passes = n` evaluates it n times instead, moving the thickness iterate
  `x <- x + omega*(F(x) - x)` after every pass with `omega = inner_relax`, the
  first n-1 passes rehearsing at `dt = 0` (no bed motion, no thickness
  diffusion) and only the last carrying the real `dt`. `omega < 1` retains
  `1 - omega` of the previous iterate — a numerical ice memory of roughly
  `1/(omega*n)` landscape steps, at n-fold cost — which damps the ice-sheet
  margin flicker. `(1, 1.0)` is the single fresh evaluation, **bit-for-bit**
  (mode A's default; modes B, C and D now default to the adaptive damping above); uplift, the carve, the erosion rate, hillslope diffusion,
  flexure, sediment, output staging and the topography commit all stay once per
  step. Mode A and the xsimlab
  adapter (both `siim.fastscape.glacial_processes` and
  `run(driver='xsimlab')`) refuse non-defaults: the adapter routes once per step
  through the process graph.
  - Non-default settings change the model, and two consequences are inherent to
    the fused closure-and-erosion kernel. On the final pass the eroder reads the
    kernel's OWN closure `H` while the state carries the `omega`-blend, so at
    `omega < 1` erosion and the committed thickness are not the same field; and
    the committed state can hold thickness where that pass's diagnostic ice flux
    is zero (at a retreating margin, `x = 100` m with `F = 0` commits 80 m at
    `omega = 0.2` while the kernel's zero-flux branch erodes fluvially). Such a
    cell decays by `1 - omega` per pass and snaps to 0 once below
    `constants.INNER_RELAX_BARE_H` (1 m), so a cell the ice has left is bare
    again rather than holding a vanishing film. The
    mode-C carve reads the
    committed blend, not the closure `H`. With `H_diffusivity > 0`, damping
    attenuates the diffusion correction — diffusion runs on the final pass only
    and the commit blends the diffused closure with the undiffused iterate —
    and the model warns. At `inner_passes > 1` the staged routing graph and
    fluxes (`receivers_out`, `stack_out`, `Qg_out`, `area_out`, `basin_out`)
    come from the final pass, routed on the last rehearsal iterate rather than
    on `H(t-1)`: the documented one-step lag is shortened, not removed.

- **New sheet knob: `routing_filter_factor` (mode B/C, standalone driver).** The
  spatial companion of the knobs above. At `f > 0` the ROUTER GRAPH alone is
  built from a thickness-weighted TRIANGULAR (chapeau) mean of the routing
  surface `zb + hc*H` over a window of half-width `max(f*H/dx - 1/2, 0)` cells —
  about `f` ice thicknesses, set by the local thickness, with no window floor;
  dry cells and cells thinner than `dx/(2f)` keep their raw value, and windows
  truncate at the domain edge. Target, kernel and `f = 2` are the settled
  choices of the smoother study (`docs/dev/sheet_smoother_choice.md`). The
  default `0.0` is off, **bit-for-bit**.
  - It is a topology smoother, not a display filter. The filtered surface feeds
    the router and nothing else: the accumulator's mass-balance surface and the
    ice-surface hillslope diffusion keep the unfiltered surface, the transect
    and the closure the raw pass iterate, and the carve, the flexure load and
    the outputs the committed state. Smoothing the routing surface redistributes
    `Q` and hence the diagnosed `H`, so a filtered run is a different model, and
    the graph can send flux along an edge that is uphill on the unfiltered
    mass-balance surface (a deliberate modelling choice — accumulation follows
    the graph's order and weights, so nothing is lost). The unfiltered-map
    residual that quantifies the difference is measured by the private sheet
    harness; no new output is exposed and the output schema is unchanged.
  - `f > 0` requires mode B/C, the standalone driver and square cells (windows
    wrap on looped axes: see the entry below), and is refused with
    `trunk_surface=True` — the fabricated trunk surface is the same seam built
    for the opposite, valley regime. Mode A and the xsimlab adapter (both
    `siim.fastscape.glacial_processes` and `run(driver='xsimlab')`) refuse a
    non-default.

- **New router: `flow_routing='mfd'`.** A
  multiple-flow-direction producer that splits a cell's flux over every
  strictly downhill 8-neighbour of the eps-filled surface, with weights
  `(S_k/S_max)^p` and `p = mfd_exponent` (new parameter, default
  `constants.MFD_EXPONENT = 1.0`). It is built for the ice-sheet regime, where
  D8 and D-inf make the flow graph a step function of the surface. Selecting it
  also turns on the gamma effective-slope normalisation, so the thickness
  closure and all three eroders see the recovered surface-gradient magnitude
  rather than its projection on the weighted receiver direction; on a cardinal
  plane the whole mode-B step then reduces exactly to the D8 one. It raises
  unless mode B or C, a transect `closure_flux` and square cells (`dx == dy`)
  — the same guards apply to
  `siim.fastscape.glacial_processes(routing='mfd')` and the adapter's new
  `MfdFlowRouter` process; mode C is covered under Behavioural changes
  above. Under MFD the `parallel_erode` default follows the threading layer
  (see above: the topological level count roughly triples). `'single'` runs are untouched
  (bit-for-bit); for `'dinf'` see the behavioural change above.
- Diagnostics built on a single receiver per cell — `extract_channel`,
  `channel_statistics` (Strahler order, Horton ratios), basin plots and trunk
  ribbons — are documented as PRIMARY-PATH views under multiple-flow routing.

## 0.9.10 — September 2026

### Behavioural changes

- **`glacial_model(mode='C')` is now native mode C.** The fastscape facade
  (`glacial_processes` / `glacial_model`) resolves the mode-C standard the way
  `siim2d` does: `trunk_surface` defaults to `None`, which turns trunk-surface
  routing on for mode C (`mode='C'`, or `mode='B'` with `carve=True`) and leaves
  it off otherwise, and `TrunkSurfaceToErode`'s `routing_relax` input now
  defaults to 0.5 (`constants.MODE_C_ROUTING_RELAX`). With no extra inputs,
  facade mode C matches `siim2d` mode C bit for bit. Before, it gave carving
  without trunk-surface routing and with `routing_relax=0`;
  `trunk_surface=False` restores that composition. Without a `routing_relax`
  input, two explicit combinations differ from `siim2d`: facade mode B with
  `trunk_surface=True` gets 0.5 (`siim2d`: 0), and facade mode C with
  `trunk_surface=False` gets 0 (`siim2d`: 0.5). `siim2d` runs, on either driver,
  are unchanged.

### Features

- Averaged landscape cross-sections now shade the ice thickness's interquartile
  range, hung on the mean bed, instead of the ice surface's (whose spread mostly
  repeated the bed's), and name it and the dashed bed quartiles in a legend, in
  still plots and animations.
- `siim.fastscape` now exports `D8FlowRouter`, `GlacialBlockUplift`,
  `HillslopeDiffusion` and `InitialTopography`, the rest of the processes
  `glacial_processes` wires in, so a fastscape model can be written out by hand
  from the public surface alone. They were already importable; this makes them
  part of the advertised API.

### Fixes

- **`SteadyStateProfile.plot` x-axis sense.** With base level on the left
  (the default) the canonical single-axis renderer plotted the profile against
  distance from *base level* under a "Distance from divide" label — the divide
  sat at `L`, base level at 0, and the sub-ice bed line was drawn away from the
  ice fill. The divide now sits at 0 and base level at `L` whichever end holds
  base level (the 2D model's reference has it on the right, which was already
  drawn correctly), matching `GeneralProfile.plot` and the model plotters.

## 0.9.9 — September 2026

### Documentation

- **Choosing glacial parameters.** A new guide, `guides/choosing_glacial_parameters`,
  says what `beta`, `ce` and `nu`, `lambda_p` and `lambda_c` set physically and
  how they trade off: β against a temperature lapse rate through a degree-day
  identity, with modern and LGM Alpine worked examples and the plausible
  10⁻³–10⁻² range across climates; the collapse of (`ce`, ℓ, U) onto the
  balancing sliding speed u_b\* and where the default sits against the
  Herman et al. (2021) erosion-rule compilation; and the measured relief and
  sliding-fraction scalings with `lambda_p` and `lambda_c`. Linked from the
  parameter reference, the configuring guide and the index; two bibliography
  entries added.

### Tests

- A `pin` test that the analytical steady state depends on (`ce`, `ell`, U)
  only through u_b\* = (U/ce)^(1/ell).

No behavioural change.

## 0.9.8 — September 2026

### Features

- **Tidier landscape layout.** Slender, map-aligned colorbars and a narrower
  hypsometry panel keep the terrain prominent; consistent panel spacing and a
  capped section height suit both square maps and wide domains.

- **Animation frame rate and frame range.** `animate_landscape` takes `fps`
  (an explicit rate, overriding `interval` on both the serial and the parallel
  render path) and `frames` — a slice, a range or a sequence of saved-frame
  indices, negatives counting from the end — so a movie can cover part of a run,
  or be re-timed, without re-running it. A windowed landscape movie freezes its
  colour scales over the selected frames alone, and an empty selection raises
  instead of writing an unplayable file. `animate_map` and `animate_profile`
  gain the same `frames` selection; they already took `fps`.

- **Averaged landscape profile.** `landscape(cross_section=...)` accepts
  `'mean'` beside a `y` in km: the section then averages over the rendered rows
  of the section grid, so its ice band is the average ice column per unit x and
  vanishes where nothing is iced, and it shows the spread behind that mean: the
  ice surface's 25-75% band across those rows shaded, the bed's quartiles dashed.
  An averaged profile draws no map locator line and skips the lake layer;
  `animate_landscape` inherits it like any other `landscape` keyword. Any other
  string is rejected.

- **Sediment reported by domain edge.** `track_sediment` widens from a flag to
  `False | True | 'basin' | 'edge' | 'both'`. `'edge'` adds
  `sediment_edge_flux_out` and `sediment_edge_cumulative_out`, shaped
  `(time, side)` over `['left', 'right', 'bottom', 'top']`: the routed sediment
  summed over each `'fixed_value'` outlet ring, corners counted once, `NaN` on
  an edge with no outlet — the volume actually leaving the domain there, rather
  than the single-outlet sample `sediment_history` takes. `True` still means
  `'basin'` and its rasters are unchanged.

- **Time-varying precipitation, documented.** `P` has accepted a length-`nt`
  series on the run clock in both models; the configuring guide now
  shows it beside the ELA and uplift series (`interp_forcing` ramp), and the
  example notebook carries a precipitation cell that reproduces the scalar run
  until its endpoints are changed. No package code changed.

## 0.9.7 — September 2026

### Behavioural changes

- **Flexure now solves in a sine basis** (`w = 0` at the padded-box edges,
  the basis fastscape's `flexure2D.f90` uses) instead of a periodic FFT with the
  domain-mean mode zeroed. Every `flexure=True` run changes. On small domains
  (`L` much less than the flexural parameter, siim's valley case) the difference
  is ~1e-2 m: a uniform load stays rigidity-supported as before. On domains much
  larger than the flexural parameter the old solve subtracted the Airy mean of a
  localized load and re-emitted it as a uniform uplift of the entire domain —
  on a 2500 x 250 km grid with `e_thickness=20e3` and a migrating uplift wave
  that was +3.5 m per 100 kyr step, raising an untouched far plateau from 290 m
  to 1460 m over 50 Myr. Localized loads now compensate locally at Airy and the
  far field stays put. The transform is scipy's type-2 sine transform with the
  per-run transfer grid cached: same fidelity as type 1 and as the fortran
  (measured), 2-3.5x faster per solve and faster than the fortran itself; it
  moves flexure numbers at the 1e-3 relative level, so the flexure-on reference
  battery regenerates once on Linux CI after this release.

- **`fixed_value` borders no longer receive flexural rebound.** Block uplift was
  already masked off on those rows and columns and they do not erode, but the
  plate solve still handed them their share of the subsidence, so a fixed edge
  beside an uplifting interior sank into a trench that then acted as the base
  level for everything draining to it (on a 2500 x 250 km run the fixed row fell
  287 m to 55 m while the interior rose 292 m to 543 m over 10 Myr). Stock
  fastscape has the same inconsistency between `BlockUplift` and `Flexure`.
  Affects `flexure=True` runs with at least one `fixed_value` edge, including
  the default topology's two fixed columns; all-looped and all-core domains are
  unchanged.

- Landscape movies render serially when system RAM cannot be measured, keeping
  the memory guard effective on platforms without usable `sysconf` results.
  Interactive viewers now report unexpected canvas-setting errors instead of
  silently continuing; failed figures are closed and the notebook's plotting
  backend is restored.

- The 1D `plot.limit_cycle` and `plot.limit_cycle_phase` research helpers are
  now private (`_limit_cycle`, `_limit_cycle_phase`); their drawing behavior is
  retained. Public basin and steady-state summaries support `plot=False` and
  are quiet by default; use `verbose=True` for printed Hack's-law, basin-history
  or steady-state summaries. Basin histories use `ref` consistently with
  profiles; `i_ref` remains a deprecated keyword alias. See the
  [plotting guide](docs/guides/outputs_and_io.md#basin-and-steady-state-summaries).

### Other

- Plotting review fixes: static profiles preserve existing artists on supplied
  axes, and 1D profiles/viewers/movies retain their original coordinate
  direction. Landscape component cleanup now runs once on the combined ice
  layers; ribbon-source thresholds preserve the veil's field-smoothing input.

## 0.9.6 — September 2026

- **2D base-level warning moved to its point of use.** A nonzero or per-side
  `bl` no longer warns at model construction; the note that the analytical
  steady-state reference stays at the datum is emitted when the analytical
  overlay is actually drawn (`profile`/`analytical` plots). Batch runs that
  never look at the overlay are quiet. 1D is unchanged.
- `siim2d.load` accepts a name with a subfolder (`'batch/run003'`), matching
  what `save` writes.

## 0.9.5 — September 2026

- **Documentation overhaul.** The public docs were rewritten as plain
  reference documentation (independently reviewed, and verified line-by-line
  against the code): interpretive discussion removed, the `first_1d_run`
  example re-parameterized (`zELA` 700 → 1400) so the profile stays above the
  base-level datum with the glacier confined to the upper reach, the
  coarse-grid mode-C/mode-B comparison stated precisely (the mode-C routing
  helpers never self-gate on grid resolution), `trunk_alpha` documented as an
  opacity floor, the fastscape `carve` sentinel documented, and the
  `siim._core` API index completed.
- **Fixed the adapter's install hint.** The `ImportError` raised by
  `import siim.fastscape` without the fastscape stack still recommended
  `pip install siim[fastscape]`; it now names the actual distribution,
  `pip install siim-lem[fastscape]`. No behavior changes.

## 0.9.4 — September 2026

- **PyPI distribution name: `siim-lem`.** PyPI's name-similarity guard refused
  the bare name `siim`, so the package installs as `pip install siim-lem`
  (the `[fastscape]` extra rides along unchanged). The import name is
  untouched: `import siim`. Nothing else changes from 0.9.3.

## 0.9.3 — September 2026

- **2D ice rendering: ice is on by default, and depth-graded.** `landscape` /
  `animate_landscape` under `style='smooth'` now resolve `field` to
  `'bedrock+ice'` (it resolved to bare `'bedrock'`, so the default render hid
  the ice and every ice keyword was a silent no-op; `field='bedrock'` is
  unchanged and still reachable). Ice without an explicit `ice_cmap` is painted
  by the new `ice_shading` knob: `'veil'` (smooth) alpha-blends a glacier ramp
  over the terrain by column depth, normalised on the RUN-GLOBAL
  `1.5 * max H_out` so movie frames are comparable, and draws its own
  colorbar; `'flat'` (raw) keeps the single opaque `ice_color`, so `style='raw'`
  is unchanged. Labels now distinguish the two thickness quantities: the
  landscape bar and section band show the local **ice column depth**, while
  `map(field='ice')` and the 1D/2D profile panels show the width-**mean**
  `H`. `show_trimline` / `trimline_color` / `trimline_lw` are renamed
  `show_margin` / `margin_color` / `margin_lw` (the outline is the current
  margin, not a trimline) with deprecated aliases, and the margin defaults are
  heavier and darker. `map(field='ice')` masks ice-free ground to a neutral
  bare tone, sizes its figure from the domain aspect and stamps the output
  time; the section names its ELA and states the vertical exaggeration; and the
  two-colorbar/hypsometry layout survives narrow `fig_width`. The smooth
  preset's `H_threshold` default drops from `100` to `0`, so **nothing is
  hidden by thickness under either style** — the veil already fades thin ice
  out, and the gate only deleted real glacierets. `H_threshold` remains fully
  functional as an explicit crop. On top of that veil the smooth preset now
  draws the RESOLVED TRUNK GLACIERS as true-width ribbons (`trunk_display`,
  `'ribbons'` under smooth and `'none'` under raw): a trunk cell is any icy
  cell downstream of one whose claimed width `W = alpha_g*H` already spans
  `trunk_width_cells` grid cells (default `1.0`), the class carried along the
  receivers to the terminus, and those cells are traced and rasterized at
  their true width — parabolic column depth on the same ramp, their own flat
  ice surface in the hillshade and the cross-section — at `trunk_alpha`
  opacity, seam-aware on looped axes. The veil is then built from the
  sub-resolution ice that is left, and `show_margin` outlines the ribbons
  only. The veil's opacity ramp changes from `0.18 + 0.82*sqrt(t/0.40)` to the
  linear `0.12 + 0.88*t/0.35`: the old front-loading made a 30 m column
  half-opaque, so an apron of sub-cell ice read as one sheet with the trunks
  barely darker streaks through it (trunk-vs-apron RGB distance 0.13, against
  0.67 for apron-vs-bare). `trunk_display='none'` renders exactly as before.

## 0.9.2 — September 2026

- **Mode B/C: the kernel now erodes the post-uplift bed.** The in-house driver
  and the Fastscape adapter both handed the mode-B/C kernel the pre-uplift bed
  and composed uplift afterwards. `fixed_value` border cells never uplift, so
  the first interior row equilibrated against the pinned outlet and was then
  lifted by `U*dt` every step — a permanent `U*dt` lip along every base-level
  border that raised the whole landscape by that amount (300 m at
  `dt = 300 kyr`). Mode A, the 1D model, and the kernel's documented contract
  already used the post-uplift bed. Mode-B/C results change by `U*dt` near
  borders (A is bit-identical); driver/adapter parity stays bit-for-bit, and
  the B/C reference battery was re-frozen on the CI capture environment.
- **Nine ice-display fixes from a mode-C visualization audit.** Lakes flood
  the true composite surface, so the cross-section matches the map (the
  `lakes` field was previously unreachable); `ice_smoothing='field'` with
  `H_threshold <= 0` raises instead of silently drawing nothing;
  `animate_landscape` freezes `z_max`/`H_max` over the run and one NaN no
  longer blanks the map; `ice_time_avg` feeds only the ice mask and depth
  colour in both extents (the footprint extent no longer time-averages the
  terrain/section); auto `z_max` covers the unsmoothed section profile;
  `_clean_ice_mask` is seam-aware on looped axes; the raster extent registers
  with the contour/trimline/section coordinates; the section bed line is
  bilinear to match the ice surface; bare-bed views are labelled "Bedrock
  elevation". Each fix carries a regression test.
- **Plateau initial surface: edges on their datums + a public builder.** The
  arctan plateau (`siim_escarpment(init_type='plateau')`, the fastscape
  `PlateauSurface`) now rescales its ramp so the fixed x-borders start exactly
  on `0` / `plateau_zo - plateau_dz`; the raw arctan only reached them
  asymptotically (with the default `plateau_frac`/`plateau_w` on a 50 km domain
  the low edge sat at `0.25*plateau_zo`, a permanent sill above the border's
  water datum). Behavioral for every plateau run (no reference-battery case
  uses one). New `siim.escarpment.plateau_topography(nx, ny, Lx, zo, frac, w,
  dz)` returns the same surface for plain `siim2d` runs together with the
  per-side base level `{'right': zo - dz}` that puts the plateau's outlet at
  its own edge elevation.
- **Per-side base level (2D).** `bl` now also accepts a dict keyed by side
  (`{'left'|'right'|'bottom'|'top': scalar or length-nt series}`), giving each
  `'fixed_value'` outlet its own water datum; unspecified sides keep
  `constants.BL`, and a datum on a non-outlet side or an unknown key raises.
  Border nodes carry their own side's datum and every interior node inherits
  the datum of the outlet its basin drains to (corners shared by two fixed
  sides take the x-side value). Scalar and series `bl` are unchanged
  bit-for-bit; per-side `bl` requires the in-house driver (the xsimlab adapter
  takes one scalar per step and raises on a dict), and — like scalar `bl` in 2D
  — is consumed by modes B/C only (2D mode A ignores `bl`). 1D rejects a dict.
  See `docs/guides/configuring_a_run.md`.

## 0.9.1 — August 2026

The standalone migration: siim's 2D model no longer requires the frozen
fastscape/xsimlab/fastscapelib-fortran stack.

**Packaging (the headline)**
- The 2D model (`siim.siim2d`, `siim.escarpment`) is now **pip-installable
  standalone**: `pip install siim` runs everything on numpy/scipy/numba/
  matplotlib/tqdm/xarray/pandas (numpy 2 supported; `xarray` and `pandas` are
  direct, unpinned core dependencies). No conda required for default use.
- `siim.fastscape` is demoted to the **optional adapter**
  (`pip install siim[fastscape]`); importing it without Fastscape raises a
  directed error, while missing conda-only runtime backends are reported when
  the corresponding stock process is used. fastscapelib-fortran remains conda-only —
  `environment.yml` is retitled the fastscape-adapter / legacy env and keeps
  its numpy<2 / xarray<2026.5 / zarr<3 quarantine (which no longer constrains
  a default install). A `dev-parity` extra ships for the future fastscapelib
  parity probe (workflow_dispatch-only CI stub pre-1.0). The published
  `environment.yml` is a lean `siim-adapter` environment rather than the full
  paper/notebook development environment.
- Importing the optional adapter no longer monkey-patches private xarray
  internals or installs a process-wide warning filter. SIIM-owned xsimlab
  compatibility warnings are suppressed only around the calls that emit them.
- The release exporter now requires a committed, clean source snapshot and a
  synchronized public target. Its dry run performs the same artifact build,
  archive inspection, clean-environment install, smoke tests, and public-tree
  diff as publication, without changing either working tree.

**Numerics owned in-house (defaults flipped at 0.9.1)**
- Time loop: an in-house framework-free driver (`siim._core.driver`) replaces
  xsimlab orchestration (bit-for-bit identical `ds_out` on the same backend;
  `driver='xsimlab'` remains as the adapter-env escape hatch).
- Flow routing: in-house numba D8 fill-then-route (+ the already-in-house
  D-inf) replaces the fortran SFR. ⚠ **Behavioral change:** routing
  tie-break/depression paths differ from fortran by construction, and siim's
  2D landscapes are multistable under them — same-seed mode-B/C runs
  self-organise equivalent-but-different drainage networks (~100–170 m rms
  apart at identical attractor statistics). Gated behaviorally
  (receiver parity vs frozen raw fortran refs, analytical oracle, and attractor
  statistics). Compare attractors, not
  snapshots, against pre-0.9.1 runs.
- Flexure: in-house scipy.fft plate solve. ⚠ Fixes the fortran `pihy`
  anisotropy bug — on `dx != dy` grids results differ from fortran **by
  design** (validated against the closed-form Kelvin point-load solution);
  the k=0 (domain-mean) mode is zeroed, matching fortran's far-field-neutral
  DST semantics.
- Hillslope diffusion: in-house numba ADI, byte-identical to `fs.diffusion`
  for siim's uniform diffusivity. Its Thomas solver is independently expressed
  from the standard tridiagonal row recurrence.
- The retired `'fortran'` options of `router_backend`/`numerics_backend` now
  raise directed errors; the params survive as the (public-contract) backend
  plug points.

**Bug fix — mode B/C outlet lip (2026-09-02)**
- The 2D mode-B/C kernel now receives the **post-uplift** bed (the kernel's
  documented contract; mode A and the 1D model already did). Both the in-house
  driver and the Fastscape adapter eroded the pre-uplift bed against the pinned
  `fixed_value` borders and composed uplift afterwards, leaving a permanent
  `U*dt` step on the first interior row at every base-level border and raising
  the whole landscape by that amount (300 m at dt = 300 kyr). ⚠ Mode-B/C
  results change by `U*dt` near borders (and negligibly elsewhere via the
  absolute-datum terms); the frozen B/C reference battery needs a sanctioned
  regeneration. Driver/adapter parity remains bit-for-bit.

**API changes (pre-release, no deprecation cycle)**
- Default `boundary_status` is now the explicit
  `['fixed_value','fixed_value','looped','looped']`, and `'core'` means plain
  non-periodic interior for **every** router (previously fortran-SFR silently
  treated top/bottom `'core'` as y-cyclic while D-inf did not — the
  split-brain is gone; ratified OQ-1(b)).
- `run(hooks=...)` is dropped on the standalone driver (raises a directed
  error); xsimlab RuntimeHooks remain available via `driver='xsimlab'`.
- `nt_out` is validated to `1 <= nt_out <= nt` (previously `nt_out > nt`
  silently produced NaN frames under xsimlab).
- An explicit analytical `lam` override now consistently sets
  `kappa_c = 1/(1-lam)` in both `GeneralProfile` and `RegimeMap`; the default
  exponent-derived closure is unchanged.
- An explicit `mu` with an exact power or Coulomb numerical law is retained for
  the analytical/reporting interpretation but now warns that the exact kernel
  uses its law-derived exponent. The effective-exponent law continues to use
  the override numerically.
- Saved-run pickles now carry an explicit format/schema version, producing SIIM
  version, concrete model identity, the original parameter dictionary, and the
  xarray Dataset. Unversioned saves are rejected with a directed error rather
  than silently interpreted under current defaults.
- The standalone and adapter drivers now consume one shared output schema, and
  construction of the numerical law record is keyword-only to make its
  cross-module parameter contract resistant to accidental reordering.

**Verification (per stage, recorded in the plan)**
- Extraction + driver: bit-for-bit against the frozen reference battery and
  xsimlab at every output frame. Flexure: Kelvin oracle relRMS ~2e-5 (square)
  / 1.5e-4 (anisotropic). Diffusion: byte-identical twin. Router: 961/961
  byte-exact receivers on tie-free surfaces, exact fixed-point
  router-invariance, attractor means within 3.1%. The reference battery is
  re-frozen from the standalone defaults and remains the standing
  bit-for-bit determinism tripwire (per-platform); the raw fortran router
  refs remain immutable parity baselines.

## 0.9.0 — July 2026

The pre-release consolidation: everything below is on `main`, suite 376,
docs `-W` green, dev-CI green. v1.0 is reserved for the public release.

**Physics & boundary conditions**
- Base-level BC redesign (mode B): water/rock separation with a time-dependent
  Dirichlet water datum `bl(t)`; true-state outputs everywhere (see
  `docs/guides/outputs_and_io.md`).
- Ice borders are OUTFLOW boundaries: zero-gradient thickness (dominant donor
  in 2D) with the bed evolved by the implicit closed-form border budget on the
  arrival slope — dt-robust at any dt, bounded at the flotation draft.
- Waterline-flotation gate as an effective-pressure ramp
  (`flotation_gate`/`flotation_ramp`, γ = 0.1 default), interior + border.
- Channel-floor datum `hc/H̄ = 1.5` implemented across the numerical models and
  the analytical bed reconstructions (see `docs/guides/concepts.md`).

**Mode C (the flagship default: mode B + sub-grid width carving)**
- Sub-grid glacier-width carving via exact power-diagram attribution
  (see `docs/guides/configuring_a_run.md`); `widening_rate` default 3.0.
- Mode-C standard: `trunk_surface=True`, `routing_relax=0.5` by default
  (anti-flicker EMA); the flux-consolidation
  machinery removed.
- D-inf routing rebuilt on the eps-filled surface (fill-based redesign) plus
  D-inf mode B.

**Performance**
- `parallel_erode` (default ON): level-scheduled parallel mode-B erosion,
  bit-for-bit with the serial eroder — coulomb steps −43% (SFR) / −32% (D-inf)
  at 201×201.
- D-inf facet scan row-parallelized; allocation-free multi-receiver Newton.
  D-inf mode-C steps −19% (power) / −10% (coulomb) on top of the above.

**Infrastructure**
- Pre-1.0 code audit complete (91 findings adjudicated).
- Sphinx docs site (Furo + MyST-NB), `-W` clean; RTD config at the root.
- Frugal dev-CI (cached micromamba: suite + docs build per PR/push).
- Packaging: conda-only for the 2D stack (`environment.yml` is the contract,
  with the numpy<2 / zarr<3 / xarray-freeze quarantine around the unmaintained
  xsimlab); the lightweight core (1D + analytical) declares pip deps.

## 0.5.0 and earlier — 2026 (unreleased development)

Internal development: the 1D profile model, the 2D fastscape/xsimlab model,
the analytical package (steady-state profiles, closure solver, regime map),
the numpy/numba numerical core, and the law_code kernel rewrite.
