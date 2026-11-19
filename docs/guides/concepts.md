# Concepts

How siim represents a glaciated landscape — background for the parameters in
{doc}`configuring_a_run` and the arrays in {doc}`outputs_and_io`.

## What the state is: modes A, B, C and D

siim can track the landscape two ways, plus two carved variants:

- **Mode A** (`'ice_surface'`) tracks a *single* field, the ice surface `z`.
  The ice thickness is re-solved from the local surface slope each step, so a
  carved trough heals the instant the ice thins — no memory. The simplest,
  fastest regime; good for steady long profiles.
- **Mode B** (`'bedrock+ice_thickness'`, the 1D default) tracks *two* fields, the
  bedrock `zb` and the ice thickness `H`, and rebuilds the surface each step.
  Carved overdeepenings persist on ice retreat: **bed memory**. In 2D, an
  unqualified `mode='B'` is plain bed memory with **no** width carving —
  carving is opt-in.
- **Mode C** (2D only) is mode B plus sub-grid **width carving** — the carved
  configuration and the **2D default** (bare `siim2d` runs, or `mode='C'`;
  `mode='B', carve_width=True` is the same thing). See the mode-C standard in
  {doc}`configuring_a_run`.
- **Mode D** (2D only) is mode C with a valley ↔ sheet reading that gives
  each cell, step by step, the valley or the ice-sheet form of the mode-C
  devices, so valley glaciers and ice sheets share one domain (see the datum
  below and {doc}`configuring_a_run`).

Modes A and B agree at steady state for the eff-exp and power laws (to ~1e-9);
the Coulomb law is benchmarked through mode B.

With bed memory (modes B, C and D), ice extent and erosion can vary in time under
constant forcing (autogenic cycling); two such runs are compared through
long-run statistics rather than individual output frames.

(toe-slope-1d)=
## The terminus in 1D: the toe slope correction

The 1D model builds each cell's rise from the glacial slope at its upstream
node. Near a terminus that slope is singular: the ice flux falls linearly to
zero there, and the steady surface steepens as `S ~ Q**-phi_t`, with
`phi_t` the sliding law's thin-ice limit (1/3 power, 1/2 Coulomb, `mu/nu`
effective exponent). An uncorrected upwind step across that singularity holds
the terminus at an arbitrary place in its cell, which costs the divide and
terminus their convergence in `dx` and drives a spurious swing of the mode-A
divide: 19 to 26 m on 50 and 25 m cells for the power and effective-exponent
laws (9 m for Coulomb on 25 m cells), and 32 to 35 m for Coulomb with the
accumulation cap on, on every grid from 200 to 25 m. siim1d therefore scales the
upwind slope of every icy node whose downstream neighbour carries less flux by
a factor of the flux ratio, exact for that surface, and gives the last icy
node the slope from the sub-grid terminus where the glacier draining to it
spans at least `constants.TOE_MIN_CELLS` (5) icy nodes (on a shorter one the
grid does not resolve the toe and that term over-corrects, so the last node
keeps the uncorrected step), in mode A's slope, the erosion
kernels and mode B's walk alike (`toe_slope_correction`, on by default; the
2D model has an experimental opt-in port, `toe_slope_correction_2d`, off by
default). With it the 1D divide and terminus converge at
first order in `dx`, and modes A and B settle on the same state. Coulomb with
the accumulation cap on is the exception that keeps a slow mode-A swing on
fine cells (see the changelog).

## The channel-floor datum: where the bed sits

The tracked bed `zb` is not the ice surface — it is the **sub-grid channel floor**
(the thalweg), sitting below it. Every surface siim builds from state is

    zs = zb + 1.5 · H          (1.5 = HC_OVER_H)

The factor 1.5 is the max/mean depth ratio of a parabolic channel cross-section:
`H` is the width-**mean** ice depth, and the centerline runs `1.5·H` deeper.
**`H` is the only quantity the physics consumes** — the flux closures, the
sliding laws, the erosion laws, and the driving stress `τ = ρg·H·S` all read the
mean depth `H`, never the `1.5`. Erosion is incision of the floor `zb`. (The
`hc_over_H` parameter of both models sets the ratio; `hc_over_H=1` recovers
the older mean-bed datum.)

The `1.5` describes a valley glacier, whose flanks are empty of ice. In mode D
each cell also carries a confinedness `s`, read from whether its ice discharge
stands markedly above that of the ice around it, in a trough: `s = 1` is a
valley channel, `s = 0` part of an ice sheet. The datum
becomes per cell, `hc_cell = 1.5·s + 1.0·(1 − s)`, so a sheet's surface is
`zb + H`, and the other valley devices (the ablation footprint, the carve, the
trunk surface, the routing filter) fade the same way; see
{doc}`configuring_a_run`.

## Sub-grid width: one glacier, many cells

A glacier of mean thickness `H` fills a valley of width

    W = α_g · H                (α_g = alpha_g, default 5)

which is many grid cells wide. That single width closure is used in three
places:

1. **The flux closure** uses the mass-conserving cross-section `α_g·H²/…` — the
   discharge a trunk carries.
2. **The carve** (mode C) erodes the whole footprint, a disc of radius
   `R = α_g·H/2` around each glaciated cell (power-diagram attribution). The
   footprint erodes terrain *above* the ice surface as well, so a widening
   glacier captures neighbouring ridges and channels and produces troughs of
   finite width.
3. **The display** draws ice across the same `W` under
   `ice_extent='footprint'` (see {doc}`outputs_and_io`).

On a grid too coarse to resolve `R = α_g·H/2` (a cell or less), carving is a
no-op and the bed evolves as in plain mode B; the mode-C routing helper stays
on (`trunk_surface`, wherever a source's `R` exceeds a cell), so the run is not
identical to `mode='B'`.
