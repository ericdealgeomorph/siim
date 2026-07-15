# Concepts

A short orientation to how siim represents a glaciated landscape — enough to read
the parameters in {doc}`configuring_a_run` and the arrays in
{doc}`outputs_and_io` with the right mental model.

## What the state is: mode A, B, and C

siim can track the landscape two ways, plus a carved variant:

- **Mode A** (`'ice_surface'`) tracks a *single* field, the ice surface `z`.
  The ice thickness is re-solved from the local surface slope each step, so a
  carved trough heals the instant the ice thins — no memory. The simplest,
  fastest regime; good for steady long profiles.
- **Mode B** (`'bedrock+ice_thickness'`, the 1D default) tracks *two* fields, the
  bedrock `zb` and the ice thickness `H`, and rebuilds the surface each step.
  Carved overdeepenings persist on ice retreat: **bed memory**. This is siim's
  native regime, where the autogenic cycling lives. In 2D, an unqualified
  `mode='B'` is plain bed memory with **no** width carving — carving is opt-in.
- **Mode C** (2D only) is mode B plus sub-grid **width carving** — the flagship
  carved configuration and the **2D default** (bare `siim2d` runs, or `mode='C'`;
  `mode='B', carve_width=True` is the same thing). See the mode-C standard in
  {doc}`configuring_a_run`.

Modes A and B agree at steady state for the eff-exp and power laws (to ~1e-9);
the Coulomb law is benchmarked through mode B.

## The channel-floor datum: where the bed sits

The tracked bed `zb` is not the ice surface — it is the **sub-grid channel floor**
(the thalweg), sitting below it. Every surface siim builds from state is

    zs = zb + 1.5 · H          (1.5 = HC_OVER_H)

The factor 1.5 is the max/mean depth ratio of a parabolic channel cross-section:
`H` is the width-**mean** ice depth, and the centerline runs `1.5·H` deeper. This
matters because **`H` is the only thing the physics consumes** — the flux
closures, the sliding laws, the erosion laws, and the driving stress `τ = ρg·H·S`
all read the mean depth `H`, never the `1.5`. Erosion is incision of the floor
`zb`. (Setting `HC_OVER_H = 1` recovers the older mean-bed datum.)

## Sub-grid width: one glacier, many cells

A glacier of mean thickness `H` fills a valley of width

    W = α_g · H                (α_g = alpha_g, default 5)

which is many grid cells wide. That single width closure shows up in three
places, and keeping them consistent is what makes the carved model honest:

1. **The flux closure** uses the mass-conserving cross-section `α_g·H²/…` — the
   discharge a trunk actually carries.
2. **The carve** (mode C) erodes the whole footprint, a disc of radius
   `R = α_g·H/2` around each glaciated cell (power-diagram attribution). Because
   the footprint eats terrain *above* the ice surface too, a widening glacier
   captures neighbouring ridges and channels — real trough width, not a
   one-cell slot.
3. **The display** draws ice across the same `W` when you ask for
   `ice_extent='footprint'` (see {doc}`outputs_and_io`).

On a grid too coarse to resolve `R = α_g·H/2` (a cell or less), carving is a
no-op and mode C looks like plain mode B.

## Autogenic cycling: compare attractors, not snapshots

With bed memory (mode B/C), the coupled glacial–fluvial system can self-organize
into MISI-style **limit cycles** — advance, carve, retreat, refill — with the
climate forcing held fixed. A practical consequence for anyone comparing runs:
two configurations that share a *steady state* need not share any single frame
once they are cycling. Compare **long-run statistics (the attractor)**, not
instantaneous snapshots, whenever the regime is autogenic.
