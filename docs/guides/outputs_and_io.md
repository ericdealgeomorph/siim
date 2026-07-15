# Outputs, plotting, and saved runs

## Plotting

Every model carries a plotter at `m.plot`:

- **1D** — `m.plot.profile()` draws the long profile (bed, ice surface, and the
  analytical reference). Returns `(fig, axes)`.
- **2D** — `m.plot.map(field='bedrock')` renders a raster field (returns the
  `Axes`); `m.plot.landscape()` gives an atlas-style shaded-relief render
  (returns `(fig, ax)`).

All three take a time index `i` (default `-1`, the last step). Colour/axis limits
are per-method: `map` and `profile` use `field_min` / `field_max`; `landscape`
uses `z_min` / `z_max` (terrain) and `H_min` / `H_max` (ice). Note the return
shapes differ — `map` hands back a single `Axes`, while `profile` and `landscape`
return `(fig, axes)` — so unpack accordingly. See {doc}`../api/model2d` for the
full plotter API.

## Displaying ice honestly

`landscape(field='bedrock+ice')` draws ice two ways:

- `ice_extent='footprint'` (default) is the **width-honest** view: ice is drawn
  across the model's claimed valley width `W = α_g·H` — the display dual of the
  sub-grid width carve, so a glacier looks as wide as the physics treats it.
- `ice_extent='cells'` is the **raw state** view: only the glaciated channel
  cells themselves.

Nothing is hidden or smoothed by default. For smooth *animations* the validated
recipe (a recommendation, not a default) is:

```python
m.plot.animate_landscape(field='bedrock+ice',
                         H_threshold=50, ice_sigma_cells=3, ice_time_avg=2)
```

`H_threshold` drops sub-threshold specks, `ice_sigma_cells` smooths the ice
outline, and `ice_time_avg=2` trailing-averages the ice layer to damp the
discrete-D8 per-step flicker (terrain stays on the exact frame). You *can* add
`min_ice_cells=6` to remove tiny components, but note it **hides real small
glacierets** — leave it off unless the specks are pure flicker.

## Output fields you can read off `m`

After `m.run()` (or `load`), the run's arrays hang off the model, all
`(time, y, x)` unless noted. Alongside each is its on-disk zarr variable name.

| attribute | zarr variable | meaning |
|-----------|---------------|---------|
| `m.z_out` | `topography__elevation` (mode A) | ice-surface elevation `zs` |
| `m.zb_out` | `topography__elevation` (mode B/C) | tracked bed = channel floor |
| `m.H_out` | `glacial_spl__ice_thickness` | width-mean ice thickness |
| `m.area_out` | `glacial_flow__area` | upstream drainage area |
| `m.Qg_out` | `glacial_flow__ice_flux` | ice flux |
| `m.Qf_out` | `glacial_flow__water_flux` | water flux |
| `m.erosion_rate_out` | `glacial_spl__erosion_rate` | erosion rate |
| `m.denudation_out` | `glacial_spl__denudation` | per-step rock removed (Δ`zb` in B incl. carve, Δ`zs` in A) — the sediment / flexural-unloading source |
| `m.receivers_out`, `m.stack_out`, `m.basin_out` | `glacial_flow__{receivers_2d,stack_2d,basin_ids}` | flow graph |
| `m.rebound_out` | `flexure__rebound` | flexural deflection (`flexure=True`) |
| `m.sediment_flux_out`, `m.eroded_volume_out` | `sediment__{flux,cumulative}` | sediment throughput + running total (`track_sediment=True`) |

`m` exposes **both** `z_out` (ice surface) and `zb_out` (bed) directly, so you
rarely reconstruct by hand. If you read the raw zarr instead, the mapping is
mode-aware: in mode B/C `topography` **is** the bed, so `z = zb + 1.5·H`
(`1.5 = HC_OVER_H`, the channel-floor datum); in mode A `topography` is the
surface and the bed is stored separately as `glacial_spl__bedrock_surface`.

## The true-state output convention

Outputs report the **true `(zb, H)` everywhere** — there is no presentation
floor at base level. With a nonzero `bl` (see {doc}`configuring_a_run`), a drowned
or relict bed shows through *below* the waterline: that is bed memory, not a bug.
Water (ocean and lakes) is a display layer only — `landscape` renders it at
`max(zs, bl)` and fills lakes for the picture, but the stored state keeps its true
drowned elevation.

## Output files

Figures and saved runs are written under `model_outputs/` in the current
working directory (not inside the installed package).

## Saving and reloading a run (2D)

```python
m.save('run_name')                 # writes under model_outputs/

from siim.siim2d import load
m = load('run_name')               # reload the saved run
```

Reloaded runs carry their output arrays and plotter, so you can re-plot or
post-process without re-running.
