# Outputs, plotting, and saved runs

## Plotting

Every model carries a plotter at `m.plot`:

- **1D** — `m.plot.profile()` draws the long profile and returns
  `(fig, axes)`.
- **2D raw fields** — `m.plot.map(field='bedrock')` displays one stored raster
  without terrain smoothing and returns the `Axes`.
- **2D channel profiles** — `m.plot.profile()` extracts a channel and returns
  `(fig, axes)`.
- **2D cartography** — `m.plot.landscape()` returns `(fig, ax)` and has the
  display presets described below.

All take an output index `i` (default `-1`, the last frame). `map` and
`profile` use `field_min` / `field_max`; `landscape` uses `z_min` / `z_max`
for terrain and `H_min` / `H_max` for ice colour. See {doc}`../api/model1d` and
{doc}`../api/model2d` for the method signatures.

## Smooth and raw landscape views

`landscape` defaults to `style='smooth'`, an atlas-style presentation. With no
other arguments it shows bedrock **and** ice (`field='bedrock+ice'`); terrain is
supersampled 4×, Gaussian-de-staircased, hillshaded, and contoured, and the
same preset:

- draws ice across the claimed sub-grid valley width (`ice_extent='footprint'`);
- shades it as a depth-graded translucent veil (`ice_shading='veil'`);
- fills the resolved trunk glaciers as true-width ribbons over that veil
  (`trunk_display='ribbons'`);
- hides nothing by thickness (`H_threshold=0`) — the veil already fades
  thin ice out, so a gate would only delete real glacierets;
- smooths the ice outline by two sub-grid pixels, and outlines the current ice
  margin (`show_margin`, formerly `show_trimline`) — the ribbons only, when
  they are drawn; and
- leaves connected-component removal and time averaging off.

This is intentionally a cartographic view, not an unmodified-cell view:

```python
m.plot.landscape()
```

Pass `field='bedrock'` for the bare bed. The veil normalises column depth on
`H_max`, which defaults to the run-global `1.5 * max H_out`, so a still and
every frame of an animation put the same colour on the same depth;
`ice_shading='flat'` restores a single opaque `ice_color`, and an explicit
`ice_cmap` overrides both.

A **trunk** cell is any icy cell downstream of one whose claimed width
`W = alpha_g*H` already spans `trunk_width_cells` grid cells (default `1.0`,
the width the grid can just resolve), the class carried along the receivers to
the terminus so a thinning tongue stays a trunk down to its toe; those cells
are traced and drawn at their true width, at a minimum opacity of
`trunk_alpha` (deep ice keeps the depth ramp's own higher value), over a
veil built from the sub-resolution ice that is left. Pass
`trunk_display='none'` for the veil alone, or raise `trunk_width_cells` to
reserve the ribbons for the widest glaciers.

For inspection use `style='raw'`. It selects `field='bedrock+ice'` with flat
ice (`ice_shading='flat'`), renders one pixel per cell, disables terrain/ice
smoothing, hillshade, contours, the margin outline and the trunk ribbons
(`trunk_display='none'`), and shows channel-cell ice (`ice_extent='cells'`).
Both presets leave `H_threshold=0`; the raw preset
additionally applies an upstream-area gate of
`1e6 m²` to suppress small-catchment specks. To display every ice-bearing cell,
disable that gate explicitly:

```python
m.plot.landscape(style='raw', area_threshold=0)
```

Every explicit keyword overrides its preset value. `animate_landscape`
inherits the smooth preset; for densely sampled animation output, an optional
anti-flicker recipe is:

```python
m.plot.animate_landscape(field='bedrock+ice',
                         H_threshold=50, ice_sigma_cells=3, ice_time_avg=2)
```

`ice_time_avg` changes only the displayed ice layer, not terrain or stored
state. `min_ice_cells=6` can remove small components, but it can also hide real
small glaciers and is therefore never enabled by a preset.

### Two ice thicknesses

`H_out` is the width-**mean** ice thickness — the quantity the flux closures,
sliding laws, erosion laws and `tau = rho*g*H*S` consume, and the one
`map(field='ice')` and the profile panels plot ("Mean ice thickness"). What
`landscape` renders is instead the local **column depth** to the flat ice
surface: `1.5*H` at the channel floor, and deeper on carved flanks, which is
why its colorbar reads "Ice column depth (m)".

## 1D arrays

The 1D model stores NumPy arrays directly. Spatial output has shape
`(nx, n_output)`; `output_times` and `zELA_out` have length `n_output`.

| attribute | meaning |
|---|---|
| `z_out`, `zb_out` | ice-surface elevation and channel-floor bed elevation |
| `H_out` | width-mean ice thickness |
| `Qg_out`, `Qf_out` | ice and water flux |
| `erosion_rate_out` | erosion rate |
| `tau_out`, `ub_out` | basal shear stress and sliding velocity |
| `B_out` | local ice mass-balance rate |
| `zo_out`, `xd_out` | divide elevation and position, shaped `(n_sides, n_output)` |
| `Lt_out`, `zLt_out` | terminus position and elevation, shaped `(n_sides, n_output)` |

The first dimension of spatial arrays follows `m.x`, which runs from `L` down
to zero. The 1D model has no built-in run `save`/`load` method.

## 2D dataset and arrays

The standalone driver builds `m.ds_out` as an **in-memory xarray Dataset**; a
normal run does not create a Zarr store. The convenience attributes below are
NumPy views of that dataset and have shape `(time, y, x)` unless noted.

| attribute | `ds_out` variable | meaning |
|---|---|---|
| `z_out` | reconstructed or `topography__elevation` | ice-surface elevation `zs` |
| `zb_out` | `topography__elevation` in B/C; `glacial_spl__bedrock_surface` in A | channel-floor bed elevation |
| `H_out` | `glacial_spl__ice_thickness` | width-mean ice thickness |
| `area_out` | `glacial_flow__area` | upstream drainage area |
| `Qg_out` | `glacial_flow__ice_flux` | ice flux |
| `Qf_out` | `glacial_flow__water_flux` | water flux |
| `erosion_rate_out` | `glacial_spl__erosion_rate` | erosion rate |
| `denudation_out` | `glacial_spl__denudation` | per-step rock removed: Δ`zb` in B/C including carving, or Δ`zs` in A |
| `receivers_out` | `glacial_flow__receivers_2d` | receiver graph |
| `stack_out` | `glacial_flow__stack_2d` | topological routing stack |
| `basin_out` | `glacial_flow__basin_ids` | basin identifiers |
| `lengths_out` | reconstructed from receivers/grid | node-to-receiver distance |
| `rebound_out` | `flexure__rebound` | flexural displacement; present when `flexure=True` |
| `sediment_flux_out` | `sediment__flux` | sediment throughput; present when `track_sediment=True` |
| `eroded_volume_out` | `sediment__cumulative` | cumulative eroded volume; present when `track_sediment=True` |

Mode B/C stores the bed as `topography__elevation` and reconstructs the ice
surface as `z = zb + 1.5*H`. Mode A stores the ice surface as topography and a
separate bed field. `1.5` is the channel-floor ratio `HC_OVER_H`; `H` remains
the width-mean thickness used by the physics.

## The true-state output convention

Outputs report the true `(zb, H)` everywhere; base level is not a presentation
floor in stored state. With nonzero `bl`, a drowned or relict bed may lie below
the waterline. That is bed memory, not a failed clamp. Ocean and lake surfaces
are display layers used by `landscape`, not substitutions in the output arrays.

## Files and output locations

Plot methods return Matplotlib objects and do not save unless their save/path
argument requests it. Relative names are placed beneath the current working
directory:

- images: `model_outputs/images/`
- movies: `model_outputs/movies/`
- 2D saved runs: `model_outputs/saved_models/`

An absolute output name is used as given.

MP4 animation methods use Matplotlib's ffmpeg writer and require a system
`ffmpeg` executable on `PATH`.

## Saving and loading a 2D run

```python
path = m.save('run_name')          # adds .pkl; returns the path written

from siim.siim2d import load
restored = load('run_name')
```

`save` writes atomically and records a save-format schema version, the
producing SIIM version, the concrete model class, the original user-parameter
dictionary, and `ds_out`. `load` checks the envelope, schema version, model
identity, required keys, and value types; the producing package version is
recorded for diagnostics. Unsupported, malformed, and old unversioned payloads
fail with a directed error; cross-version migration is not promised.

These files use Python pickle. **Load only files you trust:** unpickling can
execute code before SIIM can validate the decoded object. The validation
detects incompatible SIIM state; it is not a security sandbox.

The saved parameter dictionary preserves the user's original inputs. External
resources referenced there, such as an initial-topography CSV path, are not
embedded automatically and must still be available when the model is rebuilt.
Reloaded compatible runs carry their dataset, convenience arrays, and plotter,
so they can be plotted and post-processed without rerunning.
