# Outputs, plotting, and saved runs

## Plotting

Every model carries a plotter at `m.plot`:

- **1D** — `m.plot.profile()` draws the long profile and returns
  `(fig, axes)`.
- **2D raw fields** — `m.plot.map(field='bedrock')` displays one stored raster
  without terrain smoothing and returns the `Axes`; `view_map` is its notebook
  slider and `animate_map` its movie.
- **2D channel profiles** — `m.plot.profile()` extracts a channel and returns
  `(fig, axes)`. `m.plot.analytical()` draws the analytical steady-state
  reference alone, fitted to the channel `profile` draws (`basin_rank`, `ref`),
  with `profile`'s renderer, figure and axes. The 1D `analytical()` takes the
  same arguments, `(ax=None, bistable=True, basin_rank=0, ref=-1)`, and ignores
  `basin_rank` and `ref`.
  `view_xsection` is a notebook viewer of one raster row, with sliders for the
  snapshot and the row's y.
- **2D flux maps** — `m.plot.flux(which='ice')` maps the ice flux, `'water'`
  the water flux and `'area'` the catchment area on a log scale, and returns
  the `Axes`. It and the basin maps share `map`'s km coordinates, so a km
  overlay drawn on any of them lands in the same place.
- **2D cartography** — `m.plot.landscape()` returns `(fig, ax)` and has the
  display presets described below.

All take an output index `i` (default `-1`, the last frame). `map` and
`profile` use `field_min` / `field_max`; `landscape` uses `z_min` / `z_max`
for terrain (`z_min` defaults to −500 m; a bed below it takes the lowest
colour, and the bed colorbar gets an arrow at each end the map saturates past;
`animate_landscape` decides the arrows once for the whole movie) and
`H_min` / `H_max` for ice colour. `map` spans its field's data range over all
outputs, while `landscape` fixes the bed scale's floor at `z_min`, so its bed
colours compare across runs. See {doc}`../api/model1d` and
{doc}`../api/model2d` for the method signatures.

Each quantity has one colour wherever it is drawn. The bed is a grey line over
one translucent fill (the analytical package's `draw_profile`, the paper
renderer behind `GeneralProfile.plot` and `SteadyStateProfile.plot`, keeps its
depth-faded fill, in the same grey); ice lines and fills share one blue pair;
the ELA has one colour, the analytical plots included. Depth-coded ice maps
(`landscape`'s veil, `map(field='ice')`, `flux('ice')`) use one glacier ramp,
and `flux('water')` a ramp up to the water-flux colour of the profile panels.
`landscape`'s lakes (the `lake_color` default) and the 1D profile's ponds
share one standing-water colour. The colours live in `siim._palette.COLORS`,
which needs no matplotlib.

Grid coordinates read `x (km)` and `y (km)`, the 1D profile's included;
distance along an extracted 2D channel reads `Distance from divide (km)`.
`Mean ice thickness` is the width-mean `H` the physics consumes, and
`Ice column depth` the drawn column `hc_over_H*H` (the landscape bar and the
analytical `plot_profile` panel).

New figures are 12 inches wide for multi-panel figures and `landscape`, and
6 inches for single panels (`map`, `flux`, `hacks_law`, `analytical` and a
one-field `profile`). Every figure but `landscape`, which places its panels
itself, uses Matplotlib's constrained layout.

The bed colormap of `landscape` and `map(field='bedrock')` is Fabio Crameri's
turku with its darkest 20 % trimmed, from his *Scientific colour maps*
(version 8.0, [doi:10.5281/zenodo.8035877](https://doi.org/10.5281/zenodo.8035877)),
as cmcrameri distributes it under the MIT licence. Its colour table ships
inside siim, with that notice, so cmcrameri is not needed;
`cmap_bed` (`landscape`) or `cmap` (`map`) overrides it. Every plot takes its
font sizes from the active Matplotlib rcParams, so a house style sets them.

### Shared profile behavior

The 1D and 2D `profile`, `view_profile` and `animate_profile` methods use the
same parser and renderer. Field names are case-insensitive. Pass a name, a
list of names, or `{name: (minimum, maximum)}`; either bound may be `None`.
Automatic bounds cover finite data and analytical/forcing references across
all output snapshots, retain negative elevations, and use the displayed units
(including kPa for shear stress). Logarithmic erosion plots mask nonpositive
values; an all-zero frame is labeled explicitly. Manual limits must be finite,
ordered and positive on a log axis.

```python
fig, axes = m.plot.profile(
    fields={'elevation': None, 'ice_thickness': (0, 800)},
    fig_width=12, aspect=0.25, legend=True,
)
```

`fig_width` is inches; `aspect` is each panel's height/width. One field
defaults to a 6-inch figure at aspect 0.38, a stack of fields to 12 inches at
0.25 (panels about 3 inches tall), with legends outside the data panel. The
viewer and `view_xsection` take the same defaults and default to no legend. Bedrock, ice, water and ELA share colors; analytical references are
dashed and current ELA is solid. Basin comparisons instead use a fixed palette
to identify basins across maps and histories, with fainter lines for bedrock.
All snapshots use the same year/kyr/Myr time format.

`i` selects the displayed snapshot. In 2D, `ref` selects the channel extraction
frame and `basin_rank` selects its basin; the analytical overlay belongs to
that channel even after other extractions. The uplift curve follows that
channel's spatial forcing at the displayed output time. In 1D, `ref` and
`basin_rank` are accepted for call compatibility and have no effect. Only 1D
currently exposes `shear_stress` and `sliding_velocity`.

Static `profile(ax=...)` calls add to the supplied axes, preserving existing
measurement lines, annotations and tick formatters. Viewer and movie updates
clear their own panels between frames. Profile coordinates retain the model's
ordering: 1D runs from `m.x[0]` to `m.x[-1]` (normally `L` down to zero), while
2D extracted channels run from the divide toward the outlet. Cross sections
retain their ascending raster x coordinate.

### Basin and steady-state summaries

`largest_basins`, `largest_basins_history`, `sediment_history`, `hacks_law`
and `steady_state` support `plot=False` to calculate their results without
creating a figure. Calculations, drawing and text formatting have separate
internal helpers. Calls are quiet by default; pass `verbose=True` to
`hacks_law`, `largest_basins_history` or `steady_state` for a printed summary.

```python
history = m.plot.largest_basins_history(ref=-1, n_samples=20, plot=False)
sediment = m.plot.sediment_history(ref=-1, quantity='flux', plot=False)
ss_step = m.plot.steady_state(plot=False)
```

`ref` has the same meaning throughout the plotting API: it selects the output
frame used to extract a reference channel or choose basin outlets. Histories
keep those outlet identities across the run. The old `i_ref` keyword remains
accepted by both history methods, with a `DeprecationWarning`, until its
removal in 1.0; use `ref` in new calls. Positional reference arguments retain their meaning. Pass one
spelling; the legacy alias can override the default `ref=-1`.

`hacks_law(i=..., ref=..., basin_rank=...)` displays area at `i` on the channel
selected and fitted at `ref`. Its default reference remains the final output.
For `largest_basins(i=...)`, the ranking and snapshot both come from `i`.
The historical `t_start` and `t_end` names in `largest_basins_history` denote
**inclusive output-step indices**, not years; negative indices count backward
from the final output. Empty/reversed windows and nonpositive sample counts
are rejected.

Existing results are preserved: basin methods return metric namespaces, and
`steady_state` returns the first matching output index (or -1). `hacks_law`
returns an Axes when plotting; with `plot=False` it returns fit parameters
(`k_h`, `d`, `xo`, `L`, in SI units), scatter arrays (`distance_km`, `area_km2`)
and fitted-curve arrays (`fit_distance_km`, `fit_area_km2`). Basin history results
also expose `xt_over_L` and `zo_over_zELA`. Data times stay in years even when
displayed axes use kyr or Myr.

Public summary plots share semantic colors and readable time units.
They use constrained layouts and `fig_width` in inches. Supply `ax` to compose
them into your own figure: a pair for `largest_basins` or `steady_state`, nine
axes in row order for `largest_basins_history`, and one Axes for `hacks_law`.
`sediment_history` accepts a single series Axes or a map/series pair. Supplied
axes must belong to one figure; other axes and figure titles are preserved.

The 1D `_limit_cycle` and `_limit_cycle_phase` methods are private research
helpers. Their former public names have been removed; their drawing behavior
is retained, and their appearance is outside the public plotting style contract.

### Movies and viewers

`animate_profile` and `animate_map` retain `fps=20`; explicit `fps` determines
the encoded rate. Pass `fps=None` to derive it from `interval` in milliseconds
per frame. `animate_landscape` derives its rate from `interval` (default 42 ms)
unless an explicit `fps` is given, on both its serial and parallel render
paths. All three accept `frames` — a slice, a range or a sequence of
saved-frame indices, negatives counting from the end — and encode just those,
in the order given. A movie `path` may include `.mp4`; it is added only once. Existing `run_id`
filename conventions are preserved and take precedence over `path`.
`animate_landscape(quality='draft')` is for research movies that will soon be
replaced: at most `oversample=2` and x264's `veryfast` preset, about 1.6-1.7x
faster than the default `quality='high'`, the still render's full look.

Plotter-created movie figures close even if encoding fails. Supplied landscape
figures/axes stay open; unrelated axes are preserved. All supplied axes must
belong to the same figure. Map viewers update one image and colorbar in place.
Sliders redraw on release to avoid queuing expensive renders while dragging.

## Smooth and raw landscape views

`landscape` defaults to `style='smooth'`, an atlas-style presentation. With no
other arguments it uses a 12-inch figure and shows bedrock **and** ice (`field='bedrock+ice'`); terrain is
supersampled 4×, Gaussian-de-staircased, hillshaded, and contoured, and the
same preset:

- draws ice across the claimed sub-grid valley width (`ice_extent='footprint'`);
- shades it as a depth-graded translucent veil (`ice_shading='veil'`), whose
  opacity is an S-curve in column depth, 50 % at half `ice_opaque_depth`
  (default 100 m) and opaque from it down, its steepness set by
  `ice_alpha_sharpness` (default 2, so 20 m of ice is 6 % opaque);
- draws no trunk ribbons (`trunk_display='none'`);
- hides nothing by thickness (`H_threshold=0`) — the veil already fades
  thin ice out, so a gate would only delete real glacierets;
- smooths the ice outline by half a model cell (`ice_sigma_cells` defaults to
  half the oversample: two sub-grid pixels at the default 4), which can erase a
  glacier that occupies a single grid cell (`ice_sigma_cells=0` keeps it), and
  outlines the current ice margin (`show_margin`, formerly `show_trimline`) —
  the ribbons only, when they are asked for; and
- leaves connected-component removal off.

The hillshade is lit from `azdeg` degrees clockwise from north (default 315,
the NW) at `altdeg` above the horizon. Its relief is exaggerated by `ve`
(default 4) for the shading only. A cross-section panel is sized for `section_ve` (default 4) over
`z_min`..`z_max`, between 0.8 and 2.6 in tall, and prints the exaggeration it
got; the ice colorbar, like the bed's, gets an arrow where the model's ice
column `hc_over_H*H` runs past its `H_min`/`H_max` (`extend_ice`).

This is intentionally a cartographic view, not an unmodified-cell view:

```python
m.plot.landscape()
```

Pass `field='bedrock'` for the bare bed. The veil's colour normalises column
depth on `H_max`, which defaults to the run-global `hc_over_H * max H_out`
(the largest `hc_cell_out * H_out` in mode D), so a still and
every frame of an animation put the same colour on the same depth;
`ice_shading='flat'` restores a single opaque `ice_color`, and an explicit
`ice_cmap` overrides both.

`trunk_display='ribbons'` adds an optional layer over the veil. A **trunk**
cell is any icy cell downstream of one whose claimed width
`W = alpha_g*H` already spans `trunk_width_cells` grid cells (default `1.0`,
the width the grid can just resolve), the class carried along the receivers to
the terminus so a thinning tongue stays a trunk down to its toe; those cells
are traced and drawn at their true width, at a minimum opacity of
`trunk_alpha` (deep ice keeps the depth ramp's own higher value), over a
veil built from the sub-resolution ice that is left. Raise
`trunk_width_cells` to reserve the ribbons for the widest glaciers. They are
off by default: the footprint already draws the carve's cross-section wherever
a glacier spans more than about two cells, and on sheet runs the ribbons drew
rings, because their trunk class reads sheet ice as channel.

For inspection use `style='raw'`. It selects `field='bedrock+ice'` with flat
ice (`ice_shading='flat'`), renders one pixel per cell, disables terrain/ice
smoothing, hillshade, contours and the margin outline, and shows channel-cell
ice (`ice_extent='cells'`).
Both presets leave `H_threshold=0`; the raw preset
additionally applies an upstream-area gate of
`1e6 m²` to suppress small-catchment specks. To display every ice-bearing cell,
disable that gate explicitly:

```python
m.plot.landscape(style='raw', area_threshold=0)
```

Every explicit keyword overrides its preset value. `animate_landscape`
inherits the smooth preset. `fps` and `frames` are movie knobs rather than
`landscape` ones — encode the last twenty saved frames at ten frames per
second with:

```python
m.plot.animate_landscape(fps=10, frames=slice(-20, None))
```

`H_threshold` gates width-mean H in meters; `area_threshold` gates upstream
area in m². Both apply to footprint and ribbon sources. `min_ice_cells` cleans
the combined veil/ribbon mask once, so connected parts count as one glacier.
With `ice_smoothing='field'`, the veil retains thickness near the threshold
until the smoothed field is thresholded; ribbon-source gating is separate.
`sigma_cells` and `ice_sigma_cells`
are subgrid pixels; both default to follow `oversample` (`sigma_cells` equal to
it, `ice_sigma_cells` half of it), so an explicit value's native-grid strength
changes with `oversample` while the defaults' does not.

`min_ice_cells=6` can remove small components, but it can also hide real
small glaciers and is therefore never enabled by a preset.

`cross_section` adds the section and hypsometry panels. A `y` in km reads that
one row and marks it on the map; `'mean'` averages over the rendered rows of
the section grid instead — the interpolated `oversample` grid, whose two end
rows carry less ground than the interior ones — so the painted band is the
average ice column per unit x and vanishes where nothing is iced. The mean
section also shows the spread behind that mean — the ice thickness's 25–75%
band shaded on the mean bed, the bed's 25th and 75th percentiles as dashed
lines, both named in a legend — so a flat mean over one landscape reads
differently from a flat mean over several. The band is the spread of the ice
column, not an envelope of surfaces: a surface band would mostly repeat the
bed's spread. An averaged profile has no single y to mark, so the map gets
no locator line, and it skips the lake layer — a reduced water table is not a
water table:

```python
m.plot.landscape(cross_section='mean')
```

The hypsometry shares the section's elevation axis (`z_min`..`z_max`): bed
outside it is off the panel, and the bars are percentages of the whole area.
`hyp_max` (default 15 %) fixes the percent-area scale; a taller bar runs off
it. A `y` outside the domain warns and draws the edge row.
`animate_landscape` freezes the colour scales and the hillshade's contrast
stretch over the movie.

### Ice surface speed

`field='bedrock+speed'` draws the same ice as `'bedrock+ice'`, coloured by
ice surface speed in m/yr over grey hillshaded terrain, on a log scale
(`speed_min=20`, `speed_max=1000`, `speed_cmap='magma_r'`). Ice slower than
`speed_min` is left transparent, and faster ice saturates. `animate_landscape`
takes the same field and keywords:

```python
m.plot.landscape(field='bedrock+speed')
m.plot.animate_landscape(field='bedrock+speed', speed_max=2000)
```

Speed is not stored. `m.ice_speed(i)` recomputes it by running the flux
closure backwards on the flux and thickness the closure solved at output `i`,
and returns the surface speed `us` and the cross-section mean
`U = Q/(alpha_g*H**2)`. That pair is `Qg_out` and `H_out` unless the run
stores the closure's own, which it does only with `save_closure_pair=True`
(off by default; the rows add about 10 % to the saved output in modes B and C
and about 19 % in mode D): a transect `closure_flux` (the mode B/C/D default)
closes on the transect flux, kept as `closure_Qg_out`, and `inner_relax < 1`
commits a blend of passes, so the closure's thickness is kept as
`closure_H_out`. The split is
`us = ub + (5/4)*ud`, with depth-mean deformation `ud` and sliding
`ub = U - ud`, so `U <= us <= 1.25*U`. `'eff-exp'` has no sliding split, so
there `us = U`. Outlet (self-receiving) cells read 0: their thickness is
copied from the upstream cell, not solved from their own flux. A run with
`H_diffusivity` smooths H inside the kernel after the closure solve, so its
speeds are approximate, and so are those of a run that needs the closure
rows but did not save them (the default); `ice_speed` warns in both cases. In the
footprint view each pixel takes the speed of the source
cell whose ice surface it shows. `trunk_display='ribbons'` draws trunk
ribbons, each carrying the speed of the
cells along its path.

Two things to keep in mind when reading the map. Every icy cell is a channel
of width `W = alpha_g*H`, so thick interior ice that covers its cell as a sheet
(`W > dx`) reads slow: the same flux through the cell width would move about
`W/dx` times faster. Under `'coulomb'`, a few thin-ice cells near the yield stress `tau_c`
can show very large speeds; the colour scale saturates them.

### Two ice thicknesses

`H_out` is the width-**mean** ice thickness — the quantity the flux closures,
sliding laws, erosion laws and `tau = rho*g*H*S` consume, and the one
`map(field='ice')` and the profile panels plot ("Mean ice thickness"). What
`landscape` renders is instead the local **column depth** to the flat ice
surface: `hc_over_H*H` at the channel floor (default ratio 1.5), and deeper on carved flanks, which is
why its colorbar reads "Ice column depth (m)".

Basin histories record unavailable channels/fits as `NaN`; summary statistics
exclude missing samples and report valid counts in `result.stats[metric]['n']`.
A single output cannot establish steady state, so `steady_state()` returns -1.

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
| `receivers_out` | `glacial_flow__receivers_2d` | receiver graph (the largest-weight receiver under a multi-receiver router) |
| `stack_out` | `glacial_flow__stack_2d` | topological routing stack |
| `basin_out` | `glacial_flow__basin_ids` | basin identifiers |
| `lengths_out` | reconstructed from receivers/grid | node-to-receiver distance |
| `rebound_out` | `flexure__rebound` | flexural displacement; present when `flexure=True` |
| `sediment_flux_out` | `sediment__flux` | sediment throughput; present when `track_sediment` reports by basin |
| `eroded_volume_out` | `sediment__cumulative` | cumulative eroded volume; present when `track_sediment` reports by basin |
| `sediment_edge_flux_out` | `sediment__edge_flux` | volume leaving the domain across each edge, shaped `(time, side)`; present when `track_sediment` reports by edge |
| `sediment_edge_cumulative_out` | `sediment__edge_cumulative` | its running total, shaped `(time, side)` |
| `hc_cell_out` | `glacial_spl__hc_cell` | per-cell channel-floor ratio, `hc_over_H` (1.5) on valley cells to `1` on sheet cells; present in mode D (otherwise `None`) |
| `channel_out` | `glacial_spl__channel` | mode D's channel/sheet class, `True` on icy cells read channel, read on the frame's own thickness (bool; stored as int32); present in mode D (otherwise `None`, and `None` for mode-D runs saved before 0.10.0) |
| `closure_Qg_out` | `glacial_spl__closure_ice_flux` | the flux the thickness closure solved H from (last pass); present with `save_closure_pair=True` when `closure_flux` is a transect (otherwise `None`) |
| `closure_H_out` | `glacial_spl__closure_ice_thickness` | the kernel's thickness before the `inner_relax` blend (last pass; after any `H_diffusivity` smoothing); present with `save_closure_pair=True` when `inner_relax < 1`, as under the default damping of modes B, C and D (otherwise `None`) |

`track_sediment` chooses what the routed sediment is reported as: `False`
(off, the default), `True` or `'basin'` (the per-node rasters above),
`'edge'` (the per-domain-edge totals only) or `'both'`. Any other value is
rejected.

The two `(time, side)` arrays carry the `side` coordinate
`['left', 'right', 'bottom', 'top']` — `boundary_status` order, the same keys a
per-side `bl` dict takes. Each entry is this step's routed `sediment__flux`
summed over that edge's outlet ring, in m³ per step, with each corner node
counted exactly once (an outlet left/right edge owns its corners, otherwise the
bottom/top edge does). Only a `'fixed_value'` edge is an outlet, so a `'core'`
or `'looped'` edge reads `NaN` rather than zero: nothing can leave there, which
is not the same as nothing leaving. With every side `'fixed_value'`, the four
entries sum to the whole domain's denuded volume for that step.

### Primary-path diagnostics under multiple-flow routing

`receivers_out` carries one receiver per cell, so under `flow_routing='dinf'`
and especially `flow_routing='mfd'` it is the largest-weight link, not the whole
flow graph. Everything built on it -- `extract_channel`, `channel_statistics`
(Strahler order and the Horton ratios), the basin plots and the trunk ribbons --
is therefore a PRIMARY-PATH diagnostic under MFD: it describes the dominant
network. At the default `mfd_exponent = 1.0` the dominant link typically carries
under half of a cell's flux, so those views understate how much water and ice
travel off the primary path. `basin_out` is likewise the D8 chain on the same
eps-filled surface (the deterministic labeling both multi-receiver producers
reuse), and may disagree with the MFD largest-weight graph. The mass-carrying
fields -- `area_out`, `Qg_out`, `Qf_out` and everything the kernels compute --
use the full weighted graph and are unaffected.

Mode B/C stores the bed as `topography__elevation` and reconstructs the ice
surface as `z = zb + hc_over_H*H`. Mode A stores the ice surface as topography
and a separate bed field. `hc_over_H` is the channel-floor ratio (default 1.5,
`constants.HC_OVER_H`); `H` remains the width-mean thickness used by the
physics. In mode D the
ratio is per cell and the reconstruction is `z = zb + hc_cell_out*H`, the
surface the model balanced on; `landscape` draws its ice columns on the same
datum.

## The true-state output convention

Outputs report the true `(zb, H)` everywhere; base level is not a presentation
floor in stored state. With nonzero `bl`, a drowned or relict bed may lie below
the waterline. That is bed memory, not a failed clamp. Lakes are a display
layer of `landscape` (`field='bedrock+lakes'`): closed depressions in the bed,
flooded to their spill level and drawn only where the same preset would draw
no ice. No sea or ocean surface is drawn, and the lakes are not substituted
into the output arrays.

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
