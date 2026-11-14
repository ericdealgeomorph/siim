# Use siim's glacial processes in your own fastscape model

siim's glacial-erosion physics is packaged as a set of composable xsimlab
processes under `siim.fastscape`, for use in your own fastscape model without
siim's `siim2d` wrapper.

This is the **optional adapter** — the standalone 2D model (`siim.siim2d`)
runs its own in-house time loop and needs none of it (see {doc}`../getting_started/install`).
These snippets require the fastscape stack: `pip install siim-lem[fastscape]`
installs `fastscape` + `xsimlab`, but the Fortran backend
(`fastscapelib-fortran`) that fastscape's stock processes call at run time is
conda-only (`conda env create -f environment.yml`). Missing `fastscape` or
`xarray-simlab` makes `import siim.fastscape` raise a directed `ImportError`;
missing `fastscapelib-fortran` may instead fail when a stock Fastscape process
first runs. The snippets are not executed at build time.

## The quickest path

`glacial_model()` returns a ready-to-run xsimlab `Model` — fastscape's
`basic_model` with its fluvial `spl` / `drainage` slots replaced by siim's
glacial processes:

```python
from siim.fastscape import glacial_model

model = glacial_model(mode='B')          # bedrock + ice thickness, single-flow
```

It runs like any xsimlab model (`xsimlab.create_setup` /
`model.xsimlab.run`), with the glacial inputs the processes declare supplied in
the setup.

## Composing into your own model

For more control, `glacial_processes()` returns the dict of process-class
overrides; apply it after dropping the stock fluvial slots:

```python
from fastscape.models import basic_model
from siim.fastscape import glacial_processes

model = (basic_model
         .drop_processes(['spl', 'drainage'])
         .update_processes(glacial_processes(mode='B', routing='dinf')))
```

`glacial_processes()` is the composition source of truth for the xsimlab
adapter. Its process shells and the standalone 2D driver call the same
framework-free step functions, so a model built this way runs the same physics.

Both helpers take the same options:

| option | values | meaning |
|--------|--------|---------|
| `mode` | `'B'` (default), `'A'`, `'C'` | bedrock + ice thickness (bed memory) vs ice surface vs `'C'` = B + carve; case-insensitive, and the long names `'bedrock+ice_thickness'` / `'ice_surface'` work too. Mode `'D'` raises: it needs `siim2d`'s in-house driver |
| `carve` | `None` → off (default), `True`, `False` | sub-grid glacier-width carving (mode B); `mode='C'` is the alias that turns it on (`mode='C', carve=False` raises) |
| `trunk_surface` | `None` → mode-dependent (default), `True`, `False` | fabricated trunk-surface routing — converge trunk flow onto the centerline (mode B/C); `None` is on for mode C, off for plain B |
| `routing` | `'single'` (default), `'multiple'`, `'infinite'` | D8 single-flow, multiple-flow-direction, or Tarboton D-infinity; the router codes `'mfd'` (= `'multiple'`) and `'dinf'` (= `'infinite'`) are accepted as aliases. `'mfd'` raises for `mode='A'`, and the remaining guards (a transect `glacial_spl__closure_flux`, square cells) are enforced by the processes that own those inputs. Mode C with `trunk_surface=True` is supported; the dip input `surf2erode__trunk_dip_k` then defaults to the MFD value 1.0 (`constants.TRUNK_DIP_K_MFD`), and an explicit value wins. Set `flow__mfd_exponent` alongside it; `glacial_spl__slope_normalise` defaults to `None`, which resolves ON for any multi-receiver router (D-inf and MFD) and OFF for single-flow. |
| `flexure` | `False` (default), `True` | add glacial-isostatic flexure |
| `sediment` | `False` (default), `True` | add sediment tracking |

Every flag defaults off except `trunk_surface`, which follows the *mode-C
standard* exactly as the {class}`siim.siim2d.siim` wrapper does: on for mode C
(`mode='C'`, or `mode='B'` with `carve=True`), off otherwise.
`glacial_model(mode='C')` with no extra inputs therefore **is** native mode C:
trunk-surface routing on (`TrunkSurfaceToErode` in the `surf2erode` slot) and
`widening_rate=3`, run as `siim2d` runs mode C with
`inner_passes=1, inner_relax=1.0` (this facade routes once per step, so it has
no within-step damping). An explicit value always wins; `trunk_surface=False`
gives the plain carved composition (`GlacialSurfaceToErode`).

## The processes

Every class the helpers wire in is exported from `siim.fastscape`:
`GlacialLaw`, `GlacialFlowAccumulator`, `GlacialSPLModeA`, `GlacialSPLModeB`,
`GlacialSPLModeC`, `GlacialSurfaceToErode`, `TrunkSurfaceToErode`,
`GlacialFlexure`, `SedimentTracker`, `D8FlowRouter`, `DinfFlowRouter`,
`MfdFlowRouter`, `GlacialBlockUplift`, `HillslopeDiffusion` and
`InitialTopography`, together
with two reusable forcing processes (`WaveUplift`, `PlateauSurface`). A model
can therefore be written out as an explicit `xs.Model({...})` from these and
fastscape's stock processes alone. Across the modes only two slots change:
`glacial_spl` (`GlacialSPLModeA` / `GlacialSPLModeB` / `GlacialSPLModeC`) and
`surf2erode` (fastscape's `SurfaceAfterTectonics` / `GlacialSurfaceToErode` /
`TrunkSurfaceToErode` for the mode-C standard; under `routing='mfd'` the
helpers wire a private subclass of it whose only difference is the
`trunk_dip_k` default, 1.0, which an explicit model sets as an input). The
classes can also be
subclassed to customise behaviour. `GlacialBlockUplift` and `WaveUplift` publish
their local, unmasked uplift rate in the `border_bed_uplift` group, and the
mode-B and C processes take it as the border-bed uplift while
`glacial_spl__border_bed_uplift` is left at `None`, as siim2d does: a border bed
the ice deepened below `bl` recovers toward it at the local rate (never above
it). A value given is used alone; `0` freezes the recovery. `TrunkSurfaceToErode`
publishes `routing_elevation` in the `router_surface` group; the in-house
routers route on that surface when present and on `elevation` otherwise.

See {doc}`../api/fastscape` for the full reference.
