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
| `mode` | `'B'` (default), `'A'`, `'C'` | bedrock + ice thickness (bed memory) vs ice surface vs `'C'` = B + carve |
| `carve` | `None` → off (default), `True`, `False` | sub-grid glacier-width carving (mode B); `mode='C'` is the alias that turns it on (`mode='C', carve=False` raises) |
| `trunk_surface` | `False` (default), `True` | fabricated trunk-surface routing — converge trunk flow onto the centerline (mode B/C) |
| `routing` | `'single'` (default), `'dinf'` | D8 single-flow vs Tarboton D-infinity |
| `flexure` | `False` (default), `True` | add glacial-isostatic flexure |
| `sediment` | `False` (default), `True` | add sediment tracking |

This facade stays **explicit**: every flag defaults off, with no mode-resolved
defaults. In particular `glacial_model(mode='C')` selects B + carve but leaves
`trunk_surface` off — the *mode-C standard* (trunk-surface routing on,
routing relaxation, `widening_rate=3`) is applied only by the
{class}`siim.siim2d.siim` wrapper. To reproduce it here, pass
`carve=True, trunk_surface=True` and set `routing_relax` / `widening_rate` on
the model inputs.

## The processes

The classes the helpers wire in are exported from `siim.fastscape` (e.g.
`GlacialLaw`, `GlacialFlowAccumulator`, `GlacialSPLModeA`, `GlacialSPLModeB`,
`GlacialSPLModeC`, `GlacialSurfaceToErode`, `TrunkSurfaceToErode`,
`GlacialFlexure`, `SedimentTracker`, `DinfFlowRouter`), together with two
reusable forcing processes (`WaveUplift`, `PlateauSurface`). They can be
imported and subclassed directly to customise behaviour.

See {doc}`../api/fastscape` for the full reference.
