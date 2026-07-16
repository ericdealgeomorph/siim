# Your first 2D run

The 2D model (`siim.siim2d`) is a raster landscape model — since 0.9.1 it runs
standalone on a plain pip install (in-house time loop, flow routing, flexure,
and diffusion; see {doc}`install`). These snippets are **not executed** at
docs-build time only to keep the build fast; they run anywhere siim is
installed.

```python
from siim.siim2d import siim as siim2d

m = siim2d({'zELA': 1000, 'nx': 81, 'ny': 81, 'Lx': 2e4, 'Ly': 2e4,
            'T': 5e5, 'nt': 251})
m.run()
m.plot.map(field='bedrock')      # raster field; or m.plot.landscape() for relief
```

The 2D model takes the same kind of parameter dict as the 1D model. This is the
default carved configuration — `'bedrock+ice_thickness'` mode with sub-grid
glacier-width carving, single-flow routing, and the mode-C routing helpers
(trunk-surface routing and routing relaxation) — also spelled `mode='C'`. Those
choices are covered in {doc}`../guides/configuring_a_run`; plotting and saving
runs in {doc}`../guides/outputs_and_io`.

**One caveat on this grid.** The example's `nx = ny = 81` over `Lx = Ly = 2e4`
gives `dx = 250 m`, and the carve footprint radius `R = α_g·H/2 ≈ 2.5·H` is only
about a cell wide for typical thicknesses — so carving is effectively a sub-grid
no-op here and the run looks like plain mode B. To *see* the carving (real trough
width, channel capture), use a grid with `dx` well below `α_g·H/2` (a few tens of
metres); see {doc}`../guides/configuring_a_run`.

Figures and saved runs are written under `model_outputs/` in the working
directory.
