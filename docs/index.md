# siim — Sliding Ice Incision Model

Coupled glacial–fluvial landscape evolution. siim couples a sliding-ice
incision law to fluvial stream-power erosion on a common grid, so one model
grows mountains, carves glacial troughs, and relaxes them fluvially as the
climate forcing moves the equilibrium-line altitude (ELA) up and down.

Three front ends share one set of physics and constants (`siim.constants`):

- **1D profile model** (`siim.siim1d`) — a single glacial–fluvial long
  profile; fast and fully standalone.
- **2D landscape model** (`siim.siim2d`) — a raster model (single-flow or
  D-infinity routing, sub-grid glacier-width carving), with a rifted-margin
  escarpment variant in `siim.escarpment`. Its default *mode-C standard*
  combines bed memory, width carving, and trunk-surface routing. Since 0.9.1
  it runs standalone on a plain pip install (in-house time loop, routing,
  flexure, and diffusion).
- **Analytical steady state** (`siim.analytical`) — one closure engine behind
  three front ends (`GeneralProfile`, `MarginalCoulombProfile`,
  `SteadyStateProfile`), plus a nondimensional regime map
  (`siim.analytical.regime`) for the bistability boundary. Both numerical
  models embed it as their reference. numpy/scipy only.

siim's glacial physics is also available as an **optional** fastscape adapter
(`siim.fastscape`), for composing the physics into your own fastscape/xsimlab
model — see {doc}`guides/fastscape_processes`. It needs the fastscape stack
(`pip install siim-lem[fastscape]`, conda for the Fortran backend); the standalone
2D model above does not.

The model and theory papers document the governing equations. The current code
uses the channel-floor convention `zs = zb + 1.5*H`; manuscript copies that
still show `zs = zb + H` predate that convention and should not be used to infer
the implemented datum. Where a docstring cites a paper equation, it uses its
LaTeX `\label`, not a renumberable equation number.

**New here?** Install siim ({doc}`getting_started/install`), then run the 1D
walkthrough ({doc}`getting_started/first_1d_run`). {doc}`guides/concepts` gives
the mental model (modes, the channel-floor datum, sub-grid width); the
parameters that set the regime are explained in {doc}`guides/configuring_a_run`,
with every numerical-model key in {doc}`guides/parameter_reference`.

```{toctree}
:maxdepth: 2
:caption: Get started

getting_started/install
getting_started/first_1d_run
getting_started/first_2d_run
```

```{toctree}
:maxdepth: 2
:caption: Guides

guides/concepts
guides/configuring_a_run
guides/parameter_reference
guides/outputs_and_io
guides/fastscape_processes
```

```{toctree}
:maxdepth: 2
:caption: API reference

api/index
```

```{toctree}
:maxdepth: 1
:caption: Bibliography

references
```
