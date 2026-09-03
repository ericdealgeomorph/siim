---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: Python 3
  name: python3
---

# Your first 1D run

The 1D model (`siim.siim1d`) evolves a single glacial–fluvial long profile. It
is fast and dependency-light (numpy / scipy / numba / matplotlib). This page is
executed when the docs are built.

Construct a model from a parameter dict and run it:

```{code-cell} python
%matplotlib inline
from siim.siim1d import siim as siim1d

m = siim1d({'zELA': 1400, 'sliding_law': 'eff-exp', 'progress_bar': False})
m.run()
```

`zELA` sets the equilibrium-line altitude — the climate forcing — and
`sliding_law` selects the ice-flow law (`'eff-exp'`, `'power'`, or
`'coulomb'`). The model runs in its default `'bedrock+ice_thickness'` mode.

Plot the long profile:

```{code-cell} python
fig, axes = m.plot.profile()
fig
```

The plot shows the final frame: the bed, the ice surface, the ELA, and the
analytical steady-state reference the model embeds. A glacier occupies the
upper profile; below the terminus the profile is fluvial, graded to base level
at the outlet, and water fills closed overdeepenings carved into the bed.

The workflow is: construct from a dict, `run()`, then read the result off
`m.plot` or the output arrays. The 1D model keeps its run in memory and does not
provide the 2D model's pickle-based `save`/`load` helper. Plot methods return
Matplotlib figures; methods with a `save` option write requested files beneath
`model_outputs/` in the working directory.

Next: the parameters that set the regime are explained in
{doc}`../guides/configuring_a_run`, and every accepted key is listed in
{doc}`../guides/parameter_reference`; the 2D landscape model is in
{doc}`first_2d_run`.
