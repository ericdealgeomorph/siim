# Numerical core — `siim._core`

The fastscape-free numpy/numba kernels shared by the 1D profile model and the
2D landscape model. Importing `siim._core` (and its submodules) needs
numpy/numba only — no fastscape, xsimlab, or matplotlib — so the numerics stay
testable without the model stack. The 1D and 2D models build one
`GlacialParams` record and an integer `law_code` and call the step skeletons
directly.

## Per-run parameter record — `siim._core.params`

```{eval-rst}
.. automodule:: siim._core.params
   :members:
   :show-inheritance:
```

## Step skeletons (law_code switch) — `siim._core.skeleton`

```{eval-rst}
.. automodule:: siim._core.skeleton
   :members:
   :private-members:
```

## Scalar Newton solvers & ice-thickness closures — `siim._core.solvers`

```{eval-rst}
.. automodule:: siim._core.solvers
   :members:
   :private-members:
```

## 2D erosion loops — `siim._core.eroders`

```{eval-rst}
.. automodule:: siim._core.eroders
   :members:
   :private-members:
```

## Sub-grid glacier-width carving — `siim._core.carve`

```{eval-rst}
.. automodule:: siim._core.carve
   :members:
   :private-members:
```

## Explicit H diffusion — `siim._core.diffusion`

```{eval-rst}
.. automodule:: siim._core.diffusion
   :members:
   :private-members:
```

## Flow routing & accumulation primitives — `siim._core.routing`

```{eval-rst}
.. automodule:: siim._core.routing
   :members:
   :private-members:
```
