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

## Hillslope diffusion (ADI) — `siim._core.hillslope`

In-house alternating-direction-implicit hillslope diffuser (topography), the
standalone replacement for fastscapelib-fortran's `fs.diffusion`. Distinct from
`siim._core.diffusion` (the ice-thickness FD diffuser above).

```{eval-rst}
.. automodule:: siim._core.hillslope
   :members:
   :private-members:
```

## Spectral flexure — `siim._core.flexure`

In-house `scipy.fft` thin-plate flexure solve on the native grid, the standalone
replacement for fastscapelib-fortran's `fs.flexure`.

```{eval-rst}
.. automodule:: siim._core.flexure
   :members:
   :private-members:
```

## Flow routing & accumulation primitives — `siim._core.routing`

```{eval-rst}
.. automodule:: siim._core.routing
   :members:
   :private-members:
```

## Composition-chain step functions — `siim._core.step`

Framework-free extractions of the 2D model's per-step composition chain (one
`self`-free function per `siim.fastscape` process `run_step` body). The xsimlab
adapter shells and the standalone driver call the same functions.

```{eval-rst}
.. automodule:: siim._core.step
   :members:
   :private-members:
```

## In-house time loop — `siim._core.driver`

The standalone driver: siim's own merged step loop + two-cadence snapshot,
calling the same step functions the adapter shells call. Selected via
`run(driver='inhouse')` (default `constants.DRIVER_DEFAULT`).

```{eval-rst}
.. automodule:: siim._core.driver
   :members:
   :private-members:
```

## Output packing — `siim._core.outputs`

Packs the driver's step buffers into the exact `ds_out` contract
(variable names, `(time, y, x)` dims, dtypes, coords) that
`siim.siim2d.siim._unpack_outputs` reads.

```{eval-rst}
.. automodule:: siim._core.outputs
   :members:
   :private-members:
```
