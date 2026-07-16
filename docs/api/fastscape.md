# fastscape integration — `siim.fastscape`

The public, composable surface for using siim's glacial-erosion physics inside
your own fastscape/xsimlab model: the assembly helpers (`glacial_processes`,
`glacial_model`) and the `@xs.process` classes they wire in. This is the
**optional adapter** (`pip install siim[fastscape]`; conda for the Fortran
backend) — the standalone `siim.siim2d` model runs its own in-house driver and
does not require it, though `siim2d(...).run(driver='xsimlab')` will drive the
model through this adapter when the stack is present.

## Public surface — `siim.fastscape`

```{eval-rst}
.. automodule:: siim.fastscape

.. autofunction:: siim.fastscape.glacial_processes

.. autofunction:: siim.fastscape.glacial_model
```

## Glacial processes — `siim.fastscape.processes`

```{eval-rst}
.. automodule:: siim.fastscape.processes
   :members:
   :show-inheritance:
```

## Forcing processes — `siim.fastscape.forcing`

```{eval-rst}
.. automodule:: siim.fastscape.forcing
   :members:
   :show-inheritance:
```
