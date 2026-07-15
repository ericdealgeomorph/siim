# fastscape integration — `siim.fastscape`

The public, composable surface for using siim's glacial-erosion physics inside
your own fastscape/xsimlab model: the assembly helpers (`glacial_processes`,
`glacial_model`) and the `@xs.process` classes they wire in. The full coupled
`siim.siim2d` model is assembled from this same surface.

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
