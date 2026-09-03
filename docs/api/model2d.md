# 2D landscape model

The constructor accepts a parameter dictionary; see
{doc}`../guides/parameter_reference` for every key and mode-dependent default.

## Wrapper / dispatch — `siim.siim2d`

```{eval-rst}
.. automodule:: siim.siim2d

.. autoclass:: siim.siim2d.siim
   :members: run, save, load, extract_channel, strahler_order
   :show-inheritance:

.. autofunction:: siim.siim2d.load
```

The standalone driver uses the fastscape-free numerical core for routing,
ice-thickness closure, erosion, sub-grid carving, diffusion, and flexure; see
{doc}`core`. Optional xsimlab process shells around the same step functions are
documented in {doc}`fastscape`.

## Plotting — `siim.plotting`

```{eval-rst}
.. automodule:: siim.plotting
   :members:
   :show-inheritance:
```

### Raw 2D fields — `siim.plotting.maps`

```{eval-rst}
.. automodule:: siim.plotting.maps
   :members:
   :show-inheritance:
```

### Channel profiles & cross-sections — `siim.plotting.profiles`

```{eval-rst}
.. automodule:: siim.plotting.profiles
   :members:
   :show-inheritance:
```

### Landscape rendering — `siim.plotting.landscape`

```{eval-rst}
.. automodule:: siim.plotting.landscape
   :members:
   :show-inheritance:
```

### Basin & channel diagnostics — `siim.plotting.basins`

```{eval-rst}
.. automodule:: siim.plotting.basins
   :members:
   :show-inheritance:
```

### Steady-state diagnostics — `siim.plotting.diagnostics`

```{eval-rst}
.. automodule:: siim.plotting.diagnostics
   :members:
   :show-inheritance:
```

## Escarpment variant — `siim.escarpment`

```{eval-rst}
.. automodule:: siim.escarpment
   :members:
   :show-inheritance:
```
