# 2D landscape model

## Wrapper / dispatch — `siim.siim2d`

```{eval-rst}
.. automodule:: siim.siim2d
   :members:
   :show-inheritance:
```

The glacial xsimlab processes this model assembles (including `DinfFlowRouter`)
are documented in {doc}`fastscape`; the numba kernels they dispatch into —
ice-thickness closures, erosion loops, sub-grid carving, H diffusion, and the
D-infinity routing primitives — live in the fastscape-free numerical core, see
{doc}`core`.

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

### Deluxe landscape — `siim.plotting.landscape`

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
