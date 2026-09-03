# siim — Sliding Ice Incision Model

Coupled glacial–fluvial landscape evolution. `siim` models bedrock incision by
sliding ice and rivers together, resolving how glaciers and channels jointly
shape mountain topography over geologic time. It ships three coordinated front
ends over one physical core:

- **`siim.siim1d`** — fast 1D longitudinal-profile model
- **`siim.siim2d`** — full 2D landscape-evolution model (in-house driver, D8/D-inf
  routing, FFT flexure, ADI hillslope diffusion)
- **`siim.analytical`** — closed-form steady-state profiles and regime diagrams
  from the underlying theory

Python with NumPy/SciPy/Numba — no Fortran and no framework lock-in for the
standalone models.

## Install

```bash
pip install siim-lem
```

`siim` runs the full 1D and 2D models standalone on NumPy 2. An optional adapter
lets you drive the 2D model through [fastscape](https://fastscape.org)/xsimlab
instead of the built-in driver; that stack is conda-only:

```bash
conda env create -f environment.yml   # optional fastscape adapter
```

Requires Python ≥ 3.12.

## Quickstart

```python
from siim import siim2d

params = {
    'U': 1e-3,                          # rock uplift rate (m/yr)
    'zELA': 1500, 'P': 2, 'beta': 1e-2, # climate: ELA + mass balance
    'ce': 2e-5, 'Ko': 1e-6,             # glacial + fluvial erodibility
    'Lx': 100e3, 'Ly': 50e3,            # domain size (m)
    'nx': 201, 'ny': 101,               # grid
    'T': 5e6, 'nt': 501, 'nt_out': 51,  # 5 Myr run
    'seed': 0,                           # reproducible initial relief noise
    'mode': 'C',                        # bed-memory + sub-grid glacier carving
}

model = siim2d.siim(params)
model.run()
model.plot.landscape(i=-1)             # final topography
```

The 1D model (`from siim.siim1d import siim`) and the analytical solutions
(`from siim.analytical import GeneralProfile`) follow the same shape. See the
[documentation](https://siim.readthedocs.io) for the full parameter reference,
guides, and the theory background.

## Citing

If you use `siim`, please cite it via [`CITATION.cff`](CITATION.cff). A methods
paper is in preparation.

## License

[MIT](LICENSE) © Eric Deal
