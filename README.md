# siim — Sliding Ice Incision Model

Coupled glacial–fluvial landscape evolution. `siim` models bedrock incision by
sliding ice and rivers together, resolving how glaciers and channels jointly
shape mountain topography over geologic time. It ships three coordinated front
ends over one physical core:

- **`siim.siim1d`** — fast 1D longitudinal-profile model
- **`siim.siim2d`** — full 2D landscape-evolution model, built on the
  [fastscape](https://fastscape.org) / xsimlab framework
- **`siim.analytical`** — closed-form steady-state profiles and regime diagrams
  from the underlying theory

## Install

The 1D model, the analytical solutions, and the plotting layer install from PyPI:

```bash
pip install siim
```

The full **2D** model builds on fastscape/xsimlab and its Fortran backend
(`fastscapelib-fortran`), which are distributed via conda. Create the pinned
environment from the repo and install `siim` into it:

```bash
git clone https://github.com/ericdealgeomorph/siim.git
cd siim
conda env create -f environment.yml   # creates the 'glacial' env (pinned to numpy<2)
conda activate glacial
pip install -e . --no-deps
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
    'mode': 'C',                        # bed-memory + sub-grid glacier carving
}

model = siim2d.siim(params)
model.run()
model.plot.landscape(i=-1)             # final topography
```

(The 2D model above runs in the conda environment.) The 1D model
(`from siim.siim1d import siim`) and the analytical solutions
(`from siim.analytical import GeneralProfile`) run from the pip install alone and
follow the same shape. See the [documentation](https://siim.readthedocs.io) for
the full parameter reference, guides, and the theory background.

## Citing

If you use `siim`, please cite it via [`CITATION.cff`](CITATION.cff). A methods
paper is in preparation.

## License

[MIT](LICENSE) © Eric Deal
