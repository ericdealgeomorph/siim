# Installation

siim is a research package, installed from a clone. Since 0.9.1 the whole
model — 1D, **2D included**, analytical, plotting — is **pip-installable**: it
runs on numpy (2.x supported), scipy, numba, matplotlib, tqdm, and xarray,
with the time loop, flow routing, flexure, and hillslope diffusion all
implemented in-house. No conda, no Fortran.

## Standard install (everything, incl. the 2D model)

```bash
git clone <repository> siim
cd siim
pip install -e .          # editable; pulls the core deps (numpy, scipy, numba,
                          # matplotlib, tqdm, xarray) if not already present
```

That is the whole install:

```python
from siim.siim2d import siim as siim2d   # no fastscape/xsimlab needed
```

## Optional: the fastscape adapter (conda)

`siim.fastscape` exposes siim's glacial physics as composable
fastscape/xsimlab `@xs.process` classes, and `run(driver='xsimlab')` drives
the 2D model through xsimlab orchestration. This is **optional** — the
standalone 2D model needs none of it — and it is the one component that still
wants conda: `pip install -e .[fastscape]` can deliver `fastscape` and
`xarray-simlab` from PyPI (note PyPI's fastscape is a stale 0.1.0), but
**fastscapelib-fortran** — the Fortran backend fastscape's stock processes
call at run time — has no PyPI wheel. The validated adapter environment is
`environment.yml` (the fastscape-adapter / legacy env, which carries the
`numpy<2` / `xarray<2026.5` / `zarr<3` quarantine pins around the
unmaintained xsimlab):

```bash
conda env create -f environment.yml
conda activate glacial
pip install -e . --no-deps    # pure link — never lets pip touch the pinned env
```

Inside the managed conda environment, always install with `--no-deps` so pip
cannot upgrade the quarantined packages. Without the stack installed,
`import siim.fastscape` raises a directed `ImportError` explaining exactly
this.
