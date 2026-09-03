# Installation

The standard installation includes the 1D model, the full standalone 2D model,
the analytical solutions, and plotting. The time loop, flow routing, flexure,
and hillslope diffusion are implemented in the package; this path needs neither
conda nor Fortran.

## Standard install

Install the released package from PyPI:

```bash
python -m pip install siim-lem
```

Or install an editable checkout for development:

```bash
git clone https://github.com/ericdealgeomorph/siim.git
cd siim
python -m pip install -e .
```

Both install NumPy, SciPy, Numba, Matplotlib, tqdm, xarray, and pandas
automatically and support NumPy 2. The model imports directly after
installation:

```python
from siim.siim2d import siim as siim2d   # no fastscape/xsimlab needed
```

Initial topography can be supplied as a NumPy array, pandas DataFrame, or CSV
file without installing an additional Python package.

Exporting MP4 animations also requires a system `ffmpeg` executable on `PATH`.
It is not needed for simulations, static plots, or interactive viewers.

## Numba threading

On import, SIIM uses `os.environ.setdefault` to select Numba's `workqueue`
threading layer when `NUMBA_THREADING_LAYER` is not already set. This
process-wide default avoids a known macOS interaction between alternate
threading runtimes and the NumPy/SciPy stack. To choose another installed Numba
threading layer, set the environment variable **before** importing SIIM; an
explicit user value is never overwritten.

## Optional: the fastscape adapter (conda)

`siim.fastscape` exposes siim's glacial physics as composable
fastscape/xsimlab `@xs.process` classes, and `run(driver='xsimlab')` drives
the 2D model through xsimlab orchestration. This is **optional** — the
standalone 2D model needs none of it — and it is the one component that still
requires conda. `python -m pip install 'siim-lem[fastscape]'` can deliver `fastscape` and
`xarray-simlab` from PyPI (PyPI's fastscape is a stale 0.1.0), but
**fastscapelib-fortran** — the Fortran backend fastscape's stock processes
call at run time — has no PyPI wheel. The validated adapter environment is
`environment.yml`, which carries the `numpy<2` / `xarray<2026.5` / `zarr<3`
quarantine pins around the unmaintained xsimlab:

```bash
conda env create -f environment.yml
conda activate siim-adapter
python -m pip install -e . --no-deps  # do not alter the pinned adapter env
```

Inside the managed conda environment, always install with `--no-deps` so pip
cannot upgrade the quarantined packages. If `fastscape` or `xarray-simlab`
itself is absent, `import siim.fastscape` raises a directed `ImportError`.
Missing `fastscapelib-fortran` may appear later, when a stock Fastscape process
first calls its backend; the conda environment supplies all three components.
