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

On import, SIIM sets Numba's threading-layer priority
(`NUMBA_THREADING_LAYER_PRIORITY="tbb workqueue omp"`, through
`os.environ.setdefault`): Numba runs its TBB layer when it can load the TBB
library, and its own `workqueue` layer otherwise. Results are bit-for-bit the
same on either layer. TBB is faster: the level-scheduled eroder
(`parallel_erode`) opens one parallel region per drainage level, and TBB starts
each one several times more cheaply. Under MFD routing, mode D's default, the
parallel eroder is on by default only under TBB (see `parallel_erode` in {doc}`../guides/parameter_reference`). On
Apple silicon the eroder's per-level loops use as many threads as the Mac has
performance cores; every other parallel kernel uses Numba's full thread count.
On a 4 + 6 core laptop with TBB, a 101 x 101 grid steps 1.3-1.9x faster in
modes B and C and 1.1-1.4x faster in mode D than release 0.9.10 did.

The conda environment (`environment.yml`) installs TBB. With pip,
`python -m pip install 'siim-lem[tbb]'` adds it where PyPI has a `tbb` wheel
(x86_64 Linux, Windows, Intel Macs); on Apple silicon the extra installs
nothing and SIIM runs on `workqueue`. In a Linux virtual environment the
extra alone is not enough: pip puts `libtbb` in the environment's `lib`
directory, which the system loader does not search, so Numba does not find it
(tested with Numba 0.68). Point the loader there before starting Python:

```bash
export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

Numba falls back to `workqueue` silently when it cannot load the library;
`numba.threading_layer()` names the layer in use once a model has run, and
`python -m numba -s` reports whether TBB is available. Windows and Intel Macs
are untested.

To choose a layer yourself, set `NUMBA_THREADING_LAYER` (or your own
`NUMBA_THREADING_LAYER_PRIORITY`) **before** importing SIIM; an explicit user
value is never overwritten. `NUMBA_NUM_THREADS` and `numba.set_num_threads`
cap every kernel, the eroder included.

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
