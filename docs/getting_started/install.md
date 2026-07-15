# Installation

siim is a research package, installed from a clone. Its dependencies split by
component, which decides how you install it:

- **Lightweight core** — `import siim`, the 1D model (`siim.siim1d`), and the
  analytical package (`siim.analytical`) need only numpy, scipy, numba, and
  matplotlib (`siim.analytical`: numpy/scipy alone). These install cleanly with
  pip.
- **2D stack** — the 2D model (`siim.siim2d`) and the fastscape process layer
  (`siim.fastscape`) additionally need xsimlab, fastscape, and
  **fastscapelib-fortran**. The last compiles from Fortran and is most easily
  obtained from conda-forge, so a conda environment is the path of least
  resistance for anything 2D.

The heavy 2D dependencies load lazily — only when you import `siim.siim2d` or
`siim.fastscape` — so the lightweight components run without them.

## From a clone

**Lightweight core (1D model + analytical), any environment:**

```bash
git clone <repository> siim
cd siim
pip install -e .          # editable; pulls the core deps (numpy, scipy, numba,
                          # matplotlib, tqdm) if they are not already present
```

**Full install including the 2D model — use the conda environment.** The 2D
stack (xsimlab, fastscape, fastscapelib-fortran) is conda-only:
`fastscapelib-fortran` compiles from Fortran and has no PyPI distribution, and
the environment carries deliberate pins (`numpy<2`, `zarr<3`, an xarray freeze)
that quarantine the unmaintained xsimlab framework. `environment.yml` is the
canonical 2D dependency contract:

```bash
conda env create -f environment.yml
conda activate glacial
pip install -e . --no-deps    # pure link — never lets pip touch the pinned env
```

Inside the managed conda environment, always install with `--no-deps` so pip
cannot upgrade the quarantined packages.
