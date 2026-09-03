# Changelog

Short release notes. Public behavior and configuration are documented in the
guides and API reference under `docs/`.

## 0.9.2 — September 2026 (current)

- **Mode B/C: the kernel now erodes the post-uplift bed.** The in-house driver
  and the Fastscape adapter both handed the mode-B/C kernel the pre-uplift bed
  and composed uplift afterwards. `fixed_value` border cells never uplift, so
  the first interior row equilibrated against the pinned outlet and was then
  lifted by `U*dt` every step — a permanent `U*dt` lip along every base-level
  border that raised the whole landscape by that amount (300 m at
  `dt = 300 kyr`). Mode A, the 1D model, and the kernel's documented contract
  already used the post-uplift bed. Mode-B/C results change by `U*dt` near
  borders (A is bit-identical); driver/adapter parity stays bit-for-bit, and
  the B/C reference battery was re-frozen on the CI capture environment.
- **Nine ice-display fixes from a mode-C visualization audit.** Lakes flood
  the true composite surface, so the cross-section matches the map (the
  `lakes` field was previously unreachable); `ice_smoothing='field'` with
  `H_threshold <= 0` raises instead of silently drawing nothing;
  `animate_landscape` freezes `z_max`/`H_max` over the run and one NaN no
  longer blanks the map; `ice_time_avg` feeds only the ice mask and depth
  colour in both extents (the footprint extent no longer time-averages the
  terrain/section); auto `z_max` covers the unsmoothed section profile;
  `_clean_ice_mask` is seam-aware on looped axes; the raster extent registers
  with the contour/trimline/section coordinates; the section bed line is
  bilinear to match the ice surface; bare-bed views are labelled "Bedrock
  elevation". Each fix carries a regression test.
- **Plateau initial surface: edges on their datums + a public builder.** The
  arctan plateau (`siim_escarpment(init_type='plateau')`, the fastscape
  `PlateauSurface`) now rescales its ramp so the fixed x-borders start exactly
  on `0` / `plateau_zo - plateau_dz`; the raw arctan only reached them
  asymptotically (with the default `plateau_frac`/`plateau_w` on a 50 km domain
  the low edge sat at `0.25*plateau_zo`, a permanent sill above the border's
  water datum). Behavioral for every plateau run (no reference-battery case
  uses one). New `siim.escarpment.plateau_topography(nx, ny, Lx, zo, frac, w,
  dz)` returns the same surface for plain `siim2d` runs together with the
  per-side base level `{'right': zo - dz}` that puts the plateau's outlet at
  its own edge elevation.
- **Per-side base level (2D).** `bl` now also accepts a dict keyed by side
  (`{'left'|'right'|'bottom'|'top': scalar or length-nt series}`), giving each
  `'fixed_value'` outlet its own water datum; unspecified sides keep
  `constants.BL`, and a datum on a non-outlet side or an unknown key raises.
  Border nodes carry their own side's datum and every interior node inherits
  the datum of the outlet its basin drains to (corners shared by two fixed
  sides take the x-side value). Scalar and series `bl` are unchanged
  bit-for-bit; per-side `bl` requires the in-house driver (the xsimlab adapter
  takes one scalar per step and raises on a dict), and — like scalar `bl` in 2D
  — is consumed by modes B/C only (2D mode A ignores `bl`). 1D rejects a dict.
  See `docs/guides/configuring_a_run.md`.

## 0.9.1 — August 2026

The standalone migration: siim's 2D model no longer requires the frozen
fastscape/xsimlab/fastscapelib-fortran stack.

**Packaging (the headline)**
- The 2D model (`siim.siim2d`, `siim.escarpment`) is now **pip-installable
  standalone**: `pip install siim` runs everything on numpy/scipy/numba/
  matplotlib/tqdm/xarray/pandas (numpy 2 supported; `xarray` and `pandas` are
  direct, unpinned core dependencies). No conda required for default use.
- `siim.fastscape` is demoted to the **optional adapter**
  (`pip install siim[fastscape]`); importing it without Fastscape raises a
  directed error, while missing conda-only runtime backends are reported when
  the corresponding stock process is used. fastscapelib-fortran remains conda-only —
  `environment.yml` is retitled the fastscape-adapter / legacy env and keeps
  its numpy<2 / xarray<2026.5 / zarr<3 quarantine (which no longer constrains
  a default install). A `dev-parity` extra ships for the future fastscapelib
  parity probe (workflow_dispatch-only CI stub pre-1.0). The published
  `environment.yml` is a lean `siim-adapter` environment rather than the full
  paper/notebook development environment.
- Importing the optional adapter no longer monkey-patches private xarray
  internals or installs a process-wide warning filter. SIIM-owned xsimlab
  compatibility warnings are suppressed only around the calls that emit them.
- The release exporter now requires a committed, clean source snapshot and a
  synchronized public target. Its dry run performs the same artifact build,
  archive inspection, clean-environment install, smoke tests, and public-tree
  diff as publication, without changing either working tree.

**Numerics owned in-house (defaults flipped at 0.9.1)**
- Time loop: an in-house framework-free driver (`siim._core.driver`) replaces
  xsimlab orchestration (bit-for-bit identical `ds_out` on the same backend;
  `driver='xsimlab'` remains as the adapter-env escape hatch).
- Flow routing: in-house numba D8 fill-then-route (+ the already-in-house
  D-inf) replaces the fortran SFR. ⚠ **Behavioral change:** routing
  tie-break/depression paths differ from fortran by construction, and siim's
  2D landscapes are multistable under them — same-seed mode-B/C runs
  self-organise equivalent-but-different drainage networks (~100–170 m rms
  apart at identical attractor statistics). Gated behaviorally
  (receiver parity vs frozen raw fortran refs, analytical oracle, and attractor
  statistics). Compare attractors, not
  snapshots, against pre-0.9.1 runs.
- Flexure: in-house scipy.fft plate solve. ⚠ Fixes the fortran `pihy`
  anisotropy bug — on `dx != dy` grids results differ from fortran **by
  design** (validated against the closed-form Kelvin point-load solution);
  the k=0 (domain-mean) mode is zeroed, matching fortran's far-field-neutral
  DST semantics.
- Hillslope diffusion: in-house numba ADI, byte-identical to `fs.diffusion`
  for siim's uniform diffusivity. Its Thomas solver is independently expressed
  from the standard tridiagonal row recurrence.
- The retired `'fortran'` options of `router_backend`/`numerics_backend` now
  raise directed errors; the params survive as the (public-contract) backend
  plug points.

**Bug fix — mode B/C outlet lip (2026-09-02)**
- The 2D mode-B/C kernel now receives the **post-uplift** bed (the kernel's
  documented contract; mode A and the 1D model already did). Both the in-house
  driver and the Fastscape adapter eroded the pre-uplift bed against the pinned
  `fixed_value` borders and composed uplift afterwards, leaving a permanent
  `U*dt` step on the first interior row at every base-level border and raising
  the whole landscape by that amount (300 m at dt = 300 kyr). ⚠ Mode-B/C
  results change by `U*dt` near borders (and negligibly elsewhere via the
  absolute-datum terms); the frozen B/C reference battery needs a sanctioned
  regeneration. Driver/adapter parity remains bit-for-bit.

**API changes (pre-release, no deprecation cycle)**
- Default `boundary_status` is now the explicit
  `['fixed_value','fixed_value','looped','looped']`, and `'core'` means plain
  non-periodic interior for **every** router (previously fortran-SFR silently
  treated top/bottom `'core'` as y-cyclic while D-inf did not — the
  split-brain is gone; ratified OQ-1(b)).
- `run(hooks=...)` is dropped on the standalone driver (raises a directed
  error); xsimlab RuntimeHooks remain available via `driver='xsimlab'`.
- `nt_out` is validated to `1 <= nt_out <= nt` (previously `nt_out > nt`
  silently produced NaN frames under xsimlab).
- An explicit analytical `lam` override now consistently sets
  `kappa_c = 1/(1-lam)` in both `GeneralProfile` and `RegimeMap`; the default
  exponent-derived closure is unchanged.
- An explicit `mu` with an exact power or Coulomb numerical law is retained for
  the analytical/reporting interpretation but now warns that the exact kernel
  uses its law-derived exponent. The effective-exponent law continues to use
  the override numerically.
- Saved-run pickles now carry an explicit format/schema version, producing SIIM
  version, concrete model identity, the original parameter dictionary, and the
  xarray Dataset. Unversioned saves are rejected with a directed error rather
  than silently interpreted under current defaults.
- The standalone and adapter drivers now consume one shared output schema, and
  construction of the numerical law record is keyword-only to make its
  cross-module parameter contract resistant to accidental reordering.

**Verification (per stage, recorded in the plan)**
- Extraction + driver: bit-for-bit against the frozen reference battery and
  xsimlab at every output frame. Flexure: Kelvin oracle relRMS ~2e-5 (square)
  / 1.5e-4 (anisotropic). Diffusion: byte-identical twin. Router: 961/961
  byte-exact receivers on tie-free surfaces, exact fixed-point
  router-invariance, attractor means within 3.1%. The reference battery is
  re-frozen from the standalone defaults and remains the standing
  bit-for-bit determinism tripwire (per-platform); the raw fortran router
  refs remain immutable parity baselines.

## 0.9.0 — July 2026

The pre-release consolidation: everything below is on `main`, suite 376,
docs `-W` green, dev-CI green. v1.0 is reserved for the public release.

**Physics & boundary conditions**
- Base-level BC redesign (mode B): water/rock separation with a time-dependent
  Dirichlet water datum `bl(t)`; true-state outputs everywhere (see
  `docs/guides/outputs_and_io.md`).
- Ice borders are OUTFLOW boundaries: zero-gradient thickness (dominant donor
  in 2D) with the bed evolved by the implicit closed-form border budget on the
  arrival slope — dt-robust at any dt, bounded at the flotation draft.
- Waterline-flotation gate as an effective-pressure ramp
  (`flotation_gate`/`flotation_ramp`, γ = 0.1 default), interior + border.
- Channel-floor datum `hc/H̄ = 1.5` implemented across the numerical models and
  the analytical bed reconstructions (see `docs/guides/concepts.md`).

**Mode C (the flagship default: mode B + sub-grid width carving)**
- Sub-grid glacier-width carving via exact power-diagram attribution
  (see `docs/guides/configuring_a_run.md`); `widening_rate` default 3.0.
- Mode-C standard: `trunk_surface=True`, `routing_relax=0.5` by default
  (anti-flicker EMA); the flux-consolidation
  machinery removed.
- D-inf routing rebuilt on the eps-filled surface (fill-based redesign) plus
  D-inf mode B.

**Performance**
- `parallel_erode` (default ON): level-scheduled parallel mode-B erosion,
  bit-for-bit with the serial eroder — coulomb steps −43% (SFR) / −32% (D-inf)
  at 201×201.
- D-inf facet scan row-parallelized; allocation-free multi-receiver Newton.
  D-inf mode-C steps −19% (power) / −10% (coulomb) on top of the above.

**Infrastructure**
- Pre-1.0 code audit complete (91 findings adjudicated).
- Sphinx docs site (Furo + MyST-NB), `-W` clean; RTD config at the root.
- Frugal dev-CI (cached micromamba: suite + docs build per PR/push).
- Packaging: conda-only for the 2D stack (`environment.yml` is the contract,
  with the numpy<2 / zarr<3 / xarray-freeze quarantine around the unmaintained
  xsimlab); the lightweight core (1D + analytical) declares pip deps.

## 0.5.0 and earlier — 2026 (unreleased development)

Internal development: the 1D profile model, the 2D fastscape/xsimlab model,
the analytical package (steady-state profiles, closure solver, regime map),
the numpy/numba numerical core, and the law_code kernel rewrite.
