# Changelog

Short release notes. Public behavior and configuration are documented in the
guides and API reference under `docs/`.

## 0.9.7 — September 2026 (current)

### Behavioural changes

- **Flexure now solves in a sine basis** (`w = 0` at the padded-box edges,
  the basis fastscape's `flexure2D.f90` uses) instead of a periodic FFT with the
  domain-mean mode zeroed. Every `flexure=True` run changes. On small domains
  (`L` much less than the flexural parameter, siim's valley case) the difference
  is ~1e-2 m: a uniform load stays rigidity-supported as before. On domains much
  larger than the flexural parameter the old solve subtracted the Airy mean of a
  localized load and re-emitted it as a uniform uplift of the entire domain —
  on a 2500 x 250 km grid with `e_thickness=20e3` and a migrating uplift wave
  that was +3.5 m per 100 kyr step, raising an untouched far plateau from 290 m
  to 1460 m over 50 Myr. Localized loads now compensate locally at Airy and the
  far field stays put. The transform is scipy's type-2 sine transform with the
  per-run transfer grid cached: same fidelity as type 1 and as the fortran
  (measured), 2-3.5x faster per solve and faster than the fortran itself; it
  moves flexure numbers at the 1e-3 relative level, so the flexure-on reference
  battery regenerates once on Linux CI after this release.

- **`fixed_value` borders no longer receive flexural rebound.** Block uplift was
  already masked off on those rows and columns and they do not erode, but the
  plate solve still handed them their share of the subsidence, so a fixed edge
  beside an uplifting interior sank into a trench that then acted as the base
  level for everything draining to it (on a 2500 x 250 km run the fixed row fell
  287 m to 55 m while the interior rose 292 m to 543 m over 10 Myr). Stock
  fastscape has the same inconsistency between `BlockUplift` and `Flexure`.
  Affects `flexure=True` runs with at least one `fixed_value` edge, including
  the default topology's two fixed columns; all-looped and all-core domains are
  unchanged.

- Landscape movies render serially when system RAM cannot be measured, keeping
  the memory guard effective on platforms without usable `sysconf` results.
  Interactive viewers now report unexpected canvas-setting errors instead of
  silently continuing; failed figures are closed and the notebook's plotting
  backend is restored.

- The 1D `plot.limit_cycle` and `plot.limit_cycle_phase` research helpers are
  now private (`_limit_cycle`, `_limit_cycle_phase`); their drawing behavior is
  retained. Public basin and steady-state summaries support `plot=False` and
  are quiet by default; use `verbose=True` for printed Hack's-law, basin-history
  or steady-state summaries. Basin histories use `ref` consistently with
  profiles; `i_ref` remains a deprecated keyword alias. See the
  [plotting guide](docs/guides/outputs_and_io.md#basin-and-steady-state-summaries).

### Other

- Plotting review fixes: static profiles preserve existing artists on supplied
  axes, and 1D profiles/viewers/movies retain their original coordinate
  direction. Landscape component cleanup now runs once on the combined ice
  layers; ribbon-source thresholds preserve the veil's field-smoothing input.

## 0.9.6 — September 2026

- **2D base-level warning moved to its point of use.** A nonzero or per-side
  `bl` no longer warns at model construction; the note that the analytical
  steady-state reference stays at the datum is emitted when the analytical
  overlay is actually drawn (`profile`/`analytical` plots). Batch runs that
  never look at the overlay are quiet. 1D is unchanged.
- `siim2d.load` accepts a name with a subfolder (`'batch/run003'`), matching
  what `save` writes.

## 0.9.5 — September 2026

- **Documentation overhaul.** The public docs were rewritten as plain
  reference documentation (independently reviewed, and verified line-by-line
  against the code): interpretive discussion removed, the `first_1d_run`
  example re-parameterized (`zELA` 700 → 1400) so the profile stays above the
  base-level datum with the glacier confined to the upper reach, the
  coarse-grid mode-C/mode-B comparison stated precisely (the mode-C routing
  helpers never self-gate on grid resolution), `trunk_alpha` documented as an
  opacity floor, the fastscape `carve` sentinel documented, and the
  `siim._core` API index completed.
- **Fixed the adapter's install hint.** The `ImportError` raised by
  `import siim.fastscape` without the fastscape stack still recommended
  `pip install siim[fastscape]`; it now names the actual distribution,
  `pip install siim-lem[fastscape]`. No behavior changes.

## 0.9.4 — September 2026

- **PyPI distribution name: `siim-lem`.** PyPI's name-similarity guard refused
  the bare name `siim`, so the package installs as `pip install siim-lem`
  (the `[fastscape]` extra rides along unchanged). The import name is
  untouched: `import siim`. Nothing else changes from 0.9.3.

## 0.9.3 — September 2026

- **2D ice rendering: ice is on by default, and depth-graded.** `landscape` /
  `animate_landscape` under `style='smooth'` now resolve `field` to
  `'bedrock+ice'` (it resolved to bare `'bedrock'`, so the default render hid
  the ice and every ice keyword was a silent no-op; `field='bedrock'` is
  unchanged and still reachable). Ice without an explicit `ice_cmap` is painted
  by the new `ice_shading` knob: `'veil'` (smooth) alpha-blends a glacier ramp
  over the terrain by column depth, normalised on the RUN-GLOBAL
  `1.5 * max H_out` so movie frames are comparable, and draws its own
  colorbar; `'flat'` (raw) keeps the single opaque `ice_color`, so `style='raw'`
  is unchanged. Labels now distinguish the two thickness quantities: the
  landscape bar and section band show the local **ice column depth**, while
  `map(field='ice')` and the 1D/2D profile panels show the width-**mean**
  `H`. `show_trimline` / `trimline_color` / `trimline_lw` are renamed
  `show_margin` / `margin_color` / `margin_lw` (the outline is the current
  margin, not a trimline) with deprecated aliases, and the margin defaults are
  heavier and darker. `map(field='ice')` masks ice-free ground to a neutral
  bare tone, sizes its figure from the domain aspect and stamps the output
  time; the section names its ELA and states the vertical exaggeration; and the
  two-colorbar/hypsometry layout survives narrow `fig_width`. The smooth
  preset's `H_threshold` default drops from `100` to `0`, so **nothing is
  hidden by thickness under either style** — the veil already fades thin ice
  out, and the gate only deleted real glacierets. `H_threshold` remains fully
  functional as an explicit crop. On top of that veil the smooth preset now
  draws the RESOLVED TRUNK GLACIERS as true-width ribbons (`trunk_display`,
  `'ribbons'` under smooth and `'none'` under raw): a trunk cell is any icy
  cell downstream of one whose claimed width `W = alpha_g*H` already spans
  `trunk_width_cells` grid cells (default `1.0`), the class carried along the
  receivers to the terminus, and those cells are traced and rasterized at
  their true width — parabolic column depth on the same ramp, their own flat
  ice surface in the hillshade and the cross-section — at `trunk_alpha`
  opacity, seam-aware on looped axes. The veil is then built from the
  sub-resolution ice that is left, and `show_margin` outlines the ribbons
  only. The veil's opacity ramp changes from `0.18 + 0.82*sqrt(t/0.40)` to the
  linear `0.12 + 0.88*t/0.35`: the old front-loading made a 30 m column
  half-opaque, so an apron of sub-cell ice read as one sheet with the trunks
  barely darker streaks through it (trunk-vs-apron RGB distance 0.13, against
  0.67 for apron-vs-bare). `trunk_display='none'` renders exactly as before.

## 0.9.2 — September 2026

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
