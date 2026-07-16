# Changelog

Short release notes; the citable design records behind each item live in
`docs/` and `docs/dev/`.

## 0.9.1 — July 2026 (current)

The standalone migration (`docs/dev/standalone_migration_plan.md`, stages
S0–S5): siim's 2D model no longer requires the frozen fastscape/xsimlab/
fastscapelib-fortran stack.

**Packaging (the headline)**
- The 2D model (`siim.siim2d`, `siim.escarpment`) is now **pip-installable
  standalone**: `pip install siim` runs everything on numpy/scipy/numba/
  matplotlib/tqdm/xarray (numpy 2 supported; `xarray` added to core deps,
  unpinned). No conda required for default use.
- `siim.fastscape` is demoted to the **optional adapter**
  (`pip install siim[fastscape]`); its modules raise a directed ImportError
  without the stack. fastscapelib-fortran remains conda-only —
  `environment.yml` is retitled the fastscape-adapter / legacy env and keeps
  its numpy<2 / xarray<2026.5 / zarr<3 quarantine (which no longer constrains
  a default install). A `dev-parity` extra ships for the future fastscapelib
  parity probe (workflow_dispatch-only CI stub pre-1.0).

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
  (receiver parity vs frozen raw fortran refs, analytical oracle, attractor
  statistics; `docs/dev/router_contract.md`). Compare attractors, not
  snapshots, against pre-0.9.1 runs.
- Flexure: in-house scipy.fft plate solve. ⚠ Fixes the fortran `pihy`
  anisotropy bug — on `dx != dy` grids results differ from fortran **by
  design** (validated against the closed-form Kelvin point-load solution);
  the k=0 (domain-mean) mode is zeroed, matching fortran's far-field-neutral
  DST semantics.
- Hillslope diffusion: in-house numba ADI, byte-identical to `fs.diffusion`
  for siim's uniform diffusivity.
- The retired `'fortran'` options of `router_backend`/`numerics_backend` now
  raise directed errors; the params survive as the (public-contract) backend
  plug points.

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
- Saved-run pickles keep the same format (`{'_user_params', 'ds_out'}`,
  xarray Dataset, same variable names) — fresh saves round-trip; pre-0.9.1
  saves are not migrated (pre-release posture).

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
  Dirichlet water datum `bl(t)`; true-state outputs everywhere
  (`docs/dev/boundary_conditions.md`).
- Ice borders are OUTFLOW boundaries: zero-gradient thickness (dominant donor
  in 2D) with the bed evolved by the implicit closed-form border budget on the
  arrival slope — dt-robust at any dt, bounded at the flotation draft
  (`docs/dev/outflow_implicit_budget.md`).
- Waterline-flotation gate as an effective-pressure ramp
  (`flotation_gate`/`flotation_ramp`, γ = 0.1 default), interior + border.
- Channel-floor datum `hc/H̄ = 1.5` implemented across the numerical models and
  the analytical bed reconstructions (`docs/hc_convention_notes.md`).

**Mode C (the flagship default: mode B + sub-grid width carving)**
- Sub-grid glacier-width carving via exact power-diagram attribution
  (`docs/subgrid_width_carving.md`); `widening_rate` default 3.0.
- Mode-C standard: `trunk_surface=True`, `routing_relax=0.5` by default
  (anti-flicker EMA; `docs/dev/step_flicker.md`); the flux-consolidation
  machinery removed.
- D-inf routing rebuilt on the eps-filled surface (fill-based redesign,
  `docs/dinf_routing.md`) + D-inf mode B.

**Performance (`docs/dev/perf_audit.md`)**
- `parallel_erode` (default ON): level-scheduled parallel mode-B erosion,
  bit-for-bit with the serial eroder — coulomb steps −43% (SFR) / −32% (D-inf)
  at 201×201.
- D-inf facet scan row-parallelized; allocation-free multi-receiver Newton.
  D-inf mode-C steps −19% (power) / −10% (coulomb) on top of the above.

**Infrastructure**
- Pre-1.0 code audit complete (91 findings adjudicated;
  `docs/dev/audit_findings.md`).
- Sphinx docs site (Furo + MyST-NB), `-W` clean; RTD config at the root.
- Frugal dev-CI (cached micromamba: suite + docs build per PR/push).
- Packaging: conda-only for the 2D stack (`environment.yml` is the contract,
  with the numpy<2 / zarr<3 / xarray-freeze quarantine around the unmaintained
  xsimlab); the lightweight core (1D + analytical) declares pip deps.

## 0.5.0 and earlier — 2026 (unreleased development)

Internal development: the 1D profile model, the 2D fastscape/xsimlab model,
the analytical package (steady-state profiles, closure solver, regime map),
the numpy/numba numerical core, and the law_code kernel rewrite.
