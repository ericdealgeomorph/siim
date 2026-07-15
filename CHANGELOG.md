# Changelog

Short release notes; the citable design records behind each item live in
`docs/` and `docs/dev/`.

## 0.9.0 — July 2026 (current)

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
