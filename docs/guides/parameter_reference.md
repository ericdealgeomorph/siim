# Numerical-model parameter reference

The 1D and 2D numerical models both accept one dictionary at construction:

```python
from siim.siim1d import siim as siim1d
from siim.siim2d import siim as siim2d

m1 = siim1d({'zELA': 1200, 'progress_bar': False})
m2 = siim2d({'zELA': 1200, 'seed': 0, 'progress_bar': False})
```

Keys are case-sensitive and unknown keys raise `ValueError`. Defaults below are
the public constructor defaults; an arrow shows a sentinel that is resolved
during construction. SI units are used unless stated otherwise: metres, years,
Pascals, and combinations of them appropriate to each coefficient.

The 2D model checks these inputs at construction and raises `ValueError` for a
value outside its range, NaN or infinity included:

- `nt >= 2`, and `T` later than `to`;
- `nx, ny >= 3`, and `Lx, Ly > 0`;
- greater than 0: `beta`, `nu` (or `ell`), `lambda_p`, `lambda_c` and `tau_c`,
  and, in the escarpment variant, `wave_width` and `wave_velocity`;
- 0 or greater: `D`, `flotation_ramp` and `H_diffusivity`;
- with `flexure=True`: `lithos_density` and `asthen_density` greater than 0, and
  `e_thickness` 0 or greater (`0` is Airy isostasy), the last two scalars;
- `coulomb_clamp` strictly between 0 and 1.

This page covers the 1D and 2D numerical wrappers, including the additional
keys accepted by the escarpment variant. The analytical classes expose their
own constructor signatures in {doc}`../api/analytical`.

## Parameters accepted by both models

### Grid, time, and state

| key | 1D default | 2D default | meaning |
|---|---:|---:|---|
| `nx` | `501` | `201` | Number of nodes in x. In 1D, `dx` overrides it. A CSV/DataFrame initial condition overrides it in either model. |
| `to` | `0` | `0` | Initial model time (yr). |
| `T` | `3e6` | `3e6` | Final model time (yr); the clock is `linspace(to, T, nt)`. |
| `nt` | `3000` | `2501` | Number of master time points, including both endpoints. |
| `nt_out` | `101` | `101` | Requested saved frames. 1D also accepts `None` for every step. In 2D it must satisfy `1 <= nt_out <= nt`. |
| `mode` | `'bedrock+ice_thickness'` | `'C'` | State convention. `'A'`/`'ice_surface'` tracks the surface; `'B'`/`'bedrock+ice_thickness'` tracks bed and ice thickness. 2D `'C'` resolves to B plus width carving. 2D `'D'` is C with the valley ↔ sheet reading choosing, per cell and per step, the valley or the sheet treatment of the five valley devices (see {doc}`configuring_a_run`); it sets its own defaults for `inner_damping` (`'adaptive'`; under `'omega'`, `inner_relax` 1/3 over `inner_passes` 1), `width_hack_k` (0.25), `routing_filter_factor` (2), `flow_routing` (`'mfd'`, and with it the layer-dependent `parallel_erode` default; the only routing mode D accepts) and `trunk_surface` (`False`) wherever you leave them unset (everything else resolves as in mode C), needs the standalone driver, square cells, a finite `widening_rate` and every looped axis at least 17 cells long, and warns on cells coarser than 2.5 km. Its one dial of the reading is `channel_bed_bar`. C and D are rejected in 1D. |
| `H_diffusivity` | `None` | `None` | Ice-thickness diffusivity (m²/yr); `None` or `0` disables it, and negative values are rejected. It works in both 1D modes but only B/C in 2D. |
| `initial_topography` | `None` | `None` | NumPy array, pandas DataFrame, or CSV path. Array shape is `(nx,)` in 1D or `(ny, nx)` in 2D. The 2D initializer still adds `noise_amplitude`; set it to zero for the exact supplied elevations. See {ref}`initial-topography-formats`. |
| `progress_bar` | `True` | `True` | Show tqdm run progress. |

### Climate, uplift, and base level

| key | default | meaning |
|---|---:|---|
| `P` | `1` | Precipitation/runoff rate (m/yr), as a scalar or length-`nt` time series. For a series, its mean supplies the analytical reference. |
| `beta` | `1e-2` | Linear mass-balance gradient (yr⁻¹). Must be finite and greater than 0, in both models. |
| `zELA` | `None` | Equilibrium-line altitude (m), scalar or length-`nt` series. `None` means 1500 m, unless `zT` is given (next row). The minimum of a series supplies the analytical maximum-glaciation reference. 2D also takes an `(ny, nx)` field or an `(nt, ny, nx)` field series (the field series needs the in-house driver); any other shape, or a non-finite value, raises. A field's spatial mean stands in for its scalar value: the analytical reference and `zT` use it (for a field series, the minimum over time of the per-step mean), and the ELA drawn in plots is the mean of each output frame. |
| `zT` | `None` | All-snow elevation (m). Given without `zELA`, it sets the ELA: `zELA = zT - P/beta`. If `P` is a series, the derived ELA is also time-varying (per cell for a 2D `(ny, nx)` `zT`), and a `zT` series pairs with it step by step. An explicit `zELA` wins. |
| `U` | `1e-3` | Tectonic uplift rate (m/yr). Accepted array shapes differ by model; see below. |
| `bl` | `0` | Base-level waterline (m), scalar or length-`nt` series. A nonzero value warns that the analytical reference remains graded to datum zero. In 2D it also accepts a per-side dict with keys `'left'`, `'right'`, `'bottom'`, `'top'` (each value a scalar or a series): one datum per `'fixed_value'` outlet, with unlisted sides at `0`; a datum on a non-outlet side, or an unknown key, raises. Each interior node then takes its basin outlet's datum, and a corner shared by two fixed sides takes the x-side (`'left'`/`'right'`) value. Per-side `bl` needs the in-house driver. In 2D `bl` is consumed by modes B, C and D only — 2D mode A holds its borders at their initial elevation and ignores `bl` (scalar or per-side) entirely. |
| `flotation_gate` | `True` | Apply the waterline/effective-pressure gate to mode-B/C glacial erosion. `False` is an unbounded diagnostic control. |
| `flotation_ramp` | `0.1` | Dimensionless gate width γ. Must be nonnegative; `0` is the hard gate and values through `0.2` are the intended range. |
| `border_bed_uplift` | `None` → local `U` | Uplift of mode-B/C bed at base-level outlets. `0` freezes border recovery; accepted shapes follow `U` as described below. Under the escarpment's `uplift_type='wave'`, `None` is the wave's local rate each step. |

`U` accepts these forms:

- 1D: a scalar, `(nt,)`, `(nx,)`, `(nx, nt)`, or a flat array of size
  `nx*nt` interpreted in row-major `(nx, nt)` order.
- 2D: a scalar, `(nt,)`, `(ny, nx)`, or `(nt, ny, nx)`.

The 1D `border_bed_uplift` override is scalar. In 2D it may be scalar,
`(ny, nx)`, or `(nt, ny, nx)`; `None` reuses `U`, including its local and
time-varying structure.

### Fluvial and glacial physics

| key | default | meaning |
|---|---:|---|
| `n` | `1` | Dimensionless fluvial slope exponent. |
| `m` | `None` → `n/2` | Dimensionless fluvial discharge/area exponent. |
| `Ko` | `1e-6` | Fluvial erodibility in `E = Ko*Qf**m*S**n`; units are m^(1−3m) yr^(m−1) when `Qf` is m³/yr. In 2D, `0` turns fluvial erosion off; with `Ko` or `ce` at `0` no analytical reference is built (`m.analytical` is `None`, and profile plots draw none). |
| `Ac` | `2.5e-24` | Glen flow-law coefficient (Pa⁻³ s⁻¹). |
| `alpha_g` | `5` | Dimensionless valley width-to-mean-thickness ratio, `W = alpha_g*H`. |
| `sliding_law` | `'power'` | One of `'power'`, `'eff-exp'`, or `'coulomb'`. |
| `lambda_p` | `300` | Critical thickness scale (m) used by the power and effective-exponential laws. |
| `lambda_c` | `None` → `1000` | Regularized-Coulomb sliding length (m). |
| `nu` | `2` | Dimensionless primary steady-state glacial slope exponent. |
| `ell` | `None` → law-derived | Dimensionless glacial erosion-law exponent. Supplying it back-derives `nu` and `mu`; if both `ell` and `nu` are supplied, `ell` wins and a warning is emitted. |
| `mu` | `None` → law-derived | Advanced dimensionless flux-exponent override. It changes the analytical interpretation and the effective-exponential numerical law. The exact power and Coulomb numerical laws retain their law-derived exponents and emit a warning when `mu` is explicitly set. |
| `k` | `1` | Dimensionless accumulation-profile shape exponent used by the embedded analytical reference. |
| `ce` | `1e-5` | Glacial erodibility in `Eg = ce*(kt*ub)**ell`; units depend on `ell` (m^(1−ell) yr^(ell−1) when `kt*ub` is m/yr). In 2D, `0` turns glacial erosion off (see `Ko`); with `ce = Ko = 0` only uplift and hillslope diffusion (`D`) move the bed. |
| `hc_over_H` | `None` → `1.5` | Channel-floor datum: the centreline-to-mean depth ratio in `zs = zb + hc_over_H*H` (`1` is the sheet datum, a full-width cell). Must be finite and positive. Modes B and C build every surface from `(zb, H)` on it; 2D mode D's per-cell datum is `hc_over_H*s + 1*(1 − s)`. 2D mode A tracks the ice surface, whose dynamics do not depend on it: there it sets only the reported bed. The analytical reference (the 1D and 2D profile overlays and `rms_vs_analytical`) is built on the run's value; `SteadyStateProfile` takes it too (default 1.5), and it sets only the reported bed there, since the steady state is solved in surface and thickness. |
| `tau_c` | `1e5` | Coulomb yield stress (Pa). |
| `coulomb_clamp` | `1e-12` | Dimensionless minimum relative gap maintained from the regularized-Coulomb pole. |

The `nu`/`ell`/`mu` relationships depend on the sliding law. Prefer `nu` for
ordinary runs. Use direct exponent overrides only when you mean to depart from
the exact-law/default analytical pairing.
{doc}`choosing_glacial_parameters` explains what `beta`, `ce`/`ell`,
`lambda_p` and `lambda_c` set physically and how they trade off.

## 1D-only parameters

| key | default | meaning |
|---|---:|---|
| `L` | `5e4` | Profile length (m). |
| `dx` | `None` | Requested spacing (m); when set, `nx = int(L/dx) + 1`. |
| `xo` | `300` | Head-catchment reference length (m). |
| `k_h` | `5` | Hack-law coefficient in `A = k_h*x**d` (units m^(2−d)). |
| `d` | `1.8` | Dimensionless Hack-law exponent. |
| `sigma` | `0.45` | Dimensionless tributary-contribution parameter. |
| `lam` | `None` → `d*sigma/(d*sigma+k)` | AAR-like ratio used by the analytical reference; must lie in `[0, 1)`. An explicit value sets the closure scale `kappa_c = 1/(1-lam)`. |
| `left_bc` | `'base_level'` | Left boundary condition: `'base_level'` or `'reflecting'`. At least one side must be base level. |
| `right_bc` | `'reflecting'` | Right boundary condition: `'base_level'` or `'reflecting'`. |
| `cap_ice_accumulation` | `True` | Cap accumulation at `P`. `False` is mainly for comparison with the uncapped analytical ansatz. |
| `floating_termini` | `True` | In mode B, treat ice reaching a closed basin as floating/bed-decoupled. `False` lets a grounded toe carve its overdeepening; no effect in mode A. |

When `initial_topography=None`, 1D creates a quadratic one- or two-sided
profile from the boundary conditions. Its stored arrays use the model's native
x order, from `L` down to `0`.

## 2D-only parameters

### Domain, routing, and initial surface

| key | default | meaning |
|---|---:|---|
| `Lx`, `Ly` | `5e4`, `5e4` | Domain lengths (m). |
| `ny` | `201` | Number of nodes in y; `nx` defaults to `201`. |
| `boundary_status` | `['fixed_value', 'fixed_value', 'looped', 'looped']` | Edge status in `[left, right, y-min, y-max]` order. Values are `'fixed_value'`, `'core'`, or `'looped'`, or one value for all four sides. `'looped'` must be set on both sides of an axis or neither, and at least one side must be `'fixed_value'` (the outlet); anything else raises `ValueError`. |
| `initial_max_elevation` | `1000` | Peak of the generated tent surface (m); used only when `initial_topography=None`. |
| `noise_amplitude` | `100` | Amplitude of uniform nonnegative random relief added to the initial surface (m), whether generated or supplied. Set `0` to preserve supplied elevations exactly. |
| `seed` | `None` | Random seed for initial relief. Set an integer for reproducible runs. |
| `flow_routing` | `'single'` (`'mfd'` in mode D) | `'single'` for in-house D8 steepest descent, `'multiple'` for multiple-flow-direction spreading over every strictly downhill neighbour, or `'infinite'` for Tarboton D-infinity two-receiver flow. The router codes `'mfd'` (= `'multiple'`) and `'dinf'` (= `'infinite'`) are accepted as aliases, and the rest of this reference uses them. Both multi-receiver routers turn on `slope_normalise` by default. `'mfd'` requires mode B, C or D (mode A has only the one-cell closure), a transect `closure_flux`, and square cells (`dx == dy`); anything else raises. It is the only routing mode D accepts. It supports mode C — carving and `trunk_surface`, at the router-derived `trunk_dip_k` — where the trunk is resolved; see {doc}`configuring_a_run`. |
| `slope_normalise` | `None` | Gamma effective-slope normalisation. `None` resolves from the router: ON for both multi-receiver routers (`'dinf'`, `'mfd'`), OFF for `'single'`. A multi-receiver reduction `S = sum_k w_k (zs_i - zs_rk)/L_k` is the surface gradient PROJECTED on the weighted receiver direction `v = sum_k w_k e_k`; dividing by `gamma = |v|` recovers the gradient magnitude, exactly so on a plane. It is applied as gamma-scaled link lengths, so the thickness closure and all three eroders read the same effective slope. Two places deliberately keep the raw link lengths, where a single donor edge leaves gamma undefined: the outflow border's arrival slope and mode A's border thickness closure. `False` restores the unnormalised reduction — under D-inf that is the older behaviour (see the changelog), which returned 0.9239 of the true gradient on a plane tilted pi/8 and inflated `H` there by up to about 5 %. Under `'single'` the parameter is a no-op (one receiver, `gamma == 1` identically); under `'mfd'` an explicit `False` raises, because the raw MFD reduction is 17-24 % low by construction. |
| `mfd_exponent` | `1.0` | Spreading exponent p in the MFD weights `(S_k/S_max)^p`; positive, finite up to 50 or `inf`, inert under the other routers. p = 1 is Budd & Warner's split and the only exponent whose weighted receiver direction is exactly the surface gradient direction on a square 8-neighbour stencil; larger p concentrates toward the steepest link, and `inf` routes down it alone (equal steepest links share), which is still the MFD code path and not `flow_routing='single'`. A finite p above 50 is taken as `inf`, with a warning: there the smaller weights underflow to zero, and each further step in p changes a mode-D run by no more than repeating it does. Provisional pending the stage-S4 dome sweep. |
| `closure_flux` | `None` → mode-dependent | Which ice flux the thickness closure and the erosion laws read at a cell. `'cell'` is the cell's own routed flux. `'transect'` is the routed flux crossing a line one closure width (`alpha_g*H`) long through the cell, across its flow, so a trunk with ice-free flanks reads its own flux and a cell inside a broad ice body reads the flux of the whole width the closure assumes. `'transect_min'` is the same, but a neighbour contributes at most the cell's own flux. `None` resolves to `'transect_min'` in modes B, C and D and to `'cell'` in mode A, which accepts no other value; `'mfd'` needs a transect. Routing, `Qg_out`, the carve, the flexure load and the sediment keep the routed flux; `save_closure_pair` stores the closure's own. |
| `router_backend` | `'inhouse_d8'` | Reserved router plug point; currently this is the only accepted value. |
| `numerics_backend` | `'inhouse'` | Reserved flexure/diffusion plug point; currently this is the only accepted value. |
| `D` | `1e-3` | Hillslope diffusivity (m²/yr). |
| `parallel_erode` | `True` | Level-scheduled parallel mode-B/C eroder, bit-for-bit the serial implementation (`False`) at any thread count; on Apple silicon its per-level loops use as many threads as the Mac has performance cores. Under `flow_routing='mfd'` (mode D's default routing) the default follows Numba's threading layer instead: `True` under TBB, `False` otherwise. Multiple-flow spreading raises the topological level count from 151 to about 400 on a 201x201 grid, so on `workqueue` the per-level fork/join cost roughly triples and the parallel schedule loses; under TBB it is faster (measured in mode D; see {doc}`../getting_started/install`). |

### Width carving and mode-C routing

| key | default | meaning |
|---|---:|---|
| `carve_width` | `None` → mode-dependent | `True` for default mode C and for mode D, `False` for explicitly selected B/A. Explicit `True` requires B; `carve_width=False` with mode C or D is rejected. |
| `widening_rate` | `3` | Excess footprint erosion η: footprint rate is `(1 + eta)` times centerline incision. Must be nonnegative; `None`, positive infinity, `'inf'`, or `'infinity'` requests instant target imposition, which mode D rejects. |
| `trunk_surface` | `None` → mode-dependent | Defaults `True` for B plus carving (mode C), otherwise `False` (mode D included). It fabricates a converging surface that only the router graph reads — the mass balance and the hillslope diffusion keep the dynamic surface — and requires B/C. It combines with `routing_filter_factor > 0` only in mode D. |
| `trunk_dip_k` | `None` → router-dependent | Dimensionless cross-valley dip coefficient `S_c/S_v` for trunk-surface routing. `None` resolves to `1.0` (`constants.TRUNK_DIP_K_MFD`) under `'mfd'` and `0.6` (`constants.TRUNK_DIP_K`) under `'single'` and `'dinf'`: MFD's axis cells stop leaking flux off the trunk only at k ≥ 1 (closed-form axis leak `2s/(1+2s)`, `s = max(1−k, 0)/√2`), while D8 converges above 0.414. An explicit value is honoured; the sentinel is stored as passed and re-resolved from the constants at load time (`load()` warns when the saved `siim_version` differs). |
| `inner_damping` | `None` → mode-dependent | How each pass commits: `'omega'` (the uniform ω of `inner_relax`), `'adaptive'` or `'adaptive_capped'` (per-cell damping of the cells whose update keeps reversing, floor 0.2 or 1/3). `'adaptive'` in modes B/C/D unless `inner_relax` is set (then `'omega'`); `'omega'` in mode A. The adaptive presets require B/C/D and the standalone driver, and reject an explicit `inner_relax` other than their floor. |
| `inner_passes` | mode-dependent | Evaluations of the diagnostic ice map per landscape step (the within-step damped iteration): `2` in modes B/C under `'omega'`, `1` otherwise. Integer ≥ 1; a value above 1 requires B/C/D and the standalone driver. |
| `inner_relax` | mode-dependent | Update fraction ω of that iteration under `'omega'`: `H ← H + ω(F(H) − H)` after every pass, so `1 − ω` is retained: `0.5` in modes B/C, `1/3` in mode D, `1.0` in mode A. Under an adaptive preset it reports the preset's floor. Must be in `(0, 1]`; below 1 requires B/C/D and the standalone driver. |
| `channel_bed_bar` | `None` → `1.5` in mode D | Mode D's bed bar `k_B`: a cell reads channel only where the lowest bed within its half-width `alpha_g*H/2` lies below the bed around it (the median of the reading's ring) by more than `(k_B − 1)` times the ring's median ice thickness, so a fast stream on a plateau with no trough reads sheet. Finite and ≥ 0; 0 turns the bar off; raising it drops shallower troughs first (2.5 and above lost trunk channels on the 1000 km escarpment run). Mode D only; any other mode raises. |
| `routing_filter_factor` | `0.0` (`2` in mode D) | Routing-surface spatial filter: above 0 the router graph alone is built from a thickness-weighted triangular mean of `zb + hc·H` over a window of half-width `max(f·H/dx − ½, 0)` cells (about f ice thicknesses; 2 is the measured setting). Finite and ≥ 0; above 0 requires B/C, the standalone driver, square cells and `trunk_surface=False` (or mode D, under which the two combine). Windows wrap on looped axes. |
| `width_hack_k` | `0.5` (`0.25` in mode D) | Coefficient in `width = width_hack_k*A**width_hack_p` (units m^(1−2p)); used for below-ELA ablation. |
| `width_hack_p` | `0.5` | Dimensionless exponent in that glacier-width correction. |

The mode-C helper affects routing, so it can change the true simulation. It
is not a display filter. See {doc}`configuring_a_run` for the interaction.

### Flexure and optional diagnostics

| key | default | meaning |
|---|---:|---|
| `flexure` | `False` | Enable elastic-plate flexure from rock denudation/uplift and, by default, ice loading. The plate is periodic along a `'looped'` axis: see {ref}`flexure-looped-edges`. |
| `ice_load` | `True` | Include ice load when `flexure=True`; otherwise calculate unloading only. Each channel loads the plate with its cross-section `alpha_g*H**2`, spread over the cells it covers (a parabola of mean depth `H` across the width `alpha_g*H`, each cell counted once); in mode D a sheet cell (s = 0) loads a slab of thickness `H` instead. |
| `lithos_density` | `2800` | Lithospheric density (kg/m³), scalar or `(ny, nx)` field. |
| `asthen_density` | `3200` | Asthenospheric density (kg/m³). |
| `e_thickness` | `35e3` | Effective elastic thickness (m). |
| `save_closure_pair` | `False` | Store the closure's own flux and thickness (`closure_Qg_out`, `closure_H_out`) where they differ from `Qg_out` / `H_out`, so `ice_speed` and `field='bedrock+speed'` are exact. Off by default: the rows add about 10 % to the saved output in modes B and C and about 19 % in mode D, and without them the speed is approximate and `ice_speed` warns. See {doc}`outputs_and_io`. |
| `track_sediment` | `False` | Accumulate per-step eroded volume through the flow graph and report it: `True`/`'basin'` as per-node throughput/cumulative rasters, `'edge'` as the volume crossing each domain edge, or `'both'`. |

(flexure-looped-edges)=
#### Flexure on a looped grid

A `'looped'` axis stands for an infinitely long source, a slice from a chain
much longer than the domain, so the elastic plate is periodic along it: the load
repeats along that axis and the plate is solved in a Fourier basis there. A
non-looped axis keeps the sine basis, which holds the plate flat (`w = 0`) about
half a domain-width beyond its edges, the small-domain rigidity support. With
both axes looped no edge is held flat: the plate is an infinite periodic sheet
and the mean of the load is fully Airy-compensated, so a uniform metre of
unloading rebounds ρₛ/ρₐ ≈ 0.875 m everywhere, against a few thousandths of that
on a 20 km grid with a non-looped axis.

On a small domain this matters: the plate's flexural parameter,
α = (4D / (ρₐ g))^(1/4), is about 55 km at `e_thickness=20e3` and about 84 km at
the default 35 km, longer than a typical 20 km valley domain, so a plate held
flat beyond a looped edge would make the deflection depend on where the seam
falls and give about a third of the periodic plate's.

(initial-topography-formats)=
## Initial-topography formats

A NumPy array keeps the separately supplied grid dimensions and lengths. A
DataFrame or CSV supplies the grid as well and therefore overrides `nx`/`L` in
1D or `nx`/`ny`/`Lx`/`Ly` in 2D. These formats require pandas.

- 1D columns: `x`, `topography__elevation`, and optional `time`.
- 2D columns: `x`, `y`, `topography__elevation`, and optional `time`.

If `time` is present, only its latest value is loaded. Coordinates must form a
finite, uniformly spaced grid; the 2D grid must have no missing cells. The 1D
loader sorts x ascending, whereas the model's native `m.x` runs from `L` to
zero, so reverse an array yourself if its physical orientation is opposite to
the boundary convention you intend.

In 2D, the loaded or supplied surface is the base passed to the same initializer
as a generated surface. Uniform noise in `[0, noise_amplitude)` is then added
away from fixed-value edges. Use `noise_amplitude=0` when an imported DEM must
remain exact; otherwise set an integer `seed` to reproduce the perturbation.

## Running the 2D model through the optional adapter

`m.run()` uses the standalone in-house driver. `m.run(driver='xsimlab')` uses
the optional fastscape/xsimlab adapter and requires its conda environment. The
two drivers expose the same output contract. The adapter routes once per step,
so it runs the undamped model: modes B and C need
`inner_relax=1.0, inner_passes=1` there (their default damping needs the
in-house driver), and
mode D, `routing_filter_factor > 0`, a per-side `bl` and an `(nt, ny, nx)`
`zELA` series are in-house only. The `hooks=` argument is honored
only by the xsimlab driver; the in-house driver rejects a non-`None` value.
See {doc}`fastscape_processes` for direct process composition.

## Escarpment-variant additions

`siim.escarpment.siim_escarpment` accepts every 2D key above plus the following
keys. The uplift and initial-topography switches are independent.

| key | default | meaning |
|---|---:|---|
| `uplift_type` | `'block'` | `'block'` uses the standard `U` forcing. `'wave'` selects a moving Gaussian uplift wave and requires `delta_h`; wave uplift also requires scalar `U`. |
| `delta_h` | `None` | Target integrated uplift deposited as the wave passes (m); required with `uplift_type='wave'`. |
| `wave_width` | `2e5` | Gaussian 1/e half-width (m). |
| `wave_velocity` | `15e-3` | Wave propagation velocity (m/yr). |
| `x_escarpment` | `0` | Initial wave-centre position relative to the left edge (m). |
| `wave_calibration` | `1` | Dimensionless multiplier on the wave's peak rate; `1` deposits `delta_h` for a complete passage. |
| `U_inf` | `0` | Background uplift rate added to the wave (m/yr). For the embedded analytical reference, representative uplift is `U_inf + delta_h/(T - to)`, the mean rate over the run. |
| `init_type` | `'sloped'` | `'sloped'` uses the standard generated/imported surface. `'plateau'` selects the arctangent-smoothed plateau (exactly `0` at the left edge and `plateau_zo - plateau_dz` at the right edge) and requires `plateau_zo`; it cannot be combined with `initial_topography`. For a plain `siim2d` run, `siim.escarpment.plateau_topography` builds the same surface together with the per-side `bl` that puts the right outlet at the plateau's edge elevation. |
| `plateau_zo` | `None` | Plateau elevation (m); required with `init_type='plateau'`. |
| `plateau_dz` | `1` | Elevation change across the plateau (m), used to seed the divide. |
| `plateau_frac` | `0.8` | Fraction of the x extent occupied by the plateau. |
| `plateau_w` | `1e4` | Escarpment transition width (m). |

The plateau initializer also honors the ordinary 2D `seed`,
`noise_amplitude`, grid, and boundary parameters. See {doc}`../api/model2d` for
the class API.
