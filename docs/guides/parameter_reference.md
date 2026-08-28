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
| `mode` | `'bedrock+ice_thickness'` | `'C'` | State convention. `'A'`/`'ice_surface'` tracks the surface; `'B'`/`'bedrock+ice_thickness'` tracks bed and ice thickness. 2D `'C'` resolves to B plus width carving. C is rejected in 1D. |
| `H_diffusivity` | `None` | `None` | Ice-thickness diffusivity (m²/yr); `None` or `0` disables it, and negative values are rejected. It works in both 1D modes but only B/C in 2D. |
| `initial_topography` | `None` | `None` | NumPy array, pandas DataFrame, or CSV path. Array shape is `(nx,)` in 1D or `(ny, nx)` in 2D. The 2D initializer still adds `noise_amplitude`; set it to zero for the exact supplied elevations. See {ref}`initial-topography-formats`. |
| `progress_bar` | `True` | `True` | Show tqdm run progress. |

### Climate, uplift, and base level

| key | default | meaning |
|---|---:|---|
| `P` | `1` | Precipitation/runoff rate (m/yr), as a scalar or length-`nt` time series. For a series, its mean supplies the analytical reference. |
| `beta` | `1e-2` | Linear mass-balance gradient (yr⁻¹). |
| `zELA` | `1500` | Equilibrium-line altitude (m), scalar or length-`nt` series. The minimum of a series supplies the analytical maximum-glaciation reference. |
| `zT` | `None` | All-snow elevation (m). To derive ELA from it, pass `zELA=None`; then `zELA = zT - P/beta`. If `P` is a series, the derived ELA is also time-varying. An explicit `zELA` wins. |
| `U` | `1e-3` | Tectonic uplift rate (m/yr). Accepted array shapes differ by model; see below. |
| `bl` | `0` | Base-level waterline (m), scalar or length-`nt` series. A nonzero value warns that the analytical reference remains graded to datum zero. |
| `flotation_gate` | `True` | Apply the waterline/effective-pressure gate to mode-B/C glacial erosion. `False` is an unbounded diagnostic control. |
| `flotation_ramp` | `0.1` | Dimensionless gate width γ. Must be nonnegative; `0` is the hard gate and values through `0.2` are the intended range. |
| `border_bed_uplift` | `None` → local `U` | Uplift of mode-B/C bed at base-level outlets. `0` freezes border recovery; accepted shapes follow `U` as described below. |

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
| `Ko` | `1e-6` | Fluvial erodibility in `E = Ko*Qf**m*S**n`; units are m^(1−3m) yr^(m−1) when `Qf` is m³/yr. |
| `Ac` | `2.5e-24` | Glen flow-law coefficient (Pa⁻³ s⁻¹). |
| `alpha_g` | `5` | Dimensionless valley width-to-mean-thickness ratio, `W = alpha_g*H`. |
| `sliding_law` | `'power'` | One of `'power'`, `'eff-exp'`, or `'coulomb'`. |
| `lambda_p` | `300` | Critical thickness scale (m) used by the power and effective-exponential laws. |
| `lambda_c` | `None` → `1000` | Regularized-Coulomb sliding length (m). |
| `nu` | `2` | Dimensionless primary steady-state glacial slope exponent. |
| `ell` | `None` → law-derived | Dimensionless glacial erosion-law exponent. Supplying it back-derives `nu` and `mu`; if both `ell` and `nu` are supplied, `ell` wins and a warning is emitted. |
| `mu` | `None` → law-derived | Advanced dimensionless flux-exponent override. It changes the analytical interpretation and the effective-exponential numerical law. The exact power and Coulomb numerical laws retain their law-derived exponents and emit a warning when `mu` is explicitly set. |
| `k` | `1` | Dimensionless accumulation-profile shape exponent used by the embedded analytical reference. |
| `ce` | `1e-5` | Glacial erodibility in `Eg = ce*(kt*ub)**ell`; units depend on `ell` (m^(1−ell) yr^(ell−1) when `kt*ub` is m/yr). |
| `tau_c` | `1e5` | Coulomb yield stress (Pa). |
| `coulomb_clamp` | `1e-12` | Dimensionless minimum relative gap maintained from the regularized-Coulomb pole. |

The `nu`/`ell`/`mu` relationships depend on the sliding law. Prefer `nu` for
ordinary runs. Use direct exponent overrides only when you mean to depart from
the exact-law/default analytical pairing.

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
| `hc_over_H` | `None` → `1.5` | Centerline-to-mean depth ratio in `zs = zb + hc_over_H*H`. Must be positive. This experimental 1D override does not exist in 2D. |

When `initial_topography=None`, 1D creates a quadratic one- or two-sided
profile from the boundary conditions. Its stored arrays use the model's native
x order, from `L` down to `0`.

## 2D-only parameters

### Domain, routing, and initial surface

| key | default | meaning |
|---|---:|---|
| `Lx`, `Ly` | `5e4`, `5e4` | Domain lengths (m). |
| `ny` | `201` | Number of nodes in y; `nx` defaults to `201`. |
| `boundary_status` | `['fixed_value', 'fixed_value', 'looped', 'looped']` | Edge status in `[left, right, y-min, y-max]` order. Values are `'fixed_value'`, `'core'`, or `'looped'`; looped edges should be paired on an axis. |
| `initial_max_elevation` | `1000` | Peak of the generated tent surface (m); used only when `initial_topography=None`. |
| `noise_amplitude` | `100` | Amplitude of uniform nonnegative random relief added to the initial surface (m), whether generated or supplied. Set `0` to preserve supplied elevations exactly. |
| `seed` | `None` | Random seed for initial relief. Set an integer for reproducible runs. |
| `flow_routing` | `'single'` | `'single'` for in-house D8 steepest descent or `'dinf'` for Tarboton D-infinity multiple flow. |
| `router_backend` | `'inhouse_d8'` | Reserved router plug point; currently this is the only accepted value. |
| `numerics_backend` | `'inhouse'` | Reserved flexure/diffusion plug point; currently this is the only accepted value. |
| `D` | `1e-3` | Hillslope diffusivity (m²/yr). |
| `parallel_erode` | `True` | Level-scheduled parallel mode-B/C eroder. `False` selects the bit-for-bit serial implementation. |

### Width carving and mode-C routing

| key | default | meaning |
|---|---:|---|
| `carve_width` | `None` → mode-dependent | `True` for default mode C and `False` for explicitly selected B/A. Explicit `True` requires B; `mode='C', carve_width=False` is rejected. |
| `widening_rate` | `3` | Excess footprint erosion η: footprint rate is `(1 + eta)` times centerline incision. Must be nonnegative; `None`, positive infinity, `'inf'`, or `'infinity'` requests instant target imposition. |
| `trunk_surface` | `None` → mode-dependent | Defaults `True` for B plus carving (mode C), otherwise `False`. It fabricates a converging routing/mass-balance surface and requires B/C. |
| `trunk_dip_k` | `0.6` | Dimensionless cross-valley dip coefficient for trunk-surface routing. |
| `routing_relax` | `None` → mode-dependent | Routing-surface EMA coefficient: `0.5` for mode C, otherwise `0`. Must be in `[0, 1)` and a positive value requires B/C. |
| `width_hack_k` | `0.5` | Coefficient in `width = width_hack_k*A**width_hack_p` (units m^(1−2p)); used for below-ELA ablation. |
| `width_hack_p` | `0.5` | Dimensionless exponent in that glacier-width correction. |

The mode-C helpers affect routing and mass balance, so they can change the true
simulation. They are not display filters. See {doc}`configuring_a_run` for the
interaction and ELA-retuning guidance.

### Flexure and optional diagnostics

| key | default | meaning |
|---|---:|---|
| `flexure` | `False` | Enable elastic-plate flexure from rock denudation/uplift and, by default, ice loading. |
| `ice_load` | `True` | Include ice load when `flexure=True`; otherwise calculate unloading only. |
| `lithos_density` | `2800` | Lithospheric density (kg/m³), scalar or `(ny, nx)` field. |
| `asthen_density` | `3200` | Asthenospheric density (kg/m³). |
| `e_thickness` | `35e3` | Effective elastic thickness (m). |
| `track_sediment` | `False` | Accumulate per-step eroded volume through the flow graph and expose sediment throughput/cumulative outputs. |

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
two drivers expose the same output contract. The `hooks=` argument is honored
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
| `U_inf` | `0` | Background uplift rate added to the wave (m/yr). For the embedded analytical reference, representative uplift is `U_inf + delta_h/T`. |
| `init_type` | `'sloped'` | `'sloped'` uses the standard generated/imported surface. `'plateau'` selects the arctangent-smoothed plateau and requires `plateau_zo`; it cannot be combined with `initial_topography`. |
| `plateau_zo` | `None` | Plateau elevation (m); required with `init_type='plateau'`. |
| `plateau_dz` | `1` | Elevation change across the plateau (m), used to seed the divide. |
| `plateau_frac` | `0.8` | Fraction of the x extent occupied by the plateau. |
| `plateau_w` | `1e4` | Escarpment transition width (m). |

The plateau initializer also honors the ordinary 2D `seed`,
`noise_amplitude`, grid, and boundary parameters. See {doc}`../api/model2d` for
the class API.
