# Choosing glacial parameters

Here we discuss five important knobs for the glacial model: the mass-balance
gradient `beta`, the erosion coefficient `ce` and slope exponent `nu`, and the
two sliding length scales `lambda_p` and `lambda_c`. This page says what each
one sets, which of them trade off against each other, and where the defaults
sit against observations. Every number below was measured on the 1D model's
default configuration (L = 50 km, U = 1 mm/yr, β = 0.01/yr, P = 1 m/yr,
z_ELA = 1500 m) with the 1D model and the analytical steady state. See
{doc}`parameter_reference` for the keys, {doc}`configuring_a_run` for the
choices that set the regime, and {doc}`concepts` for the modes and the
autogenic cycle referred to below.

## The mass-balance gradient `beta` and the lapse rate

The climate model is a linear mass balance,

$$b = \beta\,(z - z_{\rm ELA}),\qquad
\text{capped at } b = P \text{ above } z_T = z_{\rm ELA} + P/\beta ,$$

so `beta` (yr⁻¹) is the rate at which the balance changes with elevation and
$P/\beta$ is the height above the ELA at which the capped balance reaches
full accumulation. A temperature
lapse rate $\Gamma$ enters through the melt: in a degree-day model with factor
DDF, differentiating the positive-degree-day sum with respect to elevation gives

$$\beta = \mathrm{DDF}\cdot 365\cdot|\Gamma|\cdot f_{\rm melt} \;+\; P\,\frac{d f_{\rm snow}}{dz},$$

with DDF in m w.e. °C⁻¹ d⁻¹ and $\Gamma$ in °C m⁻¹, where $f_{\rm melt}$ is the fraction of the year above freezing at that
elevation and the second term, the shift of the rain/snow partition, is 5–15 %
of the total near the ELA. The identity holds to about 1 % against a daily
degree-day model with an annual temperature cycle. For the
$\Gamma = -6.5$ °C/km lapse rate the first term is
$2.4\times10^{-3}\,\mathrm{DDF}\,f_{\rm melt}$ yr⁻¹ with DDF in
mm w.e. °C⁻¹ d⁻¹, i.e. $0.014\,f_{\rm melt}$ at DDF = 6. Degree-day factors of
2.5–11.6 (snow) and 5.4–20 (ice) mm w.e. °C⁻¹ d⁻¹ are tabulated by
{cite:t}`hockTemperatureIndexMelt2003`, with 3–8 typical of the Alps. Because
$f_{\rm melt}$ grows downward, the real $b(z)$ is curved, its gradient
steepening downglacier, not piecewise linear: for DDF = 6 and a 9 °C seasonal amplitude it is 0.004/yr at the ELA,
0.006/yr averaged over the first kilometre below it, and reaches the asymptote
0.014/yr only where the whole year is above freezing. Read `beta` as the
gradient averaged over the glacier's own elevation range. Three effects the
identity leaves out all push the same way: the late-summer ablation surface
below the ELA is ice, whose degree-day factor is about twice that of snow, so
snowline migration steepens the tongue gradient toward the 0.008–0.01/yr
measured on Alpine tongues; an orographic precipitation gradient adds
$f_{\rm snow}\,dP/dz$; and refreezing of meltwater lowers the gradient near
the ELA.

- **For a −6.5 °C/km lapse rate and an alpine degree-day factor, β ≈ 0.005/yr
  (0.004–0.008).** The package default 0.01/yr is a temperate maritime
  ablation-tongue value; it needs DDF ≳ 8 with a melt season longer than half
  the year, or a wetter climate (at a fixed ELA, β grows roughly as $P^{0.3}$, with
  $P^{1/3}$ as the short-melt-season limit, because a wetter climate has a
  longer melt season at the ELA).
- **Across climates the plausible range is 10⁻³ to 10⁻².** The same identity
  gives about 0.001/yr for continental or polar margins (DDF ≈ 3 with a melt
  season of ~15 % of the year) and about 0.01/yr for maritime tongues (DDF ≈ 8
  with a melt season over half the year); the default sits at the top of that
  range and the Alpine cases below in the middle.
- **The cap band follows from β.** With P = 1 m/yr full accumulation is reached
  $P/\beta$ = 100 m above the ELA at β = 0.01 and 200 m at β = 0.005, while the
  degree-day rollover to full accumulation spans 200–700 m. The narrow default
  band is a symptom of the steep default β, not a separate approximation.

Two worked examples of the translation, with typical inputs rather than fits
to a named glacier (Γ = −6.5 °C/km, DDF 4 for snow and 7 for ice, and the
seasonal amplitude and precipitation below). The model tunes the climate to
the prescribed ELA, so the mean annual temperature it implies at the ELA is a
free check on the inputs:

| | Alps, modern | Alps, LGM |
|---|---|---|
| ELA, precipitation at the ELA, seasonal amplitude | 3000 m, 1.5–2.5 m/yr, 9 °C | 2000 m, 0.4–0.8 m/yr, 9–13 °C |
| implied mean annual T at the ELA | −5 to −4 °C | −11 to −6 °C |
| melt-season fraction at the ELA | 0.31–0.36 | 0.18–0.25 |
| β at the ELA (yr⁻¹) | 0.0062–0.0074 | 0.0036–0.0050 |
| β over the ablation tongue (yr⁻¹) | 0.0078–0.0090 | 0.0051–0.0068 |
| accumulation-zone gradient (yr⁻¹) | 0.0029–0.0037 | 0.0014–0.0022 |
| rollover to full accumulation | 500–650 m | 275–355 m |
| siim `beta`, `P`, `zELA` | 0.007, 2, 3000 | 0.005, 0.6, 2000 |

The modern ELA temperature matches the standard Alpine climatology, and the
LGM case implies 9–12 °C of cooling at 2000 m, inside the reconstructed range,
so the inputs hang together; the drier LGM climate is what lowers β. The
identity does not transfer to polar settings such as the Transantarctic
Mountains, where ablation is sublimation and the melt-season fraction is
essentially zero: β from melt is nil, the balance is set by `P` alone, and the
honest siim setting puts `zELA` below the bed so the whole domain accumulates
at P.

## The erosion coefficient `ce` and slope exponent `nu`

Glacial erosion is a power of the sliding speed,

$$E_g = c_e\,u_b^{\ell},$$

with $u_b$ in m/yr (the model's `kt` factor converts its m/s sliding speed),
so $c_e$ has units (m/yr)^(1−ℓ). siim's primary exponent is the glacial
slope exponent `nu` (default 2), the power of slope in the steady-state form
$E_g \propto Q_g^{\mu} S^{\nu}$. The erosion exponent ℓ follows from it once
two things are fixed, Glen's flow-law exponent $n = 3$ and the sliding law:
the power law under its effective-exponent closure gives
$u_b \propto S^{5/3} Q_g^{4/9}$, so $\ell = 3\nu/5$ (ν = 2 gives ℓ = 1.2), and
the Coulomb law at yield gives $u_b \propto S^{2} Q_g$, so $\ell = \nu/2$
(ν = 2 gives ℓ = 1). Pass `ell` to override that mapping. Below, $c_e$ is
tabulated against ℓ, because its units depend on ℓ and the observational
compilation is fitted in ℓ; the ν rows give the siim key for each law.

**At steady state the pair collapses to one number.** Erosion balances uplift,
$E_g = U$, so every point on the glacier slides at the same speed, the
balancing sliding speed

$$u_b^* = (U/c_e)^{1/\ell},$$

the speed at which the erosion law returns the uplift rate. The sliding law
then fixes thickness and slope from the flux: the
steady-state landscape depends on $(c_e, \ell, U)$ only through $u_b^*$. This
is exact: the analytical solutions for ν = 1.3–5 under the power law and
ν = 1.6–6 under Coulomb (ℓ = 0.8–3 in both) at fixed $u_b^*$ agree to 1e-14,
and the 1D model's converged
mode-A states agree to 1.5 cm (power) and to the terminus flicker floor of
~10 m (Coulomb). The way to choose the pair is therefore to choose $u_b^*$, the
sliding speed a glacier must reach to keep pace with uplift, and then set
$c_e = U/u_b^{*\,\ell}$; the default landscape's family is tabulated
below. As with the stream-power pair (`Ko`, `n`), the family is consistent at
one uplift rate only, because the sensitivity to uplift depends on the
exponent. Over the usual range ℓ = 1–2:

| ℓ | 1.0 | 1.2 | 1.5 | 2.0 |
|---|---|---|---|---|
| ν, power law | 1.67 | 2 | 2.5 | 3.33 |
| $d\ln c_s/d\ln U$, power law | 0.46 | 0.38 | 0.30 | 0.23 |
| $d\ln z_o/d\ln U$, power law | 0.24 | 0.21 | 0.19 | 0.17 |
| ν, Coulomb | 2 | 2.4 | 3 | 4 |
| $d\ln c_s/d\ln U$, Coulomb | 0.28 | 0.23 | 0.18 | 0.12 |
| $d\ln z_o/d\ln U$, Coulomb | 0.09 | 0.08 | 0.07 | 0.06 |

Doubling U steepens the glacial reach by 38 % at ν = 1.67 but 17 % at ν = 3.33
under the power law, so two pairs that agree at 1 mm/yr differ by 17 % in
steepness at 2 mm/yr (12 % under Coulomb). The divide moves far less than the
steepness because the ELA pins the glacial reach.

**Where the default sits against observations.** The glacial erosion rule
compiled by {cite:t}`hermanImpactGlaciersMountain2021`, $\dot e = K_g |u_s|^{\ell}$
with both sides in m/yr, is the same law, so $K_g$ maps onto $c_e$ directly.
Their two drawn fits, $K_g$ = 2.7e-7 with ℓ = 2 and 4e-5 with ℓ = 1,
cross at 148 m/yr and 5.9 mm/yr, and the 106-glacier cloud scatters about
one decade either side of the ℓ = 1 line (5–95 %), a factor of ~9 either
side in $c_e$ at any exponent. Any exponent can pass through that pivot; only the ℓ = 1 rule is centred on
the cloud, so at other exponents the row below is an extrapolation off the
pivot, shown against the family that reproduces the
default landscape ($u_b^*$ = 46.4 m/yr at U = 1 mm/yr):

| ℓ | 1.0 | 1.2 | 1.5 | 2.0 | 2.5 | 3.0 |
|---|---|---|---|---|---|---|
| ν, power law | 1.67 | 2 | 2.5 | 3.33 | 4.17 | 5 |
| ν, Coulomb | 2 | 2.4 | 3 | 4 | 5 | 6 |
| $c_e$, default landscape | 2.2e-5 | 1.0e-5 | 3.2e-6 | 4.6e-7 | 6.8e-8 | 1.0e-8 |
| $c_e$, compilation centre | 4.0e-5 | 1.5e-5 | 3.3e-6 | 2.7e-7 | 2.2e-8 | 1.8e-9 |

The default $c_e$ = 1e-5 at ν = 2 (ℓ = 1.2) sits 0.17 decades below the
centre, and its erosion rate at 100 m/yr, 2.5 mm/yr, is within 10 % of the
ℓ = 2 fit and below the ℓ = 1 fit. One
caveat is large: the compiled velocities are mostly surface speeds, whereas
$u_b$ is sliding alone. On the default profile the power-law glacier slides at
only 20–30 % of its surface speed (analytical and 1D-model profiles) and the
Coulomb glacier at 90 % (the reason is `lambda_p`, below). Read against
surface speeds, the power-law default moves 0.8–1.0 decades below the centre,
to the erodible edge of the compilation, and the Coulomb default barely
moves.

## The power-law sliding length `lambda_p`

The power sliding law is

$$u_b = \frac{2A_c}{5}\,\tau^3 H \left(\frac{\lambda_p}{H}\right)^2,$$

and the depth-mean deformation speed is $u_d = (2A_c/5)\tau^3 H$, so
$u_b/u_d = (\lambda_p/H)^2$: **`lambda_p` is the thickness at which sliding
equals deformation.** Thinner ice slides, thicker ice creeps.

At steady state, with $u_b = u_b^*$, the flux closure becomes

$$\frac{Q_g}{k_t\alpha_g} = H^2 u_b^*\left(1 + \frac{H^2}{\lambda_p^2}\right),$$

so where $H \ll \lambda_p$ the thickness is independent of `lambda_p` and the
slope scales as $\lambda_p^{-2/3}$; the analytical steady state gives
$c_s \propto \lambda_p^{-12/19}$. Measured in mode A with the accumulation cap
off (the like-for-like comparison with the analytical), the 1D model and an
exact-law steady-state solve agree to about 1 %:

| `lambda_p` (m) | 300 | 500 | 1000 | 2000 |
|---|---|---|---|---|
| glacial relief (m) | 1289 | 803 | 493 | 332 |
| sliding fraction of the flow | 0.32 | 0.58 | 0.87 | 0.97 |

Relief scales as $\lambda_p^{-0.71}$ (the steepness index follows the predicted
−12/19; the glacier also shortens) against $c_e^{-0.45}$, so doubling
`lambda_p` is worth a threefold change in `ce`, and the default trunk, 580 m
thick against `lambda_p` = 300 m, is deformation-dominated. At `lambda_p`
≤ 100 m the steady state is a domain-filling glacier whose relief is set by
the domain length, reached from any initial surface and not comparable with
the values above.

## The Coulomb sliding length `lambda_c`

The regularized Coulomb law is

$$u_b = u_o\,\frac{(\tau/\tau_c)^3}{1 - (\tau/\tau_c)^3},\qquad
u_o = \lambda_c\,\frac{2A_c}{5}\,\tau_c^3 ,$$

so `lambda_c` enters only through the transition speed $u_o$ (32 m/yr at the
defaults). At steady state $(\tau/\tau_c)^3 = u_b^*/(u_o + u_b^*)$, which means
the one dimensionless group that matters is $u_b^*/u_o$. At the Coulomb default
(ν = 2, ℓ = 1, $u_b^*$ = 100 m/yr) it gives $\tau/\tau_c$ = 0.91: the default already
sits in the near-yield regime. A large `lambda_c` recovers a Weertman cubic law
in which the stress falls as $\lambda_c^{-1/3}$; a small `lambda_c` pins the
stress at $\tau_c$. Measured in mode A, cap off (the exact-law solve agrees to
1 %):

| `lambda_c` (m) | 0.1 | 1 | 10 | 30 | 100 | 300 | 1000 | 3000 | 10000 | 100000 |
|---|---|---|---|---|---|---|---|---|---|---|
| $\tau/\tau_c$ | 1.00 | 1.00 | 1.00 | 1.00 | 0.99 | 0.97 | 0.91 | 0.80 | 0.62 | 0.31 |
| glacial relief (m) | 805 | 807 | 804 | 801 | 798 | 789 | 750 | 678 | 565 | 351 |
| sliding fraction of the flow | 0.90 | 0.90 | 0.90 | 0.90 | 0.90 | 0.91 | 0.92 | 0.95 | 0.98 | 1.00 |
| $d\ln(\text{relief})/d\ln c_e$ | −0.30 | −0.30 | −0.30 | −0.30 | −0.32 | −0.35 | −0.40 | −0.46 | −0.51 | −0.56 |

Relief is flat while the stress is pinned (`lambda_c` ≲ 300 m) and then falls
as $\lambda_c^{-2/9}$ toward the Weertman limit. The sliding fraction rises as $\tau/\tau_c$
falls because the sliding speed cannot: the erosion balance pins $u_b$ at
$u_b^*$ whatever `lambda_c` is, and a larger $u_o$ lets the ice reach it at a
lower stress, so the glacier flattens and thins and the deformation speed,
$(2A_c/5)\tau^3 H$, collapses under it. The analytical Coulomb steady
state carries no `lambda_c` at all, so it is right only in the pinned regime.
The sensitivity to `ce` saturates there near −0.30, against the −1/3 the
steady-state theory gives ($c_s \propto (U/c_e)^{2/(3\nu)}$, i.e. $1/(3\ell)$;
the residual is the terminus moving with `ce`) and −5/9 in the Weertman limit and −0.45 for the default power law: yield pinning makes relief
less sensitive to erosional efficiency than any Weertman-type law, not
insensitive: a less erodible bed needs a faster $u_b^*$, which the glacier
reaches by thinning at fixed $\tau_c$, and a thinner glacier is steeper. In mode B `lambda_c` hardly
matters: all five values cycle on the default configuration, with the period
drifting from 274 to 306 kyr across two decades of `lambda_c`.
