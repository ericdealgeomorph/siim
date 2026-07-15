"""How mode-B width carving registers a glacier's ice — in plan and in the
vertical, and how the carve shapes the bed over time (docs).

A one-cell-wide glacier of mean thickness H occupies a valley of width
W = alpha_g*H, so where does the model put that ice, and how does the trough
grow? This is the single carve figure for the docs.

Panels:
  (a) PLAN — the registered ice area is the footprint {D < 0}, D the minimum
      power distance over the discs R = alpha_g*H/2 (MAT inversion + power
      diagram). Same scalar field the carve and the renderer both threshold.
  (b) CROSS-SECTION (A-A') — the duality. The carve writes a parabolic BED
      (model state); the renderer registers a transversely FLAT ice surface
      z_s = z_b(source) + hc*H (display). Both are hung from the SAME footprint;
      the ice column H(d) = z_s - z_b(d) = hc*H*(1 - (d/R)^2) is parabolic, zero
      at the rim d = R, hc*H at the center, and means to exactly H.
  (c) THE RULE, over a GROWING glacier — as H grows the footprint widens and
      the bed descends onto the parabola, step by step, by the exact carve rule
      zb <- min(zb, max(z_target, zb_pre - (1+eta)*E_c*dt)). Illustrates all
      three of its behaviours at once: rate-capped descent (early snapshots and
      the just-widened rim lag the parabola from above), no surface gate (flanks
      standing above z_s are consumed), and bed memory (a relict overdeepening
      below the parabola is kept untouched).

Run from the repo root:
    python docs/figures/make_ice_surface_registration_figure.py
Writes ice_surface_registration.png/.pdf next to this script.
"""
import pathlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from siim.constants import HC_OVER_H, ALPHA_G
from siim._core.carve import PDT_NO_SOURCE, _power_dt_2d

OUT = pathlib.Path(__file__).parent
plt.rcParams.update({'font.size': 9, 'axes.titlesize': 10})

ICE = '#cfe3f2'
ICE_EDGE = '#3a6ea5'
BED = '#16213e'
GOLD = 'goldenrod'


# ---------------------------------------------------------------- panel a
def build_scene():
    """A single sinuous trunk that widens downstream, on a unit grid. The
    cross-section column ``xc`` is placed on a locally straight, widest reach
    so the plan footprint there is a clean band of half-width R."""
    ny, nx = 70, 120
    xc = 84                                   # cross-section column (wide reach)

    def d8_path(x0, x1, yfun):
        cells, y_prev = [], int(round(yfun(x0)))
        for x in range(x0, x1 + 1):
            y = int(round(yfun(x)))
            step = 1 if y > y_prev else -1
            for yy in (range(y_prev + step, y + step, step) if y != y_prev else []):
                cells.append((yy, x))
            cells.append((y, x))
            y_prev = y
        return cells

    # gentle meander, locally flat (dy/dx ~ 0) near x = xc
    trunk = d8_path(8, 112, lambda x: 35 + 11 * np.sin((x - 84) / 26.0))
    offsets = np.full((ny, nx), PDT_NO_SOURCE)
    R_of = np.zeros((ny, nx))
    H_src = 0.20                              # thickness in grid units (R = a_g H/2)
    for k, (y, x) in enumerate(trunk):
        H = H_src * (3.0 + 6.0 * k / max(len(trunk) - 1, 1))   # widens downstream
        R = 0.5 * ALPHA_G * H
        offsets[y, x] = -R * R
        R_of[y, x] = R
    D = np.empty((ny, nx)); SRC = np.empty((ny, nx), np.int64)
    _power_dt_2d(offsets, 1.0, 1.0, D, SRC)
    # source thickness/radius on the cross-section column
    yc = int(round(35 + 11 * np.sin((xc - 84) / 26.0)))
    return dict(ny=ny, nx=nx, xc=xc, yc=yc, trunk=trunk,
                offsets=offsets, R_of=R_of, D=D, R_cs=R_of[yc, xc])


def panel_a(ax, sc):
    D, trunk, R_of = sc['D'], sc['trunk'], sc['R_of']
    ax.contourf(D, levels=[-1e9, 0.0], colors=[ICE])
    ax.contour(D, levels=[0.0], colors=[ICE_EDGE], linewidths=1.4)
    py, px = zip(*trunk)
    ax.plot(px, py, color=BED, lw=1.2)
    th = np.linspace(0, 2 * np.pi, 80)
    for k in range(6, len(trunk), 16):
        y, x = trunk[k]; R = R_of[y, x]
        ax.plot(x + R * np.cos(th), y + R * np.sin(th),
                color=ICE_EDGE, lw=0.6, alpha=0.5)
    xc, yc, R = sc['xc'], sc['yc'], sc['R_cs']
    ax.plot([xc, xc], [yc - R, yc + R], color='red', lw=1.6)
    ax.annotate("A", (xc, yc + R), (xc, yc + R + 6), color='red',
                ha='center', fontsize=9)
    ax.annotate("A'", (xc, yc - R), (xc, yc - R - 8), color='red',
                ha='center', fontsize=9)
    ax.text(2, 3, r'ice area $= \{x:\ \min_i(\|x-c_i\|^2 - R_i^2) < 0\}$'
                  '\n(footprint = union of discs)',
            fontsize=8, color=ICE_EDGE)
    ax.annotate(r'one-cell centerline', xy=(40, 39),
                xytext=(20, 60), arrowprops=dict(arrowstyle='-', lw=0.7),
                fontsize=8)
    ax.annotate(r'discs $R=\alpha_g H/2$', xy=(58, 26),
                xytext=(58, 4), arrowprops=dict(arrowstyle='-', lw=0.7),
                fontsize=8)
    ax.set_title('(a)  plan: where the ice area is registered', loc='left')
    ax.set_xlim(0, sc['nx'] - 1); ax.set_ylim(0, sc['ny'] - 1)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])


# ---------------------------------------------------------------- panel b
def panel_b(ax, sc):
    hc, H = HC_OVER_H, 300.0
    R = 0.5 * ALPHA_G * H                      # half-width in metres
    zb_s = 0.0                                 # source (centerline) bed
    zs = zb_s + hc * H                         # flat ice surface = trimline
    d = np.linspace(-1.3 * R, 1.3 * R, 601)
    inside = np.abs(d) <= R
    parab = zb_s + hc * H * (d / R) ** 2        # carved bed (model state)
    walls = zs + (np.abs(d) - R) / R * 0.9 * hc * H   # rock walls outside R
    bed = np.where(inside, parab, walls)
    y0 = zb_s - 80.0

    # rock (everything below the bed) and ice (bed -> flat surface inside R)
    ax.fill_between(d / 1e3, y0, bed, color='#e7e2d8')
    ax.fill_between(d / 1e3, parab, zs, where=inside, color=ICE, alpha=0.95)
    ax.plot(d[inside] / 1e3, parab[inside], color=BED, lw=2.2,
            label=r'carved bed $z_b(d)$  (model state)')
    ax.plot(np.where(inside, d, np.nan) / 1e3, np.where(inside, zs, np.nan),
            color=GOLD, lw=2.2, label=r'ice surface $z_s$ — flat  (display)')
    ax.plot(d[~inside] / 1e3, walls[~inside], color='#8a8170', lw=1.3,
            label='valley wall (rock)')
    for sgn in (-1, 1):
        ax.plot(sgn * R / 1e3, zs, 'o', color=GOLD, ms=4)

    # the ice column H(d) at one offset, as a vertical bracket
    db = 0.5 * R
    ax.annotate('', xy=(db / 1e3, zs),
                xytext=(db / 1e3, zb_s + hc * H * (db / R) ** 2),
                arrowprops=dict(arrowstyle='<->', lw=0.9, color='#27496d'))
    ax.text(db / 1e3 + 0.12, 0.5 * (zs + zb_s + hc * H * (db / R) ** 2),
            r'$H(d)=h_cH\,(1-(d/R)^2)$', fontsize=7.5, color='#27496d')
    ax.annotate('rim: $d=R$, $H=0$', xy=(R / 1e3, zs),
                xytext=(R / 1e3 - 0.05, zs + 150),
                arrowprops=dict(arrowstyle='->', lw=0.8), fontsize=8, ha='center')
    ax.annotate('center: $H=h_cH$\n(mean over $[-R,R]=H$)', xy=(0, zb_s),
                xytext=(-1.28 * R / 1e3, zb_s + 95),
                arrowprops=dict(arrowstyle='->', lw=0.8), fontsize=8)
    ax.set_xlabel('across-valley $d$ (km)')
    ax.set_ylabel('elevation (m)')
    ax.set_ylim(y0, zs + 230)
    ax.legend(fontsize=7, loc='upper right', framealpha=0.92)
    ax.set_title('(b)  cross-section A-A$\'$: same footprint, bed = parabola, '
                 'surface = flat', loc='left')


# ---------------------------------------------------------------- panel c
def panel_c(ax):
    """The carve rule on a GROWING glacier. As H grows the footprint widens and
    the bed descends onto the parabola, step by step, by the exact rule
    ``zb <- min(zb, max(z_target, zb_pre-(1+eta)*Ec*dt))``: rate-capped (early
    snapshots and the just-widened rim lag the parabola from above), no surface
    gate (flanks standing above z_s are consumed), and a relict overdeepening
    below the parabola is kept (bed memory)."""
    hc = HC_OVER_H
    n = 561
    d = np.linspace(-3.4e3, 3.4e3, n)
    # Pre-glacial terrain: a broad valley whose flanks stand above the eventual
    # ice surface (so the no-gate carve consumes them), plus a RELICT
    # overdeepening on the left flank, below the final parabola (kept).
    zb = 720.0 - 250.0 * np.exp(-(d / 2600.0) ** 2)
    zb -= 360.0 * np.exp(-((d + 1250.0) / 230.0) ** 2)   # relict overdeepening
    zb0 = zb.copy()

    Ec, dt, nsteps, eta = 1.1e-3, 1000.0, 460, 1.0     # source incision; widening eta
    H0, Hmax = 60.0, 300.0
    zc = 470.0                                          # channel floor (source bed)
    order = [('60 kyr', 0.20, '#9aa0b4'),
             ('150 kyr', 0.50, '#5a6184'),
             ('300 kyr', 1.0, '#101a3a')]
    snaps = {}
    for it in range(nsteps):
        frac = it / (nsteps - 1)
        H = H0 + (Hmax - H0) * min(1.0, 1.3 * frac)     # grows, then holds
        R = 0.5 * ALPHA_G * H
        zc -= Ec * dt
        target = zc + hc * H * (d / R) ** 2
        foot = np.abs(d) < R
        zb_pre = zb.copy()
        cap = zb_pre - (1.0 + eta) * Ec * dt            # rate cap from the pre-step bed
        zb[foot] = np.minimum(zb[foot], np.maximum(target[foot], cap[foot]))
        zb[np.abs(d) < 0.02 * R] = zc                   # resolved channel (self no-op)
        for name, fr, _ in order:
            if it == int(fr * nsteps) - 1:
                snaps[name] = (zb.copy(), R, zc, H)

    zb_fin, R_fin, zc_fin, H_fin = snaps['300 kyr']
    zs = zc_fin + hc * H_fin
    foot_fin = np.abs(d) < R_fin
    ax.fill_between(d / 1e3, zb_fin, zs, where=foot_fin & (zb_fin < zs),
                    color=ICE, alpha=0.85, lw=0, label='ice (300 kyr)')
    ax.plot(d / 1e3, zb0, color='#b9b2a6', ls='--', lw=1.0,
            label='pre-glacial terrain')
    for name, _, c in order:
        ax.plot(d / 1e3, snaps[name][0], color=c, lw=1.7, label='bed, ' + name)
    tgt = zc_fin + hc * H_fin * (d / R_fin) ** 2
    ax.plot(d[foot_fin] / 1e3, tgt[foot_fin], 'r:', lw=1.4,
            label='parabola (attractor)')
    ax.axhline(zs, color=GOLD, ls=':', lw=1.1, label='ice surface $z_s$')

    ax.annotate('flanks above $z_s$\nconsumed (no gate)', xy=(1.95, zs + 25),
                xytext=(1.05, zs + 235), fontsize=7.5, ha='center',
                arrowprops=dict(arrowstyle='->', lw=0.8))
    ax.annotate('relict overdeepening\nkept (bed memory)',
                xy=(-1.25, float(zb_fin[np.argmin(np.abs(d + 1250))]) + 10),
                xytext=(-3.35, 285), fontsize=7.5,
                arrowprops=dict(arrowstyle='->', lw=0.8))
    ax.annotate('rate-capped: just-widened\nrim lags the parabola',
                xy=(1.4, 410), xytext=(-0.15, zs + 150), fontsize=7.5,
                ha='center', arrowprops=dict(arrowstyle='->', lw=0.8))
    ax.text(-3.35, zs + 330,
            r'$z_b\leftarrow\min(z_b,\ \max(z_{target},\ '
            r'z_b^{\,pre}-(1{+}\eta)E_c\,dt))$  each step',
            fontsize=9)
    ax.set_xlabel('across-valley $d$ (km)')
    ax.set_ylabel('bed elevation (m)')
    ax.set_ylim(zc_fin - 60, zs + 400)
    ax.legend(fontsize=6.6, loc='upper right', ncol=2, framealpha=0.93)
    ax.set_title('(c)  the carve rule over a growing glacier: the trough '
                 'widens + deepens onto the parabola', loc='left')
    ax.grid(alpha=0.2)


def main():
    sc = build_scene()
    fig = plt.figure(figsize=(12.0, 8.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.92, 1.06], hspace=0.34, wspace=0.22)
    panel_a(fig.add_subplot(gs[0, 0]), sc)
    panel_b(fig.add_subplot(gs[0, 1]), sc)
    panel_c(fig.add_subplot(gs[1, :]))
    fig.savefig(OUT / 'ice_surface_registration.png', dpi=180, bbox_inches='tight')
    fig.savefig(OUT / 'ice_surface_registration.pdf', bbox_inches='tight')
    print('wrote', OUT / 'ice_surface_registration.png')


if __name__ == '__main__':
    main()
