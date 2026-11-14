"""Valley <-> sheet reading: the per-cell channel/sheet class of mode D.

Reads every icy cell (``H > REGIME_ICY_H``) as channel or sheet from the
DISCHARGE PER UNIT WIDTH ``q = Q/(alpha_g*H)`` of a separate routing pass over
the plain slab ``zb + H`` (the driver's ``_diagnostic_discharge``), by local
contrast, with every length relative to the glacier's own half-width
``r0 = alpha_g*H/2`` and floors in cells (``constants.REGIME_*``, where the
recipe is stated once):

1. span: ``qb``, the largest ``q`` within ``r0`` of the cell (a disc), and
   ``zmin``, the lowest bed there;
2. ring: on the 8 half-axes, the samples from ``r0 + max(REGIME_RING_GAP_HW*r0,
   REGIME_RING_GAP_CELLS*dx)`` over ``max(REGIME_RING_SPAN_HW*r0,
   REGIME_RING_SPAN_CELLS*dx)``, every in-domain sample counted (bare ground
   is ``q = 0``, ``H = 0``);
3. weak: fewer than ``REGIME_RING_SHARE`` (half) of the ring samples reach
   ``qb/REGIME_WEAK``; strong: fewer than that share reach ``q/REGIME_STRONG``;
4. bed bar (``kB > 0``): ``1 + (ring median zb - zmin)/max(ring median H,
   REGIME_BAR_MIN_H) > kB``, on weak and strong alike;
5. hysteresis: 8-connected weak regions, joined across looped edges, are
   channel when their strong cells number at least ``REGIME_CORE_CELLS`` and
   cover at least ``(alpha_g x their median H)^2``.

:func:`channel_class` returns the binary class; :func:`swap` relaxes the
devices' ``s`` toward it in time and :func:`smooth_icy` builds the datum's
``s``. A port of the prototype that was measured (variant D44 of the private
``.scratch`` harness), operation for operation, except that a region closing
on itself around a looped axis counts its cells once.
"""
import math
import warnings

import numba
import numpy as np
from scipy import ndimage as ndi
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .router import _require_square
from ..constants import (
    REGIME_ICY_H, REGIME_RING_GAP_HW, REGIME_RING_GAP_CELLS,
    REGIME_RING_SPAN_HW, REGIME_RING_SPAN_CELLS, REGIME_WEAK, REGIME_STRONG,
    REGIME_CORE_CELLS, REGIME_SWAP_STEPS, REGIME_DATUM_SIGMA, REGIME_BAR_MIN_H,
    REGIME_RING_SHARE,
)

# (dy, dx) unit steps of the four axis pairs
_AXES = np.array([[0, 1], [1, 0], [1, 1], [1, -1]], dtype=np.int64)
_EIGHT = np.ones((3, 3), dtype=bool)
_RING = np.array([REGIME_RING_GAP_HW, REGIME_RING_GAP_CELLS,
                  REGIME_RING_SPAN_HW, REGIME_RING_SPAN_CELLS], dtype=np.float64)


@numba.njit(cache=True, inline='always')
def _ray_cell(iy, ix, r, sgn, ay, ax, ny, nx, wrap_y, wrap_x):
    """``(jy, jx, inside)``: the cell ``r`` steps along ``sgn*(ay, ax)``,
    wrapped on a looped axis, ``inside`` False past a non-looped edge. The
    modulo (a division per sample) runs only past an edge."""
    jy = iy + sgn * r * ay
    jx = ix + sgn * r * ax
    if jy < 0 or jy >= ny:
        if not wrap_y:
            return jy, jx, False
        jy = jy % ny
    if jx < 0 or jx >= nx:
        if not wrap_x:
            return jy, jx, False
        jx = jx % nx
    return jy, jx, True


@numba.njit(cache=True, inline='always')
def _ring_bounds(r0, dx, ring):
    """``(lo, hi)``: the ring's inner and outer distance from a cell of
    half-width ``r0`` (``ring`` as in :func:`_evidence_kernel`)."""
    lo = r0 + max(ring[0] * r0, ring[1] * dx)
    return lo, lo + max(ring[2] * r0, ring[3] * dx)


@numba.njit(cache=True, inline='always')
def _median_inplace(a, n):
    """The median of ``a[:n]`` (``n >= 1``), REORDERING ``a``: the value of
    numba's ``np.median`` ((lower + upper)/2 for even ``n``) without its copy.
    A quickselect on a branch-free partition around the median of three; the
    values equal to the pivot are split off, so a ring over bare ground
    (H = 0 all round) costs one pass. Any NaN gives NaN, as ``np.median``
    does (the select alone would bury a NaN that never became the pivot)."""
    for i in range(n):
        if a[i] != a[i]:
            return np.nan
    k = n >> 1
    lo = 0
    hi = n - 1
    while lo < hi:
        x = a[lo]
        pivot = a[(lo + hi) >> 1]
        z = a[hi]
        if x > pivot:
            x, pivot = pivot, x
        if pivot > z:
            pivot = z
            if x > pivot:
                pivot = x
        s = lo
        for i in range(lo, hi + 1):             # a[lo:s] < pivot <= a[s:i+1]
            t = a[i]
            a[i] = a[s]
            a[s] = t
            s += t < pivot
        if k < s:
            hi = s - 1
            continue
        e = s
        for i in range(s, hi + 1):              # a[s:e] == pivot < a[e:i+1]
            t = a[i]
            a[i] = a[e]
            a[e] = t
            e += t <= pivot
        if e == s:                              # nothing equals the pivot: NaN
            return np.nan
        if k < e:                               # a[k] is the pivot
            break
        lo = e
    upper = a[k]
    if n & 1:
        return upper
    lower = a[0]                                # a[:k] <= a[k]: its largest
    for i in range(1, k):
        if a[i] > lower:
            lower = a[i]
    return (lower + upper) / 2


@numba.njit(cache=True, parallel=True)
def _evidence_kernel(q, H, zb, dx, alpha_g, kB, icy_h, ring, k_weak, k_strong,
                     min_h, share, wrap_y, wrap_x, axes, nbuf, weak, strong):
    """Fill ``weak`` and ``strong`` (False off the ice) cell by cell: steps 1-4
    of the module docstring. ``ring`` = (gap in half-widths, gap in cells,
    span in half-widths, span in cells); the constants come in as arguments
    so a cached compile never holds stale values."""
    ny, nx = H.shape
    for iy in numba.prange(ny):
        zbuf = np.empty(nbuf)
        hbuf = np.empty(nbuf)
        for ix in range(nx):
            Hi = H[iy, ix]
            weak[iy, ix] = False
            strong[iy, ix] = False
            if not Hi > icy_h:
                continue
            r0 = 0.5 * alpha_g * Hi
            # --- span: the disc of radius r0, row by row ---
            n = int(math.floor(r0 / dx))
            qb = q[iy, ix]
            zmin = zb[iy, ix]
            for a in range(-n, n + 1):
                jy = iy + a
                if jy < 0 or jy >= ny:
                    if not wrap_y:
                        continue
                    jy = jy % ny
                bmax = n                    # the row's half-span: the disc test is monotone in |b|
                while bmax >= 0 and (a * a + bmax * bmax) * dx * dx > r0 * r0:
                    bmax -= 1
                for b in range(-bmax, bmax + 1):
                    jx = ix + b
                    if jx < 0 or jx >= nx:
                        if not wrap_x:
                            continue
                        jx = jx % nx
                    if q[jy, jx] > qb:
                        qb = q[jy, jx]
                    if zb[jy, jx] < zmin:
                        zmin = zb[jy, jx]
            # --- ring: shares reaching qb/weak and q/strong, the medians ---
            lo, hi = _ring_bounds(r0, dx, ring)
            ref_w = qb / k_weak
            ref_s = q[iy, ix] / k_strong
            tot = 0
            cw = 0
            cs = 0
            for a in range(4):
                ay = axes[a, 0]
                ax = axes[a, 1]
                step = dx * math.hypot(ay, ax)
                r_lo = max(int(math.ceil(lo / step)), 1)
                r_hi = max(int(math.ceil(hi / step)) - 1, r_lo)
                for sgn in (1, -1):
                    for r in range(r_lo, r_hi + 1):
                        jy, jx, inside = _ray_cell(iy, ix, r, sgn, ay, ax, ny, nx,
                                                   wrap_y, wrap_x)
                        if not inside:
                            continue
                        qj = q[jy, jx]
                        if qj >= ref_w:
                            cw += 1
                        if qj >= ref_s:
                            cs += 1
                        zbuf[tot] = zb[jy, jx]
                        hbuf[tot] = H[jy, jx] if H[jy, jx] > icy_h else 0.0
                        tot += 1
            # a cell with no ring sample has no evidence of a surrounding sheet
            # (fw = fs = 0); with the bar on it is weighed against its own bed
            # and thickness, 1 + (zb - zmin)/max(H, min_h) > kB, which for
            # kB > 1 vetoes it unless the lowest bed it spans lies more than
            # (kB - 1)*H below its own (as the measured prototype did)
            fw = cw / tot if tot > 0 else 0.0
            fs = cs / tot if tot > 0 else 0.0
            ok = True
            if kB > 0.0 and (fw < share or fs < share):     # the bar can only veto
                if tot > 0:
                    mzb = _median_inplace(zbuf, tot)
                    mH = _median_inplace(hbuf, tot)
                else:
                    mzb = zb[iy, ix]
                    mH = Hi
                ok = 1.0 + (mzb - zmin) / max(mH, min_h) > kB
            weak[iy, ix] = ok and fw < share
            strong[iy, ix] = ok and fs < share


def _ring_reach(H, dx, alpha_g, wrap_y, wrap_x):
    """An upper bound on any icy cell's ring sample count (the ring grows with
    H, so the thickest icy cell bounds it; NaN cells are not icy). Warns where
    that ring reaches past half a looped axis and so wraps back onto the cell
    (the static guard, :func:`check_regime_grid`, cannot see the thickness)."""
    r0 = 0.5 * alpha_g * float(np.max(np.where(H > REGIME_ICY_H, H, 0.0), initial=0.0))
    lo, reach = _ring_bounds(r0, dx, _RING)
    for looped, n, name in ((wrap_y, H.shape[0], 'y'), (wrap_x, H.shape[1], 'x')):
        if looped and n * dx <= 2.0 * reach:
            warnings.warn(
                f"mode 'D': the thickest ice reads a ring more than half the looped {name} axis "
                f"({n * dx / 1e3:g} km) out: its rays wrap back onto the cell and the "
                "channel/sheet class there is unreliable.", stacklevel=3)
    return 8 * (int((reach - lo) / dx) + 3)


def _join_seams(lab, n, wrap_y, wrap_x):
    """Relabel ``lab`` (``ndi.label`` output, ``n`` regions) so regions that
    touch across a looped edge (8-connected, the corner included) share one
    label."""
    if not (wrap_y or wrap_x) or n < 2:
        return lab
    ny, nx = lab.shape
    pairs = []
    for d in (-1, 0, 1):
        if wrap_y:                       # row ny-1 meets row 0
            x = np.arange(nx)
            xb = x + d
            keep = np.ones(nx, bool) if wrap_x else (xb >= 0) & (xb < nx)
            pairs.append((lab[-1, x[keep]], lab[0, xb[keep] % nx]))
        if wrap_x:                       # column nx-1 meets column 0
            y = np.arange(ny)
            yb = y + d
            keep = np.ones(ny, bool) if wrap_y else (yb >= 0) & (yb < ny)
            pairs.append((lab[y[keep], -1], lab[yb[keep] % ny, 0]))
    a = np.concatenate([p[0] for p in pairs])
    b = np.concatenate([p[1] for p in pairs])
    link = (a > 0) & (b > 0) & (a != b)
    if not link.any():
        return lab
    # components are numbered in node order, so the background (node 0, no
    # links) stays 0
    graph = coo_matrix((np.ones(int(link.sum())), (a[link], b[link])), shape=(n + 1, n + 1))
    return connected_components(graph, directed=False)[1][lab]


def _hysteresis(strong, weak, H, dx, alpha_g, wrap_y, wrap_x):
    """Step 5: the weak regions (strong cells included) kept as channel."""
    region = weak | strong
    lab, n = ndi.label(region, structure=_EIGHT)
    lab = _join_seams(lab, n, wrap_y, wrap_x)
    cnt = np.bincount(lab[strong], minlength=n + 1).astype(np.float64)
    keep = np.zeros(n + 1, dtype=bool)
    cand = np.flatnonzero(cnt >= REGIME_CORE_CELLS)   # never 0: no strong cell is bare
    if cand.size:
        med = np.asarray(ndi.median(H[strong], labels=lab[strong], index=cand))
        keep[cand] = cnt[cand] * dx * dx >= (alpha_g * med) ** 2
    return keep[lab]


def channel_class(q, H, zb, dx, alpha_g, bed_bar, wrap_y, wrap_x):
    """The binary class of one step: True where an icy cell reads channel.

    ``q`` is the diagnostic discharge per unit width (0 off the ice), ``H`` the
    committed thickness, ``zb`` the bed the reading sees, ``bed_bar`` the dial
    ``kB`` (0 = no bar). Square cells of side ``dx``; ``wrap_y`` / ``wrap_x``
    wrap the rays, the discs and the regions on looped axes."""
    H = np.ascontiguousarray(H, dtype=np.float64)
    q = np.ascontiguousarray(q, dtype=np.float64).reshape(H.shape)
    zb = np.ascontiguousarray(zb, dtype=np.float64).reshape(H.shape)
    dx, alpha_g = float(dx), float(alpha_g)
    weak = np.empty(H.shape, dtype=np.bool_)
    strong = np.empty(H.shape, dtype=np.bool_)
    nbuf = _ring_reach(H, dx, alpha_g, wrap_y, wrap_x)
    _evidence_kernel(q, H, zb, dx, alpha_g, float(bed_bar), REGIME_ICY_H, _RING,
                     REGIME_WEAK, REGIME_STRONG, REGIME_BAR_MIN_H, REGIME_RING_SHARE, bool(wrap_y),
                     bool(wrap_x), _AXES, nbuf, weak, strong)
    return _hysteresis(strong, weak, H, dx, alpha_g, wrap_y, wrap_x)   # icy cells only


def swap(s_prev, icy_prev, chan, icy):
    """The devices' ``s`` after this step's reading: ``c = 1`` except on icy
    sheet cells (0), and ``s`` moves ``1/REGIME_SWAP_STEPS`` of the way from
    ``s_prev`` to ``c`` where the cell was icy on the last reading and is now;
    ``s = c`` elsewhere (so a first reading, ``icy_prev`` all False, is ``c``)."""
    c = np.where(icy & ~chan, 0.0, 1.0)
    return np.where(icy_prev & icy, s_prev + (1.0 / REGIME_SWAP_STEPS) * (c - s_prev), c)


def smooth_icy(s, icy, wrap_y, wrap_x):
    """``s`` Gaussian-smoothed over the icy cells (sigma ``REGIME_DATUM_SIGMA``
    cells, normalised by the icy weight, wrapped on looped axes, zero beyond
    the others); cells off the ice keep their ``s``."""
    w = icy.astype(np.float64)
    modes = ('wrap' if wrap_y else 'constant', 'wrap' if wrap_x else 'constant')
    num = ndi.gaussian_filter(s * w, REGIME_DATUM_SIGMA, mode=modes, cval=0.0)
    den = ndi.gaussian_filter(w, REGIME_DATUM_SIGMA, mode=modes, cval=0.0)
    return np.where(icy & (den > 0), num / np.where(den > 0, den, 1.0), s)


def check_regime_grid(ny, nx, dx, wrap_y, wrap_x, *, dy=None, widening_factor=None):
    """Mode D's grid guards. Square cells (the reading steps ``dx`` along both
    axes and the diagonals; ``dy`` left ``None`` is not checked); a FINITE carve
    widening factor (the carve scales its descent factor (1 + eta) by the
    source's s, which an instant (inf) factor cannot carry; ``None`` = no
    carve); and every looped axis longer than twice the ring's reach in cells,
    ``REGIME_RING_GAP_CELLS + REGIME_RING_SPAN_CELLS`` (at exactly twice, the
    two ends of an axis pair sample the same cell). Thick ice, whose ring scales with r0, can still alias on a
    looped axis shorter than about ``10*r0``: :func:`channel_class` warns at run
    time."""
    _require_square("mode='D'", dx, dy, "the channel/sheet reading is sized from dx alone")
    if widening_factor is not None and not np.isfinite(widening_factor):
        raise ValueError(
            "mode='D' requires a finite widening_rate: the valley <-> sheet map "
            "scales the carve's descent factor (1 + eta) by the source's "
            "confinedness, and an instant (inf) factor cannot be scaled.")
    reach = REGIME_RING_GAP_CELLS + REGIME_RING_SPAN_CELLS
    for looped, n, name in ((wrap_y, ny, 'y'), (wrap_x, nx, 'x')):
        if looped and n <= 2 * reach:
            raise ValueError(
                f"mode 'D' needs every looped axis at least {2 * reach + 1} cells "
                f"long (more than twice the channel/sheet reading's ring reach); the looped "
                f"{name} axis has {n}, so its rays would wrap onto themselves. "
                "Lengthen it or make it non-looped.")
