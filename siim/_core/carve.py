"""Sub-grid glacier-width carving (mode B only).

Consolidated into the numerical core in the pre-v1.0 rewrite. See
``docs/guides/concepts.md`` for the public description of sub-grid width
carving. numpy/numba only -- no model/fastscape imports.
"""
import numpy as np
import numba

# =============================================================================
# Sub-grid glacier-width carving (mode B only).
#
# The glacier's plan-view footprint is the union of variable-radius discs
# centered on the glaciated channel cells (radius = alpha_g*H/2): the exact
# medial-axis-transform inversion of "centerline + width" (Amenta, Choi &
# Kolluri 2001). Membership and per-cell source attribution come from one
# scalar field, the minimum POWER distance d^2 - R^2 (Aurenhammer 1988,
# Lemma 1: the power diagram decomposes a disc union; the field is negative
# exactly inside it), computed with the Felzenszwalb-Huttenlocher
# generalized distance transform with per-seed offsets -R^2 — two separable
# 1D lower-envelope passes, O(N), exact, argmin labels from the envelope
# bookkeeping (Felzenszwalb & Huttenlocher 2012, Thm 2.2; the same pipeline
# as Coeurjolly & Montanvert 2007's "reverse EDT").
#
# Cells inside the footprint are carved toward the parabolic cross-section
# hung from the source's ICE SURFACE zs_s = zb_s + hc_over_H*H_s: rim exactly at
# the ice surface (the trimline IS the ice surface), floor at the source's
# own bed —
#     z_target = zs_s - hc_over_H*H_s*(1 - (d/R)^2) = zb_s + hc_over_H*H_s*(d/R)^2
# A self-attributed cell's target is therefore its own bed: a structural
# no-op (the skip below is just a shortcut), so resolved channel cells
# keep their kernel erosion and the trough floor meets the channel bed
# smoothly — no centerline spine. With hc_over_H = HC_OVER_H = 3/2 (parabola
# max/mean) the cross-section's mean depth is exactly the model's mean
# depth H: bed-anchored and surface-anchored forms coincide. The disc
# radius R = alpha_g*H/2 stays keyed to the MEAN depth (width = alpha_g*H).
#
# NO surface gate: terrain standing above the source's ice surface is
# consumed too — ridge-eating is the channel-capture mechanism. Reach
# stays local through the power attribution (local sources own their
# surroundings); note this is geometry-dependent, not enforced — a thick
# cell just below a convergent head projects its full disc up-valley.
# Carving follows the doubly one-sided, never-additive, rate-capped rule
#     zb <- min(zb, max(z_target, zb_pre - widening_factor*E_c*dt))
# with E_c the centerline (source) incision this step, zb_pre the cell's bed
# BEFORE this step's erosion kernel (so the kernel's own erosion at the cell
# and the carve arbitrate, whichever is lower — never add), and
# widening_factor = 1 + eta = E_widening/E_c (eta = widening_rate >= 0; the
# footprint descends at E_widening = (1 + eta)*E_c; inf = instant U-shape
# imposition). The rule never deposits, lands on and then tracks the parabola,
# and leaves a bed already below target alone (bed memory). Source beds and
# erosion are read from pre-carve snapshots, so the result is independent of
# the cell visit order (no raster-orientation bias).
#
# Exclusions. Sources that did not erode this step (E_s <= 0: floating /
# lake-decoupled reaches) carve nothing, keeping floating_termini
# semantics. Border (self-receiving) cells are excluded BOTH ways — as
# sources (the caller leaves them out of the seed array so interior
# sources inherit their attribution rather than being shadowed) and as
# targets (their bed belongs to the border budget: flotation floor, datum
# clamp, waterline presentation — see boundary_conditions.md). Looped
# (periodic) boundaries are supported through _power_dt_2d_periodic,
# which wrap-pads the seed array along the looped axes and remaps the
# labels — exact, because only each seed's NEAREST periodic image can
# ever win or compete (see its docstring).
# =============================================================================

# Power-DT sentinel: cells whose offset is >= this are not sources. The
# seeding (_carve_offsets), the transform (_power_dt_1d), and the tests
# must agree on it exactly — never compare against a separate literal.
PDT_NO_SOURCE = 1e30


@numba.njit(cache=True)
def _carve_offsets(H_flat, rec, alpha_g, offsets_flat, seed_mask):
    """Seed array for the width-carve power transform: -R^2 with
    R = alpha_g*H/2 at icy, non-border, seed-allowed cells; PDT_NO_SOURCE
    elsewhere. Border (self-receiving) cells are left out of the seeds so
    interior sources inherit their attribution (see _carve_subgrid_width).

    ``seed_mask`` (per-cell, nonzero = allowed) can de-seed cells that must not
    anchor a disc; the Mode-C carve passes an all-ones mask (every icy interior
    cell seeds a disc). Returns the seed count (0 = nothing this step)."""
    nn = H_flat.shape[0]
    n_seed = 0
    for i in range(nn):
        if H_flat[i] > 0.0 and rec[i] != i and seed_mask[i] != 0:
            r = 0.5 * alpha_g * H_flat[i]
            offsets_flat[i] = -(r * r)
            n_seed += 1
        else:
            offsets_flat[i] = PDT_NO_SOURCE
    return n_seed


@numba.njit(cache=True)
def _power_dt_1d(f, h2, d_out, i_out):
    """1D lower envelope of parabolas: ``d_out[p] = min_q((p-q)^2*h2 + f[q])``,
    ``i_out[p] = argmin q``. Algorithm 1 of
    :cite:t:`felzenszwalbDistanceTransformsSampled2012` with a physical
    spacing-squared factor h2 (anisotropic grids) and +inf-aware
    seeding."""
    n = f.shape[0]
    v = np.empty(n, dtype=np.int64)
    z = np.empty(n + 1, dtype=np.float64)
    INF = PDT_NO_SOURCE
    k = 0
    q0 = 0
    while q0 < n and f[q0] >= INF:
        q0 += 1
    if q0 == n:                       # no sources in this line
        for p in range(n):
            d_out[p] = INF
            i_out[p] = -1
        return
    v[0] = q0
    z[0] = -INF
    z[1] = INF
    for q in range(q0 + 1, n):
        if f[q] >= INF:
            continue
        while True:
            p = v[k]
            s = ((f[q] + q * q * h2) - (f[p] + p * p * h2)) / (2.0 * h2 * (q - p))
            if s <= z[k]:
                k -= 1
            else:
                k += 1
                v[k] = q
                z[k] = s
                z[k + 1] = INF
                break
    k = 0
    for p in range(n):
        while z[k + 1] < p:
            k += 1
        q = v[k]
        d_out[p] = (p - q) * (p - q) * h2 + f[q]
        i_out[p] = q


@numba.njit(cache=True, parallel=True)
def _power_dt_2d(offsets, dy, dx, D, SRC):
    """Minimum power distance D[y,x] = min_src(dist^2 - R_src^2) over the
    grid (physical units, anisotropic spacing), and SRC = the flat index of
    the argmin source cell (-1 outside any line of sources). offsets holds
    -R^2 at source cells and >=1e30 elsewhere."""
    ny, nx = offsets.shape
    dy2 = dy * dy
    dx2 = dx * dx
    tmp = np.empty((ny, nx), dtype=np.float64)
    tmpi = np.empty((ny, nx), dtype=np.int64)
    for x in numba.prange(nx):
        f = np.empty(ny, dtype=np.float64)
        d = np.empty(ny, dtype=np.float64)
        ii = np.empty(ny, dtype=np.int64)
        for y in range(ny):
            f[y] = offsets[y, x]
        _power_dt_1d(f, dy2, d, ii)
        for y in range(ny):
            tmp[y, x] = d[y]
            tmpi[y, x] = ii[y]
    for y in numba.prange(ny):
        f = np.empty(nx, dtype=np.float64)
        d = np.empty(nx, dtype=np.float64)
        ii = np.empty(nx, dtype=np.int64)
        for x in range(nx):
            f[x] = tmp[y, x]
        _power_dt_1d(f, dx2, d, ii)
        for x in range(nx):
            D[y, x] = d[x]
            xs = ii[x]
            # xs >= 0 implies column xs held a source (INF columns never
            # enter the envelope), so tmpi[y, xs] >= 0 there.
            if xs < 0:
                SRC[y, x] = -1
            else:
                SRC[y, x] = tmpi[y, xs] * nx + xs


def _power_dt_2d_periodic(offsets, dy, dx, D, SRC, wrap_y, wrap_x):
    """Periodic-aware power transform: wrap-pad the seed array along the
    looped axes, run the ordinary FH transform on the padded grid, crop
    the central window, and remap the argmin labels to original flat
    indices (numpy wrapper; the njit kernel is untouched).

    EXACT for the carve, by two arguments. (1) Only each seed's NEAREST
    periodic image matters: farther images of the same seed have the same
    R and larger d, hence strictly worse power. (2) Footprint membership
    and in-footprint attribution only involve seeds with d < R <= R_max
    (a footprint cell's winner satisfies it, and any competitor that
    could beat the winner there satisfies it too), so padding by
    ceil(R_max/spacing) cells covers every image that can win; the
    half-circumference cap covers the R_max > L/2 regime, where the
    nearest image always lies within half a wrap. Cells outside every
    footprint (D >= 0) may in principle miss a remote image, but the
    carve skips them — and D < 0 vs >= 0 itself is exact by (2).
    """
    ny, nx = offsets.shape
    seeded = offsets < PDT_NO_SOURCE
    if not seeded.any():
        D.fill(PDT_NO_SOURCE)
        SRC.fill(-1)
        return
    r_max = float(np.sqrt(-offsets[seeded].min()))
    py = min(int(np.ceil(r_max / dy)) + 1, ny // 2 + 1) if wrap_y else 0
    px = min(int(np.ceil(r_max / dx)) + 1, nx // 2 + 1) if wrap_x else 0
    if py == 0 and px == 0:
        _power_dt_2d(offsets, dy, dx, D, SRC)
        return
    off_p = np.pad(offsets, ((py, py), (px, px)), mode='wrap')
    nxp = off_p.shape[1]
    D_p = np.empty_like(off_p)
    SRC_p = np.empty(off_p.shape, dtype=np.int64)
    _power_dt_2d(off_p, dy, dx, D_p, SRC_p)
    D[:, :] = D_p[py:py + ny, px:px + nx]
    sp = SRC_p[py:py + ny, px:px + nx]
    valid = sp >= 0
    yp, xp = np.divmod(sp[valid], nxp)
    SRC.fill(-1)
    SRC[valid] = ((yp - py) % ny) * nx + ((xp - px) % nx)


@numba.njit(cache=True)
def _carve_subgrid_width(zb_flat, zb_kern, zb_pre, H_flat, surface_out,
                         rec, D, SRC, offsets, widening_factor, hc_over_H):
    """Apply the sub-grid width carve (see block comment above) to every
    footprint cell whose bed stands above its target — bare valley walls
    AND thin-ice cells inside a bigger glacier's footprint ("nodes that
    were in a glacier and didn't know it": under broad ice cover nearly
    the whole footprint is icy, and skipping icy cells starves the carve).

    Parameters
    ----------
    zb_flat : ndarray
        Bed, MODIFIED IN PLACE (post-kernel on entry).
    zb_kern : ndarray
        Snapshot of zb_flat at carve entry (post-kernel, pre-carve) —
        source anchors are read here, so the result is independent
        of cell visit order even when sources are themselves carved.
    zb_pre : ndarray
        Bed before this step's erosion kernel — descent caps are
        measured from here, so the kernel's own erosion at a cell
        and the carve never add (whichever is lower wins).
    offsets : ndarray
        The -R^2 seed array the power transform consumed — R^2 is
        read back from it (R2 = -offsets[s]) so footprint definition
        and carve targets cannot disagree.

    A self-attributed cell's target is its own bed — a structural no-op
    (the skip is a shortcut); border (self-receiving) cells are skipped as
    targets (border budget's business) and must not be seeded as sources;
    sources with no kernel erosion this step (floating / decoupled) carve
    nothing. Updates surface_out in place for carved cells (bare: bed;
    icy: bed + hc_over_H*H — the presented surface drops with the bed)."""
    nn = zb_flat.shape[0]
    for i in range(nn):
        if D.flat[i] >= 0.0:                  # outside the footprint
            continue
        s = SRC.flat[i]
        if s < 0 or s == i:                   # unattributed / self: no-op
            continue
        if rec[s] == s or rec[i] == i:        # border source or target
            continue
        E_dt = zb_pre[s] - zb_kern[s]         # the kernel's bed drop at s
        if E_dt <= 0.0:                       # floating / no erosion at source
            continue
        # No surface gate: terrain standing above the source's ice surface
        # IS consumed (ridge-eating is the capture mechanism — the whole
        # point); the parabola rims AT the ice surface zs_s, so the
        # trimline is the ice surface itself.
        H_s = H_flat[s]
        R2 = -offsets.flat[s]                 # the seed's own R^2 (> 0)
        d2 = D.flat[i] + R2                   # squared distance to source
        target = zb_kern[s] + hc_over_H * H_s * (d2 / R2)
        lowered = zb_pre[i] - widening_factor * E_dt
        if target > lowered:
            lowered = target
        if lowered < zb_flat[i]:
            zb_flat[i] = lowered
            surface_out[i] = lowered + hc_over_H * H_flat[i]  # bare: bed; icy: bed + hc_over_H*H
