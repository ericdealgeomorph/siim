"""Shared rendering backend for the siim plotting package.

Hosts the one notebook slider viewer (``_slider_view``), the colorbar helper,
and the numba glacier-field rendering backend — rasterizers, soft hillshade,
glacier-path tracing, priority-flood lakes, variable-sigma smoothing, the
glacier-field builder, and the footprint ice-surface / smoothed-mask helpers —
relocated here from the old ``siim2d_plotting`` module. See
docs/dev/plotting_plan.md.

numpy/numba/scipy + the carve kernels at import; matplotlib/ipywidgets stay
lazy inside the viewer/colorbar so the slider path imports light.
"""

import numpy as np
import numba
from types import SimpleNamespace
from scipy.ndimage import map_coordinates, gaussian_filter

from .._output import output_path
from ..constants import HC_OVER_H
from .._core.carve import _carve_offsets, _power_dt_2d, _power_dt_2d_periodic

__all__ = [
    "_slider_view", "_profile_slider", "_add_colorbar", "output_path",
    "_compute_glacier_field", "_trace_paths_arrays", "_footprint_ice_surface",
    "_smooth_ice_mask", "_field_ice_mask", "_clean_ice_mask", "_mean_recent_H",
    "_priority_flood", "_shade_rgb_soft",
]


# =====================================================================
# Glacier-field rendering backend (relocated verbatim from siim2d_plotting)
# =====================================================================

@numba.njit(cache=True)
def _rasterize_glacier_segments(H_raster, seg_x1, seg_y1, seg_x2, seg_y2,
                                seg_h1, seg_h2, seg_w1, seg_w2,
                                nx, ny, dx_grid, dy_grid):
    """Rasterize parabolic glacier cross-sections onto H_raster, keeping the
    max H at each grid node. (Bedrock imprints are computed downstream as
    ``z_upsampled − H_raster`` rather than in a separate kernel.)
    """
    n_seg = seg_x1.shape[0]
    for s in range(n_seg):
        x1, y1, h1, w1 = seg_x1[s], seg_y1[s], seg_h1[s], seg_w1[s]
        x2, y2, h2, w2 = seg_x2[s], seg_y2[s], seg_h2[s], seg_w2[s]
        half_w_max = max(w1, w2) / 2.0

        seg_dx = x2 - x1
        seg_dy = y2 - y1
        seg_len = np.sqrt(seg_dx * seg_dx + seg_dy * seg_dy)
        if seg_len == 0:
            continue
        tx = seg_dx / seg_len
        ty = seg_dy / seg_len

        margin = half_w_max + max(dx_grid, dy_grid)
        r_min = max(0, int((min(y1, y2) - margin) / dy_grid))
        r_max = min(ny - 1, int((max(y1, y2) + margin) / dy_grid) + 1)
        c_min = max(0, int((min(x1, x2) - margin) / dx_grid))
        c_max = min(nx - 1, int((max(x1, x2) + margin) / dx_grid) + 1)

        for r in range(r_min, r_max + 1):
            gy = r * dy_grid
            for c in range(c_min, c_max + 1):
                gx = c * dx_grid
                dx_p = gx - x1
                dy_p = gy - y1
                along = dx_p * tx + dy_p * ty
                if along < 0 or along > seg_len:
                    continue
                perp = abs(-dx_p * ty + dy_p * tx)
                t = along / seg_len
                half_w = (w1 * (1 - t) + w2 * t) / 2.0
                if half_w <= 0 or perp > half_w:
                    continue
                h_interp = h1 * (1 - t) + h2 * t
                h_here = h_interp * (1 - (perp / half_w) ** 2)
                idx = r * nx + c
                if h_here > H_raster[idx]:
                    H_raster[idx] = h_here


@numba.njit(cache=True)
def _rasterize_glacier_vertices(H_raster, vx, vy, vh, vw,
                                nx, ny, dx_grid, dy_grid):
    """Rasterize parabolic disks at path vertices. Fills bend-gaps on the
    outside of sharp turns and rounds off glacier heads and termini where
    the per-segment rectangular coverage leaves slivers uncovered.
    """
    n_v = vx.shape[0]
    for s in range(n_v):
        xv, yv, hv, wv = vx[s], vy[s], vh[s], vw[s]
        half_w = wv / 2.0
        if half_w <= 0 or hv <= 0:
            continue
        margin = half_w + max(dx_grid, dy_grid)
        r_min = max(0, int((yv - margin) / dy_grid))
        r_max = min(ny - 1, int((yv + margin) / dy_grid) + 1)
        c_min = max(0, int((xv - margin) / dx_grid))
        c_max = min(nx - 1, int((xv + margin) / dx_grid) + 1)
        for r in range(r_min, r_max + 1):
            gy = r * dy_grid
            for c in range(c_min, c_max + 1):
                gx = c * dx_grid
                dxp = gx - xv
                dyp = gy - yv
                d = np.sqrt(dxp * dxp + dyp * dyp)
                if d >= half_w:
                    continue
                h_here = hv * (1 - (d / half_w) ** 2)
                idx = r * nx + c
                if h_here > H_raster[idx]:
                    H_raster[idx] = h_here


@numba.njit(cache=True)
def _rasterize_glacier_segments_HU(H_raster, U_raster,
                                   seg_x1, seg_y1, seg_x2, seg_y2,
                                   seg_h1, seg_h2, seg_w1, seg_w2,
                                   seg_V1, seg_V2,
                                   nx, ny, dx_grid, dy_grid):
    """Rasterize parabolic depth (H) and cross-section-average velocity (V).

    H_raster gets the local parabolic depth at each cell, max-on-h composited
    across overlapping segments. U_raster gets the cross-section-average ice
    velocity V interpolated linearly along the segment — uniform across the
    cross-section. V at each path node is computed upstream via mass
    conservation V = Q_g / (α_g · H̄²), so this is independent of sliding-
    law parameters and free of the (τ/τ_c)³ pole that the local-h sliding-law
    formula has.
    """
    n_seg = seg_x1.shape[0]
    for s in range(n_seg):
        x1, y1, h1, w1 = seg_x1[s], seg_y1[s], seg_h1[s], seg_w1[s]
        x2, y2, h2, w2 = seg_x2[s], seg_y2[s], seg_h2[s], seg_w2[s]
        V1, V2 = seg_V1[s], seg_V2[s]
        half_w_max = max(w1, w2) / 2.0

        seg_dx = x2 - x1
        seg_dy = y2 - y1
        seg_len = np.sqrt(seg_dx * seg_dx + seg_dy * seg_dy)
        if seg_len == 0:
            continue
        tx = seg_dx / seg_len
        ty = seg_dy / seg_len

        margin = half_w_max + max(dx_grid, dy_grid)
        r_min = max(0, int((min(y1, y2) - margin) / dy_grid))
        r_max = min(ny - 1, int((max(y1, y2) + margin) / dy_grid) + 1)
        c_min = max(0, int((min(x1, x2) - margin) / dx_grid))
        c_max = min(nx - 1, int((max(x1, x2) + margin) / dx_grid) + 1)

        for r in range(r_min, r_max + 1):
            gy = r * dy_grid
            for c in range(c_min, c_max + 1):
                gx = c * dx_grid
                dx_p = gx - x1
                dy_p = gy - y1
                along = dx_p * tx + dy_p * ty
                if along < 0 or along > seg_len:
                    continue
                perp = abs(-dx_p * ty + dy_p * tx)
                t = along / seg_len
                half_w = (w1 * (1 - t) + w2 * t) / 2.0
                if half_w <= 0 or perp > half_w:
                    continue
                h_interp = h1 * (1 - t) + h2 * t
                h_here = h_interp * (1 - (perp / half_w) ** 2)
                idx = r * nx + c
                if h_here > H_raster[idx]:
                    H_raster[idx] = h_here
                    U_raster[idx] = V1 * (1 - t) + V2 * t


@numba.njit(cache=True)
def _rasterize_glacier_vertices_HU(H_raster, U_raster,
                                   vx, vy, vh, vw, vV,
                                   nx, ny, dx_grid, dy_grid):
    """Per-vertex disk variant of _rasterize_glacier_segments_HU.
    Uses the vertex's V directly within its parabolic disk.
    """
    n_v = vx.shape[0]
    for s in range(n_v):
        xv, yv, hv, wv, Vv = vx[s], vy[s], vh[s], vw[s], vV[s]
        half_w = wv / 2.0
        if half_w <= 0 or hv <= 0:
            continue
        margin = half_w + max(dx_grid, dy_grid)
        r_min = max(0, int((yv - margin) / dy_grid))
        r_max = min(ny - 1, int((yv + margin) / dy_grid) + 1)
        c_min = max(0, int((xv - margin) / dx_grid))
        c_max = min(nx - 1, int((xv + margin) / dx_grid) + 1)
        for r in range(r_min, r_max + 1):
            gy = r * dy_grid
            for c in range(c_min, c_max + 1):
                gx = c * dx_grid
                dxp = gx - xv
                dyp = gy - yv
                d = np.sqrt(dxp * dxp + dyp * dyp)
                if d >= half_w:
                    continue
                h_here = hv * (1 - (d / half_w) ** 2)
                idx = r * nx + c
                if h_here > H_raster[idx]:
                    H_raster[idx] = h_here
                    U_raster[idx] = Vv


@numba.njit(cache=True)
def _shade_rgb_soft_kernel(rgb, z, dx, dy, ve, Lx, Ly, Lz):
    """In-place soft-light hillshade matching ``LightSource.shade_rgb(
    blend_mode='soft')``. Two passes: gradient + intensity + min/max,
    then full-range normalize + Pegtop blend.
    """
    ny, nx = z.shape
    intensity = np.empty((ny, nx), dtype=np.float64)
    inv_dx = 1.0 / dx
    inv_dy = 1.0 / dy
    inv_2dx = 0.5 * inv_dx
    inv_2dy = 0.5 * inv_dy
    imin = np.inf
    imax = -np.inf
    for i in range(ny):
        if i == 0:
            i_lo = 0; i_hi = 1; idy = inv_dy
        elif i == ny - 1:
            i_lo = ny - 2; i_hi = ny - 1; idy = inv_dy
        else:
            i_lo = i - 1; i_hi = i + 1; idy = inv_2dy
        for j in range(nx):
            if j == 0:
                j_lo = 0; j_hi = 1; idx = inv_dx
            elif j == nx - 1:
                j_lo = nx - 2; j_hi = nx - 1; idx = inv_dx
            else:
                j_lo = j - 1; j_hi = j + 1; idx = inv_2dx
            dzdx = (z[i, j_hi] - z[i, j_lo]) * ve * idx
            dzdy = (z[i_hi, j] - z[i_lo, j]) * ve * idy
            mag = np.sqrt(dzdx * dzdx + dzdy * dzdy + 1.0)
            d = (-dzdx * Lx - dzdy * Ly + Lz) / mag
            intensity[i, j] = d
            if d < imin:
                imin = d
            if d > imax:
                imax = d
    # Match matplotlib's shade_normals: contrast-stretch only when the span is
    # resolvable (> 1e-6); otherwise CLIP the raw intensity to [0, 1] so a flat
    # field blends at t = sin(alt) (~0.707, a light brightening) instead of the
    # old t = 0 darkening + noise amplification (audit m29).
    span = imax - imin
    normalize = span > 1e-6
    inv_span = 1.0 / span if normalize else 0.0
    for i in range(ny):
        for j in range(nx):
            if normalize:
                t = (intensity[i, j] - imin) * inv_span
            else:
                t = intensity[i, j]
                if t < 0.0:
                    t = 0.0
                elif t > 1.0:
                    t = 1.0
            two_t = 2.0 * t
            for c in range(3):
                r = rgb[i, j, c]
                rgb[i, j, c] = two_t * r + r * r * (1.0 - two_t)


def _shade_rgb_soft(rgb, z, dx, dy, ve, azdeg, altdeg):
    """Numba drop-in for ``LightSource(azdeg, altdeg).shade_rgb(
    rgb, z, vert_exag=ve, dx=dx, dy=dy, blend_mode='soft')``. Returns
    a fresh shaded RGB array. dy is negated to match matplotlib's
    image-row convention so output matches bit-for-bit."""
    az = np.radians(azdeg)
    alt = np.radians(altdeg)
    Lx = np.sin(az) * np.cos(alt)
    Ly = np.cos(az) * np.cos(alt)
    Lz = np.sin(alt)
    out = np.ascontiguousarray(rgb, dtype=np.float64).copy()
    z_c = np.ascontiguousarray(z, dtype=np.float64)
    _shade_rgb_soft_kernel(out, z_c, float(dx), float(-dy), float(ve),
                           Lx, Ly, Lz)
    return out


def _trace_paths_arrays(rec_2d, H_2d, area_2d, nx, ny, Lx, Ly, channel_threshold):
    """Trace head→outlet chains through the flow network.

    Each head's path runs from the head along receivers as far as the
    chain can go before either (a) running off a boundary, (b) reaching
    a glacier terminus (``H ≤ 0``), or (c) hitting a node already
    claimed by an earlier-traced path (in which case that node is
    appended as the terminus, so the chain visually connects to the
    already-traced network).

    Heads are iterated in **descending chain length**. The longest chain
    in the network claims its main stem (including any shared trunk and
    outlet segments); shorter heads join it at sub-confluences. This
    keeps the iteration order independent of grid-axis direction so the
    j-order (low-row-first) directional bias is gone.

    Returns ``(paths, xc, yc, H_flat)``.
    """
    dx_grid = Lx / (nx - 1)
    dy_grid = Ly / (ny - 1)

    # NaN receivers (zarr round-trip artifacts at boundary nodes) map to
    # SELF, not node 0 — nan_to_num's 0 is the corner cell, which could
    # grow a bogus path segment from nodes near the corner.
    rec_flat = rec_2d.flatten()
    rec = np.where(np.isnan(rec_flat), np.arange(rec_flat.size),
                   rec_flat).astype(int)
    H = H_2d.flatten()
    area = area_2d.flatten()
    nn = rec.size

    rows, cols = np.divmod(np.arange(nn), nx)
    xc = cols.astype(float) * dx_grid
    yc = rows.astype(float) * dy_grid

    is_active = (H > 0) & (area >= channel_threshold)

    donor_list = [[] for _ in range(nn)]
    for j in range(nn):
        r = rec[j]
        if r != j:
            donor_list[r].append(j)

    heads = [j for j in range(nn)
             if is_active[j] and not any(is_active[d] for d in donor_list[j])]

    def _chain_length(start):
        n = 1
        node = start
        # Bound the walk by the node count so a cyclic receiver array (only
        # possible from corrupted/hand-built input — model receivers are
        # acyclic) can't hang the sort (audit m33; defensive).
        for _ in range(nn):
            r = rec[node]
            if r == node:
                return n
            n += 1
            if H[r] <= 0:
                return n
            node = r
        return n

    heads.sort(key=_chain_length, reverse=True)

    max_step = 2 * max(dx_grid, dy_grid)
    paths = []
    visited = set()
    for head in heads:
        if head in visited:
            continue
        path = [head]
        visited.add(head)
        node = head
        while True:
            r = rec[node]
            if r == node:
                break
            # Detect periodic-boundary wraps BEFORE checking ``visited``. When
            # a wrap-receiver happens to be a node an earlier (longer) path
            # already claimed, the previous ``if r in visited`` short-circuit
            # would append it to the current path and break — drawing a single
            # segment from ``node`` (e.g. row 0) all the way to ``r`` (row
            # ny-1) that the dense resampler / rasterizer then renders as a
            # straight line across the entire domain. Treat wraps the same way
            # whether the target is visited or not: close the current path,
            # and start a new one at the wrap-target only if it's unclaimed.
            is_wrap = (abs(xc[r] - xc[node]) > max_step
                       or abs(yc[r] - yc[node]) > max_step)
            if is_wrap:
                if len(path) >= 2:
                    paths.append(path)
                if r in visited or H[r] <= 0:
                    # wrap into already-traced network or onto an ice-free
                    # terminus — nothing more to draw for this head.
                    path = []
                    break
                path = [r]
                visited.add(r)
                node = r
                continue
            if r in visited:
                path.append(r)
                break
            path.append(r)
            if H[r] <= 0:
                break
            visited.add(r)
            node = r
        if len(path) >= 2:
            paths.append(path)

    return paths, xc, yc, H


@numba.njit(cache=True)
def _priority_flood_kernel(z, wrap_y=False, wrap_x=False):
    """Priority-flood with an explicit binary min-heap over flat
    (h, idx) pairs. ~50× faster than the Python heapq version at
    800×800 (~10 ms vs ~570 ms).

    A looped axis (``wrap_y`` / ``wrap_x``) is NOT seeded as an outlet and its
    neighbour stencil wraps at the seam, so a depression straddling a periodic
    seam stays a lake instead of being split and drained out both edges (the
    same seam-awareness the physics fill / ice footprint use; audit m32).
    Bit-for-bit with the old kernel when both are False."""
    ny, nx = z.shape
    nn = ny * nx
    filled = z.copy()
    visited = np.zeros(nn, dtype=np.bool_)
    heap_h = np.empty(nn, dtype=np.float64)
    heap_idx = np.empty(nn, dtype=np.int64)
    n = 0

    # Seed: boundary cells at their own elevation, skipping looped axes (their
    # seam cells are interior, not outlets).
    if not wrap_x:
        for i in range(ny):
            for col in (0, nx - 1):
                idx = i * nx + col
                heap_h[n] = z[i, col]
                heap_idx[n] = idx
                visited[idx] = True
                c = n
                while c > 0:
                    p = (c - 1) >> 1
                    if heap_h[p] > heap_h[c]:
                        th = heap_h[p]; heap_h[p] = heap_h[c]; heap_h[c] = th
                        ti = heap_idx[p]; heap_idx[p] = heap_idx[c]; heap_idx[c] = ti
                        c = p
                    else:
                        break
                n += 1
    if not wrap_y:
        j0 = 0 if wrap_x else 1
        j1 = nx if wrap_x else nx - 1
        for j in range(j0, j1):
            for row in (0, ny - 1):
                idx = row * nx + j
                heap_h[n] = z[row, j]
                heap_idx[n] = idx
                visited[idx] = True
                c = n
                while c > 0:
                    p = (c - 1) >> 1
                    if heap_h[p] > heap_h[c]:
                        th = heap_h[p]; heap_h[p] = heap_h[c]; heap_h[c] = th
                        ti = heap_idx[p]; heap_idx[p] = heap_idx[c]; heap_idx[c] = ti
                        c = p
                    else:
                        break
                n += 1

    while n > 0:
        h = heap_h[0]
        idx = heap_idx[0]
        n -= 1
        if n > 0:
            heap_h[0] = heap_h[n]
            heap_idx[0] = heap_idx[n]
            p = 0
            while True:
                l = 2 * p + 1
                r = l + 1
                sm = p
                if l < n and heap_h[l] < heap_h[sm]:
                    sm = l
                if r < n and heap_h[r] < heap_h[sm]:
                    sm = r
                if sm == p:
                    break
                th = heap_h[p]; heap_h[p] = heap_h[sm]; heap_h[sm] = th
                ti = heap_idx[p]; heap_idx[p] = heap_idx[sm]; heap_idx[sm] = ti
                p = sm

        i = idx // nx
        j = idx - i * nx
        # 4-neighbour push (wrapping on looped axes)
        for k in range(4):
            if k == 0:
                ni = i - 1; nj = j
            elif k == 1:
                ni = i + 1; nj = j
            elif k == 2:
                ni = i; nj = j - 1
            else:
                ni = i; nj = j + 1
            if wrap_y:
                ni = ni % ny
            if wrap_x:
                nj = nj % nx
            if 0 <= ni < ny and 0 <= nj < nx:
                nidx = ni * nx + nj
                if not visited[nidx]:
                    zn = z[ni, nj]
                    new_h = h if zn < h else zn
                    filled[ni, nj] = new_h
                    visited[nidx] = True
                    heap_h[n] = new_h
                    heap_idx[n] = nidx
                    c = n
                    while c > 0:
                        p = (c - 1) >> 1
                        if heap_h[p] > heap_h[c]:
                            th = heap_h[p]; heap_h[p] = heap_h[c]; heap_h[c] = th
                            ti = heap_idx[p]; heap_idx[p] = heap_idx[c]; heap_idx[c] = ti
                            c = p
                        else:
                            break
                    n += 1
    return filled


def _priority_flood(z, wrap_y=False, wrap_x=False):
    """Fill closed depressions in a 2D DEM via priority-flood from the grid
    boundary. Returns a filled DEM ``f`` with ``f >= z`` everywhere. Each
    interior cell's filled elevation is the lowest spill elevation reachable
    along any monotonic-ascending path to the grid boundary. Depression
    cells get ``f > z`` (= the spill / outlet elevation); other cells get
    ``f == z``. ``wrap_y`` / ``wrap_x`` make a looped axis seam-aware (m32).
    """
    return _priority_flood_kernel(np.ascontiguousarray(z, dtype=np.float64),
                                  wrap_y, wrap_x)


@numba.njit(cache=True)
def _variable_sigma_smooth_kernel(padded, sigmas, ds, pad, out):
    n = sigmas.shape[0]
    for k in range(n):
        sigma = sigmas[k]
        pk = k + pad
        if sigma <= 0:
            out[k] = padded[pk]
            continue
        two_var = 2.0 * sigma * sigma
        half_w = int(np.ceil(4.0 * sigma / ds))
        ws = 0.0
        wxs = 0.0
        for di in range(-half_w, half_w + 1):
            offset = di * ds
            w = np.exp(-offset * offset / two_var)
            ws += w
            wxs += w * padded[pk + di]
        out[k] = wxs / ws


def _variable_sigma_smooth(values, sigmas, ds):
    """1D Gaussian smooth along a uniformly-spaced array with per-point σ.

    Boundary padding uses **linear extrapolation** of the local slope (not
    the boundary value). With 'nearest' padding and a monotonic approach
    to the endpoint, the smoothed value is biased *away* from the
    boundary by ~σ·|dy/ds|, in a direction set by the slope sign — a
    direction-dependent artefact (tribs entering the trunk from the north
    stop above the trunk axis, those from the south stop below it).
    Linear-extrap continues the slope, making the smoothing window
    value-symmetric and removing the bias.
    """
    values = np.ascontiguousarray(values, dtype=np.float64)
    sigmas = np.ascontiguousarray(sigmas, dtype=np.float64)
    n = values.size
    if n < 2:
        return values.copy()
    max_sigma = float(sigmas.max())
    if max_sigma <= 0:
        return values.copy()
    pad = int(np.ceil(4 * max_sigma / ds))
    slope_left = float(values[1] - values[0])
    slope_right = float(values[-1] - values[-2])
    left_pad = values[0] - np.arange(pad, 0, -1) * slope_left
    right_pad = values[-1] + np.arange(1, pad + 1) * slope_right
    padded = np.concatenate([left_pad, values, right_pad])
    out = np.empty_like(values)
    _variable_sigma_smooth_kernel(padded, sigmas, ds, pad, out)
    return out


def _compute_glacier_field(rec_2d, H_2d, area_2d, Qg_2d,
                           nx, ny, Lx, Ly, alpha_g,
                           channel_threshold, oversample, sigma_along_cells,
                           blur_sigma_sub, field, hc_over_H=HC_OVER_H):
    """Build (H_field, U_field, Q_field) for one time slice. The sole caller is
    ``landscape(show_smoothed_paths=True)``, which passes ``field='depth'``
    (``Qg_2d=None``); the ``'velocity'`` / ``'flux'`` modes are a kept
    future surface (e.g. an ice-cmap-by-velocity option), currently unreached.

    ``hc_over_H`` (the centerline/mean channel-depth ratio) is the run's value
    threaded from ``m.hc_over_H``; it defaults to the import-bound
    ``constants.HC_OVER_H`` for standalone use.
    """
    from scipy.ndimage import gaussian_filter1d

    if field not in ('depth', 'velocity', 'flux'):
        raise ValueError(
            f"field must be 'depth', 'velocity', or 'flux', got {field!r}")
    want_velocity = field in ('velocity', 'flux')

    dx_grid = Lx / (nx - 1)
    dy_grid = Ly / (ny - 1)
    nx_sub = (nx - 1) * oversample + 1
    ny_sub = (ny - 1) * oversample + 1
    dx_sub = Lx / (nx_sub - 1)
    dy_sub = Ly / (ny_sub - 1)

    paths, xc, yc, H = _trace_paths_arrays(
        rec_2d, H_2d, area_2d, nx, ny, Lx, Ly, channel_threshold)
    Qg_flat = Qg_2d.flatten() if want_velocity else None

    # Per-path chunks; concatenated to flat arrays once before rasterizing.
    seg_chunks = {k: [] for k in
                  ('x1', 'y1', 'x2', 'y2', 'h1', 'h2', 'w1', 'w2')}
    if want_velocity:
        seg_chunks['V1'] = []
        seg_chunks['V2'] = []
    v_chunks = {k: [] for k in ('x', 'y', 'h', 'w')}
    if want_velocity:
        v_chunks['V'] = []

    sigma_s = sigma_along_cells * max(dx_grid, dy_grid)
    s_step = 0.5 * min(dx_sub, dy_sub)
    max_width = min(Lx, Ly) / 2
    H_floor = 0.5

    paths_xy = []

    for path in paths:
        if len(path) < 2:
            continue
        idx = np.asarray(path)
        px = xc[idx]
        py = yc[idx]
        ph = np.maximum(H[idx], 0.0)
        pw = np.minimum(ph * alpha_g, max_width)
        pQ = np.maximum(Qg_flat[idx], 0.0) if want_velocity else None

        ds = np.sqrt(np.diff(px)**2 + np.diff(py)**2)
        s = np.concatenate([[0.0], np.cumsum(ds)])
        L = s[-1]
        if L == 0:
            continue

        n_dense = max(2, int(np.ceil(L / s_step)) + 1)
        s_dense = np.linspace(0, L, n_dense)
        px_d = np.interp(s_dense, s, px)
        py_d = np.interp(s_dense, s, py)
        ph_d = np.interp(s_dense, s, ph)
        pw_d = np.interp(s_dense, s, pw)
        pQ_d = np.interp(s_dense, s, pQ) if want_velocity else None

        ds_dense = L / (n_dense - 1)

        if sigma_s > 0:
            sigma_samples = sigma_s / ds_dense
            if sigma_samples > 0.1:
                px_d = gaussian_filter1d(px_d, sigma_samples, mode='nearest')
                py_d = gaussian_filter1d(py_d, sigma_samples, mode='nearest')
                ph_d = gaussian_filter1d(ph_d, sigma_samples, mode='nearest')
                pw_d = gaussian_filter1d(pw_d, sigma_samples, mode='nearest')
                if want_velocity:
                    pQ_d = gaussian_filter1d(pQ_d, sigma_samples, mode='nearest')

        # Variable-σ smoothing of POSITIONS and WIDTH only. The native path
        # zigzags by ±1 cell per D8-receiver step; at kilometer channel
        # widths, adjacent dense-segment wedges at differing angles
        # tessellate badly, producing jagged ice outlines and asymmetric
        # cross-flow z assignments. Smoothing σ scales with the LOCAL
        # channel width (σ = half-width), so narrow heads get little
        # smoothing and wide trunks get a lot. ph_d, pQ_d are NOT
        # smoothed here: those values are keyed to the arc-length parameter
        # s_dense, so they continue to reflect the original-path H/Q at
        # each native node even though the geometric (x, y) is now smoothed.
        sigmas_pos = 0.5 * pw_d
        if sigmas_pos.size and float(sigmas_pos.max()) > 0:
            px_d = _variable_sigma_smooth(px_d, sigmas_pos, ds_dense)
            py_d = _variable_sigma_smooth(py_d, sigmas_pos, ds_dense)
            # pw_d intentionally not smoothed: at paths terminating on a
            # visited trunk node, pw has a step jump from trib_W to
            # trunk_W in the last segment. Linear-extrap padding
            # continues that slope past the endpoint, projecting pw
            # values even larger than trunk_W; the wide variable-σ
            # window then pulls upstream pw values upward and inflates
            # the rasterized wedge into a kilometer-scale blob at every
            # confluence. pw varies smoothly along the path already
            # (H is a smooth field), so no smoothing is needed here.

        paths_xy.append((px_d.copy(), py_d.copy()))

        # centerline depth of the parabolic cross-section (= the model's
        # zs - zb under the channel-floor datum: the rendered valley floor
        # IS the tracked bed)
        ph_max = hc_over_H * ph_d

        if want_velocity:
            safe = ph_d > H_floor
            vV = np.zeros_like(ph_d)
            vV[safe] = pQ_d[safe] / (alpha_g * ph_d[safe] ** 2)

        def _seg(a): return a[:-1], a[1:]

        sx1, sx2 = _seg(px_d);   seg_chunks['x1'].append(sx1); seg_chunks['x2'].append(sx2)
        sy1, sy2 = _seg(py_d);   seg_chunks['y1'].append(sy1); seg_chunks['y2'].append(sy2)
        sh1, sh2 = _seg(ph_max); seg_chunks['h1'].append(sh1); seg_chunks['h2'].append(sh2)
        sw1, sw2 = _seg(pw_d);   seg_chunks['w1'].append(sw1); seg_chunks['w2'].append(sw2)
        if want_velocity:
            sv1, sv2 = _seg(vV); seg_chunks['V1'].append(sv1); seg_chunks['V2'].append(sv2)

        v_chunks['x'].append(px_d)
        v_chunks['y'].append(py_d)
        v_chunks['h'].append(ph_max)
        v_chunks['w'].append(pw_d)
        if want_velocity:
            v_chunks['V'].append(vV)

    H_field = np.zeros(ny_sub * nx_sub, dtype=float)
    U_field = np.zeros(ny_sub * nx_sub, dtype=float) if want_velocity else None

    def _cat(name): return np.concatenate(seg_chunks[name]) if seg_chunks[name] else None
    def _vcat(name): return np.concatenate(v_chunks[name]) if v_chunks[name] else None

    seg_x1 = _cat('x1')
    if seg_x1 is not None and seg_x1.size:
        seg_y1, seg_x2, seg_y2 = _cat('y1'), _cat('x2'), _cat('y2')
        seg_h1, seg_h2 = _cat('h1'), _cat('h2')
        seg_w1, seg_w2 = _cat('w1'), _cat('w2')
        if want_velocity:
            _rasterize_glacier_segments_HU(
                H_field, U_field, seg_x1, seg_y1, seg_x2, seg_y2,
                seg_h1, seg_h2, seg_w1, seg_w2,
                _cat('V1'), _cat('V2'),
                nx_sub, ny_sub, dx_sub, dy_sub,
            )
        else:
            _rasterize_glacier_segments(
                H_field, seg_x1, seg_y1, seg_x2, seg_y2,
                seg_h1, seg_h2, seg_w1, seg_w2,
                nx_sub, ny_sub, dx_sub, dy_sub,
            )

    v_x = _vcat('x')
    if v_x is not None and v_x.size:
        v_y, v_h, v_w = _vcat('y'), _vcat('h'), _vcat('w')
        if want_velocity:
            _rasterize_glacier_vertices_HU(
                H_field, U_field, v_x, v_y, v_h, v_w, _vcat('V'),
                nx_sub, ny_sub, dx_sub, dy_sub,
            )
        else:
            _rasterize_glacier_vertices(
                H_field, v_x, v_y, v_h, v_w,
                nx_sub, ny_sub, dx_sub, dy_sub,
            )

    H_field = H_field.reshape(ny_sub, nx_sub)
    if want_velocity:
        U_field = U_field.reshape(ny_sub, nx_sub)

    if blur_sigma_sub > 0:
        H_field = gaussian_filter(H_field, sigma=blur_sigma_sub, mode='nearest')
        if want_velocity:
            U_field = gaussian_filter(U_field, sigma=blur_sigma_sub, mode='nearest')

    Q_field = H_field * U_field if field == 'flux' else None

    return SimpleNamespace(H=H_field, U=U_field, Q=Q_field,
                            paths_xy=paths_xy)


def _footprint_ice_surface(H_in, zb_in, rec_in, alpha_g, dy, dx,
                           wrap_y, wrap_x, H_threshold=0.0,
                           area_in=None, area_threshold=0.0,
                           hc_over_H=HC_OVER_H):
    """Glacier-footprint ice surface for display — the rendering dual of
    the sub-grid width carve, built from the SAME machinery (union of
    discs R = alpha_g*H/2, power-diagram attribution; seam-aware along
    looped axes).

    Built ENTIRELY from the consistently-timed pair (zb_in, H_in) — the
    kernel-committed bed and thickness. The presented surface
    zs = zb + HC_OVER_H*H is reconstructed here rather than read from
    topography__elevation: topography is a STATE-variable snapshot while
    H/bed are OUT-variable snapshots, and at cells whose ice flickers
    between frames a mistimed z makes the source "ice surface" the bare
    bed — every neighbour then reads as a nunatak and the footprint fill
    collapses to the channel cells (1-cell-wide trunks through carved
    valleys, which is geometrically impossible: a carve-widened flank
    bed lies below the trimline by construction).

    Each footprint cell's ice surface is its attributed source's zs (the
    cross-section's transversely flat free surface), so a carved trough
    fills with ice exactly to the trimline, while footprint terrain
    still standing ABOVE the source's ice surface stays rock (rate-cap
    survivors render as nunataks). Display only — no model state is
    touched.

    Seeds (and own-ice cells) are gated by ``H > H_threshold`` and, when
    ``area_in`` is supplied, by ``area >= area_threshold`` — the area gate
    drops small-catchment specks without vetoing cells that merely lie under a
    larger glacier's footprint (those are attributed, not seeds).

    Returns ``(z_filled, ice_mask, depth)`` at native resolution:
    the ice-filled surface (max of terrain and footprint ice surface),
    the cells under ice (footprint fill OR carrying H > H_threshold
    themselves), and the local ice column z_filled - zb_in (at a source
    cell exactly HC_OVER_H*H)."""
    ny, nx = H_in.shape
    # consistently-timed presented surface (bare cells: zs = bed)
    zs = zb_in + hc_over_H * H_in
    # NaN receivers (zarr round-trip artifacts at boundary nodes) map to
    # SELF — same sanitization as _trace_paths_arrays; self-receiving
    # cells are excluded as seeds, mirroring the carve.
    rec_flat = np.asarray(rec_in, dtype=float).ravel()
    rec = np.where(np.isnan(rec_flat), np.arange(rec_flat.size),
                   rec_flat).astype(np.int64)
    # Seed/own-ice gate: column thickness AND (optionally) upstream drainage
    # area. Area gates only SOURCES — it never vetoes a cell that lies under a
    # larger glacier's footprint (that cell is attributed, not a seed).
    seed_ok = H_in > H_threshold
    if area_in is not None and area_threshold > 0:
        seed_ok = seed_ok & (np.asarray(area_in) >= area_threshold)
    H_seed = np.where(seed_ok, H_in, 0.0)
    offsets = np.empty((ny, nx))
    # Display seeds every own-ice cell (the H_seed==0 gate already excludes
    # non-sources); no de-seeding here, so allow every cell.
    seed_mask = np.ones(H_seed.size, dtype=np.int8)
    n_seed = _carve_offsets(H_seed.ravel(), rec, float(alpha_g),
                            offsets.ravel(), seed_mask)
    ice_cells = seed_ok
    if n_seed == 0:
        depth = np.where(ice_cells, hc_over_H * H_in, 0.0)
        return zs.copy(), ice_cells, depth
    D = np.empty((ny, nx))
    SRC = np.empty((ny, nx), dtype=np.int64)
    if wrap_y or wrap_x:
        _power_dt_2d_periodic(offsets, dy, dx, D, SRC, wrap_y, wrap_x)
    else:
        _power_dt_2d(offsets, dy, dx, D, SRC)
    inside = (D < 0.0) & (SRC >= 0)
    ice_top = np.full((ny, nx), -np.inf)
    ice_top[inside] = zs.flat[SRC[inside]]
    z_filled = np.maximum(zs, ice_top)
    ice_mask = (ice_top > zs + 1e-9) | ice_cells
    depth = np.where(ice_mask, np.maximum(z_filled - zb_in, 0.0), 0.0)
    return z_filled, ice_mask, depth


def _smooth_ice_mask(mask_nat, Y, X, sigma):
    """Subgrid ice mask with a smoothed (de-staircased) outline.

    The terrain in ``landscape`` gets bicubic upsampling plus a
    Gaussian blur, so an order-0 (nearest) upsample of the native ice
    mask reads as a blocky staircase against it. Instead, upsample the
    mask as a float (bilinear) onto the subgrid coordinates ``(Y, X)``,
    Gaussian-smooth it with ``sigma`` (subgrid pixels), and re-threshold
    at 1/2: the outline becomes a level set of a smoothed indicator —
    corners rounded at the ``sigma`` scale while the boundary stays
    pinned to the native cell-midpoint contour (the 1/2 level), so the
    rendered glacier neither grows nor shrinks systematically.

    ``sigma`` of 0 (or None) returns the crisp order-0 mask, the
    pre-smoothing behaviour, exactly.
    """

    if sigma and sigma > 0:
        frac = map_coordinates(np.asarray(mask_nat, dtype=float), [Y, X],
                               order=1, mode='nearest')
        return gaussian_filter(frac, sigma=float(sigma)) > 0.5
    return map_coordinates(np.asarray(mask_nat, dtype=np.uint8), [Y, X],
                           order=0, mode='nearest') > 0


def _field_ice_mask(H_nat, Y, X, sigma, H_threshold):
    """Ice mask from a smoothed CLIPPED thickness field (``ice_smoothing='field'``,
    the ``'cells'``-extent anti-flicker variant of :func:`_smooth_ice_mask`).

    Instead of smoothing the binary indicator ``H > thr`` (whose whole blobs pop
    as a cell's H oscillates across the threshold), upsample the CLIPPED field
    ``min(H, 2*thr)`` bilinearly onto the subgrid ``(Y, X)``, Gaussian-smooth it
    by ``sigma`` (subgrid px), and threshold at ``thr``. A cell that hovers at
    the threshold now moves the level set by a sub-cell amount rather than
    toggling a region; the clip at ``2*thr`` bounds a thick trunk's spatial bleed
    to about one cell. Because the result is a level set of a smoothed field it
    includes the thin (``thr/2..thr``) apron, so the drawn extent dilates
    modestly vs the raw ``H > thr`` mask. Requires ``H_threshold > 0`` (with
    ``thr = 0`` the clip zeroes the field and the mask is empty)."""
    thr = float(H_threshold)
    field = map_coordinates(np.minimum(np.asarray(H_nat, dtype=float), 2.0 * thr),
                            [Y, X], order=1, mode='nearest')
    if sigma and sigma > 0:
        field = gaussian_filter(field, sigma=float(sigma))
    return field > thr


def _clean_ice_mask(mask, min_native_cells, oversample):
    """Speckle/hole cleanup of a subgrid ice mask (``min_ice_cells``).

    Drops connected ice components smaller than ``min_native_cells`` native cells
    and fills enclosed bare holes smaller than the same, where a native cell is
    ``oversample**2`` subgrid pixels. Bare regions touching the array border are
    NEVER filled (open margin / ocean stays open). ``min_native_cells <= 0`` is a
    no-op. Ported from the display-pipeline probe (docs/dev/step_flicker.md)."""
    if not min_native_cells or min_native_cells <= 0:
        return mask
    from scipy.ndimage import label
    minpix = int(min_native_cells) * int(oversample) ** 2
    m = np.asarray(mask, dtype=bool)
    # 1. Remove small ice components.
    lab, n = label(m)
    if n:
        sizes = np.bincount(lab.ravel())
        m = m & ~np.isin(lab, np.nonzero(sizes < minpix)[0])
    # 2. Fill small enclosed bare holes — but never a bare region touching the
    #    array border (that is open margin, not an enclosed hole).
    lab, n = label(~m)
    if n:
        sizes = np.bincount(lab.ravel())
        border_labs = np.unique(np.concatenate(
            [lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
        small = sizes < minpix
        small[border_labs] = False
        m = m | np.isin(lab, np.nonzero(small)[0])
    return m


def _mean_recent_H(H_out, i, k):
    """Trailing-mean ice thickness over the last ``k`` output frames ending at
    frame ``i`` (``ice_time_avg``), clamped at the start of the run.

    Display-only smoothing for the ICE layer (mask + depth/fill inputs); the
    model state is untouched. ``i`` is normalised to a non-negative index first,
    so ``i = -1`` averages the window ending at the last frame. ``k <= 1`` returns
    frame ``i`` unchanged (identity)."""
    H_out = np.asarray(H_out)
    n = H_out.shape[0]
    i = int(i) % n
    k = max(1, int(k))
    lo = max(0, i - k + 1)
    return H_out[lo:i + 1].mean(axis=0)


def _add_colorbar(mappable, ax, label='', location='right', size='4%', pad=0.05):
    """Colorbar matched to the axes height (make_axes_locatable)."""
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    cax = make_axes_locatable(ax).append_axes(location, size=size, pad=pad)
    return ax.figure.colorbar(mappable, cax=cax, label=label)


def _slider_view(make_draw, sliders, *, figsize, fmt=None):
    """The one notebook slider viewer — used by every ``view_*``.

    A persistent ``ipympl`` canvas built **once** and redrawn in place via
    ``draw_idle`` on each slider change — no figure recreation, no inline-PNG
    flash, no page reflow. The canvas is a standalone widget built off pyplot's
    registry, so it neither switches the notebook's backend nor gets re-rendered
    inline on later cells; other (static) plots are unaffected.

    The caller supplies ``make_draw(fig)``, called once to set up ``fig`` and
    return a ``draw(*idxs)`` that renders the frame(s). ``sliders`` is a list of
    ``(n, label, start)`` tuples — one ``IntSlider`` each (``start < 0`` counts
    from the end). ``fmt(*idxs) -> str`` optionally labels a row above the canvas
    (profile viewers draw their own in-plot title instead and pass ``fmt=None``).

    Displays the viewer in place and returns ``None`` — returning the slider
    would make the notebook cell auto-display a duplicate of it.
    """
    try:
        import ipywidgets as widgets
        from IPython.display import display
    except ImportError as e:  # pragma: no cover - notebook-only path
        raise RuntimeError(
            "the interactive viewer needs ipywidgets (`pip install ipywidgets`); "
            "use the frame (`profile`/`map`) or `animate_*` method otherwise"
        ) from e
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib._pylab_helpers import Gcf

    # Build a pyplot-managed figure under the ipympl backend (the figure manager
    # is what renders the canvas as a live widget), then DETACH it from pyplot's
    # registry and restore the notebook's default backend. Detaching stops the
    # inline backend from re-rendering it as a static image on later cells;
    # restoring keeps every other (static) plot on the notebook default.
    prev_backend = matplotlib.get_backend()
    try:
        plt.switch_backend('module://ipympl.backend_nbagg')
    except Exception as e:  # pragma: no cover - ipympl missing
        raise RuntimeError(
            "the interactive viewer needs ipympl (`pip install ipympl`); use the "
            "frame (`profile`/`map`) or `animate_*` methods otherwise") from e
    try:
        with plt.ioff():
            fig = plt.figure(figsize=figsize, layout='constrained')
        for attr in ('header_visible', 'footer_visible', 'toolbar_visible'):
            try: setattr(fig.canvas, attr, False)   # trim ipympl chrome
            except Exception: pass
        try: fig.canvas.resizable = False           # no resize handle
        except Exception: pass

        draw = make_draw(fig)

        sl = []
        for n, label, start in sliders:
            v = start if (start is None or start >= 0) else n + start
            sl.append(widgets.IntSlider(
                min=0, max=n - 1, value=v or 0, step=1, description=label,
                continuous_update=True, readout=False,
                layout=widgets.Layout(width='98%')))
        tag = widgets.Label()

        def show(*_):
            idxs = [s.value for s in sl]
            draw(*idxs)
            if fmt is not None:
                tag.value = fmt(*idxs)
            fig.canvas.draw_idle()

        for s in sl:
            s.observe(show, names='value')
        show()
        children = list(sl) + ([tag] if fmt is not None else []) + [fig.canvas]
        display(widgets.VBox(children))
        Gcf.figs.pop(fig.canvas.manager.num, None)  # detach: no static re-render
    finally:
        matplotlib.use(prev_backend)                # restore the notebook default
    # No return: returning the slider would make the cell auto-display a duplicate.


def _profile_slider(frame, n_frames, n_panels, times, *, fig_width=12,
                    aspect=0.27, legend=False, start=-1, legend_loc='upper left'):
    """Shared profile slider (1D + 2D ``view_profile``): persistent canvas,
    fixed aspect, despined, integer-kyr title at the right, and an OFF-by-default
    **static** legend (same entries every frame, so they don't pop in/out as ice
    comes and goes).

    ``frame(axes, idx)`` draws every panel for output step ``idx`` into the given
    axes (clearing + setting limits itself). ``n_panels`` is the stack height;
    ``times`` is the output-time vector (yr); ``fig_width`` sets the on-screen
    size and ``aspect`` the per-panel height/width ratio.
    """
    import copy
    figsize = (fig_width, fig_width * aspect * n_panels)

    def make_draw(fig):
        axes = fig.subplots(n_panels, 1, sharex=True, squeeze=False)[:, 0]

        static = [([], []) for _ in axes]
        if legend:
            sample = np.unique(
                np.linspace(0, n_frames - 1, min(n_frames, 15)).astype(int))
            for j in sample:
                frame(axes, int(j))
                for k, ax in enumerate(axes):
                    h, l = ax.get_legend_handles_labels()
                    if len(l) > len(static[k][1]):
                        static[k] = ([copy.copy(x) for x in h], list(l))

        def draw(idx):
            frame(axes, idx)
            for k, ax in enumerate(axes):
                ax.spines[['top', 'right']].set_visible(False)
                cur = ax.get_legend()
                if cur is not None:
                    cur.remove()
                if legend:
                    ax.legend(*static[k], loc=legend_loc)
            axes[0].set_title(f"{times[idx] / 1e3:.0f} kyr", loc='right')
        return draw

    return _slider_view(make_draw, [(n_frames, 'Snapshot', start)],
                        figsize=figsize)
