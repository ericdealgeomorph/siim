"""Shared rendering backend for the siim plotting package.

Hosts the one notebook slider viewer (``_slider_view``), the colorbar helper,
and the numba glacier-field rendering backend — rasterizers, soft hillshade,
glacier-path tracing, priority-flood lakes, variable-sigma smoothing, the
glacier-field builder, and the footprint ice-surface / smoothed-mask helpers —
relocated here from the old ``siim2d_plotting`` module. See
``docs/guides/outputs_and_io.md`` for the public plotting contract.

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
    "_priority_flood", "_shade_rgb_soft", "_ice_ramp", "ICE_RAMP_STOPS",
    "_channel_closure", "_trunk_ribbons",
]

# The glacier ramp of the depth-graded ('veil') ice shading: thin apron ->
# trunk core. Deliberately avoids both ends of the bed map (gist_earth's navy
# lowlands and white summits), so ice never reads as terrain.
ICE_RAMP_STOPS = ('#eaf3fa', '#c5dff0', '#8ec4e3', '#4e9fd0', '#1f6fb0')
_ICE_RAMP_CACHE = {}


def _ice_ramp():
    """The ``ICE_RAMP_STOPS`` colormap, built lazily (matplotlib stays out of
    this module's import) and cached — ``animate_landscape`` asks per frame."""
    if 'cmap' not in _ICE_RAMP_CACHE:
        from matplotlib.colors import LinearSegmentedColormap
        _ICE_RAMP_CACHE['cmap'] = LinearSegmentedColormap.from_list(
            'siim_glacier', list(ICE_RAMP_STOPS))
    return _ICE_RAMP_CACHE['cmap']


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


# The surface-carrying twins the trunk ribbons rasterize with. The HU kernels
# are payload-agnostic (they interpolate ONE per-node scalar along the
# segment and keep the depth-winner's value), so the flat source ice surface
# rides the same code path as the velocity field, not a copied kernel.
_rasterize_glacier_segments_HZ = _rasterize_glacier_segments_HU
_rasterize_glacier_vertices_HZ = _rasterize_glacier_vertices_HU


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


def _sanitized_receivers(rec_in):
    """Flat int receiver array with NaNs mapped to SELF — the zarr round-trip
    artefact at boundary nodes that ``_trace_paths_arrays`` and
    ``_footprint_ice_surface`` both guard against (``nan_to_num``'s 0 is the
    corner cell, which would grow a bogus path from every such node)."""
    rec_flat = np.asarray(rec_in, dtype=float).ravel()
    return np.where(np.isnan(rec_flat), np.arange(rec_flat.size),
                    rec_flat).astype(np.int64)


def _channel_closure(H_2d, rec_in, seed_mask):
    """Trunk class: the DOWNSTREAM CLOSURE of ``seed_mask`` along the receivers.

    A cell joins when one of its donors is already in the class, propagated
    until nothing changes, so every icy cell downstream of a seed belongs and
    the class runs out to the terminus. A plain width cut does not: at glacial
    max on the reference run it reached 0 of 64 trunk termini, because the
    tongue thins below the cut long before it ends. The walk stops at the first
    ice-free receiver, so the class never leaves the glacier, and it is a
    hysteresis class — a cell is a trunk because of what drains INTO it.

    ``rec_in`` is the single-receiver output view (``receivers_out[i]``: the D8
    receiver, or D-inf's largest-weight one). O(N) — each cell is added once.
    """
    H_2d = np.asarray(H_2d, dtype=float)
    H = H_2d.ravel()
    rec = _sanitized_receivers(rec_in)
    cls = np.asarray(seed_mask, dtype=bool).ravel().copy()
    frontier = np.nonzero(cls)[0]
    while frontier.size:
        r = rec[frontier]
        grows = (r != frontier) & (H[r] > 0.0) & ~cls[r]
        frontier = np.unique(r[grows])
        cls[frontier] = True
    return cls.reshape(H_2d.shape)


def _trunk_ribbons(rec_2d, H_2d, area_2d, zs_2d, cells, nx, ny, Lx, Ly,
                   alpha_g, oversample, wrap_y=False, wrap_x=False,
                   hc_over_H=HC_OVER_H):
    """Rasterize the trunk class as true-width ribbons (the ``'ribbons'`` ice
    look): :func:`_compute_glacier_field`'s tracer restricted to ``cells``,
    drawn at the claimed width ``W = alpha_g*H`` with a floor of 1.5 subgrid
    pixels — the narrowest band the raster draws gap-free across a diagonal.

    Three departures from ``_compute_glacier_field``'s smoothing, each measured
    while prototyping:

    - centreline sigma is HALF THE DRAWN WIDTH (not a fixed cell count), so the
      smoother displaces a centreline by at most about its own half-width and a
      tributary's end always lands inside the trunk it joins;
    - the path END is PINNED back onto its node over the last sigma of arc
      length. ``_variable_sigma_smooth``'s linear-extrapolation padding drifts
      the endpoint by up to a sigma, which tore junctions open (one prototype
      network: 13 glaciers rendered as 332 outline pieces before the pin);
    - on a LOOPED axis the tracer closes its path at the wrap, so the step
      across the seam is drawn here explicitly, once on each side.

    Returns ``(depth, surface)`` on the oversampled grid: the parabolic column
    depth across the ribbon (centreline ``hc_over_H*H``, cross-section mean
    ``H`` — the carve convention) and the FLAT source ice surface
    ``zs = zb + hc_over_H*H`` carried along the path, so the render shades the
    glacier's own free surface rather than the bed under it. Ice-free pixels
    keep depth 0.
    """
    H_2d = np.asarray(H_2d, dtype=float)
    dx_grid = Lx / (nx - 1)
    dy_grid = Ly / (ny - 1)
    nx_sub = (nx - 1) * oversample + 1
    ny_sub = (ny - 1) * oversample + 1
    dx_sub = Lx / (nx_sub - 1)
    dy_sub = Ly / (ny_sub - 1)
    width_floor = 1.5 * max(dx_sub, dy_sub)
    max_width = min(Lx, Ly) / 2
    s_step = 0.5 * min(dx_sub, dy_sub)

    # Restricting the tracer to the class: zeroing H outside it makes every
    # non-class cell a terminus, so a ribbon tapers to a toe exactly where the
    # class ends (the tracer walks one cell past it, at zero thickness).
    H_cls = np.where(np.asarray(cells, dtype=bool), H_2d, 0.0)
    paths, xc, yc, H_flat = _trace_paths_arrays(
        rec_2d, H_cls, area_2d, nx, ny, Lx, Ly, 0.0)
    zs = np.asarray(zs_2d, dtype=float).ravel()

    seg = {k: [] for k in ('x1', 'y1', 'x2', 'y2', 'h1', 'h2',
                           'w1', 'w2', 'z1', 'z2')}
    vert = {k: [] for k in ('x', 'y', 'h', 'w', 'z')}

    for path in paths:
        if len(path) < 2:
            continue
        idx = np.asarray(path)
        px, py = xc[idx], yc[idx]
        ph = np.maximum(H_flat[idx], 0.0)
        pw = np.minimum(alpha_g * ph, max_width)
        pz = zs[idx]

        ds = np.sqrt(np.diff(px) ** 2 + np.diff(py) ** 2)
        s = np.concatenate([[0.0], np.cumsum(ds)])
        L = s[-1]
        if L == 0:
            continue
        n_dense = max(2, int(np.ceil(L / s_step)) + 1)
        s_d = np.linspace(0.0, L, n_dense)
        ds_d = L / (n_dense - 1)
        px_d = np.interp(s_d, s, px)
        py_d = np.interp(s_d, s, py)
        ph_d = np.interp(s_d, s, ph)
        pw_d = np.interp(s_d, s, pw)
        pz_d = np.interp(s_d, s, pz)

        sigma = 0.5 * np.maximum(pw_d, width_floor)
        px_raw, py_raw = px_d, py_d
        px_d = _variable_sigma_smooth(px_d, sigma, ds_d)
        py_d = _variable_sigma_smooth(py_d, sigma, ds_d)
        pin = np.clip((L - s_d) / max(sigma[-1], ds_d), 0.0, 1.0)
        px_d = pin * px_d + (1.0 - pin) * px_raw
        py_d = pin * py_d + (1.0 - pin) * py_raw

        pw_d = np.maximum(pw_d, width_floor)
        ph_d = hc_over_H * ph_d

        for key, arr in (('x', px_d), ('y', py_d), ('h', ph_d),
                         ('w', pw_d), ('z', pz_d)):
            vert[key].append(arr)
            seg[key + '1'].append(arr[:-1])
            seg[key + '2'].append(arr[1:])

    if wrap_x or wrap_y:
        # Periodic seam: the tracer CLOSES a path at a wrap (a single segment
        # from row 0 to row ny-1 would otherwise rasterize as a stripe across
        # the whole domain), so the wrapped step is never drawn. Draw it here
        # twice — once with the receiver shifted out past the seam, once with
        # the source shifted — and let the rasterizer's clipping keep the half
        # that lands on each side.
        rec = _sanitized_receivers(rec_2d)
        src = np.nonzero(H_cls.ravel() > 0.0)[0]
        rcv = rec[src]
        off_x = np.zeros(src.size)
        off_y = np.zeros(src.size)
        if wrap_x:                    # period nx*dx: column nx-1's right
            gap = xc[rcv] - xc[src]   # neighbour is column 0, ONE cell away
            off_x = -np.sign(gap) * (nx * dx_grid) * (np.abs(gap)
                                                      > 2 * dx_grid)
        if wrap_y:
            gap = yc[rcv] - yc[src]
            off_y = -np.sign(gap) * (ny * dy_grid) * (np.abs(gap)
                                                      > 2 * dy_grid)
        crosses = (off_x != 0.0) | (off_y != 0.0)
        src, rcv = src[crosses], rcv[crosses]
        off_x, off_y = off_x[crosses], off_y[crosses]
        if src.size:
            H_flat_cls = H_cls.ravel()
            h_s = hc_over_H * H_flat_cls[src]
            h_r = hc_over_H * H_flat_cls[rcv]
            w_s = np.maximum(np.minimum(alpha_g * H_flat_cls[src], max_width),
                             width_floor)
            w_r = np.maximum(np.minimum(alpha_g * H_flat_cls[rcv], max_width),
                             width_floor)
            for sx, sy in ((0.0, 0.0), (-1.0, -1.0)):
                # (0,0): source in place, receiver shifted out past the seam.
                # (-1,-1): the same bridge translated back onto the far side.
                seg['x1'].append(xc[src] + sx * off_x)
                seg['y1'].append(yc[src] + sy * off_y)
                seg['x2'].append(xc[rcv] + (sx + 1.0) * off_x)
                seg['y2'].append(yc[rcv] + (sy + 1.0) * off_y)
                seg['h1'].append(h_s)
                seg['h2'].append(h_r)
                seg['w1'].append(w_s)
                seg['w2'].append(w_r)
                seg['z1'].append(zs[src])
                seg['z2'].append(zs[rcv])

    depth = np.zeros(ny_sub * nx_sub, dtype=float)
    surface = np.zeros(ny_sub * nx_sub, dtype=float)
    if seg['x1']:
        c = {k: np.ascontiguousarray(np.concatenate(v))
             for k, v in seg.items()}
        _rasterize_glacier_segments_HZ(
            depth, surface, c['x1'], c['y1'], c['x2'], c['y2'],
            c['h1'], c['h2'], c['w1'], c['w2'], c['z1'], c['z2'],
            nx_sub, ny_sub, dx_sub, dy_sub)
    if vert['x']:
        v = {k: np.ascontiguousarray(np.concatenate(vv))
             for k, vv in vert.items()}
        _rasterize_glacier_vertices_HZ(
            depth, surface, v['x'], v['y'], v['h'], v['w'], v['z'],
            nx_sub, ny_sub, dx_sub, dy_sub)
    return depth.reshape(ny_sub, nx_sub), surface.reshape(ny_sub, nx_sub)


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


def _label_looped(mask, wrap_y, wrap_x):
    """``scipy.ndimage.label`` made seam-aware: components that touch both
    seams of a looped axis are merged into one (union-find over the seam
    pairs). The returned labels are then non-contiguous, which the
    ``bincount``/``isin`` bookkeeping in :func:`_clean_ice_mask` does not care
    about. With neither axis looped this is plain ``label``."""
    from scipy.ndimage import label
    lab, n = label(mask)
    if not n or not (wrap_y or wrap_x):
        return lab
    parent = np.arange(n + 1)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    seams = []
    if wrap_x:
        seams.append((lab[:, 0], lab[:, -1]))
    if wrap_y:
        seams.append((lab[0, :], lab[-1, :]))
    for a, b in seams:
        both = (a > 0) & (b > 0)
        for u, v in zip(a[both], b[both]):
            ru, rv = find(int(u)), find(int(v))
            if ru != rv:
                parent[max(ru, rv)] = min(ru, rv)
    return np.array([find(k) for k in range(n + 1)])[lab]


def _clean_ice_mask(mask, min_native_cells, oversample, wrap_y=False,
                    wrap_x=False):
    """Speckle/hole cleanup of a subgrid ice mask (``min_ice_cells``).

    Drops connected ice components smaller than ``min_native_cells`` native cells
    and fills enclosed bare holes smaller than the same, where a native cell is
    ``oversample**2`` subgrid pixels. Bare regions touching the array border are
    NEVER filled (open margin / ocean stays open). ``min_native_cells <= 0`` is a
    no-op. This is display-only cleanup; it does not alter model output.

    ``wrap_y``/``wrap_x`` (the model's looped axes) make the cleanup seam-aware:
    a glacier straddling a looped seam counts — and survives — as ONE component
    instead of two sub-minimum halves, and a looped edge is not "array border",
    so a bare hole across the seam is fillable. Both default ``False`` (the
    plain non-periodic behaviour)."""
    if not min_native_cells or min_native_cells <= 0:
        return mask
    minpix = int(min_native_cells) * int(oversample) ** 2
    m = np.asarray(mask, dtype=bool)
    # 1. Remove small ice components.
    lab = _label_looped(m, wrap_y, wrap_x)
    sizes = np.bincount(lab.ravel())
    m = m & ~np.isin(lab, np.nonzero(sizes < minpix)[0])
    # 2. Fill small enclosed bare holes — but never a bare region touching a
    #    non-looped array border (that is open margin, not an enclosed hole).
    lab = _label_looped(~m, wrap_y, wrap_x)
    sizes = np.bincount(lab.ravel())
    edges = ([] if wrap_y else [lab[0], lab[-1]]
             ) + ([] if wrap_x else [lab[:, 0], lab[:, -1]])
    small = sizes < minpix
    small[0] = False                       # label 0 here is the ice, not a hole
    if edges:
        small[np.unique(np.concatenate(edges))] = False
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
