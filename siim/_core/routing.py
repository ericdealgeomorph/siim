"""Flow routing & accumulation primitives for the siim numerical core.

The fastscape-free numba kernels -- SFR + D-inf accumulators, the eps-fill
priority flood, and the Tarboton (1997) D-inf routing primitives. Pulling them
out of the fastscape-importing model is what lets the solver kernels import
without the fastscape stack. The ``DinfFlowRouter`` xsimlab *process* that
consumes them lives in :mod:`siim.fastscape.processes`. numpy/numba only.
"""
import math

import numpy as np
import numba

from ..constants import MFD_GAMMA_FLOOR as _GAMMA_FLOOR

# =============================================================================
# Effective-slope normalisation (multi-receiver routers)
# =============================================================================

@numba.njit(cache=True)
def _recv_mean_direction(i, receivers, weights, nb_receivers, ny, nx, dx, dy,
                         wrap_y, wrap_x):
    """Flux-weighted mean receiver vector of cell ``i``: ``v = sum_k w_k*e_k``
    over its active receivers, with ``e_k`` the WRAP-AWARE unit vector from i to
    receiver k. Returns ``(vx, vy, gamma)`` with ``gamma = |v|``.

    One definition, two consumers (see ``constants.MFD_GAMMA_FLOOR``):

    * the transect closure orients its line perpendicular to ``v/gamma``
      (:func:`siim._core.step._transect_flux_kernel`, which is where this block
      lived and whose arithmetic it reproduces exactly);
    * ``gamma`` is the effective-slope normalisation -- the multi-receiver
      reduction ``sum_k w_k*(zs_i - zs_rk)/L_k`` equals ``g.v``, so dividing it
      by ``gamma`` turns a projected gradient back into a gradient magnitude.

    ``gamma == 0`` for a self-receiving border cell (its only link is skipped)
    and for a cell whose receiver fan is exactly balanced. Single-receiver
    routers have ``gamma == 1`` identically, so nothing they feed moves.
    """
    iy = i // nx
    ix = i - iy * nx
    Lx = nx * dx
    Ly = ny * dy
    vx = 0.0
    vy = 0.0
    for k in range(nb_receivers[i]):
        r = receivers[i, k]
        if r == i:
            continue
        ry = r // nx
        rx = r - ry * nx
        ddx = (rx - ix) * dx
        ddy = (ry - iy) * dy
        if wrap_x:
            if ddx > 0.5 * Lx:
                ddx -= Lx
            elif ddx < -0.5 * Lx:
                ddx += Lx
        if wrap_y:
            if ddy > 0.5 * Ly:
                ddy -= Ly
            elif ddy < -0.5 * Ly:
                ddy += Ly
        L = math.sqrt(ddx * ddx + ddy * ddy)
        if L > 0.0:
            vx += weights[i, k] * ddx / L
            vy += weights[i, k] * ddy / L
    return vx, vy, math.sqrt(vx * vx + vy * vy)


@numba.njit(cache=True)
def gamma_per_cell(receivers, weights, nb_receivers, ny, nx, dx, dy,
                   wrap_y, wrap_x):
    """Per-cell ``gamma = |sum_k w_k*e_k|``, UNFLOORED -- the diagnostic view
    (probes report min gamma and the fraction of icy cells at the floor)."""
    nn = ny * nx
    out = np.empty(nn)
    for i in range(nn):
        _vx, _vy, g = _recv_mean_direction(i, receivers, weights, nb_receivers,
                                           ny, nx, dx, dy, wrap_y, wrap_x)
        out[i] = g
    return out


@numba.njit(cache=True)
def gamma_scaled_lengths(receivers, weights, nb_receivers, lengths, ny, nx,
                         dx, dy, wrap_y, wrap_x):
    """Per-cell scaled link lengths ``L_eff[i, k] = gamma_i * L[i, k]``, with
    ``gamma_i`` floored at ``constants.MFD_GAMMA_FLOOR``.

    This is how the effective-slope normalisation is applied: every consumer
    that reads ``lengths`` as a slope denominator gets ``S/gamma`` for free, so
    the thickness closure and the eroders see the SAME effective slope. In the
    closure reduction (:func:`siim._core.skeleton._dinf_modeB_recv`) it gives
    ``L_bar_eff = gamma*L_bar`` with ``Z_bar`` unchanged, since both the
    ``sum w/L`` and the ``sum w*zs_r/L`` accumulators scale by ``1/gamma``.

    Inactive slots (``L <= 0``) stay 0, so the producer contract's zeroed tail
    and the slot-0 border sentinel survive untouched.
    """
    nn = ny * nx
    out = np.zeros_like(lengths)
    for i in range(nn):
        _vx, _vy, g = _recv_mean_direction(i, receivers, weights, nb_receivers,
                                           ny, nx, dx, dy, wrap_y, wrap_x)
        if g < _GAMMA_FLOOR:
            g = _GAMMA_FLOOR
        for k in range(lengths.shape[1]):
            L = lengths[i, k]
            if L > 0.0:
                out[i, k] = g * L
    return out


# =============================================================================
# Flow accumulation helpers
# =============================================================================

@numba.njit(cache=True)
def _flow_accumulate_sd(field, stack, receivers):
    for inode in stack[::-1]:
        if receivers[inode] != inode:
            field[receivers[inode]] += max(field[inode], 0)


@numba.njit(cache=True)
def _flow_accumulate_sd_2(field, field_ice, stack, receivers):
    for inode in stack[::-1]:
        r = receivers[inode]
        if r != inode:
            field[r]     += field[inode]
            field_ice[r] += max(field_ice[inode], 0)


# D-inf 2-field accumulator.
#
# D-inf's stack is in donor-first DFS order (donors come before their receivers),
# so a single forward pass distributes each cell's flux to its receivers with the
# corresponding weights. Cells with no receivers (boundaries or self-receiving
# pits) are skipped — they get their accumulation later when *their* donors hit
# them in the same pass.
@numba.njit(cache=True)
def _flow_accumulate_dinf_2(field, field_ice,
                           stack, nb_receivers, receivers, weights):
    for inode in stack:
        n_rec = nb_receivers[inode]
        if n_rec == 0:
            continue
        if n_rec == 1 and receivers[inode, 0] == inode:
            continue
        ice_donor = max(field_ice[inode], 0.0)
        for k in range(n_rec):
            irec = receivers[inode, k]
            w = weights[inode, k]
            field[irec]     += field[inode] * w
            field_ice[irec] += ice_donor * w


@numba.njit(cache=True)
def _flow_accumulate_dinf(field, stack, nb_receivers, receivers, weights):
    """Single-field D-inf accumulator. Donor-first stack."""
    for inode in stack:
        n_rec = nb_receivers[inode]
        if n_rec == 0:
            continue
        if n_rec == 1 and receivers[inode, 0] == inode:
            continue
        val = max(field[inode], 0.0)
        for k in range(n_rec):
            irec = receivers[inode, k]
            field[irec] += val * weights[inode, k]


# =============================================================================
# Tarboton (1997) D-infinity flow router
# -----------------------------------------------------------------------------
# For each cell, the steepest of 8 triangular facets is selected. Flow is
# distributed between the two cardinal/diagonal neighbors that bracket the
# chosen direction with continuous angle-based weights:
#
#     w_e1 = (π/4 - r) / (π/4)         (cardinal neighbor)
#     w_e2 =        r  / (π/4)         (diagonal neighbor)
#
# D-infinity's weights vary continuously as topography evolves, eliminating the
# per-step receiver-flip kicks that would excite the kinematic-wave mode of the
# glacial SPL under a discrete-weight MFR. Up to 2 receivers per cell.
# =============================================================================

# 8 facet offsets. Each is (e1_dj, e1_di, e2_dj, e2_di).
# Order: F0=E-NE, F1=N-NE, F2=N-NW, F3=W-NW, F4=W-SW, F5=S-SW, F6=S-SE, F7=E-SE
_DINF_E1_DJ = np.array([ 0, -1, -1,  0,  0,  1,  1,  0], dtype=np.int64)
_DINF_E1_DI = np.array([ 1,  0,  0, -1, -1,  0,  0,  1], dtype=np.int64)
_DINF_E2_DJ = np.array([-1, -1, -1, -1,  1,  1,  1,  1], dtype=np.int64)
_DINF_E2_DI = np.array([ 1,  1, -1, -1, -1, -1,  1,  1], dtype=np.int64)


@numba.njit(cache=True)
def _heap_push(hz, hi, hn, z, i):
    """Binary min-heap push on parallel (z, idx) arrays; returns new size."""
    k = hn
    hz[k] = z
    hi[k] = i
    while k > 0:
        p = (k - 1) >> 1
        if hz[p] <= hz[k]:
            break
        hz[p], hz[k] = hz[k], hz[p]
        hi[p], hi[k] = hi[k], hi[p]
        k = p
    return hn + 1


@numba.njit(cache=True)
def _heap_pop(hz, hi, hn):
    """Binary min-heap pop; returns (z, idx, new_size)."""
    z0 = hz[0]
    i0 = hi[0]
    hn -= 1
    hz[0] = hz[hn]
    hi[0] = hi[hn]
    k = 0
    while True:
        l = 2 * k + 1
        if l >= hn:
            break
        m = l
        r = l + 1
        if r < hn and hz[r] < hz[l]:
            m = r
        if hz[k] <= hz[m]:
            break
        hz[k], hz[m] = hz[m], hz[k]
        hi[k], hi[m] = hi[m], hi[k]
        k = m
    return z0, i0, hn


@numba.njit(cache=True, inline='always')
def _neighbour(j, i, dj, di, ny, nx, wrap_y, wrap_x):
    """Flat index of the cell ``(j + dj, i + di)``, wrapped on a looped axis;
    -1 past a non-looped edge."""
    jn = j + dj
    if jn < 0 or jn >= ny:
        if not wrap_y:
            return -1
        jn = jn % ny
    ii = i + di
    if ii < 0 or ii >= nx:
        if not wrap_x:
            return -1
        ii = ii % nx
    return jn * nx + ii


@numba.njit(cache=True)
def _priority_flood_eps(z_flat, ny, nx, interior_flat, eps, wrap_y, wrap_x,
                        z_fill):
    """Priority-flood depression filling with an epsilon drainage gradient
    (Barnes, Lehman & Mulla 2014): flood inward from the outlet cells
    (interior_flat == 0, the SFR self-receiving set), each cell filled to
    max(z, spill_path + eps), so on the FILLED surface every interior
    cell has a strictly lower 8-neighbour and lakes drain toward their
    spill on eps-gradients. eps accumulates to ~eps * lake diameter —
    millimetres for any realistic basin. 8-connectivity matches the D-inf
    facet neighbourhood, so routing the filled surface leaves no interior
    pits. Looped axes wrap (wrap_y rows, wrap_x columns).

    The fill is the one field with ``fill = z`` on the outlets and
    ``fill[n] = max(z[n], min over the 8 neighbours c of fill[c] + eps)``
    elsewhere (a minimax path cost, eps added per step), so every exact order
    of evaluation returns the same floats, and the heap is kept to the rims of
    the depressions. A cell reached from a FINAL cell by a step that rises at
    least eps takes ``z``, its lower bound, and is final at once: those cells
    are expanded from a plain FIFO. An expanded cell that leaves a lower
    neighbour unresolved is set aside; when the FIFO runs dry, the ones still
    holding one (the rims) enter a heap keyed by their own fill, and the lowest
    rim raises its unresolved neighbours to ``fill + eps`` — final, since every
    other final neighbour of theirs is a rim with a key at least as large and
    every cell still unresolved ends at least that high. Each cell is expanded
    once, set aside at most once and pushed at most once. ``eps = 0`` (the flat
    fill of the erosion view, ``skeleton._dinf_modeB_filled_view``) is the same
    scheme."""
    nn = ny * nx
    BIG = 1.0e308
    final = np.zeros(nn, dtype=np.uint8)
    work = np.empty(nn, dtype=np.int64)      # FIFO: the final cells to expand
    aside = np.empty(nn, dtype=np.int64)     # expanded, a lower neighbour left
    hz = np.empty(nn, dtype=np.float64)      # the rims' heap: a cell at most once
    hi = np.empty(nn, dtype=np.int64)
    head = 0
    tail = 0
    n_aside = 0
    hn = 0
    for idx in range(nn):
        if interior_flat[idx] == 0:
            z_fill[idx] = z_flat[idx]
            final[idx] = 1
            work[tail] = idx
            tail += 1
        else:
            z_fill[idx] = BIG
    while True:
        # --- expand every final cell waiting
        while head < tail:
            c = work[head]
            head += 1
            floor = z_fill[c] + eps
            j = c // nx
            i = c % nx
            left = False
            for dj in range(-1, 2):
                for di in range(-1, 2):
                    if dj == 0 and di == 0:
                        continue
                    n = _neighbour(j, i, dj, di, ny, nx, wrap_y, wrap_x)
                    if n < 0 or final[n]:
                        continue
                    if z_flat[n] >= floor:
                        z_fill[n] = z_flat[n]
                        final[n] = 1
                        work[tail] = n
                        tail += 1
                    elif z_flat[n] < floor:     # not a bare else: a NaN neighbour is skipped, as the heap flood did
                        left = True
            if left:
                aside[n_aside] = c
                n_aside += 1
        # --- the cells set aside that still hold a lower neighbour: the rims
        for t in range(n_aside):
            c = aside[t]
            floor = z_fill[c] + eps
            j = c // nx
            i = c % nx
            left = False
            for dj in range(-1, 2):
                for di in range(-1, 2):
                    if dj == 0 and di == 0:
                        continue
                    n = _neighbour(j, i, dj, di, ny, nx, wrap_y, wrap_x)
                    if n >= 0 and not final[n] and z_flat[n] < floor:
                        left = True
            if left:
                hn = _heap_push(hz, hi, hn, z_fill[c], c)
        n_aside = 0
        if hn == 0:
            break
        # --- the lowest rim raises what it still holds
        zc, c, hn = _heap_pop(hz, hi, hn)
        floor = zc + eps
        j = c // nx
        i = c % nx
        for dj in range(-1, 2):
            for di in range(-1, 2):
                if dj == 0 and di == 0:
                    continue
                n = _neighbour(j, i, dj, di, ny, nx, wrap_y, wrap_x)
                if n >= 0 and not final[n] and z_flat[n] < floor:
                    z_fill[n] = floor
                    final[n] = 1
                    work[tail] = n
                    tail += 1


@numba.njit(cache=True, parallel=True)
def _dinf_route(z_flat, ny, nx, dx, dy, interior_flat,
                rec1, rec2, w1, w2, len1, len2, slope_out,
                e1_dj, e1_di, e2_dj, e2_di, wrap_y, wrap_x):
    """D-infinity routing. Fills rec1, rec2, w1, w2, len1, len2, slope_out in place.

    The :cite:t:`tarbotonNewMethodDetermination1997` algorithm on a (possibly anisotropic)
    rectangular grid: per facet, the cardinal step d1 is dx or dy depending
    on whether e1 is an E/W or N/S neighbour, and the transverse step d2 is
    the other spacing::

        s1 = (z_c  - z_e1) / d1
        s2 = (z_e1 - z_e2) / d2
        r  = atan2(s2, s1)                   # flow angle within facet
        if r < 0:            r = 0,    s = s1            (along the cardinal)
        if r > atan2(d2,d1): r = that, s = (z_c-z_e2)/sqrt(dx²+dy²)  (diagonal)
        else:                s = sqrt(s1² + s2²)

    The facet with the largest s is selected; flow splits between e1 and e2
    in proportion to the angle within the facet (w_e2 = r / facet_angle —
    the π/4 of the square-grid formula generalises to atan2(d2, d1)).
    Boundary cells self-receive. Looped axes wrap. Routed on the
    eps-filled surface (see _priority_flood_eps), no interior cell pits.

    Row-parallel (``prange`` over j): every cell writes only its own index and
    reads only ``z_flat``, so the result is independent of thread count and
    bit-for-bit identical to the serial scan.
    """
    diag = np.sqrt(dx * dx + dy * dy)
    ang_x = np.arctan2(dy, dx)   # facet angle when e1 is E/W (d1=dx, d2=dy)
    ang_y = np.arctan2(dx, dy)   # facet angle when e1 is N/S (d1=dy, d2=dx)
    for j in numba.prange(ny):
        for i in range(nx):
            idx = j * nx + i
            if not interior_flat[idx]:
                rec1[idx] = idx; rec2[idx] = idx
                w1[idx] = 0.0; w2[idx] = 0.0
                len1[idx] = 0.0; len2[idx] = 0.0
                slope_out[idx] = 0.0
                continue
            z_c = z_flat[idx]
            best_s = 0.0
            best_f = -1
            best_r_val = 0.0
            best_d1 = dx
            best_ang = ang_x
            best_n1 = idx
            best_n2 = idx
            for f in range(8):
                jn1 = j + e1_dj[f]; in1 = i + e1_di[f]
                jn2 = j + e2_dj[f]; in2 = i + e2_di[f]
                if jn1 < 0 or jn1 >= ny:
                    if wrap_y:
                        jn1 = jn1 % ny
                    else:
                        continue
                if in1 < 0 or in1 >= nx:
                    if wrap_x:
                        in1 = in1 % nx
                    else:
                        continue
                if jn2 < 0 or jn2 >= ny:
                    if wrap_y:
                        jn2 = jn2 % ny
                    else:
                        continue
                if in2 < 0 or in2 >= nx:
                    if wrap_x:
                        in2 = in2 % nx
                    else:
                        continue
                if e1_di[f] != 0:
                    d1 = dx; d2 = dy; ang = ang_x
                else:
                    d1 = dy; d2 = dx; ang = ang_y
                z_e1 = z_flat[jn1 * nx + in1]
                z_e2 = z_flat[jn2 * nx + in2]
                s1 = (z_c  - z_e1) / d1
                s2 = (z_e1 - z_e2) / d2
                if s1 == 0.0 and s2 == 0.0:
                    continue
                r = np.arctan2(s2, s1)
                if r < 0.0:
                    r = 0.0
                    s = s1
                elif r > ang:
                    r = ang
                    s = (z_c - z_e2) / diag
                else:
                    s = np.sqrt(s1 * s1 + s2 * s2)
                if s > best_s:
                    best_s = s
                    best_f = f
                    best_r_val = r
                    best_d1 = d1
                    best_ang = ang
                    best_n1 = jn1 * nx + in1
                    best_n2 = jn2 * nx + in2
            if best_f < 0:
                rec1[idx] = idx; rec2[idx] = idx
                w1[idx] = 0.0; w2[idx] = 0.0
                len1[idx] = 0.0; len2[idx] = 0.0
                slope_out[idx] = 0.0
                continue
            r = best_r_val
            w_e2 = r / best_ang
            w_e1 = 1.0 - w_e2
            rec1[idx] = best_n1
            rec2[idx] = best_n2
            w1[idx] = w_e1; w2[idx] = w_e2
            len1[idx] = best_d1; len2[idx] = diag
            slope_out[idx] = best_s


@numba.njit(cache=True)
def _dinf_topo_stack(rec1, rec2, w1, w2, n, stack):
    """Topological sort (Kahn's algorithm) for D-inf graph. Emits cells in
    *receivers-first* order (base cells first, donors last). Caller must
    reverse for fastscape's donor-first convention."""
    out_deg = np.zeros(n, dtype=np.int64)
    for i in range(n):
        r1_real = (w1[i] > 0.0) and (rec1[i] != i)
        r2_real = (w2[i] > 0.0) and (rec2[i] != i) and (rec2[i] != rec1[i])
        if r1_real: out_deg[i] += 1
        if r2_real: out_deg[i] += 1
    # CSR donor list
    donor_count = np.zeros(n, dtype=np.int64)
    for i in range(n):
        if w1[i] > 0.0 and rec1[i] != i:
            donor_count[rec1[i]] += 1
        if w2[i] > 0.0 and rec2[i] != i and rec2[i] != rec1[i]:
            donor_count[rec2[i]] += 1
    donor_off = np.zeros(n + 1, dtype=np.int64)
    for i in range(n):
        donor_off[i + 1] = donor_off[i] + donor_count[i]
    cursor = donor_off[:-1].copy()
    donor_list = np.zeros(donor_off[n], dtype=np.int64)
    for i in range(n):
        if w1[i] > 0.0 and rec1[i] != i:
            r = rec1[i]
            donor_list[cursor[r]] = i; cursor[r] += 1
        if w2[i] > 0.0 and rec2[i] != i and rec2[i] != rec1[i]:
            r = rec2[i]
            donor_list[cursor[r]] = i; cursor[r] += 1
    # BFS from base cells (out_deg == 0) outward
    queue = np.zeros(n, dtype=np.int64)
    qhead = 0; qtail = 0
    for i in range(n):
        if out_deg[i] == 0:
            queue[qhead] = i; qhead += 1
    head = 0
    while qtail < qhead:
        j = queue[qtail]; qtail += 1
        stack[head] = j; head += 1
        for k in range(donor_off[j], donor_off[j + 1]):
            d = donor_list[k]
            out_deg[d] -= 1
            if out_deg[d] == 0:
                queue[qhead] = d; qhead += 1
    # Defensive: any unemitted cells (shouldn't happen with valid routing)
    if head < n:
        for i in range(n):
            if out_deg[i] > 0:
                stack[head] = i; head += 1


@numba.njit(cache=True)
def _dinf_pack(rec1, rec2, w1, w2, len1, len2, n,
               receivers, weights, lengths, nb_receivers):
    """Pack D-inf routing into (n, 2) receiver/weight arrays.

    Single-receiver cases (pit/boundary, both-receivers-same-cell, only one
    weight nonzero) get nb_receivers[i]=1 with the receiver at index 0.
    Two-receiver cells get nb_receivers[i]=2.
    """
    for i in range(n):
        if w1[i] == 0.0 and w2[i] == 0.0:
            receivers[i, 0] = i;        receivers[i, 1] = i
            weights[i, 0] = 1.0;        weights[i, 1] = 0.0
            lengths[i, 0] = 0.0;        lengths[i, 1] = 0.0
            nb_receivers[i] = 1
        elif rec1[i] == rec2[i]:
            receivers[i, 0] = rec1[i];  receivers[i, 1] = rec1[i]
            weights[i, 0] = w1[i] + w2[i]; weights[i, 1] = 0.0
            lengths[i, 0] = len1[i];    lengths[i, 1] = 0.0
            nb_receivers[i] = 1
        elif w2[i] == 0.0:
            receivers[i, 0] = rec1[i];  receivers[i, 1] = rec1[i]
            weights[i, 0] = w1[i];      weights[i, 1] = 0.0
            lengths[i, 0] = len1[i];    lengths[i, 1] = 0.0
            nb_receivers[i] = 1
        elif w1[i] == 0.0:
            receivers[i, 0] = rec2[i];  receivers[i, 1] = rec2[i]
            weights[i, 0] = w2[i];      weights[i, 1] = 0.0
            lengths[i, 0] = len2[i];    lengths[i, 1] = 0.0
            nb_receivers[i] = 1
        else:
            receivers[i, 0] = rec1[i];  receivers[i, 1] = rec2[i]
            weights[i, 0] = w1[i];      weights[i, 1] = w2[i]
            lengths[i, 0] = len1[i];    lengths[i, 1] = len2[i]
            nb_receivers[i] = 2


# =============================================================================
# Multiple-flow-direction (MFD) router — ``flow_routing='mfd'``
# -----------------------------------------------------------------------------
# A cell's flux is split over ALL strictly downhill 8-neighbours of the SAME
# eps-filled surface the D8/D-inf producers route (``_priority_flood_eps``), with
# weights (at p = inf, where they are exactly 1 on the steepest link and 0 on
# the others, the pack holds the steepest link alone)
#
#     w_k = (S_k / S_max)^p / sum_j (S_j / S_max)^p ,   S_k = (z_i - z_k) / L_k
#
# (``p = constants.MFD_EXPONENT``; the ``S_max`` normalisation is what keeps the
# p -> inf limit and crest conditioning well-behaved). Every admitted link
# strictly descends one scalar potential, so the graph is a DAG by construction
# — no tie-break, no flat handling, no fortran-parity obligation (ties are benign
# here: both neighbours simply get weight).
#
# PRODUCER CONTRACT (``docs/dev/router_contract.md``; the pack is (nn, 8)):
#   * active receivers occupy the CONTIGUOUS prefix ``0 .. nb_receivers[i]-1``;
#   * every inactive tail slot is zeroed in all three arrays
#     (``receivers = 0, weights = 0, lengths = 0``);
#   * ``receivers[i, 0] == i`` (with ``nb = 1``, ``w = 1``, ``L = 0``) occurs ONLY
#     at true border sinks — four consumers read slot 0 as the border marker;
#   * no duplicate links (guaranteed by rejecting periodic axes below 3 cells,
#     where the -1 and +1 offsets alias onto the same node);
#   * ``sum_k w_k == 1`` and ``lengths`` Euclidean.
# =============================================================================

# 8 neighbour offsets, in the SAME (jj outer, ii inner) scan order as
# ``_d8_receivers`` — so on a tie-free surface the largest-weight slot is the
# neighbour D8 would pick, and the primary-receiver output view agrees with D8.
_MFD_DJ = np.array([-1, -1, -1,  0,  0,  1,  1,  1], dtype=np.int64)
_MFD_DI = np.array([-1,  0,  1, -1,  1, -1,  0,  1], dtype=np.int64)


@numba.njit(cache=True, parallel=True)
def _mfd_route(z_flat, ny, nx, dx, dy, interior_flat, p,
               receivers, weights, lengths, nb_receivers, dj, di,
               wrap_y, wrap_x, link_mask, out_degree):
    """MFD receiver scan on the (eps-filled) surface. Fills ``receivers``,
    ``weights``, ``lengths`` ``(nn, 8)`` and ``nb_receivers`` ``(nn,)`` in place,
    in the producer-contract layout described in the block comment above, and,
    for :func:`_mfd_stack`, ``link_mask`` (bit f set: a link of positive weight
    to neighbour f) and ``out_degree`` (their count), both ``(nn,)`` uint8.

    One neighbour pass per cell (the 8 slopes are kept, ``S_max`` taken), then
    the active prefix is written with the raw ``(S/S_max)^p`` weights and
    normalised in place by their sum; no per-cell allocation. ``p = 1`` skips
    the power (``x**1`` is ``x``).
    At ``p = inf`` the raw weights are exactly 1 or 0 and a zero is not a
    link: the steepest link alone (equal steepest links share). A finite ``p``
    writes every downhill link, as before.
    A non-interior cell gets the slot-0 self sentinel. An interior cell with no
    strictly-lower neighbour cannot occur on the eps-filled surface (see
    ``_priority_flood_eps``); it would need ``|z|`` ~ 1e10 for ``z + eps`` to
    round back to ``z``. It is handled the way the D8/D-inf producers handle their own
    ``best_f < 0`` branch — the self sentinel — and the contract test asserts it
    never fires on an interior cell.

    Row-parallel (``prange`` over j): every cell writes only its own row of the
    output arrays and reads only ``z_flat``, so the result is independent of
    thread count and bit-for-bit identical to the serial scan."""
    nslot = receivers.shape[1]
    steepest_alone = np.isinf(p)
    link_length = np.empty(8)
    for f in range(8):
        link_length[f] = np.sqrt((dx * di[f]) ** 2 + (dy * dj[f]) ** 2)
    for j in numba.prange(ny):
        slope = np.empty(8)                  # per-row scratch: the 8 neighbours
        node = np.empty(8, dtype=np.int64)
        direction = np.empty(8, dtype=np.int64)
        for i in range(nx):
            idx = j * nx + i
            for k in range(nslot):          # zeroed tail by default
                receivers[idx, k] = 0
                weights[idx, k] = 0.0
                lengths[idx, k] = 0.0
            link_mask[idx] = 0
            out_degree[idx] = 0
            if not interior_flat[idx]:
                receivers[idx, 0] = idx     # border sink marker
                weights[idx, 0] = 1.0
                lengths[idx, 0] = 0.0
                nb_receivers[idx] = 1
                continue
            z_c = z_flat[idx]
            # --- the 8 slopes (0 past a non-looped edge) and the steepest
            s_max = 0.0
            for f in range(8):
                slope[f] = 0.0
                jn = j + dj[f]
                if jn < 0 or jn >= ny:
                    if wrap_y:
                        jn = jn % ny
                    else:
                        continue
                in_ = i + di[f]
                if in_ < 0 or in_ >= nx:
                    if wrap_x:
                        in_ = in_ % nx
                    else:
                        continue
                node[f] = jn * nx + in_
                slope[f] = (z_c - z_flat[node[f]]) / link_length[f]
                if slope[f] > s_max:
                    s_max = slope[f]
            if s_max <= 0.0:                # unreachable on the filled surface
                receivers[idx, 0] = idx
                weights[idx, 0] = 1.0
                lengths[idx, 0] = 0.0
                nb_receivers[idx] = 1
                continue
            # --- the strictly-downhill prefix, raw (S/S_max)^p weights
            nb = 0
            tot = 0.0
            for f in range(8):
                s = slope[f]
                if s <= 0.0:
                    continue
                wraw = s / s_max if p == 1.0 else (s / s_max) ** p
                if steepest_alone and wraw == 0.0:
                    continue
                receivers[idx, nb] = node[f]
                weights[idx, nb] = wraw
                lengths[idx, nb] = link_length[f]
                direction[nb] = f
                tot += wraw
                nb += 1
            nb_receivers[idx] = nb
            mask = 0
            for k in range(nb):
                weights[idx, k] = weights[idx, k] / tot
                if weights[idx, k] > 0.0:
                    mask |= 1 << direction[k]
                    out_degree[idx] += 1
            link_mask[idx] = mask


@numba.njit(cache=True, parallel=True)
def _mfd_donor_mask(link_mask, ny, nx, dj, di, wrap_y, wrap_x, donor_mask):
    """``donor_mask[r]`` bit f set: neighbour f of r links INTO r, i.e. its own
    ``link_mask`` carries the opposite direction, ``7 - f`` (the 8 offsets come
    in opposite pairs f, 7 - f). Row-parallel gather: a cell writes only its
    own entry."""
    for j in numba.prange(ny):
        for i in range(nx):
            mask = 0
            for f in range(8):
                jn = j + dj[f]
                if jn < 0 or jn >= ny:
                    if wrap_y:
                        jn = jn % ny
                    else:
                        continue
                in_ = i + di[f]
                if in_ < 0 or in_ >= nx:
                    if wrap_x:
                        in_ = in_ % nx
                    else:
                        continue
                if (link_mask[jn * nx + in_] >> (7 - f)) & 1:
                    mask |= 1 << f
            donor_mask[j * nx + i] = mask


@numba.njit(cache=True)
def _mfd_stack(link_mask, out_degree, ny, nx, dj, di, wrap_y, wrap_x, stack):
    """DONOR-FIRST topological order of the k-ary MFD graph (Kahn's algorithm),
    written into ``stack``, from :func:`_mfd_route`'s ``link_mask`` and
    ``out_degree`` (left untouched): the links of positive weight, the self
    sentinel of a border sink not among them.

    The k-ary sibling of :func:`_dinf_topo_stack` — which stays a two-receiver
    producer internal and is deliberately NOT generalised in place. No
    ``rec2 != rec1`` de-duplication is needed here: the producer emits each
    neighbour at most once (periodic axes shorter than 3 cells, where opposite
    offsets alias, are rejected by :func:`siim._core.step.route_mfd`).

    A cell's donors are read off the 8 neighbours' masks
    (:func:`_mfd_donor_mask`) and released in ASCENDING node index, the order a
    donor list filled cell by cell holds — the order of the stack sets the
    order every accumulation adds in, so it is part of the result. The offsets
    f = 0..7 ascend in node index off the domain edge; on an edge row or
    column, where a looped axis reorders them, the donors are sorted.

    Kahn emits base cells first (receivers-first); the output is REVERSED into
    ``stack`` so it lands in the donor-first convention
    ``_flow_accumulate_dinf`` walks forward — the same convention D-inf's
    ``route_dinf`` produces by reversing outside the kernel."""
    n = ny * nx
    donor_mask = np.empty(n, dtype=np.uint8)
    _mfd_donor_mask(link_mask, ny, nx, dj, di, wrap_y, wrap_x, donor_mask)
    out_deg = out_degree.copy()
    offset = np.empty(8, dtype=np.int64)
    for f in range(8):
        offset[f] = dj[f] * nx + di[f]
    donors = np.empty(8, dtype=np.int64)
    queue = np.empty(n, dtype=np.int64)      # the emission order: base cells first
    qhead = 0
    qtail = 0
    for i in range(n):
        if out_deg[i] == 0:
            queue[qhead] = i
            qhead += 1
    while qtail < qhead:
        c = queue[qtail]
        qtail += 1
        mask = donor_mask[c]
        if mask == 0:
            continue
        j = c // nx
        i = c % nx
        nd = 0
        if 0 < j < ny - 1 and 0 < i < nx - 1:
            for f in range(8):
                if (mask >> f) & 1:
                    donors[nd] = c + offset[f]
                    nd += 1
        else:
            for f in range(8):
                if (mask >> f) & 1:
                    d = ((j + dj[f]) % ny) * nx + (i + di[f]) % nx
                    k = nd                   # insertion: ascending node index
                    while k > 0 and donors[k - 1] > d:
                        donors[k] = donors[k - 1]
                        k -= 1
                    donors[k] = d
                    nd += 1
        for k in range(nd):
            d = donors[k]
            out_deg[d] -= 1
            if out_deg[d] == 0:
                queue[qhead] = d
                qhead += 1
    # Defensive: any unemitted cell (would signal a cycle — impossible on the
    # eps-filled surface) is appended so ``stack`` stays a full permutation.
    if qhead < n:
        for i in range(n):
            if out_deg[i] > 0:
                queue[qhead] = i
                qhead += 1
    for t in range(n):
        stack[n - 1 - t] = queue[t]


# =============================================================================
# In-house D8 single-flow router (fill-then-route) — the framework-free
# replacement for fastscape's fortran ``SingleFlowRouter``
# (``fs.flowroutingsingleflowdirection``). Same fill-then-route strategy as the
# D-inf primitives above: spill-fill the depressions with _priority_flood_eps,
# then a steepest-descent receiver scan on the FILLED surface that replicates
# fortran ``find_receiver`` exactly. The routing delta vs fortran is confined to
# tie/flat/sill cells (in-house routes them down the eps-fill gradient, fortran
# via MST sill-carving); on non-degenerate cells the receiver + length are
# byte-identical, so this is a behavioral/attractor swap, not bit-for-bit.
# =============================================================================

# fortran ``find_receiver`` inits its steepest-slope accumulator to tiny(0.d0)
# (FlowRouting.f90:332) and keeps a neighbour on STRICT slope > smax; replicate
# that init so the tie-break (first-in-scan-order wins) matches fortran exactly.
_D8_TINY = np.finfo(np.float64).tiny


@numba.njit(cache=True)
def _d8_receivers(z_flat, ny, nx, dx, dy, interior_flat, rec, lengths,
                  wrap_y, wrap_x):
    """Steepest-descent D8 receiver, replicating fortran ``find_receiver``
    (``FlowRouting.f90:302-362``) on the (eps-filled) surface: outer
    ``jj in {-1,0,1}`` x inner ``ii in {-1,0,1}`` scan (skipping ``(0,0)``),
    ``l = sqrt((dx*ii)**2 + (dy*jj)**2)``, keep on STRICT ``slope > smax`` with
    ``smax`` initialised to ``tiny`` — so steepest descent wins and, on a tie,
    the first neighbour in scan order does (byte-identical to fortran on
    tie-free surfaces). Non-interior cells self-receive (``rec[i]=i,
    lengths[i]=0``, the ``bounds_bc`` border contract). On the eps-filled
    surface every interior cell has a strictly-lower 8-neighbour, so it always
    finds one (no interior self-receivers => acyclic graph). Looped axes wrap.

    Serial: each cell writes only its own ``rec``/``lengths`` and reads only
    ``z_flat``, so the result is independent of iteration order (a prange over
    the outer loop would be bit-for-bit identical; kept serial per Map 4 §4)."""
    for idx in range(ny * nx):
        if interior_flat[idx] == 0:
            rec[idx] = idx
            lengths[idx] = 0.0
            continue
        j = idx // nx
        i = idx % nx
        z_c = z_flat[idx]
        smax = _D8_TINY
        best = idx
        best_l = 0.0
        for jj in range(-1, 2):
            jn = j + jj
            if jn < 0 or jn >= ny:
                if wrap_y:
                    jn = jn % ny
                else:
                    continue
            for ii in range(-1, 2):
                if jj == 0 and ii == 0:
                    continue
                in_ = i + ii
                if in_ < 0 or in_ >= nx:
                    if wrap_x:
                        in_ = in_ % nx
                    else:
                        continue
                n = jn * nx + in_
                l = np.sqrt((dx * ii) ** 2 + (dy * jj) ** 2)
                slope = (z_c - z_flat[n]) / l
                if slope > smax:
                    smax = slope
                    best = n
                    best_l = l
        rec[idx] = best
        lengths[idx] = best_l


@numba.njit(cache=True)
def _d8_stack(rec, n, stack):
    """Outlet-first topological stack for the single-receiver graph: base cells
    (``rec[i]==i``) first, then their donors (receiver-before-donor). Donor-CSR
    BFS from the base cells — a valid topological order, which is all the
    accumulators/eroders require (fortran's exact DFS visitation order is NOT
    contracted, Map 3 §3(iii)). On the eps-filled surface ``rec`` is a forest of
    trees rooted at the border/base cells, so the BFS emits every cell exactly
    once."""
    donor_count = np.zeros(n, dtype=np.int64)
    for i in range(n):
        r = rec[i]
        if r != i:
            donor_count[r] += 1
    donor_off = np.zeros(n + 1, dtype=np.int64)
    for i in range(n):
        donor_off[i + 1] = donor_off[i] + donor_count[i]
    cursor = donor_off[:-1].copy()
    donor_list = np.zeros(donor_off[n], dtype=np.int64)
    for i in range(n):
        r = rec[i]
        if r != i:
            donor_list[cursor[r]] = i
            cursor[r] += 1
    queue = np.zeros(n, dtype=np.int64)
    qhead = 0
    qtail = 0
    for i in range(n):
        if rec[i] == i:
            queue[qhead] = i
            qhead += 1
    head = 0
    while qtail < qhead:
        c = queue[qtail]
        qtail += 1
        stack[head] = c
        head += 1
        for k in range(donor_off[c], donor_off[c + 1]):
            d = donor_list[k]
            queue[qhead] = d
            qhead += 1
    # Defensive: any cell not reached (would signal a cycle — impossible on the
    # eps-filled surface) is appended so ``stack`` stays a full permutation.
    if head < n:
        seen = np.zeros(n, dtype=np.uint8)
        for t in range(head):
            seen[stack[t]] = 1
        for i in range(n):
            if not seen[i]:
                stack[head] = i
                head += 1


@numba.njit(cache=True)
def _d8_basin(rec, stack, n, basin):
    """Basin id of each cell = the index of the base (outlet) cell its receiver
    chain terminates at. One forward pass over the outlet-first ``stack`` (the
    receiver is always processed before its donors), so ``basin`` propagates
    from each outlet down its tributaries. Labeling by outlet INDEX makes the
    ids reproducible run-to-run (unlike fortran's unseeded ``random_number``
    catch labels, Map 4 §4) — but ``basin`` is a diagnostic and stays out of
    every equality gate regardless."""
    for idx in range(n):
        c = stack[idx]
        r = rec[c]
        if r == c:
            basin[c] = c
        else:
            basin[c] = basin[r]


def d8_interior_mask(border_status, ny, nx):
    """The ``interior`` mask (1 = routable interior cell, 0 = self-receiving
    boundary) directly from ``border_status`` — the in-house replacement for the
    fortran-derived ``sfr_rec != arange`` mask (Map 3 §4). Provably identical:
    after fortran ``LocalMinima`` the only self-receiving cells are ``bounds_bc``
    = exactly the 'fixed_value' border rings (``FastScape_ctx.f90:589-592``).

    'core' and 'looped' are both plain interior for EVERY router (OQ-1(b),
    ratified): 'core' is non-periodic interior (periodicity is the separate wrap
    axis, keyed on 'looped' only), aligning the SFR mask with the D-inf
    convention siim already shipped. ``border_status`` = [left, right, bottom,
    top] (``outputs.SIDES``); node index = ``j*nx + i``."""
    interior = np.ones(ny * nx, dtype=np.int8)
    bs = list(np.broadcast_to(border_status, 4))
    if bs[0] == 'fixed_value':
        interior[0::nx] = 0            # left column   (i = 0)
    if bs[1] == 'fixed_value':
        interior[nx - 1::nx] = 0       # right column  (i = nx-1)
    if bs[2] == 'fixed_value':
        interior[:nx] = 0              # bottom row    (j = 0)
    if bs[3] == 'fixed_value':
        interior[nx * (ny - 1):] = 0   # top row       (j = ny-1)
    return interior


# =============================================================================
# Topological level index for the level-scheduled parallel eroders
# (eroders._erode_modeb_*_levels, the ``parallel_erode`` toggle). A node's
# level is 1 + the max level over its (real) receivers, so every node in a
# level depends only on strictly lower levels — levels run in order, nodes
# within a level in parallel. Nodes are bucketed in stack order (stable), so
# the ordering is deterministic; correctness does not depend on it (writes
# within a level are disjoint).
# =============================================================================

@numba.njit(cache=True)
def _levels_sfr(stack, rec):
    """Level index for the SFR graph. Returns (order, offsets, nlev):
    ``order[offsets[l]:offsets[l+1]]`` are the nodes of level ``l``."""
    n = stack.shape[0]
    level = np.zeros(n, dtype=np.int64)
    nlev = 1
    for inode in stack:                       # receivers before donors
        r = rec[inode]
        if r != inode:
            lv = level[r] + 1
            level[inode] = lv
            if lv + 1 > nlev:
                nlev = lv + 1
    offsets = np.zeros(nlev + 1, dtype=np.int64)
    for i in range(n):
        offsets[level[i] + 1] += 1
    for l in range(nlev):
        offsets[l + 1] += offsets[l]
    order = np.empty(n, dtype=np.int64)
    cursor = offsets[:-1].copy()
    for inode in stack:
        lv = level[inode]
        order[cursor[lv]] = inode
        cursor[lv] += 1
    return order, offsets, nlev


@numba.njit(cache=True)
def _levels_dinf(stack, nb_receivers, receivers):
    """Level index for the D-inf graph (level = 1 + max over real receivers).
    Same return contract as :func:`_levels_sfr`; nodes are bucketed in
    receivers-first (reversed donor-first stack) order."""
    n = stack.shape[0]
    level = np.zeros(n, dtype=np.int64)
    nlev = 1
    for idx in range(n - 1, -1, -1):          # receivers before donors
        i = stack[idx]
        if receivers[i, 0] == i:
            continue
        lv = 0
        for k in range(nb_receivers[i]):
            r = receivers[i, k]
            if r == i:
                continue
            if level[r] + 1 > lv:
                lv = level[r] + 1
        level[i] = lv
        if lv + 1 > nlev:
            nlev = lv + 1
    offsets = np.zeros(nlev + 1, dtype=np.int64)
    for i in range(n):
        offsets[level[i] + 1] += 1
    for l in range(nlev):
        offsets[l + 1] += offsets[l]
    order = np.empty(n, dtype=np.int64)
    cursor = offsets[:-1].copy()
    for idx in range(n - 1, -1, -1):
        i = stack[idx]
        lv = level[i]
        order[cursor[lv]] = i
        cursor[lv] += 1
    return order, offsets, nlev
