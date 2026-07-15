"""Flow routing & accumulation primitives for the siim numerical core.

The fastscape-free numba kernels -- SFR + D-inf accumulators, the eps-fill
priority flood, and the Tarboton (1997) D-inf routing primitives. Pulling them
out of the fastscape-importing model is what lets the solver kernels import
without the fastscape stack. The ``DinfFlowRouter`` xsimlab *process* that
consumes them lives in :mod:`siim.fastscape.processes`. numpy/numba only.
"""
import numpy as np
import numba

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
    pits. Looped axes wrap (wrap_y rows, wrap_x columns)."""
    nn = ny * nx
    BIG = 1.0e308
    for i in range(nn):
        z_fill[i] = BIG
    visited = np.zeros(nn, dtype=np.uint8)
    # capacity: a cell is re-pushed only on strict improvement, at most
    # once per neighbour pop -> <= 8 pushes per cell + the seeds
    cap = 9 * nn + 16
    hz = np.empty(cap, dtype=np.float64)
    hi = np.empty(cap, dtype=np.int64)
    hn = 0
    for idx in range(nn):
        if interior_flat[idx] == 0:
            z_fill[idx] = z_flat[idx]
            hn = _heap_push(hz, hi, hn, z_flat[idx], idx)
    while hn > 0:
        zc, c, hn = _heap_pop(hz, hi, hn)
        if visited[c]:
            continue
        visited[c] = 1
        j = c // nx
        i = c % nx
        for dj in range(-1, 2):
            jn = j + dj
            if jn < 0 or jn >= ny:
                if wrap_y:
                    jn = jn % ny
                else:
                    continue
            for di in range(-1, 2):
                if dj == 0 and di == 0:
                    continue
                ii = i + di
                if ii < 0 or ii >= nx:
                    if wrap_x:
                        ii = ii % nx
                    else:
                        continue
                n = jn * nx + ii
                if visited[n]:
                    continue
                cand = z_flat[n]
                floor = zc + eps
                if cand < floor:
                    cand = floor
                if cand < z_fill[n]:
                    z_fill[n] = cand
                    hn = _heap_push(hz, hi, hn, cand, n)


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
