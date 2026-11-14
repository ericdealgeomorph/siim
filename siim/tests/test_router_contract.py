"""The router conformance battery.

Every conforming flow router runs through the HARD requirements. The
framework-free primitives (spill, flux conservation, stack validity, border
contract, seam invariance, SFR/D-inf cardinal parity) run on the in-house
producers directly (``route_d8`` / ``route_dinf``); the analytical
steady-state oracle at coarse tolerance runs a real siim2d model on the
shipped (in-house) backend. The retired fortran router passed this battery at
S4 (recorded in the plan); its arm died with the fortran flag at the S5 flip."""
import numpy as np
import pytest

from siim._core.step import route_d8, route_dinf, route_mfd
from siim._core.routing import _flow_accumulate_sd, _flow_accumulate_dinf

NY, NX = 15, 19
DX = DY = 100.0
BS_LOOP = ['fixed_value', 'fixed_value', 'looped', 'looped']
BS_FIXED = ['fixed_value'] * 4


# Every conforming producer, by kind. 'mfd' carries its exponent, so it is
# dispatched explicitly -- the helper used to map every non-D8 kind to D-inf.
def _route(kind, z, bs, p=1.0):
    if kind == 'd8':
        return route_d8(z, (NY, NX), DX, DY, bs)
    if kind == 'mfd':
        return route_mfd(z, (NY, NX), DX, DY, bs, p)
    return route_dinf(z, (NY, NX), DX, DY, bs)


def _interior(bs):
    m = np.ones(NY * NX, dtype=bool)
    bb = list(np.broadcast_to(bs, 4))
    if bb[0] == 'fixed_value': m[0::NX] = False
    if bb[1] == 'fixed_value': m[NX - 1::NX] = False
    if bb[2] == 'fixed_value': m[:NX] = False
    if bb[3] == 'fixed_value': m[NX * (NY - 1):] = False
    return m


def _area(kind, receivers, weights, nb_receivers, stack):
    field = np.ones(NY * NX)
    if kind == 'd8':
        _flow_accumulate_sd(field, stack, receivers)
    else:
        _flow_accumulate_dinf(field, stack, np.asarray(nb_receivers, np.int64),
                              np.asarray(receivers, np.int64),
                              np.asarray(weights, np.float64))
    return field


def _self_receiving(kind, receivers):
    if kind == 'd8':
        return receivers == np.arange(NY * NX)
    return receivers[:, 0] == np.arange(NY * NX)


def _pitted_surface(bs):
    """Tilts down to the left fixed border (drains -x) with a Gaussian pit near
    the middle — an interior depression the router must spill-correct."""
    jj, ii = np.meshgrid(np.arange(NY), np.arange(NX), indexing='ij')
    z = 100.0 + 0.5 * ii.astype(float)
    z -= 30.0 * np.exp(-(((ii - NX // 2) ** 2 + (jj - NY // 2) ** 2) / 6.0))
    return z


# --- HARD requirements (framework-free, both producers) ----------------------

@pytest.mark.parametrize("kind", ["d8", "dinf", "mfd"])
@pytest.mark.parametrize("bs", [BS_FIXED, BS_LOOP], ids=["fixed", "looped"])
def test_spill_and_stack_validity(kind, bs):
    z = _pitted_surface(bs)
    rec, w, lengths, nb, stack, basin = _route(kind, z, bs)
    interior = _interior(bs)
    selfr = _self_receiving(kind, rec)
    # spill: no interior cell self-receives (every interior cell drains)
    assert not np.any(interior & selfr), "an interior cell is an undrained pit"
    # stack is a permutation + a valid topological (receiver-before-donor) order
    assert len(np.unique(stack)) == NY * NX, "stack is not a permutation"
    pos = np.empty(NY * NX, dtype=np.int64)
    pos[stack] = np.arange(NY * NX)
    # The stack is a valid topological order for the accumulation walk: SFR is
    # outlet-first (receiver before donor; the accumulator walks it reversed),
    # D-inf is donor-first (donor before receiver; walked forward). Either way,
    # every donor->receiver edge is monotone in the walk direction.
    if kind == 'd8':
        assert np.all(pos[rec] <= pos), "SFR stack not outlet-first"
    else:
        for i in range(NY * NX):
            for k in range(nb[i]):
                r = rec[i, k]
                if r != i and w[i, k] > 0:
                    assert pos[i] <= pos[r], "D-inf stack not donor-first"
    # no cycles: following receivers from any interior cell reaches a border
    for start in np.where(interior)[0][::7]:
        c, seen = start, 0
        while not selfr[c] and seen <= NY * NX:
            c = rec[c] if kind == 'd8' else rec[c, np.argmax(w[c])]
            seen += 1
        assert seen <= NY * NX, "receiver chain cycles"


@pytest.mark.parametrize("kind", ["d8", "dinf", "mfd"])
@pytest.mark.parametrize("bs", [BS_FIXED, BS_LOOP], ids=["fixed", "looped"])
def test_flux_conservation(kind, bs):
    z = _pitted_surface(bs)
    rec, w, lengths, nb, stack, basin = _route(kind, z, bs)
    area = _area(kind, rec, w, nb, stack)
    # mass conservation: total drainage area gathered at outlets == n cells
    outlets = _self_receiving(kind, rec)
    assert area[outlets].sum() == pytest.approx(NY * NX)
    assert np.all(area >= 1.0 - 1e-9), "a cell gathered less than its own unit"


@pytest.mark.parametrize("kind", ["d8", "dinf", "mfd"])
def test_border_contract(kind):
    z = _pitted_surface(BS_LOOP)
    rec, w, lengths, nb, stack, basin = _route(kind, z, BS_LOOP)
    border = ~_interior(BS_LOOP)
    idx = np.arange(NY * NX)
    assert np.all(_self_receiving(kind, rec)[border]), "a border cell is not self-receiving"
    if kind == 'd8':
        assert np.all(rec[border] == idx[border])
        assert np.all(lengths[border] == 0.0)


@pytest.mark.parametrize("kind", ["d8", "dinf", "mfd"])
def test_seam_shift_invariance(kind):
    """Rolling the surface along a looped (y) axis rolls the drainage-area field
    by the same shift — the periodic seam is handled without a discontinuity."""
    z = _pitted_surface(BS_LOOP)
    rec, w, lengths, nb, stack, basin = _route(kind, z, BS_LOOP)
    area = _area(kind, rec, w, nb, stack).reshape(NY, NX)
    shift = 4
    zr = np.roll(z, shift, axis=0)
    rr, wr, lr, nbr, sr, br = _route(kind, zr, BS_LOOP)
    area_r = _area(kind, rr, wr, nbr, sr).reshape(NY, NX)
    assert np.allclose(area_r, np.roll(area, shift, axis=0), rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("kind", ["dinf", "mfd"])
def test_multi_receiver_primary_matches_d8_on_a_cardinal_plane(kind):
    """On a cardinal plane the D8 receiver == the multi-receiver producer's
    primary (largest-weight) receiver."""
    jj, ii = np.meshgrid(np.arange(NY), np.arange(NX), indexing='ij')
    z = (5.0 + 0.01 * DX * ii).astype(float)          # drains -x
    rec8, _w, _l, _nr, _s, _b = route_d8(z, (NY, NX), DX, DY, BS_FIXED)
    recm, wm, _l2, _nbm, _s2, _b2 = _route(kind, z, BS_FIXED)
    primary = recm[np.arange(NY * NX), np.argmax(wm, axis=1)]
    interior = _interior(BS_FIXED)
    assert np.array_equal(rec8[interior], primary[interior])


def test_sfr_dinf_cardinal_parity():
    """On a cardinal plane the D8 receiver == the D-inf primary receiver."""
    jj, ii = np.meshgrid(np.arange(NY), np.arange(NX), indexing='ij')
    z = (5.0 + 0.01 * DX * ii).astype(float)          # drains -x
    rec8, _w, _l, _nr, _s, _b = route_d8(z, (NY, NX), DX, DY, BS_FIXED)
    recd, wd, _l2, nbd, _s2, _b2 = route_dinf(z, (NY, NX), DX, DY, BS_FIXED)
    primary = recd[np.arange(NY * NX), np.argmax(wd, axis=1)]
    interior = _interior(BS_FIXED)
    assert np.array_equal(rec8[interior], primary[interior])


# --- analytical steady-state oracle at coarse tolerance ----------------------

def test_analytical_oracle_coarse():
    """A mode-A glacial 2D run on the shipped (in-house) router reproduces the
    1D analytical channel to a coarse tolerance (pinned <25% by test_baseline;
    the conforming-router requirement). The retired fortran backend passed the
    same oracle at S4 (14.66%; plan S4 notes)."""
    from siim.siim2d import siim as siim2d
    from siim.siim1d import siim as siim1d
    from .test_baseline import (_small_2d, _params_1d_from_2d_channel,
                                _channel_rms_pct)
    p2d = _small_2d(zELA=1000, T=1e6, nt=501, nt_out=26,
                    initial_max_elevation=500, mode='A')
    m2 = siim2d(p2d)
    m2.run()
    ch = m2.extract_channel(basin_rank=0)
    m1 = siim1d(_params_1d_from_2d_channel(p2d, ch))
    m1.run()
    rms = _channel_rms_pct(ch.z[-1], m1, ch)
    assert rms < 30.0, f"channel-vs-analytical RMS {rms:.1f}% (>30%)"


# --- MFD producer-contract invariants ----------------------------------------
# The (nn, 8) pack has structural requirements the two-receiver producers get
# for free: a contiguous ACTIVE PREFIX, a zeroed tail, the slot-0 border
# sentinel at borders only, no duplicate links. Four consumers read slot 0 as
# the border marker (skeleton.py, step.py's carve), so a stray interior
# self-receiver would silently reclassify an interior cell as base level.

@pytest.mark.parametrize("bs", [BS_FIXED, BS_LOOP], ids=["fixed", "looped"])
@pytest.mark.parametrize("p", [0.5, 1.0, 2.0, 8.0])
def test_mfd_producer_contract(bs, p):
    z = _pitted_surface(bs)
    rec, w, lengths, nb, stack, basin = _route('mfd', z, bs, p)
    nn = NY * NX
    idx = np.arange(nn)
    interior = _interior(bs)
    assert rec.shape == (nn, 8) and w.shape == (nn, 8) and lengths.shape == (nn, 8)
    # border sentinel in slot 0, and ONLY at true borders
    border = ~interior
    assert np.all(rec[border, 0] == idx[border])
    assert np.all(nb[border] == 1)
    assert np.all(w[border, 0] == 1.0)
    assert np.all(lengths[border, 0] == 0.0)
    assert not np.any(rec[interior, 0] == idx[interior]), \
        "an interior cell carries the slot-0 border sentinel"
    assert np.all(nb[interior] >= 1)
    diag = np.hypot(DX, DY)
    for i in np.where(interior)[0]:
        k = nb[i]
        act = rec[i, :k]
        # unique, compact active prefix
        assert len(set(act.tolist())) == k, f"duplicate link at cell {i}"
        assert i not in act.tolist()
        # zeroed tail
        assert np.all(rec[i, k:] == 0)
        assert np.all(w[i, k:] == 0.0)
        assert np.all(lengths[i, k:] == 0.0)
        # weights partition unit flow
        assert abs(w[i, :k].sum() - 1.0) <= 1e-15
        assert np.all(w[i, :k] > 0.0)
        # Euclidean lengths, cardinal or diagonal on this square grid
        assert np.all(np.isclose(lengths[i, :k], DX)
                      | np.isclose(lengths[i, :k], diag))
    # every active edge is monotone in the donor-first stack
    pos = np.empty(nn, dtype=np.int64)
    pos[stack] = np.arange(nn)
    for i in np.where(interior)[0]:
        for k in range(nb[i]):
            assert pos[i] <= pos[rec[i, k]], "MFD stack is not donor-first"


@pytest.mark.parametrize("bs", [BS_FIXED, BS_LOOP], ids=["fixed", "looped"])
def test_mfd_links_strictly_descend_the_filled_surface(bs):
    """Every admitted link strictly descends the eps-filled surface — which is
    what makes the graph a DAG by construction (no tie-break, no flat handling).
    Checked against the SAME fill the producer routes."""
    from siim._core.step import _filled_surface_and_mask
    z = _pitted_surface(bs)
    zf, _interior_mask, _wy, _wx, _ny, _nx, _nn = _filled_surface_and_mask(
        z, (NY, NX), bs)
    rec, w, lengths, nb, stack, basin = _route('mfd', z, bs)
    interior = _interior(bs)
    for i in np.where(interior)[0]:
        for k in range(nb[i]):
            assert zf[rec[i, k]] < zf[i], f"link {i}->{rec[i, k]} does not descend"


def test_mfd_rejects_a_non_positive_or_nan_exponent():
    z = _pitted_surface(BS_FIXED)
    # gate: inf is now a valid exponent, the steepest link alone (Eric, 2026-11-12;
    # ED-180); its behaviour is pinned in the test below
    for bad in (0.0, -1.0, float('nan')):
        with pytest.raises(ValueError, match="must be positive"):
            _route('mfd', z, BS_FIXED, bad)


@pytest.mark.pin
def test_mfd_exponent_inf_is_the_steepest_link_and_above_the_bound_is_inf():
    """``mfd_exponent=inf`` is an exact branch: every interior cell keeps its
    steepest link alone (the receiver D8 picks on this tie-free surface) at
    weight 1, and equal steepest links share. A finite exponent above
    ``MFD_EXPONENT_MAX`` is routed as inf, with a warning naming it; the bound
    itself is a finite exponent still."""
    import warnings
    from siim import constants
    z = _pitted_surface(BS_FIXED)
    rec, w, lengths, nb, stack, _ = _route('mfd', z, BS_FIXED, float('inf'))
    d8_rec = np.asarray(_route('d8', z, BS_FIXED)[0]).ravel()
    interior = _interior(BS_FIXED)
    assert interior.any() and np.all(nb[interior] == 1)
    assert np.array_equal(rec[interior, 0], d8_rec[interior])
    assert np.all(w[:, 0] == 1.0) and not w[:, 1:].any()
    # a roof along x: each ridge cell's two steepest links, W and E, are equal
    roof = np.tile(-10.0 * np.abs(np.arange(NX) - NX // 2), (NY, 1))
    _, w_roof, _, nb_roof, _, _ = route_mfd(roof, (NY, NX), DX, DY, BS_LOOP, float('inf'))
    ridge = np.arange(NY) * NX + NX // 2
    assert np.all(nb_roof[ridge] == 2) and np.all(w_roof[ridge, :2] == 0.5)
    with pytest.warns(UserWarning, match=r"mfd_exponent=200 .* taken as mfd_exponent=inf"):
        above = _route('mfd', z, BS_FIXED, 200.0)
    for got, want in zip(above, (rec, w, lengths, nb, stack)):
        assert np.array_equal(got, want)
    with warnings.catch_warnings():
        warnings.simplefilter('error', UserWarning)
        at_bound = _route('mfd', z, BS_FIXED, constants.MFD_EXPONENT_MAX)
    assert np.all(at_bound[1] > 0.0, where=np.arange(8) < at_bound[3][:, None])
    assert at_bound[3][interior].max() > 1      # still a spread, with no zero weight


def test_mfd_rejects_a_periodic_axis_shorter_than_three_cells():
    """On a 2-cell periodic axis the -1 and +1 offsets alias onto the same node,
    which would put a duplicate link in the pack."""
    z = np.zeros((2, 5)) + np.arange(5)[None, :]
    with pytest.raises(ValueError, match="at least 3 cells"):
        route_mfd(z, (2, 5), DX, DY, BS_LOOP, 1.0)
