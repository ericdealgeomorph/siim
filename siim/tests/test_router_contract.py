"""The router conformance battery (``docs/dev/router_contract.md``).

Every conforming flow router runs through the HARD requirements. The
framework-free primitives (spill, flux conservation, stack validity, border
contract, seam invariance, SFR/D-inf cardinal parity) run on the in-house
producers directly (``route_d8`` / ``route_dinf``); the analytical
steady-state oracle at coarse tolerance runs a real siim2d model on the
shipped (in-house) backend. The retired fortran router passed this battery at
S4 (recorded in the plan); its arm died with the fortran flag at the S5 flip."""
import numpy as np
import pytest

from siim._core.step import route_d8, route_dinf
from siim._core.routing import _flow_accumulate_sd, _flow_accumulate_dinf

NY, NX = 15, 19
DX = DY = 100.0
BS_LOOP = ['fixed_value', 'fixed_value', 'looped', 'looped']
BS_FIXED = ['fixed_value'] * 4


def _route(kind, z, bs):
    fn = route_d8 if kind == 'd8' else route_dinf
    return fn(z, (NY, NX), DX, DY, bs)


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

@pytest.mark.parametrize("kind", ["d8", "dinf"])
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


@pytest.mark.parametrize("kind", ["d8", "dinf"])
@pytest.mark.parametrize("bs", [BS_FIXED, BS_LOOP], ids=["fixed", "looped"])
def test_flux_conservation(kind, bs):
    z = _pitted_surface(bs)
    rec, w, lengths, nb, stack, basin = _route(kind, z, bs)
    area = _area(kind, rec, w, nb, stack)
    # mass conservation: total drainage area gathered at outlets == n cells
    outlets = _self_receiving(kind, rec)
    assert area[outlets].sum() == pytest.approx(NY * NX)
    assert np.all(area >= 1.0 - 1e-9), "a cell gathered less than its own unit"


@pytest.mark.parametrize("kind", ["d8", "dinf"])
def test_border_contract(kind):
    z = _pitted_surface(BS_LOOP)
    rec, w, lengths, nb, stack, basin = _route(kind, z, BS_LOOP)
    border = ~_interior(BS_LOOP)
    idx = np.arange(NY * NX)
    assert np.all(_self_receiving(kind, rec)[border]), "a border cell is not self-receiving"
    if kind == 'd8':
        assert np.all(rec[border] == idx[border])
        assert np.all(lengths[border] == 0.0)


@pytest.mark.parametrize("kind", ["d8", "dinf"])
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
