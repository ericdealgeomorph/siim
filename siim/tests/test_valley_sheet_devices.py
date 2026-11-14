"""Mode D's valley <-> sheet devices (``constants.HC_SHEET``): the five valley
devices consume a per-cell confinedness s (1 = valley, 0 = sheet). Built as
gates 1-6 of the valley <-> sheet switch (decisions/2026-10-17-switch-build-
decisions.md, private), whose fullness source was retired on 2026-11-03; mode
D's channel/sheet reading is now the only source of s (its own tests: test_mode_d.py).

1. each device at unit level — the V blended per SOURCE (the Astra review's two
   configurations), the carve per source, the footprint, the filter's fade, the
   border rule and the datum's ends;
2. on mode-D runs: the per-cell datum contract (``hc_cell_out``), the datum's
   plumbing into the kernel and the router, the coexistence rule
   (fabricate+blend -> filter) through the FIREWALL spies, and mode D admitting
   the two router devices together (the valley limit with the V live is
   test_mode_d.py::test_valley_limit_is_mode_c).
"""
import numpy as np
import pytest

from siim import constants as _constants
from siim._core import driver as drv
from siim._core.router import InhouseRouter
from siim._core.step import (hc_cell_field, border_inherits,
                             accumulate_glacial_flow, carve_bed,
                             filter_routing_surface, _fabricate_trunk_surface)
from siim._core.driver import _edge_border_mask
from siim.siim2d import siim as siim2d
from siim.tests.test_mode_d import _D
from siim.tests.test_sheet_knobs import _spy_firewall, _run

HC = float(_constants.HC_OVER_H)

# The mixed mode-D fixture without uplift, diffusion or flexure, so the per-step
# bed change is the denudation alone.
_MIXED = dict(_D, U=0.0, border_bed_uplift=0.0, D=0.0, flexure=False)

# The coexistence fixture: mode D's valley fixture (channel and sheet cells
# side by side, the sheet thick enough for the filter to open windows,
# f*H*(1 - s)/dx > 1/2 at dx 250) with the trunk fixture's knobs (the V on, one
# undamped pass, D 1, flexure, sediment).
_COEXIST = dict(_D, trunk_surface=True, inner_passes=1, inner_relax=1.0, D=1.0, flexure=True,
                track_sediment=True)


# ---------------------------------------------------------------------------
# 1. the devices, unit level
# ---------------------------------------------------------------------------
def _fabricate(zb, H, s_conf, alpha_g=5.0, dx=500.0, k_dip=1.0):
    ny, nx = zb.shape
    border = _edge_border_mask(['fixed_value', 'fixed_value', 'looped', 'looped'], (ny, nx))
    scratch = (np.empty((ny, nx)), np.empty((ny, nx)), np.empty((ny, nx), dtype=np.int64))
    zs = zb + HC * H
    return zs, _fabricate_trunk_surface(zs, zb, H, border, alpha_g, dx, dx, k_dip, 1e-3,
                                        *scratch, True, False, s_conf=s_conf)


@pytest.mark.pin
def test_trunk_v_is_blended_per_source():
    """The Astra review's two configurations (2026-10-17). A straight trunk
    (H 500, s = 1) between icy flanks (H 200, s = 0) keeps its WHOLE V: the
    flanks stand above the trunk on the router surface as in the unswitched V
    (a target-s blend would drop them below it and send the flow off the
    trunk). A thick sheet source (s = 0) owning thin cells (s = 1) fabricates
    nothing: those cells keep the dynamic surface."""
    ny, nx, dx = 9, 21, 500.0
    zb = np.repeat((2000.0 - 0.05 * np.arange(nx) * dx)[None, :], ny, axis=0)
    H = np.full((ny, nx), 200.0); H[4] = 500.0
    s = np.zeros((ny, nx)); s[4] = 1.0
    zs, V = _fabricate(zb, H, None)
    _, blended = _fabricate(zb, H, s)
    _, ones = _fabricate(zb, H, np.ones((ny, nx)))
    np.testing.assert_array_equal(ones, V)
    assert (V != zs).any(), "the V was inert: vacuous"
    rows = slice(2, 7)                                 # the trunk's footprint (R = 2.5 cells)
    np.testing.assert_array_equal(blended[rows, 1:-1], V[rows, 1:-1])
    assert (blended[3, 1:-1] > blended[4, 1:-1]).all() and (blended[5, 1:-1] > blended[4, 1:-1]).all()
    # a sheet source: nothing fabricated on the thin cells it owns
    H = np.full((ny, nx), 30.0); H[4, 10] = 1000.0
    s = np.ones((ny, nx)); s[4, 10] = 0.0
    zs, V = _fabricate(zb, H, None)
    _, blended = _fabricate(zb, H, s)
    assert (V != zs).sum() > 20, "the sheet source's footprint was inert: vacuous"
    np.testing.assert_array_equal(blended, zs)
    # half a source: half the V
    s[4, 10] = 0.5
    _, half = _fabricate(zb, H, s)
    np.testing.assert_allclose(half, zs + 0.5 * (V - zs), rtol=0, atol=1e-9)


def test_border_inherits_the_dominant_donor():
    """The kernel's ``dom`` rule: the icy donor with the most ice, the lowest
    index on a tie; borders without icy donors keep their value."""
    ny, nx = 5, 6
    rec = (np.arange(ny * nx) - 1).reshape(ny, nx)           # flow -x; column 0 self
    rec[:, 0] = np.arange(ny) * nx
    rec[:, -1] = np.arange(ny) * nx + nx - 1                  # column nx-1 self, no donors
    s = np.full((ny, nx), 0.2); s[:, 0] = s[:, -1] = 1.0; s[2, 1] = 0.7
    Q = np.ones((ny, nx)); Q[2, 1] = 5.0
    rec[3, 1] = 2 * nx; Q[3, 1] = 9.0; s[3, 1] = 0.4           # row 2's border: a stronger second donor
    rec[4, 1] = 1 * nx; s[4, 1] = 0.9                          # row 1's border: a tie (Q 1 vs Q 1)
    Q[0, 1] = 0.0                                              # row 0's only donor is ice-free
    out = border_inherits(s, rec.ravel(), None, Q)
    assert out.shape == s.shape
    assert out[2, 0] == 0.4, "the larger flux wins"
    assert out[1, 0] == 0.2, "a tie goes to the lower index"
    assert out[0, 0] == 1.0 and out[3, 0] == 1.0 and out[4, 0] == 1.0, "no icy donor: value kept"
    assert (out[:, -1] == 1.0).all()
    np.testing.assert_array_equal(out[:, 1:-1], s[:, 1:-1])
    assert (border_inherits(np.ones((ny, nx)), rec.ravel(), None, Q) == 1.0).all()
    # a multi-receiver router: the border may be a SECONDARY receiver of its
    # dominant donor (the D-inf/MFD kernels' dom runs over every link)
    rec2 = np.stack([rec.ravel(), rec.ravel()], axis=1); nb = np.ones(ny * nx, dtype=np.int64)
    j = 2 * nx + 2                                             # (2, 2): primary (2, 1), secondary the border
    rec2[j, 1] = 2 * nx; nb[j] = 2; Q[2, 2] = 50.0; s[2, 2] = 0.05
    out2 = border_inherits(s, rec2, nb, Q)
    assert out2[2, 0] == 0.05, "the secondary link's donor wins on flux"
    np.testing.assert_array_equal(out2[[0, 1, 3, 4]], out[[0, 1, 3, 4]])


def _border_inherits_lexsort(s, receivers, nb_receivers, ice_flux):
    """The numpy ``border_inherits`` (a lexsort over every link) that the njit
    kernel replaced on 2026-11-12, verbatim: the oracle below."""
    shape = np.shape(s)
    s = np.asarray(s, dtype=np.float64).ravel().copy()
    rec = np.asarray(receivers, dtype=np.int64)
    Q = np.asarray(ice_flux, dtype=np.float64).ravel()
    idx = np.arange(s.size)
    if rec.ndim == 1:
        i_link, r_link = idx, rec
        is_border = rec == idx
    else:
        live = np.arange(rec.shape[1])[None, :] < np.asarray(nb_receivers, dtype=np.int64)[:, None]
        i_link = np.broadcast_to(idx[:, None], rec.shape)[live]
        r_link = rec[live]
        is_border = rec[:, 0] == idx
    sel = (r_link != i_link) & ~is_border[i_link] & is_border[r_link] & (Q[i_link] > 0.0)
    if sel.any():
        i_link, r_link = i_link[sel], r_link[sel]
        order = np.lexsort((-i_link, Q[i_link], r_link))
        r_sorted, i_sorted = r_link[order], i_link[order]
        last = np.r_[r_sorted[1:] != r_sorted[:-1], True]
        s[r_sorted[last]] = s[i_sorted[last]]
    return s.reshape(shape)


@pytest.mark.pin
@pytest.mark.parametrize('k', [1, 8])
def test_border_inherits_is_the_lexsort_rule(k):
    """The njit border rule equals the numpy lexsort it replaced, bit for bit,
    on random one-link (D8) and up-to-8-link (MFD) graphs whose fluxes tie,
    vanish and go negative, with border donors and self links mixed in."""
    rng = np.random.default_rng(11)
    tied = changed = 0
    for _ in range(60):
        n = int(rng.integers(5, 300))
        rec = rng.integers(0, n, size=(n, k))
        border = np.flatnonzero(rng.random(n) < 0.2)
        rec[border, 0] = border
        nb = rng.integers(1, k + 1, size=n)
        Q = rng.choice([-1.0, 0.0, 1.0, 2.0, 3.0], size=n)
        s = rng.random(n)
        r, b = (rec[:, 0], None) if k == 1 else (rec, nb)
        got = border_inherits(s, r, b, Q)
        np.testing.assert_array_equal(got, _border_inherits_lexsort(s, r, b, Q))
        changed += int((got != s).sum())
        for j in border:                     # borders whose top flux is shared
            donors = [i for i in range(n) if rec[i, 0] != i and Q[i] > 0.0
                      and j in rec[i, :nb[i] if k > 1 else 1]]
            qmax = max((Q[i] for i in donors), default=0.0)
            tied += sum(Q[i] == qmax for i in donors) > 1
    assert tied > 20 and changed > 20, (tied, changed)


def test_hc_cell_field_ends_exact():
    assert np.array_equal(hc_cell_field(np.array([1.0, 0.0]), HC, 1.0), [HC, 1.0])


def _carve_fixture(alpha_g=5.0):
    """A 1000 m source in a trough (bed 0, eroded 1 m this step) between 2000 m
    walls on a flat grid; interior cells route down-x, the border ring is
    self-receiving."""
    ny, nx, dx = 21, 21, 500.0
    nn = ny * nx
    rec = np.arange(nn) + 1
    rec[np.arange(nn) % nx == nx - 1] -= 1
    iy, ix = np.indices((ny, nx))
    border = ((iy == 0) | (iy == ny - 1) | (ix == 0) | (ix == nx - 1)).ravel()
    rec[border] = np.arange(nn)[border]
    s = 10 * nx + 10
    zb_pre = np.full(nn, 2000.0); zb_pre[s] = 0.0
    zb_post = zb_pre.copy(); zb_post[s] = -1.0
    H = np.zeros(nn); H[s] = 1000.0
    return (ny, nx), dx, rec, zb_pre, zb_post, H, s


def _carve(wf, s_conf=None, **kw):
    shape, dx, rec, zb_pre, zb_post, H, s = _carve_fixture(**kw)
    zb = zb_post.copy()
    surf = zb + HC * H
    carve_bed(zb, H, surf, zb_pre, rec, 5.0, HC, wf, shape, dx, dx, False, False,
              s_conf=None if s_conf is None else np.asarray(s_conf).reshape(shape))
    return zb, zb_post, s


@pytest.mark.pin
def test_carve_scales_by_the_source_confinedness():
    nn = 21 * 21
    for wf in (np.inf, 4.0):
        today, _, _ = _carve(wf)
        ones, _, _ = _carve(wf, np.ones(nn))
        np.testing.assert_array_equal(ones, today)               # s = 1: today exactly
        assert (today < 2000.0).sum() > 20, "the carve did not bite"
        none, post, s = _carve(wf, np.zeros(nn))
        np.testing.assert_array_equal(none, post)                # a sheet source carves nothing
    s_half = np.ones(nn); s_half[s] = 0.5
    half, post, _ = _carve(4.0, s_half)
    carved = half < 2000.0
    carved[s] = False                                        # the source's own bed
    assert carved.any()
    np.testing.assert_allclose(half[carved], 2000.0 - 0.5 * 4.0 * 1.0)   # (wf*s)*E_dt
    full, _, _ = _carve(4.0)
    np.testing.assert_allclose(full[carved], 2000.0 - 4.0 * 1.0)
    assert half.min() >= post[s], "a footprint cell ended below the source's true bed"


def _ramp_routing():
    ny, nx, dx = 15, 25, 500.0
    x = np.arange(nx) * dx
    zs = np.repeat((2000.0 * (1 - x / x[-1]))[None, :], ny, axis=0)
    zs += 20.0 * np.sin(np.arange(ny))[:, None]                # break the y ties
    bs = ['fixed_value', 'fixed_value', 'looped', 'looped']
    rt = InhouseRouter((ny, nx), bs, 'single', dx, dx)(zs)
    return zs, (ny, nx), dx, rt


@pytest.mark.pin
def test_footprint_fades_from_the_hack_width_to_the_cell():
    zs, shape, dx, rt = _ramp_routing()

    def acc(k, s_conf=None):
        return accumulate_glacial_flow(
            zs, 0.0, 1200.0, 1e-2, 2.0, dx * dx, k, 0.5, shape, rt.stack, rt.receivers,
            rt.nb_receivers, rt.weights, rt.lengths, rt.basin, s_conf=s_conf)[0]
    hack, cell = acc(0.5), acc(0.0)
    assert not np.array_equal(hack, cell), "the Hack width never exceeded the cell: vacuous"
    np.testing.assert_array_equal(acc(0.5, np.ones(shape)), hack)
    np.testing.assert_array_equal(acc(0.5, np.zeros(shape)), cell)
    mid = acc(0.5, np.full(shape, 0.5))
    assert not np.array_equal(mid, hack) and not np.array_equal(mid, cell)


@pytest.mark.pin
def test_filter_ignores_valley_cells():
    """A valley cell (s = 1) keeps its raw surface and weighs nothing in its sheet
    neighbours' windows: a spike there leaves every other filtered value untouched."""
    ny, nx, dx = 15, 15, 500.0
    rng = np.random.default_rng(1)
    zs = 1000.0 + rng.uniform(0, 10, (ny, nx))
    H = np.full((ny, nx), 900.0)                          # windows of ~3 cells
    s = np.zeros((ny, nx)); s[7, 7] = 1.0
    H_f = H * (1.0 - s)
    base = filter_routing_surface(zs, H_f, dx, 2.0)
    spiked = zs.copy(); spiked[7, 7] += 5000.0
    got = filter_routing_surface(spiked, H_f, dx, 2.0)
    assert got[7, 7] == spiked[7, 7]
    mask = np.ones((ny, nx), bool); mask[7, 7] = False
    np.testing.assert_array_equal(got[mask], base[mask])
    assert not np.array_equal(filter_routing_surface(spiked, H, dx, 2.0)[mask], base[mask]), \
        "without the fade the spike would have leaked: vacuous"


# ---------------------------------------------------------------------------
# 2. on mode-D runs
# ---------------------------------------------------------------------------
@pytest.mark.pin
def test_datum_contract():
    on = _run(_MIXED)
    hc = on.hc_cell_out
    assert on.ds_out['glacial_spl__hc_cell'].dims == ('time', 'y', 'x')
    assert hc.shape == on.H_out.shape
    assert (hc[0] == HC).all(), "the first step is the valley treatment"
    assert (1.0 <= hc).all() and (hc <= HC).all()
    assert (hc[-1] < HC).any() and (hc[-1] == HC).any(), "the fixture is not mixed"
    assert np.array_equal(on.z_out, on.zb_out + hc * on.H_out)
    # the TRUE-bed transfer: at U = D = 0 without flexure the per-step bed change
    # is the denudation alone, carve included (test_mode_c gate B)
    for k in range(on.zb_out.shape[0] - 1):
        np.testing.assert_allclose(on.zb_out[k + 1] - on.zb_out[k],
                                   -on.denudation_out[k], atol=1e-9)
    assert on.denudation_out.max() > 0.0
    # gate: the fixture is mode D since the fullness switch was retired (2026-11-03);
    # the same mixed bed in mode C and in mode A carries no per-cell datum
    assert _run(dict(_MIXED, mode='C')).hc_cell_out is None
    assert _run(dict(_MIXED, mode='A')).hc_cell_out is None


@pytest.mark.pin
def test_datum_reaches_the_kernel_and_the_router(monkeypatch):
    """The datum's plumbing, exact: the kernel receives the TRUE bed (the
    routing surface's post-uplift bed, no datum offset) and the per-cell datum
    ``hc_cell`` (the staged ``hc_cell_out`` frame); the routing surface is
    built with the same datum on the lagged thickness; and the datum is
    ``hc_cell_field`` of the s read at the end of the previous step, smoothed
    over the icy cells (``regime.smooth_icy``), then the border rule again."""
    kernel, surfaces, s_rec, sm_rec, rt = [], [], [], [], []
    real_k, real_s, real_sm = drv.run_modeB_kernel, drv.routing_surface, drv.smooth_icy
    real_bi = drv.border_inherits

    def bi_spy(x, *a):                  # the devices' call: keep its routing for the datum's rule
        if not (sm_rec and x is sm_rec[-1]):
            rt.append(tuple(None if v is None else np.array(v) for v in a))
        return real_bi(x, *a)

    def sm_spy(*a):                     # the smoothed s, the border rule applied here on that routing
        sm_rec.append(real_sm(*a))
        s_rec.append(real_bi(sm_rec[-1], *rt[-1]))
        return sm_rec[-1]

    def kernel_spy(zb_flat, H_flat, *a, **kw):
        kernel.append((zb_flat.copy(), H_flat.copy(), kw.get('hc_cell')))
        return real_k(zb_flat, H_flat, *a, **kw)
    monkeypatch.setattr(drv, 'run_modeB_kernel', kernel_spy)
    monkeypatch.setattr(drv, 'routing_surface',
                        lambda zb, hc, H: surfaces.append((zb.copy(), np.array(hc), H.copy()))
                        or real_s(zb, hc, H))
    monkeypatch.setattr(drv, 'smooth_icy', sm_spy)
    monkeypatch.setattr(drv, 'border_inherits', bi_spy)
    on = _run(_D)                                # nt_out = nt, one pass
    assert len(kernel) == len(surfaces) == len(s_rec) == on.H_out.shape[0] - 1
    for k, (zb0, H0, hc_in) in enumerate(kernel):
        hc_k = on.hc_cell_out[k]
        H_in = np.zeros_like(hc_k) if k == 0 else on.H_out[k - 1]
        zb_r, hc_r, H_r = surfaces[k]
        np.testing.assert_array_equal(zb0, zb_r.ravel())        # the true bed
        np.testing.assert_array_equal(H0, H_in.ravel())
        np.testing.assert_array_equal(hc_in, hc_k.ravel())
        np.testing.assert_array_equal(hc_r, hc_k)
        np.testing.assert_array_equal(H_r, H_in)                # the raw lag
        expect = np.full_like(hc_k, HC) if k == 0 else hc_cell_field(s_rec[k - 1], HC, 1.0)
        np.testing.assert_array_equal(hc_k, expect)
    assert (on.hc_cell_out[-1] == 1.0).any() and (on.hc_cell_out[0] == HC).all()


@pytest.mark.pin
def test_router_devices_coexist_fabricate_blend_filter(monkeypatch):
    """The coexistence rule, pinned through the FIREWALL spies on a mode-D
    fixture where the V's footprint is live, the filter opens windows and both
    s values occur: the router input is ``filter(V*s + dynamic*(1 - s),
    H*(1 - s))`` exactly (H the kernel's entering iterate); the accumulator, the diffusion, the transect, the
    kernel, the carve, the flexure load and the staged outputs are untouched by
    either device."""
    rec = _spy_firewall(monkeypatch)
    s_rec, filt, sm_rec = [], [], []
    real_bi, real_filter, real_sm = drv.border_inherits, drv.filter_routing_surface, drv.smooth_icy

    def bi_spy(x, *a):                  # the devices' s; the datum's call (on the smoothed s) is not one
        out = real_bi(x, *a)
        if not (sm_rec and x is sm_rec[-1]):
            s_rec.append(out)
        return out
    monkeypatch.setattr(drv, 'smooth_icy', lambda *a: sm_rec.append(real_sm(*a)) or sm_rec[-1])
    monkeypatch.setattr(drv, 'border_inherits', bi_spy)

    def filter_spy(zs, H, dx, f, *wrap):
        out = real_filter(zs, H, dx, f, *wrap)
        filt.append((zs.copy(), H.copy(), out))
        return out
    monkeypatch.setattr(drv, 'filter_routing_surface', filter_spy)
    m = _run(_COEXIST, mode='D', trunk_surface=True)   # the fixture's explicit (1, 1.0): one pass
    assert m.regime_switch and m.routing_filter_factor == 2.0
    steps, shape = m.H_out.shape[0] - 1, m.H_out[0].shape   # nt_out = nt
    assert len(rec['routed']) == len(rec['fab_out']) == len(filt) == steps
    assert len(s_rec) == steps                 # once per step, after the passes
    s_used = [np.ones(shape)] + s_rec[:-1]     # the passes use the previous step's s
    # gate: since 2026-11-13 s relaxes toward the class (constants.REGIME_SWAP_STEPS),
    # so the vacuity check asks for a sheet-leaning cell, not s == 0 exactly
    assert any((s < 0.5).any() for s in s_rec) and any((s == 1).any() for s in s_rec), \
        "s took one value: vacuous"
    live = sum(int((v != d).sum()) for v, d in zip(rec['fab_out'], rec['surface']))
    moved = sum(int((o != z).sum()) for z, _, o in filt)
    assert live > 0 and moved > 0, (live, moved)
    for k in range(steps):
        s, H = s_used[k], rec['kernel_H'][k].reshape(shape)
        dyn, V = rec['surface'][k], rec['fab_out'][k]     # V: blended per source inside
        zs_in, H_in, out = filt[k]
        np.testing.assert_array_equal(rec['fab_in'][k], dyn)
        np.testing.assert_array_equal(zs_in, V)
        # gate: routing_relax (H_eff != raw H) was removed 2026-11-04: the
        # filter thickness is the raw entering iterate
        np.testing.assert_array_equal(H_in, H * (1.0 - s))
        np.testing.assert_array_equal(rec['routed'][k], out)
        np.testing.assert_array_equal(rec['accumulated'][k], dyn)
        np.testing.assert_array_equal(rec['diffused'][k], dyn)
        entering = np.zeros(shape) if k == 0 else m.H_out[k - 1]
        np.testing.assert_array_equal(rec['kernel_H'][k].reshape(shape), entering)
        np.testing.assert_array_equal(rec['transect_H'][k].ravel(), rec['kernel_H'][k])
        np.testing.assert_array_equal(rec['carve_H'][k].reshape(shape), m.H_out[k])
        np.testing.assert_array_equal(rec['flexure_H'][k], m.H_out[k])
        np.testing.assert_array_equal(rec['staged_H'][k], m.H_out[k])
        zb_d, H_d = rec['diag'][k]                        # clause 3: the committed state
        np.testing.assert_array_equal(H_d, m.H_out[k])
        np.testing.assert_array_equal(zb_d.ravel(), rec['eroded_zb'][k])   # before the carve
    assert len(rec['diag']) == steps
    assert (m.hc_cell_out[-1] < HC).any()


def test_mode_d_admits_filter_with_trunk():
    m = siim2d(dict(_D, trunk_surface=True))
    assert m.trunk_surface and m.routing_filter_factor == 2.0 and m.regime_switch
