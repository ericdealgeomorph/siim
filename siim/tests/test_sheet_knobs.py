"""The mode-B/C sheet knobs: ``inner_passes`` (n), ``inner_relax`` (omega) and
``routing_filter_factor`` (f).

A landscape step evaluates the diagnostic ice map F(H) once; ``inner_passes``
evaluates it n times, moving the thickness iterate ``x <- x + omega*(F(x) - x)``
after every pass, the first n-1 passes rehearsing at ``dt = 0``.
``routing_filter_factor`` is the separate SPATIAL
device: at f > 0 the ROUTER GRAPH alone is built from a thickness-weighted
triangular mean of the routing surface ``zb + hc*H`` over a window of half-width
``max(f*H/dx - 1/2, 0)`` cells, while the mass-balance surface and the hillslope
diffusion keep the unfiltered surface.

Gates here:

* **defaults** — explicit ``(1, 1.0)`` equals the unparameterised model on every
  output, plus the independent pre-change anchor (the 8e0315f fingerprints,
  skipped when that private baseline is absent);
* **identities** — equal-work groupings of the damped move agree exactly;
* **parity** — the production loop reproduces the sheet-harness recipe
  (`docs/dev/ice_sheet_dome_probes/sheet_recipe_study.py::recipe_update`)
  recurrence, transcribed here against a portable oracle built from the same
  ``_core`` calls (public tests are exported; ``docs/dev/`` is not);
* **isolation** — rehearsal passes move no bed and every physical commit (carve,
  hillslope diffusion, flexure, sediment, staging) runs once per step;
* **consequences** — the two the CHANGELOG documents for ``omega < 1``: the
  mode-C carve reads the committed blend, and the state retains thickness where
  the pass's diagnostic ice flux is zero while the kernel erodes it fluvially;
* **the filter** — it reproduces the settled harness formula
  (`docs/dev/sheet_smoother_choice.md`: surface target, triangular kernel, no
  window floor) against the definition transcribed here, preserves dry and
  sub-threshold cells exactly, commutes with a seam shift on looped axes, and
  at f = 2 reaches only the router (the FIREWALL pin); the fabricated trunk
  surface is pinned the same way;
* **guards** — mode A and the xsimlab adapter refuse non-defaults, f > 0 is
  refused on a non-square grid and with ``trunk_surface``, and the
  validation rejects bools, fractions, out-of-range omega and non-finite values.
"""
import hashlib
import json
import math
import os
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from siim import constants as _constants
from siim._core.router import InhouseRouter
from siim._core.step import (accumulate_glacial_flow, filter_routing_surface,
                             routing_surface, run_modeB_kernel, transect_flux,
                             _triangular_axis, _triangular_level)
from siim.siim2d import siim as siim2d


# ---------------------------------------------------------------------------
# Fixtures. ``_SHEET`` is the mode-B MFD sheet the knobs target; ``_VALLEY`` the
# small mode-C D8 valley (flexure on, looped y) that carries the carve.
# ---------------------------------------------------------------------------
_PHYSICS = dict(U=1e-3, zELA=150, beta=1e-2, P=2, alpha_g=12, Ko=2e-6, n=1,
                ce=1e-4, nu=2, sliding_law='power', lambda_p=500, k=.9,
                seed=7, initial_max_elevation=800, progress_bar=False)

_SHEET = dict(_PHYSICS, T=6e3, nt=7, nt_out=4, Lx=1e4, Ly=1e4, nx=21, ny=21,
              boundary_status=['fixed_value'] * 4, mode='B', flow_routing='mfd',
              trunk_surface=False, parallel_erode=False)

_VALLEY = dict(_PHYSICS, T=6e3, nt=7, nt_out=4, Lx=1.2e4, Ly=1.2e4, nx=25, ny=25,
               boundary_status=['fixed_value', 'fixed_value', 'looped', 'looped'],
               mode='C', flexure=True, track_sediment=True)

# The frozen-bed parity fixture: nothing but H evolves, so dt is inert and the
# recurrence is the whole of the model. ``ce``/``Ko`` are constructed at 1e-30
# and zeroed on the instance — siim2d always builds its 1-D analytical reference
# in __init__, which divides by K and by C (the documented workaround of
# docs/dev/ice_sheet_dome_probes/dome_utils.py; at 1e-30 the bed still creeps
# ~1e-3 m/step, at exactly 0 it is bit-rigid).
_FROZEN = dict(_SHEET, U=0.0, Ko=1e-30, ce=1e-30, D=0.0, bl=-1e6,
               flotation_gate=False, border_bed_uplift=0.0, width_hack_k=0.0,
               H_diffusivity=None)             # callers set their own T/nt/nt_out

# The two non-default settings every guard test tries.
_NON_DEFAULT = [dict(inner_passes=2), dict(inner_relax=0.5)]

_OUT_ARRAYS = ('H_out', 'z_out', 'zb_out', 'Qg_out', 'Qf_out', 'area_out',
               'denudation_out', 'erosion_rate_out', 'receivers_out',
               'lengths_out', 'stack_out', 't_out')

# The coverage a present pre-change fingerprint manifest must carry (the
# captured fixture, with all 12 of ``_OUT_ARRAYS``).
# gate: the mode-C D8 valley capture ran routing_relax 0.5, removed 2026-11-04
_FINGERPRINT_CASES = ('modeB_mfd_sheet',)


def _run(base, **kw):
    m = siim2d(dict(base, **kw))
    m.run()
    return m


def _frozen(**kw):
    """A frozen-bed model (built, not run): U = Ko = ce = D = bbu = 0."""
    m = siim2d(dict(_FROZEN, **kw))
    m.ce = m.Ko = 0.0
    return m


def _run_frozen(**kw):
    m = _frozen(**kw)
    m.run()
    return m


def _assert_same_outputs(a, b, names=_OUT_ARRAYS):
    for name in names:
        u, v = getattr(a, name, None), getattr(b, name, None)
        if u is None and v is None:
            continue
        assert np.array_equal(u, v), f"{name} differs"


def _spy_kernel(monkeypatch, record):
    """Patch ``driver.run_modeB_kernel``; after every call
    ``record(dt, zb_before, zb_after, H_before, H_after)`` — the *_after arrays
    are the live kernel buffers, copy them to keep them. Calls the real kernel
    imported above, so spies do not nest when a test patches twice."""
    from siim._core import driver as drv

    def spy(zb_flat, H_flat, Q, water_flux, law_code, gp, dt, *a, **kw):
        zb0, H0 = zb_flat.copy(), H_flat.copy()
        out = run_modeB_kernel(zb_flat, H_flat, Q, water_flux, law_code, gp, dt,
                               *a, **kw)
        record(dt, zb0, zb_flat, H0, H_flat)
        return out
    monkeypatch.setattr(drv, 'run_modeB_kernel', spy)


def _assert_same_dataset(a, b, extra=()):
    """The COMPLETE packed contract: every ``ds_out`` variable. ``_OUT_ARRAYS``
    is the public ``*_out`` attribute list, which omits the dataset-only
    ``uplift__uplift`` and the basin labels. ``glacial_flow__basin_ids`` is the
    one array excluded from the comparison — basin labelling is nondeterministic
    (reference-battery rule 5). ``extra`` names rows ``a`` carries beyond ``b``."""
    keys = sorted(b.ds_out.data_vars)
    assert sorted(a.ds_out.data_vars) == sorted(keys + list(extra)), keys
    assert {'uplift__uplift', 'glacial_flow__basin_ids'} <= set(keys), keys
    for k in keys:
        if k == 'glacial_flow__basin_ids':
            continue
        assert np.array_equal(a.ds_out[k].values, b.ds_out[k].values), f"{k} differs"


# ---------------------------------------------------------------------------
# The portable oracle: ONE production pass, composed from the same ``_core``
# calls the driver makes, driven from a model's resolved driver config. The
# recurrence around it is transcribed from ``recipe_update``; nothing is
# imported from ``docs/dev/`` (that tree is excluded from the public export).
# ---------------------------------------------------------------------------
class _Oracle:
    """``F(x, dt)`` — one surf2erode -> route -> accumulate -> kernel
    evaluation on a FROZEN bed, returning the kernel's closure thickness."""

    def __init__(self, model):
        self.cfg = cfg = model._build_driver_config()
        self.shape = (cfg.ny, cfg.nx)
        self.topo = np.asarray(cfg.initial_surface,
                               dtype=np.float64).reshape(self.shape).copy()
        self.dt = float(cfg.t[1] - cfg.t[0])
        # Scalar forcing only — the driver's per-step slicing is then a no-op,
        # so the oracle does not have to reproduce it.
        assert cfg.zELA_series is None and cfg.runoff_series is None
        assert cfg.bl_series is None and cfg.bl_sides is None
        assert cfg.bbu_series is None
        assert cfg.closure_flux == 'transect_min'   # F() hardcodes min_rule=True
        # The harness filters the ROUTER surface alone; production's filter is
        # reused here (its formula is gated independently, against the
        # transcribed definition, by test_filter_matches_the_definition).
        self.filter_factor = float(cfg.routing_filter_factor)
        self.filtered_cells = 0
        self.uplift = cfg.uplift_fn(0, self.dt)
        assert not self.uplift.any(), "the parity fixture must not uplift"
        self._router = InhouseRouter([cfg.ny, cfg.nx], model.boundary_status,
                                     model.flow_routing, cfg.dx, cfg.dy,
                                     model.mfd_exponent)

    def __enter__(self):
        self.route = self._router.__enter__()
        return self

    def __exit__(self, *exc):
        return self._router.__exit__(*exc)

    def F(self, x, dt_pass):
        cfg, shape = self.cfg, self.shape
        zs = routing_surface(self.topo, cfg.hc_over_H, x)
        zs_router = zs
        if self.filter_factor:
            zs_router = filter_routing_surface(zs, x, cfg.dx,
                                               self.filter_factor)
            self.filtered_cells += int((zs_router != zs).sum())
        rt = self.route(zs_router)
        ice_flux, water_flux = accumulate_glacial_flow(
            zs, self.uplift, cfg.zELA, cfg.beta, cfg.P, cfg.cell_area,
            cfg.width_hack_k, cfg.width_hack_p, shape, rt.stack, rt.receivers,
            rt.nb_receivers, rt.weights, rt.lengths, rt.basin)[:2]
        zb_flat = self.topo.flatten()
        H_flat = x.flatten()
        Q = transect_flux(ice_flux, x, rt.receivers, rt.weights,
                          rt.nb_receivers, shape, cfg.dx, cfg.dy, cfg.alpha_g,
                          cfg.wrap_y, cfg.wrap_x, min_rule=True)
        run_modeB_kernel(
            zb_flat, H_flat, Q, water_flux, cfg.law_code, cfg.gp, dt_pass,
            rt.stack, rt.receivers, rt.nb_receivers, rt.weights, rt.lengths,
            shape, cfg.dx, cfg.dy, cfg.bbu_static, cfg.bl, cfg.flotation_gate,
            cfg.flotation_ramp, cfg.parallel_erode, cfg.wrap_y, cfg.wrap_x,
            cfg.slope_normalise)
        return H_flat.reshape(shape)


# ---------------------------------------------------------------------------
# (a) Defaults: the explicit defaults are the unparameterised model, and the
#     unparameterised model is still the PRE-CHANGE model.
# ---------------------------------------------------------------------------
_BC = dict(inner_damping=_constants.MODE_BCD_INNER_DAMPING, inner_passes=_constants.INNER_PASSES)


@pytest.mark.pin
@pytest.mark.parametrize('base', [_SHEET, _VALLEY], ids=['sheet', 'valley-modeC-standard'])
def test_explicit_defaults_are_the_unparameterised_model(base):
    """The resolved (inner_passes, inner_relax) spelled out == not spelling it,
    on every output."""
    plain = _run(base)
    explicit = _run(base, **_BC)
    _assert_same_outputs(plain, explicit, _OUT_ARRAYS + (
        'sediment_flux_out', 'eroded_volume_out', 'rebound_out'))  # absent on the sheet: skipped


def _sha(a):
    a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def _fingerprint_path():
    env = os.environ.get('SIIM_FINGERPRINTS_JSON')
    if env:
        return Path(env)
    root = Path(__file__).resolve().parents[2]
    return root / '.scratch' / 'reference-8e0315f' / 'fingerprints.json'


@pytest.mark.pin
def test_prechange_fingerprints():
    """The independent pre-change anchor: the shipped-default mode-B MFD sheet
    captured before this feature existed still hashes bit-for-bit (the
    manifest's mode-C D8 valley ran ``routing_relax``, since removed).
    Default-vs-explicit-default alone proves parameter equivalence, not
    preservation of the algorithm — both arms can share a new bug.

    Private baseline (``.scratch/`` is untracked and unexported): skipped when
    absent, pointed at by ``SIIM_FINGERPRINTS_JSON`` otherwise. A manifest that
    IS present must carry the full coverage — iterating whatever it happens to
    hold lets an empty one report success."""
    # gate: same expectation (every fingerprint still hashes), now driven from
    # the REQUIRED case/array names instead of the manifest's own keys — astra:
    # `{"cases": {}}` and an empty `arrays` mapping both passed silently.
    path = _fingerprint_path()
    if not path.is_file():
        pytest.skip(f"no pre-change fingerprint baseline at {path}")
    ref = json.loads(path.read_text())
    missing = set(_FINGERPRINT_CASES) - set(ref['cases'])
    assert not missing, f"{path}: fingerprint cases missing: {sorted(missing)}"
    for name in _FINGERPRINT_CASES:
        case = ref['cases'][name]
        missing = set(_OUT_ARRAYS) - set(case['arrays'])
        assert not missing, f"{path}: {name} fingerprints missing: {sorted(missing)}"
        kwargs = {k: (np.asarray(v['__ndarray__'])
                      if isinstance(v, dict) and '__ndarray__' in v else v)
                  for k, v in case['kwargs'].items()}
        # the captures ran one fresh pass (B/C damp by default since 2026-11-04);
        # (1, 1.0) restores the mode-B model of its day
        kwargs = {'inner_passes': _constants.INNER_PASSES, 'inner_relax': _constants.INNER_RELAX, **kwargs}
        m = siim2d(kwargs)
        m.run()
        for array in _OUT_ARRAYS:
            got = getattr(m, array)
            assert _sha(got) == case['arrays'][array]['sha256'], (
                f"{name}.{array} changed against the pre-change baseline")


# ---------------------------------------------------------------------------
# (b0) Identities.
# ---------------------------------------------------------------------------
@pytest.mark.pin
def test_equal_work_grouping_identity():
    """The damped move is the same recurrence however the passes are grouped:
    on a frozen bed (dt inert, no thickness diffusion) eight passes in one step
    is four steps of two passes, EXACTLY."""
    one = _run_frozen(T=1e3, nt=2, nt_out=2, inner_passes=8, inner_relax=0.1)
    four = _run_frozen(T=4e3, nt=5, nt_out=2, inner_passes=2, inner_relax=0.1)
    assert one.H_out[-1].max() > 1.0
    assert np.array_equal(one.H_out[-1], four.H_out[-1])
    assert np.array_equal(one.zb_out[0], one.zb_out[-1]), "the bed did not stay frozen"


# ---------------------------------------------------------------------------
# (b) Harness parity at f = 0.
# ---------------------------------------------------------------------------
@pytest.mark.pin
def test_harness_recipe_parity_two_passes(monkeypatch):
    """Production at ``(inner_passes=2, inner_relax=0.2)`` reproduces the sheet
    harness's ``Recipe(passes=2, omega=.2, endpoint='mixed')`` update
    (``sheet_recipe_study.recipe_update``), transcribed here onto a portable
    oracle: H from zero, frozen bed (U = Ko = ce = bbu = 0, hillslope D = 0,
    flexure off, carve/trunk off), thickness diffusion off, flotation gate off,
    ``bl = -1e6``, ``hc = 1`` and a spatially CONSTANT fixed ELA.

    The one term that blocks parity with the harness's DEFAULT
    prescribed-balance recipe is per-pass ELA re-pinning: the harness recomputes
    zELA each pass so a(r) stays pinned (no elevation feedback), which production
    cannot do. Its ``climate_beta`` branch — a fixed ELA — is what this fixture
    matches, and the public constructor takes only a scalar or a time series, so
    the fixed ELA here is spatially constant."""
    monkeypatch.setattr(_constants, 'HC_OVER_H', 1.0)
    steps = dict(T=3e3, nt=4, nt_out=2)
    model = _run_frozen(inner_passes=2, inner_relax=0.2, **steps)
    with _Oracle(_frozen(**steps)) as o:
        x = np.zeros(o.shape)
        for _ in range(o.cfg.nt - 1):
            for p in range(2):
                final = p == 1
                mapped = o.F(x, o.dt if final else 0.0)
                x = x + 0.2 * (mapped - x)
    assert x.max() > 1.0, "the parity fixture grew no ice"
    assert np.abs(x - model.H_out[-1]).max() <= 1e-9   # the plan's gate; bitwise next
    assert np.array_equal(x, model.H_out[-1]), "expected bitwise, same kernels"


@pytest.mark.pin
def test_adaptive_preset_matches_the_transcribed_rule(monkeypatch):
    """The 'adaptive_capped' commit, transcribed onto the portable oracle: a
    per-cell weight w (start 1) halves where the move reverses sign and grows
    by 1.25 elsewhere, clipped to [1/3, 1]; x <- x + w*(F - x) with the
    bare-cell snap; the previous move carries across passes AND steps (two
    passes a step, so both seams are crossed). Literals, not the constants, so
    a changed constant fails here."""
    monkeypatch.setattr(_constants, 'HC_OVER_H', 1.0)
    steps = dict(T=6e3, nt=7, nt_out=2)
    model = _run_frozen(inner_damping='adaptive_capped', inner_passes=2, **steps)
    with _Oracle(_frozen(**steps)) as o:
        x, w, prev = np.zeros(o.shape), np.ones(o.shape), np.zeros(o.shape)
        reversals = 0
        for _ in range(o.cfg.nt - 1):
            for p in range(2):
                F = o.F(x, o.dt if p == 1 else 0.0)
                d = F - x
                rev = d * prev < 0.0
                reversals += int(rev.sum())
                w = np.clip(np.where(rev, 0.5 * w, 1.25 * w), 1.0 / 3.0, 1.0)
                prev = d
                x = x + w * d
                x = np.where((F <= 0.0) & (x < 1.0), 0.0, x)
    assert x.max() > 1.0, "the fixture grew no ice"
    assert reversals > 0, "no cell reversed: the halving branch went untested"
    assert np.array_equal(x, model.H_out[-1])


# ---------------------------------------------------------------------------
# Rehearsal isolation.
# ---------------------------------------------------------------------------
@pytest.mark.pin
def test_rehearsal_passes_run_at_dt_zero_and_commit_once(monkeypatch):
    """Intermediate passes rehearse: the kernel is called once per pass, at
    ``dt = 0`` for all but the last, and moves NO bed there. Every physical
    commit — carve, hillslope diffusion, flexure, sediment — runs once per step,
    outside the loop, and output staging still runs once per output frame."""
    from siim._core import driver as drv
    seen, counts = [], {}

    def _count(name, fn):
        def wrapper(*a, **kw):
            counts[name] = counts.get(name, 0) + 1
            return fn(*a, **kw)
        return wrapper

    def record(dt_pass, zb_before, zb_after, _H0, _H1):
        if dt_pass == 0.0:
            assert np.array_equal(zb_after, zb_before), "a rehearsal pass moved the bed"
        seen.append(dt_pass)

    _spy_kernel(monkeypatch, record)
    for name in ('carve_bed', 'glacial_flexure_step', 'accumulate_sediment',
                 '_inhouse_diffuse', '_snapshot'):
        monkeypatch.setattr(drv, name, _count(name, getattr(drv, name)))

    nt, nt_out, passes = 4, 2, 3
    m = _run(_VALLEY, T=3e3, nt=nt, nt_out=nt_out,
             inner_passes=passes, inner_relax=1.0)
    assert m.H_out[-1].max() > 0.0
    assert seen == [0.0, 0.0, 1000.0] * (nt - 1)
    for name in ('carve_bed', 'glacial_flexure_step', 'accumulate_sediment',
                 '_inhouse_diffuse'):
        assert counts[name] == nt - 1, f"{name} ran {counts[name]}x, not once a step"
    # staging is per OUTPUT frame, so nt_out — never multiplied by the passes
    assert counts['_snapshot'] == nt_out == m.H_out.shape[0]


@pytest.mark.pin
def test_rehearsal_moves_only_the_thickness():
    """On a frozen bed with the ice load off, an extra rehearsal pass changes
    the thickness iterate and NOTHING else: bed, denudation, erosion rate,
    sediment and the flexure carry are bit-identical to the one-pass run. (The
    routed diagnostics and the reconstructed ice surface ``z_out = zb + hc*H``
    do move — they read the final pass's iterate.)"""
    kw = dict(T=1e3, nt=2, nt_out=2, flexure=True, ice_load=False,
              track_sediment=True)
    one = _run_frozen(inner_passes=1, inner_relax=1.0, **kw)
    two = _run_frozen(inner_passes=2, inner_relax=1.0, **kw)
    _assert_same_outputs(one, two, ('zb_out', 'denudation_out',
                                    'erosion_rate_out', 'sediment_flux_out',
                                    'eroded_volume_out', 'rebound_out'))
    assert not np.array_equal(one.H_out[-1], two.H_out[-1])


# ---------------------------------------------------------------------------
# (d) Smoke, and the mode-C carve on the committed state.
# ---------------------------------------------------------------------------
def test_damped_sheet_smoke():
    """61^2 mode-B MFD sheet at (2, 0.2): finite, non-negative, and within a
    generous band of the one-pass run's ice volume."""
    cfg = dict(_SHEET, nx=61, ny=61, Lx=3e4, Ly=3e4, T=1e4, nt=11, nt_out=3)
    one = _run(cfg, inner_passes=1, inner_relax=1.0)
    damped = _run(cfg, inner_passes=2, inner_relax=0.2)
    H = damped.H_out
    assert np.isfinite(H).all()
    assert (H >= 0).all()
    assert H[-1].max() > 0.0
    v_one, v_damped = one.H_out[-1].sum(), H[-1].sum()
    assert 0.25 * v_one <= v_damped <= 4.0 * v_one, (v_one, v_damped)


@pytest.mark.pin
def test_mode_c_carves_on_the_committed_blend(monkeypatch):
    """Mode C carves ONCE, on the committed blend — not on the final pass's own
    closure H, which under omega < 1 is not what the state carries."""
    from siim._core import driver as drv
    carve_H, closure_H = [], []
    real_carve = drv.carve_bed

    def carve_spy(zb_flat, H_flat, *a, **kw):
        carve_H.append(np.asarray(H_flat).copy())
        return real_carve(zb_flat, H_flat, *a, **kw)

    monkeypatch.setattr(drv, 'carve_bed', carve_spy)
    _spy_kernel(monkeypatch, lambda dt, zb0, zb1, H0, H1: closure_H.append(H1.copy()))
    m = _run(_VALLEY, T=3e3, nt=4, nt_out=2, flexure=False,
             track_sediment=False,
             inner_passes=2, inner_relax=0.2)
    assert len(carve_H) == 3 and len(closure_H) == 6      # once / twice a step
    assert np.array_equal(carve_H[-1].reshape(m.H_out[-1].shape), m.H_out[-1])
    assert not np.array_equal(carve_H[-1], closure_H[-1]), (
        "the carve saw the closure H, not the blend")


@pytest.mark.pin
def test_damped_state_retains_thickness_at_zero_flux(monkeypatch):
    """The other documented consequence, pinned: under damping the committed
    state HOLDS thickness where that pass's diagnostic ice flux is zero, while
    the kernel — which reads its own closure H, not the blend — erodes the cell
    fluvially. The CHANGELOG's arithmetic, ``x = 100`` and ``F = 0`` committing
    80 m at ``omega = 0.2``; here the margin cell carries ~55 m and commits
    ~44 m, and its denudation is the ice-free one, ~100x smaller than the same
    cell's in the twin whose ELA stays put and whose glacial law does act.

    Fixture: grow the mode-B MFD sheet for seven steps,
    then raise the ELA far above the surface so the whole grid ablates and the
    closure returns H = 0 everywhere. The twin differs ONLY in that last step's
    ELA, so both enter it from a bit-identical state."""
    seen = {}                        # tag -> (H entering, closure H), last step

    def run(tag, last_ela):
        _spy_kernel(monkeypatch,
                    lambda dt, zb0, zb1, H0, H1: seen.__setitem__(tag, (H0, H1.copy())))
        nt = 9
        return _run(_SHEET, T=8e3, nt=nt, nt_out=nt,
                    zELA=[100.0] * (nt - 2) + [last_ela] * 2,
                    inner_passes=1, inner_relax=0.2)

    dry = run('dry', 1e5)            # ELA far above the surface: nothing survives
    icy = run('icy', 100.0)          # the twin: ice, and the glacial branch
    before, closure = seen['dry']
    assert np.array_equal(before, seen['icy'][0]), "the twins diverged early"
    cell = int(np.argmax(before))
    assert before[cell] > 10.0, before[cell]          # real ice to retain
    assert closure[cell] == 0.0, "the closure returned thickness, not zero flux"
    assert seen['icy'][1][cell] > 0.0, "the twin's glacial branch did not run"
    # committed = x + omega*(F - x) with F = 0, to the last bit
    assert (dry.H_out[-1].ravel()[cell]
            == before[cell] + 0.2 * (0.0 - before[cell]))
    # ...and the erosion there is the fluvial branch's, not the retained ice's
    d_dry = dry.denudation_out[-1].ravel()[cell]
    d_icy = icy.denudation_out[-1].ravel()[cell]
    assert 0.0 < d_dry < 0.1 * d_icy, (d_dry, d_icy)


# ---------------------------------------------------------------------------
# (e) Guards.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('kw', [
    dict(inner_passes=0),
    dict(inner_passes=-1),
    dict(inner_passes=1.5),
    dict(inner_passes=True),
    dict(inner_passes=np.bool_(True)),
    dict(inner_passes=np.nan),
    dict(inner_passes=np.inf),
    dict(inner_relax=0.0),
    dict(inner_relax=-0.1),
    dict(inner_relax=1.5),
    dict(inner_relax=np.nan),
    dict(inner_relax=np.inf),
    dict(inner_relax=True),
])
def test_validation_rejects(kw):
    with pytest.raises(ValueError, match='inner_'):
        siim2d(dict(_SHEET, **kw))


# No runaway on the harsh axis (dt = 300 kyr, eta = 5), both surf2erode
# providers: the default damping (re-pointed from the removed routing_relax's
# gate) must not cut the bed far below, or inflate H far above, its undamped twin.
_HNX, _HDX = 81, 250.0
_HBASE = dict(U=1e-3, P=1, beta=.5e-2, Ko=1e-6, n=1, ce=1e-5, nu=2,
              tau_c=1.2e5, lambda_c=1000, lambda_p=300.0, alpha_g=10,
              Lx=(_HNX - 1) * _HDX, Ly=(_HNX - 1) * _HDX, nx=_HNX, ny=_HNX,
              boundary_status=['fixed_value'] * 4, seed=109,
              flow_routing='single', progress_bar=False)


@pytest.fixture(scope='module')
def _warm_harsh():
    m = siim2d(dict(_HBASE, zELA=1e6, sliding_law='power', mode='B', carve_width=False,
                    initial_max_elevation=1000, T=30e6, nt=101, nt_out=2,
                    inner_passes=1, inner_relax=1.0))
    m.run()
    return m.z_out[-1]


@pytest.mark.parametrize("trunk_surface", [False, True])
def test_default_damping_no_runaway(trunk_surface, _warm_harsh):
    z0 = _warm_harsh

    def run(**damping):
        m = siim2d(dict(_HBASE, sliding_law='coulomb', initial_topography=z0,
                        zELA=float(np.quantile(z0, 0.50)), mode='C', widening_rate=5,
                        trunk_surface=trunk_surface, T=9e6, nt=31, nt_out=31, **damping))
        m.run()
        return m

    on, off = run(), run(inner_passes=1, inner_relax=1.0)
    # gate: the default damping became the adaptive_capped preset (Eric, 2026-11-04), then adaptive (2026-11-05)
    assert on.inner_damping == _constants.MODE_BCD_INNER_DAMPING
    assert on.zb_out.min() >= off.zb_out.min() - 1.0, \
        f"damped bed {on.zb_out.min():.0f} << undamped {off.zb_out.min():.0f}"
    assert on.H_out.max() <= 5.0 * off.H_out.max()


def test_routing_relax_is_removed():
    """The routing-surface relaxation is gone (modes B/C damp the ice itself);
    passing it names the replacement."""
    with pytest.raises(ValueError, match='routing_relax was removed'):
        siim2d(dict(_SHEET, routing_relax=0.5))


@pytest.mark.parametrize('kw', _NON_DEFAULT)
def test_mode_a_refuses_non_defaults(kw):
    with pytest.raises(ValueError, match="mode 'B' or 'C'"):
        siim2d(dict(_SHEET, mode='A', flow_routing='single', **kw))


def test_diffusion_warning():
    """Damping attenuates the diffusion correction: H diffusion runs on the
    final pass only, and the commit blends the diffused closure with the
    undiffused iterate."""
    with pytest.warns(UserWarning, match='attenuates the diffusion correction'):
        siim2d(dict(_SHEET, inner_relax=0.5, H_diffusivity=1e3))
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        siim2d(dict(_SHEET, inner_passes=4, inner_relax=1.0, H_diffusivity=1e3))


@pytest.mark.adapter
@pytest.mark.parametrize('kw', _NON_DEFAULT)
def test_adapter_refuses_non_defaults_directly(kw):
    """Entry point 1: a direct ``glacial_processes(...)`` call."""
    pytest.importorskip('xsimlab')
    pytest.importorskip('fastscape')
    from siim.fastscape import glacial_processes
    glacial_processes(mode='B')                       # defaults are fine
    with pytest.raises(ValueError, match='STANDALONE-only'):
        glacial_processes(mode='B', **kw)


@pytest.mark.adapter
@pytest.mark.parametrize('kw', _NON_DEFAULT + [{}])   # {}: the adaptive default
def test_adapter_refuses_non_defaults_through_the_model(kw):
    """Entry point 2: a configured ``siim2d(...).run(driver='xsimlab')``, whose
    ``_process_overrides`` would otherwise drop the settings silently."""
    pytest.importorskip('xsimlab')
    pytest.importorskip('fastscape')
    m = siim2d(dict(_SHEET, flow_routing='single', T=1e3, nt=2, nt_out=2, **kw))
    with pytest.raises(ValueError, match='STANDALONE-only'):
        m.run(driver='xsimlab')


# ---------------------------------------------------------------------------
# routing_filter_factor (f): the routing-surface spatial filter.
#
# ``_FILTERED`` is the frozen-bed parity fixture at a spacing where the window
# is actually open: the threshold is H > dx/(2f), and at dx = 500 m this
# fixture's 55 m of ice never reaches it.
# ---------------------------------------------------------------------------
_FILTERED = dict(_FROZEN, nx=61, ny=61, Lx=1e4, Ly=1e4)


# Subtracting a Fraction keeps the exact-arithmetic path exact and is bit-for-bit
# ``- 0.5`` on the float one (Fraction falls back to float for a float operand).
_HALF = Fraction(1, 2)


def _filter_definition(zs, H, dx, factor, cells=None):
    """The DEFINITION, one cell at a time — no shared structure with production
    (which sums whole-grid shifts per level and averages the deviation from
    ``zs.min()``): a cell with ``H > dx/(2f)`` has half-width
    ``r = min(f*H/dx - 1/2, max(shape))``, and its value interpolates linearly,
    in ``t = r - floor(r)``, between the thickness-weighted triangular means at
    the two bracketing integer half-widths m — each the direct sum over the
    ``(2m+1)^2`` window clipped to the domain, with weights
    ``(m+1-|di|)*(m+1-|dj|)*H[i+di, j+dj]``. Every other cell (dry, or
    sub-threshold) keeps its ``zs``.

    The means are taken of the DEVIATION from the cell's OWN ``zs`` and the
    reference added back: an exact weighted mean does not depend on the
    reference, so this both transcribes the definition and checks production's
    ``zs.min()`` choice against a different one. Arithmetic-agnostic — pass
    ``Fraction`` arrays and ``dx``/``factor`` and the result is exact.

    ``cells`` restricts the transcription to an iterable of ``(i, j)``; the
    O(w^2)-per-cell loop is affordable on the small fixtures, not on 300^2."""
    ny, nx = H.shape
    out = zs.copy()
    for i, j in (cells if cells is not None else
                 ((i, j) for i in range(ny) for j in range(nx))):
        r = min(factor * H[i, j] / dx - _HALF, max(H.shape))
        if r <= 0:
            continue
        m0 = math.floor(r)
        t = r - m0
        means = []
        for m in (m0, m0 + 1):
            numerator = denominator = 0
            for di in range(max(-m, -i), min(m, ny - 1 - i) + 1):
                for dj in range(max(-m, -j), min(m, nx - 1 - j) + 1):
                    w = (m + 1 - abs(di)) * (m + 1 - abs(dj)) * H[i + di, j + dj]
                    numerator += w * (zs[i + di, j + dj] - zs[i, j])
                    denominator += w
            means.append(numerator / denominator)
        out[i, j] = zs[i, j] + (1 - t) * means[0] + t * means[1]
    return out


@pytest.mark.pin
@pytest.mark.parametrize('base', [_SHEET, _VALLEY],   # valley: looped y, f = 0 still accepted
                         ids=['sheet', 'valley-modeC-standard'])
def test_explicit_filter_default_is_the_unparameterised_model(base):
    """``routing_filter_factor=0.0`` spelled out == not spelling it."""
    plain = _run(base)
    explicit = _run(base, routing_filter_factor=0.0)
    _assert_same_outputs(plain, explicit, _OUT_ARRAYS + (
        'sediment_flux_out', 'eroded_volume_out', 'rebound_out'))


@pytest.mark.pin
def test_filter_matches_the_definition():
    """The production filter is the settled harness variant: surface target,
    triangular kernel, no window floor. Gated against the per-cell definition
    above on (i) a synthetic field carrying dry cells and a sub-threshold fringe
    and (ii) the real routing surface the sheet fixture reaches after a few
    steps.

    Production factorises the 2-D window into two 1-D whole-grid passes and
    averages the deviation from ``zs.min()``; the oracle sums one window at a
    time around a cell's own value. Nothing but the definition is shared, so
    they agree to a few ULP of the surface, not bit-for-bit. 1e-11 is ~40 ULP of
    these 1e3 m fields; measured here: 6.8e-13 (synthetic) and 1.1e-13
    (fixture). The branches that ARE exact (dry cells, sub-threshold cells, a
    flat field, f = 0) are asserted as such."""
    rng = np.random.default_rng(20260917)
    n, dx, factor = 17, 500., 2.
    H = np.where(rng.random((n, n)) < .25, 0., rng.uniform(1., 900., (n, n)))
    zs = rng.uniform(-500., 1500., (n, n)) + H
    m = _run_frozen(**dict(_FILTERED, T=3e3, nt=4, nt_out=2))
    real_H = m.H_out[-1]
    real_dx = m.Lx / (m.grid_nx - 1)
    real_zs = m.zb_out[-1] + m.hc_over_H * real_H
    for field, thickness, spacing in ((zs, H, dx), (real_zs, real_H, real_dx)):
        raw = thickness <= spacing / (2 * factor)
        assert not raw.all(), "the filter is inactive on this field"
        got = filter_routing_surface(field, thickness, spacing, factor)
        want = _filter_definition(field, thickness, spacing, factor)
        # gate: TIGHTENED, 1e-9 -> 1e-11, and the flat surface from a tolerance
        # to exact equality — the stable sums earn both (astra review item 1).
        assert np.abs(got - want).max() <= 1e-11, np.abs(got - want).max()
        # dry and sub-threshold cells keep their raw value, exactly
        assert np.array_equal(got[raw], field[raw])
    # a flat surface is reproduced, and f = 0 is the identity
    flat = np.full((n, n), 7.)
    assert np.array_equal(filter_routing_surface(flat, H, dx, factor), flat)
    assert np.array_equal(filter_routing_surface(zs, H, dx, 0.), zs)


@pytest.mark.pin
def test_filter_matches_the_rational_example():
    """The review's 5x5 example (astra, 2026-09-17), where the definition closes
    in EXACT rationals: dx = 4, f = 2, ``zs[i, j] = 100 + 3*(5i + j)`` and the H
    below. Cell (2, 2) has radius 3/2, blending the integer-level means 2587/19
    and 1767/13 into 33602/247; the corner (0, 0) has radius 1 (weights 4, 2, 2,
    1 over the clipped window) and comes back at 104. The oracle runs twice —
    in ``Fraction`` for the exact expectation, in float for production."""
    H = np.array([[3, 1, 2, 1, 0],
                  [1, 2, 1, 4, 1],
                  [2, 1, 4, 2, 1],
                  [1, 3, 2, 1, 2],
                  [0, 1, 1, 2, 3]], dtype=float)
    zs = 100. + 3. * (5 * np.arange(5)[:, None] + np.arange(5))
    def exact(a):                     # the same integers, in exact arithmetic
        return np.array([[Fraction(int(v)) for v in row] for row in a],
                        dtype=object)

    want = _filter_definition(exact(zs), exact(H), Fraction(4), Fraction(2))
    assert want[2, 2] == Fraction(33602, 247), want[2, 2]
    assert want[0, 0] == 104, want[0, 0]
    got = filter_routing_surface(zs, H, 4., 2.)
    delta = np.abs(got - want.astype(float)).max()
    assert delta <= 1e-12, delta                      # all 25 cells


@pytest.mark.pin
def test_filter_is_stable_on_a_large_grid():
    """The sums are taken term by term because prefix sums are not accurate
    enough HERE, on a realistic grid. Differencing global prefix sums returned
    a 3 km flat surface with ~1e-7 m of noise on 1000^2 — 100x the router's
    1e-6 m fill increment, and enough to flip 26 837 receiver rows of a surface
    whose every exact weighted mean is 3000 m (astra, 2026-09-17). Averaging
    the deviation from a constant reference makes that case EXACT, and the
    non-flat error stops growing with the grid: 4.5e-13 at n = 100, 300 and
    1000 against the direct definition, where the prefix-sum form gave
    8.1e-10 / 9.3e-9 / 1.0e-7. The sampled cells below (m = 0/1 windows at this
    thickness) come back exact."""
    rng = np.random.default_rng(1)
    H = rng.uniform(250., 400., (1000, 1000))
    flat = np.full(H.shape, 3000.)
    assert np.array_equal(filter_routing_surface(flat, H, 1000., 2.), flat)
    H = rng.uniform(250., 400., (300, 300))
    zs = 3000. + rng.uniform(-10., 10., H.shape)
    got = filter_routing_surface(zs, H, 1000., 2.)
    assert not np.array_equal(got, zs), "the filter never opened a window"
    # gate: SAME tolerance, evaluated on a sample — the oracle is now per-cell
    # (independent of production's structure) and O(w^2) on 300^2 whole-grid is
    # not worth the seconds. 200 random active cells plus a whole row, which
    # carries the domain-edge truncation at both ends.
    active = np.argwhere(H > 1000. / (2 * 2.))
    cells = [tuple(c) for c in active[rng.choice(len(active), 200, False)]]
    cells += [(150, j) for j in range(H.shape[1])]
    want = _filter_definition(zs, H, 1000., 2., cells=cells)
    iy, ix = np.array(cells).T
    delta = np.abs(got[iy, ix] - want[iy, ix]).max()
    assert delta <= 1e-12 * 3000., delta


@pytest.mark.pin
def test_harness_recipe_parity_with_the_filter(monkeypatch):
    """Acceptance (b) at f = 2: production at ``(inner_passes=2,
    inner_relax=0.2, routing_filter_factor=2)`` reproduces the harness's
    recipe update with ``router_adaptive_factor=2``, on the same frozen-bed
    fixture as the f = 0 parity test but at a spacing where the window opens.
    The harness filters the ROUTER surface alone and keeps the raw ``zb + H``
    for the mass balance, the transect and the closure; the oracle does the
    same."""
    monkeypatch.setattr(_constants, 'HC_OVER_H', 1.0)
    steps = dict(T=3e3, nt=4, nt_out=2)
    model = _run_frozen(inner_passes=2, inner_relax=0.2,
                        routing_filter_factor=2.0, **dict(_FILTERED, **steps))
    with _Oracle(_frozen(routing_filter_factor=2.0,
                         **dict(_FILTERED, **steps))) as o:
        x = np.zeros(o.shape)
        for _ in range(o.cfg.nt - 1):
            for p in range(2):
                mapped = o.F(x, o.dt if p == 1 else 0.0)
                x = x + 0.2 * (mapped - x)
    assert x.max() > 1.0, "the parity fixture grew no ice"
    assert o.filtered_cells > 0, "the filter never opened a window"
    assert np.abs(x - model.H_out[-1]).max() <= 1e-9   # the plan's gate
    assert np.array_equal(x, model.H_out[-1]), "expected bitwise, same kernels"


def _spy_firewall(monkeypatch):
    """Spy every consumer of the routing surfaces and of the thickness in the
    in-house loop; returns the dict of per-call records (a device that does not
    run records nothing). Shared by the filter and trunk-surface FIREWALL
    pins."""
    from siim._core import driver as drv
    rec = {k: [] for k in ('surface', 'fab_in', 'fab_out', 'routed', 'accumulated',
                           'acc_args', 'acc_Q', 'diffused', 'diffused_out',
                           'diff_erosion', 'transect_H', 'kernel_zb', 'kernel_H',
                           'closure_H', 'carve_H', 'flexure_H', 'staged_H',
                           'staged_Q', 'diag', 'eroded_zb', 'carve_s', 'flexure_s', 'fab_s')}
    real = dict(surface=drv.routing_surface, fab=drv._fabricate_trunk_surface,
                acc=drv.accumulate_glacial_flow, diffuse=drv._inhouse_diffuse,
                sum=drv.sum_erosion, transect=drv.transect_flux,
                carve=drv.carve_bed, flexure=drv.glacial_flexure_step,
                loop=drv.run_loop, snapshot=drv._snapshot,
                diag=drv._diagnostic_discharge, filter=drv.filter_routing_surface)
    real_route = {}

    def surface(*a, **kw):
        rec['surface'].append(real['surface'](*a, **kw))
        return rec['surface'][-1]

    def fab(zs, *a, **kw):
        rec['fab_in'].append(zs.copy())
        rec['fab_s'].append(kw.get('s_conf'))
        rec['fab_out'].append(real['fab'](zs, *a, **kw))
        return rec['fab_out'][-1]

    def acc(zs, *a, **kw):
        rec['accumulated'].append(np.asarray(zs).copy())
        rec['acc_args'].append((a, kw))
        out = real['acc'](zs, *a, **kw)
        rec['acc_Q'].append(out[0].copy())
        return out

    def diffuse(zs, *a, **kw):
        rec['diffused'].append(np.asarray(zs).copy())
        rec['diffused_out'].append(real['diffuse'](zs, *a, **kw))
        return rec['diffused_out'][-1]

    def sum_erosion(glacial, diffusion):
        rec['diff_erosion'].append(np.asarray(diffusion).copy())
        return real['sum'](glacial, diffusion)

    def transect(Q, H, *a, **kw):
        rec['transect_H'].append(np.asarray(H).copy())
        return real['transect'](Q, H, *a, **kw)

    def carve(zb_flat, H_flat, *a, **kw):
        rec['carve_H'].append(np.asarray(H_flat).copy())
        rec['carve_s'].append(kw.get('s_conf'))
        return real['carve'](zb_flat, H_flat, *a, **kw)

    def flexure(topo, denudation, forcing, H, *a, **kw):
        rec['flexure_H'].append(np.asarray(H).copy())
        rec['flexure_s'].append(kw.get('s_conf'))
        return real['flexure'](topo, denudation, forcing, H, *a, **kw)

    def snapshot(buffers, jf, cur):
        rec['staged_H'].append(cur['glacial_spl__ice_thickness'].copy())
        rec['staged_Q'].append(cur['glacial_flow__ice_flux'].copy())
        return real['snapshot'](buffers, jf, cur)

    def kernel(dt, zb0, zb1, H0, H1):
        rec['kernel_zb'].append(zb0)         # the post-uplift bed
        rec['kernel_H'].append(H0)           # the raw iterate entering the pass
        rec['closure_H'].append(H1.copy())   # F(x): the pass's fresh closure
        rec['eroded_zb'].append(zb1.copy())  # the bed the kernel just eroded

    def loop(cfg):
        real_route['route'] = cfg.route

        def route(zs):
            rec['routed'].append(np.asarray(zs).copy())
            return real_route['route'](zs)
        cfg.route = route
        return real['loop'](cfg)

    def diag(cfg, zb, H, *a, **kw):
        # mode D's slab pass (FIREWALL clause 3): recorded on its own, run on
        # the real router, filter and accumulator, so the spies above see the
        # step's own consumers alone
        rec['diag'].append((np.asarray(zb).copy(), np.asarray(H).copy()))
        with monkeypatch.context() as mp:
            mp.setattr(drv, 'accumulate_glacial_flow', real['acc'])
            mp.setattr(drv, 'filter_routing_surface', real['filter'])
            mp.setattr(cfg, 'route', real_route['route'])
            return real['diag'](cfg, zb, H, *a, **kw)

    for name, spy in (('routing_surface', surface), ('_fabricate_trunk_surface', fab),
                      ('accumulate_glacial_flow', acc), ('_inhouse_diffuse', diffuse),
                      ('sum_erosion', sum_erosion), ('transect_flux', transect),
                      ('carve_bed', carve), ('glacial_flexure_step', flexure),
                      ('_snapshot', snapshot), ('run_loop', loop),
                      ('_diagnostic_discharge', diag)):
        monkeypatch.setattr(drv, name, spy)
    _spy_kernel(monkeypatch, kernel)
    return rec


@pytest.mark.pin
def test_filter_reaches_the_router_alone(monkeypatch):
    """Acceptance (c), the FIREWALL pin. At f = 2 the FILTERED surface
    constructs the router input ALONE: the accumulator's mass-balance surface
    and the ice-surface hillslope diffusion take the unfiltered
    ``routing_surface(...)``, the transect and the closure the raw pass
    iterate, and the carve, the flexure load and the staged outputs the
    committed state.

    Run DAMPED and multi-pass (n = 2, omega = 0.2) so that the fresh closure
    and the committed state are different arrays — at n = 1, omega = 1 they
    coincide and the staging assertions are vacuous — and spy ``_snapshot``
    itself, so what reaches the output buffers is checked where it is staged."""
    rec = _spy_firewall(monkeypatch)
    surfaces, routed, accumulated, diffused, staged = (
        rec[k] for k in ('surface', 'routed', 'accumulated', 'diffused',
                         'staged_H'))
    transect_H, kernel_H, closure_H, carve_H, flexure_H = (
        rec[k] for k in ('transect_H', 'kernel_H', 'closure_H', 'carve_H',
                         'flexure_H'))

    # gate: every assertion below is the SAME expectation re-indexed for the
    # damped multi-pass run this pin now makes (n passes per step instead of
    # one), plus the two the damping makes non-vacuous: the staged state is the
    # blend, not the closure (astra review item 1's coverage caveat).
    nt, n_pass = 4, 2
    m = _run(dict(_FILTERED, mode='C', flow_routing='single',
                  trunk_surface=False, flexure=True, track_sediment=True),
             T=3e3, nt=nt, nt_out=nt, routing_filter_factor=2.0,
             inner_passes=n_pass, inner_relax=0.2)
    steps = nt - 1
    assert len(routed) == len(surfaces) == len(kernel_H) == n_pass * steps
    # gate: the MECHANISM, not the fixture's growth ramp — which passes the
    # window opens on is a growth-rate detail. What is pinned is that at least
    # one pass really was filtered (otherwise every assertion below is vacuous),
    # and that on those passes the accumulator and the diffuser still received
    # the unfiltered surface bit-identically (asserted just after).
    changed = [int((r != s).sum()) for r, s in zip(routed, surfaces)]
    # ...and on a step's LAST pass, or the diffuser arm below is vacuous.
    assert any(changed[n_pass * s + n_pass - 1] > 0 for s in range(steps)), changed
    for i, zs in enumerate(surfaces):
        # the unfiltered surface, to the accumulator — once per pass
        assert np.array_equal(accumulated[i], zs)
        # the raw pass iterate, to the transect and the closure, never a
        # filtered surface
        assert np.array_equal(transect_H[i].ravel(), kernel_H[i])
    for step in range(steps):
        last = n_pass * step + n_pass - 1
        # the last pass's unfiltered surface, to the hillslope diffusion
        assert np.array_equal(diffused[step], surfaces[last])
        # the state this step ENTERS with (H = 0 on the first), to pass 0
        entering = np.zeros_like(m.H_out[0]) if step == 0 else m.H_out[step - 1]
        assert np.array_equal(kernel_H[n_pass * step].reshape(entering.shape),
                              entering)
        # the committed state — the damped BLEND, not the last pass's fresh
        # closure — to the carve, the flexure load and the staged outputs
        assert not np.array_equal(closure_H[last].reshape(entering.shape),
                                  m.H_out[step]), "omega < 1 committed F(x)"
        assert np.array_equal(carve_H[step].reshape(entering.shape),
                              m.H_out[step])
        assert np.array_equal(flexure_H[step], m.H_out[step])
        assert np.array_equal(staged[step], m.H_out[step])


# The trunk-surface FIREWALL fixture: the mode-C valley at 250 m (alpha_g*H/2 >
# dx within a few steps, so the V's footprint is live), nonzero uplift, one
# undamped pass a step, a materially active hillslope diffusion and flexure on.
_TRUNK = dict(_VALLEY, nx=31, ny=31, Lx=7.5e3, Ly=7.5e3, flow_routing='single',
              trunk_surface=True, inner_passes=1, inner_relax=1.0, D=1.0, nt_out=7)


@pytest.mark.pin
def test_trunk_surface_reaches_the_router_alone(monkeypatch):
    """The FIREWALL pin for the fabricated trunk surface: the V constructs the
    router input ALONE. The accumulator's mass-balance surface, the hillslope
    diffusion and its erosion subtraction take the dynamic surface
    ``routing_surface(post_uplift, hc, H)`` built from the kernel's own input;
    the transect and the kernel the raw iterate; the carve, the flexure load and
    the staged outputs the committed state. Whole arrays, every step; the
    fixture makes the V clause non-vacuous (asserted: a live footprint)."""
    rec = _spy_firewall(monkeypatch)
    m = _run(_TRUNK)
    hc, shape, steps = m.hc_over_H, m.H_out[0].shape, _TRUNK['nt'] - 1
    assert len(rec['routed']) == len(rec['fab_out']) == len(rec['surface']) == steps
    live = [int((v != s).sum()) for v, s in zip(rec['fab_out'], rec['surface'])]
    assert sum(live) > 0, live
    for k in range(steps):
        zs = rec['surface'][k]
        # gate: routing_relax (H_eff != raw H) was removed 2026-11-04, so the
        # dynamic surface is now the one rebuilt from the kernel's own (zb, H)
        assert np.array_equal(zs, routing_surface(rec['kernel_zb'][k].reshape(shape), hc,
                                                  rec['kernel_H'][k].reshape(shape)))
        assert np.array_equal(rec['fab_in'][k], zs)
        assert np.array_equal(rec['routed'][k], rec['fab_out'][k])
        assert np.array_equal(rec['accumulated'][k], zs)
        assert np.array_equal(rec['diffused'][k], zs)
        assert np.array_equal(rec['diff_erosion'][k],
                              zs.reshape(shape) - rec['diffused_out'][k])
        entering = np.zeros(shape) if k == 0 else m.H_out[k - 1]
        assert np.array_equal(rec['kernel_H'][k].reshape(shape), entering)
        assert np.array_equal(rec['transect_H'][k].ravel(), rec['kernel_H'][k])
        assert np.array_equal(rec['carve_H'][k].reshape(shape), m.H_out[k])
        assert np.array_equal(rec['flexure_H'][k], m.H_out[k])
        assert np.array_equal(rec['staged_H'][k], m.H_out[k])
        assert np.array_equal(rec['staged_Q'][k], rec['acc_Q'][k])


@pytest.mark.pin
def test_trunk_surface_mass_balance_is_the_dynamic_one(monkeypatch):
    """Where accumulating on the V would change the ice flux, the flux is the
    dynamic surface's. P is raised so the precipitation cap on the ice source
    (the 2D accumulator's only cap; there is no uncapped switch in 2D) is slack
    on the footprint and the V's lowered surface would move the source."""
    rec = _spy_firewall(monkeypatch)
    _run(_TRUNK, P=50.0)
    moved = 0
    for k, (a, kw) in enumerate(rec['acc_args']):
        dynamic = accumulate_glacial_flow(rec['surface'][k], *a, **kw)[0]
        on_v = accumulate_glacial_flow(rec['fab_out'][k], *a, **kw)[0]
        assert np.array_equal(rec['acc_Q'][k], dynamic)
        moved += int((on_v != dynamic).sum())
    assert moved > 0, "the V would not have moved the ice flux: vacuous"


def test_filtered_sheet_smoke():
    """Acceptance (d) with the filter: 61^2 mode-B MFD sheet at (2, 0.2, f=2) —
    finite, non-negative, and within a generous band of the unfiltered run."""
    cfg = dict(_SHEET, nx=61, ny=61, Lx=1e4, Ly=1e4, T=1e4, nt=11, nt_out=3)
    one = _run(cfg, inner_passes=2, inner_relax=0.2)
    filtered = _run(cfg, inner_passes=2, inner_relax=0.2, routing_filter_factor=2.0)
    H = filtered.H_out
    assert np.isfinite(H).all()
    assert (H >= 0).all()
    assert H[-1].max() > 0.0
    assert not np.array_equal(one.H_out[-1], H[-1]), "the filter changed nothing"
    v_one, v_filtered = one.H_out[-1].sum(), H[-1].sum()
    assert 0.25 * v_one <= v_filtered <= 4.0 * v_one, (v_one, v_filtered)


# --- filter guards ---------------------------------------------------------
@pytest.mark.parametrize('kw', [
    dict(routing_filter_factor=-1.0),
    dict(routing_filter_factor=np.nan),
    dict(routing_filter_factor=np.inf),
    dict(routing_filter_factor=True),
    dict(routing_filter_factor=np.bool_(True)),
])
def test_filter_validation_rejects(kw):
    with pytest.raises(ValueError, match='routing_filter_factor'):
        siim2d(dict(_SHEET, **kw))


# gate: the 'looped' case is retired — the filter now wraps on looped axes
# (mode D, plan-2026-11-02-mode-d B5); test_filter_commutes_with_a_seam_shift
# pins the new semantics and test_filter_accepts_a_looped_grid the facade.
@pytest.mark.parametrize('kw,match', [
    (dict(Ly=2e4, flow_routing='single'), 'square cells'),
    (dict(mode='A', flow_routing='single'), "mode 'B' or 'C'"),
    (dict(mode='C', flow_routing='single', trunk_surface=True), 'trunk_surface'),
], ids=['rectangular', 'mode-A', 'trunk-surface'])
def test_filter_geometry_guards(kw, match):
    """f > 0 is refused where the filter has no defined semantics (no
    independent dy), where there is no ice surface to filter, and alongside the
    opposite-regime trunk surface. f = 0 is always fine."""
    with pytest.raises(ValueError, match=match):
        siim2d(dict(_SHEET, routing_filter_factor=2.0, **kw))
    siim2d(dict(_SHEET, routing_filter_factor=0.0, **kw))


@pytest.mark.pin
@pytest.mark.parametrize('wrap', [(True, False), (False, True), (True, True)],
                         ids=['y', 'x', 'xy'])
def test_filter_commutes_with_a_seam_shift(wrap):
    """On a looped axis the windows wrap: rolling the input k cells along it
    rolls the output k cells, exactly (the shifts are periodic rolls summed in
    the same order). Thick cells open windows wider than half the period, which
    the wrap caps at (n-1)/2 cells."""
    rng = np.random.default_rng(11)
    ny, nx, dx = 15, 18, 500.
    H = np.where(rng.random((ny, nx)) < .3, 0., rng.uniform(1., 3000., (ny, nx)))
    zs = rng.uniform(-500., 1500., (ny, nx)) + H
    base = filter_routing_surface(zs, H, dx, 2., *wrap)
    assert not np.array_equal(base, zs)
    for axis, looped in enumerate(wrap):
        if looped:
            for k in (1, 7):
                got = filter_routing_surface(np.roll(zs, k, axis), np.roll(H, k, axis),
                                             dx, 2., *wrap)
                assert np.array_equal(got, np.roll(base, k, axis)), (axis, k)
    if not all(wrap):          # the other axis still truncates: no seam symmetry there
        axis = wrap.index(False)
        got = filter_routing_surface(np.roll(zs, 1, axis), np.roll(H, 1, axis), dx, 2., *wrap)
        assert not np.array_equal(got, np.roll(base, 1, axis))


@pytest.mark.parametrize('n', [5, 6])
def test_filter_wrap_counts_no_cell_twice(n):
    """A half-width past (n-1)/2 is capped there, so an impulse spreads to each
    cell once: weights m+1-|d| for |d| <= m with m = (n-1)//2, summing to
    (m+1)^2."""
    m = (n - 1) // 2
    impulse = np.zeros((n, 1))
    impulse[0] = 1.
    out = _triangular_axis(impulse, 3 * n, 0, wrap=True)[:, 0]
    want = np.zeros(n)
    for d in range(-m, m + 1):
        want[d % n] = m + 1 - abs(d)
    assert np.array_equal(out, want)
    assert out.sum() == (m + 1) ** 2


@pytest.mark.pin
@pytest.mark.parametrize('wrap', [(False, False), (True, False), (False, True), (True, True)],
                         ids=['open', 'y', 'x', 'xy'])
def test_filter_is_the_level_sums(wrap):
    """The per-cell kernel returns the floats of the whole-grid level sums
    (one ``_triangular_level`` per integer half-width in play, the bracketing
    pair interpolated): same terms, same order, so the router sees the same
    surface to the last bit. Windows up to 12 cells on a 15 x 18 grid cross the
    edges and, on a looped axis, hit the (n-1)//2 cap."""
    rng = np.random.default_rng(23)
    ny, nx, dx, factor = 15, 18, 500., 2.
    H = np.where(rng.random((ny, nx)) < .3, 0., rng.uniform(1., 3000., (ny, nx)))
    zs = rng.uniform(-500., 1500., (ny, nx)) + H
    radius = np.minimum(np.maximum(factor * H / dx - 0.5, 0.0), max(H.shape))
    lower = np.floor(radius).astype(int)
    zs0 = zs.min()
    dev = zs - zs0
    a, b = np.empty(H.shape), np.empty(H.shape)
    for m in np.union1d(lower, lower + 1):
        num, den = _triangular_level(dev, H, H * dev, int(m), *wrap)
        with np.errstate(invalid='ignore', divide='ignore'):
            mean = num if den is None else num / den
        a[lower == m] = mean[lower == m]
        b[lower + 1 == m] = mean[lower + 1 == m]
    want = np.where(radius > 0.0, zs0 + (a + (radius - lower) * (b - a)), zs)
    assert lower.max() >= 10 and (radius > 0.0).sum() > 100, "vacuous: no wide window"
    np.testing.assert_array_equal(filter_routing_surface(zs, H, dx, factor, *wrap), want)


def test_filter_accepts_a_looped_grid():
    """The facade no longer refuses a looped grid, and a looped filtered run
    differs from the unfiltered one (the windows opened)."""
    looped = dict(_FILTERED, boundary_status=['fixed_value', 'fixed_value', 'looped', 'looped'],
                  T=3e3, nt=4, nt_out=2)
    filtered = _frozen(routing_filter_factor=2.0, **looped)
    filtered.run()
    plain = _frozen(**looped)
    plain.run()
    assert np.isfinite(filtered.H_out).all()
    assert not np.array_equal(filtered.H_out[-1], plain.H_out[-1])


@pytest.mark.adapter
def test_adapter_refuses_the_filter_directly():
    """Entry point 1: a direct ``glacial_processes(...)`` call."""
    pytest.importorskip('xsimlab')
    pytest.importorskip('fastscape')
    from siim.fastscape import glacial_processes
    glacial_processes(mode='B', routing_filter_factor=0.0)   # the default is fine
    with pytest.raises(ValueError, match='STANDALONE-only'):
        glacial_processes(mode='B', routing_filter_factor=2.0)


@pytest.mark.adapter
def test_adapter_refuses_the_filter_through_the_model():
    """Entry point 2: a configured ``siim2d(...).run(driver='xsimlab')``, whose
    ``_process_overrides`` would otherwise drop the setting silently."""
    pytest.importorskip('xsimlab')
    pytest.importorskip('fastscape')
    # the undamped single pass, so the damping refusal does not fire first (audit #51)
    m = siim2d(dict(_SHEET, flow_routing='single', T=1e3, nt=2, nt_out=2,
                    inner_passes=1, inner_relax=1.0, routing_filter_factor=2.0))
    # gate: matched only 'STANDALONE-only', which the damping refusal also says (audit #51)
    with pytest.raises(ValueError, match='routing_filter_factor is STANDALONE-only'):
        m.run(driver='xsimlab')
