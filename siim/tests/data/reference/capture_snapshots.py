#!/usr/bin/env python
"""The reference battery — freeze + reproduce gate for the standalone 2D model.

This script runs the 2D model ON ITS SHIPPED DEFAULTS — the in-house driver +
in-house D8 router + in-house flexure/diffusion, since the S5 standalone flip
(sanctioned regeneration #4, plan decision note 11) — and freezes a small,
deterministic set of outputs into git-tracked ``.npz`` files under this
directory. The battery is the standing determinism tripwire: **the standalone
default path reproduces bit-for-bit, per platform** (any routing / numerics /
driver regression breaks it exactly). Through S0–S4 the battery pinned the
fortran path the migration diffed against; those pins died with the fortran
arms at S5 and the 14 config ``.npz`` were re-frozen from the standalone
defaults (pre-measured deltas: dinf configs 0.0 from the router; SFR mode A
~1.6 m rms; SFR B/C ~100–170 m rms — the ratified router-multistability +
in-house-flexure consequences, not gate failures).

The 3 ``router_*.npz`` are DIFFERENT: they stay the S4-frozen RAW fortran-SFR
baselines (surface + raw int receivers/stack + lengths) that
``test_d8_receiver_parity`` (PIP) diffs the in-house D8 against — never
regenerated. Their fidelity to live fortran is re-certified by the conda
(adapter) leg, which routes the FROZEN surface through fastscapelib-fortran
directly (never a regenerated surface — plan decision note 12).

Frozen artifacts are IMMUTABLE outside an explicitly decision-noted sanctioned
regeneration (five to date; see the plan). CI writes them only with
``--if-absent`` so a committed reference is never silently overwritten;
``test_reference_snapshots.py`` is the reproduce/gate.

Determinism (Map 4 §4):

* ``seed=111``, fixed grid/clock; the model seeds RNG only at ``initialize``
  (no per-step RNG), so a rerun in the same env reproduces bit-for-bit.
* ``basin`` arrays are EXCLUDED from everything frozen (the in-house labels
  are deterministic, but basin stays out of equality gates by protocol rule 5).
* Raw fortran rec/stack/length are deterministic AND platform-independent
  (integer outputs of exact IEEE float comparisons — probe CI run
  29273331972), so the router refs are gated bit-for-bit; the old zarr
  int-0->NaN read-back artifact died with the S4 raw re-freeze.
* Every config sets ``boundary_status`` EXPLICITLY to
  ``['fixed_value','fixed_value','looped','looped']`` (ratified OQ-1(b), now
  also the shipped default) — keeps the references comparable across the
  whole migration.

Usage::

    python capture_snapshots.py --write [--if-absent]   # freeze (pass 1)
    python capture_snapshots.py --check [--parallel on|off|both]  # reproduce-gate
    python capture_snapshots.py --list                  # print the matrix

``SIIM_S0_REF_DIR=<dir>`` redirects both ``--write`` and ``--check`` (and the
suite's ``test_reference_snapshots.py``) at a locally-frozen battery — required
off-CI, because the committed battery is platform-bound (see ``REF_DIR`` below).

The reproduce-gate (``--check``) reruns every config and asserts
``np.array_equal`` vs the frozen ``.npz``, under ``parallel_erode`` ON and OFF
(``constants.PARALLEL_ERODE``; serial==parallel is bit-for-bit,
``test_parallel_erode.py``). That is S0's whole point: determinism.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

# The committed reference battery is PLATFORM-BOUND: it was frozen in CI
# (linux/x86-64) and reproduces bit-for-bit only in that environment class.
# Cross-platform float differences (libm, FMA codegen) amplify through the
# autogenic dynamics to O(1) by the final frame (measured on macOS/arm64,
# 2026-07-13 — plan doc, S0 platform addendum). ``SIIM_S0_REF_DIR`` points the
# freeze/gate at a LOCAL battery (convention:
# ``siim/tests/data/reference_local/``, gitignored) so every platform gates
# bit-for-bit against references frozen on that platform; the committed
# ``.npz`` stay immutable and authoritative in CI.
REF_DIR = Path(os.environ.get('SIIM_S0_REF_DIR') or Path(__file__).resolve().parent)

# --- deterministic matrix knobs (Map 4 S0) -------------------------------------
NX = NY = 31
NT = 251
NT_OUT = 11
SEED = 111
# The standard explicit boundary (task ratified OQ-1): left/right fixed,
# top/bottom looped. NOT the 'core' default (whose meaning S4 changes).
BSTD = ['fixed_value', 'fixed_value', 'looped', 'looped']
LX = LY = 2.0e4

# The two frozen frames: a mid output time and the final output time.
MID = NT_OUT // 2          # 5  -> master step 125
FINAL = NT_OUT - 1         # 10 -> master step 250

# Fields frozen per config (mode-aware attrs on the siim2d model; basin excluded).
_FIELDS = ('z', 'zb', 'H')


def _base(**ov):
    """A small, strongly-glaciating 2D config (ice reaches the fixed borders),
    mirroring the known-glaciating test configs (test_parallel_erode/_denudation)
    but on the S0 deterministic grid/clock."""
    p = dict(
        U=1e-3, zELA=150, beta=1e-2, P=2, alpha_g=12, Ko=2e-6, n=1, ce=1e-4,
        nu=2, sliding_law='power', lambda_p=500, k=0.9,
        T=5e4, nt=NT, nt_out=NT_OUT, Lx=LX, Ly=LY, nx=NX, ny=NY, seed=SEED,
        boundary_status=list(BSTD), initial_max_elevation=800,
        # NO backend pins since the S5 standalone flip (sanctioned regeneration
        # #4): the battery runs — and therefore certifies — the SHIPPED default
        # path (in-house driver + inhouse_d8 router + in-house flexure/
        # diffusion). The S0–S4 fortran pins died with the fortran arms; the
        # 14 config .npz were re-frozen from these defaults.
        progress_bar=False)
    p.update(ov)
    return p


def _bl_series():
    """A step DROP in the water-line datum at the mid step — launches a
    knickpoint (mode B only; CLAUDE.md base-level BC)."""
    bl = np.zeros(NT)
    bl[NT // 2:] = -30.0
    return bl


def matrix():
    """Return ``{key: config}`` for the ~14-config deterministic battery:
    {mode A, B, C} x {SFR, dinf} x {flexure off; flexure on, ice_load} + a
    bl(t) forcing case on each routing (mode B). Carve is NOT an independent
    axis: mode C = B + carve (DEFAULT_MODE_2D='C')."""
    cfgs = {}
    for mode in ('A', 'B', 'C'):
        for routing in ('single', 'dinf'):
            for flex in (False, True):
                key = f"{mode}_{routing}_{'flexon' if flex else 'flexoff'}"
                ov = dict(mode=mode, flow_routing=routing)
                if flex:
                    ov.update(flexure=True, ice_load=True, e_thickness=20e3)
                cfgs[key] = _base(**ov)
    # bl(t) forcing (mode B, both routings) — the base-level forcing case.
    for routing in ('single', 'dinf'):
        cfgs[f"B_{routing}_blstep"] = _base(
            mode='B', flow_routing=routing, bl=_bl_series())
    return cfgs


def run_config(cfg, parallel_erode=None):
    """Run one config; return the two frozen frames (mid, final) of z/zb/H.

    ``parallel_erode`` overrides the model default when not None (the gate
    reruns each config under both to prove serial==parallel bit-for-bit)."""
    from siim.siim2d import siim as siim2d
    c = dict(cfg)
    if parallel_erode is not None:
        c['parallel_erode'] = parallel_erode
    m = siim2d(c)
    m.run()
    out = {}
    for f in _FIELDS:
        arr = getattr(m, f'{f}_out')
        out[f'{f}_mid'] = np.ascontiguousarray(arr[MID])
        out[f'{f}_final'] = np.ascontiguousarray(arr[FINAL])
    return out


# --- S4 router references (fortran-SFR receivers / stack / lengths) ------------
# (route_surface_fortran_sfr — the S0 model-path scaffold — was deleted at S5
#  with the fortran router arm it drove; raw_fortran_route below is the S4+
#  reference path.)

def _ibc(boundary_status):
    """fastscape ``ibc`` integer from a [left, right, top, bottom] status list
    (digit 1 per 'fixed_value'): ``left*1 + right*100 + top*1000 + bottom*10``."""
    bs = list(np.broadcast_to(boundary_status, 4))
    d = [1 if s == 'fixed_value' else 0 for s in bs]   # left, right, top, bottom
    return d[0] * 1 + d[1] * 100 + d[2] * 1000 + d[3] * 10


def raw_fortran_route(surface, boundary_status=None):
    """Route ``surface`` through fastscapelib-fortran's SFR DIRECTLY — no siim
    model, no xsimlab, no zarr round-trip — returning FLAT raw arrays
    ``(receivers, stack, lengths)``: ``receivers`` / ``stack`` 0-based int64,
    ``lengths`` float64. This is the S4-sanctioned re-freeze path
    (``tools/probe_router_determinism.py``): the old harness read the receivers
    back through the xsimlab zarr store, whose integer ``fill_value=0`` +
    xarray ``mask_and_scale`` decoded every value-0 (node-0 receiver) cell to
    NaN and promoted int->float64, so ``np.array_equal`` flagged those FIXED
    cells as 'nondeterministic'. Read raw, the fortran ``rec``/``stack``/
    ``length`` are bit-identical across repeats and processes (probe CI run
    29273331972) — deterministic, and integer outputs of exact IEEE float
    comparisons are platform-independent, so these raw refs are the exact S4
    receiver-parity baseline on every platform."""
    import fastscapelib_fortran as fs
    if boundary_status is None:
        boundary_status = list(BSTD)
    surface = np.asarray(surface, dtype=float)
    ny, nx = surface.shape
    fs.fastscape_init()
    fs.fastscape_set_nx_ny(nx, ny)
    fs.fastscape_setup()
    fs.fastscape_set_xl_yl(LX, LY)
    fs.fastscape_set_bc(_ibc(boundary_status))
    fs.fastscapecontext.h = np.ascontiguousarray(surface.ravel(), dtype=float)
    fs.flowroutingsingleflowdirection()
    rec = np.ascontiguousarray(np.array(fs.fastscapecontext.rec, dtype=np.int64) - 1)
    stack = np.ascontiguousarray(np.array(fs.fastscapecontext.stack, dtype=np.int64) - 1)
    length = np.ascontiguousarray(np.array(fs.fastscapecontext.length, dtype=np.float64))
    fs.fastscape_destroy()
    return rec, stack, length


ROUTER_SURFACE_NAMES = ('xtilt', 'diag', 'glaciated_modeC')


def router_surface(name):
    """Build ONE named battery surface to freeze fortran-SFR routing for (S4
    parity). Two synthetic NON-DEGENERATE surfaces (unique strict steepest
    descent everywhere -> exact byte-parity targets for the in-house D8) + one
    REAL glaciated mode-C snapshot with overdeepenings (quantifies the routing
    delta, study §3.5). Deterministic and self-contained."""
    if name == 'xtilt':
        # Cardinal x-tilt: highest at the right (fixed) edge, drains straight
        # left. Every interior receiver is the left neighbor — a clean D8/D-inf
        # cardinal-parity target.
        i = np.arange(NX)
        return np.broadcast_to((NX - 1 - i) * 5.0, (NY, NX)).copy()
    if name == 'diag':
        # Diagonal ramp toward the (0,0) corner — exercises diagonal receivers.
        jj, ii = np.meshgrid(np.arange(NY), np.arange(NX), indexing='ij')
        return (ii * 3.0 + jj * 2.0).astype(float)
    if name == 'glaciated_modeC':
        # A real glaciated mode-C snapshot (final bed+ice surface with
        # overdeepenings), generated deterministically here. Used ONLY when
        # freezing a ref that does not exist yet (fresh bootstrap): the
        # COMMITTED ref's surface is immutable and every re-certification
        # loads it from the .npz instead (frozen_router_surface — plan
        # decision note 12; a rerun of this function post-S5 produces a
        # DIFFERENT, in-house-routed landscape).
        from siim.siim2d import siim as siim2d
        m = siim2d(_base(mode='C', flow_routing='single'))
        m.run()
        return np.ascontiguousarray(m.z_out[FINAL])
    raise KeyError(name)


def router_surfaces():
    """All named battery surfaces (see :func:`router_surface`)."""
    return {name: router_surface(name) for name in ROUTER_SURFACE_NAMES}


def router_ref_key(name):
    return f"router_{name}"


def frozen_router_surface(name):
    """The FROZEN surface of a committed router reference — the array live
    fortran is re-driven on for every re-certification (never a regenerated
    surface: the glaciated_modeC landscape is a model output whose rerun
    changed with the S5 default flip — plan decision note 12)."""
    return np.ascontiguousarray(
        np.load(npz_path(router_ref_key(name)))['surface'], dtype=float)


# EVERY router reference array reproduces bit-for-bit run-to-run: the S4
# re-freeze (--refreeze-router, orchestrator-sanctioned at the S4 checkpoint)
# replaced the OLD zarr-round-trip receivers/stack/lengths — whose int
# fill_value=0 -> xarray mask_and_scale -> NaN self-compare made value-0 cells
# LOOK nondeterministic (probe CI run 29273331972; NOT fortran nondeterminism) —
# with RAW-INTEGER refs (raw_fortran_route, no model/zarr). Raw fortran
# rec/stack/length are deterministic AND platform-independent (integer outputs of
# exact IEEE float comparisons on the same input bytes), so the whole ref is
# gated exactly. basin/catch stays unseeded-random fortran and is never frozen.
ROUTER_DETERMINISTIC_KEYS = ('surface', 'receivers', 'stack', 'lengths', 'meta')


def compute_router_ref(name, surface):
    rec, stack, lengths = raw_fortran_route(surface)
    return dict(surface=np.ascontiguousarray(surface, dtype=float),
                receivers=rec, stack=stack, lengths=lengths,
                meta=np.array([LX, LY, NX, NY], dtype=float))


def refreeze_router_raw(ref_dir=None, verbose=True):
    """S4-sanctioned re-freeze of the three router references as RAW INTEGERS,
    IN PLACE, keeping the frozen ``surface`` byte-identical: load each existing
    ``router_{name}.npz`` surface (do NOT regenerate it — the glaciated_modeC
    surface is a platform-bound model output), drive :func:`raw_fortran_route`
    on it, and re-save with raw flat ``receivers``/``stack``/``lengths`` (+ the
    unchanged ``surface``/``meta``). Supersedes the old zarr-round-trip refs.
    Returns the count re-frozen."""
    d = Path(ref_dir) if ref_dir is not None else REF_DIR
    n = 0
    for name in ROUTER_SURFACE_NAMES:
        path = d / f"{router_ref_key(name)}.npz"
        if not path.exists():
            if verbose:
                print(f"[skip ] {path.name} (absent)")
            continue
        ref = np.load(path)
        surface = np.ascontiguousarray(ref['surface'], dtype=float)
        rec, stack, lengths = raw_fortran_route(surface)
        np.savez_compressed(path, surface=surface, receivers=rec, stack=stack,
                            lengths=lengths, meta=ref['meta'])
        n += 1
        if verbose:
            print(f"[refroze] {path.name} (raw ints; surface byte-identical)")
    return n


# --- freeze / check plumbing --------------------------------------------------

def npz_path(key):
    return REF_DIR / f"{key}.npz"


def _write_one(key, data, if_absent):
    path = npz_path(key)
    if if_absent and path.exists():
        return False
    np.savez_compressed(path, **data)
    return True


def write_all(if_absent=False, verbose=True):
    """Freeze the whole battery + router references. Returns the count written."""
    written = 0
    for key, cfg in matrix().items():
        if if_absent and npz_path(key).exists():
            if verbose:
                print(f"[skip ] {key}.npz (exists)")
            continue
        data = run_config(cfg)
        if _write_one(key, data, if_absent):
            written += 1
            if verbose:
                print(f"[write] {key}.npz")
    for name in ROUTER_SURFACE_NAMES:
        key = router_ref_key(name)
        if if_absent and npz_path(key).exists():
            # Never touch (or even recompute the surface of) a committed
            # router ref — it is the immutable S4 fortran baseline.
            if verbose:
                print(f"[skip ] {key}.npz (exists)")
            continue
        data = compute_router_ref(name, router_surface(name))
        if _write_one(key, data, if_absent):
            written += 1
            if verbose:
                print(f"[write] {key}.npz")
    return written


def _assert_equal(key, ref, got):
    for name in ref.files:
        a = ref[name]
        b = got[name]
        if not np.array_equal(a, b):
            n_diff = int(np.sum(a != b)) if a.shape == b.shape else -1
            raise AssertionError(
                f"reference {key}.npz array '{name}' did not reproduce "
                f"bit-for-bit ({n_diff} cells differ; shape ref={a.shape} "
                f"got={b.shape})")


def assert_router_ref(key, ref, got, verbose=True):
    """Verify a frozen router reference BIT-FOR-BIT: after the S4 raw re-freeze
    every array (surface + raw-integer receivers/stack + lengths + meta)
    reproduces exactly — raw fortran SFR routing is deterministic (the old
    'nondeterministic receivers' were a zarr NaN artifact; see
    :func:`raw_fortran_route`)."""
    for name in ROUTER_DETERMINISTIC_KEYS:
        if not np.array_equal(ref[name], got[name]):
            n_diff = (int(np.sum(ref[name] != got[name]))
                      if ref[name].shape == got[name].shape else -1)
            raise AssertionError(
                f"router reference {key}.npz array '{name}' did not reproduce "
                f"bit-for-bit ({n_diff} cells differ)")
    if verbose:
        print(f"[ok   ] {key} (raw router ref: bit-for-bit)")


def check_all(parallel='both', verbose=True, battery_only=False):
    """Reproduce-gate: rerun every frozen config and assert bit-for-bit vs the
    committed ``.npz``, under ``parallel_erode`` on/off. Raises on any mismatch;
    returns the number of (config, toggle) pairs verified.

    ``battery_only`` skips the 3 router-ref legs — they re-drive LIVE fortran
    (conda/adapter env only), while the battery legs are pip-runnable; the
    pip-based ``s0-capture`` workflow gates with ``--battery-only`` and the
    router re-cert runs in the conda adapter job
    (``test_router_reference_frozen``)."""
    modes = {'on': [True], 'off': [False], 'both': [True, False]}[parallel]
    verified = 0
    cfgs = matrix()
    missing = [k for k in cfgs if not npz_path(k).exists()]
    if missing:
        raise FileNotFoundError(
            f"missing frozen references: {missing[:3]}… run --write first")
    for pe in modes:
        for key, cfg in cfgs.items():
            ref = np.load(npz_path(key))
            got = run_config(cfg, parallel_erode=pe)
            _assert_equal(key, ref, got)
            verified += 1
            if verbose:
                print(f"[ok   ] {key} (parallel_erode={pe})")
    # Router references: raw fortran SFR (deterministic read raw) — gated
    # bit-for-bit on every array, re-driving live fortran on the FROZEN
    # surface (decision note 12: never a regenerated surface).
    # Parallel_erode-invariant (router != eroder), so checked once.
    # Conda-only leg (raw_fortran_route imports fastscapelib_fortran).
    if battery_only:
        if verbose:
            print("[skip ] router refs (--battery-only; conda adapter leg)")
        return verified
    for name in ROUTER_SURFACE_NAMES:
        key = router_ref_key(name)
        if not npz_path(key).exists():
            raise FileNotFoundError(f"missing router reference: {key}.npz")
        ref = np.load(npz_path(key))
        got = compute_router_ref(name, frozen_router_surface(name))
        assert_router_ref(key, ref, got, verbose=verbose)
        verified += 1
    return verified


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--write', action='store_true',
                    help='freeze the battery + router references')
    ap.add_argument('--if-absent', action='store_true',
                    help='with --write: never overwrite an existing .npz '
                         '(frozen artifacts are immutable)')
    ap.add_argument('--check', action='store_true',
                    help='reproduce-gate: rerun and assert bit-for-bit')
    ap.add_argument('--refreeze-router', action='store_true',
                    help='S4: re-freeze the 3 router refs as RAW INTEGERS in '
                         'place (loads + keeps the frozen surface byte-identical)')
    ap.add_argument('--parallel', choices=['on', 'off', 'both'], default='both',
                    help='which parallel_erode setting(s) to check (default both)')
    ap.add_argument('--battery-only', action='store_true',
                    help='with --check: skip the 3 router-ref legs (they need '
                         'live fastscapelib-fortran, i.e. the conda adapter env)')
    ap.add_argument('--list', action='store_true', help='print the matrix keys')
    args = ap.parse_args(argv)

    if args.list:
        for key in matrix():
            print(key)
        for name in ('xtilt', 'diag', 'glaciated_modeC'):
            print(router_ref_key(name))
        return 0
    if args.refreeze_router:
        n = refreeze_router_raw()
        print(f"re-froze {n} router reference(s) as raw integers in {REF_DIR}")
    if args.write:
        n = write_all(if_absent=args.if_absent)
        print(f"wrote {n} reference file(s) to {REF_DIR}")
    if args.check:
        n = check_all(parallel=args.parallel, battery_only=args.battery_only)
        print(f"verified {n} reproduce-gate check(s) bit-for-bit")
    if not (args.write or args.check or args.list or args.refreeze_router):
        ap.error('nothing to do: pass --write, --check, --refreeze-router, or --list')
    return 0


if __name__ == '__main__':
    sys.exit(main())
