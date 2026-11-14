"""Receiver arity in the shared multi-receiver implicit-erosion Newton.

``_solver_nonlinear_dinf`` solves, per cell, the backward-Euler erosion residual
summed over ALL of a cell's receivers::

    F(z) = z - zo_i + sum_k w_k * (dt / L_k**p) * A * max(0, z - z_rk)**p

F is strictly increasing in z, with its root bracketed by
``[min_k z_rk, zo_i]``.  The solver was written for the D-inf pack, where a cell
has at most two receivers, and its allocation-free rewrite (audit N31) kept the
first active receiver in one scalar pair and *overwrote* a second scalar pair
with every later receiver.  At k > 2 that silently dropped receivers
``1 .. k-2`` from the residual, from its derivative, AND from the bracket's
lower bound -- so a middle receiver holding the deepest surface could not be
reached at all.  MFD routes over up to eight receivers, so the truncation had to
go before any k > 2 producer exists.

These tests drive the solver directly against a pure-Python residual oracle
(bisection on the exact k-ary F) at k = 1, 2, 4, 8, and pin the specific
bracket failure: a dropped middle receiver that supplies the true bracket
minimum, whose correct root lies BELOW the truncated bracket and is therefore
unreachable, not merely inaccurate.
"""
import numpy as np
import pytest

from siim._core.solvers import _solver_nonlinear_dinf

# The solver's own convergence tolerance is loose by default (1e-3 m is far
# below the erosion signal it feeds).  Tighten it here so the comparison tests
# the ARITY, not the stopping rule.
TIGHT = dict(epsilon=1e-12, max_iter=400)


def _residual(z, zo_i, zr, g, p):
    """The exact k-ary residual, in plain Python -- the oracle's F."""
    f = z - zo_i
    for zr_k, g_k in zip(zr, g):
        dz = z - zr_k
        if dz > 0.0:
            f += g_k * dz ** p
    return f


def _oracle(zo_i, zr, w, L, A, p, dt):
    """Bisect the exact k-ary residual over ``[min_k z_rk, zo_i]``.

    Mirrors the solver's own early exits (no active receiver, or a cell already
    at/below its lowest receiver) so the two are comparable everywhere.
    """
    act = [k for k in range(len(L)) if L[k] > 0.0]
    if not act:
        return zo_i
    zr_a = [zr[k] for k in act]
    g_a = [w[k] * (dt / L[k] ** p) * A for k in act]
    lo = min(zr_a)
    if zo_i <= lo:
        return zo_i
    hi = zo_i
    for _ in range(200):                       # 200 halvings: exact to ~1e-58
        mid = 0.5 * (lo + hi)
        if _residual(mid, zo_i, zr_a, g_a, p) > 0.0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def _call(zo_i, zr_vals, w, L, A, p, dt):
    """Run the njit solver on a k-receiver bundle laid out the way the packs do
    (active receivers in the contiguous prefix)."""
    k = len(w)
    receivers_i = np.arange(1, k + 1, dtype=np.int64)
    z_flat = np.concatenate([[0.0], np.asarray(zr_vals, dtype=float)])
    return _solver_nonlinear_dinf(
        zo_i, k, receivers_i, np.asarray(w, dtype=float),
        np.asarray(L, dtype=float), z_flat, A, p, dt, **TIGHT)


# --- k-ary agreement with the residual oracle ---------------------------------

@pytest.mark.parametrize('k', [1, 2, 4, 8])
@pytest.mark.parametrize('p', [0.5, 1.0, 2.0],
                         ids=['sublinear', 'linear', 'superlinear'])
def test_kary_solver_matches_residual_oracle(k, p):
    """At every arity the solver must find the root of the residual summed over
    ALL receivers.  Pre-fix this passes at k = 1, 2 and fails at k = 4, 8: the
    middle receivers are missing from F, so the returned root under-erodes."""
    rng = np.random.default_rng(20260905 + k)
    zo_i = 100.0
    # Receiver surfaces spread well below the cell, weights normalised as a
    # real pack's are, lengths mixed cardinal/diagonal.
    zr = zo_i - rng.uniform(2.0, 60.0, size=k)
    w = rng.uniform(0.2, 1.0, size=k)
    w /= w.sum()
    L = np.where(rng.random(k) < 0.5, 100.0, 100.0 * np.sqrt(2.0))
    A, dt = 5.0e-3, 1.0e3

    got = _call(zo_i, zr, w, L, A, p, dt)
    want = _oracle(zo_i, zr, w, L, A, p, dt)
    assert got == pytest.approx(want, abs=1e-9), (
        f"k={k} p={p}: solver {got!r} vs k-ary oracle {want!r}")
    # The root really is interior (the case is not trivially zo_i or min zr).
    assert min(zr) < want < zo_i


@pytest.mark.parametrize('k', [1, 2, 4, 8])
def test_inactive_tail_slots_are_ignored(k):
    """Zero-length tail slots (the producer contract's inactive marker) must not
    enter the residual, whatever ``n_rec`` is passed."""
    zo_i = 50.0
    zr = np.linspace(40.0, 30.0, k)
    w = np.full(k, 1.0 / k)
    L = np.full(k, 100.0)
    A, p, dt = 1.0e-2, 1.5, 500.0

    tight = _call(zo_i, zr, w, L, A, p, dt)
    # Same bundle, padded out to 8 slots with the zeroed inactive tail.
    pad = 8 - k
    padded = _call(zo_i, np.concatenate([zr, np.zeros(pad)]),
                   np.concatenate([w, np.zeros(pad)]),
                   np.concatenate([L, np.zeros(pad)]), A, p, dt)
    assert padded == tight


# --- the bracket failure, pinned ---------------------------------------------

# A cell whose DEEPEST receiver sits in a middle slot.  Pre-fix, slots 0 and 3
# survive (the last active receiver overwrites the second scalar pair), so the
# bracket's lower bound is min(90, 90) = 90 while the true root is 57.37 -- the
# correct answer is outside the truncated bracket entirely, so no amount of
# iteration reaches it.
_BRACKET = dict(
    zo_i=100.0,
    zr=[90.0, 10.0, 90.0, 90.0],
    w=[0.05, 0.90, 0.025, 0.025],
    L=[1.0, 1.0, 1.0, 1.0],
    A=1.0, p=1.0, dt=1.0)
# Analytic: below z = 90 only the slot-1 term is active, so
# F(z) = z - 100 + 0.9*(z - 10) = 0  =>  z = 109/1.9.
_BRACKET_ROOT = 109.0 / 1.9              # 57.36842105...
# What the truncated residual (slots 0 and 3 only) converges to instead:
# F(z) = z - 100 + 0.075*(z - 90) = 0  =>  z = 106.75/1.075.
_BRACKET_TRUNCATED_ROOT = 106.75 / 1.075  # 99.30232558...


def test_dropped_middle_receiver_supplies_the_bracket_minimum():
    """The deepest receiver in a middle slot must set the bracket AND the
    residual.

    Pre-fix the solver returns ~99.3 (root of the two-slot residual, pinned
    above its bracket floor of 90); post-fix it returns 109/1.9 = 57.368...,
    the root of the full four-receiver residual -- which lies BELOW the pre-fix
    bracket, so this is an unreachable-answer failure, not a tolerance one.
    """
    b = _BRACKET
    got = _call(b['zo_i'], b['zr'], b['w'], b['L'], b['A'], b['p'], b['dt'])

    assert got == pytest.approx(_BRACKET_ROOT, abs=1e-9)
    # Guard the demonstration itself: the correct root is strictly below the
    # lower bound the truncated bracket would have used.
    assert _BRACKET_ROOT < min(b['zr'][0], b['zr'][-1])
    assert got < _BRACKET_TRUNCATED_ROOT - 40.0
    # And it agrees with the independent bisection oracle.
    assert got == pytest.approx(
        _oracle(b['zo_i'], b['zr'], b['w'], b['L'], b['A'], b['p'], b['dt']),
        abs=1e-9)


def test_bracket_minimum_in_middle_slot_sublinear_k8():
    """The same failure at k = 8 with a sub-linear exponent -- the regime the
    bracket exists for (plain Newton 2-cycles there)."""
    zo_i = 200.0
    zr = np.full(8, 190.0)
    zr[3] = 5.0                                # deepest receiver, middle slot
    w = np.full(8, 0.02)
    w[3] = 1.0 - 0.02 * 7
    L = np.full(8, 100.0)
    A, p, dt = 2.0, 0.5, 1.0e3

    got = _call(zo_i, zr, w, L, A, p, dt)
    want = _oracle(zo_i, zr, w, L, A, p, dt)
    assert got == pytest.approx(want, abs=1e-9)
    assert want < 190.0, "the case must erode past the surviving receivers"
