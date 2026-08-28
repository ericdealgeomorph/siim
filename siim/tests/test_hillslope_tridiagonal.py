"""Linear-algebra tests for the independently expressed Thomas recurrence.

The systems and dense oracle are defined only by the tridiagonal row equation:
``lower[i] * x[i-1] + diagonal[i] * x[i] + upper[i] * x[i+1] = rhs[i]``.
"""

import numpy as np
import pytest

from siim._core.hillslope import _solve_tridiagonal


def _dense_matrix(lower, diagonal, upper, size):
    """Assemble the dense matrix for the solver's full-length band convention."""
    return (np.diag(diagonal[:size])
            + np.diag(upper[:size - 1], 1)
            + np.diag(lower[1:size], -1))


def test_tridiagonal_scalar_ignores_unused_band_entries():
    lower = np.array([np.nan])
    diagonal = np.array([-4.0])
    upper = np.array([np.nan])
    rhs = np.array([10.0])
    solution = np.array([777.0])

    _solve_tridiagonal(lower, diagonal, upper, rhs, solution, 1)

    np.testing.assert_array_equal(solution, np.array([-2.5]))


@pytest.mark.parametrize(
    ('lower', 'diagonal', 'upper', 'expected'),
    [
        (
            [np.nan, 2.0, -1.0],
            [4.0, 5.0, 6.0],
            [1.0, 3.0, np.nan],
            [1.0, -2.0, 3.0],
        ),
        (
            [np.nan, 2.0, -1.0, 4.0],
            [3.0, -4.0, 5.0, 6.0],
            [-2.0, 3.0, 1.0, np.nan],
            [2.0, -1.0, 3.0, 0.5],
        ),
    ],
    ids=['asymmetric-3x3', 'mixed-sign-4x4'],
)
def test_tridiagonal_hand_solved_systems(lower, diagonal, upper, expected):
    lower = np.asarray(lower)
    diagonal = np.asarray(diagonal)
    upper = np.asarray(upper)
    expected = np.asarray(expected)
    rhs = _dense_matrix(lower, diagonal, upper, expected.size) @ expected
    solution = np.empty_like(rhs)

    _solve_tridiagonal(lower, diagonal, upper, rhs, solution, expected.size)

    np.testing.assert_allclose(solution, expected, rtol=5e-12, atol=5e-13)


@pytest.mark.parametrize('size', [2, 3, 8, 31])
@pytest.mark.parametrize('seed', [0, 17, 104])
def test_tridiagonal_matches_dense_oracle(size, seed):
    rng = np.random.default_rng(seed)
    lower = rng.uniform(-2.0, 2.0, size)
    upper = rng.uniform(-2.0, 2.0, size)
    diagonal = np.empty(size)
    for i in range(size):
        left = abs(lower[i]) if i else 0.0
        right = abs(upper[i]) if i < size - 1 else 0.0
        diagonal[i] = left + right + rng.uniform(0.5, 2.0)
    lower[0] = np.nan
    upper[-1] = np.nan
    rhs = rng.uniform(-3.0, 3.0, size)
    matrix = _dense_matrix(lower, diagonal, upper, size)
    expected = np.linalg.solve(matrix, rhs)
    solution = np.empty(size)

    _solve_tridiagonal(lower, diagonal, upper, rhs, solution, size)

    np.testing.assert_allclose(solution, expected, rtol=5e-12, atol=5e-13)
    residual = np.linalg.norm(matrix @ solution - rhs, ord=np.inf)
    scale = (np.linalg.norm(matrix, ord=np.inf)
             * np.linalg.norm(solution, ord=np.inf)
             + np.linalg.norm(rhs, ord=np.inf))
    assert residual <= 100.0 * np.finfo(float).eps * scale


def test_tridiagonal_reuses_output_without_writing_past_size():
    lower = np.array([np.nan, 2.0, -1.0, 91.0, 92.0])
    diagonal = np.array([4.0, 5.0, 6.0, 93.0, 94.0])
    upper = np.array([1.0, 3.0, np.nan, 95.0, 96.0])
    matrix = _dense_matrix(lower, diagonal, upper, 3)
    solution = np.full(5, 1234567.25)
    inputs_before = tuple(array.copy() for array in (lower, diagonal, upper))

    first = np.array([1.0, -2.0, 3.0])
    _solve_tridiagonal(lower, diagonal, upper, matrix @ first, solution, 3)
    np.testing.assert_allclose(solution[:3], first, rtol=5e-12, atol=5e-13)
    np.testing.assert_array_equal(solution[3:], np.full(2, 1234567.25))

    second = np.array([-1.0, 4.0, 0.5])
    rhs = matrix @ second
    rhs_before = rhs.copy()
    _solve_tridiagonal(lower, diagonal, upper, rhs, solution, 3)
    np.testing.assert_allclose(solution[:3], second, rtol=5e-12, atol=5e-13)
    np.testing.assert_array_equal(solution[3:], np.full(2, 1234567.25))

    for array, snapshot in zip((lower, diagonal, upper), inputs_before):
        np.testing.assert_equal(array, snapshot)
    np.testing.assert_array_equal(rhs, rhs_before)
