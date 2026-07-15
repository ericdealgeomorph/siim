"""Time-series builders for time-varying model forcing.

Each function builds the run's time vector internally from ``T`` (run time,
years) and ``nt`` (number of steps) and returns ``(t, series)``: the time
vector (handy for plotting) and the length-``nt`` forcing array that drops
straight into a model parameter dict, e.g.::

    from siim.forcing import ela_sawtooth
    t, zELA = ela_sawtooth(params['T'], params['nt'], ela_high=2400, ela_low=1400)
    params['zELA'] = zELA

Both ``siim.siim1d`` and ``siim.siim2d`` accept a scalar or a length-``nt``
array for ``zELA``, ``U``, and ``P``. numpy-only (no model import), so this
stays a cheap import.
"""
import numpy as np


def ela_sawtooth(T, nt, ela_high=1500, ela_low=300, period=100e3,
                 buildup_frac=0.88):
    """Asymmetric sawtooth ELA(t): slow glacial buildup, fast termination.

    Builds ``t = linspace(0, T, nt)`` internally. Linear ramp down from
    ``ela_high`` to ``ela_low`` over ``buildup_frac`` of each ``period`` (the
    slow drop into a glacial), then linear ramp back up over the remaining
    ``1 - buildup_frac`` (the fast termination).

    Parameters
    ----------
    T : float
        Total run time in years (the time vector spans 0 to T).
    nt : int
        Number of time steps (length of the returned array).
    ela_high, ela_low : float
        ELA (m) at the start of buildup and at termination onset.
    period : float
        Cycle length in years.
    buildup_frac : float
        Fraction of the period spent in the slow buildup (0 < frac < 1).

    Returns
    -------
    (t, ela) : tuple of ndarray
        The time vector and the ELA(t) series, each length ``nt``.
    """
    t = np.linspace(0, T, nt)
    phase = (t % period) / period
    ela = np.where(
        phase < buildup_frac,
        ela_high - (ela_high - ela_low) * phase / buildup_frac,    # slow drop
        ela_low + (ela_high - ela_low) * (phase - buildup_frac) / (1 - buildup_frac),  # fast rise
    )
    return t, ela


def uplift_step(T, nt, U_init, U_final, step_frac=0.5):
    """Step change in uplift rate partway through the run.

    Builds ``t = linspace(0, T, nt)`` internally and returns ``U_init`` before
    the step and ``U_final`` at and after it. The step falls at ``step_frac * T``
    (so ``step_frac`` is a fraction of the run time).

    Parameters
    ----------
    T : float
        Total run time in years (the time vector spans 0 to T).
    nt : int
        Number of time steps (length of the returned array).
    U_init, U_final : float
        Uplift rate (m/yr) before and after the step.
    step_frac : float
        Where the step lands, as a fraction of the run time (0 to 1).

    Returns
    -------
    (t, U) : tuple of ndarray
        The time vector and the uplift-rate series, each length ``nt``.
    """
    t = np.linspace(0, T, nt)
    U = np.where(t < step_frac * T, U_init, U_final)
    return t, U


def interp_forcing(T, nt, times, values, left=None, right=None):
    """Piecewise-linear forcing series from coarse ``(times, values)`` nodes.

    Interpolates ``values`` (defined at ``times``, in run-time years from 0)
    onto ``t = linspace(0, T, nt)``. A generic table-to-series builder, reusable
    for any scalar forcing — ``P``, ``zELA``, or ``U``::

        from siim.forcing import interp_forcing
        # falling precipitation over the run (m/yr)
        t, P = interp_forcing(params['T'], params['nt'],
                              times=[0, 1.5e6, 3e6], values=[2.0, 1.2, 0.65])
        params['P'] = P

    ``times`` are model-time years (0 → ``T``); map geological time (e.g. Ma)
    onto that axis before calling, the same way you would for ``zELA``.

    Parameters
    ----------
    T : float
        Total run time in years (the time vector spans 0 to T).
    nt : int
        Number of time steps (length of the returned array).
    times, values : array_like
        Node positions (run-time years) and the forcing value at each node.
        ``times`` must be increasing (``np.interp`` requirement).
    left, right : float, optional
        Value returned below ``times[0]`` / above ``times[-1]``. Defaults to the
        nearest endpoint (flat extrapolation, ``np.interp`` default).

    Returns
    -------
    (t, series) : tuple of ndarray
        The time vector and the interpolated forcing series, each length ``nt``.
    """
    t = np.linspace(0, T, nt)
    return t, np.interp(t, times, values, left=left, right=right)
