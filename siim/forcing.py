"""Time-series builders for time-varying model forcing.

Each function builds the run's time vector internally from the models' clock,
``linspace(to, T, nt)`` (``T`` the final time in years, ``to`` the initial time,
default 0, ``nt`` the number of steps), and returns ``(t, series)``: the time
vector (handy for plotting) and the length-``nt`` forcing array that drops
straight into a model parameter dict, e.g.::

    from siim.forcing import ela_sawtooth
    t, zELA = ela_sawtooth(params['T'], params['nt'], ela_high=2400, ela_low=1400,
                           to=params.get('to', 0.0))
    params['zELA'] = zELA

Both ``siim.siim1d`` and ``siim.siim2d`` accept a scalar or a length-``nt``
array for ``zELA``, ``U``, and ``P``. numpy-only (no model import), so this
stays a cheap import.
"""
import numpy as np


def ela_sawtooth(T, nt, ela_high=1500, ela_low=300, period=100e3,
                 buildup_frac=0.88, to=0.0):
    """Asymmetric sawtooth ELA(t): slow glacial buildup, fast termination.

    Builds ``t = linspace(to, T, nt)`` internally. Linear ramp down from
    ``ela_high`` to ``ela_low`` over ``buildup_frac`` of each ``period`` (the
    slow drop into a glacial), then linear ramp back up over the remaining
    ``1 - buildup_frac`` (the fast termination).

    Parameters
    ----------
    T : float
        Final model time in years, as the models read it (the time vector
        spans ``to`` to ``T``).
    nt : int
        Number of time steps (length of the returned array).
    to : float, optional
        Initial model time in years (default 0), the models' ``to``.
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
    t = np.linspace(to, T, nt)
    phase = ((t - to) % period) / period      # a cycle starts with the run
    ela = np.where(
        phase < buildup_frac,
        ela_high - (ela_high - ela_low) * phase / buildup_frac,    # slow drop
        ela_low + (ela_high - ela_low) * (phase - buildup_frac) / (1 - buildup_frac),  # fast rise
    )
    return t, ela


def uplift_step(T, nt, U_init, U_final, step_frac=0.5, to=0.0):
    """Step change in uplift rate partway through the run.

    Builds ``t = linspace(to, T, nt)`` internally and returns ``U_init`` before
    the step and ``U_final`` at and after it. The step falls at
    ``to + step_frac * (T - to)``, a fraction ``step_frac`` of the run.

    Parameters
    ----------
    T : float
        Final model time in years, as the models read it (the time vector
        spans ``to`` to ``T``).
    nt : int
        Number of time steps (length of the returned array).
    to : float, optional
        Initial model time in years (default 0), the models' ``to``.
    U_init, U_final : float
        Uplift rate (m/yr) before and after the step.
    step_frac : float
        Where the step lands, as a fraction of the run time (0 to 1).

    Returns
    -------
    (t, U) : tuple of ndarray
        The time vector and the uplift-rate series, each length ``nt``.
    """
    t = np.linspace(to, T, nt)
    U = np.where(t < to + step_frac * (T - to), U_init, U_final)
    return t, U


def interp_forcing(T, nt, times, values, left=None, right=None, to=0.0):
    """Piecewise-linear forcing series from coarse ``(times, values)`` nodes.

    Interpolates ``values`` (defined at ``times``, in model-time years)
    onto ``t = linspace(to, T, nt)``. A generic table-to-series builder, reusable
    for any scalar forcing — ``P``, ``zELA``, or ``U``::

        from siim.forcing import interp_forcing
        # falling precipitation over the run (m/yr)
        t, P = interp_forcing(params['T'], params['nt'],
                              times=[0, 1.5e6, 3e6], values=[2.0, 1.2, 0.65])
        params['P'] = P

    ``times`` are model-time years (``to`` → ``T``); map geological time (e.g. Ma)
    onto that axis before calling, the same way you would for ``zELA``.

    Parameters
    ----------
    T : float
        Final model time in years, as the models read it (the time vector
        spans ``to`` to ``T``).
    nt : int
        Number of time steps (length of the returned array).
    to : float, optional
        Initial model time in years (default 0), the models' ``to``.
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
    t = np.linspace(to, T, nt)
    return t, np.interp(t, times, values, left=left, right=right)
