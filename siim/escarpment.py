"""
Escarpment-specialized siim2d (thin convenience wrapper).

Rides on top of the standard ``siim2d.siim`` model, wiring two generic fastscape
forcing processes — ``WaveUplift`` and ``PlateauSurface``, which now live in
:mod:`siim.fastscape.forcing` (reusable by any fastscape model) — in through the
base class extension seams (``_default_params`` / ``_process_overrides`` /
``_forcing_input_vars``):

  - ``WaveUplift`` — a moving Gaussian uplift wave (replaces fastscape's
    ``BlockUplift``), for a retreating-escarpment / passing uplift-pulse setting.
  - ``PlateauSurface`` — an arctan-smoothed plateau initial topography
    (replaces ``InitialTopography``), for starting from a high plateau with a
    sharp escarpment edge.

Everything else (solvers, routing, plotting, analytical, save/load, channel
extraction, …) is inherited unchanged from ``siim2d.siim``, so improvements to
the core model flow through to the escarpment model automatically.

Usage::

    from siim.escarpment import siim_escarpment
    m = siim_escarpment({
        "uplift_type": "wave", "delta_h": 1500, "wave_velocity": 15e-3,
        "init_type": "plateau", "plateau_zo": 1500,
        "sliding_law": "coulomb", ...
    })
    m.run()
    m.plot.landscape()
"""

import numpy as np
from types import SimpleNamespace

from .siim2d import siim
from ._core.step import plateau_profile, plateau_surface, uplift_mask, wave_uplift


def plateau_topography(nx, ny, Lx, zo=1000.0, frac=0.8, w=10e3, dz=1.0):
    """Plateau initial surface for plain ``siim2d`` runs, with its own base level.

    Returns ``(topo, bl)``: the ``(ny, nx)`` surface — exactly 0 at the left
    edge, rising to ``zo`` across an escarpment centred at ``x = (1 - frac)*Lx``
    with transition width ``w`` (m), then dropping ``dz`` linearly to exactly
    ``zo - dz`` at the right edge — and the per-side base level
    ``{'right': zo - dz}`` to pass as the ``bl`` parameter. The right outlet's
    water line is then the plateau's own edge elevation, so every basin
    draining onto the plateau is graded to plateau height and base level does
    not incise the plateau from behind; the left outlet keeps the default
    datum. Same profile as ``siim_escarpment(init_type='plateau')``
    (:func:`siim._core.step.plateau_profile`); siim2d adds its own
    ``noise_amplitude`` (zeroed on fixed borders) on top. Usage::

        params['initial_topography'], params['bl'] = plateau_topography(
            nx, ny, Lx, zo=2000, frac=0.85, w=25e3, dz=1000)
    """
    x = np.linspace(0.0, Lx, nx)
    topo = np.tile(plateau_profile(x, zo, dz, frac, w), (ny, 1))
    return topo, {'right': zo - dz}


class siim_escarpment(siim):
    """siim2d specialized for escarpment problems.

    Accepts the full standard siim2d parameter set plus::

      uplift (uplift_type='wave'):
        uplift_type      'block' (default, standard BlockUplift) or 'wave'
        delta_h          target integrated uplift as the wave passes (m); required for 'wave'
        wave_width       1/e half-width of the wave (m)
        wave_velocity    wave propagation velocity (m/yr)
        x_escarpment     initial wave-center position relative to the left edge (m)
        wave_calibration calibration factor on the peak rate (1.0 = exact:
                         the wave deposits delta_h as it passes)
        U_inf            steady background uplift (m/yr)

      initial topography (init_type='plateau'):
        init_type        'sloped' (default, standard InitialTopography) or 'plateau'
        plateau_zo       plateau elevation (m); required for 'plateau'
        plateau_dz       slope across the plateau (m); seeds the divide
        plateau_frac     fraction of x occupied by the plateau
        plateau_w        escarpment transition width (m)

    The two switches are independent: you can mix e.g. a sloped start with a
    wave uplift, or a plateau start with block uplift.
    """

    def _default_params(self):
        d = super()._default_params()
        d.update({
            # uplift: 'block' (default) or 'wave' (moving-Gaussian WaveUplift)
            "uplift_type": "block",
            "delta_h": None,
            "wave_width": 200e3,
            "wave_velocity": 15e-3,
            "x_escarpment": 0.0,
            "wave_calibration": 1.0,
            "U_inf": 0.0,
            # initial topography: 'sloped' (default) or 'plateau' (PlateauSurface)
            "init_type": "sloped",
            "plateau_zo": None,
            "plateau_dz": 1.0,
            "plateau_frac": 0.8,
            "plateau_w": 10e3,
        })
        return d

    def set_and_check_parameters(self, user_params):
        # Base validates the merged (standard + escarpment) parameter set and
        # populates all the standard attributes (grid, time, climate, U, …).
        super().set_and_check_parameters(user_params)
        params = SimpleNamespace(**{**self._default_params(), **user_params})

        # --- uplift_type dispatch ---
        self.uplift_type = params.uplift_type
        if self.uplift_type not in ("block", "wave"):
            raise ValueError(f"uplift_type must be 'block' or 'wave', got {self.uplift_type!r}")
        if self.uplift_type == "wave":
            if params.delta_h is None:
                raise ValueError("uplift_type='wave' requires delta_h (target plateau gain, m)")
            if not np.isscalar(params.U):
                raise ValueError("uplift_type='wave' is incompatible with array-form U; "
                                 "pass scalar U or use uplift_type='block'")
            self.delta_h = float(params.delta_h)
            self.wave_width = float(params.wave_width)
            self.wave_velocity = float(params.wave_velocity)
            self.x_escarpment = float(params.x_escarpment)
            self.wave_calibration = float(params.wave_calibration)
            self.U_inf = float(params.U_inf)
            # Representative scalar uplift for the analytical reference: the
            # average rate the wave deposits over the run.
            self.U = self.U_inf + self.delta_h / self.T

        # --- init_type dispatch ---
        self.init_type = params.init_type
        if self.init_type not in ("sloped", "plateau"):
            raise ValueError(f"init_type must be 'sloped' or 'plateau', got {self.init_type!r}")
        if self.init_type == "plateau":
            if params.plateau_zo is None:
                raise ValueError("init_type='plateau' requires plateau_zo (plateau elevation, m)")
            if params.initial_topography is not None:
                raise ValueError("init_type='plateau' is incompatible with initial_topography "
                                 "array; pick one")
            self.plateau_zo = float(params.plateau_zo)
            self.plateau_dz = float(params.plateau_dz)
            self.plateau_frac = float(params.plateau_frac)
            self.plateau_w = float(params.plateau_w)

    def _process_overrides(self):
        # Adapter-only seam (xsimlab driver); lazy import keeps
        # `import siim.escarpment` stack-free (the in-house driver reaches the
        # same extracted step fns through the seams below instead).
        from .fastscape import PlateauSurface, WaveUplift
        overrides = super()._process_overrides()
        if self.init_type == "plateau":
            overrides["init_topography"] = PlateauSurface
        if self.uplift_type == "wave":
            overrides["uplift"] = WaveUplift
        return overrides

    def _forcing_input_vars(self):
        forcing = {}

        # initial topography
        if self.init_type == "plateau":
            forcing.update({
                "init_topography__plateau_zo": self.plateau_zo,
                "init_topography__plateau_dz": self.plateau_dz,
                "init_topography__plateau_frac": self.plateau_frac,
                "init_topography__plateau_w": self.plateau_w,
            })
        else:
            forcing["init_topography__elevation_init"] = self._make_initial_topo()
        if self.seed is not None:
            forcing["init_topography__seed"] = self.seed

        # uplift
        if self.uplift_type == "wave":
            forcing.update({
                "uplift__delta_h": self.delta_h,
                "uplift__wave_width": self.wave_width,
                "uplift__wave_velocity": self.wave_velocity,
                "uplift__x_escarpment": self.x_escarpment,
                "uplift__wave_calibration": self.wave_calibration,
                "uplift__U_inf": self.U_inf,
            })
        else:
            forcing["uplift__rate"] = self._make_uplift_field()

        return forcing

    # --- in-house driver seams (mirror _process_overrides/_forcing_input_vars:
    #     the same extracted step fns the WaveUplift/PlateauSurface shells call) ---

    def _driver_initial_surface(self):
        if self.init_type != "plateau":
            return super()._driver_initial_surface()
        shape = (self.grid_ny, self.grid_nx)
        x = np.linspace(0, self.Lx, self.grid_nx)
        y = np.linspace(0, self.Ly, self.grid_ny)
        return plateau_surface(
            x, y, shape, self.boundary_status, self.seed, self.noise_amplitude,
            self.plateau_zo, self.plateau_dz, self.plateau_frac, self.plateau_w)

    def _driver_uplift_fn(self):
        if self.uplift_type != "wave":
            return super()._driver_uplift_fn()
        shape = (self.grid_ny, self.grid_nx)
        x = np.linspace(0, self.Lx, self.grid_nx)
        y = np.linspace(0, self.Ly, self.grid_ny)
        mask = uplift_mask(self.boundary_status, shape)
        t = self.t

        def fn(k, dt):
            # step_start = t[k] (the xsimlab runtime arg the shell receives);
            # wave_uplift midpoint-samples at t[k] + dt/2 internally.
            return wave_uplift(x, y, shape, mask, dt, t[k], self.delta_h,
                               self.wave_width, self.wave_velocity,
                               self.x_escarpment, self.wave_calibration,
                               self.U_inf)
        return fn
