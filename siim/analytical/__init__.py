"""Analytical steady-state machinery for the coupled glacial-fluvial profile.

Lightweight by design: importing this subpackage needs numpy/scipy only (no
fastscape/xsimlab, no matplotlib until ``plot`` is called), so the theory
classes can be used for paper figures without the model stack.

Entry points::

    from siim.analytical import GeneralProfile          # general exponents
    from siim.analytical import MarginalCoulombProfile  # exact d*phi = 1 case
    from siim.analytical import SteadyStateProfile      # physical parameters

``GeneralProfile`` is the general steady state in steepness form (ks, cs,
zELA, L plus exponents); ``MarginalCoulombProfile`` is the closed-form
(arcosh) solution of the marginal-Coulomb special case and doubles as the
cross-check oracle for the general machinery; ``SteadyStateProfile`` (alias
``analytical_steady_state_solution``) maps physical model parameters onto
the same engine. ``RegimeMap`` evaluates the same closure over
(kappa = Ng/Nf, Y = zELA/zfo) grids for regime diagrams, including the
saddle-node bistability boundary. All share the ``Solution`` /
``AARResult`` result types, the shared closure solver, and the
``incomplete_beta`` kernel below.
"""
from .core import (AARResult, Solution, incomplete_beta,
                   incomplete_beta_compl)
from .profiles import GeneralProfile, MarginalCoulombProfile, sweep
from .regime import RegimeMap, SaddleNodeBoundary
from .steady_state import SteadyStateProfile, analytical_steady_state_solution

__all__ = [
    'AARResult',
    'GeneralProfile',
    'MarginalCoulombProfile',
    'RegimeMap',
    'SaddleNodeBoundary',
    'Solution',
    'SteadyStateProfile',
    'analytical_steady_state_solution',
    'incomplete_beta',
    'incomplete_beta_compl',
    'sweep',
]
