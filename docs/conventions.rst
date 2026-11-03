Conventions and model scope
===========================

* Supply one-dimensional, equally sized NumPy arrays. Use finite values,
  positive measurement errors and sorted times for data processing/searches.
  Remove invalid observations explicitly before fitting.
* Times, epochs, periods and exposure durations in the examples are in **days**.
  Use one consistent astronomical time system (for example BJD_TDB); BRUCE
  does not convert timestamps or infer a time system.
* ``t_zero`` is the primary mid-transit epoch, not a periastron epoch.
  ``incl`` and ``w`` are in **radians**, and ``0 <= e < 1`` describes an ellipse.
* ``radius_1 = R_star/a`` and ``k = R_companion/R_star``. For circular motion,
  ``b = cos(incl)/radius_1``. This relation is not the eccentric-transit
  impact-parameter formula.
* Relative out-of-transit flux is unity. ``light_3`` is contaminating light
  divided by target light. The default power-2 law is
  ``I(mu)/I(1) = 1 - c * (1 - mu**alpha)``.
* ``cadence=0`` evaluates instantaneous flux. Positive ``cadence`` averages
  equally spaced samples across the full exposure and requires an **odd**
  ``noversample >= 3`` (default 11). Check convergence for your cadence and
  ingress timescale.
* RV amplitudes ``K1``, ``K2`` and systemic velocity ``V0`` use any consistent
  velocity unit. Outputs and errors use that same unit.

Likelihood signs
----------------

``binarystar.lc``, ``rv1`` and ``rv2`` return model arrays when no observations
are supplied. With observations and errors they return a scalar **negative
log likelihood**, suitable for minimisation. For posterior sampling, negate it
before adding a log prior. ``offset=1`` includes the variance-dependent log
term; ``offset=0`` omits it. The additive Gaussian constant is omitted.
Use ``offset=1`` when varying jitter.

In contrast, ``sampler.loglike`` returns **log likelihood** (larger is better),
and template matching returns **2 times the log-likelihood improvement** over
the supplied baseline (larger is better). ``sampler.loglike`` discards non-finite
pointwise contributions; validate data and models yourself rather than relying
on this historical masking policy.

Limits and validation
---------------------

The transit implementation is an analytical approximation, not a complete
binary-star surface-integration model. Do not assume ellc's full physical
coverage. In the manuscript's one matched-law circular example, the maximum
difference from very-fine ellc integration is about 74 ppm; this is not a
global accuracy bound. Validate accuracy for your geometry and scientific
requirements, particularly for large companions and grazing transits.
The benchmark and comparison scripts live under ``paper/``.
