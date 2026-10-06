# Model assumptions and experimental numerical method

The nonlinear section-polar corrected VLM is implemented, but has not been
validated against independent aerodynamic data. All configuration values and
domain cutoffs are provisional. Dataset generation and ML training remain TODO.

## Geometry and reference conventions

Wings are symmetric, rigid, untwisted trapezoids with one named airfoil, zero
dihedral, and no fuselage or control surfaces. Full-wing planform area is S in
square metres, aspect ratio is b squared divided by S, taper is tip/root chord,
and sweep is quarter-chord sweep in degrees, positive aft. Coordinates use x
aft, y spanwise, and z up. Analytic dimensions and the AeroSandbox adapter are
implemented. Geometry IDs exclude operating conditions.

## Implemented nonlinear closure

This is a custom decambering-style boundary correction, not an established
validated decambering implementation. AeroSandbox supplies meshing and the
horseshoe-vortex influence kernel. The three-dimensional solve retains multiple
chordwise panels; it does not call AeroSandbox's nonlinear lifting-line solver.

1. Create a flat thin-surface mesh with `add_camber=False`. Airfoil camber and
   viscous lift enter through section polars instead of being counted twice.
   `spanwise_panels` is the count per semispan; `chordwise_panels` is per strip.
   Bound legs sit at panel quarter chord, collocation points at three-quarter
   chord. The prescribed wake is parallel to geometry x.
2. Assemble the normal-velocity influence matrix A from AeroSandbox horseshoes.
   Solve `A gamma = -V_inf_z - V_inf * delta_strip`, where delta is a virtual
   strip incidence in radians, applied to every chordwise panel in that strip.
   The physical geometry stays unchanged. Start with delta equal to zero.
3. Sum panel circulations within each strip. Place equivalent horseshoes at the
   full-wing quarter chord to estimate section inflow. This collapsed estimate
   avoids counting within-section chordwise bound-vortex interactions as the
   section's induced angle. It is a modeling approximation requiring validation.
4. Compute effective angle as atan2 of local z and x velocity and section speed
   from those two components. Ignore section crossflow. Section CL from the
   vortex circulation is `2 * Gamma_strip / (V_section * chord)`.
5. Query cached NeuralFoil CL/CD/CM at effective angle. Tables use freestream
   Reynolds number `rho * V_inf * local_chord / mu` and freestream Mach, frozen
   during iteration. Induced speed changes affect the lift closure and force
   normalization but do not update table Reynolds/Mach. This attached-flow
   approximation is explicit and should be tested before broad dataset use.
6. Update delta by `relaxation * (CL_polar - CL_vortex) / (2*pi)`. The thin-airfoil
   slope is an iteration preconditioner only; NeuralFoil supplies the target
   nonlinear CL. Stop only when both maximum absolute section CL mismatch and
   maximum panel circulation change normalized by `V_inf * max(chord)` meet
   `residual_tolerance`. Reject virtual corrections larger than 30 degrees.
7. Integrate panel forces using `rho * gamma * cross(V_local, bound_leg)` with
   velocities from the full chordwise lattice. Integrate section profile force
   using local chordwise dynamic pressure, strip area, and polar CD, directed
   along section inflow. Project both into freestream lift and drag axes and
   divide by freestream dynamic pressure times S. Return total CD and separate
   induced/profile contributions. Polar CM is cached but not integrated because
   this interface predicts only lift and drag.

The nonlinear coupling feeds CL back into circulation. Adding CD to a single
linear VLM result would not reproduce this method.

## Polar provenance and failure policy

Polars are computed through `Airfoil.get_aero_from_neuralfoil` with explicit
transition settings and the 360-degree extension disabled. Cache identities
include the request, airfoil-coordinate hash, and AeroSandbox/NeuralFoil/NumPy
versions. JSON writes are atomic. Piecewise-linear interpolation rejects
extrapolation. Confidence uses the lower bracket confidence; low-confidence
queries are rejected. Cached tables retain all CL/CD/CM/confidence values.

Invalid inputs, out-of-domain queries, flow reversal, and runtime/numerical/cache
errors return failed cases with error messages, runtime and null CL/CD. Exhausted
iteration limits return unconverged cases with final residual/change diagnostics
and null CL/CD. Missing dependencies raise an import error. Runtime includes
geometry, matrix construction, polar generation/cache I/O, and all iterations.
Data generation remains TODO and must save every attempt instead of filtering
failures silently.

## Domain and validation limits

The current implementation requires beta=0, Mach below 0.3, and absolute sweep
at most 30 degrees. Swept geometry enters the vortex lattice, but section polars
use chordwise strips without sweep-normal Reynolds/Mach or empirical sweep
corrections. Those limits are safeguards, not demonstrated validity ranges.

The model assumes steady potential flow, a prescribed wake, and rigid surfaces.
Section polars cannot establish correct three-dimensional separation, stall,
tip flow or interference. Numerical convergence alone is not evidence of physical
accuracy, particularly near stall or for sweep. Under-relaxed fixed-point iteration
may fail or find different branches at high angles. No post-stall accuracy is
claimed. Dependency versions checked: AeroSandbox 4.2.10, NeuralFoil 0.3.3.

Tests check actual NeuralFoil cache reuse, interpolation/domain policy, zero-lift
symmetry, positive lift/drag in an attached-flow example, drag decomposition,
section-polar lift feedback, and failure/iteration-limit reporting. Synthetic
polars are used only as labeled in-memory unit-test fixtures. Remaining work:
independent benchmark validation, quantitative mesh/tolerance studies, sensitivity
to the equivalent-strip inflow approximation, sweep treatment, and stall/domain
selection before generating a production training dataset.

## Sources

- [AeroSandbox VLM source](https://aerosandbox.readthedocs.io/en/develop/_modules/aerosandbox/aerodynamics/aero_3D/vortex_lattice_method.html)
- [NeuralFoil project](https://github.com/peterdsharpe/NeuralFoil)
- [NeuralFoil paper](https://arxiv.org/abs/2503.16323)
