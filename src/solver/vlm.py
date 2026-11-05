"""Small-angle VLM–NeuralFoil lift matching for unswept, symmetric-airfoil wings.

Uses a 2*pi-per-radian section lift slope to infer effective incidence.
This is an approximate incidence-correction model, not a validated stall model.
Only zero quarter-chord sweep is supported. Re and Mach use freestream speed.
"""

from time import perf_counter
import math
from typing import Any

import numpy as np
from scipy.linalg import lu_factor, lu_solve
from scipy.optimize import root

from .airfoil import evaluate_airfoil
from .geometry import WingGeometry, to_aerosandbox_wing


# Panel influence: calculate how each vortex changes flow across the wing.
def _influence_matrix(analysis: Any) -> np.ndarray:
    """Normal induced velocity at each collocation point per unit circulation."""
    # Save the baseline circulation and prepare the panel influence matrix.
    original = analysis.vortex_strengths.copy()
    n_panels = len(original)
    matrix = np.empty((n_panels, n_panels))

    # Measure the normal flow produced by each vortex individually.
    try:
        for j in range(n_panels):
            # Activate one unit-strength vortex to measure its effect on all panels.
            unit_gamma = np.zeros(n_panels)
            unit_gamma[j] = 1.0
            analysis.vortex_strengths = unit_gamma
            induced = analysis.get_induced_velocity_at_points(analysis.collocation_points)

            # Keep the induced velocity component normal to the wing surface.
            matrix[:, j] = np.sum(induced * analysis.normal_directions, axis=1)
    finally:
        # Restore the original wing circulation after measuring vortex influences.
        analysis.vortex_strengths = original

    return matrix


def _panel_induced_drag(analysis: Any, circulation: np.ndarray) -> np.ndarray:
    """Recompute panel drag coefficients using AeroSandbox's near-field forces."""
    analysis.vortex_strengths = circulation.copy()
    local_velocity = analysis.get_velocity_at_points(analysis.vortex_centers)
    bound_leg = analysis.right_vortex_vertices - analysis.left_vortex_vertices
    forces = (
        float(analysis.op_point.atmosphere.density())
        * np.cross(local_velocity, bound_leg)
        * circulation[:, None]
    )
    force_x_wind, _, _ = analysis.op_point.convert_axes(
        forces[:, 0], forces[:, 1], forces[:, 2],
        from_axes="geometry", to_axes="wind",
    )
    reference_force = float(analysis.op_point.dynamic_pressure()) * analysis.airplane.s_ref
    return -np.asarray(force_x_wind) / reference_force


# Lift coupling: find incidence corrections that make VLM and airfoil lift agree.
def _match_lift(
    *,
    airfoil: Any,
    gamma_initial: np.ndarray,
    gamma_response: np.ndarray,
    strip: np.ndarray,
    chord: np.ndarray,
    velocity: float,
    Re: np.ndarray,
    mach: float,
    model_size: str,
    lift_tolerance: float,
    xtol: float,
    max_evaluations: int,
) -> dict[str, Any]:
    """Solve all station lift mismatches using a numerical Jacobian."""
    n_strips = len(chord)

    # Calculate section loading and airfoil lift for a trial set of corrections.
    def evaluate(correction: np.ndarray) -> dict[str, Any]:
        # Update panel circulation for the trial incidence correction at each strip.
        gamma = gamma_initial + gamma_response @ correction

        # Sum chordwise vortex strengths to get circulation at each span station.
        gamma_strip = np.bincount(strip, weights=gamma, minlength=n_strips)

        # Convert circulation to section lift coefficient using Kutta-Joukowski.
        cl_vlm = 2 * gamma_strip / (velocity * chord)

        # Estimate effective airfoil incidence using the assumed 2*pi lift slope.
        alpha_eff = np.degrees(cl_vlm / (2 * np.pi) - correction)
        section_aero = evaluate_airfoil(airfoil, alpha_eff, Re, mach, model_size)

        # Compare airfoil and VLM lift at every span station.
        return {
            "error": section_aero["CL"] - cl_vlm,
            "gamma": gamma,
            "cl_vlm": cl_vlm,
            "alpha_deg": alpha_eff,
            "section_aero": section_aero,
        }

    # Adjust all strip incidences together until their section lift errors vanish.
    solution = root(
        fun=lambda correction: evaluate(correction)["error"],
        x0=np.zeros(n_strips),
        method="hybr",
        jac=False,
        options={"xtol": xtol, "maxfev": max_evaluations},
    )
    state = evaluate(solution.x)

    # Judge convergence by the largest section lift mismatch across the wing.
    max_error = float(np.max(np.abs(state["error"])))
    if not math.isfinite(max_error) or max_error > lift_tolerance:
        raise RuntimeError(
            f"VLM–NeuralFoil coupling failed: {solution.message}; "
            f"maximum lift mismatch = {max_error:.6g}"
        )

    # Record convergence details; small incidence changes alone do not prove lift agreement.
    state.update(
        correction_rad=solution.x,
        max_lift_error=max_error,
        residual_evaluations=int(solution.nfev),
        solver_success=bool(solution.success),
        solver_message=str(solution.message),
    )

    return state


# Coupled wing solve: build the wing, correct its loading, and collect results.
def solve_vlm_neuralfoil(
    geometry: WingGeometry,
    *,
    velocity: float = 100.0,
    alpha: float = 5.0,
    spanwise_resolution: int = 20,
    chordwise_resolution: int = 4,
    model_size: str = "large",
    lift_tolerance: float = 1e-6,
    xtol: float = 1e-8,
    max_evaluations: int = 2000,
) -> dict[str, Any]:
    """Return approximate wing CL, total CD, and station data.

    alpha is in degrees; velocity is in m/s. Station corrections are radians.
    Reynolds number is calculated from velocity, local chord, and the
    atmosphere's kinematic viscosity. The returned reference Re uses the MAC.
    Includes both halves of the wing, ordered from negative to positive y.
    Geometric twist is included in the VLM mesh and its initial circulation.
    CD combines section profile drag and near-field induced drag recomputed
    from the final corrected circulation. It represents this wing only.
    """

    # Reject invalid flow conditions and wing shapes outside this model's assumptions.
    if not math.isfinite(velocity) or velocity <= 0:
        raise ValueError("velocity must be finite and positive")
    if not math.isfinite(alpha):
        raise ValueError("alpha must be finite")
    if abs(geometry.parameters.sweep_deg) > 1e-12:
        raise ValueError("This coupling currently requires zero quarter-chord sweep")

    # Check that panel counts and the solver evaluation limit are positive integers.
    for name, value in (
        ("spanwise_resolution", spanwise_resolution),
        ("chordwise_resolution", chordwise_resolution),
        ("max_evaluations", max_evaluations),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")

    # Check the allowed lift mismatch and incidence convergence tolerance.
    for name, value in (("lift_tolerance", lift_tolerance), ("xtol", xtol)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")

    # Load the aerodynamic tools and build the wing and its airfoil.
    import aerosandbox as asb

    total_start = perf_counter()
    wing = to_aerosandbox_wing(geometry)
    airfoil = wing.xsecs[0].airfoil

    # The assumed lift slope relation requires an airfoil with no camber.
    camber = np.asarray(airfoil.local_camber(), dtype=float)
    if not np.all(np.isfinite(camber)) or np.max(np.abs(camber)) > 1e-5:
        raise ValueError("This effective-incidence model requires a symmetric airfoil")

    # Use the full wing area, span, and mean aerodynamic chord as reference dimensions.
    airplane = asb.Airplane(
        name="VLM–NeuralFoil",
        wings=[wing],
        s_ref=geometry.parameters.area_m2,
        b_ref=geometry.span_m,
        c_ref=wing.mean_aerodynamic_chord(),
    )

    # Divide the wing into spanwise and chordwise vortex panels at the specified flow state.
    analysis = asb.VortexLatticeMethod(
        airplane=airplane,
        op_point=asb.OperatingPoint(velocity=velocity, alpha=alpha, beta=0.0),
        spanwise_resolution=spanwise_resolution,
        chordwise_resolution=chordwise_resolution,
    )

    # Solve the initial inviscid wing loading before applying airfoil lift corrections.
    vlm_start = perf_counter()
    initial_aero = analysis.run()
    vlm_runtime_s = perf_counter() - vlm_start
    coupling_start = perf_counter()

    # Group panels at the same spanwise location into a single wing strip.
    y, strip = np.unique(
        np.round(analysis.vortex_centers[:, 1], 12), return_inverse=True
    )
    n_strips = len(y)
    strip_area = np.bincount(strip, weights=analysis.areas)

    # Find local chord from the trapezoidal planform on either half of the wing.
    chord = np.interp(
        np.abs(y),
        [xsec.xyz_le[1] for xsec in wing.xsecs],
        [xsec.chord for xsec in wing.xsecs],
    )

    # Calculate Reynolds number from the flow speed, chord, and atmospheric viscosity.
    atmosphere = analysis.op_point.atmosphere
    kinematic_viscosity = float(atmosphere.kinematic_viscosity())
    Re = velocity * chord / kinematic_viscosity
    reference_reynolds = velocity * float(airplane.c_ref) / kinematic_viscosity
    mach = velocity / float(atmosphere.speed_of_sound())

    # Save the baseline circulation before applying section lift corrections.
    gamma_initial = analysis.vortex_strengths.copy()

    # Relate every panel's circulation to the normal flow induced at all panels.
    matrix = _influence_matrix(analysis)
    panel_to_strip = np.eye(n_strips)[strip]

    # Find the change in normal flow required per radian of strip incidence correction.
    rhs_per_radian = (
        -velocity * analysis.normal_directions[:, 2, None] * panel_to_strip
    )

    # Solve the panel flow equations once for circulation response to strip corrections.
    gamma_response = lu_solve(lu_factor(matrix), rhs_per_radian)

    # Match VLM section lift to NeuralFoil lift at every span station.
    state = _match_lift(
        airfoil=airfoil,
        gamma_initial=gamma_initial,
        gamma_response=gamma_response,
        strip=strip,
        chord=chord,
        velocity=velocity,
        Re=Re,
        mach=mach,
        model_size=model_size,
        lift_tolerance=lift_tolerance,
        xtol=xtol,
        max_evaluations=max_evaluations,
    )
    section_aero = state["section_aero"]

    # Recompute forces with the corrected loading; running VLM again would reset it.
    panel_drag = _panel_induced_drag(analysis, state["gamma"])
    induced_cd = np.bincount(strip, weights=panel_drag) * airplane.s_ref / strip_area
    profile_drag = float(np.sum(section_aero["CD"] * strip_area) / airplane.s_ref)
    induced_drag = float(np.sum(panel_drag))

    # Collect the local flow conditions and airfoil coefficients along the full span.
    stations = [
        {
            "y_m": float(y[i]),
            "chord_m": float(chord[i]),
            "twist_deg": float(geometry.parameters.tip_twist_deg * 2 * abs(y[i]) / geometry.span_m),
            "area_m2": float(strip_area[i]),
            "Re": float(Re[i]),
            "mach": float(mach),
            "alpha_deg": float(state["alpha_deg"][i]),
            "correction_rad": float(state["correction_rad"][i]),
            "cl_vlm": float(state["cl_vlm"][i]),
            "cl": float(section_aero["CL"][i]),
            "cd": float(section_aero["CD"][i] + induced_cd[i]),
            "cd_2d": float(section_aero["CD"][i]),
            "cd_induced": float(induced_cd[i]),
            "cm": float(section_aero["CM"][i]),
            "lift_error": float(state["error"][i]),
        }
        for i in range(n_strips)
    ]

    # Assemble wing coefficients, spanwise loading, convergence details, and runtimes.
    return {
        "geometry_id": geometry.geometry_id,
        "reynolds": reference_reynolds,

        # Combine profile and corrected induced drag on the same reference area.
        "CL": float(np.sum(state["cl_vlm"] * strip_area) / airplane.s_ref),
        "CD": profile_drag + induced_drag,
        "drag_components": {"profile": profile_drag, "induced": induced_drag},

        # Include the corrected spanwise loading and panel circulation.
        "stations": stations,
        "circulation": state["gamma"],

        # Report how closely the VLM and airfoil section lift agree.
        "max_lift_error": state["max_lift_error"],
        "residual_evaluations": state["residual_evaluations"],
        "solver_success": state["solver_success"],
        "solver_message": state["solver_message"],

        # Keep the original inviscid solution for comparison only.
        "initial_vlm": {"CL": float(initial_aero["CL"]), "CDi": float(initial_aero["CD"])},

        # Report time spent on the initial VLM, coupling, and complete solve.
        "vlm_runtime_s": vlm_runtime_s,
        "coupling_runtime_s": perf_counter() - coupling_start,
        "total_runtime_s": perf_counter() - total_start,
    }
