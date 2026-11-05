"""Approximate small-angle VLM–NeuralFoil coupling for unswept, symmetric airfoils."""

from time import perf_counter
import numpy as np
from scipy.linalg import lu_factor, lu_solve
from scipy.optimize import root

from .airfoil import evaluate_airfoil
from .geometry import to_aerosandbox_wing


def influence_matrix(analysis):
    """Normal induced velocity at each collocation point per unit circulation."""
    original = analysis.vortex_strengths.copy()
    n_panels = len(original)
    matrix = np.empty((n_panels, n_panels))

    try:
        for j in range(n_panels):
            # Measure one vortex's effect on every panel.
            unit_gamma = np.zeros(n_panels)
            unit_gamma[j] = 1.0
            analysis.vortex_strengths = unit_gamma
            induced = analysis.get_induced_velocity_at_points(analysis.collocation_points)

            matrix[:, j] = np.sum(induced * analysis.normal_directions, axis=1)
    finally:
        analysis.vortex_strengths = original

    return matrix


def panel_induced_drag(analysis, circulation):
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


def match_lift(
    *, airfoil, gamma_initial, gamma_response, strip, chord,
    velocity, Re, mach, model_size, lift_tolerance, xtol, max_evaluations,
):
    """Solve all station lift mismatches using a numerical Jacobian."""
    n_strips = len(chord)

    def evaluate(correction):
        gamma = gamma_initial + gamma_response @ correction
        gamma_strip = np.bincount(strip, weights=gamma, minlength=n_strips)
        cl_vlm = 2 * gamma_strip / (velocity * chord)

        # Infer effective incidence using a 2*pi-per-radian lift slope.
        alpha_eff = np.degrees(cl_vlm / (2 * np.pi) - correction)
        section_aero = evaluate_airfoil(airfoil, alpha_eff, Re, mach, model_size)

        return {
            "error": section_aero["CL"] - cl_vlm,
            "gamma": gamma,
            "cl_vlm": cl_vlm,
            "alpha_deg": alpha_eff,
            "section_aero": section_aero,
        }

    # Find corrections that make NeuralFoil and VLM lift agree at every strip.
    solution = root(
        fun=lambda correction: evaluate(correction)["error"],
        x0=np.zeros(n_strips),
        method="hybr",
        jac=False,
        options={"xtol": xtol, "maxfev": max_evaluations},
    )
    state = evaluate(solution.x)

    max_error = float(np.max(np.abs(state["error"])))

    # Check lift agreement separately from SciPy's stopping criteria.
    state.update(
        correction_rad=solution.x,
        max_lift_error=max_error,
        lift_converged=bool(max_error <= lift_tolerance),
        residual_evaluations=int(solution.nfev),
        solver_success=bool(solution.success),
        solver_message=str(solution.message),
    )

    return state


def solve_vlm_neuralfoil(
    geometry,
    *,
    velocity=100.0,
    alpha=5.0,
    spanwise_resolution=20,
    chordwise_resolution=4,
    model_size="large",
    lift_tolerance=1e-6,
    xtol=1e-8,
    max_evaluations=2000,
):
    """Return wing CL, total CD, and station data for velocity in m/s and alpha in degrees."""

    import aerosandbox as asb

    total_start = perf_counter()
    wing = to_aerosandbox_wing(geometry)
    airfoil = wing.xsecs[0].airfoil

    airplane = asb.Airplane(
        name="VLM–NeuralFoil",
        wings=[wing],
        s_ref=geometry.parameters.area_m2,
        b_ref=geometry.span_m,
        c_ref=wing.mean_aerodynamic_chord(),
    )

    analysis = asb.VortexLatticeMethod(
        airplane=airplane,
        op_point=asb.OperatingPoint(velocity=velocity, alpha=alpha, beta=0.0),
        spanwise_resolution=spanwise_resolution,
        chordwise_resolution=chordwise_resolution,
    )

    # Solve the initial VLM loading.
    vlm_start = perf_counter()
    initial_aero = analysis.run()
    vlm_runtime_s = perf_counter() - vlm_start
    coupling_start = perf_counter()

    # Group chordwise panels into spanwise strips.
    y, strip = np.unique(
        np.round(analysis.vortex_centers[:, 1], 12), return_inverse=True
    )
    n_strips = len(y)
    strip_area = np.bincount(strip, weights=analysis.areas)

    chord = np.interp(
        np.abs(y),
        [xsec.xyz_le[1] for xsec in wing.xsecs],
        [xsec.chord for xsec in wing.xsecs],
    )

    # Calculate local Reynolds numbers and freestream Mach number.
    atmosphere = analysis.op_point.atmosphere
    kinematic_viscosity = float(atmosphere.kinematic_viscosity())
    Re = velocity * chord / kinematic_viscosity
    reference_reynolds = velocity * float(airplane.c_ref) / kinematic_viscosity
    mach = velocity / float(atmosphere.speed_of_sound())

    gamma_initial = analysis.vortex_strengths.copy()

    # Calculate circulation changes per radian of strip incidence correction.
    matrix = influence_matrix(analysis)
    panel_to_strip = np.eye(n_strips)[strip]
    rhs_per_radian = (
        -velocity * analysis.normal_directions[:, 2, None] * panel_to_strip
    )
    gamma_response = lu_solve(lu_factor(matrix), rhs_per_radian)

    state = match_lift(
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

    # Recompute drag from the corrected circulation.
    panel_drag = panel_induced_drag(analysis, state["gamma"])
    induced_cd = np.bincount(strip, weights=panel_drag) * airplane.s_ref / strip_area
    profile_drag = float(np.sum(section_aero["CD"] * strip_area) / airplane.s_ref)
    induced_drag = float(np.sum(panel_drag))

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

    return {
        "geometry_id": geometry.geometry_id(),
        "reynolds": reference_reynolds,

        "CL": float(np.sum(state["cl_vlm"] * strip_area) / airplane.s_ref),
        "CD": profile_drag + induced_drag,
        "drag_components": {"profile": profile_drag, "induced": induced_drag},

        "stations": stations,
        "circulation": state["gamma"],

        "max_lift_error": state["max_lift_error"],
        "lift_converged": state["lift_converged"],
        "residual_evaluations": state["residual_evaluations"],
        "solver_success": state["solver_success"],
        "solver_message": state["solver_message"],

        "initial_vlm": {"CL": float(initial_aero["CL"]), "CDi": float(initial_aero["CD"])},

        "vlm_runtime_s": vlm_runtime_s,
        "coupling_runtime_s": perf_counter() - coupling_start,
        "total_runtime_s": perf_counter() - total_start,
    }
