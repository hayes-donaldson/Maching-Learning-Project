import numpy as np
import pytest
import aerosandbox as asb

from solver.geometry import WingParameters, build_wing_geometry, to_aerosandbox_wing
from solver.vlm import _panel_induced_drag, solve_vlm_neuralfoil


def make_geometry(twist=0.0):
    return build_wing_geometry(WingParameters(8.0, 1.7, 0.5, 0.0, "naca0012", twist))


def make_analysis():
    geometry = make_geometry()
    wing = to_aerosandbox_wing(geometry)
    return asb.VortexLatticeMethod(
        airplane=asb.Airplane(
            wings=[wing], s_ref=geometry.parameters.area_m2,
            b_ref=geometry.span_m, c_ref=wing.mean_aerodynamic_chord(),
        ),
        op_point=asb.OperatingPoint(velocity=50.0, alpha=5.0),
        spanwise_resolution=6, chordwise_resolution=3,
    )


def test_recomputed_drag_matches_aerosandbox():
    analysis = make_analysis()
    original = analysis.run()
    recomputed = _panel_induced_drag(analysis, analysis.vortex_strengths.copy())
    np.testing.assert_allclose(recomputed.sum(), original["CD"], rtol=1e-12, atol=1e-14)


def test_induced_drag_is_quadratic_in_circulation():
    analysis = make_analysis()
    original = analysis.run()
    circulation = analysis.vortex_strengths.copy()
    zero = _panel_induced_drag(analysis, np.zeros_like(circulation))
    scaled = _panel_induced_drag(analysis, -2 * circulation)
    np.testing.assert_allclose(zero, 0.0, atol=1e-14)
    np.testing.assert_allclose(scaled.sum(), 4 * original["CD"], rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("twist", [0.0, -3.0])
def test_total_drag_uses_corrected_loading_and_integrates_station_drag(twist):
    geometry = make_geometry(twist)
    result = solve_vlm_neuralfoil(
        geometry, velocity=50.0, alpha=5.0,
        spanwise_resolution=6, chordwise_resolution=3,
    )
    components = result["drag_components"]
    assert components["induced"] > 0
    assert result["CD"] > components["profile"]
    assert abs(components["induced"] - result["initial_vlm"]["CDi"]) > 1e-8
    np.testing.assert_allclose(result["CD"], sum(components.values()), rtol=1e-12)
    for field, expected in (
        ("cd", result["CD"]),
        ("cd_2d", components["profile"]),
        ("cd_induced", components["induced"]),
    ):
        integrated = sum(s[field] * s["area_m2"] for s in result["stations"])
        np.testing.assert_allclose(
            integrated / geometry.parameters.area_m2, expected, rtol=1e-12, atol=1e-14,
        )


def test_symmetric_wing_drag_is_even_in_angle_of_attack():
    geometry = make_geometry()
    options = dict(velocity=50.0, spanwise_resolution=6, chordwise_resolution=3)
    positive = solve_vlm_neuralfoil(geometry, alpha=5.0, **options)
    negative = solve_vlm_neuralfoil(geometry, alpha=-5.0, **options)
    np.testing.assert_allclose(positive["CD"], negative["CD"], rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(positive["CL"], -negative["CL"], rtol=1e-9, atol=1e-12)
