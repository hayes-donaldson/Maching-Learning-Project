import math
import unittest

from solver.geometry import WingParameters, build_wing_geometry, to_aerosandbox_wing
from solver.solver import OperatingPoint, SolverSettings, run_solver


class GeometryTests(unittest.TestCase):
    def test_area_aspect_ratio_and_quarter_chord_sweep(self):
        p = WingParameters(8, 1.5, 0.4, 20, "naca2412")
        wing = build_wing_geometry(p)
        self.assertAlmostEqual(wing.span_m ** 2 / p.area_m2, p.aspect_ratio)
        self.assertAlmostEqual(wing.span_m * (wing.root_chord_m + wing.tip_chord_m) / 2, p.area_m2)
        self.assertAlmostEqual(wing.tip_chord_m / wing.root_chord_m, p.taper_ratio)
        sweep = math.degrees(math.atan((wing.tip_le_x_m + wing.tip_chord_m / 4 - wing.root_chord_m / 4) / (wing.span_m / 2)))
        self.assertAlmostEqual(sweep, p.sweep_deg)

    def test_identity_and_invalid_parameters(self):
        first = WingParameters(8, 1, 0.5, 0, "NACA2412")
        self.assertEqual(first.geometry_id, WingParameters(8.0, 1.0, 0.5, -0.0, "naca2412").geometry_id)
        self.assertNotEqual(first.geometry_id, WingParameters(9, 1, 0.5, 0, "naca2412").geometry_id)
        for args in [(0, 1, 1, 0, "naca0012"), (8, -1, 1, 0, "naca0012"), (8, 1, 0, 0, "naca0012"), (8, 1, 1, 90, "naca0012"), (8, 1, 1, 0, ""), (float("nan"), 1, 1, 0, "naca0012")]:
            with self.assertRaises(ValueError):
                WingParameters(*args)

    def test_aerosandbox_adapter_preserves_dimensions(self):
        import pytest
        pytest.importorskip("aerosandbox")
        wing = build_wing_geometry(WingParameters(8, 1, 0.5, 0, "naca2412"))
        converted = to_aerosandbox_wing(wing)
        self.assertAlmostEqual(converted.area(), wing.parameters.area_m2)
        self.assertAlmostEqual(converted.span(), wing.span_m)
