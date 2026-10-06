"""Aerodynamic geometry, polar, and solver interfaces, independent of ML."""

from .geometry import WingGeometry, WingParameters, build_wing_geometry
from .solver import OperatingPoint, SolverResult, SolverSettings, run_solver

__all__ = [
    "WingGeometry", "WingParameters", "build_wing_geometry", "OperatingPoint",
    "SolverResult", "SolverSettings", "run_solver",
]
