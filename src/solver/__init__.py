"""Aerodynamic geometry, polar, and solver interfaces, independent of ML."""

from .geometry import WingGeometry, WingParameters, build_wing_geometry

__all__ = [
    "WingGeometry", "WingParameters", "build_wing_geometry",
]
