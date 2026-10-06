"""Symmetric, untwisted trapezoidal wings. Lengths in m, angles in degrees."""

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any


@dataclass(frozen=True)
class WingParameters:
    """Full-wing area; tip/root taper; quarter-chord sweep; named airfoil.

    A custom airfoil must eventually use a versioned coordinate-content hash
    rather than an ambiguous name. Operating conditions are not geometry inputs.
    """

    aspect_ratio: float
    area_m2: float
    taper_ratio: float
    sweep_deg: float
    airfoil: str

    def __post_init__(self) -> None:
        for name in ("aspect_ratio", "area_m2", "taper_ratio"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.sweep_deg) or abs(self.sweep_deg) >= 90:
            raise ValueError("sweep_deg must be finite and between -90 and 90")
        if not isinstance(self.airfoil, str) or not self.airfoil.strip():
            raise ValueError("airfoil must be a nonempty identifier")

    @property
    def geometry_id(self) -> str:
        """Stable ID for exact canonical parameters, independent of flight state."""
        values = asdict(self)
        for key in ("aspect_ratio", "area_m2", "taper_ratio", "sweep_deg"):
            value = float(values[key])
            values[key] = 0.0 if value == 0 else value
        values["airfoil"] = self.airfoil.strip().lower()
        payload = json.dumps({"schema": 1, **values}, sort_keys=True, allow_nan=False)
        return "wing_" + hashlib.sha256(payload.encode()).hexdigest()[:20]


@dataclass(frozen=True)
class WingGeometry:
    """Analytic geometry specification, not an aerodynamic prediction."""

    parameters: WingParameters
    span_m: float
    root_chord_m: float
    tip_chord_m: float
    tip_le_x_m: float

    @property
    def geometry_id(self) -> str:
        return self.parameters.geometry_id


def build_wing_geometry(parameters: WingParameters) -> WingGeometry:
    """Construct dimensions for a symmetric wing with root leading edge at x=0."""
    span = math.sqrt(parameters.aspect_ratio * parameters.area_m2)
    root = 2 * parameters.area_m2 / (span * (1 + parameters.taper_ratio))
    tip = root * parameters.taper_ratio
    tip_le = span / 2 * math.tan(math.radians(parameters.sweep_deg)) + (root - tip) / 4
    return WingGeometry(parameters, span, root, tip, tip_le)


def to_aerosandbox_wing(geometry: WingGeometry) -> Any:
    """Convert analytic geometry to a symmetric AeroSandbox wing."""
    import aerosandbox as asb

    airfoil = asb.Airfoil(
        name=geometry.parameters.airfoil.strip().lower()
    )

    return asb.Wing(
        name=geometry.geometry_id,
        symmetric=True,
        xsecs=[
            # Root section on the wing centerline
            asb.WingXSec(
                xyz_le=[0.0, 0.0, 0.0],
                chord=geometry.root_chord_m,
                twist=0.0,
                airfoil=airfoil,
            ),
            # Right tip section
            asb.WingXSec(
                xyz_le=[geometry.tip_le_x_m, geometry.span_m / 2, 0.0],
                chord=geometry.tip_chord_m,
                twist=0.0,
                airfoil=airfoil,
            ),
        ],
    )
