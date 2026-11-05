"""Symmetric trapezoidal wings with linear twist. Lengths in m, angles in degrees."""

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any


# Wing inputs: define the planform, airfoil, and root-to-tip twist.
@dataclass(frozen=True)
class WingParameters:
    """Full-wing area; tip/root taper; named airfoil; tip twist relative to root.

    Root twist is zero. Positive tip twist is wash-in; negative is washout.

    A custom airfoil must eventually use a versioned coordinate-content hash
    rather than an ambiguous name. Operating conditions are not geometry inputs.
    """

    aspect_ratio: float
    area_m2: float
    taper_ratio: float
    sweep_deg: float
    airfoil: str
    tip_twist_deg: float = 0.0

    def __post_init__(self) -> None:
        # Require positive aspect ratio, wing area, and taper ratio.
        for name in ("aspect_ratio", "area_m2", "taper_ratio"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")

        # Keep quarter-chord sweep strictly between -90 and 90 degrees.
        if not math.isfinite(self.sweep_deg) or abs(self.sweep_deg) >= 90:
            raise ValueError("sweep_deg must be finite and between -90 and 90")

        # Require an airfoil name for the aerodynamic model.
        if not isinstance(self.airfoil, str) or not self.airfoil.strip():
            raise ValueError("airfoil must be a nonempty identifier")

    @property
    def geometry_id(self) -> str:
        """Stable ID for exact canonical parameters, independent of flight state."""

        # Give identical wing designs the same identifier regardless of number formatting.
        values = asdict(self)
        for key in ("aspect_ratio", "area_m2", "taper_ratio", "sweep_deg", "tip_twist_deg"):
            value = float(values[key])
            values[key] = 0.0 if value == 0 else value
        values["airfoil"] = self.airfoil.strip().lower()

        # Create a repeatable identifier from the wing design inputs.
        payload = json.dumps({"schema": 2, **values}, sort_keys=True, allow_nan=False)
        return "wing_" + hashlib.sha256(payload.encode()).hexdigest()[:20]


# Wing dimensions: store the span, root and tip chords, and tip position.
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


# Planform construction: convert the design inputs into trapezoidal wing dimensions.
def build_wing_geometry(parameters: WingParameters) -> WingGeometry:
    """Construct dimensions for a symmetric wing with root leading edge at x=0."""

    # Recover full span from aspect ratio: AR = b^2 / S.
    span = math.sqrt(parameters.aspect_ratio * parameters.area_m2)

    # Recover root and tip chords from trapezoidal area and taper ratio.
    root = 2 * parameters.area_m2 / (span * (1 + parameters.taper_ratio))
    tip = root * parameters.taper_ratio

    # Place the tip leading edge so the quarter-chord line has the requested sweep.
    tip_le = span / 2 * math.tan(math.radians(parameters.sweep_deg)) + (root - tip) / 4

    return WingGeometry(parameters, span, root, tip, tip_le)


# Aerodynamic geometry: define the root and tip sections for AeroSandbox.
def to_aerosandbox_wing(geometry: WingGeometry) -> Any:
    """Convert analytic geometry to a symmetric AeroSandbox wing."""
    import aerosandbox as asb

    # Apply the same named airfoil at the root and tip.
    airfoil = asb.Airfoil(
        name=geometry.parameters.airfoil.strip().lower()
    )

    # Define the right semispan and mirror it across the wing centerline.
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
                twist=geometry.parameters.tip_twist_deg,
                airfoil=airfoil,
            ),
        ],
    )
