"""Symmetric trapezoidal wings with linear twist. Lengths in m, angles in degrees."""

from dataclasses import asdict, dataclass
import hashlib
import json
import math


@dataclass(frozen=True)
class WingParameters:
    """Wing inputs with zero root twist; negative tip twist means washout."""

    aspect_ratio: float
    area_m2: float
    taper_ratio: float
    sweep_deg: float
    airfoil: str
    tip_twist_deg: float = 0.0

    def geometry_id(self):
        """Stable ID for exact canonical parameters, independent of flight state."""

        # Normalize formatting so identical designs have the same ID.
        values = asdict(self)
        for key in ("aspect_ratio", "area_m2", "taper_ratio", "sweep_deg", "tip_twist_deg"):
            value = float(values[key])
            values[key] = 0.0 if value == 0 else value
        values["airfoil"] = self.airfoil.strip().lower()

        payload = json.dumps({"schema": 2, **values}, sort_keys=True, allow_nan=False)
        return "wing_" + hashlib.sha256(payload.encode()).hexdigest()[:20]


@dataclass(frozen=True)
class WingGeometry:
    """Analytic geometry specification, not an aerodynamic prediction."""

    parameters: WingParameters
    span_m: float
    root_chord_m: float
    tip_chord_m: float
    tip_le_x_m: float

    def geometry_id(self):
        return self.parameters.geometry_id()


def build_wing_geometry(parameters):
    """Construct dimensions for a symmetric wing with root leading edge at x=0."""

    # AR = span^2 / area.
    span = math.sqrt(parameters.aspect_ratio * parameters.area_m2)

    # Trapezoidal area = span * (root chord + tip chord) / 2.
    root = 2 * parameters.area_m2 / (span * (1 + parameters.taper_ratio))
    tip = root * parameters.taper_ratio

    # Apply sweep at the quarter-chord line.
    tip_le = span / 2 * math.tan(math.radians(parameters.sweep_deg)) + (root - tip) / 4

    return WingGeometry(parameters, span, root, tip, tip_le)


def to_aerosandbox_wing(geometry):
    """Convert analytic geometry to a symmetric AeroSandbox wing."""
    import aerosandbox as asb

    parameters = geometry.parameters
    airfoil = asb.Airfoil(name=parameters.airfoil.strip().lower())

    # Mirror the right half-wing to create the full wing.
    return asb.Wing(
        name=geometry.geometry_id(),
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
                twist=parameters.tip_twist_deg,
                airfoil=airfoil,
            ),
        ],
    )
