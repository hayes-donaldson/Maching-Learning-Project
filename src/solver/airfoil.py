"""NeuralFoil section data. Angles passed to NeuralFoil are in degrees."""

from typing import Any

import numpy as np


# Section aerodynamics: obtain airfoil coefficients at the local flow conditions.
def evaluate_airfoil(
    airfoil: Any,
    alpha_deg: np.ndarray,
    Re: np.ndarray,
    mach: float,
    model_size: str = "large",
) -> dict[str, np.ndarray]:
    """Evaluate one airfoil at all stations in a single vectorized call."""

    # Predict section lift, drag, and pitching moment at each station's flow conditions.
    aero = airfoil.get_aero_from_neuralfoil(
        alpha=alpha_deg,
        Re=Re,
        mach=mach,
        model_size=model_size,
    )

    # Keep one value of each coefficient per span station.
    coefficients = {
        key: np.asarray(aero[key], dtype=float).ravel()
        for key in ("CL", "CD", "CM")
    }

    # Verify that every span station has finite aerodynamic coefficients.
    n_stations = np.asarray(alpha_deg).size
    for key, values in coefficients.items():
        if values.size != n_stations or not np.all(np.isfinite(values)):
            raise ValueError(f"NeuralFoil returned invalid {key} station data")

    return coefficients
