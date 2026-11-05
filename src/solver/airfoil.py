"""NeuralFoil section data. Angles passed to NeuralFoil are in degrees."""

import numpy as np


def evaluate_airfoil(airfoil, alpha_deg, Re, mach, model_size="large"):
    """Evaluate one airfoil at all stations in a single vectorized call."""

    aero = airfoil.get_aero_from_neuralfoil(
        alpha=alpha_deg,
        Re=Re,
        mach=mach,
        model_size=model_size,
    )

    return {
        key: np.asarray(aero[key], dtype=float).ravel()
        for key in ("CL", "CD", "CM")
    }
