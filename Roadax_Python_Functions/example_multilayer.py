"""
Example script: run Multilayer Analysis via the unified API.

Inputs are taken from the provided demo snippet (dual-wheel case).
"""

from typing import Any, Dict

from unified_api import run_feature


def main() -> None:
    # ---------------- INPUTS (as provided) ----------------
    Number_of_layers = 4
    Thickness_layers = [100.00, 240.00, 200.00]
    Modulus_layers   = [3000.00, 617.94, 300.11, 76.83]
    Poissons         = [0.35, 0.35, 0.35, 0.35]

    Tyre_pressure = 0.56
    wheel_load = 20000

    analysis_points = 4
    depths = [100, 100, 540, 540]
    radii_from_left = [0, 155, 0, 155]

    isbonded = True
    center_spacing = 310
    alpha_deg = 0

    wheel_set = 2

    # Build args for unified API (parameter name is 'radii')
    args: Dict[str, Any] = {
        "Number_of_layers": Number_of_layers,
        "Thickness_layers": Thickness_layers,
        "Modulus_layers": Modulus_layers,
        "Poissons": Poissons,
        "Tyre_pressure": Tyre_pressure,
        "wheel_load": wheel_load,
        "wheel_set": wheel_set,
        "analysis_points": analysis_points,
        "depths": depths,
        "radii": radii_from_left,
        "isbonded": isbonded,
        "center_spacing": center_spacing,
        "alpha_deg": alpha_deg,
    }

    # Invoke multilayer analysis via unified API
    report_arr, result_table = run_feature("multilayer_analysis", args)

    # Pretty-print outputs
    try:
        import numpy as np  # noqa: F401
        print("Report_arr shape:", getattr(report_arr, "shape", None))
    except Exception:
        pass

    print("\n--- Multilayer Analysis Result (table) ---")
    try:
        # If it's a pandas DataFrame, print nicely
        print(result_table.to_string(index=False))
    except Exception:
        # Fallback generic print
        print(result_table)


if __name__ == "__main__":
    main()
