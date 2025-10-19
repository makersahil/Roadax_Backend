"""
Example script: run Effective CBR calculation via the unified API.

Uses the provided layered system inputs and prints the Effective CBR.
"""

from typing import Any, Dict

from unified_api import run_feature


def main() -> None:
    # Inputs (exactly as in Effective_CBR_Calc.__main__ example)
    number_of_layer = 5
    thk       = [200, 300, 100, 400]
    CBR       = [10, 5, 10, 5, 8]
    Poisson_r = [0.35, 0.35, 0.35, 0.35, 0.35]

    args: Dict[str, Any] = {
        "number_of_layer": number_of_layer,
        "thk": thk,
        "CBR": CBR,
        "Poisson_r": Poisson_r,
    }

    # Run via unified API
    effective_cbr = run_feature("effective_cbr_calc", args)

    # Print result
    try:
        print(f"Effective CBR: {float(effective_cbr):.3f}")
    except Exception:
        print("Effective CBR:", effective_cbr)


if __name__ == "__main__":
    main()
