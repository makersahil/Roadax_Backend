"""
Example script: run the 'Bridge' pipeline via unified API (Type 2 design + hydrate).

It mirrors the behavior of Bridge.py but uses the facade functions
so it's consistent with the unified API.
"""

from typing import Any, Dict

from unified_api import design_then_hydrate_json


def main() -> None:
    # Design args (Type 2) from the provided snippet
    design_args: Dict[str, Any] = {
        "Type": 2,
        "Design_Traffic": 50.0,
        "Effective_Subgrade_CBR": 15.0,
        "Reliability": 90.0,
        "Va": 3.5,
        "Vbe": 11.5,
        "BT_Mod": 3000.0,
        "BC_cost": 6000.0,
        "DBM_cost": 5000.0,
        "BC_DBM_width": 7.0,
        "Base_cost": 1800.0,
        "Subbase_cost": 1200.0,
        "Base_Sub_width": 7.0,
        "cfdchk_UI": 0,
        "FS_CTB_UI": 1.4,
        "RF_UI": 1,
        "CRL_cost_UI": 3000,
        "SAMI_cost_UI": None,
        "Rtype_UI": None,
        "is_wmm_r_UI": None,
        "R_Base_UI": None,
        "is_gsb_r_UI": None,
        "R_Subbase_UI": None,
        "wmm_r_cost_UI": None,
        "gsb_r_cost_UI": None,
        "SA_M_UI": None,
        "TaA_M_UI": None,
        "TrA_M_UI": None,
        "AIL_Mod_UI": 450,
        "WMM_Mod_UI": None,
        "ETB_Mod_UI": None,
        "CTB_Mod_UI": 5000,
        "CTSB_Mod_UI": 600,
    }

    # Run pipeline: design -> hydrate (JSON-shaped output)
    result = design_then_hydrate_json(design_args)
    design_json = result.get("design", {})
    hydrate_json = result.get("hydrate", {})

    # Extract hydrate fields from JSON-safe shape
    Report_t = hydrate_json.get("Report_t")
    Cost_km = hydrate_json.get("Cost_km")
    breakdown = hydrate_json.get("breakdown", {})

    print("Report_t (Permissible, Critical, Pass/Fail):")
    print(Report_t)
    try:
        # Cost_km could be a numpy scalar already converted to Python float
        print(f"Cost (₹ lakh/km): {float(Cost_km):.3f}")
    except Exception:
        print("Cost (₹ lakh/km):", Cost_km)
    try:
        print("Cost breakdown keys:", list(breakdown.keys()))
    except Exception:
        print("Cost breakdown:", breakdown)


if __name__ == "__main__":
    main()
