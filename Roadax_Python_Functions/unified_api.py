"""
Unified, simple facade for the package callables.

Goals:
- Keep it very easy to route a single "feature" + args from an API.
- Preserve each function's native return format (tuple, dict, etc.).
- Be readable and minimal for developers.

Supported feature keys -> functions:
  design_by_type              -> Design_by_type.run_unified_pavement_design(**args)
  edit_type_to_check          -> Edit_type_to_check.hydrate_from_shared(Shared)
  critical_strain_analysis    -> Critical_Strain_Analysis.exact_criticals_with_details(**args)
  multilayer_analysis         -> Multilayer_Analysis.multilayer_main(**args)
  effective_cbr_calc          -> Effective_CBR_Calc.AIO_Effective_CBR(**args)
  permissible_strain_analysis -> Permissible_Strain_Analysis.compute_permissible_si(**args)

Convenience pipeline:
  design_then_hydrate(design_args, hydrate_args=None)
    -> runs the unified design, then feeds its Shared into hydrate_from_shared.

JSON helpers:
  run_feature_json(feature, args)
  design_then_hydrate_json(design_args, hydrate_args=None)
    Return the same information as the main files but in JSON-safe shapes.
"""

from typing import Any, Dict, Optional, Tuple

# Direct, explicit imports for clarity (OK to import heavy libs at process start)
from Design_by_type import run_unified_pavement_design
from Edit_type_to_check import hydrate_from_shared
from Critical_Strain_Analysis import exact_criticals_with_details
from Multilayer_Analysis import multilayer_main
from Effective_CBR_Calc import AIO_Effective_CBR
from Permissible_Strain_Analysis import compute_permissible_si


# Map short feature keys to callables
FUNCTIONS = {
    "design_by_type": run_unified_pavement_design,
    "edit_type_to_check": hydrate_from_shared,  # expects Shared dict
    "critical_strain_analysis": exact_criticals_with_details,
    "multilayer_analysis": multilayer_main,
    "effective_cbr_calc": AIO_Effective_CBR,
    "permissible_strain_analysis": compute_permissible_si,
}


def run_feature(feature: str, args: Dict[str, Any]) -> Any:
    """Run a single feature by key with args and return its native result.

    Example:
      run_feature("design_by_type", { ...kwargs... })
    """
    if feature not in FUNCTIONS:
        raise ValueError(f"Unknown feature: {feature}")
    func = FUNCTIONS[feature]
    return func(**(args or {}))


def design_then_hydrate(
    design_args: Dict[str, Any],
    hydrate_args: Optional[Dict[str, Any]] = None,
) -> Tuple[Any, Any]:
    """
    Convenience pipeline that mirrors the common flow:
      1) run_unified_pavement_design -> (Best, ResultsTable, Shared, TRACE, T, hFig)
      2) hydrate_from_shared(Shared)  -> (Report_t, Cost_km, breakdown, Derived)

    Returns a tuple (design_result, hydrate_result) with the functions' native outputs.
    """
    design_result = run_unified_pavement_design(**(design_args or {}))
    _, _, Shared, _, _, _ = design_result

    # Allow caller to override or inject Shared explicitly
    if hydrate_args:
        shared_override = hydrate_args.get("Shared")
        if shared_override is not None:
            Shared = shared_override

    hydrate_result = hydrate_from_shared(Shared)
    return design_result, hydrate_result


def example_type8_args() -> Dict[str, Any]:
  """
  Provide a ready-to-run, plausible argument set for a Type 8 design
  (Reinforced WMM base over CTSB), based on the project defaults used
  during smoketests.

  Returns a dict suitable for run_unified_pavement_design(**args).
  """
  return {
    "Type": 8,
    "Design_Traffic": 100.0,            # msa
    "Effective_Subgrade_CBR": 15.0,     # %
    "Reliability": 90.0,                # %
    "Va": 3.5,                          # %
    "Vbe": 11.5,                        # %
    "BT_Mod": 3000.0,                   # MPa
    # Costs and widths (illustrative)
    "BC_cost": 8000.0,                  # ₹/m^3
    "DBM_cost": 7000.0,                 # ₹/m^3
    "BC_DBM_width": 4.0,                # m
    "Base_cost": 3000.0,                # ₹/m^3
    "Subbase_cost": 2000.0,             # ₹/m^3
    "Base_Sub_width": 7.0,              # m
    # Optional/Type-specific
    "cfdchk_UI": None,
    "FS_CTB_UI": None,
    "RF_UI": None,
    "CRL_cost_UI": None,
    "SAMI_cost_UI": None,
    "Rtype_UI": 1,                      # MIF reinforcement
    "is_wmm_r_UI": 1,                   # reinforce base
    "R_Base_UI": 2.0,                   # reinforcement factor
    "is_gsb_r_UI": None,                # not used in Type 8
    "R_Subbase_UI": None,
    "wmm_r_cost_UI": 100.0,             # flat cost for reinforcement
    "gsb_r_cost_UI": None,
    # Axle mixes (not required for Type 8)
    "SA_M_UI": None,
    "TaA_M_UI": None,
    "TrA_M_UI": None,
    # Moduli
    "AIL_Mod_UI": None,
    "WMM_Mod_UI": 300.0,                # MPa
    "ETB_Mod_UI": None,
    "CTB_Mod_UI": None,
    "CTSB_Mod_UI": 600.0,               # MPa
  }


def run_example_type8(**_ignore: Any) -> Any:
  """Run the Type 8 example using example_type8_args().

  Accepts and ignores any kwargs so it can be safely placed in FUNCTIONS
  and invoked via run_feature(feature, args) with args={}.
  Returns the native tuple from run_unified_pavement_design.
  """
  return run_unified_pavement_design(**example_type8_args())


# Optionally expose the example runner as a callable feature as well
FUNCTIONS["example_type8"] = run_example_type8


def example_criticals_args() -> Dict[str, Any]:
  """Provide the critical strain analysis arguments exactly matching
  Critical_Strain_Analysis.__main__ so outputs are comparable.
  """
  Number_of_layers = 5
  Thickness_layers = [100.0, 200.0, 200.0, 250.0]
  Modulus_layers = [3000.0, 400.0, 5000.0, 800.0, 100.0]
  Poissons = [0.35, 0.35, 0.25, 0.35, 0.35]

  Eva_depth_bituminous = 100.0
  Eva_depth_base = 500.0
  Eva_depth_Subgrade = 750.0

  FS_CTB_T = 1.4
  CFD_Check = 1

  SA_M_T = [
    [185, 195, 70000],
    [175, 185, 90000],
    [165, 175, 92000],
    [155, 165, 300000],
    [145, 155, 280000],
    [135, 145, 650000],
    [125, 135, 600000],
    [115, 125, 1340000],
  ]
  TaA_M_T = [
    [390, 410, 200000],
    [370, 390, 230000],
    [350, 370, 240000],
    [330, 350, 235000],
    [310, 330, 225000],
    [290, 310, 475000],
    [270, 290, 450000],
    [250, 270, 1435000],
  ]
  TrA_M_T = [
    [585, 615, 35000],
    [555, 585, 40000],
    [525, 555, 40000],
    [495, 525, 45000],
    [465, 495, 43000],
    [435, 465, 110000],
    [405, 435, 100000],
    [375, 405, 330000],
    [345, 375, 300000],
  ]

  return {
    "Number_of_layers": Number_of_layers,
    "Thickness_layers": Thickness_layers,
    "Modulus_layers": Modulus_layers,
    "Poissons": Poissons,
    "Eva_depth_bituminous": Eva_depth_bituminous,
    "Eva_depth_base": Eva_depth_base,
    "Eva_depth_Subgrade": Eva_depth_Subgrade,
    "CFD_Check": CFD_Check,
    "FS_CTB_T": FS_CTB_T,
    "SA_M_T": SA_M_T,
    "TaA_M_T": TaA_M_T,
    "TrA_M_T": TrA_M_T,
  }


def run_example_criticals(**_ignore: Any) -> Any:
  """Run the Critical Strain Analysis example using example_criticals_args()."""
  return exact_criticals_with_details(**example_criticals_args())


FUNCTIONS["example_criticals"] = run_example_criticals


if __name__ == "__main__":
    # Print available feature keys for quick reference
    print("Available features:")
    for k in FUNCTIONS.keys():
        print(" -", k)


# ------------------------- JSON-shaping helpers -------------------------

def _to_jsonable(obj: Any) -> Any:
  """Convert numpy/pandas/matplotlib and nested containers to JSON-safe types."""
  if obj is None or isinstance(obj, (bool, int, float, str)):
    return obj
  # numpy
  try:
    import numpy as np  # type: ignore
    if isinstance(obj, np.generic):
      return obj.item()
    if isinstance(obj, np.ndarray):
      return obj.tolist()
  except Exception:
    pass
  # pandas
  try:
    import pandas as pd  # type: ignore
    if isinstance(obj, pd.DataFrame):
      return {"_type": "DataFrame", "columns": obj.columns.tolist(), "data": obj.to_dict(orient="records")}
    if isinstance(obj, pd.Series):
      return {"_type": "Series", "name": obj.name, "data": obj.to_dict()}
  except Exception:
    pass
  # matplotlib figure -> drop
  try:
    import matplotlib.figure as mpl_fig  # type: ignore
    if isinstance(obj, mpl_fig.Figure):
      return None
  except Exception:
    pass
  if isinstance(obj, dict):
    return {str(k): _to_jsonable(v) for k, v in obj.items()}
  if isinstance(obj, (list, tuple, set)):
    return [_to_jsonable(x) for x in obj]
  try:
    return str(obj)
  except Exception:
    return None


def _shape_design_by_type(out: Any) -> Any:
  try:
    Best, ResultsTable, Shared, TRACE, T, hFig = out  # type: ignore[misc]
    return {
      "Best": _to_jsonable(Best),
      "ResultsTable": _to_jsonable(ResultsTable),
      "Shared": _to_jsonable(Shared),
      "TRACE": _to_jsonable(TRACE),
      "T": _to_jsonable(T),
      "hFig": _to_jsonable(hFig),
    }
  except Exception:
    return _to_jsonable(out)


def _shape_edit_type_to_check(out: Any) -> Any:
  try:
    Report_t, Cost_km, breakdown, Derived = out  # type: ignore[misc]
    return {
      "Report_t": _to_jsonable(Report_t),
      "Cost_km": _to_jsonable(Cost_km),
      "breakdown": _to_jsonable(breakdown),
      "Derived": _to_jsonable(Derived),
    }
  except Exception:
    return _to_jsonable(out)


def _shape_multilayer(out: Any) -> Any:
  try:
    Report_arr, ResultTable = out  # type: ignore[misc]
    return {
      "Report_arr": _to_jsonable(Report_arr),
      "ResultTable": _to_jsonable(ResultTable),
    }
  except Exception:
    return _to_jsonable(out)


def _shape_permissible(out: Any) -> Any:
  try:
    vec4, Perm_Si_R, extra = out  # type: ignore[misc]
    return {
      "vec4": _to_jsonable(vec4),
      "Perm_Si_R": _to_jsonable(Perm_Si_R),
      "out": _to_jsonable(extra),
    }
  except Exception:
    return _to_jsonable(out)


def run_feature_json(feature: str, args: Dict[str, Any]) -> Any:
  """Run feature and return JSON-safe output mirroring main file semantics.

  - design_by_type            -> dict(Best, ResultsTable, Shared, TRACE, T, hFig)
  - edit_type_to_check        -> dict(Report_t, Cost_km, breakdown, Derived)
  - multilayer_analysis       -> dict(Report_arr, ResultTable)
  - permissible_strain_analysis -> dict(vec4, Perm_Si_R, out)
  - critical_strain_analysis  -> list (4 values)
  - effective_cbr_calc        -> float
  """
  out = run_feature(feature, args)
  if feature == "design_by_type":
    return _shape_design_by_type(out)
  if feature == "edit_type_to_check":
    return _shape_edit_type_to_check(out)
  if feature == "multilayer_analysis":
    return _shape_multilayer(out)
  if feature == "permissible_strain_analysis":
    return _shape_permissible(out)
  # default for the remaining features
  return _to_jsonable(out)


def design_then_hydrate_json(
  design_args: Dict[str, Any],
  hydrate_args: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
  design_result, hydrate_result = design_then_hydrate(design_args, hydrate_args)
  return {
    "design": _shape_design_by_type(design_result),
    "hydrate": _shape_edit_type_to_check(hydrate_result),
  }
