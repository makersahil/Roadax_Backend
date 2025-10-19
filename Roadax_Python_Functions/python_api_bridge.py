#!/usr/bin/env python3
"""
Simple JSON bridge for Node.js (or any process).

Reads a JSON payload from stdin and writes a JSON response to stdout.

Payloads you can send:
    1) Run a single feature (preserves that function's native return structure):
         {
             "feature": "multilayer_analysis",   # one of unified_api.FUNCTIONS keys
             "args": { ...kwargs... }
         }

    2) Run the common pipeline (design then hydrate):
         {
             "pipeline": "design_then_hydrate",
             "design_args": { ... },
             "hydrate_args": { ... }  # optional; if omitted, Shared comes from design
         }

    3) List available features:
         { "action": "features" }

Response shape (always JSON):
    { "ok": true, "data": ... }  OR  { "ok": false, "error": { type, message, details } }
"""

from __future__ import annotations

import json
import sys
from typing import Any, Dict, Tuple
import math


def _to_jsonable(obj: Any) -> Any:
    """Convert common scientific Python objects to JSON-serializable types, recursively."""
    # Primitives
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj

    # Numpy scalars/arrays
    try:
        import numpy as np  # lazy import
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except Exception:
        pass

    # Pandas objects
    try:
        import pandas as pd  # lazy import
        if isinstance(obj, pd.DataFrame):
            return {
                "_type": "DataFrame",
                "columns": obj.columns.tolist(),
                "data": obj.to_dict(orient="records"),
            }
        if isinstance(obj, pd.Series):
            return {
                "_type": "Series",
                "name": obj.name,
                "data": obj.to_dict(),
            }
    except Exception:
        pass

    # Matplotlib figure -> drop (caller shouldn't need raw binary here)
    try:
        import matplotlib.figure as mpl_fig  # lazy import
        if isinstance(obj, mpl_fig.Figure):
            return None
    except Exception:
        pass

    # Dicts and iterables
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_jsonable(x) for x in obj]

    # Fallback to string representation
    try:
        return str(obj)
    except Exception:
        return None


def _shape_design_by_type(out: Any) -> Dict[str, Any]:
    """Map run_unified_pavement_design return tuple to named fields."""
    try:
        Best, ResultsTable, Shared, TRACE, T, hFig = out  # type: ignore[misc]
        return {
            "Best": _to_jsonable(Best),
            "ResultsTable": _to_jsonable(ResultsTable),
            "Shared": _to_jsonable(Shared),
            "TRACE": _to_jsonable(TRACE),
            "T": _to_jsonable(T),
            "hFig": _to_jsonable(hFig),  # becomes null
        }
    except Exception:
        return _to_jsonable(out)


def _shape_edit_type_to_check(out: Any) -> Dict[str, Any]:
    """Map hydrate_from_shared return tuple to named fields."""
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


def _shape_feature_output(feature: str, out: Any) -> Any:
    """Return JSON-safe output while mirroring main file structures.

    - design_by_type: named dict with Best, ResultsTable, Shared, TRACE, T, hFig
    - edit_type_to_check: named dict with Report_t, Cost_km, breakdown, Derived
    - others: keep native structure but converted to JSON-safe (e.g., ndarray -> list)
    """
    if feature == "design_by_type":
        return _shape_design_by_type(out)
    if feature == "edit_type_to_check":
        return _shape_edit_type_to_check(out)
    return _to_jsonable(out)


def _handle(payload: Dict[str, Any]) -> Dict[str, Any]:
    action = (payload or {}).get("action")

    # Action: noop (health check without importing heavy modules)
    if action == "noop":
        return {"ok": True, "data": {"uptime": True, "python": sys.version}}

    # Action: list features
    if action == "features":
        from unified_api import FUNCTIONS
        return {"ok": True, "data": sorted(list(FUNCTIONS.keys()))}

    # Single feature
    feature = (payload or {}).get("feature")
    if feature:
        # Fast-path for effective CBR without importing heavy modules
        if feature == "effective_cbr_calc":
            try:
                from Effective_CBR_Calc import AIO_Effective_CBR  # type: ignore
                args = (payload or {}).get("args") or {}
                out = AIO_Effective_CBR(**args)
                return {"ok": True, "data": _to_jsonable(out)}
            except Exception as e:
                return {"ok": False, "error": {"type": type(e).__name__, "message": str(e), "details": _to_jsonable(getattr(e, "args", []))}}
        # Fast-path for critical strain analysis
        if feature == "critical_strain_analysis":
            try:
                from Critical_Strain_Analysis import exact_criticals_with_details  # type: ignore
                args = (payload or {}).get("args") or {}
                out = exact_criticals_with_details(**args)
                return {"ok": True, "data": _to_jsonable(out)}
            except Exception as e:
                return {"ok": False, "error": {"type": type(e).__name__, "message": str(e), "details": _to_jsonable(getattr(e, "args", []))}}
        # Fast-path for multilayer analysis to avoid importing unified_api (which pulls pandas-heavy modules)
        if feature == "multilayer_analysis":
            try:
                from Multilayer_Analysis import multilayer_main  # type: ignore
                args = (payload or {}).get("args") or {}
                out = multilayer_main(**args)
                try:
                    Report_arr, ResultTable = out  # type: ignore[misc]
                    shaped = {
                        "Report_arr": _to_jsonable(Report_arr),
                        "ResultTable": _to_jsonable(ResultTable),
                    }
                except Exception:
                    shaped = _to_jsonable(out)
                return {"ok": True, "data": shaped}
            except Exception as e:
                return {"ok": False, "error": {"type": type(e).__name__, "message": str(e), "details": _to_jsonable(getattr(e, "args", []))}}
        # Fast-path for permissible strain analysis
        if feature == "permissible_strain_analysis":
            try:
                from Permissible_Strain_Analysis import compute_permissible_si  # type: ignore
                args = (payload or {}).get("args") or {}
                out = compute_permissible_si(**args)
                try:
                    vec4, Perm_Si_R, extra = out  # type: ignore[misc]
                    shaped = {
                        "vec4": _to_jsonable(vec4),
                        "Perm_Si_R": _to_jsonable(Perm_Si_R),
                        "out": _to_jsonable(extra),
                    }
                except Exception:
                    shaped = _to_jsonable(out)
                return {"ok": True, "data": shaped}
            except Exception as e:
                return {"ok": False, "error": {"type": type(e).__name__, "message": str(e), "details": _to_jsonable(getattr(e, "args", []))}}
        # Fast-path for design_by_type (heavy); shape DataFrames to JSON
        if feature == "design_by_type":
            try:
                from Design_by_type import run_unified_pavement_design  # type: ignore
                args = (payload or {}).get("args") or {}
                out = run_unified_pavement_design(**args)
                try:
                    Best, ResultsTable, Shared, TRACE, T, hFig = out  # type: ignore[misc]
                    shaped = {
                        "Best": _to_jsonable(Best),
                        "ResultsTable": _to_jsonable(ResultsTable),
                        "Shared": _to_jsonable(Shared),
                        "TRACE": _to_jsonable(TRACE),
                        "T": _to_jsonable(T),
                        "hFig": None,
                    }
                except Exception:
                    shaped = _to_jsonable(out)
                return {"ok": True, "data": shaped}
            except Exception as e:
                return {"ok": False, "error": {"type": type(e).__name__, "message": str(e), "details": _to_jsonable(getattr(e, "args", []))}}
        # Fallback to unified facade for all other features
        from unified_api import run_feature_json
        try:
            args = (payload or {}).get("args") or {}
            out = run_feature_json(feature, args)
            return {"ok": True, "data": out}
        except Exception as e:
            return {"ok": False, "error": {"type": type(e).__name__, "message": str(e), "details": _to_jsonable(getattr(e, "args", []))}}

    # Pipeline: design_then_hydrate
    pipeline = (payload or {}).get("pipeline")
    if pipeline == "design_then_hydrate":
        from unified_api import design_then_hydrate_json
        try:
            design_args = (payload or {}).get("design_args") or {}
            hydrate_args = (payload or {}).get("hydrate_args") or None
            out = design_then_hydrate_json(design_args, hydrate_args)
            return {"ok": True, "data": out}
        except Exception as e:
            return {"ok": False, "error": {"type": type(e).__name__, "message": str(e), "details": _to_jsonable(getattr(e, "args", []))}}

    return {"ok": False, "error": {"type": "BadRequest", "message": "Provide either 'feature' or 'pipeline' (or action='features').", "details": payload}}


def main() -> None:
    try:
        raw = sys.stdin.read()
        payload = json.loads(raw) if raw.strip() else {}
    except Exception as e:
        resp = {"ok": False, "error": {"type": type(e).__name__, "message": f"Invalid JSON input: {e}", "details": None}}
        print(json.dumps(resp, ensure_ascii=False, allow_nan=False))
        return

    resp = _handle(payload)
    # Strict JSON (no NaN/Infinity)
    def _san(v: Any) -> Any:
        if v is None:
            return None
        if isinstance(v, float):
            return v if math.isfinite(v) else None
        if isinstance(v, (int, bool, str)):
            return v
        if isinstance(v, dict):
            return {str(k): _san(vv) for k, vv in v.items()}
        if isinstance(v, (list, tuple, set)):
            return [_san(x) for x in v]
        try:
            return _to_jsonable(v)
        except Exception:
            return None

    try:
        print(json.dumps(resp, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError):
        safe_resp = _san(resp)
        print(json.dumps(safe_resp, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
