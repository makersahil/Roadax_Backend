from __future__ import annotations
# unified_pavement_design.py
# Faithful MATLAB→Python conversion (no logic changes).
# Uses SciPy's jv aliased as besselj to match MATLAB besselj.
# Requires: numpy, pandas, scipy, matplotlib

import math
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
try:
    import pandas as pd  # type: ignore
except Exception:  # optional: allow running without pandas
    pd = None  # type: ignore
from scipy.special import jv as besselj
import matplotlib.pyplot as plt

import warnings, numpy as np
warnings.filterwarnings("ignore", category=RuntimeWarning)
np.seterr(divide="ignore", invalid="ignore")

# ---------- small helpers to mimic MATLAB semantics ----------
def cosd(x: float) -> float:
    return math.cos(math.radians(x))


def sind(x: float) -> float:
    return math.sin(math.radians(x))


def round_half_away_from_zero(x: float) -> float:
    s = 1 if x >= 0 else -1
    return s * math.floor(abs(x) + 0.5)


def quant_step(x: float, s: float) -> float:
    # MATLAB round: ties away from zero
    return s * round_half_away_from_zero(x / s)


def iff(cond: bool, a, b):
    return a if cond else b


# ============================================================
# =============== PUBLIC ENTRY POINT FUNCTION ================
# ============================================================
def run_unified_pavement_design(
    Type: int,
    Design_Traffic: float,
    Effective_Subgrade_CBR: float,
    Reliability: float,
    Va: float,
    Vbe: float,
    BT_Mod: float,
    BC_cost: float,
    DBM_cost: float,
    BC_DBM_width: float,
    Base_cost: float,
    Subbase_cost: float,
    Base_Sub_width: float,
    cfdchk_UI: Optional[int],
    FS_CTB_UI: Optional[float],
    RF_UI: Optional[float],
    CRL_cost_UI: Optional[float],
    SAMI_cost_UI: Optional[float],
    Rtype_UI: Optional[int],
    is_wmm_r_UI: Optional[int],
    R_Base_UI: Optional[float],
    is_gsb_r_UI: Optional[int],
    R_Subbase_UI: Optional[float],
    wmm_r_cost_UI: Optional[float],
    gsb_r_cost_UI: Optional[float],
    SA_M_UI: Optional[np.ndarray],
    TaA_M_UI: Optional[np.ndarray],
    TrA_M_UI: Optional[np.ndarray],
    AIL_Mod_UI: Optional[float],
    WMM_Mod_UI: Optional[float],
    ETB_Mod_UI: Optional[float],
    CTB_Mod_UI: Optional[float],
    CTSB_Mod_UI: Optional[float],
):
    # ------------------------- NORMALIZE UI EMPTIES -------------------------
    cfdchk_UI = 0 if (cfdchk_UI is None) else cfdchk_UI
    FS_CTB_UI = 1.4 if (FS_CTB_UI is None) else FS_CTB_UI
    RF_UI = 1 if (RF_UI is None) else RF_UI

    Rtype_UI = 1 if (Rtype_UI is None or (isinstance(Rtype_UI, float) and np.isnan(Rtype_UI))) else Rtype_UI
    R_Base_UI = 1 if (R_Base_UI is None or (isinstance(R_Base_UI, float) and np.isnan(R_Base_UI))) else R_Base_UI
    R_Subbase_UI = 1 if (R_Subbase_UI is None or (isinstance(R_Subbase_UI, float) and np.isnan(R_Subbase_UI))) else R_Subbase_UI

    is_wmm_r_UI = 0 if (is_wmm_r_UI is None) else is_wmm_r_UI
    is_gsb_r_UI = 0 if (is_gsb_r_UI is None) else is_gsb_r_UI

    CRL_cost_UI = 0 if (CRL_cost_UI is None) else CRL_cost_UI
    SAMI_cost_UI = 0 if (SAMI_cost_UI is None) else SAMI_cost_UI
    wmm_r_cost_UI = 0 if (wmm_r_cost_UI is None) else wmm_r_cost_UI
    gsb_r_cost_UI = 0 if (gsb_r_cost_UI is None) else gsb_r_cost_UI

    SA_M_UI = np.array([]) if (SA_M_UI is None or (isinstance(SA_M_UI, float) and np.isnan(SA_M_UI))) else np.array(SA_M_UI, dtype=float)
    TaA_M_UI = np.array([]) if (TaA_M_UI is None or (isinstance(TaA_M_UI, float) and np.isnan(TaA_M_UI))) else np.array(TaA_M_UI, dtype=float)
    TrA_M_UI = np.array([]) if (TrA_M_UI is None or (isinstance(TrA_M_UI, float) and np.isnan(TrA_M_UI))) else np.array(TrA_M_UI, dtype=float)

    ETB_Mod_UI = 800 if (ETB_Mod_UI is None) else ETB_Mod_UI
    CTSB_Mod_UI = 600 if (CTSB_Mod_UI is None) else CTSB_Mod_UI
    WMM_Mod_UI = 350 if (WMM_Mod_UI is None) else WMM_Mod_UI
    AIL_Mod_UI = 450 if (AIL_Mod_UI is None) else AIL_Mod_UI
    CTB_Mod_UI = 5000 if (CTB_Mod_UI is None) else CTB_Mod_UI

    # ------------------------- TYPE-SPECIFIC ASSIGNMENTS -------------------------
    Base_width = Base_Sub_width
    Subbase_width = Base_Sub_width

    Rtype = None
    R_Base = None
    R_Subbase = None
    is_wmm_r = None
    is_gsb_r = None
    RF_CTB = None
    cfdchk = None
    FS_CTB = None
    SA_M = np.array([])
    TaA_M = np.array([])
    TrA_M = np.array([])
    CRL_cost = 0.0
    CRL_width = 0.0
    SAMI_cost = 0.0
    wmm_r_cost = 0.0
    gsb_r_cost = 0.0
    ETB_Mod = None
    CTSB_Mod = None
    WMM_Mod = None
    AIL_Mod = None
    CTB_Mod = None

    if Type == 1:
        pass
    elif Type == 2:
        RF_CTB = RF_UI; cfdchk = cfdchk_UI; FS_CTB = FS_CTB_UI
        SA_M = SA_M_UI; TaA_M = TaA_M_UI; TrA_M = TrA_M_UI
        AIL_Mod = AIL_Mod_UI; CTB_Mod = CTB_Mod_UI; CTSB_Mod = CTSB_Mod_UI
        CRL_cost = CRL_cost_UI; CRL_width = BC_DBM_width
    elif Type == 3:
        RF_CTB = RF_UI; cfdchk = cfdchk_UI; FS_CTB = FS_CTB_UI
        SA_M = SA_M_UI; TaA_M = TaA_M_UI; TrA_M = TrA_M_UI
        CTB_Mod = CTB_Mod_UI; CTSB_Mod = CTSB_Mod_UI
        SAMI_cost = SAMI_cost_UI
    elif Type == 4:
        ETB_Mod = ETB_Mod_UI; CTSB_Mod = CTSB_Mod_UI
    elif Type == 5:
        RF_CTB = RF_UI; cfdchk = cfdchk_UI; FS_CTB = FS_CTB_UI
        SA_M = SA_M_UI; TaA_M = TaA_M_UI; TrA_M = TrA_M_UI
        AIL_Mod = AIL_Mod_UI; CTB_Mod = CTB_Mod_UI
        CRL_cost = CRL_cost_UI; CRL_width = BC_DBM_width
    elif Type == 6:
        WMM_Mod = WMM_Mod_UI; CTSB_Mod = CTSB_Mod_UI
    elif Type == 7:
        Rtype = Rtype_UI; is_wmm_r = is_wmm_r_UI; is_gsb_r = is_gsb_r_UI
        R_Base = R_Base_UI; R_Subbase = R_Subbase_UI
        wmm_r_cost = wmm_r_cost_UI; gsb_r_cost = gsb_r_cost_UI
    elif Type == 8:
        Rtype = Rtype_UI; is_wmm_r = is_wmm_r_UI; is_gsb_r = 0
        R_Base = R_Base_UI; R_Subbase = None
        wmm_r_cost = wmm_r_cost_UI; WMM_Mod = WMM_Mod_UI; CTSB_Mod = CTSB_Mod_UI
    else:
        raise ValueError("Unsupported Type. Must be 1–8.")

    if (is_wmm_r is None) or (not bool(is_wmm_r)):
        R_Base = 1
    if (is_gsb_r is None) or (not bool(is_gsb_r)):
        R_Subbase = 1

    # ------------------------- CALL SOLVER -------------------------
    Best, T, TRACE = unified_pavement_design(
        Type,
        Design_Traffic,
        Effective_Subgrade_CBR,
        Reliability,
        Va,
        Vbe,
        BT_Mod,
        Rtype,
        R_Base,
        R_Subbase,
        RF_CTB,
        cfdchk,
        FS_CTB,
        SA_M,
        TaA_M,
        TrA_M,
        BC_cost,
        BC_DBM_width,
        DBM_cost,
        Base_cost,
        Base_width,
        Subbase_cost,
        Subbase_width,
        CRL_cost,
        CRL_width,
        SAMI_cost,
        wmm_r_cost,
        gsb_r_cost,
        bool(is_wmm_r) if is_wmm_r is not None else False,
        bool(is_gsb_r) if is_gsb_r is not None else False,
        ETB_Mod,
        CTSB_Mod,
        WMM_Mod,
        AIL_Mod,
        CTB_Mod,
    )

    # Results table + optional plot
    hFig = None
    def _is_empty_df(df: Any) -> bool:
        try:
            return df is None or (hasattr(df, 'empty') and bool(getattr(df, 'empty')))
        except Exception:
            return False
    is_empty = False
    if pd is not None:
        is_empty = _is_empty_df(Best)
    else:
        # dict-shaped DataFrame fallback: {'_type':'DataFrame','columns':[...] ,'data':[...]]}
        try:
            is_empty = (Best is None) or (isinstance(Best, dict) and len(Best.get('data') or []) == 0)
        except Exception:
            is_empty = Best is None

    if is_empty:
        print("Warning: No feasible design found. Skipping table/plot.", file=sys.stderr)
        ResultsTable = pd.DataFrame() if pd is not None else {"_type": "DataFrame", "columns": [], "data": []}
    else:
        if pd is not None:
            ResultsTable, hFig = display_unified_results_table(Best, Type, bool(is_wmm_r) if is_wmm_r is not None else False, bool(is_gsb_r) if is_gsb_r is not None else False)
        else:
            # Build minimal results table without pandas/matplotlib
            if isinstance(Best, dict) and Best.get('_type') == 'DataFrame' and Best.get('data'):
                row = Best['data'][0]
                cols = [
                    'BC_mm','DBM_mm','CRL_Layer','CRL_mm','Base_Layer','Base_mm','Subbase_Layer','Subbase_mm',
                    'Nbf','Nsbr','Nctf','CFD','Cost'
                ]
                # derive BC/DBM split
                BT = float(row.get('BT', 0) or 0)
                if BT <= 40:
                    BC, DBM = BT, 0.0
                elif BT < 90:
                    BC, DBM = 30.0, BT - 30.0
                else:
                    BC, DBM = 40.0, BT - 40.0
                ResultsTable = {
                    "_type": "DataFrame",
                    "columns": cols,
                    "data": [{
                        'BC_mm': BC,
                        'DBM_mm': DBM,
                        'CRL_Layer': ('CRL (AIL)' if (Type in (2,5) and float(row.get('CRL', 0) or 0) > 0) else ("CRL (SAMI)" if Type==3 and float(row.get('CRL',0) or 0)>0 else '')),
                        'CRL_mm': float(row.get('CRL', 0) or 0),
                        'Base_Layer': layer_names_by_type(Type, bool(is_wmm_r_UI), bool(is_gsb_r_UI))[0],
                        'Base_mm': float(row.get('Base', 0) or 0),
                        'Subbase_Layer': layer_names_by_type(Type, bool(is_wmm_r_UI), bool(is_gsb_r_UI))[1],
                        'Subbase_mm': float(row.get('Subbase', 0) or 0),
                        'Nbf': float(row.get('Nbf', float('nan'))),
                        'Nsbr': float(row.get('Nsbr', float('nan'))),
                        'Nctf': float(row.get('Nctf', float('nan'))),
                        'CFD': float(row.get('CFD', float('nan'))),
                        'Cost': float(row.get('Cost', float('nan'))),
                    }]
                }
            else:
                ResultsTable = {"_type": "DataFrame", "columns": [], "data": []}

    # ------------------------- BUILD SHARED -------------------------
    Shared: Dict[str, Any] = {}

    # Design inputs
    Shared["Design_Traffic"] = Design_Traffic
    Shared["Effective_Subgrade_CBR"] = Effective_Subgrade_CBR
    Shared["Reliability"] = Reliability
    Shared["Va"] = Va
    Shared["Vbe"] = Vbe
    Shared["BT_Mod"] = BT_Mod
    Shared["Type"] = Type

    # Reinforcement/CTB/CFD sets
    Shared["Rtype"] = Rtype_UI
    Shared["R_Base"] = R_Base_UI
    Shared["R_Subbase"] = R_Subbase_UI
    Shared["is_wmm_r"] = is_wmm_r_UI
    Shared["is_gsb_r"] = is_gsb_r_UI

    Shared["RF_CTB"] = RF_CTB
    Shared["cfdchk"] = cfdchk
    Shared["FS_CTB"] = FS_CTB
    Shared["SA_M"] = SA_M
    Shared["TaA_M"] = TaA_M
    Shared["TrA_M"] = TrA_M

    # Costs & widths
    Shared["costs"] = dict(
        BC_cost=BC_cost,
        DBM_cost=DBM_cost,
        BC_DBM_width=BC_DBM_width,
        Base_cost=Base_cost,
        Base_width=Base_width,
        Subbase_cost=Subbase_cost,
        Subbase_width=Subbase_width,
        CRL_cost=CRL_cost,
        CRL_width=CRL_width,
        SAMI_cost=SAMI_cost,
        wmm_r_cost=wmm_r_cost,
        gsb_r_cost=gsb_r_cost,
    )

    # Thicknesses from Best
    if pd is not None:
        if Best is not None and not Best.empty:
            Shared["BT_thk"] = float(Best["BT"].iloc[0])
            Shared["CRL_thk"] = float(Best["CRL"].iloc[0]) if "CRL" in Best.columns else 0.0
            Shared["Base_thk"] = float(Best["Base"].iloc[0])
            Shared["Subbase_thk"] = float(Best["Subbase"].iloc[0])
            if "Cost" in Best.columns:
                Shared["Cost"] = float(Best["Cost"].iloc[0])
            if "Nbf" in Best.columns:
                Shared["Nbf"] = float(Best["Nbf"].iloc[0])
            if "Nsbr" in Best.columns:
                Shared["Nsbr"] = float(Best["Nsbr"].iloc[0])
            if "Nctf" in Best.columns:
                Shared["Nctf"] = float(Best["Nctf"].iloc[0])
            if "CFD" in Best.columns:
                Shared["CFD"] = float(Best["CFD"].iloc[0])
        else:
            Shared["BT_thk"], Shared["CRL_thk"], Shared["Base_thk"], Shared["Subbase_thk"] = 0.0, 0.0, 0.0, 0.0
    else:
        # dict-shaped Best
        if isinstance(Best, dict) and Best.get('_type') == 'DataFrame' and Best.get('data'):
            row = Best['data'][0]
            Shared["BT_thk"] = float(row.get('BT', 0) or 0)
            Shared["CRL_thk"] = float(row.get('CRL', 0) or 0)
            Shared["Base_thk"] = float(row.get('Base', 0) or 0)
            Shared["Subbase_thk"] = float(row.get('Subbase', 0) or 0)
            if 'Cost' in row:
                Shared["Cost"] = float(row.get('Cost') or 0)
            if 'Nbf' in row:
                Shared["Nbf"] = float(row.get('Nbf') or 0)
            if 'Nsbr' in row:
                Shared["Nsbr"] = float(row.get('Nsbr') or 0)
            if 'Nctf' in row:
                Shared["Nctf"] = float(row.get('Nctf') or 0)
            if 'CFD' in row:
                Shared["CFD"] = float(row.get('CFD') or 0)
        else:
            Shared["BT_thk"], Shared["CRL_thk"], Shared["Base_thk"], Shared["Subbase_thk"] = 0.0, 0.0, 0.0, 0.0

    # Subgrade modulus from CBR
    CBR = Shared["Effective_Subgrade_CBR"]
    Shared["Subgrade_mod"] = (CBR * 10) if CBR <= 5 else 17.6 * (CBR ** 0.64)
    Shared["Subgrade_thk"] = float("inf")
    Shared["Subgrade_v"] = 0.35

    # Poisson’s ratios
    Shared["BT_v"] = 0.35
    Shared["In_v"] = 0.35
    Shared["Base_v"] = 0.25 if Type in [2, 3, 4, 5] else 0.35
    Shared["Subbase_v"] = 0.25 if Type in [2, 3, 4, 6, 8] else 0.35

    # Interlayer modulus
    if Type == 3:
        Shared["In_mod"] = 0
    elif Shared["CRL_thk"] > 0:
        Shared["In_mod"] = AIL_Mod_UI
    else:
        Shared["In_mod"] = 0

    granularE = lambda hs, Es: 0.2 * (hs ** 0.45) * Es  # noqa

    # Type-wise modulus
    if Type == 1:
        Eg = granularE((Shared["Base_thk"] + Shared["Subbase_thk"]), Shared["Subgrade_mod"])
        Shared["Base_mod"] = Eg
        Shared["Subbase_mod"] = Eg
    elif Type == 2:
        Shared["Base_mod"] = CTB_Mod_UI
        Shared["Subbase_mod"] = CTSB_Mod_UI
    elif Type == 3:
        Shared["Base_mod"] = CTB_Mod_UI
        Shared["Subbase_mod"] = CTSB_Mod_UI
    elif Type == 4:
        Shared["Base_mod"] = ETB_Mod_UI
        Shared["Subbase_mod"] = CTSB_Mod_UI
    elif Type == 5:
        Shared["Base_mod"] = CTB_Mod_UI
        Shared["Subbase_mod"] = granularE(Shared["Subbase_thk"], Shared["Subgrade_mod"])
    elif Type == 6:
        Shared["Base_mod"] = WMM_Mod_UI
        Shared["Subbase_mod"] = CTSB_Mod_UI
    elif Type == 7:
        Subbase_mod = 0.2 * (Shared["Subbase_thk"] ** 0.45) * Shared["Subgrade_mod"]
        nE = 2; ThicknessE = Shared["Subbase_thk"]; EE = [Subbase_mod, Shared["Subgrade_mod"]]; vE = [0.35, 0.35]
        EMr = AIO_EffectiveMr(nE, ThicknessE, EE, vE)
        Base_mod = 0.2 * (Shared["Base_thk"] ** 0.45) * EMr
        if Rtype_UI == 1:
            BaseR_mod = Base_mod * R_Base_UI
            SubbaseR_mod = Subbase_mod * R_Subbase_UI
        else:
            a2 = 0.249 * (math.log10(Base_mod * 145.038)) - 0.977
            a3 = 0.227 * (math.log10(Subbase_mod * 145.038)) - 0.839
            BaseR_mod = (10 ** ((0.977 + R_Base_UI * a2) / 0.249)) / 145.038
            SubbaseR_mod = 10 ** ((0.839 + R_Subbase_UI * a3) / 0.227) / 145.038
        Shared["Base_mod"] = BaseR_mod
        Shared["Subbase_mod"] = SubbaseR_mod
    elif Type == 8:
        nE = 2; ThicknessE = Shared["Subbase_thk"]; EE = [CTSB_Mod_UI, Shared["Subgrade_mod"]]; vE = [0.25, 0.35]
        EMr = AIO_EffectiveMr(nE, ThicknessE, EE, vE)
        Base_mod = min(WMM_Mod_UI, 0.2 * (Shared["Base_thk"] ** 0.45) * EMr)
        if Rtype_UI == 1:
            BaseR_mod = Base_mod * R_Base_UI
        else:
            a2 = 0.249 * (math.log10(Base_mod * 145.038)) - 0.977
            BaseR_mod = (10 ** ((0.977 + R_Base_UI * a2) / 0.249)) / 145.038
        Shared["Base_mod"] = BaseR_mod
        Shared["Subbase_mod"] = CTSB_Mod_UI

    # Layer names
    baseName, subName = layer_names_by_type(Type, bool(is_wmm_r_UI), bool(is_gsb_r_UI))
    if Type == 3:
        Shared["Interlayer_name"] = "CRL (SAMI)"
    elif (Type in [2, 5]) and Shared["CRL_thk"] > 0:
        Shared["Interlayer_name"] = "CRL (AIL)"
    else:
        Shared["Interlayer_name"] = ""
    Shared["Bituminous_name"] = "Bituminous (BC+DBM)"
    Shared["Subgrade_name"] = "Subgrade"
    Shared["Layer_Names"] = [Shared["Bituminous_name"], Shared["Interlayer_name"], baseName, subName, Shared["Subgrade_name"]]

    print(
        f"Type={Type} | BT={Shared['BT_thk']} mm | Base={Shared['Base_thk']} mm | Subbase={Shared['Subbase_thk']} mm | "
        f"Ebase={Shared.get('Base_mod', float('nan'))} MPa | Esub={Shared.get('Subbase_mod', float('nan'))} MPa",
        file=sys.stderr,
    )

    return Best, ResultsTable, Shared, TRACE, T, hFig


# ============================================================
# ======================= CORE SOLVER ========================
# ============================================================
def unified_pavement_design(
    Type: int,
    Design_Traffic: float,
    Effective_Subgrade_CBR: float,
    Reliability: float,
    Va: float,
    Vbe: float,
    BT_Mod: float,
    Rtype: Optional[int],
    R_Base: Optional[float],
    R_Subbase: Optional[float],
    RF_CTB: Optional[float],
    cfdchk: Optional[int],
    FS_CTB: Optional[float],
    SA_M: np.ndarray,
    TaA_M: np.ndarray,
    TrA_M: np.ndarray,
    BC_cost: float,
    BC_DBM_width: float,
    DBM_cost: float,
    Base_cost: float,
    Base_width: float,
    Subbase_cost: float,
    Subbase_width: float,
    CRL_cost: float,
    CRL_width: float,
    SAMI_cost: float,
    wmm_r_cost: float,
    gsb_r_cost: float,
    is_wmm_r: bool,
    is_gsb_r: bool,
    ETB_Mod: Optional[float],
    CTSB_Mod: Optional[float],
    WMM_Mod: Optional[float],
    AIL_Mod: Optional[float],
    CTB_Mod: Optional[float],
):
    # ------------------------- DERIVED INPUTS -------------------------
    N = Design_Traffic
    N_design = N * 1e6
    Subgrade_CBR = Effective_Subgrade_CBR
    reliability = Reliability
    Vbec = Vbe
    Vac = Va

    # ------------------------- MODULI & FLAGS BY TYPE -------------------------
    hasCRL = False
    Base_Modulus = None
    Subbase_Modulus = None
    CRL_Modulus = 0

    if Type == 1:
        pass
    elif Type == 4:
        Base_Modulus = ETB_Mod; Subbase_Modulus = CTSB_Mod
    elif Type == 6:
        Base_Modulus = WMM_Mod; Subbase_Modulus = CTSB_Mod
    elif Type == 7:
        pass
    elif Type == 2:
        hasCRL = True
        Base_Modulus = CTB_Mod; Subbase_Modulus = CTSB_Mod; CRL_Modulus = AIL_Mod
    elif Type == 3:
        Base_Modulus = CTB_Mod; Subbase_Modulus = CTSB_Mod
        hasCRL = False; CRL_Modulus = 0
    elif Type == 5:
        hasCRL = True
        Base_Modulus = CTB_Mod; Subbase_Modulus = None; CRL_Modulus = AIL_Mod
    elif Type == 8:
        Base_Modulus = WMM_Mod; Subbase_Modulus = CTSB_Mod
    else:
        raise ValueError("Unsupported Type")

    # ------------------------- THICKNESS LIMITS -------------------------
    if   Type == 1: Base_MThk, Subbase_MThk = 150, 150
    elif Type == 4: Base_MThk, Subbase_MThk = 100, 200
    elif Type == 6: Base_MThk, Subbase_MThk = 150, 200
    elif Type == 7: Base_MThk, Subbase_MThk = 150, 150
    elif Type == 2: Base_MThk, Subbase_MThk = 100, 200
    elif Type == 3: Base_MThk, Subbase_MThk = 100, 200
    elif Type == 5: Base_MThk, Subbase_MThk = 100, 150
    elif Type == 8: Base_MThk, Subbase_MThk = 150, 200
    Base_MXThk, Subbase_MXThk = 500, 300

    # ------------------------- INITIAL THICKNESSES -------------------------
    if Type in [2, 3, 5]:
        BC_Thk1, DBM_Thk1, CRL_Thk1 = 40, 110, (100 if hasCRL else 0)
    else:
        BC_Thk1, DBM_Thk1, CRL_Thk1 = 30, 120, 0
    Base_Thk1, Subbase_Thk1 = 200, 200
    BT_Thk1 = BC_Thk1 + DBM_Thk1

    # ------------------------- GLOBALS / STEPS -------------------------
    Btstep, Basestep, Subbasestep, CRLstep = 5, 5, 50, 10
    eqtol = 1e-9

    quant5 = lambda x: quant_step(x, 5)
    quantSub = lambda x: quant_step(x, Subbasestep)

    # ====================== MEMOIZATION SUPPORT ======================
    cache_eval: Dict[str, Any] = {}
    cache_cost: Dict[str, Any] = {}
    make_key_235 = lambda T, BT, CRL, BAS, SUB: f"{int(T)}|{int(BT)}|{int(CRL)}|{int(BAS)}|{int(SUB)}"
    make_key_1467 = lambda T, BT, BAS, SUB: f"{int(T)}|{int(BT)}|{int(BAS)}|{int(SUB)}"

    def clampQ_235(BT, CRL, Base, Subb, pol):
        BTq = pol["enforce_BT40band"](pol["enforce_BT_Ngt20"](pol["quant5"](BT)))
        CRLq = max(pol["quant5"](CRL), 0)
        Baseq = min(max(pol["quant5"](Base), pol["Base_MThk"]), pol["Base_MXThk"])
        Subbq = min(max(pol["quantSub"](Subb), pol["Sub_MThk"]), pol["Sub_MXThk"])
        return BTq, CRLq, Baseq, Subbq

    def clampQ_1467(BT, Base, Subb, pol):
        BTq = pol["enforce_BT40band"](pol["enforce_BT_Ngt20"](pol["quant5"](BT)))
        Baseq = min(max(pol["quant5"](Base), pol["Base_MThk"]), pol["Base_MXThk"])
        Subbq = min(max(pol["quantSub"](Subb), pol["Sub_MThk"]), pol["Sub_MXThk"])
        return BTq, Baseq, Subbq

    def eval_state_cached(Type_, thk, pars_, pol_):
        isT235 = pars_["isT235"]
        if isT235:
            BTq, CRLq, Baseq, Subbq = clampQ_235(thk[0], thk[1], thk[2], thk[3], pol_)
            key = make_key_235(Type_, BTq, CRLq, Baseq, Subbq)
        else:
            BTq, Baseq, Subbq = clampQ_1467(thk[0], thk[1], thk[2], pol_)
            key = make_key_1467(Type_, BTq, Baseq, Subbq)
            CRLq = 0
        if key in cache_eval:
            return cache_eval[key]

        if isT235:
            Crit, Perm = FN_Cri_Cal_BF_SR_FF(
                Type_,
                [BTq, CRLq, Baseq, Subbq],
                pars_["N_design"],
                pars_["Subgrade_CBR"],
                pars_["BT_Mod"],
                pars_["Va"],
                pars_["Vbe"],
                pars_["reliability"],
                pars_["Base_Modulus"],
                pars_["RF_CTB"],
                pars_["Subbase_Modulus"],
                pars_["CRL_Mod_pass"],
            )
            Crit_ms = np.array(Crit) * 1e6
            Perm_ms = np.array(Perm) * 1e6
            CFDv = 0.0
            if pars_["useCFD"]:
                CFDv = FN_Cri_Cal_CTB_CFD(
                    Type_,
                    [BTq, CRLq, Baseq, Subbq],
                    pars_["Subgrade_CBR"],
                    pars_["BT_Mod"],
                    pars_["Base_Modulus"],
                    pars_["Subbase_Modulus"],
                    pars_["CRL_Mod_pass"],
                    pars_["FS_CTB"],
                    pars_["SA_M"],
                    pars_["TaA_M"],
                    pars_["TrA_M"],
                )
            if key in cache_cost:
                Costv = cache_cost[key]
            else:
                Costv = cost_unified(
                    Type_,
                    BTq,
                    CRLq,
                    Baseq,
                    Subbq,
                    pars_["BC_DBM_width"],
                    pars_["BC_cost"],
                    pars_["DBM_cost"],
                    pars_["CRL_cost"],
                    pars_["CRL_width"],
                    pars_["Base_width"],
                    pars_["Base_cost"],
                    pars_["Subbase_width"],
                    pars_["Subbase_cost"],
                    pars_["SAMI_cost"],
                    pars_["wmm_r_cost"],
                    pars_["gsb_r_cost"],
                    pars_["is_wmm_r"],
                    pars_["is_gsb_r"],
                )
                cache_cost[key] = Costv
            SDbf = float(Perm_ms[0] - Crit_ms[0])
            SDsbr = float(Perm_ms[1] - Crit_ms[1])
            SDctf = float(Perm_ms[2] - Crit_ms[2])
            out = dict(BT=BTq, CRL=CRLq, Base=Baseq, Sub=Subbq, SDbf=SDbf, SDsbr=SDsbr, SDctf=SDctf, CFD=CFDv, Cost=float(Costv))
        else:
            Crit, Perm = AIO_N_Cri_Cal_BF_SR(
                Type_,
                [BTq, Baseq, Subbq],
                pars_["N_design"],
                pars_["Subgrade_CBR"],
                pars_["BT_Mod"],
                pars_["Va"],
                pars_["Vbe"],
                pars_["reliability"],
                pars_["Base_Modulus"],
                pars_["Subbase_Modulus"],
                pars_["Rtype"],
                pars_["R_Base"],
                pars_["R_Subbase"],
            )
            Crit_ms = np.array(Crit) * 1e6
            Perm_ms = np.array(Perm) * 1e6
            if key in cache_cost:
                Costv = cache_cost[key]
            else:
                Costv = cost_unified(
                    Type_,
                    BTq,
                    0,
                    Baseq,
                    Subbq,
                    pars_["BC_DBM_width"],
                    pars_["BC_cost"],
                    pars_["DBM_cost"],
                    pars_["CRL_cost"],
                    pars_["CRL_width"],
                    pars_["Base_width"],
                    pars_["Base_cost"],
                    pars_["Subbase_width"],
                    pars_["Subbase_cost"],
                    pars_["SAMI_cost"],
                    pars_["wmm_r_cost"],
                    pars_["gsb_r_cost"],
                    pars_["is_wmm_r"],
                    pars_["is_gsb_r"],
                )
                cache_cost[key] = Costv
            SDbf = float(Perm_ms[0] - Crit_ms[0])
            SDsbr = float(Perm_ms[1] - Crit_ms[1])
            out = dict(BT=BTq, CRL=0, Base=Baseq, Sub=Subbq, SDbf=SDbf, SDsbr=SDsbr, SDctf=float("nan"), CFD=0.0, Cost=float(Costv))

        cache_eval[key] = out
        return out

    # ------------------------- ENFORCERS / HELPERS -------------------------
    isT235 = (Type in [2, 3, 5])
    minBT100_active = (isT235 and N > 20)

    def enforce_BT40band(BT):
        if BT < 30:
            return 30
        if 30 <= BT <= 40:
            return BT
        if 40 < BT < 80:
            return 40
        return BT

    def enforce_BT_min100(BTq):
        return BTq if not minBT100_active else max(BTq, 100)

    def enforce_BT_Ngt20(BT):
        return enforce_BT_min100(BT)

    def apply_bounds_1467(BT, Base, Subb):
        return (
            enforce_BT40band(enforce_BT_Ngt20(quant5(BT))),
            min(max(quant5(Base), Base_MThk), Base_MXThk),
            min(max(quantSub(Subb), Subbase_MThk), Subbase_MXThk),
        )

    def apply_bounds_235(BT, CRL, Base, Subb):
        return (
            enforce_BT40band(enforce_BT_Ngt20(quant5(BT))),
            max(quant5(CRL), 0),
            min(max(quant5(Base), Base_MThk), Base_MXThk),
            min(max(quantSub(Subb), Subbase_MThk), Subbase_MXThk),
        )

    useCFD = isT235 and (cfdchk == 1)

    # ------------------------- PACK PARS/POL -------------------------
    pars = dict(
        Type=Type,
        isT235=isT235,
        useCFD=useCFD,
        N_design=N_design,
        Subgrade_CBR=Subgrade_CBR,
        BT_Mod=BT_Mod,
        Va=Va,
        Vbe=Vbe,
        reliability=reliability,
        Base_Modulus=Base_Modulus,
        RF_CTB=RF_CTB,
        Subbase_Modulus=Subbase_Modulus,
        CRL_Mod_pass=CRL_Modulus if hasCRL else 0,
        FS_CTB=FS_CTB,
        SA_M=SA_M,
        TaA_M=TaA_M,
        TrA_M=TrA_M,
        BC_DBM_width=BC_DBM_width,
        BC_cost=BC_cost,
        DBM_cost=DBM_cost,
        CRL_width=CRL_width,
        CRL_cost=CRL_cost,
        Base_width=Base_width,
        Base_cost=Base_cost,
        Subbase_width=Subbase_width,
        Subbase_cost=Subbase_cost,
        SAMI_cost=SAMI_cost,
        wmm_r_cost=wmm_r_cost,
        gsb_r_cost=gsb_r_cost,
        is_wmm_r=bool(is_wmm_r),
        is_gsb_r=bool(is_gsb_r),
        Rtype=Rtype,
        R_Base=R_Base,
        R_Subbase=R_Subbase,
    )

    # ------------------------- POLICIES / HELPERS -------------------------
    # NOTE: these replicate the MATLAB lambdas exactly
    def quantS(x, s):
        return s * round_half_away_from_zero(x / s)

    quant5 = lambda x: quantS(x, 5)
    quantSub = lambda x: quantS(x, Subbasestep)  # enforce 50-mm grid for Subbase

    minBT100_active = (isT235 and (Design_Traffic > 20))

    def enforce_BT40band(BT):
        # ((BT<30)*30 + (30<=BT<=40)*BT + (40<BT<80)*40 + (BT>=80)*BT)
        if BT < 30:
            return 30
        if BT <= 40:
            return BT
        if BT < 80:
            return 40
        return BT

    def enforce_BT_min100(BT):
        if not minBT100_active:
            return BT
        return max(BT, 100)

    enforce_BT_Ngt20 = enforce_BT_min100

    def clampQ_1467(BT, Base, Subb):
        BTq = enforce_BT40band(enforce_BT_Ngt20(quant5(BT)))
        Baseq = min(max(quant5(Base), Base_MThk), Base_MXThk)
        Subbq = min(max(quantSub(Subb), Subbase_MThk), Subbase_MXThk)  # <<< 50-mm
        return BTq, Baseq, Subbq

    def clampQ_235(BT, CRL, Base, Subb):
        BTq = enforce_BT40band(enforce_BT_Ngt20(quant5(BT)))
        CRLq = max(quant5(CRL), 0)
        Baseq = min(max(quant5(Base), Base_MThk), Base_MXThk)
        Subbq = min(max(quantSub(Subb), Subbase_MThk), Subbase_MXThk)  # <<< 50-mm
        return BTq, CRLq, Baseq, Subbq

    pol = dict(
        quant5=quant5,
        quantSub=quantSub,
        enforce_BT40band=enforce_BT40band,
        enforce_BT_Ngt20=enforce_BT_Ngt20,
        Base_MThk=Base_MThk,
        Base_MXThk=Base_MXThk,
        Sub_MThk=Subbase_MThk,
        Sub_MXThk=Subbase_MXThk,
        BT_step=Btstep,
        Base_step=Basestep,
        Sub_step=Subbasestep,
    )

    # ====================== MEMOIZATION SUPPORT ======================
    cache_eval: dict[str, dict] = {}
    cache_cost: dict[str, float] = {}

    def make_key_235(T, BT, CRL, BAS, SUB):
        return f"{T}|{int(BT)}|{int(CRL)}|{int(BAS)}|{int(SUB)}"

    def make_key_1467(T, BT, BAS, SUB):
        return f"{T}|{int(BT)}|{int(BAS)}|{int(SUB)}"

    def cost_cached(BTq, CRLq, Baseq, Subbq):
        if isT235:
            key = make_key_235(pars["Type"], BTq, CRLq, Baseq, Subbq)
        else:
            key = make_key_1467(pars["Type"], BTq, Baseq, Subbq)
        if key in cache_cost:
            return cache_cost[key]
        c = cost_unified(
            pars["Type"], BTq, (CRLq if isT235 else 0), Baseq, Subbq,
            pars["BC_DBM_width"], pars["BC_cost"], pars["DBM_cost"],
            pars["CRL_cost"], pars["CRL_width"],
            pars["Base_width"], pars["Base_cost"],
            pars["Subbase_width"], pars["Subbase_cost"],
            pars["SAMI_cost"], pars["wmm_r_cost"], pars["gsb_r_cost"],
            pars["is_wmm_r"], pars["is_gsb_r"]
        )
        cache_cost[key] = c
        return c

    def eval_state_cached(thk_tuple):
        """
        Returns dict with BT, CRL, Base, Sub, SDbf, SDsbr, SDctf, CFD, Cost
        (mirrors MATLAB struct packing)
        """
        if isT235:
            BT, CRL, Base, Subb = clampQ_235(*thk_tuple)
            key = make_key_235(pars["Type"], BT, CRL, Base, Subb)
        else:
            BT, Base, Subb = clampQ_1467(*thk_tuple)
            CRL = 0
            key = make_key_1467(pars["Type"], BT, Base, Subb)

        if key in cache_eval:
            return cache_eval[key]

        if isT235:
            Crit, Perm = FN_Cri_Cal_BF_SR_FF(
                pars["Type"], [BT, CRL, Base, Subb], pars["N_design"], pars["Subgrade_CBR"],
                pars["BT_Mod"], pars["Va"], pars["Vbe"], pars["reliability"],
                pars["Base_Modulus"], pars["RF_CTB"], pars["Subbase_Modulus"], pars["CRL_Mod_pass"]
            )
            Crit_ms = [c * 1e6 for c in Crit]
            Perm_ms = [p * 1e6 for p in Perm]
            CFDv = 0.0
            if pars["useCFD"]:
                CFDv = FN_Cri_Cal_CTB_CFD(
                    pars["Type"], [BT, CRL, Base, Subb], pars["Subgrade_CBR"], pars["BT_Mod"],
                    pars["Base_Modulus"], pars["Subbase_Modulus"], pars["CRL_Mod_pass"],
                    pars["FS_CTB"], pars["SA_M"], pars["TaA_M"], pars["TrA_M"]
                )
            Costv = cost_cached(BT, CRL, Base, Subb)
            SDbf = Perm_ms[0] - Crit_ms[0]
            SDsbr = Perm_ms[1] - Crit_ms[1]
            SDctf = Perm_ms[2] - Crit_ms[2]
            out = dict(BT=BT, CRL=CRL, Base=Base, Sub=Subb,
                       SDbf=SDbf, SDsbr=SDsbr, SDctf=SDctf, CFD=CFDv, Cost=Costv)
        else:
            Crit, Perm = AIO_N_Cri_Cal_BF_SR(
                pars["Type"], [BT, Base, Subb], pars["N_design"], pars["Subgrade_CBR"],
                pars["BT_Mod"], pars["Va"], pars["Vbe"], pars["reliability"],
                pars["Base_Modulus"], pars["Subbase_Modulus"],
                pars["Rtype"], pars["R_Base"], pars["R_Subbase"]
            )
            Crit_ms = [c * 1e6 for c in Crit]
            Perm_ms = [p * 1e6 for p in Perm]
            Costv = cost_cached(BT, 0, Base, Subb)
            SDbf = Perm_ms[0] - Crit_ms[0]
            SDsbr = Perm_ms[1] - Crit_ms[1]
            out = dict(BT=BT, CRL=0, Base=Base, Sub=Subb,
                       SDbf=SDbf, SDsbr=SDsbr, SDctf=float("nan"), CFD=0.0, Cost=Costv)

        cache_eval[key] = out
        return out

    # ------------------------- COARSE FEASIBILITY SEARCH -------------------------
    maxItA = 100
    BT_Thk_N = [math.nan] * maxItA
    Base_Thk_N = [math.nan] * maxItA
    Subbase_Thk_N = [math.nan] * maxItA
    CRL_Thk_N = [math.nan] * maxItA if isT235 else None

    D_BT = [0.0] * (maxItA + 2)
    D_Base = [0.0] * (maxItA + 2)
    D_Subbase = [0.0] * (maxItA + 2)

    Sidiff_BTF = [math.nan] * maxItA
    Sidiff_SBR = [math.nan] * maxItA
    Sidiff_CTF = [math.nan] * maxItA if isT235 else None
    CFD_N = [math.nan] * maxItA if isT235 else None

    def iff(cond, a, b):
        return a if cond else b

    for it in range(1, maxItA + 1):
        k = it - 1  # 0-based index
        # propagate thicknesses
        if it == 1:
            BT_Thk_N[k] = BT_Thk1
            Base_Thk_N[k] = Base_Thk1
            Subbase_Thk_N[k] = Subbase_Thk1
            if isT235:
                CRL_Thk_N[k] = (0 if Type == 3 else CRL_Thk1)
        else:
            BT_Thk_N[k] = BT_Thk_N[k - 1] + D_BT[it]
            Base_Thk_N[k] = Base_Thk_N[k - 1] + D_Base[it]
            Subbase_Thk_N[k] = Subbase_Thk_N[k - 1] + D_Subbase[it]
            if isT235:
                CRL_Thk_N[k] = CRL_Thk_N[k - 1]

        # policy & limits
        BT_Thk_N[k] = enforce_BT40band(enforce_BT_Ngt20(BT_Thk_N[k]))
        Base_Thk_N[k] = min(max(quant5(Base_Thk_N[k]), Base_MThk), Base_MXThk)
        Subbase_Thk_N[k] = min(max(quantSub(Subbase_Thk_N[k]), Subbase_MThk), Subbase_MXThk)
        if isT235 and Type == 3:
            CRL_Thk_N[k] = 0

        # evaluate
        if isT235:
            st = eval_state_cached((BT_Thk_N[k], CRL_Thk_N[k], Base_Thk_N[k], Subbase_Thk_N[k]))
            Sidiff_BTF[k] = st["SDbf"]
            Sidiff_SBR[k] = st["SDsbr"]
            Sidiff_CTF[k] = st["SDctf"]
            CFD_N[k] = st["CFD"]
        else:
            st = eval_state_cached((BT_Thk_N[k], Base_Thk_N[k], Subbase_Thk_N[k]))
            Sidiff_BTF[k] = st["SDbf"]
            Sidiff_SBR[k] = st["SDsbr"]

        # governing difference for Base/Sub
        if isT235:
            bucket = [Sidiff_SBR[k]]
            if not (Sidiff_CTF[k] is None or math.isnan(Sidiff_CTF[k])):
                bucket.append(Sidiff_CTF[k])
            if useCFD and CFD_N[k] is not None and not math.isnan(CFD_N[k]):
                bucket.append(1 - CFD_N[k])
            bucket = [b for b in bucket if b is not None and not math.isnan(b)]
            Ndiff_GOV = Sidiff_SBR[k] if len(bucket) == 0 else min(bucket)
        else:
            Ndiff_GOV = Sidiff_SBR[k]

        # controllers
        if isT235:
            ok_CTF = (Sidiff_CTF[k] is not None) and (Sidiff_CTF[k] > 0)
            ok_CFD = (not useCFD) or (CFD_N[k] < 1)
            if not (ok_CTF and ok_CFD):
                D_BT[it + 1] = (Btstep if Sidiff_BTF[k] < 0 else 0)
            else:
                D_BT[it + 1] = (-Btstep if Sidiff_BTF[k] > 0 else (Btstep if Sidiff_BTF[k] < 0 else 0))
        else:
            D_BT[it + 1] = (-Btstep if Sidiff_BTF[k] > 0 else (Btstep if Sidiff_BTF[k] < 0 else 0))

        if (it % 2) == 0:
            D_Subbase[it + 1] = (-Subbasestep if Ndiff_GOV > 0 else (Subbasestep if Ndiff_GOV < 0 else 0))
        else:
            D_Subbase[it + 1] = 0

        if (it % 2) == 1:
            D_Base[it + 1] = (-Basestep if Ndiff_GOV > 0 else (Basestep if Ndiff_GOV < 0 else 0))
        else:
            D_Base[it + 1] = 0

        if isT235 and useCFD and CFD_N[k] is not None and CFD_N[k] >= 1:
            D_Base[it + 1] = max(D_Base[it + 1], Basestep)

        # early exits (index guards)
        ok_CTF = (not isT235) or (Sidiff_CTF[k] is not None and Sidiff_CTF[k] > 0)
        ok_CFD = (not useCFD) or (CFD_N[k] < 1 if isT235 else True)
        if it > 20:
            if (Sidiff_BTF[k - 2] < 0 and Sidiff_BTF[k - 1] < 0) and \
               (Sidiff_BTF[k] > 0 and Sidiff_SBR[k] > 0 and ok_CTF and ok_CFD):
                break
        if it > 20 and Sidiff_BTF[k] > 0 and Sidiff_SBR[k] > 0 and ok_CTF and ok_CFD:
            if abs(Sidiff_BTF[k - 4] - Sidiff_BTF[k]) < 1e-12 and \
               abs(Sidiff_SBR[k - 4] - Sidiff_SBR[k]) < 1e-12:
                break

    # ---- Build AAA: [Nbf Nsbr Nctf CFD BT CRL Base Sub]
    rows_AAA = []
    for i in range(maxItA):
        if math.isnan(Sidiff_BTF[i]) or math.isnan(Sidiff_SBR[i]):
            continue
        if isT235:
            if math.isnan(Sidiff_CTF[i]) or math.isnan(CFD_N[i]) or math.isnan(BT_Thk_N[i]) or math.isnan(Base_Thk_N[i]) or math.isnan(Subbase_Thk_N[i]):
                continue
            if Sidiff_BTF[i] > 0 and Sidiff_SBR[i] > 0 and Sidiff_CTF[i] > 0 and (not useCFD or CFD_N[i] < 1):
                rows_AAA.append([
                    Sidiff_BTF[i], Sidiff_SBR[i], Sidiff_CTF[i], CFD_N[i],
                    BT_Thk_N[i], CRL_Thk_N[i], Base_Thk_N[i], Subbase_Thk_N[i]
                ])
        else:
            if Sidiff_BTF[i] > 0 and Sidiff_SBR[i] > 0:
                rows_AAA.append([
                    Sidiff_BTF[i], Sidiff_SBR[i], math.nan, 0.0,
                    BT_Thk_N[i], 0.0, Base_Thk_N[i], Subbase_Thk_N[i]
                ])

    if len(rows_AAA) == 0:
        warnings.warn("No feasible coarse options (AAA). Adjust minima or increase BT/Base/Subbase.")
        Best = pd.DataFrame()
        T = pd.DataFrame()
        TRACE = np.empty((0, 9))
        return Best, T, TRACE

    AAA = np.array(rows_AAA, dtype=float)
    # cost AAA via cache; append as col 9 and sort
    costs = []
    for r in AAA:
        BTi, CRLi, Basei, Subi = r[4], r[5], r[6], r[7]
        costs.append(cost_cached(BTi, CRLi, Basei, Subi))
    AAAc = np.column_stack([AAA, np.array(costs)])
    # unique rows
    AAAc = np.unique(np.round(AAAc, 12), axis=0)
    AAAc = AAAc[np.argsort(AAAc[:, 8])]  # sort by cost col 9 (0-based idx 8)

    # ------------------------- LOCAL NEIGHBORHOOD (AANZ) -------------------------
    BT0, CRL0, Base0, Sub0 = AAAc[0, 4], AAAc[0, 5], AAAc[0, 6], AAAc[0, 7]
    BT_steps = np.array([-Btstep, 0, Btstep])
    Base_steps = np.array([-Basestep, 0, Basestep])
    Sub_steps = np.array([-Subbasestep, 0, Subbasestep])
    CRL_steps = np.array([0])  # keep CRL fixed by default

    AANZ = []
    for dBT in BT_steps:
        for dB in Base_steps:
            for dS in Sub_steps:
                if isT235:
                    for dC in CRL_steps:
                        BT = BT0 + dBT;
                        CRL = CRL0 + dC;
                        Base = Base0 + dB;
                        Subb = Sub0 + dS
                        BT, CRL, Base, Subb = clampQ_235(BT, CRL, Base, Subb)
                        if Type == 3:
                            CRL = 0.0
                        st = eval_state_cached((BT, CRL, Base, Subb))
                        if not (st["SDbf"] > 0 and st["SDsbr"] > 0 and st["SDctf"] > 0 and (
                                not useCFD or st["CFD"] < 1)):
                            continue
                        AANZ.append([st["BT"], st["CRL"], st["Base"], st["Sub"],
                                     st["SDbf"], st["SDsbr"], st["SDctf"], st["CFD"], st["Cost"]])
                else:
                    BT = BT0 + dBT;
                    Base = Base0 + dB;
                    Subb = Sub0 + dS
                    BT, Base, Subb = clampQ_1467(BT, Base, Subb)
                    st = eval_state_cached((BT, Base, Subb))
                    if not (st["SDbf"] > 0 and st["SDsbr"] > 0):
                        continue
                    AANZ.append([st["BT"], 0.0, st["Base"], st["Sub"],
                                 st["SDbf"], st["SDsbr"], math.nan, 0.0, st["Cost"]])

    # If nothing in the local neighborhood, seed with the cheapest AAA candidate
    if len(AANZ) == 0:
        warnings.warn("AANZ empty; using cheapest AAA only.")
        # AAAc columns: [Nbf Nsbr Nctf CFD BT CRL Base Sub Cost]
        AANZ = [[AAAc[0, 4], AAAc[0, 5], AAAc[0, 6], AAAc[0, 7],
                 AAAc[0, 0], AAAc[0, 1], AAAc[0, 2], AAAc[0, 3], AAAc[0, 8]]]

    # Dedup/round; ensure 2D with expected 9 columns
    AANZ = np.unique(np.round(np.array(AANZ, dtype=float), 12), axis=0)
    AANZ = np.asarray(AANZ, dtype=float).reshape(-1, 9)

    # Feasibility mask (robust for single-row)
    if isT235:
        mask = (AANZ[:, 4] > 0) & (AANZ[:, 5] > 0) & (AANZ[:, 6] > 0) & ((~useCFD) | (AANZ[:, 7] < 1))
    else:
        mask = (AANZ[:, 4] > 0) & (AANZ[:, 5] > 0)
    mask = np.asarray(mask, dtype=bool).ravel()
    AANZ = AANZ[mask]

    # Fallback if filtering removed everything
    if AANZ.size == 0:
        warnings.warn("AANZ empty after filtering; using cheapest AAA candidate.")
        # AAAc columns: [Nbf Nsbr Nctf CFD BT CRL Base Sub Cost]
        AANZ = np.array([[AAAc[0, 4], AAAc[0, 5], AAAc[0, 6], AAAc[0, 7],
                          AAAc[0, 0], AAAc[0, 1], AAAc[0, 2], AAAc[0, 3], AAAc[0, 8]]], dtype=float)

    # Sort by Cost
    AANZ = AANZ[np.argsort(AANZ[:, 8])]

    # ------------------------- DIRECTIONAL DESCENT -------------------------
    varyCRL = isT235 and hasCRL and not np.allclose(AANZ[:, 1], 0)

    BTc, CRLc, Basec, Subc = AANZ[0, 0], AANZ[0, 1], AANZ[0, 2], AANZ[0, 3]
    Nbf_c, Nsbr_c, Nctf_c, CFD_c, Costc = AANZ[0, 4], AANZ[0, 5], AANZ[0, 6], AANZ[0, 7], AANZ[0, 8]
    TRACE = [[BTc, CRLc, Basec, Subc, Nbf_c, Nsbr_c, Nctf_c, CFD_c, Costc]]

    eqtol = 1e-9
    tolC = 1e-9
    max_iter = 60

    def step_mult(d, step):
        return (abs(d) < eqtol) or (abs((d % step)) < eqtol)

    def steps_ok(BT0, BT1, Ba0, Ba1, Su0, Su1, CRL0, CRL1):
        return step_mult(BT1 - BT0, Btstep) and \
               step_mult(Ba1 - Ba0, Basestep) and \
               step_mult(Su1 - Su0, Subbasestep) and \
               step_mult(CRL1 - CRL0, 10)  # CRLstep = 10 in MATLAB

    def feas_ok(row):
        if isT235:
            return (row[4] > 0 and row[5] > 0 and row[6] > 0) and ((not useCFD) or (row[7] < 1))
        else:
            return (row[4] > 0 and row[5] > 0)

    def unit_permm(layer_width, layer_cost):
        # ₹ lakh per mm per km
        return (layer_width * layer_cost) / 100000.0

    def unit_permm_BT(BT):
        # marginal BT cost per mm obeying BC/DBM split
        if BT <= 40:
            return unit_permm(pars["BC_DBM_width"], pars["BC_cost"])
        else:
            return unit_permm(pars["BC_DBM_width"], pars["DBM_cost"])

    def ensure_feasible(Type_, BT, CRL, Base, Subb):
        SDbf, SDsbr, SDctf, CFDv, Costv = eval_NR_COST(Type_, (BT, CRL, Base, Subb))
        ok_ = (SDbf > 0 and SDsbr > 0 and SDctf > 0 and ((not useCFD) or CFDv < 1))
        return ok_, Costv, (SDbf, SDsbr, SDctf, CFDv)

    def ensure_feasible1467(Type_, BT, Base, Subb):
        SDbf, SDsbr, SDctf, CFDv, Costv = eval_NR_COST(Type_, (BT, 0, Base, Subb))
        ok_ = (SDbf > 0 and SDsbr > 0)
        return ok_, Costv, (SDbf, SDsbr, math.nan, 0.0)

    def eval_NR_COST(Type_, thkVec):
        BT = thkVec[0]
        if isT235:
            CRL, Base, Subb = thkVec[1], thkVec[2], thkVec[3]
            st = eval_state_cached((BT, CRL, Base, Subb))
            return st["SDbf"], st["SDsbr"], st["SDctf"], st["CFD"], st["Cost"]
        else:
            Base, Subb = thkVec[1], thkVec[2]
            st = eval_state_cached((BT, Base, Subb))
            return st["SDbf"], st["SDsbr"], float("nan"), 0.0, st["Cost"]

    def trade_BT_up(Type_, BTc_, CRLc_, Basec_, Subc_, Btstep_):
        BTn_ = BTc_ + Btstep_
        if isT235:
            CRLn_ = CRLc_
            BTn_, CRLn_, Basen_, Subbn_ = clampQ_235(BTn_, CRLn_, Basec_, Subc_)
            ok_, Costn_, _ = ensure_feasible(Type_, BTn_, CRLn_, Basen_, Subbn_)
            if not ok_:
                return BTc_, CRLc_, Basec_, Subc_, math.inf, False
        else:
            BTn_, Basen_, Subbn_ = clampQ_1467(BTn_, Basec_, Subc_)
            ok_, Costn_, _ = ensure_feasible1467(Type_, BTn_, Basen_, Subbn_)
            if not ok_:
                return BTc_, 0.0, Basec_, Subc_, math.inf, False
            CRLn_ = 0.0

        uBT = unit_permm_BT(BTn_)
        uBase = unit_permm(pars["Base_width"], pars["Base_cost"])
        uSub = unit_permm(pars["Subbase_width"], pars["Subbase_cost"])

        # shave Base/Sub in descending per-mm cost order if it
        # reduces *total* cost while keeping feasibility.
        max_it = 100
        Costn = Costn_
        Basen = Basen_
        Subbn = Subbn_
        for _ in range(max_it):
            # try the more expensive layer first
            pairs = [("base", uBase), ("sub", uSub)]
            pairs.sort(key=lambda t: -t[1])
            shaved = False

            for which, _u in pairs:
                if which == "base":
                    if Basen <= Base_MThk:
                        continue
                    B_try, S_try = Basen - Basestep, Subbn
                else:
                    if Subbn <= Subbase_MThk:
                        continue
                    B_try, S_try = Basen, Subbn - Subbasestep

                if isT235:
                    BTt, CRLt, Bt, St = clampQ_235(BTn_, CRLn_, B_try, S_try)
                    feas, Ct, _ = ensure_feasible(Type_, BTt, CRLt, Bt, St)
                else:
                    BTt, Bt, St = clampQ_1467(BTn_, B_try, S_try)
                    feas, Ct, _ = ensure_feasible1467(Type_, BTt, Bt, St)

                if not feas:
                    continue
                if Ct + 1e-12 < Costn:
                    Basen, Subbn, Costn = Bt, St, Ct
                    shaved = True
                    break

            if not shaved:
                break

        return BTn_, CRLn_, Basen, Subbn, Costn, True

        # ===== Directional descent (unified) =====

    varyCRL = isT235 and hasCRL and (not np.allclose(AANZ[:, 1], 0))

    BTc, CRLc, Basec, Subc = AANZ[0, 0], AANZ[0, 1], AANZ[0, 2], AANZ[0, 3]
    Nbf_c, Nsbr_c, Nctf_c, CFD_c, Costc = AANZ[0, 4], AANZ[0, 5], AANZ[0, 6], AANZ[0, 7], AANZ[0, 8]
    TRACE = [[BTc, CRLc, Basec, Subc, Nbf_c, Nsbr_c, Nctf_c, CFD_c, Costc]]

    tolC = 1e-9
    max_iter = 60

    for _ in range(max_iter):
        # neighbor masks w.r.t current point
        if varyCRL:
            maskBT = (np.isclose(AANZ[:, 1], CRLc, atol=eqtol) &
                      np.isclose(AANZ[:, 2], Basec, atol=eqtol) &
                      np.isclose(AANZ[:, 3], Subc, atol=eqtol) &
                      (~np.isclose(AANZ[:, 0], BTc, atol=eqtol)))
            maskCRL = (np.isclose(AANZ[:, 0], BTc, atol=eqtol) &
                       np.isclose(AANZ[:, 2], Basec, atol=eqtol) &
                       np.isclose(AANZ[:, 3], Subc, atol=eqtol) &
                       (~np.isclose(AANZ[:, 1], CRLc, atol=eqtol)))
            maskBAS = (np.isclose(AANZ[:, 0], BTc, atol=eqtol) &
                       np.isclose(AANZ[:, 1], CRLc, atol=eqtol) &
                       np.isclose(AANZ[:, 3], Subc, atol=eqtol) &
                       (~np.isclose(AANZ[:, 2], Basec, atol=eqtol)))
            maskSUB = (np.isclose(AANZ[:, 0], BTc, atol=eqtol) &
                       np.isclose(AANZ[:, 1], CRLc, atol=eqtol) &
                       np.isclose(AANZ[:, 2], Basec, atol=eqtol) &
                       (~np.isclose(AANZ[:, 3], Subc, atol=eqtol)))
        else:
            maskBT = (np.isclose(AANZ[:, 2], Basec, atol=eqtol) &
                      np.isclose(AANZ[:, 3], Subc, atol=eqtol) &
                      (~np.isclose(AANZ[:, 0], BTc, atol=eqtol)))
            maskBAS = (np.isclose(AANZ[:, 0], BTc, atol=eqtol) &
                       np.isclose(AANZ[:, 3], Subc, atol=eqtol) &
                       (~np.isclose(AANZ[:, 2], Basec, atol=eqtol)))
            maskSUB = (np.isclose(AANZ[:, 0], BTc, atol=eqtol) &
                       np.isclose(AANZ[:, 2], Basec, atol=eqtol) &
                       (~np.isclose(AANZ[:, 3], Subc, atol=eqtol)))
            maskCRL = np.zeros_like(maskBT, dtype=bool)

        sBT = sCRL = sBase = sSub = np.nan
        if np.any(maskBT):
            dC = AANZ[maskBT, 8] - Costc
            dX = AANZ[maskBT, 0] - BTc
            sBT = np.median(dC / dX)
        if varyCRL and np.any(maskCRL):
            dC = AANZ[maskCRL, 8] - Costc
            dX = AANZ[maskCRL, 1] - CRLc
            sCRL = np.median(dC / dX)
        if np.any(maskBAS):
            dC = AANZ[maskBAS, 8] - Costc
            dX = AANZ[maskBAS, 2] - Basec
            sBase = np.median(dC / dX)
        if np.any(maskSUB):
            dC = AANZ[maskSUB, 8] - Costc
            dX = AANZ[maskSUB, 3] - Subc
            sSub = np.median(dC / dX)

        # OLS fallback (rank-safe)
        X = np.column_stack([AANZ[:, 0:4], np.ones((AANZ.shape[0], 1))])
        y = AANZ[:, 8]
        vary_cols = np.any(np.abs(X - X[0, :]) > 1e-12, axis=0)
        vary_cols[-1] = True
        Xr = X[:, vary_cols]
        if Xr.size > 0 and Xr.shape[0] >= Xr.shape[1]:
            try:
                br, *_ = np.linalg.lstsq(Xr, y, rcond=None)
            except np.linalg.LinAlgError:
                br = np.zeros(Xr.shape[1])
            b = np.zeros(X.shape[1])
            b[vary_cols] = br
        else:
            try:
                b = X.T @ np.linalg.pinv(X @ X.T) @ y
            except np.linalg.LinAlgError:
                b = np.zeros(X.shape[1])

        if np.isnan(sBT):   sBT = b[0]
        if varyCRL and np.isnan(sCRL): sCRL = b[1]
        if np.isnan(sBase): sBase = b[2 if varyCRL else 1]
        if np.isnan(sSub):  sSub = b[3 if varyCRL else 2]

        # unit moves (downhill)
        dBT_unit = Btstep * (-np.sign(sBT)) if abs(sBT) > 1e-12 else 0.0
        dBase_unit = Basestep * (-np.sign(sBase)) if abs(sBase) > 1e-12 else 0.0
        dSub_unit = Subbasestep * (-np.sign(sSub)) if abs(sSub) > 1e-12 else 0.0
        dCRL_unit = (10 * (-np.sign(sCRL)) if (varyCRL and abs(sCRL) > 1e-12) else 0.0)

        # try cost-ratio-aware BT increase with shaving
        BTi, CRLi, Basei, Subi, Costi, ok = trade_BT_up(Type, BTc, CRLc, Basec, Subc, Btstep)
        if ok:
            # lattice check
            def steps_ok(BT0, BT1, Ba0, Ba1, Su0, Su1, CRL0, CRL1):
                def step_mult(d, step):
                    return (abs(d) < 1e-12) or (abs((d % step)) < 1e-12)

                return step_mult(BT1 - BT0, Btstep) and \
                       step_mult(Ba1 - Ba0, Basestep) and \
                       step_mult(Su1 - Su0, Subbasestep) and \
                       step_mult(CRL1 - CRL0, 10)

            st = eval_state_cached((BTi, CRLi, Basei, Subi)) if isT235 else eval_state_cached((BTi, Basei, Subi))
            cand_bt = [st["BT"], st["CRL"] if isT235 else 0.0, st["Base"], st["Sub"],
                       st["SDbf"], st["SDsbr"], st["SDctf"] if isT235 else np.nan,
                       st["CFD"] if isT235 else 0.0, st["Cost"]]
            if steps_ok(BTc, cand_bt[0], Basec, cand_bt[2], Subc, cand_bt[3], CRLc, cand_bt[1]) and \
                    feas_ok(cand_bt) and (cand_bt[-1] + tolC < Costc):
                BTc, CRLc, Basec, Subc, Costc = cand_bt[0], cand_bt[1], cand_bt[2], cand_bt[3], cand_bt[-1]
                Nbf_c, Nsbr_c, Nctf_c, CFD_c = cand_bt[4], cand_bt[5], cand_bt[6], cand_bt[7]
                TRACE.append([BTc, CRLc, Basec, Subc, Nbf_c, Nsbr_c, Nctf_c, CFD_c, Costc])
                continue

        # candidate moves
        CAND = np.unique(np.array([
            [dBT_unit, dCRL_unit if varyCRL else 0.0, dBase_unit, dSub_unit],
            [0.0, dCRL_unit if varyCRL else 0.0, dBase_unit, dSub_unit],
            [dBT_unit, 0.0, 0.0, 0.0],
            [0.0, 0.0, dBase_unit, 0.0],
            [0.0, 0.0, 0.0, dSub_unit],
        ], dtype=float), axis=0)
        CAND = CAND[np.any(np.abs(CAND) > 0, axis=1)]

        best_local = [BTc, CRLc, Basec, Subc, Nbf_c, Nsbr_c, Nctf_c, CFD_c, Costc]
        found_better = False

        for m in (1, 2):
            for row in CAND:
                dBT, dCRL, dBase, dSub = m * row[0], m * row[1], m * row[2], m * row[3]
                BTn, CRLn, Basen, Subn = BTc + dBT, CRLc + dCRL, Basec + dBase, Subc + dSub

                if isT235:
                    BTn, CRLn, Basen, Subn = clampQ_235(BTn, CRLn, Basen, Subn)
                    if Type == 3:
                        CRLn = 0.0
                    st = eval_state_cached((BTn, CRLn, Basen, Subn))
                    cand = [st["BT"], st["CRL"], st["Base"], st["Sub"], st["SDbf"], st["SDsbr"],
                            st["SDctf"], st["CFD"], st["Cost"]]
                else:
                    BTn, Basen, Subn = clampQ_1467(BTn, Basen, Subn)
                    st = eval_state_cached((BTn, Basen, Subn))
                    cand = [st["BT"], 0.0, st["Base"], st["Sub"], st["SDbf"], st["SDsbr"], np.nan, 0.0, st["Cost"]]

                # exact step lattice & feasibility
                def step_mult(d, step):
                    return (abs(d) < 1e-12) or (abs((d % step)) < 1e-12)

                ok_lattice = (step_mult(cand[0] - BTc, Btstep) and
                              step_mult(cand[2] - Basec, Basestep) and
                              step_mult(cand[3] - Subc, Subbasestep) and
                              step_mult(cand[1] - CRLc, 10))
                if not ok_lattice or not feas_ok(cand):
                    continue

                if cand[-1] + tolC < best_local[-1]:
                    best_local = cand
                    found_better = True

        if not found_better:
            break

        # commit
        BTc, CRLc, Basec, Subc = best_local[0], best_local[1], best_local[2], best_local[3]
        Nbf_c, Nsbr_c, Nctf_c, CFD_c, Costc = best_local[4], best_local[5], best_local[6], best_local[7], best_local[8]
        TRACE.append(best_local)

    # final safety: strip any infeasible trace rows (robust for single-row)
    TRACE = np.asarray(TRACE, dtype=float)
    if TRACE.ndim == 1:
        TRACE = TRACE.reshape(1, -1)

    if isT235:
        m = (TRACE[:, 4] > 0) & (TRACE[:, 5] > 0) & (TRACE[:, 6] > 0) & ((not useCFD) | (TRACE[:, 7] < 1))
    else:
        m = (TRACE[:, 4] > 0) & (TRACE[:, 5] > 0)

    m = np.asarray(m, dtype=bool).ravel()
    if m.size == 0:
        TRACE = np.empty((0, 9), dtype=float)
    else:
        TRACE = TRACE[m]

    AANR = TRACE[np.argsort(TRACE[:, 8])] if TRACE.size else np.empty((0, 9), dtype=float)

    # consolidate and report
    rows = []
    if AANR.size:
        rows.append(AANR[0, :])
    if AANZ.size:
        rows.append(AANZ[0, :])
    rows.append(np.array([AAAc[0, 4], AAAc[0, 5], AAAc[0, 6], AAAc[0, 7],
                          AAAc[0, 0], AAAc[0, 1], AAAc[0, 2], AAAc[0, 3], AAAc[0, 8]]))
    rows = np.unique(np.round(np.vstack(rows), 12), axis=0)
    rows = rows[np.argsort(rows[:, 8])]

    cols = ['BT', 'CRL', 'Base', 'Subbase', 'Nbf', 'Nsbr', 'Nctf', 'CFD', 'Cost']
    if pd is not None:
        T = pd.DataFrame(rows, columns=cols)
        if not isT235:
            T = T[['BT', 'CRL', 'Base', 'Subbase', 'Nbf', 'Nsbr', 'Cost']]
        if T.empty:
            Best = pd.DataFrame()
        else:
            Best = T.iloc[[0]]
    else:
        # dict-shaped DataFrame fallback
        if not isT235:
            cols_use = ['BT', 'CRL', 'Base', 'Subbase', 'Nbf', 'Nsbr', 'Cost']
            data = [
                {
                    'BT': float(r[0]), 'CRL': 0.0, 'Base': float(r[2]), 'Subbase': float(r[3]),
                    'Nbf': float(r[4]), 'Nsbr': float(r[5]), 'Cost': float(r[8])
                } for r in rows
            ]
        else:
            cols_use = cols
            data = [
                {
                    'BT': float(r[0]), 'CRL': float(r[1]), 'Base': float(r[2]), 'Subbase': float(r[3]),
                    'Nbf': float(r[4]), 'Nsbr': float(r[5]), 'Nctf': float(r[6]), 'CFD': float(r[7]), 'Cost': float(r[8])
                } for r in rows
            ]
        T = {"_type": "DataFrame", "columns": cols_use, "data": data}
        Best = {"_type": "DataFrame", "columns": cols_use, "data": [data[0]]} if data else {"_type": "DataFrame", "columns": cols_use, "data": []}

    return Best, T, TRACE


# ============================================================
# ================ Display / cost / labels ===================
# ============================================================



def crl_layer_name_by_type(Type: int, CRL: float) -> str:
    if CRL <= 0:
        return ''
    if Type in (2, 5):
        return 'CRL (AIL)'
    if Type == 3:
        return 'CRL (SAMI)'
    return 'CRL'


def layer_names_by_type(Type: int, is_wmm_r: bool, is_gsb_r: bool) -> Tuple[str, str]:
    if Type == 1:
        return 'Base (WMM)', 'Subbase (GSB)'
    if Type == 4:
        return 'Base (ETB)', 'Subbase (CTSB)'
    if Type == 6:
        return 'Base (WMM)', 'Subbase (CTSB)'
    if Type == 7:
        base = 'Base (Reinforced WMM)' if is_wmm_r else 'Base (WMM)'
        sub = 'Subbase (Reinforced GSB)' if is_gsb_r else 'Subbase (GSB)'
        return base, sub
    if Type in (2, 3):
        return 'Base (CTB)', 'Subbase (CTSB)'
    if Type == 5:
        return 'Base (CTB)', 'Subbase (GSB)'
    if Type == 8:
        return 'Base (Reinforced)', 'Subbase (CTSB)'
    return 'Base', 'Subbase'


import warnings

def cost_unified(
    Type: int, BT: float, CRL: float, Base: float, Subb: float,
    BC_DBM_width: float, BC_cost: float, DBM_cost: float,
    CRL_cost: float, CRL_width: float,
    Base_width: float, Base_cost: float,
    Subbase_width: float, Subbase_cost: float,
    SAMI_cost: float,
    wmm_r_cost: float, gsb_r_cost: float,
    is_wmm_r: bool, is_gsb_r: bool
) -> float:
    # split BT
    if BT <= 40:
        BC_mm, DBM_mm = BT, 0.0
    elif BT < 90:
        BC_mm, DBM_mm = 30.0, BT - 30.0
    else:
        BC_mm, DBM_mm = 40.0, BT - 40.0

    CRL_mm  = CRL if (Type in (2, 5)) else 0.0
    Base_mm = Base
    Sub_mm  = Subb

    # Flats (₹/km)
    SAMI_Flat   = (BC_DBM_width * 1000.0) * SAMI_cost if (Type == 3 and SAMI_cost >= 0) else 0.0
    WMM_R_Flat  = (Base_width    * 1000.0) * wmm_r_cost if ((Type in (7, 8)) and is_wmm_r and wmm_r_cost >= 0) else 0.0
    GSB_R_Flat  = (Subbase_width * 1000.0) * gsb_r_cost if ((Type in (7, 8)) and is_gsb_r and gsb_r_cost >= 0) else 0.0

    # Layer costs (₹) for 1 km length
    cost_val = (
        BC_mm   * 0.001 * BC_DBM_width  * 1000.0 * BC_cost   +
        DBM_mm  * 0.001 * BC_DBM_width  * 1000.0 * DBM_cost  +
        CRL_mm  * 0.001 * CRL_width     * 1000.0 * CRL_cost  +
        Base_mm * 0.001 * Base_width    * 1000.0 * Base_cost +
        Sub_mm  * 0.001 * Subbase_width * 1000.0 * Subbase_cost +
        SAMI_Flat + WMM_R_Flat + GSB_R_Flat
    ) / 100000.0  # scale to ₹ lakh

    return float(cost_val)


def FN_Cri_Cal_BF_SR_FF(
    Type: int, Thk: List[float], N_design: float, Subgrade_CBR: float,
    BT_Mod: float, Va: float, Vbe: float, reliability: float,
    Base_Mod: float, RF_CTB: float, Subbase_Mod: float, CRL_Mod: float
) -> Tuple[List[float], List[float]]:
    # Subgrade modulus
    if Subgrade_CBR <= 5:
        Subgrade_Mod = Subgrade_CBR * 10.0
    else:
        Subgrade_Mod = 17.6 * (Subgrade_CBR ** 0.64)

    BT_Thk, CRL_Thk, Base_Thk, Subbase_Thk = Thk
    load = 20000.0
    typre = 0.56
    alpha = 0.0

    if Type == 2:
        n = 5
        Thickness = [BT_Thk, CRL_Thk, Base_Thk, Subbase_Thk]
        E = [BT_Mod, CRL_Mod, Base_Mod, Subbase_Mod, Subgrade_Mod]
        v = [0.35, 0.35, 0.25, 0.25, 0.35]
        isbonded = True

        Eva_depth = BT_Thk
        Mat_SigmaSi_BT = AIO_SigmaSi(n, Thickness, E, v, isbonded, typre, load, alpha, Eva_depth, 1, 2)

        Eva_depth = sum(Thickness)
        Mat_SigmaSi_SUB = AIO_SigmaSi(n, Thickness, E, v, isbonded, typre, load, alpha, Eva_depth, 4, 5)

        Eva_depth = sum(Thickness[:3]); typre_tmp = 0.8
        Mat_SigmaSi_CTB = AIO_SigmaSi(n, Thickness, E, v, isbonded, typre_tmp, load, alpha, Eva_depth, 3, 4)

    elif Type == 3:
        n = 4
        Thickness = [BT_Thk, Base_Thk, Subbase_Thk]
        E = [BT_Mod, Base_Mod, Subbase_Mod, Subgrade_Mod]
        v = [0.35, 0.25, 0.25, 0.35]
        isbonded = True

        Eva_depth = BT_Thk
        Mat_SigmaSi_BT = AIO_SigmaSi(n, Thickness, E, v, isbonded, typre, load, alpha, Eva_depth, 1, 2)

        Eva_depth = sum(Thickness)
        Mat_SigmaSi_SUB = AIO_SigmaSi(n, Thickness, E, v, isbonded, typre, load, alpha, Eva_depth, 3, 4)

        Eva_depth = sum(Thickness[:2]); typre_tmp = 0.8
        Mat_SigmaSi_CTB = AIO_SigmaSi(n, Thickness, E, v, isbonded, typre_tmp, load, alpha, Eva_depth, 2, 3)

    elif Type == 5:
        # Granular Subbase modulus depends on thickness
        Subbase_Mod_eff = 0.2 * (Subbase_Thk ** 0.45) * Subgrade_Mod
        n = 5
        Thickness = [BT_Thk, CRL_Thk, Base_Thk, Subbase_Thk]
        E = [BT_Mod, CRL_Mod, Base_Mod, Subbase_Mod_eff, Subgrade_Mod]
        v = [0.35, 0.35, 0.25, 0.35, 0.35]
        isbonded = True

        Eva_depth = BT_Thk
        Mat_SigmaSi_BT = AIO_SigmaSi(n, Thickness, E, v, isbonded, typre, load, alpha, Eva_depth, 1, 2)

        Eva_depth = sum(Thickness)
        Mat_SigmaSi_SUB = AIO_SigmaSi(n, Thickness, E, v, isbonded, typre, load, alpha, Eva_depth, 4, 5)

        Eva_depth = sum(Thickness[:3]); typre_tmp = 0.8
        Mat_SigmaSi_CTB = AIO_SigmaSi(n, Thickness, E, v, isbonded, typre_tmp, load, alpha, Eva_depth, 3, 4)
    else:
        raise ValueError("Unsupported Type in FN_Cri_Cal_BF_SR_FF")

    # critical (epsilon_t, epsilon_r) in bituminous; epsilon_z in subgrade; (epsilon_t,epsilon_r) in CTB
    Critical_BSi   = float(np.max(np.abs(Mat_SigmaSi_BT[:, 8:10])))
    Critical_SubSi = float(np.max(np.abs(Mat_SigmaSi_SUB[:, 7])))
    Critical_CTBSi = float(np.max(np.abs(Mat_SigmaSi_CTB[:, 8:10])))

    Permissible_Si = AIO_PermissibleSi(reliability, N_design, BT_Mod, Va, Vbe, Base_Mod, RF_CTB)
    Critical_Si = [Critical_BSi, Critical_SubSi, Critical_CTBSi]
    return Critical_Si, Permissible_Si


def FN_Cri_Cal_CTB_CFD(
    Type: int, Thk: List[float], Subgrade_CBR: float, BT_Mod: float,
    Base_Mod: float, Subbase_Mod: float, CRL_Mod: float, FS_CTB: float,
    SA_M: np.ndarray, TaA_M: np.ndarray, TrA_M: np.ndarray
) -> float:
    # Subgrade modulus
    if Subgrade_CBR <= 5:
        Subgrade_Mod = Subgrade_CBR * 10.0
    else:
        Subgrade_Mod = 17.6 * (Subgrade_CBR ** 0.64)

    BT_Thk, CRL_Thk, Base_Thk, Subbase_Thk = Thk

    if Type == 2:
        n = 5
        Thickness = [BT_Thk, CRL_Thk, Base_Thk, Subbase_Thk]
        E = [BT_Mod, CRL_Mod, Base_Mod, Subbase_Mod, Subgrade_Mod]
        v = [0.35, 0.35, 0.25, 0.25, 0.35]
        upperlayer, lowerlayer = 3, 4
        Eva_depth = sum(Thickness[:3]); typre = 0.8; alpha = 0.0; isbonded = True
    elif Type == 3:
        n = 4
        Thickness = [BT_Thk, Base_Thk, Subbase_Thk]
        E = [BT_Mod, Base_Mod, Subbase_Mod, Subgrade_Mod]
        v = [0.35, 0.25, 0.25, 0.35]
        upperlayer, lowerlayer = 2, 3
        Eva_depth = sum(Thickness[:2]); typre = 0.8; alpha = 0.0; isbonded = True
    elif Type == 5:
        Subbase_Mod_eff = 0.2 * (Subbase_Thk ** 0.45) * Subgrade_Mod
        n = 5
        Thickness = [BT_Thk, CRL_Thk, Base_Thk, Subbase_Thk]
        E = [BT_Mod, CRL_Mod, Base_Mod, Subbase_Mod_eff, Subgrade_Mod]
        v = [0.35, 0.35, 0.25, 0.35, 0.35]
        upperlayer, lowerlayer = 3, 4
        Eva_depth = sum(Thickness[:3]); typre = 0.8; alpha = 0.0; isbonded = True
    else:
        raise ValueError("Unsupported Type for CFD.")

    def cfd_from_axle_set(ax_arr: np.ndarray, wheels: int, rep_mult: float) -> float:
        CFD_x = 0.0
        if ax_arr.size == 0:
            return 0.0
        for i in range(ax_arr.shape[0]):
            load_kN = np.mean(ax_arr[i, 0:2])  # kN per axle
            load = (load_kN * 1000.0) / wheels
            reps = rep_mult * float(ax_arr[i, 2])

            Mat = AIO_SigmaSi(n, Thickness, E, v, isbonded, typre, load, alpha, Eva_depth, upperlayer, lowerlayer)
            # tension in CTB (take max of sigma_t, sigma_r)
            sig = float(np.max(np.abs(Mat[:, 3:5])))
            SR = sig / FS_CTB
            NF = 10.0 ** ((0.972 - SR) / 0.0825)
            CFD_x += reps / NF
            if CFD_x > 1.0:
                break
        return CFD_x

    CFD_SA  = cfd_from_axle_set(SA_M,  4, 1.0)
    CFD_TaA = cfd_from_axle_set(TaA_M, 8, 2.0)
    CFD_TrA = cfd_from_axle_set(TrA_M,12, 3.0)
    return float(CFD_SA + CFD_TaA + CFD_TrA)


def AIO_N_Cri_Cal_BF_SR(
    Type: int, Thk: List[float], N_design: float, Subgrade_CBR: float,
    BT_Mod: float, Va: float, Vbe: float, reliability: float,
    Base_Mod: Optional[float], Subbase_Mod: Optional[float],
    Rtype: Optional[int], R_Base: Optional[float], R_Subbase: Optional[float]
) -> Tuple[List[float], List[float]]:
    BT_Thk, Base_Thk, Subbase_Thk = Thk
    if Subgrade_CBR <= 5:
        Subgrade_Mod = Subgrade_CBR * 10.0
    else:
        Subgrade_Mod = 17.6 * (Subgrade_CBR ** 0.64)

    load = 20000.0; typre = 0.56; alpha = 0.0

    if Type == 1:
        Granular_Thk = Subbase_Thk + Base_Thk
        Granular_Mod = 0.2 * (Granular_Thk ** 0.45) * Subgrade_Mod
        n = 3
        Thickness = [BT_Thk, Granular_Thk]
        E = [BT_Mod, Granular_Mod, Subgrade_Mod]
        v = [0.35, 0.35, 0.35]; isbonded = True

        Eva_depth = BT_Thk
        Mat_SigmaSi_BT = AIO_SigmaSi(n, Thickness, E, v, isbonded, typre, load, alpha, Eva_depth, 1, 2)

        Eva_depth = sum(Thickness)
        Mat_SigmaSi_SUB = AIO_SigmaSi(n, Thickness, E, v, isbonded, typre, load, alpha, Eva_depth, 2, 3)

    elif Type in (4, 6):
        n = 4
        Thickness = [BT_Thk, Base_Thk, Subbase_Thk]
        E = [BT_Mod, Base_Mod, Subbase_Mod, Subgrade_Mod]
        v = [0.35, 0.35, 0.25, 0.35]; isbonded = True

        Eva_depth = BT_Thk
        Mat_SigmaSi_BT = AIO_SigmaSi(n, Thickness, E, v, isbonded, typre, load, alpha, Eva_depth, 1, 2)

        Eva_depth = sum(Thickness)
        Mat_SigmaSi_SUB = AIO_SigmaSi(n, Thickness, E, v, isbonded, typre, load, alpha, Eva_depth, 3, 4)

    elif Type == 7:
        Subbase_Mod_eff = 0.2 * (Subbase_Thk ** 0.45) * Subgrade_Mod
        nE = 2; ThicknessE = Subbase_Thk; EE = [Subbase_Mod_eff, Subgrade_Mod]; vE = [0.35, 0.35]
        EMr = AIO_EffectiveMr(nE, ThicknessE, EE, vE)
        Base_Mod_eff = 0.2 * (Base_Thk ** 0.45) * EMr

        if Rtype == 1:  # MIF
            BaseR_Mod    = Base_Mod_eff * R_Base
            SubbaseR_Mod = Subbase_Mod_eff * R_Subbase
        else:  # LCR
            a2 = 0.249 * (math.log10(Base_Mod_eff * 145.038)) - 0.977
            a3 = 0.227 * (math.log10(Subbase_Mod_eff * 145.038)) - 0.839
            BaseR_Mod    = (10.0 ** ((0.977 + R_Base * a2) / 0.249)) / 145.038
            SubbaseR_Mod = (10.0 ** ((0.839 + R_Subbase * a3) / 0.227)) / 145.038

        n = 4
        Thickness = [BT_Thk, Base_Thk, Subbase_Thk]
        E = [BT_Mod, BaseR_Mod, SubbaseR_Mod, Subgrade_Mod]
        v = [0.35, 0.35, 0.35, 0.35]; isbonded = True

        Eva_depth = BT_Thk
        Mat_SigmaSi_BT = AIO_SigmaSi(n, Thickness, E, v, isbonded, typre, load, alpha, Eva_depth, 1, 2)

        Eva_depth = sum(Thickness)
        Mat_SigmaSi_SUB = AIO_SigmaSi(n, Thickness, E, v, isbonded, typre, load, alpha, Eva_depth, 3, 4)

    elif Type == 8:
        # CTSB over subgrade defines EMr; Base is WMM with reinforcement only on BASE
        nE = 2; ThicknessE = Subbase_Thk; EE = [Subbase_Mod, Subgrade_Mod]; vE = [0.25, 0.35]
        EMr = AIO_EffectiveMr(nE, ThicknessE, EE, vE)
        Base_Mod_eff = min(Base_Mod, 0.2 * (Base_Thk ** 0.45) * EMr)

        if Rtype == 1:  # MIF
            BaseR_Mod    = Base_Mod_eff * R_Base
            SubbaseR_Mod = Subbase_Mod
        else:          # LCR
            a2 = 0.249 * (math.log10(Base_Mod_eff * 145.038)) - 0.977
            BaseR_Mod = (10.0 ** ((0.977 + R_Base * a2) / 0.249)) / 145.038
            SubbaseR_Mod = Subbase_Mod

        n = 4
        Thickness = [BT_Thk, Base_Thk, Subbase_Thk]
        E = [BT_Mod, BaseR_Mod, SubbaseR_Mod, Subgrade_Mod]
        v = [0.35, 0.35, 0.25, 0.35]; isbonded = True

        Eva_depth = BT_Thk
        Mat_SigmaSi_BT = AIO_SigmaSi(n, Thickness, E, v, isbonded, typre, load, alpha, Eva_depth, 1, 2)

        Eva_depth = sum(Thickness)
        Mat_SigmaSi_SUB = AIO_SigmaSi(n, Thickness, E, v, isbonded, typre, load, alpha, Eva_depth, 3, 4)

    else:
        raise ValueError("Unsupported Type in AIO_N_Cri_Cal_BF_SR")

    Critical_BSi   = float(np.max(np.abs(Mat_SigmaSi_BT[:, 8:10])))
    Critical_SubSi = float(np.max(np.abs(Mat_SigmaSi_SUB[:, 7])))

    Permissible_Si = AIO_PermissibleSi(reliability, N_design, BT_Mod, Va, Vbe, None)
    Critical_Si = [Critical_BSi, Critical_SubSi]
    return Critical_Si, Permissible_Si


def AIO_SigmaSi(
    n: int, Thickness: List[float], E: List[float], v: List[float],
    isbonded: bool, typre: float, load: float, alpha: float,
    Eva_depth: float, upperlayer: int, lowerlayer: int
) -> np.ndarray:
    # Points for OUTER (O) – dual wheel centers at r=0 and r=310 mm
    PointsW1O = [dict(r=0.0,   z=Eva_depth, layer=upperlayer),
                 dict(r=0.0,   z=Eva_depth, layer=lowerlayer)]
    PointsW2O = [dict(r=310.0, z=Eva_depth, layer=upperlayer),
                 dict(r=310.0, z=Eva_depth, layer=lowerlayer)]

    q = typre
    a = math.sqrt(load/(math.pi*q))

    Results_W1_O = AIO_R(n, Thickness, E, v, isbonded, PointsW1O, a, q)
    Results_W2_O = AIO_R(n, Thickness, E, v, isbonded, PointsW2O, a, q)
    Results_O    = AIO_wheel_superposition(E, v, alpha, PointsW1O, PointsW2O, Results_W1_O, Results_W2_O)

    # Points for INNER (C) – at r=155 mm for both wheels
    PointsW1C = [dict(r=155.0, z=Eva_depth, layer=upperlayer),
                 dict(r=155.0, z=Eva_depth, layer=lowerlayer)]
    PointsW2C = [dict(r=155.0, z=Eva_depth, layer=upperlayer),
                 dict(r=155.0, z=Eva_depth, layer=lowerlayer)]

    Results_W1_C = AIO_R(n, Thickness, E, v, isbonded, PointsW1C, a, q)
    Results_W2_C = AIO_R(n, Thickness, E, v, isbonded, PointsW2C, a, q)
    Results_C    = AIO_wheel_superposition(E, v, alpha, PointsW1C, PointsW2C, Results_W1_C, Results_W2_C)

    def row_from(res):
        return np.array([
            Eva_depth,              # 0
            res["r"],               # 1
            res["sigma_z"],         # 2
            res["sigma_t"],         # 3 (global y)
            res["sigma_r"],         # 4 (global x)
            res["tau_rz"],          # 5 (we keep name, but it's tau_xz after rotation)
            res["w"],               # 6
            res["epsilon_z"],       # 7
            res["epsilon_t"],       # 8
            res["epsilon_r"],       # 9
        ], dtype=float)

    Mat = np.vstack([
        row_from({**Results_O[0], "r": PointsW1O[0]["r"]}),
        row_from({**Results_O[1], "r": PointsW1O[1]["r"]}),
        row_from({**Results_C[0], "r": PointsW1C[0]["r"]}),
        row_from({**Results_C[1], "r": PointsW1C[1]["r"]}),
    ])
    return Mat


def AIO_PermissibleSi(
    reliability: float, N_design: float, BT_Mod: float, Va: float, Vbe: float,
    Base_Mod: Optional[float], RF_CTB: Optional[float] = 1.0
) -> List[float]:
    # Bituminous fatigue and rutting permissible strains
    M = 4.84 * ((Vbe / (Va + Vbe)) - 0.69)
    C = 10.0 ** M

    if reliability == 80:
        si_rut_permissible  = 1.0 / ((N_design / (4.1656e-8)) ** (1.0 / 4.5337))
        si_bfat_permissible = 1.0 / ((N_design / ((C * 1.6064e-4) * ((1.0 / BT_Mod) ** 0.854))) ** (1.0 / 3.89))
    else:
        si_rut_permissible  = 1.0 / ((N_design / (1.41e-8)) ** (1.0 / 4.5337))
        si_bfat_permissible = 1.0 / ((N_design / ((C * 0.5161e-4) * ((1.0 / BT_Mod) ** 0.854))) ** (1.0 / 3.89))

    if Base_Mod is None:
        return [si_bfat_permissible, si_rut_permissible]
    else:
        if RF_CTB is None or RF_CTB == 0:
            RF_CTB = 1.0
        si_cfat_permissible = (((113000.0 / (Base_Mod ** 0.804)) + 191.0) /
                               ((N_design / RF_CTB) ** (1.0 / 12.0))) * 1e-6
        return [si_bfat_permissible, si_rut_permissible, si_cfat_permissible]


def AIO_EffectiveMr(n: int, Thickness: float, E: List[float], v: List[float]) -> float:
    # surface deflection from single wheel, then back-calculate Mr
    load = 40000.0
    tyre = 0.56
    alpha = 0.0
    q = tyre
    a = math.sqrt(load / (math.pi * q))
    isbonded = True

    Points = [dict(r=0.0, z=0.0, layer=1)]
    Results = AIO_R(n, [Thickness], E, v, isbonded, Points, a, q)
    surf_def = Results[0]["w"]
    EMr = 2.0 * (1.0 - v[0] ** 2) * q * a / surf_def
    return float(EMr)


def AIO_R(
    n: int, Thickness: List[float], E: List[float], v: List[float],
    isbonded: bool, Points: List[Dict[str, float]], a: float, q: float
) -> List[Dict[str, float]]:
    # precompute z, H
    z = [sum(Thickness[:j+1]) for j in range(n-1)]
    H = z[-1] if z else 1.0
    alpha = a / H
    dl = 0.2 / alpha

    # init results
    Results = []
    for _ in range(len(Points)):
        Results.append(dict(sigma_z=0.0, sigma_r=0.0, sigma_t=0.0,
                            tau_rz=0.0, w=0.0, u=0.0))

    # integrate m from 0 to 200 with 4-pt Gauss on each subinterval
    l = 0.0
    while l <= 200.0 + 1e-12:
        for gauss_point in (1, 2, 3, 4):
            if gauss_point == 1:
                m  = (l + dl/2.0) - 0.86114 * (dl/2.0)
                fc = 0.34786 * (dl/2.0)
            elif gauss_point == 2:
                m  = (l + dl/2.0) - 0.33998 * (dl/2.0)
                fc = 0.65215 * (dl/2.0)
            elif gauss_point == 3:
                m  = (l + dl/2.0) + 0.33998 * (dl/2.0)
                fc = 0.65215 * (dl/2.0)
            else:
                m  = (l + dl/2.0) + 0.86114 * (dl/2.0)
                fc = 0.34786 * (dl/2.0)

            Res_hat = AIO_R_hat(n, Thickness, E, v, isbonded, m, Points, H)

            if m != 0.0:
                for j in range(len(Points)):
                    rho = Points[j]["r"] / H
                    J1 = besselj(1, m * alpha)
                    J0 = besselj(0, m * alpha)
                    # multiply/add as in MATLAB
                    Results[j]["sigma_z"] += fc * (q * alpha * Res_hat[j]["sigma_z"] / m * J1)
                    Results[j]["sigma_r"] += fc * (q * alpha * Res_hat[j]["sigma_r"] / m * J1)
                    Results[j]["sigma_t"] += fc * (q * alpha * Res_hat[j]["sigma_t"] / m * J1)
                    Results[j]["tau_rz"]  += fc * (q * alpha * Res_hat[j]["tau_rz"]  / m * J1)
                    Results[j]["w"]       += fc * (q * alpha * Res_hat[j]["w"]       / m * J1)
                    Results[j]["u"]       += fc * (q * alpha * Res_hat[j]["u"]       / m * J1)
        l += dl

    # convert to strains with Hooke’s law (plane strain in 3D iso)
    for j in range(len(Points)):
        ii = int(Points[j]["layer"]) - 1
        Ez, Er, Et = Results[j]["sigma_z"], Results[j]["sigma_r"], Results[j]["sigma_t"]
        vv = v[ii]; EE = E[ii]
        Results[j]["epsilon_z"] = (Ez - vv * (Er + Et)) / EE
        Results[j]["epsilon_r"] = (Er - vv * (Ez + Et)) / EE
        Results[j]["epsilon_t"] = (Et - vv * (Er + Ez)) / EE

    return Results


def AIO_R_hat(
    n: int, Thickness: List[float], E: List[float], v: List[float], isbonded: bool,
    m: float, Points: List[Dict[str, float]], H: float
) -> List[Dict[str, float]]:
    # layer boundaries
    z = [sum(Thickness[:i+1]) for i in range(n-1)]
    lambda_arr = [zz / H for zz in z]
    lambda_arr.append(float("inf"))

    F = [math.exp(-m * (lambda_arr[0] - 0.0))]
    R = [(E[0] / E[1]) * (1.0 + v[1]) / (1.0 + v[0])]
    for i in range(1, n-1):
        F.append(math.exp(-m * (lambda_arr[i] - lambda_arr[i-1])))
        R.append((E[i] / E[i+1]) * (1.0 + v[i+1]) / (1.0 + v[i]))
    F.append(math.exp(-m * (lambda_arr[n-1] - lambda_arr[n-2])))

    M_list = []
    for i in range(n-1):
        li = lambda_arr[i]
        F_i = F[i]
        F_ip1 = F[i+1]
        R_i = R[i] if i < len(R) else 1.0

        if isbonded:
            M1 = np.array([
                [1,         F_i,                         -(1 - 2*v[i] - m*li),        (1 - 2*v[i] + m*li) * F_i],
                [1,        -F_i,                          2*v[i] + m*li,             (2*v[i] - m*li) * F_i],
                [1,         F_i,                          1 + m*li,                  -(1 - m*li) * F_i],
                [1,        -F_i,                         -(2 - 4*v[i] - m*li),      -(2 - 4*v[i] + m*li) * F_i],
            ], dtype=float)

            M2 = np.array([
                [F_ip1, 1,                           -(1 - 2*v[i+1] - m*li) * F_ip1,  1 - 2*v[i+1] + m*li],
                [F_ip1, -1,                          ( 2*v[i+1] + m*li) * F_ip1,      2*v[i+1] - m*li],
                [R_i*F_ip1, R_i,                     ( 1 + m*li) * R_i * F_ip1,      -(1 - m*li) * R_i],
                [R_i*F_ip1, -R_i,                    -(2 - 4*v[i+1] - m*li) * R_i * F_ip1, -(2 - 4*v[i+1] + m*li) * R_i],
            ], dtype=float)
        else:
            M1 = np.array([
                [1,         F_i,                         -(1 - 2*v[i] - m*li),            (1 - 2*v[i] + m*li) * F_i],
                [1,         F_i,                          1 + m*li,                       -(1 - m*li) * F_i],
                [1,        -F_i,                          2*v[i] + m*li,                   (2*v[i] - m*li) * F_i],
                [0,         0,                            0,                                0],
            ], dtype=float)

            M2 = np.array([
                [F_ip1,           1,                         -(1 - 2*v[i+1] - m*li) * F_ip1,   1 - 2*v[i+1] + m*li],
                [R_i*F_ip1,       R_i,                       ( 1 + m*li) * R_i * F_ip1,       -(1 - m*li) * R_i],
                [0,               0,                         0,                                0],
                [F_ip1,          -1,                          ( 2*v[i+1] + m*li),               2*v[i+1] - m*li],
            ], dtype=float)

        # Solve M1 x = M2 for transfer matrix
        X = np.linalg.solve(M1, M2)
        M_list.append(X)

    MM = np.eye(4)
    for Mi in M_list:
        MM = MM @ Mi
    MM = MM[:, [1, 3]]  # columns 2 and 4 (0-based)

    # boundary at surface
    li1 = lambda_arr[0]
    b11 = math.exp(-li1 * m)
    b21 = math.exp(-li1 * m)
    b12 = 1.0
    b22 = -1.0

    c11 = -(1 - 2*v[0]) * math.exp(-m * li1)
    c21 =  2 * v[0]      * math.exp(-m * li1)
    c12 =  1 - 2*v[0]
    c22 =  2 * v[0]

    k11 = b11 * MM[0,0] + b12 * MM[1,0] + c11 * MM[2,0] + c12 * MM[3,0]
    k12 = b11 * MM[0,1] + b12 * MM[1,1] + c11 * MM[2,1] + c12 * MM[3,1]
    k21 = b21 * MM[0,0] + b22 * MM[1,0] + c21 * MM[2,0] + c22 * MM[3,0]
    k22 = b21 * MM[0,1] + b22 * MM[1,1] + c21 * MM[2,1] + c22 * MM[3,1]

    A = [0.0]*n; B = [0.0]*n; C = [0.0]*n; D = [0.0]*n
    A[n-1] = 0.0
    denom = (k11 * k22 - k12 * k21)
    B[n-1] = k22 / denom if abs(denom) > 1e-14 else 0.0
    C[n-1] = 0.0
    D[n-1] = 1.0 / (k12 - (k22 * k11 / k21)) if abs(k21) > 1e-14 else 0.0

    for i in range(n-2, -1, -1):
        XX = M_list[i] @ np.array([A[i+1], B[i+1], C[i+1], D[i+1]], dtype=float)
        A[i], B[i], C[i], D[i] = [float(XX[0]), float(XX[1]), float(XX[2]), float(XX[3])]

    # compute response at each point
    Results = []
    for j in range(len(Points)):
        rho = Points[j]["r"] / H
        lmm = Points[j]["z"] / H
        ii = int(Points[j]["layer"]) - 1

        if ii != 0:
            term1 = (A[ii] - C[ii] * (1 - 2*v[ii] - m*lmm)) * math.exp(-m * (lambda_arr[ii]   - lmm))
            term2 = (B[ii] + D[ii] * (1 - 2*v[ii] + m*lmm)) * math.exp(-m * (lmm - lambda_arr[ii-1]))
            sigma_z = -m * besselj(0, m*rho) * (term1 + term2)

            term1r = (A[ii] + C[ii] * (1 + m*lmm)) * math.exp(-m * (lambda_arr[ii]   - lmm))
            term2r = (B[ii] - D[ii] * (1 - m*lmm)) * math.exp(-m * (lmm - lambda_arr[ii-1]))
            sigma_r = (m*besselj(0, m*rho) - besselj(1, m*rho)/rho) * (term1r + term2r) \
                      + 2*v[ii]*m*besselj(0, m*rho) * (C[ii]*math.exp(-m*(lambda_arr[ii]-lmm)) - D[ii]*math.exp(-m*(lmm - lambda_arr[ii-1])))

            sigma_t = (besselj(1, m*rho)/rho) * (term1r + term2r) \
                      + 2*v[ii]*m*besselj(0, m*rho) * (C[ii]*math.exp(-m*(lambda_arr[ii]-lmm)) - D[ii]*math.exp(-m*(lmm - lambda_arr[ii-1])))

            tau_rz = m*besselj(1, m*rho) * ((A[ii] + C[ii]*(2*v[ii] + m*lmm)) * math.exp(-m*(lambda_arr[ii]-lmm)) -
                                            (B[ii] - D[ii]*(2*v[ii] - m*lmm)) * math.exp(-m*(lmm - lambda_arr[ii-1])))

            w = -H*(1 + v[ii]) / E[ii] * besselj(0, m*rho) * (
                (A[ii] - C[ii]*(2 - 4*v[ii] - m*lmm)) * math.exp(-m*(lambda_arr[ii]-lmm)) -
                (B[ii] + D[ii]*(2 - 4*v[ii] + m*lmm)) * math.exp(-m*(lmm - lambda_arr[ii-1]))
            )
            u = H*(1 + v[ii]) / E[ii] * besselj(1, m*rho) * (
                (A[ii] + C[ii]*(1 + m*lmm)) * math.exp(-m*(lambda_arr[ii]-lmm)) +
                (B[ii] - D[ii]*(1 - m*lmm)) * math.exp(-m*(lmm - lambda_arr[ii-1]))
            )

            if rho == 0.0:
                sigma_r = (m*besselj(0, m*rho) - m/2.0) * (term1r + term2r) + \
                          2*v[ii]*m*besselj(0, m*rho) * (C[ii]*math.exp(-m*(lambda_arr[ii]-lmm)) - D[ii]*math.exp(-m*(lmm - lambda_arr[ii-1])))
                sigma_t = (m/2.0) * (term1r + term2r) + \
                          2*v[ii]*m*besselj(0, m*rho) * (C[ii]*math.exp(-m*(lambda_arr[ii]-lmm)) - D[ii]*math.exp(-m*(lmm - lambda_arr[ii-1])))
        else:
            term1 = (A[ii] - C[ii] * (1 - 2*v[ii] - m*lmm)) * math.exp(-m * (lambda_arr[ii] - lmm))
            term2 = (B[ii] + D[ii] * (1 - 2*v[ii] + m*lmm)) * math.exp(-m * (lmm - 0.0))
            sigma_z = -m * besselj(0, m*rho) * (term1 + term2)

            term1r = (A[ii] + C[ii] * (1 + m*lmm)) * math.exp(-m * (lambda_arr[ii] - lmm))
            term2r = (B[ii] - D[ii] * (1 - m*lmm)) * math.exp(-m * (lmm - 0.0))
            sigma_r = (m*besselj(0, m*rho) - besselj(1, m*rho)/rho) * (term1r + term2r) + \
                      2*v[ii]*m*besselj(0, m*rho) * (C[ii]*math.exp(-m*(lambda_arr[ii]-lmm)) - D[ii]*math.exp(-m*(lmm - 0.0)))

            sigma_t = (besselj(1, m*rho)/rho) * (term1r + term2r) + \
                      2*v[ii]*m*besselj(0, m*rho) * (C[ii]*math.exp(-m*(lambda_arr[ii]-lmm)) - D[ii]*math.exp(-m*(lmm - 0.0)))

            tau_rz = m*besselj(1, m*rho) * ((A[ii] + C[ii]*(2*v[ii] + m*lmm)) * math.exp(-m*(lambda_arr[ii]-lmm)) -
                                            (B[ii] - D[ii]*(2*v[ii] - m*lmm)) * math.exp(-m*(lmm - 0.0)))

            w = -H*(1 + v[ii]) / E[ii] * besselj(0, m*rho) * (
                (A[ii] - C[ii]*(2 - 4*v[ii] - m*lmm)) * math.exp(-m*(lambda_arr[ii]-lmm)) -
                (B[ii] + D[ii]*(2 - 4*v[ii] + m*lmm)) * math.exp(-m*(lmm - 0.0))
            )
            u = H*(1 + v[ii]) / E[ii] * besselj(1, m*rho) * (
                (A[ii] + C[ii]*(1 + m*lmm)) * math.exp(-m*(lambda_arr[ii]-lmm)) +
                (B[ii] - D[ii]*(1 - m*lmm)) * math.exp(-m*(lmm - 0.0))
            )

            if rho == 0.0:
                sigma_r = (m*besselj(0, m*rho) - m/2.0) * (term1r + term2r) + \
                          2*v[ii]*m*besselj(0, m*rho) * (C[ii]*math.exp(-m*(lambda_arr[ii]-lmm)) - D[ii]*math.exp(-m*(lmm - 0.0)))
                sigma_t = (m/2.0) * (term1r + term2r) + \
                          2*v[ii]*m*besselj(0, m*rho) * (C[ii]*math.exp(-m*(lambda_arr[ii]-lmm)) - D[ii]*math.exp(-m*(lmm - 0.0)))

        Results.append(dict(sigma_z=sigma_z, sigma_r=sigma_r, sigma_t=sigma_t, tau_rz=tau_rz, w=w, u=u))

    return Results


def AIO_wheel_superposition(
    E: List[float], v: List[float], alpha: float,
    PointsW1: List[Dict[str, float]], PointsW2: List[Dict[str, float]],
    Results_W1: List[Dict[str, float]], Results_W2: List[Dict[str, float]]
) -> List[Dict[str, float]]:
    # rotate (sigma_r, sigma_t) into global (x,y) by angle alpha and sum the two wheel effects
    def rot_xy(sr, st, a_deg):
        c = cosd(a_deg); s = sind(a_deg)
        Sigma_x = sr*(c**2) + st*(s**2)
        Sigma_y = sr*(s**2) + st*(c**2)
        return Sigma_x, Sigma_y

    # Upper (index 0) at upperlayer
    Sx1, Sy1 = rot_xy(Results_W1[0]["sigma_r"], Results_W1[0]["sigma_t"], alpha)
    Sx2, Sy2 = rot_xy(Results_W2[0]["sigma_r"], Results_W2[0]["sigma_t"], alpha)

    Sigma_x_L2 = Sx1 + Sx2
    Sigma_y_L2 = Sy1 + Sy2
    Sigma_z_L2 = Results_W1[0]["sigma_z"] + Results_W2[0]["sigma_z"]
    tau_xz_L2  = Results_W1[0]["tau_rz"] * cosd(alpha) + Results_W2[0]["tau_rz"] * cosd(alpha)

    # Hooke to get strains
    # Use the upperlayer index from PointsW1[0]
    LayerT = int(PointsW1[0]["layer"]) - 1
    E2 = E[LayerT]; v2 = v[LayerT]
    ex2 = (Sigma_x_L2 - v2*(Sigma_y_L2 + Sigma_z_L2))/E2
    ey2 = (Sigma_y_L2 - v2*(Sigma_x_L2 + Sigma_z_L2))/E2
    ez2 = (Sigma_z_L2 - v2*(Sigma_x_L2 + Sigma_y_L2))/E2

    upper = dict(
        sigma_z=Sigma_z_L2, sigma_t=Sigma_y_L2, sigma_r=Sigma_x_L2,
        tau_rz=tau_xz_L2, w=Results_W1[0]["w"] + Results_W2[0]["w"], u=0.0,
        epsilon_z=ez2, epsilon_t=ey2, epsilon_r=ex2
    )

    # Lower (index 1) at lowerlayer
    Sx1b, Sy1b = rot_xy(Results_W1[1]["sigma_r"], Results_W1[1]["sigma_t"], alpha)
    Sx2b, Sy2b = rot_xy(Results_W2[1]["sigma_r"], Results_W2[1]["sigma_t"], alpha)

    Sigma_x_L3 = Sx1b + Sx2b
    Sigma_y_L3 = Sy1b + Sy2b
    Sigma_z_L3 = Results_W1[1]["sigma_z"] + Results_W2[1]["sigma_z"]
    tau_xz_L3  = Results_W1[1]["tau_rz"] * cosd(alpha) + Results_W2[1]["tau_rz"] * cosd(alpha)

    LayerD = int(PointsW2[1]["layer"]) - 1
    E3 = E[LayerD]; v3 = v[LayerD]
    ex3 = (Sigma_x_L3 - v3*(Sigma_y_L3 + Sigma_z_L3))/E3
    ey3 = (Sigma_y_L3 - v3*(Sigma_x_L3 + Sigma_z_L3))/E3
    ez3 = (Sigma_z_L3 - v3*(Sigma_x_L3 + Sigma_y_L3))/E3

    lower = dict(
        sigma_z=Sigma_z_L3, sigma_t=Sigma_y_L3, sigma_r=Sigma_x_L3,
        tau_rz=tau_xz_L3, w=Results_W1[1]["w"] + Results_W2[1]["w"], u=0.0,
        epsilon_z=ez3, epsilon_t=ey3, epsilon_r=ex3
    )

    return [upper, lower]


# ------------------- presentation helpers -------------------

def crl_layer_name_by_type(Type: int, CRL: float) -> str:
    if CRL <= 0:
        return ""
    if Type in (2, 5):
        return "CRL (AIL)"
    if Type == 3:
        return "CRL (SAMI)"
    return "CRL"


def layer_names_by_type(Type: int, is_wmm_r: bool, is_gsb_r: bool) -> Tuple[str, str]:
    if Type == 1:
        return "Base (WMM)", "Subbase (GSB)"
    if Type == 4:
        return "Base (ETB)", "Subbase (CTSB)"
    if Type == 6:
        return "Base (WMM)", "Subbase (CTSB)"
    if Type == 7:
        base = "Base (Reinforced WMM)" if is_wmm_r else "Base (WMM)"
        sub  = "Subbase (Reinforced GSB)" if is_gsb_r else "Subbase (GSB)"
        return base, sub
    if Type in (2, 3):
        return "Base (CTB)", "Subbase (CTSB)"
    if Type == 5:
        return "Base (CTB)", "Subbase (GSB)"
    if Type == 8:
        return "Base (Reinforced)", "Subbase (CTSB)"
    return "Base", "Subbase"


def display_unified_results_table(Best: pd.DataFrame, Type: int, is_wmm_r: bool, is_gsb_r: bool):
    assert isinstance(Best, pd.DataFrame) and len(Best) == 1, "Best must be a 1-row DataFrame."

    BT = float(Best.loc[Best.index[0], "BT"])
    CRL = float(Best.loc[Best.index[0], "CRL"]) if "CRL" in Best.columns else 0.0
    Base = float(Best.loc[Best.index[0], "Base"])
    Subb = float(Best.loc[Best.index[0], "Subbase"])
    Cost = float(Best.loc[Best.index[0], "Cost"])
    Nbf  = float(Best.loc[Best.index[0], "Nbf"])
    Nsbr = float(Best.loc[Best.index[0], "Nsbr"])
    Nctf = float(Best.loc[Best.index[0], "Nctf"]) if "Nctf" in Best.columns else float("nan")
    CFD  = float(Best.loc[Best.index[0], "CFD"])  if "CFD" in Best.columns  else float("nan")

    # BC/DBM split
    if BT <= 40:
        BC, DBM = BT, 0.0
    elif BT < 90:
        BC, DBM = 30.0, BT - 30.0
    else:
        BC, DBM = 40.0, BT - 40.0

    baseName, subName = layer_names_by_type(Type, is_wmm_r, is_gsb_r)
    crlName = crl_layer_name_by_type(Type, CRL)

    ResultsTable = pd.DataFrame([{
        "BC_mm": BC, "DBM_mm": DBM,
        "CRL_Layer": crlName, "CRL_mm": CRL,
        "Base_Layer": baseName, "Base_mm": Base,
        "Subbase_Layer": subName, "Subbase_mm": Subb,
        "Nbf": Nbf, "Nsbr": Nsbr, "Nctf": Nctf, "CFD": CFD, "Cost": Cost
    }])

    # Build labels & thicknesses in the exact order you stack them
    layer_labels = [subName, baseName]
    layer_thk    = [Subb,    Base]
    if CRL > 0:
        layer_labels.append(crlName)
        layer_thk.append(CRL)
    layer_labels.extend(["DBM", "BC"])
    layer_thk.extend([DBM,    BC])

    # Fresh figure (important if function is called multiple times)
    fig = plt.figure("Pavement Thickness", figsize=(5, 4))
    fig.clf()
    ax = fig.add_subplot(111)

    # Draw the stacked bars and keep the handles we want in the legend
    bottom = 0.0
    handles, labels = [], []
    for label, thk in zip(layer_labels, layer_thk):
        if thk <= 0:
            continue
        bar = ax.bar([1], [thk], width=0.5, bottom=[bottom])
        handles.append(bar[0])
        labels.append(label)
        ax.text(1, bottom + thk/2.0, f"{int(thk)} mm", ha="center", va="center", fontsize=9)
        bottom += thk

    ax.set_xticks([1], [""])
    ax.set_ylabel("Thickness (mm)")
    ax.set_title("Pavement Layer Stack")
    ax.set_ylim(0, max(1.0, 1.1 * bottom))
    ax.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()

    return ResultsTable, fig


# ------------------- convenience helpers / demo -------------------

def export_results_to_csv(results_table: pd.DataFrame, path: str) -> str:
    """
    Save the 1-row results table to CSV and return the path.
    """
    if results_table is None or results_table.empty:
        raise ValueError("No results to export.")
    results_table.to_csv(path, index=False)
    return path


if __name__ == "__main__":
    # Minimal smoketest demo with plausible defaults
    Type = 8
    Best, ResultsTable, Shared, TRACE, T, hFig = run_unified_pavement_design(
        Type=Type,
        Design_Traffic=100.0,                   # msa
        Effective_Subgrade_CBR=15,            # %
        Reliability=90.0,                      # %
        Va=3.5,                                # %
        Vbe=11.5,                              # %
        BT_Mod=3000.0,                         # MPa, bituminous
        BC_cost=8000.0,                        # ₹/m^3 (illustrative)
        DBM_cost=7000.0,                       # ₹/m^3
        BC_DBM_width=4,                      # m
        Base_cost=3000.0,                      # ₹/m^3 (CTB)
        Subbase_cost=2000.0,                   # ₹/m^3 (CTSB)
        Base_Sub_width=7.0,                    # m
        cfdchk_UI=None,                           # enable CFD check
        FS_CTB_UI=None,                         # safety factor
        RF_UI=None,                             # reliability factor for CTB fatigue
        CRL_cost_UI=None,                    # ₹/m^3 (AIL)
        SAMI_cost_UI=None,                      # not used for Type 2
        Rtype_UI=1,                         # N/A for Type 2
        is_wmm_r_UI=1,                      # N/A for Type 2
        R_Base_UI=2,                        # N/A for Type 2
        is_gsb_r_UI=None,                      # N/A for Type 2
        R_Subbase_UI=None,                     # N/A for Type 2
        wmm_r_cost_UI=100,                    # N/A
        gsb_r_cost_UI=None,                    # N/A
        SA_M_UI=None,                          #np.array([[185, 195, 100000], [175, 185, 100000], ], dtype=float,),                           # single line axle mix (kN, kN, reps factor)
        TaA_M_UI=None,                         #np.array([[390, 410, 2000000], [370, 390, 2300000], ], dtype=float,),
        TrA_M_UI=None,                         #np.array([[585, 615, 3500000], [555, 585, 4000000], ], dtype=float,),
        AIL_Mod_UI=None,                       # MPa
        WMM_Mod_UI=300,                       # N/A
        ETB_Mod_UI=None,                       # N/A
        CTB_Mod_UI=None,                     # MPa
        CTSB_Mod_UI=600                      # MPa
    )
    print(Shared)


