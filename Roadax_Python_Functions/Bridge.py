# driver.py
from Design_by_type import run_unified_pavement_design
from Edit_type_to_check import hydrate_from_shared

# 1) Run the unified solver to get Shared
Best, ResultsTable, Shared, TRACE, T, hFig = run_unified_pavement_design(
    Type=2,
    Design_Traffic=50.0,
    Effective_Subgrade_CBR=15.0,
    Reliability=90.0,
    Va=3.5,
    Vbe=11.5,
    BT_Mod=3000.0,
    BC_cost=6000.0,
    DBM_cost=5000.0,
    BC_DBM_width=7.0,
    Base_cost=1800.0,
    Subbase_cost=1200.0,
    Base_Sub_width=7.0,
    cfdchk_UI=0,           # enable CFD if you want it checked
    FS_CTB_UI=1.4,         # CTB safety factor
    RF_UI=1,             # reliability factor for CTB fatigue
    CRL_cost_UI=3000,    # only used for Types 2 & 5
    SAMI_cost_UI=None,     # only used for Type 3
    Rtype_UI=None, is_wmm_r_UI=None, R_Base_UI=None,
    is_gsb_r_UI=None, R_Subbase_UI=None,
    wmm_r_cost_UI=None, gsb_r_cost_UI=None,
    SA_M_UI=None,  # example axle set (kN, kN, reps multiplier)
    TaA_M_UI=None, TrA_M_UI=None,
    AIL_Mod_UI=450, WMM_Mod_UI=None, ETB_Mod_UI=None,
    CTB_Mod_UI=5000, CTSB_Mod_UI=600
)

# 2) Feed Shared into hydrate_from_shared
Report_t, Cost_km, breakdown, Derived = hydrate_from_shared(Shared)

# 3) Use the outputs
print("Report_t (Permissible, Critical, Pass/Fail):")
print(Report_t)
print(f"Cost (₹ lakh/km): {Cost_km:.3f}")
print("Cost breakdown keys:", list(breakdown.keys()))
