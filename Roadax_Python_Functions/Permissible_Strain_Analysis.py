# permissible_si.py
import math
from typing import Optional, Tuple, List, Dict, Any


def compute_permissible_si(Design_Traffic: float,
                           Reliability: int,
                           Va: float,
                           Vbe: float,
                           BT_Mod: float,
                           Base_ctb: int,
                           Base_Mod: Optional[float],
                           RF_CTB: Optional[float]
                           ) -> Tuple[List[float], List[Optional[float]], Dict[str, Any]]:
    """
    Direct translation of MATLAB compute_permissible_si with identical logic.
    Returns:
      vec4  : [Bituminous; Base; 1; Subgrade] (microstrain; Base is NaN if no CTB)
      Perm_Si_R : [si_bfat, si_rut, si_cfat] in microstrain (si_cfat is None if no CTB)
      out   : dict with named fields mirroring MATLAB struct
    """

    # --------- input checks (no implicit defaults) ---------
    assert (isinstance(Design_Traffic, (int, float)) and Design_Traffic > 0), "Design_Traffic must be > 0."
    assert Reliability in (80, 90), "Reliability must be 80 or 90."
    assert isinstance(Va, (int, float)) and isinstance(Vbe, (int, float)), "Va and Vbe must be scalars."
    assert isinstance(BT_Mod, (int, float)) and BT_Mod > 0, "BT_Mod must be > 0."
    assert Base_ctb in (0, 1), "Base_ctb must be 0 or 1."
    if Base_ctb == 1:
        assert (Base_Mod is not None and isinstance(Base_Mod, (int, float)) and Base_Mod > 0), \
            "Base_Mod is required and must be >0 when Base_ctb==1."
        assert (RF_CTB is not None and isinstance(RF_CTB, (int, float)) and RF_CTB > 0), \
            "RF_CTB is required and must be >0 when Base_ctb==1."
    else:
        # must be "empty" when CTB not present
        assert Base_Mod is None, "Base_Mod must be None when Base_ctb==0."
        assert RF_CTB is None, "RF_CTB must be None when Base_ctb==0."

    # --------- compute N_design and permissible strains ---------
    N_design = Design_Traffic * 1e6

    if Base_ctb == 1:
        Permissible_Si = AIO_PermissibleSi(Reliability, N_design, BT_Mod, Va, Vbe, Base_Mod, RF_CTB)
    else:
        # pass "empty" for Base_Mod and RF -> si_cfat = None
        Permissible_Si = AIO_PermissibleSi(Reliability, N_design, BT_Mod, Va, Vbe, None, None)

    # scale to microstrain (MATLAB: Permissible_Si .* 1e6)
    # preserve None for third element if no CTB
    Permissible_Si_R: List[Optional[float]] = [
        Permissible_Si[0] * 1e6 if Permissible_Si[0] is not None else None,
        Permissible_Si[1] * 1e6 if Permissible_Si[1] is not None else None,
        (Permissible_Si[2] * 1e6) if Permissible_Si[2] is not None else None,
    ]

    # Map to named outputs; if si_cfat is "[]", store NaN in vec4 (numeric)
    Si_bfat_micro = float(Permissible_Si_R[0])  # always numeric
    Si_rut_micro = float(Permissible_Si_R[1])   # always numeric
    if Permissible_Si_R[2] is not None:
        Si_cfat_micro = float(Permissible_Si_R[2])
    else:
        Si_cfat_micro = float("nan")  # MATLAB would have [] but 4x1 numeric vector needs NaN

    # Final 4x1 output: [Bituminous; Base; 1; Subgrade]
    vec4 = [Si_bfat_micro, Si_cfat_micro, 1.0, Si_rut_micro]

    out = {
        "Bituminous_micro": Si_bfat_micro,
        "Subgrade_micro": Si_rut_micro,
        "Base_micro": Si_cfat_micro,
        "Vector4": vec4,
    }
    return vec4, Permissible_Si_R, out


def AIO_PermissibleSi(reliability: int,
                      N_design: float,
                      BT_Mod: float,
                      Va: float,
                      Vbe: float,
                      Base_Mod: Optional[float],
                      RF: Optional[float]) -> List[Optional[float]]:
    """
    Returns [si_bfat, si_rut, si_cfat] in STRAIN (not micro), matching MATLAB.
    If Base_Mod is None (no CTB), si_cfat is returned as None (MATLAB []).
    """
    # M, C as in MATLAB
    M = 4.84 * ((Vbe / (Va + Vbe)) - 0.69)
    C = 10 ** M

    if reliability == 80:
        si_rut_permissible = 1.0 / ((N_design / (4.1656e-8)) ** (1.0 / 4.5337))
        si_bfat_permissible = 1.0 / ((N_design / ((C * 1.6064e-4) * ((1.0 / BT_Mod) ** 0.854))) ** (1.0 / 3.89))
    else:  # reliability == 90
        si_rut_permissible = 1.0 / ((N_design / (1.41e-8)) ** (1.0 / 4.5337))
        si_bfat_permissible = 1.0 / ((N_design / ((C * 0.5161e-4) * ((1.0 / BT_Mod) ** 0.854))) ** (1.0 / 3.89))

    if Base_Mod is None:
        si_cfat_permissible: Optional[float] = None  # MATLAB []
    else:
        # RF must be provided together with Base_Mod
        si_cfat_permissible = (((113000.0 / (Base_Mod ** 0.804)) + 191.0) / ((N_design / RF) ** (1.0 / 12.0))) * 1e-6

    return [si_bfat_permissible, si_rut_permissible, si_cfat_permissible]


if __name__ == "__main__":
    # -------- inputs (exactly as your MATLAB script) --------
    Design_Traffic = 300    # msa
    Reliability = 80        # 80 or 90
    Va = 3.5
    Vbe = 11.5
    BT_Mod = 3000

    # Base (CTB) present
    Base_ctb = 1         # if CTB then 1 else 0
    Base_Mod = 600.0     # MPa if CBT zero, consider None
    RF_CTB = 1.0         # if CBT zero, consider None

    vec4, Perm_Si_R, out = compute_permissible_si(
        Design_Traffic, Reliability, Va, Vbe, BT_Mod,
        Base_ctb, Base_Mod, RF_CTB
    )

    print("4x1 Output Vector = [Bituminous; Base; 1; Subgrade]")
    # Pretty print like MATLAB disp
    for val in vec4:
        if math.isnan(val):
            print("NaN")
        else:
            print(f"{val:.6f}")
