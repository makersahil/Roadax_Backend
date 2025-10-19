# criticals_with_details.py
# Exact MATLAB-to-Python conversion (no logic changes).
# Uses SciPy's jv aliased as besselj to match MATLAB besselj calls.

import math
from typing import List, Dict, Any, Tuple

import numpy as np
# pandas is not required by the core logic; make it optional to avoid hard dependency
try:
    import pandas as pd  # type: ignore  # (not used by the core)
except Exception:  # pragma: no cover
    pd = None  # optional
from scipy.special import jv as besselj  # Bessel J_v(x)


def cosd(x: float) -> float:
    return math.cos(math.radians(x))


def sind(x: float) -> float:
    return math.sin(math.radians(x))


def exact_criticals_with_details(
    Number_of_layers: int,
    Thickness_layers: List[float],
    Modulus_layers: List[float],
    Poissons: List[float],
    Eva_depth_bituminous: float,
    Eva_depth_base: float,
    Eva_depth_Subgrade: float,
    CFD_Check: int,
    FS_CTB_T: float,
    SA_M_T: np.ndarray,
    TaA_M_T: np.ndarray,
    TrA_M_T: np.ndarray,
) -> np.ndarray:
    # --------------------------- start of original logic ---------------------------
    interface_depths = np.cumsum(Thickness_layers)

    # --- Bituminous interface ---
    idx_bt_arr = np.where(interface_depths == Eva_depth_bituminous)[0]
    if idx_bt_arr.size == 0:
        raise ValueError("Eva_depth must coincide with a layer interface depth.")
    else:
        idx_bt = int(idx_bt_arr[0]) + 1  # MATLAB 1-based
        upper_layer_bt = idx_bt
        lower_layer_bt = idx_bt + 1

    isbonded = True
    alpha = 0.0
    Tyre_pressure = 0.56
    wheel_load = 20000.0

    Mat_SigmaSi_BT = AIO_SigmaSi(
        Number_of_layers,
        Thickness_layers,
        Modulus_layers,
        Poissons,
        isbonded,
        Tyre_pressure,
        wheel_load,
        alpha,
        Eva_depth_bituminous,
        upper_layer_bt,
        lower_layer_bt,
    )

    Critical_BSi = np.max(np.abs(Mat_SigmaSi_BT[:, 8:10])) * (10**6)

    # --- Subgrade interface ---
    idx_subgrade_arr = np.where(interface_depths == Eva_depth_Subgrade)[0]
    if idx_subgrade_arr.size == 0:
        raise ValueError("Eva_depth must coincide with a layer interface depth.")
    else:
        idx_subgrade = int(idx_subgrade_arr[0]) + 1  # MATLAB 1-based
        upper_layer_subgrade = idx_subgrade
        lower_layer_subgrade = idx_subgrade + 1

    MatSR = AIO_SigmaSi(
        Number_of_layers,
        Thickness_layers,
        Modulus_layers,
        Poissons,
        isbonded,
        Tyre_pressure,
        wheel_load,
        alpha,
        Eva_depth_Subgrade,
        upper_layer_subgrade,
        lower_layer_subgrade,
    )

    Critical_SubgradeSi = np.max(np.abs(MatSR[:, 7])) * (10**6)

    # --- Base interface ---
    idx_base_arr = np.where(interface_depths == Eva_depth_base)[0]
    if idx_base_arr.size == 0:
        raise ValueError("Eva_depth must coincide with a layer interface depth.")
    else:
        idx_base = int(idx_base_arr[0]) + 1  # MATLAB 1-based
        upper_layer_base = idx_base
        lower_layer_base = idx_base + 1

    typre_c = 0.8  # MPa
    MatBase = AIO_SigmaSi(
        Number_of_layers,
        Thickness_layers,
        Modulus_layers,
        Poissons,
        isbonded,
        typre_c,
        wheel_load,
        alpha,
        Eva_depth_base,
        upper_layer_base,
        lower_layer_base,
    )

    Critical_CTBSi = np.max(np.abs(MatBase[:, 8:10])) * (10**6)

    if CFD_Check == 1:
        n_ca = Number_of_layers
        Thickness_ca = Thickness_layers
        E_ca = Modulus_layers
        v_ca = Poissons
        isbonded_ca = True
        Eva_depth_ca = Eva_depth_base
        typre_ca = 0.8
        alpha_ca = 0.0
        top_ca = upper_layer_base
        bot_ca = lower_layer_base

        CFD_SA = 0.0
        CFD_TaA = 0.0
        CFD_TrA = 0.0

        SA_M = np.array(SA_M_T, dtype=float)
        TaA_M = np.array(TaA_M_T, dtype=float)
        TrA_M = np.array(TrA_M_T, dtype=float)
        FS_CTB = FS_CTB_T

        # Single axle
        for i in range(SA_M.shape[0]):
            load_ca1 = np.mean(SA_M[i, 0:2]) * 1000.0 / 4.0
            reps = SA_M[i, 2]
            Mat = AIO_SigmaSi(
                n_ca,
                Thickness_ca,
                E_ca,
                v_ca,
                isbonded_ca,
                typre_ca,
                load_ca1,
                alpha_ca,
                Eva_depth_ca,
                top_ca,
                bot_ca,
            )
            sig = np.max(np.abs(Mat[:, 3:5]))
            SR = sig / FS_CTB
            NF = 10 ** ((0.972 - SR) / 0.0825)
            CFD_SA = CFD_SA + reps / NF

        # Tandem axle
        for i in range(TaA_M.shape[0]):
            load_ca2 = np.mean(TaA_M[i, 0:2]) * 1000.0 / 8.0
            reps = 2.0 * TaA_M[i, 2]
            Mat = AIO_SigmaSi(
                n_ca,
                Thickness_ca,
                E_ca,
                v_ca,
                isbonded_ca,
                typre_ca,
                load_ca2,
                alpha_ca,
                Eva_depth_ca,
                top_ca,
                bot_ca,
            )
            sig = np.max(np.abs(Mat[:, 3:5]))
            SR = sig / FS_CTB
            NF = 10 ** ((0.972 - SR) / 0.0825)
            CFD_TaA = CFD_TaA + reps / NF
            if CFD_SA + CFD_TaA > 1:
                break

        # Tridem axle
        for i in range(TrA_M.shape[0]):
            load_ca3 = np.mean(TrA_M[i, 0:2]) * 1000.0 / 12.0
            reps = 3.0 * TrA_M[i, 2]
            Mat = AIO_SigmaSi(
                n_ca,
                Thickness_ca,
                E_ca,
                v_ca,
                isbonded_ca,
                typre_ca,
                load_ca3,
                alpha_ca,
                Eva_depth_ca,
                top_ca,
                bot_ca,
            )
            sig = np.max(np.abs(Mat[:, 3:5]))
            SR = sig / FS_CTB
            NF = 10 ** ((0.972 - SR) / 0.0825)
            CFD_TrA = CFD_TrA + reps / NF
            if CFD_SA + CFD_TaA + CFD_TrA > 1:
                break

        CFD = CFD_SA + CFD_TaA + CFD_TrA
    else:
        CFD = float("nan")

    # Final 4x1 vector
    out_vec = np.array([Critical_BSi, Critical_CTBSi, CFD, Critical_SubgradeSi], dtype=float)
    return out_vec


# ================== Local subfunctions (unchanged core) ==================
def AIO_SigmaSi(
    n: int,
    Thickness: List[float],
    E: List[float],
    v: List[float],
    isbonded: bool,
    typre: float,
    load: float,
    alpha: float,
    Eva_depth: float,
    upperlayer: int,
    lowerlayer: int,
) -> np.ndarray:

    PointsW1O = [
        {"r": 0.0, "z": Eva_depth, "layer": int(upperlayer)},
        {"r": 0.0, "z": Eva_depth, "layer": int(lowerlayer)},
    ]

    q = typre
    a = math.sqrt(load / (math.pi * q))

    Results_W1_O = AIO_R(n, Thickness, E, v, isbonded, PointsW1O, a, q)

    PointsW2O = [
        {"r": 310.0, "z": Eva_depth, "layer": int(upperlayer)},
        {"r": 310.0, "z": Eva_depth, "layer": int(lowerlayer)},
    ]

    Results_W2_O = AIO_R(n, Thickness, E, v, isbonded, PointsW2O, a, q)
    Results_O = AIO_wheel_superposition(E, v, alpha, PointsW1O, PointsW2O, Results_W1_O, Results_W2_O)

    PointsW1C = [
        {"r": 155.0, "z": Eva_depth, "layer": int(upperlayer)},
        {"r": 155.0, "z": Eva_depth, "layer": int(lowerlayer)},
    ]

    Results_W1_C = AIO_R(n, Thickness, E, v, isbonded, PointsW1C, a, q)

    PointsW2C = [
        {"r": 155.0, "z": Eva_depth, "layer": int(upperlayer)},
        {"r": 155.0, "z": Eva_depth, "layer": int(lowerlayer)},
    ]

    Results_W2_C = AIO_R(n, Thickness, E, v, isbonded, PointsW2C, a, q)
    Results_C = AIO_wheel_superposition(E, v, alpha, PointsW1C, PointsW2C, Results_W1_C, Results_W2_C)

    Mat_SigmaSi = np.array(
        [
            [
                Eva_depth,
                PointsW1O[0]["r"],
                Results_O[0]["sigma_z"],
                Results_O[0]["sigma_t"],
                Results_O[0]["sigma_r"],
                Results_O[0]["tau_rz"],
                Results_O[0]["w"],
                Results_O[0]["epsilon_z"],
                Results_O[0]["epsilon_t"],
                Results_O[0]["epsilon_r"],
            ],
            [
                Eva_depth,
                PointsW1O[1]["r"],
                Results_O[1]["sigma_z"],
                Results_O[1]["sigma_t"],
                Results_O[1]["sigma_r"],
                Results_O[1]["tau_rz"],
                Results_O[1]["w"],
                Results_O[1]["epsilon_z"],
                Results_O[1]["epsilon_t"],
                Results_O[1]["epsilon_r"],
            ],
            [
                Eva_depth,
                PointsW1C[0]["r"],
                Results_C[0]["sigma_z"],
                Results_C[0]["sigma_t"],
                Results_C[0]["sigma_r"],
                Results_C[0]["tau_rz"],
                Results_C[0]["w"],
                Results_C[0]["epsilon_z"],
                Results_C[0]["epsilon_t"],
                Results_C[0]["epsilon_r"],
            ],
            [
                Eva_depth,
                PointsW1C[1]["r"],
                Results_C[1]["sigma_z"],
                Results_C[1]["sigma_t"],
                Results_C[1]["sigma_r"],
                Results_C[1]["tau_rz"],
                Results_C[1]["w"],
                Results_C[1]["epsilon_z"],
                Results_C[1]["epsilon_t"],
                Results_C[1]["epsilon_r"],
            ],
        ],
        dtype=float,
    )

    return Mat_SigmaSi


def AIO_R(
    n: int,
    Thickness: List[float],
    E: List[float],
    v: List[float],
    isbonded: bool,
    Points: List[Dict[str, float]],
    a: float,
    q: float,
) -> List[Dict[str, float]]:

    z = np.cumsum(np.array(Thickness, dtype=float))
    H = float(z[n - 2])  # last interface depth

    alpha = a / H
    dl = 0.2 / alpha

    Results: List[Dict[str, float]] = []
    for _ in range(len(Points)):
        Results.append(
            {"sigma_z": 0.0, "sigma_r": 0.0, "sigma_t": 0.0, "tau_rz": 0.0, "w": 0.0, "u": 0.0}
        )

    # integrate m from 0:dl:200 with 4-pt Gauss rule inside each slab (as in MATLAB)
    l_vals = np.arange(0.0, 200.0 + 1e-12, dl)
    for l in l_vals:
        for gauss_point in (1, 2, 3, 4):
            if gauss_point == 1:
                m = (l + dl / 2.0) - 0.86114 * (dl / 2.0)
                fc = 0.34786 * (dl / 2.0)
            elif gauss_point == 2:
                m = (l + dl / 2.0) - 0.33998 * (dl / 2.0)
                fc = 0.65215 * (dl / 2.0)
            elif gauss_point == 3:
                m = (l + dl / 2.0) + 0.33998 * (dl / 2.0)
                fc = 0.65215 * (dl / 2.0)
            else:  # gauss_point == 4
                m = (l + dl / 2.0) + 0.86114 * (dl / 2.0)
                fc = 0.34786 * (dl / 2.0)

            Result = AIO_R_hat(n, Thickness, E, v, isbonded, m, Points)

            if m != 0.0:
                J1 = float(besselj(1, m * alpha))
                for j in range(len(Points)):
                    Results[j]["sigma_z"] += fc * (q * alpha * Result[j]["sigma_z"] / m * J1)
                    Results[j]["sigma_r"] += fc * (q * alpha * Result[j]["sigma_r"] / m * J1)
                    Results[j]["sigma_t"] += fc * (q * alpha * Result[j]["sigma_t"] / m * J1)
                    Results[j]["tau_rz"] += fc * (q * alpha * Result[j]["tau_rz"] / m * J1)
                    Results[j]["w"] += fc * (q * alpha * Result[j]["w"] / m * J1)
                    Results[j]["u"] += fc * (q * alpha * Result[j]["u"] / m * J1)

    # strains
    for j in range(len(Points)):
        ii = int(Points[j]["layer"])  # MATLAB 1-based
        Ei = float(E[ii - 1])
        vi = float(v[ii - 1])

        sz = Results[j]["sigma_z"]
        sr = Results[j]["sigma_r"]
        st = Results[j]["sigma_t"]

        Results[j]["epsilon_z"] = (sz - vi * (sr + st)) / Ei
        Results[j]["epsilon_r"] = (sr - vi * (sz + st)) / Ei
        Results[j]["epsilon_t"] = (st - vi * (sr + sz)) / Ei

    return Results


def AIO_R_hat(
    n: int,
    Thickness: List[float],
    E: List[float],
    v: List[float],
    isbonded: bool,
    m: float,
    Points: List[Dict[str, float]],
) -> List[Dict[str, float]]:

    z = np.cumsum(np.array(Thickness, dtype=float))
    H = float(z[n - 2])

    lam = np.zeros(n, dtype=float)
    lam[: n - 1] = z / H
    lam[n - 1] = np.inf

    F = np.zeros(n, dtype=float)
    R = np.zeros(n - 1, dtype=float)

    # 1-based equivalents
    F[0] = math.exp(-m * (lam[0] - 0.0))
    R[0] = (E[0] / E[1]) * (1.0 + v[1]) / (1.0 + v[0])

    for i1b in range(2, n):  # i = 2..n-1 (1-based)
        i = i1b - 1  # 0-based
        F[i] = math.exp(-m * (lam[i] - lam[i - 1]))
        R[i] = (E[i] / E[i + 1]) * (1.0 + v[i + 1]) / (1.0 + v[i])

    F[n - 1] = math.exp(-m * (lam[n - 1] - lam[n - 2]))

    M_list: List[np.ndarray] = []

    for i1b in range(1, n):  # i = 1..n-1 (1-based)
        i = i1b - 1  # 0-based

        if isbonded:
            M1 = np.zeros((4, 4), dtype=float)
            M2 = np.zeros((4, 4), dtype=float)

            M1[0, 0] = 1
            M1[1, 0] = 1
            M1[2, 0] = 1
            M1[3, 0] = 1

            M1[0, 1] = F[i]
            M1[1, 1] = -F[i]
            M1[2, 1] = F[i]
            M1[3, 1] = -F[i]

            M1[0, 2] = -(1 - 2 * v[i] - m * lam[i])
            M1[1, 2] = 2 * v[i] + m * lam[i]
            M1[2, 2] = 1 + m * lam[i]
            M1[3, 2] = -(2 - 4 * v[i] - m * lam[i])

            M1[0, 3] = (1 - 2 * v[i] + m * lam[i]) * F[i]
            M1[1, 3] = (2 * v[i] - m * lam[i]) * F[i]
            M1[2, 3] = -(1 - m * lam[i]) * F[i]
            M1[3, 3] = -(2 - 4 * v[i] + m * lam[i]) * F[i]

            M2[0, 0] = F[i + 1]
            M2[1, 0] = F[i + 1]
            M2[2, 0] = R[i] * F[i + 1]
            M2[3, 0] = R[i] * F[i + 1]

            M2[0, 1] = 1
            M2[1, 1] = -1
            M2[2, 1] = R[i]
            M2[3, 1] = -R[i]

            M2[0, 2] = -(1 - 2 * v[i + 1] - m * lam[i]) * F[i + 1]
            M2[1, 2] = (2 * v[i + 1] + m * lam[i]) * F[i + 1]
            M2[2, 2] = (1 + m * lam[i]) * R[i] * F[i + 1]
            M2[3, 2] = -(2 - 4 * v[i + 1] - m * lam[i]) * R[i] * F[i + 1]

            M2[0, 3] = 1 - 2 * v[i + 1] + m * lam[i]
            M2[1, 3] = (2 * v[i + 1] - m * lam[i])
            M2[2, 3] = -(1 - m * lam[i]) * R[i]
            M2[3, 3] = -(2 - 4 * v[i + 1] + m * lam[i]) * R[i]
        else:
            M1 = np.zeros((4, 4), dtype=float)
            M2 = np.zeros((4, 4), dtype=float)

            M1[0, 0] = 1
            M1[1, 0] = 1
            M1[2, 0] = 1
            M1[3, 0] = 0

            M1[0, 1] = F[i]
            M1[1, 1] = F[i]
            M1[2, 1] = -F[i]
            M1[3, 1] = 0

            M1[0, 2] = -(1 - 2 * v[i] - m * lam[i])
            M1[1, 2] = 1 + m * lam[i]
            M1[2, 2] = 2 * v[i] + m * lam[i]
            M1[3, 2] = 0

            M1[0, 3] = (1 - 2 * v[i] + m * lam[i]) * F[i]
            M1[1, 3] = -(1 - m * lam[i]) * F[i]
            M1[2, 3] = (2 * v[i] - m * lam[i]) * F[i]
            M1[3, 3] = 0

            M2[0, 0] = F[i + 1]
            M2[1, 0] = R[i] * F[i + 1]
            M2[2, 0] = 0
            M2[3, 0] = F[i + 1]

            M2[0, 1] = 1
            M2[1, 1] = R[i]
            M2[2, 1] = 0
            M2[3, 1] = -1

            M2[0, 2] = -(1 - 2 * v[i + 1] - m * lam[i]) * F[i + 1]
            M2[1, 2] = (1 + m * lam[i]) * R[i] * F[i + 1]
            M2[2, 2] = 0
            M2[3, 2] = (2 * v[i + 1] + m * lam[i]) * F[i + 1]

            M2[0, 3] = 1 - 2 * v[i + 1] + m * lam[i]
            M2[1, 3] = -(1 - m * lam[i]) * R[i]
            M2[2, 3] = 0
            M2[3, 3] = 2 * v[i + 1] - m * lam[i]

        X = np.linalg.solve(M1, M2)
        M_list.append(X)

    MM = np.diag([1.0, 1.0, 1.0, 1.0])
    for i in range(n - 1):
        MM = MM @ M_list[i]
    MM = MM[:, [1, 3]]  # columns 2 and 4 (1-based)

    b11 = math.exp(-lam[0] * m)
    b21 = math.exp(-lam[0] * m)
    b12 = 1.0
    b22 = -1.0

    c11 = -(1.0 - 2.0 * v[0]) * math.exp(-m * lam[0])
    c21 = 2.0 * v[0] * math.exp(-m * lam[0])
    c12 = 1.0 - 2.0 * v[0]
    c22 = 2.0 * v[0]

    k11 = b11 * MM[0, 0] + b12 * MM[1, 0] + c11 * MM[2, 0] + c12 * MM[3, 0]
    k12 = b11 * MM[0, 1] + b12 * MM[1, 1] + c11 * MM[2, 1] + c12 * MM[3, 1]
    k21 = b21 * MM[0, 0] + b22 * MM[1, 0] + c21 * MM[2, 0] + c22 * MM[3, 0]
    k22 = b21 * MM[0, 1] + b22 * MM[1, 1] + c21 * MM[2, 1] + c22 * MM[3, 1]

    A = np.zeros(n, dtype=float)
    B = np.zeros(n, dtype=float)
    C = np.zeros(n, dtype=float)
    D = np.zeros(n, dtype=float)

    A[n - 1] = 0.0
    denom = (k11 * k22 - k12 * k21)
    B[n - 1] = k22 / denom
    C[n - 1] = 0.0
    D[n - 1] = 1.0 / (k12 - k22 * k11 / k21)

    for i in range(n - 2, -1, -1):  # i = n-1..1 (1-based)
        XX = M_list[i] @ np.array([A[i + 1], B[i + 1], C[i + 1], D[i + 1]], dtype=float)
        A[i] = XX[0]
        B[i] = XX[1]
        C[i] = XX[2]
        D[i] = XX[3]

    Results: List[Dict[str, float]] = []
    for j in range(len(Points)):
        rho = Points[j]["r"] / H
        lmm = Points[j]["z"] / H
        ii = int(Points[j]["layer"])  # 1..n

        J0 = float(besselj(0, m * rho))
        J1 = float(besselj(1, m * rho))

        if ii != 1:
            term1 = (A[ii - 1] - C[ii - 1] * (1 - 2 * v[ii - 1] - m * lmm)) * math.exp(-m * (lam[ii - 1] - lmm))
            term2 = (B[ii - 1] + D[ii - 1] * (1 - 2 * v[ii - 1] + m * lmm)) * math.exp(
                -m * (lmm - lam[ii - 2])
            )

            sigma_z = -m * J0 * (term1 + term2)

            if rho == 0.0:
                sigma_r = (m * J0 - m / 2.0) * (
                    (A[ii - 1] + C[ii - 1] * (1 + m * lmm)) * math.exp(-m * (lam[ii - 1] - lmm))
                    + (B[ii - 1] - D[ii - 1] * (1 - m * lmm)) * math.exp(-m * (lmm - lam[ii - 2]))
                ) + 2 * v[ii - 1] * m * J0 * (
                    C[ii - 1] * math.exp(-m * (lam[ii - 1] - lmm)) - D[ii - 1] * math.exp(-m * (lmm - lam[ii - 2]))
                )
                sigma_t = (m / 2.0) * (
                    (A[ii - 1] + C[ii - 1] * (1 + m * lmm)) * math.exp(-m * (lam[ii - 1] - lmm))
                    + (B[ii - 1] - D[ii - 1] * (1 - m * lmm)) * math.exp(-m * (lmm - lam[ii - 2]))
                ) + 2 * v[ii - 1] * m * J0 * (
                    C[ii - 1] * math.exp(-m * (lam[ii - 1] - lmm)) - D[ii - 1] * math.exp(-m * (lmm - lam[ii - 2]))
                )
            else:
                sigma_r = (m * J0 - J1 / rho) * (
                    (A[ii - 1] + C[ii - 1] * (1 + m * lmm)) * math.exp(-m * (lam[ii - 1] - lmm))
                    + (B[ii - 1] - D[ii - 1] * (1 - m * lmm)) * math.exp(-m * (lmm - lam[ii - 2]))
                ) + 2 * v[ii - 1] * m * J0 * (
                    C[ii - 1] * math.exp(-m * (lam[ii - 1] - lmm)) - D[ii - 1] * math.exp(-m * (lmm - lam[ii - 2]))
                )
                sigma_t = (J1 / rho) * (
                    (A[ii - 1] + C[ii - 1] * (1 + m * lmm)) * math.exp(-m * (lam[ii - 1] - lmm))
                    + (B[ii - 1] - D[ii - 1] * (1 - m * lmm)) * math.exp(-m * (lmm - lam[ii - 2]))
                ) + 2 * v[ii - 1] * m * J0 * (
                    C[ii - 1] * math.exp(-m * (lam[ii - 1] - lmm)) - D[ii - 1] * math.exp(-m * (lmm - lam[ii - 2]))
                )

            tau_rz = m * J1 * (
                (A[ii - 1] + C[ii - 1] * (2 * v[ii - 1] + m * lmm)) * math.exp(-m * (lam[ii - 1] - lmm))
                - (B[ii - 1] - D[ii - 1] * (2 * v[ii - 1] - m * lmm)) * math.exp(-m * (lmm - lam[ii - 2]))
            )
            w = -H * (1 + v[ii - 1]) / E[ii - 1] * J0 * (
                (A[ii - 1] - C[ii - 1] * (2 - 4 * v[ii - 1] - m * lmm)) * math.exp(-m * (lam[ii - 1] - lmm))
                - (B[ii - 1] + D[ii - 1] * (2 - 4 * v[ii - 1] + m * lmm)) * math.exp(-m * (lmm - lam[ii - 2]))
            )
            u = H * (1 + v[ii - 1]) / E[ii - 1] * J1 * (
                (A[ii - 1] + C[ii - 1] * (1 + m * lmm)) * math.exp(-m * (lam[ii - 1] - lmm))
                + (B[ii - 1] - D[ii - 1] * (1 - m * lmm)) * math.exp(-m * (lmm - lam[ii - 2]))
            )
        else:
            term1 = (A[ii - 1] - C[ii - 1] * (1 - 2 * v[ii - 1] - m * lmm)) * math.exp(-m * (lam[ii - 1] - lmm))
            term2 = (B[ii - 1] + D[ii - 1] * (1 - 2 * v[ii - 1] + m * lmm)) * math.exp(-m * (lmm - 0.0))

            sigma_z = -m * J0 * (term1 + term2)

            if rho == 0.0:
                sigma_r = (m * J0 - m / 2.0) * (
                    (A[ii - 1] + C[ii - 1] * (1 + m * lmm)) * math.exp(-m * (lam[ii - 1] - lmm))
                    + (B[ii - 1] - D[ii - 1] * (1 - m * lmm)) * math.exp(-m * (lmm - 0.0))
                ) + 2 * v[ii - 1] * m * J0 * (
                    C[ii - 1] * math.exp(-m * (lam[ii - 1] - lmm)) - D[ii - 1] * math.exp(-m * (lmm - 0.0))
                )
                sigma_t = (m / 2.0) * (
                    (A[ii - 1] + C[ii - 1] * (1 + m * lmm)) * math.exp(-m * (lam[ii - 1] - lmm))
                    + (B[ii - 1] - D[ii - 1] * (1 - m * lmm)) * math.exp(-m * (lmm - 0.0))
                ) + 2 * v[ii - 1] * m * J0 * (
                    C[ii - 1] * math.exp(-m * (lam[ii - 1] - lmm)) - D[ii - 1] * math.exp(-m * (lmm - 0.0))
                )
            else:
                sigma_r = (m * J0 - J1 / rho) * (
                    (A[ii - 1] + C[ii - 1] * (1 + m * lmm)) * math.exp(-m * (lam[ii - 1] - lmm))
                    + (B[ii - 1] - D[ii - 1] * (1 - m * lmm)) * math.exp(-m * (lmm - 0.0))
                ) + 2 * v[ii - 1] * m * J0 * (
                    C[ii - 1] * math.exp(-m * (lam[ii - 1] - lmm)) - D[ii - 1] * math.exp(-m * (lmm - 0.0))
                )
                sigma_t = (J1 / rho) * (
                    (A[ii - 1] + C[ii - 1] * (1 + m * lmm)) * math.exp(-m * (lam[ii - 1] - lmm))
                    + (B[ii - 1] - D[ii - 1] * (1 - m * lmm)) * math.exp(-m * (lmm - 0.0))
                ) + 2 * v[ii - 1] * m * J0 * (
                    C[ii - 1] * math.exp(-m * (lam[ii - 1] - lmm)) - D[ii - 1] * math.exp(-m * (lmm - 0.0))
                )

            tau_rz = m * J1 * (
                (A[ii - 1] + C[ii - 1] * (2 * v[ii - 1] + m * lmm)) * math.exp(-m * (lam[ii - 1] - lmm))
                - (B[ii - 1] - D[ii - 1] * (2 * v[ii - 1] - m * lmm)) * math.exp(-m * (lmm - 0.0))
            )
            w = -H * (1 + v[ii - 1]) / E[ii - 1] * J0 * (
                (A[ii - 1] - C[ii - 1] * (2 - 4 * v[ii - 1] - m * lmm)) * math.exp(-m * (lam[ii - 1] - lmm))
                - (B[ii - 1] + D[ii - 1] * (2 - 4 * v[ii - 1] + m * lmm)) * math.exp(-m * (lmm - 0.0))
            )
            u = H * (1 + v[ii - 1]) / E[ii - 1] * J1 * (
                (A[ii - 1] + C[ii - 1] * (1 + m * lmm)) * math.exp(-m * (lam[ii - 1] - lmm))
                + (B[ii - 1] - D[ii - 1] * (1 - m * lmm)) * math.exp(-m * (lmm - 0.0))
            )

        Results.append(
            {"sigma_z": sigma_z, "sigma_r": sigma_r, "sigma_t": sigma_t, "tau_rz": tau_rz, "w": w, "u": u}
        )

    return Results


def AIO_wheel_superposition(
    E: List[float],
    v: List[float],
    alpha: float,
    PointsW1: List[Dict[str, float]],
    PointsW2: List[Dict[str, float]],
    Results_W1: List[Dict[str, float]],
    Results_W2: List[Dict[str, float]],
) -> List[Dict[str, float]]:
    # helper rotation
    def rot_xy(sr: float, st: float, a: float) -> Tuple[float, float]:
        return (
            sr * (cosd(a) ** 2) + st * (sind(a) ** 2),  # Sigma_x
            sr * (sind(a) ** 2) + st * (cosd(a) ** 2),  # Sigma_y
        )

    LayerT = int(PointsW1[0]["layer"])
    LayerD = int(PointsW2[1]["layer"])

    # UPPER (index 1)
    Sx1, Sy1 = rot_xy(Results_W1[0]["sigma_r"], Results_W1[0]["sigma_t"], alpha)
    Sx2, Sy2 = rot_xy(Results_W2[0]["sigma_r"], Results_W2[0]["sigma_t"], alpha)

    Sigma_x_L2 = Sx1 + Sx2
    Sigma_y_L2 = Sy1 + Sy2
    Sigma_z_L2 = Results_W1[0]["sigma_z"] + Results_W2[0]["sigma_z"]
    tau_xz_L2 = Results_W1[0]["tau_rz"] * cosd(alpha) + Results_W2[0]["tau_rz"] * cosd(alpha)

    E2 = float(E[LayerT - 1])
    v2 = float(v[LayerT - 1])
    ex2 = (Sigma_x_L2 - v2 * (Sigma_y_L2 + Sigma_z_L2)) / E2
    ey2 = (Sigma_y_L2 - v2 * (Sigma_x_L2 + Sigma_z_L2)) / E2
    ez2 = (Sigma_z_L2 - v2 * (Sigma_x_L2 + Sigma_y_L2)) / E2

    out1 = {
        "sigma_z": Sigma_z_L2,
        "sigma_t": Sigma_y_L2,
        "sigma_r": Sigma_x_L2,
        "tau_rz": tau_xz_L2,
        "w": Results_W1[0]["w"] + Results_W2[0]["w"],
        "u": 0.0,
        "epsilon_z": ez2,
        "epsilon_t": ey2,
        "epsilon_r": ex2,
    }

    # LOWER (index 2)
    Sx1b, Sy1b = rot_xy(Results_W1[1]["sigma_r"], Results_W1[1]["sigma_t"], alpha)
    Sx2b, Sy2b = rot_xy(Results_W2[1]["sigma_r"], Results_W2[1]["sigma_t"], alpha)

    Sigma_x_L3 = Sx1b + Sx2b
    Sigma_y_L3 = Sy1b + Sy2b
    Sigma_z_L3 = Results_W1[1]["sigma_z"] + Results_W2[1]["sigma_z"]
    tau_xz_L3 = Results_W1[1]["tau_rz"] * cosd(alpha) + Results_W2[1]["tau_rz"] * cosd(alpha)

    E3 = float(E[LayerD - 1])
    v3 = float(v[LayerD - 1])
    ex3 = (Sigma_x_L3 - v3 * (Sigma_y_L3 + Sigma_z_L3)) / E3
    ey3 = (Sigma_y_L3 - v3 * (Sigma_x_L3 + Sigma_z_L3)) / E3
    ez3 = (Sigma_z_L3 - v3 * (Sigma_x_L3 + Sigma_y_L3)) / E3

    out2 = {
        "sigma_z": Sigma_z_L3,
        "sigma_t": Sigma_y_L3,
        "sigma_r": Sigma_x_L3,
        "tau_rz": tau_xz_L3,
        "w": Results_W1[1]["w"] + Results_W2[1]["w"],
        "u": 0.0,
        "epsilon_z": ez3,
        "epsilon_t": ey3,
        "epsilon_r": ex3,
    }

    return [out1, out2]


# ---------------- Example usage (matches your MATLAB script) ----------------
if __name__ == "__main__":
    Number_of_layers = 5
    Thickness_layers = [100.0, 200.0, 200.0, 250.0]
    Modulus_layers = [3000.0, 400.0, 5000.0, 800.0, 100.0]
    Poissons = [0.35, 0.35, 0.25, 0.35, 0.35]

    Eva_depth_bituminous = 100.0
    Eva_depth_base = 500.0
    Eva_depth_Subgrade = 750.0

    FS_CTB_T = 1.4
    CFD_Check = 1

    SA_M_T = np.array(
        [
            [185, 195, 70000],
            [175, 185, 90000],
            [165, 175, 92000],
            [155, 165, 300000],
            [145, 155, 280000],
            [135, 145, 650000],
            [125, 135, 600000],
            [115, 125, 1340000],
        ],
        dtype=float,
    )
    TaA_M_T = np.array(
        [
            [390, 410, 200000],
            [370, 390, 230000],
            [350, 370, 240000],
            [330, 350, 235000],
            [310, 330, 225000],
            [290, 310, 475000],
            [270, 290, 450000],
            [250, 270, 1435000],
        ],
        dtype=float,
    )
    TrA_M_T = np.array(
        [
            [585, 615, 35000],
            [555, 585, 40000],
            [525, 555, 40000],
            [495, 525, 45000],
            [465, 495, 43000],
            [435, 465, 110000],
            [405, 435, 100000],
            [375, 405, 330000],
            [345, 375, 300000],
        ],
        dtype=float,
    )

    out_vec = exact_criticals_with_details(
        Number_of_layers,
        Thickness_layers,
        Modulus_layers,
        Poissons,
        Eva_depth_bituminous,
        Eva_depth_base,
        Eva_depth_Subgrade,
        CFD_Check,
        FS_CTB_T,
        SA_M_T,
        TaA_M_T,
        TrA_M_T,
    )

    # Equivalent of MATLAB "out_vec"
    print(out_vec)
