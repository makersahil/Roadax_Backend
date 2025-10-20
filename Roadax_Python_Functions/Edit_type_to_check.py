# Expects `Shared` produced by run_unified_pavement_design(...)

import math
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
# pandas is optional; core logic does not require it
try:
    import pandas as pd  # type: ignore
except Exception:  # ModuleNotFoundError or others
    pd = None  # sentinel; not used in core
from scipy.special import jv as besselj  # Bessel J_v(x)


def cosd(x: float) -> float:
    return math.cos(math.radians(x))


def sind(x: float) -> float:
    return math.sin(math.radians(x))


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



def hydrate_from_shared(Shared: Dict[str, Any]) -> Tuple[np.ndarray, float, Dict[str, float], Dict[str, Any]]:
    """
    HYDRATE_FROM_SHARED (Python)
    Inputs
      Shared : dict produced by run_unified_pavement_design (contains design, layer properties, costs, flags, spectra, etc.)
    Outputs
      Report_t        : (4 x 3) ndarray -> [Permissible, Critical, Pass/Fail] for:
                         [1] BT fatigue, [2] CTB fatigue (AUSTROADS), [3] CTB CFD (AASHTO),
                         [4] Subgrade rutting
      Cost_km         : total cost per km (legacy scale: ₹ / 1e5 = ₹ lakh)
      cost_breakdown  : dict with per-line raw subtotals and total_scaled
      Derived         : dict of useful intermediates (enums, names, vectors, indices, criteria, flags, etc.)
    """
    sget = _sget

    # ---- Design parameters ----
    Design_Traffic         = sget(Shared, 'Design_Traffic',         300.0)
    Effective_Subgrade_CBR = sget(Shared, 'Effective_Subgrade_CBR',   5.0)
    Reliability            = sget(Shared, 'Reliability',             80.0)
    Va                     = sget(Shared, 'Va',                       3.5)
    Vbe                    = sget(Shared, 'Vbe',                     11.5)

    # ---- Final thicknesses from Code-1's Best ----
    BT_thk       = float(sget(Shared, 'BT_thk',       150.0))
    In_thk       = float(sget(Shared, 'CRL_thk',        0.0))    # CRL acts as Interlayer
    Base_thk     = float(sget(Shared, 'Base_thk',       0.0))
    Subbase_thk  = float(sget(Shared, 'Subbase_thk',    0.0))
    Subgrade_thk = float(sget(Shared, 'Subgrade_thk',   float('inf')))  # semi-infinite by default

    # ---- Moduli (already computed in Code 1) ----
    BT_mod       = float(sget(Shared, 'BT_Mod',       3000.0))
    In_mod       = float(sget(Shared, 'In_mod',          0.0))
    Base_mod     = float(sget(Shared, 'Base_mod',        0.0))
    Subbase_mod  = float(sget(Shared, 'Subbase_mod',     0.0))
    Subgrade_mod = float(sget(Shared, 'Subgrade_mod',   50.0))

    # ---- Poisson's ratios (CTB/ETB/CTSB = 0.25; others 0.35) ----
    BT_v        = float(sget(Shared, 'BT_v',        0.35))
    In_v        = float(sget(Shared, 'In_v',        0.35))
    Base_v      = float(sget(Shared, 'Base_v',      0.35))
    Subbase_v   = float(sget(Shared, 'Subbase_v',   0.35))
    Subgrade_v  = float(sget(Shared, 'Subgrade_v',  0.35))

    # ---- CFD / CTB parameters & spectra (if provided) ----
    RF_CTB = sget(Shared, 'RF_CTB', None)
    FS_CTB = sget(Shared, 'FS_CTB', None)
    SA_M   = np.array(sget(Shared, 'SA_M',   []), dtype=float).reshape(-1, 3) if sget(Shared, 'SA_M',   []) is not None else np.empty((0,3))
    TaA_M  = np.array(sget(Shared, 'TaA_M',  []), dtype=float).reshape(-1, 3) if sget(Shared, 'TaA_M',  []) is not None else np.empty((0,3))
    TrA_M  = np.array(sget(Shared, 'TrA_M',  []), dtype=float).reshape(-1, 3) if sget(Shared, 'TrA_M',  []) is not None else np.empty((0,3))
    # cfdchk present but not needed here; we compute CFD if CTB+spectra+FS_CTB
    # cfdchk = int(sget(Shared, 'cfdchk', 0))

    # ---- Reinforcement flags ----
    is_wmm_r = bool(sget(Shared, 'is_wmm_r', 0))
    is_gsb_r = bool(sget(Shared, 'is_gsb_r', 0))

    # ---- Costs & widths (from Shared.costs) ----
    C = sget(Shared, 'costs', {}) if isinstance(sget(Shared, 'costs', {}), dict) else {}
    BC_cost       = _csget(C, 'BC_cost',       0.0)
    DBM_cost      = _csget(C, 'DBM_cost',      0.0)
    BC_DBM_width  = _csget(C, 'BC_DBM_width',  0.0)
    Base_cost     = _csget(C, 'Base_cost',     0.0)
    Base_width    = _csget(C, 'Base_width',    0.0)
    Subbase_cost  = _csget(C, 'Subbase_cost',  0.0)
    Subbase_width = _csget(C, 'Subbase_width', 0.0)
    CRL_cost      = _csget(C, 'CRL_cost',      0.0)
    CRL_width     = _csget(C, 'CRL_width',     0.0)
    SAMI_cost     = _csget(C, 'SAMI_cost',     0.0)
    wmm_r_cost    = _csget(C, 'wmm_r_cost',    0.0)
    gsb_r_cost    = _csget(C, 'gsb_r_cost',    0.0)

    BT_width = BC_DBM_width  # for convenience

    # ---- Map Code-1 Type → enums ----
    Type = int(sget(Shared, 'Type', 1))

    # Bituminous_layer: 1=BC | 2=BC+DBM | 3=Modified | 4=Other
    # Interlayer:       1=AIL | 2=SAMI | 3=Other | 4=None
    # Base_layer:       1=WMM | 2=CTB | 3=ETB | 4=Reinf WMM | 5=Other | 6=None
    # Subbase_layer:    1=GSB | 2=CTSB| 3=Reinf GSB | 4=Other | 5=None
    Bituminous_layer = 2
    Interlayer       = 4
    Base_layer       = 6
    Subbase_layer    = 5

    if   Type == 1: Interlayer=4; Base_layer=1; Subbase_layer=1   # WMM + GSB
    elif Type == 2: Interlayer=1; Base_layer=2; Subbase_layer=2   # CTB + AIL + CTSB
    elif Type == 3: Interlayer=2; Base_layer=2; Subbase_layer=2   # CTB + SAMI + CTSB
    elif Type == 4: Interlayer=4; Base_layer=3; Subbase_layer=2   # ETB + CTSB
    elif Type == 5: Interlayer=1; Base_layer=2; Subbase_layer=1   # CTB + AIL + GSB
    elif Type == 6: Interlayer=4; Base_layer=1; Subbase_layer=2   # WMM + CTSB
    elif Type == 7: Interlayer=4; Base_layer=4; Subbase_layer=3   # Reinforced WMM + Reinforced GSB
    elif Type == 8: Interlayer=4; Base_layer=4; Subbase_layer=2   # Reinforced WMM + CTSB

    # ---- Enforce SAMI as zero-thickness/zero-modulus interlayer ----
    if Interlayer == 2:  # SAMI
        In_thk = 0.0
        In_mod = 0.0

    # ---- ν overrides by enum (safety: CTB/ETB/CTSB → 0.25) ----
    if Base_layer in (2, 3):      Base_v    = 0.25
    if Subbase_layer == 2:        Subbase_v = 0.25

    # ---- Pick interlayer cost/width consistent with enum ----
    if   Interlayer == 1:  # AIL / CRL
        In_cost, In_width = CRL_cost, CRL_width
    elif Interlayer == 2:  # SAMI (flat cost elsewhere)
        In_cost, In_width = SAMI_cost, max(BT_width, CRL_width)
    else:
        In_cost, In_width = 0.0, 0.0

    # ---- Names (for reporting) ----
    Bituminous_opts = ['BC','BC+DBM','Modified Mix','Other']
    Interlayer_opts = ['AIL','SAMI','Other','None']
    Base_opts       = ['WMM','CTB','ETB','Reinforced WMM','Other','None']
    Subbase_opts    = ['GSB','CTSB','Reinforced GSB','Other','None']

    Bituminous_name = Bituminous_opts[Bituminous_layer-1]
    Interlayer_name = Interlayer_opts[Interlayer-1]
    Base_name       = Base_opts[Base_layer-1]
    Subbase_name    = Subbase_opts[Subbase_layer-1]
    Subgrade_name   = 'Subgrade'

    # ---- Flags for checks ----
    BT_F     = 1
    CTB_AUST = (RF_CTB is not None) and (Base_layer == 2)
    CTB_AASH = (FS_CTB is not None) and (Base_layer == 2) and (SA_M.size > 0)
    SB_R     = 1

    # ------------------------- DESIGN CHECK BASICS ------------------------
    N = float(Design_Traffic)
    N_design = N * 1e6
    reliability = float(Reliability)

    # ------------------------- FINAL LAYER VECTORS ------------------------
    # Ensure subgrade defaults if needed
    if not math.isfinite(Subgrade_thk):
        Subgrade_thk = float('inf')

    if (Subgrade_mod is None) or (Subgrade_mod <= 0):
        CBR = sget(Shared, 'Effective_Subgrade_CBR', None)
        if CBR is not None:
            CBR = float(CBR)
            Subgrade_mod = 10.0 * CBR if CBR <= 5 else 17.6 * (CBR ** 0.64)
        else:
            Subgrade_mod = 50.0
    if Subgrade_v is None:
        Subgrade_v = 0.35

    # Order: [Bituminous, Interlayer, Base, Subbase, Subgrade]
    Thk_T       = np.array([BT_thk,   In_thk,   Base_thk,   Subbase_thk,   Subgrade_thk], dtype=float)
    Mod_T       = np.array([BT_mod,   In_mod,   Base_mod,   Subbase_mod,   Subgrade_mod], dtype=float)
    v_T         = np.array([BT_v,     In_v,     Base_v,     Subbase_v,     Subgrade_v  ], dtype=float)
    Layer_Names = [Bituminous_name, Interlayer_name, Base_name, Subbase_name, Subgrade_name]

    # Keep finite, positive layers AND always keep Subgrade (last element)
    idxSG = len(Thk_T) - 1
    mask = (Thk_T > 0) & np.isfinite(Thk_T)
    mask[idxSG] = True

    Layer_Names = [nm for nm, keep in zip(Layer_Names, mask) if keep]
    Thk_T       = Thk_T[mask]
    Mod_T       = Mod_T[mask]
    v_T         = v_T[mask]

    # Finite-layer thickness vector (exclude subgrade)
    Thk_TC = Thk_T[:-1]

    # Locate Base (3rd in original ordering) in masked arrays (1-based for AIO_SigmaSi)
    base_index_original = 3  # 1-based in the original 5-layer ordering
    valid_idx = [i for i, keep in enumerate((np.array([True, True, True, True, True]) & np.array([True, True, True, True, True])), start=1) if True]
    # original indices are 1..5; we need to see if the Base layer (index=3) survived masking:
    # build original mask to know which ones survived
    orig_Thk_T = np.array([BT_thk, In_thk, Base_thk, Subbase_thk, Subgrade_thk], dtype=float)
    orig_mask  = (orig_Thk_T > 0) & np.isfinite(orig_Thk_T)
    orig_mask[-1] = True
    valid_idx = [i for i, keep in enumerate(orig_mask, start=1) if keep]
    try:
        final_base_pos = valid_idx.index(base_index_original) + 1  # back to 1-based
    except ValueError:
        final_base_pos = None

    # ------------------------- CRITERIA CALCULATIONS ------------------------
    # BT fatigue
    if BT_F == 1:
        n_b = len(Thk_T)
        Thickness_b = Thk_TC.tolist()
        E_b = Mod_T.tolist()
        v_b = v_T.tolist()
        isbonded_b = True
        Eva_depth_b = float(Thk_T[0])
        load0_b = 20000.0
        typre_b = 0.56
        alpha_b = 0.0
        top_b = 1
        bot_b = 2
        MatBT = AIO_SigmaSi(n_b, Thickness_b, E_b, v_b, isbonded_b, typre_b, load0_b, alpha_b, Eva_depth_b, top_b, bot_b)
        # columns: 0 Eva_depth, 1 r, 2 sigma_z, 3 sigma_t, 4 sigma_r, 5 tau_rz, 6 w, 7 ez, 8 et, 9 er
        Critical_BSi = float(np.nanmax(np.abs(MatBT[:, 8:10])))
    else:
        MatBT = np.empty((0, 10))
        Critical_BSi = float('nan')

    # Subgrade rutting
    if SB_R == 1:
        n_s = len(Thk_T)
        Thickness_s = Thk_TC.tolist()
        E_s = Mod_T.tolist()
        v_s = v_T.tolist()
        isbonded_s = True
        Eva_depth_s = float(np.sum(Thk_TC))
        load0_s = 20000.0
        typre_s = 0.56
        alpha_s = 0.0
        top_s = n_s - 1
        bot_s = n_s
        MatSR = AIO_SigmaSi(n_s, Thickness_s, E_s, v_s, isbonded_s, typre_s, load0_s, alpha_s, Eva_depth_s, top_s, bot_s)
        Critical_SubSi = float(np.nanmax(np.abs(MatSR[:, 7])))
    else:
        MatSR = np.empty((0, 10))
        Critical_SubSi = float('nan')

    # CTB fatigue (AUSTROADS) — principal pair at CTB interface (modeled by max of eps_t/eps_r magnitudes)
    if CTB_AUST and (final_base_pos is not None):
        n_c = len(Thk_T)
        Thickness_c = Thk_TC.tolist()
        E_c = Mod_T.tolist()
        v_c = v_T.tolist()
        isbonded_c = True
        Eva_depth_c = float(np.sum(Thk_TC[:final_base_pos]))
        load0_c = 20000.0
        typre_c = 0.8
        alpha_c = 0.0
        top_c = int(final_base_pos)
        bot_c = int(final_base_pos + 1)
        MatC = AIO_SigmaSi(n_c, Thickness_c, E_c, v_c, isbonded_c, typre_c, load0_c, alpha_c, Eva_depth_c, top_c, bot_c)
        Critical_CTBSi = float(np.nanmax(np.abs(MatC[:, 8:10])))
    else:
        MatC = np.empty((0, 10))
        Critical_CTBSi = float('nan')

    # CTB cracking (AASHTO CFD)
    if CTB_AASH and (final_base_pos is not None):
        n_ca = len(Thk_T)
        Thickness_ca = Thk_TC.tolist()
        E_ca = Mod_T.tolist()
        v_ca = v_T.tolist()
        isbonded_ca = True
        Eva_depth_ca = float(np.sum(Thk_TC[:final_base_pos]))
        typre_ca = 0.8
        alpha_ca = 0.0
        top_ca = int(final_base_pos)
        bot_ca = int(final_base_pos + 1)

        CFD_SA  = 0.0
        CFD_TaA = 0.0
        CFD_TrA = 0.0

        def _cfd_from(ax_arr: np.ndarray, wheels: int, rep_mult: float) -> float:
            if ax_arr.size == 0:
                return 0.0
            cfd = 0.0
            for i in range(ax_arr.shape[0]):
                load_kN = float(np.mean(ax_arr[i, 0:2]))
                load = (load_kN * 1000.0) / wheels
                reps = rep_mult * float(ax_arr[i, 2])

                Mat = AIO_SigmaSi(n_ca, Thickness_ca, E_ca, v_ca, isbonded_ca, typre_ca, load, alpha_ca, Eva_depth_ca, top_ca, bot_ca)
                # take max of |sigma_t|, |sigma_r|
                sig = float(np.nanmax(np.abs(Mat[:, 3:5])))
                SR  = sig / float(FS_CTB)
                NF  = 10.0 ** ((0.972 - SR) / 0.0825)
                cfd += (reps / NF)
                if cfd > 1.0:
                    break
            return cfd

        CFD_SA  = _cfd_from(SA_M,  4, 1.0)
        CFD_TaA = _cfd_from(TaA_M, 8, 2.0)
        CFD_TrA = _cfd_from(TrA_M, 12, 3.0)
        CFD = float(CFD_SA + CFD_TaA + CFD_TrA)
    else:
        CFD = float('nan')

    # Assemble criticals (MPa→microstrain for strain terms)
    Critical_Si = np.array([
        Critical_BSi * 1e6,
        Critical_CTBSi * 1e6,
        CFD,
        Critical_SubSi * 1e6
    ], dtype=float)

    # Permissible strains (returned as strains; convert to microstrain for reporting)
    Base_Mod_for_perm = None if (final_base_pos is None) else float(Mod_T[final_base_pos-1])
    Perm_raw = AIO_PermissibleSi(reliability, N_design, float(Mod_T[0]), float(Va), float(Vbe), Base_Mod_for_perm, RF_CTB)
    Permissible_Si = np.array(Perm_raw, dtype=float) * 1e6  # [bfat, rut, (opt) cfat]*1e6

    # Row-by-row permissibles to match 4 checks
    def pick_perm(idx, fallback=np.nan):
        try:
            return float(Permissible_Si[idx])
        except Exception:
            return float('nan') if math.isnan(fallback) else float(fallback)

    RPermissible_Si = np.array([
        pick_perm(0),                   # BT fatigue
        (pick_perm(2) if CTB_AUST else np.nan),   # CTB fatigue (AUSTROADS)
        (1.0 if CTB_AASH else np.nan),            # CFD ≤ 1
        pick_perm(1)                    # Subgrade rutting
    ], dtype=float)

    # Pass/fail (NaN comparisons -> False, as in MATLAB)
    D_Check = (Critical_Si <= RPermissible_Si)
    D_Check = np.where(np.isnan(Critical_Si) | np.isnan(RPermissible_Si), False, D_Check)

    Report_t = np.column_stack([RPermissible_Si, Critical_Si, D_Check.astype(float)])

    # ------------ Compute cost (per km, legacy scale) ------------
    costs = dict(
        BC_cost=BC_cost, DBM_cost=DBM_cost, BC_DBM_width=BT_width,
        CRL_cost=In_cost, CRL_width=In_width,
        Base_cost=Base_cost, Base_width=Base_width,
        Subbase_cost=Subbase_cost, Subbase_width=Subbase_width,
        SAMI_cost=SAMI_cost, wmm_r_cost=wmm_r_cost, gsb_r_cost=gsb_r_cost
    )
    flags = dict(is_wmm_r=is_wmm_r, is_gsb_r=is_gsb_r)
    Cost_km, cost_breakdown = cost_unified_v2(Type, BT_thk, In_thk, Base_thk, Subbase_thk, costs, flags)

    # ------------ Pack Derived for downstream use ------------
    Derived: Dict[str, Any] = {}
    Derived["Type"]            = Type
    Derived["enums"]           = dict(Bituminous_layer=Bituminous_layer, Interlayer=Interlayer,
                                      Base_layer=Base_layer, Subbase_layer=Subbase_layer)
    Derived["names"]           = dict(Bituminous=Bituminous_name, Interlayer=Interlayer_name,
                                      Base=Base_name, Subbase=Subbase_name, Subgrade=Subgrade_name)
    Derived["flags"]           = dict(BT_F=BT_F, CTB_AUST=bool(CTB_AUST), CTB_AASH=bool(CTB_AASH), SB_R=SB_R)
    Derived["N_design"]        = N_design
    Derived["Va_Vbe"]          = [Va, Vbe]
    Derived["Thk_T"]           = Thk_T
    Derived["Mod_T"]           = Mod_T
    Derived["v_T"]             = v_T
    Derived["Layer_Names"]     = Layer_Names
    Derived["Thk_TC"]          = Thk_TC
    Derived["final_base_pos"]  = final_base_pos
    Derived["Mats"]            = dict(MatBT=MatBT, MatSR=MatSR, MatC=MatC)
    Derived["Perms"]           = Permissible_Si
    Derived["Criticals"]       = Critical_Si
    Derived["RowPerms"]        = RPermissible_Si
    Derived["PassFail"]        = D_Check
    Derived["costs"]           = costs
    Derived["cost_flags"]      = flags

    return Report_t, float(Cost_km), cost_breakdown, Derived


# --------------------- helpers & cost (v2) ---------------------

def _is_empty(x: Any) -> bool:
    if x is None:
        return True
    if isinstance(x, float) and math.isnan(x):
        return True
    if isinstance(x, np.ndarray):
        return x.size == 0
    try:
        return (len(x) == 0)  # type: ignore[arg-type]
    except Exception:
        return False

def _sget(S: Dict[str, Any], key: str, default: Any) -> Any:
    """Safe getter: if missing/empty/NaN → default; else value as-is."""
    if isinstance(S, dict) and key in S:
        v = S[key]
        if not _is_empty(v):
            return v
    return default

def _csget(costs: Dict[str, Any], key: str, default: float) -> float:
    v = costs.get(key, default) if isinstance(costs, dict) else default
    if _is_empty(v):
        return float(default)
    return float(v)

def cost_unified_v2(Type: int, BT: float, CRL: float, Base: float, Subb: float,
                    costs: Dict[str, float], flags: Dict[str, bool]) -> Tuple[float, Dict[str, float]]:
    """
    Matches the MATLAB cost_unified_v2 behavior.
    Returns:
      Cost (₹/km divided by 1e5 → ₹ lakh/km), and a breakdown dict (raw subtotals + total_scaled).
    """
    g  = lambda s, k, d: float(s.get(k, d)) if isinstance(s, dict) and (s.get(k, d) is not None) else float(d)
    gb = lambda s, k, d: bool(s.get(k, d)) if isinstance(s, dict) else bool(d)

    BC_cost       = g(costs, 'BC_cost', 0.0)
    DBM_cost      = g(costs, 'DBM_cost', 0.0)
    BC_DBM_width  = g(costs, 'BC_DBM_width', 0.0)
    CRL_cost      = g(costs, 'CRL_cost', 0.0)
    CRL_width     = g(costs, 'CRL_width', 0.0)
    Base_cost     = g(costs, 'Base_cost', 0.0)
    Base_width    = g(costs, 'Base_width', 0.0)
    Subbase_cost  = g(costs, 'Subbase_cost', 0.0)
    Subbase_width = g(costs, 'Subbase_width', 0.0)
    SAMI_cost     = g(costs, 'SAMI_cost', 0.0)
    wmm_r_cost    = g(costs, 'wmm_r_cost', 0.0)
    gsb_r_cost    = g(costs, 'gsb_r_cost', 0.0)

    is_wmm_r      = gb(flags, 'is_wmm_r', False)
    is_gsb_r      = gb(flags, 'is_gsb_r', False)

    # Split BT into BC & DBM
    if BT <= 40:
        BC_mm, DBM_mm = BT, 0.0
    elif BT < 90:
        BC_mm, DBM_mm = 30.0, BT - 30.0
    else:
        BC_mm, DBM_mm = 40.0, BT - 40.0

    useCRL = (Type in (2, 5)) and (CRL > 0.0)
    CRL_mm = (CRL if useCRL else 0.0)

    useSAMI = (Type == 3)
    SAMI_Flat = (BC_DBM_width * 1000.0) * SAMI_cost if (useSAMI and (SAMI_cost > 0.0)) else 0.0

    WMM_R_Flat = 0.0
    GSB_R_Flat = 0.0
    if Type in (7, 8):
        if is_wmm_r and (wmm_r_cost > 0.0):
            WMM_R_Flat = (Base_width    * 1000.0) * wmm_r_cost
        if is_gsb_r and (gsb_r_cost > 0.0):
            GSB_R_Flat = (Subbase_width * 1000.0) * gsb_r_cost

    c_BC   = BC_mm  * 0.001 * BC_DBM_width  * 1000.0 * BC_cost
    c_DBM  = DBM_mm * 0.001 * BC_DBM_width  * 1000.0 * DBM_cost
    c_CRL  = CRL_mm * 0.001 * CRL_width     * 1000.0 * CRL_cost
    c_Base = Base   * 0.001 * Base_width    * 1000.0 * Base_cost
    c_Subb = Subb   * 0.001 * Subbase_width * 1000.0 * Subbase_cost

    subtotal_raw = c_BC + c_DBM + c_CRL + c_Base + c_Subb + SAMI_Flat + WMM_R_Flat + GSB_R_Flat
    Cost = subtotal_raw / 1e5  # legacy scale (₹ lakh / km)

    breakdown = dict(
        BC=c_BC,
        DBM=c_DBM,
        CRL=c_CRL,
        Base=c_Base,
        Subbase=c_Subb,
        SAMI_Flat=SAMI_Flat,
        WMM_R_Flat=WMM_R_Flat,
        GSB_R_Flat=GSB_R_Flat,
        subtotal_raw=subtotal_raw,
        total_scaled=Cost
    )
    return float(Cost), breakdown


#Shared = dict(
#    Type=2,  # CTB + AIL + CTSB
#    Design_Traffic=300, Effective_Subgrade_CBR=5, Reliability=80, Va=3.5, Vbe=11.5,
#    BT_thk=150, CRL_thk=0, Base_thk=200, Subbase_thk=200, Subgrade_thk=float('inf'),
#    BT_Mod=3000, In_mod=0, Base_mod=1500, Subbase_mod=600, Subgrade_mod=50,
#    RF_CTB=1.0, FS_CTB=1.0, SA_M=[[80, 80, 1e6]], TaA_M=[], TrA_M=[],
#    costs=dict(BC_cost=6000, DBM_cost=5000, BC_DBM_width=7.0,
#               Base_cost=1800, Base_width=7.0, Subbase_cost=1200, Subbase_width=7.0,
#               CRL_cost=2000, CRL_width=7.0, SAMI_cost=250, wmm_r_cost=0, gsb_r_cost=0)
#)

#Report_t, Cost_km, breakdown, D = hydrate_from_shared(Shared)

#print(Report_t)