# effective_cbr_layered.py
import math
from typing import List, Dict, Any

import numpy as np
from scipy.special import jv as besselj  # Bessel J_n


def AIO_Effective_CBR(number_of_layer: int,
                      thk: List[float],
                      CBR: List[float],
                      Poisson_r: List[float]) -> float:
    """
    Direct translation of AIO_Effective_CBR (MATLAB) with identical logic.
    """
    # ---------------------- basic validation ----------------------
    n = number_of_layer
    assert n >= 2, 'number_of_layer must be >= 2.'
    assert np.ndim(thk) == 1 and len(thk) == n - 1, 'thk must be 1 x (n-1).'
    assert np.ndim(CBR) == 1 and len(CBR) == n, 'CBR must be 1 x n.'
    assert np.ndim(Poisson_r) == 1 and len(Poisson_r) == n, 'Poisson_r must be 1 x n.'

    # Ensure row vectors (conceptually)
    Thickness = list(thk)
    CBR_vec   = list(CBR)
    v         = list(Poisson_r)

    # ------------------- CBR → modulus (vectorized) -------------------
    Mod = [0.0] * n
    for i in range(n):
        if CBR_vec[i] <= 5:
            Mod[i] = 10.0 * CBR_vec[i]
        else:
            Mod[i] = 17.6 * (CBR_vec[i] ** 0.64)  # MPa

    # ---------------------- effective modulus ----------------------
    E_mod = AIO_EffectiveMr(n, Thickness, Mod, v)  # scalar MPa

    # ------------------- modulus → effective CBR -------------------
    if E_mod <= 50.0:
        Effective_CBR = E_mod / 10.0
    else:
        Effective_CBR = (E_mod / 17.6) ** (1.0 / 0.64)

    return Effective_CBR


def AIO_EffectiveMr(n: int,
                    Thickness: List[float],
                    E: List[float],
                    v: List[float]) -> float:
    load = 40000.0
    tyre = 0.56
    q = tyre
    a = math.sqrt(load / (math.pi * q))
    isbonded = True

    PointsW1O = [{'r': 0.0, 'z': 0.0, 'layer': 1}]  # 1-based layer index
    Results_W1_O = AIO_R(n, Thickness, E, v, isbonded, PointsW1O, a, q)
    surf_def = Results_W1_O[0]['w']
    EMr = 2.0 * (1.0 - v[0] ** 2) * q * a / surf_def
    return EMr


def AIO_R(n: int,
          Thickness: List[float],
          E: List[float],
          v: List[float],
          isbonded: bool,
          Points: List[Dict[str, float]],
          a: float,
          q: float) -> List[Dict[str, float]]:
    # Interfaces and total thickness
    z_interfaces = np.cumsum(np.array(Thickness, dtype=float))  # length n-1
    H = float(z_interfaces[-1])

    alpha = a / H
    dl = 0.2 / alpha

    # Initialize accumulators per point
    Results = []
    for _ in range(len(Points)):
        Results.append({
            'sigma_z': 0.0, 'sigma_r': 0.0, 'sigma_t': 0.0,
            'tau_rz': 0.0, 'w': 0.0, 'u': 0.0
        })

    # Integrate l = 0:dl:200, with 4-pt Gaussian in each sub-interval
    l = 0.0
    while l <= 200.0 + 1e-12:
        mids = (l + dl / 2.0) + np.array([-0.86114, -0.33998, 0.33998, 0.86114]) * (dl / 2.0)
        fcs = np.array([0.34786, 0.65215, 0.65215, 0.34786]) * (dl / 2.0)

        Hat_sets = [AIO_R_hat(n, Thickness, E, v, isbonded, float(m), Points) for m in mids]

        for gp in range(4):
            m = float(mids[gp])
            fc = float(fcs[gp])

            if m != 0.0:
                J1 = float(besselj(1, m * alpha))
                factor = q * alpha * J1 / m

                for j in range(len(Points)):
                    Rj = Hat_sets[gp][j]
                    Results[j]['sigma_z'] += fc * (factor * Rj['sigma_z'])
                    Results[j]['sigma_r'] += fc * (factor * Rj['sigma_r'])
                    Results[j]['sigma_t'] += fc * (factor * Rj['sigma_t'])
                    Results[j]['tau_rz']  += fc * (factor * Rj['tau_rz'])
                    Results[j]['w']       += fc * (factor * Rj['w'])
                    Results[j]['u']       += fc * (factor * Rj['u'])

        l += dl

    # Strains (Hooke’s law) at each point (1-based layer index)
    for j in range(len(Points)):
        ii = int(Points[j]['layer'])  # 1..n
        Ei = float(E[ii - 1])
        vi = float(v[ii - 1])

        sigz = Results[j]['sigma_z']
        sigr = Results[j]['sigma_r']
        sigt = Results[j]['sigma_t']

        Results[j]['epsilon_z'] = (sigz - vi * (sigr + sigt)) / Ei
        Results[j]['epsilon_r'] = (sigr - vi * (sigz + sigt)) / Ei
        Results[j]['epsilon_t'] = (sigt - vi * (sigr + sigz)) / Ei

    return Results


def AIO_R_hat(n: int,
              Thickness: List[float],
              E: List[float],
              v: List[float],
              isbonded: bool,
              m: float,
              Points: List[Dict[str, float]]) -> List[Dict[str, float]]:
    Thickness = np.array(Thickness, dtype=float)
    E = np.array(E, dtype=float)
    v = np.array(v, dtype=float)

    # Interfaces and normalized depths
    z = np.cumsum(Thickness)       # length n-1
    H = float(z[-1])
    lam = z / H                    # λ(1..n-1)
    lam = np.concatenate([lam, [np.inf]])  # λ(n)=inf

    # Exponentials & ratios
    F = np.zeros(n, dtype=float)
    Rr = np.zeros(n - 1, dtype=float)

    F[0] = math.exp(-m * (lam[0] - 0.0))
    Rr[0] = (E[0] / E[1]) * (1.0 + v[1]) / (1.0 + v[0])
    for i in range(1, n - 1):
        F[i] = math.exp(-m * (lam[i] - lam[i - 1]))
        Rr[i] = (E[i] / E[i + 1]) * (1.0 + v[i + 1]) / (1.0 + v[i])
    F[n - 1] = math.exp(-m * (lam[n - 1] - lam[n - 2]))

    # Build/solve M1 \ M2 per interface
    M_list = []
    for i in range(n - 1):
        Fi = F[i]
        Fip1 = F[i + 1]
        lam_i = lam[i]
        vi = v[i]
        vip1 = v[i + 1]
        Ri = Rr[i]

        if isbonded:
            M1 = np.zeros((4, 4))
            M2 = np.zeros((4, 4))

            M1[0, 0] = 1; M1[1, 0] = 1; M1[2, 0] = 1; M1[3, 0] = 1
            M1[0, 1] = Fi; M1[1, 1] = -Fi; M1[2, 1] = Fi; M1[3, 1] = -Fi
            M1[0, 2] = -(1 - 2 * vi - m * lam_i)
            M1[1, 2] = 2 * vi + m * lam_i
            M1[2, 2] = 1 + m * lam_i
            M1[3, 2] = -(2 - 4 * vi - m * lam_i)
            M1[0, 3] = (1 - 2 * vi + m * lam_i) * Fi
            M1[1, 3] = (2 * vi - m * lam_i) * Fi
            M1[2, 3] = -(1 - m * lam_i) * Fi
            M1[3, 3] = -(2 - 4 * vi + m * lam_i) * Fi

            M2[0, 0] = Fip1; M2[1, 0] = Fip1; M2[2, 0] = Ri * Fip1; M2[3, 0] = Ri * Fip1
            M2[0, 1] = 1;    M2[1, 1] = -1;   M2[2, 1] = Ri;       M2[3, 1] = -Ri
            M2[0, 2] = -(1 - 2 * vip1 - m * lam_i) * Fip1
            M2[1, 2] = (2 * vip1 + m * lam_i) * Fip1
            M2[2, 2] = (1 + m * lam_i) * Ri * Fip1
            M2[3, 2] = -(2 - 4 * vip1 - m * lam_i) * Ri * Fip1
            M2[0, 3] = 1 - 2 * vip1 + m * lam_i
            M2[1, 3] = 2 * vip1 - m * lam_i
            M2[2, 3] = -(1 - m * lam_i) * Ri
            M2[3, 3] = -(2 - 4 * vip1 + m * lam_i) * Ri
        else:
            M1 = np.zeros((4, 4))
            M2 = np.zeros((4, 4))

            M1[0, 0] = 1; M1[1, 0] = 1; M1[2, 0] = 1; M1[3, 0] = 0
            M1[0, 1] = Fi; M1[1, 1] = Fi; M1[2, 1] = -Fi; M1[3, 1] = 0
            M1[0, 2] = -(1 - 2 * vi - m * lam_i)
            M1[1, 2] = 1 + m * lam_i
            M1[2, 2] = 2 * vi + m * lam_i
            M1[3, 2] = 0
            M1[0, 3] = (1 - 2 * vi + m * lam_i) * Fi
            M1[1, 3] = -(1 - m * lam_i) * Fi
            M1[2, 3] = (2 * vi - m * lam_i) * Fi
            M1[3, 3] = 0

            M2[0, 0] = Fip1; M2[1, 0] = Ri * Fip1; M2[2, 0] = 0;     M2[3, 0] = Fip1
            M2[0, 1] = 1;    M2[1, 1] = Ri;        M2[2, 1] = 0;     M2[3, 1] = -1
            M2[0, 2] = -(1 - 2 * vip1 - m * lam_i) * Fip1
            M2[1, 2] = (1 + m * lam_i) * Ri * Fip1
            M2[2, 2] = 0
            M2[3, 2] = (2 * vip1 + m * lam_i) * Fip1
            M2[0, 3] = 1 - 2 * vip1 + m * lam_i
            M2[1, 3] = -(1 - m * lam_i) * Ri
            M2[2, 3] = 0
            M2[3, 3] = 2 * vip1 - m * lam_i

        X = np.linalg.solve(M1, M2)  # exact MATLAB backslash analog for square systems
        M_list.append(X)

    MM = np.eye(4)
    for Mi in M_list:
        MM = MM @ Mi
    MM = MM[:, [1, 3]]  # columns 2 and 4 (0-based)

    b11 = math.exp(-lam[0] * m)
    b21 = math.exp(-lam[0] * m)
    b12 = 1.0
    b22 = -1.0

    c11 = -(1 - 2 * v[0]) * math.exp(-m * lam[0])
    c21 = 2 * v[0] * math.exp(-m * lam[0])
    c12 = 1 - 2 * v[0]
    c22 = 2 * v[0]

    k11 = b11 * MM[0, 0] + b12 * MM[1, 0] + c11 * MM[2, 0] + c12 * MM[3, 0]
    k12 = b11 * MM[0, 1] + b12 * MM[1, 1] + c11 * MM[2, 1] + c12 * MM[3, 1]
    k21 = b21 * MM[0, 0] + b22 * MM[1, 0] + c21 * MM[2, 0] + c22 * MM[3, 0]
    k22 = b21 * MM[0, 1] + b22 * MM[1, 1] + c21 * MM[2, 1] + c22 * MM[3, 1]

    A = np.zeros(n)
    B = np.zeros(n)
    C = np.zeros(n)
    D = np.zeros(n)

    B[n - 1] = k22 / (k11 * k22 - k12 * k21)
    D[n - 1] = 1.0 / (k12 - k22 * k11 / k21)

    for i in range(n - 2, -1, -1):
        vec = M_list[i] @ np.array([A[i + 1], B[i + 1], C[i + 1], D[i + 1]])
        A[i], B[i], C[i], D[i] = vec.tolist()

    outs: List[Dict[str, float]] = []
    for P in Points:
        rho = float(P['r']) / H
        lmm = float(P['z']) / H
        ii = int(P['layer'])  # 1..n

        def J0(x): return float(besselj(0, x))
        def J1(x): return float(besselj(1, x))

        if ii != 1:
            e1 = math.exp(-m * (lam[ii - 1] - lmm))
            e2 = math.exp(-m * (lmm - lam[ii - 2]))

            sigma_z = -m * J0(m * rho) * (
                (A[ii - 1] - C[ii - 1] * (1 - 2 * v[ii - 1] - m * lmm)) * e1 +
                (B[ii - 1] + D[ii - 1] * (1 - 2 * v[ii - 1] + m * lmm)) * e2
            )

            if rho != 0.0:
                common = ((A[ii - 1] + C[ii - 1] * (1 + m * lmm)) * e1 +
                          (B[ii - 1] - D[ii - 1] * (1 - m * lmm)) * e2)
                sigma_r = (m * J0(m * rho) - J1(m * rho) / rho) * common + \
                          2 * v[ii - 1] * m * J0(m * rho) * (C[ii - 1] * e1 - D[ii - 1] * e2)
                sigma_t = (J1(m * rho) / rho) * common + \
                          2 * v[ii - 1] * m * J0(m * rho) * (C[ii - 1] * e1 - D[ii - 1] * e2)
            else:
                common = ((A[ii - 1] + C[ii - 1] * (1 + m * lmm)) * e1 +
                          (B[ii - 1] - D[ii - 1] * (1 - m * lmm)) * e2)
                sigma_r = (m * J0(m * rho) - m / 2.0) * common + \
                          2 * v[ii - 1] * m * J0(m * rho) * (C[ii - 1] * e1 - D[ii - 1] * e2)
                sigma_t = (m / 2.0) * common + \
                          2 * v[ii - 1] * m * J0(m * rho) * (C[ii - 1] * e1 - D[ii - 1] * e2)

            tau_rz = m * J1(m * rho) * (
                (A[ii - 1] + C[ii - 1] * (2 * v[ii - 1] + m * lmm)) * e1 -
                (B[ii - 1] - D[ii - 1] * (2 * v[ii - 1] - m * lmm)) * e2
            )

            w = -H * (1 + v[ii - 1]) / E[ii - 1] * J0(m * rho) * (
                (A[ii - 1] - C[ii - 1] * (2 - 4 * v[ii - 1] - m * lmm)) * e1 -
                (B[ii - 1] + D[ii - 1] * (2 - 4 * v[ii - 1] + m * lmm)) * e2
            )

            u = H * (1 + v[ii - 1]) / E[ii - 1] * J1(m * rho) * (
                (A[ii - 1] + C[ii - 1] * (1 + m * lmm)) * e1 +
                (B[ii - 1] - D[ii - 1] * (1 - m * lmm)) * e2
            )

        else:
            e1 = math.exp(-m * (lam[ii - 1] - lmm))
            e0 = math.exp(-m * (lmm - 0.0))

            sigma_z = -m * J0(m * rho) * (
                (A[ii - 1] - C[ii - 1] * (1 - 2 * v[ii - 1] - m * lmm)) * e1 +
                (B[ii - 1] + D[ii - 1] * (1 - 2 * v[ii - 1] + m * lmm)) * e0
            )

            if rho != 0.0:
                common = ((A[ii - 1] + C[ii - 1] * (1 + m * lmm)) * e1 +
                          (B[ii - 1] - D[ii - 1] * (1 - m * lmm)) * e0)
                sigma_r = (m * J0(m * rho) - J1(m * rho) / rho) * common + \
                          2 * v[ii - 1] * m * J0(m * rho) * (C[ii - 1] * e1 - D[ii - 1] * e0)
                sigma_t = (J1(m * rho) / rho) * common + \
                          2 * v[ii - 1] * m * J0(m * rho) * (C[ii - 1] * e1 - D[ii - 1] * e0)
            else:
                common = ((A[ii - 1] + C[ii - 1] * (1 + m * lmm)) * e1 +
                          (B[ii - 1] - D[ii - 1] * (1 - m * lmm)) * e0)
                sigma_r = (m * J0(m * rho) - m / 2.0) * common + \
                          2 * v[ii - 1] * m * J0(m * rho) * (C[ii - 1] * e1 - D[ii - 1] * e0)
                sigma_t = (m / 2.0) * common + \
                          2 * v[ii - 1] * m * J0(m * rho) * (C[ii - 1] * e1 - D[ii - 1] * e0)

            tau_rz = m * J1(m * rho) * (
                (A[ii - 1] + C[ii - 1] * (2 * v[ii - 1] + m * lmm)) * e1 -
                (B[ii - 1] - D[ii - 1] * (2 * v[ii - 1] - m * lmm)) * e0
            )

            w = -H * (1 + v[ii - 1]) / E[ii - 1] * J0(m * rho) * (
                (A[ii - 1] - C[ii - 1] * (2 - 4 * v[ii - 1] - m * lmm)) * e1 -
                (B[ii - 1] + D[ii - 1] * (2 - 4 * v[ii - 1] + m * lmm)) * e0
            )

            u = H * (1 + v[ii - 1]) / E[ii - 1] * J1(m * rho) * (
                (A[ii - 1] + C[ii - 1] * (1 + m * lmm)) * e1 +
                (B[ii - 1] - D[ii - 1] * (1 - m * lmm)) * e0
            )

        outs.append({
            'sigma_z': sigma_z, 'sigma_r': sigma_r, 'sigma_t': sigma_t,
            'tau_rz': tau_rz, 'w': w, 'u': u
        })

    return outs


if __name__ == "__main__":
    # ---------------- inputs (exactly as MATLAB) ----------------
    number_of_layer = 5
    thk       = [200, 300, 100, 400]
    CBR       = [10, 5, 10, 5, 8]
    Poisson_r = [0.35, 0.35, 0.35, 0.35, 0.35]

    Effective_CBR = AIO_Effective_CBR(number_of_layer, thk, CBR, Poisson_r)
    print(f"\nThe Effective CBR of the layered system is: {Effective_CBR:.3f}")
