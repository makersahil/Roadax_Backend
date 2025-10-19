# multilayer_dualwheel.py
import math
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
# pandas is optional for JSON bridge; if unavailable, we'll return a dict-shaped table
try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore
from scipy.special import jv as besselj  # Bessel J_n


# --------------------------- small helpers ---------------------------
def cosd(x: float) -> float:
    return math.cos(math.radians(x))

def sind(x: float) -> float:
    return math.sin(math.radians(x))


# --------------------------- main wrapper ---------------------------
def multilayer_main(Number_of_layers: int,
                    Thickness_layers: List[float],
                    Modulus_layers: List[float],
                    Poissons: List[float],
                    Tyre_pressure: float,
                    wheel_load: float,
                    wheel_set: int,
                    analysis_points: int,
                    depths: List[float],
                    radii: List[float],
                    isbonded: bool,
                    center_spacing: float,
                    alpha_deg: float) -> Tuple[np.ndarray, Any]:
    """
    Direct translation of MATLAB 'multilayer_main' with identical logic/flow.
    Returns (Report as ndarray, ResultTable as pandas DataFrame).
    """

    # ----------------------------- INPUT CHECKS -----------------------------
    n = Number_of_layers
    assert n >= 2, 'Number_of_layers must be >= 2.'
    assert np.ndim(Thickness_layers) == 1 and len(Thickness_layers) == (n - 1), 'Thickness_layers must be 1x(n-1).'
    assert np.ndim(Modulus_layers) == 1 and len(Modulus_layers) == n, 'Modulus_layers must be 1x n.'
    assert np.ndim(Poissons) == 1 and len(Poissons) == n, 'Poissons must be 1x n.'
    assert Tyre_pressure > 0, 'Tyre_pressure must be >0.'
    assert wheel_load > 0, 'wheel_load must be >0.'
    assert wheel_set in (1, 2), 'wheel_set must be 1 or 2.'
    assert analysis_points >= 1, 'analysis_points must be >=1.'
    assert np.ndim(depths) == 1 and np.ndim(radii) == 1 and len(depths) == len(radii) == analysis_points, \
        'depths/radii length must equal analysis_points.'
    assert isinstance(isbonded, (bool, np.bool_)) or isbonded in (0, 1), 'isbonded must be logical.'
    assert center_spacing >= 0, 'center_spacing must be >=0.'
    # alpha_deg: any scalar

    # ------------------------- LAYOUT / CONTACT PARAMS ----------------------
    Thickness = list(Thickness_layers)       # 1 x (n-1)
    E = list(Modulus_layers)                 # 1 x n
    v = list(Poissons)                       # 1 x n
    q = float(Tyre_pressure)
    a = math.sqrt(wheel_load / (math.pi * q))   # contact radius (per wheel)
    alpha = float(alpha_deg)                    # deg
    s = float(center_spacing)

    # ------------------------------ MAIN LOOP --------------------------------
    Report: List[List[float]] = []
    interface_depths = np.cumsum(Thickness_layers)  # 1 x (n-1)

    for ij in range(1, analysis_points + 1):
        # prepare per-iteration containers (kept for parity with MATLAB)
        Points: List[Dict[str, Any]] = []
        PointsW1: List[Dict[str, Any]] = []
        PointsW2: List[Dict[str, Any]] = []
        PointsW1O: List[Dict[str, Any]] = []
        PointsW2O: List[Dict[str, Any]] = []

        z = float(depths[ij - 1])
        R_left = float(radii[ij - 1])
        if abs(R_left) < 1e-12:
            R_left = 0.0

        # Check if z sits exactly on an interface (within tolerance)
        tol = 1e-9
        idx_if = None
        diffs = np.abs(interface_depths - z)
        where_if = np.where(diffs <= tol)[0]
        if where_if.size > 0:
            idx_if = int(where_if[0])                # 0-based
            is_interface = True
            upper_layer = idx_if + 1                 # 1-based
            lower_layer = idx_if + 2                 # 1-based
            layer_within: Optional[int] = None
        else:
            is_interface = False
            k = np.where(z > interface_depths)[0]
            if k.size == 0:
                layer_within = 1
            else:
                layer_within = int(k[-1] + 2)        # +2 -> below that interface (1-based)
            upper_layer = None
            lower_layer = None

        if wheel_set == 1:
            # ------------------------- SINGLE WHEEL -------------------------
            if not is_interface:
                Points.append({'r': R_left, 'z': z, 'layer': layer_within})
                Results_W1 = AIO_R(n, Thickness, E, v, bool(isbonded), Points, a, q)

                R = Results_W1[0]
                Report.append([ij, z, R_left, R['sigma_z'], R['sigma_t'], R['sigma_r'],
                               R['tau_rz'], R['w'], R['epsilon_z'], R['epsilon_t'], R['epsilon_r']])
            else:
                Points.append({'r': R_left, 'z': z, 'layer': upper_layer})
                Points.append({'r': R_left, 'z': z, 'layer': lower_layer})

                Results_W1 = AIO_R(n, Thickness, E, v, bool(isbonded), Points, a, q)

                Ru, Rl = Results_W1[0], Results_W1[1]
                Report.append([ij, z, R_left, Ru['sigma_z'], Ru['sigma_t'], Ru['sigma_r'],
                               Ru['tau_rz'], Ru['w'], Ru['epsilon_z'], Ru['epsilon_t'], Ru['epsilon_r']])
                Report.append([ij, z, R_left, Rl['sigma_z'], Rl['sigma_t'], Rl['sigma_r'],
                               Rl['tau_rz'], Rl['w'], Rl['epsilon_z'], Rl['epsilon_t'], Rl['epsilon_r']])

        else:
            # -------------------------- DUAL WHEELS -------------------------
            r1 = abs(R_left)          # to LEFT center (x=0)
            r2 = abs(R_left - s)      # to RIGHT center (x=+s)
            if r1 < 1e-12:
                r1 = 0.0
            if r2 < 1e-12:
                r2 = 0.0

            if not is_interface:
                # one layer; evaluate each wheel at same (z, layer_within)
                PointsW1.append({'r': r1, 'z': z, 'layer': layer_within})
                PointsW2.append({'r': r2, 'z': z, 'layer': layer_within})

                Results_W1 = AIO_R(n, Thickness, E, v, bool(isbonded), PointsW1, a, q)
                Results_W2 = AIO_R(n, Thickness, E, v, bool(isbonded), PointsW2, a, q)

                # superpose (alpha rotation of local (r,t) → global (x,y)) EXACTLY as MATLAB lines
                Sigma_x = (Results_W1[0]['sigma_r'] * (cosd(alpha) ** 2) +
                           Results_W1[0]['sigma_t'] * (sind(alpha) ** 2) +
                           Results_W2[0]['sigma_r'] * (cosd(alpha) ** 2) +
                           Results_W2[0]['sigma_t'] * (sind(alpha) ** 2))

                Sigma_y = (Results_W1[0]['sigma_r'] * (sind(alpha) ** 2) +
                           Results_W1[0]['sigma_t'] * (cosd(alpha) ** 2) +
                           Results_W2[0]['sigma_r'] * (sind(alpha) ** 2) +
                           Results_W2[0]['sigma_t'] * (cosd(alpha) ** 2))

                Sigma_z = Results_W1[0]['sigma_z'] + Results_W2[0]['sigma_z']
                Tau_xz  = (Results_W1[0]['tau_rz'] * cosd(alpha) +
                           Results_W2[0]['tau_rz'] * cosd(alpha))

                ii = layer_within
                Ez = E[ii - 1]
                vz = v[ii - 1]
                ez = (Sigma_z - vz * (Sigma_x + Sigma_y)) / Ez
                ex = (Sigma_x - vz * (Sigma_y + Sigma_z)) / Ez   # epsilon_x -> 'epsilon_r' column
                ey = (Sigma_y - vz * (Sigma_x + Sigma_z)) / Ez   # epsilon_y -> 'epsilon_t' column

                w_sum = Results_W1[0]['w'] + Results_W2[0]['w']

                # NOTE: Column order matches MATLAB line (Sigma_z, Sigma_t=ey, Sigma_r=ex)
                Report.append([ij, z, R_left, Sigma_z, ey, ex, Tau_xz, w_sum, ez, ey, ex])

            else:
                # interface; evaluate upper & lower for each wheel then superpose
                PointsW1O.append({'r': r1, 'z': z, 'layer': upper_layer})
                PointsW1O.append({'r': r1, 'z': z, 'layer': lower_layer})

                PointsW2O.append({'r': r2, 'z': z, 'layer': upper_layer})
                PointsW2O.append({'r': r2, 'z': z, 'layer': lower_layer})

                Results_W1_O = AIO_R(n, Thickness, E, v, bool(isbonded), PointsW1O, a, q)
                Results_W2_O = AIO_R(n, Thickness, E, v, bool(isbonded), PointsW2O, a, q)

                Results_O = AIO_wheel_superposition(E, v, alpha, PointsW1O, PointsW2O, Results_W1_O, Results_W2_O)

                Ru, Rl = Results_O[0], Results_O[1]
                Report.append([ij, z, R_left, Ru['sigma_z'], Ru['sigma_t'], Ru['sigma_r'],
                               Ru['tau_rz'], Ru['w'], Ru['epsilon_z'], Ru['epsilon_t'], Ru['epsilon_r']])
                Report.append([ij, z, R_left, Rl['sigma_z'], Rl['sigma_t'], Rl['sigma_r'],
                               Rl['tau_rz'], Rl['w'], Rl['epsilon_z'], Rl['epsilon_t'], Rl['epsilon_r']])

    VarNames = ['Point', 'Depth', 'Radius_from_left', 'Sigma_z', 'Sigma_t',
                'Sigma_r', 'Tau_xz', 'w', 'ez', 'et', 'er']
    Report_arr = np.array(Report, dtype=float) if len(Report) else np.zeros((0, len(VarNames)))
    if pd is not None:
        ResultTable = pd.DataFrame(Report_arr, columns=VarNames)
    else:
        # Fallback structure resembling the JSON-shaped DataFrame used by the bridge
        ResultTable = {
            "_type": "DataFrame",
            "columns": VarNames,
            "data": [
                {k: float(v) for k, v in zip(VarNames, row)} for row in Report_arr.tolist()
            ],
        }
    return Report_arr, ResultTable


# --------------------------- solver: AIO_R ---------------------------
def AIO_R(n: int,
          Thickness: List[float],
          E: List[float],
          v: List[float],
          isbonded: bool,
          Points: List[Dict[str, float]],
          a: float,
          q: float) -> List[Dict[str, float]]:

    z_interfaces = np.cumsum(np.array(Thickness, dtype=float))  # 1..(n-1)
    H = float(z_interfaces[-1])

    alpha = a / H
    dl = 0.2 / alpha

    # initialize accumulators
    Results = []
    for _ in range(len(Points)):
        Results.append({'sigma_z': 0.0, 'sigma_r': 0.0, 'sigma_t': 0.0,
                        'tau_rz': 0.0, 'w': 0.0, 'u': 0.0})

    # integrate m from 0:dl:200 with 4-pt Gaussian inside each interval
    for l in np.arange(0.0, 200.0 + 1e-12, dl):
        mids = (l + dl / 2.0) + np.array([-0.86114, -0.33998, 0.33998, 0.86114]) * (dl / 2.0)
        fcs = np.array([0.34786, 0.65215, 0.65215, 0.34786]) * (dl / 2.0)

        Hat_all = [AIO_R_hat(n, Thickness, E, v, isbonded, float(m), Points) for m in mids]

        for gp in range(4):
            m = float(mids[gp])
            fc = float(fcs[gp])
            if m != 0.0:
                J1 = float(besselj(1, m * alpha))
                factor = q * alpha * J1 / m

                for j in range(len(Points)):
                    Rj = Hat_all[gp][j]
                    Results[j]['sigma_z'] += fc * (factor * Rj['sigma_z'])
                    Results[j]['sigma_r'] += fc * (factor * Rj['sigma_r'])
                    Results[j]['sigma_t'] += fc * (factor * Rj['sigma_t'])
                    Results[j]['tau_rz']  += fc * (factor * Rj['tau_rz'])
                    Results[j]['w']       += fc * (factor * Rj['w'])
                    Results[j]['u']       += fc * (factor * Rj['u'])

    # strains via Hooke (isotropic) at each point (1-based layer index)
    for j in range(len(Points)):
        ii = int(Points[j]['layer'])     # 1..n
        Ei = float(E[ii - 1])
        vi = float(v[ii - 1])
        sigz = Results[j]['sigma_z']
        sigr = Results[j]['sigma_r']
        sigt = Results[j]['sigma_t']
        Results[j]['epsilon_z'] = (sigz - vi * (sigr + sigt)) / Ei
        Results[j]['epsilon_r'] = (sigr - vi * (sigz + sigt)) / Ei
        Results[j]['epsilon_t'] = (sigt - vi * (sigr + sigz)) / Ei

    return Results


# --------------------------- solver: AIO_R_hat ---------------------------
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

    z = np.cumsum(Thickness)
    H = float(z[-1])

    lam = z / H                         # 1..(n-1)
    lam = np.concatenate([lam, [np.inf]])  # λ(n)=inf

    F = np.zeros(n, dtype=float)
    R = np.zeros(n - 1, dtype=float)

    F[0] = math.exp(-m * (lam[0] - 0.0))
    R[0] = (E[0] / E[1]) * (1.0 + v[1]) / (1.0 + v[0])
    for i in range(1, n - 1):
        F[i] = math.exp(-m * (lam[i] - lam[i - 1]))
        R[i] = (E[i] / E[i + 1]) * (1.0 + v[i + 1]) / (1.0 + v[i])
    F[n - 1] = math.exp(-m * (lam[n - 1] - lam[n - 2]))

    M_list: List[np.ndarray] = []
    for i in range(n - 1):
        Fi = F[i]; Fip1 = F[i + 1]
        lam_i = lam[i]
        vi = v[i]; vip1 = v[i + 1]
        Ri = R[i]

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

        X = np.linalg.solve(M1, M2)
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

    A = np.zeros(n); B = np.zeros(n); C = np.zeros(n); D = np.zeros(n)
    denom = (k11 * k22 - k12 * k21)
    B[n - 1] = k22 / denom
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

        outs.append({'sigma_z': sigma_z, 'sigma_r': sigma_r, 'sigma_t': sigma_t,
                     'tau_rz': tau_rz, 'w': w, 'u': u})

    return outs


# --------------------------- superposition helper ---------------------------
def AIO_wheel_superposition(E: List[float], v: List[float], alpha: float,
                            PointsW1: List[Dict[str, float]],
                            PointsW2: List[Dict[str, float]],
                            Results_W1: List[Dict[str, float]],
                            Results_W2: List[Dict[str, float]]) -> List[Dict[str, float]]:
    """
    Direct translation of MATLAB AIO_wheel_superposition (field mapping preserved).
    """
    LayerT = int(PointsW1[0]['layer'])
    LayerD = int(PointsW2[1]['layer'])

    # rotate local (r,t) to global (x,y) by alpha (deg)
    def rot_xy(sr: float, st: float, a: float) -> Tuple[float, float]:
        return (sr * (cosd(a) ** 2) + st * (sind(a) ** 2),
                sr * (sind(a) ** 2) + st * (cosd(a) ** 2))

    # ---- UPPER (index 1)
    Sx1, Sy1 = rot_xy(Results_W1[0]['sigma_r'], Results_W1[0]['sigma_t'], alpha)
    Sx2, Sy2 = rot_xy(Results_W2[0]['sigma_r'], Results_W2[0]['sigma_t'], alpha)

    Sigma_x_L2 = Sx1 + Sx2
    Sigma_y_L2 = Sy1 + Sy2
    Sigma_z_L2 = Results_W1[0]['sigma_z'] + Results_W2[0]['sigma_z']
    tau_xz_L2  = Results_W1[0]['tau_rz'] * cosd(alpha) + Results_W2[0]['tau_rz'] * cosd(alpha)

    E2 = E[LayerT - 1]; v2 = v[LayerT - 1]
    ex2 = (Sigma_x_L2 - v2 * (Sigma_y_L2 + Sigma_z_L2)) / E2
    ey2 = (Sigma_y_L2 - v2 * (Sigma_x_L2 + Sigma_z_L2)) / E2
    ez2 = (Sigma_z_L2 - v2 * (Sigma_x_L2 + Sigma_y_L2)) / E2

    out_upper = {'sigma_z': Sigma_z_L2, 'sigma_t': Sigma_y_L2, 'sigma_r': Sigma_x_L2,
                 'tau_rz': tau_xz_L2, 'w': Results_W1[0]['w'] + Results_W2[0]['w'], 'u': 0.0,
                 'epsilon_z': ez2, 'epsilon_t': ey2, 'epsilon_r': ex2}

    # ---- LOWER (index 2)
    Sx1b, Sy1b = rot_xy(Results_W1[1]['sigma_r'], Results_W1[1]['sigma_t'], alpha)
    Sx2b, Sy2b = rot_xy(Results_W2[1]['sigma_r'], Results_W2[1]['sigma_t'], alpha)

    Sigma_x_L3 = Sx1b + Sx2b
    Sigma_y_L3 = Sy1b + Sy2b
    Sigma_z_L3 = Results_W1[1]['sigma_z'] + Results_W2[1]['sigma_z']
    tau_xz_L3  = Results_W1[1]['tau_rz'] * cosd(alpha) + Results_W2[1]['tau_rz'] * cosd(alpha)

    E3 = E[LayerD - 1]; v3 = v[LayerD - 1]
    ex3 = (Sigma_x_L3 - v3 * (Sigma_y_L3 + Sigma_z_L3)) / E3
    ey3 = (Sigma_y_L3 - v3 * (Sigma_x_L3 + Sigma_z_L3)) / E3
    ez3 = (Sigma_z_L3 - v3 * (Sigma_x_L3 + Sigma_y_L3)) / E3

    out_lower = {'sigma_z': Sigma_z_L3, 'sigma_t': Sigma_y_L3, 'sigma_r': Sigma_x_L3,
                 'tau_rz': tau_xz_L3, 'w': Results_W1[1]['w'] + Results_W2[1]['w'], 'u': 0.0,
                 'epsilon_z': ez3, 'epsilon_t': ey3, 'epsilon_r': ex3}

    return [out_upper, out_lower]


# --------------------------- demo run (matches your MATLAB script) ---------------------------
if __name__ == "__main__":
    # ---------------- INPUTS ----------------
    Number_of_layers = 4
    Thickness_layers = [100.00, 240.00, 200.00]
    Modulus_layers   = [3000.00, 617.94, 300.11, 76.83]
    Poissons         = [0.35, 0.35, 0.35, 0.35]

    Tyre_pressure = 0.56
    wheel_load = 20000

    analysis_points = 4
    depths = [100, 100, 540, 540]
    radii_from_left = [0, 155, 0, 155]

    isbonded = True
    center_spacing = 310
    alpha_deg = 0

    wheel_set = 2

    Report_dual, Table_dual = multilayer_main(
        Number_of_layers,
        Thickness_layers,
        Modulus_layers,
        Poissons,
        Tyre_pressure,
        wheel_load,
        wheel_set,
        analysis_points,
        depths,
        radii_from_left,
        isbonded,
        center_spacing,
        alpha_deg
    )

    print('--- Dual-wheel result (table) ---')
    print(Table_dual.to_string(index=False))
