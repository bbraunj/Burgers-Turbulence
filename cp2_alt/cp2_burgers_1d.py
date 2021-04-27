# References:
# -----------
# [1] San et. al. 2014 Numerical.
#     https://doi.org/10.1016/j.compfluid.2013.11.006
# [2] Maulik et. al. 2018 Adaptive. https://doi.org/10.1002/fld.4489
# [3] Jiang et. al. 1999. https://doi.org/10.1006/jcph.1999.6207
#
import matplotlib.pyplot as plt
from numba import jit
import numpy as np
from pathlib import Path


def main(cases=[{"recon_scheme": "WENO-5", "limiter": None, "nx": 2000}]):

    results = []
    for case in cases:
        recon_scheme = case["recon_scheme"]
        limiter = case["limiter"]
        nx = case["nx"]

        l_ = "" if limiter is None else f", {limiter} Limiter"
        title = (f"\nSolving Burgers Problem Using {recon_scheme} "
                 f"Reconstruction{l_}, nx={nx}")
        print(title + "\n" + "="*len(title))

        result = outer_loop(recon_scheme, limiter, nx)
        results.append(result)

    plot_results(results)


def plot_results(results):
    plt.clf()
    for r in results:
        y = r["q"]
        l_ = "" if r["limiter"] is None else f", {r['limiter']}"
        label = f"{r['recon_scheme']}{l_}, nx={r['nx']}"

        plt.plot(np.linspace(0, r["lx"], r["nx"]), y,
                 linewidth=0.4, label=label)

    plt.legend()
    plt.ylabel("u")
    plt.xlabel("x")
    plt.suptitle(f"Solutions to Burgers Problem at t={r['t']:.4f}s")
    Path("output").mkdir(exist_ok=True)
    plt.savefig("output/Results_u.png", dpi=400)


def outer_loop(reconstruction_scheme, limiter, nx):
    lx = 1
    dx = lx/nx
    q = get_IC_q(nx, dx)
    tmax = 0.1  # s
    time = 0

    print(f"nx = {nx}")
    while time < tmax:
        dt = get_dt(q, nx, dx)

        # Make sure we don't go past tmax
        dt = (dt
              if time+dt < tmax
              else tmax - time)
        time += dt

        q = tvdrk3(nx, dx, dt, q, reconstruction_scheme, limiter)
        print(f"  * time: {time:.4f} s", end="\r")

    print("\n  * Done")

    result = {
        "lx": lx,
        "nx": nx,
        "q": q[2:nx+2],
        "t": time,
        "recon_scheme": reconstruction_scheme,
        "limiter": limiter
    }

    return result


def tvdrk3(nx, dx, dt, q_n, reconstruction_scheme, limiter):
    """
    3rd Order Runge-Kutta time integrator.
    """
    i = np.arange(3, nx+2)
    q1 = np.copy(q_n)
    q1[i, :] = q_n[i, :] + dt*rhs(nx, dx, q_n, reconstruction_scheme, limiter)

    q2 = np.copy(q_n)
    q2[i, :] = (0.75*q_n[i, :] + 0.25*q1[i, :]
                + 0.25*dt*rhs(nx, dx, q1, reconstruction_scheme, limiter))

    q_np1 = np.copy(q_n)
    q_np1[i, :] = ((1/3)*q_n[i, :] + (2/3)*q2[i, :]
                   + (2/3)*dt*rhs(nx, dx, q2, reconstruction_scheme, limiter))

    return q_np1


def rhs(nx, dx, q, reconstruction_scheme, limiter):
    """
    Note::
        q.shape == (nx+5, :)
                    ^^^^ Specifically from 0:(nx+5)
        rhs.shape == (nx-1, :)
                      ^^^^ Specifically from 1:nx + 2
    """
    # Periodic boundary conditions
    q[2, :] = q[nx+1, :]
    q[1, :] = q[nx, :]
    q[0, :] = q[nx-1, :]
    q[nx+2, :] = q[3, :]
    q[nx+3, :] = q[4, :]
    q[nx+4, :] = q[5, :]

    # Reconstruction Schemes
    if reconstruction_scheme == "MUSCL":
        qL_ip12, qR_ip12 = muscl(q, nx, limiter)
    elif reconstruction_scheme == "WENO-3":
        qL_ip12, qR_ip12 = weno_3(q, nx)
    elif reconstruction_scheme == "WENO-5":
        qL_ip12, qR_ip12 = weno_5(q, nx)
    elif reconstruction_scheme == "WENO-5C":
        qL_ip12, qR_ip12 = weno_5c(q, nx)
    # ----------------------

    c_ip12 = get_wave_speeds(q, nx)
    FR = get_flux(qR_ip12)
    FL = get_flux(qL_ip12)

    # Riemann Solvers
    F_ip12 = rusanov_riemann(FR, FL, qR_ip12, qL_ip12, c_ip12)
    # ---------------

    i = np.arange(1, nx) + 2
    rhs = -(F_ip12[i, :] - F_ip12[i-1, :]) / dx
    #       F_ip12         F_im12

    return rhs


def rusanov_riemann(FR, FL, qR_ip12, qL_ip12, c_ip12):
    F_ip12 = 0.5*((FR + FL) - c_ip12*(qR_ip12 - qL_ip12))

    return F_ip12


# ==============================================
# =========== Reconstruction Schemes ===========
# ==============================================
def muscl(q, nx, limiter):
    #
    # MUSCL reconstruction scheme
    #
    qL_ip12, qR_ip12 = [np.zeros(q.shape) for _ in range(2)]
    k = 0.5

    # Limiters
    # --------
    def van_leer_limiter(r):
        limiter = (r + abs(r)) / (1 + r)
        limiter[np.where(np.isnan(limiter))] = 2
        return limiter

    def van_albada_limiter(r):
        limiter = (r**2 + r) / (r**2 + 1)
        limiter[np.where(np.isnan(limiter))] = 1
        return limiter

    def minmod_limiter(r):
        limiter = np.stack([
            np.zeros(r.shape),
            np.stack([r, np.ones(r.shape)], axis=0).min(axis=0),
        ], axis=0).max(axis=0)
        limiter[np.where(np.isnan(limiter))] = 1
        return limiter

    def superbee_limiter(r):
        limiter = np.stack([
            np.zeros(r.shape),
            np.stack([2*r, np.ones(r.shape)], axis=0).min(axis=0),
            np.stack([r, 2*np.ones(r.shape)], axis=0).min(axis=0),
        ], axis=0).max(axis=0)
        limiter[np.where(np.isnan(limiter))] = 2
        return limiter

    def monotonized_central_limiter(r):
        limiter = np.stack([
            np.zeros(r.shape),
            np.stack([2*r, 0.5*(1+r), 2*np.ones(r.shape)],
                     axis=0).min(axis=0),
        ], axis=0).max(axis=0)
        limiter[np.where(np.isnan(limiter))] = 2
        return limiter

    limiter_fcn = {
        "Van Leer": van_leer_limiter,
        "Van Albada": van_albada_limiter,
        "Min-Mod": minmod_limiter,
        "Superbee": superbee_limiter,
        "Monotonized Central": monotonized_central_limiter,
    }[limiter]
    # --------

    # Positive reconstruction @ i+1/2
    i = np.arange(0, nx) + 2

    r = (q[i] - q[i-1]) / (q[i+1] - q[i])
    qL_ip12[i] = q[i] + 0.25*(
        (1-k)*limiter_fcn(1/r)*(q[i] - q[i-1]) +
        (1+k)*limiter_fcn(r)*(q[i+1] - q[i])
    )

    # Negative reconstruction @ (i-1/2) + 1
    i += 1

    r = (q[i] - q[i-1]) / (q[i+1] - q[i])
    qR_ip12[i-1] = q[i] - 0.25*(
        (1+k)*limiter_fcn(1/r)*(q[i] - q[i-1]) +
        (1-k)*limiter_fcn(r)*(q[i+1] - q[i])
    )

    return qL_ip12, qR_ip12


def weno_3(q, nx):
    #
    # 3rd order WENO reconstruction scheme
    #
    b0, b1, qL_ip12, qR_ip12 = [np.zeros(q.shape) for _ in range(4)]

    # Positive reconstruction @ i+1/2
    # Smoothness indicators
    i = np.arange(-1, nx+1) + 2
    b0[i] = (q[i] - q[i-1])**2
    b1[i] = (q[i+1] - q[i])**2

    # Linear weighting coefficients
    d0 = 1/3
    d1 = 2/3

    # Nonliear weights
    eps = 1e-6
    a0 = d0 / (b0[i] + eps)**2
    a1 = d1 / (b1[i] + eps)**2

    w0 = a0 / (a0 + a1)
    w1 = a1 / (a0 + a1)

    q0 = -q[i-1] + 3*q[i]
    q1 = q[i] + q[i+1]
    qL_ip12[1:(nx+3)] = (w0/2)*q0 + (w1/2)*q1

    # Negative reconstruction @ (i-1/2) + 1
    i += 1
    # Smoothness indicators
    b0[i] = (q[i] - q[i-1])**2
    b1[i] = (q[i+1] - q[i])**2

    # Linear weighting coefficients
    d0 = 2/3
    d1 = 1/3

    # Nonliear weights
    eps = 1e-6
    a0 = d0 / (b0[i] + eps)**2
    a1 = d1 / (b1[i] + eps)**2

    w0 = a0 / (a0 + a1)
    w1 = a1 / (a0 + a1)

    q0 = q[i-1] + q[i]
    q1 = 3*q[i] - q[i+1]
    qR_ip12[1:(nx+3)] = (w0/2)*q0 + (w1/2)*q1

    return qL_ip12, qR_ip12


def weno_5(q, nx):
    """
    WENO 5th Order reconstruction scheme [2]

                 Cell
                  |
      Boundary -| | |- Boundary
                v v v
    |-*-|-*-|...|-*-|...|-*-|
      0   1     ^ i ^     n
            i-1/2   i+1/2

    - Based on cell definition above, estimate q_i-1/2 and q_i+1/2 using
      nodes values from cells on the left (q^L) or cells on the right (q^R).

    # Note::
    #     There may be ghost points on either end of the computational domain.
    #     For this reason, s & e are inputs to define the indices of the start
    #     and end of the computational domain. All points before and after
    #     these indices are ghost points.

    Returns:
        tuple: A tuple containing q^L_i+1/2 and q^R_i+1/2 respectively.
    """
    b0, b1, b2, qL_ip12, qR_ip12 = [np.zeros(q.shape) for _ in range(5)]

    # Smoothness indicators
    i = np.arange(0, nx+1) + 2
    b0[i, :] = ((13/12)*(q[i-2, :] - 2*q[i-1, :] + q[i, :])**2
                + (1/4)*(q[i-2, :] - 4*q[i-1, :] + 3*q[i, :])**2)
    b1[i, :] = ((13/12)*(q[i-1, :] - 2*q[i, :] + q[i+1, :])**2
                + (1/4)*(q[i-1, :] - q[i+1, :])**2)
    b2[i, :] = ((13/12)*(q[i, :] - 2*q[i+1, :] + q[i+2, :])**2
                + (1/4)*(3*q[i, :] - 4*q[i+1, :] + q[i+2, :])**2)

    # Linear weighting coefficients
    d0 = 1/10
    d1 = 6/10
    d2 = 3/10

    # Nonliear weights
    eps = 1e-6
    i = np.arange(0, nx) + 2
    a0 = d0 / (b0[i, :] + eps)**2
    a1 = d1 / (b1[i, :] + eps)**2
    a2 = d2 / (b2[i, :] + eps)**2

    w0 = a0 / (a0 + a1 + a2)
    w1 = a1 / (a0 + a1 + a2)
    w2 = a2 / (a0 + a1 + a2)

    # Positive reconstruction @ i+1/2
    q0 = 2*q[i-2, :] - 7*q[i-1, :] + 11*q[i, :]
    q1 = -q[i-1, :] + 5*q[i, :] + 2*q[i+1, :]
    q2 = 2*q[i, :] + 5*q[i+1, :] - q[i+2, :]
    qL_ip12[i, :] = (w0/6)*q0 + (w1/6)*q1 + (w2/6)*q2

    # Negative reconstruction @ i-1/2 + 1 (so i+1/2)
    i += 1
    a0 = d0 / (b0[i, :] + eps)**2
    a1 = d1 / (b1[i, :] + eps)**2
    a2 = d2 / (b2[i, :] + eps)**2

    w0 = a0 / (a0 + a1 + a2)
    w1 = a1 / (a0 + a1 + a2)
    w2 = a2 / (a0 + a1 + a2)

    q0 = 2*q[i+2, :] - 7*q[i+1, :] + 11*q[i, :]
    q1 = -q[i+1, :] + 5*q[i, :] + 2*q[i-1, :]
    q2 = 2*q[i, :] + 5*q[i-1, :] - q[i-2, :]
    qR_ip12[i-1, :] = (w0/6)*q0 + (w1/6)*q1 + (w2/6)*q2

    return qL_ip12, qR_ip12


def weno_5c(q, nx):
    #
    # Periodic Compact 5th Order WENO reconstruction
    #
    qL_ip12, qR_ip12 = [np.zeros(q.shape) for _ in range(2)]
    q = q[:, 0]

    # Positive reconstruction @ i+1/2
    i = np.arange(0, nx) + 2
    v1, v2, v3, v4, v5 = (q[i-2], q[i-1], q[i], q[i+1], q[i+2])
    a1, a2, a3, b1, b2, b3 = CRWENO5(v1, v2, v3, v4, v5)

    aa = a1
    bb = a2
    cc = a3
    rr = b1*q[i-1] + b2*q[i] + b3*q[i+1]

    qL_ip12[i, 0] = TDMA_cyclic(aa, bb, cc, rr, aa[0], cc[-1])

    # Negative reconstruction @ i-1/2 + 1 (so i+1/2)
    i += 1
    v1, v2, v3, v4, v5 = (q[i+2], q[i+1], q[i], q[i-1], q[i-2])
    a1, a2, a3, b1, b2, b3 = CRWENO5(v1, v2, v3, v4, v5)

    aa = a3
    bb = a2
    cc = a1
    rr = b1*q[i+1] + b2*q[i] + b3*q[i-1]

    qR_ip12[i-1, 0] = TDMA_cyclic(aa, bb, cc, rr, aa[0], cc[-1])

    return qL_ip12, qR_ip12


def CRWENO5(v1, v2, v3, v4, v5):
    # Smoothness indicators
    s0 = ((13/12)*(v1 - 2*v2 + v3)**2 + (1/4)*(v1 - 4*v2 + 3*v3)**2)
    s1 = ((13/12)*(v2 - 2*v3 + v4)**2 + (1/4)*(v2 - v4)**2)
    s2 = ((13/12)*(v3 - 2*v4 + v5)**2 + (1/4)*(3*v3 - 4*v4 + v5)**2)

    # Linear weighting coefficients
    d0 = 1/5
    d1 = 1/2
    d2 = 3/10

    # Nonliear weights
    eps = 1e-6
    c0 = d0 / (s0 + eps)**2
    c1 = d1 / (s1 + eps)**2
    c2 = d2 / (s2 + eps)**2

    w0 = c0 / (c0 + c1 + c2)
    w1 = c1 / (c0 + c1 + c2)
    w2 = c2 / (c0 + c1 + c2)

    # Tridiagonal coefficients LHS
    a1 = (2/3)*w0 + (1/3)*w1
    a2 = (1/3)*w0 + (2/3)*(w1+w2)
    a3 = (1/3)*w2

    # Tridiagonal coefficients RHS
    b1 = (1/6)*w0
    b2 = (1/6)*(5*(w0+w1) + w2)
    b3 = (1/6)*(w1 + 5*w2)

    return a1, a2, a3, b1, b2, b3


@jit(nopython=True, nogil=True)
def TDMA_cyclic(a, b, c, d, alpha, beta):
    nf = d.shape[0]
    ac = np.copy(a)
    bc = np.copy(b)
    cc = np.copy(c)
    dc = np.copy(d)

    gamma = 10.0*b[0]
    bc[0] = b[0] - gamma
    bc[-1] = b[-1] - alpha*beta/gamma
    y = TDMAsolver(ac, bc, cc, dc)

    u = np.zeros(nf)
    u[0] = gamma
    u[-1] = alpha
    q = TDMAsolver(ac, bc, cc, u)

    x = y - q*((y[0] + (beta/gamma)*y[-1]) /
               (1 + q[0] + (beta/gamma)*q[-1]))

    return x


@jit(nopython=True, nogil=True)
def TDMAsolver(a, b, c, d):
    '''
    TDMA solver, a b c d can be NumPy array type or Python list type.
    refer to http://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
    and to http://www.cfd-online.com/Wiki/Tridiagonal_matrix_algorithm_-
    _TDMA_(Thomas_algorithm)
    '''
    nf = d.shape[0]  # number of equations
    ac = np.copy(a)
    bc = np.copy(b)
    cc = np.copy(c)
    dc = np.copy(d)
    # ac, bc, cc, dc = map(np.array, (a, b, c, d))  # copy arrays
    for i in range(1, nf):
        mc = ac[i-1]/bc[i-1]
        bc[i] = bc[i] - mc*cc[i-1]
        dc[i] = dc[i] - mc*dc[i-1]

    xc = bc
    xc[-1] = dc[-1]/bc[-1]

    for i in range(nf-2, -1, -1):
        xc[i] = (dc[i]-cc[i]*xc[i+1])/bc[i]

    return xc


# ==========================================
# =========== Problem Quantities ===========
# ==========================================
def get_IC_q(nx, dx):
    q = np.zeros((nx+5, 1))
    q[(int(0.5/dx)+2):(int(0.6/dx)+2), :] = 1

    return q


def get_dt(q, nx, dx):
    cfl = 0.5
    c_ip12 = get_wave_speeds(q, nx)
    dt = np.nanmin(cfl * dx / c_ip12)

    return dt


def get_wave_speeds(q, nx):
    c_ip12 = np.zeros(q.shape)
    i = np.arange(0, nx) + 2
    c_ip12[i, :] = abs(np.vstack([
        q[i-2, :], q[i-1, :], q[i, :],
        q[i+1, :], q[i+2, :], q[i+3, :]
    ])).max(axis=0)

    return c_ip12


def get_flux(q):
    F = q**2 / 2

    return F


if __name__ == "__main__":
    # main(cases=[
    #     {"recon_scheme": "WENO-5", "limiter": None, "nx": 2000},  # Base-line
    #     {"recon_scheme": "WENO-3", "limiter": None, "nx": 100},
    #     {"recon_scheme": "WENO-5", "limiter": None, "nx": 100},
    # ])
    main(cases=[
        {"recon_scheme": "WENO-5", "limiter": None, "nx": 2000},  # Base-line
        {"recon_scheme": "MUSCL", "limiter": "Van Leer", "nx": 100},
        {"recon_scheme": "MUSCL", "limiter": "Van Albada", "nx": 100},
        {"recon_scheme": "MUSCL", "limiter": "Min-Mod", "nx": 100},
        {"recon_scheme": "MUSCL", "limiter": "Superbee", "nx": 100},
        {"recon_scheme": "MUSCL", "limiter": "Monotonized Central", "nx": 100},
    ])
