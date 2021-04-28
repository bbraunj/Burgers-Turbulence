# References:
# -----------
# [1] San et. al. 2014 Numerical.
#     https://doi.org/10.1016/j.compfluid.2013.11.006
# [2] Maulik et. al. 2018 Adaptive. https://doi.org/10.1002/fld.4489
# [3] Jiang et. al. 1999. https://doi.org/10.1006/jcph.1999.6207
#
import json
import matplotlib.pyplot as plt
from numba import jit
import numpy as np
from pathlib import Path
import random
import time
import warnings

warnings.simplefilter("ignore")


def main(cases=[
    {"recon_scheme": "WENO-5", "limiter": None, "nx": 2000, "ns": 4,
     "nu": 5e-4, "tmax": 0.05}
], export_as_DNS=False, DNS_nu=5e-4, output_dir=None):

    DNS_dir = Path(f"DNS/nu={DNS_nu:.5f}")
    if export_as_DNS:
        assert len(cases) == 1
        assert cases[0]["nu"] == DNS_nu
        assert not DNS_dir.exists()
    elif not DNS_dir.exists():
        print(f"\nWARNING: '{DNS_dir}' directory does not exist. Proceeding "
              "without DNS results to compare with.")

    results = []
    for case in cases:
        start_time = time.time()

        recon_scheme = case["recon_scheme"]  # Reconstruction Scheme
        limiter = case["limiter"]  # Limiter for MUSCL scheme
        nx = case["nx"]  # Number of cells in x direction
        ns = case["ns"]  # Number of samples
        nu = case["nu"]  # Viscosity
        tmax = case["tmax"]

        l_ = "\n" if limiter is None else f", {limiter} Limiter\n"
        title = (f"\nSolving Burgers Problem Using {recon_scheme} "
                 f"Reconstruction{l_}nx={nx}, ns={ns}, nu={nu:.1e}"
                 f", tmax={tmax:.2f}")
        print(title + "\n" + "="*len(title.split("\n")[1]))

        result = outer_loop(recon_scheme, limiter, nx, ns, nu, tmax)
        results.append(result)

        stop_time = time.time()
        print(f"Execution Time: {stop_time-start_time:.4f} s")

    if export_as_DNS:
        export_DNS_results(results[0], DNS_dir)
    else:
        if DNS_dir.exists():
            DNS_results = load_DNS_results(DNS_dir)
            results.insert(0, DNS_results)

        plot_results(results, output_dir)


def outer_loop(reconstruction_scheme, limiter, nx, ns, nu, tmax):
    lx = 2*np.pi
    dx = lx/nx
    q = get_IC_q(nx, dx, ns)
    time = 0

    q_hist = [q]
    E_hist = []   # Average kinetic energy
    Et_hist = []  # Dissipation rate of kinetic energy
    while time < tmax:
        dt = get_dt(q, nx, dx)

        # Make sure we don't go past tmax
        dt = (dt
              if time+dt < tmax
              else tmax - time)
        # Also make sure we get a data point at 0.05 s
        dt = (0.05-time
              if time < 0.05 and time+dt > 0.05
              else dt)
        time += dt
        E_k0 = get_KE_k_space(q, nx)
        E0 = get_total_KE(E_k0)

        q = tvdrk3(q, nx, dx, dt, nu, reconstruction_scheme, limiter)
        q_hist.append(q)

        E_k = get_KE_k_space(q, nx)
        E = get_total_KE(E_k)
        Et = -(E-E0)/dt
        D = get_total_KE_dissipation_rate(E_k, nu)
        E_hist.append({"E": E, "E_k": E_k, "t": time, "dt": dt})
        Et_hist.append({"Et": Et, "D": D, "t": time, "dt": dt})

        print(f"  * time: {time:.4f} s", end="\r")

    print("\n  * Done")

    result = {
        "lx": lx,
        "nx": nx,
        "ns": ns,
        "nu": nu,
        "q_hist": q_hist,
        "E_hist": E_hist,
        "Et_hist": Et_hist,
        "t": time,
        "recon_scheme": reconstruction_scheme,
        "limiter": limiter
    }

    return result


def get_KE_k_space(q, nx):
    # Get average kinetic energy of the domain (0:nx)
    # E = (1/nx) * 0.5*np.sum(u**2)

    u = q
    u_k = np.fft.fft(u, axis=0, norm="forward")

    Es = np.zeros((nx-1, u.shape[-1]))
    Es = 0.5*np.abs(u_k[3:nx+2])**2  # Energy spectrum
    i = np.arange(1, int(nx/2)-1)
    E_k = np.sum(0.5*(Es[i] + Es[nx-1-i]), axis=-1) / q.shape[-1]

    return E_k


def get_total_KE(E_k):
    # E_k goes from 0 < k < km
    # and the domain is -km < k < km
    # --> Multiply by 2 to get energy over symmetric domain
    EE = 2*np.sum(E_k)

    return EE


def get_total_KE_dissipation_rate(E_k, nu):
    # E_k goes from 0 < k < km
    # and the domain is -km < k < km
    # --> Multiply by 2 to get energy over symmetric domain
    # Also, E_k = 0.5*u_k**2, but D = nu * k**2 * u_k**2
    # --> Multiply by 2 again to get E_k -> u_k**2
    # (Multiply by 4 in total)
    ksq = np.arange(1, len(E_k)+1)**2
    DD = 4*nu*np.sum(ksq*E_k)

    return DD


def tvdrk3(q_n, nx, dx, dt, nu, reconstruction_scheme, limiter):
    """
    3rd Order Runge-Kutta time integrator.
    """
    i = np.arange(3, nx+2)
    q1 = np.copy(q_n)
    q1[i, :] = q_n[i, :] + dt*rhs(q_n, nx, dx, nu, reconstruction_scheme,
                                  limiter)
    q1 = perbc(q1, nx)

    q2 = np.copy(q_n)
    q2[i, :] = (0.75*q_n[i, :] + 0.25*q1[i, :]
                + 0.25*dt*rhs(q1, nx, dx, nu, reconstruction_scheme, limiter))
    q2 = perbc(q2, nx)

    q_np1 = np.copy(q_n)
    q_np1[i, :] = ((1/3)*q_n[i, :] + (2/3)*q2[i, :]
                   + (2/3)*dt*rhs(q2, nx, dx, nu, reconstruction_scheme,
                                  limiter))
    q_np1 = perbc(q_np1, nx)

    return q_np1


def rhs(q, nx, dx, nu, reconstruction_scheme, limiter):
    """
    Note::
        q.shape == (nx+5, :)
                    ^^^^ Specifically from 0:(nx+5)
        rhs.shape == (nx-1, :)
                      ^^^^ Specifically from 1:nx + 2
    """
    # Reconstruction Schemes
    if "MUSCL" in reconstruction_scheme:
        qL_ip12, qR_ip12 = muscl(q, nx, limiter, reconstruction_scheme)
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

    # Viscous contribution
    uxx = c4ddp(q[2:(nx+3)], nx+1, dx)  # 0 <= x <= nx+1

    rhs += nu*uxx[1:-1]

    return rhs


def perbc(q, nx):
    q[2, :] = q[nx+1, :]
    q[1, :] = q[nx, :]
    q[0, :] = q[nx-1, :]
    q[nx+2, :] = q[3, :]
    q[nx+3, :] = q[4, :]
    q[nx+4, :] = q[5, :]

    return q


def rusanov_riemann(FR, FL, qR_ip12, qL_ip12, c_ip12):
    F_ip12 = 0.5*((FR + FL) - c_ip12*(qR_ip12 - qL_ip12))

    return F_ip12


# ==============================================
# =========== Reconstruction Schemes ===========
# ==============================================
def muscl(q, nx, limiter, muscl_type):
    #
    # MUSCL reconstruction scheme
    #
    qL_ip12, qR_ip12 = [np.zeros(q.shape) for _ in range(2)]

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

    def no_limiter(r):
        return 1

    limiter = "None" if limiter is None else limiter
    limiter_fcn = {
        "Van Leer": van_leer_limiter,
        "Van Albada": van_albada_limiter,
        "Min-Mod": minmod_limiter,
        "Superbee": superbee_limiter,
        "Monotonized Central": monotonized_central_limiter,
        "None": no_limiter,
    }[limiter]
    # --------

    # MUSCL Type Determines k_ value
    # ------------------------------
    k_ = {
        "MUSCL-KT": [1, -1],
        "MUSCL-Quick": 0.5,
        "MUSCL-Central": 1,
        "MUSCL-Upwind": -1,
        "MUSCL-Fromm": 0,
        "MUSCL-3rd": 1/3,
    }[muscl_type]
    muscl_kt = True if muscl_type == "MUSCL-KT" else False
    # ------------------------------

    # Positive reconstruction @ i+1/2
    i = np.arange(0, nx) + 2
    r = (q[i] - q[i-1]) / (q[i+1] - q[i])

    k = k_ if not muscl_kt else k_[0]
    qL_ip12[i] = q[i] + 0.25*(
        (1-k)*limiter_fcn(1/r)*(q[i] - q[i-1]) +
        (1+k)*limiter_fcn(r)*(q[i+1] - q[i])
    )

    # Negative reconstruction @ (i-1/2) + 1
    i += 1

    r = (q[i] - q[i-1]) / (q[i+1] - q[i])
    k = k_ if not muscl_kt else k_[1]
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

    # Positive reconstruction @ i+1/2
    i = np.arange(0, nx) + 2
    v1, v2, v3, v4, v5 = (q[i-2], q[i-1], q[i], q[i+1], q[i+2])
    a1, a2, a3, b1, b2, b3 = CRWENO5(v1, v2, v3, v4, v5)

    aa = a1
    bb = a2
    cc = a3
    rr = b1*q[i-1] + b2*q[i] + b3*q[i+1]

    for n in range(q.shape[-1]):
        qL_ip12[i, n] = TDMA_cyclic(aa[:, n], bb[:, n], cc[:, n], rr[:, n],
                                    aa[0, n], cc[-1, n])

    # Negative reconstruction @ i-1/2 + 1 (so i+1/2)
    i += 1
    v1, v2, v3, v4, v5 = (q[i+2], q[i+1], q[i], q[i-1], q[i-2])
    a1, a2, a3, b1, b2, b3 = CRWENO5(v1, v2, v3, v4, v5)

    aa = a3
    bb = a2
    cc = a1
    rr = b1*q[i+1] + b2*q[i] + b3*q[i-1]

    for n in range(q.shape[-1]):
        qR_ip12[i-1, n] = TDMA_cyclic(aa[:, n], bb[:, n], cc[:, n], rr[:, n],
                                      aa[0, n], cc[-1, n])

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


# ============================================
# =========== Viscous Contribution ===========
# ============================================
def c4ddp(f, N, h):
    #
    # 4th order compact scheme 2nd derivative periodic
    #
    # f_ = f minus repeating part (f[0] = f[-1])
    f_ = f[:-1]
    f_pp = np.zeros(f.shape)
    a = np.zeros(f_.shape)
    b = np.zeros(f_.shape)
    c = np.zeros(f_.shape)
    r = np.zeros(f_.shape)

    a[:] = 1 / 10
    b[:] = 1
    c[:] = 1 / 10

    i = np.arange(0, N-2)
    r[i] = (6/5)*(1/h**2)*(f_[i+1] - 2*f_[i] + f_[i-1])

    r[-1] = (6/5)*(1/h**2)*(f_[0] - 2*f_[-1] + f_[-2])

    alpha = 1/10
    beta = 1/10
    if len(f.shape) > 1:
        for n in range(f.shape[-1]):
            f_pp[:-1, n] = TDMA_cyclic(a[:, n], b[:, n], c[:, n], r[:, n], alpha,
                                     beta)
    else:
        f_pp[:-1] = TDMA_cyclic(a, b, c, r, alpha, beta)

    # Verify the cyclic Thomas algorithm is correct
    # f_pp2 = tri_solver_np_cyclic(a[0], b[0], c[0], r, N)
    # assert np.allclose(f_pp, f_pp2)

    f_pp[-1] = f_pp[0]

    return f_pp


# ==========================================
# =========== Problem Quantities ===========
# ==========================================
def get_IC_q(nx, dx, ns):
    q = np.zeros((nx+5, ns))

    # Shock from 0.5 <= x < 0.6
    # q[(int(0.5/dx)+2):(int(0.6/dx)+2), :] = 1

    # Wave numbers
    lx = nx*dx
    kx = np.zeros((nx+5, ns))
    i = np.stack([np.arange(int(nx/2))]*ns, axis=1)
    kx[2:(int(nx/2)+2)] = 2*np.pi*i / lx
    kx[(int(nx/2)+2):(nx+2)] = 2*np.pi*(i - int(nx/2)) / lx

    k0 = 10
    A = (2/(3*np.sqrt(np.pi))) * k0**-5
    E_k = A*kx**4 * np.exp(-(kx/k0)**2)

    E = get_total_KE(E_k[3:(int(nx/2)+1), 0])  # Should be == 0.5

    # Velocity in Fourier space
    # u_k = np.sqrt(2*E_k)
    rand = np.zeros((nx, ns))
    for n in range(ns):
        rand[int(nx/2):, n] = [random.uniform(0, 1) for _ in range(int(nx/2))]
        rand[:int(nx/2), n] = -rand[int(nx/2):, n]
    u_k = np.sqrt(2*E_k[2:(nx+2)]) * np.exp(2j*np.pi*rand)

    # Velocity in physical space
    u = np.fft.ifft(u_k, axis=0, norm="forward")
    u = np.real(np.fft.ifft(u_k, axis=0, norm="forward"))

    # # Periodicity
    # u[nx+2] = u[2]

    q[2:(nx+2)] = u
    q = perbc(q, nx)
    E_k1 = get_KE_k_space(q, nx)
    E1 = get_total_KE(E_k1)

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


# ==========================================
# ================ Plotting ================
# ==========================================
def plot_results(results, output_dir):
    print("\n* Plotting results")
    plt.clf()
    for r in results:
        nx = r["nx"]
        y = r["q_hist"][-1][2:(nx+2), 0]
        l_ = "" if r["limiter"] is None else f", {r['limiter']}"
        label = (f"{r['recon_scheme']}{l_}, nx={r['nx']}, ns={r['ns']}"
                 f", $\\nu$={r['nu']:.1e}")

        plt.plot(np.linspace(0, r["lx"], r["nx"]), y,
                 linewidth=0.4, label=label)

    plt.legend(bbox_to_anchor=(0, -0.15), loc="upper left")
    plt.ylabel("u")
    plt.xlabel("x")
    plt.suptitle(f"Solutions to Burgers Problem at t={r['t']:.4f}s")
    out_dir = "output" if output_dir is None else f"output/{output_dir}"
    Path(out_dir).mkdir(exist_ok=True)
    plt.savefig(f"{out_dir}/Results_u.png", dpi=400, bbox_inches="tight")

    plot_E_hist(results, output_dir)
    print(f"  * Done. Plots saved to '{out_dir}/*.png'")


def plot_E_hist(results, output_dir):
    # E_hist.append({"E": E, "E_k": E_k, "t": time, "dt": dt})
    # Et_hist.append({"Et": Et, "t": time, "dt": dt})
    for E_hist_key, E_key in zip(["E_hist", "Et_hist"], ["E", "Et"]):
        plt.clf()
        for r in results:
            E = [_[E_key] for _ in r[E_hist_key] if not np.isnan(_[E_key])]
            t = [_["t"] for _ in r[E_hist_key] if not np.isnan(_["t"])]
            if len(t) > len(E):
                t = t[:len(E)]
            elif len(E) > len(t):
                E = E[:len(t)]

            l_ = "" if r["limiter"] is None else f", {r['limiter']}"
            label = (f"{r['recon_scheme']}{l_}, nx={r['nx']}, ns={r['ns']}"
                     f", $\\nu$={r['nu']:.1e}")

            plt.plot(t, E, linewidth=0.5, label=label)

            # if E_key == "Et":  # Plot the formula-value of dissipation rate
            #     D = [_["D"] for _ in r[E_hist_key] if not np.isnan(_["D"])]
            #     label = (f"D, {r['recon_scheme']}{l_}, nx={r['nx']}"
            #              f", ns={r['ns']}, $\\nu$={r['nu']:.1e}")
            #     plt.plot(t, D, linewidth=0.5, label=label)

        plt.legend(bbox_to_anchor=(0, -0.15), loc="upper left")
        plt.ylabel(E_key)
        plt.xlabel("t")
        # plt.yscale("log")
        # plt.xscale("log")
        E_var = ("Kinetic Energy"
                 if E_key == "E"
                 else "Kinetic Energy Dissipation Rate")
        plt.suptitle(f"{E_var} vs. Time for Burgers Problem")
        out_dir = "output" if output_dir is None else f"output/{output_dir}"
        Path(out_dir).mkdir(exist_ok=True)
        plt.savefig(f"{out_dir}/Results_{E_key}_vs_t.png", bbox_inches="tight")

    plt.clf()
    for r in results:
        ts = np.array([_["t"] for _ in r["E_hist"] if not np.isnan(_["t"])])
        idt = np.abs(ts - 0.05).argmin()
        E_k0 = r["E_hist"][idt]["E_k"]
        t = r["E_hist"][idt]["t"]
        k = np.arange(len(E_k0))

        l_ = "" if r["limiter"] is None else f", {r['limiter']}"
        label = (f"{r['recon_scheme']}{l_}, nx={r['nx']}, ns={r['ns']}"
                 f", $\\nu$={r['nu']:.1e}")

        plt.plot(k, E_k0, linewidth=0.5, label=label)

    plt.legend(bbox_to_anchor=(0, -0.15), loc="upper left")
    plt.ylabel("E(k)")
    plt.xlabel("k")
    plt.yscale("log")
    plt.xscale("log")
    plt.ylim([1e-11, 1e-1])
    plt.xlim([1e0, 1e4])
    plt.suptitle(f"E(k) vs. k for Burgers Problem at t={t:.4f}s")
    out_dir = "output" if output_dir is None else f"output/{output_dir}"
    Path(out_dir).mkdir(exist_ok=True)
    plt.savefig(f"{out_dir}/Results_Ek_vs_k.png", bbox_inches="tight")


# ============================================================
# ================ Saving DNS Results to disk ================
# ============================================================
def export_DNS_results(DNS_results, DNS_dir):
    DNS_dir.mkdir(parents=True, exist_ok=True)

    np.save(DNS_dir/"q_hist.npy", DNS_results["q_hist"])
    np.save(DNS_dir/"E_hist.npy",
            [_["E"] for _ in DNS_results["E_hist"]])
    np.save(DNS_dir/"Ek_hist.npy",
            [_["E_k"] for _ in DNS_results["E_hist"]])
    np.save(DNS_dir/"Et_hist.npy",
            [_["Et"] for _ in DNS_results["Et_hist"]])
    np.save(DNS_dir/"D_hist.npy",
            [_["D"] for _ in DNS_results["Et_hist"]])
    np.save(DNS_dir/"t_hist.npy",
            [_["t"] for _ in DNS_results["E_hist"]])
    np.save(DNS_dir/"dt_hist.npy",
            [_["dt"] for _ in DNS_results["E_hist"]])

    other_data = {
        "lx": DNS_results["lx"],
        "nx": DNS_results["nx"],
        "ns": DNS_results["ns"],
        "nu": DNS_results["nu"],
        "t": DNS_results["t"],
        "recon_scheme": DNS_results["recon_scheme"] + " (DNS)",
        "limiter": DNS_results["limiter"],
    }

    with open(DNS_dir/"other_data.json", "w") as F:
        json.dump(other_data, F)

    print(f"\nExported {DNS_results['recon_scheme']} results to "
          f"'{DNS_dir}' directory.")


def load_DNS_results(DNS_dir):
    with open(DNS_dir/"other_data.json", "r") as F:
        DNS_results = json.load(F)

    q_hist = np.load(DNS_dir/"q_hist.npy")
    E_hist = np.load(DNS_dir/"E_hist.npy")
    Ek_hist = np.load(DNS_dir/"Ek_hist.npy")
    Et_hist = np.load(DNS_dir/"Et_hist.npy")
    D_hist = np.load(DNS_dir/"D_hist.npy")
    t_hist = np.load(DNS_dir/"t_hist.npy")
    dt_hist = np.load(DNS_dir/"dt_hist.npy")

    DNS_results.update({"E_hist": [
        {"E": E, "E_k": E_k, "t": t, "dt": dt}
        for E, E_k, t, dt
        in zip(E_hist, Ek_hist, t_hist, dt_hist)
    ]})
    DNS_results.update({"Et_hist": [
        {"Et": Et, "D": D, "t": t, "dt": dt}
        for Et, D, t, dt
        in zip(Et_hist, D_hist, t_hist, dt_hist)
    ]})
    DNS_results.update({"q_hist": q_hist})

    print(f"\nLoaded {DNS_results['recon_scheme']} results from "
          f"'{DNS_dir}' directory.")

    return DNS_results


if __name__ == "__main__":
    # Simulation Configurations
    # =========================
    MUSCL_base = {"limiter": None, "nx": 2**9, "ns": 2**5,
                  "nu": 5e-4, "tmax": 0.2}
    MUSCL_Q = MUSCL_base | {"recon_scheme": "MUSCL-Quick"}
    MUSCL_KT = MUSCL_base | {"recon_scheme": "MUSCL-KT"}
    MUSCL_CS = MUSCL_base | {"recon_scheme": "MUSCL-Central"}
    MUSCL_3rd = MUSCL_base | {"recon_scheme": "MUSCL-3rd"}
    MUSCL_Up = MUSCL_base | {"recon_scheme": "MUSCL-Upwind"}
    MUSCL_Fr = MUSCL_base | {"recon_scheme": "MUSCL-Fromm"}

    WENO_3 = {"recon_scheme": "WENO-3", "limiter": None,
              "nu": 5e-4, "tmax": 0.2}
    WENO_5 = WENO_3 | {"recon_scheme": "WENO-5"}
    # =========================

    # DNS
    # ===
    main(cases=[WENO_5 | {"nx": 2**15, "ns": 2**5, "nu": 2e-4}],
         export_as_DNS=True, DNS_nu=2e-4)

    # =========================================================================
    #                            Test Cases Control
    # =========================================================================
    comparisons = {
        "WENO_5_Resolution_Comparison": [
            WENO_5 | {"nx": 2**9, "ns": 2**5},  # nx=512, ns=32
            WENO_5 | {"nx": 2**10, "ns": 2**5},
            WENO_5 | {"nx": 2**11, "ns": 2**5},
            WENO_5 | {"nx": 2**12, "ns": 2**5},
            WENO_5 | {"nx": 2**13, "ns": 2**5},
            WENO_5 | {"nx": 2**14, "ns": 2**5},
            # nx=2**15 is considered our "DNS" results
        ],
        "WENO_3_Viscosity_Comparison": [  # nx=512, ns=32
            WENO_3 | {"nx": 2**9, "ns": 2**5, "nu": 0},
            WENO_3 | {"nx": 2**9, "ns": 2**5, "nu": 2e-4},
            WENO_3 | {"nx": 2**9, "ns": 2**5, "nu": 5e-4},
            WENO_3 | {"nx": 2**9, "ns": 2**5, "nu": 20e-4},
            WENO_3 | {"nx": 2**9, "ns": 2**5, "nu": 100e-4},
        ],
        "WENO_5_Viscosity_Comparison": [  # nx=512, ns=32
            WENO_5 | {"nx": 2**9, "ns": 2**5, "nu": 0},
            WENO_5 | {"nx": 2**9, "ns": 2**5, "nu": 2e-4},
            WENO_5 | {"nx": 2**9, "ns": 2**5, "nu": 5e-4},
            WENO_5 | {"nx": 2**9, "ns": 2**5, "nu": 20e-4},
            WENO_5 | {"nx": 2**9, "ns": 2**5, "nu": 100e-4},
        ],
        "MUSCL_Q_Limiters_Comparison": [  # nx=512, ns=32
            MUSCL_Q | {"limiter": "Van Leer"},
            MUSCL_Q | {"limiter": "Van Albada"},
            MUSCL_Q | {"limiter": "Min-Mod"},
            MUSCL_Q | {"limiter": "Superbee"},
            MUSCL_Q | {"limiter": "Monotonized Central"},
        ],
        "MUSCL_KT_Limiters_Comparison": [  # nx=512, ns=32
            MUSCL_KT | {"limiter": "Van Leer"},
            MUSCL_KT | {"limiter": "Van Albada"},
            MUSCL_KT | {"limiter": "Min-Mod"},
            MUSCL_KT | {"limiter": "Superbee"},
            MUSCL_KT | {"limiter": "Monotonized Central"},
        ],
        "MUSCL_CS_Limiters_Comparison": [  # nx=512, ns=32
            MUSCL_CS | {"limiter": "Van Leer"},
            MUSCL_CS | {"limiter": "Van Albada"},
            MUSCL_CS | {"limiter": "Min-Mod"},
            MUSCL_CS | {"limiter": "Superbee"},
            MUSCL_CS | {"limiter": "Monotonized Central"},
        ],
        "MUSCL_3rd_Limiters_Comparison": [  # nx=512, ns=32
            MUSCL_3rd | {"limiter": "Van Leer"},
            MUSCL_3rd | {"limiter": "Van Albada"},
            MUSCL_3rd | {"limiter": "Min-Mod"},
            MUSCL_3rd | {"limiter": "Superbee"},
            MUSCL_3rd | {"limiter": "Monotonized Central"},
        ],
        "MUSCL_Upwind_Limiters_Comparison": [  # nx=512, ns=32
            MUSCL_Up | {"limiter": "Van Leer"},
            MUSCL_Up | {"limiter": "Van Albada"},
            MUSCL_Up | {"limiter": "Min-Mod"},
            MUSCL_Up | {"limiter": "Superbee"},
            MUSCL_Up | {"limiter": "Monotonized Central"},
        ],
        "MUSCL_Fromm_Limiters_Comparison": [  # nx=512, ns=32
            MUSCL_Fr | {"limiter": "Van Leer"},
            MUSCL_Fr | {"limiter": "Van Albada"},
            MUSCL_Fr | {"limiter": "Min-Mod"},
            MUSCL_Fr | {"limiter": "Superbee"},
            MUSCL_Fr | {"limiter": "Monotonized Central"},
        ],
        "MUSCL_Type_Comparison": [  # nx=512, ns=32
            MUSCL_Q,
            MUSCL_KT,
            MUSCL_CS,
            MUSCL_3rd,
            MUSCL_Up,
            MUSCL_Fr,
        ],
        "MUSCL_Superbee_Type_Comparison": [  # nx=512, ns=32
            MUSCL_Q | {"limiter": "Superbee"},
            MUSCL_KT | {"limiter": "Superbee"},
            MUSCL_CS | {"limiter": "Superbee"},
            MUSCL_3rd | {"limiter": "Superbee"},
            MUSCL_Up | {"limiter": "Superbee"},
            MUSCL_Fr | {"limiter": "Superbee"},
        ],
        "MUSCL_Q_Superbee_Viscosity_Comparison": [  # nx=512, ns=32
            MUSCL_Q | {"limiter": "Superbee", "nu": 0},
            MUSCL_Q | {"limiter": "Superbee", "nu": 2e-4},
            MUSCL_Q | {"limiter": "Superbee", "nu": 5e-4},
            MUSCL_Q | {"limiter": "Superbee", "nu": 20e-4},
            MUSCL_Q | {"limiter": "Superbee", "nu": 100e-4},
        ],
        "Best_MUSCL_vs_WENO_3_5": [  # TODO
            WENO_3 | {"nx": 2**11, "ns": 2**5},
            WENO_5 | {"nx": 2**11, "ns": 2**5},
        ]
    }

    comparisons_to_run = [
        # "WENO_5_Resolution_Comparison",
        "WENO_3_Viscosity_Comparison",
        "WENO_5_Viscosity_Comparison",
        # "MUSCL_Q_Limiters_Comparison",
        # "MUSCL_KT_Limiters_Comparison",
        # "MUSCL_CS_Limiters_Comparison",
        # "MUSCL_3rd_Limiters_Comparison",
        # "MUSCL_Upwind_Limiters_Comparison",
        # "MUSCL_Fromm_Limiters_Comparison",
        # "MUSCL_Type_Comparison",
        # "MUSCL_Superbee_Type_Comparison",
        "MUSCL_Q_Superbee_Viscosity_Comparison",
    ]

    # for comparison_name in comparisons_to_run:
    #     start_time = time.time()

    #     print("\n" + "#"*79 + "\n" +
    #           f"Running Comparison: '{comparison_name}'".center(79) +
    #           "\n" + "#"*79)
    #     main(cases=comparisons[comparison_name], output_dir=comparison_name)

    #     stop_time = time.time()
    #     print(f"\nTotal Execution Time: {stop_time-start_time:.4f} s")
    # =========================================================================
