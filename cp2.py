# References:
# -----------
# [1] Maulik et. al. 2020. https://doi.org/10.1016/j.cam.2020.112866
#
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def main():
    title = "Solving Taylor Green Vortex Problem"
    print(title + "\n" + "="*len(title))

    results = []
    # for nx in [2000, 200]:
    for pwr in [4]:
        lx = ly = lz = 2*np.pi
        nx = ny = nz = 2**pwr
        dx = lx/nx
        dy = ly/ny
        dz = lz/nz

        q = get_IC_q(nx, ny, nz, dx, dy, dz)
        tmax = 10  # s
        time = 0

        while time < tmax:
            dt = get_dt(q, nx, ny, nz, dx, dy, dz)
            time += dt

            q = tvdrk3(nx, ny, nz, dx, dy, dz, dt, q)
            print(f"  * time: {time:.4f} s", end="\n")

        print("\n  * Done")
        results.append({
            "lx": lx,
            "nx": nx,
            "q": q[2:nx+2, 2:ny+2, 2:nz+2, :],
            "t": time,
        })

    # plot_results(results)


def plot_results(results):
    # TODO: Make this more of what we actually want
    ylabel = "u"
    plt.clf()
    for r in results:
        y = r["u"]
        plt.plot(np.linspace(0, r["lx"], r["nx"]), y,
                 linewidth=1, label=f"nx={r['nx']}")

    plt.legend()
    plt.ylabel(ylabel)
    plt.xlabel("x")
    plt.suptitle("Solutions to Taylor Green Vortex Problem at "
                 f"t={r['t']:.4f}s")
    Path("output").mkdir(exist_ok=True)
    plt.savefig(f"output/Results_{ylabel}.png")


def tvdrk3(nx, ny, nz, dx, dy, dz, dt, q_n):
    """
    3rd Order Runge-Kutta time integrator.
    """
    q1 = np.copy(q_n)
    q1[3:(nx+2), 3:(ny+2), 3:(nz+2), :] = (
        q_n[3:(nx+2), 3:(ny+2), 3:(nz+2), :] +
        dt*rhs(nx, ny, nz, dx, dy, dz, q_n)
    )
    print("---")
    print(f"avg(q1[0]) = {np.average(q1[:, :, :, 0])}")
    print(f"avg(q1[1]) = {np.average(q1[:, :, :, 1])}")
    print(f"avg(q1[2]) = {np.average(q1[:, :, :, 2])}")
    print(f"avg(q1[3]) = {np.average(q1[:, :, :, 3])}")
    print(f"avg(q1[4]) = {np.average(q1[:, :, :, 4])}")
    print("---")

    # import pudb; pudb.set_trace()
    q2 = np.copy(q_n)
    q2[3:(nx+2), 3:(ny+2), 3:(nz+2), :] = (
        0.75*q_n[3:(nx+2), 3:(ny+2), 3:(nz+2), :] +
        0.25*q1[3:(nx+2), 3:(ny+2), 3:(nz+2), :] +
        0.25*dt*rhs(nx, ny, nz, dx, dy, dz, q1)
    )
    print(f"avg(q2[0]) = {np.average(q2[:, :, :, 0])}")
    print(f"avg(q2[1]) = {np.average(q2[:, :, :, 1])}")
    print(f"avg(q2[2]) = {np.average(q2[:, :, :, 2])}")
    print(f"avg(q2[3]) = {np.average(q2[:, :, :, 3])}")
    print(f"avg(q2[4]) = {np.average(q2[:, :, :, 4])}")
    print("---")

    q_np1 = np.copy(q_n)
    q_np1[3:(nx+2), 3:(ny+2), 3:(nz+2), :] = (
        (1/3)*q_n[3:(nx+2), 3:(ny+2), 3:(nz+2), :] +
        (2/3)*q2[3:(nx+2), 3:(ny+2), 3:(nz+2), :] +
        (2/3)*dt*rhs(nx, ny, nz, dx, dy, dz, q2)
    )
    print(f"avg(q_np1[0]) = {np.average(q_np1[:, :, :, 0])}")
    print(f"avg(q_np1[1]) = {np.average(q_np1[:, :, :, 1])}")
    print(f"avg(q_np1[2]) = {np.average(q_np1[:, :, :, 2])}")
    print(f"avg(q_np1[3]) = {np.average(q_np1[:, :, :, 3])}")
    print(f"avg(q_np1[4]) = {np.average(q_np1[:, :, :, 4])}")
    print("---")

    return q_np1


def rhs(nx, ny, nz, dx, dy, dz, q):
    """
    Note::
        len(q) == nx+5
                  ^^^^ Specifically from 0:(nx+5)
        len(rhs) == nx-1
                    ^^^^ Specifically from 3:(nx+2)
    """
    # 1st order non-reflective (transmissive) boundary conditions
    n = [nx, ny, nz]
    for axis in [0, 1, 2]:
        q = q.swapaxes(0, axis)
        q[2] = q[3]
        q[1] = q[4]
        q[0] = q[5]
        q[n[axis]+2] = q[n[axis]+1]
        q[n[axis]+3] = q[n[axis]]
        q[n[axis]+4] = q[n[axis]-1]
        q = q.swapaxes(0, axis)

    # qL_ip12.shape = qR_ip12.shape = ... = (nx, ny, nz, 5)
    qL_ip12, qR_ip12 = weno_5(q, axis=0)
    qL_jp12, qR_jp12 = weno_5(q, axis=1)
    qL_kp12, qR_kp12 = weno_5(q, axis=2)
    # qL_ip12, qR_ip12 = weno_5z(q)

    # c_ip12.shape = c_jp12.shape = c_kp12 = (nx, ny, nz)
    c_ip12, c_jp12, c_kp12 = get_wave_speeds(q, nx, ny, nz)
    # Make these the same shape as the 3D vectors
    # (:, :, :) -> (:, :, :, 5)
    c_ip12 = np.stack([c_ip12]*5, axis=3)
    c_jp12 = np.stack([c_jp12]*5, axis=3)
    c_kp12 = np.stack([c_kp12]*5, axis=3)

    # Rusanov's Riemann Solver
    # ========================
    F, G, H = get_inviscid_fluxes(q)
    # F.shape = G.shape = H.shape = (nx+5, ny+5, nz+5, 5)
    # * The range 2:(n+2) excludes ghost points
    FR = F[2:(nx+2), 2:(ny+2), 2:(nz+2), :]*qR_ip12
    FL = F[2:(nx+2), 2:(ny+2), 2:(nz+2), :]*qL_ip12
    F_ip12 = 0.5*((FR + FL) - c_ip12*(qR_ip12 - qL_ip12))
    # F_im12.shape = F_ip12.shape = (nx-1, ny-1, nz-1)
    F_im12 = F_ip12[:-1, :-1, :-1, :]
    F_ip12 = F_ip12[1:, :-1, :-1, :]

    GR = G[2:(nx+2), 2:(ny+2), 2:(nz+2), :]*qR_jp12
    GL = G[2:(nx+2), 2:(ny+2), 2:(nz+2), :]*qL_jp12
    G_jp12 = 0.5*((GR + GL) - c_jp12*(qR_jp12 - qL_jp12))
    G_jm12 = G_jp12[:-1, :-1, :-1, :]
    G_jp12 = G_jp12[:-1, 1:, :-1, :]

    HR = H[2:(nx+2), 2:(ny+2), 2:(nz+2), :]*qR_kp12
    HL = H[2:(nx+2), 2:(ny+2), 2:(nz+2), :]*qL_kp12
    H_kp12 = 0.5*((HR + HL) - c_kp12*(qR_kp12 - qL_kp12))
    H_km12 = H_kp12[:-1, :-1, :-1, :]
    H_kp12 = H_kp12[:-1, :-1, 1:, :]

    # Fv.shape = Gv.shape = Hv.shape = (nx+3, ny+3, nz+3)
    # - Outer ghost points excluded
    Fv, Gv, Hv = get_viscous_fluxes(q, nx, ny, nz, dx, dy, dz)
    # * The range 1:(n+1) excludes ghost points
    FvR = Fv[1:(nx+1), 1:(ny+1), 1:(nz+1), :]*qR_ip12
    FvL = Fv[1:(nx+1), 1:(ny+1), 1:(nz+1), :]*qL_ip12
    Fv_ip12 = 0.5*((FvR + FvL) - c_ip12*(qR_ip12 - qL_ip12))
    # Fv_im12.shape = Fv_ip12.shape = (nx-1, ny-1, nz-1)
    Fv_im12 = Fv_ip12[:-1, :-1, :-1, :]
    Fv_ip12 = Fv_ip12[1:, :-1, :-1, :]

    GvR = Gv[1:(nx+1), 1:(ny+1), 1:(nz+1), :]*qR_jp12
    GvL = Gv[1:(nx+1), 1:(ny+1), 1:(nz+1), :]*qL_jp12
    Gv_jp12 = 0.5*((GvR + GvL) - c_jp12*(qR_jp12 - qL_jp12))
    Gv_jm12 = Gv_jp12[:-1, :-1, :-1, :]
    Gv_jp12 = Gv_jp12[:-1, 1:, :-1, :]

    HvR = Hv[1:(nx+1), 1:(ny+1), 1:(nz+1), :]*qR_kp12
    HvL = Hv[1:(nx+1), 1:(ny+1), 1:(nz+1), :]*qL_kp12
    Hv_kp12 = 0.5*((HvR + HvL) - c_kp12*(qR_kp12 - qL_kp12))
    Hv_km12 = Hv_kp12[:-1, :-1, :-1, :]
    Hv_kp12 = Hv_kp12[:-1, :-1, 1:, :]
    # ========================

    rhs = (
        (Fv_ip12 - Fv_im12) / dx +
        (Gv_jp12 - Gv_jm12) / dy +
        (Hv_kp12 - Hv_km12) / dz -
        (F_ip12 - F_im12) / dx -
        (G_jp12 - G_jm12) / dy -
        (H_kp12 - H_km12) / dz
    )

    return rhs


def weno_5(q, axis=0):
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
    q = q.swapaxes(0, axis)
    n = q.shape[0]
    i = np.arange(2, n-2)

    # Smoothness indicators
    b0 = ((13/12)*(q[i-2] - 2*q[i-1] + q[i])**2
          + (1/4)*(q[i-2] - 4*q[i-1] + 3*q[i])**2)
    b1 = ((13/12)*(q[i-1] - 2*q[i] + q[i+1])**2
          + (1/4)*(q[i-1] - q[i+1])**2)
    b2 = ((13/12)*(q[i] - 2*q[i+1] + q[i+2])**2
          + (1/4)*(3*q[i] - 4*q[i+1] + q[i+2])**2)

    # Linear weighting coefficients
    d0 = 1/10
    d1 = 6/10
    d2 = 3/10

    # Nonliear weights
    eps = 1e-6
    i_ = np.arange(n-5)
    a0 = d0 / (b0[i_] + eps)**2
    a1 = d1 / (b1[i_] + eps)**2
    a2 = d2 / (b2[i_] + eps)**2

    w0 = a0 / (a0 + a1 + a2)
    w1 = a1 / (a0 + a1 + a2)
    w2 = a2 / (a0 + a1 + a2)

    ii = np.arange(2, n-3)
    qL_ip12 = ((w0/6)*(2*q[ii-2] - 7*q[ii-1] + 11*q[ii])
               + (w1/6)*(-q[ii-1] + 5*q[ii] + 2*q[ii+1])
               + (w2/6)*(2*q[ii] + 5*q[ii+1] - q[ii+2]))

    a0 = d2 / (b0[i_+1] + eps)**2
    a1 = d1 / (b1[i_+1] + eps)**2
    a2 = d0 / (b2[i_+1] + eps)**2

    w0 = a0 / (a0 + a1 + a2)
    w1 = a1 / (a0 + a1 + a2)
    w2 = a2 / (a0 + a1 + a2)

    qR_ip12 = ((w0/6)*(-q[ii-1] + 5*q[ii] + 2*q[ii+1])
               + (w1/6)*(2*q[ii] + 5*q[ii+1] - q[ii+2])
               + (w2/6)*(11*q[ii+1] - 7*q[ii+2] + 2*q[ii+3]))

    qL_ip12 = qL_ip12.swapaxes(0, axis)
    qR_ip12 = qR_ip12.swapaxes(0, axis)

    # Trim other axes to match new length (nx/ny/nz)
    for a in [0, 1, 2]:
        if a == axis:
            continue

        qL_ip12 = qL_ip12.swapaxes(0, a)
        qR_ip12 = qR_ip12.swapaxes(0, a)

        n = qL_ip12.shape[0]
        qL_ip12 = qL_ip12[2:(n-3)]
        qR_ip12 = qR_ip12[2:(n-3)]

        qL_ip12 = qL_ip12.swapaxes(0, a)
        qR_ip12 = qR_ip12.swapaxes(0, a)

    return qL_ip12, qR_ip12


def weno_5z(q):
    """
    WENO 5th Order reconstruction scheme (Z Scheme)

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
    n = len(q)
    i = np.arange(2, n-2)

    # Smoothness indicators
    b0 = ((13/12)*(q[i-2, :] - 2*q[i-1, :] + q[i, :])**2
          + (1/4)*(q[i-2, :] - 4*q[i-1, :] + 3*q[i, :])**2)
    b1 = ((13/12)*(q[i-1, :] - 2*q[i, :] + q[i+1, :])**2
          + (1/4)*(q[i-1, :] - q[i+1, :])**2)
    b2 = ((13/12)*(q[i, :] - 2*q[i+1, :] + q[i+2, :])**2
          + (1/4)*(3*q[i, :] - 4*q[i+1, :] + q[i+2, :])**2)

    # Linear weighting coefficients
    d0 = 1/10
    d1 = 6/10
    d2 = 3/10

    # Nonliear weights
    eps = 1e-20
    i_ = np.arange(n-5)
    a0 = d0*(1 + (abs(b0[i_, :] - b2[i_, :])/(b0[i_, :] + eps))**2)
    a1 = d1*(1 + (abs(b0[i_, :] - b2[i_, :])/(b1[i_, :] + eps))**2)
    a2 = d2*(1 + (abs(b0[i_, :] - b2[i_, :])/(b2[i_, :] + eps))**2)

    w0 = a0 / (a0 + a1 + a2)
    w1 = a1 / (a0 + a1 + a2)
    w2 = a2 / (a0 + a1 + a2)

    ii = np.arange(2, n-3)
    qL_ip12 = ((w0/6)*(2*q[ii-2, :] - 7*q[ii-1, :] + 11*q[ii, :])
               + (w1/6)*(-q[ii-1, :] + 5*q[ii, :] + 2*q[ii+1, :])
               + (w2/6)*(2*q[ii, :] + 5*q[ii+1, :] - q[ii+2, :]))

    a0 = d2*(1 + (abs(b0[i_+1, :] - b2[i_+1, :])/(b0[i_+1, :] + eps))**2)
    a1 = d1*(1 + (abs(b0[i_+1, :] - b2[i_+1, :])/(b1[i_+1, :] + eps))**2)
    a2 = d0*(1 + (abs(b0[i_+1, :] - b2[i_+1, :])/(b2[i_+1, :] + eps))**2)

    w0 = a0 / (a0 + a1 + a2)
    w1 = a1 / (a0 + a1 + a2)
    w2 = a2 / (a0 + a1 + a2)

    qR_ip12 = ((w0/6)*(-q[ii-1, :] + 5*q[ii, :] + 2*q[ii+1, :])
               + (w1/6)*(2*q[ii, :] + 5*q[ii+1, :] - q[ii+2, :])
               + (w2/6)*(11*q[ii+1, :] - 7*q[ii+2, :] + 2*q[ii+3, :]))

    return qL_ip12, qR_ip12


# ==========================================
# =========== Problem Quantities ===========
# ==========================================
def constants():
    Re = 1600
    Pr = 0.72
    Ma = 0.08
    gamma = 7/5  # Adiabatic index

    return Re, Pr, Ma, gamma


def get_IC_q(nx, ny, nz, dx, dy, dz):
    lx = nx*dx
    ly = ny*dy
    lz = nz*dz

    X = np.linspace(-2*dx, lx+(3*dx), nx+5)
    Y = np.linspace(-2*dy, ly+(3*dy), ny+5)
    Z = np.linspace(-2*dz, lz+(3*dz), nz+5)
    xx, yy, zz = np.meshgrid(X, Y, Z)
    # Just part of how np.meshgrid works... ¯\_(ツ)_/¯
    xx = xx.swapaxes(0, 1)
    yy = yy.swapaxes(0, 1)

    Re, Pr, Ma, gamma = constants()

    rho0 = 1
    u0 = np.sin(xx)*np.cos(yy)*np.cos(zz)
    v0 = -np.cos(xx)*np.sin(yy)*np.cos(zz)
    w0 = np.zeros((nx+5, ny+5, nz+5))
    p0 = (1/(gamma * Ma**2)) + (
        (np.cos(2*xx) + np.cos(2*yy)) * (np.cos(2*zz) + 2) / 16
    )

    e0 = (p0 / (rho0*(gamma-1))) + 0.5*rho0*(u0**2 + v0**2 + w0**2)

    q = np.zeros((nx+5, ny+5, nz+5, 5))
    q[:, :, :, 0] = rho0
    q[:, :, :, 1] = rho0*u0
    q[:, :, :, 2] = rho0*v0
    q[:, :, :, 3] = rho0*w0
    q[:, :, :, 4] = rho0*e0

    return q


def get_dt(q, nx, ny, nz, dx, dy, dz):
    cfl = 0.5

    c_ip12, c_jp12, c_kp12 = get_wave_speeds(q, nx, ny, nz)
    dtx = np.nanmin(cfl * dx / c_ip12)
    dty = np.nanmin(cfl * dy / c_jp12)
    dtz = np.nanmin(cfl * dz / c_kp12)

    dt = min([dtx, dty, dtz])

    return dt


def get_wave_speeds(q, nx, ny, nz):
    Re, Pr, Ma, gamma = constants()

    rho = q[:, :, :, 0]
    u = q[:, :, :, 1] / rho
    v = q[:, :, :, 2] / rho
    w = q[:, :, :, 3] / rho
    e = q[:, :, :, 4] / rho

    p = rho*(gamma-1)*(e - 0.5*(u**2 + v**2 + w**2))
    a = np.sqrt(gamma*p/rho)

    ii = np.arange(2, nx+2)
    jj = np.arange(2, ny+2)
    kk = np.arange(2, nz+2)

    c_ip12 = np.nanmax(abs(np.stack([
        # u[ii-2, 2:(ny+2), 2:(nz+2)],
        # (u[ii-2, 2:(ny+2), 2:(nz+2)]-a[ii-2, 2:(ny+2), 2:(nz+2)]),
        # (u[ii-2, 2:(ny+2), 2:(nz+2)]+a[ii-2, 2:(ny+2), 2:(nz+2)]),
        # # --
        # u[ii-1, 2:(ny+2), 2:(nz+2)],
        # (u[ii-1, 2:(ny+2), 2:(nz+2)]-a[ii-1, 2:(ny+2), 2:(nz+2)]),
        # (u[ii-1, 2:(ny+2), 2:(nz+2)]+a[ii-1, 2:(ny+2), 2:(nz+2)]),
        # # --
        u[ii, 2:(ny+2), 2:(nz+2)],
        (u[ii, 2:(ny+2), 2:(nz+2)]-a[ii, 2:(ny+2), 2:(nz+2)]),
        (u[ii, 2:(ny+2), 2:(nz+2)]+a[ii, 2:(ny+2), 2:(nz+2)]),
        # --
        u[ii+1, 2:(ny+2), 2:(nz+2)],
        (u[ii+1, 2:(ny+2), 2:(nz+2)]-a[ii+1, 2:(ny+2), 2:(nz+2)]),
        (u[ii+1, 2:(ny+2), 2:(nz+2)]+a[ii+1, 2:(ny+2), 2:(nz+2)]),
        # # --
        # u[ii+2, 2:(ny+2), 2:(nz+2)],
        # (u[ii+2, 2:(ny+2), 2:(nz+2)]-a[ii+2, 2:(ny+2), 2:(nz+2)]),
        # (u[ii+2, 2:(ny+2), 2:(nz+2)]+a[ii+2, 2:(ny+2), 2:(nz+2)]),
        # # --
        # u[ii+3, 2:(ny+2), 2:(nz+2)],
        # (u[ii+3, 2:(ny+2), 2:(nz+2)]-a[ii+3, 2:(ny+2), 2:(nz+2)]),
        # (u[ii+3, 2:(ny+2), 2:(nz+2)]+a[ii+3, 2:(ny+2), 2:(nz+2)]),
    ], axis=0)), axis=0)

    c_jp12 = np.nanmax(abs(np.stack([
        # u[2:(nx+2), jj-2, 2:(nz+2)],
        # (u[2:(nx+2), jj-2, 2:(nz+2)]-a[2:(nx+2), jj-2, 2:(nz+2)]),
        # (u[2:(nx+2), jj-2, 2:(nz+2)]+a[2:(nx+2), jj-2, 2:(nz+2)]),
        # # --
        # u[2:(nx+2), jj-1, 2:(nz+2)],
        # (u[2:(nx+2), jj-1, 2:(nz+2)]-a[2:(nx+2), jj-1, 2:(nz+2)]),
        # (u[2:(nx+2), jj-1, 2:(nz+2)]+a[2:(nx+2), jj-1, 2:(nz+2)]),
        # # --
        u[2:(nx+2), jj, 2:(nz+2)],
        (u[2:(nx+2), jj, 2:(nz+2)]-a[2:(nx+2), jj, 2:(nz+2)]),
        (u[2:(nx+2), jj, 2:(nz+2)]+a[2:(nx+2), jj, 2:(nz+2)]),
        # --
        u[2:(nx+2), jj+1, 2:(nz+2)],
        (u[2:(nx+2), jj+1, 2:(nz+2)]-a[2:(nx+2), jj+1, 2:(nz+2)]),
        (u[2:(nx+2), jj+1, 2:(nz+2)]+a[2:(nx+2), jj+1, 2:(nz+2)]),
        # # --
        # u[2:(nx+2), jj+2, 2:(nz+2)],
        # (u[2:(nx+2), jj+2, 2:(nz+2)]-a[2:(nx+2), jj+2, 2:(nz+2)]),
        # (u[2:(nx+2), jj+2, 2:(nz+2)]+a[2:(nx+2), jj+2, 2:(nz+2)]),
        # # --
        # u[2:(nx+2), jj+3, 2:(nz+2)],
        # (u[2:(nx+2), jj+3, 2:(nz+2)]-a[2:(nx+2), jj+3, 2:(nz+2)]),
        # (u[2:(nx+2), jj+3, 2:(nz+2)]+a[2:(nx+2), jj+3, 2:(nz+2)]),
    ], axis=0)), axis=0)

    c_kp12 = np.nanmax(abs(np.stack([
        # u[2:(nx+2), 2:(ny+2), kk-2],
        # (u[2:(nx+2), 2:(ny+2), kk-2]-a[2:(nx+2), 2:(ny+2), kk-2]),
        # (u[2:(nx+2), 2:(ny+2), kk-2]+a[2:(nx+2), 2:(ny+2), kk-2]),
        # # --
        # u[2:(nx+2), 2:(ny+2), kk-1],
        # (u[2:(nx+2), 2:(ny+2), kk-1]-a[2:(nx+2), 2:(ny+2), kk-1]),
        # (u[2:(nx+2), 2:(ny+2), kk-1]+a[2:(nx+2), 2:(ny+2), kk-1]),
        # # --
        u[2:(nx+2), 2:(ny+2), kk],
        (u[2:(nx+2), 2:(ny+2), kk]-a[2:(nx+2), 2:(ny+2), kk]),
        (u[2:(nx+2), 2:(ny+2), kk]+a[2:(nx+2), 2:(ny+2), kk]),
        # --
        u[2:(nx+2), 2:(ny+2), kk+1],
        (u[2:(nx+2), 2:(ny+2), kk+1]-a[2:(nx+2), 2:(ny+2), kk+1]),
        (u[2:(nx+2), 2:(ny+2), kk+1]+a[2:(nx+2), 2:(ny+2), kk+1]),
        # # --
        # u[2:(nx+2), 2:(ny+2), kk+2],
        # (u[2:(nx+2), 2:(ny+2), kk+2]-a[2:(nx+2), 2:(ny+2), kk+2]),
        # (u[2:(nx+2), 2:(ny+2), kk+2]+a[2:(nx+2), 2:(ny+2), kk+2]),
        # # --
        # u[2:(nx+2), 2:(ny+2), kk+3],
        # (u[2:(nx+2), 2:(ny+2), kk+3]-a[2:(nx+2), 2:(ny+2), kk+3]),
        # (u[2:(nx+2), 2:(ny+2), kk+3]+a[2:(nx+2), 2:(ny+2), kk+3]),
    ], axis=0)), axis=0)

    return c_ip12, c_jp12, c_kp12


def get_inviscid_fluxes(q):
    """
    Returns:
        tuple: Tuple containing F, G, and H fluxes, corresponding to fluxes in
            the x, y and z directions respectively.
    """
    Re, Pr, Ma, gamma = constants()

    rho = q[:, :, :, 0]
    u = q[:, :, :, 1] / rho
    v = q[:, :, :, 2] / rho
    w = q[:, :, :, 3] / rho
    e = q[:, :, :, 4] / rho

    p = rho*(gamma-1)*(e - 0.5*(u**2 + v**2 + w**2))
    h = e + p/rho

    F = np.zeros(q.shape)
    F[:, :, :, 0] = rho*u
    F[:, :, :, 1] = rho*u**2 + p
    F[:, :, :, 2] = rho*u*v
    F[:, :, :, 3] = rho*u*w
    F[:, :, :, 4] = rho*u*h

    G = np.zeros(q.shape)
    G[:, :, :, 0] = rho*v
    G[:, :, :, 1] = rho*v*u
    G[:, :, :, 2] = rho*v**2 + p
    G[:, :, :, 3] = rho*v*w
    G[:, :, :, 4] = rho*v*h

    H = np.zeros(q.shape)
    H[:, :, :, 0] = rho*w
    H[:, :, :, 1] = rho*w*u
    H[:, :, :, 2] = rho*w*v
    H[:, :, :, 3] = rho*w**2 + p
    H[:, :, :, 4] = rho*w*h

    return F, G, H


def get_viscous_fluxes(q, nx, ny, nz, dx, dy, dz):
    """
    [1]

    Returns:
        tuple: Tuple containing F, G, and H fluxes, corresponding to fluxes in
            the x, y and z directions respectively.
    """
    Re, Pr, Ma, gamma = constants()

    # Maybe replace zeros in rho with 1e-6?
    rho = q[:, :, :, 0]
    u = q[:, :, :, 1] / rho
    v = q[:, :, :, 2] / rho
    w = q[:, :, :, 3] / rho
    e = q[:, :, :, 4] / rho

    p = rho*(gamma-1)*(e - 0.5*(u**2 + v**2 + w**2))
    T = Ma**2 * gamma * p / rho

    # Dynamic viscosity
    T_ref = 300  # K
    S = 110.4  # K
    mu = T**1.5 * (1 + S/T_ref) / (T + S/T_ref)

    # Shear stresses, using 1st order central derivatives
    ii = np.arange(1, nx+4)
    jj = np.arange(1, ny+4)
    kk = np.arange(1, nz+4)

    # Make this the right shape for calculations
    mu = mu[1:-1, 1:-1, 1:-1]

    # You would think you could do:
    #     u[ii+1, jj, kk]
    # but that gives an array of shape (nx+3,) rather than (nx+3, ny+3, nz+3)
    duk_dxk_term = (2/3)*(
        (u[ii+1, 1:-1, 1:-1] - u[ii-1, 1:-1, 1:-1]) / (2*dx) +
        (v[1:-1, jj+1, 1:-1] - v[1:-1, jj-1, 1:-1]) / (2*dy) +
        (w[1:-1, 1:-1, kk+1] - w[1:-1, 1:-1, kk-1]) / (2*dz)
    )
    t_xx = (mu/Re) * (
        (u[ii+1, 1:-1, 1:-1] - u[ii-1, 1:-1, 1:-1]) / (2*dx) +
        (u[ii+1, 1:-1, 1:-1] - u[ii-1, 1:-1, 1:-1]) / (2*dx) - duk_dxk_term
    )
    t_yy = (mu/Re) * (
        (v[1:-1, jj+1, 1:-1] - v[1:-1, jj-1, 1:-1]) / (2*dy) +
        (v[1:-1, jj+1, 1:-1] - v[1:-1, jj-1, 1:-1]) / (2*dy) - duk_dxk_term
    )
    t_zz = (mu/Re) * (
        (w[1:-1, 1:-1, kk+1] - w[1:-1, 1:-1, kk-1]) / (2*dz) +
        (w[1:-1, 1:-1, kk+1] - w[1:-1, 1:-1, kk-1]) / (2*dz) - duk_dxk_term
    )

    t_xy = (mu/Re) * (
        (u[1:-1, jj+1, 1:-1] - u[1:-1, jj-1, 1:-1]) / (2*dy) +
        (v[ii+1, 1:-1, 1:-1] - v[ii-1, 1:-1, 1:-1]) / (2*dx)  # - duk_dxk_term
        #                                                       ^^^^^^^^^^^^^^
        #                          Kronecker delta cancels this because i != j
    )
    t_xz = (mu/Re) * (
        (u[1:-1, 1:-1, kk+1] - u[1:-1, 1:-1, kk-1]) / (2*dz) +
        (w[ii+1, 1:-1, 1:-1] - w[ii-1, 1:-1, 1:-1]) / (2*dx)  # - duk_dxk_term
    )
    t_yz = (mu/Re) * (
        (v[1:-1, 1:-1, kk+1] - v[1:-1, 1:-1, kk-1]) / (2*dz) +
        (w[1:-1, jj+1, 1:-1] - w[1:-1, jj-1, 1:-1]) / (2*dy)  # - duk_dxk_term
    )

    # Heat fluxes, using 1st order central derivative
    q_x = -mu / (Re*Pr*Ma**2 * (gamma-1)) * (
        (T[ii+1, 1:-1, 1:-1] - T[ii-1, 1:-1, 1:-1]) / (2*dx)
    )
    q_y = -mu / (Re*Pr*Ma**2 * (gamma-1)) * (
        (T[1:-1, jj+1, 1:-1] - T[1:-1, jj-1, 1:-1]) / (2*dy)
    )
    q_z = -mu / (Re*Pr*Ma**2 * (gamma-1)) * (
        (T[1:-1, 1:-1, kk+1] - T[1:-1, 1:-1, kk-1]) / (2*dz)
    )

    # Make these the right shape for calculations
    u = u[1:-1, 1:-1, 1:-1]
    v = v[1:-1, 1:-1, 1:-1]
    w = w[1:-1, 1:-1, 1:-1]
    T = T[1:-1, 1:-1, 1:-1]

    Fv = np.zeros(q[1:-1, 1:-1, 1:-1, :].shape)
    Fv[:, :, :, 0] = 0
    Fv[:, :, :, 1] = t_xx
    Fv[:, :, :, 2] = t_xy
    Fv[:, :, :, 3] = t_xz
    Fv[:, :, :, 4] = u*t_xx + v*t_xy + w*t_xz - q_x

    Gv = np.zeros(q[1:-1, 1:-1, 1:-1, :].shape)
    Gv[:, :, :, 0] = 0
    Gv[:, :, :, 1] = t_xy
    Gv[:, :, :, 2] = t_yy
    Gv[:, :, :, 3] = t_yz
    Gv[:, :, :, 4] = u*t_xy + v*t_yy + w*t_yz - q_y

    Hv = np.zeros(q[1:-1, 1:-1, 1:-1, :].shape)
    Hv[:, :, :, 0] = 0
    Hv[:, :, :, 1] = t_xz
    Hv[:, :, :, 2] = t_yz
    Hv[:, :, :, 3] = t_zz
    Hv[:, :, :, 4] = u*t_xz + v*t_yz + w*t_zz - q_z

    return Fv, Gv, Hv


if __name__ == "__main__":
    main()
