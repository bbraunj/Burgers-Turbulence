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
    print("---")
    print(f"min(q_n[0]) = {np.min(q_n[:, :, :, 0])}")
    print(f"min(q_n[1]) = {np.min(q_n[:, :, :, 1])}")
    print(f"min(q_n[2]) = {np.min(q_n[:, :, :, 2])}")
    print(f"min(q_n[3]) = {np.min(q_n[:, :, :, 3])}")
    print(f"min(q_n[4]) = {np.min(q_n[:, :, :, 4])}")
    print("---")
    q1 = np.copy(q_n)
    q1[3:(nx+2), 3:(ny+2), 3:(nz+2), :] = (
        q_n[3:(nx+2), 3:(ny+2), 3:(nz+2), :] +
        dt*rhs(nx, ny, nz, dx, dy, dz, q_n)
    )
    q1 = perbc_xyz(q1, nx, ny, nz)
    print(f"min(q1[0]) = {np.min(q1[:, :, :, 0])}")
    print(f"min(q1[1]) = {np.min(q1[:, :, :, 1])}")
    print(f"min(q1[2]) = {np.min(q1[:, :, :, 2])}")
    print(f"min(q1[3]) = {np.min(q1[:, :, :, 3])}")
    print(f"min(q1[4]) = {np.min(q1[:, :, :, 4])}")
    print("---")

    q2 = np.copy(q1)
    q2[3:(nx+2), 3:(ny+2), 3:(nz+2), :] = (
        0.75*q_n[3:(nx+2), 3:(ny+2), 3:(nz+2), :] +
        0.25*q1[3:(nx+2), 3:(ny+2), 3:(nz+2), :] +
        0.25*dt*rhs(nx, ny, nz, dx, dy, dz, q1)
    )
    q2 = perbc_xyz(q2, nx, ny, nz)
    print(f"min(q2[0]) = {np.min(q2[:, :, :, 0])}")
    print(f"min(q2[1]) = {np.min(q2[:, :, :, 1])}")
    print(f"min(q2[2]) = {np.min(q2[:, :, :, 2])}")
    print(f"min(q2[3]) = {np.min(q2[:, :, :, 3])}")
    print(f"min(q2[4]) = {np.min(q2[:, :, :, 4])}")
    print("---")

    q_np1 = np.copy(q2)
    q_np1[3:(nx+2), 3:(ny+2), 3:(nz+2), :] = (
        (1/3)*q_n[3:(nx+2), 3:(ny+2), 3:(nz+2), :] +
        (2/3)*q2[3:(nx+2), 3:(ny+2), 3:(nz+2), :] +
        (2/3)*dt*rhs(nx, ny, nz, dx, dy, dz, q2)
    )
    q_np1 = perbc_xyz(q_np1, nx, ny, nz)
    print(f"min(q_np1[0]) = {np.min(q_np1[:, :, :, 0])}")
    print(f"min(q_np1[1]) = {np.min(q_np1[:, :, :, 1])}")
    print(f"min(q_np1[2]) = {np.min(q_np1[:, :, :, 2])}")
    print(f"min(q_np1[3]) = {np.min(q_np1[:, :, :, 3])}")
    print(f"min(q_np1[4]) = {np.min(q_np1[:, :, :, 4])}")
    print("---")

    return q_np1


def rhs(nx, ny, nz, dx, dy, dz, q):
    """
    Note::
        q.shape == (nx+5, ny+5, nz+5)
                    ^^^^^^^^^^^^^^^^
                    Specifically from (0:(nx+5), 0:(ny+5), 0:(nz+5))
        rhs.shape == (nx-1, ny-1, nz-1)
                      ^^^^^^^^^^^^^^^^
                      Specifically from (3:(nx+2), 3:(ny+2), 3:(nz+2))
    """
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
    FR = F[2:(nx+2), 3:(ny+2), 3:(nz+2), :]*qR_ip12
    FL = F[2:(nx+2), 3:(ny+2), 3:(nz+2), :]*qL_ip12
    F_ip12 = 0.5*((FR + FL) - c_ip12*(qR_ip12 - qL_ip12))
    F_im12 = F_ip12[:-1, :, :, :]
    F_ip12 = F_ip12[1:, :, :, :]
    # F_im12.shape = F_ip12.shape = (nx-1, ny-1, nz-1)

    GR = G[3:(nx+2), 2:(ny+2), 3:(nz+2), :]*qR_jp12
    GL = G[3:(nx+2), 2:(ny+2), 3:(nz+2), :]*qL_jp12
    G_jp12 = 0.5*((GR + GL) - c_jp12*(qR_jp12 - qL_jp12))
    G_jm12 = G_jp12[:, :-1, :, :]
    G_jp12 = G_jp12[:, 1:, :, :]

    HR = H[3:(nx+2), 3:(ny+2), 2:(nz+2), :]*qR_kp12
    HL = H[3:(nx+2), 3:(ny+2), 2:(nz+2), :]*qL_kp12
    H_kp12 = 0.5*((HR + HL) - c_kp12*(qR_kp12 - qL_kp12))
    H_km12 = H_kp12[:, :, :-1, :]
    H_kp12 = H_kp12[:, :, 1:, :]
    # ========================

    # Fv.shape = Gv.shape = Hv.shape = (nx+3, ny+3, nz+3)
    # - Outer ghost points excluded
    Fv_ip12, Gv_jp12, Hv_kp12 = get_viscous_fluxes(q, nx, ny, nz, dx, dy, dz)

    # Fv_ip12.shape = (nx, ny-1, nz-1)
    Fv_im12 = Fv_ip12[:-1, :, :, :]
    Fv_ip12 = Fv_ip12[1:, :, :, :]
    # Fv_ip12.shape = (nx-1, ny-1, nz-1)

    # Gv_jp12.shape = (nx-1, ny, nz-1)
    Gv_jm12 = Gv_jp12[:, :-1, :, :]
    Gv_jp12 = Gv_jp12[:, 1:, :, :]
    # Gv_jp12.shape = (nx-1, ny-1, nz-1)

    # Hv_kp12.shape = (nx-1, ny-1, nz)
    Hv_km12 = Hv_kp12[:, :, :-1, :]
    Hv_kp12 = Hv_kp12[:, :, 1:, :]
    # Hv_kp12.shape = (nx-1, ny-1, nz-1)

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

    # Trim other axes to (1:nx, 1:ny, 1:nz)
    for a in [0, 1, 2]:
        if a == axis:
            continue

        qL_ip12 = qL_ip12.swapaxes(0, a)
        qR_ip12 = qR_ip12.swapaxes(0, a)

        n = qL_ip12.shape[0]
        qL_ip12 = qL_ip12[3:(n-3)]
        qR_ip12 = qR_ip12[3:(n-3)]

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


def perbc_xyz(q, nx, ny, nz):
    """
    Periodic boundary conditions in x/y/z directions

    Note::
        This does not modify the ghost points in the corners (#):

            # # * * * ... * # # #
            # # * * * ... * # # #
            * * * * * ... * * * *
            * * * * * ... * * * *
            ...
            * * * * * ... * * * *
            # # * * * ... * # # #
            # # * * * ... * # # #
            # # * * * ... * # # #
    """

    # Front of domain
    q[3:(nx+2), 2, 3:(nz+2), :] = q[3:(nx+2), ny+1, 3:(nz+2), :]  # j=ny
    q[3:(nx+2), 1, 3:(nz+2), :] = q[3:(nx+2), ny, 3:(nz+2), :]    # j=ny-1
    q[3:(nx+2), 0, 3:(nz+2), :] = q[3:(nx+2), ny-1, 3:(nz+2), :]  # j=ny-2

    # Back of domain
    q[3:(nx+2), ny+2, 3:(nz+2), :] = q[3:(nx+2), 3, 3:(nz+2), :]  # j=0
    q[3:(nx+2), ny+3, 3:(nz+2), :] = q[3:(nx+2), 4, 3:(nz+2), :]  # j=1
    q[3:(nx+2), ny+4, 3:(nz+2), :] = q[3:(nx+2), 5, 3:(nz+2), :]  # j=2
    #           ^^^^-- Last ghost point

    # Left of domain (x)
    q[2, 3:(ny+2), 3:(nz+2), :] = q[nx+1, 3:(ny+2), 3:(nz+2), :]
    q[1, 3:(ny+2), 3:(nz+2), :] = q[nx, 3:(ny+2), 3:(nz+2), :]
    q[0, 3:(ny+2), 3:(nz+2), :] = q[nx-1, 3:(ny+2), 3:(nz+2), :]

    # Right of domain (x)
    q[nx+2, 3:(ny+2), 3:(nz+2), :] = q[3, 3:(ny+2), 3:(nz+2), :]
    q[nx+3, 3:(ny+2), 3:(nz+2), :] = q[4, 3:(ny+2), 3:(nz+2), :]
    q[nx+4, 3:(ny+2), 3:(nz+2), :] = q[5, 3:(ny+2), 3:(nz+2), :]

    # Left of domain (z)
    q[3:(nx+2), 3:(ny+2), 2, :] = q[3:(nx+2), 3:(ny+2), nz+1, :]
    q[3:(nx+2), 3:(ny+2), 1, :] = q[3:(nx+2), 3:(ny+2), nz, :]
    q[3:(nx+2), 3:(ny+2), 0, :] = q[3:(nx+2), 3:(ny+2), nz-1, :]

    # Right of domain (z)
    q[3:(nx+2), 3:(nz+2), nz+2, :] = q[3:(nx+2), 3:(ny+2), 3, :]
    q[3:(nx+2), 3:(nz+2), nz+3, :] = q[3:(nx+2), 3:(ny+2), 4, :]
    q[3:(nx+2), 3:(nz+2), nz+4, :] = q[3:(nx+2), 3:(ny+2), 5, :]

    return q


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
    a = np.sqrt(gamma*p[2:(nx+3), 2:(ny+3), 2:(nz+3)] /
                rho[2:(nx+3), 2:(ny+3), 2:(nz+3)])

    ii = np.arange(2, nx+2)
    c_ip12 = np.nanmax(abs(np.stack([
        # u[ii-2, 2:(ny+2), 2:(nz+2)],
        # (u[ii-2, 2:(ny+2), 2:(nz+2)]-a[ii-2, 2:(ny+2), 2:(nz+2)]),
        # (u[ii-2, 2:(ny+2), 2:(nz+2)]+a[ii-2, 2:(ny+2), 2:(nz+2)]),
        # # --
        # u[ii-1, 2:(ny+2), 2:(nz+2)],
        # (u[ii-1, 2:(ny+2), 2:(nz+2)]-a[ii-1, 2:(ny+2), 2:(nz+2)]),
        # (u[ii-1, 2:(ny+2), 2:(nz+2)]+a[ii-1, 2:(ny+2), 2:(nz+2)]),
        # # --
        u[ii, 3:(ny+2), 3:(nz+2)],
        (u[ii, 3:(ny+2), 3:(nz+2)]-a[:-1, 1:-1, 1:-1]),
        (u[ii, 3:(ny+2), 3:(nz+2)]+a[:-1, 1:-1, 1:-1]),
        # --
        u[ii+1, 3:(ny+2), 3:(nz+2)],
        (u[ii+1, 3:(ny+2), 3:(nz+2)]-a[1:, 1:-1, 1:-1]),
        (u[ii+1, 3:(ny+2), 3:(nz+2)]+a[1:, 1:-1, 1:-1]),
        # # --
        # u[ii+2, 2:(ny+2), 2:(nz+2)],
        # (u[ii+2, 2:(ny+2), 2:(nz+2)]-a[ii+2, 2:(ny+2), 2:(nz+2)]),
        # (u[ii+2, 2:(ny+2), 2:(nz+2)]+a[ii+2, 2:(ny+2), 2:(nz+2)]),
        # # --
        # u[ii+3, 2:(ny+2), 2:(nz+2)],
        # (u[ii+3, 2:(ny+2), 2:(nz+2)]-a[ii+3, 2:(ny+2), 2:(nz+2)]),
        # (u[ii+3, 2:(ny+2), 2:(nz+2)]+a[ii+3, 2:(ny+2), 2:(nz+2)]),
    ], axis=0)), axis=0)

    jj = np.arange(2, ny+2)
    c_jp12 = np.nanmax(abs(np.stack([
        # u[2:(nx+2), jj-2, 2:(nz+2)],
        # (u[2:(nx+2), jj-2, 2:(nz+2)]-a[2:(nx+2), jj-2, 2:(nz+2)]),
        # (u[2:(nx+2), jj-2, 2:(nz+2)]+a[2:(nx+2), jj-2, 2:(nz+2)]),
        # # --
        # u[2:(nx+2), jj-1, 2:(nz+2)],
        # (u[2:(nx+2), jj-1, 2:(nz+2)]-a[2:(nx+2), jj-1, 2:(nz+2)]),
        # (u[2:(nx+2), jj-1, 2:(nz+2)]+a[2:(nx+2), jj-1, 2:(nz+2)]),
        # # --
        u[3:(nx+2), jj, 3:(nz+2)],
        (u[3:(nx+2), jj, 3:(nz+2)]-a[1:-1, :-1, 1:-1]),
        (u[3:(nx+2), jj, 3:(nz+2)]+a[1:-1, :-1, 1:-1]),
        # --
        u[3:(nx+2), jj+1, 3:(nz+2)],
        (u[3:(nx+2), jj+1, 3:(nz+2)]-a[1:-1, 1:, 1:-1]),
        (u[3:(nx+2), jj+1, 3:(nz+2)]+a[1:-1, 1:, 1:-1]),
        # # --
        # u[2:(nx+2), jj+2, 2:(nz+2)],
        # (u[2:(nx+2), jj+2, 2:(nz+2)]-a[2:(nx+2), jj+2, 2:(nz+2)]),
        # (u[2:(nx+2), jj+2, 2:(nz+2)]+a[2:(nx+2), jj+2, 2:(nz+2)]),
        # # --
        # u[2:(nx+2), jj+3, 2:(nz+2)],
        # (u[2:(nx+2), jj+3, 2:(nz+2)]-a[2:(nx+2), jj+3, 2:(nz+2)]),
        # (u[2:(nx+2), jj+3, 2:(nz+2)]+a[2:(nx+2), jj+3, 2:(nz+2)]),
    ], axis=0)), axis=0)

    kk = np.arange(2, nz+2)
    c_kp12 = np.nanmax(abs(np.stack([
        # u[2:(nx+2), 2:(ny+2), kk-2],
        # (u[2:(nx+2), 2:(ny+2), kk-2]-a[2:(nx+2), 2:(ny+2), kk-2]),
        # (u[2:(nx+2), 2:(ny+2), kk-2]+a[2:(nx+2), 2:(ny+2), kk-2]),
        # # --
        # u[2:(nx+2), 2:(ny+2), kk-1],
        # (u[2:(nx+2), 2:(ny+2), kk-1]-a[2:(nx+2), 2:(ny+2), kk-1]),
        # (u[2:(nx+2), 2:(ny+2), kk-1]+a[2:(nx+2), 2:(ny+2), kk-1]),
        # # --
        u[3:(nx+2), 3:(ny+2), kk],
        (u[3:(nx+2), 3:(ny+2), kk]-a[1:-1, 1:-1, :-1]),
        (u[3:(nx+2), 3:(ny+2), kk]+a[1:-1, 1:-1, :-1]),
        # --
        u[3:(nx+2), 3:(ny+2), kk+1],
        (u[3:(nx+2), 3:(ny+2), kk+1]-a[1:-1, 1:-1, 1:]),
        (u[3:(nx+2), 3:(ny+2), kk+1]+a[1:-1, 1:-1, 1:]),
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
        tuple: Tuple containing F, G, and H, corresponding to inviscid fluxes
            in the x, y and z directions respectively, where quantities lie in
            the cell center.
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
    Returns:
        tuple: Tuple containing Fv_ip12, Gv_jp12, and Hv_kp12,
            corresponding to the viscous fluxes in the x, y and z directions
            respectively, where quantities lie on the cell boundaries in their
            respective directions.
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

    #
    # 6th-Order Finite Volume derivatives, interpolated to lie on cell
    # boundaries. These are used to find Fv_ip12, Gv_jp12, and Hv_kp12, the
    # fluxes in X, Y, and Z directions respectively, also lying on the cell
    # boundaries.
    #

    # X Direction
    # -----------
    # t_xx -> ux, vy, wz
    # t_xy -> uy, vx
    # t_xz -> uz, wx
    # q_x -> Tx
    # --> Derivatives to find: ux, uy, uz, vx, vy, wx, wz, Tx

    # Cross derivatives for y/z derivatives
    jj = np.arange(3, ny+2)
    cd_uy = 0.1*(
        (15/(2*dy))*(u[:, jj+1, 3:(nz+2)] - u[:, jj-1, 3:(nz+2)]) +
        (-6/(4*dy))*(u[:, jj+2, 3:(nz+2)] - u[:, jj-2, 3:(nz+2)]) +
        (1/(6*dy))*(u[:, jj+3, 3:(nz+2)] - u[:, jj-3, 3:(nz+2)])
    )
    cd_vy = 0.1*(
        (15/(2*dy))*(v[:, jj+1, 3:(nz+2)] - v[:, jj-1, 3:(nz+2)]) +
        (-6/(4*dy))*(v[:, jj+2, 3:(nz+2)] - v[:, jj-2, 3:(nz+2)]) +
        (1/(6*dy))*(v[:, jj+3, 3:(nz+2)] - v[:, jj-3, 3:(nz+2)])
    )

    kk = np.arange(3, nz+2)
    cd_uz = 0.1*(
        (15/(2*dz))*(u[:, 3:(ny+2), kk+1] - u[:, 3:(ny+2), kk-1]) +
        (-6/(4*dz))*(u[:, 3:(ny+2), kk+2] - u[:, 3:(ny+2), kk-2]) +
        (1/(6*dz))*(u[:, 3:(ny+2), kk+3] - u[:, 3:(ny+2), kk-3])
    )
    cd_wz = 0.1*(
        (15/(2*dz))*(w[:, 3:(ny+2), kk+1] - w[:, 3:(ny+2), kk-1]) +
        (-6/(4*dz))*(w[:, 3:(ny+2), kk+2] - w[:, 3:(ny+2), kk-2]) +
        (1/(6*dz))*(w[:, 3:(ny+2), kk+3] - w[:, 3:(ny+2), kk-3])
    )

    # Fluxes @ cell interfaces (i+1/2)
    def interpolate_to_ip12_interface(q_i, nx, ny, nz):
        ii = np.arange(2, nx+2)

        if q_i.shape[1] == ny+5 and q_i.shape[2] == nz+5:
            q_ip12 = (1/60)*(
                37*(q_i[ii+1, 3:(ny+2), 3:(nz+2)] +
                    q_i[ii, 3:(ny+2), 3:(nz+2)])
                - 8*(q_i[ii+2, 3:(ny+2), 3:(nz+2)] +
                     q_i[ii-1, 3:(ny+2), 3:(nz+2)])
                + (q_i[ii+3, 3:(ny+2), 3:(nz+2)] +
                   q_i[ii-2, 3:(ny+2), 3:(nz+2)])
            )
        else:  # Assume it's a cross-derivative, shape (nx+5, ny-1, nz-1)
            q_ip12 = (1/60)*(
                37*(q_i[ii+1, :, :] + q_i[ii, :, :])
                - 8*(q_i[ii+2, :, :] + q_i[ii-1, :, :])
                + (q_i[ii+3, :, :] + q_i[ii-2, :, :])
            )

        return q_ip12

    uy_ip12 = interpolate_to_ip12_interface(cd_uy, nx, ny, nz)
    vy_ip12 = interpolate_to_ip12_interface(cd_vy, nx, ny, nz)
    uz_ip12 = interpolate_to_ip12_interface(cd_uz, nx, ny, nz)
    wz_ip12 = interpolate_to_ip12_interface(cd_wz, nx, ny, nz)

    # x derivatives @ cell interfaces
    def ddx_ip12(q_i, nx, ny, nz):
        ii = np.arange(2, nx+2)

        qx_ip12 = (1/180)*(
            (245/dx)*(q_i[ii+1, 3:(ny+2), 3:(nz+2)] -
                      q_i[ii, 3:(ny+2), 3:(nz+2)])
            + (-75/(3*dx))*(q_i[ii+2, 3:(ny+2), 3:(nz+2)] -
                            q_i[ii-1, 3:(ny+2), 3:(nz+2)])
            + (10/(5*dx))*(q_i[ii+3, 3:(ny+2), 3:(nz+2)] -
                           q_i[ii-2, 3:(ny+2), 3:(nz+2)])
        )

        return qx_ip12

    ux_ip12 = ddx_ip12(u, nx, ny, nz)
    vx_ip12 = ddx_ip12(v, nx, ny, nz)
    wx_ip12 = ddx_ip12(w, nx, ny, nz)
    Tx_ip12 = ddx_ip12(T, nx, ny, nz)

    # Other values interpolated to cell interfaces (i+1/2)
    mu_ip12 = interpolate_to_ip12_interface(mu, nx, ny, nz)
    u_ip12 = interpolate_to_ip12_interface(u, nx, ny, nz)
    v_ip12 = interpolate_to_ip12_interface(v, nx, ny, nz)
    w_ip12 = interpolate_to_ip12_interface(w, nx, ny, nz)

    t_xx_ip12 = (mu_ip12/Re)*(2*ux_ip12 - (2/3)*(ux_ip12 + vy_ip12 + wz_ip12))
    t_xy_ip12 = (mu_ip12/Re)*(uy_ip12 + vx_ip12)
    t_xz_ip12 = (mu_ip12/Re)*(uz_ip12 + wx_ip12)
    q_x_ip12 = -(mu_ip12 / (Re*Pr*Ma**2 * (gamma-1))) * Tx_ip12

    # Flux in X-dir
    Fv_ip12 = np.zeros((nx, ny-1, nz-1, 5))
    Fv_ip12[:, :, :, 0] = 0
    Fv_ip12[:, :, :, 1] = t_xx_ip12
    Fv_ip12[:, :, :, 2] = t_xy_ip12
    Fv_ip12[:, :, :, 3] = t_xz_ip12
    Fv_ip12[:, :, :, 4] = (u_ip12*t_xx_ip12 + v_ip12*t_xy_ip12 +
                           w_ip12*t_xz_ip12 - q_x_ip12)

    # Y Direction
    # -----------
    # t_xy -> uy, vx
    # t_yy -> ux, vy, wz
    # t_yz -> vz, wy
    # q_y -> Ty
    # --> Derivatives to find: ux, uy, vx, vy, vz, wy, wz, Ty

    # Cross derivatives for x/z derivatives
    ii = np.arange(3, nx+2)
    cd_ux = 0.1*(
        (15/(2*dx))*(u[ii+1, :, 3:(nz+2)] - u[ii-1, :, 3:(nz+2)]) +
        (-6/(4*dx))*(u[ii+2, :, 3:(nz+2)] - u[ii-2, :, 3:(nz+2)]) +
        (1/(6*dx))*(u[ii+3, :, 3:(nz+2)] - u[ii-3, :, 3:(nz+2)])
    )
    cd_vx = 0.1*(
        (15/(2*dx))*(v[ii+1, :, 3:(nz+2)] - v[ii-1, :, 3:(nz+2)]) +
        (-6/(4*dx))*(v[ii+2, :, 3:(nz+2)] - v[ii-2, :, 3:(nz+2)]) +
        (1/(6*dx))*(v[ii+3, :, 3:(nz+2)] - v[ii-3, :, 3:(nz+2)])
    )

    kk = np.arange(3, nz+2)
    cd_vz = 0.1*(
        (15/(2*dz))*(v[3:(nx+2), :, kk+1] - v[3:(nx+2), :, kk-1]) +
        (-6/(4*dz))*(v[3:(nx+2), :, kk+2] - v[3:(nx+2), :, kk-2]) +
        (1/(6*dz))*(v[3:(nx+2), :, kk+3] - v[3:(nx+2), :, kk-3])
    )
    cd_wz = 0.1*(
        (15/(2*dz))*(w[3:(nx+2), :, kk+1] - w[3:(nx+2), :, kk-1]) +
        (-6/(4*dz))*(w[3:(nx+2), :, kk+2] - w[3:(nx+2), :, kk-2]) +
        (1/(6*dz))*(w[3:(nx+2), :, kk+3] - w[3:(nx+2), :, kk-3])
    )

    # Fluxes @ cell interfaces (j+1/2)
    def interpolate_to_jp12_interface(q_j, nx, ny, nz):
        jj = np.arange(2, ny+2)

        if q_j.shape[0] == nx+5 and q_j.shape[2] == nz+5:
            q_jp12 = (1/60)*(
                37*(q_j[3:(nx+2), jj+1, 3:(nz+2)] +
                    q_j[3:(nx+2), jj, 3:(nz+2)])
                - 8*(q_j[3:(nx+2), jj+2, 3:(nz+2)] +
                     q_j[3:(nx+2), jj-1, 3:(nz+2)])
                + (q_j[3:(nx+2), jj+3, 3:(nz+2)] +
                   q_j[3:(nx+2), jj-2, 3:(nz+2)])
            )
        else:  # Assume it's a cross-derivative, shape (nx-1, ny+5, nz-1)
            q_jp12 = (1/60)*(
                37*(q_j[:, jj+1, :] + q_j[:, jj, :])
                - 8*(q_j[:, jj+2, :] + q_j[:, jj-1, :])
                + (q_j[:, jj+3, :] + q_j[:, jj-2, :])
            )

        return q_jp12

    ux_jp12 = interpolate_to_jp12_interface(cd_ux, nx, ny, nz)
    vx_jp12 = interpolate_to_jp12_interface(cd_vx, nx, ny, nz)
    vz_jp12 = interpolate_to_jp12_interface(cd_vz, nx, ny, nz)
    wz_jp12 = interpolate_to_jp12_interface(cd_wz, nx, ny, nz)

    # y derivatives @ cell interfaces
    def ddy_jp12(q_j, nx, ny, nz):
        jj = np.arange(2, ny+2)

        qy_jp12 = (1/180)*(
            (245/dy)*(q_j[3:(nx+2), jj+1, 3:(nz+2)] -
                      q_j[3:(nx+2), jj, 3:(nz+2)])
            + (-75/(3*dy))*(q_j[3:(nx+2), jj+2, 3:(nz+2)] -
                            q_j[3:(nx+2), jj-1, 3:(nz+2)])
            + (10/(5*dy))*(q_j[3:(nx+2), jj+3, 3:(nz+2)] -
                           q_j[3:(nx+2), jj-2, 3:(nz+2)])
        )

        return qy_jp12

    uy_jp12 = ddy_jp12(u, nx, ny, nz)
    vy_jp12 = ddy_jp12(v, nx, ny, nz)
    wy_jp12 = ddy_jp12(w, nx, ny, nz)
    Ty_jp12 = ddy_jp12(T, nx, ny, nz)

    # Other values interpolated to cell interfaces (j+1/2)
    mu_jp12 = interpolate_to_jp12_interface(mu, nx, ny, nz)
    u_jp12 = interpolate_to_jp12_interface(u, nx, ny, nz)
    v_jp12 = interpolate_to_jp12_interface(v, nx, ny, nz)
    w_jp12 = interpolate_to_jp12_interface(w, nx, ny, nz)

    t_xy_jp12 = (mu_jp12/Re)*(uy_jp12 + vx_jp12)
    t_yy_jp12 = (mu_jp12/Re)*(2*vy_jp12 - (2/3)*(ux_jp12 + vy_jp12 + wz_jp12))
    t_yz_jp12 = (mu_jp12/Re)*(vz_jp12 + wy_jp12)
    q_y_jp12 = -(mu_jp12 / (Re*Pr*Ma**2 * (gamma-1))) * Ty_jp12

    # Flux in Y-dir
    Gv_jp12 = np.zeros((nx-1, ny, nz-1, 5))
    Gv_jp12[:, :, :, 0] = 0
    Gv_jp12[:, :, :, 1] = t_xy_jp12
    Gv_jp12[:, :, :, 2] = t_yy_jp12
    Gv_jp12[:, :, :, 3] = t_yz_jp12
    Gv_jp12[:, :, :, 4] = (u_jp12*t_xy_jp12 + v_jp12*t_yy_jp12 +
                           w_jp12*t_yz_jp12 - q_y_jp12)

    # Z Direction
    # -----------
    # t_xz -> uz, wx
    # t_yz -> vz, wy
    # t_zz -> ux, vy, wz
    # q_z -> Tz
    # --> Derivatives to find: ux, uz, vy, vz, wx, wy, wz, Tz

    # Cross derivatives for x/y derivatives
    ii = np.arange(3, nx+2)
    cd_ux = 0.1*(
        (15/(2*dx))*(u[ii+1, 3:(ny+2), :] - u[ii-1, 3:(ny+2), :]) +
        (-6/(4*dx))*(u[ii+2, 3:(ny+2), :] - u[ii-2, 3:(ny+2), :]) +
        (1/(6*dx))*(u[ii+3, 3:(ny+2), :] - u[ii-3, 3:(ny+2), :])
    )
    cd_wx = 0.1*(
        (15/(2*dx))*(w[ii+1, 3:(ny+2), :] - w[ii-1, 3:(ny+2), :]) +
        (-6/(4*dx))*(w[ii+2, 3:(ny+2), :] - w[ii-2, 3:(ny+2), :]) +
        (1/(6*dx))*(w[ii+3, 3:(ny+2), :] - w[ii-3, 3:(ny+2), :])
    )

    jj = np.arange(3, ny+2)
    cd_vy = 0.1*(
        (15/(2*dy))*(v[3:(nx+2), jj+1, :] - v[3:(nx+2), jj-1, :]) +
        (-6/(4*dy))*(v[3:(nx+2), jj+2, :] - v[3:(nx+2), jj-2, :]) +
        (1/(6*dy))*(v[3:(nx+2), jj+3, :] - v[3:(nx+2), jj-3, :])
    )
    cd_wy = 0.1*(
        (15/(2*dy))*(w[3:(nx+2), jj+1, :] - w[3:(nx+2), jj-1, :]) +
        (-6/(4*dy))*(w[3:(nx+2), jj+2, :] - w[3:(nx+2), jj-2, :]) +
        (1/(6*dy))*(w[3:(nx+2), jj+3, :] - w[3:(nx+2), jj-3, :])
    )

    # Fluxes @ cell interfaces (k+1/2)
    def interpolate_to_kp12_interface(q_k, nx, ny, nz):
        kk = np.arange(2, nz+2)

        if q_k.shape[0] == nx+5 and q_k.shape[1] == ny+5:
            q_kp12 = (1/60)*(
                37*(q_k[3:(nx+2), 3:(ny+2), kk+1] +
                    q_k[3:(nx+2), 3:(ny+2), kk])
                - 8*(q_k[3:(nx+2), 3:(ny+2), kk+2] +
                     q_k[3:(nx+2), 3:(ny+2), kk-1])
                + (q_k[3:(nx+2), 3:(ny+2), kk+3] +
                   q_k[3:(nx+2), 3:(ny+2), kk-2])
            )
        else:  # Assume it's a cross-derivative, shape (nx-1, ny-1, nz+5)
            q_kp12 = (1/60)*(
                37*(q_k[:, :, kk+1] + q_k[:, :, kk])
                - 8*(q_k[:, :, kk+2] + q_k[:, :, kk-1])
                + (q_k[:, :, kk+3] + q_k[:, :, kk-2])
            )

        return q_kp12

    ux_kp12 = interpolate_to_kp12_interface(cd_ux, nx, ny, nz)
    wx_kp12 = interpolate_to_kp12_interface(cd_wx, nx, ny, nz)
    vy_kp12 = interpolate_to_kp12_interface(cd_vy, nx, ny, nz)
    wy_kp12 = interpolate_to_kp12_interface(cd_wy, nx, ny, nz)

    # z derivatives @ cell interfaces
    def ddz_kp12(q_k, nx, ny, nz):
        kk = np.arange(2, nz+2)

        qz_kp12 = (1/180)*(
            (245/dy)*(q_k[3:(nx+2), 3:(ny+2), kk+1] -
                      q_k[3:(nx+2), 3:(ny+2), kk])
            + (-75/(3*dy))*(q_k[3:(nx+2), 3:(ny+2), kk+2] -
                            q_k[3:(nx+2), 3:(ny+2), kk-1])
            + (10/(5*dy))*(q_k[3:(nx+2), 3:(ny+2), kk+3] -
                           q_k[3:(nx+2), 3:(ny+2), kk-2])
        )

        return qz_kp12

    uz_kp12 = ddz_kp12(u, nx, ny, nz)
    vz_kp12 = ddz_kp12(v, nx, ny, nz)
    wz_kp12 = ddz_kp12(w, nx, ny, nz)
    Tz_kp12 = ddz_kp12(T, nx, ny, nz)

    # Other values interpolated to cell interfaces (k+1/2)
    mu_kp12 = interpolate_to_kp12_interface(mu, nx, ny, nz)
    u_kp12 = interpolate_to_kp12_interface(u, nx, ny, nz)
    v_kp12 = interpolate_to_kp12_interface(v, nx, ny, nz)
    w_kp12 = interpolate_to_kp12_interface(w, nx, ny, nz)

    t_xz_kp12 = (mu_kp12/Re)*(uz_kp12 + wx_kp12)
    t_yz_kp12 = (mu_kp12/Re)*(vz_kp12 + wy_kp12)
    t_zz_kp12 = (mu_kp12/Re)*(2*wz_kp12 - (2/3)*(ux_kp12 + vy_kp12 + wz_kp12))
    q_z_kp12 = -(mu_kp12 / (Re*Pr*Ma**2 * (gamma-1))) * Tz_kp12

    # Flux in Z-dir
    Hv_kp12 = np.zeros((nx-1, ny-1, nz, 5))
    Hv_kp12[:, :, :, 0] = 0
    Hv_kp12[:, :, :, 1] = t_xz_kp12
    Hv_kp12[:, :, :, 2] = t_yz_kp12
    Hv_kp12[:, :, :, 3] = t_zz_kp12
    Hv_kp12[:, :, :, 4] = (u_kp12*t_xz_kp12 + v_kp12*t_yz_kp12 +
                           w_kp12*t_zz_kp12 - q_z_kp12)

    return Fv_ip12, Gv_jp12, Hv_kp12


if __name__ == "__main__":
    main()
