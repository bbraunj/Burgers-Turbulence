# References:
# -----------
# [1] San et. al. 2014 Numerical.
#     https://doi.org/10.1016/j.compfluid.2013.11.006
# [2] Maulik et. al. 2018 Adaptive. https://doi.org/10.1002/fld.4489
# [3] Jiang et. al. 1999. https://doi.org/10.1006/jcph.1999.6207
#
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def main(problem="Burgers"):
    title = f"Solving {problem} Problem"
    print(title + "\n" + "="*len(title))

    results = []
    # for nx in [2000, 200]:
    for nx in [200]:
        lx = 1
        dx = lx/nx
        q = get_IC_q(nx, dx, problem)
        tmax = get_tmax(problem)  # s
        time = 0

        print(f"nx = {nx}")
        while time < tmax:
            dt = get_dt(q, dx, problem)
            time += dt

            q = tvdrk3(nx, dx, dt, q, problem)
            print(f"  * time: {time:.4f} s", end="\r")

        print("\n  * Done")
        # if problem in ["Sod Shock Tube", "Brio-Wu Shock Tube"]:
        #     print(f"rho = {q[:, 0]}")
        results.append({
            "lx": lx,
            "nx": nx,
            "q": q[2:nx+2],
            "t": time,
        })

    plot_results(results, problem)


def plot_results(results, problem):
    quantities_to_plot = {
        "Burgers": ["u"],
        "Shallow Water": ["h"],
        "Sod Shock Tube": ["rho"],
        "Brio-Wu Shock Tube": ["rho", "u"],
    }
    for ylabel in quantities_to_plot[problem]:
        plt.clf()
        for r in results:
            if problem == "Burgers":
                y = r["q"]
            elif problem == "Shallow Water":
                rho = 1
                h = r["q"][:, 0] / rho
                y = h
            elif problem == "Sod Shock Tube":
                rho = r["q"][:, 0]
                y = rho
            elif problem == "Brio-Wu Shock Tube":
                rho = r["q"][:, 0]
                u = r["q"][:, 1] / rho
                y = rho if ylabel == "rho" else u

            plt.plot(np.linspace(0, r["lx"], r["nx"]), y,
                     linewidth=1, label=f"nx={r['nx']}")

        plt.legend()
        plt.ylabel(ylabel)
        plt.xlabel("x")
        plt.suptitle(f"Solutions to {problem} Problem at t={r['t']:.4f}s")
        Path("output").mkdir(exist_ok=True)
        plt.savefig(f"output/Results_{problem.replace(' ', '_')}_Problem_"
                    f"{ylabel}.png")


def tvdrk3(nx, dx, dt, q_n, problem):
    """
    3rd Order Runge-Kutta time integrator.
    """
    i = np.arange(3, nx+2)
    q1 = np.copy(q_n)
    q1[i, :] = q_n[i, :] + dt*rhs(nx, dx, q_n, problem)

    q2 = np.copy(q_n)
    q2[i, :] = (0.75*q_n[i, :] + 0.25*q1[i, :]
                + 0.25*dt*rhs(nx, dx, q1, problem))

    q_np1 = np.copy(q_n)
    q_np1[i, :] = ((1/3)*q_n[i, :] + (2/3)*q2[i, :]
                   + (2/3)*dt*rhs(nx, dx, q2, problem))

    return q_np1


def rhs(nx, dx, q, problem):
    """
    Note::
        len(q) == nx+5
                  ^^^^ Specifically from 0:(nx+5)
        len(rhs) == nx-1
                    ^^^^ Specifically from 3:(nx+2)
    """
    # 1st order non-reflective (transmissive) boundary conditions
    q[2, :] = q[3, :]
    q[1, :] = q[4, :]
    q[0, :] = q[5, :]
    q[nx+2, :] = q[nx+1, :]
    q[nx+3, :] = q[nx, :]
    q[nx+4, :] = q[nx-1, :]

    # len(qR_ip12) == len(qL_ip12) == nx
    qL_ip12, qR_ip12 = weno_5(q)
    # qL_ip12, qR_ip12 = weno_5z(q)

    # len(c_ip12) == nx
    c_ip12 = get_wave_speeds(q, problem)

    # Rusanov's Riemann Solver
    # ========================
    F = get_flux(q, problem)
    # len(FR) == len(FL) == nx
    # * The range 2:(nx+2) excludes ghost points
    FR = F[2:(nx+2), :]*qR_ip12
    FL = F[2:(nx+2), :]*qL_ip12

    F_ip12 = 0.5*((FR + FL) - (c_ip12*(qR_ip12 - qL_ip12).T).T)
    # len(F_im12) == len(F_ip12) == nx-1
    F_im12 = F_ip12[:-1]
    F_ip12 = F_ip12[1:]

    rhs = -(F_ip12 - F_im12) / dx

    return rhs


def weno_5(q):
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
    a0 = d0 / (b0[i_, :] + eps)**2
    a1 = d1 / (b1[i_, :] + eps)**2
    a2 = d2 / (b2[i_, :] + eps)**2

    w0 = a0 / (a0 + a1 + a2)
    w1 = a1 / (a0 + a1 + a2)
    w2 = a2 / (a0 + a1 + a2)

    ii = np.arange(2, n-3)
    qL_ip12 = ((w0/6)*(2*q[ii-2, :] - 7*q[ii-1, :] + 11*q[ii, :])
               + (w1/6)*(-q[ii-1, :] + 5*q[ii, :] + 2*q[ii+1, :])
               + (w2/6)*(2*q[ii, :] + 5*q[ii+1, :] - q[ii+2, :]))

    a0 = d2 / (b0[i_+1, :] + eps)**2
    a1 = d1 / (b1[i_+1, :] + eps)**2
    a2 = d0 / (b2[i_+1, :] + eps)**2

    w0 = a0 / (a0 + a1 + a2)
    w1 = a1 / (a0 + a1 + a2)
    w2 = a2 / (a0 + a1 + a2)

    qR_ip12 = ((w0/6)*(-q[ii-1, :] + 5*q[ii, :] + 2*q[ii+1, :])
               + (w1/6)*(2*q[ii, :] + 5*q[ii+1, :] - q[ii+2, :])
               + (w2/6)*(11*q[ii+1, :] - 7*q[ii+2, :] + 2*q[ii+3, :]))

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
def get_IC_q(nx, dx, problem):
    #
    # Use:
    #   - 2 ghost points on LHS of domain
    #   - 3 ghost points on RHS of domain

    if problem == "Burgers":
        q = np.zeros((nx+5, 1))
        q[(int(0.5/dx)+2):(int(0.6/dx)+2), :] = 1
    elif problem == "Shallow Water":
        q = np.ones((nx+5, 2))
        rho = 1
        h0 = np.ones(nx+5)
        h0[0:(int(0.5/dx)+2)] = 10
        u0 = 0

        q[:, 0] = rho*h0
        q[:, 1] = rho*h0*u0
    elif problem == "Sod Shock Tube":
        rho0 = np.ones(nx+5)
        u0 = 0
        p0 = np.ones(nx+5)

        rho0[(int(0.5/dx)+2):] = 0.125
        p0[(int(0.5/dx)+2):] = 0.1

        gamma = 7/5  # Adiabatic index
        E0 = (p0 / (gamma-1)) + 0.5*rho0*u0**2

        q = np.zeros((nx+5, 3))
        # Euler equations
        q[:, 0] = rho0
        q[:, 1] = rho0*u0
        q[:, 2] = E0
    elif problem == "Brio-Wu Shock Tube":
        rho0 = np.ones(nx+5)
        u0 = 0
        v0 = 0
        w0 = 0
        Bx0 = 0.75
        By0 = np.ones(nx+5)
        Bz0 = 0
        p0 = np.ones(nx+5)

        rho0[(int(0.5/dx)+2):] = 0.125
        By0[(int(0.5/dx)+2):] = -1

        gamma = 7/5  # Adiabatic index
        E0 = ((p0 / (gamma-1)) + 0.5*rho0*(u0**2 + v0**2 + w0**2)
              + 0.5*(Bx0**2 + By0**2 + Bz0**2))

        q = np.zeros((nx+5, 7))
        # MHD equations
        q[:, 0] = rho0
        q[:, 1] = rho0*u0
        q[:, 2] = rho0*v0
        q[:, 3] = rho0*w0
        q[:, 4] = By0
        q[:, 5] = Bz0
        q[:, 6] = E0

    return q


def get_tmax(problem):
    if problem == "Burgers":
        tmax = 0.075
    elif problem == "Shallow Water":
        tmax = 0.1
    elif problem == "Sod Shock Tube" or problem == "Brio-Wu Shock Tube":
        tmax = 0.2

    return tmax


def get_dt(q, dx, problem):
    cfl = {
        "Burgers": 0.5,
        "Shallow Water": 0.5,
        "Sod Shock Tube": 0.5,
        "Brio-Wu Shock Tube": 0.5,
    }[problem]

    c_ip12 = get_wave_speeds(q, problem)
    dt = np.nanmin(cfl * dx / c_ip12)

    return dt


def get_wave_speeds(q, problem):
    n = len(q)
    if problem == "Burgers":
        i = np.arange(2, n-3)
        c_ip12 = abs(np.vstack([q[i-2, :], q[i-1, :], q[i, :],
                                q[i+1, :], q[i+2, :], q[i+3, :]])).max(axis=0)
    elif problem == "Shallow Water":
        rho = 1
        g = 1
        h = q[:, 0] / rho
        u = q[:, 1] / q[:, 0]
        a = 2*np.sqrt(g*h)
    elif problem == "Sod Shock Tube":
        gamma = 7/5  # Adiabatic index
        rho = q[:, 0]
        u = q[:, 1] / rho
        E = q[:, 2]
        p = (gamma-1) * (E - 0.5*rho*u**2)
        a = np.sqrt(gamma*p/rho)
    elif problem == "Brio-Wu Shock Tube":
        gamma = 7/5  # Adiabatic index
        rho = q[:, 0]
        u = q[:, 1] / rho
        v = q[:, 2] / rho
        w = q[:, 3] / rho
        Bx = 0.75
        By = q[:, 4]
        Bz = q[:, 5]
        E = q[:, 6]
        p = (gamma-1) * (E - 0.5*rho*(u**2 + v**2 + w**2)
                         - 0.5*(Bx**2 + By**2 + Bz**2))
        a = np.sqrt(gamma*p/rho)

        # [3] Eqns. between 2.24 & 2.25
        bx = Bx / np.sqrt(rho)
        by = By / np.sqrt(rho)
        bz = Bz / np.sqrt(rho)
        b = np.sqrt(bx**2 + by**2 + bz**2)

        c_a = abs(bx)
        c_f = np.sqrt(0.5*(
            a**2 + b**2 + np.sqrt((a**2 + b**2)**2 - 4*a**2*bx**2)
        ))
        c_s = np.sqrt(0.5*(
            a**2 + b**2 - np.sqrt((a**2 + b**2)**2 - 4*a**2*bx**2)
        ))

        i = np.arange(2, n-3)
        c_ip12 = abs(np.vstack([
            u[i-2], (u[i-2]-c_a[i-2]), (u[i-2]-c_s[i-2]), (u[i-2]-c_f[i-2]),
            (u[i-2]+c_a[i-2]), (u[i-2]+c_s[i-2]), (u[i-2]+c_f[i-2]),
            # --
            u[i-1], (u[i-1]-c_a[i-1]), (u[i-1]-c_s[i-1]), (u[i-1]-c_f[i-1]),
            (u[i-1]+c_a[i-1]), (u[i-1]+c_s[i-1]), (u[i-1]+c_f[i-1]),
            # --
            u[i], (u[i]-c_a[i]), (u[i]-c_s[i]), (u[i]-c_f[i]),
            (u[i]+c_a[i]), (u[i]+c_s[i]), (u[i]+c_f[i]),
            # --
            u[i+1], (u[i+1]-c_a[i+1]), (u[i+1]-c_s[i+1]), (u[i+1]-c_f[i+1]),
            (u[i+1]+c_a[i+1]), (u[i+1]+c_s[i+1]), (u[i+1]+c_f[i+1]),
            # --
            u[i+2], (u[i+2]-c_a[i+2]), (u[i+2]-c_s[i+2]), (u[i+2]-c_f[i+2]),
            (u[i+2]+c_a[i+2]), (u[i+2]+c_s[i+2]), (u[i+2]+c_f[i+2]),
            # --
            u[i+3], (u[i+3]-c_a[i+3]), (u[i+3]-c_s[i+3]), (u[i+3]-c_f[i+3]),
            (u[i+3]+c_a[i+3]), (u[i+3]+c_s[i+3]), (u[i+3]+c_f[i+3]),
        ])).max(axis=0)

    if problem in ["Shallow Water", "Sod Shock Tube"]:
        i = np.arange(2, n-3)
        c_ip12 = abs(np.vstack([
            u[i-2], (u[i-2]-a[i-2]), (u[i-2]+a[i-2]),
            u[i-1], (u[i-1]-a[i-1]), (u[i-1]+a[i-1]),
            u[i], (u[i]-a[i]), (u[i]+a[i]),
            u[i+1], (u[i+1]-a[i+1]), (u[i+1]+a[i+1]),
            u[i+2], (u[i+2]-a[i+2]), (u[i+2]+a[i+2]),
            u[i+3], (u[i+3]-a[i+3]), (u[i+3]+a[i+3]),
        ])).max(axis=0)

    return c_ip12


def get_flux(q, problem):
    if problem == "Burgers":
        F = q**2 / 2
    elif problem == "Shallow Water":
        rho = 1
        g = 1
        u = q[:, 1] / q[:, 0]
        h = q[:, 0] / rho

        F = np.zeros(q.shape)
        F[:, 0] = rho*h*u
        F[:, 1] = rho*h*u**2 + 0.5*rho*g*h**2
    elif problem == "Sod Shock Tube":
        gamma = 7/5  # Adiabatic index
        rho = q[:, 0]
        u = q[:, 1] / rho
        E = q[:, 2]
        p = (gamma-1) * (E - 0.5*rho*u**2)

        F = np.zeros(q.shape)
        F[:, 0] = rho*u
        F[:, 1] = rho*u**2 + p
        F[:, 2] = u*(E + p)
    elif problem == "Brio-Wu Shock Tube":
        gamma = 7/5  # Adiabatic index
        rho = q[:, 0]
        u = q[:, 1] / rho
        v = q[:, 2] / rho
        w = q[:, 3] / rho
        Bx = 0.75
        By = q[:, 4]
        Bz = q[:, 5]
        E = q[:, 6]
        p = (gamma-1) * (E - 0.5*rho*(u**2 + v**2 + w**2)
                         - 0.5*(Bx**2 + By**2 + Bz**2))
        p_star = p + 0.5*(Bx**2 + By**2 + Bz**2)

        F = np.zeros(q.shape)
        F[:, 0] = rho*u
        F[:, 1] = rho*u**2 + p_star - Bx**2
        F[:, 2] = rho*u*v - Bx*By
        F[:, 3] = rho*u*w - Bx*Bz
        F[:, 4] = By*u - Bx*v
        F[:, 5] = Bz*u - Bx*w
        F[:, 6] = u*(E + p_star) - Bx*(Bx*u + By*v + Bz*w)

    return F


if __name__ == "__main__":
    main(problem="Burgers")
    main(problem="Shallow Water")
    main(problem="Sod Shock Tube")
    main(problem="Brio-Wu Shock Tube")
