// References:
// -----------
// [1] San et. al. 2014 Numerical.
//     https://doi.org/10.1016/j.compfluid.2013.11.006
// [2] Maulik et. al. 2018 Adaptive. https://doi.org/10.1002/fld.4489
// [3] Jiang et. al. 1999. https://doi.org/10.1006/jcph.1999.6207

#define EIGEN_FFTW_DEFAULT
#include "matplotlibcpp.h"
#include <chrono>
#include <cmath>
#include <complex>
#include <iomanip>
#include <iostream>
#include <math.h>
#include <vector>
#include <utility>
#include <Eigen/Dense>
#include <unsupported/Eigen/FFT>

using namespace Eigen;
namespace plt = matplotlibcpp;

const double pi{ 3.1415926535897 };

// Supporting Code
std::chrono::nanoseconds dummy_time;
struct Stopwatch {
  Stopwatch(std::chrono::nanoseconds& result)
    : name{ "No_name" },
    result{ result },
    start{ std::chrono::high_resolution_clock::now() } { }
  Stopwatch(std::string name, std::chrono::nanoseconds& result)
    : name{ name },
    result{ result },
    start{ std::chrono::high_resolution_clock::now() } { }
  Stopwatch(std::string name)
    : name{ name },
    result{ dummy_time },
    start{ std::chrono::high_resolution_clock::now() } { }

  ~Stopwatch() {
    result = std::chrono::high_resolution_clock::now() - start;
    std::cout << "-[" << name << "] Ended. Elapsed Time: " <<
      std::setprecision(4) << result.count()/1e6 << " ms" << std::endl;
  }
private:
  const std::string name;
  std::chrono::nanoseconds& result;
  const std::chrono::time_point<std::chrono::high_resolution_clock> start;
};

// Forward declarations
// --------------------
std::vector<size_t> get_nxs_from_cmd_line(int argc, char** argv, size_t default_nx = std::pow(2, 4));

// Present results
void plot_E_ks(const std::vector<ArrayXd>& E_ks, const std::vector<size_t>& nxs);
ArrayXd get_KE_k_space(const ArrayXXd& q, const size_t nx);

// Main algorithm
template <typename T>
ArrayXd outer_loop(const size_t nx, const size_t ns, const T nu, const T tmax);

template <typename T>
ArrayXXd tvdrk3(const ArrayXXd& q_n, const size_t nx, const T dx, const T dt, const T nu);

template <typename T>
ArrayXXd rhs(const ArrayXXd& q, const size_t nx, const T dx, const T nu);

// Reconstruction Scheme
std::tuple<ArrayXXd, ArrayXXd> weno_5(const ArrayXXd& q, const size_t nx);

// Viscous Contribution
template <typename T>
ArrayXXd c4ddp(const ArrayXXd& f, const T h);

template <typename T>
ArrayXd TDMA_cyclic(ArrayXd a, ArrayXd b, ArrayXd c, ArrayXd d, const T alpha, const T beta);

ArrayXd TDMAsolver(ArrayXd a, ArrayXd b, ArrayXd c, ArrayXd d);

// Supporting functions
template <typename T>
ArrayXXd get_IC_q(const size_t ns, const size_t nx, const T dx);

template <typename T>
T get_dt(ArrayXXd q, const size_t nx, const T dx);

ArrayXXd perbc(ArrayXXd q, const size_t nx);
ArrayXXd get_wave_speeds(const ArrayXXd& q, const size_t nx);
ArrayXXd get_flux(const ArrayXXd& q);
ArrayXXd rusanov_riemann(const ArrayXXd& FR,
                         const ArrayXXd& FL,
                         const ArrayXXd& qR_ip12,
                         const ArrayXXd& qL_ip12,
                         const ArrayXXd& c_ip12);
// ---------------------

int main(int argc, char** argv) {
  const auto nxs = get_nxs_from_cmd_line(argc, argv);
  const size_t ns = std::pow(2, 6);
  const double nu = 5e-4;
  const double tmax = 0.2;

  std::vector<ArrayXd> E_ks;
  {
    Stopwatch stopwatch{ "All Runs" };
    for (size_t nx : nxs) {
      Stopwatch stopwatch{ "NX="+std::to_string(nx) };
      ArrayXd E_k = outer_loop<double>(nx, ns, nu, tmax);
      E_ks.emplace_back(E_k);
    }
  }

  plot_E_ks(E_ks, nxs);
  return 0;
}

std::vector<size_t> get_nxs_from_cmd_line(int argc, char** argv, size_t default_nx) {
  std::vector<size_t> nxs(default_nx);

  if (argc > 1) {
    nxs.clear();
    for (size_t i{ 1 }; i<argc; i++) {
      size_t nx = strtol(argv[i], NULL, 10);
      if (strtol(argv[i], NULL, 10) <= 0) {
        std::cerr << "Invalid input! Must enter a positive integer for NX." << std::endl;
      }
      nxs.emplace_back(nx);
    }
  }

  return nxs;
}

void plot_E_ks(const std::vector<ArrayXd>& E_ks, const std::vector<size_t>& nxs) {
  std::string filename = "E_k_nx=";
  for (size_t nx : nxs) filename += std::to_string(nx) + "_";
  filename.pop_back();
  filename += ".pdf";

  std::cout << "\n* Plotting '" << filename << "'" << std::endl;

  for (size_t i{}; i<nxs.size(); i++) {
    VectorXd x = VectorXd::LinSpaced(E_ks[i].rows(), 0, E_ks[i].rows());
    std::vector xv(x.begin(), x.end());
    std::vector E_kv(E_ks[i].begin(), E_ks[i].end());
    plt::named_loglog("NX="+std::to_string(nxs[i]), xv, E_kv);
  }
  plt::legend();
  plt::save(filename);
  std::cout << "  * Done" << std::endl;
}


// ================================================================================================== 
// ========================================= Main Algorithm ========================================= 
// ================================================================================================== 
template <typename T>
ArrayXd outer_loop(const size_t nx, const size_t ns, const T nu, const T tmax) {
  const T lx{ 2*pi };
  const T dx{ lx / nx };
  ArrayXXd q = get_IC_q(ns, nx, dx);
  T time{ 0 };

  size_t n_iterations{};
  std::cout << "\nSolving Burgers Problem Using WENO-5 Reconstruction";
  std::cout << "\nnx=" << nx << ", ns=" << ns << ", nu=" << nu << ", tmax=" << tmax << std::endl;
  T dt;
  while (time < tmax) {
    auto start = std::chrono::high_resolution_clock::now();

    n_iterations++;
    dt = get_dt(q, nx, dx);

    // Make sure we don't go past tmax
    dt = (time + dt) < tmax ? dt : tmax - time;
    // Also make sure we get a data point at 0.05 s
    dt = (time < 0.05 && time+dt > 0.05) ? (0.05 - time) : dt;
    time += dt;

    // Advance a time step using TVDRK3 time integration
    q = tvdrk3(q, nx, dx, dt, nu);

    auto stop = std::chrono::high_resolution_clock::now();
    auto elapsed = stop - start;
    std::cout << std::setprecision(4);
    std::cout << "  * time: " << std::fixed << time << " s, ";
    std::cout << "dt: " << std::scientific << dt << " s, ";
    std::cout << "took: " << std::fixed << elapsed.count()/1e6 << " ms  \r";
    std::cout << std::flush;
  }

  std::cout << "\n  * Done (" << n_iterations << " iterations)" << std::endl;

  ArrayXd E_k = get_KE_k_space(q, nx);
  return E_k;
}


ArrayXd get_KE_k_space(const ArrayXXd& q, const size_t nx) {
  // Get the average kinetic energy of the domain (0:nx)
  // E = (1/nx) * 0.5*sum(u**2)

  ArrayXXd u_k = ArrayXXd::Zero(q.rows(), q.cols());
  FFT<double> fft{};
  Eigen::VectorXcd u_ki;
  for (size_t i{}; i<q.cols(); i++) {
    Eigen::VectorXcd q_i = q.col(i);
    fft.fwd(u_ki, q_i);
    u_k.col(i) = u_ki.real() / q.rows();
  }

  ArrayXXd Es = 0.5*pow(u_k(seqN(3, nx-1), all), 2);  // Energy spectrum
  ArrayXi i = ArrayXi::LinSpaced(nx/2 - 2, 1, nx/2 - 1);
  ArrayXd E_k = (0.5*(Es(i, all) + Es(nx-1-i, all))).rowwise().sum() / q.cols();

  return E_k;
}


template <typename T>
ArrayXXd tvdrk3(const ArrayXXd& q_n, const size_t nx, const T dx, const T dt, const T nu) {
  // 3rd Order Runge-Kutta time integrator.
  ArrayXXd q1( q_n );
  auto i = ArrayXi::LinSpaced(nx-1, 3, nx+2);
  q1(i, all) = q_n(i, all) + dt*rhs(q_n, nx, dx, nu);
  q1 = perbc(q1, nx);

  ArrayXXd q2( q_n );
  q2(i, all) = 0.75*q_n(i, all) + 0.25*q1(i, all) + 0.25*dt*rhs(q1, nx, dx, nu);
  q2 = perbc(q2, nx);

  ArrayXXd q_np1( q_n );
  q_np1(i, all) = (1./3)*q_n(i, all) + (2./3)*q2(i, all) + (2./3)*dt*rhs(q2, nx, dx, nu);
  q_np1 = perbc(q_np1, nx);

  return q_np1;
}


template <typename T>
ArrayXXd rhs(const ArrayXXd& q, const size_t nx, const T dx, const T nu) {
  // Note:
  //   q.shape() == {nx+5, ns}
  //                 ^^^^ Specifically from 0:(nx+5)
  //   rhs.shape() == {nx-1, ns}
  //                   ^^^^ Specifically from 1:(nx+2)

  // Reconstruction Scheme
  auto [qL_ip12, qR_ip12] = weno_5(q, nx);

  ArrayXXd c_ip12 = get_wave_speeds(q, nx);
  ArrayXXd FR = get_flux(qR_ip12);
  ArrayXXd FL = get_flux(qL_ip12);

  // Riemann Solvers
  ArrayXXd F_ip12 = rusanov_riemann(FR, FL, qR_ip12, qL_ip12, c_ip12);

  auto i = ArrayXi::LinSpaced(nx-1, 1, nx) + 2;
  ArrayXXd rhs = -(F_ip12(i, all) - F_ip12(i-1, all)) / dx;
  //               F_ip12           F_im12

  // Viscous contribution
  ArrayXXd uxx = c4ddp(q(seqN(2, nx+1), all), dx);  // 0 <= x <= nx+1
  
  rhs += nu*uxx(seq(1, last-1), all);

  return rhs;
}

ArrayXXd rusanov_riemann(const ArrayXXd& FR,
                         const ArrayXXd& FL,
                         const ArrayXXd& qR_ip12,
                         const ArrayXXd& qL_ip12,
                         const ArrayXXd& c_ip12) {
  ArrayXXd F_ip12 = 0.5*((FR + FL) - c_ip12*(qR_ip12 - qL_ip12));

  return F_ip12;
}

std::tuple<ArrayXXd, ArrayXXd> weno_5(const ArrayXXd& q, const size_t nx) {
  // WENO 5th Order reconstruction scheme [2]
  //                                                                            
  //              Cell
  //               |
  //   Boundary -| | |- Boundary
  //             v v v
  // |-*-|-*-|...|-*-|...|-*-|
  //   0   1     ^ i ^     n
  //         i-1/2   i+1/2
  //                                                                            
  // - Based on cell definition above, estimate q_i-1/2 and q_i+1/2 using
  //   nodes values from cells on the left (q^L) or cells on the right (q^R).
  //                                                                            
  // > Note::
  // >     There may be ghost points on either end of the computational domain.
  // >     For this reason, s & e are inputs to define the indices of the start
  // >     and end of the computational domain. All points before and after
  // >     these indices are ghost points.
  //                                                                            
  // Returns:
  //     std::tuple: A tuple containing q^L_i+1/2 and q^R_i+1/2 respectively.

  const size_t ns = q.cols();

  ArrayXXd b0      = ArrayXXd::Zero(nx+5, ns);
  ArrayXXd b1      = ArrayXXd::Zero(nx+5, ns);
  ArrayXXd b2      = ArrayXXd::Zero(nx+5, ns);

  ArrayXXd qL_ip12 = ArrayXXd::Zero(nx+5, ns);
  ArrayXXd qR_ip12 = ArrayXXd::Zero(nx+5, ns);

  // Linear weighting coefficients
  double d0 = 1./10;
  double d1 = 6./10;
  double d2 = 3./10;
  double eps = 1e-6;

  // Smoothness indicators
  auto i = ArrayXi::LinSpaced(nx+1, 0, nx+1) + 2;
  b0(i, all) = ((13./12)*pow(q(i-2, all) - 2*q(i-1, all) +   q(i, all), 2)
                + (1./4)*pow(q(i-2, all) - 4*q(i-1, all) + 3*q(i, all), 2));
  b1(i, all) = ((13./12)*pow(q(i-1, all) - 2*q(i, all) + q(i+1, all), 2)
                + (1./4)*pow(q(i-1, all)               - q(i+1, all), 2));
  b2(i, all) = ((13./12)*pow(  q(i, all) - 2*q(i+1, all) + q(i+2, all), 2)
                + (1./4)*pow(3*q(i, all) - 4*q(i+1, all) + q(i+2, all), 2));

  // Nonlinear weights
  auto i2 = ArrayXi::LinSpaced(nx, 0, nx) + 2;
  ArrayXXd a0 = d0 / pow(b0(i2, all) + eps, 2);
  ArrayXXd a1 = d1 / pow(b1(i2, all) + eps, 2);
  ArrayXXd a2 = d2 / pow(b2(i2, all) + eps, 2);

  ArrayXXd w0 = a0 / (a0 + a1 + a2);
  ArrayXXd w1 = a1 / (a0 + a1 + a2);
  ArrayXXd w2 = a2 / (a0 + a1 + a2);

  // Positive reconstruction @ i+1/2
  ArrayXXd q0 = 2*q(i2-2, all) - 7*q(i2-1, all) + 11*q(i2, all);
  ArrayXXd q1 =  -q(i2-1, all) + 5*q(i2, all)   +  2*q(i2+1, all);
  ArrayXXd q2 = 2*q(i2, all)   + 5*q(i2+1, all) -    q(i2+2, all);
  qL_ip12(i2, all) = (w0/6)*q0 + (w1/6)*q1 + (w2/6)*q2;

  // Negative reconstruction @ i-1/2 + 1 (so i+1/2)
  auto i3 = ArrayXi::LinSpaced(nx, 0, nx) + 3;
  a0 = d0 / pow(b0(i3, all) + eps, 2);
  a1 = d1 / pow(b1(i3, all) + eps, 2);
  a2 = d2 / pow(b2(i3, all) + eps, 2);

  w0 = a0 / (a0 + a1 + a2);
  w1 = a1 / (a0 + a1 + a2);
  w2 = a2 / (a0 + a1 + a2);

  q0 = 2*q(i3+2, all) - 7*q(i3+1, all) + 11*q(i3, all);
  q1 =  -q(i3+1, all) + 5*q(i3, all)   +  2*q(i3-1, all);
  q2 = 2*q(i3, all)   + 5*q(i3-1, all) -    q(i3-2, all);
  qR_ip12(i3-1, all) = (w0/6)*q0 + (w1/6)*q1 + (w2/6)*q2;

  return std::make_tuple(qL_ip12, qR_ip12);
}


template <typename T>
ArrayXXd c4ddp(const ArrayXXd& f, const T h) {
  //
  // 4th order compact scheme 2nd derivative periodic
  //
  // f_ = f minus repeating part (f[0] = f[-1])
  const size_t nx = f.rows();
  const size_t ns = f.cols();

  ArrayXXd f_ = f(seq(0, nx-1), all);
  ArrayXXd f_pp = ArrayXXd::Zero(nx, ns);
  ArrayXXd r    = ArrayXXd::Zero(nx-1, ns);

  ArrayXXd a = (1./10) * ArrayXXd::Ones(nx-1, ns);
  ArrayXXd b =      1. * ArrayXXd::Ones(nx-1, ns);
  ArrayXXd c = (1./10) * ArrayXXd::Ones(nx-1, ns);

  auto i = ArrayXi::LinSpaced(nx-2, 1, nx-2);
  r(i, all)    = (6./5)*(1/std::pow(h, 2))*(f_(i+1, all) - 2*f_(i, all)    + f_(i-1, all));
  r(0, all)    = (6./5)*(1/std::pow(h, 2))*(f_(1, all)   - 2*f_(0, all)    + f_(last, all));
  r(last, all) = (6./5)*(1/std::pow(h, 2))*(f_(0, all)   - 2*f_(last, all) + f_(last-1, all));

  const T alpha = 1./10;
  const T beta = 1./10;
  for (size_t s{}; s<ns; s++) {
    f_pp(seq(0, last-1), s) = TDMA_cyclic(a.col(s), b.col(s), c.col(s), r.col(s), alpha, beta);
  }
  f_pp(last, all) = f_pp.row(0);

  return f_pp;
}


template <typename T>
ArrayXd TDMA_cyclic(ArrayXd a, ArrayXd b, ArrayXd c, ArrayXd d, const T alpha, const T beta) {

  const size_t nf = a.rows();

  const T gamma = 10.*b(0);
  b(0) = b(0) - gamma;
  b(nf-1) = b(nf-1) - alpha*beta/gamma;
  ArrayXd y = TDMAsolver(a, b, c, d);

  ArrayXd u = ArrayXd::Zero(nf);
  u(0) = gamma;
  u(nf-1) = alpha;
  ArrayXd q = TDMAsolver(a, b, c, u);

  ArrayXd x = y - q*((y(0) + (beta/gamma)*y(nf-1)) /
                     (1 + q(0) + (beta/gamma)*q(nf-1)));

  return x;
}


ArrayXd TDMAsolver(ArrayXd a, ArrayXd b, ArrayXd c, ArrayXd d) {
  // TDMA solver, a b c d can be NumPy array type or Python list type.
  // refer to http://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
  // and to http://www.cfd-online.com/Wiki/Tridiagonal_matrix_algorithm_-
  // _TDMA_(Thomas_algorithm)

  const size_t nf = a.rows();  // number of equations

  double m;
  for (size_t i{ 1 }; i<nf; i++) {
    m = a(i-1) / b(i-1);
    b(i) = b(i) - m*c(i-1);
    d(i) = d(i) - m*d(i-1);
  }

  ArrayXd x( b );
  x(nf-1) = d(nf-1) / b(nf-1);

  for (size_t i{ nf-2 }; i>0; i--) {
    x(i) = (d(i) - c(i)*x(i+1)) / b(i);
  }
  x(0) = (d(0) - c(0)*x(1)) / b(0);  // Doing i=0 in the for loop isn't working for some reason.

  return x;
}


// -------------------------------------------- 
// -------------- Misc Functions -------------- 
// -------------------------------------------- 
template <typename T>
ArrayXXd get_IC_q(const size_t ns, const size_t nx, const T dx) {
  using namespace std::literals::complex_literals;

  // nx+5 because 2 ghost points at lhs of domain, 3 ghost points at rhs of domain
  ArrayXXd q = ArrayXXd::Zero(nx+5, ns);
  double lx = nx*dx;

  // Wave numbers
  ArrayXXd s = ArrayXXd::Zero(nx/2, ns);
  for (size_t i{}; i<nx/2; i++) {
    s(i, all) = ArrayXXd::Ones(1, ns)*i;
  }

  ArrayXXd kx = ArrayXXd::Zero(nx+5, ns);
  kx(seqN(2, nx/2), all)      = 2*pi*s / lx;
  kx(seqN(nx/2+2, nx/2), all) = 2*pi*(s - nx/2) / lx;

  // Initial energy spectrum in wavenumber space
  const double k0 = 10.;
  const double A = ( 2. / (3.*sqrt(pi)) ) * pow(k0, -5);
  ArrayXXd E_k = A*pow(kx, 4) * exp(-pow(kx/k0, 2));

  // Velocity in Fourier space
  ArrayXXd r = ArrayXXd::Random(nx, ns);
  r(seq(0, nx/2), all) = -r(seq(0, nx/2), all);

  ArrayXXcd u_k = sqrt(2.*E_k(seqN(2, nx), all)) * exp(2.0i * pi * r);

  // Velocity in physical space
  ArrayXXd u = ArrayXXd::Zero(nx, ns);
  FFT<double> fft{};
  Eigen::VectorXcd u_i;
  for (size_t i{}; i<ns; i++) {
    Eigen::VectorXcd u_ki = u_k.col(i);
    fft.fwd(u_i, u_ki);
    u.col(i) = u_i.real();
  }

  q(seqN(2, nx), all) = u;
  q = perbc(q, nx);

  return q;
}

ArrayXXd perbc(ArrayXXd q, const size_t nx) {
  q.row(2) = q.row(nx+1);
  q.row(1) = q.row(nx);
  q.row(0) = q.row(nx-1);
  q.row(nx+2) = q.row(3);
  q.row(nx+3) = q.row(4);
  q.row(nx+4) = q.row(5);

  return q;
}

template <typename T>
T get_dt(ArrayXXd q, const size_t nx, const T dx) {
  const T cfl = 0.5;
  ArrayXXd c_ip12 = get_wave_speeds(q, nx);
  T dt = (cfl * dx / c_ip12).minCoeff();

  return dt;
}

ArrayXXd get_wave_speeds(const ArrayXXd& q, const size_t nx) {
  ArrayXXd c_ip12 = ArrayXXd::Zero(q.rows(), q.cols());
  for (size_t s{}; s<q.cols(); s++) {
    for (size_t i{ 2 }; i<nx+2; i++) {
      c_ip12(i, s) = abs(q)(seq(i-2, i+3), s).maxCoeff();
    }
  }

  return c_ip12;
}

ArrayXXd get_flux(const ArrayXXd& q) {
  ArrayXXd F = pow(q, 2) / 2.;

  return F;
}