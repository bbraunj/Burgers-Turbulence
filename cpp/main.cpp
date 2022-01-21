#define EIGEN_FFTW_DEFAULT
#include <chrono>
#include <cmath>
#include <complex>
#include <iomanip>
#include <iostream>
#include <math.h>
#include <utility>
#include <Eigen/Dense>
#include <unsupported/Eigen/FFT>

using namespace Eigen;


// Function declarations
// ---------------------
template <typename T>
void outer_loop(const size_t nx, const size_t ns, const T nu, const T tmax);
template <typename T>
ArrayXXd tvdrk3(ArrayXXd q_n, const size_t nx, const T dx, const T dt, const T nu);
template <typename T>
ArrayXXd rhs(ArrayXXd q, const size_t nx, const T dx, const T nu);
ArrayXXd rusanov_riemann(ArrayXXd FR,
                         ArrayXXd FL,
                         ArrayXXd qR_ip12,
                         ArrayXXd qL_ip12,
                         ArrayXXd c_ip12);
ArrayXXd perbc(ArrayXXd q, const size_t nx);

// Reconstruction Scheme
std::tuple<ArrayXXd, ArrayXXd> weno_5(ArrayXXd q, const size_t nx);

// Viscous Contribution
template <typename T>
ArrayXXd c4ddp(ArrayXXd f, const T h);
template <typename T>
ArrayXd TDMA_cyclic(ArrayXd a, ArrayXd b, ArrayXd c, ArrayXd d, const T alpha, const T beta);
ArrayXd TDMAsolver(ArrayXd a, ArrayXd b, ArrayXd c, ArrayXd d);

// Problem Quantities
template <typename T>
ArrayXXd get_IC_q(const size_t ns, const size_t nx, const T dx);
template <typename T>
T get_dt(ArrayXXd q, const size_t nx, const T dx);
ArrayXXd get_wave_speeds(ArrayXXd q, const size_t nx);
ArrayXXd get_flux(ArrayXXd q);
// ---------------------

const double pi{ 3.1415926535897 };

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


int main(int argc, char** argv) {
  // Get nx from command line if we can
  size_t nx = std::pow(2, 4);
  if (argc > 1) { nx = strtol(argv[1], NULL, 10); }

  // const size_t nx = vm["nx"].as<size_t>();
  const size_t ns = std::pow(2, 6);
  const double nu = 5e-4;
  const double tmax = 0.2;

  std::chrono::nanoseconds elapsed_ns;
  {
    Stopwatch stopwatch{ "Overall", elapsed_ns };
    outer_loop<double>(nx, ns, nu, tmax);
  }
  std::cout << "\nExecution time: " << std::setprecision(4) << elapsed_ns.count()/1e9 << " s" << std::endl;
}


template <typename T>
void outer_loop(const size_t nx, const size_t ns, const T nu, const T tmax) {
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
    // std::cout << "Before get_dt" << std::endl;
    dt = get_dt(q, nx, dx);
    std::cout << "    * Calculated dt: " << dt << std::endl;

    // Make sure we don't go past tmax
    dt = (time + dt) < tmax ? dt : tmax - time;
    // Also make sure we get a data point at 0.05 s
    dt = (time < 0.05 && time+dt > 0.05) ? (0.05 - time) : dt;
    time += dt;

    // Advance a time step using TVDRK3 time integration
    q = tvdrk3(q, nx, dx, dt, nu);

    auto stop = std::chrono::high_resolution_clock::now();
    auto elapsed = stop - start;
    std::cout << "  * time: " << std::setprecision(4) << time << " s"
      << ", took: " << elapsed.count()/1e9 << " s" << std::endl;
  }

  std::cout << "\n  * Done (" << n_iterations << " iterations)" << std::endl;
}


template <typename T>
ArrayXXd tvdrk3(ArrayXXd q_n, const size_t nx, const T dx, const T dt, const T nu) {
  // Stopwatch stopwatch{ "TVDRK3" };
  // 3rd Order Runge-Kutta time integrator.
  // std::cout << "tvdrk3:" << std::endl;
  ArrayXXd q1( q_n );
  auto i = ArrayXi::LinSpaced(nx-1, 3, nx+2);
  q1(i, all) = q_n(i, all) + dt*rhs(q_n, nx, dx, nu);
  // std::cout << "* q1: " << q1(5, 0) << std::endl;
  q1 = perbc(q1, nx);

  ArrayXXd q2( q_n );
  q2(i, all) = 0.75*q_n(i, all) + 0.25*q1(i, all) + 0.25*dt*rhs(q1, nx, dx, nu);
  // std::cout << "* q2: " << q2(5, 0) << std::endl;
  q2 = perbc(q2, nx);

  ArrayXXd q_np1( q_n );
  q_np1(i, all) = (1./3)*q_n(i, all) + (2./3)*q2(i, all) + (2./3)*dt*rhs(q2, nx, dx, nu);
  // std::cout << "* q_np1: " << q_np1(5, 0) << std::endl;
  q_np1 = perbc(q_np1, nx);

  return q_np1;
}


template <typename T>
ArrayXXd rhs(ArrayXXd q, const size_t nx, const T dx, const T nu) {
  // Note:
  //   q.shape() == {nx+5, ns}
  //                 ^^^^ Specifically from 0:(nx+5)
  //   rhs.shape() == {nx-1, ns}
  //                   ^^^^ Specifically from 1:(nx+2)

  // Stopwatch stopwatch{ "rhs" };
  // Reconstruction Scheme
  auto [qL_ip12, qR_ip12] = weno_5(q, nx);
  // xt::xarray<T> [qL_ip12, qR_ip12] = weno_5(q, nx);

  ArrayXXd c_ip12 = get_wave_speeds(q, nx);
  ArrayXXd FR = get_flux(qR_ip12);
  ArrayXXd FL = get_flux(qL_ip12);
  // std::cout << "  + FL: " << FL(5, 0) << std::endl;
  // std::cout << "  + FR: " << FR(5, 0) << std::endl;

  // Riemann Solvers
  ArrayXXd F_ip12 = rusanov_riemann(FR, FL, qR_ip12, qL_ip12, c_ip12);
  // std::cout << "  + F_ip12: " << F_ip12(5, 0) << std::endl;

  auto i = ArrayXi::LinSpaced(nx-1, 1, nx) + 2;
  ArrayXXd rhs = -(F_ip12(i, all) - F_ip12(i-1, all)) / dx;
  //               F_ip12           F_im12

  // Viscous contribution
  ArrayXXd uxx = c4ddp(q(seqN(2, nx+1), all), dx);  // 0 <= x <= nx+1
  
  rhs += nu*uxx(seq(1, last-1), all);
  // rhs += nu*xt::view(uxx, xt::range(1, nx), xt::all());

  return rhs;
}

ArrayXXd rusanov_riemann(ArrayXXd FR,
                         ArrayXXd FL,
                         ArrayXXd qR_ip12,
                         ArrayXXd qL_ip12,
                         ArrayXXd c_ip12) {
  ArrayXXd F_ip12 = 0.5*((FR + FL) - c_ip12*(qR_ip12 - qL_ip12));

  return F_ip12;
}

std::tuple<ArrayXXd, ArrayXXd> weno_5(ArrayXXd q, const size_t nx) {
  // Stopwatch stopwatch{ "WENO-5" };
  const size_t ns = q.cols();
  // const auto domain_shape = {q.shape()[0]-5, q.shape()[1]};
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
  // const auto i = xt::range(2, nx+2);
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
  // const auto i2 = xt::range(3, nx+3);
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

  // std::pair<xt::xarray<T>, xt::xarray<T>> p(qL_ip12, qR_ip12);
  // std::cout << "  + qL_ip12: " << qL_ip12(5, 0) << std::endl;
  // std::cout << "  + qR_ip12: " << qR_ip12(5, 0) << std::endl;
  return std::make_tuple(qL_ip12, qR_ip12);
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
ArrayXXd c4ddp(ArrayXXd f, const T h) {
  //
  // 4th order compact scheme 2nd derivative periodic
  //
  // f_ = f minus repeating part (f[0] = f[-1])
  //        plus last nonrepeating inserted at index 0 (f_[0] = f[-2])
  // xt::xarray<T> f_   = xt::zeros<T>(f.shape());
  // xt::view(f_, xt::range(1, N), xt::all()) = xt::view(f, xt::range(0, N-1), xt::all());
  // xt::view(f_, 0, xt::all()) = xt::view(f, N-2, xt::all());
  // Stopwatch stopwatch{ "C4DDP" };
  const size_t nx = f.rows();
  const size_t ns = f.cols();

  ArrayXXd f_ = f(seq(0, nx-1), all);
  ArrayXXd f_pp = ArrayXXd::Zero(nx, ns);
  // xt::xarray<T> r    = xt::zeros<T>(xt::view(f, xt::range(0, N-1), xt::all()).shape());
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
T get_dt(ArrayXXd q, const size_t nx, const T dx) {
  // Stopwatch stopwatch{ "Get_dt" };
  const T cfl = 0.5;
  ArrayXXd c_ip12 = get_wave_speeds(q, nx);
  T dt = (cfl * dx / c_ip12).minCoeff();

  return dt;
}

ArrayXXd get_wave_speeds(ArrayXXd q, const size_t nx) {
  ArrayXXd c_ip12 = ArrayXXd::Zero(q.rows(), q.cols());
  for (size_t s{}; s<q.cols(); s++) {
    for (size_t i{ 2 }; i<nx+2; i++) {
      c_ip12(i, s) = abs(q)(seq(i-2, i+3), s).maxCoeff();
    }
  }

  return c_ip12;
}

ArrayXXd get_flux(ArrayXXd q) {
  ArrayXXd F = pow(q, 2) / 2.;

  return F;
}