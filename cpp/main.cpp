#include <chrono>
#include <cmath>
#include <complex>
#include <iomanip>
#include <iostream>
#include <math.h>
#include <set>
#include <utility>
#include <xtensor/xarray.hpp>
#include <xtensor/xcomplex.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xmath.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xview.hpp>
#include <xtensor-fftw/basic.hpp>


// Function declarations
// ---------------------
template <typename T>
void outer_loop(const size_t nx, const size_t ns, const T nu, const T tmax);
template <typename T>
xt::xarray<T> tvdrk3(xt::xarray<T> q_n, const size_t nx, const T dx, const T dt, const T nu);
template <typename T>
xt::xarray<T> rhs(xt::xarray<T> q, const size_t nx, const T dx, const T nu);
template <typename T>
xt::xarray<T> rusanov_riemann(xt::xarray<T> FR,
                              xt::xarray<T> FL,
                              xt::xarray<T> qR_ip12,
                              xt::xarray<T> qL_ip12,
                              xt::xarray<T> c_ip12);
template <typename T>
xt::xarray<T> perbc(xt::xarray<T> q, const size_t nx);

// Reconstruction Scheme
template <typename T>
std::tuple<xt::xarray<T>, xt::xarray<T>> weno_5(xt::xarray<T> q, const size_t nx);

// Viscous Contribution
template <typename T>
xt::xarray<T> c4ddp(xt::xarray<T> f, const size_t N, const T h);
template <typename T>
xt::xarray<T> TDMA_cyclic(xt::xarray<T> a, xt::xarray<T> b, xt::xarray<T> c, xt::xarray<T> d, const T alpha, const T beta);
template <typename T>
xt::xarray<T> TDMAsolver(xt::xarray<T> a, xt::xarray<T> b, xt::xarray<T> c, xt::xarray<T> d);

// Problem Quantities
template <typename T>
xt::xarray<T> get_IC_q(const size_t ns, const size_t nx, const T dx);
template <typename T>
T get_dt(xt::xarray<T> q, const size_t nx, const T dx);
template <typename T>
xt::xarray<T> get_wave_speeds(xt::xarray<T> q, const size_t nx);
template <typename T>
xt::xarray<T> get_flux(xt::xarray<T> q);
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
  xt::xarray<T> q = get_IC_q(ns, nx, dx);
  T time{ 0 };

  size_t n_iterations{};
  std::cout << "\nSolving Burgers Problem Using WENO-5 Reconstruction";
  std::cout << "\nnx=" << nx << ", ns=" << ns << ", nu=" << nu << ", tmax=" << tmax << std::endl;
  T dt;
  while (time < tmax) {
    n_iterations++;
    // std::cout << "Before get_dt" << std::endl;
    dt = get_dt(q, nx, dx);
    std::cout << "    * Calculated dt: " << dt << std::endl;

    // Make sure we don't go past tmax
    dt = (time + dt) < tmax ? dt : tmax - time;
    // Also make sure we get a data point at 0.05 s
    dt = (time < 0.05 && time+dt > 0.05) ? (0.05 - time) : dt;
    time += dt;
    std::cout << "  * time: " << std::setprecision(4) << time << " s" << ", dt: " << dt << " s" << std::endl;

    // Advance a time step using TVDRK3 time integration
    q = tvdrk3(q, nx, dx, dt, nu);

  }

  std::cout << "\n  * Done (" << n_iterations << " iterations)" << std::endl;
}


template <typename T>
xt::xarray<T> tvdrk3(xt::xarray<T> q_n, const size_t nx, const T dx, const T dt, const T nu) {
  // Stopwatch stopwatch{ "TVDRK3" };
  // 3rd Order Runge-Kutta time integrator.
  // std::cout << "tvdrk3:" << std::endl;
  xt::xarray<T> q1( q_n );
  auto i = xt::range(3, nx+2);
  xt::view(q1, i, xt::all()) = (
    xt::view(q_n, i, xt::all()) +
    dt * rhs(q_n, nx, dx, nu)
  );
  // std::cout << "* q1: " << q1(5, 0) << std::endl;
  q1 = perbc(q1, nx);

  xt::xarray<T> q2( q_n );
  xt::view(q2, i, xt::all()) = (
    0.75 * xt::view(q_n, i, xt::all()) +
    0.25 * xt::view(q1,  i, xt::all()) +
    0.25 * dt * rhs(q1, nx, dx, nu)
  );
  // std::cout << "* q2: " << q2(5, 0) << std::endl;
  q2 = perbc(q2, nx);

  xt::xarray<T> q_np1( q_n );
  xt::view(q_np1, i, xt::all()) = (
    (1./3) * xt::view(q_n, i, xt::all()) +
    (2./3) * xt::view(q2,  i, xt::all()) +
    (2./3) * dt * rhs(q2, nx, dx, nu)
  );
  // std::cout << "* q_np1: " << q_np1(5, 0) << std::endl;
  q_np1 = perbc(q_np1, nx);

  return q_np1;
}


template <typename T>
xt::xarray<T> rhs(xt::xarray<T> q, const size_t nx, const T dx, const T nu) {
  // Note:
  //   q.shape() == {nx+5, ns}
  //                 ^^^^ Specifically from 0:(nx+5)
  //   rhs.shape() == {nx-1, ns}
  //                   ^^^^ Specifically from 1:(nx+2)

  // Stopwatch stopwatch{ "rhs" };
  // Reconstruction Scheme
  auto [qL_ip12, qR_ip12] = weno_5(q, nx);
  // xt::xarray<T> [qL_ip12, qR_ip12] = weno_5(q, nx);

  xt::xarray<T> c_ip12 = get_wave_speeds(q, nx);
  xt::xarray<T> FR = get_flux(qR_ip12);
  xt::xarray<T> FL = get_flux(qL_ip12);
  // std::cout << "  + FL: " << FL(5, 0) << std::endl;
  // std::cout << "  + FR: " << FR(5, 0) << std::endl;

  // Riemann Solvers
  xt::xarray<T> F_ip12 = rusanov_riemann(FR, FL, qR_ip12, qL_ip12, c_ip12);
  // std::cout << "  + F_ip12: " << F_ip12(5, 0) << std::endl;

  xt::xarray<T> rhs = -(
      xt::view(F_ip12, xt::range(3, nx+2), xt::all())  // F_ip12
    - xt::view(F_ip12, xt::range(2, nx+1), xt::all())  // F_im12
  ) / dx;

  // Viscous contribution
  xt::xarray<T> uxx = c4ddp(
    static_cast<xt::xarray<T>>(xt::view(q, xt::range(2, nx+3), xt::all())),
    nx+1, dx
  );
  // std::cout << "  + uxx: " << uxx(5, 0) << std::endl;

  // rhs += nu*xt::view(uxx, xt::range(1, nx), xt::all());

  return rhs;
}

template <typename T>
xt::xarray<T> rusanov_riemann(xt::xarray<T> FR,
                              xt::xarray<T> FL,
                              xt::xarray<T> qR_ip12,
                              xt::xarray<T> qL_ip12,
                              xt::xarray<T> c_ip12) {
  xt::xarray<T> F_ip12 = 0.5*((FR + FL) - c_ip12*(qR_ip12 - qL_ip12));

  return F_ip12;
}

template <typename T>
std::tuple<xt::xarray<T>, xt::xarray<T>> weno_5(xt::xarray<T> q, const size_t nx) {
  // Stopwatch stopwatch{ "WENO-5" };
  const auto q_shape = q.shape();
  const auto domain_shape = {q.shape()[0]-5, q.shape()[1]};
  xt::xarray<T> b0      = xt::zeros<T>(q_shape);
  xt::xarray<T> b1      = xt::zeros<T>(q_shape);
  xt::xarray<T> b2      = xt::zeros<T>(q_shape);

  xt::xarray<T> a0      = xt::zeros<T>(domain_shape);
  xt::xarray<T> a1      = xt::zeros<T>(domain_shape);
  xt::xarray<T> a2      = xt::zeros<T>(domain_shape);

  xt::xarray<T> w0      = xt::zeros<T>(domain_shape);
  xt::xarray<T> w1      = xt::zeros<T>(domain_shape);
  xt::xarray<T> w2      = xt::zeros<T>(domain_shape);

  xt::xarray<T> qL_ip12 = xt::zeros<T>(q_shape);
  xt::xarray<T> qR_ip12 = xt::zeros<T>(q_shape);

  // Linear weighting coefficients
  T d0 = 1./10;
  T d1 = 6./10;
  T d2 = 3./10;
  T eps = 1e-6;

  // Smoothness indicators
  xt::view(b0, xt::range(2, nx+3), xt::all()) = (
    (13./12)*xt::square((    xt::view(q, xt::range(0, nx+1), xt::all())  // i-2
                         - 2*xt::view(q, xt::range(1, nx+2), xt::all())  // i-1
                         +   xt::view(q, xt::range(2, nx+3), xt::all())  // i
                        ))
    + (1./4)*xt::square((    xt::view(q, xt::range(0, nx+1), xt::all())  // i-2
                         - 4*xt::view(q, xt::range(1, nx+2), xt::all())  // i-1
                         + 3*xt::view(q, xt::range(2, nx+3), xt::all())  // i
                        ))
  );
  xt::view(b1, xt::range(2, nx+3), xt::all()) = (
    (13./12)*xt::square((    xt::view(q, xt::range(1, nx+2), xt::all())  // i-1
                         - 2*xt::view(q, xt::range(2, nx+3), xt::all())  // i
                         +   xt::view(q, xt::range(3, nx+4), xt::all())  // i+1
                         ))
    + (1./4)*xt::square((  xt::view(q, xt::range(1, nx+2), xt::all())    // i-1
                         - xt::view(q, xt::range(3, nx+4), xt::all())    // i+1
                        ))
  );
  xt::view(b2, xt::range(2, nx+3), xt::all()) = (
    (13./12)*xt::square((    xt::view(q, xt::range(2, nx+3), xt::all())  // i
                         - 2*xt::view(q, xt::range(3, nx+4), xt::all())  // i+1
                         +   xt::view(q, xt::range(4, nx+5), xt::all())  // i+2
                        ))
    + (1./4)*xt::square((  3*xt::view(q, xt::range(2, nx+3), xt::all())  // i
                         - 4*xt::view(q, xt::range(3, nx+4), xt::all())  // i+1
                         +   xt::view(q, xt::range(4, nx+5), xt::all())  // i+2
                        ))
  );

  // Nonlinear weights
  // const auto i = xt::range(2, nx+2);
  a0 = d0 / xt::square(xt::view(b0, xt::range(2, nx+2), xt::all()) + 1e-6);
  a1 = d1 / xt::square(xt::view(b1, xt::range(2, nx+2), xt::all()) + 1e-6);
  a2 = d2 / xt::square(xt::view(b2, xt::range(2, nx+2), xt::all()) + 1e-6);

  xt::xarray<T> a_sum = (a0 + a1 + a2);
  w0 = a0 / a_sum;
  w1 = a1 / a_sum;
  w2 = a2 / a_sum;

  // Positive reconstruction @ i+1/2
  xt::xarray<T> q0 = (
       2.*xt::view(q, xt::range(0, nx),   xt::all())  // i-2
    -  7.*xt::view(q, xt::range(1, nx+1), xt::all())  // i-1
    + 11.*xt::view(q, xt::range(2, nx+2), xt::all())  // i
  );
  xt::xarray<T> q1 = (
        -xt::view(q, xt::range(1, nx+1), xt::all())   // i-1
    + 5.*xt::view(q, xt::range(2, nx+2), xt::all())   // i
    + 2.*xt::view(q, xt::range(3, nx+3), xt::all())   // i+1
  );
  xt::xarray<T> q2 = (
      2.*xt::view(q, xt::range(2, nx+2), xt::all())   // i
    + 5.*xt::view(q, xt::range(3, nx+3), xt::all())   // i+1
    -    xt::view(q, xt::range(4, nx+4), xt::all())   // i+2
  );
  xt::view(qL_ip12, xt::range(2, nx+2), xt::all()) = (w0/6)*q0 + (w1/6)*q1 + (w2/6)*q2;

  // Negative reconstruction @ i-1/2 + 1 (so i+1/2)
  // const auto i2 = xt::range(3, nx+3);
  a0 = d0 / xt::square(xt::view(b0, xt::range(3, nx+3), xt::all()) + eps);
  a1 = d1 / xt::square(xt::view(b1, xt::range(3, nx+3), xt::all()) + eps);
  a2 = d2 / xt::square(xt::view(b2, xt::range(3, nx+3), xt::all()) + eps);

  a_sum = (a0 + a1 + a2);
  w0 = a0 / a_sum;
  w1 = a1 / a_sum;
  w2 = a2 / a_sum;

  q0 = (
       2.*xt::view(q, xt::range(4, nx+4), xt::all())  // i2+2
    -  7.*xt::view(q, xt::range(3, nx+3), xt::all())  // i2+1
    + 11.*xt::view(q, xt::range(2, nx+2), xt::all())  // i2
  );
  q1 = (
        -xt::view(q, xt::range(3, nx+3), xt::all())   // i2+1
    + 5.*xt::view(q, xt::range(2, nx+2), xt::all())   // i2
    + 2.*xt::view(q, xt::range(1, nx+1), xt::all())   // i2-1
  );
  q2 = (
      2.*xt::view(q, xt::range(2, nx+2), xt::all())   // i2
    + 5.*xt::view(q, xt::range(1, nx+1), xt::all())   // i2-1
    -    xt::view(q, xt::range(0, nx),   xt::all())   // i2-2
  );
  xt::view(qR_ip12, xt::range(2, nx+2), xt::all()) = (w0/6)*q0 + (w1/6)*q1 + (w2/6)*q2;

  // std::pair<xt::xarray<T>, xt::xarray<T>> p(qL_ip12, qR_ip12);
  // std::cout << "  + qL_ip12: " << qL_ip12(5, 0) << std::endl;
  // std::cout << "  + qR_ip12: " << qR_ip12(5, 0) << std::endl;
  return std::make_tuple(qL_ip12, qR_ip12);
}


template <typename T>
xt::xarray<T> TDMA_cyclic(
    xt::xarray<T> a,
    xt::xarray<T> b,
    xt::xarray<T> c,
    xt::xarray<T> d,
    const T alpha, const T beta
    ) {

  const size_t nf = a.shape()[0];

  const T gamma = 10.*b[0];
  b[0] = b[0] - gamma;
  b[nf-1] = b[nf-1] - alpha*beta/gamma;
  xt::xarray<T> y = TDMAsolver(a, b, c, d);

  xt::xarray<T> u = xt::zeros<T>({nf});
  u[0] = gamma;
  u[nf-1] = alpha;
  xt::xarray<T> q = TDMAsolver(a, b, c, u);

  xt::xarray<T> x = y - q*((y[0] + (beta/gamma)*y[nf-1]) /
                           (1 + q[0] + (beta/gamma)*q[nf-1]));

  return x;
}


template <typename T>
xt::xarray<T> TDMAsolver(
    xt::xarray<T> a,
    xt::xarray<T> b,
    xt::xarray<T> c,
    xt::xarray<T> d
    ) {

  const size_t nf = a.shape()[0];  // number of equations

  T m;
  for (size_t i{ 1 }; i<nf; i++) {
    m = a[i-1] / b[i-1];
    b[i] = b[i] - m*c[i-1];
    d[i] = d[i] - m*d[i-1];
  }

  xt::xarray<T> x( b );
  x[nf-1] = d[nf-1] / b[nf-1];

  for (size_t i{ nf-2 }; i>0; i--) {
    x[i] = (d[i] - c[i]*x[i+1]) / b[i];
  }
  x[0] = (d[0] - c[0]*x[1]) / b[0];  // Doing i=0 in the for loop isn't working for some reason.

  return x;
}


template <typename T>
xt::xarray<T> get_IC_q(const size_t ns, const size_t nx, const T dx) {
  using xarray = xt::xarray<T>;
  using namespace std::literals::complex_literals;

  // nx+5 because 2 ghost points at lhs of domain, 3 ghost points at rhs of domain
  xarray q = xt::zeros<T>({nx+5, ns});
  T lx = nx*dx;

  // Wave numbers
  xarray s = xt::zeros<T>({nx/2, ns});
  for (size_t i{}; i<nx/2; i++) {
    xt::view(s, i, xt::all()) = xt::ones<T>({ns})*i;
  }

  xarray kx = xt::zeros<T>({nx+5, ns});
  xt::view(kx, xt::range(2, (nx/2 + 2)), xt::all())        = 2*pi*s / lx;
  xt::view(kx, xt::range((nx/2 + 2), (nx + 2)), xt::all()) = 2*pi*(s - nx/2) / lx;

  // Initial energy spectrum in wavenumber space
  const T k0 = 10.;
  const T A = ( 2. / (3.*sqrt(pi)) ) * pow(k0, -5);
  xarray E_k = A*xt::pow<4>(kx) * xt::exp(-xt::square(kx/k0));

  // Velocity in Fourier space
  xarray r = xt::random::rand<double>({nx, ns});
  xt::view(r, xt::range(0, nx/2), xt::all()) = -xt::view(r, xt::range(0, nx/2), xt::all());
  xt::xarray<std::complex<T>> u_k = xt::sqrt(2.*xt::view(E_k, xt::range(2, nx+2), xt::all()))
                                  * xt::exp(2.0i * pi * r);

  // Velocity in physical space
  xarray u = xt::zeros<double>({nx, ns});
  for (size_t i{}; i<ns; i++) {
    xt::xarray<std::complex<double>> u_ki = xt::view(u_k, xt::all(), i);
    xt::view(u, xt::all(), i) = xt::real(xt::fftw::fft(u_ki));
  }

  xt::view(q, xt::range(2, nx+2), xt::all()) = u;
  q = perbc(q, nx);

  return q;
}

template <typename T>
xt::xarray<T> perbc(xt::xarray<T> q, const size_t nx) {
  xt::view(q, 2, xt::all()) = xt::view(q, nx+1, xt::all());
  xt::view(q, 1, xt::all()) = xt::view(q, nx, xt::all());
  xt::view(q, 0, xt::all()) = xt::view(q, nx-1, xt::all());
  xt::view(q, nx+2, xt::all()) = xt::view(q, 3, xt::all());
  xt::view(q, nx+3, xt::all()) = xt::view(q, 4, xt::all());
  xt::view(q, nx+4, xt::all()) = xt::view(q, 5, xt::all());

  return q;
}

template <typename T>
xt::xarray<T> c4ddp(xt::xarray<T> f, const size_t N, const T h) {
  //
  // 4th order compact scheme 2nd derivative periodic
  //
  // f_ = f minus repeating part (f[0] = f[-1])
  //        plus last nonrepeating inserted at index 0 (f_[0] = f[-2])
  // xt::xarray<T> f_   = xt::zeros<T>(f.shape());
  // xt::view(f_, xt::range(1, N), xt::all()) = xt::view(f, xt::range(0, N-1), xt::all());
  // xt::view(f_, 0, xt::all()) = xt::view(f, N-2, xt::all());
  // Stopwatch stopwatch{ "C4DDP" };
  xt::xarray<T> f_ = xt::view(f, xt::range(0, N-1), xt::all());
  xt::xarray<T> f_pp = xt::zeros<T>(f.shape());
  // xt::xarray<T> r    = xt::zeros<T>(xt::view(f, xt::range(0, N-1), xt::all()).shape());
  xt::xarray<T> r    = xt::zeros<T>(f_.shape());

  xt::xarray<T> a = (1./10) * xt::ones<T>(r.shape());
  xt::xarray<T> b =      1. * xt::ones<T>(r.shape());
  xt::xarray<T> c = (1./10) * xt::ones<T>(r.shape());

  xt::view(r, xt::range(1, N-2), xt::all()) = (6./5)*(1./pow(h, 2))*(
         xt::view(f_, xt::range(2, N-1), xt::all())  // i+1
    - 2.*xt::view(f_, xt::range(1, N-2), xt::all())  // i
    +    xt::view(f_, xt::range(0, N-3), xt::all())  // i-1
  );
  xt::view(r, 0, xt::all()) = (6./5)*(1./pow(h, 2))*(
         xt::view(f_, 1,   xt::all())  // i+1
    - 2.*xt::view(f_, 0,   xt::all())  // i
    +    xt::view(f_, N-1, xt::all())  // i-1
  );
  xt::view(r, N-1, xt::all()) = (6./5)*(1./pow(h, 2))*(
         xt::view(f_, 0,   xt::all())  // i+1
    - 2.*xt::view(f_, N-1, xt::all())  // i
    +    xt::view(f_, N-2, xt::all())  // i-1
  );

  const T alpha = 1./10;
  const T beta = 1./10;
  const size_t dim = f.dimension();
  const size_t ns = f.shape()[dim-1];
  for (size_t s{}; s<ns; s++) {
    xt::view(f_pp, xt::range(0, N-1), s) = TDMA_cyclic(
      static_cast<xt::xarray<T>>(xt::view(a, xt::all(), s)),
      static_cast<xt::xarray<T>>(xt::view(b, xt::all(), s)),
      static_cast<xt::xarray<T>>(xt::view(c, xt::all(), s)),
      static_cast<xt::xarray<T>>(xt::view(r, xt::all(), s)),
      alpha, beta
    );
  }
  xt::view(f_pp, N-1, xt::all()) = xt::view(f_pp, 0, xt::all());

  return f_pp;
}

template <typename T>
T get_dt(xt::xarray<T> q, const size_t nx, const T dx) {
  // Stopwatch stopwatch{ "Get_dt" };
  const T cfl = 0.5;
  xt::xarray<T> c_ip12 = get_wave_speeds(q, nx);
  T dt = *xt::nanmin<T>(cfl * dx / c_ip12).begin();

  return dt;
}

template <typename T>
xt::xarray<T> get_wave_speeds(xt::xarray<T> q, const size_t nx) {
  xt::xarray<T> c_ip12 = xt::zeros<T>(q.shape());
  // xt::xarray<T> stacked_q = xt::xtuple(
  //   static_cast<xt::xarray<T>>(xt::view(q, xt::range(0, nx), xt::all())),
  //   static_cast<xt::xarray<T>>(xt::view(q, xt::range(1, nx+1), xt::all())),
  //   static_cast<xt::xarray<T>>(xt::view(q, xt::range(2, nx+2), xt::all())),
  //   static_cast<xt::xarray<T>>(xt::view(q, xt::range(3, nx+3), xt::all())),
  //   static_cast<xt::xarray<T>>(xt::view(q, xt::range(4, nx+4), xt::all())),
  //   static_cast<xt::xarray<T>>(xt::view(q, xt::range(5, nx+5), xt::all()))
  // );
  xt::xarray<T> abs_q = xt::abs(q);
  for (size_t s{}; s<q.shape()[1]; s++) {
    for (size_t i{ 2 }; i<nx+2; i++) {
      auto r = xt::view(abs_q, xt::range(i-2, i+3), s);
      std::set<T> sorted_r(r.begin(), r.end());
      c_ip12(i, s) = *--sorted_r.end();
    }
  }
  // std::cout << "q:\n" << q << std::endl;
  // std::cout << "c_ip12:\n" << c_ip12 << std::endl;
  // xt::view(c_ip12, xt::range(2, nx+2), xt::all()) = xt::amax(xt::vstack(xt::xtuple(
  //   static_cast<xt::xarray<T>>(xt::view(q, xt::range(0, nx), xt::all())),
  //   static_cast<xt::xarray<T>>(xt::view(q, xt::range(1, nx+1), xt::all())),
  //   static_cast<xt::xarray<T>>(xt::view(q, xt::range(2, nx+2), xt::all())),
  //   static_cast<xt::xarray<T>>(xt::view(q, xt::range(3, nx+3), xt::all())),
  //   static_cast<xt::xarray<T>>(xt::view(q, xt::range(4, nx+4), xt::all())),
  //   static_cast<xt::xarray<T>>(xt::view(q, xt::range(5, nx+5), xt::all()))
  // )), {0});

  return c_ip12;
}

template <typename T>
xt::xarray<T> get_flux(xt::xarray<T> q) {
  xt::xarray<T> F = xt::square(q) / 2.;

  return F;
}