#define EIGEN_FFTW_DEFAULT
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <Eigen/Dense>
#include <unsupported/Eigen/FFT>
#include <math.h>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace Eigen;

const double pi{ 3.1415926535897 };
const size_t nx{ 10 };
const size_t ns{ 5 };
const double lx{ 2*pi };
const double dx{ lx / nx };

using namespace std::literals::complex_literals;

ArrayXXd perbc(ArrayXXd q, const size_t nx);

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


int main() {
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
  std::cout << "u:\n" << u << std::endl;
  std::cout << "u.maxCoeff(): " << u.maxCoeff() << std::endl;

  q(seqN(2, nx), all) = u;
  q = perbc(q, nx);

  ArrayXXd c_ip12 = ArrayXXd::Zero(q.rows(), q.cols());
  for (size_t s{}; s<q.cols(); s++) {
    for (size_t i{ 2 }; i<nx+2; i++) {
      c_ip12(i, s) = abs(q)(seq(i-2, i+3), s).maxCoeff();
    }
  }
  std::cout << "c_ip12:\n" << c_ip12 << std::endl;

  const double cfl = 0.5;
  std::cout << "cfl * dx / c_ip12:\n" << (cfl * dx / c_ip12) << std::endl;
  std::cout << "dt: " << (cfl * dx / c_ip12).minCoeff() << std::endl;

  std::cout << "q:\n" << q << std::endl;
  auto i = ArrayXi::LinSpaced(3, 0, 3);
  for (size_t n{}; n<(q.rows()-3); n++) {
    std::cout << "q[" << n << ":" << n+3 << ", :]:\n" << q(i+n, all) << std::endl;
  }
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