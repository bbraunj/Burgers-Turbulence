#include <cmath>
#include <complex>
#include <math.h>
#include <iostream>
#include <vector>
#include <xtensor/xadapt.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xcomplex.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xview.hpp>
#include <xtensor-fftw/basic.hpp>

const double pi{ 3.1415926535897 };
const size_t nx{ 10 };
const size_t ns{ 5 };
const double lx{ 2*pi };
const double dx{ lx / nx };

using namespace std::literals::complex_literals;

int main() {
//   using xarray = xt::xarray<double>;
  // xt::xarray<double> s = xt::arange<double>(0, nx);
  // xt::xarray<double> b = xt::arange<double>(0, nx);
  // s.reshape({1, nx});
  // b.reshape({1, nx});
  // for (size_t i{ 1 }; i<ns; i++) {
  //     s = xt::concatenate(xt::xtuple(s, b));
  // }
  // std::cout << xt::adapt(s.shape()) << std::endl;
  // std::cout << s << std::endl;

//   xt::xarray<double> s2 = xt::zeros<double>({nx, ns});
//   for (size_t i{}; i<nx; i++) {
//       xt::view(s2, i, xt::all()) = xt::ones<double>({ns})*i;
//   }
//   std::cout << xt::adapt(s2.shape()) << std::endl;
//   std::cout << s2 << std::endl;

//   xarray r = xt::random::rand<double>({nx, ns});
//   std::cout << xt::adapt(r.shape()) << std::endl;
//   std::cout << r << std::endl;

  using xarray = xt::xarray<double>;
  // nx+5 because 2 ghost points at lhs of domain, 3 ghost points at rhs of domain
  xarray q = xt::zeros<double>({nx+5, ns});
  double lx = nx*dx;

  // Wave numbers
  xarray s = xt::zeros<double>({nx/2, ns});
  for (size_t i{}; i<nx/2; i++) {
    xt::view(s, i, xt::all()) = xt::ones<double>({ns})*i;
  }

  xarray kx = xt::zeros<double>({nx+5, ns});
  xt::view(kx, xt::range(2, (nx/2 + 2)), xt::all())        = 2*pi*s / lx;
  xt::view(kx, xt::range((nx/2 + 2), (nx + 2)), xt::all()) = 2*pi*(s - nx/2) / lx;

  // Initial energy spectrum in wavenumber space
  const size_t k0 = 10;
  auto A = ( 2 / (3*sqrt(pi)) ) * pow(k0, -5);
  xarray E_k = A*xt::pow(kx, 4) * xt::exp(-xt::pow(kx/k0, 2));

  // Velocity in Fourier space
  xarray r = xt::random::rand<double>({nx, ns});
  xt::view(r, xt::range(0, nx/2), xt::all()) = -xt::view(r, xt::range(0, nx/2), xt::all());
  xt::xarray<std::complex<double>> u_k = xt::sqrt(2*xt::view(E_k, xt::range(2, nx+2), xt::all()))
                                       * xt::exp(2.0i * pi * r);

  // Velocity in physical space
  xarray u = xt::zeros<double>({nx, ns});
  for (size_t i{}; i<ns; i++) {
    xt::xarray<std::complex<double>> u_ki = xt::view(u_k, xt::all(), i);
    xt::view(u, xt::all(), i) = xt::real(xt::fftw::fft(u_ki));
  }

  xt::view(q, xt::range(2, nx+2), xt::all()) = u;
//   q = perbc(q, nx);

//   std::cout << xt::adapt(u.shape()) << std::endl;
  std::cout << "s:\n" << s << std::endl;
  std::cout << "kx:\n" << kx << std::endl;
  std::cout << "E_k:\n" << std::scientific << E_k << std::endl;
  std::cout << "u_k:\n" << u_k << std::endl;
  std::cout << "u:\n" << u << std::endl;
  std::cout << "q:\n" << q << std::endl;

  xarray c_ip12 = xt::zeros<double>(q.shape());
  xarray test = xt::vstack(xt::xtuple(
    xarray(xt::view(q, xt::range(0, nx), xt::all())),
    xarray(xt::view(q, xt::range(1, nx+1), xt::all())),
    xarray(xt::view(q, xt::range(2, nx+2), xt::all())),
    xarray(xt::view(q, xt::range(3, nx+3), xt::all())),
    xarray(xt::view(q, xt::range(4, nx+4), xt::all())),
    xarray(xt::view(q, xt::range(5, nx+5), xt::all()))
  ));
  xt::view(c_ip12, xt::range(2, nx+2), xt::all()) = xt::amax(test, {0});
  std::cout << "test" << xt::adapt(test.shape()) << std::endl;
  std::cout << test << std::endl;
  std::cout << "c_ip12" << xt::adapt(c_ip12.shape()) << std::endl;
  std::cout << c_ip12 << std::endl;

  const double cfl = 0.5;
  auto dt = xt::nanmin<double>(cfl * dx / c_ip12);
  std::cout << (cfl * dx / c_ip12) << std::endl;
  std::cout << "dt = " << dt << std::endl;
}