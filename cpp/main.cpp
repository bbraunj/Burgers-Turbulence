#include <cmath>
#include <iostream>
#include <vector>


template <typename T>
std::pair<std::vector<T>, std::vector<T>> weno_5( std::vector<T> q, const size_t nx ) {
  using vecT = std::vector<T>;
  vecT b0      (nx+5, 0);
  vecT b1      (nx+5, 0);
  vecT b2      (nx+5, 0);

  vecT a0      (nx+5, 0);
  vecT a1      (nx+5, 0);
  vecT a2      (nx+5, 0);

  vecT w0      (nx+5, 0);
  vecT w1      (nx+5, 0);
  vecT w2      (nx+5, 0);

  vecT q0      (nx+5, 0);
  vecT q1      (nx+5, 0);
  vecT q2      (nx+5, 0);

  vecT qL_ip12 (nx+5, 0);
  vecT qR_ip12 (nx+5, 0);

  // Linear weighting coefficients
  float d0 = 1/10;
  float d1 = 6/10;
  float d2 = 3/10;
  float eps = 1e-6;

  for (size_t i{ 2 }; i<nx+3; i++) {
    // Smoothness indicators
    b0[i] = ((13/12)*pow( q[i-2] - 2*q[i-1] +   q[i], 2 )
             + (1/4)*pow( q[i-2] - 4*q[i-1] + 3*q[i], 2 ));
    b1[i] = ((13/12)*pow( q[i-1] - 2*q[i] + q[i+1], 2 )
             + (1/4)*pow( q[i-1] - q[i+1], 2 ));
    b2[i] = ((13/12)*pow(   q[i] - 2*q[i+1] + q[i+2], 2 )
             + (1/4)*pow( 3*q[i] - 4*q[i+1] + q[i+2], 2 ));
  }

  // Positive reconstruction @ i+1/2
  for (size_t i{ 2 }; i<nx+2; i++) {
    // Nonlinear weights
    a0[i] = d0 / pow( b0[i] + eps, 2 );
    a1[i] = d1 / pow( b1[i] + eps, 2 );
    a2[i] = d2 / pow( b2[i] + eps, 2 );

    T a_sum{ a0[i] + a1[i] + a2[i] };
    w0[i] = a0[i] / a_sum;
    w1[i] = a1[i] / a_sum;
    w2[i] = a2[i] / a_sum;

    // Positive reconstruction
    q0[i] = 2*q[i-2] - 7*q[i-1] + 11*q[i];
    q1[i] = -q[i-1]  + 5*q[i]   + 2*q[i+1];
    q2[i] = 2*q[i]   + 5*q[i+1] - q[i+2];
    qL_ip12[i] = (w0[i] / 6)*q0[i]
               + (w1[i] / 6)*q1[i]
               + (w2[i] / 6)*q2[i];
  }

  // Negative reconstruction @ i-1/2 + 1 (so i+1/2)
  for (size_t i{ 3 }; i<nx+3; i++) {
    // Nonlinear weights
    a0[i] = d0 / pow( b0[i] + eps, 2 );
    a1[i] = d1 / pow( b1[i] + eps, 2 );
    a2[i] = d2 / pow( b2[i] + eps, 2 );

    T a_sum{ a0[i] + a1[i] + a2[i] };
    w0[i] = a0[i] / a_sum;
    w1[i] = a1[i] / a_sum;
    w2[i] = a2[i] / a_sum;

    // Positive reconstruction
    q0[i] = 2*q[i+2] - 7*q[i+1] + 11*q[i];
    q1[i] = -q[i+1]  + 5*q[i]   + 2*q[i-1];
    q2[i] = 2*q[i]   + 5*q[i-1] - q[i-2];
    qR_ip12[i-1] = (w0[i] / 6)*q0[i]
                 + (w1[i] / 6)*q1[i]
                 + (w2[i] / 6)*q2[i];
  }
}


template <typename T>
std::vector<T> TDMAsolver(
    std::vector<T> a,
    std::vector<T> b,
    std::vector<T> c,
    std::vector<T> d
    ) {

  const size_t n = a.size();

  T m;
  for (size_t i{ 1 }; i<n; i++) {
    m = a[i-1] / b[i-1];
    b[i] = b[i] - m*c[i-1];
    d[i] = d[i] - m*d[i-1];
  }

  std::vector<T> x( b );
  x[n-1] = d[n-1] / b[n-1];

  for (size_t i{ n-2 }; i>=0; i--) {
    x[i] = (d[i] - c[i]*x[i+1]) / b[i];
  }

  return x;
}


template <typename T>
std::vector<T> TDMA_cyclic(
    std::vector<T> a,
    std::vector<T> b,
    std::vector<T> c,
    std::vector<T> d,
    const T alpha, const T beta
    ) {

  const size_t n = a.size();

  const T gamma = 10*b[0];
  b[0] = b[0] - gamma;
  b[n-1] = b[n-1] - alpha*beta/gamma;
  std::vector<T> y = TDMAsolver(a, b, c, d);

  std::vector<T> u(n, 0);
  u[0] = gamma;
  u[n-1] = alpha;
  std::vector<T> q = TDMAsolver(a, b, c, u);

  std::vector<T> x(n, 0);
  for (size_t i{}; i<n; i++) {
    x[i] = y[i] - q[i]*((y[0] + (beta/gamma)*y[n-1]) /
                        (1 + q[0] + (beta/gamma)*q[n-1]));
  }

  return x;
}


template <typename T>
std::vector<std::vector<T>> rhs(
    std::vector<std::vector<T>> q,
    const size_t ns,
    const size_t nx,
    const T dx,
    const T nu
    ) {

  using vecT = std::vector<T>;

  for (size_t s{}; s<ns; s++) {
    vecT& qs = &q[s];

    auto& [qL_ip12, qR_ip12] = weno_5(qs, nx);

    vecT c_ip12 = get_wave_speeds(q, nx);
    vecT FR = get_flux(qR_ip12);
    vecT FL = get_flux(qL_ip12);

    // Riemann Solver
    vecT F_ip12 = rusanov_riemann(FR, FL, qR_ip12, qL_ip12, c_ip12);

    // Viscous contribution
    vecT uxx = c4ddp(qs.begin()+2, qs.cend(), nx+1, dx);

    vecT rhs(nx, 0);
    for (size_t i{}; i<nx; i++) {
      //         F_ip12        F_im12
      rhs[i] -= (F_ip12[i+3] - F_ip12[i+2]) / dx;
      rhs[i] += nu*uxx[i+1];
    }
  }
}


int main() {
  using namespace std;
  cout << "Hello World!";
}
