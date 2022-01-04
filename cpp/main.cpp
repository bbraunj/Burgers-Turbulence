#include <vector>
#include <iostream>


template <typename T>
std::vector<T> TDMAsolver(
    std::vector<T> a,
    std::vector<T> b,
    std::vector<T> c,
    std::vector<T> d
    ) {

  const size_t n = a.size();

  T m;
  for (size_t i=1; i<n; i++) {
    m = a[i-1] / b[i-1];
    b[i] = b[i] - m*c[i-1];
    d[i] = d[i] - m*d[i-1];
  }

  std::vector<T> x( b );
  x[n-1] = d[n-1] / b[n-1];

  for (size_t i=n-2; i>=0; i--) {
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
  for (size_t i=0; i<n; i++) {
    x[i] = y[i] - q[i]*((y[0] + (beta/gamma)*y[n-1]) /
                        (1 + q[0] + (beta/gamma)*q[n-1]));
  }

  return x;
}


int main() {
  using namespace std;
  cout << "Hello World!";
}
