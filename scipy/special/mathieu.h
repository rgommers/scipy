/* UFunc scalar kernels for Mathieu functions.
 *
 * These are based on the work of Stuart Brorson (see
 * https://github.com/scipy/xsf/pull/99), whose code has been refactored by
 * Scipy developers so that only parts fit for xsf as a library of simple
 * numerical kernels which can be used on both CPU and GPU are included in
 * xsf. The below code uses dstevr from LAPACK to calculate eigenvalues and
 * eigenvectors of symmetric tridiagonal recurrence matrices, and thus is not
 * fit for xsf.
 *
 * A full write up of the mathematics behind Stuart Brorson's implementation can
 * be found at https://github.com/brorson/ScipyMathieuPaper.
 *
 * The below kernels are stateful functors. `mathieu_xem`, a template which can
 * compute either of the angular mathieu functions `mathieu_cem` and
 * `mathieu_sem` each cache the fourier coefficients corresponding to the last
 * seen pair of `m` and `q`. Such caches live only during the process of
 * iteration for a single call to the ufunc. They do not persist between calls
 * to the ufunc.
 *
 * The ufuncs corresponding to these stateful kernels which perform caching
 * should be wrapped with `scipy._ufunc_tools._with_cache_optimization` so that
 * ufunc iteration will proceed in the optimal order for cache reuse (with `m`
 * and `q` varying most slowly and `x` varying most quickly). `mathieu_cv`, a
 * template which can compute either of the characteristic value functions
 * `mathieu_a` and `mathieu_b` is also a stateful functor, but it performs no
 * caching and thus does not need
 * `scipy._ufunc_tools._with_cache_optimization`. It is stateful so that it can
 * hold on to `std::vector` instances, to resize and reuse as needed, rather
 * than created and destroyed in each ufunc iteration.  */

#pragma once

#include <algorithm>
#include <cmath>
#include <exception>
#include <limits>
#include <new>
#include <vector>

#include <xsf/error.h>
#include <xsf/mathieu.h>
#include <xsf/mathieu_legacy.h>

#include "mdspan_helpers.h"
#include "tridiagonal.h"

namespace special {

/* Largest number of terms of the Fourier series we're willing to sum. Beyond
 * this the computation is reported as SF_ERROR_NO_RESULT rather than attempted.
 */
constexpr int MAX_PARTIAL_SUM_N = 100000;

/* Stateful functor for mathieu characteristic values
 *
 * Computes characteristic values of the even angular mathieu function ce
 * when the template argument FuncParity = xsf::mathieu::Parity::Even and
 * of the odd angular mathieu function se when FuncParity =
 * xsf::mathieu::Parity::Odd.
 */
template <xsf::mathieu::Parity FuncParity, typename T> struct mathieu_cv {

  std::vector<double> D;
  std::vector<double> E;
  /* The solver is itself a stateful functor so that it can hold on to
   * std::vectors to resize and reuse as needed. */
  eigvalsh_tridiagonal solver;

  T operator()(T m, T q) {
    using namespace xsf::mathieu;
    auto constexpr Even = Parity::Even;
    auto constexpr Odd = Parity::Odd;
    constexpr const char *name = (FuncParity == Even) ? "mathieu_a" : "mathieu_b";

    /* ``m`` must be a non-negative (positive for the odd functions) integer,
     * and both arguments must be finite. Note that ``m != floor(m)`` does not
     * rule out infinities, and that a non-finite ``m`` would make the cast to
     * CBLAS_INT below undefined behavior. */
    bool domain_error =
        !std::isfinite(m) || (m != std::floor(m)) || !std::isfinite(q);
    if constexpr (FuncParity == Even) {
      domain_error = domain_error || (m < 0);
    } else {
      domain_error = domain_error || (m <= 0);
    }
    if (domain_error) {
      xsf::set_error(name, SF_ERROR_DOMAIN, NULL);
      return std::numeric_limits<T>::quiet_NaN();
    }

    if (m > 500) {
      // Fallback to Zhang and Jin implementation for very high order.
      if constexpr (FuncParity == Even) {
        return xsf::cem_cva(m, q);
      }
      return xsf::sem_cva(m, q);
    }

    auto int_m = static_cast<CBLAS_INT>(m);
    /* Use the same truncation as the Fourier coefficient computation in
     * mathieu_coeffs; a q independent truncation loses many digits for large
     * |q|. */
    auto N = static_cast<CBLAS_INT>(
        get_partial_sum_N(static_cast<int>(int_m), static_cast<double>(q)));

    try {
      // Make sure allocation actually succeeds.
      D.resize(N);
      E.resize(N - 1);
    } catch (const std::exception &) {
      xsf::set_error(name, SF_ERROR_MEMORY, NULL);
      return std::numeric_limits<T>::quiet_NaN();
    }

    // Generate recurrence matrix.
    if (int_m % 2) {
      make_matrix<FuncParity, Odd>(q, as_mdspan(D), as_mdspan(E));
    } else {
      make_matrix<FuncParity, Even>(q, as_mdspan(D), as_mdspan(E));
    }

    auto idx = cv_index<FuncParity>(int_m);
    double eigenvalue;
    auto status = solver(D, E, idx, eigenvalue);
    if (status != SF_ERROR_OK) {
      xsf::set_error(name, status, NULL);
      return std::numeric_limits<T>::quiet_NaN();
    }

    // Pull out the characteristic value from among the eigenvalues.
    return static_cast<T>(eigenvalue);
  }
};

/* Compute fourier coefficients of mathieu functions. These are the values of
 * the eigenvector corresponding to the characteristic value. FuncParity works
 * as it did for mathieu_cv.
 *
 * Like mathieu_cv, this is stateful so that it can reuse std::vector instances,
 * but no caching is performed here.
 */
template <xsf::mathieu::Parity FuncParity> struct mathieu_coeffs {
  std::vector<double> D;
  std::vector<double> E;
  eigh_tridiagonal solver;

  sf_error_t operator()(int m, double q, std::vector<double> &X) {
    using namespace xsf::mathieu;
    auto constexpr Even = Parity::Even;
    auto constexpr Odd = Parity::Odd;

    auto N = X.size();

    if (N == 0) {
      return SF_ERROR_OK;
    }

    try {
      // Make sure allocation actually succeeds.
      D.resize(N);
      E.resize(N - 1);
    } catch (const std::exception &) {
      return SF_ERROR_MEMORY;
    }

    // Generate recurrence matrix.
    if (m % 2) {
      make_matrix<FuncParity, Odd>(q, as_mdspan(D), as_mdspan(E));
    } else {
      make_matrix<FuncParity, Even>(q, as_mdspan(D), as_mdspan(E));
    }

    auto idx = cv_index<FuncParity>(m);
    double eigenvalue;
    auto status = solver(D, E, idx, X, eigenvalue);
    if (status != SF_ERROR_OK) {
      return status;
    }

    if constexpr (FuncParity == Even) {
      if (m % 2 == 0) {
        /* This normalization step is required in the even/even case. */
        X[0] /= M_SQRT2;
      }
    }

    return SF_ERROR_OK;
  }
};

/* Computes even and odd angular mathieu functions ce and se depending on
 * template parameter FuncParity
 *
 * This caches the fourier coefficients so that they can be reused as x varies
 * while m and q stay fixed during the course of ufunc iteration. A ufunc using
 * this kernel should use scipy.special._ufunc_tools._with_cache_optimization.
 * Following SciPy's longstanding unorthodox behavior, the angle x is taken in
 * units of degrees.
 */
template <xsf::mathieu::Parity FuncParity, typename T> struct mathieu_xem {
  double last_q = std::numeric_limits<double>::quiet_NaN();
  int last_m = -1;
  mathieu_coeffs<FuncParity> get_coefs;
  std::vector<double> coefs;
  void operator()(T m, T q, T x, T &out, T &out_diff) {
    using namespace xsf::mathieu;
    auto constexpr Even = Parity::Even;
    auto constexpr Odd = Parity::Odd;
    constexpr const char *name =
        (FuncParity == Even) ? "mathieu_cem" : "mathieu_sem";

    double q_d = static_cast<double>(q);
    double x_d = static_cast<double>(x);
    double out_d, out_diff_d;

    /* ``m`` must be a non-negative integer and ``q`` must be finite. Note that
     * ``m != floor(m)`` does not rule out infinities, and that a non-finite
     * ``m`` or ``q`` would make the cast to int below, respectively the
     * computation of the truncation size in get_partial_sum_N, undefined
     * behavior. */
    if ((m < 0) || !std::isfinite(m) || m != std::floor(m) ||
        !std::isfinite(q) || std::isnan(x)) {
      out = std::numeric_limits<T>::quiet_NaN();
      out_diff = std::numeric_limits<T>::quiet_NaN();
      xsf::set_error(name, SF_ERROR_DOMAIN, NULL);
      last_m = -1; // invalidate cache upon error
      return;
    }

    /* Orders this large always exceed the truncation size limit checked below.
     * Rejecting them here keeps the cast to int well defined. */
    if (m > MAX_PARTIAL_SUM_N) {
      out = std::numeric_limits<T>::quiet_NaN();
      out_diff = std::numeric_limits<T>::quiet_NaN();
      xsf::set_error(name, SF_ERROR_NO_RESULT, NULL);
      last_m = -1; // invalidate cache upon error
      return;
    }

    auto int_m = static_cast<int>(m);
    if constexpr (FuncParity == Odd) {
      if (int_m == 0) {
        out = static_cast<T>(0);
        out_diff = static_cast<T>(0);
        last_m = -1; // invalidate cache since no coefs in this case.
        return;
      }
    }

    /* Check if either q or m has changed, and if so recompute the fourier
     * coefficients. */
    if (q_d != last_q || int_m != last_m) {
      // Chooses the number of terms of the Fourier series to sum.
      auto N = get_partial_sum_N(int_m, q_d);
      if (N <= 0 || N > MAX_PARTIAL_SUM_N) {
        out = std::numeric_limits<T>::quiet_NaN();
        out_diff = std::numeric_limits<T>::quiet_NaN();
        xsf::set_error(name, SF_ERROR_NO_RESULT, NULL);
        last_m = -1; // invalidate cache upon error
        return;
      }

      try {
        // Make sure allocation actually succeeds.
        coefs.resize(N);
      } catch (const std::exception &) {
        out = std::numeric_limits<T>::quiet_NaN();
        out_diff = std::numeric_limits<T>::quiet_NaN();
        xsf::set_error(name, SF_ERROR_MEMORY, NULL);
        last_m = -1; // invalidate cache upon error
        return;
      }

      auto status = get_coefs(int_m, q_d, coefs);
      if (status != SF_ERROR_OK) {
        out = std::numeric_limits<T>::quiet_NaN();
        out_diff = std::numeric_limits<T>::quiet_NaN();
        last_m = -1; // invalidate cache upon error
        xsf::set_error(name, status, NULL);
        return;
      }
      last_q = q_d;
      last_m = int_m;
    }
    /* Compute mathieu function and its derivative by summing the fourier
     * series. */
    if (int_m % 2) {
      sum_fourier_series<FuncParity, Odd, AngleUnitPolicy::Degrees>(
          as_mdspan(coefs), x_d, out_d, out_diff_d);
    } else {
      sum_fourier_series<FuncParity, Even, AngleUnitPolicy::Degrees>(
          as_mdspan(coefs), x_d, out_d, out_diff_d);
    }
    out = static_cast<T>(out_d);
    out_diff = static_cast<T>(out_diff_d);
  }
};

// Non-stateful overloads for use in Cython special.
inline double mathieu_a(double m, double q) {
  return mathieu_cv<xsf::mathieu::Parity::Even, double>{}(m, q);
}

inline double mathieu_b(double m, double q) {
  return mathieu_cv<xsf::mathieu::Parity::Odd, double>{}(m, q);
}

inline void mathieu_cem(double m, double q, double x, double &out,
                        double &out_diff) {
  return mathieu_xem<xsf::mathieu::Parity::Even, double>{}(m, q, x, out,
                                                           out_diff);
}

inline void mathieu_sem(double m, double q, double x, double &out,
                        double &out_diff) {
  return mathieu_xem<xsf::mathieu::Parity::Odd, double>{}(m, q, x, out,
                                                          out_diff);
}

} // namespace special
