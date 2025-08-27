"""
Mixed Bergomi models.
"""

import warnings
from math import factorial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import IntegrationWarning, dblquad, quad
from scipy.optimize import brentq, fsolve
from scipy.special import beta, eval_hermitenorm, hyp2f1
from scipy.stats import norm

from utils import plot_approx

"""
--- Global Constants for Numerical Stability and Clarity ---
"""

# Clipping bounds to prevent overflow/underflow errors in np.exp()
EXP_LOWER_BOUND = -700
EXP_UPPER_BOUND = 700

# Small number to prevent division by zero
EPSILON = 1e-9


"""
--- Model Setting ---
"""

# --- Mixed Bergomi Model ---


class MixedBergomi:
    """
    Implementation of the mixed Bergomi model.

    Parameters
    ----------
    omega1, omega2: float
                    Vol-of-vol for each process.
    lam: float
         Mixing weight.
    k: float
       Mean-reversion speed.
    X0: float
        Initial log-spot variance.
    T: float
       Time to maturity.
    Delta: float
           VIX derivative tenor.
    """

    def __init__(self, params):
        """
        Initialize the mixed Bergomi model.
        See class docstring for parameter definitions.
        """

        self.omega1, self.omega2, self.lam, self.k, self.X0, self.T, self.Delta = params
        self.omega = [self.omega1, self.omega2]

    def mean(self):
        """
        Compute the mean of each squared VIX proxy.
        """

        exp_term_T = 1 - np.exp(-2 * self.k * self.T)
        exp_term_Delta = 1 - np.exp(-2 * self.k * self.Delta)
        common_factor = exp_term_T * exp_term_Delta / (8 * self.k**2 * self.Delta)

        means = self.X0 - (np.array(self.omega) ** 2 * common_factor)
        return means[0], means[1]

    def variance(self):
        """
        Compute the variance of each squared VIX proxy.
        """

        exp_term_T = 1 - np.exp(-2 * self.k * self.T)
        exp_term_Delta = (1 - np.exp(-self.k * self.Delta)) ** 2
        common_factor = exp_term_T * exp_term_Delta / (2 * self.k**3 * self.Delta**2)

        variances = np.array(self.omega) ** 2 * common_factor
        return variances[0], variances[1]

    def coefficients(self):
        """
        Calculates the coefficients in the weak VIX derivative price approximation
        for the mixed Bergomi model.
        """

        gamma = np.zeros((3, len(self.omega)))

        # Pre-compute common exponential terms for efficiency
        exp_2kT = 1 - np.exp(-2 * self.k * self.T)
        exp_2k_delta = 1 - np.exp(-2 * self.k * self.Delta)
        exp_k_delta = 1 - np.exp(-self.k * self.Delta)

        # Intermediate terms derived from the model's covariance structure
        term12 = (
            -1
            + self.k
            * self.Delta
            * (1 + np.exp(-2 * self.k * self.Delta))
            / exp_2k_delta
        )
        term14 = (
            (2 + self.k * self.Delta) * np.exp(-self.k * self.Delta)
            - 2
            + self.k * self.Delta
        )
        term22 = (
            2 * self.k * self.Delta * (1 + np.exp(-self.k * self.Delta))
            + np.exp(-2 * self.k * self.Delta) * (2 * self.k * self.Delta + 3)
            - 3
        )
        term32 = (
            self.k * self.Delta
            - 2
            + np.exp(-self.k * self.Delta) * (2 + self.k * self.Delta)
        )

        for j in range(len(self.omega)):
            omega_j2 = self.omega[j] ** 2
            omega_j4 = self.omega[j] ** 4

            # γ_1,j
            term11 = omega_j4 / (128 * self.k**4 * self.Delta**2)
            term13 = omega_j2 / (8 * self.k**3 * self.Delta**2)
            gamma[0, j] = (
                term11 * term12 * exp_2kT**2 * exp_2k_delta**2
                + term13 * term14 * exp_2kT * exp_k_delta
            )

            # γ_2,j
            term21 = omega_j4 / (48 * self.k**5 * self.Delta**3)
            gamma[1, j] = -term21 * term22 * exp_2kT**2 * exp_k_delta**2

            # γ_3,j
            term31 = omega_j4 / (16 * self.k**6 * self.Delta**4)
            gamma[2, j] = term31 * term32 * exp_2kT**2 * exp_k_delta**3

        return gamma

    def get_bc(self):
        """
        Compute the necessary parameters for the function g,
        g (y) := ( 1 + b * exp (c * y))^(1 / 2).

        """

        mu1, mu2 = self.mean()
        sigma1, sigma2 = np.sqrt(self.variance())

        if sigma1 < sigma2:
            b_factor = (1 - self.lam) / self.lam
            exp_arg_b = mu2 - mu1 + (sigma2 - sigma1) * sigma1 / 2
            b = b_factor * np.exp(np.clip(exp_arg_b, EXP_LOWER_BOUND, EXP_UPPER_BOUND))
            c = sigma2 - sigma1
        else:
            b_factor = self.lam / (1 - self.lam)
            exp_arg_b = mu1 - mu2 + (sigma1 - sigma2) * sigma2 / 2
            b = b_factor * np.exp(np.clip(exp_arg_b, EXP_LOWER_BOUND, EXP_UPPER_BOUND))
            c = sigma1 - sigma2

        return b, c


# --- Mixed Bergomi Model ---


class MixedRoughBergomi:
    """
    Implementation of the mixed rough Bergomi model.

    Parameters
    ----------
    eta1, eta2: float
                    Vol-of-vol for each process.
    lam: float
         Mixing weight.
    H: float
       Hurst parameter.
    X0: float
        Initial log-spot variance.
    T: float
       Time to maturity.
    Delta: float
           VIX derivative tenor.
    """

    def __init__(self, params):
        """
        Initialize the mixed Bergomi model.
        See class docstring for parameter definitions.
        """

        self.eta1, self.eta2, self.lam, self.H, self.X0, self.T, self.Delta = params
        self.eta = [self.eta1, self.eta2]

    def mean(self):
        """
        Compute the mean of each squared VIX proxy.
        """

        h_term = 2 * self.H + 1
        num = (self.T + self.Delta) ** h_term - self.Delta**h_term - self.T**h_term
        den = 4 * self.Delta * self.H * h_term

        means = self.X0 - (np.array(self.eta) ** 2 * num) / den
        return means[0], means[1]

    def variance(self):
        """
        Compute the variance of each squared VIX proxy.
        """

        h_term1 = self.H + 0.5
        h_term2 = 2 * self.H + 2

        term1 = (np.array(self.eta) ** 2) / (self.Delta**2 * h_term1**2)
        term2 = (
            (self.T + self.Delta) ** h_term2 - self.Delta**h_term2 + self.T**h_term2
        ) / h_term2
        # The hypergeometric function arises from integrating the fractional kernel
        hyp_geo = hyp2f1(-h_term1, self.H + 1.5, self.H + 2.5, -self.T / self.Delta)
        term3 = (
            2
            * beta(1, self.H + 1.5)
            * (self.Delta**h_term1)
            * (self.T ** (self.H + 1.5))
            * hyp_geo
        )

        variances = term1 * (term2 - term3)
        return variances[0], variances[1]

    # Helper methods for the integrands in the `coefficients` calculation.
    def integrand_gamma2(self, t, u):
        """
        Computer the integrands necessary in the 'coefficients' calculation.

        :param t: float or np.ndarray
                  Lower time(s).
        :param u: float or np.ndarray
                  Upper time(s) (must satisfy u > t).
        """

        term1 = (self.T * t + self.Delta) ** (self.H + 0.5) - (self.T * t) ** (
            self.H + 0.5
        )
        term2 = (self.T + self.Delta * u) ** (2 * self.H) - (self.Delta * u) ** (
            2 * self.H
        )
        term3 = (self.T * t + self.Delta * u) ** (self.H - 0.5)
        return term1 * term2 * term3

    def omega_gamma3(self, u):
        """
        Compute the necessary function of the integrand necessary in the 'coefficients'
        calculation

        :param u: float or np.ndarray
                  Time(s).
        """

        delta_ratio = self.Delta / self.T

        h_term = self.H + 0.5
        term1 = ((1 - u) ** h_term) * (delta_ratio**h_term) * beta(1, h_term)

        hyp1 = hyp2f1(
            -h_term,
            h_term,
            h_term + 1,
            -(1 + delta_ratio * u) / (delta_ratio * (1 - u)),
        )
        term2 = ((1 + delta_ratio * u) ** h_term) * hyp1

        hyp2 = hyp2f1(-h_term, h_term, h_term + 1, -u / (1 - u))
        term3 = ((delta_ratio * u) ** h_term) * hyp2

        hyp3 = hyp2f1(-h_term + 1, h_term + 1, h_term + 2, -1 / (delta_ratio * u))
        term4 = beta(1, h_term + 1) * ((delta_ratio * u) ** (h_term - 1)) * hyp3
        return term1 * (term2 - term3) - term4

    def integrand_gamma3(self, t, u):
        """
        Computer the integrands necessary in the 'coefficients' calculation.

        :param t: float or np.ndarray
                  Lower time(s).
        :param u: float or np.ndarray
                  Upper time(s) (must satisfy u > t).
        """

        omega_val = self.omega_gamma3(u)
        delta_ratio = self.Delta / self.T
        term1 = (t + delta_ratio) ** (self.H + 0.5) - t ** (self.H + 0.5)
        term2 = (t + delta_ratio * u) ** (self.H - 0.5)
        return term1 * term2 * omega_val

    def coefficients(self):
        """
        Calculates the coefficients in the weak VIX derivative price approximation
        for the mixed rough Bergomi model.
        """

        sigma_sq = self.variance()
        gamma = np.zeros((3, len(self.eta)))

        h_term1 = 2 * self.H + 1
        h_term2 = 4 * self.H + 1

        # BUG FIX: Wrap integrals in try-except to prevent crashes on non-convergence.
        try:
            integral_gamma2 = dblquad(
                self.integrand_gamma2, 0, 1, lambda t: 0, lambda t: 1
            )[0]
            integral_gamma3 = dblquad(
                self.integrand_gamma3, 0, 1, lambda t: 0, lambda t: 1
            )[0]
        except Exception as e:
            print(f"Warning: Integration for gamma coefficients failed: {e}")
            gamma[:] = np.nan  # Set all coefficients to NaN if integration fails
            return gamma

        for j in range(len(self.eta)):
            eta_j2 = self.eta[j] ** 2
            eta_j4 = self.eta[j] ** 4

            term14 = (
                (self.T + self.Delta) ** h_term1 - self.Delta**h_term1 - self.T**h_term1
            ) / (self.Delta * h_term1)

            # γ_1,j
            term11 = (
                (self.T + self.Delta) ** h_term2 + self.Delta**h_term2 - self.T**h_term2
            ) / (self.Delta * h_term2)
            hyp11 = hyp2f1(-2 * self.H, h_term1, 2 * self.H + 2, -self.Delta / self.T)
            term15 = (
                2
                * beta(1, h_term1)
                * (self.Delta ** (2 * self.H))
                * (self.T ** (2 * self.H))
                * hyp11
            )
            gamma[0, j] = (
                (eta_j4 / (32 * self.H**2)) * (term11 - term14**2 - term15)
                + (eta_j2 * term14) / (4 * self.H)
                - sigma_sq[j] / 2
            )

            # γ_2,j
            term21 = -(eta_j4 * self.T) / (2 * self.Delta * self.H * h_term1)
            term23 = (eta_j2 * sigma_sq[j] * term14) / (4 * self.H)
            gamma[1, j] = term21 * integral_gamma2 + term23

            # γ_3,j
            term31 = (eta_j4 * self.T ** (4 * self.H + 2)) / (
                2 * self.Delta**2 * (self.H + 0.5) ** 2
            )
            gamma[2, j] = term31 * integral_gamma3 - sigma_sq[j] ** 2 / 2

        return gamma

    def get_bc(self):
        """
        Compute the necessary parameters for the function g.
        g (y) := ( 1 + b * exp (c * y))^(1 / 2)

        """

        mu1, mu2 = self.mean()
        sigma1, sigma2 = np.sqrt(self.variance())

        if sigma1 < sigma2:
            b_factor = (1 - self.lam) / self.lam
            exp_arg_b = mu2 - mu1 + (sigma2 - sigma1) * sigma1 / 2
            b = b_factor * np.exp(np.clip(exp_arg_b, EXP_LOWER_BOUND, EXP_UPPER_BOUND))
            c = sigma2 - sigma1
        else:
            b_factor = self.lam / (1 - self.lam)
            exp_arg_b = mu1 - mu2 + (sigma1 - sigma2) * sigma2 / 2
            b = b_factor * np.exp(np.clip(exp_arg_b, EXP_LOWER_BOUND, EXP_UPPER_BOUND))
            c = sigma1 - sigma2

        return b, c


"""
--- Pricing Approximation Methods ---
"""


# --- Hermite Polynomial Expansion Method---
class HermiteApproximation:
    """
    Compute the VIX derivative price using the probabilist's Hermite polynomials.

    Parameters
    ----------
    kappa : float
        The strike price of the VIX calls/puts.
    model: The instance of a Bergomi model, either mixed Bergomi model or
        mixed rough Bergomi model.
    p: int
       The payoff type: 1 for calls,
                        0 for futures,
                        others for puts.
    """

    def __init__(self, kappa, model, p):
        """
        Initializes the Hermite polynomial expansion with model parameters.
        See class docstring for parameter definitions.
        """

        self.kappa = kappa
        self.lam = model.lam
        self.T = model.T
        self.mu1, self.mu2 = model.mean()
        self.var1, self.var2 = model.variance()
        self.sigma1 = np.sqrt(self.var1)
        self.sigma2 = np.sqrt(self.var2)
        self.p = p
        self.gamma = model.coefficients()
        self.b, self.c = model.get_bc()

    def prob_hermite_poly(self, n, y):
        """
        Evaluate the probabilist's Hermite polynomial.

        :param n: int
                  Order of Hermite polynomial. (n > 0)
        :param y: float or np.ndarray
                  Points at which to evaluate the polynomial.
        """

        if not isinstance(n, int) or n < 0:
            raise ValueError("Order n must be a non-negative integer.")

        is_scalar = np.isscalar(y)
        y_arr = np.asarray(y, dtype=float)

        # eval_hermitenorm handles both scalar and array inputs
        res_arr = np.ones_like(y_arr) if n == 0 else eval_hermitenorm(n, y_arr)

        return res_arr.item() if is_scalar else res_arr

    def hermite_phi_product(self, n, y):
        """
        Compute the product of the probabilist's Hermite polynomial and the PDF
        of the standard normal random variable.

        :param n: int
                  Order of Hermite polynomial. (n > 0)
        :param y: float or np.ndarray
                  Points at which to evaluate the polynomial.
        """

        if not isinstance(n, int) or n < 0:
            raise ValueError("Order n must be a non-negative integer.")

        is_scalar = np.isscalar(y)
        y_arr = np.asarray(y, dtype=float)

        if n == 0:
            res_arr = np.ones_like(y_arr) * norm.pdf(y)
        else:
            res_arr = eval_hermitenorm(n, y) * norm.pdf(y)
        return res_arr.item() if is_scalar else res_arr

    def func_g(self, y):
        """
        Compute the function g, g (y) := ( 1 + b * exp (c * y))^(1 / 2).

        :param y: float or np.ndarray
                  Input value(s) for y.
        """

        is_scalar = np.isscalar(y)
        y_arr = np.asarray(y, dtype=float)

        arg_exp_g = self.c * y_arr
        clipped_arg_exp_g = np.clip(arg_exp_g, EXP_LOWER_BOUND, EXP_UPPER_BOUND)

        exp_val = np.exp(clipped_arg_exp_g)
        b_exp_val = self.b * exp_val
        g_sq = 1.0 + b_exp_val

        # Create a mask for valid (positive) term0 to avoid log(0) or log(-)
        valid_mask = g_sq > EPSILON

        g_val = np.zeros_like(y_arr, dtype=float)

        if np.any(valid_mask):
            # Isolate the valid terms for log calculation
            v_b_exp_val = b_exp_val[valid_mask]

            # Calculate in log-space to prevent overflow
            log_g = 0.5 * np.log1p(v_b_exp_val)
            g_val[valid_mask] = np.exp(np.clip(log_g, EXP_LOWER_BOUND, EXP_UPPER_BOUND))

        return g_val.item() if is_scalar else g_val

    def func_g1(self, y):
        """
        Compute the first-order partial derivative of g (y) with respect to the mean
        of each process in the mixed Bergomi model or mixed rough Bergomi model.

        :param y: float or np.ndarray
                  Input value(s) for y.
        """

        is_scalar = np.isscalar(y)
        y_arr = np.asarray(y, dtype=float)

        arg_exp_g = self.c * y_arr
        clipped_arg_exp_g = np.clip(arg_exp_g, EXP_LOWER_BOUND, EXP_UPPER_BOUND)

        exp_val = np.exp(clipped_arg_exp_g)
        b_exp_val = self.b * exp_val
        g_sq = 1.0 + b_exp_val

        # Create a mask for valid (positive) term0 to avoid log(0) or log(-)
        valid_mask = g_sq > EPSILON

        g1_val = np.zeros_like(y_arr, dtype=float)

        if np.any(valid_mask):
            # Isolate the valid terms for log calculation
            v_b_exp_val = b_exp_val[valid_mask]
            v_clipped_arg = clipped_arg_exp_g[valid_mask]

            # Calculate in log-space to prevent overflow
            log_numerator = np.log(0.5 * self.b) + v_clipped_arg
            log_denominator = 0.5 * np.log1p(
                v_b_exp_val
            )  # Use log1p for precision: log(1+x)

            # Combine logs and exponential back
            log_g1 = log_numerator - log_denominator
            g1_val[valid_mask] = np.exp(
                np.clip(log_g1, EXP_LOWER_BOUND, EXP_UPPER_BOUND)
            )

        if self.sigma1 <= self.sigma2:
            return -g1_val.item() if is_scalar else -g1_val
        else:
            return g1_val.item() if is_scalar else g1_val

    def func_g2(self, y):
        """
        Compute the second-order partial derivative of g (y) with respect to the mean
        of each process in the mixed Bergomi model or mixed rough Bergomi model.
        Using a numerically stable log-space computation to prevent overflow.

        :param y: float or np.ndarray
                  Input value(s) for y.
        """

        is_scalar = np.isscalar(y)
        y_arr = np.asarray(y, dtype=float)

        arg_exp_g = self.c * y_arr
        clipped_arg_exp_g = np.clip(arg_exp_g, EXP_LOWER_BOUND, EXP_UPPER_BOUND)

        exp_val = np.exp(clipped_arg_exp_g)
        b_exp_val = self.b * exp_val
        g_sq = 1.0 + b_exp_val

        # Create a mask for valid (positive) term0 to avoid log(0) or log(-)
        valid_mask = g_sq > EPSILON

        g2_val = np.zeros_like(y_arr, dtype=float)

        if np.any(valid_mask):
            # Isolate the valid terms for log calculation
            v_b_exp_val = b_exp_val[valid_mask]
            v_clipped_arg = clipped_arg_exp_g[valid_mask]

            # Calculate in log-space to prevent overflow
            log_numerator = (
                np.log(0.25 * self.b) + v_clipped_arg + np.log(2 + v_b_exp_val)
            )
            log_denominator = 1.5 * np.log1p(
                v_b_exp_val
            )  # Use log1p for precision: log(1+x)

            # Combine logs and exponential back
            log_g2 = log_numerator - log_denominator
            g2_val[valid_mask] = np.exp(
                np.clip(log_g2, EXP_LOWER_BOUND, EXP_UPPER_BOUND)
            )

        return g2_val.item() if is_scalar else g2_val

    def func_g3(self, y):
        """
        Compute the third-order partial derivative of g (y) with respect to the mean
        of each process in the mixed Bergomi model or mixed rough Bergomi model.
        Using a numerically stable log-space computation to prevent overflow.

        :param y: float or np.ndarray
                  Input value(s) for y.
        """

        is_scalar = np.isscalar(y)
        y_arr = np.asarray(y, dtype=float)

        arg_exp_g = self.c * y_arr
        clipped_arg_exp_g = np.clip(arg_exp_g, EXP_LOWER_BOUND, EXP_UPPER_BOUND)
        exp_val = np.exp(clipped_arg_exp_g)
        b_exp_val = self.b * exp_val
        g_sq = 1.0 + b_exp_val

        # Create a mask for valid (positive) term0
        valid_mask = g_sq > EPSILON

        g3_val = np.zeros_like(y_arr, dtype=float)

        if np.any(valid_mask):
            # Isolate the valid terms for calculation
            v_b = self.b
            x = b_exp_val[valid_mask]
            v_clipped_arg = clipped_arg_exp_g[valid_mask]

            # --- DEFINITIVE HYBRID FIX ---
            # We need two different methods to calculate log(4 + 2x + x^2)
            # depending on the magnitude of x.

            log_of_num_term = np.zeros_like(x)

            # Define a threshold to switch between methods
            large_x_threshold = 1e50

            # Create masks for large and small values of x
            large_x_mask = x > large_x_threshold
            small_x_mask = ~large_x_mask

            # Case 1: x is small or moderate. Direct calculation is safe.
            # This avoids the 1/x overflow.
            if np.any(small_x_mask):
                x_small = x[small_x_mask]
                log_of_num_term[small_x_mask] = np.log(4 + 2 * x_small + x_small**2)

            # Case 2: x is very large. Use the factorization to avoid x**2 overflow.
            if np.any(large_x_mask):
                x_large = x[large_x_mask]
                log_x = np.log(v_b) + v_clipped_arg[large_x_mask]

                # Calculate 1/x first. This is safe as x is large.
                inv_x = 1 / x_large
                log1p_arg = 2 * inv_x + (2 * inv_x) ** 2
                log_of_num_term[large_x_mask] = 2 * log_x + np.log1p(log1p_arg)
            # --- FIX ENDS ---

            # Calculate the rest in log-space as before
            log_numerator = np.log(0.125 * v_b) + v_clipped_arg + log_of_num_term
            log_denominator = 2.5 * np.log1p(x)  # Use x directly here

            # Combine logs and exponential back
            log_g3 = log_numerator - log_denominator
            g3_val[valid_mask] = np.exp(
                np.clip(log_g3, EXP_LOWER_BOUND, EXP_UPPER_BOUND)
            )

        if self.sigma1 <= self.sigma2:
            return -g3_val.item() if is_scalar else -g3_val
        else:
            return g3_val.item() if is_scalar else g3_val

    def weight_calculation(self, n, func_g_):
        """
        Compute the weight for the Hermite expansion.

        :param n: int
                  Order of the Hermite polynomial.
        :param func_g_: callable
                        The function g or its derivative
                        (e.g., self.func_g, self.func_g1).
        """

        def integrand(y):
            return func_g_(y) * self.hermite_phi_product(n, y)

        integral = np.nan
        err = np.inf

        try:
            # First attempt: integrate over the full range
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always", IntegrationWarning)
                integral, err = quad(
                    integrand,
                    -np.inf,
                    np.inf,
                    limit=50000,
                    epsabs=EPSILON,
                    epsrel=EPSILON,
                )

            # If the first attempt raised a warning, split at 0
            if w:
                print("Initial integration warned. Splitting at 0.")
                with warnings.catch_warnings(record=True) as w2:
                    warnings.simplefilter("always", IntegrationWarning)
                    integral1, err1 = quad(
                        integrand,
                        -np.inf,
                        0,
                        limit=50000,
                        epsabs=EPSILON,
                        epsrel=EPSILON,
                    )
                    integral2, err2 = quad(
                        integrand,
                        0,
                        np.inf,
                        limit=50000,
                        epsabs=EPSILON,
                        epsrel=EPSILON,
                    )
                    integral = integral1 + integral2
                    err = err1 + err2

                # If splitting at 0 ALSO warned, split the negative part at -10
                if w2:
                    print("Second integration warned. Splitting at -10.")
                    # Note: No need for another warning catch here unless you have
                    # further nesting
                    integral1, err1 = quad(
                        integrand,
                        -np.inf,
                        -10,
                        limit=50000,
                        epsabs=EPSILON,
                        epsrel=EPSILON,
                    )
                    integral2, err2 = quad(
                        integrand, -10, 0, limit=50000, epsabs=EPSILON, epsrel=EPSILON
                    )
                    integral3, err3 = quad(
                        integrand,
                        0,
                        np.inf,
                        limit=50000,
                        epsabs=EPSILON,
                        epsrel=EPSILON,
                    )
                    integral = integral1 + integral2 + integral3
                    err = err1 + err2 + err3

            # Final check on the estimated error
            if err > abs(integral) * 0.01 and err > 1e-7:
                pass  # Warning about high integration error

        except Exception as e:
            print(
                f"Warning: Integration failed for n={n} "
                f"with function {func_g_.__name__}. Error: {e}"
            )
            integral = np.nan

        return integral / float(factorial(n))

    def g_approx(self, y, N, w=None):
        """
        Compute the Nth-order approximation for g(y).

        :param y: float or np.ndarray
                  Input value(s) for y.
        :param N: int
                  The order of the Hermite polynomial used for approximation.
        :param w: float or np.ndarray
                  Pre-calculated weights for the Hermite expansion.
                  If None, weights are calculated.
        """

        is_scalar = np.isscalar(y)
        y_arr = np.asarray(y, dtype=float)
        g_approx = np.zeros_like(y_arr, dtype=float)

        if w is None:
            weights = w or [
                self.weight_calculation(i, self.func_g(y_arr)) for i in range(N + 1)
            ]
        elif len(w) != N + 1:
            raise ValueError(
                f"Provided weights length {len(w)} doesn't match N+1 ({N + 1})"
            )
        else:
            weights = w

        for n, weight in enumerate(weights):
            if not np.isnan(weight):
                g_approx += weight * self.prob_hermite_poly(n, y_arr)
            else:
                g_approx += np.full_like(y_arr, np.nan)
        return g_approx.item() if is_scalar else g_approx

    def d1_finding(self):
        """
        Find the value of d1 in the Black-Scholes price function, which is defined
        as 'A' in the paper. Use Brent's method or fsolve for the objective function.
        """

        # Function F(x) = lambda * exp(mu1 + sigma1*x)
        # + (1-lambda) * exp(mu2 + sigma2*x) - kappa^2
        def objective(x):
            term1 = self.lam * np.exp(self.mu1 + self.sigma1 * x)
            term2 = (1 - self.lam) * np.exp(self.mu2 + self.sigma2 * x)
            return term1 + term2 - self.kappa**2

        try:
            # Use brentq for robust root finding within a bracketing interval
            d1 = brentq(objective, -20, 20)  # Adjusted range for robustness
            return d1
        except ValueError:
            # Fallback to fsolve if brentq fails (e.g., no sign change in interval)
            print(
                "Warning: brentq failed, falling back to fsolve. Consider "
                "adjusting brentq interval."
            )
            d1 = fsolve(objective, 0)[0]  # Initial guess at 0
            return d1
        except Exception as e:
            print(f"Error in d1_finding: {e}")
            return np.nan  # Return NaN if root finding completely fails

    def calculate_I_N(self, weights):
        """
        Compute the integral part of I_N, I_1N, I_2N, I_3N in the VIX call/put prices.

        :param weights: float or np.ndarray
                        Pre-calculated weights for the optimal N.
        """

        A = self.d1_finding()

        B = A - self.sigma1 / 2 if self.sigma1 < self.sigma2 else A - self.sigma2 / 2

        if self.p == 1:
            if len(weights) == 1:
                h = norm.cdf(-B)
            else:
                h = np.zeros_like(weights)
                h[0] = norm.cdf(-B)
                for i in range(1, len(weights)):
                    h[i] = self.hermite_phi_product(i - 1, B)
        elif self.p == 0:
            h = np.ones_like(weights)
        else:
            if len(weights) == 1:
                h = norm.cdf(B)
            else:
                h = np.zeros_like(weights)
                h[0] = norm.cdf(B)
                for i in range(1, len(weights)):
                    h[i] = -self.hermite_phi_product(i - 1, B)

        integral = weights * h

        return integral

    def coefficients(self):
        """
        Compute the coefficients in the Hermite expansion.
        """

        c = np.zeros(4)

        if self.sigma1 < self.sigma2:
            a = self.lam**0.5 * np.exp(self.mu1 / 2 + self.var1 / 8)
            c[0] = (
                1
                + 0.5 * self.gamma[0, 0]
                + 0.25 * self.gamma[1, 0]
                + 0.125 * self.gamma[2, 0]
            )
            c[1] = (
                self.gamma[0, 0]
                - self.gamma[0, 1]
                + self.gamma[1, 0] * (1 - 0.5 * self.sigma2 / self.sigma1)
                - self.gamma[1, 1] * 0.5 * self.sigma1 / self.sigma2
                + self.gamma[2, 0] * (3 / 4 - 0.5 * self.sigma2 / self.sigma1)
                - self.gamma[2, 1] * 0.25 * self.sigma1**2 / self.sigma2**2
            )
            c[2] = (
                self.gamma[1, 0] * (1 - self.sigma2 / self.sigma1)
                + self.gamma[1, 1] * (1 - self.sigma1 / self.sigma2)
                + self.gamma[2, 0]
                * (
                    1.5
                    - 2 * self.sigma2 / self.sigma1
                    + 0.5 * self.sigma2**2 / self.sigma1**2
                )
                + self.gamma[2, 1]
                * (self.sigma1 / self.sigma2 - self.sigma1**2 / self.sigma2**2)
            )
            c[3] = self.gamma[2, 0] * (
                1 - 2 * self.sigma2 / self.sigma1 + self.sigma2**2 / self.sigma1**2
            ) - self.gamma[2, 1] * (
                1 - 2 * self.sigma1 / self.sigma2 + self.sigma1**2 / self.sigma2**2
            )
        else:
            a = (1 - self.lam) ** 0.5 * np.exp(self.mu2 / 2 + self.var2 / 8)
            c[0] = (
                1
                + 0.5 * self.gamma[0, 1]
                + 0.25 * self.gamma[1, 1]
                + 0.125 * self.gamma[2, 1]
            )
            c[1] = (
                self.gamma[0, 0]
                - self.gamma[0, 1]
                + self.gamma[1, 0] * 0.5 * self.sigma2 / self.sigma1
                - self.gamma[1, 1] * (1 - 0.5 * self.sigma1 / self.sigma2)
                + self.gamma[2, 0] * 0.25 * self.sigma2**2 / self.sigma1**2
                - self.gamma[2, 1] * (3 / 4 - 0.5 * self.sigma1 / self.sigma2)
            )
            c[2] = (
                self.gamma[1, 0] * (1 - self.sigma2 / self.sigma1)
                + self.gamma[1, 1] * (1 - self.sigma1 / self.sigma2)
                + self.gamma[2, 0]
                * (self.sigma2 / self.sigma1 - self.sigma2**2 / self.sigma1**2)
                + self.gamma[2, 1]
                * (
                    1.5
                    - 2 * self.sigma1 / self.sigma2
                    + 0.5 * self.sigma1**2 / self.sigma2**2
                )
            )
            c[3] = self.gamma[2, 0] * (
                1 - 2 * self.sigma2 / self.sigma1 + self.sigma2**2 / self.sigma1**2
            ) - self.gamma[2, 1] * (
                1 - 2 * self.sigma1 / self.sigma2 + self.sigma1**2 / self.sigma2**2
            )

        return c * a

    def vix_derivative_price(self, best_N):
        """
        Compute the VIX derivative price using the Hermite expansion.
        :param best_N: int
                       The optimal order N for the Hermite approximation to use
                       for call/put options.
                       For futures (p=0), this will be overridden to 0.
        """

        A = self.d1_finding()

        # Determine the effective N based on p
        # If p=0 (futures), optimal order is 0. Otherwise, use the provided best_N_star.
        effective_N = 0 if self.p == 0 else best_N

        # Pre-calculate weights for all approximation functions using
        # the effective_N_star
        weights = np.zeros((4, effective_N + 1))
        weights[0] = [
            self.weight_calculation(n, self.func_g) for n in range(effective_N + 1)
        ]
        weights[1] = [
            self.weight_calculation(n, self.func_g1) for n in range(effective_N + 1)
        ]
        weights[2] = [
            self.weight_calculation(n, self.func_g2) for n in range(effective_N + 1)
        ]
        weights[3] = [
            self.weight_calculation(n, self.func_g3) for n in range(effective_N + 1)
        ]

        # Calculate I_N, I_1N, I_2N, I_3N using the pre-calculated weights and
        # effective_N_star
        integral_N = np.array(
            [np.sum(self.calculate_I_N(weights[i])) for i in range(4)]
        )

        c = self.coefficients()

        # Combine terms based on derivative type (p)
        if self.p == 0:  # Futures
            price = np.sum(c * integral_N)
        elif self.p == 1:  # Call option
            price = np.sum(c * integral_N) - self.kappa * norm.cdf(-A)
        else:  # Put option
            price = self.kappa * norm.cdf(A) - np.sum(c * integral_N)

        return price

    def implied_vol_taylor(self, best_N, type):
        """
        Compute the implied volatility using the Taylor's theorem.
        :param best_N: int
                       The optimal order N for the Hermite approximation to use
                       for call/put options.
        :param type: int
                     The expansion type: 1 for Method 1,
                                         2 for Method 2,
                                         3 for Method 3.
        """

        A = self.d1_finding()

        weights = np.zeros((4, best_N + 1))
        weights[0] = [
            self.weight_calculation(n, self.func_g) for n in range(best_N + 1)
        ]
        weights[1] = [
            self.weight_calculation(n, self.func_g1) for n in range(best_N + 1)
        ]
        weights[2] = [
            self.weight_calculation(n, self.func_g2) for n in range(best_N + 1)
        ]
        weights[3] = [
            self.weight_calculation(n, self.func_g3) for n in range(best_N + 1)
        ]

        integral_N = np.array([self.calculate_I_N(weights[i]) for i in range(4)])

        c = self.coefficients()

        coeff_pdf = sum(ci * sum(li[1:]) for ci, li in zip(c, integral_N, strict=False))

        if self.sigma1 < self.sigma2:
            B = A - self.sigma1 / 2
            # a = self.lam ** 0.5 * np.exp(self.mu1 / 2 + self.var1 / 8)
            # f = a * integral_N[0, 0] / norm.cdf(-B)
            f = sum(
                ci * li[0] for ci, li in zip(c, integral_N, strict=False)
            ) / norm.cdf(-B)
            if type == 1:
                k = np.log(self.kappa)
                x0 = k - A * self.sigma1 / 2 + self.var1 / 8
                iv0 = 0.5 * self.sigma1 / np.sqrt(self.T)
            else:
                x0 = np.log(f)
                if type == 2:
                    k = x0 + B * self.sigma1 / 2 + self.var1 / 8
                    iv0 = 0.5 * self.sigma1 / np.sqrt(self.T)
                else:
                    k = np.log(self.kappa)
                    if (B**2 - 2 * (x0 - k)) >= 0:
                        iv01 = (-B + np.sqrt(B**2 - 2 * (x0 - k))) / np.sqrt(self.T)
                        iv02 = (-B - np.sqrt(B**2 - 2 * (x0 - k))) / np.sqrt(self.T)
                        conditions = [
                            (iv01 < 0) & (iv02 < 0),
                            (iv01 >= 0) & (iv02 < 0),
                            (iv01 < 0) & (iv02 >= 0),
                        ]
                        choices = [np.nan, iv01, iv02]
                        iv0 = np.select(conditions, choices, np.minimum(iv01, iv02))
                        if np.isnan(iv0):
                            raise ValueError(
                                "The zeroth-order implied volatility is less than 0"
                                ", this method is not applicable."
                            )
                    else:
                        raise ValueError(
                            "The discriminant is less than 0, this method is not "
                            "applicable."
                        )

        else:
            B = A - self.sigma2 / 2
            # a = (1 - self.lam) ** 0.5 * np.exp(self.mu2 / 2 + self.var2 / 8)
            # f = a * integral_N[0, 0] / norm.cdf(-B)
            f = sum(
                ci * li[0] for ci, li in zip(c, integral_N, strict=False)
            ) / norm.cdf(-B)
            if type == 1:
                k = np.log(self.kappa)
                iv0 = 0.5 * self.sigma2 / np.sqrt(self.T)
                x0 = k - A * self.sigma2 / 2 + self.var2 / 8
            else:
                x0 = np.log(f)
                if type == 2:
                    k = x0 + B * self.sigma2 / 2 + self.var2 / 8
                    iv0 = 0.5 * self.sigma2 / np.sqrt(self.T)
                else:
                    k = np.log(self.kappa)
                    if (B**2 - 2 * (x0 - k)) >= 0:
                        iv01 = (-B + np.sqrt(B**2 - 2 * (x0 - k))) / np.sqrt(self.T)
                        iv02 = (-B - np.sqrt(B**2 - 2 * (x0 - k))) / np.sqrt(self.T)
                        conditions = [
                            (iv01 < 0) & (iv02 < 0),
                            (iv01 >= 0) & (iv02 < 0),
                            (iv01 < 0) & (iv02 >= 0),
                        ]
                        choices = [np.nan, iv01, iv02]
                        iv0 = np.select(conditions, choices, np.minimum(iv01, iv02))
                        if np.isnan(iv0):
                            raise ValueError(
                                "The zeroth-order implied volatility is less than 0"
                                ", this method is not applicable."
                            )
                    else:
                        raise ValueError(
                            "The discriminant is less than 0, this method is "
                            "not applicable."
                        )

        vega = np.exp(x0) * np.sqrt(self.T) * norm.pdf(B)

        if type == 1:
            extra_term = (f - np.exp(x0)) * norm.cdf(-B)
        elif type == 2:
            extra_term = (np.exp(k) - self.kappa) * norm.cdf(-A)
        else:
            extra_term = self.kappa * (
                norm.cdf(-B - iv0 * np.sqrt(self.T)) - norm.cdf(-A)
            )

        iv1 = coeff_pdf / vega
        iv2 = extra_term / vega

        iv = iv0 + iv1
        iv_e = iv0 + iv1 + iv2

        return iv, iv_e

    def implied_vol_bell_polynomial(self, best_N, type):
        """
        Compute the implied volatility using the Taylor's theorem.
        :param best_N: int
                       The optimal order N for the Hermite approximation to use
                       for call/put options.
        :param type: int
                     The expansion type: 1 for Method 1,
                                         2 for Method 2,
                                         3 for Method 3.
        """

        A = self.d1_finding()

        weights = np.zeros((4, best_N + 1))
        weights[0] = [
            self.weight_calculation(n, self.func_g) for n in range(best_N + 1)
        ]
        weights[1] = [
            self.weight_calculation(n, self.func_g1) for n in range(best_N + 1)
        ]
        weights[2] = [
            self.weight_calculation(n, self.func_g2) for n in range(best_N + 1)
        ]
        weights[3] = [
            self.weight_calculation(n, self.func_g3) for n in range(best_N + 1)
        ]

        integral_N = np.array([self.calculate_I_N(weights[i]) for i in range(4)])

        c = self.coefficients()

        f = sum(ci * wi[0] for ci, wi in zip(c, weights, strict=False))

        if self.sigma1 < self.sigma2:
            B = A - self.sigma1 / 2
            if type == 1:
                k = np.log(self.kappa)
                x0 = k - A * self.sigma1 / 2 + self.var1 / 8
                iv0 = 0.5 * self.sigma1 / np.sqrt(self.T)
            else:
                x0 = np.log(f)
                if type == 2:
                    iv0 = 0.5 * self.sigma1 / np.sqrt(self.T)
                    k = x0 + B * self.sigma1 / 2 + self.var1 / 8
                else:
                    k = np.log(self.kappa)
                    if (B**2 - 2 * (x0 - k)) >= 0:
                        iv01 = (-B + np.sqrt(B**2 - 2 * (x0 - k))) / np.sqrt(self.T)
                        iv02 = (-B - np.sqrt(B**2 - 2 * (x0 - k))) / np.sqrt(self.T)
                        conditions = [
                            (iv01 < 0) & (iv02 < 0),
                            (iv01 >= 0) & (iv02 < 0),
                            (iv01 < 0) & (iv02 >= 0),
                        ]
                        choices = [np.nan, iv01, iv02]
                        iv0 = np.select(conditions, choices, np.minimum(iv01, iv02))
                        if np.isnan(iv0):
                            raise ValueError(
                                "The zeroth-order implied volatility is less than 0"
                                ", this method is not applicable."
                            )
                    else:
                        raise ValueError(
                            "The discriminant is less than 0, this method is not "
                            "applicable."
                        )
        else:
            B = A - self.sigma2 / 2
            if type == 1:
                k = np.log(self.kappa)
                iv0 = 0.5 * self.sigma2 / np.sqrt(self.T)
                x0 = k - A * self.sigma2 / 2 + self.var2 / 8
            else:
                x0 = np.log(f)
                if type == 2:
                    iv0 = 0.5 * self.sigma2 / np.sqrt(self.T)
                    k = x0 + B * self.sigma2 / 2 + self.var2 / 8
                else:
                    k = np.log(self.kappa)
                    if (B**2 - 2 * (x0 - k)) >= 0:
                        iv01 = (-B + np.sqrt(B**2 - 2 * (x0 - k))) / np.sqrt(self.T)
                        iv02 = (-B - np.sqrt(B**2 - 2 * (x0 - k))) / np.sqrt(self.T)
                        conditions = [
                            (iv01 < 0) & (iv02 < 0),
                            (iv01 >= 0) & (iv02 < 0),
                            (iv01 < 0) & (iv02 >= 0),
                        ]
                        choices = [np.nan, iv01, iv02]
                        iv0 = np.select(conditions, choices, np.minimum(iv01, iv02))
                        if np.isnan(iv0):
                            raise ValueError(
                                "The zeroth-order implied volatility is less than 0"
                                ", this method is not applicable."
                            )
                    else:
                        raise ValueError(
                            "The discriminant is less than 0, this method is "
                            "not applicable."
                        )

        vega = np.exp(x0) * np.sqrt(self.T) * norm.pdf(B)

        if type == 1:
            extra_term = (f - np.exp(x0)) * norm.cdf(-B)
        elif type == 2:
            extra_term = (np.exp(k) - self.kappa) * norm.cdf(-A)
        else:
            extra_term = self.kappa * (
                norm.cdf(-B - iv0 * np.sqrt(self.T)) - norm.cdf(-A)
            )

        v1 = np.sum(c[0] * integral_N[0][1:])
        v2 = np.sum(c[1] * integral_N[1][1:])
        v3 = np.sum(c[2] * integral_N[2][1:])
        v4 = np.sum(c[3] * integral_N[3][1:])

        m = x0 - k

        m1 = m**2 / (iv0**3 * self.T) - iv0 * self.T / 4
        m2 = m1**2 - 3 * m**2 / (iv0**4 * self.T) - self.T / 4
        m3 = (
            m1**3
            - 9 * m**4 / (iv0**7 * self.T**2)
            + 3 * m**2 / (2 * iv0**3)
            + 12 * m**2 / (iv0**5 * self.T)
            + 3 * iv0 * self.T**2 / 16
        )

        iv1 = (v1 + v2) / vega
        iv2 = (v3 + v4) / vega - 0.5 * iv1**2 * m1
        iv3 = (v3 + v4) / vega - iv1 * iv2 * m1 - 1 / 6 * iv1**3 * m2
        iv4 = (
            v4 / vega
            - (iv1 * iv3 + 0.5 * iv2**2) * m1
            - 0.5 * iv1**2 * iv2 * m2
            - 1 / 24 * iv1**4 * m3
        )

        iv = iv0 + iv1 + iv2
        iv_e = iv0 + iv1 + iv2 + iv3 + iv4

        return iv, iv_e

    def implied_vol_bell_polynomial_e(self, best_N, type):
        """
        Compute the implied volatility using the implied volatility theorem with the
        Bell polynomial version of the Faa di Bruno's formula.
        :param best_N: int
                       The optimal order N for the Hermite approximation to use for
                       call/put options.
        :param type: int
                     The expansion type: 1 for Method 1,
                                         2 for Method 2,
                                         3 for Method 3.
        """

        best_N = 4

        A = np.asarray(self.d1_finding())

        weights = np.zeros((4, best_N + 1))
        weights[0] = [
            self.weight_calculation(n, self.func_g) for n in range(best_N + 1)
        ]
        weights[1] = [
            self.weight_calculation(n, self.func_g1) for n in range(best_N + 1)
        ]
        weights[2] = [
            self.weight_calculation(n, self.func_g2) for n in range(best_N + 1)
        ]
        weights[3] = [
            self.weight_calculation(n, self.func_g3) for n in range(best_N + 1)
        ]

        c = self.coefficients()

        f = sum(ci * wi[0] for ci, wi in zip(c, weights, strict=False))

        psi = np.zeros(best_N)

        if self.sigma1 < self.sigma2:
            B = A - self.sigma1 / 2
            for i in range(1, best_N + 1):
                psi[i - 1] = sum(ci * wi[i] for ci, wi in zip(c, weights, strict=False))
            if type == 1:
                k = np.log(self.kappa)
                x0 = k - A * self.sigma1 / 2 + self.var1 / 8
                iv0 = 0.5 * self.sigma1 / np.sqrt(self.T)
            else:
                x0 = np.log(f)
                if type == 2:
                    k = x0 + B * self.sigma1 / 2 + self.var1 / 8
                    iv0 = 0.5 * self.sigma1 / np.sqrt(self.T)
                else:
                    k = np.log(self.kappa)
                    if (B**2 - 2 * (x0 - k)) >= 0:
                        iv01 = (-B + np.sqrt(B**2 - 2 * (x0 - k))) / np.sqrt(self.T)
                        iv02 = (-B - np.sqrt(B**2 - 2 * (x0 - k))) / np.sqrt(self.T)
                        conditions = [
                            (iv01 < 0) & (iv02 < 0),
                            (iv01 >= 0) & (iv02 < 0),
                            (iv01 < 0) & (iv02 >= 0),
                        ]
                        choices = [np.nan, iv01, iv02]
                        iv0 = np.select(conditions, choices, np.minimum(iv01, iv02))
                        if np.isnan(iv0):
                            raise ValueError(
                                "The zeroth-order implied volatility is less than 0"
                                ", this method is not applicable."
                            )
                    else:
                        raise ValueError(
                            "The discriminant is less than 0, this method is not "
                            "applicable."
                        )
        else:
            B = A - self.sigma2 / 2
            # a = (1 - self.lam) ** 0.5 * np.exp(self.mu2 / 2 + self.var2 / 8)
            # f = a * integral_N[0, 0] / norm.cdf(-B)
            for i in range(1, best_N + 1):
                psi[i - 1] = sum(ci * wi[i] for ci, wi in zip(c, weights, strict=False))
            if type == 1:
                k = np.log(self.kappa)
                x0 = k - A * self.sigma2 / 2 + self.var2 / 8
                iv0 = 0.5 * self.sigma2 / np.sqrt(self.T)
            else:
                x0 = np.log(f)
                if type == 2:
                    k = x0 + B * self.sigma2 / 2 + self.var2 / 8
                    iv0 = 0.5 * self.sigma2 / np.sqrt(self.T)
                else:
                    k = np.log(self.kappa)
                    if (B**2 - 2 * (x0 - k)) >= 0:
                        iv01 = (-B + np.sqrt(B**2 - 2 * (x0 - k))) / np.sqrt(self.T)
                        iv02 = (-B - np.sqrt(B**2 - 2 * (x0 - k))) / np.sqrt(self.T)
                        conditions = [
                            (iv01 < 0) & (iv02 < 0),
                            (iv01 >= 0) & (iv02 < 0),
                            (iv01 < 0) & (iv02 >= 0),
                        ]
                        choices = [np.nan, iv01, iv02]
                        iv0 = np.select(conditions, choices, np.minimum(iv01, iv02))
                        if np.isnan(iv0):
                            raise ValueError(
                                "The zeroth-order implied volatility is less than 0"
                                ", this method is not applicable."
                            )
                    else:
                        raise ValueError(
                            "The discriminant is less than 0, this method is not "
                            "applicable."
                        )

        if type == 1:
            extra_term = (f - np.exp(x0)) * norm.cdf(-B)
        elif type == 2:
            extra_term = (np.exp(k) - self.kappa) * norm.cdf(-A)
        else:
            extra_term = self.kappa * (
                norm.cdf(-B - iv0 * np.sqrt(self.T)) - norm.cdf(-A)
            )

        f = np.exp(x0)

        m = x0 - k

        m1 = m**2 / (iv0**3 * self.T) - iv0 * self.T / 4
        m2 = m1**2 - 3 * m**2 / (iv0**4 * self.T) - self.T / 4
        m3 = (
            m1**3
            - 9 * m**4 / (iv0**7 * self.T**2)
            + 3 * m**2 / (2 * iv0**3)
            + 12 * m**2 / (iv0**5 * self.T)
            + 3 * iv0 * self.T**2 / 16
        )

        # iv1 = (psi[0] - psi[1] * iv0 * np.sqrt(self.T) + 2 * psi[2] * (iv0 * np.sqrt(self.T)) ** 2) / (f * iv0 * np.sqrt(self.T))
        # iv2 = (psi[1] - 2 * psi[2] * iv0 * np.sqrt(self.T)) / f * (- m / (iv0 * self.T) + iv0 / 2) - 0.5 * iv1 ** 2 * m1
        # iv3 = psi[2] * np.sqrt(self.T) / f * (m ** 2 / (iv0 ** 2 * self.T ** 2) - (m + 1) / self.T + iv0 ** 2 / 4) - \
        #       iv1 * iv2 * m1 - 1 / 6 * iv1 ** 3 * m2

        iv1 = psi[0] / (f * np.sqrt(self.T)) + psi[1] / (f * np.sqrt(self.T)) * B
        iv2 = psi[2] / (f * np.sqrt(self.T)) * (B**2 - 1) - 0.5 * iv1**2 * m1
        iv3 = (
            psi[3] / (f * np.sqrt(self.T)) * (B**3 - 3 * B)
            - iv1 * iv2 * m1
            - 1 / 6 * iv1**3 * m2
        )

        iv = iv0 + iv1 + iv2 + iv3

        return iv

    def parameters_out(self):
        """
        Prints out the mean, variance and coefficients.
        """

        A = self.d1_finding()

        parameter = {
            "mu1": [self.mu1],
            "mu2": [self.mu2],
            "sigma1": [self.sigma1],
            "sigma2": [self.sigma2],
            "A": [A],
        }
        para = pd.DataFrame(parameter)
        print("Parameters: \n", para)
        print("Coefficients: ", self.gamma)


"""
--- Numerical Experiments Setting ---
"""

# --- Optimal Order of the Hermite Expansion ---


class optimal_N:
    """
    Decide the optimal order of the Hermite expansion for the function g.

    Parameters
    ----------
    obj : callable
          The Hermite approximation class, from where we can import the necessary
          functions.
    N_max: int
           The maximum order of the Hermite expansion.
    T: float
       Time to maturity.
    """

    def __init__(self, obj, N_max, M, T):
        """
        Initialize the optimization problem used to confirm the optimal order.
        See class docstring for parameter definitions.
        """

        self.y_samples = np.random.normal(0, 1, size=M)
        self.y_plot = np.linspace(-10, 10, M, dtype=float)

        self.g_y_samples = obj.func_g(self.y_samples)
        self.g_y_plot = obj.func_g(self.y_plot)

        self.phi_plot = norm.pdf(self.y_plot)

        self.all_weights = [
            obj.weight_calculation(n, obj.func_g) for n in range(N_max + 1)
        ]

        self.obj = obj
        self.N = N_max
        self.T = T

    def optimization(self):
        """
        Compute the minimum order that make the squared error less than 10^(-5).
        """

        if np.any(np.isnan(self.g_y)):
            return (
                0,
                np.full(self.N + 1, np.nan, dtype=float),
                self.y_eval,
                self.g_y,
                [np.nan] * (self.N + 1),
            )

        F_errors = np.empty(self.N + 1, dtype=float)
        F_errors.fill(np.nan)

        for N in range(self.N + 1):
            weights_for_N = self.all_weights[: N + 1]
            if np.any(np.isnan(weights_for_N)):
                F_errors[N] = np.nan
                continue

            gN_approx = self.obj.g_approx(self.y_eval, N, weights_for_N)

            if np.any(np.isnan(gN_approx)):
                F_errors[N] = np.nan
            else:
                mse = np.mean((gN_approx - self.g_y) ** 2)
                F_errors[N] = mse
                # errors = np.abs(gN_approx - self.g_y)
                # F_errors[N] = np.maximum(errors)

        best_N = 0

        threshold_N = False
        for N in range(self.N + 1):
            if not np.isnan(F_errors[N]) and F_errors[N] < 1e-5:
                best_N = N
                threshold_N = True
                break

        # If no F_errors less than 1e-5 were found, then set best_N to the value
        # that makes the F_errors minimum (if valid errors exist)
        if not threshold_N and not np.all(np.isnan(F_errors)):
            try:
                best_N = int(np.nanargmin(F_errors))
            except ValueError:
                pass  # best_N remains 0 if all F_errors are NaN

        return best_N, F_errors[best_N]

    def plot_g(self):
        """
        Plot the actual function g and its N-th order Hermite expansion.
        """

        best_N, F_errors = self.optimization()
        weights_N = self.all_weights[: best_N + 1]
        gN_approx = self.obj.g_approx(self.y_plot, best_N, weights_N)

        plt.figure(figsize=(12, 7))

        # Plot actual g(y)
        plt.plot(
            self.y_plot,
            self.g_y_plot,
            label="Actual $g(y)$",
            color="blue",
            linestyle="-",
            linewidth=2,
        )

        # Plot Hermite approximation
        if np.any(np.isnan(gN_approx)):
            # print(f"Warning: Hermite Approximated g(y) for N*={N_optimal}
            # contains NaN values.")
            plt.plot(
                self.y_plot,
                gN_approx,
                label=f"Hermite Approx. $g(y)$ (N={best_N}, NaN present)",
                color="red",
                linestyle="--",
            )
        else:
            plt.plot(
                self.y_plot,
                gN_approx,
                label=f"Hermite Approx. $g(y)$ (N={best_N})",
                color="red",
                linestyle="--",
            )

        plt.title(f"Actual vs. Approximated $g(y)$ for T = {self.T:.4f}")
        plt.xlabel("$y$")
        plt.ylabel("$g(y)$")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.ylim(bottom=0)
        plt.axhline(0, color="black", linewidth=0.5)
        plt.axvline(0, color="black", linewidth=0.5)
        plt.show()


# --- One Dimensional Gauss-Hermite Quadrature Method---


def gauss_hermite_approx(N_gauss, kappa, model, p):
    """
    Compute the VIX derivative price using one dimensional Gauss-Hermite quadrature
    method.

    Parameters
    ----------
    N_gauss : int
        Number of nodes for the Gauss-Hermite quadrature.
    kappa : float
        The strike price of the VIX call/put.
    model : object
        The instance of a Bergomi model, mixed Bergomi model or mixed rough Bergomi
        model.
    p : int
        The payoff type: 1 for calls, 0 for futures, others for puts.

    Returns
    -------
    float
        The VIX derivative price.
    """
    mu1, mu2 = model.mean()
    sigma1, sigma2 = np.sqrt(model.variance())
    lam = model.lam
    X0 = model.X0
    T = model.T
    Delta = model.Delta

    nodes, weights = roots_hermite(N_gauss)

    # Change of variable: let z = sqrt(2)*x so that the expectation becomes
    # E[f(Z)] = 1/sqrt(pi) * \int_{-\infty}^{\infty} f(sqrt(2)*x) e^{-x^2} dx.
    z = np.sqrt(2) * nodes
    weights_trans = weights / np.sqrt(np.pi)

    F = lam * np.exp(mu1 + sigma1 * z) + (1 - lam) * np.exp(mu2 + sigma2 * z)

    if p == 1:
        payoff = np.maximum(np.sqrt(F) - kappa, 0)
    elif p == 0:
        payoff = np.sqrt(F)
    else:
        payoff = np.maximum(kappa - np.sqrt(F), 0)

    # The main term in the weak VIX derivative price approximation
    main = np.sum(weights_trans * payoff)

    # First order derivative of the payoff function
    def payoff_func(x):
        if p == 1:
            return np.where(x > kappa**2, 1 / (2 * np.sqrt(x)), 0)
        elif p == 0:
            return 1 / (2 * np.sqrt(x))
        else:
            return np.where(x < kappa**2, -1 / (2 * np.sqrt(x)), 0)

    # If the input instance is the mixed rough Bergomi model
    if isinstance(model, MixedRoughBergomi):
        vol1, vol2 = model.eta
        H = model.H
        term0 = 2 * H + 1
        term1 = (T + Delta) ** term0 - Delta**term0 - T**term0
        term2 = 2 * Delta * H * term0
        integral = term1 / term2
    else:
        vol1, vol2 = model.omega
        k = model.k
        term1 = 1 / (4 * k**2 * Delta)
        term2 = 1 - np.exp(-2 * k * T)
        term3 = 1 - np.exp(-2 * k * Delta)
        integral = term1 * term2 * term3

    term01 = lam * np.exp(mu1 + sigma1 * z)
    if vol1 == 0:
        term11 = (1 - lam) * np.exp(mu2 + sigma2 * z)
    else:
        term11 = (1 - lam) * np.exp(
            vol2 / 2 * (vol1 - vol2) * integral
            + (1 - vol2 / vol1) * X0
            + vol2 / vol1 * (mu1 + sigma1 * z)
        )
    psi1 = payoff_func(term01 + term11) * term01

    if vol2 == 0:
        term02 = lam * np.exp(mu1 + sigma1 * z)
    else:
        term02 = lam * np.exp(
            vol1 / 2 * (vol2 - vol1) * integral
            + (1 - vol1 / vol2) * X0
            + vol1 / vol2 * (mu2 + sigma2 * z)
        )
    term12 = (1 - lam) * np.exp(mu2 + sigma2 * z)
    psi2 = payoff_func(term02 + term12) * term12

    # Correction terms Pi,1
    p1 = np.zeros(3)

    p1[0] = np.sum(weights_trans * psi1)
    p1[1] = 1 / sigma1 * np.sum(weights_trans * z * psi1)
    p1[2] = 1 / sigma1**2 * np.sum(weights_trans * (z**2 - 1) * psi1)

    # Correction terms Pi,2
    p2 = np.zeros(3)

    p2[0] = np.sum(weights_trans * psi2)
    p2[1] = 1 / sigma2 * np.sum(weights_trans * z * psi2)
    p2[2] = 1 / sigma2**2 * np.sum(weights_trans * (z**2 - 1) * psi2)

    gamma = model.coefficients()

    if vol1 == 0:
        price = main + np.sum(gamma[:, 1] * p2)
    elif vol2 == 0:
        price = main + np.sum(gamma[:, 0] * p1)
    else:
        price = main + np.sum(gamma[:, 0] * p1) + np.sum(gamma[:, 1] * p2)

    return price


# --- Implied Volatility ---


def implied_vol(x, y, T, market):
    """
    Compute the implied volatility by the root-finding method.

    :param x: float
              The spot price of the underlying asset.
              Here is the VIX futures price.
    :param y: float
              The strike price of the option.
    :param T: float
              The maturity of the option.
    :param market: float
                   The call/put price obtained through the approximation or the market.
    """

    def objective(z):
        """
        The objective function of the root-finding method construct using the
        Black-Scholes call price function.

        :param z: float
                  The volatility in the Black-Scholes price function.
        """

        d1 = np.log(x / y) / (z * np.sqrt(T)) + z * np.sqrt(T) / 2
        d2 = d1 - z * np.sqrt(T)
        bs_call = x * norm.cdf(d1) - y * norm.cdf(d2)
        return bs_call - market

    # Brent's method to solve for the implied volatility
    try:
        implied_vol = brentq(objective, 1e-13, 7.0)  # Bounds for volatility (0 to 500%)
        return implied_vol
    except ValueError:
        return np.nan


# --- VIX Derivative Price Approximation ---


def main_mb():
    """
    Main function to define the mixed Bergomi model scenarios and trigger the analysis.
    """

    k = 1
    X0 = np.log(0.2**2)
    T = [1.0 / 12.0, 3.0 / 12.0, 6.0 / 12.0]
    Delta = 1.0 / 12.0
    kappa_list = np.linspace(0.1, 0.4, 10)

    N_max = 20
    N_gauss = 80
    M = 30000

    p = 1

    # Scenario 1
    omega1 = 0.5
    omega2 = 6
    lam = 0.3

    # Scenario 2
    omega1 = 10
    omega2 = 2
    lam = 0.2

    call_approx = np.zeros((len(T), len(kappa_list)))
    call_ref = np.zeros((len(T), len(kappa_list)))

    iv_approx_3 = np.zeros((3, len(T), len(kappa_list)))
    iv_approx = np.zeros((len(T), len(kappa_list)))
    iv_ref = np.zeros((len(T), len(kappa_list)))

    log_moneyness = np.zeros((len(T), len(kappa_list)))

    j = 0

    for t in T:
        print(f"Processing for T = {t:.4f}.")

        para = [omega1, omega2, lam, k, X0, t, Delta]
        mb = MixedBergomi(para)

        hermite_object = HermiteApproximation(kappa_list[0], mb, p)

        res_optimal = optimal_N(hermite_object, N_max, M, t)
        # res_optimal.plot_g()

        best_N, F_error = res_optimal.optimization()

        print(f"\nOptimal Hermite order, N = {best_N}.")
        print(f"\nThe sum of the squared error, Errors = {F_error}.")

        futures_gauss_hermite = gauss_hermite_approx(N_gauss, kappa_list[0], mb, p=0)
        futures_hermite = HermiteApproximation(
            kappa_list[0], mb, p=0
        ).vix_derivative_price(best_N)

        call_ref[j] = np.array(
            [gauss_hermite_approx(N_gauss, kappa, mb, p=1) for kappa in kappa_list]
        )
        call_approx[j] = np.array(
            [
                HermiteApproximation(kappa, mb, p=1).vix_derivative_price(best_N)
                for kappa in kappa_list
            ]
        )

        iv_ref[j] = np.array(
            [
                implied_vol(futures_hermite, kappa_list[i], t, call_approx[j, i])
                for i in range(len(kappa_list))
            ]
        )
        iv_approx[j] = np.array(
            [
                HermiteApproximation(kappa, mb, p=1).implied_vol_taylor(best_N, 2)[0]
                for kappa in kappa_list
            ]
        )

        # iv_approx[j] = np.array([HermiteApproximation(kappa, mb, p=1).implied_vol_bell_polynomial_e(best_N, 2) for kappa in kappa_list])

        log_moneyness[j] = np.log(futures_gauss_hermite / kappa_list)

        j += 1

    # plot_approx(call_approx, call_ref, kappa_list, "Call option price")
    # print_approx(call_approx, call_ref, kappa_list)
    plot_approx(iv_approx, iv_ref, kappa_list, "Implied volatility")
    # plot_approx(iv_approx, iv_ref, log_moneyness, "Implied volatility")


def main_mrb():
    """
    Main function to define the mixed rough Bergomi model scenarios and trigger
    the analysis.
    """

    H = 0.1
    X0 = np.log(0.235**2)
    T = [1.0 / 12.0, 3.0 / 12.0, 6.0 / 12.0]
    Delta = 1.0 / 12.0
    kappa_list = np.linspace(0.1, 0.4, 10)

    N_max = 20
    N_gauss = 80
    M = 30000

    p = 1

    # Scenario 1
    eta1 = 1.4
    eta2 = 0.7
    lam = 0.3

    # Scenario 2
    eta1 = 2
    eta2 = 0.2
    lam = 0.4

    call_approx = np.zeros((len(T), len(kappa_list)))
    call_ref = np.zeros((len(T), len(kappa_list)))

    iv_approx_3 = np.zeros((3, len(T), len(kappa_list)))
    iv_approx = np.zeros((len(T), len(kappa_list)))
    iv_ref = np.zeros((len(T), len(kappa_list)))

    j = 0

    for t in T:
        print(f"Processing for T = {t:.4f}.")

        para = [eta1, eta2, lam, H, X0, t, Delta]
        mrb = MixedRoughBergomi(para)

        hermite_object = HermiteApproximation(kappa_list[0], mrb, p)
        # hermite_object.parameters_out()

        res_optimal = optimal_N(hermite_object, N_max, M, t)
        # res_optimal.plot_g()

        best_N, F_error = res_optimal.optimization()

        print(f"\nOptimal Hermite order, N = {best_N}.")
        print(f"\nThe sum of the squared error, Errors = {F_error}.")

        futures_gauss_hermite = gauss_hermite_approx(N_gauss, kappa_list[0], mrb, p=0)
        futures_hermite = HermiteApproximation(
            kappa_list[0], mrb, p=0
        ).vix_derivative_price(best_N)

        call_ref[j] = np.array(
            [gauss_hermite_approx(N_gauss, kappa, mrb, p=1) for kappa in kappa_list]
        )
        call_approx[j] = np.array(
            [
                HermiteApproximation(kappa, mrb, p=1).vix_derivative_price(best_N)
                for kappa in kappa_list
            ]
        )

        iv_ref[j] = np.array(
            [
                implied_vol(futures_gauss_hermite, kappa_list[i], t, call_ref[j, i])
                for i in range(len(kappa_list))
            ]
        )
        # iv_approx[j] = np.array(
        #     [
        #         HermiteApproximation(kappa, mrb, p=1).implied_vol_taylor(best_N, 2)[0]
        #         for kappa in kappa_list
        #     ]
        # )

        # iv_approx[j] = np.array(
        #     [
        #         HermiteApproximation(kappa, mrb, p=1).implied_vol_bell_polynomial(
        #             best_N, 2
        #         )[0]
        #         for kappa in kappa_list
        #     ]
        # )
        iv_approx[j] = np.array(
            [
                HermiteApproximation(kappa, mrb, p=1).implied_vol_bell_polynomial_e(
                    best_N, 2
                )
                for kappa in kappa_list
            ]
        )

        j += 1

        # log_moneyess = np.log(futures_hermite / kappa_list)
        # print(futures_gauss_hermite)
        # print(futures_hermite)
        # print(log_moneyess)

    # plot_approx(call_approx, call_ref, kappa_list, "Call option price")
    # print_approx(call_approx, call_ref, kappa_list)
    plot_approx(iv_approx, iv_ref, kappa_list, "Implied volatility")
    # print(iv_approx, iv_ref, kappa_list)


if __name__ == "__main__":
    main_mrb()
