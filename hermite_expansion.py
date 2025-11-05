import numpy as np
import warnings
from math import factorial
from scipy.stats import norm
from scipy.optimize import brentq, fsolve
from scipy.integrate import quad, IntegrationWarning

import utils

# Clipping bounds to prevent overflow/underflow errors in np.exp()
EXP_LOWER_BOUND = -700
EXP_UPPER_BOUND = 700

# Small number to prevent division by zero
EPSILON = 1e-9


class HermiteExpansion:
    """
    Compute the VIX derivative price using the probabilist's Hermite polynomials.

    Parameters
    ----------
    model: The instance of a Bergomi model, either mixed Bergomi model or mixed rough Bergomi model.
    T: float
        Maturity.
    rule: int, optional
        The rule of the implied volatility expansion.
        rule=1, choose the higher volatility as the base,
        rule=-1, choose the lower volatility as the base
    """

    def __init__(self, model, T, rule):
        """
        Initializes the Hermite polynomial expansion with model parameters.
        See class docstring for parameter definitions.
        """
        if rule not in [-1, 1]:
            raise ValueError("The rule must be 1 or -1.")
        if T < 0:
            raise ValueError("Maturity T must be positive.")\
        
        self.model = model
        self.T = T
        self.r = rule

        self.lbd = model.lbd
        self.mu1 = model.inst1.mean(T)
        self.mu2 = model.inst2.mean(T)
        self.sigma1 = np.sqrt(model.inst1.variance(T))
        self.sigma2 = np.sqrt(model.inst2.variance(T))

        self.gamma_11 = model.inst1.gamma_1(T)
        self.gamma_21 = model.inst1.gamma_2(T)
        self.gamma_31 = model.inst1.gamma_3(T)
        self.gamma_12 = model.inst2.gamma_1(T)
        self.gamma_22 = model.inst2.gamma_2(T)
        self.gamma_32 = model.inst2.gamma_3(T)

        if rule == -1:
            self.a, self.b, self.c = model.abc(T, EXP_LOWER_BOUND, EXP_UPPER_BOUND)
        elif rule == 1:
            self.a, self.b, self.c = model.abc_reverse(T, EXP_LOWER_BOUND, EXP_UPPER_BOUND)


    def g(self, y):
        """
        Compute the function g, g (y) := ( 1 + b * exp (c * y))^(1 / 2).

        Parameters
        ----------
        y: float or np.ndarray
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
    
    def g1(self, y):
        """
        Compute the first-order partial derivative of g (y)
        with respect to the mean of each process
        in the mixed Bergomi model or mixed rough Bergomi model.

        Parameters
        ----------
        y: float or np.ndarray
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
            log_denominator = 0.5 * np.log1p(v_b_exp_val)  # Use log1p for precision: log(1+x)

            # Combine logs and exponential back
            log_g1 = log_numerator - log_denominator
            g1_val[valid_mask] = np.exp(np.clip(log_g1, EXP_LOWER_BOUND, EXP_UPPER_BOUND))
        
        if self.r == -1 and self.sigma1 <= self.sigma2: 
            return -g1_val.item() if is_scalar else -g1_val
        elif self.r == 1 and self.sigma1 >= self.sigma2:
            return -g1_val.item() if is_scalar else -g1_val
        else:
            return g1_val.item() if is_scalar else g1_val


    def g2(self, y):
        """
        Compute the second-order partial derivative of g (y)
        with respect to the mean of each process
        in the mixed Bergomi model or mixed rough Bergomi model.
        Using a numerically stable log-space computation to prevent overflow.

        Parameters
        ----------
        y: float or np.ndarray
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
            log_numerator = np.log(0.25 * self.b) + v_clipped_arg + np.log(2 + v_b_exp_val)
            log_denominator = 1.5 * np.log1p(v_b_exp_val)  # Use log1p for precision: log(1+x)

            # Combine logs and exponential back
            log_g2 = log_numerator - log_denominator
            g2_val[valid_mask] = np.exp(np.clip(log_g2, EXP_LOWER_BOUND, EXP_UPPER_BOUND))

        return g2_val.item() if is_scalar else g2_val

    def g3(self, y):
        """
        Compute the third-order partial derivative of g (y)
        with respect to the mean of each process
        in the mixed Bergomi model or mixed rough Bergomi model.
        Using a numerically stable log-space computation to prevent overflow.

        Parameters
        ----------
        y: float or np.ndarray
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
                log_of_num_term[small_x_mask] = np.log(4 + 2 * x_small + x_small ** 2)

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
            g3_val[valid_mask] = np.exp(np.clip(log_g3, EXP_LOWER_BOUND, EXP_UPPER_BOUND))

        if self.r == -1 and self.sigma1 <= self.sigma2:
            return -g3_val.item() if is_scalar else -g3_val
        elif self.r == 1 and self.sigma1 >= self.sigma2:
            return -g3_val.item() if is_scalar else -g3_val
        else:
            return g3_val.item() if is_scalar else g3_val

    def weight_calculation(self, n, func_g):
        """
        Compute the weights for the n-th order Hermite expansion of the main term and its derivatives.

        Parameters
        ----------
        n: int
            Order of the Hermite polynomial.
        func_g: callable
                The function g or its derivative.
                (e.g., self.g, self.g1)
        """

        integrand = lambda y: func_g(y) * utils.hermite_phi_product(n, y)

        integral = np.nan
        err = np.inf

        try:
            # First attempt: integrate over the full range
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always", IntegrationWarning)
                integral, err = quad(integrand, -np.inf, np.inf, epsabs=EPSILON, epsrel=EPSILON)

            # If the first attempt raised a warning, split at 0
            if w:
                # print("Initial integration warned. Splitting at 0.")
                with warnings.catch_warnings(record=True) as w2:
                    warnings.simplefilter("always", IntegrationWarning)
                    integral1, err1 = quad(integrand, -np.inf, 0, epsabs=EPSILON, epsrel=EPSILON)
                    integral2, err2 = quad(integrand, 0, np.inf, epsabs=EPSILON, epsrel=EPSILON)
                    integral = integral1 + integral2
                    err = err1 + err2

                # If splitting at 0 ALSO warned, split the negative part at -10
                if w2:
                    print("Second integration warned. Splitting at -10.")
                    # Note: No need for another warning catch here unless you have further nesting
                    integral1, err1 = quad(integrand, -np.inf, -10, epsabs=EPSILON, epsrel=EPSILON)
                    integral2, err2 = quad(integrand, -10, 0, epsabs=EPSILON, epsrel=EPSILON)
                    integral3, err3 = quad(integrand, 0, np.inf, epsabs=EPSILON, epsrel=EPSILON)
                    integral = integral1 + integral2 + integral3
                    err = err1 + err2 + err3

            # Final check on the estimated error
            if err > abs(integral) * 0.01 and err > 1e-7:
                pass  # Warning about high integration error

        except Exception as e:
            print(f"Warning: Integration failed for n={n} with function {func_g.__name__}. Error: {e}")
            integral = np.nan

        return integral / float(factorial(n))
    
    def g_proxy(self, y, n, weights=None):
        """
        Compute the nth-order hermite expansion for g(y).

        Parameters
        ----------
        y: float or np.ndarray
            Input value(s) for y.
        n: int
            The order of the Hermite expansion.
        weights: float or np.ndarray
                Pre-calculated weights for the Hermite expansion.
                If None, weights are calculated.
        """

        is_scalar = np.isscalar(y)
        y_arr = np.asarray(y, dtype=float)
        g_approx = np.zeros_like(y_arr, dtype=float)

        if weights is None:
            weights_n = [self.weight_calculation(i, self.g) for i in range(n + 1)]
        elif len(weights) != n + 1:
            raise ValueError(f"Provided weights length {len(weights)} doesn't match n+1 ({n + 1})")
        else:
            weights_n = weights

        for n, weight in enumerate(weights_n):
            if not np.isnan(weight):
                g_approx += weight * utils.prob_hermite_poly(n, y_arr)
            else:
                g_approx += np.full_like(y_arr, np.nan)
        return g_approx.item() if is_scalar else g_approx
    
    def optimal_order(self, n_max, n_mc):
        """
        Decide the optimal order of the Hermite expansion using the non-linear least square method.

        Parameters
        ----------
        n_max: int
            The maximum order of the Hermite expansion.
        n_mc: int
            The number of the Monte-Carlo samples.
        """

        y_eval = np.linspace(-10, 10, n_mc, dtype=float)
        g_y = self.g(y_eval)
        weights_n_max = [self.weight_calculation(n, self.g) for n in range(n_max + 1)]

        if np.any(np.isnan(g_y)):
            return 0, np.full(n_max + 1, np.nan, dtype=float), y_eval, g_y, [np.nan] * (n_max + 1)

        F_errors = np.empty(n_max + 1, dtype=float)
        F_errors.fill(np.nan)

        for n in range(n_max+ 1):
            weights_for_n = weights_n_max[: n + 1]

            if np.any(np.isnan(weights_for_n)):
                F_errors[n] = np.nan
                continue

            g_proxy_n = self.g_proxy(y_eval, n, weights_for_n)

            if np.any(np.isnan(g_proxy_n)):
                F_errors[n] = np.nan
            else:
                mse = np.mean((g_proxy_n - g_y) ** 2)
                F_errors[n] = mse

        optimal_n = 0

        threshold_n = False
        for n in range(n_max + 1):
            if not np.isnan(F_errors[n]) and F_errors[n] < 1e-5:
                optimal_n = n
                threshold_n = True
                break

        # If no F_errors less than 1e-5 were found, then set best_N to the value
        # that makes the F_errors minimum (if valid errors exist)
        if not threshold_n:
            if not np.all(np.isnan(F_errors)):
                try:
                    optimal_n = int(np.nanargmin(F_errors))
                except ValueError:
                    pass  # best_N remains 0 if all F_errors are NaN

        return optimal_n

    def d1_finding(self, kappa):
        """
        Find the value of d1 in the Black-Scholes price function, which is defined as 'A' in the paper.
        Use Brent's method or fsolve for the objective function.

        Parameters
        ----------
        kappa: float
            Strike price.
        """

        # Function F(x) = lambda * exp(mu1 + sigma1*x) + (1-lambda) * exp(mu2 + sigma2*x) - kappa^2
        def objective_func(x):
            term1 = self.lbd * np.exp(self.mu1 + self.sigma1 * x)
            term2 = (1 - self.lbd) * np.exp(self.mu2 + self.sigma2 * x)
            return term1 + term2 - kappa ** 2

        try:
            # Use brentq for robust root finding within a bracketing interval
            d1 = brentq(objective_func, -20, 20)  # Adjusted range for robustness
            return d1
        
        except ValueError:
            # Fallback to fsolve if brentq fails (e.g., no sign change in interval)
            print("Warning: brentq failed, falling back to fsolve. Consider adjusting brentq interval.")
            d1 = fsolve(objective_func, 0)[0]  # Initial guess at 0
            return d1
        
        except Exception as e:
            print(f"Error in d1_finding: {e}")
            return np.nan  # Return NaN if root finding completely fails

    def integral_n(self, kappa, opttype, weights_n):
        """
        Compute I_N, I_1N, I_2N, I_3N in the N-th Hermite expansionin for the VIX options price.
        
        Parameters
        ----------
        kappa: float
            Strike price.
        opttype: int, optional
            Option type: 1 for call, -1 for put, and 0 for futures.
        weights_n: float or np.ndarray
                Pre-calculated weights for the optimal n.
        """

        A = self.d1_finding(kappa)

        if self.r == -1:
            sigma_base = min(self.sigma1, self.sigma2)
        elif self.r == 1:
            sigma_base = max(self.sigma1, self.sigma2)

        B = A - sigma_base / 2

        if opttype in [-1, 1]:
            h = np.zeros_like(weights_n)
            h[0] = norm.cdf(-opttype * B)
            for i in range(1, len(weights_n)):
                h[i] = opttype * utils.hermite_phi_product(i-1, B)
        else:
            h = np.ones_like(weights_n)

        integral = weights_n * h

        return integral
    
    def c_0(self):
        """
        Compute the first coefficients in the Hermite expansion.
        """

        if (self.r == -1 and self.sigma1 < self.sigma2) or \
            (self.r == 1 and self.sigma1 > self.sigma2):
            c_0 = 1 + 0.5 * self.gamma_11 + 0.25 * self.gamma_21 + 0.125 * self.gamma_31
        else:
            c_0 = 1 + 0.5 * self.gamma_12 + 0.25 * self.gamma_22 + 0.125 * self.gamma_32
        
        return c_0 * self.a
    
    def c_1(self):
        """
        Compute the second coefficients in the Hermite expansion.
        """

        if (self.r == -1 and self.sigma1 < self.sigma2) or \
            (self.r == 1 and self.sigma1 > self.sigma2):
            c_1 = (self.gamma_11 - self.gamma_12
                   + self.gamma_21 * (1 - 0.5 * self.sigma2 / self.sigma1)
                   - self.gamma_22 * 0.5 * self.sigma1 / self.sigma2
                   + self.gamma_31 * (3 / 4 - 0.5 * self.sigma2 / self.sigma1)
                   - self.gamma_32 * 0.25 * self.sigma1**2 / self.sigma2**2)
        else:
            c_1 = (self.gamma_11 - self.gamma_12
                   + self.gamma_21 * 0.5 * self.sigma2 / self.sigma1
                   - self.gamma_22 * (1 - 0.5 * self.sigma1 / self.sigma2)
                   + self.gamma_31 * 0.25 * self.sigma2**2 / self.sigma1**2
                   - self.gamma_32 * (3 / 4 - 0.5 * self.sigma1 / self.sigma2))
            
        return c_1 * self.a

    def c_2(self):
        """
        Compute the third coefficients in the Hermite expansion.
        """

        if (self.r == -1 and self.sigma1 < self.sigma2) or \
            (self.r == 1 and self.sigma1 > self.sigma2):
            c_2 = (self.gamma_21 * (1 - self.sigma2 / self.sigma1)
                   + self.gamma_22 * (1 - self.sigma1 / self.sigma2)
                   + self.gamma_31 * (1.5 - 2 * self.sigma2 / self.sigma1 + 0.5 * self.sigma2**2 / self.sigma1**2)
                   + self.gamma_32 * (self.sigma1 / self.sigma2 - self.sigma1**2 / self.sigma2**2))
        else:
            c_2 = (self.gamma_21 * (1 - self.sigma2 / self.sigma1)
                   + self.gamma_22 * (1 - self.sigma1 / self.sigma2)
                   + self.gamma_31 * (self.sigma2 / self.sigma1 - self.sigma2**2 / self.sigma1**2)
                   + self.gamma_32 * (1.5 - 2 * self.sigma1 / self.sigma2 + 0.5 * self.sigma1**2 / self.sigma2**2))
            
        return c_2 * self.a

    def c_3(self):
        """
        Compute the forth coefficients in the Hermite expansion.
        """

        c_3 = (self.gamma_31 * (1 - 2 * self.sigma2 / self.sigma1 + self.sigma2**2 / self.sigma1**2)
               - self.gamma_32 * (1 - 2 * self.sigma1 / self.sigma2 + self.sigma1**2 / self.sigma2**2))
            
        return c_3 * self.a

    def vix_option_price_expan(self, kappa, opttype, n_max, n_mc):
        """
        Compute the VIX option price using the Hermite expansion.

        Parameters
        ----------
        kappa: float
            Strike price.
        opttype: int, optional
            Option type: 1 for call, -1 for put, and 0 for futures.
        n_max: int
            The maximum order of the Hermite expansion.
        n_mc: int
            The number of the Monte-Carlo samples.
        """

        A = self.d1_finding(kappa)

        optimal_n = self.optimal_order(n_max, n_mc)

        # If opttype = 0 (futures), optimal order is 0. Otherwise, use the provided optimal order.
        n = 0 if opttype == 0 else optimal_n

        weights_0 = np.array([self.weight_calculation(n, self.g) for n in range(n + 1)])
        weights_1 = np.array([self.weight_calculation(n, self.g1) for n in range(n + 1)])
        weights_2 = np.array([self.weight_calculation(n, self.g2) for n in range(n + 1)])
        weights_3 = np.array([self.weight_calculation(n, self.g3) for n in range(n + 1)])

        i_0 = np.sum(self.integral_n(kappa, opttype, weights_0))
        i_1 = np.sum(self.integral_n(kappa, opttype, weights_1))
        i_2 = np.sum(self.integral_n(kappa, opttype, weights_2))
        i_3 = np.sum(self.integral_n(kappa, opttype, weights_3))

        c_0 = self.c_0()
        c_1 = self.c_1()
        c_2 = self.c_2()
        c_3 = self.c_3()

        if opttype in [-1, 1]:
            price = opttype * (c_0 * i_0 + c_1 * i_1 + c_2 * i_2 + c_3 * i_3 
                               - kappa * norm.cdf(-opttype * A))
        else:
            price = c_0 * i_0 + c_1 * i_1 + c_2 * i_2 + c_3 * i_3

        return price

    def implied_vol_expan(self, k, n_max, n_mc, method):
        """
        Compute the implied volatility using the Hermite expansion.
        
        Parameters
        ----------
        k: float or ndarray
            Log-moneyness of the VIX option.
        opttype: int, optional
            Option type: 1 for call, -1 for put, and 0 for futures.
        n_max: int
            The maximum order of the Hermite expansion.
        n_mc: int
            The number of the Monte-Carlo samples.
        method: int, optional
            Rewritten form: 1 for defining the log-spot price, 2 for defing the log-strike price.
        """

        optimal_n = self.optimal_order(n_max, n_mc)

        weights_0 = np.array([self.weight_calculation(n, self.g) for n in range(optimal_n + 1)])
        weights_1 = np.array([self.weight_calculation(n, self.g1) for n in range(optimal_n + 1)])
        weights_2 = np.array([self.weight_calculation(n, self.g2) for n in range(optimal_n + 1)])
        weights_3 = np.array([self.weight_calculation(n, self.g3) for n in range(optimal_n + 1)])

        c_0 = self.c_0()
        c_1 = self.c_1()
        c_2 = self.c_2()
        c_3 = self.c_3()

        vix_fut = c_0 * weights_0[0] + c_1 * weights_1[0] + c_2 * weights_2[0] + c_3 * weights_3[0]

        kappa = vix_fut * np.exp(k)

        A = self.d1_finding(kappa)

        if (self.r == -1 and self.sigma1 < self.sigma2) or \
            (self.r == 1 and self.sigma1 > self.sigma2):
            B = A - self.sigma1 / 2
            iv0 = 0.5 * self.sigma1 / np.sqrt(self.T)

            if method == 1:
                k0 = np.log(kappa)
                x0 = k0 - B * self.sigma1 / 2 - self.sigma1**2 / 8
            else:
                x0 = np.log(vix_fut)
                k0 = x0 + B * self.sigma1 / 2 + self.sigma1**2 / 8
        else:
            B = A - self.sigma2 / 2
            iv0 = 0.5 * self.sigma2 / np.sqrt(self.T)

            if method == 1:
                k0 = np.log(kappa)
                x0 = k0 - B * self.sigma2 / 2 - self.sigma2**2 / 8
            else:
                x0 = np.log(vix_fut)
                k0 = x0 + B * self.sigma2 / 2 + self.sigma2**2 / 8
        
        he = [utils.prob_hermite_poly(n, B) for n in range(optimal_n)]

        coeff = c_0 * weights_0[1: ] + c_1 * weights_1[1: ] + c_2 * weights_2[1: ] + c_3 * weights_3[1: ]

        if method == 1:
            extra_term = (vix_fut - np.exp(x0)) * norm.cdf(-B)
        else:
            extra_term = (np.exp(k) - kappa) * norm.cdf(-A)

        iv1 = np.sum(coeff * he) / (np.exp(x0) * np.sqrt(self.T))
        iv2 = extra_term / (np.exp(x0) * np.sqrt(self.T) * norm.pdf(B))

        iv = iv0 + iv1
        iv_e = iv0 + iv1 + iv2

        return iv, iv_e
    

    def implied_vol_expan_fixed(self, k, method):
        """
        Compute the implied volatility using the Hermite expansion.
        
        Parameters
        ----------
        k: float or ndarray
            Log-moneyness of the VIX option.
        opttype: int, optional
            Option type: 1 for call, -1 for put, and 0 for futures.
        n_max: int
            The maximum order of the Hermite expansion.
        n_mc: int
            The number of the Monte-Carlo samples.
        method: int, optional
            Rewritten form: 1 for defining the log-spot price, 2 for defing the log-strike price.
        """

        optimal_n = 10

        weights_0 = np.array([self.weight_calculation(n, self.g) for n in range(optimal_n + 1)])
        weights_1 = np.array([self.weight_calculation(n, self.g1) for n in range(optimal_n + 1)])
        weights_2 = np.array([self.weight_calculation(n, self.g2) for n in range(optimal_n + 1)])
        weights_3 = np.array([self.weight_calculation(n, self.g3) for n in range(optimal_n + 1)])

        c_0 = self.c_0()
        c_1 = self.c_1()
        c_2 = self.c_2()
        c_3 = self.c_3()

        vix_fut = c_0 * weights_0[0] + c_1 * weights_1[0] + c_2 * weights_2[0] + c_3 * weights_3[0]

        kappa = vix_fut * np.exp(k)

        A = self.d1_finding(kappa)

        if (self.r == -1 and self.sigma1 < self.sigma2) or \
            (self.r == 1 and self.sigma1 > self.sigma2):
            B = A - self.sigma1 / 2
            iv0 = 0.5 * self.sigma1 / np.sqrt(self.T)

            if method == 1:
                k0 = np.log(kappa)
                x0 = k0 - B * self.sigma1 / 2 - self.sigma1**2 / 8
            else:
                x0 = np.log(vix_fut)
                k0 = x0 + B * self.sigma1 / 2 + self.sigma1**2 / 8
        else:
            B = A - self.sigma2 / 2
            iv0 = 0.5 * self.sigma2 / np.sqrt(self.T)

            if method == 1:
                k0 = np.log(kappa)
                x0 = k0 - B * self.sigma2 / 2 - self.sigma2**2 / 8
            else:
                x0 = np.log(vix_fut)
                k0 = x0 + B * self.sigma2 / 2 + self.sigma2**2 / 8
        
        he = [utils.prob_hermite_poly(n, B) for n in range(optimal_n)]

        coeff = c_0 * weights_0[1: ] + c_1 * weights_1[1: ] + c_2 * weights_2[1: ] + c_3 * weights_3[1: ]

        iv1 = np.sum(coeff * he) / (np.exp(x0) * np.sqrt(self.T))

        iv = iv0 + iv1

        return iv
    

    def implied_vol_expan_bell(self, kappa, n_max, n_mc, method):
        """
        Compute the implied volatility using the Hermite expansion.
        
        Parameters
        ----------
        kappa: float
            Strike price.
        opttype: int, optional
            Option type: 1 for call, -1 for put, and 0 for futures.
        n_max: int
            The maximum order of the Hermite expansion.
        n_mc: int
            The number of the Monte-Carlo samples.
        method: int, optional
            Rewritten form: 1 for defining the log-spot price, 2 for defing the log-strike price.
        """

        A = self.d1_finding(kappa)

        optimal_n = self.optimal_order(n_max, n_mc)

        weights_0 = np.array([self.weight_calculation(n, self.g) for n in range(optimal_n + 1)])
        weights_1 = np.array([self.weight_calculation(n, self.g1) for n in range(optimal_n + 1)])
        weights_2 = np.array([self.weight_calculation(n, self.g2) for n in range(optimal_n + 1)])
        weights_3 = np.array([self.weight_calculation(n, self.g3) for n in range(optimal_n + 1)])

        c_0 = self.c_0()
        c_1 = self.c_1()
        c_2 = self.c_2()
        c_3 = self.c_3()

        vix_fut = c_0 * weights_0[0] + c_1 * weights_1[0] + c_2 * weights_2[0] + c_3 * weights_3[0]

        if (self.r == -1 and self.sigma1 < self.sigma2) or \
            (self.r == 1 and self.sigma1 > self.sigma2):
            B = A - self.sigma1 / 2
            iv0 = 0.5 * self.sigma1 / np.sqrt(self.T)

            if method == 1:
                k = np.log(kappa)
                x0 = k - B * self.sigma1 / 2 - self.sigma1**2 / 8
            else:
                x0 = np.log(vix_fut)
                k = x0 + B * self.sigma1 / 2 + self.sigma1**2 / 8
        else:
            B = A - self.sigma2 / 2
            iv0 = 0.5 * self.sigma2 / np.sqrt(self.T)

            if method == 1:
                k = np.log(kappa)
                x0 = k - B * self.sigma2 / 2 - self.sigma2**2 / 8
            else:
                x0 = np.log(vix_fut)
                k = x0 + B * self.sigma2 / 2 + self.sigma2**2 / 8

        if method == 1:
            extra_term = (vix_fut - np.exp(x0)) * norm.cdf(-B)
        else:
            extra_term = (np.exp(k) - kappa) * norm.cdf(-A)

        he = np.array([utils.prob_hermite_poly(n, B) for n in range(optimal_n)])

        coeff = c_0 * weights_0[1: ] + c_1 * weights_1[1: ] + c_2 * weights_2[1: ] + c_3 * weights_3[1: ]

        v = coeff * he

        vega = np.exp(x0) * np.sqrt(self.T) * norm.pdf(B)\
        
        m = x0 - k
        divi1 = m**2 / (iv0**3 * self.T) - iv0 * self.T / 4
        divi2 = divi1**2 - 3 * m**2 / (iv0**4 * self.T) - self.T / 4
        divi3 = divi1**3 - 9 * m**4 / (iv0**7 * self.T**2) + 3 * m**2 / (2 * iv0**3) + 12 * m**2 / (iv0**5 * self.T) + 3 * iv0 * self.T**2 / 16

        iv1 = v[0] / vega
        iv2 = v[1] / vega - 0.5 * iv1**2 * divi1
        iv3 = v[2] / vega - iv1 * iv2 * divi1 - 1 / 6 * iv1**3 * divi2
        iv4 = v[3] / vega - (iv1 * iv3 + 0.5 * iv2**2) * divi1 - 0.5 * iv1**2 * iv2 * divi2 - 1 / 24 * iv1**4 * divi3
        iv5 = extra_term / vega

        iv = iv0 + iv1 + iv2 + iv3 + iv4
        iv_e = iv0 + iv1 + iv2 + iv3 + iv4 + iv5

        return iv, iv_e
