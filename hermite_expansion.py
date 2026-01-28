import numpy as np
import numba as nb
import warnings
from math import factorial
from scipy.stats import norm
from scipy.optimize import brentq, newton
from scipy.integrate import quad, IntegrationWarning

import utils

# Clipping bounds to prevent overflow/underflow errors in np.exp()
EXP_LOWER_BOUND = -700
EXP_UPPER_BOUND = 700

# Small number to prevent division by zero
EPSILON = 1e-9

@nb.jit(nopython=True, cache=True, fastmath=True)
def _numba_g(y_arr, b, c, lower, upper, epsilon):
    n = y_arr.shape[0]
    out = np.zeros(n, dtype=np.float64)
    
    for i in range(n):
        y = y_arr[i]
        arg_exp = c * y

        if arg_exp < lower: arg_exp = lower
        if arg_exp > upper: arg_exp = upper
        
        b_exp_val = b * np.exp(arg_exp)
        g_sq = 1.0 + b_exp_val
        
        if g_sq > epsilon:
            log_g = 0.5 * np.log1p(b_exp_val)

            if log_g < lower: log_g = lower
            if log_g > upper: log_g = upper
            
            out[i] = np.exp(log_g)
        else:
            out[i] = 0.0
            
    return out

@nb.jit(nopython=True, cache=True, fastmath=True)
def _numba_g1(y_arr, b, c, lower, upper, epsilon):
    n = y_arr.shape[0]
    out = np.zeros(n, dtype=np.float64)
    log_05_b = np.log(0.5 * b)
    
    for i in range(n):
        y = y_arr[i]
        arg_exp = c * y

        if arg_exp < lower: arg_exp = lower
        if arg_exp > upper: arg_exp = upper
        
        b_exp_val = b * np.exp(arg_exp)
        g_sq = 1.0 + b_exp_val
        
        if g_sq > epsilon:
            log_num = log_05_b + arg_exp
            log_den = 0.5 * np.log1p(b_exp_val)
            
            log_g1 = log_num - log_den
            
            if log_g1 < lower: log_g1 = lower
            if log_g1 > upper: log_g1 = upper
            
            out[i] = np.exp(log_g1)
        else:
            out[i] = 0.0
    return out

@nb.jit(nopython=True, cache=True, fastmath=True)
def _numba_g2(y_arr, b, c, lower, upper, epsilon):
    n = y_arr.shape[0]
    out = np.zeros(n, dtype=np.float64)
    log_025_b = np.log(0.25 * b)
    
    for i in range(n):
        y = y_arr[i]
        arg_exp = c * y
        if arg_exp < lower: arg_exp = lower
        if arg_exp > upper: arg_exp = upper
        
        b_exp_val = b * np.exp(arg_exp)
        g_sq = 1.0 + b_exp_val
        
        if g_sq > epsilon:
            log_num = log_025_b + arg_exp + np.log(2.0 + b_exp_val)
            log_den = 1.5 * np.log1p(b_exp_val)
            
            log_g2 = log_num - log_den
            
            if log_g2 < lower: log_g2 = lower
            if log_g2 > upper: log_g2 = upper
            
            out[i] = np.exp(log_g2)
        else:
            out[i] = 0.0
    return out

@nb.jit(nopython=True, cache=True, fastmath=True)
def _numba_g3(y_arr, b, c, lower, upper, epsilon):
    n = y_arr.shape[0]
    out = np.zeros(n, dtype=np.float64)
    log_0125_b = np.log(0.125 * b)
    large_x_threshold = 1e50
    
    for i in range(n):
        y = y_arr[i]
        arg_exp = c * y
        if arg_exp < lower: arg_exp = lower
        if arg_exp > upper: arg_exp = upper
        
        x = b * np.exp(arg_exp)
        g_sq = 1.0 + x
        
        if g_sq > epsilon:
            if x > large_x_threshold:

                log_x = np.log(b) + arg_exp
                inv_x = 1.0 / x
                log1p_arg = 2.0 * inv_x + (2.0 * inv_x)**2
                log_of_num_term = 2.0 * log_x + np.log1p(log1p_arg)
            else:
                log_of_num_term = np.log(4.0 + 2.0 * x + x**2)
            
            log_num = log_0125_b + arg_exp + log_of_num_term
            log_den = 2.5 * np.log1p(x)
            
            log_g3 = log_num - log_den
            
            if log_g3 < lower: log_g3 = lower
            if log_g3 > upper: log_g3 = upper
            
            out[i] = np.exp(log_g3)
        else:
            out[i] = 0.0
    return out

class HermiteExpansion:
    """
    Implementation of the Hermite expansion for the VIX option pricing and implied volatility approximation
    under the mixed Bergomi model and the mixed rough Bergomi model.

    Parameters
    ----------
    """

    def __init__(self, rule, mu1, mu2, sigma1, sigma2, lbd, gamma_11, gamma_21, gamma_31, gamma_12, gamma_22, gamma_32):
        """
        Initialize the Hemrite expansion.
        See class docstring for parameter definitions.
        """
        if rule not in [-1, 1]:
            raise ValueError("Hemite expansion opttype must be either -1 or 1.")
        
        self.rule = rule
        self.mu1 = mu1
        self.mu2 = mu2
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.lbd = lbd

        self.gamma_11 = gamma_11
        self.gamma_21 = gamma_21
        self.gamma_31 = gamma_31
        self.gamma_12 = gamma_12
        self.gamma_22 = gamma_22
        self.gamma_32 = gamma_32

        if (rule == -1 and sigma1 < sigma2) or \
            (rule == 1 and sigma1 > sigma2):
            self.a = lbd**0.5 * np.exp(mu1 / 2 + sigma1**2 / 8)
            b_factor = (1 - lbd) / lbd
            exp_arg_b = mu2 - mu1 + (sigma2 - sigma1) * sigma1 / 2
            self.b = b_factor * np.exp(np.clip(exp_arg_b, EXP_LOWER_BOUND, EXP_UPPER_BOUND))
            self.c = sigma2 - sigma1
            self.sign = -1.0
        else:
            self.a = (1 - lbd)**0.5 * np.exp(mu2 / 2 + sigma2**2 / 8)
            b_factor = lbd / (1 - lbd)
            exp_arg_b = mu1 - mu2 + (sigma1 - sigma2) * sigma2 / 2
            self.b = b_factor * np.exp(np.clip(exp_arg_b, EXP_LOWER_BOUND, EXP_UPPER_BOUND))
            self.c = sigma1 - sigma2
            self.sign = 1.0

    def coefficients(self):
        """
        Compute the coefficients in the Hermite expansion.
        """
        sigma1 = self.sigma1
        sigma2 = self.sigma2
        gamma_11 = self.gamma_11
        gamma_21 = self.gamma_21
        gamma_31 = self.gamma_31
        gamma_12 = self.gamma_12
        gamma_22 = self.gamma_22
        gamma_32 = self.gamma_32
        a = self.a

        if self.sign == -1.0:
            c_0 = 1 + 0.5 * gamma_11 + 0.25 * gamma_21 + 0.125 * gamma_31
            c_1 = (gamma_11 - gamma_12
                   + gamma_21 * (1 - 0.5 * sigma2 / sigma1)
                   - gamma_22 * 0.5 * sigma1 / sigma2
                   + gamma_31 * (3 / 4 - 0.5 * sigma2 / sigma1)
                   - gamma_32 * 0.25 * sigma1**2 / sigma2**2)
            c_2 = (gamma_21 * (1 - sigma2 / sigma1)
                   + gamma_22 * (1 - sigma1 / sigma2)
                   + gamma_31 * (1.5 - 2 * sigma2 / sigma1 + 0.5 * sigma2**2 / sigma1**2)
                   + gamma_32 * (sigma1 / sigma2 - sigma1**2 / sigma2**2))
        else:
            c_0 = 1 + 0.5 * gamma_12 + 0.25 * gamma_22 + 0.125 * gamma_32
            c_1 = (gamma_11 - gamma_12
                   + gamma_21 * 0.5 * sigma2 / sigma1
                   - gamma_22 * (1 - 0.5 * sigma1 / sigma2)
                   + gamma_31 * 0.25 * sigma2**2 / sigma1**2
                   - gamma_32 * (3 / 4 - 0.5 * sigma1 / sigma2))
            c_2 = (gamma_21 * (1 - sigma2 / sigma1)
                   + gamma_22 * (1 - sigma1 / sigma2)
                   + gamma_31 * (sigma2 / sigma1 - sigma2**2 / sigma1**2)
                   + gamma_32 * (1.5 - 2 * sigma1 / sigma2 + 0.5 * sigma1**2 / sigma2**2))

        c_3 = (gamma_31 * (1 - 2 * sigma2 / sigma1 + sigma2**2 / sigma1**2)
               - gamma_32 * (1 - 2 * sigma1 / sigma2 + sigma1**2 / sigma2**2))
        
        c_0 = c_0 * a
        c_1 = c_1 * a
        c_2 = c_2 * a
        c_3 = c_3 * a
        
        return c_0, c_1, c_2, c_3

    def g(self, y):
        """
        Compute the function g, g (y) := ( 1 + b * exp (c * y))^(1 / 2).

        Parameters
        ----------
        y: float or np.ndarray
            Input value(s) for y.
        """
        is_scalar = np.isscalar(y)
        y_arr = np.atleast_1d(y).astype(float)
        
        g_val = _numba_g(y_arr, self.b, self.c, EXP_LOWER_BOUND, EXP_UPPER_BOUND, EPSILON)

        return g_val.item() if is_scalar else g_val

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

        try:
            # First attempt: integrate over the full range
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always", IntegrationWarning)
                integral, err = quad(integrand, -np.inf, np.inf, epsabs=EPSILON, epsrel=EPSILON)

            # If the first attempt raised a warning, split at 0
            if w:
                with warnings.catch_warnings(record=True) as w2:
                    warnings.simplefilter("always", IntegrationWarning)
                    integral1, err1 = quad(integrand, -np.inf, 0, epsabs=EPSILON, epsrel=EPSILON)
                    integral2, err2 = quad(integrand, 0, np.inf, epsabs=EPSILON, epsrel=EPSILON)
                    integral = integral1 + integral2
                    err = err1 + err2

                # If splitting at 0 ALSO warned, split the negative part at -10
                if w2:
                    with warnings.catch_warnings(record=True) as w3:
                        warnings.simplefilter("always", IntegrationWarning)
                        integral1, err1 = quad(integrand, -np.inf, -10, epsabs=EPSILON, epsrel=EPSILON)
                        integral2, err2 = quad(integrand, -10, 0, epsabs=EPSILON, epsrel=EPSILON)
                        integral3, err3 = quad(integrand, 0, np.inf, epsabs=EPSILON, epsrel=EPSILON)
                        integral = integral1 + integral2 + integral3
                        err = err1 + err2 + err3

                        if w3:
                            warnings.filterwarnings('ignore', category=Warning, module='scipy')
                            integral1, err1 = quad(integrand, -np.inf, -10, epsabs=EPSILON, epsrel=EPSILON)
                            integral2, err2 = quad(integrand, -10, 0, epsabs=EPSILON, epsrel=EPSILON)
                            integral3, err3 = quad(integrand, 0, 10, epsabs=EPSILON, epsrel=EPSILON)
                            integral4, err4 = quad(integrand, 10, np.inf, epsabs=EPSILON, epsrel=EPSILON)
                            integral = integral1 + integral2 + integral3 + integral4
                            err = err1 + err2 + err3 + err4

            if err > abs(integral) * 0.01 and err > 1e-7:
                pass

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
            weights_n = [self.weight_calculation(i, self.g,) for i in range(n + 1)]
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
        n_mc : int
            Number of Monte Carlo paths.
        """
        n_mc = 10**6
        y_eval = np.linspace(-13, 13, n_mc, dtype=float)

        g_y = self.g(y_eval)
        weights_n_max = [self.weight_calculation(n, self.g) for n in range(n_max + 1)]

        if np.any(np.isnan(g_y)):
            return 0, np.full(n_max+1, np.nan, dtype=float), y_eval, g_y, [np.nan] * (n_max+1)

        F_errors = np.empty(n_max + 1, dtype=float)
        F_errors.fill(np.nan)

        for n in range(n_max+ 1):
            weights_for_n = weights_n_max[: n + 1]

            if np.any(np.isnan(weights_for_n)):
                F_errors[n] = np.nan
                continue

            g_proxy_n = self.g_proxy(y_eval, n, weights=weights_for_n)

            if np.any(np.isnan(g_proxy_n)):
                F_errors[n] = np.nan
            else:
                mse = np.mean((g_proxy_n - g_y) ** 2)
                F_errors[n] = mse

        optimal_n = 0

        threshold_n = False
        for n in range(n_max + 1):
            if not np.isnan(F_errors[n]) and F_errors[n] < 1e-7:
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
        y_arr = np.atleast_1d(y).astype(float)
        
        g1_val = _numba_g1(y_arr, self.b, self.c, EXP_LOWER_BOUND, EXP_UPPER_BOUND, EPSILON)
            
        g1_val = self.sign * g1_val
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
        y_arr = np.atleast_1d(y).astype(float)
        
        g2_val = _numba_g2(y_arr, self.b, self.c, EXP_LOWER_BOUND, EXP_UPPER_BOUND, EPSILON)
        
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
        y_arr = np.atleast_1d(y).astype(float)
        
        g3_val = _numba_g3(y_arr, self.b, self.c, EXP_LOWER_BOUND, EXP_UPPER_BOUND, EPSILON)
            
        g3_val = self.sign * g3_val
        return g3_val.item() if is_scalar else g3_val
    
    def d1_finding(self, kappa):
        """
        Find the value of d1 in the Black-Scholes price function, which is defined as 'A' in the paper.
        Use Newton or Brent's method for the objective function.

        Parameters
        ----------
        kappa: float or nd.array
            Strike price(s).
        """

        kappas = np.asarray(kappa)
        orig_shape = kappas.shape
        kappas_flat = kappas.ravel()

        def f(x, k_val):
            arg1 = self.mu1 + self.sigma1 * x
            arg2 = self.mu2 + self.sigma2 * x
            
            if arg1 > EXP_UPPER_BOUND or arg2 > EXP_UPPER_BOUND:
                raise OverflowError("Exponent too large for Newton")
            elif arg1 < EXP_LOWER_BOUND or arg2 < EXP_LOWER_BOUND:
                raise OverflowError("Exponent too small for Newton")
                
            val = self.lbd * np.exp(arg1) + (1 - self.lbd) * np.exp(arg2) - k_val**2
            return val

        def f_prime(x, k_val):
            arg1 = self.mu1 + self.sigma1 * x
            arg2 = self.mu2 + self.sigma2 * x
            
            if arg1 > EXP_UPPER_BOUND or arg2 > EXP_UPPER_BOUND:
                raise OverflowError("Exponent too large for Newton")
            elif arg1 < EXP_LOWER_BOUND or arg2 < EXP_LOWER_BOUND:
                raise OverflowError("Exponent too small for Newton")
                
            term1 = self.lbd * self.sigma1 * np.exp(arg1)
            term2 = (1 - self.lbd) * self.sigma2 * np.exp(arg2)
            
            deriv = term1 + term2
            
            if deriv <= 1e-16:
                raise RuntimeError("Derivative too close to zero")
                
            return deriv

        def solve_single(k_val):

            if k_val <= 1e-8: 
                return -999.0 

            safe_lbd = max(self.lbd, 1e-10)

            if self.sigma1 > 1e-6:
                guess = (2 * np.log(k_val) - np.log(safe_lbd) - self.mu1) / self.sigma1
            else:
                guess = 0.0
            
            try:
                root = newton(f, x0=guess, fprime=f_prime, args=(k_val,), maxiter=50)
                if not np.isfinite(root):
                    raise ValueError("NaN result")
                return root
                
            except (RuntimeError, OverflowError, ValueError, ArithmeticError):
                pass

            def f_brent(x):
                arg1 = self.mu1 + self.sigma1 * x
                arg2 = self.mu2 + self.sigma2 * x

                if arg1 > EXP_UPPER_BOUND or arg2 > EXP_UPPER_BOUND:
                    return 1e100 
                return self.lbd * np.exp(arg1) + (1 - self.lbd) * np.exp(arg2) - k_val**2
            
            step = 0.5
            low, high = guess, guess
            val_guess = f_brent(guess)
            limit = 50

            if val_guess < 0:
                for _ in range(limit):
                    high += step
                    step *= 2.0
                    if f_brent(high) > 0:
                        break
            else:
                for _ in range(limit):
                    low -= step
                    step *= 2.0
                    if f_brent(low) < 0:
                        break
            
            try:
                return brentq(f_brent, low, high, xtol=1e-4, maxiter=100)
            except Exception:
                return np.nan

        d1s = np.empty_like(kappas_flat, dtype=float)
        for i, k_val in enumerate(kappas_flat):
            d1s[i] = solve_single(float(k_val))

        d1s = d1s.reshape(orig_shape)
        
        if d1s.ndim == 0:
            return float(d1s.item())
        return d1s
    
    def integral_n(self, kappa, opttype, weights_n=None):
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

        kappa = np.atleast_1d(np.asarray(kappa))

        if weights_n is not None:
            weights_n = np.atleast_1d(np.asarray(weights_n, dtype=float))

        n_weights = int(weights_n.shape[0])

        if self.rule == -1:
            sigma_base = min(self.sigma1, self.sigma2)
        elif self.rule == 1:
            sigma_base = max(self.sigma1, self.sigma2)

        if opttype in [-1, 1]:
            A = self.d1_finding(kappa)
            B = A - sigma_base / 2 
            
            h = np.zeros((n_weights, kappa.shape[0]))
            h[0] = norm.cdf(-opttype * B)
            for i in range(1, n_weights):
                h[i] = opttype * utils.hermite_phi_product(i-1, B)
        else:
            h = np.ones((n_weights, kappa.shape[0]))

        integral = weights_n.reshape(-1, 1) * h

        return integral
    
    def vix_opt_price(self, kappa, opttype, n_max, n_mc, optimal_n=None):
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
        n_mc : int
            Number of Monte Carlo paths.
        optimal_n : int
            The optimal order for the Hermite expansion. (default is None)
        """
        kappa = np.atleast_1d(np.asarray(kappa))

        if optimal_n == None:
            optimal_n = self.optimal_order(n_max, n_mc)

        n = 0 if opttype == 0 else optimal_n

        weights_0 = np.array([self.weight_calculation(n, self.g) for n in range(n + 1)])
        weights_1 = np.array([self.weight_calculation(n, self.g1) for n in range(n + 1)])
        weights_2 = np.array([self.weight_calculation(n, self.g2) for n in range(n + 1)])
        weights_3 = np.array([self.weight_calculation(n, self.g3) for n in range(n + 1)])

        c_0, c_1, c_2, c_3 = self.coefficients()

        if opttype == 0:
            price = c_0 * weights_0[0] + c_1 * weights_1[0] + c_2 * weights_2[0] + c_3 * weights_3[0]
        else: 
            i_0 = np.sum(self.integral_n(kappa, opttype, weights_0), axis=0)
            i_1 = np.sum(self.integral_n(kappa, opttype, weights_1), axis=0)
            i_2 = np.sum(self.integral_n(kappa, opttype, weights_2), axis=0)
            i_3 = np.sum(self.integral_n(kappa, opttype, weights_3), axis=0)
            A = self.d1_finding(kappa)
            price = opttype * (c_0 * i_0 + c_1 * i_1 + c_2 * i_2 + c_3 * i_3 
                               - kappa * norm.cdf(-opttype * A))

        return price
    
    def vix_implied_vol(self, k, T, n_max, n_mc, formula, optimal_n=None):
        """
        Compute the implied volatility using the Hermite expansion.
        
        Parameters
        ----------
        k: float or ndarray
            Log-moneyness of the VIX option.
        T: float
            Maturity.
        n_max: int
            The maximum order of the Hermite expansion.
        n_mc : int
            Number of Monte Carlo paths.
        formula: int, optional
            Rewritten form: 
            1 for defining the log-spot price,
            2 for defining the log-strike price,
            3 for defining the both.
        optimal_n : int
            The optimal order for the Hermite expansion. (default is None)
        return_opt: str, optional
            If 'iv', return only the implied volatility.
            If 'all', return both the futures price and the implied volatility.
        """
        if optimal_n == None:
            optimal_n = self.optimal_order(n_max, n_mc)

        weights_0 = np.array([self.weight_calculation(n, self.g) for n in range(optimal_n + 1)])
        weights_1 = np.array([self.weight_calculation(n, self.g1) for n in range(optimal_n + 1)])
        weights_2 = np.array([self.weight_calculation(n, self.g2) for n in range(optimal_n + 1)])
        weights_3 = np.array([self.weight_calculation(n, self.g3) for n in range(optimal_n + 1)])

        c_0, c_1, c_2, c_3 = self.coefficients()
        vix_fut = c_0 * weights_0[0] + c_1 * weights_1[0] + c_2 * weights_2[0] + c_3 * weights_3[0]

        kappa = vix_fut * np.exp(k)

        A = self.d1_finding(kappa)

        if self.sign == -1.0:
            B = A - self.sigma1 / 2
            iv0 = 0.5 * self.sigma1 / np.sqrt(T)
            k1 = np.log(kappa)
            x1 = k1 - B * self.sigma1 / 2 - self.sigma1**2 / 8
            x2 = np.log(vix_fut)
            k2 = x2 + B * self.sigma1 / 2 + self.sigma1**2 / 8
            x3 = np.log(self.a * weights_0[0])
            k3 = x3 + B * self.sigma1 / 2 + self.sigma1**2 / 8
        else:
            B = A - self.sigma2 / 2
            iv0 = 0.5 * self.sigma2 / np.sqrt(T)
            k1 = np.log(kappa)
            x1 = k1 - B * self.sigma2 / 2 - self.sigma2**2 / 8
            x2 = np.log(vix_fut)
            k2 = x2 + B * self.sigma2 / 2 + self.sigma2**2 / 8
            x3 = np.log(self.a * weights_0[0])
            k3 = x3 + B * self.sigma2 / 2 + self.sigma2**2 / 8

        if formula == 1:
            x0 = x1
            k0 = k1
            extra_term = (vix_fut - np.exp(x0)) * norm.cdf(-B)
        elif formula == 2:
            x0 = x2
            k0 = k2
            extra_term = (np.exp(k0) - kappa) * norm.cdf(-A)
        else:
            x0 = 0.5 * (x1 + x2)
            k0 = 0.5 * (k1 + k2)
            # x0 = x3
            # k0 = k3
            extra_term = ((vix_fut - np.exp(x0)) * norm.cdf(-B)
                         + (np.exp(k0) - kappa) * norm.cdf(-A))
            
        if optimal_n > 0:
            he = np.vstack([utils.prob_hermite_poly(n, B) for n in range(optimal_n)])
            coeff = c_0 * weights_0[1: ] + c_1 * weights_1[1: ] + c_2 * weights_2[1: ] + c_3 * weights_3[1: ]
            iv1 = (coeff[:, None] * he).sum(axis=0) / (np.exp(x0) * np.sqrt(T))
        else:
            iv1 = 0.0
            
        iv2 = extra_term / (np.exp(x0) * np.sqrt(T) * norm.pdf(B))

        iv = iv0 + iv1
        iv_e = iv0 + iv1 + iv2

        return vix_fut, iv, iv_e
    
    def discarded_term_opt_price(self, k, T, n_max, n_mc, formula, optimal_n=None):
        """
        """
        if optimal_n == None:
            optimal_n = self.optimal_order(n_max, n_mc)

        weights_0 = np.array([self.weight_calculation(n, self.g) for n in range(optimal_n + 1)])
        weights_1 = np.array([self.weight_calculation(n, self.g1) for n in range(optimal_n + 1)])
        weights_2 = np.array([self.weight_calculation(n, self.g2) for n in range(optimal_n + 1)])
        weights_3 = np.array([self.weight_calculation(n, self.g3) for n in range(optimal_n + 1)])

        c_0, c_1, c_2, c_3 = self.coefficients()
        vix_fut = c_0 * weights_0[0] + c_1 * weights_1[0] + c_2 * weights_2[0] + c_3 * weights_3[0]

        kappa = vix_fut * np.exp(k)

        A = self.d1_finding(kappa)

        if self.sign == -1.0:
            B = A - self.sigma1 / 2
            k1 = np.log(kappa)
            x1 = k1 - B * self.sigma1 / 2 - self.sigma1**2 / 8
            x2 = np.log(vix_fut)
            k2 = x2 + B * self.sigma1 / 2 + self.sigma1**2 / 8
            x3 = np.log(self.a * weights_0[0])
            k3 = x3 + B * self.sigma1 / 2 + self.sigma1**2 / 8
        else:
            B = A - self.sigma2 / 2
            k1 = np.log(kappa)
            x1 = k1 - B * self.sigma2 / 2 - self.sigma2**2 / 8
            x2 = np.log(vix_fut)
            k2 = x2 + B * self.sigma2 / 2 + self.sigma2**2 / 8
            x3 = np.log(self.a * weights_0[0])
            k3 = x3 + B * self.sigma2 / 2 + self.sigma2**2 / 8

        if formula == 1:
            x0 = x1
            k0 = k1
            extra_term = (vix_fut - np.exp(x0)) * norm.cdf(-B)
        elif formula == 2:
            x0 = x2
            k0 = k2
            extra_term = (np.exp(k0) - kappa) * norm.cdf(-A)
        else:
            x0 = 0.5 * (x1 + x2)
            k0 = 0.5 * (k1 + k2)
            extra_term = ((vix_fut - np.exp(x0)) * norm.cdf(-B)
                         + (np.exp(k0) - kappa) * norm.cdf(-A))

        return extra_term
    
    def weight_calculation_proxy(self, n, func_g, n_quad=80):
        """
        Compute the weights for the n-th order Hermite expansion using Gauss-Hermite quadrature.
        Optimized to replace scipy.integrate.quad.

        Parameters
        ----------
        n: int
            Order of the Hermite polynomial.
        func_g: callable
                The function g or its derivative.
        n_quad: int, optional
            Number of quadrature points (default 100).
        """

        nodes, weights = np.polynomial.hermite.hermgauss(n_quad)
        
        y_nodes = np.sqrt(2.0) * nodes
        g_vals = func_g(y_nodes)
        h_vals = utils.prob_hermite_poly(n, y_nodes)
        
        integral = (1.0 / np.sqrt(np.pi)) * np.sum(weights * g_vals * h_vals)

        return integral / float(factorial(n))
    
    def vix_opt_price_cal(self, kappa, opttype, n_max, n_mc, optimal_n=None):
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
        n_mc : int
            Number of Monte Carlo paths.
        optimal_n : int
            The optimal order for the Hermite expansion. (default is None)
        """
        kappa = np.atleast_1d(np.asarray(kappa))

        if optimal_n == None:
            optimal_n = self.optimal_order(n_max, n_mc)

        n = 0 if opttype == 0 else optimal_n

        weights_0 = np.array([self.weight_calculation_proxy(n, self.g) for n in range(n + 1)])
        weights_1 = np.array([self.weight_calculation_proxy(n, self.g1) for n in range(n + 1)])
        weights_2 = np.array([self.weight_calculation_proxy(n, self.g2) for n in range(n + 1)])
        weights_3 = np.array([self.weight_calculation_proxy(n, self.g3) for n in range(n + 1)])

        c_0, c_1, c_2, c_3 = self.coefficients()
        
        if opttype == 0:
            price = c_0 * weights_0[0] + c_1 * weights_1[0] + c_2 * weights_2[0] + c_3 * weights_3[0]
        else:
            i_0 = np.sum(self.integral_n(kappa, opttype, weights_0), axis=0)
            i_1 = np.sum(self.integral_n(kappa, opttype, weights_1), axis=0)
            i_2 = np.sum(self.integral_n(kappa, opttype, weights_2), axis=0)
            i_3 = np.sum(self.integral_n(kappa, opttype, weights_3), axis=0)
            A = self.d1_finding_proxy(kappa)
            price = opttype * (c_0 * i_0 + c_1 * i_1 + c_2 * i_2 + c_3 * i_3 
                               - kappa * norm.cdf(-opttype * A))

        return price
    
    def vix_implied_vol_cal(self, k, T, n_max, n_mc, formula, optimal_n=None):
        """
        Compute the implied volatility using the Hermite expansion.
        
        Parameters
        ----------
        k: float or ndarray
            Log-moneyness of the VIX option.
        T: float
            Maturity.
        n_max: int
            The maximum order of the Hermite expansion.
        n_mc : int
            Number of Monte Carlo paths.
        formula: int, optional
            Rewritten form: 
            1 for defining the log-spot price,
            2 for defining the log-strike price,
            3 for defining the both.
        optimal_n : int
            The optimal order for the Hermite expansion. (default is None)
        return_opt: str, optional
            If 'iv', return only the implied volatility.
            If 'all', return both the futures price and the implied volatility.
        """
        if optimal_n == None:
            optimal_n = self.optimal_order(n_max, n_mc)

        weights_0 = np.array([self.weight_calculation_proxy(n, self.g) for n in range(optimal_n + 1)])
        weights_1 = np.array([self.weight_calculation_proxy(n, self.g1) for n in range(optimal_n + 1)])
        weights_2 = np.array([self.weight_calculation_proxy(n, self.g2) for n in range(optimal_n + 1)])
        weights_3 = np.array([self.weight_calculation_proxy(n, self.g3) for n in range(optimal_n + 1)])

        c_0, c_1, c_2, c_3 = self.coefficients()
        vix_fut = c_0 * weights_0[0] + c_1 * weights_1[0] + c_2 * weights_2[0] + c_3 * weights_3[0]

        kappa = vix_fut * np.exp(k)

        A = self.d1_finding(kappa)

        if self.sign == -1.0:
            B = A - self.sigma1 / 2
            iv0 = 0.5 * self.sigma1 / np.sqrt(T)
            k1 = np.log(kappa)
            x1 = k1 - B * self.sigma1 / 2 - self.sigma1**2 / 8
            x2 = np.log(vix_fut)
            k2 = x2 + B * self.sigma1 / 2 + self.sigma1**2 / 8
        else:
            B = A - self.sigma2 / 2
            iv0 = 0.5 * self.sigma2 / np.sqrt(T)
            k1 = np.log(kappa)
            x1 = k1 - B * self.sigma2 / 2 - self.sigma2**2 / 8
            x2 = np.log(vix_fut)
            k2 = x2 + B * self.sigma2 / 2 + self.sigma2**2 / 8

        if formula == 1:
            x0 = x1
            k0 = k1
            extra_term = (vix_fut - np.exp(x0)) * norm.cdf(-B)
        elif formula == 2:
            x0 = x2
            k0 = k2
            extra_term = (np.exp(k0) - kappa) * norm.cdf(-A)
        else:
            x0 = 0.5 * (x1 + x2)
            k0 = 0.5 * (k1 + k2)
            extra_term = ((vix_fut - np.exp(x0)) * norm.cdf(-B)
                         + (np.exp(k0) - kappa) * norm.cdf(-A))
            
        if optimal_n > 0:
            he = np.vstack([utils.prob_hermite_poly(n, B) for n in range(optimal_n)])
            coeff = c_0 * weights_0[1: ] + c_1 * weights_1[1: ] + c_2 * weights_2[1: ] + c_3 * weights_3[1: ]
            iv1 = (coeff[:, None] * he).sum(axis=0) / (np.exp(x0) * np.sqrt(T))
        else:
            iv1 = 0.0
            
        iv2 = extra_term / (np.exp(x0) * np.sqrt(T) * norm.pdf(B))

        iv = iv0 + iv1
        iv_e = iv0 + iv1 + iv2

        return vix_fut, iv, iv_e