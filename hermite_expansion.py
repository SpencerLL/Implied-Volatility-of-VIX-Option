import numpy as np
import numba as nb
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
            
            # clip result
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


class OptimalOrder:

    def __init__(self, model, T, rule, n_max, n_mc):
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
        self.n_max = n_max
        self.n_mc = n_mc

        self.a, self.b, self.c = model.abc(T, rule, EXP_LOWER_BOUND, EXP_UPPER_BOUND)
    
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
   
    
    def optimal_order(self):
        """
        Decide the optimal order of the Hermite expansion using the non-linear least square method.

        Parameters
        ----------
        n_max: int
            The maximum order of the Hermite expansion.
        n_mc: int
            The number of the Monte-Carlo samples.
        """

        y_eval = np.linspace(-10, 10, self.n_mc, dtype=float)
        g_y = self.g(y_eval)
        weights_n_max = [self.weight_calculation(n, self.g) for n in range(self.n_max + 1)]

        if np.any(np.isnan(g_y)):
            return 0, np.full(self.n_max + 1, np.nan, dtype=float), y_eval, g_y, [np.nan] * (self.n_max + 1)

        F_errors = np.empty(self.n_max + 1, dtype=float)
        F_errors.fill(np.nan)

        for n in range(self.n_max+ 1):
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
        for n in range(self.n_max + 1):
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

    def __init__(self, model, T, rule, optimal_order):
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
        self.optimal_order = optimal_order

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

        self.a, self.b, self.c = model.abc(T, rule, EXP_LOWER_BOUND, EXP_UPPER_BOUND)


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
        
        sign = 1.0
        if self.r == -1 and self.sigma1 <= self.sigma2:
            sign = -1.0
        elif self.r == 1 and self.sigma1 >= self.sigma2:
            sign = -1.0
            
        g1_val = sign * g1_val
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
        
        sign = 1.0
        if self.r == -1 and self.sigma1 <= self.sigma2:
            sign = -1.0
        elif self.r == 1 and self.sigma1 >= self.sigma2:
            sign = -1.0
            
        g3_val = sign * g3_val
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

            # Final check on the estimated error
            if err > abs(integral) * 0.01 and err > 1e-7:
                pass  # Warning about high integration error

        except Exception as e:
            print(f"Warning: Integration failed for n={n} with function {func_g.__name__}. Error: {e}")
            integral = np.nan

        return integral / float(factorial(n))
    

    def d1_finding(self, kappa):
        """
        Find the value of d1 in the Black-Scholes price function, which is defined as 'A' in the paper.
        Use Brent's method or fsolve for the objective function.

        Parameters
        ----------
        kappa: float or nd.array
            Strike price(s).
        """
        kappas = np.asarray(kappa)
        orig_shape = kappas.shape
        kappas_flat = kappas.ravel()
        
        def objective_func(x, kappa):
            term1 = self.lbd * np.exp(self.mu1 + self.sigma1 * x)
            term2 = (1 - self.lbd) * np.exp(self.mu2 + self.sigma2 * x)
            return term1 + term2 - kappa ** 2

        def root_finding(kappa):
            a, b = -20.0, 20.0
            try:
                fa = objective_func(a, kappa)
                fb = objective_func(b, kappa)
                if np.sign(fa) == np.sign(fb):
                    for m in [40, 60, 80, 100]:
                        a2, b2 = -m, m
                        fa2, fb2 = objective_func(a2, kappa), objective_func(b2, kappa)
                        if np.sign(fa2) != np.sign(fb2):
                            a, b = a2, b2
                            break

                if np.sign(objective_func(a, kappa)) != np.sign(objective_func(b, kappa)):
                    return brentq(lambda x: objective_func(x, kappa), a, b)
            except Exception:
                pass

            try:
                root = fsolve(lambda x: objective_func(x, kappa), x0=0.0)
                return float(root[0])
            except Exception:
                return np.nan
            
        d1s = np.empty_like(kappas_flat, dtype=float)
        for i, k_i in enumerate(kappas_flat):
            d1s[i] = root_finding(float(k_i))

        
        d1s = d1s.reshape(orig_shape)
   
        if np.isscalar(kappa):
            return float(d1s.item())
        return d1s


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
        kappa = np.atleast_1d(np.asarray(kappa))
        weights_n = np.atleast_1d(np.asarray(weights_n, dtype=float))
        n_weights = int(weights_n.shape[0])

        if self.r == -1:
            sigma_base = min(self.sigma1, self.sigma2)
        elif self.r == 1:
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

    def vix_option_price_expan(self, kappa, opttype):
        """
        Compute the VIX option price using the Hermite expansion.

        Parameters
        ----------
        kappa: float
            Strike price.
        opttype: int, optional
            Option type: 1 for call, -1 for put, and 0 for futures.
        """
        kappa = np.atleast_1d(np.asarray(kappa))

        optimal_n = self.optimal_order

        n = 0 if opttype == 0 else optimal_n

        weights_0 = np.array([self.weight_calculation(n, self.g) for n in range(n + 1)])
        weights_1 = np.array([self.weight_calculation(n, self.g1) for n in range(n + 1)])
        weights_2 = np.array([self.weight_calculation(n, self.g2) for n in range(n + 1)])
        weights_3 = np.array([self.weight_calculation(n, self.g3) for n in range(n + 1)])

        i_0 = np.sum(self.integral_n(kappa, opttype, weights_0), axis=0)
        i_1 = np.sum(self.integral_n(kappa, opttype, weights_1), axis=0)
        i_2 = np.sum(self.integral_n(kappa, opttype, weights_2), axis=0)
        i_3 = np.sum(self.integral_n(kappa, opttype, weights_3), axis=0)

        c_0, c_1, c_2, c_3 = self.c_0(), self.c_1(), self.c_2(), self.c_3()

        combo = c_0 * i_0 + c_1 * i_1 + c_2 * i_2 + c_3 * i_3

        if opttype == 0:
            price = combo
        else: 
            A = self.d1_finding(kappa)
            price = opttype * (c_0 * i_0 + c_1 * i_1 + c_2 * i_2 + c_3 * i_3 
                               - kappa * norm.cdf(-opttype * A))

        return price

    def implied_vol_expan(self, k, method):
        """
        Compute the implied volatility using the Hermite expansion.
        
        Parameters
        ----------
        k: float or ndarray
            Log-moneyness of the VIX option.
        opttype: int, optional
            Option type: 1 for call, -1 for put, and 0 for futures.
        method: int, optional
            Rewritten form: 1 for defining the log-spot price, 2 for defing the log-strike price.
        """

        optimal_n = self.optimal_order

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

            k1 = np.log(kappa)
            x1 = k1 - B * self.sigma1 / 2 - self.sigma1**2 / 8
            x2 = np.log(vix_fut)
            k2 = x2 + B * self.sigma1 / 2 + self.sigma1**2 / 8
        else:
            B = A - self.sigma2 / 2
            iv0 = 0.5 * self.sigma2 / np.sqrt(self.T)

            k1 = np.log(kappa)
            x1 = k1 - B * self.sigma2 / 2 - self.sigma2**2 / 8
            x2 = np.log(vix_fut)
            k2 = x2 + B * self.sigma2 / 2 + self.sigma2**2 / 8

        if method == 1:
            x0 = x1
            k0 = k1
            extra_term = (vix_fut - np.exp(x0)) * norm.cdf(-B)
        elif method == 2:
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
            iv1 = (coeff[:, None] * he).sum(axis=0) / (np.exp(x0) * np.sqrt(self.T))
        else:
            iv1 = 0.0
            
        iv2 = extra_term / (np.exp(x0) * np.sqrt(self.T) * norm.pdf(B))

        iv = iv0 + iv1
        iv_e = iv0 + iv1 + iv2

        return iv, iv_e
    
    def implied_vol_expan_cal(self, k, F_mkt, method):
        """
        Compute the implied volatility using the Hermite expansion.
        
        Parameters
        ----------
        k: float or ndarray
            Log-moneyness of the VIX option.
        opttype: int, optional
            Option type: 1 for call, -1 for put, and 0 for futures.
        method: int, optional
            Rewritten form: 1 for defining the log-spot price, 2 for defing the log-strike price.
        """

        optimal_n = self.optimal_order

        weights_0 = np.array([self.weight_calculation(n, self.g) for n in range(optimal_n + 1)])
        weights_1 = np.array([self.weight_calculation(n, self.g1) for n in range(optimal_n + 1)])
        weights_2 = np.array([self.weight_calculation(n, self.g2) for n in range(optimal_n + 1)])
        weights_3 = np.array([self.weight_calculation(n, self.g3) for n in range(optimal_n + 1)])

        c_0 = self.c_0()
        c_1 = self.c_1()
        c_2 = self.c_2()
        c_3 = self.c_3()

        vix_fut = F_mkt

        kappa = vix_fut * np.exp(k)

        A = self.d1_finding(kappa)

        if (self.r == -1 and self.sigma1 < self.sigma2) or \
            (self.r == 1 and self.sigma1 > self.sigma2):
            B = A - self.sigma1 / 2
            iv0 = 0.5 * self.sigma1 / np.sqrt(self.T)

            k1 = np.log(kappa)
            x1 = k1 - B * self.sigma1 / 2 - self.sigma1**2 / 8
            x2 = np.log(vix_fut)
            k2 = x2 + B * self.sigma1 / 2 + self.sigma1**2 / 8
        else:
            B = A - self.sigma2 / 2
            iv0 = 0.5 * self.sigma2 / np.sqrt(self.T)

            k1 = np.log(kappa)
            x1 = k1 - B * self.sigma2 / 2 - self.sigma2**2 / 8
            x2 = np.log(vix_fut)
            k2 = x2 + B * self.sigma2 / 2 + self.sigma2**2 / 8

        if method == 1:
            x0 = x1
            k0 = k1
            extra_term = (vix_fut - np.exp(x0)) * norm.cdf(-B)
        elif method == 2:
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
            iv1 = (coeff[:, None] * he).sum(axis=0) / (np.exp(x0) * np.sqrt(self.T))
        else:
            iv1 = 0.0
            
        iv2 = extra_term / (np.exp(x0) * np.sqrt(self.T) * norm.pdf(B))

        iv = iv0 + iv1
        iv_e = iv0 + iv1 + iv2

        return iv, iv_e
  