import numpy as np
from math import factorial
from scipy.stats import norm
from scipy.integrate import quad
from scipy.optimize import brentq, newton

import utils
from black_scholes import BlackScholes, BlackScholesLog
from hermite_expansion import HermiteExpansion

# Clipping bounds to prevent overflow/underflow errors in np.exp()
EXP_LOWER_BOUND = -700
EXP_UPPER_BOUND = 700

# Small number to prevent division by zero
EPSILON = 1e-9

class StandardBergomi:
    """
    Implementation of the standard Bergomi model.

    Parameters
    ----------
    omega: float
        Vol-of-vol for each process.
    k: float
        Mean-reversion speed.
    xi0: callable
        Initial forward instantaneous variance.
    delta_vix: float, optional
        Time window for the VIX calculation (default is 1/12).
    """

    def __init__(self, omega, k, xi0, delta_vix=1.0/12.0):
        """
        Initialize the standard Bergomi model.
        See class docstring for parameter definitions.
        """
        if omega <= 0.0:
            raise ValueError("Volatility of volatility omega must be positive.")
        if not callable(xi0):
            raise ValueError("xi0 must be a callable function.")
        t_test = np.linspace(1e-10, 10, 1000)
        if not np.all(xi0(t_test) > np.array([0.0])):
            raise ValueError("xi0 must be positive for all t >= 0.")
        
        self.omega = omega
        self.k = k
        self.xi0 = xi0
        self.xi0_0 = self.xi0(np.zeros(1))[0]
        self.delta_vix = delta_vix

        self.x0 = lambda t: np.log(xi0(t))
        self.x0_0 = self.x0(np.zeros(1))[0]
        self.x0_flat = self._is_xi0_flat()

    def _is_xi0_flat(self):
        """
        Check if the forward variance curve xi0 is flat.
        """
        t_test = np.linspace(1e-10, 10, 1000)
        return np.allclose(self.xi0(t_test), self.xi0_0)
    
    def kernel(self, u, t):
        """
        Compute the rough Bergomi kernel function.

        Parameters
        ----------
        u : float or np.ndarray
            Upper time(s) (must satisfy u > t).
        t : float or np.ndarray
            Lower time(s).
        """
        return self.omega * np.exp(-self.k * (u - t))
    
    def fut_vix2(self, T, n_quad=30, quad_scipy=True):
        r"""
        Compute the fair value of a VIX squared futures contract at maturity T, which
        corresponds to

            E[VIX_T^2] = 1/delta \int_{T}^{T+delta} \xi_0^u du = nu (\xi_0^\cdot).

        Parameters
        ----------
        T : float
            Maturity of the VIX future (must be non-negative).
        n_quad : int, optional
            Number of quadrature points for numerical integration, if not using scipy
            (default is 30).
        quad_scipy : bool, optional
            If True, use scipy's quad for integration (default is True).
        """
        if T < 0:
            raise ValueError("Maturity T must be non-negative.")
        if n_quad < 1:
            raise ValueError("n_quad must be at least 1.")

        if quad_scipy:
            integral, _ = quad(lambda u: self.xi0(u), T, T + self.delta_vix)

            return integral / self.delta_vix
        
        else:
            nodes, weights = utils.gauss_legendre(0.0, 1.0, n_quad)
            return np.dot(weights, self.xi0(T + self.delta_vix * nodes))
    
    def nu0_K(self, t, T, n_quad=30, quad_scipy=True):
        r"""
        Compute the normalized integral of the kernel over the VIX proxy time interval.

        \nu (K^\cdot (t)) = 
            (1/delta) * \int_{T}^{T+delta} xi0(u) * kernel(u, t) du / F_{VIX^2}

        This is used in the calculation of gamma coefficients for the VIX proxy
        expansion.

        Parameters
        ----------
        t : float
            Lower time of the kernel.
        T : float
            Start of the VIX window.
        n_quad : int, optional
            Number of quadrature points for numerical integration, if not using scipy
            (default is 30).
        quad_scipy : bool, optional
            If True, use scipy's quad for integration (default is True).
        """
        if T < 0:
            raise ValueError("Maturity T must be non-negative.")
        if n_quad < 1:
            raise ValueError("n_quad must be at least 1.")
        
        fvix2 = self.fut_vix2(T, n_quad=n_quad, quad_scipy=quad_scipy)

        if quad_scipy:
        
            val, _ = quad(
                lambda u: self.xi0(u) * self.kernel(u, t),
                T,
                T + self.delta_vix
            )

            return (val / (fvix2 * self.delta_vix))
        else:
            nodes, weights = utils.gauss_legendre(0.0, 1.0, n_quad)
            u_grid = T + self.delta_vix * nodes
            K_mat = self.kernel(u=u_grid[:, None], t=t)
            return weights * self.xi0(u_grid) @ K_mat / fvix2
    
    def nu0_K2(self, t, T, n_quad=30, quad_scipy=True):
        r"""
        Compute the normalized integral of the kernel over the VIX proxy time interval.

        nu (K^\cdot (t)^2) = 
            (1/delta) * \int_{T}^{T+delta} xi0(u) * kernel(u, t)^2 du / F_{VIX^2}

        This is used in the calculation of gamma coefficients for the VIX proxy
        expansion.

        Parameters
        ----------
        t : float
            Lower time of the kernel.
        T : float
            Start of the VIX window.
        n_quad : int, optional
            Number of quadrature points for numerical integration, if not using scipy
            (default is 30).
        quad_scipy : bool, optional
            If True, use scipy's quad for integration (default is True).
        """
        if T < 0:
            raise ValueError("Maturity T must be non-negative.")
        if n_quad < 1:
            raise ValueError("n_quad must be at least 1.")
        
        fvix2 = self.fut_vix2(T, n_quad=n_quad, quad_scipy=quad_scipy)
        
        if quad_scipy:

            val, _ = quad(
                lambda u: self.xi0(u) * (self.kernel(u, t)**2),
                T,
                T + self.delta_vix
            )

            return val / (fvix2 * self.delta_vix)
        else:
            nodes, weights = utils.gauss_legendre(0.0, 1.0, n_quad)
            u_grid = T + self.delta_vix * nodes
            K_sq = self.kernel(u=u_grid[:, None], t=t)**2
            return (weights * self.xi0(u_grid)) @ K_sq / fvix2
    
    def mean(self, T, n_quad=30, quad_scipy=True):
        """
        Compute the mean of the squared VIX proxy.

        Parameters
        ----------
        T: float
            Maturity.
        n_quad : int, optional
            Number of quadrature points for numerical integration, if not using scipy
            (default is 30).
        quad_scipy : bool, optional
            If True, use scipy's quad for integration (default is True).
        """
        if T <= 0:
            raise ValueError("Maturity T must be positive.")
        
        fvix2 = self.fut_vix2(T, n_quad=n_quad, quad_scipy=quad_scipy)
        
        if quad_scipy:

            def integrand(t):
                return self.nu0_K2(t, T)

            return np.log(fvix2) - 0.5 * quad(integrand, 0, T)[0]

        else:
            nodes, weights = utils.gauss_legendre(0.0, 1.0, n_quad)
            t_grid = T * nodes
            u_grid = T + self.delta_vix * nodes
            vec_u = weights * self.xi0(u_grid)
            K_sq = self.kernel(u=u_grid[None, :], t=t_grid[:, None])**2
            return np.log(fvix2) - (0.5 * T / fvix2) * weights @ (K_sq @ vec_u)

    def mean_flat(self, T):
        """
        Compute the mean of the squared VIX proxy when the forward variance curve xi0 is flat.

        Parameters
        ----------
        T: float
            Maturity.
        """
        if T <= 0:
            raise ValueError("Maturity T must be positive.")
        
        term = (self.omega**2 / (8 * self.k**2 * self.delta_vix)
                * (1 - np.exp(-2 * self.k * T))
                * (1 - np.exp(-2 * self.k * self.delta_vix)))
        return self.x0_0 - term
    
    def variance(self, T, n_quad=30, quad_scipy=True):
        """
        Compute the variance of the VIX proxy.

        Parameters
        ----------
        T : float
            Maturity.
        n_quad : int, optional
            Number of quadrature points for numerical integration, if not using scipy
            (default is 30).
        quad_scipy : bool, optional
            If True, use scipy's quad for integration (default is True).
        """
        if T <= 0:
            raise ValueError("Maturity T must be positive.")
    
        if quad_scipy:

            def integrand(t):
                return self.nu0_K(t, T)**2

            return quad(integrand, 0, T)[0]
        
        else:
            fvix2 = self.fut_vix2(T, n_quad=n_quad, quad_scipy=quad_scipy)
            nodes, weights = utils.gauss_legendre(0.0, 1.0, n_quad)
            t_grid = T * nodes
            u_grid = T + self.delta_vix * nodes
            vec_u = weights * self.xi0(u_grid)
            K_mat = self.kernel(u=u_grid[None, :], t=t_grid[:, None])
            nu0_k_vals = (K_mat @ vec_u) / fvix2
            return T * np.dot(weights, nu0_k_vals**2)
    
    def variance_flat(self, T):
        """
        Compute the variance of the squared VIX proxy when the forward variance curve xi0 is flat.

        Parameters
        ----------
        T: float
            Maturity.
        """
        if T <= 0:
            raise ValueError("Maturity T must be positive.")
        
        var = (self.omega**2 / (2 * self.k**3 * self.delta_vix**2)
                 * (1 - np.exp(-2 * self.k * T))
                 * (1 - np.exp(-self.k * self.delta_vix))**2)
        return var
    
    def gamma_1(self, T, n_quad=30, quad_scipy=False):
        """
        Compute the first-order gamma coefficient of the VIX option price proxy.

        Parameters
        ----------
        T : float
            Maturity of the VIX future.
        n_quad : int, optional
            Number of quadrature points for numerical integration, if not using scipy
            (default is 30).
        quad_scipy : bool, optional
            If True, use scipy's quad for integration (default is False).
        """
        if T <= 0:
            raise ValueError("Maturity T must be positive.")
        if n_quad < 1:
            raise ValueError("n_quad must be at least 1.")
        
        fvix2 = self.fut_vix2(T, n_quad=n_quad, quad_scipy=quad_scipy)
        
        if quad_scipy:

            def I1(u):
                val, _ = quad(
                    lambda t: self.kernel(u, t)**2 - self.nu0_K2(t, T),
                    0.0, T
                )
                return val

            def I2(u):
                val, _ = quad(
                    lambda t: (self.kernel(u, t) - self.nu0_K(t, T))**2,
                    0.0, T
                )
                return val
            
            def outer_integrand(u):
                term1 = 0.125 * (I1(u)**2)
                term2 = 0.5 * I2(u)
                return self.xi0(u) * (term1 + term2) / (fvix2 * self.delta_vix)

            gamma_1, _ = quad(outer_integrand, T, T + self.delta_vix)
            return gamma_1

        else:
            nodes, weights = utils.gauss_legendre(0.0, 1.0, n_quad)
            t_grid = T * nodes
            u_grid = T + self.delta_vix * nodes
            xi_weights = weights * self.xi0(u_grid)
            K_mat = self.kernel(u=u_grid[:, None], t=t_grid[None, :])
            K_sq = K_mat**2
            
            nu0_K = (xi_weights @ K_mat) / fvix2
            nu0_K2 = (xi_weights @ K_sq) / fvix2
            
            int_K_sq_dt = K_sq @ weights
            
            term_I1 = T * (int_K_sq_dt - np.dot(weights, nu0_K2))
            
            term_B = K_mat @ (weights * nu0_K)
            term_C = np.dot(weights, nu0_K**2)
            term_I2 = T * (int_K_sq_dt - 2 * term_B + term_C)
            
            result = xi_weights @ (0.125 * term_I1**2 + 0.5 * term_I2)
            return result / fvix2
    
    def gamma_1_flat(self, T):
        """
        Compute the first-order gamma coefficient of the VIX option price proxy
        when the forward variance curve xi0 is flat.

        Parameters
        ----------
        T: float
            Maturity.
        """
        if T <= 0:
            raise ValueError("Maturity T must be positive.")

        term1 = 1 - np.exp(-2 * self.k * T)
        term2 = 1 - np.exp(-2 * self.k * self.delta_vix)
        term3 = 1 - np.exp(-self.k * self.delta_vix)

        gamma_11 = (self.omega**4 / (128 * self.k**4 * self.delta_vix**2)
                    * (-1 + self.k * self.delta_vix
                       * (1 + np.exp(-2 * self.k * self.delta_vix)) / term2)
                    * term1**2
                    * term2**2)
        
        gamma_12 = (self.omega**2 / (8 * self.k**3 * self.delta_vix**2)
                    * ((2 + self.k * self.delta_vix) * np.exp(-self.k * self.delta_vix)
                       - 2 + self.k * self.delta_vix)
                    * term1
                    * term3)

        return gamma_11 + gamma_12
    
    def gamma_2(self, T, n_quad=30, quad_scipy=False):
        """
        Compute the second-order gamma coefficient of the VIX option price proxy.

        Parameters
        ----------
        T : float
            Maturity of the VIX future.
        n_quad : int, optional
            Number of quadrature points for numerical integration, if not using scipy. (Default is 30)
        quad_scipy : bool, optional
            If True, use scipy's quad for integration (default is False).
        """
        if T <= 0:
            raise ValueError("Maturity T must be positive.")
        if n_quad < 1:
            raise ValueError("n_quad must be at least 1.")
        
        fvix2 = self.fut_vix2(T, n_quad=n_quad, quad_scipy=quad_scipy)
        
        if quad_scipy:
             
            def I1(u):
                val, _ = quad(
                    lambda t: self.kernel(u, t)**2 - self.nu0_K2(t, T),
                    0.0, T
                )
                return val
             
            def I2(u):
                val, _ = quad(
                    lambda t: self.nu0_K(t, T) * (self.kernel(u, t) - self.nu0_K(t, T)),
                    0.0, T
                )
                return val
             
            def outer_integrand(u):
                return  - 0.5 * self.xi0(u) * I1(u) * I2(u) / (fvix2 * self.delta_vix)
             
            gamma_2, _ = quad(outer_integrand, T, T + self.delta_vix)
            return gamma_2
        else:
            nodes, weights = utils.gauss_legendre(0.0, 1.0, n_quad)
            t_grid = T * nodes
            u_grid = T + self.delta_vix * nodes
            K_mat = self.kernel(u=u_grid[:, None], t=t_grid[None, :])
            K_sq = K_mat**2
            xi_weights = weights * self.xi0(u_grid)
            
            nu0_K = (xi_weights @ K_mat) / fvix2
            nu0_K2 = (xi_weights @ K_sq) / fvix2
            weighted_nu = weights * nu0_K
            
            term_I1 = T * (K_mat @ weighted_nu - np.dot(weights, nu0_K**2))
            
            term_I2 = T * (K_sq @ weights - np.dot(weights, nu0_K2))
            
            gamma_2 = -0.5 * (xi_weights @ (term_I1 * term_I2)) / fvix2
            return gamma_2
    
    def gamma_2_flat(self, T):
        """
        Compute the second-order gamma coefficient of the VIX option price proxy
        when the forward variance curve xi0 is flat.

        Parameters
        ----------
        T: float
            Maturity.
        """
        if T <= 0:
            raise ValueError("Maturity T must be positive.")

        term1 = 1 - np.exp(-2 * self.k * T)
        term2 = 1 - np.exp(-self.k * self.delta_vix)

        gamma_2 = (- self.omega**4 / (48 * self.k**5 * self.delta_vix**3)
                   * (2 * self.k * self.delta_vix * (1 + np.exp(-self.k * self.delta_vix))
                      + np.exp(-2 * self.k * self.delta_vix) * (2 * self.k * self.delta_vix + 3) - 3)
                   * term1**2
                   * term2**2)
        
        return gamma_2
    
    def gamma_3(self, T, n_quad=30, quad_scipy=False):
        """
        Compute the second-order gamma coefficient of the VIX option price proxy.

        Parameters
        ----------
        T : float
            Maturity of the VIX future.
        n_quad : int, optional
            Number of quadrature points for numerical integration, if not using scipy. (Default is 30)
        quad_scipy : bool, optional
            If True, use scipy's quad for integration (default is False)
        """
        if T <= 0:
            raise ValueError("Maturity T must be positive.")
        if n_quad < 1:
            raise ValueError("n_quad must be at least 1.")
        
        fvix2 = self.fut_vix2(T, n_quad=n_quad, quad_scipy=quad_scipy)
        
        if quad_scipy:

            def I(u):
                val, _ = quad(
                    lambda t: self.nu0_K(t, T) * (self.kernel(u, t) - self.nu0_K(t, T)),
                    0.0, T
                )
                return val
            
            def outer_integrand(u):
                return 0.5 * self.xi0(u) * I(u)**2 / (fvix2 * self.delta_vix)
             
            gamma_3, _ = quad(outer_integrand, T, T + self.delta_vix)
            return gamma_3
        else:
            nodes, weights = utils.gauss_legendre(0.0, 1.0, n_quad)
            t_grid = T * nodes
            u_grid = T + self.delta_vix * nodes 
            xi_weights = weights * self.xi0(u_grid)
            K_mat = self.kernel(u=u_grid[:, None], t=t_grid[None, :])
            nu0_K = (xi_weights @ K_mat) / fvix2
            I_u = T * (K_mat @ (weights * nu0_K) - np.dot(weights, nu0_K**2))
            gamma_3 = 0.5 * (xi_weights @ I_u**2) / fvix2
            return gamma_3
    
    def gamma_3_flat(self, T):
        """
        Compute the third-order gamma coefficient of the VIX option price proxy
        when the forward variance curve xi0 is flat.

        Parameters
        ----------
        T: float
            Maturity.
        """
        if T <= 0:
            raise ValueError("Maturity T must be positive.")

        term1 = 1 - np.exp(-2 * self.k * T)
        term2 = 1 - np.exp(-self.k * self.delta_vix)

        gamma_3 = (self.omega**4 / (16 * self.k**6 * self.delta_vix**4)
                   * (self.k * self.delta_vix - 2
                      + np.exp(-self.k * self.delta_vix) * (2 + self.k * self.delta_vix))
                   * term1**2
                   * term2**3)
        
        return gamma_3
    
    def vix_opt_price_proxy(self, kappa, T, opttype):
        """
        Approximate the price of the VIX option using the proxy expansion.

        Parameters
        ----------
        kappa: float
            Strike price.
        T: float
            Maturity.
        opttype: int, optional
            Option type: 1 for call, -1 for put.
        """
        if opttype not in [-1, 1]:
            raise ValueError("Option type opttype must be either -1 (put) or 1 (call).")
        if T <= 0:
            raise ValueError("Maturity T must be positive.")

        if not self.x0_flat:
            mu = self.mean(T)
            sigma = np.sqrt(self.variance(T))
            gamma_1 = self.gamma_1(T)
            gamma_2 = self.gamma_2(T)
            gamma_3 = self.gamma_3(T)
        else:
            mu = self.mean_flat(T)
            sigma = np.sqrt(self.variance_flat(T))
            gamma_1 = self.gamma_1_flat(T)
            gamma_2 = self.gamma_2_flat(T)
            gamma_3 = self.gamma_3_flat(T)

        values = {
            "s": np.exp(mu / 2 + sigma**2 / 8),
            "k": kappa,
            "sigma": sigma / (2 * np.sqrt(T)),
            "T": T,
            "opttype": opttype
        }

        bs = BlackScholes(**values)

        price_0 = bs.price()
        price_1 = 1 / 2 * np.exp(mu / 2 + sigma**2 / 8) * bs.delta()
        price_2 = price_1 / 2 + np.exp(mu + sigma**2 / 4) / 4 * bs.gamma()
        price_3 = -price_1 / 2 + 3 * price_2 / 2 + 1 / 8 * np.exp(3 * mu / 2 + 3 * sigma**2 / 8) * bs.speed()

        return price_0 + gamma_1 * price_1 + gamma_2 * price_2 + gamma_3 * price_3

    def vix_fut_price_proxy(self, T):
        """
        Approximate the price of the VIX option using the proxy expansion.

        Parameters
        ----------
        T: float
            Maturity.
        """
        if T <= 0:
            raise ValueError("Maturity T must be positive.")

        if not self.x0_flat:
            mu = self.mean(T)
            sigma = np.sqrt(self.variance(T))
            gamma_1 = self.gamma_1(T)
            gamma_2 = self.gamma_2(T)
            gamma_3 = self.gamma_3(T)
        else:
            mu = self.mean_flat(T)
            sigma = np.sqrt(self.variance_flat(T))
            gamma_1 = self.gamma_1_flat(T)
            gamma_2 = self.gamma_2_flat(T)
            gamma_3 = self.gamma_3_flat(T)

        x = np.exp(mu / 2 + sigma**2 / 8)

        return x * (1 + gamma_1 / 2 + gamma_2 / 4 + gamma_3 / 8)
  
    def vix_opt_price(self, kappa, T, opttype, n_time, n_space):
        """
        Approximate the price of the VIX option using the two-dimensional quadrature.

        Parameters
        ----------
        kappa: float
            Strike price.
        T: float
            Maturity.
        opttype: int, optional
            Option type: 1 for call, -1 for put, and 0 for futures.
        n_time: int
            Number of quadrature nodes for the time interval.
        n_space: int
            Number of quadrature nodes for the space interval.
        """
        if T <= 0:
            raise ValueError("Maturity T must be positive.")
        
        if self.k > 0:
            var = (1 - np.exp(-2 * self.k * T)) / (2 * self.k)
        else:
            var = T

        def vix2(x):
            """
            Rewrite the squared VIX index using the Markovian representation.
            """   

            def f(u, x):
                term1 = self.omega * np.exp(-self.k * (u - T)) * x
                term2 = 0.5 * self.omega**2 * np.exp(-2 * self.k * (u - T)) * var

                return np.exp (term1 - term2)
            
            nodes, weights = utils.gauss_legendre(T, T+self.delta_vix, n_time)

            integral = 0
            for (u, w) in zip(nodes, weights):

                if not self.x0_flat:
                    integral += w * f(u, x) * self.xi0(u)
                else:
                    integral += w * f(u, x) * np.exp(self.x0_0)

            return integral / self.delta_vix
        
        nodes, weights = utils.gauss_hermite(var, n_space)

        prices = 0
        for (x, w) in zip(nodes, weights):

            if opttype == 1: 
                prices += w * np.maximum(0, np.sqrt(vix2(x)) - kappa)
            elif opttype == -1:
                prices += w * np.maximum(0, -np.sqrt(vix2(x)) + kappa)
            else:
                prices += w * np.sqrt(vix2(x))
        
        return prices
    
    def vix_opt_price_expan(self, kappa, T, opttype, return_opt="price"):
        """
        """
        if T <= 0:
            raise ValueError("Maturity T must be positive.")
        
        if not self.x0_flat:
            mu = self.mean(T)
            sigma = np.sqrt(self.variance(T))
            gamma_1 = self.gamma_1(T)
            gamma_2 = self.gamma_2(T)
            gamma_3 = self.gamma_3(T)
        else:
            mu = self.mean_flat(T)
            sigma = np.sqrt(self.variance_flat(T))
            gamma_1 = self.gamma_1_flat(T)
            gamma_2 = self.gamma_2_flat(T)
            gamma_3 = self.gamma_3_flat(T)

        if opttype == 0:
            x = np.exp(mu / 2 + sigma**2 / 8)
            price = x * (1 + gamma_1 / 2 + gamma_2 / 4 + gamma_3 / 8)
        else:
            values = {
                "x": mu / 2 + sigma**2 / 8,
                "y": np.log(kappa),
                "sigma": sigma / (2 * np.sqrt(T)),
                "T": T,
                "opttype": opttype
            }

            bs = BlackScholesLog(**values)
            price_0 = bs.price()
            price_1 = bs.price_derivative(1)
            price_2 = bs.price_derivative(2)
            price_3 = bs.price_derivative(3)
            price = price_0 + gamma_1 / 2 * price_1 + gamma_2 / 4 * price_2 + gamma_3 / 8 * price_3
            discarded_terms = (gamma_1 / 2 + gamma_2 / 4 + gamma_3 / 8) * bs.price_derivative(1)
        
        if return_opt == "all":
            return price, discarded_terms
        else:
            return price

    def implied_vol_proxy(self, k, T, return_opt="iv"):
        """
        Compute the implied volatility implied from the approximation of VIX option prices
        using the root-find method.

        Parameters
        ----------
        k: float or ndarray
            Log-moneyness of the VIX option.
        T: float
            Maturity.
        return_opt: str, optional
            If 'iv', return only the implied volatility.
            If 'all', return both the futures price and the implied volatility.
        """
        if T <= 0:
            raise ValueError("Maturity T must be positive.")
        
        k = np.atleast_1d(np.asarray(k))
        F = self.vix_fut_price_proxy(T)
        kappa = F * np.exp(k)
        
        opttype = 2 * (kappa >= F) - 1
        otm_price = np.array(
            [
                self.vix_opt_price_proxy(k, T, opttype_i) for k, opttype_i in zip(kappa, opttype)
            ]
        )

        ivs = utils.implied_vol_bisection(F, kappa, T, otm_price, opttype)

        if return_opt == "all":
            return F, ivs
        else:
            return ivs
    
    def vix_implied_vol(self, k, T, n_time, n_space, return_opt="iv"):
        """
        Compute the implied volatility implied from the approximation of VIX option prices
        using the root-find method.

        Parameters
        ----------
        k: float or ndarray
            Log-moneyness of the VIX option.
        T: float
            Maturity.
        n_time: int
            Number of quadrature nodes for the time interval.
        n_space: int
            Number of quadrature nodes for the space interval.
        return_opt: str, optional
            If 'iv', return only the implied volatility.
            If 'all', return both the futures price and the implied volatility.
        """
        if T <= 0:
            raise ValueError("Maturity T must be positive.")
        
        k = np.atleast_1d(np.asarray(k))
        F = self.vix_opt_price(0, T, opttype=0, n_time=n_time, n_space=n_space)
        kappa = F * np.exp(k)
        
        opttype = 2 * (kappa >= F) - 1
        otm_price = np.array(
            [
                self.vix_opt_price(k, T, opttype_i, n_time, n_space) for k, opttype_i in zip(kappa, opttype)
            ]
        )

        ivs = utils.implied_vol_bisection(F, kappa, T, otm_price, opttype)

        if return_opt == "all":
            return F, ivs
        else:
            return ivs
    
    def implied_vol_expan(self, k, T, return_opt="iv"):
        """
        Approximate the implied volatility of the VIX option using the Taylor expansion.

        Parameters
        ----------
        k: float or ndarray
            Log-moneyness of the VIX option.
        T: float
            Maturity.
        return_opt: str, optional
            If 'iv', return only the implied volatility.
            If 'all', return both the futures price and the implied volatility.
        """
        if T <= 0:
            raise ValueError("Maturity T must be positive.")
        
        k = np.atleast_1d(np.asarray(k))
        F = self.vix_fut_price_proxy(T)
        kappa = F * np.exp(k)

        if not self.x0_flat:
            mu = self.mean(T)
            sigma = np.sqrt(self.variance(T))
            gamma_2 = self.gamma_2(T)
            gamma_3 = self.gamma_3(T)
        else:
            mu = self.mean_flat(T)
            sigma = np.sqrt(self.variance_flat(T))
            gamma_2 = self.gamma_2_flat(T)
            gamma_3 = self.gamma_3_flat(T)

        tilde_sigma = sigma / np.sqrt(T)
        m = mu / 2 + sigma**2 / 8 - np.log(kappa)

        iv_0 = tilde_sigma / 2

        iv_1 = (gamma_2 / (2 * tilde_sigma * T)
                + 3 * gamma_3 / (8 * tilde_sigma * T)
                - gamma_3 * m / (tilde_sigma**3 * T**2))
        
        ivs = iv_0 + iv_1

        if return_opt == "all":
            return F, ivs
        else:
            return ivs
    
    def vix_opt_price_proxy_mixed(self, kappa, T, omega2, lbd, opttype, n_gauss):
        """
        Approximate the price of the VIX option using the proxy expansion
        in the mixed standard Bergomi model.

        Parameters
        ----------
        kappa: float
            Strike price.
        T: float
            Maturity.
        omega2: float
            Vol-of-vol of the second factor.
        lbd: float
            Weight of the second factor.
        opttype: int, optional
            Option type: 1 for call, -1 for put.
        n_gauss: int
            Number of quadrature nodes for the space interval.
        """
        if T <= 0:
            raise ValueError("Maturity T must be positive.")
        if omega2 <= 0.0:
            raise ValueError("Vol-of-vol omega2 must be positive.")

        kappa = np.atleast_1d(np.asarray(kappa))

        bergomi_omega2 = self.__class__(
            omega=omega2,
            k=self.k,
            xi0=self.xi0,
            delta_vix=self.delta_vix
        )

        fvix2 = self.fut_vix2(T)

        if not self.x0_flat:
            mu1 = self.mean(T) - np.log(fvix2)
            mu2 = bergomi_omega2.mean(T) - np.log(fvix2)
            sigma1 = np.sqrt(self.variance(T))
            sigma2 = np.sqrt(bergomi_omega2.variance(T))
            gamma_11 = self.gamma_1(T)
            gamma_21 = self.gamma_2(T)
            gamma_31 = self.gamma_3(T)
            gamma_12 = bergomi_omega2.gamma_1(T)
            gamma_22 = bergomi_omega2.gamma_2(T)
            gamma_32 = bergomi_omega2.gamma_3(T)
        else:
            mu1 = self.mean_flat(T) - self.x0_0
            mu2 = bergomi_omega2.mean_flat(T) - self.x0_0
            sigma1 = np.sqrt(self.variance_flat(T))
            sigma2 = np.sqrt(bergomi_omega2.variance_flat(T))
            gamma_11 = self.gamma_1_flat(T)
            gamma_21 = self.gamma_2_flat(T)
            gamma_31 = self.gamma_3_flat(T)
            gamma_12 = bergomi_omega2.gamma_1_flat(T)
            gamma_22 = bergomi_omega2.gamma_2_flat(T)
            gamma_32 = bergomi_omega2.gamma_3_flat(T)

        z, weights_trans = utils.gauss_hermite(1, n_gauss)

        arg_1 = mu1 + sigma1 * z
        arg_2 = mu2 + sigma2 * z
        arg_1 = np.clip(arg_1, EXP_LOWER_BOUND, EXP_UPPER_BOUND) 
        arg_2 = np.clip(arg_2, EXP_LOWER_BOUND, EXP_UPPER_BOUND)

        if not self.x0_flat:
            F = fvix2 * (lbd * np.exp(arg_1) 
                                + (1 - lbd) * np.exp(arg_2))
        else:
            F = np.exp(self.x0_0) * (lbd * np.exp(arg_1) 
                                + (1 - lbd) * np.exp(arg_2))
        
        if opttype in [-1, 1]:
            payoff = np.maximum(opttype * (np.sqrt(F) - kappa[:, np.newaxis]), 0)
        elif opttype == 0:
            payoff = np.tile(np.sqrt(F), (len(kappa), 1))

        price_0 = np.sum(weights_trans * payoff, axis=1)
        
        def payoff_derivative(x):
            sqrt_x = np.sqrt(np.maximum(x, EPSILON))
            if opttype == 1:
                condition = x > kappa[:, np.newaxis] ** 2
                return np.where(condition, 1 / (2 * sqrt_x), 0)
            elif opttype == 0:
                deriv = 1 / (2 * np.sqrt(x))
                return np.tile(deriv, (len(kappa), 1))
            else:
                condition = x < kappa[:, np.newaxis] ** 2
                return np.where(condition, -1 / (2 * sqrt_x), 0)
        
        def psi1(x):
            term1 = lbd * np.exp(np.clip(x, EXP_LOWER_BOUND, EXP_UPPER_BOUND))

            if self.omega < EPSILON:
                term2 = (1 - lbd) * np.exp(np.clip(mu2, EXP_LOWER_BOUND, EXP_UPPER_BOUND))
            else:
                mu0 = mu1 / (self.omega**2)

                exp_arg = omega2 * (self.omega - omega2) * (-mu0) + omega2 / self.omega * x
                term2 = (1 - lbd) * np.exp(np.clip(exp_arg, EXP_LOWER_BOUND, EXP_UPPER_BOUND))
            
            if not self.x0_flat:
                total = fvix2 * (term1 + term2)
                psi1 = payoff_derivative(total) *  fvix2 * term1
            else:
                total = np.exp(self.x0_0) * (term1 + term2)
                psi1 = payoff_derivative(total) * np.exp(self.x0_0) * term1
            return psi1
        
        def psi2(x):
            term1 = (1 - lbd) * np.exp(np.clip(x, EXP_LOWER_BOUND, EXP_UPPER_BOUND))

            if omega2 < EPSILON:
                term2 = lbd * np.exp(mu1)
            else:
                mu0 = mu2 / (omega2**2)

                exp_arg = self.omega * (omega2 - self.omega) * (-mu0) + self.omega / omega2 * x

                term2 = lbd * np.exp(np.clip(exp_arg, EXP_LOWER_BOUND, EXP_UPPER_BOUND))
            
            if not self.x0_flat:
                total = fvix2 * (term1 + term2)
                psi2 = payoff_derivative(total) * fvix2 * term1
            else:
                total = np.exp(self.x0_0) * (term1 + term2)
                psi2 = payoff_derivative(total) * np.exp(self.x0_0) * term1
            return psi2

        arg1_vec = mu1 + sigma1 * z
        arg2_vec = mu2 + sigma2 * z

        val_psi1 = psi1(arg1_vec)
        val_psi2 = psi2(arg2_vec)

        safe_sigma1 = sigma1 if sigma1 > 1e-6 else np.inf 
        safe_sigma2 = sigma2 if sigma2 > 1e-6 else np.inf
       
        price_1 = (gamma_11 * np.sum(weights_trans * val_psi1, axis=1)
                   + gamma_12 * np.sum(weights_trans * val_psi2, axis=1))
        
        price_2 = (gamma_21 * np.sum(weights_trans * z * val_psi1, axis=1) / safe_sigma1
                   + gamma_22 * np.sum(weights_trans * z * val_psi2, axis=1) / safe_sigma2)
        
        price_3 = (gamma_31 * np.sum(weights_trans * (z**2 - 1) * val_psi1, axis=1) / safe_sigma1**2
                   + gamma_32 * np.sum(weights_trans * (z**2 - 1) * val_psi2, axis=1) / safe_sigma2**2)
        
        price  = price_0 + price_1 + price_2 + price_3
        
        return price

    def vix_opt_price_mixed(self, kappa, T, omega2, lbd, opttype, n_time, n_space):
        """
        Approximate the price of the VIX option in the mixed standard Bergomi model
        using the two-dimensional quadrature.

        Parameters
        ----------
        kappa: float, 
            Strike price.
        T: float
            Maturity.
        omega2: float
            Vol-of-vol of the second factor.
        lbd: float
            Weight of the second factor.
        opttype: int, optional
            Option type: 1 for call, -1 for put, and 0 for futures.
        n_time: int
            Number of quadrature nodes for the time interval.
        n_space: int
            Number of quadrature nodes for the space interval.
        """
        if T <= 0:
            raise ValueError("Maturity T must be positive.")
        if omega2 <= 0.0:
            raise ValueError("Vol-of-vol omega2 must be positive.")
        
        if self.k > 0:
            var = (1 - np.exp(-2 * self.k * T)) / (2 * self.k)
        else:
            var = T

        def vix2(x, omega):
            """
            Rewrite the squared VIX index using the Markovian representation.
            """   

            def f(u, x):
                term1 = omega * np.exp(-self.k * (u - T)) * x
                term2 = 0.5 * omega**2 * np.exp(-2 * self.k * (u - T)) * var

                return np.exp (term1 - term2)
            
            nodes, weights = utils.gauss_legendre(T, T+self.delta_vix, n_time)

            integral = 0
            for (u, w) in zip(nodes, weights):
                if not self.x0_flat:
                    integral += w * f(u, x) * self.xi0(u)
                else:
                    integral += w * f(u, x) * np.exp(self.x0_0)

            return integral / self.delta_vix
        
        nodes, weights = utils.gauss_hermite(var, n_space)

        prices = 0
        for (x, w) in zip(nodes, weights):

            vix2_mixed = lbd * vix2(x, self.omega) + (1 - lbd) * vix2(x, omega2)

            if opttype == 1: 
                prices += w * np.maximum(0, np.sqrt(vix2_mixed) - kappa)
            elif opttype == -1:
                prices += w * np.maximum(0, -np.sqrt(vix2_mixed) + kappa)
            else:
                prices += w * np.sqrt(vix2_mixed)
        
        return prices
    
    def implied_vol_proxy_mixed(self, k, T, omega2, lbd, n_gauss, return_opt="iv"):
        """
        Compute the implied volatility implied from the approximation of VIX option prices
        in the mixe standard Bergomi model using the root-find method.

        Parameters
        ----------
        k: float or ndarray
            Log-moneyness of the VIX option.
        T: float
            Maturity.
        omega2: float
            Vol-of-vol of the second factor.
        lbd: float
            Weight of the second factor.
        n_gauss: int
                Number of nodes for the one-dimensional Gauss-Hermite quadrature.
        return_opt: str, optional
            If 'iv', return only the implied volatility.
            If 'all', return both the futures price and the implied volatility.
        """
        if T <= 0:
            raise ValueError("Maturity T must be positive.")
        if omega2 <= 0.0:
            raise ValueError("Vol-of-vol omega2 must be positive.")

        k = np.atleast_1d(np.asarray(k))
        F = self.vix_opt_price_proxy_mixed(0, T, omega2=omega2, lbd=lbd, opttype=0, n_gauss=n_gauss)
        kappa = F * np.exp(k)
        
        opttype = 2 * (kappa >= F) - 1
        otm_price = np.array(
            [
                self.vix_opt_price_proxy_mixed(k, T, omega2, lbd, opttype_i, n_gauss)[0]
                for k, opttype_i in zip(kappa, opttype)
            ]
        )

        ivs = utils.implied_vol_bisection(F, kappa, T, otm_price, opttype)

        if return_opt == "all":
            return F, ivs
        else:
            return ivs 
  
    def vix_implied_vol_mixed(self, k, T, omega2, lbd, n_time, n_space, return_opt="iv"):
        """
        Compute the implied volatility implied from the approximation of VIX option prices
        using the root-find method.

        Parameters
        ----------
        k: float or ndarray
            Log-moneyness of the VIX option.
        T: float
            Maturity.
        omega2: float
            Vol-of-vol of the second factor.
        lbd: float
            Weight of the second factor.
        n_time: int
            Number of quadrature nodes for the time interval.
        n_space: int
            Number of quadrature nodes for the space interval.
        return_opt: str, optional
            If 'iv', return only the implied volatility.
            If 'all', return both the futures price and the implied volatility.
        """
        if T <= 0:
            raise ValueError("Maturity T must be positive.")
        if omega2 <= 0.0:
            raise ValueError("Vol-of-vol omega2 must be positive.")

        k = np.atleast_1d(np.asarray(k))
        F = self.vix_opt_price_mixed(0, T, omega2, lbd, opttype=0, n_time=n_time, n_space=n_space)
        kappa = F * np.exp(k)
        
        opttype = 2 * (kappa >= F) - 1
        otm_price = np.array(
            [
                self.vix_opt_price_mixed(k, T, omega2, lbd, opttype=opttype_i, n_time=n_time, n_space=n_space)
                for k, opttype_i in zip(kappa, opttype)
            ]
        )

        ivs = utils.implied_vol_bisection(F, kappa, T, otm_price, opttype)

        if return_opt == "all":
            return F, ivs
        else:
            return ivs

    def vix_opt_price_expan_mixed(self, kappa, T, rule, omega2, lbd, opttype, n_max, n_mc, optimal_n=None, return_opt="price"):
        """
        Compute the VIX option price using the Hermite expansion.

        Parameters
        ----------
        kappa: float
            Strike price.
        T: float
            Maturity.
        rule: int, optional
            The rule of the implied volatility expansion.
            rule=1, choose the higher volatility as the base,
            rule=-1, choose the lower volatility as the base
        omega2: float
            Vol-of-vol of the second factor.
        lbd: float
            Weight of the second factor.
        opttype: int, optional
            Option type: 1 for call, -1 for put, and 0 for futures.
        n_max: int
            The maximum order of the Hermite expansion.
        n_mc : int
            Number of Monte Carlo paths.
        optimal_n : int
            The optimal order for the Hermite expansion. (default is None)
        """
        if T <= 0:
            raise ValueError("Maturity T must be positive.")
        if omega2 <= 0.0:
            raise ValueError("Vol-of-vol omega2 must be positive.")
        
        kappa = np.atleast_1d(np.asarray(kappa))

        bergomi_omega2 = self.__class__(
            omega=omega2,
            k=self.k,
            xi0=self.xi0,
            delta_vix=self.delta_vix
        )

        if not self.x0_flat:
            mu1 = self.mean(T)
            mu2 = bergomi_omega2.mean(T)
            sigma1 = np.sqrt(self.variance(T))
            sigma2 = np.sqrt(bergomi_omega2.variance(T))
            gamma_11 = self.gamma_1(T)
            gamma_21 = self.gamma_2(T)
            gamma_31 = self.gamma_3(T)
            gamma_12 = bergomi_omega2.gamma_1(T)
            gamma_22 = bergomi_omega2.gamma_2(T)
            gamma_32 = bergomi_omega2.gamma_3(T)
        else:
            mu1 = self.mean_flat(T)
            mu2 = bergomi_omega2.mean_flat(T)
            sigma1 = np.sqrt(self.variance_flat(T))
            sigma2 = np.sqrt(bergomi_omega2.variance_flat(T))
            gamma_11 = self.gamma_1_flat(T)
            gamma_21 = self.gamma_2_flat(T)
            gamma_31 = self.gamma_3_flat(T)
            gamma_12 = bergomi_omega2.gamma_1_flat(T)
            gamma_22 = bergomi_omega2.gamma_2_flat(T)
            gamma_32 = bergomi_omega2.gamma_3_flat(T)

        expan = HermiteExpansion(rule, mu1, mu2, sigma1, sigma2, lbd, gamma_11, gamma_21, gamma_31, gamma_12, gamma_22, gamma_32)
        price = expan.vix_opt_price(kappa, opttype, n_max, n_mc, optimal_n)
        
        formulas = {
            "Formula 1": 1,
            "Formula 2": 2,
            "Formula 3": 3
        }
        
        discarded_term = {formula: expan.discarded_term_opt_price(kappa, T, n_max, n_mc, f) for formula, f in formulas.items()}
        
        if return_opt=="all":
            return price, discarded_term
        else:
            return price
    
    def implied_vol_expan_mixed(self, k, T, rule, omega2, lbd, n_max, n_mc, formula, optimal_n=None, return_opt="iv"):
        """
        Compute the implied volatility using the Hermite expansion.
        
        Parameters
        ----------
        k: float or ndarray
            Log-moneyness of the VIX option.
        T: float
            Maturity.
        rule: int, optional
            The rule of the implied volatility expansion.
            rule=1, choose the higher volatility as the base,
            rule=-1, choose the lower volatility as the base
        omega2: float
            Vol-of-vol of the second factor.
        lbd: float
            Weight of the second factor.
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
        if T <= 0:
            raise ValueError("Maturity T must be positive.")
        if omega2 <= 0.0:
            raise ValueError("Vol-of-vol omega2 must be positive.")
        
        bergomi_omega2 = self.__class__(
            omega=omega2,
            k=self.k,
            xi0=self.xi0,
            delta_vix=self.delta_vix
        )
        
        if not self.x0_flat:
            mu1 = self.mean(T)
            mu2 = bergomi_omega2.mean(T)
            sigma1 = np.sqrt(self.variance(T))
            sigma2 = np.sqrt(bergomi_omega2.variance(T))
            gamma_11 = self.gamma_1(T)
            gamma_21 = self.gamma_2(T)
            gamma_31 = self.gamma_3(T)
            gamma_12 = bergomi_omega2.gamma_1(T)
            gamma_22 = bergomi_omega2.gamma_2(T)
            gamma_32 = bergomi_omega2.gamma_3(T)
        else:
            mu1 = self.mean_flat(T)
            mu2 = bergomi_omega2.mean_flat(T)
            sigma1 = np.sqrt(self.variance_flat(T))
            sigma2 = np.sqrt(bergomi_omega2.variance_flat(T))
            gamma_11 = self.gamma_1_flat(T)
            gamma_21 = self.gamma_2_flat(T)
            gamma_31 = self.gamma_3_flat(T)
            gamma_12 = bergomi_omega2.gamma_1_flat(T)
            gamma_22 = bergomi_omega2.gamma_2_flat(T)
            gamma_32 = bergomi_omega2.gamma_3_flat(T)

        expan = HermiteExpansion(rule, mu1, mu2, sigma1, sigma2, lbd, gamma_11, gamma_21, gamma_31, gamma_12, gamma_22, gamma_32)
        vix_fut, iv, _ = expan.vix_implied_vol(k, T, n_max, n_mc, formula, optimal_n)

        if return_opt == "all":
            return vix_fut, iv
        else:
            return iv
    
    def vix_opt_price_expan_mixed_cal(self, kappa, T, rule, omega2, lbd, opttype, n_max, n_mc, optimal_n=None):
        """
        Compute the VIX option price using the Hermite expansion.

        Parameters
        ----------
        kappa: float
            Strike price.
        T: float
            Maturity.
        rule: int, optional
            The rule of the implied volatility expansion.
            rule=1, choose the higher volatility as the base,
            rule=-1, choose the lower volatility as the base
        omega2: float
            Vol-of-vol of the second factor.
        lbd: float
            Weight of the second factor.
        opttype: int, optional
            Option type: 1 for call, -1 for put, and 0 for futures.
        n_max: int
            The maximum order of the Hermite expansion.
        n_mc : int
            Number of Monte Carlo paths.
        optimal_n : int
            The optimal order for the Hermite expansion. (default is None)
        """
        if T <= 0:
            raise ValueError("Maturity T must be positive.")
        if omega2 < 0:
            raise ValueError("Vol-of-vol omega2 must be non-negative.")
        
        kappa = np.atleast_1d(np.asarray(kappa))

        bergomi_omega2 = self.__class__(
            omega=omega2,
            k=self.k,
            xi0=self.xi0,
            delta_vix=self.delta_vix
        )

        if not self.x0_flat:
            mu1 = self.mean(T, n_quad=30, quad_scipy=False)
            mu2 = bergomi_omega2.mean(T, n_quad=30, quad_scipy=False)
            sigma1 = np.sqrt(self.variance(T, n_quad=30, quad_scipy=False))
            sigma2 = np.sqrt(bergomi_omega2.variance(T, n_quad=30, quad_scipy=False))
            gamma_11 = self.gamma_1(T, n_quad=30, quad_scipy=False)
            gamma_21 = self.gamma_2(T, n_quad=30, quad_scipy=False)
            gamma_31 = self.gamma_3(T, n_quad=30, quad_scipy=False)
            gamma_12 = bergomi_omega2.gamma_1(T, n_quad=30, quad_scipy=False)
            gamma_22 = bergomi_omega2.gamma_2(T, n_quad=30, quad_scipy=False)
            gamma_32 = bergomi_omega2.gamma_3(T, n_quad=30, quad_scipy=False)
        else:
            mu1 = self.mean_flat(T)
            mu2 = bergomi_omega2.mean_flat(T)
            sigma1 = np.sqrt(self.variance_flat(T))
            sigma2 = np.sqrt(bergomi_omega2.variance_flat(T))
            gamma_11 = self.gamma_1_flat(T)
            gamma_21 = self.gamma_2_flat(T)
            gamma_31 = self.gamma_3_flat(T)
            gamma_12 = bergomi_omega2.gamma_1_flat(T)
            gamma_22 = bergomi_omega2.gamma_2_flat(T)
            gamma_32 = bergomi_omega2.gamma_3_flat(T)

        expan = HermiteExpansion(rule, mu1, mu2, sigma1, sigma2, lbd, gamma_11, gamma_21, gamma_31, gamma_12, gamma_22, gamma_32)
        price = expan.vix_opt_price_cal(kappa, opttype, n_max, n_mc, optimal_n)
        
        return price
    
    def implied_vol_expan_mixed_cal(self, k, T, rule, omega2, lbd, n_max, n_mc, formula, optimal_n=None, return_opt="iv"):
        """
        Compute the implied volatility using the Hermite expansion.
        
        Parameters
        ----------
        k: float or ndarray
            Log-moneyness of the VIX option.
        T: float
            Maturity.
        rule: int, optional
            The rule of the implied volatility expansion.
            rule=1, choose the higher volatility as the base,
            rule=-1, choose the lower volatility as the base
        omega2: float
            Vol-of-vol of the second factor.
        lbd: float
            Weight of the second factor.
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
        if T <= 0:
            raise ValueError("Maturity T must be positive.")
        if omega2 <= 0.0:
            raise ValueError("Vol-of-vol omega2 must be positive.")
        
        bergomi_omega2 = self.__class__(
            omega=omega2,
            k=self.k,
            xi0=self.xi0,
            delta_vix=self.delta_vix
        )
        
        if not self.x0_flat:
            mu1 = self.mean(T, n_quad=30, quad_scipy=False)
            mu2 = bergomi_omega2.mean(T, n_quad=30, quad_scipy=False)
            sigma1 = np.sqrt(self.variance(T, n_quad=30, quad_scipy=False))
            sigma2 = np.sqrt(bergomi_omega2.variance(T, n_quad=30, quad_scipy=False))
            gamma_11 = self.gamma_1(T, n_quad=30, quad_scipy=False)
            gamma_21 = self.gamma_2(T, n_quad=30, quad_scipy=False)
            gamma_31 = self.gamma_3(T, n_quad=30, quad_scipy=False)
            gamma_12 = bergomi_omega2.gamma_1(T, n_quad=30, quad_scipy=False)
            gamma_22 = bergomi_omega2.gamma_2(T, n_quad=30, quad_scipy=False)
            gamma_32 = bergomi_omega2.gamma_3(T, n_quad=30, quad_scipy=False)
        else:
            mu1 = self.mean_flat(T)
            mu2 = bergomi_omega2.mean_flat(T)
            sigma1 = np.sqrt(self.variance_flat(T))
            sigma2 = np.sqrt(bergomi_omega2.variance_flat(T))
            gamma_11 = self.gamma_1_flat(T)
            gamma_21 = self.gamma_2_flat(T)
            gamma_31 = self.gamma_3_flat(T)
            gamma_12 = bergomi_omega2.gamma_1_flat(T)
            gamma_22 = bergomi_omega2.gamma_2_flat(T)
            gamma_32 = bergomi_omega2.gamma_3_flat(T)

        expan = HermiteExpansion(rule, mu1, mu2, sigma1, sigma2, lbd, gamma_11, gamma_21, gamma_31, gamma_12, gamma_22, gamma_32)
        vix_fut, iv, iv_e = expan.vix_implied_vol_cal(k, T, n_max, n_mc, formula, optimal_n)

        if return_opt == "all":
            return vix_fut, iv, iv_e
        else:
            return iv, iv_e