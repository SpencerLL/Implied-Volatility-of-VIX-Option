import numpy as np
import warnings
from math import factorial
from scipy.stats import norm
from scipy.optimize import brentq, newton
from scipy.linalg import eigvals, cholesky
from scipy.special import beta, hyp2f1
from scipy.integrate import dblquad, quad

import utils
from black_scholes import BlackScholes, BlackScholesLog
from hermite_expansion import HermiteExpansion

# Clipping bounds to prevent overflow/underflow errors in np.exp()
EXP_LOWER_BOUND = -700
EXP_UPPER_BOUND = 700

# Small number to prevent division by zero
EPSILON = 1e-9

class RoughBergomi:
    """
    Implementation of the rough Bergomi model.

    Parameters
    ----------
    eta: float
        Vol-of-vol for each process.
    H: float
        Hurst parameter of the fractional Brownian motion (must be positive).
    xi0: callable
        Initial forward instantaneous variance.
    delta_vix: float, optional
        Time window for the VIX calculation (default is 1/12).
    """

    def __init__(self, eta, H, xi0, delta_vix=1.0/12.0):
        """
        Initialize the rough Bergomi model.
        See class docstring for parameter definitions.
        """
        if eta <= 0.0:
            raise ValueError("Volatility of volatility eta must be positive.")
        if H <= 0.0:
            raise ValueError("Hurst parameter H must be positive.")
        if not callable(xi0):
            raise ValueError("xi0 must be a callable function.")
        t_test = np.linspace(1e-10, 10, 1000)
        if not np.all(xi0(t_test) > np.array([0.0])):
            raise ValueError("xi0 must be positive for all t >= 0.")
        
        self.eta = eta
        self.H = H
        self.xi0 = xi0
        self.xi0_0 = self.xi0(np.zeros(1))[0]
        self.delta_vix = delta_vix

        self.x0 = lambda t: np.log(self.xi0(t))
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
        return self.eta * (u - t)**(self.H - 0.5)
    
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
        
        term0 = 2 * self.H + 1

        term11 = self.eta**2 * ((T + self.delta_vix) ** term0 - self.delta_vix ** term0 - T**term0)
        term12 = 4 * self.delta_vix * self.H * term0

        return self.x0_0 - term11 / term12
    
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

        hyp1 = hyp2f1(-self.H-0.5, self.H+1.5, self.H+2.5, -T/self.delta_vix)

        term0 = 2 * self.H + 2

        term1 = self.eta**2 / (self.delta_vix**2 * (self.H + 0.5)**2)
        term2 = (((T + self.delta_vix)**term0 - self.delta_vix**term0 + T**term0) / term0
                 - 2 * beta(1, self.H+1.5) * self.delta_vix**(self.H+0.5) * T**(self.H+1.5) * hyp1)
        return term1 * term2
    
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
            If True, use scipy's quad for integration (default is False)
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
        
        sigma2 = self.variance_flat(T=T)

        hyp1 = hyp2f1(-2 * self.H, 2 * self.H + 1, 2 * self.H + 2, -self.delta_vix/T)
        term0 = 4 * self.H + 1
        term1 = 2 * self.H + 1

        gamma_11 = (self.eta**4 / (32 * self.H**2)
                    * (((T + self.delta_vix)**term0 + self.delta_vix**term0 - T**term0) / (self.delta_vix * term0)
                       - (((T + self.delta_vix)**term1 - self.delta_vix**term1 - T**term1) / (self.delta_vix * term1))**2
                       - 2 * beta(1, term1) * self.delta_vix**(2 * self.H) * T**(2 * self.H) * hyp1))
        
        gamma_12 = (self.eta**2 * ((T + self.delta_vix)**term1 - self.delta_vix**term1 - T**term1)
                    / (4 * self.delta_vix * self.H * term1))

        gamma_1 = gamma_11 + gamma_12 - sigma2 / 2

        return gamma_1
    
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
        
        sigma2 = self.variance_flat(T)
        term0 = 2 * self.H + 1

        def integrand_func(t, u):

            intergand = (((T * t + self.delta_vix)**(self.H + 0.5) - (T * t)**(self.H + 0.5))
                         * ((T + self.delta_vix * u)**(2 * self.H) - (self.delta_vix * u)**(2 * self.H))
                         * (T * t + self.delta_vix * u)**(self.H - 0.5))
        
            return intergand

        gamma_21 = (-(self.eta**4 * T) / (2 * self.delta_vix * self.H * term0)
                    * dblquad(integrand_func, 0, 1, lambda t: 0, lambda t: 1)[0])
        
        gamma_22 = ((self.eta**2 * sigma2) / (4 * self.delta_vix * self.H * term0)
                    * ((T + self.delta_vix)**term0 - self.delta_vix**term0 - T**term0))

        gamma_2 = gamma_21 + gamma_22
        
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
        
        sigma2 = self.variance_flat(T)

        def integrand_func(t, u):

            def omega_func(x):

                delta = self.delta_vix / T
                term0 = self.H + 0.5

                hyp1 = hyp2f1(-term0, term0, term0+1, -(1+delta*x)/(delta*(1-x)))
                hyp2 = hyp2f1(-term0, term0, term0+1, -x/(1 - x))
                hyp3 = hyp2f1(-term0+1, term0+1, term0+2, -1/(delta*x))

                omega_1 =  ((1 - x)**term0 * delta**term0 * beta(1, term0)
                            * ((1 + delta * x)**term0 * hyp1 - (delta * x)**term0 * hyp2))

                omega_2 = beta(1, term0+1) * (delta * x)**(term0 - 1) * hyp3

                omega = omega_1 - omega_2
                return omega

            omega = omega_func(u)
            delta = self.delta_vix / T

            integrand = (((t + delta)**(self.H + 0.5) - t**(self.H + 0.5))
                         * (t + delta * u)**(self.H - 0.5)
                         * omega)

            return integrand

        gamma_3 = ((self.eta**4 * T**(4 * self.H + 2)) / (2 * self.delta_vix**2 * (self.H + 0.5)**2)
                   * dblquad(integrand_func, 0, 1, lambda t: 0, lambda t: 1)[0]
                   - sigma2**2 / 2)
        
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

    def cov_matrix_vix(self, T, n_disc):
        """
        Compute the covariance matrix of the Gaussian vector (X_T^{u_i}) for 1 <= i <= n,
        where u_i = T + delta * i / n, and delta = 1 / 12.

        Parameters
        ----------
        T: float
            Maturity.
        n_disc : int
            Number of time discretization steps.
        """
        if T <= 0:
            raise ValueError("Maturity T must be positive.")
        
        tab_t = np.linspace(T, T + self.delta_vix, n_disc + 1)

        cov = np.zeros((n_disc + 1, n_disc + 1))

        for i in range(n_disc + 1):
            for j in range(i, n_disc + 1):
                if j==i:
                    cov[i,j] = ((self.eta**2 / (2 * self.H))
                                * (tab_t[i]**(2 * self.H) - (tab_t[i] - T)**(2 * self.H)))
                else:
                    term0 = tab_t[j] - tab_t[i]
                    term1 = self.H + 0.5
                    
                    cov_1 = self.eta**2 * term0**(term1 - 1) / term1
                    cov_21 = tab_t[i]**term1 * hyp2f1(-term1+1, term1, term1+1, -tab_t[i]/term0)
                    cov_22 = (tab_t[i] - T)**term1 * hyp2f1(-term1+1, term1, term1+1, -(tab_t[i]-T)/term0)
                    
                    cov[i,j] = cov_1 * (cov_21 - cov_22)
                    cov[j,i] = cov[i,j]

        return cov
    
    def cholesky_cov_matrix_vix(self, T, n_disc):
        """
        Compute the lower-triangular Cholesky factor of the covariance matrix of the
        Gaussian vector (X_T^{u_i}) for 1 <= i <= n, where u_i = T + delta * i / n
        and delta = 1 / 12.

        Parameters
        ----------
        T: float
            Maturity.
        n_mc : int
            Number of Monte Carlo paths.
        n_disc : int
            Number of time discretization steps.
        """
        if T <= 0:
            raise ValueError("Maturity T must be positive.")

        cov = self.cov_matrix_vix(T=T, n_disc=n_disc)[1:, 1:]

        # eigenvalues = eigvals(cov)

        try:
            L = cholesky(cov, lower=True)
        except Exception:
            jitter = 1e-12
            max_jitter = 1e-6
            # success = False
            
            while jitter <= max_jitter:
                try:
                    cov_fix = cov + np.eye(cov.shape[0]) * jitter
                    L = cholesky(cov_fix, lower=True)
                    # success = True
                    break
                except Exception:
                    jitter *= 10

        return L
    
    def vix_opt_price(self, kappa, T, opttype, n_disc, n_mc, seed=None):
        """
        Approximate the price of the VIX option using the Monte-Carlo method.

        Parameters
        ----------
        kappa: float
            Strike price.
        T: float
            Maturity.
        opttype: int, optional
            Option type: 1 for call, -1 for put, and 0 for futures.
        n_disc : int
            Number of time discretization steps.
        n_mc : int
            Number of Monte Carlo paths.
        seed : int or None, optional
            Random seed for reproducibility. Default is None.
        """
        if T <= 0:
            raise ValueError("Maturity T must be positive.")
        if seed is not None:
            np.random.seed(seed)

        tab_t = np.linspace(T, T + self.delta_vix, n_disc + 1)
        cov = self.cov_matrix_vix(T=T, n_disc=n_disc)[1:, 1:]
        L = self.cholesky_cov_matrix_vix(T=T, n_disc=n_disc)

        if not self.x0_flat:
            mu = self.x0(tab_t) - self.eta**2 / (4 * self.H) * (tab_t**(2 * self.H) - (tab_t - T)**(2 * self.H))
        else: 
            mu = self.x0_0 - self.eta**2 / (4 * self.H) * (tab_t**(2 * self.H) - (tab_t - T)**(2 * self.H))
        mu = mu[1:]
        mu_mean = np.mean(mu)

        sigma2_mean = np.sum(cov) / n_disc**2
        sigma_mean = np.sqrt(sigma2_mean)

        samples = np.zeros(n_mc)
        z = np.random.randn(n_mc, n_disc)

        for i in range(n_mc):
            # right-point scheme
            X_T = mu + L @ z[i, :]
            VIX2_R = np.mean(np.exp(X_T))
            exp_meanX = np.exp(np.mean(X_T))

            if opttype == 1:
                # opttype == 1: call option
                payoff1 = max(0, np.sqrt(VIX2_R) - kappa)
                payoff2 = max(0, np.sqrt(exp_meanX) - kappa)
                samples[i] = payoff1 - payoff2
            elif opttype == -1:
                # opttype == -1: put option
                payoff1 = max(0, - np.sqrt(VIX2_R) + kappa)
                payoff2 = max(0, - np.sqrt(exp_meanX) + kappa)
                samples[i] = payoff1 - payoff2
            else:
                payoff1 = np.sqrt(VIX2_R)
                payoff2 = np.sqrt(exp_meanX)
                samples[i] = payoff1 - payoff2

        s = np.exp(mu_mean / 2 + sigma2_mean / 8)

        # control varible
        if opttype in [1, -1]:
            # opttype == 1: call option 
            cv = BlackScholes(s=s, k=kappa, sigma = sigma_mean / (2 * np.sqrt(T)), T=T, opttype=opttype).price()
        else:
            cv = s

        estimate = np.mean(samples) + cv

        return estimate
    
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
    
    def vix_implied_vol(self, k, T, n_disc, n_mc, seed=None, return_opt="iv"):
        """
        Compute the implied volatility implied from the approximation of VIX option prices
        using the root-find method.

        Parameters
        ----------
        k: float or ndarray
            Log-moneyness of the VIX option.
        T: float
            Maturity.
        n_disc : int
            Number of time discretization steps.
        n_mc : int
            Number of Monte Carlo paths.
        seed : int or None, optional
            Random seed for reproducibility. Default is None.
        return_opt: str, optional
            If 'iv', return only the implied volatility.
            If 'all', return both the futures price and the implied volatility.
        """
        if T <= 0:
            raise ValueError("Maturity T must be positive.")

        if seed is not None:
            np.random.seed(seed)

        k = np.atleast_1d(np.asarray(k))
        F = self.vix_opt_price(0, T, opttype=0, n_disc=n_disc, n_mc=n_mc, seed=seed)
        kappa = F * np.exp(k)
        
        opttype = 2 * (kappa >= F) - 1
        otm_price = np.array(
            [
                self.vix_opt_price(k, T, opttype_i, n_disc, n_mc, seed) for k, opttype_i in zip(kappa, opttype)
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
    
    def vix_opt_price_proxy_mixed(self, kappa, T, eta2, lbd, opttype, n_gauss):
        """
        Approximate the price of the VIX option using the proxy expansion
        in the mixed rough Bergomi model.

        Parameters
        ----------
        kappa: float
            Strike price.
        T: float
            Maturity.
        eta2: float
            Vol-of-vol of the second factor.
        lbd: float
            Weight of the second factor.
        opttype: int, optional
            Option type: 1 for call, -1 for put.
        n_gauss: int
            Number of quadrature points for the Gauss-Hermite quadrature.
        """
        if T <= 0:
            raise ValueError("Maturity T must be positive.")
        if eta2 < 0:
            raise ValueError("Vol-of-vol eta2 must be non-negative.")

        kappa = np.atleast_1d(np.asarray(kappa))

        rbergomi_eta2 = self.__class__(
            eta=eta2,
            H=self.H,
            xi0=self.xi0,
            delta_vix=self.delta_vix
        )
        
        fvix2 = self.fut_vix2(T)

        if not self.x0_flat:
            mu1 = self.mean(T) - np.log(fvix2)
            mu2 = rbergomi_eta2.mean(T) - np.log(fvix2)
            sigma1 = np.sqrt(self.variance(T))
            sigma2 = np.sqrt(rbergomi_eta2.variance(T))
            gamma_11 = self.gamma_1(T)
            gamma_21 = self.gamma_2(T)
            gamma_31 = self.gamma_3(T)
            gamma_12 = rbergomi_eta2.gamma_1(T)
            gamma_22 = rbergomi_eta2.gamma_2(T)
            gamma_32 = rbergomi_eta2.gamma_3(T)
        else:
            mu1 = self.mean_flat(T) - self.x0_0
            mu2 = rbergomi_eta2.mean_flat(T) - self.x0_0
            sigma1 = np.sqrt(self.variance_flat(T))
            sigma2 = np.sqrt(rbergomi_eta2.variance_flat(T))
            gamma_11 = self.gamma_1_flat(T)
            gamma_21 = self.gamma_2_flat(T)
            gamma_31 = self.gamma_3_flat(T)
            gamma_12 = rbergomi_eta2.gamma_1_flat(T)
            gamma_22 = rbergomi_eta2.gamma_2_flat(T)
            gamma_32 = rbergomi_eta2.gamma_3_flat(T)

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

            if self.eta < EPSILON:
                term2 = (1 - lbd) * np.exp(np.clip(mu2, EXP_LOWER_BOUND, EXP_UPPER_BOUND))
            else:
                mu0 = mu1 / (self.eta**2)

                exp_arg = eta2 * (self.eta - eta2) * (-mu0) + eta2 / self.eta * x
                term2 = (1 - lbd) * np.exp(np.clip(exp_arg, EXP_LOWER_BOUND, EXP_UPPER_BOUND))
            
            if not self.x0_flat:
                total = fvix2 * (term1 + term2)
                psi1 = payoff_derivative(total) * fvix2 * term1
            else:
                total = np.exp(self.x0_0) * (term1 + term2)
                psi1 = payoff_derivative(total) * np.exp(self.x0_0) * term1
            return psi1

        def psi2(x):
            term1 = (1 - lbd) * np.exp(np.clip(x, EXP_LOWER_BOUND, EXP_UPPER_BOUND))

            if eta2 < EPSILON:
                term2 = lbd * np.exp(mu1)
            else:
                mu0 = mu2 / (eta2**2)

                exp_arg = self.eta * (eta2 - self.eta) * (-mu0) + self.eta / eta2 * x

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

    def vix_index_mixed(self, T, eta2, lbd, n_disc, n_mc, seed = None):
        """
        Simulated the path for the VIX index in the mixe Bergomi model using the Monte-Carlo method.

        Parameters
        ----------
        T: float
            Maturity.
        eta2: float
            Vol-of-vol of the second factor.
        lbd: float
            Weight of the second factor.
        n_disc : int
            Number of time discretization steps.
        n_mc : int
            Number of Monte Carlo paths.
        seed : int or None, optional
            Random seed for reproducibility. Default is None.
        """
        if T <= 0:
            raise ValueError("Maturity T must be positive.")
        if eta2 < 0:
            raise ValueError("Vol-of-vol eta2 must be non-negative.")

        if seed is not None:
            np.random.seed(seed)

        tab_t = np.linspace(T, T + self.delta_vix, n_disc + 1)

        rbergomi_eta2 = self.__class__(
            eta=eta2,
            H=self.H,
            xi0=self.xi0,
            delta_vix=self.delta_vix
        )

        L1 = self.cholesky_cov_matrix_vix(T=T, n_disc=n_disc)
        L2 = rbergomi_eta2.cholesky_cov_matrix_vix(T=T, n_disc=n_disc)

        if not self.x0_flat:
            mu1 = self.x0(tab_t) - self.eta**2 / (4 * self.H) * (tab_t**(2 * self.H) - (tab_t - T)**(2 * self.H))
            mu2 = self.x0(tab_t) - eta2**2 / (4 * self.H) * (tab_t**(2 * self.H) - (tab_t - T)**(2 * self.H))
        else:
            mu1 = self.x0_0 - self.eta**2 / (4 * self.H) * (tab_t**(2 * self.H) - (tab_t - T)**(2 * self.H))
            mu2 = self.x0_0 - eta2**2 / (4 * self.H) * (tab_t**(2 * self.H) - (tab_t - T)**(2 * self.H))

        mu1 = mu1[1:]
        mu2 = mu2[1:]

        z = np.random.randn(n_mc, n_disc)
        samples = np.zeros(n_mc)

        z_T = z.T
        x1_T = mu1[:, None] + L1 @ z_T
        x2_T = mu2[:, None] + L2 @ z_T

        combined_exp = lbd * np.exp(x1_T) + (1 - lbd) * np.exp(x2_T)
        samples = np.mean(combined_exp, axis=0)

        return np.sqrt(samples)
    
    def vix_opt_price_mixed(self, kappa, T, eta2, lbd, opttype, n_disc, n_mc, seed = None):
        """
        Approximate the price of the VIX option using the Monte-Carlo method.

        Parameters
        ----------
        kappa: float
            Strike price.
        T: float
            Maturity.
        eta2: float
            Vol-of-vol of the second factor.
        lbd: float
            Weight of the second factor.
        opttype: int, optional
            Option type: 1 for call, -1 for put, and 0 for futures.
        n_disc : int
            Number of time discretization steps.
        n_mc : int
            Number of Monte Carlo paths.
        seed : int or None, optional
            Random seed for reproducibility. Default is None.
        """
        if T <= 0:
            raise ValueError("Maturity T must be positive.")
        if eta2 < 0:
            raise ValueError("Vol-of-vol eta2 must be non-negative.")
        
        if seed is not None:
            np.random.seed(seed)
        
        kappa = np.atleast_1d(np.asarray(kappa))
        vix = self.vix_index_mixed(T, eta2, lbd, n_disc, n_mc, seed)

        if opttype == 1:
            samples = np.maximum(0, vix - kappa[:, np.newaxis])
        elif opttype == -1:
            samples = np.maximum(0, - vix + kappa[:, np.newaxis])
        else:
            samples = np.tile(vix, (len(kappa), 1))

        estimate = np.mean(samples, axis=1)
        return estimate
    
    def implied_vol_proxy_mixed(self, k, T, eta2, lbd, n_gauss, return_opt="iv"):
        """
        Compute the implied volatility implied from the approximation of VIX option prices
        in the mixed rough Bergomi model using the root-find method.

        Parameters
        ----------
        k: float or ndarray
            Log-moneyness of the VIX option.
        T: float
            Maturity.
        eta2: float
            Vol-of-vol of the second factor.
        lbd: float
            Weight of the second factor.
        n_gauss: int
            Number of quadrature points for the Gauss-Hermite quadrature.
        return_opt: str, optional
            If 'iv', return only the implied volatility.
            If 'all', return both the futures price and the implied volatility.
        """
        if T <= 0:
            raise ValueError("Maturity T must be positive.")
        if eta2 < 0:
            raise ValueError("Vol-of-vol eta2 must be non-negative.")

        k = np.atleast_1d(np.asarray(k))
        F = self.vix_opt_price_proxy_mixed(0, T, eta2, lbd, opttype=0, n_gauss=n_gauss)
        kappa = F * np.exp(k)
        
        opttype = 2 * (kappa >= F) - 1
        otm_price = np.array(
            [
                self.vix_opt_price_proxy_mixed(kappa=k, T=T, eta2=eta2, lbd=lbd,
                                                opttype=opttype_i, n_gauss=n_gauss)[0]
                for k, opttype_i in zip(kappa, opttype)
            ]
        )

        ivs = utils.implied_vol_bisection(F, kappa, T, otm_price, opttype)

        if return_opt == "all":
            return F, ivs
        else:
            return ivs

    def vix_implied_vol_mixed(self, k, T, eta2, lbd, n_disc, n_mc, seed=None, return_opt="iv"):
        """
        Compute the implied volatility implied from the approximation of VIX option prices
        using the root-find method.

        Parameters
        ----------
        k: float or ndarray
            Log-moneyness of the VIX option.
        T: float
            Maturity.
        eta2: float
            Vol-of-vol of the second factor.
        lbd: float
            Weight of the second factor.
        n_disc : int
            Number of time discretization steps.
        n_mc : int
            Number of Monte Carlo paths.
        seed : int or None, optional
            Random seed for reproducibility. Default is None.
        return_opt: str, optional
            If 'iv', return only the implied volatility.
            If 'all', return both the futures price and the implied volatility.
        """
        if T <= 0:
            raise ValueError("Maturity T must be positive.")
        if seed is not None:
            np.random.seed(seed)

        k = np.atleast_1d(np.asarray(k))
        F = self.vix_opt_price_mixed(0, T, eta2, lbd, opttype=0, n_disc=n_disc, n_mc=n_mc, seed=seed)[0]
        kappa = F * np.exp(k)

        opttype = 2 * (kappa >= F) - 1
        otm_price = np.array(
            [
                self.vix_opt_price_mixed(k, T, eta2, lbd, opttype=opttype_i, n_disc=n_disc, n_mc=n_mc, seed=seed)[0]
                for k, opttype_i in zip(kappa, opttype)
            ]
        )

        ivs = utils.implied_vol_bisection(F, kappa, T, otm_price, opttype)

        if return_opt == "all":
            return F, ivs
        else:
            return ivs

    def vix_opt_price_expan_mixed(self, kappa, T, rule, eta2, lbd, opttype, n_max, n_mc, optimal_n=None, return_opt="price"):
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
        eta2: float
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
        if eta2 < 0:
            raise ValueError("Vol-of-vol eta2 must be non-negative.")

        kappa = np.atleast_1d(np.asarray(kappa))

        rbergomi_eta2 = self.__class__(
            eta=eta2,
            H=self.H,
            xi0=self.xi0,
            delta_vix=self.delta_vix
        )

        if not self.x0_flat:
            mu1 = self.mean(T)
            mu2 = rbergomi_eta2.mean(T)
            sigma1 = np.sqrt(self.variance(T))
            sigma2 = np.sqrt(rbergomi_eta2.variance(T))
            gamma_11 = self.gamma_1(T)
            gamma_21 = self.gamma_2(T)
            gamma_31 = self.gamma_3(T)
            gamma_12 = rbergomi_eta2.gamma_1(T)
            gamma_22 = rbergomi_eta2.gamma_2(T)
            gamma_32 = rbergomi_eta2.gamma_3(T)
        else:
            mu1 = self.mean_flat(T)
            mu2 = rbergomi_eta2.mean_flat(T)
            sigma1 = np.sqrt(self.variance_flat(T))
            sigma2 = np.sqrt(rbergomi_eta2.variance_flat(T))
            gamma_11 = self.gamma_1_flat(T)
            gamma_21 = self.gamma_2_flat(T)
            gamma_31 = self.gamma_3_flat(T)
            gamma_12 = rbergomi_eta2.gamma_1_flat(T)
            gamma_22 = rbergomi_eta2.gamma_2_flat(T)
            gamma_32 = rbergomi_eta2.gamma_3_flat(T)
        
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
    
    def implied_vol_expan_mixed(self, k, T, rule, eta2, lbd, n_max, n_mc, formula, optimal_n=None, return_opt="iv"):
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
        eta2: float
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
        if eta2 < 0:
            raise ValueError("Vol-of-vol eta2 must be non-negative.")

        rbergomi_eta2 = self.__class__(
            eta=eta2,
            H=self.H,
            xi0=self.xi0,
            delta_vix=self.delta_vix
        )

        if not self.x0_flat:
            mu1 = self.mean(T)
            mu2 = rbergomi_eta2.mean(T)
            sigma1 = np.sqrt(self.variance(T))
            sigma2 = np.sqrt(rbergomi_eta2.variance(T))
            gamma_11 = self.gamma_1(T)
            gamma_21 = self.gamma_2(T)
            gamma_31 = self.gamma_3(T)
            gamma_12 = rbergomi_eta2.gamma_1(T)
            gamma_22 = rbergomi_eta2.gamma_2(T)
            gamma_32 = rbergomi_eta2.gamma_3(T)
        else:
            mu1 = self.mean_flat(T)
            mu2 = rbergomi_eta2.mean_flat(T)
            sigma1 = np.sqrt(self.variance_flat(T))
            sigma2 = np.sqrt(rbergomi_eta2.variance_flat(T))
            gamma_11 = self.gamma_1_flat(T)
            gamma_21 = self.gamma_2_flat(T)
            gamma_31 = self.gamma_3_flat(T)
            gamma_12 = rbergomi_eta2.gamma_1_flat(T)
            gamma_22 = rbergomi_eta2.gamma_2_flat(T)
            gamma_32 = rbergomi_eta2.gamma_3_flat(T)
        
        expan = HermiteExpansion(rule, mu1, mu2, sigma1, sigma2, lbd, gamma_11, gamma_21, gamma_31, gamma_12, gamma_22, gamma_32)
        vix_fut, iv, _ = expan.vix_implied_vol(k, T, n_max, n_mc, formula, optimal_n)

        if return_opt == "all":
            return vix_fut, iv
        else:
            return iv
        
    def gamma_2_proxy_flat(self, T, n_quad=30):
        """
        Compute the proxy for the second-order gamma coefficient of the VIX option price proxy
        using Gauss-Legendre quadrature.

        Parameters
        ----------
        T : float
            Maturity of the VIX future (must be non-negative).
        n_quad : int, optional
            Number of quadrature points for numerical integration, if not using scipy
            (default is 30).
        """
        if T <= 0:
            raise ValueError("Maturity T must be positive.")
        
        sigma2 = self.variance_flat(T)
        term0 = 2 * self.H + 1

        nodes, weights = np.polynomial.legendre.leggauss(n_quad)
        nodes = 0.5 * (nodes + 1)
        weights = 0.5 * weights

        t = nodes[:, None]
        u = nodes[None, :]

        term1 = (T * t + self.delta_vix)**(self.H + 0.5) - (T * t)**(self.H + 0.5)
        term2 = (T + self.delta_vix * u)**(2 * self.H) - (self.delta_vix * u)**(2 * self.H)
        term3 = (T * t + self.delta_vix * u)**(self.H - 0.5)

        integrand = term1 * term2 * term3
        integral = np.sum(weights[:, None] * weights[None, :] * integrand)

        gamma_21 = (-(self.eta**4 * T) / (2 * self.delta_vix * self.H * term0) * integral)
        
        gamma_22 = ((self.eta**2 * sigma2) / (4 * self.delta_vix * self.H * term0)
                    * ((T + self.delta_vix)**term0 - self.delta_vix**term0 - T**term0))

        gamma_2 = gamma_21 + gamma_22
        
        return gamma_2

    def gamma_3_proxy_flat(self, T, n_quad=30):
        """
        Compute the proxy for the third-order gamma coefficient of the VIX option price proxy
        using Gauss-Legendre quadrature.

        Parameters
        ----------
        T : float
            Maturity of the VIX future (must be non-negative).
        n_quad : int, optional
            Number of quadrature points for numerical integration, if not using scipy
            (default is 30).
        """
        if T <= 0:
            raise ValueError("Maturity T must be positive.")
        
        sigma2 = self.variance_flat(T)

        nodes, weights = np.polynomial.legendre.leggauss(n_quad)
        nodes = 0.5 * (nodes + 1)
        weights = 0.5 * weights
        u = nodes
        delta = self.delta_vix / T

        term0 = self.H + 0.5

        z1 = -(1 + delta * u) / (delta * (1 - u))
        hyp1 = hyp2f1(-term0, term0, term0+1, z1)

        z2 = -u / (1 - u)
        hyp2 = hyp2f1(-term0, term0, term0+1, z2)

        z3 = -1 / (delta * u)
        hyp3 = hyp2f1(-term0+1, term0+1, term0+2, z3)

        omega_1 = ((1 - u)**term0 * delta**term0 * beta(1, term0)
                   * ((1 + delta * u)**term0 * hyp1 - (delta * u)**term0 * hyp2))

        omega_2 = beta(1, term0+1) * (delta * u)**(term0 - 1) * hyp3

        omega = omega_1 - omega_2

        t_grid = nodes[:, None]
        u_grid = nodes[None, :]
        omega_grid = omega[None, :] # omega 只依赖于 u

        term1 = (t_grid + delta)**(self.H + 0.5) - t_grid**(self.H + 0.5)
        term2 = (t_grid + delta * u_grid)**(self.H - 0.5)
        
        integrand = term1 * term2 * omega_grid
        integral = np.sum(weights[:, None] * weights[None, :] * integrand)

        gamma_3 = ((self.eta**4 * T**(4 * self.H + 2)) / (2 * self.delta_vix**2 * (self.H + 0.5)**2)
                   * integral
                   - sigma2**2 / 2)
        
        return gamma_3
    
    def vix_opt_price_proxy_mixed_cal(self, kappa, T, eta2, lbd, opttype, n_gauss):
        """
        Approximate the price of the VIX option using the proxy expansion
        in the mixed rough Bergomi model.

        Parameters
        ----------
        kappa: float
            Strike price.
        T: float
            Maturity.
        eta2: float
            Vol-of-vol of the second factor.
        lbd: float
            Weight of the second factor.
        opttype: int, optional
            Option type: 1 for call, -1 for put.
        n_gauss: int
            Number of quadrature points for the Gauss-Hermite quadrature.
        """
        if T <= 0:
            raise ValueError("Maturity T must be positive.")
        if eta2 < 0:
            raise ValueError("Vol-of-vol eta2 must be non-negative.")

        kappa = np.atleast_1d(np.asarray(kappa))

        rbergomi_eta2 = self.__class__(
            eta=eta2,
            H=self.H,
            xi0=self.xi0,
            delta_vix=self.delta_vix
        )

        fvix2 = self.fut_vix2(T, n_quad=30, quad_scipy=False)

        if not self.x0_flat:
            mu1 = self.mean(T, n_quad=30, quad_scipy=False) - np.log(fvix2)
            mu2 = rbergomi_eta2.mean(T, n_quad=30, quad_scipy=False) - np.log(fvix2)
            sigma1 = np.sqrt(self.variance(T, n_quad=30, quad_scipy=False))
            sigma2 = np.sqrt(rbergomi_eta2.variance(T, n_quad=30, quad_scipy=False))
            gamma_11 = self.gamma_1(T, n_quad=30, quad_scipy=False)
            gamma_21 = self.gamma_2(T, n_quad=30, quad_scipy=False)
            gamma_31 = self.gamma_3(T, n_quad=30, quad_scipy=False)
            gamma_12 = rbergomi_eta2.gamma_1(T, n_quad=30, quad_scipy=False)
            gamma_22 = rbergomi_eta2.gamma_2(T, n_quad=30, quad_scipy=False)
            gamma_32 = rbergomi_eta2.gamma_3(T, n_quad=30, quad_scipy=False)
        else:
            mu1 = self.mean_flat(T) - self.x0_0
            mu2 = rbergomi_eta2.mean_flat(T) - self.x0_0
            sigma1 = np.sqrt(self.variance_flat(T))
            sigma2 = np.sqrt(rbergomi_eta2.variance_flat(T))
            gamma_11 = self.gamma_1_flat(T)
            gamma_21 = self.gamma_2_proxy_flat(T)
            gamma_31 = self.gamma_3_proxy_flat(T)
            gamma_12 = rbergomi_eta2.gamma_1_flat(T)
            gamma_22 = rbergomi_eta2.gamma_2_proxy_flat(T)
            gamma_32 = rbergomi_eta2.gamma_3_proxy_flat(T)

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

            if self.eta < EPSILON:
                term2 = (1 - lbd) * np.exp(np.clip(mu2, EXP_LOWER_BOUND, EXP_UPPER_BOUND))
            else:
                mu0 = mu1 / (self.eta**2)

                exp_arg = eta2 * (self.eta - eta2) * (-mu0) + eta2 / self.eta * x
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

            if eta2 < EPSILON:
                term2 = lbd * np.exp(mu1)
            else:
                mu0 = mu2 / (eta2**2)

                exp_arg = self.eta * (eta2 - self.eta) * (-mu0) + self.eta / eta2 * x

                term2 = lbd * np.exp(np.clip(exp_arg, EXP_LOWER_BOUND, EXP_UPPER_BOUND))
            
            if not self.x0_flat:
                total = fvix2 * (term1 + term2)
                psi2 = payoff_derivative(total) *  fvix2 * term1
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
    
    def implied_vol_proxy_mixed_cal(self, k, T, eta2, lbd, n_gauss, return_opt="iv"):
        """
        Compute the implied volatility implied from the approximation of VIX option prices
        in the mixed rough Bergomi model using the root-find method.

        Parameters
        ----------
        k: float or ndarray
            Log-moneyness of the VIX option.
        T: float
            Maturity.
        eta2: float
            Vol-of-vol of the second factor.
        lbd: float
            Weight of the second factor.
        n_gauss: int
            Number of quadrature points for the Gauss-Hermite quadrature.
        return_opt: str, optional
            If 'iv', return only the implied volatility.
            If 'all', return both the futures price and the implied volatility.
        """
        if T <= 0:
            raise ValueError("Maturity T must be positive.")
        if eta2 < 0:
            raise ValueError("Vol-of-vol eta2 must be non-negative.")

        k = np.atleast_1d(np.asarray(k))
        F = self.vix_opt_price_proxy_mixed_cal(0, T, eta2, lbd, opttype=0, n_gauss=n_gauss)
        kappa = F * np.exp(k)
        
        opttype = 2 * (kappa >= F) - 1
        otm_price = np.array(
            [
                self.vix_opt_price_proxy_mixed_cal(kappa=k, T=T, eta2=eta2, lbd=lbd,
                                                opttype=opttype_i, n_gauss=n_gauss)[0]
                for k, opttype_i in zip(kappa, opttype)
            ]
        )

        ivs = utils.implied_vol_bisection(F, kappa, T, otm_price, opttype)

        if return_opt == "all":
            return F, ivs
        else:
            return ivs
    
    def vix_opt_price_expan_mixed_cal(self, kappa, T, rule, eta2, lbd, opttype, n_max, n_mc, optimal_n=None):
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
        eta2: float
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
        if eta2 < 0:
            raise ValueError("Vol-of-vol eta2 must be non-negative.")

        kappa = np.atleast_1d(np.asarray(kappa))

        rbergomi_eta2 = self.__class__(
            eta=eta2,
            H=self.H,
            xi0=self.xi0,
            delta_vix=self.delta_vix
        )

        if not self.x0_flat:
            mu1 = self.mean(T, n_quad=30, quad_scipy=False)
            mu2 = rbergomi_eta2.mean(T, n_quad=30, quad_scipy=False)
            sigma1 = np.sqrt(self.variance(T, n_quad=30, quad_scipy=False))
            sigma2 = np.sqrt(rbergomi_eta2.variance(T, n_quad=30, quad_scipy=False))
            gamma_11 = self.gamma_1(T, n_quad=30, quad_scipy=False)
            gamma_21 = self.gamma_2(T, n_quad=30, quad_scipy=False)
            gamma_31 = self.gamma_3(T, n_quad=30, quad_scipy=False)
            gamma_12 = rbergomi_eta2.gamma_1(T, n_quad=30, quad_scipy=False)
            gamma_22 = rbergomi_eta2.gamma_2(T, n_quad=30, quad_scipy=False)
            gamma_32 = rbergomi_eta2.gamma_3(T, n_quad=30, quad_scipy=False)
        else:
            mu1 = self.mean_flat(T)
            mu2 = rbergomi_eta2.mean_flat(T)
            sigma1 = np.sqrt(self.variance_flat(T))
            sigma2 = np.sqrt(rbergomi_eta2.variance_flat(T))
            gamma_11 = self.gamma_1_flat(T)
            gamma_21 = self.gamma_2_proxy_flat(T)
            gamma_31 = self.gamma_3_proxy_flat(T)
            gamma_12 = rbergomi_eta2.gamma_1_flat(T)
            gamma_22 = rbergomi_eta2.gamma_2_proxy_flat(T)
            gamma_32 = rbergomi_eta2.gamma_3_proxy_flat(T)

        expan = HermiteExpansion(rule, mu1, mu2, sigma1, sigma2, lbd, gamma_11, gamma_21, gamma_31, gamma_12, gamma_22, gamma_32)
        price = expan.vix_opt_price_cal(kappa, opttype, n_max, n_mc, optimal_n)

        return price
    
    def implied_vol_expan_mixed_cal(self, k, T, rule, eta2, lbd, n_max, n_mc, formula, optimal_n=None, return_opt="iv"):
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
        eta2: float
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
        if eta2 < 0:
            raise ValueError("Vol-of-vol eta2 must be non-negative.")

        rbergomi_eta2 = self.__class__(
            eta=eta2,
            H=self.H,
            xi0=self.xi0,
            delta_vix=self.delta_vix
        )

        if not self.x0_flat:
            mu1 = self.mean(T, n_quad=30, quad_scipy=False)
            mu2 = rbergomi_eta2.mean(T, n_quad=30, quad_scipy=False)
            sigma1 = np.sqrt(self.variance(T, n_quad=30, quad_scipy=False))
            sigma2 = np.sqrt(rbergomi_eta2.variance(T, n_quad=30, quad_scipy=False))
            gamma_11 = self.gamma_1(T, n_quad=30, quad_scipy=False)
            gamma_21 = self.gamma_2(T, n_quad=30, quad_scipy=False)
            gamma_31 = self.gamma_3(T, n_quad=30, quad_scipy=False)
            gamma_12 = rbergomi_eta2.gamma_1(T, n_quad=30, quad_scipy=False)
            gamma_22 = rbergomi_eta2.gamma_2(T, n_quad=30, quad_scipy=False)
            gamma_32 = rbergomi_eta2.gamma_3(T, n_quad=30, quad_scipy=False)
        else:
            mu1 = self.mean_flat(T)
            mu2 = rbergomi_eta2.mean_flat(T)
            sigma1 = np.sqrt(self.variance_flat(T))
            sigma2 = np.sqrt(rbergomi_eta2.variance_flat(T))
            gamma_11 = self.gamma_1_flat(T)
            gamma_21 = self.gamma_2_proxy_flat(T)
            gamma_31 = self.gamma_3_proxy_flat(T)
            gamma_12 = rbergomi_eta2.gamma_1_flat(T)
            gamma_22 = rbergomi_eta2.gamma_2_proxy_flat(T)
            gamma_32 = rbergomi_eta2.gamma_3_proxy_flat(T)
        
        expan = HermiteExpansion(rule, mu1, mu2, sigma1, sigma2, lbd, gamma_11, gamma_21, gamma_31, gamma_12, gamma_22, gamma_32)
        vix_fut, iv, _ = expan.vix_implied_vol_cal(k, T, n_max, n_mc, formula, optimal_n)

        if return_opt == "all":
            return vix_fut, iv
        else:
            return iv