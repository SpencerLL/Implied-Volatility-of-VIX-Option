import os
import numpy as np

import utils
from black_scholes import BlackScholesLog


class StandardBergomi:
    """
    Implementation of the standard Bergomi model with the assumption that
    the initial forward variance curve is flat.

    Parameters
    ----------
    omega: float
            Vol-of-vol for each process.
    k: float
        Mean-reversion speed.
    xi: float
        Initial forward instantaneous variance.
    """

    def __init__(self, omega, k, xi0):
        """
        Initialize the standard Bergomi model.
        See class docstring for parameter definitions.
        """
        if omega <= 0.0:
            raise ValueError("Volatility of volatility omega must be positive.")
        
        
        self.omega = omega
        self.k = k
        self.xi0 = xi0
        self.delta_vix = 1.0 / 12.0

        self.x0 = np.log(xi0)

    def mean(self, T):
        """
        Compute the mean of the squared VIX proxy.

        Parameters
        ----------
        T: float
            Maturity.
        """
        term = (self.omega**2 / (8 * self.k**2 * self.delta_vix)
                * (1 - np.exp(-2 * self.k * T))
                * (1 - np.exp(-2 * self.k * self.delta_vix)))
        return self.x0 - term
    
    def variance(self, T):
        """
        Compute the variance of the squared VIX proxy.

        Parameters
        ----------
        T: float
            Maturity.
        """
        var = (self.omega**2 / (2 * self.k**3 * self.delta_vix**2)
                 * (1 - np.exp(-2 * self.k * T))
                 * (1 - np.exp(-self.k * self.delta_vix))**2)
        return var
    
    def gamma_1(self, T):
        """
        Compute the first-order gamma coefficient of the VIX option price proxy.

        Parameters
        ----------
        T: float
            Maturity.
        """

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
    
    def gamma_2(self, T):
        """
        Compute the second-order gamma coefficient of the VIX option price proxy.

        Parameters
        ----------
        T: float
            Maturity.
        """

        term1 = 1 - np.exp(-2 * self.k * T)
        term2 = 1 - np.exp(-self.k * self.delta_vix)

        gamma_2 = (- self.omega**4 / (48 * self.k**5 * self.delta_vix**3)
                   * (2 * self.k * self.delta_vix * (1 + np.exp(-self.k * self.delta_vix))
                      + np.exp(-2 * self.k * self.delta_vix) * (2 * self.k * self.delta_vix + 3) - 3)
                   * term1**2
                   * term2**2)
        
        return gamma_2
    
    def gamma_3(self, T):
        """
        Compute the third-order gamma coefficient of the VIX option price proxy.

        Parameters
        ----------
        T: float
            Maturity.
        """

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

        mu = self.mean(T=T)
        sigma = np.sqrt(self.variance(T=T))

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

        gamma_1 = self.gamma_1(T=T)
        gamma_2 = self.gamma_2(T=T)
        gamma_3 = self.gamma_3(T=T)

        return price_0 + gamma_1 / 2 * price_1 + gamma_2 / 4 * price_2 + gamma_3 / 8 * price_3
    

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

        mu = self.mean(T=T)
        sigma = np.sqrt(self.variance(T=T))

        x = np.exp(mu / 2 + sigma**2 / 8)

        gamma_1 = self.gamma_1(T=T)
        gamma_2 = self.gamma_2(T=T)
        gamma_3 = self.gamma_3(T=T)

        return x * (1 + gamma_1 / 2 + gamma_2 / 4 + gamma_3 / 8)


    def implied_vol_proxy(self, k, T):
        """
        Compute the implied volatility implied from the approximation of VIX option prices
        using the root-find method.

        Parameters
        ----------
        k: float or ndarray
            Log-moneyness of the VIX option.
        T: float
            Maturity.
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

        return utils.implied_vol_bisection(F, kappa, T, otm_price, opttype)
    
    
    def implied_vol_expan(self, k, T):
        """
        Approximate the implied volatility of the VIX option using the Taylor expansion.

        Parameters
        ----------
        k: float or ndarray
            Log-moneyness of the VIX option.
        T: float
            Maturity.
        """
        if T <= 0:
            raise ValueError("Maturity T must be positive.")
        
        k = np.atleast_1d(np.asarray(k))

        F = self.vix_fut_price_proxy(T)

        kappa = F * np.exp(k)

        mu = self.mean(T=T)
        sigma = np.sqrt(self.variance(T=T))

        tilde_sigma = sigma / np.sqrt(T)
        m = mu / 2 + sigma**2 / 8 - np.log(kappa)

        gamma_2 = self.gamma_2(T=T)
        gamma_3 = self.gamma_3(T=T)

        iv_0 = tilde_sigma / 2

        iv_1 = (gamma_2 / (2 * tilde_sigma * T)
                + 3 * gamma_3 / (8 * tilde_sigma * T)
                - gamma_3 * m / (tilde_sigma**3 * T**2))
        
        return iv_0 + iv_1
    
    
    def vix_price(self, kappa, T, opttype, n_time, n_space):
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
                integral += w * f(u, x) * self.xi0

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
    
    def vix_implied_vol(self, k, T, n_time, n_space):
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
        """
        if T <= 0:
            raise ValueError("Maturity T must be positive.")
        
        k = np.atleast_1d(np.asarray(k))

        F = self.vix_price(0, T, opttype=0, n_time=n_time, n_space=n_space)

        kappa = F * np.exp(k)
        
        opttype = 2 * (kappa >= F) - 1

        otm_price = np.array(
            [
                self.vix_price(k, T, opttype_i, n_time, n_space) for k, opttype_i in zip(kappa, opttype)
            ]
        )

        return utils.implied_vol_bisection(F, kappa, T, otm_price, opttype)