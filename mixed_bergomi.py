import os
import numpy as np

import utils
import standard_bergomi, rough_bergomi

# Clipping bounds to prevent overflow/underflow errors in np.exp()
EXP_LOWER_BOUND = -700
EXP_UPPER_BOUND = 700

# Small number to prevent division by zero
EPSILON = 1e-9

class MixedStandardBergomi:
    """
    Implementation of the mixed standard Bergomi model with the assumption that
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

    def __init__(self, omega, k, xi0, lbd):
        """
        Initialize the mixed rough Bergomi model.
        See class docstring for parameter definitions.
        """
        
        self.omega = np.atleast_1d(np.asarray(omega, dtype=float))

        if np.any(self.omega <= 0.0):
            raise ValueError("Volatility of volatility eta must be positive.")
        
        self.omega1, self.omega2 = omega
        self.k = k
        self.xi0 = xi0
        self.lbd = lbd

        self.delta_vix = 1.0 / 12.0
        self.x0 = np.log(xi0)

        self.inst1 = standard_bergomi.StandardBergomi(self.omega1, k, xi0)
        self.inst2 = standard_bergomi.StandardBergomi(self.omega2, k, xi0)
        

    def vix_price_proxy(self, kappa, T, opttype, n_gauss):
        """
        Compute the VIX derivative price using one dimensional Gauss-Hermite quadrature method.

        Parameters
        ----------
        kappa: float
            Strike price.
        T: float
            Maturity.
        opttype: int, optional
            Option type: 1 for call, -1 for put, and 0 for futures.
        n_gauss: int
                Number of nodes for the one-dimensional Gauss-Hermite quadrature.
        """
        if T <= 0:
            raise ValueError("Maturity T must be positive.")

        kappa = np.atleast_1d(np.asarray(kappa))

        mu1 = self.inst1.mean(T) - self.x0
        mu2 = self.inst2.mean(T) - self.x0
        sigma1 = np.sqrt(self.inst1.variance(T))
        sigma2 = np.sqrt(self.inst2.variance(T))

        gamma_11 = self.inst1.gamma_1(T)
        gamma_21 = self.inst1.gamma_2(T)
        gamma_31 = self.inst1.gamma_3(T)
        gamma_12 = self.inst2.gamma_1(T)
        gamma_22 = self.inst2.gamma_2(T)
        gamma_32 = self.inst2.gamma_3(T)

        z, weights_trans = utils.gauss_hermite(1, n_gauss)

        F = np.exp(self.x0) * (self.lbd * np.exp(mu1 + sigma1 * z) 
                               + (1 - self.lbd) * np.exp(mu2 + sigma2 * z))
        
        if opttype in [-1, 1]:
            payoff = np.maximum(opttype * (np.sqrt(F) - kappa[:, np.newaxis]), 0)
        elif opttype == 0:
            payoff = np.tile(np.sqrt(F), (len(kappa), 1))

        price_0 = np.sum(weights_trans * payoff, axis=1)
        
        def payoff_derivative(x):
            if opttype == 1:
                condition = x > kappa[:, np.newaxis] ** 2
                return np.where(condition, 1 / (2 * np.sqrt(x)), 0)
            elif opttype == 0:
                deriv = 1 / (2 * np.sqrt(x))
                return np.tile(deriv, (len(kappa), 1))
            else:
                condition = x < kappa[:, np.newaxis] ** 2
                return np.where(condition, -1 / (2 * np.sqrt(x)), 0)
        
        mu0 = mu1 / (self.omega1**2)

        def psi1(x):
            term1 = self.lbd * np.exp(x)
            if self.omega1 == 0:
                term2 = (1 - self.lbd) * np.exp(mu2)
            else:
                term2 = (1 - self.lbd) * np.exp(self.omega2 * (self.omega1 - self.omega2) * (-mu0)
                                                + self.omega2 / self.omega1 * x)
            return payoff_derivative(np.exp(self.x0) * (term1 + term2)) * np.exp(self.x0) * term1
        
        def psi2(x):
            term1 = (1 - self.lbd) * np.exp(x)
            if self.omega2 == 0:
                term2 = self.lbd * np.exp(mu1)
            else:
                term2 = self.lbd * np.exp(self.omega1 * (self.omega2 - self.omega1) * (-mu0)
                                           + self.omega1 / self.omega2 * x)
            return payoff_derivative(np.exp(self.x0) * (term1 + term2)) * np.exp(self.x0) * term1
    
        price_1 = (gamma_11 * np.sum(weights_trans * psi1(mu1 + sigma1 * z), axis=1)
                   + gamma_12 * np.sum(weights_trans * psi2(mu2 + sigma2 * z), axis=1))
        
        price_2 = (gamma_21 * np.sum(weights_trans * z * psi1(mu1 + sigma1 * z), axis=1) / sigma1
                   + gamma_22 * np.sum(weights_trans * z * psi2(mu2 + sigma2 * z), axis=1) / sigma2)
        
        price_3 = (gamma_31 * np.sum(weights_trans * (z**2 - 1) * psi1(mu1 + sigma1 * z), axis=1) / sigma1**2
                   + gamma_32 * np.sum(weights_trans * (z**2 - 1) * psi2(mu2 + sigma2 * z), axis=1) / sigma2**2)
        
        price  = price_0 + price_1 + price_2 + price_3
        
        return price
    
    def implied_vol_proxy(self, k, T, n_gauss):
        """
        Compute the implied volatility implied from the approximation of VIX option prices
        using the root-find method.

        Parameters
        ----------
        k: float or ndarray
            Log-moneyness of the VIX option.
        T: float
            Maturity.
        n_gauss: int
                Number of nodes for the one-dimensional Gauss-Hermite quadrature.
        """
        if T <= 0:
            raise ValueError("Maturity T must be positive.")

        k = np.atleast_1d(np.asarray(k))

        F = self.vix_price_proxy(0, T, opttype=0, n_gauss=n_gauss)

        kappa = F * np.exp(k)
        
        opttype = 2 * (kappa >= F) - 1

        otm_price = np.array(
            [
                self.vix_price_proxy(k, T, opttype=opttype_i, n_gauss=n_gauss)[0]
                for k, opttype_i in zip(kappa, opttype)
            ]
        )

        return utils.implied_vol_bisection(F, kappa, T, otm_price, opttype)
    
    def implied_vol_proxy_cal(self, k, T, F_mkt, n_gauss):
        """
        Compute the implied volatility implied from the approximation of VIX option prices
        using the root-find method.

        Parameters
        ----------
        k: float or ndarray
            Log-moneyness of the VIX option.
        T: float
            Maturity.
        n_gauss: int
                Number of nodes for the one-dimensional Gauss-Hermite quadrature.
        """
        if T <= 0:
            raise ValueError("Maturity T must be positive.")

        k = np.atleast_1d(np.asarray(k))

        kappa = F_mkt * np.exp(k)
        
        opttype = 2 * (kappa >= F_mkt) - 1

        otm_price = np.array(
            [
                self.vix_price_proxy(k, T, opttype=opttype_i, n_gauss=n_gauss)[0]
                for k, opttype_i in zip(kappa, opttype)
            ]
        )

        return utils.implied_vol_bisection(F_mkt, kappa, T, otm_price, opttype)
    

    def vix_price(self, kappa, T, opttype, n_time, n_space):
        """
        Approximate the price of the VIX option using the two-dimensional quadrature.

        Parameters
        ----------
        kappa: float, 
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
                integral += w * f(u, x) * self.xi0

            return integral / self.delta_vix
        
        nodes, weights = utils.gauss_hermite(var, n_space)

        prices = 0
        for (x, w) in zip(nodes, weights):

            vix2_mixed = self.lbd * vix2(x, self.omega1) + (1 - self.lbd) * vix2(x, self.omega2)

            if opttype == 1: 
                prices += w * np.maximum(0, np.sqrt(vix2_mixed) - kappa)
            elif opttype == -1:
                prices += w * np.maximum(0, -np.sqrt(vix2_mixed) + kappa)
            else:
                prices += w * np.sqrt(vix2_mixed)
        
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
                self.vix_price(k, T, opttype=opttype_i, n_time=n_time, n_space=n_space)
                for k, opttype_i in zip(kappa, opttype)
            ]
        )

        return utils.implied_vol_bisection(F, kappa, T, otm_price, opttype)
    

    def abc(self, T, rule, EXP_LOWER_BOUND, EXP_UPPER_BOUND):
        """
        Compute the necessary parameters for the function g.
        g (y) := ( 1 + b * exp (c * y))^(1 / 2)
        for the Hermite expansion.
        
        Parameters
        ----------
        T: float
            Maturity.
        rule: int, optional
            The rule of the implied volatility expansion.
            rule=1, choose the higher volatility as the base,
            rule=-1, choose the lower volatility as the base
        """
        mu1 = self.inst1.mean(T)
        mu2 = self.inst2.mean(T)
        sigma1 = np.sqrt(self.inst1.variance(T))
        sigma2 = np.sqrt(self.inst2.variance(T))

        if (rule == -1 and sigma1 < sigma2) or \
            (rule == 1 and sigma1 > sigma2):
            a = self.lbd**0.5 * np.exp(mu1 / 2 + sigma1**2 / 8)
            b_factor = (1 - self.lbd) / self.lbd
            exp_arg_b = mu2 - mu1 + (sigma2 - sigma1) * sigma1 / 2
            b = b_factor * np.exp(np.clip(exp_arg_b, EXP_LOWER_BOUND, EXP_UPPER_BOUND))
            c = sigma2 - sigma1
        else:
            a = (1 - self.lbd)**0.5 * np.exp(mu2 / 2 + sigma2**2 / 8)
            b_factor = self.lbd / (1 - self.lbd)
            exp_arg_b = mu1 - mu2 + (sigma1 - sigma2) * sigma2 / 2
            b = b_factor * np.exp(np.clip(exp_arg_b, EXP_LOWER_BOUND, EXP_UPPER_BOUND))
            c = sigma1 - sigma2

        return a, b, c


class MixedRoughBergomi:
    """
    Implementation of the mixed rough Bergomi model with the assumption that
    the initial forward variance curve is flat.

    Parameters
    ----------
    eta: float
        Vol-of-vol for each process.
    H: float
        Hurst parameter of the fractional Brownian motion (must be positive).
    xi: float
        Initial forward instantaneous variance.
    """

    def __init__(self, eta, H, xi0, lbd):
        """
        Initialize the mixed rough Bergomi model.
        See class docstring for parameter definitions.
        """
        
        if H <= 0.0:
            raise ValueError("Hurst parameter H must be positive.")
        
        self.eta = np.atleast_1d(np.asarray(eta, dtype=float))

        if np.any(self.eta <= 0.0):
            raise ValueError("Volatility of volatility eta must be positive.")
        
        self.eta1, self.eta2 = eta
        self.H = H
        self.xi0 = xi0
        self.lbd = lbd

        self.delta_vix = 1.0 / 12.0
        self.x0 = np.log(xi0)

        self.inst1 = rough_bergomi.RoughBergomi(self.eta1, H, xi0)
        self.inst2 = rough_bergomi.RoughBergomi(self.eta2, H, xi0)
        

    def vix_price_proxy(self, kappa, T, opttype, n_gauss):
        """
        Compute the VIX derivative price using one dimensional Gauss-Hermite quadrature method.

        Parameters
        ----------
        kappa: float or ndarray
            Strike price(s).
        T: float
            Maturity.
        opttype: int, optional
            Option type: 1 for call, -1 for put, and 0 for futures.
        n_gauss: int
            Number of nodes for the one-dimensional Gauss-Hermite quadrature.
        """
        kappa = np.atleast_1d(np.asarray(kappa))

        mu1 = self.inst1.mean(T) - self.x0
        mu2 = self.inst2.mean(T) - self.x0
        sigma1 = np.sqrt(self.inst1.variance(T))
        sigma2 = np.sqrt(self.inst2.variance(T))

        gamma_11 = self.inst1.gamma_1(T)
        gamma_21 = self.inst1.gamma_2(T)
        gamma_31 = self.inst1.gamma_3(T)
        gamma_12 = self.inst2.gamma_1(T)
        gamma_22 = self.inst2.gamma_2(T)
        gamma_32 = self.inst2.gamma_3(T)

        z, weights_trans = utils.gauss_hermite(1, n_gauss)

        F = np.exp(self.x0) * (self.lbd * np.exp(mu1 + sigma1 * z) 
                               + (1 - self.lbd) * np.exp(mu2 + sigma2 * z))

        if opttype in [-1, 1]:
            payoff = np.maximum(opttype * (np.sqrt(F) - kappa[:, np.newaxis]), 0)
        elif opttype == 0:
            payoff = np.tile(np.sqrt(F), (len(kappa), 1))

        price_0 = np.sum(weights_trans * payoff, axis=1)
        
        def payoff_derivative(x):
            if opttype == 1:
                condition = x > kappa[:, np.newaxis] ** 2
                return np.where(condition, 1 / (2 * np.sqrt(x)), 0)
            elif opttype == 0:
                deriv = 1 / (2 * np.sqrt(x))
                return np.tile(deriv, (len(kappa), 1))
            else:
                condition = x < kappa[:, np.newaxis] ** 2
                return np.where(condition, -1 / (2 * np.sqrt(x)), 0)
        
        mu0 = mu1 / (self.eta1**2)

        def psi1(x):
            term1 = self.lbd * np.exp(x)
            if self.eta1 == 0:
                term2 = (1 - self.lbd) * np.exp(mu2)
            else:
                term2 = (1 - self.lbd) * np.exp(self.eta2 * (self.eta1 - self.eta2) * (-mu0)
                                                + self.eta2 / self.eta1 * x)
            return payoff_derivative(np.exp(self.x0) * (term1 + term2)) * np.exp(self.x0) * term1
        
        def psi2(x):
            term1 = (1 - self.lbd) * np.exp(x)
            if self.eta2 == 0:
                term2 = self.lbd * np.exp(mu1)
            else:
                term2 = self.lbd * np.exp(self.eta1 * (self.eta2 - self.eta1) * (-mu0)
                                           + self.eta1 / self.eta2 * x)
            return payoff_derivative(np.exp(self.x0) * (term1 + term2)) * np.exp(self.x0) * term1
    
        price_1 = (gamma_11 * np.sum(weights_trans * psi1(mu1 + sigma1 * z), axis=1)
                   + gamma_12 * np.sum(weights_trans * psi2(mu2 + sigma2 * z), axis=1))
        
        price_2 = (gamma_21 * np.sum(weights_trans * z * psi1(mu1 + sigma1 * z), axis=1) / sigma1
                   + gamma_22 * np.sum(weights_trans * z * psi2(mu2 + sigma2 * z), axis=1) / sigma2)
        
        price_3 = (gamma_31 * np.sum(weights_trans * (z**2 - 1) * psi1(mu1 + sigma1 * z), axis=1) / sigma1**2
                   + gamma_32 * np.sum(weights_trans * (z**2 - 1) * psi2(mu2 + sigma2 * z), axis=1) / sigma2**2)
        
        price  = price_0 + price_1 + price_2 + price_3
        
        return price
    
    def implied_vol_proxy(self, k, T, n_gauss):
        """
        Compute the implied volatility implied from the approximation of VIX option prices
        using the root-find method.

        Parameters
        ----------
        k: float or ndarray
            Log-moneyness of the VIX option.
        T: float
            Maturity.
        n_gauss: int
                Number of nodes for the one-dimensional Gauss-Hermite quadrature.
        """
        if T <= 0:
            raise ValueError("Maturity T must be positive.")

        k = np.atleast_1d(np.asarray(k))

        F = self.vix_price_proxy(0, T, opttype=0, n_gauss=n_gauss)

        kappa = F * np.exp(k)
        
        opttype = 2 * (kappa >= F) - 1

        otm_price = np.array(
            [
                self.vix_price_proxy(k, T, opttype=opttype_i, n_gauss=n_gauss)[0]
                for k, opttype_i in zip(kappa, opttype)
            ]
        )

        return utils.implied_vol_bisection(F, kappa, T, otm_price, opttype)
    

    def implied_vol_proxy_cal(self, k, T, F_mkt, n_gauss):
        """
        Compute the implied volatility implied from the approximation of VIX option prices
        using the root-find method.

        Parameters
        ----------
        k: float or ndarray
            Log-moneyness of the VIX option.
        T: float
            Maturity.
        n_gauss: int
                Number of nodes for the one-dimensional Gauss-Hermite quadrature.
        """
        if T <= 0:
            raise ValueError("Maturity T must be positive.")

        k = np.atleast_1d(np.asarray(k))

        kappa = F_mkt * np.exp(k)
        
        opttype = 2 * (kappa >= F_mkt) - 1

        otm_price = np.array(
            [
                self.vix_price_proxy(k, T, opttype=opttype_i, n_gauss=n_gauss)[0]
                for k, opttype_i in zip(kappa, opttype)
            ]
        )

        return utils.implied_vol_bisection(F_mkt, kappa, T, otm_price, opttype)
    
    
    def vix_index(self, T, n_disc, n_mc, seed = None):
        """
        Simulated the path for the VIX index using the Monte-Carlo method.

        Parameters
        ----------
        T: float
            Maturity.
        n_disc : int
            Number of time discretization steps.
        n_mc : int
            Number of Monte Carlo paths.
        seed : int or None, optional
            Random seed for reproducibility. Default is None.
        """

        if seed is not None:
            np.random.seed(seed)

        tab_t = np.linspace(T, T + self.delta_vix, n_disc + 1)
        L1 = self.inst1.cholesky_cov_matrix_vix(T=T, n_disc=n_disc)
        L2 = self.inst2.cholesky_cov_matrix_vix(T=T, n_disc=n_disc)

        mu1 = self.x0 - self.eta1**2 / (4 * self.H) * (tab_t**(2 * self.H) - (tab_t - T)**(2 * self.H))
        mu1 = mu1[1:]
        mu2 = self.x0 - self.eta2**2 / (4 * self.H) * (tab_t**(2 * self.H) - (tab_t - T)**(2 * self.H))
        mu2 = mu2[1:]

        z = np.random.randn(n_mc, n_disc)
        samples = np.zeros(n_mc)

        z_T = z.T
        x1_T = mu1[:, None] + L1 @ z_T
        x2_T = mu2[:, None] + L2 @ z_T

        combined_exp = self.lbd * np.exp(x1_T) + (1 - self.lbd) * np.exp(x2_T)
        samples = np.mean(combined_exp, axis=0)

        return np.sqrt(samples)


    def vix_price(self, kappa, T, opttype, n_disc, n_mc, seed = None):
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

        kappa = np.atleast_1d(np.asarray(kappa))
        
        if seed is not None:
            np.random.seed(seed)
        
        vix = self.vix_index(T, n_disc, n_mc, seed)

        if opttype == 1:
            # opttype == 1: call option
            samples = np.maximum(0, vix - kappa[:, np.newaxis])
        elif opttype == -1:
            # opttype == -1: put option
            samples = np.maximum(0, - vix + kappa[:, np.newaxis])
        else:
            samples = np.tile(vix, (len(kappa), 1))

        estimate = np.mean(samples, axis=1)
        return estimate
    
    def vix_implied_vol(self, k, T, n_disc, n_mc, seed = None):
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
        """
        if T <= 0:
            raise ValueError("Maturity T must be positive.")
        
        if seed is not None:
            np.random.seed(seed)

        k = np.atleast_1d(np.asarray(k))

        F = self.vix_price(0, T, opttype=0, n_disc=n_disc, n_mc=n_mc, seed=seed)[0]

        kappa = F * np.exp(k)

        opttype = 2 * (kappa >= F) - 1

        otm_price = np.array(
            [
                self.vix_price(k, T, opttype=opttype_i, n_disc=n_disc, n_mc=n_mc, seed=seed)[0]
                for k, opttype_i in zip(kappa, opttype)
            ]
        )

        return utils.implied_vol_bisection(F, kappa, T, otm_price, opttype)
    
    def abc(self, T, rule, EXP_LOWER_BOUND, EXP_UPPER_BOUND):
        """
        Compute the necessary parameters for the function g.
        g (y) := ( 1 + b * exp (c * y))^(1 / 2)
        for the Hermite expansion.
        
        Parameters
        ----------
        T: float
            Maturity.
        rule: int, optional
            The rule of the implied volatility expansion.
            rule=1, choose the higher volatility as the base,
            rule=-1, choose the lower volatility as the base
        """
        mu1 = self.inst1.mean(T)
        mu2 = self.inst2.mean(T)
        sigma1 = np.sqrt(self.inst1.variance(T))
        sigma2 = np.sqrt(self.inst2.variance(T))

        if (rule == -1 and sigma1 < sigma2) or \
            (rule == 1 and sigma1 > sigma2):
            a = self.lbd**0.5 * np.exp(mu1 / 2 + sigma1**2 / 8)
            b_factor = (1 - self.lbd) / self.lbd
            exp_arg_b = mu2 - mu1 + (sigma2 - sigma1) * sigma1 / 2
            b = b_factor * np.exp(np.clip(exp_arg_b, EXP_LOWER_BOUND, EXP_UPPER_BOUND))
            c = sigma2 - sigma1
        else:
            a = (1 - self.lbd)**0.5 * np.exp(mu2 / 2 + sigma2**2 / 8)
            b_factor = self.lbd / (1 - self.lbd)
            exp_arg_b = mu1 - mu2 + (sigma1 - sigma2) * sigma2 / 2
            b = b_factor * np.exp(np.clip(exp_arg_b, EXP_LOWER_BOUND, EXP_UPPER_BOUND))
            c = sigma1 - sigma2

        return a, b, c
    
