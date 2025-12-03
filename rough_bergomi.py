import os
import numpy as np
from scipy.linalg import eigvals, cholesky
from scipy.special import beta, hyp2f1
from scipy.integrate import dblquad

import utils
from black_scholes import BlackScholes, BlackScholesLog


class RoughBergomi:
    """
    Implementation of the rough Bergomi model with the assumption that
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

    def __init__(self, eta, H, xi0):
        """
        Initialize the rough Bergomi model.
        See class docstring for parameter definitions.
        """
        self.eta = np.atleast_1d(np.asarray(eta, dtype=float))
        if np.any(self.eta <= 0.0):
            raise ValueError("Volatility of volatility eta must be positive.")
        
        if H <= 0.0:
            raise ValueError("Hurst parameter H must be positive.")
        
        self.eta = eta
        self.H = H
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
        term0 = 2 * self.H + 1

        term11 = self.eta**2 * ((T + self.delta_vix) ** term0 - self.delta_vix ** term0 - T**term0)
        term12 = 4 * self.delta_vix * self.H * term0

        return self.x0 - term11 / term12
    

    def variance(self, T):
        """
        Compute the variance of the squared VIX proxy.

        Parameters
        ----------
        T: float
            Maturity.
        """

        hyp1 = hyp2f1(-self.H-0.5, self.H+1.5, self.H+2.5, -T/self.delta_vix)

        term0 = 2 * self.H + 2

        term1 = self.eta**2 / (self.delta_vix**2 * (self.H + 0.5)**2)
        term2 = (((T + self.delta_vix)**term0 - self.delta_vix**term0 + T**term0) / term0
                 - 2 * beta(1, self.H+1.5) * self.delta_vix**(self.H+0.5) * T**(self.H+1.5) * hyp1)
        return term1 * term2


    def gamma_1(self, T):
        """
        Compute the first-order gamma coefficient of the VIX option price proxy.

        Parameters
        ----------
        T: float
            Maturity.
        """

        sigma2 = self.variance(T=T)

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
    

    def gamma_2(self, T):
        """
        Compute the second-order gamma coefficient of the VIX option price proxy.

        Parameters
        ----------
        T: float
            Maturity.
        """
        sigma2 = self.variance(T=T)

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
    

    def gamma_3(self, T):
        """
        Compute the third-order gamma coefficient of the VIX option price proxy.

        Parameters
        ----------
        T: float
            Maturity.
        """
        sigma2 = self.variance(T=T)

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

        cov = self.cov_matrix_vix(T=T, n_disc=n_disc)[1:, 1:]

        eigenvalues = eigvals(cov)

        if np.any(eigenvalues <= 0):
            # print("Covariance matrix is not positive definite.")
            cov += np.eye(cov.shape[0]) * 1e-11
            L = cholesky(cov, lower=True)
        else:
            L = cholesky(cov, lower=True)

        return L
    
    def vix_price(self, kappa, T, opttype, n_disc, n_mc, seed=None):
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
        if seed is not None:
            np.random.seed(seed)

        tab_t = np.linspace(T, T + self.delta_vix, n_disc + 1)
        cov = self.cov_matrix_vix(T=T, n_disc=n_disc)[1:, 1:]
        L = self.cholesky_cov_matrix_vix(T=T, n_disc=n_disc)

        mu = self.x0 - self.eta**2 / (4 * self.H) * (tab_t**(2 * self.H) - (tab_t - T)**(2 * self.H))
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
    
    def vix_implied_vol(self, k, T, n_disc, n_mc, seed=None):
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

        F = self.vix_price(0, T, opttype=0, n_disc=n_disc, n_mc=n_mc, seed=seed)

        kappa = F * np.exp(k)
        
        opttype = 2 * (kappa >= F) - 1

        otm_price = np.array(
            [
                self.vix_price(k, T, opttype_i, n_disc, n_mc, seed) for k, opttype_i in zip(kappa, opttype)
            ]
        )

        return utils.implied_vol_bisection(F, kappa, T, otm_price, opttype)


