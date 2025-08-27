"""
Rough Bergomi models.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import dblquad
from scipy.linalg import cholesky, eigvals
from scipy.special import beta, hyp2f1, roots_hermite
from scipy.stats import norm

from black_scholes import BlackScholes, BlackScholesLog
from mixed_bergomi import MixedRoughBergomi
from utils import implied_vol, relative_error


# Monte-Carlo simulation (rBergomi model)
class MonteCarloRoughBergomi:
    def __init__(self, eta, H, X0, T, kappa, N, M, Delta=1.0 / 12.0):
        self.eta = eta
        self.H = H
        self.X0 = X0
        self.T = T
        self.Delta = Delta
        self.kappa = kappa
        self.N = N
        self.M = M

        self.t = np.linspace(T, T + self.Delta, N + 1)
        self.mu = X0 - ((eta**2) / (4 * H)) * (
            self.t ** (2 * H) - (self.t - T) ** (2 * H)
        )

    def covariance(self):
        cov = np.zeros((len(self.t), len(self.t)))
        for i in range(len(self.t)):
            for j in range(i, len(self.t)):
                if j == i:
                    cov[i, j] = ((self.eta**2) / (2 * self.H)) * (
                        self.t[i] ** (2 * self.H) - (self.t[i] - self.T) ** (2 * self.H)
                    )
                else:
                    term01 = self.t[j] - self.t[i]
                    term02 = self.H + 0.5
                    term1 = (self.eta**2) * (term01 ** (term02 - 1)) / term02
                    term2 = (self.t[i] ** term02) * hyp2f1(
                        -term02 + 1, term02, term02 + 1, -self.t[i] / term01
                    )
                    term3 = ((self.t[i] - self.T) ** term02) * hyp2f1(
                        -term02 + 1, term02, term02 + 1, -(self.t[i] - self.T) / term01
                    )
                    cov[i, j] = term1 * (term2 - term3)
                    cov[j, i] = cov[i, j]
        return cov

    def call(self):
        mu = self.mu[1:]
        mu_P = np.mean(self.mu[1:])
        cov = self.covariance()[1:, 1:]

        eigenvalues = np.asarray(eigvals(cov))
        if np.any(eigenvalues <= 0):
            # print("Covariance matrix is not positive definite.")
            cov += np.eye(cov.shape[0]) * 1e-11
            L = cholesky(cov, lower=True)
        else:
            L = cholesky(cov, lower=True)

        samples = np.zeros(self.M)
        Z = np.random.randn(self.M, self.N)

        for i in range(self.M):
            # right-point scheme
            XT = mu + L @ Z[i, :]
            VIX2_R = np.mean(np.exp(XT))
            exp_meanX = np.exp(np.mean(XT))

            payoff1 = max(0, np.sqrt(VIX2_R) - self.kappa)
            payoff2 = max(0, np.sqrt(exp_meanX) - self.kappa)
            samples[i] = payoff1 - payoff2

        sigma2_P = np.sum(cov) / self.N**2
        sigma_P = np.sqrt(sigma2_P)

        x = np.exp(mu_P / 2 + sigma2_P / 8)
        y = self.kappa
        z = sigma_P / 2

        # control variate
        cv = BlackScholes(x, y, z, self.T).call()
        # print(cv)

        estimate = np.mean(samples) + cv

        # wide = 1.96 * np.std(samples) / np.sqrt(self.M)
        # ci_lower = estimate - wide
        # ci_upper = estimate + wide
        # print(f"The wide of 95% confidence interval: {wide}")
        # print(f"95% Confidence interval: [{ci_lower}, {ci_upper}]")

        return estimate

    def put(self):
        mu = self.mu[1:]
        mu_P = np.mean(self.mu[1:])
        cov = self.covariance()[1:, 1:]

        eigenvalues = np.asarray(eigvals(cov))
        if np.any(eigenvalues <= 0):
            # print("Covariance matrix is not positive definite.")
            cov += np.eye(cov.shape[0]) * 1e-11
            L = cholesky(cov, lower=True)
        else:
            L = cholesky(cov, lower=True)

        samples = np.zeros(self.M)
        Z = np.random.randn(self.M, self.N)

        for i in range(self.M):
            # right-point scheme
            XT = mu + L @ Z[i, :]
            VIX2_R = np.mean(np.exp(XT))
            exp_meanX = np.exp(np.mean(XT))

            payoff1 = max(0, -np.sqrt(VIX2_R) + self.kappa)
            payoff2 = max(0, -np.sqrt(exp_meanX) + self.kappa)
            samples[i] = payoff1 - payoff2

        sigma2_P = np.sum(cov) / self.N**2
        sigma_P = np.sqrt(sigma2_P)

        x = np.exp(mu_P / 2 + sigma2_P / 8)
        y = self.kappa
        z = sigma_P / 2

        # control variate
        cv = BlackScholes(x, y, z, self.T).put()

        estimate = np.mean(samples) + cv

        # wide = 1.96 * np.std(samples) / np.sqrt(self.M)
        # ci_lower = estimate - wide
        # ci_upper = estimate + wide
        # print(f"The wide of 95% confidence interval: {wide}")
        # print(f"95% Confidence interval: [{ci_lower}, {ci_upper}]")

        return estimate

    def future(self):
        mu = self.mu[1:]
        mu_P = np.mean(self.mu[1:])
        cov = self.covariance()[1:, 1:]

        eigenvalues = np.asarray(eigvals(cov))
        if np.any(eigenvalues <= 0):
            # print("Covariance matrix is not positive definite.")
            cov += np.eye(cov.shape[0]) * 1e-11
            L = cholesky(cov, lower=True)
        else:
            L = cholesky(cov, lower=True)

        samples = np.zeros(self.M)
        Z = np.random.randn(self.M, self.N)

        for i in range(self.M):
            # right-point scheme
            XT = mu + L @ Z[i, :]
            VIX2_R = np.mean(np.exp(XT))
            exp_meanX = np.exp(np.mean(XT))

            payoff1 = np.sqrt(VIX2_R)
            payoff2 = np.sqrt(exp_meanX)
            samples[i] = payoff1 - payoff2

        sigma2_P = np.sum(cov) / self.N**2

        # control variate
        cv = np.exp(mu_P / 2 + sigma2_P / 8)
        # print(cv)

        estimate = np.mean(samples) + cv

        # wide = 1.96 * np.std(samples) / np.sqrt(self.M)
        # ci_lower = estimate - wide
        # ci_upper = estimate + wide
        # print(f"The wide of 95% confidence interval: {wide}")
        # print(f"95% Confidence interval: [{ci_lower}, {ci_upper}]")

        return estimate


# Weak approximation (rBergomi model)
class WeakApproxRoughBergomi:
    def __init__(self, eta, H, X0, T, kappa, Delta=1.0 / 12.0):
        self.eta = eta
        self.H = H
        self.X0 = X0
        self.T = T
        self.Delta = Delta
        self.kappa = kappa

    def mean(self):
        term0 = 2 * self.H + 1
        term1 = (self.eta**2) * (
            (self.T + self.Delta) ** term0 - self.Delta**term0 - self.T**term0
        )
        term2 = 4 * self.Delta * self.H * term0
        return self.X0 - term1 / term2

    def variance(self):
        term0 = 2 * self.H + 2
        term1 = (self.eta**2) / ((self.Delta**2) * ((self.H + 0.5) ** 2))
        term2 = (
            (self.T + self.Delta) ** term0 - self.Delta**term0 + self.T**term0
        ) / term0
        hyp1 = hyp2f1(-self.H - 0.5, self.H + 1.5, self.H + 2.5, -self.T / self.Delta)
        term3 = (
            2
            * beta(1, self.H + 1.5)
            * (self.Delta ** (self.H + 0.5))
            * (self.T ** (self.H + 1.5))
            * hyp1
        )
        return term1 * (term2 - term3)

    def integrand_gamma2(self, t, u):
        term1 = (self.T * t + self.Delta) ** (self.H + 0.5) - (self.T * t) ** (
            self.H + 0.5
        )
        term2 = (self.T + self.Delta * u) ** (2 * self.H) - (self.Delta * u) ** (
            2 * self.H
        )
        term3 = (self.T * t + self.Delta * u) ** (self.H - 0.5)
        return term1 * term2 * term3

    def omega_gamma3(self, u):
        delta = self.Delta / self.T

        term0 = self.H + 0.5
        term1 = ((1 - u) ** term0) * (delta**term0) * beta(1, term0)

        hyp1 = hyp2f1(-term0, term0, term0 + 1, -(1 + delta * u) / (delta * (1 - u)))
        term2 = ((1 + delta * u) ** term0) * hyp1

        hyp2 = hyp2f1(-term0, term0, term0 + 1, -u / (1 - u))
        term3 = ((delta * u) ** term0) * hyp2

        hyp3 = hyp2f1(-term0 + 1, term0 + 1, term0 + 2, -1 / (delta * u))
        term4 = beta(1, term0 + 1) * ((delta * u) ** (term0 - 1)) * hyp3
        return term1 * (term2 - term3) - term4

    def integrand_gamma3(self, t, u):
        omega = self.omega_gamma3(u)
        delta = self.Delta / self.T
        term1 = (t + delta) ** (self.H + 0.5) - t ** (self.H + 0.5)
        term2 = (t + delta * u) ** (self.H - 0.5)
        return term1 * term2 * omega

    def coefficients(self):
        sigma2 = self.variance()

        gamma = np.zeros(4)

        gamma[0] = 1

        term10 = 4 * self.H + 1
        term11 = (
            (self.T + self.Delta) ** term10 + self.Delta**term10 - self.T**term10
        ) / (self.Delta * term10)
        term13 = 2 * self.H + 1
        term14 = (
            (self.T + self.Delta) ** term13 - self.Delta**term13 - self.T**term13
        ) / (self.Delta * term13)
        hyp11 = hyp2f1(
            -2 * self.H, 2 * self.H + 1, 2 * self.H + 2, -self.Delta / self.T
        )
        term15 = (
            2
            * beta(1, term13)
            * (self.Delta ** (2 * self.H))
            * (self.T ** (2 * self.H))
            * hyp11
        )
        term161 = (self.eta**2) * (
            (self.T + self.Delta) ** term13 - self.Delta**term13 - self.T**term13
        )
        term162 = 4 * self.Delta * self.H * term13
        gamma[1] = (
            ((self.eta**4) / (32 * (self.H**2))) * (term11 - term14**2 - term15)
            + term161 / term162
            - sigma2 / 2
        )

        term20 = 2 * self.H + 1
        term21 = -((self.eta**4) * self.T) / (2 * self.Delta * self.H * term20)
        term22 = dblquad(self.integrand_gamma2, 0, 1, lambda t: 0, lambda t: 1)[0]
        term231 = (
            (self.eta**2)
            * sigma2
            * ((self.T + self.Delta) ** term20 - self.Delta**term20 - self.T**term20)
        )
        term232 = 4 * self.Delta * self.H * term20
        gamma[2] = term21 * term22 + term231 / term232

        term311 = ((self.eta) ** 4) * (self.T ** (4 * self.H + 2))
        term312 = 2 * (self.Delta**2) * ((self.H + 0.5) ** 2)
        term32 = dblquad(self.integrand_gamma3, 0, 1, lambda t: 0, lambda t: 1)[0]
        gamma[3] = (term311 / term312) * term32 - (sigma2**2) / 2
        return gamma

    def call(self):
        mu = self.mean()
        sigma2 = self.variance()
        sigma = np.sqrt(sigma2)

        x = np.exp(mu / 2 + sigma2 / 8)
        y = self.kappa
        z = sigma / 2

        BS = BlackScholes(x, y, z, self.T)

        P = np.zeros(4)

        P[0] = BS.call()
        # print(P[0])

        P[1] = (x / 2) * BS.delta_call()

        P[2] = P[1] / 2 + ((x**2) / 4) * BS.gamma()

        P[3] = -P[1] / 2 + (3 * P[2]) / 2 + ((x**3) / 8) * BS.speed()

        gamma = self.coefficients()
        return np.sum(gamma * P)

    def put(self):
        mu = self.mean()
        sigma2 = self.variance()

        x = np.exp(mu / 2 + sigma2 / 8)
        y = self.kappa
        z = np.sqrt(sigma2) / 2

        BS = BlackScholes(x, y, z, self.T)

        P = np.zeros(4)

        P[0] = BS.put()
        # print(P[0])

        P[1] = (x / 2) * BS.delta_put()

        P[2] = P[1] / 2 + ((x**2) / 4) * BS.gamma()

        P[3] = -P[1] / 2 + (3 * P[2]) / 2 + ((x**3) / 8) * BS.speed()

        gamma = self.coefficients()
        return np.sum(gamma * P)

    def future(self):
        mu = self.mean()
        sigma2 = self.variance()

        x = np.exp(mu / 2 + sigma2 / 8)

        P = np.zeros(4)

        for i in range(4):
            P[i] = (2 ** (-i)) * x

        gamma = self.coefficients()
        return np.sum(gamma * P)

    def call_trans(self):
        mu = self.mean()
        sigma2 = self.variance()

        x = mu / 2 + sigma2 / 8
        k = np.log(self.kappa)
        sigma = np.sqrt(sigma2 / self.T) / 2
        T = self.T

        BS = BlackScholesLog(x, k, sigma, T)

        P = np.zeros(4)

        P[0] = BS.call()
        # print(P[0])

        P[1] = (1 / 2) * BS.delta_call()

        P[2] = (1 / 4) * BS.gamma()

        P[3] = (1 / 8) * BS.speed()

        gamma = self.coefficients()
        return np.sum(gamma * P)

    def implied_vol(self):
        mu = self.mean()
        sigma2 = self.variance()

        x = mu / 2 + sigma2 / 8
        k = np.log(self.kappa)

        # Original Implied volatility expansion - functional
        """
        sigma = np.sqrt(sigma2 / self.T) / 2
        T = self.T

        BS = BS_formula_log(x, k, sigma, T)
        gamma = self.coefficients()

        v = np.zeros(4)

        v[0] = BS.call() * gamma[0]

        v[1] = (1 / 2) * BS.delta_call() * gamma[1]

        v[2] = (1 / 4) * BS.gamma() * gamma[2]

        v[3] = (1 / 8) * BS.speed() * gamma[3]

        iv = np.zeros(4)

        iv[0] = sigma

        iv[1] = v[1] / BS.vega()

        iv[2] = (v[2] - (1 / 2) * iv[1] ** 2 * BS.vomma()) / BS.vega()

        iv[3] = (
            (v[3] - iv[1] * iv[2] * BS.vomma() - (1 / 6) * (iv[1] ** 3) * BS.vo_3())
                    / BS.vega()
        )
        # """

        # Original Implied volatility expansion - explicit
        """
        sigma = np.sqrt(sigma2/self.T)
        T = self.T

        iv = np.zeros(4)
        gamma = self.coefficients()

        iv[0] = sigma/2

        d1 = 2*(x-k)/(sigma*np.sqrt(T)) + sigma*np.sqrt(T)/4
        term0 = norm.cdf(d1)/norm.pdf(d1)
        xk = x-k

        iv[1] = gamma[1]/(2*np.sqrt(T)) * term0

        term21 = gamma[2]/(4*np.sqrt(T)) * term0
        term22 = gamma[2]/(2*sigma*T)
        term23 = iv[1]**2 * ((4*xk**2)/(sigma**3*T) - sigma*T/16)
        iv[2] = term21 + term22 - term23

        term31 = gamma[3] / (8 * np.sqrt(T)) * term0
        term32 = gamma[3] / (2 * sigma * T)
        term33 = gamma[3]/2 * (2*xk/(sigma**3*T**2) + 1/(4*sigma*T))
        term34 = iv[1]*iv[2] * (8*xk**2/(sigma**3*T) - sigma*T/8)
        term35 = iv[1]**3/6 * (64*xk**4/(sigma**6*T**2) + sigma**2*T**2/64
                               -2*xk**2/(sigma**3) - 48*xk**2/(sigma**4*T) - T/4)

        iv[3] = term31 + term32 - term33 - term34 - term35
        # """

        # New implied volatility expansion - functional
        # '''
        sigma = np.sqrt(sigma2 / self.T) / 2
        T = self.T

        BS = BlackScholesLog(x, k, sigma, T)
        gamma = self.coefficients()

        d1 = (x - k) / (sigma * np.sqrt(T)) + (sigma * np.sqrt(T)) / 2
        # print(d1)

        parameter = {"mu": [mu], "sigma": [sigma2], "x": [x], "k": [k], "d1": [d1]}
        para = pd.DataFrame(parameter)
        print("Parameters: ", para)

        # v1 = np.exp(x) * norm.cdf(d1)

        v2 = np.exp(x) * norm.pdf(d1) * 1 / (sigma * np.sqrt(T))

        v3 = np.exp(x) * norm.pdf(d1) * (1 / (sigma * np.sqrt(T))) ** 2 * (-d1)

        # iv = np.zeros(4)
        iv = np.zeros(6)

        iv[0] = sigma

        iv[1] = (v2 * (gamma[2] / 4 + gamma[3] / 8)) / BS.vega()

        iv[2] = (
            (v2 + v3) * gamma[3] / 8 - (1 / 2) * iv[1] ** 2 * BS.vomma()
        ) / BS.vega()

        # iv[3] = (
        #     v1 * (gamma[1] / 2 + gamma[2] / 4 + gamma[3] / 8)
        #     - iv[1] * iv[2] * BS.vomma()
        #     - (1 / 6) * (iv[1] ** 3) * BS.vo_3()
        # ) / BS.vega()

        # iv[3] = (
        #     v1 * (gamma[2] / 4 + gamma[3] / 8)
        #     - iv[1] * iv[2] * BS.vomma()
        #     - (1 / 6) * (iv[1] ** 3) * BS.vo_3()
        # ) / BS.vega()

        # iv[4] = (
        #     v1 * (gamma[1] / 2)
        #     - (1 / 2) * (2 * iv[1] * iv[3] + iv[2] ** 2) * BS.vomma()
        #     - (1 / 2) * iv[1] ** 2 * iv[2] * BS.vo_3()
        #     - (1 / 24) * iv[1] ** 4 * BS.vo_4()
        # ) / BS.vega()

        # iv[5] = (
        #     v1 * gamma[1] / 2
        #     - (iv[1] * iv[4] + iv[2] * iv[3]) * BS.vomma()
        #     - (1 / 2) * (iv[1] ** 2 * iv[3] + iv[1] * iv[2] ** 2) * BS.vo_3()
        #     - (1 / 3) * iv[1] ** 3 * iv[2] * BS.vo_4()
        #     - (1 / 120) * iv[1] ** 5 * BS.vo_5()
        # ) / BS.vega()

        # '''

        # New implied volatility expansion - explicit
        """
        sigma = np.sqrt(sigma2 / self.T)
        T = self.T

        iv = np.zeros(4)
        gamma = self.coefficients()

        iv[0] = sigma / 2

        d1 = 2 * (x - k) / (sigma * np.sqrt(T)) + sigma * np.sqrt(T) / 4
        term0 = norm.cdf(d1) / norm.pdf(d1)
        # term0 = norm.cdf(d1) / norm.pdf(d1)
        xk = x - k

        iv[1] = gamma[2]/(2*sigma*T) + gamma[3]/(4*sigma*T)

        term2 = (4*xk**2)/(sigma**3*T) - sigma*T/16
        iv[2] = (
            gamma[3]/(4*sigma*T) - gamma[3]/(2*sigma**2 * T**(3/2)) * d1
            - iv[1]**2 * term2
        )

        term31 = (gamma[1]/2 + gamma[2]/4 + gamma[3]/8) * term0 / np.sqrt(T)
        term32 = iv[1]*iv[2] * (8* xk**2/(sigma**3*T) - sigma*T/8)
        term33 = iv[1]**3 / 6 * (64 * xk**4/(sigma**6 * T**2) + sigma**2 * T**2/64
                               -2 * xk**2/(sigma**2) - 48 * xk**2/(sigma**4*T) - T/4)
        iv[3] = term31 - term32 - term33
        # """

        # Bompis's method
        """
        sigma = np.sqrt(sigma2 / self.T)
        T = self.T

        iv = np.zeros(4)
        gamma = self.coefficients()

        iv[0] = sigma / 2

        iv[1] = gamma[2]/(2*sigma*T) + 3 * gamma[3]/(8*sigma*T)

        iv[2] = - gamma[3]/(sigma**3 * T**2) * (x-k)
        # """

        # print(iv)

        return np.sum(iv)


# different values of eta - call option
def case1():
    H = 0.1
    eta = np.linspace(0.1, 1.5, 10)
    X0 = np.log(0.235**2)
    T = [1 / 12, 3 / 12, 6 / 12]
    kappa = 0.2
    N = 300
    M = 10**5

    VIX_mc_1m = 100 * np.array(
        [MonteCarloRoughBergomi(e, H, X0, T[0], kappa, N, M).call() for e in eta]
    )
    VIX_mc_3m = 100 * np.array(
        [MonteCarloRoughBergomi(e, H, X0, T[1], kappa, N, M).call() for e in eta]
    )
    VIX_mc_6m = 100 * np.array(
        [MonteCarloRoughBergomi(e, H, X0, T[2], kappa, N, M).call() for e in eta]
    )

    VIX_approx_1m = 100 * np.array(
        [WeakApproxRoughBergomi(e, H, X0, T[0], kappa).call() for e in eta]
    )
    VIX_approx_3m = 100 * np.array(
        [WeakApproxRoughBergomi(e, H, X0, T[1], kappa).call() for e in eta]
    )
    VIX_approx_6m = 100 * np.array(
        [WeakApproxRoughBergomi(e, H, X0, T[2], kappa).call() for e in eta]
    )

    rel_error_1m = relative_error(VIX_approx_1m, VIX_mc_1m)
    rel_error_3m = relative_error(VIX_approx_3m, VIX_mc_3m)
    rel_error_6m = relative_error(VIX_approx_6m, VIX_mc_6m)

    _, axs = plt.subplots(1, 2, figsize=(10, 5))

    # Top subplot: VIX prices
    axs[0].plot(eta, VIX_mc_1m, "b.", label="ref 1M")
    axs[0].plot(eta, VIX_mc_3m, "g+", label="ref 3M")
    axs[0].plot(eta, VIX_mc_6m, "rx", label="ref 6M")
    axs[0].plot(eta, VIX_approx_1m, "b:", label="approx 1M")
    axs[0].plot(eta, VIX_approx_3m, "g--", label="approx 3M")
    axs[0].plot(eta, VIX_approx_6m, "r-.", label="approx 6M")
    axs[0].set_xlabel("eta")
    axs[0].set_ylabel("VIX call price (%)")
    axs[0].legend(
        loc="lower right",  # Position of the legend
        ncol=2,
        frameon=True,  # Add a border to the legend
    )
    axs[0].grid(True)

    # Bottom subplot: Relative errors
    axs[1].plot(eta, rel_error_1m, "b.:", label="1M")
    axs[1].plot(eta, rel_error_3m, "g+--", label="3M")
    axs[1].plot(eta, rel_error_6m, "rx-.", label="6M")
    axs[1].set_xlabel("eta")
    axs[1].set_ylabel("Relative error (%)")
    axs[1].legend(
        loc="upper left",  # Position of the legend
        ncol=3,
        frameon=True,  # Add a border to the legend
    )
    axs[1].grid(True)

    plt.tight_layout()

    # Save the plot as a file
    output_file = "vix_call_plot with different eta.png"
    plt.savefig(output_file, dpi=300)


# different values of H
def case2():
    H = np.linspace(0.05, 0.4, 10)
    eta = 1
    X0 = np.log(0.235**2)
    T = [1 / 12, 3 / 12, 6 / 12]
    Delta = 1 / 12
    kappa = 0.2
    N = 300
    M = 10**6

    VIX_mc_1m = 100 * np.array(
        [MonteCarloRoughBergomi(eta, h, X0, T[0], Delta, kappa, N, M).call() for h in H]
    )
    VIX_mc_3m = 100 * np.array(
        [MonteCarloRoughBergomi(eta, h, X0, T[1], Delta, kappa, N, M).call() for h in H]
    )
    VIX_mc_6m = 100 * np.array(
        [MonteCarloRoughBergomi(eta, h, X0, T[2], Delta, kappa, N, M).call() for h in H]
    )

    VIX_approx_1m = 100 * np.array(
        [WeakApproxRoughBergomi(eta, h, X0, T[0], Delta, kappa).call() for h in H]
    )
    VIX_approx_3m = 100 * np.array(
        [WeakApproxRoughBergomi(eta, h, X0, T[1], Delta, kappa).call() for h in H]
    )
    VIX_approx_6m = 100 * np.array(
        [WeakApproxRoughBergomi(eta, h, X0, T[2], Delta, kappa).call() for h in H]
    )

    rel_error_1m = relative_error(VIX_approx_1m, VIX_mc_1m)
    rel_error_3m = relative_error(VIX_approx_3m, VIX_mc_3m)
    rel_error_6m = relative_error(VIX_approx_6m, VIX_mc_6m)

    _, axs = plt.subplots(1, 2, figsize=(10, 5))

    axs[0].plot(H, VIX_mc_1m, "b.", label="ref 1M")
    axs[0].plot(H, VIX_mc_3m, "g+", label="ref 3M")
    axs[0].plot(H, VIX_mc_6m, "rx", label="ref 6M")
    axs[0].plot(H, VIX_approx_1m, "b:", label="approx 1M")
    axs[0].plot(H, VIX_approx_3m, "g--", label="approx 3M")
    axs[0].plot(H, VIX_approx_6m, "r-.", label="approx 6M")
    axs[0].set_xlabel("H")
    axs[0].set_ylabel("VIX call price (%)")
    axs[0].legend(
        loc="lower left",  # Position of the legend
        ncol=2,
        frameon=True,  # Add a border to the legend
    )
    axs[0].grid(True)

    axs[1].plot(H, rel_error_1m, "b.:", label="1M")
    axs[1].plot(H, rel_error_3m, "g+--", label="3M")
    axs[1].plot(H, rel_error_6m, "rx-.", label="6M")
    axs[1].set_xlabel("H")
    axs[1].set_ylabel("Relative Error (%)")
    axs[1].legend(
        loc="upper right",  # Position of the legend
        ncol=3,
        frameon=True,  # Add a border to the legend
    )
    axs[1].grid(True)

    plt.tight_layout()

    # Save the plot as a file
    output_file = "vix_call_plot with different H.png"
    plt.savefig(output_file, dpi=300)


# implied volatility - root-finding method
def case3():
    H = 0.1
    eta = 1
    X0 = np.log(0.235**2)
    T = [1 / 12, 3 / 12, 6 / 12]
    Delta = 1 / 12
    kappa = np.linspace(0.1, 0.4, 10)
    N = 300
    M = 10**6

    VIX_F_mc_1m = MonteCarloRoughBergomi(
        eta, H, X0, T[0], Delta, kappa[0], N, M
    ).future()
    VIX_F_mc_3m = MonteCarloRoughBergomi(
        eta, H, X0, T[1], Delta, kappa[0], N, M
    ).future()
    VIX_F_mc_6m = MonteCarloRoughBergomi(
        eta, H, X0, T[2], Delta, kappa[0], N, M
    ).future()

    VIX_c_mc_1m = np.array(
        [MonteCarloRoughBergomi(eta, H, X0, T[0], Delta, k, N, M).call() for k in kappa]
    )
    VIX_c_mc_3m = np.array(
        [MonteCarloRoughBergomi(eta, H, X0, T[1], Delta, k, N, M).call() for k in kappa]
    )
    VIX_c_mc_6m = np.array(
        [MonteCarloRoughBergomi(eta, H, X0, T[2], Delta, k, N, M).call() for k in kappa]
    )

    VIX_iv_mc_1m = np.array(
        [implied_vol(VIX_F_mc_1m, kappa[i], T[0], VIX_c_mc_1m[i]) for i in range(10)]
    )
    VIX_iv_mc_3m = np.array(
        [implied_vol(VIX_F_mc_3m, kappa[i], T[1], VIX_c_mc_3m[i]) for i in range(10)]
    )
    VIX_iv_mc_6m = np.array(
        [implied_vol(VIX_F_mc_6m, kappa[i], T[2], VIX_c_mc_6m[i]) for i in range(10)]
    )

    VIX_F_approx_1m = WeakApproxRoughBergomi(eta, H, X0, T[0], Delta, kappa[0]).future()
    VIX_F_approx_3m = WeakApproxRoughBergomi(eta, H, X0, T[1], Delta, kappa[0]).future()
    VIX_F_approx_6m = WeakApproxRoughBergomi(eta, H, X0, T[2], Delta, kappa[0]).future()

    VIX_c_approx_1m = np.array(
        [WeakApproxRoughBergomi(eta, H, X0, T[0], Delta, k).call() for k in kappa]
    )
    VIX_c_approx_3m = np.array(
        [WeakApproxRoughBergomi(eta, H, X0, T[1], Delta, k).call() for k in kappa]
    )
    VIX_c_approx_6m = np.array(
        [WeakApproxRoughBergomi(eta, H, X0, T[2], Delta, k).call() for k in kappa]
    )

    VIX_iv_approx_1m = np.array(
        [
            implied_vol(VIX_F_approx_1m, kappa[i], T[0], VIX_c_approx_1m[i])
            for i in range(10)
        ]
    )
    VIX_iv_approx_3m = np.array(
        [
            implied_vol(VIX_F_approx_3m, kappa[i], T[1], VIX_c_approx_3m[i])
            for i in range(10)
        ]
    )
    VIX_iv_approx_6m = np.array(
        [
            implied_vol(VIX_F_approx_6m, kappa[i], T[2], VIX_c_approx_6m[i])
            for i in range(10)
        ]
    )

    rel_error_1m = relative_error(VIX_iv_approx_1m, VIX_iv_mc_1m)
    rel_error_3m = relative_error(VIX_iv_approx_3m, VIX_iv_mc_3m)
    rel_error_6m = relative_error(VIX_iv_approx_6m, VIX_iv_mc_6m)

    fig4, axs = plt.subplots(1, 2, figsize=(10, 5))

    axs[0].plot(kappa, VIX_iv_mc_1m, "b.", label="ref 1M")
    axs[0].plot(kappa, VIX_iv_approx_1m, "b:", label="approx 1M")
    axs[0].plot(kappa, VIX_iv_mc_3m, "g+", label="ref 3M")
    axs[0].plot(kappa, VIX_iv_approx_3m, "g--", label="approx 3M")
    axs[0].plot(kappa, VIX_iv_mc_6m, "rx", label="ref 6M")
    axs[0].plot(kappa, VIX_iv_approx_6m, "r-.", label="approx 6M")
    # axs[0].set_xticks(axs[1].get_xticks())
    # axs[0].set_xticklabels(axs[1].get_xticklabels())
    axs[0].set_xlabel("Strike price")
    axs[0].set_ylabel("Implied volatility")
    axs[0].legend(
        loc="center",  # Position of the legend
        ncol=3,
        frameon=True,  # Add a border to the legend
    )
    axs[0].grid(True)

    axs[1].plot(kappa, rel_error_1m, "b.:", label="1M")
    axs[1].plot(kappa, rel_error_3m, "g+--", label="3M")
    axs[1].plot(kappa, rel_error_6m, "rx-.", label="6M")
    axs[1].set_xlabel("Strike price")
    axs[1].set_ylabel("Relative error (%)")
    axs[1].legend(
        loc="lower right",
        ncol=3,
        frameon=True,
    )
    axs[1].grid(True)

    plt.tight_layout()

    output_file = "vix_implied_vol_plot with different strike price_a.png"
    plt.savefig(output_file, dpi=300)


# implied volatility - Matt's method
def case4():
    H = 0.1
    eta = 1
    X0 = np.log(0.235**2)
    T = [1 / 12, 3 / 12, 6 / 12]
    Delta = 1 / 12
    kappa = np.linspace(0.1, 0.4, 10)
    N = 300
    M = 10**6

    VIX_F_mc_1m = MonteCarloRoughBergomi(
        eta, H, X0, T[0], Delta, kappa[0], N, M
    ).future()
    VIX_F_mc_3m = MonteCarloRoughBergomi(
        eta, H, X0, T[1], Delta, kappa[0], N, M
    ).future()
    VIX_F_mc_6m = MonteCarloRoughBergomi(
        eta, H, X0, T[2], Delta, kappa[0], N, M
    ).future()

    VIX_c_mc_1m = np.array(
        [MonteCarloRoughBergomi(eta, H, X0, T[0], Delta, k, N, M).call() for k in kappa]
    )
    VIX_c_mc_3m = np.array(
        [MonteCarloRoughBergomi(eta, H, X0, T[1], Delta, k, N, M).call() for k in kappa]
    )
    VIX_c_mc_6m = np.array(
        [MonteCarloRoughBergomi(eta, H, X0, T[2], Delta, k, N, M).call() for k in kappa]
    )

    VIX_iv_mc_1m = np.array(
        [
            implied_vol(VIX_F_mc_1m, kappa[i], T[0], VIX_c_mc_1m[i])
            for i in range(len(kappa))
        ]
    )
    VIX_iv_mc_3m = np.array(
        [
            implied_vol(VIX_F_mc_3m, kappa[i], T[1], VIX_c_mc_3m[i])
            for i in range(len(kappa))
        ]
    )
    VIX_iv_mc_6m = np.array(
        [
            implied_vol(VIX_F_mc_6m, kappa[i], T[2], VIX_c_mc_6m[i])
            for i in range(len(kappa))
        ]
    )

    VIX_iv_expan_1m = np.array(
        [
            WeakApproxRoughBergomi(eta, H, X0, T[0], Delta, k).implied_vol()
            for k in kappa
        ]
    )
    VIX_iv_expan_3m = np.array(
        [
            WeakApproxRoughBergomi(eta, H, X0, T[1], Delta, k).implied_vol()
            for k in kappa
        ]
    )
    VIX_iv_expan_6m = np.array(
        [
            WeakApproxRoughBergomi(eta, H, X0, T[2], Delta, k).implied_vol()
            for k in kappa
        ]
    )

    rel_error_1m_e = relative_error(VIX_iv_expan_1m, VIX_iv_mc_1m)
    rel_error_3m_e = relative_error(VIX_iv_expan_3m, VIX_iv_mc_3m)
    rel_error_6m_e = relative_error(VIX_iv_expan_6m, VIX_iv_mc_6m)

    _, axs = plt.subplots(1, 2, figsize=(10, 5))

    axs[0].plot(kappa, VIX_iv_mc_1m, "b.", label="ref 1M")
    axs[0].plot(kappa, VIX_iv_mc_3m, "g+", label="ref 3M")
    axs[0].plot(kappa, VIX_iv_mc_6m, "rx", label="ref 6M")
    axs[0].plot(kappa, VIX_iv_expan_1m, "b:", label="expan 1M")
    axs[0].plot(kappa, VIX_iv_expan_3m, "g--", label="expan 3M")
    axs[0].plot(kappa, VIX_iv_expan_6m, "r-.", label="expan 6M")
    # axs[0].set_xticks(axs[1].get_xticks())
    # axs[0].set_xticklabels(axs[1].get_xticklabels())
    axs[0].set_xlabel("Strike price")
    axs[0].set_ylabel("Implied volatility")
    axs[0].legend(
        loc="upper right",
        ncol=2,
        frameon=True,
    )
    axs[0].grid(True)

    axs[1].plot(kappa, rel_error_1m_e, "b.:", label="1M")
    axs[1].plot(kappa, rel_error_3m_e, "g+--", label="3M")
    axs[1].plot(kappa, rel_error_6m_e, "rx-.", label="6M")
    axs[1].set_xlabel("Strike price")
    axs[1].set_ylabel("Relative error (%)")
    axs[1].legend(
        loc="upper right",
        ncol=3,
        frameon=True,
    )
    axs[1].grid(True)

    plt.tight_layout()

    # Save the plot as a file
    output_file = "vix_implied_vol_plot with different strike price_b.png"
    plt.savefig(output_file, dpi=300)


# compare case 3 and case 4
def case5():
    H = 0.1
    eta = 1
    X0 = np.log(0.235**2)
    T = [1 / 12, 3 / 12, 6 / 12]
    Delta = 1 / 12
    kappa = np.linspace(0.1, 0.4, 10)
    N = 300
    M = 10**6

    VIX_F_mc_1m = MonteCarloRoughBergomi(
        eta, H, X0, T[0], Delta, kappa[0], N, M
    ).future()
    VIX_F_mc_3m = MonteCarloRoughBergomi(
        eta, H, X0, T[1], Delta, kappa[0], N, M
    ).future()
    VIX_F_mc_6m = MonteCarloRoughBergomi(
        eta, H, X0, T[2], Delta, kappa[0], N, M
    ).future()

    VIX_c_mc_1m = np.array(
        [MonteCarloRoughBergomi(eta, H, X0, T[0], Delta, k, N, M).call() for k in kappa]
    )
    VIX_c_mc_3m = np.array(
        [MonteCarloRoughBergomi(eta, H, X0, T[1], Delta, k, N, M).call() for k in kappa]
    )
    VIX_c_mc_6m = np.array(
        [MonteCarloRoughBergomi(eta, H, X0, T[2], Delta, k, N, M).call() for k in kappa]
    )

    VIX_iv_mc_1m = np.array(
        [implied_vol(VIX_F_mc_1m, kappa[i], T[0], VIX_c_mc_1m[i]) for i in range(10)]
    )
    VIX_iv_mc_3m = np.array(
        [implied_vol(VIX_F_mc_3m, kappa[i], T[1], VIX_c_mc_3m[i]) for i in range(10)]
    )
    VIX_iv_mc_6m = np.array(
        [implied_vol(VIX_F_mc_6m, kappa[i], T[2], VIX_c_mc_6m[i]) for i in range(10)]
    )

    VIX_F_approx_1m = WeakApproxRoughBergomi(eta, H, X0, T[0], Delta, kappa[0]).future()
    VIX_F_approx_3m = WeakApproxRoughBergomi(eta, H, X0, T[1], Delta, kappa[0]).future()
    VIX_F_approx_6m = WeakApproxRoughBergomi(eta, H, X0, T[2], Delta, kappa[0]).future()

    VIX_c_approx_1m = np.array(
        [WeakApproxRoughBergomi(eta, H, X0, T[0], Delta, k).call() for k in kappa]
    )
    VIX_c_approx_3m = np.array(
        [WeakApproxRoughBergomi(eta, H, X0, T[1], Delta, k).call() for k in kappa]
    )
    VIX_c_approx_6m = np.array(
        [WeakApproxRoughBergomi(eta, H, X0, T[2], Delta, k).call() for k in kappa]
    )

    VIX_iv_approx_1m = np.array(
        [
            implied_vol(VIX_F_approx_1m, kappa[i], T[0], VIX_c_approx_1m[i])
            for i in range(10)
        ]
    )
    VIX_iv_approx_3m = np.array(
        [
            implied_vol(VIX_F_approx_3m, kappa[i], T[1], VIX_c_approx_3m[i])
            for i in range(10)
        ]
    )
    VIX_iv_approx_6m = np.array(
        [
            implied_vol(VIX_F_approx_6m, kappa[i], T[2], VIX_c_approx_6m[i])
            for i in range(10)
        ]
    )

    rel_error_1m = relative_error(VIX_iv_approx_1m, VIX_iv_mc_1m)
    rel_error_3m = relative_error(VIX_iv_approx_3m, VIX_iv_mc_3m)
    rel_error_6m = relative_error(VIX_iv_approx_6m, VIX_iv_mc_6m)

    VIX_iv_expan_1m = np.array(
        [
            WeakApproxRoughBergomi(eta, H, X0, T[0], Delta, k).implied_vol()
            for k in kappa
        ]
    )
    VIX_iv_expan_3m = np.array(
        [
            WeakApproxRoughBergomi(eta, H, X0, T[1], Delta, k).implied_vol()
            for k in kappa
        ]
    )
    VIX_iv_expan_6m = np.array(
        [
            WeakApproxRoughBergomi(eta, H, X0, T[2], Delta, k).implied_vol()
            for k in kappa
        ]
    )

    rel_error_1m_e = relative_error(VIX_iv_expan_1m, VIX_iv_mc_1m)
    rel_error_3m_e = relative_error(VIX_iv_expan_3m, VIX_iv_mc_3m)
    rel_error_6m_e = relative_error(VIX_iv_expan_6m, VIX_iv_mc_6m)

    pd.set_option("display.precision", 7)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", None)

    implied_vol_1m = {
        "Reference": VIX_iv_mc_1m,
        "Approximation": VIX_iv_approx_1m,
        "Relative error 1": rel_error_1m,
        "Expansion": VIX_iv_expan_1m,
        "Relative error 2": rel_error_1m_e,
    }
    result_1m = pd.DataFrame(implied_vol_1m)
    result_1m.index = kappa
    print(result_1m)

    implied_vol_3m = {
        "Reference": VIX_iv_mc_3m,
        "Approximation": VIX_iv_approx_3m,
        "Relative error 1": rel_error_3m,
        "Expansion": VIX_iv_expan_3m,
        "Relative error 2": rel_error_3m_e,
    }
    result_3m = pd.DataFrame(implied_vol_3m)
    result_3m.index = kappa
    print(result_3m)

    implied_vol_6m = {
        "Reference": VIX_iv_mc_6m,
        "Approximation": VIX_iv_approx_6m,
        "Relative error 1": rel_error_6m,
        "Expansion": VIX_iv_expan_6m,
        "Relative error 2": rel_error_6m_e,
    }
    result_6m = pd.DataFrame(implied_vol_6m)
    result_6m.index = kappa
    print(result_6m)

    fig3, axs = plt.subplots(2, 2, figsize=(10, 8))

    axs[0, 0].plot(kappa, VIX_iv_mc_1m, "b.", label="ref 1M")
    axs[0, 0].plot(kappa, VIX_iv_approx_1m, "b:", label="approx 1M")
    axs[0, 0].plot(kappa, VIX_iv_mc_3m, "g+", label="ref 3M")
    axs[0, 0].plot(kappa, VIX_iv_approx_3m, "g--", label="approx 3M")
    axs[0, 0].plot(kappa, VIX_iv_mc_6m, "rx", label="ref 6M")
    axs[0, 0].plot(kappa, VIX_iv_approx_6m, "r-.", label="approx 6M")
    axs[0, 0].set_xlabel("Strike price")
    axs[0, 0].set_ylabel("Implied volatility")
    axs[0, 0].legend(
        # loc='center',  # Position of the legend
        ncol=3,
        frameon=True,  # Add a border to the legend
    )
    axs[0, 0].grid(True)

    axs[1, 0].plot(kappa, rel_error_1m, "b.:", label="1M")
    axs[1, 0].plot(kappa, rel_error_3m, "g+--", label="3M")
    axs[1, 0].plot(kappa, rel_error_6m, "rx-.", label="6M")
    axs[1, 0].set_xlabel("Strike price")
    axs[1, 0].set_ylabel("Relative error (%)")
    axs[1, 0].legend(
        # loc='lower right',
        ncol=3,
        frameon=True,
    )
    axs[1, 0].grid(True)

    axs[0, 1].plot(kappa, VIX_iv_mc_1m, "b.", label="ref 1M")
    axs[0, 1].plot(kappa, VIX_iv_expan_1m, "b:", label="expan 1M")
    axs[0, 1].plot(kappa, VIX_iv_mc_3m, "g+", label="ref 3M")
    axs[0, 1].plot(kappa, VIX_iv_expan_3m, "g--", label="expan 3M")
    axs[0, 1].plot(kappa, VIX_iv_mc_6m, "rx", label="ref 6M")
    axs[0, 1].plot(kappa, VIX_iv_expan_6m, "r-.", label="expan 6M")
    axs[0, 1].set_xlabel("Strike price")
    axs[0, 1].set_ylabel("Implied volatility")
    axs[0, 1].legend(
        # loc='upper right',
        ncol=3,
        frameon=True,
    )
    axs[0, 1].grid(True)

    axs[1, 1].plot(kappa, rel_error_1m_e, "b.:", label="1M")
    axs[1, 1].plot(kappa, rel_error_3m_e, "g+--", label="3M")
    axs[1, 1].plot(kappa, rel_error_6m_e, "rx-.", label="6M")
    axs[1, 1].set_xlabel("Strike price")
    axs[1, 1].set_ylabel("Relative error (%)")
    axs[1, 1].legend(
        # loc='upper right',
        ncol=3,
        frameon=True,
    )
    axs[1, 1].grid(True)

    plt.tight_layout()

    # Save the plot as a file
    output_file = "vix_implied_vol_plot with different strike price_rough.png"
    plt.savefig(output_file, dpi=300)


# different values of eta - future
def case6():
    H = 0.1
    eta = np.linspace(0.1, 1.5, 10)
    X0 = np.log(0.235**2)
    T = [1 / 12, 3 / 12, 6 / 12]
    Delta = 1 / 12
    kappa = 0.2
    N = 300
    M = 10**6

    VIX_mc_1m = 100 * np.array(
        [
            MonteCarloRoughBergomi(e, H, X0, T[0], Delta, kappa, N, M).future()
            for e in eta
        ]
    )
    VIX_mc_3m = 100 * np.array(
        [
            MonteCarloRoughBergomi(e, H, X0, T[1], Delta, kappa, N, M).future()
            for e in eta
        ]
    )
    VIX_mc_6m = 100 * np.array(
        [
            MonteCarloRoughBergomi(e, H, X0, T[2], Delta, kappa, N, M).future()
            for e in eta
        ]
    )

    VIX_approx_1m = 100 * np.array(
        [WeakApproxRoughBergomi(e, H, X0, T[0], Delta, kappa).future() for e in eta]
    )
    VIX_approx_3m = 100 * np.array(
        [WeakApproxRoughBergomi(e, H, X0, T[1], Delta, kappa).future() for e in eta]
    )
    VIX_approx_6m = 100 * np.array(
        [WeakApproxRoughBergomi(e, H, X0, T[2], Delta, kappa).future() for e in eta]
    )

    rel_error_1m = relative_error(VIX_approx_1m, VIX_mc_1m)
    rel_error_3m = relative_error(VIX_approx_3m, VIX_mc_3m)
    rel_error_6m = relative_error(VIX_approx_6m, VIX_mc_6m)

    fig1, axs = plt.subplots(1, 2, figsize=(10, 5))

    # Top subplot: VIX prices
    axs[0].plot(eta, VIX_mc_1m, "b.", label="ref 1M")
    axs[0].plot(eta, VIX_mc_3m, "g+", label="ref 3M")
    axs[0].plot(eta, VIX_mc_6m, "rx", label="ref 6M")
    axs[0].plot(eta, VIX_approx_1m, "b:", label="approx 1M")
    axs[0].plot(eta, VIX_approx_3m, "g--", label="approx 3M")
    axs[0].plot(eta, VIX_approx_6m, "r-.", label="approx 6M")
    axs[0].set_xlabel("eta")
    axs[0].set_ylabel("VIX call price (%)")
    axs[0].legend(
        loc="lower right",  # Position of the legend
        ncol=2,
        frameon=True,  # Add a border to the legend
    )
    axs[0].grid(True)

    # Bottom subplot: Relative errors
    axs[1].plot(eta, rel_error_1m, "b.:", label="1M")
    axs[1].plot(eta, rel_error_3m, "g+--", label="3M")
    axs[1].plot(eta, rel_error_6m, "rx-.", label="6M")
    axs[1].set_xlabel("eta")
    axs[1].set_ylabel("Relative error (%)")
    axs[1].legend(
        loc="upper left",  # Position of the legend
        ncol=3,
        frameon=True,  # Add a border to the legend
    )
    axs[1].grid(True)

    plt.tight_layout()

    # Save the plot as a file
    output_file = "vix_future_plot with different eta.png"
    plt.savefig(output_file, dpi=300)


# test
def test():
    # Parameters in Chap. 02
    H = 0.1
    eta = 1
    X0 = np.log(0.235**2)
    T = 1 / 12
    Delta = 1 / 12
    kappa = 0.1
    # N = 300
    # M = 5 * 10**4
    iv_approx = WeakApproxRoughBergomi(eta, H, X0, T, Delta, kappa).implied_vol()
    print(f"Estimated implied volatility (weak approx.): {iv_approx}")

    # call_approx = weak_approx_rb(eta, H, X0, T, Delta, kappa).call_trans()
    # print(f"Estimated call option price (weak approx.): {call_approx}")
    #
    # N = 400
    # M = 3 * 10 ** 6
    # # M = 100
    # call_mc = MC_rb(eta, H, X0, T, Delta, kappa, N, M).call()
    # print(f"Estimated call option price (MC simulation): {call_mc}")


if __name__ == "__main__":
    case1()
    # test()
