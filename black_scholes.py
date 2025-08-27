"""
Black-Scholes models.
"""

import numpy as np
from scipy.stats import norm


# Black-Scholes formula (General ver.)
class BlackScholes:
    def __init__(self, x, y, z, T):
        self.x = x
        self.y = y
        self.z = z

        self.d1 = (np.log(x / y) / z) + z / 2
        self.d2 = self.d1 - z

        self.sigma = z / np.sqrt(T)
        self.T = T

    # call price
    def call(self):
        return self.x * norm.cdf(self.d1) - self.y * norm.cdf(self.d2)

    # put price
    def put(self):
        return -self.x * norm.cdf(-self.d1) + self.y * norm.cdf(-self.d2)

    # delta - call
    def delta_call(self):
        return norm.cdf(self.d1)

    # delta - put
    def delta_put(self):
        return -norm.cdf(-self.d1)

    # gamma
    def gamma(self):
        return (1 / (self.x * self.z)) * norm.pdf(self.d1)

    # speed
    def speed(self):
        return -(self.gamma() / self.x) * (
            (np.log(self.x / self.y) / (self.z**2)) + 1.5
        )

    # vega
    def vega(self):
        # return self.x * norm.pdf(self.d1)
        return self.x * np.sqrt(self.T) * norm.pdf(self.d1)

    # vomma
    def vomma(self):
        # term0 = (np.log(self.x / self.y) ** 2 / (self.sigma ** 3)) - (self.sigma) / 4
        # return self.x * norm.pdf(self.d1) * term0
        term0 = (np.log(self.x / self.y) ** 2 / (self.sigma**3 * self.T)) - (
            self.sigma * self.T
        ) / 4
        return self.x * np.sqrt(self.T) * norm.pdf(self.d1) * term0

    def vo_3(self):
        # term0 = np.log(self.x / self.y) ** 2 / (self.sigma ** 3) - (self.sigma) / 4
        # term1 = 3 * np.log(self.x / self.y) ** 2 / (self.sigma ** 4) + 1 / 4
        # return self.x * norm.pdf(self.d1) * (term0 ** 2 - term1)
        term0 = (
            np.log(self.x / self.y) ** 2 / (self.sigma**3 * self.T)
            - (self.sigma * self.T) / 4
        )
        term1 = 3 * np.log(self.x / self.y) ** 2 / (self.sigma**4 * self.T) + self.T / 4
        return self.x * np.sqrt(self.T) * norm.pdf(self.d1) * (term0**2 - term1)


# Black-Scholes formula (Log-spot ver.)
class BlackScholesLog:
    def __init__(self, x, k, sigma, T):
        self.x = x
        self.k = k
        self.sigma = sigma
        self.T = T

        self.d1 = (x - k) / (sigma * np.sqrt(T)) + (sigma * np.sqrt(T)) / 2
        # print(self.d1)
        self.d2 = self.d1 - (sigma * np.sqrt(T))

    # call price
    def call(self):
        return np.exp(self.x) * norm.cdf(self.d1) - np.exp(self.k) * norm.cdf(self.d2)

    # put price
    # def put(self):
    #     return -np.exp(self.x) * norm.cdf(-self.d1) + np.exp(self.k) * norm.cdf(
    #         -self.d2
    #     )

    # first-order partial derivative - call
    def delta_call(self):
        return np.exp(self.x) * norm.cdf(self.d1)

    # delta - put
    # def delta_put(self):
    #     return -norm.cdf(-self.d1)

    # second-order partial derivative - call
    def gamma(self):
        term1 = np.exp(self.x) * norm.cdf(self.d1)
        term2 = (
            np.exp(self.x) * norm.pdf(self.d1) * (1 / (self.sigma * np.sqrt(self.T)))
        )
        return term1 + term2

    # third-order partial derivative - call
    def speed(self):
        term1 = np.exp(self.x) * norm.cdf(self.d1)
        term2 = (
            2
            * np.exp(self.x)
            * norm.pdf(self.d1)
            * (1 / (self.sigma * np.sqrt(self.T)))
        )
        term3 = (
            np.exp(self.x)
            * norm.pdf(self.d1)
            * ((1 / (self.sigma * np.sqrt(self.T))) ** 2)
            * (-self.d1)
        )
        return term1 + term2 + term3

    # first-order partial derivative wrt volatility
    def vega(self):
        return np.exp(self.x) * np.sqrt(self.T) * norm.pdf(self.d1)

    # second-order partial derivative wrt volatility
    def vomma(self):
        term1 = np.exp(self.x) * np.sqrt(self.T) * norm.pdf(self.d1)
        term2 = self.d1 * self.d2 / self.sigma
        return term1 * term2

    # second-order partial derivative wrt volatility
    def vo_3(self):
        term1 = np.exp(self.x) * np.sqrt(self.T) * norm.pdf(self.d1)
        term2 = self.d1 * self.d2 * (self.d1 + self.d2) - 1
        return term1 * term2 / self.sigma**2

    def vo_4(self):
        term1 = np.exp(self.x) * np.sqrt(self.T) * norm.pdf(self.d1)
        term2 = self.d1 * self.d2 * (
            self.d1**2 + 4 * self.d1 * self.d2 + self.d2**2 - 3
        ) - (self.d1 + self.d2)
        return term1 * term2 / self.sigma**3

    def vo_5(self):
        term1 = np.exp(self.x) * np.sqrt(self.T) * norm.pdf(self.d1)
        term2 = (
            self.d1
            * self.d2
            * (
                self.d1**3
                + 9 * self.d1**2 * self.d2
                + 9 * self.d1 * self.d2**2
                + self.d2**3
                - 6 * self.d1
                - 6 * self.d2
            )
        )
        term3 = -(self.d1**2 + 4 * self.d1 * self.d2 + self.d2**2 - 3)
        return term1 * (term2 + term3) / self.sigma**4
