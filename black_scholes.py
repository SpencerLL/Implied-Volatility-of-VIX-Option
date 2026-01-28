import os
import numpy as np
from scipy.stats import norm
from scipy.special import binom


class BlackScholes:
    """
    The general version of Black-Scholes model.

    Parameters
    ----------
    s: float
        Spot price.
    k: float
        Strike price.
    simga: float
            Volatility of the underlying asset.
    T: float
        Time to maturity.
    opttype: int or ndarray, optional
            Option type: 1 for call, -1 for put, and 0 for futures.
    """
    
    def __init__(self, s, k, sigma, T, opttype):
        """
        Initialize the Black-Scholes model.
        See class docstring for parameter definitions.
        """

        self.s = np.asarray(s, dtype=float)
        self.k = np.asarray(k, dtype=float)
        self.sigma = np.asarray(sigma, dtype=float)
        self.T = np.asarray(T, dtype=float)
        self.opttype = np.asarray(opttype) 

        if np.any(self.s <= 0.0):
            raise ValueError("Initial spot price s must be positive.")
        
        if np.any(self.sigma <= 0.0):
            raise ValueError("The volatility of option must be positive.")
        

        self.d1 = np.log(s / k) / (self.sigma * np.sqrt(T)) + sigma * np.sqrt(T) / 2
        self.d2 = self.d1 - self.sigma * np.sqrt(T)

    def price(self):
        """
        Compute the call/put option price.
        """
        price = self.opttype * (self.s * norm.cdf(self.opttype * self.d1) 
                               - self.k * norm.cdf(self.opttype * self.d2))
        if price.ndim == 0:
            return price.item()

        return price

    def delta(self):
        """
        Compute the Greek delta for the call/put.
        First order partial derivative of the price function
        with respect to the spot price.
        """

        return self.opttype * norm.cdf(self.opttype * self.d1)


    def gamma(self):
        """
        Compute the Greek gamma.
        Second order partial derivative of the price function
        with respect to the spot price.
        """
        return (1 / (self.s * self.sigma * np.sqrt(self.T))) * norm.pdf(self.d1)


    def speed(self):
        """
        Compute the Greek speed.
        Third order partial derivative of the price function
        with respect to the spot price.
        """
        return (-self.gamma() / self.s
                * (np.log(self.s / self.k) / ((self.sigma * np.sqrt(self.T))**2) +  1.5))


    def vega(self):
        """
        Compute the Greek vega.
        First order partial derivative of the price function with respect to the volatility.
        """
        return self.s * np.sqrt(self.T) * norm.pdf(self.d1)


    def vomma(self):
        """
        Compute the Greek vomma.
        Second order partial derivative of the price function with respect to the volatility.
        """
        term = ((np.log(self.s / self.k)**2 / (self.sigma**3 * self.T))
                 - (self.sigma * self.T) / 4)
        return self.s * np.sqrt(self.T) * norm.pdf(self.d1) * term


class BlackScholesLog:
    """
    The special version of Black-Scholes model using the log-spot and log-strike.

    Parameters
    ----------
    x: float
        Log-Spot price.
    y: float
        Log-strike price.
    simga: float
            Volatility of the underlying asset.
    T: float
        Maturity.
    """

    def __init__(self, x, y, sigma, T, opttype):
        """
        Initialize the Black-Scholes model.
        See class docstring for parameter definitions.
        """

        self.x = x
        self.y = y
        self.sigma = sigma
        self.T = T
        self.opttype = opttype

        self.d1 = (x-y) / (sigma * np.sqrt(T)) + (sigma * np.sqrt(T)) / 2
        self.d2 = self.d1 - (sigma * np.sqrt(T))


    def price(self):
        """
        Compute the call/put option price.
        """

        return self.opttype * (np.exp(self.x) * norm.cdf(self.opttype * self.d1) 
                               - np.exp(self.y) * norm.cdf(self.opttype * self.d2))


    def _prob_hermite_poly(self, i: int, x):
        """
        Calculates the j-th probabilist's Hermite polynomial at point x
        using the recurrence relation.

        Parameters
        ----------
        i: int
            The order of the probabilist's Hermite polynomial.
        x: float
            The evaluate point.
        """
        if i == 0:
            return 1
        if i == 1:
            return x

        # Iteratively compute using the recurrence relation:
        # He_{n+1}(x) = x * He_n(x) - n * He_{n-1}(x)
        he_nm2 = 1
        he_nm1 = x
        for n in range(2, i + 1):
            he_n = x * he_nm1 - (n - 1) * he_nm2
            he_nm2 = he_nm1
            he_nm1 = he_n
        return he_nm1

    def price_derivative(self, n: int):
        """
        Compute the n-th order partial derivative of the call/put price function
        with respect to the log-spot price.

        Parameters
        ----------
        n: int
            The order of the partial derivative.
        """
        if n < 1:
            raise ValueError("The order of the derivative 'n' must be an integer greater than or equal to 1.")
        
        first_term = self.opttype * np.exp(self.x) * norm.cdf(self.opttype * self.d1)

        # The summation term is only added for n >= 2
        if n >= 2:
            summation = 0
            for i in range(1, n):
                he_i_minus_1 = self._prob_hermite_poly(i - 1, self.d1)
                term_i = (binom(n - 1, i) 
                          * ((-1)**(i - 1))
                          * he_i_minus_1 / (self.sigma**i * self.T**(i / 2.0)))
                summation += term_i
        
            second_term = np.exp(self.x) * norm.pdf(self.d1) * summation
            return first_term + second_term
        else: # n == 1
            return first_term

    def vega(self):
        """
        Compute the Greek vega.
        First order partial derivative of the price function with respect to the volatility.
        """
        return np.exp(self.x) * np.sqrt(self.T) * norm.pdf(self.d1)


    def vomma(self):
        """
        Compute the Greek vomma.
        Second order partial derivative of the price function with respect to the volatility.
        """
        term1 = np.exp(self.x) * np.sqrt(self.T) * norm.pdf(self.d1)
        term2 = ((self.x - self.k)**2 / (self.sigma**3 * self.T)
                  - self.sigma * self.T / 4)
        return term1 * term2