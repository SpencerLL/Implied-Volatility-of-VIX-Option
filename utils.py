import os

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.stats import norm
from scipy.special import eval_hermitenorm

from black_scholes import BlackScholes


def gauss_legendre(a, b, n):
        """
        Compute the Gauss-Legendre quadrature points and weights on the interval [a, b].

        Parameters
        ----------
        a : float
            Lower bound of the integration interval.
        b : float
            Upper bound of the integration interval.
        n : int
            Number of quadrature points.
        """
        nodes, weights = np.polynomial.legendre.leggauss(n)
        nodes_a_b = 0.5 * (b - a) * nodes + 0.5 * (b + a)
        weights_a_b = 0.5 * (b - a) * weights
        return nodes_a_b, weights_a_b


def gauss_hermite(var, n: int):
        """
        Compute the Gauss-Hermite quadrature points and weights on the interval [a, b].

        Parameters
        ----------
        var: float
            Variance of the random variable.
        n : int
            Number of quadrature points.
        """
        nodes, weights = np.polynomial.hermite.hermgauss(n)

        nodes_trans = np.sqrt(2 * var) * nodes
        weights_trans = weights / np.sqrt(np.pi)
        return nodes_trans, weights_trans


def prob_hermite_poly(n, y):
    """
    Evaluate the probabilist's Hermite polynomial.

    Parameters
    ----------
    n: int
        Order of Hermite polynomial. (n > 0)
    y: float or np.ndarray
        Points at which to evaluate the polynomial.
    """

    if not isinstance(n, int) or n < 0:
        raise ValueError("Order n must be a non-negative integer.")

    is_scalar = np.isscalar(y)
    y_arr = np.asarray(y, dtype=float)

    if n == 0:
        res_arr = np.ones_like(y_arr)
    else:
        # eval_hermitenorm handles both scalar and array inputs
        res_arr = eval_hermitenorm(n, y_arr)

    return res_arr.item() if is_scalar else res_arr


def hermite_phi_product(n, y):
    """
    Compute the product of the probabilist's Hermite polynomial and the PDF
    of the standard normal random variable.

    Parameters
    ----------
    n: int
        Order of Hermite polynomial. (n > 0)
    y: float or ndarray
        Points at which to evaluate the polynomial.
    """

    if not isinstance(n, int) or n < 0:
        raise ValueError("Order n must be a non-negative integer.")

    is_scalar = np.isscalar(y)
    y_arr = np.asarray(y, dtype=float)

    if n == 0:
        res_arr = np.ones_like(y_arr) * norm.pdf(y)
    else:
        res_arr = eval_hermitenorm(n, y) * norm.pdf(y)
    return res_arr.item() if is_scalar else res_arr


def implied_vol_brentq(f, k, T, market, opttype):
    """
    Compute the implied volatility using the Brent's method.

    Parameters
    ----------
    f: float
        Futures price.
    k: float
        Strike price.
    T: float
        Time to maturity.
    market: float
        Observed market price of the option.
    opttype: int, optional
            Option type: 1 for call, -1 for put, and 0 for futures.
    """

    if (f <= 0) or (k <= 0) or (T <= 0) or (market <= 0):
        return np.nan

    try:
        
        result = optimize.root_scalar(
            f = lambda sigma: BlackScholes(f, k, sigma, T, opttype).price - market,
            bracket=[1e-10, 5.0],
            method="brentq",
        )
       
        return result.root if result.converged else np.nan
    except ValueError:
        return np.nan
    

def implied_vol_bisection(f, k, T, market, opttype, TOL=1e-5, MAX_ITER=1000):
    """
    Compute the implied volatility using a bisection method.

    Parameters
    ----------
    f: float
        Futures price.
    k: float or ndarray
        Strike price.
    T: float
        Time to maturity.
    market: float
        Observed market price of the option.
    opttype: int or ndarray, optional
            Option type: 1 for call, -1 for put, and 0 for futures.
    TOL : float, optional
        Tolerance for convergence of the implied volatility.
    MAX_ITER : int, optional
        Maximum number of iterations for the bisection method.
    """
    k = np.atleast_1d(k)
    market = np.atleast_1d(market)
    opttype = np.full_like(k, opttype)

    if k.shape != market.shape:
        raise ValueError("k and market must have the same shape.")

    if not np.all(np.abs(opttype) == 1):
        raise ValueError("opttype must be either 1 or -1.")

    f = float(f)
    T = float(T)

    if T <= 0 or f <= 0:
        return np.full_like(k, np.nan)
    
    IMPVOL_MIN = 1e-10
    IMPVOL_MAX = 5.0

    low = IMPVOL_MIN * np.ones_like(k)
    high = IMPVOL_MAX * np.ones_like(k)
    mid = 0.5 * (low + high)

    for _ in range(MAX_ITER):

        price = BlackScholes(f, k, mid, T, opttype).price()

        small_market_mask = np.abs(market) < 1e-7

        diff = np.where(small_market_mask,
                        price - market,
                        (price - market) / market)

        if np.all(np.abs(diff) < TOL):
            return mid

        mask = diff > 0
        high[mask] = mid[mask]
        low[~mask] = mid[~mask]
        mid = 0.5 * (low + high)

    # raise ValueError("Implied volatility did not converge.")

    print("Implied volatility did not converge for all log(K/F) values.")

    # Set mid to NaN where the tolerance is not met
    mid = np.where(np.abs(diff) < TOL, mid, np.nan)
    return mid


def implied_vol_mc(f, k, T, mc_error: bool = False):
    """
    Compute the implied volatility using Monte Carlo simulated prices.

    Parameters
    ----------
    f: float
        Futures price.
    k: float or ndarray
        Strike price.
    T: float
        Time to maturity.
    mc_error : bool, optional
        If True, computes the 95% confidence interval for the implied volatility.
    """

    F = np.mean(f)
    k = np.atleast_1d(k)
    
    # opttype: 1 for call, -1 for put, depending on moneyness
    opttype = 2 * (k >= F) - 1  # 1 if K >= F (call), -1 if K < F (put)
    payoff = np.maximum(opttype[None, :] * (f[:, None] - k[None, :]), 0.0)
    otm_price = np.mean(payoff, axis=0)
    otm_impvol = implied_vol_bisection(F, k, T, otm_price, opttype)

    if mc_error:
        error_95 = 1.96 * np.std(payoff, axis=0) / f.shape[0] ** 0.5
        otm_impvol_high = implied_vol_bisection(
            f, k, T, otm_price + error_95, opttype
        )
        otm_impvol_low = implied_vol_bisection(
            f, k, T, otm_price - error_95, opttype
        )
        return {
            "otm_impvol": otm_impvol,
            "otm_impvol_high": otm_impvol_high,
            "otm_impvol_low": otm_impvol_low,
            "error_95": error_95,
            "otm_price": otm_price,
        }

    return otm_impvol


def signed_rel_error(reference, approx):
    """
    Compute the signed relative errors (%).

    Parameters
    ----------
    approx: float or np.ndarray
            Approximation.
    ref: float or np.ndarray
        Reference values.
    """

    error = (approx - reference) / reference * 100
    return error


def rel_error(reference, approx):
    """
    Compute the relative errors (%).

    Parameters
    ----------
    approx: float or np.ndarray
            Approximation.
    ref: float or np.ndarray
        Reference values.
    """

    error = np.abs(approx - reference) / np.abs(reference) * 100
    return error


def plot_style_4(kappa, maturities, reference, approx, expansion, x_title, y_title):
    """
    The formate of the plots for the comparation between the reference value, approximation value
    and the expansion values.

    Parameters
    ----------
    kappa: np.ndarray
        Strike prices.
    maturities: 

    reference: np.ndarray
            Reference values.
    approx: np.ndarray
            Approxiamtion values.
    expansion: np.ndarray
            Expansion values.
    x_title: str
            The title of the x-axis.  
    y_title: str
            The title of the y-axis.   
    """
    COLORS = ["blue", "green", "red"]

    rel_error_approx = {
        maturity: signed_rel_error(reference[maturity], approx[maturity]) for maturity in maturities.keys()
    }
    rel_error_expan = {
        maturity: signed_rel_error(reference[maturity], expansion[maturity]) for maturity in maturities.keys()
    }

    fig, axs = plt.subplots(2, 2, figsize = (10, 8))
    # fig, axs = plt.subplots(2, 2, figsize = (14, 14))

    for i, maturity in enumerate(maturities.keys()):
        axs[0, 0].plot(
            kappa, reference[maturity], "x", color=COLORS[i], label=f"ref({maturity})"
        )
        axs[0, 0].plot(
            kappa, approx[maturity], "--", color=COLORS[i], label=f"approx({maturity})",
        )
        axs[0, 1].plot(
            kappa, reference[maturity], "x", color=COLORS[i], label=f"ref({maturity})"
        )
        axs[0, 1].plot(
            kappa, expansion[maturity], "--", color=COLORS[i], label=f"expan({maturity})",
        )
        axs[1, 0].plot(
            kappa, rel_error_approx[maturity], "+--", color=COLORS[i], label=f"{maturity}"
        )
        axs[1, 1].plot(
            kappa, rel_error_expan[maturity], "+--", color=COLORS[i], label=f"{maturity}"
        )
    axs[0, 0].set_xlabel(x_title)
    axs[0, 1].set_xlabel(x_title)
    axs[1, 0].set_xlabel(x_title)
    axs[1, 1].set_xlabel(x_title)
    axs[0, 0].set_ylabel(y_title)
    axs[0, 1].set_ylabel(y_title)
    axs[1, 0].set_ylabel("Relative error (%)")
    axs[1, 1].set_ylabel("Relative error (%)")

    axs[0, 0].legend(ncol=3, frameon=True)
    axs[0, 1].legend(ncol=3, frameon=True)
    axs[1, 0].legend(ncol=3, frameon=True)
    axs[1, 1].legend(ncol=3, frameon=True)

    axs[0, 0].grid(True)
    axs[0, 1].grid(True)
    axs[1, 0].grid(True)
    axs[1, 1].grid(True)

    plt.tight_layout()

    plt.show()

def year_fraction(start_dt, end_dt, basis=365.0):
    """
    """
    return (end_dt - start_dt).days / basis