"""
Utility functions for option pricing and implied volatility.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize
from scipy.stats import norm


def relative_abs_error(approx, reference):
    """
    Compute the relative error (absolute) between approximation and reference values.

    Parameters
    ----------
    approx : np.ndarray
        Array of approximate values.
    reference : np.ndarray
        Array of reference values.

    Returns
    -------
    error : np.ndarray
        Array of relative errors (in percent).
    """
    return np.abs(relative_error(approx, reference))


def relative_error(approx, reference):
    """
    Compute the signed relative error between approximation and reference values.

    Parameters
    ----------
    approx : np.ndarray
        Array of approximate values.
    reference : np.ndarray
        Array of reference values.

    Returns
    -------
    error : np.ndarray
        Array of signed relative errors (in percent).
    """
    error = (approx - reference) / reference * 100
    # error_per = np.char.add(np.char.mod("%0.7f", error_per0), "%")
    return error


def plot_approx(approx, reference, kappa_list, type):
    """
    Plot the Hermite approximation results alongside reference values and their
    relative errors.

    Parameters
    ----------
    approx : np.ndarray
        Array of results computed by Hermite approximation.
        Shape: (n_periods, n_strikes).
    reference : np.ndarray
        Array of reference values. Shape: (n_periods, n_strikes).
    kappa_list : list or np.ndarray
        List of strike prices.
    type : str
        Type of approximation (e.g., 'call price', 'put price', 'implied volatility').

    Returns
    -------
    None
        Displays the plot with approximation, reference, and relative error curves.
    """

    rel_errors = np.zeros_like(approx)
    rel_errors[0] = relative_error(approx[0], reference[0])
    rel_errors[1] = relative_error(approx[1], reference[1])
    rel_errors[2] = relative_error(approx[2], reference[2])

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    #
    axs[0].plot(kappa_list, reference[0], "b.", label="ref. 1M")
    axs[0].plot(kappa_list, reference[1], "g+", label="ref. 3M")
    axs[0].plot(kappa_list, reference[2], "rx", label="ref. 6M")

    axs[0].plot(kappa_list, approx[0], "b:", label="approx. 1M")
    axs[0].plot(kappa_list, approx[1], "g--", label="approx. 3M")
    axs[0].plot(kappa_list, approx[2], "r-.", label="approx. 6M")

    axs[0].set_xlabel("Strike price")
    axs[0].set_ylabel(type)

    axs[0].legend(
        # loc='upper right',
        ncol=2,
        frameon=True,
    )
    axs[0].grid(True)

    axs[1].plot(kappa_list, rel_errors[0], "b.:", label="1M")
    axs[1].plot(kappa_list, rel_errors[1], "g+--", label="3M")
    axs[1].plot(kappa_list, rel_errors[2], "rx-.", label="6M")
    axs[1].set_xlabel("Strike price")
    axs[1].set_ylabel("Relative error (%)")
    axs[1].legend(
        # loc='upper right',
        ncol=3,
        frameon=True,
    )
    axs[1].grid(True)

    plt.tight_layout()

    # Save the plot as a file
    # output_file = "vix_implied_vol_plot with different strike price_b.png"
    # plt.savefig(output_file, dpi=300)
    plt.show()


def print_approx(approx, reference, kappa_list):
    """
    Print the results of Hermite approximations and their relative errors.

    Parameters
    ----------
    approx : np.ndarray
        Array of results computed by Hermite approximation.
        Shape: (n_periods, n_strikes).
    reference : np.ndarray
        Array of reference values. Shape: (n_periods, n_strikes).
    kappa_list : list or np.ndarray
        List of strike prices.

    Returns
    -------
    None
        Prints a DataFrame with strike prices, reference values, approximations, and
        relative errors.
    """

    rel_errors = np.zeros_like(approx)

    rel_errors[0] = relative_error(approx[0], reference[0])
    rel_errors[1] = relative_error(approx[1], reference[1])
    rel_errors[2] = relative_error(approx[2], reference[2])

    for i in range(len(rel_errors)):
        results = {
            "Strike price": kappa_list,
            "Quadrature": reference[i],
            "Hermite": approx[i],
            "Rel. error": rel_errors[i],
        }
        results = pd.DataFrame(results)
        results.index = pd.Index(range(1, len(results) + 1))

        pd.set_option("display.max_rows", None)
        pd.set_option("display.max_columns", None)

        print("Numerical results: \n", results)


# Implied volatility (root-finding method)
def implied_vol(x, y, T, market):
    # Black-Scholes formula for VIX options
    def objective(z):
        d1 = np.log(x / y) / (z * np.sqrt(T)) + z * np.sqrt(T) / 2
        d2 = d1 - z * np.sqrt(T)
        bs_call = x * norm.cdf(d1) - y * norm.cdf(d2)
        # bs_call = BS_formula(x, y, z).call()
        return bs_call - market

    # Brent's method to solve for the implied volatility
    try:
        implied_vol = optimize.brentq(
            objective, 1e-8, 5.0
        )  # Bounds for volatility (0 to 500%)
        return implied_vol
    except ValueError:
        return np.nan
