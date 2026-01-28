import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from standard_bergomi import StandardBergomi
from rough_bergomi import RoughBergomi

def calib_one_maturity(T, kappa, fwd_mkt, iv_mkt, params_init, params_num, model="rough"):
    """
    Calibrate Mixed Rough Bergomi model to single-day market vix smile using the root_finding method
    with the Monte Carlo method.

    Parameters
    ----------
    T : float
        Maturity of the VIX options to calibrate to.
    kappa : np.ndarray
        Array of strikes of the VIX options.
    fwd_mkt : float
        Market VIX futures price.
    iv_mkt : np.ndarray
        Market implied volatilities of the VIX options.
    params_init : tuple
        Initial guess for the parameters (eta1, eta2, H, xi0, lbd).
    params_num: int
        Number of parameters to calibrate (4 or 5).
    model: str, optional
        if "rough", use the rough Bergomi model
        if "standard" use the standard Bergomi model
    """

    k = np.log(kappa / fwd_mkt).astype(float)
    iv_mkt = iv_mkt.astype(float)
    fwd_mkt = float(fwd_mkt)

    if params_num == 4:
        vol_of_vol1_init, vol_of_vol2_init, xi0_init, lbd_init = params_init
        if model == "standard":
            rate_init = 1
        else: 
            rate_init = 0.1
    else:
        vol_of_vol1_init, vol_of_vol2_init, rate_init, xi0_init, lbd_init = params_init

    delta_vix = 30 / 365.25

    def solve_xi0(vol_of_vol1, vol_of_vol2, rate, lbd):

        def residual_x(x):

            x = float(np.atleast_1d(x)[0])
            xi0 = float(np.exp(x))

            xi0_func = lambda u: np.ones_like(u) * xi0

            if model == "standard":
                inst = StandardBergomi(vol_of_vol1, rate, xi0_func, delta_vix)
                n_time = 50
                n_space = 50
                fwd_model = inst.vix_opt_price_mixed(0, kappa, T, vol_of_vol2, lbd, opttype=0, n_time=n_time, n_space=n_space)
            else:
                inst = RoughBergomi(vol_of_vol1, rate, xi0_func, delta_vix)
                n_disc = 300
                n_mc = 10**4
                seed = 777
                fwd_model = inst.vix_opt_price_mixed(0, T, vol_of_vol2, lbd, opttype = 0, n_disc=n_disc, n_mc=n_mc, seed=seed)

            return fwd_mkt - fwd_model
        
        try:
            x_guess = np.log(xi0_init)

            x_sol = least_squares(
                residual_x,
                x0=[x_guess]
            )

            if x_sol.success and np.isfinite(x_sol.x[0]):
                return float(np.exp(x_sol.x[0]))
        
        except ValueError:
            return np.nan
        
    def residuals_params(params):

        if params_num == 4:
            vol_of_vol1, vol_of_vol2, lbd = params
            rate = rate_init
        else:
            vol_of_vol1, vol_of_vol2, rate, lbd = params

        current_xi0 = solve_xi0(vol_of_vol1, vol_of_vol2, rate, lbd)
        current_xi0_func = lambda u: np.ones_like(u) * current_xi0

        if model == "standard":
            inst = StandardBergomi(vol_of_vol1, rate, current_xi0_func, delta_vix)
            n_time = 50
            n_space = 50
            iv_model = inst.vix_implied_vol_mixed(k, T, vol_of_vol2, lbd, n_time, n_space)
        else:
            inst = RoughBergomi(vol_of_vol1, rate, current_xi0_func, delta_vix)
            n_disc = 300
            n_mc = 10**4
            seed = 777
            iv_model = inst.vix_implied_vol_mixed(k, T, vol_of_vol2, lbd, n_disc, n_mc, seed)

        return iv_mkt - iv_model

    if params_num == 4:
        lower_bounds = [1e-4, 1e-4, 0.0] 
        upper_bounds = [20.0, 20.0, 1.0]
        bounds_to_use = (lower_bounds, upper_bounds)

        params_sol = least_squares(
            residuals_params,
            x0=[vol_of_vol1_init, vol_of_vol2_init, lbd_init],
            bounds=bounds_to_use,
            method="trf"
        )

        vol_of_vol1_cal, vol_of_vol2_cal, lbd_cal = params_sol.x
        rate_cal = rate_init
    else:
        lower_bounds = [1e-4, 1e-4, 9e-3, 0.0] 
        upper_bounds = [20.0, 20.0, 0.5-1e-3, 1.0]
        bounds_to_use = (lower_bounds, upper_bounds)

        params_sol = least_squares(
            residuals_params,
            x0=[vol_of_vol1_init, vol_of_vol2_init, rate_init, lbd_init],
            bounds=bounds_to_use,
            method="trf"
        )
        
        vol_of_vol1_cal, vol_of_vol2_cal, rate_cal, lbd_cal = params_sol.x

    xi0_cal = solve_xi0(vol_of_vol1_cal, vol_of_vol2_cal, rate_cal, lbd_cal)

    return {
        "T": float(T),
        "vol_of_vol1": vol_of_vol1_cal,
        "vol_of_vol2": vol_of_vol2_cal,
        "rate": rate_cal,
        "xi0": xi0_cal,
        "lbd": lbd_cal,
        "cost": params_sol.cost
        }

def calib_one_maturity_proxy(T, kappa, fwd_mkt, iv_mkt, params_init, params_num, n_gauss, model="rough"):
    """
    Calibrate Mixed Rough Bergomi model to single-day market vix smile using the root_finding method.
    The calibration procedure follows these steps:
    1) For given (eta1, eta2, H, lbd), solve for xi0 such that model VIX futures price matches market VIX futures price.
    2) Using the xi0 obtained in step (1), compute model implied volatilities of VIX options.
    3) Compute the residuals between market and model implied volatilities.

    Parameters
    ----------
    T : float
        Maturity of the VIX options to calibrate to.
    kappa : np.ndarray
        Array of strikes of the VIX options.
    fwd_mkt : float
        Market VIX futures price.
    iv_mkt : np.ndarray
        Market implied volatilities of the VIX options.
    params_init : tuple
        Initial guess for the parameters (eta1, eta2, H, xi0, lbd).
    params_num: int
        Number of parameters to calibrate (4 or 5).
    n_gauss : int
        Number of Gauss-Hermite quadrature points.
    model: str, optional
        if "rough", use the rough Bergomi model
        if "standard" use the standard Bergomi model
    """

    k = np.log(kappa / fwd_mkt).astype(float)
    iv_mkt = iv_mkt.astype(float)
    fwd_mkt = float(fwd_mkt)

    if params_num == 4:
        vol_of_vol1_init, vol_of_vol2_init, xi0_init, lbd_init = params_init
        if model == "standard":
            rate_init = 1
        else: 
            rate_init = 0.1
    else:
        vol_of_vol1_init, vol_of_vol2_init, rate_init, xi0_init, lbd_init = params_init

    delta_vix = 30 / 365.25

    def solve_xi0(vol_of_vol1, vol_of_vol2, rate, lbd):

        def residual_x(x):

            x = float(np.atleast_1d(x)[0])
            xi0 = float(np.exp(x))

            xi0_func = lambda u: np.ones_like(u) * xi0

            if model == "standard":
                inst = StandardBergomi(vol_of_vol1, rate, xi0_func, delta_vix)
                fwd_model = inst.vix_opt_price_proxy_mixed(0, T, vol_of_vol2, lbd, opttype=0, n_gauss=n_gauss)
            else:
                inst = RoughBergomi(vol_of_vol1, rate, xi0_func, delta_vix)
                fwd_model = inst.vix_opt_price_proxy_mixed_cal(0, T, vol_of_vol2, lbd, opttype = 0, n_gauss=n_gauss)

            return fwd_mkt - fwd_model
        
        try:
            x_guess = np.log(xi0_init)

            x_sol = least_squares(
                residual_x,
                x0=[x_guess]
            )

            if x_sol.success and np.isfinite(x_sol.x[0]):
                return float(np.exp(x_sol.x[0]))
        
        except ValueError:
            return np.nan
        
    def residuals_params(params):

        if params_num == 4:
            vol_of_vol1, vol_of_vol2, lbd = params
            rate = rate_init
        else:
            vol_of_vol1, vol_of_vol2, rate, lbd = params

        current_xi0 = solve_xi0(vol_of_vol1, vol_of_vol2, rate, lbd)

        current_xi0_func = lambda u: np.ones_like(u) * current_xi0

        if model == "standard":
            inst = StandardBergomi(vol_of_vol1, rate, current_xi0_func, delta_vix)
            iv_model = inst.implied_vol_proxy_mixed(k, T, vol_of_vol2, lbd, n_gauss)
        else:
            inst = RoughBergomi(vol_of_vol1, rate, current_xi0_func, delta_vix)
            iv_model = inst.implied_vol_proxy_mixed_cal(k, T, vol_of_vol2, lbd, n_gauss)

        return iv_mkt - iv_model
    
    if params_num == 4:
        lower_bounds = [1e-4, 1e-4, 0.0] 
        upper_bounds = [20.0, 20.0, 1.0]
        bounds_to_use = (lower_bounds, upper_bounds)

        params_sol = least_squares(
            residuals_params,
            x0=[vol_of_vol1_init, vol_of_vol2_init, lbd_init],
            bounds=bounds_to_use,
            method="trf"
        )

        vol_of_vol1_cal, vol_of_vol2_cal, lbd_cal = params_sol.x
        rate_cal = rate_init
    else:
        lower_bounds = [1e-4, 1e-4, 9e-3, 0.0] 
        upper_bounds = [20.0, 20.0, 0.5-1e-3, 1.0]
        bounds_to_use = (lower_bounds, upper_bounds)

        params_sol = least_squares(
            residuals_params,
            x0=[vol_of_vol1_init, vol_of_vol2_init, rate_init, lbd_init],
            bounds=bounds_to_use,
            method="trf"
        )
        
        vol_of_vol1_cal, vol_of_vol2_cal, rate_cal, lbd_cal = params_sol.x
    
    xi0_cal = solve_xi0(vol_of_vol1_cal, vol_of_vol2_cal, rate_cal, lbd_cal)

    return {
        "T": float(T),
        "vol_of_vol1": vol_of_vol1_cal,
        "vol_of_vol2": vol_of_vol2_cal,
        "rate": rate_cal,
        "xi0": xi0_cal,
        "lbd": lbd_cal,
        "cost": params_sol.cost
        }

def calib_one_maturity_expan(T, kappa, fwd_mkt, iv_mkt, params_init, params_num, n_max, n_mc, formula, model="rough"):
    """
    Calibrate Mixed Rough Bergomi model to single-day market vix smile using the explicit implied volatility expansion.
    The calibration procedure follows these steps:
    1) For given (eta1, eta2, H, lbd), solve for xi0 such that model VIX futures price matches market VIX futures price.
    2) Using the xi0 obtained in step (1), compute model implied volatilities of VIX options.
    3) Compute the residuals between market and model implied volatilities.

    Parameters
    ----------
    T : float
        Maturity of the VIX options to calibrate to.
    kappa : np.ndarray
        Array of strikes of the VIX options.
    fwd_mkt : float
        Market VIX futures price.
    iv_mkt : np.ndarray
        Market implied volatilities of the VIX options.
    params_init : tuple
        Initial guess for the parameters (eta1, eta2, H, xi0, lbd).
    params_num: int
        Number of parameters to calibrate (4 or 5).
    n_max : int
        Maximum order of the Hermite expansion.
    n_mc : int
        Number of Monte Carlo samples for estimating optimal order.
    formula : int
        Formula form for the implied volatility expansion.
    model: str, optional
        if "rough", use the rough Bergomi model
        if "standard" use the standard Bergomi model
    """
    k = np.log(kappa / fwd_mkt).astype(float)
    iv_mkt = iv_mkt.astype(float)
    fwd_mkt = float(fwd_mkt)

    if params_num == 4:
        vol_of_vol1_init, vol_of_vol2_init, xi0_init, lbd_init = params_init
        if model == "standard":
            rate_init = 1
        else: 
            rate_init = 0.1
    else:
        vol_of_vol1_init, vol_of_vol2_init, rate_init, xi0_init, lbd_init = params_init
        
    delta_vix = 30 / 365.25
    rule = -1
    optimal_n = 10

    def solve_xi0(vol_of_vol1, vol_of_vol2, rate, lbd):

        def residual_x(x):

            x = float(np.atleast_1d(x)[0])
            xi0 = float(np.exp(x))

            xi0_func = lambda u: np.ones_like(u) * xi0

            if model == "standard":
                inst = StandardBergomi(vol_of_vol1, rate, xi0_func, delta_vix)
                fwd_model = inst.vix_opt_price_expan_mixed_cal(0, T, rule, vol_of_vol2, lbd, opttype=0, n_max=n_max, n_mc=n_mc, optimal_n=optimal_n)
            else:
                inst = RoughBergomi(vol_of_vol1, rate, xi0_func, delta_vix)
                fwd_model = inst.vix_opt_price_expan_mixed_cal(0, T, rule, vol_of_vol2, lbd, opttype=0, n_max=n_max, n_mc=n_mc, optimal_n=optimal_n)

            return fwd_mkt - fwd_model
        
        try:
            x_guess = np.log(xi0_init)

            x_sol = least_squares(
                residual_x,
                x0=[x_guess]
            )

            if x_sol.success and np.isfinite(x_sol.x[0]):
                return float(np.exp(x_sol.x[0]))

        except ValueError:
            
            return np.nan
        
    def residuals_params(params):

        if params_num == 4:
            vol_of_vol1, vol_of_vol2, lbd = params
            rate = rate_init
        else:
            vol_of_vol1, vol_of_vol2, rate, lbd = params

        current_xi0 = solve_xi0(vol_of_vol1, vol_of_vol2, rate, lbd)
        current_xi0_func = lambda u: np.ones_like(u) * current_xi0

        if model == "standard":
            inst = StandardBergomi(vol_of_vol1, rate, current_xi0_func, delta_vix)
            iv_model, _ = inst.implied_vol_expan_mixed(k, T, rule, vol_of_vol2, lbd, n_max, n_mc, formula, optimal_n)
        else:
            inst = RoughBergomi(vol_of_vol1, rate, current_xi0_func, delta_vix)
            iv_model = inst.implied_vol_expan_mixed_cal(k, T, rule, vol_of_vol2, lbd, n_max, n_mc, formula, optimal_n)

        return iv_mkt - iv_model
    
    if params_num == 4:
        lower_bounds = [1e-4, 1e-4, 0.0] 
        upper_bounds = [20.0, 20.0, 1.0]
        bounds_to_use = (lower_bounds, upper_bounds)

        params_sol = least_squares(
            residuals_params,
            x0=[vol_of_vol1_init, vol_of_vol2_init, lbd_init],
            bounds=bounds_to_use,
            method="trf"
        )

        vol_of_vol1_cal, vol_of_vol2_cal, lbd_cal = params_sol.x
        rate_cal = rate_init
    else:
        lower_bounds = [1e-4, 1e-4, 9e-3, 0.0] 
        upper_bounds = [20.0, 20.0, 0.5-1e-3, 1.0]
        bounds_to_use = (lower_bounds, upper_bounds)

        params_sol = least_squares(
            residuals_params,
            x0=[vol_of_vol1_init, vol_of_vol2_init, rate_init, lbd_init],
            bounds=bounds_to_use,
            method="trf"
        )
        
        vol_of_vol1_cal, vol_of_vol2_cal, rate_cal, lbd_cal = params_sol.x
    
    xi0_cal = solve_xi0(vol_of_vol1_cal, vol_of_vol2_cal, rate_cal, lbd_cal)

    return {
        "T": float(T),
        "vol_of_vol1": vol_of_vol1_cal,
        "vol_of_vol2": vol_of_vol2_cal,
        "rate": rate_cal,
        "xi0": xi0_cal,
        "lbd": lbd_cal,
        "cost": params_sol.cost
        }

def repricing_calibrated_model(market_data, calibrated_params, model="rough"):
    """

    model: str, optional
        if "rough", use the rough Bergomi model
        if "standard" use the standard Bergomi model
    """

    has_bid_ask = 'Bid' in market_data.columns and 'Ask' in market_data.columns

    results = []
    calib_Ts = calibrated_params.index.unique().sort_values()

    for T in calib_Ts:
        
        g = market_data[np.isclose(market_data['Texp'], T, atol=1e-4)]

        if g.empty:
            continue

        market_T = g['Texp'].iloc[0]
        fwd_mkt = g['Fwd'].iloc[0]
        mid_ivs = g['Mid']
        strikes = g['Strike']
        log_strikes = np.log(strikes / fwd_mkt)

        if has_bid_ask:
            bid_ivs = g['Bid']
            ask_ivs = g['Ask']

        N = len(log_strikes)

        sort_indices = np.argsort(log_strikes)

        params_for_T = calibrated_params.loc[T]
        vol_of_vol1 = params_for_T['vol_of_vol1']
        vol_of_vol2 = params_for_T['vol_of_vol2']
        rate = params_for_T['rate']
        xi0 = params_for_T['xi0']
        lbd = params_for_T['lbd']

        xi0_func = lambda u: np.ones_like(u) * xi0

        delta_vix = 30 / 365.25

        if model == "standard":
            inst = StandardBergomi(vol_of_vol1, rate, xi0_func, delta_vix)
            n_time = 80
            n_space = 80
            fwd_model, iv_model = inst.vix_implied_vol_mixed(log_strikes, market_T, vol_of_vol2, lbd, n_time, n_space, return_opt='all')
        else:
            inst = RoughBergomi(vol_of_vol1, rate, xi0_func, delta_vix)
            n_disc = 300
            n_mc = 10**6
            seed = 777
            fwd_model, iv_model = inst.vix_implied_vol_mixed(log_strikes, market_T, vol_of_vol2, lbd, n_disc, n_mc, seed, return_opt='all')

        data_dict = {
            'Texp': [market_T] * N,
            'strikes': strikes.iloc[sort_indices],
            'iv_mkt': mid_ivs.iloc[sort_indices],
            'iv_cal': iv_model,
            'fwd_mkt': [fwd_mkt] * N,
            'fwd_cal': [fwd_model] * N
        }

        if has_bid_ask:
            data_dict['bid_iv_mkt'] = bid_ivs.iloc[sort_indices]
            data_dict['ask_iv_mkt'] = ask_ivs.iloc[sort_indices]

        temp_df = pd.DataFrame(data_dict)

        results.append(temp_df)
    
    if not results:
        return pd.DataFrame()

    df_calibrated_results = pd.concat(results)

    return df_calibrated_results

def repricing_calibrated_model_proxy(market_data, calibrated_params, model="rough"):
    """

    model: str, optional
        if "rough", use the rough Bergomi model
        if "standard" use the standard Bergomi model
    """

    has_bid_ask = 'Bid' in market_data.columns and 'Ask' in market_data.columns

    results = []
    calib_Ts = calibrated_params.index.unique().sort_values()

    for T in calib_Ts:
        
        g = market_data[np.isclose(market_data['Texp'], T, atol=1e-4)]

        if g.empty:
            continue

        market_T = g['Texp'].iloc[0]
        fwd_mkt = g['Fwd'].iloc[0]
        mid_ivs = g['Mid']
        strikes = g['Strike']
        log_strikes = np.log(strikes / fwd_mkt)

        if has_bid_ask:
            bid_ivs = g['Bid']
            ask_ivs = g['Ask']

        N = len(log_strikes)

        sort_indices = np.argsort(log_strikes)

        params_for_T = calibrated_params.loc[T]
        vol_of_vol1 = params_for_T['vol_of_vol1']
        vol_of_vol2 = params_for_T['vol_of_vol2']
        rate = params_for_T['rate']
        xi0 = params_for_T['xi0']
        lbd = params_for_T['lbd']

        xi0_func = lambda u: np.ones_like(u) * xi0

        n_gauss = 120
        delta_vix = 30 / 365.25

        if model == "standard":
            inst = StandardBergomi(vol_of_vol1, rate, xi0_func, delta_vix)
            fwd_model, iv_model = inst.implied_vol_proxy_mixed(log_strikes, market_T, vol_of_vol2, lbd, n_gauss, return_opt='all')
        else:
            inst = RoughBergomi(vol_of_vol1, rate, xi0_func, delta_vix)
            fwd_model, iv_model = inst.implied_vol_proxy_mixed(log_strikes, market_T, vol_of_vol2, lbd, n_gauss, return_opt='all')

        data_dict = {
            'Texp': [market_T] * N,
            'strikes': strikes.iloc[sort_indices],
            'iv_mkt': mid_ivs.iloc[sort_indices],
            'iv_cal': iv_model,
            'fwd_mkt': [fwd_mkt] * N,
            'fwd_cal': [fwd_model] * N
        }

        if has_bid_ask:
            data_dict['bid_iv_mkt'] = bid_ivs.iloc[sort_indices]
            data_dict['ask_iv_mkt'] = ask_ivs.iloc[sort_indices]

        temp_df = pd.DataFrame(data_dict)

        results.append(temp_df)
    
    if not results:
        return pd.DataFrame()

    df_calibrated_results = pd.concat(results)

    return df_calibrated_results

def repricing_calibrated_model_expan(market_data, calibrated_params, formula,  model="rough"):
    """

    model: str, optional
        if "rough", use the rough Bergomi model
        if "standard" use the standard Bergomi model
    """

    has_bid_ask = 'Bid' in market_data.columns and 'Ask' in market_data.columns

    results = []
    calib_Ts = calibrated_params.index.unique().sort_values()

    for T in calib_Ts:
        
        g = market_data[np.isclose(market_data['Texp'], T, atol=1e-4)]

        if g.empty:
            continue

        market_T = g['Texp'].iloc[0]
        fwd_mkt = g['Fwd'].iloc[0]
        mid_ivs = g['Mid']
        strikes = g['Strike']
        log_strikes = np.log(strikes / fwd_mkt)

        if has_bid_ask:
            bid_ivs = g['Bid']
            ask_ivs = g['Ask']

        N = len(log_strikes)

        sort_indices = np.argsort(log_strikes)

        params_for_T = calibrated_params.loc[T]
        vol_of_vol1 = params_for_T['vol_of_vol1']
        vol_of_vol2 = params_for_T['vol_of_vol2']
        rate = params_for_T['rate']
        xi0 = params_for_T['xi0']
        lbd = params_for_T['lbd']

        xi0_func = lambda u: np.ones_like(u) * xi0

        n_max = 20
        n_mc = 10**6
        rule = -1
        delta_vix = 30 / 365.25

        if model == "standard":
            inst = StandardBergomi(vol_of_vol1, rate, xi0_func, delta_vix)
            fwd_model, iv_model, _ = inst.implied_vol_expan_mixed(log_strikes, market_T, rule, vol_of_vol2, lbd, n_max, n_mc, formula, return_opt='all')
        else:
            inst = RoughBergomi(vol_of_vol1, rate, xi0_func, delta_vix)
            fwd_model, iv_model, _ = inst.implied_vol_expan_mixed(log_strikes, market_T, rule, vol_of_vol2, lbd, n_max, n_mc, formula, return_opt='all')

        data_dict = {
            'Texp': [market_T] * N,
            'strikes': strikes.iloc[sort_indices],
            'iv_mkt': mid_ivs.iloc[sort_indices],
            'iv_cal': iv_model,
            'fwd_mkt': [fwd_mkt] * N,
            'fwd_cal': [fwd_model] * N
        }

        if has_bid_ask:
            data_dict['bid_iv_mkt'] = bid_ivs.iloc[sort_indices]
            data_dict['ask_iv_mkt'] = ask_ivs.iloc[sort_indices]

        temp_df = pd.DataFrame(data_dict)

        results.append(temp_df)

    df_calibrated_results = pd.concat(results)
    return df_calibrated_results

def calib_global_maturity(T, kappa, fwd_mkt, iv_mkt, params_init, params_num, model="rough"):
    """
    Calibrate Mixed Rough Bergomi model to multiple-days market vix smiles using the root_finding method
    with the Monte-Carlo simulation.

    Parameters
    ----------
    T : list of float
        All maturities [T1, T2, ...].
    kappa : list of np.ndarray
        List of strike arrays for each maturity.
    fwd_mkt : list of float
        Market VIX futures prices for each maturity.
    iv_mkt : list of np.ndarray
        Market IVs for each maturity.
    params_init : tuple
        Initial guess for the parameters 
        [b0, b1, b2, tau1, tau2, vol_of_vol1, vol_of_vol2, rate, lbd].
    params_num: int
        Number of parameters to calibrate (4 or 5).
    model: str, optional
        if "rough", use the rough Bergomi model
        if "standard" use the standard Bergomi model
    """
    if np.any(np.isnan(fwd_mkt)):
        raise ValueError("Input 'fwd_mkt' contains NaNs.")
    iv_mkt_flat = np.concatenate(iv_mkt)
    if np.any(np.isnan(iv_mkt_flat)):
        raise ValueError("Input 'iv_mkt' contains NaNs.")

    if params_num == 4:
        if model == "standard":
            rate_init = 1
        else: 
            rate_init = 0.1

    delta_vix = 30 / 365.25

    def solve_xi0(params_model):

        if params_num == 4:
            vol_of_vol1, vol_of_vol2, lbd = params_model
            rate = rate_init
        else:
            vol_of_vol1, vol_of_vol2, rate, lbd = params_model
        
        def residual_xi0(params):

            b0, b1, b2, tau1, tau2 = params

            if b0 + b1 <= 1e-5:
                return np.ones_like(fwd_mkt) * 100.0
            
            if tau1 < 1e-4 or tau2 < 1e-4:
                return np.ones_like(fwd_mkt) * 100.0

            def current_xi0_func(u):
                return b0 + b1 * np.exp(-u / tau1) + b2 * (u / tau2) * np.exp(-u / tau2)

            fwd_model = []

            try:

                for T_i in T:

                    if model == "standard":
                        inst = StandardBergomi(vol_of_vol1, rate, current_xi0_func, delta_vix)
                        n_time = 50
                        n_space = 50
                        fwd_model_i = inst.vix_opt_price_mixed(0, T_i, vol_of_vol2, lbd, opttype=0, n_time=n_time, n_space=n_space)
                    else:
                        inst = RoughBergomi(vol_of_vol1, rate, current_xi0_func, delta_vix)
                        n_disc = 300
                        n_mc = 10**4
                        seed = 777
                        fwd_model_i = inst.vix_opt_price_mixed(0, T_i, vol_of_vol2, lbd, opttype = 0, n_disc=n_disc, n_mc=n_mc, seed=seed)

                    fwd_model.append(float(fwd_model_i))

                fwd_model = np.asarray(fwd_model)

                if np.any(np.isnan(fwd_model)):
                     return np.ones_like(fwd_mkt) * 100.0
                
                return np.array(fwd_mkt) - fwd_model
            
            except ValueError:
                return np.ones_like(fwd_mkt) * 100.0
            except Exception as e:
                return np.ones_like(fwd_mkt) * 100.0
        
        lower_bounds = [1e-4, -1.0, -1.0, 1e-2, 1e-2] 
        upper_bounds = [1.0, 1.0, 1.0, 20.0, 20.0]
        bounds_to_use = (lower_bounds, upper_bounds)

        xi0_sol = least_squares(
            residual_xi0,
            x0=params_init[:5],
            bounds=bounds_to_use,
            method="trf"
        )
        return xi0_sol.x
        
    def residuals_params(params):

        if params_num == 4:
            vol_of_vol1, vol_of_vol2, lbd = params
            rate = rate_init
        else:
            vol_of_vol1, vol_of_vol2, rate, lbd = params

        b0, b1, b2, tau1, tau2 = solve_xi0(params)

        def current_xi0_func(u):
            return b0 + b1 * np.exp(-u / tau1) + b2 * (u / tau2) * np.exp(-u / tau2)
        
        iv_model = []

        for i, T_i in enumerate(T):

            kappa_i = kappa[i]
            fwd_i = fwd_mkt[i]
            k_i = np.log(kappa_i / fwd_i)

            if model == "standard":
                inst = StandardBergomi(vol_of_vol1, rate, current_xi0_func, delta_vix)
                n_time = 50
                n_space = 50
                iv_model_i = inst.vix_implied_vol_mixed(k_i, T_i, vol_of_vol2, lbd, n_time, n_space)
            else:
                inst = RoughBergomi(vol_of_vol1, rate, current_xi0_func, delta_vix)
                n_disc = 300
                n_mc = 10**4
                seed = 777
                iv_model_i = inst.vix_implied_vol_mixed(k_i, T_i, vol_of_vol2, lbd, n_disc, n_mc, seed)

            iv_model.append(np.asarray(iv_model_i))

        iv_model_flat = np.concatenate(iv_model)

        if np.any(np.isnan(iv_model_flat)):
                return np.ones_like(iv_mkt_flat) * 100.0

        error_iv = iv_mkt_flat - iv_model_flat
        return np.asarray(error_iv)

    if params_num == 4:
        lower_bounds = [1e-4, 1e-4, 0.0] 
        upper_bounds = [20.0, 20.0, 1.0]
        bounds_to_use = (lower_bounds, upper_bounds)
    else:
        lower_bounds = [1e-4, 1e-4, 9e-3, 0.0] 
        upper_bounds = [20.0, 20.0, 0.5-1e-3, 1.0]
        bounds_to_use = (lower_bounds, upper_bounds)

    params_sol = least_squares(
        residuals_params,
        x0=params_init[5:],
        bounds=bounds_to_use,
        method="trf"
    )

    b0_cal, b1_cal, b2_cal, tau1_cal, tau2_cal = solve_xi0(params_sol.x)

    if params_num == 4:
        vol_of_vol1_cal, vol_of_vol2_cal, lbd_cal = params_sol.x
        rate_cal = rate_init
    else:
        vol_of_vol1_cal, vol_of_vol2_cal, rate_cal, lbd_cal = params_sol.x

    return {
        "b0": b0_cal,
        "b1": b1_cal,
        "b2": b2_cal,
        "tau1": tau1_cal,
        "tau2": tau2_cal,
        "vol_of_vol1": vol_of_vol1_cal,
        "vol_of_vol2": vol_of_vol2_cal,
        "rate": rate_cal,
        "lbd": lbd_cal,
        "cost": params_sol.cost
    }

def calib_global_maturity_proxy(T, kappa, fwd_mkt, iv_mkt, params_init, params_num, n_gauss, model="rough"):
    """
    Calibrate Mixed Rough Bergomi model to multiple-days market vix smiles using the root_finding method
    with the Monte-Carlo simulation.

    Parameters
    ----------
    T : list of float
        All maturities [T1, T2, ...].
    kappa : list of np.ndarray
        List of strike arrays for each maturity.
    fwd_mkt : list of float
        Market VIX futures prices for each maturity.
    iv_mkt : list of np.ndarray
        Market IVs for each maturity.
    params_init : tuple
        Initial guess for the parameters 
        [b0, b1, b2, tau1, tau2, vol_of_vol1, vol_of_vol2, rate, lbd].
    params_num: int
        Number of parameters to calibrate (4 or 5).
    n_gauss: int
        Number of nodes for the one-dimensional Gauss-Hermite quadrature.
    model: str, optional
        if "rough", use the rough Bergomi model
        if "standard" use the standard Bergomi model
    """
    if np.any(np.isnan(fwd_mkt)):
        raise ValueError("Input 'fwd_mkt' contains NaNs.")
    iv_mkt_flat = np.concatenate(iv_mkt)
    if np.any(np.isnan(iv_mkt_flat)):
        raise ValueError("Input 'iv_mkt' contains NaNs.")

    if params_num == 4:
        if model == "standard":
            rate_init = 1
        else: 
            rate_init = 0.1

    delta_vix = 30 / 365.25

    def solve_xi0(params_model):

        if params_num == 4:
            vol_of_vol1, vol_of_vol2, lbd = params_model
            rate = rate_init
        else:
            vol_of_vol1, vol_of_vol2, rate, lbd = params_model
        
        def residual_xi0(params):

            b0, b1, b2, tau1, tau2 = params

            if b0 + b1 <= 1e-5:
                return np.ones_like(fwd_mkt) * 100.0
            
            if tau1 < 1e-4 or tau2 < 1e-4:
                return np.ones_like(fwd_mkt) * 100.0

            def current_xi0_func(u):
                return b0 + b1 * np.exp(-u / tau1) + b2 * (u / tau2) * np.exp(-u / tau2)

            fwd_model = []

            try:

                for T_i in T:

                    if model == "standard":
                        inst = StandardBergomi(vol_of_vol1, rate, current_xi0_func, delta_vix)
                        fwd_model_i = inst.vix_opt_price_proxy_mixed(0, T_i, vol_of_vol2, lbd, opttype=0, n_gauss=n_gauss)
                    else:
                        inst = RoughBergomi(vol_of_vol1, rate, current_xi0_func, delta_vix)
                        fwd_model_i = inst.vix_opt_price_proxy_mixed_cal(0, T_i, vol_of_vol2, lbd, opttype=0, n_gauss=n_gauss)

                    fwd_model.append(float(fwd_model_i))

                fwd_model = np.asarray(fwd_model)

                if np.any(np.isnan(fwd_model)):
                     return np.ones_like(fwd_mkt) * 100.0
                
                return np.array(fwd_mkt) - fwd_model
            
            except ValueError:
                return np.ones_like(fwd_mkt) * 100.0
            except Exception as e:
                return np.ones_like(fwd_mkt) * 100.0
        
        lower_bounds = [1e-4, -1.0, -1.0, 1e-2, 1e-2] 
        upper_bounds = [1.0, 1.0, 1.0, 20.0, 20.0]
        bounds_to_use = (lower_bounds, upper_bounds)

        xi0_sol = least_squares(
            residual_xi0,
            x0=params_init[:5],
            bounds=bounds_to_use,
            method="trf"
        )
        return xi0_sol.x
        
    def residuals_params(params):

        if params_num == 4:
            vol_of_vol1, vol_of_vol2, lbd = params
            rate = rate_init
        else:
            vol_of_vol1, vol_of_vol2, rate, lbd = params

        b0, b1, b2, tau1, tau2 = solve_xi0(params)

        def current_xi0_func(u):
            return b0 + b1 * np.exp(-u / tau1) + b2 * (u / tau2) * np.exp(-u / tau2)
        
        iv_model = []

        for i, T_i in enumerate(T):

            kappa_i = kappa[i]
            fwd_i = fwd_mkt[i]
            k_i = np.log(kappa_i / fwd_i)

            if model == "standard":
                inst = StandardBergomi(vol_of_vol1, rate, current_xi0_func, delta_vix)
                iv_model_i = inst.implied_vol_proxy_mixed(k_i, T_i, vol_of_vol2, lbd, n_gauss)
            else:
                inst = RoughBergomi(vol_of_vol1, rate, current_xi0_func, delta_vix)
                iv_model_i = inst.implied_vol_proxy_mixed_cal(k_i, T_i, vol_of_vol2, lbd, n_gauss)

            iv_model.append(np.asarray(iv_model_i))

        iv_model_flat = np.concatenate(iv_model)

        if np.any(np.isnan(iv_model_flat)):
                return np.ones_like(iv_mkt_flat) * 100.0

        error_iv = iv_mkt_flat - iv_model_flat
        return np.asarray(error_iv)

    if params_num == 4:
        lower_bounds = [1e-4, 1e-4, 0.0] 
        upper_bounds = [20.0, 20.0, 1.0]
        bounds_to_use = (lower_bounds, upper_bounds)
    else:
        lower_bounds = [1e-4, 1e-4, 9e-3, 0.0] 
        upper_bounds = [20.0, 20.0, 0.5-1e-3, 1.0]
        bounds_to_use = (lower_bounds, upper_bounds)

    params_sol = least_squares(
        residuals_params,
        x0=params_init[5:],
        bounds=bounds_to_use,
        method="trf"
    )

    b0_cal, b1_cal, b2_cal, tau1_cal, tau2_cal = solve_xi0(params_sol.x)

    if params_num == 4:
        vol_of_vol1_cal, vol_of_vol2_cal, lbd_cal = params_sol.x
        rate_cal = rate_init
    else:
        vol_of_vol1_cal, vol_of_vol2_cal, rate_cal, lbd_cal = params_sol.x

    return {
        "b0": b0_cal,
        "b1": b1_cal,
        "b2": b2_cal,
        "tau1": tau1_cal,
        "tau2": tau2_cal,
        "vol_of_vol1": vol_of_vol1_cal,
        "vol_of_vol2": vol_of_vol2_cal,
        "rate": rate_cal,
        "lbd": lbd_cal,
        "cost": params_sol.cost
    }

def calib_global_maturity_expan(T, kappa, fwd_mkt, iv_mkt, params_init, params_num, n_max, n_mc, formula, model="rough"):
    """
    Calibrate Mixed Rough Bergomi model to multiple-days market vix smiles using the explicit Hermite expansion.

    Parameters
    ----------
    T : list of float
        All maturities [T1, T2, ...].
    kappa : list of np.ndarray
        List of strike arrays for each maturity.
    fwd_mkt : list of float
        Market VIX futures prices for each maturity.
    iv_mkt : list of np.ndarray
        Market IVs for each maturity.
    params_init : tuple
        Initial guess for the parameters 
        [b0, b1, b2, tau1, tau2, vol_of_vol1, vol_of_vol2, rate, lbd].
    params_num: int
        Number of parameters to calibrate (4 or 5).
    n_max: int
            The maximum order of the Hermite expansion.
    n_mc : int
        Number of Monte Carlo paths.
    model: str, optional
        if "rough", use the rough Bergomi model
        if "standard" use the standard Bergomi model
    """
    if np.any(np.isnan(fwd_mkt)):
        raise ValueError("Input 'fwd_mkt' contains NaNs.")
    iv_mkt_flat = np.concatenate(iv_mkt)
    if np.any(np.isnan(iv_mkt_flat)):
        raise ValueError("Input 'iv_mkt' contains NaNs.")

    if params_num == 4:
        if model == "standard":
            rate_init = 1
        else: 
            rate_init = 0.1

    rule = -1
    optimal_n = 7
    delta_vix = 30 / 365.25

    def solve_xi0(params_model):

        if params_num == 4:
            vol_of_vol1, vol_of_vol2, lbd = params_model
            rate = rate_init
        else:
            vol_of_vol1, vol_of_vol2, rate, lbd = params_model
        
        def residual_xi0(params):

            b0, b1, b2, tau1, tau2 = params

            if b0 + b1 <= 1e-5:
                return np.ones_like(fwd_mkt) * 100.0
            
            if tau1 < 1e-4 or tau2 < 1e-4:
                return np.ones_like(fwd_mkt) * 100.0

            def current_xi0_func(u):
                return b0 + b1 * np.exp(-u / tau1) + b2 * (u / tau2) * np.exp(-u / tau2)

            fwd_model = []

            try:

                for T_i in T:

                    if model == "standard":
                        inst = StandardBergomi(vol_of_vol1, rate, current_xi0_func, delta_vix)
                        fwd_model_i = inst.vix_opt_price_expan_mixed_cal(0, T_i, rule, vol_of_vol2, lbd, opttype=0, n_max=n_max, n_mc=n_mc, optimal_n=0)
                    else:
                        inst = RoughBergomi(vol_of_vol1, rate, current_xi0_func, delta_vix)
                        fwd_model_i = inst.vix_opt_price_expan_mixed_cal(0, T_i, rule, vol_of_vol2, lbd, opttype=0, n_max=n_max, n_mc=n_mc, optimal_n=0)
                    
                    fwd_model.append(float(fwd_model_i))

                fwd_model = np.asarray(fwd_model)
                    

                if np.any(np.isnan(fwd_model)):
                     return np.ones_like(fwd_mkt) * 100.0
                
                return np.array(fwd_mkt) - fwd_model
            
            except ValueError:
                return np.ones_like(fwd_mkt) * 100.0
            except Exception as e:
                return np.ones_like(fwd_mkt) * 100.0
        
        lower_bounds = [1e-4, -1.0, -1.0, 1e-2, 1e-2] 
        upper_bounds = [1.0, 1.0, 1.0, 20.0, 20.0]
        bounds_to_use = (lower_bounds, upper_bounds)

        xi0_sol = least_squares(
            residual_xi0,
            x0=params_init[:5],
            bounds=bounds_to_use,
            method="trf"
        )
        return xi0_sol.x
        
    def residuals_params(params):

        if params_num == 4:
            vol_of_vol1, vol_of_vol2, lbd = params
            rate = rate_init
        else:
            vol_of_vol1, vol_of_vol2, rate, lbd = params

        b0, b1, b2, tau1, tau2 = solve_xi0(params)

        def current_xi0_func(u):
            return b0 + b1 * np.exp(-u / tau1) + b2 * (u / tau2) * np.exp(-u / tau2)
        
        iv_model = []

        for i, T_i in enumerate(T):

            kappa_i = kappa[i]
            fwd_i = fwd_mkt[i]
            k_i = np.log(kappa_i / fwd_i)

            if model == "standard":
                inst = StandardBergomi(vol_of_vol1, rate, current_xi0_func, delta_vix)
                iv_model_i = inst.implied_vol_expan_mixed_cal(k_i, T_i, rule, vol_of_vol2, lbd, n_max, n_mc, formula, optimal_n)
            else:
                inst = RoughBergomi(vol_of_vol1, rate, current_xi0_func, delta_vix)
                iv_model_i = inst.implied_vol_expan_mixed_cal(k_i, T_i, rule, vol_of_vol2, lbd, n_max, n_mc, formula, optimal_n)

            iv_model.append(np.asarray(iv_model_i))

        iv_model_flat = np.concatenate(iv_model)
        if np.any(np.isnan(iv_model_flat)):
            return np.ones_like(iv_mkt_flat) * 100.0

        error_iv = iv_mkt_flat - iv_model_flat
        return np.asarray(error_iv)

    if params_num == 4:
        lower_bounds = [1e-4, 1e-4, 0.0] 
        upper_bounds = [20.0, 20.0, 1.0]
        bounds_to_use = (lower_bounds, upper_bounds)
    else:
        lower_bounds = [1e-4, 1e-4, 9e-3, 0.0] 
        upper_bounds = [20.0, 20.0, 0.5-1e-3, 1.0]
        bounds_to_use = (lower_bounds, upper_bounds)

    params_sol = least_squares(
        residuals_params,
        x0=params_init[5:],
        bounds=bounds_to_use,
        method="trf"
    )

    b0_cal, b1_cal, b2_cal, tau1_cal, tau2_cal = solve_xi0(params_sol.x)

    if params_num == 4:
        vol_of_vol1_cal, vol_of_vol2_cal, lbd_cal = params_sol.x
        rate_cal = rate_init
    else:
        vol_of_vol1_cal, vol_of_vol2_cal, rate_cal, lbd_cal = params_sol.x

    return {
        "b0": b0_cal,
        "b1": b1_cal,
        "b2": b2_cal,
        "tau1": tau1_cal,
        "tau2": tau2_cal,
        "vol_of_vol1": vol_of_vol1_cal,
        "vol_of_vol2": vol_of_vol2_cal,
        "rate": rate_cal,
        "lbd": lbd_cal,
        "cost": params_sol.cost
    }

def repricing_calibrated_model_global(market_data, calibrated_params, model="rough"):
    """
    Reprice the calibrated model globally using the root finding method with the Monte Carlo simulation.

    Parameters
    ----------
    market_data : pd.DataFrame
        DataFrame containing market data with columns:
        'Texp', 'Fwd', 'Strike', 'Mid', optionally 'Bid' and 'Ask'.
    calibrated_params : pd.DataFrame
        DataFrame containing calibrated parameters for each maturity.
    model: str, optional
        if "rough", use the rough Bergomi model
        if "standard" use the standard Bergomi model
    """
    if isinstance(calibrated_params, pd.DataFrame):
        params = calibrated_params.iloc[0]
    else:
        params = calibrated_params

    b0 = params['b0']
    b1 = params['b1']
    b2 = params['b2']
    tau1 = params['tau1']
    tau2 = params['tau2']
    vol_of_vol1 = params['vol_of_vol1']
    vol_of_vol2 = params['vol_of_vol2']
    rate = params['rate']
    lbd = params['lbd']

    def xi0_func(u):
        return b0 + b1 * np.exp(-u / tau1) + b2 * (u / tau2) * np.exp(-u / tau2)

    delta_vix = 30 / 365.25
    has_bid_ask = 'Bid' in market_data.columns and 'Ask' in market_data.columns

    results = []

    for T, g in market_data.groupby('Texp'):

        fwd_mkt = g['Fwd'].iloc[0]
        mid_ivs = g['Mid']
        strikes = g['Strike']
        log_strikes = np.log(strikes / fwd_mkt)

        if has_bid_ask:
            bid_ivs = g['Bid']
            ask_ivs = g['Ask']

        N = len(log_strikes)
        sort_indices = np.argsort(log_strikes)

        if model == "standard":
            inst = StandardBergomi(vol_of_vol1, rate, xi0_func, delta_vix)
            n_time = 80
            n_space = 80
            fwd_model, iv_model = inst.vix_implied_vol_mixed(log_strikes, T, vol_of_vol2, lbd, n_time, n_space, return_opt='all')
        else:
            inst = RoughBergomi(vol_of_vol1, rate, xi0_func, delta_vix)
            n_disc = 300
            n_mc = 10**6
            seed = 777
            fwd_model, iv_model = inst.vix_implied_vol_mixed(log_strikes, T, vol_of_vol2, lbd, n_disc, n_mc, seed, return_opt='all')

        data_dict = {
            'Texp': [T] * N,
            'strikes': strikes.iloc[sort_indices],
            'iv_mkt': mid_ivs.iloc[sort_indices],
            'iv_cal': iv_model,
            'fwd_mkt': [fwd_mkt] * N,
            'fwd_cal': [fwd_model] * N
        }

        if has_bid_ask:
            data_dict['bid_iv_mkt'] = bid_ivs.iloc[sort_indices]
            data_dict['ask_iv_mkt'] = ask_ivs.iloc[sort_indices]

        temp_df = pd.DataFrame(data_dict)

        results.append(temp_df)
    
    if not results:
        return pd.DataFrame()

    df_calibrated_results = pd.concat(results)

    return df_calibrated_results

def repricing_calibrated_model_global_proxy(market_data, calibrated_params, model="rough"):
    """
    Reprice the calibrated model globally using the root finding method with the Monte Carlo simulation.

    Parameters
    ----------
    market_data : pd.DataFrame
        DataFrame containing market data with columns:
        'Texp', 'Fwd', 'Strike', 'Mid', optionally 'Bid' and 'Ask'.
    calibrated_params : pd.DataFrame
        DataFrame containing calibrated parameters for each maturity.
    model: str, optional
        if "rough", use the rough Bergomi model
        if "standard" use the standard Bergomi model
    """

    if isinstance(calibrated_params, pd.DataFrame):
        params = calibrated_params.iloc[0]
    else:
        params = calibrated_params

    b0 = params['b0']
    b1 = params['b1']
    b2 = params['b2']
    tau1 = params['tau1']
    tau2 = params['tau2']
    vol_of_vol1 = params['vol_of_vol1']
    vol_of_vol2 = params['vol_of_vol2']
    rate = params['rate']
    lbd = params['lbd']

    def xi0_func(u):
        return b0 + b1 * np.exp(-u / tau1) + b2 * (u / tau2) * np.exp(-u / tau2)

    n_gauss = 120
    delta_vix = 30 / 365.25
    has_bid_ask = 'Bid' in market_data.columns and 'Ask' in market_data.columns

    results = []

    for T, g in market_data.groupby('Texp'):

        fwd_mkt = g['Fwd'].iloc[0]
        mid_ivs = g['Mid']
        strikes = g['Strike']
        log_strikes = np.log(strikes / fwd_mkt)

        if has_bid_ask:
            bid_ivs = g['Bid']
            ask_ivs = g['Ask']

        N = len(log_strikes)
        sort_indices = np.argsort(log_strikes)

        if model == "standard":
            inst = StandardBergomi(vol_of_vol1, rate, xi0_func, delta_vix)
            fwd_model, iv_model = inst.implied_vol_proxy_mixed(log_strikes, T, vol_of_vol2, lbd, n_gauss, return_opt='all')
        else:
            inst = RoughBergomi(vol_of_vol1, rate, xi0_func, delta_vix)
            fwd_model, iv_model = inst.implied_vol_proxy_mixed(log_strikes, T, vol_of_vol2, lbd, n_gauss, return_opt='all')

        data_dict = {
            'Texp': [T] * N,
            'strikes': strikes.iloc[sort_indices],
            'iv_mkt': mid_ivs.iloc[sort_indices],
            'iv_cal': iv_model,
            'fwd_mkt': [fwd_mkt] * N,
            'fwd_cal': [fwd_model] * N
        }

        if has_bid_ask:
            data_dict['bid_iv_mkt'] = bid_ivs.iloc[sort_indices]
            data_dict['ask_iv_mkt'] = ask_ivs.iloc[sort_indices]

        temp_df = pd.DataFrame(data_dict)

        results.append(temp_df)
    
    if not results:
        return pd.DataFrame()

    df_calibrated_results = pd.concat(results)

    return df_calibrated_results

def repricing_calibrated_model_global_expan(market_data, calibrated_params, formula, model="rough"):
    """
    Reprice the calibrated model globally using the root finding method with the Monte Carlo simulation.

    Parameters
    ----------
    market_data : pd.DataFrame
        DataFrame containing market data with columns:
        'Texp', 'Fwd', 'Strike', 'Mid', optionally 'Bid' and 'Ask'.
    calibrated_params : pd.DataFrame
        DataFrame containing calibrated parameters for each maturity.
    model: str, optional
        if "rough", use the rough Bergomi model
        if "standard" use the standard Bergomi model
    """
    if isinstance(calibrated_params, pd.DataFrame):
        params = calibrated_params.iloc[0]
    else:
        params = calibrated_params

    b0 = params['b0']
    b1 = params['b1']
    b2 = params['b2']
    tau1 = params['tau1']
    tau2 = params['tau2']
    vol_of_vol1 = params['vol_of_vol1']
    vol_of_vol2 = params['vol_of_vol2']
    rate = params['rate']
    lbd = params['lbd']

    def xi0_func(u):
        return b0 + b1 * np.exp(-u / tau1) + b2 * (u / tau2) * np.exp(-u / tau2)

    n_max = 20
    n_mc = 10**6
    rule = -1
    delta_vix = 30 / 365.25
    has_bid_ask = 'Bid' in market_data.columns and 'Ask' in market_data.columns

    results = []

    for T, g in market_data.groupby('Texp'):

        fwd_mkt = g['Fwd'].iloc[0]
        mid_ivs = g['Mid']
        strikes = g['Strike']
        log_strikes = np.log(strikes / fwd_mkt)

        if has_bid_ask:
            bid_ivs = g['Bid']
            ask_ivs = g['Ask']

        N = len(log_strikes)
        sort_indices = np.argsort(log_strikes)

        if model == "standard":
            inst = StandardBergomi(vol_of_vol1, rate, xi0_func, delta_vix)
            fwd_model, iv_model, _ = inst.implied_vol_expan_mixed(log_strikes, T, rule, vol_of_vol2, lbd, n_max, n_mc, formula, return_opt='all')
        else:
            inst = RoughBergomi(vol_of_vol1, rate, xi0_func, delta_vix)
            fwd_model, iv_model, _ = inst.implied_vol_expan_mixed(log_strikes, T, rule, vol_of_vol2, lbd, n_max, n_mc, formula, return_opt='all')

        data_dict = {
            'Texp': [T] * N,
            'strikes': strikes.iloc[sort_indices],
            'iv_mkt': mid_ivs.iloc[sort_indices],
            'iv_cal': iv_model,
            'fwd_mkt': [fwd_mkt] * N,
            'fwd_cal': [fwd_model] * N
        }

        if has_bid_ask:
            data_dict['bid_iv_mkt'] = bid_ivs.iloc[sort_indices]
            data_dict['ask_iv_mkt'] = ask_ivs.iloc[sort_indices]

        temp_df = pd.DataFrame(data_dict)

        results.append(temp_df)
    
    if not results:
        return pd.DataFrame()

    df_calibrated_results = pd.concat(results)

    return df_calibrated_results