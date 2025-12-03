import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares, brentq

import utils
from mixed_bergomi import MixedRoughBergomi, MixedStandardBergomi
from hermite_expansion import OptimalOrder, HermiteExpansion

def calib_one_maturity_proxy(T, kappa, fwd_mkt, iv_mkt, params_init, H, n_gauss):

    k = np.log(kappa / fwd_mkt).astype(float)
    iv_mkt = iv_mkt.astype(float)

    eta1_init, eta2_init, xi0_init, lbd_init = params_init

    def solve_xi0(eta1, eta2, lbd):

        def residual_x(x):

            x = float(np.atleast_1d(x)[0])
            xi0 = float(np.exp(x))

            mrb = MixedRoughBergomi(eta=[eta1, eta2], H=H, xi0=xi0, lbd=lbd)

            fwd_model = mrb.vix_price_proxy(kappa=0, T=T, opttype=0, n_gauss=n_gauss)

            return fwd_mkt - fwd_model
        
        try:
            x_sol = brentq(residual_x, -10.0, 10.0, xtol=1e-8)
            return float(np.exp(x_sol))
        
        except ValueError:
            x_guess = np.log(xi0_init)

            x_sol = least_squares(
                residual_x,
                x0=[x_guess]
            )

            if x_sol.success and np.isfinite(x_sol.x[0]):
                return float(np.exp(x_sol.x[0]))
            
            return np.nan
        
    xi0_cal0 = solve_xi0(eta1=eta1_init, eta2=eta2_init, lbd=lbd_init)

    def residuals_params(params):

        eta1, eta2, lbd = params

        mrb = MixedRoughBergomi(eta=[eta1, eta2], H=H, xi0=xi0_cal0, lbd=lbd)

        iv_model = mrb.implied_vol_proxy_cal(k=k, T=T, F_mkt=fwd_mkt, n_gauss=n_gauss)

        return iv_mkt - iv_model
    
    lower_bounds = [0.0, 0.0, 0.0]
    upper_bounds = [np.inf, np.inf, 1.0]
    bounds_to_use = (lower_bounds, upper_bounds)
    
    params_sol = least_squares(
        residuals_params,
        x0=[eta1_init, eta2_init, lbd_init],
        bounds=bounds_to_use,
        method="trf"
    )

    eta1_cal, eta2_cal, lbd_cal = params_sol.x
    xi0_cal = solve_xi0(eta1_cal, eta2_cal, lbd_cal)

    return {
        "T": float(T), "eta1": eta1_cal, "eta2": eta2_cal,
        "H": H, "xi0": xi0_cal, "lbd": lbd_cal, "cost": params_sol.cost
        }

def calib_one_maturity_expan(T, kappa, fwd_mkt, iv_mkt, params_init, H, n_max, n_mc, method):

    k = np.log(kappa / fwd_mkt).astype(float)
    iv_mkt = iv_mkt.astype(float)

    eta1_init, eta2_init, xi0_init, lbd_init = params_init

    def solve_xi0(eta1, eta2, lbd):

        def residual_x(x):

            x = float(np.atleast_1d(x)[0])
            xi0 = float(np.exp(x))

            mrb = MixedRoughBergomi(eta=[eta1, eta2], H=H, xi0=xi0, lbd=lbd)

            optimal_order = OptimalOrder(mrb, T, rule=-1, n_max=n_max, n_mc=n_mc).optimal_order()
            expan_inst = HermiteExpansion(model=mrb, T=T, rule=-1, optimal_order=optimal_order)

            fwd_model = expan_inst.vix_option_price_expan(kappa=0, opttype=0)

            return fwd_mkt - fwd_model
        
        try:
            x_sol = brentq(residual_x, -10.0, 10.0, xtol=1e-8)
            return float(np.exp(x_sol))
        
        except ValueError:
            x_guess = np.log(xi0_init)

            x_sol = least_squares(
                residual_x,
                x0=[x_guess]
            )

            if x_sol.success and np.isfinite(x_sol.x[0]):
                return float(np.exp(x_sol.x[0]))
            
            return np.nan
        
    xi0_cal0 = solve_xi0(eta1=eta1_init, eta2=eta2_init, lbd=lbd_init)

    def residuals_params(params):
        eta1, eta2, lbd = params

        mrb = MixedRoughBergomi(eta=[eta1, eta2], H=H, xi0=xi0_cal0, lbd=lbd)
        optimal_order = OptimalOrder(mrb, T, rule=-1, n_max=n_max, n_mc=n_mc).optimal_order()
        expan_inst = HermiteExpansion(model=mrb, T=T, rule=-1, optimal_order=optimal_order)

        if method == 1:
            iv_model = expan_inst.implied_vol_expan(k=k, method=method)[0]
        else:
            iv_model = expan_inst.implied_vol_expan_cal(k=k, F_mkt=fwd_mkt, method=method)[0]

        return iv_mkt - iv_model
    
    lower_bounds = [0.0, 0.0, 0.0]
    upper_bounds = [np.inf, np.inf, 1.0]
    bounds_to_use = (lower_bounds, upper_bounds)
    
    params_sol = least_squares(
        residuals_params,
        x0=[eta1_init, eta2_init, lbd_init],
        bounds=bounds_to_use,
        method="trf"
    )

    eta1_cal, eta2_cal, lbd_cal = params_sol.x
    xi0_cal = solve_xi0(eta1_cal, eta2_cal, lbd_cal)

    return {
        "T": float(T), "eta1": eta1_cal, "eta2": eta2_cal,
        "H": H, "xi0": xi0_cal, "lbd": lbd_cal, "cost": params_sol.cost
        }

def calib_global_maturity_expan(market_data_list, params_init, H, n_max, n_mc, method):
    """
    """
    
    eta1_init, eta2_init, xi0_init, lbd_init = params_init

    def solve_xi0_for_maturity(eta1, eta2, lbd, T_target, fwd_target):
        
        def residual_x(x):
            x = float(np.atleast_1d(x)[0])
            xi0 = float(np.exp(x))

            mrb = MixedRoughBergomi(eta=[eta1, eta2], H=H, xi0=xi0, lbd=lbd)

            optimal_order = OptimalOrder(mrb, T_target, rule=-1, n_max=n_max, n_mc=n_mc).optimal_order()
            expan_inst = HermiteExpansion(model=mrb, T=T_target, rule=-1, optimal_order=optimal_order)
            fwd_model = expan_inst.vix_option_price_expan(kappa=0, opttype=0)

            return fwd_target - fwd_model

        try:
            x_sol = brentq(residual_x, -10.0, 10.0, xtol=1e-8)
            return float(np.exp(x_sol))
        except ValueError:
            x_guess = np.log(xi0_init) if xi0_init > 0 else 0.0
            res = least_squares(residual_x, x0=[x_guess])
            if res.success:
                return float(np.exp(res.x[0]))
            return np.nan


    def global_residuals(params):
        eta1, eta2, lbd = params
        all_residuals = []

        for data in market_data_list:
            T_i = data['T']
            fwd_i = data['fwd_mkt']
            iv_mkt_i = data['iv_mkt'].astype(float)
            kappa_i = data['kappa']
            
            k_i = np.log(kappa_i / fwd_i).astype(float)

            xi0_i = solve_xi0_for_maturity(eta1, eta2, lbd, T_i, fwd_i)
            
            if np.isnan(xi0_i):
                return np.ones_like(iv_mkt_i) * 999.0

            mrb = MixedRoughBergomi(eta=[eta1, eta2], H=H, xi0=xi0_i, lbd=lbd)
            optimal_order = OptimalOrder(mrb, T=T_i, rule=-1, n_max=n_max, n_mc=n_mc).optimal_order()
            expan_inst = HermiteExpansion(model=mrb, T=T_i, rule=-1, optimal_order=optimal_order)

            if method == 1:
                iv_model_i = expan_inst.implied_vol_expan(k=k_i, method=method)[0]
            else:
                iv_model_i = expan_inst.implied_vol_expan_cal(k=k_i, F_mkt=fwd_i, method=method)[0]

            residuals_i = iv_mkt_i - iv_model_i
            all_residuals.append(residuals_i)

        return np.concatenate(all_residuals)

    lower_bounds = [0.0, 0.0, 0.0]
    upper_bounds = [np.inf, np.inf, 1.0]
    
    params_sol = least_squares(
        global_residuals,
        x0=[eta1_init, eta2_init, lbd_init],
        bounds=(lower_bounds, upper_bounds),
        method="trf",
        verbose=1 
    )

    eta1_cal, eta2_cal, lbd_cal = params_sol.x

    results = {
        "global_params": {"eta1": eta1_cal, "eta2": eta2_cal, "lbd": lbd_cal, "H": H},
        "cost": params_sol.cost,
        "term_structure": []
    }

    for data in market_data_list:
        xi0_final = solve_xi0_for_maturity(eta1_cal, eta2_cal, lbd_cal, data['T'], data['fwd_mkt'])
        results["term_structure"].append({
            "T": data['T'],
            "xi0": xi0_final
        })

    return results

def repricing_calibrated_model_proxy(market_data, calibrated_params):
    """
    """
    results = []
    for T, g in market_data.groupby('Texp'):

        fwd_mkt = g['Fwd'].iloc[0]
        mid_ivs = g['Mid']
        strikes = g['Strike']
        log_strikes = np.log(strikes / fwd_mkt)

        N = len(log_strikes)

        sort_indices = np.argsort(log_strikes)

        params_for_T = calibrated_params.loc[T]
        eta1=params_for_T['eta1']
        eta2=params_for_T['eta2']
        H=params_for_T['H']
        xi0=params_for_T['xi0']
        lbd=params_for_T['lbd']

        n_gauss = 120

        mrb = MixedRoughBergomi(eta=[eta1, eta2], H=H, xi0=xi0, lbd=lbd)

        fwd_proxy = mrb.vix_price_proxy(kappa=0, T=T, opttype=0, n_gauss=n_gauss)

        iv_proxy_sorted = mrb.implied_vol_proxy(k=log_strikes.iloc[sort_indices], T=T, n_gauss=n_gauss)

        temp_df = pd.DataFrame({
            'Texp': [T] * N,
            'log_strike': log_strikes.iloc[sort_indices],
            'iv_mkt': mid_ivs.iloc[sort_indices],
            'iv_cal': iv_proxy_sorted,
            'fwd_mkt': [fwd_mkt] * N,
            'fwd_cal': [fwd_proxy] * N
        })

        results.append(temp_df)

    df_calibrated_results = pd.concat(results)

    return df_calibrated_results

def repricing_calibrated_model_expan(market_data, calibrated_params, method):

    results = []
    for T, g in market_data.groupby('Texp'):

        fwd_mkt = g['Fwd'].iloc[0]
        mid_ivs = g['Mid']
        strikes = g['Strike']
        log_strikes = np.log(strikes / fwd_mkt)

        N = len(log_strikes)

        sort_indices = np.argsort(log_strikes)

        params_for_T = calibrated_params.loc[T]
        eta1=params_for_T['eta1']
        eta2=params_for_T['eta2']
        H=params_for_T['H']
        xi0=params_for_T['xi0']
        lbd=params_for_T['lbd']

        n_max = 20
        n_mc = 10**6

        mrb = MixedRoughBergomi(eta=[eta1, eta2], H=H, xi0=xi0, lbd=lbd)
        optimal_order = OptimalOrder(mrb, T, rule=-1, n_max=n_max, n_mc=n_mc).optimal_order()
        expan_inst = HermiteExpansion(mrb, T, rule=-1, optimal_order=optimal_order)

        fwd_proxy = expan_inst.vix_option_price_expan(kappa=0, opttype=0)

        iv_proxy_sorted = expan_inst.implied_vol_expan(k=log_strikes.iloc[sort_indices], method=method)[0]

        temp_df = pd.DataFrame({
            'Texp': [T] * N,
            'log_strike': log_strikes.iloc[sort_indices],
            'iv_mkt': mid_ivs.iloc[sort_indices],
            'iv_cal': iv_proxy_sorted,
            'fwd_mkt': [fwd_mkt] * N,
            'fwd_cal': [fwd_proxy] * N
        })

        results.append(temp_df)

    df_calibrated_results = pd.concat(results)
    return df_calibrated_results


def calibrated_plots_fwd(calibrated_results):
    """
    """

    plot_data = calibrated_results.groupby('Texp')[['fwd_mkt', 'fwd_cal']].first().reset_index()
    plot_data = plot_data.sort_values(by='Texp')

    x_texp = plot_data['Texp']
    y_fwd1 = plot_data['fwd_mkt']
    y_fwd2 = plot_data['fwd_cal']

    rel_errors = utils.rel_error(y_fwd1, y_fwd2)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9, 3.5))

    axes[0].plot(x_texp, y_fwd1, marker='o', linestyle='None', color="blue", markersize=5, label="Market")
    axes[0].plot(x_texp, y_fwd2, color='orange', linewidth=2.5, label="Model")
    axes[0].legend(fontsize=12)
    axes[0].grid(True)
    axes[0].set_xlabel('Maturity (years)')
    axes[0].set_ylabel('VIX futures')

    axes[1].plot(x_texp, rel_errors, marker='x', linestyle='None',color='green')
    axes[1].set_xlabel("Maturity (years)")
    axes[1].set_ylabel("Relative error (%)")

    plt.show()
    

def calibrated_plots_vix_smiles(calibrated_results):
    """
    """

    groups_for_plot = list(calibrated_results.groupby('Texp'))

    nrows = len(groups_for_plot)
    ncols = 2

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(9, nrows * 3.5))

    if nrows == 1:
        axes_rows = np.array([axes])
    else:
        axes_rows = axes

    for i, (T, g) in enumerate(groups_for_plot):
        
        ax_left = axes_rows[i, 0]
        ax_right = axes_rows[i, 1]

        log_strikes = g['log_strike']
        iv_mkt = g['iv_mkt']
        iv_model = g['iv_cal']
       
        rel_error = utils.rel_error(iv_mkt, iv_model)
            
        ax_left.set_title(f"T = {T:.3f}", fontsize=18, fontweight='bold')
        ax_left.set_xlabel("Log-strike")
        ax_left.set_ylabel("Implied volatility")
        
        ax_left.plot(log_strikes, iv_mkt, marker='o', linestyle='None', 
                    color='blue', markersize=5, label="Market")
        ax_left.plot(log_strikes, iv_model, color='orange', linewidth=2.5, label="Model")
        ax_left.legend(fontsize=12)
        ax_left.grid(True)

        ax_right.set_title(f"T = {T:.3f}", fontsize=18, fontweight='bold')
        ax_right.set_xlabel("Log-strike")
        ax_right.set_ylabel("Relative error (%)")
        ax_right.plot(log_strikes, rel_error, marker='x', linestyle='None',color='green')

        ax_right.grid(True)

    plt.tight_layout()
    plt.show()
    