import os
import pandas as pd
import numpy as np
from joblib import Parallel, delayed 
import multiprocessing 

import calibration

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"


methods_to_run = [1, 2, 3]


N_CORES = max(1, multiprocessing.cpu_count() - 1) 


def worker_single_fixed(T, g, params_init, H, N_MAX, N_MC, method):
    """
    """

    fwd_mkt = g['Fwd'].unique()[0]
    kappa_list = g['Strike'].to_numpy()
    mid_ivs     = g['Mid'].to_numpy()

    res = calibration.calib_one_maturity_expan(
        T=T,
        kappa=kappa_list,
        fwd_mkt=fwd_mkt,
        iv_mkt=mid_ivs,
        params_init=params_init,
        H=H,
        n_max=N_MAX,
        n_mc=N_MC,
        method=method
    )
    return T, res


def worker_single_random(T, g, params_init_list, H, N_MAX, N_MC, method, baseline_res=None):
    """
    """

    fwd_mkt = g['Fwd'].unique()[0]
    kappa_list = g['Strike'].to_numpy()
    mid_ivs     = g['Mid'].to_numpy()

    best_result_for_T = None
    lowest_cost = np.inf

    if baseline_res is not None:
        base_cost = baseline_res.get('cost')
        if base_cost is not None and not np.isnan(base_cost):
            lowest_cost = base_cost
            best_result_for_T = baseline_res

    for params_init in params_init_list:

        res = calibration.calib_one_maturity_expan(
            T=T,
            kappa=kappa_list,
            fwd_mkt=fwd_mkt,
            iv_mkt=mid_ivs,
            params_init=params_init,
            H=H,
            n_max=N_MAX,
            n_mc=N_MC,
            method=method
        )

        current_cost = res.get('cost')

        if current_cost is not None and current_cost < lowest_cost:
            lowest_cost = current_cost
            best_result_for_T = res
            best_result_for_T['initial_guess'] = params_init

    if best_result_for_T is not None:
        print(f"  -> Best Cost for T={T:.3f}: {lowest_cost:.6f}")
        return T, best_result_for_T
    else:
        print(f"  -> Warning: Calibration failed for T={T:.3f}")
        failed_res = {
            "T": T, "eta1": np.nan, "eta2": np.nan, 
            "H": H, "xi0": np.nan, "lbd": np.nan, 
            "cost": np.nan, "initial_guess": None
        }
        return T, failed_res


def expansion_calib_fixed(market_data_df, H, N_MAX, N_MC, params_init_fixed, method):
    """
    """

    print(f"Using {N_CORES} cores for parallel processing.")
    
    groups = list(market_data_df.groupby('Texp'))

    print(f"\nStart processing formula {method} (Fixed Guess)")
        
    results_list = Parallel(n_jobs=N_CORES)(
        delayed(worker_single_fixed)(T, g, params_init_fixed, H, N_MAX, N_MC, method)
        for T, g in groups
    )
    results = dict(results_list)

    df_results_params = pd.DataFrame(results).T

    if not df_results_params.empty:
        df_results_params = df_results_params.sort_index()
        df_results_params.index.name = 'T'
        df_results_params.index = df_results_params.index.astype(float)

        if 'T' in df_results_params.columns:
            df_results_params = df_results_params.drop(columns=['T'])

        print("\nCalibration results (Parameters):")
        print(df_results_params)

        df_calibrated_results = calibration.repricing_calibrated_model_expan(market_data_df, df_results_params, method=method)
    else:
        print(f"\nNo calibration results obtained for formula {method}.")

    return df_calibrated_results

def expansion_calib_random(market_data_df, H, N_MAX, N_MC, eta_range, xi0_range, lbd_range, N_SAMPLES, method):
    """
    """

    print(f"Using {N_CORES} cores for parallel processing.")
    
    groups = list(market_data_df.groupby('Texp'))

    np.random.seed(777) 
    eta1_samples = np.random.uniform(eta_range[0], eta_range[1], N_SAMPLES)
    eta2_samples = np.random.uniform(eta_range[0], eta_range[1], N_SAMPLES)
    xi0_samples  = np.random.uniform(xi0_range[0], xi0_range[1], N_SAMPLES)
    lbd_samples  = np.random.uniform(lbd_range[0], lbd_range[1], N_SAMPLES)
    params_init_list = list(zip(eta1_samples, eta2_samples, xi0_samples, lbd_samples))

    print(f"\nStart processing formula {method} (Random Guess)")
        
    results_list = Parallel(n_jobs=N_CORES)(
        delayed(worker_single_random)(T, g, params_init_list, H, N_MAX, N_MC, method)
        for T, g in groups
    )
    results = dict(results_list)

    df_results_params = pd.DataFrame(results).T

    if not df_results_params.empty:
        df_results_params = df_results_params.sort_index()
        df_results_params.index.name = 'T'
        df_results_params.index = df_results_params.index.astype(float)
            
        if 'T' in df_results_params.columns:
            df_results_params = df_results_params.drop(columns=['T'])

        print("\nCalibration results (Parameters):")
        print(df_results_params)

        df_calibrated_results = calibration.repricing_calibrated_model_expan(market_data_df, df_results_params, method=method)
    else:
        print(f"\nNo calibration results obtained for Method {method}.")

    return df_calibrated_results
