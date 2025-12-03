import os
import pandas as pd
import numpy as np
from joblib import Parallel, delayed 
import multiprocessing 

import calibration


os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

data_path = r"vix_implied_vol_20230215.csv"

N_CORES = max(1, multiprocessing.cpu_count() - 1) 


def build_market_data_list(group_df):
    """
    """

    market_data_list = []
    unique_Ts_in_bucket = group_df['Texp'].unique()
    
    for T_sub in unique_Ts_in_bucket:
        sub_g = group_df[group_df['Texp'] == T_sub]
        data_item = {
            "T": float(T_sub),
            "fwd_mkt": float(sub_g['Fwd'].unique()[0]),
            "kappa": sub_g['Strike'].to_numpy().astype(float),
            "iv_mkt": sub_g['Mid'].to_numpy().astype(float)
        }
        market_data_list.append(data_item)
    return unique_Ts_in_bucket, market_data_list


def process_global_result(global_res, lowest_cost, H, initial_guess=None):
    """
    """
    
    g_params = global_res['global_params']
    
    term_results = []
    for term_res in global_res['term_structure']:
        T_val = term_res['T']
        
        term_results.append({
            T_val: {
                "T": T_val,
                "eta1": g_params['eta1'],
                "eta2": g_params['eta2'],
                "H": g_params['H'],
                "lbd": g_params['lbd'],
                "xi0": term_res['xi0'],
                "cost": lowest_cost,
                "initial_guess": initial_guess
            }
        })
    return term_results


def worker_global_fixed(interval, group_df, params_init, H, N_MAX, N_MC, method):
    """
    """
    
    if group_df.empty:
        return []

    unique_Ts, market_data_list = build_market_data_list(group_df)

    try:
        global_res = calibration.calib_global_maturity_expan(
            market_data_list=market_data_list,
            params_init=params_init,
            H=H,
            n_max=N_MAX,
            n_mc=N_MC,
            method=method
        )
        
        print(f"  -> Success! Cost: {global_res['cost']:.6f}")
        return process_global_result(global_res, global_res['cost'], H)
            
    except Exception as e:
        print(f"!!! Calibration failed for bucket {interval}: {e}")

        nan_results = []
        for T_sub in unique_Ts:
            nan_results.append({
                T_sub: {
                    "T": T_sub, "eta1": np.nan, "eta2": np.nan, 
                    "H": H, "xi0": np.nan, "lbd": np.nan, 
                    "cost": np.nan, "initial_guess": None
                }
            })
        return nan_results


def worker_global_random(interval, group_df, params_init_list, H, N_MAX, N_MC, method):
    """
    """

    if group_df.empty:
        return []

    unique_Ts, market_data_list = build_market_data_list(group_df)
    
    best_global_res = None
    lowest_cost = np.inf
    best_init_guess = None


    for params_init in params_init_list:
        try:
            global_res = calibration.calib_global_maturity_expan(
                market_data_list=market_data_list,
                params_init=params_init,
                H=H,
                n_max=N_MAX,
                n_mc=N_MC,
                method=method
            )
        except Exception:
            continue 
            
        current_cost = global_res.get('cost')

        if current_cost is not None and current_cost < lowest_cost:
            lowest_cost = current_cost
            best_global_res = global_res
            best_init_guess = params_init
                
    if best_global_res is not None:
        print(f"  -> Best Cost for Bucket: {lowest_cost:.6f}")
        return process_global_result(best_global_res, lowest_cost, H, initial_guess=best_init_guess)
    else:
        print(f"  -> Warning: Calibration failed for all guesses in Bucket {interval}")

        nan_results = []
        for T_sub in unique_Ts:
            nan_results.append({
                T_sub: {
                    "T": T_sub, "eta1": np.nan, "eta2": np.nan, 
                    "H": H, "xi0": np.nan, "lbd": np.nan, 
                    "cost": np.nan, "initial_guess": None
                }
            })
        return nan_results


def expansion_global_calib_fixed(market_data_df, H, N_MAX, N_MC, params_init_fixed, method):
    """
    """

    print(f"Using {N_CORES} cores for parallel processing.")
    
    max_T = market_data_df['Texp'].max()
    bins = np.arange(0, max_T + 1.0/12.0 + 0.001, 1.0/12.0)
    market_data_df['Bucket'] = pd.cut(market_data_df['Texp'], bins=bins, right=False)
    
    bucket_groups = list(market_data_df.groupby('Bucket'))

    print(f"\nStart processing formula {method} (Fixed Guess)")
        
    results_list_nested = Parallel(n_jobs=N_CORES)(
        delayed(worker_global_fixed)(interval, g, params_init_fixed, H, N_MAX, N_MC, method)
        for interval, g in bucket_groups
    )
        
    results = {}
    for bucket_results in results_list_nested:
        for term_res_dict in bucket_results:
            results.update(term_res_dict)


    df_results_params = pd.DataFrame(results).T

    if not df_results_params.empty:
        df_results_params = df_results_params.sort_index()
        df_results_params.index.name = 'T'
        df_results_params.index = df_results_params.index.astype(float)

        if 'T' in df_results_params.columns:
            df_results_params = df_results_params.drop(columns=['T'])

        print("\nCalibration results (Parameters):")
        print(df_results_params)

    else:
        print(f"\nNo calibration results obtained for Method {method}..")

    df_calibrated_results = calibration.repricing_calibrated_model_expan(market_data_df, df_results_params, method=method)
    return df_calibrated_results

def expansion_global_calib_random(market_data_df, H, N_MAX, N_MC, eta_range, xi0_range, lbd_range, N_SAMPLES, method):
    """
    """
    print(f"Using {N_CORES} cores for parallel processing.")

    max_T = market_data_df['Texp'].max()
    bins = np.arange(0, max_T + 1.0/12.0 + 0.001, 1.0/12.0)
    market_data_df['Bucket'] = pd.cut(market_data_df['Texp'], bins=bins, right=False)
    
    bucket_groups = list(market_data_df.groupby('Bucket'))

    np.random.seed(777) 
    eta1_samples = np.random.uniform(eta_range[0], eta_range[1], N_SAMPLES)
    eta2_samples = np.random.uniform(eta_range[0], eta_range[1], N_SAMPLES)
    xi0_samples  = np.random.uniform(xi0_range[0], xi0_range[1], N_SAMPLES)
    lbd_samples  = np.random.uniform(lbd_range[0], lbd_range[1], N_SAMPLES)
    params_init_list = list(zip(eta1_samples, eta2_samples, xi0_samples, lbd_samples))

    print(f"\nStart processing Method = {method} (Random Guess)")
        
    results_list_nested = Parallel(n_jobs=N_CORES)(
        delayed(worker_global_random)(interval, g, params_init_list, H, N_MAX, N_MC, method)
        for interval, g in bucket_groups
    )
        
    results = {}
    for bucket_results in results_list_nested:
        for term_res_dict in bucket_results:
            results.update(term_res_dict)

    df_results_params = pd.DataFrame(results).T

    if not df_results_params.empty:
        df_results_params = df_results_params.sort_index()
        df_results_params.index.name = 'T'
        df_results_params.index = df_results_params.index.astype(float)

        if 'T' in df_results_params.columns:
            df_results_params = df_results_params.drop(columns=['T'])
            
        print("\nCalibration results (Parameters):")
        print(df_results_params)
    else:
        print(f"\nNo calibration results obtained for Method {method}.")

    df_calibrated_results = calibration.repricing_calibrated_model_expan(market_data_df, df_results_params, method=method)
    return df_calibrated_results