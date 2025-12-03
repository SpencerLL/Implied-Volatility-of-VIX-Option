import os
import pandas as pd
import numpy as np
from joblib import Parallel, delayed 
import multiprocessing 
import calibration

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"



N_CORES = max(1, multiprocessing.cpu_count() - 1) 



def worker_calib_fixed(T, g, params_init, H, N_GAUSS):
    """
    """

    fwd_mkt = g['Fwd'].unique()[0]
    kappa_list = g['Strike'].to_numpy()
    mid_ivs     = g['Mid'].to_numpy()

    res = calibration.calib_one_maturity_proxy(
        T=T,
        kappa=kappa_list,
        fwd_mkt=fwd_mkt,
        iv_mkt=mid_ivs,
        params_init=params_init,
        H=H,
        n_gauss=N_GAUSS
    )
    return T, res


def worker_calib_random(T, g, params_init_list, H, N_GAUSS):
    """
    """

    fwd_mkt = g['Fwd'].unique()[0]
    kappa_list = g['Strike'].to_numpy()
    mid_ivs     = g['Mid'].to_numpy()

    best_result_for_T = None
    lowest_cost = np.inf

    for params_init in params_init_list:

        try:
            res = calibration.calib_one_maturity_proxy(
                T=T,
                kappa=kappa_list,
                fwd_mkt=fwd_mkt,
                iv_mkt=mid_ivs,
                params_init=params_init,
                H=H,
                n_gauss=N_GAUSS
            )
            
            current_cost = res.get('cost')
            
            if current_cost is not None and np.isfinite(current_cost) and current_cost < lowest_cost:
                lowest_cost = current_cost
                best_result_for_T = res
                best_result_for_T['initial_guess'] = params_init

        except ValueError as e:
            print(f"  -> Skipping invalid guess for T={T:.3f} due to ValueError: {e}")
            continue
        except OverflowError:
            print(f"  -> Skipping invalid guess for T={T:.3f} due to OverflowError.")
            continue
        except Exception as e:
            print(f"  -> Skipping guess for T={T:.3f} due to unexpected error: {e}")
            continue
    
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


def root_finding_calib_fixed(market_data_df, H, N_GAUSS, params_init_fixed):
    """
    """

    print(f"Using {N_CORES} cores for parallel processing.")
    
    groups = list(market_data_df.groupby('Texp'))
    
    results_list_fixed = Parallel(n_jobs=N_CORES)(
        delayed(worker_calib_fixed)(T, g, params_init_fixed, H, N_GAUSS)
        for T, g in groups
    )

    results = dict(results_list_fixed)
    df_results_params = pd.DataFrame(results).T

    if not df_results_params.empty:
        df_results_params = df_results_params.sort_index()
        df_results_params.index.name = 'T'
        df_results_params.index = df_results_params.index.astype(float)  

        if 'T' in df_results_params.columns:
            df_results_params = df_results_params.drop(columns=['T']) 

        print("\nCalibration results (Parameters) - Fixed Guess:")
        print(df_results_params)
        
        df_calibrated_results = calibration.repricing_calibrated_model_proxy(market_data_df, df_results_params)
    else:
        print("\nNo calibration results obtained (Fixed Guess). Skipping plots.")

    
    return df_calibrated_results


def root_finding_calib_random(market_data_df, H, N_GAUSS, eta_range, xi0_range, lbd_range, N_SAMPLES):
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
    
    results_list_random = Parallel(n_jobs=N_CORES)(
        delayed(worker_calib_random)(T, g, params_init_list, H, N_GAUSS)
        for T, g in groups
    )

    results = dict(results_list_random)
    df_results_params = pd.DataFrame(results).T

    if not df_results_params.empty:
        df_results_params = df_results_params.sort_index()
        df_results_params.index.name = 'T'
        df_results_params.index = df_results_params.index.astype(float)

        if 'T' in df_results_params.columns:
            df_results_params = df_results_params.drop(columns=['T'])

        print("\nCalibration results (Parameters) - Random Guess:")
        print(df_results_params)
        
        df_calibrated_results = calibration.repricing_calibrated_model_proxy(market_data_df, df_results_params)
        # calibration.calibrated_plots_fwd(df_calibrated_results, name="root_finding_2")
        # calibration.calibrated_plots_vix_smiles(df_calibrated_results, name="root_finding_2")
    else:
        print("\nNo calibration results obtained (Random Guess). Skipping plots.")

    print("\nAll calibration tasks completed.")

    return df_calibrated_results
