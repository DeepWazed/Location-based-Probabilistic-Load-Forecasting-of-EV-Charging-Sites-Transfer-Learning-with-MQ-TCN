import numpy as np
import pandas as pd
import math
from math import sqrt
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, mean_squared_log_error

def performance_metrics(result_df):

    # compute MAE
    mae = mean_absolute_error(result_df['y_test_org'], result_df['y_test_pred'])

    # compute R2 score
    r2 = r2_score(result_df['y_test_org'], result_df['y_test_pred'])

    # compute MSE
    mse = mean_squared_error(result_df['y_test_org'], result_df['y_test_pred'])

    # compute RMSE
    rmse = math.sqrt(mse)

    # compute MAPE (Mean Absolute Percentage Error)
    def mean_absolute_percentage_error(y_true, y_pred):
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    mape = mean_absolute_percentage_error(result_df['y_test_org'], result_df['y_test_pred'])

    # compute NRMSE (Normalized Root Mean Squared Error)
    def normalized_root_mean_squared_error(y_true, y_pred):
        rmse = math.sqrt(mean_squared_error(y_true, y_pred))
        return rmse / (np.max(y_true) - np.min(y_true))

    nrmse = normalized_root_mean_squared_error(result_df['y_test_org'], result_df['y_test_pred'])

    # compute RMLSE (Root Mean Log Squared Error)
    def root_mean_log_squared_error(y_true, y_pred):
        rmsle = math.sqrt(np.mean(np.log1p(y_true) - np.log1p(y_pred))**2)
        return rmsle

    rmlse = root_mean_log_squared_error(result_df['y_test_org'], result_df['y_test_pred'])
    
    # compute SMAPE (Symmetric Mean Absolute Percentage Error)
    def symmetric_mean_absolute_percentage_error(y_true, y_pred):
        return 100 * 2 * np.mean(np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred)))
        
    smape = symmetric_mean_absolute_percentage_error(result_df['y_test_org'], result_df['y_test_pred'])

    # compute MSLE (Mean Squared Logarithmic Error)
    msle = mean_squared_log_error(result_df['y_test_org'], result_df['y_test_pred'])
    
    performance_metrics_summary = pd.DataFrame({
        'MAE': [mae],
        'R2': [r2],
        'MSE': [mse],
        'RMSE': [rmse],
        'MAPE': [mape],
        'NRMSE': [nrmse],
        'RMLSE': [rmlse],
        'SMAPE': [smape],
        'MSLE': [msle]
    })

    return performance_metrics_summary

    

def evaluate_PICP_WS(df, confidence):
    y_pred_upper = df['y_test_pred_q2']
    y_pred_lower = df['y_test_pred_q1']
    y_test = df['y_test_org']

    idx_oobl = np.where(y_test < y_pred_lower)[0]
    idx_oobu = np.where(y_test > y_pred_upper)[0]

    PICP = np.sum((y_test > y_pred_lower) & (y_test <= y_pred_upper)) / len(y_test) * 100

    WS = (np.sum(y_pred_upper - y_pred_lower) +
          np.sum(2 * (y_pred_lower[idx_oobl] - y_test[idx_oobl]) / confidence) +
          np.sum(2 * (y_test[idx_oobu] - y_pred_upper[idx_oobu]) / confidence)) / len(y_test)

    PINC = 1 - WS / (upper_bound - lower_bound)

    print ("PICP of testing set: {:.2f}%".format(PICP))
    print ("WS of testing set: {:.2f}".format(WS))
    print("PINC of testing set: {:.2f}%".format(PINC * 100))

    return PICP, PINC, WS

# IF WE NEED LATER 
# upper_bound = 0.95
# lower_bound = 0.05
# confidence = 1 - (upper_bound - lower_bound)
# PICP, PINC, WS = evaluate_PICP_WS(base_result_df_daily, confidence)




def quantile_performance_matrix(y_test_org, y_test_pred_q2, y_test_pred_q3):
    mse1 = mean_squared_error(y_test_org, y_test_pred_q2)
    rmse1 = sqrt(mse1)
    nrmse1 = rmse1 / (y_test_org.max() - y_test_org.min())

    mse2 = mean_squared_error(y_test_org, y_test_pred_q3)
    rmse2 = sqrt(mse2)
    nrmse2 = rmse2 / (y_test_org.max() - y_test_org.min())

    smape1 = 100 * (2 * np.abs(y_test_org - y_test_pred_q2) / (np.abs(y_test_org) + np.abs(y_test_pred_q2))).mean()
    smape2 = 100 * (2 * np.abs(y_test_org - y_test_pred_q3) / (np.abs(y_test_org) + np.abs(y_test_pred_q3))).mean()

    y_test_org_diff = y_test_org.diff(1).dropna()
    mase1 = mean_absolute_error(y_test_org, y_test_pred_q2) / (np.abs(y_test_org_diff).mean())
    mase2 = mean_absolute_error(y_test_org, y_test_pred_q3) / (np.abs(y_test_org_diff).mean())
    
    rmlse1 = np.sqrt(np.mean(np.log1p(y_test_org) - np.log1p(y_test_pred_q2))**2)
    rmlse2 = np.sqrt(np.mean(np.log1p(y_test_org) - np.log1p(y_test_pred_q3))**2)
    
    nd1 = (np.abs(y_test_org - y_test_pred_q2) / (y_test_org.max() - y_test_org.min())).mean()
    nd2 = (np.abs(y_test_org - y_test_pred_q3) / (y_test_org.max() - y_test_org.min())).mean()


    error_summary = pd.DataFrame({
        'MSE1': [mse1],
        'MSE2': [mse2],
        'RMSE1': [rmse1],
        'RMSE2': [rmse2],
        'NRMSE1': [nrmse1],
        'NRMSE2': [nrmse2],
        'SMAPE1': [smape1],
        'SMAPE2': [smape2],
        'MASE1': [mase1],
        'MASE2': [mase2],
        'RMLSE1': [rmlse1],
        'RMLSE2': [rmlse2],
        'ND1': [nd1],
        'ND2': [nd2]
    })

    return error_summary
