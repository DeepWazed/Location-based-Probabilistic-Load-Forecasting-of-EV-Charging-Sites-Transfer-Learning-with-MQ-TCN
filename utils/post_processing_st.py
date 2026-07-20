import numpy as np
import torch
import torch.nn as nn
import pandas as pd

def process_quantile_data(y_test, y_pred, quantiles, scaling_type, scaling_params, timestamp_list, train_df, val_df, scaled_df, n_past, n_future, stride = 24):

    y_pred_q1 = y_pred[:,:,0].reshape(y_pred.shape[0], -1)
    y_pred_q2 = y_pred[:,:,1].reshape(y_pred.shape[0], -1)
    y_pred_q3 = y_pred[:,:,2].reshape(y_pred.shape[0], -1)

    columns = ['t+' + str(i) for i in range(1, y_test.shape[1] + 1)]
    y_test_df = pd.DataFrame(y_test.reshape(y_test.shape[0], -1), columns=columns)
    y_pred_q1_df = pd.DataFrame(y_pred_q1.reshape(y_pred_q1.shape[0], -1), columns=columns)
    y_pred_q2_df = pd.DataFrame(y_pred_q2.reshape(y_pred_q2.shape[0], -1), columns=columns)
    y_pred_q3_df = pd.DataFrame(y_pred_q3.reshape(y_pred_q3.shape[0], -1), columns=columns)

    col_index = y_test_df.columns.get_loc('t+1')
    scaling_param = scaling_params['Smoothed_kWhDeliveredTotal']

    if scaling_type == 'robust':
        robust_center, robust_scale = np.array(scaling_param)
        y_test_df.iloc[:, col_index] = y_test_df.iloc[:, col_index] * robust_scale + robust_center
        y_pred_q1_df.iloc[:, col_index] = y_pred_q1_df.iloc[:, col_index] * robust_scale + robust_center
        y_pred_q2_df.iloc[:, col_index] = y_pred_q2_df.iloc[:, col_index] * robust_scale + robust_center
        y_pred_q3_df.iloc[:, col_index] = y_pred_q3_df.iloc[:, col_index] * robust_scale + robust_center

    elif scaling_type == 'min_max':
        min_value, max_value = np.array(scaling_param)
        y_test_df.iloc[:, col_index] = y_test_df.iloc[:, col_index] * (max_value - min_value) + min_value
        y_pred_q1_df.iloc[:, col_index] = y_pred_q1_df.iloc[:, col_index] * (max_value - min_value) + min_value
        y_pred_q2_df.iloc[:, col_index] = y_pred_q2_df.iloc[:, col_index] * (max_value - min_value) + min_value
        y_pred_q3_df.iloc[:, col_index] = y_pred_q3_df.iloc[:, col_index] * (max_value - min_value) + min_value


    y_test_df = y_test_df.iloc[:, 0]
    y_pred_q1_df = y_pred_q1_df.iloc[:, 0]
    y_pred_q2_df = y_pred_q2_df.iloc[:, 0]
    y_pred_q3_df = y_pred_q3_df.iloc[:, 0]
    y_test_org = y_test_df.copy()
    y_test_pred_q1 = y_pred_q1_df.copy()
    y_test_pred_q2 = y_pred_q2_df.copy()
    y_test_pred_q3 = y_pred_q3_df.copy()

    timestamp_indices = range(len(train_df) + len(val_df) + n_past, len(scaled_df) - n_future + 1, stride)
    doneChargingTime_subset = [timestamp_list[i] for i in timestamp_indices]

    result_df = pd.DataFrame({'y_test_org': y_test_org, 'y_test_pred_q1': y_test_pred_q1, 'y_test_pred_q2': y_test_pred_q2, 'y_test_pred_q3': y_test_pred_q3})
    
    for column in result_df.columns:
        if column != 'y_test_org':
            result_df[column] = result_df[column].apply(lambda x: 0 if x < 0 else x)
    result_df.insert(0, 'doneChargingTime', doneChargingTime_subset)
    result_df['doneChargingTime'] = pd.to_datetime(result_df['doneChargingTime'])
    result_df_daily_mean = result_df.copy()
    result_df_daily_mean = result_df_daily_mean.set_index('doneChargingTime')
    result_df_daily_mean = result_df_daily_mean.resample('D').mean()
    result_df_daily_mean = result_df_daily_mean.reset_index()

    result_df_daily_total = result_df.copy()
    result_df_daily_total = result_df_daily_total.set_index('doneChargingTime')
    result_df_daily_total = result_df_daily_total.resample('D').sum()
    result_df_daily_total = result_df_daily_total.reset_index()

    return result_df, result_df_daily_total, result_df_daily_mean
