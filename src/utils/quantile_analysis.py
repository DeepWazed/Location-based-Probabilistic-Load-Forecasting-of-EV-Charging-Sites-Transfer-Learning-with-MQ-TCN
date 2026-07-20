import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import math
from math import sqrt
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, mean_squared_log_error

def pinball_loss(outputs, targets, quantiles, reduction="mean"):
    assert len(quantiles) == outputs.shape[-1], "Number of quantiles should match number of outputs"
    assert outputs.shape[:-1] == targets.shape, "Output and target shape mismatch"

    loss = 0

    for i, q in enumerate(quantiles):
        err = targets - outputs[:, :, i]
        loss += torch.max((q - 1) * err, q * err)
    
    if reduction == "mean":
        return torch.mean(loss) / len(quantiles)  
    elif reduction == "sum":
        return torch.sum(loss)
    elif reduction == "none":
        return loss  
    else:
        raise ValueError(f"Invalid reduction type: {reduction}")
    
    
def quantile_test(device, model, test_loader, quantiles, reduction):
    test_losses = []  
    preds = []
    model.to(device)
    model.eval()
    with torch.no_grad():
        for x_con_batch, x_cat_batch, y_batch in test_loader:
            x_con_batch, x_cat_batch, y_batch = x_con_batch.to(device), x_cat_batch.to(device), y_batch.to(device)
            outputs = model(x_con_batch, x_cat_batch)
            loss = pinball_loss(outputs, y_batch, quantiles, reduction)
            test_loss = loss.item() * x_con_batch.size(0)
            test_losses.append(test_loss)  
            preds.append(outputs.cpu().numpy())
    
    preds = np.concatenate(preds, axis=0)
    test_loss = np.sum(test_losses) / len(test_loader.dataset)
    test_losses = [loss / len(test_loader) for loss in test_losses]
    print(f'Test Loss: {test_loss:.4f}')
    print(f"Individual Quantile Loss: {test_losses}")
    return test_loss, test_losses, preds
    
    
    
def process_quantile_data(y_test, y_pred, quantiles, scaling_type, scaling_params, timestamp_list, train_df, val_df, scaled_df, n_past, n_future):

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

    if y_test.shape[0] == 128:
        doneChargingTime_subset = timestamp_list[len(train_df)+len(val_df)+n_past:len(scaled_df)-(n_future)+1]
        doneChargingTime_subset = doneChargingTime_subset[:128]
    else:
        doneChargingTime_subset = timestamp_list[len(train_df)+len(val_df)+n_past:len(scaled_df)-(n_future)+1]
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


def process_multi_quantile_data(y_test, y_pred, quantiles, scaling_type, scaling_params, timestamp_list, train_df, val_df, scaled_df, n_past, n_future):

    y_pred_q1 = y_pred[:,:,0].reshape(y_pred.shape[0], -1)
    y_pred_q2 = y_pred[:,:,1].reshape(y_pred.shape[0], -1)
    y_pred_q3 = y_pred[:,:,2].reshape(y_pred.shape[0], -1)
    y_pred_q4 = y_pred[:,:,3].reshape(y_pred.shape[0], -1)
    y_pred_q5 = y_pred[:,:,4].reshape(y_pred.shape[0], -1)
    y_pred_q6 = y_pred[:,:,5].reshape(y_pred.shape[0], -1)

    columns = ['t+' + str(i) for i in range(1, y_test.shape[1] + 1)]
    y_test_df = pd.DataFrame(y_test.reshape(y_test.shape[0], -1), columns=columns)
    y_pred_q1_df = pd.DataFrame(y_pred_q1.reshape(y_pred_q1.shape[0], -1), columns=columns)
    y_pred_q2_df = pd.DataFrame(y_pred_q2.reshape(y_pred_q2.shape[0], -1), columns=columns)
    y_pred_q3_df = pd.DataFrame(y_pred_q3.reshape(y_pred_q3.shape[0], -1), columns=columns)
    y_pred_q4_df = pd.DataFrame(y_pred_q4.reshape(y_pred_q4.shape[0], -1), columns=columns)
    y_pred_q5_df = pd.DataFrame(y_pred_q5.reshape(y_pred_q5.shape[0], -1), columns=columns)
    y_pred_q6_df = pd.DataFrame(y_pred_q6.reshape(y_pred_q6.shape[0], -1), columns=columns)

    col_index = y_test_df.columns.get_loc('t+1')
    scaling_param = scaling_params['Smoothed_kWhDeliveredTotal']

    if scaling_type == 'robust':
        robust_center, robust_scale = np.array(scaling_param)
        y_test_df.iloc[:, col_index] = y_test_df.iloc[:, col_index] * robust_scale + robust_center
        y_pred_q1_df.iloc[:, col_index] = y_pred_q1_df.iloc[:, col_index] * robust_scale + robust_center
        y_pred_q2_df.iloc[:, col_index] = y_pred_q2_df.iloc[:, col_index] * robust_scale + robust_center
        y_pred_q3_df.iloc[:, col_index] = y_pred_q3_df.iloc[:, col_index] * robust_scale + robust_center
        y_pred_q4_df.iloc[:, col_index] = y_pred_q4_df.iloc[:, col_index] * robust_scale + robust_center
        y_pred_q5_df.iloc[:, col_index] = y_pred_q5_df.iloc[:, col_index] * robust_scale + robust_center
        y_pred_q6_df.iloc[:, col_index] = y_pred_q6_df.iloc[:, col_index] * robust_scale + robust_center

    elif scaling_type == 'min_max':
        min_value, max_value = np.array(scaling_param)
        y_test_df.iloc[:, col_index] = y_test_df.iloc[:, col_index] * (max_value - min_value) + min_value
        y_pred_q1_df.iloc[:, col_index] = y_pred_q1_df.iloc[:, col_index] * (max_value - min_value) + min_value
        y_pred_q2_df.iloc[:, col_index] = y_pred_q2_df.iloc[:, col_index] * (max_value - min_value) + min_value
        y_pred_q3_df.iloc[:, col_index] = y_pred_q3_df.iloc[:, col_index] * (max_value - min_value) + min_value
        y_pred_q4_df.iloc[:, col_index] = y_pred_q4_df.iloc[:, col_index] * (max_value - min_value) + min_value
        y_pred_q5_df.iloc[:, col_index] = y_pred_q5_df.iloc[:, col_index] * (max_value - min_value) + min_value
        y_pred_q6_df.iloc[:, col_index] = y_pred_q6_df.iloc[:, col_index] * (max_value - min_value) + min_value


    y_test_df = y_test_df.iloc[:, 0]
    y_pred_q1_df = y_pred_q1_df.iloc[:, 0]
    y_pred_q2_df = y_pred_q2_df.iloc[:, 0]
    y_pred_q3_df = y_pred_q3_df.iloc[:, 0]
    y_pred_q4_df = y_pred_q4_df.iloc[:, 0]
    y_pred_q5_df = y_pred_q5_df.iloc[:, 0]
    y_pred_q6_df = y_pred_q6_df.iloc[:, 0]

    y_test_org = y_test_df.copy()
    y_test_pred_q1 = y_pred_q1_df.copy()
    y_test_pred_q2 = y_pred_q2_df.copy()
    y_test_pred_q3 = y_pred_q3_df.copy()
    y_test_pred_q4 = y_pred_q4_df.copy()
    y_test_pred_q5 = y_pred_q5_df.copy()
    y_test_pred_q6 = y_pred_q6_df.copy()

    print("I am here")

    print(len(timestamp_list))

    if y_test.shape[0] == 128:
        doneChargingTime_subset = timestamp_list[len(train_df)+len(val_df)+n_past:len(scaled_df)-(n_future)+1]
        doneChargingTime_subset = doneChargingTime_subset[:128]
    else:
        doneChargingTime_subset = timestamp_list[len(train_df)+len(val_df)+n_past:len(scaled_df)-(n_future)+1]
        print(len(doneChargingTime_subset))
    result_df = pd.DataFrame({'y_test_org': y_test_org, 'y_test_pred_q1': y_test_pred_q1, 'y_test_pred_q2': y_test_pred_q2, 'y_test_pred_q3': y_test_pred_q3, 'y_test_pred_q4': y_test_pred_q4, 'y_test_pred_q5': y_test_pred_q5, 'y_test_pred_q6': y_test_pred_q6})
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
    
    
def quantile_performance_metrics(y_test_org, y_test_pred_q1, y_test_pred_q2, y_test_pred_q3):
    mse1 = mean_squared_error(y_test_org, y_test_pred_q1)
    rmse1 = sqrt(mse1)
    nrmse1 = rmse1 / (y_test_org.max() - y_test_org.min())
    
    mse2 = mean_squared_error(y_test_org, y_test_pred_q2)
    rmse2 = sqrt(mse2)
    nrmse2 = rmse2 / (y_test_org.max() - y_test_org.min())

    mse3 = mean_squared_error(y_test_org, y_test_pred_q3)
    rmse3 = sqrt(mse3)
    nrmse3 = rmse3 / (y_test_org.max() - y_test_org.min())
    
    smape1 = 100 * (2 * np.abs(y_test_org - y_test_pred_q1) / (np.abs(y_test_org) + np.abs(y_test_pred_q1))).mean()
    smape2 = 100 * (2 * np.abs(y_test_org - y_test_pred_q2) / (np.abs(y_test_org) + np.abs(y_test_pred_q2))).mean()
    smape3 = 100 * (2 * np.abs(y_test_org - y_test_pred_q3) / (np.abs(y_test_org) + np.abs(y_test_pred_q3))).mean()

    y_test_org_diff = y_test_org.diff(1).dropna()
    mase1 = mean_absolute_error(y_test_org, y_test_pred_q1) / (np.abs(y_test_org_diff).mean())
    mase2 = mean_absolute_error(y_test_org, y_test_pred_q2) / (np.abs(y_test_org_diff).mean())
    mase3 = mean_absolute_error(y_test_org, y_test_pred_q3) / (np.abs(y_test_org_diff).mean())
    
    rmlse1 = np.sqrt(np.mean((np.log1p(y_test_org) - np.log1p(y_test_pred_q1))**2))
    rmlse2 = np.sqrt(np.mean((np.log1p(y_test_org) - np.log1p(y_test_pred_q2))**2))
    rmlse3 = np.sqrt(np.mean((np.log1p(y_test_org) - np.log1p(y_test_pred_q3))**2))

    
    nd1 = (np.abs(y_test_org - y_test_pred_q1) / (y_test_org.max() - y_test_org.min())).mean()
    nd2 = (np.abs(y_test_org - y_test_pred_q2) / (y_test_org.max() - y_test_org.min())).mean()
    nd3 = (np.abs(y_test_org - y_test_pred_q3) / (y_test_org.max() - y_test_org.min())).mean()


    error_summary = pd.DataFrame({
        'MSE_Q1': [mse1],
        'MSE_Q2': [mse2],
        'MSE_Q3': [mse3],
        'RMSE_Q1': [rmse1],
        'RMSE_Q2': [rmse2],
        'RMSE_Q3': [rmse3],
        'NRMSE_Q1': [nrmse1],
        'NRMSE_Q2': [nrmse2],
        'NRMSE_Q3': [nrmse3],
        'SMAPE_Q1': [smape1],
        'SMAPE_Q2': [smape2],
        'SMAPE_Q3': [smape3],
        'MASE_Q1': [mase1],
        'MASE_Q2': [mase2],
        'MASE_Q3': [mase3],
        'RMLSE_Q1': [rmlse1],
        'RMLSE_Q2': [rmlse2],
        'RMLSE_Q3': [rmlse3],
        'ND_Q1': [nd1],
        'ND_Q2': [nd2],
        'ND_Q3': [nd3]
    })

    return error_summary