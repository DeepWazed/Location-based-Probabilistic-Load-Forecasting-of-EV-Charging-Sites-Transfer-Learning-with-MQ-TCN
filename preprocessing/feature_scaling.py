import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, MinMaxScaler


def robust_scaling(df, robust_cols, sin_cos_cols, column_name):
    scaled_df = df.copy()

    robust_scaler = RobustScaler()
    scaling_params = {}

    for col in robust_cols:
        if col == column_name:
            scaled_df[col] = robust_scaler.fit_transform(scaled_df[[col]])
            scaling_params[col] = (robust_scaler.center_[0], robust_scaler.scale_[0])

    for col in sin_cos_cols:
        scaled_df[col + '_sin'] = np.sin(2 * np.pi * scaled_df[col] / scaled_df[col].max())
        scaled_df[col + '_cos'] = np.cos(2 * np.pi * scaled_df[col] / scaled_df[col].max())
        scaled_df.drop(col, axis=1, inplace=True)

    return scaled_df, scaling_params


def min_max_scaling(df, minmax_cols, sin_cos_cols, column_name):
    scaled_df = df.copy()
    scaler = MinMaxScaler()
    scaling_params = {}

    for col in minmax_cols:
        if col == column_name:
            scaling_params[col] = (scaled_df[col].min(), scaled_df[col].max())
        scaled_df[col] = scaler.fit_transform(scaled_df[[col]])

    for col in sin_cos_cols:
        scaled_df[col + '_sin'] = np.sin(2 * np.pi * scaled_df[col] / scaled_df[col].max())
        scaled_df[col + '_cos'] = np.cos(2 * np.pi * scaled_df[col] / scaled_df[col].max())
        scaled_df.drop(col, axis=1, inplace=True)

    return scaled_df, scaling_params
    
    
def one_hot_encode_column(dataframe, column_name):
    one_hot_encoded = pd.get_dummies(dataframe[column_name], prefix=column_name)
    dataframe = pd.concat([dataframe, one_hot_encoded], axis=1)
    dataframe = dataframe.drop(column_name, axis=1)
    # dataframe[one_hot_encoded.columns] = dataframe[one_hot_encoded.columns].astype(int)

    return dataframe
