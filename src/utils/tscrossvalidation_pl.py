import numpy as np
import torch
from torch.utils.data import TensorDataset

class BlockingTimeSeriesSplit():
    def __init__(self, n_splits, cv_train_size):
        self.n_splits = n_splits
        self.cv_train_size = cv_train_size

    def get_n_splits(self, X, y, groups):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        k_fold_size = n_samples // self.n_splits
        indices = np.arange(n_samples)

        margin = 0
        for i in range(self.n_splits):
            start = i * k_fold_size
            stop = start + k_fold_size
            mid = int(self.cv_train_size * (stop - start)) + start
            yield indices[start: mid], indices[mid + margin: stop]

def ts_kfold_cv(df, cat_features, n_past, n_future, device, test_size=0.1, k_fold=4, cv_train_size=0.8):
    cols = df.columns.tolist()
    cols.insert(0, cols.pop(cols.index('Smoothed_kWhDeliveredTotal')))
    df = df[cols]
    df = df.drop('doneChargingTime', axis=1)
    df_cat = df[cat_features].astype(np.int32) 
    df_num = df.drop(cat_features, axis=1).astype(np.float16)  

    df_cat = df_cat.to_numpy()
    df_num = df_num.to_numpy()

    X_num = []
    X_cat = []
    y = []

    for i in range(n_past, len(df) - n_future + 1):
        X_num.append(df_num[i - n_past:i, :])
        X_cat.append(df_cat[i - n_past:i, :])
        y.append(df_num[i:i + n_future, 0])

    full_input_num, full_input_cat, full_target = np.array(X_num), np.array(X_cat), np.array(y)

    train_size = 1 - test_size
    full_input_num = torch.tensor(full_input_num, dtype=torch.float16).to(device)  
    full_input_cat = torch.tensor(full_input_cat, dtype=torch.int).to(device)  

    input_train_num = full_input_num[:int(len(full_input_num) * train_size)]
    input_train_cat = full_input_cat[:int(len(full_input_cat) * train_size)]
    input_test_num = torch.tensor(full_input_num[int(len(full_input_num) * train_size):], dtype=torch.float16).to(device)  
    input_test_cat = torch.tensor(full_input_cat[int(len(full_input_cat) * train_size):], dtype=torch.int).to(device)  

    full_target = torch.tensor(full_target, dtype=torch.float16).to(device) 
    target_train = full_target[:int(len(full_target) * train_size)]
    target_test = full_target[int(len(full_target) * train_size):]

    tscv = BlockingTimeSeriesSplit(n_splits=k_fold, cv_train_size=cv_train_size)
    K_fold = []

    for i, (train_index, valid_index) in enumerate(tscv.split(input_train_num)):
        cv_train_x_num, cv_valid_x_num = input_train_num[train_index], input_train_num[valid_index]
        cv_train_x_cat, cv_valid_x_cat = input_train_cat[train_index], input_train_cat[valid_index]
        cv_train_y, cv_valid_y = target_train[train_index], target_train[valid_index]

        train_data = TensorDataset(cv_train_x_num, cv_train_x_cat, cv_train_y)
        valid_data = TensorDataset(cv_valid_x_num, cv_valid_x_cat, cv_valid_y)

        K_fold.append([train_data, valid_data])

    test_data = TensorDataset(input_test_num, input_test_cat, target_test)

    return int(len(full_input_num) * train_size), K_fold, test_data