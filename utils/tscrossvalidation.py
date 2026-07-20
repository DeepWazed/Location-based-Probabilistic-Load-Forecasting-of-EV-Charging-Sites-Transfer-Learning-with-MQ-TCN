import numpy as np
import torch
from torch.utils.data import TensorDataset


class BlockingTimeSeriesSplit():
    def __init__(self, n_splits,cv_train_size):
        self.n_splits = n_splits
        self.cv_train_size=cv_train_size
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

def ts_kfold_cv(df, cat_features, n_past, n_future, test_size=0.1, k_fold=4, cv_train_size=0.8):

    cols = df.columns.tolist()
    cols.insert(0, cols.pop(cols.index('Smoothed_kWhDeliveredTotal')))
    df = df[cols]
    df = df.drop('doneChargingTime', axis=1)
    df_cat=df[cat_features].astype(int)
    df_num=df.drop(cat_features, axis=1)
    df_num=df_num.astype(np.float64)

    df_cat = df_cat.to_numpy()
    df_num = df_num.to_numpy()

    X_num = []
    X_cat = []
    y = []

    for i in range(n_past, len(df) - n_future + 1):
        X_num.append(df_num[i - n_past:i, :])
        X_cat.append(df_cat[i - n_past:i, :])
        y.append(df_num[i:i + n_future, 0])

    full_input_num,full_input_cat, full_target = np.array(X_num), np.array(X_cat), np.array(y)

    # print('Input continuous data has shape == {}.'.format(full_input_num.shape))
    # print('Input categorical data has shape == {}.'.format(full_input_cat.shape))
    # print('Target data has shape == {}.'.format(full_target.shape))

    """
    Split the dataset into train, and stand alone test sets
    """
    train_size = 1 - test_size
    full_input_num = np.asarray(full_input_num)
    full_input_cat = np.asarray(full_input_cat)
    full_input_num = full_input_num.astype(float)
    full_input_cat = full_input_cat.astype(int)
    input_train_num = full_input_num[:int(len(full_input_num) * train_size)]
    input_train_cat = full_input_cat[:int(len(full_input_cat) * train_size)]
    input_test_num = full_input_num[int(len(full_input_num) * train_size):]
    input_test_cat = full_input_cat[int(len(full_input_cat) * train_size):]

    full_target = np.asarray(full_target)
    full_target = full_target.astype(float)
    target_train = full_target[:int(len(full_target) * train_size)]
    target_test = full_target[int(len(full_target) * train_size):]

    """
    split train set into k folds
    """
    tscv = BlockingTimeSeriesSplit(n_splits=k_fold, cv_train_size=cv_train_size)
    K_fold = []

    for i, (train_index, valid_index) in enumerate(tscv.split(input_train_num)):
        cv_train_x_num, cv_valid_x_num = input_train_num[train_index], input_train_num[valid_index]
        cv_train_x_cat, cv_valid_x_cat = input_train_cat[train_index], input_train_cat[valid_index]
        cv_train_y, cv_valid_y = target_train[train_index], target_train[valid_index]
        # print(f"Fold {i + 1}: Train size {cv_train_x_num.shape[0]}, Valid size {cv_valid_x_num.shape[0]}")

        "as tensor"
        train_data = TensorDataset(torch.from_numpy(cv_train_x_num).float(),
                                   torch.from_numpy(cv_train_x_cat).int(),
                                   torch.from_numpy(cv_train_y).float())

        valid_data = TensorDataset(torch.from_numpy(cv_valid_x_num).float(),
                                   torch.from_numpy(cv_valid_x_cat).int(),
                                   torch.from_numpy(cv_valid_y).float())

        K_fold.append([train_data, valid_data])

    test_start_index = int(len(full_input_num) * train_size)

    # print(f"Test set starts from index {test_start_index + n_past} and ends at index {test_end_index + n_past + n_future -1}.") # To see the values

    # print(f"Test set starts from index {test_start_index} and ends at index {test_end_index}.")


    test_data = TensorDataset(torch.from_numpy(input_test_num).float(),
                              torch.from_numpy(input_test_cat).int(),
                              torch.from_numpy(target_test).float())
    # print(target_test.shape)

    # print(target_test[0])
    # print("Staring:\n")
    # print(input_test_num[0])
    # print("ENDing:\n")
    # print(input_test_num[-1])


    return test_start_index, K_fold, test_data