import torch
import numpy as np
import pandas as pd

class DataPreparation:
    def __init__(self, n_future, n_past, n_categorical_features):
        self.n_future = n_future
        self.n_past = n_past
        self.n_categorical_features = n_categorical_features

    def prepare_data(self, dataframe, device):
        cols = dataframe.columns.tolist()
        cols.insert(0, cols.pop(cols.index('Smoothed_kWhDeliveredTotal')))
        dataframe = dataframe[cols]

        dataframe = dataframe.drop('doneChargingTime', axis=1)

        # print(dataframe.head(5))
        # print("UniqueFee_Free values:", dataframe['UniqueFee_Free'].unique())

        data = dataframe.to_numpy()  
        # print("Data types of each column:")
        # for col, dtype in zip(dataframe.columns, dataframe.dtypes):
        #     print(f"{col}: {dtype}")             

        X = []
        y = []
        for i in range(self.n_past, len(data) - self.n_future + 1):
            X.append(data[i - self.n_past:i, :])
            y.append(data[i:i + self.n_future, 0])

        X, y = np.array(X), np.array(y)

        
        # X_con = X[:, :, :-self.n_categorical_features]
        # X_cat = X[:, :, -self.n_categorical_features:].astype(np.int32)
        
        X_con = X[:, :, :-self.n_categorical_features].astype(np.float16)  
        X_cat = X[:, :, -self.n_categorical_features:].astype(np.int32) 

        # print("X_con dtype:", X_con.dtype)
        # print("X_cat dtype:", X_cat.dtype) 

        # print(X.dtype)

        X_tensor = torch.tensor(X, dtype=torch.float16).to(device)  
        X_con_tensor = torch.tensor(X_con, dtype=torch.float16).to(device) 
        X_cat_tensor = torch.tensor(X_cat, dtype=torch.int).to(device)  
        y_tensor = torch.tensor(y, dtype=torch.float16).to(device)

        return X_tensor, X_con_tensor, X_cat_tensor, y_tensor