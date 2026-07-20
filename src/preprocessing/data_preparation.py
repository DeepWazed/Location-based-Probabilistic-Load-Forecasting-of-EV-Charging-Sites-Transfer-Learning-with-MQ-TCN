import numpy as np
import pandas as pd


class DataPreparation:
    def __init__(self, n_future, n_past, n_categorical_features):
        self.n_future = n_future
        self.n_past = n_past
        self.n_categorical_features = n_categorical_features

    def prepare_data(self, dataframe):
        
        cols = dataframe.columns.tolist()
        cols.insert(0, cols.pop(cols.index('Smoothed_kWhDeliveredTotal')))
        dataframe = dataframe[cols]

        data = dataframe.drop('doneChargingTime', axis=1).to_numpy()  

        X = []
        y = []
        for i in range(self.n_past, len(data) - self.n_future + 1):
            X.append(data[i - self.n_past:i, :])
            y.append(data[i:i + self.n_future, 0])

        X, y = np.array(X), np.array(y)
        
        ## adding extra: Might need to remove if not works
        
        X_con = X[:, :, :-self.n_categorical_features]
        X_cat = X[:, :, -self.n_categorical_features:].astype(int)
        

        return X, X_con, X_cat, y

