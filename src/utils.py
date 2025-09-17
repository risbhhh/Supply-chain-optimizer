import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def group_series_to_matrix(df, sku, date_col="date", value_col="demand"):
    sub = df[df['sku'] == sku].sort_values(date_col)
    return sub[value_col].values.astype(float)


class SeriesScaler:
    def __init__(self):
        self.scaler = StandardScaler()

    def fit_transform(self, series):
        series = series.reshape(-1, 1)
        scaled = self.scaler.fit_transform(series).flatten()
        return scaled

    def inverse_transform(self, scaled):
        arr = np.array(scaled).reshape(-1, 1)
        inv = self.scaler.inverse_transform(arr).flatten()
        return inv
