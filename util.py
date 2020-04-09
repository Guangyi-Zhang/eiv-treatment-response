import pandas as pd
import numpy as np


def dataset_from_df(df, nut):
    y_idx = ~df['glucose'].isnull()
    y = df['glucose'][y_idx].values
    t = df['time'][y_idx].values

    # not null && nut > threshold
    rx_idx = ((~df[nut[0]].isnull()) & (df[nut].sum(axis=1)>10)).values
    rx = df[nut][rx_idx].values
    tx = df['time'][rx_idx].values
    yx = ((df['glucose'].fillna(method='bfill') + df['glucose'].fillna(method='ffill')) / 2)[rx_idx].values

    txi = np.nonzero(rx_idx)[0]
    ti = np.nonzero(y_idx)[0]

    return (t,y,ti), (tx,yx,rx,txi)


def get_lag(t, tx):
    d = t.reshape((-1,1)) - tx.reshape((1, -1))
    return d
