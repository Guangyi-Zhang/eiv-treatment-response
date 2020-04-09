import numpy as np
import pandas as pd
from scipy import interpolate

import matplotlib
import matplotlib.pyplot as plt


def metric1(y, trend, treated):
    reg = treated != 0
    #return (trend - y)[reg].var() / trend[reg].var()
    return (trend)[reg].var() / y[reg].var()


def metric2(y, trend, treated):
    reg = treated != 0
    #sgn = np.sign(treated[reg].sum()) # np.sign((m.y[i] - y_hat)[reg].sum())
    #return sgn * treated[reg].var() / (trend - y)[reg].var()
    return (trend+treated)[reg].var() / y[reg].var() - metric1(y, trend, treated)


def metric_mse(y, trend, treated, t, tx, window=None, active=False, realtime=False):
    '''
    Treatment is constrained to positive.
    It turns out active encourages narrow bumps.
    '''
    if active:
        idx = treated > 1e-5
    else:
        idx = treated > -1

    if window is not None:
        idx = t < -1 # all False
        for tx_ in tx:
            idx |= (t >= tx_-window[0]) & (t <= tx_+window[1])

    if realtime:
        x = np.arange(len(y))
        x_ = np.concatenate([x[~idx], [len(y)]])
        y_ = np.concatenate([y[~idx], [0]])
        f = interpolate.interp1d(x_, y_, kind='previous')
        trd = f(x)[idx]
    else:
        trd = trend[idx]

    yhat = trd + treated[idx]
    return ((y[idx] - yhat)**2).mean()


def metric_var(y, trend, treated, t, tx, window=None):
    '''
    Treatment is constrained to positive.
    '''
    idx = treated > -1

    if window is not None:
        idx = t < -1 # all False
        for tx_ in tx:
            idx |= (t >= tx_-window[0]) & (t <= tx_+window[1])

    trd = trend[idx]
    tr = treated[idx]
    yhat = trd + tr
    return np.abs(yhat.var() - y[idx].var())
