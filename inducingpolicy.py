import numpy as np
import pymc3 as pm


def inducing_policy3(n_inducing_points, t, y, tx, bwin=60, awin=180):
    '''
    uniform excluding the regions of treatment
    '''
    idx_t = t > -1 # all True
    for tx_ in tx:
        idx_t &= (t > tx_+awin) | (t < tx_-bwin)
    t_valid = t[idx_t]

    idx = np.array([i_ for i_ in range(0, len(t_valid), len(t_valid)//n_inducing_points)])
    tu = np.stack([t_valid[idx], t_valid[idx]], 1)
    return tu, None


def inducing_policy2(n_inducing_points, t, y, window=10*60):
    '''
    kmeans([t,y]), then remove y>=window_mean
    '''
    feat = np.stack([t, y], 1)
    tu = pm.gp.util.kmeans_inducing_points(n_inducing_points, feat)

    idx, idx_exc = [], []
    for i, (t_, y_) in enumerate(tu):
        mean = y[(t > (t_-window//2)) & (t < (t_+window//2))].mean()
        if y_ < mean:
            idx.append(i)
        else:
            idx_exc.append(i)
    return tu[np.array(idx)], tu[np.array(idx_exc)]


def inducing_policy1(n_inducing_points, t, y):
    '''
    kmeans([t,y]), then remove y>=mean
    '''
    feat = np.stack([t, y], 1)
    tu = pm.gp.util.kmeans_inducing_points(n_inducing_points, feat)
    idx = tu[:,1] < y.mean()
    return tu[idx,:], tu[~idx,:]


def inducing_policy0(n_inducing_points, t, y):
    '''
    uniform
    '''
    idx = np.array([i_ for i_ in range(0, len(t), len(t)//n_inducing_points)])
    tu = np.stack([t[idx], y[idx]], 1)
    return tu, None
