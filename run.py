import pymc3 as pm
import theano
import theano.tensor as tt
theano.config.openmp = False

import numpy as np
import pandas as pd
import pickle
import functools
import itertools
import sys
import argparse

from model import GPTrendIndividualModel, GPTrendHierModel
from util import dataset_from_df, get_lag


nut = ['STARCH', 'SUGAR', 'FIBC', 'FAT', 'PROT']
low, high = 0, 24*60


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default='test')
    parser.add_argument("--model", type=str, default='IndividualModel')
    parser.add_argument("--trend", type=str, default='GPTrend')
    parser.add_argument("--n_inducing_points", type=int, default=None)
    parser.add_argument("--lengthscale", type=int, default=None)
    parser.add_argument("--ids", type=str, default='1,2')
    parser.add_argument("--step", type=str, default='NUTS')
    parser.add_argument("--n_sample", type=int, default=500)
    parser.add_argument("--n_tune", type=int, default=500)
    parser.add_argument("--sparse", action='store_true')
    parser.add_argument("--fast", action='store_true')
    parser.add_argument("--inducing_policy", type=str, default='policy1')
    parser.add_argument("--time_uncertainty", action='store_true')
    parser.add_argument("--feature", type=str, default='')
    parser.add_argument("--testset", type=str, default=None)
    parser.add_argument("--covariate", action='store_true')
    parser.add_argument("--covariate_sd", type=float, default=0.01)
    parser.add_argument("--skipsample", action='store_true')
    parser.add_argument("--days", type=int, default=3)
    parser.add_argument("--nppc", type=int, default=100)
    parser.add_argument("--target_accept", type=float, default=0.8)
    parser.add_argument("--nostd", action='store_true')
    parser.add_argument("--dataset", type=str, default='dataset/public_dataset.csv')
    parser.add_argument("--nointercept", action='store_true')
    parser.add_argument("--hier_sd_h", type=float, default=1)
    parser.add_argument("--hier_sd_ls", type=float, default=3)
    return parser.parse_args()


def load_data(args):
    glucose = pd.read_csv(args.dataset)
    ids = np.asarray([args.ids.split(',')], int).flatten()
    dfs = [glucose[glucose['id']==_id] for _id in ids]
    return ids, dfs


def choose_model(args):
    args_dict = vars(args)
    if args.model == 'GPTrendIndividualModel':
        m = GPTrendIndividualModel(**args_dict)
    elif args.model == 'GPTrendHierModel':
        m = GPTrendHierModel(**args_dict)
    else:
        raise Exception('Unknown model: {}'.format(args.model))
    return m


def prepare_dataset(id_, df, args):
    (t,y,ti), (tx,yx,x,txi) = dataset_from_df(df, nut)

    if args.testset == 'day':
        tp = np.linspace(low, high*3, 500) # for plotting

        t_beg_test = 60*24*(args.days)
        t_end_train = 60*24*(args.days)
        idx_test = t > t_beg_test
        idx_train = t <= t_end_train
        idx_xtest = tx > t_beg_test
        idx_xtrain = tx <= t_end_train
        return (t[idx_train], y[idx_train], tx[idx_xtrain], x[idx_xtrain]),\
                (t[idx_test], y[idx_test], tx[idx_xtest], x[idx_xtest]),\
                (tp, None, tx, x, idx_xtrain)
    else:
        raise Exception('Unknown testset: {}'.format(args.testset))


def build_model(m, args, ids, dfs):
    for id_, df_ in zip(ids, dfs):
        (t,y,tx,x), (tt,yt,txt,xt), (tp, _, txp, xp, txp_idx) = prepare_dataset(id_, df_, args)
        m.add((t,y,tx,x,None), (tt,yt,txt,xt), (tp,None,txp,xp,txp_idx))

    m.build()


if __name__ == '__main__':
    args = parse_args()
    print('task:', args.task)
    args_dict = vars(args)
    if args.fast:
        theano.config.mode = 'FAST_COMPILE'

    ################
    # Train
    ################
    if args.skipsample:
        store = pickle.load(open('trace/{}.pkl'.format(args.task), 'rb'))
        ids = store['ids']
        dfs = store['dfs']
    else:
        ids, dfs = load_data(args)
    m = choose_model(args)
    build_model(m, args, ids, dfs)
    if args.skipsample:
        trace = store['trace']
    else:
        if args.step == 'HMC':
            trace = m.sample(n=args.n_sample, tune=args.n_tune, step='HamiltonianMC', target_accept=args.target_accept)
        elif args.step == 'Metropolis':
            trace = m.sample(n=args.n_sample, tune=args.n_tune, step='Metropolis')
        else:
            trace = m.sample(n=args.n_sample, tune=args.n_tune, step='NUTS')

    store = {'trace': trace,
            'ids': ids,
            'dfs': dfs,
            'model': m,
            }
    pickle.dump(store, open('trace/{}.pkl'.format(args.task), 'bw'))

    ppc = m.predict(trace, n=args.nppc)
    store = {'ppc': ppc,
            }
    pickle.dump(store, open('trace/{}_ppc.pkl'.format(args.task), 'bw'))
