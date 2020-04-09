import numpy as np
import pymc3 as pm
import theano
import theano.tensor as tt
from sklearn.preprocessing import PolynomialFeatures
from scipy.cluster.vq import kmeans, vq
import sys
import random
theano.config.openmp = False

from util import get_lag
from inducingpolicy import inducing_policy0, inducing_policy1, inducing_policy2, inducing_policy3


def treatment_bell(lag, lengthscale, head=2):
    return tt.exp(- (lag - head*lengthscale)**2 / lengthscale**2)


class BasicModel(object):
    def __init__(self, **kwargs):
        self.n = 0
        self.t, self.tt, self.tp = [], [], []
        self.y, self.yt = [], []
        self.tx, self.txt, self.txp = [], [], []
        self.txc, self.txct, self.txcp = [], [], []
        self.x, self.xt, self.xp = [], [], []
        self.txp_idx = []
        self.model = pm.Model()
        self.trace = None
        self.ppc = None
        self.trend = []
        self.feature = kwargs['feature']
        self.covariate = kwargs['covariate']
        self.covariate_sd = kwargs['covariate_sd']
        self.nostd = kwargs['nostd']

    def add(self, training, testing, plotting, trend=None):
        self.n = self.n + 1
        self.trend.append(trend)

        (t,y,tx,x,txc), (tt,yt,txt,xt), (tp,_,txp,xp,txp_idx) = training, testing, plotting
        self.xdim = x.shape[1]
        self.t.append(t)
        self.y.append(y)
        self.tx.append(tx)
        self.x.append(x)
        self.txc.append(txc)

        self.tt.append(tt)
        self.yt.append(yt)
        self.txt.append(txt)
        self.xt.append(xt)

        self.tp.append(tp)
        self.txp.append(txp)
        self.xp.append(xp)
        self.txp_idx.append(txp_idx)

    def preprocess(self):
        # Feature transformation
        xs, xts, xps = [], [], []
        for x, xt, xp in zip(self.x, self.xt, self.xp):
            if 'log' in self.feature:
                x = np.log(x+1)
                xt = np.log(xt+1)
                xp = np.log(xp+1)
            if 'sqrt' in self.feature:
                x = np.sqrt(x)
                xt = np.sqrt(xt)
                xp = np.sqrt(xp)
            if 'poly2' in self.feature:
                poly = PolynomialFeatures(2, include_bias=False, interaction_only=True)
                x = poly.fit_transform(x)
                xt = poly.transform(xt)
                xp = poly.transform(xp)
            self.xdim = x.shape[1]
            xs.append(x)
            xts.append(xt)
            xps.append(xp)
        self.x = xs
        self.xt = xts
        self.xp = xps

        if not self.nostd: # standardize
            X = np.vstack(self.x)
            m, s = X.mean(axis=0), X.std(axis=0)
            self.xmn, self.xstd = m, s
            if np.any(s < 1e-4):
                print('DEBUG: std, ', s)
            self.x = [(x-m)/s for x in self.x]
            self.xt = [(xt-m)/s for xt in self.xt]
            self.xp = [(xp-m)/s for xp in self.xp]

    def build(self):
        pass

    def sample(self, n=500, tune=500, step='NUTS', **kwargs):
        with self.model:
            if 'cores' in kwargs:
                nc = kwargs.get('cores', 2)
                kwargs.pop('cores')
            else:
                nc = 2

            if step == 'Metropolis':
                s = pm.Metropolis(vars=self.model.free_RVs, **kwargs)
                nc = 1
            elif step == 'NUTS':
                s = pm.NUTS(vars=self.model.free_RVs, **kwargs)
            elif step == 'HamiltonianMC':
                s = pm.HamiltonianMC(vars=self.model.free_RVs, **kwargs)
            else:
                s = pm.NUTS(vars=self.model.free_RVs, **kwargs)

            return pm.sample(n, tune=tune, step=s, cores=nc)

    def get_ppc(self, suffix, test_only, delay):
        return []

    def predict(self, trace, n=100, suffix='', test_only=False, delay=True):
        '''
        Note: all patients share one t_s for now.
        '''
        with self.model:
            to_ppc = self.get_ppc(suffix, test_only, delay)
            ppc = pm.sample_ppc(trace, vars=to_ppc, samples=n)
            return ppc


class GPTrendModel(BasicModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.n_inducing_points = kwargs['n_inducing_points']
        self.lengthscale = kwargs['lengthscale']
        self.sparse = kwargs['sparse']
        self.inducing_policy = kwargs['inducing_policy']
        self.gp = []
        self.tu = []
        self.tu_exc = []

    def build_gp(self):
        with self.model:
            tdim = 1
            if self.lengthscale is None:
                #ls_se = pm.HalfFlat('ls_se')
                #ls_se = pm.Normal('ls_se', mu=50, sd=10)
                ls_se = pm.HalfNormal('ls_se', sd=10, shape=self.n) + 10

                #ls_se = pm.Cauchy('ls_se', alpha=50, beta=1, shape=self.n)
                #ls_se = tt.log(np.exp(5) + tt.exp(ls_se)) # Softplus
            else:
                ls_se = [self.lengthscale] * self.n
            nu_se = pm.HalfNormal('nu_se', sd=10, shape=self.n)
            c = pm.HalfNormal('c', sd=10, shape=self.n)

            for i, (t, y, tx, x) in enumerate(zip(self.t, self.y, self.tx, self.x)):
                # Kernel
                K_se = nu_se[i] * pm.gp.cov.ExpQuad(tdim, ls_se[i])
                K_c = pm.gp.cov.Constant(c[i])
                K = K_se + K_c

                mu = pm.gp.mean.Zero()
                if self.n_inducing_points:
                    if self.inducing_policy == 'policy0':
                        tu, tu_exc = inducing_policy0(self.n_inducing_points, t, y)
                    elif self.inducing_policy == 'policy1':
                        tu, tu_exc = inducing_policy1(self.n_inducing_points, t, y)
                    elif self.inducing_policy == 'policy2':
                        tu, tu_exc = inducing_policy2(self.n_inducing_points, t, y)
                    elif self.inducing_policy == 'policy3':
                        tu, tu_exc = inducing_policy3(self.n_inducing_points, t, y, tx, bwin=60, awin=180)
                    self.tu.append(tu)
                    self.tu_exc.append(tu_exc)
                if self.sparse:
                    #gp = pm.gp.MarginalSparse(mean_func=mu, cov_func=K, approx="DTC")
                    gp = pm.gp.MarginalSparse(mean_func=mu, cov_func=K, approx="FITC")
                else:
                    gp = pm.gp.Marginal(mean_func=mu, cov_func=K)

                self.gp.append(gp)

    def get_ppc(self, suffix, test_only, delay):
        to_ppc = []
        for i in range(self.n):
            trend = self.gp[i].conditional('trend{}{}'.format(i,suffix), self.t[i][:,None])
            trend_test = self.gp[i].conditional('trend_test{}{}'.format(i,suffix), self.tt[i][:,None])
            trend_plot = self.gp[i].conditional('trend_plot{}{}'.format(i,suffix), self.tp[i][:,None])
            to_ppc += [trend, trend_test, trend_plot]
        return to_ppc


class IndividualModel(BasicModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.time_uncertainty = kwargs['time_uncertainty']
        self.nointercept = kwargs['nointercept']
        self.txv = []
        self.tr_l = []
        self.tr_h = []
        self.tr_hv = []
        self.treated = []

    def preprocess_tt(self, patient_idx, x, xv=None, add_xv=False):
        '''
        Theano version, plus errors-in-variables.
        poly2 is missing.
        xdim stays the same for now
        '''
        with self.model:
            if add_xv  and xv is None:
                xv = pm.Normal('xv{}'.format(patient_idx), mu=0, sd=self.covariate_sd,
                #xv = pm.Normal('xv{}'.format(patient_idx), mu=1, sd=self.covariate_sd,
                #xv = pm.Laplace('xv{}'.format(patient_idx), mu=1, b=self.covariate_sd,
                    shape=len(self.tx[patient_idx]))
                #xv = 1 / xv # symmetic
                self.xv[patient_idx] = xv
            # add the rv before transformation of x
            #if add_xv:
            #    x = x * xv[:,None]
            # transformation
            if 'log' in self.feature:
                x = tt.log(x+1)
                if add_xv:
                    x = x + xv[:,None]
            if 'sqrt' in self.feature:
                #TODO xv not impl
                x = tt.sqrt(x)
        return x

    def build_treated(self):
        self.build_common_prior()
        if self.covariate: # _tt has no poly2 by now
            self.xv = [None] * self.n
            self.x = [ self.preprocess_tt(i, self.x[i], add_xv=True) for i in range(self.n) ]
        else:
            self.preprocess()
        self.build_prior()

        with self.model:
            for i, (t, y, tx, x) in enumerate(zip(self.t, self.y, self.tx, self.x)):
                lsp = self.wl.tag.test_value.shape
                if len(lsp) == 2:
                    tr_l = tt.dot(x, self.wl[i]) + self.bl[i]
                else:
                    tr_l = tt.dot(x, self.wl) + self.bl[i]
                tr_l = tt.log(1 + tt.exp(tr_l)) # Softplus

                tr_h = tt.dot(x, self.wh[i]) + self.bh[i]
                tr_h = tt.log(1 + tt.exp(tr_h)) # Softplus
                #tr_hv = tr_h + self.sigma_h[i] * pm.Normal('tr_h_v{}_'.format(i), mu=0, sd=1, shape=len(tx))
                tr_hv = tr_h

                treated = tt.zeros(y.shape[0])
                lag = get_lag(t, tx)
                if self.time_uncertainty:
                    txv = self.delay[i] + self.sigma_tx[i] * pm.Normal('txv{}_'.format(i), mu=0, sd=1, shape=(len(tx)))
                for j in range(tx.shape[0]):
                    if self.time_uncertainty:
                        lagv = lag[:,j] + txv[j]
                    else:
                        lagv = lag[:,j]
                    tr_i = tr_hv[j] * treatment_bell(lagv, tr_l[j])
                    treated = treated + tr_i

                if self.time_uncertainty:
                    self.txv.append(txv)
                self.tr_l.append(tr_l)
                self.tr_h.append(tr_h)
                self.tr_hv.append(tr_hv)
                self.treated.append(treated)

    def build_common_prior(self):
        ''' No use of xdim allowed. '''
        n = self.n
        with self.model:
            if self.nointercept:
                self.bl = np.zeros(n)
                self.bh = np.zeros(n)
            else:
                self.bl = pm.HalfNormal('bl', sd=3, shape=n)
                #self.bl = pm.HalfCauchy('bl', beta=1, shape=n) + 10

                #self.bh = pm.Normal('bh', mu=0, sd=3, shape=n)
                self.bh = pm.HalfNormal('bh', sd=3, shape=n)

            #self.sigma_h = pm.HalfNormal('sigma_h', sd=1, shape=n)
            #self.sigma_h = pm.HalfCauchy('sigma_h', beta=0.5, shape=n)
            if self.time_uncertainty:
                self.delay = pm.Normal('delay', mu=0, sd=10, shape=n)
                self.sigma_tx = pm.HalfNormal('sigma_tx', sd=10, shape=n)

            #self.sigma = pm.HalfCauchy('sigma', 2.5, shape=n)
            #self.sigma = pm.HalfNormal('sigma', sd=1, shape=n)
            self.sigma = pm.HalfNormal('sigma', sd=0.1, shape=n)

    def build_prior(self):
        n, xdim = self.n, self.xdim
        with self.model:
            #self.wl = pm.Cauchy('wl', alpha=0, beta=1, shape=(xdim))
            self.wl = pm.Normal('wl', mu=0, sd=5, shape=(n, xdim))
            self.wh = pm.Normal('wh', mu=0, sd=5, shape=(n, xdim))

    def get_ppc(self, suffix, test_only, delay):
        to_ppc = []
        for i in range(self.n):
            to_ppc += self.__get_ppc_for_patient(i, suffix, test_only, delay)
        return to_ppc

    def __cal_treated(self, patient_idx, which='training', delay=False):
        i = patient_idx
        if which == 'training':
            t, tx, x = self.t[i], self.tx[i], self.x[i]
        elif which == 'testing':
            t, tx, x = self.tt[i], self.txt[i], self.xt[i]
        else:
            t, tx, x = self.tp[i], self.txp[i], self.xp[i]
        txp_train_idx = set(self.txp_idx[i])

        lag = get_lag(t, tx)
        treated = tt.zeros(t.shape)
        tr_h = tt.dot(x, self.wh[i]) + self.bh[i]
        tr_h = tt.log(1 + tt.exp(tr_h)) # Softplus

        lsp = self.wl.tag.test_value.shape
        if len(lsp) == 2:
            tr_l = tt.dot(x, self.wl[i]) + self.bl[i]
        else:
            tr_l = tt.dot(x, self.wl) + self.bl[i]
        tr_l = tt.log(1 + tt.exp(tr_l)) # Softplus

        for j in range(tx.shape[0]):
            if self.time_uncertainty and (which=='training' or which=='plotting' and j in txp_train_idx):
                lagv = lag[:,j] + self.txv[i][j]
                tr_i = tr_h[j] * treatment_bell(lagv, tr_l[j])
            else:
                if self.time_uncertainty and delay:
                    lagv = lag[:,j] + self.delay[i]
                else:
                    lagv = lag[:,j]
                tr_i = tr_h[j] * treatment_bell(lagv, tr_l[j])
            treated = treated + tr_i

        return treated, tr_h, tr_l

    def __get_ppc_for_patient(self, patient_idx, suffix, test_only, delay):
        i = patient_idx
        if self.covariate:
            self.xt[i] = self.preprocess_tt(i, self.xt[i], add_xv=False)
            self.xp[i] = self.preprocess_tt(i, self.xp[i], add_xv=False)

        if not test_only:
            treated, h, l       = self.__cal_treated(patient_idx, which='training')
            treated = pm.Deterministic('treated{}{}'.format(i,suffix), treated)
            h = pm.Deterministic('h{}{}'.format(i,suffix), h)
            l = pm.Deterministic('l{}{}'.format(i,suffix), l)

            treated_p, h_p, l_p = self.__cal_treated(patient_idx, which='plotting')
            treated_p = pm.Deterministic('treated_plot{}{}'.format(i,suffix), treated_p)
            h_p = pm.Deterministic('h_plot{}{}'.format(i,suffix), h_p)
            l_p = pm.Deterministic('l_plot{}{}'.format(i,suffix), l_p)

            hv = pm.Deterministic('hv{}{}'.format(i,suffix), self.tr_hv[i])

        treated_t, h_t, l_t = self.__cal_treated(patient_idx, which='testing', delay=delay)
        treated_t = pm.Deterministic('treated_test{}{}'.format(i,suffix), treated_t)
        h_t = pm.Deterministic('h_test{}{}'.format(i,suffix), h_t)
        l_t = pm.Deterministic('l_test{}{}'.format(i,suffix), l_t)

        if not test_only:
            to_ppc = [treated, h, l, treated_t, h_t, l_t, treated_p, h_p, l_p, hv]
        else:
            to_ppc = [treated_t, h_t, l_t]

        if not test_only and self.time_uncertainty:
            shift = pm.Deterministic('shift{}{}'.format(i,suffix), self.txv[i])
            to_ppc.append(shift)

        return to_ppc


class GPTrendIndividualModel(GPTrendModel, IndividualModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self):
        self.build_gp()
        self.build_treated()

        with self.model:
            for i, (t, y, tx, x) in enumerate(zip(self.t, self.y, self.tx, self.x)):
                # Likelihood
                if self.n_inducing_points and self.sparse:
                    self.gp[i].marginal_likelihood('y_obs{}'.format(i), X=t[:,None],
                            Xu=self.tu[i][:,0][:,None], y=y-self.treated[i], noise=self.sigma[i])
                else:
                    self.gp[i].marginal_likelihood('y_obs{}'.format(i), X=t[:,None],
                            y=y-self.treated[i], noise=self.sigma[i])

    def get_ppc(self, suffix, test_only, delay):
        ext1 = GPTrendModel.get_ppc(self, suffix, test_only, delay)
        ext2 = IndividualModel.get_ppc(self, suffix, test_only, delay)
        return ext1 + ext2


class HierModel(IndividualModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hier_sd_h = kwargs['hier_sd_h']
        self.hier_sd_ls = kwargs['hier_sd_ls']

    def build_prior(self):
        n, xdim = self.n, self.xdim

        with self.model:
            #self.wl = pm.Normal('wl', sd=10, shape=(xdim))
            #self.wl = pm.Cauchy('wl', alpha=0, beta=2, shape=(n, xdim))

            self.wl0 = pm.Normal('wl0', mu=0, sd=5, shape=xdim)
            #self.wl0 = pm.Cauchy('wl0', alpha=0, beta=1, shape=xdim)
            #self.sigma_wl = pm.HalfCauchy('sigma_wl', beta=0.1, shape=xdim)
            self.sigma_wl = pm.HalfNormal('sigma_wl', sd=self.hier_sd_ls, shape=xdim)
            self.wl = pm.MvNormal('wl', mu=self.wl0,
                    cov=tt.diag(self.sigma_wl), shape=(n, xdim))

            self.wh0 = pm.Normal('wh0', mu=0, sd=5, shape=xdim)
            #self.sigma_wh = pm.HalfCauchy('sigma_wh', beta=1, shape=xdim)
            self.sigma_wh = pm.HalfNormal('sigma_wh', sd=self.hier_sd_h, shape=xdim)
            self.wh = pm.MvNormal('wh', mu=self.wh0,
                    cov=tt.diag(self.sigma_wh), shape=(n, xdim))
                    #cov=tt.diag(np.array([5]*xdim)), shape=(n, xdim))


class GPTrendHierModel(GPTrendModel, HierModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self):
        self.build_gp()
        self.build_treated()

        with self.model:
            for i, (t, y, tx, x) in enumerate(zip(self.t, self.y, self.tx, self.x)):
                # Likelihood
                if self.n_inducing_points:
                    self.gp[i].marginal_likelihood('y_obs{}'.format(i), X=t[:,None],
                            Xu=self.tu[i][:,0][:,None], y=y-self.treated[i], noise=self.sigma[i])
                else:
                    self.gp[i].marginal_likelihood('y_obs{}'.format(i), X=t[:,None],
                            y=y-self.treated[i], noise=self.sigma[i])

    def get_ppc(self, suffix, test_only, delay):
        ext1 = GPTrendModel.get_ppc(self, suffix, test_only, delay)
        ext2 = HierModel.get_ppc(self, suffix, test_only, delay)
        return ext1 + ext2
