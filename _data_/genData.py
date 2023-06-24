import sys
import numpy as np
import numpy.random as npr
import GPy
from scipy.ndimage import gaussian_filter1d

from blockCompute import spMutliFullBlockCompute


def genData(dataset, seedVal=None, **kwargs):
    
    
    # Set seed
    if seedVal is None: seedVal = int(1e+6)
    np.random.seed(seedVal)
    print('Seed is set to {}'.format(seedVal))
    
    
    if dataset == 'ggToy':
        
        ######## read arguments ########
        nout = kwargs['nout']
        nlf = kwargs['nlf']
        precisionG = kwargs['precisionG']
        sensitivity = kwargs['sensitivity']
        precisionU = kwargs['precisionU']
        N = kwargs['N']
        
        if 'bias' not in kwargs:
            bias = np.repeat(0, nout)
        else: bias = kwargs['bias']
        
        if nout != len(precisionG): sys.exit('Wrong input')
        if len(precisionG) != len(sensitivity): sys.exit('Wrong input')
        if len(precisionU) > 1: sys.exit('Not implemented')
        
        
        ######## Generate samples #######
        x = np.linspace(-1, 1, num=N)[:, None]
        X = dict()
        for i in range(nout): X[i] = x
        
        mu = np.repeat(0, N*nout)[:, None]
        for i in range(nout):
            if bias[i] != 0: 
                mu[i*N:(i+1)*N, ] = np.repeat(bias[i], N)[:, None]
        
        fullCov = spMutliFullBlockCompute(
            nout=nout, X=X, precisionU=precisionU, precisionG=precisionG, 
            sensitivity=sensitivity
        )
        
        Ytrue = np.random.multivariate_normal(mu[:,0], fullCov)
        Ytrue = Ytrue.reshape((N, nout), order='F')
        Ytrue = {i: Ytrue[:, i][:,None] for i in range(Ytrue.shape[1])}
        Yobs = dict()
        for i in range(nout):
            Yobs[i] = Ytrue[i] + \
                0.1*np.std(Ytrue[i])*np.random.multivariate_normal(
                    np.zeros(N), np.identity(N))[:,None]
        
        return dict(X=X, y=Yobs, ytrue=Ytrue, bias=bias)
        
    elif dataset == 'ggToyMissing':

        ######## Get the dataset ########        
        bdata = genData(dataset='ggToy', seedVal=seedVal, **kwargs)
        
        ######## read arguments #########
        nout = kwargs['nout']
        Nobs = kwargs['Nobs']
        N = kwargs['N']
        missingLB, missingUB = kwargs['missing_range']
        if 'diffX' not in kwargs: diffX = False
        else: diffX = kwargs['diffX']
        
        ######## observations ###########
        missingUnit = [0]
        obsIdx = {i: np.sort(np.random.choice(range(N), size=Nobs, replace=False))
                  for i in range(nout)}
        if not diffX : 
            for i in range(nout): obsIdx[i] = obsIdx[0]
        for i in range(nout):
            if i in missingUnit: 
                obsIdx[i] = obsIdx[i][
                    (missingLB > bdata['X'][i].flatten()[obsIdx[i]]) | \
                         (bdata['X'][i].flatten()[obsIdx[i]] > missingUB)]
        
        gdata = dict(
            X = {i: bdata['X'][i][obsIdx[i]] for i in range(nout)},
            y = {i: bdata['y'][i][obsIdx[i]] for i in range(nout)},
            true = bdata
        )

        return gdata
    
    elif dataset == 'nonlinear':
        
        ######## read arguments #########
        nout = kwargs['nout']
        N = kwargs['N']
        param_w1 = kwargs['param_w1']
        param_w2 = kwargs['param_w2']
        noise = kwargs['noise']


        ######## Generate Samples ########
        x = np.linspace(0, 10, num=N)[:, None]
        X = dict()
        for i in range(nout): X[i] = x

        w1 = npr.normal(loc=param_w1[0], scale=param_w1[1], size=nout)
        w2 = npr.uniform(low=param_w2[0], high=param_w2[1], size=nout)

        Ytrue, Yobs = dict(), dict()
        for i in range(nout): 
            Ytrue[i] = (.3*(X[i])**2 - 2*np.sin(w1[i]*np.pi*(X[i])) + w2[i])
            Yobs[i] = Ytrue[i] + noise*npr.multivariate_normal(
                np.zeros(N), np.identity(N)
            )[:, None]

        return dict(X=X, y=Yobs, ytrue=Ytrue)

    elif dataset == 'convolution':

        ######## read arguments #########
        nout = kwargs['nout']
        N = kwargs['N']
        param_u = kwargs['param_u']
        param_flen = kwargs['param_flen']
        param_fvar = kwargs['param_fvar']
        noise = kwargs['noise']

        ####### generate samples ########
        x = np.linspace(-1, 1, num=N)[:, None]
        X = dict()
        for i in range(nout): X[i] = x
        k_uu = GPy.kern.RBF(
            input_dim=1, lengthscale=param_u[0], variance=param_u[1]
        ).K(x,x)
        u = npr.multivariate_normal(np.zeros((N)),k_uu,1).T

        Ytrue, Yobs = dict(), dict()
        for i in range(nout):
            flen = npr.uniform(param_flen[0], param_flen[1])
            fvar = npr.uniform(param_fvar[0], param_fvar[1])
            Ytrue[i] = fvar * gaussian_filter1d(
                input=u.squeeze(), sigma=flen)[:,None]
            Yobs[i] = Ytrue[i] + noise * ( 
                np.random.multivariate_normal(
                    np.zeros(N), np.identity(N))[:,None]
            )

        return dict(X=X, y=Yobs, ytrue=Ytrue)

def genMissing(bdata, missing_range, Nobs=None, missingUnit=[0], 
        diffX=False, seedVal=None
    ):

    # Set seed
    if seedVal is not None: 
        np.random.seed(seedVal)
        print('Seed is set to {}'.format(seedVal))
    
    nout = len(bdata['X'])
    N = {k: val.shape[0] for k, val in bdata['X'].items()}
    missingLB, missingUB = missing_range

    if Nobs is not None:
        obsIdx = {
            k: np.sort(np.random.choice(range(val), size=Nobs, replace=False))
            for k, val in N.items()
        }
    else:
        obsIdx = {k: np.arange(val) for k, val in N.items()}
    
    # if not diffX : 
    #     for i in obsIdx.keys(): obsIdx[i] = obsIdx[0]
    
    for i in range(nout):
        if i in missingUnit: 
            obsIdx[i] = obsIdx[i][
                (missingLB > bdata['X'][i].flatten()[obsIdx[i]]) | \
                        (bdata['X'][i].flatten()[obsIdx[i]] > missingUB)]

    gdata = dict(
            X = {i: bdata['X'][i][obsIdx[i]] for i in range(nout)},
            y = {i: bdata['y'][i][obsIdx[i]] for i in range(nout)},
            true = bdata,
            selected_index = obsIdx
        )
    
    return gdata
        
def cen_to_fed(bdata, type='cross-unit'):

    nout = len(bdata['X'])
    if type == 'cross-unit':
        
        fed_data_all = dict()
        for i_out in range(nout):
            fed_data_all[i_out] = dict()
            fed_data_all[i_out]['X'] = {0: bdata['X'][i_out]}
            fed_data_all[i_out]['y'] = {0: bdata['y'][i_out]}
            fed_data_all[i_out]['true'] = dict(
                X=bdata['true']['X'][i_out], y=bdata['true']['ytrue'][i_out]
            )

        return fed_data_all
        
        
        
        