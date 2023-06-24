import numpy as np
import warnings
import sys
from math import log10, floor
import GPy
import GPy.util as util
from GPy.util.linalg import tdot

def _unscaled_dist(X, X2=None):
    """
    Compute the Euclidean distance between each row of X and X2, or between
    each pair of rows of X if X2 is None.
    """
    #X, = self._slice_X(X)
    if X2 is None:
        Xsq = np.sum(np.square(X),1)
        r2 = -2.*tdot(X) + (Xsq[:,None] + Xsq[None,:])
        util.diag.view(r2)[:,]= 0. # force diagnoal to be zero: sometime numerically a little negative
        r2 = np.clip(r2, 0, np.inf)
        return np.sqrt(r2)
    else:
        #X2, = self._slice_X(X2)
        X1sq = np.sum(np.square(X),1)
        X2sq = np.sum(np.square(X2),1)
        r2 = -2.*np.dot(X, X2.T) + (X1sq[:,None] + X2sq[None,:])
        r2 = np.clip(r2, 0, np.inf)
        return np.sqrt(r2)

def dist2(X, X2=None):
    
    return np.square(_unscaled_dist(X, X2))


def var_trans(variable, transIdx, transType='exp'):  

    if transType == 'exp':
        return np.where(transIdx, util.functions.clip_exp(variable), variable)    
    elif transType == 'log':
        return np.where(transIdx, np.log(variable), variable) ##unstable log?
    else: 
        sys.exit("The given transType is not implemented yet.")


def complex_log(x):
    
    return np.where(x>0, np.log(x), np.log(x+0j))
        
        
def gradCheck(x, f, fp, model=None, **kwargs):
    
    eps = 1e-6
    
    nparam = len(x)
    deltaf = np.zeros(nparam)
    step = np.zeros(nparam)
    
    for i in range(nparam):
        step[i] = 1.
        fplus = f(x + eps*step, **kwargs)
        fminus = f(x - eps*step, **kwargs)
        deltaf[i] = .5*(fplus-fminus)/eps
        step[i] = 0
    
    gradient = fp(x,**kwargs)
    delta = deltaf - gradient
    
    if model is not None:
        if isinstance(model.param, list):
            paramName = model.paramName
        elif isinstance(model.param, dict):
            paramName = []
            for pname, p in model.param.items():
                if p.ndim > 1:
                    paramName += [
                        "{} {},{}".format(pname, i, j) 
                        for i in range(p.shape[0]) for j in range(p.shape[1]) 
                    ]
                else:
                    paramName += ["{} {}".format(pname, i) for i in range(p.shape[0])]
        paramName += ['beta {}'.format(i) for i in range(model.nout)]
        
    else:
        paramName = ['param {}'.format(i) for i in range(nparam)]


    print("--> Gradient Check!")
    for i in range(nparam):
        print('{: <13} |analytic: {:10.6f} |numerical: {:10.6f} |diff: {:7.4f}'.format(
            paramName[i], gradient[i], deltaf[i], delta[i]))


def compare_exact(first, second):
    
    """Return whether two dicts of arrays are exactly equal"""
    
    if first.keys() != second.keys():
        return False
    return all(np.array_equal(first[key], second[key]) for key in first)

def compare_approximate(first, second):
    
    """Return whether two dicts of arrays are roughly equal"""
    
    if first.keys() != second.keys():
        return False
    return all(np.allclose(first[key], second[key]) for key in first)

def calculate_mse(pred, obs):
    
    return np.sum((pred - obs)**2) / pred.shape[0]