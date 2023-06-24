import sys
import numpy as np
from GPtools.kernCompute import ggKernCompute, ggxggKernCompute

def spMutliFullBlockCompute(nout, X, precisionU, precisionG, sensitivity):
    
    if (nout != len(X)) or (nout != len(precisionG)) or (nout != len(sensitivity)) :
        sys.exit('incorrect dim- number of output: {}'.format(nout))
    
    N_all = np.concatenate(list(X.values()), 0).shape[0]
    K = np.zeros((N_all, N_all))
    
    startOne = 0; endOne = X[0].shape[0]
    for i in range(nout):
        
        # compute block diagonal
        H = [precisionU[0], precisionG[i], sensitivity[i]]
        K[startOne:endOne, startOne:endOne] = ggKernCompute(H=H, x=X[i], x2=X[i])['K']
        
        # compute block off-diagonal
        startTwo = 0; endTwo = X[0].shape[0]
        if i > 0:
            for j in range(i):
                H = [precisionG[i], precisionG[j], precisionU[0], sensitivity[i], sensitivity[j]]
                K[startOne:endOne, startTwo:endTwo] = ggxggKernCompute(H=H, x=X[i], x2=X[j])['K']
                K[startTwo:endTwo, startOne:endOne] = K[startOne:endOne, startTwo:endTwo].T
                startTwo = startTwo + X[j].shape[0]; endTwo = endTwo + X[j+1].shape[0]
        
        if i < nout-1 : 
            startOne = startOne + X[i].shape[0]; endOne = endOne + X[i+1].shape[0]
    
    return K
