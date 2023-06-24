import numpy as np
from GPtools.utils import dist2


def ggKernCompute(H, x, x2=None):
    
    if x2 is None: x2 = x
    
    Pr = H[0]
    Pqr = H[1]
    sensitivity = H[2]
    Prinv = 1/Pr
    Pqinv = 1/Pqr
    Pinv = Prinv + 2*Pqinv
    
    P = 1/Pinv
    
    n2 = dist2(x, x2)
    Kbase = np.exp(-.5*P*n2)
    
    K = sensitivity**2 * Kbase
    
    return dict(K=K, Kbase=Kbase, Prinv=Prinv, Pqinv=Pqinv, P=P, n2=n2)


def ggKernDiagCompute(H, x, x2=None):
    
    if x2 is None: x2 = x
    
    sensitivity = H
    
    K = sensitivity ** 2 * np.ones((x.shape[0],1))
    
    return dict(K=K, sens=sensitivity)


def ggxggKernCompute(H, x, x2=None):
    
    if x2 is None: x2 = x
    
    Pqr = H[0]
    Psr = H[1]
    Pr = H[2]
    Pqrinv = 1/Pqr
    Psrinv = 1/Psr
    Prinv = 1/Pr
    Pinv = Pqrinv + Psrinv + Prinv
    P = 1/Pinv
    
    Sqr = H[3]
    Ssr = H[4]
    
    n2 = dist2(x, x2)
    fNumPqr = (2*Pqrinv + Prinv)**(1/4)
    fNumPsr = (2*Psrinv + Prinv)**(1/4)
    fDen = np.prod(Pinv)**(1/2)
    factor = Sqr*Ssr*fNumPqr*fNumPsr / fDen
    fsensq = Ssr*fNumPqr*fNumPsr / fDen
    fsenss = Sqr*fNumPqr*fNumPsr / fDen
    
    Kbase = np.exp(-.5*P*n2)
    K = factor*Kbase
    
    return dict(K=K, Kbase=Kbase, Pqrinv=Pqrinv, Psrinv=Psrinv, Prinv=Prinv, P=P,
                fsensq=fsensq, fsenss=fsenss, n2=n2)


def ggxGaussianKernCompute(H, x, x2=None):
    
    if x2 is None: x2 = x
    
    Pqr = H[0]    # precision G
    Pr = H[1]     # precision U
    Pqrinv = 1/Pqr
    Prinv = 1/Pr
    Pinv = Pqrinv + Prinv
    P = 1/Pinv
    sensitivity = H[2] # sensitivity G

    n2 = dist2(x, x2)
    
    fNumPqr = (2*Pqrinv + Prinv)**(1/4)
    fNumPr = Pr**(-1/4)
    fDen = Pinv**(1/2)
    factor = sensitivity * fNumPqr * fNumPr / fDen
    fsens = fNumPqr * fNumPr / fDen    
    
    Kbase = np.exp(-.5*P*n2)
    K = factor*Kbase

    return dict(K=K, Kbase=Kbase, Pqrinv=Pqrinv, 
                Prinv=Prinv, P=P, fsens=fsens, n2=n2)


def rbfKernCompute(H, x, x2=None):
    
    if x2 is None: x2 = x
    
    precisionU = H[0]
    variance = H[1]
    
    n2 = dist2(x, x2)
    
    wi2 = .5*precisionU
    Kbase = np.exp(-n2*wi2)
    K = variance*Kbase
    
    return dict(K=K, Kbase=Kbase, n2=n2)


def rbfKernDiagCompute(H, x, x2=None):
    
    if x2 is None: x2 = x
    
    precision = H[0]
    variance = H[1]
    
    K = variance * np.ones((x.shape[0],1))
    
    return dict(K=K, pre=precision, var=variance)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    