import numpy as np
from GPtools.kernCompute import (
    rbfKernCompute, rbfKernDiagCompute, ggKernCompute, ggxggKernCompute, 
    ggxGaussianKernCompute, ggKernDiagCompute,
)

def rbfKernGradient(H, partialMat, x, x2=None):
    
    if x2 is None: x2 = x
    
    kernRes = rbfKernCompute(H=H, x=x, x2=x2)
    K = kernRes['K']
    Kbase = kernRes['Kbase']
    n2 = kernRes['n2']
    matGradPreU = np.sum(-.5*K*n2*partialMat)
    matGradSens = np.sum(Kbase*partialMat)
    
    return dict(matGradPreU=matGradPreU, matGradSens=matGradSens)

def rbfKernDiagGradient(H, partialMat, x, x2=None):

    if x2 is None: x2 = x
    kernRes = rbfKernDiagCompute(H=H, x=x, x2=x2)
    precision, variance = kernRes['pre'], kernRes['var']
    
    matGradPre = 0
    matGradVar = np.sum(np.diag(partialMat))

    return dict(matGradPre=matGradPre, matGradVar=matGradVar)


def ggKernGradient(H, partialMat, x, x2=None):
    
    if x2 is None: x2 = x
    
    kernRes = ggKernCompute(H, x=x, x2=x2)
    K = kernRes['K']
    Kbase=kernRes['Kbase']
    Prinv=kernRes['Prinv']
    Pqinv=kernRes['Pqinv']
    P=kernRes['P']
    dist=kernRes['n2']
    sensitivity = H[2]
    
    matGradPqr = - np.sum( partialMat * K * (Pqinv*P)**2 * dist )
    matGradPr  = - 0.5 * np.sum( partialMat * K * ((Prinv*P)**2 * dist) )
    gradSensq = 2 * sensitivity * np.sum(Kbase * partialMat)
    
    return dict(matGradPqr=matGradPqr, matGradPr=matGradPr, gradSensq=gradSensq)


def ggKernDiagGradient(H, partialMat, x, x2=None):
    
    if x2 is None: x2=x
    
    kernRes = ggKernDiagCompute(H, x)
    sensitivity = kernRes['sens']
    gradSensq = 2*sensitivity*np.sum(np.diag(partialMat))
    
    return dict(gradSensq=gradSensq)
    

def ggxggKernGradient(H, partialMat, x, x2=None):
    
    if x2 is None: x2 = x
    
    kernRes = ggxggKernCompute(H=H, x=x, x2=x2)
    K = kernRes['K']
    Kbase = kernRes['Kbase']
    Pqrinv = kernRes['Pqrinv']
    Psrinv = kernRes['Psrinv']
    Prinv = kernRes['Prinv']
    P = kernRes['P']
    fsensq = kernRes['fsensq']
    fsenss = kernRes['fsenss']
    dist = kernRes['n2']
    
    preFactorPqr = 1/(2*Pqrinv + Prinv)
    preFactorPsr = 1/(2*Psrinv + Prinv)
    preFactorPr = (1/(2*Pqrinv + Prinv)) + (1/(2*Psrinv + Prinv))
    
    oneMat = np.ones((x.shape[0], x2.shape[0]))
    
    matGradPr = np.sum( 
        0.5*K*(Prinv*Prinv*(oneMat*P - 0.5*oneMat*preFactorPr - P*dist*P))*partialMat 
    )
    matGradPqr = np.sum( 
        0.5*K*(Pqrinv*Pqrinv*(oneMat*P - oneMat*preFactorPqr - P*dist*P))*partialMat 
    )
    matGradPsr = np.sum(
        0.5*K*(Psrinv*Psrinv*(oneMat*P - oneMat*preFactorPsr - P*dist*P))*partialMat 
    )
    GradSensq = fsensq * np.sum( Kbase*partialMat )
    GradSenss = fsenss * np.sum( Kbase*partialMat )
    
    return dict(
        matGradPr=matGradPr, matGradPqr=matGradPqr, matGradPsr=matGradPsr,
        GradSensq=GradSensq, GradSenss=GradSenss
    )
    

def ggxGaussianKernGradient(H, partialMat, x, x2=None):
    
    if x2 is None: x2 = x
    
    kernRes = ggxGaussianKernCompute(H=H, x=x, x2=x2)
    K=kernRes['K']
    Kbase=kernRes['Kbase']
    Pqrinv=kernRes['Pqrinv']
    Prinv=kernRes['Prinv']
    P=kernRes['P']
    fsens=kernRes['fsens']
    dist=kernRes['n2']
    
    oneMat = np.ones((x.shape[0], x2.shape[0]))
    
    preFactorPqr = 1/(2*Pqrinv + Prinv)
    preFactorPr = 1/Prinv + (1/(2*Pqrinv + Prinv))
    
    matGradPr = np.sum(
        0.5*partialMat*K*(Prinv*Prinv*(oneMat*P - 0.5*oneMat*preFactorPr- P*dist*P)) 
    )
    matGradPqr = np.sum( 
        0.5*partialMat*K*(Pqrinv*Pqrinv*(oneMat*P - oneMat*preFactorPqr- P*dist*P)) 
    )
    
    gradSenss = fsens*np.sum(partialMat*Kbase);
    
    return dict(
        matGradPr=matGradPr,
        matGradPqr=matGradPqr,
        gradSenss=gradSenss
    )

def whiteKernGradient(partialMat):
    
    return np.sum(np.diag(partialMat))
    
    
    
    
    
    
    
    
    