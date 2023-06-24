import sys
import warnings
import numpy as np
import numpy.random as npr
from pyrsistent import v
from sklearn.preprocessing import PolynomialFeatures

from GPtools.kernCompute import (
    ggKernCompute, ggxggKernCompute, rbfKernCompute, ggxGaussianKernCompute
)
from GPtools.kernGradient import (
    rbfKernGradient, ggKernGradient, ggxGaussianKernGradient, 
    whiteKernGradient, ggKernDiagGradient, rbfKernDiagGradient
)
from GPtools.utils import (
    var_trans, complex_log, compare_exact, compare_approximate
)
from GPy.util.linalg import pdinv, symmetrify, dpotrs, dpotri
from GPy.util import choleskies
from scipy.linalg.blas import dtrmm
from copy import copy

from likelihoods.gaussian import Gaussian


class vmgp(object):
    
    '''
    A class for Multi-output Gaussian Processes with Variational Inducing Kernals
    '''
    
    def __init__(self, X, y, **kwargs):
        
        #------------------------------ data ---------------------------------#
        self.X = X
        self.y = y
        self.nout = len(X)
        self.nobs = {i_out: data.shape[0] for i_out, data in X.items()}
        self.nlf = kwargs['nlf']
        self.nfed = kwargs.get('nfed')
        if self.nfed is None:
            self.nfed = 1
        self.ndim = X[0].shape[1]
        xrange = kwargs.get('xrange')
        if xrange is not None:
            self.minX, self.maxX= xrange
        else:
            self.minX, self.maxX = np.min(X[0]), np.max(X[0])        
          
        #------------------------ model parameters --------------------------#
        self.fixInducing = kwargs['fixInducing']
        self.numInducing = kwargs['numInducing']
        self.addIndKern = kwargs['addIndKern']
        self.priorMean = kwargs.get('priorMean') # 'poly-2'
        if self.priorMean is None:
            self.priorMean = 'None'
        elif self.priorMean[:4] == 'poly':
            self.poly_degree = int(self.priorMean[-1])
        
        
        # create parameters
        self.nparam, self.param = self.create_param()
        self.paramName = list(self.param.keys())
        self.beta = np.repeat(np.exp(-2), self.nout)
        self.inducingPt = kwargs.get('inducingPt')
        if self.inducingPt is not None:
            if not self.numInducing == self.inducingPt.shape[0]:
                raise ValueError("numInducing should be equal to the dim of inducingPt")
        else:
            factorInducing = .1; med = .5*(self.maxX - self.minX)
            self.inducingPt = np.linspace(
                    self.minX-factorInducing*med, self.maxX+factorInducing*med, 
                    num=self.numInducing
            )[:, None]
        # Non-fixed inducing Pt needs to be implemented.
        
        # determine nonnegativity
        self.nonnegative = dict()
        for param_name, params in self.param.items():
            if 'pre ' in param_name or 'var ' in param_name:
                self.nonnegative[param_name] = np.ones_like(params, dtype=bool)
            else: 
                self.nonnegative[param_name] = np.zeros_like(params, dtype=bool)
        
        # mean
        if 'mean' not in kwargs: 
            self.mean = {
                i_out: np.zeros((data.shape[0], 1)) for i_out, data in y.items()
            }
        else: self.mean = kwargs['mean']
        
        self.Eqflogpyf = None
        self.KLqupu = None
        self.elbo = None

        self.X_batch = copy(self.X)
        self.y_batch = copy(self.y)
        self.mean_batch = copy(self.mean)
        self.batch_size = copy(self.nobs)

        self.pred_mean_gp = dict()
        self.pred_mean = dict()
        self.pred_var = dict()

        self.compute_mean(param=self.param)
        self.compute_mat(param=self.param)


    def compute_mean(self, param):

        """
        Calculate prior mean
        """

        if self.priorMean[:4] == 'poly':
            polynomial_features = PolynomialFeatures(degree=self.poly_degree)
            
            poly_b_ = {
                i: param['poly b{}'.format(i)]
                for i in range(self.poly_degree+1)
            }

            for i_out in range(self.nout):
                xp_i = polynomial_features.fit_transform(self.X[i_out])
                poly_b = np.array([
                    poly_b_[i][i_out] for i in range(self.poly_degree+1)
                ])[:, None]

                self.mean[i_out] = xp_i @ poly_b

    
    def compute_mat(self, param):

        """
        Calculate important matrices
        """

        #---------------------- compute y - mean ----------------------------#
        self.m_batch = {
            m: self.y_batch[m]-self.mean_batch[m] for m in range(self.nout)
        }
        
        #---------------------- compute matrices ----------------------------#
        # get hyperparameters
        pre_f = param['pre diff f']
        sen_f = param['sen diff f']
        pre_u = param['pre RBF u']
        var_u = param['var RBF u']
        if self.addIndKern: 
            pre_rbf_f = param['pre RBF f']
            var_rbf_f = param['var RBF f']
        
        # q(u)
        self.muu, self.Lu, self.Su, self.Suinv = dict(), dict(), dict(), dict()
        for l in range(self.nlf):
            self.muu[l] = param['mean vardist u'][:,[l]]
            self.Lu[l] = np.asarray(
                choleskies.flat_to_triang(param['cov vardist u'][:,[l]])
            )[0]
            self.Su[l] = np.dot(self.Lu[l], self.Lu[l].T)
            self.Suinv[l], _ = dpotri(np.asfortranarray(self.Lu[l]), lower=1)
            symmetrify(self.Suinv[l])

            # safety
            if np.any(np.isinf(self.Suinv[l])):
                raise ValueError("Suinv: Cholesky representation unstable")

        # p(f_m|u)        
        ## compute Kuu
        u, self.Kuu = dict(), dict()
        for l in range(self.nlf):
            u[l] = self.inducingPt
            self.Kuu[l] = (
                rbfKernCompute([pre_u[l], 1], u[l])['K'] 
                + var_u[l]*np.eye(self.numInducing)
            )

        ## compute Kfu & Kff (blockdiag)
        self.Kfu, self.Kff, self.Kff_diag = dict(), dict(), dict()
        for m in range(self.nout):
            # Kfu
            self.Kfu[m] = dict()
            for l in range(self.nlf):
                self.Kfu[m][l] = ggxGaussianKernCompute(
                    H=[pre_f[m,l], pre_u[l], sen_f[m,l]], x=self.X_batch[m], x2=u[l]
                )['K']
            
            # Kff
            self.Kff[m] = np.zeros(
                (self.X_batch[m].shape[0], self.X_batch[m].shape[0])
            )
            for l in range(self.nlf):
                self.Kff[m] += ggxggKernCompute(
                    H=[pre_f[m,l], pre_f[m,l], pre_u[l], sen_f[m,l], sen_f[m,l]],
                    x=self.X_batch[m], x2=self.X_batch[m]
                )['K']
            self.Kff[m] += 1e-9 * np.eye(self.X_batch[m].shape[0])
            if self.addIndKern: 
                self.Kff[m] += rbfKernCompute(
                    H=[pre_rbf_f[m], var_rbf_f[m]], x=self.X_batch[m]
                )['K']

            #Kff_diag
            self.Kff_diag[m] = np.diag(self.Kff[m])[:, None]
        
        ## compute Kuuinv & log(det(Kuu))
        self.Kuuinv, self.sqrtKuu = dict(), dict()
        self.logDetKuu, self.logDetKuuT = dict(), dict()

        for l in range(self.nlf):
            self.Kuuinv[l], self.sqrtKuu[l], _, self.logDetKuu[l] = (
                pdinv(self.Kuu[l], 10)
            )
            self.logDetKuuT[l] = self.logDetKuu[l]

        # q(f_m) = \int\int p(f_m|u) q(u) du    
        self.KfuKuuinv, self.KfuKuuinvmuu, self.sqrtSuKuuinvKuf = dict(), dict(), dict()
        self.qf_mu, self.qf_var, self.qf_cov = dict(), dict(), dict()
        
        for m in range(self.nout):

            self.KfuKuuinv[m], self.KfuKuuinvmuu[m], self.sqrtSuKuuinvKuf[m] = dict(), dict(), dict()
            self.qf_mu[m] = np.zeros((self.X_batch[m].shape[0], 1))
            self.qf_var[m] = np.zeros_like(self.Kff_diag[m])
            self.qf_cov[m] = np.zeros_like(self.Kff[m])

            for l in range(self.nlf):

                KuuinvKuf_ml = np.dot(self.Kuuinv[l], self.Kfu[m][l].T)
                self.KfuKuuinv[m][l] = KuuinvKuf_ml.T
                self.qf_mu[m] += np.dot(self.KfuKuuinv[m][l], self.muu[l])
                self.sqrtSuKuuinvKuf[m][l] = np.dot(self.Lu[l].T, KuuinvKuf_ml)
                self.qf_var[m] += (
                    self.Kff_diag[m]
                    + np.sum(np.square(self.sqrtSuKuuinvKuf[m][l]), 0)[:, None]
                    - np.sum(KuuinvKuf_ml * self.Kfu[m][l].T, 0)[:, None]
                )
                self.qf_cov[m] += (
                    self.Kff[m]
                    + np.dot(np.dot(self.KfuKuuinv[m][l], self.Su[l]), KuuinvKuf_ml)
                    - np.dot(self.Kfu[m][l], KuuinvKuf_ml)
                )       
        
        # safety check
        for m, qf_m_var in self.qf_var.items():
            if (qf_m_var < 0).any(): print('qf_{}_var negative!'.format(m)) 

        
        
    def calculate_KL(self):

        """
        Calculates the KL divergence term in ELBO
        """
        
        KL = 0
        for l in range(self.nlf):
            KL += (
                .5 * np.sum(self.Kuuinv[l] * self.Su[l])
                + .5 * np.dot(self.muu[l].T, np.dot(self.Kuuinv[l], self.muu[l]))
                - .5 * self.numInducing
                + .5 * 2 * np.sum(np.log(np.abs(np.diag(self.sqrtKuu[l]))))
                - .5 * 2 * np.sum(np.log(np.abs(np.diag(self.Lu[l]))))
            )

        
        return KL
    
    
    def ELBO(self, H, **kwargs):
        
        """
        Calculates Evidence Lower Bound (ELBO).
        """

        batch_idx = kwargs.get('batch_idx')
        self.model_update(H, batch_idx=batch_idx)
        batch_factor = sum(self.nobs.values()) / sum(self.batch_size.values())

        Eqflogpyf_i = [
            Gaussian(sigma=1/self.beta[m]**.5).var_exp(
                self.m_batch[m], self.qf_mu[m], self.qf_var[m]
            ) for m in range(self.nout)
        ]
        self.Eqflogpyf = np.sum([np.sum(i) for i in Eqflogpyf_i])
        
        self.KLqupu = self.calculate_KL().squeeze()
        
        self.elbo = self.nfed * batch_factor * self.Eqflogpyf - self.KLqupu
        
        return - self.elbo
    
    
    def gradELBO(self, H, **kwargs):
        
        """
        Calculates gradients of ELBO
        """

        batch_idx = kwargs.get('batch_idx')
        self.model_update(H, batch_idx=batch_idx)
        batch_factor = sum(self.nobs.values()) / sum(self.batch_size.values())
        
        # calculate matrices
        u = {l: self.inducingPt for l in range(self.nlf)}
        
        VEdqmu, VEdqv, VEdlik_v, VEdY = dict(), dict(), dict(), dict()

        for m in range(self.nout):
            VE_exp_dev_res = Gaussian(1/self.beta[m]**.5).var_exp_derivatives(
                self.m_batch[m], self.qf_mu[m], self.qf_var[m]
            ) 
            VEdqmu[m], VEdqv[m], VEdlik_v[m], VEdY[m] = (
                self.nfed * VE_exp_dev_res[0] * batch_factor,
                self.nfed * VE_exp_dev_res[1] * batch_factor,
                self.nfed * VE_exp_dev_res[2] * batch_factor,
                self.nfed * VE_exp_dev_res[3] * batch_factor,
            )
        VEdlik_v = {m: np.sum(VEdlik_v[m]) for m in range(self.nout)}
        if self.priorMean[:4] == 'poly':
            VEdb = dict()
            for m in range(self.nout):
                xp_i = - PolynomialFeatures(
                    degree=self.poly_degree
                ).fit_transform(self.X_batch[m])
                VEdb[m] = xp_i.T @ VEdY[m]

        # KL terms
        dKLdmuu, dKLdSu, dKLdKuu = dict(), dict(), dict()

        for l in range(self.nlf):
            dKLdmuu[l] = self.Kuuinv[l] @ self.muu[l]
            dKLdSu[l] = .5 * (self.Kuuinv[l] - self.Suinv[l])
            dKLdKuu[l] = .5 * (
                self.Kuuinv[l] 
                - self.Kuuinv[l] @ self.Su[l] @ self.Kuuinv[l]
                - self.Kuuinv[l] @ (self.muu[l] @ self.muu[l].T) @ self.Kuuinv[l].T
            )
                
        # VE terms
        dVEdmuu, dVEdSu, dVEdKuu = dict(), dict(), dict()
        dVEdKuf, dVEdKffdiag = {m:{} for m in range(self.nout)}, dict() 

        for l in range(self.nlf):
            
            dVEdmuu_l = np.zeros((self.numInducing, 1))
            dVEdSu_l = np.zeros((self.numInducing, self.numInducing))
            dVEdKuu_l = np.zeros((self.numInducing, self.numInducing))
            for m in range(self.nout):
                
                #
                dVEdmuu_l += self.KfuKuuinv[m][l].T @ VEdqmu[m]
                
                #
                KuuinvKufVEdqv = np.ascontiguousarray(
                    self.KfuKuuinv[m][l].T * VEdqv[m].T
                )
                B = KuuinvKufVEdqv @ self.KfuKuuinv[m][l]
                dVEdSu_l += B 
                
                #
                C = B @ self.Su[l] @ self.Kuuinv[l]
                KuuinvKufVEdqmu = self.KfuKuuinv[m][l].T @ VEdqmu[m]
                D = KuuinvKufVEdqmu @ (self.Kuuinv[l] @ self.muu[l]).T
                dVEdKuu_l += ( B - C - C.T - D ) 
    
                #
                tmp = 2. * (self.Su[l] @ self.Kuuinv[l] - np.eye(self.numInducing))
                dVEdKuf_ml = (
                    (self.Kuuinv[l] @ self.muu[l] @ VEdqmu[m].T)
                    + (tmp.T @ KuuinvKufVEdqv)
                ) 
                dVEdKuf[m][l] = dVEdKuf_ml
                
                #
                dVEdKffdiag[m] = VEdqv[m] 
            

            dVEdmuu[l] = dVEdmuu_l
            dVEdSu[l] = dVEdSu_l            
            dVEdKuu[l] = .5 * (dVEdKuu_l + dVEdKuu_l.T)

        # Sum of VE and KL terms
        dLdmuu, dLdSu, dLdKuu, dLdLu = dict(), dict(), dict(), dict()
        for l in range(self.nlf):
            dLdmuu[l] = dVEdmuu[l] - dKLdmuu[l]
            dLdSu[l] = dVEdSu[l] - dKLdSu[l]
            dLdKuu[l] = dVEdKuu[l] - dKLdKuu[l]
        dLdKuf = dVEdKuf
        dLdKffdiag = dVEdKffdiag
        dLdlik_v = VEdlik_v

        # Convert Su gradients to Lu gradients
        for l in range(self.nlf):
            dLdLu[l] = 2. * (dLdSu[l] @ self.Lu[l])
            dLdLu[l] = np.asarray(choleskies.triang_to_flat(dLdLu[l][None, :,:]))


        #----------------- compute dL/dmat * dmat/dparams -------------------#
        # get hyperparameters
        pre_f = self.param['pre diff f']
        sen_f = self.param['sen diff f']
        pre_u = self.param['pre RBF u']
        var_u = self.param['var RBF u']
        if self.addIndKern: 
            pre_rbf_f = self.param['pre RBF f']
            var_rbf_f = self.param['var RBF f']
        
        # get factors related to variable transformation
        factorParam = copy(self.param)
        factorParam = {
            key: np.where(self.nonnegative[key], value, 1) 
            for key, value in factorParam.items()
        }
        
        factor_pre_f = factorParam['pre diff f']
        factor_sen_f = factorParam['sen diff f']
        factor_pre_u = factorParam['pre RBF u']
        factor_var_u = factorParam['var RBF u']
        if self.addIndKern:
            factor_pre_rbf_f = factorParam['pre RBF f']
            factor_var_rbf_f = factorParam['var RBF f']
        factor_beta = self.beta 


        ## params in KffDiag 
        dLdKffdiagParam = dict()
        for m in range(self.nout):
            dLdKffdiagParam[m] = dict()
            for l in range(self.nlf):
                H = sen_f[m,l]
                dLdKffdiagParam[m][l] = ggKernDiagGradient(
                    H, dLdKffdiag[m].squeeze(), x=self.X_batch[m]
                )

        if self.addIndKern:
            dLdKffdiagParam_Ind = dict()
            for m in range(self.nout):
                H = [pre_rbf_f[m], var_rbf_f[m]]
                dLdKffdiagParam_Ind[m] = rbfKernDiagGradient(
                    H, dLdKffdiag[m].squeeze(), x=self.X_batch[m]
                )
        
        ## params in Kuf 
        dLdKufParam = dict()
        for m in range(self.nout):
            dLdKufParam[m] = dict()
            for l in range(self.nlf):
                H = [pre_f[m,l], pre_u[l], sen_f[m,l]]
                dLdKufParam[m][l] = ggxGaussianKernGradient(
                    H, dLdKuf[m][l], x=u[l], x2=self.X_batch[m]
                )
       
        ## params in Kuu
        dLdKuuParam = dict()
        for l in range(self.nlf):    
            H = [pre_u[l], 1]
            dLdKuuParam[l] = rbfKernGradient(H, partialMat=dLdKuu[l], x=u[l])
        
        dLdKuuWhiteParam = dict()
        for l in range(self.nlf):
            dLdKuuWhiteParam[l] = whiteKernGradient(partialMat=dLdKuu[l])
        
        ## params beta
        dLdbetas = dict()
        for m in range(self.nout):
            dLdbetas[m] = - dLdlik_v[m] * 1/(self.beta[m]**2) 


        #------------------- compute dL/dparams * factors -------------------#
        
        dLdpre_f = np.zeros((self.nout, self.nlf))
        dLdsen_f = np.zeros((self.nout, self.nlf))
        dLdpre_u = [0 for q in range(self.nlf)]
        dLdvar_u = [0 for q in range(self.nlf)]
        if self.addIndKern:
            dLdpre_rbf_f = [0 for m in range(self.nout)]
            dLdvar_rbf_f = [0 for m in range(self.nout)]
        if self.priorMean[:4] == 'poly':
            dLdb = {
                i: [0 for m in range(self.nout)] 
                for i in range(self.poly_degree+1)
            }
        dLdbeta = [0 for m in range(self.nout)]
        
        for m in range(self.nout):
            for l in range(self.nlf):
                dLdpre_f[m,l] = dLdKufParam[m][l]['matGradPqr'] * factor_pre_f[m,l]
                dLdsen_f[m,l] = (
                    + self.nlf * dLdKffdiagParam[m][l]['gradSensq'] 
                    + dLdKufParam[m][l]['gradSenss']
                ) * factor_sen_f[m,l]
                # print(dLdKffdiagParam[m][l]['gradSensq'], dLdKufParam[m][l]['gradSenss']);
                # import sys; sys.exit()
        
        for l in range(self.nlf):
            for m in range(self.nout):
                dLdpre_u[l] += dLdKufParam[m][l]['matGradPr'] * factor_pre_u[l]
            dLdpre_u[l] += dLdKuuParam[l]['matGradPreU'] * factor_pre_u[l]
            dLdvar_u[l] += dLdKuuWhiteParam[l] * factor_var_u[l]
        
        if self.addIndKern:             
            for m in range(self.nout):
                dLdpre_rbf_f[m] = (
                    dLdKffdiagParam_Ind[m]['matGradPre'] * factor_pre_rbf_f[m]
                )
                dLdvar_rbf_f[m] = (
                    dLdKffdiagParam_Ind[m]['matGradVar'] * factor_var_rbf_f[m]
                )

        if self.priorMean[:4] == 'poly':
            for i in range(self.poly_degree + 1):
                dLdb[i] = [VEdb[m].squeeze()[i] for m in range(self.nout)]

        for m in range(self.nout):
            dLdbeta[m] = dLdbetas[m] * factor_beta[m]
        
        # aggregation to grad
        grad = list()
        grad += list(dLdpre_f.flatten())
        grad += list(dLdsen_f.flatten())
        if self.addIndKern: 
            grad += list(dLdpre_rbf_f)
            grad += list(dLdvar_rbf_f)
        if self.priorMean[:4] == 'poly':
            for i in range(self.poly_degree + 1):
                grad += dLdb[i] #[0 for __ in range(self.nout)] ###
        grad += list(dLdpre_u)
        grad += list(dLdvar_u)
        grad += list(np.concatenate(list(dLdmuu.values()),1).flatten())
        grad += list(np.concatenate(list(dLdLu.values()),1).flatten())
        grad += list(dLdbeta)
        
        return - np.array(grad)
        

    def model_update(self, H, batch_idx=None, verbose=False):
        
        if len(H) != (self.nparam + self.nout):
            warnings.warn('Different lengths of given parameter!')
        tmp_previous = self.X_batch

        # Get batch
        if batch_idx is not None:
            for i_out, indice in batch_idx.items():
                self.batch_size[i_out] = indice.shape[0]
        else: 
            self.batch_size = self.nobs
            batch_idx = {
                i_out: np.arange(data.shape[0]) for i_out, data in self.X.items()
            }

        self.X_batch = {m: self.X[m][batch_idx[m], :] for m in range(self.nout)}
        self.y_batch = {m: self.y[m][batch_idx[m], :] for m in range(self.nout)}
        
        # Get hyperparameters and variance
        new_param, new_beta = self.param_list_to_dict(H)
        
        # Transform non-negative parameters
        param_tr = copy(new_param)
        for key, param in new_param.items():
            param_tr[key] = var_trans(
                param, self.nonnegative[key], transType='exp'
            )
            
        beta_tr = var_trans(
            new_beta, np.ones_like(new_beta, dtype=bool), transType='exp'
        )
        
        # Set the parameters & Compute matrices if any changes exist
        # if compare_exact(self.param, param_tr) \
        #   and np.array_equal(self.beta, beta_tr) \
        #   and compare_exact(tmp_previous, self.X_batch):
        #     if verbose: print('Parameters need not to be updated.')
        #     return
        # else:
        self.compute_mean(param=param_tr)
        self.mean_batch = {
            m: self.mean[m][batch_idx[m], :] for m in range(self.nout)
        }
        self.compute_mat(param=param_tr)
        self.param = param_tr
        self.beta = beta_tr
        
    
    def param_list_to_dict(self, H):

        '''
        Convert parameter structure from list to dictionary
        '''

        new_param = copy(self.param); new_beta = copy(self.beta); start_idx = 0
        
        new_param['pre diff f'] = H[
            start_idx : start_idx + self.nout*self.nlf
        ].reshape((self.nout, self.nlf))
        start_idx += self.nout*self.nlf
        
        new_param['sen diff f'] = H[
            start_idx : start_idx + self.nout*self.nlf
        ].reshape((self.nout, self.nlf))
        start_idx += self.nout*self.nlf

        if self.addIndKern:
            
            new_param['pre RBF f'] = H[start_idx : start_idx + self.nout]
            start_idx += self.nout
            
            new_param['var RBF f'] = H[start_idx : start_idx + self.nout]
            start_idx += self.nout
        
        if self.priorMean[:4] == 'poly':
            for i in range(self.poly_degree + 1):
                new_param['poly b{}'.format(i)] = H[start_idx : start_idx + self.nout]
                start_idx += self.nout            

        new_param['pre RBF u'] = H[start_idx : start_idx + self.nlf]
        start_idx += self.nlf
        
        new_param['var RBF u'] = H[start_idx : start_idx + self.nlf]
        start_idx += self.nlf
        
        # needs to be extended to the case with multiple lat funcs
        new_param['mean vardist u'] = H[
            start_idx : start_idx + self.numInducing*self.nlf
        ].reshape((self.numInducing, self.nlf))
        start_idx += self.numInducing*self.nlf  
        
        new_param['cov vardist u'] = H[
            start_idx : start_idx + int(self.numInducing*(self.numInducing+1)/2)*self.nlf
        ].reshape((int(self.numInducing*(self.numInducing+1)/2), self.nlf))
        start_idx += int(self.numInducing*(self.numInducing+1)/2)*self.nlf
        
        new_beta = H[start_idx : start_idx + self.nout]
        
        return new_param, new_beta

    
    def create_param(self):

        '''
        Create parameters 
        '''
        
        param = dict()
        
        # param fm
        param['pre diff f'] = np.ones((self.nout, self.nlf))
        param['sen diff f'] = np.ones((self.nout, self.nlf))
        
        if self.addIndKern:
            param['pre RBF f'] = np.ones((self.nout,))
            param['var RBF f'] = np.ones((self.nout,))
            
        if self.priorMean[:4] == 'poly':
            for i in range(self.poly_degree+1):
                param['poly b{}'.format(i)] = np.zeros((self.nout,))
        
        if not self.fixInducing:
            sys.exit('Optimizing Inducing Pts is not implemented yet.')
        
        # param u
        param['pre RBF u'] = np.ones((self.nlf,))
        param['var RBF u'] = np.ones((self.nlf,))
        
        # param variational distribution
        param['mean vardist u'] = 2.5 * npr.randn(self.numInducing, self.nlf) #np.zeros((self.numInducing, self.nlf)) 
        param['cov vardist u'] = np.asarray(
            choleskies.triang_to_flat(
                np.tile(np.eye(self.numInducing)[None, :, :], (self.nlf, 1, 1))
            )     
        )
        
        nparam = np.sum([i.size for i in param.values()])
        
        return nparam, param
    

    def predict(self, Xpred, i_out, H=None):
        
        '''
        Calculate a predictive distribution at Xpred
        '''

        if H is not None: self.model_update(H)

        u = {l: self.inducingPt for l in range(self.nlf)}

        # get hyperparameters
        pre_f = self.param['pre diff f']
        sen_f = self.param['sen diff f']

        if self.addIndKern: 
            pre_rbf_f = self.param['pre RBF f']
            var_rbf_f = self.param['var RBF f']
        
        if self.priorMean[:4] == 'poly':
            poly_b_ = dict()
            for i in range(self.poly_degree+1):
                poly_b_[i] = self.param['poly b{}'.format(i)]

        pre_u = self.param['pre RBF u']
        var_u = self.param['var RBF u']

        # Kfstaru
        Kfstaru = dict()
        for l in range(self.nlf):
            Kfstaru[l] = ggxGaussianKernCompute(
                H=[pre_f[i_out,l], pre_u[l], sen_f[i_out,l]], x=Xpred, x2=u[l]
            )['K']

        # Kfstarfstar_diag
        Kfstarfstar = np.zeros((Xpred.shape[0], Xpred.shape[0]))
        for l in range(self.nlf):
            Kfstarfstar += ggxggKernCompute(
                H=[pre_f[i_out,l], pre_f[i_out,l], pre_u[l], sen_f[i_out,l], sen_f[i_out,l]],
                x=Xpred, x2=Xpred
            )['K'] + 1e-9*np.eye(Xpred.shape[0])
        Kfstarfstar_diag = np.diag(Kfstarfstar)[:, None]

        if self.addIndKern: 
            Kww = rbfKernCompute(
                H=[pre_rbf_f[i_out], var_rbf_f[i_out]], x=self.X[i_out]
            )['K']
            Kwstarw = rbfKernCompute(
                H=[pre_rbf_f[i_out], var_rbf_f[i_out]], x=Xpred, x2=self.X[i_out]
            )['K']
            KwstarwKwwy = Kwstarw @ Kww @ self.y[i_out]


        #  q(ystar)
        KfstaruKuuinv = dict()
        self.pred_mean[i_out] = np.zeros_like(Xpred)
        for l in range(self.nlf):
            KfstaruKuuinv[l] = np.dot(Kfstaru[l], self.Kuuinv[l])
            self.pred_mean[i_out] += np.dot(KfstaruKuuinv[l], self.muu[l])


        if self.addIndKern: 
            self.pred_mean[i_out] += KwstarwKwwy
        
        if self.priorMean[:4] == 'poly':
            Xpredpoly = PolynomialFeatures(
                degree=self.poly_degree).fit_transform(Xpred)
            poly_b_pred = np.array([
                poly_b_[i][i_out] for i in range(self.poly_degree+1)
            ])[:, None]
            self.pred_mean[i_out] += Xpredpoly @ poly_b_pred

        self.pred_var[i_out] = Kfstarfstar_diag + 1/self.beta[i_out]
        for l in range(self.nlf):
            sqrtSuKuuinvKufstar = np.dot(self.Lu[l].T, KfstaruKuuinv[l].T)
            self.pred_var[i_out] += (
                + np.sum(np.square(sqrtSuKuuinvKufstar), 0)[:, None]
                - np.sum(KfstaruKuuinv[l].T * Kfstaru[l].T, 0)[:, None]
            )

        return self.pred_mean[i_out], self.pred_var[i_out]


    def get_param(self):

        '''
        Return untransformed mgp parameters in a list form
        '''
        
        # Get hyperparameters and variance
        param_untr, beta_untr = copy(self.param), copy(self.beta)
        
        # untransform non-negative parameters (log)
        for key, param in self.param.items():
            param_untr[key] = var_trans(
                param, self.nonnegative[key], transType='log'
            )
            
        beta_untr = var_trans(
            self.beta, np.ones_like(beta_untr, dtype=bool), transType='log'
        )

        # dict to array (Order must be in correspondance with list_to_dict)
        param_array = list()
        
        param_array += list(param_untr['pre diff f'].flatten())
        param_array += list(param_untr['sen diff f'].flatten())
        if self.addIndKern:
            param_array += list(param_untr['pre RBF f'].flatten())
            param_array += list(param_untr['var RBF f'].flatten())
        if self.priorMean[:4] == 'poly':
            for i in range(self.poly_degree + 1):
                param_array += list(param_untr['poly b{}'.format(i)])
        param_array += list(param_untr['pre RBF u'])
        param_array += list(param_untr['var RBF u'])
        param_array += list(param_untr['mean vardist u'].flatten())
        param_array += list(param_untr['cov vardist u'].flatten())
        param_array += list(beta_untr)


        return np.array(param_array)