import sys
import math
import numpy as np
from numpy import newaxis
import scipy.linalg as npalg
import scipy.stats as stats
import os
from ..utils.linalg import *
from ..utils.se_ard import *
from ..utils.misc import *
from ..utils.metrics import *
from scipy.special import erf
import copy
from scipy.cluster.vq import kmeans2
from scipy.spatial.distance import cdist
import pdb
import warnings
warnings.filterwarnings('error', category=RuntimeWarning)
# warnings.filterwarnings("ignore", category=DeprecationWarning)
# warnings.filterwarnings("ignore", category=UserWarning)

class SGP_PEP:
    def __init__(self, Xtrain, ytrain, M, alpha=1.0, lik='gauss', gh_deg=5):
        """ Sparse Gaussian processes using Power EP
        Xtrain, ytrain: training inputs and outputs
        M: number of inducing points
        alpha: power in power EP, could be between 0 and 1
        lik: observation likelihood type, e.g. gauss or cdf
        gh_deg: degree for gauss-hermite quadrature, when lik='cdf'
        """
        self.Xtrain = Xtrain
        self.ytrain = ytrain

        self.N = Xtrain.shape[0]
        self.M = M
        self.D = Xtrain.shape[1]
        self.alpha = alpha
        self.lik = lik
        self.gh_deg = gh_deg

        # init hypers and placeholders
        self.ls = np.zeros((self.D, ))
        self.sf = 0
        self.sn = 0
        self.zu = np.zeros((self.M, self.D))

        # EP loop variables
        self.gamma = np.zeros(M)
        self.beta = np.zeros((M, M))
        self.variances = np.empty((self.N, 1))
        self.variances.fill(1e20)
        self.means = np.zeros((self.N, 1))

        self.inferred = False
        self.jitter = 1e-6

    def reset_pep(self):
        # EP loop variables
        self.gamma = np.zeros(self.M)
        self.beta = np.zeros((self.M, self.M))
        self.variances = np.empty((self.N, 1))
        self.variances.fill(1e20)
        self.means = np.zeros((self.N, 1))

    def get_pep_variables(self):
        return (copy.deepcopy(self.gamma), 
                copy.deepcopy(self.beta),
                copy.deepcopy(self.variances), 
                copy.deepcopy(self.means))

    def set_pep_variables(self, arr):
        self.gamma = arr[0]
        self.beta = arr[1]
        self.variances = arr[2]
        self.means = arr[3]

    def update_kuu(self):
        """ compute Kuu given the hypers
        """
        ls = self.ls
        sf = self.sf
        M = self.M
        zu_id = self.zu
        self.Kuu = compute_kernel(2*ls, 2*sf, self.zu, self.zu)  
        self.Kuu += np.diag(self.jitter * np.ones((M, )))
        self.Kuuinv = matrixInverse(self.Kuu)

    def update_pep_variables(self):
        self.gamma = np.dot(self.Kuuinv, self.m)
        self.beta = np.dot(self.Kuuinv, np.dot(self.Kuu - self.V, self.Kuuinv))

    def update_hypers(self, params, update_pep=False):
        """ update hypers
        """
        self.ls = params['ls'].copy()
        self.sf = params['sf'].copy()
        self.sn = params['sn'].copy()
        self.zu = params['zu'].copy()
        self.update_kuu()
        if update_pep:
            self.update_pep_variables()

    def update_posterior(self):
        Kfu = compute_kernel(2*self.ls, 2*self.sf, self.Xtrain, self.zu)
        KuuinvKuf = np.dot(self.Kuuinv, Kfu.T)
        means = self.means[:, 0]
        variances = self.variances[:, 0]
        T2u = np.dot(KuuinvKuf / variances, KuuinvKuf.T)
        # T2unew = np.sum(np.einsum('an,bn->abn', KuuinvKuf * variances, KuuinvKuf), axis=2)
        T1u = np.dot(KuuinvKuf, means / variances)
        # T1unew = np.sum(KuuinvKuf * means, axis=1)
        Vinv = self.Kuuinv + T2u
        
        V = matrixInverse(Vinv)
        m = np.dot(V, T1u)
        self.gamma = np.dot(self.Kuuinv, m)
        self.beta = np.dot(self.Kuuinv, np.dot(self.Kuu - V, self.Kuuinv))
        
        return (m, V, Vinv)

    def pep_iterate(
        self, 
        KuuinvKufi, 
        Kfiu, 
        Kfifi, 
        xi, 
        yi, 
        compute_logZ, 
        dlogZt_dm_handle,
        dlogZt_dm2_handle,
        gamma,
        alpha,
        beta,
        mean_i,
        variance_i,
        compute_posterior_func,
        compute_phi_mvg_func,
        logZt_func,
        dlogZt_dv_func,
        dlogZt_dsn_func,
        Kuuinv,
        ls,
        sf,
        zu
        ):
        """
        Run an iteration of power-ep given a datapoint
        returning the new values of mean and variance of the appproximate site factor
        and new parameters of the approximate posterior, gamma and beta
        """
        # Deletion
        p_i = KuuinvKufi
        k_i = Kfiu
        # Note h_si for the deletion uses the non \i version of beta
        h_si = p_i - np.dot(beta, k_i)

        dlogZd_dmi2 = 1.0 / (variance_i / alpha - np.dot(k_i, h_si))
        dlogZd_dmi = -dlogZd_dmi2 * (mean_i - np.dot(k_i, gamma))

        gamma_si = gamma + h_si * dlogZd_dmi
        beta_si = beta - np.outer(h_si, h_si)*dlogZd_dmi2

        # Projection
        h = p_i - np.dot(beta_si, k_i)
        m_si_i = np.dot(k_i, gamma_si)
        v_si_ii = Kfifi - np.dot(np.dot(k_i, beta_si), k_i)
        phi_cav = 0.0
        logZtilted = 0.0
        dlogZtilted = 0.0

        dlogZ_dmi = dlogZt_dm_handle(yi, m_si_i, v_si_ii, alpha)
        dlogZ_dmi2 = dlogZt_dm2_handle(yi, m_si_i, v_si_ii, alpha)

        gamma_new = gamma_si + h * dlogZ_dmi
        beta_new = beta_si - np.outer(h, h) * dlogZ_dmi2

        # Inclusion
        var_i_new = - 1.0 / dlogZ_dmi2 - np.dot(k_i, h)
        mean_i_new = m_si_i - dlogZ_dmi / dlogZ_dmi2

        var_new = 1 / (1 / var_i_new + 1 / variance_i * (1 - alpha))
        mean_div_var_i_new = (mean_i_new / var_i_new + 
                mean_i / variance_i * (1 - alpha))
        mean_new = mean_div_var_i_new * var_new

        if compute_logZ:
            (m_cav, V_cav) = compute_posterior_func(gamma_si, beta_si)
            phi_cav = compute_phi_mvg_func(m_cav, V_cav)
            logZtilted = logZt_func(yi, m_si_i, v_si_ii, alpha)
            dlogZ_dvi = dlogZt_dv_func(yi, m_si_i, v_si_ii, alpha)
            KuuinvMcav = np.dot(Kuuinv, m_cav)
            dlogZtilted_dKuu_via_mi = -np.outer(dlogZ_dmi * KuuinvMcav, p_i)
            KuuinvVcavKuuinvKufi = np.dot(Kuuinv, np.dot(V_cav, p_i))
            temp1 = -np.outer(KuuinvVcavKuuinvKufi, p_i*dlogZ_dvi)
            temp2 = temp1.T
            temp3 = np.outer(p_i, p_i*dlogZ_dvi)
            dlogZtilted_dKuu_via_vi = temp1 + temp2 + temp3
            dlogZtilted_dKuu = dlogZtilted_dKuu_via_mi + dlogZtilted_dKuu_via_vi
            dlogZtilted_dKfu_via_mi = dlogZ_dmi * KuuinvMcav
            dlogZtilted_dKfu_via_vi = 2 * dlogZ_dvi * (-p_i + KuuinvVcavKuuinvKufi)
            dlogZtilted_dKfu = dlogZtilted_dKfu_via_mi + dlogZtilted_dKfu_via_vi
            
            dlogZtilted_dsf = (2*np.sum(dlogZtilted_dKfu * k_i) 
                    + 2*dlogZ_dvi*np.exp(2*sf))
            ls2 = np.exp(2*ls)
            M = zu.shape[0]
            D = zu.shape[1]
            ones_M = np.ones((M, ))
            ones_D = np.ones((D, ))
            xi_minus_zu = np.outer(ones_M, xi) - zu
            temp1 = np.outer(k_i, ones_D) * 0.5 * xi_minus_zu**2
            dlogZtilted_dls = 2*np.dot(dlogZtilted_dKfu, temp1) * 1.0 / ls2
            temp2 = xi_minus_zu * np.outer(ones_M, 1.0 / ls2 )
            dlogZtilted_dzu = np.outer(dlogZtilted_dKfu * k_i, ones_D) * temp2

            dlogZtilted_dsn = dlogZt_dsn_func(
                    yi, m_si_i, v_si_ii, alpha)

            dlogZtilted = {
                    'ls': dlogZtilted_dls, 
                    'sf': dlogZtilted_dsf,
                    'sn': dlogZtilted_dsn, 
                    'zu': dlogZtilted_dzu, 
                    'Kuu': dlogZtilted_dKuu}

        return (var_new, mean_new, gamma_new, beta_new, 
                logZtilted, dlogZtilted, phi_cav)
    
    def run_pep(self, indices, no_ep_sweeps, compute_logZ=False, sequential=False):
        if sequential:
            return self.run_pep_sequential(indices, no_ep_sweeps, compute_logZ)
        else:
            return self.run_pep_parallel(indices, no_ep_sweeps, compute_logZ)

    def run_pep_sequential(self, indices, no_ep_sweeps, compute_logZ=False):
        Nb = len(indices)
        log_lik = 0.0
        grads = {'ls': np.zeros((self.D, )), 
                'sf': 0, 
                'sn': 0, 
                'zu': np.zeros((self.M, self.D))}
        if no_ep_sweeps != 0:
            X = self.Xtrain[indices, :]
            Kfu = compute_kernel(2*self.ls, 2*self.sf, X, self.zu)
            KuuinvKuf = np.dot(self.Kuuinv, Kfu.T)
            Kff_diag = compute_kernel_diag(2*self.ls, 2*self.sf, X)
            grads_logZtilted = {
                    'ls': np.zeros((self.D, )), 
                    'sf': 0, 
                    'sn': 0, 
                    'zu': np.zeros((self.M, self.D)), 
                    'Kuu': np.zeros((self.M, self.M))}
        for k in xrange(no_ep_sweeps):
            find_log_lik = compute_logZ and (k == no_ep_sweeps-1)
            # loop through all points
            for j in range(len(indices)):
                i = indices[j]
                # print i, j
                (var_i_new, mean_i_new, gamma_new, beta_new, 
                        logZtilted, dlogZtilted, phi_cav) = \
                                self.pep_iterate(KuuinvKuf[:, j], Kfu[j, :], Kff_diag[j], 
                                        self.Xtrain[i, :], self.ytrain[i], 
                                        find_log_lik,
                                        self.dlogZtilted_dm,
                                        self.dlogZtilted_dm2,
                                        self.gamma,
                                        self.alpha,
                                        self.beta,
                                        self.means[i],
                                        self.variances[i],
                                        self.compute_posterior,
                                        self.compute_phi_mvg,
                                        self.logZtilted,
                                        self.dlogZtilted_dv,
                                        self.dlogZtilted_dsn,
                                        self.Kuuinv,
                                        self.ls,
                                        self.sf,
                                        self.zu)
                self.gamma = gamma_new
                self.beta = beta_new
                self.means[i] = mean_i_new
                self.variances[i] = var_i_new

                if find_log_lik:
                    log_lik += logZtilted + phi_cav
                    for name in ['ls', 'sf', 'sn', 'zu', 'Kuu']:
                        grads_logZtilted[name] += dlogZtilted[name]

            (m, V) = self.compute_posterior(self.gamma, self.beta)
            self.m = m
            self.V = V
            if find_log_lik:
                N = self.N
                # scale logZtilted and phi_cav accordingly
                scale_logZtilted = N * 1.0 / Nb / self.alpha
                log_lik = log_lik * scale_logZtilted
                phi_pos = self.compute_phi_mvg(m, V)
                phi_prior = self.compute_phi_mvg(np.zeros(m.shape), self.Kuu)
                scale_post = (-len(self.ytrain) * 1.0 / self.alpha + 1.0)

                log_lik += scale_post * phi_pos - phi_prior


                self.log_lik = log_lik

                # compute the gradients
                Kuuinv = self.Kuuinv
                Vmm = V + np.outer(m, m)
                S = - Kuuinv + np.dot(Kuuinv, np.dot(Vmm, Kuuinv))
                S = S + 2*scale_logZtilted * grads_logZtilted['Kuu']

                dhyp = d_trace_MKzz_dhypers(2*self.ls, 2*self.sf, self.zu, S, self.Kuu)
                grads['sf'] = dhyp[0] + scale_logZtilted * grads_logZtilted['sf']
                grads['ls'] = dhyp[1] + scale_logZtilted * grads_logZtilted['ls']
                grads['zu'] = dhyp[2]/2 + scale_logZtilted * grads_logZtilted['zu']
                grads['sn'] = scale_logZtilted * grads_logZtilted['sn']

        self.inferred = True
        error = False # todo
        return log_lik, grads, error

    def run_pep_parallel(self, indices, no_ep_sweeps, compute_logZ=False):
        Nb = len(indices)
        log_lik = 0.0
        grads = {
                'ls': np.zeros((self.D, )), 
                'sf': 0, 
                'sn': 0, 
                'zu': np.zeros((self.M, self.D))
                }
        if no_ep_sweeps != 0:
            X = self.Xtrain[indices, :]
            Kfu = compute_kernel(2*self.ls, 2*self.sf, X, self.zu)
            KuuinvKuf = np.dot(self.Kuuinv, Kfu.T)
            Kff_diag = compute_kernel_diag(2*self.ls, 2*self.sf, X)
            grads_logZtilted = {
                    'ls': np.zeros((self.D, )), 
                    'sf': 0, 
                    'sn': 0, 
                    'zu': np.zeros((self.M, self.D)), 
                    'Kuu': np.zeros((self.M, self.M))}
        error = False
        for k in xrange(no_ep_sweeps):
            # print k
            Xbatch = self.Xtrain[indices, :]
            ybatch = self.ytrain[indices]
            find_log_lik = compute_logZ and (k == no_ep_sweeps-1)

            # perform parallel updates
            # deletion
            p_i = KuuinvKuf[:, :, newaxis].transpose((1, 0, 2))
            k_i = Kfu[:, :, newaxis] 
            k_ii = Kff_diag[:, newaxis]
            gamma = self.gamma[:, newaxis]
            beta = self.beta
            alpha = self.alpha
            h_si = p_i - np.einsum('ab,kbc->kac', beta, k_i)
            variance_i_ori = self.variances[indices, :]
            variance_i = variance_i_ori[:, :, newaxis]
            mean_i_ori = self.means[indices, :]
            mean_i = mean_i_ori[:, :, newaxis]
            dlogZd_dmi2 = 1.0 / (variance_i/alpha - 
                np.sum(k_i * h_si, axis=1, keepdims=True))
            dlogZd_dmi = -dlogZd_dmi2 * (mean_i - 
                np.sum(k_i * gamma, axis=1, keepdims=True))
            hd1 = h_si * dlogZd_dmi
            hd2h = np.einsum('abc,adc->abd', h_si, h_si) * dlogZd_dmi2
            gamma_si = gamma + hd1
            beta_si = beta - hd2h

            # projection
            h = p_i - np.einsum('abc,acd->abd', beta_si, k_i)
            m_si_i = np.einsum('abc,abc->ac', k_i, gamma_si)
            v_si_ii = k_ii - np.einsum('abc,abd,adc->ac', k_i, beta_si, k_i)

            dlogZ_dmi = np.zeros(m_si_i.shape)
            dlogZ_dmi2 = np.zeros(m_si_i.shape)
            dlogZ_dvi = np.zeros(m_si_i.shape)
            logZtilted = np.zeros(m_si_i.shape)
            try:
                # todo: remove this loop
                for i in range(Nb):
                    m_ii = m_si_i[i, 0]
                    v_ii = v_si_ii[i, 0]
                    y_ii = ybatch[i]

                    logZtilted[i] = self.logZtilted(y_ii, m_ii, v_ii, alpha)
                    dlogZ_dmi[i] = self.dlogZtilted_dm(y_ii, m_ii, v_ii, alpha)
                    dlogZ_dmi2[i] = self.dlogZtilted_dm2(y_ii, m_ii, v_ii, alpha)
                    dlogZ_dvi[i] = self.dlogZtilted_dv(y_ii, m_ii, v_ii, alpha)

                var_i_new = -1.0 / dlogZ_dmi2 - np.sum(k_i * h, axis=1)
                mean_i_new = m_si_i - dlogZ_dmi / dlogZ_dmi2

                var_new_parallel = 1 / (1 / var_i_new + 1 / variance_i_ori * (1 - alpha))
                mean_div_var_i_new = (mean_i_new / var_i_new + 
                    mean_i_ori / variance_i_ori * (1 - alpha))
                mean_new_parallel = mean_div_var_i_new * var_new_parallel

                rho = 0.5

                n1_new = 1.0 / var_new_parallel
                n2_new = mean_new_parallel / var_new_parallel

                n1_ori = 1.0 / variance_i_ori
                n2_ori = mean_i_ori / variance_i_ori

                n1_damped = rho * n1_new + (1.0 - rho) * n1_ori
                n2_damped = rho * n2_new + (1.0 - rho) * n2_ori

                var_new_parallel = 1.0 / n1_damped
                mean_new_parallel = var_new_parallel * n2_damped 

                # update means and variances
                self.means[indices, :] = mean_new_parallel
                self.variances[indices, :] = var_new_parallel

                # update gamma and beta
                (m, V, Vinv) = self.update_posterior()
                self.m = m
                self.V = V
                if find_log_lik:
                    N = self.N
                    # compute cavity covariance
                    betacavKuu = np.einsum('abc,cd->abd', beta_si, self.Kuu)
                    mcav = np.einsum('bc,acd->abd', self.Kuu, gamma_si)
                    Vcav = self.Kuu - np.einsum('bc,acd->abd', self.Kuu, betacavKuu)
                    signV, logdetV = np.linalg.slogdet(V)
                    signKuu, logdetKuu = np.linalg.slogdet(self.Kuu)
                    Vinvm = np.dot(Vinv, m)
                    term1 = 0.5 * (logdetV - logdetKuu + np.dot(m, Vinvm))

                    
                    tn = 1.0 / var_new_parallel
                    gn = mean_new_parallel
                    wnVcav = np.einsum('abc,abd->adc', p_i, Vcav)
                    wnVcavwn = np.einsum('abc,abd->ac', wnVcav, p_i)
                    wnVcavVinvm = np.sum(wnVcav * Vinvm[:, newaxis], axis=1)
                    wnV = np.einsum('abc,bd->adc', p_i, V)
                    wnVwn = np.sum(wnV * p_i, axis=1)
                    mwn = np.einsum('b,abc->ac', m, p_i)
                    oneminuswnVwn = 1 - self.alpha * tn * wnVwn

                    term2a = 0.5 * self.alpha * tn**2 * gn**2 * wnVcavwn
                    term2b = - gn * tn * wnVcavVinvm
                    term2c = 0.5 * tn * mwn**2 / oneminuswnVwn
                    term2d = -0.5 / alpha * np.log(oneminuswnVwn)
                    term2 = N / Nb * np.sum(term2a + term2b + term2c + term2d)

                    scale_logZtilted = N / Nb / self.alpha
                    term3 = scale_logZtilted * np.sum(logZtilted)

                    log_lik = term1 + term2 + term3

                    KuuinvMcav = np.einsum('bc,acd->abd', self.Kuuinv, mcav)
                    dlogZt_dmiKuuinvMcav = dlogZ_dmi[:, newaxis, :] * KuuinvMcav
                    dlogZt_dKuu_via_mi = -np.einsum('abc,adc->abd', dlogZt_dmiKuuinvMcav, p_i)
                    
                    VcavKuuinvKufi = np.einsum('abc,acd->abd', Vcav, p_i)
                    KuuinvVcavKuuinvKufi = np.einsum('bc,acd->abd', self.Kuuinv, VcavKuuinvKufi)
                    p_idlogZ_dvi = p_i * dlogZ_dvi[:, newaxis, :]
                    temp1 = - np.einsum('abc,adc->abd', KuuinvVcavKuuinvKufi, p_idlogZ_dvi)
                    temp2 = np.transpose(temp1, [0, 2, 1])
                    temp3 = np.einsum('abc,adc->abd', p_i, p_idlogZ_dvi)
                    dlogZt_dKuu_via_vi = temp1 + temp2 + temp3
                    dlogZt_dKuu = np.sum(dlogZt_dKuu_via_mi + dlogZt_dKuu_via_vi, axis=0)

                    dlogZt_dKfu_via_mi = dlogZt_dmiKuuinvMcav
                    dlogZt_dKfu_via_vi = 2 * dlogZ_dvi[:, newaxis, :] * (-p_i + KuuinvVcavKuuinvKufi)
                    dlogZt_dKfu = dlogZt_dKfu_via_mi + dlogZt_dKfu_via_vi
                    dlogZt_dsf = (2*np.sum(dlogZt_dKfu * k_i) 
                        + 2*np.sum(dlogZ_dvi*np.exp(2*self.sf)))
                    ls2 = np.exp(2*self.ls)
                    M = self.zu.shape[0]
                    D = self.zu.shape[1]
                    ones_M = np.ones((Nb, M))
                    ones_D = np.ones((Nb, D))
                    xi_minus_zu = np.einsum('km,kd->kmd', ones_M, Xbatch) - self.zu
                    
                    temp1 = np.einsum('kma,kd->kmd', k_i, ones_D) * 0.5 * xi_minus_zu**2
                    dlogZt_dls = 2.0*np.sum(dlogZt_dKfu * temp1) / ls2
                    temp2 = xi_minus_zu * np.einsum('km,d->kmd', ones_M, 1.0 / ls2 )
                    dlogZt_dzu = np.sum(np.einsum('kma,kd->kmd', dlogZt_dKfu * k_i, ones_D) * temp2, axis=0)

                    dlogZt_dsn = 0
                    for i in range(m_si_i.shape[0]):
                        dlogZt_dsn += self.dlogZtilted_dsn(ybatch[i], m_si_i[i], 
                            v_si_ii[i], self.alpha)

                    self.log_lik = log_lik

                    # compute the gradients
                    Kuuinv = self.Kuuinv
                    Vmm = V + np.outer(m, m)
                    S = - Kuuinv + np.dot(Kuuinv, np.dot(Vmm, Kuuinv))
                    S = S + 2*scale_logZtilted * dlogZt_dKuu

                    dhyp = d_trace_MKzz_dhypers(2*self.ls, 2*self.sf, self.zu, S, self.Kuu)
                    grads['sf'] = dhyp[0] + scale_logZtilted * dlogZt_dsf
                    grads['ls'] = dhyp[1] + scale_logZtilted * dlogZt_dls
                    grads['zu'] = dhyp[2]/2 + scale_logZtilted * dlogZt_dzu
                    grads['sn'] = scale_logZtilted * dlogZt_dsn
            except (RuntimeWarning, np.linalg.linalg.LinAlgError):
                print "exception: ignore this update"
                mean_new_parallel = mean_i_ori
                var_new_parallel = variance_i_ori
                error = True
        self.inferred = True
        return log_lik, grads, error

    def init_hypers(self):
        ls = self.__estimate_ls(self.Xtrain)
        sn = np.array([np.log(0.1*np.std(self.ytrain))])
        sf = np.array([np.log(1*np.std(self.ytrain))])
        sf[sf < -5] = 0
        sn[sn < -5] = np.log(0.1)

        # first layer
        M = self.M
        D = self.D
        Ntrain = self.Xtrain.shape[0]
        if Ntrain < 20000:
            centroids, label = kmeans2(self.Xtrain, M, minit='points')
        else:
            randind = np.random.permutation(Ntrain)
            centroids = self.Xtrain[randind[0:M], :]
        zu = centroids

        # dict to hold hypers and pseudo inputs
        params = {'ls': ls,
                  'sf': sf,
                  'zu': centroids,
                  'sn': sn}

        return params

    def get_hypers(self):
        params = {'ls': self.ls,
                  'sf': self.sf,
                  'zu': self.zu,
                  'sn': self.sn}
        return params

    def __estimate_ls(self, X):
        Ntrain = X.shape[0]
        if Ntrain < 10000:
            X1 = np.copy(X)
        else:
            randind = np.random.permutation(Ntrain)
            X1 = X[randind[0:(5*self.M)], :]

        dist = cdist(X1, X1, 'euclidean')
        D = X1.shape[1]
        N = X1.shape[0]
        triu_ind = np.triu_indices(N)
        ls = np.zeros((D, ))
        d2imed = np.median(dist[triu_ind])
        for i in range(D):
            ls[i] = np.log(d2imed / 2 + 1e-16)
        return ls

    def train(self, no_epochs, n_per_mb, Xtest=None, ytest=None, 
                reinit_hypers=True, compute_test=False, lrate=0.001, 
                no_ep_sweeps=10, fixed_params=[], print_trace=False, print_epoch=1):
        adamobj = self.init_adam(adamlrate=lrate)
        if reinit_hypers:
            init_params = self.init_hypers()
            params = init_params
        else:
            params = self.get_hypers()

        ind = 1
        check = False
        test_nll = []
        test_err = []
        energies = []
        Ntrain = self.Xtrain.shape[0]
        batch_idxs = make_batches(Ntrain, n_per_mb)
        try:
            epoch = 0

            if compute_test:
                mf, vf = self.predict_diag(Xtest)
                if self.lik == 'gauss':
                    vf = vf + np.exp(2*self.sn)
                test_nlli = compute_nll(ytest, mf, vf, self.lik)
                test_errori = compute_error(ytest, mf, vf, self.lik)
                if print_trace:
                    print ("\t test nll: %.5f, test error: %.5f" 
                            % (test_nlli, test_errori))
                test_nll.append(test_nlli)
                test_err.append(test_errori)
            while (not check):
                last_params = copy.deepcopy(params)
                last_pep_vars = self.get_pep_variables()
                try:
                    permutation = np.random.choice(range(Ntrain), Ntrain, replace=False)
                    if print_trace:
                        printProgress(0, len(batch_idxs), 
                                prefix = 'Epoch %d:' % epoch, 
                                suffix = 'Complete', 
                                barLength = 60)
                    for i, idxs in enumerate(batch_idxs):
                        # update hypers
                        update_pep = True
                        if epoch == 0 and i == 0:
                            update_pep = False
                        
                        self.update_hypers(params, update_pep=update_pep)
                        energy, grad, error = self.compute_energy(params, 
                                permutation[idxs], no_ep_sweeps=no_ep_sweeps)
                        if not error:
                            # print params
                            params = self.update_adam(adamobj, 
                                    params, grad, ind, 1.0, fixed=fixed_params) 
                            # print params
                            ind += 1
                            if print_trace:
                                printProgress(i+1, len(batch_idxs), 
                                        prefix = 'Epoch %d:' % epoch, 
                                        suffix = 'Complete, energy %.4f' % energy,
                                        barLength = 60)
                    epoch += 1
                    # lrate = 1.01*lrate
                    # adamobj = self.set_adam_lrate(adamobj, lrate)
                except (RuntimeWarning, Exception) as e:
                    print e
                    print 'runtime warning: skipping one epoch, reducing lrate'
                    params = last_params
                    self.set_pep_variables(last_pep_vars)
                    # pdb.set_trace()
                    lrate = 0.5*lrate
                    adamobj = self.set_adam_lrate(adamobj, lrate)
                    # increase epoch?
                    epoch += 1
                

                # TODO: check convergence
                converged = False
                check = (epoch >= no_epochs) or converged
                if compute_test and epoch % print_epoch == 0:
                    mf, vf = self.predict_diag(Xtest)
                    if self.lik == 'gauss':
                        vf = vf + np.exp(2*self.sn)
                    test_nlli = compute_nll(ytest, mf, vf, self.lik)
                    test_errori = compute_error(ytest, mf, vf, self.lik)
                    if print_trace:
                        print ("\tlogZ: %.5f, test nll: %.5f, test error: %.5f" 
                                % (energy, test_nlli, test_errori))
                    test_nll.append(test_nlli)
                    test_err.append(test_errori)
                    energies.append(energy)
                else:
                    energies.append(energy)
                    # if print_trace:
                    #     print "\t logZ: %.5f" % (energy)

        except KeyboardInterrupt:
            print 'Caught KeyboardInterrupt ...'

        # except np.linalg.linalg.LinAlgError:
        #     print 'linalg error ...'
        self.update_hypers(params)
        return test_nll, test_err, energies

    def predict(self, X_T):
        """
        Returns the mean and covariance for some test points.

        Does not include likelihood noise
        """
        # All vectors in columns
        Kfu = compute_kernel(2*self.ls, 2*self.sf, X_T, self.zu)
        Kff = compute_kernel(2*self.ls, 2*self.sf, X_T, X_T)
        m = np.dot(Kfu, self.gamma)
        V = Kff - Kfu.dot(self.beta).dot(Kfu.T)
        return (m, V)

    def predict_diag(self, Xtest):
        """
        Returns the marginal mean and variance for some test points.

        Does not include likelihood noise
        """
        Ntest = Xtest.shape[0]
        m = np.zeros((Ntest, ))
        v = np.zeros((Ntest, ))
        for i in range(Ntest):
            Xi = Xtest[i, :]
            Kfu = compute_kernel(2*self.ls, 2*self.sf, Xi, self.zu)
            Kff = compute_kernel(2*self.ls, 2*self.sf, Xi, Xi)
            m[i] = np.dot(Kfu, self.gamma)
            v[i] = Kff - Kfu.dot(self.beta).dot(Kfu.T)

        return m, v

    def compute_energy(self, params, train_indices, no_ep_sweeps=10):
        # run power-ep
        self.run_pep(train_indices, no_ep_sweeps-1, compute_logZ=False)
        log_lik, grads, error = self.run_pep(train_indices, 1, compute_logZ=True)

        return log_lik, grads, error

    def compute_posterior(self, gamma, beta):
        m_pos = np.dot(self.Kuu, gamma)
        V_pos = self.Kuu - np.dot(self.Kuu, np.dot(beta, self.Kuu))
        return (m_pos, V_pos)

    def compute_phi_mvg(self, m, V):
        try:
            (sign, logdet) = np.linalg.slogdet(V)
            phi = 0.5 * logdet
            phi += 0.5 * np.dot(m.T, np.dot(matrixInverse(V), m))
            # phi += 0.5 * np.dot(m.T, npalg.solve(V, m))
            phi += 0.5 * V.shape[0] * np.log(2*np.pi)
        except Exception, e:
            # pdb.set_trace()
            phi = 0
        return phi
       
    def logZtilted(self, y_i, m_si_i, v_si_ii, alpha):
        if self.lik == 'gauss':
            return self.gauss_logZtilted(y_i, m_si_i, v_si_ii, alpha, np.exp(2*self.sn))
        elif self.lik == 'cdf':
            return self.cdf_logZtilted(y_i, m_si_i, v_si_ii, alpha, self.gh_deg)

    def dlogZtilted_dm(self, y_i, m_si_i, v_si_ii, alpha):
        if self.lik == 'gauss':
            return self.gauss_dlogZtilted_dm(y_i, m_si_i, v_si_ii, alpha, np.exp(2*self.sn))
        elif self.lik == 'cdf':
            return self.cdf_dlogZtilted_dm(y_i, m_si_i, v_si_ii, alpha, self.gh_deg)

    def dlogZtilted_dv(self, y_i, m_si_i, v_si_ii, alpha):
        if self.lik == 'gauss':
            return self.gauss_dlogZtilted_dv(y_i, m_si_i, v_si_ii, alpha, np.exp(2*self.sn))
        elif self.lik == 'cdf':
            return self.cdf_dlogZtilted_dv(y_i, m_si_i, v_si_ii, alpha, self.gh_deg)

    def dlogZtilted_dsn(self, y_i, m_si_i, v_si_ii, alpha):
        if self.lik == 'gauss':
            return (2.0*np.exp(2*self.sn)*
                    self.gauss_dlogZtilted_dvy(y_i, m_si_i, v_si_ii, alpha, np.exp(2*self.sn)))
        elif self.lik == 'cdf':
            return 0

    def dlogZtilted_dm2(self, y_i, m_si_i, v_si_ii, alpha):
        if self.lik == 'gauss':
            return self.gauss_dlogZtilted_dm2(y_i, m_si_i, v_si_ii, alpha, np.exp(2*self.sn))
        elif self.lik == 'cdf':
            return self.cdf_dlogZtilted_dm2(y_i, m_si_i, v_si_ii, alpha, self.gh_deg)

    def gauss_logZtilted(self, y_i, m_si_i, v_si_ii, alpha, v_y):
        logZtilted = -0.5 * alpha * np.log(2*np.pi) + 0.5 * (1.0 - alpha) * np.log(v_y) \
            - 0.5 * np.log(alpha) - 0.5 * np.log(v_si_ii + v_y/alpha) \
            - 0.5 * (y_i - m_si_i)**2.0 / (v_si_ii + v_y/alpha)
        return logZtilted 

    def gauss_dlogZtilted_dm(self, y_i, m_si_i, v_si_ii, alpha, v_y):
        return (y_i-m_si_i)/(v_y/alpha+v_si_ii)

    def gauss_dlogZtilted_dvy(self, y_i, m_si_i, v_si_ii, alpha, v_y):
        return 0.5 * (1.0 - alpha) / v_y - 0.5 / alpha / (v_si_ii + v_y/alpha) \
               + 0.5 * (y_i - m_si_i)**2.0 / (v_si_ii + v_y/alpha)**2 / alpha 

    def gauss_dlogZtilted_dv(self, y_i, m_si_i, v_si_ii, alpha, v_y):
        return - 0.5 / (v_y / alpha + v_si_ii) + 0.5 * (y_i - m_si_i)**2.0 / (v_si_ii + v_y/alpha)**2

    def gauss_dlogZtilted_dm2(self, y_i, m_si_i, v_si_ii, alpha, v_y):
        return -1./(v_y/alpha+v_si_ii)

    __gh_points = None
    def _gh_points(self, T=20):
        if self.__gh_points is None:
            self.__gh_points = np.polynomial.hermite.hermgauss(T)
        return self.__gh_points

    def cdf_logZtilted(self, y, m, v, alpha, deg):

        logZtitled = 0.0
        if alpha == 1.0:
            t = y * m / np.sqrt(1+v)
            Z = 0.5 * (1 + math.erf(t / np.sqrt(2)))
            eps = 1e-16
            logZtilted = np.log(Z + eps)
        else:
            gh_x, gh_w = self._gh_points(deg)
            ts = gh_x * np.sqrt(2*v) + m
            pdfs = 0.5 * (1 + erf(y*ts / np.sqrt(2)))
            logZtilted = np.log(np.dot(pdfs**alpha, gh_w) / np.sqrt(np.pi)) 

        return logZtilted 

    def cdf_dlogZtilted_dm(self, y, m, v, alpha, deg):
        dm = 0.0
        if alpha == 1.0:
            t = y * m / np.sqrt(1 + v)
            Z = 0.5 * (1 + math.erf(t / np.sqrt(2)))
            eps = 1e-16
            Zeps = Z + eps
            beta = 1 / Zeps / np.sqrt(1 + v) * 1/np.sqrt(2*np.pi) * np.exp(-t**2.0 / 2)
            dm = y*beta
        else:
            
            gh_x, gh_w = self._gh_points(deg)   
            eps = 1e-8
            ts = gh_x * np.sqrt(2*v) + m
            pdfs = 0.5 * (1 + erf(y*ts / np.sqrt(2))) + eps
            Ztilted = np.dot(pdfs**alpha, gh_w) / np.sqrt(np.pi)
            dZdm = np.dot(gh_w, pdfs**(alpha-1.0)*np.exp(-ts**2/2)) * y * alpha / np.pi / np.sqrt(2)
            dm = dZdm / Ztilted + eps

        return dm

    def cdf_dlogZtilted_dv(self, y, m, v, alpha, deg):
        dv = 0.0
        if alpha == 1.0:
            t = y * m / np.sqrt(1 + v)
            Z = 0.5 * (1 + math.erf(t / np.sqrt(2)))
            eps = 1e-16
            Zeps = Z + eps
            dv = - 0.5 * y * m / Zeps / (1 + v)**1.5 * 1/np.sqrt(2*np.pi) * np.exp(-t**2.0 / 2)
        else:
            
            gh_x, gh_w = self._gh_points(deg)   
            eps = 1e-8    
            ts = gh_x * np.sqrt(2*v) + m
            pdfs = 0.5 * (1 + erf(y*ts / np.sqrt(2))) + eps
            Ztilted = np.dot(pdfs**alpha, gh_w) / np.sqrt(np.pi)
            dZdv = np.dot(gh_w, pdfs**(alpha-1.0)*np.exp(-ts**2/2) * gh_x) * y * alpha / np.pi / np.sqrt(2) / np.sqrt(2*v)
            dv = dZdv / Ztilted + eps

        return dv
 
    def cdf_dlogZtilted_dm2(self, y, m, v, alpha, deg):
        dm2 = 0
        if alpha == 1.0:
            t = y * m / np.sqrt(1 + v)
            Z = 0.5 * (1 + math.erf(t / np.sqrt(2)))
            eps = 1e-16
            Zeps = Z + eps
            beta = 1 / Zeps / np.sqrt(1 + v) * 1/np.sqrt(2*np.pi) * np.exp(-t**2.0 / 2)
            dm2 = - (beta**2 + m*y*beta/(1+v))
        else:
            
            gh_x, gh_w = self._gh_points(deg)
            eps = 1e-8
            ts = gh_x * np.sqrt(2*v) + m
            pdfs = 0.5 * (1 + erf(y*ts / np.sqrt(2))) + eps
            Ztilted = np.dot(pdfs**alpha, gh_w) / np.sqrt(np.pi)
            dZdm = np.dot(gh_w, pdfs**(alpha-1)*np.exp(-ts**2/2)) * y * alpha / np.pi / np.sqrt(2)
            dZdm2 = np.dot(gh_w, (alpha-1)*pdfs**(alpha-2)*np.exp(-ts**2)/np.sqrt(2*np.pi)  
                - pdfs**(alpha-1) * y * ts * np.exp(-ts**2/2) ) * alpha / np.pi / np.sqrt(2)

            dm2 = -dZdm**2 / Ztilted**2 + dZdm2 / Ztilted + eps
        return dm2
 
    def init_adam(self, adamlrate=0.001):
        alpha = {'ls': adamlrate,
                 'sf': adamlrate,
                 'sn': adamlrate,
                 'zu': adamlrate}
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        # init 1st moment and 2nd moment vectors
        v_zu = np.zeros((self.M, self.D))
        v_ls = np.zeros((self.D, ))
        v_sf = 0
        v_sn = 0
        v_all = {'zu': v_zu, 'ls': v_ls,
                 'sf': v_sf, 'sn': v_sn}

        m_all = copy.deepcopy(v_all)
        m_hat = copy.deepcopy(v_all)
        v_hat = copy.deepcopy(v_all)

        adamobj = {'alpha': alpha, 'beta1': beta1, 'beta2': beta2, 'eps': eps,
                   'm': m_all, 'v': v_all, 'm_hat': m_hat, 'v_hat': v_hat}

        return adamobj

    def set_adam_lrate(self, adamobj, new_lrate):
        for name in ['ls', 'sf', 'sn', 'zu']:
            adamobj['alpha'][name] = new_lrate
        return adamobj

    def update_adam(self, adamobj, params, grad, iterno, dec_rate=1.0, fixed=[]):
        # update ADAM params and model params
        param_names = ['ls', 'sf', 'sn', 'zu']

        alpha = adamobj['alpha']
        beta1 = adamobj['beta1']
        beta2 = adamobj['beta2']
        eps = adamobj['eps']

        for name in param_names:
            if not (name in fixed):
                # get gradients
                g = grad[name]
                # compute running average of grad and grad^2
                # update biased first moment estimate
                adamobj['m'][name] = beta1 * adamobj['m'][name] + \
                                        (1.0 - beta1) * g
                # update biased second moment estimate
                adamobj['v'][name] = beta2 * adamobj['v'][name] + \
                                        (1.0 - beta2) * g**2.0
                # compute bias-corrected first and second moment estimates
                adamobj['m_hat'][name] = adamobj['m'][name] / (1.0 - beta1**iterno)
                adamobj['v_hat'][name] = adamobj['v'][name] / (1.0 - beta2**iterno)
                # update model params
                curval = params[name]
                delta = dec_rate * alpha[name] * adamobj['m_hat'][name] / \
                    (np.sqrt(adamobj['v_hat'][name]) + eps)
                params[name] = curval + delta



        return params
