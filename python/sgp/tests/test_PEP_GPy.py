from ..pep.PEP_reg import PEP
import GPy
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

np.random.seed(20)


N = 20 # Number of data points
M = 2 # Number of inducing points
X = np.c_[np.linspace(0, 10, N)] # Data X values
X_B = np.c_[(np.max(X)-np.min(X))*np.random.uniform(0, 1, M)+np.min(X)] + 2
lik_noise_var = 0.1
X_T = np.c_[np.linspace(0,10, 100)] # use the same covariance matrix!
X_X_T = np.vstack((X, X_T))
k = GPy.kern.RBF(input_dim=1, lengthscale=1, variance=1)
Y_full = np.c_[np.random.multivariate_normal(
        np.zeros(X_X_T.shape[0]), 
        k.K(X_X_T)+np.eye(X_X_T.shape[0])*lik_noise_var)]
Y = np.c_[Y_full[:N]]
Y_T = np.c_[Y_full[N:]]


plt.figure(figsize=(20,10))
plt.scatter(X, Y, color='k')
X_plot = np.c_[np.linspace(-2, 12, 500)]

k = GPy.kern.RBF(input_dim=1, lengthscale=1, variance=1)
model = GPy.models.SparseGPRegression(X,Y,kernel=k,Z=X_B)
model.name = 'VFE'
model.Gaussian_noise.variance = lik_noise_var
model.unfix()
model.Z.unconstrain()
model.optimize('bfgs', messages=True, max_iters=2e3)
(m, V) = model.predict(X_plot, full_cov=False)
plt.plot(X_plot, m,'g', label='VFE')
plt.plot(X_plot, m+2*np.sqrt(V),'g--')
plt.plot(X_plot, m-2*np.sqrt(V),'g--')

(m_B, V_B) = model.predict(X_B, full_cov=False)
plt.scatter(X_B, m_B, color='g')

vfe_lml = model.log_likelihood()
print 'VFE: ', vfe_lml


k = GPy.kern.RBF(input_dim=1, lengthscale=1, variance=1)
model = GPy.models.SparseGPRegression(X,Y,kernel=k,Z=X_B)
model.name = 'FITC'
model.inference_method = GPy.inference.latent_function_inference.FITC()
model.Gaussian_noise.variance = lik_noise_var
model.unfix()
model.Z.unconstrain()
model.optimize('bfgs', messages=True, max_iters=2e3)
(m, V) = model.predict(X_plot, full_cov=False)
plt.plot(X_plot, m,'b', label='FITC')
plt.plot(X_plot, m+2*np.sqrt(V),'b--')
plt.plot(X_plot, m-2*np.sqrt(V),'b--')

(m_B, V_B) = model.predict(X_B, full_cov=False)
plt.scatter(X_B, m_B, color='b')

fitc_lml = model.log_likelihood()
print 'FITC: ', fitc_lml


alpha = 0.5
k = GPy.kern.RBF(input_dim=1, lengthscale=1, variance=1)
model = GPy.models.SparseGPRegression(X,Y,kernel=k,Z=X_B)
model.name = 'POWER-EP'
model.inference_method = PEP(alpha=alpha)
model.Gaussian_noise.variance = lik_noise_var
model.unfix()
# print model.checkgrad()
model.optimize('bfgs', messages=True, max_iters=2e3)
# model.optimize_restarts(num_restarts = 10)
(m, V) = model.predict(X_plot, full_cov=False)
plt.plot(X_plot, m,'r', label='Power-EP, alpha %.2f' % alpha)
plt.plot(X_plot, m+2*np.sqrt(V),'r--')
plt.plot(X_plot, m-2*np.sqrt(V),'r--')

(m_B, V_B) = model.predict(X_B, full_cov=False)
plt.scatter(X_B, m_B, color='r')

pep_lml = model.log_likelihood()
print 'Power EP: ', pep_lml

k = GPy.kern.RBF(input_dim=1, lengthscale=1, variance=1)
model = GPy.models.GPRegression(X,Y,k, noise_var=lik_noise_var)
model.name = 'FULL'
model.Gaussian_noise.variance = lik_noise_var
model.unfix()
# print model.checkgrad()
model.optimize('bfgs', messages=True, max_iters=2e3)
# model.optimize_restarts(num_restarts = 10)
(m, V) = model.predict(X_plot, full_cov=False)
plt.plot(X_plot, m,'k', label='FULL GP')
plt.plot(X_plot, m+2*np.sqrt(V),'k--')
plt.plot(X_plot, m-2*np.sqrt(V),'k--')
full_lml = model.log_likelihood()
print 'FULL: ', full_lml
plt.legend()

plt.show()