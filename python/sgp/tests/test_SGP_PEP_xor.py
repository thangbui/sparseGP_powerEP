from ..pep.SGP_PEP import SGP_PEP

import math
import numpy as np
import scipy.stats as stats
from ..utils.metrics import *

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time
np.random.seed(1)



M = 30
no_epochs = 200
minibatch_size = 200
no_ep_sweeps = 1
alpha = 0.8

# creating xor datasets
N_train = 500
x = np.random.randn(N_train, 2)
y = np.reshape(1 * np.logical_xor(x[:, 0] > 0, x[:, 1] > 0), [N_train, 1])
y[np.where(y == 0)] = -1

X_train = x
X_test = x
y_train = y
y_test = y

t0 = time.time()

s = SGP_PEP(X_train, y_train, M, alpha, lik='cdf')
test_nll, test_err, energies = s.train(no_epochs=no_epochs, n_per_mb=minibatch_size, reinit_hypers=True, 
    no_ep_sweeps=no_ep_sweeps, lrate=0.001, fixed_params=[], 
    compute_test=True, print_trace=True, Xtest=X_test, ytest=y_test)

t1 = time.time()
total_time = t1 - t0
print 'training time %.3f' % total_time

plt.figure()
plt.plot(np.array(test_nll))
plt.xlabel('iteration')
plt.ylabel('nll')

plt.figure()
plt.plot(np.array(test_err))
plt.xlabel('iteration')
plt.ylabel('error')

plt.figure()
plt.plot(np.array(energies))
plt.xlabel('iteration')
plt.ylabel('logZ')

# We make predictions for the test set
mf, vf = s.predict_diag(X_test)
# We compute the test error and log lik
test_nll = compute_nll(y_test, mf, vf, s.lik)
test_error = compute_error(y_test, mf, vf, s.lik)
print 'test_error: %.3f' % test_error
print 'test nll: %.3f' % test_nll

# Define a class that forces representation of float to look a certain way
# This remove trailing zero so '1.0' becomes '1'
class nf(float):
     def __repr__(self):
         str = '%.1f' % (self.__float__(),)
         if str[-1]=='0':
             return '%.0f' % self.__float__()
         else:
             return '%.1f' % self.__float__()

delta = 0.05
x1 = np.arange(-3, 3, delta)
x2 = np.arange(-3, 3, delta)
X1, X2 = np.meshgrid(x1, x2)
X11 = X1.reshape([X1.shape[0]*X1.shape[1], 1])
X22 = X2.reshape([X2.shape[0]*X2.shape[1], 1])
X_plot = np.concatenate((X11, X22), axis=1)

m, v = s.predict_diag(X_plot)
pred_prob = stats.norm.cdf(m / np.sqrt(1 + v))
pred_prob = np.reshape(pred_prob, (X1.shape[0], X1.shape[1]))
# levels = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
levels = np.array([0.5])

plt.figure()
plt.imshow(pred_prob, interpolation='nearest',
           extent=(X1.min(), X1.max(), X2.min(), X2.max()), aspect='auto',
           origin='lower', cmap=plt.cm.Paired)
cs = plt.contour(X1, X2, pred_prob, levels, cmap=cm.brg, linewidths=2,
                       linetypes='--')
plt.scatter(X_train[:, 0], X_train[:, 1], marker='o', s=25, c=y_train, cmap=plt.cm.Paired)
# plt.scatter(X_test[:, 0], X_test[:, 1], marker='o', s=25, c=y_test, cmap=plt.cm.Paired)
    
zud = s.zu
plt.scatter(zud[:, 0], zud[:, 1], marker='o', s=25,  c='k')
cs.levels = [nf(val) for val in cs.levels ]
# Label levels with specially formatted floats
if plt.rcParams["text.usetex"]:
     fmt = r'%r'
else:
     fmt = '%r'
plt.clabel(cs, cs.levels, inline=True, fmt=fmt, fontsize=10)
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.xticks(())
plt.yticks(())
plt.xlim(np.min(x1), np.max(x1))
plt.ylim(np.min(x2), np.max(x2))
plt.show()
