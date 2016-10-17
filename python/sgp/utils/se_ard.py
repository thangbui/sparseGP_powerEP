import numpy as np
import scipy.linalg   as spla
from scipy.spatial.distance import cdist


def compute_kernel(lls, lsf, x, z):
    ls = np.exp(lls)
    sf = np.exp(lsf)

    if x.ndim == 1:
        x = x[ None, : ]

    if z.ndim == 1:
        z = z[ None, : ]

    r2 = cdist(x, z, 'seuclidean', V = ls)**2.0  
    k = sf * np.exp(-0.5*r2)
    return k

def compute_kernel_diag(lls, lsf, x):
    ls = np.exp(lls)
    sf = np.exp(lsf)

    if x.ndim == 1:
        x = x[ None, : ]

    k = sf * np.ones((x.shape[0], ))
    return k


def d_trace_MKzz_dhypers(lls, lsf, z, M, Kzz):

    dKzz_dlsf = Kzz
    ls = np.exp(lls)

    # This is extracted from the R-code of Scalable EP for GP Classification by DHL and JMHL

    gr_lsf = np.sum(M * dKzz_dlsf)

    # This uses the vact that the distance is v^21^T - vv^T + 1v^2^T, where v is a vector with the l-dimension
    # of the inducing points. 

    Ml = 0.5 * M * Kzz
    Xl = z * np.outer(np.ones(z.shape[ 0 ]), 1.0 / np.sqrt(ls))
    gr_lls = np.dot(np.ones(Ml.shape[ 0 ]), np.dot(Ml.T, Xl**2)) + np.dot(np.ones(Ml.shape[ 0 ]), np.dot(Ml, Xl**2)) \
    - 2.0 * np.dot(np.ones(Xl.shape[ 0 ]), (Xl * np.dot(Ml, Xl)))

    Xbar = z * np.outer(np.ones(z.shape[ 0 ]), 1.0 / ls)
    Mbar1 = - M.T * Kzz
    Mbar2 = - M * Kzz
    gr_z = (Xbar * np.outer(np.dot(np.ones(Mbar1.shape[ 0 ]) , Mbar1), np.ones(Xbar.shape[ 1 ])) - np.dot(Mbar1, Xbar)) +\
        (Xbar * np.outer(np.dot(np.ones(Mbar2.shape[ 0 ]) , Mbar2), np.ones(Xbar.shape[ 1 ])) - np.dot(Mbar2, Xbar))

    # The cost of this function is dominated by five matrix multiplications with cost M^2 * D each where D is 
    # the dimensionality of the data!!!

    return gr_lsf, gr_lls, gr_z