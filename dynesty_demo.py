# Python 3 compatability
from __future__ import division, print_function
from six.moves import range

# system functions that are always useful to have
import time, sys, os

# basic numeric setup
import numpy as np
from numpy import linalg

# inline plotting
# matplotlib inline

# plotting
import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# seed the random number generator
np.random.seed(5647)

# re-defining plotting defaults
from matplotlib import rcParams
rcParams.update({'xtick.major.pad': '7.0'})
rcParams.update({'xtick.major.size': '7.5'})
rcParams.update({'xtick.major.width': '1.5'})
rcParams.update({'xtick.minor.pad': '7.0'})
rcParams.update({'xtick.minor.size': '3.5'})
rcParams.update({'xtick.minor.width': '1.0'})
rcParams.update({'ytick.major.pad': '7.0'})
rcParams.update({'ytick.major.size': '7.5'})
rcParams.update({'ytick.major.width': '1.5'})
rcParams.update({'ytick.minor.pad': '7.0'})
rcParams.update({'ytick.minor.size': '3.5'})
rcParams.update({'ytick.minor.width': '1.0'})
rcParams.update({'font.size': 30})

import dynesty
ndim = 3  # number of dimensions
C = np.identity(ndim)  # set covariance to identity matrix
C[C==0] = 0.95  # set off-diagonal terms (strongly correlated)
Cinv = linalg.inv(C)  # precision matrix
lnorm = -0.5 * (np.log(2 * np.pi) * ndim + np.log(linalg.det(C)))  # ln(normalization)

# 3-D correlated multivariate normal log-likelihood
def loglikelihood(x):
    """Multivariate normal log-likelihood."""
    return -0.5 * np.dot(x, np.dot(Cinv, x)) + lnorm

# prior transform
def prior_transform(u):
    """Transforms our unit cube samples `u` to a flat prior between -10. and 10. in each variable."""
    return 10. * (2. * u - 1.)

# gradient of log-likelihood *with respect to u*
# -> d(lnl)/du = d(lnl)/dv * dv/du
# dv/du = 1. / prior(v)
def gradient(x):
    """Multivariate normal log-likelihood gradient."""
    dlnl_dv = -np.dot(Cinv, x)  # standard gradient
    jac = np.diag(np.full_like(x, 20.))  # Jacobian
    return np.dot(jac, dlnl_dv)  # transformed gradient

# initialize our nested sampler
sampler = dynesty.NestedSampler(loglikelihood, prior_transform, ndim, nlive=1500)

# sample from the target distribution
sampler.run_nested()

res = sampler.results  # grab our results
print('Keys:', res.keys(),'\n')  # print accessible keys
res.summary()  # print a summary


# ADDED BY ME TO TRY SAVING THINGS
import pickle
direc = '/Users/jonathancohn/Documents/dyn_mod/nest_out/'
out_name = direc + 'dynesty_demo_output.pkl'
with open(out_name, 'wb+') as newfile:  # 'wb' because binary format
    pickle.dump(res, newfile, pickle.HIGHEST_PROTOCOL)  # res.samples
    print('results pickle dumped!')

with open(out_name, 'rb') as pk:
    u = pickle._Unpickler(pk)
    u.encoding = 'latin1'
    dyn_res = u.load()
print(dyn_res['samples'].shape)

# 3-D plots of position and likelihood, colored by weight
fig = plt.figure(figsize=(30, 10))
ax = fig.add_subplot(121, projection='3d')

# plotting the initial run
p = ax.scatter(res.samples[:, 0], res.samples[:, 1], res.samples[:, 2],
               marker='o', c=np.exp(res.logwt) * 1e7, linewidths=(0.,), cmap='coolwarm')
ax.set_xlim(-10., 10.)
ax.set_xticks(np.linspace(-10., 10., 5))
ax.set_xlabel(r'$x$', labelpad=25)
ax.set_ylim(-10., 10.)
ax.set_yticks(np.linspace(-10., 10., 5))
ax.set_ylabel(r'$y$', labelpad=25)
ax.set_zlim(-10., 10.)
ax.set_zticks(np.linspace(-10., 10., 5))
ax.set_zlabel(r'$z$', labelpad=25)
ax.set_title('Initial Run')
cb = fig.colorbar(p)
cb.set_label('Weight (1e-6)', labelpad=50., rotation=270.)
plt.tight_layout()
plt.show()

# plotting the extended run
fig = plt.figure(figsize=(30, 10))
ax = fig.add_subplot(121, projection='3d')
p = ax.scatter(dyn_res['samples'][:, 0], dyn_res['samples'][:, 1], dyn_res['samples'][:, 2],
               marker='o', c=np.exp(dyn_res['logwt']) * 1e7, linewidths=(0.,), cmap='coolwarm')
ax.set_xlim(-10., 10.)
ax.set_xticks(np.linspace(-10., 10., 5))
ax.set_xlabel(r'$x$', labelpad=25)
ax.set_ylim(-10., 10.)
ax.set_yticks(np.linspace(-10., 10., 5))
ax.set_ylabel(r'$y$', labelpad=25)
ax.set_zlim(-10., 10.)
ax.set_zticks(np.linspace(-10., 10., 5))
ax.set_zlabel(r'$z$', labelpad=25)
ax.set_title('Initial Run')
cb = fig.colorbar(p)
cb.set_label('Weight (1e-6)', labelpad=50., rotation=270.)
plt.tight_layout()
plt.show()

# How to do quantiles!
from dynesty import utils as dyfunc
weights = np.exp(dyn_res['logwt'] - dyn_res['logz'][-1])  # normalized weights
for i in range(dyn_res['samples'].shape[1]):  # for each parameter
    quantiles_3 = dyfunc.quantile(dyn_res['samples'][:, i], [0.025, 0.5, 0.975], weights=weights)
    quantiles_1 = dyfunc.quantile(dyn_res['samples'][:, i], [0.16, 0.5, 0.84], weights=weights)
    print(quantiles_3)
    print(quantiles_1)

from dynesty import plotting as dyplot
# initialize figure
fig, axes = plt.subplots(3, 7, figsize=(35, 15))
axes = axes.reshape((3, 7))  # reshape axes

# add white space
[a.set_frame_on(False) for a in axes[:, 3]]
[a.set_xticks([]) for a in axes[:, 3]]
[a.set_yticks([]) for a in axes[:, 3]]

# plot initial run (res1; left)
fg, ax = dyplot.cornerplot(res, color='blue', truths=np.zeros(ndim),
                           truth_color='black', show_titles=True,
                           max_n_ticks=3, quantiles=None,
                           fig=(fig, axes[:, :3]))

# plot extended run (res2; right)
fg, ax = dyplot.cornerplot(dyn_res, color='dodgerblue', truths=np.zeros(ndim),
                           truth_color='black', show_titles=True,
                           quantiles=None, max_n_ticks=3,
                           fig=(fig, axes[:, 4:]))
plt.show()
