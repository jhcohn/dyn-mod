# Python 3 compatability
# from __future__ import division, print_function
# from six.moves import range

import dynesty
import pickle
#import _pickle as pickle

# basic numeric setup
import numpy as np

# plotting
import matplotlib
from matplotlib import pyplot as plt

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
rcParams.update({'font.size': 15})


def my_own_thing(results, par_labels, ax_labels, quantiles, ax_lims=None):
    # results should be dyn_res['samples']
    roundto = 2  # 4
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))  # 3 rows, 3 cols of subplots; because there are 9 free params
    fs = 12
    labels = np.array(['mbh', 'xloc', 'yloc', 'sig0', 'inc', 'PAdisk', 'vsys', 'ml_ratio', 'f'])
    axes_order = [[0, 0], [2, 0], [2, 1], [2, 2], [1, 0], [1, 1], [1, 2], [0, 1], [0, 2]]
    for i in range(len(results[0])):
        row, col = axes_order[i]
        if labels[i] == 'mbh' or labels[i] == 'PAdisk':
            bins = 1000
        else:
            bins = 100
        chain = results[:, i]
        weight = np.ones_like(chain) * 2e-3
        axes[row, col].hist(chain, bins=bins, color="b", histtype="step", weights=weight)  # axes[i]
        print(quantiles[i], 'look')
        # percs = np.percentile(chain, [.15, 50., 99.85])  # 3sigma
        # axes[row, col].axvline(percs[1], color='b', ls='--')  # axes[i]
        # axes[row, col].axvspan(percs[0], percs[2], color='b', alpha=0.25)
        axes[row, col].axvline(quantiles[i][1], color='b', ls='--')  # axes[i]
        axes[row, col].axvspan(quantiles[i][0], quantiles[i][2], color='b', alpha=0.25)
        axes[row, col].tick_params('both', labelsize=fs)
        axes[row, col].set_title(par_labels[i] + ': ' + str(round(quantiles[i][1], roundto)) + ' (+'
                                 + str(round(quantiles[i][2] - quantiles[i][1], roundto)) + ', -'
                                 + str(round(quantiles[i][1] - quantiles[i][0], roundto)) + ')', fontsize=fs)
        axes[row, col].set_xlabel(ax_labels[i], fontsize=fs)
        if ax_lims is not None:
            axes[row, col].set_xlim(ax_lims[i][0], ax_lims[i][1])
    plt.tight_layout()
    plt.show()

direc = '/Users/jonathancohn/Documents/dyn_mod/nest_out/'
# OLDER
#out_name = direc + 'dyndyncluster_test_n8_1568739677.2582278_tempsave.pkl'  # 'dynesty_demo_output.pkl'
#out_name = direc + 'dyndyncluster_test_n8_1568739677.1305604_tempsave.pkl'
#out_name = direc + 'dyndyn2_test_n8_1568739677.1305604_tempsave.pkl'

# WORKED
# out_name = direc + 'dyndyncluster_test_n8_1568739677.2582278_tempsave.pkl'  # 2698 old prior
# out_name = direc + 'dyndyn3258_test_n8_dlogz1.5_1569249221.642805_tempsave.pkl'  # 3258 old TINY prior
# out_name = direc + 'dyndyn3258_test_n8_dlogz1.5_1569249222.4063346_tempsave.pkl'  # 3258 old TINY prior (same as above)
# out_name = direc + 'dyndyn3258_test_n8_dlogz15_1569434823.359717_tempsave.pkl'  # 3258 old TINY prior
# out_name = direc + 'dyndyn3258_newpri_test_n8_dlogz1.5_1569620562.4751067_tempsave.pkl'  # 3258 newprior TYPO (PRIOR for f was BAD 0.5<f<0.65) (AHA!)
# out_name = direc + 'dyndyn3258_newpri_test_n8_dlogz1.5_1569620562.355544_tempsave.pkl'  # 3258 newprior TYPO same as above!
# out_name = direc + 'dyndyn_newpri_test_n8_1569798600.5749478_end.pkl' # 2698 newprior GOOD (altho typo 1e8<mbh<1e18)
# out_name = direc + 'dyndyn_newpri_test_n8_1569620951.5581834_tempsave.pkl' # 2698 newprior GOOD (same as above)
# out_name = direc + 'dyndyn105newpri_test_n8_1569855263.300289_tempsave.pkl'  # 2698 newprior 4x8(?) REALLY BAD WHY
# out_name = direc + 'dyndyn3258_test_n8_dlogz1.5_1570221683.4872622_tempsave.pkl'  # 3258 old-prior style good
# out_name = direc + 'dyndyn3258_newpri_test_n8_dlogz1.5_1570464872.8310175_tempsave.pkl'  # 3258 newprior BAD WHY
# out_name = direc + 'dyndyn105newpri_test_n8_1570545604.3403502_tempsave.pkl'  # 2698 newprior 4x8(?) REALLY BAD WHY
# out_name = direc + 'dyndyn105newpri_test_n8_ds3_1570633918.9553041_tempsave.pkl'  # 2698 neworior 3x6 REALLY BAD WHY
# out_name = direc + 'dyndyn105newpri_test_n8_1570545604.3403502_tempsave.pkl'  # 2698 neworior 4x8 REALLY BAD WHY
# out_name = direc + 'dyndyn3258_newpri_2_test_n8_dlogz0.15_1570807500.0458014_tempsave.pkl'  # 3258 newprior 2 REALLY BAD STILL WHYYYY (BECAUSE MAXCALL IS REACHED!!!)
out_name = direc + 'dyndyn3258_narrowpri_n8_dlogz1.5_1571063422.9054961_tempsave.pkl'  # 3258 narrow priors, yep still works

with open(out_name, 'rb') as pk:
    u = pickle._Unpickler(pk)
    u.encoding = 'latin1'
    dyn_res = u.load()  # pickle.load(pk)  #
print(dyn_res['samples'].shape)

# 3-D plots of position and likelihood, colored by weight
# fig = plt.figure(figsize=(30, 10))
# ax = fig.add_subplot(121, projection='3d')

# How to do quantiles!
labels = np.array(['mbh', 'xloc', 'yloc', 'sig0', 'inc', 'PAdisk', 'vsys', 'ml_ratio', 'f'])
ax_lab = np.array([r'$\log_{10}$(M$_{\odot}$)', 'pixels', 'pixels', 'km/s', 'deg', 'deg', 'km/s',
                   r'M$_{\odot}$/L$_{\odot}$', 'unitless'])
from dynesty import utils as dyfunc
weights = np.exp(dyn_res['logwt'] - dyn_res['logz'][-1])  # normalized weights
three_sigs = []
for i in range(dyn_res['samples'].shape[1]):  # for each parameter
    quantiles_3 = dyfunc.quantile(dyn_res['samples'][:, i], [0.0015, 0.5, 0.9985], weights=weights)
    quantiles_2 = dyfunc.quantile(dyn_res['samples'][:, i], [0.025, 0.5, 0.975], weights=weights)
    quantiles_1 = dyfunc.quantile(dyn_res['samples'][:, i], [0.16, 0.5, 0.84], weights=weights)
    print(labels[i])
    if i == 0:
        print(np.log10(quantiles_3), quantiles_3)
        print(np.log10(quantiles_2), quantiles_2)
        print(np.log10(quantiles_1), quantiles_1)
        three_sigs.append(np.log10(quantiles_3))
    else:
        print(quantiles_3)
        print(quantiles_2)
        print(quantiles_1)
        three_sigs.append(quantiles_3)

from dynesty import plotting as dyplot

'''
# initialize figure
fig, axes = plt.subplots(3, 7, figsize=(35, 15))
axes = axes.reshape((3, 7))  # reshape axes

# add white space
[a.set_frame_on(False) for a in axes[:, 3]]
[a.set_xticks([]) for a in axes[:, 3]]
[a.set_yticks([]) for a in axes[:, 3]]
# '''
sig1 = [0.16, 0.5, 0.84]
sig2 = [0.025, 0.5, 0.975]
sig3 = [0.0015, 0.5, 0.9985]

vax = np.zeros(shape=len(labels))
vwidth = np.zeros(shape=(len(labels), 2))
with open('/Users/jonathancohn/Documents/dyn_mod/param_files/Ben_A1_errors.txt') as a1:
    for line in a1:
        cols = line.split()
        if not line.startswith('#'):
            if cols[0] == 'mbh':
                vax[np.where(labels == cols[0])] = np.log10(float(cols[1]))
                vwidth[np.where(labels == cols[0]), 0] = np.log10(float(cols[1]) - float(cols[2]))
                vwidth[np.where(labels == cols[0]), 1] = np.log10(float(cols[1]) + float(cols[2]))
            else:
                vax[np.where(labels == cols[0])] = float(cols[1])
                vwidth[np.where(labels == cols[0]), 0] = float(cols[1]) - float(cols[2])
                vwidth[np.where(labels == cols[0]), 1] = float(cols[1]) + float(cols[2])
print(vax)
print(vwidth)
print(vax[labels=='sig0'])
print(vwidth[labels=='sig0'])

logm = True
if logm:
    dyn_res['samples'][:, 0] = np.log10(dyn_res['samples'][:, 0])
    labels[0] = 'log mbh'  # r'log$_{10}$mbh'

ax_lims = None
if '3258' in out_name:
    ax_lims = [[9.2, 9.6], [126.2, 127.8], [150.2, 151.8], [7.5, 14.], [66., 70.], [19.13, 19.23], [6447., 6462.],
               [1.52, 1.8], [0.93, 1.2]]
    ax_lims = None
else:
    ax_lims = [[9.3, 9.6], [126.2, 127.8], [150.2, 151.8], [7.5, 14.], [66., 70.], [19.13, 19.23], [6447., 6462.],
               [1.52, 1.8], [0.93, 1.2]]
    #ax_lims=None

my_own_thing(dyn_res['samples'], labels, ax_lab, three_sigs, ax_lims=ax_lims)
print(oop)

# plot initial run (res1; left)

# TO EDIT SOURCE CODE: open /Users/jonathancohn/anaconda3/envs/three/lib/python3.6/site-packages/dynesty/plotting.py

# USE THIS FOR COMPARING 3258 TO BEN'S A1 MODEL:
#fg, ax = dyplot.cornerplot(dyn_res, color='blue', show_titles=True, max_n_ticks=3, quantiles=sig1, labels=labels,
#                           compare_med=vax, compare_width=vwidth)
#plt.show()
# OTHERWISE USE THIS:
fg, ax = dyplot.cornerplot(dyn_res, color='blue', show_titles=True, max_n_ticks=3, quantiles=sig1, labels=labels)

# fig=(fig, axes), truths=np.zeros(ndim), truth_color='black',

plt.show()


'''
# 3258 dyndyn3258 priors:
    # NARROWEST PRIORS based on short MCMC
    cube[0] = 10 ** (cube[0] * 0.1 + 9.3)  # mbh: log-uniform prior 1e9.3:1e9.4
    cube[1] = cube[1] + 361.5  # xloc: uniform prior 361.5:362.5
    cube[2] = cube[2] * 0.5 + 354.75  # yloc: uniform prior 354.75:355.25
    cube[3] = cube[3] * 10. + 5.  # sig0: uniform prior 5:15
    cube[4] = cube[4] * 2. + 45.4  # inc: uniform prior 45.4:47.4 (low pri=49.83663935008522 from MGE q)
    cube[5] = cube[5] * 5. + 166.25  # PAdisk: uniform prior 166.25:167.25
    cube[6] = cube[6] * 2. + 2760.  # vsys: uniform prior 2760:2762
    cube[7] = cube[7] * 0.2 + 3.05  # mlratio: uniform prior 3.05:3.25
    cube[8] = cube[8] * 0.03 + 1.005  # f: uniform prior 1.005:1.035

# 3258 newpri 2
    cube[0] = 10 ** (cube[0] * 2. + 8.)  # mbh: log-uniform prior 1e9.3:1e9.4
    cube[1] = cube[1] * 5. + 360.  # xloc: uniform prior 360:365
    cube[2] = cube[2] * 3. + 353.  # yloc: uniform prior 353:356
    cube[3] = cube[3] * 15.  # sig0: uniform prior 0:15
    cube[4] = cube[4] * 40. + 45.  # inc: uniform prior 44.5:89.5 (low pri=44.01472043806415 from MGE q)
    cube[5] = cube[5] * 110. + 100.  # PAdisk: uniform prior 100:210
    cube[6] = cube[6] * 50. + 2730.  # vsys: uniform prior 2730:2780
    cube[7] = cube[7] * 5. + 0.5  # mlratio: uniform prior 0.5:5.5
    cube[8] = cube[8] + 0.5  # f: uniform prior 0.5:1.5

# 3258 newpri 3
    cube[0] = 10 ** (cube[0] * 2. + 8.)  # mbh: log-uniform prior 1e9.3:1e9.4
    cube[1] = cube[1] * 15. + 355.  # xloc: uniform prior 360:365
    cube[2] = cube[2] * 15. + 347.  # yloc: uniform prior 353:356
    cube[3] = cube[3] * 15.  # sig0: uniform prior 0:15
    cube[4] = cube[4] * 30. + 46.  # inc: uniform prior 44.5:89.5 (low pri=44.01472043806415 from MGE q)
    cube[5] = cube[5] * 100. + 100.  # PAdisk: uniform prior 100:210
    cube[6] = cube[6] * 50. + 2730.  # vsys: uniform prior 2730:2780
    cube[7] = cube[7] * 5. + 0.5  # mlratio: uniform prior 0.5:5.5
    cube[8] = cube[8] + 0.5  # f: uniform prior 0.5:1.5

# 3258 newpri 4
    # NARROWEST PRIORS based on short MCMC
    cube[0] = 10 ** (cube[0] * 2. + 8.)  # mbh: log-uniform prior 1e9.3:1e9.4
    cube[1] = cube[1] * 5. + 360.  # xloc: uniform prior 360:365
    cube[2] = cube[2] * 9. + 351.  # yloc: uniform prior 353:356
    cube[3] = cube[3] * 15.  # sig0: uniform prior 0:15
    cube[4] = cube[4] * 20. + 45.  # inc: uniform prior 44.5:89.5 (low pri=44.01472043806415 from MGE q)
    cube[5] = cube[5] * 45. + 135.  # PAdisk: uniform prior 100:210
    cube[6] = cube[6] * 50. + 2730.  # vsys: uniform prior 2730:2780
    cube[7] = cube[7] * 3. + 1.  # mlratio: uniform prior 0.5:5.5
    cube[8] = cube[8] + 0.5  # f: uniform prior 0.5:1.5
'''

'''
# 2698 dyndyn2 priors:
    # NARROWEST PRIORS based on short MCMC
    cube[0] = 10 ** (cube[0] * 0.8 + 9.2)  # mbh: log-uniform prior 10^9.2:10^10
    cube[1] = cube[1] * 3.5 + 126.  # xloc: uniform prior 126:129.5
    cube[2] = cube[2] * 1.75 + 150.25  # yloc: uniform prior 150.25:152
    cube[3] *= 15.  # sig0: uniform prior 0:15
    cube[4] = cube[4] * 5. + 66.  # inc: uniform prior 66:71 (low pri=49.83663935008522 from MGE q)
    cube[5] = cube[5] * 4. + 17.  # PAdisk: uniform prior 17:21
    cube[6] = cube[6] * 50. + 6440.  # vsys: uniform prior 6440:6490
    cube[7] = cube[7] * 0.45 + 1.45  # mlratio: uniform prior 1.45:1.90
    cube[8] = cube[8] * 0.85 + 0.85  # f: uniform prior 0.85:1.7

# 2698 dyndyn_newpri priors (GOOD OUTPUT)
    cube[0] = 10 ** (cube[0] * 2. + 8.)  # mbh: log-uniform prior 10^8:10^10
    cube[1] = cube[1] * 6. + 124.  # xloc: uniform prior 124:130
    cube[2] = cube[2] * 4. + 149.  # yloc: uniform prior 149:153
    cube[3] *= 15.  # sig0: uniform prior 0:15
    cube[4] = cube[4] * 5. + 66.  # inc: uniform prior 66:71 (low pri=49.83663935008522 from MGE q)
    cube[5] = cube[5] * 4. + 17.  # PAdisk: uniform prior 17:21
    cube[6] = cube[6] * 50. + 6440.  # vsys: uniform prior 6440:6490
    cube[7] = cube[7] + 1.  # mlratio: uniform prior 1:2
    cube[8] = cube[8] * 0.85 + 0.85  # f: uniform prior 0.85:1.7

# 2698 dyndyn_newpri 2
    cube[0] = 10 ** (cube[0] * 2. + 8.)  # mbh: log-uniform prior 10^8:10^10
    cube[1] = cube[1] * 6. + 124.  # xloc: uniform prior 124:130
    cube[2] = cube[2] * 4. + 149.  # yloc: uniform prior 149:153
    cube[3] *= 15.  # sig0: uniform prior 0:15
    cube[4] = cube[4] * 20. + 60.  # inc: uniform prior 66:71 (low pri=49.83663935008522 from MGE q)
    cube[5] = cube[5] * 35. + 10.  # PAdisk: uniform prior 17:21
    cube[6] = cube[6] * 100. + 6400.  # vsys: uniform prior 6440:6490
    cube[7] = cube[7] + 1.  # mlratio: uniform prior 1:2
    cube[8] = cube[8] * 1.5 + 0.5  # f: uniform prior 0.85:1.7
'''

'''
# 2698 105newpri
    cube[0] = 10 ** (cube[0] * 2. + 8.)  # mbh: log-uniform prior 10^8:10^10
    cube[1] = cube[1] * 6. + 124.  # xloc: uniform prior 123:131
    cube[2] = cube[2] * 6. + 148.  # yloc: uniform prior 147:155
    cube[3] *= 15.  # sig0: uniform prior 0:15
    cube[4] = cube[4] * 39. + 50.  # inc: uniform prior 50:89 (low pri=49.83663935008522 from MGE q)
    cube[5] = cube[5] * 45.  # PAdisk: uniform prior 0:45
    cube[6] = cube[6] * 100. + 6400.  # vsys: uniform prior 6400:6500
    cube[7] = cube[7] * 4.9 + 0.1  # mlratio: uniform prior 0.1:5
    cube[8] = cube[8] * 1.5 + 0.5  # f: uniform prior 0.5:2
    
# 2698 105newpri 2
    cube[0] = 10 ** (cube[0] * 2. + 8.)  # mbh: log-uniform prior 10^8:10^10
    cube[1] = cube[1] * 4. + 125.  # xloc: uniform prior 123:131
    cube[2] = cube[2] * 4. + 149.  # yloc: uniform prior 147:155
    cube[3] *= 15.  # sig0: uniform prior 0:15
    cube[4] = cube[4] * 20. + 60.  # inc: uniform prior 50:89 (low pri=49.83663935008522 from MGE q)
    cube[5] = cube[5] * 20. + 10.  # PAdisk: uniform prior 0:45
    cube[6] = cube[6] * 100. + 6400.  # vsys: uniform prior 6400:6500
    cube[7] = cube[7] * 1.9 + 0.1  # mlratio: uniform prior 0.1:2
    cube[8] = cube[8] * 1.5 + 0.5  # f: uniform prior 0.5:2
'''