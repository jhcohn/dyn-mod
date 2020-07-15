# Python 3 compatability
# from __future__ import division, print_function
# from six.moves import range

import dynesty
import pickle
from dynesty import utils as dyfunc
import dynamical_model as dm

from pathlib import Path

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

result = []
with open('groupmtg/kinemetry_out_vorbin_avgdat_snr10.txt', 'r') as f:
    for line in f:
        if not line.startswith('#'):
            cols = line.split()
            for c in cols:
                result.append(float(c))

k0, k1, k5, k51, rad, pa, q, er_q, er_pa, erk1, erk5, erk51 = [], [], [], [], [], [], [], [], [], [], [], []

interval = int(len(result) / 12.)
for j in range(len(result)):
    if j < interval:
        k0.append(result[j])
    elif interval <= j < 2 * interval:
        k1.append(result[j])
    elif 2 * interval <= j < 3 * interval:
        k5.append(result[j])
    elif 3 * interval <= j < 4 * interval:
        k51.append(result[j])
    elif 4 * interval <= j < 5 * interval:
        rad.append(result[j])
    elif 5 * interval <= j < 6 * interval:
        pa.append(result[j])
    elif 6 * interval <= j < 7 * interval:
        q.append(result[j])
    elif 7 * interval <= j < 8 * interval:
        er_q.append(result[j])
    elif 8 * interval <= j < 9 * interval:
        er_pa.append(result[j])
    elif 9 * interval <= j < 10 * interval:
        erk1.append(result[j])
    elif 10 * interval <= j < 11 * interval:
        erk5.append(result[j])
    elif 11 * interval <= j < 12 * interval:
        erk51.append(result[j])

plt.errorbar(rad, pa, yerr=er_pa, fmt='ko')
plt.xlabel('Radius [arcsec]')
plt.ylabel('Position Angle [deg]')
plt.show()

plt.errorbar(rad, q, yerr=er_q, fmt='bo')
plt.xlabel('Radius [arcsec]')
plt.ylabel('q')
plt.show()

plt.errorbar(rad, k1, yerr=erk1, fmt='ko')
plt.xlabel('Radius [arcsec]')
plt.ylabel(r'k$_1$ [km/s]')
plt.show()

plt.errorbar(rad, k51, yerr=erk51, fmt='bo')
plt.xlabel('Radius [arcsec]')
plt.ylabel(r'k$_5$/k$_1$')
plt.show()

