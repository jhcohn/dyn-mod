from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import AxesGrid
import os
from scipy import interpolate

base = '/Users/jonathancohn/Documents/dyn_mod/'
gf = 'galfit_u2698/'

re_skysub = base + 'ugc2698_f160w_pxfr075_pxs010_drz_rapidnuc_sci_nonan_e.fits'
mod1 = base + gf + 'galfit_out_u2698_rrenewsky_sersic.fits'
mod2 = base + gf + 'galfit_out_u2698_rrenewsky_2sersic.fits'
mod3 = base + gf + 'galfit_out_u2698_rrenewsky_2sersic_centered.fits'
mod4 = base + gf + 'galfit_out_u2698_rrenewsky_3sersic_centered.fits'
mod5 = base + gf + 'galfit_out_u2698_rrenewsky_4sersic_centered.fits'
# input: galfit_sersic_rre.txt, output galfit.123

mask = base + gf + 'f160w_maskedgemask_px010.fits'

with fits.open(mod1) as hdu:
    print(hdu.info())
    # print(hdu_h[0].header)
    hdr = hdu[0].header
    dat = hdu[0].data
    data = hdu[1].data  # input image
    model = hdu[2].data  # model
    residual = hdu[3].data  # residual
    resid = (data - model) / data

with fits.open(mod2) as hdu:
    print(hdu.info())
    # print(hdu_h[0].header)
    hdr2 = hdu[0].header
    dat2 = hdu[0].data
    data2 = hdu[1].data  # input image
    model2 = hdu[2].data  # model
    residual2 = hdu[3].data  # residual
    resid2 = (data2 - model2) / data2

with fits.open(mod3) as hdu:
    print(hdu.info())
    # print(hdu_h[0].header)
    hdr3 = hdu[0].header
    dat3 = hdu[0].data
    data3 = hdu[1].data  # input image
    model3 = hdu[2].data  # model
    residual3 = hdu[3].data  # residual
    resid3 = (data3 - model3) / data3

with fits.open(mod4) as hdu:
    print(hdu.info())
    # print(hdu_h[0].header)
    hdr4 = hdu[0].header
    dat4 = hdu[0].data
    data4 = hdu[1].data  # input image
    model4 = hdu[2].data  # model
    residual4 = hdu[3].data  # residual
    resid4 = (data4 - model4) / data4

with fits.open(mod5) as hdu:
    print(hdu.info())
    # print(hdu_h[0].header)
    hdr5 = hdu[0].header
    dat5 = hdu[0].data
    data5 = hdu[1].data  # input image
    model5 = hdu[2].data  # model
    residual5 = hdu[3].data  # residual
    resid5 = (data5 - model5) / data5

with fits.open(mask) as hdu:
    maskdata = hdu[0].data

#data -= maskdata * data
#model -= maskdata * model

#maskdata[maskdata == 0] = -1
#maskdata[maskdata == 1] = 0
#maskdata[maskdata == -1] = 1
#data *= maskdata
#data[data == 0.] = 337.5

print(np.amax(data), np.amin(data))
print(np.amax(model), np.amin(model))
print(np.amax(residual), np.amin(residual))
# resid = abs(data - model) / data
# resid[maskdata == 1] = 0.
print(np.nanmax(resid[resid != np.inf]), np.nanmin(resid[resid != -np.inf]))

#plt.imshow((model - model2)/model, origin='lower')#, vmax=0.1, vmin=-0.1)
#plt.colorbar()
#plt.show()

from scipy.special import gamma, gammainc
from scipy.stats.distributions import chi2
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2.html  # chi2.ppf is inverse cdf of chi^2 yay!

def lrt(chi2mod1, chi2mod2):

    return chi2mod1 - chi2mod2


def bic(npt, npar, loglike):  # npt = number of data points, npar = number of free params, loglike = ln(max likelihood)
    # like ~ exp(-chi^2/2) (maximizes likelihood for min chi^2) -> loglike ~ -chi^2/2

    return npar*np.log(npt) - 2*loglike


def aic(npar, loglike):  # npar = number of free params, loglike = ln(max likelihood)
    # like ~ exp(-chi^2/2) (maximizes likelihood for min chi^2) -> loglike ~ -chi^2/2

    return 2*npar - 2*loglike


#fig, ax = plt.subplots(3,3, figsize=(16,12))
fig, ax = plt.subplots(4,3, figsize=(16,16))
im0 = ax[0][0].imshow(data, origin='lower', vmax=1e4, vmin=0)
ax[0][0].set_ylabel(r'1-sersic')
ax[0][0].set_title(r'Data')
cbar = fig.colorbar(im0, ax=ax[0][0], pad=0.02)
im1 = ax[0][1].imshow(model, origin='lower', vmax=1e4, vmin=0)
ax[0][1].set_title(r'Model')
cbar1 = fig.colorbar(im1, ax=ax[0][1], pad=0.02)
# im2 = ax[2].imshow(resid, origin='lower', vmax=1., vmin=0.)
# ax[2].set_title(r'|data - model| / data')
im2 = ax[0][2].imshow(resid, origin='lower', vmax=0.2, vmin=-0.2, cmap='RdBu')
ax[0][2].set_title(r'(data - model) / data')
cbar2 = fig.colorbar(im2, ax=ax[0][2], pad=0.02)

'''  #
im10 = ax[1][0].imshow(data2, origin='lower', vmax=1e4, vmin=0)
ax[1][0].set_title(r'(2-sersic)')
cbar10 = fig.colorbar(im10, ax=ax[1][0], pad=0.02)
im11 = ax[1][1].imshow(model2, origin='lower', vmax=1e4, vmin=0)
ax[1][1].set_title(r'(2-sersic)')
cbar11 = fig.colorbar(im11, ax=ax[1][1], pad=0.02)
# im2 = ax[2].imshow(resid, origin='lower', vmax=1., vmin=0.)
# ax[2].set_title(r'|data - model| / data')
im12 = ax[1][2].imshow(resid2, origin='lower', vmax=0.2, vmin=-0.2, cmap='RdBu')
ax[1][2].set_title(r'(2-sersic)')
cbar12 = fig.colorbar(im12, ax=ax[1][2], pad=0.02)
# '''

im20 = ax[1][0].imshow(data3, origin='lower', vmax=1e4, vmin=0)
ax[1][0].set_ylabel(r'2-sersic')
cbar20 = fig.colorbar(im20, ax=ax[1][0], pad=0.02)
im21 = ax[1][1].imshow(model3, origin='lower', vmax=1e4, vmin=0)
#ax[1][1].set_title(r'(2-sersic, constrained)')
cbar21 = fig.colorbar(im21, ax=ax[1][1], pad=0.02)
# im2 = ax[2].imshow(resid, origin='lower', vmax=1., vmin=0.)
# ax[2].set_title(r'|data - model| / data')
im22 = ax[1][2].imshow(resid3, origin='lower', vmax=0.2, vmin=-0.2, cmap='RdBu')
#ax[1][2].set_title(r'(2-sersic, constrained)')
cbar22 = fig.colorbar(im22, ax=ax[1][2], pad=0.02)

im30 = ax[2][0].imshow(data4, origin='lower', vmax=1e4, vmin=0)
ax[2][0].set_ylabel(r'3-sersic')
cbar30 = fig.colorbar(im30, ax=ax[2][0], pad=0.02)
im31 = ax[2][1].imshow(model4, origin='lower', vmax=1e4, vmin=0)
#ax[2][1].set_title(r'(3-sersic, constrained)')
cbar31 = fig.colorbar(im31, ax=ax[2][1], pad=0.02)
# im2 = ax[2].imshow(resid, origin='lower', vmax=1., vmin=0.)
# ax[2].set_title(r'|data - model| / data')
im32 = ax[2][2].imshow(resid4, origin='lower', vmax=0.2, vmin=-0.2, cmap='RdBu')
#ax[2][2].set_title(r'(3-sersic, constrained)')
cbar32 = fig.colorbar(im32, ax=ax[2][2], pad=0.02)

im40 = ax[3][0].imshow(data5, origin='lower', vmax=1e4, vmin=0)
ax[3][0].set_ylabel(r'4-sersic')
cbar40 = fig.colorbar(im40, ax=ax[3][0], pad=0.02)
im41 = ax[3][1].imshow(model5, origin='lower', vmax=1e4, vmin=0)
#ax[3][1].set_title(r'(4-sersic, constrained)')
cbar41 = fig.colorbar(im41, ax=ax[3][1], pad=0.02)
# im2 = ax[2].imshow(resid, origin='lower', vmax=1., vmin=0.)
# ax[2].set_title(r'|data - model| / data')
im42 = ax[3][2].imshow(resid5, origin='lower', vmax=0.2, vmin=-0.2, cmap='RdBu')
#ax[3][2].set_title(r'(4-sersic, constrained)')
cbar42 = fig.colorbar(im42, ax=ax[3][2], pad=0.02)

plt.show()

# chi^2_1sersic = 2569389.000, Ndof_s1 = 1464996; chi^2_s2 = 1942792.250, Ndof_s2 = 1464989;
# chi^2_s2cons =  1130432.000, Ndof_s2cons = 1464992
# chi^2_s3 = 758686.500; Ndof_s3 = 1464988; chi^2_s4 = 746227.875, Ndof_s4 = 1464984
# [Ndof = N_pixels - N_freeparams]
# free params s1 = 7; free params s2 = 14; free params s2cons = 11; free pars s3c = 15
# Taking likelihood_ratio_test_statistic lambda_LR = -2(l(mod0) - l(mod1)):
# and taking l(mod) = nl(exp(-chi^2/2)) = -chi^2/2'
# lambda_LR = (chi^2_mod0 - chi^2_mod1)
# lambda_LR(1sersic, 2sersic) = 2569389 - 1942792.25 = 626596.75, Ndof = N_s2 - N1s1 =
# lambda_LR(1sersic, 2sersic_cons) = 2569389 - 1130432 = 1438957

# print(lrt(2569389. / 1464996, 1942792. / 1464989), 'unconstrained')
#print(lrt(2569389., 1942792.), 'unconstrained')
#print(chi2.ppf(.9999994, 7), 'unc diff')
# print(lrt(2569389. / 1464996, 1130432. / 1464992), 'constrained')
print(lrt(758686.5, 746227.875), '4sc vs 3sc')
print(lrt(1130432., 758686.5), '3sc vs 2sc')
print(lrt(2569389., 1130432.), '2sc vs 1s')
print(chi2.ppf(.9999994, 4), 'cons diff')
#print(lrt(1130432. / 1464992, 1942792. / 1464989), 'constrained vs unc')
#print(aic(7, -2569389 / 2.), 's1')
#print(aic(14, -1942792.25 / 2.), 's2 uncons')
#print(aic(11, -1130432 / 2.), 's2 cons')
#print(bic(1464996, 7, -2569389/2.), 's1')
#print(bic(1464989, 14, -1942792.25/2.), 's2 uncons')
#print(bic(1464992, 11, -1130432/2.), 's2 cons')
# print(lrt(1130432., 1690149.875))  # Cons vs MGE, delta Ndof = 1464992-1464967 = 25

# See HERE: for 99.7% confidence interval, would want alpha = 0.003 https://www.itl.nist.gov/div898/handbook/apr/section2/apr233.htm
# See HERE: https://www.itl.nist.gov/div898/handbook/eda/section3/eda3674.htm (Lower-tail critical values) If LRT < crit
# value for given alpha and delta-(degrees of freedom), null hypothesis ruled out!
# SO: LRT: rules out s1 in favor of s2un (nu=7, LRT=0.4277) at better than 0.001. Rules out s2un in favor of s2c (nu=3,
#     LRT=0.5545) at between 0.10 and 0.05. Rules out s1 in favor of s2c (nu=4, LRT=0.9822) at better than
print(oop)

#sersic_galfit_fit_