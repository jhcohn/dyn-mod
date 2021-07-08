from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate, optimize, ndimage
from photutils.centroids import centroid_com, centroid_2dg  # centroid_quadratic,
import time
from astropy import convolution


def gaussian(height, center_x, center_y, width_x, width_y):
    """Returns a gaussian function with the given parameters"""
    return lambda x,y: height*np.exp(-(((center_x-x)/width_x)**2+((center_y-y)/width_y)**2)/2)

def moments(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution by calculating its
    moments """
    total = data.sum()
    X, Y = np.indices(data.shape)
    x = (X*data).sum()/total
    y = (Y*data).sum()/total
    col = data[:, int(y)]
    width_x = np.sqrt(np.abs((np.arange(col.size)-x)**2*col).sum()/col.sum())
    row = data[int(x), :]
    width_y = np.sqrt(np.abs((np.arange(row.size)-y)**2*row).sum()/row.sum())
    height = data.max()
    return height, x, y, width_x, width_y

def fitgaussian(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution found by a fit"""
    params = moments(data)
    errorfunction = lambda p: np.ravel(gaussian(*p)(*np.indices(data.shape)) -
                                 data)
    p, success = optimize.leastsq(errorfunction, params)
    return p


def fitconvolve(sigma, data1, data2):
    """

    :return:
    """

    if sigma[0] <= 1e-15 or np.isinf(sigma[0]) or sigma[1] <= 1e-15 or np.isinf(sigma[1]):
        zero = np.inf

    else:
        # convolve I-band PSF (data1) with kernel with unknown FWHM (sigma[0], sigma[1]) to match H-band PSF (data2)
        zero = np.nansum(ndimage.gaussian_filter(data1, [sigma[0], sigma[1]]) - data2)

    return [zero, 0.]

def fitsigma(sigma, data1, data2):

    return np.nansum(ndimage.gaussian_filter(data1, [sigma[0], sigma[1]]) - data2)

def vfit(sigma, *kwargs):

    return [fitsigma(sigma, *kwargs), 0.]

def twoD_gaussian(x, height, center_x, center_y, width_x, width_y):
    """Returns a gaussian function with the given parameters"""
    return height*np.exp(-(((center_x-x[0])/width_x)**2+((center_y-x[1])/width_y)**2)/2)

def do_2dgaussfit(data, plots=False):
    data = np.nan_to_num(data)
    params = fitgaussian(data)
    fit = gaussian(*params)
    (height, x, y, width_x, width_y) = params
    print(params)

    if plots:
        plt.contour(fit(*np.indices(data.shape)))
        ax = plt.gca()

        plt.text(0.95, 0.05, """x : %.1f\ny : %.1f\nwidth_x : %.1f\nwidth_y : %.1f""" % (x, y, width_x, width_y),
                 fontsize=16, horizontalalignment='right', verticalalignment='bottom', transform=ax.transAxes)
        plt.show()
    return params


g2698 = '/Users/jonathancohn/Documents/dyn_mod/galfit/u2698/'
gf = '/Users/jonathancohn/Documents/dyn_mod/galfit/'
fj = '/Users/jonathancohn/Documents/dyn_mod/for_jonathan/'
hst_p = '/Users/jonathancohn/Documents/hst_pgc11179/'
hst_n = '/Users/jonathancohn/Documents/hst_ngc384/'
n384_h = fj + 'NGC0384_F160W_drz_sci.fits'
n384_adj_h = fj + 'NGC0384_F160W_drz_sci_adjusted.fits'
n384_hmask = fj + 'NGC0384_F160W_drz_mask.fits'
n384_hmask_extended = fj + 'NGC0384_F160W_drz_mask_extended.fits'
n384_f814w_pixscale03 = fj + 'NGC0384_F814W_drc_sci.fits'
psfH = fj + 'psfH.fits'

p11179_h = fj + 'PGC11179_F160W_drz_sci.fits'
p11179_adj_h = fj + 'PGC11179_F160W_drz_sci_adjusted.fits'

# DRIZZLED F814W IMAGES
n384_f814w_pxf08 = hst_n + 'n384_f814w_drizflc_pxf08_006_sci.fits'
p11179_f814w_pxf08 = hst_p + 'p11179_f814w_drizflc_006_sci.fits'  # pxf08!

# FINAL PSFs
p11179_driz_f814w_psf = hst_p + 'p11179_f814w_drizflc_pxf08_006_psf_take2_sci.fits'
p11179_driz_f814w_psfcen = hst_p + 'p11179_f814w_drizflc_pxf08_006_psf_centroid_sci.fits'
p11179_driz_f814w_psfben = hst_p + 'p11179_f814w_drizflc_pxf08_006_psf_benstyle_sci.fits'
n384_driz_f814w_psf = hst_n + 'n384_f814w_drizflc_pxf08_006_psf_take2_sci.fits'
n384_driz_f814w_psfcen = hst_n + 'n384_f814w_drizflc_pxf08_006_psf_centroid_sci.fits'
n384_driz_f814w_psfben = hst_n + 'n384_f814w_drizflc_pxf08_006_psf_benstyle_sci.fits'

zp_H = 24.662  # https://www.stsci.edu/hst/instrumentation/wfc3/data-analysis/photometric-calibration/ir-photometric-calibration
zp_I = 24.699  # ACTUALLY UVIS1!! (UVIS2=24.684)
# https://www.stsci.edu/hst/instrumentation/wfc3/data-analysis/photometric-calibration/uvis-photometric-calibration

with fits.open(p11179_driz_f814w_psf) as hdu:
    hdr_pgc_psf_f814w = hdu[0].header
    dat_pgc_psf_f814w = hdu[0].data

with fits.open(p11179_driz_f814w_psfcen) as hdu:
    hdr_pgc_psfcen_f814w = hdu[0].header
    dat_pgc_psfcen_f814w = hdu[0].data

with fits.open(p11179_driz_f814w_psfben) as hdu:
    hdr_pgc_psfben_f814w = hdu[0].header
    dat_pgc_psfben_f814w = hdu[0].data

with fits.open(n384_driz_f814w_psf) as hdu:
    hdr_ngc_psf_f814w = hdu[0].header
    dat_ngc_psf_f814w = hdu[0].data

with fits.open(n384_driz_f814w_psfcen) as hdu:
    hdr_ngc_psfcen_f814w = hdu[0].header
    dat_ngc_psfcen_f814w = hdu[0].data

with fits.open(n384_driz_f814w_psfben) as hdu:
    hdr_ngc_psfben_f814w = hdu[0].header
    dat_ngc_psfben_f814w = hdu[0].data

with fits.open(psfH) as hdu:
    hdr_psfH = hdu[0].header
    dat_psfH = hdu[0].data

with fits.open(n384_f814w_pxf08) as hdu:
    hdr_nI = hdu[0].header
    dat_nI = hdu[0].data

with fits.open(p11179_f814w_pxf08) as hdu:
    hdr_pI = hdu[0].header
    dat_pI = hdu[0].data

res = 0.06  # arcsec/pix

# METHOD 2
t0 = time.time()
print('start')
def twoD_gaussian(x, height, center_x, center_y, width_x, width_y):
    """Returns a gaussian function with the given parameters"""
    return height*np.exp(-(((center_x-x[0])/width_x)**2+((center_y-x[1])/width_y)**2)/2)

def conv(pars, dataI):
    height, center_x, center_y, width_x, width_y = pars
    #height, width_x, width_y = pars
    #center_x, center_y = 40., 40.

    x = np.linspace(0, len(dataI), len(dataI))
    y = np.linspace(0, len(dataI[0]), len(dataI[0]))
    x, y = np.meshgrid(x, y)

    twoDG = np.exp(-(((center_x - x) / width_x) ** 2 + ((center_y - y) / width_y) ** 2) / 2)

    return convolution.convolve(dataI * height, twoDG)

def zero_conv(pars, dataH, dataI):

    icon = conv(pars, dataI)

    # return np.abs(np.nansum(icon - dataH))
    return np.abs(np.nansum((icon - dataH)**2))
    # return np.exp(-np.nansum(icon - dataH))


# psfH peak is centered on 40,40; current peak is at NGC 1927,1922 -> 1882:1882+81, 1887:1887+81
# dat_psfi = dat_ngc_psfcen_f814w[1882:1882+81, 1887:1887+81]  # correct!
dat_psfi = dat_pgc_psfcen_f814w[1606:1606+81, 1542:1542+81]
# psfH peak is centered on 40,40; current peak is at PGC 1646.4766,1582.24407 -> 1606:1606+81, 1542:1542+81
#g2di = do_2dgaussfit(dat_psfi)  # [1.04957099e-03 1.46476646e+02 8.22440674e+01 8.65402657e-01 8.51768218e-01] # p11179
#plt.imshow(dat_psfi, origin='lower')
#plt.colorbar()
#plt.show()
#print(oops)
g2di = do_2dgaussfit(dat_psfi)  # [1.10704459e-03 4.01447195e+01 4.01799546e+01 8.18216509e-01 8.47028033e-01] # n384
g2dh = do_2dgaussfit(dat_psfH)  # [ 0.04847133 40.03064685 39.91339118  1.45843277  1.49161195]  # PSF H

# dat_psfi *= g2dh[0] / g2di[0]


diff_g2d = np.sqrt(g2dh**2 - g2di**2)
print(diff_g2d)
guess_g2d_offset = (160., 40., 40., 1.2, 1.2)
#guess_g2d_offset = (160., 1.2, 1.2)
# guess_g2d_best = (159., 40., 40., diff_g2d[3], diff_g2d[4])
#z = zero_conv(pars=guess_g2d_best, dataH=dat_psfH, dataI=dat_psfi)
#print(z, 'check')
#z = zero_conv(pars=guess_g2d_offset, dataH=dat_psfH, dataI=dat_psfi)
#print(z, 'check')
sol = optimize.fmin(zero_conv, x0=guess_g2d_offset, args=(dat_psfH, dat_psfi))
print(sol, 'fmin solution')
print(diff_g2d)
# NGC 384:
# [120.0546269   40.21214241  40.35472934   1.09384867   1.06637347] fmin solution (including centerx,y) 0.0004500262
# [121.55822138   1.13873817   1.15121924] fmin solution (NOT including centerx,y) 0.0010782303
# PGC 11179:
# [121.84288026  40.1249686   40.02693034   1.09615983   1.05934121] fmin solution (including centerx,y) 0.00045021565
# [121.82100697   1.10433544   1.05780877] fmin solution (NOT including centerx,y) 0.0005105181

# MAKE GALAXY PLOTS
sol[0] = 1.  # comment out for withamp; else, noamp!
sol[1] = sol[1]+60.  # 200 vs 80 -> midpoint 100 instead of 40, so add 60!
sol[2] = sol[2]+60.  # 200 vs 80 -> midpoint 100 instead of 40, so add 60!
# icon = conv(sol, dataI=dat_nI[1820:2021, 1830:2031])  # NGC 384
icon = conv(sol, dataI=dat_pI[1550:1751, 1500:1701])  # PGC 11179

fig, ax = plt.subplots(1, 2, figsize=(12,5))
plt.subplots_adjust(wspace=0.05)
#im0 = ax[0].imshow(dat_nI[1820:2021, 1830:2031], origin='lower', vmax=150.)  # NGC 384
im0 = ax[0].imshow(dat_pI[1550:1751, 1500:1701], origin='lower', vmax=90.)  # PGC 11179
cb0 = fig.colorbar(im0, ax=ax[0], pad=0.02)
ax[0].text(5, 170, 'I-band (original)', color='w')
# im0 = ax[1].imshow(icon, origin='lower', vmax=150.)  # NGC 384 (noamp)
# im0 = ax[1].imshow(icon, origin='lower', vmax=12000.)  # NGC 384 (withamp)
im0 = ax[1].imshow(icon, origin='lower', vmax=90.)  # PGC 11179 (noamp)
# im0 = ax[1].imshow(icon, origin='lower', vmax=6800.)  # PGC 11179 (withamp)
cb0 = fig.colorbar(im0, ax=ax[1], pad=0.02)
ax[1].text(5, 170, 'I-band (convolved)', color='w')
plt.show()
print(oops)
# n384_psfhomog_2dgauss_5pars_iband_noamp.png  # p11179_psfhomog_2dgauss_5pars_iband_noamp
#

# MAKE PSF PLOTS
icon = conv(sol, dataI=dat_psfi)
print(zero_conv(pars=sol, dataH=dat_psfH, dataI=dat_psfi), 'test the solution!')

fig, ax = plt.subplots(2, 2, figsize=(8,8))
fig.subplots_adjust(wspace=0.1, hspace=0.1)
im0 = ax[0][0].imshow(dat_psfi, origin='lower')
ax[0][0].text(5, 70, 'I-band PSF', color='w')
cbar0 = fig.colorbar(im0, ax=ax[0][0], pad=0.02)
im01 = ax[0][1].imshow(dat_psfH, origin='lower')
ax[0][1].text(5, 70, 'H-band PSF', color='w')
cbar01 = fig.colorbar(im01, ax=ax[0][1], pad=0.02)
im1 = ax[1][0].imshow(icon, origin='lower')
ax[1][0].text(5, 70, 'convolved I-band', color='w')
cbar1 = fig.colorbar(im1, ax=ax[1][0], pad=0.02)
im2 = ax[1][1].imshow((dat_psfH - icon), origin='lower')
ax[1][1].text(5, 70, r'H-band $-$ convolved I-band', color='w')
cbar2 = fig.colorbar(im2, ax=ax[1][1], pad=0.02)
plt.show()
print(oop)
#xx = optimize.fsolve(vfit, args=(dat_psfi, dat_psfH), x0=[143.4, 2])
#print(xx, vfit(xx, dat_psfi, dat_psfH))

# print(oop)
print(np.nanmax(dat_psfi), np.nanmax(dat_psfH))  # 0.001129155 0.05456335
ssp_n = ndimage.gaussian_filter(dat_psfi, [1.2, 1.2]) # convolve with kernel!
plt.imshow(ssp_n - dat_psfi, origin='lower')
plt.colorbar()
plt.show()
print(np.nansum(dat_psfi - dat_psfH))  # -0.980911
print(np.nansum(ssp_n - dat_psfi))  # 1.1641532e-10
print(np.nansum(ssp_n - dat_psfH))  # -0.9809109
print(oops)


print(oops)

solveit = optimize.fsolve(fitconvolve, args=(dat_psfi, dat_psfH), x0=[1.2, 1.2])
print(solveit, [diff_sigma_nx, diff_sigma_ny], 'solution, initial guess')
print(time.time()-t0, 'time!')
print(fitconvolve(solveit, dat_psfi, dat_psfH), 'should be 0 -- did solution work?')
print(oops)


# METHOD 1
# calculate sigmas from 2D gaussians
gp = do_2dgaussfit(dat_pgc_psfcen_f814w[1500:1700, 1500:1700])  # [1.04957099e-03 1.64647665e+03 1.58224407e+03 8.65402658e-01 8.51768218e-01]
gn = do_2dgaussfit(dat_ngc_psfcen_f814w[1820:2020, 1800:2020])  # [1.10704459e-03 1.92214472e+03 1.92717995e+03 8.18216509e-01 8.47028032e-01]
gH = do_2dgaussfit(dat_psfH)  # [ 0.04847133 40.03064685 39.91339118  1.45843277  1.49161195]

#p_fwhm = np.mean([p_fwhm_x, p_fwhm_y])
#n_fwhm = np.mean([n_fwhm_x, n_fwhm_y])
#H_fwhm = np.mean([H_fwhm_x, H_fwhm_y])

diff_sigma_px = np.sqrt(gH[3]**2 - gp[3]**2) # diff in quadrature!  # 1.2072894820619162
diff_sigma_py = np.sqrt(gH[4]**2 - gp[4]**2) # diff in quadrature!  # 1.2277824434215172
diff_sigma_nx = np.sqrt(gH[3]**2 - gn[3]**2) # diff in quadrature!  # 1.1739269102067171
diff_sigma_ny = np.sqrt(gH[4]**2 - gn[4]**2) # diff in quadrature!  # 1.2244987215858145
print(diff_sigma_nx)
print(diff_sigma_ny)
print(diff_sigma_px)
print(diff_sigma_py)

tn = time.time()
ssp_n = ndimage.gaussian_filter(dat_ngc_psfcen_f814w, [diff_sigma_nx, diff_sigma_ny]) # convolve with kernel!
gal_n = ndimage.gaussian_filter(dat_nI, [diff_sigma_nx, diff_sigma_ny]) # convolve with kernel!
t1 = time.time()
print('time in gaussian filter', t1 - tn)
ssp_p = ndimage.gaussian_filter(dat_pgc_psfcen_f814w, [diff_sigma_px, diff_sigma_py]) # convolve with kernel!
gal_p = ndimage.gaussian_filter(dat_pI, [diff_sigma_px, diff_sigma_py]) # convolve with kernel!
print('time in gaussian filter', time.time() - t1)
gn2 = do_2dgaussfit(ssp_n[1820:2020, 1800:2020])  # [1.04957099e-03 1.64647665e+03 1.58224407e+03 8.65402658e-01 8.51768218e-01]
gp2 = do_2dgaussfit(ssp_p[1500:1700, 1500:1700])  # [1.04957099e-03 1.64647665e+03 1.58224407e+03 8.65402658e-01 8.51768218e-01]
gn = do_2dgaussfit(dat_ngc_psfcen_f814w[1820:2020, 1800:2020])  # [1.04957099e-03 1.64647665e+03 1.58224407e+03 8.65402658e-01 8.51768218e-01]
gp = do_2dgaussfit(dat_pgc_psfcen_f814w[1500:1700, 1500:1700])
# print(gn2)
# print(gp2)
# print(gn)
# print(gp)
print(gH)
plt.imshow(gal_n[1820:2020, 1830:2030], origin='lower', vmax=150.)
plt.colorbar()
plt.show()
plt.imshow(dat_nI[1820:2020, 1830:2030], origin='lower', vmax=150.)
plt.colorbar()
plt.show()
plt.imshow(gal_p[1550:1750, 1500:1700], origin='lower', vmax=90.)
plt.colorbar()
plt.show()
plt.imshow(dat_pI[1550:1750, 1500:1700], origin='lower', vmax=90.)
plt.colorbar()
plt.show()
print(oops)


'''  # NGC 384 FIND CENTERS
plt.imshow(dat_n5, origin='lower', vmax=6700, vmin=0.)
plt.colorbar()
plt.plot(1994, 154, 'm*')  # max
plt.plot(1994.5, 153.8, 'w*')  # 2D gauss
plt.plot(1994.4, 153.2, 'c*')  # centroid
plt.show()
plt.imshow(dat_nl, origin='lower', vmax=14000, vmin=0.)
plt.colorbar()
plt.plot(2039, 199, 'm*')
plt.plot(2039.9, 198.9, 'w*')
plt.plot(2039.5, 198.4, 'c*')
plt.show()
plt.imshow(dat_no, origin='lower', vmax=9900, vmin=0.)
plt.colorbar()
plt.plot(2084, 244, 'm*')
plt.plot(2110.6, 253.9, 'w*')
plt.plot(2084.8, 243.8, 'c*')
plt.show()
print(oop)

find_center(dat_n5, 1950, 100, 7000.)
# maxloc: [54, 44] # this is [y x]! -> [1994 154] n5
# 2D gauss: 4.44897434e+01 5.37789283e+01 [1994.5, 153.8] n5
# cen: 44.041795233578284 53.24022886329969 [1994.4, 153.2] n5

find_center(dat_nl, 2000, 150, 14000.)
# maxloc: [49, 39] -> [2039 199] nl
# 2D gauss: 3.98992626e+01  4.89351953e+01 [2039.9, 198.9] nl
# cen: 39.50042823306298 48.44560475888629 [2039.5, 198.4] nl
#print(oop)

find_center(dat_no, 2050, 200, 9900.)
# maxloc [44, 34] no -> [2084 244]
# 2D gauss: 6.05688644e+01  5.38515316e+01 [2110.6, 253.9] no
# cen: 34.777073215971605 43.77436076055096 [2084.8, 243.8] no
print(oop)
# '''  #

# FROM HERE NGC 384 I-BAND PSFs
# 2005.8, 199.7 (for rmq?) atv aperture photometry!
psf_5 = np.zeros(shape=dat_n5.shape)  # rmq -> pattstep = 0, no pattern, no dither. Just center it over galaxy!
psf_l = np.zeros(shape=dat_nl.shape)  # rxq -> pattstep = 1, pattern1=LINE (0,0) [must be 0,0, since step=2 exists]
psf_o = np.zeros(shape=dat_no.shape)  # rzq -> pattstep = 2, pattern1=LINE (2.5, 2.5)


# BEGIN JONELLE STYLE
# BASED ON MAX PIXELS
nxctr_5,nyctr_5 = 1994,154
nxctr_l,nyctr_l = 2039,199
nxctr_o,nyctr_o = 2084,244

# dat_psfm, *psfx, *psfz all centered at 44,44
# PUT 44,44 at galaxy center!!
psf_5[nyctr_5-44:nyctr_5+45, nxctr_5-44:nxctr_5+45] = dat_psf5  # correct!!!!!!
psf_l[nyctr_l-44:nyctr_l+45, nxctr_l-44:nxctr_l+45] = dat_psfl  # correct!!!!!!
psf_o[nyctr_o-44:nyctr_o+45, nxctr_o-44:nxctr_o+45] = dat_psfo  # correct!!!!!!


# BEGIN HYBRID & BEN STYLE!!!!
# cen: 44.041795233578284 53.24022886329969 [1994.4, 153.2] n5
# cen: 39.50042823306298 48.44560475888629 [2039.5, 198.4] nl
# cen: 34.777073215971605 43.77436076055096 [2084.8, 243.8] no
# BASED ON CENTROIDS (USE FOR BOTH CENTROID/HYBRID STYLE AND BEN STYLE)
nxctr_5,nyctr_5 = 1994.041795233578284,153.24022886329969
nxctr_l,nyctr_l = 2039.50042823306298,198.44560475888629
nxctr_o,nyctr_o = 2084.777073215971605,243.77436076055096
