from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate, optimize, ndimage
from photutils.centroids import centroid_com, centroid_2dg  # centroid_quadratic,
import time

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
# ssp_n = ndimage.gaussian_filter(dat_nI, [diff_sigma_nx, diff_sigma_ny]) # convolve with kernel!
t1 = time.time()
print('time in gaussian filter', t1 - tn)
ssp_p = ndimage.gaussian_filter(dat_pgc_psfcen_f814w, [diff_sigma_px, diff_sigma_py]) # convolve with kernel!
# ssp_p = ndimage.gaussian_filter(dat_pI, [diff_sigma_px, diff_sigma_py]) # convolve with kernel!
print('time in gaussian filter', time.time() - t1)
gn2 = do_2dgaussfit(ssp_n[1820:2020, 1800:2020])  # [1.04957099e-03 1.64647665e+03 1.58224407e+03 8.65402658e-01 8.51768218e-01]
gp2 = do_2dgaussfit(ssp_p[1500:1700, 1500:1700])  # [1.04957099e-03 1.64647665e+03 1.58224407e+03 8.65402658e-01 8.51768218e-01]
print(gn2)
print(gp2)
print(gH)
plt.imshow(ssp_n, origin='lower')
plt.colorbar()
plt.show()
plt.imshow(ssp_p, origin='lower')
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
