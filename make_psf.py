from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from scipy.ndimage import interpolation
import scipy.optimize as opt
from photutils.centroids import centroid_com, centroid_2dg  # centroid_quadratic,

# 2D GAUSSIAN!
def twoD_Gaussian(coords, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    x, y = coords
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)))

    return g.ravel()


def find_center(idata, xi, yi, peak):
    '''
    :param idata: flc image
    :param xi: start of x range to explore with centroid
    :param yi: start of y range to explore with centroid
    :param peak: initial guess for amplitude of 2D gaussian
    :return:
    '''

    # :param xf: end of x range to explore with centroid (xi+100)
    # :param yf: end of y range to explore with centroid (yi+100)

    idata = np.nan_to_num(idata)

    xf = xi + 100
    yf = yi + 100
    idata_small = idata[yi:yf, xi:xf]

    from numpy import unravel_index
    maxes = unravel_index(idata_small.argmax(), idata_small.shape)
    print(maxes, 'maxloc!')

    initial_guess = (peak, 50, 50, 5., 5., 0.87, 0.)  # amplitude, x0, y0, sigma_x, sigma_y, theta [rad], offset

    x = np.linspace(0, len(idata_small), len(idata_small))
    y = np.linspace(0, len(idata_small[0]), len(idata_small[0]))
    x, y = np.meshgrid(x, y)

    gg = twoD_Gaussian((x, y), peak, 50, 50, 5., 5., 0.87, 0.)
    print(gg.shape)
    print(idata.shape)

    import scipy.optimize as opt
    popt, pcov = opt.curve_fit(twoD_Gaussian, (x, y), idata_small.ravel(), p0=initial_guess)
    print('2D gaussian (amp, x, y, sigx, sigy, theta, offset)', popt)
    print('above here is 2D gauss result!')

    xcen, ycen = centroid_2dg(idata_small, mask=np.isnan(idata_small))
    print('centroid', xcen, ycen)


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

n384_psf_f814w = hst_n + 'n384_f814w00.fits'
p11179_h = fj + 'PGC11179_F160W_drz_sci.fits'
p11179_adj_h = fj + 'PGC11179_F160W_drz_sci_adjusted.fits'

# PSFs
p11179_rmq_psf = hst_p + 'p11179_rmq_psf00.fits'
p11179_rxq_psf = hst_p + 'p11179_rxq_psf00.fits'
p11179_rzq_psf = hst_p + 'p11179_rzq_psf00.fits'
p11179_f814w_psfm = hst_p + 'psf_ic0b14rmq_flc.fits'
p11179_f814w_psfx = hst_p + 'psf_ic0b14rxq_flc.fits'
p11179_f814w_psfz = hst_p + 'psf_ic0b14rzq_flc.fits'
n384_f814w_psf5 = hst_n + 'psf_ic0b09v5q_flc.fits'
n384_f814w_psfl = hst_n + 'psf_ic0b09vlq_flc.fits'
n384_f814w_psfo = hst_n + 'psf_ic0b09voq_flc.fits'
# FLCs
p11179_f814w_m = hst_p + 'ic0b14rmq_flc.fits'
p11179_f814w_x = hst_p + 'ic0b14rxq_flc.fits'
p11179_f814w_z = hst_p + 'ic0b14rzq_flc.fits'
n384_f814w_5 = hst_n + 'ic0b09v5q_flc.fits'
n384_f814w_l = hst_n + 'ic0b09vlq_flc.fits'
n384_f814w_o = hst_n + 'ic0b09voq_flc.fits'

zp_H = 24.662  # https://www.stsci.edu/hst/instrumentation/wfc3/data-analysis/photometric-calibration/ir-photometric-calibration
zp_I = 24.699  # ACTUALLY UVIS1?!?! (UVIS2=24.684)
# https://www.stsci.edu/hst/instrumentation/wfc3/data-analysis/photometric-calibration/uvis-photometric-calibration

with fits.open(p11179_rmq_psf) as hdu:
    hdr_p0 = hdu[0].header
    dat_psfm = hdu[0].data

with fits.open(p11179_rxq_psf) as hdu:
    hdr_p1 = hdu[0].header
    dat_psfx = hdu[0].data

with fits.open(p11179_rzq_psf) as hdu:
    hdr_p2 = hdu[0].header
    dat_psfz = hdu[0].data

with fits.open(p11179_f814w_m) as hdu:
    hdr_pm = hdu[0].header
    dat_pm = hdu[4].data

with fits.open(p11179_f814w_x) as hdu:
    hdr_px = hdu[0].header
    dat_px = hdu[4].data

with fits.open(p11179_f814w_z) as hdu:
    hdr_pz = hdu[0].header
    dat_pz = hdu[4].data

# TO DO: ACTUALLY calculate image center based on center of rmq, rxq, rzq flc fits files
# 2005.8, 199.7 (for rmq?) atv aperture photometry!
psf_m = np.zeros(shape=dat_pm.shape)  # rmq -> pattstep = 0, no pattern, no dither. Just center it over galaxy!
psf_x = np.zeros(shape=dat_px.shape)  # rxq -> pattstep = 1, pattern1=LINE (0,0) [must be 0,0, since step=2 exists]
psf_z = np.zeros(shape=dat_pz.shape)  # rzq -> pattstep = 2, pattern1=LINE (2.5, 2.5)
# IF it should be LINE-3PT: (0,0) , (2.33,2.33), (4.67,4.67)
# For xc0, yc0: POS TARG = -1.7952, -1.9130 [unit = arcsec]! (https://www.stsci.edu/itt/APT_help20/WFC3/appendixC3.html)
## For UVIS: POS TARG X ~ 0.0396 ["/pix] * x [pix]; POS TARG Y ~ 0.0027 ["/pix] * x [pix] + 0.0395 ["/pix] * y [pix]
## -> x [pix] = POS TARG X / 0.0396; y [pix] = (POS TARG Y - 0.0027 * x) / 0.0395
#  Also clarified here: http://guaix.fis.ucm.es/~agpaz/Instrumentacion_Espacio_2010/Espacio_Docs/HST/hst_c15_phaseII.pdf
# SAME FOR BOTH GALAXIES!! Just Change xctr, yctr

pxctr_m,pyctr_m = 2006,200
pxctr_x,pyctr_x = 2052,245
pxctr_z,pyctr_z = 2097,291

# dat_psfm, *psfx, *psfz all centered at 44,44
# PUT 44,44 at galaxy center!!
f_psfm = np.zeros(shape=dat_pm.shape)
f_psfx = np.zeros(shape=dat_px.shape)
f_psfz = np.zeros(shape=dat_pz.shape)

f_psfm[pyctr_m-44:pyctr_m+45, pxctr_m-44:pxctr_m+45] = dat_psfm  # correct!!!!!!
f_psfx[pyctr_x-44:pyctr_x+45, pxctr_x-44:pxctr_x+45] = dat_psfx  # correct!!!!!!
f_psfz[pyctr_z-44:pyctr_z+45, pxctr_z-44:pxctr_z+45] = dat_psfz  # correct!!!!!!

#fits.writeto('/Users/jonathancohn/Documents/dyn_mod/psf_rmq.fits', f_psfm)
#fits.writeto('/Users/jonathancohn/Documents/dyn_mod/psf_rxq.fits', f_psfx)
#fits.writeto('/Users/jonathancohn/Documents/dyn_mod/psf_rzq.fits', f_psfz)

#plt.imshow(f_psfm, origin='lower')  # CORRECT!!!!!!
#plt.colorbar()
#plt.plot(pxctr_m, pyctr_m, 'r*')
#plt.show()

with fits.open(p11179_f814w_psfz, 'update') as hdu:
    hdu[4].data = f_psfz
    # hdu.writeto('psf_rzq_test.fits')
    hdu.flush()
    # can view these directly e.g. using: "ds9 ic0b14rxq_flc.fits[4]" on the command line!

with fits.open(p11179_f814w_psfx, 'update') as hdu:
    hdu[4].data = f_psfx
    # hdu.writeto('psf_rxq_test.fits')
    hdu.flush()
    # can view these directly e.g. using: "ds9 ic0b14rxq_flc.fits[4]" on the command line!

with fits.open(p11179_f814w_psfm, 'update') as hdu:
    hdr = hdu[0].header
    dat0 = hdu[0].data
    # dats = hdu['SCI'].data
    dat1 = hdu[1].data  # sci uvis2
    dat2 = hdu[2].data  # err uvis2
    dat3 = hdu[3].data  # data quality uvis2
    dat4 = hdu[4].data  # sci uvis1
    dat5 = hdu[5].data  # err uvis1
    dat6 = hdu[6].data  # dq uvis1
    hdu[4].data = f_psfm
    # hdu.writeto('psf_rmq_test.fits')
    hdu.flush()
    # can view these directly e.g. using: "ds9 ic0b14rmq_flc.fits[4]" on the command line!
    print(dat0)
    # print(dats.shape)
    print(dat1.shape)
    print(dat2.shape)
    print(dat3.shape)
    print(dat4.shape)
    print(dat5.shape)

print(oops)

'''  # INTERPOLATION STUFF
# xctr,yctr = pxctr_m,pyctr_m  # nxctr,nyctr
#xc0 = xctr+(-1.7952 / 0.0396)  # = -45.3333333
#xc1 = xctr+0
#xc2 = xctr+2.5
#yc0 = yctr+((-1.9130 - 0.0027 * (-1.7952 / 0.0396)) / 0.0395)  # = -45.3316456
#yc1 = yctr+0
#yc2 = yctr+2.5

xorig = np.zeros(shape=len(dat_pm[0]))
yorig = np.zeros(shape=len(dat_pm))
origres = 0.0337078652
for i in range(len(xorig)):
    xorig[i] = origres * (i - len(dat_pm[0]) / 2.)
for i in range(len(yorig)):
    yorig[i] = origres * (i - len(dat_pm) / 2.)

xx = np.zeros(shape=len(psf_m))
xm = np.zeros(shape=len(psf_x))
xz = np.zeros(shape=len(psf_z))
ym = np.zeros(shape=len(psf_m[0]))
yx = np.zeros(shape=len(psf_x[0]))
yz = np.zeros(shape=len(psf_z[0]))
targetres = 0.06  # arcsec/pix; same as i_data res!

for i in range(len(xm)):
    xm[i] = targetres * (i - pxctr_m)
    xx[i] = targetres * (i - pxctr_x)
    xz[i] = targetres * (i - pxctr_z)
for i in range(len(ym)):
    ym[i] = targetres * (i - pyctr_m)
    yx[i] = targetres * (i - pyctr_x)
    yz[i] = targetres * (i - pyctr_z)
    
#f_psf_m = interpolate.interp2d(xorig,yorig,dat_psfm)
#f_psf_x = interpolate.interp2d(xorig,yorig,dat_psfx)
#f_psf_z = interpolate.interp2d(xorig,yorig,dat_psfz)

#psf_m = f_psf_m(xm, ym)
#psf_x = f_psf(xx, yx)
#psf_z = f_psf(xz, yz)
#plt.imshow(psf_0, origin='lower')  # 1535, 1601
#plt.colorbar()
#plt.show()
#psftest = f_psf(xorig,yorig)
#print(xorig, yorig)
#print(x0, y0)
#print(oop)

plt.imshow(psf_m, origin='lower')
plt.colorbar()
plt.show()

plt.imshow(psf_x, origin='lower')
plt.colorbar()
plt.show()

plt.imshow(psf_z, origin='lower')
plt.colorbar()
plt.show()
#fits.writeto('/Users/jonathancohn/Documents/dyn_mod/psf0.fits', psf_0)
#fits.writeto('/Users/jonathancohn/Documents/dyn_mod/psf1.fits', psf_1)
#fits.writeto('/Users/jonathancohn/Documents/dyn_mod/psf2.fits', psf_2)
print(oop)
# '''  #

#print(fits.printdiff(hst_n + 'ic0b09v5q_flc.fits', hst_n + 'psf_ic0b09v5q_flc.fits'))
#print(oops)

with fits.open(hst_n + 'ic0b09v5q_flc.fits') as hdu:
    hdr = hdu[0].header
    dat0 = hdu[0].data
    dats = hdu['SCI'].data
    dat1 = hdu[1].data  # sci uvis2
    dat2 = hdu[2].data  # err uvis2
    dat3 = hdu[3].data  # data quality uvis2
    dat4 = hdu[4].data  # sci uvis1
    dat5 = hdu[5].data  # err uvis1
    dat6 = hdu[6].data  # dq uvis1
    # can view these directly e.g. using: "ds9 ic0b14rmq_raw.fits[4]" on the command line!
    print(dat0)
    print(dats.shape)
    print(dat1.shape)
    print(dat2.shape)
    print(dat3.shape)
    print(dat4.shape)
    print(dat5.shape)
print(oop)


with fits.open(hst_n + 'ic0b09v5q_flc.fits', 'update') as hdu:
    hdr_v5q = hdu[0].header
    dat_v5q = hdu[1].data
    hdr_v5q['history'] = 'Replaced ic0b09v5q_flc.fits science image with PSF data!'
    print(hdu.info())
#fits.writeto(n384_f814w_psf5, psf_0, hdr_v5q)
    # data_v5q = psf_0
    hdu['SCI'].data = psf_0
    #hdu[1].data = psf_0
    hdu[0].header = hdr_v5q
    print(hdu.info())
    #hdu.flush()
    fits.writeto(n384_f814w_psf5, psf_0, hdr_v5q, 'sci', 0, overwrite=True)

    #print(hdu.info())

with fits.open(hst_n + 'ic0b09vlq_flc.fits', 'update') as hdu:
    hdr_vlq = hdu[0].header
    dat_vlq = hdu[0].data

    hdr_vlq['history'] = 'Replaced ic0b09vlq_flc.fits science image with PSF data!'
#fits.writeto(n384_f814w_psfl, psf_1, hdr_vlq)
    #fits.update(n384_f814w_psfl, psf_1, hdr_vlq, 'sci')
    # dat_vlq = psf_1
    #hdu['SCI',1].data = psf_1
    hdu[1].data = psf_1
    hdu[0].header = hdr_vlq
    hdu.flush()

with fits.open(hst_n + 'ic0b09voq_flc.fits', 'update') as hdu:
    hdr_voq = hdu[0].header
    dat_voq = hdu[0].data

    hdr_voq['history'] = 'Replaced ic0b09voq_flc.fits science image with PSF data!'
#fits.writeto(n384_f814w_psfo, psf_2, hdr_voq)
    # fits.update(n384_f814w_psfo, psf_2, hdr_voq, 'sci')
    # dat_voq = psf_2
    #hdu['SCI',1].data = psf_2
    hdu[1].data = psf_2
    hdu[0].header = hdr_voq
    hdu.flush()
print(oop)

# '''  # PGC 11179 EXAMINE PSF DITHERS
plt.imshow(psf_0[1575:1725, 1505:1655], origin='lower')  # center at 26,30 (lean toward lower)
plt.colorbar()
plt.show()
plt.imshow(psf_1[1575:1725, 1505:1655], origin='lower')  # center at 76,72 (lean toward lower)
plt.colorbar()
plt.show()
plt.imshow(psf_2[1575:1725, 1505:1655], origin='lower')  # center at 78,74 (lean toward higher)
plt.colorbar()
plt.show()

with fits.open(hst_p + 'ic0b14rmq_flc.fits', 'rb+') as hdu:
    hdr_rmq = hdu[0].header
    dat_rmq = hdu[0].data

hdr_rmq['history'] = 'Replaced ic0b14rmq_flc.fits science image with PSF data!'
#fits.writeto(p11179_f814w_psfm, psf_0, hdr_rmq)
fits.update(p11179_f814w_psfm, psf_0, hdr_rmq, 'sci')

with fits.open(hst_p + 'ic0b14rxq_flc.fits', 'rb+') as hdu:
    hdr_rxq = hdu[0].header
    dat_rxq = hdu[0].data

hdr_rxq['history'] = 'Replaced ic0b14rxq_flc.fits science image with PSF data!'
#fits.writeto(p11179_f814w_psfx, psf_1, hdr_rxq)
fits.update(p11179_f814w_psfx, psf_1, hdr_rxq, 'sci')

with fits.open(hst_p + 'ic0b14rzq_flc.fits', 'rb+') as hdu:
    hdr_rzq = hdu[0].header
    dat_rzq = hdu[0].data

hdr_rzq['history'] = 'Replaced ic0b14rzq_flc.fits science image with PSF data!'
# fits.writeto(p11179_f814w_psfz, psf_2, hdr_rzq)
fits.update(p11179_f814w_psfz, psf_2, hdr_rzq, 'sci')
print(oop)
# '''  #

# P11179 FIND CENTER OF FLC FILES
find_center(dat_pz, 2050, 240, 6300.)
# maxloc: [2097, 291] STABLE
# 2D gauss: 4.63267593e+01  4.97281038e+01 [2096.3, 289.7] UNSTABLE
# cen: 45.85933876735593 49.2277176176716 [2095.9, 289.2] STABLE

find_center(dat_px, 2000, 200, 9000.)
# maxloc: [2052, 245] STABLE
# 2D gauss: 5.10612198e+01 4.43809128e+01 [2051.1, 244.4] UNSTABLE
# cen: 50.543907361864704 43.931833647663524 [2050.5, 243.9] STABLE
#print(oop)

find_center(dat_pm, 1950, 150, 3000.)
# maxloc [2006, 200] STABLE
# 2D gauss: 5.56384595e+01  4.90790934e+01 [2005.6, 199.1] UNSTABLE
# cen: 55.08392222855715 48.60211849076918 [2005.1, 198.6] STABLE
print(oop)

plt.imshow(dat_pz, origin='lower', vmax=6300., vmin=0)  # max 2097, 291, xi=2050, yi=240
plt.colorbar()
plt.show()

plt.imshow(dat_px, origin='lower', vmax=9000., vmin=0)  # max 2052, 245, xi=2000, yi=200
plt.colorbar()
plt.show()

plt.imshow(dat_pm, origin='lower', vmax=3900., vmin=0)
plt.plot(1950+5.56158850e+01, 150+4.90380694e+01, 'r*')  # gauss
# max = 2006, 200
plt.plot(1950+56., 150+50., 'm+')  # max
plt.plot(1950+55.05, 150+48.6, 'c+')  # centroid
plt.colorbar()
plt.show()
print(oop)

find_center(dat_pz, 2050, 240, 6300.)
# maxloc: [2097, 291]
# 2D gauss: 4.63267593e+01  4.97281038e+01 [2096.3, 289.7]
# cen: 45.85933876735593 49.2277176176716 [2095.9, 289.2]

find_center(dat_px, 2000, 200, 9000.)
# maxloc: [2052, 245]
# 2D gauss: 5.10612198e+01 4.43809128e+01 [2051.1, 244.4]
# cen: 50.543907361864704 43.931833647663524 [2050.5, 243.9]
#print(oop)

find_center(dat_pm, 1950, 150, 3000.)
# maxloc [2006, 200]
# 2D gauss: 5.56384595e+01  4.90790934e+01 [2005.6, 199.1]
# cen: 55.08392222855715 48.60211849076918 [2005.1, 198.6]

# rmq tests:
# maxloc:  # 1950, 150
# 50, 56

# 1960, 160, 3000.
# 2D gauss: 8.44637745e+01  8.15378479e+01
# cen: 45.0676098014803 38.58309272573514

# 1950, 150, 3000.
# 2D gauss: 5.56384595e+01  4.90790934e+01
# cen: 55.08392222855715 48.60211849076918

# 1950, 150, 2000.
# 2D gauss: 5.56540490e+01  4.91025709e+01
# cen: 55.08392222855715 48.60211849076918

# 1940, 140, 3000.
# 2D gauss: 4.32271081e+01  4.19935679e+01
# cen: 65.02754530560121 58.53671045948828

# 1940, 140, 2000.
# 2D gauss: 6.57007019e+01  5.91450037e+01
# cen: 65.02754530560121 58.53671045948828
