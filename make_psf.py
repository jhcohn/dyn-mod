from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate, optimize
from scipy.ndimage import interpolation
from photutils.centroids import centroid_com, centroid_2dg  # centroid_quadratic,

# 2D GAUSSIAN!
def twoD_Gaussian(coords, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    x, y = coords
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)))

    return g.ravel()


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

def do_2dgaussfit(data):
    data = np.nan_to_num(data)
    params = fitgaussian(data)
    fit = gaussian(*params)

    plt.contour(fit(*np.indices(data.shape)))
    ax = plt.gca()
    (height, x, y, width_x, width_y) = params
    print(params)

    plt.text(0.95, 0.05, """x : %.1f\ny : %.1f\nwidth_x : %.1f\nwidth_y : %.1f""" % (x, y, width_x, width_y),
             fontsize=16, horizontalalignment='right', verticalalignment='bottom', transform=ax.transAxes)
    plt.show()


def gauss2d_simple(coords, amplitude, xo, yo, sigma_x, sigma_y):
    x, y = coords
    return amplitude * np.exp(-(((xo-x)/sigma_x)**2 + ((yo-y)/sigma_y)**2)/2)


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
    idata_small[idata_small > 1.9e4] = 115.
    #plt.imshow(idata_small, origin='lower')
    #plt.colorbar()
    #plt.show()

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
    popt, pcov = optimize.curve_fit(twoD_Gaussian, (x, y), idata_small.ravel(), p0=initial_guess)
    print('2D gaussian (amp, x, y, sigx, sigy, theta, offset)', popt)
    print('above here is 2D gauss result!')

    xcen, ycen = centroid_2dg(idata_small, mask=np.isnan(idata_small))
    print('centroid', xcen, ycen)


def get_psf_fwhm(psf, xcen=None, ycen=None):
    '''

    :param psf: drizzled PSF
    :param xcen: center of PSF in x
    :param ycen: center of PSF in y
    :return:
    '''

    psf = np.nan_to_num(psf)

    #psf = psf[1922-50:1922+50,1927-50:1927+50]
    #xcen = 50
    #ycen = 50

    if xcen is None:
        xcen, ycen = centroid_2dg(psf, mask=np.isnan(psf))
        # xcen, ycen = 1582.2446553629625, 1646.47696347699  # P11179 centroid (hybrid) method
        # xcen, ycen = 1582.244686019545, 1646.4769456777465  # P11179 ben method
        print('centroid', xcen, ycen)

    # 1582, 1648 (index starting at 1) -> 1581, 1647 (index starting at 0)
    #initial_guess = (np.nanmax(psf), xcen, ycen, 0.7, 0.7, 0., 0.)  # amp, x0, y0, sigma_x, sigma_y, theta [rad], offset
    initial_guess = (np.nanmax(psf), xcen, ycen, 0.8, 0.8, 0., 0.)  # amp, x0, y0, sigma_x, sigma_y, theta [rad], offset

    x = np.linspace(0, len(psf), len(psf))
    y = np.linspace(0, len(psf[0]), len(psf[0]))
    x, y = np.meshgrid(x, y)

    print('start fit')
    gg = twoD_Gaussian((x, y), np.nanmax(psf), xcen, ycen, 0.7, 0.7, 0., 0.)
    popt, pcov = optimize.curve_fit(twoD_Gaussian, (x, y), psf.ravel(), p0=initial_guess)
    print('2D gaussian (amp, x, y, sigx, sigy, theta, offset)', popt)

    datafit = twoD_Gaussian((x, y), *popt)
    plt.imshow(psf, origin='lower')  # fwhm = 2.355 * sigma
    plt.text(popt[1]-5, popt[2]+5, r'$\sigma_x=$' + str(round(popt[3], 4)) + r', $\sigma_y=$' + str(round(popt[4], 4)),
             color='w')
    plt.contour(x, y, datafit.reshape(len(psf[0]), len(psf)), 5, colors='w')
    plt.show()


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


# FINAL PSFs
psfH = fj + 'psfH.fits'
p11179_driz_f814w_psf = hst_p + 'p11179_f814w_drizflc_pxf08_006_psf_take2_sci.fits'
p11179_driz_f814w_psfcen = hst_p + 'p11179_f814w_drizflc_pxf08_006_psf_centroid_sci.fits'
p11179_driz_f814w_psfben = hst_p + 'p11179_f814w_drizflc_pxf08_006_psf_benstyle_sci.fits'
n384_driz_f814w_psf = hst_n + 'n384_f814w_drizflc_pxf08_006_psf_take2_sci.fits'
n384_driz_f814w_psfcen = hst_n + 'n384_f814w_drizflc_pxf08_006_psf_centroid_sci.fits'
n384_driz_f814w_psfben = hst_n + 'n384_f814w_drizflc_pxf08_006_psf_benstyle_sci.fits'
# 03"/pix SCALE FINAL PSFs FOR TESTING
p11179_driz_f814w_psf_03 = hst_p + 'p11179_f814w_drizflc_pxf08_003_psf_take2_sci.fits'
p11179_driz_f814w_psfcen_03 = hst_p + 'p11179_f814w_drizflc_pxf08_003_psf_centroid_sci.fits'
p11179_driz_f814w_psfben_03 = hst_p + 'p11179_f814w_drizflc_pxf08_003_psf_benstyle_sci.fits'
n384_driz_f814w_psf_03 = hst_n + 'n384_f814w_drizflc_pxf08_003_psf_take2_sci.fits'
n384_driz_f814w_psfcen_03 = hst_n + 'n384_f814w_drizflc_pxf08_003_psf_centroid_sci.fits'
n384_driz_f814w_psfben_03 = hst_n + 'n384_f814w_drizflc_pxf08_003_psf_benstyle_sci.fits'
# TINY TIM PSFs
p11179_rmq_psf = hst_p + 'p11179_rmq_psf00.fits'
p11179_rxq_psf = hst_p + 'p11179_rxq_psf00.fits'
p11179_rzq_psf = hst_p + 'p11179_rzq_psf00.fits'
n384_v5q_psf = hst_n + 'n384_v5q_psf00.fits'
n384_vlq_psf = hst_n + 'n384_vlq_psf00.fits'
n384_voq_psf = hst_n + 'n384_voq_psf00.fits'
# PSFs COPIED FROM FLCs FOR INPUT INTO DRIZZLE
p11179_f814w_psfm = hst_p + 'psf_ic0b14rmq_flc.fits'  # jonelle style!
p11179_f814w_psfx = hst_p + 'psf_ic0b14rxq_flc.fits'
p11179_f814w_psfz = hst_p + 'psf_ic0b14rzq_flc.fits'
p11179_f814w_psfcenm = hst_p + 'psfcen_ic0b14rmq_flc.fits'  # hybrid style!
p11179_f814w_psfcenx = hst_p + 'psfcen_ic0b14rxq_flc.fits'
p11179_f814w_psfcenz = hst_p + 'psfcen_ic0b14rzq_flc.fits'
p11179_f814w_psfbenm = hst_p + 'psfben_ic0b14rmq_flc.fits'  # ben style!
p11179_f814w_psfbenx = hst_p + 'psfben_ic0b14rxq_flc.fits'
p11179_f814w_psfbenz = hst_p + 'psfben_ic0b14rzq_flc.fits'
n384_f814w_psf5 = hst_n + 'psf_ic0b09v5q_flc.fits'  # jonelle style
n384_f814w_psfl = hst_n + 'psf_ic0b09vlq_flc.fits'
n384_f814w_psfo = hst_n + 'psf_ic0b09voq_flc.fits'
n384_f814w_psfcen5 = hst_n + 'psfcen_ic0b09v5q_flc.fits'  # hybrid style
n384_f814w_psfcenl = hst_n + 'psfcen_ic0b09vlq_flc.fits'
n384_f814w_psfceno = hst_n + 'psfcen_ic0b09voq_flc.fits'
n384_f814w_psfben5 = hst_n + 'psfben_ic0b09v5q_flc.fits'  # ben style!
n384_f814w_psfbenl = hst_n + 'psfben_ic0b09vlq_flc.fits'
n384_f814w_psfbeno = hst_n + 'psfben_ic0b09voq_flc.fits'
# FLCs
p11179_f814w_m = hst_p + 'ic0b14rmq_flc.fits'
p11179_f814w_x = hst_p + 'ic0b14rxq_flc.fits'
p11179_f814w_z = hst_p + 'ic0b14rzq_flc.fits'
n384_f814w_5 = hst_n + 'ic0b09v5q_flc.fits'
n384_f814w_l = hst_n + 'ic0b09vlq_flc.fits'
n384_f814w_o = hst_n + 'ic0b09voq_flc.fits'

zp_H = 24.662  # https://www.stsci.edu/hst/instrumentation/wfc3/data-analysis/photometric-calibration/ir-photometric-calibration
zp_I = 24.699  # ACTUALLY UVIS1!! (UVIS2=24.684)
# https://www.stsci.edu/hst/instrumentation/wfc3/data-analysis/photometric-calibration/uvis-photometric-calibration

with fits.open(psfH) as hdu:
    hdr_psfh = hdu[0].header
    dat_psfh = hdu[0].data

with fits.open(p11179_driz_f814w_psf) as hdu:
    hdr_pgc_psf_f814w = hdu[0].header
    dat_pgc_psf_f814w = hdu[0].data

with fits.open(p11179_driz_f814w_psfcen) as hdu:
    hdr_pgc_psfcen_f814w = hdu[0].header
    dat_pgc_psfcen_f814w = hdu[0].data

with fits.open(p11179_driz_f814w_psfben) as hdu:
    hdr_pgc_psfben_f814w = hdu[0].header
    dat_pgc_psfben_f814w = hdu[0].data

with fits.open(p11179_driz_f814w_psf_03) as hdu:
    hdr_pgc_psf_f814w_03 = hdu[0].header
    dat_pgc_psf_f814w_03 = hdu[0].data

with fits.open(p11179_driz_f814w_psfcen_03) as hdu:
    hdr_pgc_psfcen_f814w_03 = hdu[0].header
    dat_pgc_psfcen_f814w_03 = hdu[0].data

with fits.open(p11179_driz_f814w_psfben_03) as hdu:
    hdr_pgc_psfben_f814w_03 = hdu[0].header
    dat_pgc_psfben_f814w_03 = hdu[0].data

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

with fits.open(n384_driz_f814w_psf) as hdu:
    hdr_ngc_psf_f814w = hdu[0].header
    dat_ngc_psf_f814w = hdu[0].data

with fits.open(n384_driz_f814w_psfcen) as hdu:
    hdr_ngc_psfcen_f814w = hdu[0].header
    dat_ngc_psfcen_f814w = hdu[0].data

with fits.open(n384_driz_f814w_psfben) as hdu:
    hdr_ngc_psfben_f814w = hdu[0].header
    dat_ngc_psfben_f814w = hdu[0].data

with fits.open(n384_driz_f814w_psf_03) as hdu:
    hdr_ngc_psf_f814w_03 = hdu[0].header
    dat_ngc_psf_f814w_03 = hdu[0].data

with fits.open(n384_driz_f814w_psfcen_03) as hdu:
    hdr_ngc_psfcen_f814w_03 = hdu[0].header
    dat_ngc_psfcen_f814w_03 = hdu[0].data

with fits.open(n384_driz_f814w_psfben_03) as hdu:
    hdr_ngc_psfben_f814w_03 = hdu[0].header
    dat_ngc_psfben_f814w_03 = hdu[0].data

with fits.open(n384_v5q_psf) as hdu:
    hdr_n0 = hdu[0].header
    dat_psf5 = hdu[0].data

with fits.open(n384_vlq_psf) as hdu:
    hdr_n1 = hdu[0].header
    dat_psfl = hdu[0].data

with fits.open(n384_voq_psf) as hdu:
    hdr_n2 = hdu[0].header
    dat_psfo = hdu[0].data

with fits.open(n384_f814w_5) as hdu:
    hdr_n5 = hdu[0].header
    dat_n5 = hdu[4].data

with fits.open(n384_f814w_l) as hdu:
    hdr_nl = hdu[0].header
    dat_nl = hdu[4].data

with fits.open(n384_f814w_o) as hdu:
    hdr_no = hdu[0].header
    dat_no = hdu[4].data

#plt.imshow(dat_psfh, origin='lower')
#plt.colorbar()
#plt.show()
do_2dgaussfit(dat_psfm)  # [ 0.11947732 43.99928028 43.99479415  0.96549582  0.9681218 ]
do_2dgaussfit(dat_psfx)  # [ 0.11950119 43.99918033 43.99485935  0.96535846  0.96801687]
do_2dgaussfit(dat_psfz)  # [ 0.11950464 43.99917304 43.99504149  0.96536614  0.96799807]
do_2dgaussfit(dat_psf5)  # [ 0.11947835 43.99929119 43.99470546  0.96547872  0.96820728]
do_2dgaussfit(dat_psfo)  # [ 0.11949347 43.99917936 43.99496006  0.96547602  0.96806489]
do_2dgaussfit(dat_psfl)  # [ 0.11949002 43.99912657 43.9947759   0.96539584  0.96815537]
#do_2dgaussfit(dat_psfh)  # [ 0.04847133 40.03064685 39.91339118  1.45843277  1.49161195]
# do_2dgaussfit(dat_pgc_psfben_f814w_03[3250:3350, 3110:3210])  # 3166, 3295
# [3250:3350, 3110:3210] -> [2.98088782e-04 4.34327133e+01 5.49931741e+01 1.57468666e+00 1.58472555e+00]
# do_2dgaussfit(dat_pgc_psfcen_f814w_03[3250:3350, 3110:3210])  # 3166, 3295
# [3250:3350, 3110:3210] -> [2.98080786e-04 4.34327533e+01 5.49931119e+01 1.57470757e+00 1.58476141e+00]
# do_2dgaussfit(dat_pgc_psf_f814w_03[3250:3350, 3110:3210])  # 3163, 3295
# [3250:3350, 3110:3210] -> [3.32459023e-04 4.44731343e+01 5.27570091e+01 1.47076506e+00 1.46177679e+00]
# do_2dgaussfit(dat_ngc_psf_f814w_03[3700:4000, 3700:4000])  #
# [3800:3900, 3800:3900] -> [3.48986047e-04 4.39221921e+01 5.47504082e+01 1.38635260e+00 1.53387333e+00]
# do_2dgaussfit(dat_ngc_psfcen_f814w_03)  #
# [3800:3900, 3800:3900] -> [3.09420628e-04 4.47922338e+01 5.48503835e+01 1.54821846e+00 1.57455480e+00]
# do_2dgaussfit(dat_ngc_psfben_f814w_03[3800:3900, 3800:3900])  #
# [3800:3900, 3800:3900] -> [3.09404579e-04 4.47921413e+01 5.48503251e+01 1.54826558e+00 1.57462895e+00]
print(ops)
# https://scipy-cookbook.readthedocs.io/items/FittingData.html
# height, x, y, width_x, width_y
do_2dgaussfit(dat_pgc_psf_f814w)  # [1.19039658e-03 1.64700640e+03 1.58110278e+03 7.78752111e-01 7.83891454e-01]
#[1500:1700, 1500:1700] -> [1.19039658e-03 1.47006402e+02 8.11027805e+01 7.78752111e-01 7.83891454e-01]
do_2dgaussfit(dat_pgc_psfcen_f814w)  # [1.04957099e-03 1.64647665e+03 1.58224407e+03 8.65402658e-01 8.51768218e-01]
# [1500:1700, 1500:1700] -> [1.04957099e-03 1.46476646e+02 8.22440674e+01 8.65402657e-01 8.51768218e-01]
do_2dgaussfit(dat_pgc_psfben_f814w)  # [1.04958903e-03 1.64647663e+03 1.58224410e+03 8.65400337e-01 8.51753571e-01]
# [1500:1700, 1500:1700] -> [1.04958903e-03 1.46476627e+02 8.22440989e+01 8.65400337e-01 8.51753571e-01]
do_2dgaussfit(dat_ngc_psf_f814w)  # [1.20795599e-03 1.92174249e+03 1.92713734e+03 7.75526286e-01 8.20020663e-01]
# [1820:2020, 1800:2020] -> [1.20795599e-03 1.01742489e+02 1.27137337e+02 7.75526286e-01 8.20020662e-01]
do_2dgaussfit(dat_ngc_psfcen_f814w)  # [1.10704459e-03 1.92214472e+03 1.92717995e+03 8.18216509e-01 8.47028032e-01]
# [1820:2020, 1800:2020] -> [1.10704459e-03 1.02144719e+02 1.27179955e+02 8.18216509e-01 8.47028033e-01]
do_2dgaussfit(dat_ngc_psfben_f814w)  # [1.10700135e-03 1.92214469e+03 1.92717991e+03 8.18223999e-01 8.47072199e-01]
# [1820:2020, 1800:2020] -> [1.10700135e-03 1.02144690e+02 1.27179907e+02 8.18223999e-01 8.47072199e-01]
print(oops)

plt.imshow(dat_ngc_psfben_f814w, origin='lower')
plt.colorbar()
plt.show()

plt.imshow(dat_pgc_psfcen_f814w, origin='lower')
plt.colorbar()
plt.show()
find_center(dat_pgc_psfcen_f814w, 1530, 1600, 1e-3)  # 1582.2446553629625, 1646.47696347699  # hybrid style
find_center(dat_ngc_psfcen_f814w, 1875, 1870, 1e-3)  #  # hybrid style  # 1927.180832938775 1922.1457266707657

#get_psf_fwhm(dat_pgc_psf_f814w, 1581, 1647)  # jonelle style
# get_psf_fwhm(dat_pgc_psfcen_f814w, 1582, 1646)  # 1582.2446553629625, 1646.47696347699  # hybrid style
#get_psf_fwhm(dat_pgc_psfben_f814w)  # , 1582, 1646)  # 1582.244686019545 1646.4769456777465  # ben style
# get_psf_fwhm(dat_ngc_psf_f814w, 1927, 1922)  # jonelle style  # n384_drizzled_psf_with2dgauss
#get_psf_fwhm(dat_ngc_psfcen_f814w)  #  # hybrid style  # 1927.180832938775 1922.1457266707657
# get_psf_fwhm(dat_ngc_psfben_f814w, 1927.180785848962, 1922.1456981411623)  # ben style  # 1927.180785848962 1922.1456981411623
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

#plt.imshow(psf_5, origin='lower')  # CORRECT!!!!!!
#plt.colorbar()
#plt.plot(nxctr_5, nyctr_5, 'r*')
#plt.show()
#print(oops)

# '''  # WRITE OUT NGC 384 PSFs (centered using peak)
# MAKE SURE to replace UVIS2 science image with zeroes!!!!! THIS WORKED!!!!!!!!!!!!!!!!!!!!
# p11179_f814w_drizflc_pxf08_006_psf_take2_sci.fits
with fits.open(n384_f814w_psf5, 'update') as hdu:
    hdu[1].data = np.zeros(shape=psf_5.shape)
    hdu[4].data = psf_5
    hdu.flush()

with fits.open(n384_f814w_psfl, 'update') as hdu:
    hdu[1].data = np.zeros(shape=psf_l.shape)
    hdu[4].data = psf_l
    hdu.flush()

with fits.open(n384_f814w_psfo, 'update') as hdu:
    hdu[1].data = np.zeros(shape=psf_o.shape)
    hdu[4].data = psf_o
    hdu.flush()
    # can view these directly e.g. using: "ds9 ic0b14rmq_flc.fits[4]" on the command line!
print(oops)



# BEGIN HYBRID & BEN STYLE!!!!
# cen: 44.041795233578284 53.24022886329969 [1994.4, 153.2] n5
# cen: 39.50042823306298 48.44560475888629 [2039.5, 198.4] nl
# cen: 34.777073215971605 43.77436076055096 [2084.8, 243.8] no
# BASED ON CENTROIDS (USE FOR BOTH CENTROID/HYBRID STYLE AND BEN STYLE)
nxctr_5,nyctr_5 = 1994.041795233578284,153.24022886329969
nxctr_l,nyctr_l = 2039.50042823306298,198.44560475888629
nxctr_o,nyctr_o = 2084.777073215971605,243.77436076055096

# HYBRID & BEN STYLE
# Setup orig tiny tim PSF axes
xorig = np.zeros(shape=len(dat_psf5))
yorig = np.zeros(shape=len(dat_psf5[0]))
for i in range(len(xorig)):
    if (len(xorig) % 2.) == 0:
        xorig[i] = i - len(xorig) / 2.
    else:
        xorig[i] = i - (len(xorig)-1) / 2.
    # xorig = i - 44
for i in range(len(yorig)):
    if (len(yorig) % 2) == 0:
        yorig[i] = i - len(yorig) / 2.
    else:
        yorig[i] = i - (len(yorig)-1) / 2.
    # yorig = i - 44

# Make pixel grids (HYBRID & BEN STYLE)
x05 = np.zeros(shape=len(dat_n5[0]))
y05 = np.zeros(shape=len(dat_n5))
x0l = np.zeros(shape=len(dat_nl[0]))
y0l = np.zeros(shape=len(dat_nl))
x0o = np.zeros(shape=len(dat_no[0]))
y0o = np.zeros(shape=len(dat_no))
for i in range(len(x05)):
    x05[i] = i - nxctr_5
    x0l[i] = i - nxctr_l
    x0o[i] = i - nxctr_o
for i in range(len(y05)):
    y05[i] = i - nyctr_5
    y0l[i] = i - nyctr_l
    y0o[i] = i - nyctr_o


# HYBRID STYLE!!!
f_psf_5 = interpolate.interp2d(xorig,yorig,dat_psf5)
f_psf_l = interpolate.interp2d(xorig,yorig,dat_psfl)
f_psf_o = interpolate.interp2d(xorig,yorig,dat_psfo)

interpd_psf5 = f_psf_5(x05, y05)
interpd_psfl = f_psf_l(x0l, y0l)
interpd_psfo = f_psf_o(x0o, y0o)

# '''  # WRITE OUT P11179 PSFs (2D interpolated) HYBRID STYLE
with fits.open(n384_f814w_psfcen5, 'update') as hdu:
    hdu[1].data = np.zeros(shape=interpd_psf5.shape)
    hdu[4].data = interpd_psf5
    hdu.flush()
with fits.open(n384_f814w_psfcenl, 'update') as hdu:
    hdu[1].data = np.zeros(shape=interpd_psfl.shape)
    hdu[4].data = interpd_psfl
    hdu.flush()
with fits.open(n384_f814w_psfceno, 'update') as hdu:
    hdu[1].data = np.zeros(shape=interpd_psfo.shape)
    hdu[4].data = interpd_psfo
    hdu.flush()
print(oop)
# '''  #
plt.imshow(interpd_psf5, origin='lower')
plt.colorbar()
plt.show()
plt.imshow(interpd_psfl, origin='lower')
plt.colorbar()
plt.show()
plt.imshow(interpd_psfo, origin='lower')
plt.colorbar()
plt.show()
print(oop)


# BEN STYLE!!!
f_psf_5 = interpolate.interp2d(xorig,yorig,dat_psf5)
f_psf_l = interpolate.interp2d(xorig,yorig,dat_psf5)
f_psf_o = interpolate.interp2d(xorig,yorig,dat_psf5)

interpd_psf5 = f_psf_5(x05, y05)
interpd_psfl = f_psf_l(x0l, y0l)
interpd_psfo = f_psf_o(x0o, y0o)

# '''  # WRITE OUT P11179 PSFs (2D interpolated) BEN STYLE
with fits.open(n384_f814w_psfben5, 'update') as hdu:
    hdu[1].data = np.zeros(shape=interpd_psf5.shape)
    hdu[4].data = interpd_psf5
    hdu.flush()
with fits.open(n384_f814w_psfbenl, 'update') as hdu:
    hdu[1].data = np.zeros(shape=interpd_psfl.shape)
    hdu[4].data = interpd_psfl
    hdu.flush()
with fits.open(n384_f814w_psfbeno, 'update') as hdu:
    hdu[1].data = np.zeros(shape=interpd_psfo.shape)
    hdu[4].data = interpd_psfo
    hdu.flush()
print(oop)
plt.imshow(interpd_psf5, origin='lower')
plt.colorbar()
plt.show()
plt.imshow(interpd_psfl, origin='lower')
plt.colorbar()
plt.show()
plt.imshow(interpd_psfo, origin='lower')
plt.colorbar()
plt.show()
print(oop)



# BELOW HERE: PGC 11179 I-BAND PSFs
# ACTUALLY calculate image center based on center of rmq, rxq, rzq flc fits files
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

# cen: 55.08392222855715 48.60211849076918 [2005.1, 198.6] rmq
# cen: 50.543907361864704 43.931833647663524 [2050.5, 243.9] rxq
# cen: 45.85933876735593 49.2277176176716 [2095.9, 289.2] rzq
# BASED ON CENTROIDS
pxctr_m,pyctr_m = 2005.08392222855715,198.60211849076918
pxctr_x,pyctr_x = 2050.543907361864704,243.931833647663524
pxctr_z,pyctr_z = 2095.85933876735593,289.2277176176716

# BEN STYLE
# Setup orig tiny tim PSF axes
zz = np.zeros(shape=88)
xorig = np.zeros(shape=len(dat_psfm))
yorig = np.zeros(shape=len(dat_psfm[0]))
for i in range(len(xorig)):
    if (len(xorig) % 2.) == 0:
        xorig[i] = i - len(xorig) / 2.
    else:
        xorig[i] = i - (len(xorig)-1) / 2.
    # xorig = i - 44
for i in range(len(yorig)):
    if (len(yorig) % 2) == 0:
        yorig[i] = i - len(yorig) / 2.
    else:
        yorig[i] = i - (len(yorig)-1) / 2.
    # yorig = i - 44

# Make pixel grids
x0m = np.zeros(shape=len(dat_pm[0]))
y0m = np.zeros(shape=len(dat_pm))
x0x = np.zeros(shape=len(dat_px[0]))
y0x = np.zeros(shape=len(dat_px))
x0z = np.zeros(shape=len(dat_pz[0]))
y0z = np.zeros(shape=len(dat_pz))
for i in range(len(x0m)):
    x0m[i] = i - pxctr_m
    x0x[i] = i - pxctr_x
    x0z[i] = i - pxctr_z
for i in range(len(y0m)):
    y0m[i] = i - pyctr_m
    y0x[i] = i - pyctr_x
    y0z[i] = i - pyctr_z

f_psf_m = interpolate.interp2d(xorig,yorig,dat_psfx)
f_psf_x = interpolate.interp2d(xorig,yorig,dat_psfx)
f_psf_z = interpolate.interp2d(xorig,yorig,dat_psfx)

interpd_psfm = f_psf_m(x0m, y0m)
interpd_psfx = f_psf_x(x0x, y0x)
interpd_psfz = f_psf_z(x0z, y0z)

# '''  # WRITE OUT P11179 PSFs (2D interpolated)
with fits.open(p11179_f814w_psfbenm, 'update') as hdu:
    hdu[1].data = np.zeros(shape=interpd_psfm.shape)
    hdu[4].data = interpd_psfm
    hdu.flush()
with fits.open(p11179_f814w_psfbenx, 'update') as hdu:
    hdu[1].data = np.zeros(shape=interpd_psfx.shape)
    hdu[4].data = interpd_psfx
    hdu.flush()
with fits.open(p11179_f814w_psfbenz, 'update') as hdu:
    hdu[1].data = np.zeros(shape=interpd_psfz.shape)
    hdu[4].data = interpd_psfz
    hdu.flush()
print(oop)
plt.imshow(interpd_psfz, origin='lower')
plt.colorbar()
plt.show()

plt.imshow(interpd_psfx, origin='lower')
plt.colorbar()
plt.show()

plt.imshow(interpd_psfm, origin='lower')
plt.colorbar()
plt.show()
print(oop)


# HYBRID STYLE
# Setup orig tiny tim PSF axes
zz = np.zeros(shape=88)
xorig = np.zeros(shape=len(dat_psfm))
yorig = np.zeros(shape=len(dat_psfm[0]))
for i in range(len(xorig)):
    if (len(xorig) % 2.) == 0:
        xorig[i] = i - len(xorig) / 2.
    else:
        xorig[i] = i - (len(xorig)-1) / 2.
    # xorig = i - 44
for i in range(len(yorig)):
    if (len(yorig) % 2) == 0:
        yorig[i] = i - len(yorig) / 2.
    else:
        yorig[i] = i - (len(yorig)-1) / 2.
    # yorig = i - 44

# Make pixel grids
x0m = np.zeros(shape=len(dat_pm[0]))
y0m = np.zeros(shape=len(dat_pm))
x0x = np.zeros(shape=len(dat_px[0]))
y0x = np.zeros(shape=len(dat_px))
x0z = np.zeros(shape=len(dat_pz[0]))
y0z = np.zeros(shape=len(dat_pz))
for i in range(len(x0m)):
    x0m[i] = i - pxctr_m
    x0x[i] = i - pxctr_x
    x0z[i] = i - pxctr_z
for i in range(len(y0m)):
    y0m[i] = i - pyctr_m
    y0x[i] = i - pyctr_x
    y0z[i] = i - pyctr_z

f_psf_m = interpolate.interp2d(xorig,yorig,dat_psfm)
f_psf_x = interpolate.interp2d(xorig,yorig,dat_psfx)
f_psf_z = interpolate.interp2d(xorig,yorig,dat_psfz)

interpd_psfm = f_psf_m(x0m, y0m)
interpd_psfx = f_psf_x(x0x, y0x)
interpd_psfz = f_psf_z(x0z, y0z)

# '''  # WRITE OUT P11179 PSFs (2D interpolated)
with fits.open(p11179_f814w_psfcenm, 'update') as hdu:
    hdu[1].data = np.zeros(shape=interpd_psfm.shape)
    hdu[4].data = interpd_psfm
    hdu.flush()
with fits.open(p11179_f814w_psfcenx, 'update') as hdu:
    hdu[1].data = np.zeros(shape=interpd_psfx.shape)
    hdu[4].data = interpd_psfx
    hdu.flush()
with fits.open(p11179_f814w_psfcenz, 'update') as hdu:
    hdu[1].data = np.zeros(shape=interpd_psfz.shape)
    hdu[4].data = interpd_psfz
    hdu.flush()
print(oop)
# '''  #

plt.imshow(interpd_psfz, origin='lower')
plt.colorbar()
plt.show()

plt.imshow(interpd_psfx, origin='lower')
plt.colorbar()
plt.show()

plt.imshow(interpd_psfm, origin='lower')
plt.colorbar()
plt.show()
print(oop)

# JONELLE STYLE
# BASED ON MAX PIXELS
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


'''  # WRITE OUT P11179 PSFs (centered using peak)
# MAKE SURE to replace UVIS2 science image with zeroes!!!!! THIS WORKED!!!!!!!!!!!!!!!!!!!!
# p11179_f814w_drizflc_pxf08_006_psf_take2_sci.fits
with fits.open(p11179_f814w_psfz, 'update') as hdu:
    hdu[1].data = np.zeros(shape=f_psfz.shape)
    hdu[4].data = f_psfz
    # hdu.writeto('psf_rzq_test.fits')
    hdu.flush()
    # can view these directly e.g. using: "ds9 ic0b14rxq_flc.fits[4]" on the command line!

with fits.open(p11179_f814w_psfx, 'update') as hdu:
    hdu[1].data = np.zeros(shape=f_psfx.shape)
    hdu[4].data = f_psfx
    # hdu.writeto('psf_rxq_test.fits')
    hdu.flush()
    # can view these directly e.g. using: "ds9 ic0b14rxq_flc.fits[4]" on the command line!

with fits.open(p11179_f814w_psfm, 'update') as hdu:
    hdu[1].data = np.zeros(shape=f_psfm.shape)
    hdu[4].data = f_psfm
    #dat0 = hdu[0].data
    # dats = hdu['SCI'].data
    #dat1 = hdu[1].data  # sci uvis2
    #dat2 = hdu[2].data  # err uvis2
    #dat3 = hdu[3].data  # data quality uvis2
    #dat4 = hdu[4].data  # sci uvis1
    #dat5 = hdu[5].data  # err uvis1
    #dat6 = hdu[6].data  # dq uvis1
    hdu[4].data = f_psfm
    # hdu.writeto('psf_rmq_test.fits')
    #print(dat0)
    # print(dats.shape)
    #print(dat1.shape)
    #print(dat2.shape)
    #print(dat3.shape)
    #print(dat4.shape)
    #print(dat5.shape)
    hdu.flush()
    # can view these directly e.g. using: "ds9 ic0b14rmq_flc.fits[4]" on the command line!
print(oops)
# '''  #

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


with fits.open(hst_n + 'imfitn384_f814w_drizflc_pxf08_006_psf_take2_sci.fits', 'update') as hdu:
    hdr = hdu[0].header
    hdr['BUNIT'] = 'COUNTS'
    hdr['history'] = 'replaced BUNIT (was ELECTRONS/S) solely to make imfit work'
    hdu[0].header = hdr
    hdu.flush()
print(oops)