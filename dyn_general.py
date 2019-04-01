import numpy as np
import astropy.io.fits as fits
# from astropy.table import Table
import matplotlib.pyplot as plt
from matplotlib import rc
from scipy import integrate, signal, interpolate
from scipy.ndimage import filters, interpolation
# from astropy import modeling
import time
from astropy import convolution
import argparse
from pathlib import Path
from astropy.modeling.models import Box2D
from astropy.nddata.utils import block_reduce
from astropy.modeling.models import Ellipse2D
# from plotbin import display_pixels as dp
# from regions import read_crtf, CRTFParser, DS9Parser, read_ds9, PolygonPixelRegion, PixCoord


# constants
class Constants:  # CONFIRMED
    # a bunch of useful astronomy constants

    def __init__(self):
        self.c = 2.99792458 * 10 ** 8  # m/s
        self.pc = 3.086 * 10 ** 16  # m [or use for m per pc]
        self.G = 6.67 * 10 ** -11  # kg^-1 * m^3 * s^-2
        self.M_sol = 1.989 * 10 ** 30  # kg [or use for kg / solar mass]
        self.H0 = 70  # km/s/Mpc
        self.arcsec_per_rad = 206265.  # arcsec per radian
        self.m_per_km = 10. ** 3  # m per km
        self.G_pc = self.G * self.M_sol * (1. / self.pc) / self.m_per_km ** 2  # Msol^-1 * pc * km^2 * s^-2
        self.c_kms = self.c / self.m_per_km


def make_beam(grid_size=99, res=1., amp=1., x0=0., y0=0., x_std=1., y_std=1., rot=0., fits_name=None):  # CONFIRMED
    """
    Generate a beam spread function

    :param grid_size: size of grid (must be odd!)
    :param res: resolution of the grid (arcsec/pixel)
    :param amp: amplitude of the 2d gaussian
    :param x0: mean of x axis of 2d gaussian
    :param y0: mean of y axis of 2d gaussian
    :param x_std: FWHM of beam in x (to use for standard deviation of Gaussian) (in arcsec)
    :param y_std: FWHM of beam in y (to use for standard deviation of Gaussian) (in arcsec)
    :param rot: rotation angle in radians
    :param fits_name: this name will be the filename to which the beam fits file is written (if None, write no file)

    return the synthesized beam
    """

    # SET RESOLUTION OF BEAM GRID THE SAME AS FLUXES GRID
    x_beam = [0.] * grid_size
    y_beam = [0.] * grid_size

    # set center of the beam grid axes (find the central pixel number along each axis: same for x_beam and y_beam)
    ctr = (len(x_beam) + 1.) / 2.  # +1 bc python starts counting at 0, grid_size is odd

    # grid_size must be odd, so fill in the axes with resolution * ((i+1) - ctr)!
    for i in range(len(x_beam)):
        x_beam[i] = res * ((i + 1) - ctr)  # (arcsec/pix) * N_pixels = arcsec
        y_beam[i] = res * ((i + 1) - ctr)  # (arcsec/pix) * N_pixels = arcsec

    # SET UP MESHGRID
    xx, yy = np.meshgrid(x_beam, y_beam)

    # SET UP PSF 2D GAUSSIAN VARIABLES
    n = 2.35482  # Convert from FWHM to std_dev: http://mathworld.wolfram.com/GaussianFunction.html
    x_std /= n
    y_std /= n
    a = np.cos(rot) ** 2 / (2 * x_std ** 2) + np.sin(rot) ** 2 / (2 * y_std ** 2)
    b = -np.sin(2 * rot) / (4 * x_std ** 2) + np.sin(2 * rot) / (4 * y_std ** 2)
    c = np.sin(rot) ** 2 / (2 * x_std ** 2) + np.cos(rot) ** 2 / (2 * y_std ** 2)

    # CALCULATE PSF, NORMALIZE IT TO AMPLITUDE
    synth_beam = np.exp(-(a * (xx - x0) ** 2 + 2 * b * (xx - x0) * (yy - y0) + c * (yy - y0) ** 2))
    A = amp / np.amax(synth_beam)  # normalize it so the input amplitude is the real amplitude
    synth_beam *= A

    # IF write_fits, WRITE PSF TO FITS FILE
    # if fits_name is not None:
    if not Path(fits_name).exists():
        hdu = fits.PrimaryHDU(synth_beam)
        hdul = fits.HDUList([hdu])
        hdul.writeto(fits_name)

    return synth_beam


def get_sig(r=None, sig0=1., r0=1., mu=1., sig1=0.):  # CONFIRMED
    """
    :param r: 2D array of radius values [r(x, y)]
    :param sig0: uniform sigma_turb component (velocity units)
    :param r0: scale radius value (same units as r)
    :param mu: same units as r
    :param sig1: additional sigma term (same units as sig0)
    :return: dictionary of the three different sigma shapes
    """

    sigma = {'flat': sig0, 'gauss': sig1 + sig0 * np.exp(-(r - r0) ** 2 / (2 * mu ** 2)),
             'exp': sig1 + sig0 * np.exp(-r / r0)}

    return sigma


def pvd(data_cube, theta, z_ax, x_arcsec, R, v_sys):  # BUCKET UNCONFIRMED
    """
    Build position-velocity diagram (PVD) from input data cube
    :param data_cube: input data cube
    :param theta: angle through which to rotate the data cube so that the disk semi-major axis is along the +y axis
    :return: position velocity diagram (PVD)
    """
    hdu = fits.open(data_cube)
    data = hdu[0].data[0]  # header = hdu[0].header

    # REBIN IN GROUPS OF 4x4 PIXELS
    rebinned = []  # np.zeros(shape=(len(z_ax), len(fluxes), len(fluxes[0])))
    t_rebin = time.time()
    for z in range(len(data)):
        subarrays = blockshaped(data[z, :, :], 4, 4)  # bin the data in groups of 4x4 pixels

        # each pixel in the new, rebinned data cube is the mean of each 4x4 set of original pixels
        reshaped = np.mean(np.mean(subarrays, axis=-1), axis=-1).reshape((int(len(data[0]) / 4.),
                                                                          int(len(data[0][0]) / 4.)))
        rebinned.append(reshaped)
    print("Rebinning the cube done in {0} s".format(time.time() - t_rebin))  # 0.5 s
    data = np.asarray(rebinned)
    print(data.shape)

    col = np.zeros_like(data[0])
    for x in range(len(data[0, 0, :])):
        for y in range(len(data[z, :, 0])):
            if len(data[0, 0, :]) / 2. == x:  # if at center point, include  # BUCKET INCLUDE BH OFFSET
                if len(data[z, :, 0]) / 2. == y:  # if at center point, include  # BUCKET INCLUDE BH OFFSET
                    col[y, x] = 1.
            # if x,y pixels are both before halfway point (to get bot-L quarter in python)
            # elif x,y pixels are both after halfway point (to get top-R quarter in python)
            elif (x < len(data[0, 0, :]) / 2. and y < len(data[0, :, 0]) / 2.) or \
                    (x > len(data[0, 0, :]) / 2. and y > len(data[0, :, 0]) / 2.):
                if (theta - 0.5) * np.pi / 180. <= \
                        np.abs(np.arctan((len(data[0, :, 0]) / 2. - y) / (len(data[0, 0, :]) / 2. - x))) \
                        <= (26.7 + 0.5) * np.pi / 180.:  # arbitrarily chose 0.5 degrees as tolerance on angle
                    col[y, x] = 1.
            else:
                col[y, x] = 0.
    # plt.imshow(col, origin='lower')
    # plt.plot(80, 80, 'w*')
    # plt.show()

    data_masked = np.asarray([data[z][:, :] * col[:, :] for z in range(len(data))])
    pvd_fill = np.zeros(shape=(len(data), len(data[0])))
    for z in range(len(data_masked)):
        for c in range(len(data_masked[0][0])):
            pvd_fill[z, c] = np.sum(data_masked[z, :, c])  # sum each column, append results of sum to pvd_fill

    # np.asarray(x_arcsec)[::2] keeps the 0th element of x_arcsec, & every other element after, e.g. 0th, 2nd, 4th, etc.

    plt.contourf(np.asarray(x_arcsec)[::2], z_ax - v_sys, pvd_fill, 600, vmin=np.amin(pvd_fill),
                 vmax=np.amax(pvd_fill), cmap='viridis')
    plt.plot(0, 0, 'w*')
    plt.xlabel(r'Offset [arcsec]', fontsize=20)  # BUCKET THIS IS NOT CORRECT (want offset on 26.7deg axis, not plain x)
    plt.ylabel(r'velocity [km/s]', fontsize=20)  # BUCKET data doesn't match Barth+2016 quite perfectly right now
    plt.colorbar()
    plt.xlim(-2., 2.)
    plt.ylim(-675., 675)
    plt.show()
    plt.close()

    return data_masked, pvd_fill


def get_fluxes(data_cube, data_mask, write_name=None):  # CONFIRMED
    """
    Use to integrate line profiles to get fluxes from data cube!

    :param data_cube: input data cube of observations
    :param data_mask: mask for each slice of data, for construction of the weight map
    :param write_name: name of fits file to which to write collapsed cube (if None, write no file)

    :return: collapsed data cube (i.e. flux map), len(z_ax), intrinsic freq of observed line, and input data cube!
    """
    hdu = fits.open(data_cube)
    data = hdu[0].data[0]  # data[0] --> z, y, x (121, 700, 700)

    hdu_m = fits.open(data_mask)
    mask = hdu_m[0].data  # this is hdu_m[0].data, NOT hdu[0].data[0], unlike the data_cube above

    z_len = len(hdu[0].data[0])  # store the number of velocity slices in the data cube
    freq1 = float(hdu[0].header['CRVAL3'])  # starting frequency in the data cube
    f_step = float(hdu[0].header['CDELT3'])  # frequency step in the data cube
    f_0 = float(hdu[0].header['RESTFRQ'])
    freq_axis = np.arange(freq1, freq1 + (z_len * f_step), f_step)  # [bluest, ..., reddest]
    # NOTE: something is forcing the inclusion of the endpoint, which arange normally doesn't include
    # However, when cutting the endpoint one f_step sooner, arange doesn't include the endpoint...
    # So, I'm including the extra point above, then cutting it off, so that it gives the correct array:
    # NOTE: NGC_3258 DATA DOES *NOT* HAVE THIS ISSUE, SOOOOOOOOOO *SHRUG* COMMENTING OUT THE HACK/FIX BELOW
    # freq_axis = freq_axis[:-1]

    # Collapse the fluxes! Sum over all slices, multiplying each slice by the slice's mask and by the frequency step
    collapsed_fluxes = np.zeros(shape=(len(data[0]), len(data[0][0])))
    for zi in range(len(data)):
        collapsed_fluxes += data[zi] * mask[zi] * abs(f_step)

    # plt.imshow(collapsed_fluxes, origin='lower')
    # plt.colorbar()
    # plt.show()
    hdu.close()
    hdu_m.close()
    # collapsed_fluxes *= 1000.  # to bring to "regular" numbers for lucy process; will undo after lucy process

    # if write_name is not None:
    if not Path(write_name).exists():
        hdu = fits.PrimaryHDU(collapsed_fluxes)
        hdul = fits.HDUList([hdu])
        hdul.writeto(write_name)  # '/Users/jonathancohn/Documents/dyn_mod/' +
        hdul.close()
    return collapsed_fluxes, freq_axis, f_0, data, abs(f_step)


def blockshaped(arr, nrow, ncol):  # CONFIRMED
    """
    Function to use for rebinning data

    :param arr: array getting rebinned (2d)
    :param nrow: number of pixels in a row to get rebinned
    :param ncol: number of pixels in a column to get rebinned (same as nrow for n x n rebinning)
    :return: blocks of nrow x ncol subarrays from the input array
    """

    h, w = arr.shape
    return arr.reshape(h // nrow, nrow, -1, ncol).swapaxes(1, 2).reshape(-1, nrow, ncol)


def rebin(data, n):
    """
    Rebin some cube (data or model) in groups of n x n pixels

    :param data: input cube (data or model) (can now also be one slice of the cube, e.g. a 2darray mask)
    :param n: size of pixel binning (e.g. n=4 --> rebins the date in sets of 4x4 pixels)
    :return: rebinned cube
    """

    rebinned = []
    if len(data.shape) == 2:
        subarrays = blockshaped(data, n, n)  # bin the data in groups of nxn (4x4) pixels
        # each pixel in the new, rebinned data cube is the mean of each 4x4 set of original pixels
        # reshaped = np.mean(np.mean(subarrays, axis=-1), axis=-1).reshape((int(len(data[0]) / 4.),
        #                                                                   int(len(data[0][0]) / 4.)))
        reshaped = n**2 * np.mean(np.mean(subarrays, axis=-1), axis=-1).reshape((int(len(data) / n),
                                                                                 int(len(data[0]) / n)))
        rebinned.append(reshaped)
    else:
        for z in range(len(data)):
            subarrays = blockshaped(data[z, :, :], n, n)  # bin the data in groups of nxn (4x4) pixels
            # each pixel in the new, rebinned data cube is the mean of each 4x4 set of original pixels
            # reshaped = np.mean(np.mean(subarrays, axis=-1), axis=-1).reshape((int(len(data[0]) / 4.),
            #                                                                   int(len(data[0][0]) / 4.)))
            reshaped = n**2 * np.mean(np.mean(subarrays, axis=-1), axis=-1).reshape((int(len(data[0]) / n),
                                                                                     int(len(data[0][0]) / n)))
            rebinned.append(reshaped)
    # print('rebinned')
    return np.asarray(rebinned)


def ellipse_fitting(cube, x0_sub, y0_sub, res, pa_disk, inc):
    """
    Create an elliptical mask, within which we will do all of the actual fitting

    :param cube: down-sampled version of the sub-cube around the emission (model or data; using only for the dimensions)
    :param x0_sub: x-pixel location of BH, in coordinates of the sub-cube [pix]
    :param y0_sub: y-pixel location of BH, in coordinates of the sub-cube [pix]
    :param res: resolution of the pixel scale [arcsec/pix]
    :param pa_disk: position angle of the disk [radians]
    :param inc: inclination angle of the disk [deg]

    :return: masked ellipse array, to mask out everything in model cube outside of ellipse we want to fit
    """

    # Define the Fitting Ellipse
    a = 1. / res  # size of semimajor axis, in pixels
    b = (1. / res) * np.cos(np.deg2rad(inc))  # size of semiminor axis, in pixels
    # a = 34.28
    # b = 23.68
    ell = Ellipse2D(amplitude=1., x_0=x0_sub, y_0=y0_sub, a=a, b=b, theta=pa_disk)
    y_e, x_e = np.mgrid[0:len(cube[0]), 0:len(cube[0][0])]  # make sure this is the downsampled cube!

    # Select the regions of the ellipse we want to fit
    ellipse_mask = ell(x_e, y_e)

    return ellipse_mask


def check_ellipse(data, res, xo, yo, major, minor, theta):
    """
    Test x0,y0 point: is it within ellipse defined by major,minor axis and theta position angle?

    :param data: data (or model) cube
    :param res: resolution of grid [arcsec/pix]
    :param xo: pixel point (x) to check
    :param yo: pixel point (y) to check
    :param major: major axis of ellipse [arcsec]
    :param minor: minor axis of ellipse [arcsec]
    :param theta: position angle of ellipse [radians]

    :return: data cube with points outside the ellipse masked
    """

    tests = []
    mask = np.zeros(shape=(len(data[0][0]), len(data[0])))  # array of indices, corresponding to x,y data
    for i in range(len(data[0][0])):  # (x axis)
        for j in range(len(data[0])):  # (y axis)
            maj = major / res  # ellipse major axis
            mi = minor / res  # ellipse minor axis
            # NOTE: THE TEST_PT EQUATION BELOW IS FOR theta_rot MEASURED FROM +y, SO ADD +90deg TO theta
            # https://stackoverflow.com/questions/7946187/point-and-ellipse-rotated-position-test-algorithm
            theta_rot = theta + np.pi/2.  # + np.pi/4.
            test_pt = ((np.cos(theta_rot) * (i - (len(data[0][0])/2. + xo)) + np.sin(theta_rot) *
                        (j - (len(data[0])/2. + yo))) / maj) ** 2 \
                + ((np.sin(theta_rot) * (i - (len(data[0][0])/2. + xo)) - np.cos(theta_rot) *
                    (j - (len(data[0])/2. + yo))) / mi) ** 2
            tests.append(test_pt)
            mask[i, j] = test_pt

    import numpy.ma as ma
    data_masked = np.zeros(shape=data.shape)
    for k in range(len(data)):
        data_masked[k] = ma.masked_where(mask > 1., data[k])

    return data_masked


def model_grid(resolution=0.05, s=10, x_loc=0., y_loc=0., mbh=4 * 10 ** 8, inc=60., vsys=None, dist=17.,
               theta=np.deg2rad(-200.), data_cube=None, data_mask=None, lucy_output=None, out_name=None,
               enclosed_mass=None, ml_ratio=1., sig_type='flat', grid_size=31, sig_params=[1., 1., 1., 1.], f_w=1.,
               x_fwhm=0.052, y_fwhm=0.037, pa=64., menc_type=False, ds=None, lucy_in=None, lucy_b=None, lucy_mask=None,
               lucy_o=None, lucy_it=10, chi2=False, pb=None, zrange=None, xyrange=None, xyerr=None, mge_f=None):
    """
    Build grid for dynamical modeling!

    :param resolution: resolution of observations [arcsec/pixel]
    :param s: oversampling factor
    :param x_loc: the location of the BH, as measured along the x axis of the data cube [pixels]
    :param y_loc: the location of the BH, as measured along the y axis of the data cube [pixels]
    :param mbh: supermassive black hole mass [solar masses]
    :param inc: inclination of the galaxy [degrees]
    :param vsys: if given, the systemic velocity [km/s]
    :param dist: distance to the galaxy [Mpc]
    :param theta: angle from the +x_obs axis counterclockwise to the blueshifted side of the disk (-x_disk) [radians]
    :param data_cube: input data cube of observations
    :param data_mask: input mask cube of each slice of the data, for constructing the weight map
    :param lucy_output: output from running lucy on data cube and beam PSF
    :param out_name: output name of the fits file to which to save the output v_los image (if None, don't save image)
    :param enclosed_mass: file including data of the enclosed stellar mass of the galaxy (1st column should be radius
        r in kpc and second column should be M_stars(<r) or velocity_circ(R) due to stars within R)
    :param mge_f: file including MGE fit parameters that you can use to create v_circ(R) due to stars, from mge_vcirc.py
    :param ml_ratio: The mass-to-light ratio of the galaxy
    :param sig_type: code for the type of sigma_turb we're using. Can be 'flat', 'exp', or 'gauss'
    :param sig_params: list of parameters to be plugged into the get_sig() function. Number needed varies by sig_type
    :param f_w: multiplicative weight factor for the line profiles
    :param grid_size: the pixel grid size to use for the make_beam() function
    :param x_fwhm: FWHM in the x-direction of the ALMA beam (arcsec) to use for the make_beam() function
    :param y_fwhm: FWHM in the y-direction of the ALMA beam (arcsec) to use for the make_beam() function
    :param pa: position angle (in degrees) to use for the make_beam() function
    :param menc_type: Select how you incorporate enclosed stellar mass [True for mass(R) or False for velocity(R)]
    :param ds: downsampling factor to use when averaging pixels together for actual model-data comparison
    :param lucy_in: file name of input summed flux map to use for lucy process (if lucy_output doesn't exist)
    :param lucy_b: file name of input beam (built in make_beam function) to use for lucy process (if lucy_output doesn't
        exist)
    :param lucy_o: file name that will become the lucy_output, used in lucy process (if lucy_output doesn't exist)
    :param lucy_mask: file name of collapsed mask file to use in lucy process (if lucy_output doesn't exist)
    :param lucy_it: number of iterations to run in lucy process (if lucy_output doesn't exist)
    :param chi2: if True, record chi^2 of model vs data! If False, ignore, and return convolved model cube
    :param pb: the primary beam image, typically stored as, e.g., /product/NGC3258_CO_image.flux.fits
    :param zrange: the slices of the data cube where the data shows up (i.e. isn't just noise) [zi, zf]
    :param xyrange: the subset of the cube (in pixels) that we actually want to run data on [xi, xf, yi, yf]
    :param xyerr: the subset of the cube (in pixels) used for estimating the noise [xi, xf, yi, yf]

    :return: convolved model cube (if chi2 is False); chi2 (if chi2 is True)
    """
    ## BUCKET BUCKET: NEXT TO DO: INCORPORATE xyrange, AND TALK TO BEN SOME MORE ABOUT CHOOSING AN APPROPRIATE RANGE
    t_begin = time.time()
    # INSTANTIATE ASTRONOMICAL CONSTANTS
    constants = Constants()

    # MAKE ALMA BEAM  x_std=major; y_std=minor; rot=90-PA
    beam = make_beam(grid_size=grid_size, res=resolution, x_std=x_fwhm, y_std=y_fwhm, rot=np.deg2rad(90. - pa),
                     fits_name=lucy_b)  # this function now auto-creates the beam file lucy_b if it doesn't yet exist

    # COLLAPSE THE DATA CUBE
    fluxes, freq_ax, f_0, input_data, fstep = get_fluxes(data_cube, data_mask, write_name=lucy_in)
    # print(fluxes.shape)

    # DECONVOLVE FLUXES WITH BEAM PSF
    # to use pyraf, must be in the "three" environment ("source activate three" or "tres" on command line)
    # to use iraf, must ALSO "source activate iraf27" on command line
    if not Path(lucy_output).exists():
        t_pyraf = time.time()
        import pyraf
        from pyraf import iraf
        from iraf import stsdas, analysis, restore  # THIS WORKED!!!!
        restore.lucy(lucy_in, lucy_b, lucy_o, niter=lucy_it, maskin=lucy_mask, goodpixval=1, limchisq=1E-3)
        print('lucy process done in ' + str(time.time() - t_pyraf) + 's')  # ~10.6s
        if lucy_output is None:  # lucy_output should be defined, but just in case:
            lucy_output = lucy_o[:-3]  # don't include "[0]" that's required on the end for lucy

    hdu = fits.open(lucy_output)
    lucy_out = hdu[0].data
    hdu.close()

    # SUBPIXELS (RESHAPING DATA sxs)
    # take deconvolved flux map (output from lucy), assign each subpixel flux=(real pixel flux)/s**2 --> weight map
    print('start ')
    t0 = time.time()
    if s == 1:
        subpix_deconvolved = lucy_out
    else:
        # subpix_deconvolved is identical to lucy_out, just with sxs subpixels for each pixel & the total flux conserved
        subpix_deconvolved = np.zeros(shape=(len(lucy_out) * s, len(lucy_out[0]) * s))  # 300*s, 300*s
        for ypix in range(len(lucy_out)):
            for xpix in range(len(lucy_out[0])):
                subpix_deconvolved[(ypix * s):(ypix + 1) * s, (xpix * s):(xpix + 1) * s] = lucy_out[ypix, xpix] / s ** 2
    # print('deconvolution took {0} s'.format(time.time() - t0))  # ~0.3 s (7.5s for 1280x1280 array)

    # SET UP VELOCITY AXIS
    if vsys is None:
        v_sys = constants.H0 * dist
    else:
        v_sys = vsys

    # convert from frequency (Hz) to velocity (km/s), with freq_ax in Hz
    z_ax = np.asarray([v_sys + ((f_0 - freq) / freq) * (constants.c / constants.m_per_km) for freq in freq_ax])  # v_opt
    # z_ax = np.asarray([v_sys + ((f_0 - freq)/f_0) * (constants.c / constants.m_per_km) for freq in freq_ax])  # v_rad

    # RESCALE subpix_deconvolved, z_ax TO ONLY CONTAIN THE SUB-CUBE REGION WHERE EMISSION ACTUALLY EXISTS
    subpix_deconvolved = subpix_deconvolved[int(s * xyrange[2]):int(s * xyrange[3]),
                                            int(s * xyrange[0]):int(s * xyrange[1])]  # stored: y,x
    z_ax = z_ax[int(zrange[0]):int(zrange[1])]
    freq_ax = freq_ax[int(zrange[0]):int(zrange[1])]

    # MUST ALSO RESCALE x_loc, y_loc PIXEL VALUES TO CORRESPOND TO SUB-CUBE!
    x_loc = x_loc - xyrange[0]  # x_loc - xi
    y_loc = y_loc - xyrange[2]  # y_loc - yi

    # SET UP OBSERVATION AXES
    # initialize all values along axes at 0., with axes length set to equal input data cube * s
    y_obs = [0.] * len(subpix_deconvolved)
    x_obs = [0.] * len(subpix_deconvolved[0])

    # Define coordinates to be 0,0 at center of the observed axes (find the central pixel number along each axis)
    if len(x_obs) % 2. == 0:  # if even
        x_ctr = (len(x_obs)) / 2.  # set the center of the axes (in pixel number)
        for i in range(len(x_obs)):
            x_obs[i] = resolution * (i - x_ctr) / s  # (arcsec/pix) * N_subpixels / (subpixels/pix) = arcsec
    else:  # elif odd
        x_ctr = (len(x_obs) + 1.) / 2.  # +1 bc python starts counting at 0
        for i in range(len(x_obs)):
            x_obs[i] = resolution * ((i + 1) - x_ctr) / s
    # repeat for y-axis
    if len(y_obs) % 2. == 0:
        y_ctr = (len(y_obs)) / 2.
        for i in range(len(y_obs)):
            y_obs[i] = resolution * (i - y_ctr) / s
    else:
        y_ctr = (len(y_obs) + 1.) / 2.
        for i in range(len(y_obs)):
            y_obs[i] = resolution * ((i + 1) - y_ctr) / s

    # SET BH OFFSET from center [in arcsec], based on the input BH pixel position
    # x_loc,y_loc are in pixels, rather than subpixels (so DON'T divide locs by s)
    x_bhoff = (x_loc - x_ctr / s) * resolution  # (pix - subpix/(subpix/pix)) * (arcsec/pix) = arcsec
    y_bhoff = (y_loc - y_ctr / s) * resolution

    # CONVERT FROM ARCSEC TO PHYSICAL UNITS (pc)
    x_bhoff = dist * 10 ** 6 * np.tan(x_bhoff / constants.arcsec_per_rad)  # tan(off) = x_disk/dist --> x = d*tan(off)
    y_bhoff = dist * 10 ** 6 * np.tan(y_bhoff / constants.arcsec_per_rad)  # 206265 arcsec/rad
    # print('BH is at [pc]: ', x_bhoff, y_bhoff)

    # convert all x,y observed grid positions to pc
    x_obs = np.asarray([dist * 10 ** 6 * np.tan(x / constants.arcsec_per_rad) for x in x_obs])  # 206265 arcsec/rad
    y_obs = np.asarray([dist * 10 ** 6 * np.tan(y / constants.arcsec_per_rad) for y in y_obs])  # 206265 arcsec/rad

    # at each x,y spot in grid, calculate what x_disk and y_disk are, then calculate R, v, etc.
    # CONVERT FROM x_obs, y_obs TO x_disk, y_disk (still in pc)
    x_disk = (x_obs[None, :] - x_bhoff) * np.cos(theta) + (y_obs[:, None] - y_bhoff) * np.sin(theta)  # 2d array
    y_disk = (y_obs[:, None] - y_bhoff) * np.cos(theta) - (x_obs[None, :] - x_bhoff) * np.sin(theta)  # 2d array
    # print('x, y disk', x_disk.shape, y_disk.shape)

    # CALCULATE THE RADIUS (R) OF EACH POINT (x_disk, y_disk) IN THE DISK (pc)
    R = np.sqrt((y_disk ** 2 / np.cos(np.deg2rad(inc)) ** 2) + x_disk ** 2)  # radius R of each point in the disk (2d array)

    '''  #
    # PRINT PVD
    pvd(data_cube, theta, z_ax, x_obs, R, v_sys)  # x_obs = x in arcsec
    print(oops)
    # '''  #

    # CALCULATE ENCLOSED MASS BASED ON MBH AND ENCLOSED STELLAR MASS
    # THEN CALCULATE KEPLERIAN VELOCITY
    t_mass = time.time()
    '''  # WHEN I SWITCH TO NEW PARAM FILE FORMAT
    if menc_type == 0:  # if calculating v(R) due to stars directly from MGE parameters
        import sys
        sys.path.insert(0, '/Users/jonathancohn/Documents/jam/')  # lets me import file from different folder/path
        import mge_vcirc_mine as mvm

        comp, surf_pots, sigma_pots, qobs = mvm.load_mge(enclosed_mass)  # load the MGE parameters
        v_c = mvm.mge_vcirc(surf_pots * ml_ratio, sigma_pots, qobs, np.rad2deg(inc), 0., dist, R)  # v_circ due to stars

        # CALCULATE KEPLERIAN VELOCITY OF ANY POINT (x_disk, y_disk) IN THE DISK WITH RADIUS R (km/s)
        vel = np.sqrt((constants.G_pc * mbh / R) + v_c**2)
    elif menc_type == 1:  # elif using a file with stellar mass(R)
        radii = []
        v_circ = []
        with open(enclosed_mass) as em:  # note: current file has units v_circ^2/(M/L) --> v_circ = np.sqrt(col * (M/L))
            for line in em:
                cols = line.split()  # note: currently using model "B1" = 2nd col in file (file has 4 cols of models)
                radii.append(float(cols[0]))  # file lists radii in pc
                v_circ.append(float(cols[1]))  # v^2 / (M/L) --> units (km/s)^2 / (M_sol/L_sol)
        v_c_r = interpolate.interp1d(radii, v_circ, fill_value='extrapolate')  # create a function to interpolate v_circ

        # CALCULATE KEPLERIAN VELOCITY OF ANY POINT (x_disk, y_disk) IN THE DISK WITH RADIUS R (km/s)
        vel = np.sqrt(v_c_r(R) * ml_ratio + (constants.G_pc * mbh / R))  # velocities sum in quadrature
    elif menc_type == 2:  # elif using a file directly with v_circ as a function of R due to stellar mass
        radii = []
        m_stellar = []
        with open(enclosed_mass) as em:
            for line in em:
                cols = line.split()
                cols[1] = cols[1].replace('D', 'e')
                radii.append(float(cols[0]) * 10 ** 3)  # file lists radii in kpc; convert to pc
                m_stellar.append(float(cols[1]))  # solar masses
        m_star_r = interpolate.interp1d(radii, m_stellar, kind='cubic', fill_value='extrapolate')  # create a function
        ml_const = ml_ratio / 7.35  # because mass file assumes a mass-to-light ratio of 7.35
        m_R = mbh + ml_const * m_star_r(R)  # Use m_star_r function to interpolate mass at all radii R (2d array)

        # CALCULATE KEPLERIAN VELOCITY OF ANY POINT (x_disk, y_disk) IN THE DISK WITH RADIUS R (km/s)
        vel = np.sqrt(constants.G_pc * m_R / R)  # Keplerian velocity vel at each point in the disk
    # '''  # WHEN I SWITCH OT NEW PARAM FILE FORMAT
    if Path(mge_f).exists():  # if calculating v(R) due to stars directly from MGE parameters
        import sys
        sys.path.insert(0, '/Users/jonathancohn/Documents/jam/')  # lets me import file from different folder/path
        import mge_vcirc_mine as mvm

        comp, surf_pots, sigma_pots, qobs = mvm.load_mge(mge_f)  # load the MGE parameters
        # v_c = mvm.mge_vcirc(surf_pots * ml_ratio, sigma_pots, qobs, np.rad2deg(inc), 0., dist, R)  # v_circ due to stars
        # note: mge_vcirc currently breaks if R has more than one dimension, so:
        rads = np.logspace(-2, 10, 30)
        v_c = mvm.mge_vcirc(surf_pots * ml_ratio, sigma_pots, qobs, inc, 0., dist, rads)
        v_c_func = interpolate.interp1d(rads, v_c, fill_value='extrapolate')  # create a function to interpolate v_circ

        # CALCULATE KEPLERIAN VELOCITY OF ANY POINT (x_disk, y_disk) IN THE DISK WITH RADIUS R (km/s)
        # vel = np.sqrt((constants.G_pc * mbh / R) + v_c**2)
        # note: mge_vcirc currently breaks if R has more than one dimension, so:
        vel = np.sqrt(v_c_func(R)**2 + (constants.G_pc * mbh / R))  # velocities sum in quadrature
    elif menc_type:  # if using a file with stellar mass(R)
        radii = []
        m_stellar = []
        with open(enclosed_mass) as em:
            for line in em:
                cols = line.split()
                cols[1] = cols[1].replace('D', 'e')
                radii.append(float(cols[0]) * 10 ** 3)  # file lists radii in kpc; convert to pc
                m_stellar.append(float(cols[1]))  # solar masses
        m_star_r = interpolate.interp1d(radii, m_stellar, kind='cubic', fill_value='extrapolate')  # create a function
        ml_const = ml_ratio / 7.35  # because mass file assumes a mass-to-light ratio of 7.35
        m_R = mbh + ml_const * m_star_r(R)  # Use m_star_r function to interpolate mass at all radii R (2d array)

        # CALCULATE KEPLERIAN VELOCITY OF ANY POINT (x_disk, y_disk) IN THE DISK WITH RADIUS R (km/s)
        vel = np.sqrt(constants.G_pc * m_R / R)  # Keplerian velocity vel at each point in the disk
    else:  # elif using a file with v_circ as a function of R due to stellar mass
        radii = []
        v_circ = []
        with open(enclosed_mass) as em:  # note: current file has units v_circ^2/(M/L) --> v_circ = np.sqrt(col * (M/L))
            for line in em:
                cols = line.split()  # note: currently using model "B1" = 2nd col in file (file has 4 cols of models)
                radii.append(float(cols[0]))  # file lists radii in pc
                v_circ.append(float(cols[1]))  # v^2 / (M/L) --> units (km/s)^2 / (M_sol/L_sol)
        v_c_r = interpolate.interp1d(radii, v_circ, fill_value='extrapolate')  # create a function to interpolate v_circ

        # CALCULATE KEPLERIAN VELOCITY OF ANY POINT (x_disk, y_disk) IN THE DISK WITH RADIUS R (km/s)
        vel = np.sqrt(v_c_r(R) * ml_ratio + (constants.G_pc * mbh / R))  # velocities sum in quadrature
    # print('Time elapsed in assigning enclosed masses is {0} s'.format(time.time() - t_mass))  # ~3.5s

    # CALCULATE LINE-OF-SIGHT VELOCITY AT EACH POINT (x_disk, y_disk) IN THE DISK (km/s)
    alpha = abs(np.arctan(y_disk / (np.cos(np.deg2rad(inc)) * x_disk)))  # alpha meas. from +x (minor axis) toward +y (major axis)
    sign = x_disk / abs(x_disk)  # (+x now back to redshifted side, so don't need extra minus sign back in front!)
    v_los = sign * abs(vel * np.cos(alpha) * np.sin(np.deg2rad(inc)))  # THIS IS CURRENTLY CORRECT
    # print('los')

    # SET LINE-OF-SIGHT VELOCITY AT THE BLACK HOLE CENTER TO BE 0, SUCH THAT IT DOES NOT BLOW UP
    center = (R == 0.)  # Doing this is only relevant if we have pixel located exactly at the center
    v_los[center] = 0.  # if any point is at x_disk, y_disk = (0., 0.), set velocity there = 0.

    # CALCULATE OBSERVED VELOCITY
    v_obs = v_sys - v_los  # observed velocity v_obs at each point in the disk

    # CALCULATE VELOCITY PROFILES
    sigma = get_sig(r=R, sig0=sig_params[0], r0=sig_params[1], mu=sig_params[2], sig1=sig_params[3])[sig_type]

    zred = vsys / constants.c_kms

    # CONVERT v_los TO OBSERVED FREQUENCY MAP
    freq_obs = (f_0 / (1+zred)) * (1 - v_los / constants.c_kms)

    '''  #
    plt.imshow(v_los, origin='lower', cmap='RdBu_r', vmax=np.amax([np.amax(v_los), -np.amin(v_los)]),
               vmin=-np.amax([np.amax(v_los), -np.amin(v_los)]))
    cbar = plt.colorbar()
    cbar.set_label(r'km/s', fontsize=20, rotation=0, labelpad=20)
    plt.show()
    plt.imshow(freq_obs, origin='lower', vmax=np.amax(freq_obs), vmin=np.amin(freq_obs), cmap='viridis',
               extent=[np.amin(x_obs), np.amax(x_obs), np.amin(y_obs), np.amax(y_obs)])
    cbar = plt.colorbar()
    cbar.set_label(r'Hz', fontsize=20, rotation=0, labelpad=20)
    plt.show()
    # '''  #

    # CONVERT OBSERVED DISPERSION (turbulent) TO FREQUENCY WIDTH
    sigma_grid = np.zeros(shape=(v_los.shape)) + sigma
    delta_freq_obs = (f_0 / (1 + zred)) * (sigma_grid / constants.c_kms)

    # WEIGHTS FOR LINE PROFILES: apply weights to gaussian velocity profiles for each subpixel
    weight = subpix_deconvolved  # / 1000.  # dividing by 1000 bc multiplying map by 1000 earlier  [Jy/beam Hz]

    # WEIGHT CURRENTLY IN UNITS OF Jy/beam * Hz --> need to get it in units of Jy/beam to match data
    weight *= f_w / np.sqrt(2 * np.pi * delta_freq_obs**2)  # divide to get correct units

    # BEN_LUCY COMPARISON ONLY (only use for comparing to model with Ben's lucy map, which is in different units)
    weight *= fstep * 6.783  # (undo dividing by 1000 above, bc test isn't using the flux part where I do that [no longer doing])

    # BUILD GAUSSIAN LINE PROFILES!!!
    t_mod = time.time()
    cube_model = np.zeros(shape=(len(freq_ax), len(freq_obs), len(freq_obs[0])))  # initialize model cube
    for fr in range(len(freq_ax)):
        # print(fr)
        cube_model[fr] = weight * np.exp(-(freq_ax[fr] - freq_obs) ** 2 / (2 * delta_freq_obs ** 2))
    # print('cube model constructed in ' + str(time.time() - t_mod) + ' s')  # 22s

    # RE-SAMPLE BACK TO CORRECT PIXEL SCALE (take average of sxs sub-pixels for real alma pixel) --> intrinsic data cube
    t_z = time.time()
    if s == 1:
        intrinsic_cube = cube_model
    else:
        intrinsic_cube = rebin(cube_model, s)
        # intrinsic_cube = block_reduce(cube_model, s, np.mean)
    # print("intrinsic cube done in {0} s".format(time.time() - t_z))
    # print("start to intrinsic done in {0} s".format(time.time() - t0))

    # CONVERT INTRINSIC TO OBSERVED
    # take velocity slice from intrinsic data cube, convolve with alma beam --> observed data cube
    convolved_cube = np.zeros(shape=intrinsic_cube.shape)  # approx ~1e-6 to 3e-6s per pixel
    ts = time.time()
    for z in range(len(z_ax)):
        # print(z)
        # tl = time.time()
        convolved_cube[z, :, :] = convolution.convolve(intrinsic_cube[z, :, :], beam)  # CONFIRMED!
        # print("Convolution loop " + str(z) + " took {0} seconds".format(time.time() - tl))  # 0.03s/loop for 100x100pix
    # print('convolved! Total convolution loop took {0} seconds'.format(time.time() - ts))  # 170.9s
    print('total model constructed in {0} seconds'.format(time.time() - t_begin))  # ~213s

    # ONLY WANT TO FIT WITHIN ELLIPTICAL REGION! APPLY ELLIPTICAL MASK
    ell_mask = ellipse_fitting(convolved_cube, x_loc, y_loc, resolution, theta, inc)
    # hdu = fits.PrimaryHDU(ell_mask)
    # hdul = fits.HDUList([hdu])
    # hdul.writeto('/Users/jonathancohn/Documents/dyn_mod/outputs/NGC_3258_fitting_ellipse.fits')
    convolved_cube *= ell_mask
    input_data_masked = input_data[int(zrange[0]):int(zrange[1]), int(xyrange[2]):int(xyrange[3]),
                        int(xyrange[0]):int(xyrange[1])] * ell_mask
    # '''  #
    if chi2:
        chi_sq = 0.  # initialize chisq
        # https://en.wikipedia.org/wiki/Reduced_chi-squared_statistic
        # https://en.wikipedia.org/wiki/Variance
        # https://en.wikipedia.org/wiki/Standard_deviation

        # compare the data to the model by binning each in groups of dsxds pixels (separate from s)
        data_4 = rebin(input_data_masked, ds)
        ap_4 = rebin(convolved_cube, ds)

        '''  #
        inds_to_try2 = np.asarray([[10, 10], [10, 15], [15, 10]])
        for i in range(len(inds_to_try2)):
            print(inds_to_try2[i][0], inds_to_try2[i][1])
            plt.plot(z_ax, ap_4[:, inds_to_try2[i][1], inds_to_try2[i][0]], 'r-',
                     label=r'Astropy conv')  # ap_out
            plt.plot(z_ax, data_4[:, inds_to_try2[i][1], inds_to_try2[i][0]], 'b:', label=r'Data')  # data_in
            plt.axvline(x=v_sys, color='k')
            plt.title(str(inds_to_try2[i][0]) + ', ' + str(inds_to_try2[i][1]))  # ('no x,y offset')
            plt.legend()
            plt.show()
        # '''  #

        # MODEL NOISE!
        # BUCKET: ADJUST SO NOISE IS ONLY CALCULATED ONCE IN EMCEE PROCESS, NOT EACH ITERATION
        ell_4 = rebin(ell_mask, ds)
        noise = []
        nums = []
        cs = []
        z_ind = 0  # the actual index for the model-data comparison cubes
        for z in range(int(zrange[0]), int(zrange[1])):  # for each relevant freq slice (ignore slices with only noise)
            # ESTIMATE NOISE (RMS) IN ORIGINAL DATA CUBE [z, y, x]
            # plt.imshow((ap_4[z_ind] - data_4[z_ind])**2, origin='lower')
            # plt.colorbar()
            # plt.show()
            # noise.append(np.sqrt(np.mean(input_data[z, int(xyerr[2]):int(xyerr[3]), int(xyerr[0]):int(xyerr[1])] ** 2)))

            # noise2.append(np.sqrt(np.mean(input_data[z, 390:440, 370:420] ** 2)))  # 260:360, 210:310
            # noise2 trying to do proper variance. For large N, Variance ~= std^2!
            noise.append(np.std(input_data[z, int(xyerr[2]):int(xyerr[3]), int(xyerr[0]):int(xyerr[1])]))
            # noise2.append(np.sqrt(np.mean((input_data[z, int(xyerr[2]):int(xyerr[3]), int(xyerr[0]):int(xyerr[1])]
            #               - np.mean(input_data[z, int(xyerr[2]):int(xyerr[3]), int(xyerr[0]):int(xyerr[1])]))**2)))

            nums.append(np.sum((ap_4[z_ind] - data_4[z_ind])**2))
            chi_sq += np.sum((ap_4[z_ind] - data_4[z_ind])**2 / noise[z_ind]**2)  # calculate chisq!
            cs.append(np.sum((ap_4[z_ind] - data_4[z_ind])**2 / noise[z_ind]**2))

            z_ind += 1  # the actual index for the model-data comparison cubes

        # CALCULATE REDUCED CHI^2
        all_pix = np.ndarray.flatten(ell_4)  # all fitted pixels in each slice [len = 625 (yep)] [525 masked, 100 not]
        masked_pix = all_pix[all_pix != 0]  # all_pix, but this time only the pixels that are actually inside ellipse
        n_pts = len(masked_pix) * (zrange[1] - zrange[0])  # total number of pixels being compared = 4600 [100*46]
        n_params = 12  # number of free parameters
        print(r'Supposedly reduced chi^2=', chi_sq / (n_pts - n_params))  # 4284.80414208  # 12300204.6088
        '''  #
        # plt.plot(freq_ax/1e9, np.asarray(cs) / len(masked_pix), 'ro', label=r'$\chi^2$')
        plt.plot(freq_ax / 1e9, np.asarray(noise)**2, 'k*', label=r'Variance')
        plt.plot(freq_ax / 1e9, nums, 'b+', label=r'$\chi^2$ Numerator')
        plt.legend(loc='center right')
        plt.yscale('log')
        plt.xlabel(r'GHz')
        plt.show()
        plt.plot(freq_ax, noise, 'k*')
        plt.show()
        print(oops)
        # '''  #

        return chi_sq
    else:
        # WRITE OUT RESULTS TO FITS FILE
        if not Path(out_name).exists():
            hdu = fits.PrimaryHDU(convolved_cube)
            hdul = fits.HDUList([hdu])
            hdul.writeto(out_name)
            print('written!')

        return convolved_cube


def par_dicts2(parfile, q=False):
    """
    Return dictionaries that contain file names, parameter names, and initial guesses, for free and fixed parameters

    :param parfile: the parameter file
    :return: params (the free parameters, fixed parameters, and input files), priors (prior boundaries as {'param_name':
        [min, max]} dictionaries)
    """

    params = {}
    with open(parfile, 'r') as pf:
        for line in pf:
            if not line.startswith('#'):
                cols = line.split()
                if cols[0] == 'free':
                    params[cols[1]] = float(cols[2])
                    priors[cols[1]] = [float(cols[3]), float(cols[4])]
                elif cols[0] == 'float':
                    params[cols[1]] = float(cols[2])
                elif cols[0] == 'int':
                    params[cols[1]] = int(cols[2])
                elif cols[0] == 'str':
                    params[cols[1]] = cols[2]

    if q:
        import sys
        sys.path.insert(0, '/Users/jonathancohn/Documents/jam/')  # lets me import file from different folder/path
        import mge_vcirc_mine as mvm
        comp, surf_pots, sigma_pots, qobs = mvm.load_mge(params['mass'])

        return params, priors, qobs

    else:
        return params, priors


def par_dicts(parfile, q=False):
    """
    Return dictionaries that contain file names, parameter names, and initial guesses, for free and fixed parameters

    :param parfile: the parameter file
    :return: params (the free parameters), fixed_pars (fixed parameters), files (file names), and priors (prior
        boundaries as {'param_name': [min, max]} dictionaries)
    """

    params = {}
    fixed_pars = {}
    files = {}
    priors = {}

    # READ IN PARAMS FORM THE PARAMETER FILE
    with open(parfile, 'r') as pf:
        for line in pf:
            if line.startswith('Pa'):
                par_names = line.split()[1:]  # ignore the "Param" str in the first column
            elif line.startswith('Primax'):
                primax = line.split()[1:]
            elif line.startswith('Primin'):
                primin = line.split()[1:]
            elif line.startswith('V'):
                par_vals = line.split()[1:]
            elif line.startswith('Other_p'):
                fixed_names = line.split()[1:]
            elif line.startswith('Other_v'):
                fixed_vals = line.split()[1:]
            elif line.startswith('T'):
                file_types = line.split()[1:]
            elif line.startswith('F'):
                file_names = line.split()[1:]

    for n in range(len(par_names)):
        params[par_names[n]] = float(par_vals[n])
        priors[par_names[n]] = [float(primin[n]), float(primax[n])]

    for n in range(len(fixed_names)):
        if fixed_names[n] == 's_type' or fixed_names[n] == 'mtype':
            fixed_pars[fixed_names[n]] = fixed_vals[n]
        elif fixed_names[n] == 'gsize' or fixed_names[n] == 's':
            fixed_pars[fixed_names[n]] = int(fixed_vals[n])
        else:
            fixed_pars[fixed_names[n]] = float(fixed_vals[n])

    for n in range(len(file_types)):
        files[file_types[n]] = file_names[n]

    if q:
        import sys
        sys.path.insert(0, '/Users/jonathancohn/Documents/jam/')  # lets me import file from different folder/path
        import mge_vcirc_mine as mvm
        comp, surf_pots, sigma_pots, qobs = mvm.load_mge(files['mge'])

        return params, fixed_pars, files, priors, qobs
    else:
        return params, fixed_pars, files, priors


if __name__ == "__main__":
    # MAKE SURE I HAVE ACTIVATED THE iraf27 ENVIRONMENT!!!
    t0_true = time.time()
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument('--parfile')

    args = vars(parser.parse_args())
    print(args['parfile'])

    params, fixed_pars, files, priors = par_dicts(args['parfile'])
    # print(params)
    # print(fixed_pars)
    # print(files)

    # Make nice plot fonts
    rc('font', **{'family': 'serif', 'serif': ['Times']})
    rc('text', usetex=True)

    # CREATE OUTNAME BASED ON INPUT PARS
    pars_str = ''
    for key in params:
        pars_str += str(params[key]) + '_'
    out = '/Users/jonathancohn/Documents/dyn_mod/outputs/NGC_3258_general_' + pars_str + '_subcube_ellmask_bl2_noell.fits'

    # If the lucy process hasn't been done yet, and the mask cube also hasn't been collapsed yet, create collapsed mask
    if not Path(files['lucy']).exists() and not Path(files['lucy_mask']).exists():
        hdu_m = fits.open(files['mask'])
        fullmask = hdu_m[0].data  # The mask is stored in hdu_m[0].data, NOT hdu_m[0].data[0]
        print(fullmask.shape)
        collapsed_mask = integrate.simps(fullmask, axis=0)
        for i in range(len(collapsed_mask)):
            for j in range(len(collapsed_mask[0])):
                if collapsed_mask[i, j] != 0.:
                    collapsed_mask[i, j] = 1.
        hdu1 = fits.PrimaryHDU(collapsed_mask)
        hdul1 = fits.HDUList([hdu1])
        hdul1.writeto(files['lucy_mask'])

    # CREATE MODEL CUBE!
    chisqs = []
    for sig0 in [4.6, 4.75, 4.9, 5., 101.]:  # sig_params=[params['sig0'],
    # for vsys in [1900., 2000., 2200., 2700., 2760.76, 3000.]:  # vsys=params['vsys']
    # for mbh in [2.25e9, 2.3e9, 2.35e9, 2.4e9, 2.45e9]:  # mbh=params['mbh']
        out_cube = model_grid(resolution=fixed_pars['resolution'], s=fixed_pars['s'], x_loc=params['xloc'],
                              y_loc=params['yloc'], mbh=params['mbh'], inc=params['inc'], vsys=params['vsys'],
                              dist=fixed_pars['dist'], theta=np.deg2rad(params['PAdisk']), data_cube=files['data'],
                              data_mask=files['mask'], lucy_output=files['lucy'], out_name=out, ml_ratio=params['ml_ratio'],
                              mge_f=files['mge'], enclosed_mass=files['mass'], menc_type=fixed_pars['mtype']==True,
                              sig_type=fixed_pars['s_type'], grid_size=fixed_pars['gsize'], x_fwhm=fixed_pars['x_fwhm'],
                              y_fwhm=fixed_pars['y_fwhm'], pa=fixed_pars['PAbeam'], ds=int(fixed_pars['ds']),
                              sig_params=[sig0, params['r0'], params['mu'], params['sig1']],
                              f_w=params['f'], lucy_in=files['lucy_in'], lucy_b=files['lucy_b'], lucy_o=files['lucy_o'],
                              lucy_mask=files['lucy_mask'], lucy_it=fixed_pars['lucy_it'], chi2=True, pb=files['pb'],
                              zrange=[int(fixed_pars['zi']), int(fixed_pars['zf'])],
                              xyrange=[int(fixed_pars['xi']), int(fixed_pars['xf']), int(fixed_pars['yi']),
                                       int(fixed_pars['yf'])], xyerr=[int(fixed_pars['xerr0']), int(fixed_pars['xerr1']),
                                                                      int(fixed_pars['yerr0']), int(fixed_pars['yerr1'])])
        chisqs.append(out_cube)
    print('True Total: ' + str(time.time() - t0_true))  # 3.93s YAY!  # 23s for 6 models (16.6 for 4 models)
    print(chisqs)


    '''
    params, priors = par_dicts(args['parfile'])
    out_cube = model_grid(resolution=pars['resolution'], s=pars['s'], x_loc=pars['xloc'], y_loc=pars['yloc'],
                          mbh=pars['mbh'], inc=pars['inc'], vsys=pars['vsys'], dist=pars['dist'],
                          theta=np.deg2rad(pars['PAdisk']), data_cube=pars['data'], data_mask=pars['mask'], chi2=True,
                          lucy_output=pars['lucy'], out_name=out, ml_ratio=pars['ml_ratio'], grid_size=pars['gsize'],
                          enclosed_mass=pars['mass'], menc_type=pars['mtype'], sig_type=pars['s_type'], f_w=pars['f'],
                          x_fwhm=pars['x_fwhm'], y_fwhm=pars['y_fwhm'], pa=pars['PAbeam'], lucy_it=pars['lucy_it'],
                          sig_params=[pars['sig0'], pars['r0'], pars['mu'], pars['sig1']], lucy_mask=pars['lucy_mask'],
                          lucy_in=pars['lucy_in'], lucy_b=pars['lucy_b'], lucy_o=pars['lucy_o'], ds=pars['ds'],
                          zrange=[pars['zi'], pars['zf']], xyrange=[pars['xi'], pars['xf'], pars['yi'], pars['yf']],
                          xyerr=[pars['xerr0'], pars['xerr1'], pars['yerr0'], pars['yerr1']])
    
    chi2 = dg.model_grid(
        # FREE PARAMETERS (entered as list of params because these change each iteration)
        x_loc=params[pars.keys().index('xloc')],
        y_loc=params[pars.keys().index('yloc')],
        mbh=params[pars.keys().index('mbh')],
        inc=params[pars.keys().index('inc')],
        vsys=params[pars.keys().index('vsys')],
        theta=np.deg2rad(params[pars.keys().index('PAdisk')]),
        ml_ratio=params[pars.keys().index('ml_ratio')],
        sig_params=[params[pars.keys().index('sig0')], params[pars.keys().index('r0')],
                    params[pars.keys().index('mu')], params[pars.keys().index('sig1')]],
        f_w=params[pars.keys().index('f')],
        # FIXED PARAMETERS
        sig_type=pars['s_type'], menc_type=pars['mtype'], ds=pars['ds'], grid_size=pars['gsize'], s=pars['s'],
        x_fwhm=pars['x_fwhm'], y_fwhm=pars['y_fwhm'], pa=pars['PAbeam'], dist=pars['dist'], lucy_it=pars['lucy_it'], 
        resolution=pars['resolution'], xyrange=[pars['xi'], pars['xf'], pars['yi'], pars['yf']],
        zrange=[pars['zi'], pars['zf']], xyerr=[pars['xerr0'], pars['xerr1'], pars['yerr0'], pars['yerr1']],
        # FILES
        enclosed_mass=pars['mass'], lucy_in=pars['lucy_in'], lucy_output=pars['lucy'], lucy_b=pars['lucy_b'],
        lucy_o=pars['lucy_o'], lucy_mask=pars['lucy_mask'], data_cube=pars['data'], data_mask=pars['mask'],
        # OTHER PARAMETERS
        out_name=None, chi2=True)
    '''

# xf=410,yi=320;xi=310,yi=400 --> want divisible by 4: so...good!

# TAKE TRUNCATED MODEL AND PUT BACK IN ZEROS FULL SIZE MODEL CUBE TO COMPARE TO DATA
