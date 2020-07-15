import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import patches
from mpl_toolkits.axes_grid1 import AxesGrid
from scipy import integrate, signal, interpolate, misc
from scipy.ndimage import filters, interpolation
import time
from astropy import convolution
import argparse
from pathlib import Path
from astropy.modeling.models import Box2D
from astropy.nddata.utils import block_reduce
from astropy.modeling.models import Ellipse2D
import sys
# sys.path.insert(0, '/Users/jonathancohn/Documents/jam/')  # lets me import file from different folder/path
import mge_vcirc_mine as mvm  # import mge_vcirc code
#from idlpy import *


def innerint(rvals, avals, sigr3, del_r, h=0):

    int1_a3 = np.zeros(shape=len(rvals))
    for i in range(1, len(rvals)):  # for i=1.,n_elements(rvals)-1 (BC IDL INDEXING INCLUSIVE!)
        int1_a3[i] = np.sum(rvals[i:] * sigr3[i:] * del_r / np.sqrt(rvals[i:] ** 2 - (avals[i - 1] - h) ** 2))

    return int1_a3


def integral22(rad, dda):

    int22 = integrate.quad(integrand22, 0, rad, args=(rad, dda))[0]

    return int22


def doubl_grand(R, sig_func, incl, f_convert, h=1e-5):


    f_dg = lambda a, r: 1 / np.sqrt(R**2 - a**2)

    int_result = integrate.dblquad(f_dg, a, np.inf, lambda r: 0, lambda r: R)

    return int_result


def integrand22(a, rad, dda):

    #if rad == a:
    #    integ22 = 0
    #else:
    integ22 = a * dda / np.sqrt(rad**2 - a**2)

    return integ22


def integral2(rad, rmax, sigma_func, inclination, conversion_factor):

    int2 = integrate.quad(integrand2, 0, rad, args=(rad, rmax, sigma_func, inclination, conversion_factor))[0]
    print(int2, 'int2 done')

    return int2


def integrand2(a, rad, rmax, sigma_func, inclination, conversion_factor):

    # print('integrand 2 get ready')
    # da = 0.1
    #integ2 = misc.derivative(integral1, a, dx=da, args=(sigma_func, inclination, conversion_factor)) * a\
    #         / np.sqrt(rad[-1]**2 - a**2)
    integ2 = deriv(a, rmax, sigma_func, inclination, conversion_factor, h=1e-5) * a / np.sqrt(rad**2 - a**2)
    # integ2 = 186.82929873466492 * a / np.sqrt(rad[-1]**2 - a**2)
    # print('integrand 2 built', integ2)

    return integ2


def deriv(a, rmax, sigma_func, inclination, conversion_factor, h=1e-5):
    # h is included for easy manual derivation

    deriv = (integral1(a+h, rmax, sigma_func, inclination, conversion_factor) -
             integral1(a, rmax, sigma_func, inclination, conversion_factor)) / h

    return deriv


def integral1(a, rmax, sigma_func, inclination, conversion_factor):

    #print('pre inner integral')
    # replacing np.inf with rmax
    int1 = integrate.quad(integrand1, a, rmax, args=(sigma_func, a, inclination, conversion_factor))[0]
    # print('inner int calculated')
    # print(int1)

    return int1


def integrand1(r, sigma_func, a, inclination, conversion_factor):

    #print('integrand inner get ready')
    integ1 = r * sigma_func(r) * np.cos(inclination) * conversion_factor / np.sqrt(r ** 2 - a ** 2)
    #print('integrand inner built')

    return integ1


def par_dicts(parfile, q=False):
    """
    Return dictionaries that contain file names, parameter names, and initial guesses, for free and fixed parameters

    :param parfile: the parameter file
    :param q: True if I'm using mge vcirc
    :return: params (the free parameters, fixed parameters, and input files), priors (prior boundaries as {'param_name':
        [min, max]} dictionaries), qobs (mge parameter qobs; only included if q=True)
    """

    params = {}
    priors = {}
    nfree = 0
    with open(parfile, 'r') as pf:
        for line in pf:
            if not line.startswith('#'):
                cols = line.split()
                if len(cols) > 0:  # ignore empty lines
                    if cols[0] == 'free':  # free parameters are floats with priors
                        nfree += 1
                        params[cols[1]] = float(cols[2])
                        priors[cols[1]] = [float(cols[3]), float(cols[4])]
                    elif cols[0] == 'float':  # fixed floats
                        params[cols[1]] = float(cols[2])
                    elif cols[0] == 'int':  # fixed ints
                        params[cols[1]] = int(cols[2])
                    elif cols[0] == 'str':  # fixed strings
                        params[cols[1]] = cols[2]

    if q:
        import sys
        sys.path.insert(0, '/Users/jonathancohn/Documents/jam/')  # lets me import file from different folder/path
        import mge_vcirc_mine as mvm
        comp, surf_pots, sigma_pots, qobs = mvm.load_mge(params['mass'])

        return params, priors, nfree, qobs

    else:
        return params, priors, nfree


def model_prep(lucy_out=None, lucy_mask=None, lucy_b=None, lucy_in=None, lucy_it=None, data=None, data_mask=None,
               grid_size=None, res=1., x_std=1., y_std=1., pa=0., ds=4, zrange=None, xyerr=None, theta_ell=0, q_ell=0,
               xell=0, yell=0):
    """

    :param lucy_out: output from running lucy on data cube and beam PSF; if it doesn't exist, create it!
    :param lucy_mask: file name of collapsed mask file to use in lucy process (if lucy_out doesn't exist)
    :param lucy_b: file name of input beam (built in make_beam()) to use for lucy process (if lucy_out doesn't exist)
    :param lucy_in: file name of input summed flux map to use for lucy process (if lucy_out doesn't exist)
    :param lucy_it: number of iterations to run in lucy process (if lucy_out doesn't exist)
    :param data: input data cube of observations
    :param data_mask: input mask cube of each slice of the data, for constructing the weight map
    :param grid_size: the pixel grid size to use for the make_beam() function
    :param res: resolution of observations [arcsec/pixel]
    :param x_std: FWHM in the x-direction of the ALMA beam (arcsec) to use for the make_beam() function
    :param y_std: FWHM in the y-direction of the ALMA beam (arcsec) to use for the make_beam() function
    :param pa: position angle (in degrees) to use for the make_beam() function
    :param ds: down-sampling factor for pixel scale used to calculate chi^2
    :param zrange: range of frequency slices containing emission [zi, zf]
    :param xyerr: x & y pixel region, on the down-sampled pixel scale, where the noise is calculated [xi, xf, yi, yf]
    :param theta_ell: position angle of the annuli for the gas mass, same as for the ellipse fitting region [radians]
    :param q_ell: axis ratio q of the annuli for the gas mass, same as for the ellipse fitting region [unitless]
    :param xell: x center of elliptical annuli for the gas mass, same as for the ellipse fitting region [pixels]
    :param yell: y center of elliptical annuli for the gas mass, same as for the ellipse fitting region [pixels]

    :return: lucy mask, lucy output, synthesized beam, flux map, frequency axis, f_0, freq step, input data cube
    """

    # If the lucy process hasn't been done yet, and the mask cube also hasn't been collapsed yet, create collapsed mask
    hdu_m = fits.open(data_mask)
    fullmask = hdu_m[0].data  # The mask is stored in hdu_m[0].data, NOT hdu_m[0].data[0]
    print(fullmask.shape)
    hdu_m.close()
    if not Path(lucy_out).exists() and not Path(lucy_mask).exists():
        collapsed_mask = integrate.simps(fullmask, axis=0)   # integrate the mask, collapsing to 2D
        for i in range(len(collapsed_mask)):
            for j in range(len(collapsed_mask[0])):
                if collapsed_mask[i, j] != 0.:
                    collapsed_mask[i, j] = 1.  # make it a mask: if data exist in given pix, pix=1; else, pix=0
        hdu1 = fits.PrimaryHDU(collapsed_mask)
        hdul1 = fits.HDUList([hdu1])
        hdul1.writeto(lucy_mask)  # write out to mask file
        print('mask created!')

    # MAKE ALMA BEAM  x_std=major; y_std=minor; rot=90-PA; auto-create the beam file lucy_b if it doesn't yet exist
    beam = make_beam(grid_size=grid_size, res=res, x_std=x_std, y_std=y_std, rot=np.deg2rad(90. - pa), fits_name=lucy_b)

    # COLLAPSE THE DATA CUBE
    fluxes, freq_ax, f_0, input_data, fstep = get_fluxes(data, data_mask, write_name=lucy_in)

    # DECONVOLVE FLUXES WITH BEAM PSF
    if not Path(lucy_out).exists():  # to use iraf, must "source activate iraf27" on command line
        t_pyraf = time.time()
        from skimage import restoration  # need to be in "three" environment (source activate three == tres)
        hduin = fits.open(lucy_in)
        l_in = hduin[0].data
        hdub = fits.open(lucy_b)
        l_b = hdub[0].data
        hduin.close()
        hdub.close()
        l_in[l_in < 0.] = 0.
        # https://github.com/scikit-image/scikit-image/issues/2551 (it was returning nans)
        # open skimage/restoration/deconvolution.py
        # find this line (~line 389): im_deconv *= convolve_method(relative_blur, psf_mirror, 'same')
        # Insert a new line before that line: relative_blur[np.isnan(relative_blur)] = 0
        l_o = restoration.richardson_lucy(l_in, l_b, lucy_it, clip=False)  # need clip=False
        print('lucy process done in ' + str(time.time() - t_pyraf) + 's')  # ~1s

        hdu1 = fits.PrimaryHDU(l_o)
        hdul1 = fits.HDUList([hdu1])
        hdul1.writeto(lucy_out)  # write out to lucy_out file
        print(lucy_out)

    hdu = fits.open(lucy_out)
    lucy_out = hdu[0].data
    hdu.close()

    # ESTIMATE NOISE (RMS) IN ORIGINAL DATA CUBE [z, y, x]
    #print(input_data.shape)
    #fig, ax = plt.subplots(1)
    #ax.imshow(fluxes, origin='lower')
    #rect = patches.Rectangle((xyerr[0], xyerr[2]), xyerr[1] - xyerr[0], xyerr[3] - xyerr[2], linewidth=2, edgecolor='w',
    #                         facecolor='none')
    #ax.add_patch(rect)
    #plt.show()
    #print(oop)
    noise_4 = rebin(input_data, ds)  # rebin the noise to the pixel scale on which chi^2 will be calculated
    noise = []  # For large N, Variance ~= std^2
    for z in range(zrange[0], zrange[1]):  # for each relevant freq slice
        # noise.append(np.std(noise_4[z, xyerr[2]:xyerr[3], xyerr[0]:xyerr[1]]))  # ~variance
        noise.append(np.std(noise_4[z, int(xyerr[2]/ds):int(xyerr[3]/ds), int(xyerr[0]/ds):int(xyerr[1]/ds)]))

    #print(noise)
    #print(oop)
    #plt.plot(freq_ax[zrange[0]:zrange[1]] / 1e9, noise, 'k+')#, label='How my code is currently set up')
    #plt.xlabel('GHz')
    #plt.ylabel('std [Jy/beam]')
    #plt.legend()
    #plt.show()
    #print(oop)

    # CALCULATE FLUX MAP FOR GAS MASS ESTIMATE
    # CALCULATE VELOCITY WIDTH  # vsys = 6454.9 estimated based on various test runs; see Week of 2020-05-04 on wiki
    v_width = 2.99792458e5 * (1 + (6454.9 / 2.99792458e5)) * fstep / f_0  # velocity width [km/s] = c*(1+v/c)*fstep/f0

    # CONSTRUCT THE FLUX MAP IN UNITS Jy km/s beam^-1
    collapsed_fluxes_vel = np.zeros(shape=(len(input_data[0]), len(input_data[0][0])))
    for zi in range(len(input_data)):
        collapsed_fluxes_vel += input_data[zi] * fullmask[zi] * v_width
    # collapsed_fluxes_vel[collapsed_fluxes_vel < 0] = 0.  # ignore negative values? probably not?

    # DEFINE SEMI-MAJOR AXES FOR SAMPLING, THEN CALCULATE THE MEAN SURFACE BRIGHTNESS INSIDE ELLIPTICAL ANNULI
    semi_major = np.linspace(0., 100., num=85)  # [pix]
    co_surf, co_errs = annuli_sb(collapsed_fluxes_vel, semi_major, theta_ell, q_ell, xell, yell)  # CO [Jy km/s beam^-1]
    co_ell_sb = np.asarray(co_surf)  # put the output list of mean CO surface brightnesses in an array
    co_ell_rad = (2. * semi_major[1:] + q_ell * semi_major[1:]) / 3.  # mean_ellipse_radius = (2a + b)/3
    # semi_major[1:] because 1 less annulus than ellipse

    return lucy_mask, lucy_out, beam, fluxes, freq_ax, f_0, fstep, input_data, noise, co_ell_sb, co_ell_rad


def make_beam(grid_size=99, res=1., amp=1., x0=0., y0=0., x_std=1., y_std=1., rot=0., fits_name=None):
    """
    Generate a beam spread function for the ALMA beam

    :param grid_size: size of grid (must be odd!)
    :param res: resolution of the grid (arcsec/pixel)
    :param amp: amplitude of the 2d gaussian
    :param x0: mean of x axis of 2d gaussian
    :param y0: mean of y axis of 2d gaussian
    :param x_std: FWHM of beam in x [beam major axis] (to use for standard deviation of Gaussian) (in arcsec)
    :param y_std: FWHM of beam in y [beam minor axis] (to use for standard deviation of Gaussian) (in arcsec)
    :param rot: rotation angle in radians
    :param fits_name: this name will be the filename to which the beam fits file is written (if None, write no file)

    :return the synthesized beam
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
    if not Path(fits_name).exists():
        hdu = fits.PrimaryHDU(synth_beam)
        hdul = fits.HDUList([hdu])
        hdul.writeto(fits_name)

    return synth_beam


def get_sig(r=None, sig0=1., r0=1., mu=1., sig1=0.):
    """
    :param r: 2D array of radius values = r(x, y) [length units]
    :param sig0: uniform sigma_turb component [velocity units]
    :param r0: expectation of the distribution; i.e. scale radius value [same units as r]
    :param mu: standard deviation of the distribution [same units as r]
    :param sig1: sigma offset [same units as sig0]
    :return: dictionary of the three different sigmas
    """

    sigma = {'flat': sig0, 'gauss': sig1 + sig0 * np.exp(-(r - r0) ** 2 / (2 * mu ** 2)),
             'exp': sig1 + sig0 * np.exp(-r / r0)}

    return sigma


def get_fluxes(data_cube, data_mask, write_name=None):
    """
    Integrate over the frequency axis of a data cube to get a flux map!

    :param data_cube: input data cube of observations
    :param data_mask: mask for each slice of data, for construction of the flux map
    :param write_name: name of fits file to which to write the flux map (only writes if write_name does not yet exist)

    :return: collapsed data cube (i.e. flux map), len(z_ax), intrinsic freq of observed line, input data cube
    """
    hdu = fits.open(data_cube)
    data = hdu[0].data[0]  # data[0] contains: z, y, x (121, 700, 700)

    hdu_m = fits.open(data_mask)
    mask = hdu_m[0].data  # this is hdu_m[0].data, NOT hdu[0].data[0], unlike the data_cube above

    z_len = len(hdu[0].data[0])  # store the number of velocity slices in the data cube
    freq1 = float(hdu[0].header['CRVAL3'])  # starting frequency in the data cube
    f_step = float(hdu[0].header['CDELT3'])  # frequency step in the data cube  # note: fstep is negative for NGC_3258
    f_0 = float(hdu[0].header['RESTFRQ'])
    freq_axis = np.arange(freq1, freq1 + (z_len * f_step), f_step)  # [bluest, ..., reddest]
    # NOTE: For NGC1332, this includes endpoint (arange shouldn't). However, if cut endpoint at fstep-1, arange doesn't
    # include it...So, for 1332, include the extra point above, then cutting it off: freq_axis = freq_axis[:-1]
    # NOTE: NGC_3258 DATA DOES *NOT* HAVE THIS ISSUE, SOOOOOOOOOO COMMENT OUT FOR NOW!

    # Collapse the fluxes! Sum over all slices, multiplying each slice by the slice's mask and by the frequency step
    collapsed_fluxes = np.zeros(shape=(len(data[0]), len(data[0][0])))
    for zi in range(len(data)):
        collapsed_fluxes += data[zi] * mask[zi] * abs(f_step)
        #     velwidth = self.c_kms * (1 + self.zred) * self.fstep / self.f_0

    hdu.close()  # close data
    hdu_m.close()  # close mask

    collapsed_fluxes[collapsed_fluxes < 0] = 0.

    if not Path(write_name).exists():
        hdu = fits.PrimaryHDU(collapsed_fluxes)
        hdul = fits.HDUList([hdu])
        hdul.writeto(write_name)  # '/Users/jonathancohn/Documents/dyn_mod/' +
        hdul.close()
    return collapsed_fluxes, freq_axis, f_0, data, abs(f_step)


def blockshaped(arr, nrow, ncol):
    """
    Function to use for rebinning data

    :param arr: the array getting rebinned (2d)
    :param nrow: number of pixels in a row to get rebinned
    :param ncol: number of pixels in a column to get rebinned (same as nrow for n x n rebinning)
    :return: blocks of nrow x ncol subarrays from the input array
    """

    h, w = arr.shape
    return arr.reshape(h // nrow, nrow, -1, ncol).swapaxes(1, 2).reshape(-1, nrow, ncol)


def rebin(data, n):
    """
    Rebin data or model cube (or one slice of a cube) in blocks of n x n pixels

    :param data: input data cube, model cube, or slice of a cube, e.g. a 2Darray
    :param n: size of pixel binning (e.g. n=4 rebins the date in blocks of 4x4 pixels)
    :return: rebinned cube or slice
    """

    rebinned = []
    if len(data.shape) == 2:
        subarrays = blockshaped(data, n, n)  # bin the data in groups of nxn (4x4) pixels
        # each pixel in the new, rebinned data cube is the mean of each 4x4 set of original pixels
        reshaped = n**2 * np.mean(np.mean(subarrays, axis=-1), axis=-1).reshape((int(len(data) / n),
                                                                                 int(len(data[0]) / n)))
        rebinned.append(reshaped)
    else:
        for z in range(len(data)):  # for each slice
            subarrays = blockshaped(data[z, :, :], n, n)  # bin the data as above
            reshaped = n**2 * np.mean(np.mean(subarrays, axis=-1), axis=-1).reshape((int(len(data[0]) / n),
                                                                                     int(len(data[0][0]) / n)))
            rebinned.append(reshaped)

    return np.asarray(rebinned)


def ellipse_fitting(cube, rfit, x0_sub, y0_sub, res, pa_disk, q):
    """
    Create an elliptical mask, within which we will do the actual fitting

    :param cube: sub-cube, cut to x,y,z regions where emission exists in data (model or data; using only for dimensions)
    :param rfit: the radius of the disk region to be fit [arcsec]
    :param x0_sub: x-pixel location of BH, in coordinates of the sub-cube [pix]
    :param y0_sub: y-pixel location of BH, in coordinates of the sub-cube [pix]
    :param res: resolution of the pixel scale [arcsec/pix]
    :param pa_disk: position angle of the disk [radians]
    :param q: axis ratio of the ellipse = cos(disk_inc) [unitless]

    :return: masked ellipse array, to mask out everything in model cube outside of the ellipse we want to fit
    """

    # Define the Fitting Ellipse
    a = rfit / res  # size of semimajor axis, in pixels
    b = (rfit / res) * q  # size of semiminor axis, in pixels

    ell = Ellipse2D(amplitude=1., x_0=x0_sub, y_0=y0_sub, a=a, b=b, theta=pa_disk)  # create elliptical region
    y_e, x_e = np.mgrid[0:len(cube[0]), 0:len(cube[0][0])]  # make sure this grid is the size of the downsampled cube!

    # Select the regions of the ellipse we want to fit
    ellipse_mask = ell(x_e, y_e)

    return ellipse_mask


def annuli_sb(flux_map, semi_major_axes, position_angle, axis_ratio, x_center, y_center, incl_gas=False):
    """
    Create an array of elliptical annuli, in which to calculate the mean CO surface brightness

    :param flux_map: CO surface brightness flux map, collapsed with the velocity width [Jy km/s beam^-1]
    :param semi_major_axes: the semi-major axes of the elliptical annuli to put on the flux map [pix]
    :param position_angle: position angle of the disk [radians]
    :param axis_ratio: axis ratio of the ellipse = cos(disk_inc) [unit-less]
    :param x_center: x-pixel location of BH, in coordinates of the flux_map [pix]
    :param y_center: y-pixel location of BH, in coordinates of the flux_map [pix]

    :return: array of mean CO flux calculated in each elliptical annulus on the flux map
    """
    ellipses = []

    for sma in semi_major_axes:
        # create elliptical region
        # print(sma, x_center, y_center, sma * axis_ratio, position_angle, len(flux_map), len(flux_map[0]))
        ell = Ellipse2D(amplitude=1., x_0=x_center, y_0=y_center, a=sma, b=sma * axis_ratio, theta=position_angle)
        y_e, x_e = np.mgrid[0:len(flux_map), 0:len(flux_map[0])]  # this grid is the size of the flux_map!

        # Select the regions of the ellipse we want to fit
        ellipses.append(ell(x_e, y_e))
        #plt.imshow(ell(x_e, y_e), origin='lower')
        #plt.colorbar()
        #plt.show()
    annuli = []
    average_co = []
    errs_co = np.zeros(shape=(2, len(ellipses) - 1))
    for e in range(len(ellipses) - 1):  # because indexed to e+1
        annuli.append(ellipses[e+1] - ellipses[e])
        npix = np.sum(annuli[e])
        annulus_flux = annuli[e] * flux_map
        #plt.imshow(ellipses[e] + ellipses[e+1], origin='lower')
        #plt.colorbar()
        #plt.show()
        annulus_flux[annulus_flux == 0] = np.nan
        average_co.append(np.nanmean(annulus_flux))
        # average_co.append(np.sum(annulus_flux) / npix)  # not just np.mean(annulus_flux) bc includes 0s!
        errs_co[:, e] = np.nanpercentile(annulus_flux, [16., 84.]) / npix  # 68% confidence interval
        # print(errs_co[:, e])
        #plt.imshow(annulus_flux, origin='lower')
        #plt.colorbar()
        #plt.show()

    return average_co, errs_co


class ModelGrid:

    def __init__(self, resolution=0.05, os=4, x_loc=0., y_loc=0., mbh=4e8, inc=np.deg2rad(60.), vsys=None, vrad=0.,
                 kappa=0., omega=0., dist=17., theta=np.deg2rad(200.), input_data=None, lucy_out=None, vtype='orig',
                 out_name=None, beam=None, rfit=1., q_ell=1., theta_ell=0., xell=360., yell=350., bl=False,
                 enclosed_mass=None, menc_type=0, ml_ratio=1., sig_type='flat', sig_params=None, f_w=1., noise=None,
                 ds=None, zrange=None, xyrange=None, reduced=False, freq_ax=None, f_0=0., fstep=0., opt=True,
                 quiet=False, n_params=8, data_mask=None, f_he=1.36, r21=0.7, alpha_co10=3.1, incl_gas=False,
                 co_rad=None, co_sb=None, gas_norm=1e5, gas_radius=5, z_fixed=0.02152, pvd_width=None, kin_file=None):
        # Astronomical Constants:
        self.c = 2.99792458 * 10 ** 8  # [m / s]
        self.pc = 3.086 * 10 ** 16  # [m / pc]
        self.G = 6.67 * 10 ** -11  # [kg^-1 * m^3 * s^-2]
        self.M_sol = 1.989 * 10 ** 30  # [kg / solar mass]
        self.H0 = 70  # [km/s/Mpc]
        self.arcsec_per_rad = 206265.  # [arcsec / radian]
        self.m_per_km = 10. ** 3  # [m / km]
        self.G_pc = self.G * self.M_sol * (1. / self.pc) / self.m_per_km ** 2  # G [Msol^-1 * pc * km^2 * s^-2] (gross)
        self.c_kms = self.c / self.m_per_km  # [km / s]
        # Input Parameters
        self.z_fixed = z_fixed  # fixed redshift, used to transform the luminosity dist to the angular diameter dist
        self.resolution = resolution  # resolution of observations [arcsec/pixel]
        self.os = os  # oversampling factor
        self.x_loc = x_loc  # the location of the BH, as measured along the x axis of the data cube [pixels]
        self.y_loc = y_loc  # the location of the BH, as measured along the y axis of the data cube [pixels]
        self.mbh = mbh  # supermassive black hole mass [solar masses]
        self.inc = inc  # inclination of the galaxy [radians]
        self.vsys = vsys  # the systemic velocity [km/s]
        self.vrad = vrad  # optional radial inflow term [km/s]
        self.kappa = kappa  # optional radial inflow term; tie inflow to the overall line-of-sight velocity [unitless]
        self.omega = omega  # optional velocity coefficient, used with kappa for radial inflow [unitless]
        self.vtype = vtype  # 'vrad', 'kappa', 'omega', any other value for original (no radial velocity component)
        self.dist = dist  # angular diameter distance to the galaxy [Mpc]
        self.pc_per_ac = self.dist * 1e6 / self.arcsec_per_rad  # small angle formula (convert dist to pc, from Mpc)
        self.pc_per_pix = self.dist * 1e6 / self.arcsec_per_rad * self.resolution  # small angle formula, as above
        self.pc_per_sp = self.pc_per_pix / self.os  # pc per subpixel (over-sampling pc pixel scale)
        self.theta = theta  # angle from +x_obs axis counterclockwise to the disk's blueshifted side (-x_disk) [radians]
        self.zred = self.vsys / self.c_kms  # redshift
        self.input_data = input_data  # input 3D data cube of observations
        self.data_mask = data_mask  # 3D strictmask cube, used to mask the data
        self.lucy_out = lucy_out  # deconvolved 2D fluxmap, output from running lucy-richardson on fluxmap and beam PSF
        self.out_name = out_name  # optional; output name of the fits file to which to save the convolved model cube
        self.beam = beam  # synthesized alma beam (output from model_ins)
        self.rfit = rfit  # disk radius (elliptical semi-major axis) within which we compare the model and data [arcsec]
        self.q_ell = q_ell  # axis ratio q of fitting ellipse [unitless]
        self.theta_ell = theta_ell  # same as theta, but held fixed; used for the ellipse fitting region [radians]
        self.xell = xell  # same as x_loc, but held fixed; used for the ellipse fitting region [pixels]
        self.yell = yell  # same as y_loc, but held fixed; used for the ellipse fitting region [pixels]
        self.bl = bl  # lucy weight map unit indicator (bl=False or 0 --> Jy/beam * Hz; bl=True or 1 --> Jy/beam * km/s)
        self.enclosed_mass = enclosed_mass  # MGE file name, or similar file containing M(R) information [file]
        self.menc_type = menc_type  # Determines how stellar mass is included [0->MGE; 1->v(R); 2->M(R)]
        self.ml_ratio = ml_ratio  # The mass-to-light ratio of the galaxy [Msol / Lsol]
        self.sig_type = sig_type  # Str determining what type of sigma to use. Can be 'flat', 'exp', or 'gauss'
        self.sig_params = sig_params  # array of sigma parameters: [sig0, r0, mu, sig1]
        self.f_w = f_w  # multiplicative weight factor for the line profiles [unitless]
        self.noise = noise  # array of the estimated (ds x ds-binned) noise per slice (within the zrange) [Jy/beam]
        self.ds = ds  # downsampling factor to use when averaging pixels together for actual model-data comparison [int]
        self.zrange = zrange  # array with the slices of the data cube where real emission shows up [zi, zf]
        self.xyrange = xyrange  # array with the subset of the cube (in pixels) that contains emission [xi, xf, yi, yf]
        self.reduced = reduced  # if True, ModelGrid.chi2() returns the reduced chi^2 instead of the regular chi^2
        self.freq_ax = freq_ax  # array of the frequency axis in the data cube, from bluest to reddest frequency [Hz]
        self.f_0 = f_0  # rest frequency of the observed line in the data cube [Hz]
        self.fstep = fstep  # frequency step in the frequency axis [Hz]
        self.opt = opt  # frequency axis velocity convention; opt=True -> optical; opt=False -> radio
        self.quiet = quiet  # if quiet=True, suppress printing out stuff!
        self.n_params = n_params  # number of free parameters being fit, as counted from the param file in par_dicts()
        self.co_rad = co_rad  # mean elliptical radii of annuli used in calculation of co_sb [pix]
        self.co_sb = co_sb  # mean CO surface brightness in elliptical annuli [Jy km/s beam^-1]
        self.f_he = f_he  # additional fraction of gas that is helium (f_he = 1 + helium mass fraction)
        self.r21 = r21  # CO(2-1)/CO(1-0) SB ratio (see pg 6-8 in Boizelle+17)
        self.alpha_co10 = alpha_co10  # CO(1-0) to H2 conversion factor (see pg 6-8 in Boizelle+17)
        self.incl_gas = incl_gas  # if True, include gas mass in calculations
        self.gas_norm = gas_norm  # best-fit exponential coefficient for gas mass calculation [Msol/pix^2]
        self.gas_radius = gas_radius  # best-fit scale radius for gas-mass calculation [pix]
        self.pvd_width = pvd_width  # width (in pixels) for the PVD extraction
        self.kin_file = kin_file  # filename to save info for kinemetry input into IDL later
        # Parameters to be built in create_grid(), convolve_cube(), or chi2 functions inside the class
        self.z_ax = None  # velocity axis, constructed from freq_ax, f_0, and vsys, based on opt
        self.xobs = None
        self.yobs = None
        self.weight = None  # 2D weight map, constructed from lucy_output (the deconvolved fluxmap)
        self.freq_obs = None  # the 2D line-of-sight velocity map, converted to frequency
        self.delta_freq_obs = None  # the 2D turbulent velocity map, converted to delta-frequency
        self.clipped_data = None  # the data sub-cube that we compare to the model, clipped by zrange and xyrange
        self.convolved_cube = None  # the model cube: create from convolving the intrinsic model cube with the ALMA beam
        self.ell_mask = None  # mask defined by the elliptical fitting region, before downsampling
        self.ell_ds = None  # mask defined by the elliptical fitting region, created on ds x ds down-sampled pixels
    """
    Build grid for dynamical modeling!
    
    Class structure following: https://www.w3schools.com/python/python_classes.asp

    Class functions:
        grids: calculates weight map, freq_obs map, and delta_freq_obs map
        gas_mass: if incl_gas, calculate velocity due to gas mass
        convolution: grids must be run first; create the intrinsic model cube and convolve it with the ALMA beam
        chi2: convolution must be run first; calculates chi^2 and/or reduced chi^2
        output_cube: convolution must be run first; store model cube in a fits file
        test_ellipse: grids must be run first; use to check how the fitting-ellipse looks with respect to the weight map
        moment_0: convolution must be run first; create 0th moment map
        moment_12: convolution must be run first; create 1st or 2nd moment map    
    """

    def grids(self):
        t_grid = time.time()

        # SUBPIXELS (reshape deconvolved flux map [lucy_out] sxs subpixels, so subpix has flux=(real pixel flux)/s**2)
        if not self.quiet:
            print('start')
        if self.os == 1:  # subpix_deconvolved == lucy_out, but with sxs subpixels per pixel & total flux conserved
            subpix_deconvolved = self.lucy_out
        else:
            subpix_deconvolved = np.zeros(shape=(len(self.lucy_out) * self.os, len(self.lucy_out[0]) * self.os))
            for ypix in range(len(self.lucy_out)):
                for xpix in range(len(self.lucy_out[0])):
                    subpix_deconvolved[(ypix * self.os):(ypix + 1) * self.os, (xpix * self.os):(xpix + 1) * self.os] = \
                        self.lucy_out[ypix, xpix] / self.os ** 2

        # convert from frequency (Hz) to velocity (km/s), with freq_ax in Hz
        if self.opt:  # optical convention
            z_ax = np.asarray([self.vsys + ((self.f_0 - freq) / freq) * (self.c / self.m_per_km) for freq in
                               self.freq_ax])
        else:  # radio convention
            z_ax = np.asarray([self.vsys + ((self.f_0 - freq) / self.f_0) * (self.c / self.m_per_km) for freq in
                               self.freq_ax])

        # RESCALE subpix_deconvolved, z_ax, freq_ax TO CONTAIN ONLY THE SUB-CUBE REGION WHERE EMISSION ACTUALLY EXISTS
        subpix_deconvolved = subpix_deconvolved[self.os * self.xyrange[2]:self.os * self.xyrange[3],
                             self.os * self.xyrange[0]:self.os * self.xyrange[1]]  # stored: y,x
        self.z_ax = z_ax[self.zrange[0]:self.zrange[1]]
        self.freq_ax = self.freq_ax[self.zrange[0]:self.zrange[1]]

        # RESCALE (x_loc, y_loc) AND (xell, yell) PIXEL VALUES TO CORRESPOND TO SUB-CUBE PIXEL LOCATIONS!
        x_loc = self.x_loc - self.xyrange[0]  # x_loc - xi
        y_loc = self.y_loc - self.xyrange[2]  # y_loc - yi

        self.xell = self.xell - self.xyrange[0]
        self.yell = self.yell - self.xyrange[2]

        # SET UP OBSERVATION AXES: initialize x,y axes at 0., with lengths = os * len(input data cube axes)
        y_obs_ac = np.asarray([0.] * len(subpix_deconvolved))
        x_obs_ac = np.asarray([0.] * len(subpix_deconvolved[0]))

        # Define coordinates to be 0,0 at center of the observed axes (find the central pixel number along each axis)
        if len(x_obs_ac) % 2. == 0:  # if even
            x_ctr = (len(x_obs_ac)) / 2.  # set the center of the axes (in pixel number)
            for i in range(len(x_obs_ac)):
                x_obs_ac[i] = self.resolution * (i - x_ctr) / self.os  # (arcsec/pix) * N_subpix / (subpix/pix) = arcsec
        else:  # elif odd
            x_ctr = (len(x_obs_ac) + 1.) / 2.  # +1 bc python starts counting at 0
            for i in range(len(x_obs_ac)):
                x_obs_ac[i] = self.resolution * ((i + 1) - x_ctr) / self.os
        if len(y_obs_ac) % 2. == 0:
            y_ctr = (len(y_obs_ac)) / 2.
            for i in range(len(y_obs_ac)):
                y_obs_ac[i] = self.resolution * (i - y_ctr) / self.os
        else:
            y_ctr = (len(y_obs_ac) + 1.) / 2.
            for i in range(len(y_obs_ac)):
                y_obs_ac[i] = self.resolution * ((i + 1) - y_ctr) / self.os

        # SET BH OFFSET from center [in arcsec], based on input BH pixel position (*_loc in pixels; *_ctr in subpixels)
        x_bh_ac = (x_loc - x_ctr / self.os) * self.resolution  # [pix - subpix/(subpix/pix)] * [arcsec/pix] = arcsec
        y_bh_ac = (y_loc - y_ctr / self.os) * self.resolution

        # CONVERT FROM ARCSEC TO PHYSICAL UNITS (pc)
        x_bhoff = self.dist * 10 ** 6 * np.tan(x_bh_ac / self.arcsec_per_rad)  # tan(off) = xdisk/dist -> x = d*tan(off)
        y_bhoff = self.dist * 10 ** 6 * np.tan(y_bh_ac / self.arcsec_per_rad)  # 206265 arcsec/rad

        # convert all x,y observed grid positions to pc
        x_obs = np.asarray([self.dist * 10 ** 6 * np.tan(x / self.arcsec_per_rad) for x in x_obs_ac])
        y_obs = np.asarray([self.dist * 10 ** 6 * np.tan(y / self.arcsec_per_rad) for y in y_obs_ac])

        self.xobs = x_obs_ac
        self.yobs = y_obs_ac

        # CONVERT FROM x_obs, y_obs TO 2D ARRAYS OF x_disk, y_disk [arcsec] and [pc] versions
        x_disk_ac = (x_obs_ac[None, :] - x_bh_ac) * np.cos(self.theta) +\
                    (y_obs_ac[:, None] - y_bh_ac) * np.sin(self.theta)  # arcsec
        y_disk_ac = (y_obs_ac[:, None] - y_bh_ac) * np.cos(self.theta) -\
                    (x_obs_ac[None, :] - x_bh_ac) * np.sin(self.theta)  # arcsec
        x_disk = (x_obs[None, :] - x_bhoff) * np.cos(self.theta) + (y_obs[:, None] - y_bhoff) * np.sin(self.theta)  # pc
        y_disk = (y_obs[:, None] - y_bhoff) * np.cos(self.theta) - (x_obs[None, :] - x_bhoff) * np.sin(self.theta)  # pc

        # CALCULATE THE RADIUS (R) OF EACH POINT (x_disk, y_disk) IN THE DISK [pc]
        R_ac = np.sqrt((y_disk_ac ** 2 / np.cos(self.inc) ** 2) + x_disk_ac ** 2)  # radius R [arcsec]
        R = np.sqrt((y_disk ** 2 / np.cos(self.inc) ** 2) + x_disk ** 2)  # radius of each pt in disk (2d array) [pc]

        # CALCULATE KEPLERIAN VELOCITY DUE TO ENCLOSED STELLAR MASS
        vg = 0  # default to ignoring the gas mass!
        if self.incl_gas:  # If incl_mass, overwrite vg with a gas mass estimate, then add it in quadrature to velocity!
            t_gas = time.time()  # FANTSTIC THIS ONLY ADDS ~0.015 seconds! DEPENDS ON len(rvals); using R, takes ~300s
            # pix per beam = 2pi sigx sigy [pix^2]
            pix_per_beam = 2. * np.pi * (0.197045 / self.resolution / 2.35482) * (0.103544 / self.resolution / 2.35482)
            pc2_per_beam = pix_per_beam * self.pc_per_pix**2  # pc^2 per beam = pix/beam * pc^2/pix

            co_annuli_radii = self.co_rad * self.pc_per_pix  # convert input annulus mean radii from pix to pc
            co_annuli_sb = np.nan_to_num(self.co_sb)  # replace NaNs with 0s in input CO mean surface brightness profile
            print('cosb done')
            # Set up integration bounds, numerical step size, vectors r & a (see Binney & Tremaine eqn 2.157)
            # maxr >> than maximum CO radius, s.t. relative gravitational potential contributions are small compared to
            # those near the disk edge.
            min_r = 0.  # integration lower bound [pix or pc]
            max_r = 1300.  # upper bound [pc]; disk peak <~100pc, extend <~700pc; maxr >2x max CO radius  # 510 for edge
            nr = 500  # 500  # number of steps used in integration process
            del_r = (max_r - min_r) / nr  # integration step size [pc]
            avals = np.linspace(min_r,max_r,nr)  # [pc]  # range(min_r,max_r,(max_r-min_r)/del_r)
            rvals = np.linspace(min_r,max_r,nr)  # [pc]  # range(min_r,max_r,(max_r-min_r)/del_r)

            # convert from Jy km/s to Msol (Boizelle+17; Carilli & Walter 13, S2.4: https://arxiv.org/pdf/1301.0371.pdf)
            msol_per_jykms = 3.25e7 * self.alpha_co10 * self.f_he * self.dist ** 2 / \
                             ((1 + self.zred) * self.r21 * (self.f_0/1e9) ** 2)  # f_0 in GHz, not Hz?!
            # equation for (1+z)^3 is for observed freq, but using rest freq -> nu0^2 = (nu*(1+z))^2
            # units on 3.25e7 are [K Jy^-1 pc^2/Mpc^2 GHz^2] --> total units [Msol (Jy km/s)^-1]

            # Fit the CO distribution w/ an exp profile (w/ scale radius & norm), then construct Sigma(R) for R=rvals
            # CASE (2)
            # radius_pc = self.gas_radius * self.pc_per_pix  # convert free parameter from [pix] to [pc]
            # gas_norm_pc = self.gas_norm / self.pc_per_pix ** 2  # convert [Jy km/s / pix] to [Jy km/s / pc^2]
            # sigr2 = gas_norm_pc * np.cos(self.inc) * msol_per_jykms * np.exp(-rvals / radius_pc)  # [Msol pc^-2]

            # CASE (3)
            # Interpolate CO surface brightness vs elliptical mean radii, to construct Sigma(rvals).
            # Units [Jy km/s/beam] * [Msol/(Jy km/s)] / [pc^2/beam] = [Msol/pc^2]

            # Test setting inner-most annulus to 0 (no discernible change in final v_c,gas)
            #co_annuli_sb = np.insert(co_annuli_sb, 0, 0)
            #co_annuli_radii = np.insert(co_annuli_radii, 0, 0)

            sigr3_func_r = interpolate.interp1d(co_annuli_radii, co_annuli_sb, kind='quadratic', fill_value='extrapolate')
            sigr3 = sigr3_func_r(rvals) * np.cos(self.inc) * msol_per_jykms / pc2_per_beam  # Msol pc^-2
            plt.plot(co_annuli_radii, co_annuli_sb * np.cos(self.inc) * msol_per_jykms / pc2_per_beam, 'ro',
                     label='CO flux map')
            plt.plot(rvals, sigr3, 'b+', label='Interpolation')
            plt.ylabel(r'Surface density [M$_{\odot} / $pc$^2$]')
            plt.xlabel(r'Mean elliptical radius [pc]')
            plt.legend()
            plt.show()

            # ESTIMATE GAS MASS
            #i1 = integrate.quad(sigr3_func_r, 0, rvals[-1])[0]
            #print(i1)
            #i1 *= np.cos(self.inc) * msol_per_jykms  # convert to Msol/pc^2
            #print(np.log10(i1))  # 8.71174327436427
            # END ESTIMATE GAS MASS

            # BUCKET TESTING PYTHON INTEGRATION
            # '''  #
            print('testing python integration')
            #int2 = integrate.quad(deriv, 0, rvals[-1], args=(rvals, sigr3_func_r, self.inc, msol_per_jykms))
            #print(int2)
            #print(oop)
            # int2 = integrate.quad(integrand2, 0, rvals[-1], args=(rvals, sigr3_func_r, self.inc, msol_per_jykms))
            # grand = integrand2(rvals[-1], rvals, sigr3_func_r, self.inc, msol_per_jykms)
            #print(int2, 'grand')
            #td = time.time()
            #dda = misc.derivative(integral1, rvals[-1], dx=0.001, args=(sigr3_func_r, self.inc, msol_per_jykms))
            int1 = np.zeros(shape=len(avals))
            int1h = np.zeros(shape=len(avals))
            hh = 1e-5
            for av in range(len(avals)):
                int1[av] = integral1(avals[av], np.inf, sigr3_func_r, self.inc, msol_per_jykms / pc2_per_beam)
                int1h[av] = integral1(avals[av] - hh, np.inf, sigr3_func_r, self.inc, msol_per_jykms / pc2_per_beam)
                # dda.append(deriv(rv, np.inf, sigr3_func_r, self.inc, msol_per_jykms, 1e-5))
            dda = (int1 - int1h) / hh
            print(dda)
            zerocut = 580  # 530
            dda[rvals > zerocut] = 0
            # dda[dda > 0] = 0.
            # dda[np.abs(dda) > 1e6] = 0.

            from scipy.interpolate import UnivariateSpline as unsp
            int1[rvals > zerocut] = 0.
            interp_int1 = unsp(rvals, int1)
            # hint that smoothing factor needs to be big: https://stackoverflow.com/questions/8719754/scipy-interpolate-univariatespline-not-smoothing-regardless-of-parameters
            interp_int1.set_smoothing_factor(9e8)  # 1e8  # 1e9 smoothed
            intintr = interp_int1(rvals)
            # intintr[rvals > 530] = 0.
            #plt.plot(rvals, int1, 'k+')
            #plt.plot(rvals, intintr, 'r-')
            #plt.show()
            dda_sp = interp_int1.derivative()
            dda_sp_r = dda_sp(rvals)
            # dda_sp_r[rvals > 530] = 0
            #plt.plot(dda, 'ko')
            #plt.plot(dda_sp_r, 'r+')
            #plt.show()

            # print(oop)
            #dda = deriv(rvals[-1], sigr3_func_r, self.inc, msol_per_jykms, 1e-5)  # if use h too small, deriv explodes
            #print(dda, 'hi')
            #dda = deriv(rvals[0], sigr3_func_r, self.inc, msol_per_jykms, 1e-5)  # if use h too small, deriv explodes
            #print(dda, 'hey')
            # print(oop)
            #print(time.time() - td)
            #td2 = time.time()
            #dda0 = deriv(rvals[0], sigr3_func_r, self.inc, msol_per_jykms, 1e-5)
            #print(time.time() - td2)
            #print(dda, dda0)
            #print(oop)
            int2 = np.zeros(shape=len(rvals))
            int2_before = np.zeros(shape=len(rvals))
            for rv in range(len(rvals[1:])):
                #trv = time.time()
                #dda = deriv(rvals[rv], sigr3_func_r, self.inc, msol_per_jykms, 1e-5)
                int2[rv] = integral22(rvals[rv], dda_sp_r[rv])
                int2_before[rv] = integral22(rvals[rv], dda[rv])
                #print(rvals[rv], time.time() - trv)
                # deriv(rv, sigr3_func_r, self.inc, msol_per_jykms, 1e-5) * rv / np.sqrt(rvals[-1]**2 - rv**2)
            #    int2[rv] = integral2(rv, sigr3_func_r, self.inc, msol_per_jykms)
            print(int2, 'int2')
            #plt.plot(int2_before, 'ko')
            #plt.plot(int2, 'r+')
            #plt.show()
            #print(oop)
            #dda = integral1(rvals[-1], sigr3_func_r, self.inc, msol_per_jykms)
            # print(dda, 'deriv')
            #int2 = int22(integrand22, 0, rvals[-1], args=(rvals, dda, sigr3_func_r, self.inc, msol_per_jykms))
            #int1 = integral1(rvals[-1], sigr3_func_r, self.inc, msol_per_jykms)
            #print(int1, 'look')  # could maybe go to a higher rvals value than 1500, but it relatively levels off here
            # print(oop)
            #integral_2 = integral2(rvals, sigr3_func_r, self.inc, msol_per_jykms)
            #print(integral_2)
            # int2 = 186.82929873466492 * np.asarray(rvals)
            vcg = np.sqrt(-4 * self.G_pc * int2)
            vcg = np.nan_to_num(vcg)
            print(vcg)
            vcg_func = interpolate.interp1d(rvals, vcg, kind='quadratic', fill_value='extrapolate')
            vcgr = vcg_func(R)
            alpha = abs(np.arctan(y_disk / (np.cos(self.inc) * x_disk)))  # measure alpha from +x (minor ax) to +y (maj ax)
            sign = x_disk / abs(x_disk)  # (+x now back to redshifted side, so don't need extra minus sign back in front!)
            #vcgr = sign * abs(vcgr * np.cos(alpha) * np.sin(self.inc))  # v_los > 0 -> redshift; v_los < 0 -> blueshift
            plt.imshow(vcgr, origin='lower', extent=[x_obs[0], x_obs[-1], y_obs[0], y_obs[-1]])  # , cmap='RdBu_r')  #, vmin=-50, vmax=50)  #
            cbar = plt.colorbar()
            cbar.set_label(r'km/s')
            plt.xlabel(r'x\_obs [pc]')
            plt.ylabel(r'y\_obs [pc]')
            plt.show()

            #hdu = fits.PrimaryHDU(vcgr)
            #hdul = fits.HDUList([hdu])
            #hdul.writeto('/Users/jonathancohn/Documents/dyn_mod/groupmtg/ugc_2698_newmasks/vlosgas_smooth5e8.fits')
            #print(oop)
            # '''  #
            # BUCKET END TESTING PYTHON INTEGRATION

            # Calculate the (inner) integral (see eqn 2.157 from Binney & Tremaine)
            # int1_a2 = np.zeros(shape=len(rvals))
            int1_a3 = np.zeros(shape=len(rvals))
            for i in range(1, len(rvals)):  # for i=1.,n_elements(rvals)-1 (BC IDL INDEXING INCLUSIVE!)
                # int1_a2[i] = np.sum(rvals[i:] * sigr2[i:] * del_r / np.sqrt(rvals[i:]**2 - avals[i-1]**2))
                int1_a3[i] = np.sum(rvals[i:] * sigr3[i:] * del_r / np.sqrt(rvals[i:]**2 - avals[i-1]**2))

            plt.plot(rvals, int1_a3, 'bo', markerfacecolor='none', label="Ben's integration")
            plt.plot(rvals, int1, 'k+', label="My integration")
            plt.plot(rvals, intintr, 'r-', label="Smoothed spline interpolation of my integration")
            plt.legend()
            plt.show()
            #hh = 1e-5
            #int1[rvals > 550] = 0.
            #plt.plot(rvals, int1, 'k+', label='My integral')
            #plt.plot(rvals, int1_a3, 'bo', label="Ben's integration", markerfacecolor='none')
            #plt.legend()
            #plt.show()
            #fx_minus_h = innerint(rvals, avals, sigr3, del_r, h=hh)
            #fx = innerint(rvals, avals, sigr3, del_r, h=0)
            #dda3 = (fx - fx_minus_h) / hh
            #plt.plot(dda, 'k+')
            #plt.plot(dda3, 'bo')
            #plt.show()
            # print(oop)

            #dda3 = np.zeros(shape=len(rvals))
            #hh = 1e-9  # robust for at LEAST 1e-9 <= h <= 1e-3
            #for i in range(1, len(rvals)):  # for i=1.,n_elements(rvals)-1 (BC IDL INDEXING INCLUSIVE!)
            #    # int1_a2[i] = np.sum(rvals[i:] * sigr2[i:] * del_r / np.sqrt(rvals[i:]**2 - avals[i-1]**2))
            #    dhi = np.sum(rvals[i:] * sigr3[i:] * del_r / np.sqrt(rvals[i:]**2 - (avals[i-1]+hh)**2))
            #    dlo = np.sum(rvals[i:] * sigr3[i:] * del_r / np.sqrt(rvals[i:]**2 - avals[i-1]**2))
            #    dda3[i] = (dhi - dlo) / hh

            #int1r = interpolate.interp1d(rvals, fx, kind='quadratic', fill_value='extrapolate')
            #plt.imshow(int1r(R), origin='lower')
            #plt.colorbar()
            #plt.show()

            # Crude numerical differential wrt radius (d/da) for 2nd (outer) integral (see eqn 2.157 Binney & Tremaine)
            # int1_dda2 = np.zeros(shape=len(rvals))
            int1_dda3 = np.zeros(shape=len(rvals))

            # int1_dda2[1:] = (int1_a2[1:] - int1_a2[0:-1]) / del_r
            int1_dda3[1:] = (int1_a3[1:] - int1_a3[0:-1]) / del_r  # Offset indices in int1_a* by 1 so diff -> deriv


            plt.plot(rvals, int1_dda3, 'bo', markerfacecolor='none', label="Ben's derivative")
            plt.plot(rvals, dda, 'k+', label="My old derivative")
            plt.plot(rvals, dda_sp_r, 'r-', label="Derivative at each point from the spline function")
            plt.legend()
            plt.show()

            #plt.plot(int1_dda3, 'ko')
            #plt.show()

            #plt.plot(dda3 - int1_dda3, 'ko')
            #plt.ylabel("My derivative - Ben's derivative")
            #plt.xlabel("Radius")
            #plt.show()

            print(int1_dda3, 'hi')
            #ddar = interpolate.interp1d(rvals, dda3, kind='quadratic', fill_value='extrapolate')
            #ddarr = ddar(R)
            #ddarr[np.abs(ddarr) >= 5e3] = 0.
            #plt.imshow(ddarr, origin='lower')
            #plt.colorbar()
            #plt.show()
            # Calculate the second (outer) integral (eqn 2.157 Binney & Tremaine)
            # int2_r2 = np.zeros(shape=len(avals))
            int2_r3 = np.zeros(shape=len(avals))
            for i in range(1, len(rvals) - 1):  # only go to len(avals)-1 (in IDL: -2) bc index rvals[i+1]
                # int2_r2[i] = np.sum(avals[0:i] * int1_dda2[0:i] / np.sqrt(rvals[i+1]**2 - avals[0:i]**2) * del_r)
                int2_r3[i] = np.sum(avals[0:i] * int1_dda3[0:i] / np.sqrt(rvals[i+1]**2 - avals[0:i]**2) * del_r)

            print(int2_r3, 'int2')
            # print(oop)

            int22 = interpolate.interp1d(rvals, int2_r3, kind='quadratic', fill_value='extrapolate')
            int22r = int22(R)
            #plt.imshow(int22r, origin='lower')
            #plt.colorbar()
            #plt.show()
            plt.plot(rvals, np.sqrt(-4 * self.G_pc * int2_r3), 'bo', markerfacecolor='none', label="Ben's vcirc,gas")
            plt.plot(rvals, np.sqrt(-4 * self.G_pc * int2_before), 'k+', label="My old vcirc,gas")
            plt.plot(rvals, np.sqrt(-4 * self.G_pc * int2), 'r-', label="vcirc,gas based on the spline function")
            plt.legend()
            plt.show()

            plt.plot(rvals, int2_r3, 'bo', markerfacecolor='none', label="Ben's outer integral")
            plt.plot(rvals, int2_before, 'k+', label="My old outer integral")
            plt.plot(rvals, int2, 'r-', label="My outer integral based on the spline function")
            plt.legend()
            plt.show()

            plt.plot(rvals, np.sqrt(-4 * self.G_pc * int2_r3), 'bo', markerfacecolor='none', label="Ben's vcirc,gas")
            plt.plot(rvals, np.sqrt(-4 * self.G_pc * int2_before), 'k+', label="My old vcirc,gas")
            plt.plot(rvals, np.sqrt(-4 * self.G_pc * int2), 'r-', label="vcirc,gas based on the spline function")
            plt.legend()
            plt.show()

            # Numerical v_cg solution assuming an exponential mass distribution (vc2) & one following the CO sb (vc3)
            # vc2 = np.sqrt(np.abs(-4 * self.G_pc * int2_r2))
            #vc3 = np.sqrt(np.abs(-4 * self.G_pc * int2_r3))
            print(np.amax(int2_r3), np.amin(int2_r3))
            vc3 = np.sqrt(-4 * self.G_pc * int2_r3)
            print(np.amax(vc3), np.amin(vc3))
            #print(oop)
            #vc3 = np.nan_to_num(vc3)

            # INTERPOLATE/EXTRAPOLATE FROM velocity(rvals) TO velcoity(R)
            # vc2_r = interpolate.interp1d(rvals, vc2, kind='zero', fill_value='extrapolate')
            vc3_r = interpolate.interp1d(rvals, vc3, kind='quadratic', fill_value='extrapolate')

            # Note that since potentials are additive, sum up the velocity contributions in quadrature:
            # vg = vc2_r(R)
            # vg = vc3_r(R)
            alpha = abs(np.arctan(y_disk / (np.cos(self.inc) * x_disk)))  # measure alpha from +x (minor ax) to +y (maj ax)
            sign = x_disk / abs(x_disk)  # (+x now back to redshifted side, so don't need extra minus sign back in front!)
            vg = sign * abs(vcgr * np.cos(alpha) * np.sin(self.inc))  # v_los > 0 -> redshift; v_los < 0 -> blueshift
            plt.imshow(vg, origin='lower', extent=[x_obs[0], x_obs[-1], y_obs[0], y_obs[-1]], cmap='RdBu_r')
            cbar = plt.colorbar()
            cbar.set_label(r'km/s')
            plt.xlabel(r'x\_obs [pc]')
            plt.ylabel(r'y\_obs [pc]')
            plt.show()
            print(oop)
            if not self.quiet:
                print(time.time() - t_gas, ' seconds spent in gas calculation')

        if self.menc_type == 0:  # if calculating v(R) due to stars directly from MGE parameters
            if not self.quiet:
                print('mge')
            test_rad = np.linspace(np.amin(R_ac), np.amax(R_ac), 100)  # create an array of test radii

            comp, surf_pots, sigma_pots, qobs = mvm.load_mge(self.enclosed_mass)  # load the MGE parameters

            v_c = mvm.mge_vcirc(surf_pots * self.ml_ratio, sigma_pots, qobs, np.rad2deg(self.inc), 0., self.dist,
                                test_rad)  # calculate v_c,star
            v_c_r = interpolate.interp1d(test_rad, v_c, kind='cubic', fill_value='extrapolate')  # interpolate v_c(R)

            # CALCULATE KEPLERIAN VELOCITY OF ANY POINT (x_disk, y_disk) IN THE DISK WITH RADIUS R (km/s)
            vel = np.sqrt((self.G_pc * self.mbh / R) + v_c_r(R_ac)**2 + vg**2)
        elif self.menc_type == 1:  # elif using a file with v_circ(R) due to stellar mass
            if not self.quiet:
                print('v(r)')
            radii = []
            v_circ = []
            with open(self.enclosed_mass) as em:  # current file has units v_c^2/(M/L) --> v_c = np.sqrt(col * (M/L))
                for line in em:
                    cols = line.split()  # note: using Ben's model "B1" = 2nd col in file (file has 4 cols of models)
                    radii.append(float(cols[0]))  # file lists radii in pc
                    v_circ.append(float(cols[1]))  # v^2 / (M/L) --> units (km/s)^2 / (M_sol/L_sol)

            v_c_r = interpolate.interp1d(radii, v_circ, fill_value='extrapolate')  # create function: interpolate v_c(R)

            # CALCULATE KEPLERIAN VELOCITY OF ANY POINT (x_disk, y_disk) IN THE DISK WITH RADIUS R (km/s)
            vel = np.sqrt(v_c_r(R) * self.ml_ratio + (self.G_pc * self.mbh / R) + vg**2)  # velocities sum in quadrature

        elif self.menc_type == 2:  # elif using a file directly with stellar mass as a function of R
            if not self.quiet:
                print('M(R)')
            radii = []
            m_stellar = []
            with open(self.enclosed_mass) as em:
                for line in em:
                    cols = line.split()
                    cols[1] = cols[1].replace('D', 'e')  # for some reason this file uses (e.g.) 0.5D08 instead of 0.5e8
                    radii.append(float(cols[0]) * 10 ** 3)  # file lists radii in kpc; convert to pc
                    m_stellar.append(float(cols[1]))  # solar masses
            m_star_r = interpolate.interp1d(radii, m_stellar, kind='cubic', fill_value='extrapolate')  # create fcn
            ml_const = self.ml_ratio / 7.35  # because mass file assumes a mass-to-light ratio of 7.35
            m_R = self.mbh + ml_const * m_star_r(R)  # function interpolates mass at all radii R (2d array)

            # CALCULATE KEPLERIAN VELOCITY OF ANY POINT (x_disk, y_disk) IN THE DISK WITH RADIUS R (km/s)
            vel = np.sqrt(self.G_pc * m_R / R + vg**2)  # Keplerian velocity vel at each point in the disk

        # CALCULATE LINE-OF-SIGHT VELOCITY AT EACH POINT (x_disk, y_disk) IN THE DISK (km/s)
        alpha = abs(np.arctan(y_disk / (np.cos(self.inc) * x_disk)))  # measure alpha from +x (minor ax) to +y (maj ax)
        sign = x_disk / abs(x_disk)  # (+x now back to redshifted side, so don't need extra minus sign back in front!)
        v_los = sign * abs(vel * np.cos(alpha) * np.sin(self.inc))  # v_los > 0 -> redshift; v_los < 0 -> blueshift

        # INCLUDE NEW RADIAL VELOCITY TERM
        vrad_sign = y_disk / abs(y_disk)  # With this sign convention: vrad > 0 -> outflow; vrad < 0 -> inflow!
        if self.vtype == 'vrad':
            v_los += self.vrad * vrad_sign * abs(np.sin(alpha) * np.sin(self.inc))  # See notebook for derivation!
        elif self.vtype == 'kappa':  # use just kappa
            v_los += self.kappa * vrad_sign * abs(vel * np.sin(alpha) * np.sin(self.inc))
        elif self.vtype == 'omega':  # use omega and kappa both!
            v_los = self.omega * v_los + self.kappa * vrad_sign * abs(vel * np.sin(alpha) * np.sin(self.inc))

        # SET LINE-OF-SIGHT VELOCITY AT THE BLACK HOLE CENTER TO BE 0, SUCH THAT IT DOES NOT BLOW UP
        center = (R == 0.)  # Doing this is only relevant if we have pixel located exactly at the center
        v_los[center] = 0.  # if any point is at x_disk, y_disk = (0., 0.), set velocity there = 0.

        # CALCULATE VELOCITY PROFILES
        sigma = get_sig(r=R, sig0=self.sig_params[0], r0=self.sig_params[1], mu=self.sig_params[2],
                        sig1=self.sig_params[3])[self.sig_type]

        # CONVERT v_los TO OBSERVED FREQUENCY MAP
        self.freq_obs = (self.f_0 / (1+self.zred)) * (1 - v_los / self.c_kms)  # convert v_los to f_obs

        # CONVERT OBSERVED DISPERSION (turbulent) TO FREQUENCY WIDTH
        sigma_grid = np.zeros(shape=R.shape) + sigma  # make sigma (whether already R-shaped or constant) R-shaped
        self.delta_freq_obs = (self.f_0 / (1 + self.zred)) * (sigma_grid / self.c_kms)  # convert sigma to delta_f

        # WEIGHTS FOR LINE PROFILES: apply weights to gaussian velocity profiles for each subpixel
        self.weight = subpix_deconvolved  # [Jy/beam Hz]

        # WEIGHT CURRENTLY IN UNITS OF Jy/beam * Hz --> need to get it in units of Jy/beam to match data
        self.weight *= self.f_w / np.sqrt(2 * np.pi * self.delta_freq_obs**2)  # divide to get correct units
        # NOTE: MAYBE multiply by fstep here, after collapsing and lucy process, rather than during collapse
        # NOTE: would then need to multiply by ~1000 factor or something large there, bc otherwise lucy cvg too fast

        # print(self.c_kms * (1 - (self.fstep / self.f_0) * (1+self.zred)))

        # BEN_LUCY COMPARISON ONLY (only use for comparing to model with Ben's lucy map, which is in different units)
        if self.bl:  # (multiply by vel_step because Ben's lucy map is actually in Jy/beam km/s units)
            self.weight *= self.fstep * 6.783  # NOTE: 6.783 == beam size / channel width in km/s
            # channel width = 1.537983987579E+07 Hz --> v_width = self.c * (1 - self.f_0 / (self.fstep / (1+self.zred)))

        if not self.quiet:
            print(str(time.time() - t_grid) + ' seconds in grids()')


        '''
        if self.kin_file is not None:
            print(self.kin_file)
            with open(self.kin_file, 'w+') as kv:  # 'kin_velbin.txt'
                kv.write('#######################\n')
                kv.write('   XBIN   YBIN   VEL   \n')
                kv.write('#######################\n')
                for i in range(len(v_los)):
                    for j in range(len(v_los[0])):
                        kv.write('   ' + str(i) + '   ' + str(j) + '   ' + str(v_los[i, j]) + '\n')
            print('done')
        '''
        # DO KINEMETRY (XBIN, YBIN: 1D arrays with X,Y coords describing map; MOMENT: 1D array with kin.moment e.g velocity values at XBIN,YBIN positions)
        # x_obs, y_obs, v_los, X0=self.xloc, Y0=self.yloc, NAME='UGC2698_kinemetry'
        # ALTERNATIVELY: x = range(self.xyrange[1]-self.xyrange[0]), y = range(self.xyrange[3]-self.xyrange[2]), and scale=self.resolution, X0=self.xloc, Y0=self.yloc
        # Should I set IMG? IMG=self.subpix_deconvolved? Or do I want the not-deconvolved fluxmap?
        # KINEMETRY(x_obs, y_obs, v_los,
        # KINEMETRY, xbin, ybin, velbin, rad, pa, q, cf, ntrm=6, scale=0.8, $
	    #           ERROR=er_velbin, name='NGC2974',er_cf=er_cf, er_pa=er_pa, $
	    #           er_q=er_q, /plot, /verbose




    def convolution(self):
        # BUILD GAUSSIAN LINE PROFILES!!!
        cube_model = np.zeros(shape=(len(self.freq_ax), len(self.freq_obs), len(self.freq_obs[0])))  # initialize cube
        for fr in range(len(self.freq_ax)):
            cube_model[fr] = self.weight * np.exp(-(self.freq_ax[fr] - self.freq_obs) ** 2 /
                                                  (2 * self.delta_freq_obs ** 2))

        # RE-SAMPLE BACK TO CORRECT PIXEL SCALE (take avg of sxs sub-pixels for real alma pixel) --> intrinsic data cube
        if self.os == 1:
            intrinsic_cube = cube_model
        else:
            intrinsic_cube = rebin(cube_model, self.os)  # intrinsic_cube = block_reduce(cube_model, self.os, np.mean)

        tc = time.time()
        # CONVERT INTRINSIC TO OBSERVED (convolve each slice of intrinsic_cube with ALMA beam --> observed data cube)
        self.convolved_cube = np.zeros(shape=intrinsic_cube.shape)  # approx ~1e-6 to 3e-6s per pixel
        for z in range(len(self.z_ax)):
            self.convolved_cube[z, :, :] = convolution.convolve(intrinsic_cube[z, :, :], self.beam)
        print('convolution loop ' + str(time.time() - tc))

        # BUCKET WAS USING THIS TO GENERATE KINEMETRY INPUT
        #self.generate_kinemetry_input(model=False, mom=2, snr=10)#, filename='u2698_moment_vorbin_avgdat_snr10.txt')
        #print(oop)


    def vorbinning(self, snr, m1=None, cube=None, filename=None):
        """

        :param snr: target Signal-to-Noise Ratio
        :param m1: moment map to be rebuilt on scale of voronoi-binned map (ie. avg the moment map in each bin)
        :param cube: model cube to be rebuilt on scale of voronoi-binned map (ie. average the line profiles in each bin)
                     and then use that voronoi-binned cube to generate the moment map later
        :param filename: file to which to save XBIN, YBIN, moment map
        :return:
        """

        hdu_m = fits.open(self.data_mask)
        data_mask = hdu_m[0].data  # The mask is stored in hdu_m[0].data, NOT hdu_m[0].data[0]
        hdu_m.close()

        # estimate the 2D collapsed signal, and estimate a constant noise
        sig = np.zeros(shape=self.input_data[0].shape)
        noi = 0
        for z in range(len(self.input_data)):
            sig += self.input_data[z] * data_mask[z] / len(self.input_data)
            noi += np.mean(self.input_data[z, params['yerr0']:params['yerr1'], params['xerr0']:params['xerr1']]) /\
                   len(self.input_data)
        import vorbin
        from vorbin.voronoi_2d_binning import voronoi_2d_binning

        sig = sig[self.xyrange[2]:self.xyrange[3], self.xyrange[0]:self.xyrange[1]]
        print(noi)

        signal_input = []
        noise_input = []
        x_in = []  # used as arcsec-scale input
        y_in = []  # used as arcsec-scale input
        xpix = []  # just store pixel number
        ypix = []  # just store pixel number
        if len(sig) % 2. == 0:  # if even
            yctr = (len(sig)) / 2.  # set the center of the axes (in pixel number)
        else:  # elif odd
            yctr = (len(sig) + 1.) / 2.  # +1 bc python starts counting at 0
        if len(sig[0]) % 2 == 0.:
            xctr = (len(sig[0])) / 2.  # set the center of the axes (in pixel number)
        else:  # elif odd
            xctr = (len(sig[0]) + 1.) / 2.  # +1 bc python starts counting at 0

        for yy in range(len(sig)):
            for xx in range(len(sig[0])):
                if sig[yy, xx] != 0:  # don't include pixels that have been masked out!
                    xpix.append(xx)
                    ypix.append(yy)
                    x_in.append(xx - xctr)  # pixel scale, with 0 at center
                    y_in.append(yy - yctr)  # pixel scale, with 0 at center
                    noise_input.append(noi)
                    signal_input.append(sig[yy, xx])

        target_snr = snr
        signal_input = np.asarray(signal_input)
        noise_input = np.asarray(noise_input)
        x_in = np.asarray(x_in) * self.resolution  # convert to arcsec-scale
        y_in = np.asarray(y_in) * self.resolution  # convert to arcsec-scale

        # Perform voronoi binning! The vectors (binNum, xNode, yNode, xBar, yBar, sn, nPixels, scale) are *output*
        binNum, xNode, yNode, xBar, yBar, sn, nPixels, scale = voronoi_2d_binning(x_in, y_in, signal_input, noise_input,
                                                                                  target_snr, plot=1, quiet=1)
        plt.show()
        print(binNum, sn, nPixels)  # len=# of pix, bin # for each pix; len=# of bins: SNR/bin; len=# of bins: # pix/bin

        if cube is not None:
            print('Averaging Voronoi binning on the cube')
            flat_binned_cube = np.zeros(shape=(len(cube), max(binNum) + 1))  # flatten & bin the cube as func(slice)
            for zc in range(len(cube)):
                for xy in range(len(x_in)):
                    flat_binned_cube[zc, binNum[xy]] += cube[zc, ypix[xy], xpix[xy]] / nPixels[binNum[xy]]

            # convert the flattened binned cube into a vector where each slice has the same size as the x & y inputs
            full_binned_cube = np.zeros(shape=(len(cube), len(binNum)))
            for zc in range(len(cube)):
                for xy in range(len(x_in)):
                    full_binned_cube[zc, xy] += flat_binned_cube[zc, binNum[xy]]

            # convert the full binned cube back to the same size as the input cube, now with the contents voronoi binned
            cube_vb = np.zeros(shape=cube.shape)
            for zc in range(len(cube)):
                for xy in range(len(x_in)):
                    cube_vb[zc, ypix[xy], xpix[xy]] = flat_binned_cube[zc, xy]

            self.convolved_cube = cube_vb
            mg.moment_12(abs_diff=False, incl_beam=False, norm=False, mom=1)
            mg.moment_12(abs_diff=False, incl_beam=False, norm=False, mom=2)

            return cube_vb

        elif m1 is not None:
            print('Averaging Voronoi binning on the moment map')
            flattened_binned_m1 = np.zeros(shape=max(binNum)+1)  # flatten and bin the moment 1 map
            for xy in range(len(x_in)):
                flattened_binned_m1[binNum[xy]] += m1[ypix[xy], xpix[xy]] / nPixels[binNum[xy]]

            # convert the flattened binned moment 1 map into a vector of the same size as the x & y inputs
            full_binned_m1 = np.zeros(shape=len(binNum))
            for xy in range(len(x_in)):
                full_binned_m1[xy] += flattened_binned_m1[binNum[xy]]

            '''  #
            dir = '/Users/jonathancohn/Documents/dyn_mod/'
            with open(dir+'voronoi_2d_binning_2698_output.txt', 'r') as vb:
                for line in vb:
                    if not line.startswith('#'):
                        cols = line.split()
                        flattened_binned_m1[int(cols[2])] += m1[int(cols[1]), int(cols[0])] / nPixels[int(cols[2])]
    
            full_binned_m1 = np.zeros(shape=len(binNum))
            dir = '/Users/jonathancohn/Documents/dyn_mod/'
            with open(dir+'voronoi_2d_binning_2698_output.txt', 'r') as vb:
                ii = 0
                for line in vb:
                    if not line.startswith('#'):
                        cols = line.split()
                        full_binned_m1[ii] += flattened_binned_m1[int(cols[2])]
                        ii += 1
            # '''

            # '''
            if filename is not None:
                dir = '/Users/jonathancohn/Documents/dyn_mod/'
                with open(dir+filename, 'w+') as vb:  # dir+'u2698_moment_vorbin_snr15.txt'
                    vb.write('# targetSN=' + str(target_snr) + '\n')
                    vb.write('#######################\n')
                    vb.write('   XBIN   YBIN   VEL   \n')
                    vb.write('#######################\n')
                    # vb.write('# x y binNum\n')
                    for xy in range(len(x_in)):
                        vb.write(str(x_in[xy]) + ' ' + str(y_in[xy]) + ' ' + str(full_binned_m1[xy]) + '\n')
                        # m1_vb[y_in[xy], x_in[xy]] = full_binned_m1[xy]
                    #for xy in range(len(x_in)):
                    #    vb.write(str(x_in[xy]) + ' ' + str(y_in[xy]) + ' ' + str(binNum[xy]) + ' ' + str(full_binned_m1[xy]) +
                    #             '\n')
                    #    m1_vb[y_in[xy], x_in[xy]] = full_binned_m1[xy]
                # '''

                ###  dlogz 1.1431055564826238 thresh 0.02 nc 518024 niter 9051
            # create the binned moment map for display
            m1_vb = np.zeros(shape=m1.shape)
            for xy in range(len(x_in)):
                m1_vb[ypix[xy], xpix[xy]] = full_binned_m1[xy]

            plt.imshow(m1_vb, origin='lower', cmap='RdBu_r')  # plot it!
            plt.colorbar()
            plt.show()

            return m1_vb


    def chi2(self):
        # ONLY WANT TO FIT WITHIN ELLIPTICAL REGION! CREATE ELLIPSE MASK
        self.ell_mask = ellipse_fitting(self.convolved_cube, self.rfit, self.xell, self.yell, self.resolution,
                                        self.theta_ell, self.q_ell)  # create ellipse mask

        # CREATE A CLIPPED DATA CUBE SO THAT WE'RE LOOKING AT THE SAME EXACT x,y,z REGION AS IN THE MODEL CUBE
        self.clipped_data = self.input_data[self.zrange[0]:self.zrange[1], self.xyrange[2]:self.xyrange[3],
                                            self.xyrange[0]:self.xyrange[1]]

        #self.clipped_data = self.vorbinned_cube(snr=8)
        cube_vb = self.vorbinned_cube(snr=8)

        vb_ds = rebin(cube_vb, self.ds)

        for ix,iy in [[4,6],[10,8],[12,8],[7,5],[14,9],[15,10]]:
            print(ix,iy)
            self.line_profiles_comparevb(cube_vb, ix, iy, noise2=self.noise, show_freq=False)
        print(oop)

        # self.convolved_cube *= ell_mask  # mask the convolved model cube
        # self.input_data_masked = self.input_data[self.zrange[0]:self.zrange[1], self.xyrange[2]:self.xyrange[3],
        #                          self.xyrange[0]:self.xyrange[1]] * ell_mask  # mask the input data cube

        # REBIN THE ELLIPSE MASK BY THE DOWN-SAMPLING FACTOR
        #fig = plt.figure(figsize=(12,8))
        #plt.imshow(self.ell_mask, origin='lower')
        #plt.colorbar()
        #plt.show()
        self.ell_ds = rebin(self.ell_mask, self.ds)[0]  # rebin the mask by the down-sampling factor
        #fig = plt.figure(figsize=(12,8))
        #plt.imshow(self.ell_ds, origin='lower')
        #plt.colorbar()
        #plt.show()
        #self.ell_ds[self.ell_ds < 0.5] = 0.  # if averaging instead of summing
        self.ell_ds[self.ell_ds < self.ds**2 / 2.] = 0.  # set all pix < 50% "inside" the ellipse to be outside -> mask
        self.ell_ds = np.nan_to_num(self.ell_ds / np.abs(self.ell_ds))  # set all points in ellipse = 1, convert nan->0
        #fig = plt.figure(figsize=(12,8))
        #plt.imshow(self.ell_ds, origin='lower')
        #plt.colorbar()
        #plt.show()
        #print(oop)
        # REBIN THE DATA AND MODEL BY THE DOWN-SAMPLING FACTOR: compare data and model in binned groups of dsxds pix
        data_ds = rebin(self.clipped_data, self.ds)
        ap_ds = rebin(self.convolved_cube, self.ds)

        # APPLY THE ELLIPTICAL MASK TO MODEL CUBE & INPUT DATA
        data_ds *= self.ell_ds
        ap_ds *= self.ell_ds
        n_pts = np.sum(self.ell_ds) * len(self.z_ax)  # total number of pixels compared in chi^2 calculation!

        chi_sq = 0.  # initialize chi^2
        cs = []  # initialize chi^2 per slice

        z_ind = 0  # the actual index for the model-data comparison cubes
        for z in range(self.zrange[0], self.zrange[1]):  # for each relevant freq slice (ignore slices with only noise)
            chi_sq += np.sum((ap_ds[z_ind] - data_ds[z_ind])**2 / self.noise[z_ind]**2)  # calculate chisq!
            cs.append(np.sum((ap_ds[z_ind] - data_ds[z_ind])**2 / self.noise[z_ind]**2))  # chisq per slice
            # np.std(x) = sqrt(mean(abs(x - x.mean())**2))

            z_ind += 1  # the actual index for the model-data comparison cubes

        if not self.quiet:
            print(np.sum(self.ell_ds), len(self.z_ax), n_pts)
            print(r'chi^2=', chi_sq)

        if self.reduced:  # CALCULATE REDUCED CHI^2
            chi_sq /= (n_pts - self.n_params)  # convert to reduced chi^2; else just return full chi^2
            if not self.quiet:
                print(r'Reduced chi^2=', chi_sq)
                print(n_pts - self.n_params)

        if n_pts == 0.:  # PROBLEM WARNING
            print(self.resolution, self.xell, self.yell, self.theta_ell, self.q_ell, self.rfit)
            print('WARNING! STOP! There are no pixels inside the fitting ellipse! n_pts = ' + str(n_pts))
            chi_sq = np.inf

        return chi_sq  # Reduced or Not depending on reduced = True or False


    def line_profiles_comparevb(self, cube_vb, ix, iy, noise2=None, show_freq=False):
        # compare line profiles between regular cube and voronoi-binned cube
        f_sys = self.f_0 / (1 + self.zred)
        print(ix, iy)
        data_ds = rebin(self.clipped_data, self.ds)
        vb_ds = rebin(cube_vb, self.ds)

        hdu_m = fits.open(self.data_mask)
        data_mask = hdu_m[0].data  # The mask is stored in hdu_m[0].data, NOT hdu_m[0].data[0]
        v_width = 2.99792458e5 * (1 + (6454.9 / 2.99792458e5)) * self.fstep / self.f_0  # velocity width [km/s] = c*(1+v/c)*fstep/f0
        mask_ds = rebin(data_mask[self.zrange[0]:self.zrange[1], self.xyrange[2]:self.xyrange[3],
                        self.xyrange[0]:self.xyrange[1]], self.ds)

        if show_freq:
            plt.plot(self.freq_ax / 1e9, vb_ds[:, iy, ix], 'r*', label=r'Voronoi-binned')
            plt.plot(self.freq_ax / 1e9, data_ds[:, iy, ix], 'k+', label=r'Data')
            plt.plot(self.freq_ax / 1e9, self.noise, 'k--', label=r'Noise (std)')
            plt.axvline(x=f_sys / 1e9, color='k', label=r'$f_{sys}$')
            plt.xlabel(r'Frequency [GHz]')
        else:
            vel_ax = []
            for v in range(len(self.freq_ax)):
                vel_ax.append(self.c_kms * (1. - (self.freq_ax[v] / self.f_0) * (1 + self.zred)))
            dv = vel_ax[1] - vel_ax[0]
            #vel_ax.insert(0, vel_ax[0])
            #plt.errorbar(vel_ax, data_ds[:, iy, ix], yerr=self.noise, color='k', marker='+', label=r'Data')
            plt.fill_between(vel_ax, data_ds[:, iy, ix] - self.noise, data_ds[:, iy, ix] + self.noise, color='k',
                             step='mid', alpha=0.3)
            plt.step(vel_ax, data_ds[:, iy, ix], color='k', where='mid', label=r'Data')  # width=vel_ax[1] - vel_ax[0], alpha=0.4
            #plt.plot(vel_ax + dv/2., data_ds[:, iy, ix], ls='steps', color='k', label=r'Data')  # width=vel_ax[1] - vel_ax[0], alpha=0.4
            #plt.plot(vel_ax, ap_ds[:, iy, ix], color='r', marker='+', ls='none', label=r'Model')  # 'r+'
            plt.fill_between(vel_ax, vb_ds[:, iy, ix] - noise2, vb_ds[:, iy, ix] + noise2, color='b', step='mid',
                             alpha=0.3)
            plt.step(vel_ax, vb_ds[:, iy, ix], color='b', where='mid', label=r'Voronoi-binned')  # width=vel_ax[1] - vel_ax[0], alpha=0.5
            plt.axvline(x=0., color='k', ls='--', label=r'v$_{\text{sys}}$')
            plt.xlabel(r'Line-of-sight velocity [km/s]')
        plt.legend()
        plt.ylabel(r'Flux Density [Jy/beam]')
        plt.show()


    def line_profiles(self, ix, iy, show_freq=False):  # compare line profiles at the given indices ix, iy
        f_sys = self.f_0 / (1 + self.zred)
        print(ix, iy)
        data_ds = rebin(self.clipped_data, self.ds)
        ap_ds = rebin(self.convolved_cube, self.ds)

        hdu_m = fits.open(self.data_mask)
        data_mask = hdu_m[0].data  # The mask is stored in hdu_m[0].data, NOT hdu_m[0].data[0]
        v_width = 2.99792458e5 * (1 + (6454.9 / 2.99792458e5)) * self.fstep / self.f_0  # velocity width [km/s] = c*(1+v/c)*fstep/f0
        mask_ds = rebin(data_mask[self.zrange[0]:self.zrange[1], self.xyrange[2]:self.xyrange[3],
                        self.xyrange[0]:self.xyrange[1]], self.ds)

        collapse_flux_v = np.zeros(shape=(len(data_ds[0]), len(data_ds[0][0])))
        for zi in range(len(data_ds)):
            collapse_flux_v += data_ds[zi] * mask_ds[zi] * v_width
            # self.clipped_data[zi] * data_mask[zi, self.xyrange[2]:self.xyrange[3], self.xyrange[0]:self.xyrange[1]]* v_width
        #plt.imshow(collapse_flux_v, origin='lower')
        #cbar = plt.colorbar()
        #cbar.set_label(r'Jy km s$^{-1}$ beam$^{-1}$', rotation=270, labelpad=20.)
        #plt.plot(ix, iy, 'w*')
        #plt.show()
        #plt.imshow(ap_ds[20], origin='lower')
        #plt.show()
        #print(oop)
        if show_freq:
            plt.plot(self.freq_ax / 1e9, ap_ds[:, iy, ix], 'r*', label=r'Model')
            plt.plot(self.freq_ax / 1e9, data_ds[:, iy, ix], 'k+', label=r'Data')
            plt.plot(self.freq_ax / 1e9, self.noise, 'k--', label=r'Noise (std)')
            plt.axvline(x=f_sys / 1e9, color='k', label=r'$f_{sys}$')
            plt.xlabel(r'Frequency [GHz]')
        else:
            vel_ax = []
            for v in range(len(self.freq_ax)):
                vel_ax.append(self.c_kms * (1. - (self.freq_ax[v] / self.f_0) * (1 + self.zred)))
            dv = vel_ax[1] - vel_ax[0]
            #vel_ax.insert(0, vel_ax[0])
            #plt.errorbar(vel_ax, data_ds[:, iy, ix], yerr=self.noise, color='k', marker='+', label=r'Data')
            plt.fill_between(vel_ax, data_ds[:, iy, ix] - self.noise, data_ds[:, iy, ix] + self.noise, color='k',
                             step='mid', alpha=0.3)
            plt.step(vel_ax, data_ds[:, iy, ix], color='k', where='mid', label=r'Data')  # width=vel_ax[1] - vel_ax[0], alpha=0.4
            #plt.plot(vel_ax + dv/2., data_ds[:, iy, ix], ls='steps', color='k', label=r'Data')  # width=vel_ax[1] - vel_ax[0], alpha=0.4
            #plt.plot(vel_ax, ap_ds[:, iy, ix], color='r', marker='+', ls='none', label=r'Model')  # 'r+'
            plt.step(vel_ax, ap_ds[:, iy, ix], color='b', where='mid', label=r'Model')  # width=vel_ax[1] - vel_ax[0], alpha=0.5
            plt.axvline(x=0., color='k', ls='--', label=r'v$_{\text{sys}}$')
            plt.xlabel(r'Line-of-sight velocity [km/s]')
        plt.legend()
        plt.ylabel(r'Flux Density [Jy/beam]')
        plt.show()



    def pvd(self):
        from pvextractor import pvextractor
        from pvextractor.geometry import path
        from pvextractor import extract_pv_slice
        print(len(self.clipped_data[0]), len(self.clipped_data[0][0]), len(self.clipped_data))
        # path1 = path.Path([(0, 0), (len(self.clipped_data[0]), len(self.clipped_data[0][0]))], width=self.pvd_width)
        path1 = path.Path([(self.xyrange[0], self.xyrange[2]), (self.xyrange[1], self.xyrange[3])], width=self.pvd_width)
        # [(x0,y0), (x1,y1)]
        print(self.input_data.shape)
        pvd_dat, slice = extract_pv_slice(self.input_data[self.zrange[0]:self.zrange[1]], path1)
        print(self.convolved_cube.shape)
        path2 = path.Path([(0, 0), (self.xyrange[1] - self.xyrange[0], self.xyrange[3] - self.xyrange[2])], width=self.pvd_width)
        pvd_dat2, slice2 = extract_pv_slice(self.convolved_cube, path2)
        print(slice)

        vel_ax = []
        for v in range(len(self.freq_ax)):
            vel_ax.append(self.c_kms * (1. - (self.freq_ax[v] / self.f_0) * (1 + self.zred)))

        fig, ax = plt.subplots(3, 1, sharex=True)
        plt.subplots_adjust(hspace=0.02)
        print(slice.shape)

        x_rad = np.zeros(shape=len(slice[0]))
        if len(slice[0]) % 2. == 0:  # if even
            xr_c = (len(slice[0])) / 2.  # set the center of the axes (in pixel number)
            for i in range(len(slice[0])):
                x_rad[i] = self.resolution * (i - xr_c) # (arcsec/pix) * N_pix = arcsec
        else:  # elif odd
            xr_c = (len(slice[0]) + 1.) / 2.  # +1 bc python starts counting at 0
            for i in range(len(slice[0])):
                x_rad[i] = self.resolution * ((i + 1) - xr_c)

        #from mpl_toolkits.axes_grid1 import make_axes_locatable
        #divider1 = make_axes_locatable(ax[0])
        print(x_rad, len(x_rad))
        print(vel_ax)
        # CONVERT FROM Jy/beam TO mJy/beam
        slice *= 1e3
        slice2 *= 1e3
        vmin = np.amin([slice, slice2])
        vmax = np.amax([slice, slice2])
        p1 = ax[0].pcolormesh(x_rad, vel_ax, slice, vmin=vmin, vmax=vmax)  # x_rad[0], x_rad[-1]
        fig.colorbar(p1, ax=ax[0], ticks=[-0.5, 0, 0.5, 1, 1.5], pad=0.02)
        #ax[0].imshow(slice, origin='lower', extent=[x_rad[0], x_rad[-1], vel_ax[0], vel_ax[-1]])  # x_rad[0], x_rad[-1]
        #cax1 = divider1.append_axes('right', size='5%', pad=0.05)
        #fig.colorbar(im1, cax=cax1, orientation='vertical')

        p2 = ax[1].pcolormesh(x_rad, vel_ax, slice2, vmin=vmin, vmax=vmax)  # x_rad[0], x_rad[-1]
        cb2 = fig.colorbar(p2, ax=ax[1], ticks=[-0.5, 0, 0.5, 1, 1.5], pad=0.02)
        cb2.set_label(r'mJy beam$^{-1}$', rotation=270, labelpad=20.)
        #ax[1].imshow(slice2, origin='lower', extent=[x_rad[0], x_rad[-1], vel_ax[0], vel_ax[-1]])
        #ax[1].colorbar()

        # p3 = ax[2].pcolormesh(x_rad, vel_ax, slice - slice2, vmin=np.amin([vmin, slice - slice2]), vmax=vmax)
        p3 = ax[2].pcolormesh(x_rad, vel_ax, slice - slice2, vmin=np.amin(slice - slice2), vmax=np.amax(slice - slice2))
        fig.colorbar(p3, ax=ax[2], ticks=[-1., -0.5, 0, 0.5, 1], pad=0.02)
        #ax[2].imshow(slice - slice2, origin='lower', extent=[x_rad[0], x_rad[-1], vel_ax[0], vel_ax[-1]])
        #ax[2].colorbar()

        #ax[2].set_xticks([2, 27, 52, 77, 102])
        #ax[2].set_xticklabels([x_rad[2], x_rad[27], x_rad[52], x_rad[77], x_rad[102]])

        ax[1].set_ylabel('Velocity [km/s]')
        plt.xlabel('Distance [arcsec]')
        #plt.colorbar()
        plt.show()


    def output_cube(self):  # if outputting actual cube itself
        if not Path(self.out_name).exists():  # WRITE OUT RESULTS TO FITS FILE
            hdu = fits.PrimaryHDU(self.convolved_cube)
            hdul = fits.HDUList([hdu])
            hdul.writeto(self.out_name)
            print('written!')


    def test_ellipse(self):
        # USE BELOW FOR TESTING
        cf = rebin(rebin(self.weight, self.os), self.ds)[0]  # re-binned weight map, for reference
        plt.imshow(self.ell_ds * cf, origin='lower')  # masked weight map
        plt.title('4x4-binned ellipse * weight map')
        plt.colorbar()
        plt.show()

        plt.imshow(cf, origin='lower')  # re-binned weight map by itself, for reference
        plt.title('4x4-binned weight map')
        plt.colorbar()
        plt.show()


    def vorbinned_cube(self, snr=8):
        """
        Create moment map within voronoi-binned regions, based on averaging the line profiles in each voronoi bin

        :param snr: target Signal-to-Noise Ratio

        :return:
        """
        import vorbin
        from vorbin.voronoi_2d_binning import voronoi_2d_binning

        hdu_m = fits.open(self.data_mask)  # open data mask
        data_mask = hdu_m[0].data  # the mask is stored in hdu_m[0].data, NOT hdu_m[0].data[0]
        hdu_m.close()

        sig = np.zeros(shape=self.input_data[0].shape)  # estimate the 2D collapsed signal
        noi = 0  # estimate a constant noise
        for z in range(len(self.input_data)):
            sig += self.input_data[z] * data_mask[z] / len(self.input_data)
            noi += np.mean(self.input_data[z, params['yerr0']:params['yerr1'], params['xerr0']:params['xerr1']]) / \
                   len(self.input_data)

        sig = sig[self.xyrange[2]:self.xyrange[3], self.xyrange[0]:self.xyrange[1]]
        # print(noi)

        self.clipped_data = self.input_data[self.zrange[0]:self.zrange[1], self.xyrange[2]:self.xyrange[3],
                                            self.xyrange[0]:self.xyrange[1]]

        signal_input = []
        noise_input = []
        x_in = []  # used as arcsec-scale input
        y_in = []  # used as arcsec-scale input
        xpix = []  # just store pixel number
        ypix = []  # just store pixel number
        if len(sig) % 2. == 0:  # if even
            yctr = (len(sig)) / 2.  # set the center of the axes (in pixel number)
        else:  # elif odd
            yctr = (len(sig) + 1.) / 2.  # +1 bc python starts counting at 0
        if len(sig[0]) % 2 == 0.:
            xctr = (len(sig[0])) / 2.  # set the center of the axes (in pixel number)
        else:  # elif odd
            xctr = (len(sig[0]) + 1.) / 2.  # +1 bc python starts counting at 0

        for yy in range(len(sig)):
            for xx in range(len(sig[0])):
                if sig[yy, xx] != 0:  # don't include pixels that have been masked out!
                    xpix.append(xx)
                    ypix.append(yy)
                    x_in.append(xx - xctr)  # pixel scale, with 0 at center
                    y_in.append(yy - yctr)  # pixel scale, with 0 at center
                    noise_input.append(noi)
                    signal_input.append(sig[yy, xx])

        target_snr = snr
        signal_input = np.asarray(signal_input)
        noise_input = np.asarray(noise_input)
        x_in = np.asarray(x_in) * self.resolution  # convert to arcsec-scale
        y_in = np.asarray(y_in) * self.resolution  # convert to arcsec-scale

        # Perform voronoi binning! The vectors (binNum, xNode, yNode, xBar, yBar, sn, nPixels, scale) are *output*
        binNum, xNode, yNode, xBar, yBar, sn, nPixels, scale = voronoi_2d_binning(x_in, y_in, signal_input, noise_input,
                                                                                  target_snr, plot=1, quiet=1)
        plt.show()
        # print(binNum, sn, nPixels)  # len=# of pix, bin # for each pix; len=# of bins: SNR/bin; len=# of bins: # pix/bin

        flat_binned_cube = np.zeros(shape=(len(self.clipped_data), max(binNum) + 1))  # flatten & bin cube as f(slice)
        for zc in range(len(self.clipped_data)):
            for xy in range(len(x_in)):
                flat_binned_cube[zc, binNum[xy]] += self.clipped_data[zc, ypix[xy], xpix[xy]] / nPixels[binNum[xy]]

        # convert the flattened binned cube into a vector where each slice has the same size as the x & y inputs
        full_binned_cube = np.zeros(shape=(len(self.clipped_data), len(binNum)))
        for zc in range(len(self.clipped_data)):
            for xy in range(len(x_in)):
                full_binned_cube[zc, xy] += flat_binned_cube[zc, binNum[xy]]

        # convert the full binned cube back to the same size as the input cube, now with the contents voronoi binned
        cube_vb = np.zeros(shape=self.clipped_data.shape)
        print(full_binned_cube.shape)  # 57, 2875
        print(flat_binned_cube.shape)  # 57, 716
        print(self.clipped_data.shape)  # 57, 64, 84
        print(max(xpix), max(ypix))  # 80, 59
        for zc in range(len(self.clipped_data)):
            for xy in range(len(x_in)):
                cube_vb[zc, ypix[xy], xpix[xy]] = full_binned_cube[zc, xy]

        return cube_vb


    def moment_0(self, abs_diff, incl_beam, norm, samescale=False):
        """
        Create 0th moment map

        :param abs_diff: True or False; if True, show absolute value of the residual
        :param incl_beam: True or False; if True, include beam inset in the data panel
        :param norm: True or False; if True, normalize residual by the data
        :param samescale: True or False; if True, show the residual on the same scale as the data & model
        :return: moment map plot
        """
        # if using equation from https://www.atnf.csiro.au/people/Tobias.Westmeier/tools_hihelpers.php#moments
        hdu_m = fits.open(self.data_mask)
        data_mask = hdu_m[0].data  # The mask is stored in hdu_m[0].data, NOT hdu_m[0].data[0]
        hdu_m.close()
        vel_ax = []
        velwidth = self.c_kms * (1 + self.zred) * self.fstep / self.f_0
        for v in range(len(self.freq_ax)):
            vel_ax.append(self.c_kms * (1. - (self.freq_ax[v] / self.f_0) * (1 + self.zred)))

        # full cube strictmask, clipped to the appropriate zrange
        # (NOTE: would need to clip to xyrange, & rebin with ds, to compare data_ds & ap_ds. Seems wrong thing to do.)
        clipped_mask = data_mask[self.zrange[0]:self.zrange[1], self.xyrange[2]:self.xyrange[3],
                                 self.xyrange[0]:self.xyrange[1]]

        data_masked_m0 = np.zeros(shape=self.ell_mask.shape)
        for z in range(len(vel_ax)):
            data_masked_m0 += abs(velwidth) * self.clipped_data[z] * clipped_mask[z]

        model_masked_m0 = np.zeros(shape=(len(self.convolved_cube[0]), len(self.convolved_cube[0][0])))
        for zi in range(len(self.convolved_cube)):
            model_masked_m0 += self.convolved_cube[zi] * abs(velwidth) * clipped_mask[zi]

        '''
        fig = plt.figure()
        grid = AxesGrid(fig, 111,
                        nrows_ncols=(3, 1),
                        axes_pad=0.01,
                        cbar_mode='single',
                        cbar_location='right',
                        cbar_pad=0.1)
        '''
        fig, ax = plt.subplots(3, 1)
        plt.subplots_adjust(hspace=0.02)

        # CONVERT TO mJy
        data_masked_m0 *= 1e3
        model_masked_m0 *= 1e3

        subtr = 0.
        if incl_beam:
            beam_overlay = np.zeros(shape=self.ell_mask.shape)
            print(self.beam.shape, beam_overlay.shape)
            # beam_overlay[:self.beam.shape[0], (beam_overlay.shape[1] - self.beam.shape[1]):] = self.beam
            beam_overlay[:self.beam.shape[0] - 6, (beam_overlay.shape[1] - self.beam.shape[1]) + 6:] = self.beam[6:, :-6]
            print(beam_overlay.shape, self.beam.shape)
            beam_overlay *= np.amax(data_masked_m0) / np.amax(beam_overlay)
            data_masked_m0 += beam_overlay
            subtr = beam_overlay
        vmin = np.amin([np.nanmin(model_masked_m0), np.nanmin(data_masked_m0)])
        vmax = np.amax([np.nanmax(model_masked_m0), np.nanmax(data_masked_m0)])
        cbartitle0 = r'mJy Km s$^{-1}$ beam$^{-1}$'

        im0 = ax[0].imshow(data_masked_m0, vmin=vmin, vmax=vmax, origin='lower')
        ax[0].set_title(r'Moment 0 (top - bottom: data, model, residual)')
        cbar = fig.colorbar(im0, ax=ax[0], pad=0.02)
        cbar.set_label(cbartitle0, rotation=270, labelpad=20.)

        im1 = ax[1].imshow(model_masked_m0, vmin=vmin, vmax=vmax, origin='lower')
        cbar1 = fig.colorbar(im1, ax=ax[1], pad=0.02)
        cbar1.set_label(cbartitle0, rotation=270, labelpad=20.)

        title0 = 'Moment 0 residual (model-data)'
        titleabs = 'Moment 0 residual abs(model-data)'
        diff = model_masked_m0 - (data_masked_m0 - subtr)
        if norm:
            diff /= data_masked_m0
            diff = np.nan_to_num(diff)
            print(np.nanquantile(diff, [0.16, 0.5, 0.84]), 'typical differences; 0.16, 0.5, 0.84!')
            title0 += ' / data'
            titleabs += ' / data'
            cbartitle0 = 'Ratio [Residual / Data]'
        if samescale:
            if abs_diff:
                diff = np.abs(diff)
            im2 = ax[2].imshow(diff, vmin=vmin, vmax=vmax, origin='lower')
        else:  # then residual scale
            # im2 = ax[2].imshow(diff, origin='lower', vmin=np.nanmin([diff, -diff]), vmax=np.nanmax([diff, -diff]))
            im2 = ax[2].imshow(diff, origin='lower', vmin=np.nanmin(diff), vmax=np.nanmax(diff))
        cbar2 = fig.colorbar(im2, ax=ax[2], pad=0.02)
        cbar2.set_label(cbartitle0, rotation=270, labelpad=20.)

        ax[0].set_xticklabels([])
        ax[1].set_xticklabels([])
        ax[2].set_xlabel(r'x [pixels]', fontsize=20)  # x [arcsec]
        ax[0].set_ylabel(r'y [pixels]', fontsize=20)  # y [arcsec]
        ax[1].set_ylabel(r'y [pixels]', fontsize=20)  # y [arcsec]
        ax[2].set_ylabel(r'y [pixels]', fontsize=20)  # y [arcsec]

        plt.show()


    def moment_12(self, abs_diff, incl_beam, norm, mom, samescale=False, snr=5, filename=None):
        """
        Create 1st or 2nd moment map
        :param abs_diff: True or False; if True, show absolute value of the residual
        :param incl_beam: True or False; if True, include beam inset in the data panel
        :param norm: True or False; if True, normalize residual by the data
        :param mom: moment, 1 or 2
        :param samescale: True or False; if True, show the residual on the same scale as the data & model

        :return: moment map plot
        """

        hdu_m = fits.open(self.data_mask)
        data_mask = hdu_m[0].data  # The mask is stored in hdu_m[0].data, NOT hdu_m[0].data[0]
        hdu_m.close()
        vel_ax = []
        velwidth = self.c_kms * (1 + self.zred) * self.fstep / self.f_0
        for v in range(len(self.freq_ax)):
            vel_ax.append(self.c_kms * (1. - (self.freq_ax[v] / self.f_0) * (1 + self.zred)))

        # full cube strictmask, clipped to the appropriate zrange
        # (NOTE: would need to clip to xyrange, & rebin with ds, to compare data_ds & ap_ds. Seems wrong thing to do.)
        clipped_mask = data_mask[self.zrange[0]:self.zrange[1], self.xyrange[2]:self.xyrange[3],
                                 self.xyrange[0]:self.xyrange[1]]

        model_numerator = np.zeros(shape=(len(self.convolved_cube[0]), len(self.convolved_cube[0][0])))
        model_denominator = np.zeros(shape=(len(self.convolved_cube[0]), len(self.convolved_cube[0][0])))
        for zi in range(len(self.convolved_cube)):
            model_numerator += vel_ax[zi] * self.convolved_cube[zi] * clipped_mask[zi]
            model_denominator += self.convolved_cube[zi] * clipped_mask[zi]
        model_mom = model_numerator / model_denominator

        data_numerator = np.zeros(shape=(len(self.convolved_cube[0]), len(self.convolved_cube[0][0])))
        data_denominator = np.zeros(shape=(len(self.convolved_cube[0]), len(self.convolved_cube[0][0])))
        for zi in range(len(self.convolved_cube)):
            data_numerator += vel_ax[zi] * self.clipped_data[zi] * clipped_mask[zi]
            data_denominator += self.clipped_data[zi] * clipped_mask[zi]
        data_mom = data_numerator / data_denominator

        if mom == 2:
            m2_num = np.zeros(shape=(len(self.convolved_cube[0]), len(self.convolved_cube[0][0])))
            m2_den = np.zeros(shape=(len(self.convolved_cube[0]), len(self.convolved_cube[0][0])))
            for zi in range(len(self.convolved_cube)):
                m2_num += (vel_ax[zi] - model_mom)**2 * self.convolved_cube[zi] * clipped_mask[zi]
                m2_den += self.convolved_cube[zi] * clipped_mask[zi]
            m2 = np.sqrt(m2_num / m2_den) # * d1  # BUCKET: no need for MASKING using d1?

            d2_num = np.zeros(shape=(len(self.convolved_cube[0]), len(self.convolved_cube[0][0])))
            d2_n2 = np.zeros(shape=(len(self.convolved_cube[0]), len(self.convolved_cube[0][0])))
            d2_den = np.zeros(shape=(len(self.convolved_cube[0]), len(self.convolved_cube[0][0])))
            for zi in range(len(self.convolved_cube)):
                d2_n2 += self.clipped_data[zi] * (vel_ax[zi] - data_mom)**2 * clipped_mask[zi] # * mask2d
                d2_num += (vel_ax[zi] - data_mom)**2 * self.clipped_data[zi] * clipped_mask[zi] # * mask2d
                d2_den += self.clipped_data[zi] * clipped_mask[zi] # * mask2d
            dfig = np.zeros(shape=(len(self.convolved_cube[0]), len(self.convolved_cube[0][0]))) + 1.  # create mask
            dfig2 = np.zeros(shape=(len(self.convolved_cube[0]), len(self.convolved_cube[0][0]))) + 1.  # create mask
            dfig[d2_n2 < 0.] = 0.  # d2_n2 matches d2_den on the sign aspect
            dfig2[d2_den < 0.] = 0.

            d2_num[d2_num < 0] = 0.  # BUCKET ADDING TO GET RID OF NANs
            d2 = np.sqrt(d2_num / d2_den) # * d1  # BUCKET: no need for MASKING using d1?

            '''
            fig = plt.figure()
            grid = AxesGrid(fig, 111,
                            nrows_ncols=(3, 1),
                            axes_pad=0.01,
                            cbar_mode='single',
                            cbar_location='right',
                            cbar_pad=0.1)
            i = 0
            '''
            fig, ax = plt.subplots(3, 1)
            plt.subplots_adjust(hspace=0.02)

            subtr = 0.
            if incl_beam:
                d2 = np.nan_to_num(d2)
                beam_overlay = np.zeros(shape=self.ell_mask.shape)
                beam_overlay[:self.beam.shape[0], (beam_overlay.shape[1] - self.beam.shape[1]):] = self.beam
                print(beam_overlay.shape, self.beam.shape)
                beam_overlay *= np.amax(d2) / np.amax(beam_overlay)
                d2 += beam_overlay
                subtr = beam_overlay

            #for ax in grid:
            vmin2 = np.amin([np.nanmin(d2), np.nanmin(m2)])
            vmax2 = np.amax([np.nanmax(d2), np.nanmax(m2)])
            cbartitle2 = r'km/s'
            im0 = ax[0].imshow(d2, origin='lower', vmin=vmin2, vmax=vmax2)  # , cmap='RdBu_r'
            ax[0].set_title(r'Moment 2 (top - bottom: data, model, residual)')
            cbar = fig.colorbar(im0, ax=ax[0], pad=0.02)
            cbar.set_label(cbartitle2, rotation=270, labelpad=20.)

            im1 = ax[1].imshow(m2, origin='lower', vmin=vmin2, vmax=vmax2)  # , cmap='RdBu_r'
            cbar2 = fig.colorbar(im1, ax=ax[1], pad=0.02)
            cbar2.set_label(cbartitle2, rotation=270, labelpad=20.)

            diff = m2 - (d2 - subtr)
            title2 = 'Moment 2 residual (model-data)'
            titleabs2 = 'Moment 2 residual abs(model-data)'
            if norm:
                diff /= d2
                print(np.nanquantile(diff, [0.16, 0.5, 0.84]), 'look median!')
                title2 += ' / data'
                titleabs2 += ' / data'
                cbartitle2 = 'Ratio [Residual / Data]'
            if abs_diff:
                diff = np.abs(diff)
                #ax.set_title(titleabs2)
            #else:
                #ax.set_title(title2)
            if samescale:
                im2 = ax[2].imshow(diff, origin='lower', vmin=vmin2, vmax=vmax2)  # , cmap='RdBu'
            else:  # residscale
                #im2 = ax.imshow(diff, origin='lower', vmin=np.nanmin([diff, -diff]),
                #                vmax=np.nanmax([diff, -diff]))
                print(np.nanmin(diff), np.nanmax(diff))
                im2 = ax[2].imshow(diff, origin='lower', vmin=np.nanmin(diff), vmax=np.nanmax(diff))  # np.nanmin(diff)
                #ax.set_title(title2)
            cbar2 = fig.colorbar(im2, ax=ax[2], pad=0.02)
            cbar2.set_label(cbartitle2, rotation=270, labelpad=20.)

            ax[0].set_xticklabels([])
            ax[1].set_xticklabels([])
            ax[2].set_xlabel(r'x [pixels]', fontsize=20)  # x [arcsec]
            ax[0].set_ylabel(r'y [pixels]', fontsize=20)  # y [arcsec]
            ax[1].set_ylabel(r'y [pixels]', fontsize=20)  # y [arcsec]
            ax[2].set_ylabel(r'y [pixels]', fontsize=20)  # y [arcsec]

            plt.show()
            # self.vorbinning(m1=np.nan_to_num(d2), snr=snr, filename=filename)


        elif mom == 1:
            '''
            fig = plt.figure()
            grid = AxesGrid(fig, 111,
                            nrows_ncols=(1, 3),
                            axes_pad=0.01,
                            cbar_mode='single',
                            cbar_location='right',
                            cbar_pad=0.1)
            i = 0
            '''
            fig, ax = plt.subplots(3, 1)
            plt.subplots_adjust(hspace=0.02)

            subtr = 0.
            if incl_beam:
                beam_overlay = np.zeros(shape=self.ell_mask.shape)
                beam_overlay[:self.beam.shape[0], (beam_overlay.shape[1] - self.beam.shape[1]):] = self.beam
                print(beam_overlay.shape, self.beam.shape)
                beam_overlay *= np.amax(data_mom) / np.amax(beam_overlay)
                data_mom += beam_overlay
                subtr = beam_overlay

            cbartitle1 = r'km/s'
            data_mom[np.abs(data_mom) > 1e3] = 0
            print(np.nanmax(data_mom), np.nanmin(data_mom), np.nanmax(model_mom), np.nanmin(model_mom))
            vmin1 = np.amin([np.nanmin(data_mom), np.nanmin(model_mom)])
            vmax1 = np.amax([np.nanmax(data_mom), np.nanmax(model_mom)])
            im0 = ax[0].imshow(data_mom, origin='lower', vmin=vmin1, vmax=vmax1, cmap='RdBu_r')
            ax[0].set_title(r'Moment 1 (top - bottom: data, model, residual)')
            cbar0 = fig.colorbar(im0, ax=ax[0], ticks=[-500, -250, 0., 250.], pad=0.02)
            cbar0.set_label(cbartitle1, rotation=270, labelpad=20.)

            im1 = ax[1].imshow(model_mom, origin='lower', vmin=vmin1, vmax=vmax1, cmap='RdBu_r')
            #ax[1].set_title(r'Moment 1 (model)')
            cbar1 = fig.colorbar(im1, ax=ax[1], ticks=[-500, -250, 0., 250.], pad=0.02)
            cbar1.set_label(cbartitle1, rotation=270, labelpad=20.)
            title1 = 'Moment 1 residual (model - data)'
            diff = model_mom - (data_mom - subtr)
            if norm:
                diff /= data_mom
                print(np.nanquantile(diff, [0.16, 0.5, 0.84]), 'look median!')
                title1 += ' / data'
                cbartitle1 = 'Ratio [Residual / Data]'
            if samescale:
                im2 = ax[2].imshow(diff, origin='lower', vmin=vmin1, vmax=vmax1, cmap='RdBu')  # , cmap='RdBu'
            else:  # resid scale
                vn = np.amax([-150, np.nanmin(diff)])
                vx = np.amin([150, np.nanmax(diff)])
                #im2 = ax[2].imshow(diff, origin='lower', vmin=np.nanmin(diff), vmax=np.nanmax(diff))
                im2 = ax[2].imshow(diff, origin='lower', vmin=vn, vmax=vx)
                #im2 = ax[2].imshow(diff, origin='lower', vmin=np.nanmin([diff, -diff]),
                #                   vmax=np.nanmax([diff, -diff]))  # cmap='RdBu'
            # ax.set_title(title1)
            cbar2 = fig.colorbar(im2, ax=ax[2], pad=0.02)
            cbar2.set_label(cbartitle1, rotation=270, labelpad=20.)

            ax[0].set_xticklabels([])
            ax[1].set_xticklabels([])
            ax[2].set_xlabel(r'x [pixels]', fontsize=20)  # x [arcsec]
            ax[0].set_ylabel(r'y [pixels]', fontsize=20)  # y [arcsec]
            ax[1].set_ylabel(r'y [pixels]', fontsize=20)  # y [arcsec]
            ax[2].set_ylabel(r'y [pixels]', fontsize=20)  # y [arcsec]

            plt.show()

            # NOTE HERE IS VORONOI BINNING STUFF
            for i in range(len(data_mom)):
                for j in range(len(data_mom[0])):
                    if np.isnan(data_mom[i, j]):
                        data_mom[i, j] = 0.
            # self.vorbinning(m1=data_mom, snr=snr, filename=filename)  # 4, filename='u2698_moment_vorbin_snr4_ac.txt')

            if self.kin_file is not None:
                print(self.kin_file)
                #for i in range(len(data_mom)):
                #    for j in range(len(data_mom[0])):
                #        if np.isnan(data_mom[i, j]):
                #            data_mom[i, j] = 0.
                #self.vorbinning(m1=data_mom)
                '''
                with open(self.kin_file, 'w+') as kv:  # 'kin_velbin.txt'
                    kv.write('#######################\n')
                    kv.write('   XBIN   YBIN   VEL   \n')
                    kv.write('#######################\n')
                    for i in range(len(data_mom)):
                        for j in range(len(data_mom[0])):
                            if np.isnan(data_mom[i, j]):
                                data_mom[i, j] = 0.
                            # kv.write('   ' + str(i) + '   ' + str(j) + '   ' + str(data_mom[i, j]) + '\n')
                            kv.write('   ' + str(self.xobs[i]) + '   ' + str(self.yobs[j]) + '   ' + str(data_mom[i, j])
                                     + '\n')
                    plt.imshow(data_mom, origin='lower')
                    plt.colorbar()
                    plt.show()
                print('done')
                '''

            return data_mom


    def generate_kinemetry_input(self, model=False, mom=1, snr=8, filename=None):
        """
        Create moment map within voronoi-binned regions, based on averaging the line profiles in each voronoi bin

        :param model: True or False; if True, generating kinemetry input from model cube; else, generating from data
        :param mom: moment, 1 or 2
        :param snr: target Signal-to-Noise Ratio
        :param filename: file to which to save XBIN, YBIN, moment map

        :return:
        """
        import vorbin
        from vorbin.voronoi_2d_binning import voronoi_2d_binning

        hdu_m = fits.open(self.data_mask)  # open data mask
        data_mask = hdu_m[0].data  # the mask is stored in hdu_m[0].data, NOT hdu_m[0].data[0]
        hdu_m.close()

        sig = np.zeros(shape=self.input_data[0].shape)  # estimate the 2D collapsed signal
        noi = 0  # estimate a constant noise
        for z in range(len(self.input_data)):
            sig += self.input_data[z] * data_mask[z] / len(self.input_data)
            noi += np.mean(self.input_data[z, params['yerr0']:params['yerr1'], params['xerr0']:params['xerr1']]) / \
                   len(self.input_data)

        sig = sig[self.xyrange[2]:self.xyrange[3], self.xyrange[0]:self.xyrange[1]]
        # print(noi)

        self.clipped_data = self.input_data[self.zrange[0]:self.zrange[1], self.xyrange[2]:self.xyrange[3],
                                            self.xyrange[0]:self.xyrange[1]]

        signal_input = []
        noise_input = []
        x_in = []  # used as arcsec-scale input
        y_in = []  # used as arcsec-scale input
        xpix = []  # just store pixel number
        ypix = []  # just store pixel number
        if len(sig) % 2. == 0:  # if even
            yctr = (len(sig)) / 2.  # set the center of the axes (in pixel number)
        else:  # elif odd
            yctr = (len(sig) + 1.) / 2.  # +1 bc python starts counting at 0
        if len(sig[0]) % 2 == 0.:
            xctr = (len(sig[0])) / 2.  # set the center of the axes (in pixel number)
        else:  # elif odd
            xctr = (len(sig[0]) + 1.) / 2.  # +1 bc python starts counting at 0

        for yy in range(len(sig)):
            for xx in range(len(sig[0])):
                if sig[yy, xx] != 0:  # don't include pixels that have been masked out!
                    xpix.append(xx)
                    ypix.append(yy)
                    x_in.append(xx - xctr)  # pixel scale, with 0 at center
                    y_in.append(yy - yctr)  # pixel scale, with 0 at center
                    noise_input.append(noi)
                    signal_input.append(sig[yy, xx])

        target_snr = snr
        signal_input = np.asarray(signal_input)
        noise_input = np.asarray(noise_input)
        x_in = np.asarray(x_in) * self.resolution  # convert to arcsec-scale
        y_in = np.asarray(y_in) * self.resolution  # convert to arcsec-scale

        # Perform voronoi binning! The vectors (binNum, xNode, yNode, xBar, yBar, sn, nPixels, scale) are *output*
        binNum, xNode, yNode, xBar, yBar, sn, nPixels, scale = voronoi_2d_binning(x_in, y_in, signal_input, noise_input,
                                                                                  target_snr, plot=1, quiet=1)
        plt.show()
        # print(binNum, sn, nPixels)  # len=# of pix, bin # for each pix; len=# of bins: SNR/bin; len=# of bins: # pix/bin

        flat_binned_cube = np.zeros(shape=(len(self.clipped_data), max(binNum) + 1))  # flatten & bin cube as f(slice)
        for zc in range(len(self.clipped_data)):
            for xy in range(len(x_in)):
                flat_binned_cube[zc, binNum[xy]] += self.clipped_data[zc, ypix[xy], xpix[xy]] / nPixels[binNum[xy]]

        # convert the flattened binned cube into a vector where each slice has the same size as the x & y inputs
        full_binned_cube = np.zeros(shape=(len(self.clipped_data), len(binNum)))
        for zc in range(len(self.clipped_data)):
            for xy in range(len(x_in)):
                full_binned_cube[zc, xy] += flat_binned_cube[zc, binNum[xy]]

        # convert the full binned cube back to the same size as the input cube, now with the contents voronoi binned
        cube_vb = np.zeros(shape=self.clipped_data.shape)
        print(full_binned_cube.shape)  # 57, 2875
        print(flat_binned_cube.shape)  # 57, 716
        print(self.clipped_data.shape)  # 57, 64, 84
        print(max(xpix), max(ypix))  # 80, 59
        for zc in range(len(self.clipped_data)):
            for xy in range(len(x_in)):
                cube_vb[zc, ypix[xy], xpix[xy]] = full_binned_cube[zc, xy]

        vel_ax = []  # convert freq axis to velocity axis
        for v in range(len(self.freq_ax)):
            vel_ax.append(self.c_kms * (1. - (self.freq_ax[v] / self.f_0) * (1 + self.zred)))

        # full cube strictmask, clipped to the appropriate zrange
        clipped_mask = data_mask[self.zrange[0]:self.zrange[1], self.xyrange[2]:self.xyrange[3],
                                 self.xyrange[0]:self.xyrange[1]]

        # Calculate Moment 1 (model)
        model_numerator = np.zeros(shape=(len(self.convolved_cube[0]), len(self.convolved_cube[0][0])))
        model_denominator = np.zeros(shape=(len(self.convolved_cube[0]), len(self.convolved_cube[0][0])))
        for zi in range(len(self.convolved_cube)):
            model_numerator += vel_ax[zi] * self.convolved_cube[zi] * clipped_mask[zi]
            model_denominator += self.convolved_cube[zi] * clipped_mask[zi]
        model_mom = model_numerator / model_denominator

        # Calculate Moment 1 (data)
        data_numerator = np.zeros(shape=(len(self.convolved_cube[0]), len(self.convolved_cube[0][0])))
        data_denominator = np.zeros(shape=(len(self.convolved_cube[0]), len(self.convolved_cube[0][0])))
        for zi in range(len(self.convolved_cube)):
            data_numerator += vel_ax[zi] * self.clipped_data[zi] * clipped_mask[zi]
            data_denominator += self.clipped_data[zi] * clipped_mask[zi]
        data_mom = data_numerator / data_denominator

        if mom == 2:  # Calculate Moment 2 (model, then data)
            m2_num = np.zeros(shape=(len(self.convolved_cube[0]), len(self.convolved_cube[0][0])))
            m2_den = np.zeros(shape=(len(self.convolved_cube[0]), len(self.convolved_cube[0][0])))
            for zi in range(len(self.convolved_cube)):
                m2_num += (vel_ax[zi] - model_mom)**2 * self.convolved_cube[zi] * clipped_mask[zi]
                m2_den += self.convolved_cube[zi] * clipped_mask[zi]
            m2 = np.sqrt(m2_num / m2_den)

            d2_num = np.zeros(shape=(len(self.convolved_cube[0]), len(self.convolved_cube[0][0])))
            d2_n2 = np.zeros(shape=(len(self.convolved_cube[0]), len(self.convolved_cube[0][0])))
            d2_den = np.zeros(shape=(len(self.convolved_cube[0]), len(self.convolved_cube[0][0])))
            for zi in range(len(self.convolved_cube)):
                d2_n2 += self.clipped_data[zi] * (vel_ax[zi] - data_mom)**2 * clipped_mask[zi] # * mask2d
                d2_num += (vel_ax[zi] - data_mom)**2 * self.clipped_data[zi] * clipped_mask[zi] # * mask2d
                d2_den += self.clipped_data[zi] * clipped_mask[zi] # * mask2d
            d2_num[d2_num < 0] = 0.  # BUCKET ADDING TO GET RID OF NANs
            d2 = np.sqrt(d2_num / d2_den) # * d1  # BUCKET: no need for MASKING using d1?
            d2 = np.nan_to_num(d2)
            #print(np.argwhere(np.isnan(d2)))
            data_mom = d2  # replace moment 1 with moment 2 (data)
            model_mom = m2  # replace moment 1 with moment 2 (model)

        if model:  # are we generating kinemetry for the model moment map?
            moment = model_mom
        else:  # or are we generating kinemetry for the data moment map?
            moment = data_mom

        flattened_binned_m1 = np.zeros(shape=max(binNum) + 1)  # flatten and bin the selected moment map
        for xy in range(len(x_in)):
            flattened_binned_m1[binNum[xy]] += moment[ypix[xy], xpix[xy]] / nPixels[binNum[xy]]

        # convert the flattened binned moment map into a vector of the same size as the x & y inputs
        full_binned_m1 = np.zeros(shape=len(binNum))
        for xy in range(len(x_in)):
            full_binned_m1[xy] += flattened_binned_m1[binNum[xy]]

        if filename is not None:
            dir = '/Users/jonathancohn/Documents/dyn_mod/'
            with open(dir + filename, 'w+') as vb:  # dir+'u2698_moment_vorbin_snr15.txt'
                vb.write('# targetSN=' + str(target_snr) + '\n')
                vb.write('#######################\n')
                vb.write('   XBIN   YBIN   VEL   \n')
                vb.write('#######################\n')
                for xy in range(len(x_in)):
                    vb.write(str(x_in[xy]) + ' ' + str(y_in[xy]) + ' ' + str(full_binned_m1[xy]) + '\n')

        m1_vb = np.zeros(shape=moment.shape)  # create the binned moment map for display
        for xy in range(len(x_in)):
            m1_vb[ypix[xy], xpix[xy]] = full_binned_m1[xy]

        vmax = np.nanmax([m1_vb, -m1_vb])
        vmin = np.nanmin([m1_vb, -m1_vb])
        cmap = 'RdBu_r'
        if mom == 2:
            vmax = np.nanmax(m1_vb)
            vmin = np.nanmin(m1_vb)
            cmap = 'plasma'
        plt.imshow(m1_vb, origin='lower', cmap=cmap, vmax=vmax, vmin=vmin)
        plt.colorbar()
        plt.show()  # plot it!


    def kin_pa(self):

        from pafit import fit_kinematic_pa as fkpa

        xbin, ybin = np.random.uniform(low=[-30, -20], high=[30, 20], size=(100, 2)).T
        print(xbin)
        inc = 60.  # assumed galaxy inclination
        r = np.sqrt(xbin ** 2 + (ybin / np.cos(np.radians(inc))) ** 2)  # Radius in the plane of the disk
        a = 40  # Scale length in arcsec
        vr = 2000 * np.sqrt(r) / (r + a)  # Assumed velocity profile
        vel = vr * np.sin(np.radians(inc)) * xbin / r  # Projected velocity field

        plt.clf()
        ang, ang_err, v_syst = fkpa.fit_kinematic_pa(xbin, ybin, vel, debug=True, nsteps=30)
        plt.show()


def test_qell2(params, l_in, q_ell, rfit, pa, figname):
    #ell_mask = ellipse_fitting(input_data, rfit, params['xell'], params['yell'], params['resolution'], pa, q_ell)

    fig = plt.figure()
    ax = plt.gca()
    plt.imshow(l_in, origin='lower')
    plt.colorbar()
    e1 = patches.Ellipse((params['xell'], params['yell']), 2 * rfit / params['resolution'],
                         2 * rfit / params['resolution'] * q_ell, angle=pa, linewidth=2, edgecolor='w', fill=False)
    ax.add_patch(e1)
    plt.title(r'q = ' + str(q_ell) + r', PA = ' + str(pa) + ' deg, rfit = ' + str(rfit) + ' arcsec')
    if figname is None:
        plt.show()
    else:
        figname += '_' + str(q_ell) + '_' + str(pa) + '_' + str(rfit) + '.png'
        plt.savefig(figname, dpi=300)
    plt.clf()


if __name__ == "__main__":
    # MAKE SURE I HAVE ACTIVATED THE iraf27 ENVIRONMENT IF I NEED TO RUN LUCY DECONVOLUTION!!!
    t0_true = time.time()
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument('--parfile')

    args = vars(parser.parse_args())

    # Load parameters from the parameter file
    params, priors, n_free = par_dicts(args['parfile'])

    # Make nice plot fonts
    rc('font', **{'family': 'serif', 'serif': ['Times']})
    rc('text', usetex=True)

    # CREATE OUTPUT FILENAME BASED ON INPUT PARAMETERS
    pars_str = ''
    for key in params:
        pars_str += str(params[key]) + '_'
    out = '/Users/jonathancohn/Documents/dyn_mod/outputs/test_' + pars_str + '.fits'

    # CREATE THINGS THAT ONLY NEED TO BE CALCULATED ONCE (collapse fluxes, lucy, noise)
    mod_ins = model_prep(data=params['data'], ds=params['ds'], lucy_out=params['lucy'], lucy_mask=params['lucy_mask'],
                         lucy_b=params['lucy_b'], lucy_in=params['lucy_in'], lucy_it=params['lucy_it'],
                         data_mask=params['mask'], grid_size=params['gsize'], res=params['resolution'],
                         x_std=params['x_fwhm'], y_std=params['y_fwhm'], pa=params['PAbeam'],
                         xyerr=[params['xerr0'], params['xerr1'], params['yerr0'], params['yerr1']],
                         zrange=[params['zi'], params['zf']], q_ell=params['q_ell'],
                         theta_ell=np.deg2rad(params['theta_ell']), xell=params['xell'], yell=params['yell'])

    lucy_mask, lucy_out, beam, fluxes, freq_ax, f_0, fstep, input_data, noise, co_ell_sb, co_ell_rad = mod_ins

    hduin = fits.open(params['lucy_in'])
    l_in = hduin[0].data
    hduin.close()

    #dir = '/Users/jonathancohn/Documents/dyn_mod/groupmtg/ugc_2698_newmasks/'
    #for rfit in [0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 0.9, 1.]:
    #    test_qell2(params, l_in, 0.38, rfit, 19, dir)
    #for theta in [19]:
    #    for q in [0.38, 0.4, 0.5]:
    #        for rfit in [0.65, 0.7, 0.75]:
    #            test_qell2(params, l_in, q, rfit, theta, dir)
    #print(oop)
    # ig = params['incl_gas'] == 'True'

    # CREATE MODEL CUBE!
    out = params['outname']
    t0m = time.time()

    v = []
    x = []
    y = []
    with open('kinemetry/NGC2974_SAURON_kinematics.dat', 'r') as f:
        c = 0
        for line in f:
            c += 1
            if c >= 3:
                cols = line.split()
                x = cols[1]
                y = cols[2]
                v = cols[3]

    mg = ModelGrid(resolution=params['resolution'], os=params['os'], x_loc=params['xloc'], y_loc=params['yloc'],
                   mbh=params['mbh'], inc=np.deg2rad(params['inc']), vsys=params['vsys'], dist=params['dist'],
                   theta=np.deg2rad(params['PAdisk']), input_data=input_data, lucy_out=lucy_out, out_name=out,
                   beam=beam, rfit=params['rfit'], enclosed_mass=params['mass'], ml_ratio=params['ml_ratio'],
                   sig_type=params['s_type'], zrange=[params['zi'], params['zf']], menc_type=params['mtype'],
                   sig_params=[params['sig0'], params['r0'], params['mu'], params['sig1']], f_w=params['f'],
                   ds=params['ds'], noise=noise, reduced=True, freq_ax=freq_ax, q_ell=params['q_ell'],
                   theta_ell=np.deg2rad(params['theta_ell']), xell=params['xell'], yell=params['yell'], fstep=fstep,
                   f_0=f_0, bl=params['bl'], xyrange=[params['xi'], params['xf'], params['yi'], params['yf']],
                   n_params=n_free, data_mask=params['mask'], incl_gas=params['incl_gas']=='True', vrad=params['vrad'],
                   kappa=params['kappa'], omega=params['omega'], co_rad=co_ell_rad, co_sb=co_ell_sb,
                   pvd_width=(params['x_fwhm']+params['y_fwhm'])/params['resolution']/2., #)
                   kin_file='/Users/jonathancohn/Documents/dyn_mod/ugc_2698/ugc_2698_kinpc.txt')
    # gas_norm=params['gas_norm'], gas_radius=params['gas_radius']

    mg.grids()
    mg.convolution()
    chi_sq = mg.chi2()
    # x_fwhm=0.197045, y_fwhm=0.103544 -> geometric mean = sqrt(0.197045*0.103544) = 0.142838
    # mg.pvd()
    # mg.vorbinning()
    # filename='u2698_moment_vorbin_snr4_ac.txt'
    #mg.moment_12(abs_diff=False, incl_beam=False, norm=False, mom=2, snr=4, filename=None)
                 #filename='u2698_moment_vorbin_snr10_ac.txt')
    # For moment 2: SNR 5 bad, 10 good, 7 has ~1 bad pixel near center still; 8 a little rough but no more bad pixels
    # print(oop)
    # mg.moment_0(abs_diff=False, incl_beam=True, norm=False)
    # mg.moment_12(abs_diff=False, incl_beam=False, norm=False, mom=1)
    # mg.moment_12(abs_diff=False, incl_beam=False, norm=False, mom=2)

    mg.line_profiles(10, 8)  # center?
    mg.line_profiles(11, 8)  # center?
    mg.line_profiles(12, 8)  # red?
    mg.line_profiles(9, 7)  # blue?
    mg.line_profiles(7, 5)  # decent blue [recently using this]
    mg.line_profiles(14, 9)  # good red [recently using this]
    mg.line_profiles(15, 10)  # decent red [recently using this]

    #mg.line_profiles(7, 5)  # decent blue
    #mg.line_profiles(4, 6)  # blue orig (not great)
    #mg.line_profiles(6, 6)  # blue okay? (meh)
    #mg.line_profiles(10, 9)  # near ctr orig (meh)
    #mg.line_profiles(14, 8)  # decent red
    #mg.line_profiles(14, 9)  # good red
    #mg.line_profiles(14, 10)  # decent red
    #mg.line_profiles(15, 9)  # good red
    #mg.line_profiles(15, 10)  # decent red

    '''  #
    mg.line_profiles(8, 8)
    mg.line_profiles(9, 8)
    mg.line_profiles(9, 7)
    mg.line_profiles(9, 6)
    mg.line_profiles(10, 7)
    mg.line_profiles(11, 10)
    mg.line_profiles(12, 8)
    mg.line_profiles(12, 9)
    mg.line_profiles(14, 8)  # decent red
    mg.line_profiles(14, 9)  # good red
    mg.line_profiles(14, 10)  # decent red
    mg.line_profiles(15, 9)  # good red
    mg.line_profiles(15, 10)  # decent red
    for ii in range(6, 10):
        for jj in range(5, 9):
            mg.line_profiles(ii, jj)
    # not great blues: 6,5 // 6,8 // 7,8 / 8,5 / 9,7 / 9,8
    # okayish blues 6,6 // 6,7 // 7,6 // 7,7 / 8,6 / 8,7 / 9,5 / 9,6
    # maybe reasonable blues 7,5
    #mg.line_profiles(9, 9)
    #mg.line_profiles(9, 10)
    #mg.line_profiles(10, 9)  # near ctr orig
    #mg.line_profiles(10, 8)  # near ctr
    #mg.line_profiles(11, 8)  # near ctr
    #mg.line_profiles(10, 10)
    #mg.line_profiles(4, 6)  # blue orig
    #mg.line_profiles(6, 6)  # blue
    # mg.line_profiles(3, 3)
    # mg.line_profiles(6, 4)
    # mg.line_profiles(4, 4)
    #mg.line_profiles(14, 10)  # red
    #mg.line_profiles(16, 12)
    # '''  #

    print(time.time() - t0m, ' seconds')
    print('True Total time: ' + str(time.time() - t0_true) + ' seconds')  # ~1 second for a cube of 84x64x49

# ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00']
# [pale blue, orange red,  green,   pale pink, red brown,  lavender, pale gray,    red, yellow green]