import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import patches
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
from scipy.interpolate import UnivariateSpline as unsp


def integral22(rad, dda):

    int22 = integrate.quad(integrand22, 0, rad, args=(rad, dda))[0]

    return int22


def integrand22(a, rad, dda):

    integ22 = a * dda / np.sqrt(rad**2 - a**2)

    return integ22


def integral1(a, rmax, sigma_func, inclination, conversion_factor):

    int1 = integrate.quad(integrand1, a, rmax, args=(sigma_func, a, inclination, conversion_factor))[0]

    return int1


def integrand1(r, sigma_func, a, inclination, conversion_factor):

    integ1 = r * sigma_func(r) * np.cos(inclination) * conversion_factor / np.sqrt(r ** 2 - a ** 2)

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
        # import sys
        # sys.path.insert(0, '/Users/jonathancohn/Documents/jam/')  # lets me import file from different folder/path
        # import mge_vcirc_mine as mvm
        comp, surf_pots, sigma_pots, qobs = mvm.load_mge(params['mass'])

        return params, priors, nfree, qobs

    else:
        return params, priors, nfree


def model_prep(lucy_out=None, lucy_mask=None, lucy_b=None, lucy_in=None, lucy_it=None, data=None, data_mask=None,
               grid_size=None, res=1., x_std=1., y_std=1., pa=0., ds=4, ds2=4, zrange=None, xyerr=None, theta_ell=0,
               q_ell=0, xell=0, yell=0, avg=True):
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
    :param ds2: down-sampling factor for pixel scale used to calculate chi^2, allowing for non-square down-sampling
    :param zrange: range of frequency slices containing emission [zi, zf]
    :param xyerr: x & y pixel region, on the down-sampled pixel scale, where the noise is calculated [xi, xf, yi, yf]
    :param theta_ell: position angle of the annuli for the gas mass, same as for the ellipse fitting region [radians]
    :param q_ell: axis ratio q of the annuli for the gas mass, same as for the ellipse fitting region [unitless]
    :param xell: x center of elliptical annuli for the gas mass, same as for the ellipse fitting region [pixels]
    :param yell: y center of elliptical annuli for the gas mass, same as for the ellipse fitting region [pixels]
    :param avg: averaging vs summing within the rebin() function

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
    cut_y = len(input_data[0]) % ds2  # is the cube y-dimension divisible by ds2? If not, cut remainder from cube edge
    cut_x = len(input_data[0][0]) % ds  # is the cube x-dimension divisible by ds2? If not, cut remainder from cube edge
    if cut_x == 0:  # if there is no remainder:
        cut_x = -len(input_data[0][0])  # don't want to cut anything below
    if cut_y == 0:  # if there is no remainder:
        cut_y = -len(input_data[0])  # don't want to cut anything below
    noise_ds = rebin(input_data[:, :-cut_y, :-cut_x], ds2, ds, avg=avg)  # down-sample noise to the chi^2 pixel scale

    noise = []  # For large N, Variance ~= std^2
    for z in range(zrange[0], zrange[1]):  # for each relevant freq slice, calculte std (aka rms) ~variance
        noise.append(np.std(noise_ds[z, int(xyerr[2]/ds2):int(xyerr[3]/ds2), int(xyerr[0]/ds):int(xyerr[1]/ds)]))

    # CALCULATE VELOCITY WIDTH  # vsys = 6454.9 estimated from various test runs; see eg. Week of 2020-05-04 on wiki
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
    # semi_major[1:] because there is 1 less annulus than there are ellipses

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


def rebin(data, nr, nc, avg=True):
    """
    Rebin data or model cube (or one slice of a cube) in blocks of n x n pixels

    :param data: input data cube, model cube, or slice of a cube, e.g. a 2Darray
    :param nr: size of pixel binning, rows (e.g. nr=4, nc=4 rebins the date in blocks of 4x4 pixels)
    :param nc: size of pixel binning, columns
    :param avg: if True, return the mean within each rebinnined pixel, rather than the sum. If False, return the sum.
    :return: rebinned cube or slice
    """

    if avg:
        nn = 1.
    else:
        nn = nr * nc

    rebinned = []
    if len(data.shape) == 2:  # if binning a 2D array: bin data in groups of nr x nc pixels
        subarrays = blockshaped(data, nr, nc)  # subarrays.shape = len(data)*len(data[0]) / (nr*nc)
        # each pixel in the new, rebinned data cube is the mean of each nr x nc set of original pixels
        reshaped = nn * np.mean(np.mean(subarrays, axis=-1), axis=-1).reshape((int(len(data) / nr),
                                                                               int(len(data[0]) / nc)))
        rebinned.append(reshaped)  # reshaped.shape = (len(data) / nr, len(data[0]) / nc)
    else:  # if binning a 3D array
        for z in range(len(data)):  # bin each slice of the data in groups of nr x nc pixels
            subarrays = blockshaped(data[z, :, :], nr, nc)  # subarrays.shape = len(data[0])*len(data[0][0]) / (nr*nc)
            reshaped = nn * np.mean(np.mean(subarrays, axis=-1), axis=-1).reshape((int(len(data[0]) / nr),
                                                                                   int(len(data[0][0]) / nc)))
            rebinned.append(reshaped)  # reshaped.shape = (len(data[0]) / nr, len(data[0][0]) / nc)

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


def gas_vel(resolution, co_rad, co_sb, dist, f_0, inc_fixed, zfixed=0.02152):
    """

    :param radius_array: 2D R array grid built in the ModelGrid class
    :param resolution: resolution of observations [arcsec/pixel]
    :param co_rad: mean elliptical radii of annuli used in calculation of co_sb [pix]
    :param co_sb: mean CO surface brightness in elliptical annuli [Jy km/s beam^-1]
    :param dist: angular diameter distance to the galaxy [Mpc]
    :param f_0: rest frequency of the observed line in the data cube [Hz]
    :param inc_fixed: fixed inclination, used for the Binney & Tremaine integration [radians]
    :param zfixed: fixed redshift, used for unit conversion

    :return: circular gas velocity interpolation function
    """
    G_pc = (6.67 * 10**-11) * 1.989e30 * (1 / 3.086e16) / 1e3**2  # G [Msol^-1 * pc * km^2 * s^-2]
    f_he = 1.36  # additional fraction of gas that is helium (f_he = 1 + helium mass fraction)
    r21 = 0.7  # CO(2-1)/CO(1-0) SB ratio (see pg 6-8 in Boizelle+17)
    alpha_co10 = 3.1  # CO(1-0) to H2 conversion factor (see pg 6-8 in Boizelle+17)

    # pix per beam = 2pi sigx sigy [pix^2]
    pix_per_beam = 2. * np.pi * (0.197045 / resolution / 2.35482) * (0.103544 / resolution / 2.35482)
    pc_per_pix = dist * 1e6 / 206265 * resolution
    pc2_per_beam = pix_per_beam * pc_per_pix ** 2  # pc^2 per beam = pix/beam * pc^2/pix

    co_radii = co_rad * pc_per_pix  # convert input annulus mean radii from pix to pc
    co_sb = np.nan_to_num(co_sb)  # replace NaNs with 0s in input CO mean surface brightness profile

    # Set up integration bounds, numerical step size, vectors r & a (see Binney & Tremaine eqn 2.157)
    # use maxr >> than maximum CO radius, so potential contributions at maxr << those near the disk edge.
    min_r = 0.  # integration lower bound [pix or pc]
    max_r = 1300.  # upper bound [pc]; disk peaks <~100pc, extends <~700pc  # 510 for edge
    nr = 500  # 500  # number of steps used in integration process
    avals = np.linspace(min_r, max_r, nr)  # [pc]  # range(min_r,max_r,(max_r-min_r)/del_r)
    rvals = np.linspace(min_r, max_r, nr)  # [pc]  # range(min_r,max_r,(max_r-min_r)/del_r)

    # convert from Jy km/s to Msol (Boizelle+17; Carilli & Walter 13, S2.4: https://arxiv.org/pdf/1301.0371.pdf)
    msol_per_jykms = 3.25e7 * alpha_co10 * f_he * dist ** 2 / \
                     ((1 + zfixed) * r21 * (f_0 / 1e9) ** 2)  # f_0 in GHz, not Hz
    # equation for (1+z)^3 is for observed freq, but using rest freq -> nu0^2 = (nu*(1+z))^2
    # units on 3.25e7 are [K Jy^-1 pc^2/Mpc^2 GHz^2] --> total units [Msol (Jy km/s)^-1]

    # Interpolate CO surface brightness vs elliptical mean radii, to construct Sigma(rvals).
    # Units [Jy km/s/beam] * [Msol/(Jy km/s)] / [pc^2/beam] = [Msol/pc^2]
    sigr3_func_r = interpolate.interp1d(co_radii, co_sb, kind='quadratic', fill_value='extrapolate')

    # PYTHON INTEGRATION: calculate the inner integral (see eqn 2.157 from Binney & Tremaine)
    int1 = np.zeros(shape=len(avals))
    for av in range(len(avals)):
        int1[av] = integral1(avals[av], np.inf, sigr3_func_r, inc_fixed, msol_per_jykms / pc2_per_beam)
    zerocut = 580  # [pc] select point at or just beyond the disk edge
    int1[rvals > zerocut] = 0.  # set all points outside the disk edge = 0 (sometimes get spurious points)

    interp_int1 = unsp(rvals, int1)  # interpolate the inner integral using UnivariateSpline
    interp_int1.set_smoothing_factor(9e8)  # smooth the inner integral, so it will be well-behaved

    dda_sp = interp_int1.derivative()  # calculate the derivative of the spline smoothed integral
    dda_sp_r = dda_sp(rvals)  # define the derivative on the rvals grid

    int2 = np.zeros(shape=len(rvals))  # calculate the outer integral! # this integral takes ~0.35 seconds
    for rv in range(len(rvals[1:])):
        int2[rv] = integral22(rvals[rv], dda_sp_r[rv])

    vcg = np.sqrt(-4 * G_pc * int2)  # calculate the circular velocity due to the gas!
    vcg = np.nan_to_num(vcg)  # set all NaNs (as a result of negative sqrt values) = 0

    # create a function to interpolate from vcg(rvals) to vcg(R)
    vcg_func = interpolate.interp1d(rvals, vcg, kind='quadratic', fill_value='extrapolate')
    # vg = vcg_func(radius_array)  # interpolate vcg onto vcg(R)

    return vcg_func


def map_averaging(m1, xpix, ypix, binNum, x_in, nPixels):
    """

    :param m1: input moment map, calculated in ModelGrid
    :param xpix: output from ModelGrid.just_the_bins
    :param ypix: output from ModelGrid.just_the_bins
    :param binNum: output from ModelGrid.just_the_bins
    :param x_in: output from ModelGrid.just_the_bins
    :param nPixels: output from ModelGrid.just_the_bins
    :return: m1, with pixels averaged in the Voronoi bin regions generated in ModelGrid.just_the_bins
    """

    flattened_binned_m1 = np.zeros(shape=max(binNum) + 1)  # flatten and bin the moment 1 map
    for xy in range(len(x_in)):
        flattened_binned_m1[binNum[xy]] += m1[ypix[xy], xpix[xy]] / nPixels[binNum[xy]]

    # convert the flattened binned moment 1 map into a vector of the same size as the x & y inputs
    full_binned_m1 = np.zeros(shape=len(binNum))
    for xy in range(len(x_in)):
        full_binned_m1[xy] += flattened_binned_m1[binNum[xy]]

    # create the binned moment maps for display
    m1_vb = np.zeros(shape=m1.shape)
    for xy in range(len(x_in)):
        m1_vb[ypix[xy], xpix[xy]] = full_binned_m1[xy]

    return m1_vb


class ModelGrid:

    def __init__(self, resolution=0.05, os=4, x_loc=0., y_loc=0., mbh=4e8, inc=np.deg2rad(60.), vsys=None, vrad=0.,
                 kappa=0., omega=0., dist=17., theta=np.deg2rad(200.), input_data=None, lucy_out=None, vtype='orig',
                 out_name=None, beam=None, rfit=1., q_ell=1., theta_ell=0., xell=360., yell=350., bl=False,
                 enclosed_mass=None, menc_type=0, ml_ratio=1., sig_type='flat', sig_params=None, f_w=1., noise=None,
                 ds=None, ds2=None, zrange=None, xyrange=None, reduced=False, freq_ax=None, f_0=0., fstep=0., opt=True,
                 quiet=False, n_params=8, data_mask=None, avg=True, f_he=1.36, r21=0.7, alpha_co10=3.1, incl_gas=False,
                 co_rad=None, co_sb=None, z_fixed=0.02152, pvd_width=None, vcg_func=None):
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
        self.ds2 = ds2  # downsampling factor (same as self.ds, but for second dimension) [int]
        self.zrange = zrange  # array with the slices of the data cube where real emission shows up [zi, zf]
        self.xyrange = xyrange  # array with the subset of the cube (in pixels) that contains emission [xi, xf, yi, yf]
        self.reduced = reduced  # if True, ModelGrid.chi2() returns the reduced chi^2 instead of the regular chi^2
        self.freq_ax = freq_ax  # array of the frequency axis in the data cube, from bluest to reddest frequency [Hz]
        self.f_0 = f_0  # rest frequency of the observed line in the data cube [Hz]
        self.fstep = fstep  # frequency step in the frequency axis [Hz]
        self.opt = opt  # frequency axis velocity convention; opt=True -> optical; opt=False -> radio
        self.quiet = quiet  # if quiet=True, suppress printing out stuff!
        self.n_params = n_params  # number of free parameters being fit, as counted from the param file in par_dicts()
        self.avg = avg  # averaging vs summing within the rebin() function
        self.co_rad = co_rad  # mean elliptical radii of annuli used in calculation of co_sb [pix]
        self.co_sb = co_sb  # mean CO surface brightness in elliptical annuli [Jy km/s beam^-1]
        self.f_he = f_he  # additional fraction of gas that is helium (f_he = 1 + helium mass fraction)
        self.r21 = r21  # CO(2-1)/CO(1-0) SB ratio (see pg 6-8 in Boizelle+17)
        self.alpha_co10 = alpha_co10  # CO(1-0) to H2 conversion factor (see pg 6-8 in Boizelle+17)
        self.incl_gas = incl_gas  # if True, include gas mass in calculations
        self.pvd_width = pvd_width  # width (in pixels) for the PVD extraction
        self.vcg_func = vcg_func  # gas circular velocity interpolation function, returns v_c,gas(R) in units of km/s
        # Parameters to be built in create_grid(), convolve_cube(), or chi2 functions inside the class
        self.z_ax = None  # velocity axis, constructed from freq_ax, f_0, and vsys, based on opt
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
        convolution: grids must be run first; create the intrinsic model cube and convolve it with the ALMA beam
        chi2: convolution must be run first; calculates chi^2 and/or reduced chi^2
        line_profiles: chi2 must be run first; plot the line profile of a given x,y (binned pixel) 
        pvd: chi2 must be run first; generate the position-velocity diagram
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
        if self.incl_gas:  # If incl_mass, overwrite vg with v_circ,gas estimate, then add it in quadrature to velocity!
            t_gas = time.time()  # Adds ~5s for nr=200, ~13s for nr=500
            vg = self.vcg_func(R)

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


    def convolution(self):
        # BUILD GAUSSIAN LINE PROFILES!!!
        cube_model = np.zeros(shape=(len(self.freq_ax), len(self.freq_obs), len(self.freq_obs[0])))  # initialize cube
        for fr in range(len(self.freq_ax)):
            cube_model[fr] = self.weight * np.exp(-(self.freq_ax[fr] - self.freq_obs) ** 2 /
                                                  (2 * self.delta_freq_obs ** 2))

        # RE-SAMPLE BACK TO CORRECT PIXEL SCALE (take avg of sxs sub-pixels for real alma pixel) --> intrinsic data cube
        #if self.os == 1:
        #    intrinsic_cube = cube_model
        #else:  # intrinsic_cube = block_reduce(cube_model, self.os, np.mean)
        intrinsic_cube = rebin(cube_model, self.os, self.os, avg=False)  # this must use avg=False

        tc = time.time()
        # CONVERT INTRINSIC TO OBSERVED (convolve each slice of intrinsic_cube with ALMA beam --> observed data cube)
        self.convolved_cube = np.zeros(shape=intrinsic_cube.shape)  # approx ~1e-6 to 3e-6s per pixel
        for z in range(len(self.z_ax)):
            self.convolved_cube[z, :, :] = convolution.convolve(intrinsic_cube[z, :, :], self.beam)
        print('convolution loop ' + str(time.time() - tc))


    def chi2(self):
        # ONLY WANT TO FIT WITHIN ELLIPTICAL REGION! CREATE ELLIPSE MASK
        self.ell_mask = ellipse_fitting(self.convolved_cube, self.rfit, self.xell, self.yell, self.resolution,
                                        self.theta_ell, self.q_ell)  # create ellipse mask

        # CREATE A CLIPPED DATA CUBE SO THAT WE'RE LOOKING AT THE SAME EXACT x,y,z REGION AS IN THE MODEL CUBE
        self.clipped_data = self.input_data[self.zrange[0]:self.zrange[1], self.xyrange[2]:self.xyrange[3],
                                            self.xyrange[0]:self.xyrange[1]]

        # self.convolved_cube *= ell_mask  # mask the convolved model cube
        # self.input_data_masked = self.input_data[self.zrange[0]:self.zrange[1], self.xyrange[2]:self.xyrange[3],
        #                          self.xyrange[0]:self.xyrange[1]] * ell_mask  # mask the input data cube

        # REBIN THE ELLIPSE MASK BY THE DOWN-SAMPLING FACTOR
        self.ell_ds = rebin(self.ell_mask, self.ds2, self.ds, avg=self.avg)[0]  # rebin mask by the downsampling factor
        '''  #
        fig = plt.figure()
        ax = plt.gca()
        plt.imshow(self.ell_ds, origin='lower')
        plt.colorbar()
        from matplotlib import patches
        e1 = patches.Ellipse((self.xell / self.ds, self.yell / self.ds2), 2 * self.rfit / self.resolution,
                             2 * self.rfit / self.resolution * self.q_ell, angle=np.rad2deg(self.theta_ell),
                             linewidth=2, edgecolor='w', fill=False)  # np.rad2deg(params['theta_ell'])
        print(e1)
        e1.width /= self.ds
        e1.height /= self.ds
        #e1.x /= self.ds
        #e1.y /= self.ds2
        ax.add_patch(e1)
        plt.plot(self.xell / self.ds, self.yell / self.ds, 'w*')
        #plt.plot(self.xell, self.yell, 'w*')
        plt.show()
        #print(oop)
        # '''  #

        if self.avg:  # if averaging instead of summing in rebin()
            self.ell_ds[self.ell_ds < 0.5] = 0.  # pixels < 50% "inside" the ellipse are masked
        else:  # if summing instead of averaging in rebin()
            self.ell_ds[self.ell_ds < self.ds * self.ds2 / 2.] = 0.  # pixels < 50% "inside" the ellipse are masked
        self.ell_ds = np.nan_to_num(self.ell_ds / np.abs(self.ell_ds))  # set all points in ellipse = 1, convert nan->0

        '''  #
        fig = plt.figure()
        ax = plt.gca()
        plt.imshow(self.ell_ds, origin='lower')
        plt.colorbar()
        from matplotlib import patches
        e1 = patches.Ellipse((self.xell / self.ds, self.yell / self.ds2), 2 * self.rfit / self.resolution,
                             2 * self.rfit / self.resolution * self.q_ell, angle=np.rad2deg(self.theta_ell),
                             linewidth=2, edgecolor='w', fill=False)  # np.rad2deg(params['theta_ell'])
        print(e1)
        e1.width /= self.ds
        e1.height /= self.ds
        #e1.x /= self.ds
        #e1.y /= self.ds2
        ax.add_patch(e1)
        plt.plot(self.xell / self.ds, self.yell / self.ds, 'w*')
        #plt.plot(self.xell, self.yell, 'w*')
        plt.show()
        # '''  #

        # REBIN THE DATA AND MODEL BY THE DOWN-SAMPLING FACTOR: compare data and model in binned groups of dsxds pix
        data_ds = rebin(self.clipped_data, self.ds2, self.ds, avg=self.avg)
        ap_ds = rebin(self.convolved_cube, self.ds2, self.ds, avg=self.avg)

        print(data_ds.shape)
        print(self.convolved_cube.shape)
        #print(oops)
        ell_2 = ellipse_fitting(data_ds, self.rfit, self.xell / self.ds, self.yell / self.ds2,
                                self.resolution * self.ds, self.theta_ell, self.q_ell)  # create ellipse mask
        '''  #
        fig = plt.figure()
        ax = plt.gca()
        plt.imshow(ell_2, origin='lower')
        plt.colorbar()
        #plt.show()
        from matplotlib import patches
        e1 = patches.Ellipse((self.xell / self.ds, self.yell / self.ds2), 2 * self.rfit / self.resolution / self.ds,
                             2 * self.rfit / self.resolution * self.q_ell / self.ds2, angle=np.rad2deg(self.theta_ell),
                             linewidth=2, edgecolor='w', fill=False)  # np.rad2deg(params['theta_ell'])
        print(e1)
        #e1.width /= self.ds
        #e1.height /= self.ds
        #e1.x /= self.ds
        #e1.y /= self.ds2
        ax.add_patch(e1)
        print('ell2')
        plt.plot(self.xell / self.ds, self.yell / self.ds, 'w*')
        #plt.plot(self.xell, self.yell, 'w*')
        plt.show()
        # '''  #

        # APPLY THE ELLIPTICAL MASK TO MODEL CUBE & INPUT DATA
        data_ds *= self.ell_ds
        ap_ds *= self.ell_ds
        n_pts = np.sum(self.ell_ds) * len(self.z_ax)  # total number of pixels compared in chi^2 calculation!
        data_2 = data_ds * ell_2  # BUCKET
        ap_2 = ap_ds * ell_2  # BUCKET
        n_2 = np.sum(ell_2) * len(self.z_ax)  # BUCKET

        chi_sq = 0.  # initialize chi^2
        cs = []  # initialize chi^2 per slice
        chi_2 = 0.  # BUCKET

        z_ind = 0  # the actual index for the model-data comparison cubes
        for z in range(self.zrange[0], self.zrange[1]):  # for each relevant freq slice (ignore slices with only noise)
            chi_sq += np.sum((ap_ds[z_ind] - data_ds[z_ind])**2 / self.noise[z_ind]**2)  # calculate chisq!
            cs.append(np.sum((ap_ds[z_ind] - data_ds[z_ind])**2 / self.noise[z_ind]**2))  # chisq per slice
            # np.std(x) = sqrt(mean(abs(x - x.mean())**2))
            chi_2 += np.sum((ap_2[z_ind] - data_2[z_ind])**2 / self.noise[z_ind]**2)  # BUCKET

            z_ind += 1  # the actual index for the model-data comparison cubes

        if not self.quiet:
            print(np.sum(self.ell_ds), len(self.z_ax), n_pts)
            print(r'chi^2=', chi_sq)

        if self.reduced:  # CALCULATE REDUCED CHI^2
            chi_sq /= (n_pts - self.n_params)  # convert to reduced chi^2; else just return full chi^2
            chi_2 /= (n_2 - self.n_params)  # BUCKET
            if not self.quiet:
                print(r'Reduced chi^2=', chi_sq)
                print(n_pts - self.n_params)
                print(r'OR reduced chi^2 = ', chi_2)  # BUCKET
                print(n_2 - self.n_params)  # BUCKET

        if n_pts == 0.:  # PROBLEM WARNING
            print(self.resolution, self.xell, self.yell, self.theta_ell, self.q_ell, self.rfit)
            print('WARNING! STOP! There are no pixels inside the fitting ellipse! n_pts = ' + str(n_pts))
            chi_sq = np.inf

        return chi_sq  # Reduced or Not depending on reduced = True or False


    def line_profiles(self, ix, iy, show_freq=False):  # compare line profiles at the given indices ix, iy
        # RESCALE (x_loc, y_loc) AND (xell, yell) PIXEL VALUES TO CORRESPOND TO SUB-CUBE PIXEL LOCATIONS!
        x_locvb = (self.x_loc - self.xyrange[0]) / self.ds  # x_loc - xi
        y_locvb = (self.y_loc - self.xyrange[2]) / self.ds2  # y_loc - yi

        # SET UP OBSERVATION AXES: initialize x,y axes at 0., with lengths = sub_cube.shape
        y_obs_acvb = np.asarray([0.] * len(self.ell_ds))
        x_obs_acvb = np.asarray([0.] * len(self.ell_ds[0]))

        # Define coordinates to be 0,0 at center of the observed axes (find the central pixel number along each axis)
        for i in range(len(x_obs_acvb)):
            x_obs_acvb[i] = self.resolution * (i - x_locvb) / self.ds  # (arcsec/pix) * N_pix = arcsec
        for i in range(len(y_obs_acvb)):
            y_obs_acvb[i] = self.resolution * (i - y_locvb) / self.ds2

        f_sys = self.f_0 / (1 + self.zred)
        print(ix, iy)
        data_ds = rebin(self.clipped_data, self.ds2, self.ds, avg=self.avg)
        ap_ds = rebin(self.convolved_cube, self.ds2, self.ds, avg=self.avg)

        hdu_m = fits.open(self.data_mask)
        data_mask = hdu_m[0].data  # The mask is stored in hdu_m[0].data, NOT hdu_m[0].data[0]
        hdu_m.close()
        v_width = 2.99792458e5 * (1 + (6454.9 / 2.99792458e5)) * self.fstep / self.f_0  # velocity width [km/s] = c*(1+v/c)*fstep/f0
        mask_ds = rebin(data_mask[self.zrange[0]:self.zrange[1], self.xyrange[2]:self.xyrange[3],
                        self.xyrange[0]:self.xyrange[1]], self.ds2, self.ds, avg=self.avg)

        collapse_flux_v = np.zeros(shape=(len(data_ds[0]), len(data_ds[0][0])))
        for zi in range(len(data_ds)):
            collapse_flux_v += data_ds[zi] * mask_ds[zi] * v_width
            # self.clipped_data[zi] * data_mask[zi, self.xyrange[2]:self.xyrange[3], self.xyrange[0]:self.xyrange[1]]* v_width

        if show_freq:
            plt.plot(self.freq_ax / 1e9, ap_ds[:, iy, ix], 'r*', label=r'Model')
            plt.plot(self.freq_ax / 1e9, data_ds[:, iy, ix], 'k+', label=r'Data')
            plt.plot(self.freq_ax / 1e9, self.noise, 'k--', label=r'Noise (std)')
            plt.axvline(x=f_sys / 1e9, color='k', label=r'$f_{sys}$')
            plt.xlabel(r'Frequency [GHz]')
            plt.show()
        else:
            vel_ax = []
            for v in range(len(self.freq_ax)):
                vel_ax.append(self.c_kms * (1. - (self.freq_ax[v] / self.f_0) * (1 + self.zred)))

            fig, ax = plt.subplots(len(ix), 1, figsize=(6, 12), sharex=True)
            plt.subplots_adjust(hspace=0.)

            for ii in range(len(ix)):
                xp = round(x_obs_acvb[ix[ii]], 3)
                yp = round(y_obs_acvb[iy[ii]], 3)
                dlabel = None
                mlabel = None
                if ii == 0:
                    dlabel = r'Data'
                    mlabel = r'Model'
                norm = np.amax(data_ds[:, iy[ii], ix[ii]])
                ax[ii].fill_between(vel_ax, (data_ds[:, iy[ii], ix[ii]] - self.noise) / norm,
                                    (data_ds[:, iy[ii], ix[ii]] + self.noise) / norm, color='k', step='mid', alpha=0.3)
                #ax[ii].fill_between(vel_ax, data_ds[:, iy[ii], ix[ii]] - self.noise,
                #                    data_ds[:, iy[ii], ix[ii]] + self.noise, color='k', step='mid', alpha=0.2)
                ax[ii].step(vel_ax, data_ds[:, iy[ii], ix[ii]] / norm, color='k', where='mid', label=dlabel)
                ax[ii].step(vel_ax, ap_ds[:, iy[ii], ix[ii]] / norm, color='b', where='mid', label=mlabel)
                # ax[ii].axvline(x=0., color='k', ls='--', label=r'v$_{\text{sys}}$')
                ax[ii].text(-200, 0.9, r'x=' + str(xp) + r'", y=' + str(yp) + r'"')
            ax[-1].set_xlabel(r'Line-of-sight velocity [km/s]')
            # ax[1].set_ylabel(r'Flux Density [arbitrary]')  # [Jy/beam]
            ax[0].legend(loc='upper right')
            fig.text(0.04, 0.5, r'Flux Density [arbitrary]', va='center', rotation='vertical')
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

        fig, ax = plt.subplots(3, 1, sharex=True, figsize=(6,18))
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
        cf = rebin(rebin(self.weight, self.os, self.os), self.ds2, self.ds)[0]  # re-binned weight map, for reference
        plt.imshow(self.ell_ds * cf, origin='lower')  # masked weight map
        plt.title('4x4-binned ellipse * weight map')
        plt.colorbar()
        plt.show()

        plt.imshow(cf, origin='lower')  # re-binned weight map by itself, for reference
        plt.title('4x4-binned weight map')
        plt.colorbar()
        plt.show()


    def vorbinning(self, snr, m1=None, cube=None, filename=None, cbar_lab=None, cmap='RdBu_r'):
        """

        :param snr: target Signal-to-Noise Ratio
        :param m1: moment map to be rebuilt on scale of voronoi-binned map (ie. avg the moment map in each bin)
        :param cube: model cube to be rebuilt on scale of voronoi-binned map (ie. average the line profiles in each bin)
                     and then use that voronoi-binned cube to generate the moment map later
        :param filename: file to which to save XBIN, YBIN, moment map
        :param cbar_lab: colorbar label (Jy km/s beam^-1 for Moment 0, km/s for Moment 1 and Moment 2)
        :param extent: array, extent=[left, right, bottom, top] in data coordinates
        :return:
        """

        # RESCALE (x_loc, y_loc) AND (xell, yell) PIXEL VALUES TO CORRESPOND TO SUB-CUBE PIXEL LOCATIONS!
        x_locvb = self.x_loc - self.xyrange[0]  # x_loc - xi
        y_locvb = self.y_loc - self.xyrange[2]  # y_loc - yi

        # SET UP OBSERVATION AXES: initialize x,y axes at 0., with lengths = sub_cube.shape
        y_obs_acvb = np.asarray([0.] * len(m1))
        x_obs_acvb = np.asarray([0.] * len(m1[0]))

        # Define coordinates to be 0,0 at center of the observed axes (find the central pixel number along each axis)
        for i in range(len(x_obs_acvb)):
            x_obs_acvb[i] = self.resolution * (i - x_locvb)  # (arcsec/pix) * N_pix = arcsec
        for i in range(len(y_obs_acvb)):
            y_obs_acvb[i] = self.resolution * (i - y_locvb)

        hdu_m = fits.open(self.data_mask)
        data_mask = hdu_m[0].data  # The mask is stored in hdu_m[0].data, NOT hdu_m[0].data[0]
        hdu_m.close()

        # estimate the 2D collapsed signal, and estimate a constant noise
        sig = np.zeros(shape=self.input_data[0].shape)
        noi = 0
        for z in range(len(self.input_data)):
            sig += self.input_data[z] * data_mask[z] / len(self.input_data)
            noi += np.mean(self.input_data[z, params['yerr0']:params['yerr1'], params['xerr0']:params['xerr1']]) / \
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
                                                                                  target_snr, plot=0, quiet=1)
        # plt.show()
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
            flattened_binned_m1 = np.zeros(shape=max(binNum) + 1)  # flatten and bin the moment 1 map
            for xy in range(len(x_in)):
                flattened_binned_m1[binNum[xy]] += m1[ypix[xy], xpix[xy]] / nPixels[binNum[xy]]

            # convert the flattened binned moment 1 map into a vector of the same size as the x & y inputs
            full_binned_m1 = np.zeros(shape=len(binNum))
            for xy in range(len(x_in)):
                full_binned_m1[xy] += flattened_binned_m1[binNum[xy]]

            # '''
            if filename is not None:
                dir = '/Users/jonathancohn/Documents/dyn_mod/'
                with open(dir + filename, 'w+') as vb:  # dir+'u2698_moment_vorbin_snr15.txt'
                    vb.write('# targetSN=' + str(target_snr) + '\n')
                    vb.write('#######################\n')
                    vb.write('   XBIN   YBIN   VEL   \n')
                    vb.write('#######################\n')
                    # vb.write('# x y binNum\n')
                    for xy in range(len(x_in)):
                        vb.write(str(x_in[xy]) + ' ' + str(y_in[xy]) + ' ' + str(full_binned_m1[xy]) + '\n')
                        # m1_vb[y_in[xy], x_in[xy]] = full_binned_m1[xy]
                    # for xy in range(len(x_in)):
                    #    vb.write(str(x_in[xy]) + ' ' + str(y_in[xy]) + ' ' + str(binNum[xy]) + ' ' + str(full_binned_m1[xy]) +
                    #             '\n')
                    #    m1_vb[y_in[xy], x_in[xy]] = full_binned_m1[xy]
                # '''

                ###  dlogz 1.1431055564826238 thresh 0.02 nc 518024 niter 9051
            # create the binned moment map for display
            m1_vb = np.zeros(shape=m1.shape)
            for xy in range(len(x_in)):
                m1_vb[ypix[xy], xpix[xy]] = full_binned_m1[xy]

            #plt.imshow(m1_vb, origin='lower', cmap=cmap, extent=[x_obs_acvb[0], x_obs_acvb[-1], y_obs_acvb[0],
            #                                                     y_obs_acvb[-1]])  # plot it!
            #cbar = plt.colorbar()
            #cbar.set_label(cbar_lab, rotation=270., labelpad=20.)
            #plt.xlabel('x [arcsec]')
            #plt.ylabel('y [arcsec]')
            #plt.show()

            return m1_vb


    def just_the_bins(self, snr):
        """

        :param snr: target Signal-to-Noise Ratio
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
            noi += np.mean(self.input_data[z, params['yerr0']:params['yerr1'], params['xerr0']:params['xerr1']]) / \
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
                                                                                  target_snr, plot=0, quiet=1)
        # plt.show()
        print(binNum, sn, nPixels)  # len=# of pix, bin # for each pix; len=# of bins: SNR/bin; len=# of bins: # pix/bin

        return xpix, ypix, binNum, x_in, nPixels


    def vor_moms(self, incl_beam, snr=10, just_data=True):
        """
        Calculate moment maps, average them within voronoi bins
        # using equations from https://www.atnf.csiro.au/people/Tobias.Westmeier/tools_hihelpers.php#moments

        :param incl_beam: True or False; if True, show absolute value of the residual in moment 0 data panel
        :param snr: voronoi binning target signal-to-noise ratio
        :param just_data: if True, plot just the moments of the data (don't include model and residual)
        :return:
        """
        # OPEN STRICTMASK
        hdu_m = fits.open(self.data_mask)
        data_mask = hdu_m[0].data  # The mask is stored in hdu_m[0].data, NOT hdu_m[0].data[0]
        hdu_m.close()

        # CREATE VELOCITY AXIS FROM FREQUENCY AXIS
        vel_ax = []
        velwidth = self.c_kms * (1 + self.zred) * self.fstep / self.f_0
        for v in range(len(self.freq_ax)):
            vel_ax.append(self.c_kms * (1. - (self.freq_ax[v] / self.f_0) * (1 + self.zred)))

        # full cube strictmask, clipped to the appropriate zrange
        clipped_mask = data_mask[self.zrange[0]:self.zrange[1], self.xyrange[2]:self.xyrange[3],
                                 self.xyrange[0]:self.xyrange[1]]

        # CALCULATE MOMENT 0 for data, then for model
        data_masked_m0 = np.zeros(shape=self.ell_mask.shape)
        for z in range(len(vel_ax)):
            data_masked_m0 += abs(velwidth) * self.clipped_data[z] * clipped_mask[z]  # SUM_z data[z] * mask[z] * dz

        model_masked_m0 = np.zeros(shape=(len(self.convolved_cube[0]), len(self.convolved_cube[0][0])))
        for zi in range(len(self.convolved_cube)):
            model_masked_m0 += self.convolved_cube[zi] * abs(velwidth) * clipped_mask[zi]  # SUM_z model[z]*mask[z]*dz

        # CONVERT TO mJy
        data_masked_m0 *= 1e3
        model_masked_m0 *= 1e3

        subtr = 0.
        if incl_beam:  # if including beam overlay
            beam_overlay = np.zeros(shape=self.ell_mask.shape)  # overlay the beam on the same scale as the moment map
            # print(self.beam.shape, beam_overlay.shape)
            # beam_overlay[:self.beam.shape[0], (beam_overlay.shape[1] - self.beam.shape[1]):] = self.beam
            beam_overlay[:self.beam.shape[0] - 6, (beam_overlay.shape[1] - self.beam.shape[1]) + 6:] = self.beam[6:,:-6]
            print(beam_overlay.shape, self.beam.shape)
            beam_overlay *= np.amax(data_masked_m0) / np.amax(beam_overlay)  # scale so beam shows up well on colormap
            data_masked_m0 += beam_overlay  # display on the moment 0 data panel
            subtr = beam_overlay

        # CALCULATE RESIDUAL
        residual_m0 = (data_masked_m0 - subtr) - model_masked_m0

        # AVERAGE EACH MAP WITHING THE VORONOI BIN
        xpix, ypix, binNum, x_in, nPixels = self.just_the_bins(snr=snr)

        # CALCULATE NUMERATOR AND DENOMINATOR USED IN MOMENT 1 & 2, FOR MODEL THEN FOR DATA
        model_numerator = np.zeros(shape=(len(self.convolved_cube[0]), len(self.convolved_cube[0][0])))
        model_denominator = np.zeros(shape=(len(self.convolved_cube[0]), len(self.convolved_cube[0][0])))
        for zi in range(len(self.convolved_cube)):
            model_numerator += vel_ax[zi] * self.convolved_cube[zi] * clipped_mask[zi]
            model_denominator += self.convolved_cube[zi] * clipped_mask[zi]
        model_m1 = model_numerator / model_denominator

        data_numerator = np.zeros(shape=(len(self.convolved_cube[0]), len(self.convolved_cube[0][0])))
        data_denominator = np.zeros(shape=(len(self.convolved_cube[0]), len(self.convolved_cube[0][0])))
        for zi in range(len(self.convolved_cube)):
            data_numerator += vel_ax[zi] * self.clipped_data[zi] * clipped_mask[zi]
            data_denominator += self.clipped_data[zi] * clipped_mask[zi]
        data_m1 = data_numerator / data_denominator
        #print(np.nanmin(data_m1))
        #print(oop)

        # CALCULATE MOMENT 2
        m2_num = np.zeros(shape=(len(self.convolved_cube[0]), len(self.convolved_cube[0][0])))  # numerator
        m2_den = np.zeros(shape=(len(self.convolved_cube[0]), len(self.convolved_cube[0][0])))  # denominator
        for zi in range(len(self.convolved_cube)):
            m2_num += (vel_ax[zi] - model_m1) ** 2 * self.convolved_cube[zi] * clipped_mask[zi]
            m2_den += self.convolved_cube[zi] * clipped_mask[zi]
        model_m2 = np.sqrt(m2_num / m2_den)  # * d1  # BUCKET: no need for MASKING using d1?

        d2_num = np.zeros(shape=(len(self.convolved_cube[0]), len(self.convolved_cube[0][0])))
        d2_n2 = np.zeros(shape=(len(self.convolved_cube[0]), len(self.convolved_cube[0][0])))
        d2_den = np.zeros(shape=(len(self.convolved_cube[0]), len(self.convolved_cube[0][0])))
        for zi in range(len(self.convolved_cube)):
            d2_n2 += self.clipped_data[zi] * (vel_ax[zi] - data_m1) ** 2 * clipped_mask[zi]  # * mask2d
            d2_num += (vel_ax[zi] - data_m1) ** 2 * self.clipped_data[zi] * clipped_mask[zi]  # * mask2d
            d2_den += self.clipped_data[zi] * clipped_mask[zi]  # * mask2d

        d2_num[d2_num < 0] = 0.  # BUCKET ADDING TO GET RID OF NANs
        data_m2 = np.sqrt(d2_num / d2_den)  # * d1  # BUCKET: no need for MASKING using d1?
        data_m2 = np.nan_to_num(data_m2)
        residual_m2 = data_m2 - model_m2

        # AVERAGE EACH MOMENT MAP WITHIN THE VORONOI BINS
        d0 = map_averaging(data_masked_m0, xpix, ypix, binNum, x_in, nPixels)
        m0 = map_averaging(model_masked_m0, xpix, ypix, binNum, x_in, nPixels)
        r0 = map_averaging(residual_m0, xpix, ypix, binNum, x_in, nPixels)
        cbar_0 = r'mJy km/s beam$^{-1}$'  # same for data, model, residual for moment 0
        cmap_0 = 'viridis'  # same for data, model, residual for moment 0
        min0 = np.amin([np.nanmin(m0), np.nanmin(d0)])
        max0 = np.amax([np.nanmax(m0), np.nanmax(d0)])

        data_m1[np.abs(data_m1) > 1e3] = 0  # get rid of edge effects
        residual_m1 = (data_m1 - subtr) - model_m1  # calculate residual
        d1 = map_averaging(data_m1, xpix, ypix, binNum, x_in, nPixels)
        m1 = map_averaging(model_m1, xpix, ypix, binNum, x_in, nPixels)
        r1 = map_averaging(residual_m1, xpix, ypix, binNum, x_in, nPixels)
        cbar_1 = r'km/s'  # same for data, model, residual for moment 1
        cmap_dm1 = 'RdBu_r'  # for data, model for moment 1
        cmap_r1 = 'viridis'  # for residual for moment 1
        min1 = np.amin([np.nanmin(m1), np.nanmin(d1)])
        max1 = np.amax([np.nanmax(m1), np.nanmax(d1)])

        d2 = map_averaging(data_m2, xpix, ypix, binNum, x_in, nPixels)
        m2 = map_averaging(model_m2, xpix, ypix, binNum, x_in, nPixels)
        r2 = map_averaging(residual_m2, xpix, ypix, binNum, x_in, nPixels)
        cbar_2 = r'km/s'  # same for data, model, residual for moment 2
        cmap_2 = 'viridis'  # same for data, model, residual for moment 2
        min2 = np.amin([np.nanmin(m2), np.nanmin(d2)])
        max2 = np.amax([np.nanmax(m2), np.nanmax(d2)])

        # CALCULATE SUB-CUBE ARCSEC EXTENT
        # RESCALE (x_loc, y_loc) AND (xell, yell) PIXEL VALUES TO CORRESPOND TO SUB-CUBE PIXEL LOCATIONS!
        x_locvb = self.x_loc - self.xyrange[0]  # x_loc - xi
        y_locvb = self.y_loc - self.xyrange[2]  # y_loc - yi

        # SET UP OBSERVATION AXES: initialize x,y axes at 0., with lengths = sub_cube.shape
        y_obs_acvb = np.asarray([0.] * len(m1))
        x_obs_acvb = np.asarray([0.] * len(m1[0]))

        # Define coordinates to be 0,0 at center of the observed axes (find the central pixel number along each axis)
        for i in range(len(x_obs_acvb)):
            x_obs_acvb[i] = self.resolution * (i - x_locvb)  # (arcsec/pix) * N_pix = arcsec
        for i in range(len(y_obs_acvb)):
            y_obs_acvb[i] = self.resolution * (i - y_locvb)

        extent = [x_obs_acvb[0], x_obs_acvb[-1], y_obs_acvb[0], y_obs_acvb[-1]]  # left right bottom top

        if just_data:
            fig, ax = plt.subplots(3, 1, figsize=(6, 18))  # rows, cols, figsize=(width, height)
            plt.subplots_adjust(hspace=0.02, wspace=0.02)

            # PLOT MOMENT 0
            imd0 = ax[0].imshow(d0, vmin=min0, vmax=max0, origin='lower', extent=extent, cmap=cmap_0)
            cbard0 = fig.colorbar(imd0, ax=ax[0], pad=0.02)
            cbard0.set_label(cbar_0, rotation=270, labelpad=20.)

            ax[0].set_xticklabels([])
            # ax[0].set_yticklabels([])
            ax[0].set_ylabel(r'y [arcsec]', fontsize=20)  # y [arcsec]

            # PLOT MOMENT 1
            imd1 = ax[1].imshow(d1, vmin=min1, vmax=max1, origin='lower', extent=extent, cmap=cmap_dm1)
            cbard1 = fig.colorbar(imd1, ax=ax[1], pad=0.02)
            cbard1.set_label(cbar_1, rotation=270, labelpad=20.)

            ax[1].set_xticklabels([])
            # ax[1].set_yticklabels([])
            ax[1].set_ylabel(r'y [arcsec]', fontsize=20)  # y [arcsec]

            # PLOT MOMENT 2
            imd2 = ax[2].imshow(d2, vmin=min2, vmax=max2, origin='lower', extent=extent, cmap=cmap_2)
            cbard2 = fig.colorbar(imd2, ax=ax[2], pad=0.02)
            cbard2.set_label(cbar_2, rotation=270, labelpad=20.)

            #ax[2].set_xticklabels([])
            #ax[2].set_yticklabels([])
            ax[2].set_xlabel(r'x [arcsec]', fontsize=20)  # x [arcsec]
            ax[2].set_ylabel(r'y [arcsec]', fontsize=20)  # y [arcsec]

            plt.show()

        else:
            # START PLOTTING
            fig, ax = plt.subplots(3, 3, figsize=(12,8))
            plt.subplots_adjust(hspace=0.02, wspace=0.02)

            # PLOT MOMENT 0
            imd0 = ax[0][0].imshow(d0, vmin=min0, vmax=max0, origin='lower', extent=extent, cmap=cmap_0)
            cbard0 = fig.colorbar(imd0, ax=ax[0][0], pad=0.02)
            #cbard0.set_label(cbar_0, rotation=270, labelpad=20.)

            imm0 = ax[0][1].imshow(m0, vmin=min0, vmax=max0, origin='lower', extent=extent, cmap=cmap_0)
            cbarm0 = fig.colorbar(imm0, ax=ax[0][1], pad=0.02)
            #cbarm0.set_label(cbar_0, rotation=270, labelpad=20.)

            imr0 = ax[0][2].imshow(r0, origin='lower', vmin=np.nanmin(r0), vmax=np.nanmax(r0), extent=extent, cmap=cmap_0)
            cbar2r0 = fig.colorbar(imr0, ax=ax[0][2], pad=0.02)
            cbar2r0.set_label(cbar_0, rotation=270, labelpad=20.)

            ax[0][0].set_xticklabels([])
            #ax[0][0].set_yticklabels([])
            ax[0][1].set_xticklabels([])
            ax[0][1].set_yticklabels([])
            ax[0][2].set_xticklabels([])
            ax[0][2].set_yticklabels([])
            #ax[0][0].set_xlabel(r'x [arcsec]', fontsize=20)  # x [arcsec]
            ax[0][0].set_ylabel(r'y [arcsec]', fontsize=20)  # y [arcsec]
            #ax[0][1].set_xlabel(r'x [arcsec]', fontsize=20)  # y [arcsec]
            #ax[0][1].set_ylabel(r'y [arcsec]', fontsize=20)  # y [arcsec]
            #ax[0][2].set_xlabel(r'x [arcsec]', fontsize=20)  # x [arcsec]
            #ax[0][2].set_ylabel(r'y [arcsec]', fontsize=20)  # y [arcsec]


            # PLOT MOMENT 1
            imd1 = ax[1][0].imshow(d1, vmin=min1, vmax=max1, origin='lower', extent=extent, cmap=cmap_dm1)
            cbard1 = fig.colorbar(imd1, ax=ax[1][0], pad=0.02)
            #cbard1.set_label(cbar_1, rotation=270, labelpad=20.)

            imm1 = ax[1][1].imshow(m1, vmin=min1, vmax=max1, origin='lower', extent=extent, cmap=cmap_dm1)
            cbarm1 = fig.colorbar(imm1, ax=ax[1][1], pad=0.02)
            #cbarm1.set_label(cbar_1, rotation=270, labelpad=20.)

            imr1 = ax[1][2].imshow(r1, origin='lower', vmin=np.nanmin(r1), vmax=np.nanmax(r1), extent=extent, cmap=cmap_r1)
            cbarr1 = fig.colorbar(imr1, ax=ax[1][2], pad=0.02)
            cbarr1.set_label(cbar_1, rotation=270, labelpad=20.)

            ax[1][0].set_xticklabels([])
            #ax[1][0].set_yticklabels([])
            ax[1][1].set_xticklabels([])
            ax[1][1].set_yticklabels([])
            ax[1][2].set_xticklabels([])
            ax[1][2].set_yticklabels([])
            #ax[1][0].set_xlabel(r'x [arcsec]', fontsize=20)  # x [arcsec]
            ax[1][0].set_ylabel(r'y [arcsec]', fontsize=20)  # y [arcsec]
            #ax[1][1].set_xlabel(r'x [arcsec]', fontsize=20)  # y [arcsec]
            #ax[1][1].set_ylabel(r'y [arcsec]', fontsize=20)  # y [arcsec]
            #ax[1][2].set_xlabel(r'x [arcsec]', fontsize=20)  # x [arcsec]
            #ax[1][2].set_ylabel(r'y [arcsec]', fontsize=20)  # y [arcsec]


            # PLOT MOMENT 2
            imd2 = ax[2][0].imshow(d2, vmin=min2, vmax=max2, origin='lower', extent=extent, cmap=cmap_2)
            cbard2 = fig.colorbar(imd2, ax=ax[2][0], pad=0.02)
            #cbard2.set_label(cbar_2, rotation=270, labelpad=20.)

            imm2 = ax[2][1].imshow(m2, vmin=min2, vmax=max2, origin='lower', extent=extent, cmap=cmap_2)
            cbarm2 = fig.colorbar(imm2, ax=ax[2][1], pad=0.02)
            #cbarm2.set_label(cbar_2, rotation=270, labelpad=20.)

            imr2 = ax[2][2].imshow(r2, origin='lower', vmin=np.nanmin(r2), vmax=np.nanmax(r2), extent=extent, cmap=cmap_2)
            cbarr2 = fig.colorbar(imr2, ax=ax[2][2], pad=0.02)
            cbarr2.set_label(cbar_2, rotation=270, labelpad=20.)

            #ax[2][1].set_xticklabels([])
            ax[2][1].set_yticklabels([])
            #ax[2][2].set_xticklabels([])
            ax[2][2].set_yticklabels([])
            ax[2][0].set_xlabel(r'x [arcsec]', fontsize=20)  # x [arcsec]
            ax[2][0].set_ylabel(r'y [arcsec]', fontsize=20)  # y [arcsec]
            ax[2][1].set_xlabel(r'x [arcsec]', fontsize=20)  # y [arcsec]
            #ax[2][1].set_ylabel(r'y [arcsec]', fontsize=20)  # y [arcsec]
            ax[2][2].set_xlabel(r'x [arcsec]', fontsize=20)  # x [arcsec]
            #ax[2][2].set_ylabel(r'y [arcsec]', fontsize=20)  # y [arcsec]

            plt.show()




    def moment_0(self, abs_diff, incl_beam, norm, samescale=False, snr=10):
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

        self.vorbinning(m1=data_masked_m0, snr=snr, filename=None, cbar_lab=r'mJy km/s beam$^{-1}$', cmap='viridis')
        self.vorbinning(m1=model_masked_m0, snr=snr, filename=None, cbar_lab=r'mJy km/s beam$^{-1}$', cmap='viridis')
        self.vorbinning(m1=diff, snr=snr, filename=None, cbar_lab=r'mJy km/s beam$^{-1}$', cmap='viridis')


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

            d2 = np.nan_to_num(d2)
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
            self.vorbinning(m1=np.nan_to_num(d2), snr=snr, filename=None, cbar_lab='km/s', cmap='viridis')
            self.vorbinning(m1=m2, snr=snr, filename=None, cbar_lab='km/s', cmap='viridis')
            self.vorbinning(m1=diff, snr=snr, filename=None, cbar_lab='km/s', cmap='viridis')


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
            self.vorbinning(m1=data_mom, snr=snr, filename=None, cbar_lab='km/s', cmap='RdBu_r')
            self.vorbinning(m1=model_mom, snr=snr, filename=None, cbar_lab='km/s', cmap='RdBu_r')
            self.vorbinning(m1=diff, snr=snr, filename=None, cbar_lab='km/s', cmap='viridis')
            #print(oop)

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

    if 'ds2' not in params:
        params['ds2'] = params['ds']

    # DECIDE HERE WHETHER TO AVG IN THE REBIN() FUNCTION (avging=True) OR SUM (avging=False)
    avging = False

    # CREATE THINGS THAT ONLY NEED TO BE CALCULATED ONCE (collapse fluxes, lucy, noise)
    mod_ins = model_prep(data=params['data'], ds=params['ds'], ds2=params['ds2'], lucy_out=params['lucy'],
                         lucy_mask=params['lucy_mask'],
                         lucy_b=params['lucy_b'], lucy_in=params['lucy_in'], lucy_it=params['lucy_it'],
                         data_mask=params['mask'], grid_size=params['gsize'], res=params['resolution'],
                         x_std=params['x_fwhm'], y_std=params['y_fwhm'], pa=params['PAbeam'],
                         xyerr=[params['xerr0'], params['xerr1'], params['yerr0'], params['yerr1']],
                         zrange=[params['zi'], params['zf']], avg=avging, q_ell=params['q_ell'],
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
    inc_fixed = np.deg2rad(67.7)  # based on fiducial model (67.68 deg)
    vcg_in = None
    if params['incl_gas'] == 'True':
        zfix_u2698 = 0.02152
        zfix_p11179 = 0.0224706  # https://califaserv.caha.es/CALIFA_WEB/public_html/?q=content/califa-explorer-v01&califaid=9007
        zfix_n384 = 0.01412
        vcg_in = gas_vel(params['resolution'], co_ell_rad, co_ell_sb, params['dist'], f_0, inc_fixed, zfixed=zfix_u2698)

    out = params['outname']
    t0m = time.time()
    mg = ModelGrid(resolution=params['resolution'], os=params['os'], x_loc=params['xloc'], y_loc=params['yloc'],
                   mbh=params['mbh'], inc=np.deg2rad(params['inc']), vsys=params['vsys'], dist=params['dist'],
                   theta=np.deg2rad(params['PAdisk']), input_data=input_data, lucy_out=lucy_out, out_name=out,
                   beam=beam, rfit=params['rfit'], enclosed_mass=params['mass'], ml_ratio=params['ml_ratio'],
                   sig_type=params['s_type'], zrange=[params['zi'], params['zf']], menc_type=params['mtype'],
                   sig_params=[params['sig0'], params['r0'], params['mu'], params['sig1']], f_w=params['f'],
                   ds=params['ds'], ds2=params['ds2'], noise=noise, reduced=True, freq_ax=freq_ax, q_ell=params['q_ell'],
                   theta_ell=np.deg2rad(params['theta_ell']), xell=params['xell'], yell=params['yell'], fstep=fstep,
                   f_0=f_0, bl=params['bl'], xyrange=[params['xi'], params['xf'], params['yi'], params['yf']],
                   n_params=n_free, data_mask=params['mask'], incl_gas=params['incl_gas']=='True', vrad=params['vrad'],
                   kappa=params['kappa'], omega=params['omega'], co_rad=co_ell_rad, co_sb=co_ell_sb, avg=avging,
                   pvd_width=(params['x_fwhm']*params['y_fwhm'])/params['resolution']/2., vcg_func=vcg_in)

    # x_fwhm=0.197045, y_fwhm=0.103544 -> geometric mean = sqrt(0.197045*0.103544) = 0.142838; regular mean = 0.1502945
    mg.grids()
    mg.convolution()
    chi_sq = mg.chi2()
    # mg.vor_moms(incl_beam=True, snr=10, just_data=False)
    mg.vor_moms(incl_beam=False, snr=10, just_data=False)
    mg.pvd()
    print(oop)
    # LP at 16,11 is great!
    xs = [7, 14, 15]
    ys = [5, 9, 10]
    xs2 = [9, 8, 13, 13]
    ys2 = [7, 8, 8, 9]
    xs3 = [4, 10, 12]
    ys3 = [6, 8, 8]
    xtest = [4, 7, 9, 14, 16, 13, 16]  # ok, good, meh, good, great, okay, okay
    ytest = [6, 5, 7, 9, 11, 11, 8]
    xtalk = [4, 7, 13, 16]
    ytalk = [6, 5, 11, 11]
    mg.line_profiles(xtalk, ytalk)
    #mg.vor_moms(incl_beam=True, snr=10)
    #mg.vor_moms(incl_beam=False, snr=10)
    #mg.pvd()
    print(oop)
    #mg.moment_0(abs_diff=False, incl_beam=True, norm=False)
    #mg.moment_12(abs_diff=False, incl_beam=False, norm=False, mom=1)
    #mg.moment_12(abs_diff=False, incl_beam=False, norm=False, mom=2)

    mg.line_profiles(13, 8)
    mg.line_profiles(9, 7)
    mg.line_profiles(9, 8)
    mg.line_profiles(8, 8)
    mg.line_profiles(13, 9)
    print(oop)

    mg.line_profiles(10, 8)  # center?
    mg.line_profiles(11, 8)  # center?
    mg.line_profiles(12, 8)  # red?
    mg.line_profiles(9, 7)  # blue?
    #mg.line_profiles(14, 8)  # decent red [was using this]
    #mg.line_profiles(14, 10)  # decent red [was using this]
    #mg.line_profiles(15, 9)  # good red

    # Good examples
    #mg.line_profiles(7, 5)  # decent blue [recently using this]
    #mg.line_profiles(14, 9)  # good red [recently using this]
    #mg.line_profiles(15, 10)  # decent red [recently using this]

    # meh examples
    #mg.line_profiles(4, 6)  # blue orig (not great)
    #mg.line_profiles(6, 6)  # blue okay? (meh)
    #mg.line_profiles(10, 9)  # near ctr orig (meh)
    #mg.line_profiles(13, 8)  # red [not bad]

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
