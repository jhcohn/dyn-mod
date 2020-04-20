import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt
from matplotlib import rc
from scipy import integrate, signal, interpolate
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

        return params, priors, qobs, nfree

    else:
        return params, priors, nfree


def model_prep(lucy_out=None, lucy_mask=None, lucy_b=None, lucy_in=None, lucy_it=None, data=None, data_mask=None,
               grid_size=None, res=1., x_std=1., y_std=1., pa=0., ds=4, zrange=None, xyerr=None):
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

    :return: lucy mask, lucy output, synthesized beam, flux map, frequency axis, f_0, freq step, input data cube
    """

    # If the lucy process hasn't been done yet, and the mask cube also hasn't been collapsed yet, create collapsed mask
    if not Path(lucy_out).exists() and not Path(lucy_mask).exists():
        hdu_m = fits.open(data_mask)
        fullmask = hdu_m[0].data  # The mask is stored in hdu_m[0].data, NOT hdu_m[0].data[0]
        print(fullmask.shape)
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

    noise_4 = rebin(input_data, ds)
    noise = []  # ESTIMATE NOISE (RMS) IN ORIGINAL DATA CUBE [z, y, x]  # For large N, Variance ~= std^2
    for z in range(zrange[0], zrange[1]):  # for each relevant freq slice
        noise.append(np.std(noise_4[z, xyerr[2]:xyerr[3], xyerr[0]:xyerr[1]]))  # ~variance
        #         noise.append(np.std(noise_4[z, int(xyerr[2]/ds):int(xyerr[3]/ds), int(xyerr[0]/ds):int(xyerr[1]/ds)]))

    return lucy_mask, lucy_out, beam, fluxes, freq_ax, f_0, fstep, input_data, noise


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


class ModelGrid:

    def __init__(self, resolution=0.05, os=4, x_loc=0., y_loc=0., mbh=4e8, inc=np.deg2rad(60.), vsys=None, vrad=0.,
                 dist=17., theta=np.deg2rad(200.), input_data=None, lucy_out=None, out_name=None, beam=None, rfit=1.,
                 q_ell=1., theta_ell=0., xell=360., yell=350., bl=False, enclosed_mass=None, menc_type=0,
                 ml_ratio=1., sig_type='flat', sig_params=None, f_w=1., noise=None, ds=None, zrange=None, xyrange=None,
                 reduced=False, freq_ax=None, f_0=0., fstep=0., opt=True, quiet=False, n_params=12):
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
        self.resolution = resolution
        self.os = os
        self.x_loc = x_loc
        self.y_loc = y_loc
        self.mbh = mbh
        self.inc = inc
        self.vsys = vsys
        self.vrad = vrad
        self.dist = dist
        self.theta = theta
        self.input_data = input_data
        self.lucy_out = lucy_out
        self.out_name = out_name
        self.beam = beam
        self.rfit = rfit
        self.q_ell = q_ell
        self.theta_ell = theta_ell
        self.xell = xell
        self.yell = yell
        self.bl = bl
        self.enclosed_mass = enclosed_mass
        self.menc_type = menc_type
        self.ml_ratio = ml_ratio
        self.sig_type = sig_type
        self.sig_params = sig_params
        self.f_w = f_w
        self.noise = noise
        self.ds = ds
        self.zrange = zrange
        self.xyrange = xyrange
        self.reduced = reduced
        self.freq_ax = freq_ax
        self.f_0 = f_0
        self.fstep = fstep
        self.opt = opt
        self.quiet = quiet
        self.n_params = n_params
        # Parameters to be built in create_grid() or convolve_cube() functions inside the class
        self.weight = None
        self.z_ax = None
        self.freq_obs = None
        self.zred = None
        self.input_data_masked = None
        self.delta_freq_obs = None
        self.convolved_cube = None
    """
    Build grid for dynamical modeling!
    
    Class structure following: https://www.w3schools.com/python/python_classes.asp

    :param resolution: resolution of observations [arcsec/pixel]
    :param os: oversampling factor
    :param x_loc: the location of the BH, as measured along the x axis of the data cube [pixels]
    :param y_loc: the location of the BH, as measured along the y axis of the data cube [pixels]
    :param mbh: supermassive black hole mass [solar masses]
    :param inc: inclination of the galaxy [radians]
    :param vsys: if given, the systemic velocity [km/s]
    :param vrad: radial inflow term [km/s]
    :param dist: distance to the galaxy [Mpc]
    :param theta: angle from the +x_obs axis counterclockwise to the blueshifted side of the disk (-x_disk) [radians]
    :param input_data: input data cube of observations
    :param lucy_out: output from running lucy on data cube and beam PSF
    :param out_name: output name of the fits file to which to save the output v_los image (if None, don't save image)
    :param beam: synthesized alma beam (output from model_ins)
    :param q_ell: axis ratio q of fitting ellipse [unitless]
    :param theta_ell: same as theta above, but held fixed during model fitting; used for the ellipse fitting region
    :param xell: same as x_loc above, but held fixed during model fitting; used for the ellipse fitting region
    :param yell: same as y_loc above, but held fixed during model fitting; used for the ellipse fitting region
    :param enclosed_mass: file including data of the enclosed stellar mass of the galaxy (1st column should be radius
        r in kpc and second column should be M_stars(<r) or velocity_circ(R) due to stars within R)
    :param menc_type: Select how you incorporate enclosed stellar mass [0 for MGE; 1 for v(R) file; 2 for M(R) file]
    :param ml_ratio: The mass-to-light ratio of the galaxy
    :param sig_type: code for the type of sigma_turb we're using. Can be 'flat', 'exp', or 'gauss'
    :param sig_params: list of parameters to be plugged into the get_sig() function. Number needed varies by sig_type
    :param f_w: multiplicative weight factor for the line profiles
    :param noise: array of the estimated (ds x ds-binned) noise per slice (within the desired zrange)
    :param rfit: disk radius within which we will compare model and data [arcsec]
    :param ds: downsampling factor to use when averaging pixels together for actual model-data comparison
    :param chi2: if True, record chi^2 of model vs data! If False, ignore, and return convolved model cube
    :param zrange: the slices of the data cube where the data shows up (i.e. isn't just noise) [zi, zf]
    :param xyrange: the subset of the cube (in pixels) that we actually want to run data on [xi, xf, yi, yf]
    :param reduced: if True, return reduced chi^2 instead of regular chi^2 (requires chi2=True, too)
    :param freq_ax: array of the frequency axis in the data cube, from bluest to reddest frequency [Hz]
    :param f_0: rest frequency of the observed line in the data cube [Hz]
    :param fstep: frequency step in the frequency axis [Hz]
    :param bl: lucy weight map unit indicator (bl=False or 0 --> Jy/beam * Hz; bl=True or 1 --> Jy/beam km/s)
    :param opt: frequency axis velocity convention; if opt=True, optical velocity; if opt=False, radio velocity
    :param quiet: suppress printing out stuff!
    :param n_params: number of free parameters being fit (from param file!)

    Class with functions:
        create_grid: creates model grid from input params, calculates weight map, freq map, and delta-freq map
        convolve_cube: requires create_grid to be run first; convolve the model cube with the ALMA beam
        chi2: requires create_grid and convolve_cube to be run first; calculates chi2 and/or reduced chi2
        output_cube: not actively using; requires create_frid and convolve_cube to be run first; create output cube file        
        # BUCKET TO DO: include moment map functions too!
    """

    def create_grid(self):
        t_grid = time.time()

        # SUBPIXELS (reshape deconvolved flux map [lucy_out] sxs subpixels, so subpix has flux=(real pixel flux)/s**2)
        if not self.quiet:
            print('start')
        if self.os == 1:  # subpix_deconvolved is identical to lucy_out, with sxs subpixels per pixel & total flux conserved
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
        x_obs = np.asarray([self.dist * 10 ** 6 * np.tan(x / self.arcsec_per_rad) for x in x_obs_ac])  # 206265 arcsec/rad
        y_obs = np.asarray([self.dist * 10 ** 6 * np.tan(y / self.arcsec_per_rad) for y in y_obs_ac])  # 206265 arcsec/rad

        # CONVERT FROM x_obs, y_obs TO 2D ARRAYS OF x_disk, y_disk [arcsec] and [pc] versions
        x_disk_ac = (x_obs_ac[None, :] - x_bh_ac) * np.cos(self.theta) +\
                    (y_obs_ac[:, None] - y_bh_ac) * np.sin(self.theta)  # arcsec
        y_disk_ac = (y_obs_ac[:, None] - y_bh_ac) * np.cos(self.theta) -\
                    (x_obs_ac[None, :] - x_bh_ac) * np.sin(self.theta)  # arcsec
        x_disk = (x_obs[None, :] - x_bhoff) * np.cos(self.theta) + (y_obs[:, None] - y_bhoff) * np.sin(self.theta)  # pc
        y_disk = (y_obs[:, None] - y_bhoff) * np.cos(self.theta) - (x_obs[None, :] - x_bhoff) * np.sin(self.theta)  # pc

        # CALCULATE THE RADIUS (R) OF EACH POINT (x_disk, y_disk) IN THE DISK [pc]
        R_ac = np.sqrt((y_disk_ac ** 2 / np.cos(self.inc) ** 2) + x_disk_ac ** 2)  # radius R (arcsec)
        R = np.sqrt((y_disk ** 2 / np.cos(self.inc) ** 2) + x_disk ** 2)  # radius of each point in the disk (2d array)

        # CALCULATE KEPLERIAN VELOCITY DUE TO ENCLOSED STELLAR MASS
        if self.menc_type == 0:  # if calculating v(R) due to stars directly from MGE parameters
            if not self.quiet:
                print('mge')
            test_rad = np.linspace(np.amin(R_ac), np.amax(R_ac), 100)  # create an array of test radii

            comp, surf_pots, sigma_pots, qobs = mvm.load_mge(self.enclosed_mass)  # load the MGE parameters

            v_c = mvm.mge_vcirc(surf_pots * self.ml_ratio, sigma_pots, qobs, np.rad2deg(self.inc), 0., self.dist,
                                test_rad)  # calculate v_c,star
            v_c_r = interpolate.interp1d(test_rad, v_c, kind='cubic', fill_value='extrapolate')  # interpolate v_c(R)

            # CALCULATE KEPLERIAN VELOCITY OF ANY POINT (x_disk, y_disk) IN THE DISK WITH RADIUS R (km/s)
            vel = np.sqrt((self.G_pc * self.mbh / R) + v_c_r(R_ac)**2)
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
            vel = np.sqrt(v_c_r(R) * self.ml_ratio + (self.G_pc * self.mbh / R))  # velocities sum in quadrature

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
            vel = np.sqrt(self.G_pc * m_R / R)  # Keplerian velocity vel at each point in the disk

        # CALCULATE LINE-OF-SIGHT VELOCITY AT EACH POINT (x_disk, y_disk) IN THE DISK (km/s)
        alpha = abs(np.arctan(y_disk / (np.cos(self.inc) * x_disk)))  # measure alpha from +x (minor ax) to +y (maj ax)
        sign = x_disk / abs(x_disk)  # (+x now back to redshifted side, so don't need extra minus sign back in front!)
        v_los = sign * abs(vel * np.cos(alpha) * np.sin(self.inc))  # v_los > 0 -> redshift; v_los < 0 -> blueshift

        # INCLUDE NEW RADIAL VELOCITY TERM
        vrad_sign = y_disk / abs(y_disk)  # With this sign convention: vrad > 0 -> outflow; vrad < 0 -> inflow!
        v_los += self.vrad * vrad_sign * abs(np.sin(alpha) * np.sin(self.inc))  # See notebook for derivation!

        # SET LINE-OF-SIGHT VELOCITY AT THE BLACK HOLE CENTER TO BE 0, SUCH THAT IT DOES NOT BLOW UP
        center = (R == 0.)  # Doing this is only relevant if we have pixel located exactly at the center
        v_los[center] = 0.  # if any point is at x_disk, y_disk = (0., 0.), set velocity there = 0.

        # CALCULATE VELOCITY PROFILES
        sigma = get_sig(r=R, sig0=self.sig_params[0], r0=self.sig_params[1], mu=self.sig_params[2],
                        sig1=self.sig_params[3])[self.sig_type]

        # CONVERT v_los TO OBSERVED FREQUENCY MAP
        self.zred = self.vsys / self.c_kms  # redshift
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

        # print(self.c_kms * (1 - (fstep / f_0) * (1+zred)))

        # BEN_LUCY COMPARISON ONLY (only use for comparing to model with Ben's lucy map, which is in different units)
        if self.bl:  # (multiply by vel_step because Ben's lucy map is actually in Jy/beam km/s units)
            self.weight *= self.fstep * 6.783  # NOTE: 6.783 == beam size / channel width in km/s
            # channel width = 1.537983987579E+07 Hz --> v_width = c * (1 - f_0 / (fstep / (1+zred)))
            # 2698: channel width = 1.537983987576E+07 --> vwidth = 3e5 * (1 -

        if not self.quiet:
            print(str(time.time() - t_grid) + ' seconds in create_grid()')


    def convolve_cube(self):
        # BUILD GAUSSIAN LINE PROFILES!!!
        cube_model = np.zeros(shape=(len(self.freq_ax), len(self.freq_obs), len(self.freq_obs[0])))  # initialize model cube
        for fr in range(len(self.freq_ax)):
            cube_model[fr] = self.weight * np.exp(-(self.freq_ax[fr] - self.freq_obs) ** 2 /
                                                  (2 * self.delta_freq_obs ** 2))

        # RE-SAMPLE BACK TO CORRECT PIXEL SCALE (take avg of sxs sub-pixels for real alma pixel) --> intrinsic data cube
        if self.os == 1:
            intrinsic_cube = cube_model
        else:
            intrinsic_cube = rebin(cube_model, self.os)  # intrinsic_cube = block_reduce(cube_model, s, np.mean)

        tc = time.time()
        # CONVERT INTRINSIC TO OBSERVED (convolve each slice of intrinsic_cube with ALMA beam --> observed data cube)
        self.convolved_cube = np.zeros(shape=intrinsic_cube.shape)  # approx ~1e-6 to 3e-6s per pixel
        for z in range(len(self.z_ax)):
            self.convolved_cube[z, :, :] = convolution.convolve(intrinsic_cube[z, :, :], self.beam)
        print('convolution loop ' + str(time.time() - tc))


    def chi2(self):
        # ONLY WANT TO FIT WITHIN ELLIPTICAL REGION! APPLY ELLIPTICAL MASK TO MODEL CUBE & INPUT DATA
        ell_mask = ellipse_fitting(self.convolved_cube, self.rfit, self.xell, self.yell, self.resolution,
                                   self.theta_ell, self.q_ell)  # create ellipse mask
        unmasked_model = self.convolved_cube
        unmasked_data = self.input_data[self.zrange[0]:self.zrange[1], self.xyrange[2]:self.xyrange[3],
                                 self.xyrange[0]:self.xyrange[1]]
        self.convolved_cube *= ell_mask  # mask the convolved model cube
        self.input_data_masked = self.input_data[self.zrange[0]:self.zrange[1], self.xyrange[2]:self.xyrange[3],
                                 self.xyrange[0]:self.xyrange[1]] * ell_mask  # mask the input data cube

        ell_4 = rebin(ell_mask, self.ds)  # rebin the mask by the down-sampling factor
        all_pix = np.ndarray.flatten(ell_4)  # all pixels in each slice
        masked_pix = all_pix[all_pix >= 8]  # all_pix, but this time only the pixels that are "inside" the ellipse
        n_pts = len(masked_pix) * len(self.z_ax)  # number of total pixels to be compared in chi^2 calculation!
        print(len(masked_pix), len(self.z_ax), n_pts)

        data_4u = rebin(unmasked_data, self.ds)
        ap_4u = rebin(unmasked_model, self.ds)
        ell_4[ell_4 < 8] = 0.
        ell_4u = np.nan_to_num(ell_4 / np.abs(ell_4))
        data_4u *= ell_4u
        ap_4u *= ell_4u[0]
        npu = np.sum(ell_4u) * len(self.z_ax)
        print(npu)  # == n_pts, good!
        #plt.imshow(ell_4[0], origin='lower')
        #plt.colorbar()
        #plt.show()

        chi_sq = 0.  # initialize chi^2
        cs = []  # initialize chi^2 per slice

        # DOWN-SAMPLING: compare the data to the model by binning each in groups of dsxds pixels (separate from os)
        data_4 = rebin(self.input_data_masked, self.ds)
        ap_4 = rebin(self.convolved_cube, self.ds)

        # DEFINE THE ELLIPSE MASK ON THE DOWN-SAMPLED PIXEL SCALE!
        ell_mask = ellipse_fitting(ap_4, self.rfit, self.xell / self.ds, self.yell / self.ds, self.resolution * self.ds,
                                   self.theta_ell, self.q_ell)
        ell_4 = ell_mask
        # THE BELOW IMAGES SUGGEST THAT THE HIGHER-UP-DEFINED ell_4 WORKS BETTER
        #print(np.sum(ell_mask))
        #cf = rebin(rebin(self.weight, self.ds), self.ds)[0]
        #plt.imshow(ell_mask * cf, origin='lower')  # ell_mask
        #plt.title('Ellipse constructed on 4x4-binned grid')
        #plt.colorbar()
        #plt.show()

        #plt.imshow(ell_4[0] * cf, origin='lower')  # ell_mask
        #plt.title('4x4-binned ellipse')
        #plt.colorbar()
        #plt.show()
        #print(oop)


        z_ind = 0  # the actual index for the model-data comparison cubes
        for z in range(self.zrange[0], self.zrange[1]):  # for each relevant freq slice (ignore slices with only noise)
            chi_sq += np.sum((ap_4u[z_ind] - data_4u[z_ind])**2 / self.noise[z_ind]**2)  # calculate chisq!
            cs.append(np.sum((ap_4u[z_ind] - data_4u[z_ind])**2 / self.noise[z_ind]**2))  # chisq per slice

            z_ind += 1  # the actual index for the model-data comparison cubes

        # CALCULATE REDUCED CHI^2
        all_pix = np.ndarray.flatten(ell_4)  # all fitted pixels in each slice [len = 625 (yep)] [525 masked, 100 not]
        # masked_pix = all_pix[all_pix >= 8]  # all_pix, but this time only the pixels that are actually inside ellipse
        masked_pix = all_pix[all_pix != 0]  # all_pix, but this time only the pixels that are actually inside ellipse
        n_pts = len(masked_pix) * len(self.z_ax)  # (zrange[1] - zrange[0])  # total number of pixels being compared
        print(n_pts, len(masked_pix))  # rfit=1.2 --> 6440 (140/slice); 6716 (146/slice) for fully inclusive ellipse
        # print(oop)  # rfit=1.05 --> 4968 (108/slice)  # rfit=1.136 --> 5704 (124/slice) (see 16 April 2018 meeting)
        print(r'chi^2=', chi_sq)
        print(chi_sq / (npu - self.n_params))
        # if not quiet:
            # print('total model constructed in {0} seconds'.format(time.time() - t_begin))  # ~213s

        if self.reduced:
            print(r'Reduced chi^2=', chi_sq / (n_pts - self.n_params))  # n_params = number of free parameters (from load)
            chi_sq /= (n_pts - self.n_params)  # convert to reduced chi^2; else just return full chi^2
        if n_pts == 0.:
            print(self.resolution, self.xell, self.yell)
            print('WARNING! STOP! There are no pixels inside the fitting ellipse! n_pts = ' + str(n_pts))
            chi_sq = np.inf
        return chi_sq  # Reduced or Not depending on reduced = True or False


    def output_cube(self):  # if outputting actual cube itself
        inds_to_try2 = np.asarray([[10, 10], [10, 15], [15, 10]])  # plot a few line profiles
        import test_dyn_funcs as tdf
        f_sys = self.f_0 / (1+self.zred)
        tdf.compare(self.input_data_masked, self.convolved_cube, self.freq_ax / 1e9, inds_to_try2, f_sys / 1e9, 4)  # plot them!

        if not Path(self.out_name).exists():  # WRITE OUT RESULTS TO FITS FILE
            hdu = fits.PrimaryHDU(self.convolved_cube)
            hdul = fits.HDUList([hdu])
            hdul.writeto(self.out_name)
            print('written!')

        return self.convolved_cube


def test_qell2(input_data, params, l_in, q_ell, rfit, pa, figname):
    figname += '_' + str(q_ell) + '_' + str(pa) + '_' + str(rfit) + '.png'
    ell_mask = ellipse_fitting(input_data, rfit, params['xell'], params['yell'], params['resolution'], pa, q_ell)

    fig = plt.figure()
    ax = plt.gca()
    plt.imshow(l_in, origin='lower')
    plt.colorbar()
    from matplotlib import patches
    e1 = patches.Ellipse((params['xell'], params['yell']), 2 * rfit / params['resolution'],
                         2 * rfit / params['resolution'] * q_ell, angle=pa, linewidth=2, edgecolor='w', fill=False)
    ax.add_patch(e1)
    plt.title(r'q = ' + str(q_ell) + r', PA = ' + str(pa) + ' deg, rfit = ' + str(rfit) + ' arcsec')
    plt.savefig(figname, dpi=300)


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
                         zrange=[params['zi'], params['zf']])

    lucy_mask, lucy_out, beam, fluxes, freq_ax, f_0, fstep, input_data, noise = mod_ins

    hduin = fits.open(params['lucy_in'])
    l_in = hduin[0].data
    hduin.close()

    # CREATE MODEL CUBE!
    out = params['outname']
    mg = ModelGrid(resolution=params['resolution'], os=params['s'], x_loc=params['xloc'], y_loc=params['yloc'],
                   mbh=params['mbh'], inc=np.deg2rad(params['inc']), vsys=params['vsys'], dist=params['dist'],
                   theta=np.deg2rad(params['PAdisk']), input_data=input_data, lucy_out=lucy_out, out_name=out,
                   beam=beam, rfit=params['rfit'], enclosed_mass=params['mass'], ml_ratio=params['ml_ratio'],
                   sig_type=params['s_type'], zrange=[params['zi'], params['zf']], menc_type=params['mtype'],
                   sig_params=[params['sig0'], params['r0'], params['mu'], params['sig1']], f_w=params['f'],
                   ds=params['ds'], noise=noise, reduced=True, freq_ax=freq_ax, q_ell=params['q_ell'],
                   theta_ell=np.deg2rad(params['theta_ell']), xell=params['xell'], yell=params['yell'], fstep=fstep,
                   f_0=f_0, bl=params['bl'], xyrange=[params['xi'], params['xf'], params['yi'], params['yf']],
                   n_params=n_free)
    mg.create_grid()
    mg.convolve_cube()
    chi_sq = mg.chi2()

    print('True Total time: ' + str(time.time() - t0_true) + ' seconds')  # 3.93s YAY!  # 23s for 6 models (16.6 for 4 models)
    print(r'$\chi^2_{\nu}$ = ' + str(chi_sq))  # NOTE: right now, chi^2_nu (with my lucy + v(R)) = 1.9, chi^2_nu (with my lucy + mge) = 9.98
