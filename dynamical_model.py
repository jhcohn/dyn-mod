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

        return params, priors, nfree, qobs

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
                 kappa=0., omega=0., dist=17., theta=np.deg2rad(200.), input_data=None, lucy_out=None, vtype='orig',
                 out_name=None, beam=None, rfit=1., q_ell=1., theta_ell=0., xell=360., yell=350., bl=False,
                 enclosed_mass=None, menc_type=0, ml_ratio=1., sig_type='flat', sig_params=None, f_w=1., noise=None,
                 ds=None, zrange=None, xyrange=None, reduced=False, freq_ax=None, f_0=0., fstep=0., opt=True,
                 quiet=False, n_params=8, data_mask=None, f_he=1.37, r21=0.7, alpha_co10=3.1, incl_gas=False,
                 gas_norm=1e5, gas_radius=5):
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
        self.dist = dist  # distance to the galaxy [Mpc]
        self.pc_per_ac = self.dist * 1e6 / self.arcsec_per_rad  # small angle formula (convert dist to pc, from Mpc)
        self.pc_per_pix = self.dist * 1e6 / self.arcsec_per_rad * self.resolution  # small angle formula, as above
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
        self.f_he = f_he  # additional fraction of gas that is helium (f_he = 1 + helium mass fraction)
        self.r21 = r21  # CO(2-1)/CO(1-0) SB ratio (see pg 6-8 in Boizelle+17)
        self.alpha_co10 = alpha_co10  # CO(1-0) to H2 conversion factor (see pg 6-8 in Boizelle+17)
        self.incl_gas = incl_gas  # if True, include gas mass in calculations
        self.gas_norm = gas_norm  # best-fit exponential coefficient for gas mass calculation [Msol/pix^2]
        self.gas_radius = gas_radius  # best-fit scale radius for gas-mass calculation [pix]
        # Parameters to be built in create_grid(), convolve_cube(), or chi2 functions inside the class
        self.z_ax = None  # velocity axis, constructed from freq_ax, f_0, and vsys, based on opt
        self.weight = None  # 2D weight map, constructed from lucy_output (the deconvolved fluxmap)
        self.freq_obs = None  # the 2D line-of-sight velocity map, converted to frequency
        self.delta_freq_obs = None  # the 2D turbulent velocity map, converted to delta-frequency
        self.clipped_data = None  # the data sub-cube that we compare to the model, clipped by zrange and xyrange
        self.convolved_cube = None  # the model cube: create from convolving the intrinsic model cube with the ALMA beam
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
        vg = 0  # default to ignoring the gas mass!
        if self.incl_gas:  # If incl_mass, overwrite vg with a gas mass estimate, add it in quadrature to velocity!
            vg = gas_mass()
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
            v_los *= self.omega
            v_los += self.kappa * vrad_sign * abs(vel * np.sin(alpha) * np.sin(self.inc))

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


    def gas_mass(self):
        """
        # DO CO PROFILE CALCULATIONS FOR GAS MASS
        :return: the velocity due to the gas mass
        """
        # CALCULATE VELOCITY WIDTH
        v_width = self.c_kms * (1 + self.zred) * self.fstep / self.f_0  # velocity width [Jy km/s beam^-1]

        # BUCKET: move construction of collapsed_fluxes_vel to model_prep(), so only do it once? (Can't, bc self.zred)
        hdu_m = fits.open(self.data_mask)  # open the strictmask
        mask = hdu_m[0].data
        hdu_m.close()

        # CONSTRUCT THE FLUX MAP IN UNITS Jy km/s beam^-1
        collapsed_fluxes_vel = np.zeros(shape=(len(self.input_data[0]), len(self.input_data[0][0])))
        for zi in range(len(self.input_data)):
            collapsed_fluxes_vel += self.input_data[zi] * mask[zi] * v_width
        # collapsed_fluxes_vel[collapsed_fluxes_vel < 0] = 0.  # ignore negative values? probably not?

        # DEFINE SEMI-MAJOR AXES FOR SAMPLING, THEN CALCULATE THE MEAN SURFACE BRIGHTNESS INSIDE ELLIPTICAL ANNULI
        semi_majors = np.logspace(0., 1.84, num=12)  # [pix] make these params (0, 1.84, num=12) class inputs?
        # semi_major_axes from 0.02 to 1.4 arcsec = 1 to ~70 pix (~10^1.84 pix) [res = 0.02 arcsec/pix]
        co_surf, co_errs = annuli_sb(collapsed_fluxes_vel, semi_majors, np.deg2rad(self.theta_ell), self.q_ell,
                                     self.xell, self.yell)
        co_ell_sb = np.asarray(co_surf)  # convert the output list of mean CO surface brightnesses into an array
        semi_majors_pc = semi_majors[1:] * self.resolution  # convert semi-major axes from pix to arcsec!
        # THEN, convert co_ell_rad to be the mean annulus radius, not sma value!
        co_ell_rad = (2. * semi_majors_pc + self.q_ell * semi_majors_pc) / 3.  # mean_ellipse_radius = (2a + b)/3

        # Set up integration bounds, numerical integration step size, & vectors "avals" & "rvals" for each integration.
        # maxr >> than the maximum CO radius s.t. the relative grav. potential contributions are small compared to near
        # the disk edge.
        del_r = 0.1  # integration step size [pc]
        min_r = 0.  # integration lower bound [pix or pc]
        max_r = 65. * self.pc_per_pix  # 450.  # integration upper bound [pc; semi-major axis on flux map is ~30pix]
        avals = range(min_r,max_r,(max_r-min_r)/del_r)  # [pc]
        rvals = range(min_r,max_r,(max_r-min_r)/del_r)  # [pc]

        # set up conversion factor: transfer from Jy km/s beam^-1 to Msol / pc^2
        # See eqn (1) in Boizelle+17 (from Carilli & Walter 13, S2.4: https://arxiv.org/pdf/1301.0371.pdf)
        jykms_to_msol = 3.25e7 * self.alpha_co10 * self.f_he * self.dist**2 /\
                        ((1 + self.zred) * self.r21 * (self.f_0 / 1e9)**2)

        # Fit the CO distribution w/ an exp profile (w/ scale radius & norm), then construct Sigma(R) for R=rvals
        radius_pc = self.gas_radius * self.pc_per_pix  # convert free parameter from [pix] to [pc]
        gas_norm_pc = self.gas_norm / self.pc_per_pix ** 2  # convert free parameter from [Msol/pix^2] to [Msol/pc^2]
        sigr2 = gas_norm_pc * np.cos(self.inc) * jykms_to_msol * np.exp(-rvals / radius_pc)

        # Interpolate CO surface brightness vs elliptical mean radii, to construct \Sigma(R) for R=rvals
        # create a function that returns the CO surface brightness (in Jy km/s beam^-1) for a set of radii (in pc)
        sigr3_func_r = interpolate.interp1d(co_ell_rad, co_ell_sb, kind='spline', fill_value='extrapolate')
        sigr3 = sigr3_func_r(rvals) * np.cos(self.inc) * jykms_to_msol  # calculate Msol/pc^2 profile for rvals!

        # Calculate the 1st (inner) integral, then compute its differential (d/da) (for cases (2) and (3))
        # See eqn 2.157 from Binney & Tremaine
        int1_a2 = np.zeros(shape=len(rvals))  # int1_a2=make_array(n_elements(rvals),value=0d)
        int1_a3 = np.zeros(shape=len(rvals))
        for i in range(1, len(rvals)):  # for i=1.,n_elements(rvals)-1 # rvals-1 BC IDL INDEXING INCLUSIVE!
            int1_a2[i] = np.sum(rvals[i:] * sigr2[i:] * del_r / np.sqrt(rvals[i:]**2 - avals[i-1]**2))
            int1_a3[i] = np.sum(rvals[i:] * sigr3[i:] * del_r / np.sqrt(rvals[i:]**2 - avals[i-1]**2))

        # Compute crude numerical differential wrt radius (d/da) for 2nd (outer) integral (eqn 2.157 Binney & Tremaine)
        int1_dda2 = np.zeros(shape=len(rvals))
        int1_dda3 = np.zeros(shape=len(rvals))
        # Offset indices in int1_a* by 1 so that the difference is like a derivative
        int1_dda2[1:] = (int1_a2[1:] - int1_a2[0:-1]) / del_r
        int1_dda3[1:] = (int1_a3[1:] - int1_a3[0:-1]) / del_r

        # Calculate the second (outer) integral (eqn 2.157 Binney & Tremaine)
        int2_r2 = np.zeros(shape=len(avals))  # int2_r2=make_array(n_elements(avals),value=0d)
        int2_r3 = np.zeros(shape=len(avals))
        for i in range(1, len(avals) - 1):  # only go to len(avals)-1 (in IDL: -2) bc index rvals[i+1]
            int2_r2[i] = np.sum(avals[0:i] * int1_dda2[0:i] / np.sqrt(rvals[i+1]**2 - avals[0:i]**2) * del_r)
            int2_r3[i] = np.sum(avals[0:i] * int1_dda3[0:i] / np.sqrt(rvals[i+1]**2 - avals[0:i]**2) * del_r)

        # Numerical v_cg solution assuming an exponential mass distribution (vc2) & one following the CO sb (vc3)
        vc2 = np.sqrt(np.abs(-4*self.G_pc*int2_r2))
        vc3 = np.sqrt(np.abs(-4*self.G_pc*int2_r3))

        # Note that since potentials are additive, sum up the velocity contributions in quadrature:
        v_cg = vc3

        return v_cg


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
            intrinsic_cube = rebin(cube_model, self.os)  # intrinsic_cube = block_reduce(cube_model, s, np.mean)

        tc = time.time()
        # CONVERT INTRINSIC TO OBSERVED (convolve each slice of intrinsic_cube with ALMA beam --> observed data cube)
        self.convolved_cube = np.zeros(shape=intrinsic_cube.shape)  # approx ~1e-6 to 3e-6s per pixel
        for z in range(len(self.z_ax)):
            self.convolved_cube[z, :, :] = convolution.convolve(intrinsic_cube[z, :, :], self.beam)
        print('convolution loop ' + str(time.time() - tc))


    def chi2(self):
        # ONLY WANT TO FIT WITHIN ELLIPTICAL REGION! CREATE ELLIPSE MASK
        ell_mask = ellipse_fitting(self.convolved_cube, self.rfit, self.xell, self.yell, self.resolution,
                                   self.theta_ell, self.q_ell)  # create ellipse mask

        # CREATE A CLIPPED DATA CUBE SO THAT WE'RE LOOKING AT THE SAME EXACT x,y,z REGION AS IN THE MODEL CUBE
        self.clipped_data = self.input_data[self.zrange[0]:self.zrange[1], self.xyrange[2]:self.xyrange[3],
                                            self.xyrange[0]:self.xyrange[1]]

        # self.convolved_cube *= ell_mask  # mask the convolved model cube
        # self.input_data_masked = self.input_data[self.zrange[0]:self.zrange[1], self.xyrange[2]:self.xyrange[3],
        #                          self.xyrange[0]:self.xyrange[1]] * ell_mask  # mask the input data cube

        # REBIN THE ELLIPSE MASK BY THE DOWN-SAMPLING FACTOR
        self.ell_ds = rebin(ell_mask, self.ds)[0]  # rebin the mask by the down-sampling factor
        self.ell_ds[self.ell_ds < self.ds**2 / 2.] = 0.  # set all pix < 50% "inside" the ellipse to be outside -> mask
        self.ell_ds = np.nan_to_num(self.ell_ds / np.abs(self.ell_ds))  # set all points in ellipse = 1, convert nan->0

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

            z_ind += 1  # the actual index for the model-data comparison cubes

        if not self.quiet:
            print(np.sum(self.ell_ds), len(self.z_ax), n_pts)
            print(r'chi^2=', chi_sq)

        if self.reduced:  # CALCULATE REDUCED CHI^2
            chi_sq /= (n_pts - self.n_params)  # convert to reduced chi^2; else just return full chi^2
            if not self.quiet:
                print(r'Reduced chi^2=', chi_sq)

        if n_pts == 0.:  # PROBLEM WARNING
            print(self.resolution, self.xell, self.yell, self.theta_ell, self.q_ell, self.rfit)
            print('WARNING! STOP! There are no pixels inside the fitting ellipse! n_pts = ' + str(n_pts))
            chi_sq = np.inf

        return chi_sq  # Reduced or Not depending on reduced = True or False


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


    def moment_0(self, abs_diff, incl_beam, norm):
        """
        Create 0th moment map

        :param abs_diff: True or False; if True, show absolute value of the residual
        :param incl_beam: True or False; if True, include beam inset in the data panel
        :param norm: True or False; if True, normalize residual by the data
        :return: moment map plot
        """
        # if using equation from https://www.atnf.csiro.au/people/Tobias.Westmeier/tools_hihelpers.php#moments
        vel_ax = []
        velwidth = self.c_kms * (1 + self.zred) * self.fstep / self.f_0
        for v in range(len(self.freq_ax)):
            vel_ax.append(self.c_kms * (1. - (self.freq_ax[v] / self.f_0) * (1 + self.zred)))

        # full cube strictmask, clipped to the appropriate zrange
        # (NOTE: would need to clip to xyrange, & rebin with ds, to compare data_ds & ap_ds. Seems wrong thing to do.)
        clipped_mask = self.data_mask[self.zrange[0]:self.zrange[1]]

        data_masked_m0 = np.zeros(shape=self.ell_mask.shape)
        for z in range(len(vel_ax)):
            data_masked_m0 += abs(velwidth) * self.clipped_data[z] * clipped_mask[z]

        model_masked_m0 = np.zeros(shape=(len(self.convolved_cube[0]), len(self.convolved_cube[0][0])))
        for zi in range(len(convolved_cube)):
            model_masked_m0 += self.convolved_cube[zi] * abs(velwidth) * clipped_mask[zi]

        fig = plt.figure()
        grid = AxesGrid(fig, 111,
                        nrows_ncols=(1, 3),
                        axes_pad=0.01,
                        cbar_mode='single',
                        cbar_location='right',
                        cbar_pad=0.1)
        i = 0

        # CONVERT TO mJy
        data_masked_m0 *= 1e3
        model_masked_m0 *= 1e3

        subtr = 0.
        if incl_beam:
            beam_overlay = np.zeros(shape=self.ell_mask.shape)
            beam_overlay[:self.beam.shape[0], (beam_overlay.shape[1] - self.beam.shape[1]):] = self.beam
            print(beam_overlay.shape, self.beam.shape)
            beam_overlay *= np.amax(data_masked_m0) / np.amax(beam_overlay)
            data_masked_m0 += beam_overlay
            subtr = beam_overlay
        for ax in grid:
            vmin = np.amin([np.nanmin(model_masked_m0), np.nanmin(data_masked_m0)])
            vmax = np.amax([np.nanmax(model_masked_m0), np.nanmax(data_masked_m0)])
            cbartitle0 = r'mJy/beam'
            if i == 0:
                im = ax.imshow(data_masked_m0, vmin=vmin, vmax=vmax, origin='lower')
                ax.set_title(r'Moment 0 (data)')
            elif i == 1:
                im = ax.imshow(model_masked_m0, vmin=vmin, vmax=vmax, origin='lower')
                ax.set_title(r'Moment 0 (model)')
            elif i == 2:
                title0 = 'Moment 0 residual (model-data)'
                titleabs = 'Moment 0 residual abs(model-data)'
                diff = model_masked_m0 - (data_masked_m0 - subtr)
                if norm:
                    diff /= data_masked_m0
                    print(np.nanquantile(diff, [0.16, 0.5, 0.84]), 'typical differences; 0.16, 0.5, 0.84!')
                    title0 += ' / data'
                    titleabs += ' / data'
                    cbartitle0 = 'Ratio [Residual / Data]'
                if samescale:
                    if abs_diff:
                        diff = np.abs(diff)
                        ax.set_title(titleabs)
                    else:
                        ax.set_title(title0)
                    im = ax.imshow(diff, vmin=vmin, vmax=vmax, origin='lower')
                else:  # then residual scale
                    im = ax.imshow(diff, origin='lower', vmin=np.nanmin([diff, -diff]), vmax=np.nanmax([diff, -diff]))
                    ax.set_title(title0)
            i += 1

            ax.set_xlabel(r'x [pixels]', fontsize=20)  # x [arcsec]
            ax.set_ylabel(r'y [pixels]', fontsize=20)  # y [arcsec]

        cbar = grid.cbar_axes[0].colorbar(im)
        cax = grid.cbar_axes[0]
        axis = cax.axis[cax.orientation]
        axis.label.set_text(cbartitle0)
        plt.show()


    def moment_12(self, abs_diff, incl_beam, norm, mom):
        """
        Create 1st or 2nd moment map
        :param abs_diff: True or False; if True, show absolute value of the residual
        :param incl_beam: True or False; if True, include beam inset in the data panel
        :param norm: True or False; if True, normalize residual by the data
        :param mom: moment, 1 or 2
        :return: moment map plot
        """
        vel_ax = []
        velwidth = self.c_kms * (1 + self.zred) * self.fstep / self.f_0
        for v in range(len(self.freq_ax)):
            vel_ax.append(self.c_kms * (1. - (self.freq_ax[v] / self.f_0) * (1 + self.zred)))

        # full cube strictmask, clipped to the appropriate zrange
        # (NOTE: would need to clip to xyrange, & rebin with ds, to compare data_ds & ap_ds. Seems wrong thing to do.)
        clipped_mask = self.data_mask[self.zrange[0]:self.zrange[1]]

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
            for zi in range(len(convolved_cube)):
                m2_num += (vel_ax[zi] - model_mom)**2 * self.convolved_cube[zi] * clipped_mask[zi]
                m2_den += self.convolved_cube[zi] * clipped_mask[zi]
            m2 = np.sqrt(m2_num / m2_den) # * d1  # BUCKET: no need for MASKING using d1?

            d2_num = np.zeros(shape=(len(self.convolved_cube[0]), len(self.convolved_cube[0][0])))
            d2_n2 = np.zeros(shape=(len(self.convolved_cube[0]), len(self.convolved_cube[0][0])))
            d2_den = np.zeros(shape=(len(self.convolved_cube[0]), len(self.convolved_cube[0][0])))
            for zi in range(len(convolved_cube)):
                d2_n2 += self.clipped_data[zi] * (vel_ax[zi] - dmap)**2 * clipped_mask[zi] # * mask2d
                d2_num += (vel_ax[zi] - dmap)**2 * self.clipped_data[zi] * clipped_mask[zi] # * mask2d
                d2_den += self.clipped_data[zi] * clipped_mask[zi] # * mask2d
            dfig = np.zeros(shape=(len(self.convolved_cube[0]), len(self.convolved_cube[0][0]))) + 1.  # create mask
            dfig2 = np.zeros(shape=(len(self.convolved_cube[0]), len(self.convolved_cube[0][0]))) + 1.  # create mask
            dfig[d2_n2 < 0.] = 0.  # d2_n2 matches d2_den on the sign aspect
            dfig2[d2_den < 0.] = 0.
            d2 = np.sqrt(d2_num / d2_den) # * d1  # BUCKET: no need for MASKING using d1?

            fig = plt.figure()
            grid = AxesGrid(fig, 111,
                            nrows_ncols=(1, 3),
                            axes_pad=0.01,
                            cbar_mode='single',
                            cbar_location='right',
                            cbar_pad=0.1)
            i = 0

            subtr = 0.
            if incl_beam:
                d2 = np.nan_to_num(d2)
                beam_overlay = np.zeros(shape=self.ell_mask.shape)
                beam_overlay[:self.beam.shape[0], (beam_overlay.shape[1] - self.beam.shape[1]):] = self.beam
                print(beam_overlay.shape, self.beam.shape)
                beam_overlay *= np.amax(d2) / np.amax(beam_overlay)
                d2 += beam_overlay
                subtr = beam_overlay

            for ax in grid:
                vmin2 = np.amin([np.nanmin(d2), np.nanmin(m2)])
                vmax2 = np.amax([np.nanmax(d2), np.nanmax(m2)])
                cbartitle2 = r'km/s'
                if i == 0:
                    im = ax.imshow(d2, origin='lower', vmin=vmin2, vmax=vmax2)  # , cmap='RdBu_r'
                    ax.set_title(r'Moment 2 (data)')
                elif i == 1:
                    im = ax.imshow(m2, origin='lower', vmin=vmin2, vmax=vmax2)  # , cmap='RdBu_r'
                    ax.set_title(r'Moment 2 (model)')
                elif i == 2:
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
                        ax.set_title(titleabs2)
                    else:
                        ax.set_title(title2)
                    if samescale:
                        im = ax.imshow(diff, origin='lower', vmin=vmin2, vmax=vmax2)  # , cmap='RdBu'
                    else:  # residscale
                        im = ax.imshow(diff, origin='lower', vmin=np.nanmin([diff, -diff]),
                                       vmax=np.nanmax([diff, -diff]))
                        ax.set_title(title2)
                i += 1

                ax.set_xlabel(r'x [pixels]', fontsize=20)  # x [arcsec]
                ax.set_ylabel(r'y [pixels]', fontsize=20)  # y [arcsec]

            cbar = ax.cax.colorbar(im)
            cbar = grid.cbar_axes[0].colorbar(im)
            cax = grid.cbar_axes[0]
            axis = cax.axis[cax.orientation]
            axis.label.set_text(cbartitle2)
            plt.show()

        elif mom == 1:
            fig = plt.figure()
            grid = AxesGrid(fig, 111,
                            nrows_ncols=(1, 3),
                            axes_pad=0.01,
                            cbar_mode='single',
                            cbar_location='right',
                            cbar_pad=0.1)
            i = 0

            subtr = 0.
            if incl_beam:
                beam_overlay = np.zeros(shape=self.ell_mask.shape)
                beam_overlay[:self.beam.shape[0], (beam_overlay.shape[1] - self.beam.shape[1]):] = self.beam
                print(beam_overlay.shape, self.beam.shape)
                beam_overlay *= np.amax(data_mom) / np.amax(beam_overlay)
                data_mom += beam_overlay
                subtr = beam_overlay

            for ax in grid:
                cbartitle1 = r'km/s'
                vmin1 = np.amin([np.nanmin(data_mom), np.nanmin(model_mom)])
                vmax1 = np.amax([np.nanmax(data_mom), np.nanmax(model_mom)])
                if i == 0:
                    im = ax.imshow(data_mom, origin='lower', vmin=vmin1, vmax=vmax1, cmap='RdBu_r')
                    ax.set_title(r'Moment 1 (data)')
                elif i == 1:
                    im = ax.imshow(model_mom, origin='lower', vmin=vmin1, vmax=vmax1, cmap='RdBu_r')
                    ax.set_title(r'Moment 1 (model)')
                elif i == 2:
                    title1 = 'Moment 1 residual (model - data)'
                    diff = model_mom - (data_mom - subtr)
                    if norm:
                        diff /= data_mom
                        print(np.nanquantile(diff, [0.16, 0.5, 0.84]), 'look median!')
                        title1 += ' / data'
                        cbartitle1 = 'Ratio [Residual / Data]'
                    if samescale:
                        im = ax.imshow(diff, origin='lower', vmin=vmin1, vmax=vmax1, cmap='RdBu')  # , cmap='RdBu'
                    else:  # resid scale
                        im = ax.imshow(diff, origin='lower', vmin=np.nanmin([diff, -diff]),
                                       vmax=np.nanmax([diff, -diff]))  # cmap='RdBu'
                    ax.set_title(title1)
                i += 1

                ax.set_xlabel(r'x [pixels]', fontsize=20)  # x [arcsec]
                ax.set_ylabel(r'y [pixels]', fontsize=20)  # y [arcsec]

            cbar = ax.cax.colorbar(im)
            cbar = grid.cbar_axes[0].colorbar(im)
            cax = grid.cbar_axes[0]
            axis = cax.axis[cax.orientation]
            axis.label.set_text(cbartitle1)
            plt.show()



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


def annuli_sb(flux_map, semi_major_axes, position_angle, axis_ratio, x_center, y_center):
    """
    Create an array of elliptical annuli, in which to calculate the mean CO surface brightness

    :param flux_map: CO surface brightness flux map
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
        print(sma, x_center, y_center, sma * axis_ratio, position_angle, len(flux_map), len(flux_map[0]))
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
        print(errs_co[:, e])
        #plt.imshow(annulus_flux, origin='lower')
        #plt.colorbar()
        #plt.show()

    return average_co, errs_co


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
    mg = ModelGrid(resolution=params['resolution'], os=params['os'], x_loc=params['xloc'], y_loc=params['yloc'],
                   mbh=params['mbh'], inc=np.deg2rad(params['inc']), vsys=params['vsys'], dist=params['dist'],
                   theta=np.deg2rad(params['PAdisk']), input_data=input_data, lucy_out=lucy_out, out_name=out,
                   beam=beam, rfit=params['rfit'], enclosed_mass=params['mass'], ml_ratio=params['ml_ratio'],
                   sig_type=params['s_type'], zrange=[params['zi'], params['zf']], menc_type=params['mtype'],
                   sig_params=[params['sig0'], params['r0'], params['mu'], params['sig1']], f_w=params['f'],
                   ds=params['ds'], noise=noise, reduced=True, freq_ax=freq_ax, q_ell=params['q_ell'],
                   theta_ell=np.deg2rad(params['theta_ell']), xell=params['xell'], yell=params['yell'], fstep=fstep,
                   f_0=f_0, bl=params['bl'], xyrange=[params['xi'], params['xf'], params['yi'], params['yf']],
                   n_params=n_free, data_mask=params['mask'])
    mg.grids()
    mg.convolution()
    chi_sq = mg.chi2()
    lab = r'$\chi^2'
    if reduced:
        lab = r'$\chi^2_{\nu}$'

    print('True Total time: ' + str(time.time() - t0_true) + ' seconds')  # ~1 second for a cube of 84x64x49
    print(lab + r' = ' + str(chi_sq))
