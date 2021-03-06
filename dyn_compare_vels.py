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
sys.path.insert(0, '/Users/jonathancohn/Documents/jam/')  # lets me import file from different folder/path
import mge_vcirc_mine as mvm  # import mge_vcirc code


class Constants:  # a bunch of useful astronomy constants

    def __init__(self):
        self.c = 2.99792458 * 10 ** 8  # [m / s]
        self.pc = 3.086 * 10 ** 16  # [m / pc]
        self.G = 6.67 * 10 ** -11  # [kg^-1 * m^3 * s^-2]
        self.M_sol = 1.989 * 10 ** 30  # [kg / solar mass]
        self.H0 = 70  # [km/s/Mpc]
        self.arcsec_per_rad = 206265.  # [arcsec / radian]
        self.m_per_km = 10. ** 3  # [m / km]
        self.G_pc = self.G * self.M_sol * (1. / self.pc) / self.m_per_km ** 2  # G [Msol^-1 * pc * km^2 * s^-2] (gross)
        self.c_kms = self.c / self.m_per_km  # [km / s]


def make_beam(grid_size=99, res=1., amp=1., x0=0., y0=0., x_std=1., y_std=1., rot=0., fits_name=None):
    """
    Generate a beam spread function for the ALMA beam

    :param grid_size: size of grid (must be odd!)
    :param res: resolution of the grid (arcsec/pixel)
    :param amp: amplitude of the 2d gaussian
    :param x0: mean of x axis of 2d gaussian
    :param y0: mean of y axis of 2d gaussian
    :param x_std: FWHM of beam in x (to use for standard deviation of Gaussian) (in arcsec)
    :param y_std: FWHM of beam in y (to use for standard deviation of Gaussian) (in arcsec)
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

    hdu.close()  # close data
    hdu_m.close()  # close mask

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


def model_grid(resolution=0.05, s=10, x_loc=0., y_loc=0., mbh=4 * 10 ** 8, inc=np.deg2rad(60.), vsys=None, dist=17.,
               theta=np.deg2rad(200.), input_data=None, lucy_out=None, out_name=None, beam=None, q_ell=1., theta_ell=0.,
               bl=False, enclosed_mass=None, menc_type=False, ml_ratio=1., sig_type='flat', sig_params=None, f_w=1.,
               noise=None, rfit=1., ds=None, chi2=False, zrange=None, xyrange=None, reduced=False, freq_ax=None, f_0=0.,
               fstep=0., opt=True, quiet=False, xell=360., yell=350.):
    """
    Build grid for dynamical modeling!

    :param resolution: resolution of observations [arcsec/pixel]
    :param s: oversampling factor
    :param x_loc: the location of the BH, as measured along the x axis of the data cube [pixels]
    :param y_loc: the location of the BH, as measured along the y axis of the data cube [pixels]
    :param mbh: supermassive black hole mass [solar masses]
    :param inc: inclination of the galaxy [radians]
    :param vsys: if given, the systemic velocity [km/s]
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

    :return: convolved model cube (if chi2 is False); chi2 (if chi2 is True)
    """
    t_begin = time.time()

    # INSTANTIATE ASTRONOMICAL CONSTANTS
    constants = Constants()

    # SUBPIXELS (reshape deconvolved flux map [lucy_out] sxs subpixels, so subpix has flux=(real pixel flux)/s**2)
    if not quiet:
        print('start')
    if s == 1:  # subpix_deconvolved is identical to lucy_out, with sxs subpixels per pixel & total flux conserved
        subpix_deconvolved = lucy_out
    else:
        subpix_deconvolved = np.zeros(shape=(len(lucy_out) * s, len(lucy_out[0]) * s))
        for ypix in range(len(lucy_out)):
            for xpix in range(len(lucy_out[0])):
                subpix_deconvolved[(ypix * s):(ypix + 1) * s, (xpix * s):(xpix + 1) * s] = lucy_out[ypix, xpix] / s ** 2

    # convert from frequency (Hz) to velocity (km/s), with freq_ax in Hz
    if opt:  # v_opt
        z_ax = np.asarray([vsys + ((f_0 - freq) / freq) * (constants.c / constants.m_per_km) for freq in freq_ax])
    else:  # v_rad
        z_ax = np.asarray([vsys + ((f_0 - freq) / f_0) * (constants.c / constants.m_per_km) for freq in freq_ax])

    # RESCALE subpix_deconvolved, z_ax, freq_ax TO CONTAIN ONLY THE SUB-CUBE REGION WHERE EMISSION ACTUALLY EXISTS
    subpix_deconvolved = subpix_deconvolved[s * xyrange[2]:s * xyrange[3], s * xyrange[0]:s * xyrange[1]]  # stored: y,x
    z_ax = z_ax[zrange[0]:zrange[1]]
    freq_ax = freq_ax[zrange[0]:zrange[1]]

    # RESCALE x_loc, y_loc PIXEL VALUES TO CORRESPOND TO SUB-CUBE PIXEL LOCATIONS!
    x_loc = x_loc - xyrange[0]  # x_loc - xi
    y_loc = y_loc - xyrange[2]  # y_loc - yi

    xell = xell - xyrange[0]
    yell = yell - xyrange[2]

    # SET UP OBSERVATION AXES: initialize x,y axes at 0., with lengths = s * len(input data cube axes)
    y_obs_ac = np.asarray([0.] * len(subpix_deconvolved))
    x_obs_ac = np.asarray([0.] * len(subpix_deconvolved[0]))

    # Define coordinates to be 0,0 at center of the observed axes (find the central pixel number along each axis)
    if len(x_obs_ac) % 2. == 0:  # if even
        x_ctr = (len(x_obs_ac)) / 2.  # set the center of the axes (in pixel number)
        for i in range(len(x_obs_ac)):
            x_obs_ac[i] = resolution * (i - x_ctr) / s  # (arcsec/pix) * N_subpixels / (subpixels/pix) = arcsec
    else:  # elif odd
        x_ctr = (len(x_obs_ac) + 1.) / 2.  # +1 bc python starts counting at 0
        for i in range(len(x_obs_ac)):
            x_obs_ac[i] = resolution * ((i + 1) - x_ctr) / s
    if len(y_obs_ac) % 2. == 0:
        y_ctr = (len(y_obs_ac)) / 2.
        for i in range(len(y_obs_ac)):
            y_obs_ac[i] = resolution * (i - y_ctr) / s
    else:
        y_ctr = (len(y_obs_ac) + 1.) / 2.
        for i in range(len(y_obs_ac)):
            y_obs_ac[i] = resolution * ((i + 1) - y_ctr) / s

    # SET BH OFFSET from center [in arcsec], based on input BH pixel position (note: _loc in pixels; _ctr in subpixels)
    x_bh_ac = (x_loc - x_ctr / s) * resolution  # units (pix - subpix/[subpix/pix]) * (arcsec/pix) = arcsec
    y_bh_ac = (y_loc - y_ctr / s) * resolution

    # CONVERT FROM ARCSEC TO PHYSICAL UNITS (pc)
    x_bhoff = dist * 10 ** 6 * np.tan(x_bh_ac / constants.arcsec_per_rad)  # tan(off) = x_disk/dist --> x = d*tan(off)
    y_bhoff = dist * 10 ** 6 * np.tan(y_bh_ac / constants.arcsec_per_rad)  # 206265 arcsec/rad

    # convert all x,y observed grid positions to pc
    x_obs = np.asarray([dist * 10 ** 6 * np.tan(x / constants.arcsec_per_rad) for x in x_obs_ac])  # 206265 arcsec/rad
    y_obs = np.asarray([dist * 10 ** 6 * np.tan(y / constants.arcsec_per_rad) for y in y_obs_ac])  # 206265 arcsec/rad

    # CONVERT FROM x_obs, y_obs TO x_disk, y_disk [pc]
    x_disk_ac = (x_obs_ac[None, :] - x_bh_ac) * np.cos(theta) + (y_obs_ac[:, None] - y_bh_ac) * np.sin(theta)  # arcsec
    y_disk_ac = (y_obs_ac[:, None] - y_bh_ac) * np.cos(theta) - (x_obs_ac[None, :] - x_bh_ac) * np.sin(theta)  # arcsec
    x_disk = (x_obs[None, :] - x_bhoff) * np.cos(theta) + (y_obs[:, None] - y_bhoff) * np.sin(theta)  # 2d array
    y_disk = (y_obs[:, None] - y_bhoff) * np.cos(theta) - (x_obs[None, :] - x_bhoff) * np.sin(theta)  # 2d array

    # CALCULATE THE RADIUS (R) OF EACH POINT (x_disk, y_disk) IN THE DISK [pc]
    R_ac = np.sqrt((y_disk_ac ** 2 / np.cos(inc) ** 2) + x_disk_ac ** 2)  # radius R (arcsec)
    R = np.sqrt((y_disk ** 2 / np.cos(inc) ** 2) + x_disk ** 2)  # radius R of each point in the disk (2d array)

    # CALCULATE KEPLERIAN VELOCITY DUE TO ENCLOSED STELLAR MASS
    if menc_type == 0:  # if calculating v(R) due to stars directly from MGE parameters
        if not quiet:
            print('mge')
        test_rad = np.linspace(np.amin(R_ac), np.amax(R_ac), 100)

        comp, surf_pots, sigma_pots, qobs = mvm.load_mge(enclosed_mass)  # load the MGE parameters

        v_c = mvm.mge_vcirc(surf_pots * ml_ratio, sigma_pots, qobs, np.rad2deg(inc), 0., dist, test_rad)  # v_c,star
        v_c_r = interpolate.interp1d(test_rad, v_c, kind='cubic', fill_value='extrapolate')

        # CALCULATE KEPLERIAN VELOCITY OF ANY POINT (x_disk, y_disk) IN THE DISK WITH RADIUS R (km/s)
        vel = np.sqrt((constants.G_pc * mbh / R) + v_c_r(R_ac)**2)
    elif menc_type == 1:  # elif using a file with v_circ(R) due to stellar mass
        if not quiet:
            print('v(r)')
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

    elif menc_type == 2:  # elif using a file directly with stellar mass as a function of R
        if not quiet:
            print('M(R)')
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

    # CALCULATE LINE-OF-SIGHT VELOCITY AT EACH POINT (x_disk, y_disk) IN THE DISK (km/s)
    alpha = abs(np.arctan(y_disk / (np.cos(inc) * x_disk)))  # alpha meas. from +x (minor axis) toward +y (major axis)
    sign = x_disk / abs(x_disk)  # (+x now back to redshifted side, so don't need extra minus sign back in front!)
    v_los = sign * abs(vel * np.cos(alpha) * np.sin(inc))  # THIS IS CURRENTLY CORRECT

    # SET LINE-OF-SIGHT VELOCITY AT THE BLACK HOLE CENTER TO BE 0, SUCH THAT IT DOES NOT BLOW UP
    center = (R == 0.)  # Doing this is only relevant if we have pixel located exactly at the center
    v_los[center] = 0.  # if any point is at x_disk, y_disk = (0., 0.), set velocity there = 0.

    # CALCULATE VELOCITY PROFILES
    sigma = get_sig(r=R, sig0=sig_params[0], r0=sig_params[1], mu=sig_params[2], sig1=sig_params[3])[sig_type]

    # CONVERT v_los TO OBSERVED FREQUENCY MAP
    zred = vsys / constants.c_kms #vsys  # redshift
    freq_obs = (f_0 / (1+zred)) * (1 - v_los / constants.c_kms)  # convert v_los to f_obs

    # CONVERT OBSERVED DISPERSION (turbulent) TO FREQUENCY WIDTH
    sigma_grid = np.zeros(shape=R.shape) + sigma  # make sigma (whether already R-shaped or constant) R-shaped
    delta_freq_obs = (f_0 / (1 + zred)) * (sigma_grid / constants.c_kms)  # convert sigma to delta_f

    # WEIGHTS FOR LINE PROFILES: apply weights to gaussian velocity profiles for each subpixel
    weight = subpix_deconvolved  # [Jy/beam Hz]

    # WEIGHT CURRENTLY IN UNITS OF Jy/beam * Hz --> need to get it in units of Jy/beam to match data
    weight *= f_w / np.sqrt(2 * np.pi * delta_freq_obs**2)  # divide to get correct units
    # NOTE: MAYBE multiply by fstep here, after collapsing and lucy process, rather than during collapse
    # NOTE: would then need to multiply by ~1000 factor or something else large there, bc otherwise lucy cvg too fast

    # BEN_LUCY COMPARISON ONLY (only use for comparing to model with Ben's lucy map, which is in different units)
    if bl:  # (multiply by vel_step because Ben's lucy map is actually in Jy/beam km/s units)
        weight *= fstep * 6.783  # NOTE: 6.783 == beam size / channel width in km/s
        # channel width = 1.537983987579E+07 Hz --> v_width = c * (1 - f_0 / (fstep / (1+zred)))

    # BUILD GAUSSIAN LINE PROFILES!!!
    cube_model = np.zeros(shape=(len(freq_ax), len(freq_obs), len(freq_obs[0])))  # initialize model cube
    for fr in range(len(freq_ax)):
        cube_model[fr] = weight * np.exp(-(freq_ax[fr] - freq_obs) ** 2 / (2 * delta_freq_obs ** 2))

    # RE-SAMPLE BACK TO CORRECT PIXEL SCALE (take average of sxs sub-pixels for real alma pixel) --> intrinsic data cube
    if s == 1:
        intrinsic_cube = cube_model
    else:
        intrinsic_cube = rebin(cube_model, s)  # intrinsic_cube = block_reduce(cube_model, s, np.mean)

    # CONVERT INTRINSIC TO OBSERVED (convolve each slice of intrinsic_cube with alma beam --> observed data cube)
    convolved_cube = np.zeros(shape=intrinsic_cube.shape)  # approx ~1e-6 to 3e-6s per pixel
    for z in range(len(z_ax)):
        convolved_cube[z, :, :] = convolution.convolve(intrinsic_cube[z, :, :], beam)  # CONVOLVE CUBE WITH BEAM

    # ONLY WANT TO FIT WITHIN ELLIPTICAL REGION! APPLY ELLIPTICAL MASK TO MODEL CUBE & INPUT DATA
    ell_mask = ellipse_fitting(convolved_cube, rfit, xell, yell, resolution, theta_ell, q_ell)  # create ellipse mask
    convolved_cube *= ell_mask
    input_data_masked = input_data[zrange[0]:zrange[1], xyrange[2]:xyrange[3], xyrange[0]:xyrange[1]] * ell_mask

    # compare the data to the model by binning each in groups of dsxds pixels (separate from s)
    ell_4 = rebin(ell_mask, ds)
    data_4 = rebin(input_data_masked, ds)
    model_4 = rebin(convolved_cube, ds)

    plt.imshow(data_4[23], origin='lower')
    plt.colorbar()
    plt.show()

    indices = [[10,11], [11,11]] #UGC 2698: #[18,12] #NGC 3258: #[[13, 11]]  # [11, 13]
    print(data_4.shape, ell_4.shape)
    #for z in range(len(z_ax)):
    #    plt.imshow(data_4[z] * ell_4[0], origin='lower')
    #    plt.colorbar()
    #    plt.show()
    #            [12, 20], [20,12], [10, 10], [int(len(data_4[1]) / 2.), int(len(data_4[1]) / 2.)],
    #            [int(len(data_4[1]) / 2.) + 1, int(len(data_4[1]) / 2.) + 1]]
    # str(x_loc + xyrange[0])
    compare(data_4, model_4, freq_ax / (10**9), indices, f_0 / (10**9 * (1+zred)), '', noise, ers=True)
    #compare(data_4, model_4, freq_ax / 1e9, indices, (f_0 / (1+(6421. / constants.c_kms))) / 1e9, 'run2 peak2', noise)

    # '''  #
    # Collapsed cube flux map
    plt.title(r'Model flux map [Jy/beam]')
    plt.imshow(np.sum(model_4, axis=0), origin='lower')  # cs_unsum[16] - cs_unsum2[16]
    cbar = plt.colorbar()
    cbar.set_label(r'Flux [Jy/beam]', fontsize=20, rotation=90, labelpad=20)
    plt.xlabel(r'x [4x4 binned pixels]', fontsize=20)  # x [arcsec]  # x [4x4 binned pixels]
    plt.ylabel(r'y [4x4 binned pixels]', fontsize=20)  # y [arcsec]  # y [4x4 binned pixels]
    plt.show()
    # '''  #

    all_pix = np.ndarray.flatten(ell_4)  # all fitted pixels in each slice [len = 625 (yep)] [525 masked, 100 not]
    masked_pix = all_pix[all_pix != 0]  # all_pix, but this time only the pixels that are actually inside ellipse
    n_pts = len(masked_pix) * len(z_ax)  # (zrange[1] - zrange[0])  # total number of pixels being compared
    print(n_pts)
    print(len(masked_pix))

    chi_sq = 0.  # initialize chi^2
    cs = []  # chi^2 per slice
    cs_unsum = []
    z_ind = 0  # the actual index for the model-data comparison cubes
    for z in range(zrange[0], zrange[1]):  # for each relevant freq slice (ignore slices with only noise)
        chi_sq += np.sum((model_4[z_ind] - data_4[z_ind]) ** 2 / noise[z_ind] ** 2)  # calculate chisq!
        cs.append(np.sum((model_4[z_ind] - data_4[z_ind]) ** 2 / noise[z_ind] ** 2))  # chisq per slice
        cs_unsum.append((model_4[z_ind] - data_4[z_ind]) ** 2 / noise[z_ind] ** 2)
        z_ind += 1
    print(chi_sq)
    print(np.sum(ell_4), 'look!')

    #v_los = block_reduce(v_los, s, np.mean)
    #v_los = block_reduce(v_los, ds, np.mean) * ell_4[0]
    print(v_los.shape)
    #v_los *= ell_4[0]

    return v_los, x_obs_ac, y_obs_ac, cs, freq_ax, cs_unsum, ell_4


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
    with open(parfile, 'r') as pf:
        for line in pf:
            if not line.startswith('#'):
                cols = line.split()
                if len(cols) > 0:  # ignore empty lines
                    if cols[0] == 'free':  # free parameters are floats with priors
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

        return params, priors, qobs

    else:
        return params, priors


def model_prep(lucy_out=None, lucy_mask=None, lucy_b=None, lucy_in=None, lucy_o=None, lucy_it=None, data=None,
               data_mask=None, grid_size=None, res=1., x_std=1., y_std=1., pa=0., ds=4, zrange=None, xyerr=None):
    """

    :param lucy_out: output from running lucy on data cube and beam PSF; if it doesn't exist, create it!
    :param lucy_mask: file name of collapsed mask file to use in lucy process (if lucy_out doesn't exist)
    :param lucy_b: file name of input beam (built in make_beam()) to use for lucy process (if lucy_out doesn't exist)
    :param lucy_in: file name of input summed flux map to use for lucy process (if lucy_out doesn't exist)
    :param lucy_o: file name that will become the lucy_out, used in lucy process (if lucy_out doesn't exist)
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

    # MAKE ALMA BEAM  x_std=major; y_std=minor; rot=90-PA; auto-create the beam file lucy_b if it doesn't yet exist
    beam = make_beam(grid_size=grid_size, res=res, x_std=x_std, y_std=y_std, rot=np.deg2rad(90. - pa), fits_name=lucy_b)

    # COLLAPSE THE DATA CUBE
    fluxes, freq_ax, f_0, input_data, fstep = get_fluxes(data, data_mask, write_name=lucy_in)

    # DECONVOLVE FLUXES WITH BEAM PSF
    if not Path(lucy_out).exists():  # to use iraf, must "source activate iraf27" on command line
        t_pyraf = time.time()
        '''  #
        import pyraf
        from pyraf import iraf
        from iraf import stsdas, analysis, restore
        restore.lucy(lucy_in, lucy_b, lucy_o, niter=lucy_it, maskin=lucy_mask, goodpixval=1, limchisq=1E-3)  # lucy
        print('lucy process done in ' + str(time.time() - t_pyraf) + 's')  # ~10.6s
        if lucy_out is None:  # lucy_output should be defined, but just in case:
            lucy_out = lucy_o[:-3]  # don't include "[0]" on the file name ("[0]" is required for lucy process)
        # '''  #
        from skimage import restoration  # need to be in "three" environment (source activate three == tres)
        hduin = fits.open(lucy_in)
        l_in = hduin[0].data
        hdub = fits.open(lucy_b)
        l_b = hdub[0].data
        hduin.close()
        hdub.close()
        # https://github.com/scikit-image/scikit-image/issues/2551 (it was returning nans)
        # add to skimage/restoration/deconvolution.py (before line 389): relative_blur[np.isnan(relative_blur)] = 0
        l_o = restoration.richardson_lucy(l_in, l_b, lucy_it, clip=False)  # need clip=False
        print('lucy process done in ' + str(time.time() - t_pyraf) + 's')  # ~1s

        hdu1 = fits.PrimaryHDU(l_o)
        hdul1 = fits.HDUList([hdu1])
        hdul1.writeto(lucy_out)  # write out to lucy_out file

    hdu = fits.open(lucy_out)
    lucy_out = hdu[0].data
    hdu.close()

    noise_4 = rebin(input_data, ds)
    noise = []  # ESTIMATE NOISE (RMS) IN ORIGINAL DATA CUBE [z, y, x]  # For large N, Variance ~= std^2
    for z in range(zrange[0], zrange[1]):  # for each relevant freq slice
        noise.append(np.std(noise_4[z, xyerr[2]:xyerr[3], xyerr[0]:xyerr[1]]))  # ~variance

    return lucy_mask, lucy_out, beam, fluxes, freq_ax, f_0, fstep, input_data, noise


def compare(data, model, z_ax, inds_to_try2, v_sys, title, noise, ers=False, other_vsys=False):

    for i in range(len(inds_to_try2)):
        print(inds_to_try2[i][0], inds_to_try2[i][1])
        plt.plot(z_ax, model[:, inds_to_try2[i][1], inds_to_try2[i][0]], 'r*', label=r'Model')  # r-
        if ers:
            plt.plot(z_ax, noise, 'ko', label=r'Noise')
        plt.plot(z_ax, data[:, inds_to_try2[i][1], inds_to_try2[i][0]], 'b+', label=r'Data')  # b:
        plt.axvline(x=v_sys, color='k', label=r'v$_{\text{sys}}$')
        if other_vsys:
            plt.axvline(x=(f_0 / (1+(6421./(2.99792458*1e5)))) / 1e9, color='k', ls='--',
                        label=r'v$_{\text{sys}}=6421$ km/s')
        # plt.title(str(inds_to_try2[i][0]) + ', ' + str(inds_to_try2[i][1]))  # ('no x,y offset')
        plt.legend()
        plt.xlabel(r'Frequency [GHz]')
        plt.ylabel(r'Flux Density [Jy/beam]')
        chisq = np.sum((model[:, inds_to_try2[i][1], inds_to_try2[i][0]] - data[:, inds_to_try2[i][1], inds_to_try2[i][0]])**2
                       / np.asarray(noise)**2)  # calculate chisq!

        plt.title(title + '4x4-binned pixel (' + str(inds_to_try2[i][0]) + ',' + str(inds_to_try2[i][1]) +
            ') with ' + r'$\chi^2 = $' + str(chisq))
        #plt.title('xloc = ' + title + ', at 4x4-binned pixel (' + str(inds_to_try2[i][0]) + ',' + str(inds_to_try2[i][1]) +
        #          ') with ' + r'$\chi^2 = $' + str(chisq))
        plt.show()
        plt.close()


if __name__ == "__main__":
    # MAKE SURE I HAVE ACTIVATED THE iraf27 ENVIRONMENT!!!
    t0_true = time.time()
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument('--parfile')

    args = vars(parser.parse_args())

    # Load parameters from the parameter file
    params, priors = par_dicts(args['parfile'])

    # Make nice plot fonts
    rc('font', **{'family': 'serif', 'serif': ['Times']})
    rc('text', usetex=True)

    # CREATE OUTPUT FILENAME BASED ON INPUT PARAMETERS
    pars_str = ''
    for key in params:
        pars_str += str(params[key]) + '_'
    out = '/Users/jonathancohn/Documents/dyn_mod/outputs/NGC_3258_general_' + pars_str + '_subcube_ellmask_bl2.fits'

    # CREATE THINGS THAT ONLY NEED TO BE CALCULATED ONCE (collapse fluxes, lucy, noise)
    mod_ins = model_prep(data=params['data'], ds=params['ds'], lucy_out=params['lucy'], lucy_mask=params['lucy_mask'],
                         lucy_b=params['lucy_b'], lucy_in=params['lucy_in'], lucy_o=params['lucy_o'],
                         lucy_it=params['lucy_it'], data_mask=params['mask'], grid_size=params['gsize'],
                         res=params['resolution'], x_std=params['x_fwhm'], y_std=params['y_fwhm'], pa=params['PAbeam'],
                         xyerr=[params['xerr0'], params['xerr1'], params['yerr0'], params['yerr1']],
                         zrange=[params['zi'], params['zf']])

    lucy_mask, lucy_out, beam, fluxes, freq_ax, f_0, fstep, input_data, noise = mod_ins

    # CREATE MODEL CUBE!
    out = params['outname']
    outs = model_grid(resolution=params['resolution'], s=params['s'], x_loc=params['xloc'], y_loc=params['yloc'],
                      mbh=params['mbh'], inc=np.deg2rad(params['inc']), vsys=params['vsys'], dist=params['dist'],
                      theta=np.deg2rad(params['PAdisk']), input_data=input_data, lucy_out=lucy_out, out_name=out,
                      beam=beam, rfit=params['rfit'], enclosed_mass=params['mass'], ml_ratio=params['ml_ratio'],
                      sig_type=params['s_type'], zrange=[params['zi'], params['zf']], menc_type=params['mtype'],
                      sig_params=[params['sig0'], params['r0'], params['mu'], params['sig1']], f_w=params['f'],
                      ds=params['ds'], noise=noise, chi2=True, reduced=True, freq_ax=freq_ax, q_ell=params['q_ell'],
                      theta_ell=np.deg2rad(params['theta_ell']), fstep=fstep, f_0=f_0, bl=params['bl'],
                      xyrange=[params['xi'], params['xf'], params['yi'], params['yf']], xell=params['xell'],
                      yell=params['yell'])
    vlos, x_obs, y_obs, cs, f_ax, cs_unsum, ell4 = outs

    params, priors = par_dicts('/Users/jonathancohn/Documents/dyn_mod/param_files/ngc_3258binexc_xcl_params.txt')
    # CREATE THINGS THAT ONLY NEED TO BE CALCULATED ONCE (collapse fluxes, lucy, noise)
    mod_ins = model_prep(data=params['data'], ds=params['ds'], lucy_out=params['lucy'], lucy_mask=params['lucy_mask'],
                         lucy_b=params['lucy_b'], lucy_in=params['lucy_in'], lucy_o=params['lucy_o'],
                         lucy_it=params['lucy_it'], data_mask=params['mask'], grid_size=params['gsize'],
                         res=params['resolution'], x_std=params['x_fwhm'], y_std=params['y_fwhm'], pa=params['PAbeam'],
                         xyerr=[params['xerr0'], params['xerr1'], params['yerr0'], params['yerr1']],
                         zrange=[params['zi'], params['zf']])

    # 362.0412  # 362.0228
    out2 = model_grid(resolution=params['resolution'], s=params['s'], x_loc=params['xloc'], y_loc=params['yloc'],
                      mbh=params['mbh'], inc=np.deg2rad(params['inc']), vsys=params['vsys'], dist=params['dist'],
                      theta=np.deg2rad(params['PAdisk']), input_data=input_data, lucy_out=lucy_out, out_name=out,
                      beam=beam, rfit=params['rfit'], enclosed_mass=params['mass'], ml_ratio=params['ml_ratio'],
                      sig_type=params['s_type'], zrange=[params['zi'], params['zf']], menc_type=params['mtype'],
                      sig_params=[params['sig0'], params['r0'], params['mu'], params['sig1']], f_w=params['f'],
                      ds=params['ds'], noise=noise, chi2=True, reduced=True, freq_ax=freq_ax, q_ell=params['q_ell'],
                      theta_ell=np.deg2rad(params['theta_ell']), fstep=fstep, f_0=f_0, bl=params['bl'],
                      xyrange=[params['xi'], params['xf'], params['yi'], params['yf']],  xell=params['xell'],
                      yell=params['yell'])
    vlos2, x_obs, y_obs, cs2, f_ax, cs_unsum2, ell42 = out2

    vdiff = vlos - vlos2
    plt.imshow(vdiff, origin='lower', cmap='RdBu_r', vmax=np.amax([np.amax(vdiff), -np.amin(vdiff)]),
               vmin=-np.amax([np.amax(vdiff), -np.amin(vdiff)]))
    cbar = plt.colorbar()
    plt.title(r'vlos (median xloc) - vlos (lower CL xloc)')
    cbar.set_label(r'km/s', fontsize=20, rotation=0, labelpad=20)
    plt.xlabel(r'x [pixels]', fontsize=20)  # x [arcsec]
    plt.ylabel(r'y [pixels]', fontsize=20)  # y [arcsec]
    plt.show()
    maxes = []
    argm = []
    for i in range(len(vdiff)):
        maxes.append(np.amax(vdiff[i]))
        argm.append(np.argmax(vdiff[i]))
    j = np.argmax(maxes)
    print(argm[j], j)

    print(np.amax(vdiff), 'look')
    print(vdiff[11, 13], 'look')

    print(np.amax(cs_unsum[16] - cs_unsum2[16]), np.amin(cs_unsum[16] - cs_unsum2[16]))
    cdiff = (cs_unsum[16] - cs_unsum2[16])
    huh = cdiff > 0.0
    hm = cdiff == 0.0
    print(np.sum(huh), np.sum(hm))

    cd = []
    for i in range(len(cs_unsum)):
        cd.append(cs_unsum[i] - cs_unsum2[i])
    cd = np.asarray(cd)
    print(cd.shape)
    cd = np.sum(cd, axis=0)
    print(cd.shape)

    c1 = np.sum(cs_unsum, axis=0)

    all_pix = np.ndarray.flatten(ell4)  # all fitted pixels in each slice [len = 625 (yep)] [525 masked, 100 not]
    masked_pix = all_pix[all_pix != 0]  # all_pix, but this time only the pixels that are actually inside ellipse
    n_pts = len(masked_pix) * len(f_ax)
    red_cs = np.sum(cs_unsum) / (n_pts - 9.)

    # plt.title(r'median xloc $\chi^2$ - lower CL xloc $\chi^2$, total $\Delta \chi^2 = $' + str(round(np.sum(cd), 4)))
    plt.title(r'$\chi^2_\nu$ = ' + str(round(red_cs, 4)))
    # plt.title(r'$\chi^2_{\nu} = $' + str(round(np.sum(c1) / np.sum(ell4) / len(cs_unsum), 4)))
    #  (at $\nu = $' + str(round(f_ax[16] / 10**9, 4)) + ' GHz)'
    # plt.imshow(c1 / np.sum(ell4) / len(cs_unsum), origin='lower')  # cs_unsum[16] - cs_unsum2[16]
    plt.imshow(c1, origin='lower')  # cs_unsum[16] - cs_unsum2[16]
    cbar = plt.colorbar()
    # cbar.set_label(r'$\Delta$[(model - data)$^2 / $ noise$^2$]', fontsize=20, rotation=90, labelpad=20)
    cbar.set_label(r'$\chi^2 = $(model - data)$^2 / $ noise$^2', fontsize=20, rotation=90, labelpad=20)
    plt.xlabel(r'x [4x4 binned pixels]', fontsize=20)  # x [arcsec]  # x [4x4 binned pixels]
    plt.ylabel(r'y [4x4 binned pixels]', fontsize=20)  # y [arcsec]  # y [4x4 binned pixels]
    plt.show()

    plt.plot(f_ax / 10**9, np.asarray(cs) - np.asarray(cs2), 'b*', label=r'median xloc $\chi^2 - $ lower CL xloc $\chi^2$')
    # plt.plot(f_ax / 10**9, cs, 'b*', label='median xloc')
    # plt.plot(f_ax / 10**9, cs2, 'k+', label='0.0016 offset xloc')
    plt.axhline(y=np.median(np.asarray(cs) - np.asarray(cs2)), color='r', label='median difference')
    plt.axhline(y=np.mean(np.asarray(cs) - np.asarray(cs2)), color='k', label='mean difference')
    plt.legend(loc='lower left')
    plt.xlabel(r'Frequency [GHz]')
    plt.ylabel(r'$\Delta \chi^2$ per slice')
    plt.show()

    plt.imshow(vlos - vlos2, origin='lower', cmap='RdBu_r', vmax=np.amax([np.amax(vlos - vlos2), -np.amin(vlos - vlos2)]),
               vmin=-np.amax([np.amax(vlos - vlos2), -np.amin(vlos - vlos2)]),
               )
    cbar = plt.colorbar()
    cbar.set_label(r'km/s', fontsize=20, rotation=0, labelpad=20)
    plt.xlabel(r'x [pixels]', fontsize=20)  # x [arcsec]
    plt.ylabel(r'y [pixels]', fontsize=20)  # y [arcsec]
    plt.show()

