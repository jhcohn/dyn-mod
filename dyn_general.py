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
    # plt.imshow(data[27], origin='lower')  # <--THIS RESULTS IN THE RIGHT ORIENTATION!!!
    # plt.show()
    # print(oops)

    hdu_m = fits.open(data_mask)
    mask = hdu_m[0].data  # this is hdu_m[0].data, NOT hdu[0].data[0], unlike the data_cube above

    z_len = len(hdu[0].data[0])  # store the number of velocity slices in the data cube
    freq1 = float(hdu[0].header['CRVAL3'])  # starting frequency in the data cube
    f_step = float(hdu[0].header['CDELT3'])  # frequency step in the data cube
    f_0 = float(hdu[0].header['RESTFRQ'])
    freq_axis = np.arange(freq1, freq1 + (z_len * f_step), f_step)  # [bluest, ..., reddest]
    # print(freq_axis)
    # print(oop)
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
    collapsed_fluxes *= 1000.  # to bring to "regular" numbers for lucy process; will undo after lucy process

    # if write_name is not None:
    if not Path(write_name).exists():
        hdu = fits.PrimaryHDU(collapsed_fluxes)
        hdul = fits.HDUList([hdu])
        hdul.writeto(write_name)  # '/Users/jonathancohn/Documents/dyn_mod/' +
        hdul.close()
    return collapsed_fluxes, freq_axis, f_0, data


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

    :param data: input cube (data or model)
    :param n: size of pixel binning (e.g. n=4 --> rebins the date in sets of 4x4 pixels)
    :return: rebinned cube
    """

    rebinned = []
    for z in range(len(data)):
        subarrays = blockshaped(data[z, :, :], n, n)  # bin the data in groups of nxn (4x4) pixels
        # each pixel in the new, rebinned data cube is the mean of each 4x4 set of original pixels
        # reshaped = np.mean(np.mean(subarrays, axis=-1), axis=-1).reshape((int(len(data[0]) / 4.),
        #                                                                   int(len(data[0][0]) / 4.)))
        reshaped = n**2 * np.mean(np.mean(subarrays, axis=-1), axis=-1).reshape((int(len(data[0]) / n),
                                                                                 int(len(data[0][0]) / n)))
        rebinned.append(reshaped)
    print('rebinned')
    return np.asarray(rebinned)


def model_grid(resolution=0.05, s=10, x_off=0., y_off=0., mbh=4 * 10 ** 8, inc=np.deg2rad(60.), vsys=None, dist=17.,
               theta=np.deg2rad(-200.), data_cube=None, data_mask=None, lucy_output=None, out_name=None,
               enclosed_mass=None, ml_ratio=1., sig_type='flat', grid_size=31, sig_params=[1., 1., 1., 1.], f_w=1.,
               x_fwhm=0.052, y_fwhm=0.037, pa=64., menc_type=False,  lucy_in=None, lucy_b=None, lucy_mask=None,
               lucy_o=None):
    """
    Build grid for dynamical modeling!

    :param resolution: resolution of observations [arcsec/pixel]
    :param s: oversampling factor
    :param x_off: the location of the BH, offset from the center in the +x direction [arcsec]
    :param y_off: the location of the BH, offset from the center in the +y direction [arcsec]
    :param mbh: supermassive black hole mass [solar masses]
    :param inc: inclination of the galaxy [radians]
    :param vsys: if given, the systemic velocity [km/s]
    :param dist: distance to the galaxy [Mpc]
    :param theta: angle from the +x_obs axis counterclockwise to the blueshifted side of the disk (-x_disk) [radians]
    :param data_cube: input data cube of observations
    :param data_mask: input mask cube of each slice of the data, for constructing the weight map
    :param lucy_output: output from running lucy on data cube and beam PSF
    :param out_name: output name of the fits file to which to save the output v_los image (if None, don't save image)
    :param enclosed_mass: file including data of the enclosed stellar mass of the galaxy (1st column should be radius
        r in kpc and second column should be M_stars(<r) or velocity_circ(R) due to stars within R)
    :param ml_ratio: The mass-to-light ratio of the galaxy
    :param sig_type: code for the type of sigma_turb we're using. Can be 'flat', 'exp', or 'gauss'
    :param sig_params: list of parameters to be plugged into the get_sig() function. Number needed varies by sig_type
    :param f_w: multiplicative weight factor for the line profiles
    :param grid_size: the pixel grid size to use for the make_beam() function
    :param x_fwhm: FWHM in the x-direction of the ALMA beam (arcsec) to use for the make_beam() function
    :param y_fwhm: FWHM in the y-direction of the ALMA beam (arcsec) to use for the make_beam() function
    :param pa: position angle (in degrees) to use for the make_beam() function
    :param menc_type: Select how you incorporate enclosed stellar mass [True for mass(R) or False for velocity(R)]
    :param lucy_in: file name of input summed flux map to use for lucy process (if lucy_output is None)
    :param lucy_b: file name of input beam (built in make_beam function) to use for lucy process (if lucy_output is None)
    :param lucy_o: file name that will become the lucy_output, used in lucy process (if lucy_output is None)
    :param lucy_mask: file name of collapsed mask file to use in lucy process (if lucy_output is None)

    :return: observed line-of-sight velocity [km/s]
    """
    # INSTANTIATE ASTRONOMICAL CONSTANTS
    constants = Constants()

    # MAKE ALMA BEAM  # grid_size anything (must = collapsed datacube size for lucy); x_std=major; y_std=minor; rot=PA
    # beam = make_beam(grid_size=grid_size, res=resolution, x_std=x_fwhm, y_std=y_fwhm, rot=np.deg2rad(90. - pa))
    beam = make_beam(grid_size=grid_size, res=resolution, x_std=x_fwhm, y_std=y_fwhm, rot=np.deg2rad(90. - pa),
                     fits_name=lucy_b)  # this function now auto-creates the beam file lucy_b if it doesn't yet exist

    # COLLAPSE THE DATA CUBE
    fluxes, freq_ax, f_0, input_data = get_fluxes(data_cube, data_mask, write_name=lucy_in)
    # fluxes, freq_ax, f_0, input_data = get_fluxes(data_cube, data_mask)
    print(fluxes.shape)

    # plt.imshow(fluxes, origin='lower')
    # plt.colorbar()
    # plt.show()

    # DECONVOLVE FLUXES WITH BEAM PSF
    # to use pyraf, must be in the "three" environment ("source activate three" or "tres" on command line)
    # to use iraf, must ALSO "source activate iraf27" on command line
    if not Path(lucy_output).exists():
        t_pyraf = time.time()
        import pyraf
        from pyraf import iraf
        from iraf import stsdas, analysis, restore  # THIS WORKED!!!!
        restore.lucy(lucy_in, lucy_b, lucy_o, niter=10, maskin=lucy_mask, goodpixval=1, limchisq=1E-3)
        # CONFIRMED THIS WORKS!!!!!!!
        print('lucy process done in ' + str(time.time() - t_pyraf) + 's')  # ~10.6s
        if lucy_output is None:  # lucy_output should be defined, but just in case:
            lucy_output = lucy_o[:-3]  # don't include "[0]" that's required on the end for lucy
        # print(oops)

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
    print('deconvolution took {0} s'.format(time.time() - t0))  # ~0.3 s (7.5s for 1280x1280 array)
    # plt.imshow(subpix_deconvolved, origin='lower')  # looks good!  # extent=[xmin, xmax, ymin, ymax]
    # plt.show()

    # GAUSSIAN STEP
    # get gaussian velocity profile for each subpixel, apply weights to gaussians (weights = subpix_deconvolved output)

    # SET UP VELOCITY AXIS
    if vsys is None:
        v_sys = constants.H0 * dist
    else:
        v_sys = vsys

    # convert from frequency (Hz) to velocity (km/s), with freq_ax in Hz
    print(freq_ax[0], freq_ax[-1])
    # f_0 = 2.305380000000e11  # intrinsic frequency of CO(2-1) line
    z_ax = np.asarray([v_sys + ((f_0 - freq) / freq) * (constants.c / constants.m_per_km) for freq in freq_ax])  # v_opt
    # z_ax = np.asarray([v_sys + ((f_0 - freq)/f_0) * (constants.c / constants.m_per_km) for freq in freq_ax])  # v_rad
    # print(z_ax[1] - z_ax[0])  # 20.1 km/s yay! (newfiles --> 20km/s [19.9917398153], [19.8979441208] --> okay good?)
    # print(oops)

    # SET UP OBSERVATION AXES
    # initialize all values along axes at 0., but with a length equal to axis length [arcsec] * oversampling factor /
    # resolution [arcsec / pixel]  --> units of pixels along the observed axes
    y_obs = [0.] * len(lucy_out) * s
    x_obs = [0.] * len(lucy_out[0]) * s

    # set center of the observed axes (find the central pixel number along each axis)
    if len(x_obs) % 2. == 0:  # if even
        x_ctr = (len(x_obs)) / 2.  # set the center of the axes (in pixel number)
        for i in range(len(x_obs)):
            x_obs[i] = resolution * (i - x_ctr) / s  # (arcsec/pix) * N_subpixels / (subpixels/pix) = arcsec
    else:  # elif odd
        x_ctr = (len(x_obs) + 1.) / 2.  # +1 bc python starts counting at 0
        for i in range(len(x_obs)):
            x_obs[i] = resolution * ((i + 1) - x_ctr) / s  # (arcsec/pix) * N_subpixels / (subpixels/pix) = arcsec
    # repeat for y-axis
    if len(y_obs) % 2. == 0:
        y_ctr = (len(y_obs)) / 2.
        for i in range(len(y_obs)):
            y_obs[i] = resolution * (i - y_ctr) / s
            # y_obs[i] = resolution * (y_ctr - i) / s
    else:
        y_ctr = (len(y_obs) + 1.) / 2.
        for i in range(len(y_obs)):
            y_obs[i] = resolution * ((i + 1) - y_ctr) / s
            # y_obs[i] = resolution * (y_ctr - (i + 1)) / s

    # SET BH POSITION [in arcsec], based on the input offset values
    # DON'T divide offset positions by s, unless offset positions are in subpixels instead of pixels
    x_bhctr = (x_off - x_ctr / s) * resolution
    y_bhctr = (y_off - y_ctr / s) * resolution
    # x_bhctr = x_off  # NEW: using arcsec inputs!
    # y_bhctr = y_off  # NEW: using arcsec inputs!

    # CONVERT FROM ARCSEC TO PHYSICAL UNITS (pc)
    # tan(angle) = x/d where d=dist and x=disk_radius --> x = d*tan(angle), where angle = arcsec / arcsec_per_rad
    # convert BH position from arcsec to pc
    x_bhctr = dist * 10 ** 6 * np.tan(x_bhctr / constants.arcsec_per_rad)
    y_bhctr = dist * 10 ** 6 * np.tan(y_bhctr / constants.arcsec_per_rad)
    print('BH is at [pc]: ', x_bhctr, y_bhctr)

    # convert all x,y observed grid positions to pc
    x_arcsec = x_obs
    y_arcsec = y_obs
    x_obs = np.asarray([dist * 10 ** 6 * np.tan(x / constants.arcsec_per_rad) for x in x_obs])  # 206265 arcsec/rad
    y_obs = np.asarray([dist * 10 ** 6 * np.tan(y / constants.arcsec_per_rad) for y in y_obs])  # 206265 arcsec/rad


    # at each x,y spot in grid, calculate what x_disk and y_disk are, then calculate R, v, etc.
    # CONVERT FROM x_obs, y_obs TO x_disk, y_disk (still in pc)
    x_disk = (x_obs[None, :] - x_bhctr) * np.cos(theta) + (y_obs[:, None] - y_bhctr) * np.sin(theta)  # 2d array
    y_disk = (y_obs[:, None] - y_bhctr) * np.cos(theta) - (x_obs[None, :] - x_bhctr) * np.sin(theta)  # 2d array
    print('x, y disk', x_disk.shape, y_disk.shape)

    # CALCULATE THE RADIUS (R) OF EACH POINT (x_disk, y_disk) IN THE DISK (pc)
    R = np.sqrt((y_disk ** 2 / np.cos(inc) ** 2) + x_disk ** 2)  # radius R of each point in the disk (2d array)
    # print(R.shape)

    '''  #
    # PRINT PVD
    pvd(data_cube, theta, z_ax, x_obs, R, v_sys)  # x_obs = x in arcsec
    print(oops)
    # '''  #

    '''  #
    # this looks right!!!!!
    plt.contourf(x_obs, y_obs, R, 600, vmin=np.amin(R), vmax=np.amax(R), cmap='viridis')
    plt.xlabel(r'x [pc]', fontsize=30)
    plt.ylabel(r'y [pc]', fontsize=30)
    plt.ylim(min(y_obs), max(y_obs))
    plt.xlim(min(x_obs), max(x_obs))
    plt.plot(y_bhctr, x_bhctr, 'k*', markersize=20)
    cbar = plt.colorbar()
    cbar.set_label(r'pc', fontsize=30, rotation=0, labelpad=25)  # pc,  # km/s
    plt.show()
    print(oops)
    # '''  #

    # CALCULATE ENCLOSED MASS BASED ON MBH AND ENCLOSED STELLAR MASS
    # CREATE A FUNCTION TO INTERPOLATE (AND EXTRAPOLATE) ENCLOSED STELLAR M(R)
    # THEN CALCULATE KEPLERIAN VELOCITY
    t_mass = time.time()
    if menc_type:  # if using a file with stellar mass(R)
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
    else:  # elif using a file with circular velocity due to stellar mass as a function of R
        # note: current file has velocities given in v_circ^2/(M/L) --> v_circ = np.sqrt(col * (M/L))
        # note: currently using model "B1" --> use 2nd col in the file (this file has 4 cols of different models)
        radii = []
        v_circ = []
        with open(enclosed_mass) as em:
            for line in em:
                cols = line.split()
                radii.append(float(cols[0]))  # file lists radii in pc
                v_circ.append(np.sqrt(float(cols[1])) * ml_ratio)  # km/s
        v_c_r = interpolate.interp1d(radii, v_circ, fill_value='extrapolate')  # create a function
        # print(v_c_r(1.), (v_c_r(5.) / ml_ratio)**2, (v_c_r(500) / ml_ratio)**2, (v_c_r(2000) / ml_ratio)**2)
        vel = v_c_r(R) + np.sqrt(constants.G_pc * mbh / R)  # use v_c_r function to interpolate velocity due to stars
        # BUCKET BUCKET check is ^this right???

    print('vel')
    print('Time elapsed in assigning enclosed masses is {0} s'.format(time.time() - t_mass))  # ~3.5s

    # CALCULATE LINE-OF-SIGHT VELOCITY AT EACH POINT (x_disk, y_disk) IN THE DISK (km/s)
    alpha = abs(np.arctan(y_disk / (np.cos(inc) * x_disk)))  # alpha meas. from +x (minor axis) toward +y (major axis)
    sign = x_disk / abs(x_disk)  # (+x now back to redshifted side, so don't need extra minus sign back in front!)
    v_los = sign * abs(vel * np.cos(alpha) * np.sin(inc))  # THIS IS CURRENTLY CORRECT
    print('los')

    # SET LINE-OF-SIGHT VELOCITY AT THE BLACK HOLE CENTER TO BE 0, SUCH THAT IT DOES NOT BLOW UP
    # if any point is at x_disk, y_disk = (0., 0.), set velocity there = 0.
    center = (R == 0.)  # Doing this is only relevant if we have pixel located exactly at the center
    v_los[center] = 0.
    # print(center, v_los[center])

    # CALCULATE OBSERVED VELOCITY
    v_obs = v_sys - v_los  # observed velocity v_obs at each point in the disk
    print('v_obs')

    print(np.amax(v_los), np.amin(v_los))
    plt.imshow(v_los, origin='lower', cmap='RdBu_r', vmax=np.amax([np.amax(v_los), -np.amin(v_los)]),
               vmin=-np.amax([np.amax(v_los), -np.amin(v_los)]))
    cbar = plt.colorbar()
    cbar.set_label(r'km/s', fontsize=30, rotation=0, labelpad=25)
    plt.show()

    # CALCULATE VELOCITY PROFILES
    sigma = get_sig(r=R, sig0=sig_params[0], r0=sig_params[1], mu=sig_params[2], sig1=sig_params[3])[sig_type]
    # print(sigma)

    zred = vsys / constants.c_kms  # 0.005210938295185531
    print(zred)
    print(f_0)

    # CONVERT v_los TO OBSERVED FREQUENCY MAP
    freq_obs = (f_0 / (1+zred)) * (1 - v_los / constants.c_kms)
    print(np.amax(freq_ax), np.amin(freq_ax))      # (229341791123.0, 227496210337.90479)
    print(np.amax(freq_obs), np.amin(freq_obs))    # (229757280000.37192, 227111459583.84235)

    plt.imshow(freq_obs, origin='lower', vmax=np.amax(freq_obs), vmin=np.amin(freq_obs), cmap='viridis',
               extent=[np.amin(x_obs), np.amax(x_obs), np.amin(y_obs), np.amax(y_obs)])
    plt.colorbar()
    plt.show()

    # 60th slice of data -> ~central slice (i.e. ~v_sys) --> freq_ax = 2.28434e11


    # CONVERT OBSERVED DISPERSION (turbulent) TO FREQUENCY WIDTH
    sigma_grid = np.zeros(shape=(v_los.shape)) + sigma
    delta_freq_obs = (f_0 / (1 + zred)) * (sigma_grid / constants.c_kms)  # 1e11 / 1e5 ~ 1e6

    print(np.amax(sigma), np.amin(sigma))
    print(np.amax(delta_freq_obs), np.amin(delta_freq_obs))
    print(np.argmax(delta_freq_obs, axis=-1))
    print(delta_freq_obs[np.argmax(delta_freq_obs, axis=0)])
    plt.imshow(delta_freq_obs, origin='lower', vmin=np.amin(delta_freq_obs), vmax=np.amax(delta_freq_obs))
    plt.colorbar()
    plt.show()
    print(freq_obs.shape)

    # print(len(freq_ax))
    # print(freq_ax)

    weight = subpix_deconvolved / 1000.  # dividing by 1000 bc multiplying map by 1000 earlier  [Jy/beam km/s]

    # WEIGHT CURRENTLY IN UNITS OF Jy/beam * Hz --> need to get it in units of Jy/beam to match data
    weight /= np.sqrt(2 * np.pi * delta_freq_obs**2)  # divide to get correct units

    # plt.imshow(weight, origin='lower')
    # plt.colorbar()
    # plt.show()

    # BUILD GAUSSIAN LINE PROFILES!!!
    t_mod = time.time()
    cube_model = np.zeros(shape=(len(freq_ax), len(freq_obs), len(freq_obs[0])))  # initialize model cube
    for fr in range(len(freq_ax)):
        print(fr)
        cube_model[fr] = weight * f_w * np.exp(-(freq_ax[fr] - freq_obs) ** 2 / (2 * delta_freq_obs ** 2))
    print('cube model constructed in ' + str(time.time() - t_mod) + ' s')  # 34s
    # print(cube_model.shape)

    # x_disk = (x_obs[None, :] - x_bhctr) * np.cos(theta) + (y_obs[:, None] - y_bhctr) * np.sin(theta)  # 2d array
    # y_disk = (y_obs[:, None] - y_bhctr) * np.cos(theta) - (x_obs[None, :] - x_bhctr) * np.sin(theta)  # 2d array

    # RESAMPLE
    # RE-SAMPLE BACK TO CORRECT PIXEL SCALE (take average of sxs sub-pixels for real alma pixel) --> intrinsic data cube
    t_z = time.time()
    if s == 1:
        intrinsic_cube = cube_model
    else:
        intrinsic_cube = rebin(cube_model, s)
    print("intrinsic cube done in {0} s".format(time.time() - t_z))  # 15s
    print("start to intrinsic done in {0} s".format(time.time() - t0))  # 47s (including stops to close/save figs)

    # CONVERT INTRINSIC TO OBSERVED
    # take velocity slice from intrinsic data cube, convolve with alma beam --> observed data cube
    convolved_cube = np.zeros(shape=intrinsic_cube.shape)  # 61, 300, 300
    # beam = make_beam(grid_size=35, x_std=0.044, y_std=0.039, rot=np.deg2rad(90-64.))
    ts = time.time()
    for z in range(len(z_ax)):
        print(z)
        tl = time.time()
        convolved_cube[z, :, :] = convolution.convolve(intrinsic_cube[z, :, :], beam)  # CONFIRMED!
        print("Convolution loop " + str(z) + " took {0} seconds".format(time.time() - tl))
    print('convolved! Total convolution loop took {0} seconds'.format(time.time() - ts))  # ~1.5s x len(z_ax) --> 175s
    # approx ~3e-6s per pixel. So estimate as len(z_ax)*len(x)*len(y)*3e-6 seconds (within a factor of 3)

    # WRITE OUT RESULTS TO FITS FILE
    if not Path(out_name).exists():
        hdu = fits.PrimaryHDU(convolved_cube)
        hdul = fits.HDUList([hdu])
        hdul.writeto(out_name)
        print('written!')
    # '''  #

    return convolved_cube


if __name__ == "__main__":
    # MAKE SURE I HAVE ACTIVATED THE three AND iraf27 ENVIRONMENTS!!!
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument('--parfile')

    args = vars(parser.parse_args())
    # kwargs = {}
    # for key in args.keys():
    #     kwargs[key] = args[key]
    # print(kwargs['parfile'])

    params = {}
    fixed_pars = {}
    files = {}
    print(args['parfile'])

    with open(args['parfile'], 'r') as pf:
        for line in pf:
            print(line)
            if line.startswith('P'):
                par_names = line.split()[1:]  # ignore the "Value" str in the first column
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

    for n in range(len(fixed_names)):
        if fixed_names[n] == 's_type':
            fixed_pars[fixed_names[n]] = fixed_vals[n]
        elif fixed_names[n] == 'gsize' or fixed_names[n] == 's':
            fixed_pars[fixed_names[n]] = int(fixed_vals[n])
        else:
            fixed_pars[fixed_names[n]] = float(fixed_vals[n])

    for n in range(len(file_types)):
        files[file_types[n]] = file_names[n]
    # print(params)
    # print(fixed_pars)
    # print(files)
    '''
    # BLACK HOLE MASS (M_sol), RESOLUTION (ARCSEC), VELOCITY SPACING (KM/S)
    mbh = 6.64 * 10 ** 8  # 6.86 * 10 ** 8  # , 20.1 (v_spacing no longer necessary)
    resolution = 0.01  # was 0.01, now 0.04 because we're binning the data  # now binning later->0.01  # 0.05  # 0.044
    # DOUBLE CHECK THE ABOVE: 0.04 or 0.044

    # X, Y OFFSETS (PIXELS)
    x_off, y_off = 0., 0.  # 5., 17.  # +6., -4.  # 631. - 1280./2., 1280/2. - 651  # pixels  0., 0.

    # VELOCITY DISPERSION PARAMETERS
    sig0 = 32.1  # 22.2  # km/s
    r0, mu, sig1 = 1., 1., 1.  # using for sigma(R) (Gaussian or exponential)
    s_type = 'flat'

    # DISTANCE (Mpc), INCLINATION (rad), POSITION ANGLE (rad)
    dist = 22.3  # Mpc
    inc = np.deg2rad(85.2)  # 83.
    theta = np.deg2rad(26.7)  # + 180.)  # (180. + 116.7)  # 116.7 - 90.
    vsys = 1562.2  # km/s
    # Note: 22.3 * 10**6 * tan(1280 * 0.01 / 206265) = 1383.85 pc --> 1383.85 pc / 1280 pix = 1.08 pc/pix

    # OVERSAMPLING FACTOR
    s = 1  # set to 1 to match letter

    # ENCLOSED MASS FILE, CONSTANT BY WHICH TO MULTIPLY THE M/L RATIO
    enc_mass = 'ngc1332_enclosed_stellar_mass'
    ml_ratio = 7.83
    ml_const = ml_ratio / 7.35  # because enc_mass file assumes a ml_ratio of 7.35
    # ml_const = 1.065  # 7.83 / 7.35 # 7.53 / 7.35 (1.024)

    # ALMA BEAM PARAMS
    gsize = 31  # size of grid (must be odd)
    x_fwhm = 0.052  # arcsec
    y_fwhm = 0.037  # arcsec
    pa = 64.  # position angle (deg)

    # MAKE ALMA BEAM  # grid_size anything (must = collapsed datacube size for lucy); x_std=major; y_std=minor; rot=PA
    beam = make_beam(grid_size=gsize, res=resolution, x_std=x_fwhm, y_std=y_fwhm, rot=np.deg2rad(90. - pa))
    # beam = make_beam(grid_size=gsize, res=resolution, x_std=x_fwhm, y_std=y_fwhm, rot=np.deg2rad(90. - pa),
    #                  fits_name='newfiles_fullsize_beam' + str(gsize) + 'res_fwhm.fits')

    cube = '/Users/jonathancohn/Documents/dyn_mod/NGC_1332_newfiles/NGC1332_CO21_C3_MS_bri_20kms.pbcor.fits'
    d_mask = '/Users/jonathancohn/Documents/dyn_mod/NGC_1332_newfiles/NGC1332_CO21_C3_MS_bri_20kms_strictmask.mask.fits'
    # lucy = '/Users/jonathancohn/Documents/dyn_mod/newfiles_fullsize_masked_xy_beam31resfwhm_1000_limchi1e-3lucy_collapsedmask_n5.fits'
    # lucy = '/Users/jonathancohn/Documents/dyn_mod/newfiles_fullsize_masked_xy_beam31resfwhm_1000_limchi1e-3lucy_summed_n5.fits'
    lucy = '/Users/jonathancohn/Documents/dyn_mod/newfiles_fullsize_masked_xy_beam31resfwhm1_1000_limchi1e-3lucy_summed_n5.fits'
    # beam = 'newfiles_beam31res_fwhm.fits'
    out = '/Users/jonathancohn/Documents/dyn_mod/NGC_1332_freqcube_summed_apconv_n5_beam' + str(gsize) + 'fwhm_spline_s' + str(
        s) + '.fits'

    lucy_in = '/Users/jonathancohn/Documents/dyn_mod/NGC1332_newfiles_fullsize_masked_summed_1000.fits'
    lucy_b = '/Users/jonathancohn/Documents/dyn_mod/newfiles_fullsize_beam31res_fwhm.fits'
    lucy_o = '/Users/jonathancohn/Documents/dyn_mod/pyraf_out_n5.fits[0]'
    lucy_mask = '/Users/jonathancohn/Documents/dyn_mod/NGC_1332_newfiles/collapsed_mask_fullsize.fits'
    '''
    # Make nice plot fonts
    rc('font', **{'family': 'serif', 'serif': ['Times']})
    rc('text', usetex=True)

    # CREATE OUTNAME BASED ON INPUT PARS
    pars_str = ''
    for key in params:
        pars_str += str(params[key]) + '_'
    out = '/Users/jonathancohn/Documents/dyn_mod/outputs/NGC_3258_general_' + pars_str + '_take2.fits'

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
    out_cube = model_grid(resolution=fixed_pars['resolution'], s=fixed_pars['s'], x_off=params['xoff'],
                          y_off=params['yoff'], mbh=params['mbh'], inc=np.deg2rad(params['inc']), vsys=params['vsys'],
                          dist=fixed_pars['dist'], theta=np.deg2rad(params['PAdisk']), data_cube=files['data'],
                          data_mask=files['mask'], lucy_output=files['lucy'], out_name=out, ml_ratio=params['ml_ratio'],
                          enclosed_mass=files['mass'], menc_type=files['mtype']==True, sig_type=fixed_pars['s_type'],
                          grid_size=fixed_pars['gsize'], x_fwhm=fixed_pars['x_fwhm'], y_fwhm=fixed_pars['y_fwhm'],
                          pa=fixed_pars['PAbeam'], sig_params=[params['sig0'], params['r0'], params['mu'], params['sig1']],
                          f_w=params['f'], lucy_in=files['lucy_in'], lucy_b=files['lucy_b'], lucy_o=files['lucy_o'],
                          lucy_mask=files['lucy_mask'])

    # /Users/jonathancohn/Documents/dyn_mod/ngc3258_general_lucyout_n10.fits
    '''
    out_cube = model_grid(resolution=resolution, s=s, x_off=x_off, y_off=y_off, mbh=mbh, inc=inc, dist=dist, vsys=vsys,
                          theta=theta, data_cube=cube, data_mask=d_mask, lucy_output=lucy, out_name=out, incl_fig=0,
                          ml_const=ml_const, enclosed_mass=enc_mass, sig_type=s_type, grid_size=gsize, x_fwhm=x_fwhm,
                          y_fwhm=y_fwhm, pa=pa, sig_params=[sig0, r0, mu, sig1], lucy_in=lucy_in, lucy_b=lucy_b,
                          lucy_o=lucy_o, lucy_mask=lucy_mask)
    '''