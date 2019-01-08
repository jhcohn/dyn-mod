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
from plotbin import display_pixels as dp
from regions import read_crtf, CRTFParser, DS9Parser, read_ds9, PolygonPixelRegion, PixCoord



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


def check_beam(beam_name):  # BUCKET UNCONFIRMED (may need to flip y like I did for data)
    hdu = fits.open(beam_name)
    data = hdu[0].data  # header = hdu[0].header
    print(data.shape)
    print(data)
    # for i in range(len(data[0])):
    #     plt.plot(np.arange(len(data[0])), data[i, :])
    plt.plot(np.arange(len(data)), data[:, int(len(data) / 2.)])
    plt.show()


def make_beam(grid_size=100, amp=1., x0=0., y0=0., x_std=1., y_std=1., rot=0., fits_name=None):  # BUCKET UNCONFIRMED
    """
    Use to generate a beam psf (and to create a beam fits file to use in lucy

    :param grid_size: size
    :param amp: amplitude of the 2d gaussian
    :param x0: mean of x axis of 2d gaussian
    :param y0: mean of y axis of 2d gaussian
    :param x_std: standard deviation of Gaussian in x
    :param y_std: standard deviation of Gaussian in y
    :param rot: rotation angle in radians
    :param fits_name: this name will be the filename to which the beam fits file is written (if None, write no file)

    return the synthesized beam
    """

    # SET UP MESHGRID
    x_beam = np.linspace(-1., 1., grid_size)
    y_beam = np.linspace(-1., 1., grid_size)
    xx, yy = np.meshgrid(x_beam, y_beam)

    # SET UP PSF 2D GAUSSIAN VARIABLES
    a = np.cos(rot) ** 2 / (2 * x_std ** 2) + np.sin(rot) ** 2 / (2 * y_std ** 2)
    b = -np.sin(2 * rot) / (4 * x_std ** 2) + np.sin(2 * rot) / (4 * y_std ** 2)
    c = np.sin(rot) ** 2 / (2 * x_std ** 2) + np.cos(rot) ** 2 / (2 * y_std ** 2)

    # CALCULATE PSF, NORMALIZE IT TO AMPLITUDE
    synth_beam = np.exp(-(a * (xx - x0) ** 2 + 2 * b * (xx - x0) * (yy - y0) + c * (yy - y0) ** 2))
    A = amp / np.amax(synth_beam)
    synth_beam *= A

    # IF write_fits, WRITE PSF TO FITS FILE
    if fits_name is not None:
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
        reshaped = np.mean(np.mean(subarrays, axis=-1), axis=-1).reshape((int(len(data[0])/4.),
                                                                          int(len(data[0][0])/4.)))
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
            elif (x < len(data[0, 0, :]) / 2. and y < len(data[0, :, 0]) / 2.) or\
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

    data_masked = np.asarray([data[z][:,:] * col[:,:] for z in range(len(data))])
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


def get_fluxes(data_cube, int_slice1=14, int_slice2=63, x_off=0., y_off=0., resolution=0., write_name=None):  # CONFIRMED
    """
    Use to integrate line profiles to get fluxes from data cube!

    :param data_cube: input data cube of observations
    :param write_name: name of fits file to which to write collapsed cube (if None, write no file)

    :return: collapsed data cube (i.e. integrated line profiles i.e. area under curve i.e. flux), z_length (len(z_ax))
    """
    # hdu = fits.open(data_cube)
    hdu = fits.open('/Users/jonathancohn/Documents/dyn_mod/NGC_1332_newfiles/NGC1332_CO21_C3_MS_bri_20kms.pbcor.fits')
    # print(hdu[0].header)  # header
    # CTYPE1: RA (x_obs)
    # CRVAL1 = 5.157217100000Ee+1 [RA of reference pix]
    # CDELT1 = 1.944444444444E-5 [increment at the reference pixel]
    # CRPIX1 = 1.510000000000E+2 [coordinate of reference pixel]
    # CUNIT1 = DEG
    # CTYPE2 = DEC (y_obs)
    # CRVAL2 = -2.133536900000E+1
    # CDELT2 = CDELT1
    # CRPIX2 = CRPIS1
    # CUNIT2 = DEG
    # CTYPE3 = FREQ
    # CRVAL3 = 2.297843878460E+11
    # CDELT3 = -1.537983987576E+7  # 15.4 Mhz <-> 20.1 km/s
    # CRPIX3 = 1.
    # CUNIT3 = Hz

    # len(hdu[0].data[0]) = 61 (velocity!); len(hdu[0].data[0][1]) = 300 (x?), len(hdu[0].data[0][1][299]) = 300 (y?)
    data = hdu[0].data[0]
    # plt.imshow(data[27], origin='lower')  # <--THIS RESULTS IN THE RIGHT ORIENTATION!!!
    # plt.show()
    # print(oops)

    hdu_m = fits.open('/Users/jonathancohn/Documents/dyn_mod/NGC_1332_newfiles/NGC1332_CO21_C3_MS_bri_20kms_strictmask.mask.fits')
    mask = hdu_m[0].data[0]

    # data1 = data[:, ::-1, :]
    # print(data1[60][-250-1][200], 'flipped')
    # print(data[60][250][200], 'orig')

    # REBIN IN GROUPS OF 4x4 PIXELS
    rebinned = []  # np.zeros(shape=(len(z_ax), len(fluxes), len(fluxes[0])))
    rebinned_m = []
    t_rebin = time.time()
    for z in range(len(hdu[0].data[0])):
        subarrays = blockshaped(data[z, :, :], 4, 4)  # bin the data in groups of 4x4 pixels
        subarrays_m = blockshaped(mask[z, :, :], 4, 4)
        # data[z, ::-1, :] flips the y-axis, which is stored in python in the reverse order (problem.png)

        # Take the mean along the first axis of each subarray in subarrays, then take the mean along the
        # other (s-length) axis, such that what remains is a 1d array of the means of the sxs subarrays. Then reshape
        # into the correct lengths
        reshaped = np.mean(np.mean(subarrays, axis=-1), axis=-1).reshape((int(len(data[0])/4.),
                                                                          int(len(data[0][0])/4.)))
        reshaped_m = np.mean(np.mean(subarrays_m, axis=-1), axis=-1).reshape((int(len(mask[0])/4.),
                                                                              int(len(mask[0][0])/4.)))
        rebinned.append(reshaped)
        rebinned_m.append(reshaped_m)
    print("Rebinning the cube done in {0} s".format(time.time() - t_rebin))  # 0.5 s
    data = np.asarray(rebinned)
    mask = np.asarray(rebinned_m)
    # print(data.shape)
    # plt.imshow(data[20, :, :])
    # plt.show()
    # LOOKS GOOD!

    combined = []
    for zi in range(len(data)):
        combined.append(data[zi] * mask[zi])
    combined = np.asarray(combined)
    # plt.imshow(combined[35], origin='lower')
    # plt.show()
    # LOOKS GOOD!


    '''
    ### BUCKET ADDING IN R
    constants = Constants()

    y_obs = [0.] * len(data[0])
    x_obs = [0.] * len(data[0][0])

    # set center of the observed axes (find the central pixel number along each axis)
    if len(x_obs) % 2. == 0:  # if even
        x_ctr = (len(x_obs)) / 2.  # set the center of the axes (in pixel number)
        for i in range(len(x_obs)):
            x_obs[i] = resolution * (i - x_ctr)  # (arcsec/pix) * N_pixels = arcsec
    else:  # elif odd
        x_ctr = (len(x_obs) + 1.) / 2.  # +1 bc python starts counting at 0
        for i in range(len(x_obs)):
            x_obs[i] = resolution * ((i + 1) - x_ctr)  # (arcsec/pix) * N_pixels = arcsec
    # repeat for y-axis
    if len(y_obs) % 2. == 0:
        y_ctr = (len(y_obs)) / 2.
        for i in range(len(y_obs)):
            y_obs[i] = resolution * (i - y_ctr)
            # y_obs[i] = resolution * (y_ctr - i)
    else:
        y_ctr = (len(y_obs) + 1.) / 2.
        for i in range(len(y_obs)):
            y_obs[i] = resolution * ((i + 1) - y_ctr)
            # y_obs[i] = resolution * (y_ctr - (i + 1))

    # SET BH POSITION [in arcsec], based on the input offset values
    # DON'T divide offset positions by s, unless offset positions are in subpixels instead of pixels
    x_bhctr = x_off * resolution
    y_bhctr = y_off * resolution

    # CONVERT FROM ARCSEC TO PHYSICAL UNITS (pc)
    # tan(angle) = x/d where d=dist and x=disk_radius --> x = d*tan(angle), where angle = arcsec / arcsec_per_rad
    # convert BH position from arcsec to pc
    x_bhctr = dist * 10 ** 6 * np.tan(x_bhctr / constants.arcsec_per_rad)
    y_bhctr = dist * 10 ** 6 * np.tan(y_bhctr / constants.arcsec_per_rad)
    print('BH is at [pc]: ', x_bhctr, y_bhctr)

    # convert all x,y observed grid positions to pc
    x_obs = np.asarray([dist * 10 ** 6 * np.tan(x / constants.arcsec_per_rad) for x in x_obs])  # 206265 arcsec/rad
    y_obs = np.asarray([dist * 10 ** 6 * np.tan(y / constants.arcsec_per_rad) for y in y_obs])  # 206265 arcsec/rad
    # print((x_obs[1] - x_obs[0])*s)  # 7.56793447282
    # print(x_obs[0], x_obs[s])  # (-1134.5595077622734, -1126.9915732895627)

    # at each x,y spot in grid, calculate what x_disk and y_disk are, then calculate R, v, etc.
    # CONVERT FROM x_obs, y_obs TO x_disk, y_disk (still in pc)
    x_disk = (x_obs[None, :] - x_bhctr) * np.cos(theta) + (y_obs[:, None] - y_bhctr) * np.sin(theta)  # 2d array
    y_disk = (y_obs[:, None] - y_bhctr) * np.cos(theta) - (x_obs[None, :] - x_bhctr) * np.sin(theta)  # 2d array
    print('x, y disk', x_disk.shape, y_disk.shape)

    # CALCULATE THE RADIUS (R) OF EACH POINT (x_disk, y_disk) IN THE DISK (pc)
    R = np.sqrt((y_disk ** 2 / np.cos(inc) ** 2) + x_disk ** 2)  # radius R of each point in the disk (2d array)

    for i in range(len(data)):
        data[i, R > 50] = 0.
    ### BUCKET END ADDING IN R
    '''

    # collapsed_fluxes = integrate.simps(data[int_slice1:int_slice2], axis=0)  # according to my python terminal tests
    collapsed_fluxes = integrate.simps(combined, axis=0)
    plt.imshow(collapsed_fluxes, origin='lower')
    plt.show()
    # CORRECT ORIENTATION FOR PYTHON
    # NOTE: SLICES ALONE NOT GOOD ENOUGH!
    '''  #
    # Open CASA window (in terminal, type): /Applications/CASA.app/Contents/MacOS/casapy
    # THEN type:
    # viewer('/Users/jonathancohn/Documents/dyn_mod/NGC1332_01_casacopy.fits')
    # VIEWER/VISUALIZATION INFO:
    # https://casa.nrao.edu/casadocs/casa-5.1.1/image-cube-visualization/viewing-images-and-cubes
    # https://casa.nrao.edu/docs/UserMan/casa_cookbook008.html
    # IMPORT FITS INFO:
    # https://casa.nrao.edu/Release3.3.0/docs/UserMan/UserMansu302.html
    # REGIONS INFO:
    # https://casa.nrao.edu/Release4.1.0/doc/UserMan/UserMansu347.html
    # https://casa.nrao.edu/docs/UserMan/casa_cookbook008.html (cmd+f "Regions and the Region Manager")
    # https://casa.nrao.edu/casadocs/casa-5.1.0/image-cube-visualization/regions-in-the-viewer
    # READING REGIONS FILES INTO PYTHON:
    # https://media.readthedocs.org/pdf/astropy-regions/latest/astropy-regions.pdf (Chapter 9, pg 39)
    regions = read_ds9('/Users/jonathancohn/Documents/dyn_mod/regions/NGC1332_01_casacopy_slice14_take2.reg')
    print(regions)
    regions = regions[0]
    print(regions)
    artist = regions.as_artist()
    axes = plt.gca()
    axes.set_aspect('equal')
    axes.add_artist(artist)
    plt.show()
    print(oops)
    # '''  #

    '''  #
    # #CRTFv0 CASA Region Text Format version 0
    # poly [[51.57178622deg, -21.33521372deg], [51.57172392deg, -21.33516360deg], [51.57170693deg, -21.33517679deg], [51.57177206deg, -21.33523086deg]] coord=ICRS, corr=[I], linewidth=1, linestyle=-, symsize=1, symthick=1, color=magenta, font=Helvetica, fontsize=11, fontstyle=normal, usetex=false

    # TO READ IN: cut all extra words, kill space after poly,

    # BUCKET HAVING ISSUES READING IN CRTF FILE
    with open('/Users/jonathancohn/Documents/dyn_mod/regions/NGC1332_01_casacopy_slice14_take2.crtf', 'r') as crtf_file:
        for line in crtf_file:
            if line.startswith('#'):
                pass
            else:
                reg = CRTFParser(line)  # still get error: "Not a valid CRTF line: '{0}'.".format(line))"
    print(reg)
    # reg_string = 'circle[[42deg, 43deg], 3deg], coord=J2000, color=green '
    # print(CRTFParser(reg_string))
    # print(oops)
    reg = read_crtf('/Users/jonathancohn/Documents/dyn_mod/regions/NGC1332_01_casacopy_slice14_take2.crtf')
    artist = reg.as_artist()
    axes = plt.gca()
    axes.set_aspect('equal')
    axes.add_artist(artist)
    plt.show()
    # '''  #

    '''
    # BUCKET: checking adding in R
    # collapsed_fluxes = integrate.simps(data, axis=0)
    if R is not None:
        print(collapsed_fluxes.shape)
        plt.imshow(collapsed_fluxes, origin='lower')
        # plt.plot(len(collapsed_fluxes)/2., len(collapsed_fluxes[0])/2., 'w*')  # this is centered yay!
        plt.colorbar()
        plt.show()
        # print(oops)
    '''

    z_len = len(hdu[0].data[0])  # store the number of velocity slices in the data cube
    freq1 = float(hdu[0].header['CRVAL3'])
    f_step = float(hdu[0].header['CDELT3'])
    freq_axis = np.arange(freq1, freq1 + (z_len * f_step), f_step)  # [bluest, less blue, ..., reddest]
    print(freq1)
    hdu.close()

    if write_name is not None:
        hdu = fits.PrimaryHDU(collapsed_fluxes)
        # [::-1,:])  # HAVE TO FLIP THE ORIENTATION FOR FITS BC I HATE EVERYTHING, EXCEPT MAYBE NOT?
        hdul = fits.HDUList([hdu])
        hdul.writeto(write_name)

    return collapsed_fluxes, freq_axis, data


def blockshaped(arr, nrow, ncol):  # CONFIRMED
    h, w = arr.shape
    return arr.reshape(h // nrow, nrow, -1, ncol).swapaxes(1, 2).reshape(-1, nrow, ncol)


# BUCKET UNCONFIRMED: (*ALSO* NEED TO REDEFINE PARAMS)
def model_grid(resolution=0.05, s=10, x_off=0., y_off=0., mbh=4 * 10 ** 8, inc=np.deg2rad(60.), vsys=None, dist=17.,
               theta=np.deg2rad(-200.), data_cube=None, lucy_output=None, out_name=None, incl_fig=False,
               enclosed_mass=None, ml_const=1., sig_type='flat', beam=None, sig_params=[1., 1., 1., 1.]):
    """
    Build grid for dynamical modeling!

    :param resolution: resolution of observations [arcsec]
    :param s: oversampling factor
    :param x_off: the location of the BH, offset from the center in the +x direction [pixels] (taken directly from data)
    :param y_off: the location of the BH, offset from the center in the +y direction [pixels] (taken directly from data)
    :param mbh: supermassive black hole mass [solar masses]
    :param inc: inclination of the galaxy [radians]
    :param vsys: if given, the systemic velocity [km/s]
    :param dist: distance to the galaxy [Mpc]
    :param theta: angle from the redshifted side of the disk (+y_disk) counterclockwise to the +y_obs axis [radians]
        (angle input must be negative to go counterclockwise)
        :param theta: NOW: angle from the the +y_obs axis counterclockwise to the redshifted side of the disk (+y_disk)
        [radians] (angle input must be negative to go counterclockwise)
    :param data_cube: input data cube of observations
    :param lucy_output: output from running lucy on data cube and beam PSF
    :param out_name: output name of the fits file to which to save the output v_los image (if None, don't save image)
    :param incl_fig: if True, print figure of 2d plane of observed line-of-sight velocity
    :param enclosed_mass: file including data of the enclosed stellar mass of the galaxy (1st column should be radius
        r in kpc and second column should be M_stars(<r))
    :param ml_const: Constant by which to multiply the M/L ratio, in case the M/L ratio is not that which was used to
        calculate the enclosed stellar masses in the enclosed_mass file
    :param sig_type: code for the type of sigma_turb we're using. Can be 'flat', 'exp', or 'gauss'
    :param beam: the alma beam with which the data were observed, as output by the make_beam() function
    :param sig_params: list of parameters to be plugged into the get_sig() function. Number needed varies by sig_type

    :return: observed line-of-sight velocity [km/s]
    """
    # INSTANTIATE ASTRONOMICAL CONSTANTS
    constants = Constants()

    # COLLAPSE THE DATA CUBE
    # fluxes, freq_ax, data_rebinned = get_fluxes(data_cube, write_name='NGC1332_newdata_c1463_collapsed_xy.fits')
    fluxes, freq_ax, data_rebinned = get_fluxes(data_cube, x_off=x_off, y_off=y_off, resolution=resolution)
    # ^rebins data in 4x4 pixels

    # DECONVOLVE FLUXES WITH BEAM PSF
    # source activate iraf27; in /Users/jonathancohn/iraf/, type xgterm; in xgterm, load stsdas, analysis, restore
    # then: lucy input_image.fits psf.fits output_image.fits niter=15 [defaults for both adu and noise, play with niter]
    # currently done in iraf outside python. Output: lucy_output='lucy_out_n5.fits', for niter=5
    # NOTE: the data cube for NGC1332 has redshifted side at the bottom left
    hdu = fits.open(lucy_output)
    lucy_out = hdu[0].data
    hdu.close()
    # plt.imshow(lucy_out, origin='lower')  # looks correct
    # plt.show()

    # NOW INCLUDE ENCLOSED STELLAR MASS (interpolated below, after R is defined)
    radii = []
    m_stellar = []
    with open(enclosed_mass) as em:
        for line in em:
            cols = line.split()
            cols[1] = cols[1].replace('D', 'e')
            radii.append(float(cols[0]) * 10 ** 3)  # file lists radii in kpc; convert to pc
            m_stellar.append(float(cols[1]))  # solar masses

    # SUBPIXELS (RESHAPING DATA sxs)
    # take deconvolved flux map (output from lucy), assign each subpixel flux=(real pixel flux)/s**2
    # --> these are the weights to apply to gaussians
    print('start ')
    t0 = time.time()
    # subpix_deconvolved is identical to lucy_out, just with sxs subpixels for each pixel and the total flux conserved
    subpix_deconvolved = np.zeros(shape=(len(lucy_out) * s, len(lucy_out[0]) * s))  # 300*s, 300*s
    for ypix in range(len(lucy_out)):
        for xpix in range(len(lucy_out[0])):
            subpix_deconvolved[(ypix * s):(ypix + 1) * s, (xpix * s):(xpix + 1) * s] = lucy_out[ypix, xpix] / s ** 2
    print('deconvolution took {0} s'.format(time.time() - t0))  # ~0.3 s (7.5s for 1280x1280 array)
    # plt.imshow(subpix_deconvolved, origin='lower')  # looks good!
    # plt.show()

    # GAUSSIAN STEP
    # get gaussian velocity profile for each subpixel, apply weights to gaussians (weights = subpix_deconvolved output)

    # SET UP VELOCITY AXIS
    if vsys is None:
        v_sys = constants.H0 * dist
    else:
        v_sys = vsys

    # convert from frequency (Hz) to velocity (km/s), with freq_ax in Hz
    # CO(2-1) lands in 2.2937*10^11 Hz (15.4 * 10^6 Hz corresponds to 20.1 km/s  # from Barth+2016)
    f_0 = 2.29369132e11  # intrinsic frequency of CO(2-1) line
    z_ax = np.asarray([v_sys - ((freq - f_0)/f_0) * (constants.c / constants.m_per_km) for freq in freq_ax])
    # z_ax = np.asarray([v_sys - ((f_0 - freq) / freq) * (constants.c / constants.m_per_km) for freq in freq_ax])
    print(z_ax[1] - z_ax[0])  # 20.1 km/s yay!

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
    x_bhctr = x_off * resolution
    y_bhctr = y_off * resolution

    # CONVERT FROM ARCSEC TO PHYSICAL UNITS (pc)
    # tan(angle) = x/d where d=dist and x=disk_radius --> x = d*tan(angle), where angle = arcsec / arcsec_per_rad
    # convert BH position from arcsec to pc
    x_bhctr = dist * 10 ** 6 * np.tan(x_bhctr / constants.arcsec_per_rad)
    y_bhctr = dist * 10 ** 6 * np.tan(y_bhctr / constants.arcsec_per_rad)
    print('BH is at [pc]: ', x_bhctr, y_bhctr)

    # convert all x,y observed grid positions to pc
    x_obs = np.asarray([dist * 10 ** 6 * np.tan(x / constants.arcsec_per_rad) for x in x_obs])  # 206265 arcsec/rad
    y_obs = np.asarray([dist * 10 ** 6 * np.tan(y / constants.arcsec_per_rad) for y in y_obs])  # 206265 arcsec/rad
    # print((x_obs[1] - x_obs[0])*s)  # 7.56793447282
    # print(x_obs[0], x_obs[s])  # (-1134.5595077622734, -1126.9915732895627)

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
    plt.imshow(R, origin='lower')  # this looks right!
    plt.show()

    plt.imshow(y_disk, origin='lower')  # this looks right!
    plt.colorbar()
    plt.show()
    
    y_obs2 = np.zeros(shape=R.shape)  # this looks right!
    for i in range(len(y_obs)):
        y_obs2[i, :] = y_obs[i]
    plt.imshow(y_obs2, origin='lower')
    cbar = plt.colorbar()
    plt.show()
    x_obs2 = np.zeros(shape=R.shape)  # this looks right!
    for i in range(len(x_obs)):
        x_obs2[:, i] = x_obs[i]
    plt.imshow(x_obs2, origin='lower')
    cbar = plt.colorbar()
    plt.show()
    
    plt.contourf(x_disk, y_disk, R)  # this looks right!
    cbar = plt.colorbar()
    plt.show()
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
    cbar.set_label(r'km/s', fontsize=30, rotation=0, labelpad=25)  # pc,  # km/s
    plt.show()
    print(oops)
    # '''  #

    # CALCULATE ENCLOSED MASS BASED ON MBH AND ENCLOSED STELLAR MASS
    # CREATE A FUNCTION TO INTERPOLATE (AND EXTRAPOLATE) ENCLOSED STELLAR M(R)
    t_mass = time.time()
    m_star_r = interpolate.interp1d(radii, m_stellar, fill_value='extrapolate')  # this creates a function
    m_R = mbh + ml_const * m_star_r(R)  # Use m_star_r function to interpolate mass at all radii R (2d array)
    ''' #
    Rplot = R.ravel()
    m_Rplot = m_R.ravel()
    #for row in range(len(R)):
    #    plt.plot(R[row], m_R[row] - mbh, 'k-')
    plt.scatter(Rplot, m_Rplot)
    plt.yscale('log')
    plt.ylabel(r'$\log_{10}$[M$_*$($<$R)/M$_{\odot}$]', fontsize=20)
    plt.xlabel(r'R [pc]', fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.tick_params(axis='both', which='minor', labelsize=10)
    plt.show()
    print(oops)
    # '''  #
    # print(m_R)
    # print(m_R.shape)
    print('Time elapsed in assigning enclosed masses is {0} s'.format(time.time() - t_mass))  # ~3.5s

    # DRAW ELLIPSE  # UNCONFIRMED!!!!!
    from matplotlib import patches
    # USE 50pc FOR MODEL FITTING
    # USE 4.3" x 0.7" FOR FIG 2 REPLICATION
    # major = 50
    # minor = 50*np.cos(inc)
    major = dist * 10 ** 6 * np.tan(4.3 / constants.arcsec_per_rad)
    minor = dist * 10 ** 6 * np.tan(0.7 / constants.arcsec_per_rad)
    par = np.linspace(0, 2 * np.pi, 100)
    ell = np.asarray([major * np.cos(par), minor * np.sin(par)])
    theta_rot = theta  # 26.7*np.pi/180.  # 153*pi/180. (90.+116.7)*np.pi/180.
    rot_mat = np.asarray([[np.cos(theta_rot), -np.sin(theta_rot)], [np.sin(theta_rot), np.cos(theta_rot)]])
    ell_rot = np.zeros((2, ell.shape[1]))
    for i in range(ell.shape[1]):
        ell_rot[:, i] = np.dot(rot_mat, ell[:, i])
    # ell = patches.Ellipse((x_bhctr, y_bhctr), width=50, height=50*np.cos(inc), angle=theta*180./np.pi)
    print(ell_rot)

    # CALCULATE KEPLERIAN VELOCITY OF ANY POINT (x_disk, y_disk) IN THE DISK WITH RADIUS R (km/s)
    vel = np.sqrt(constants.G_pc * m_R / R)  # Keplerian velocity vel at each point in the disk
    print('vel')

    # CALCULATE LINE-OF-SIGHT VELOCITY AT EACH POINT (x_disk, y_disk) IN THE DISK (km/s)
    # alpha = POSITION ANGLE AROUND THE DISK, measured from +x (minor axis) toward +y (major axis)
    # alpha = np.arctan((y_disk * np.cos(inc)) / x_disk)
    # v_obs = v_sys - v_los = v_sys - v(R)*sin(i)*cos(alpha)
    # = v_sys - sqrt(GM/R)*sin(i)*cos(alpha)
    # = v_sys - sqrt(GM)*sin(i)*cos(alpha)/[(x_obs / cos(i))**2 + y_obs**2]^(1/4)
    # = v_sys - sqrt(GM)*sin(i)*[1 /sqrt((x_obs / (y_obs*cos(i)))**2 + 1)]/[(x_obs/cos(i))**2 + y_obs**2]^(1/4)
    # NOTE: 1/sqrt((x_obs / (y_obs*cos(i)))**2 + 1) = y_obs/sqrt((x_obs/cos(i))**2 + y_obs**2) = y_obs / sqrt(R)
    # v_obs = v_sys - sqrt(GM)*sin(i)*y_obs / [(x_obs / cos(i))**2 + y_obs**2]^(3/4)
    # v_los = np.sqrt(constants.G_pc * m_R) * np.sin(inc) * y_disk / ((x_disk / np.cos(inc)) ** 2 + y_disk ** 2) ** (
    #             3 / 4)
    v_los = np.sqrt(constants.G_pc * m_R) * np.sin(inc) * x_disk / ((y_disk / np.cos(inc)) ** 2 + x_disk ** 2) ** (
                  3 / 4)
    # line-of-sight velocity v_los at each point in the disk
    print('los')

    # ALTERNATIVE CALCULATION FOR v_los
    alpha = abs(np.arctan(y_disk / (np.cos(inc) * x_disk)))  # alpha meas. from +x (minor axis) toward +y (major axis)
    sign = y_disk / abs(y_disk)  # if np.pi/2 < alpha < 3*np.pi/2, alpha < 0.
    v_los2 = sign * abs(vel * np.sin(alpha) * np.sin(inc))  # print(v_los - v_los2)  # 0 YAY!

    # SET LINE-OF-SIGHT VELOCITY AT THE BLACK HOLE CENTER TO BE 0, SUCH THAT IT DOES NOT BLOW UP
    # if any point as at x_disk, y_disk = (0., 0.), set velocity there = 0.
    # Only relevant if we have pixel located exactly at the center
    center = (R == 0.)
    v_los[center] = 0.
    v_los2[center] = 0.
    # print(center, v_los[center])

    # CALCULATE OBSERVED VELOCITY
    v_obs = v_sys - v_los  # observed velocity v_obs at each point in the disk
    print('v_obs')

    '''  #
    # PLOT USING MICHELE CAPPELLARI'S display_pixels.py CODE
    x_obs2 = np.zeros(shape=R.shape)  # this looks right!
    for i in range(len(x_obs)):
        x_obs2[:, i] = x_obs[i]
    y_obs2 = np.zeros(shape=R.shape)  # this looks right!
    for i in range(len(y_obs)):
        y_obs2[i, :] = y_obs[i]

    dp.display_pixels(x_obs2, y_obs2, subpix_deconvolved, pixelsize=x_obs[1] - x_obs[0])
    plt.show()
    print(oops)

    dp.display_pixels(x_obs2, y_obs2, v_obs, pixelsize=x_obs[1] - x_obs[0])
    plt.show()
    print(oops)

    plt.imshow(v_obs, origin='lower')
    plt.show()
    # '''  #

    # SELECT ALL VELOCITIES WITHIN ELLIPTICAL REGION CENTERED ON BH
    '''  #
    # THIS LOOKS RIGHT!!!! (ellipse TBC)
    indices = R < 50.
    figure = plt.figure(figsize=(6,6))
    ax = plt.gca()
    # plt.contourf(x_obs, y_obs, v_obs, 600, vmin=np.amin(v_obs), vmax=np.amax(v_obs), cmap='RdBu_r')
    plt.imshow(v_obs, extent=[-350, 350,-350, 350], cmap='RdBu_r', origin='lower')
    # plt.plot(x_bhctr + ell_rot[0, :], y_bhctr + ell_rot[1, :], 'k')
    plt.xlabel(r'x [pc]', fontsize=30)
    plt.ylabel(r'y [pc]', fontsize=30)
    plt.xlim(-350, 350)
    plt.ylim(-350, 350)
    ax.set_xticks([-300., -200., -100., 0., 100., 200., 300.,])
    ax.set_yticks([-300., -200., -100., 0., 100., 200., 300.,])
    ax.set_xticklabels([-300., -200., -100., 0., 100., 200., 300.])
    ax.set_yticklabels([-300., -200., -100., 0., 100., 200., 300.])
    plt.colorbar()
    # THIS PLOT INDICATES x_obs = -y_data, y_obs = x_data, so this shows x_dat vs -y_dat
    plt.show()
    # print(oops)

    # THIS (probably) LOOKS RIGHT!!!
    v_ctr = v_obs
    x_reg = x_obs
    y_reg = y_obs
    v_ctr[R > 50] = v_sys  # ignore all velocities outside R=50pc ellipse
    plt.contourf(x_reg, y_reg, v_ctr, 600, vmin=np.amin(v_ctr), vmax=np.amax(v_ctr), cmap='RdBu_r')
    plt.colorbar()
    plt.show()
    print(oops)
    # '''  #

    '''  #
    # RECREATE BARTH+2016 FIG2, CO(2-1) LINE PROFILE INTEGRATED OVER ELLIPSE
    # FIRST, MAKE MODEL VERSION OF FIG2
    # hdu = fits.open(data_cube)
    # data_vs3 = hdu[0].data[0]  # header = hdu[0].header
    data_vs3 = data_rebinned

    obs_vels = []
    mask2 = []
    major = dist * 10 ** 6 * np.tan(4.3 / 0.04 / constants.arcsec_per_rad)  # was 0.01, rebinned to 0.04
    minor = dist * 10 ** 6 * np.tan(0.7 / 0.04 / constants.arcsec_per_rad)  # was 0.01, rebinned to 0.04
    for x in range(len(x_obs)):
        print(x)
        for y in range(len(y_obs)):
            test_pt = ((np.cos(theta_rot)*(x - x_bhctr) + np.sin(theta_rot)*(y - y_bhctr))/major)**2\
                + ((np.sin(theta_rot)*(x - x_bhctr) - np.cos(theta_rot)*(y-y_bhctr))/minor)**2
            if test_pt <= 1:  # IF POINT IS WITHIN ELLIPSE, INCLUDE THIS MODEL VELOCITY
                obs_vels.append(v_obs[x, y])
            mask2.append(test_pt)
    print(obs_vels)
    obs_vels = np.asarray(obs_vels)
    mask2 = np.asarray(mask2)
    print(obs_vels.shape)
    import numpy.ma as ma
    data_vs2 = []
    for k in range(len(data_vs3)):
        data_vs2.append(ma.masked_where(mask2 > 1., data_vs3[k]))
    # data_vs2 = np.asarray(data_vs2)
    # data_vs2 = ma.masked_where(mask > 1., data_vs[30])
    print(np.asarray(data_vs2).shape)
    data_vs2 = np.asarray(data_vs2)
    o_vels = []
    for i in range(len(data_vs2)):  # for each velocity slice
        print(i)
        # add = np.sum(data_vs[i, inds])  # sum up the data corresponding to indices that fall within ellipse
        # add = np.sum(data_vs2[i])  # WAS THIS, TRYING TRAPZ INSTEAD:
        add = np.trapz(data_vs2[i])
        add = np.trapz(add)
        # data_vs[i, inds].shape = 640, 640
        o_vels.append(add)
    print(len(o_vels))
    # plt.hist(v_obs[R <= 300], len(z_ax), edgecolor='k', facecolor=None)  # points within radius rather than ellipse
    plt.bar(z_ax, o_vels, width=(z_ax[1] - z_ax[0]),  edgecolor='k', facecolor='b', alpha=0.5)  # data_vs[:, R < 50]
    # plt.hist(vels_f, len(z_ax), edgecolor='k')  # data_vs[:, R < 50]
    plt.axvline(x=v_sys, color='k')
    plt.xlabel(r'Observed velocity [km/s]')
    plt.ylabel(r'N [model]')
    plt.show()
    # print(oops)

    # RECREATE BARTH+2016 FIG2, CO(2-1) LINE PROFILE INTEGRATED OVER ELLIPSE
    # NOW, MAKE DATA VERSION OF FIG 2
    # hdu = fits.open(data_cube)
    # data_vs = hdu[0].data[0]  # header = hdu[0].header
    data_vs = data_rebinned  # note: the x,y shape here has been corrected already
    print(data_vs.shape)  # (75, 640, 640)
    # for xo, yo in [[6., 0.], [8., 0.], [4., 0.], [6., -2.], [8., -2.], [4., -2.], [6., 2.], [8., 2.], [4., 2.]]:
    # [3., 20.], [5., 20.], [7, 20.], [3., 17.], [5., 17.], [7., 17.], [3., 15.], [5., 15.], [7., 15.]]:
    for xo, yo in [[0., 0.,]]:
        vels = []
        acceptable_pixels = []
        inds = np.zeros(shape=(len(data_vs[0][0]), len(data_vs[0])))  # array of indices, corresponding to x,y data
        # TEST each x,y point: is it within ellipse?
        tests = []
        mask = np.zeros(shape=(len(data_vs[0][0]), len(data_vs[0])))  # array of indices, corresponding to x,y data
        for i in range(len(data_vs[0][0])):  # 640 (x axis)
            for j in range(len(data_vs[0])):  # 640 (y axis)
                res = 0.04  # arcsec/pix  # 0.01 now, but need to bin to 4x4 pixels
                maj = 4.3 / res  # ellipse major axis
                mi = 0.7 / res  # ellipse minor axis
                # NOTE: THE TEST_PT EQUATION BELOW IS FOR theta_rot MEASURED FROM +y, SO ADD +90deg TO theta
                # https://stackoverflow.com/questions/7946187/point-and-ellipse-rotated-position-test-algorithm
                theta_rot = theta + np.pi/2.  # + np.pi/4.
                test_pt = ((np.cos(theta_rot) * (i - (len(data_vs[0][0])/2. + xo)) + np.sin(theta_rot) *
                            (j - (len(data_vs[0])/2. + yo))) / maj) ** 2 \
                    + ((np.sin(theta_rot) * (i - (len(data_vs[0][0])/2. + xo)) - np.cos(theta_rot) *
                        (j - (len(data_vs[0])/2. + yo))) / mi) ** 2
                tests.append(test_pt)
                mask[i, j] = test_pt
                if test_pt <= 1:  # if point within ellipse
                    inds[i, j] = True  # int(1)  # set index = True
                    acceptable_pixels.append([i, j])  # list of acceptable pixels (not using anymore)
                else:  # if point NOT within ellipse
                    inds[i, j] = False  # int(0)  # set index = False
        # inds = inds.astype(int)  # make sure each entry is an int, bc apparently setting each int(entry) isn't good enough
        # print(inds)

        import numpy.ma as ma
        data_vs2 = []
        for k in range(len(data_vs)):
            data_vs2.append(ma.masked_where(mask > 1., data_vs[k]))

        # data_vs2 = np.asarray(data_vs2)
        # data_vs2 = ma.masked_where(mask > 1., data_vs[30])
        print(np.asarray(data_vs2).shape)
        # print(tests)
        c = 0
        for t in tests:
            if t < 1:
                c += 1
        print(inds)
        # print(data_vs[0][inds].shape)
        print(inds.shape)

        # for pixset in acceptable_pixels:
        #     print(pixset[0], pixset[1])
        #     # pixset = [293, 271]  # 293, 271 is pretty good
        #     profile = []
        #     for k in range(len(data_vs)):
        #         profile.append(data_vs[k][pixset[0]][pixset[1]])
        #     plt.plot(z_ax, profile)
        #     plt.axvline(x=v_sys, color='k')
        #     plt.show()

        # print(data_vs[0, inds])centers
        # print(data_vs[0, inds].shape)
        # print(inds[0, 0])  # inds[320, 320] = 1; inds[0, 0] = 0
        for i in range(len(data_vs2)):  # for each velocity slice
            print(i)
            # add = np.sum(data_vs[i, inds])  # sum up the data corresponding to indices that fall within ellipse
            # add = np.sum(data_vs2[i])  # WAS THIS, TRYING TRAPZ INSTEAD:
            add = np.trapz(data_vs2[i])
            add = np.trapz(add)
            # data_vs[i, inds].shape = 640, 640
            vels.append(add)
        print(vels)
        print(len(vels))
        plt.bar(z_ax, vels, width=(z_ax[1] - z_ax[0]), edgecolor='k', facecolor='r', alpha=0.5)  # data_vs[:, R < 50]
        plt.bar(z_ax, o_vels, width=(z_ax[1] - z_ax[0]), edgecolor='k', facecolor='b', alpha=0.5)  # data_vs[:, R < 50]
        # plt.hist(vels_f, len(z_ax), edgecolor='k')  # data_vs[:, R < 50]
        plt.title(str(xo) + ', ' + str(yo))
        plt.axvline(x=v_sys, color='k')
        plt.xlabel(r'Observed velocity [km/s]')
        plt.ylabel(r'N [model, blue; data, red]')
        plt.show()
    print(oops)
    # '''  #

    # CALCULATE VELOCITY PROFILES
    sigma = get_sig(r=R, sig0=sig_params[0], r0=sig_params[1], mu=sig_params[2], sig1=sig_params[3])[sig_type]
    print(sigma)
    obs3d = []  # data cube, v_obs but with a wavelength axis, too!
    weight = subpix_deconvolved
    # BUCKET TRYING ELLIPSE HERE
    major_ax = 50  # pc (see paragraph 3 of S4.2 of Barth+2016 letter)
    minor_ax = 50*np.cos(inc)  # pc (see paragraph 3 of S4.2 of Barth+2016 letter)
    theta_rot = theta + np.pi/4
    for x in range(len(x_obs)):
        print(x)
        for y in range(len(y_obs)):
            test_pt = ((np.cos(theta_rot)*x_obs[x] + np.sin(theta_rot)*y_obs[y])/major_ax)**2\
                + ((np.sin(theta_rot)*x_obs[x] - np.cos(theta_rot)*y_obs[y])/minor_ax)**2
            if test_pt >= 1:  # IF POINT IS OUTSIDE ELLIPSE, IGNORE THIS REGION
                weight[x, y] = 0.
            # NOTE: MAYBE BETTER, BUT REALLY HARD TO SAY
    # END BUCKET
    # weight[R>100] = 0.
    # weight[weight < 1.5e-3] = 0.  # trying something! Right idea but too fluctuate-y still
    plt.imshow(weight, origin='lower')
    plt.show()
    # weight /= np.sqrt(2 * np.pi * sigma)
    # NOTE: BUCKET: weight = weight / (sqrt(2*pi)*sigma) ??
    # print(weight, np.amax(weight), np.amin(weight))
    tz1 = time.time()
    for z in range(len(z_ax)):
        print(z)
        # print(z_ax[z] - v_obs[int(len(v_obs)-1),int(len(v_obs)-1)])  #, z_ax[z], v_obs)
        obs3d.append(weight * np.exp(-(z_ax[z] - v_obs) ** 2 / (2 * sigma ** 2)))
        # obs3d.append(np.exp(-(z_ax[z] - v_obs) ** 2 / (2 * sigma ** 2)))
    print('obs3d')
    obs3d = np.asarray(obs3d)  # has shape 61, 600, 600, which is good
    print('obs3d took {0} s'.format(time.time() - tz1))  # ~3.7 s; 14s for s=2 1280x1280
    print(obs3d.shape)
    for ind in [20, 27, 30, 36, 37, 40]:  # BUCKET SUPER GROSS BAD OKAY DEFINITELY NEED TO RE-DO WEIGHT MAP
        plt.imshow(obs3d[ind], origin='lower')
        plt.show()
    print(oops)

    '''  #
    # OBS3D Profiles
    # [315, 337], [337, 325], [340, 340], [350, 325], [350, 350],
    #                            [325, 350], [320, 360], [315, 315], [337, 315], [270, 411],
    #                            [339, 329]])  # [411, 370]->[270,411]; [329, 301], [346, 405]
    plt.plot(z_ax, obs3d[:,315*s,337*s], 'k--')
    plt.plot(z_ax, obs3d[:,350*s,325*s], 'r-')
    plt.plot(z_ax, obs3d[:,315*s,350*s], 'm-')
    # plt.plot(z_ax, obs3d[:350*s,400*s], 'b-')
    plt.plot(z_ax, obs3d[:,316*s,336*s], 'g-')  # 163,166
    plt.plot(z_ax, obs3d[:,310*s,380*s], 'b-')  # 163,166
    plt.plot(z_ax, obs3d[:,10*s,450*s], 'k-')  # 10, 400
    plt.axvline(x=v_sys, color='k')
    # red solid, blue solid, dotted for along minor axis
    plt.show()
    print(oops)
    # '''  #

    '''  #
    # view 2d slice of obs3d (i.e. view one velocity-slice in 2d-space)
    fig = plt.figure()
    thing2 = obs3d[31, :, :]
    print(np.amax(thing2))
    plt.contourf(-y_obs, x_obs, thing2, 600, vmin=np.amin(thing2), vmax=np.amax(thing2), cmap='viridis')
    plt.colorbar()
    plt.show()
    print(oops)
    # '''  #

    '''  #
    # view several velocity profiles at once
    print(obs3d.shape)
    mid = 323
    diff = 40
    for xs in range(mid, mid+diff):
        print(xs)
        for ys in range(mid, mid+diff):
            plt.plot(z_ax, obs3d[:,xs,ys], 'b--', alpha=0.5)
    for xs in range(mid-diff, mid):
        print(xs)
        for ys in range(mid-diff, mid):
            plt.plot(z_ax, obs3d[:,xs,ys], 'r--', alpha=0.5)
    plt.axvline(x=v_sys, color='k')
    plt.show()
    # '''  #

    ''' #
    # PRINT OBSERVED VELOCITY v_obs
    fig = plt.figure(figsize=(12, 10))
    #plt.contourf(-y_obs, x_obs, v_obs, 600, vmin=np.amin(v_obs), vmax=np.amax(v_obs), cmap='RdBu_r')
    #plt.ylabel(r'x [pc]', fontsize=30)
    #plt.xlabel(r'y [pc]', fontsize=30)
    #plt.xlim(max(y_obs), min(y_obs))
    #plt.ylim(min(x_obs), max(x_obs))
    #plt.plot(-y_bhctr, x_bhctr, 'k*', markersize=20)
    #plt.plot(-y_bhctr+ell_rot[0,:], x_bhctr+ell_rot[1,:], 'k')
    print(x_bhctr, y_bhctr)
    plt.contourf(x_obs, y_obs, v_obs, 600, vmin=np.amin(v_obs), vmax=np.amax(v_obs), cmap='RdBu_r')
    plt.xlabel(r'x [pc]', fontsize=30)
    plt.ylabel(r'y [pc]', fontsize=30)
    plt.ylim(min(y_obs), max(y_obs))
    plt.xlim(min(x_obs), max(x_obs))
    plt.plot(x_bhctr, y_bhctr, 'k*', markersize=20)
    plt.plot(ell_rot[0,:]+x_bhctr, ell_rot[1,:]+y_bhctr, 'k')
    cbar = plt.colorbar()
    cbar.set_label(r'km/s', fontsize=30, rotation=0, labelpad=25)  # pc,  # km/s
    plt.show()
    print(oops)
    # '''  #

    '''  #
    # write out obs3d to fits file
    hdu = fits.PrimaryHDU(obs3d)
    hdul = fits.HDUList([hdu])
    hdul.writeto('1332_obs3d_vlos2_correctweight_s2.fits')
    print(oops)
    # '''  #

    '''  #
    t_fig = time.time()
    fig = plt.figure(figsize=(6, 5))
    print(v_obs.shape)
    # plt.contourf(x_obs, y_obs, v_obs, 600, vmin=np.amin(v_obs), vmax=np.amax(v_obs), cmap='RdBu_r')
    thing = weight  # v_obs  # obs3d[24,:,:]  # v_obs, v_los, obs3d[30,:,:], R
    print(thing[150,150])  # YES THIS NOW CORRECT
    plt.contourf(-y_obs, x_obs, thing, 600, vmin=np.amin(thing), vmax=np.amax(thing), cmap='RdBu_r')
    # viridis, RdBu_r, inferno
    print(theta)

    # plt.plot([0., 1.*np.cos(np.deg2rad(theta))], [0., 1.*np.sin(np.deg2rad(theta))], color='k', lw=3)
    cbar = plt.colorbar()

    # hdu = fits.PrimaryHDU(v_obs)
    # hdul = fits.HDUList([hdu])
    # print(len(hdu.data[0]), len(hdu.data))  # 1800, 1800
    # hdul.writeto('ngc1332_vobs.fits')

    # plt.xlabel(r'x [pc]', fontsize=30)
    # plt.ylabel(r'y [pc]', fontsize=30)
    # plt.xlim(min(x_obs), max(x_obs))
    # plt.ylim(min(y_obs), max(y_obs))

    plt.xlabel(r'y [pc]', fontsize=30)
    plt.ylabel(r'x [pc]', fontsize=30)
    plt.xlim(max(y_obs), min(y_obs))
    plt.ylim(min(x_obs), max(x_obs))
    # cbar.set_label(r'km/s', fontsize=30, rotation=0, labelpad=20)  # pc,  # km/s
    print('hey')
    plt.show()
    print('Image plotted in {0} s'.format(time.time() - t_fig))  # ~255 seconds for s=6
    print(oops)
    # '''  #

    # RESAMPLE
    # RE-SAMPLE BACK TO CORRECT PIXEL SCALE (take average of sxs sub-pixels for real alma pixel) --> intrinsic data cube
    t_z = time.time()
    intrinsic_cube = []  # np.zeros(shape=(len(z_ax), len(fluxes), len(fluxes[0])))
    for z2 in range(len(z_ax)):
        print(z2)
        # break each (s*len(realx), s*len(realy))-sized velocity slice of obs3d into an array comprised of sxs subarrays
        # i.e. break the 300s x 300s array into blocks of sxs arrays, so that I can take the means of each block
        subarrays = blockshaped(obs3d[z2, :, :], s, s)

        # Take the mean along the first (s-length) axis of each subarray in subarrays, then take the mean along the
        # other (s-length) axis, such that what remains is a 1d array of the means of the sxs subarrays. Then reshape
        # into the correct real x_pixel by real y_pixel lengths
        reshaped = np.mean(np.mean(subarrays, axis=-1), axis=-1).reshape((len(fluxes), len(fluxes[0])))
        intrinsic_cube.append(reshaped)
    print("intrinsic cube done in {0} s".format(time.time() - t_z))  # 0.34 s YAY! (1.3 s for s=6)
    print("start to intrinsic done in {0} s".format(time.time() - t0))  # 6.2s for s=6
    intrinsic_cube = np.asarray(intrinsic_cube)
    # print(oops)
    # print(intrinsic_cube.shape)  # 61, 300, 300

    ''' #
    plt.plot(z_ax, intrinsic_cube[:,158,168], 'k-')
    plt.plot(z_ax, intrinsic_cube[:,149,153], 'r-')
    plt.plot(z_ax, intrinsic_cube[:,165,178], 'b-')
    plt.plot(z_ax, intrinsic_cube[:152,172], 'm--')
    plt.plot(z_ax, intrinsic_cube[:,176,157], 'g--')  # 163,166
    plt.axvline(x=v_sys, color='k')
    print(v_sys)
    # red solid, blue solid, dotted for along minor axis
    plt.show()
    # '''

    '''  #
    hdu = fits.PrimaryHDU(intrinsic_cube)
    hdul = fits.HDUList([hdu])
    hdul.writeto('1332_intrinsic_newdata01_n15_s2.fits')
    print('written!')
    print(oops)

    if out_name is not None:
        hdu = fits.PrimaryHDU(intrinsic_cube2)
        hdul = fits.HDUList([hdu])
        hdul.writeto('name.fits')
    # '''  #

    # CONVERT INTRINSIC TO OBSERVED
    # take velocity slice from intrinsic data cube, convolve with alma beam --> observed data cube
    '''
    beam_psf = make_beam(grid_size=len(intrinsic_cube[0, :, :]), x_std=0.319, y_std=0.233, rot=np.deg2rad(90-78.4))
    # print(beam_psf.shape)  # 300, 300
    convolved_cube = []
    start_time = time.time()
    for z3 in range(len(z_ax)):
        t1 = time.time()
        print(z3)
        # CONVOLVE intrinsic_cube[z3, :, :] with alma beam
        # alma_beam needs to have same dimensions as intrinsic_cube[:, :, z3]
        convolved_step = signal.fftconvolve(intrinsic_cube[z3, :, :], beam_psf, mode='same')
        # print(convolved_step.shape)
        convolved_cube.append(convolved_step)
        # convolved_cube.append(signal.convolve2d(intrinsic_cube[:, :, z3], beam_psf, mode='same'))
        # print(convolved_cube[z3].shape)  # 300, 300
        print("Convolution loop " + str(z3) + " took {0} seconds".format(time.time() - t1))
        # mode=same keeps output size same
    convolved_cube = np.asarray(convolved_cube)
    print('convolved! Total convolution loop took {0} seconds'.format(time.time() - start_time))
    '''

    convolved_cube = np.zeros(shape=intrinsic_cube.shape)  # 61, 300, 300
    # beam = make_beam(grid_size=35, x_std=0.319, y_std=0.233, rot=np.deg2rad(90-78.4))  # 29
    # beam = make_beam(grid_size=35, x_std=0.044, y_std=0.039, rot=np.deg2rad(90-64.))
    # beam_psf = make_beam(grid_size=len(intrinsic_cube[0,:,:])-1, x_std=0.319, y_std=0.233, rot=np.deg2rad(90-78.4))
    ts = time.time()
    for z in range(len(z_ax)):
        print(z)
        tl = time.time()
        # I think the problem is I need to set origin?
        convolved_cube[z, :, :] = convolution.convolve(intrinsic_cube[z, :, :], beam)
        # convolved_cube[z,:,:] = filters.convolve(intrinsic_cube[z,:,:], beam, mode='same')
        print("Convolution loop " + str(z) + " took {0} seconds".format(time.time() - tl))
    print('convolved! Total convolution loop took {0} seconds'.format(time.time() - ts))

    '''  #
    # COMPARE LINE PROFILES!!!
    # inds_to_try = np.asarray([[645, 628], [651, 631], [655, 633]])  # 670, 640; 625, 640
    # inds_to_try = np.asarray([[320, 312], [326, 316], [332, 320]])  # 670, 640; 625, 640
    # inds_to_try = np.asarray([[320, 313], [326, 316], [332, 319]])  # 670, 640; 625, 640
    inds_to_try = np.asarray([[340, 310], [337, 315], [334, 320], [10, 450]])
    # from above, data center should be around data[:, 320+5, 320+17]
    colors = ['g', 'k', 'm', 'b']

    # fluxes, freq_ax
    fig = plt.figure(figsize=(12, 10))
    plt.contourf(x_obs, y_obs, v_obs, 600, vmin=np.amin(v_obs), vmax=np.amax(v_obs), cmap='RdBu_r')
    cbar = plt.colorbar()
    plt.xlabel(r'x [pc]', fontsize=30)
    plt.ylabel(r'y [pc]', fontsize=30)
    plt.xlim(min(x_obs), max(x_obs))
    plt.ylim(min(y_obs), max(y_obs))
    for i in range(len(inds_to_try)):
        plt.plot(x_obs[inds_to_try[i][0] * s], y_obs[inds_to_try[i][1] * s], colors[i] + '*', markersize=20)
    plt.plot(x_bhctr + ell_rot[0, :], y_bhctr + ell_rot[1, :], 'k')
    cbar.set_label(r'km/s', fontsize=30, rotation=0, labelpad=20)  # pc,  # km/s
    plt.show()

    print(z_ax)
    print(freq_ax)
    hdu = fits.open(data_cube)
    data = hdu[0].data[0]  # header = hdu[0].header
    for i in range(len(inds_to_try)):
        print(data[0, :, :].shape)
        # plt.plot(z_ax, data[:, inds_to_try[i][0], inds_to_try[i][1]], 'k--')  # 670, 640; 722, 614; 625, 640
        # 293, 271
        # plt.plot(z_ax, data[:, 640-inds_to_try[i][1], inds_to_try[i][0]], 'k--')  # 670, 640; 722, 614; 625, 640
        plt.plot(z_ax, data[:, 293, 271], 'k--')  # 670, 640; 722, 614; 625, 640
        # plt.plot(z_ax, convolved_cube[:, inds_to_try[i][0], inds_to_try[i][1]],
        plt.plot(z_ax, convolved_cube[:, 293, 271],
                 colors[i] + '-')  # 670, 640; 722, 614; 625, 640
        plt.axvline(x=v_sys, color='k')
        plt.show()

    inds_to_try2 = np.asarray([[315, 337], [337, 325], [340, 340], [350, 325], [350, 350],
                               [325, 350], [320, 360], [315, 315], [337, 315], [270, 411],
                               [339, 329], [10, 450]])  # [411, 370]->[270,411]; [329, 301], [346, 405]
    # systemic with red and blue bumps, very red (no data/slight systemic), very slight red, quite red, very slight red,
    # systemic, blue/systemic, red/systemic, slight red, slight blue!, quite red again wtf
    # NOT DUE TO CONVOLUTION! NOT DUE TO INTRINSIC_CUBE STEP EITHER!
    colors2 = ['r', 'm', 'k', 'g', 'b', 'r', 'm', 'k', 'g', 'b', 'r', 'm']
    for i in range(len(inds_to_try2)):
        # plt.plot(z_ax, data[:, inds_to_try[i][0], inds_to_try[i][1]], 'k--')  # 670, 640; 722, 614; 625, 640
        plt.plot(z_ax, data[:, 640 - inds_to_try2[i][1], inds_to_try2[i][0]], 'k--')  # 670, 640; 722, 614; 625, 640
        plt.plot(z_ax, convolved_cube[:, inds_to_try2[i][0], inds_to_try2[i][1]],
                 colors2[i] + '-')  # 670, 640; 722, 614; 625, 640
        plt.axvline(x=v_sys, color='k')
        plt.show()
        # print(oops)
    # '''  #

    # ROTATE convolved_cube TO MATCH ORIGINAL DATA CUBE
    # convolved_cube = np.swapaxes(convolved_cube, 1, 2)  # still not right...
    # convolved_cube = np.rot90(convolved_cube, axes=(1, 2))
    # print(np.swapaxes(convolved_cube, 1, 2).shape)

    # '''  #
    # TRY THINGS AGAIN
    # hdu = fits.open(data_cube)
    # data_vs = hdu[0].data[0]  # header = hdu[0].header
    data_vs = data_rebinned
    print(data_vs.shape)  # (75, 640, 640)
    # for xo, yo in [[6., 0.], [8., 0.], [4., 0.], [6., -2.], [8., -2.], [4., -2.], [6., 2.], [8., 2.], [4., 2.]]:
    # , [3., 20.], [5., 20.], [7, 20.], [3., 17.], [5., 17.], [7., 17.], [3., 15.], [5., 15.], [7., 15.]
    for xo, yo in [[0., 0.]]:
        vels = []
        acceptable_pixels = []
        inds = np.zeros(shape=(len(data_vs[0]), len(data_vs[0][0])))  # array of indices, corresponding to x,y data
        # TEST each x,y point: is it within ellipse?
        tests = []
        mask1 = np.zeros(shape=(len(data_vs[0]), len(data_vs[0][0])))  # array of indices, corresponding to x,y data
        for i in range(len(data_vs[0][0])):  # 640 (x axis)
            for j in range(len(data_vs[0])):  # 640 (y axis)
                res = 0.04  # arcsec/pix  # inherently 0.01; using 0.04 because binned the data
                maj = 4.3 / res  # ellipse major axis
                mi = 0.7 / res  # ellipse minor axis
                theta_rot = theta  # + np.pi/4.
                test_pt = ((np.cos(theta_rot) * (i - (len(data_vs[0][0])/2. + xo)) + np.sin(theta_rot) *
                            (j - (len(data_vs[0])/2. + yo))) / maj) ** 2 \
                    + ((np.sin(theta_rot) * (i - (len(data_vs[0][0])/2. + xo)) - np.cos(theta_rot) *
                        (j - (len(data_vs[0])/2. + yo))) / mi) ** 2
                tests.append(test_pt)
                mask1[i, j] = test_pt
                if test_pt <= 1:  # if point within ellipse
                    inds[i, j] = True  # int(1)  # set index = True
                    acceptable_pixels.append([i, j])  # list of acceptable pixels (not using anymore)
                else:  # if point NOT within ellipse
                    inds[i, j] = False  # int(0)  # set index = False
        # inds = inds.astype(int)
        # print(inds)

        import numpy.ma as ma
        data_vs2 = []
        conv2 = []
        for k in range(len(data_vs)):
            data_vs2.append(ma.masked_where(mask1 > 1., data_vs[k]))
            conv2.append(ma.masked_where(mask1 > 1., convolved_cube[k]))
        data_vs2 = np.asarray(data_vs2)
        conv2 = np.asarray(conv2)
        # data_vs2 = ma.masked_where(mask > 1., data_vs[30])
        print(np.asarray(data_vs2).shape)
        # print(tests)
        c = 0
        for t in tests:
            if t < 1:
                c += 1
        print(inds)
        # print(data_vs[0][inds].shape)
        print(inds.shape)

        '''
        inds_to_try2 = np.asarray([[315, 337], [337, 325], [340, 340], [350, 325], [350, 350],
                                   [325, 350], [320, 360], [315, 315], [337, 315], [270, 411],
                                   [339, 329], [10, 450]])  # [411, 370]->[270,411]; [329, 301], [346, 405]
        '''
        inds_to_try2 = np.asarray([[83, 83], [85, 84], [77, 76], [78, 74], [90, 88], [90, 65]])
        # systemic with red and blue bumps, very red (no data/slight systemic), very slight red, quite red, very slight red,
        # systemic, blue/systemic, red/systemic, slight red, slight blue!, quite red again wtf
        # NOT DUE TO CONVOLUTION! NOT DUE TO INTRINSIC_CUBE STEP EITHER!
        colors2 = ['r', 'm', 'k', 'g', 'b', 'r', 'm', 'k', 'g', 'b', 'r', 'm']
        for i in range(len(inds_to_try2)):
            print(i)
            print(inds_to_try2[i][0], inds_to_try2[i][1])
            # plt.plot(z_ax, data[:, inds_to_try[i][0], inds_to_try[i][1]], 'k--')  # 670, 640; 722, 614; 625, 640
            plt.plot(z_ax, data_vs2[:, inds_to_try2[i][0], inds_to_try2[i][1]], 'k--')  # 670, 640; 722, 614; 625, 640
            plt.plot(z_ax, conv2[:, inds_to_try2[i][0], inds_to_try2[i][1]], colors2[i] + '-')
            plt.axvline(x=v_sys, color='k')
            plt.title('no x,y offset')
            plt.show()

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title('data ' + str(xo) + ', ' + str(yo))
        plt.imshow(data_vs2[20])  # (data_vs[45])  # (data_vs[30])
        plt.gca().invert_yaxis()
        ax.set_aspect('equal')
        plt.colorbar(orientation='vertical')
        plt.show()

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title('conv ' + str(xo) + ', ' + str(yo))
        plt.imshow(conv2[20])  # (data_vs[45])  # (data_vs[30])
        plt.gca().invert_yaxis()
        ax.set_aspect('equal')
        plt.colorbar(orientation='vertical')
        plt.show()
    # END TRY THINGS AGAIN
    # '''  #

    # WRITE OUT RESULTS TO FITS FILE
    if out_name is not None:
        hdu = fits.PrimaryHDU(convolved_cube)
        hdul = fits.HDUList([hdu])
        hdul.writeto(out_name)
        print('written!')
        # NOTE: right now all 0s

    # PRINT 2D DISK ROTATION FIG
    if incl_fig:
        fig = plt.figure(figsize=(12, 10))
        plt.contourf(-y_obs, x_obs, v_obs, 600, vmin=np.amin(v_obs), vmax=np.amax(v_obs), cmap='RdBu_r')
        cbar = plt.colorbar()
        plt.xlabel(r'y [pc]', fontsize=30)
        plt.ylabel(r'x [pc]', fontsize=30)
        plt.xlim(max(y_obs), min(y_obs))
        plt.ylim(min(x_obs), max(x_obs))
        cbar.set_label(r'km/s', fontsize=30, rotation=0, labelpad=20)  # pc,  # km/s
        plt.show()

        fig2 = plt.figure(figsize=(12, 10))
        for x in range(len(x_obs)):
            xtemp = [x_obs[x]] * len(y_obs)
            plt.scatter(xtemp, y_obs, c=v_obs[x, :], vmin=np.amin(v_obs), vmax=np.amax(v_obs), s=25,
                        cmap='RdBu_r')  # viridis, RdBu_r, inferno
        plt.colorbar()
        plt.xlabel(r'x [Mpc]', fontsize=30)
        plt.ylabel(r'y [Mpc]', fontsize=30)
        plt.xlim(min(x_obs), max(x_obs))
        plt.ylim(min(y_obs), max(y_obs))
        plt.show()
    # '''  #

    return convolved_cube


def spec(cube, x_pix, y_pix, print_it=False):
    """
    View the full spectrum across all velocity channels at the (x,y) pixel location of your choice in the cube

    :param cube: output convolved cube from model_grid() function, with shape=(N_xpixels, N_ypixels, N_vel_channels)
    :param x_pix: x_pixel location at which to view spectrum
    :param y_pix: y_pixel location at which to view spectrum
    :param print_it: plot and show the given spectrum

    return spectrum: full spectrum across all velocity channels in cube, at spatial location (x_pixel, y_pixel)
    """
    spectrum = np.asarray([cube[:, x_pix, y_pix]])

    if print_it:
        fig = plt.figure(figsize=(12, 10))
        channels = np.arange(stop=len(spectrum), step=1.) + 1.
        plt.scatter(channels, spectrum, 'bo', s=25)
        plt.xlabel(r'Velocity channel', fontsize=30)
        plt.ylabel(r'Flux per channel', fontsize=30)
        plt.xlim(channels[0], channels[-1])
        plt.ylim(0., max(spectrum))
        plt.show()

    return spectrum


if __name__ == "__main__":
    # OLD DATA
    ''' #
    cube, lucy, out = 'NGC_1332_CO21.pbcor.fits', 'lucy_out_n15.fits', '1332_apconvolve_n15_size35.fits'
    mbh, resolution, spacing = 6.*10**8, 0.07, 20.1
    # NOTE: NGC 1332 center appears to be x=168, y=158 (out of (300,300), starting with 1) --> expected at 149.5
    # My code would currently put it at 150,150 which is really the 151st pixel because python starts counting at 0
    # ds9 x-axis is my -y axis: y_off = -(168-151) = -18
    # ds9 y-axis is my +x axis: x_off = (158-151) = 8
    x_off, y_off = 8., -18.
    s = 6
    # sigma = 25.  # * (1 + np.exp(-R[x, y])) # sig: const, c1+c2*e^(-r/r0), c1+exp[-(x-mu)^2 / (2*sig^2)]
    # sig1 = 0.  # km/s
    # sig0 = 50.  # 268.  # km/s
    # r0 = 60.7  # pc
    ## sigma = sig0 * np.exp(-R / r0) + sig1
    ## sigma = sig0
    # mu = (not discussed anywhere in paper?)
    # sigma = sig0 * np.exp(-(R - r0)**2 / (2 * mu**2)) + sig1
    sig_type, sig0, r0, mu, sig1 = 'flat', 25., 1., 1., 1.
    # psf = make_beam(grid_size=100, x_std=0.319, y_std=0.233, rot=np.deg2rad(90-78.4))
    beam = make_beam(grid_size=35, x_std=0.319, y_std=0.233, rot=np.deg2rad(90-78.4))  # , fits_name='testbeam.fits'
    # fits_name='psf_out.fits')  # rot: start at +90, subtract 78.4 to get 78.4
    dist = 22.3  # Mpc
    inc = np.deg2rad(83.)
    theta = np.deg2rad(-333.)
    # x_width=21., y_width=21., n_channels=60,
    # NOTE: from Fig3 of Barth+2016, their theta=~117 deg appears to be measured from +x to redshifted side of disk
    #       since I want redshifted side of disk to +y (counterclockwise), and my +y is to the left in their image,
    #       I want -(360 - (117-90)) = -333 deg
    # beam_axes = [0.319, 0.233]  # arcsecs
    # beam_major_axis_PA = 78.4  # degrees
    # '''

    # ''' #
    # NEW DATA
    # DEFINE INPUT PARAMETERS
    # DATA CUBE, LUCY OUTPUT, AND NAME FOR OUTPUT FILE (cube size 1280x1280x75)
    # for lucy: http://stsdas.stsci.edu/cgi-bin/gethelp.cgi?lucy
    '''
    cube = '/Users/jonathancohn/Documents/dyn_mod/2015_N1332_data/product/calibrated_source_coline.pbcor.fits'
    lucy = '/Users/jonathancohn/Documents/dyn_mod/newdata_lucy_n15.fits'
    out = '1332_newdata_apconv_n15_size35_s2_fixes3.fits'
    '''
    cube = '/Users/jonathancohn/Documents/dyn_mod/NGC1332_01_calibrated_source_coline.pbcor.fits'
    # lucy = '/Users/jonathancohn/Documents/dyn_mod/newdata01_binnedandclipped_beam77_lucy_n15.fits'
    lucy = '/Users/jonathancohn/Documents/dyn_mod/newdata01_binnedandclipped_beam77_lucy_n15_xy.fits'
    # newdata01_beam77_lucy_n15.fits'
    out = '1332_newdata01_apconv_n15_gsize35_4x4bin_xy_50pccollapse.fits'

    # BLACK HOLE MASS (M_sol), RESOLUTION (ARCSEC), VELOCITY SPACING (KM/S)
    mbh = 6.64 * 10 ** 8  # 6.86 * 10 ** 8  # , 20.1 (v_spacing no longer necessary)
    resolution = 0.04  # was 0.01, now 0.04 because we're binning the data  # 0.05  # 0.044

    # X, Y OFFSETS (PIXELS)
    # NEW DATA 01: CALL IT: 326, 316 (observed pixel coords) (out of 640, 640) --> x_off = +7, y_off = -4
    x_off, y_off = 0., 0.  # 5., 17.  # +6., -4.  # 631. - 1280./2., 1280/2. - 651  # pixels  0., 0.

    # VELOCITY DISPERSION PARAMETERS
    sig0 = 32.1  # 22.2  # km/s
    r0, mu, sig1 = 1., 1., 1.  # not currently using these!
    s_type = 'flat'

    # DISTANCE (Mpc), INCLINATION (rad), POSITION ANGLE (rad)
    dist = 22.3  # Mpc
    inc = np.deg2rad(85.2)  # 83.
    theta = np.deg2rad(26.7)# + 180.)  # (180. + 116.7)  # -333 (-333.3 from 116.7: -(360 - (116.7-90)) = -333.3
    vsys = 1562.2  # km/s
    # Note: 22.3 * 10**6 * tan(640 * 0.01 / 206265) = 692 pc (3460 pc if use 0.05?)
    # 692 pc / 640 pix =~ 1.1 pc/pix

    # OVERSAMPLING FACTOR
    s = 2

    # ENCLOSED MASS FILE, CONSTANT BY WHICH TO MULTIPLY THE M/L RATIO
    enc_mass = 'ngc1332_enclosed_stellar_mass'
    ml_const = 1.065  # 7.83 / 7.35 # 1.024  # 7.53 / 7.35

    # BEAM PARAMS
    gsize = 35  # size of grid
    x_fwhm = 0.052  # arcsec
    y_fwhm = 0.037  # arcsec
    pos = 64.  # deg

    # MAKE ALMA BEAM  # grid_size anything (must = collapsed datacube size for lucy); x_std=major; y_std=minor; rot=PA
    beam = make_beam(grid_size=gsize, x_std=x_fwhm, y_std=y_fwhm, rot=np.deg2rad(90. - pos))
    # , fits_name='newbeam77.fits')
    # newbeam = make_beam(grid_size=1280, x_std=0.052, y_std=0.037, rot=np.deg2rad(90-64.),
    # fits_name='newdata_beam.fits')
    # '''

    # Make nice plot fonts
    rc('font', **{'family': 'serif', 'serif': ['Times']})
    rc('text', usetex=True)

    # CREATE GRID!
    out_cube = model_grid(resolution=resolution, s=s, x_off=x_off, y_off=y_off, mbh=mbh, inc=inc, dist=dist, vsys=vsys,
                          theta=theta, data_cube=cube, lucy_output=lucy, out_name=out, incl_fig=0, ml_const=ml_const,
                          enclosed_mass=enc_mass, sig_type=s_type, beam=beam, sig_params=[sig0, r0, mu, sig1])

    # vel_slice = spec(out_cube, x_pix=149, y_pix=149, print_it=True)
