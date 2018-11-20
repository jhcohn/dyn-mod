import numpy as np
import astropy.io.fits as fits
# from astropy.table import Table
import matplotlib.pyplot as plt
from matplotlib import rc
from scipy import integrate, signal, interpolate
from scipy.ndimage import filters
# from astropy import modeling
import time
from astropy import convolution

"""
From Barth 2001 (eqn 4; pg 9): line profile in velocity units of a given slit position and CCD row is:
    f = S * P * np.exp( (v - v_p - v_d * M * (x-x_c))**2 / (-2 * (sig_p**2 + sig_t**2 + sig_LSF**2)) )
where:
S = surface brightness at pixel i,j
P = value (at pixel i,j) of a PSF centered at pixel i,j
v_p = projected LOS velocity at i,j
v_d = bin size of the CCD in pixels along dispersion axis in wavelength range of interest (25.2 km/s)
x_c = x-position of slit center
(x-x_c) is in units of pixels
v_d * M * (x-x_c) = shift due to nonzero width of the slit and its projection onto the STIS CCD
    (accounts for the fact that wavelength recorded for a given photon depends on where photon enters
    slit along the x-axis)
M = ratio of plate scales in the dispersion and spatial directions (anamorphic magnification factor)
    (in wavelength range of interest, M=0.93 for G750M grating)
sig_p = bulk velocity dispersion for gas in circumnuclear disk, projected along LOS
sig_t = thermal velocity dispersion in disk
sif_LSF = gaussian width of point-source line-spread function (LSF) for STIS

FOR ME:
NOTE: do convolution step after rebinning to regular pixels, s.t. value of s doesn't affect time cost
However, each subpixel IS modeled as a gaussian
Step-by-step: (1) model each subpixel velocity as a gaussian. (2) line profile at each pixel location is
    weighted by surface brightness map, assuming each subsampled pixel (at a given pixel location) is
    equivalent. (3) Then, rebin the data by averaging each s x s set of subpixel profiles to form a
    single profile. (4) Then, with these single profiles, do the lucy step.

NOTE: STEP 3 above is the one I was missing I think!


NOTE: central velocity should be circular vel, not obs?
NOTE: care about shape for weights as function of radius
NOTE: for weights, always include small f in front!
NOTE: doesn't need to be physical, just match observations

Jonelle tips:
(1) collapse data cube (numerically integrate obs line profiles to get area under curve i.e. flux)
    --> map of fluxes.
(2) run that through lucy (feed in beam, deconvolve it) --> map of deconvolved fluxes
(3) take deconvolved map, take subpixels, assign each subpixel to have same flux (set from deconvolved map)
    --> weights to apply to gaussian
(4) resample back to correct pixel scale (take average of 100 (10x10) subpixels for real alma pixel)
    --> intrinsic data cube
(5) take velocity slice, convolve with alma beam --> observed data cube
ALTERNATIVE: (what we think is done in Barth+2001 paper)
(1) Instead of formal integral, sum values (sum profiles at each location)
# is weight just the number out in front, or is it divided by sqrt(2pi*sigma)

# when get cube, look at slices
# use 1332 for model inputs for now (including beam size etc.)
# also adopt best fit intrinsic velocity dispersion


S = same for all subsampled s pixels, sum these up and that's what I get at real pixel i,j
P = do I define this myself? This is also the psf input into lucy (along with the regular image) right?
S * p together are wrapped into the weights I do above
v_p = what I've been calling v_obs
v_d, x_c, M not relevant for me!
sig_p = what is "bulk" vel dispersion? Is that just the difference btwn most + and most - velocities?
sig_t = how do I get this?
sig_LSF = I assume there's something equivalent I should use for radio beam spread function?
Wrap up all the sigs into one sig_turb? (Barth+2016) (define sigma as constant c, c + exp, or c + gauss)
NOTE: exponential sig = c1 + c2*np.exp(-r / r0) --> c1, c2, and r0 are free params
NOTE: gaussian sig = c1 + c2*np.exp(-(r-r0)**2 / (2*mu**2)) --> c1, c2, r0, and mu are free params 
    f = S * P * np.exp(-(v - v_p)**2 / (2 * sig_turb**2))
"""


# constants
class Constants:
    # a bunch of useful astronomy constants

    def __init__(self):
        self.c = 3. * 10 ** 8  # m/s
        self.pc = 3.086 * 10 ** 16  # m
        self.G = 6.67 * 10 ** -11  # kg^-1 * m^3 * s^-2
        self.M_sol = 2. * 10 ** 30  # kg
        self.H0 = 70  # km/s/Mpc
        self.arcsec_per_rad = 206265  # arcsec per radian
        self.m_per_km = 10 ** 3  # m per km
        self.G_pc = self.G * self.M_sol * (1 / self.pc) / self.m_per_km ** 2  # Msol^-1 * pc * km^2 * s^-2


'''
def lucy_iraf():

    iraf.stsdas(_doprint=0)
    iraf.analysis(_doprint=0)
    iraf.restore(_doprint=0)

    iraf.unlearn(iraf.stsdas)
    iraf.unlearn(iraf.analysis)
    iraf.unlearn(iraf.restore)

    iraf.nfprepare()
'''


def check_beam(beam_name):
    hdu = fits.open(beam_name)
    data = hdu[0].data  # header = hdu[0].header
    print(data.shape)
    print(data)
    # for i in range(len(data[0])):
    #     plt.plot(np.arange(len(data[0])), data[i, :])
    plt.plot(np.arange(len(data)), data[:, int(len(data) / 2.)])
    plt.show()


def make_beam(grid_size=100, amp=1., x0=0., y0=0., x_std=1., y_std=1., rot=0., fits_name=None):
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


def get_sig(r=None, sig0=1., r0=1., mu=1., sig1=0.):
    """
    :param r: array of radius values
    :param sig0: uniform sigma_turb component (velocity units)
    :param r0: scale radius value (same units as r)
    :param mu: same units as r
    :param sig1: additional sigma term (same units as sig0)
    :return: dictionary of the three different sigma shapes
    """

    sigma = {'flat': sig0, 'gauss': sig1 + sig0 * np.exp(-(r - r0) ** 2 / (2 * mu ** 2)),
             'exp': sig1 + sig0 * np.exp(-r / r0)}

    return sigma


def get_fluxes(data_cube, write_name=None):
    """
    Use to integrate line profiles to get fluxes from data cube!

    :param data_cube: input data cube of observations
    :param write_name: name of fits file to which to write collapsed cube (if None, write no file)

    :return: collapsed data cube (i.e. integrated line profiles i.e. area under curve i.e. flux), z_length (len(z_ax))
    """
    hdu = fits.open(data_cube)
    # print(hdu[0].header)
    # CTYPE1: RA
    # CRVAL1 = 5.157217100000Ee+1 [RA of reference pix]
    # CDELT1 = 1.944444444444E-5 [increment at the reference pixel]
    # CRPIX1 = 1.510000000000E+2 [coordinate of reference pixel]
    # CUNIT1 = DEG
    # CTYPE2 = DEC
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
    data = hdu[0].data[0]  # header = hdu[0].header
    # print(data[:,1,1])
    # collapsed_fluxes = np.sum(data, axis=0)  # this sums over all velocity slices at each pixel
    collapsed_fluxes = integrate.simps(data[14:63], axis=0)  # according to my python terminal tests
    print(collapsed_fluxes.shape)

    z_len = len(hdu[0].data[0])  # store the number of velocity slices in the data cube
    freq1 = float(hdu[0].header['CRVAL3'])
    f_step = float(hdu[0].header['CDELT3'])
    freq_axis = np.arange(freq1, freq1 + (z_len * f_step), f_step)
    print(freq1)
    hdu.close()

    if write_name is not None:
        hdu = fits.PrimaryHDU(collapsed_fluxes)
        hdul = fits.HDUList([hdu])
        hdul.writeto(write_name)

    return collapsed_fluxes, freq_axis


def blockshaped(arr, nrow, ncol):
    h, w = arr.shape
    return (arr.reshape(h // nrow, nrow, -1, ncol).swapaxes(1, 2).reshape(-1, nrow, ncol))


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
    # fluxes, freq_ax = get_fluxes(data_cube, write_name='NGC1332_newdata01_clipped1463_collapsed.fits')
    fluxes, freq_ax = get_fluxes(data_cube)
    # , write_name='1332_integratedcollapsed.fits'
    # , write=True, write_name='NGC1332_collapsed.fits')
    # print('n_channels: ', n_channels)  # 61 (paper says only 60?)

    # DECONVOLVE FLUXES WITH BEAM PSF
    # source activate iraf27; in /Users/jonathancohn/iraf/, type xgterm; in xgterm, load stsdas, analysis, restore
    # then: lucy input_image.fits psf.fits output_image.fits niter=15 [defaults for both adu and noise, play with niter]
    # currently done in iraf outside python. Output: lucy_output='lucy_out_n5.fits', for niter=5
    # NOTE: the data cube for NGC1332 has redshifted side at the bottom left
    hdu = fits.open(lucy_output)
    lucy_out = hdu[0].data
    hdu.close()

    # NOW INCLUDE ENCLOSED STELLAR MASS
    # BUCKET INTERPOLATE
    radii = []
    m_stellar = []
    with open(enclosed_mass) as em:
        for line in em:
            cols = line.split()
            cols[1] = cols[1].replace('D', 'e')
            radii.append(float(cols[0]) * 10 ** 3)  # file lists radii in kpc; convert to pc
            m_stellar.append(float(cols[1]))  # solar masses

    # SUBPIXELS
    # take deconvolved flux map (output from lucy), take subpixels, assign each subpixel flux=(real pixel flux)/s**2
    # --> these are the weights to apply to gaussians
    print('start ')
    t0 = time.time()
    # subpix_deconvolved is identical to lucy_out, just with sxs subpixels for each pixel and the total flux conserved

    # RESHAPING TO sxs SIZE
    subpix_deconvolved = np.zeros(shape=(len(lucy_out) * s, len(lucy_out[0]) * s))  # 300*s, 300*s
    for xpix in range(len(lucy_out)):
        for ypix in range(len(lucy_out[0])):
            subpix_deconvolved[(xpix * s):(xpix + 1) * s, (ypix * s):(ypix + 1) * s] = lucy_out[xpix, ypix] / s ** 2
    print('deconvolution took {0} s'.format(time.time() - t0))  # ~0.3 s (7.5s for 1280x1280 array)
    # subpix_deconvolved is in (x_dat, ydat) = (-y_obs, x_obs)

    # GAUSSIAN STEP
    # get gaussian velocity for each subpixel, apply weights to gaussians (weights = subpix_deconvolved output)

    # SET UP VELOCITY AXIS
    if vsys is None:
        v_sys = constants.H0 * dist
    else:
        v_sys = vsys
    # 15.4 * 10^6 Hz corresponds to 20.1 km/s  # from Barth+2016

    # convert from frequency (Hz) to velocity (km/s), with freq_ax in Hz
    # CO(2-1) lands in 2.2937*10^11 Hz
    f_0 = 2.29369132e11  # intrinsic frequency of CO(2-1) line
    # z_ax = np.asarray([v_sys - ((freq - f_0)/f_0) * (constants.c / constants.m_per_km) for freq in freq_ax])
    z_ax = np.asarray([v_sys - ((f_0 - freq) / freq) * (constants.c / constants.m_per_km) for freq in freq_ax])
    print(z_ax[1] - z_ax[0])  # 20.1 km/s yay!

    # SET UP OBSERVATION AXES
    # initialize all values along axes at 0., but with a length equal to axis length [arcsec] * oversampling factor /
    # resolution [arcsec / pixel]  --> units of pixels along the observed axes
    '''
    x_obs = [0.] * int(x_width * s / resolution)
    y_obs = [0.] * int(y_width * s / resolution)
    '''
    # y_obs = [0.] * len(lucy_out[0]) * s
    # x_obs = [0.] * len(lucy_out) * s
    y_obs = [0.] * len(lucy_out) * s
    x_obs = [0.] * len(lucy_out[0]) * s
    # Initialize obs axes, given x_dat = -y_obs, y_dat = x_obs
    # print(x_width * s / resolution, 'is this float a round integer? hopefully!')

    # set center of the observed axes (find the central pixel number along each axis)
    if len(x_obs) % 2. == 0:  # if even
        x_ctr = (len(x_obs)) / 2  # set the center of the axes (in pixel number)
        for i in range(len(x_obs)):
            x_obs[i] = resolution * (i - x_ctr) / s  # (arcsec/pix) * N_subpixels / (subpixels/pix) = arcsec
    else:  # elif odd
        x_ctr = (len(x_obs) + 1.) / 2  # +1 bc python starts counting at 0
        for i in range(len(x_obs)):
            x_obs[i] = resolution * ((i + 1) - x_ctr) / s  # (arcsec/pix) * N_subpixels / (subpixels/pix) = arcsec
    # repeat for y-axis
    if len(y_obs) % 2. == 0:
        y_ctr = (len(y_obs)) / 2
        for i in range(len(y_obs)):
            y_obs[i] = resolution * (i - y_ctr) / s
    else:
        y_ctr = (len(y_obs) + 1.) / 2
        for i in range(len(y_obs)):
            y_obs[i] = resolution * ((i + 1) - y_ctr) / s

    '''
    x_ctr = (len(x_obs) + 1.) / 2  # +1 bc python starts counting at 0
    y_ctr = (len(y_obs) + 1.) / 2  # +1 bc python starts counting at 0

    # now fill in the x,y observed axis values [arcsec], based on the axis length, oversampling size, and resolution
    for i in range(len(x_obs)):
        x_obs[i] = resolution * ((i + 1) - x_ctr) / s  # (arcsec/pix) * subpixels / (subpixels/pix) = arcsec
    for i in range(len(y_obs)):
        y_obs[i] = resolution * ((i + 1) - y_ctr) / s
    '''

    # SET BH POSITION [in arcsec], based on the input offset values
    # DON'T divide offset positions by s, unless offset positions are in subpixels instead of pixels
    # currently estimating offsets by pixels, not subpixels
    # the offsets should be in subpixels?
    # x_bhctr = x_off * resolution  # / s
    # y_bhctr = y_off * resolution  # / s
    x_bhctr = y_off * resolution
    y_bhctr = -x_off * resolution

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
    # x_disk = (x_obs[:, None] - x_bhctr) * np.cos(theta) - (y_obs[None, :] - y_bhctr) * np.sin(theta)  # 2d array
    # y_disk = (y_obs[None, :] - y_bhctr) * np.cos(theta) + (x_obs[:, None] - x_bhctr) * np.sin(theta)  # 2d array
    # NOTE: if I use the below definition, then I'm just saying theta goes clockwise now
    x_disk = (x_obs[None, :] - x_bhctr) * np.cos(theta) + (y_obs[:, None] - y_bhctr) * np.sin(theta)  # 2d array
    y_disk = (y_obs[:, None] - y_bhctr) * np.cos(theta) - (x_obs[None, :] - x_bhctr) * np.sin(theta)  # 2d array
    print('x, y disk', x_disk.shape, y_disk.shape)
    # print(x_disk)
    # print(y_disk)

    # CALCULATE THE RADIUS (R) OF EACH POINT (x_disk, y_disk) IN THE DISK (pc)
    R = np.sqrt((x_disk ** 2 / np.cos(inc) ** 2) + y_disk ** 2)  # radius R of each point in the disk (2d array)
    # print(R.shape)

    '''  #
    plt.contourf(x_obs, y_obs, R, 600, vmin=np.amin(R), vmax=np.amax(R), cmap='viridis')
    plt.xlabel(r'x [pc]', fontsize=30)
    plt.ylabel(r'y [pc]', fontsize=30)
    plt.ylim(min(y_obs), max(y_obs))
    plt.xlim(min(x_obs), max(x_obs))
    plt.plot(x_bhctr, y_bhctr, 'k*', markersize=20)
    cbar = plt.colorbar()
    cbar.set_label(r'km/s', fontsize=30, rotation=0, labelpad=25)  # pc,  # km/s
    plt.show()
    print(oops)
    # '''  #

    # CALCULATE ENCLOSED MASS BASED ON MBH AND ENCLOSED STELLAR MASS
    # look up spline interpolation (eg interpol in idl: provide x,y of r array, masses array, then give it R)
    t_mass = time.time()

    # INTRODUCE MASS AS A FUNCTION OF RADIUS (INCLUDE ENCLOSED STELLAR MASS)
    # CREATE A FUNCTION TO INTERPOLATE (AND EXTRAPOLATE) ENCLOSED STELLAR M(R)
    m_star_r = interpolate.interp1d(radii, m_stellar, fill_value='extrapolate')
    m_R = mbh + ml_const * m_star_r(R)  # 2d array
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

    print(m_R)
    print('Time elapsed in assigning enclosed masses is {0} s'.format(time.time() - t_mass))  # ~3.5s
    # print(m_R)
    # print(m_R.shape)

    # DRAW ELLIPSE
    from matplotlib import patches
    # USE 50pc FOR MODEL FITTING
    # USE 4.3" x 0.7" FOR FIG 2 REPLICATION
    # major = 50
    # minor = 50*np.cos(inc)
    major = dist * 10 ** 6 * np.tan(4.3 / constants.arcsec_per_rad)
    minor = dist * 10 ** 6 * np.tan(0.7 / constants.arcsec_per_rad)
    par = np.linspace(0, 2 * np.pi, 100)
    ell = np.asarray([major * np.cos(par), minor * np.sin(par)])
    theta_rot = theta + np.pi/2.  # 26.7*np.pi/180.  # 153*pi/180. (90.+116.7)*np.pi/180.
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
    v_los = np.sqrt(constants.G_pc * m_R) * np.sin(inc) * y_disk / ((x_disk / np.cos(inc)) ** 2 + y_disk ** 2) ** (
                3 / 4)
    # line-of-sight velocity v_los at each point in the disk
    print('los')

    # ALTERNATIVE CALCULATION FOR v_los
    alpha = abs(np.arctan((y_disk * np.cos(inc)) / x_disk))  # alpha meas. from +x (minor axis) toward +y (major axis)
    sign = y_disk / abs(y_disk)  # if np.pi/2 < alpha < 3*np.pi/2, alpha < 0.
    v_los2 = sign * abs(vel * np.sin(alpha) * np.sin(inc))
    # print(v_los - v_los2)  # 0 YAY!

    # SET LINE-OF-SIGHT VELOCITY AT THE BLACK HOLE CENTER TO BE 0, SUCH THAT IT DOES NOT BLOW UP
    # if any point as at x_disk, y_disk = (0., 0.), set velocity there = 0.
    # Only relevant if we have pixel located exactly at the center
    center = (R == 0.)
    v_los[center] = 0.
    v_los2[center] = 0.
    # print(center, v_los[center])

    # CALCULATE OBSERVED VELOCITY
    v_obs = v_sys - v_los2  # observed velocity v_obs at each point in the disk
    print('v_obs')

    # SELECT ALL VELOCITIES WITHIN ELLIPTICAL REGION CENTERED ON BH
    '''  #
    indices = R < 50.
    plt.contourf(x_obs, y_obs, v_obs, 600, vmin=np.amin(v_obs), vmax=np.amax(v_obs), cmap='RdBu_r')
    plt.plot(x_bhctr + ell_rot[0, :], y_bhctr + ell_rot[1, :], 'k')
    plt.xlabel(r'x [pc]', fontsize=30)
    plt.ylabel(r'y [pc]', fontsize=30)
    plt.colorbar()
    # THIS PLOT INDICATES x_obs = -y_data, y_obs = x_data, so this shows x_dat vs -y_dat
    plt.show()

    v_ctr = v_obs
    x_reg = x_obs
    y_reg = y_obs
    v_ctr[R > 50] = v_sys  # ignore all velocities outside R=50pc ellipse
    plt.contourf(x_reg, y_reg, v_ctr, 600, vmin=np.amin(v_ctr), vmax=np.amax(v_ctr), cmap='RdBu_r')
    plt.colorbar()
    plt.show()

    plt.hist(v_obs[R <= 300], len(z_ax), edgecolor='k', facecolor=None)
    plt.axvline(x=v_sys, color='k')
    plt.xlabel(r'Observed velocity [km/s]')
    plt.ylabel(r'N [model]')
    plt.show()
    print(oops)
    # '''  #

    # '''  #
    # MAKE MODEL VERSION OF FIG2
    hdu = fits.open(data_cube)
    data_vs3 = hdu[0].data[0]  # header = hdu[0].header

    obs_vels = []
    mask2 = []
    major = dist * 10 ** 6 * np.tan(4.3 / 0.01 / constants.arcsec_per_rad)
    minor = dist * 10 ** 6 * np.tan(0.7 / 0.01 / constants.arcsec_per_rad)
    for x in range(len(x_obs)):
        print(x)
        for y in range(len(y_obs)):
            test_pt = ((np.cos(theta_rot)*(x - x_bhctr) + np.sin(theta_rot)*(y - y_bhctr))/major)**2\
                + ((np.sin(theta_rot)*(x - x_bhctr) - np.cos(theta_rot)*(y-y_bhctr))/minor)**2
            if test_pt <= 1:
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
        add = np.sum(data_vs2[i])
        # data_vs[i, inds].shape = 640, 640
        o_vels.append(add)
    # plt.hist(v_obs[R <= 300], len(z_ax), edgecolor='k', facecolor=None)  # points within radius rather than ellipse
    plt.bar(z_ax, o_vels, width=(z_ax[1] - z_ax[0]), edgecolor='k')  # data_vs[:, R < 50]
    # plt.hist(vels_f, len(z_ax), edgecolor='k')  # data_vs[:, R < 50]
    plt.axvline(x=v_sys, color='k')
    plt.xlabel(r'Observed velocity [km/s]')
    plt.ylabel(r'N [model]')
    plt.show()
    # '''  #

    # RECREATE BARTH+2016 FIG2, CO(2-1) LINE PROFILE INTEGRATED OVER ELLIPSE
    # '''  #
    hdu = fits.open(data_cube)
    data_vs = hdu[0].data[0]  # header = hdu[0].header
    print(data_vs.shape)  # (75, 640, 640)
    # for xo, yo in [[6., 0.], [8., 0.], [4., 0.], [6., -2.], [8., -2.], [4., -2.], [6., 2.], [8., 2.], [4., 2.]]:
    # [3., 20.], [5., 20.], [7, 20.], [3., 17.], [5., 17.], [7., 17.], [3., 15.], [5., 15.], [7., 15.]]:
    for xo, yo in [[0., 0.,]]:
        vels = []
        acceptable_pixels = []
        inds = np.zeros(shape=(len(data_vs[0]), len(data_vs[0][0])))  # array of indices, corresponding to x,y data
        # TEST each x,y point: is it within ellipse?
        tests = []
        mask = np.zeros(shape=(len(data_vs[0]), len(data_vs[0][0])))  # array of indices, corresponding to x,y data
        for i in range(len(data_vs[0][0])):  # 640 (x axis)
            for j in range(len(data_vs[0])):  # 640 (y axis)
                res = 0.01  # arcsec/pix  # 0.01 now, but need to bin to 4x4 pixels
                maj = 4.3 / res  # ellipse major axis
                mi = 0.7 / res  # ellipse minor axis
                theta_rot = theta + np.pi/4.
                test_pt = ((np.cos(theta_rot) * (i - (320 + xo)) + np.sin(theta_rot) * (j - (320 + yo))) / maj) ** 2 \
                          + ((np.sin(theta_rot) * (i - (320 + xo)) - np.cos(theta_rot) * (j - (320 + yo))) / mi) ** 2
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

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title(str(xo) + ', ' + str(yo))
        plt.imshow(data_vs2[20])  # (data_vs[45])  # (data_vs[30])
        plt.gca().invert_yaxis()
        ax.set_aspect('equal')
        plt.colorbar(orientation='vertical')
        plt.show()

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
            add = np.sum(data_vs2[i])
            # data_vs[i, inds].shape = 640, 640
            vels.append(add)
        print(vels)
        print(len(vels))
        plt.bar(z_ax, vels, width=(z_ax[1] - z_ax[0]), edgecolor='k')  # data_vs[:, R < 50]
        # plt.hist(vels_f, len(z_ax), edgecolor='k')  # data_vs[:, R < 50]
        plt.title(str(xo) + ', ' + str(yo))
        plt.axvline(x=v_sys, color='k')
        plt.xlabel(r'Observed velocity [km/s]')
        plt.ylabel(r'N [model]')
        plt.show()
    print(oops)
    # '''  #

    # CALCULATE VELOCITY PROFILES
    sigma = get_sig(r=R, sig0=sig_params[0], r0=sig_params[1], mu=sig_params[2], sig1=sig_params[3])[sig_type]
    print(sigma)
    obs3d = []  # data cube, v_obs but with a wavelength axis, too!
    weight = subpix_deconvolved
    # weight /= np.sqrt(2 * np.pi * sigma)
    # NOTE: BUCKET: weight = weight / (sqrt(2*pi)*sigma)
    # print(weight, np.amax(weight), np.amin(weight))  # most ~0.003, max 0.066, min 1e-15
    tz1 = time.time()
    for z in range(len(z_ax)):
        print(z)
        # print(z_ax[z] - v_obs[int(len(v_obs)-1),int(len(v_obs)-1)])  #, z_ax[z], v_obs)
        obs3d.append(weight * np.exp(-(z_ax[z] - v_obs) ** 2 / (2 * sigma ** 2)))
    print('obs3d')
    obs3d = np.asarray(obs3d)  # has shape 61, 600, 600, which is good
    print('obs3d took {0} s'.format(time.time() - tz1))  # ~3.7 s; 14s for s=2 1280x1280
    print(obs3d.shape)

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
    convolved_cube = np.rot90(convolved_cube, axes=(1, 2))
    # print(np.swapaxes(convolved_cube, 1, 2).shape)

    # '''  #
    # TRY THINGS AGAIN
    hdu = fits.open(data_cube)
    data_vs = hdu[0].data[0]  # header = hdu[0].header
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
                res = 0.01  # arcsec/pix  # should be 0.1, use 0.01 while finding things
                maj = 4.3 / res  # ellipse major axis
                mi = 0.7 / res  # ellipse minor axis
                theta_rot = theta + np.pi/4.
                test_pt = ((np.cos(theta_rot) * (i - (320 + xo)) + np.sin(theta_rot) * (j - (320 + yo))) / maj) ** 2 \
                          + ((np.sin(theta_rot) * (i - (320 + xo)) - np.cos(theta_rot) * (j - (320 + yo))) / mi) ** 2
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

        inds_to_try2 = np.asarray([[315, 337], [337, 325], [340, 340], [350, 325], [350, 350],
                                   [325, 350], [320, 360], [315, 315], [337, 315], [270, 411],
                                   [339, 329], [10, 450]])  # [411, 370]->[270,411]; [329, 301], [346, 405]
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
    lucy = '/Users/jonathancohn/Documents/dyn_mod/newdata01_beam77_lucy_n15.fits'
    out = '1332_newdata01_apconv_rot90-12_n15_gsize35_tryagain.fits'

    # BLACK HOLE MASS (M_sol), RESOLUTION (ARCSEC), VELOCITY SPACING (KM/S)
    mbh = 6.86 * 10 ** 8  # , 20.1 (v_spacing no longer necessary)
    resolution = 0.01  # 0.05  # 0.044

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
    theta = np.deg2rad(26.7 + 180.)  # (180. + 116.7)  # -333 (-333.3 from 116.7: -(360 - (116.7-90)) = -333.3
    vsys = 1562.2  # km/s
    # Note: 22.3 * 10**6 * tan(640 * 0.01 / 206265) = 692 pc (3460 pc if use 0.05?)
    # 692 pc / 640 pix =~ 1.1 pc/pix

    # OVERSAMPLING FACTOR
    s = 1

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
