import numpy as np
import astropy.io.fits as fits
# from astropy.table import Table
import matplotlib.pyplot as plt
from scipy import integrate
from scipy import signal
# from astropy import modeling
import time
# from astropy import convolution

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
        self.c = 3.*10**8  # m/s
        self.pc = 3.086*10**16  # m
        self.G = 6.67*10**-11  # kg^-1 * m^3 * s^-2
        self.M_sol = 2.*10**30  # kg
        self.H0 = 70  # km/s/Mpc
        self.arcsec_per_rad = 206265  # arcsec per radian
        self.m_per_km = 10**3  # m per km
        self.G_pc = self.G * self.M_sol * (1 / self.pc) / self.m_per_km**2  # Msol^-1 * pc * km^2 * s^-2


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


def make_beam_psf(grid_size=100, amp=1., x0=0., y0=0., x_std=1., y_std=1., rot=0., fits_name=None):
    """
    Use to generate a beam psf (and to create a beam psf fits file to use in lucy

    :param grid_size: size
    :param amp: amplitude of the 2d gaussian
    :param x0: mean of x axis of 2d gaussian
    :param y0: mean of y axis of 2d gaussian
    :param x_std: standard deviation of Gaussian in x
    :param y_std: standard deviation of Gaussian in y
    :param rot: rotation angle in radians
    :param fits_name: this name will be the filename to which the psf fits file is written (if None, write no file)

    return the beam psf
    """

    # SET UP MESHGRID
    x_psf = np.linspace(-1., 1., grid_size)
    y_psf = np.linspace(-1., 1., grid_size)
    xx, yy = np.meshgrid(x_psf, y_psf)

    # SET UP PSF 2D GAUSSIAN VARIABLES
    a = np.cos(rot)**2 / (2*x_std**2) + np.sin(rot)**2 / (2*y_std**2)
    b = -np.sin(2*rot) / (4*x_std**2) + np.sin(2*rot) / (4*y_std**2)
    c = np.sin(rot)**2 / (2*x_std**2) + np.cos(rot)**2 / (2*y_std**2)

    # CALCULATE PSF, NORMALIZE IT TO AMPLITUDE
    psf = np.exp(-(a*(xx-x0)**2 + 2*b*(xx-x0)*(yy-y0) + c*(yy-y0)**2))
    A = amp / np.amax(psf)
    psf *= A

    # IF write_fits, WRITE PSF TO FITS FILE
    if fits_name is not None:
        hdu = fits.PrimaryHDU(psf)
        hdul = fits.HDUList([hdu])
        hdul.writeto(fits_name)

    return psf


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
    collapsed_fluxes = integrate.simps(data, axis=0)  # according to my python terminal tests
    print(collapsed_fluxes.shape)

    z_len = len(hdu[0].data[0])  # store the number of velocity slices in the data cube
    freq1 = float(hdu[0].header['CRVAL3'])
    f_step = float(hdu[0].header['CDELT3'])
    freq_axis = np.arange(freq1, freq1 + (z_len * f_step), f_step)
    # BUCKET: this results in 62 frequency channels, not 61 (or 60) --> problem?
    hdu.close()

    if write_name is not None:
        hdu = fits.PrimaryHDU(collapsed_fluxes)
        hdul = fits.HDUList([hdu])
        hdul.writeto(write_name)

    return collapsed_fluxes, freq_axis


def model_grid(x_width=0.975, y_width=2.375, resolution=0.05, s=10, n_channels=52, spacing=20.1, x_off=0., y_off=0.,
               mbh=4*10**8, inc=np.deg2rad(60.), dist=17., theta=np.deg2rad(-200.), data_cube=None, lucy_output=None,
               out_name=None, incl_fig=False, enclosed_mass=None, ml_const=1.):
    """
    Build grid for dynamical modeling!

    :param x_width: width of observations along x-axis [arcsec]
    :param y_width: width of observations along y-axis [arcsec]
    :param resolution: resolution of observations [arcsec]
    :param s: oversampling factor
    :param n_channels: number of frequency (or wavelength) channels in the wavelength axis of the data cube
    :param spacing: spacing of the channels in the wavelength axis of the data cube [km/s] [or km/s/pixel? Barth+16,pg7]
    :param x_off: the location of the BH, offset from the center in the +x direction [pixels]
    :param y_off: the location of the BH, offset from the center in the +y direction [pixels]
    :param mbh: supermassive black hole mass [solar masses]
    :param inc: inclination of the galaxy [radians]
    :param dist: distance to the galaxy [Mpc]
    :param theta: angle from the redshifted side of the disk (+y_disk) counterclockwise to the +y_obs axis [radians]
        (angle input must be negative to go counterclockwise)
    :param data_cube: input data cube of observations
    :param lucy_output: output from running lucy on data cube and beam PSF
    :param out_name: output name of the fits file to which to save the output v_los image (if None, don't save image)
    :param incl_fig: if True, print figure of 2d plane of observed line-of-sight velocity
    :param enclosed_mass: file including data of the enclosed stellar mass of the galaxy (1st column should be radius
        r in kpc and second column should be M_stars(<r))
    :param ml_const: Constant by which to multiply the M/L ratio, in case the M/L ratio is not that which was used to
        calculate the enclosed stellar masses in the enclosed_mass file

    :return: observed line-of-sight velocity [km/s]
    """
    # INSTANTIATE ASTRONOMICAL CONSTANTS
    constants = Constants()

    # COLLAPSE THE DATA CUBE
    # fluxes, n_channels = get_fluxes(data_cube)  # , write=True, write_name='NGC1332_collapsed.fits')
    fluxes, freq_ax = get_fluxes(data_cube)
    # , write_name='1332_integratedcollapsed.fits'
    # , write=True, write_name='NGC1332_collapsed.fits')
    # print('n_channels: ', n_channels)  # 61 (paper says only 60?)

    # DECONVOLVE FLUXES WITH BEAM PSF
    # currently done in iraf outside python. Output: lucy_output='lucy_out_n5.fits', for niter=5
    # NOTE: the data cube for NGC1332 has redshifted side at the bottom left
    hdu = fits.open(lucy_output)
    lucy_out = hdu[0].data
    hdu.close()

    # NOW INCLUDE ENCLOSED STELLAR MASS
    radii = []
    m_stellar = []
    with open(enclosed_mass) as em:
        for line in em:
            cols = line.split()
            cols[1] = cols[1].replace('D', 'e')
            radii.append(float(cols[0]) * 10**3)  # file lists radii in kpc; convert to pc
            m_stellar.append(float(cols[1]))  # solar masses

    # SUBPIXELS
    # take deconvolved flux map (output from lucy), take subpixels, assign each subpixel flux=(real pixel flux)/s**2
    # --> these are the weights to apply to gaussians
    print('start ')
    t0 = time.time()
    # subpix_deconvolved is identical to lucy_out, just with sxs subpixels for each pixel and the total flux conserved

    # RESHAPING
    subpix_deconvolved = np.zeros(shape=(len(lucy_out) * s, len(lucy_out[0]) * s))  # 300*s, 300*s
    for x_subpixel in range(len(subpix_deconvolved)):
        for y_subpixel in range(len(subpix_deconvolved[0])):
            xpix = int(x_subpixel / s)
            ypix = int(y_subpixel / s)
            subpix_deconvolved[x_subpixel, y_subpixel] = lucy_out[xpix, ypix] / s**2
            # USING THIS ORIENTATION IS CORRECT IF subpix_deconvolved = weight IS PLOTTED with x,y = -y_obs, x_obs
    print('deconvolution took {0} s'.format(time.time() - t0))  # ~10.3 s
    # print(subpix_deconvolved.shape)  # (1800,1800)

    # GAUSSIAN STEP
    # get gaussian velocity for each subpixel, apply weights to gaussians (weights = subpix_deconvolved output)

    # SET UP VELOCITY AXIS
    v_sys = constants.H0 * dist
    # z_ax = np.linspace(v_sys - n_channels * spacing / 2, v_sys + n_channels * spacing / 2, n_channels + 1)
    # 15.4 * 10^6 Hz corresponds to 20.1 km/s  # from Barth+2016
    # convert from frequency (Hz) to velocity (km/s), with freq_ax in Hz
    # print(freq_ax)
    # CO(2-1) lands in 2.2937*10^11 Hz
    f_0 = 2.29369132e11
    z_ax = np.asarray([v_sys - ((freq-f_0)/f_0) * (constants.c / constants.m_per_km) for freq in freq_ax])
    # print(z_ax[1] - z_ax[0])  # 20.1 km/s yay!

    # SET UP OBSERVATION AXES
    # initialize all values along axes at 0., but with a length equal to axis length [arcsec] * oversampling factor /
    # resolution [arcsec / pixel]  --> units of pixels along the observed axes
    '''
    x_obs = [0.] * int(x_width * s / resolution)
    y_obs = [0.] * int(y_width * s / resolution)
    '''

    # NOTE: I JUST SWITCHED x, y TO WHAT I THINK THEY SHOULD BE
    y_obs = [0.] * len(lucy_out[0]) * s
    x_obs = [0.] * len(lucy_out) * s
    # print(x_width * s / resolution, 'is this float a round integer? hopefully!')

    # set center of the observed axes (find the central pixel number along each axis)
    if len(x_obs) % 2. == 0:  # if even
        x_ctr = (len(x_obs)) / 2
        for i in range(len(x_obs)):
            x_obs[i] = resolution * (i - x_ctr) / s  # (arcsec/pix) * subpixels / (subpixels/pix) = arcsec
    else:
        x_ctr = (len(x_obs) + 1.) / 2  # +1 bc python starts counting at 0
        for i in range(len(x_obs)):
            x_obs[i] = resolution * ((i + 1) - x_ctr) / s  # (arcsec/pix) * subpixels / (subpixels/pix) = arcsec
    if len(y_obs) % 2. == 0:
        y_ctr = (len(y_obs)) / 2
        for i in range(len(y_obs)):
            y_obs[i] = resolution * (i + - y_ctr) / s
    else:
        y_ctr = (len(y_obs) + 1.) / 2  # +1 bc python starts counting at 0
        for i in range(len(y_obs)):
            y_obs[i] = resolution * ((i + 1) - y_ctr) / s
    # print(x_ctr, y_ctr, len(lucy_out), len(x_obs))

    '''
    x_ctr = (len(x_obs) + 1.) / 2  # +1 bc python starts counting at 0
    y_ctr = (len(y_obs) + 1.) / 2  # +1 bc python starts counting at 0

    # now fill in the x,y observed axis values [arcsec], based on the axis length, oversampling size, and resolution
    for i in range(len(x_obs)):
        x_obs[i] = resolution * ((i + 1) - x_ctr) / s  # (arcsec/pix) * subpixels / (subpixels/pix) = arcsec
    for i in range(len(y_obs)):
        y_obs[i] = resolution * ((i + 1) - y_ctr) / s
    '''

    # SET BH POSITION [arcsec], based on the input offset values
    # DON'T divide offset positions by s, unless offset positions are in subpixels instead of pixels
    # currentyl estimating offsets by pixels, not subpixels
    x_bhctr = x_off * resolution  # / s
    y_bhctr = -y_off * resolution  # / s  # BUCKET: note: use -y_off because I have to plot this in -y

    # CONVERT FROM ARCSEC TO PHYSICAL UNITS (Mpc)
    # tan(angle) = x/d where d=dist and x=disk_radius --> x = d*tan(angle), where angle = arcsec / arcsec_per_rad

    # convert BH position from arcsec to pc
    x_bhctr = dist * 10**6 * np.tan(x_bhctr / constants.arcsec_per_rad)
    y_bhctr = dist * 10**6 * np.tan(y_bhctr / constants.arcsec_per_rad)
    print(x_bhctr, y_bhctr)

    # convert all x,y observed grid positions to pc
    # x_obs = np.asarray([dist * 10**6 * np.tan(x / constants.arcsec_per_rad) for x in x_obs])  # 206265 arcsec/rad
    # y_obs = np.asarray([dist * 10**6 * np.tan(y / constants.arcsec_per_rad) for y in y_obs])  # 206265 arcsec/rad
    x_obs = np.asarray([dist * 10**6 * np.tan(x / constants.arcsec_per_rad) for x in x_obs])  # 206265 arcsec/rad
    y_obs = np.asarray([dist * 10**6 * np.tan(y / constants.arcsec_per_rad) for y in y_obs])  # 206265 arcsec/rad
    print((x_obs[1] - x_obs[0])*s)  # 7.56793447282
    print(x_obs[0], x_obs[s])  # (-1134.5595077622734, -1126.9915732895627)

    # at each x,y spot in grid, calculate what x_disk and y_disk are, then calculate R, v, etc.
    # CONVERT FROM x_obs, y_obs TO x_disk, y_disk (still in pc)
    x_disk = (x_obs[:, None] - x_bhctr) * np.cos(theta) - (y_obs[None, :] - y_bhctr) * np.sin(theta)
    y_disk = (y_obs[None, :] - y_bhctr) * np.cos(theta) + (x_obs[:, None] - x_bhctr) * np.sin(theta)
    print('x, y disk', x_disk.shape, y_disk.shape)
    # print(x_disk)
    # print(y_disk)

    # CALCULATE THE RADIUS (R) OF EACH POINT (x_disk, y_disk) IN THE DISK (pc)
    R = np.sqrt((x_disk ** 2 / np.cos(inc) ** 2) + y_disk ** 2)  # radius R of each point in the disk
    print(R.shape)

    # CALCULATE ENCLOSED MASS BASED ON MBH AND ENCLOSED STELLAR MASS
    # BUCKET THERE HAS TO BE A FASTER WAY
    t_mass = time.time()
    m_R = np.zeros(shape=(len(R), len(R[0])))  # mass as a function of R
    for x in range(len(R)):
        print(x)
        for y in range(len(R[0])):
            for rad in range(len(radii) - 1):
                if radii[rad] <= R[x, y] < radii[rad+1]:
                    m_R[x, y] = mbh + ml_const * m_stellar[rad]
                elif R[x, y] > radii[-1]:
                    m_R[x, y] = mbh + ml_const * m_stellar[-1]
    print('Time elapsed in assigning enclosed masses is {0} s'.format(time.time() - t_mass))  # 12s w/ s=2; 118s w/ s=6
    print(m_R)

    # CALCULATE KEPLERIAN VELOCITY OF ANY POINT (x_disk, y_disk) IN THE DISK WITH RADIUS R (km/s)
    vel = np.sqrt(constants.G_pc * m_R / R)  # Keplerian velocity vel at each point in the disk
    # vel = np.sqrt(constants.G_pc * mbh / R)  # Keplerian velocity vel at each point in the disk
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
    v_los = np.sqrt(constants.G_pc * m_R) * np.sin(inc) * y_disk / ((x_disk / np.cos(inc)) ** 2 + y_disk ** 2) ** (3/4)
    # line-of-sight velocity v_los at each point in the disk
    print('los')

    # ALTERNATIVE CALCULATION FOR v_los
    alpha = abs(np.arctan((y_disk * np.cos(inc)) / x_disk))  # alpha meas. from +x (minor axis) toward +y (major axis)
    sign = y_disk / abs(y_disk)  # if np.pi/2 < alpha < 3*np.pi/2, alpha < 0.
    v_los2 = sign * abs(vel * np.sin(alpha) * np.sin(inc))
    # print(v_los - v_los2)  # 0 YAY!
    # print(v_los.shape)  # (300*s,300*s) yay!

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
    # print(v_obs - v_los)

    sigma = 10.  # * (1 + np.exp(-R[x, y])) # sig: const, c1+c2*e^(-r/r0), c1+exp[-(x-mu)^2 / (2*sig^2)]
    sig1 = 0.  # km/s
    sig0 = 268.  # km/s
    r0 = 60.7  # pc
    sigma = sig0 * np.exp(-R / r0) + sig1
    obs3d = []  # data cube, v_obs but with a wavelength axis, too!
    weight = subpix_deconvolved
    weight /= np.sqrt(2 * np.pi * sigma)
    # NOTE: BUCKET: weight = weight / (sqrt(2*pi)*sigma)
    # print(weight, np.amax(weight), np.amin(weight))  # most ~0.003, max 0.066, min 1e-15
    for z in range(len(z_ax)):
        obs3d.append(weight * np.exp(-(z_ax[z] - v_obs) ** 2 / (2 * sigma ** 2)))
    obs3d = np.asarray(obs3d)  # has shape 61, 600, 600, which is good
    print('obs3d')

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
    central = obs3d[:,300,300]
    central1 = obs3d[:,298,298]
    central2 = obs3d[:,299,301]
    central3 = obs3d[:,301,299]
    nonctr = obs3d[:,0,0]
    # plt.plot(z_ax, central1, 'b--')
    # plt.plot(z_ax, central2, 'g--')
    # plt.plot(z_ax, central3, 'm--')
    # plt.plot(z_ax, central, 'k--')
    # plt.plot(z_ax, nonctr, 'k-')
    # plt.axvline(x=v_sys, color='r')
    # plt.show()
    print(obs3d[:,300,300])
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
    # RATE-DETERMINING STEP
    t_z = time.time()
    intrinsic_cube = np.zeros(shape=(len(z_ax), len(fluxes), len(fluxes[0])))
    for z2 in range(len(z_ax)):
        print(z2)
        for xreal in range(len(intrinsic_cube[z2, :, 0])):
            for yreal in range(len(intrinsic_cube[z2, 0, :])):
                intrinsic_cube[z2, xreal, yreal] =\
                    np.mean(obs3d[z2, int(xreal * s):int((xreal + 1) * s), int(yreal * s):int((yreal + 1) * s)])
    print("intrinsic cube done in {0} s".format(time.time() - t_z))  # ~37-45 s (better than ~68!)

    plt.plot(z_ax, intrinsic_cube[:,158,168], 'k-')
    plt.plot(z_ax, intrinsic_cube[:,149,153], 'r-')
    plt.plot(z_ax, intrinsic_cube[:,165,178], 'b-')
    plt.plot(z_ax, intrinsic_cube[:152,172], 'm--')  #
    plt.plot(z_ax, intrinsic_cube[:,163,166], 'g--')
    plt.axvline(x=v_sys, color='k')
    print(v_sys)
    # red solid, blue solid, dotted for along minor axis
    plt.show()

    '''
    hdu = fits.PrimaryHDU(intrinsic_cube)
    hdul = fits.HDUList([hdu])
    hdul.writeto('1332_intrinsic_enclmass_n15_s2.fits')
    print('written!')
    print(oops)
        
    if out_name is not None:
        hdu = fits.PrimaryHDU(intrinsic_cube2)
        hdul = fits.HDUList([hdu])
        hdul.writeto('name.fits')
    # '''

    # CONVERT INTRINSIC TO OBSERVED
    # take velocity slice from intrinsic data cube, convolve with alma beam --> observed data cube
    beam_psf = make_beam_psf(grid_size=len(intrinsic_cube[0, :, :]), x_std=0.319, y_std=0.233, rot=np.deg2rad(90-78.4))
    # print(beam_psf.shape)  # 300, 300
    convolved_cube = []
    start_time = time.time()
    for z3 in range(len(z_ax)):
        t1 = time.time()
        print(z3)
        # CONVOLVE intrinsic_cube[:, :, z3] with alma beam
        # alma_beam needs to have same dimensions as intrinsic_cube[:, :, z3]
        # BUCKET: MAYBE TRY signal.fftconvolve()?
        convolved_step = signal.fftconvolve(intrinsic_cube[z3, :, :], beam_psf, mode='same')
        # convolved_cube.append(signal.convolve2d(intrinsic_cube[:, :, z3], beam_psf, mode='same'))
        print(convolved_step.shape)
        convolved_cube.append(convolved_step)
        print(convolved_cube[z3].shape)  # 300, 300  # now it's 61, 300 :(
        print("Convolution loop " + str(z3) + " took {0} seconds".format(time.time() - t1))
        # ISSUE: each signal.convolve2d step takes ~43-56s (takes ~16 seconds in terminal for same sized objects...)
        # mode=same keeps output size same
        # welp according to Barth+2016 this is the most time-consuming step of the modeling calculations
    convolved_cube = np.asarray(convolved_cube)
    print('convolved! Total convolution loop took {0} seconds'.format(time.time() - start_time))
    # BUCKET: CURRENTLY 300 x 61 x 61 ????

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

        '''
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
        '''

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

    cube = 'NGC_1332_CO21.pbcor.fits'

    psf = make_beam_psf(grid_size=100, x_std=0.319, y_std=0.233, rot=np.deg2rad(90-78.4))  # ,
    # fits_name='psf_out.fits')  # rot: start at +90, subtract 78.4 to get 78.4

    # NOTE: NGC 1332 center appears to be x=168, y=158 (out of (300,300), starting with 1) --> expected at 149.5
    # My code would currently put it at 150,150 which is really the 151st pixel because python starts counting at 0
    # ds9 x-axis is my -y axis: y_off = -(168-151) = -18
    # ds9 y-axis is my +x axis: x_off = (158-151) = 8

    out_cube = model_grid(resolution=0.07, s=2, spacing=20.1, x_off=8., y_off=-18., mbh=6.*10**8, inc=np.deg2rad(83.),
                          dist=22.3, theta=np.deg2rad(-333.), data_cube=cube, lucy_output='lucy_out_n15.fits',
                          out_name='1332_convolved_corrweight_n15_s2.fits', incl_fig=True,
                          enclosed_mass='ngc1332_enclosed_stellar_mass')

    vel_slice = spec(out_cube, x_pix=149, y_pix=149, print_it=True)

    # x_width=21., y_width=21., n_channels=60,
    # NOTE: from Fig3 of Barth+2016, their theta=~117 deg appears to be measured from +x to redshifted side of disk
    #       since I want redshifted side of disk to +y (counterclockwise), and my +y is to the left in their image,
    #       I want -(360 - (117-90)) = -333 deg
    # beam_axes = [0.319, 0.233]  # arcsecs
    # beam_major_axis_PA = 78.4  # degrees
