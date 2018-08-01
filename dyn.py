import numpy as np
import astropy.io.fits as fits
from astropy.table import Table
import matplotlib.pyplot as plt


# constants
class Constants:
    # a bunch of useful astronomy constants

    def __init__(self):
        self.c = 3.*10**8  # m/s
        self.pc = 3.086*10**16  # m
        self.G = 6.67*10**-11  # kg^-1 * m^3 * s^-2
        self.M_sol = 2.*10**30  # kg
        self.H0 = 70  # km/s/Mpc
        self.rad_to_arcsec = 206265  # arcsec per radian
        self.m_to_km = 10**-3  # km per m
        self.G_pc = self.G * self.M_sol * (1 / self.pc) * self.m_to_km**2  # Msol^-1 * pc * km^2 * s^-2


def model_grid(x_width=0.975, y_width=2.375, resolution=0.05, s=10, n_channels=52, spacing=20, x_off=0., y_off=0.,
               mbh=4*10**8, inc=np.deg2rad(60.), dist=17., theta=np.deg2rad(-200.), outname=None, incl_fig=False):
    """
    Build grid for dynamical modeling!

    :param x_width: width of observations along x-axis [arcsec]
    :param y_width: width of observations along y-axis [arcsec]
    :param resolution: resolution of observations [arcsec]
    :param s: oversampling factor
    :param n_channels: number of frequency (or wavelength) channels in the wavelength axis of the data cube
    :param spacing: spacing of the channels in the wavelength axis of the data cube [km/s]
    :param x_off: the location of the BH, offset from the center in the +x direction [pixels]
    :param y_off: the location of the BH, offset from the center in the +y direction [pixels]
    :param mbh: supermassive black hole mass [solar masses]
    :param inc: inclination of the galaxy [radians]
    :param dist: distance to the galaxy [Mpc]
    :param theta: angle from the redshifted side of the disk (+y_disk) counterclockwise to the +y_obs axis [radians]
        (angle input must be negative to go counterclockwise)
    :param outname: the output name of the fits file to which to save the outupt v_los image (if None, don't save image)
    :param incl_fig: if True, print figure of 2d plane of observed line-of-sight velocity

    :return: observed line-of-sight velocity [km/s]
    """
    constants = Constants()

    # SET UP VELOCITY AXIS
    v_sys = 0.#-constants.H0 * dist
    z_ax = np.linspace(v_sys - n_channels * spacing / 2, v_sys + n_channels * spacing / 2, n_channels + 1)

    # SET UP OBSERVATION AXES
    # initialize all values along axes at 0., but with a length equal to axis length [arcsec] * oversampling factor /
    # resolution [arcsec / pixel]  --> units of pixels along the observed axes
    x_obs = [0.] * int(x_width * s / resolution)
    y_obs = [0.] * int(y_width * s / resolution)
    # print(x_width * s / resolution, 'is this float a round integer? hopefully!')

    # set center of the observed axes (find the central pixel number along each axis)
    x_ctr = (len(x_obs) + 1) / 2  # +1 bc python starts counting at 0, and len is odd
    y_ctr = (len(y_obs) + 1) / 2  # +1 bc python starts counting at 0, and len is odd

    # now fill in the x,y observed axis values [arcsec], based on the axis length, oversampling size, and resolution
    for i in range(len(x_obs)):
        x_obs[i] = resolution * ((i + 1) - x_ctr) / s
    for i in range(len(y_obs)):
        y_obs[i] = resolution * ((i + 1) - y_ctr) / s

    # SET BH POSITION [arcsec], based on the input offset values
    x_bhctr = x_off * resolution / s
    y_bhctr = y_off * resolution / s

    # CONVERT FROM ARCSEC TO PHYSICAL UNITS (Mpc)
    # tan(angle) = x/d where d=dist and x=disk_radius --> x = d*tan(angle), where angle = arcsec / arcsec_per_rad

    # convert BH position from arcsec to pc
    x_bhctr = dist * np.tan(x_bhctr / constants.rad_to_arcsec) * 10**6
    y_bhctr = dist * np.tan(y_bhctr / constants.rad_to_arcsec) * 10**6

    # convert all x,y observed grid positions to pc
    x_obs = np.asarray([dist * 10**6 * np.tan(x / constants.rad_to_arcsec) for x in x_obs])  # 206265 arcsec/rad
    y_obs = np.asarray([dist * 10**6 * np.tan(y / constants.rad_to_arcsec) for y in y_obs])  # 206265 arcsec/rad

    # at each x,y spot in grid, calculate what x_disk and y_disk are, then calculate R, v, etc.
    R = np.zeros(shape=(len(x_obs), len(y_obs)))  # radius R of each point in the disk
    vel = np.zeros(shape=(len(x_obs), len(y_obs)))  # Keplerian velocity vel at each point in the disk
    v_los = np.zeros(shape=(len(x_obs), len(y_obs)))  # line-of-sight velocity v_los at each point in the disk
    v_obs = np.zeros(shape=(len(x_obs), len(y_obs)))  # observed velocity v_obs at each point in the disk
    v_obs3d = np.zeros(shape=(len(z_ax), len(x_obs), len(y_obs)))  # data cube, v_obs but with a wavelength axis, too!
    for x in range(len(x_obs)):
        for y in range(len(y_obs)):
            # CONVERT FROM x_obs, y_obs TO x_disk, y_disk (still in pc)
            x_disk = (x_obs[x] - x_bhctr) * np.cos(theta) - (y_obs[y] - y_bhctr) * np.sin(theta)
            y_disk = (y_obs[y] - y_bhctr) * np.cos(theta) + (x_obs[x] - x_bhctr) * np.sin(theta)

            # CALCULATE THE RADIUS (R) OF EACH POINT (x_disk, y_disk) IN THE DISK (pc)
            R[x, y] = np.sqrt((x_disk ** 2 / np.cos(inc) ** 2) + y_disk ** 2)

            # CALCULATE KEPLERIAN VELOCITY OF ANY POINT (x_disk, y_disk) IN THE DISK WITH RADIUS R (km/s)
            vel[x, y] = np.sqrt(constants.G_pc * mbh / R[x, y]) # * (constants.pc / constants.m_to_km)
            # 3.086 * 10 ** 19  # Mpc to km

            # CALCULATE LINE-OF-SIGHT VELOCITY AT EACH POINT (x_disk, y_disk) IN THE DISK (km/s)
            # alpha = np.arctan((y_disk * np.cos(inc)) / x_disk)  # alpha = POSITION ANGLE AROUND THE DISK
            # alpha is measured from +x (minor axis) toward +y (major axis)
            # v_obs = v_sys - v_los = v_sys - v(R)*sin(i)*cos(alpha)
            # = v_sys - sqrt(GM/R)*sin(i)*cos(alpha)
            # = v_sys - sqrt(GM)*sin(i)*cos(alpha)/[(x_obs / cos(i))**2 + y_obs**2]^(1/4)
            # = v_sys - sqrt(GM)*sin(i)*[1 /sqrt((x_obs / (y_obs*cos(i)))**2 + 1)]/[(x_obs/cos(i))**2 + y_obs**2]^(1/4)
            # NOTE: 1/sqrt((x_obs / (y_obs*cos(i)))**2 + 1) = y_obs/sqrt((x_obs/cos(i))**2 + y_obs**2) = y_obs / sqrt(R)
            #       = v_sys - sqrt(GM)*sin(i)*y_obs / [(x_obs / cos(i))**2 + y_obs**2]^(3/4)
            v_los[x, y] = np.sqrt(constants.G_pc * mbh) * np.sin(inc) * y_disk / \
                          ((x_disk / np.cos(inc)) ** 2 + y_disk ** 2) ** (3 / 4)

            # SET LINE-OF-SIGHT VELOCITY AT THE BLACK HOLE CENTER TO BE 0, SUCH THAT IT DOES NOT BLOW UP
            if x_obs[x] == x_bhctr and y_obs[y] == y_bhctr:
                v_los[x, y] = 0.

            # CALCULATE OBSERVED VELOCITY
            v_obs[x, y] = v_sys - v_los[x, y]

            # CALCULATE WAVELENGTH AXIS
            for z in range(len(z_ax)):
                A = 1. / 10.
                sigma = 50. * (1 + np.exp(-R[x, y]))  # sig: const, c1+c2*e^(-r/r0), const+gaussian
                # sigma = 50.*(1+np.exp(-R[x, y]))  # sig: const, c1+c2*e^(-r/r0), const+gaussian  # didn't change much
                v_obs3d[z, x, y] = A * np.exp(-(z_ax[z] - v_obs[x, y]) ** 2 / (2 * sigma ** 2))
                if x == 97 and y == 238:  # corresponds to pixels 98 & 240, which are max
                    print(v_obs3d[z, x, y], x, y, z_ax[z], v_obs[x, y])

    # WRITE OUT RESULTS TO FITS FILE
    if outname is not None:
        hdu = fits.PrimaryHDU(v_obs3d)
        hdul = fits.HDUList([hdu])
        hdul.writeto(outname)

    # PRINT 2D DISK ROTATION FIG
    if incl_fig:

        fig = plt.figure(figsize=(12, 10))
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

    return v_obs3d


if __name__ == "__main__":

    v = model_grid(outname=None, incl_fig=True)
    print(v)
