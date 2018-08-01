import numpy as np
import astropy.io.fits as fits
from astropy.table import Table
import matplotlib.pyplot as plt

outname = 'delete.fits'# 'output_vel_3d_expsig_vsys.fits'
x_off = 0#10
y_off = 0#15
# NEW CENTER at 113, 215 (out of max 205, 459 --> orig center 103, 230)
# MINE: 253 / 475, and 108 / 195
## 238 + 15 = 253, 98 + 10 = 108 hooray!!

# constants
c = 3*10**5
m_to_Mpc = 3.24*10**-23
km_to_Mpc = 3.24*10**-20  # Mpc per km
G0 = 6.67 * 10 ** -11  # kg^-1 * m^3 * s^-2 * [2*10**30 kg / M_sol] * [3.24*10**-23 Mpc/m]^3
Msol = 2*10**30  # kg
G = G0 * Msol * m_to_Mpc**3  # Mpc^3 * Msol^-1 * s^-2
G1 = G0 * Msol * km_to_Mpc * (10**-3)**3
print(G1)
print(1./233.)
# G = 1/233.  # pc * Msol^-1 * (km / s)^2
H0 = 70  # km/s/Mpc
print(G)

# system
angle_conv = 180  # basically, add this to any angle I use, bc I want to measure them from my -x axis rather than my +x
Mbh = 4*10**8  # M_sol
inc = np.deg2rad(60.)  # degrees
dist = 17  # Mpc
v_sys = -H0 * dist  # 0.  # H0 * dist
print(v_sys)
theta = np.deg2rad(-200.)  # angle measured from positive (i.e. redshifted) disk axis, counterclockwise to +y_obs axis
# negative to go counterclockwise

# AT EACH POINT: 3rd dimension using wavelength (velocity!) steps (step size changeable! also maybe total length adjustable) as per Barth 1332 paper
# ** NOTE: Barth+2016, S4.3 (pg 8, RHS): 20 km/s velocity spacing?
# ** NOTE: Barth+2016, S4.4 (pg 9, LHS): 52 frequency channels

# make a gaussian centered on velocity of that point, width set by intrinsic vel disp (free parameter)
# look for 1332 cube, sum along wavelengths to make image
# ** NOTE: found in email! When open fits in ds9, use cube window that opens up to cycle through freqs to see cube data!

# (install pyraf!) (needs 2.7 environment!)
# ** NOTE: ALREADY INSTALLED IN iraf27 environ

# Task (in pyraf): lucy (stdas) (stsdas?)
# ** NOTE: need to install something separately?

# take collapsed cube, deconvolve with beam (also in Aaron's paper)
# going from 1 pix to 10 subsampled: assume equal flux in each --> height of gaussian --> now have all params required to describe profile
# lucy does the whole beam convolution thing
# convolving thing is how we get height of Gaussian
# above eqn will give intrinsic velocity gaussian, but we don't measure intrinsic (hence why we do convolution thing)
'''
# prepare to set up z axis (velocity axis) for data cube:
# should it look something like this??
# z_ax = np.linspace(-100., 100., 11)  # where 11 = 1 + (100 - (-100))/20; creates 10 channels, each w/ width 20 km/s

n_channels = 52
spacing = 20  # km/s
z_ax = np.linspace(-n_channels * spacing / 2, n_channels * spacing / 2, n_channels + 1)

gaussian: = A * exp(-(v-vrot)**2 / (2*sigma^2))

sig: const, const+sig1*e^(-r/r0), const+gaussian
'''
# SETTING UP z AXIS
n_channels = 134  # 52  # from Barth+2016
spacing = 20  # km/s
z_ax = np.linspace(v_sys - n_channels * spacing / 2, v_sys + n_channels * spacing / 2, n_channels + 1)

# grid setup
'''
Let y be along the major axis, x be along the minor axis.
'''
s = 10  # oversampling factor
x_width = 0.975  # arcsec
y_width = 2.375  # arcsec
resolution = 0.05  # arcsec
# Number of points resolved N = disk_radius / resolution (+1 because of linspace)
print(.975*10/.05)  # want 1->475, 1->195 (centered [zeroed] at 238, 98
x_obs1 = np.linspace(-x_width/2., x_width/2., int((x_width / resolution)*s + 1))  # units of arcsec
y_obs1 = np.linspace(-y_width/2., y_width/2., int((y_width / resolution)*s + 1))  # units of arcsec
# x_obs = np.linspace(0., x_width, int((x_width / resolution) * s + 1))  # units of arcsec
# y_obs = np.linspace(0., y_width, int((y_width / resolution) * s + 1))  # units of arcsec
x_step = 0.05
y_step = 0.05
x_obs = [0.] * int(x_width*s/resolution)
y_obs = [0.] * int(y_width*s/resolution)
x_ctr = (len(x_obs) + 1) / 2  # +1 bc python starts counting at 0, and len is odd
y_ctr = (len(y_obs) + 1) / 2  # +1 bc python starts counting at 0, and len is odd
for i in range(len(x_obs)):
    x_obs[i] = x_step * ((i+1) - x_ctr) / s
for i in range(len(y_obs)):
    y_obs[i] = y_step * ((i+1) - y_ctr) / s
# print(x_obs)  # YES!
# print(x_obs1)
# print(len(x_obs))  # YES!
x_bhctr = x_off * x_step / s
y_bhctr = y_off * y_step / s

print(x_width*s/resolution, 'is this float a round integer? hopefully!')

# my x,y are both -2?
# each pixel is step 0.05071 / 10

# x_obs = np.asarray([(x / np.cos(inc))**2 for x in x_disk])
# y_obs = y_disk

# convert from arcsec to physical units (Mpc)
# tan(theta) = x/d where d=dist and theta=disk_radius --> x = d*tan(theta)
x_bhctr = dist * np.tan(x_bhctr / 206265)
y_bhctr = dist * np.tan(y_bhctr / 206265)
print(x_bhctr, y_bhctr)

x_obs1 = np.asarray([dist * np.tan(x / 206265) for x in x_obs1])  # 206265 arcsec/rad
y_obs1 = np.asarray([dist * np.tan(y / 206265) for y in y_obs1])  # 206265 arcsec/rad
x_obs = np.asarray([dist*np.tan(x / 206265) for x in x_obs])  # 206265 arcsec/rad
y_obs = np.asarray([dist*np.tan(y / 206265) for y in y_obs])  # 206265 arcsec/rad
print(x_obs1)
print(x_obs)
print(len(x_obs))
# print(oops)  # stop it here for a sec

# at each x,y spot in grid, calculate what x_disk, y_disk are, then calculate R, v, etc.
R = np.zeros(shape=(len(x_obs), len(y_obs)))
vel = np.zeros(shape=(len(x_obs), len(y_obs)))
v_los = np.zeros(shape=(len(x_obs), len(y_obs)))
v_obs = np.zeros(shape=(len(x_obs), len(y_obs)))
v_obs3d = np.zeros(shape=(len(z_ax), len(x_obs), len(y_obs)))
for x in range(len(x_obs)):
    for y in range(len(y_obs)):
        x_disk = (x_obs[x] - x_bhctr)*np.cos(theta) - (y_obs[y] - y_bhctr)*np.sin(theta)
        y_disk = (y_obs[y] - y_bhctr)*np.cos(theta) + (x_obs[x] - x_bhctr)*np.sin(theta)

        # velocity depends on radius R of point in the disk
        R[x, y] = np.sqrt((x_disk**2 / np.cos(inc)**2) + y_disk**2)

        # Keplerian velocity of any point (x, y) in the disk (i.e. with radius R(x, y))
        vel[x, y] = np.sqrt(G * Mbh / R[x, y]) * 3.086*10**19  # Mpc/s to km/s

        # Convert Keplerian velocity to LOS velocity
        alpha = np.arctan((y_disk*np.cos(inc)) / x_disk)  # alpha correct?
        # alpha measured from +x (minor axis) toward +y (major axis)
        # v_los[x, y] = vel[x, y] * np.sin(alpha) * np.sin(inc)  # - sign because blueshifts are negative (convention)
        # -vel * cos(a) * sin(inc) OR +vel * sin(a) * sin(inc) ?!

        # calculate keplerian velocity
        v_los[x, y] = np.sqrt(G * Mbh) * 3.086*10**19 * np.sin(inc) * y_disk /\
            ((x_disk / np.cos(inc))**2 + y_disk**2)**(3/4)
        # if x_obs[x] == 0. and y_obs[y] == 0:  # don't let center blow up!
        if x_obs[x] == x_bhctr and y_obs[y] == y_bhctr:
            v_los[x, y] = 0.

        # calculate observed velocity
        v_obs[x, y] = v_sys - v_los[x, y]

        '''
        # CALCULATE Z AXIS (aka velocity axis!)
        for z in range(len(z_ax)):
            A = 1. / 10.  # NOTE: can interpret thiss via flux, or surface brightness, or (etc.)
            sigma = 50.*(1+np.exp(-R[x, y]))  # sig: const, c1+c2*e^(-r/r0), const+gaussian
            # sigma = 50.*(1+np.exp(-R[x, y]))  # sig: const, c1+c2*e^(-r/r0), const+gaussian  # didn't change much
            v_obs3d[z, x, y] = A * np.exp(-(z_ax[z] - v_obs[x, y])**2 / (2 * sigma**2))
            if x == 97 and y == 238:  # corresponds to pixels 98 & 240, which are max
                print(v_obs3d[z, x, y], x, y, z_ax[z], v_obs[x, y])
        '''
        # NOTE: try printing out gaussian velocity spectrum at each x,y (or some x,y)

# print(v_los)
print(v_obs)
# print(len(v_obs))  # 196
# print(len(v_obs[0]), len(v_obs[:,0]), len(v_obs[0,:]), len(x_obs), len(y_obs))  # 476 196 476 196 476

# v_obs = v_sys - v_los = v_sys - v(R)*sin(i)*cos(alpha)
#       = v_sys - sqrt(GM/R)*sin(i)*cos(alpha)
#       = v_sys - sqrt(GM)*sin(i)*cos(alpha)/[(x_obs / cos(i))**2 + y_obs**2]^(1/4)
#       = v_sys - sqrt(GM)*sin(i)*[1 / sqrt((x_obs / (y_obs*cos(i)))**2 + 1)]/[(x_obs / cos(i))**2 + y_obs**2]^(1/4)
# NOTE: 1/sqrt((x_obs / (y_obs*cos(i)))**2 + 1) = y_obs/sqrt((x_obs/cos(i))**2 + y_obs**2) = y_obs / sqrt(R)
#       = v_sys - sqrt(GM)*sin(i)*y_obs / [(x_obs / cos(i))**2 + y_obs**2]^(3/4)

# n = np.zeros(shape=(10, 10))
hdu = fits.PrimaryHDU(v_obs3d)
hdul = fits.HDUList([hdu])
# hdul.append(fits.PrimaryHDU())
# for img in export_array:
#    hdul.append(fits.ImageHDU(data=img))
hdul.writeto(outname)

fig = plt.figure(figsize=(12, 10))
for x in range(len(x_obs)):
    xtemp = [x_obs[x]] * len(y_obs)
    plt.scatter(xtemp, y_obs, c=v_los[x, :], vmin=-10**3., vmax=10**3., s=25, cmap='RdBu_r')  # viridis, RdBu_r, inferno
print(theta)
# plt.plot([0., 1.*np.cos(np.deg2rad(theta))], [0., 1.*np.sin(np.deg2rad(theta))], color='k', lw=3)
plt.colorbar()
plt.xlabel(r'x [Mpc]', fontsize=30)
plt.ylabel(r'y [Mpc]', fontsize=30)
plt.xlim(min(x_obs), max(x_obs))
plt.ylim(min(y_obs), max(y_obs))
print('hey')
plt.show()

'''
for x in range(len(x_obs)):
    for y in range(len(y_obs)):
        print(x,y)
        plt.scatter(x_obs[x], y_obs[y], c=v_obs[x, y], vmin=-50., vmax=50., s=25, cmap='Greys')
plt.colorbar()
print('hey')
plt.show()
'''
# t = Table(v_obs)
# t.write('new_table.fits')

