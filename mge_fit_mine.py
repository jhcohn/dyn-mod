#!/usr/bin/env python

"""
    This script obtains an MGE fit from a galaxy image using the mge_fit_sectors package, based on mge_fit_example.py

"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from os import path

from find_galaxy import find_galaxy
from mge_fit_1d import mge_fit_1d
from sectors_photometry import sectors_photometry
from mge_fit_sectors import mge_fit_sectors
from mge_print_contours import mge_print_contours
from mge_fit_sectors_twist import mge_fit_sectors_twist
from sectors_photometry_twist import sectors_photometry_twist
from mge_print_contours_twist import mge_print_contours_twist


def fwhm_est(arr):
    amp = np.amax(arr)
    xctr,yctr = np.where(arr == amp)
    xctr = xctr[0]
    yctr = yctr[0]
    half_max = amp / 2.  # find when function crosses line half_max (when sign of diff flips)
    # take the 'derivative' of signum(half_max - Y[])
    dx = np.sign(half_max - np.array(arr[xctr, 0:-1])) - np.sign(half_max - np.array(arr[xctr, 1:]))
    dy = np.sign(half_max - np.array(arr[0:-1, yctr])) - np.sign(half_max - np.array(arr[1:, yctr]))
    # find the left and right most indexes
    print(dx)
    print(dy)
    dx_idx_l = np.where(dx > 0)[0][0]
    dx_idx_r = np.where(dx < 0)[-1][0]
    dy_idx_l = np.where(dy > 0)[0][0]
    dy_idx_r = np.where(dy < 0)[-1][0]
    return dx_idx_r - dx_idx_l, dy_idx_r - dy_idx_l  # return the differences (full width)


def gaussian_2d(x, y, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp(-(a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)))
    return g.ravel()


def gauss_function(x, a, x0, sigma):

    return a*np.exp(-(x-x0)**2/(2*sigma**2))


def fit_u2698():
    """
    This function fits an mge to UGC 2698

    Mask the object before MGE fitting.

    We model an HST/WFC3/F160W image of UGC 2698.

    """

    scale1 = 0.1     # (arcsec/pixel) PC1. This is used as scale and flux reference!

    base = '/Users/jonathancohn/Documents/dyn_mod/galfit_u2698/'
    file = base + 'ugc2698_f160w_pxfr075_pxs010_ahcorr_rapidnuc_sci_nonan.fits'  # dust-corrected UGC 2698 H-band, from Ben
    hdu = fits.open(file)
    img1 = hdu[0].data
    img1 -= 0.37  # subtract sky
    hdu.close()

    maskfile = base + 'f160w_ahcorr_mask_px010.fits'
    hdu = fits.open(maskfile)
    mask1 = hdu[0].data  # Must be Boolean with False=masked in sectors_photometry(). Image is 1=Masked, 0=Unmasked
    hdu.close()

    mask2 = base + 'f160w_combinedmask_px010.fits'
    hdu = fits.open(mask2)
    mask_comb = hdu[0].data  # Must be Boolean with False=masked in sectors_photometry(). Image is 1+ = Masked.
    hdu.close()

    maskimg = mask_comb + mask1

    maskimg[maskimg == 0] = -1
    maskimg[maskimg > 0] = False
    maskimg[maskimg < 0] = True
    maskimg = maskimg.astype(bool)

    test_img = img1 * maskimg
    # The geometric parameters below were obtained using my FIND_GALAXY program
    f = find_galaxy(test_img, binning=1, fraction=0.1, level=None, nblob=1, plot=True, quiet=False)
    plt.show()
    ang1 = 96.3 #173.8  # -8.13  # Based on Nuker profile fits in GALFIT
    xc1 = 489.81  # 491.08  # Based on Nuker profile fits in GALFIT
    yc1 = 879.91  # 880.89  # Based on Nuker profile fits in GALFIT
    eps = 0.284  # 1 - 0.73  # I've been finding q = 0.73 with the Nuker profile in GALFIT
    # above uses fraction=0.1, binning=5. Using fraction=0.05: ang1=172.0,xc1=490.21,yc1=880.40,eps=0.305
    # Using fraction=0.2: ang1=178.8,xcl=487.16,ycl=878.78,eps=0.248
    # Using fraction=0.01: ang1=178.8,xcl=491.51,ycl=879.79,eps=0.312
    # fraction=0.1,binning=3: ang1=173.8,xcl=489.82,ycl=879.88,eps=0.284
    # fraction=0.1,binning=1: ang1=96.3,xcl=489.84,ycl=879.92,eps=0.284 [Astro PA = 173.7]

    # print(oop)

    plt.clf()
    s1 = sectors_photometry(img1, eps, ang1, xc1, yc1, minlevel=0, mask=maskimg, plot=1)
    plt.show()  # Allow plot to appear on the screen

    radius = s1.radius
    angle = s1.angle
    counts = s1.counts

    # The PSF needs to be the one for the high-resolution image used in the centre.
    # Here this is the WFC3/F160W image (we use a Gaussian PSF for simplicity)
    psffile = base + 'ugc2698_f160w_pxfr075_pxs010_rapid_psf_drz_sci_clipped2no0.fits'
    hdu = fits.open(psffile)
    psfimg = hdu[0].data
    hdu.close()
    amp = np.amax(psfimg)
    xctr, yctr = np.where(psfimg == amp)
    xctr = xctr[0]
    yctr = yctr[0]

    #print(np.amax(psfimg))
    #print(np.where(psfimg==np.amax(psfimg)))
    #plt.imshow(psfimg)
    #plt.colorbar()
    #plt.show()

    #diffs = fwhm_est(psfimg)
    #print(diffs)

    import scipy.optimize as opt
    xarr = range(len(psfimg))
    yarr = range(len(psfimg[1]))
    poptx, pcovx = opt.curve_fit(gauss_function, xarr, psfimg[:, yctr], p0=[0.11658926, xctr, 1.])
    popty, pcovy = opt.curve_fit(gauss_function, yarr, psfimg[xctr, :], p0=[0.11658926, yctr, 1.])
    print(poptx, popty, amp, xctr, yctr)
    sigmapsf = np.mean([poptx[2], popty[2]])
    # fwhm = 2 * np.sqrt(2 * np.log(2)) * sig_avg
    # print(sigmapsf)  # 0.892
    ngauss = 10
    #print(radius)
    #print(angle)
    #print(counts)
    #print(oop)

    # Do the actual MGE fit
    # *********************** IMPORTANT ***********************************
    # For the final publication-quality MGE fit one should include the line:
    #
    # from mge_fit_sectors_regularized import mge_fit_sectors_regularized as mge_fit_sectors
    #
    # at the top of this file and re-run the procedure.
    # See the documentation of mge_fit_sectors_regularized for details.
    # *********************************************************************
    plt.clf()
    m = mge_fit_sectors(radius, angle, counts, eps, ngauss=ngauss, sigmapsf=sigmapsf, scale=scale1, plot=1, linear=0)
    print(m.sol)
    plt.show()  # Allow plot to appear on the screen

    # Plot MGE contours of the HST image
    plt.clf()
    mge_print_contours(img1, ang1, xc1, yc1, m.sol, scale=scale1, binning=4, sigmapsf=sigmapsf)
    # binning=4, magrange=9
    plt.show()

    write_out = False
    outname = base + 'ugc_2698_f160w_ahcorr_mge.txt'
    if write_out:
        with open(outname, 'w+') as o:
            o.write('# UGC 2698 MGE using mge_fit_mine.py\n')
            o.write('# Counts Sigma_pix qObs\n')
            for j in range(len(m.sol[0])):
                o.write(str(m.sol[0][j]) + ' ' + str(m.sol[1][j]) + ' ' + str(m.sol[2][j]) + '\n')
        print('written!')



if __name__ == '__main__':

    print("\nFitting UGC 2698-----------------------------------\n")
    fit_u2698()

'''
############################################
  Total_Counts  Sigma_Pixels      qObs
############################################
      9342.36      1.82213            1
      1999.81      2.72397            1
        12111      5.01764            1
      30723.2      7.44695     0.605945
      68232.8      12.5351     0.769006
      93288.7      25.0499     0.753742
      99019.7      54.1949     0.663729
       162588      88.9077     0.707725
       166389      174.463     0.733191
       196010      505.144     0.785429
'''
