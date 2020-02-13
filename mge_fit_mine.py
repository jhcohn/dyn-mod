#!/usr/bin/env python

"""
    This script obtains an MGE fit from a galaxy image using the mge_fit_sectors package, based on mge_fit_example.py

"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from os import path
import scipy.optimize as opt

from find_galaxy import find_galaxy
from mge_fit_1d import mge_fit_1d
from sectors_photometry import sectors_photometry
from mge_fit_sectors import mge_fit_sectors
from mge_print_contours import mge_print_contours
from mge_fit_sectors_twist import mge_fit_sectors_twist
from sectors_photometry_twist import sectors_photometry_twist
from mge_print_contours_twist import mge_print_contours_twist


def fit_psf(psf_file, plotem=False):
    """
    This function fits an mge to a PSF

    We model the PSF of our HST/WFC3/F160W image of UGC 2698.
    :return: the mge parameters fit to the PSF
    """
    hdu = fits.open(psf_file)
    psfimg = hdu[0].data
    hdu.close()

    f_p = find_galaxy(psfimg, binning=1, fraction=0.02, level=None, nblob=1, plot=plotem, quiet=False)
    if plotem:
        plt.show()
        plt.clf()
    psf_ang1 = f_p.theta  # theta = 151.3, xctr = 51.37, yctr = 50.44, eps = 0.031, major axis = 28.9 pix
    psf_xc1 = f_p.xmed
    psf_yc1 = f_p.ymed
    psf_eps = f_p.eps

    # sectors photometry first! Specify just want 1 sector
    sp = sectors_photometry(psfimg, psf_eps, psf_ang1, psf_xc1, psf_yc1, n_sectors=1, minlevel=0, plot=plotem)
    if plotem:
        plt.show()  # Allow plot to appear on the screen
        plt.clf()

    m1d = mge_fit_1d(sp.radius, sp.counts, plot=plotem)
    if plotem:
        plt.show()  # Allow plot to appear on the screen
        plt.clf()

    return m1d


def fit_u2698(input_image, input_mask, psf_file, pfrac=0.01, num=None, write_out=None, plots=False, qlims=None,
              cps=True, sky=0.37):
    """
    This function fits an mge to UGC 2698

    Mask the object before MGE fitting.

    We model an HST/WFC3/F160W image of UGC 2698.

    cps: if input_image is in units of counts per second, set cps=True; else if in counts, set cps=False
    """

    scale1 = 0.1     # (arcsec/pixel) PC1. This is used as scale and flux reference!

    hdu = fits.open(input_image)
    img1 = hdu[0].data
    img1 -= sky  # subtract sky  # NOTE: ALSO SAVED NEW VERSIONS OF IMGS WITH SKY SUBTRACTED FOR USE IN GALFIT
    # NOTE: both ahcorr and regH images have the same flux values
    hdu.close()

    hdu = fits.open(input_mask)
    maskimg = hdu[0].data  # Must be Boolean with False=masked in sectors_photometry(). Image is 1+ = Masked.
    hdu.close()

    maskimg[maskimg == 0] = -1
    maskimg[maskimg > 0] = False
    maskimg[maskimg < 0] = True
    maskimg = maskimg.astype(bool)

    test_img1 = img1 * maskimg

    # The geometric parameters below were obtained using my FIND_GALAXY program
    f = find_galaxy(img1, binning=1, fraction=pfrac, level=None, nblob=1, plot=plots, quiet=False)
    if plots:
        plt.show()
        plt.clf()
    ang1 = f.theta
    xc1 = f.xmed
    yc1 = f.ymed
    eps = f.eps
    print(xc1, yc1, ang1, eps)

    s1 = sectors_photometry(img1, eps, ang1, xc1, yc1, minlevel=0, mask=maskimg, plot=plots)  # sky subtr, so minlevel=0
    if plots:
        plt.show()  # Allow plot to appear on the screen
        plt.clf()

    radius = s1.radius
    angle = s1.angle
    counts = s1.counts

    # PSF for the WFC3/F160W image (we use a Gaussian PSF for simplicity)
    m_psf = fit_psf(psf_file)
    sigma_psf = m_psf.sol[1]
    norm_psf = m_psf.sol[0] / np.sum(m_psf.sol[0])
    print(norm_psf, np.sum(norm_psf))

    # Do the actual MGE fit
    # *********************** IMPORTANT ***********************************
    # For the final publication-quality MGE fit one should include the line:
    # from mge_fit_sectors_regularized import mge_fit_sectors_regularized as mge_fit_sectors
    # at the top of this file and re-run the procedure.
    # See the documentation of mge_fit_sectors_regularized for details.
    # *********************************************************************
    if num is None:
        m = mge_fit_sectors(radius, angle, counts, eps, sigmapsf=sigma_psf, normpsf=norm_psf, scale=scale1, plot=plots,
                            linear=True, qbounds=qlims, ngauss=num)
    else:
        m = mge_fit_sectors(radius, angle, counts, eps, sigmapsf=sigma_psf, normpsf=norm_psf, scale=scale1, plot=plots,
                            linear=False, qbounds=qlims, ngauss=num)
    print(m.sol)
    if plots:
        plt.show()  # Allow plot to appear on the screen
        plt.clf()

    # Plot MGE contours of the HST image
    mge_print_contours(img1, ang1, xc1, yc1, m.sol, scale=scale1, binning=1, sigmapsf=sigma_psf, normpsf=norm_psf)
    # magrange=9
    plt.show()

    if write_out is not None:
        outname = write_out
        with open(outname, 'w+') as o:
            o.write('# UGC 2698 MGE using mge_fit_mine.py\n')
            o.write('# ang = find_galaxy.theta: "Position angle measured clock-wise from the image X axis"\n')
            cs = 'Counts'
            if cps:
                cs = 'Counts_per_sec'
            o.write('# ' + cs + ' Sigma_pix qObs xc yc ang\n')
            for j in range(len(m.sol[0])):
                o.write(str(m.sol[0][j]) + ' ' + str(m.sol[1][j]) + ' ' + str(m.sol[2][j]) + ' ' + str(xc1) + ' ' +
                        str(yc1) + ' ' + str(ang1) + '\n')
        print('written!')


def display_mod(galfit_out=None):
    """
    This function displays the model mge of UGC 2698, from fit_ugc2698 -> GALFIT -> out_galfit.py -> input here!

    """
    base = '/Users/jonathancohn/Documents/dyn_mod/galfit_u2698/'

    mags = []
    fwhms = []
    qs = []
    with open(base + galfit_out, 'r') as go:
        for line in go:
            cols = line.split()
            if line.startswith('A)'):
                file = cols[1]
            elif line.startswith('D)'):
                psf_file = cols[1]
            elif line.startswith('F)'):
                mask = cols[1]
            elif line.startswith('J)'):
                zp = float(cols[1])
            elif line.startswith(' 1)'):
                yc1 = float(cols[1])
                xc1 = float(cols[2])
            elif line.startswith(' 3)'):
                mags.append(float(cols[1]))  # integrated mags
            elif line.startswith(' 4)'):
                fwhms.append(float(cols[1]))  # fwhm_pix
            elif line.startswith(' 9)'):
                qs.append(float(cols[1]))  # qObs

    scale1 = 0.1  # (arcsec/pixel) PC1. This is used as scale and flux reference!

    norm = True
    if 'ahcorr' in file:
        norm = False
        print(file)

    hdu = fits.open(file)
    img1 = hdu[0].data
    hdrn = hdu[0].header
    img1 -= 0.37  # subtract sky
    hdu.close()

    hdu = fits.open(mask)
    maskimg = hdu[0].data  # Must be Boolean with False=masked in sectors_photometry(). Image is 1+ = Masked.
    hdu.close()

    maskimg[maskimg == 0] = -1
    maskimg[maskimg > 0] = False
    maskimg[maskimg < 0] = True
    maskimg = maskimg.astype(bool)

    test_img = img1 * maskimg

    # The geometric parameters below were obtained using my FIND_GALAXY program
    f = find_galaxy(img1, binning=1, fraction=0.1, level=None, nblob=1, plot=True, quiet=False)
    plt.show()
    '''
    # USING AHCORR (DUST-CORRECTED H-BAND), REGULAR MASK (and binning=1, fraction=0.1)
    /Users/jonathancohn/Documents/dyn_mod/galfit_u2698/ugc2698_f160w_pxfr075_pxs010_ahcorr_rapidnuc_sci_nonan.fits
    Pixels used: 195802
    Peak Img[j, k]: 491 880
    Mean (j, k): 489.99 879.95
    Theta (deg): 96.3
    Astro PA (deg): 173.7
    Eps: 0.284
    Major axis (pix): 324.5
 
    # USING REGULAR H-BAND, PLUS DUST MASK (and binning=1, fraction = 0.1)
    Pixels used: 195770
    Peak Img[j, k]: 491 880
    Mean (j, k): 489.92 880.01
    Theta (deg): 96.3
    Astro PA (deg): 173.7
    Eps: 0.284
    Major axis (pix): 324.6
    '''
    print(oop)
    ang1 = 96.3
    xc1 = 491.1538  # 489.92
    yc1 = 880.6390  # 880.01
    # xc1 = 491.1538  # 489.92
    # yc1 = 880.6390  # 880.01
    print(xc1, yc1, 'look!')
    eps = 0.284
    sma = 324.6
    # astro PA = 173.7

    # '''  #
    plt.clf()
    s1 = sectors_photometry(img1, eps, ang1, xc1, yc1, minlevel=0, mask=maskimg, plot=0)
    # plt.show()  # Allow plot to appear on the screen

    radius = s1.radius
    angle = s1.angle
    counts = s1.counts
    # '''  #

    # The PSF needs to be the one for the high-resolution image used in the centre.
    # Here this is the WFC3/F160W image (we use a Gaussian PSF for simplicity)
    m_psf = fit_psf(psf_file)
    sigma_psf = m_psf.sol[1]
    norm_psf = m_psf.sol[0] / np.sum(m_psf.sol[0])

    # From FIND GALAXY (with PSF)
    # Pixels used: 999
    # Peak Img[j, k]: 51 50
    # Mean (j, k): 51.37 50.44
    # Theta (deg): 151.3
    # Astro PA (deg): 118.7
    # Eps: 0.031
    # Major axis (pix): 28.9

    '''  #
    plt.clf()
    linear = False
    if num is None:
        linear = True
        m = mge_fit_sectors(radius, angle, counts, eps, sigmapsf=sigma_psf, normpsf=norm_psf, scale=scale1, plot=1,
                            linear=linear, qbounds=qlims, ngauss=num)  # ngauss=ngauss,
    else:
        m = mge_fit_sectors(radius, angle, counts, eps, sigmapsf=sigma_psf, normpsf=norm_psf, scale=scale1, plot=1,
                            linear=linear, qbounds=qlims, ngauss=num)  # ngauss=ngauss,
    print(m.sol)
    plt.show()  # Allow plot to appear on the screen
    # '''  #

    sols = np.zeros(shape=(3,len(qs)))  # axes: total_counts, sigma_pix, qObs
    for i in range(len(mags)):
        sols[0][i] = 10 ** (0.4 * (zp - mags[i]))  # counts
        sols[1][i] = fwhms[i] / 2.355  # https://en.wikipedia.org/wiki/Full_width_at_half_maximum
        sols[2][i] = qs[i]
    print(sols)

    # Plot MGE contours of the HST image
    plt.clf()
    if norm:
        mge_print_contours(img1 / 898.467164, ang1, xc1, yc1, sols, scale=scale1, binning=4, sigmapsf=sigma_psf,
                           normpsf=norm_psf)
    else:
        mge_print_contours(img1, ang1, xc1, yc1, sols, scale=scale1, binning=4, sigmapsf=sigma_psf, normpsf=norm_psf)

    plt.show()


if __name__ == '__main__':

    print("\nFitting UGC 2698-----------------------------------\n")

    base = '/Users/jonathancohn/Documents/dyn_mod/galfit_u2698/'
    ahcorr_cps = base + 'ugc2698_f160w_pxfr075_pxs010_ahcorr_rapidnuc_sci_nonan_cps.fits'
    ahcorr_counts = base + 'ugc2698_f160w_pxfr075_pxs010_ahcorr_rapidnuc_sci_nonan_counts.fits'
    reg_mask = base + 'f160w_maskedgemask_px010.fits'
    regH_img = base + 'ugc2698_f160w_pxfr075_pxs010_drz_rapidnuc_sci_nonan_cps.fits'
    regH_counts = base + 'ugc2698_f160w_pxfr075_pxs010_drz_rapidnuc_sci_nonan_counts.fits'
    comb_mask = base + 'f160w_combinedmask_px010.fits'
    psf_file = base + 'ugc2698_f160w_pxfr075_pxs010_rapid_psf_drz_sci_clipped2no0.fits'

    out_ah5 = base + 'mge_ugc_2698_ahcorr_n10_05_pf001_cps.txt'
    out_ah6 = base + 'mge_ugc_2698_ahcorr_n10_06_pf001_cps.txt'  # None  # 'ugc_2698_ahcorr_n11_mge_04.txt'
    out_ah4 = base + 'ugc_2698_ahcorr_n9_mge_04_psff002.txt'
    out_rh6 = base + 'ugc_2698_regH_n9_mge_06_psff002.txt'
    out_rh4 = base + 'ugc_2698_regH_n9_mge_04_psff002.txt'
    out_ah = base + 'mge_ugc_2698_ahcorr_n10_pf001_cps.txt'
    out_rh = base + 'mge_ugc_2698_regH_n10_pf001_cps.txt'
    out_ahcounts = base + 'mge_ugc_2698_ahcorr_linear_pf001_counts.txt'  # mge_ugc_2698_ahcorr_n10_pf001_counts
    out_rhcounts = base + 'mge_ugc_2698_regH_linear_pf001_counts.txt'
    out_rhcps = base + 'mge_ugc_2698_regH_linear_pf001_cps.txt'

    sky_counts = 339.493665331
    rms_counts = 21.5123034564
    sky_cps = 0.37709936
    rms_cps = 0.0239433389818
    num = None  # 10
    ql6 = [0.6, 1.]
    ql5 = [0.5, 1.]
    ql4 = [0.4, 1.]
    pf = 0.01

    fit_u2698(regH_counts, comb_mask, psf_file, pfrac=pf, num=num, write_out=out_rhcounts, qlims=None, plots=True,
              cps=False, sky=sky_counts)
    print(oop)
    fit_u2698(regH_img, comb_mask, psf_file, pfrac=pf, num=num, write_out=out_rhcps, qlims=None, plots=True,
              cps=True, sky=sky_cps)
    print(oop)
    fit_u2698(ahcorr_counts, reg_mask, psf_file, pfrac=pf, num=num, write_out=out_ahcounts, qlims=None, plots=True,
              cps=False, sky=sky_counts)  # 338.8114
    print(oop)
    fit_u2698(ahcorr_img, reg_mask, psf_file, pfrac=pf, num=num, write_out=out_ah5, qlims=ql5, plots=True)
    fit_u2698(regH_img, comb_mask, psf_file, pfrac=pf, num=num, write_out=out_rh, qlims=None, plots=True)
    print(oop)
    fit_u2698(ahcorr_img, reg_mask, psf_file, pfrac=pf, num=num, write_out=out_ah, qlims=None, plots=True)
    fit_u2698(regH_img, comb_mask, psf_file, pfrac=pf, num=num, write_out=out_rh, qlims=None, plots=True)
    print(oop)
    fit_u2698(ahcorr_img, reg_mask, psf_file, pfrac=0.02, num=num, write_out=None, qlims=ql5, plots=False)
    fit_u2698(ahcorr_img, reg_mask, psf_file, pfrac=0.03, num=num, write_out=None, qlims=ql5, plots=False)
    fit_u2698(ahcorr_img, reg_mask, psf_file, pfrac=0.04, num=num, write_out=None, qlims=ql5, plots=False)
    fit_u2698(ahcorr_img, reg_mask, psf_file, pfrac=0.05, num=num, write_out=None, qlims=ql5, plots=False)
    print(oop)
    fit_u2698(ahcorr_img, reg_mask, psf_file, num=num, write_out=out_ah4, qlims=ql4, plots=True)
    fit_u2698(ahcorr_img, reg_mask, psf_file, num=num, write_out=out_ah6, qlims=ql6, plots=True)
    fit_u2698(regH_img, comb_mask, psf_file, num=num, write_out=out_rh4, qlims=ql4, plots=True)
    fit_u2698(regH_img, comb_mask, psf_file, num=num, write_out=out_rh6, qlims=ql6, plots=True)
    print(oop)
    '''  #
    IMG1
    fit_u2698(ahcorr_img, reg_mask, psf_file, pfrac=0.05, num=num, write_out=None, qlims=ql5, plots=False)
    # 491.4505251361511, 879.66976803463479 (a bit shifted; probably should be ~880-881)
    fit_u2698(ahcorr_img, reg_mask, psf_file, pfrac=0.03, num=num, write_out=None, qlims=ql5, plots=False)
    # 491.36437904527077, 879.9134055002786 (a bit shifted, but better than above)
    fit_u2698(ahcorr_img, reg_mask, psf_file, pfrac=0.01, num=num, write_out=None, qlims=ql5, plots=False)
    # 491.36437904527077, 879.9134055002786 (a bit shifted,same as above)
    fit_u2698(ahcorr_img, reg_mask, psf_file, pfrac=0.0075, num=num, write_out=None, qlims=ql5, plots=False)
    # 491.36437904527077, 879.9134055002786 (same as above)
    fit_u2698(ahcorr_img, reg_mask, psf_file, pfrac=0.005, num=num, write_out=None, qlims=ql5, plots=False)
    # 491.4431968696681, 879.9734944374386 (a bit better)
    fit_u2698(ahcorr_img, reg_mask, psf_file, pfrac=0.0025, num=num, write_out=None, qlims=ql5, plots=False)
    # 491.44627677000807, 879.95515611820576 (a tiny bit worse)
    fit_u2698(ahcorr_img, reg_mask, psf_file, pfrac=0.002, num=num, write_out=None, qlims=ql5, plots=False)
    # 491.45790454942824, 879.95950711328237
    fit_u2698(ahcorr_img, reg_mask, psf_file, pfrac=0.0015, num=num, write_out=None, qlims=ql5, plots=False)
    # 491.48646540287677, 879.99319790554887
    fit_u2698(ahcorr_img, reg_mask, psf_file, pfrac=0.001, num=num, write_out=None, qlims=ql5, plots=False)
    # 811.45635993532062, 1179.050969939296 HORRIBLE!
    
    fit_u2698(ahcorr_img, reg_mask, psf_file, pfrac=0.01, num=num, write_out=None, qlims=ql5, plots=False)
    #491.355681831 879.929309641
    fit_u2698(ahcorr_img, reg_mask, psf_file, pfrac=0.02, num=num, write_out=None, qlims=ql5, plots=False)
    #491.382178879 879.988405578
    fit_u2698(ahcorr_img, reg_mask, psf_file, pfrac=0.03, num=num, write_out=None, qlims=ql5, plots=False)
    #491.364379045 879.9134055
    fit_u2698(ahcorr_img, reg_mask, psf_file, pfrac=0.04, num=num, write_out=None, qlims=ql5, plots=False)
    #491.429928147 879.648566635
    fit_u2698(ahcorr_img, reg_mask, psf_file, pfrac=0.05, num=num, write_out=None, qlims=ql5, plots=False)
    #491.450525136 879.669768035
    fit_u2698(ahcorr_img, reg_mask, psf_file, pfrac=0.06, num=num, write_out=None, qlims=ql5, plots=False)
    #491.540395065 879.730832415
    fit_u2698(ahcorr_img, reg_mask, psf_file, pfrac=0.07, num=num, write_out=None, qlims=ql5, plots=False)
    #491.497614815 879.821165695
    fit_u2698(ahcorr_img, reg_mask, psf_file, pfrac=0.08, num=num, write_out=None, qlims=ql5, plots=False)
    #683.090132296 1058.50026125
    fit_u2698(ahcorr_img, reg_mask, psf_file, pfrac=0.09, num=num, write_out=None, qlims=ql5, plots=False)
    #681.794088312 1057.28524263
    
    TESTIMG
    fit_u2698(ahcorr_img, reg_mask, psf_file, pfrac=0.001, num=num, write_out=None, qlims=ql5, plots=False)
    #491.4500028 879.952142988
    fit_u2698(ahcorr_img, reg_mask, psf_file, pfrac=0.002, num=num, write_out=None, qlims=ql5, plots=False)
    #491.437416947 879.972154554
    fit_u2698(ahcorr_img, reg_mask, psf_file, pfrac=0.003, num=num, write_out=None, qlims=ql5, plots=False)
    #491.497414716 879.932725165
    fit_u2698(ahcorr_img, reg_mask, psf_file, pfrac=0.004, num=num, write_out=None, qlims=ql5, plots=False)
    #491.489475065 879.961478177
    fit_u2698(ahcorr_img, reg_mask, psf_file, pfrac=0.005, num=num, write_out=None, qlims=ql5, plots=False)
    #491.496320619 879.953463891
    fit_u2698(ahcorr_img, reg_mask, psf_file, pfrac=0.006, num=num, write_out=None, qlims=ql5, plots=False)
    #491.391923676 879.931646223
    fit_u2698(ahcorr_img, reg_mask, psf_file, pfrac=0.007, num=num, write_out=None, qlims=ql5, plots=False)
    #491.418584102 879.932159354
    fit_u2698(ahcorr_img, reg_mask, psf_file, pfrac=0.008, num=num, write_out=None, qlims=ql5, plots=False)
    #491.545252285 879.951583076
    fit_u2698(ahcorr_img, reg_mask, psf_file, pfrac=0.009, num=num, write_out=None, qlims=ql5, plots=False)
    #491.607532648 879.883632699
    # '''  #


    '''  #
    fit_u2698(ahcorr_img, reg_mask, psf_file, pfrac=0.01, num=num, write_out=None, qlims=ql6, plots=True)
    fit_u2698(ahcorr_img, reg_mask, psf_file, pfrac=0.02, num=num, write_out=None, qlims=ql6, plots=True)
    fit_u2698(ahcorr_img, reg_mask, psf_file, pfrac=0.03, num=num, write_out=None, qlims=ql6, plots=True)
    fit_u2698(ahcorr_img, reg_mask, psf_file, pfrac=0.04, num=num, write_out=None, qlims=ql6, plots=True)
    print(oop)
    # '''  #

    display_mod(galfit_out='galfit.72')
    # IMPORTANT NOTE: m_Vega zeropoint for F160W = m_AB - 1.39 = 25.95 - 1.39 = 24.56 (F160W~H?)
    # http://www.astronomy.ohio-state.edu/~martini/usefuldata.html (conversion AB-Vega)
    # https://hst-docs.stsci.edu/display/WFC3IHB/9.3+Calculating+Sensitivities+from+Tabulated+Data (F160W AB zeropoint)
    # https://github.tamu.edu/joncohn/gas-dynamical-modeling/wiki/Week-of-2019-12-09 (did this on this page)
    # regH: 'ugc2698_f160w_pxfr075_pxs010_drz_rapidnuc_sci_no0.fits'
    # dust mask: 'f160w_combinedmask_px010.fits'
    # ahcorr: 'ugc2698_f160w_pxfr075_pxs010_ahcorr_rapidnuc_sci_nonan.fits'
    # reg mask: 'f160w_maskedgemask_px010.fits'
    # 'galfit.73': regH, dust mask, q>=0.6 (norm=True)
    # 'galfit.75': regH, dust mask, q>=0.4 (norm=True)
    # 'galfit.77': ahcorr, reg mask, q>=0.4 (norm=False)
    # 'galfit.72': ahcorr, reg mask, q>=0.55 (norm=False)
    print(oop)

    fit_u2698(img, mask, write_out=outname, num=num, qlims=ql)

# NOTE: GOT 0.05 ERROR WHEN I USED ZP J) 25.95, BUT NOT WHEN I USED ZP J) 24.697
