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


def sb_prof3(smas, intens, mge_sols, qbounds=[0., 1.]):
    # Select an x and y plot range that is the same for all plots
    #
    minrad = np.min(mge_sols.radius) * mge_sols.scale
    maxrad = np.max(mge_sols.radius) * mge_sols.scale
    mincnt = np.min(mge_sols.counts)
    maxcnt = np.max(mge_sols.counts)
    xran = minrad * (maxrad / minrad) ** np.array([-0.02, +1.02])
    yran = mincnt * (maxcnt / mincnt) ** np.array([-0.05, +1.05])
    n = mge_sols.sectors.size
    dn = int(round(n / 6.))
    nrows = (n - 1) // dn + 1  # integer division
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(8, 16))
    fig.subplots_adjust(hspace=0.01)
    # fig.text(0.04, 0.5, 'counts', va='center', rotation='vertical')
    # fig.text(0.96, 0.5, 'error (%)', va='center', rotation='vertical')
    #ax[-1, 0].set_xlabel("arcsec")
    #ax[-1, 1].set_xlabel("arcsec")
    ax[1].set_xlabel('arcsec')

    ax[0].set_xlim(0.5, 400.)  # xran
    ax[0].set_ylim(2e-3, 3.*10**3)  # yran
    ax[1].set_ylim(-1, 1.)

    '''
    OPTIONAL OUTPUT KEYWORDS:
        SOL - Output keyword containing a 3xNgauss array with the
              best-fitting solution:
              1) sol[0,*] = TotalCounts, of the Gaussians components.
                  The relation TotalCounts = Height*(2*!PI*Sigma**2*qObs)
                  can be used compute the Gaussian central surface
                  brightness (Height)
              2) sol[1,*] = Sigma, is the dispersion of the best-fitting
                  Gaussians in pixels.
              3) sol[2,*] = qObs, is the observed axial ratio of the
                  best-fitting Gaussian components.
    '''
    just_sols = mge_sols.sol
    # rarray = np.logspace(-3, 3., 100)
    total = np.zeros(shape=np.array(smas).shape)
    for each_gauss in range(len(just_sols[0])):
        norm = just_sols[0, each_gauss] / (just_sols[1, each_gauss]**2 * 2*np.pi*just_sols[2, each_gauss])
        yarray = norm * np.exp(-0.5 * (smas / just_sols[1, each_gauss]) ** 2)
        total += yarray
        ax[0].loglog(smas, yarray, 'r--')
    ax[0].loglog(smas, total, 'r-', label=r'MGE Fit Sectors model', linewidth=2)

    ax[1].plot(smas, (intens - total) / intens, 'ko', markersize=4)
    ax[1].axhline(y=0., color='k', ls='--')

    #ax[row, 1].semilogx(r, mge_sols.err[w] * 100, 'C0o')
    #ax[row, 1].axhline(linestyle='--', color='C1', linewidth=2)
    #ax[row, 1].yaxis.tick_right()
    #ax[row, 1].yaxis.set_label_position("right")
    #ax[row, 1].set_ylim([-19.5, 20])

    # plt.errorbar(isolist.sma, isolist.grad, yerr=isolist.grad_r_error, fmt='--', markersize=4)
    ax[0].plot(smas, intens, 'b--', label='Data (from Ellipse task)', linewidth=4)
    #plt.xlabel('Semimajor Axis Length log(pix)')
    #plt.xscale('log')
    ax[0].set_ylabel('Mean counts along semi-major axis')
    ax[1].set_ylabel(r'Residual [(data - model) / data]',)
    ax[0].legend()
    ax[0].set_title(r'UGC 2698 surface brightness profile, qbounds=' + str(qbounds))
    plt.show()
    plt.clf()


def sb_prof2(data, xctr, yctr, a0, eps, pa, mge_sols):
    from photutils.isophote import EllipseGeometry
    geometry = EllipseGeometry(x0=xctr, y0=yctr, sma=a0, eps=eps, pa=pa * np.pi / 180.)
    from photutils import EllipticalAperture
    aper = EllipticalAperture((geometry.x0, geometry.y0), geometry.sma, geometry.sma * (1 - geometry.eps), geometry.pa)
    plt.imshow(data, origin='lower')
    aper.plot(color='w')
    plt.show()
    from photutils.isophote import Ellipse
    ellipse = Ellipse(data, geometry)
    isolist = ellipse.fit_image()
    print(isolist)
    print(isolist.grad)
    print(isolist.sma)
    print(isolist.intens)

    # Select an x and y plot range that is the same for all plots
    #
    minrad = np.min(mge_sols.radius) * mge_sols.scale
    maxrad = np.max(mge_sols.radius) * mge_sols.scale
    mincnt = np.min(mge_sols.counts)
    maxcnt = np.max(mge_sols.counts)
    xran = minrad * (maxrad / minrad) ** np.array([-0.02, +1.02])
    yran = mincnt * (maxcnt / mincnt) ** np.array([-0.05, +1.05])
    n = mge_sols.sectors.size
    dn = int(round(n / 6.))
    nrows = (n - 1) // dn + 1  # integer division
    fig, ax = plt.subplots(1, 1)
    # fig.subplots_adjust(hspace=0.01)
    fig.text(0.04, 0.5, 'counts', va='center', rotation='vertical')
    fig.text(0.96, 0.5, 'error (%)', va='center', rotation='vertical')
    #ax[-1, 0].set_xlabel("arcsec")
    #ax[-1, 1].set_xlabel("arcsec")
    ax.set_xlabel('arcsec')
    row = 0
    w = np.nonzero(mge_sols.angle == mge_sols.sectors[n-1])[0]
    w = w[np.argsort(mge_sols.radius[w])]
    r = mge_sols.radius[w] * mge_sols.scale
    txt = "$%.f^\circ$" % mge_sols.sectors[n-1]
    ax.set_xlim(xran)
    ax.set_ylim(yran)
    ax.loglog(r, mge_sols.counts[w], 'C0o')
    ax.loglog(r, mge_sols.yfit[w], 'C1', linewidth=2)
    # ax.text(0.98, 0.95, txt, ha='right', va='top', transform=ax[row, 0].transAxes)
    ax.text(0.98, 0.95, txt, ha='right', va='top', transform=ax.transAxes)
    ax.loglog(r, mge_sols.gauss[w, :] * mge_sols.weights[None, :])
    #ax[row, 1].semilogx(r, mge_sols.err[w] * 100, 'C0o')
    #ax[row, 1].axhline(linestyle='--', color='C1', linewidth=2)
    #ax[row, 1].yaxis.tick_right()
    #ax[row, 1].yaxis.set_label_position("right")
    #ax[row, 1].set_ylim([-19.5, 20])

    plt.errorbar(isolist.sma, isolist.grad, yerr=isolist.grad_r_error, fmt='--', markersize=4)
    #plt.xlabel('Semimajor Axis Length log(pix)')
    #plt.xscale('log')
    plt.ylabel('Mean intensity')
    plt.show()
    print(oop)


def sb_profile(image, out_tab, a0, x0, y0, eps0, theta0):
    # also make 1D surface brightness profiles
    # *e.g. iraf or pyraf ellipse task
    # BUCKET: run ellipse task on main galaxy (dust-corr) and on GALFIT output, then plot both
    # ellipse task should show 1D surf brightness
    import pyraf
    from pyraf import iraf
    from iraf import stsdas, analysis, isophote
    from iraf import stsdas, fitting
    #pyraf.epar(isophote.ellipse)
    isophote.ellipse(input=image, output=out_tab)#, a0=a0, x0=x0, y0=y0, eps0=eps0, teta0=theta0)
    # in terminal with: source activate iraf27: pyraf: stsdas: analysis: isophote: epar ellipse:
    # # PSET geompar: x0 879.95, y0 489.99, ellip0 0.284, pa0 96.3 (-6.3), sma0 324.5
    print(e)
    # .c1h
    print(e[0])
    print(e[1])
    print('next:')
    #f1 = fitting.gfit1d()#input=out_tab + ' a**1/4 mag', output='outgfit1.tab')
    #print(f1)
    print(oop)
    # restore.lucy(lucy_in, lucy_b, lucy_o, niter=lucy_it, maskin=lucy_mask, goodpixval=1, limchisq=1E-3)  # lucy
    # print('lucy process done in ' + str(time.time() - t_pyraf) + 's')  # ~10.6s


def fit_1d():
    """
    Usage example for mge_fit_1d().
    This example reproduces Figure 3 in Cappellari (2002)
    It takes <1s on a 2.5 GHz computer

    """
    n = 300  # number of sampled points
    x = np.logspace(np.log10(0.01), np.log10(300), n)  # logarithmically spaced radii
    y = (1 + x)**-4  # The profile must be logarithmically sampled!
    plt.clf()
    p = mge_fit_1d(x, y, ngauss=16, plot=True)
    plt.show()  # allow the plot to appear in certain situations


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


def gaussian_2d(xx, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    x = xx[0]
    y = xx[1]
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp(-(a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)))
    return g.ravel()


def gauss_function(x, a, x0, sigma):

    return a*np.exp(-(x-x0)**2/(2*sigma**2))


def fit_psf(psf_file, plotem=False):
    """
    This function fits an mge to a PSF

    Mask the object before MGE fitting.

    We model an HST/WFC3/F160W image of UGC 2698.
    :return: 
    """
    scale1 = 0.1     # (arcsec/pixel) PC1. This is used as scale and flux reference!

    hdu = fits.open(psf_file)
    psfimg = hdu[0].data
    hdu.close()

    f = find_galaxy(psfimg, binning=1, fraction=0.1, level=None, nblob=1, plot=True, quiet=False)
    if plotem:
        plt.show()
    plt.clf()
    # theta = 151.3, xctr = 51.37, yctr = 50.44, eps = 0.31, major axis = 28.9 pix
    eps = 0.31
    ang1 = 151.3
    xc1 = 51.37
    yc1 = 50.44
    sma = 28.9

    # sb_profile(a0=28.9, x0=51.37, y0=50.44, eps0=0.31, theta0=151.3)

    # sectors photometry first! Specify just want 1 sector
    plt.clf()
    s1 = sectors_photometry(psfimg, eps, ang1, xc1, yc1, n_sectors=1, minlevel=0, mask=None, plot=1)
    print('hi')
    if plotem:
        plt.show()  # Allow plot to appear on the screen

    plt.clf()
    m1d = mge_fit_1d(s1.radius, s1.counts, plot=True)
    # ngauss=ngauss,
    if plotem:
        plt.show()  # Allow plot to appear on the screen
    plt.clf()

    '''  #
    x = np.zeros(shape=psfimg.shape)
    xctr = 51.37
    yctr = 50.44
    for i in range(len(x)):
        for j in range(len(x[0])):
            x[i, j] = np.sqrt((i-xctr)**2 + (j-yctr)**2)

    x1 = np.arange(len(x))
    x2 = np.arange(len(x[0]))
    xx, yy = np.meshgrid(x1, x2)
    xs = [xx, yy]

    init_guess = [np.amax(psfimg), xctr, yctr, 0.9, 0.9, np.deg2rad(151.3), 0.]

    popt, pcov = opt.curve_fit(gaussian_2d, xs, np.ravel(psfimg), p0=init_guess)
    print(popt, pcov)
    # 0.1369, 50.4615, 51.3456, 0.94123, 0.91545, 4.48e7 (equivalent to 2.831 rad = 162.2 deg), 2.59e-5

    print(x.shape, psfimg.shape)
    print(oop)
    plt.imshow(x, origin='lower')
    plt.colorbar()
    plt.show()
    # R = np.sqrt(xx**2 + xy**2)
    # '''  #

    return m1d


def psf_2dgauss(psfimg):

    # Use base cmap to create transparent

    mycmap = plt.cm.Blues
    mycmap._init()
    mycmap._lut[:,-1] = np.linspace(0, 0.6, 259)

    # Import image and get x and y extents
    x1 = np.arange(len(psfimg))
    x2 = np.arange(len(psfimg[0]))
    xx, yy = np.meshgrid(x1, x2)

    # Plot image and overlay colormap
    fig, ax = plt.subplots(1, 1)
    ax.imshow(psfimg, origin='lower', cmap=plt.cm.Reds)
    Gauss = gaussian_2d((xx, yy), 0.1369, 50.4615, 51.2456, 0.94123, 0.91545, 4.48e7, 2.59e-5)
    # gaussian_2d(xx, amplitude, xo, yo, sigma_x, sigma_y, theta, offset)
    cb = ax.contourf(xx, yy, Gauss.reshape(xx.shape[0], yy.shape[1]), 15, cmap=mycmap)
    plt.colorbar(cb)
    plt.show()


def fit_u2698(input_image, input_mask, psf_file, num=None, write_out=None, plots=False, qlims=None):
    """
    This function fits an mge to UGC 2698

    Mask the object before MGE fitting.

    We model an HST/WFC3/F160W image of UGC 2698.

    """

    scale1 = 0.1     # (arcsec/pixel) PC1. This is used as scale and flux reference!

    hdu = fits.open(input_image)
    img1 = hdu[0].data
    hdrn = hdu[0].header
    img1 -= 0.37  # subtract sky
    hdu.close()

    hdu = fits.open(input_mask)
    maskimg = hdu[0].data  # Must be Boolean with False=masked in sectors_photometry(). Image is 1+ = Masked.
    hdu.close()

    maskimg[maskimg == 0] = -1
    maskimg[maskimg > 0] = False
    maskimg[maskimg < 0] = True
    maskimg = maskimg.astype(bool)

    test_img = img1 * maskimg

    # The geometric parameters below were obtained using my FIND_GALAXY program
    f = find_galaxy(test_img, binning=5, fraction=0.2, level=None, nblob=1, plot=plots, quiet=False)
    if plots:
        plt.show()
    plt.clf()
    ang1 = f.theta
    xc1 = f.xmed
    yc1 = f.ymed
    eps = f.eps
    sma = f.majoraxis
    # ahcorr, binning=1, fraction=0.05: 490.40 880.37, theta=98.0, eps=0.304
    # ahcorr, binning=3, fraction=0.05: 490.39 880.37, theta=97.9, eps=0.305
    # ahcorr, binning=5, fraction=0.05: 490.38 880.38, theta=98.0, eps=0.305
    # ahcorr, binning=1, fraction=0.1: 489.99 879.95, theta=96.3, eps=0.284
    # ahcorr, binning=3, fraction=0.1: 490.00 879.91, theta=96.2, eps=0.284
    # ahcorr, binning=5, fraction=0.1: 489.99 879.93, theta=96.2, eps=0.284
    # ahcorr, binning=1, fraction=0.2: 487.69 878.85, theta=91.4, eps=0.251
    # ahcorr, binning=3, fraction=0.2: 487.53 878.84, theta=91.2, eps=0.249
    # ahcorr, binning=5, fraction=0.2: 487.52 878.88, theta=91.2, eps=0.249

    # regH, binning=1, fraction=0.05: 490.34 880.47, theta=98.0, eps=0.304
    # regH, binning=3, fraction=0.05: 490.33 880.47, theta=97.9, eps=0.305
    # regH, binning=5, fraction=0.05: 490.32 880.49, theta=98.0, eps=0.305
    # regH, binning=1, fraction=0.1: 489.92 880.01, theta=96.3, eps=0.284
    # regH, binning=3, fraction=0.1: 489.93 879.97, theta=96.2, eps=0.284
    # regH, binning=5, fraction=0.1: 489.92 880.00, theta=96.2, eps=0.284
    # regH, binning=1, fraction=0.1: 487.52 878.86, theta=91.4, eps=0.251
    # regH, binning=3, fraction=0.1: 487.35 878.86, theta=91.2, eps=0.249
    # regH, binning=5, fraction=0.1: 487.34 878.90, theta=91.2, eps=0.248

    # WHERE DID THESE COME FROM -- from Nuker profile fits in GALFIT? Or?
    # xc1 = 491.1538
    # yc1 = 880.6390

    plt.clf()
    s1 = sectors_photometry(img1, eps, ang1, xc1, yc1, minlevel=0, mask=maskimg, plot=plots)
    if plots:
        plt.show()  # Allow plot to appear on the screen

    radius = s1.radius
    angle = s1.angle
    counts = s1.counts

    # The PSF needs to be the one for the high-resolution image used in the centre.
    # Here this is the WFC3/F160W image (we use a Gaussian PSF for simplicity)
    m_psf = fit_psf(psf_file)
    print(m_psf.sol)
    sigma_psf = m_psf.sol[1]
    norm_psf = m_psf.sol[0] / np.sum(m_psf.sol[0])
    print(norm_psf)
    print(np.sum(norm_psf))

    # Do the actual MGE fit
    # *********************** IMPORTANT ***********************************
    # For the final publication-quality MGE fit one should include the line:
    # from mge_fit_sectors_regularized import mge_fit_sectors_regularized as mge_fit_sectors
    # at the top of this file and re-run the procedure.
    # See the documentation of mge_fit_sectors_regularized for details.
    # *********************************************************************
    plt.clf()
    if num is None:
        m = mge_fit_sectors(radius, angle, counts, eps, sigmapsf=sigma_psf, normpsf=norm_psf, scale=scale1, plot=plots,
                            linear=True, qbounds=qlims, ngauss=num)
    else:
        m = mge_fit_sectors(radius, angle, counts, eps, sigmapsf=sigma_psf, normpsf=norm_psf, scale=scale1, plot=plots,
                            linear=False, qbounds=qlims, ngauss=num)
    print(m.sol)
    if plots:
        plt.show()  # Allow plot to appear on the screen

    # Plot MGE contours of the HST image
    plt.clf()
    mge_print_contours(img1, ang1, xc1, yc1, m.sol, scale=scale1, binning=1, sigmapsf=sigma_psf, normpsf=norm_psf)
    # magrange=9
    plt.show()

    if write_out is not None:
        outname = write_out
        with open(outname, 'w+') as o:
            o.write('# UGC 2698 MGE using mge_fit_mine.py\n')
            o.write('# Counts Sigma_pix qObs\n')
            for j in range(len(m.sol[0])):
                o.write(str(m.sol[0][j]) + ' ' + str(m.sol[1][j]) + ' ' + str(m.sol[2][j]) + '\n')
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
    f = find_galaxy(test_img, binning=1, fraction=0.1, level=None, nblob=1, plot=True, quiet=False)
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

    # dust-corrected H-band, without dust-mask (10 gaussians)
    base = '/Users/jonathancohn/Documents/dyn_mod/galfit_u2698/'
    ahcorr_img = base + 'ugc2698_f160w_pxfr075_pxs010_ahcorr_rapidnuc_sci_nonan.fits'
    reg_mask = base + 'f160w_maskedgemask_px010.fits'
    regH_img = base + 'ugc2698_f160w_pxfr075_pxs010_drz_rapidnuc_sci_no0.fits'
    comb_mask = base + 'f160w_combinedmask_px010.fits'
    psf_file = base + 'ugc2698_f160w_pxfr075_pxs010_rapid_psf_drz_sci_clipped2no0.fits'

    outname = base + 'ugc_2698_ahcorr_n10_mge_06.txt'  # None  # 'ugc_2698_ahcorr_n11_mge_04.txt'
    num = 10
    ql = [0.6, 1.]

    # fit_u2698(ahcorr_img, reg_mask, psf_file, num=num, write_out=outname, qlims=ql)
    fit_u2698(regH_img, comb_mask, psf_file, num=num, write_out=outname, qlims=ql)
    print(oop)

    display_mod(galfit_out='galfit.72')
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



  Total_Counts  Sigma_Pixels      qObs
############################################
      9104.31      1.80429            1
      1994.43      2.73585            1
      10603.9      4.69608            1
      30319.6       7.4939     0.602674
      23105.5       10.078     0.827892
      53889.6      14.1119     0.745498
      90717.1      25.9335     0.761944
      83508.1      55.3055     0.630831
       153750      81.6541      0.73941
      33830.9       147.68            1
      12248.8      164.898     0.404042
       129416      168.936     0.684078
      36981.4      294.837            1
         3191      558.949     0.184736
       172209      558.949     0.738044
++++++++++++++++++++++++++++++++++++++++++++


############################################
  Total_Counts  Sigma_Pixels      qObs
############################################
      1062.94      1.83785     0.702801
      9605.44      2.55936     0.704422
      37898.2      5.63516      0.70118
      1735.33      7.69404     0.191735
      45599.8      10.3421     0.760172
      43784.9      15.1333     0.748806
      88172.8      26.0086     0.764245
        82879      55.0517     0.630104
       154973      81.6569     0.738075
      34088.9      147.728            1
      11501.7      167.444      0.39655
       129631      168.857      0.68322
        37018      294.871            1
      3153.37      558.949     0.184362
       172182      558.949     0.738117
++++++++++++++++++++++++++++++++++++++++++++


# BELOW: using better 2D-gaussian-based PSF and not using the dust mask
############################################
  Total_Counts  Sigma_Pixels      qObs
############################################
      1157.91      1.82863     0.701628
      9430.11      2.54726     0.702732
        37000      5.58782     0.700525
      1742.02      7.66752     0.189333
      31953.1      9.78372     0.751368
      34462.8      12.6929      0.76841
      26868.4      16.8582     0.735097
      85918.2      26.2533     0.766337
      80699.6      55.1753     0.625563
       155313      81.1197     0.739494
      30804.3      147.723            1
      14434.3      166.975     0.415181
       130802      167.728     0.694388
      37233.6      294.281            1
      3351.58      558.949      0.18673
       172155      558.949     0.738193
++++++++++++++++++++++++++++++++++++++++++++


# MGE FIT 1D FOR PSF:
############################################
 Computation time: 0.08 seconds
  Total Iterations:  8
Nonzero Gaussians:  8
 Unused Gaussians:  7
 Chi2: 1.192 
 STDEV: 0.1606
 MEANABSDEV: 0.09709
############################################
 Total_Counts      Sigma
############################################
     0.146607     0.679721
     0.162065      1.36287
  0.000591738      3.27422
   0.00519833       4.2905
   0.00183743      6.95761
   0.00111433      10.4357
   0.00021939      19.0168
  0.000112128      38.3946
############################################
look
[[1.46607120e-01 1.62064550e-01 5.91737513e-04 5.19833186e-03
  1.83743233e-03 1.11433121e-03 2.19389570e-04 1.12127639e-04]
 [6.79720513e-01 1.36287440e+00 3.27422206e+00 4.29049532e+00
  6.95760865e+00 1.04357095e+01 1.90167687e+01 3.83946290e+01]]


# BELOW: MGE for 2698 using the PSF from mgefit1d
############################################
  Total_Counts  Sigma_Pixels      qObs
############################################
      1614.18      1.88377     0.695413
       9624.5      2.53562     0.695396
      37366.8      5.60134      0.69543
      1920.69      7.47509     0.191258
      29510.8      9.61615     0.753008
      54363.7      13.5609     0.750675
      91670.3      25.3629     0.762905
        84953      54.5601     0.632941
       154734      81.7813     0.738034
        32438      147.737            1
      12764.7      166.029     0.404675
       129929      168.694     0.688262
      37001.7      294.766            1
      3209.33      558.949     0.184371
       172204      558.949     0.738063


# Same as immediately above, but with qbounds][0.4, 1.0]
############################################
  Total_Counts  Sigma_Pixels      qObs
############################################
      3250.33      2.04194          0.4
      8070.39      2.60906     0.723649
      34123.1      5.57814     0.729482
       6459.8      6.48883          0.4
        45336      10.4746      0.75228
      42386.7      15.0798     0.748743
      88146.9        25.95     0.763055
      82669.8      54.9177     0.630267
       154801      81.5853     0.737208
      38367.1      147.348            1
      11470.9      166.308          0.4
       123244      168.288      0.67855
      36150.6      296.136            1
      5566.44      296.864          0.4
       172801      558.949     0.736923
++++++++++++++++++++++++++++++++++++++++++++

# Same as immediately above, but with qbounds][0.5, 1.0]
############################################
  Total_Counts  Sigma_Pixels      qObs
############################################
      8313.43      2.22579     0.556158
      7933.53      3.81251     0.920391
      18092.3      5.61535         0.55
      23377.1       6.7435     0.727228
      48969.2      11.2483     0.758479
      32992.7       15.695     0.740096
      87784.8      25.8775     0.764633
      81025.4      54.5412      0.63036
       156381       81.464     0.730236
      38353.4      144.002            1
        74216      164.445     0.742556
      43249.4      167.139         0.55
        22989      208.773         0.55
      35787.1      296.649            1
       173480      558.949     0.735674
++++++++++++++++++++++++++++++++++++++++++++

run PSF in MGE code (start with find galaxy etc)
use that to characterize PSF later (1-D PSF)
mge fit 1d
then normalize output gaussians

also make 1D surface brightness profiles
*e.g. iraf or pyraf ellipse task
# BUCKET: run ellipse task on main galaxy (dust-corr) and on GALFIT output, then plot both
# ellipse task should show 1D surf brightness

For my prelim: I'll do similar first-pass gas modeling of more CEGs or of other galaxies at upper-mass end
'''


# NOTE: GOT 0.05 ERROR WHEN I USED ZP J) 25.95, BUT NOT WHEN I USED ZP J) 24.697