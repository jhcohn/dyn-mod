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

from scipy import ndimage


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
              persec=False, sky=0.37):
    """
    This function fits an mge to UGC 2698

    Mask the object before MGE fitting.

    We model an HST/WFC3/F160W image of UGC 2698.

    persec: if input_image is in units of electrons per second, set persec=True; else if in electrons, set persec=False
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
            es = 'Electrons'
            if persec:
                es = 'Electrons_per_sec'
            o.write('# ' + es + ' Sigma_pix qObs xc yc ang\n')
            for j in range(len(m.sol[0])):
                o.write(str(m.sol[0][j]) + ' ' + str(m.sol[1][j]) + ' ' + str(m.sol[2][j]) + ' ' + str(xc1) + ' ' +
                        str(yc1) + ' ' + str(ang1) + '\n')
        print('written!')


def display_mod(galfit_out=None, texp=898.467164, sky=339.493665331):
    """
    This function displays the model mge of UGC 2698, from fit_ugc2698 -> GALFIT -> out_galfit.py -> input here!

    """
    mags = []
    fwhms = []
    qs = []
    pas = []
    xc1 = []
    yc1 = []
    component = None
    with open(galfit_out, 'r') as go:
        for line in go:
            cols = line.split()
            if line.startswith('A)'):
                file = cols[1]
            elif line.startswith('B)'):
                outfile = cols[1]
            elif line.startswith('D)'):
                psf_file = cols[1]
            elif line.startswith('F)'):
                mask = cols[1]
            elif line.startswith('J)'):
                zp = float(cols[1])
            elif line.startswith('K)'):
                scale1 = float(cols[1])  # (arcsec/pixel); cols[1] = dx, cols[2] = dy; for this case, dx = dy = 0.1
            elif line.startswith(' 0)'):
                component = cols[1]
            elif line.startswith(' 1)') and component == 'sky':
                sky = float(cols[1])
            elif line.startswith(' 1)') and component == 'gaussian':
                yc1.append(float(cols[1]))  # swap from GALFIT to MGE units, so swap x & y
                xc1.append(float(cols[2]))  # swap from GALFIT to MGE units, so swap x & y
            elif line.startswith(' 3)') and component == 'gaussian':
                mags.append(float(cols[1]))  # integrated mags
            elif line.startswith(' 4)') and component == 'gaussian':
                fwhms.append(float(cols[1]))  # fwhm_pix
            elif line.startswith(' 9)') and component == 'gaussian':
                qs.append(float(cols[1]))  # qObs
            elif line.startswith('10)') and component == 'gaussian':
                pas.append(float(cols[1]))

    # scale1 = 0.1
    hdu = fits.open(file)
    img1 = hdu[0].data
    img1 -= sky  # subtract sky
    hdu.close()

    hdu = fits.open(outfile)
    mod_img = hdu[2].data
    mod_img -= sky
    hdu.close()

    # '''  #
    # The PSF needs to be the one for the high-resolution image used in the centre.
    # Here this is the WFC3/F160W image (we use a Gaussian PSF for simplicity)
    if not psf_file.startswith('/'):
        psf_file = '/Users/jonathancohn/Documents/dyn_mod/galfit_u2698/' + psf_file
    m_psf = fit_psf(psf_file)
    sigma_psf = m_psf.sol[1]
    norm_psf = m_psf.sol[0] / np.sum(m_psf.sol[0])

    sols = np.zeros(shape=(3,len(qs)))  # axes: electrons, sigma_pix, qObs
    for i in range(len(mags)):
        sols[0][i] = texp * 10 ** (0.4 * (zp - mags[i]))  # electrons
        sols[1][i] = fwhms[i] / 2.355  # https://en.wikipedia.org/wiki/Full_width_at_half_maximum
        sols[2][i] = qs[i]

    # Plot MGE contours of the HST image
    xc = xc1[0]
    yc = yc1[0]
    pa = pas[0]
    '''  #
    # PASTE IN CONTOUR CODE, SO I CAN EDIT IT TO INCLUDE MULTPLE x,y VALUES
    sigmapsf = np.atleast_1d(sigma_psf)
    normpsf = np.atleast_1d(norm_psf)

    lum, sigma, q = sols

    # Analytic convolution with an MGE circular Gaussian
    # Eq.(4,5) in Cappellari (2002)
    #
    model = 0.
    for lumj, sigj, qj in zip(lum, sigma, q):
        for sigP, normP in zip(sigmapsf, normpsf):
            sx = np.sqrt(sigj**2 + sigP**2)
            sy = np.sqrt((sigj*qj)**2 + sigP**2)
            n = img1.shape
            ang = np.radians(-pa)  # (pos_ang - 90.) ; pos_ang = 90 - pa ; pos_ang - 90 = 90 - pa - 90 = -pa
            x, y = np.ogrid[:n[0], :n[1]] - np.array([xc, yc])

            xcosang = np.cos(ang) / (np.sqrt(2.) * sx) * x
            ysinang = np.sin(ang) / (np.sqrt(2.) * sx) * y
            xsinang = np.sin(ang) / (np.sqrt(2.) * sy) * x
            ycosang = np.cos(ang) / (np.sqrt(2.) * sy) * y

            im = (xcosang + ysinang) ** 2 + (ycosang - xsinang) ** 2

            g = np.exp(-im)

            model += lumj*normP/(2.*np.pi*sx*sy) * g
    '''  #
    peak = img1[int(round(xc)), int(round(yc))]  # xc, yc
    magrange = 10.
    levels = peak * 10**(-0.4*np.arange(0, magrange, 0.5)[::-1]) # 0.5 mag/arcsec^2 steps

    binning = 1
    if binning is None:
        binning = 1
    else:
        #model = ndimage.filters.gaussian_filter(model, binning/2.355)
        #model = ndimage.zoom(model, 1./binning, order=1)
        img1 = ndimage.filters.gaussian_filter(img1, binning/2.355)
        img1 = ndimage.zoom(img1, 1./binning, order=1)
        mod_img = ndimage.filters.gaussian_filter(mod_img, binning/2.355)
        mod_img = ndimage.zoom(mod_img, 1./binning, order=1)

    ax = plt.gca()
    ax.axis('equal')
    ax.set_adjustable('box-forced')
    s = img1.shape

    scale = scale1
    if scale is None:
        extent = [0, s[1], 0, s[0]]
        plt.xlabel("pixels")
        plt.ylabel("pixels")
    else:
        extent = np.array([0, s[1], 0, s[0]])*scale*binning
        plt.xlabel("arcsec")
        plt.ylabel("arcsec")

    cnt = ax.contour(img1, levels, colors = 'k', linestyles='solid', extent=extent)
    ax.contour(mod_img, levels, colors='r', linestyles='solid', extent=extent)  # model

    mask = None
    # hdu = fits.open(mask)
    # mask = hdu[0].data
    # hdu.close()

    if mask is not None:
        a = np.ma.masked_array(mask, mask)
        ax.imshow(a, cmap='autumn_r', interpolation='nearest', origin='lower',
                  extent=extent, zorder=3, alpha=0.7)

    #mge_print_contours(img1, 90. - pas[0], xc1, yc1, sols, scale=scale1, binning=4, sigmapsf=sigma_psf,
    #                   normpsf=norm_psf)

    plt.show()


def sbprof1d(input_models, scales, types, fmts, mlabs, pfrac=0.01, plots=False, sky=337.5, es=False, akint=1354.46,
             mud=False, mgefile=None, resid=False, comp_angle=0.):
    """
    This function fits an mge to UGC 2698

    Mask the object before MGE fitting.

    We model an HST/WFC3/F160W image of UGC 2698.

    es: if input_image is in units of electrons per second, set es=True; else if in electrons, set es=False
    scales: arcsec/pix
    """
    if resid:
        fig, ax = plt.subplots(2, 1, sharex=True, figsize=(8, 16))
        fig.subplots_adjust(hspace=0.01)

    if comp_angle != 0:
        for a in range(len(comp_angle)):
            for i in range(len(input_models)):
                hdu = fits.open(input_models[i])
                img1 = hdu[0].data
                if types[i] != 'akin':
                    print('hi')
                    img1 -= sky  # subtract sky  # NOTE: ALSO SAVED NEW VERSIONS OF IMGS WITH SKY SUBTRACTED FOR USE IN GALFIT
                else:
                    pass
                hdu.close()

                # The geometric parameters below were obtained using my FIND_GALAXY program
                f = find_galaxy(img1, binning=1, fraction=pfrac, level=None, nblob=1, plot=plots, quiet=False)
                if plots:
                    plt.show()
                    plt.clf()
                ang1 = f.theta
                xc1 = f.xmed
                yc1 = f.ymed
                eps = f.eps

                s1 = sectors_photometry(img1, eps, ang1, xc1, yc1, minlevel=0, plot=plots)  # sky subtr, so minlevel=0
                if plots:
                    plt.show()  # Allow plot to appear on the screen
                    plt.clf()

                radius = s1.radius
                angle = s1.angle  # polar angle of the surface brightness measurements, taken from the galaxy major axis.
                counts = s1.counts

                # if mud:  # display with y units of mag / arcsec^2
                #    mu = []
                #    for c in range(len(counts)):
                #        mu.append((24.6949 - 2.5 * np.log10(counts[c])) / (scales[i] ** 2))
                #    counts = np.asarray(mu)

                num = angle == comp_angle[a]
                cc = counts[angle == comp_angle[a]]
                rad = radius[angle == comp_angle[a]] * scales[i]

                if resid:
                    ax[0].loglog(rad, cc, fmts[a, i], label=mlabs[a, i])
                    if i == 0:
                        modc = cc
                        modr = rad
                    elif i == 1:
                        datc = cc
                        datr = rad
                else:
                    plt.plot(rad, cc, fmts[a, i], label=mlabs[a, i])
            if resid:
                ax[1].plot(datr, (datc - modc) / datc, fmts[a, i])
                ax[1].axhline(y=0, color='k', ls='--')
                ax[0].set_ylabel('Surface brightness (electrons) \n along the major and minor axes')
                ax[1].set_ylabel(r'Residual [(data - model) / data]')
                ax[1].set_xlabel(r'arcsec')
                ax[0].legend()
            else:
                plt.xscale('log')
                plt.yscale('log')
                # plt.xlim(1e-2, 11)
                # plt.ylim(8e4, 3e6)
                plt.xlabel('arcsec')
                plt.ylabel('Surface brightness (electrons) \n along the major and minor axes')
                plt.legend()

    else:
        for i in range(len(input_models)):
            hdu = fits.open(input_models[i])
            img1 = hdu[0].data
            if types[i] != 'akin':
                print('hi')
                img1 -= sky  # subtract sky  # NOTE: ALSO SAVED NEW VERSIONS OF IMGS WITH SKY SUBTRACTED FOR USE IN GALFIT
                # img1 /= 898.467164
            else:
                #if es and i == 0 or i == 1 or i == 2:
                #    img1 *= akint
                #if i == 3:
                #    img1 *= 1354.46 / 898.467164

                # img1 = hdu[2].data  # model
                # if es and i == 3:
                    #img1 = akint[i]  # 898.467164  # 1354.46
                pass

            print(np.amax(img1), 'max')
            # NOTE: both ahcorr and regH images have the same flux values
            hdu.close()

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

            s1 = sectors_photometry(img1, eps, ang1, xc1, yc1, minlevel=0, plot=plots)  # sky subtr, so minlevel=0
            if plots:
                plt.show()  # Allow plot to appear on the screen
                plt.clf()

            radius = s1.radius
            angle = s1.angle  # polar angle of the surface brightness measurements, taken from the galaxy major axis.
            counts = s1.counts

            #if mud:  # display with y units of mag / arcsec^2
            #    mu = []
            #    for c in range(len(counts)):
            #        mu.append(24.695 + 5 * np.log10(0.08) + 2.5 * np.log10(texp) - 2.5 * np.log10(counts[c]) - 0.075)
            #    counts = np.asarray(mu)

            num = angle == comp_angle
            cc = counts[angle == comp_angle]
            print(np.amax(cc), 'max cc')
            rad = radius[angle == comp_angle] * scales[i]

            print(np.sum(num))
            #ra = []
            #co = []
            #for r in range(len(radius)):
            #    if angle[r] == 0.:
            #        ra.append(radius[r] * scale)
            #        co.append(counts[r])
            # plt.plot(ra, co, fmts[i], label=mlabs[i])
            if resid:
                ax[0].loglog(rad, cc, fmts[i], label=mlabs[i])
                if i == 0:
                    modc = cc
                    modr = rad
                elif i == 1:
                    datc = cc
                    datr = rad
            else:
                plt.plot(rad, cc, fmts[i], label=mlabs[i])
        if resid:
            ax[1].plot(datr, (datc-modc) / datc, 'ko')
            ax[1].axhline(y=0, color='k', ls='--')
            ax[0].set_ylabel('Surface brightness (electrons) \n along the major axis')
            ax[1].set_ylabel(r'Residual [(data - model) / data]')
            ax[1].set_xlabel(r'arcsec')
            ax[0].legend()
        else:
            plt.xscale('log')
            plt.yscale('log')
            #plt.xlim(1e-2, 11)
            #plt.ylim(8e4, 3e6)
            plt.xlabel('arcsec')
            plt.ylabel('Surface brightness (electrons) along the major axis')
            plt.legend()
    plt.show()


if __name__ == '__main__':

    print("\nFitting UGC 2698-----------------------------------\n")

    base = '/Users/jonathancohn/Documents/dyn_mod/galfit_u2698/'

    # TRY: not dividing by texp to do the exposure time=1 with akin's model
    # AND compare only to mine calculated from fixed models!

    '''  #
    mods = [base+'galfit_out_rrenewsky_test.fits', base+'galfit_out_rhenewsky_test.fits',
            base+'galfit_out_ahenewsky_test.fits', base+'galfit_out_u2698_akin_esnodt.fits']
    # '''  #

    '''  #
    # PRODUCING PLOTS FOR GITHUB FINAL MGEs PAGE:
    # https://github.tamu.edu/joncohn/gas-dynamical-modeling/wiki/Final-MGEs-for-UGC-2698
    modre = [base+'galfit_out_u2698_rrenewsky_linear_pf001_zp24_model.fits',
             base + 'ugc2698_f160w_pxfr075_pxs010_drz_rapidnuc_sci_nonan_e.fits']
    modrepsf = [base+'galfit_outpsf_u2698_rre_linear_pf001_zp24_model.fits',
                base + 'ugc2698_f160w_pxfr075_pxs010_drz_rapidnuc_sci_nonan_e.fits']
    modah = [base+'galfit_out_u2698_ahenewsky_linear_pf001_zp24_model.fits',
             base + 'ugc2698_f160w_pxfr075_pxs010_ahcorr_rapidnuc_sci_nonan_counts.fits']
    modahpsf = [base+'galfit_outpsf_u2698_ahe_linear_pf001_zp24_model.fits',
                base + 'ugc2698_f160w_pxfr075_pxs010_ahcorr_rapidnuc_sci_nonan_counts.fits']
    modrh = [base+'galfit_out_u2698_rhenewsky_model.fits',
             base + 'ugc2698_f160w_pxfr075_pxs010_drz_rapidnuc_sci_nonan_counts.fits']
    modrhpsf = [base+'galfit_outpsf_u2698_rhe_linear_pf001_zp24_model.fits',
                base + 'ugc2698_f160w_pxfr075_pxs010_drz_rapidnuc_sci_nonan_counts.fits']
    scale = [0.1, 0.1]
    typs = [None, None]
    labre = [r'GALFIT model (HST $H$-band + regular mask)', r'Data (HST $H$-band)']
    labrepsf = [r'GALFIT model (HST $H$-band + regular mask + AGN)', r'Data (HST $H$-band)']
    labrh = [r'GALFIT model (HST $H$-band + dust mask)', r'Data (Dust-masked HST $H$-band)']
    labrhpsf = [r'GALFIT model (HST $H$-band + dust mask + AGN)', r'Data (Dust-masked HST $H$-band)']
    labah = ['GALFIT model (Dust-corrected \n' + r'HST $H$-band + regular mask)', r'Data (Dust-corrected HST $H$-band)']
    labahpsf = ['GALFIT model (Dust-corrected \n' + r'HST $H$-band + regular mask + AGN)',
                r'Data (Dust-corrected HST $H$-band)']
    fmt = ['b-', 'm--']

    compmods = [modre[0], modrh[0], modah[0]]
    compscale = [0.1, 0.1, 0.1]
    comptype = [None, None, None]
    complab = np.asarray([[labre[0] + ', major axis', labrh[0] + ', major axis',
                           r'GALFIT model (Dust-corrected HST $H$-band + regular mask), major axis'],
                          [labre[0] + ', minor axis', labrh[0] + ', minor axis',
                           r'GALFIT model (Dust-corrected HST $H$-band + regular mask), minor axis']])
    fmts = np.asarray([['go', 'mo', 'bo'], ['gx', 'mx', 'bx']])
    comp_ang = [0., 90.]
    sbprof1d(compmods, compscale, comptype, fmts, complab, resid=False, comp_angle=comp_ang)
    print(oop)

    #sbprof1d(modre, scale, type, fmt, labre, akint=1., es=False, resid=True)
    #sbprof1d(modrepsf, scale, type, fmt, labrepsf, akint=1., es=False, resid=True)
    #sbprof1d(modrh, scale, type, fmt, labrh, akint=1., es=False, resid=True)
    #sbprof1d(modrhpsf, scale, type, fmt, labrhpsf, akint=1., es=False, resid=True)
    sbprof1d(modah, scale, typs, fmt, labah, akint=1., es=False, resid=True)
    sbprof1d(modahpsf, scale, typs, fmt, labahpsf, akint=1., es=False, resid=True)
    print(oop)
    # '''  #



    mods = [base+'galfit_out_u2698_rre_esnodt.fits', base+'galfit_out_u2698_rhe_esnodt.fits',
            base+'galfit_out_u2698_ahe_esnodt.fits', base+'galfit_out_u2698_akin_esnodt.fits']
            # [base+'galfit_out_u2698_rrenewsky_linear_pf001_zp24_model.fits',
            # base+'galfit_out_u2698_rhenewsky_model.fits',
            # base+'galfit_out_u2698_ahenewsky_linear_pf001_zp24_model.fits',
            # base+'galfit_out_u2698_ben_fixedmod_zp24.fits']
            # base+'galfit_out_ahenewsky_test.fits']
            # base+'galfit_out_u2698_akin_fixedmod_zp24_corrdt_es_newxy.fits']  # es=False # _es_newxy.fits']  # _adjrange0box  # es=True
    scales = [0.1, 0.1, 0.1, 0.06]  # 0.06]  # 0.1]
    types = ['akin', 'akin', 'akin', 'akin']
    akt = 1  # 898.467164 # 1354.46  # 1354.46  # 898.467164
    fmts = ['k-', 'm-', 'b-', 'r--']
    mod_labs = [r'GALFIT model (reg H + reg mask)', r'GALFIT model (reg H + dust mask)',
                r'GALFIT model (ahcorr + reg mask)',
                # r"Ben's model (reg H + reg mask)"]
                # r'GALFIT model (ahcorr + reg mask, calculated as a fixed model)']
                r"Akin's model (reg H + reg mask; different t_exp and scale)"]

    sbprof1d(mods, scales, types, fmts, mod_labs, akint=akt, es=True)
    print(oop)

    ahcorr_cps = base + 'ugc2698_f160w_pxfr075_pxs010_ahcorr_rapidnuc_sci_nonan_cps.fits'
    ahcorr_counts = base + 'ugc2698_f160w_pxfr075_pxs010_ahcorr_rapidnuc_sci_nonan_counts.fits'
    ahe = base + 'ugc2698_f160w_pxfr075_pxs010_ahcorr_rapidnuc_sci_nonan_e.fits'
    reg_mask = base + 'f160w_maskedgemask_px010.fits'

    regH_img = base + 'ugc2698_f160w_pxfr075_pxs010_drz_rapidnuc_sci_nonan_cps.fits'
    regH_counts = base + 'ugc2698_f160w_pxfr075_pxs010_drz_rapidnuc_sci_nonan_counts.fits'
    re = base + 'ugc2698_f160w_pxfr075_pxs010_drz_rapidnuc_sci_nonan_e.fits'
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
    out_ahcps = base + 'mge_ugc_2698_ahcorr_linear_pf001_cps.txt'
    out_ahe =  base + 'mge_ugc_2698_ahe_linear_pf001.txt'  # ahcorr, regmask, units=e

    out_rhcounts = base + 'mge_ugc_2698_regH_linear_pf001_counts.txt'
    out_rhcps = base + 'mge_ugc_2698_regH_linear_pf001_cps.txt'
    out_rhe = base + 'mge_ugc_2698_rhe_linear_pf001.txt'  # regH, combmask, units=e

    out_rr_counts = base + 'mge_ugc_2698_regH_regm_linear_pf001_counts.txt'
    out_rr_cps = base + 'mge_ugc_2698_regH_regm_linear_pf001_cps.txt'
    out_rre = base + 'mge_ugc_2698_rre_linear_pf001.txt'  # regH, regmask, units=e

    sky_e = 337.5  # 339.493665331
    rms_e = 22.4  # 21.5123034564
    # sky_cps = 0.377858732438
    # rms_cps = 0.0239433389818

    num = None  # 10
    ql6 = [0.6, 1.]
    ql5 = [0.5, 1.]
    ql4 = [0.4, 1.]

    pf = 0.01

    ### do regH with reg (not dust) mask
    ### then do regH with reg (not dust) mask WITH PSF in GALFIT (remove innermost mge component and replace with psf for first guess)
    ### then do regH with dust mask: make sure sky is correct, try just doing it directly how I've been doing it
    ### then do regH with dust mask, with PSF (same as before; pop inner component output by mge fit sectors and replace with psf)

    display_mod(base + 'galfit.122')  # galfit.106 (rre); galfit.107 (rhe); galfit.112 (ahe); galfit.113 (rrepsf);
    # galfit.116 (rhepsf); galfit.118 (ahepsf); galfit.120 (rrenewsky); galfit.121 (rhenewsky); galfit.122 (ahenewsky)
    print(oop)

    fit_u2698(re, reg_mask, psf_file, pfrac=pf, num=num, write_out=out_rre, plots=False, persec=False, sky=sky_e)
    fit_u2698(re, comb_mask, psf_file, pfrac=pf, num=num, write_out=out_rhe, plots=False, persec=False, sky=sky_e)
    fit_u2698(ahe, reg_mask, psf_file, pfrac=pf, num=num, write_out=out_ahe, plots=False, persec=False, sky=sky_e)

    # display_mod(galfit_out='galfit.72')
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

# NOTE: GOT 0.05 ERROR WHEN I USED ZP J) 25.95, BUT NOT WHEN I USED ZP J) 24.697
