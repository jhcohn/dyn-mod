from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import AxesGrid
import os
from scipy import interpolate


gf = '/Users/jonathancohn/Documents/dyn_mod/galfit/'
fj = '/Users/jonathancohn/Documents/dyn_mod/for_jonathan/'
base = '/Users/jonathancohn/Documents/dyn_mod/galfit_u2698/'
# out = base + 'galfit_regH_combmask_mge_n10_q06_zp25.fits'
# out = base + 'galfit_ahcorr_mge_n10_055_zp25.fits'  # 'galfit_mge_055_zp25.fits'
# 'galfit_mge_055_zp24_myconstraint_reset_take2.fits'
# 'galfit_mge_055_zp24_cadj.fits'  # 'galfit_out_mge_055_zp24_myconstraint.fits'
# 'galfit_mge0.fits'  # 'imgblock_masked8.fits' # 'imgblock_masked3.fits'  # 'imgblock_test49.fits'
abs = 0


def sb_profile(image, out_tab, mask=None):
    # run ellipse task on input galaxy and on GALFIT output (ellipse task should show 1D surf brightness)
    import pyraf
    from pyraf import iraf  # need to be in iraf27 python environment!
    from iraf import stsdas, analysis, isophote
    from iraf import stsdas, fitting
    # print(image, out_tab)
    if mask is not None:
        isophote.ellipse(input=image, output=out_tab, dqf=mask)  #, a0=a0, x0=x0, y0=y0, eps0=eps0, teta0=theta0)
    else:
        isophote.ellipse(input=image, output=out_tab)  #, a0=a0, x0=x0, y0=y0, eps0=eps0, teta0=theta0)
    # NOTE: THIS CREATES A FILE BUT DOESN'T SAVE TABLE IN USEFUL MANNER. OPEN TABLE AND PASTE THE PRINTED OUTPUT INTO IT
    # in terminal with: source activate iraf27: pyraf: stsdas: analysis: isophote: epar ellipse:
    # # PSET geompar: x0 879.95, y0 489.99, ellip0 0.284, pa0 96.3 (-6.3), sma0 324.5


def tab_to_sb(out_tab):
    smas = []
    intens = []
    with open(out_tab, 'r') as efile:  # '/Users/jonathancohn/Documents/mge/out_tab1_saved.txt'
        for line in efile:
            if not line.startswith('#'):
                cols = line.split()
                c1 = ''
                parens = 0
                for char in cols[1]:
                    if char == '(':
                        parens += 1
                    elif parens == 0:
                        c1 += char
                smas.append(float(cols[0]))
                intens.append(float(c1))
    smas.reverse()
    intens.reverse()
    # print(smas)
    # print(intens)
    return smas, intens


def sb_compare3(in_filename, out_filename, just_out, table_img, table_mod, table_ben, qbounds=[0., 1.], sky=None,
                modtype=None, es=False, mask=None):
    """

    :param in_filename:
    :param out_filename:
    :param just_out:
    :param table_img:
    :param table_mod:
    :param table_ben:
    :param qbounds:
    :param sky: default to None now that I account for sky in GALFIT fitting!
    :param counts: if img in units of counts per sec, counts=False; if img in units of counts, counts=True
    :param modtype:
    :return:
    """

    ###
    if not os.path.exists(just_out):
        with fits.open(out_filename) as hdu:
            # print(hdu.info())
            # print(hdu_h[0].header)
            hdr = hdu[0].header
            if modtype == 'akin':
                data = hdu[0].data
                fits.writeto(just_out, data, hdr)
            elif modtype is None:
                data = hdu[0].data
                data1 = hdu[1].data  # input image
                data2 = hdu[2].data  # model
                data3 = hdu[3].data  # residual
                fits.writeto(just_out, data2, hdr)

    if not os.path.exists(table_img):
        sb_profile(in_filename, table_img, mask=mask)  # image
        print('When these finish, copy-paste what was printed in the terminal into the newly created table files here')
        print('copy-paste from "# Semi-" down to the line with "CPU seconds", NOT including the line with "CPU seconds')
        print('Then, re-run this script (out_galfit.py)')
        print('This table is ' + table_img)
    if not os.path.exists(table_mod):
        sb_profile(just_out, table_mod)  # image
        print('When these finish, copy-paste what was printed in the terminal into the newly created table files here')
        print('copy-paste from "# Semi-" down to the line with "CPU seconds", NOT including the line with "CPU seconds')
        print('Then, re-run this script (out_galfit.py)')
        print('This table is ' + table_mod)
        print(oops)
    ###

    # Create 1D surface brightness profile arrays for image, model from tables output by Ellipse task
    smas_img, intens_img = tab_to_sb(table_img)
    smas_mod, intens_mod = tab_to_sb(table_mod)
    smas_ben, intens_ben = tab_to_sb(table_ben)

    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(8, 16))
    fig.subplots_adjust(hspace=0.01)
    # fig.text(0.04, 0.5, 'counts', va='center', rotation='vertical')
    # fig.text(0.96, 0.5, 'error (%)', va='center', rotation='vertical')
    ax[1].set_xlabel('arcsec')

    if es:
        ax[0].set_ylim(200., 3. * 10 ** 6)  # 2e-3, 3.*10**3
    else:
        ax[0].set_ylim(8e-2, 3.*10**3)  # 2e-3, 3.*10**3
    ax[0].set_xlim(0.5, 400.)
    ax[1].set_ylim(-0.15, 0.15)  # -0.4, 0.4)  #

    # Interpolate the model to have the same sma values as the image, so I can take residuals!
    f = interpolate.interp1d(smas_mod, intens_mod, fill_value="extrapolate")  # x, y
    if es:
        intens_ben = [ib * 898.467164 for ib in intens_ben]
        fben = interpolate.interp1d(smas_ben, intens_ben, fill_value="extrapolate")  # x, y
    else:
        fben = interpolate.interp1d(smas_ben, intens_ben, fill_value="extrapolate")  # x, y
    if sky is not None:
        intens_mod_samex = f(smas_img) - sky
        intens_ben_samex = fben(smas_img) - sky
    else:
        print(smas_img)
        intens_mod_samex = f(smas_img)
        intens_ben_samex = fben(smas_img)

    # if modtype == 'akin':
    #     intens_mod_samex = [ims * 1354.46 for ims in intens_mod_samex]


    # Add in Ben's MGE as well!
    if not os.path.exists(table_ben):
        # BUCKET FIX THIS!
        total = np.zeros(shape=np.array(smas_img).shape)
        counts = []
        sigmas = []
        qs = []
        with open(table_ben, 'r') as tb:
            counter = 0
            for line in tb:
                counter += 1
                if counter > 2:
                    cols = line.split()
                    counts.append(10 ** ((25.95 - float(cols[1])) / 2.5))
                    sigmas.append(float(cols[3]) / (2 * np.sqrt(2 * np.log(2))))
                    qs.append(float(cols[5]))
        counts = np.array(counts)
        sigmas = np.array(sigmas)
        qs = np.array(qs)
        print(counts)
        print(sigmas)
        for each_gauss in range(len(counts)):
            norm = counts[each_gauss] / (sigmas[each_gauss]**2 * 2*np.pi*qs[each_gauss])
            print(norm)
            yarray = norm * np.exp(-0.5 * (smas_img / sigmas[each_gauss]) ** 2)
            total += yarray
            # ax[0].loglog(smas, yarray, 'r--')
        # ax[0].loglog(smas_img, total, 'r-', label=r"Ben's MGE parameters", linewidth=2)

    benlab = r"Ben's MGE parameters"
    ylab = 'Mean counts along semi-major axis'
    if es:
        ylab = 'Mean counts (electrons) along semi-major axis'
        benlab += r" (multiplied by the exposure time)"
    ax[0].loglog(smas_img, intens_ben_samex, 'r-', label=benlab, linewidth=2)

    # SHOULD SKY BE SUBTRACTED HERE?!
    ax[0].loglog(smas_img, np.array(intens_img), 'k--', label=r'Data (profile from Ellipse task)', linewidth=2)
    ax[0].loglog(smas_img, intens_mod_samex, 'b-', label=r'GALFIT Model (profile from Ellipse task)', linewidth=2)
    # smas_mod, intens_mod

    ax[1].plot(smas_img, (np.array(intens_img) - np.array(intens_mod_samex)) / np.array(intens_img), 'bo', markersize=4)
    # ax[1].plot(smas_img, (np.array(intens_img) - np.array(total)) / np.array(intens_img), 'ro', markersize=4)
    ax[1].plot(smas_img, (np.array(intens_img) - np.array(intens_ben_samex)) / np.array(intens_img), 'ro', markersize=4)
    ax[1].axhline(y=0., color='k', ls='--')

    ax[0].set_ylabel(ylab)
    ax[1].set_ylabel(r'Residual [(data - model) / data]')
    ax[0].legend()
    if qbounds is not None:
        ax[0].set_title(r'UGC 2698 surface brightness profile, qbounds=' + str(qbounds))
    else:
        ax[0].set_title(r'UGC 2698 surface brightness profile')
    plt.show()
    plt.clf()


def sb_prof_compare(in_filename, out_filename, just_out, table_img, table_mod, qbounds=[0., 1.], sky=None, scale=0.1,
                    modtype=None, es=True, mask=None, glx='UGC 2698'):
    '''

    :param in_filename:
    :param out_filename:
    :param just_out:
    :param table_img:
    :param table_mod:
    :param qbounds:
    :param sky: default to None now that I account for sky in GALFIT fitting!
    :param modtype:
    :return:
    '''

    ###
    if not os.path.exists(just_out):
        with fits.open(out_filename) as hdu:
            print(hdu.info())
            # print(hdu_h[0].header)
            hdr = hdu[0].header
            if modtype == 'akin':
                data = hdu[0].data
                fits.writeto(just_out, data, hdr)
            else:
                data = hdu[0].data
                data1 = hdu[1].data  # input image
                data2 = hdu[2].data  # model
                data3 = hdu[3].data  # residual
                fits.writeto(just_out, data2, hdr)

    if not os.path.exists(table_img):
        sb_profile(in_filename, table_img, mask=mask)  # image
        print('When these finish, copy-paste what was printed in the terminal into the newly created table files here')
        print('copy-paste from "# Semi-" down to the line with "CPU seconds", NOT including the line with "CPU seconds')
        print('Then, re-run this script (out_galfit.py)')
        print('This table is ' + table_img)
    if not os.path.exists(table_mod):
        sb_profile(just_out, table_mod)  # image
        print('When these finish, copy-paste what was printed in the terminal into the newly created table files here')
        print('copy-paste from "# Semi-" down to the line with "CPU seconds", NOT including the line with "CPU seconds')
        print('Then, re-run this script (out_galfit.py)')
        print('This table is ' + table_mod)
        print(oops)
    ###

    # Create 1D surface brightness profile arrays for image, model from tables output by Ellipse task
    smas_img, intens_img = tab_to_sb(table_img)
    smas_mod, intens_mod = tab_to_sb(table_mod)
    smas_mod = np.asarray(smas_mod) * scale
    smas_img = np.asarray(smas_img) * scale

    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(8, 16))
    fig.subplots_adjust(hspace=0.01)
    # fig.text(0.04, 0.5, 'counts', va='center', rotation='vertical')
    # fig.text(0.96, 0.5, 'error (%)', va='center', rotation='vertical')
    ax[1].set_xlabel('arcsec')

    if es:
        ax[0].set_ylim(200., 3. * 10 ** 6)  # 2e-3, 3.*10**3
        ax[0].set_ylabel('Mean counts (electrons) along semi-major axis')
    else:
        ax[0].set_ylim(8e-2, 3.*10**3)  # 2e-3, 3.*10**3
    ax[0].set_xlim(0.5, 400.)
    ax[1].set_ylim(-0.15, 0.15)  # -0.4, 0.4)  #

    f = interpolate.interp1d(smas_mod, intens_mod, fill_value="extrapolate")  # x, y
    if sky is not None:
        intens_mod_samex = f(smas_img) - sky
    else:
        intens_mod_samex = f(smas_img)

    # intens_mod_samex = [ims * 898.467164 / 1354.46 for ims in intens_mod_samex]

    ax[0].loglog(smas_img, np.array(intens_img), 'b--', label=r'Data (profile from Ellipse task)', linewidth=2)
    ax[0].loglog(smas_img, intens_mod_samex, 'r-', label=r'GALFIT Model (profile from Ellipse task)', linewidth=2)
    # , with t_exp~898

    ax[1].plot(smas_img, (np.array(intens_img) - np.array(intens_mod_samex)) / (np.array(intens_img)), 'ko', markersize=4)
    ax[1].axhline(y=0., color='k', ls='--')

    ax[1].set_ylabel(r'Residual [(data - model) / data]')
    ax[0].legend()
    if qbounds is not None:
        ax[0].set_title(glx + r' surface brightness profile, qbounds=' + str(qbounds))
    else:
        ax[0].set_title(glx + r' surface brightness profile')
    plt.show()
    plt.clf()


def plot3(img, masked_img=None, normed=True, clipreg=None, indivs=False, cliprange=False):
    with fits.open(img) as hdu:
        print(hdu.info())
        # print(hdu_h[0].header)
        hdr = hdu[0].header
        data = hdu[0].data
        data1 = hdu[1].data  # input image
        data2 = hdu[2].data  # model
        data3 = hdu[3].data  # residual

    if masked_img is not None:
        with fits.open(masked_img) as hdu:
            print(hdu.info())
            # print(hdu_h[0].header)
            hdr_masked = hdu[0].header
            data_masked = hdu[0].data
            data4 = data3 / data_masked
    else:
        data4 = data3 / data1  # residual scaled to the input image
    # data5 = (data2 - data1 - sky) / data1
    # data5[np.isnan(data5)] = 0.
    dataset = [data1, data2, data3, data4]

    for dat in dataset:
        dat[np.isfinite(dat) == False] = 0.  # change infinities to 0.

    if indivs:
        for dat in dataset:
            # plt.imshow(dat, origin='lower', vmin=np.amin([dat, -dat]), vmax=np.amax([dat, -dat]), cmap='RdBu')
            plt.imshow(dat, origin='lower', vmin=np.amin(dat[1]), vmax=np.amax(dat[1]))
            plt.colorbar()
            plt.show()

    labels = ['Input image', 'Model', 'Residual']  # [(data - model) / data]
    i = 0

    if clipreg is not None:
        for d in range(len(dataset)):
            if len(clipreg) == 4:
                dataset[d] = dataset[d][clipreg[0]:clipreg[1], clipreg[2]:clipreg[3]]
            else:
                dataset[d] = dataset[d][clipreg:-clipreg, clipreg:-clipreg]

    if normed:
        labels[2] += ' (normalized to data)'

    fig = plt.figure()
    grid = AxesGrid(fig, 111,
                    nrows_ncols=(1, 3),
                    axes_pad=0.01,
                    cbar_mode='single',
                    cbar_location='right',
                    cbar_pad=0.1)
    for ax in grid:
        if cliprange:
            if normed and i == 2:
                im = ax.imshow(dataset[3], origin='lower', vmin=np.amin([-dataset[3], dataset[3]]),
                               vmax=np.amax([-dataset[3], dataset[3]]), cmap='RdBu')  # , np.nanmax(data4)
            else:
                im = ax.imshow(dataset[i], origin='lower', vmin=np.amin(dataset[1]), vmax=np.amax(dataset[1]))

        elif i == 0 or i == 1:
            im = ax.imshow(dataset[i], origin='lower', vmin=np.amin([np.nanmin(dataset[0]), np.nanmin(dataset[1])]),
                           vmax=np.amax([np.nanmax(dataset[0]), np.nanmax(dataset[1])]))  # , cmap='RdBu_r'
        else:
            if normed:
                im = ax.imshow(dataset[3], origin='lower', vmin=np.amin([-dataset[3], dataset[3]]),
                               vmax=np.amax([-dataset[3], dataset[3]]), cmap='RdBu')  # , np.nanmax(data4)
            else:
                im = ax.imshow(np.abs(dataset[i]), origin='lower', vmin=np.nanmin(dataset[2]),
                               vmax=np.nanmax(dataset[2]),
                               cmap='RdBu')  # , np.nanmax(data4)
        ax.set_title(labels[i])
        i += 1
        ax.set_xlabel(r'x [pixels]', fontsize=20)  # x [arcsec]
        ax.set_ylabel(r'y [pixels]', fontsize=20)  # y [arcsec]
    cbar = ax.cax.colorbar(im)
    cbar = grid.cbar_axes[0].colorbar(im)
    plt.show()
    plt.clf()


def compare_sbs(img_labs=None, mod_labs=None, colors=None, table_imgs=None, table_mods=None, psf=False, sky=None,
                es=True):
    """

    :param img_labs: list of labels for the data
    :param mod_labs: list of labels for the models
    :param colors: list of colors to use for the different cases (regH+reg mask, regH+dust, ahcorr+reg)
    :param table_imgs: list of table_img files
    :param table_mods: list of table_mod files
    :param psf: if showing models with PSF (AGN) included, psf=True; else, showing models with Akin's table included
    :param sky: default to None now that I account for sky in GALFIT fitting!
    :param es: if img in units of electrons per sec, es=False; if img in units of electrons, es=True
    :return:
    """
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(8, 16))
    fig.subplots_adjust(hspace=0.01)
    # fig.text(0.04, 0.5, 'counts', va='center', rotation='vertical')
    # fig.text(0.96, 0.5, 'error (%)', va='center', rotation='vertical')
    ax[1].set_xlabel('arcsec')
    if es:
        ax[0].set_ylim(200., 3. * 10 ** 6)  # 2e-3, 3.*10**3
    else:
        ax[0].set_ylim(8e-2, 3.*10**3)  # 2e-3, 3.*10**3
    ax[0].set_xlim(0.5, 400.)
    ax[1].set_ylim(-0.15, 0.15)  # -0.4, 0.4)  #

    for i in range(len(table_imgs)):
        table_img = table_imgs[i]
        table_mod = table_mods[i]
        img_lab = img_labs[i]  # r'Data (profile from Ellipse task)'
        mod_lab = mod_labs[i]  # r'GALFIT Model (profile from Ellipse task)'
        clr = colors[i]

        # Create 1D surface brightness profile arrays for image, model from tables output by Ellipse task
        smas_img, intens_img = tab_to_sb(table_img)
        smas_mod, intens_mod = tab_to_sb(table_mod)

        # Interpolate the model to have the same sma values as the image, so I can take residuals!
        f = interpolate.interp1d(smas_mod, intens_mod, fill_value="extrapolate")  # x, y

        if sky is not None:
            intens_mod_samex = f(smas_img) - sky
        else:
            print(smas_img)
            intens_mod_samex = f(smas_img)

        # PLOT 1D profiles
        if not psf and i == len(table_imgs) - 1:  # if including Akin table & on the last iter (i.e. on Akin's table)
            ax[0].loglog(smas_img, np.array(intens_img), colors[0] + '--', label=None, linewidth=2)
            ax[0].loglog(smas_img, intens_mod_samex, clr + ':', label=mod_lab, linewidth=2)
        else:
            ax[0].loglog(smas_img, np.array(intens_img), clr + '--', label=img_lab, linewidth=2)
            ax[0].loglog(smas_img, intens_mod_samex, clr + '-', label=mod_lab, linewidth=2)

        # PLOT residuals
        res = (np.array(intens_img) - np.array(intens_mod_samex)) / np.array(intens_img)
        res2 = np.abs(res[:-20])
        print(img_lab)
        print(np.amax(res), np.amin(res))
        indx = np.where(res2==np.amax(res2))[0][0]
        print(smas_img[indx], res[indx], 'r, max')
        print(np.percentile(res, [16., 50., 84.]))
        ax[1].plot(smas_img, res, clr + 'o', markersize=4)

    ylab = 'Mean counts along semi-major axis'
    if es:
        ylab = 'Mean counts (electrons) along semi-major axis'

    # Plot line at y (residual) = 0 in lower panel
    ax[1].axhline(y=0., color='k', ls='--')

    ax[0].set_ylabel(ylab)
    ax[1].set_ylabel(r'Residual [(data - model) / data]')
    ax[0].legend()
    ax[0].set_title(r'UGC 2698 surface brightness profile')
    plt.show()
    plt.clf()


def sb_skyrm(imgs, table_imgs, table_mods, masks, outs, just_outs_skysub, sky=337.5, modtype=None):

    for i in range(len(table_imgs)):
        if not os.path.exists(just_outs_skysub[i]):
            with fits.open(outs[i]) as hdu:
                print(hdu.info())
                # print(hdu_h[0].header)
                hdr = hdu[0].header
                if modtype == 'akin':
                    data = hdu[0].data - sky
                    hdr['HISTORY'] = 'subtracted 337.5 from model image to remove the sky'
                    fits.writeto(just_outs_skysub[i], data, hdr)
                elif modtype is None:
                    data = hdu[0].data
                    data1 = hdu[1].data  # input image
                    data2 = hdu[2].data - sky  # model
                    data3 = hdu[3].data  # residual
                    hdr['HISTORY'] = 'subtracted 337.5 from model image to remove the sky'
                    fits.writeto(just_outs_skysub[i], data2, hdr)
        if not os.path.exists(table_imgs[i]):
            sb_profile(imgs[i], table_imgs[i], mask=masks[i])  # image
            print('When finished, copy-paste what was printed in the terminal into the newly created table file here')
            print('copy-paste from "# Semi-" down to (NOT including) the line with "CPU seconds"')
            print('Then, re-run this script (out_galfit.py)')
            print('This table is ' + table_imgs[i])
        else:
            pass
        if not os.path.exists(table_mods[i]):
            sb_profile(just_outs_skysub[i], table_mods[i])  # image
            print('When finished, copy-paste what was printed in the terminal into the newly created table files here')
            print('copy-paste from "# Semi-" down to (NOT including) the line with "CPU seconds"')
            print('Then, re-run this script (out_galfit.py)')
            print('This table is ' + table_mods[i])
            print(oops)
        else:
            pass


def compare_modsbs(mod_labs=None, colors=None, table_mods=None, pix_scales=None, psf=False, es=True, xlims=[0.05, 4.]):
    """

    :param mod_labs: list of labels for the models
    :param colors: list of colors to use for the different cases (regH+reg mask, regH+dust, ahcorr+reg)
    :param table_mods: list of table_mod files
    :param pix_scales: list of pixel scales for each model [arcsec/pix]
    :param psf: if showing models with PSF (AGN) included, psf=True; else, showing models with Akin's table included
    :param es: if img in units of electrons per sec, es=False; if img in units of electrons, es=True
    :return:
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    # fig.subplots_adjust(hspace=0.01)
    ax.set_xlabel('arcsec')
    if es:
        ax.set_ylim(2e3, 3e6)  # (200., 3. * 10 ** 6)  # 2e-3, 3.*10**3  # 8e4, 3e6
    else:
        ax.set_ylim(8e-2, 3.*10**3)  # 2e-3, 3.*10**3
    ax.set_xlim(xlims[0], xlims[1])  # (0.5, 400.)

    for i in range(len(table_mods)):
        table_mod = table_mods[i]
        mod_lab = mod_labs[i]  # r'GALFIT Model (profile from Ellipse task)'
        clr = colors[i]

        # Create 1D surface brightness profile arrays for model from tables output by Ellipse task
        smas_mod, intens_mod = tab_to_sb(table_mod)
        smas_mod = np.asarray(smas_mod) * pix_scales[i]

        # Interpolate the model to have the same sma values as the image, so I can take residuals!
        # f = interpolate.interp1d(smas_mod, intens_mod, fill_value="extrapolate")  # x, y

        # PLOT 1D profiles
        if not psf and i == len(table_mods) - 1:  # if including Akin table & on the last iter (i.e. on Akin's table)
            ax.loglog(smas_mod, intens_mod, clr + ':', label=mod_lab, linewidth=2)
        else:
            ax.loglog(smas_mod, intens_mod, clr + '-', label=mod_lab, linewidth=2)

    ylab = 'Mean counts along semi-major axis'
    if es:
        ylab = 'Mean counts (electrons) along semi-major axis'

    ax.set_ylabel(ylab)
    ax.legend()
    ax.set_title(r'UGC 2698 surface brightness profile')
    plt.show()
    plt.clf()


###
ahe_skysub = base + 'ugc2698_f160w_pxfr075_pxs010_ahcorr_rapidnuc_sci_nonan_e_skysub.fits'
re_skysub = base + 'ugc2698_f160w_pxfr075_pxs010_drz_rapidnuc_sci_nonan_e_skysub.fits'
re_akin_skysub = 'ugc2698_f160w_pxfr075_pxs010_drz_rapidnuc_sci_nonan_e_akintexp_skysub.fits'

ahe = base + 'ugc2698_f160w_pxfr075_pxs010_ahcorr_rapidnuc_sci_nonan_e.fits'
ahe_masked = base + 'ugc2698_f160w_pxfr075_pxs010_ahcorr_rapidnuc_sci_nonan_e_mem.fits'
re = base + 'ugc2698_f160w_pxfr075_pxs010_drz_rapidnuc_sci_nonan_e.fits'
re_masked = base + 'ugc2698_f160w_pxfr075_pxs010_drz_rapidnuc_sci_nonan_e_cm.fits'
rre_masked = base + 'ugc2698_f160w_pxfr075_pxs010_drz_rapidnuc_sci_nonan_e_mem.fits'
table_img_ahe = 'out_tab_ahe.txt'
table_img_ahe_skyrm = 'out_tab_ahe_skyrm.txt'
table_img_re = 'out_tab_re.txt'
table_img_re_skyrm = 'out_tab_reskyrm.txt'
table_img_re_cm = 'out_tab_re_cm.txt'  # cmasked re
table_img_re_cm_skyrm = 'out_tab_re_cm_skyrm.txt'
table_img_reakin_skyrm = 'out_tab_reakin_skyrm.txt'
table_ben = base + 'out_tab_mge_Ben_fixed_mod.txt'

cmask = 'f160w_combinedmask_px010.fits'  # edge, regular, and dust mask
qb = None

table_mod_rre = base + 'out_tab_rre_linear_pf001_mod.txt'  # _galfit101.txt for the bad constraintfile run
table_mod_rrepsf = base + 'out_tabpsf_rre_linear_pf001_mod.txt'
table_mod_rrenewsky = 'out_tab_rrenewsky_linear_pf001_mod.txt'
table_mod_rrenewsky_skyrm = 'out_tab_rrenewsky_skyrm_linear_pf001_mod.txt'
table_mod_rhe = base + 'out_tab_rhe_linear_pf001_mod_galfit105.txt'
table_mod_rhepsf = base + 'out_tabpsf_rhe_linear_pf001_mod.txt'
table_mod_rhenewsky = base + 'out_tab_rhenewsky_mod.txt'
table_mod_rhenewsky_skyrm = base + 'out_tab_rhenewsky_skyrm_mod.txt'
# table_mod_rhe = base + 'out_tab_rhe_linear_pf001_mod_badfit.txt' (inner gaussian fixed to 0.01)
table_mod_rhe_restart = base + 'out_tab_rhe_galfit106restart.txt'  # start from the best-fit positions from the cvgd rre run)
table_mod_ahe = base + 'out_tab_ahe_linear_pf001_mod.txt'
table_mod_ahepsf = base + 'out_tabpsf_ahe_linear_pf001_mod.txt'
table_mod_ahenewsky = base + 'out_tab_ahenewsky_linear_pf001_mod.txt'
table_mod_ahenewsky_skyrm = base + 'out_tab_ahenewsky_skyrm_linear_pf001_mod.txt'
table_akin = base + 'out_tab_akin_fixedmod_zp24.txt'
table_akin_dt = base + 'out_tab_akin_fixedmod_zp24_dt.txt'
table_akin_t898 = base + 'out_tab_akin_fixedmod_zp24_t898.txt'
table_akin_corr = base + 'out_tab_akin_fixedmod_zp24_corr.txt'
table_akin_corrdt = base + 'out_tab_akin_fixedmod_zp24_corrdt.txt'
table_akin_corrdtt2 = base + 'out_tab_akin_fixedmod_zp24_corrdtt2.txt'
table_akin_corrdt_skyrm = base + 'out_tab_akin_fixedmod_zp24_corrdt_skyrm.txt'
table_akin_corrdt_es = base + 'out_tab_akin_fixedmod_zp24_corrdt_es.txt'
table_ben_fix = base + 'out_tab_ben_fixedmod_zp24.txt'

output_block_rre = base + 'galfit_out_u2698_rre_linear_pf001_zp24.fits'
output_block_rrepsf = base + 'galfit_outpsf_u2698_rre_linear_pf001_zp24.fits'
output_block_rrenewsky = base + 'galfit_out_u2698_rrenewsky_linear_pf001_zp24.fits'
# output_block_rhe = base + 'galfit_out_u2698_rhe_linear_pf001_zp24.fits'
output_block_rhe_restart = base + 'galfit_out_u2698_rhe_galfit106restart.fits'
output_block_rhepsf = base + 'galfit_outpsf_u2698_rhe_linear_pf001_zp24.fits'
output_block_rhenewsky = base + 'galfit_out_u2698_rhenewsky.fits'
output_block_ahe = base + 'galfit_out_u2698_ahe_linear_pf001_zp24.fits'
output_block_ahepsf = base + 'galfit_outpsf_u2698_ahe_linear_pf001_zp24.fits'
output_block_ahenewsky = base + 'galfit_out_u2698_ahenewsky_linear_pf001_zp24.fits'
output_block_akin = base + 'galfit_out_u2698_akin_fixedmod_zp24.fits'
output_block_akin_dt = base + 'galfit_out_u2698_akin_fixedmod_zp24_dt.fits'
output_block_akin_t898 = base + 'galfit_out_u2698_akin_fixedmod_zp24_t898.fits'
output_block_akin_corr = base + 'galfit_out_u2698_akin_fixedmod_zp24_corr.fits'
output_block_akin_corrdt = base + 'galfit_out_u2698_akin_fixedmod_zp24_corrdt.fits'
output_block_akin_corrdtt2 = base + 'galfit_out_u2698_akin_fixedmod_zp24_corrdtt2.fits'
output_block_akin_corrdt_es = base + 'galfit_out_u2698_akin_fixedmod_zp24_corrdt_es.fits'
output_block_ben_fix = base + 'galfit_out_u2698_ben_fixedmod_zp24.fits'

just_output_rre = base + 'galfit_out_u2698_rre_linear_pf001_zp24_model.fits'
just_output_rrepsf = base + 'galfit_outpsf_u2698_rre_linear_pf001_zp24_model.fits'
just_output_rrenewsky = base + 'galfit_out_u2698_rrenewsky_linear_pf001_zp24_model.fits'
just_output_rhe = base + 'galfit_out_u2698_rhe_linear_pf001_zp24_model.fits'
just_output_rhe_restart = base + 'galfit_out_u2698_rhe_galfit106restart_model.fits'
just_output_rhepsf = base + 'galfit_outpsf_u2698_rhe_linear_pf001_zp24_model.fits'
just_output_rhenewsky = base + 'galfit_out_u2698_rhenewsky_model.fits'
just_output_ahe = base + 'galfit_out_u2698_ahe_linear_pf001_zp24_model.fits'
just_output_ahepsf = base + 'galfit_outpsf_u2698_ahe_linear_pf001_zp24_model.fits'
just_output_ahenewsky = base + 'galfit_out_u2698_ahenewsky_linear_pf001_zp24_model.fits'
just_output_akin = base + 'galfit_out_u2698_akin_fixedmod_zp24_model.fits'
just_output_akin_dt = base + 'galfit_out_u2698_akin_fixedmod_zp24_modelz_dt.fits'
just_output_akin_t898 = base + 'galfit_out_u2698_akin_fixedmod_zp24_t898_model.fits'
just_output_akin_corr = base + 'galfit_out_u2698_akin_fixedmod_zp24_corr_model.fits'
just_output_akin_corrdt = base + 'galfit_out_u2698_akin_fixedmod_zp24_corrdt_model.fits'
just_output_akin_corrdtt2 = base + 'galfit_out_u2698_akin_fixedmod_zp24_corrdtt2_model.fits'
just_output_akin_corrdt_es = base + 'galfit_out_u2698_akin_fixedmod_zp24_corrdt_es_model.fits'
just_output_ben_fix = base + 'galfit_out_u2698_ben_fixedmod_zp24_model.fits'

# sb_prof_compare(re, output_block_akin_corrdt_es, just_output_akin_corrdt_es, table_img_re, table_akin_corrdt_es, qbounds=qb, es=True, modtype='akin')
# sb_prof_compare(re, output_block_akin_corr, just_output_akin_corr, table_img_re, table_akin_corr, qbounds=qb, es=True, modtype='akin')
# sb_prof_compare(re, output_block_akin_corrdt, just_output_akin_corrdt, table_img_re, table_akin_corrdt, qbounds=qb, es=True, modtype='akin')
# sb_prof_compare(re, output_block_akin_dt, just_output_akin_dt, table_img_re, table_akin_dt, qbounds=qb, es=True, modtype='akin')

# ''' #### PGC 11179 PA free, n9
plot3(gf+'p11179/galfit_out_p11179_reg_linear_n9_pafree.fits', clipreg=[1300,1500,1370,1570], indivs=False, cliprange=True)
sb_prof_compare(fj+'PGC11179_F160W_drz_sci.fits', gf+'p11179/galfit_out_p11179_reg_linear_n9_pafree.fits',
                gf+'p11179/galfit_out_p11179_reg_linear_n9_pafree_model.fits',
                gf+'p11179/out_tab_p11179_reg_linear_n9_pafree_img.txt',
                gf + 'p11179/out_tab_p11179_reg_linear_n9_pafree_mod.txt', scale=0.06, modtype='reg', es=True,
                mask=None, glx='PGC 11179')
print(oop)
#### PGC 11179 PA free '''
''' #### PGC 11179 PA free
plot3(gf+'p11179/galfit_out_p11179_reg_linear_pafree.fits', clipreg=[1300,1500,1370,1570], indivs=False, cliprange=True)
sb_prof_compare(fj+'PGC11179_F160W_drz_sci.fits', gf+'p11179/galfit_out_p11179_reg_linear_pafree.fits',
                gf+'p11179/galfit_out_p11179_reg_linear_pafree_model.fits',
                gf+'p11179/out_tab_p11179_reg_linear_pafree_img.txt',
                gf + 'p11179/out_tab_p11179_reg_linear_pafree_mod.txt', scale=0.06, modtype='reg', es=True, mask=None,
                glx='PGC 11179')
print(oop)
#### PGC 11179 PA free '''
''' #### PGC 11179 #
sb_prof_compare(fj+'PGC11179_F160W_drz_sci.fits', gf+'p11179/galfit_out_p11179_reg_linear.fits',
                gf+'p11179/galfit_out_p11179_reg_linear_model.fits', gf+'p11179/out_tab_p11179_reg_linear_img.txt',
                gf + 'p11179/out_tab_p11179_reg_linear_mod.txt', scale=0.06, modtype='reg', es=True, mask=None,
                glx='PGC 11179')
print(oop)
plot3(gf+'p11179/galfit_out_p11179_reg_linear.fits', clipreg=[1300,1500,1370,1570], indivs=False, cliprange=True)
# clipreg [y,y,x,x]
print(oop)
#### PGC 11179 '''

imgs = [re, re, ahe, re_akin_skysub]
masks = [None, cmask, None, None]
outs = [output_block_rrenewsky, output_block_rhenewsky, output_block_ahenewsky, output_block_akin_corrdt]
just_outs_skysub = [just_output_rrenewsky, just_output_rhenewsky, just_output_ahenewsky, just_output_akin_corrdt]

table_imgs = [table_img_re_skyrm, table_img_re_cm_skyrm, table_img_ahe_skyrm, table_img_reakin_skyrm]
table_imgs_psf = [table_img_re, table_img_re_cm, table_img_ahe]
table_mods = [table_mod_rrenewsky_skyrm, table_mod_rhenewsky_skyrm, table_mod_ahenewsky_skyrm, table_akin_corrdtt2]
table_mods_psf = [table_mod_rrepsf, table_mod_rhepsf, table_mod_ahepsf]
colors = ['k', 'm', 'b', 'r']
colors_psf = ['k', 'm', 'b']
img_labs = [r'Data (reg H + reg mask)', r'Data (reg H + dust mask data)', r'Data (ahcorr + reg mask)', None]
img_labs_psf = [r'Data (reg H + reg mask)', r'Data (reg H + dust mask data)', r'Data (ahcorr + reg mask)']
mod_labs = [r'GALFIT model (reg H + reg mask)', r'GALFIT model (reg H + dust mask)',
            'GALFIT model (ahcorr + reg mask)', r"Akin's model (reg H + reg mask; different t_exp and scale)"]
mod_labs_psf = [r'GALFIT model (reg H + reg mask, psf)', r'GALFIT model (reg H + dust mask data, psf)',
                'GALFIT model (ahcorr + reg mask, psf)']
pix_scales = [0.1, 0.1, 0.1, 0.06]
xlims = [0.01, 20]  #[0.01, 10]  # [0.05, 4.]

#sb_prof_compare(re, output_block_akin, just_output_akin, table_img_re, table_akin, qbounds=qb, es=True, scale=0.06,
#                modtype='akin')
#print(oop)


#sb_prof_compare(re, output_block_akin_corrdtt2, just_output_akin_corrdtt2, table_img_re, table_akin_corrdtt2,
#                qbounds=qb, es=True, modtype='akin')

#compare_modsbs(mod_labs, colors, table_mods, pix_scales, psf=False, es=True, xlims=xlims)
#print(oop)

#sb_skyrm(imgs, table_imgs, table_mods, masks, outs, just_outs_skysub, sky=337.5, modtype=None)
#print(oop)

#compare_modsbs(mod_labs, colors, table_mods, pix_scales, psf=False, es=True, xlims=xlims)
#print(oop)
# compare_sbs(img_labs_psf, mod_labs_psf, colors_psf, table_imgs_psf, table_mods_psf, psf=True, es=True)
compare_sbs(img_labs, mod_labs, colors, table_imgs, table_mods, es=True)
print(oop)

sb_prof_compare(re, output_block_akin, just_output_akin, table_img_re, table_akin, qbounds=qb, es=True, modtype='akin')
print(oop)
sb_prof_compare(re, output_block_ben_fix, just_output_ben_fix, table_img_re, table_ben_fix, qbounds=qb, es=True, modtype='akin')
print(oop)

# ahe newsky
sb_compare3(ahe, output_block_ahenewsky, just_output_ahenewsky, table_img_ahe, table_mod_ahenewsky, table_ben, qbounds=qb, es=True)
sb_prof_compare(ahe_masked, output_block_ahenewsky, just_output_ahenewsky, table_img_ahe, table_mod_ahenewsky, qbounds=qb)
plot3(output_block_ahenewsky, clipreg=[450,550,825,925], indivs=False, cliprange=True)  # clipreg=30, =60
print(oop)

# rhe newsky (finally got a decent(?) rhe image using the mask input in the Ellipse task)
sb_compare3(re, output_block_rhenewsky, just_output_rhenewsky, table_img_re_cm, table_mod_rhenewsky, table_ben, qbounds=qb,
            es=True, mask='f160w_combinedmask_px010.fits')
sb_prof_compare(re, output_block_rhenewsky, just_output_rhenewsky, table_img_re_cm, table_mod_rhenewsky,
                qbounds=qb, mask='f160w_combinedmask_px010.fits')
plot3(output_block_rhenewsky, clipreg=[450,550,825,925], indivs=False, cliprange=True)  # , masked_img=re_masked
print(oop)

# rre newsky
sb_compare3(re, output_block_rrenewsky, just_output_rrenewsky, table_img_re, table_mod_rrenewsky, table_ben, qbounds=qb, es=True)
sb_prof_compare(rre_masked, output_block_rrenewsky, just_output_rrenewsky, table_img_re, table_mod_rrenewsky, qbounds=qb)
plot3(output_block_rrenewsky, clipreg=[450,550,825,925], indivs=False, cliprange=True)#, masked_img=rre_masked)
print(oop)

# ahe, PSF
sb_compare3(ahe, output_block_ahepsf, just_output_ahepsf, table_img_ahe, table_mod_ahepsf, table_ben, qbounds=qb, es=True)
sb_prof_compare(ahe_masked, output_block_ahepsf, just_output_ahepsf, table_img_ahe, table_mod_ahepsf, qbounds=qb)
plot3(output_block_ahepsf, clipreg=[450,550,825,925], indivs=False, cliprange=True)  # clipreg=30, =60

# rhe, PSF
sb_compare3(re, output_block_rhepsf, just_output_rhepsf, table_img_re, table_mod_rhepsf, table_ben, qbounds=qb, es=True)
sb_prof_compare(re_masked, output_block_rhepsf, just_output_rhepsf, table_img_re, table_mod_rhepsf, qbounds=qb)
plot3(output_block_rhepsf, clipreg=[450,550,825,925], indivs=False, cliprange=True)  # , masked_img=re_masked

# rre, PSF (NOTE: input X0, Y0 guess in terminal to find object in Ellipse; used 880.8350 & 491.0701 from galfit.113)
sb_compare3(re, output_block_rrepsf, just_output_rrepsf, table_img_re, table_mod_rrepsf, table_ben, qbounds=qb, es=True)
sb_prof_compare(rre_masked, output_block_rrepsf, just_output_rrepsf, table_img_re, table_mod_rrepsf, qbounds=qb)
plot3(output_block_rrepsf, clipreg=[450,550,825,925], indivs=False, cliprange=True)#, masked_img=rre_masked)

# rre
sb_compare3(re, output_block_rre, just_output_rre, table_img_re, table_mod_rre, table_ben, qbounds=qb, es=True)
sb_prof_compare(rre_masked, output_block_rre, just_output_rre, table_img_re, table_mod_rre, qbounds=qb)
plot3(output_block_rre, clipreg=[450,550,825,925], indivs=False, cliprange=True)#, masked_img=rre_masked)

# rhe
sb_compare3(re_masked, output_block_rhe_restart, just_output_rhe_restart, table_img_re_cm, table_mod_rhe_restart,
            table_ben, qbounds=qb, es=True)
sb_compare3(re, output_block_rhe_restart, just_output_rhe_restart, table_img_re, table_mod_rhe_restart,
            table_ben, qbounds=qb, es=True)
sb_prof_compare(re_masked, output_block_rhe_restart, just_output_rhe_restart, table_img_re, table_mod_rhe_restart,
                qbounds=qb)
plot3(output_block_rhe_restart, clipreg=[450,550,825,925], indivs=False, cliprange=True)  # , masked_img=re_masked
#sb_compare3(re, output_block_rhe, just_output_rhe, table_img_re, table_mod_rhe, table_ben, qbounds=qb, es=True)
#sb_prof_compare(re_masked, output_block_rhe, just_output_rhe, table_img_re, table_mod_rhe, qbounds=qb)
#plot3(output_block_rhe, clipreg=[450,550,825,925], indivs=False, cliprange=True)  # clipreg=30, =60

# ahe
sb_compare3(ahe, output_block_ahe, just_output_ahe, table_img_ahe, table_mod_ahe, table_ben, qbounds=qb, es=True)
sb_prof_compare(ahe_masked, output_block_ahe, just_output_ahe, table_img_ahe, table_mod_ahe, qbounds=qb)
plot3(output_block_ahe, clipreg=[450,550,825,925], indivs=False, cliprange=True)  # clipreg=30, =60

'''  #
# NEW GALFIT OUTPUT (using counts instead of cps, with a linear fit from MGE fit sectors)
ahcorr = base + 'ugc2698_f160w_pxfr075_pxs010_ahcorr_rapidnuc_sci_nonan_counts.fits'  # image that was input into galfit
masked_ahcorr = base + 'ugc2698_f160w_pxfr075_pxs010_ahcorr_rapidnuc_sci_nonan_counts_maskedgemask.fits'
output_block = base + 'galfit_out_ahcorr_linear_pf001_counts_zp24.fits'
just_output = base + 'galfit_out_ahcorr_linear_pf001_counts_zp24_model.fits'
table_img = base + 'out_tab_ahcorr_linear_pf001_counts_img.txt'
table_mod = base + 'out_tab_ahcorr_linear_pf001_counts_mod.txt'
# table_ben = base + 'wiki_Ben_ahcorr_mge_out.txt'
table_ben = base + 'out_tab_mge_Ben_fixed_mod.txt'
qb = None

sb_compare3(ahcorr, output_block, just_output, table_img, table_mod, table_ben, qbounds=qb, sky=None, counts=True)
sb_prof_compare(masked_ahcorr, output_block, just_output, table_img, table_mod, qbounds=qb, sky=None)
plot3(output_block, clipreg=[450,550,825,925], indivs=False, cliprange=True)  # clipreg=30, =60
# out, table_img, table_mod, qbounds=[0., 1.]
print(oop)
###
# '''  #

'''  #
###
# NEW GALFIT OUTPUT (using counts instead of cps)
ahcorr = base + 'ugc2698_f160w_pxfr075_pxs010_ahcorr_rapidnuc_sci_nonan_counts.fits'  # image that was input into galfit
masked_ahcorr = base + 'ugc2698_f160w_pxfr075_pxs010_ahcorr_rapidnuc_sci_nonan_counts_maskedgemask.fits'
output_block = base + 'galfit_out_ahcorr_n10_pf001_counts_zp24.fits'
just_output = base + 'galfit_out_ahcorr_n10_pf001_counts_zp24_model.fits'
table_img = base + 'out_tab_ahcorr_n10_pf001_counts_img.txt'
table_mod = base + 'out_tab_ahcorr_n10_pf001_counts_mod.txt'
# table_ben = base + 'wiki_Ben_ahcorr_mge_out.txt'
table_ben = base + 'out_tab_mge_Ben_fixed_mod.txt'
qb = None

sb_compare3(ahcorr, output_block, just_output, table_img, table_mod, table_ben, qbounds=qb, sky=None, counts=True)
sb_prof_compare(masked_ahcorr, output_block, just_output, table_img, table_mod, qbounds=qb, sky=None)
plot3(output_block, clipreg=[450,550,825,925], indivs=False, cliprange=True)  # clipreg=30, =60
# out, table_img, table_mod, qbounds=[0., 1.]
print(oop)
###
# '''  #

'''  #
###
# NEW GALFIT OUTPUT (since corrections to mge_fit_mine.py and convert_mge.py were made)
ahcorr = base + 'ugc2698_f160w_pxfr075_pxs010_ahcorr_rapidnuc_sci_nonan_cps.fits'  # image that was input into galfit
regH = base + 'ugc2698_f160w_pxfr075_pxs010_drz_rapidnuc_sci_nonan_cps.fits'  # image that was input into galfit
masked_ahcorr = base + 'ugc2698_f160w_pxfr075_pxs010_ahcorr_rapidnuc_sci_nonan_cps_maskedgemask.fits'
output_block = base + 'galfit_out_ahcorr_n10_05_pf001_cps_zp24.fits'
just_output = base + 'galfit_out_ahcorr_n10_05_pf001_cps_zp24_model.fits'
table_img = base + 'out_tab_ahcorr_n10_05_pf001_cps_img.txt'
table_mod = base + 'out_tab_ahcorr_n10_05_pf001_cps_mod.txt'
# table_ben = base + 'wiki_Ben_ahcorr_mge_out.txt'
table_ben = base + 'out_tab_mge_Ben_fixed_mod.txt'
qb = [0.5, 1.0]

sb_compare3(ahcorr, output_block, just_output, table_img, table_mod, table_ben, qbounds=qb, sky=None)
sb_prof_compare(masked_ahcorr, output_block, just_output, table_img, table_mod, qbounds=qb, sky=None)
plot3(output_block, clipreg=[450,550,825,925], indivs=False, cliprange=True)  # clipreg=30, =60
# out, table_img, table_mod, qbounds=[0., 1.]
print(oop)
###
# '''  #

'''  #
# GALFIT OUTPUT: TAKE THE OUTPUT FROM RUNNING BEN'S galfit.parameters.txt IN GALFIT, THEN RUN THAT OUTPUT IN GALFIT
# AS A FIXED MODEL
masked_input_img = base + 'ugc2698_f160w_pxfr075_pxs010_ahcorr_rapidnuc_sci_nonan_maskedgemask.fits'
output_img = base + 'galfit_mge_ben_fixed.fits'
just_output = base + 'galfit_mge_ben_fixed_model.fits'
table_img = base + 'out_tab_mge_Ben_fixed_img.txt'
table_mod = base + 'out_tab_mge_Ben_fixed_mod.txt'
sb_prof_compare(masked_input_img, output_img, just_output, table_img, table_mod, modtype='akin', qbounds=[0.4, 1.0])
print(oop)

# '''  #

'''  #
# GALFIT OUTPUT: TAKE THE OUTPUT FROM RUNNING AKIN'S MGE (CORRECT CONVERSION) IN GALFIT, THEN RUN THAT OUTPUT IN GALFIT
# AS A FIXED MODEL
# masked_input_img = base + 'ugc2698_f160w_pxfr075_pxs010_ahcorr_rapidnuc_sci_nonan_maskedgemask.fits'
input_img = base + 'ugc2698_f160w_pxfr075_pxs010_drz_rapidnuc_sci_no0.fits'
output_img = base + 'galfit_akin_mge_fixedmodel_out_corrected.fits'
just_output = base + 'galfit_akin_mge_fixedmodel_out_corrected_model.fits'
table_img = base + 'out_tab_akin_fixedmodel_corrected_img.txt'
table_mod = base + 'out_tab_akin_fixedmodel_corrected_mod.txt'
sb_prof_compare(input_img, output_img, just_output, table_img, table_mod, modtype='akin', qbounds=[0.4, 1.0])
# sb_prof_compare(masked_input_img, output_img, just_output, table_img, table_mod, modtype='akin', qbounds=[0.4, 1.0])
print(oop)

# '''  #

'''  #
# GALFIT OUTPUT FOR CASE 1: AKIN'S MGE (CORRECTED: FIXED MODEL RUN IN GALFIT JUST TO CREATE THE MODEL IMAGE)
out = base + 'galfit_akin_mge_fixedmodel_out_corrected.fits'  # _zp25
input_img = base + 'ugc2698_f160w_pxfr075_pxs010_drz_rapidnuc_sci_no0.fits'
output_img = base + 'galfit_akin_mge_fixedmodel_out_corrected.fits'  # fits file output by galfit
just_output = base + 'galfit_akin_mge_fixedmodel_out_corrected_model.fits'  # new file to store just the model from galfit
table_img = base + 'out_tab_akin_fixedmodel_corrected_img.txt'  # new file to store table output from Ellipse task on the image
table_mod = base + 'out_tab_akin_fixedmodel_corrected_mod.txt'  # new file to store table output from Ellipse task on the model
# table_ben = base + 'wiki_Ben_ahcorr_mge.txt'  # file storing Ben's MGE
table_ben = base + 'out_tab_mge_Ben_fixed_mod.txt'
# plot3(out, clipreg=[450,550,825,925], indivs=False, cliprange=True)  # clipreg=30, =60
sb_compare3(input_img, output_img, just_output, table_img, table_mod, table_ben, qbounds=None, sky=None, modtype='akin')
sb_prof_compare(input_img, output_img, just_output, table_img, table_mod, qbounds=None, sky=None, modtype='akin')
# galfit_akin_mge_fixedmodel_out.fits
# '''  #

'''  #
# GALFIT OUTPUT FOR CASE 2: MGE using regular H-band, with dust mask; q>=0.4
# Note: n10 used MGE with fixed num=10, returned 9 nonzero gaussians
out = base + 'galfit_regH_combmask_mge_n10_q04_zp25.fits'
input_img = base + 'ugc2698_f160w_pxfr075_pxs010_drz_rapidnuc_sci_no0.fits'  # image that was input into galfit
output_img = base + 'galfit_regH_combmask_mge_n10_q04_zp25.fits'  # fits file output by galfit
just_output = base + 'galfit_regH_combmask_mge_n10_q04_zp25_model.fits'  # new file to store just the model from galfit
table_img = base + 'out_tab_regH_combmask_n10_q04_img.txt'  # new file to store table output from Ellipse task on the image
table_mod = base + 'out_tab_regH_combmask_n10_q04_mod.txt'  # new file to store table output from Ellipse task on the model
# table_ben = base + 'wiki_Ben_ahcorr_mge.txt'  # file storing Ben's MGE
table_ben = base + 'out_tab_mge_Ben_fixed_mod.txt'
plot3(out, clipreg=[450,550,825,925], indivs=False, cliprange=True)  # clipreg=30, =60
sb_compare3(input_img, output_img, just_output, table_img, table_mod, table_ben, qbounds=[0.4, 1.0], sky=None)
sb_prof_compare(input_img, output_img, just_output, table_img, table_mod, qbounds=[0.4, 1.0], sky=None)

print(oop)
# '''  #

'''  #
# GALFIT OUTPUT FOR CASE 2: MGE using regular H-band, with dust mask, q>=0.6
# Note: n10 used MGE with fixed num=10, returned 9 nonzero gaussians
# galfit_out = 'galfit.73'
out = base + 'galfit_regH_combmask_mge_n10_q06_zp25.fits'
input_img = base + 'ugc2698_f160w_pxfr075_pxs010_drz_rapidnuc_sci_no0.fits'  # image that was input into galfit
output_img = base + 'galfit_regH_combmask_mge_n10_q06_zp25.fits'  # fits file output by galfit
just_output = base + 'galfit_regH_combmask_mge_n10_q06_zp25_model.fits'  # new file to store just the model from galfit
table_img = base + 'out_tab_regH_combmask_n10_q06_img.txt'  # new file to store table output from Ellipse task on the image
table_mod = base + 'out_tab_regH_combmask_n10_q06_mod.txt'  # new file to store table output from Ellipse task on the model
# table_ben = base + 'wiki_Ben_ahcorr_mge.txt'  # file storing Ben's MGE
table_ben = base + 'out_tab_mge_Ben_fixed_mod.txt'
sb_compare3(input_img, output_img, just_output, table_img, table_mod, table_ben, qbounds=[0.6, 1.0], sky=None)
sb_prof_compare(input_img, output_img, just_output, table_img, table_mod, qbounds=[0.6, 1.0], sky=None)
plot3(out, clipreg=[450,550,825,925], indivs=False, cliprange=True)  # clipreg=30, =60

print(oop)
# '''  #

###

'''  #
# GALFIT OUTPUT FOR CASE 3: MGE using dust-corrected H-band, no dust mask (just regular mask), lower q bound 0.4
# out = base + 'galfit_ahcorr_mge_n10_055_zp25.fits'  # 'galfit_mge_055_zp25.fits'
masked_input_img = base + 'ugc2698_f160w_pxfr075_pxs010_ahcorr_rapidnuc_sci_nonan_maskedgemask.fits'
output_img = base + 'galfit_ahcorr_mge_n11_04_zp25.fits'
just_output = base + 'galfit_ahcorr_mge_n11_04_zp25_model.fits'
table_img = base + 'out_tab_ahcorr_n11_04_img.txt'
table_mod = base + 'out_tab_ahcorr_n11_04_mod.txt'
# table_ben = base + 'wiki_Ben_ahcorr_mge_out.txt'
table_ben = base + 'out_tab_mge_Ben_fixed_mod.txt'
sb_compare3(masked_input_img, output_img, just_output, table_img, table_mod, table_ben, qbounds=[0.4, 1.0])
sb_prof_compare(masked_input_img, output_img, just_output, table_img, table_mod, qbounds=[0.4, 1.0])
plot3(output_img, clipreg=[450,550,825,925], indivs=False, cliprange=True)
print(oop)
# out, table_img, table_mod, qbounds=[0., 1.]
# ''' #

''' #
# GALFIT OUTPUT FOR CASE 3: MGE using dust-corrected H-band, no dust mask (just regular mask)
# out = base + 'galfit_ahcorr_mge_n10_055_zp25.fits'  # 'galfit_mge_055_zp25.fits'
masked_input_img = base + 'ugc2698_f160w_pxfr075_pxs010_ahcorr_rapidnuc_sci_nonan_maskedgemask.fits'
output_img = base + 'galfit_ahcorr_mge_n10_055_zp25.fits'
just_output = base + 'galfit_ahcorr_mge_n10_055_zp25_model.fits'
table_img = base + 'out_tab_ahcorr_n10_img.txt'
table_mod = base + 'out_tab_ahcorr_n10_mod.txt'
# table_ben = base + 'wiki_Ben_ahcorr_mge.txt'
table_ben = base + 'out_tab_mge_Ben_fixed_mod.txt'
sb_compare3(masked_input_img, output_img, just_output, table_img, table_mod, table_ben, qbounds=[0.55, 1.0])
sb_prof_compare(masked_input_img, output_img, just_output, table_img, table_mod, qbounds=[0.55, 1.0])
# out, table_img, table_mod, qbounds=[0., 1.]
print(oop)
# '''  #

plot3(out, clipreg=[450,550,825,925], indivs=False, cliprange=True)  # clipreg=30, =60
# [400,600,800,1000]
