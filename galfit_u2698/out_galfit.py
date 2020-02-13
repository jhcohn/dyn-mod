from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import AxesGrid
import os
from scipy import interpolate


base = '/Users/jonathancohn/Documents/dyn_mod/galfit_u2698/'
out = base + 'galfit_regH_combmask_mge_n10_q06_zp25.fits'
# out = base + 'galfit_ahcorr_mge_n10_055_zp25.fits'  # 'galfit_mge_055_zp25.fits'
# 'galfit_mge_055_zp24_myconstraint_reset_take2.fits'
# 'galfit_mge_055_zp24_cadj.fits'  # 'galfit_out_mge_055_zp24_myconstraint.fits'
# 'galfit_mge0.fits'  # 'imgblock_masked8.fits' # 'imgblock_masked3.fits'  # 'imgblock_test49.fits'
abs = 0


def sb_profile(image, out_tab):
    # run ellipse task on input galaxy and on GALFIT output (ellipse task should show 1D surf brightness)
    import pyraf
    from pyraf import iraf
    from iraf import stsdas, analysis, isophote
    from iraf import stsdas, fitting
    # print(image, out_tab)
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
                modtype=None, counts=False):
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
        sb_profile(in_filename, table_img)  # image
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

    if counts:
        ax[0].set_ylim(200., 3. * 10 ** 6)  # 2e-3, 3.*10**3
    else:
        ax[0].set_ylim(8e-2, 3.*10**3)  # 2e-3, 3.*10**3
    ax[0].set_xlim(0.5, 400.)
    ax[1].set_ylim(-0.15, 0.15)  # -0.4, 0.4)  #

    # Interpolate the model to have the same sma values as the image, so I can take residuals!
    f = interpolate.interp1d(smas_mod, intens_mod)  # x, y
    if counts:
        intens_ben = [ib * 898.467164 for ib in intens_ben]
        fben = interpolate.interp1d(smas_ben, intens_ben)  # x, y
    else:
        fben = interpolate.interp1d(smas_ben, intens_ben)  # x, y
    if sky is not None:
        intens_mod_samex = f(smas_img) - sky
        intens_ben_samex = fben(smas_img) - sky
    else:
        intens_mod_samex = f(smas_img)
        intens_ben_samex = fben(smas_img)

    # Add in Ben's MGE as well!
    '''
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
    ax[0].loglog(smas_img, total, 'r-', label=r"Ben's MGE parameters", linewidth=2)
    '''
    benlab = r"Ben's MGE parameters"
    if counts:
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

    ax[0].set_ylabel('Mean counts along semi-major axis')
    ax[1].set_ylabel(r'Residual [(data - model) / data]')
    ax[0].legend()
    if qbounds is not None:
        ax[0].set_title(r'UGC 2698 surface brightness profile, qbounds=' + str(qbounds))
    else:
        ax[0].set_title(r'UGC 2698 surface brightness profile')
    plt.show()
    plt.clf()


def sb_prof_compare(in_filename, out_filename, just_out, table_img, table_mod, qbounds=[0., 1.], sky=None,
                    modtype=None):
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
            elif modtype is None:
                data = hdu[0].data
                data1 = hdu[1].data  # input image
                data2 = hdu[2].data  # model
                data3 = hdu[3].data  # residual
                fits.writeto(just_out, data2, hdr)

    if not os.path.exists(table_img):
        sb_profile(in_filename, table_img)  # image
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

    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(8, 16))
    fig.subplots_adjust(hspace=0.01)
    # fig.text(0.04, 0.5, 'counts', va='center', rotation='vertical')
    # fig.text(0.96, 0.5, 'error (%)', va='center', rotation='vertical')
    ax[1].set_xlabel('arcsec')

    ax[0].set_xlim(0.5, 400.)
    ax[0].set_ylim(8e-2, 1e8) #3.*10**3)  # 2e-3, 3.*10**3
    ax[1].set_ylim(-0.15, 0.15)  # -0.4, 0.4)  #

    f = interpolate.interp1d(smas_mod, intens_mod)  # x, y
    if sky is not None:
        intens_mod_samex = f(smas_img) - sky
    else:
        intens_mod_samex = f(smas_img)

    ax[0].loglog(smas_img, np.array(intens_img), 'b--', label=r'Data (profile from Ellipse task)', linewidth=2)
    ax[0].loglog(smas_img, intens_mod_samex, 'r-', label=r'GALFIT Model (profile from Ellipse task)', linewidth=2)
    # smas_mod, intens_mod

    ax[1].plot(smas_img, (np.array(intens_img) - np.array(intens_mod_samex)) / (np.array(intens_img)), 'ko', markersize=4)
    ax[1].axhline(y=0., color='k', ls='--')

    ax[0].set_ylabel('Mean counts along semi-major axis')
    ax[1].set_ylabel(r'Residual [(data - model) / data]')
    ax[0].legend()
    if qbounds is not None:
        ax[0].set_title(r'UGC 2698 surface brightness profile, qbounds=' + str(qbounds))
    else:
        ax[0].set_title(r'UGC 2698 surface brightness profile')
    plt.show()
    plt.clf()


def plot3(img, normed=True, clipreg=None, indivs=False, cliprange=False):
    with fits.open(img) as hdu:
        print(hdu.info())
        # print(hdu_h[0].header)
        hdr = hdu[0].header
        data = hdu[0].data
        data1 = hdu[1].data  # input image
        data2 = hdu[2].data  # model
        data3 = hdu[3].data  # residual

    data4 = data3 / data1  # residual scaled to the input image
    dataset = [data1, data2, data3, data4]

    for dat in dataset:
        dat[np.isfinite(dat) == False] = 0.  # change infinities to 0.

    print(np.amax(dataset[1]), np.amin(dataset[1]))

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
            print(dataset[d].shape)
            if len(clipreg) == 4:
                dataset[d] = dataset[d][clipreg[0]:clipreg[1], clipreg[2]:clipreg[3]]
            else:
                dataset[d] = dataset[d][clipreg:-clipreg, clipreg:-clipreg]
            print(dataset[d].shape)

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
                print(np.amin(dataset[1]), np.amax(dataset[1]))
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


# '''  #
###
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
