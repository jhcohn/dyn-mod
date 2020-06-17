# Python 3 compatability
# from __future__ import division, print_function
# from six.moves import range

import dynesty
import pickle
#import _pickle as pickle
from dynesty import utils as dyfunc
import dynamical_model as dm

from pathlib import Path

# basic numeric setup
import numpy as np

# plotting
import matplotlib
from matplotlib import pyplot as plt

# re-defining plotting defaults
from matplotlib import rcParams
rcParams.update({'xtick.major.pad': '7.0'})
rcParams.update({'xtick.major.size': '7.5'})
rcParams.update({'xtick.major.width': '1.5'})
rcParams.update({'xtick.minor.pad': '7.0'})
rcParams.update({'xtick.minor.size': '3.5'})
rcParams.update({'xtick.minor.width': '1.0'})
rcParams.update({'ytick.major.pad': '7.0'})
rcParams.update({'ytick.major.size': '7.5'})
rcParams.update({'ytick.major.width': '1.5'})
rcParams.update({'ytick.minor.pad': '7.0'})
rcParams.update({'ytick.minor.size': '3.5'})
rcParams.update({'ytick.minor.width': '1.0'})
rcParams.update({'font.size': 15})


def my_own_xy(results, par_labels, ax_labels, quantiles, ax_lims=None, compare_err=False, comp2=False, err=1, fs=12):
    # results should be dyn_res['samples']
    roundto = 3  # 2  # 4
    fig, axes = plt.subplots(1, 2, figsize=(12, 8))  # 3 rows, 3 cols of subplots; because there are 9 free params
    labels = np.array(['xloc', 'yloc'])
    print(results.shape, 'look!')
    for i in range(2):
        bins = 5000
        chain = results[:, i]
        print(np.quantile(chain, [.16, .5, .84]), 'look')
        weight = np.ones_like(chain) * 2e-4
        axes[i].hist(chain, bins=bins, color="b", histtype="step", weights=weight)  # axes[i]
        axes[i].axvline(quantiles[i][1], color='b', ls='--')  # axes[i]
        axes[i].axvspan(quantiles[i][0], quantiles[i][2], color='b', alpha=0.25)
        axes[i].tick_params('both', labelsize=fs)
        axes[i].set_title(par_labels[i] + ': ' + str(round(quantiles[i][1], roundto)) + ' (+'
                                 + str(round(quantiles[i][2] - quantiles[i][1], roundto)) + ', -'
                                 + str(round(quantiles[i][1] - quantiles[i][0], roundto)) + ')', fontsize=fs)
        axes[i].set_xlabel(ax_labels[i], fontsize=fs)

        if compare_err:
            with open('/Users/jonathancohn/Documents/dyn_mod/param_files/Ben_A1_errors.txt') as a1:
                for line in a1:
                    cols = line.split()
                    if cols[0] == labels[i]:
                        vax = float(cols[1])
                        vax_width = float(cols[2])
            print(vax, 'vax', labels[i])
            if labels[i] == 'mbh':
                axes[i].axvline(np.log10(vax), ls='-', color='k')
                print('here', np.log10(vax), np.log10(vax - vax_width), np.log10(vax + vax_width))
                axes[i].axvspan(np.log10(vax - vax_width), np.log10(vax + vax_width), hatch='/', color='k',
                                       fill=False, alpha=0.5)
            else:
                axes[i].axvline(vax, ls='-', color='k')
                axes[i].axvspan(vax - vax_width, vax + vax_width, hatch='/', color='k', fill=False, alpha=0.5)
        elif comp2:
            with open('/Users/jonathancohn/Documents/dyn_mod/param_files/Ben_A1_sampling.txt') as a1:
                for line in a1:
                    cols = line.split()
                    if cols[0] == labels[i]:
                        vax = float(cols[1])
                        vax_width1 = float(cols[2])
                        vax_width3 = float(cols[3])
            if err == 1:
                vax_width = vax_width1
            elif err == 3:
                vax_width = vax_width3
            if labels[i] == 'mbh':
                axes[i].axvline(np.log10(vax), ls='-', color='k')
                print('here', np.log10(vax), np.log10(vax - vax_width), np.log10(vax + vax_width))
                axes[i].axvspan(np.log10(vax - vax_width), np.log10(vax + vax_width), hatch='/', color='k',
                                       fill=False, alpha=0.5)
            else:
                axes[i].axvline(vax, ls='-', color='k')
                axes[i].axvspan(vax - vax_width, vax + vax_width, hatch='/', color='k', fill=False, alpha=0.5)
        if ax_lims is not None:
            axes[i].set_xlim(ax_lims[i][0], ax_lims[i][1])
    plt.tight_layout()
    plt.show()


def table_it(things, parfiles, models, parlabels, sig=3):

    hdr = '| model | '
    hdrl = '| --- |'
    for la in range(len(parlabels)):
        hdrl += ' --- |'
        hdr += parlabels[la] + ' | '

    texlines = '| '
    lines = '| '
    for t in range(len(things)):
        print(t, len(things), len(parfiles), len(models))
        params, priors, nfree, qobs = dm.par_dicts(parfiles[t], q=True)  # get params and file names from output parfile
        mod_ins = dm.model_prep(data=params['data'], ds=params['ds'], lucy_out=params['lucy'], lucy_b=params['lucy_b'],
                                lucy_mask=params['lucy_mask'], lucy_in=params['lucy_in'], lucy_it=params['lucy_it'],
                                data_mask=params['mask'], grid_size=params['gsize'], res=params['resolution'],
                                x_std=params['x_fwhm'], y_std=params['y_fwhm'], zrange=[params['zi'], params['zf']],
                                xyerr=[params['xerr0'], params['xerr1'], params['yerr0'], params['yerr1']],
                                pa=params['PAbeam'])
        lucy_mask, lucy_out, beam, fluxes, freq_ax, f_0, fstep, input_data, noise, co_rad, co_sb = mod_ins
        vrad = None
        kappa = None
        omega = None
        if 'vrad' in parlabels:
            vrad = params['vrad']
        elif 'omega' in parlabels:
            kappa = params['kappa']
            omega = params['omega']
        elif 'kappa' in parlabels:
            kappa = params['kappa']

        mg = dm.ModelGrid(x_loc=params['xloc'], y_loc=params['yloc'], mbh=params['mbh'], ml_ratio=params['ml_ratio'],
            inc=np.deg2rad(params['inc']), vsys=params['vsys'], theta=np.deg2rad(params['PAdisk']), vrad=vrad,
            kappa=kappa, omega=omega, f_w=params['f'], os=params['os'], enclosed_mass=params['mass'],
            sig_params=[params['sig0'], params['r0'], params['mu'], params['sig1']], resolution=params['resolution'],
            lucy_out=lucy_out, out_name=None, beam=beam, rfit=params['rfit'], zrange=[params['zi'], params['zf']],
            dist=params['dist'], input_data=input_data, sig_type=params['s_type'], menc_type=params['mtype'],
            theta_ell=np.deg2rad(params['theta_ell']), xell=params['xell'],yell=params['yell'], q_ell=params['q_ell'],
            ds=params['ds'], reduced=True, f_0=f_0, freq_ax=freq_ax, noise=noise, bl=params['bl'], fstep=fstep,
            xyrange=[params['xi'], params['xf'], params['yi'], params['yf']], n_params=nfree)

        mg.grids()
        mg.convolution()
        chi2 = mg.chi2()
        print(chi2)
        print(models[t])

        fmt = "{{0:{0}}}".format('.2f').format
        fmt3 = "{{0:{0}}}".format('.3f').format
        chititle = r"${{0}}$".format(fmt3(chi2))
        altchititle = r"{0}".format(fmt3(chi2))
        texlines += models[t] + ' | ' + chititle + ' | '
        lines += models[t] + ' | ' + altchititle + ' | '

        with open(things[t], 'rb') as pk:
            u = pickle._Unpickler(pk)
            u.encoding = 'latin1'
            dyn_res = u.load()  #
            # dyn_res = pickle.load(pk)  #

        weights = np.exp(dyn_res['logwt'] - dyn_res['logz'][-1])  # normalized weights

        for i in range(dyn_res['samples'].shape[1]):  # for each parameter
            quants = [0.0015, 0.5, 0.9985]
            if sig == 1 or sig == 'mod':
                quants = [0.16, 0.5, 0.84]
            elif sig == 2:
                quants = [0.025, 0.5, 0.975]
            elif sig == 3:
                quants = [0.0015, 0.5, 0.9985]
            q = dyfunc.quantile(dyn_res['samples'][:, i], quants, weights=weights)
            if i == 0 and 'nobh' not in parfiles[t]:
                q = np.log10(q)
            title = r"${{{0}}}_{{-{1}}}^{{+{2}}}$"
            alttitle = r"{0} -{1}/+{2}"
            if sig == 'mod':
                mod = (2*4597) ** (1/4)  # BUCKET 4606 changes if the fitting region changes
                texlines += title.format(fmt(q[1]), fmt((q[1] - q[0]) * mod), fmt((q[2] - q[1]) * mod)) + ' | '
                lines += alttitle.format(fmt(q[1]), fmt((q[1] - q[0]) * mod), fmt((q[2] - q[1]) * mod)) + ' | '
            else:
                texlines += title.format(fmt(q[1]), fmt(q[1] - q[0]), fmt(q[2] - q[1])) + ' | '
                lines += alttitle.format(fmt(q[1]), fmt(q[1] - q[0]), fmt(q[2] - q[1])) + ' | '
            #lines += str(q[1].format('.2f').format) + ' +' + str((q[2]-q[1]).format('.2f').format) + ' -' +\
            #         str((q[1] - q[0]).format('.2f').format) + ' | '
        lines += '\n| '
        texlines += '\n| '

    return hdr, hdrl, lines, texlines


def my_own_thing(results, par_labels, ax_labels, quantiles, ax_lims=None, compare_err=False, comp2=False, err=1, fs=12,
                 savefig=None, vrad=False):
    # results should be dyn_res['samples']
    roundto = 3  # 2  # 4
    npar = len(par_labels)
    if npar == 10:
        fig, axes = plt.subplots(2, 5, figsize=(20, 12))  # 2 rows, 5 cols of subplots; because there are 10 free params
        # labels = np.array(['mbh', 'xloc', 'yloc', 'sig0', 'inc', 'PAdisk', 'vsys', 'ml_ratio', 'f',]) vrad, kappa, etc
        axes_order = [[0, 0], [1, 0], [1, 1], [1, 2], [0, 3], [0, 4], [1, 3], [0, 1], [0, 2], [1, 4]]
    elif npar == 11:
        fig, axes = plt.subplots(3, 4, figsize=(20, 12))  # 3 rows, 4 cols of subplots; because there are 11 free params
        # labels =   ['mbh', 'xloc', 'yloc', 'sig0', 'inc', 'PAdisk', 'vsys', 'ml_ratio', 'f', kappa, omega], etc
        axes_order = [[0, 0], [1, 1], [1, 2], [1, 3], [0, 3], [1, 0], [2, 0], [0, 1], [0, 2], [2, 1], [2, 2]]
    elif npar == 9:
        fig, axes = plt.subplots(3, 3, figsize=(20, 12))  # 3 rows, 3 cols of subplots; because there are 9 free params
        # labels = np.array(['mbh', 'xloc', 'yloc', 'sig0', 'inc', 'PAdisk', 'vsys', 'ml_ratio', 'f'])
        axes_order = [[0, 0], [2, 0], [2, 1], [2, 2], [1, 0], [1, 1], [1, 2], [0, 1], [0, 2]]
    elif npar == 8:
        fig, axes = plt.subplots(2, 4, figsize=(20, 12))  # 2 rows, 4 cols of subplots; because there are 8 free params
        # labels = np.array(['xloc', 'yloc', 'sig0', 'inc', 'PAdisk', 'vsys', 'ml_ratio', 'f'])
        axes_order = [[0, 0], [0, 1], [1, 0], [0, 2], [0, 3], [1, 1], [1, 2], [1, 3]]
    for i in range(len(results[0])):
        row, col = axes_order[i]
        if compare_err or comp2:
            print('yes', par_labels[i])
            if par_labels[i] == 'yloc' or par_labels[i] == 'xloc':
                bins = 3200
            elif par_labels[i] == 'sig0':
                bins = 200  # 1000
            elif par_labels[i] == 'f' or par_labels[i] == 'PAdisk' or par_labels[i] == 'vsys' or par_labels[i] == 'mbh':
                bins = 1500
            else:
                bins = 800
        else:
            if par_labels[i] == 'mbh':  # or par_labels[i] == 'PAdisk':
                bins = 400 #1000
            else:
                bins = 100
        chain = results[:, i]
        weight = np.ones_like(chain) * 2e-3
        axes[row, col].hist(chain, bins=bins, color="b", histtype="step", weights=weight)  # axes[i]
        # print(quantiles[i], 'look')
        # percs = np.percentile(chain, [.15, 50., 99.85])  # 3sigma
        # axes[row, col].axvline(percs[1], color='b', ls='--')  # axes[i]
        # axes[row, col].axvspan(percs[0], percs[2], color='b', alpha=0.25)
        axes[row, col].axvline(quantiles[i][1], color='b', ls='--')  # axes[i]
        axes[row, col].axvspan(quantiles[i][0], quantiles[i][2], color='b', alpha=0.25)
        axes[row, col].tick_params('both', labelsize=fs)
        axes[row, col].set_title(par_labels[i] + ': ' + str(round(quantiles[i][1], roundto)) + ' (+'
                                 + str(round(quantiles[i][2] - quantiles[i][1], roundto)) + ', -'
                                 + str(round(quantiles[i][1] - quantiles[i][0], roundto)) + ')', fontsize=fs)
        axes[row, col].set_xlabel(ax_labels[i], fontsize=fs)

        if compare_err:
            with open('/Users/jonathancohn/Documents/dyn_mod/param_files/Ben_A1_errors.txt') as a1:
                for line in a1:
                    cols = line.split()
                    if cols[0] == par_labels[i]:
                        vax = float(cols[1])
                        vax_width = float(cols[2])
            print(vax, 'vax', par_labels[i])
            if par_labels[i] == 'mbh':
                axes[row, col].axvline(np.log10(vax), ls='-', color='k')
                print('here', np.log10(vax), np.log10(vax - vax_width), np.log10(vax + vax_width))
                axes[row, col].axvspan(np.log10(vax - vax_width), np.log10(vax + vax_width), hatch='/', color='k',
                                       fill=False, alpha=0.5)
            else:
                axes[row, col].axvline(vax, ls='-', color='k')
                axes[row, col].axvspan(vax - vax_width, vax + vax_width, hatch='/', color='k', fill=False, alpha=0.5)
        elif comp2:
            with open('/Users/jonathancohn/Documents/dyn_mod/param_files/Ben_A1_sampling.txt') as a1:
                for line in a1:
                    cols = line.split()
                    if cols[0] == par_labels[i]:
                        vax = float(cols[1])
                        vax_width1 = float(cols[2])
                        vax_width3 = float(cols[3])
            if err == 1:
                vax_width = vax_width1
            elif err == 3:
                vax_width = vax_width3
            if par_labels[i] == 'mbh':
                axes[row, col].axvline(np.log10(vax), ls='-', color='k')
                print('here', np.log10(vax), np.log10(vax - vax_width), np.log10(vax + vax_width))
                axes[row, col].axvspan(np.log10(vax - vax_width), np.log10(vax + vax_width), hatch='/', color='k',
                                       fill=False, alpha=0.5)
            else:
                axes[row, col].axvline(vax, ls='-', color='k')
                axes[row, col].axvspan(vax - vax_width, vax + vax_width, hatch='/', color='k', fill=False, alpha=0.5)
        if ax_lims is not None:
            axes[row, col].set_xlim(ax_lims[i][0], ax_lims[i][1])
    plt.tight_layout()
    if savefig is not None:
        plt.savefig(savefig)
    plt.show()

direc = '/Users/jonathancohn/Documents/dyn_mod/nest_out/'
grp = '/Users/jonathancohn/Documents/dyn_mod/groupmtg/'
inpf = None
# OLDER
#out_name = direc + 'dyndyncluster_test_n8_1568739677.2582278_tempsave.pkl'  # 'dynesty_demo_output.pkl'
#out_name = direc + 'dyndyncluster_test_n8_1568739677.1305604_tempsave.pkl'
#out_name = direc + 'dyndyn2_test_n8_1568739677.1305604_tempsave.pkl'

# WORKED
# out_name = direc + 'dyndyncluster_test_n8_1568739677.2582278_tempsave.pkl'  # 2698 old prior
# out_name = direc + 'dyndyn3258_test_n8_dlogz1.5_1569249221.642805_tempsave.pkl'  # 3258 old TINY prior
# thing = 'dyndyn3258_test_n8_dlogz1.5_1569249222.4063346_tempsave.pkl'  # 3258 old TINY prior (same as above)
# out_name = direc + 'dyndyn3258_test_n8_dlogz15_1569434823.359717_tempsave.pkl'  # 3258 old TINY prior
# out_name = direc + 'dyndyn3258_newpri_test_n8_dlogz1.5_1569620562.4751067_tempsave.pkl'  # 3258 newprior TYPO (PRIOR for f was BAD 0.5<f<0.65) (AHA!)
# out_name = direc + 'dyndyn3258_newpri_test_n8_dlogz1.5_1569620562.355544_tempsave.pkl'  # 3258 newprior TYPO same as above!
# out_name = direc + 'dyndyn_newpri_test_n8_1569798600.5749478_end.pkl' # 2698 newprior GOOD (altho typo 1e8<mbh<1e18)
# out_name = direc + 'dyndyn_newpri_test_n8_1569620951.5581834_tempsave.pkl' # 2698 newprior GOOD (same as above)
# out_name = direc + 'dyndyn105newpri_test_n8_1569855263.300289_tempsave.pkl'  # 2698 newprior 4x8(?) REALLY BAD WHY
# out_name = direc + 'dyndyn3258_test_n8_dlogz1.5_1570221683.4872622_tempsave.pkl'  # 3258 old-prior style good
# out_name = direc + 'dyndyn3258_newpri_test_n8_dlogz1.5_1570464872.8310175_tempsave.pkl'  # 3258 newprior BAD WHY
# out_name = direc + 'dyndyn105newpri_test_n8_1570545604.3403502_tempsave.pkl'  # 2698 newprior 4x8(?) REALLY BAD WHY
# out_name = direc + 'dyndyn105newpri_test_n8_ds3_1570633918.9553041_tempsave.pkl'  # 2698 neworior 3x6 REALLY BAD WHY
# out_name = direc + 'dyndyn105newpri_test_n8_1570545604.3403502_tempsave.pkl'  # 2698 neworior 4x8 REALLY BAD WHY
# out_name = direc + 'dyndyn3258_newpri_2_test_n8_dlogz0.15_1570807500.0458014_tempsave.pkl'  # 3258 newprior 2 REALLY BAD STILL WHYYYY (BECAUSE MAXCALL IS REACHED!!!)
# out_name = direc + 'dyndyn3258_narrowpri_n8_dlogz1.5_1571063422.9054961_tempsave.pkl'  # 3258 narrow priors, yep still works
# thing = 'dyndyn105newpri_2_maxcmil_n8_ds3_1571235624.2196844_tempsave.pkl'  # 2698 105newpri2 3x6 BAD PA NOT IN RADS
# thing = 'dyndyn105newpri_2_maxcmil_n8_ds3_1571410705.2218955_end.pkl'  # 2698 105newpri2 3x6 end BAD PA NOT IN RADS
# thing = 'dyndyn105newpri_3_maxcmil_n8_ds3_1571323790.6415393_tempsave.pkl'  # 2698 105newpri3 3x6 (fine)
# thing = 'dyndyn105newpri_3_maxcmil_n8_ds3_1571374539.7854428_end.pkl'  # 2698 105newpri3 3x6 (fine)
# thing = 'dyndyn105newpri_maxcmil_n8_ds3_1571092540.929372_tempsave.pkl'  # 2698 105newpri 3x6 VERY BAD (must've hit mil)
# thing = 'dyndyn105newpri_maxcmil_n8_ds4_1571092684.7476249_tempsave.pkl'  # 2698 105newpri 4x8 VERY BAD (must've hit mil)
# thing = 'dyndyn_newpri_2_n8_1571236232.359415_tempsave.pkl'  # 2698 newpri2 BAD because hit mil (getting closer tho)
# thing = 'dyndyn_newpri_2_maxc10mil_n8_1571596549.628827_tempsave.pkl'  # 2698 newpri2 BAD because PA NOT IN RADS
# thing = 'dyndyn_newpri_2_maxc10mil_n8_1571981879.8668501_end.pkl'  # 2698 newpri2 BAD because PA NOT IN RADS
# thing = 'dyndyn_newpri_3_maxc10mil_n8_0.02_1572037132.3993847_tempsave.pkl'  # 2698 newpri3 GOOD PA CORRECTED!
# thing = 'dyndyn3258_newpri_5_max10mil_test_n8_dlogz0.15_1572883759.2018988_tempsave.pkl'  # 3258 newpri5 PA GOOD!
# thing = 'dyndyn3258_newpri_5_max10mil_n8_dlogz0.15_1573010828.1336136_end.pkl'  # 3258 newpri5 PA GOOD end! (same good)
# thing = 'dyndyn3258_xy_max10mil_n8_dlogz0.15_1574217317.0507047_end.pkl'  # 3258 xy free, else fixed
# NEW MASKS BELOW
# RHE 2
# thing = 'dyndyn_newpri_3_maxc10mil_n8_mask2rhe_1586505950.4425063_end.pkl'  # 2698 mask2 rhe (sig0 prior now too low)
# thing = 'dyndyn_newpri_3_maxc10mil_n8_mask2rhe_1586825623.941504_end.pkl'  # 2698 mask2 rhe sig0 extended!
# name = 'ugc_2698_newmasks/u2698_nest_mask2rhe_sig0extend_3sig.png'
# cornername = 'ugc_2698_newmasks/u2698_nest_mask2rhe_sig0extend_corner_3sig.png'
# inpf = 'ugc_2698/ugc_2698_newmask2_rhe_n8.txt'
# mod = 'rhe baseline'
# RHE LAX
# thing = 'dyndyn_newpri_3_maxc10mil_n8_masklaxrhe_1586572307.33264_end.pkl'  # 2698 masklax rhe (bad priors)
# thing = 'dyndyn_newpri_3_maxc10mil_n8_masklaxrhe_1586800510.8196273_end.pkl'  # 2698 masklax rhe prior extended!
# name = 'ugc_2698_newmasks/u2698_nest_masklaxrhe_priextend_3sig.png'
# cornername = 'ugc_2698_newmasks/u2698_nest_masklaxrhe_priextend_corner_3sig.png'
# inpf = 'ugc_2698/ugc_2698_newmasklax_rhe_n8.txt'
# mod = 'rhe lax'
# RHE STRICT
# thing = 'dyndyn_newpri_3_maxc10mil_n8_maskstrictrhe_1586573455.940675_end.pkl'  # 2698 maskstrict rhe (bad priors)
# thing = 'dyndyn_newpri_3_maxc10mil_n8_maskstrictrhe_1586801056.355033_end.pkl'  # 2698 maskstrict rhe prior extended!
# name = 'ugc_2698_newmasks/u2698_u2698_nest_maskstrictrhe_priextend_3sig.png'
# cornername = 'ugc_2698_newmasks/u2698_nest_maskstrictrhe_priextend_corner_3sig.png'
# inpf = 'ugc_2698/ugc_2698_newmaskstrict_rhe_n8.txt'
# mod = 'rhe strict'
# AKIN 2
# thing = 'dyndyn_newpri_3_maxc10mil_n8_mask2akin_1586613935.0107005_end.pkl'  # 2698 mask2 akin's mge (bad priors)
# thing = 'dyndyn_newpri_3_maxc10mil_n8_mask2akin_1586820818.1132705_end.pkl'  # 2698 mask2 akin's mge prior extended!
# name = 'ugc_2698_newmasks/u2698_nest_mask2akin_priextend_3sig.png'
# cornername = 'ugc_2698_newmasks/u2698_nest_mask2akin_priextend_corner_3sig.png'
# inpf = 'ugc_2698/ugc_2698_newmask2_akin_n8.txt'
# mod = 'akin baseline'
# AKIN 2
dictakin2 = {'pkl': 'dyndyn_newpri_3_maxc10mil_n8_mask2akin_1586820818.1132705_end.pkl', # 2698 mask2 akin (good priors)
             'name': 'ugc_2698_newmasks/u2698_nest_mask2akin_priextend_3sig.png',
             'cornername': 'ugc_2698_newmasks/u2698_nest_mask2akin_priectend_corner_3sig.png',
             'inpf': 'ugc_2698/ugc_2698_newmask2_akin_n8.txt', 'outpf': 'ugc_2698/ugc_2698_newmask2_akin_n8_out.txt',
             'mod': 'akin baseline', 'extra_params': None}
# RRE 2
dictrre2 = {'pkl': 'dyndyn_newpri_3_maxc10mil_n8_mask2rre_1586801196.254912_end.pkl', # 2698 mask2 rre (good priors)
            'name': 'ugc_2698_newmasks/u2698_nest_mask2rre_3sig.png',
            'cornername': 'ugc_2698_newmasks/u2698_nest_mask2rre_corner_3sig.png',
            'inpf': 'ugc_2698/ugc_2698_newmask2_rre_n8.txt', 'outpf': 'ugc_2698/ugc_2698_newmask2_rre_n8_out.txt',
            'mod': 'rre baseline', 'extra_params': None}
# RRE LAX
dictrrelax = {'pkl': 'dyndyn_newpri_3_maxc10mil_n8_masklaxrre_1586982579.9110975_end.pkl', # 2698 masklax rre (good priors)
              'name': 'ugc_2698_newmasks/u2698_nest_masklaxrre_3sig.png',
              'cornername': 'ugc_2698_newmasks/u2698_nest_masklaxrre_corner_3sig.png',
              'inpf': 'ugc_2698/ugc_2698_newmasklax_rre_n8.txt', 'outpf': 'ugc_2698/ugc_2698_newmasklax_rre_n8_out.txt',
              'mod': 'rre lax', 'extra_params': None}
# RRE STRICT
dictrrestrict = {'pkl': 'dyndyn_newpri_3_maxc10mil_n8_maskstrictrre_1586987192.3806183_end.pkl', # 2698 maskstrict rre (good priors)
                 'name': 'ugc_2698_newmasks/u2698_nest_maskstrictrre_3sig.png',
                 'cornername': 'ugc_2698_newmasks/u2698_nest_maskstrictrre_corner_3sig.png',
                 'inpf': 'ugc_2698/ugc_2698_newmaskstrict_rre_n8.txt',
                 'outpf': 'ugc_2698/ugc_2698_newmaskstrict_rre_n8_out.txt', 'mod': 'rre strict', 'extra_params': None}
# AHE 2
dictahe2 = {'pkl': 'dyndyn_newpri_3_maxc10mil_n8_mask2ahe_1586560467.2338874_end.pkl', # 2698 mask2 ahe (good priors)
            'name': 'ugc_2698_newmasks/u2698_nest_mask2ahe_sig0extend_3sig.png',
            'cornername': 'ugc_2698_newmasks/u2698_nest_mask2ahe_sig0extend_corner_3sig.png',
            'inpf': 'ugc_2698/ugc_2698_newmask2_ahe_n8.txt', 'outpf': 'ugc_2698/ugc_2698_newmask2_ahe_n8_out.txt',
            'mod': 'ahe baseline', 'extra_params': None}
# AHE LAX
dictahelax = {'pkl': 'dyndyn_newpri_3_maxc10mil_n8_masklaxahe_1586985474.5780168_end.pkl', # 2698 masklax ahe (good priors)
              'name': 'ugc_2698_newmasks/u2698_nest_masklaxahe_3sig.png',
              'cornername': 'ugc_2698_newmasks/u2698_nest_masklaxahe_corner_3sig.png',
              'inpf': 'ugc_2698/ugc_2698_newmasklax_ahe_n8.txt', 'outpf': 'ugc_2698/ugc_2698_newmasklax_ahe_n8_out.txt',
              'mod': 'ahe lax', 'extra_params': None}
# AHE STRICT
dictahestrict = {'pkl': 'dyndyn_newpri_3_maxc10mil_n8_maskstrictahe_1586987362.5191674_end.pkl', # 2698 maskstrict ahe (good priors)
                 'name': 'ugc_2698_newmasks/u2698_nest_maskstrictahe_3sig.png',
                 'cornername': 'ugc_2698_newmasks/u2698_nest_maskstrictahe_corner_3sig.png',
                 'inpf': 'ugc_2698/ugc_2698_newmaskstrict_ahe_n8.txt',
                 'outpf': 'ugc_2698/ugc_2698_newmaskstrict_ahe_n8_out.txt', 'mod': 'ahe strict', 'extra_params': None}
# RHE 2
dictrhe2 = {'pkl': 'dyndyn_newpri_3_maxc10mil_n8_mask2rhe_1586825623.941504_end.pkl', # 2698 mask2 rhe (good priors)
            'name': 'ugc_2698_newmasks/u2698_nest_mask2rhe_sig0extend_3sig.png',
            'cornername': 'ugc_2698_newmasks/u2698_nest_mask2rhe_sig0extend_corner_3sig.png',
            'inpf': 'ugc_2698/ugc_2698_newmask2_rhe_n8.txt', 'outpf': 'ugc_2698/ugc_2698_newmask2_rhe_n8_out.txt',
            'mod': 'rhe baseline', 'extra_params': None}
# RHE LAX
dictrhelax = {'pkl': 'dyndyn_newpri_3_maxc10mil_n8_masklaxrhe_1586800510.8196273_end.pkl', # 2698 masklax rhe (good priors)
              'name': 'ugc_2698_newmasks/u2698_nest_masklaxrhe_priextend_3sig.png',
              'cornername': 'ugc_2698_newmasks/u2698_nest_masklaxrhe_priextend_corner_3sig.png',
              'inpf': 'ugc_2698/ugc_2698_newmasklax_rhe_n8.txt', 'outpf': 'ugc_2698/ugc_2698_newmasklax_rhe_n8_out.txt',
              'mod': 'rhe lax', 'extra_params': None}
# RHE STRICT
dictrhestrict = {'pkl': 'dyndyn_newpri_3_maxc10mil_n8_maskstrictrhe_1586801056.355033_end.pkl', # 2698 maskstrict rhe (good priors)
                 'name': 'ugc_2698_newmasks/u2698_nest_maskstrictrhe_priextend_3sig.png',
                 'cornername': 'ugc_2698_newmasks/u2698_nest_maskstrictrhe_priextend_corner_3sig.png',
                 'inpf': 'ugc_2698/ugc_2698_newmaskstrict_rhe_n8.txt',
                 'outpf': 'ugc_2698/ugc_2698_newmaskstrict_rhe_n8_out.txt', 'mod': 'rhe strict', 'extra_params': None}

# RHE 2 VRAD
dictrhe2vrad = {'pkl': 'dyndyn_newpri_3_maxc10mil_n8_mask2rhevrad_1587724128.296138_end.pkl',  # 2698 mask2 rhe, vrad
                'name': 'ugc_2698_newmasks/u2698_nest_mask2rhevrad_3sig.png',
                'cornername': 'ugc_2698_newmasks/u2698_nest_mask2rhevrad_corner_3sig.png',
                'inpf': 'ugc_2698/ugc_2698_newmask2_rhe_n8_vrad.txt',
                'outpf': 'ugc_2698/ugc_2698_newmask2_rhe_n8_vrad_out.txt',
                'mod': 'rhe baseline',
                'extra_params': [['vrad', 'km/s']]}

# RHE 2 KAPPA
dictrhe2kappa = {'pkl': 'u2698_mask2_rhe_kappa_10000000_8_0.01_1588375516.5221143_end.pkl',  # 2698 mask2 rhe, kappa
                 'name': 'ugc_2698_newmasks/u2698_nest_mask2rhekappa_3sig.png',
                 'cornername': 'ugc_2698_newmasks/u2698_nest_mask2rhekappa_corner_3sig.png',
                 'inpf': 'ugc_2698/ugc_2698_newmask2_rhe_n8_kappa.txt',
                 'outpf': 'ugc_2698/ugc_2698_newmask2_rhe_n8_kappa_out.txt',
                 'mod': 'rhe baseline',
                 'extra_params': [['kappa', 'unitless']]}

# RHE 2 OMEGA
dictrhe2omega = {'pkl': 'u2698_mask2_rhe_omega_10000000_8_0.01_1588463109.9259367_end.pkl',  # 2698 mask2 rhe, omega
                 'name': 'ugc_2698_newmasks/u2698_nest_mask2rheomega_3sig.png',
                 'cornername': 'ugc_2698_newmasks/u2698_nest_mask2rheomega_corner_3sig.png',
                 'inpf': 'ugc_2698/ugc_2698_newmask2_rhe_n8_omega.txt',
                 'outpf': 'ugc_2698/ugc_2698_newmask2_rhe_n8_omega_out.txt',
                 'mod': 'rhe baseline',
                 'extra_params': [['kappa', 'unitless'], ['omega', 'unitless']]}

# RHE BASELINE GAS
dictrhe2gas = {'pkl': 'u2698_baseline_rhe_orig_gas_10000000_8_0.02_1588986032.6169796_end.pkl', # 2698 mask2 rhe gas
               'name': 'ugc_2698_newmasks/u2698_nest_baseline_rhe_orig_gas_3sig.png',
               'cornername': 'ugc_2698_newmasks/u2698_nest_baseline_rhe_orig_gas_corner_3sig.png',
               'inpf': 'ugc_2698/ugc_2698_baseline_rhe_orig_gas.txt',
               'outpf': 'ugc_2698/ugc_2698_baseline_rhe_orig_gas_out.txt',
               'mod': 'rhe baseline gas', 'extra_params': None}

# RHE BASELINE DIST85
dictrhe2d85 = {'pkl': 'u2698_d85_baseline_rhe_orig_nog_10000000_8_0.02_1589829467.780973_end.pkl', # 2698 mask2 rhe d85
               'name': 'ugc_2698_newmasks/u2698_nest_d85_baseline_rhe_orig_nog_3sig.png',
               'cornername': 'ugc_2698_newmasks/u2698_nest_d85_baseline_rhe_orig_nog_corner_3sig.png',
               'inpf': 'ugc_2698/ugc_2698_d85_baseline_rhe_orig_nog.txt',  # ugc_2698_d85_baseline_rhe_orig_nog.txt
               'outpf': 'ugc_2698/ugc_2698_d85_baseline_rhe_orig_nog_out.txt',
               'mod': 'rhe baseline nog d85', 'extra_params': None}  # Runtime: 4.56889167 hours (131584.08 sec)

# RHE BASELINE DIST85, ORIG FITTING ELLIPSE
dictrhe2d85o = {'pkl': 'u2698_d85oell_baseline_rhe_orig_nog_10000000_8_0.02_1589945564.533565_end.pkl', # 2698 mask2 rhe d85oell
                'name': 'ugc_2698_newmasks/u2698_nest_d85oell_baseline_rhe_orig_nog_3sig.png',
                'cornername': 'ugc_2698_newmasks/u2698_nest_d85oell_baseline_rhe_orig_nog_corner_3sig.png',
                'inpf': 'ugc_2698/ugc_2698_d85oell_baseline_rhe_orig_nog.txt',  # ugc_2698_d85_baseline_rhe_orig_nog.txt
                'outpf': 'ugc_2698/ugc_2698_d85oell_baseline_rhe_orig_nog_out.txt',
                'mod': 'rhe baseline nog d85oell', 'extra_params': None}  # Runtime: 4.28580069 hours (123431.06 sec)

# RHE BASELINE Beam51, DIST85
dictrhe2b51 = {'pkl': 'u2698_b51_d85_baseline_rhe_orig_nog_10000000_8_0.02_1589860625.1720965_end.pkl', # 2698 mask2 rhe b51 d85
               'name': 'ugc_2698_newmasks/u2698_nest_b51_d85_baseline_rhe_orig_nog_3sig.png',
               'cornername': 'ugc_2698_newmasks/u2698_nest_b51_d85_baseline_rhe_orig_nog_corner_3sig.png',
               'inpf': 'ugc_2698/ugc_2698_b51_d85_baseline_rhe_orig_nog.txt',
               'outpf': 'ugc_2698/ugc_2698_b51_d85_baseline_rhe_orig_nog_out.txt',
               'mod': 'rhe baseline nog b51 d85', 'extra_params': None}  # Runtime: 6.7734625 hours (195075.72 sec)

# RHE BASELINE n250
dictrhe2n25 = {'pkl': 'u2698_n250_baseline_rhe_orig_nog_10000000_8_0.02_1589930146.9680398_end.pkl', # 2698 mask2 rhe n250
               'name': 'ugc_2698_newmasks/u2698_nest_n250_baseline_rhe_orig_nog_3sig.png',
               'cornername': 'ugc_2698_newmasks/u2698_nest_n250_baseline_rhe_orig_nog_corner_3sig.png',
               'inpf': 'ugc_2698/ugc_2698_n250_baseline_rhe_orig_nog.txt',
               'outpf': 'ugc_2698/ugc_2698_n250_baseline_rhe_orig_nog_out.txt',
               'mod': 'rhe baseline n250', 'extra_params': None}  # Runtime: 3.37993923611 hours (97342.25 sec)

# RHE BASELINE n1000
dictrhe2n1k = {'pkl': 'u2698_n1000_baseline_rhe_orig_nog_10000000_8_0.02_1589932643.558555_end.pkl', # 2698 mask2 rhe n1000
               'name': 'ugc_2698_newmasks/u2698_nest_n1000_baseline_rhe_orig_nog_3sig.png',
               'cornername': 'ugc_2698_newmasks/u2698_nest_n1000_baseline_rhe_orig_nog_corner_3sig.png',
               'inpf': 'ugc_2698/ugc_2698_n1000_baseline_rhe_orig_nog.txt',
               'outpf': 'ugc_2698/ugc_2698_n1000_baseline_rhe_orig_nog_out.txt',
               'mod': 'rhe baseline n1000', 'extra_params': None}  # Runtime: 7.47817535 hours (215371.45 sec)

# RHE BASELINE dlz01
dictrhe2dlz01 = {'pkl': 'u2698_dlz01_baseline_rhe_orig_nog_10000000_8_0.1_1589923485.539471_end.pkl', # 2698 mask2 rhe dlz01
                 'name': 'ugc_2698_newmasks/u2698_nest_dlz01_baseline_rhe_orig_nog_3sig.png',
                 'cornername': 'ugc_2698_newmasks/u2698_nest_dlz01_baseline_rhe_orig_nog_corner_3sig.png',
                 'inpf': 'ugc_2698/ugc_2698_dlz01_baseline_rhe_orig_nog.txt',
                 'outpf': 'ugc_2698/ugc_2698_dlz01_baseline_rhe_orig_nog_out.txt',
                 'mod': 'rhe baseline dlz01', 'extra_params': None}  # Runtime: 5.05 hours (145415.41 sec)

# RHE BASELINE dlz0001
dictrhe2dlz0001 = {'pkl': 'u2698_dlz0001_baseline_rhe_orig_nog_10000000_8_0.001_1589921482.0451794_end.pkl', # 2698 mask2 rhe dlz0001
                   'name': 'ugc_2698_newmasks/u2698_nest_dlz0001_baseline_rhe_orig_nog_3sig.png',
                   'cornername': 'ugc_2698_newmasks/u2698_nest_dlz0001_baseline_rhe_orig_nog_corner_3sig.png',
                   'inpf': 'ugc_2698/ugc_2698_dlz0001_baseline_rhe_orig_nog.txt',
                   'outpf': 'ugc_2698/ugc_2698_dlz0001_baseline_rhe_orig_nog_out.txt',
                   'mod': 'rhe baseline dlz0001', 'extra_params': None}  # Runtime: 4.45 hours (128087.74 sec)

# RHE BASELINE Beam15, DIST85
dictrhe2gs5 = {'pkl': 'u2698_gs5_d85_baseline_rhe_orig_nog_10000000_8_0.02_1590624161.6140797_end.pkl', # 2698 mask2 rhe gs5 d85
               'name': 'ugc_2698_newmasks/u2698_nest_gs5_d85_baseline_rhe_orig_nog_3sig.png',
               'cornername': 'ugc_2698_newmasks/u2698_nest_gs5_d85_baseline_rhe_orig_nog_corner_3sig.png',
               'inpf': 'ugc_2698/ugc_2698_gs5_d85_baseline_rhe_orig_nog.txt',
               'outpf': 'ugc_2698/ugc_2698_gs5_d85_baseline_rhe_orig_nog_out.txt',
               'mod': 'rhe baseline nog gs5 d85', 'extra_params': None}  # Runtime: 5.48489583 hours (157965.00 sec)

# RHE BASELINE Beam15, DIST85
dictrhe2gs15 = {'pkl': 'u2698_gs15_d85_baseline_rhe_orig_nog_10000000_8_0.02_1590439381.911974_end.pkl', # 2698 mask2 rhe gs15 d85
                'name': 'ugc_2698_newmasks/u2698_nest_gs15_d85_baseline_rhe_orig_nog_3sig.png',
                'cornername': 'ugc_2698_newmasks/u2698_nest_gs15_d85_baseline_rhe_orig_nog_corner_3sig.png',
                'inpf': 'ugc_2698/ugc_2698_gs15_d85_baseline_rhe_orig_nog.txt',
                'outpf': 'ugc_2698/ugc_2698_gs15_d85_baseline_rhe_orig_nog_out.txt',
                'mod': 'rhe baseline nog gs15 d85', 'extra_params': None}  # Runtime: 3.52119931 hours (101410.54 sec)

# RHE BASELINE Beam71, DIST85
dictrhe2gs71 = {'pkl': 'u2698_gs71_d85_baseline_rhe_orig_nog_10000000_8_0.02_1590463525.87706_end.pkl', # 2698 mask2 rhe gs71 d85
                'name': 'ugc_2698_newmasks/u2698_nest_gs71_d85_baseline_rhe_orig_nog_3sig.png',
                'cornername': 'ugc_2698_newmasks/u2698_nest_gs71_d85_baseline_rhe_orig_nog_corner_3sig.png',
                'inpf': 'ugc_2698/ugc_2698_gs71_d85_baseline_rhe_orig_nog.txt',
                'outpf': 'ugc_2698/ugc_2698_gs71_d85_baseline_rhe_orig_nog_out.txt',
                'mod': 'rhe baseline nog gs71 d85', 'extra_params': None}  # Runtime: 9.84351667 hours (283493.28 sec)

# RHE BASELINE os2, d91 // Runtime: 3.86155764 hours (111212.86 s), 99910 total calls, 32514 in initial stage
dictrhe2os2 = {'pkl': 'u2698_os2_d91_baseline_rhe_orig_nog_10000000_8_0.02_1590693577.7494018_end.pkl',
               'name': 'ugc_2698_newmasks/u2698_nest_os2_d91_baseline_rhe_orig_nog_3sig.png',
               'cornername': 'ugc_2698_newmasks/u2698_nest_os2_d91_baseline_rhe_orig_nog_corner_3sig.png',
               'inpf': 'ugc_2698/ugc_2698_os2_d91_baseline_rhe_orig_nog.txt',
               'outpf': 'ugc_2698/ugc_2698_os2_d91_baseline_rhe_orig_nog_out.txt',
               'mod': 'rhe baseline (os2, d91)', 'extra_params': None}

# RHE BASELINE os3, d91 // Runtime: 3.53416181 hours (101783.86 s), 83881 total calls, 29171 in initial stage
dictrhe2os3 = {'pkl': 'u2698_os3_d91_baseline_rhe_orig_nog_10000000_8_0.02_1590692584.260687_end.pkl',
               'name': 'ugc_2698_newmasks/u2698_nest_os3_d91_baseline_rhe_orig_nog_3sig.png',
               'cornername': 'ugc_2698_newmasks/u2698_nest_os3_d91_baseline_rhe_orig_nog_corner_3sig.png',
               'inpf': 'ugc_2698/ugc_2698_os3_d91_baseline_rhe_orig_nog.txt',
               'outpf': 'ugc_2698/ugc_2698_os3_d91_baseline_rhe_orig_nog_out.txt',
               'mod': 'rhe baseline (os3, d91)', 'extra_params': None}

# RHE BASELINE os4, d91 // Runtime: 3.38293333 hours (97428.48 s), 74588 total calls, 28648 in initial stage
dictrhe2os4 = {'pkl': 'u2698_os4_d91_baseline_rhe_orig_nog_10000000_8_0.02_1590692888.5767071_end.pkl',
               'name': 'ugc_2698_newmasks/u2698_nest_os4_d91_baseline_rhe_orig_nog_3sig.png',
               'cornername': 'ugc_2698_newmasks/u2698_nest_os4_d91_baseline_rhe_orig_nog_corner_3sig.png',
               'inpf': 'ugc_2698/ugc_2698_os4_d91_baseline_rhe_orig_nog.txt',
               'outpf': 'ugc_2698/ugc_2698_os4_d91_baseline_rhe_orig_nog_out.txt',
               'mod': 'rhe baseline (os4, d91)', 'extra_params': None}

# RHE BASELINE os6, d91 // Runtime: 4.91630278 hours (141589.52 s), 92306 total calls, 28768 in initial stage
dictrhe2os6 = {'pkl': 'u2698_os6_d91_baseline_rhe_orig_nog_10000000_8_0.02_1590699379.3008854_end.pkl',
               'name': 'ugc_2698_newmasks/u2698_nest_os6_d91_baseline_rhe_orig_nog_3sig.png',
               'cornername': 'ugc_2698_newmasks/u2698_nest_os6_d91_baseline_rhe_orig_nog_corner_3sig.png',
               'inpf': 'ugc_2698/ugc_2698_os6_d91_baseline_rhe_orig_nog.txt',
               'outpf': 'ugc_2698/ugc_2698_os6_d91_baseline_rhe_orig_nog_out.txt',
               'mod': 'rhe baseline (os6, d91)', 'extra_params': None}
'''
# Note: pre-batch 
| rhe baseline (os6, d91) | 1.303 (5990.287) | 9.39 -0.03/+0.04 | 126.90 -0.33/+0.33 | 150.96 -0.28/+0.29 | 17.14 -2.60/+2.84 | 67.67 -0.91/+0.91 | 18.74 -0.82/+0.80 | 6454.76 -2.56/+2.23 | 1.71 -0.05/+0.04 | 1.10 -0.04/+0.05 | 

'''

# RHE BASELINE os8, d91 // Runtime: 5.45515556 hours (157108.48 s); 80759 total calls, 28235 in initial stage
dictrhe2os8 = {'pkl': 'u2698_os8_d91_baseline_rhe_orig_nog_10000000_8_0.02_1590701499.9093819_end.pkl',
               'name': 'ugc_2698_newmasks/u2698_nest_os8_d91_baseline_rhe_orig_nog_3sig.png',
               'cornername': 'ugc_2698_newmasks/u2698_nest_os8_d91_baseline_rhe_orig_nog_corner_3sig.png',
               'inpf': 'ugc_2698/ugc_2698_os8_d91_baseline_rhe_orig_nog.txt',
               'outpf': 'ugc_2698/ugc_2698_os8_d91_baseline_rhe_orig_nog_out.txt',
               'mod': 'rhe baseline (os8, d91)', 'extra_params': None}

# RHE BASELINE os10, d91 // Runtime: 6.35648819 hours (183066.86 s); 74868 total calls; 28623 in initial stage
dictrhe2os10 = {'pkl': 'u2698_os10_d91_baseline_rhe_orig_nog_10000000_8_0.02_1590705487.6493137_end.pkl',
                'name': 'ugc_2698_newmasks/u2698_nest_os10_d91_baseline_rhe_orig_nog_3sig.png',
                'cornername': 'ugc_2698_newmasks/u2698_nest_os10_d91_baseline_rhe_orig_nog_corner_3sig.png',
                'inpf': 'ugc_2698/ugc_2698_os10_d91_baseline_rhe_orig_nog.txt',
                'outpf': 'ugc_2698/ugc_2698_os10_d91_baseline_rhe_orig_nog_out.txt',
                'mod': 'rhe baseline (os10, d91)', 'extra_params': None}

# RHE BASELINE os12, d91 // Runtime: 9.00957396 hours (259475.73 s); 81068 total calls; 28501 in initial stage
dictrhe2os12 = {'pkl': 'u2698_os12_d91_baseline_rhe_orig_nog_10000000_8_0.02_1590715106.442782_end.pkl',
                'name': 'ugc_2698_newmasks/u2698_nest_os12_d91_baseline_rhe_orig_nog_3sig.png',
                'cornername': 'ugc_2698_newmasks/u2698_nest_os12_d91_baseline_rhe_orig_nog_corner_3sig.png',
                'inpf': 'ugc_2698/ugc_2698_os12_d91_baseline_rhe_orig_nog.txt',
                'outpf': 'ugc_2698/ugc_2698_os12_d91_baseline_rhe_orig_nog_out.txt',
                'mod': 'rhe baseline (os12, d91)', 'extra_params': None}

# RHE BASELINE os14, d91 // Runtime: 12.6801323 hours (365187.81 s); 91474 total calls; 27918 in initial stage
dictrhe2os14 = {'pkl': 'u2698_os14_d91_baseline_rhe_orig_nog_10000000_8_0.02_1590729076.410538_end.pkl',
                'name': 'ugc_2698_newmasks/u2698_nest_os14_d91_baseline_rhe_orig_nog_3sig.png',
                'cornername': 'ugc_2698_newmasks/u2698_nest_os14_d91_baseline_rhe_orig_nog_corner_3sig.png',
                'inpf': 'ugc_2698/ugc_2698_os14_d91_baseline_rhe_orig_nog.txt',
                'outpf': 'ugc_2698/ugc_2698_os14_d91_baseline_rhe_orig_nog_out.txt',
                'mod': 'rhe baseline (os14, d91)', 'extra_params': None}

# RHE BASELINE os16, d91 // Runtime: 13.230291 hours (381032.38 s); 83169 total calls; 28864 in initial stage
dictrhe2os16 = {'pkl': 'u2698_os16_d91_baseline_rhe_orig_nog_10000000_8_0.02_1590732620.6191862_end.pkl',
                'name': 'ugc_2698_newmasks/u2698_nest_os16_d91_baseline_rhe_orig_nog_3sig.png',
                'cornername': 'ugc_2698_newmasks/u2698_nest_os16_d91_baseline_rhe_orig_nog_corner_3sig.png',
                'inpf': 'ugc_2698/ugc_2698_os16_d91_baseline_rhe_orig_nog.txt',
                'outpf': 'ugc_2698/ugc_2698_os16_d91_baseline_rhe_orig_nog_out.txt',
                'mod': 'rhe baseline (os16, d91)', 'extra_params': None}

# RHE BASELINE priset2 // Runtime: 5.82 hrs (167512.72 s) (~3.33 hrs initial); 127117 total calls; 39273 initial calls
dictrhe2set2 = {'pkl': 'u2698_priset2_d91_baseline_rhe_orig_nog_10000000_8_0.02_1591064673.5489726_end.pkl',
                'name': 'ugc_2698_newmasks/u2698_nest_priset2_d91_baseline_rhe_orig_nog_3sig_fullpri.png',
                'cornername': 'ugc_2698_newmasks/u2698_nest_priset2_d91_baseline_rhe_orig_nog_corner_3sig.png',
                'inpf': 'ugc_2698/ugc_2698_priset2_d91_baseline_rhe_orig_nog.txt',
                'outpf': 'ugc_2698/ugc_2698_priset2_d91_baseline_rhe_orig_nog_out.txt',
                'mod': 'rhe baseline (priset2, d91)', 'extra_params': None}

# RHE BASELINE priset3 // Runtime: 4.34 hrs (125138.24 s) (~3.33 hrs initial); 98311 total calls; 44026 initial calls
#dictrhe2set3 = {'pkl': 'u2698_priset3_d91_baseline_rhe_orig_nog_10000000_8_0.02_1591056752.7983973_end.pkl',
#                'name': 'ugc_2698_newmasks/u2698_nest_priset3_d91_baseline_rhe_orig_nog_3sig.png',
#                'cornername': 'ugc_2698_newmasks/u2698_nest_priset3_d91_baseline_rhe_orig_nog_corner_3sig.png',
#                'inpf': 'ugc_2698/ugc_2698_priset3_d91_baseline_rhe_orig_nog.txt',
#                'outpf': 'ugc_2698/ugc_2698_priset3_d91_baseline_rhe_orig_nog_out.txt',
#                'mod': 'rhe baseline (priset3, d91)', 'extra_params': None}

# RHE BASELINE NOBH // Runtime: 3.32 hrs (95689.48 s) (~1.8hr in initial); 86416 total calls; 32600 initial calls
dictrhe2nobh = {'pkl': 'u2698_nobh_baseline_rhe_orig_nog_10000000_8_0.02_1591290916.2051768_end.pkl',
                'name': 'ugc_2698_newmasks/u2698_nest_nobh_baseline_rhe_orig_nog_3sig.png',
                'cornername': 'ugc_2698_newmasks/u2698_nest_nobh_baseline_rhe_orig_nog_corner_3sig.png',
                'inpf': 'ugc_2698/ugc_2698_nobh_baseline_rhe_orig_nog.txt',
                'outpf': 'ugc_2698/ugc_2698_nobh_baseline_rhe_orig_nog_out.txt',
                'mod': 'rhe baseline (nobh, d91)', 'extra_params': None}

# RHE BASELINE PRISET3 PROPER // 7.21 hrs (207733.30s); 157996 total calls; 49752 initial calls
dictrhe2set3 = {'pkl': 'u2698_priset3_d91_baseline_rhe_orig_nog_10000000_8_0.02_1591304348.7719812_end.pkl',
                'name': 'ugc_2698_newmasks/u2698_nest_priset32_d91_baseline_rhe_orig_nog_3sig.png',
                'cornername': 'ugc_2698_newmasks/u2698_nest_priset32_d91_baseline_rhe_orig_nog_corner_3sig.png',
                'inpf': 'ugc_2698/ugc_2698_priset32_d91_baseline_rhe_orig_nog.txt',
                'outpf': 'ugc_2698/ugc_2698_priset32_d91_baseline_rhe_orig_nog_out.txt',
                'mod': 'rhe baseline (priset32, d91)', 'extra_params': None}

# RE_RUNNING MGE SETS: RRE BASELINE // 5.36hrs (154349.38s); 121481 total calls; 51351 initial calls
dictrre2set3 = {'pkl': 'u2698_baseline_rre_orig_nog_10000000_8_0.02_1591298519.005904_end.pkl',
                'name': 'ugc_2698_newmasks/u2698_nest_baseline_rre_orig_nog_3sig.png',
                'cornername': 'ugc_2698_newmasks/u2698_nest_baseline_rre_orig_nog_corner_3sig.png',
                'inpf': 'ugc_2698/ugc_2698_baseline_rre_orig_nog.txt',
                'outpf': 'ugc_2698/ugc_2698_baseline_rre_orig_nog_out.txt',
                'mod': 'rre baseline (d91)', 'extra_params': None}

# RE_RUNNING MGE SETS: AHE BASELINE // 6.31hrs (181747.31s); 144642 total calls; 44556 initial calls
dictahe2set3 = {'pkl': 'u2698_baseline_ahe_orig_nog_10000000_8_0.02_1591301727.9228723_end.pkl',
                'name': 'ugc_2698_newmasks/u2698_nest_baseline_ahe_orig_nog_3sig.png',
                'cornername': 'ugc_2698_newmasks/u2698_nest_baseline_ahe_orig_nog_corner_3sig.png',
                'inpf': 'ugc_2698/ugc_2698_baseline_ahe_orig_nog.txt',
                'outpf': 'ugc_2698/ugc_2698_baseline_ahe_orig_nog_out.txt',
                'mod': 'ahe baseline (d91)', 'extra_params': None}

# RE_RUNNING MGE SETS: AKIN BASELINE // 13.73hrs (395424.19s); 165523 total calls; 55631 initial calls
dictakin2set3 = {'pkl': 'u2698_baseline_akin_orig_nog_10000000_8_0.02_1591329308.3132455_end.pkl',
                 'name': 'ugc_2698_newmasks/u2698_nest_baseline_akin_orig_nog_3sig.png',
                 'cornername': 'ugc_2698_newmasks/u2698_nest_baseline_akin_orig_nog_corner_3sig.png',
                 'inpf': 'ugc_2698/ugc_2698_baseline_akin_orig_nog.txt',
                 'outpf': 'ugc_2698/ugc_2698_baseline_akin_orig_nog_out.txt',
                 'mod': 'akin baseline (d91)', 'extra_params': None}

# RHE BASELINE RFIT 1.0 // 6.04hrs (174034.81s); 141325 total calls; 45170 initial calls
dictrhe2rfit1 = {'pkl': 'u2698_rfit10_baseline_rhe_orig_nog_10000000_8_0.02_1591306649.6288242_end.pkl',
                 'name': 'ugc_2698_newmasks/u2698_nest_rfit10_baseline_rhe_orig_nog_3sig.png',
                 'cornername': 'ugc_2698_newmasks/u2698_nest_rfit10_baseline_rhe_orig_nog_corner_3sig.png',
                 'inpf': 'ugc_2698/ugc_2698_rfit10_baseline_rhe_orig_nog.txt',
                 'outpf': 'ugc_2698/ugc_2698_rfit10_baseline_rhe_orig_nog_out.txt',
                 'mod': 'rhe baseline rfit10', 'extra_params': None}

# RHE BASELINE RFIT 05 // 9.60hrs (276603.16s); 197189 total calls; 51459 initial calls
dictrhe2rfit05 = {'pkl': 'u2698_rfit05_baseline_rhe_orig_nog_10000000_8_0.02_1591597783.1773756_end.pkl',
                  'name': 'ugc_2698_newmasks/u2698_nest_rfit05_baseline_rhe_orig_nog_3sig.png',
                  'cornername': 'ugc_2698_newmasks/u2698_nest_rfit05_baseline_rhe_orig_nog_corner_3sig.png',
                  'inpf': 'ugc_2698/ugc_2698_rfit05_baseline_rhe_orig_nog.txt',
                  'outpf': 'ugc_2698/ugc_2698_rfit05_baseline_rhe_orig_nog_out.txt',
                  'mod': 'rhe baseline rfit05', 'extra_params': None}

# RHE BASELINE RFIT 03 // original 21019 calls; latest at 281155 calls
dictrhe2rfit03 = {'pkl': 'u2698_rfit03_baseline_rhe_orig_nog_10000000_8_10.0_1592252035.7531602_tempsave.pkl',
                  'name': 'ugc_2698_newmasks/u2698_nest_rfit03_baseline_rhe_orig_nog_3sig.png',
                  'cornername': 'ugc_2698_newmasks/u2698_nest_rfit03_baseline_rhe_orig_nog_corner_3sig.png',
                  'inpf': 'ugc_2698/ugc_2698_rfit03_baseline_rhe_orig_nog.txt',
                  'outpf': 'ugc_2698/ugc_2698_rfit03_baseline_rhe_orig_nog_out.txt',
                  'mod': 'rhe baseline rfit03', 'extra_params': None}
# rfit03: 310743 323595 333650

# RHE BASELINE EXPSIG // 37.97hrs (1093493.88s); 848668 total calls; 432880 initial calls
dictrhe2exp = {'pkl': 'u2698_expsig_baseline_rhe_orig_nog_10000000_8_0.02_1591464018.0182757_end.pkl',
               'name': 'ugc_2698_newmasks/u2698_nest_expsig_baseline_rhe_orig_nog_3sig.png',
               'cornername': 'ugc_2698_newmasks/u2698_nest_expsig_baseline_rhe_orig_nog_corner_3sig.png',
               'inpf': 'ugc_2698/ugc_2698_expsig_baseline_rhe_orig_nog.txt',
               'outpf': 'ugc_2698/ugc_2698_expsig_baseline_rhe_orig_nog_out.txt',
               'mod': 'rhe baseline expsig', 'extra_params': [['r0', 'pc'], ['sig1', 'km/s']]}

# RHE BASELINE JUSTEXV // 4.87hrs (140325.02s); 95871 total calls; 43280 initial calls
dictrhe2jexv = {'pkl': 'u2698_justexv2582_10000000_8_0.02_1592357354.1659029_end.pkl',
                'name': 'ugc_2698_newmasks/u2698_nest_justexv_3sig.png',
                'cornername': 'ugc_2698_newmasks/u2698_nest_justexv_corner_3sig.png',
                'inpf': 'ugc_2698/ugc_2698_justexv.txt',
                'outpf': 'ugc_2698/ugc_2698_justexv_out.txt',
                'mod': 'rhe baseline exv', 'extra_params': None}

# CHOOSE DICTIONARY, DEFINE LABELS
dict = dictrhe2jexv
if 'nobh' in dict['pkl']:
    labels = np.array(['xloc', 'yloc', 'sig0', 'inc', 'PAdisk', 'vsys', 'ml_ratio', 'f'])
    ax_lab = np.array(['pixels', 'pixels', 'km/s', 'deg', 'deg', 'km/s', r'M$_{\odot}$/L$_{\odot}$', 'unitless'])
    tablabs = np.array(['reduced chi^2', 'xloc [pix]', 'yloc [pix]', 'sig0 [km/s]', 'inc [deg]',
                        'PAdisk [deg]', 'vsys [km/s]', 'ml_ratio [Msol/Lsol]', 'f [unitless]'])
else:
    labels = np.array(['mbh', 'xloc', 'yloc', 'sig0', 'inc', 'PAdisk', 'vsys', 'ml_ratio', 'f'])
    ax_lab = np.array([r'$\log_{10}$(M$_{\odot}$)', 'pixels', 'pixels', 'km/s', 'deg', 'deg', 'km/s',
                       r'M$_{\odot}$/L$_{\odot}$', 'unitless'])
    tablabs = np.array(['reduced chi^2', 'log10(mbh) [Msol]', 'xloc [pix]', 'yloc [pix]', 'sig0 [km/s]', 'inc [deg]',
                        'PAdisk [deg]', 'vsys [km/s]', 'ml_ratio [Msol/Lsol]', 'f [unitless]'])
if dict['extra_params'] is not None:
    for par in dict['extra_params']:
        labels = np.append(labels, par[0])
        ax_lab = np.append(ax_lab, par[1])
        tablabs = np.append(tablabs, par[0] + ' [' + par[1] + ']')

print(labels)

'''  #
<details><summary>rfit = 0.3 arcsec</summary>

* First I show the ellipse with rfit = 0.3 arcsec (using a PA = 19 deg, q = 0.38). Then I show the normal posterior plots.

   ![ellipse](https://github.tamu.edu/joncohn/gas-dynamical-modeling/blob/master/2020-06-01/ellipse_0.38_19_0.3.png)

   ![1D posteriors](https://github.tamu.edu/joncohn/gas-dynamical-modeling/blob/master/2020-06-01/u2698_nest_rfit03_baseline_rhe_orig_nog_3sig.png)

   ![corner plot](https://github.tamu.edu/joncohn/gas-dynamical-modeling/blob/master/2020-06-01/u2698_nest_rfit03_baseline_rhe_orig_nog_corner_3sig.png)

</details>

# ONLY table_it *AFTER* OUT FILE CREATED
hd, hl, li, tx = table_it([direc + dict['pkl']], [dict['outpf']], [dict['mod']], tablabs, sig=1)
print(hd)
print(hl)
print(li)
print(tx)
print(oop)
# '''  #

'''  #
# TABLE
thinglist = [direc + 'dyndyn_newpri_3_maxc10mil_n8_0.02_1572037132.3993847_tempsave.pkl', direc + dictakin2['pkl'],
             direc + dictrre2['pkl'], direc + dictrrelax['pkl'], direc + dictrrestrict['pkl'], direc + dictahe2['pkl'],
             direc + dictahelax['pkl'], direc + dictahestrict['pkl'], direc + dictrhe2['pkl'],
             direc + dictrhelax['pkl'], direc + dictrhestrict['pkl']]
paramfiles = ['ugc_2698/ugc_2698_newpriout_n8.txt', dictakin2['outpf'], dictrre2['outpf'], dictrrelax['outpf'],
              dictrrestrict['outpf'], dictahe2['outpf'], dictahelax['outpf'], dictahestrict['outpf'], dictrhe2['outpf'],
              dictrhelax['outpf'], dictrhestrict['outpf']]
modlist = ['old cube', 'akin baseline', 'rre baseline', 'rre lax', 'rre strict', 'ahe baseline', 'ahe lax', 'ahe strict',
           'rhe baseline', 'rhe lax', 'rhe strict']
hd, hl, li, tx = table_it(thinglist, paramfiles, modlist, tablabs)
print(hd)
print(hl)
print(li)
print(tx)
print(oop)
# '''  #

out_name = direc + dict['pkl']


# https://stackoverflow.com/questions/2121874/python-pickling-after-changing-a-modules-directory
with open(out_name, 'rb') as pk:
    u = pickle._Unpickler(pk)
    u.encoding = 'latin1'
    dyn_res = u.load()  #
    # dyn_res = pickle.load(pk)  #
print(dyn_res['samples'].shape)

# 3-D plots of position and likelihood, colored by weight
# fig = plt.figure(figsize=(30, 10))
# ax = fig.add_subplot(121, projection='3d')

# How to do quantiles!
weights = np.exp(dyn_res['logwt'] - dyn_res['logz'][-1])  # normalized weights
three_sigs = []
one_sigs = []
if dict['inpf'] is not None:
    with open(dict['inpf'], 'r') as inpff:
        outpf = dict['inpf'][:-4] + '_out.txt'
        print(dict['inpf'], outpf)
        if not Path(outpf).exists():
            with open(outpf, 'w+') as outpff:
                idx = 0
                for line in inpff:
                    if line.startswith('free'):
                        insert = str(dyfunc.quantile(dyn_res['samples'][:, idx], [0.0015, 0.5, 0.9985],
                                                     weights=weights)[1])
                        idx += 1
                        cols = line.split()
                        cols[2] = insert
                        line = ' '.join(cols) + '\n'
                    outpff.write(line)


for i in range(dyn_res['samples'].shape[1]):  # for each parameter
    quantiles_3 = dyfunc.quantile(dyn_res['samples'][:, i], [0.0015, 0.5, 0.9985], weights=weights)
    quantiles_2 = dyfunc.quantile(dyn_res['samples'][:, i], [0.025, 0.5, 0.975], weights=weights)
    quantiles_1 = dyfunc.quantile(dyn_res['samples'][:, i], [0.16, 0.5, 0.84], weights=weights)
    print(labels[i])
    if 'xy' in dict['pkl']:
        print(quantiles_3)
        print(quantiles_2)
        print(quantiles_1)
        three_sigs.append(quantiles_3)
        one_sigs.append(quantiles_1)
    elif i == 0 and 'nobh' not in dict['pkl']:
        print(np.log10(quantiles_3), quantiles_3)
        print(np.log10(quantiles_2), quantiles_2)
        print(np.log10(quantiles_1), quantiles_1)
        three_sigs.append(np.log10(quantiles_3))
        one_sigs.append(np.log10(quantiles_1))
    else:
        print(quantiles_3)
        print(quantiles_2)
        print(quantiles_1)
        three_sigs.append(quantiles_3)
        one_sigs.append(quantiles_1)
from dynesty import plotting as dyplot

'''
# initialize figure
fig, axes = plt.subplots(3, 7, figsize=(35, 15))
axes = axes.reshape((3, 7))  # reshape axes

# add white space
[a.set_frame_on(False) for a in axes[:, 3]]
[a.set_xticks([]) for a in axes[:, 3]]
[a.set_yticks([]) for a in axes[:, 3]]
# '''
sig1 = [0.16, 0.5, 0.84]
sig2 = [0.025, 0.5, 0.975]
sig3 = [0.0015, 0.5, 0.9985]

vax = np.zeros(shape=len(labels))
vwidth = np.zeros(shape=(len(labels), 2))
with open('/Users/jonathancohn/Documents/dyn_mod/param_files/Ben_A1_errors.txt') as a1:
    for line in a1:
        cols = line.split()
        if not line.startswith('#'):
            if cols[0] == 'mbh':
                vax[np.where(labels == cols[0])] = np.log10(float(cols[1]))
                vwidth[np.where(labels == cols[0]), 0] = np.log10(float(cols[1]) - float(cols[2]))
                vwidth[np.where(labels == cols[0]), 1] = np.log10(float(cols[1]) + float(cols[2]))
            else:
                vax[np.where(labels == cols[0])] = float(cols[1])
                vwidth[np.where(labels == cols[0]), 0] = float(cols[1]) - float(cols[2])
                vwidth[np.where(labels == cols[0]), 1] = float(cols[1]) + float(cols[2])
print(vax)
print(vwidth)
print(vax[labels=='sig0'])
print(vwidth[labels=='sig0'])

logm = True
if logm and 'xy' not in dict['pkl'] and 'nobh' not in dict['pkl']:
    dyn_res['samples'][:, 0] = np.log10(dyn_res['samples'][:, 0])
    labels[0] = 'log mbh'  # r'log$_{10}$mbh'

ax_lims = None
if 'xy' in dict['pkl']:
    labels = np.array(['xloc', 'yloc'])
    ax_lab = ['pixels', 'pixels']
    ax_lims = [[361.97, 362.08], [354.84, 354.975]]
    my_own_xy(dyn_res['samples'], labels, ax_lab, one_sigs, ax_lims=ax_lims, comp2=True, err=1)  # one_sigma
    # my_own_xy(dyn_res['samples'], labels, ax_lab, three_sigs, ax_lims=ax_lims, comp2=True, err=3)  # three_sigma
    print(oop)
    '''
    [362.021449724303, 362.0429280235011, 362.04897481883864]
    [362.038033081247, 362.0429280235011, 362.04651339175155]
    [362.0414293129913, 362.0429280235011, 362.04455921103]
    xloc
    [354.87933747089113, 354.8935055084055, 354.9029591740998]
    [354.8857249803428, 354.8935055084055, 354.8994498474415]
    [354.8901495992812, 354.8935055084055, 354.8965309525797]
    yloc

    '''

elif '3258' in out_name:
    # ax_lims = [[9.2, 9.6], [126.2, 127.8], [150.2, 151.8], [7.5, 14.], [66., 70.], [19.13, 19.23], [6447., 6462.],
    #           [1.52, 1.8], [0.93, 1.2]]
    # ax_lims = None
    # ax_lims = [[9.36, 9.4], [361.97, 362.08], [354.75, 355.], [7., 11.], [45.5, 46.5], [166.3, 167], [2760.5, 2761],
    #            [3.1, 3.22], [1.015, 1.03]]  # USE THESE IF NOT COMPARING TO BEN'S A1 SAMPLING ERRORS
    ax_lims = [[9.364, 9.391], [361.97, 362.08], [354.84, 354.975], [8.5, 10.75], [45.5, 46.5], [166.4, 167.1],
               [2760.5, 2761], [3.08, 3.23], [1.015, 1.03]]
elif 'maskstrict' in out_name and 'rhe' in out_name:
    ax_lims = [[9.3, 9.6], [126.2, 127.8], [150.2, 151.8], [7.8, 21.4], [66., 72.], [15., 23.], [6447., 6462.],
               [1.52, 1.8], [0.93, 1.2]]  # [19.13, 19.23]
elif 'mask' in out_name and 'rre' in out_name:
    ax_lims = [[9.0, 9.3], [126.2, 127.8], [150.2, 151.8], [7.8, 21.4], [66., 70.], [15., 23.], [6447., 6462.],
               [1.7, 2.2], [0.93, 1.2]]  # [19.13, 19.23]
elif 'mask' in out_name and 'akin' in out_name:
    ax_lims = [[9.3, 9.6], [126.2, 127.8], [150.2, 151.8], [7.8, 21.4], [66., 70.], [15., 23.], [6447., 6462.],
               [1.1, 1.7], [0.93, 1.2]]  # [19.13, 19.23]
elif 'vrad' in out_name and 'rhe' in out_name:
    ax_lims = [[9.3, 9.6], [126.2, 127.8], [150.2, 151.8], [7.8, 21.4], [66., 72.], [15., 23.], [6447., 6462.],
               [1.52, 1.8], [0.93, 1.2], [-25., 35.]]
elif 'kappa' in out_name and 'rhe' in out_name:
    ax_lims = [[9.2, 9.5], [126.2, 127.8], [150.2, 151.8], [12.5, 21.4], [66., 71.], [17., 21.], [6447., 6462.],
               [1.61, 1.95], [0.93, 1.2], [-0.05, 0.05]]
elif 'omega' in out_name and 'rhe' in out_name:
    ax_lims = [[9.1, 9.8], [126.2, 127.8], [150.2, 151.8], [12.5, 23.], [66., 71.], [16.5, 21.], [6447., 6462.],
               [1.52, 2.2], [0.93, 1.2], [-0.1, 0.1], [0.7, 1.]]
elif 'mask' in out_name:
    ax_lims = [[9.3, 9.6], [126.2, 127.8], [150.2, 151.8], [7.8, 21.4], [66., 70.], [15., 23.], [6447., 6462.],
               [1.52, 1.8], [0.93, 1.2]]  # [19.13, 19.23]
elif 'expsig' in out_name:
    ax_lims = [[8., 10.], [124., 128.], [148, 152], [0., 100.], [52.4, 85], [5., 35.], [6405, 6505],
               [0.3, 3.], [0.5, 1.5], [0., 100.], [0., 100.]]  # [19.13, 19.23]
elif 'set3' in out_name or 'set2' in out_name or 'rre' in out_name or 'ahe' in out_name or 'akin' in out_name\
        or 'rfit' in out_name or 'jexv' in out_name:
    ax_lims = [[8., 10.], [124., 128.], [148, 152], [0., 40.], [52.4, 85], [5., 35.], [6405, 6505],
               [0.3, 3.], [0.5, 1.5]]  # [19.13, 19.23]
elif 'nobh' in out_name:
    ax_lims = [[124., 128.], [148, 152], [0., 40.], [52.4, 85], [5., 35.], [6405, 6505],
               [0.3, 3.], [0.5, 1.5]]  # [19.13, 19.23]
elif 'd91' in out_name or 'rhe' in out_name:
    ax_lims = [[9.32, 9.47], [126.2, 127.8], [150.2, 151.8], [13., 21.4], [66., 69.5], [17., 20.8], [6450.5, 6459.5],
               [1.59, 1.82], [1.01, 1.2]]  # [19.13, 19.23]
elif 'd85' in out_name:
    ax_lims = [[9.3, 9.47], [126.2, 127.8], [150.2, 151.8], [13., 21.4], [66., 69.5], [17., 20.8], [6450.5, 6459.5],
               [1.69, 1.91], [1.01, 1.2]]  # [19.13, 19.23]
else:
    ax_lims = [[9.3, 9.47], [126.2, 127.8], [150.2, 151.8], [13., 21.4], [66., 69.5], [17., 20.8], [6450.5, 6459.5],
               [1.63, 1.81], [1.01, 1.2]]  # [19.13, 19.23]
    # ax_lims=None

# USE FOR INCLUDING BEN'S A1 ERRORS
#my_own_thing(dyn_res['samples'], labels, ax_lab, one_sigs, ax_lims=ax_lims, compare_err=True)  # three_sigs
#print(oop)
# my_own_thing(dyn_res['samples'], labels, ax_lab, one_sigs, ax_lims=ax_lims, comp2=True, err=1)  # one_sigma
# my_own_thing(dyn_res['samples'], labels, ax_lab, three_sigs, ax_lims=ax_lims, comp2=True, err=3)  # three_sigma

# ELSE USE THIS
vrad = False
if 'vrad' in dict['name']:
    vrad = True
my_own_thing(dyn_res['samples'], labels, ax_lab, three_sigs, ax_lims=ax_lims, savefig=grp + dict['name'], vrad=vrad)
             # fs=8)

# plot initial run (res1; left)

# TO EDIT SOURCE CODE: open /Users/jonathancohn/anaconda3/envs/three/lib/python3.6/site-packages/dynesty/plotting.py

# USE THIS FOR COMPARING 3258 TO BEN'S A1 MODEL:
# fg, ax = dyplot.cornerplot(dyn_res, color='blue', show_titles=True, max_n_ticks=3, quantiles=sig1, labels=labels,
#                            compare_med=vax, compare_width=vwidth)
# plt.show()

# OTHERWISE USE THIS:
ndim = len(labels)
factor = 2.0  # size of side of one panel
lbdim = 0.5 * factor  # size of left/bottom margin
trdim = 0.2 * factor  # size of top/right margin
whspace = 0.05  # size of width/height margin
plotdim = factor * ndim + factor * (ndim - 1.) * whspace  # plot size
dim = lbdim + plotdim + trdim  # total size
fig, axes = plt.subplots(ndim, ndim, figsize=(1.7*dim, dim))
fg, ax = dyplot.cornerplot(dyn_res, color='blue', show_titles=True, max_n_ticks=3, quantiles=sig3, labels=labels,
                           fig=(fig, axes))
plt.savefig(grp + dict['cornername'])
#plt.show()
# '''  #
# ONLY table_it *AFTER* OUT FILE CREATED
hd, hl, li, tx = table_it([direc + dict['pkl']], [dict['outpf']], [dict['mod']], tablabs)
print(hd)
print(hl)
print(li)
print(tx)
# '''  #
print(oop)


'''
# 3258 dyndyn3258 priors:
    # NARROWEST PRIORS based on short MCMC
    cube[0] = 10 ** (cube[0] * 0.1 + 9.3)  # mbh: log-uniform prior 1e9.3:1e9.4
    cube[1] = cube[1] + 361.5  # xloc: uniform prior 361.5:362.5
    cube[2] = cube[2] * 0.5 + 354.75  # yloc: uniform prior 354.75:355.25
    cube[3] = cube[3] * 10. + 5.  # sig0: uniform prior 5:15
    cube[4] = cube[4] * 2. + 45.4  # inc: uniform prior 45.4:47.4 (low pri=49.83663935008522 from MGE q)
    cube[5] = cube[5] * 5. + 166.25  # PAdisk: uniform prior 166.25:167.25
    cube[6] = cube[6] * 2. + 2760.  # vsys: uniform prior 2760:2762
    cube[7] = cube[7] * 0.2 + 3.05  # mlratio: uniform prior 3.05:3.25
    cube[8] = cube[8] * 0.03 + 1.005  # f: uniform prior 1.005:1.035

# 3258 newpri 2
    cube[0] = 10 ** (cube[0] * 2. + 8.)  # mbh: log-uniform prior 1e9.3:1e9.4
    cube[1] = cube[1] * 5. + 360.  # xloc: uniform prior 360:365
    cube[2] = cube[2] * 3. + 353.  # yloc: uniform prior 353:356
    cube[3] = cube[3] * 15.  # sig0: uniform prior 0:15
    cube[4] = cube[4] * 40. + 45.  # inc: uniform prior 44.5:89.5 (low pri=44.01472043806415 from MGE q)
    cube[5] = cube[5] * 110. + 100.  # PAdisk: uniform prior 100:210
    cube[6] = cube[6] * 50. + 2730.  # vsys: uniform prior 2730:2780
    cube[7] = cube[7] * 5. + 0.5  # mlratio: uniform prior 0.5:5.5
    cube[8] = cube[8] + 0.5  # f: uniform prior 0.5:1.5

# 3258 newpri 3
    cube[0] = 10 ** (cube[0] * 2. + 8.)  # mbh: log-uniform prior 1e9.3:1e9.4
    cube[1] = cube[1] * 15. + 355.  # xloc: uniform prior 360:365
    cube[2] = cube[2] * 15. + 347.  # yloc: uniform prior 353:356
    cube[3] = cube[3] * 15.  # sig0: uniform prior 0:15
    cube[4] = cube[4] * 30. + 46.  # inc: uniform prior 44.5:89.5 (low pri=44.01472043806415 from MGE q)
    cube[5] = cube[5] * 100. + 100.  # PAdisk: uniform prior 100:210
    cube[6] = cube[6] * 50. + 2730.  # vsys: uniform prior 2730:2780
    cube[7] = cube[7] * 5. + 0.5  # mlratio: uniform prior 0.5:5.5
    cube[8] = cube[8] + 0.5  # f: uniform prior 0.5:1.5

# 3258 newpri 4
    cube[0] = 10 ** (cube[0] * 2. + 8.)  # mbh: log-uniform prior 1e9.3:1e9.4
    cube[1] = cube[1] * 5. + 360.  # xloc: uniform prior 360:365
    cube[2] = cube[2] * 9. + 351.  # yloc: uniform prior 353:356
    cube[3] = cube[3] * 15.  # sig0: uniform prior 0:15
    cube[4] = cube[4] * 20. + 45.  # inc: uniform prior 44.5:89.5 (low pri=44.01472043806415 from MGE q)
    cube[5] = cube[5] * 45. + 135.  # PAdisk: uniform prior 100:210
    cube[6] = cube[6] * 50. + 2730.  # vsys: uniform prior 2730:2780
    cube[7] = cube[7] * 3. + 1.  # mlratio: uniform prior 0.5:5.5
    cube[8] = cube[8] + 0.5  # f: uniform prior 0.5:1.5

# 3258 newpri 5
    cube[0] = 10 ** (cube[0] * 2. + 8.)  # mbh: log-uniform prior 1e9.3:1e9.4
    cube[1] = cube[1] * 5. + 360.  # xloc: uniform prior 360:365
    cube[2] = cube[2] * 5. + 353.  # yloc: uniform prior 353:356
    cube[3] = cube[3] * 15.  # sig0: uniform prior 0:15
    cube[4] = cube[4] * 15. + 45.  # inc: uniform prior 44.5:89.5 (low pri=44.01472043806415 from MGE q)
    cube[5] = cube[5] * 45. + 135.  # PAdisk: uniform prior 100:210
    cube[6] = cube[6] * 50. + 2730.  # vsys: uniform prior 2730:2780
    cube[7] = cube[7] * 2. + 2.  # mlratio: uniform prior 0.5:5.5
    cube[8] = cube[8] + 0.5  # f: uniform prior 0.5:1.5

# 3258 newpriN 5
    cube[0] = 10 ** (cube[0] * 2. + 8.)  # mbh: log-uniform prior 1e9.3:1e9.4
    cube[1] = cube[1] * 5. + 360.  # xloc: uniform prior 360:365
    cube[2] = cube[2] * 5. + 353.  # yloc: uniform prior 353:356
    cube[3] = cube[3] * 15.  # sig0: uniform prior 0:15
    cube[4] = cube[4] * 15. + 45.  # inc: uniform prior 44.5:89.5 (low pri=44.01472043806415 from MGE q)
    cube[5] = cube[5] * 20. + 150.  # PAdisk: uniform prior 100:210
    cube[6] = cube[6] * 50. + 2730.  # vsys: uniform prior 2730:2780
    cube[7] = cube[7] * 2. + 2.  # mlratio: uniform prior 0.5:5.5
    cube[8] = cube[8] + 0.5  # f: uniform prior 0.5:1.5
'''

'''
# 2698 dyndyn2 priors:
    # NARROWEST PRIORS based on short MCMC
    cube[0] = 10 ** (cube[0] * 0.8 + 9.2)  # mbh: log-uniform prior 10^9.2:10^10
    cube[1] = cube[1] * 3.5 + 126.  # xloc: uniform prior 126:129.5
    cube[2] = cube[2] * 1.75 + 150.25  # yloc: uniform prior 150.25:152
    cube[3] *= 15.  # sig0: uniform prior 0:15
    cube[4] = cube[4] * 5. + 66.  # inc: uniform prior 66:71 (low pri=49.83663935008522 from MGE q)
    cube[5] = cube[5] * 4. + 17.  # PAdisk: uniform prior 17:21
    cube[6] = cube[6] * 50. + 6440.  # vsys: uniform prior 6440:6490
    cube[7] = cube[7] * 0.45 + 1.45  # mlratio: uniform prior 1.45:1.90
    cube[8] = cube[8] * 0.85 + 0.85  # f: uniform prior 0.85:1.7

# 2698 dyndyn_newpri priors (GOOD OUTPUT)
    cube[0] = 10 ** (cube[0] * 2. + 8.)  # mbh: log-uniform prior 10^8:10^10
    cube[1] = cube[1] * 6. + 124.  # xloc: uniform prior 124:130
    cube[2] = cube[2] * 4. + 149.  # yloc: uniform prior 149:153
    cube[3] *= 15.  # sig0: uniform prior 0:15
    cube[4] = cube[4] * 5. + 66.  # inc: uniform prior 66:71 (low pri=49.83663935008522 from MGE q)
    cube[5] = cube[5] * 4. + 17.  # PAdisk: uniform prior 17:21
    cube[6] = cube[6] * 50. + 6440.  # vsys: uniform prior 6440:6490
    cube[7] = cube[7] + 1.  # mlratio: uniform prior 1:2
    cube[8] = cube[8] * 0.85 + 0.85  # f: uniform prior 0.85:1.7

# 2698 dyndyn_newpri 2
    cube[0] = 10 ** (cube[0] * 2. + 8.)  # mbh: log-uniform prior 10^8:10^10
    cube[1] = cube[1] * 6. + 124.  # xloc: uniform prior 124:130
    cube[2] = cube[2] * 4. + 149.  # yloc: uniform prior 149:153
    cube[3] *= 15.  # sig0: uniform prior 0:15
    cube[4] = cube[4] * 20. + 60.  # inc: uniform prior 66:71 (low pri=49.83663935008522 from MGE q)
    cube[5] = cube[5] * 35. + 10.  # PAdisk: uniform prior 17:21
    cube[6] = cube[6] * 100. + 6400.  # vsys: uniform prior 6440:6490
    cube[7] = cube[7] + 1.  # mlratio: uniform prior 1:2
    cube[8] = cube[8] * 1.5 + 0.5  # f: uniform prior 0.85:1.7

# 2698 dyndyn_newpri 3 (NARROWER ON ALL BUT PA, FOLLOWING newpri 2 results)
    cube[0] = 10 ** (cube[0] * 0.6 + 9.2)  # mbh: log-uniform prior 10^8:10^10
    cube[1] = cube[1] * 2.6 + 125.8  # xloc: uniform prior 124:130
    cube[2] = cube[2] * 2. + 150.  # yloc: uniform prior 149:153
    cube[3] = cube[3] * 7.5 + 7.5  # sig0: uniform prior 0:15
    cube[4] = cube[4] * 10. + 63.  # inc: uniform prior 66:71 (low pri=49.83663935008522 from MGE q)
    cube[5] = cube[5] * 35. + 10.  # PAdisk: uniform prior 17:21
    cube[6] = cube[6] * 30. + 6435.  # vsys: uniform prior 6440:6490
    cube[7] = cube[7] * 0.6 + 1.4  # mlratio: uniform prior 1:2
    cube[8] = cube[8] * 0.65 + 0.75  # f: uniform prior 0.85:1.7
# relative time taken: nc 79745 niter 15290 (end: nc 154073 niter 32858)
# CPU time: 303927.34s (~10.5 hours over 8 CPUs)
# runtime: 39369.12614393234
'''

'''
# 2698 105newpri
    cube[0] = 10 ** (cube[0] * 2. + 8.)  # mbh: log-uniform prior 10^8:10^10
    cube[1] = cube[1] * 6. + 124.  # xloc: uniform prior 123:131
    cube[2] = cube[2] * 6. + 148.  # yloc: uniform prior 147:155
    cube[3] *= 15.  # sig0: uniform prior 0:15
    cube[4] = cube[4] * 39. + 50.  # inc: uniform prior 50:89 (low pri=49.83663935008522 from MGE q)
    cube[5] = cube[5] * 45.  # PAdisk: uniform prior 0:45
    cube[6] = cube[6] * 100. + 6400.  # vsys: uniform prior 6400:6500
    cube[7] = cube[7] * 4.9 + 0.1  # mlratio: uniform prior 0.1:5
    cube[8] = cube[8] * 1.5 + 0.5  # f: uniform prior 0.5:2
    
# 2698 105newpri 2
    cube[0] = 10 ** (cube[0] * 2. + 8.)  # mbh: log-uniform prior 10^8:10^10
    cube[1] = cube[1] * 4. + 125.  # xloc: uniform prior 123:131
    cube[2] = cube[2] * 4. + 149.  # yloc: uniform prior 147:155
    cube[3] *= 15.  # sig0: uniform prior 0:15
    cube[4] = cube[4] * 20. + 60.  # inc: uniform prior 50:89 (low pri=49.83663935008522 from MGE q)
    cube[5] = cube[5] * 20. + 10.  # PAdisk: uniform prior 0:45
    cube[6] = cube[6] * 100. + 6400.  # vsys: uniform prior 6400:6500
    cube[7] = cube[7] * 1.9 + 0.1  # mlratio: uniform prior 0.1:2
    cube[8] = cube[8] * 1.5 + 0.5  # f: uniform prior 0.5:2
# relative time taken: nc 515184 niter 18808
# CPU time: 1276895.38s (44 hours/8 CPUs)

# 2698 105newpri 3
    cube[0] = 10 ** (cube[0] * 2. + 8.)  # mbh: log-uniform prior 10^8:10^10
    cube[1] = cube[1] * 4. + 125.  # xloc: uniform prior 123:131
    cube[2] = cube[2] * 4. + 149.  # yloc: uniform prior 147:155
    cube[3] *= 15.  # sig0: uniform prior 0:15
    cube[4] = cube[4] * 10. + 65.  # inc: uniform prior 50:89 (low pri=49.83663935008522 from MGE q)
    cube[5] = cube[5] * 10. + 15.  # PAdisk: uniform prior 0:45
    cube[6] = cube[6] * 50. + 6440.  # vsys: uniform prior 6400:6500
    cube[7] = cube[7] * 1.9 + 0.1  # mlratio: uniform prior 0.1:5
    cube[8] = cube[8] * 1.5 + 0.5  # f: uniform prior 0.5:2
# relative time taken: nc 115840 niter 18389
# CPU time: 365789.09s (12 hours/8 CPUs)
'''
