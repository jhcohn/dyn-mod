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


def big_table(output_dicts, logged=False, avg=True):

    texlines = '\\begin{table}\n    \\begin{center}\n    \\textbf{Table 3} \\\\\n    Model test results \\\\\n    ' +\
               '\\begin{tabular}{ |l|r|r|r| }\n    \hline\n    Model & \multicolumn{1}{|l|}{$\chi^2_\\nu$}' +\
               ' & \multicolumn{1}{|l|}{\mbh [$\\times10^9M_\odot$]} & $\Delta$\mbh \\\\\n    \hline\n    \hline\n'

    direc = '/Users/jonathancohn/Documents/dyn_mod/nest_out/'
    mbh_fiducial = 2461189947.064265 / 1e9

    table_names = {'ahe': 'dust-corrected', 'dlogz0.001': 'dlogz = 0.001', 'ds48': 'ds $=4\\times8$',
                   'ds510': 'ds $=5\\times10$', 'exp': 'expsig', 'fiducial': 'fiducial', 'fullpriors': 'wide priors',
                   'gas': 'gas', 'kappa': '$\kappa$', 'lucyn5': 'Lucy n = 5', 'lucyn15': 'Lucy n = 15',
                   'nlive1000': 'nlive = 1000', 'os1': '$s=1$', 'os2': '$s=2$', 'os3': '$s=3$', 'os6': '$s=6$',
                   'os8': '$s=8$', 'os10': '$s=10$', 'os12': '$s=12$', 'rfit0.3': 'r$_{\\text{ell}}=0\\farcs{3}$',
                   'rfit0.4': 'r$_{\\text{ell}}=0\\farcs{4}$', 'rfit0.5': 'r$_{\\text{ell}}=0\\farcs{5}$',
                   'rfit0.6': 'r$_{\\text{ell}}=0\\farcs{6}$', 'rfit0.8': 'r$_{\\text{ell}}=0\\farcs{8}$',
                   'rre': 'original $H$-band', 'vrad': 'v$_{\\text{rad}}$'}

    for od in output_dicts:
        thing = direc + output_dicts[od]['pkl']
        parfile = output_dicts[od]['outpf']
        model = output_dicts[od]['mod']
        print(model)

        params, priors, nfree, qobs = dm.par_dicts(parfile, q=True)  # get params and file names from output parfile

        if 'ds2' not in params:
            params['ds2'] = params['ds']

        mod_ins = dm.model_prep(data=params['data'], ds=params['ds'], ds2=params['ds2'], lucy_out=params['lucy'],
                                lucy_b=params['lucy_b'], lucy_mask=params['lucy_mask'], lucy_in=params['lucy_in'],
                                lucy_it=params['lucy_it'], data_mask=params['mask'], grid_size=params['gsize'],
                                res=params['resolution'], x_std=params['x_fwhm'], y_std=params['y_fwhm'], avg=avg,
                                xyerr=[params['xerr0'], params['xerr1'], params['yerr0'], params['yerr1']],
                                zrange=[params['zi'], params['zf']], theta_ell=np.deg2rad(params['theta_ell']),
                                xell=params['xell'], yell=params['yell'], q_ell=params['q_ell'], pa=params['PAbeam'])

        lucy_mask, lucy_out, beam, fluxes, freq_ax, f_0, fstep, input_data, noise, co_rad, co_sb = mod_ins
        vrad = None
        kappa = None
        omega = None
        if model == 'vrad':
            vrad = params['vrad']
        elif model == 'omega':
            kappa = params['kappa']
            omega = params['omega']
        elif model == 'kappa':
            kappa = params['kappa']

        inc_fixed = np.deg2rad(67.7)  # based on fiducial model (67.68 deg)
        vcg_in = None
        if params['incl_gas'] == 'True':
            vcg_in = dm.gas_vel(params['resolution'], co_rad, co_sb, params['dist'], f_0, inc_fixed, zfixed=0.02152)

        mg = dm.ModelGrid(x_loc=params['xloc'], y_loc=params['yloc'], mbh=params['mbh'], ml_ratio=params['ml_ratio'],
            inc=np.deg2rad(params['inc']), vsys=params['vsys'], theta=np.deg2rad(params['PAdisk']), vrad=vrad,
            kappa=kappa, omega=omega, f_w=params['f'], os=params['os'], enclosed_mass=params['mass'],
            sig_params=[params['sig0'], params['r0'], params['mu'], params['sig1']], resolution=params['resolution'],
            lucy_out=lucy_out, out_name=None, beam=beam, rfit=params['rfit'], zrange=[params['zi'], params['zf']],
            dist=params['dist'], input_data=input_data, sig_type=params['s_type'], menc_type=params['mtype'],
            theta_ell=np.deg2rad(params['theta_ell']), xell=params['xell'],yell=params['yell'], q_ell=params['q_ell'],
            ds=params['ds'], ds2=params['ds2'], reduced=True, f_0=f_0, freq_ax=freq_ax, noise=noise, bl=params['bl'],
            fstep=fstep, xyrange=[params['xi'], params['xf'], params['yi'], params['yf']], n_params=nfree,
            data_mask=params['mask'], incl_gas=params['incl_gas']=='True', co_rad=co_rad, co_sb=co_sb, vcg_func=vcg_in,
            pvd_width=params['x_fwhm'] / params['resolution'], avg=avg, quiet=True)

        mg.grids()
        mg.convolution()
        chi2, chi2_nu = mg.chi2()
        # print(model)
        print(chi2, chi2_nu)

        fmt = "{{0:{0}}}".format('.2f').format
        fmt3 = "{{0:{0}}}".format('.3f').format
        chititle = r"${{{0}}}$".format(fmt3(chi2))
        chinutitle = r"${{{0}}}$".format(fmt3(chi2_nu))
        # texlines += '    ' + table_names[model] + ' & ' + chititle + ' (' + chinutitle + ') & '
        texlines += '    ' + table_names[model] + ' & ' + chinutitle + ' & '

        with open(thing, 'rb') as pk:
            u = pickle._Unpickler(pk)
            u.encoding = 'latin1'
            dyn_res = u.load()  #
            # dyn_res = pickle.load(pk)  #

        weights = np.exp(dyn_res['logwt'] - dyn_res['logz'][-1])  # normalized weights

        quants = [0.0015, 0.5, 0.9985]
        if sig == 1 or sig == 'mod':
            quants = [0.16, 0.5, 0.84]
        elif sig == 2:
            quants = [0.025, 0.5, 0.975]
        elif sig == 3:
            quants = [0.0015, 0.5, 0.9985]

        mbh_q = dyfunc.quantile(dyn_res['samples'][:, 0], quants, weights=weights)
        if logged:
            q = np.log10(mbh_q)
        else:
            q = np.asarray(mbh_q) / 1e9

        dmbh = 100 * (q[1] - mbh_fiducial) / mbh_fiducial

        title = r"${{{0}}}_{{-{1}}}^{{+{2}}}$"
        if sig == 'mod':
            mod = (2*4597) ** (1/4)  # BUCKET 4606 changes if the fitting region changes
            texlines += title.format(fmt(q[1]), fmt((q[1] - q[0]) * mod), fmt((q[2] - q[1]) * mod)) + ' & '
        else:
            texlines += title.format(fmt(q[1]), fmt(q[1] - q[0]), fmt(q[2] - q[1])) + ' & ' + fmt(dmbh) + '\% \\\\\n' +\
                        '    \hline\n'

    texlines += '    \end{tabular}\n    \end{center}\n    \caption{\\textbf{to do}}\n    \label{tab_compare}\n' +\
                '\end{table}'

    return texlines


def table_it(things, parfiles, models, parlabels, sig=3, avg=True, logged=False, percent_diff=False):

    hdr = '| model | '
    hdrl = '| --- |'
    for la in range(len(parlabels)):
        hdrl += ' --- |'
        hdr += parlabels[la] + ' | '

    texlines = '& '
    lines = '| '

    for t in range(len(things)):
        print(t, len(things), len(parfiles), len(models))
        params, priors, nfree, qobs = dm.par_dicts(parfiles[t], q=True)  # get params and file names from output parfile

        if 'ds2' not in params:
            params['ds2'] = params['ds']

        mod_ins = dm.model_prep(data=params['data'], ds=params['ds'], ds2=params['ds2'], lucy_out=params['lucy'],
                                lucy_b=params['lucy_b'], lucy_mask=params['lucy_mask'], lucy_in=params['lucy_in'],
                                lucy_it=params['lucy_it'], data_mask=params['mask'], grid_size=params['gsize'],
                                res=params['resolution'], x_std=params['x_fwhm'], y_std=params['y_fwhm'], avg=avg,
                                xyerr=[params['xerr0'], params['xerr1'], params['yerr0'], params['yerr1']],
                                zrange=[params['zi'], params['zf']], theta_ell=np.deg2rad(params['theta_ell']),
                                xell=params['xell'], yell=params['yell'], q_ell=params['q_ell'], pa=params['PAbeam'])

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

        inc_fixed = np.deg2rad(67.7)  # based on fiducial model (67.68 deg)
        vcg_in = None
        if params['incl_gas'] == 'True':
            vcg_in = dm.gas_vel(params['resolution'], co_rad, co_sb, params['dist'], f_0, inc_fixed, zfixed=0.02152)

        mg = dm.ModelGrid(x_loc=params['xloc'], y_loc=params['yloc'], mbh=params['mbh'], ml_ratio=params['ml_ratio'],
            inc=np.deg2rad(params['inc']), vsys=params['vsys'], theta=np.deg2rad(params['PAdisk']), vrad=vrad,
            kappa=kappa, omega=omega, f_w=params['f'], os=params['os'], enclosed_mass=params['mass'],
            sig_params=[params['sig0'], params['r0'], params['mu'], params['sig1']], resolution=params['resolution'],
            lucy_out=lucy_out, out_name=None, beam=beam, rfit=params['rfit'], zrange=[params['zi'], params['zf']],
            dist=params['dist'], input_data=input_data, sig_type=params['s_type'], menc_type=params['mtype'],
            theta_ell=np.deg2rad(params['theta_ell']), xell=params['xell'],yell=params['yell'], q_ell=params['q_ell'],
            ds=params['ds'], ds2=params['ds2'], reduced=True, f_0=f_0, freq_ax=freq_ax, noise=noise, bl=params['bl'],
            fstep=fstep, xyrange=[params['xi'], params['xf'], params['yi'], params['yf']], n_params=nfree,
            data_mask=params['mask'], incl_gas=params['incl_gas']=='True', co_rad=co_rad, co_sb=co_sb, vcg_func=vcg_in,
            pvd_width=(params['x_fwhm'] + params['y_fwhm']) / params['resolution'] / 2., avg=avg)

        mg.grids()
        mg.convolution()
        chi2, chi2_nu = mg.chi2()
        print(models[t])
        print(chi2, chi2_nu)

        fmt = "{{0:{0}}}".format('.2f').format
        fmt3 = "{{0:{0}}}".format('.3f').format
        chititle = r"${{0}}$".format(fmt3(chi2))
        chinutitle = r"${{0}}$".format(fmt3(chi2_nu))
        altchititle = r"{0}".format(fmt3(chi2))
        altchinutitle = r"{0}".format(fmt3(chi2_nu))
        texlines += models[t] + ' & ' + chititle + ' & ' + chinutitle + ' & '
        lines += models[t] + ' | ' + altchititle + ' | ' + altchinutitle + ' | '

        with open(things[t], 'rb') as pk:
            u = pickle._Unpickler(pk)
            u.encoding = 'latin1'
            dyn_res = u.load()  #
            # dyn_res = pickle.load(pk)  #

        weights = np.exp(dyn_res['logwt'] - dyn_res['logz'][-1])  # normalized weights

        quants = [0.0015, 0.5, 0.9985]
        if sig == 1 or sig == 'mod':
            quants = [0.16, 0.5, 0.84]
        elif sig == 2:
            quants = [0.025, 0.5, 0.975]
        elif sig == 3:
            quants = [0.0015, 0.5, 0.9985]

        for i in range(dyn_res['samples'].shape[1]):  # for each parameter
            q = dyfunc.quantile(dyn_res['samples'][:, i], quants, weights=weights)
            print(q)
            if i == 0 and 'nobh' not in parfiles[t]:
                if logged:
                    q = np.log10(q)
                else:
                    q = np.asarray(q) / 1e9
            title = r"${{{0}}}_{{-{1}}}^{{+{2}}}$"
            alttitle = r"{0} -{1}/+{2}"
            if sig == 'mod':
                mod = (2*4597) ** (1/4)  # BUCKET 4606 changes if the fitting region changes
                texlines += title.format(fmt(q[1]), fmt((q[1] - q[0]) * mod), fmt((q[2] - q[1]) * mod)) + ' & '
                lines += alttitle.format(fmt(q[1]), fmt((q[1] - q[0]) * mod), fmt((q[2] - q[1]) * mod)) + ' | '
            else:
                texlines += title.format(fmt(q[1]), fmt(q[1] - q[0]), fmt(q[2] - q[1])) + ' & '
                lines += alttitle.format(fmt(q[1]), fmt(q[1] - q[0]), fmt(q[2] - q[1])) + ' | '
            #lines += str(q[1].format('.2f').format) + ' +' + str((q[2]-q[1]).format('.2f').format) + ' -' +\
            #         str((q[1] - q[0]).format('.2f').format) + ' | '

        if percent_diff:
            hdr += 'MBH % difference |'
            fid_bh = 2461189947.064265
            mod_bh = dyfunc.quantile(dyn_res['samples'][:, 0], quants, weights=weights)[1]
            lines += str(fmt(100 * (mod_bh - fid_bh) / fid_bh)) + '% |'
            texlines += str(fmt(100 * (mod_bh - fid_bh) / fid_bh)) + '%'

        lines += '\n| '
        texlines += '\n| '


    return hdr, hdrl, lines, texlines


def my_own_thing(results, par_labels, ax_labels, quantiles, ax_lims=None, fs=20, savefig=None):
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
        if par_labels[i] == r'$\log_{10}(M_{\text{BH}}/$M$_{\odot})$':  # or par_labels[i] == 'PAdisk':
            bins = 400  # 400 #1000
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

        if ax_lims is not None:
            axes[row, col].set_xlim(ax_lims[i][0], ax_lims[i][1])
    plt.tight_layout()
    if savefig is not None:
        plt.savefig(savefig)
    plt.show()


def output_dictionaries(err):
    fid_dict = {'pkl': 'ugc_2698_finaltests_fiducial_10000000_8_0.02_1598991563.9127946_end.pkl',
                'name': 'finaltests/u2698_finaltests_fiducial_' + str(err) + 'sig.png',
                'cornername': 'finaltests/u2698_finaltests_fiducial_corner_' + str(err) + 'sig.png',
                'inpf': 'ugc_2698/ugc_2698_finaltests_fiducial.txt',
                'outpf': 'ugc_2698/ugc_2698_finaltests_fiducial_out.txt',
                'mod': 'fiducial', 'extra_params': None}

    ahe_dict = {'pkl': 'ugc_2698_finaltests_ahe_10000000_8_0.02_1598996196.6371024_end.pkl',
                'name': 'finaltests/u2698_finaltests_ahe_' + str(err) + 'sig.png',
                'cornername': 'finaltests/u2698_finaltests_ahe_corner_' + str(err) + 'sig.png',
                'inpf': 'ugc_2698/ugc_2698_finaltests_ahe.txt',
                'outpf': 'ugc_2698/ugc_2698_finaltests_ahe_out.txt',
                'mod': 'ahe', 'extra_params': None}

    rre_dict = {'pkl': 'ugc_2698_finaltests_rre_10000000_8_0.02_1599004482.6025503_end.pkl',
                'name': 'finaltests/u2698_finaltests_rre_' + str(err) + 'sig.png',
                'cornername': 'finaltests/u2698_finaltests_rre_corner_' + str(err) + 'sig.png',
                'inpf': 'ugc_2698/ugc_2698_finaltests_rre.txt',
                'outpf': 'ugc_2698/ugc_2698_finaltests_rre_out.txt',
                'mod': 'rre', 'extra_params': None}

    dlz_dict = {'pkl': 'ugc_2698_finaltests_dlogz0.001_10000000_8_0.001_1598998076.8872557_end.pkl',
                'name': 'finaltests/u2698_finaltests_dlogz0.001_' + str(err) + 'sig.png',
                'cornername': 'finaltests/u2698_finaltests_dlogz0.001_corner_' + str(err) + 'sig.png',
                'inpf': 'ugc_2698/ugc_2698_finaltests_dlogz0.001.txt',
                'outpf': 'ugc_2698/ugc_2698_finaltests_dlogz0.001_out.txt',
                'mod': 'dlogz0.001', 'extra_params': None}

    nlv_dict = {'pkl': 'ugc_2698_finaltests_nlive1000_10000000_8_0.02_1599014686.3287234_end.pkl',
                'name': 'finaltests/u2698_finaltests_nlive1000_' + str(err) + 'sig.png',
                'cornername': 'finaltests/u2698_finaltests_nlive1000_corner_' + str(err) + 'sig.png',
                'inpf': 'ugc_2698/ugc_2698_finaltests_nlive1000.txt',
                'outpf': 'ugc_2698/ugc_2698_finaltests_nlive1000_out.txt',
                'mod': 'nlive1000', 'extra_params': None}

    d48_dict = {'pkl': 'ugc_2698_finaltests_ds48_10000000_8_0.02_1598991134.1821778_end.pkl',
                'name': 'finaltests/u2698_finaltests_ds48_' + str(err) + 'sig.png',
                'cornername': 'finaltests/u2698_finaltests_ds48_corner_' + str(err) + 'sig.png',
                'inpf': 'ugc_2698/ugc_2698_finaltests_ds48.txt',
                'outpf': 'ugc_2698/ugc_2698_finaltests_ds48_out.txt',
                'mod': 'ds48', 'extra_params': None}

    d51_dict = {'pkl': 'ugc_2698_finaltests_ds510_10000000_8_0.02_1598996125.0661013_end.pkl',
                'name': 'finaltests/u2698_finaltests_ds510_' + str(err) + 'sig.png',
                'cornername': 'finaltests/u2698_finaltests_ds510_corner_' + str(err) + 'sig.png',
                'inpf': 'ugc_2698/ugc_2698_finaltests_ds510.txt',
                'outpf': 'ugc_2698/ugc_2698_finaltests_ds510_out.txt',
                'mod': 'ds510', 'extra_params': None}

    exp_dict = {'pkl': 'ugc_2698_finaltests_exp_10000000_8_0.02_1599221245.840947_end.pkl',  # ~297000 func calls; ~638013 (batch stage)
                'name': 'finaltests/u2698_finaltests_exp_' + str(err) + 'sig.png',
                'cornername': 'finaltests/u2698_finaltests_exp_corner_' + str(err) + 'sig.png',
                'inpf': 'ugc_2698/ugc_2698_finaltests_exp.txt',
                'outpf': 'ugc_2698/ugc_2698_finaltests_exp_out.txt',
                'mod': 'exp', 'extra_params': [['r0', 'pc'], ['sig1', 'km/s']]}

    ful_dict = {'pkl': 'ugc_2698_finaltests_fullpriors_10000000_8_0.02_1599057523.4734955_end.pkl',
                'name': 'finaltests/u2698_finaltests_fullpriors_' + str(err) + 'sig.png',
                'cornername': 'finaltests/u2698_finaltests_fullpriors_corner_' + str(err) + 'sig.png',
                'inpf': 'ugc_2698/ugc_2698_finaltests_fullpriors.txt',
                'outpf': 'ugc_2698/ugc_2698_finaltests_fullpriors_out.txt',
                'mod': 'fullpriors', 'extra_params': None}

    gas_dict = {'pkl': 'ugc_2698_finaltests_gas_10000000_8_0.02_1598991110.5280113_end.pkl',
                'name': 'finaltests/u2698_finaltests_gas_' + str(err) + 'sig.png',
                'cornername': 'finaltests/u2698_finaltests_gas_corner_' + str(err) + 'sig.png',
                'inpf': 'ugc_2698/ugc_2698_finaltests_gas.txt',
                'outpf': 'ugc_2698/ugc_2698_finaltests_gas_out.txt',
                'mod': 'gas', 'extra_params': None}

    kap_dict = {'pkl': 'ugc_2698_finaltests_kappa_10000000_8_0.02_1599130684.4075377_end.pkl',  # ~327116 func calls temp, ~413422 temp2 (prelim cvg)
                'name': 'finaltests/u2698_finaltests_kappa_' + str(err) + 'sig.png',
                'cornername': 'finaltests/u2698_finaltests_kappa_corner_' + str(err) + 'sig.png',
                'inpf': 'ugc_2698/ugc_2698_finaltests_kappa.txt',
                'outpf': 'ugc_2698/ugc_2698_finaltests_kappa_out.txt',
                'mod': 'kappa', 'extra_params': [['kappa', 'unitless']]}

    s01_dict = {'pkl': 'ugc_2698_finaltests_os1_10000000_8_0.02_1598993131.9236333_end.pkl',
                'name': 'finaltests/u2698_finaltests_os1_' + str(err) + 'sig.png',
                'cornername': 'finaltests/u2698_finaltests_os1_corner_' + str(err) + 'sig.png',
                'inpf': 'ugc_2698/ugc_2698_finaltests_os1.txt',
                'outpf': 'ugc_2698/ugc_2698_finaltests_os1_out.txt',
                'mod': 'os1', 'extra_params': None}

    s02_dict = {'pkl': 'ugc_2698_finaltests_os2_10000000_8_0.02_1598993294.4051502_end.pkl',
                'name': 'finaltests/u2698_finaltests_os2_' + str(err) + 'sig.png',
                'cornername': 'finaltests/u2698_finaltests_os2_corner_' + str(err) + 'sig.png',
                'inpf': 'ugc_2698/ugc_2698_finaltests_os2.txt',
                'outpf': 'ugc_2698/ugc_2698_finaltests_os2_out.txt',
                'mod': 'os2', 'extra_params': None}

    s03_dict = {'pkl': 'ugc_2698_finaltests_os3_10000000_8_0.02_1598987576.2494218_end.pkl',
                'name': 'finaltests/u2698_finaltests_os3_' + str(err) + 'sig.png',
                'cornername': 'finaltests/u2698_finaltests_os3_corner_' + str(err) + 'sig.png',
                'inpf': 'ugc_2698/ugc_2698_finaltests_os3.txt',
                'outpf': 'ugc_2698/ugc_2698_finaltests_os3_out.txt',
                'mod': 'os3', 'extra_params': None}

    s06_dict = {'pkl': 'ugc_2698_finaltests_os6_10000000_8_0.02_1599013569.0664136_end.pkl',
                'name': 'finaltests/u2698_finaltests_os6_' + str(err) + 'sig.png',
                'cornername': 'finaltests/u2698_finaltests_os6_corner_' + str(err) + 'sig.png',
                'inpf': 'ugc_2698/ugc_2698_finaltests_os6.txt',
                'outpf': 'ugc_2698/ugc_2698_finaltests_os6_out.txt',
                'mod': 'os6', 'extra_params': None}

    s08_dict = {'pkl': 'ugc_2698_finaltests_os8_10000000_8_0.02_1599021614.7866514_end.pkl',
                'name': 'finaltests/u2698_finaltests_os8_' + str(err) + 'sig.png',
                'cornername': 'finaltests/u2698_finaltests_os8_corner_' + str(err) + 'sig.png',
                'inpf': 'ugc_2698/ugc_2698_finaltests_os8.txt',
                'outpf': 'ugc_2698/ugc_2698_finaltests_os8_out.txt',
                'mod': 'os8', 'extra_params': None}

    s10_dict = {'pkl': 'ugc_2698_finaltests_os10_10000000_8_0.02_1599027485.286418_end.pkl',
                'name': 'finaltests/u2698_finaltests_os10_' + str(err) + 'sig.png',
                'cornername': 'finaltests/u2698_finaltests_os10_corner_' + str(err) + 'sig.png',
                'inpf': 'ugc_2698/ugc_2698_finaltests_os10.txt',
                'outpf': 'ugc_2698/ugc_2698_finaltests_os10_out.txt',
                'mod': 'os10', 'extra_params': None}

    s12_dict = {'pkl': 'ugc_2698_finaltests_os12_10000000_8_0.02_1599039316.859327_end.pkl',
                'name': 'finaltests/u2698_finaltests_os12_' + str(err) + 'sig.png',
                'cornername': 'finaltests/u2698_finaltests_os12_corner_' + str(err) + 'sig.png',
                'inpf': 'ugc_2698/ugc_2698_finaltests_os12.txt',
                'outpf': 'ugc_2698/ugc_2698_finaltests_os12_out.txt',
                'mod': 'os12', 'extra_params': None}

    r03_dict = {'pkl': 'ugc_2698_finaltests_rfit0.3_10000000_8_0.02_1598973877.4783337_tempsave.pkl',  # ~325800 calls; ~649210 temp2 (unchanged from temp); ~958458 temp3
                'name': 'finaltests/u2698_finaltests_rfit0.3_' + str(err) + 'sig_temp3.png',
                'cornername': 'finaltests/u2698_finaltests_rfit0.3_corner_' + str(err) + 'sig_temp3.png',
                'inpf': 'ugc_2698/ugc_2698_finaltests_rfit0.3.txt',
                'outpf': 'ugc_2698/ugc_2698_finaltests_rfit0.3_out_temp3.txt',
                'mod': 'rfit0.3', 'extra_params': None}

    r04_dict = {'pkl': 'ugc_2698_finaltests_rfit0.4_10000000_8_0.02_1598973877.4808373_tempsave.pkl',  # ~293900 calls; ~627888 temp2 (identical to temp); ~918764 temp3
                'name': 'finaltests/u2698_finaltests_rfit0.4_' + str(err) + 'sig_temp3.png',
                'cornername': 'finaltests/u2698_finaltests_rfit0.4_corner_' + str(err) + 'sig_temp3.png',
                'inpf': 'ugc_2698/ugc_2698_finaltests_rfit0.4.txt',
                'outpf': 'ugc_2698/ugc_2698_finaltests_rfit0.4_out_temp3.txt',
                'mod': 'rfit0.4', 'extra_params': None}

    r05_dict = {'pkl': 'ugc_2698_finaltests_rfit0.5_10000000_8_0.02_1599019066.0246398_end.pkl',
                'name': 'finaltests/u2698_finaltests_rfit0.5_' + str(err) + 'sig.png',
                'cornername': 'finaltests/u2698_finaltests_rfit0.5_corner_' + str(err) + 'sig.png',
                'inpf': 'ugc_2698/ugc_2698_finaltests_rfit0.5.txt',
                'outpf': 'ugc_2698/ugc_2698_finaltests_rfit0.5_out.txt',
                'mod': 'rfit0.5', 'extra_params': None}

    r06_dict = {'pkl': 'ugc_2698_finaltests_rfit0.6_10000000_8_0.02_1599003992.1003041_end.pkl',
                'name': 'finaltests/u2698_finaltests_rfit0.6_' + str(err) + 'sig.png',
                'cornername': 'finaltests/u2698_finaltests_rfit0.6_corner_' + str(err) + 'sig.png',
                'inpf': 'ugc_2698/ugc_2698_finaltests_rfit0.6.txt',
                'outpf': 'ugc_2698/ugc_2698_finaltests_rfit0.6_out.txt',
                'mod': 'rfit0.6', 'extra_params': None}

    r08_dict = {'pkl': 'ugc_2698_finaltests_rfit0.8_10000000_8_0.02_1598994597.4476626_end.pkl',
                'name': 'finaltests/u2698_finaltests_rfit0.8_' + str(err) + 'sig.png',
                'cornername': 'finaltests/u2698_finaltests_rfit0.8_corner_' + str(err) + 'sig.png',
                'inpf': 'ugc_2698/ugc_2698_finaltests_rfit0.8.txt',
                'outpf': 'ugc_2698/ugc_2698_finaltests_rfit0.8_out.txt',
                'mod': 'rfit0.8', 'extra_params': None}

    vra_dict = {'pkl': 'ugc_2698_finaltests_vrad_10000000_8_0.02_1599163402.2385688_end.pkl',  # ~233582 calls, temp2 299242 (after prelim cvg), temp3 ~603270
                'name': 'finaltests/u2698_finaltests_vrad_' + str(err) + 'sig.png',
                'cornername': 'finaltests/u2698_finaltests_vrad_corner_' + str(err) + 'sig.png',
                'inpf': 'ugc_2698/ugc_2698_finaltests_vrad.txt',
                'outpf': 'ugc_2698/ugc_2698_finaltests_vrad_out.txt',
                'mod': 'vrad', 'extra_params': [['vrad', 'km/s']]}

    l05_dict = {'pkl': 'ugc_2698_finaltests_lucyn5_10000000_8_0.02_1599618047.3316298_end.pkl',
                'name': 'finaltests/u2698_finaltests_lucyn5_' + str(err) + 'sig.png',
                'cornername': 'finaltests/u2698_finaltests_lucyn5_corner_' + str(err) + 'sig.png',
                'inpf': 'ugc_2698/ugc_2698_finaltests_lucyn5.txt',
                'outpf': 'ugc_2698/ugc_2698_finaltests_lucyn5_out.txt',
                'mod': 'lucyn5', 'extra_params': None}

    l15_dict = {'pkl': 'ugc_2698_finaltests_lucyn15_10000000_8_0.02_1599615320.6710668_end.pkl',
                'name': 'finaltests/u2698_finaltests_lucyn15_' + str(err) + 'sig.png',
                'cornername': 'finaltests/u2698_finaltests_lucyn15_corner_' + str(err) + 'sig.png',
                'inpf': 'ugc_2698/ugc_2698_finaltests_lucyn15.txt',
                'outpf': 'ugc_2698/ugc_2698_finaltests_lucyn15_out.txt',
                'mod': 'lucyn15', 'extra_params': None}

    lvb_dict = {'pkl': 'ugc_2698_finaltests_lucyvb_10000000_8_0.02_1600323488.5804818_end.pkl',
                'name': 'finaltests/u2698_finaltests_lucyvb_' + str(err) + 'sig.png',
                'cornername': 'finaltests/u2698_finaltests_lucyvb_corner_' + str(err) + 'sig.png',
                'inpf': 'ugc_2698/ugc_2698_finaltests_lucyvb.txt',
                'outpf': 'ugc_2698/ugc_2698_finaltests_lucyvb_out.txt',
                'mod': 'lucyvb', 'extra_params': None}

    results_dict = {
                    'fiducial': fid_dict,#
                    'ahe': ahe_dict,#
                    'rre': rre_dict,#
                    'os1': s01_dict,  #
                    'os2': s02_dict,  #
                    'os3': s03_dict,  #
                    'os6': s06_dict,  #
                    'os8': s08_dict,  #
                    'os10': s10_dict,  #
                    'os12': s12_dict,  #
                    'ds48': d48_dict,#
                    'ds510': d51_dict,#
                    'rfit0.3': r03_dict,  #
                    'rfit0.4': r04_dict,  #
                    'rfit0.5': r05_dict,  #
                    'rfit0.6': r06_dict,  #
                    'rfit0.8': r08_dict,  #
                    'gas': gas_dict,#
                    'kappa': kap_dict,#
                    'vrad': vra_dict,
                    'exp': exp_dict,#
                    'lucyn5': l05_dict,#
                    'lucyn15': l15_dict,#
                    'lucyvb': lvb_dict,#
                    'nlive': nlv_dict,#
                    'dlogz': dlz_dict,#
                    'fullpriors': ful_dict,#
                    }

    return results_dict


# DEFINE DIRECTORIES
direc = '/Users/jonathancohn/Documents/dyn_mod/nest_out/'
grp = '/Users/jonathancohn/Documents/dyn_mod/groupmtg/'
sig = 3  # 1 # 3  # show 1sigma errors or 3sigma errors

# CHOOSE DICTIONARY
dict = output_dictionaries(sig)['fiducial']

if 'nobh' in dict['pkl']:
    # labels = np.array(['xloc', 'yloc', 'sig0', 'inc', 'PAdisk', 'vsys', 'ml_ratio', 'f'])
    labels = np.array([r'$x_0$', r'$y_0$', r'$\sigma_0$', r'$\iota$', r'$\Gamma$', r'v$_{\text{sys}}$', r'$M/L$',
                       r'$f_0$'])
    ax_lab = np.array(['pixels', 'pixels', 'km $s^{-1}$', 'deg', 'deg', 'km $s^{-1}$', r'M$_{\odot}$/L$_{\odot}$',
                       'unitless'])
    tablabs = np.array(['reduced chi^2', 'xloc [pix]', 'yloc [pix]', 'sig0 [km/s]', 'inc [deg]',
                        'PAdisk [deg]', 'vsys [km/s]', 'ml_ratio [Msol/Lsol]', 'f [unitless]'])
else:
    # labels = np.array(['mbh', 'xloc', 'yloc', 'sig0', 'inc', 'PAdisk', 'vsys', 'ml_ratio', 'f'])
    # labels = np.array([r'M$_{\text{BH}}$', r'$x_0$', r'$y_0$', r'$\sigma_0$', r'$\iota$', r'$\Gamma$',
    labels = np.array([r'$\log_{10}(M_{\text{BH}}/$M$_{\odot})$', r'$x_0$', r'$y_0$', r'$\sigma_0$', r'$\iota$',
                       r'$\Gamma$', r'v$_{\text{sys}}$', r'$M/L$', r'$f_0$'])
    # ax_lab = np.array([r'$\log_{10}$(M$_{\odot}$)', 'pixels', 'pixels', 'km/s', 'deg', 'deg', 'km/s',
    #                    r'M$_{\odot}$/L$_{\odot}$', 'unitless'])
    ax_lab = np.array([r'$\log_{10}$(M$_{\text{BH}}/$M$_{\odot}$)', 'pixels', 'pixels', 'km $s^{-1}$', 'deg', 'deg',
                       'km $s^{-1}$', r'M$_{\odot}$/L$_{\odot}$', 'unitless'])
    tablabs = np.array(['chi^2' 'reduced chi^2', 'log10(mbh) [Msol]', 'xloc [pix]', 'yloc [pix]', 'sig0 [km/s]',
                        'inc [deg]', 'PAdisk [deg]', 'vsys [km/s]', 'ml_ratio [Msol/Lsol]', 'f [unitless]'])
if dict['extra_params'] is not None:
    for par in dict['extra_params']:
        labels = np.append(labels, par[0])
        ax_lab = np.append(ax_lab, par[1])
        tablabs = np.append(tablabs, par[0] + ' [' + par[1] + ']')

print(labels)

'''  #
# ONLY table_it *AFTER* OUT FILE CREATED
#tl = big_table(output_dictionaries(sig))
#print(tl)
#print(oop)

hd, hl, li, tx = table_it([direc + dict['pkl']], [dict['outpf']], [dict['mod']], tablabs, sig=sig, logged=True,
                          percent_diff=True)
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

# PREPARE QUANTILES!
weights = np.exp(dyn_res['logwt'] - dyn_res['logz'][-1])  # normalized weights

# WRITE OUTFILE!
three_sigs = []
one_sigs = []
with open(dict['inpf'], 'r') as inpff:
    # outpf = dict['inpf'][:-4] + '_out.txt'
    outpf = dict['outpf']
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

# CALCULATE QUANTILES!
for i in range(dyn_res['samples'].shape[1]):  # for each parameter
    quantiles_3 = dyfunc.quantile(dyn_res['samples'][:, i], [0.0015, 0.5, 0.9985], weights=weights)
    quantiles_2 = dyfunc.quantile(dyn_res['samples'][:, i], [0.025, 0.5, 0.975], weights=weights)
    quantiles_1 = dyfunc.quantile(dyn_res['samples'][:, i], [0.16, 0.5, 0.84], weights=weights)
    print(labels[i])
    if i == 0 and 'nobh' not in dict['pkl']:
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

sig1 = [0.16, 0.5, 0.84]
sig2 = [0.025, 0.5, 0.975]
sig3 = [0.0015, 0.5, 0.9985]

sigs = three_sigs  # set default to 3sig
qts = sig3  # set default to 3sig

if sig == 1:
    qts = sig1
    sigs = one_sigs
elif sig == 3:
    qts = sig3
    sigs = three_sigs

logm = True
if logm and 'nobh' not in dict['pkl']:
    dyn_res['samples'][:, 0] = np.log10(dyn_res['samples'][:, 0])
    # labels[0] = 'log mbh'  # r'log$_{10}$mbh'
    # labels[0] = r'log$_{{10}}($M$_{\text{BH}})$'
    # labels[0] = '$\\log_{10}(M_{\\text{BH}})$'

ax_lims = None

if 'fullpriors' in out_name:
    ax_lims = [[6., 12.], [116., 140.], [140., 160.], [0., 200.], [52.4, 89.], [0., 89.], [5000., 8100.], [0.1, 10.],
               [0.1, 2.5]]
elif 'exp' in out_name:
    ax_lims = [[8., 10.], [124., 128.], [148, 152], [0., 100.], [52.4, 89], [5., 35.], [6405, 6505], [0.3, 3.],
               [0.5, 1.5], [0., 100.], [0., 100.]]  # [19.13, 19.23]
elif 'vrad' in out_name:
    ax_lims = [[8., 10.], [124., 128.], [148, 152], [0., 40.], [52.4, 89], [5., 35.], [6405, 6505], [0.3, 3.],
               [0.5, 1.5], [-50., 50.]]
elif 'kappa' in out_name:
    ax_lims = [[8., 10.], [124., 128.], [148, 152], [0., 40.], [52.4, 89], [5., 35.], [6405, 6505], [0.3, 3.],
               [0.5, 1.5], [-1., 1.]]
else:
    ax_lims = [[9.15, 9.65], [124., 128.], [148, 152], [0., 40.], [52.4, 89], [5., 35.], [6405, 6505], [0.3, 3.],
               [0.5, 1.5]]  # 8., 10.

import matplotlib as mpl
# mpl.rcParams['font.size'] = 20
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']  # for \text command
mpl.rcParams['xtick.labelsize'] = 20
mpl.rcParams['ytick.labelsize'] = 20

# ELSE USE THIS
my_own_thing(dyn_res['samples'], labels, ax_lab, one_sigs, ax_lims=ax_lims, savefig=grp + dict['name'])
print(oop)

# plot initial run (res1; left)

# TO EDIT SOURCE CODE: open /Users/jonathancohn/anaconda3/envs/three/lib/python3.6/site-packages/dynesty/plotting.py

# MAKE CORNER PLOT
ndim = len(labels)
factor = 2.0  # size of side of one panel
lbdim = 0.5 * factor  # size of left/bottom margin
trdim = 0.2 * factor  # size of top/right margin
whspace = 0.05  # size of width/height margin
plotdim = factor * ndim + factor * (ndim - 1.) * whspace  # plot size
dim = lbdim + plotdim + trdim  # total size
fig, axes = plt.subplots(ndim, ndim, figsize=(1.7*dim, dim))
fg, ax = dyplot.cornerplot(dyn_res, color='blue', show_titles=True, title_kwargs={'fontsize': 30}, max_n_ticks=3,
                           quantiles=qts, labels=labels, label_kwargs={'fontsize': 30}, fig=(fig, axes))
# plt.savefig(grp + dict['cornername'])
plt.savefig(grp + 'finaltests/u2698_finaltests_fiducial_corner_3sig_font30tick20.png')
#plt.show()

'''  #
# ONLY table_it *AFTER* OUT FILE CREATED
hd, hl, li, tx = table_it([direc + dict['pkl']], [dict['outpf']], [dict['mod']], tablabs, sig=sig, logged=True,
                          percent_diff=True)
print(hd)
print(hl)
print(li)
print(tx)
# '''  #
