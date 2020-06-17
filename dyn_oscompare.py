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


direc = '/Users/jonathancohn/Documents/dyn_mod/nest_out/'
grp = '/Users/jonathancohn/Documents/dyn_mod/groupmtg/'


os_to_compare = ['u2698_os2_d91_baseline_rhe_orig_nog_10000000_8_0.02_1590693577.7494018_end.pkl',
                 'u2698_os3_d91_baseline_rhe_orig_nog_10000000_8_0.02_1590692584.260687_end.pkl',
                 'u2698_os4_d91_baseline_rhe_orig_nog_10000000_8_0.02_1590692888.5767071_end.pkl',
                 'u2698_os6_d91_baseline_rhe_orig_nog_10000000_8_0.02_1590699379.3008854_end.pkl',
                 'u2698_os8_d91_baseline_rhe_orig_nog_10000000_8_0.02_1590701499.9093819_end.pkl',
                 'u2698_os10_d91_baseline_rhe_orig_nog_10000000_8_0.02_1590705487.6493137_end.pkl',
                 'u2698_os12_d91_baseline_rhe_orig_nog_10000000_8_0.02_1590715106.442782_end.pkl',
                 'u2698_os14_d91_baseline_rhe_orig_nog_10000000_8_0.02_1590729076.410538_end.pkl',
                 'u2698_os16_d91_baseline_rhe_orig_nog_10000000_8_0.02_1590732620.6191862_end.pkl']

os_parfiles = ['ugc_2698/ugc_2698_os2_d91_baseline_rhe_orig_nog_out.txt',
               'ugc_2698/ugc_2698_os3_d91_baseline_rhe_orig_nog_out.txt',
               'ugc_2698/ugc_2698_os4_d91_baseline_rhe_orig_nog_out.txt',
               'ugc_2698/ugc_2698_os6_d91_baseline_rhe_orig_nog_out.txt',
               'ugc_2698/ugc_2698_os8_d91_baseline_rhe_orig_nog_out.txt',
               'ugc_2698/ugc_2698_os10_d91_baseline_rhe_orig_nog_out.txt',
               'ugc_2698/ugc_2698_os12_d91_baseline_rhe_orig_nog_out.txt',
               'ugc_2698/ugc_2698_os14_d91_baseline_rhe_orig_nog_out.txt',
               'ugc_2698/ugc_2698_os16_d91_baseline_rhe_orig_nog_out.txt']

chisq_array = np.zeros(shape=(len(os_to_compare)))
mbh_array = np.zeros(shape=(len(os_to_compare), 3))
os_order = []

for osf in range(len(os_to_compare)):
    with open(direc + os_to_compare[osf], 'rb') as pk:
        u = pickle._Unpickler(pk)
        u.encoding = 'latin1'
        dyn_res = u.load()  #
        # dyn_res = pickle.load(pk)  #

    weights = np.exp(dyn_res['logwt'] - dyn_res['logz'][-1])  # normalized weights

    # 0th index is MBH (dyn_res['samples'][:, i] is BH mass when i = 0)
    quantiles_3 = dyfunc.quantile(dyn_res['samples'][:, 0], [0.0015, 0.5, 0.9985], weights=weights)
    quantiles_2 = dyfunc.quantile(dyn_res['samples'][:, 0], [0.025, 0.5, 0.975], weights=weights)
    quantiles_1 = dyfunc.quantile(dyn_res['samples'][:, 0], [0.16, 0.5, 0.84], weights=weights)
    print(np.log10(quantiles_3), quantiles_3)

    mbh_array[osf, 1] = quantiles_3[1]  # quantiles_1  # quantiles_2
    mbh_array[osf, 0] = quantiles_3[1] - quantiles_3[0]
    mbh_array[osf, 2] = quantiles_3[2] - quantiles_3[1]

    params, priors, nfree, qobs = dm.par_dicts(os_parfiles[osf], q=True)  # get params and file names from output parfile
    mod_ins = dm.model_prep(data=params['data'], ds=params['ds'], lucy_out=params['lucy'], lucy_b=params['lucy_b'],
                            lucy_mask=params['lucy_mask'], lucy_in=params['lucy_in'], lucy_it=params['lucy_it'],
                            data_mask=params['mask'], grid_size=params['gsize'], res=params['resolution'],
                            x_std=params['x_fwhm'], y_std=params['y_fwhm'], zrange=[params['zi'], params['zf']],
                            xyerr=[params['xerr0'], params['xerr1'], params['yerr0'], params['yerr1']],
                            pa=params['PAbeam'])
    lucy_mask, lucy_out, beam, fluxes, freq_ax, f_0, fstep, input_data, noise, co_rad, co_sb = mod_ins

    mg = dm.ModelGrid(x_loc=params['xloc'], y_loc=params['yloc'], mbh=params['mbh'], ml_ratio=params['ml_ratio'],
        inc=np.deg2rad(params['inc']), vsys=params['vsys'], theta=np.deg2rad(params['PAdisk']), vrad=None,
        kappa=None, omega=None, f_w=params['f'], os=params['os'], enclosed_mass=params['mass'],
        sig_params=[params['sig0'], params['r0'], params['mu'], params['sig1']], resolution=params['resolution'],
        lucy_out=lucy_out, out_name=None, beam=beam, rfit=params['rfit'], zrange=[params['zi'], params['zf']],
        dist=params['dist'], input_data=input_data, sig_type=params['s_type'], menc_type=params['mtype'],
        theta_ell=np.deg2rad(params['theta_ell']), xell=params['xell'],yell=params['yell'], q_ell=params['q_ell'],
        ds=params['ds'], reduced=False, f_0=f_0, freq_ax=freq_ax, noise=noise, bl=params['bl'], fstep=fstep,
        xyrange=[params['xi'], params['xf'], params['yi'], params['yf']], n_params=nfree)

    mg.grids()
    mg.convolution()
    chisq_array[osf] = mg.chi2()

    os_order.append(params['os'])


plt.plot(os_order, chisq_array, 'ko')
plt.ylabel(r'$\chi^2$')
plt.xlabel(r'Oversampling factor')
plt.show()

plt.errorbar(os_order, mbh_array[:, 1], yerr=[mbh_array[:, 0], mbh_array[:, 2]], fmt='ko')
plt.yscale("log")
plt.ylabel(r'Black Hole Mass [M$_{\odot}$]')
plt.xlabel(r'Oversampling factor')
plt.show()
