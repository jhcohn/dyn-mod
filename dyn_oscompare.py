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


os_to_compare = ['ugc_2698_finaltests_os1_10000000_8_0.02_1598993131.9236333_end.pkl',
                 'ugc_2698_finaltests_os2_10000000_8_0.02_1598993294.4051502_end.pkl',
                 'ugc_2698_finaltests_os3_10000000_8_0.02_1598987576.2494218_end.pkl',
                 'ugc_2698_finaltests_fiducial_10000000_8_0.02_1598991563.9127946_end.pkl',
                 'ugc_2698_finaltests_os6_10000000_8_0.02_1599013569.0664136_end.pkl',
                 'ugc_2698_finaltests_os8_10000000_8_0.02_1599021614.7866514_end.pkl',
                 'ugc_2698_finaltests_os10_10000000_8_0.02_1599027485.286418_end.pkl',
                 'ugc_2698_finaltests_os12_10000000_8_0.02_1599039316.859327_end.pkl']

os_parfiles = ['ugc_2698/ugc_2698_finaltests_os1_out.txt', 'ugc_2698/ugc_2698_finaltests_os2_out.txt',
               'ugc_2698/ugc_2698_finaltests_os3_out.txt', 'ugc_2698/ugc_2698_finaltests_fiducial_out.txt',
               'ugc_2698/ugc_2698_finaltests_os6_out.txt', 'ugc_2698/ugc_2698_finaltests_os8_out.txt',
               'ugc_2698/ugc_2698_finaltests_os10_out.txt', 'ugc_2698/ugc_2698_finaltests_os12_out.txt']

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
        ds=params['ds'], ds2=params['ds'], reduced=False, f_0=f_0, freq_ax=freq_ax, noise=noise, bl=params['bl'],
        fstep=fstep, xyrange=[params['xi'], params['xf'], params['yi'], params['yf']], n_params=nfree)

    mg.grids()
    mg.convolution()
    chisq_array[osf] = mg.chi2()

    os_order.append(params['os'])


plt.figure(figsize=(8,6))
plt.errorbar(os_order, mbh_array[:, 1]/1e9, yerr=[mbh_array[:, 0]/1e9, mbh_array[:, 2]/1e9], fmt='ko')
plt.ylim(2, 3)
#plt.yscale("log")
plt.ylabel(r'Black Hole Mass [$\frac{M_{\odot}}{1e9}$]')
plt.xlabel(r'Oversampling factor')
plt.show()

plt.figure(figsize=(8,6))
plt.plot(os_order, chisq_array, 'ko')
plt.ylabel(r'$\chi^2$')
plt.xlabel(r'Oversampling factor')
plt.show()

'''  # 
os_to_compare = [#'u2698_exv_os1_baseline_rhe_orig_nog_10000000_8_0.02_1594865056.9097123_end.pkl',
                 'u2698_exv_os2_baseline_rhe_orig_nog_10000000_8_0.02_1593211158.390702_end.pkl',
                 'u2698_exv_os3_baseline_rhe_orig_nog_10000000_8_0.02_1593216852.5147808_end.pkl',
                 'u2698_exv_os4_baseline_rhe_orig_nog_10000000_8_0.02_1593211776.966962_end.pkl',
                 'u2698_exv_os6_baseline_rhe_orig_nog_10000000_8_0.02_1593218525.8919191_end.pkl',
                 'u2698_exv_os8_baseline_rhe_orig_nog_10000000_8_0.02_1593227152.5059612_end.pkl',
                 'u2698_exv_os10_baseline_rhe_orig_nog_10000000_8_0.02_1593239740.6758_end.pkl',
                 'u2698_exv_os12_baseline_rhe_orig_nog_10000000_8_0.02_1593265549.3681285_end.pkl',
                 'u2698_exv_os14_baseline_rhe_orig_nog_10000000_8_0.02_1593261183.7849855_end.pkl',
                 'u2698_exv_os16_baseline_rhe_orig_nog_10000000_8_0.02_1593336953.3834007_end.pkl']

os_parfiles = [#'ugc_2698/ugc_2698_exv_os1_baseline_rhe_orig_nog_out.txt',
               'ugc_2698/ugc_2698_exv_os2_baseline_rhe_orig_nog_out.txt',
               'ugc_2698/ugc_2698_exv_os3_baseline_rhe_orig_nog_out.txt',
               'ugc_2698/ugc_2698_exv_os4_baseline_rhe_orig_nog_out.txt',
               'ugc_2698/ugc_2698_exv_os6_baseline_rhe_orig_nog_out.txt',
               'ugc_2698/ugc_2698_exv_os8_baseline_rhe_orig_nog_out.txt',
               'ugc_2698/ugc_2698_exv_os10_baseline_rhe_orig_nog_out.txt',
               'ugc_2698/ugc_2698_exv_os12_baseline_rhe_orig_nog_out.txt',
               'ugc_2698/ugc_2698_exv_os14_baseline_rhe_orig_nog_out.txt',
               'ugc_2698/ugc_2698_exv_os16_baseline_rhe_orig_nog_out.txt']
# '''  #