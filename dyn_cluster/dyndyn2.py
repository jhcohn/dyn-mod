#!/usr/bin/env python
from __future__ import division, print_function
import os, sys, time
import argparse

import pickle
import numpy as np
from numpy import linalg

import dynesty
from dynesty import DynamicNestedSampler
import dyn_model as dm

# PARALLEL
from dynesty.dynamicsampler import stopping_function, weight_function, _kld_error
from dynesty.utils import *
from multiprocessing import Pool

# see https://github.com/bd-j/prospector/blob/master/prospect/fitting/nested.py
#   https://dynesty.readthedocs.io/en/latest/api.html
# https://github.com/bd-j/prospector/blob/master/prospect/fitting/fitting.py
# https://github.com/bd-j/prospector/blob/master/scripts/prospector_dynesty.py

# seed the random number generator
np.random.seed(5647)


def prior(theta):
    """
    Transform prior from uniform cube ranging from 0:1 for each parameter, to each parameter's prior

    :param cube: uniform 0:1 for each parameter; adjust so each parameter ranges between prior boundaries

    :return: prior cube
    """
    '''  #
    # WIDE PRIORS
    cube[0] = 10**(cube[0]*13. - 1.)  # mbh: log-uniform prior from 10^-1 to 10^12
    cube[1] *= 24. + 116.  # xloc: uniform prior 116:140
    cube[2] *= 20. + 140.  # yloc: uniform prior 140:160
    cube[3] *= 200.  # sig0: uniform prior 0:200
    cube[4] *= 40.15 + 49.84  # inc: uniform prior 49.84:89.99 (lower bound=49.83663935008522 set by q in MGE)
    cube[5] *= 90.  # PAdisk: uniform prior 0:90
    cube[6] *= 5000. + 3100.  # vsys: uniform prior 5000:8100
    cube[7] *= 9.9 + 0.1  # mlratio: uniform prior 0.1:10
    cube[8] *= 2.4 + 0.1  # f: uniform prior 0.1:2.5
    '''  #
    '''  #
    # NARROWER PRIORS [even narrower, based on short MCMC]
    cube[0] = 10**(cube[0] * 3. + 8.)  # mbh: log-uniform prior 10^8:10^11 [**(cube[0] + 9.): 10^9:10^10]
    cube[1] *= 20. + 118.  # xloc: uniform prior 118:138 [*= 3.5 + 126.: 126:129.5]
    cube[2] *= 15. + 143.  # yloc: uniform prior 143:158 [*= 1.75 + 150.25: 150.25:152]
    cube[3] *= 200.  # sig0: uniform prior 0:200 [*= 15.: 0:15]
    cube[4] *= 40.15 + 49.84  # inc: uniform prior 49.84:89.99 (low pri=49.83663935008522 from MGE q) [*=5 + 66.: 66:71]
    cube[5] *= 90.  # PAdisk: uniform prior 0:90 [*= 4. + 17.: 17:21]
    cube[6] *= 1000. + 6100.  # vsys: uniform prior 6100:7100 [*= 50 + 6440.: 6440:6490]
    cube[7] *= 9.9 + 0.1  # mlratio: uniform prior 0.1:10 [*= 0.45 + 1.45: 1.45:1.90]
    cube[8] *= 2.4 + 0.1  # f: uniform prior 0.1:2.5 [*= 0.85 + 0.85: 0.85:1.7]
    # '''
    cube = np.array(theta)

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

    return cube


def lnprob(cube):
    """
    Not computing full posterior probability, so just use chi squared (P propto exp(-chi^2/2) --> ln(P) ~ -chi^2 / 2

    :param cube: parameter cube (see prior)

    :return: -0.5 * chi^2 (ln likelihood)
    """

    chi2 = dm.model_grid(
        # FREE PARAMETERS
        x_loc=cube[1], y_loc=cube[2], mbh=cube[0], inc=np.deg2rad(cube[4]), vsys=cube[6], theta=cube[5],
        ml_ratio=cube[7], sig_params=[cube[3], params['r0'], params['mu'], params['sig1']], f_w=cube[8],

        # FIXED PARAMETERS
        resolution=params['resolution'],
        s=params['s'],
        dist=params['dist'],
        input_data=input_data,
        lucy_out=lucy_out,
        out_name=None,
        beam=beam,
        rfit=params['rfit'],
        enclosed_mass=params['mass'],
        sig_type=params['s_type'],
        zrange=[params['zi'], params['zf']],
        menc_type=params['mtype'],
        ds=params['ds'],
        noise=noise,
        chi2=True,
        reduced=False,
        freq_ax=freq_ax,
        q_ell=params['q_ell'],
        theta_ell=np.deg2rad(params['theta_ell']),
        xell=params['xell'],
        yell=params['yell'],
        fstep=fstep, f_0=f_0,
        bl=params['bl'],
        xyrange=[params['xi'], params['xf'], params['yi'], params['yf']])

    # return pri + (-0.5 * chi2)
    return -0.5 * chi2


# MORE POOL STUFF
def halt(message):
    """Exit, closing pool safely."""
    print(message)
    try:
        pool.close()
    except:
        pass
    sys.exit(0)


# pool = None
# nprocs = 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument('--p')  # parfile
    args = vars(parser.parse_args())
    # PARAMETER FILE
    parfile = args['p']  # '/scratch/user/joncohn/dyn_cluster/ugc_2698/ugc_2698_clusterparams.txt'  # BUCKET
    params, priors, qobs = dm.par_dicts(parfile, q=True)  # get dicts of params and file names from parameter file
    print(priors['inc'])
    qint_pri = np.amax(np.rad2deg(np.arccos(np.sqrt((400*qobs**2 - 1.)/399.))))
    priors['inc'][0] = np.amax([priors['inc'][0], np.rad2deg(np.arccos(np.amin(qobs)))])
    priors['inc'][0] = np.amax([priors['inc'][0], qint_pri])
    print(priors['inc'])
    # np.sqrt(qobs**2 - np.cos(inc)**2)/np.sin(inc) > 0.05 --> sin(inc) < sqrt(qobs**2 - np.cos(inc)**2)/.05
    # --> sin^2(inc) < qobs^2/0.05 - cos^2(inc)/0.05 --> sin^2(inc) + cos^2(inc)/0.05^2 < qobs^2/0.05^2
    # NOTE: sin^2(x) + C*cos^2(x) = 0.5*(C*cos(2x) + C + 1 - cos(2x)) = 0.5(cos(2x)(C-1) + (C+1))
    # --> 0.5(19*cos(2inc) + 21) < 20*qobs^2 --> 399cos^2(inc)+1 < 400*q^2 --> inc < arccos(sqrt[(400q^2 - 1)/399])

    ndim = len(priors)  # number of dimensions = number of free parameters

    # name of the output files
    direc = '/scratch/user/joncohn/dyn_cluster/nest_out/'

    mod_ins = dm.model_prep(data=params['data'], ds=params['ds'], lucy_out=params['lucy'], lucy_b=params['lucy_b'], lucy_mask=params['lucy_mask'], lucy_in=params['lucy_in'], lucy_o=params['lucy_o'], lucy_it=params['lucy_it'], data_mask=params['mask'], grid_size=params['gsize'], res=params['resolution'], x_std=params['x_fwhm'], y_std=params['y_fwhm'], pa=params['PAbeam'], zrange=[params['zi'], params['zf']], xyerr=[params['xerr0'], params['xerr1'], params['yerr0'], params['yerr1']])
    lucy_mask, lucy_out, beam, fluxes, freq_ax, f_0, fstep, input_data, noise = mod_ins
    thresh = 0.02  # 0.01
    maxc = 100000
    # n_eff = 100000
    out_name = direc + 'dyndyn2_test_n' + str(params['nprocs']) + '_' + str(time.time()) + '_tempsave' +  '.pkl'

    if params['nprocs'] > 1:    
        pool = Pool(processes=params['nprocs'])
        psampler = dynesty.DynamicNestedSampler(lnprob, prior, ndim, pool=pool, use_pool={'update_bound': False}, queue_size=params['nprocs'], dlogz=thresh)  # queue_size=nprocs, first_update={'min_ncall': 5000, 'min_eff': 50.}
        ncall = psampler.ncall
        niter = psampler.it - 1
        for results in psampler.sample_initial(maxcall=maxc):
            ncall += results[9]
            niter += 1
            delta_logz = results[-1]
            print('dlogz ' + str(delta_logz), 'thresh ' + str(thresh), 'nc ' + str(ncall), 'niter ' + str(niter))
            pass
        while True:
            stop = stopping_function(psampler.results)  # evaluate stop
            if not stop:
                logl_bounds = weight_function(psampler.results)  # derive bounds
                print('look1')
                for results in psampler.sample_batch(logl_bounds=logl_bounds, maxcall=maxc):
                    ncall += results[4]  # worst, ustar, vstar, loglstar, nc...
                    niter += 1
                    print('nc ' + str(ncall), 'niter ' + str(niter))
                    pass
                print('look2')
                psampler.combine_runs()  # add new samples to previous results
                res1 = psampler.results
                tp = time.time()
                with open(out_name, 'wb+') as newfile:  # 'wb' bc binary
                    pickle.dump(res1, newfile, pickle.HIGHEST_PROTOCOL)
                    print('results pickle dumped in ' + str(time.time()-tp))
            else:
                break
    else:
        psampler = dynesty.DynamicNestedSampler(lnprob, prior, ndim, dlogz=thresh)  # queue_size=nprocs, first_update={'min_ncall': 5000, 'min_eff': 50.}
        ncall = psampler.ncall
        niter = psampler.it - 1
        for results in psampler.sample_initial(maxcall=maxc):
            ncall += results[9]
            niter += 1
            delta_logz = results[-1]
            print('dlogz ' + str(delta_logz), 'thresh ' + str(thresh), 'nc ' + str(ncall), 'niter ' + str(niter))
            pass
        while True:
            stop = stopping_function(psampler.results)  # evaluate stop
            if not stop:
                logl_bounds = weight_function(psampler.results)  # derive bounds
                print('look1')
                for results in psampler.sample_batch(logl_bounds=logl_bounds, maxcall=maxc):
                    ncall += results[4]  # worst, ustar, vstar, loglstar, nc...
                    niter += 1
                    print('nc ' + str(ncall), 'niter ' + str(niter))
                    pass
                print('look2')
                psampler.combine_runs()  # add new samples to previous results
                res1 = psampler.results
                tp = time.time()
                with open(out_name, 'wb+') as newfile:  # 'wb' bc binary
                    pickle.dump(res1, newfile, pickle.HIGHEST_PROTOCOL)
                    print('results pickle dumped in ' + str(time.time()-tp))
            else:
                break
    # psampler.run_nested()
    # dynamic nested sampling loop. After initial baseline run using a constant number of live points, dynamically
    # allocates additional (nested) samples to optimize apecified weight function until stopping criterion reached.
    # dlogz=0.01; NOTE: default dlogz=1e-3*(nlive-1)+0.01 (bc default add_live=True; otherwise default dlogz=0.01)
    # maybe try n_effective=1e7 in run_nested() (put ceiling on number of posterior samples)
    # could also try setting maxiter_init=? (put ceiling on max iterations for initial baseline nested sampling run)
    res = psampler.results
    import pickle
    out_name = direc + 'dyndyn2_test_n' + str(params['nprocs']) + '_' + str(time.time()) + '_end.pkl'
    with open(out_name, 'wb+') as newfile:  # 'wb' because binary format
        pickle.dump(res, newfile, pickle.HIGHEST_PROTOCOL)  # res.samples
        print('results pickle dumped!')
    print(out_name)
    # run Dynamic Nested Sampler (Dynamic -> better for posteriors)
    # dsampler = DynamicNestedSampler(lnprob, prior, ndim, bound='multi')
    # dsampler.run_nested()
    # res = dsampler.results

    # sampler = dynesty.NestedSampler(lnprob, prior, ndim, nlive=1500)
    # sampler.run_nested()
    # res = sampler.results  # grab our results

    print('Keys:', res.keys(), '\n')  # print accessible keys
    # res.summary()  # print a summary (broken: needs nlive to be defined?)

    from dynesty import utils as dyfunc
    weights = np.exp(res['logwt'] - res['logz'][-1])  # normalized weights
    for i in range(dyn_res['samples'].shape[1]):  # for each parameter
        quantiles_3 = dyfunc.quantile(res['samples'][:, i], [0.00135, 0.5, 0.99865], weights=weights)
        quantiles_2 = dyfunc.quantile(res['samples'][:, i], [0.02275, 0.5, 0.977], weights=weights)
        quantiles_1 = dyfunc.quantile(res['samples'][:, i], [0.16, 0.5, 0.84], weights=weights)
        print(quantiles_3, r'$3\sigma$')
        print(quantiles_2, r'$2\sigma$')
        print(quantiles_1, r'$1\sigma$')


    halt('Finished')

