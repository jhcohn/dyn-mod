#!/usr/bin/env python
from __future__ import division, print_function
import os, sys, time
import argparse

import pickle
import numpy as np
from numpy import linalg

import dynesty
from dynesty import DynamicNestedSampler
import dynamical_model as dm

# PARALLEL
from dynesty.dynamicsampler import stopping_function, weight_function, _kld_error
from dynesty.utils import *
from multiprocessing import Pool
from dynesty import utils as dyfunc

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
    cube = np.array(theta)

    # uniform priors (or, in case of mbh, log-uniform prior) from priors['param'][0] to priors['param'][1]
    cube[0] = 10 ** (cube[0] * (priors['mbh'][1] - priors['mbh'][0]) + priors['mbh'][0])  # mbh: log-uniform prior
    cube[1] = cube[1] * (priors['xloc'][1] - priors['xloc'][0]) + priors['xloc'][0]  # xloc: uniform prior
    cube[2] = cube[2] * (priors['yloc'][1] - priors['yloc'][0]) + priors['yloc'][0]  # yloc: uniform prior
    cube[3] = cube[3] * (priors['sig0'][1] - priors['sig0'][0]) + priors['sig0'][0]  # sig0: uniform prior
    cube[4] = cube[4] * (priors['inc'][1] - priors['inc'][0]) + priors['inc'][0]  # inc: uniform prior
    cube[5] = cube[5] * (priors['PAdisk'][1] - priors['PAdisk'][0]) + priors['PAdisk'][0]  # PAdisk: uniform prior
    cube[6] = cube[6] * (priors['vsys'][1] - priors['vsys'][0]) + priors['vsys'][0]  # vsys: uniform prior
    cube[7] = cube[7] * (priors['ml_ratio'][1] - priors['ml_ratio'][0]) + priors['ml_ratio'][0]  # ml_ratio: uniform pri
    cube[8] = cube[8] * (priors['f'][1] - priors['f'][0]) + priors['f'][0]  # f: uniform prior

    # OPTIONAL FREE PARAMETERS
    if params['s_type'] == 'flat':
        if params['vtype'] == 'vrad':
            cube[9] = cube[9] * (priors['vrad'][1] - priors['vrad'][0]) + priors['vrad'][0]  # vrad: uniform prior
        elif params['vtype'] == 'kappa':
            cube[9] = cube[9] * (priors['kappa'][1] - priors['kappa'][0]) + priors['kappa'][0]  # kappa: uniform prior
        elif params['vtype'] == 'omega':
            cube[9] = cube[9] * (priors['kappa'][1] - priors['kappa'][0]) + priors['kappa'][0]  # kappa: uniform prior
            cube[10] = cube[10] * (priors['omega'][1] - priors['omega'][0]) + priors['omega'][0]  # omega: uniform prior
    elif params['s_type'] == 'exp':
        cube[9] = cube[9] * (priors['sig1'][1] - priors['sig1'][0]) + priors['sig1'][0]  # sig1: uniform prior
        cube[10] = cube[10] * (priors['r0'][1] - priors['r0'][0]) + priors['r0'][0]  # r0: uniform prior
        if params['vtype'] == 'vrad':
            cube[11] = cube[11] * (priors['vrad'][1] - priors['vrad'][0]) + priors['vrad'][0]  # vrad: uniform prior
        elif params['vtype'] == 'kappa':
            cube[11] = cube[11] * (priors['kappa'][1] - priors['kappa'][0]) + priors['kappa'][0]  # kappa: uniform prior
        elif params['vtype'] == 'omega':
            cube[11] = cube[11] * (priors['kappa'][1] - priors['kappa'][0]) + priors['kappa'][0]  # kappa: uniform prior
            cube[12] = cube[12] * (priors['omega'][1] - priors['omega'][0]) + priors['omega'][0]  # omega: uniform prior
    elif params['s_type'] == 'gauss':
        cube[9] = cube[9] * (priors['sig1'][1] - priors['sig1'][0]) + priors['sig1'][0]  # sig1: uniform prior
        cube[10] = cube[10] * (priors['r0'][1] - priors['r0'][0]) + priors['r0'][0]  # r0: uniform prior
        cube[11] = cube[11] * (priors['mu'][1] - priors['mu'][0]) + priors['mu'][0]  # mu: uniform prior
        if params['vtype'] == 'vrad':
            cube[12] = cube[12] * (priors['vrad'][1] - priors['vrad'][0]) + priors['vrad'][0]  # vrad: uniform prior
        elif params['vtype'] == 'kappa':
            cube[12] = cube[12] * (priors['kappa'][1] - priors['kappa'][0]) + priors['kappa'][0]  # kappa: uniform prior
        elif params['vtype'] == 'omega':
            cube[12] = cube[12] * (priors['kappa'][1] - priors['kappa'][0]) + priors['kappa'][0]  # kappa: uniform prior
            cube[13] = cube[13] * (priors['omega'][1] - priors['omega'][0]) + priors['omega'][0]  # omega: uniform prior

    return cube


def lnprob(cube):
    """
    Not computing full posterior probability, so just use chi squared (P propto exp(-chi^2/2) --> ln(P) ~ -chi^2 / 2

    :param cube: parameter cube (see prior)

    :return: -0.5 * chi^2 (ln likelihood)
    """

    # SET UP OPTIONAL PARAMETERS
    vrad, kappa, omega = None, None, None  # optional radial velocity parameters
    r0, mu, sig1 = params['r0'], params['mu'], params['sig1']  # optional turbulent velocity dispersion parameters

    # ASSIGN PARAMETERS TO BE FREE IF WE WANT THEM TO BE FREE
    if params['s_type'] == 'flat':
        if params['vtype'] == 'vrad':  # if vrad
            vrad = cube[9]
        elif params['vtype'] == 'kappa':
            kappa = cube[9]
        elif params['vtype'] == 'omega':
            kappa = cube[9]
            omega = cube[10]
    elif params['s_type'] == 'exp':  # if vrad & exp
        sig1 = cube[9]
        r0 = cube[10]
        if params['vtype'] == 'vrad':  # if vrad
            vrad = cube[11]
        elif params['vtype'] == 'kappa':
            kappa = cube[11]
        elif params['vtype'] == 'omega':
            kappa = cube[11]
            omega = cube[12]
    elif params['s_type'] == 'gauss':  # if vrad & gauss
        sig1 = cube[9]
        r0 = cube[10]
        mu = cube[11]
        if params['vtype'] == 'vrad':  # if vrad
            vrad = cube[12]
        elif params['vtype'] == 'kappa':  # if kappa
            kappa = cube[12]
        elif params['vtype'] == 'omega':  # if omega
            kappa = cube[12]
            omega = cube[13]

    # INSTANTIATE THE MODEL
    mg = dm.ModelGrid(
        # FREE PARAMETERS
        x_loc=cube[1], y_loc=cube[2], mbh=cube[0], inc=np.deg2rad(cube[4]), vsys=cube[6], theta=np.deg2rad(cube[5]),
        ml_ratio=cube[7], sig_params=[cube[3], r0, mu, sig1], f_w=cube[8], vrad=vrad, kappa=kappa, omega=omega,
        # FIXED PARAMETERS
        n_params=nfree,
        vtype=params['vtype'],
        sig_type=params['s_type'],
        resolution=params['resolution'],
        os=params['os'],
        ds=params['ds'],
        dist=params['dist'],
        input_data=input_data,
        lucy_out=lucy_out,
        beam=beam,
        noise=noise,
        menc_type=params['mtype'],
        enclosed_mass=params['mass'],
        bl=params['bl'],
        out_name=None,
        rfit=params['rfit'],
        q_ell=params['q_ell'],
        theta_ell=np.deg2rad(params['theta_ell']),
        xell=params['xell'],
        yell=params['yell'],
        reduced=False,
        freq_ax=freq_ax,
        fstep=fstep,
        f_0=f_0,
        zrange=[params['zi'], params['zf']],
        xyrange=[params['xi'], params['xf'], params['yi'], params['yf']],
        incl_gas=params['incl_gas']=='True',
        co_rad=co_ell_rad,
        co_sb=co_ell_sb)
    mg.grids()  # CREATE THE MODEL GRID
    mg.convolution()  # CONVOLVE THE MODEL CUBE
    chi2 = mg.chi2()  # CALCULATE CHI2

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)  # introduce argparser for command line args
    parser.add_argument('--p')  # parfile is the only arg used here (with --p on the command line)
    args = vars(parser.parse_args())  # this makes the parfile arg = args['p']

    t00 = time.time()  # start time

    # PARAMETER FILE
    parfile = args['p']
    params, priors, nfree, qobs = dm.par_dicts(parfile, q=True)  # get dicts of params and file names from param file

    # make sure inc prior does not conflict with q
    print(priors['inc'])
    qint_pri = np.amax(np.rad2deg(np.arccos(np.sqrt((400*qobs**2 - 1.)/399.))))
    priors['inc'][0] = np.amax([priors['inc'][0], np.rad2deg(np.arccos(np.amin(qobs)))])
    priors['inc'][0] = np.amax([priors['inc'][0], qint_pri])
    print(priors['inc'])
    # q and inclination math
    # np.sqrt(qobs**2 - np.cos(inc)**2)/np.sin(inc) > 0.05 --> sin(inc) < sqrt(qobs**2 - np.cos(inc)**2)/.05
    # --> sin^2(inc) < qobs^2/0.05 - cos^2(inc)/0.05 --> sin^2(inc) + cos^2(inc)/0.05^2 < qobs^2/0.05^2
    # NOTE: sin^2(x) + C*cos^2(x) = 0.5*(C*cos(2x) + C + 1 - cos(2x)) = 0.5(cos(2x)(C-1) + (C+1))
    # --> 0.5(19*cos(2inc) + 21) < 20*qobs^2 --> 399cos^2(inc)+1 < 400*q^2 --> inc < arccos(sqrt[(400q^2 - 1)/399])

    ndim = len(priors)  # number of dimensions = number of free parameters = nfree

    # name of the output files
    direc = '/scratch/user/joncohn/dyn_cluster/nest_out/'

    # PREPARE THINGS FOR MODEL INPUTS
    mod_ins = dm.model_prep(data=params['data'], ds=params['ds'], lucy_out=params['lucy'], lucy_b=params['lucy_b'],
                            lucy_mask=params['lucy_mask'], lucy_in=params['lucy_in'], lucy_it=params['lucy_it'],
                            data_mask=params['mask'], grid_size=params['gsize'], res=params['resolution'],
                            x_std=params['x_fwhm'], y_std=params['y_fwhm'], pa=params['PAbeam'],
                            xyerr=[params['xerr0'], params['xerr1'], params['yerr0'], params['yerr1']],
                            zrange=[params['zi'], params['zf']])
    lucy_mask, lucy_out, beam, fluxes, freq_ax, f_0, fstep, input_data, noise, co_ell_sb, co_ell_rad = mod_ins

    thresh = params['thresh']  # dlogz threshold
    maxc = params['maxc']  # max calls allowed
    # Set pkl output base that will be the same for tempsave and end save files
    basename = direc + params['outname'] + '_' + str(maxc) + '_' + str(params['nprocs']) + '_' + str(thresh) + '_'
    out_name = basename + str(time.time()) + '_tempsave' +  '.pkl'  # tempsave file

    if params['nprocs'] > 1:  # if multi-processing
        pool = Pool(processes=params['nprocs'])  # define pool
        # run Dynamic Nested Sampler (Dynamic is better for looking at posteriors)
        psampler = dynesty.DynamicNestedSampler(lnprob, prior, ndim, pool=pool, use_pool={'update_bound': False},
                                                queue_size=params['nprocs'], dlogz=thresh)
        ncall = psampler.ncall
        niter = psampler.it - 1
        # loop through the sampler until the dlogz threshold (thresh) is met, or until maxc calls have been made!
        for results in psampler.sample_initial(maxcall=maxc, dlogz=thresh):
            sys.stdout.flush()  # print things
            ncall += results[9]  # results[9] contains the ncalls made
            niter += 1
            delta_logz = results[-1]
            print('dlogz ' + str(delta_logz), 'thresh ' + str(thresh), 'nc ' + str(ncall), 'niter ' + str(niter))
            pass
        while True:  # once the dlogz threshold is met, start the batch stage! Sample the results in batches
            print('batch stage')
            res1 = psampler.results
            # Save things to the tempsave file at start of each loop, so if it doesn't converge we still get results!
            with open(out_name, 'wb+') as newfile:  # 'wb' bc binary
                pickle.dump(res1, newfile, pickle.HIGHEST_PROTOCOL)
                print('results pickle dumped: ' + out_name)  # takes ~0.01 - 0.06 seconds
            stop = stopping_function(psampler.results)  # evaluate stopping conditions
            if not stop:
                logl_bounds = weight_function(psampler.results)  # derive bounds
                print('stopping conditions not yet met')
                for results in psampler.sample_batch(logl_bounds=logl_bounds, maxcall=maxc):  # iterate the sample batch
                    sys.stdout.flush()  # print things
                    ncall += results[4]  # worst, ustar, vstar, loglstar, nc...
                    niter += 1
                    print('nc ' + str(ncall), 'niter ' + str(niter))
                    pass
                psampler.combine_runs()  # add new samples to previous results
            else:  # stopping conditions are met! Break the loop!
                break
    else:  # if not multi-processing
        # run Dynamic Nested Sampler (Dynamic is better for looking at posteriors)
        psampler = dynesty.DynamicNestedSampler(lnprob, prior, ndim, dlogz=thresh)
        ncall = psampler.ncall
        niter = psampler.it - 1
        # loop through the sampler until the dlogz threshold (thresh) is met, or until maxc calls have been made!
        for results in psampler.sample_initial(maxcall=maxc):
            ncall += results[9]  # results[9] contains the ncalls made
            niter += 1
            delta_logz = results[-1]
            print('dlogz ' + str(delta_logz), 'thresh ' + str(thresh), 'nc ' + str(ncall), 'niter ' + str(niter))
            pass
        while True:  # once the dlogz threshold is met, start the batch stage! Sample the results in batches
            print('batch stage')
            res1 = psampler.results
            # Save things to the tempsave file at start of each loop, so if it doesn't converge we still get results!
            with open(out_name, 'wb+') as newfile:  # 'wb' bc binary
                pickle.dump(res1, newfile, pickle.HIGHEST_PROTOCOL)
                print('results pickle dumped: ' + out_name)  # takes ~0.01 - 0.06 seconds
            stop = stopping_function(psampler.results)  # evaluate stopping conditions
            if not stop:
                logl_bounds = weight_function(psampler.results)  # derive bounds
                print('stopping conditions not yet met')
                for results in psampler.sample_batch(logl_bounds=logl_bounds, maxcall=maxc):  # iterate the sample batch
                    ncall += results[4]  # worst, ustar, vstar, loglstar, nc...
                    niter += 1
                    print('nc ' + str(ncall), 'niter ' + str(niter))
                    pass
                psampler.combine_runs()  # add new samples to previous results
            else:  # stopping conditions are met! Break the loop!
                break

    # IF THE CODE REACHES HERE, THE SAMPLING IS COMPLETE!
    print('runtime', time.time() - t00)

    # SAVE THE RESULTS
    res = psampler.results  # results
    endname = basename + str(time.time()) + '_end.pkl'  # final output pkl containing full results, with basename root
    with open(endname, 'wb+') as newfile:  # 'wb' because binary format
        pickle.dump(res, newfile, pickle.HIGHEST_PROTOCOL)
        print('results pickle dumped!')
    print(endname)  # output pkl file name

    print('Keys:', res.keys(), '\n')  # print accessible keys

    # PRINT OUT THE 1, 2, and 3-sigma quantiles for each free parameter
    weights = np.exp(res['logwt'] - res['logz'][-1])  # normalized weights
    for i in range(res['samples'].shape[1]):  # for each parameter
        quantiles_3 = dyfunc.quantile(res['samples'][:, i], [0.00135, 0.5, 0.99865], weights=weights)
        quantiles_2 = dyfunc.quantile(res['samples'][:, i], [0.02275, 0.5, 0.977], weights=weights)
        quantiles_1 = dyfunc.quantile(res['samples'][:, i], [0.16, 0.5, 0.84], weights=weights)
        print(quantiles_3, r'$3\sigma$')
        print(quantiles_2, r'$2\sigma$')
        print(quantiles_1, r'$1\sigma$')

    halt('Finished')  # halt the multi-processing so that the code stops!
