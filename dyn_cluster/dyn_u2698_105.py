import dyn_105mod as dm

import sys
import numpy as np
import emcee
import pickle
import time
import argparse
import matplotlib.pyplot as plt


def next_pow_two(n):		
    i = 1		
    while i < n:		
        i = i << 1		
    return i		
def autocorr_func_1d(x, norm=True):		
    x = np.atleast_1d(x)		
    if len(x.shape) != 1:		
        print(len(x.shape))		
        raise ValueError("invalid dimensions for 1D autocorrelation function")		
    n = next_pow_two(len(x))		
    # Compute the FFT and then (from that) the auto-correlation function		
    f = np.fft.fft(x - np.mean(x), n=2*n)		
    acf = np.fft.ifft(f * np.conjugate(f))[:len(x)].real		
    acf /= 4*n		
    # Optionally normalize		
    if norm:		
        acf /= acf[0]		
    return acf		
# Automated windowing procedure following Sokal (1989)		
def auto_window(taus, c):		
    m = np.arange(len(taus)) < c * taus		
    if np.any(m):		
        return np.argmin(m)		
    return len(taus) - 1		
# Following the suggestion from Goodman & Weare (2010)		
def autocorr_gw2010(y, c=5.0):		
    f = autocorr_func_1d(np.mean(y, axis=0))		
    taus = 2.0*np.cumsum(f)-1.0		
    window = auto_window(taus, c)		
    return taus[window]


def test_dyn_m(theta, mod_ins=None, params=None):
    """
    Get the chi squared of the model!

    :param theta: the parameter(s) being varied --> in this case, mbh
    :param mod_ins: constant input model things
    :param params: all other model parameters (held fixed)

    :return: chi^2 (NOT reduced chi^2)
    """
    np.random.seed(123)

    lucy_mask, lucy_out, beam, fluxes, freq_ax, f_0, fstep, input_data, noise = mod_ins

    chi2 = dm.model_grid(
        # FREE
        mbh=theta[0],
        resolution=params['resolution'],
        x_loc=params['xloc'],
        y_loc=params['yloc'],
        inc=np.deg2rad(params['inc']),
        vsys=params['vsys'],
        theta=np.deg2rad(params['PAdisk']),
        ml_ratio=params['ml_ratio'],
        sig_params=[params['sig0'], params['r0'], params['mu'], params['sig1']],
        f_w=params['f'],
        # FIXED
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
        fstep=fstep,
        f_0=f_0,
        bl=params['bl'],
        xyrange=[params['xi'], params['xf'], params['yi'], params['yf']])

    return chi2


def test_dyn(theta, par_dict=None, mod_ins=None, params=None):
    """
    Get the chi squared of the model!

    :param theta: the parameter(s) being varied --> in this case, all potentially free params
    :param par_dict: the parameter(s) being varied
    :param mod_ins: constant input model things
    :param params: all other model parameters (held fixed)

    :return: chi^2 (NOT reduced chi^2)
    """
    np.random.seed(123)

    lucy_mask, lucy_out, beam, fluxes, freq_ax, f_0, fstep, input_data, noise = mod_ins

    chi2 = dm.model_grid(
        # FREE PARAMETERS
        x_loc=theta[par_dict.index('xloc')],  # par_dict.keys().index('xloc') (and etc beneath)
        y_loc=theta[par_dict.index('yloc')],
        mbh=theta[par_dict.index('mbh')],
        inc=np.deg2rad(theta[par_dict.index('inc')]),
        vsys=theta[par_dict.index('vsys')],
        theta=np.deg2rad(theta[par_dict.index('PAdisk')]),
        ml_ratio=theta[par_dict.index('ml_ratio')],
        sig_params=[theta[par_dict.index('sig0')],
                    params['r0'],
                    params['mu'],
                    params['sig1']],
                    #theta[par_dict.index('r0')],
                    #theta[par_dict.index('mu')],
                    #theta[par_dict.index('sig1')]],
        f_w=theta[par_dict.index('f')],
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

    return chi2


def lnprior_m(theta, priors, param_names):
    """
    Calculate the ln of the prior

    :param theta: input parameters
    :param priors: prior boundary dictionary, with structure: {'param_name': [pri_min, pri_max]}
    :param param_names: parameter dictionary key names
    :return: ln of the prior (0, ie flat, if theta within boundaries; -inf if theta outside of boundaries)
    """

    lnp = 0.  # apply a flat prior, so equivalent probability for all values within the prior range
    for p in range(len(theta)):  # for each parameter in the param vector theta
        if theta[p] < priors[0] or theta[p] > priors[1]:
            lnp = -np.inf  # kill anything that is outside the prior boundaries

    return lnp


def lnprior(theta, priors, param_names):
    """
    Calculate the ln of the prior

    :param theta: input parameters
    :param priors: prior boundary dictionary, with structure: {'param_name': [pri_min, pri_max]}
    :param param_names: parameter dictionary key names
    :return: ln of the prior (0, ie flat, if theta within boundaries; -inf if theta outside of boundaries)
    """

    lnp = 0.  # apply a flat prior, so equivalent probability for all values within the prior range
    for p in range(len(theta)):  # for each parameter in the param vector theta
        if theta[p] < priors[param_names[p]][0] or theta[p] > priors[param_names[p]][1]:
            lnp = -np.inf  # kill anything that is outside the prior boundaries

    return lnp


def lnprob(theta, mod_ins=None, priors=None, param_names=None, params=None):
    """
    Not computing full posterior probability, so just use chi squared (P propto exp(-chi^2/2) --> ln(P) ~ -chi^2 / 2

    :param theta: input parameter vector
    :param mod_ins: constant input model things
    :param priors: prior boundary dictionary, with structure: {'param_name': [pri_min, pri_max]}
    :param param_names: list of parameter names
    :param params: dictionary of fixed input parameters (e.g. resolution, beam params, etc.)

    :return: -0.5 * chi^2 (ln likelihood)
    """
    pri = lnprior(theta, priors, param_names)
    if pri == -np.inf:
        chi2 = 1.  # doesn't matter, because -inf + (-0.5 * 1) = -inf
    else:
        chi2 = test_dyn(theta=theta, mod_ins=mod_ins, par_dict=param_names, params=params)

    # print('lnprob stuff', pri, chi2, pri + (-0.5 * chi2))
    return pri + (-0.5 * chi2)


def lnprob_m(theta, mod_ins=None, params=None, priors=None, param_names=None):
    """
    Not computing full posterior probability, so just use chi squared (P propto exp(-chi^2/2) --> ln(P) ~ -chi^2 / 2

    :param theta: input parameter vector
    :param mod_ins: constant input model things
    :param params: dictionary of input parameters (e.g. resolution, beam params, etc.)
    :param priors: prior boundary dictionary, with structure: {'param_name': [pri_min, pri_max]}
    :param param_names: list of parameter names

    :return: -0.5 * chi^2 (ln likelihood)
    """
    pri = lnprior_m(theta, priors, param_names)
    if pri == -np.inf:
        chi2 = 1.  # doesn't matter, because -inf + (-0.5 * 1) = -inf
    else:
        chi2 = test_dyn_m(theta, params=params, mod_ins=mod_ins)

    return pri + (-0.5 * chi2)


def do_emcee(nwalkers=250, burn=100, steps=1000, printer=0, all_free=True, parfile=None, pool=None, save=True, nthr=4, q=True):

    t0_mc = time.time()
    # BUCKET set q=True once mge working
    # params, fixed_pars, files, priors, qobs = dg.par_dicts(parfile, q=True)  # get dicts of params and file names from parameter file
    # AVOID ERROR! --> all q^2 - cos(inc)^2 > 0 --> q^2 > cos(inc)^2 -> cos(inc) < q
    # params, fixed_pars, files, priors = dg.par_dicts(parfile, q=False)  # get dicts of params and file names from parameter file
    # BUCKET UNCOMMENT BELOW WHEN USING MGE!
    if q:
        print('Yes q')
        params, priors, qobs = dm.par_dicts(parfile, q=True)  # get dicts of params and file names from parameter file 
        print(priors['inc'])
        print(qobs)
        print(np.amax(np.rad2deg(np.arccos(np.sqrt((400*qobs**2 - 1.)/399.)))))
        qint_pri = np.amax(np.rad2deg(np.arccos(np.sqrt((400*qobs**2 - 1.)/399.))))
        priors['inc'][0] = np.amax([priors['inc'][0], np.rad2deg(np.arccos(np.amin(qobs)))])  # BUCKET TURN ON ONCE MGE WORKING
        priors['inc'][0] = np.amax([priors['inc'][0], qint_pri])
    else:
        print('No q')
        params, priors = dm.par_dicts(parfile, q=False)  # get dicts of params and file names from parameter file
        qobs = None
    print(priors['inc'])
    # np.sqrt(qobs**2 - np.cos(inc)**2)/np.sin(inc) > 0.05 --> sin(inc) < sqrt(qobs**2 - np.cos(inc)**2)/.05
    # --> sin^2(inc) < qobs^2/0.05 - cos^2(inc)/0.05 --> sin^2(inc) + cos^2(inc)/0.05^2 < qobs^2/0.05^2
    # NOTE: sin^2(x) + C*cos^2(x) = 0.5*(C*cos(2x) + C + 1 - cos(2x)) = 0.5(cos(2x)(C-1) + (C+1))
    # --> 0.5(19*cos(2inc) + 21) < 20*qobs^2 --> 399cos^2(inc)+1 < 400*q^2 --> inc < arccos(sqrt[(400q^2 - 1)/399])

    ndim = len(priors)  # number of dimensions = number of free parameters
    direc = '/scratch/user/joncohn/emcee_out/'

    mod_ins = dm.model_prep(data=params['data'], ds=params['ds'], lucy_out=params['lucy'], lucy_b=params['lucy_b'], lucy_mask=params['lucy_mask'], lucy_in=params['lucy_in'], lucy_o=params['lucy_o'], lucy_it=params['lucy_it'], data_mask=params['mask'], grid_size=params['gsize'], res=params['resolution'], x_std=params['x_fwhm'], y_std=params['y_fwhm'], pa=params['PAbeam'], zrange=[params['zi'], params['zf']], xyerr=[params['xerr0'], params['xerr1'], params['yerr0'], params['yerr1']])

    if not all_free:
        ndim = 1
        param_names = ['mbh']
        p0_guess = np.asarray([params['mbh']])  # initial guess

        # SET UP RANDOM CLUSTER OF POINTS NEAR INITIAL GUESS
        walkers = np.zeros(shape=(nwalkers, len(p0_guess)))  # initialize walkers; there are nwalkers for each parameter
        stepper_full = np.zeros_like(walkers)  # initializes stepper for each parameter
        for w in range(nwalkers):  # for each walker
            for p in range(len(p0_guess)):  # for each parameter
                # select random number, within 20% of param value (except for a few possible large steps)
                # adjuster = np.random.choice(np.concatenate((np.linspace(-0.2, 0.2, 200), np.linspace(-0.9, -0.2, 10),
                #                                             np.linspace(0.2, 0.9, 10))))
                adjuster = np.random.choice(np.linspace(-0.02, 0.02, 200))
                walkers[w, p] = p0_guess[p] * (1 + adjuster)
                # stepper_full[w, p] = p0_guess[p] * (1 + adjuster)

        # the actual p0 array needs to be the cluster of initialized walkers
        p0 = walkers

        # main interface for emcee is EmceeSampler:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_m, args=[params, mod_ins, priors['mbh'], param_names],
                                        pool=pool, nthreads=nthr)
        pool.close()
        print('p0', p0)
        print('Burning in!')
        pos, prob, state = sampler.run_mcmc(p0, burn)
        # pos = final position of walkers after the 100 steps
        sampler.reset()  # clears walkers, and clears bookkeeping parameters in sampler so we get a fresh start

        print('Running MCMC!')
        sampler.run_mcmc(pos, steps)  # production run of 1000 steps (probably overkill, for now)
        print("printing...")

        # Another good test of whether or not sampling went well: mean acceptance fraction should be 0.25 to 0.5
        print("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)))

        for i in range(ndim):
            outfile = direc + 'flatchain_' + param_names[i] + '_' + str(nwalkers) + '_' + str(burn) + '_' + str(
                steps) + '.pkl'
            print(outfile)
            with open(outfile, 'wb') as newfile:  # 'wb' because binary format
                pickle.dump(sampler.flatchain[:, i], newfile, pickle.HIGHEST_PROTOCOL)
                print('pickle dumped!')

        # You can make histograms of these samples to get an estimate of the density that you were sampling
        print('time in emcee ' + str(t0_mc - time.time()))
        if printer:
            fig = plt.figure()

            plt.hist(np.log10(sampler.flatchain), 100, color="k", histtype="step")  # axes[i]
            percs = np.percentile(np.log10(sampler.flatchain), [16., 50., 84.])
            threepercs = np.percentile(np.log10(sampler.flatchain), [0.15, 50., 99.85])  # 3sigma

            plt.axvline(percs[1], color='k', ls='-')  # axes[i]
            plt.axvline(percs[0], color='k', ls='--')  # axes[i]
            plt.axvline(percs[2], color='k', ls='--')  #
            plt.axvline(threepercs[0], color='b', ls='--')  # axes[i]
            plt.axvline(threepercs[2], color='b', ls='--')  #
            plt.tick_params('both', labelsize=8)
            # plt.title("Dimension {0:d}".format(i))
            plt.title('mbh' + ': ' + str(round(percs[1], 2)) + ' (+' + str(round(percs[2] - percs[1], 2)) + ', -'
                      + str(round(percs[1] - percs[0], 2)) + ')', fontsize=8)

            plt.show()
            print(oop)

    # SET UP "HYPERPARAMETER" VALUES IN 50 DIMENSIONS
    # ndim = 50

    p0_guess = [] # initialize parameter vector
    param_names = []  # initialize parameter names vector
    free_p = {}
    for key in priors:  # for each parameter
        p0_guess.append(params[key])  # initial guess
        param_names.append(key)  # parameter name
        free_p[key] = params[key]
    p0_guess = np.asarray(p0_guess)
    # print('p0', p0_guess)

    # SET UP RANDOM CLUSTER OF POINTS NEAR INITIAL GUESS
    walkers = np.zeros(shape=(nwalkers, len(p0_guess)))  # initialize walkers; there are nwalkers for each parameter
    print('did it change')
    for w in range(nwalkers):  # for each walker
        for p in range(len(p0_guess)):  # for each parameter
            if param_names[p] == 'mbh':  # BUCKET I think this worked
                # print('mbh here', p0_guess[p])
                adjuster = np.random.choice(np.linspace(0.5, 2., 200.))
                walkers[w, p] = p0_guess[p] * adjuster
            elif param_names[p] == 'sig1' or param_names[p] == 'sig0' or param_names[p] == 'mu' or param_names[p] == 'r0':
                walkers[w,p] = np.random.choice(np.linspace(0.4, 10., 200.)) 
            elif param_names[p] == 'xloc':
                walkers[w,p] = np.random.choice(np.linspace(129.01, 131.01))
            elif param_names[p] == 'yloc':
                walkers[w,p] = np.random.choice(np.linspace(149.01, 151.01))
            elif param_names[p] == 'inc':
                walkers[w,p] = np.random.choice(np.linspace(66., 73.))
            elif param_names[p] == 'PAdisk':
                walkers[w,p] = np.random.choice(np.linspace(10., 30.))
            elif param_names[p] == 'vsys':
                walkers[w,p] = np.random.choice(np.linspace(6370., 6570.))
            elif param_names[p] == 'ml_ratio':
                walkers[w,p] = np.random.choice(np.linspace(1.5, 2.5))
            elif param_names[p] == 'f':
                walkers[w,p] = np.random.choice(np.linspace(0.96, 1.04))
	    '''  #
	    elif param_names[p] == 'sig1' or param_names[p] == 'sig0' or param_names[p] == 'mu' or param_names[p] == 'r0':
                adjuster = np.random.choice(np.linspace(0.2, 2., 200.))
                walkers[w, p] = p0_guess[p] * adjuster
	    # select random number, within 20% of param value (except for a few possible large steps)
            # adjuster = np.random.choice(np.concatenate((np.linspace(-0.2, 0.2, 200), np.linspace(-0.9, -0.2, 10),
            #                                             np.linspace(0.2, 0.9, 10))))
            else:
                adjuster = np.random.choice(np.linspace(-0.02, 0.02, 200))
                # print(walkers.shape, w, p, p0_guess.shape, p0_guess[p], adjuster, ndim)
                walkers[w, p] = p0_guess[p] * (1 + adjuster)
            # '''
	    '''
            if param_names[p] == 'mbh':
                stepper_full[w, p] = np.random.choice(np.logspace(-1., 1., 200))
            else:
                stepper_full[w, p] = np.random.choice(np.linspace(-1., 1., 200))
            '''

    # initialize walkers: start all walkers in a cluster near p0
    # for wa in range(nwalkers):  # for each set of walkers
    #     newstep = stepper_full[wa, :]  # each row is a list of steps for all the params
    #     walkers[wa, :] = p0_guess + newstep  # initialize walkers

    # the actual p0 array needs to be the cluster of initialized walkers
    p0 = walkers
    print(param_names)
    print('p0', p0)
    print(param_names.index('xloc'))
    print(p0[param_names.index('mbh')])
    print(p0[:, param_names.index('mbh')])
    # print(oop)

    # main interface for emcee is EmceeSampler:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=[mod_ins, priors, param_names, params], pool=pool, threads=nthr)
                                                        #(p0_guess, priors=priors, param_names=param_names,
    #                                                      fixed_pars=fixed_pars, files=files))
    # ^don't need theta in lnprob() because of setup of EmceeSampler function, right?

    # pool.close()  # BUCKET REMEMBER THIS

    # call lbprob as lnprob(p, means, icov) [where p is the position of a single walker. If no args parameter provided,
    # the calling sequence would be lnprob(p) instead.]
    # Note: using delta chi^2 instead of lnprob?
    # run a few burn-in steps in your MCMC chain to let the walkers explore the parameter space a bit and get settled
    # into the maximum of the density. We'll run a burn-in of burn=100 steps, starting from our initial guess p0:
    # NO LONGER USING BURN IN
    # print('Burning in!')
    # pos, prob, state = sampler.run_mcmc(p0, burn)
    # pos = final position of walkers after the 100 steps
    # sampler.reset()  # clears walkers, and clears bookkeeping parameters in sampler so we get a fresh start

    # RUN EMCEE, SAVING PROGRESS AND CHECKING CONVERGENCE!
    print('Running MCMC!')

    unds = 0
    pf = ''
    for i in parfile:
        if i == '_':
            unds += 1
        elif unds == 2:
            pf += i
    direc += 'qflat_u2698_bcluster_strict_' + pf + '_ds' + str(params['ds']) + '_lucy' + str(params['lucy_it']) + '_' + str(time.time()) + '_'

    tempchain = direc + str(nwalkers) + '_' + str(burn) + '_' + str(steps) + 'tempchain.pkl'
    # https://stackoverflow.com/questions/50294428/implement-early-stopping-in-emcee-given-a-convergence-criterion

    old_tau = np.inf
    chunk = 100  # 20
    cvg = False
    autocorr = np.empty(steps)
    for ii, result in enumerate(sampler.sample(p0, iterations=steps, storechain=True)):  # iterations=500, storechain=False
        ii += 1  # sampler.iteration == ii
        print('iteration ' + str(ii) + ' complete')
        print('time taken so far', time.time() - t0_mc)
        if (ii == 0) or (ii % chunk) == 0:
            tstore = time.time()
            position, lprob, rstate = result  # result[0]
            f = open(tempchain, "wb")
            # for k in range(len(position)):  # position.shape[0]
            pickle.dump(sampler.chain, f, pickle.HIGHEST_PROTOCOL)
            # f.write("{0:4d} {1:s}\n".format(k, " ".join(str(position[k]))))
            f.close()
            '''  # COMMENTING OUT BELOW BECAUSE TAKING UP TIME!
            N = np.exp(np.linspace(np.log(100), np.log(len(sampler.chain[0,:,0])), 10)).astype(int)		
            gw2010 = np.zeros(shape=(len(N), len(sampler.chain[0,:,0])))		
            for p in range(len(sampler.chain[0,:,0])):		
                pchain = sampler.chain[:,p,:]		
                # Compute the estimators for a few different chain lengths		
                const_tau = N / 50.		
                # gw2010 = np.empty(len(N))		
                # new = np.empty(len(N))		
                for j, n in enumerate(N):		
                    gw2010[j, p] = autocorr_gw2010(pchain[:, :n])		
                    # new[j] = autocorr_new(sampler.chain[:, :n])		
            print('gw2010 and tau=N/50 estimates:')		
            print(gw2010[-1, :])		
            print(const_tau[-1])
            # '''  #
            # BUCKET COMMENTING OUT CVG STUFF FOR NOW (weird that it insta-worked...)
            # if np.all(gw2010[-1, :] < ii / 50.) and np.all(gw2010[-2] < ii / 50.):		
            #     cvg = True		
            # print(cvg)
            # if cvg:
            #     break
            '''  #
            tau = sampler.get_autocorr_time()  # tol=0)
            print('tau', tau)  # all nans originally, started getting some not nans at iteration 10! hooray!
            autocorr[ii - 1] = np.mean(tau)
            cvg = np.all(tau * 75 < ii)  # < sampler.iteration
            # print(cvg)
            cvg &= np.all(np.abs(old_tau - tau) / tau < 0.01)  # want change in tau to be < 1% (shows chains stable!)
            # '''  #
            # old_tau = tau
            print(time.time() - tstore, 'overhead')  # 0.01s, 0.003s, etc.
        else:
            pass

    # pos1, prob1, state1 = sampler.run_mcmc(pos, steps)  # production run of 1000 steps (probably overkill, for now)
    # sampler.run_mcmc(pos, steps)  # production run of 1000 steps (probably overkill, for now)  # No longer need to do this because done in loop above!
    print("printing...")

    '''  #
    # look at how well things are burned in: http://dfm.io/emcee/current/user/line/#maximum-likelihood-estimation
    # sampler.chain.shape = (nwalkers, steps, ndim)  # gives parameter values for each walker at each step in chaim
    for i in ndim:
        plt.plot(sampler.chain[:, :, i])  # maybe?
        plt.show()
    
    # '''  #

    # sampler now has a property EnsembleSampler.chain that is a numpy array with shape (250, 1000, 50). More useful
    # object is the EnsembleSampler.flatchain which has the shape (250000, 50) and contains all samples reshaped into a
    # flat list. So we now have 250000 unbiased samples of the density p(x").

    # Another good test of whether or not sampling went well: check the mean acceptance fraction of the ensemble:
    #  EnsembleSampler.acceptance_fraction()
    print("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)))
    print('want between 0.25 and 0.5')  # should be 0.25 to 0.5

    if save:
	unds = 0
        pf = ''
        for i in parfile:
            if i == '_':
                unds += 1
            elif unds == 2:
                pf += i
        direc += str(time.time()) + '_'
        # direc += 'mpi' + pf + '_'
        out_chain = direc + str(nwalkers) + '_' + str(burn) + '_' + str(steps) + '_fullchain.pkl'
        with open(out_chain, 'wb') as newfile:  # 'wb' because binary format
            pickle.dump(sampler.chain, newfile, pickle.HIGHEST_PROTOCOL)
            print('full chain pickle dumped!')
        out_flatchain = direc + str(nwalkers) + '_' + str(burn) + '_' + str(steps) + '_full_flatchain.pkl'
        with open(out_flatchain, 'wb') as newfile:  # 'wb' because binary format
            pickle.dump(sampler.chain, newfile, pickle.HIGHEST_PROTOCOL)
            print('flatchain pickle dumped!')
        '''  #
        for i in range(ndim):
            outfile = direc + str(nwalkers) + '_' + str(burn) + '_' + str(steps) + '_flatchain_' + param_names[i]\
                      + '.pkl'
            print(outfile)
            with open(outfile, 'wb') as newfile:  # 'wb' because binary format
                pickle.dump(sampler.flatchain[:, i], newfile, pickle.HIGHEST_PROTOCOL)
                print('pickle dumped!')
        # '''  #
        # with open(direc + str(nwalkers) + '_' + str(burn) + '_' + str(steps) + '_gw2010.pkl', 'wb') as newfile:		
        #     pickle.dump(gw2010, newfile, pickle.HIGHEST_PROTOCOL)		
        # with open(direc + str(nwalkers) + '_' + str(burn) + '_' + str(steps) + '_Nover50.pkl', 'wb') as newfile:		
        #     pickle.dump(const_tau, newfile, pickle.HIGHEST_PROTOCOL)

    # You can make histograms of these samples to get an estimate of the density that you were sampling
    print('time in emcee ' + str(time.time() - t0_mc))
    if printer:
        import corner
        samples = sampler.flatchain.reshape((-1, ndim))  # .chain[:, 50:, :]
        # truths here are just input value
        fig = corner.corner(samples, labels=free_p.keys(), truths=[v for v in free_p.values()])
        fig.savefig(direc + str(nwalkers) + '_' + str(burn) + '_' + str(steps) + "_triangle.png")
        # fig = plt.figure()
        # axes = [plt.subplot(221), plt.subplot(222), plt.subplot(223), plt.subplot(224)]
        # PLOT CONVERGENCE TEST
        for i in range(ndim):
            print("{0:3d}\t{1: 5.4f}\t\t{2:5.4f}\t\t{3:3.2f}".format(
                ndim,
                sampler.chain[:, i].reshape(-1, sampler.chain[:, i].shape[-1]).mean(axis=0)[0],
                sampler.chain[:, i].reshape(-1, sampler.chain[:, i].shape[-1]).std(axis=0)[0],
                gelman_rubin(sampler.chain[:, :, i], 1)))  # gelman_rubin(sampler.chain[:, :, i], 1)[0]))
            # sampler.chain has shape (nwalkers, niters, ndim)

            xmin = 500
            chain_length = sampler.chain[:, :, i].shape[1]
            step_sampling = np.arange(xmin, chain_length, 50)
            rhat = np.array([gelman_rubin(sampler.chain[:, :, i][:, :steps, :], 1)[0] for steps in step_sampling])
            # print(np.amax(rhat), np.amin(rhat))
            # if not numpy.isnan(rhat).any():
            #     plt.plot(step_sampling, rhat)
            #     plt.show()

        # PLOT POSTERIORS
        fig, axes = plt.subplots(3, 4)  # 3 rows, 4 cols of subplots; because there are 12 free params
        row = 0
        col = 0
        for i in range(ndim):
            if i == 4 or i == 8:  # or i == 12:
                row += 1
                col = 0

            if free_p.keys()[i] == 'mbh':
                axes[row, col].hist(np.log10(sampler.flatchain[:, i]), 100, color="k", histtype="step")  # axes[i]
                percs = np.percentile(np.log10(sampler.flatchain[:, i]), [16., 50., 84.])
                threepercs = np.percentile(np.log10(sampler.flatchain[:, i]), [0.15, 50., 99.85])  # 3sigma
            else:
                axes[row, col].hist(sampler.flatchain[:, i], 100, color="k", histtype="step")  # axes[i]
                percs = np.percentile(sampler.flatchain[:, i], [16., 50., 84.])
                threepercs = np.percentile(sampler.flatchain[:, i], [0.15, 50., 99.85])  # 3sigma
            # print([key for key in params], percs)
            # print(threepercs)
            axes[row, col].axvline(percs[1], ls='-')  # axes[i]
            axes[row, col].axvline(percs[0], ls='--')  # axes[i]
            axes[row, col].axvline(percs[2], ls='--')  #
            axes[row, col].tick_params('both', labelsize=8)
            # plt.title("Dimension {0:d}".format(i))
            axes[row, col].set_title(free_p.keys()[i] + ': ' + str(round(percs[1],4)) + ' (+'
                                     + str(round(percs[2] - percs[1], 4)) + ', -'
                                     + str(round(percs[1] - percs[0], 4)) + ')', fontsize=8)
            col += 1

        plt.show()
        plt.close()
        for i in range(ndim):
            plt.plot(sampler.flatchain[:, i])
            plt.title(free_p.keys()[i])
            plt.xlabel(r'Iteration')
            plt.ylabel('Value')
            plt.show()

    return sampler.flatchain


def gelman_rubin(chain, ndim):
    # http://joergdietrich.github.io/emcee-convergence.html
    # http://www.stat.columbia.edu/~gelman/research/published/itsim.pdf
    ssq = np.var(chain, axis=1, ddof=ndim)
    W_avg = np.mean(ssq, axis=0)
    tb = np.mean(chain, axis=1)
    tbb = np.mean(tb, axis=0)
    cm = chain.shape[0]
    cn = chain.shape[1]
    B_var = cn / (cm - 1) * np.sum((tbb - tb)**2, axis=0)
    var_t_sq = W_avg * (cn - 1) / cn + B_var / cn
    Rhat = np.sqrt(var_t_sq / W_avg)  # just a number (at least, if ndim=1, i.e. if chain=sampler.chain[:,:,i])
    return Rhat


def halt(message):
    """Exit, closing pool safely."""
    print(message)
    try:
        pool.close()
    except:
        pass
    sys.exit(0)


if __name__ == "__main__":
    # MAKE SURE I HAVE ACTIVATED THE three ENVIRONMENT!!!
    t0_full = time.time()
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument('--parfile')
    parser.add_argument('--w')
    parser.add_argument('--b')
    parser.add_argument('--s')
    args = vars(parser.parse_args())

    #from multiprocessing import Pool
    #
    #with Pool() as pool:
    #    flatchain = do_emcee(nwalkers=26, burn=1, steps=1, printer=1, parfile=args['parfile'], all_free=True,
    #                         pool=pool, save=True)
    #    print("Multiprocessing took {0:.1f} seconds".format(multi_time))
    #    print("{0:.1f} times faster than serial".format(serial_time / multi_time))


    # '''  #
    from emcee.utils import MPIPool

    pool = MPIPool()
    if not pool.is_master():
        print('a')
        pool.wait()
        sys.exit(0)
    print('hi')
    
    try:
        print('b')

        pool = MPIPool(debug=False, loadbalance=True)
        print('pool')
        if not pool.is_master():
            # Wait for instructions from the master process.
            print('c')
            pool.wait()
            sys.exit(0)
    except(ImportError, ValueError):
        pool = None
        print('Not using MPI')
    # '''  #

    # do_emcee(nwalkers=250, burn=100, steps=100, printer=0, parfile=None)
    print('here we go')
    flatchain = do_emcee(nwalkers=int(args['w']), burn=int(args['b']), steps=int(args['s']), printer=0, parfile=args['parfile'], all_free=True, pool=pool, save=True, nthr=12, q=True)
    # flatchain = do_emcee(nwalkers=100, burn=100, steps=100, printer=1, parfile=args['parfile'])
    print(flatchain.shape)
    print('full time ' + str(time.time() - t0_full))
    # print(oops)
    halt('Finished')

