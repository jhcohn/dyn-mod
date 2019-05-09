import dyn_general as dg
import dyn_model as dm

import numpy as np
import emcee
import pickle
import time
import argparse
import matplotlib.pyplot as plt


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
        fstep=fstep,
        f_0=f_0,
        bl=params['bl'],
        xyrange=[params['xi'], params['xf'], params['yi'], params['yf']])

    return chi2


def test_dyn(theta, par_dict=None, mod_ins=None, params=None):
    """
    Get the chi squared of the model!

    :param theta: the parameter(s) being varied --> in this case, all potentially free params
    :param params: the parameter(s) being varied AND
    :param mod_ins: constant input model things
    :param params: all other model parameters (held fixed)

    :return: chi^2 (NOT reduced chi^2)
    """
    np.random.seed(123)

    lucy_mask, lucy_out, beam, fluxes, freq_ax, f_0, fstep, input_data, noise = mod_ins

    chi2 = dm.model_grid(
        # FREE PARAMETERS
        x_loc=theta[par_dict.keys().index('xloc')],
        y_loc=theta[par_dict.keys().index('yloc')],
        mbh=theta[par_dict.keys().index('mbh')],
        inc=np.deg2rad(theta[par_dict.keys().index('inc')]),
        vsys=theta[par_dict.keys().index('vsys')],
        theta=np.deg2rad(theta[par_dict.keys().index('PAdisk')]),
        ml_ratio=theta[par_dict.keys().index('ml_ratio')],
        sig_params=[theta[par_dict.keys().index('sig0')],
                    theta[par_dict.keys().index('r0')],
                    theta[par_dict.keys().index('mu')],
                    theta[par_dict.keys().index('sig1')]],
        f_w=theta[par_dict.keys().index('f')],
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
        chi2 = test_dyn(theta=theta, mod_ins=mod_ins, par_dict=priors, params=params)

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


def do_emcee(nwalkers=250, burn=100, steps=1000, printer=0, all_free=True, parfile=None, pool=None, save=True):

    t0_mc = time.time()
    # BUCKET set q=True once mge working
    # params, fixed_pars, files, priors, qobs = dg.par_dicts(parfile, q=True)  # get dicts of params and file names from parameter file
    # AVOID ERROR! --> all q^2 - cos(inc)^2 > 0 --> q^2 > cos(inc)^2 -> cos(inc) < q
    # params, fixed_pars, files, priors = dg.par_dicts(parfile, q=False)  # get dicts of params and file names from parameter file
    params, priors, qobs = dm.par_dicts(parfile, q=True)  # get dicts of params and file names from parameter file
    priors['inc'][0] = np.amax([priors['inc'][0], np.rad2deg(np.arccos(np.amin(qobs)))])  # BUCKET TURN ON ONCE MGE WORKING

    ndim = len(priors)  # number of dimensions = number of free parameters
    direc = '/Users/jonathancohn/Documents/dyn_mod/emcee_out/'

    mod_ins = dm.model_prep(data=params['data'], ds=params['ds'], lucy_out=params['lucy'], lucy_b=params['lucy_b'],
                            lucy_mask=params['lucy_mask'], lucy_in=params['lucy_in'], lucy_o=params['lucy_o'],
                            lucy_it=params['lucy_it'], data_mask=params['mask'], grid_size=params['gsize'],
                            res=params['resolution'], x_std=params['x_fwhm'], y_std=params['y_fwhm'],
                            pa=params['PAbeam'], zrange=[params['zi'], params['zf']],
                            xyerr=[params['xerr0'], params['xerr1'], params['yerr0'], params['yerr1']])

    '''  #
    params, priors = par_dicts(args['parfile'])

    # CREATE OUTNAME BASED ON INPUT PARS
    pars_str = ''
    for key in params:
        pars_str += str(params[key]) + '_'
    out = '/Users/jonathancohn/Documents/dyn_mod/outputs/NGC_3258_general_' + pars_str + '_subcube_ellmask_bl2.fits'

    mod_ins = model_prep(data=params['data'], lucy_out=params['lucy'], lucy_mask=params['lucy_mask'],
                         lucy_b=params['lucy_b'], lucy_in=params['lucy_in'], lucy_o=params['lucy_o'],
                         lucy_it=params['lucy_it'], data_mask=params['mask'], grid_size=params['gsize'],
                         res=params['resolution'], x_std=params['x_fwhm'], y_std=params['y_fwhm'], pa=params['PAbeam'])

    lucy_mask, lucy_out, beam, fluxes, freq_ax, f_0, fstep, input_data = mod_ins

    # CREATE MODEL CUBE!
    out = params['outname']  # '/Users/jonathancohn/Documents/dyn_mod/outputs/NGC_3258_general_' + pars_str + '_subcube_ellmask_bl2.fits'
    chisq = model_grid(resolution=params['resolution'], s=params['s'], x_loc=params['xloc'],
                       y_loc=params['yloc'], mbh=params['mbh'], inc=np.deg2rad(params['inc']), vsys=params['vsys'],
                       dist=params['dist'], theta=np.deg2rad(params['PAdisk']), input_data=input_data,
                       lucy_out=lucy_out, out_name=out, beam=beam, inc_star=params['inc_star'],
                       enclosed_mass=params['mass'], ml_ratio=params['ml_ratio'], sig_type=params['s_type'],
                       sig_params=[params['sig0'], params['r0'], params['mu'], params['sig1']], f_w=params['f'],
                       rfit=params['rfit'], menc_type=params['mtype'], ds=int(params['ds']),
                       chi2=True, zrange=[params['zi'], params['zf']], mge_f=params['mge'],
                       xyrange=[params['xi'], params['xf'], params['yi'], params['yf']],
                       xyerr=[params['xerr0'], params['xerr1'], params['yerr0'], params['yerr1']],
                       reduced=True, freq_ax=freq_ax, f_0=f_0, fstep=fstep)
    # '''  #




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
                                        pool=pool)
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
    for w in range(nwalkers):  # for each walker
        for p in range(len(p0_guess)):  # for each parameter
            # select random number, within 20% of param value (except for a few possible large steps)
            # adjuster = np.random.choice(np.concatenate((np.linspace(-0.2, 0.2, 200), np.linspace(-0.9, -0.2, 10),
            #                                             np.linspace(0.2, 0.9, 10))))
            adjuster = np.random.choice(np.linspace(-0.02, 0.02, 200))
            print(walkers.shape, w, p, p0_guess.shape, p0_guess[p], adjuster, ndim)
            walkers[w, p] = p0_guess[p] * (1 + adjuster)
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

    # main interface for emcee is EmceeSampler:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=[mod_ins, priors, param_names, params], pool=pool)
                                                        #(p0_guess, priors=priors, param_names=param_names,
    #                                                      fixed_pars=fixed_pars, files=files))
    # ^don't need theta in lnprob() because of setup of EmceeSampler function, right?

    # pool.close()  # BUCKET REMEMBER THIS

    # call lbprob as lnprob(p, means, icov) [where p is the position of a single walker. If no args parameter provided,
    # the calling sequence would be lnprob(p) instead.]
    # Note: using delta chi^2 instead of lnprob?
    # run a few burn-in steps in your MCMC chain to let the walkers explore the parameter space a bit and get settled
    # into the maximum of the density. We'll run a burn-in of burn=100 steps, starting from our initial guess p0:
    print('Burning in!')
    pos, prob, state = sampler.run_mcmc(p0, burn)
    # pos = final position of walkers after the 100 steps
    sampler.reset()  # clears walkers, and clears bookkeeping parameters in sampler so we get a fresh start

    print('Running MCMC!')
    # pos1, prob1, state1 = sampler.run_mcmc(pos, steps)  # production run of 1000 steps (probably overkill, for now)
    sampler.run_mcmc(pos, steps)  # production run of 1000 steps (probably overkill, for now)
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
        for i in range(ndim):
            outfile = direc + str(nwalkers) + '_' + str(burn) + '_' + str(steps) + '_flatchain_' + param_names[i]\
                      + '.pkl'
            print(outfile)
            with open(outfile, 'wb') as newfile:  # 'wb' because binary format
                pickle.dump(sampler.flatchain[:, i], newfile, pickle.HIGHEST_PROTOCOL)
                print('pickle dumped!')

    # You can make histograms of these samples to get an estimate of the density that you were sampling
    print('time in emcee ' + str(time.time() - t0_mc))
    if printer:
        import corner
        samples = sampler.flatchain.reshape((-1, ndim))  # .chain[:, 50:, :]
        fig = corner.corner(samples, labels=free_p.keys(), truths=free_p.values())  # truths here is just input value
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

            '''  #
            plt.plot(step_sampling, rhat, label="{:d}-dim".format(ndim), linewidth=2)

            ax = plt.gca()
            xmax = ax.get_xlim()[1]
            plt.hlines(1.1, xmin, xmax, linestyles="--")
            plt.ylabel("$\hat R$")
            plt.xlabel("chain length")
            plt.ylim(1, 2)
            plt.show()
            # '''  #

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
    Rhat = np.sqrt(var_t_sq / W_avg)
    print(Rhat.shape, 'Rhat shape')
    return Rhat


if __name__ == "__main__":
    # MAKE SURE I HAVE ACTIVATED THE iraf27 ENVIRONMENT!!!
    t0_full = time.time()
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument('--parfile')
    args = vars(parser.parse_args())

    '''  #
    import emcee
    from emcee.utils import MPIPool

    pool = MPIPool()
    if not pool.is_master():
        print('a')
        pool.wait()
        sys.exit(0)
    print('hi')

    try:
        print('b')
        from emcee.utils import MPIPool

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

    # do_emcee(nwalkers=250, burn=100, steps=1000, printer=0, parfile=None)
    flatchain = do_emcee(nwalkers=150, burn=1, steps=100, printer=1, parfile=args['parfile'], all_free=True, pool=None,
                         save=True)
    # flatchain = do_emcee(nwalkers=100, burn=100, steps=100, printer=1, parfile=args['parfile'])
    print(flatchain.shape)
    print('full time ' + str(time.time() - t0_full))
