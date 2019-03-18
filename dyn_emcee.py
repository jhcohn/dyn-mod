import dyn_general as dg

import numpy as np
import emcee
import pickle


def test_dyn(params=None, fixed_pars=None, files=None):
    np.random.seed(123)

    chi2 = dg.model_grid(resolution=fixed_pars['resolution'], s=fixed_pars['s'], x_loc=params['xloc'],
                         y_loc=params['yloc'], mbh=params['mbh'], inc=np.deg2rad(params['inc']), vsys=params['vsys'],
                         dist=fixed_pars['dist'], theta=np.deg2rad(params['PAdisk']), data_cube=files['data'],
                         data_mask=files['mask'], lucy_output=files['lucy'], out_name=out, ml_ratio=params['ml_ratio'],
                         enclosed_mass=files['mass'], menc_type=files['mtype'] == True, sig_type=fixed_pars['s_type'],
                         grid_size=fixed_pars['gsize'], x_fwhm=fixed_pars['x_fwhm'], y_fwhm=fixed_pars['y_fwhm'],
                         sig_params=[params['sig0'], params['r0'], params['mu'], params['sig1']],
                         pa=fixed_pars['PAbeam'], f_w=params['f'], lucy_in=files['lucy_in'], lucy_b=files['lucy_b'],
                         lucy_o=files['lucy_o'], lucy_mask=files['lucy_mask'], lucy_it=fixed_pars['lucy_it'], chi2=True,
                         zrange=[fixed_pars['zi'], fixed_pars['zf']], xyrange=[fixed_pars['xi'], fixed_pars['xf'],
                                                                               fixed_pars['yi'], fixed_pars['yf']])

    return chi2


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


def chisq(theta, priors=None, param_names=None, fixed_pars=fixed_pars, files=files):
    """
    Not computing full posterior probability, so just use chi squared (P propto exp(-chi^2/2) --> ln(P) ~ -chi^2 / 2

    :param theta: input parameter vector
    :param priors: prior boundary dictionary, with structure: {'param_name': [pri_min, pri_max]}
    :param param_names: list of parameter names
    :param fixed_pars: dictionary of fixed input parameters (e.g. resolution, beam params, etc.)
    :param files: dictionary containing names of files

    :return: -0.5 * chi^2 (ln likelihood)
    """
    pri = lnprior(theta, priors, param_names)
    if pri == -np.inf:
        chi2 = 1.  # doesn't matter, because -inf + (-0.5 * 1) = -inf
    else:
        chi2 = test_dyn(theta, fixed_pars=fixed_pars, files=files)

    return pri + (-0.5 * chi2)


def do_emcee(nwalkers=250, burn=100, steps=1000, printer=0, parfile=None, pri_maxes=None, pri_mins=None):

    params, fixed_pars, files, priors = dg.par_dicts(parfile)  # get dicts of params and file names from parameter file
    ndim = len(params)  # number of dimensions = number of free parameters

    # SET UP "HYPERPARAMETER" VALUES IN 50 DIMENSIONS
    # ndim = 50

    p0 = [] # initialize parameter vector
    param_names = []  # initialize parameter names vector
    for key in params:  # for each parameter
        p0.append(params[key])  # initial guess
        param_names.append(key)  # parameter name
    p0 = np.asarray(p0)
    print('p0', p0)

    # SET UP RANDOM CLUSTER OF POINTS NEAR INITIAL GUESS
    walkers = np.zeros(shape=(nwalkers, len(p0)))  # initialize walkers; there are nwalkers for each parameter
    stepper_full = np.zeros_like(walkers)  # initializes stepper for each parameter
    for w in range(nwalkers):  # for each walker
        for p in range(len(p0)):  # for each parameter
            # select random number, within 20% of param value (except for a few possible large steps)
            adjuster = np.random.choice(np.concatenate((np.linspace(-0.2, 0.2, 200), np.linspace(-0.9, -0.2, 10),
                                                        np.linspace(0.2, 0.9, 10))))
            stepper_full[w, p] = p0[p] * (1 + adjuster)
            '''
            if param_names[p] == 'mbh':
                stepper_full[w, p] = np.random.choice(np.logspace(-1., 1., 200))
            else:
                stepper_full[w, p] = np.random.choice(np.linspace(-1., 1., 200))
            '''

    # initialize walkers: start all walkers in a cluster near p0
    for wa in range(nwalkers):  # for each set of walkers
        newstep = stepper_full[wa, :]  # each row is a list of steps for all the params
        walkers[wa, :] = p0 + newstep  # initialize walkers

    # the actual p0 array needs to be the cluster of initialized walkers
    p0 = walkers

    '''
    means = np.random.rand(ndim)  # returns random numbers between 0, 1 with shape ndim

    cov = 0.5 - np.random.rand(ndim ** 2).reshape((ndim, ndim))  # NxN covariance matrix
    cov = np.triu(cov)  # takes array, and returns just the upper triangle of the array
    cov += cov.T - np.diag(cov.diagonal())
    cov = np.dot(cov, cov)

    icov = np.linalg.inv(cov)  # inverse of covariance matrix

    # guess a starting point for each of the 250 walkers. position will be an ndim vector, so initial guess will be
    # 250x(ndim) array (or list of 250 arrays each with ndim elements)
    # nwalkers = 250
    p0 = np.random.rand(ndim * nwalkers).reshape((nwalkers, ndim))  # initial (bad) guess: between 0 and 1 for each

    # main interface for emcee is EmceeSampler:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=[means, icov])
    '''

    # main interface for emcee is EmceeSampler:
    sampler = emcee.EnsembleSampler(nwalkers, ndim,
                                    chisq(priors=priors, param_names=param_names, fixed_pars=fixed_pars, files=files))
    # ^don't need theta in chisq() because of setup of EmceeSampler function, right?

    # call lbprob as lnprob(p, means, icov) [where p is the position of a single walker. If no args parameter provided,
    # the calling sequence would be lnprob(p) instead.]
    # Note: using delta chi^2 instead of lnprob?
    # run a few burn-in steps in your MCMC chain to let the walkers explore the parameter space a bit and get settled
    # into the maximum of the density. We'll run a burn-in of burn=100 steps, starting from our initial guess p0:
    print('p0', p0)
    print('Burning in!')
    pos, prob, state = sampler.run_mcmc(p0, burn)
    # pos = final position of walkers after the 100 steps
    sampler.reset()  # clears walkers, and clears bookkeeping parameters in sampler so we get a fresh start

    print('Running MCMC!')
    # pos1, prob1, state1 = sampler.run_mcmc(pos, steps)  # production run of 1000 steps (probably overkill, for now)
    sampler.run_mcmc(pos, steps)  # production run of 1000 steps (probably overkill, for now)
    print("printing...")

    # sampler now has a property EnsembleSampler.chain that is a numpy array with shape (250, 1000, 50). More useful
    # object is the EnsembleSampler.flatchain which has the shape (250000, 50) and contains all samples reshaped into a
    # flat list. So we now have 250000 unbiased samples of the density p(x").

    # Another good test of whether or not sampling went well: check the mean acceptance fraction of the ensemble:
    #  EnsembleSampler.acceptance_fraction()
    print("Mean acceptance fraction: {0:.3f}"
          .format(np.mean(sampler.acceptance_fraction)))  # should be 0.25 to 0.5

    # params = ['inc', 'mbh', 'gamma', 'beta']
    for i in range(ndim):
        outfile = jam_dir + 'flatchain_' + param_names[i] + '-' + str(walk) + '-' + str(burn) + '-' + str(step) + '.pkl'
        print(outfile)
        with open(outfile, 'wb') as newfile:  # 'wb' because binary format
            pickle.dump(sampler.flatchain[:, i], newfile, pickle.HIGHEST_PROTOCOL)
            print('pickle dumped!')

    # You can make histograms of these samples to get an estimate of the density that you were sampling
    if printer:
        # fig = plt.figure()
        # axes = [plt.subplot(221), plt.subplot(222), plt.subplot(223), plt.subplot(224)]
        fig, axes = plt.subplots(3, 4)  # 3 rows, 4 cols of subplots; because there are 12 free params
        for i in range(ndim):
            axes[i].hist(sampler.flatchain[:, i], 100, color="k", histtype="step")
            percs = np.percentile(sampler.flatchain[:, i], [16., 50., 84.])
            threepercs = np.percentile(sampler.flatchain[:, i], [0.15, 50., 99.85])  # 3sigma
            print(params[i], percs)
            print(threepercs)
            axes[i].axvline(percs[1], ls='-')
            axes[i].axvline(percs[0], ls='--')
            axes[i].axvline(percs[2], ls='--')
            # plt.title("Dimension {0:d}".format(i))
            axes[i].title(params[i] + ': ' + str(percs[1]) + ' (+' + str(percs[2] - percs[1]) + ', -'
                          + str(percs[1] - percs[0]) + ')')

        plt.show()

    return sampler.flatchain


if __name__ == "__main__":
    # MAKE SURE I HAVE ACTIVATED THE three AND iraf27 ENVIRONMENTS!!!
    do_emcee(nwalkers=250, burn=100, steps=1000, printer=0, parfile=None, pri_maxes=None, pri_mins=None)

    # BUCKET: DIVIDE BY ERROR! BEST TO DEFINE ERROR DIFFERENTLY EACH SLICE, BUT FOR SMALL DISKS (e.g. NGC3258), DOESN'T
    # MATTER, AND CAN JUST USE THE SAME ELLIPTICAL REGION IN EACH SLICE, NOT CONTAINING ANY LINE EMISSION.

    # TO DO EVENTUALLY: ADD IN MGE STEP TO ESTIMATE v(R) (or M(R))
    # TO DO NOW!!! TWEAK MODEL TO ONLY RUN ON SMALL SUBSECTION OF CUBE!
