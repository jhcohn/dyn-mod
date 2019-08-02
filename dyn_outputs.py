import numpy as np
from astropy.io import fits
from scipy import integrate
import matplotlib.pyplot as plt
import pickle
import corner

import dyn_model as dm


def blockshaped(arr, nrow, ncol):  # CONFIRMED
    h, w = arr.shape
    return arr.reshape(h // nrow, nrow, -1, ncol).swapaxes(1, 2).reshape(-1, nrow, ncol)


def rebin(data, n):
    rebinned = []
    for z in range(len(data)):
        subarrays = blockshaped(data[z, :, :], n, n)  # bin the data in groups of nxn (4x4) pixels
        # each pixel in the new, rebinned data cube is the mean of each 4x4 set of original pixels
        # reshaped = np.mean(np.mean(subarrays, axis=-1), axis=-1).reshape((int(len(data[0]) / 4.),
        #                                                                   int(len(data[0][0]) / 4.)))
        reshaped = n**2 * np.mean(np.mean(subarrays, axis=-1), axis=-1).reshape((int(len(data[0]) / n),
                                                                                 int(len(data[0][0]) / n)))
        rebinned.append(reshaped)
    print('rebinned')
    return np.asarray(rebinned)


def compare(data, model, z_ax, inds_to_try2, v_sys, n):
    data_4 = rebin(data, n)
    ap_4 = rebin(model, n)

    for i in range(len(inds_to_try2)):
        print(inds_to_try2[i][0], inds_to_try2[i][1])
        plt.plot(z_ax, ap_4[:, inds_to_try2[i][1], inds_to_try2[i][0]], 'r+', label=r'Model')  # r-
        plt.plot(z_ax, data_4[:, inds_to_try2[i][1], inds_to_try2[i][0]], 'b+', label=r'Data')  # b:
        plt.axvline(x=v_sys, color='k', label=r'v$_{\text{sys}}$')
        # plt.title(str(inds_to_try2[i][0]) + ', ' + str(inds_to_try2[i][1]))  # ('no x,y offset')
        plt.legend()
        plt.xlabel(r'Frequency [GHz]')
        plt.ylabel(r'Flux Density [Jy/beam]')
        plt.show()
        plt.close()


def sig_prof(pfiles, R, labs):

    ls = ['k-', 'b--', 'r:', 'g-.']
    for p in range(len(pfiles)):
        params1, priors1 = dm.par_dicts(pfiles[p], q=False)  # get dicts of params and file names from parameter file
        print(params1['sig0'], params1['r0'], params1['mu'], params1['sig1'], params1['s_type'])
        sig = dm.get_sig(R, params1['sig0'], params1['r0'], params1['mu'], params1['sig1'])[params1['s_type']]

        plt.plot(R, sig, ls[p], label=labs[p])
    plt.xlabel(r'R [pc]')
    plt.ylabel(r'$\sigma(R)$ [km/s]')
    plt.xscale('log')
    plt.legend()
    plt.show()


def output_clipped(parfile, nwalkers, burn, nsteps, direct, clip=100, pfile_true=None, init_guess=None):
    """
    Print out interesting outputs from a given emcee run

    :param parfile:
    :param walkers:
    :param burn:
    :param steps:
    :param direc:
    :return:
    """

    params, priors = dm.par_dicts(parfile, q=False)  # get dicts of params and file names from parameter file

    # BUCKET NEED TO FIND OUT HOW TO AUTOMATE GETTING THIS ORDER RIGHT!
    pars = ['sig1', 'mbh', 'f', 'PAdisk', 'yloc', 'xloc', 'mu', 'sig0', 'vsys', 'r0', 'ml_ratio', 'inc']
    ax_lab = ['km/s', r'$\log_{10}$(M$_{\odot}$)', 'unitless', 'deg', 'pixels', 'xloc', 'pc', 'km/s', 'km/s', 'pc',
              r'M$_{\odot}$/L$_{\odot}$', 'deg']
    chains = direct + str(nwalkers) + '_' + str(burn) + '_' + str(nsteps) + '_fullchain.pkl'
    # chains = '/Users/jonathancohn/Documents/dyn_mod/chain.pkl'

    with open(chains, 'rb') as pk:
        u = pickle._Unpickler(pk)
        u.encoding = 'latin1'
        chain1 = u.load()
    print(chain1[:, clip:, 0].shape)
    for par in range(len(pars)):
        chain = np.ndarray.flatten(chain1[:, clip:, par])  # CLIP, THEN FLATTEN

        if init_guess is not None:
            par_init, pri_init = dm.par_dicts(init_guess, q=False)
            vax_init = par_init[pars[par]]
        else:
            vax_init = params[pars[par]]

        if pfile_true is not None:
            par_true, pri_true = dm.par_dicts(pfile_true, q=False)
            vax = par_true[pars[par]]
        else:
            vax = params[pars[par]]

        if pars[par] == 'mbh':
            plt.hist(np.log10(chain), 100, color="k", histtype="step")  # axes[i]
            percs = np.percentile(np.log10(chain), [16., 50., 84.])
            threepercs = np.percentile(np.log10(chain), [0.15, 50., 99.85])  # 3sigma
            plt.axvline(np.log10(vax_init), ls='--', color='r')
            plt.axvline(np.log10(vax), ls='-', color='k')
        else:
            plt.hist(chain, 100, color="k", histtype="step")  # axes[i]
            percs = np.percentile(chain, [16., 50., 84.])
            threepercs = np.percentile(chain, [0.15, 50., 99.85])  # 3sigma
            print(params[pars[par]])
            plt.axvline(vax_init, ls='--', color='r')
            plt.axvline(vax, ls='-', color='k')

        plt.axvline(percs[1], ls='-', color='b')  # axes[i]
        plt.axvline(percs[0], ls='--', color='b')  # axes[i]
        plt.axvline(percs[2], ls='--', color='b')  #
        plt.tick_params('both', labelsize=16)
        plt.xlabel(ax_lab[par])
        # plt.title("Dimension {0:d}".format(i))
        plt.title(pars[par] + ': ' + str(round(percs[1], 4)) + ' (+'
                  + str(round(percs[2] - percs[1], 4)) + ', -'
                  + str(round(percs[1] - percs[0], 4)) + ')', fontsize=16)
        plt.show()


def output(parfile, walkers, burn, steps, direc, pfile_true=None):
    """
    Print out interesting outputs from a given emcee run

    :param parfile:
    :param walkers:
    :param burn:
    :param steps:
    :param direc:
    :return:
    """

    params, priors = dm.par_dicts(parfile, q=False)  # get dicts of params and file names from parameter file

    ax_lab = [r'$\log_{10}$(M$_{\odot}$)', 'deg', 'deg', 'pixels', 'pixels', 'km/s', 'km/s', 'km/s', 'pc', 'pc',
              'unitless', r'M$_{\odot}$/L$_{\odot}$']
    pars = ['mbh', 'inc', 'PAdisk', 'xloc', 'yloc', 'vsys', 'sig1', 'sig0', 'mu', 'r0', 'f', 'ml_ratio']

    for par in range(len(pars)):
        # chains = 'emcee_out/flatchain_' + pars[par] + '_' + str(walkers) + '_' + str(burn) + '_' +str(steps) + '.pkl'
        chains = direc + str(walkers) + '_' + str(burn) + '_' +str(steps) + '_flatchain_' + pars[par] + '.pkl'

        with open(chains, 'rb') as pk:
            u = pickle._Unpickler(pk)
            u.encoding = 'latin1'
            chain = u.load()

            if pfile_true is not None:
                par_true, pri_true = dm.par_dicts(pfile_true, q=False)  # get dicts of params and file names from parameter file
                vax = par_true[pars[par]]
            else:
                vax = params[pars[par]]

            if pars[par] == 'mbh':
                plt.hist(np.log10(chain), 100, color="k", histtype="step")  # axes[i]
                percs = np.percentile(np.log10(chain), [16., 50., 84.])
                threepercs = np.percentile(np.log10(chain), [0.15, 50., 99.85])  # 3sigma
                plt.axvline(np.log10(vax), ls='-', color='k')
            else:
                plt.hist(chain, 100, color="k", histtype="step")  # axes[i]
                percs = np.percentile(chain, [16., 50., 84.])
                threepercs = np.percentile(chain, [0.15, 50., 99.85])  # 3sigma
                print(params[pars[par]])
                plt.axvline(vax, ls='-', color='k')

            plt.axvline(percs[1], ls='-', color='b')  # axes[i]
            plt.axvline(percs[0], ls='--', color='b')  # axes[i]
            plt.axvline(percs[2], ls='--', color='b')  #
            plt.tick_params('both', labelsize=16)
            plt.xlabel(ax_lab[par])
            # plt.title("Dimension {0:d}".format(i))
            plt.title(pars[par] + ': ' + str(round(percs[1],4)) + ' (+'
                                     + str(round(percs[2] - percs[1], 4)) + ', -'
                                     + str(round(percs[1] - percs[0], 4)) + ')', fontsize=16)
            plt.show()


def outcorner(parfile, fullchain, burn):  # direc, walkers, burn, steps,
    params, priors = dm.par_dicts(parfile, q=False)  # get dicts of params and file names from parameter file
    free_p = {}
    for key in priors:  # for each parameter
        free_p[key] = params[key]
    # order = mbh, f, PAdisk, yloc, xloc, sig0, vsys, ml_ratio, inc
    # fullchain = direc + str(walkers) + '_' + str(burn) + '_' + str(steps) + '_fullchain.pkl'
    with open(fullchain, 'rb') as pk:
        u = pickle._Unpickler(pk)
        u.encoding = 'latin1'
        fchain = u.load()
    print(fchain.shape)
    fchain = fchain[:, burn:, :]
    samples = fchain.reshape((-1, len(fchain[0, 0, :])))  # (-1, ndim)  # .chain[:, 50:, :]
    print(samples.shape, 'here')
    from matplotlib import rcParams
    rcParams["font.size"] = 4
    fig = corner.corner(samples, bins=20)  # labels=[v for v in free_p.keys()], truths=[v for v in free_p.values()], show_titles=True, )  # truths here is B1 value
    plt.show()


def temp_out(direct, clip, end, pfile_true, init_guess):

    params, priors = dm.par_dicts(init_guess, q=False)  # get dicts of params and file names from parameter file

    # BUCKET NEED TO FIND OUT HOW TO AUTOMATE GETTING THIS ORDER RIGHT!
    pars = ['sig1', 'mbh', 'f', 'PAdisk', 'yloc', 'xloc', 'mu', 'sig0', 'vsys', 'r0', 'ml_ratio', 'inc']
    ax_lab = ['km/s', r'$\log_{10}$(M$_{\odot}$)', 'unitless', 'deg', 'pixels', 'pixels', 'pc', 'km/s', 'km/s', 'pc',
              r'M$_{\odot}$/L$_{\odot}$', 'deg']

    with open(direct, 'rb') as pk:
        u = pickle._Unpickler(pk)
        u.encoding = 'latin1'
        chain1 = u.load()
    print(chain1.shape)
    print(chain1[:, clip:end, 0].shape)

    # ADDED
    fig, axes = plt.subplots(2, 3)  # 3 rows, 4 cols of subplots; because there are 12 free params
    i = 0
    row = 0
    col = 0
    # /ADDED
    for par in range(len(pars)):
        chain = np.ndarray.flatten(chain1[:, clip:end, par])  # CLIP, THEN FLATTEN

        if init_guess is not None:
            par_init, pri_init = dm.par_dicts(init_guess, q=False)
            vax_init = par_init[pars[par]]
        else:
            vax_init = params[pars[par]]

        if pfile_true is not None:
            par_true, pri_true = dm.par_dicts(pfile_true, q=False)
            vax = par_true[pars[par]]
        else:
            vax = params[pars[par]]

        # '''  #
        # ADDED
        i += 1
        print(i, row, col)
        if i == 10:  #  if i == 4:  # if i == 4 or i == 8:
            row += 1
            col = 0
        if 7 <= i <= 12:  # if 0 <= i <= 6:
            if pars[par] == 'mbh':
                axes[row, col].hist(np.log10(chain), 100, color="k", histtype="step")  # axes[i]
                percs = np.percentile(np.log10(chain), [16., 50., 84.])
                threepercs = np.percentile(np.log10(chain), [0.15, 50., 99.85])  # 3sigma
                axes[row, col].axvline(np.log10(vax_init), ls='--', color='r')
                axes[row, col].axvline(np.log10(vax), ls='-', color='k')
            else:
                axes[row, col].hist(chain, 100, color="k", histtype="step")  # axes[i]
                percs = np.percentile(chain, [16., 50., 84.])
                threepercs = np.percentile(chain, [0.15, 50., 99.85])  # 3sigma
                axes[row, col].axvline(vax_init, ls='--', color='r')
                axes[row, col].axvline(vax, ls='-', color='k')
            # print([key for key in params], percs)
            # print(threepercs)
            axes[row, col].axvline(percs[1], ls='-')  # axes[i]
            axes[row, col].axvline(percs[0], ls='--')  # axes[i]
            axes[row, col].axvline(percs[2], ls='--')  #
            axes[row, col].tick_params('both', labelsize=8)
            # plt.title("Dimension {0:d}".format(i))
            axes[row, col].set_title(pars[par] + ': ' + str(round(percs[1],4)) + ' (+'
                                     + str(round(percs[2] - percs[1], 4)) + ', -'
                                     + str(round(percs[1] - percs[0], 4)) + ')', fontsize=8)
            axes[row, col].set_xlabel(ax_lab[par], fontsize=8)
            col += 1

    plt.tick_params('both', labelsize=8)
    # plt.title("Dimension {0:d}".format(i))
    plt.show()
    plt.close()
    # /ADDED
    '''  #
    
        if pars[par] == 'mbh':
            plt.hist(np.log10(chain), 100, color="k", histtype="step")  # axes[i]
            percs = np.percentile(np.log10(chain), [16., 50., 84.])
            threepercs = np.percentile(np.log10(chain), [0.15, 50., 99.85])  # 3sigma
            plt.axvline(np.log10(vax_init), ls='--', color='r')
            plt.axvline(np.log10(vax), ls='-', color='k')
        else:
            plt.hist(chain, 100, color="k", histtype="step")  # axes[i]
            percs = np.percentile(chain, [16., 50., 84.])
            threepercs = np.percentile(chain, [0.15, 50., 99.85])  # 3sigma
            print(params[pars[par]])
            plt.axvline(vax_init, ls='--', color='r')
            plt.axvline(vax, ls='-', color='k')

        plt.axvline(percs[1], ls='-', color='b')  # axes[i]
        plt.axvline(percs[0], ls='--', color='b')  # axes[i]
        plt.axvline(percs[2], ls='--', color='b')  #
        plt.tick_params('both', labelsize=16)
        plt.xlabel(ax_lab[par])
        # plt.title("Dimension {0:d}".format(i))
        plt.title(pars[par] + ': ' + str(round(percs[1], 4)) + ' (+'
                  + str(round(percs[2] - percs[1], 4)) + ', -'
                  + str(round(percs[1] - percs[0], 4)) + ')', fontsize=16)
        plt.show()
        # '''  #


def gelman_rubin_sarah(mtrace):
    Rhat = {}
    for var in range(len(mtrace[0, 0, :])):
        x = np.array(mtrace[:, :, var].T)
        num_samples = x.shape[1]

        # Calculate between-chain variance
        B = num_samples * np.var(np.mean(x, axis=1), axis=0, ddof=1)
        # Calculate within-chain variance
        W = np.mean(np.var(x, axis=1, ddof=1), axis=0)

        # Estimate of marginal posterior variance
        Vhat = W * (num_samples - 1) / num_samples + B / num_samples

        Rhat[var] = np.sqrt(Vhat / W)

    return Rhat


def gelman_rubin(walkers=100, burn=100, steps=100, ndim=12, base=None, direc=None, clip=None, end=None):
    # http://joergdietrich.github.io/emcee-convergence.html
    # http://www.stat.columbia.edu/~gelman/research/published/itsim.pdf

    if direc is not None:
        chains = direc
        with open(chains, 'rb') as pk:
            u = pickle._Unpickler(pk)
            u.encoding = 'latin1'
            chain = u.load()
        chain = chain[:, clip:end, :]
    else:
        chains = base + str(walkers) + '_' + str(burn) + '_' + str(steps) + '_fullchain.pkl'
        with open(chains, 'rb') as pk:
            u = pickle._Unpickler(pk)
            u.encoding = 'latin1'
            chain = u.load()

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


def auto_window(taus, c):  # Automated windowing procedure following Sokal (1989)
    m = np.arange(len(taus)) < c * taus
    if np.any(m):
        return np.argmin(m)
    return len(taus) - 1


def autocorr_gw2010(y, c=5.0):  # Following the suggestion from Goodman & Weare (2010)
    f = autocorr_func_1d(np.mean(y, axis=0))
    taus = 2.0*np.cumsum(f)-1.0
    window = auto_window(taus, c)
    return taus[window]


def autocorr_new(y, c=5.0):
    f = np.zeros(y.shape[1])
    for yy in y:
        f += autocorr_func_1d(yy)  # norm=False (having norm=True causing nan for some, bc dividing by 0 (acf[0] = 0.)!
    f /= len(y)
    taus = 2.0*np.cumsum(f)-1.0
    window = auto_window(taus, c)
    return taus[window]


def cvg(fullchain, clip, end, vline=None):

    with open(fullchain, 'rb') as pk:
        u = pickle._Unpickler(pk)
        u.encoding = 'latin1'
        chain1 = u.load()
    print(chain1.shape)
    chain1 = chain1[:, clip:end, :]
    print(chain1.shape)

    #out_chain = fullchain[:-4] + '_resaved.pkl'
    #with open(out_chain, 'wb') as newfile:  # 'wb' because binary format
    #    pickle.dump(chain1, newfile, pickle.HIGHEST_PROTOCOL)
    #print(oop)

    pars = ['sig1', 'mbh', 'f', 'PAdisk', 'yloc', 'xloc', 'mu', 'sig0', 'vsys', 'r0', 'ml_ratio', 'inc']

    N = np.exp(np.linspace(np.log(100), np.log(100000), 20)).astype(int)
    # N = np.exp(np.linspace(np.log(100), np.log(len(chain1[0, :, 0])), 10)).astype(int)
    gw2010 = np.zeros(shape=(len(N), len(chain1[0, 0, :])))
    new = np.zeros(shape=(len(N), len(chain1[0, 0, :])))
    for p in range(len(chain1[0, 0, :])):
        # chain = np.ndarray.flatten(chain1[:, :, p])  # CLIP, THEN FLATTEN
        # for s in range(len(chain1[0, :, p])):
        pchain = chain1[:, :, p]
        print(pchain.shape)
        # Compute the estimators for a few different chain lengths
        for j, n in enumerate(N):
            gw2010[j, p] = autocorr_gw2010(pchain[:, :n])
            new[j, p] = autocorr_new(pchain[:, :n])

    print(gw2010)
    print(new)
    for i in range(len(gw2010[0, :])):
        '''
        if i == 0:
            # lab = "New"
            lab = "G\&W 2010"
        else:
            lab = None
        '''  #
        lab = pars[i]
        if pars[i] == 'sig0' or pars[i] == 'r0' or pars[i] == 'mu':
            plt.loglog(N, gw2010[:, i], 'k-', label=lab)  # 'b-',   # N = 50*tau_n50, because tau_n50 = N / 50
        else:
            plt.loglog(N, gw2010[:,i], label=lab)  # 'b-',   # N = 50*tau_n50, because tau_n50 = N / 50
    plt.loglog(N, N / 50., 'k--', label=r"$\tau = N/50$")
    plt.axvline(x=len(chain1[0, :, 0]), color='r')  # N_steps actually taken in chain
    if vline is not None:
        for x in vline:
            plt.axvline(x=x)#, color='b')
    plt.xlabel("number of steps, $N$")
    plt.ylabel(r"$\tau$ estimates")
    plt.legend(fontsize=14)
    plt.show()


def check_cvg(directory):
    gw = directory + '_gw2010.pkl'
    const_tau = directory + '_Nover50.pkl'

    with open(gw, 'rb') as estimate:
        u = pickle._Unpickler(estimate)
        u.encoding = 'latin1'
        gw2010 = u.load()
    with open(const_tau, 'rb') as estimate:
        u = pickle._Unpickler(estimate)
        u.encoding = 'latin1'
        tau_n50 = u.load()

    for i in range(len(gw2010[0,:])):
        if i == 0:
            lab = "G\&W 2010"
        else:
            lab = None
        plt.loglog(50*tau_n50, gw2010[:,i], 'b-', label=lab)  # N = 50*tau_n50, because tau_n50 = N / 50
    plt.loglog(50*tau_n50, tau_n50, 'k--', label=r"$\tau = N/50$")
    plt.xlabel("number of samples, $N$")
    plt.ylabel(r"$\tau$ estimates")
    plt.legend(fontsize=14)
    plt.show()


def compare_sigs(pfile1, pfile2, q1=True, q2=False):
    p1 = dm.par_dicts(pfile1, q1)
    if q1:
        par1, pri1, qobs1 = p1
    else:
        par1, pri1 = p1

    p2 = dm.par_dicts(pfile2, q2)
    if q2:
        par2, pri2, qobs2 = p2
    else:
        par2, pri2 = p2

    pars = ['sig1', 'mu', 'sig0', 'r0']
    ax_lab = ['km/s', 'pc', 'km/s', 'pc']

    rads = np.logspace(-2, 3, 200)  # pc
    sig1_1 = par1['sig1']
    sig0_1 = par1['sig0']
    mu_1 = par1['mu']
    r0_1 = par1['r0']

    sigma_1 = {'flat': sig0_1, 'gauss': sig1_1 + sig0_1 * np.exp(-(rads - r0_1) ** 2 / (2 * mu_1 ** 2)),
               'exp': sig1_1 + sig0_1 * np.exp(-rads / r0_1)}

    sig1_2 = par2['sig1']
    sig0_2 = par2['sig0']
    mu_2 = par2['mu']
    r0_2 = par2['r0']

    sigma_2 = {'flat': sig0_2, 'gauss': sig1_2 + sig0_2 * np.exp(-(rads - r0_2) ** 2 / (2 * mu_2 ** 2)),
               'exp': sig1_2 + sig0_2 * np.exp(-rads / r0_2)}

    plt.plot(rads, sigma_1[par1['s_type']], 'k--', label=r'Cohn')
    plt.plot(rads, sigma_2[par2['s_type']], 'b:', label=r'Boizelle')
    plt.legend(loc='upper right')
    plt.ylabel(r'$\sigma$ [km/s]')
    plt.xlabel(r'Radius [pc]')
    plt.show()


def plot_all(fullchain, clip, end, pfile_true, init_guess, flatsig=False, save=False, xcl=False):

    params, priors = dm.par_dicts(init_guess, q=False)  # get dicts of params and file names from parameter file

    if xcl and flatsig:
        fig, axes = plt.subplots(2, 4, figsize=(18, 9))  # 3 rows, 3 cols of subplots; because there are 9 free params
        # BUCKET NEED TO FIND OUT HOW TO AUTOMATE GETTING THIS ORDER RIGHT!
        pars = ['mbh', 'f', 'PAdisk', 'yloc', 'sig0', 'vsys', 'ml_ratio', 'inc']
        ax_lab = [r'$\log_{10}$(M$_{\odot}$)', 'unitless', 'deg', 'pixels', 'km/s', 'km/s',
                  r'M$_{\odot}$/L$_{\odot}$', 'deg']
        axes_order = [[0, 0], [0, 2], [1, 0], [1, 2], [1, 3], [1, 1], [0, 1], [0, 3]]
    elif flatsig:
        fig, axes = plt.subplots(3, 3, figsize=(16, 12))  # 3 rows, 3 cols of subplots; because there are 9 free params
        # BUCKET NEED TO FIND OUT HOW TO AUTOMATE GETTING THIS ORDER RIGHT!
        pars = ['mbh', 'f', 'PAdisk', 'yloc', 'xloc', 'sig0', 'vsys', 'ml_ratio', 'inc']
        ax_lab = [r'$\log_{10}$(M$_{\odot}$)', 'unitless', 'deg', 'pixels', 'pixels', 'km/s', 'km/s',
                  r'M$_{\odot}$/L$_{\odot}$', 'deg']
        axes_order = [[0, 0], [0, 2], [1, 1], [2, 1], [2, 0], [2, 2], [1, 2], [0, 1], [1, 0]]
        # mbh, mlratio, inc, padisk ; f, vsys, xloc, yloc ; sig1, sig0, r0, mu
    else:
        fig, axes = plt.subplots(3, 4, figsize=(18, 12))  # 3 rows, 4 cols of subplots; because there are 9 free params
        # BUCKET NEED TO FIND OUT HOW TO AUTOMATE GETTING THIS ORDER RIGHT!
        pars = ['sig1', 'mbh', 'f', 'PAdisk', 'yloc', 'xloc', 'mu', 'sig0', 'vsys', 'r0', 'ml_ratio', 'inc']
        ax_lab = ['km/s', r'$\log_{10}$(M$_{\odot}$)', 'unitless', 'deg', 'pixels', 'pixels', 'pc', 'km/s', 'km/s',
                  'pc', r'M$_{\odot}$/L$_{\odot}$', 'deg']
        axes_order = [[2, 0], [0, 0], [1, 0], [0, 3], [1, 3], [1, 2], [2, 3], [2, 1], [1, 1], [2, 2], [0, 1], [0, 2]]
        # mbh, mlratio, inc, padisk ; f, vsys, xloc, yloc ; sig1, sig0, r0, mu

    with open(fullchain, 'rb') as pk:
        u = pickle._Unpickler(pk)
        u.encoding = 'latin1'
        chain1 = u.load()
    print(chain1.shape)
    print(chain1[:, clip:end, 0].shape)

    for par in range(len(pars)):
        chain = np.ndarray.flatten(chain1[:, clip:end, par])  # CLIP, THEN FLATTEN

        if init_guess is not None:
            par_init, pri_init = dm.par_dicts(init_guess, q=False)
            vax_init = par_init[pars[par]]
        else:
            vax_init = params[pars[par]]

        if pfile_true is not None:
            par_true, pri_true = dm.par_dicts(pfile_true, q=False)
            vax = par_true[pars[par]]
        else:
            vax = params[pars[par]]

        if flatsig:
            with open('/Users/jonathancohn/Documents/dyn_mod/param_files/Ben_A1_errors.txt') as a1:
                for line in a1:
                    cols = line.split()
                    if cols[0] == pars[par]:
                        vax = float(cols[1])
                        vax_width = float(cols[2])

        row, col = axes_order[par]
        fs = 11
        if pars[par] == 'mbh':
            axes[row, col].hist(np.log10(chain), 2000, color="b", histtype="step")  # axes[i]
            percs = np.percentile(np.log10(chain), [16., 50., 84.])
            threepercs = np.percentile(np.log10(chain), [0.15, 50., 99.85])  # 3sigma
            if flatsig:
                axes[row, col].axvline(np.log10(vax), ls='-', color='k')
                print('here', np.log10(vax), np.log10(vax-vax_width), np.log10(vax+vax_width))
                axes[row, col].axvspan(np.log10(vax-vax_width), np.log10(vax+vax_width), hatch='/', color='k',
                                       fill=False, alpha=0.5)
            else:
                axes[row, col].axvline(np.log10(vax), ls='-', color='k')
            # axes[row, col].axvline(np.log10(vax_init), ls='--', color='r')
        else:
            if pars[par] == 'xloc' or pars[par] == 'yloc' or pars[par] == 'vsys':
                bin = 4000
            elif pars[par] == 'PAdisk':
                bin = 2000
            else:
                bin = 200
            axes[row, col].hist(chain, bin, color="b", histtype="step")  # axes[i]
            percs = np.percentile(chain, [16., 50., 84.])
            threepercs = np.percentile(chain, [0.15, 50., 99.85])  # 3sigma
            if flatsig:
                axes[row, col].axvline(vax, ls='-', color='k')
                axes[row, col].axvspan(vax-vax_width, vax+vax_width, hatch='/', color='k', fill=False, alpha=0.5)
            else:
                axes[row, col].axvline(vax, ls='-', color='k')
            # axes[row, col].axvline(vax_init, ls='--', color='r')
        # print([key for key in params], percs)
        # print(threepercs)
        axes[row, col].axvline(percs[1], color='b', ls='--')  # axes[i]
        axes[row, col].axvspan(percs[0], percs[2], color='b', alpha=0.25)
        # axes[row, col].axvline(percs[0], ls='--')  # axes[i]
        # axes[row, col].axvline(percs[2], ls='--')  #
        axes[row, col].tick_params('both', labelsize=fs)
        # plt.title("Dimension {0:d}".format(i))
        axes[row, col].set_title(pars[par] + ': ' + str(round(percs[1], 4)) + ' (+'
                                 + str(round(percs[2] - percs[1], 4)) + ', -'
                                 + str(round(percs[1] - percs[0], 4)) + ')', fontsize=fs)
        axes[row, col].set_xlabel(ax_lab[par], fontsize=fs)
        print(percs[0] - (percs[1] - percs[0]), percs[2] + (percs[2] - percs[1]))
        # '''  #  Comment here to remove xlims
        if pars[par] == 'vsys' or pars[par] == 'yloc':
            axes[row, col].set_xlim(percs[0] - 14 * (percs[1] - percs[0]), percs[2] + 14 * (percs[2] - percs[1]))
        elif pars[par] == 'xloc':
            axes[row, col].set_xlim(percs[0] - 40 * (percs[1] - percs[0]), percs[2] + 40 * (percs[2] - percs[1]))
        elif flatsig and pars[par] == 'sig0':
            axes[row, col].set_xlim(7., 10.)
        elif pars[par] == 'sig0' or pars[par] == 'r0' or pars[par] == 'mu':
            axes[row, col].set_xlim(percs[0] - 2 * (percs[1] - percs[0]), percs[2] + 2 * (percs[2] - percs[1]))
        else:
            axes[row, col].set_xlim(percs[0] - 8 * (percs[1] - percs[0]), percs[2] + 8 * (percs[2] - percs[1]))
        #  '''  #
    # plt.tick_params('both', labelsize=11)
    plt.tight_layout()
    # plt.title("Dimension {0:d}".format(i))
    if save:
        # plt.savefig('/Users/jonathancohn/Documents/dyn_mod/groupmtg/inex5_250_0_4000/inex5_burn' + str(clip) +
        #             '_end' + str(end) + '-all_save.png', dpi=300)
        # plt.savefig('/Users/jonathancohn/Documents/dyn_mod/groupmtg/binex8_250_0_2000/binex8_r500_burn' + str(clip) +
        #             '_end' + str(end) + '-all_save.png', dpi=300)
        plt.savefig('/Users/jonathancohn/Documents/dyn_mod/groupmtg/binexc_500_0_20000/binexc_burn' + str(clip) +
                    '_end' + str(end) + '-all_save.png', dpi=300)
    plt.show()
    plt.close()


def param_changes(fullchain, clips, ends, flatsig=False):
    """
    How does the median and 68% confidence interval of each parameter change over the course of the chain

    :param fullchain:
    :param clips:
    :param ends:
    :return:
    """

    if flatsig:
        # BUCKET NEED TO FIND OUT HOW TO AUTOMATE GETTING THIS ORDER RIGHT!
        pars = ['mbh', 'f', 'PAdisk', 'yloc', 'xloc', 'sig0', 'vsys', 'ml_ratio', 'inc']
    else:
        # BUCKET NEED TO FIND OUT HOW TO AUTOMATE GETTING THIS ORDER RIGHT!
        pars = ['sig1', 'mbh', 'f', 'PAdisk', 'yloc', 'xloc', 'mu', 'sig0', 'vsys', 'r0', 'ml_ratio', 'inc']

    with open(fullchain, 'rb') as pk:
        u = pickle._Unpickler(pk)
        u.encoding = 'latin1'
        chain1 = u.load()
    print(chain1.shape)

    for par in range(len(pars)):
        meds = []
        lows = []
        highs = []
        widths = []
        for i in range(len(ends)):
            clip = clips[i]
            end = ends[i]
            # print(chain1[:, clip:end, par].shape)
            chain = np.ndarray.flatten(chain1[:, clip:end, par])  # CLIP, THEN FLATTEN

            if pars[par] == 'mbh':
                percs = np.percentile(np.log10(chain), [16., 50., 84.])
                threepercs = np.percentile(np.log10(chain), [0.15, 50., 99.85])  # 3sigma
            else:
                percs = np.percentile(chain, [16., 50., 84.])
                threepercs = np.percentile(chain, [0.15, 50., 99.85])  # 3sigma
            meds.append(percs[1])
            lows.append(percs[0])
            highs.append(percs[2])
            widths.append(percs[2] - percs[1])
        print(pars[par], round((np.amax(meds) - np.amin(meds)) / np.median(meds), 4))
        print(pars[par], round(np.median(meds), 4), round(np.amax(meds) - np.amin(meds), 4),
              round(np.amax(highs) - np.amin(highs), 4), round(np.amax(lows) - np.amin(lows), 4),
              round(np.amax(np.asarray(highs) - np.asarray(lows)) - np.amin(np.asarray(highs) - np.asarray(lows)), 4))

    # plt.title("Dimension {0:d}".format(i))


if __name__ == "__main__":

    nwalkers = 250 #250#250  # 250  # 250  # 150  # 250#150
    burn = 0  # 1  # 1  # 1  # 100#50
    nsteps = 301 #500  # 301  # 300  # 100  # 100#100

    base = '/Users/jonathancohn/Documents/dyn_mod/'
    # pf = base + 'param_files/ngc_3258_params.txt'
    # direct = '/Users/jonathancohn/Documents/dyn_mod/emcee_out/'
    # direct = '/Users/jonathancohn/Documents/dyn_mod/cluster_out/' + 'mpi3258bl_'
    # pf = base + 'param_files/ngc_3258bl_params.txt'

    # direct = '/Users/jonathancohn/Documents/dyn_mod/cluster_out/' + 'mpi3258binex_'
    # pf = base + 'param_files/ngc_3258bl_params.txt'
    # direct = '/Users/jonathancohn/Documents/dyn_mod/cluster_out/' + '3258inex_'
    # pf = base + 'param_files/ngc_3258inex_params.txt'

    #direct = '/Users/jonathancohn/Documents/dyn_mod/cluster_out/' + 'mpi3258binex_'
    #pf = base + 'param_files/ngc_3258bl_params.txt'

    # direct = '/Users/jonathancohn/Documents/dyn_mod/cluster_out/' + 'mpi3258binex_'
    # pf = base + 'param_files/ngc_3258bl_params.txt'

    # direct = '/Users/jonathancohn/Documents/dyn_mod/cluster_out/' + 'mpi3258inex2_'
    # pf = base + 'param_files/ngc_3258inex2out_params.txt'  #     pf = base + 'param_files/ngc_3258inex2_params.txt'

    # COMPARE SIGMA_TURB
    '''  #
    # compare_sigs(base + 'param_files/ngc_3258inex5out_params.txt', base + 'param_files/ngc_3258bl_params.txt')
    rads = np.logspace(-2., 3.5, 100)
    sig_prof([base + 'param_files/ngc_3258bl_params.txt', base + 'param_files/ngc_3258inex5out_params.txt',
              base + 'param_files/ngc_3258binex8out_params.txt', base + 'param_files/ngc_3258bl3out_params.txt'],
             rads, ['B1', 'inex', 'binex', 'bl'])
    print(oop)
    # '''  #

    '''  #
    # xcl run!
    direct = '/Users/jonathancohn/Documents/dyn_mod/cluster_out/' + 'xcl_3258binexc_1564617972.15_'
    pf = base + 'param_files/ngc_3258binexc_xcl_params.txt'  # pf = base + 'param_files/ngc_3258inex2_params.txt'
    bl = base + 'param_files/ngc_3258bl_params.txt'
    plot_all(direct + '500_0_1000_fullchain.pkl', clip=500, end=1000, pfile_true=bl, init_guess=pf, flatsig=True, xcl=True,
             save=False)
    print(oop)
    # '''  #

    ''' #
    # not sure (500walk 500steps)
    direct = '/Users/jonathancohn/Documents/dyn_mod/cluster_out/' + 'flat3258binexc_1564523818.43_'
    pf = base + 'param_files/ngc_3258binexc_params.txt'  # pf = base + 'param_files/ngc_3258inex2_params.txt'
    bl = base + 'param_files/ngc_3258bl_params.txt'
    plot_all(direct + '500_0_500_fullchain.pkl', clip=400, end=500, pfile_true=bl, init_guess=pf, flatsig=True, save=False)
    print(oop)
    # ''' #

    # '''  #
    # binexc run and binexc fixed ellipse run
    direct = '/Users/jonathancohn/Documents/dyn_mod/cluster_out/' + 'flat3258binexc_'
    direct_ok = '/Users/jonathancohn/Documents/dyn_mod/cluster_out/' + 'old_flat3258binexc_'
    pf = base + 'param_files/ngc_3258binexc_params.txt'  #     pf = base + 'param_files/ngc_3258inex2_params.txt'
    bl = base + 'param_files/ngc_3258bl_params.txt'
    directr = direct + '1564759210.8_500_0_20000_fullchain.pkl'  # '500_0_20000tempchain.pkl'
    cvg(directr, clip=0, end=-1, vline=[14750])
    #temp_out(directr, clip=0, end=-1, pfile_true=base + 'param_files/ngc_3258bl_params.txt', init_guess=pf)
    plot_all(directr, clip=2000, end=20000, pfile_true=bl, init_guess=pf, flatsig=True, save=False)
    plot_all(direct_ok + '500_0_20000tempchain.pkl', clip=2000, end=20000, pfile_true=bl, init_guess=pf, flatsig=True, save=False)

    # ends = [700, 800, 870, 900, 1000, 1500, 2000]  # 500, 600,
    # param_changes(direct + '500_0_20000_fullchain.pkl', clips=np.asarray(ends) - 500, ends=ends)
    # for end in [7500, 10000, 15000, 18000, 20001]:
    #for end in [200, 300, 400, 500, 1000]:
    #    plot_all(direct + '500_0_20000_fullchain.pkl', clip=end - 100, end=end, pfile_true=bl, init_guess=pf, flatsig=True,
    #             save=False)  # clip=4000, clip=end-5000
    plot_all(direct_ok + '500_0_20000_fullchain.pkl', clip=0, end=4000, pfile_true=bl, init_guess=pf, flatsig=True,
             save=False)  # clip=4000, end=20001
    outcorner(pf, direct + '500_0_20000_fullchain.pkl', 19950)
    cvg(direct + '500_0_20000_fullchain.pkl', clip=0, end=-1, vline=[15000])
    # output_clipped(pf, 500, 0, 20000, direct, clip=1600, pfile_true=bl, init_guess=pf)
    r = gelman_rubin(direc=direct + '500_0_20000_fullchain.pkl', clip=0, end=-1)
    print(r)
    # '''  #

    '''  #
    direct = '/Users/jonathancohn/Documents/dyn_mod/cluster_out/' + 'cvg3258binex_'
    pf = base + 'param_files/ngc_3258binex_params.txt'  #     pf = base + 'param_files/ngc_3258inex2_params.txt'
    bl = base + 'param_files/ngc_3258bl_params.txt'
    ends = [700, 800, 870, 900, 1000, 1500, 2000]  # 500, 600,
    param_changes(direct + '250_0_2000_fullchain.pkl', clips=np.asarray(ends) - 500, ends=ends)
    print(oop)
    # for end in [500, 600, 700, 800, 870, 900, 1000, 1500, 2000]:
        # plot_all(direct + '250_0_2000_fullchain.pkl', clip=400, end=end, pfile_true=bl, init_guess=pf, save=False)  # end-500
    cvg(direct + '250_0_2000_fullchain.pkl', clip=0, end=-1)
    plot_all(direct + '250_0_2000_fullchain.pkl', clip=1600, end=2000, pfile_true=bl, init_guess=pf)
    output_clipped(pf, 250, 0, 2000, direct, clip=1600, pfile_true=bl, init_guess=pf)
    r = gelman_rubin(direc=direct + '250_0_2000_fullchain.pkl', clip=0, end=-1)
    print(r)
    # '''  #

    '''  #
    direct = '/Users/jonathancohn/Documents/dyn_mod/cluster_out/' + 'cvg3258bl_'
    bl = base + 'param_files/ngc_3258bl_params.txt'
    pf = base + 'param_files/ngc_3258bl_params.txt'  #     pf = base + 'param_files/ngc_3258inex2_params.txt'
    outcorner(pf, direct + '250_0_1501_fullchain.pkl', 1000)
    plot_all(direct + '250_0_1501_fullchain.pkl', clip=1000, end=1501, pfile_true=bl, init_guess=pf)
    cvg(direct + '250_0_1501_fullchain.pkl', clip=0, end=1501)
    # output_clipped(pf, 250, 0, 1501, direct, clip=1000, pfile_true=pf, init_guess=None)
    r = gelman_rubin(direc=direct + '250_0_1501_fullchain.pkl', clip=0, end=-1)
    print(r)
    # '''  #

    '''  #
    direct = '/Users/jonathancohn/Documents/dyn_mod/cluster_out/' + 'qcvg3258inex_'
    bl = base + 'param_files/ngc_3258bl_params.txt'
    pf = base + 'param_files/ngc_3258inex_params.txt'  #     pf = base + 'param_files/ngc_3258inex2_params.txt'
    # plot_all(direct + '250_0_4000_fullchain.pkl', clip=500, end=1000, pfile_true=bl, init_guess=pf, save=True)
    outcorner(pf, direct + '250_0_4000_fullchain.pkl', 3900)
    plot_all(direct + '250_0_4000_fullchain.pkl', clip=3000, end=4000, pfile_true=bl, init_guess=pf)
    cvg(direct + '250_0_4000_fullchain.pkl', clip=0, end=4000)
    # output_clipped(pf, 250, 0, 4000, direct, clip=3000, pfile_true=base + 'param_files/ngc_3258bl_params.txt', init_guess=pf)
    r = gelman_rubin(direc=direct + '250_0_4000_fullchain.pkl', clip=0, end=-1)
    print(r)
    # '''  #

    '''  #
    direct = '/Users/jonathancohn/Documents/dyn_mod/cluster_out/' + '250_0_2000tempchain.pkl'
    pf = base + 'param_files/ngc_3258inex_params.txt'  #     pf = base + 'param_files/ngc_3258inex2_params.txt'
    temp_out(direct, clip=1400, end=1820, pfile_true=base + 'param_files/ngc_3258bl_params.txt', init_guess=pf)
    r = gelman_rubin(direc=direct, clip=0, end=1820)
    print(r)
    cvg(direct, clip=0, end=1820)
    # '''  #

    '''  #
    direct = '/Users/jonathancohn/Documents/dyn_mod/cluster_out/' + '250_0_1500tempchain.pkl'
    pf = base + 'param_files/ngc_3258inex_params.txt'  #     pf = base + 'param_files/ngc_3258inex2_params.txt'
    temp_out(direct, clip=1000, end=1380, pfile_true=base + 'param_files/ngc_3258bl_params.txt', init_guess=pf)
    r = gelman_rubin(direc=direct, clip=0, end=1380)
    print(r)
    cvg(direct, clip=0, end=1380)
    # '''  #

    '''  #
    direct = '/Users/jonathancohn/Documents/dyn_mod/cluster_out/' + '250_0_1000tempchain.pkl'
    pf = base + 'param_files/ngc_3258bl_params.txt'  #     pf = base + 'param_files/ngc_3258inex2_params.txt'
    r = gelman_rubin(direc=direct, clip=0, end=180)
    print(r)
    cvg(direct, clip=0, end=180)  # turnover first appears at step 181
    temp_out(direct, clip=0, end=180, pfile_true=base + 'param_files/ngc_3258bl_params.txt', init_guess=pf)
    check_cvg(direct + str(nwalkers) + '_' + str(burn) + '_' + str(nsteps))
    # '''  #

    #outcorner(pf, nwalkers, burn, nsteps, direct, ndim=12)

    #output(pf, nwalkers, burn, nsteps, direct, pfile_true=base + 'param_files/ngc_3258bl_params.txt')

    # output_clipped(pf, nwalkers, burn, nsteps, direct, clip=300, pfile_true=base + 'param_files/ngc_3258bl_params.txt')
    output_clipped(pf, nwalkers, burn, nsteps, direct, clip=200, pfile_true=base + 'param_files/ngc_3258bl_params.txt',
                   init_guess=pf)

    # output(pf, nwalkers, burn, nsteps, direct, pfile_true=base + 'param_files/ngc_3258bl_params.txt')

    print(gelman_rubin(nwalkers, burn, nsteps, 1, direct))

    chains = direct + str(nwalkers) + '_' + str(burn) + '_' + str(nsteps) + '_fullchain.pkl'
    with open(chains, 'rb') as pk:
        u = pickle._Unpickler(pk)
        u.encoding = 'latin1'
        chain = u.load()
    print(gelman_rubin_sarah(chain))

    for i in range(len(chain[0,0,:])):
        import emcee
        tau = emcee.autocorr.integrated_time(chain[:,:,i])
        print(tau)
