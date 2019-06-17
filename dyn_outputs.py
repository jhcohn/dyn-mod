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


def outcorner(parfile, walkers, burn, steps, direc, ndim):
    params, priors = dm.par_dicts(parfile, q=False)  # get dicts of params and file names from parameter file
    free_p = {}
    for key in priors:  # for each parameter
        free_p[key] = params[key]
    flatchain = direc + str(walkers) + '_' + str(burn) + '_' + str(steps) + '_fullchain.pkl'
    with open(flatchain, 'rb') as pk:
        u = pickle._Unpickler(pk)
        u.encoding = 'latin1'
        fchain = u.load()
    samples = fchain.reshape((-1, ndim))  # .chain[:, 50:, :]
    print(samples.shape, 'here')
    fig = corner.corner(samples, labels=[v for v in free_p.keys()], truths=[v for v in free_p.values()])  # truths here is just input value
    plt.show()


def temp_out(direct, clip, end, pfile_true, init_guess):

    params, priors = dm.par_dicts(init_guess, q=False)  # get dicts of params and file names from parameter file

    # BUCKET NEED TO FIND OUT HOW TO AUTOMATE GETTING THIS ORDER RIGHT!
    pars = ['sig1', 'mbh', 'f', 'PAdisk', 'yloc', 'xloc', 'mu', 'sig0', 'vsys', 'r0', 'ml_ratio', 'inc']
    ax_lab = ['km/s', r'$\log_{10}$(M$_{\odot}$)', 'unitless', 'deg', 'pixels', 'xloc', 'pc', 'km/s', 'km/s', 'pc',
              r'M$_{\odot}$/L$_{\odot}$', 'deg']

    with open(direct, 'rb') as pk:
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

        '''  #
        if pars[par] == 'mbh':
            plt.hist(np.log10(chain), 100, color="k", histtype="step")  # axes[i]
            percs = np.percentile(np.log10(chain), [16., 50., 84.])
            threepercs = np.percentile(np.log10(chain), [0.15, 50., 99.85])  # 3sigma
            plt.axvline(np.log10(vax_init), ls='--', color='r')
            plt.axvline(np.log10(vax), ls='-', color='k')
        else:
        # '''  #
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


def cvg(fullchain, clip, end):

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

    #N = np.exp(np.linspace(np.log(100), np.log(10000), 20)).astype(int)
    N = np.exp(np.linspace(np.log(100), np.log(len(chain1[0, :, 0])), 10)).astype(int)
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
        if i == 0:
            lab = "New"
            # lab = "G\&W 2010"
        else:
            lab = None
        plt.loglog(N, new[:,i], 'b-', label=lab)  # N = 50*tau_n50, because tau_n50 = N / 50
    plt.loglog(N, N / 50., 'k--', label=r"$\tau = N/50$")
    plt.xlabel("number of samples, $N$")
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

    # '''  #
    direct = '/Users/jonathancohn/Documents/dyn_mod/cluster_out/' + '250_0_1500tempchain.pkl'
    pf = base + 'param_files/ngc_3258inex_params.txt'  #     pf = base + 'param_files/ngc_3258inex2_params.txt'
    r = gelman_rubin(direc=direct, clip=0, end=1389)
    print(r)
    cvg(direct, clip=0, end=1389)
    temp_out(direct, clip=1000, end=1389, pfile_true=base + 'param_files/ngc_3258bl_params.txt', init_guess=pf)
    # '''  #

    '''  #
    direct = '/Users/jonathancohn/Documents/dyn_mod/cluster_out/' + '250_0_1000tempchain.pkl'
    pf = base + 'param_files/ngc_3258bl_params.txt'  #     pf = base + 'param_files/ngc_3258inex2_params.txt'
    r = gelman_rubin(direc=direct, clip=0, end=192)
    print(r)
    cvg(direct, clip=0, end=192)  # turnover first appears at step 181
    temp_out(direct, clip=0, end=192, pfile_true=base + 'param_files/ngc_3258bl_params.txt', init_guess=pf)
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
