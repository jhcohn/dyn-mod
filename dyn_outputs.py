import numpy as np
from astropy.io import fits
from scipy import integrate
import matplotlib.pyplot as plt
import pickle

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


def output(parfile, walkers, burn, steps, direc):
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
        chains = direc + str(walkers) + '_' + str(burn) + '_' +str(steps) + '_flatchain_' + pars[par]  + '.pkl'
        with open(chains, 'rb') as pk:
            u = pickle._Unpickler(pk)
            u.encoding = 'latin1'
            chain = u.load()

            if pars[par] == 'mbh':
                plt.hist(np.log10(chain), 100, color="k", histtype="step")  # axes[i]
                percs = np.percentile(np.log10(chain), [16., 50., 84.])
                plt.axvline(np.log10(params[pars[par]]), ls='-', color='k')
            else:
                plt.hist(chain, 100, color="k", histtype="step")  # axes[i]
                percs = np.percentile(chain, [16., 50., 84.])
                print(params[pars[par]])
                plt.axvline(params[pars[par]], ls='-', color='k')

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


def outcorner(parfile, walkers, burn, steps, direc):
    params, priors = dm.par_dicts(parfile, q=False)  # get dicts of params and file names from parameter file
    free_p = {}
    for key in priors:  # for each parameter
        free_p[key] = params[key]
    flatchain = direc + str(walkers) + '_' + str(burn) + '_' + str(steps) + '_full_flatchain.pkl'
    with open(flatchain, 'rb') as pk:
        u = pickle._Unpickler(pk)
        u.encoding = 'latin1'
        fchain = u.load()
    samples = fchain.reshape((-1, ndim))  # .chain[:, 50:, :]
    fig = corner.corner(samples, labels=free_p.keys(), truths=free_p.values())  # truths here is just input value
    plt.show()


def gelman_rubin(walkers, burn, steps, ndim, base):
    # http://joergdietrich.github.io/emcee-convergence.html
    # http://www.stat.columbia.edu/~gelman/research/published/itsim.pdf

    chains = base + str(walkers) + '_' + str(burn) + '_' + str(steps) + '_fullchain.pkl'

    with open(base + chains, 'rb') as pk:
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


if __name__ == "__main__":

    nwalkers = 150
    burn = 50
    nsteps = 100

    base = '/Users/jonathancohn/Documents/dyn_mod/'
    pf = base + 'param_files/ngc_3258_params.txt'
    direct = '/Users/jonathancohn/Documents/dyn_mod/emcee_out/'

    output(pf, nwalkers, burn, nsteps, direct)

    print(gelman_rubin(nwalkers, burn, nsteps, 1, direct))
