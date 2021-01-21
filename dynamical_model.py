import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import patches
from scipy import integrate, signal, interpolate, misc
from scipy.ndimage import filters, interpolation
import time
from astropy import convolution
import argparse
from pathlib import Path
from astropy.modeling.models import Box2D
from astropy.nddata.utils import block_reduce
from astropy.modeling.models import Ellipse2D
import sys
# sys.path.insert(0, '/Users/jonathancohn/Documents/jam/')  # lets me import file from different folder/path
import mge_vcirc_mine as mvm  # import mge_vcirc code
from scipy.interpolate import UnivariateSpline as unsp


def projected_distance(ra1, ra2, dec1, dec2, dist1, dist2):
    """

    :param ra1: RA position of object 1 [deg]
    :param ra2: RA position of object 2 [deg]
    :param dec1: DEC position of object 1 [deg]
    :param dec2: DEC position of object 1 [deg]
    :param dist1: Distance to object 1 [pc]
    :param dist2: Distance to object 2 [pc]
    :return: angle of separation [rad], angle of separation [deg], distances of separation [pc] (at dist of each object)
    """

    ra1 = np.deg2rad(ra1)
    ra2 = np.deg2rad(ra2)
    dec1 = np.deg2rad(dec1)
    dec2 = np.deg2rad(dec2)

    sep_rad = np.arccos(np.sin(dec1) * np.sin(dec2) + np.cos(dec1) * np.cos(dec2) * np.cos(ra1 - ra2))  # radians
    sep_deg = np.rad2deg(np.arccos(np.sin(dec1) * np.sin(dec2) + np.cos(dec1) * np.cos(dec2) * np.cos(ra1 - ra2)))

    d_sep1 = dist1 * sep_rad  # pc
    d_sep2 = dist2 * sep_rad  # pc

    return sep_rad, sep_deg, d_sep1, d_sep2


def mge_sum(mge, pc_per_ac):
    """
    Calculate total Luminosity of the MGE

    :param mge: MGE file, containing columns: j [component number], I [Lsol,H/pc^2], sigma[arcsec], q[unitless]
    :param pc_per_ac: parsec per arcsec scaling
    :return: the total luminosity from integrating the MGE
    """

    intensities = []
    sigmas = []
    qs = []
    with open(mge, 'r') as mfile:
        for line in mfile:
            if not line.startswith('#'):
                cols = line.split()
                intensities.append(float(cols[1]))  # cols[0] = component number, cols[1] = intensity[Lsol,H/pc^2]
                sigmas.append(float(cols[2]))  # cols[2] = sigma[arcsec]
                qs.append(float(cols[3]))  # cols[3] = qObs[unitless]

    sum = 0
    for i in range(len(intensities)):
        # volume under 2D gaussian function = 2 pi A sigma_x sigma_y
        sum += 2 * np.pi * intensities[i] * qs[i] * (pc_per_ac * sigmas[i]) ** 2  # convert sigma to pc
    print(sum)  # Lsol,H

    return sum


def mbh_relation_full(mbhs, mbh_errs, lum_ks, lum_errs, masses, mass_errs, sigmas, sigma_errs, galaxies, fmts, ml_locs,
                      msig_locs, mm_locs, savefig=None):
    """
    Calculate MBH relations

    :param mbhs: array of black hole mass measurements
    :param mbh_errs: array of uncertainties on the black hole mass measurements [[low_0, hi_0], [low_1, high_1], ...]
    :param lum_ks: array (same length as mbhs) of luminosities (K-band)
    :param lum_errs: array of uncertainties on the luminosities (K-band) [[low_0, hi_0], [low_1, high_1], ...]
    :param masses: array (same length as mbhs) of masses (H-band luminosity * M/L_H)
    :param mass_errs: array of uncertainties on the masses [[low_0, hi_0], [low_1, high_1], ...]
    :param sigmas: array (same length as mbhs) of sigmas (assuming full galaxies = bulge)
    :param sigma_errs: array of uncertainties on the sigmas [[low_0, hi_0], [low_1, high_1], ...]
    :param: galaxies: array (same length as mbhs) of galaxy labels, eg ['UGC 2698', 'NGC 1277', 'NGC 1271', 'MRK 1216']
    :param: fmts: array (same length as mbhs) of colors and symbols for each galaxy, eg ['bD', 'rs', 'rs', 'rs']
    :param: ml_locs: array (same length as mbhs) of [x,y] text locations on M-L for each galaxy, eg [[1.2e11, 5e9], ...]
    :param: msig_locs: same as ml_locs, but for text locations on M-sigma
    :param: mm_locs: same as ml_locs, but for text locations on M-M
    :param: savefig: file save name for .png image
    :return:
    """

    # initialize figure
    fig, ax = plt.subplots(3, 1, figsize=(10, 20))  # subplots(rows, cols); ax [0,1,2] -> [M-sig, M-L, M-M]
    plt.subplots_adjust(hspace=0.125)

    # SET UP M-SIG PANEL, RELATIONS
    sig_x = np.logspace(1.7, 2.7)  # ~50 - ~500 km/s
    kormendy_and_ho = 0.310e9 * (sig_x / 200) ** 4.38  # 0.309e9
    saglia = 10 ** (4.868 * np.log10(sig_x) - 2.827)  # log MBH = a * log sigma + ZP (CorePowerEClassPC) -> scatter=0.38
    # CorePowerEClassPC = Core & power-law ellipticals, classical bulges, & classical component (Saglia+16 Table 8)
    # Saglia relations, scatter: pg54-55 https://arxiv.org/pdf/1304.7762.pdf
    vdbosch = 10 ** (-4.00 + 5.35 * np.log10(sig_x))
    mcconnell_and_ma = 10 ** (8.32 + 5.64 * np.log10(sig_x / 200.))

    # relations + intrinsic scatter
    ax[0].fill_between(sig_x, saglia * (10 ** 0.38), saglia * (10 ** -0.38), color='k', alpha=0.2)
    ax[0].plot(sig_x, saglia, 'k-', label='Saglia et al. (2016)')  # (CorePowerEClassPC) -> scatter=0.38
    ax[0].plot(sig_x, kormendy_and_ho, 'g--', label='Kormendy \& Ho (2013)')
    ax[0].plot(sig_x, vdbosch, 'm-.', label='van den Bosch (2016)')  # scatter 0.49 dex
    ax[0].plot(sig_x, mcconnell_and_ma, color='darkorange', linestyle=':', label='McConnell \& Ma (2013)')

    # SET UP M-LUM PANEL, RELATIONS
    lum_x = np.logspace(7., 13.)
    kormendy_and_ho = 0.544e9 * (lum_x / 1e11) ** 1.22  # 0.542, 1.21
    lasker = 10 ** (8.56 + 0.75 * (np.log10(lum_x) - 11.))  # intrinsic scatter = 0.46

    # include relations + intrinsic scatter
    ax[1].plot(lum_x, lasker, 'k-', label='Lasker et al. (2014)')  # plot the lasker relation
    ax[1].fill_between(lum_x, lasker * (10 ** .46), lasker * (10 ** -.46), color='k', alpha=0.2)
    ax[1].plot(lum_x, kormendy_and_ho, 'g--', label='Kormendy \& Ho (2013)')  # plot the KH13 relation
    ax[1].fill_between(lum_x, kormendy_and_ho * (10 ** (-0.33)), kormendy_and_ho * (10 ** 0.33), color='g', alpha=0.2)
    # 10^(.4*.31)-1 (https://faculty.virginia.edu/skrutskie/astr3130.s16/notes/astr3130_lec12.pdf)

    # SET UP M-M PANEL, RELATIONS     # TO DO: correct relations for mass?!!!!!?!?!
    mass_x = np.logspace(8, 12.7)
    kormendy_and_ho = 0.49e9 * (mass_x / 1e11) ** 1.17
    saglia = 10 ** (0.846 * np.log10(mass_x) - 0.713)
    mcconnell_and_ma = 10 ** (8.46 + 1.05 * np.log10(mass_x / 1e11))
    sani = 10 ** (8.16 + 0.79 * np.log10(mass_x) - 11)
    savorgnan = 10 ** (8.56 + 1.04 * np.log10(mass_x / (10 ** 10.81)))

    # relations + intrinsic scatter
    ax[2].plot(mass_x, saglia, 'k-', label='Saglia et al. (2016)')
    #print(saglia * 10 ** .431)
    #print(saglia * 10 ** -.431)
    ax[2].fill_between(mass_x, saglia * (10 ** 0.431), saglia * (10 ** -0.431), color='k', alpha=0.2)
    ax[2].plot(mass_x, savorgnan, 'm-.', label='Savorgnan (2016)')
    ax[2].plot(mass_x, kormendy_and_ho, 'g--', label='Kormendy \& Ho (2013)')
    ax[2].plot(mass_x, mcconnell_and_ma, color='darkorange', linestyle=':', label='McConnell \& Ma (2013)')

    # PLOT MBHs!
    #lumxerrs = [1e11, 1e11, 3e10, 5e10]
    #massxerrs = [1e11, 7e10, 3e10, 7e10]
    for i in range(len(mbhs)):
        ax[0].errorbar(sigmas[i], mbhs[i], xerr=[[sigma_errs[i][0]], [sigma_errs[i][1]]],
                       yerr=[[mbh_errs[i][0]], [mbh_errs[i][1]]], fmt=fmts[i])
        ax[0].text(msig_locs[i][0], msig_locs[i][1], galaxies[i], color=fmts[i][0])
        # ax[1].errorbar(lum_ks[i], mbhs[i], xerr=[[lum_errs[i][0]], [lum_errs[i][1]]],
        #                yerr=[[mbh_errs[i][0]], [mbh_errs[i][1]]], fmt=fmts[i])
        ax[1].errorbar(lum_ks[i], mbhs[i], xuplims=np.array([True]), xerr=[0.2*lum_ks[i]],
                       yerr=[[mbh_errs[i][0]], [mbh_errs[i][1]]], fmt=fmts[i])
        ax[1].text(ml_locs[i][0], ml_locs[i][1], galaxies[i], color=fmts[i][0])
        # ax[2].errorbar(masses[i], mbhs[i], xerr=[[mass_errs[i][0]], [mass_errs[i][1]]],
        #                yerr=[[mbh_errs[i][0]], [mbh_errs[i][1]]], fmt=fmts[i])
        ax[2].errorbar(masses[i], mbhs[i], xuplims=np.array([True]), xerr=[0.2*masses[i]],
                       yerr=[[mbh_errs[i][0]], [mbh_errs[i][1]]], fmt=fmts[i])
        ax[2].text(mm_locs[i][0], mm_locs[i][1], galaxies[i], color=fmts[i][0])

    # SET AXIS DETAILS
    ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    ax[0].set_xlabel(r'$\sigma_{\star}$ [km/s]')
    ax[0].set_ylabel(r'M$_{\rm{BH}}$ [M$_{\odot}$]')
    ax[0].legend()
    ax[0].set_xlim(50, 500)
    ax[0].set_ylim(1e6, 2.5e10)

    ax[1].set_xscale('log')
    ax[1].set_yscale('log')
    ax[1].set_xlabel(r'L$_{\rm{K,bul}}$ [L$_{\odot}$]')
    ax[1].set_ylabel(r'M$_{\rm{BH}}$ [M$_{\odot}$]')
    ax[1].legend()
    ax[1].set_xlim(1e8, 2e12)
    ax[1].set_ylim(1e6, 2.5e10)

    ax[2].set_xscale('log')
    ax[2].set_yscale('log')
    ax[2].set_xlabel(r'M$_{\rm{bul}}$ [M$_{\odot}$]')
    ax[2].set_ylabel(r'M$_{\rm{BH}}$ [M$_{\odot}$]')
    ax[2].legend()
    ax[2].set_xlim(1e8, 5e12)
    ax[2].set_ylim(1e6, 2.5e10)

    plt.savefig('/Users/jonathancohn/Documents/dyn_mod/groupmtg/finaltests/' + savefig)

    plt.show()


def mbh_relations(mbh, mbh_err, lum_k=None, mass_k=None, sigma=None, xerr=None, incl_past=True):
    """
    Calculate MBH relations

    :param mbh: array of black hole mass measurements
    :param mbh_err: array of uncertainties on the black hole mass measurements [[low_0, hi_0], [low_1, high_1], ...]
    :param lum_k: if showing M-L, array (same length as mbh) of luminosities
    :param mass_k: if showing M-M, array (same length as mbh) of stellar masses
    :param sigma: if showing M-sigma, array (same length as mbh) of sigmas
    :param xerr: error on lum_k, mass_k, or sigma
    :param incl_past: if True, include other CEGs on the plot
    :return:
    """
    if lum_k is not None:
        fig = plt.figure(figsize=(8,6))
        lum_x = np.logspace(7., 13.)
        kormendy_and_ho = 0.544e9 * (lum_x / 1e11) ** 1.22  # 0.542, 1.21
        kah_up = 0.611e9 * (lum_x / 1e11) ** 1.22  # 1.30 (1.21)
        kah_down = 0.481e9 * (lum_x / 1e11) ** 1.22  # 1.12 (1.21)

        lasker = 10**(8.56 + 0.75*(np.log10(lum_x) - 11.))  # intrinsic scatter = 0.46

        plt.plot(lum_x, lasker, 'k-', label='Lasker et al. (2014)')
        plt.fill_between(lum_x, lasker * (10**.46), lasker * (10**-.46), color='k', alpha=0.2)  # include relations + intrinsic scatter

        plt.plot(lum_x, kormendy_and_ho, 'g--', label='Kormendy \& Ho (2013)')
        # plt.fill_between(lum_x, kah_up, kah_down, color='k', alpha=0.3)
        # plt.fill_between(lum_x, kormendy_and_ho * 10**(0.4*(-0.31)), kormendy_and_ho * 10**(0.4*(0.31)),
        #                 color='k', alpha=0.3)
        # plt.fill_between(lum_x, kah_up + kormendy_and_ho * 10**(0.4*(-0.31)), kah_down - kormendy_and_ho*(10**(0.4*0.31)-1),  # WAS USING THIS, VERY CLOSE BUT NOT QUITE RIGHT
        #                  color='k', alpha=0.2)  # include relations + intrinsic scatter
        # plt.fill_between(lum_x, kormendy_and_ho * (1 + 10**(0.4*(-0.31))), kormendy_and_ho * (1 - (10**(0.4*0.31)-1)),
        plt.fill_between(lum_x, kormendy_and_ho * (10**(-0.33)), kormendy_and_ho * (10**0.33),  # 10^(.4*.31)-1 (https://faculty.virginia.edu/skrutskie/astr3130.s16/notes/astr3130_lec12.pdf)
                        color='g', alpha=0.2)  # include relations + intrinsic scatter

        # PAGE 54-55 https://arxiv.org/pdf/1304.7762.pdf:
        # -2.5log10(L2/L1) = m2 - m1 -> 10^[(m1-m2)/2.5] = L2/L1 -> L2 = L1*10^[0.4(m1-m2)] ->
        # -> L1 - L2 = L1(1 - 10^[0.4(m1-m2)]) -> L2 - L1 = delta L = L1 (10^[0.4(-delta mag)] - 1)
        # LK = LH * 10^(0.4(MH - MK))

        if incl_past:
            plt.errorbar(7.7e10, 4.9e9, xerr=2.8e10, yerr=1.6e9, fmt='rs')  # NGC 1277
            plt.text(1.2e11, 5.5e9, 'N1277', color='r')
            plt.errorbar(4.6e10, 3e9, xerr=[[2.7e10], [2.7e10]], yerr=[[1.1e9], [1e9]], fmt='rs')  # N1271
            plt.text(1.3e10, 3.2e9, 'N1271', color='r')
            plt.errorbar(9.6e10, 4.9e9, xerr=[[7.9e10], [4.6e10]], yerr=1.7e9, fmt='rs')  # MARK 1216
            plt.text(2.75e10, 5.5e9, 'M1216', color='r')

        for item in range(len(mbh)):
            #plt.errorbar(lum_k[item], mbh[item], yerr=[[mbh_err[item][0]], [mbh_err[item][1]]], fmt='bD')
            plt.errorbar(lum_k[item], mbh[item], yerr=[[mbh_err[item][0]], [mbh_err[item][1]]], xerr=xerr, color='b',
                         marker='D')
        plt.text(9.5e10, 2.3e9, 'U2698', color='b')
            # , assuming an H-K color of 0.2 (Vazdekis et al. 1996) and a K-band solar absolute magnitude of 3.29.
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(r'L$_{\rm{K,bul}}$ [L$_{\odot}$]')
        plt.ylabel(r'M$_{\rm{BH}}$ [M$_{\odot}$]')
        plt.legend()
        plt.xlim(1e8, 2e12)
        plt.ylim(1e6, 2.5e10)
        plt.show()


    elif sigma is not None:
        fig, ax = plt.subplots(1, figsize=(8, 6))
        sig_x = np.logspace(1.7, 2.7)  # ~50 - ~500 km/s
        kormendy_and_ho = 0.310e9 * (sig_x / 200) ** 4.38  # 0.309e9
        kah_up = 0.346e9 * (sig_x / 200) ** 4.38  # 4.67
        kah_down = 0.276e9 * (sig_x / 200) ** 4.38  # 4.09

        # saglia = 10**(5.246 * np.log10(sig_x) - 3.77)  # log MBH = a * log sigma + ZP
        #saglia_up = 10**((5.246+0.274) * np.log10(sig_x) - 3.77+0.631)  # log MBH = (a+da) * log sigma + ZP + dZP (0.631)
        #saglia_down = 10**((5.246-0.274) * np.log10(sig_x) - 3.77-0.631)  # log MBH = (a-da) * log sigma + ZP - dZP
        # da = 0.274, dZP = 0.631, rms = 0.459
        saglia = 10**(4.868 * np.log10(sig_x) - 2.827)  # log MBH = a * log sigma + ZP (CorePowerEClassPC) -> scatter=0.38
        # CorePowerEClassPC = Core & power-law ellipticals, classical bulges, & classical component (Saglia+16 Table 8)

        vdbosch = 10**(-4.00 + 5.35*np.log10(sig_x))

        mcconnell_and_ma = 10**(8.32 + 5.64*np.log10(sig_x / 200.))

        plt.fill_between(sig_x, saglia*(10**0.38), saglia*(10**-0.38), color='k', alpha=0.2)  # relations + intrinsic scatter  # CORRECT
        plt.plot(sig_x, saglia, 'k-', label='Saglia et al. (2016)')
        # relations and scatter from pg54-55 https://arxiv.org/pdf/1304.7762.pdf

        plt.plot(sig_x, kormendy_and_ho, 'g--', label='Kormendy \& Ho (2013)')
        # plt.fill_between(lum_x, kah_up, kah_down, color='k', alpha=0.3)
        #plt.fill_between(lum_x, kormendy_and_ho * 10**(0.4*(-0.31)), kormendy_and_ho * 10**(0.4*(0.31)),
        #                 color='k', alpha=0.3)
        # plt.fill_between(sig_x, kah_up*(1 + 0.28), kah_down*(1 - 0.28), color='k', alpha=0.2)  # relations + intrinsic scatter  # less sure

        plt.plot(sig_x, vdbosch, 'm-.', label='van den Bosch (2016)')  # scatter 0.49 dex

        plt.plot(sig_x, mcconnell_and_ma, color='darkorange', linestyle=':', label='McConnell \& Ma (2013)')  # scatter 0.38 dex

        if incl_past:
            plt.errorbar(333, 4.9e9, xerr=0., yerr=1.6e9, fmt='rs')  # NGC 1277  # BUCKET TO DO: van den Bosch+12 for error on sigma
            plt.text(320, 7e9, 'N1277', color='r')
            plt.errorbar(276, 3e9, xerr=[[4], [73]], yerr=[[1.1e9], [1e9]], fmt='rs')  # N1271
            plt.text(220, 3.2e9, 'N1271', color='r')
            plt.errorbar(308, 4.9e9, xerr=[[6], [16]], yerr=1.7e9, fmt='rs')  # MARK 1216
            plt.text(245, 5.5e9, 'M1216', color='r')

        for item in range(len(mbh)):
            plt.errorbar(sigma[item], mbh[item], yerr=[[mbh_err[item][0]], [mbh_err[item][1]]], xerr=xerr, color='b',
                         marker='D')
            #plt.errorbar(sigma[item], mbh[item], yerr=[[mbh_err[item][0]], [mbh_err[item][1]]], xerr=xerr, color='b',
            #             marker=None)
            plt.text(310, 1.5e9, 'U2698', color='b')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(r'$\sigma$ [km/s]')
        plt.ylabel(r'M$_{\rm{BH}}$ [M$_{\odot}$]')
        plt.legend()
        plt.xlim(50, 500)
        plt.ylim(1e6, 2.5e10)
        plt.show()

    elif mass_k is not None:
        fig = plt.figure(figsize=(8,6))
        mass_x = np.logspace(8, 12.7)
        # TO DO: correct relations for mass!
        kormendy_and_ho = 0.49e9 * (mass_x / 1e11) ** 1.17
        saglia = 10 ** (0.846 * np.log10(mass_x) - 0.713)
        mcconnell_and_ma = 10 ** (8.46 + 1.05 * np.log10(mass_x / 1e11))
        sani = 10 ** (8.16 + 0.79 * np.log10(mass_x) - 11)
        savorgnan = 10 ** (8.56 + 1.04 * np.log10(mass_x / (10 ** 10.81)))

        plt.plot(mass_x, saglia, 'k-', label='Saglia et al. (2016)')
        print(saglia * 10**.431)
        print(saglia * 10**-.431)
        plt.fill_between(mass_x, saglia * (10 ** 0.431), saglia * (10 ** -0.431), color='k', alpha=0.2)  # relations + intrinsic scatter  # CORRECT

        plt.plot(mass_x, savorgnan, 'm-.', label='Savorgnan (2016)')

        plt.plot(mass_x, kormendy_and_ho, 'g--', label='Kormendy \& Ho (2013)')

        plt.plot(mass_x, mcconnell_and_ma, color='darkorange', linestyle=':', label='McConnell \& Ma (2013)')

        # plt.plot(mass_x, sani, color='b', linestyle=(0, (3, 5, 1, 5, 1, 5)), label='Sani (2011)')  # dashdotdotted

        if incl_past:
            plt.errorbar(1.6e11, 4.9e9, yerr=1.6e9, fmt='rs')  # NGC 1277  # TO DO: correct error  # xerr=2.8e10,
            plt.text(1.2e11, 8e9, 'N1277', color='r')
            plt.errorbar(1.0e11, 3e9, yerr=[[1.1e9], [1e9]], fmt='rs')  # N1271  # TO DO: correct error  # xerr=[[2.7e10], [2.7e10]],
            plt.text(1.3e10, 3.2e9, 'N1271', color='r')
            plt.errorbar(1.1e11, 4.9e9, xerr=[[0.9e11], [0.5e11]], yerr=1.7e9, fmt='rs')  # MARK 1216
            plt.text(3e10, 5.5e9, 'M1216', color='r')

        for item in range(len(mbh)):
            plt.errorbar(mass_k[item], mbh[item], yerr=[[mbh_err[item][0]], [mbh_err[item][1]]], fmt='bD')
            plt.text(1.3e11, 2.3e9, 'U2698', color='b')
            # , assuming an H-K color of 0.2 (Vazdekis et al. 1996) and a K-band solar absolute magnitude of 3.29.
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(r'M$_{\rm{bulge}}$ [M$_{\odot}$]')
        plt.ylabel(r'M$_{\rm{BH}}$ [M$_{\odot}$]')
        import matplotlib.ticker as mticker
        plt.legend()
        plt.xlim(1e8, 5e12)
        plt.ylim(1e6, 2.5e10)
        plt.show()


def integral22(rad, dda):

    int22 = integrate.quad(integrand22, 0, rad, args=(rad, dda))[0]

    return int22


def integrand22(a, rad, dda):

    integ22 = a * dda / np.sqrt(rad**2 - a**2)

    return integ22


def integral1(a, rmax, sigma_func, inclination, conversion_factor):

    int1 = integrate.quad(integrand1, a, rmax, args=(sigma_func, a, inclination, conversion_factor))[0]

    return int1


def integrand1(r, sigma_func, a, inclination, conversion_factor):

    integ1 = r * sigma_func(r) * np.cos(inclination) * conversion_factor / np.sqrt(r ** 2 - a ** 2)

    return integ1


def par_dicts(parfile, q=False):
    """
    Return dictionaries that contain file names, parameter names, and initial guesses, for free and fixed parameters

    :param parfile: the parameter file
    :param q: True if I'm using mge vcirc
    :return: params (the free parameters, fixed parameters, and input files), priors (prior boundaries as {'param_name':
        [min, max]} dictionaries), qobs (mge parameter qobs; only included if q=True)
    """

    params = {}
    priors = {}
    nfree = 0
    with open(parfile, 'r') as pf:
        for line in pf:
            if not line.startswith('#'):
                cols = line.split()
                if len(cols) > 0:  # ignore empty lines
                    if cols[0] == 'free':  # free parameters are floats with priors
                        nfree += 1
                        params[cols[1]] = float(cols[2])
                        priors[cols[1]] = [float(cols[3]), float(cols[4])]
                    elif cols[0] == 'float':  # fixed floats
                        params[cols[1]] = float(cols[2])
                    elif cols[0] == 'int':  # fixed ints
                        params[cols[1]] = int(cols[2])
                    elif cols[0] == 'str':  # fixed strings
                        params[cols[1]] = cols[2]

    if q:
        # import sys
        # sys.path.insert(0, '/Users/jonathancohn/Documents/jam/')  # lets me import file from different folder/path
        # import mge_vcirc_mine as mvm
        comp, surf_pots, sigma_pots, qobs = mvm.load_mge(params['mass'])

        return params, priors, nfree, qobs

    else:
        return params, priors, nfree


def model_prep(lucy_out=None, lucy_mask=None, lucy_b=None, lucy_in=None, lucy_it=None, data=None, data_mask=None,
               grid_size=None, res=1., x_std=1., y_std=1., pa=0., ds=4, ds2=4, zrange=None, xyerr=None, theta_ell=0,
               q_ell=0, xell=0, yell=0, avg=True):
    """

    :param lucy_out: output from running lucy on data cube and beam PSF; if it doesn't exist, create it!
    :param lucy_mask: file name of collapsed mask file to use in lucy process (if lucy_out doesn't exist)
    :param lucy_b: file name of input beam (built in make_beam()) to use for lucy process (if lucy_out doesn't exist)
    :param lucy_in: file name of input summed flux map to use for lucy process (if lucy_out doesn't exist)
    :param lucy_it: number of iterations to run in lucy process (if lucy_out doesn't exist)
    :param data: input data cube of observations
    :param data_mask: input mask cube of each slice of the data, for constructing the weight map
    :param grid_size: the pixel grid size to use for the make_beam() function
    :param res: resolution of observations [arcsec/pixel]
    :param x_std: FWHM in the x-direction of the ALMA beam (arcsec) to use for the make_beam() function
    :param y_std: FWHM in the y-direction of the ALMA beam (arcsec) to use for the make_beam() function
    :param pa: position angle (in degrees) to use for the make_beam() function
    :param ds: down-sampling factor for pixel scale used to calculate chi^2
    :param ds2: down-sampling factor for pixel scale used to calculate chi^2, allowing for non-square down-sampling
    :param zrange: range of frequency slices containing emission [zi, zf]
    :param xyerr: x & y pixel region, on the down-sampled pixel scale, where the noise is calculated [xi, xf, yi, yf]
    :param theta_ell: position angle of the annuli for the gas mass, same as for the ellipse fitting region [radians]
    :param q_ell: axis ratio q of the annuli for the gas mass, same as for the ellipse fitting region [unitless]
    :param xell: x center of elliptical annuli for the gas mass, same as for the ellipse fitting region [pixels]
    :param yell: y center of elliptical annuli for the gas mass, same as for the ellipse fitting region [pixels]
    :param avg: averaging vs summing within the rebin() function

    :return: lucy mask, lucy output, synthesized beam, flux map, frequency axis, f_0, freq step, input data cube
    """

    # If the lucy process hasn't been done yet, and the mask cube also hasn't been collapsed yet, create collapsed mask
    hdu_m = fits.open(data_mask)
    fullmask = hdu_m[0].data  # The mask is stored in hdu_m[0].data, NOT hdu_m[0].data[0]
    print(fullmask.shape)
    hdu_m.close()
    if not Path(lucy_out).exists() and not Path(lucy_mask).exists():
        collapsed_mask = integrate.simps(fullmask, axis=0)   # integrate the mask, collapsing to 2D
        for i in range(len(collapsed_mask)):
            for j in range(len(collapsed_mask[0])):
                if collapsed_mask[i, j] != 0.:
                    collapsed_mask[i, j] = 1.  # make it a mask: if data exist in given pix, pix=1; else, pix=0
        hdu1 = fits.PrimaryHDU(collapsed_mask)
        hdul1 = fits.HDUList([hdu1])
        hdul1.writeto(lucy_mask)  # write out to mask file
        print('mask created!')

    # MAKE ALMA BEAM  x_std=major; y_std=minor; rot=90-PA; auto-create the beam file lucy_b if it doesn't yet exist
    beam = make_beam(grid_size=grid_size, res=res, x_std=x_std, y_std=y_std, rot=np.deg2rad(90. - pa), fits_name=lucy_b)

    # COLLAPSE THE DATA CUBE
    fluxes, freq_ax, f_0, input_data, fstep = get_fluxes(data, data_mask, write_name=lucy_in)

    # DECONVOLVE FLUXES WITH BEAM PSF
    if not Path(lucy_out).exists():  # to use iraf, must "source activate iraf27" on command line
        t_pyraf = time.time()
        from skimage import restoration  # need to be in "three" environment (source activate three == tres)
        hduin = fits.open(lucy_in)
        l_in = hduin[0].data
        hdub = fits.open(lucy_b)
        l_b = hdub[0].data
        hduin.close()
        hdub.close()
        l_in[l_in < 0.] = 0.
        # https://github.com/scikit-image/scikit-image/issues/2551 (it was returning nans)
        # open skimage/restoration/deconvolution.py
        # find this line (~line 389): im_deconv *= convolve_method(relative_blur, psf_mirror, 'same')
        # Insert a new line before that line: relative_blur[np.isnan(relative_blur)] = 0
        l_o = restoration.richardson_lucy(l_in, l_b, lucy_it, clip=False)  # need clip=False
        print('lucy process done in ' + str(time.time() - t_pyraf) + 's')  # ~1s

        hdu1 = fits.PrimaryHDU(l_o)
        hdul1 = fits.HDUList([hdu1])
        hdul1.writeto(lucy_out)  # write out to lucy_out file
        print(lucy_out)

    hdu = fits.open(lucy_out)
    lucy_out = hdu[0].data
    hdu.close()

    # ESTIMATE NOISE (RMS) IN ORIGINAL DATA CUBE [z, y, x] (these cuts are safe, as there is no emission near cube edge)
    cut_y = len(input_data[0]) % ds2  # is the cube y-dimension divisible by ds2? If not, cut remainder from cube edge
    cut_x = len(input_data[0][0]) % ds  # is the cube x-dimension divisible by ds2? If not, cut remainder from cube edge
    if cut_x == 0:  # if there is no remainder:
        cut_x = -len(input_data[0][0])  # don't want to cut anything
    if cut_y == 0:  # if there is no remainder:
        cut_y = -len(input_data[0])  # don't want to cut anything
    noise_ds = rebin(input_data[:, :-cut_y, :-cut_x], ds2, ds, avg=avg)  # down-sample noise to the chi^2 pixel scale

    noise = []  # For large N, Variance ~= std^2
    for z in range(zrange[0], zrange[1]):  # for each relevant freq slice, calculte std (aka rms) ~variance
        noise.append(np.std(noise_ds[z, int(xyerr[2]/ds2):int(xyerr[3]/ds2), int(xyerr[0]/ds):int(xyerr[1]/ds)],
                            ddof=1))  # ddof=1 makes variance ~ 1/(N-1) instead of 1/N

    # CALCULATE VELOCITY WIDTH  # vsys = 6454.9 estimated from various test runs; see eg. Week of 2020-05-04 on wiki
    v_width = 2.99792458e5 * (1 + (6454.9 / 2.99792458e5)) * fstep / f_0  # velocity width [km/s] = c*(1+v/c)*fstep/f0

    # CONSTRUCT THE FLUX MAP IN UNITS Jy km/s beam^-1
    collapsed_fluxes_vel = np.zeros(shape=(len(input_data[0]), len(input_data[0][0])))
    for zi in range(len(input_data)):
        collapsed_fluxes_vel += input_data[zi] * fullmask[zi] * v_width

    # DEFINE SEMI-MAJOR AXES FOR SAMPLING, THEN CALCULATE THE MEAN SURFACE BRIGHTNESS INSIDE ELLIPTICAL ANNULI
    semi_major = np.linspace(0., 100., num=85)  # [pix]  # num=66)
    #semi_major = np.linspace(0., 85., num=1000)  # [pix]  # np.linspace(0., 100., num=85.)
    #semi_major = np.linspace(0., 85., num=5000)  # [pix]
    co_surf, co_errs = annuli_sb(collapsed_fluxes_vel, semi_major, theta_ell, q_ell, xell, yell)  # CO [Jy km/s beam^-1]
    co_ell_sb = np.asarray(co_surf)  # put the output list of mean CO surface brightnesses in an array
    co_ell_rad = (2. * semi_major[1:] + q_ell * semi_major[1:]) / 3.  # mean_ellipse_radius = (2a + b)/3
    # semi_major[1:] because there is 1 less annulus than there are ellipses

    return lucy_mask, lucy_out, beam, fluxes, freq_ax, f_0, fstep, input_data, noise, co_ell_sb, co_ell_rad


def make_beam(grid_size=99, res=1., amp=1., x0=0., y0=0., x_std=1., y_std=1., rot=0., fits_name=None):
    """
    Generate a beam spread function for the ALMA beam

    :param grid_size: size of grid (must be odd!)
    :param res: resolution of the grid (arcsec/pixel)
    :param amp: amplitude of the 2d gaussian
    :param x0: mean of x axis of 2d gaussian
    :param y0: mean of y axis of 2d gaussian
    :param x_std: FWHM of beam in x [beam major axis] (to use for standard deviation of Gaussian) (in arcsec)
    :param y_std: FWHM of beam in y [beam minor axis] (to use for standard deviation of Gaussian) (in arcsec)
    :param rot: rotation angle in radians
    :param fits_name: this name will be the filename to which the beam fits file is written (if None, write no file)

    :return the synthesized beam
    """

    # SET RESOLUTION OF BEAM GRID THE SAME AS FLUXES GRID
    x_beam = [0.] * grid_size
    y_beam = [0.] * grid_size

    # set center of the beam grid axes (find the central pixel number along each axis: same for x_beam and y_beam)
    ctr = (len(x_beam) + 1.) / 2.  # +1 bc python starts counting at 0, grid_size is odd

    # grid_size must be odd, so fill in the axes with resolution * ((i+1) - ctr)!
    for i in range(len(x_beam)):
        x_beam[i] = res * ((i + 1) - ctr)  # (arcsec/pix) * N_pixels = arcsec
        y_beam[i] = res * ((i + 1) - ctr)  # (arcsec/pix) * N_pixels = arcsec

    # SET UP MESHGRID
    xx, yy = np.meshgrid(x_beam, y_beam)

    # SET UP PSF 2D GAUSSIAN VARIABLES
    n = 2.35482  # Convert from FWHM to std_dev: http://mathworld.wolfram.com/GaussianFunction.html
    x_std /= n
    y_std /= n
    a = np.cos(rot) ** 2 / (2 * x_std ** 2) + np.sin(rot) ** 2 / (2 * y_std ** 2)
    b = -np.sin(2 * rot) / (4 * x_std ** 2) + np.sin(2 * rot) / (4 * y_std ** 2)
    c = np.sin(rot) ** 2 / (2 * x_std ** 2) + np.cos(rot) ** 2 / (2 * y_std ** 2)

    # CALCULATE PSF, NORMALIZE IT TO AMPLITUDE
    synth_beam = np.exp(-(a * (xx - x0) ** 2 + 2 * b * (xx - x0) * (yy - y0) + c * (yy - y0) ** 2))
    A = amp / np.amax(synth_beam)  # normalize it so the input amplitude is the real amplitude
    synth_beam *= A

    # IF write_fits, WRITE PSF TO FITS FILE
    if not Path(fits_name).exists():
        hdu = fits.PrimaryHDU(synth_beam)
        hdul = fits.HDUList([hdu])
        hdul.writeto(fits_name)

    return synth_beam


def get_sig(r=None, sig0=1., r0=1., mu=1., sig1=0.):
    """
    :param r: 2D array of radius values = r(x, y) [length units]
    :param sig0: uniform sigma_turb component [velocity units]
    :param r0: expectation of the distribution; i.e. scale radius value [same units as r]
    :param mu: standard deviation of the distribution [same units as r]
    :param sig1: sigma offset [same units as sig0]
    :return: dictionary of the three different sigmas
    """

    sigma = {'flat': sig0, 'gauss': sig1 + sig0 * np.exp(-(r - r0) ** 2 / (2 * mu ** 2)),
             'exp': sig1 + sig0 * np.exp(-r / r0)}

    return sigma


def get_fluxes(data_cube, data_mask, write_name=None):
    """
    Integrate over the frequency axis of a data cube to get a flux map!

    :param data_cube: input data cube of observations
    :param data_mask: mask for each slice of data, for construction of the flux map
    :param write_name: name of fits file to which to write the flux map (only writes if write_name does not yet exist)

    :return: collapsed data cube (i.e. flux map), len(z_ax), intrinsic freq of observed line, input data cube
    """
    hdu = fits.open(data_cube)
    data = hdu[0].data[0]  # data[0] contains: z, y, x (121, 700, 700)

    hdu_m = fits.open(data_mask)
    mask = hdu_m[0].data  # this is hdu_m[0].data, NOT hdu[0].data[0], unlike the data_cube above

    z_len = len(hdu[0].data[0])  # store the number of velocity slices in the data cube
    freq1 = float(hdu[0].header['CRVAL3'])  # starting frequency in the data cube
    f_step = float(hdu[0].header['CDELT3'])  # frequency step in the data cube  # note: fstep is negative for NGC_3258
    f_0 = float(hdu[0].header['RESTFRQ'])
    freq_axis = np.arange(freq1, freq1 + (z_len * f_step), f_step)  # [bluest, ..., reddest]
    # NOTE: For NGC1332, this includes endpoint (arange shouldn't). However, if cut endpoint at fstep-1, arange doesn't
    # include it...So, for 1332, include the extra point above, then cutting it off: freq_axis = freq_axis[:-1]
    # NOTE: NGC_3258 DOES *NOT* HAVE THIS ISSUE (nor does UGC 2698), SOOOOOOOOOO COMMENT OUT FOR NOW!

    # Collapse the fluxes! Sum over all slices, multiplying each slice by the slice's mask and by the frequency step
    collapsed_fluxes = np.zeros(shape=(len(data[0]), len(data[0][0])))
    for zi in range(len(data)):
        collapsed_fluxes += data[zi] * mask[zi] * abs(f_step)

    hdu.close()  # close data
    hdu_m.close()  # close mask

    collapsed_fluxes[collapsed_fluxes < 0] = 0.

    if not Path(write_name).exists():  # if lucy_in file (flux map to be lucy-deconvolved) does not exist, create it!
        hdu = fits.PrimaryHDU(collapsed_fluxes)
        hdul = fits.HDUList([hdu])
        hdul.writeto(write_name)  # '/Users/jonathancohn/Documents/dyn_mod/' +
        hdul.close()
    return collapsed_fluxes, freq_axis, f_0, data, abs(f_step)


def blockshaped(arr, nrow, ncol):
    """
    Function to use for rebinning data

    :param arr: the array getting rebinned (2d)
    :param nrow: number of pixels in a row to get rebinned
    :param ncol: number of pixels in a column to get rebinned (same as nrow for n x n rebinning)
    :return: blocks of nrow x ncol subarrays from the input array
    """

    h, w = arr.shape
    return arr.reshape(h // nrow, nrow, -1, ncol).swapaxes(1, 2).reshape(-1, nrow, ncol)


def rebin(data, nr, nc, avg=True):
    """
    Rebin data or model cube (or one slice of a cube) in blocks of n x n pixels

    :param data: input data cube, model cube, or slice of a cube, e.g. a 2Darray
    :param nr: size of pixel binning, rows (e.g. nr=4, nc=4 rebins the data in blocks of 4x4 pixels)
    :param nc: size of pixel binning, columns
    :param avg: if True, return the mean within each rebinnined pixel, rather than the sum. If False, return the sum.
    :return: rebinned cube or slice
    """

    if avg:
        nn = 1.
    else:
        nn = nr * nc

    rebinned = []
    if len(data.shape) == 2:  # if binning a 2D array: bin data in groups of nr x nc pixels
        subarrays = blockshaped(data, nr, nc)  # subarrays.shape = len(data)*len(data[0]) / (nr*nc)
        # each pixel in the new, rebinned data cube is the mean of each nr x nc set of original pixels
        reshaped = nn * np.mean(np.mean(subarrays, axis=-1), axis=-1).reshape((int(len(data) / nr),
                                                                               int(len(data[0]) / nc)))
        rebinned.append(reshaped)  # reshaped.shape = (len(data) / nr, len(data[0]) / nc)
    else:  # if binning a 3D array
        for z in range(len(data)):  # bin each slice of the data in groups of nr x nc pixels
            subarrays = blockshaped(data[z, :, :], nr, nc)  # subarrays.shape = len(data[0])*len(data[0][0]) / (nr*nc)
            reshaped = nn * np.mean(np.mean(subarrays, axis=-1), axis=-1).reshape((int(len(data[0]) / nr),
                                                                                   int(len(data[0][0]) / nc)))
            rebinned.append(reshaped)  # reshaped.shape = (len(data[0]) / nr, len(data[0][0]) / nc)

    return np.asarray(rebinned)


def ellipse_fitting(cube, rfit, x0_sub, y0_sub, res, pa_disk, q):
    """
    Create an elliptical mask, within which we will do the actual fitting

    :param cube: sub-cube, cut to x,y,z regions where emission exists in data (model or data; using only for dimensions)
    :param rfit: the radius of the disk region to be fit [arcsec]
    :param x0_sub: x-pixel location of BH, in coordinates of the sub-cube [pix]
    :param y0_sub: y-pixel location of BH, in coordinates of the sub-cube [pix]
    :param res: resolution of the pixel scale [arcsec/pix]
    :param pa_disk: position angle of the disk [radians]
    :param q: axis ratio of the ellipse = cos(disk_inc) [unitless]

    :return: masked ellipse array, to mask out everything in model cube outside of the ellipse we want to fit
    """

    # Define the Fitting Ellipse
    a = rfit / res  # size of semimajor axis, in pixels
    b = (rfit / res) * q  # size of semiminor axis, in pixels

    ell = Ellipse2D(amplitude=1., x_0=x0_sub, y_0=y0_sub, a=a, b=b, theta=pa_disk)  # create elliptical region
    y_e, x_e = np.mgrid[0:len(cube[0]), 0:len(cube[0][0])]  # make sure this grid is the size of the downsampled cube!

    # Select the regions of the ellipse we want to fit
    ellipse_mask = ell(x_e, y_e)

    return ellipse_mask


def annuli_sb(flux_map, semi_major_axes, position_angle, axis_ratio, x_center, y_center):
    """
    Create an array of elliptical annuli, in which to calculate the mean CO surface brightness

    :param flux_map: CO surface brightness flux map, collapsed with the velocity width [Jy km/s beam^-1]
    :param semi_major_axes: the semi-major axes of the elliptical annuli to put on the flux map [pix]
    :param position_angle: position angle of the disk [radians]
    :param axis_ratio: axis ratio of the ellipse = cos(disk_inc) [unit-less]
    :param x_center: x-pixel location of BH, in coordinates of the flux_map [pix]
    :param y_center: y-pixel location of BH, in coordinates of the flux_map [pix]

    :return: array of mean CO flux calculated in each elliptical annulus on the flux map
    """
    ellipses = []

    for sma in semi_major_axes:  # create elliptical region for each annulus
        ell = Ellipse2D(amplitude=1., x_0=x_center, y_0=y_center, a=sma, b=sma * axis_ratio, theta=position_angle)
        y_e, x_e = np.mgrid[0:len(flux_map), 0:len(flux_map[0])]  # this grid is the size of the flux_map!

        ellipses.append(ell(x_e, y_e))  # Select the regions of the ellipse we want to fit

    annuli = []
    average_co = []
    errs_co = np.zeros(shape=(2, len(ellipses) - 1))
    ns = []
    for e in range(len(ellipses) - 1):  # because indexed to e+1
        annuli.append(ellipses[e+1] - ellipses[e])
        npix = np.sum(annuli[e])
        annuli[e][annuli[e] == 0] = np.nan  # turn mask into 1s and NaNs instead of 1s and 0s, so that 0s in flux = ok
        annulus_flux = annuli[e] * flux_map
        #annulus_flux[annulus_flux == 0] = np.nan  # set 0s equal to NaNs
        average_co.append(np.nanmean(annulus_flux))  # calculate the mean, ignoring NaNs
        errs_co[:, e] = np.nanpercentile(annulus_flux, [16., 84.]) / npix  # use 68% confidence interval as errors
        ns.append(npix)

    return average_co, errs_co


def gas_vel(resolution, co_rad, co_sb, dist, f_0, inc_fixed, zfixed=0.02152):
    """

    :param radius_array: 2D R array grid built in the ModelGrid class
    :param resolution: resolution of observations [arcsec/pixel]
    :param co_rad: mean elliptical radii of annuli used in calculation of co_sb [pix]
    :param co_sb: mean CO surface brightness in elliptical annuli [Jy km/s beam^-1]
    :param dist: angular diameter distance to the galaxy [Mpc]
    :param f_0: rest frequency of the observed line in the data cube [Hz]
    :param inc_fixed: fixed inclination, used for the Binney & Tremaine integration [radians]
    :param zfixed: fixed redshift, used for unit conversion

    :return: circular gas velocity interpolation function
    """
    G_pc = (6.67 * 10**-11) * 1.989e30 * (1 / 3.086e16) / 1e3**2  # G [Msol^-1 * pc * km^2 * s^-2]
    f_he = 1.36  # additional fraction of gas that is helium (f_he = 1 + helium mass fraction)
    r21 = 0.7  # CO(2-1)/CO(1-0) SB ratio (see pg 6-8 in Boizelle+17)
    alpha_co10 = 3.1  # CO(1-0) to H2 conversion factor (see pg 6-8 in Boizelle+17)

    # pix per beam = 2pi sigx sigy [pix^2]
    pix_per_beam = 2. * np.pi * (0.197045 / resolution / 2.35482) * (0.103544 / resolution / 2.35482)
    pc_per_pix = dist * 1e6 / 206265 * resolution
    pc2_per_beam = pix_per_beam * pc_per_pix ** 2  # pc^2 per beam = pix/beam * pc^2/pix

    co_radii = co_rad * pc_per_pix  # convert input annulus mean radii from pix to pc
    co_sb = np.nan_to_num(co_sb)  # replace NaNs with 0s in input CO mean surface brightness profile

    # Set up integration bounds, numerical step size, vectors r & a (see Binney & Tremaine eqn 2.157)
    # use maxr >> than maximum CO radius, so potential contributions at maxr << those near the disk edge.
    min_r = 0.  # integration lower bound [pix or pc]
    max_r = 1300.  # upper bound [pc]; disk peaks <~100pc, extends <~700pc  # 510 for edge
    nr = 500  # 500  # number of steps used in integration process
    avals = np.linspace(min_r, max_r, nr)  # [pc]  # range(min_r,max_r,(max_r-min_r)/del_r)
    rvals = np.linspace(min_r, max_r, nr)  # [pc]  # range(min_r,max_r,(max_r-min_r)/del_r)

    # convert from Jy km/s to Msol (Boizelle+17; Carilli & Walter 13, S2.4: https://arxiv.org/pdf/1301.0371.pdf)
    msol_per_jykms = 3.25e7 * alpha_co10 * f_he * dist ** 2 / \
                     ((1 + zfixed) * r21 * (f_0 / 1e9) ** 2)  # f_0 in GHz, not Hz
    # equation for (1+z)^3 is for observed freq, but using rest freq -> nu0^2 = (nu*(1+z))^2
    # units on 3.25e7 are [K Jy^-1 pc^2/Mpc^2 GHz^2] --> total units [Msol (Jy km/s)^-1]

    # Interpolate CO surface brightness vs elliptical mean radii, to construct Sigma(rvals).
    # Units [Jy km/s/beam] * [Msol/(Jy km/s)] / [pc^2/beam] = [Msol/pc^2]
    sigr3_func_r = interpolate.interp1d(co_radii, co_sb, kind='quadratic', fill_value='extrapolate')

    #plt.errorbar(co_radii, co_sb, fmt='ko')#, yerr=errs_co)
    #plt.plot(np.linspace(0,750), sigr3_func_r(np.linspace(0,750)), 'b--')
    #plt.show()

    # 1277 RA, DEC: 49.964542	41.573528
    # 1271 RA, DEC: 49.797000	41.353250
    # 1275 RA, DEC: 49.950667	41.511696
    # 2698 RA, DEC: 50.512000	40.863889

    # ESTIMATE GAS MASS
    #i1 = integrate.quad(sigr3_func_r, 0, np.inf)[0]
    #print(i1)  # 47.448953273905644
    #i1 = 2 * np.pi * integrate.quad(sigr3_func_r, 0, rvals[-1])[0]  # 45.47 * 2pi!
    #print(0.04879794212243681 * np.cos(inc_fixed) * msol_per_jykms)  # 552840.5698900325
    #print(1.190971117285626 * np.cos(inc_fixed) * msol_per_jykms)  # 13492723.720823068
    # SHOULD THIS BE 2*pi*integrate.quad(sigr3_func_r, 0, rvals[-1])[0]
    # i2 = integrate.quadrature(sigr3_func_r, 0, rvals[-1])[0]  # 45.50
    # i3 = integrate.romb(co_sb, dx=co_radii[2]-co_radii[1])  # 44.66 (use linspace with num=66 -> 65 samples (1+2^n)
    #print(i1)  # 283.7109152673565 TADA
    # 48.10314883101475
    #print(i1 * np.cos(inc_fixed))
    #print(28.27 * np.cos(inc_fixed) * msol_per_jykms)  # 320275860.64137024
    #print(282.720 * np.cos(inc_fixed) * msol_per_jykms)  # 3202985189.9726987 -> log10(M) = 9.50555493
    # ^without inc: 8440988792.877579 -> log10(M) = 9.92639332
    #i1 *= np.cos(inc_fixed) * msol_per_jykms  # Jy km/s/beam * [Msol/(Jy km/s)]
    #print(np.log10(i1))  # 9.507074443222836
    # 8.736371904617355
    #print(oop)
    # END ESTIMATE GAS MASS

    # PYTHON INTEGRATION: calculate the inner integral (see eqn 2.157 from Binney & Tremaine)
    int1 = np.zeros(shape=len(avals))
    for av in range(len(avals)):
        int1[av] = integral1(avals[av], np.inf, sigr3_func_r, inc_fixed, msol_per_jykms / pc2_per_beam)
    zerocut = 580  # [pc] select point at or just beyond the disk edge
    int1[rvals > zerocut] = 0.  # set all points outside the disk edge = 0 (sometimes get spurious points)

    interp_int1 = unsp(rvals, int1)  # interpolate the inner integral using UnivariateSpline
    interp_int1.set_smoothing_factor(9e8)  # smooth the inner integral, so it will be well-behaved

    dda_sp = interp_int1.derivative()  # calculate the derivative of the spline smoothed integral
    dda_sp_r = dda_sp(rvals)  # define the derivative on the rvals grid

    int2 = np.zeros(shape=len(rvals))  # calculate the outer integral! # this integral takes ~0.35 seconds
    for rv in range(len(rvals[1:])):
        int2[rv] = integral22(rvals[rv], dda_sp_r[rv])

    vcg = np.sqrt(-4 * G_pc * int2)  # calculate the circular velocity due to the gas!
    vcg = np.nan_to_num(vcg)  # set all NaNs (as a result of negative sqrt values) = 0

    # create a function to interpolate from vcg(rvals) to vcg(R)
    vcg_func = interpolate.interp1d(rvals, vcg, kind='quadratic', fill_value='extrapolate')

    return vcg_func  # vg = vcg_func(radius_array), to interpolate vcg onto vcg(R)



def map_averaging(m1, xpix, ypix, binNum, x_in, nPixels):
    """
    Function for averaging moment maps within voronoi bins

    :param m1: input moment map, calculated in ModelGrid
    :param xpix: output from ModelGrid.just_the_bins
    :param ypix: output from ModelGrid.just_the_bins
    :param binNum: output from ModelGrid.just_the_bins
    :param x_in: output from ModelGrid.just_the_bins
    :param nPixels: output from ModelGrid.just_the_bins
    :return: m1, with pixels averaged in the Voronoi bin regions generated in ModelGrid.just_the_bins
    """

    flattened_binned_m1 = np.zeros(shape=max(binNum) + 1)  # flatten and bin the moment 1 map
    for xy in range(len(x_in)):
        flattened_binned_m1[binNum[xy]] += m1[ypix[xy], xpix[xy]] / nPixels[binNum[xy]]

    # convert the flattened binned moment 1 map into a vector of the same size as the x & y inputs
    full_binned_m1 = np.zeros(shape=len(binNum))
    for xy in range(len(x_in)):
        full_binned_m1[xy] += flattened_binned_m1[binNum[xy]]

    # create the binned moment maps for display
    m1_vb = np.zeros(shape=m1.shape)
    for xy in range(len(x_in)):
        m1_vb[ypix[xy], xpix[xy]] = full_binned_m1[xy]

    return m1_vb


class ModelGrid:

    def __init__(self, resolution=0.05, os=4, x_loc=0., y_loc=0., mbh=4e8, inc=np.deg2rad(60.), vsys=None, vrad=0.,
                 kappa=0., omega=0., dist=17., theta=np.deg2rad(200.), input_data=None, lucy_out=None, vtype='orig',
                 out_name=None, beam=None, rfit=1., q_ell=1., theta_ell=0., xell=360., yell=350., bl=False,
                 enclosed_mass=None, menc_type=0, ml_ratio=1., sig_type='flat', sig_params=None, f_w=1., noise=None,
                 ds=None, ds2=None, zrange=None, xyrange=None, reduced=False, freq_ax=None, f_0=0., fstep=0., opt=True,
                 quiet=False, n_params=8, data_mask=None, avg=True, f_he=1.36, r21=0.7, alpha_co10=3.1, incl_gas=False,
                 co_rad=None, co_sb=None, z_fixed=0.02152, pvd_width=None, vcg_func=None, sqrt2n=False):
        # Astronomical Constants:
        self.c = 2.99792458 * 10 ** 8  # [m / s]
        self.pc = 3.086 * 10 ** 16  # [m / pc]
        self.G = 6.67 * 10 ** -11  # [kg^-1 * m^3 * s^-2]
        self.M_sol = 1.989 * 10 ** 30  # [kg / solar mass]
        self.H0 = 70  # [km/s/Mpc]
        self.arcsec_per_rad = 206265.  # [arcsec / radian]
        self.m_per_km = 10. ** 3  # [m / km]
        self.G_pc = self.G * self.M_sol * (1. / self.pc) / self.m_per_km ** 2  # G [Msol^-1 * pc * km^2 * s^-2] (gross)
        self.c_kms = self.c / self.m_per_km  # [km / s]
        # Input Parameters
        self.z_fixed = z_fixed  # fixed redshift, used to transform the luminosity dist to the angular diameter dist
        self.resolution = resolution  # resolution of observations [arcsec/pixel]
        self.os = os  # oversampling factor
        self.x_loc = x_loc  # the location of the BH, as measured along the x axis of the data cube [pixels]
        self.y_loc = y_loc  # the location of the BH, as measured along the y axis of the data cube [pixels]
        self.mbh = mbh  # supermassive black hole mass [solar masses]
        self.inc = inc  # inclination of the galaxy [radians]
        self.vsys = vsys  # the systemic velocity [km/s]
        self.vrad = vrad  # optional radial inflow term [km/s]
        self.kappa = kappa  # optional radial inflow term; tie inflow to the overall line-of-sight velocity [unitless]
        self.omega = omega  # optional velocity coefficient, used with kappa for radial inflow [unitless]
        self.vtype = vtype  # 'vrad', 'kappa', 'omega', any other value for original (no radial velocity component)
        self.dist = dist  # angular diameter distance to the galaxy [Mpc]
        self.pc_per_ac = self.dist * 1e6 / self.arcsec_per_rad  # small angle formula (convert dist to pc, from Mpc)
        self.pc_per_pix = self.dist * 1e6 / self.arcsec_per_rad * self.resolution  # small angle formula, as above
        self.pc_per_sp = self.pc_per_pix / self.os  # pc per subpixel (over-sampling pc pixel scale)
        self.theta = theta  # angle from +x_obs axis counterclockwise to the disk's blueshifted side (-x_disk) [radians]
        self.zred = self.vsys / self.c_kms  # redshift
        self.input_data = input_data  # input 3D data cube of observations
        self.data_mask = data_mask  # 3D strictmask cube, used to mask the data
        self.lucy_out = lucy_out  # deconvolved 2D fluxmap, output from running lucy-richardson on fluxmap and beam PSF
        self.out_name = out_name  # optional; output name of the fits file to which to save the convolved model cube
        self.beam = beam  # synthesized alma beam (output from model_ins)
        self.rfit = rfit  # disk radius (elliptical semi-major axis) within which we compare the model and data [arcsec]
        self.q_ell = q_ell  # axis ratio q of fitting ellipse [unitless]
        self.theta_ell = theta_ell  # same as theta, but held fixed; used for the ellipse fitting region [radians]
        self.xell = xell  # same as x_loc, but held fixed; used for the ellipse fitting region [pixels]
        self.yell = yell  # same as y_loc, but held fixed; used for the ellipse fitting region [pixels]
        self.bl = bl  # lucy weight map unit indicator (bl=False or 0 --> Jy/beam * Hz; bl=True or 1 --> Jy/beam * km/s)
        self.enclosed_mass = enclosed_mass  # MGE file name, or similar file containing M(R) information [file]
        self.menc_type = menc_type  # Determines how stellar mass is included [0->MGE; 1->v(R); 2->M(R)]
        self.ml_ratio = ml_ratio  # The mass-to-light ratio of the galaxy [Msol / Lsol]
        self.sig_type = sig_type  # Str determining what type of sigma to use. Can be 'flat', 'exp', or 'gauss'
        self.sig_params = sig_params  # array of sigma parameters: [sig0, r0, mu, sig1]
        self.f_w = f_w  # multiplicative weight factor for the line profiles [unitless]
        self.noise = noise  # array of the estimated (ds x ds-binned) noise per slice (within the zrange) [Jy/beam]
        self.ds = ds  # downsampling factor to use when averaging pixels together for actual model-data comparison [int]
        self.ds2 = ds2  # downsampling factor (same as self.ds, but for second dimension) [int]
        self.zrange = zrange  # array with the slices of the data cube where real emission shows up [zi, zf]
        self.xyrange = xyrange  # array with the subset of the cube (in pixels) that contains emission [xi, xf, yi, yf]
        self.reduced = reduced  # if True, ModelGrid.chi2() returns the reduced chi^2 instead of the regular chi^2
        self.freq_ax = freq_ax  # array of the frequency axis in the data cube, from bluest to reddest frequency [Hz]
        self.f_0 = f_0  # rest frequency of the observed line in the data cube [Hz]
        self.fstep = fstep  # frequency step in the frequency axis [Hz]
        self.opt = opt  # frequency axis velocity convention; opt=True -> optical; opt=False -> radio
        self.quiet = quiet  # if quiet=True, suppress printing out stuff!
        self.n_params = n_params  # number of free parameters being fit, as counted from the param file in par_dicts()
        self.avg = avg  # averaging vs summing within the rebin() function
        self.co_rad = co_rad  # mean elliptical radii of annuli used in calculation of co_sb [pix]
        self.co_sb = co_sb  # mean CO surface brightness in elliptical annuli [Jy km/s beam^-1]
        self.f_he = f_he  # additional fraction of gas that is helium (f_he = 1 + helium mass fraction)
        self.r21 = r21  # CO(2-1)/CO(1-0) SB ratio (see pg 6-8 in Boizelle+17)
        self.alpha_co10 = alpha_co10  # CO(1-0) to H2 conversion factor (see pg 6-8 in Boizelle+17)
        self.incl_gas = incl_gas  # if True, include gas mass in calculations
        self.pvd_width = pvd_width  # width (in pixels) for the PVD extraction
        self.vcg_func = vcg_func  # gas circular velocity interpolation function, returns v_c,gas(R) in units of km/s
        self.sqrt2n = sqrt2n  # divide chi^2 by sqrt(2N) for stat things that don't really apply to this work anyway
        # Parameters to be built in create_grid(), convolve_cube(), or chi2 functions inside the class
        self.z_ax = None  # velocity axis, constructed from freq_ax, f_0, and vsys, based on opt
        self.weight = None  # 2D weight map, constructed from lucy_output (the deconvolved fluxmap)
        self.freq_obs = None  # the 2D line-of-sight velocity map, converted to frequency
        self.delta_freq_obs = None  # the 2D turbulent velocity map, converted to delta-frequency
        self.clipped_data = None  # the data sub-cube that we compare to the model, clipped by zrange and xyrange
        self.convolved_cube = None  # the model cube: create from convolving the intrinsic model cube with the ALMA beam
        self.ell_mask = None  # mask defined by the elliptical fitting region, before downsampling
        self.ell_ds = None  # mask defined by the elliptical fitting region, created on ds x ds down-sampled pixels
    """
    Build grid for dynamical modeling!
    
    Class structure following: https://www.w3schools.com/python/python_classes.asp

    Class functions:
        grids: calculates weight map, freq_obs map, and delta_freq_obs map
        convolution: grids must be run first; create the intrinsic model cube and convolve it with the ALMA beam
        chi2: convolution must be run first; calculates chi^2 and/or reduced chi^2
        line_profiles: chi2 must be run first; plot the line profile of a given x,y (binned pixel) 
        pvd: chi2 must be run first; generate the position-velocity diagram
        output_cube: convolution must be run first; store model cube in a fits file
        test_ellipse: grids must be run first; use to check how the fitting-ellipse looks with respect to the weight map
        moment_0: convolution must be run first; create 0th moment map
        moment_12: convolution must be run first; create 1st or 2nd moment map
    """

    def grids(self):
        t_grid = time.time()

        # SUBPIXELS (reshape deconvolved flux map [lucy_out] sxs subpixels, so subpix has flux=(real pixel flux)/s**2)
        if not self.quiet:
            print('start')
        if self.os == 1:  # subpix_deconvolved == lucy_out, but with sxs subpixels per pixel & total flux conserved
            subpix_deconvolved = self.lucy_out + 0.  # setting = lucy_out (without e.g. +0) results in model not working
        else:
            subpix_deconvolved = np.zeros(shape=(len(self.lucy_out) * self.os, len(self.lucy_out[0]) * self.os))
            for ypix in range(len(self.lucy_out)):
                for xpix in range(len(self.lucy_out[0])):
                    subpix_deconvolved[(ypix * self.os):(ypix + 1) * self.os, (xpix * self.os):(xpix + 1) * self.os] = \
                        self.lucy_out[ypix, xpix] / self.os ** 2

        # convert from frequency (Hz) to velocity (km/s), with freq_ax in Hz
        if self.opt:  # optical convention
            z_ax = np.asarray([self.vsys + ((self.f_0 - freq) / freq) * (self.c / self.m_per_km) for freq in
                               self.freq_ax])
        else:  # radio convention
            z_ax = np.asarray([self.vsys + ((self.f_0 - freq) / self.f_0) * (self.c / self.m_per_km) for freq in
                               self.freq_ax])

        # RESCALE subpix_deconvolved, z_ax, freq_ax TO CONTAIN ONLY THE SUB-CUBE REGION WHERE EMISSION ACTUALLY EXISTS
        subpix_deconvolved = subpix_deconvolved[self.os * self.xyrange[2]:self.os * self.xyrange[3],
                                                self.os * self.xyrange[0]:self.os * self.xyrange[1]]  # stored: y,x


        self.z_ax = z_ax[self.zrange[0]:self.zrange[1]]
        self.freq_ax = self.freq_ax[self.zrange[0]:self.zrange[1]]

        # RESCALE (x_loc, y_loc) AND (xell, yell) PIXEL VALUES TO CORRESPOND TO SUB-CUBE PIXEL LOCATIONS!
        x_loc = self.x_loc - self.xyrange[0]  # x_loc - xi
        y_loc = self.y_loc - self.xyrange[2]  # y_loc - yi

        self.xell = self.xell - self.xyrange[0]
        self.yell = self.yell - self.xyrange[2]

        # SET UP OBSERVATION AXES: initialize x,y axes at 0., with lengths = os * len(input data cube axes)
        y_obs_ac = np.asarray([0.] * len(subpix_deconvolved))
        x_obs_ac = np.asarray([0.] * len(subpix_deconvolved[0]))

        # Define coordinates to be 0,0 at center of the observed axes (find the central pixel number along each axis)
        if len(x_obs_ac) % 2. == 0:  # if even
            x_ctr = (len(x_obs_ac)) / 2.  # set the center of the axes (in pixel number)
            for i in range(len(x_obs_ac)):
                x_obs_ac[i] = self.resolution * (i - x_ctr) / self.os  # (arcsec/pix) * N_subpix / (subpix/pix) = arcsec
        else:  # elif odd
            x_ctr = (len(x_obs_ac) + 1.) / 2.  # +1 bc python starts counting at 0
            for i in range(len(x_obs_ac)):
                x_obs_ac[i] = self.resolution * ((i + 1) - x_ctr) / self.os
        if len(y_obs_ac) % 2. == 0:
            y_ctr = (len(y_obs_ac)) / 2.
            for i in range(len(y_obs_ac)):
                y_obs_ac[i] = self.resolution * (i - y_ctr) / self.os
        else:
            y_ctr = (len(y_obs_ac) + 1.) / 2.
            for i in range(len(y_obs_ac)):
                y_obs_ac[i] = self.resolution * ((i + 1) - y_ctr) / self.os

        # SET BH OFFSET from center [in arcsec], based on input BH pixel position (*_loc in pixels; *_ctr in subpixels)
        x_bh_ac = (x_loc - x_ctr / self.os) * self.resolution  # [pix - subpix/(subpix/pix)] * [arcsec/pix] = arcsec
        y_bh_ac = (y_loc - y_ctr / self.os) * self.resolution

        # CONVERT FROM ARCSEC TO PHYSICAL UNITS (pc)
        x_bhoff = self.dist * 10 ** 6 * np.tan(x_bh_ac / self.arcsec_per_rad)  # tan(off) = xdisk/dist -> x = d*tan(off)
        y_bhoff = self.dist * 10 ** 6 * np.tan(y_bh_ac / self.arcsec_per_rad)  # 206265 arcsec/rad

        # convert all x,y observed grid positions to pc
        x_obs = np.asarray([self.dist * 10 ** 6 * np.tan(x / self.arcsec_per_rad) for x in x_obs_ac])
        y_obs = np.asarray([self.dist * 10 ** 6 * np.tan(y / self.arcsec_per_rad) for y in y_obs_ac])

        # CONVERT FROM x_obs, y_obs TO 2D ARRAYS OF x_disk, y_disk [arcsec] and [pc] versions
        x_disk_ac = (x_obs_ac[None, :] - x_bh_ac) * np.cos(self.theta) +\
                    (y_obs_ac[:, None] - y_bh_ac) * np.sin(self.theta)  # arcsec
        y_disk_ac = (y_obs_ac[:, None] - y_bh_ac) * np.cos(self.theta) -\
                    (x_obs_ac[None, :] - x_bh_ac) * np.sin(self.theta)  # arcsec
        x_disk = (x_obs[None, :] - x_bhoff) * np.cos(self.theta) + (y_obs[:, None] - y_bhoff) * np.sin(self.theta)  # pc
        y_disk = (y_obs[:, None] - y_bhoff) * np.cos(self.theta) - (x_obs[None, :] - x_bhoff) * np.sin(self.theta)  # pc

        # CALCULATE THE RADIUS (R) OF EACH POINT (x_disk, y_disk) IN THE DISK [pc]
        R_ac = np.sqrt((y_disk_ac ** 2 / np.cos(self.inc) ** 2) + x_disk_ac ** 2)  # radius R [arcsec]
        R = np.sqrt((y_disk ** 2 / np.cos(self.inc) ** 2) + x_disk ** 2)  # radius of each pt in disk (2d array) [pc]
        self.R = R

        # CALCULATE KEPLERIAN VELOCITY DUE TO ENCLOSED STELLAR MASS
        vg = 0  # default to ignoring the gas mass!
        if self.incl_gas:  # If incl_mass, overwrite vg with v_circ,gas estimate, then add it in quadrature to velocity!
            t_gas = time.time()  # Adds ~5s for nr=200, ~13s for nr=500
            vg = self.vcg_func(R)

            if not self.quiet:
                print(time.time() - t_gas, ' seconds spent in gas calculation')

        if self.menc_type == 0:  # if calculating v(R) due to stars directly from MGE parameters
            if not self.quiet:
                print('mge')
            test_rad = np.linspace(np.amin(R_ac), np.amax(R_ac), 100)  # create an array of test radii [arcsec]

            comp, surf_pots, sigma_pots, qobs = mvm.load_mge(self.enclosed_mass)  # load the MGE parameters

            v_c = mvm.mge_vcirc(surf_pots * self.ml_ratio, sigma_pots, qobs, np.rad2deg(self.inc), 0., self.dist,
                                test_rad)  # calculate v_c,star
            v_c_r = interpolate.interp1d(test_rad, v_c, kind='cubic', fill_value='extrapolate')  # interpolate v(R[ac])

            # CALCULATE KEPLERIAN VELOCITY OF ANY POINT (x_disk, y_disk) IN THE DISK WITH RADIUS R (km/s)
            vel = np.sqrt((self.G_pc * self.mbh / R) + v_c_r(R_ac)**2 + vg**2)  # G_pc -> use R [pc], v_c_r needs R [ac]

        elif self.menc_type == 1:  # elif using a file with v_circ(R) due to stellar mass
            if not self.quiet:
                print('v(r)')
            radii = []
            v_circ = []
            with open(self.enclosed_mass) as em:  # current file has units v_c^2/(M/L) --> v_c = np.sqrt(col * (M/L))
                for line in em:
                    cols = line.split()  # note: using Ben's model "B1" = 2nd col in file (file has 4 cols of models)
                    radii.append(float(cols[0]))  # file lists radii in pc
                    v_circ.append(float(cols[1]))  # v^2 / (M/L) --> units (km/s)^2 / (M_sol/L_sol)

            v_c_r = interpolate.interp1d(radii, v_circ, fill_value='extrapolate')  # create function: interpolate v_c(R)

            # CALCULATE KEPLERIAN VELOCITY OF ANY POINT (x_disk, y_disk) IN THE DISK WITH RADIUS R (km/s)
            vel = np.sqrt(v_c_r(R) * self.ml_ratio + (self.G_pc * self.mbh / R) + vg**2)  # velocities sum in quadrature

        elif self.menc_type == 2:  # elif using a file directly with stellar mass as a function of R
            if not self.quiet:
                print('M(R)')
            radii = []
            m_stellar = []
            with open(self.enclosed_mass) as em:
                for line in em:
                    cols = line.split()
                    cols[1] = cols[1].replace('D', 'e')  # for some reason this file uses (e.g.) 0.5D08 instead of 0.5e8
                    radii.append(float(cols[0]) * 10 ** 3)  # file lists radii in kpc; convert to pc
                    m_stellar.append(float(cols[1]))  # solar masses
            m_star_r = interpolate.interp1d(radii, m_stellar, kind='cubic', fill_value='extrapolate')  # create fcn
            ml_const = self.ml_ratio / 7.35  # because mass file assumes a mass-to-light ratio of 7.35
            m_R = self.mbh + ml_const * m_star_r(R)  # function interpolates mass at all radii R (2d array)

            # CALCULATE KEPLERIAN VELOCITY OF ANY POINT (x_disk, y_disk) IN THE DISK WITH RADIUS R (km/s)
            vel = np.sqrt(self.G_pc * m_R / R + vg**2)  # Keplerian velocity vel at each point in the disk

        # CALCULATE LINE-OF-SIGHT VELOCITY AT EACH POINT (x_disk, y_disk) IN THE DISK (km/s)
        alpha = abs(np.arctan(y_disk / (np.cos(self.inc) * x_disk)))  # measure alpha from +x (minor ax) to +y (maj ax)
        sign = x_disk / abs(x_disk)  # (+x now back to redshifted side, so don't need extra minus sign back in front!)
        v_los = sign * abs(vel * np.cos(alpha) * np.sin(self.inc))  # v_los > 0 -> redshift; v_los < 0 -> blueshift

        # INCLUDE NEW RADIAL VELOCITY TERM
        vrad_sign = y_disk / abs(y_disk)  # With this sign convention: vrad > 0 -> outflow; vrad < 0 -> inflow!
        if self.vtype == 'vrad':
            v_los += self.vrad * vrad_sign * abs(np.sin(alpha) * np.sin(self.inc))  # See notebook for derivation!
        elif self.vtype == 'kappa':  # use just kappa
            v_los += self.kappa * vrad_sign * abs(vel * np.sin(alpha) * np.sin(self.inc))
        elif self.vtype == 'omega':  # use omega and kappa both!
            v_los = self.omega * v_los + self.kappa * vrad_sign * abs(vel * np.sin(alpha) * np.sin(self.inc))

        # SET LINE-OF-SIGHT VELOCITY AT THE BLACK HOLE CENTER TO BE 0, SUCH THAT IT DOES NOT BLOW UP
        center = (R == 0.)  # Doing this is only relevant if we have pixel located exactly at the center
        v_los[center] = 0.  # if any point is at x_disk, y_disk = (0., 0.), set velocity there = 0.

        #vkap = abs(self.kappa * vrad_sign * abs(vel * np.sin(alpha) * np.sin(self.inc)))
        #print(vkap[R < 1.6], np.median(vkap), np.amax(vkap), np.amin(vkap))
        #print(np.percentile(vkap, [0.15, 50., 99.85]))  #[16., 50., 84.]))
        #plt.imshow(self.kappa * vrad_sign * abs(vel * np.sin(alpha) * np.sin(self.inc)), origin='lower')
        #plt.colorbar()
        #plt.show()
        #print(oop)

        # CALCULATE VELOCITY PROFILES
        sigma = get_sig(r=R, sig0=self.sig_params[0], r0=self.sig_params[1], mu=self.sig_params[2],
                        sig1=self.sig_params[3])[self.sig_type]

        # CONVERT v_los TO OBSERVED FREQUENCY MAP
        self.freq_obs = (self.f_0 / (1+self.zred)) * (1 - v_los / self.c_kms)  # convert v_los to f_obs

        # CONVERT OBSERVED DISPERSION (turbulent) TO FREQUENCY WIDTH
        sigma_grid = np.zeros(shape=R.shape) + sigma  # make sigma (whether already R-shaped or constant) R-shaped
        self.delta_freq_obs = (self.f_0 / (1 + self.zred)) * (sigma_grid / self.c_kms)  # convert sigma to delta_f

        # WEIGHTS FOR LINE PROFILES: apply weights to gaussian velocity profiles for each subpixel
        self.weight = subpix_deconvolved  # [Jy/beam Hz]

        # WEIGHT CURRENTLY IN UNITS OF Jy/beam * Hz --> need to get it in units of Jy/beam to match data
        self.weight *= self.f_w / np.sqrt(2 * np.pi * self.delta_freq_obs**2)  # divide to get correct units
        # NOTE: MAYBE multiply by fstep here, after collapsing and lucy process, rather than during collapse
        # NOTE: would then need to multiply by ~1000 factor or something large there, bc otherwise lucy cvg too fast

        # BEN_LUCY COMPARISON ONLY (only use for comparing to model with Ben's lucy map, which is in different units)
        if self.bl:  # (multiply by vel_step because Ben's lucy map is actually in Jy/beam km/s units)
            self.weight *= self.fstep * 6.783  # NOTE: 6.783 == beam size / channel width in km/s
            # channel width = 1.537983987579E+07 Hz --> v_width = self.c * (1 - self.f_0 / (self.fstep / (1+self.zred)))

        if not self.quiet:
            print(str(time.time() - t_grid) + ' seconds in grids()')


    def convolution(self):
        # BUILD GAUSSIAN LINE PROFILES!!!
        cube_model = np.zeros(shape=(len(self.freq_ax), len(self.freq_obs), len(self.freq_obs[0])))  # initialize cube
        for fr in range(len(self.freq_ax)):
            cube_model[fr] = self.weight * np.exp(-(self.freq_ax[fr] - self.freq_obs) ** 2 /
                                                  (2 * self.delta_freq_obs ** 2))

        # RE-SAMPLE BACK TO CORRECT PIXEL SCALE (take avg of sxs sub-pixels for real alma pixel) --> intrinsic data cube
        if self.os == 1:
            intrinsic_cube = cube_model
        else:  # intrinsic_cube = block_reduce(cube_model, self.os, np.mean)
            intrinsic_cube = rebin(cube_model, self.os, self.os, avg=False)  # this must use avg=False

        tc = time.time()
        # CONVERT INTRINSIC TO OBSERVED (convolve each slice of intrinsic_cube with ALMA beam --> observed data cube)
        self.convolved_cube = np.zeros(shape=intrinsic_cube.shape)  # approx ~1e-6 to 3e-6s per pixel
        for z in range(len(self.z_ax)):
            self.convolved_cube[z, :, :] = convolution.convolve(intrinsic_cube[z, :, :], self.beam)
        print('convolution loop ' + str(time.time() - tc))


    def chi2(self):
        # CREATE A CLIPPED DATA CUBE SO THAT WE'RE LOOKING AT THE SAME EXACT x,y,z REGION AS IN THE MODEL CUBE
        self.clipped_data = self.input_data[self.zrange[0]:self.zrange[1], self.xyrange[2]:self.xyrange[3],
                                            self.xyrange[0]:self.xyrange[1]]

        # self.convolved_cube *= ell_mask  # mask the convolved model cube
        # self.input_data_masked = self.input_data[self.zrange[0]:self.zrange[1], self.xyrange[2]:self.xyrange[3],
        #                          self.xyrange[0]:self.xyrange[1]] * ell_mask  # mask the input data cube

        # REBIN THE DATA AND MODEL BY THE DOWN-SAMPLING FACTOR: compare data and model in binned groups of dsxds pix
        data_ds = rebin(self.clipped_data, self.ds2, self.ds, avg=self.avg)
        ap_ds = rebin(self.convolved_cube, self.ds2, self.ds, avg=self.avg)

        # CALCULATE THE ELLIPSE MASK DIRECTLY ON THE DOWN-SAMPLED SCALE
        self.ell_ds = ellipse_fitting(data_ds, self.rfit, self.xell / self.ds, self.yell / self.ds2,
                                      self.resolution * self.ds, self.theta_ell, self.q_ell)  # create ellipse mask

        # APPLY THE ELLIPTICAL MASK TO MODEL CUBE & INPUT DATA: ONLY COMPARE DATA & MODEL WITHIN THIS FITTING ELLIPSE!
        data_ds *= self.ell_ds
        ap_ds *= self.ell_ds
        n_pts = np.sum(self.ell_ds) * len(self.z_ax)  # total number of pixels compared in chi^2 calculation!

        chi_sq = 0.  # initialize chi^2
        cs = []  # initialize chi^2 per slice

        z_ind = 0  # the actual index for the model-data comparison cubes
        chi_disk = np.zeros(shape=ap_ds[0].shape)
        for z in range(self.zrange[0], self.zrange[1]):  # for each relevant freq slice (ignore slices with only noise)
            chi_sq += np.sum((ap_ds[z_ind] - data_ds[z_ind])**2 / self.noise[z_ind]**2)  # calculate chisq!
            cs.append(np.sum((ap_ds[z_ind] - data_ds[z_ind])**2 / self.noise[z_ind]**2))  # chisq per slice
            chi_disk += (ap_ds[z_ind] - data_ds[z_ind])**2 / self.noise[z_ind]**2

            z_ind += 1  # the actual index for the model-data comparison cubes

        if self.sqrt2n:
            chi_sq /= np.sqrt(2. * n_pts)

        if n_pts == 0.:  # PROBLEM WARNING
            print(self.resolution, self.xell, self.yell, self.theta_ell, self.q_ell, self.rfit)
            print('WARNING! STOP! There are no pixels inside the fitting ellipse! n_pts = ' + str(n_pts))
            return np.inf

        if not self.quiet:
            print(np.sum(self.ell_ds), len(self.z_ax), n_pts)
            print(r'chi^2=', chi_sq)
            print('reduced = ', chi_sq / (n_pts - self.n_params))
            print(r'dof =', n_pts - self.n_params)

        if self.reduced:  # CALCULATE REDUCED CHI^2, RETURN BOTH REDUCED & FULL
            return chi_sq, chi_sq / (n_pts - self.n_params)  # convert to reduced chi^2; else just return full chi^2

        else:  # RETURN FULL CHI^2
            return chi_sq  # Reduced or Not depending on reduced = True or False


    def scaling_rels(self, rel=0, err_type='systematic', magsolH=3.37, magsolK=3.27, magsolV=4.87, HK=0.2, VK=3.1):
        """
        Print scaling relations!
        TO DO: add rel=2 (mbh-bulge mass relation)
        TO DO: move this to dynesty_out.py

        :param rel: which scaling relation! rel=0: mbh-sigma. rel=1: mbh-lum. rel=2: mbh-mass.
        :param err_type: errors on UGC 2698 to plot ('systematic' or 'statistical')
        :param magsolH: assumed absolute magnitude of the Sun in the H-band
        :param magsolK: assumed absolute magnitude of the Sun in the K-band
        :param magsolV: assumed absolute magnitude of the Sun in the V-band
        :param HK: assumed H-K for the galaxy
        :param VK: assumed V-K for the galaxy
        :return:
        """

        # UGC 2698 calculations
        lum_H = mge_sum(self.enclosed_mass, self.pc_per_ac)
        abs_H = magsolH - 2.5 * np.log10(lum_H)  # MsolH = 3.37
        abs_K = abs_H - HK  # H-K = 0.2 -> K = H-0.2
        lum_K = 10 ** (0.4 * (magsolK - abs_K))  # LK / Lsol = 10^(.4(MsolK - MK))
        LK_fromH = lum_H * 10 ** (0.4 * (HK + magsolK - magsolH))
        Mgal = lum_H * self.ml_ratio
        # MsolK: 3.27, MsolH: 3.37,
        # print(lum_K, lum_H, self.ml_ratio, Mgal)  # 237030217843.6 216174127928.3 1.7034101349154922 368233200419.5
        # print(oop)

        #print(mge_sum('mge_mrk1216.txt', 94e6 / 206265), 'mrk1216')  # 117014480545.67628
        #print(mge_sum('mge_ngc1271.txt', 80e6 / 206265), 'ngc1271')  # 70176058033.44998
        #print(mge_sum('mge_ngc1277.txt', 71e6 / 206265), 'ngc1277')  # 17491088092.82888 # -> lum_H = 2.77e11
        #print(oop)
        # pc_per_ac = self.dist * 1e6 / self.arcsec_per_rad
        # take dists from Yildirim+2017 Table 1: MRK1216 d=94Mpc. NGC1271 d=80Mpc. NGC1277 d=71Mpc.

        # FROM WALSH+2017:
        # (Conclusion): MRK 1216 total lum L_K = 1.4e11, total stellar mass = 1.6e11
        # (Sec7): NGC 1277 total lum L_V = 1.7e10 (L_K=, total stellar mass = 1.6e11
        # (Sec7): NGC 1271 total lum L_H = 7.2e10, total stellar mass = 1.0e11


        # '''  #
        # OLD
        # lum_H1271 = 7.2e10
        # lum_V1277 = 1.7e10
        # lum_K1216 = 1.4e11

        # L_V 1277: 0.18147530956875887 e11
        # L_H 1271: 0.7476219879607321 e11
        # L_H 1216: 1.225292156991971 e11
        # Take V - K = 3.1
        # H - K = 0.2

        lum_H1271 = 7.476219879607321e10
        lum_V1277 = 1.8147530956875887e10
        lum_H1216 = 1.225292156991971e11
        #LK_fromV1277 = lum_V1277 * 10 ** (0.4 * 3.)  # V-K = 3.0
        # print(LK_fromV1277, lum_V1277 * 10 ** (0.4 * 3.1))  # 269431842718.3894 295426140887.394
        #LK_fromH1271 = lum_H1271 * 10 ** (0.4 * 0.2)  # H-K = 0.2

        abs_H1216 = magsolH - 2.5 * np.log10(lum_H1216)  # MsolH = 3.37
        abs_K1216 = abs_H1216 - HK  # H-K = 0.2 -> K = 0.2-H
        lum_K1216 = 10 ** (0.4 * (magsolK - abs_K1216))  # LK / Lsol = 10^(.4(MsolK - MK))
        LK_fromH1216 = lum_H1216 * 10 ** (0.4 * (HK + magsolK - magsolH))

        # abs_mag_H = abs_mag_sun_H - 2.5 D * alog10(totlum_H)
        abs_H1271 = magsolH - 2.5 * np.log10(lum_H1271)  # MsolH = 3.37
        abs_K1271 = abs_H1271 - HK  # H-K = 0.2 -> K = 0.2-H
        lum_K1271 = 10 ** (0.4 * (magsolK - abs_K1271))  # LK / Lsol = 10^(.4(MsolK - MK))
        LK_fromH1271 = lum_H1271 * 10 ** (0.4 * (HK + magsolK - magsolH))

        abs_V1277 = magsolV - 2.5 * np.log10(lum_V1277)  # MsolV = 4.83
        abs_K1277 = abs_V1277 - VK  # V-K = 3.0
        lum_K1277 = 10 ** (0.4 * (magsolK - abs_K1277))  # MsolK = 3.27
        LK_fromV1277 = lum_V1277 * 10 ** (0.4 * (VK + magsolK - magsolV))  # for LK = 1e11, need x~3.484
        # need (V-K) + magsolK - magsolV ~= 1.924
        # NOTE: actual Mass 1277 = 1.581e11
        # NOTE: Based on total LK, MRK 1216 total LH = 1.06200861e11, with M/L = 1.3. Hmm need LH~1.23e11
        ## 1.23e11 * 10^(0.4(0.2 + 3.27-3.37)) = 1.349e11 = LK.
        ## Using LK = 1.4e11, get LH = 1.28e11
        # print(lum_K/1e11, lum_K1277/1e11, lum_K1271/1e11)
        # print(LK_fromH/1e11, LK_fromV1277/1e11, LK_fromH1271/1e11)

        #print(lum_K/1e11, lum_K1216/1e11, lum_K1277/1e11, lum_K1271/1e11)  # 2.370302178436111 1.3435061340469487 0.7224662201773862 0.8197512087561656
        #print(LK_fromH/1e11, LK_fromH1216/1e11, LK_fromV1277/1e11, LK_fromH1271/1e11)  # 2.370302178436111 1.3435061340469487 0.7224662201773862 0.8197512087561656
        #print(oop)
        # print(LK_fromH1271)  # 86563039292.45374mge_sum
        mbh_relation_full(mbhs=[self.mbh, 4.9e9, 3e9, 4.9e9],
                          sigmas=[304, 317, 295, 308],
                          lum_ks=[lum_K, lum_K1277, lum_K1271, LK_fromH1216],  # [lum_K, lum_K1277, lum_K1271, 1.4e11],  # [2.5e11, 7.7e10, 8.1e10, 1.3e11]  # 7.7e10, 4.6e10, 9.6e10],
                          masses=[Mgal, 1.68772037899e11, 1.04667078315e11, 1.59287980409e11],
                          # [Mgal, 1.67e11, 1.0e11, 1.6e11],  # [Mgal, 1.6e11, 1.0e11, 1.6e11]
                          # [3.8e11, 1.3e11, 1.1e11, 2.2e11],  # 1.6e11, 1.0e11, 1.1e11]
                          mbh_errs=[[0.78e9, 0.70e9], [1.6e9, 1.6e9], [1.1e9, 1e9], [1.7e9, 1.7e9]],
                          sigma_errs=[[6, 6], [5, 5], [6, 6], [7, 7]],
                          lum_errs=[[0.,0.], [0,0], [0,0], [0,0]],  # [2.8e10, 2.8e10], [2.7e10, 2.7e10], [7.9e10, 4.6e10]],  # TO DO FIX 0s!
                          mass_errs=[[0.,0.], [0.,0.], [0.,0.], [0,0]],  # [0.9e11, 0.5e11]],  # TO DO FIX 0s!
                          galaxies=['U2698', 'N1277', 'N1271', 'M1216'],
                          fmts=['bD', 'rs', 'rs', 'rs'],  # ['bD', 'rs', 'mo', 'k*'],  # ['bD', 'rs', 'rs', 'rs'],
                          msig_locs=[[310, 1.5e9], [318, 6.5e9], [247, 3.2e9], [257, 5.4e9]],
                          ml_locs=[[1.1e11, 1.6e9], [3.4e10, 5.65e9], [3.9e10, 2e9], [1.4e11, 5.8e9]],
                          mm_locs=[[1.7e11, 2.7e9], [1.85e11, 5.2e9], [4.5e10, 2e9], [6.75e10, 5.65e9]],
                          # [[1.7e11, 2.8e9], [1.8e11, 5e9], [4.9e10, 1.9e9], [7.5e10, 5.65e9]],
                          savefig='new_mge_calcs.png')  #  'scaling_rels_10_20_uplims_a.png')  # _b.png'
        # mbhs, mbh_errs, lum_ks, lum_errs, masses, mass_errs, sigmas, sigma_errs, galaxies, fmts, ml_locs, msig_locs, mm_locs
        print(oop)

        # msig_locs=[[310, 1.5e9], [318, 6.5e9], [247, 3.2e9], [257, 5.4e9]],
        # ml_locs=[[1.15e11, 2.3e9], [1.4e11, 5.5e9], [4e10, 2e9], [3.5e10, 5.5e9]],
        # mm_locs=[[1.6e11, 2.3e9], [2.4e11, 5e9], [4.8e10, 3.1e9], [5.5e10, 5.5e9]],

        # '''  #

        if err_type == 'statistical':
            err_low = 5.5e7
            err_hi = 5.6e7
        else:
            err_low = 0.78e9
            err_hi = 0.70e9

        if rel == 0:
            mbh_relations([self.mbh], [[err_low, err_hi]], sigma=[304], xerr=[[6], [6]])
            # mbh_relations([10 ** 9.39], [[0.71e9, 0.7e9]], sigma=[304], xerr=[[6], [6]])

        elif rel == 1:
            lum_H = mge_sum(self.enclosed_mass, self.pc_per_ac)
            # lum_H = 10**11.3348
            LKcorr = lum_H * 10 ** (0.4 * 0.2)
            # print(lum_H, LKcorr)
            mbh_relations([self.mbh], [[err_low, err_hi]], lum_k=[LKcorr])
            # mbh_relations([10 ** 9.39], [[0.71e9, 0.7e9]], lum_k=[LKcorr])
        elif rel == 2:
            lum_H = mge_sum(self.enclosed_mass, self.pc_per_ac)
            M_H = lum_H * self.ml_ratio
            mbh_relations([self.mbh], [[err_low, err_hi]], mass_k=[M_H])
            print(M_H, self.ml_ratio)

        # def mbh_relations(mbh, mbh_err, lum_k=None, mass_k=None, sigma=None, incl_past=True):
        print('done')


    def mge_sbprof(self):
        """
        Calculate mass profile of the MGE along the major axis

        :param mge: MGE file, containing columns: j [component number], I [Lsol,H/pc^2], sigma[arcsec], q[unitless]
        :param pc_per_ac: parsec per arcsec scaling
        :return: the total luminosity from integrating the MGE
        """

        intensities = []
        sigmas = []
        qs = []
        with open(self.enclosed_mass, 'r') as mfile:
            for line in mfile:
                if not line.startswith('#'):
                    cols = line.split()
                    intensities.append(float(cols[1]))  # cols[0] = component number, cols[1] = intensity[Lsol,H/pc^2]
                    sigmas.append(float(cols[2]))  # cols[2] = sigma[arcsec]
                    qs.append(float(cols[3]))  # cols[3] = qObs[unitless]

        gaus = []

        rad = np.logspace(-1., 0.5, 25)  # arcsec

        for i in range(len(intensities)):
            # volume under 2D gaussian function = 2 pi A sigma_x sigma_y
            # gaus.append(2 * np.pi * intensities[i] * qs[i] * (self.pc_per_ac * sigmas[i]) ** 2)  # convert sigma to pc

            area = np.pi * (rad * self.pc_per_ac)**2 * qs[i]  # pc^2
            gaus.append(intensities[i] * np.exp(-rad**2 / (2 * sigmas[i]**2)) * self.ml_ratio * area)

        for i in range(len(gaus)):
            plt.plot(rad, gaus[i], 'k--')
        plt.axhline(y=self.mbh)
        plt.show()

        print(oop)

    def line_profiles(self, ix, iy, show_freq=False):  # compare line profiles at the given indices ix, iy
        f_sys = self.f_0 / (1 + self.zred)
        print(ix, iy)
        data_ds = rebin(self.clipped_data, self.ds2, self.ds, avg=self.avg)
        ap_ds = rebin(self.convolved_cube, self.ds2, self.ds, avg=self.avg)

        hdu_m = fits.open(self.data_mask)
        data_mask = hdu_m[0].data  # The mask is stored in hdu_m[0].data, NOT hdu_m[0].data[0]
        v_width = 2.99792458e5 * (1 + (6454.9 / 2.99792458e5)) * self.fstep / self.f_0  # velocity width [km/s] = c*(1+v/c)*fstep/f0
        mask_ds = rebin(data_mask[self.zrange[0]:self.zrange[1], self.xyrange[2]:self.xyrange[3],
                        self.xyrange[0]:self.xyrange[1]], self.ds2, self.ds, avg=self.avg)

        collapse_flux_v = np.zeros(shape=(len(data_ds[0]), len(data_ds[0][0])))
        for zi in range(len(data_ds)):
            collapse_flux_v += data_ds[zi] * mask_ds[zi] * v_width
            # self.clipped_data[zi] * data_mask[zi, self.xyrange[2]:self.xyrange[3], self.xyrange[0]:self.xyrange[1]]* v_width

        '''  #
        #plt.imshow(self.weight, origin='lower')
        #plt.show()
        fig = plt.figure()
        ax = plt.gca()
        plt.imshow(self.ell_ds, origin='lower')
        plt.colorbar()
        from matplotlib import patches
        e1 = patches.Ellipse((self.xell / self.ds, self.yell / self.ds), 2 * self.rfit / (self.resolution * self.ds), 2 * self.rfit / (self.resolution * self.ds) * self.q_ell,
                             angle=np.rad2deg(self.theta_ell),
                             linewidth=2, edgecolor='w', fill=False)  # np.rad2deg(params['theta_ell'])
        print(e1)
        ax.add_patch(e1)
        plt.plot(self.xell / self.ds, self.yell / self.ds, 'w*')
        #plt.title(r'q = ' + str(self.q_ell) + r', PA = ' + str(params['theta_ell']) + ' deg, rfit = ' + str(params['rfit'])
        #          + ' arcsec')
        plt.show()
        #print(oop)
        # '''  #

        '''  #
        plt.imshow(collapse_flux_v * self.ell_ds, origin='lower')
        cbar = plt.colorbar()
        cbar.set_label(r'Jy km s$^{-1}$ beam$^{-1}$', rotation=270, labelpad=20.)
        ax = plt.gca()
        from matplotlib import patches
        print(2 * self.rfit / (self.resolution * self.ds))
        print(2 * self.rfit / (self.resolution * self.ds2) * self.q_ell)
        print(self.xell, self.yell)

        e1 = patches.Ellipse((self.xell / self.ds, self.yell / self.ds2), 2 * self.rfit / (self.resolution * self.ds), 2 * self.rfit / (self.resolution * self.ds2) * self.q_ell,
                             angle=np.rad2deg(self.theta_ell), linewidth=2, edgecolor='w', fill=False)
        ax.add_patch(e1)
        #plt.plot(10.7125, 8.225, 'w*')  # ix, iy, 'w*'
        for i in range(len(ix)):
            plt.plot(ix[i], iy[i], 'w*')
        plt.show()
        #plt.imshow(ap_ds[20], origin='lower')
        #plt.show()
        #print(oop)
        # '''  #
        if show_freq:
            plt.plot(self.freq_ax / 1e9, ap_ds[:, iy, ix], 'r*', label=r'Model')
            plt.plot(self.freq_ax / 1e9, data_ds[:, iy, ix], 'k+', label=r'Data')
            plt.plot(self.freq_ax / 1e9, self.noise, 'k--', label=r'Noise (std)')
            plt.axvline(x=f_sys / 1e9, color='k', label=r'$f_{sys}$')
            plt.xlabel(r'Frequency [GHz]')
        else:
            vel_ax = []
            for v in range(len(self.freq_ax)):
                vel_ax.append(self.c_kms * (1. - (self.freq_ax[v] / self.f_0) * (1 + self.zred)))
            dv = vel_ax[1] - vel_ax[0]
            #vel_ax.insert(0, vel_ax[0])
            #plt.errorbar(vel_ax, data_ds[:, iy, ix], yerr=self.noise, color='k', marker='+', label=r'Data')
            plt.fill_between(vel_ax, data_ds[:, iy, ix] - self.noise, data_ds[:, iy, ix] + self.noise, color='k',
                             step='mid', alpha=0.3)
            plt.step(vel_ax, data_ds[:, iy, ix], color='k', where='mid', label=r'Data')  # width=vel_ax[1] - vel_ax[0], alpha=0.4
            #plt.plot(vel_ax + dv/2., data_ds[:, iy, ix], ls='steps', color='k', label=r'Data')  # width=vel_ax[1] - vel_ax[0], alpha=0.4
            #plt.plot(vel_ax, ap_ds[:, iy, ix], color='r', marker='+', ls='none', label=r'Model')  # 'r+'
            plt.step(vel_ax, ap_ds[:, iy, ix], color='b', where='mid', label=r'Model')  # width=vel_ax[1] - vel_ax[0], alpha=0.5
            plt.axvline(x=0., color='k', ls='--', label=r'v$_{\text{sys}}$')
            # plt.xlabel(r'Line-of-sight velocity [km/s]')
            import matplotlib as mpl
            mpl.rcParams['text.usetex'] = True
            mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']  # for \text command
            plt.xlabel(r'Velocity $v_{\text{LOS}} - v_{\text{sys}}$ [km/s]')
        plt.legend()
        plt.ylabel(r'Flux Density [Jy/beam]')
        plt.show()


    def pvd(self):
        from pvextractor import pvextractor
        from pvextractor.geometry import path
        from pvextractor import extract_pv_slice
        print(len(self.clipped_data[0]), len(self.clipped_data[0][0]), len(self.clipped_data))
        # path1 = path.Path([(0, 0), (len(self.clipped_data[0]), len(self.clipped_data[0][0]))], width=self.pvd_width)

        xc = self.x_loc - self.xyrange[0]  # x_loc - xi
        yc = self.y_loc - self.xyrange[2]  # y_loc - yi
        extend = 40
        x0 = xc - extend * np.cos(np.deg2rad(self.theta))
        xf = xc + extend * np.cos(np.deg2rad(self.theta))
        y0 = yc - extend * np.sin(np.deg2rad(self.theta))
        yf = yc + extend * np.sin(np.deg2rad(self.theta))

        # path1 = path.Path([(self.xyrange[0], self.xyrange[2]), (self.xyrange[1], self.xyrange[3])], width=self.pvd_width)
        path1 = path.Path([(x0, y0), (xf, yf)], width=self.pvd_width)
        # [(x0,y0), (x1,y1)]
        print(self.input_data.shape)
        # pvd_dat, slice = extract_pv_slice(self.input_data[self.zrange[0]:self.zrange[1]], path1)
        pvd_dat, slice = extract_pv_slice(self.clipped_data, path1)
        print(self.convolved_cube.shape)
        # path2 = path.Path([(0, 0), (self.xyrange[1] - self.xyrange[0], self.xyrange[3] - self.xyrange[2])], width=self.pvd_width)
        pvd_dat2, slice2 = extract_pv_slice(self.convolved_cube, path1)  # path2
        print(slice)

        vel_ax = []
        for v in range(len(self.freq_ax)):
            vel_ax.append(self.c_kms * (1. - (self.freq_ax[v] / self.f_0) * (1 + self.zred)))

        x_rad = np.zeros(shape=len(slice[0]))
        if len(slice[0]) % 2. == 0:  # if even
            xr_c = (len(slice[0])) / 2.  # set the center of the axes (in pixel number)
            for i in range(len(slice[0])):
                x_rad[i] = self.resolution * (i - xr_c) # (arcsec/pix) * N_pix = arcsec
        else:  # elif odd
            xr_c = (len(slice[0]) + 1.) / 2.  # +1 bc python starts counting at 0
            for i in range(len(slice[0])):
                x_rad[i] = self.resolution * ((i + 1) - xr_c)

        #from mpl_toolkits.axes_grid1 import make_axes_locatable
        #divider1 = make_axes_locatable(ax[0])
        # CONVERT FROM Jy/beam TO mJy/beam
        slice *= 1e3 * self.pvd_width
        slice1 = slice
        slice2 *= 1e3 * self.pvd_width
        vmin = np.amin([slice, slice2])
        vmax = np.amax([slice, slice2])

        '''  # RMS REGIONS
        print(slice2.shape)  # 57, 79
        regs = [[46, 56, 1, 11], [33, 43, 13, 23], [13, 23, 55, 65], [1, 11, 67, 77]]
        #plt.imshow(slice, origin='lower')
        #plt.plot([regs[3][0], regs[3][1]], [regs[3][2], regs[3][2]], 'k-')
        #plt.plot([regs[3][0], regs[3][1]], [regs[3][3], regs[3][3]], 'k-')
        #plt.plot([regs[3][0], regs[3][0]], [regs[3][2], regs[3][3]], 'k-')
        #plt.plot([regs[3][1], regs[3][1]], [regs[3][2], regs[3][3]], 'k-')
        #plt.show()
        #print(oop)

        rmsds = []  # CALCULATE RMS DEVIATION IN 4 NOISE REGIONS
        for reg in regs:
            print(reg, reg[0], reg[1], reg[2], reg[3])
            noisereg = slice1[reg[0]:reg[1], reg[2]:reg[3]]
            n = np.shape(noisereg)[0] * np.shape(noisereg)[1]
            print(noisereg.shape, n)
            rms = np.sqrt((1. / n) * (np.sum(noisereg ** 2)))
            rms_deviation = np.sqrt((1. / n) * np.sum((np.mean(noisereg) - noisereg) ** 2))
            print(rms, rms_deviation)
            rmsds.append(rms_deviation)
        rmsds = np.asarray(rmsds)
        print(np.mean(rmsds))  # 2.15210524474533
        # print(oop)

        # MAX VELOCITIES DETECTED [detection = signal >= 3 * mean(rms deviation)]
        vel_ax = np.asarray(vel_ax)
        for col in range(20, 64):  #36 ;; 28, 53  # 79 cols -> midpoint 39.5 (39 = 40th index): 35,45 = 35,36,37,38,39,40,41,42,43,44
            signal_where = np.where(slice[:, col] > 2*np.mean(rmsds))[0]
            # print(signal_where, slice[:, col])
            vel_max = vel_ax[signal_where]
            # print(vel_ax)
            # print(vel_max, signal_where, slice[:, col])
            print(col, np.amax(vel_max), np.amin(vel_max), x_rad[col])
        # max: 474.75214482, -479.95001906
        print(oop)
        
        # RESIDUAL LEVELS
        pvddatblah, slicemask = extract_pv_slice(self.convolved_cube, path1)  # path2
        slicemask *= 1e3 * self.pvd_width
        slicemask[slicemask < 2.] = 0.
        print(np.percentile(slicemask[slicemask!=0.], [16., 50., 84.]))
        slicemask[slicemask > 0.] = 1.
        signal = slice * slicemask
        print(np.percentile(signal[signal != 0], [16., 50., 84.]), 'hi')
        sfrac = abs((slice - slice2) / slice)
        print(np.amax(slice))
        # sfrac[abs(sfrac) >= 100] = 0.
        sfrac *= slicemask
        print(np.percentile(sfrac[sfrac!=0.], [16., 50., 84.]))
        plt.imshow(sfrac, origin='lower')
        plt.colorbar()
        plt.show()
        print(oop)
        # '''

        fig, ax = plt.subplots(3, 1, sharex=True, figsize=(6, 12))
        plt.subplots_adjust(hspace=0.02)
        p1 = ax[0].pcolormesh(x_rad, vel_ax, slice, vmin=vmin, vmax=vmax, cmap='Greys')  # x_rad[0], x_rad[-1]
        fig.colorbar(p1, ax=ax[0], pad=0.02)  # ticks=[-0.5, 0, 0.5, 1, 1.5],
        #ax[0].imshow(slice, origin='lower', extent=[x_rad[0], x_rad[-1], vel_ax[0], vel_ax[-1]])  # x_rad[0], x_rad[-1]
        #cax1 = divider1.append_axes('right', size='5%', pad=0.05)
        #fig.colorbar(im1, cax=cax1, orientation='vertical')
        p2 = ax[1].pcolormesh(x_rad, vel_ax, slice2, vmin=vmin, vmax=vmax, cmap='Greys')  # x_rad[0], x_rad[-1]
        cb2 = fig.colorbar(p2, ax=ax[1], pad=0.02)  # ticks=[-0.5, 0, 0.5, 1, 1.5],
        cb2.set_label(r'mJy beam$^{-1}$', rotation=270, labelpad=20.)
        #ax[1].imshow(slice2, origin='lower', extent=[x_rad[0], x_rad[-1], vel_ax[0], vel_ax[-1]])
        #ax[1].colorbar()

        # p3 = ax[2].pcolormesh(x_rad, vel_ax, slice - slice2, vmin=np.amin([vmin, slice - slice2]), vmax=vmax)
        p3 = ax[2].pcolormesh(x_rad, vel_ax, slice - slice2, vmin=np.amin(slice - slice2), vmax=np.amax(slice - slice2),
                              cmap='Greys')
        fig.colorbar(p3, ax=ax[2], pad=0.02)  # ticks=[-1., -0.5, 0, 0.5, 1],
        #ax[2].imshow(slice - slice2, origin='lower', extent=[x_rad[0], x_rad[-1], vel_ax[0], vel_ax[-1]])
        #ax[2].colorbar()

        #ax[2].set_xticks([2, 27, 52, 77, 102])
        #ax[2].set_xticklabels([x_rad[2], x_rad[27], x_rad[52], x_rad[77], x_rad[102]])

        import matplotlib as mpl
        mpl.rcParams['text.usetex'] = True
        mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']  # for \text command
        ax[1].set_ylabel(r'$v_{\text{LOS}} - v_{\text{sys}}$ [km s$^{-1}$]')
        plt.xlabel('Distance [arcsec]')
        #plt.colorbar()
        plt.show()


    def output_cube(self):  # if outputting actual cube itself
        if not Path(self.out_name).exists():  # WRITE OUT RESULTS TO FITS FILE
            hdu = fits.PrimaryHDU(self.convolved_cube)
            hdul = fits.HDUList([hdu])
            hdul.writeto(self.out_name)
            print('written!')


    def test_ellipse(self):
        # USE BELOW FOR TESTING
        cf = rebin(rebin(self.weight, self.os, self.os), self.ds2, self.ds)[0]  # re-binned weight map, for reference
        plt.imshow(self.ell_ds * cf, origin='lower')  # masked weight map
        plt.title('4x4-binned ellipse * weight map')
        plt.colorbar()
        plt.show()

        plt.imshow(cf, origin='lower')  # re-binned weight map by itself, for reference
        plt.title('4x4-binned weight map')
        plt.colorbar()
        plt.show()


    def just_the_bins(self, snr, pars_backup=None):
        """
        Calculate voronoi bins from flux map

        :param snr: target Signal-to-Noise Ratio
        :return:
        """
        try:
            params
        except NameError:
            print("params is undefined!")
            params = pars_backup
        print(params, self.input_data.shape)
        hdu_m = fits.open(self.data_mask)
        data_mask = hdu_m[0].data  # The mask is stored in hdu_m[0].data, NOT hdu_m[0].data[0]
        hdu_m.close()

        # estimate the 2D collapsed signal, and estimate a constant noise
        sig = np.zeros(shape=self.input_data[0].shape)
        noi = 0
        for z in range(len(self.input_data)):
            sig += self.input_data[z] * data_mask[z] / len(self.input_data)
            noi += np.mean(self.input_data[z, params['yerr0']:params['yerr1'], params['xerr0']:params['xerr1']]) / \
                   len(self.input_data)
        from vorbin.voronoi_2d_binning import voronoi_2d_binning

        sig = sig[self.xyrange[2]:self.xyrange[3], self.xyrange[0]:self.xyrange[1]]
        print(noi)

        signal_input = []
        noise_input = []
        x_in = []  # used as arcsec-scale input
        y_in = []  # used as arcsec-scale input
        xpix = []  # just store pixel number
        ypix = []  # just store pixel number
        if len(sig) % 2. == 0:  # if even
            yctr = (len(sig)) / 2.  # set the center of the axes (in pixel number)
        else:  # elif odd
            yctr = (len(sig) + 1.) / 2.  # +1 bc python starts counting at 0
        if len(sig[0]) % 2 == 0.:
            xctr = (len(sig[0])) / 2.  # set the center of the axes (in pixel number)
        else:  # elif odd
            xctr = (len(sig[0]) + 1.) / 2.  # +1 bc python starts counting at 0

        for yy in range(len(sig)):
            for xx in range(len(sig[0])):
                if sig[yy, xx] != 0:  # don't include pixels that have been masked out!
                    xpix.append(xx)
                    ypix.append(yy)
                    x_in.append(xx - xctr)  # pixel scale, with 0 at center
                    y_in.append(yy - yctr)  # pixel scale, with 0 at center
                    noise_input.append(noi)
                    signal_input.append(sig[yy, xx])

        target_snr = snr
        signal_input = np.asarray(signal_input)
        noise_input = np.asarray(noise_input)
        x_in = np.asarray(x_in) * self.resolution  # convert to arcsec-scale
        y_in = np.asarray(y_in) * self.resolution  # convert to arcsec-scale

        # Perform voronoi binning! The vectors (binNum, xNode, yNode, xBar, yBar, sn, nPixels, scale) are *output*
        binNum, xNode, yNode, xBar, yBar, sn, nPixels, scale = voronoi_2d_binning(x_in, y_in, signal_input, noise_input,
                                                                                  target_snr, plot=0, quiet=1)
        # plt.show()  # don't plot
        # print(binNum, sn, nPixels)  # bin num for each pix [len=num pixels]; SNR/bin [num bins]; pix/bin [num bins]

        return xpix, ypix, binNum, x_in, nPixels


    def vor_moms(self, incl_beam, snr=10, just_data=False, fs=20, pars_backup=None, frac=False):
        """
        Calculate moment maps, average them within voronoi bins
        # using equations from https://www.atnf.csiro.au/people/Tobias.Westmeier/tools_hihelpers.php#moments

        :param incl_beam: True or False; if True, show absolute value of the residual in moment 0 data panel
        :param snr: voronoi binning target signal-to-noise ratio
        :param just_data: if True, plot just the moments of the data (don't include model and residual)
        :return:
        """
        # OPEN STRICTMASK
        hdu_m = fits.open(self.data_mask)
        data_mask = hdu_m[0].data  # The mask is stored in hdu_m[0].data, NOT hdu_m[0].data[0]
        hdu_m.close()

        # CREATE VELOCITY AXIS FROM FREQUENCY AXIS
        vel_ax = []
        velwidth = self.c_kms * (1 + self.zred) * self.fstep / self.f_0
        for v in range(len(self.freq_ax)):
            vel_ax.append(self.c_kms * (1. - (self.freq_ax[v] / self.f_0) * (1 + self.zred)))

        # full cube strictmask, clipped to the appropriate zrange
        clipped_mask = data_mask[self.zrange[0]:self.zrange[1], self.xyrange[2]:self.xyrange[3],
                                 self.xyrange[0]:self.xyrange[1]]

        # CALCULATE MOMENT 0 for data, for voronoi-binned lucy-input flux map
        data_lucy_flux = np.zeros(shape=self.convolved_cube[0].shape)
        for z in range(len(vel_ax)):
            data_lucy_flux += abs(self.fstep) * self.clipped_data[z] * clipped_mask[z]  # SUM_z data[z] * mask[z] * dz

        # CALCULATE MOMENT 0 for data, then for model
        data_masked_m0 = np.zeros(shape=self.convolved_cube[0].shape)
        for z in range(len(vel_ax)):
            data_masked_m0 += abs(velwidth) * self.clipped_data[z] * clipped_mask[z]  # SUM_z data[z] * mask[z] * dz

        model_masked_m0 = np.zeros(shape=(len(self.convolved_cube[0]), len(self.convolved_cube[0][0])))
        for zi in range(len(self.convolved_cube)):
            model_masked_m0 += self.convolved_cube[zi] * abs(velwidth) * clipped_mask[zi]  # SUM_z model[z]*mask[z]*dz

        # CONVERT TO mJy
        data_masked_m0 *= 1e3
        model_masked_m0 *= 1e3

        subtr = 0.
        if incl_beam:  # if including beam overlay
            # TO DO: beam overlay not working right now?!
            beam_overlay = np.zeros(shape=self.convolved_cube[0].shape)  # overlay the beam on the same scale as the moment map
            # print(self.beam.shape, beam_overlay.shape)
            # print(self.beam.shape, self.convolved_cube.shape)
            # beam_overlay[:self.beam.shape[0], (beam_overlay.shape[1] - self.beam.shape[1]):] = self.beam
            ebeam = patches.Ellipse((-0.65,-0.5), params['x_fwhm'], params['y_fwhm'],
                                    angle=-(90 + params['PAbeam']), linewidth=2, edgecolor='w', fill=False)
            # beam_overlay[:self.beam.shape[0] - 6, (beam_overlay.shape[1] - self.beam.shape[1]) + 6:] = self.beam[6:,:-6]
            beam_overlay[:21, 63:] = self.beam[5:26, 5:26]
            print(beam_overlay.shape, self.beam.shape)
            beam_overlay *= np.amax(data_masked_m0) / np.amax(beam_overlay)  # scale so beam shows up well on colormap
            #data_masked_m0 += beam_overlay  # display on the moment 0 data panel
            #subtr = beam_overlay

        # CALCULATE RESIDUAL
        residual_m0 = (data_masked_m0 - subtr) - model_masked_m0
        residual_frac_m0 = np.nan_to_num(((data_masked_m0 - subtr) - model_masked_m0) / data_masked_m0, nan=10.)

        # AVERAGE EACH MAP WITHING THE VORONOI BIN -- CALCULATE THE VORONOI BINS!
        xpix, ypix, binNum, x_in, nPixels = self.just_the_bins(snr=snr, pars_backup=pars_backup)

        # AVERAGE lucy-in fluxmap within voronoi bins
        dlucy = map_averaging(data_lucy_flux, xpix, ypix, binNum, x_in, nPixels)
        dlucy_full = np.zeros(shape=self.lucy_out.shape)
        dlucy_full[self.xyrange[2]:self.xyrange[3], self.xyrange[0]:self.xyrange[1]] = dlucy
        dl_name = '/Users/jonathancohn/Documents/dyn_mod/ugc_2698/ugc_2698_fluxmap_20.3kms_strictmask2_voronoibin.fits'
        if not Path(dl_name).exists():  # if voronoi-binned lucy_in file does not exist, create it!
            hdu = fits.PrimaryHDU(dlucy_full)
            hdul = fits.HDUList([hdu])
            hdul.writeto(dl_name)
            hdul.close()
            print(oop)

        # CALCULATE NUMERATOR AND DENOMINATOR USED IN MOMENT 1 & 2, FOR MODEL THEN FOR DATA
        model_numerator = np.zeros(shape=(len(self.convolved_cube[0]), len(self.convolved_cube[0][0])))
        model_denominator = np.zeros(shape=(len(self.convolved_cube[0]), len(self.convolved_cube[0][0])))
        for zi in range(len(self.convolved_cube)):
            model_numerator += vel_ax[zi] * self.convolved_cube[zi] * clipped_mask[zi]
            model_denominator += self.convolved_cube[zi] * clipped_mask[zi]
        model_m1 = model_numerator / model_denominator

        data_numerator = np.zeros(shape=(len(self.convolved_cube[0]), len(self.convolved_cube[0][0])))
        data_denominator = np.zeros(shape=(len(self.convolved_cube[0]), len(self.convolved_cube[0][0])))
        for zi in range(len(self.convolved_cube)):
            data_numerator += vel_ax[zi] * self.clipped_data[zi] * clipped_mask[zi]
            data_denominator += self.clipped_data[zi] * clipped_mask[zi]
        data_m1 = data_numerator / data_denominator

        # CALCULATE MOMENT 2
        m2_num = np.zeros(shape=(len(self.convolved_cube[0]), len(self.convolved_cube[0][0])))  # numerator
        m2_den = np.zeros(shape=(len(self.convolved_cube[0]), len(self.convolved_cube[0][0])))  # denominator
        for zi in range(len(self.convolved_cube)):
            m2_num += (vel_ax[zi] - model_m1) ** 2 * self.convolved_cube[zi] * clipped_mask[zi]
            m2_den += self.convolved_cube[zi] * clipped_mask[zi]
        model_m2 = np.sqrt(m2_num / m2_den)

        d2_num = np.zeros(shape=(len(self.convolved_cube[0]), len(self.convolved_cube[0][0])))
        d2_n2 = np.zeros(shape=(len(self.convolved_cube[0]), len(self.convolved_cube[0][0])))
        d2_den = np.zeros(shape=(len(self.convolved_cube[0]), len(self.convolved_cube[0][0])))
        for zi in range(len(self.convolved_cube)):
            d2_n2 += self.clipped_data[zi] * (vel_ax[zi] - data_m1) ** 2 * clipped_mask[zi]  # * mask2d
            d2_num += (vel_ax[zi] - data_m1) ** 2 * self.clipped_data[zi] * clipped_mask[zi]  # * mask2d
            d2_den += self.clipped_data[zi] * clipped_mask[zi]  # * mask2d

        d2_num[d2_num < 0] = 0.  # ADDING TO GET RID OF NANs
        data_m2 = np.sqrt(d2_num / d2_den)
        data_m2 = np.nan_to_num(data_m2)
        residual_m2 = data_m2 - model_m2
        residual_frac_m2 = np.nan_to_num((data_m2 - model_m2) / data_m2, nan=10.)

        # AVERAGE EACH MOMENT MAP WITHIN THE VORONOI BINS
        d0 = map_averaging(data_masked_m0, xpix, ypix, binNum, x_in, nPixels)
        m0 = map_averaging(model_masked_m0, xpix, ypix, binNum, x_in, nPixels)
        r0 = map_averaging(residual_m0, xpix, ypix, binNum, x_in, nPixels)
        r0frac = map_averaging(residual_frac_m0, xpix, ypix, binNum, x_in, nPixels)
        cbar_0 = r'mJy km s$^{-1}$ beam$^{-1}$'  # same for data, model, residual for moment 0
        cmap_0 = 'viridis'  # same for data, model, residual for moment 0
        min0 = np.amin([np.nanmin(m0), np.nanmin(d0)])
        max0 = np.amax([np.nanmax(m0), np.nanmax(d0)])
        #if incl_beam:
        #    d0 += beam_overlay  # OPTION 1 (looks a bit too much like an extension of the galaxy)

        data_m1[np.abs(data_m1) > 1e3] = 0  # get rid of edge effects
        residual_m1 = (data_m1 - subtr) - model_m1  # calculate residual
        residual_frac_m1 = np.nan_to_num(((data_m1 - subtr) - model_m1) / data_m1, nan=10.)
        d1 = map_averaging(data_m1, xpix, ypix, binNum, x_in, nPixels)
        m1 = map_averaging(model_m1, xpix, ypix, binNum, x_in, nPixels)
        r1 = map_averaging(residual_m1, xpix, ypix, binNum, x_in, nPixels)
        r1frac = map_averaging(residual_frac_m1, xpix, ypix, binNum, x_in, nPixels)
        cbar_1 = r'km s$^{-1}$'  # same for data, model, residual for moment 1
        cmap_dm1 = 'RdBu_r'  # for data, model for moment 1
        cmap_r1 = 'viridis'  # for residual for moment 1
        min1 = np.amin([np.nanmin(m1), np.nanmin(d1)])
        max1 = np.amax([np.nanmax(m1), np.nanmax(d1)])

        d2 = map_averaging(data_m2, xpix, ypix, binNum, x_in, nPixels)
        m2 = map_averaging(model_m2, xpix, ypix, binNum, x_in, nPixels)
        r2 = map_averaging(residual_m2, xpix, ypix, binNum, x_in, nPixels)
        r2frac = map_averaging(residual_frac_m2, xpix, ypix, binNum, x_in, nPixels)
        cbar_2 = r'km s$^{-1}$'  # same for data, model, residual for moment 2
        cmap_2 = 'viridis'  # same for data, model, residual for moment 2
        min2 = np.amin([np.nanmin(m2), np.nanmin(d2)])
        max2 = np.amax([np.nanmax(m2), np.nanmax(d2)])

        # CALCULATE SUB-CUBE ARCSEC EXTENT
        # RESCALE (x_loc, y_loc) AND (xell, yell) PIXEL VALUES TO CORRESPOND TO SUB-CUBE PIXEL LOCATIONS!
        x_locvb = self.x_loc - self.xyrange[0]  # x_loc - xi
        y_locvb = self.y_loc - self.xyrange[2]  # y_loc - yi

        # SET UP OBSERVATION AXES: initialize x,y axes at 0., with lengths = sub_cube.shape
        y_obs_acvb = np.asarray([0.] * len(m1))
        x_obs_acvb = np.asarray([0.] * len(m1[0]))

        # Define coordinates to be 0,0 at center of the observed axes (find the central pixel number along each axis)
        for i in range(len(x_obs_acvb)):
            x_obs_acvb[i] = -self.resolution * (i - x_locvb)  # (arcsec/pix) * N_pix = arcsec  # - bc RA increases left
        for i in range(len(y_obs_acvb)):
            y_obs_acvb[i] = self.resolution * (i - y_locvb)

        extent = [x_obs_acvb[0], x_obs_acvb[-1], y_obs_acvb[0], y_obs_acvb[-1]]  # left right bottom top

        '''  #
        r0frac[abs(r0frac) > .7] = 0.
        r1frac[abs(r1frac) > .7] = 0.
        r2frac[abs(r2frac) > .7] = 0.
        #print(np.percentile(r0frac[r0frac != 0.], [16., 50., 84.]))
        # print(np.percentile(r1frac[r1frac != 0.], [16., 50., 84.]))
        #print(np.percentile(r2frac[r2frac != 0.], [16., 50., 84.]))
        # 70% CUTOFF (ignoring large outliers) ||| NO CUTOFF
        # [0.11095245 0.36982081 0.6115537 ]    ||| [0.11807345 0.42726565 0.73992959]  # (90% cutoff -> median 0.41)
        # [-0.02858211  0.01388819  0.0806571 ] ||| [-0.02856374  0.01625211  0.12162416]
        # [0.03480282 0.19415179 0.42092524]    ||| [-4.55112186e+306  1.53471847e-001  4.74627680e-001]
        # print(oop)
        #print(np.nanmax(d1), np.nanmin(d1))
        #print(np.nanmax(d2))
        # print(self.convolved_cube.shape)  # 57, 64, 84
        # cubeR = self.R[self.xyrange[2]:self.xyrange[3], self.xyrange[0]:self.xyrange[1]]
        # print(cubeR[34, 40], cubeR[34, 41], cubeR[35, 40], cubeR[35, 41])
        # mom0 pc: 259.56308283369617 264.44614505943844 257.091046329428 261.98709015724785
        # mom1 pc: [163.0853738  160.46565412 157.85176795 164.99981543 162.40552092
        # 182.56246847 179.95691494 187.12025143 184.53206897 191.72944879
        # 189.15849744 198.94669192 196.38644047 193.83252891 203.63162447
        # 201.0879059  198.55080156 208.3582256  205.83079758 257.19277722
        # 254.78244345 262.18831282 259.79095415]
        # print(cubeR[25, 45], cubeR[26, 48])

        # print(cubeR.shape)  # 64, 84 yay!
        #print(cubeR[abs(r0frac) > 0.5])
        #print(cubeR[abs(r1frac) > 0.5])
        #print(cubeR[abs(r2frac) > 0.5])
        #plt.imshow(r2frac, origin='lower')
        #plt.colorbar()
        #plt.show()
        #print(oop)
        
        x_disk = -x_obs_acvb[None, :] * np.cos(self.theta) + y_obs_acvb[:, None] * np.sin(self.theta)  # arcsec
        y_disk = y_obs_acvb[:, None] * np.cos(self.theta) + x_obs_acvb[None, :] * np.sin(self.theta)  # arcsec

        short_R = np.sqrt((y_disk ** 2 / np.cos(self.inc) ** 2) + x_disk ** 2)  # arcsec
        print(x_obs_acvb)
        print(y_obs_acvb)

        #totrad = np.zeros(shape=(len(x_obs_acvb), len(y_obs_acvb)))
        #for x in range(len(x_obs_acvb)):
        #    for y in range(len(y_obs_acvb)):
        #        totrad[x, y] = np.sqrt(x_obs_acvb[x]**2 + y_obs_acvb[y]**2)
        plt.imshow(r2frac, origin='lower')#, extent=extent)
        plt.colorbar()
        lvl = [0.1, 0.2, 0.3, 0.4, 0.5]
        plt.contour(short_R, lvl, colors='w')
        plt.show()
        print(oop)

        if frac:
            r0 = r0frac
            r1 = r1frac
            r2 = r2frac
        # '''

        if just_data:
            fig, ax = plt.subplots(3, 1, figsize=(6, 18))  # rows, cols, figsize=(width, height)
            plt.subplots_adjust(hspace=0.02, wspace=0.02)
            plt.gca().set_aspect('equal', adjustable='box')

            # PLOT MOMENT 0
            imd0 = ax[0].imshow(d0, vmin=min0, vmax=max0, origin='lower', extent=extent, cmap=cmap_0)
            cbard0 = fig.colorbar(imd0, ax=ax[0], pad=0.02)
            cbard0.set_label(cbar_0, rotation=270, labelpad=20.)
            if incl_beam:
                ax[0][0].add_patch(ebeam)

            ax[0].set_xticklabels([])
            ax[0].set_ylabel(r'$\Delta$ DEC [arcsec]', fontsize=fs)  # y [arcsec]

            # PLOT MOMENT 1
            imd1 = ax[1].imshow(d1, vmin=min1, vmax=max1, origin='lower', extent=extent, cmap=cmap_dm1)
            cbard1 = fig.colorbar(imd1, ax=ax[1], pad=0.02)
            cbard1.set_label(cbar_1, rotation=270, labelpad=20.)

            ax[1].set_xticklabels([])
            ax[1].set_ylabel(r'$\Delta$ DEC [arcsec]', fontsize=fs)  # y [arcsec]

            # PLOT MOMENT 2
            imd2 = ax[2].imshow(d2, vmin=min2, vmax=max2, origin='lower', extent=extent, cmap=cmap_2)
            cbard2 = fig.colorbar(imd2, ax=ax[2], pad=0.02)
            cbard2.set_label(cbar_2, rotation=270, labelpad=20.)

            ax[2].set_xlabel(r'$\Delta$ RA [arcsec]', fontsize=fs)  # x [arcsec]
            ax[2].set_ylabel(r'$\Delta$ DEC [arcsec]', fontsize=fs)  # y [arcsec]

            plt.show()

        else:
            # START PLOTTING
            rc('text', usetex=False)
            fig, ax = plt.subplots(3, 3, figsize=(13,8))  # (12,8)
            plt.gca().set_aspect('equal', adjustable='box')
            plt.subplots_adjust(hspace=0.02, wspace=0.02)

            # PLOT MOMENT 0
            imd0 = ax[0][0].imshow(d0, vmin=min0, vmax=max0, origin='lower', extent=extent, cmap=cmap_0)
            cbard0 = fig.colorbar(imd0, ax=ax[0][0], pad=0.02)
            # cbard0.set_label(cbar_0, rotation=270, labelpad=20.)
            if incl_beam:
                ax[0][0].add_patch(ebeam)

            imm0 = ax[0][1].imshow(m0, vmin=min0, vmax=max0, origin='lower', extent=extent, cmap=cmap_0)
            cbarm0 = fig.colorbar(imm0, ax=ax[0][1], pad=0.02)
            # cbarm0.set_label(cbar_0, rotation=270, labelpad=20.)

            imr0 = ax[0][2].imshow(r0, origin='lower', vmin=np.nanmin(r0), vmax=np.nanmax(r0), extent=extent, cmap=cmap_0)
            cbar2r0 = fig.colorbar(imr0, ax=ax[0][2], pad=0.02)
            cbar2r0.set_label(cbar_0, rotation=270, labelpad=20.)

            # only label in first row is on y axis
            ax[0][0].set_xticklabels([])  # xticks shared, yticks not shared!
            ax[0][1].set_xticklabels([])  # xticks shared, yticks shared
            ax[0][1].set_yticklabels([])
            ax[0][2].set_xticklabels([])  # xticks shared, yticks shared
            ax[0][2].set_yticklabels([])
            ax[0][0].set_ylabel('Moment 0\n' + '\n' + r'$\Delta$ DEC [arcsec]', fontsize=fs)  # y [arcsec]

            # PLOT MOMENT 1
            imd1 = ax[1][0].imshow(d1, vmin=min1, vmax=max1, origin='lower', extent=extent, cmap=cmap_dm1)
            cbard1 = fig.colorbar(imd1, ax=ax[1][0], pad=0.02)
            # cbard1.set_label(cbar_1, rotation=270, labelpad=20.)

            imm1 = ax[1][1].imshow(m1, vmin=min1, vmax=max1, origin='lower', extent=extent, cmap=cmap_dm1)
            cbarm1 = fig.colorbar(imm1, ax=ax[1][1], pad=0.02)
            # cbarm1.set_label(cbar_1, rotation=270, labelpad=20.)

            imr1 = ax[1][2].imshow(r1, origin='lower', vmin=np.nanmin(r1), vmax=np.nanmax(r1), extent=extent, cmap=cmap_r1)
            cbarr1 = fig.colorbar(imr1, ax=ax[1][2], pad=0.02)
            cbarr1.set_label(cbar_1, rotation=270, labelpad=20.)

            # only label in second row is on y axis
            ax[1][0].set_xticklabels([])  # xticks shared, yticks not shared!
            ax[1][1].set_xticklabels([])  # xticks shared, yticks shared
            ax[1][1].set_yticklabels([])
            ax[1][2].set_xticklabels([])  # xticks shared, yticks shared
            ax[1][2].set_yticklabels([])
            ax[1][0].set_ylabel('Moment 1\n' + '\n' + r'$\Delta$ DEC [arcsec]', fontsize=fs)  # y [arcsec]

            # PLOT MOMENT 2
            imd2 = ax[2][0].imshow(d2, vmin=min2, vmax=max2, origin='lower', extent=extent, cmap=cmap_2)
            cbard2 = fig.colorbar(imd2, ax=ax[2][0], pad=0.02)
            # cbard2.set_label(cbar_2, rotation=270, labelpad=20.)

            imm2 = ax[2][1].imshow(m2, vmin=min2, vmax=max2, origin='lower', extent=extent, cmap=cmap_2)
            cbarm2 = fig.colorbar(imm2, ax=ax[2][1], pad=0.02)
            # cbarm2.set_label(cbar_2, rotation=270, labelpad=20.)

            imr2 = ax[2][2].imshow(r2, origin='lower', vmin=np.nanmin(r2), vmax=np.nanmax(r2), extent=extent, cmap=cmap_2)
            cbarr2 = fig.colorbar(imr2, ax=ax[2][2], pad=0.02)
            cbarr2.set_label(cbar_2, rotation=270, labelpad=20.)

            # only left-most panel has y axis label, but all panels in bottom row have x axis label
            ax[2][1].set_yticklabels([])  # xticks not shared, yticks shared!
            ax[2][2].set_yticklabels([])  # xticks not shared, yticks shared!
            ax[2][0].set_ylabel('Moment 2\n' + '\n' + r'$\Delta$ DEC [arcsec]', fontsize=fs)  # y [arcsec]
            ax[2][0].set_xlabel(r'$\Delta$ RA [arcsec]', fontsize=fs)  # x [arcsec]
            ax[2][1].set_xlabel(r'$\Delta$ RA [arcsec]', fontsize=fs)  # y [arcsec]
            ax[2][2].set_xlabel(r'$\Delta$ RA [arcsec]', fontsize=fs)  # x [arcsec]

            ax[0][0].set_title(r'Data', fontsize=fs)
            ax[0][1].set_title(r'Model', fontsize=fs)
            ax[0][2].set_title(r'Residual', fontsize=fs)

            plt.show()


    def vorm0(self, incl_beam=True, snr=10, params=None, beamloc=(-1., -1.), ymatch=32, xmatch=66, bcolor='w'):
        """
        Calculate moment0 data map, average it within voronoi bins
        # using equations from https://www.atnf.csiro.au/people/Tobias.Westmeier/tools_hihelpers.php#moments

        :param incl_beam: True or False
        :param snr: voronoi binning target signal-to-noise ratio
        :return:
        """
        # OPEN STRICTMASK
        hdu_m = fits.open(self.data_mask)
        data_mask = hdu_m[0].data  # The mask is stored in hdu_m[0].data, NOT hdu_m[0].data[0]
        hdu_m.close()

        # CREATE VELOCITY AXIS FROM FREQUENCY AXIS
        vel_ax = []
        velwidth = self.c_kms * (1 + self.zred) * self.fstep / self.f_0
        for v in range(len(self.freq_ax)):
            vel_ax.append(self.c_kms * (1. - (self.freq_ax[v] / self.f_0) * (1 + self.zred)))

        # full cube strictmask, clipped to the appropriate zrange
        clipped_mask = data_mask[self.zrange[0]:self.zrange[1], self.xyrange[2]:self.xyrange[3],
                                 self.xyrange[0]:self.xyrange[1]]

        # CALCULATE MOMENT 0 for data, for voronoi-binned lucy-input flux map
        data_lucy_flux = np.zeros(shape=self.convolved_cube[0].shape)
        for z in range(len(vel_ax)):
            data_lucy_flux += abs(self.fstep) * self.clipped_data[z] * clipped_mask[z]  # SUM_z data[z] * mask[z] * dz

        # CALCULATE MOMENT 0 for data, then for model
        data_masked_m0 = np.zeros(shape=self.convolved_cube[0].shape)
        for z in range(len(vel_ax)):
            data_masked_m0 += abs(velwidth) * self.clipped_data[z] * clipped_mask[z]  # SUM_z data[z] * mask[z] * dz

        # CONVERT TO mJy
        data_masked_m0 *= 1e3

        if incl_beam:  # if including beam overlay
            beam_overlay = np.zeros(shape=self.convolved_cube[0].shape)  # overlay beam on the same scale as moment map
            ebeam = patches.Ellipse(beamloc, params['x_fwhm'], params['y_fwhm'],  # 0.65,-0.5
                                    angle=-(90 + params['PAbeam']), linewidth=1.5, edgecolor=bcolor, fill=False)
            beam_overlay[:21, 63:] = self.beam[5:26, 5:26]
            beam_overlay *= np.amax(data_masked_m0) / np.amax(beam_overlay)  # scale so beam shows up well on colormap
        else:
            ebeam = None

        # AVERAGE EACH MAP WITHING THE VORONOI BIN -- CALCULATE THE VORONOI BINS!
        xpix, ypix, binNum, x_in, nPixels = self.just_the_bins(snr=snr, pars_backup=params)

        # AVERAGE lucy-in fluxmap within voronoi bins
        dlucy = map_averaging(data_lucy_flux, xpix, ypix, binNum, x_in, nPixels)
        dlucy_full = np.zeros(shape=self.lucy_out.shape)
        dlucy_full[self.xyrange[2]:self.xyrange[3], self.xyrange[0]:self.xyrange[1]] = dlucy

        # AVERAGE EACH MOMENT MAP WITHIN THE VORONOI BINS
        d0 = map_averaging(data_masked_m0, xpix, ypix, binNum, x_in, nPixels)
        cbar_0 = r'mJy km s$^{-1}$ beam$^{-1}$'  # same for data, model, residual for moment 0
        cmap_0 = 'viridis'  # same for data, model, residual for moment 0
        min0 = np.nanmin(d0)
        max0 = np.nanmax(d0)

        # CALCULATE SUB-CUBE ARCSEC EXTENT
        # RESCALE (x_loc, y_loc) AND (xell, yell) PIXEL VALUES TO CORRESPOND TO SUB-CUBE PIXEL LOCATIONS!
        x_locvb = self.x_loc - self.xyrange[0]  # x_loc - xi
        y_locvb = self.y_loc - self.xyrange[2]  # y_loc - yi

        # SET UP OBSERVATION AXES: initialize x,y axes at 0., with lengths = sub_cube.shape
        y_obs_acvb = np.asarray([0.] * len(d0))
        x_obs_acvb = np.asarray([0.] * len(d0[0]))

        print(x_locvb, y_locvb)  # 42.85356933052101 32.96256939040606
        x_locvb = xmatch - self.xyrange[0]
        y_locvb = ymatch - self.xyrange[2]
        # print(xmatch, ymatch, self.x_loc, self.y_loc)  # 118, 160, 126.85, 150.96

        #x_locvb = xmatch - self.xyrange[0]

        #x_locvb = 100 - 84  # 150 - 84
        #y_locvb = 162 - 118  # 150 - 118

        # Define coordinates to be 0,0 at center of the observed axes (find the central pixel number along each axis)
        for i in range(len(x_obs_acvb)):
            # -self.resolution * (i - x_locvb)
            x_obs_acvb[i] = -self.resolution * (i - x_locvb)  # (arcsec/pix) * N_pix = arcsec  # - bc RA increases left
        for i in range(len(y_obs_acvb)):
            # self.resolution * (i - y_locvb)
            y_obs_acvb[i] = self.resolution * (i - y_locvb)

        extent = [x_obs_acvb[0], x_obs_acvb[-1], y_obs_acvb[0], y_obs_acvb[-1]]  # left right bottom top
        # [1.32, -0.34, -0.64, 0.62]
        print(extent, self.resolution)
        # print(oop)

        return d0, min0, max0, extent, cmap_0, ebeam


    def moment_0(self, abs_diff, incl_beam, norm, samescale=False):
        """
        Create 0th moment map

        :param abs_diff: True or False; if True, show absolute value of the residual
        :param incl_beam: True or False; if True, include beam inset in the data panel
        :param norm: True or False; if True, normalize residual by the data
        :param samescale: True or False; if True, show the residual on the same scale as the data & model
        :return: moment map plot
        """
        # if using equation from https://www.atnf.csiro.au/people/Tobias.Westmeier/tools_hihelpers.php#moments
        hdu_m = fits.open(self.data_mask)
        data_mask = hdu_m[0].data  # The mask is stored in hdu_m[0].data, NOT hdu_m[0].data[0]
        vel_ax = []
        velwidth = self.c_kms * (1 + self.zred) * self.fstep / self.f_0
        for v in range(len(self.freq_ax)):
            vel_ax.append(self.c_kms * (1. - (self.freq_ax[v] / self.f_0) * (1 + self.zred)))

        # full cube strictmask, clipped to the appropriate zrange
        clipped_mask = data_mask[self.zrange[0]:self.zrange[1], self.xyrange[2]:self.xyrange[3],
                                 self.xyrange[0]:self.xyrange[1]]

        data_masked_m0 = np.zeros(shape=self.convolved_cube[0].shape)
        for z in range(len(vel_ax)):
            data_masked_m0 += abs(velwidth) * self.clipped_data[z] * clipped_mask[z]

        model_masked_m0 = np.zeros(shape=(len(self.convolved_cube[0]), len(self.convolved_cube[0][0])))
        for zi in range(len(self.convolved_cube)):
            model_masked_m0 += self.convolved_cube[zi] * abs(velwidth) * clipped_mask[zi]

        fig, ax = plt.subplots(3, 1)
        plt.subplots_adjust(hspace=0.02)

        # CONVERT TO mJy
        data_masked_m0 *= 1e3
        model_masked_m0 *= 1e3

        subtr = 0.
        if incl_beam:
            beam_overlay = np.zeros(shape=self.convolved_cube[0].shape)
            print(self.beam.shape, beam_overlay.shape)
            # beam_overlay[:self.beam.shape[0], (beam_overlay.shape[1] - self.beam.shape[1]):] = self.beam
            beam_overlay[:self.beam.shape[0] - 6, (beam_overlay.shape[1] - self.beam.shape[1]) + 6:] = self.beam[6:, :-6]
            print(beam_overlay.shape, self.beam.shape)
            beam_overlay *= np.amax(data_masked_m0) / np.amax(beam_overlay)
            data_masked_m0 += beam_overlay
            subtr = beam_overlay
        vmin = np.amin([np.nanmin(model_masked_m0), np.nanmin(data_masked_m0)])
        vmax = np.amax([np.nanmax(model_masked_m0), np.nanmax(data_masked_m0)])
        cbartitle0 = r'mJy Km s$^{-1}$ beam$^{-1}$'

        im0 = ax[0].imshow(data_masked_m0, vmin=vmin, vmax=vmax, origin='lower')
        ax[0].set_title(r'Moment 0 (top - bottom: data, model, residual)')
        cbar = fig.colorbar(im0, ax=ax[0], pad=0.02)
        cbar.set_label(cbartitle0, rotation=270, labelpad=20.)

        im1 = ax[1].imshow(model_masked_m0, vmin=vmin, vmax=vmax, origin='lower')
        cbar1 = fig.colorbar(im1, ax=ax[1], pad=0.02)
        cbar1.set_label(cbartitle0, rotation=270, labelpad=20.)

        title0 = 'Moment 0 residual (model-data)'
        titleabs = 'Moment 0 residual abs(model-data)'
        diff = model_masked_m0 - (data_masked_m0 - subtr)
        if norm:
            diff /= data_masked_m0
            diff = np.nan_to_num(diff)
            print(np.nanquantile(diff, [0.16, 0.5, 0.84]), 'typical differences; 0.16, 0.5, 0.84!')
            title0 += ' / data'
            titleabs += ' / data'
            cbartitle0 = 'Ratio [Residual / Data]'
        if samescale:
            if abs_diff:
                diff = np.abs(diff)
            im2 = ax[2].imshow(diff, vmin=vmin, vmax=vmax, origin='lower')
        else:  # then residual scale
            # im2 = ax[2].imshow(diff, origin='lower', vmin=np.nanmin([diff, -diff]), vmax=np.nanmax([diff, -diff]))
            im2 = ax[2].imshow(diff, origin='lower', vmin=np.nanmin(diff), vmax=np.nanmax(diff))
        cbar2 = fig.colorbar(im2, ax=ax[2], pad=0.02)
        cbar2.set_label(cbartitle0, rotation=270, labelpad=20.)

        ax[0].set_xticklabels([])
        ax[1].set_xticklabels([])
        ax[2].set_xlabel(r'x [pixels]', fontsize=20)  # x [arcsec]
        ax[0].set_ylabel(r'y [pixels]', fontsize=20)  # y [arcsec]
        ax[1].set_ylabel(r'y [pixels]', fontsize=20)  # y [arcsec]
        ax[2].set_ylabel(r'y [pixels]', fontsize=20)  # y [arcsec]

        plt.show()


    def moment_12(self, abs_diff=False, incl_beam=False, norm=False, mom=1, samescale=False):
        """
        Create 1st or 2nd moment map
        :param abs_diff: True or False; if True, show absolute value of the residual
        :param incl_beam: True or False; if True, include beam inset in the data panel
        :param norm: True or False; if True, normalize residual by the data
        :param mom: moment, 1 or 2
        :param samescale: True or False; if True, show the residual on the same scale as the data & model

        :return: moment map plot
        """

        hdu_m = fits.open(self.data_mask)
        data_mask = hdu_m[0].data  # The mask is stored in hdu_m[0].data, NOT hdu_m[0].data[0]

        vel_ax = []
        velwidth = self.c_kms * (1 + self.zred) * self.fstep / self.f_0
        for v in range(len(self.freq_ax)):
            vel_ax.append(self.c_kms * (1. - (self.freq_ax[v] / self.f_0) * (1 + self.zred)))

        # full cube strictmask, clipped to the appropriate zrange
        clipped_mask = data_mask[self.zrange[0]:self.zrange[1], self.xyrange[2]:self.xyrange[3],
                                 self.xyrange[0]:self.xyrange[1]]

        model_numerator = np.zeros(shape=(len(self.convolved_cube[0]), len(self.convolved_cube[0][0])))
        model_denominator = np.zeros(shape=(len(self.convolved_cube[0]), len(self.convolved_cube[0][0])))
        for zi in range(len(self.convolved_cube)):
            model_numerator += vel_ax[zi] * self.convolved_cube[zi] * clipped_mask[zi]
            model_denominator += self.convolved_cube[zi] * clipped_mask[zi]
        model_mom = model_numerator / model_denominator

        data_numerator = np.zeros(shape=(len(self.convolved_cube[0]), len(self.convolved_cube[0][0])))
        data_denominator = np.zeros(shape=(len(self.convolved_cube[0]), len(self.convolved_cube[0][0])))
        for zi in range(len(self.convolved_cube)):
            data_numerator += vel_ax[zi] * self.clipped_data[zi] * clipped_mask[zi]
            data_denominator += self.clipped_data[zi] * clipped_mask[zi]
        data_mom = data_numerator / data_denominator

        if mom == 2:
            m2_num = np.zeros(shape=(len(self.convolved_cube[0]), len(self.convolved_cube[0][0])))
            m2_den = np.zeros(shape=(len(self.convolved_cube[0]), len(self.convolved_cube[0][0])))
            for zi in range(len(self.convolved_cube)):
                m2_num += (vel_ax[zi] - model_mom)**2 * self.convolved_cube[zi] * clipped_mask[zi]
                m2_den += self.convolved_cube[zi] * clipped_mask[zi]
            m2 = np.sqrt(m2_num / m2_den)

            d2_num = np.zeros(shape=(len(self.convolved_cube[0]), len(self.convolved_cube[0][0])))
            d2_n2 = np.zeros(shape=(len(self.convolved_cube[0]), len(self.convolved_cube[0][0])))
            d2_den = np.zeros(shape=(len(self.convolved_cube[0]), len(self.convolved_cube[0][0])))
            for zi in range(len(self.convolved_cube)):
                d2_n2 += self.clipped_data[zi] * (vel_ax[zi] - data_mom)**2 * clipped_mask[zi] # * mask2d
                d2_num += (vel_ax[zi] - data_mom)**2 * self.clipped_data[zi] * clipped_mask[zi] # * mask2d
                d2_den += self.clipped_data[zi] * clipped_mask[zi] # * mask2d
            dfig = np.zeros(shape=(len(self.convolved_cube[0]), len(self.convolved_cube[0][0]))) + 1.  # create mask
            dfig2 = np.zeros(shape=(len(self.convolved_cube[0]), len(self.convolved_cube[0][0]))) + 1.  # create mask
            dfig[d2_n2 < 0.] = 0.  # d2_n2 matches d2_den on the sign aspect
            dfig2[d2_den < 0.] = 0.

            d2_num[d2_num < 0] = 0.  # ADDING TO GET RID OF NANs
            d2 = np.sqrt(d2_num / d2_den)

            fig, ax = plt.subplots(3, 1)
            plt.subplots_adjust(hspace=0.02)

            subtr = 0.
            if incl_beam:
                d2 = np.nan_to_num(d2)
                beam_overlay = np.zeros(shape=self.convolved_cube[0].shape)
                beam_overlay[:self.beam.shape[0], (beam_overlay.shape[1] - self.beam.shape[1]):] = self.beam
                print(beam_overlay.shape, self.beam.shape)
                beam_overlay *= np.amax(d2) / np.amax(beam_overlay)
                d2 += beam_overlay
                subtr = beam_overlay

            #for ax in grid:
            vmin2 = np.amin([np.nanmin(d2), np.nanmin(m2)])
            vmax2 = np.amax([np.nanmax(d2), np.nanmax(m2)])
            cbartitle2 = r'km/s'
            im0 = ax[0].imshow(d2, origin='lower', vmin=vmin2, vmax=vmax2)  # , cmap='RdBu_r'
            ax[0].set_title(r'Moment 2 (top - bottom: data, model, residual)')
            cbar = fig.colorbar(im0, ax=ax[0], pad=0.02)
            cbar.set_label(cbartitle2, rotation=270, labelpad=20.)

            im1 = ax[1].imshow(m2, origin='lower', vmin=vmin2, vmax=vmax2)  # , cmap='RdBu_r'
            cbar2 = fig.colorbar(im1, ax=ax[1], pad=0.02)
            cbar2.set_label(cbartitle2, rotation=270, labelpad=20.)

            diff = m2 - (d2 - subtr)
            title2 = 'Moment 2 residual (model-data)'
            titleabs2 = 'Moment 2 residual abs(model-data)'
            if norm:
                diff /= d2
                print(np.nanquantile(diff, [0.16, 0.5, 0.84]), 'look median!')
                title2 += ' / data'
                titleabs2 += ' / data'
                cbartitle2 = 'Ratio [Residual / Data]'
            if abs_diff:
                diff = np.abs(diff)
                #ax.set_title(titleabs2)
            #else:
                #ax.set_title(title2)
            if samescale:
                im2 = ax[2].imshow(diff, origin='lower', vmin=vmin2, vmax=vmax2)  # , cmap='RdBu'
            else:  # residscale
                #im2 = ax.imshow(diff, origin='lower', vmin=np.nanmin([diff, -diff]),
                #                vmax=np.nanmax([diff, -diff]))
                print(np.nanmin(diff), np.nanmax(diff))
                im2 = ax[2].imshow(diff, origin='lower', vmin=np.nanmin(diff), vmax=np.nanmax(diff))  # np.nanmin(diff)
                #ax.set_title(title2)
            cbar2 = fig.colorbar(im2, ax=ax[2], pad=0.02)
            cbar2.set_label(cbartitle2, rotation=270, labelpad=20.)

            ax[0].set_xticklabels([])
            ax[1].set_xticklabels([])
            ax[2].set_xlabel(r'x [pixels]', fontsize=20)  # x [arcsec]
            ax[0].set_ylabel(r'y [pixels]', fontsize=20)  # y [arcsec]
            ax[1].set_ylabel(r'y [pixels]', fontsize=20)  # y [arcsec]
            ax[2].set_ylabel(r'y [pixels]', fontsize=20)  # y [arcsec]

            plt.show()

        elif mom == 1:
            fig, ax = plt.subplots(3, 1)
            plt.subplots_adjust(hspace=0.02)

            subtr = 0.
            if incl_beam:
                beam_overlay = np.zeros(shape=self.convolved_cube[0].shape)
                beam_overlay[:self.beam.shape[0], (beam_overlay.shape[1] - self.beam.shape[1]):] = self.beam
                print(beam_overlay.shape, self.beam.shape)
                beam_overlay *= np.amax(data_mom) / np.amax(beam_overlay)
                data_mom += beam_overlay
                subtr = beam_overlay

            cbartitle1 = r'km/s'
            data_mom[np.abs(data_mom) > 1e3] = 0
            print(np.nanmax(data_mom), np.nanmin(data_mom), np.nanmax(model_mom), np.nanmin(model_mom))
            vmin1 = np.amin([np.nanmin(data_mom), np.nanmin(model_mom)])
            vmax1 = np.amax([np.nanmax(data_mom), np.nanmax(model_mom)])
            im0 = ax[0].imshow(data_mom, origin='lower', vmin=vmin1, vmax=vmax1, cmap='RdBu_r')
            ax[0].set_title(r'Moment 1 (top - bottom: data, model, residual)')
            cbar0 = fig.colorbar(im0, ax=ax[0], ticks=[-500, -250, 0., 250.], pad=0.02)
            cbar0.set_label(cbartitle1, rotation=270, labelpad=20.)

            im1 = ax[1].imshow(model_mom, origin='lower', vmin=vmin1, vmax=vmax1, cmap='RdBu_r')
            #ax[1].set_title(r'Moment 1 (model)')
            cbar1 = fig.colorbar(im1, ax=ax[1], ticks=[-500, -250, 0., 250.], pad=0.02)
            cbar1.set_label(cbartitle1, rotation=270, labelpad=20.)
            title1 = 'Moment 1 residual (model - data)'
            diff = model_mom - (data_mom - subtr)
            if norm:
                diff /= data_mom
                print(np.nanquantile(diff, [0.16, 0.5, 0.84]), 'look median!')
                title1 += ' / data'
                cbartitle1 = 'Ratio [Residual / Data]'
            if samescale:
                im2 = ax[2].imshow(diff, origin='lower', vmin=vmin1, vmax=vmax1, cmap='RdBu')  # , cmap='RdBu'
            else:  # resid scale
                vn = np.amax([-150, np.nanmin(diff)])
                vx = np.amin([150, np.nanmax(diff)])
                #im2 = ax[2].imshow(diff, origin='lower', vmin=np.nanmin(diff), vmax=np.nanmax(diff))
                im2 = ax[2].imshow(diff, origin='lower', vmin=vn, vmax=vx)
                #im2 = ax[2].imshow(diff, origin='lower', vmin=np.nanmin([diff, -diff]),
                #                   vmax=np.nanmax([diff, -diff]))  # cmap='RdBu'
            # ax.set_title(title1)
            cbar2 = fig.colorbar(im2, ax=ax[2], pad=0.02)
            cbar2.set_label(cbartitle1, rotation=270, labelpad=20.)

            ax[0].set_xticklabels([])
            ax[1].set_xticklabels([])
            ax[2].set_xlabel(r'x [pixels]', fontsize=20)  # x [arcsec]
            ax[0].set_ylabel(r'y [pixels]', fontsize=20)  # y [arcsec]
            ax[1].set_ylabel(r'y [pixels]', fontsize=20)  # y [arcsec]
            ax[2].set_ylabel(r'y [pixels]', fontsize=20)  # y [arcsec]

            plt.show()


    def kin_pa(self):

        from pafit import fit_kinematic_pa as fkpa

        xbin, ybin = np.random.uniform(low=[-30, -20], high=[30, 20], size=(100, 2)).T
        print(xbin)
        inc = 60.  # assumed galaxy inclination
        r = np.sqrt(xbin ** 2 + (ybin / np.cos(np.radians(inc))) ** 2)  # Radius in the plane of the disk
        a = 40  # Scale length in arcsec
        vr = 2000 * np.sqrt(r) / (r + a)  # Assumed velocity profile
        vel = vr * np.sin(np.radians(inc)) * xbin / r  # Projected velocity field

        plt.clf()
        ang, ang_err, v_syst = fkpa.fit_kinematic_pa(xbin, ybin, vel, debug=True, nsteps=30)
        plt.show()


def test_qell2(params, l_in, q_ell, rfit, pa, figname):
    #ell_mask = ellipse_fitting(input_data, rfit, params['xell'], params['yell'], params['resolution'], pa, q_ell)

    fig = plt.figure()
    ax = plt.gca()
    plt.imshow(l_in, origin='lower')
    plt.colorbar()
    e1 = patches.Ellipse((params['xell'], params['yell']), 2 * rfit / params['resolution'],
                         2 * rfit / params['resolution'] * q_ell, angle=pa, linewidth=2, edgecolor='w', fill=False)
    ax.add_patch(e1)
    plt.title(r'q = ' + str(q_ell) + r', PA = ' + str(pa) + ' deg, rfit = ' + str(rfit) + ' arcsec')
    if figname is None:
        plt.show()
    else:
        figname += '_' + str(q_ell) + '_' + str(pa) + '_' + str(rfit) + '.png'
        plt.savefig(figname, dpi=300)
    plt.clf()


if __name__ == "__main__":
    # MAKE SURE I HAVE ACTIVATED THE iraf27 ENVIRONMENT IF I NEED TO RUN LUCY DECONVOLUTION!!!
    t0_true = time.time()
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument('--parfile')

    args = vars(parser.parse_args())

    # Load parameters from the parameter file
    params, priors, n_free = par_dicts(args['parfile'])

    # Make nice plot fonts
    rc('font', **{'family': 'serif', 'serif': ['Times']})
    rc('text', usetex=True)

    # CREATE OUTPUT FILENAME BASED ON INPUT PARAMETERS
    pars_str = ''
    for key in params:
        pars_str += str(params[key]) + '_'
    out = '/Users/jonathancohn/Documents/dyn_mod/outputs/test_' + pars_str + '.fits'

    if 'ds2' not in params:
        params['ds2'] = params['ds']

    if 'sqrt2n' not in params:
        params['sqrt2n'] = False
    elif params['sqrt2n'] == 1:
        params['sqrt2n'] = True
    else:
        params['sqrt2n'] = False

    # DECIDE HERE WHETHER TO AVG IN THE REBIN() FUNCTION (avging=True) OR SUM (avging=False)
    avging = True

    # CREATE THINGS THAT ONLY NEED TO BE CALCULATED ONCE (collapse fluxes, lucy, noise)
    mod_ins = model_prep(data=params['data'], ds=params['ds'], ds2=params['ds2'], lucy_out=params['lucy'],
                         lucy_mask=params['lucy_mask'],
                         lucy_b=params['lucy_b'], lucy_in=params['lucy_in'], lucy_it=params['lucy_it'],
                         data_mask=params['mask'], grid_size=params['gsize'], res=params['resolution'],
                         x_std=params['x_fwhm'], y_std=params['y_fwhm'], pa=params['PAbeam'],
                         xyerr=[params['xerr0'], params['xerr1'], params['yerr0'], params['yerr1']],
                         zrange=[params['zi'], params['zf']], avg=avging, q_ell=params['q_ell'],
                         theta_ell=np.deg2rad(params['theta_ell']), xell=params['xell'], yell=params['yell'])

    lucy_mask, lucy_out, beam, fluxes, freq_ax, f_0, fstep, input_data, noise, co_ell_sb, co_ell_rad = mod_ins

    hduin = fits.open(params['lucy_in'])
    l_in = hduin[0].data
    hduin.close()

    #dir = '/Users/jonathancohn/Documents/dyn_mod/groupmtg/ugc_2698_newmasks/'
    #for rfit in [0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 0.9, 1.]:
    #    test_qell2(params, l_in, 0.38, rfit, 19, dir)
    #for theta in [19]:
    #    for q in [0.38, 0.4, 0.5]:
    #        for rfit in [0.65, 0.7, 0.75]:
    #            test_qell2(params, l_in, q, rfit, theta, dir)
    #print(oop)
    # ig = params['incl_gas'] == 'True'

    # Calculate projected distance if you want
    # 1277 RA, DEC: 49.964542	41.573528
    # 1271 RA, DEC: 49.797000	41.353250
    # 1275 RA, DEC: 49.950667	41.511696
    # 2698 RA, DEC: 50.512000	40.863889
    # dists = projected_distance(50.512000, 49.950667, 40.863889, 41.511696, 91e6, 76e6)  # UGC 2698 vs NGC 1275
    # print(dists)  # (0.01349781247173701, 0.7733676872895765, 1228300.934928068, 1025833.7478520127)
    # print(oop)

    # CREATE MODEL CUBE!
    inc_fixed = np.deg2rad(67.7)  # based on fiducial model (67.68 deg)
    vcg_in = None
    if params['incl_gas'] == 'True':
        vcg_in = gas_vel(params['resolution'], co_ell_rad, co_ell_sb, params['dist'], f_0, inc_fixed, zfixed=0.02152)

    out = params['outname']
    t0m = time.time()
    mg = ModelGrid(resolution=params['resolution'], os=params['os'], x_loc=params['xloc'], y_loc=params['yloc'],
                   mbh=params['mbh'], inc=np.deg2rad(params['inc']), vsys=params['vsys'], dist=params['dist'],
                   theta=np.deg2rad(params['PAdisk']), input_data=input_data, lucy_out=lucy_out, out_name=out,
                   beam=beam, rfit=params['rfit'], enclosed_mass=params['mass'], ml_ratio=params['ml_ratio'],
                   sig_type=params['s_type'], zrange=[params['zi'], params['zf']], menc_type=params['mtype'],
                   sig_params=[params['sig0'], params['r0'], params['mu'], params['sig1']], f_w=params['f'],
                   ds=params['ds'], ds2=params['ds2'], noise=noise, reduced=True, freq_ax=freq_ax, q_ell=params['q_ell'],
                   theta_ell=np.deg2rad(params['theta_ell']), xell=params['xell'], yell=params['yell'], fstep=fstep,
                   f_0=f_0, bl=params['bl'], xyrange=[params['xi'], params['xf'], params['yi'], params['yf']],
                   n_params=n_free, data_mask=params['mask'], incl_gas=params['incl_gas']=='True', vrad=params['vrad'],
                   kappa=params['kappa'], omega=params['omega'], co_rad=co_ell_rad, co_sb=co_ell_sb, avg=avging,
                   pvd_width=0.142838/params['resolution'], vcg_func=vcg_in, sqrt2n=params['sqrt2n'])
    # pvd_width = (params['x_fwhm']*params['y_fwhm'])/params['resolution']/2.

    # x_fwhm=0.197045, y_fwhm=0.103544 -> geometric mean = sqrt(0.197045*0.103544) = 0.142838; regular mean = 0.1502945
    mg.grids()
    mg.convolution()
    chi_sq = mg.chi2()  # 6495.965711236455 (1.2275067481550368)  # 6498.030199144044 (1.227896863027975)
    # mg.moment_0(False, False, False)
    mg.scaling_rels(rel=2)
    # mg.vor_moms(incl_beam=True, fs=12, pars_backup=params, frac=True)
    # mg.pvd()
    print(oop)
    xtalk = [4, 7, 13, 16]
    ytalk = [6, 5, 11, 11]
    xtalk = 14
    ytalk = 9
    #mg.line_profiles(xtalk, ytalk)
    print(oop)
    mg.mge_sbprof()
    mg.scaling_rels(rel=2)
    # mg.vor_moms(incl_beam=True)
    # mg.pvd()
    # mg.line_profiles(5,6)
    #xtalk = [4, 7, 13, 16]
    #ytalk = [6, 5, 11, 11]
    #xtalk = 14
    #ytalk = 9
    #mg.line_profiles(xtalk, ytalk)
    print(oop)
    #mg.moment_0(abs_diff=False, incl_beam=True, norm=False)
    #mg.moment_12(abs_diff=False, incl_beam=False, norm=False, mom=1)
    #mg.moment_12(abs_diff=False, incl_beam=False, norm=False, mom=2)
    # mg.moment_12(mom=1)
    mg.line_profiles(13, 8)
    mg.line_profiles(9, 7)
    mg.line_profiles(9, 8)
    mg.line_profiles(8, 8)
    mg.line_profiles(13, 9)
    print(oop)

    mg.line_profiles(10, 8)  # center?
    mg.line_profiles(11, 8)  # center?
    mg.line_profiles(12, 8)  # red?
    mg.line_profiles(9, 7)  # blue?
    #mg.line_profiles(14, 8)  # decent red [was using this]
    #mg.line_profiles(14, 10)  # decent red [was using this]
    #mg.line_profiles(15, 9)  # good red

    # Good examples
    #mg.line_profiles(7, 5)  # decent blue [recently using this]
    #mg.line_profiles(14, 9)  # good red [recently using this]
    #mg.line_profiles(15, 10)  # decent red [recently using this]

    # meh examples
    #mg.line_profiles(4, 6)  # blue orig (not great)
    #mg.line_profiles(6, 6)  # blue okay? (meh)
    #mg.line_profiles(10, 9)  # near ctr orig (meh)
    #mg.line_profiles(13, 8)  # red [not bad]

    '''  #
    mg.line_profiles(8, 8)
    mg.line_profiles(9, 8)
    mg.line_profiles(9, 7)
    mg.line_profiles(9, 6)
    mg.line_profiles(10, 7)
    mg.line_profiles(11, 10)
    mg.line_profiles(12, 8)
    mg.line_profiles(12, 9)
    mg.line_profiles(14, 8)  # decent red
    mg.line_profiles(14, 9)  # good red
    mg.line_profiles(14, 10)  # decent red
    mg.line_profiles(15, 9)  # good red
    mg.line_profiles(15, 10)  # decent red
    for ii in range(6, 10):
        for jj in range(5, 9):
            mg.line_profiles(ii, jj)
    # not great blues: 6,5 // 6,8 // 7,8 / 8,5 / 9,7 / 9,8
    # okayish blues 6,6 // 6,7 // 7,6 // 7,7 / 8,6 / 8,7 / 9,5 / 9,6
    # maybe reasonable blues 7,5
    #mg.line_profiles(9, 9)
    #mg.line_profiles(9, 10)
    #mg.line_profiles(10, 9)  # near ctr orig
    #mg.line_profiles(10, 8)  # near ctr
    #mg.line_profiles(11, 8)  # near ctr
    #mg.line_profiles(10, 10)
    #mg.line_profiles(4, 6)  # blue orig
    #mg.line_profiles(6, 6)  # blue
    # mg.line_profiles(3, 3)
    # mg.line_profiles(6, 4)
    # mg.line_profiles(4, 4)
    #mg.line_profiles(14, 10)  # red
    #mg.line_profiles(16, 12)
    # '''  #

    print(time.time() - t0m, ' seconds')
    print('True Total time: ' + str(time.time() - t0_true) + ' seconds')  # ~1 second for a cube of 84x64x49

'''
        fig, ax = plt.subplots(1, 3)
        hdu_m = fits.open('ugc_2698/ugc_2698_20.3_strict2_lucyout_n10.fits')
        fiduciallucy = hdu_m[0].data
        hdu_m.close()
        im0 = ax[0].imshow(fiduciallucy, origin='lower')
        cbar = fig.colorbar(im0, ax=ax[0], pad=0.02)
        im1 = ax[1].imshow(self.lucy_out, origin='lower')
        cbar = fig.colorbar(im1, ax=ax[1], pad=0.02)
        im2 = ax[2].imshow(fiduciallucy - self.lucy_out, origin='lower')
        cbar = fig.colorbar(im2, ax=ax[2], pad=0.02)
        plt.show()
        print(oop)
'''

'''
    params, priors, n_free, qobs = par_dicts(args['parfile'], q=True)

    # make sure inc prior does not conflict with q
    print(priors['inc'])
    qint_pri = np.amax(np.rad2deg(np.arccos(np.sqrt((400*qobs**2 - 1.)/399.))))
    priors['inc'][0] = np.amax([priors['inc'][0], np.rad2deg(np.arccos(np.amin(qobs)))])
    priors['inc'][0] = np.amax([priors['inc'][0], qint_pri])
    print(priors['inc'])
    print(oop)
'''