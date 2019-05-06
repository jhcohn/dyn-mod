import numpy as np
from astropy.io import fits
from scipy import integrate
import matplotlib.pyplot as plt


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


if __name__ == "__main__":

    base = '/Users/jonathancohn/Documents/dyn_mod/'

    import pickle

    import dyn_model as dm
    params, priors = dm.par_dicts(base + 'param_files/ngc_3258bl_params.txt', q=False)  # get dicts of params and file names from parameter file

    ax_lab = [r'$\log_{10}$(M$_{\odot}$)', 'deg', 'deg', 'pixels', 'pixels', 'km/s', 'km/s', 'km/s', 'pc', 'pc', 'unitless', r'M$_{\odot}$/L$_{\odot}$']
    pars = ['mbh', 'inc', 'PAdisk', 'xloc', 'yloc', 'vsys', 'sig1', 'sig0', 'mu', 'r0', 'f', 'ml_ratio']
    for par in range(len(pars)):
        chains = 'emcee_out/flatchain_' + pars[par] + '_100_1_50.pkl'

        with open(base + chains, 'rb') as pk:
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
                                     + str(round(percs[2] - percs[1], 2)) + ', -'
                                     + str(round(percs[1] - percs[0], 2)) + ')', fontsize=16)
            plt.show()
    print(oop)


    # hdu = fits.open(base + 'NGC_1332_newfiles/NGC1332_CO21_C3_MS_bri_20kms.pbcor.fits')
    hdu = fits.open(base + 'ngc_3258/ngc3258_CO21_bri.pbcor.fits')
    data_in = hdu[0].data[0]
    z_len = len(data_in)  # store the number of velocity slices in the data cube
    freq1 = float(hdu[0].header['CRVAL3'])
    f_step = float(hdu[0].header['CDELT3'])
    # f_0 = 2.305380000000e11  # 2.29369132e11  # intrinsic frequency of CO(2-1) line  # 2.30538e11
    f_0 = float(hdu[0].header['RESTFRQ'])
    hdu.close()
    freq_axis = np.arange(freq1, freq1 + (z_len * f_step), f_step)  # [bluest, less blue, ..., reddest]
    # freq_axis = freq_axis[:-1]
    # print(freq_axis[0], freq_axis[-1])  # 2.29909, 2.28777 [e11]
    v_sys = 2760.76  # 1562.2  # km/s
    z_ax = np.asarray([((f_0 - freq) / freq) * (3.*10**5) for freq in freq_axis])  # v_opt, km/s
    # print(z_ax)
    # print(oop)

    # hdu = fits.open(base + 'ngc1332_things/NGC_1332_fullsize_idl_n5_beam31_s1.fits')
    # idl_out = hdu[0].data
    # hdu.close()

    # hdu = fits.open(base + 'ngc1332_things/NGC_1332_fullsize_filtconv_n5_beam31_s1.fits')
    # filt_out = hdu[0].data
    # hdu.close()

    # hdu = fits.open(base + 'NGC_1332_fullsize_apconv_n5_beam31fwhm_s1_weightdiv_corrsig.fits')  # _zred
    # hdu = fits.open(base + 'NGC_1332_fullsize_apconv_n5_beam31fwhm_s1.fits')
    # hdu = fits.open(base+ 'NGC_1332_freqcube_summed_apconv_n5_beam31fwhm_s1.fits')
    # hdu = fits.open(base + 'NGC_1332_freqcube_summed_apconv_n5_beam31fwhm_spline_s1.fits')
    # hdu = fits.open(base + 'outputs/NGC_3258_general_7.42_2386000000.0_1.02_166.8_354.89_362.04_-0.46_4.75_2760.76_0.14_3.16_46.0__benlucycorr.fits')
    # hdu = fits.open(base + 'outputs/NGC_3258_general_7.42_3.16_2386000000.0_1.02_166.8_2760.76_-0.46_4.75_0.14_362.04_354.89_46.0__pcscale.fits')
    # hdu = fits.open(base + 'outputs/NGC_3258_general_7.42_2386000000.0_1.02_166.8_354.89_362.04_-0.46_4.75_2760.76_0.14_3.16_46.0__subcube.fits')
    # hdu = fits.open(base + 'outputs/NGC_3258_general_7.42_2386000000.0_1.02_166.8_354.89_362.04_-0.46_4.75_2760.76_0.14_3.16_46.0__subcube_ellmask_bl2.fits')
    hdu = fits.open(base + 'outputs/NGC_3258_general_7.42_2386000000.0_1.02_166.8_354.89_362.04_-0.46_4.75_2760.76_0.14_3.16_46.0__subcube_ellmask_bl2_noell.fits')
    # hdu = fits.open(base + 'outputs/ngc_3258_benlucy_outcube.fits')

    ap_out = hdu[0].data
    hdu.close()

    hdu = fits.open('/Users/jonathancohn/Documents/dyn_mod/outputs/NGC_3258_fitting_ellipse.fits')
    ell_fits = hdu[0].data
    hdu.close()

    plt.imshow(ell_fits, origin='lower')
    plt.colorbar()
    plt.show()

    print(ap_out.shape)
    print(data_in.shape)
    data_in = data_in[39:85, 310:410, 310:410]
    print(data_in.shape)

    data_in *= ell_fits
    ap_out *= ell_fits

    idx = 59  # 26
    idx = 20
    print(np.amin((ap_out[idx] - data_in[idx])/data_in[idx]), np.amax((ap_out[idx] - data_in[idx])/data_in[idx]),
          np.median((ap_out[idx] - data_in[idx])/data_in[idx]))
    plt.imshow((ap_out[idx] - data_in[idx])/data_in[idx], origin='lower')
    # plt.imshow((idl_out[26] - data_in[26])/data_in[26], origin='lower')
    plt.colorbar()
    # plt.xlim(700,750)
    # plt.ylim(720,750)
    # plt.plot(848, 720, 'w*')
    plt.show()
    # print(oop)
    plt.imshow(data_in[idx], origin='lower')
    plt.colorbar()
    # plt.plot(848, 720, 'w*')
    plt.show()

    data_4 = rebin(data_in, 4)
    # idl_4 = rebin(idl_out, 4)
    # filt_4 = rebin(filt_out)
    ap_4 = rebin(ap_out, 4)

    # CREATED ONCE
    # hdu = fits.PrimaryHDU(data_4)
    # hdul = fits.HDUList([hdu])
    # hdul.writeto(base + 'NGC_1332_newfiles/NGC1332_CO21_C3_MS_bri_20kms_bin4x4.pbcor.fits')

    '''
    # UNCOMMENT THIS LATER
    hdu = fits.PrimaryHDU(ap_4)
    hdul = fits.HDUList([hdu])
    hdul.writeto(base + 'outputs/NGC_3258_general_4x4binned.fits')
    # hdul.writeto(base + 'NGC_1332_freqcube_summed_apconv_n5_beam31fwhm_spline_s1_4x4binned.fits')
    # hdul.writeto(base + 'NGC_1332_freqcube_summed_apconv_n5_beam31fwhm_s1_4x4binned.fits')
    
    hdu = fits.PrimaryHDU(data_4 - ap_4)
    hdul = fits.HDUList([hdu])
    hdul.writeto(base + 'NGC_3258_data_minus_freqcube_4x4binned.fits')
    '''
    # hdul.writeto(base + 'NGC_1332_data_minus_freqcube_spline_4x4binned.fits')
    # print(oops)
    #hdu = fits.PrimaryHDU(data_4 - ap_4)
    #hdul = fits.HDUList([hdu])
    #hdul.writeto('data_minus_apconv_4x4_fixed.fits')
    # print(oop)
    # 212, 180 --> 212*4
    # x=2 --> x_0 = 2, 3
    # x=2, n=4 --> x_0 = 4,5,6,7 = (x-1)*n
    # x=4, n=4 --> x_0 = 12,13,14,15 = (x-1)*n:(x*n) = 12:16 YAY!
    # y
    # 3, 4 (n=2): [(3/2):(3+1)/2] 22 23 32 33
    # 0, 0 (n=2): [(0/2):(0/2)+1] 00, 01, 10, 11

    #plt.imshow(idl_4[23], origin='lower')
    #plt.colorbar()
    #plt.plot(212, 180, 'w*')
    #plt.show()
    print(ap_4.shape, data_4.shape)
    chi_sq_num = 0.
    for z in range(len(data_4)):
        chi_sq_num += np.sum((ap_4[z] - data_4[z]) ** 2)
    print(chi_sq_num)  # ~ -1160 for take2, ~7.2 for pcscale. Good?

    # inds_to_try2 = np.asarray([[212, 180], [159, 159], [165, 155], [155, 165], [160, 160], [170, 170], [155, 155], [165, 165]])
    # inds_to_try2 = np.asarray([[178, 178], [200, 172], [168, 155], [155, 168], [135, 140], [125, 150]])
    # inds_to_try2 = np.asarray([[89, 92], [95, 89], [92, 92], [91, 86]])  # red side, blue side, center top, center bot
    inds_to_try2 = np.asarray([[10,10], [10,15], [15,10]])

    for i in range(len(inds_to_try2)):
        print(inds_to_try2[i][0], inds_to_try2[i][1])

        '''
        data_full = []
        for ii in range((inds_to_try2[i][1])*4,((inds_to_try2[i][1]+1)*4)):
            for jj in range((inds_to_try2[i][0])*4, ((inds_to_try2[i][0]+1)*4)):
                data_full.append(data_in[:, ii, jj])
        data_full = np.asarray(data_full)
        # print(data_full.shape)  # 4x4 x 75 --> 16, 75
    
        data_stack = np.zeros(shape=(data_full[0].shape))
        for l in range(len(data_full)):
            # plt.plot(z_ax, data_full[l], 'k--')  # confirmed
            data_stack += data_full[l]
        # plt.plot(z_ax, data_stack, 'g--')  # confirmed identical to how Data is calculated
        '''
        plt.plot(z_ax[39:85], ap_4[:, inds_to_try2[i][1], inds_to_try2[i][0]], 'r-', label=r'Astropy conv')  # ap_out  # 0.01217/.00031 *
                 # / np.amax(ap_out[:, inds_to_try2[i][1], inds_to_try2[i][0]]), 'r-', label=r'Astropy conv')
        # plt.plot(z_ax, idl_out[:, inds_to_try2[i][1], inds_to_try2[i][0]], 'k--', label=r'IDL conv')  # idl_out
                 # / np.amax(idl_out[:, inds_to_try2[i][1], inds_to_try2[i][0]]), 'k--', label=r'IDL conv')
        # plt.plot(z_ax, filt_out[:, inds_to_try2[i][1], inds_to_try2[i][0]], 'b:', label=r'Scipy conv')
                 # / np.amax(filt_out[:, inds_to_try2[i][1], inds_to_try2[i][0]]), 'b:', label=r'Scipy conv')
        plt.plot(z_ax[39:85], data_4[:, inds_to_try2[i][1], inds_to_try2[i][0]], 'b:', label=r'Data')  # data_in
        plt.axvline(x=v_sys, color='k')
        plt.title(str(inds_to_try2[i][0]) + ', ' + str(inds_to_try2[i][1]))  # ('no x,y offset')
        plt.legend()
        plt.show()
        # base = '/Users/jonathancohn/Documents/dyn_mod/lps/sum_div_filtconv_'  # div_'
        # plt.savefig(base + 'newfiles_lp_s' + str(s) + '_' + str(inds_to_try2[i][0]) + '_' + str(inds_to_try2[i][1]))
        plt.close()
    print(oop)

    '''  #
    idl_filt = []
    idl_ap = []
    for z in range(len(idl_out)):
        print(z)
        idl_filt.append((idl_out[z] - filt_out[z]))
        idl_ap.append((idl_out[z] - ap_out[z]))
        if 35 < z < 39:
            plt.imshow(idl_filt[z], origin='lower')
            plt.colorbar()
            plt.show()
    
            plt.imshow(idl_ap[z], origin='lower')
            plt.colorbar()
            plt.show()
    
    idl_filt = np.asarray(idl_filt)
    idl_ap = np.asarray(idl_ap)
    
    idl_filt_coll = integrate.simps(idl_filt, axis=0)
    idl_ap_coll = integrate.simps(idl_ap, axis=0)
    
    plt.imshow(idl_filt_coll, origin='lower')
    plt.colorbar()
    plt.show()
    
    plt.imshow(idl_ap_coll, origin='lower')
    plt.colorbar()
    plt.show()
    print(oops)
    # '''  #

    obs3d = np.asarray([[1., 2., 3., 4.], [1., 2., 3., 4.], [5., 6., 7., 8.], [9., 10., 11., 12.]])
    s = 2
    print(obs3d.shape)

    '''
    subarrays = blockshaped(obs3d[:, :], 4, 4)  # bin the data in groups of 4x4 pixels
    # data[z, ::-1, :] flips the y-axis, which is stored in python in the reverse order (problem.png)
    
    # Take the mean along the first axis of each subarray in subarrays, then take the mean along the
    # other (s-length) axis, such that what remains is a 1d array of the means of the sxs subarrays. Then reshape
    # into the correct lengths
    reshaped = np.mean(np.mean(subarrays, axis=-1), axis=-1).reshape((int(len(data[0])/4.),
                                                                      int(len(data[0][0])/4.)))
    reshaped_m = np.mean(np.mean(subarrays_m, axis=-1), axis=-1).reshape((int(len(mask[0])/4.),
                                                                          int(len(mask[0][0])/4.)))
    rebinned.append(reshaped)
    rebinned_m.append(reshaped_m)
    print("Rebinning the cube done in {0} s".format(time.time() - t_rebin))  # 0.5 s
    '''

    intrinsic_cube = []  # np.zeros(shape=(len(z_ax), len(fluxes), len(fluxes[0])))
    # break each (s*len(realx), s*len(realy))-sized velocity slice of obs3d into an array comprised of sxs subarrays
    # i.e. break the 300s x 300s array into blocks of sxs arrays, so that I can take the means of each block
    subarrays = blockshaped(obs3d[:, :], s, s)

    print('sub:')
    print(subarrays)

    # Take the mean along the first (s-length) axis of each subarray in subarrays, then take the mean along the
    # other (s-length) axis, such that what remains is a 1d array of the means of the sxs subarrays. Then reshape
    # into the correct real x_pixel by real y_pixel lengths
    reshaped = s**2 * np.mean(np.mean(subarrays, axis=-1), axis=-1).reshape((2, 2))
    reshaped2 = np.sum(np.sum(subarrays, axis=-1), axis=-1).reshape((2, 2))
    print('reshape:')
    print(reshaped)

    print('reshaped2:')
    print(reshaped2)
