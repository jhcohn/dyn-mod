import dyn_model as dm
import astropy.io.fits as fits
import numpy as np
import argparse

if __name__ == "__main__":
    # MAKE SURE I HAVE ACTIVATED THE iraf27 ENVIRONMENT!!!
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument('--parfile')

    args = vars(parser.parse_args())

    # Load parameters from the parameter file
    params, priors = dm.par_dicts(args['parfile'])
    mod_ins = dm.model_prep(data=params['data'], ds=params['ds'], lucy_out=params['lucy'], lucy_mask=params['lucy_mask'],
                         lucy_b=params['lucy_b'], lucy_in=params['lucy_in'], lucy_o=params['lucy_o'],
                         lucy_it=params['lucy_it'], data_mask=params['mask'], grid_size=params['gsize'],
                         res=params['resolution'], x_std=params['x_fwhm'], y_std=params['y_fwhm'], pa=params['PAbeam'],
                         xyerr=[params['xerr0'], params['xerr1'], params['yerr0'], params['yerr1']],
                         zrange=[params['zi'], params['zf']])

    lucy_mask, lucy_out, beam, fluxes, freq_ax, f_0, fstep, input_data, noise = mod_ins

    hduin = fits.open(params['lucy_in'])
    l_in = hduin[0].data
    hduin.close()

    out = params['outname']

    mbhs = [9.625, 9.4375]
    xlocs = [128.2, 127.1]
    ylocs = [150.375, 151.1]
    sig0s = [2., 10.5]
    incs = [69.1, 68.1]
    pas = [18.6, 19.2]
    vsyss = [6478., 6452.]
    ml = 1.6728
    fs = [1.353, 1.0595]
    chis = []
    used_pars = []
    count = 0
    for mbh in mbhs:
        for xloc in xlocs:
            for yloc in ylocs:
                for sig0 in sig0s:
                    for inc in incs:
                        for pa in pas:
                            for vsys in vsyss:
                                for f in fs:
                                    print(count)
                                    chis.append(dm.model_grid(resolution=params['resolution'], s=params['s'],
                                                              x_loc=xloc,
                                                          y_loc=yloc, mbh=mbh, inc=np.deg2rad(inc), vsys=vsys,
                                                          dist=params['dist'], theta=np.deg2rad(pa),
                                                          input_data=input_data, lucy_out=lucy_out, out_name=out,
                                                          beam=beam, rfit=params['rfit'], enclosed_mass=params['mass'],
                                                          ml_ratio=ml, sig_type=params['s_type'],  f_w=f,
                                                          zrange=[params['zi'], params['zf']], menc_type=params['mtype'],
                                                          sig_params=[sig0, params['r0'], params['mu'],
                                                                      params['sig1']],
                                                          ds=params['ds'], noise=noise, chi2=True, reduced=True,
                                                          freq_ax=freq_ax, q_ell=params['q_ell'],
                                                          theta_ell=np.deg2rad(params['theta_ell']),
                                                          xell=params['xell'], yell=params['yell'], fstep=fstep,
                                                          f_0=f_0, bl=params['bl'],
                                                          xyrange=[params['xi'], params['xf'], params['yi'],
                                                                   params['yf']]))
                                    used_pars.append(str(mbh) + '-' + str(xloc) + '-' + str(yloc) + '-' + str(sig0)
                                                     + '-' + str(inc) + '-' + str(pa) + '-' + str(vsys) + '-' +
                                                     str(ml) + '-' + str(f))
                                    count += 1
    print(chis)
    print(used_pars)

    print(np.percentile(chis, [3., 16., 50., 84., 97.]))
    idx = np.argmin(chis)
    print(chis[idx], used_pars[idx])

