# basic numeric setup
import numpy as np

# SETTINGS / CHOICES
pri = 'priset3'  # 'wide', 'mid', 'narrow', 'priset2', 'priset3'
masktype = 'baseline'  # 'strict'  # 'lax'  # 'baseline'
mgetype = 'rhe'  # 'rhe'  # 'ahe'  # 'rhe'  # 'rre'  # 'akin'
os = 4  # 1 2 3 4 6 8 10 12 14 16
gs = 31  # beam grid size
sigtype = 'flat'  # flat, exp, gauss
rfit = 0.7
vrad = False  # include radial velocity
kappa = False  # include radial velocity with kappa
omega = False  # include radial velocity with sub-Keplerian motion
gas = False  # incl_gas
zi = 25  # 29
zf = 82  # 78

if masktype == 'ext2582':
    zi = 25
    zf = 82
elif masktype == 'ext2285':
    zi = 22
    zf = 85

gas_label = 'nog'
if gas:
    gas_label = 'gas'

vtype = 'orig'  # orig, vrad, omega, kappa
if vrad:
    vtype = 'vrad'  # orig, vrad, omega, kappa
elif omega:
    vtype = 'omega'  # orig, vrad, omega, kappa
elif kappa:
    vtype = 'kappa'  # orig, vrad, omega, kappa

# run_type = sigtype + 'sig_' + masktype + '_' + mgetype + '_' + vtype + '_' + gas_label
# run_type = 'rfit10_' + masktype + '_' + mgetype + '_' + vtype + '_' + gas_label
# run_type = masktype + '_' + mgetype + '_' + vtype + '_' + gas_label
# run_type = 'exv_' + 'os' + str(os) + '_' + masktype + '_' + mgetype + '_' + vtype + '_' + gas_label
run_type = 'exv_' + masktype + '_' + mgetype + '_' + vtype + '_' + gas_label
# run_type  = 'os' + str(os) + '_d91_' + masktype + '_' + mgetype + '_' + vtype + '_' + gas_label
# run_type  = 'os' + str(os) + '_' + masktype + '_' + mgetype + '_' + vtype + '_' + gas_label
    # 'b' + str(gs) + '_d85_' +  masktype + '_' + mgetype + '_' + vtype + '_' + gas_label


def write_runfile(run_base):
    """

    :param run_base: new run file base, including directory
    :return:
    """

    nest_base = '/Users/jonathancohn/Documents/dyn_mod/dyn_cluster/param_files/'

    with open(nest_base + 'nest_2698_' + run_base + '.lsf', 'w+') as rn:
        rn.write('# NECESSARY JOB SPECIFICATIONS)\n')
        rn.write('# BSUB -J nest_2698_' + run_base + '.lsf\n')
        rn.write('# BSUB -L /bin/bash\n')
        rn.write('# BSUB -W 89:00 #200:00\n')
        rn.write('# BSUB -n 8\n')
        rn.write('# BSUB -R "span[ptile=8]" \n')
        rn.write('# BSUB -R "rusage[mem=2560]"\n')
        rn.write('# BSUB -M 2560\n')
        rn.write('# BSUB -u joncohn@tamu.edu \n')
        rn.write('# BSUB -o out_nest_2698_' + run_base + '.%J\n\n')

        rn.write('ml purge\n')
        rn.write('source /home/joncohn/mydyn.sh\n\n')

        rn.write('python dyndyn_n.py --p=ugc_2698/ugc_2698_' + run_base + '.txt')

        return nest_base + 'nest_2698_' + run_base + '.lsf'


def newpar_scratch(parname, priors, nest_pars, fixed_str, fixed_fl, fixed_int, files):
    """

    :param parname: new parameter file name, including directory
    :param priors: dictionary of priors for each free parameter
    :param nest_pars: dictionary of dynesty parameters
    :param fixed_str: dictionary of fixed str parameters
    :param fixed_fl: dictionary of fixed float parameters
    :param fixed_int: dictionary of fixed int parameters
    :param files: dictionary of files, including directory (local vs cluster)
    :return: parname
    """
    with open(parname, 'w+') as newp:
        newp.write('# TYPE PARAMETER VALUE PRIOR_MIN PRIOR_MAX\n')
        newp.write('# FREE PARAMS\n')
        for key in priors:
            guess = str(priors[key][0] + (priors[key][1] - priors[key][0]) / 2)
            newp.write('free ' + key + ' ' + guess + ' ' + str(priors[key][0]) + ' ' + str(priors[key][1]) + '\n')
        newp.write('\n')
        newp.write('# DYNESTY PARAMS\n')
        for key in nest_pars:
            newp.write(nest_pars[key][0] + ' ' + key + ' ' + str(nest_pars[key][1]) + '\n')
        newp.write('\n')
        newp.write('# FIXED PARAMS\n')
        for key in fixed_str:
            newp.write('str ' + key + ' ' + fixed_str[key] + '\n')
        for key in fixed_fl:
            newp.write('float ' + key + ' ' + str(fixed_fl[key]) + '\n')
        for key in fixed_int:
            newp.write('int ' + key + ' ' + str(fixed_int[key]) + '\n')
        newp.write('\n')
        newp.write('# FILES\n')
        for key in files:
            newp.write('str ' + key + ' ' + files[key] + '\n')
        newp.write('\n')

    return parname


def write_newpar(parname, copyf, priors, fixed, files):
    """

    :param parname: new parameter file name, including directory
    :param copyf: parameter file used as the foundation for the new file (copy this file)
    :param priors: dictionary of priors for each free parameter
    :param fixed: dictionary of fixed parameters
    :param files: dictionary of files, including directory (local vs cluster)
    :return: parname
    """

    print(copyf)
    with open(parname, 'w+') as newp:
        with open(copyf, 'r') as cpf:
            section = 'freepars'
            for line in cpf:
                if line.startswith('free') and priors['mbh'] != []:
                    cols = line.split()
                    cols[3] = str(priors[cols[1]][0])
                    cols[4] = str(priors[cols[1]][1])
                    line = ' '.join(cols) + '\n'
                elif line.startswith('# DYNESTY') or line.startswith('# FIXED FLOATS'):
                    section = 'fixed'
                elif (line.startswith('int') or line.startswith('float') or line.startswith('str')) and\
                        section == 'fixed':
                    cols = line.split()
                    if cols[1] not in priors:
                        cols[2] = str(fixed[cols[1]])
                        line = ' '.join(cols) + '\n'
                elif line.startswith('# FILES'):
                    section = 'files'
                elif line.startswith('str') and section == 'files':
                    cols = line.split()
                    cols[2] = str(files[cols[1]])
                    line = ' '.join(cols) + '\n'

                newp.write(line)

    return parname


dm = '/Users/jonathancohn/Documents/dyn_mod/'
dc = '/scratch/user/joncohn/dyn_cluster'
localpars = dm + 'ugc_2698/'
clusterpars = dm + 'dyn_cluster/param_files/'
clusterloc = dc + '/ugc_2698/'

masks = {'baseline': 'UGC2698_C4_CO21_bri_20.3kms_jonathan_casaimviewhand_strictmask2.fits',
         'strict': 'UGC2698_C4_CO21_bri_20.3kms_jonathan_casaimviewhand_strictmaskstrict.fits',
         'lax': 'UGC2698_C4_CO21_bri_20.3kms_jonathan_casaimviewhand_strictmasklax.fits',
         'ext2582': 'UGC2698_C4_CO21_bri_20.3kms_jonathan_casaimviewhand_strictmaskext2582.fits',
         'ext2285': 'UGC2698_C4_CO21_bri_20.3kms_jonathan_casaimviewhand_strictmaskext2285.fits'}

lucys = {'baseline': 'ugc_2698_20.3_strict2_lucyout_n10.fits',
         'strict': 'ugc_2698_20.3_strictstrict_lucyout_n10.fits',
         'lax': 'ugc_2698_20.3_strictlax_lucyout_n10.fits',
         'ext2582': 'ugc_2698_20.3_strict2582_lucyout_n10.fits',
         'ext2285': 'ugc_2698_20.3_strict2285_lucyout_n10.fits'}

if gs != 31:
    lucys = {'baseline': 'ugc_2698_20.3_strict2_lucyout_g' + str(gs) + '_n10.fits',
             'strict': 'ugc_2698_20.3_strictstrict_lucyout_g' + str(gs) + '_n10.fits',
             'lax': 'ugc_2698_20.3_strictlax_lucyout_g' + str(gs) + 'n10.fits',
             'ext2582': 'ugc_2698_20.3_strict2582_lucyout' + str(gs) + '_n10.fits',
             'ext2285': 'ugc_2698_20.3_strict2285_lucyout' + str(gs) + '_n10.fits'}

lucy_masks = {'baseline': 'ugc_2698_collapsemask_20.3kms_jonathan_casaimview_strictmask2.fits',
              'strict': 'ugc_2698_collapsemask_20.3kms_jonathan_casaimview_strictmaskstrict.fits',
              'lax': 'ugc_2698_collapsemask_20.3kms_jonathan_casaimview_strictmasklax.fits',
              'ext2582': 'ugc_2698_collapsemask_20.3kms_jonathan_casaimview_strictmask2582.fits',
              'ext2285': 'ugc_2698_collapsemask_20.3kms_jonathan_casaimview_strictmask2285.fits'}

fluxmaps = {'baseline': 'ugc_2698_fluxmap_20.3kms_jonathan_casaimview_strictmask2.fits',
            'strict': 'ugc_2698_fluxmap_20.3kms_jonathan_casaimview_strictmask2strict.fits',
            'lax': 'ugc_2698_fluxmap_20.3kms_jonathan_casaimview_strictmasklax.fits',
            'ext2582': 'ugc_2698_fluxmap_20.3kms_jonathan_casaimview_strictmask2582.fits',
            'ext2285': 'ugc_2698_fluxmap_20.3kms_jonathan_casaimview_strictmask2285.fits'}

mges = {'ahe': 'ugc_2698_ahe_mge.txt', 'rhe': 'ugc_2698_rhe_mge.txt', 'rre': 'ugc_2698_rre_mge.txt',
        'ahepsf': 'ugc_2698_ahepsf_mge.txt', 'rhepsf': 'ugc_2698_rhepsf_mge.txt', 'rrepsf': 'ugc_2698_rrepsf_mge.txt',
        'akin': 'yildirim_table_2698.txt'}


# FIXED PARAMS
dyn_pars = {'nlive': ['int', 250], 'nprocs': ['int', 8], 'maxc': ['int', 10000000], 'thresh': ['float', 0.02]}

fixed_str = {# OUTPUT RUN BASE NAME
             'outname': 'u2698_' + run_type,
             's_type': sigtype,
             'vtype': vtype,  # orig, vrad, omega, kappa
             'incl_gas': str(gas)}

fixed = {#'r0': 0.5,  # COULD BE FREE IN SOME MODELS
         #'mu': 0.5,  # COULD BE FREE IN SOME MODELS
         #'sig1': 5.,  # COULD BE FREE IN SOME MODELS
         #'vrad': 0.,  # COULD BE FREE IN SOME MODELS
         #'kappa': 0.,  # COULD BE FREE IN SOME MODELS
         #'omega': 1.,  # COULD BE FREE IN SOME MODELS
         # ELLIPTICAL REGION PARS
         'xell': 126.85, 'yell': 150.9, 'theta_ell': 19., 'q_ell': 0.38, 'rfit': rfit,
         # yell between 150.8~few to 150.9; could do 150.85; or round so xell 127, yell 151, qell .36-.38, adjust rfit?
         # theta_ell 18.88 : 19.19 ;; q_ell cos(67.7 deg) ;; rfit arcsec
         # DYNESTY PARAMS
         # 'nlive': 250, 'nprocs': 8, 'maxc': 10000000, 'thresh': 0.02,
         # OUTPUT RUN BASE NAME
         # 'outname': 'u2698_' + run_type,
         # ALWAYS FIXED PARAMETERS (THESE DON'T CHANGE!)
         'resolution': 0.02,  # arcsec/pix
         'x_fwhm': 0.197045,
         'y_fwhm': 0.103544,
         'PAbeam': 9.271,
         'dist': 91.,  # 85.29,  #89.,  # Mpc'
         # OTHER FIXED PARAMETERS
         #'zi': zi, # cube range: zi:zf,
         #'zf': zf,
         #'xi': 84,
         #'xf': 168,
         #'yi': 118,
         #'yf': 182,
         #'xerr0': 144,  # xerr & yerr must be divisible by ds
         #'xerr1': 168,
         #'yerr0': 96,
         #'yerr1': 120,
         #'os': os,
         #'ds': 4,
         #'gsize': gs,
         #'mtype': 0,  # mge file
         #'bl': 0,
         # 's_type': sigtype,
         # 'vtype': vtype,  # orig, vrad, omega, kappa
         # 'incl_gas': str(gas),
         #'lucy_it': 10
         }

fixed_int = {  # FIXED ints
    'zi': zi,  # cube range: zi:zf,
    'zf': zf,
    'xi': 84,
    'xf': 168,
    'yi': 118,
    'yf': 182,
    'xerr0': 144,  # xerr & yerr must be divisible by ds
    'xerr1': 168,
    'yerr0': 96,
    'yerr1': 120,
    'os': os,
    'ds': 4,
    'gsize': gs,
    'mtype': 0,  # mge file
    'bl': 0,
    'lucy_it': 10}

priors = {'mbh': [0, 12], 'xloc': [116, 140], 'yloc': [140, 160], 'sig0': [0, 200], 'inc': [0, 89],
          'PAdisk': [0, 90], 'vsys': [5000, 8100], 'ml_ratio': [0.1, 10], 'f': [0.1, 2.5]}
midpriors = {'mbh': [9., 9.8], 'xloc': [125.5, 129], 'yloc': [149, 153], 'sig0': [5, 25], 'inc': [63, 73],
             'PAdisk': [16, 22], 'vsys': [6440, 6490], 'ml_ratio': [1, 2.5], 'f': [0.7, 1.4]}
narrowpri = {'mbh': [9.1, 9.7], 'xloc': [125.5, 128], 'yloc': [150, 152], 'sig0': [12, 24], 'inc': [65, 71],
             'PAdisk': [15, 22], 'vsys': [6445, 6465], 'ml_ratio': [1.3, 2.4], 'f': [0.9, 1.2]}
priset2 = {'mbh': [8.5, 9.7], 'xloc': [124, 128], 'yloc': [148, 152], 'sig0': [5, 25], 'inc': [0, 85],
          'PAdisk': [10, 30], 'vsys': [6425, 6485], 'ml_ratio': [0.8, 2.4], 'f': [0.5, 1.5]}
priset3 = {'mbh': [8, 10.], 'xloc': [124, 128], 'yloc': [148, 152], 'sig0': [0, 40], 'inc': [0, 89],
          'PAdisk': [5, 35], 'vsys': [6405, 6505], 'ml_ratio': [0.3, 3.], 'f': [0.5, 1.5]}


if vrad:
    priors['vrad'] = [-50, 50]
    midpriors['vrad'] = [-20, 35]
    narrowpri['vrad'] = [-5, 25]
    priset2['vrad'] = [-5, 25]
    priset3['vrad'] = [-50, 50]
    fixed['kappa'] = 0.
    fixed['omega'] = 1.
elif kappa or omega:
    fixed['vrad'] = 0.
    priors['kappa'] = [-1, 1]
    midpriors['kappa'] = [-0.5, 0.5]
    narrowpri['kappa'] = [-0.1, 0.1]
    priset2['kappa'] = [-0.2, 0.2]
    priset3['kappa'] = [-1, 1]
    if omega:
        priors['omega'] = [0, 1]
        midpriors['omega'] = [0.5, 1]
        narrowpri['omega'] = [0.75, 1]
        priset2['omega'] = [0.7, 1]
        priset3['omega'] = [0, 1]
    else:
        fixed['omega'] = 1.
else:
    fixed['vrad'] = 0.
    fixed['kappa'] = 0.
    fixed['omega'] = 1.

# 'gauss': sig1 + sig0 * np.exp(-(r - r0) ** 2 / (2 * mu ** 2)) ;;; 'exp': sig1 + sig0 * np.exp(-r / r0)
if sigtype == 'exp' or sigtype == 'gauss':
    priors['r0'] = [0, 100]
    midpriors['r0'] = [0, 50]
    narrowpri['r0'] = [0, 20]
    priset2['r0'] = [0, 50]
    priset3['r0'] = [0, 100]
    priors['sig1'] = [0, 100]
    midpriors['sig1'] = [0, 50]
    narrowpri['sig1'] = [0, 20]
    priset2['sig1'] = [0, 50]
    priset3['sig1'] = [0, 100]
    priset3['sig0'] = [0, 100]
    if sigtype == 'gauss':
        priors['mu'] = [0, 100]
        midpriors['mu'] = [0, 50]
        narrowpri['mu'] = [0, 10]
        priset2['mu'] = [0, 50]
        priset3['mu'] = [0, 100]
    else:
        fixed['mu'] = 1.
else:
    fixed['r0'] = 0.5
    fixed['mu'] = 1.
    fixed['sig1'] = 0.

# LOCAL FILES
localfiles = {'data': localpars + 'UGC2698_C4_CO21_bri_20.3kms.pbcor.fits',
              'mask': localpars + masks[masktype], 'lucy': localpars + lucys[masktype],
              'lucy_mask': localpars + lucy_masks[masktype], 'lucy_in': localpars + fluxmaps[masktype],
              'lucy_b': localpars + 'ugc_2698_beam' + str(gs) + '.fits', 'lucy_o': localpars + lucys[masktype] + '[0]',
              'mass': localpars + mges[mgetype]}

# CLUSTER FILES
clusterfiles = {'data': clusterloc + 'UGC2698_C4_CO21_bri_20.3kms.pbcor.fits',
                'mask': clusterloc + masks[masktype], 'lucy': clusterloc + lucys[masktype],
                'lucy_mask': localpars + lucy_masks[masktype], 'lucy_in': clusterloc + fluxmaps[masktype],
                'lucy_b': clusterloc + 'ugc_2698_beam' + str(gs) + '.fits',
                'lucy_o': clusterloc + lucys[masktype] + '[0]', 'mass': clusterloc + mges[mgetype]}

newpar = 'ugc_2698_' + run_type + '.txt'
locnewpar = localpars + newpar
clusternewpar = clusterpars + newpar

copyf = localpars + 'ugc_2698_baseline_rhe_orig_gas.txt'

# SET PRIOR CHOICE
use_priors = priors
if pri == 'wide':
    use_priors = priors
elif pri == 'mid':
    use_priors = midpriors
elif pri == 'narrow':
    use_priors = narrowpri
elif pri == 'priset2':
    use_priors = priset2
elif pri == 'priset3':
    use_priors = priset3

# MAKE THE LOCAL VERSION
# newpl = write_newpar(locnewpar, copyf, use_priors, fixed, localfiles)
newpl = newpar_scratch(locnewpar, use_priors, dyn_pars, fixed_str, fixed, fixed_int, localfiles)
print(newpl)

# AND THE CLUSTER VERSION TO SCP OVER
# newpc = write_newpar(clusternewpar, copyf, use_priors, fixed, clusterfiles)
newpc = newpar_scratch(clusternewpar, use_priors, dyn_pars, fixed_str, fixed, fixed_int, clusterfiles)
print(newpc)

# THEN WRITE CLUSTER RUNFILE TO SCP OVER
submit_file = write_runfile(run_type)
print(submit_file)

'''  #
# PARAM INFO:
# INFO FREE PARAMS:
# mbh=black_hole_mass [solar masses]
# xloc=x_position_of_BH [pixels]
# yloc=y_position_of_BH [pixels]
# sig0=sigma_velocity_dispersion [km/s]
# r0=sigma_scale_radius [pc]
# mu=sigma_variance [pc]
# sig1=sigma_offset [km/s]
# inc=inclination_angle [deg]
# PAdisk=disk_PA_from_xobs_axis_to_blueshifted_axis [deg]
# vsys=systemic_velocity [km/s]
# ml_ratio=mass_to_light_ratio [solar]
# f=gaussian_line_profile_normalization_constant

# INFO FIXED PARAMS
# q_ell=axis_ratio_of_fitting_ellipse [unitless, based on disk inc]
# resolution=pixel_scale [arcsec/pix]
# rfit=disk_radius_within_which_we_will_compare_model_and_data [arcsec]
# s_type=type_of_sigma ['flat', 'gauss', or 'exp']
# dist=galaxy_angular_size_distance [Mpc]
# s=oversampling_factor [number]
# gsize=grid_size_for_beam [pixel number; must be odd]
# x_fwhm=fwhm_of_beam_in_x_direction [arcsec]
# y_fwhm=fwhm_of_beam_in_y_direction [arcsec]
# PAbeam=beam_position_angle [deg]
# inc_star=inclination_of_stars_in_galaxy [deg]
# zi=slice_number_in_input_data_where_data_starts_showing_up [python, so first slice is 0]
# zf=slice_number_in_input_data_where_data_stops_showing_up [python, so first slice is 0, and zstop is NOT included]
# ds=downsampling_factor [int]
# bl=units_on_collapsed_flux [0=Jy/beam * Hz, 1=Jy/beam * km/s]
# mtype=type_of_file_describing_enclosed_stellar_mass [0=mge parameters, 1=mass as function of radius; 2=circular velocity as function of radius]

# INFO FILES
# data=data_cube
# mask=mask_cube
# lucy=output_from_lucy [Lucy output file]
# lucy_in=lucy_input_flux_map [if no output_from_lucy]
# lucy_b=beam_to_use_with_lucy [if no output_from_lucy]
# lucy_o=filename_to_save_lucy_output [if no output_from_lucy]
# lucy_mask=mask_cube_collapsed_to_2d [if no output_from_lucy]
# mass=file_describing_enclosed_stellar_mass [m(R) if mtype=2, v(R) if mtype=1, mge parameters if mtype=0]
#  '''


#                 colscluster = line.split()
#                 colscluster[2] = str(clusterfiles[colscluster[1]])
#                 clusterline = ' '.join(colscluster) + '\n'

    # AHE 2
# thing = 'dyndyn_newpri_3_maxc10mil_n8_mask2ahe_1586514104.468642_end.pkl'  # 2698 mask2 ahe (sig0 prior now too low)
# thing = 'dyndyn_newpri_3_maxc10mil_n8_mask2ahe_1586560467.2338874_end.pkl'  # 2698 mask2 ahe sig0 extended!
# name = 'ugc_2698_newmasks/u2698_nest_mask2ahe_sig0extend_3sig.png'
# cornername = 'ugc_2698_newmasks/u2698_nest_mask2ahe_sig0extend_corner_3sig.png'
# inpf = 'ugc_2698/ugc_2698_newmask2_ahe_n8.txt'
# mod = 'ahe baseline'
# RHE 2
# thing = 'dyndyn_newpri_3_maxc10mil_n8_mask2rhe_1586505950.4425063_end.pkl'  # 2698 mask2 rhe (sig0 prior now too low)
# thing = 'dyndyn_newpri_3_maxc10mil_n8_mask2rhe_1586825623.941504_end.pkl'  # 2698 mask2 rhe sig0 extended!
# name = 'ugc_2698_newmasks/u2698_nest_mask2rhe_sig0extend_3sig.png'
# cornername = 'ugc_2698_newmasks/u2698_nest_mask2rhe_sig0extend_corner_3sig.png'
# inpf = 'ugc_2698/ugc_2698_newmask2_rhe_n8.txt'
# mod = 'rhe baseline'
# AKIN 2
# thing = 'dyndyn_newpri_3_maxc10mil_n8_mask2akin_1586613935.0107005_end.pkl'  # 2698 mask2 akin's mge (bad priors)
# thing = 'dyndyn_newpri_3_maxc10mil_n8_mask2akin_1586820818.1132705_end.pkl'  # 2698 mask2 akin's mge prior extended!
# name = 'ugc_2698_newmasks/u2698_nest_mask2akin_priextend_3sig.png'
# cornername = 'ugc_2698_newmasks/u2698_nest_mask2akin_priextend_corner_3sig.png'
# inpf = 'ugc_2698/ugc_2698_newmask2_akin_n8.txt'
# mod = 'akin baseline'
# RRE 2
# thing = 'dyndyn_newpri_3_maxc10mil_n8_mask2rre_1586586220.7801452_end.pkl'  # 2698 mask2 rre (bad priors)
# thing = 'dyndyn_newpri_3_maxc10mil_n8_mask2rre_1586801196.254912_end.pkl'  # 2698 mask2 rre prior extended!
# name = 'ugc_2698_newmasks/u2698_nest_mask2rre_priextend_3sig.png'
# cornername = 'ugc_2698_newmasks/u2698_nest_mask2rre_priextend_corner_3sig.png'
# inpf = 'ugc_2698/ugc_2698_newmask2_rre_n8.txt'
# mod = 'rre baseline'
# RHE LAX
# thing = 'dyndyn_newpri_3_maxc10mil_n8_masklaxrhe_1586572307.33264_end.pkl'  # 2698 masklax rhe (bad priors)
# thing = 'dyndyn_newpri_3_maxc10mil_n8_masklaxrhe_1586800510.8196273_end.pkl'  # 2698 masklax rhe prior extended!
# name = 'ugc_2698_newmasks/u2698_nest_masklaxrhe_priextend_3sig.png'
# cornername = 'ugc_2698_newmasks/u2698_nest_masklaxrhe_priextend_corner_3sig.png'
# inpf = 'ugc_2698/ugc_2698_newmasklax_rhe_n8.txt'
# mod = 'rhe lax'
# RHE STRICT
# thing = 'dyndyn_newpri_3_maxc10mil_n8_maskstrictrhe_1586573455.940675_end.pkl'  # 2698 maskstrict rhe (bad priors)
# thing = 'dyndyn_newpri_3_maxc10mil_n8_maskstrictrhe_1586801056.355033_end.pkl'  # 2698 maskstrict rhe prior extended!
# name = 'ugc_2698_newmasks/u2698_u2698_nest_maskstrictrhe_priextend_3sig.png'
# cornername = 'ugc_2698_newmasks/u2698_nest_maskstrictrhe_priextend_corner_3sig.png'
# inpf = 'ugc_2698/ugc_2698_newmaskstrict_rhe_n8.txt'
# mod = 'rhe strict'
