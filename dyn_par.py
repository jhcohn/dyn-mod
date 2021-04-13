# basic numeric setup
import numpy as np


def write_runfile(run_base, galaxy='2698', folder='ugc_2698', dynpy='dyndyn_n.py'):
    """

    :param run_base: new run file base, including directory
    :return:
    """

    nest_base = '/Users/jonathancohn/Documents/dyn_mod/dyn_cluster/param_files/'

    with open(nest_base + 'nest_' + galaxy + '_' + run_base + '.lsf', 'w+') as rn:
        rn.write('# NECESSARY JOB SPECIFICATIONS)\n')
        rn.write('# BSUB -J nest_' + galaxy + '_' + run_base + '.lsf\n')
        rn.write('# BSUB -L /bin/bash\n')
        rn.write('# BSUB -W 89:00 #200:00\n')
        rn.write('# BSUB -n 8\n')
        rn.write('# BSUB -R "span[ptile=8]" \n')
        rn.write('# BSUB -R "rusage[mem=2560]"\n')
        rn.write('# BSUB -M 2560\n')
        rn.write('# BSUB -u joncohn@tamu.edu \n')
        rn.write('# BSUB -o out_nest_' + galaxy + '_' + run_base + '.%J\n\n')

        rn.write('ml purge\n')
        rn.write('source /home/joncohn/mydyn.sh\n\n')

        rn.write('python ' + dynpy + ' --p=' + folder + '/' + folder + '_' + run_base + '.txt')

        return nest_base + 'nest_' + galaxy + '_' + run_base + '.lsf'


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


def cube_regions(ds, ds2, gal='2698'):
    """

    :param ds: down-sampling factor (x)
    :param ds2: down-sampling factor (y)
    :param gal: galaxy keyword
    :return: [error region xi, xf, yi, yf], [sub-cube region xi, xf, yi, yf]
    """
    # 2698 fiducial: xi = 84 xf = 168 yi = 118 yf = 182
    # Try to make these ranges divisible by 4, 5, and 10 (e.g. xf-xi=80, 100, 120, 140, 160)
    # 11179: xerr0 = 190 xerr1 = 214 yerr0 = 102 yerr1 = 126
    # 11179: xi = 104  # 108 xf = 204  # 208 yi = 80  # 92 yf = 240  # 232
    # 384: xerr0 = 248 xerr1 = 272 yerr0 = 143 yerr1 = 167
    # 384: xi = 150  # 158 xf = 270  # 262 yi = 132  # 142 yf = 252  # 242

    err_dict = {'2698': {'11': [144, 168, 96, 120],
                         '44': [144, 168, 96, 120],  # [xerr0, xerr1, yerr0, yerr1]
                         '84': [144, 168, 96, 120],
                         '105': [141, 171, 95, 120],
                         '63': [144, 168, 96, 120]},
                '11179': {'44': [190, 214, 102, 126],
                          '84': [190, 214, 102, 126],
                          '105': [187, 217, 101, 126],
                          '63': [190, 214, 102, 126]},
                '384': {'44': [248, 272, 143, 167],
                        '84': [248, 272, 143, 167],
                        '105': [245, 275, 142, 167],
                        '63': [248, 272, 143, 167]}
                }
    sub_dict = {'2698': {'11': [84, 168, 118, 182],
                         '44': [84, 168, 118, 182],  # [xi, xf, yi, yf]
                         '84': [82, 170, 118, 182],
                         '105': [81, 171, 115, 185],
                         '63': [84, 168, 117, 183]},
                '11179': {'44': [104, 204, 80, 240],
                          '84': [102, 206, 80, 240],
                          '105': [104, 204, 80, 240],
                          '63': [103, 205, 79, 241]},
                '384': {'44': [150, 270, 132, 252],
                        '84': [150, 270, 132, 252],
                        '105': [150, 270, 132, 252],
                        '63': [150, 270, 132, 252]}
                }

    return err_dict[gal][str(ds2) + str(ds)], sub_dict[gal][str(ds2) + str(ds)]


def make_the_files(galaxy, base_id, pri, masktype, mgetype, os, gs, sigtype, fixedsig, vrad, kappa, omega, gas, ds, ds2,
                   nlive, dlogz, lucyn, lucyvb=False):
    """
    Main function for handling parameter input options! Functions to write out parameter files and lsf files are called
        inside this function.

    :param galaxy: galaxy keyword ('2698', '384', '11179')
    :param base_id: base id for the run name (e.g. 'finaltests', 'preliminary', etc.)
    :param pri: priorset keyword ('priset3', etc.)
    :param masktype: mask keyword ('baseline', etc.)
    :param mgetype: mge keyword ('rhe', etc.)
    :param os: pixel oversampling factor (4, etc.)
    :param gs: beam grid size (create beam on a grid of gs x gs pixels)
    :param sigtype: sigma type ('flat', 'exp', 'gauss')
    :param fixedsig: False or float (if float: hold the sigma_turb parameter fixed to this value)
    :param vrad: True or False (if True: including radial velocity component as a free parameter, constant with radius)
    :param kappa: True or False (if True: including radial velocity component with kappa)
    :param omega: True or False (if True: including radial velocity component with omega; also uses kappa, although the
        keyword kappa should be kept False)
    :param gas: True or False (if True: include a calculation of the gas mass)
    :param ds: down-sampling factor (ds x ds2 block-averaging) in the x-direction (4, etc.)
    :param ds2: down-sampling factor (ds x ds2 block-averaging) in the y-direction (4, etc.)
    :param nlive: number of live points to use in the dynesty fit (250, etc.)
    :param dlogz: convergence threshold to use in dynesty (0.02, etc.)
    :param lucyn: number of iterations to use in the lucy-richardson deconvolution
    :param lucyvb: True if this is the model where the lucy-input fluxmap has been voronoi-binned before deconvolution
    :return:
    """

    # OLD PARAMS NOW ASSIGNED BY GALAXY
    # :param rfit: fitting ellipse semi-major axis [arcsec] (0.7, etc.)
    # :param zi: first frequency slice index in the subcube [included; start counting at 0] (25, etc.)
    # :param zf: final frequency slice index in the subcube [not included; start counting at 0] (82, etc.)

    # if masktype == 'ext2582':
    #     zi = 25
    #     zf = 82
    # elif masktype == 'ext2285':
    #     zi = 22
    #     zf = 85

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

    err_region, sub_cube = cube_regions(ds, ds2, gal=galaxy)
    xerr0, xerr1, yerr0, yerr1 = err_region
    xi, xf, yi, yf = sub_cube

    dm = '/Users/jonathancohn/Documents/dyn_mod/'
    dc = '/scratch/user/joncohn/dyn_cluster'
    clusterpars = dm + 'dyn_cluster/param_files/'

    if galaxy == '2698':

        zi = 25  # 29
        zf = 82  # 78

        cube = 'UGC2698_C4_CO21_bri_20.3kms.pbcor.fits'
        folder = 'ugc_2698'

        masks = {'baseline': 'UGC2698_C4_CO21_bri_20.3kms_jonathan_casaimviewhand_strictmask2.fits',
                 'strict': 'UGC2698_C4_CO21_bri_20.3kms_jonathan_casaimviewhand_strictmaskstrict.fits',
                 'lax': 'UGC2698_C4_CO21_bri_20.3kms_jonathan_casaimviewhand_strictmasklax.fits',
                 'ext2582': 'UGC2698_C4_CO21_bri_20.3kms_jonathan_casaimviewhand_strictmaskext2582.fits',
                 'ext2285': 'UGC2698_C4_CO21_bri_20.3kms_jonathan_casaimviewhand_strictmaskext2285.fits'}

        if lucyvb:
            lucys = {'baseline': 'ugc_2698_20.3_strict2_voronoibin_lucyout_n' + str(lucyn) + '.fits'}
        else:
            lucys = {'baseline': 'ugc_2698_20.3_strict2_lucyout_n' + str(lucyn) + '.fits',
                     'strict': 'ugc_2698_20.3_strictstrict_lucyout_n' + str(lucyn) + '.fits',
                     'lax': 'ugc_2698_20.3_strictlax_lucyout_n' + str(lucyn) + '.fits',
                     'ext2582': 'ugc_2698_20.3_strict2582_lucyout_n' + str(lucyn) + '.fits',
                     'ext2285': 'ugc_2698_20.3_strict2285_lucyout_n' + str(lucyn) + '.fits'}

        if gs != 31:
            lucys = {'baseline': 'ugc_2698_20.3_strict2_lucyout_g' + str(gs) + '_n' + str(lucyn) + '.fits',
                     'strict': 'ugc_2698_20.3_strictstrict_lucyout_g' + str(gs) + '_n' + str(lucyn) + '.fits',
                     'lax': 'ugc_2698_20.3_strictlax_lucyout_g' + str(gs) + 'n' + str(lucyn) + '.fits',
                     'ext2582': 'ugc_2698_20.3_strict2582_lucyout_g' + str(gs) + '_n' + str(lucyn) + '.fits',
                     'ext2285': 'ugc_2698_20.3_strict2285_lucyout_g' + str(gs) + '_n' + str(lucyn) + '.fits'}

        lucy_masks = {'baseline': 'ugc_2698_collapsemask_20.3kms_jonathan_casaimview_strictmask2.fits',
                      'strict': 'ugc_2698_collapsemask_20.3kms_jonathan_casaimview_strictmaskstrict.fits',
                      'lax': 'ugc_2698_collapsemask_20.3kms_jonathan_casaimview_strictmasklax.fits',
                      'ext2582': 'ugc_2698_collapsemask_20.3kms_jonathan_casaimview_strictmask2582.fits',
                      'ext2285': 'ugc_2698_collapsemask_20.3kms_jonathan_casaimview_strictmask2285.fits'}

        if lucyvb:
            fluxmaps = {'baseline': 'ugc_2698_fluxmap_20.3kms_strictmask2_voronoibin.fits'}
        else:
            fluxmaps = {'baseline': 'ugc_2698_fluxmap_20.3kms_jonathan_casaimview_strictmask2.fits',
                        'strict': 'ugc_2698_fluxmap_20.3kms_jonathan_casaimview_strictmask2strict.fits',
                        'lax': 'ugc_2698_fluxmap_20.3kms_jonathan_casaimview_strictmasklax.fits',
                        'ext2582': 'ugc_2698_fluxmap_20.3kms_jonathan_casaimview_strictmask2582.fits',
                        'ext2285': 'ugc_2698_fluxmap_20.3kms_jonathan_casaimview_strictmask2285.fits'}

        mges = {'ahe': 'ugc_2698_ahe_mge.txt', 'rhe': 'ugc_2698_rhe_mge.txt', 'rre': 'ugc_2698_rre_mge.txt',
                'ahepsf': 'ugc_2698_ahepsf_mge.txt', 'rhepsf': 'ugc_2698_rhepsf_mge.txt',
                'rrepsf': 'ugc_2698_rrepsf_mge.txt',
                'akin': 'yildirim_table_2698.txt'}

        xell = 126.85
        yell = 150.9
        xrange = [116, 140]
        yrange = [140, 160]
        theta_ell = 19.
        parange = [0, 35]
        q_ell = 0.38
        rfit = 0.7

        # HEADER INFO & DISTANCE
        resolution = 0.02  # arcsec/pix
        x_fwhm = 0.197045  # arcsec
        y_fwhm = 0.103544  # arcsec
        PAbeam = 9.271  # deg
        dist = 91.  # Mpc
        vsysrange = [6405, 6505]
        zfix = 0.02152

    elif galaxy == '11179':  # BUCKET: UPDATE  (done): masks (baseline), lucys (baseline), lucy_masks (baseline),
        # fluxmaps (baseline), mges (akin)
        cube = 'PGC11179_C4_CO21_bri_MS_20kms.pbcor.fits'
        folder = 'pgc_11179'

        zi = 6  # ~4 slices before first with emission
        zf = 56  # ~4 slices after last with emission

        masks = {'baseline': 'PGC11179_C4_CO21_bri_MS_20kms_jonathan_casaimviewhand_strictmask.fits',
                 'strict': 'PGC11179_C4_CO21_bri_MS_20kms_jonathan_casaimviewhand_strictmaskstrict.fits',
                 'lax': 'PGC11179_C4_CO21_bri_MS_20kms_jonathan_casaimviewhand_strictmasklax.fits',
                 'ext0756': 'PGC11179_C4_CO21_bri_MS_20kms_jonathan_casaimviewhand_strictmaskext0756.fits'}

        lucys = {'baseline': 'pgc_11179_20_strict_lucyout_n' + str(lucyn) + '.fits',
                 'strict': 'pgc_11179_20_strictstrict_lucyout_n' + str(lucyn) + '.fits',
                 'lax': 'pgc_11179_20_strictlax_lucyout_n' + str(lucyn) + '.fits',
                 'ext0756': 'pgc_11179_20_strict0756_lucyout_n' + str(lucyn) + '.fits'}

        if gs != 31:
            lucys = {'baseline': 'pgc_11179_20_strict_lucyout_g' + str(gs) + '_n' + str(lucyn) + '.fits',
                     'strict': 'pgc_11179_20_strictstrict_lucyout_g' + str(gs) + '_n' + str(lucyn) + '.fits',
                     'lax': 'pgc_11179_20_strictlax_lucyout_g' + str(gs) + '_n' + str(lucyn) + '.fits',
                     'ext0756': 'pgc_11179_20_strict0756_lucyout_g' + str(gs) + '_n' + str(lucyn) + '.fits'}

        lucy_masks = {'baseline': 'pgc_11179_collapsemask_20kms_jonathan_casaimview_strictmask.fits',
                      'strict': 'pgc_11179_collapsemask_20kms_jonathan_casaimview_strictmaskstrict.fits',
                      'lax': 'pgc_11179_collapsemask_20kms_jonathan_casaimview_strictmasklax.fits',
                      'ext0756': 'pgc_11179_collapsemask_20kms_jonathan_casaimview_strictmask0756.fits'}

        fluxmaps = {'baseline': 'pgc_11179_fluxmap_20kms_jonathan_casaimview_strictmask.fits',
                    'strict': 'pgc_11179_fluxmap_20kms_jonathan_casaimview_strictmaskstrict.fits',
                    'lax': 'pgc_11179_fluxmap_20kms_jonathan_casaimview_strictmasklax.fits',
                    'ext0756': 'pgc_11179_fluxmap_20kms_jonathan_casaimview_strictmask0756.fits'}

        mges = {'ahe': 'pgc_11179_ahe_mge.txt', 'rhe': 'pgc_11179_rhe_mge.txt', 'rre': 'pgc_11179_rre_mge.txt',
                'ahepsf': 'pgc_11179_ahepsf_mge.txt', 'rhepsf': 'pgc_11179_rhepsf_mge.txt',
                'rrepsf': 'pgc_11179_rrepsf_mge.txt',
                'akin': 'pgc_11179_mge_akin.txt'}  # 'yildirim_table_11179.txt'}

        xell = 155.  # pix
        yell = 155.  # pix
        xrange = [150., 160.]  # [140., 160.]
        yrange = [150., 160.]  # [150., 170.]
        theta_ell = 70.  # ellipse PA [deg] (counter-clockwise from x)
        parange = [60., 90.]
        q_ell = 0.35  # akin MGE min q=0.2793 -> arccos(sqrt((400*.2793^2 - 1.)/399.)) = 74.03 deg ;; take i~80. deg
        rfit = 1.8

        # HEADER INFO & DISTANCE
        resolution = 0.03  # arcsec/pix
        x_fwhm = 0.29  # arcsec
        y_fwhm = 0.16  # arcsec
        PAbeam = 86.97  # 86.78  # PA of beam [deg] (as defined in casa/the ALMA data)
        dist = 98.  # Mpc
        # https://ned.ipac.caltech.edu/byname?objname=PGC%2011179&hconst=67.8&omegam=0.308&omegav=0.692&wmap=4&corr_z=1
        vsysrange = [6600, 7600]
        zfix = 0.02289

    elif galaxy == '384':  # BUCKET: UPDATE (done): masks (baseline), lucys (baseline), lucy_masks (baseline), fluxmaps
        # (baseline), mges (akin)
        cube = 'NGC384_C4_CO21_bri_20kms.pbcor.fits'
        folder = 'ngc_384'

        zi = 40  # ~4 slices before first with emission
        zf = 83  # ~4 slices after last with emission

        masks = {'baseline': 'NGC384_C4_CO21_bri_20kms_jonathan_casaimviewhand_strictmask.fits',
                 'strict': 'NGC384_C4_CO21_bri_20kms_jonathan_casaimviewhand_strictmaskstrict.fits',
                 'lax': 'NGC384_C4_CO21_bri_20kms_jonathan_casaimviewhand_strictmasklax.fits',
                 'ext3983': 'NGC384_C4_CO21_bri_20kms_jonathan_casaimviewhand_strictmaskext3983.fits'}

        lucys = {'baseline': 'ngc_384_20_strict_lucyout_n' + str(lucyn) + '.fits',
                 'strict': 'ngc_384_20_strictstrict_lucyout_n' + str(lucyn) + '.fits',
                 'lax': 'ngc_384_20_strictlax_lucyout_n' + str(lucyn) + '.fits',
                 'ext3983': 'ngc_384_20_strict3983_lucyout_n' + str(lucyn) + '.fits'}

        if gs != 31:
            lucys = {'baseline': 'ngc_384_20_strict_lucyout_g' + str(gs) + '_n' + str(lucyn) + '.fits',
                     'strict': 'ngc_384_20_strictstrict_lucyout_g' + str(gs) + '_n' + str(lucyn) + '.fits',
                     'lax': 'ngc_384_20_strictlax_lucyout_g' + str(gs) + '_n' + str(lucyn) + '.fits',
                     'ext3983': 'ngc_384_20_strict3983_lucyout_g' + str(gs) + '_n' + str(lucyn) + '.fits'}

        lucy_masks = {'baseline': 'ngc_384_collapsemask_20kms_jonathan_casaimview_strictmask.fits',
                      'strict': 'ngc_384_collapsemask_20kms_jonathan_casaimview_strictmaskstrict.fits',
                      'lax': 'ngc_384_collapsemask_20kms_jonathan_casaimview_strictmasklax.fits',
                      'ext3983': 'ngc_384_collapsemask_20kms_jonathan_casaimview_strictmask3983.fits'}

        fluxmaps = {'baseline': 'ngc_384_fluxmap_20kms_jonathan_casaimview_strictmask.fits',
                    'strict': 'ngc_384_fluxmap_20kms_jonathan_casaimview_strictmaskstrict.fits',
                    'lax': 'ngc_384_fluxmap_20kms_jonathan_casaimview_strictmasklax.fits',
                    'ext3983': 'ngc_384_fluxmap_20kms_jonathan_casaimview_strictmask3983.fits'}

        mges = {'ahe': 'ngc_384_ahe_mge.txt', 'rhe': 'ngc_384_rhe_mge.txt', 'rre': 'ngc_384_rre_mge.txt',
                'ahepsf': 'ngc_384_ahepsf_mge.txt', 'rhepsf': 'ngc_384_rhepsf_mge.txt',
                'rrepsf': 'ngc_384_rrepsf_mge.txt',
                'akin': 'ngc_384_mge_akin.txt'}  # 'yildirim_table_384.txt'}

        xell = 208.  # pix
        yell = 193.  # pix
        xrange = [200., 220.]
        yrange = [180., 210.]
        theta_ell = 43.  # ellipse PA [deg] (counter-clockwise from x)
        parange = [15., 75.]
        q_ell = 0.50  # akin mge min_q=0.5361 -> 57.7 deg -> take inc=60 deg, q~.50
        rfit = 1.8

        # HEADER INFO & DISTANCE
        resolution = 0.04  # arcsec/pix
        x_fwhm = 0.301921666  # arcsec
        y_fwhm = 0.161281541  # arcsec
        PAbeam = 16.17017173767  # deg
        dist = 61.  # Mpc
        # https://ned.ipac.caltech.edu/byname?objname=NGC%20384&hconst=67.8&omegam=0.308&omegav=0.692&wmap=4&corr_z=1
        vsysrange = [4000, 4500]
        zfix = 0.01412

    # CREATE FILE/MODEL RUN NAME
    dfid = ''  # difference from fiducial model!
    if pri != 'priset3':
        dfid += '_' + pri
    if masktype != 'baseline':
        dfid += '_' + masktype
    if mgetype != 'rhe':
        dfid += '_' + mgetype
    if os != 4:
        dfid += '_os' + str(os)
    if gs != 31:
        dfid += '_' + str(gs)
    if fixedsig:
        dfid += '_fixsig' + str(fixedsig)
    if sigtype != 'flat':
        dfid += '_' + sigtype
    if rfit != 0.7:  # != 0.7: (for UGC 2698); != 1.8: (for NGC 384 and PGC 11179)
        dfid += '_rfit' + str(rfit)
    if vtype != 'orig':
        dfid += '_' + vtype
    if gas_label != 'nog':
        dfid += '_' + gas_label
    if ds != 4 or ds2 != 4:
        dfid += '_ds' + str(ds) + str(ds2)
    if dlogz != 0.02:
        dfid += '_dlogz' + str(dlogz)
    if nlive != 250:
        dfid += '_nlive' + str(nlive)
    if gs != 31:
        dfid += '_gs' + str(gs)
    if lucyn != 10:
        dfid += '_lucyn' + str(lucyn)
    if lucyvb:
        dfid += '_lucyvb'

    if dfid == '':
        dfid += '_fiducial'
    run_type = base_id + dfid

    # SET PARAM FILE LOCATIONS
    localpars = dm + folder + '/'
    clusterloc = dc + '/' + folder + '/'

    # BUCKET: UPDATE BELOW FOR 2698, 384, AND 11179

    # FIXED PARAMS
    # DYNESTY PARAMS
    dyn_pars = {'nlive': ['int', nlive], 'nprocs': ['int', 8], 'maxc': ['int', 10000000], 'thresh': ['float', dlogz]}

    fixed_str = {  # OUTPUT RUN BASE NAME & OTHER FIXED STRINGS
        'outname': folder + '_' + run_type,  # 'u2698_'
        's_type': sigtype,
        'vtype': vtype,  # orig, vrad, omega, kappa
        'incl_gas': str(gas)}

    fixed = {  # FIXED FLOATS: ELLIPTICAL REGION PARS, HEADER INFO, ETC.
        'xell': xell, 'yell': yell, 'theta_ell': theta_ell, 'q_ell': q_ell, 'rfit': rfit,
        # yell between 150.8~few to 150.9; could do 150.85; or round so xell 127, yell 151, qell .36-.38, adjust rfit?
        # theta_ell 18.88 : 19.19 ;; q_ell cos(67.7 deg) ;; rfit arcsec
        # HEADER INFO & DISTANCE
        'resolution': resolution,  # arcsec/pix
        'x_fwhm': x_fwhm,
        'y_fwhm': y_fwhm,
        'PAbeam': PAbeam,
        'dist': dist,  # 91.  # 85.29,  #89.,  # Mpc'
        'zfix': zfix
        }

    if 'pri' == 'weird' or 'pri' == 'pretest':
        fixed['inc'] = 89.9  # 11179
        fixed['PAdisk'] = 70.  # 11179

    fixed_int = {  # FIXED INTs
        'zi': zi,  # cube range: zi:zf,
        'zf': zf,
        'xi': xi,  # 84
        'xf': xf,  # 168
        'yi': yi,  # 118
        'yf': yf,  # 182
        'xerr0': xerr0,  # 144 # xerr & yerr must be divisible by ds
        'xerr1': xerr1,  # 168
        'yerr0': yerr0,  # 96
        'yerr1': yerr1,  # 120
        'os': os,
        'ds': ds,
        'ds2': ds2,
        'gsize': gs,
        'mtype': 0,  # mge file
        'bl': 0,
        'lucy_it': lucyn}

    priors = {'mbh': [0, 12], 'xloc': xrange, 'yloc': yrange, 'sig0': [0, 200], 'inc': [0, 89],
              'PAdisk': [0, 90], 'vsys': [5000, 8100], 'ml_ratio': [0.1, 10], 'f': [0.1, 2.5]}
    midpriors = {'mbh': [9., 9.8], 'xloc': [125.5, 129], 'yloc': [149, 153], 'sig0': [5, 25], 'inc': [63, 73],
                 'PAdisk': [16, 22], 'vsys': [6440, 6490], 'ml_ratio': [1, 2.5], 'f': [0.7, 1.4]}
    narrowpri = {'mbh': [9.1, 9.7], 'xloc': [125.5, 128], 'yloc': [150, 152], 'sig0': [12, 24], 'inc': [65, 71],
                 'PAdisk': [15, 22], 'vsys': [6445, 6465], 'ml_ratio': [1.3, 2.4], 'f': [0.9, 1.2]}
    priset2 = {'mbh': [8.5, 9.7], 'xloc': [124, 128], 'yloc': [148, 152], 'sig0': [5, 25], 'inc': [0, 85],
               'PAdisk': [10, 30], 'vsys': [6425, 6485], 'ml_ratio': [0.8, 2.4], 'f': [0.5, 1.5]}
    # priset3 = {'mbh': [8, 10.], 'xloc': [124, 128], 'yloc': [148, 152], 'sig0': [0, 40], 'inc': [0, 89],
    #            'PAdisk': [5, 35], 'vsys': [6405, 6505], 'ml_ratio': [0.3, 3.], 'f': [0.5, 1.5]}

    prepri = {'mbh': [7.5, 10.5], 'xloc': xrange, 'yloc': yrange, 'sig0': [0, 200], 'inc': [0, 89.9],
              'PAdisk': parange, 'vsys': vsysrange, 'ml_ratio': [0.1, 10.], 'f': [0.2, 5.]}

    priset3 = {'mbh': [8., 10.], 'xloc': xrange, 'yloc': yrange, 'sig0': [0, 100], 'inc': [0, 89.9],
               'PAdisk': parange, 'vsys': vsysrange, 'ml_ratio': [0.3, 3.], 'f': [0.5, 1.5]}

    fullpriors = {'mbh': [6, 12], 'xloc': [116, 140], 'yloc': [140, 160], 'sig0': [0, 200], 'inc': [0, 89],
                  'PAdisk': [0, 89], 'vsys': [5000, 8100], 'ml_ratio': [0.1, 10], 'f': [0.1, 2.5]}

    weird_11179 = {'mbh': [0., 11.], 'xloc': [143, 173], 'yloc': [133, 163], 'sig0': [0, 1e3], 'vsys': [6800, 7200],
                   'ml_ratio': [0., 5.], 'f': [0.2, 2.]}

    # COULD BE FREE OR FIXED: vrad, kappa, omega (depends on vtype)
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

    # USING FIXED SIGMA?
    if fixedsig:
        del priors['sig0']
        del midpriors['sig0']
        del narrowpri['sig0']
        del priset2['sig0']
        del priset3['sig0']
        del fullpriors['sig0']
        del prepri['sig0']
        fixed['sig0'] = fixedsig

    # COULD BE FREE OR FIXED: r0, sig1, mu (depends on sigtype)
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
    localfiles = {'data': localpars + cube,
                  'mask': localpars + masks[masktype], 'lucy': localpars + lucys[masktype],
                  'lucy_mask': localpars + lucy_masks[masktype], 'lucy_in': localpars + fluxmaps[masktype],
                  'lucy_b': localpars + folder + '_beam' + str(gs) + '.fits',
                  'lucy_o': localpars + lucys[masktype] + '[0]',
                  'mass': localpars + mges[mgetype]}

    # CLUSTER FILES
    clusterfiles = {'data': clusterloc + cube,
                    'mask': clusterloc + masks[masktype], 'lucy': clusterloc + lucys[masktype],
                    'lucy_mask': localpars + lucy_masks[masktype], 'lucy_in': clusterloc + fluxmaps[masktype],
                    'lucy_b': clusterloc + folder + '_beam' + str(gs) + '.fits',
                    'lucy_o': clusterloc + lucys[masktype] + '[0]', 'mass': clusterloc + mges[mgetype]}

    newpar = folder + '_' + run_type + '.txt'
    locnewpar = localpars + newpar
    clusternewpar = clusterpars + newpar

    # copyf = localpars + folder + 'ugc_2698_baseline_rhe_orig_gas.txt'

    # SET PRIOR CHOICE
    prior_dict = {'wide': priors,
                  'mid': midpriors,
                  'narrow': narrowpri,
                  'prepri': prepri,
                  'priset2': priset2,
                  'priset3': priset3,
                  'fullpriors': fullpriors,
                  'weird': weird_11179}

    use_priors = prior_dict[pri]

    pyfile = 'dyndyn_all.py'
    '''  #
    pyfile = 'dyndyn_n.py'
    if 'fixbh' in base_id:
        use_priors.pop('mbh')
        fixed['mbh'] = base_id[5:]  # base_id must be in format fixbhMBH e.g. fixbh2.46e9, so base_id[5:] = 2.46e9
        pyfile = 'dyndyn_fixbh.py'
    # '''  #
    #use_priors = priors
    #if pri == 'wide':
    #    use_priors = priors
    #elif pri == 'mid':
    #    use_priors = midpriors
    #elif pri == 'narrow':
    #    use_priors = narrowpri
    #elif pri == 'priset2':
    #    use_priors = priset2
    #elif pri == 'priset3':
    #    use_priors = priset3

    # MAKE THE LOCAL VERSION
    # newpl = write_newpar(locnewpar, copyf, use_priors, fixed, localfiles)
    newpl = newpar_scratch(locnewpar, use_priors, dyn_pars, fixed_str, fixed, fixed_int, localfiles)
    print(newpl)

    # AND THE CLUSTER VERSION TO SCP OVER
    # newpc = write_newpar(clusternewpar, copyf, use_priors, fixed, clusterfiles)
    newpc = newpar_scratch(clusternewpar, use_priors, dyn_pars, fixed_str, fixed, fixed_int, clusterfiles)
    print(newpc)

    # THEN WRITE CLUSTER RUNFILE TO SCP OVER
    submit_file = write_runfile(run_type, galaxy=galaxy, folder=folder, dynpy=pyfile)
    print(submit_file)


# SETTINGS / CHOICES
galaxy = '2698'  # '11179' # 2698, 11179, 384
# bid = 'fixbh7500000000' # 'weird'  # preliminary, finaltests, pre2, pretest, pre3, weird
bid = 'finaltests'
# fixbh numbers: median 2461189947.064265, +/-1sig 2526552381.586655,2395489164.9312243 ;;
# +/-3sig 2667403123.112414,2275153534.726537 ;; +/-systemic 2.46+0.70=3160000000,2.46-0.78=1680000000
pri = 'priset3'  # 'prepri', 'wide', 'mid', 'narrow', 'priset2', 'priset3' (fiducial), 'fullpriors'
masktype = 'baseline'  # 'strict'  # 'lax'  # 'baseline'
mgetype = 'rhe'  # 'rhe'  # 'rhe'  # 'ahe'  # 'rhe'  # 'rre'  # 'akin'
os = 4  # 1 2 3 4 6 8 10 12 14 16
gs = 31  # beam grid size
fixedsig = 10  # hold turbulent velocity dispersion (sigma) fixed: float or False
sigtype = 'flat'  # flat, exp, gauss
# rfit = 0.7
vrad = False  # include radial velocity
kappa = False  # include radial velocity with kappa
omega = False  # include radial velocity with sub-Keplerian motion
gas = False  # incl_gas
# zi = 25  # 29
# zf = 82  # 78
# total pixels in xi:xf must be divisible by ds2; total pixels in yi:yf must be divisible by ds (same for xerr & yerr)
ds = 4
ds2 = 4
####### ds = #4x4  #8x4  #10x5 #6x3  #10x10
#xerr0 = 144  # 144 # 144 # 141 # 144 # 141
#xerr1 = 168  # 168 # 168 # 171 # 168 # 171
#yerr0 = 96   # 96  # 96  # 95  # 96  # 93
#yerr1 = 120  # 120 # 120 # 120 # 120 # 123
#xi = 84      # 84  # 82  # 81  # 84  # 81
#xf = 168     # 168 # 170 # 171 # 168 # 171
#yi = 118     # 118 # 118 # 115 # 117 # 115
#yf = 182     # 182 # 182 # 185 # 183 # 185
# ds=4x4 -> xi=84,xf=168, yi=118,yf=182. ds=8x4 -> xi=82,xf=170, yi=118,yf=182. ds=10x5 -> xi=81,xf=171, yi=115,yf=185.
# ds=6x3 -> xi=84,xf=168, yi=117,yf=183. ds=10x10 -> xi=81,xf=171, yi=115, yf=185
# ds=4x4 OR =8x4 OR =6x3 -> xerr0=144,xerr1=168, yerr0=96,yerr1=120. ds=10x5 -> xerr0=141,xerr1=171, yerr0=95,yerr1=120.
# ds=10x10 -> xerr0=141,xerr1=171, yerr0=93,yerr1=123.
# confirmed: this error region has no overlap with the strictmask
lucyn = 10  # 5 10 15
lucyvb = False  # False  True
nlive = 250  # 250  # 1000
dlogz = 0.02  # 0.02  # 0.001

# MAKE THE FILES!
# for os in [1, 2, 3, 4, 6, 8, 10, 12]:
# for mgetype in ['rhe', 'ahe', 'rre']:
#for rfit in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
#    make_the_files(galaxy, pri, masktype, mgetype, os, gs, sigtype, rfit, vrad, kappa, omega, gas, zi, zf, ds, ds2, nlive,
#                   dlogz)
# USED FOR 2698
# make_the_files(galaxy, pri, masktype, mgetype, os, gs, sigtype, rfit, vrad, kappa, omega, gas, zi, zf, ds, ds2, nlive,
#                dlogz, lucyn, lucyvb=lucyvb)
# MADE MORE GENERAL:
make_the_files(galaxy, bid, pri, masktype, mgetype, os, gs, sigtype, fixedsig, vrad, kappa, omega, gas, ds, ds2, nlive,
               dlogz, lucyn, lucyvb)


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

'''
hdu = fits.open('ugc_2698/ugc_2698_20.3_strict2_lucyout_n15.fits')
data15 = hdu[0].data  # data[0] contains: z, y, x (121, 700, 700)
hdu.close()
hdu = fits.open('ugc_2698/ugc_2698_20.3_strict2_lucyout_n10.fits')
data10 = hdu[0].data  # data[0] contains: z, y, x (121, 700, 700)
hdu.close()
hdu = fits.open('ugc_2698/ugc_2698_20.3_strict2_lucyout_n5.fits')
data5 = hdu[0].data  # data[0] contains: z, y, x (121, 700, 700)
hdu.close()
fig, ax = plt.subplots(2, 3)
im5 = ax[0][0].imshow(data5, origin='lower')
cbar5 = fig.colorbar(im5, ax=ax[0][0], pad=0.02)
im10 = ax[0][1].imshow(data10, origin='lower')
cbar10 = fig.colorbar(im10, ax=ax[0][1], pad=0.02)
im15 = ax[0][2].imshow(data15, origin='lower')
cbar15 = fig.colorbar(im15, ax=ax[0][2], pad=0.02)
im5r = ax[1][0].imshow((data5 - data10)/data10, origin='lower')
cbar5r = fig.colorbar(im5r, ax=ax[1][0], pad=0.02)
im10r = ax[1][1].imshow((data10 - data10)/data10, origin='lower')
cbar10r = fig.colorbar(im10r, ax=ax[1][1], pad=0.02)
im15r = ax[1][2].imshow((data15 - data10)/data10, origin='lower')
cbar15r = fig.colorbar(im15r, ax=ax[1][2], pad=0.02)
plt.show()
print(oop)
'''