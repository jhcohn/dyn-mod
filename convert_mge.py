import numpy as np
import argparse


def paper_to_galfit(table_file, mag_sol=3.32, pix_scale=0.06, t_exp=1354.46, zp=24.6949, apcorr=0.0, dust=0.075,
                    xctr=None, yctr=None, pa=-7.5697, dist=89.*1e6, img=None, mask=None, copy_file=None,
                    galfit_out=None, new_galfit=None, convert_to_app=True, compare_file=None):
    """
    Convert from table built from Akin's table.pdf units to an MGE with units I can use as input for GALFIT, and write
    that new GALFIT file!

    :param table_file: table file e.g. from Akin, with cols component_number, I_Lsol/pc^2, sigma_arcsec, q
    :param mag_sol: absolute magnitude of the sun [in relevant band in question; in Akin's/our case, H-band]
    :param pix_scale: pixel scale [arcsec/pix]
    :param t_exp: exposure time [s]
    :param zp: zeropoint [of relevant band in question; in our/Akin's case, F160W (H-band)]
    :param apcorr: infinite aperture correction, applied for surface brightness measurements (readme_mge_fit_sectors)
    :param dust: assumed extinction correction [for relevant band in question; in our/Akin's case, H-band]
    :param xctr: x center
    :param yctr: y center
    :param pa: position angle
    :param dist: distance to the galaxy [pc]
    :param img: input image
    :param mask: mask image
    :param copy_file: GALFIT input file of which we're copying the structure
    :param galfit_out: filename to use as the output data image block in the new GALFIT file we're writing
    :param new_galfit: the new GALFIT file we're writing
    :param convert_to_app: if True, convert from absolute to apparent magnitude
    """

    # Akin's table.pdf caption (TABLE 15) lists M_Hsol = 3.32
    # Akin's caption makes an assumption of a dust correction of 0.075 mags
    # NOTE: Akin's image is 0.06"/pix; exposure time = 1354.46 seconds
    # Use eqns 2 and 3 from readme_mge_fit_sectors.pdf to convert Akin's table to peak counts C_0, then eqn 1 for C_0->
    # total counts, then total counts -> integrated mag as usual.
    # eqn3: I_H [Lsol/pc^2] = (64800/pi)^2 * 10^(0.4*(Msol_H - mu_H))
    ## Solve for mu_H! mu_H = Msol_H - log10((np.pi / 64800)**2 * float(cols[1]) *****
    # eqn2: mu_H = zeropoint + 0.1 + 5log10(pix_scale) + 2.5log10(exp_time) - 2.5log(C_0) - A_H
    ## pix_scale is defined s.t. sigma_pix * pix_scale = sigma_arcsec
    # = 25.95 + 0.1 + 5log10(0.06) + 2.5log10(1354.46) - 2.5log10(C_0) - 0.075
    ## Solve for C_0: mu_H - 25.95 - 0.1 + 0.075 = 5log10(0.06) + 2.5(log10(1354.46) - log10(C_0))
    ### Let consts = -25.95 - 0.1 + 0.075
    ### 10^((mu_H + consts)/2.5) = pix_scale^2  * (exp_time / C_0)
    ### C_0 = pix_scale**2 * exp_time * 10**(-0.4 * (mu_H+consts)) *****
    # eqn1: C_0 = total_counts / (2*pi*sigma_pix**2 * qObs)
    ## Solve for total_counts: total_counts = C_0 * (2 * np.pi * sigma_pix**2 * qObs) *****
    # Finally: mags = zeropoint - 2.5 * log10(total_counts) *****

    mags = []
    fwhms = []
    qs = []
    with open(table_file, 'r') as tf:
        for line in tf:
            if not line.startswith('#'):
                cols = line.split()
                mu = mag_sol - 2.5 * np.log10((np.pi / 64800)**2 + float(cols[1]))  # cols[1] = I_H [Lsol/pc^2]
                c0 = pix_scale**2 * t_exp * 10**(0.4 * (zp + apcorr - dust - mu))
                total_counts = c0 * 2 * np.pi * (float(cols[2]) / pix_scale)**2 * float(cols[3])
                mag = zp - 2.5*np.log10(total_counts)
                if convert_to_app:
                    mag = mag + 5. * np.log10(dist) - 5.
                mags.append(mag)  # above: cols[2] = sigma_arcsec, cols[3] = qObs
                fwhms.append(2.355 * float(cols[2]) / pix_scale)  # fwhm = 2.355 * np.array(sigma_arcsec) / pix_scale
                qs.append(float(cols[3]))

    if compare_file is not None:
        with open(compare_file, 'w+') as cf:
            cf.write('# Component Integrated_mags FWHM_pix qObs\n')
            for m in range(len(mags)):
                cf.write(str(m+1) + ' ' + str(mags[m]) + ' ' + str(fwhms[m]) + ' ' + str(qs[m]) + '\n')
        print(tada)

    with open(new_galfit, 'w+') as new_g:
        with open(copy_file, 'r') as cf:
            for line in cf:
                wline = line
                if not line.startswith('# Component number: '):
                    if img is not None and line.startswith('A)'):
                        wline = 'A) '  + img + '      # Input data image (FITS file)' + '\n'
                    elif line.startswith('B)'):
                        wline = 'B) '  + galfit_out + '      # Output data image block' + '\n'
                    elif mask is not None and line.startswith('F)'):
                        wline = 'F) '  + mask + '      # Bad pixel mask (FITS image or ASCII coord list)' + '\n'
                    elif line.startswith('G)'):
                        wline = '# G) None      # File with parameter constraints (ASCII file)' + '\n'
                        # No constraint file!
                    elif line.startswith('P)'):
                        wline = 'P) 1                   # Choose: 0=optimize, 1=model, 2=imgblock, 3=subcomps' + '\n\n'
                    elif line.startswith('H)') or line.startswith('I)') or line.startswith('J)') or \
                            line.startswith('K)') or line.startswith('O)'):
                        wline = line
                    new_g.write(wline)
                else:
                    break

        for i in range(len(mags)):
            new_g.write('\n')
            new_g.write('# Component number: ' + str(i+1) + '\n')
            new_g.write(' 0) gaussian               #  Component type\n')
            new_g.write(' 1) ' + str(yctr+1) + ' '  + str(xctr+1) + ' 2 2  #  Position x, y\n')  # xy reversed in python
            new_g.write(' 3) ' + str(mags[i]) + ' 0 # Integrated magnitude\n')
            new_g.write(' 4) ' + str(fwhms[i]) + ' 0 # FWHM [pix]\n')
            new_g.write(' 5) 0.0000      0          #     ----- \n')
            new_g.write(' 6) 0.0000      0          #     ----- \n')
            new_g.write(' 7) 0.0000      0          #     ----- \n')
            new_g.write(' 8) 0.0000      0          #     ----- \n')
            new_g.write(' 9) ' + str(qs[i]) + ' 0 # Axis ratio (b/a)\n')
            new_g.write('10) ' + str(pa) + '     2          #  Position angle (PA) [deg: Up=0, Left=90]\n')
            new_g.write(' Z) 0                      #  Skip this model in output image?  (yes=1, no=0)\n')


def mge_to_galfit(mge_file, zeropoint, img=None, mask=None, copy_file=None, galfit_out=None, write_new=None,
                  new_galfit=None, constraint=None, texp=None, sky=None, rms=None, gain=1.0):  # pa=-7.5697

    mags = []
    fwhms = []
    qs = []
    xs = []
    ys = []
    pas = []

    with open(mge_file, 'r') as f:
        for line in f:
            cols = line.split()
            if not line.startswith('#'):
                if texp is not None:
                    mags.append(zeropoint - 2.5 * np.log10(float(cols[0]) / texp))  # - 2.5*np.log10(gain)
                else:
                    mags.append(zeropoint - 2.5 * np.log10(float(cols[0])))
                fwhms.append(2 * np.sqrt(2 * np.log(2.)) * float(cols[1]))  # FWHM =~ 2.355 sigma
                qs.append(float(cols[2]))
                xs.append(float(cols[3]))
                ys.append(float(cols[4]))
                pas.append(90. - float(cols[5]))
                # MGE find_galaxy: Position angle measured clock-wise from the image X axis
                # GALFIT: PA = 0 if the semi-major axis is parallel to the Y-axis, and increases counter-clockwise
                # gain=2.5 for WFC3 and untis of counts (not counts/sec)
                # For counts: need to adjust mag guesses by -2.5log10(counts_zp) (counts_zp = texp (/gain?))

    if write_new is not None:
        with open(write_new, 'w+') as newfile:
            newfile.write('# UGC 2698 MGE output using mge_fit_mine.py, converted to GALFIT input units\n')
            newfile.write('# param file: ' + new_galfit + '\n')
            newfile.write('# pa = the GALFIT position angle: "The position angle is 0 if the semi-major axis is aligned'
                          ' parallel to the Y-axis and increases toward the counter-clockwise direction.\n')
            newfile.write('# Mags FWHM_pixels qObs xc yc pa\n')  # PAs
            for i in range(len(mags)):
                newfile.write(str(mags[i]) + ' ' + str(fwhms[i]) + ' ' + str(qs[i]) + ' ' + str(ys[i]) + ' ' +
                              str(xs[i]) + ' ' + str(pas[i]) + '\n')  # x,y reversed in python

    print(len(mags))
    with open(new_galfit, 'w+') as new_g:
        with open(copy_file, 'r') as cf:
            for line in cf:
                wline = line
                if not line.startswith('# Component number: '):
                    if line.startswith('#  Chi'):
                        wline = '# run with: galfit -skyped ' + str(sky) + ' -skyrms ' + str(rms) + ' ' + new_galfit +\
                                '\n'
                    elif line.startswith('A)'):
                        wline = 'A) '  + img + '      # Input data image (FITS file)\n'
                    elif line.startswith('B)'):
                        wline = 'B) '  + galfit_out + '      # Output data image block\n'
                    elif line.startswith('F)'):
                        wline = 'F) '  + mask + '      # Bad pixel mask (FITS image or ASCII coord list)\n'
                    elif line.startswith('G)'):
                        wline = 'G) '  + constraint + '      # File with parameter constraints (ASCII file)\n'
                    elif line.startswith('J)'):
                        wline = 'J) ' + str(zeropoint) + '              # Magnitude photometric zeropoint \n'
                    elif line.startswith('H)') or line.startswith('I)') or line.startswith('K)') or \
                            line.startswith('O)') or line.startswith('P)'):
                        wline = line
                    new_g.write(wline)
                else:
                    break

        for i in range(len(mags)):
            new_g.write('\n')
            new_g.write('# Component number: ' + str(i+1) + '\n')
            new_g.write(' 0) gaussian               #  Component type\n')
            new_g.write(' 1) ' + str(ys[i]) + ' '  + str(xs[i]) + ' 2 2  #  Position x, y\n')  # x,y reversed in python
            new_g.write(' 3) ' + str(mags[i]) + ' 1 # Integrated magnitude\n')
            new_g.write(' 4) ' + str(fwhms[i]) + ' 1 # FWHM [pix]\n')
            new_g.write(' 5) 0.0000      0          #     ----- \n')
            new_g.write(' 6) 0.0000      0          #     ----- \n')
            new_g.write(' 7) 0.0000      0          #     ----- \n')
            new_g.write(' 8) 0.0000      0          #     ----- \n')
            new_g.write(' 9) ' + str(qs[i]) + ' 1 # Axis ratio (b/a)\n')
            new_g.write('10) ' + str(pas[i]) + '     2          #  Position angle (PA) [deg: Up=0, Left=90]\n')
            new_g.write(' Z) 0                      #  Skip this model in output image?  (yes=1, no=0)\n')

    return mags, fwhms, qs, pas


def write_constraintfile(output=None, mag_bright=None, mag_dim=None, re_lo=None, re_hi=None, q_lo=None, num=10):
    with open(output, 'w+') as newfile:
        newfile.write('# Component/    parameter   constraint	Comment\n')
        if mag_bright is not None:
            newfile.write('1     mag    ' + str(mag_bright) + ' to '+ str(mag_dim) + '\n')
        if re_lo is not None:
            newfile.write('1     re     ' + str(re_lo) + ' to '+ str(re_hi) + '\n')
        nums = np.arange(num) + 1  # +1 because python starts indexing at 0, galfit starts at 1
        string = ''
        for n in nums:
            string += str(n) + '_'
        string = string[:-1]
        newfile.write(string + '     x     offset\n')
        newfile.write(string + '     y     offset\n')
        newfile.write(string + '     pa    offset\n')
        if q_lo is not None:
            for n in nums:
                if n < 10:
                    newfile.write(str(n) + '     q    ' + str(q_lo) + ' to 1.0\n')
                else:  # if n>=10:
                    newfile.write(str(n) + '    q    ' + str(q_lo) + ' to 1.0\n')


def galfit_to_wiki(galfit_out, table_file):
    '''

    :param galfit_out: e.g. galfit.63
    :param table_file: output table file to copy over to the wiki
    '''

    with open(table_file, 'w+') as tf:
        tf.write('| Integrated mag | FWHM [pix] | qObs |\n| ----- |-----| ----- |\n')
        with open(galfit_out, 'r') as go:
            for line in go:
                entry = None
                if line.startswith(' 3)') or line.startswith(' 4)'):
                    cols = line.split()
                    entry = '| ' + cols[1] + ' '
                elif line.startswith(' 9)'):
                    cols = line.split()
                    entry = '| ' + cols[1] + '\n'
                if entry is not None:
                    tf.write(entry)

def galfit_to_cap(galfit_out, zeropoint, write_new=None):
    """
    DON'T USE THIS!

    :param galfit_out:
    :param zeropoint:
    :param xctr:
    :param yctr:
    :param pa:
    :param img:
    :param mask:
    :param write_new:
    :return:
    """

    mags = []
    fwhms = []
    qs = []
    with open(galfit_out, 'r') as go:
        for line in go:
            cols = line.split()
            if line.startswith(' 3)'):
                mags.append(cols[1])
            elif line.startswith(' 4)'):
                fwhms.append(cols[1])
            elif line.startswith(' 9)'):
                qs.append(cols[1])

    counts = []
    sigmas = []
    for i in range(len(mags)):
        counts.append(10 ** (0.4 * (zeropoint - mags[i])))
        sigmas.append(fwhms[i] / 2.355)  # https://en.wikipedia.org/wiki/Full_width_at_half_maximum

    if write_new is not None:
        with open(write_new, 'w+') as newfile:
            newfile.write('# Counts Sigma_pix qObs\n')  # PAs
            for i in range(len(mags)):
                newfile.write(str(counts[i]) + ' ' + str(sigmas[i]) + ' ' + str(qs[i]) + '\n')  # ' ' + str(pas[i]) +

    return mags, fwhms, qs


if __name__ == "__main__":
    base = '/Users/jonathancohn/Documents/mge/'
    gf = '/Users/jonathancohn/Documents/dyn_mod/galfit_u2698/'

    # galfit_to_wiki(gf + 'galfit.78', gf + 'wiki_Ben_ahcorr_mge_out.txt')
    # galfit_to_wiki(gf + 'galfit.88', gf + 'wiki_galfit_out_u2698_ahcorr_linear_pf001_counts.txt')
    # print(oop)

    ### AFTER CORRECTING STUFF ###
    zp = 24.6949  # 24.697  # 25.95  # 24.697
    texp = 898.467164
    sky_e = 339.493665331
    rms_e = 21.5123034564
    sky_eps = 0.377858732438
    rms_eps = 0.0239433389818
    # MGEs
    mge_rhe = gf + 'mge_ugc_2698_rhe_linear_pf001.txt'
    mge_ahe = gf + 'mge_ugc_2698_ahe_linear_pf001.txt'
    mge_rre = gf + 'mge_ugc_2698_rre_linear_pf001.txt'
    # N components; q limits
    n_ahe = 10
    n_rhe = 11
    n_rre = 11
    qs = None
    # New converted MGE text files (to write)
    conv_ahe = gf + 'conv_mge_ugc_2698_ahe_linear_pf001.txt'
    conv_rhe = gf + 'conv_mge_ugc_2698_rhe_linear_pf001.txt'
    conv_rre = gf + 'conv_mge_ugc_2698_rre_linear_pf001.txt'
    # IMGs
    re = base + 'ugc2698_f160w_pxfr075_pxs010_drz_rapidnuc_sci_nonan_e.fits'  # regH image!
    ahe = base + 'ugc2698_f160w_pxfr075_pxs010_ahcorr_rapidnuc_sci_nonan_e.fits'  # ahcorr image!
    # Masks
    comb_mask = gf + 'f160w_combinedmask_px010.fits'  # include dust mask!
    reg_mask = gf + 'f160w_maskedgemask_px010.fits'  # don't include dust mask!
    # New GALFIT param files!
    pars_gal_ahe = gf + 'galfit_params_u2698_ahe_linear_pf001_zp24.txt'
    pars_gal_rhe = gf + 'galfit_params_u2698_rhe_linear_pf001_zp24.txt'
    pars_gal_rre = gf + 'galfit_params_u2698_rre_linear_pf001_zp24.txt'
    # New GALFIT output names (GALFIT output image blocks)
    out_gal_ahe = gf + 'galfit_out_u2698_ahe_linear_pf001_zp24.txt'
    out_gal_rhe = gf + 'galfit_out_u2698_rhe_linear_pf001_zp24.txt'
    out_gal_rre = gf + 'galfit_out_u2698_rre_linear_pf001_zp24.txt'
    # Constraint files
    cons_8 = gf + 'cons_ugc_2698_n8.txt'
    cons_9 = gf + 'cons_ugc_2698_n9.txt'
    cons_10 = gf + 'cons_ugc_2698_n10.txt'  # ahe
    cons_11 = gf + 'cons_ugc_2698_n11.txt'  # rhe, rre
    cons_12 = gf + 'cons_ugc_2698_n12.txt'
    # Copy file (galfit file of which I'm copying the structure)
    copyf = gf + 'galfit_params_mge_055_zp25.txt'

    # ahe, rhe, rre
    cv = mge_to_galfit(mge_ahe, zp, img=ahe, mask=reg_mask, constraint=cons_10, copy_file=copyf, galfit_out=out_gal_ahe,
                       write_new=conv_ahe, new_galfit=pars_gal_ahe, sky=sky_e, rms=rms_e, texp=texp)
    cv = mge_to_galfit(mge_rhe, zp, img=re, mask=comb_mask, constraint=cons_11, copy_file=copyf, galfit_out=out_gal_rhe,
                       write_new=conv_rhe, new_galfit=pars_gal_rhe, sky=sky_e, rms=rms_e, texp=texp)
    cv = mge_to_galfit(mge_rre, zp, img=re, mask=reg_mask, constraint=cons_11, copy_file=copyf, galfit_out=out_gal_rre,
                       write_new=conv_rre, new_galfit=pars_gal_rre, sky=sky_e, rms=rms_e, texp=texp)

    '''  #
    # OLD STUFF BELOW
    # write constraint files
    write_constraintfile(output=cons_8, q_lo=qs, num=8)  # q_los[c]
    write_constraintfile(output=cons_9, q_lo=qs, num=9)  # q_los[c]
    write_constraintfile(output=cons_10, q_lo=qs, num=10)  # q_los[c]
    write_constraintfile(output=cons_11, q_lo=qs, num=11)  # q_los[c]
    write_constraintfile(output=cons_12, q_lo=qs, num=12)  # q_los[c]

    print(oop)
    # REGREG
    write_constraintfile(output=cons_regreg_cps, q_lo=qs, num=n_regreg)  # q_los[c]
    cv = mge_to_galfit(mge_regreg_cps, zp, img=regH_cps, mask=reg_mask, constraint=cons_regreg_cps,
                       copy_file=copyf, galfit_out=galfit_out_regreg_cps, write_new=mge_conv_regreg_cps,
                       new_galfit=galfit_par_regreg_cps, sky=sky_cps, rms=rms_cps, texp=None)

    write_constraintfile(output=cons_regreg_counts, q_lo=qs, num=n_regreg)  # q_los[c]
    cv = mge_to_galfit(mge_regreg_counts, zp, img=regH_cps, mask=reg_mask, constraint=cons_regreg_counts,
                       copy_file=copyf, galfit_out=galfit_out_regreg_counts, write_new=mge_conv_regreg_counts,
                       new_galfit=galfit_par_regreg_counts, sky=sky_counts, rms=rms_counts, texp=texp)
    print(oop)
    # REGH
    write_constraintfile(output=cons_regH_cps, q_lo=qs, num=n_regH_cps)  # q_los[c]
    cv = mge_to_galfit(mge_regH_cps, zp, img=regH_cps, mask=comb_mask, constraint=cons_regH_cps,
                       copy_file=copyf, galfit_out=galfit_out_regH_cps, write_new=mge_conv_regH_cps,
                       new_galfit=galfit_par_regH_cps, sky=sky_cps, rms=rms_cps, texp=None)
    write_constraintfile(output=cons_regH_counts, q_lo=qs, num=n_regH_counts)  # q_los[c]
    cv = mge_to_galfit(mge_regH_counts, zp, img=regH_counts, mask=comb_mask, constraint=cons_regH_counts,
                       copy_file=copyf, galfit_out=galfit_out_regH_counts, write_new=mge_conv_regH_counts,
                       new_galfit=galfit_par_regH_counts, sky=sky_counts, rms=rms_counts, texp=texp)
    print(oop)
    # AHCORR
    write_constraintfile(output=cons_counts, q_lo=qs, num=n_counts)  # q_los[c]
    cv = mge_to_galfit(mge_counts, zp, img=ahcorr_counts, mask=reg_mask, constraint=cons_counts, copy_file=copyf,
                       galfit_out=galfit_out_counts, write_new=mge_conv_counts, new_galfit=galfit_par_counts)
    print(oop)
    # '''  #

    '''  #
    # galfit_to_wiki(gf+'galfit.73', gf+'wiki_regH_combmask_n10_q06.txt')
    # print(oop)

    mge_in = gf+'ugc_2698_regH_combmask_n10_mge_06.txt'
    con_file = gf+'galfit_regH_combmask_mge_constraintfile_n10_q06.txt'
    write_constraintfile(output=con_file, mag_bright=None, mag_dim=None, re_lo=None, re_hi=None, q_lo=q_lo, num=9)
    cv = mge_to_galfit(mge_in, zp, img=gf+'ugc2698_f160w_pxfr075_pxs010_drz_rapidnuc_sci_no0.fits',
                       mask=gf+'f160w_combinedmask_px010.fits',
                       constraint=con_file,  # constraint file to use in galfit
                       copy_file=gf + 'galfit_params_mge_055_zp25.txt',  # galfit file I'm copying the structure of
                       galfit_out=gf + 'galfit_regH_combmask_mge_n10_q06_zp25.fits',  # output in galfit file
                       write_new=base + 'mge_fit_mine_params_regH_combinedmask_n10_q06_zp25.txt',  # converted mge
                       new_galfit=gf + 'galfit_params_regH_combinedmask_mge_n10_q06_zp25.txt')  # galfit run file
    print(oop)
    # '''  #
