import numpy as np
import argparse


def paper_to_galfit(table_file, pix_scale=0.1, write_new=None, zp=25.95):
    # NOTE: Akin's image is 0.06"/pix; exposure time = 1354.46 seconds
    # Use eqns 2 and 3 from readme_mge_fit_sectors.pdf to convert Akin's table to peak counts C_0, then eqn 1 for C_0->
    # total counts, then total counts -> integrated mag as usual.
    # eqn3: I_H [Lsol/pc^2] = (64800/pi)^2 * 10^(0.4*(Msol_H - mu_H))
    ## Solve for mu_H! mu_H = Msol_H - log10((np.pi / 64800)**2 * float(cols[1])
    # eqn2: mu_H = zeropoint + 0.1 + 5log10(pix_scale) + 2.5log10(exp_time) - 2.5log(C_0) - A_H
    ## pix_scale is defined s.t. sigma_pix * pix_scale = sigma_arcsec
    # = 25.95 + 0.1 + 5log10(0.06) + 2.5log10(1354.46) - 2.5log10(C_0) - 0.075
    ## Solve for C_0: I - 25.95 - 0.1 + 0.075 = 5log10(0.06) + 2.5(log10(1354.46) - log10(C_0))
    ### 10^((I + consts)/2.5) = pix_scale^2  * (exp_time / C_0)
    ### C_0 = pix_scale**2 * exp_time * 10**(-0.4 * (I+consts))
    # eqn1: C_0 = total_counts / (2*pi*sigma_pix**2 * qObs)
    ## Solve for total_counts: total_counts = C_0 * (2 * np.pi * sigma_pix**2 * qObs)

    # ALL BELOW IS WRONG
    # To make sure I'm comparing them properly (since the listed units are slightly different), I need to convert
    # between Akin's table (Lsol/pc^2) and the galfit output (integrated mags):*Ben's (old, not-dust-corrected) MGE file
    # assumed: M_H,sol=3.37 mag (adopted from Willmer 2018), and a spatial scale of 414 arcsec/pc. Therefore,
    # M_H = -2.5 log10(L/Lsol) + M_H,sol = -2.5 log10(L/Lsol)+3.37. However, Akin's table has L/pc^2 listed, not just
    # L. To get the total integrated L from each component, I need to integrate over the 2D gaussian. The integral of a
    # 1D gaussian = sqrt(2*pi)*sigma, so the integral of a 2D gaussian = 2*pi*sigma^2*q. Therefore,
    # L_tot,H = 2*pi*sigma^2*q*(L_H/pc^2), where sigma is in units of pc. For sigma in units of arcsec,
    # sigma_pc = sigma_arcsec * 414 (pc/arcsec). Therefore, M_H = -2.5 log10(2 pi sigma^2 q (L/pc^2)) + 3.37.
    # BUT Akin's table.pdf caption (TABLE 15) lists M_Hsol = 3.32
    # ALSO Akin's caption makes an assumption of a dust correction of 0.075 mags. Therefore, zp = 3.32 + 0.075 = 3.395

    mags = []
    sigmas = []
    qs = []
    with open(table_file, 'r') as tf:
        for line in tf:
            if not line.startswith('#'):
                cols = line.split()

                mags.append(-2.5 * np.log10(2*np.pi*(414.*float(cols[2]))**2 * float(cols[3]) * float(cols[1])) + 3.395)
                # -2.5 log10(2 pi sigma^2 q (L/pc^2)) + 3.395  # 3.32
                sigmas.append(float(cols[2]))
                qs.append(float(cols[3]))

    fwhms = 2.355 * np.array(sigmas) / pix_scale
    if write_new is not None:
        with open(write_new, 'w+') as newfile:
            newfile.write('# Mags FWHM_pixels qObs\n')  # PAs
            for i in range(len(mags)):
                newfile.write(str(mags[i]) + ' ' + str(fwhms[i]) + ' ' + str(qs[i]) + '\n')


def app_to_ab(table_file, write_new=None):
    mags = []
    fwhms = []
    qs = []
    with open(table_file, 'r') as tf:
        for line in tf:
            if not line.startswith('#'):
                cols = line.split()
                mags.append(float(cols[0]) + 5*np.log10(89.*1e6) - 5.)
                fwhms.append(cols[1])
                qs.append(cols[2])

    if write_new is not None:
        with open(write_new, 'w+') as newfile:
            newfile.write('# Mags FWHM_pixels qObs\n')  # PAs
            for i in range(len(mags)):
                newfile.write(str(mags[i]) + ' ' + str(fwhms[i]) + ' ' + str(qs[i]) + '\n')


def yildirim_to_galfit(yildirim, xctr=880.7209, yctr=491.2107, pa=-7.5697, img=None, mask=None, copy_file=None,
                       galfit_out=None, new_galfit=None):
    mags = []
    fwhms = []
    qs = []
    with open(yildirim, 'r') as f:
        for line in f:
            cols = line.split()
            if not line.startswith('#'):
                mags.append(float(cols[0]))
                fwhms.append(float(cols[1]))
                qs.append(float(cols[2]))

    print(len(mags))
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
                        wline = 'P) 1                   # Choose: 0=optimize, 1=model, 2=imgblock, 3=subcomps'
                    elif line.startswith('H)') or line.startswith('I)') or line.startswith('J)') or \
                            line.startswith('K)') or line.startswith('O)'):
                        wline = line
                        # line += '\n'
                    new_g.write(wline)
                else:
                    break

        for i in range(len(mags)):
            new_g.write('\n')
            new_g.write('# Component number: ' + str(i+1) + '\n')
            new_g.write(' 0) gaussian               #  Component type\n')
            new_g.write(' 1) ' + str(xctr) + ' '  + str(yctr) + ' 2 2  #  Position x, y\n')
            new_g.write(' 3) ' + str(mags[i]) + ' 0 # Integrated magnitude\n')
            new_g.write(' 4) ' + str(fwhms[i]) + ' 0 # FWHM [pix]\n')
            new_g.write(' 5) 0.0000      0          #     ----- \n')
            new_g.write(' 6) 0.0000      0          #     ----- \n')
            new_g.write(' 7) 0.0000      0          #     ----- \n')
            new_g.write(' 8) 0.0000      0          #     ----- \n')
            new_g.write(' 9) ' + str(qs[i]) + ' 0 # Axis ratio (b/a)\n')
            new_g.write('10) ' + str(pa) + '     2          #  Position angle (PA) [deg: Up=0, Left=90]\n')
            new_g.write(' Z) 0                      #  Skip this model in output image?  (yes=1, no=0)\n')

    return mags, fwhms, qs


def mge_to_galfit(mge_file, zeropoint, xctr=880.7209, yctr=491.2107, pa=-7.5697, img=None, mask=None, copy_file=None,
                  galfit_out=None, write_new=None, new_galfit=None, constraint=None):

    mags = []
    fwhms = []
    qs = []
    pas = []

    with open(mge_file, 'r') as f:
        for line in f:
            cols = line.split()
            if not line.startswith('#'):
                mags.append(zeropoint - 2.5 * np.log10(float(cols[0])))
                fwhms.append(2 * np.sqrt(2 * np.log(2.)) * float(cols[1]))
                # https://en.wikipedia.org/wiki/Full_width_at_half_maximum
                qs.append(float(cols[2]))
                if len(cols) > 3:
                    pas.append(float(cols[3]))

    if write_new is not None:
        with open(write_new, 'w+') as newfile:
            newfile.write('# Mags FWHM_pixels qObs\n')  # PAs
            for i in range(len(mags)):
                newfile.write(str(mags[i]) + ' ' + str(fwhms[i]) + ' ' + str(qs[i]) + '\n')  # ' ' + str(pas[i]) +

    print(len(mags))
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
                    elif constraint is not None and line.startswith('G)'):
                        wline = 'G) '  + constraint + '      # File with parameter constraints (ASCII file)' + '\n'
                    elif line.startswith('H)') or line.startswith('I)') or line.startswith('J)') or \
                            line.startswith('K)') or line.startswith('O)') or line.startswith('P)'):
                        wline = line
                        # line += '\n'
                    new_g.write(wline)
                else:
                    break

        for i in range(len(mags)):
            new_g.write('\n')
            new_g.write('# Component number: ' + str(i+1) + '\n')
            new_g.write(' 0) gaussian               #  Component type\n')
            new_g.write(' 1) ' + str(xctr) + ' '  + str(yctr) + ' 2 2  #  Position x, y\n')
            new_g.write(' 3) ' + str(mags[i]) + ' 1 # Integrated magnitude\n')
            new_g.write(' 4) ' + str(fwhms[i]) + ' 1 # FWHM [pix]\n')
            new_g.write(' 5) 0.0000      0          #     ----- \n')
            new_g.write(' 6) 0.0000      0          #     ----- \n')
            new_g.write(' 7) 0.0000      0          #     ----- \n')
            new_g.write(' 8) 0.0000      0          #     ----- \n')
            new_g.write(' 9) ' + str(qs[i]) + ' 1 # Axis ratio (b/a)\n')
            new_g.write('10) ' + str(pa) + '     2          #  Position angle (PA) [deg: Up=0, Left=90]\n')
            new_g.write(' Z) 0                      #  Skip this model in output image?  (yes=1, no=0)\n')

    return mags, fwhms, qs, pas


def mge_to_galfit_orig(mge_file, zeropoint, img=None, mask=None, copy_file=None, galfit_out=None, write_new=None,
                       new_galfit=None, constraint=None):

    mags = []
    fwhms = []
    qs = []
    pas = []

    with open(mge_file, 'r') as f:
        for line in f:
            cols = line.split()
            if not line.startswith('#'):
                mags.append(zeropoint - 2.5 * np.log10(float(cols[0])))
                fwhms.append(2 * np.sqrt(2 * np.log(2.)) * float(cols[1]))
                # https://en.wikipedia.org/wiki/Full_width_at_half_maximum
                qs.append(float(cols[2]))
                if len(cols) > 3:
                    pas.append(float(cols[3]))

    if write_new is not None:
        with open(write_new, 'w+') as newfile:
            newfile.write('# Mags FWHM_pixels qObs\n')  # PAs
            for i in range(len(mags)):
                newfile.write(str(mags[i]) + ' ' + str(fwhms[i]) + ' ' + str(qs[i]) + '\n')  # ' ' + str(pas[i]) +

    print(len(mags))
    nums = [3, 4, 9]
    with open(new_galfit, 'w+') as new_g:
        with open(copy_file, 'r') as cf:
            comp = -1  # idx starts at 0
            for line in cf:
                if comp < len(mags):
                    if line.startswith('# Component number: '):
                        comp += 1  # idx starts at 0
                    for n in range(len(nums)):
                        if img is not None and line.startswith('A)'):
                            line = 'A) '  + img + '      # Input data image (FITS file)' + '\n'
                        elif line.startswith('B)'):
                            line = 'B) '  + galfit_out + '      # Output data image block' + '\n'
                        elif mask is not None and line.startswith('F)'):
                            line = 'F) '  + mask + '      # Bad pixel mask (FITS image or ASCII coord list)' + '\n'
                        elif constraint is not None and line.startswith('G)'):
                            line = 'G) '  + constraint + '      # File with parameter constraints (ASCII file)' + '\n'
                        elif line.startswith(' ' + str(nums[n]) + ')'):
                            cols = line.split()
                            if nums[n] == 3:
                                replace = mags[comp]
                            elif nums[n] == 4:
                                replace = fwhms[comp]
                            elif nums[n] == 9:
                                replace = qs[comp]
                            cols[1] = replace
                            line = ''
                            for col in cols:
                                line += ' ' + str(col)
                            line += '\n'
                    new_g.write(line)

        if len(mags) >= 10:
            for i in range(10, len(mags)):
                new_g.write('\n')
                new_g.write('# Component number: ' + str(i+1) + '\n')
                new_g.write(' 0) gaussian               #  Component type\n')
                new_g.write(' 1) 880.7209 491.2107 2 2  #  Position x, y\n')
                new_g.write(' 3) ' + str(mags[i]) +  ' 1 # Integrated magnitude\n')
                new_g.write(' 4) ' + str(fwhms[i]) + ' 1 # FWHM [pix]\n')
                new_g.write(' 5) 0.0000      0          #     ----- \n')
                new_g.write(' 6) 0.0000      0          #     ----- \n')
                new_g.write(' 7) 0.0000      0          #     ----- \n')
                new_g.write(' 8) 0.0000      0          #     ----- \n')
                new_g.write(' 9) ' + str(qs[i]) + ' 1 # Axis ratio (b/a)\n')
                new_g.write('10) -7.5697     2          #  Position angle (PA) [deg: Up=0, Left=90]\n')
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


if __name__ == "__main__":
    zp = 25.95 # 24.697  # 25.95  # 24.697
    base = '/Users/jonathancohn/Documents/mge/'
    gf = '/Users/jonathancohn/Documents/dyn_mod/galfit_u2698/'

    #galfit_to_wiki(gf + 'galfit_params_ahcorr_mge_n11_04_zp25.txt', gf + 'wiki_ahcorr_n11_04_mge.txt')  # wiki_mge055_zp25_take4.txt
    #galfit_to_wiki(gf + 'galfit.parameters.txt', gf + 'wiki_Ben_ahcorr_mge.txt')  # wiki_mge055_zp25_take4.txt
    #galfit_to_wiki(gf + 'galfit.72', gf + 'wiki_mge_ahcorr_n10_055_zp25.txt')  # wiki_mge055_zp25_take4.txt
    # galfit_to_wiki(gf + 'galfit.78', gf + 'wiki_Ben_ahcorr_mge_out.txt')
    # print(oop)

    ###

    # '''  #
    # WHAT I SHOULD HAVE BEEN DOING WITH AKIN'S STUFF
    paper_to_galfit(base+'yildirim_table_2698.txt', write_new=gf+'yildirim_galfit_zp25.txt')
    app_to_ab(gf+'yildirim_galfit_zp25.txt', write_new=gf+'yildirim_appmag_table_2698_zp25.txt')

    cv = yildirim_to_galfit(gf+'yildirim_appmag_table_2698_zp25.txt',
                            img=gf+'ugc2698_f160w_pxfr075_pxs010_drz_rapidnuc_sci_no0.fits',
                            mask=gf+'f160w_maskedgemask_px010.fits',
                            copy_file=gf + 'galfit_params_mge_055_zp25.txt',  # galfit file I'm copying the structure of
                            galfit_out=gf + 'galfit_akin_mge_fixedmodel_out_zp25.fits',  # output in galfit file
                            new_galfit=gf + 'galfit_params_akin_mge_fixedmodel_zp25.txt')  # galfit run file
    print(oop)

    '''  #
    # Akin-like, regular H-band, regular (NO dust-correction) mask, using n=43 (but got out only n=12 anyway)
    mge_in = gf+'ugc_2698_akin_n43_mge_06.txt'  # 'ugc_2698_regH_combmask_n10_q055_mge.txt'
    # mge_in = gf+'ugc_2698_akin_linear_mge_06.txt'  # 'ugc_2698_regH_combmask_n10_q055_mge.txt'
    # N=11=input, N=10=output!
    con_file = gf+'galfit_akin_mge_constraintfile_n43_q06_conq2.txt'
    write_constraintfile(output=con_file, mag_bright=None, mag_dim=None, re_lo=None, re_hi=None, q_lo=0.60, num=11)
    print(oop)
    cv = mge_to_galfit(mge_in, zp, img=gf+'ugc2698_f160w_pxfr075_pxs010_drz_rapidnuc_sci_no0.fits',
                       mask=gf+'f160w_maskedgemask_px010.fits',
                       constraint=con_file,  # constraint file to use in galfit
                       copy_file=gf + 'galfit_params_mge_055_zp25.txt',  # galfit file I'm copying the structure of
                       galfit_out=gf + 'galfit_akin_mge_n43_q06_zp25.fits',  # output in galfit file
                       write_new=base + 'mge_fit_mine_params_akin_n43_q06_zp25.txt',  # converted mge
                       new_galfit=gf + 'galfit_params_akin_mge_n43_q06_zp25.txt')  # galfit run file
    write_constraintfile(output=con_file, mag_bright=None, mag_dim=None, re_lo=None, re_hi=None, q_lo=0.60, num=12)
    print(oop)
    # '''  #

    '''  #
    # Akin-like, regular H-band, regular (NO dust-correction) mask, linear
    mge_in = gf+'ugc_2698_akin_linear_mge_06.txt'  # 'ugc_2698_regH_combmask_n10_q055_mge.txt'
    # mge_in = gf+'ugc_2698_akin_linear_mge_06.txt'  # 'ugc_2698_regH_combmask_n10_q055_mge.txt'
    # N=11=input, N=10=output!
    cv = mge_to_galfit(mge_in, zp, img=gf+'ugc2698_f160w_pxfr075_pxs010_drz_rapidnuc_sci_no0.fits',
                       mask=gf+'f160w_maskedgemask_px010.fits',
                       constraint=gf+'galfit_akin_mge_constraintfile_n10_06.txt',  # constraint file to use in galfit
                       copy_file=gf + 'galfit_params_mge_055_zp25.txt',  # galfit file I'm copying the structure of
                       galfit_out=gf + 'galfit_akin_mge_06_zp25.fits',  # output in galfit file
                       write_new=base + 'mge_fit_mine_params_akin_06_zp25.txt',  # converted mge
                       new_galfit=gf + 'galfit_params_akin_mge_06_zp25_new.txt')  # galfit run file
    print(oop)
    # '''  #

    ###

    ''' #
    # regular H-band, dust corrected mask, N=10 (output n=9), q>=0.4
    galfit_to_wiki(gf+'galfit.75', gf+'wiki_regH_combmask_mge_n10_n8_q04.txt')
    print(oop)
    mge_in = gf + 'ugc_2698_regH_combmask_n10_mge_04.txt'  # MGE output by mge_fit_mine.py, to convert to GALFIT's input
    con_file = gf + 'galfit_regH_combmask_mge_constraintfile_n10_q04.txt'  # new constraint file to create
    write_constraintfile(output=con_file, mag_bright=None, mag_dim=None, re_lo=None, re_hi=None, q_lo=0.40, num=9)
    cv = mge_to_galfit(mge_in, zp, img=gf + 'ugc2698_f160w_pxfr075_pxs010_drz_rapidnuc_sci_no0.fits',
                       mask=gf + 'f160w_combinedmask_px010.fits',
                       constraint=con_file,  # constraint file to use in galfit
                       copy_file=gf + 'galfit_params_mge_055_zp25.txt',  # galfit file I'm copying the structure of
                       galfit_out=gf + 'galfit_regH_combmask_mge_n10_q04_zp25.fits',  # output in galfit file
                       write_new=base + 'mge_fit_mine_params_regH_combinedmask_n10_q04_zp25.txt',  # converted mge
                       new_galfit=gf + 'galfit_params_regH_combinedmask_mge_n10_q04_zp25.txt')  # galfit run file
    print(oop)
    # ''' #

    '''  #
    # regular H-band, dust corrected mask, N=10 (output n=9), q>=0.6
    galfit_to_wiki(gf+'galfit.73', gf+'wiki_regH_combmask_n10_q06.txt')
    print(oop)

    mge_in = gf+'ugc_2698_regH_combmask_n10_mge_06.txt'
    con_file = gf+'galfit_regH_combmask_mge_constraintfile_n10_q06.txt'
    write_constraintfile(output=con_file, mag_bright=None, mag_dim=None, re_lo=None, re_hi=None, q_lo=0.60, num=9)
    cv = mge_to_galfit(mge_in, zp, img=gf+'ugc2698_f160w_pxfr075_pxs010_drz_rapidnuc_sci_no0.fits',
                       mask=gf+'f160w_combinedmask_px010.fits',
                       constraint=con_file,  # constraint file to use in galfit
                       copy_file=gf + 'galfit_params_mge_055_zp25.txt',  # galfit file I'm copying the structure of
                       galfit_out=gf + 'galfit_regH_combmask_mge_n10_q06_zp25.fits',  # output in galfit file
                       write_new=base + 'mge_fit_mine_params_regH_combinedmask_n10_q06_zp25.txt',  # converted mge
                       new_galfit=gf + 'galfit_params_regH_combinedmask_mge_n10_q06_zp25.txt')  # galfit run file
    print(oop)
    # '''  #

    '''  #
    # regular H-band, dust corrected mask, N=10
    mge_in = gf+'ugc_2698_regH_combmask_n11_q055_mge.txt'  # 'ugc_2698_regH_combmask_n10_q055_mge.txt'
    # N=11=input, N=10=output!
    cv = mge_to_galfit(mge_in, zp, img=gf+'ugc2698_f160w_pxfr075_pxs010_drz_rapidnuc_sci_no0.fits',
                       mask=gf+'f160w_combinedmask_px010.fits',
                       constraint=gf+'galfit_f160w_mge_constraintfile_n10_055.txt',  # constraint file to use in galfit
                       copy_file=gf + 'galfit_params_mge_055_zp25.txt',  # galfit file I'm copying the structure of
                       galfit_out=gf + 'galfit_regH_combmask_mge_n10_055_zp25.fits',  # output in galfit file
                       write_new=base + 'mge_fit_mine_params_regH_combmask_n10_055_zp25.txt',  # converted mge
                       new_galfit=gf + 'galfit_params_regH_combmask_mge_n10_055_zp25.txt')  # galfit run file
    print(oop)
    # '''  #

    '''  #
    mge_in = gf + 'ugc_2698_regH_combmask_mge.txt[
    # FOR regular H-band, with dust-corrected mask
    converted_vals = mge_to_galfit(mge_in, zp, img=gf+'ugc2698_f160w_pxfr075_pxs010_drz_rapidnuc_sci_no0.fits',
                                   mask=gf+'f160w_combinedmask_px010.fits',
                                   copy_file=gf + 'galfit_params_mge_055_zp25.txt',
                                   galfit_out=gf + 'galfit_regH_combmask_mge_055_zp25.fits',  # output in galfit file
                                   write_new=base + 'mge_fit_mine_params_regH_combmask_055_zp25.txt',  # converted mge
                                   new_galfit=gf + 'galfit_params_regH_combmask_mge_055_zp25.txt')  # galfit run file
    # '''  #

    ###

    '''  #
    # dust-corrected H-band, no dust mask, N=11, qbounds=[0.4, 1.]
    galfit_to_wiki(gf+'galfit.77', gf+'wiki_ahcorr_mge_n11_q04.txt')
    print(oop)

    # N=11=input, N=10=output!
    mge_in = gf+'ugc_2698_ahcorr_n11_mge_04.txt'
    cv = mge_to_galfit(mge_in, zp, img=gf+'ugc2698_f160w_pxfr075_pxs010_ahcorr_rapidnuc_sci_nonan.fits',
                       mask=gf+'f160w_maskedgemask_px010.fits',
                       constraint=gf+'galfit_f160w_mge_constraintfile_n11_04.txt',  # constraint file to use in galfit
                       copy_file=gf + 'galfit_params_mge_055_zp25.txt',  # galfit file I'm copying the structure of
                       galfit_out=gf + 'galfit_ahcorr_mge_n11_04_zp25.fits',  # output in galfit file
                       write_new=base + 'mge_fit_mine_params_ahcorr_n11_04_zp25.txt',  # converted mge
                       new_galfit=gf + 'galfit_params_ahcorr_mge_n11_04_zp25.txt')  # galfit run file
    print(oop)
    # '''  #

    '''  #
    # dust-corrected H-band, no dust mask, N=10, qbounds=[0.4, 1.]
    # N=10=input, N=9=output!
    mge_in = gf+'ugc_2698_ahcorr_n10_mge_04.txt'
    cv = mge_to_galfit(mge_in, zp, img=gf+'ugc2698_f160w_pxfr075_pxs010_ahcorr_rapidnuc_sci_nonan.fits',
                       mask=gf+'f160w_maskedgemask_px010.fits',
                       constraint=gf+'galfit_f160w_mge_constraintfile_n10_04.txt',  # constraint file to use in galfit
                       copy_file=gf + 'galfit_params_mge_055_zp25.txt',  # galfit file I'm copying the structure of
                       galfit_out=gf + 'galfit_ahcorr_mge_n10_04_zp25.fits',  # output in galfit file
                       write_new=base + 'mge_fit_mine_params_ahcorr_n10_04_zp25.txt',  # converted mge
                       new_galfit=gf + 'galfit_params_ahcorr_mge_n10_04_zp25.txt')  # galfit run file
    print(oop)
    # '''  #

    '''  #
    # dust-corrected H-band, no dust mask, N=10
    # N=11=input, N=10=output!
    mge_in = gf+'ugc_2698_ahcorr_n10_mge.txt'
    cv = mge_to_galfit(mge_in, zp, img=gf+'ugc2698_f160w_pxfr075_pxs010_ahcorr_rapidnuc_sci_nonan.fits',
                       mask=gf+'f160w_maskedgemask_px010.fits',
                       constraint=gf+'galfit_f160w_mge_constraintfile_n10_055.txt',  # constraint file to use in galfit
                       copy_file=gf + 'galfit_params_mge_055_zp25.txt',  # galfit file I'm copying the structure of
                       galfit_out=gf + 'galfit_ahcorr_mge_n10_055_zp25.fits',  # output in galfit file
                       write_new=base + 'mge_fit_mine_params_ahcorr_n10_055_zp25.txt',  # converted mge
                       new_galfit=gf + 'galfit_params_ahcorr_mge_n10_055_zp25.txt')  # galfit run file
    print(oop)
    # '''  #

    '''  #
    # FOR dust-corrected H-band, with no dust mask
    # args['file'] = base+'ugc_2698_f160w_ahcorr_mge.txt'
    converted_vals = mge_to_galfit(args['file'], zp, mask=gf+'f160w_maskedgemask_px010.fits',
                                   copy_file=gf + 'galfit_pars2.txt',
                                   galfit_out=gf + 'galfit_mge_055_zp25.fits', write_new=base+'mge_fit_mine_params_055_zp25.txt',
                                   new_galfit=gf + 'galfit_params_mge_055_zp25.txt')
    # '''  #
    print(cv)
