import numpy as np
import argparse


def mge_to_galfit(mge_file, zeropoint, img=None, mask=None, copy_file=None, galfit_out=None, write_new=None,
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
    #print(oop)

    # '''  #
    # Akin-like, regular H-band, regular (NO dust-correction) mask, linear
    mge_in = gf+'ugc_2698_akin_linear_mge_06.txt'  # 'ugc_2698_regH_combmask_n10_q055_mge.txt'
    # N=11=input, N=10=output!
    cv = mge_to_galfit(mge_in, zp, img=gf+'ugc2698_f160w_pxfr075_pxs010_drz_rapidnuc_sci_no0.fits',
                       mask=gf+'f160w_maskedgemask_px010.fits',
                       constraint=gf+'galfit_akin_mge_constraintfile_n10_06.txt',  # constraint file to use in galfit
                       copy_file=gf + 'galfit_params_mge_055_zp25.txt',  # galfit file I'm copying the structure of
                       galfit_out=gf + 'galfit_akin_mge_06_zp25.fits',  # output in galfit file
                       write_new=base + 'mge_fit_mine_params_akin_06_zp25.txt',  # converted mge
                       new_galfit=gf + 'galfit_params_akin_mge_06_zp25.txt')  # galfit run file
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

    # '''
    # dust-corrected H-band, no dust mask, N=11, qbounds=[0.4, 1.]
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

    # '''
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

    # '''
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
