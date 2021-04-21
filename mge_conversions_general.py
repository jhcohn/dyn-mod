import numpy as np
import argparse
from pathlib import Path
from astropy.io import fits


def paper_to_galfit(table_file, mag_sol=3.32, pix_scale=0.06, t_exp=1354.46, zp=24.6949, apcorr=0.0, dust=0.075,
                    xctr=None, yctr=None, pa=-8.14, img=None, mask=None, copy_file=None, es=False, galfit_out=None,
                    new_galfit=None, compare_file=None, logged=False, dt=True):
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
    :param compare_file: make a simple text file with the converted components
    :param logged: logged=True if the surface density is in log units [log10(Lsun/pc^2)]; else False [Lsun/pc^2]
    :param es: make the galfit file in electrons/s instead of electrons
    """

    # Akin's table.pdf caption (TABLE 15) lists M_Hsol = 3.32
    # Akin's caption makes an assumption of a dust correction of 0.075 mags
    # NOTE: Akin's image is 0.06"/pix; exposure time = 1354.46 seconds
    # Use eqns 2 and 3 from readme_mge_fit_sectors.pdf to convert Akin's table to peak counts C_0, then eqn 1 for C_0->
    # total counts, then total counts -> integrated mag as usual.
    # eqn3: I_H [Lsol/pc^2] = (64800/pi)^2 * 10^(0.4*(Msol_H - mu_H))
    ## Solve for mu_H! mu_H = Msol_H - log10((np.pi / 64800)**2 * float(cols[1]) *****
    # eqn2: mu_H = zeropoint + apcorr + 5log10(pix_scale) + 2.5log10(exp_time) - 2.5log(C_0) - A_H
    ## apcorr=0, because we're using the ZP for the correct aperture
    ## pix_scale is defined s.t. sigma_pix * pix_scale = sigma_arcsec
    # = ZP + apcorr + 5log10(0.06) + 2.5log10(1354.46) - 2.5log10(C_0) - 0.075
    ## Solve for C_0: mu_H - ZP - apcorr + 0.075 = 5log10(0.06) + 2.5(log10(1354.46) - log10(C_0))
    ### Let consts = -ZP + 0.075 - apcorr
    ### 10^((mu_H + consts)/2.5) = pix_scale^2  * (exp_time / C_0)
    ### C_0 = pix_scale**2 * exp_time * 10**(-0.4 * (mu_H+consts)) *****
    # eqn1: C_0 = total_counts / (2*pi*sigma_pix**2 * qObs)
    ## NOTE: in our images, we have units of electrons instead of counts, but this is okay/good bc our ZP is in e-/s
    ## Solve for total_counts: total_counts = C_0 * (2 * np.pi * sigma_pix**2 * qObs) *****
    # Finally: mags = zeropoint - 2.5 * log10(total_counts) *****
    # NOTE: Akin's paper takes inc = 67 deg

    mags = []
    fwhms = []
    qs = []
    with open(table_file, 'r') as tf:
        for line in tf:
            if not line.startswith('#'):
                cols = line.split()
                if logged:
                    mu = mag_sol - 2.5 * np.log10((np.pi / 64800) ** 2 * 10**float(cols[1]))  # cols[1] = log10(I_H)
                else:
                    mu = mag_sol - 2.5 * np.log10((np.pi / 64800)**2 * float(cols[1]))  # cols[1] = I_H [Lsol/pc^2]
                c0 = pix_scale**2 * t_exp * 10**(0.4 * (zp + apcorr - dust - mu))
                total_counts = c0 * 2 * np.pi * (float(cols[2]) / pix_scale)**2 * float(cols[3])
                #if es:
                #    total_counts /= t_exp
                if dt:
                    mag = zp - 2.5*np.log10(total_counts / t_exp)
                else:
                    mag = zp - 2.5 * np.log10(total_counts)
                mags.append(mag)  # above: cols[2] = sigma_arcsec, cols[3] = qObs
                fwhms.append(2.355 * float(cols[2]) / pix_scale)  # fwhm = 2.355 * np.array(sigma_arcsec) / pix_scale
                qs.append(float(cols[3]))

    if compare_file is not None:
        with open(compare_file, 'w+') as cf:
            cf.write('# Component Integrated_mags FWHM_pix qObs\n')
            for m in range(len(mags)):
                cf.write(str(m+1) + ' ' + str(mags[m]) + ' ' + str(fwhms[m]) + ' ' + str(qs[m]) + '\n')

    with open(new_galfit, 'w+') as new_g:
        with open(copy_file, 'r') as cf:
            for line in cf:
                wline = line
                if not line.startswith('# Component number: '):
                    if es and line.startswith('A)'):
                        wline = 'A) none      # Input data image (FITS file)' + '\n'
                    elif img is not None and line.startswith('A)'):
                        wline = 'A) '  + img + '      # Input data image (FITS file)' + '\n'
                        # wline = 'A) none      # Input data image (FITS file)' + '\n'
                    elif line.startswith('B)'):
                        wline = 'B) '  + galfit_out + '      # Output data image block' + '\n'
                    elif line.startswith('D)'):
                        wline = 'D) none          # Input PSF image and (optional) diffusion kernel\n'
                    elif es and line.startswith('F)'):
                        wline = 'F) none      # Bad pixel mask (FITS image or ASCII coord list)' + '\n'
                    elif mask is not None and line.startswith('F)'):
                        wline = 'F) '  + mask + '      # Bad pixel mask (FITS image or ASCII coord list)' + '\n'
                    elif line.startswith('G)'):
                        wline = 'G) none      # File with parameter constraints (ASCII file)\n'
                    elif line.startswith('J)'):
                        wline = 'J) ' + str(zp) + '              # Magnitude photometric zeropoint \n'
                    elif line.startswith('K)'):
                        wline = 'K) ' + str(pix_scale) + ' ' + str(pix_scale) + '        # Plate scale (dx dy)' +\
                                '   [arcsec per pixel]\n'
                    elif line.startswith('P)'):
                        wline = 'P) 1                   # Choose: 0=optimize, 1=model, 2=imgblock, 3=subcomps' + '\n\n'
                    elif line.startswith('H)') or line.startswith('I)') or line.startswith('O)'):
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


def mge_to_psfgalfit(mge_file, zeropoint, img=None, mask=None, copy_file=None, galfit_out=None, write_new=None,
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

        # ADD SKY COMPONENT!
        new_g.write('# Component number: 0\n')
        new_g.write(' 0) sky                    #  Component type\n')
        new_g.write(' 1) ' + str(sky) + '    0          #  Sky background at center of fitting region [ADUs]\n')
        new_g.write(' 2) 0.000e+00      0       #  dsky/dx (sky gradient in x)     [ADUs/pix]\n')
        new_g.write(' 3) 0.000e+00      0       #  dsky/dy (sky gradient in y)     [ADUs/pix]\n')
        new_g.write(' Z) 0                      #  Skip this model in output image?  (yes=1, no=0)\n')

        for i in range(len(mags)):
            new_g.write('\n')
            if i == 0:
                # PSF fit.
                new_g.write('# Component number: ' + str(i+1) + '\n')
                new_g.write(' 0) psf                    #  Component type\n')
                new_g.write(' 1) ' + str(ys[i]) + ' '  + str(xs[i]) + ' 2 2  #  Position x, y\n')  # x,y reversed in python
                new_g.write(' 3) ' + str(mags[i]) + ' 1 # Total magnitude\n')
                new_g.write(' Z) 0                      #  Skip this model in output image?  (yes=1, no=0)\n')
            else:
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


def mge_to_galfit(mge_file, zeropoint, img=None, mask=None, copy_file=None, galfit_out=None, write_converted=None,
                  new_galfit=None, constraint=None, sky=None, rms=None, glx='UGC 2698'):
                  # texp=None, gain=None, ax1=1620, ax2=1231, scale=0.100):  # pa=-7.5697

    hdu = fits.open(img)
    hdr = hdu[0].header
    hdu.close()

    ax1 = hdr['NAXIS1']
    ax2 = hdr['NAXIS2']
    texp = hdr['EXPTIME']
    scale = hdr['D005SCAL']
    gain = hdr['CCDGAIN']

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
                    # mags.append(zeropoint - 2.5 * np.log10(float(cols[0]) / texp))  # - 2.5*np.log10(gain)
                    mags.append(zeropoint - 2.5 * np.log10(float(cols[0]) / float(texp)) - 2.5*np.log10(float(gain)))
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

    if write_converted is not None:
        with open(write_converted, 'w+') as newfile:
            newfile.write('# ' + glx + ' MGE output using mge_fit_mine.py, converted to GALFIT input units\n')
            newfile.write('# param file: ' + new_galfit + '\n')
            newfile.write('# pa = the GALFIT position angle: "The position angle is 0 if the semi-major axis is aligned'
                          ' parallel to the Y-axis and increases toward the counter-clockwise direction.\n')
            newfile.write('# Mags FWHM_pixels qObs xc yc pa\n')  # PAs
            for i in range(len(mags)):
                newfile.write(str(mags[i]) + ' ' + str(fwhms[i]) + ' ' + str(qs[i]) + ' ' + str(ys[i]) + ' ' +
                              str(xs[i]) + ' ' + str(pas[i]) + '\n')  # x,y reversed in python

    if constraint is not None and not Path(constraint).exists():
        print(len(mags), constraint)
        write_constraintfile(output=constraint, q_lo=None, num=len(mags), psf=False, sky=False)

    with open(new_galfit, 'w+') as new_g:
        with open(copy_file, 'r') as cf:
            for line in cf:
                wline = line
                if not line.startswith('# Component number: '):
                    if line.startswith('#  Chi'):
                        if sky is None:
                            wline = '# run with: galfit ' + new_galfit + '\n'
                        else:
                            wline = '# run with: galfit -skyped ' + str(sky) + ' -skyrms ' + str(rms) + ' ' +\
                                    new_galfit + '\n'
                    elif line.startswith('A)'):
                        wline = 'A) '  + img + '      # Input data image (FITS file)\n'
                    elif line.startswith('B)'):
                        wline = 'B) '  + galfit_out + '      # Output data image block\n'
                    elif line.startswith('F)'):
                        wline = 'F) '  + mask + '      # Bad pixel mask (FITS image or ASCII coord list)\n'
                    elif line.startswith('G)'):
                        wline = 'G) '  + constraint + '      # File with parameter constraints (ASCII file)\n'
                    elif line.startswith('H)'):
                        wline = 'H) 1    ' + str(ax1) + ' 1    ' + str(ax2) + ' # Image region to fit (xmin xmax ymin '\
                                + 'ymax)\n'
                    elif line.startswith('J)'):
                        wline = 'J) ' + str(zeropoint) + '              # Magnitude photometric zeropoint \n'
                    elif line.startswith('K)'):
                        wline = 'K) ' + str(scale) + '  ' + str(scale) + '        # Plate scale (dx dy)   [arcsec per'\
                                + 'pixel]\n'
                    elif line.startswith('I)') or line.startswith('O)') or line.startswith('P)'):
                        wline = line
                    new_g.write(wline)
                else:  # as soon as we reach the part of the file where we fill in mge components...
                    break  # ...stop copying from this past file! Write in those components separately below

        if sky is not None:
            # ADD SKY COMPONENT!
            new_g.write('# Component number: 0\n')
            new_g.write(' 0) sky                    #  Component type\n')
            new_g.write(' 1) ' + str(sky) + '    0          #  Sky background at center of fitting region [ADUs]\n')
            new_g.write(' 2) 0.000e+00      0       #  dsky/dx (sky gradient in x)     [ADUs/pix]\n')
            new_g.write(' 3) 0.000e+00      0       #  dsky/dy (sky gradient in y)     [ADUs/pix]\n')
            new_g.write(' Z) 0                      #  Skip this model in output image?  (yes=1, no=0)\n')

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


def write_constraintfile(output=None, mag_bri=None, mag_dim=None, re_lo=None, re_hi=None, q_lo=None, num=10, sky=True,
                         psf=False):
    with open(output, 'w+') as newfile:
        newfile.write('# Component/    parameter   constraint	Comment\n')
        if mag_bri is not None:
            newfile.write('1     mag    ' + str(mag_bri) + ' to '+ str(mag_dim) + '\n')
        if re_lo is not None:
            newfile.write('1     re     ' + str(re_lo) + ' to '+ str(re_hi) + '\n')
        nums = np.arange(num) + 1  # +1 because python starts indexing at 0, galfit starts at 1
        string = ''
        if sky:  # if including a sky component
            nums = nums + 1  # don't need to do anything with the sky component
        for n in nums:
            string += str(n) + '_'
        string = string[:-1]
        newfile.write(string + '     x     offset\n')
        newfile.write(string + '     y     offset\n')
        if psf:
            string = string[2:]
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
        component = None
        with open(galfit_out, 'r') as go:
            for line in go:
                cols = line.split()
                if line.startswith(' 0)'):
                    component = cols[1]
                entry = None
                if component == 'gaussian':
                    if line.startswith(' 3)') or line.startswith(' 4)'):
                        entry = '| ' + cols[1] + ' '
                    elif line.startswith(' 9)'):
                        entry = '| ' + cols[1] + '\n'
                    if entry is not None:
                        tf.write(entry)


def galfit_to_mge(galfit_out, zeropoint, texp, write_new=None, glx='UGC 2698'):
    """

    :param galfit_out: the galfit restart file produced by a galfit run that has converged
    :param zeropoint: zeropoint (vega mag; e/s) used in the galfit run
    :param texp: exposure time of the image used in GALFIT (for an image in units of electrons)
    :param write_new: MGE fit sectors-units input mge file name
    :return:
    """

    output_imgblock = None

    xs = []
    ys = []
    pas = []
    es = []
    sigmas = []
    qs = []
    component = None
    with open(galfit_out, 'r') as go:
        for line in go:
            cols = line.split()
            if line.startswith('B)'):
                output_imgblock = cols[1]
            if line.startswith(' 0)'):
                component = cols[1]
            elif line.startswith(' 1)') and component == 'gaussian':
                xs.append(cols[1])
                ys.append(cols[2])
            elif line.startswith(' 3)') and component == 'gaussian':
                es.append(str(texp * 10 ** (0.4 * (zeropoint - float(cols[1])))))
            elif line.startswith(' 4)') and component == 'gaussian':
                sigmas.append(str(float(cols[1]) / 2.355))  # https://en.wikipedia.org/wiki/Full_width_at_half_maximum
            elif line.startswith(' 9)') and component == 'gaussian':
                qs.append(cols[1])
            elif line.startswith('10)') and component == 'gaussian':
                pas.append(str(90 - float(cols[1])))

    if write_new is not None:
        with open(write_new, 'w+') as newfile:
            newfile.write('# ' + glx + ' MGE output from GALFIT, converted back to MGE units\n')
            newfile.write('# output galfit img block: ' + output_imgblock + '\n')
            newfile.write('# galfit restart file: ' + galfit_out + '\n')
            newfile.write('# ang = find_galaxy.theta: "Position angle measured clock-wise from the image X axis"\n')
            newfile.write('# Electrons Sigma_pix qObs x y ang\n')
            for i in range(len(es)):
                newfile.write(es[i] + ' ' + sigmas[i] + ' ' + qs[i] + ' ' + ys[i] + ' ' + xs[i] + ' ' + pas[i] + '\n')

    return es, sigmas, qs, ys, xs, pas


def galfit_to_paper(galfit_out, zeropoint, texp, mag_sol=3.37, pix_scale=0.1, apcorr=0., dust=0.075, write_new=None):
    """

    :param galfit_out: the galfit restart file produced by a galfit run that has converged
    :param zeropoint: zeropoint (vega mag; e/s) used in the galfit run
    :param texp: exposure time of the image used in GALFIT (for an image in units of electrons)
    :param write_new: paper-units file name
    :return:
    """

    output_imgblock = None

    xs = []
    ys = []
    pas = []
    pas_m90 = []
    ctots = []  # electrons (total counts)
    sigmas = []  # [pixels]
    qs = []  # unitless
    component = None
    with open(galfit_out, 'r') as go:
        for line in go:
            cols = line.split()
            if line.startswith('B)'):
                output_imgblock = cols[1]
            if line.startswith(' 0)'):
                component = cols[1]
            elif line.startswith(' 1)') and component == 'gaussian':
                xs.append(cols[1])
                ys.append(cols[2])
            elif line.startswith(' 3)') and component == 'gaussian':
                ctots.append(texp * 10 ** (0.4 * (zeropoint - float(cols[1]))))
            elif line.startswith(' 4)') and component == 'gaussian':
                sigmas.append(float(cols[1]) / 2.355 * pix_scale)
                # https://en.wikipedia.org/wiki/Full_width_at_half_maximum
            elif line.startswith(' 9)') and component == 'gaussian':
                qs.append(float(cols[1]))
            elif line.startswith('10)') and component == 'gaussian':
                pas.append(cols[1])
                pas_m90.append(str(90 - float(cols[1])))

    Is = []  # [Lsol / pc^2]
    Ltots = []
    for c in range(len(ctots)):
        c_0 = ctots[c] / (2 * np.pi * (sigmas[c] / pix_scale)**2 * qs[c])
        mu = zeropoint + apcorr + 5*np.log10(pix_scale) + 2.5*np.log10(texp) - 2.5*np.log10(c_0) - dust
        Is.append((64800 / np.pi)**2 * 10**(0.4 * (mag_sol - mu)))

        Ltots.append(2 * np.pi * Is[c] * sigmas[c]**2 * qs[c])  # need distance in here somewhere?


    '''
                    if logged:
                    mu = mag_sol - 2.5 * np.log10((np.pi / 64800) ** 2 * 10**float(cols[1]))  # cols[1] = log10(I_H)
                else:
                    mu = mag_sol - 2.5 * np.log10((np.pi / 64800)**2 * float(cols[1]))  # cols[1] = I_H [Lsol/pc^2]
                c0 = pix_scale**2 * t_exp * 10**(0.4 * (zp + apcorr - dust - mu))
                total_counts = c0 * 2 * np.pi * (float(cols[2]) / pix_scale)**2 * float(cols[3])
                mag = zp - 2.5*np.log10(total_counts)
                if convert_to_app:
                    mag = mag + 5. * np.log10(dist) - 5.
                mags.append(mag)  # above: cols[2] = sigma_arcsec, cols[3] = qObs
                fwhms.append(2.355 * float(cols[2]) / pix_scale)  # fwhm = 2.355 * np.array(sigma_arcsec) / pix_scale
    '''

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

    # For M_Hsol use: http://mips.as.arizona.edu/~cnaw/sun.html
    # (this is 3.37, for WFC3 F160W under vegamag; cite Wilmar et al 2018 [ApJS paper])

    if write_new is not None:
        with open(write_new, 'w+') as newfile:
            newfile.write('# UGC 2698 MGE output from GALFIT, converted to paper units\n')
            newfile.write('# Calculated using zeropoint =' + str(zeropoint) + ', exposure time=' + str(texp) + ' s, ' +
                          'pixel scale=' + str(pix_scale) + ' arcsec/pix, an extinction correction of ' + str(dust) +
                          " mag in the H-band, " + "and eqns 1-3 of Cappellari's MGE fit sectors readme\n")
            newfile.write('# output galfit img block: ' + output_imgblock + '\n')
            newfile.write("# galfit restart file for converged run: " + galfit_out + '\n')
            newfile.write('# PA found in GALFIT = ' + str(pas[0]) + ': "The position angle is 0 if the semi-major' +
                          'axis is aligned parallel to the Y-axis and increases toward the counter-clockwise' +
                          'direction"\n')
            newfile.write('# Component I[Lsol,H/pc^2] Sigma[arcsec] qObs[unitless]\n')  # ' Ltot[Lsol,H]\n')
            for i in range(len(Is)):
                newfile.write(str(i) + ' ' + str(Is[i]) + ' ' + str(sigmas[i]) + ' ' + str(qs[i]) + '\n')


def convert_mges():
    base = '/Users/jonathancohn/Documents/mge/'
    gf = '/Users/jonathancohn/Documents/dyn_mod/galfit_u2698/'


    zp = 24.6949
    texp = 898.467164
    sky_e = 337.5  # 339.493665331  # sky_eps = 0.377858732438
    rms_e = 22.4  # 21.5123034564  # rms_eps = 0.0239433389818

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
    re = gf + 'ugc2698_f160w_pxfr075_pxs010_drz_rapidnuc_sci_nonan_e.fits'  # regH image!
    ahe = gf + 'ugc2698_f160w_pxfr075_pxs010_ahcorr_rapidnuc_sci_nonan_e.fits'  # ahcorr image!
    # Masks
    comb_mask = gf + 'f160w_combinedmask_px010.fits'  # include dust mask!
    reg_mask = gf + 'f160w_maskedgemask_px010.fits'  # don't include dust mask!
    # New GALFIT param files!
    pars_gal_ahe = gf + 'galfit_params_u2698_ahe_linear_pf001_zp24.txt'
    pars_gal_rhe = gf + 'galfit_params_u2698_rhe_linear_pf001_zp24.txt'
    pars_gal_rre = gf + 'galfit_params_u2698_rre_linear_pf001_zp24.txt'
    # New GALFIT output names (GALFIT output image blocks .fits)
    out_gal_ahe = gf + 'galfit_out_u2698_ahe_linear_pf001_zp24.fits'
    out_gal_rhe = gf + 'galfit_out_u2698_rhe_linear_pf001_zp24.fits'
    out_gal_rre = gf + 'galfit_out_u2698_rre_linear_pf001_zp24.fits'
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


def convert_mge_galfitpsf():
    gf = '/Users/jonathancohn/Documents/dyn_mod/galfit_u2698/'

    zp = 24.6949
    texp = 898.467164
    sky_e = 337.5  # 339.493665331  # sky_eps = 0.377858732438
    rms_e = 22.4  # 21.5123034564  # rms_eps = 0.0239433389818

    # MGEs
    mge_rhe = gf + 'mge_ugc_2698_rhe_linear_pf001.txt'
    mge_ahe = gf + 'mge_ugc_2698_ahe_linear_pf001.txt'
    mge_rre = gf + 'mge_ugc_2698_rre_linear_pf001.txt'
    # New converted MGE text files (to write)
    conv_ahe = gf + 'convpsf_mge_ugc_2698_ahe_linear_pf001.txt'
    conv_rhe = gf + 'convpsf_mge_ugc_2698_rhe_linear_pf001.txt'
    conv_rre = gf + 'convpsf_mge_ugc_2698_rre_linear_pf001.txt'
    # IMGs
    re = gf + 'ugc2698_f160w_pxfr075_pxs010_drz_rapidnuc_sci_nonan_e.fits'  # regH image!
    ahe = gf + 'ugc2698_f160w_pxfr075_pxs010_ahcorr_rapidnuc_sci_nonan_e.fits'  # ahcorr image!
    # Masks
    comb_mask = gf + 'f160w_combinedmask_px010.fits'  # include dust mask!
    reg_mask = gf + 'f160w_maskedgemask_px010.fits'  # don't include dust mask!
    # New GALFIT param files!
    pars_gal_ahe = gf + 'galfit_paramspsf_u2698_ahe_linear_pf001_zp24.txt'
    pars_gal_rhe = gf + 'galfit_paramspsf_u2698_rhe_linear_pf001_zp24.txt'
    pars_gal_rre = gf + 'galfit_paramspsf_u2698_rre_linear_pf001_zp24.txt'
    # New GALFIT output names (GALFIT output image blocks .fits)
    out_gal_ahe = gf + 'galfit_outpsf_u2698_ahe_linear_pf001_zp24.fits'
    out_gal_rhe = gf + 'galfit_outpsf_u2698_rhe_linear_pf001_zp24.fits'
    out_gal_rre = gf + 'galfit_outpsf_u2698_rre_linear_pf001_zp24.fits'
    # Constraint files
    cons_8 = gf + 'conspsf_ugc_2698_n8.txt'
    cons_9 = gf + 'conspsf_ugc_2698_n9.txt'
    cons_10 = gf + 'conspsf_ugc_2698_n10.txt'  # ahe
    cons_11 = gf + 'conspsf_ugc_2698_n11.txt'  # rhe, rre
    cons_12 = gf + 'conspsf_ugc_2698_n12.txt'
    # Copy file (galfit file of which I'm copying the structure)
    copyf = gf + 'galfit_params_mge_055_zp25.txt'

    qs = None
    write_constraintfile(output=cons_8, q_lo=qs, num=8, psf=True)  # q_los[c]
    write_constraintfile(output=cons_9, q_lo=qs, num=9, psf=True)  # q_los[c]
    write_constraintfile(output=cons_10, q_lo=qs, num=10, psf=True)  # q_los[c]
    write_constraintfile(output=cons_11, q_lo=qs, num=11, psf=True)  # q_los[c]
    write_constraintfile(output=cons_12, q_lo=qs, num=12, psf=True)  # q_los[c]

    # ahe, rhe, rre
    cv = mge_to_psfgalfit(mge_ahe, zp, img=ahe, mask=reg_mask, constraint=cons_10, copy_file=copyf, galfit_out=out_gal_ahe,
                       write_new=conv_ahe, new_galfit=pars_gal_ahe, sky=sky_e, rms=rms_e, texp=texp)
    cv = mge_to_psfgalfit(mge_rhe, zp, img=re, mask=comb_mask, constraint=cons_11, copy_file=copyf, galfit_out=out_gal_rhe,
                       write_new=conv_rhe, new_galfit=pars_gal_rhe, sky=sky_e, rms=rms_e, texp=texp)
    cv = mge_to_psfgalfit(mge_rre, zp, img=re, mask=reg_mask, constraint=cons_11, copy_file=copyf, galfit_out=out_gal_rre,
                       write_new=conv_rre, new_galfit=pars_gal_rre, sky=sky_e, rms=rms_e, texp=texp)


def print_mge_table(paper_file):

    # PRINT OUT THE MGE FOR PAPER TABLE

    js = []
    Is = []  # intensities (Lsol,H/pc^2)
    sigmas = []  # [arcsec]
    qs = []  # unitless
    with open(paper_file, 'r') as pf:
        for line in pf:
            if not line.startswith('#'):
                cols = line.split()
                js.append(float(cols[0]))
                Is.append(float(cols[1]))
                sigmas.append(float(cols[2]))
                qs.append(float(cols[3]))

    fmt3 = "{{0:{0}}}".format('.3f').format
    for j in range(len(js)):
        print(str(int(j+1)) + ' & ' + str(fmt3(np.log10(Is[j]))) + ' & ' + str(fmt3(sigmas[j])) + ' & ' +
              str(fmt3(qs[j])) + ' \\\\')


if __name__ == "__main__":
    base = '/Users/jonathancohn/Documents/mge/'
    dm = '/Users/jonathancohn/Documents/dyn_mod/'
    u2698 = '/Users/jonathancohn/Documents/dyn_mod/ugc_2698/'
    fj = '/Users/jonathancohn/Documents/dyn_mod/for_jonathan/'
    # gf = '/Users/jonathancohn/Documents/dyn_mod/galfit_u2698/'
    gu = '/Users/jonathancohn/Documents/dyn_mod/galfit_u2698/'
    gf = '/Users/jonathancohn/Documents/dyn_mod/galfit/'
    gp = gf + 'p11179/'

    reg_p11179 = {'img': fj+'PGC11179_F160W_drz_sci.fits', 'mask': fj+'PGC11179_F160W_drz_mask.fits',
                  'mge': gp+'mge_pgc_11179_reg_linear.txt', 'conv': gp+'convmge_pgc_11179_reg_linear.txt',
                  'out': gp+'galfit_out_p11179_reg_linear.fits', 'new' :gp+'galfit_in_p11179_reg_linear.txt',
                  'cons': gp+'constraint_p11179_n10.txt', 'outconv': base+'convgalfit_out_p11179_reg_linear.txt',
                  'glx': 'PGC 11179', 'zp': 24.662, 'exp': 1354.463046}
    # zeropoint: https://www.stsci.edu/hst/instrumentation/wfc3/data-analysis/photometric-calibration/ir-photometric-calibration#section-cc19dbfc-8f60-4870-8765-43810de39924

    galfit_to_mge(gp + 'galfit.01', reg_p11179['zp'], reg_p11179['exp'], reg_p11179['outconv'])
    print(oop)
    mge_to_galfit(reg_p11179['mge'], zeropoint=reg_p11179['zp'], img=reg_p11179['img'], mask=reg_p11179['mask'],
                  copy_file=gu+'galfit_params_mge_055_zp25.txt', galfit_out=reg_p11179['out'],
                  write_converted=reg_p11179['conv'], new_galfit=reg_p11179['new'], constraint=reg_p11179['cons'],
                  sky=None, rms=None, glx=reg_p11179['glx'])
    print(oops)

    zp = 24.6949
    texp = 898.467164
    sky_e = 337.5  # 339.493665331  # sky_eps = 0.377858732438
    rms_e = 22.4  # 21.5123034564  # rms_eps = 0.0239433389818

    '''
    galfit_to_paper(gf + 'galfit.120', zp, texp, write_new=dm + 'ugc_2698_rre_mge.txt')
    galfit_to_paper(gf + 'galfit.121', zp, texp, write_new=dm + 'ugc_2698_rhe_mge.txt')
    galfit_to_paper(gf + 'galfit.122', zp, texp, write_new=dm + 'ugc_2698_ahe_mge.txt')
    galfit_to_paper(gf + 'galfit.113', zp, texp, write_new=dm + 'ugc_2698_rrepsf_mge.txt')
    galfit_to_paper(gf + 'galfit.116', zp, texp, write_new=dm + 'ugc_2698_rhepsf_mge.txt')
    galfit_to_paper(gf + 'galfit.118', zp, texp, write_new=dm + 'ugc_2698_ahepsf_mge.txt')
    '''
    print_mge_table(u2698 + 'ugc_2698_rhe_mge.txt')
    print(oop)

    # galfit_to_wiki(gf + 'galfit.122', gf + 'wiki_galfit_out_u2698_ahenewsky_linear_pf001_zp24.txt')
    # galfit_to_mge(gf + 'galfit.122', zp, texp, gf + 'mconv_galfit_out_u2698_ahenewsky_linear_pf001_zp24.txt')
    # galfit_to_wiki(gf + 'galfit.121', gf + 'wiki_galfit_out_u2698_rhenewsky_zp24.txt')
    # galfit_to_mge(gf + 'galfit.121', zp, texp, gf + 'mconv_galfit_out_u2698_rhenewsky_zp24.txt')
    # galfit_to_wiki(gf + 'galfit.120', gf + 'wiki_galfit_out_u2698_rrenewsky_linear_pf001_zp24.txt')
    # galfit_to_mge(gf + 'galfit.120', zp, texp, gf + 'mconv_galfit_out_u2698_rrenewsky_linear_pf001_zp24.txt')
    # galfit_to_wiki(gf + 'galfit.118', gf + 'wiki_galfit_out_u2698_ahepsf_linear_pf001_zp24.txt')
    # galfit_to_mge(gf + 'galfit.118', zp, texp, gf + 'mconv_galfit_out_u2698_ahepsf_linear_pf001_zp24.txt')
    # galfit_to_wiki(gf + 'galfit.116', gf + 'wiki_galfit_out_u2698_rhepsf_linear_pf001_zp24.txt')
    # galfit_to_mge(gf + 'galfit.116', zp, texp, gf + 'mconv_galfit_out_u2698_rhepsf_linear_pf001_zp24.txt')
    # galfit_to_wiki(gf + 'galfit.113', gf + 'wiki_galfit_out_u2698_rrepsf_linear_pf001_zp24.txt')
    # galfit_to_mge(gf + 'galfit.113', zp, texp, gf + 'mconv_galfit_out_u2698_rrepsf_linear_pf001_zp24.txt')
    # galfit_to_wiki(gf + 'galfit.78', gf + 'wiki_Ben_ahcorr_mge_out.txt')
    # galfit_to_wiki(gf + 'galfit.112', gf + 'wiki_galfit_out_u2698_ahe_linear_pf001_zp24.txt')
    # galfit_to_mge(gf + 'galfit.112', zp, texp, gf + 'mconv_galfit_out_u2698_ahe_linear_pf001_zp24.txt')
    #galfit_to_wiki(gf + 'galfit.107', gf + 'wiki_galfit_out_u2698_rhe_galfit106restart.txt')
    #galfit_to_mge(gf + 'galfit.107', zp, texp, gf + 'mconv_galfit_out_u2698_rre_galfit106restart.txt')
    #galfit_to_wiki(gf + 'galfit.106', gf + 'wiki_galfit_out_u2698_rre_linear_pf001_zp24.txt')
    #galfit_to_mge(gf + 'galfit.106', zp, texp, gf + 'mconv_galfit_out_u2698_rre_linear_pf001_zp24.txt')

    # convert_mges()
    # convert_mge_galfitpsf()
    # print(oop)

    # '''  #
    # PAPER TO GALFIT: ahe newsky electrons/s units (use dt=False, es=True)!
    paper_to_galfit(dm + 'ugc_2698_ahe_mge.txt', mag_sol=3.37, pix_scale=0.1, t_exp=898.467164, zp=24.695, apcorr=0.0,
                    dust=0.075, xctr=490.19072764757743, yctr=879.72232410674587, pa=-7.9971,
                    img='none', mask='none', copy_file=gf+'galfit_params_mge_055_zp25.txt', es=True, dt=False,
                    galfit_out=gf+'galfit_out_u2698_ahe_esnodt.fits',  # _t898; _dt; _corr; _corrdt
                    new_galfit=gf+'galfit_params_u2698_ahe_esnodt.txt', # _t898; _dt
                    compare_file=gf+'conv_galfit_input_ahe_esnodt.txt')  # _t898; _dt

    # PAPER TO GALFIT: rhe newsky electrons/s units (use dt=False, es=True)!
    paper_to_galfit(dm + 'ugc_2698_rhe_mge.txt', mag_sol=3.37, pix_scale=0.1, t_exp=898.467164, zp=24.695, apcorr=0.0,
                    dust=0.075, xctr=490.13910728473815, yctr=879.68141083841419, pa=-7.9913,
                    img='none', mask='none', copy_file=gf+'galfit_params_mge_055_zp25.txt', es=True, dt=False,
                    galfit_out=gf+'galfit_out_u2698_rhe_esnodt.fits',  # _t898; _dt; _corr; _corrdt
                    new_galfit=gf+'galfit_params_u2698_rhe_esnodt.txt', # _t898; _dt
                    compare_file=gf+'conv_galfit_input_rhe_esnodt.txt')  # _t898; _dt

    # PAPER TO GALFIT: rre newsky electrons/s units (use dt=False, es=True)!
    paper_to_galfit(dm + 'ugc_2698_rre_mge.txt', mag_sol=3.37, pix_scale=0.1, t_exp=898.467164, zp=24.695, apcorr=0.0,
                    dust=0.075, xctr=490.07093770235764, yctr=879.83348090657728, pa=-7.9997,
                    img='none', mask='none', copy_file=gf+'galfit_params_mge_055_zp25.txt', es=True, dt=False,
                    galfit_out=gf+'galfit_out_u2698_rre_esnodt.fits',  # _t898; _dt; _corr; _corrdt
                    new_galfit=gf+'galfit_params_u2698_rre_esnodt.txt', # _t898; _dt
                    compare_file=gf+'conv_galfit_input_rre_esnodt.txt')  # _t898; _dt
    print(oop)
    # '''  #

    # '''  #
    # PAPER TO GALFIT: Akin  # corrdt is correct! Now try with electrons/s units (use dt=False, es=True)!
    # paper_to_galfit(base + 'yildirim_table_2698.txt', mag_sol=3.32, pix_scale=0.1, t_exp=texp, zp=24.695,
    paper_to_galfit(base + 'yildirim_table_2698.txt', mag_sol=3.32, pix_scale=0.06, t_exp=1354.46, zp=24.695,  # 1354.46
                    # apcorr=0.0, dust=0.075, xctr=491.0699, yctr=880.8322, pa=108.61-116.7549279549141, dist=89. * 1e6,
                    apcorr=0.0, dust=0.075, xctr=490.85792403050715, yctr=880.0990585664232, pa=108.61-116.7549279549141, dist=89. * 1e6,
                    # new x,yctrs based on x,y from find_galaxy run on rrenewsky output model
                    img='none', mask='none', copy_file=gf+'galfit_params_mge_055_zp25.txt', es=True, dt=False,
                    galfit_out=gf+'galfit_out_u2698_akin_esnodt.fits',  # _t898; _dt; _corr; _corrdt
                    new_galfit=gf+'galfit_params_u2698_akin_esnodt.txt', # _t898; _dt
                    compare_file=gf+'conv_galfit_input_akin_esnodt.txt')  # _t898; _dt
    print(oop)

    # PAPER TO GALFIT: Ben N3258
    paper_to_galfit(u2698 + 'ben_n3258_mge.txt', mag_sol=3.37, pix_scale=0.08, t_exp=1., zp=24.695, logged=True,
                    apcorr=0.0, dust=0.075, xctr=491.0699, yctr=880.8322, pa=77., dist=31.9 * 1e6,
                    img='none', mask='none', copy_file=gf+'galfit_params_mge_055_zp25.txt',
                    galfit_out=gf+'galfit_out_n3258_ben_test.fits',
                    new_galfit=gf+'galfit_params_n3258_ben_test.txt',
                    compare_file=gf+'conv_galfit_input_n3258_ben_test.txt')
    print(oop)


    # PAPER TO GALFIT: Akin  # corrdt is correct! Now try with electrons/s units!
    # paper_to_galfit(base + 'yildirim_table_2698.txt', mag_sol=3.32, pix_scale=0.1, t_exp=texp, zp=24.695,
    paper_to_galfit(base + 'yildirim_table_2698.txt', mag_sol=3.32, pix_scale=0.06, t_exp=1354.46, zp=24.695,  # 1354.46
                    # apcorr=0.0, dust=0.075, xctr=491.0699, yctr=880.8322, pa=108.61-116.7549279549141, dist=89. * 1e6,
                    apcorr=0.0, dust=0.075, xctr=490.85792403050715, yctr=880.0990585664232, pa=108.61-116.7549279549141, dist=89. * 1e6,
                    # new x,yctrs based on x,y from find_galaxy run on rrenewsky output model
                    img='none', mask='none', copy_file=gf+'galfit_params_mge_055_zp25.txt', es=True,
                    galfit_out=gf+'galfit_out_u2698_akin_fixedmod_zp24_corrdt_es_newxy.fits',  # _t898; _dt; _corr; _corrdt
                    new_galfit=gf+'galfit_params_u2698_akin_fixedmod_zp24_corrdt_es_newxy.txt', # _t898; _dt
                    compare_file=gf+'conv_galfit_input_akin_fixedmod_zp24_corrdt_es_newxy.txt')  # _t898; _dt
    # 3 * 450s
    print(oop)

    # PAPER TO GALFIT: Akin  # corrdt is correct!
    # paper_to_galfit(base + 'yildirim_table_2698.txt', mag_sol=3.32, pix_scale=0.1, t_exp=texp, zp=24.695,
    paper_to_galfit(base + 'yildirim_table_2698.txt', mag_sol=3.32, pix_scale=0.06, t_exp=1354.46, zp=24.695,  # 1354.46
                    apcorr=0.0, dust=0.075, xctr=491.0699, yctr=880.8322, pa=108.61-116.7549279549141, dist=89. * 1e6,
                    img=gf+'ugc2698_f160w_pxfr075_pxs010_drz_rapidnuc_sci_nonan_e_akintexp2.fits',  # 'none',
                    mask='none',
                    copy_file=gf+'galfit_params_mge_055_zp25.txt',
                    galfit_out=gf+'galfit_out_u2698_akin_fixedmod_zp24_corrdtt2.fits',  # _t898; _dt; _corr; _corrdt
                    new_galfit=gf+'galfit_params_u2698_akin_fixedmod_zp24_corrdtt2.txt', # _t898; _dt
                    compare_file=gf+'conv_galfit_input_akin_fixedmod_zp24_corrdtt2.txt')  # _t898; _dt
    # 3 * 450s
    print(oop)
    # TEST PAPER TO GALFIT TO PAPER: this worked! yildirim_convert_back.txt matches yildirim_table_2698.txt!
    galfit_to_paper(gf + 'galfit_params_u2698_akin_fixedmod_zp24_corrdt_es_newxy.txt', zeropoint=24.695, texp=1354.46,
                    mag_sol=3.32, pix_scale=0.06, apcorr=0., dust=0.075, write_new=gf + 'yildirim_convert_back.txt')
    print(oop)
    # PAPER TO GALFIT: Ben (rre)
    paper_to_galfit(u2698 + 'ugc_2698_mge.txt', mag_sol=3.37, pix_scale=0.1, t_exp=texp, zp=24.695, logged=True,
                    apcorr=0.0, dust=0.075, xctr=491.0699, yctr=880.8322, pa=109.08-116.7549279549141, dist=89.*1e6,
                    img=gf+'ugc2698_f160w_pxfr075_pxs010_drz_rapidnuc_sci_nonan_e.fits',  # 'none',
                    mask=gf+'f160w_maskedgemask_px010.fits',
                    copy_file=gf+'galfit_params_mge_055_zp25.txt',
                    galfit_out=gf+'galfit_out_u2698_ben_fixedmod_zp24.fits',  # _divtexp  # _re
                    new_galfit=gf+'galfit_params_u2698_ben_fixedmod_zp24.txt', # _divtexp  # _re
                    compare_file=gf+'conv_galfit_input_ben_fixedmod_zp24.txt')  # _divtexp  # _re
    print(oop)

    # TEST PAPER TO GALFIT: rre_mge: this worked! So conversions are at least self-consistent
    paper_to_galfit(dm + 'ugc_2698_rre_mge.txt', mag_sol=3.37, pix_scale=0.1, t_exp=texp, zp=24.695, apcorr=0.0,
                    dust=0.075, xctr=491.0699, yctr=880.8322, pa=-7.9997, dist=89.*1e6,
                    img=gf + 'ugc2698_f160w_pxfr075_pxs010_drz_rapidnuc_sci_nonan_e.fits',
                    mask=gf + 'f160w_maskedgemask_px010.fits',
                    copy_file=gf + 'galfit_params_mge_055_zp25.txt',
                    galfit_out=gf + 'test.fits',
                    new_galfit=gf + 'galfit_params_from_ugc_2698_rre_mge.txt',
                    compare_file=gf + 'conv_galfit_input_from_ugc_2698_rre_mge.txt', dt=True)
    print(oop)
    '''  #

    '''  #
    # write constraint files
    write_constraintfile(output=cons_8, q_lo=qs, num=8)  # q_los[c]
    write_constraintfile(output=cons_9, q_lo=qs, num=9)  # q_los[c]
    write_constraintfile(output=cons_10, q_lo=qs, num=10)  # q_los[c]
    write_constraintfile(output=cons_11, q_lo=qs, num=11)  # q_los[c]
    write_constraintfile(output=cons_12, q_lo=qs, num=12)  # q_los[c]
    # '''  #