import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy import wcs
from os import path
import scipy.optimize as opt
from scipy import ndimage
from scipy.ndimage import interpolation
from matplotlib import gridspec
import matplotlib as mpl
import dynamical_model as dm

mpl.rcParams['text.usetex'] = True
# mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']  # for \text command


def rot(image, x0, y0, angle=116.755, reshape=True):
    # https://stackoverflow.com/questions/46657423/rotated-image-coordinates-after-scipy-ndimage-interpolation-rotate
    image_rotated = interpolation.rotate(image, angle=angle, reshape=reshape)  # CORRECT ROTATION! Now center it ahh
    # im_rot = rotate(image,angle)
    original_center = (np.array(image.shape[:2][::-1])-1)/2.
    rotated_center = (np.array(image_rotated.shape[:2][::-1])-1)/2.
    original_offset = [x0, y0] - original_center
    a = np.deg2rad(angle)
    new_offset = np.array([original_offset[0]*np.cos(a) + original_offset[1]*np.sin(a),
                           -original_offset[0]*np.sin(a) + original_offset[1]*np.cos(a)])
    new_point = new_offset + rotated_center
    return image_rotated, new_point


def display_mod(galfit_out=None, texp=898.467164, sky=339.493665331):
    """
    This function displays the model mge of UGC 2698, from fit_ugc2698 -> GALFIT -> out_galfit.py -> input here!

    """
    mags = []
    fwhms = []
    qs = []
    pas = []
    xc1 = []
    yc1 = []
    component = None
    file = None
    outfile = None
    with open(galfit_out, 'r') as go:
        for line in go:
            cols = line.split()
            if line.startswith('A)'):
                file = cols[1]
            elif line.startswith('B)'):
                outfile = cols[1]
            elif line.startswith('D)'):
                psf_file = cols[1]
            elif line.startswith('F)'):
                mask = cols[1]
            elif line.startswith('J)'):
                zp = float(cols[1])
            elif line.startswith('K)'):
                scale1 = float(cols[1])  # (arcsec/pixel); cols[1] = dx, cols[2] = dy; for this case, dx = dy = 0.1
            elif line.startswith(' 0)'):
                component = cols[1]
            elif line.startswith(' 1)') and component == 'sky':
                sky = float(cols[1])
            elif line.startswith(' 1)') and component == 'gaussian':
                yc1.append(float(cols[1]))  # swap from GALFIT to MGE units, so swap x & y
                xc1.append(float(cols[2]))  # swap from GALFIT to MGE units, so swap x & y
            elif line.startswith(' 3)') and component == 'gaussian':
                mags.append(float(cols[1]))  # integrated mags
            elif line.startswith(' 4)') and component == 'gaussian':
                fwhms.append(float(cols[1]))  # fwhm_pix
            elif line.startswith(' 9)') and component == 'gaussian':
                qs.append(float(cols[1]))  # qObs
            elif line.startswith('10)') and component == 'gaussian':
                pas.append(float(cols[1]))

    # scale1 = 0.1
    hdu = fits.open(file)
    img1 = hdu[0].data
    img1 -= sky  # subtract sky
    hdu.close()

    hdu = fits.open(outfile)
    mod_img = hdu[2].data
    mod_img -= sky
    hdu.close()

    sols = np.zeros(shape=(3,len(qs)))  # axes: electrons, sigma_pix, qObs
    for i in range(len(mags)):
        sols[0][i] = texp * 10 ** (0.4 * (zp - mags[i]))  # electrons
        sols[1][i] = fwhms[i] / 2.355  # https://en.wikipedia.org/wiki/Full_width_at_half_maximum
        sols[2][i] = qs[i]

    # Plot MGE contours of the HST image
    xc = xc1[0]
    yc = yc1[0]
    pa = pas[0]

    peak = img1[int(round(xc)), int(round(yc))]  # xc, yc
    magrange = 10.
    levels = peak * 10**(-0.4*np.arange(0, magrange, 0.5)[::-1]) # 0.5 mag/arcsec^2 steps

    binning = 1
    if binning is None:
        binning = 1
    else:
        img1 = ndimage.filters.gaussian_filter(img1, binning/2.355)
        img1 = ndimage.zoom(img1, 1./binning, order=1)
        mod_img = ndimage.filters.gaussian_filter(mod_img, binning/2.355)
        mod_img = ndimage.zoom(mod_img, 1./binning, order=1)

    s = img1.shape
    scale = scale1
    if scale is None:
        extentc = [0, s[1], 0, s[0]]
        xylabel = "pixels"
    else:
        extentc = np.array([0, s[1], 0, s[0]]) * scale * binning
        xylabel = "arcsec"

    return img1, mod_img, levels, extentc, xylabel


def dust_and_co(parfile, beamloc=(-1.,-1.), incl_beam=True, xmatch=66, ymatch=32, bcolor='w', avging=True, vrad=None,
                kappa=None, omega=None, snr=10):

    # Read parameter file, create params dictionary
    params, priors, nfree, qobs = dm.par_dicts(parfile, q=True)  # get params and file names from output parfile

    if 'ds2' not in params:  # if down-sampling is square
        params['ds2'] = params['ds']  # set undefined second down-sampling factor equal to the first

    # Run through model prep, create items needed in model
    mod_ins = dm.model_prep(data=params['data'], ds=params['ds'], ds2=params['ds2'], lucy_out=params['lucy'],
                            lucy_b=params['lucy_b'], lucy_mask=params['lucy_mask'], lucy_in=params['lucy_in'],
                            lucy_it=params['lucy_it'], data_mask=params['mask'], grid_size=params['gsize'],
                            res=params['resolution'], x_std=params['x_fwhm'], y_std=params['y_fwhm'], avg=avging,
                            xyerr=[params['xerr0'], params['xerr1'], params['yerr0'], params['yerr1']],
                            zrange=[params['zi'], params['zf']], theta_ell=np.deg2rad(params['theta_ell']),
                            xell=params['xell'], yell=params['yell'], q_ell=params['q_ell'], pa=params['PAbeam'])

    lucy_mask, lucy_out, beam, fluxes, freq_ax, f_0, fstep, input_data, noise, co_rad, co_sb = mod_ins

    # Check if radial velocity is used in the model
    if 'vrad' in parfile:
        vrad = params['vrad']
    elif 'omega' in parfile:
        kappa = params['kappa']
        omega = params['omega']
    elif 'kappa' in parfile:
        kappa = params['kappa']

    # Check if gas mass is included in the model
    inc_fixed = np.deg2rad(67.7)  # based on fiducial model (67.68 deg)
    vcg_in = None
    if params['incl_gas'] == 'True':
        vcg_in = dm.gas_vel(params['resolution'], co_rad, co_sb, params['dist'], f_0, inc_fixed, zfixed=0.02152)

    # Initialize the model class
    mg = dm.ModelGrid(x_loc=params['xloc'], y_loc=params['yloc'], mbh=params['mbh'], ml_ratio=params['ml_ratio'],
                      inc=np.deg2rad(params['inc']), vsys=params['vsys'], theta=np.deg2rad(params['PAdisk']), vrad=vrad,
                      kappa=kappa, omega=omega, f_w=params['f'], os=params['os'], enclosed_mass=params['mass'],
                      sig_params=[params['sig0'], params['r0'], params['mu'], params['sig1']],
                      resolution=params['resolution'],
                      lucy_out=lucy_out, out_name=None, beam=beam, rfit=params['rfit'],
                      zrange=[params['zi'], params['zf']],
                      dist=params['dist'], input_data=input_data, sig_type=params['s_type'], menc_type=params['mtype'],
                      theta_ell=np.deg2rad(params['theta_ell']), xell=params['xell'], yell=params['yell'],
                      q_ell=params['q_ell'],
                      ds=params['ds'], ds2=params['ds2'], reduced=True, f_0=f_0, freq_ax=freq_ax, noise=noise,
                      bl=params['bl'],
                      fstep=fstep, xyrange=[params['xi'], params['xf'], params['yi'], params['yf']], n_params=nfree,
                      data_mask=params['mask'], incl_gas=params['incl_gas'] == 'True', co_rad=co_rad, co_sb=co_sb,
                      vcg_func=vcg_in,
                      pvd_width=params['x_fwhm'] / params['resolution'], avg=avging, quiet=True)

    mg.grids()  # compute the model grid
    mg.convolution()  # convolve the model
    chi2, chi2_nu = mg.chi2()  # calcuate chi^2 because this step creates self.clipped_data, which we need in mg.vorm0

    # Calculate the voronoi-binned moment0 for the data
    d0, min0, max0, extent, cmap_0, ebeam = mg.vorm0(incl_beam=incl_beam, snr=snr, params=params, beamloc=beamloc,
                                                     ymatch=ymatch, xmatch=xmatch, bcolor=bcolor)

    return d0, min0, max0, extent, cmap_0, ebeam


def make_fig1(hband='galfit_u2698/ugc2698_f160w_pxfr075_pxs010_drz_rapidnuc_sci.fits',
              iband='galfit_u2698/ugc2698_f814w_pxfr075_pxs010_drc_align_sci.fits',
              parfile=None, zp_I=24.684, zp_H=24.6949, texp_I=805., texp_H=898.467164, xi=830-1, xf=930-1, yi=440-1,
              yf=543-1, x_galcen=880.8322-1, y_galcen=491.0699-1, resolution=0.1, galfit_out='galfit_u2698/galfit.121',
              sky=339.493665331, incl_beam=True, arcsec=False):  # sky=337.5
    # 491.33878521953153 879.9361312648741

    hduI = fits.open(iband)
    idat = hduI[0].data
    hduI.close()
    hduH = fits.open(hband)
    hdat = hduH[0].data
    hhdr = hduH[0].header
    hduH.close()

    xref = 809.6162680387491
    yref = 615.3175137042995

    x_galint = 881-1#+1 # 878 # 881
    y_galint = 491-1#+1  # 492 # 491

    # FROM FIND GALAXY! IF NOT USING FIND GALAXY, COMMENT THESE OUT
    #x_galcen = 879.9361312648741
    #y_galcen = 491.33878521953153

    # GALFIT RHE cen: 880.6707 491.1347
    # RHE xloc yloc = 126.85356933052101 150.96256939040606

    #
    locations_find_galaxy = {'x_galcen': 879.9361312648741, 'y_galcen': 491.33878521953153, 'hlabel_x': 1.8, 'hlabel_y':
                             1.8, 'ihlabel_y': -2.15, 'beamloc': (1.8,-2.), 'alma_x': 115, 'alma_y': 158}
    locations_galfit = {'x_galcen': 879.8322, 'y_galcen': 490.0699, 'hlabel_x': 2., 'hlabel_y': 1.7, 'ihlabel_y': -2.2,
                        'beamloc': (1.85, -2.), 'alma_x': 116, 'alma_y': 160}
    locations_zerozero = {'x_galcen': 879.6707, 'y_galcen': 490.1347, 'hlabel_x': 2., 'hlabel_y': 1.7, 'ihlabel_y':
                          -2.2, 'beamloc': (1.85, -2.), 'alma_x': 126.85356933052101, 'alma_y': 150.96256939040606}
    locations_galfitrhe = {'x_galcen': 879.6707, 'y_galcen': 490.1347, 'hlabel_x': 2., 'hlabel_y': 1.7, 'ihlabel_y':
                           -2.2, 'beamloc': (1.85, -2.), 'alma_x': 118, 'alma_y': 160}
    # locations_zerozero must be wrong -- ALMA: RA, DEC 03:22:02.898, 40.51.50.068, HST: 3:22:02.914, 40.51.50.248
    # (matches ALMA 118 160, not 127 151)
    using_locs = locations_galfitrhe  # locations_find_galaxy

    x_galcen = using_locs['x_galcen']
    y_galcen = using_locs['y_galcen']
    hx = using_locs['hlabel_x']
    hy = using_locs['hlabel_y']
    ihy = using_locs['ihlabel_y']
    bl = using_locs['beamloc']
    xmatch = using_locs['alma_x']
    ymatch = using_locs['alma_y']

    hdua = fits.open('/Users/jonathancohn/Documents/dyn_mod/ugc_2698/UGC2698_C4_CO21_bri_20.3kms.pbcor.fits')
    hdra = hdua[0].header
    hdua.close()

    # MAKE ALMA RA, DEC ARRAYS
    ras = []
    decs = []
    scale = 1.
    if arcsec:
        scale = 3600.
    for xx in range(hdra['NAXIS1']):
        ras.append(((xx - hdra['CRPIX1']) * hdra['CDELT1'] + hdra['CRVAL1'])*scale)
    for yy in range(hdra['NAXIS2']):
        decs.append(((yy - hdra['CRPIX2']) * hdra['CDELT2'] + hdra['CRVAL2'])*scale)
    # ras[150] = 50.51190972220555  # 50.512743055538884 : 50.51108194442777
    # decs[150] = 40.86389444444445  # 40.863061111111115 : 40.86472222222223

    # MAKE HST RA, DEC ARRAYS
    if arcsec:  # in arcsec units
        radec_file = 'hband_ra_dec6.txt'  # *6.txt: same as *5.txt but in arcsec (*= 3600) ;; INCORRECT SETUP
    else:  # in deg units
        radec_file = 'hband_ra_dec7.txt'  # *5.txt: same as *6.txt but in deg ;; *7.txt corrected!!

    rah = np.zeros(shape=hdat.shape)
    dech = np.zeros(shape=hdat.shape)
    with open('/Users/jonathancohn/Documents/dyn_mod/ugc_2698/' + radec_file, 'r') as hfile:
        for line in hfile:
            if not line.startswith('#'):
                cols = line.split()
                rah[int(cols[1]), int(cols[0])] = float(cols[2])
                dech[int(cols[1]), int(cols[0])] = float(cols[3])

    # CONVERT IMAGES TO MAG UNITS!
    # zp_I = 24.684  # 24.712 (UVIS1)  # 24.684 (UVIS2; according to header it's UVIS2!)
    #  https://www.stsci.edu/hst/instrumentation/wfc3/data-analysis/photometric-calibration/uvis-photometric-calibration
    magi = zp_I - 2.5 * np.log10(idat * texp_I)  # image in counts/sec; texp = 805s, mag = zp - 2.5log10(total_counts)
    magi_cps = zp_I - 2.5 * np.log10(idat)
    # zp_H = 24.6949
    # https://www.stsci.edu/hst/instrumentation/wfc3/data-analysis/photometric-calibration/ir-photometric-calibration#se
    # ction-cc19dbfc-8f60-4870-8765-43810de39924
    magh = zp_H - 2.5 * np.log10(hdat * texp_H)  # image in counts/sec; texp = 898.467164s, mag = zp - 2.5log10(tot)
    magh_cps = zp_H - 2.5 * np.log10(hdat)

    # MAKE ZOOM-IN IMAGES
    magi_zoom = magi[yi:yf, xi:xf]
    magh_zoom = magh[yi:yf, xi:xf]
    magh_cps_zoom = magh_cps[yi:yf, xi:xf]
    magi_cps_zoom = magi_cps[yi:yf, xi:xf]
    vmax = 17.
    vmin = 9.  # -40

    # CREATE CONTOUR IMAGES
    img1, mod_img, levels, extentc, xylabel = display_mod(galfit_out, texp_H, sky)

    # MAKE COLOR IMG, EXCESS IMG
    ih_cps_zoom = magi_cps_zoom - magh_cps_zoom
    ih_zoom = magi_zoom - magh_zoom
    ih_excess = magi - magh - np.nanmedian(magi - magh)
    ih_excess_zoom = magi_zoom - magh_zoom - np.nanmedian(magi_zoom - magh_zoom)

    # DEFINE ZOOM-IN REFERENCE COORDINATES
    xgal_short = x_galcen - xi
    ygal_short = y_galcen - yi

    xref_short = xref - xi
    yref_short = yref - yi

    x_galint_short = x_galint - xi
    y_galint_short = y_galint - yi

    # ROTATE HST RA, DEC ARRAYS
    rah_rot, refpix_rot = rot(rah, hhdr['CRPIX1'], hhdr['CRPIX2'], angle=-116.755, reshape=True)
    rah_rot_nors, refpix_rot_nors = rot(rah, hhdr['CRPIX1'], hhdr['CRPIX2'], angle=-116.755, reshape=False)
    dech_rot, galint_rot = rot(dech, x_galint, y_galint, angle=-116.755, reshape=True)

    # ROTATE CONTOUR IMAGES!
    img1_rot, galint_rot = rot(img1, x_galint, y_galint, angle=-116.755, reshape=True)
    mod_img_rot, ref_rot = rot(mod_img, xref, yref, angle=-116.755, reshape=True)

    # ROTATE ZOOM-IN CONTOUR IMAGES
    img1zoom_rot, galint_short_rot = rot(img1[yi:yf, xi:xf], x_galint_short, y_galint_short, angle=-116.755,
                                         reshape=True)
    modzoom_rot, galint_short_rot = rot(mod_img[yi:yf, xi:xf], x_galint_short, y_galint_short, angle=-116.755,
                                        reshape=True)

    # ROTATE ZOOM-IN IMAGES
    ih_rot, galint_short_rot = rot(ih_cps_zoom, x_galint_short, y_galint_short, angle=-116.755, reshape=True)
    magi_rot, galshort_rot = rot(magi_zoom, xgal_short, ygal_short, angle=-116.755, reshape=True)
    magh_rot, refshort_rot = rot(magh_zoom, xref_short, yref_short, angle=-116.755, reshape=True)

    #plt.imshow(ih_rot, origin='lower', vmin=1., vmax=3.)
    #plt.plot(galint_short_rot[0], galint_short_rot[1], 'r*')  # Good enough
    #plt.show()
    #print(oop)
    print(galint_short_rot)

    # SET UP AXIS EXTENTS (FULL AXES)
    trueres = 0.1  # arcsec/deg (this is preserved in the rotation!)
    fullx_rot = np.zeros(shape=len(img1_rot[0]))  # img1, img1_rot are stored y, x
    fully_rot = np.zeros(shape=len(img1_rot))
    for i in range(len(fullx_rot)):  # negative bc RA increases to left
        fullx_rot[i] = -trueres * (i - galint_rot[0])
    for i in range(len(fully_rot)):
        fully_rot[i] = trueres * (i - galint_rot[1])

    # SET UP AXIS EXTENTS (SHORT AXES)
    # (rah_rot[600,600] - rah_rot[600, 601]) * 3600  # 2.777777777e-5 (deg) * 3600 (arcsec/deg) = 0.10000000001184617
    xax_refshortrot = np.zeros(shape=len(ih_rot[0]))  # Using HST reference pixel!
    xax_galshortrot = np.zeros(shape=len(ih_rot[0]))  # Using galaxy center from Cappellari's code!
    xax_galintshortrot = np.zeros(shape=len(ih_rot[0]))  # Using galaxy center from Cappellari's code, rounded to int!
    yax_refshortrot = np.zeros(shape=len(ih_rot))  # Using HST reference pixel!
    yax_galshortrot = np.zeros(shape=len(ih_rot))  # Using galaxy center from Cappellari's code!
    yax_galintshortrot = np.zeros(shape=len(ih_rot))  # Using galaxy center from Cappellari's code, rounded to int!
    for i in range(len(xax_refshortrot)):  # negative (-trueres) because RA increases to left
        xax_refshortrot[i] = -trueres * (i - refshort_rot[0])  # (arcsec/pix) * N_pix = arcsec
        xax_galshortrot[i] = -trueres * (i - galshort_rot[0])  # (arcsec/pix) * N_pix = arcsec
        xax_galintshortrot[i] = -trueres * (i - galint_short_rot[0])  # (arcsec/pix) * N_pix = arcsec
    for i in range(len(yax_refshortrot)):
        yax_refshortrot[i] = trueres * (i - refshort_rot[1])  # (arcsec/pix) * N_pix = arcsec
        yax_galshortrot[i] = trueres * (i - galshort_rot[1])  # (arcsec/pix) * N_pix = arcsec
        yax_galintshortrot[i] = trueres * (i - galint_short_rot[1])  # (arcsec/pix) * N_pix = arcsec

    '''  # I FIGURED IT OUT?!?!
    plt.imshow(ih_cps_zoom, origin='lower', vmin=1., vmax=3.)
    plt.plot(x_galint_short, y_galint_short, 'r*')
    plt.show()

    plt.imshow(ih_rot, origin='lower', vmin=1., vmax=3.)
    plt.plot(galint_short_rot[0], galint_short_rot[1], 'r*')  # Seems good enough
    plt.show()

    print(ih_rot.shape)  # 136, 137
    print(len(xax_galintshortrot), len(yax_galintshortrot))  # 136, 137

    plt.imshow(ih_rot, origin='lower', vmin=1., vmax=3., extent=[xax_galintshortrot[0], xax_galintshortrot[-1],
                                                                 yax_galintshortrot[0], yax_galintshortrot[-1]])
    plt.plot(0.,0., 'r*')  # THIS IS WRONG!!!!! IT'S OFFSET BY 2 PIXELS WTFFFFFFFFF
    plt.show()
    #print(oop)

    y1 = np.zeros(shape=len(ih_rot))  # Using galaxy center from Cappellari's code, rounded to int!
    x1 = np.zeros(shape=len(ih_rot[0]))  # Using galaxy center from Cappellari's code, rounded to int!
    for i in range(len(y1)):
        y1[i] = (i - galint_short_rot[1]) * 0.1
    for j in range(len(x1)):
        x1[j] = -(j - galint_short_rot[0]) * 0.1

    plt.imshow(ih_rot, origin='lower', vmin=1., vmax=3., extent=[x1[0], x1[-1], y1[0], y1[-1]])
    plt.plot(0.,0., 'r*')  # THIS IS WRONG!!!!! IT'S OFFSET BY 2 PIXELS WTFFFFFFFFF
    plt.show()

    # galint_short_rot = [67.32473545 68.83940949]
    print(xax_galintshortrot)  # with +2, sign flips btw indices 66 & 67 (entries 67 & 68); w/o +2: 68 & 69 (69 & 70)
    print(yax_galintshortrot)  # with -2, sign flips btw indices 69 & 70 (entries 70 & 71); w/o -2: 67 & 68 (68 & 69)
    print(oops)
    # '''

    #print(ih_rot.shape, len(xax_galintshortrot), len(yax_galintshortrot), 'hi')  # (136, 137) 137 136 hi # SOLVED IT!
    #ext_galintshortrot = [xax_galintshortrot[0], xax_galintshortrot[-1], yax_galintshortrot[0], yax_galintshortrot[-1]]
    #plt.imshow(ih_rot, origin='lower', vmin=1., vmax=3., extent=ext_galintshortrot)
    #plt.plot(0.,0., 'r*')
    #plt.show()

    # CUT DOWN EVEN SHORTER!
    leftcut = 45
    rightcut = -45
    botcut = 46
    topcut = -45
    xax_galintshortrot = xax_galintshortrot[botcut:topcut]  # len 46 right now
    xax_galshortrot = xax_galshortrot[botcut:topcut]  # len 46 right now
    xax_refshortrot = xax_refshortrot[botcut:topcut]  # len 46 right now
    yax_galintshortrot = yax_galintshortrot[leftcut:rightcut]  # len 46 right now
    yax_galshortrot = yax_galshortrot[leftcut:rightcut]  # len 46 right now
    yax_refshortrot = yax_refshortrot[leftcut:rightcut]  # len 46 right now

    # CUT DOWN THE IMAGES TO MATCH!
    magh_rot = magh_rot[leftcut:rightcut, botcut:topcut]
    magi_rot = magi_rot[leftcut:rightcut, botcut:topcut]
    ih_rot = ih_rot[leftcut:rightcut, botcut:topcut]
    #ih_rot = ih_rot[botcut:topcut, leftcut:rightcut]
    img1zoom_rot = img1zoom_rot[leftcut:rightcut, botcut:topcut]
    modzoom_rot = modzoom_rot[leftcut:rightcut, botcut:topcut]

    # SAVE EXTENTS, CHOOSE WHICH SHORT EXTENT TO USE
    ext_refshortrot = [xax_refshortrot[0], xax_refshortrot[-1], yax_refshortrot[0], yax_refshortrot[-1]]
    ext_galshortrot = [xax_galshortrot[0], xax_galshortrot[-1], yax_galshortrot[0], yax_galshortrot[-1]]
    ext_galintshortrot = [xax_galintshortrot[0], xax_galintshortrot[-1], yax_galintshortrot[0], yax_galintshortrot[-1]]
    fullext_rot = [fullx_rot[0], fullx_rot[-1], fully_rot[0], fully_rot[-1]]
    shortext_rot = ext_galshortrot  # ext_galintshortrot, ext_galshortrot, ext_refshortrot
    print(ext_galshortrot)
    print(ext_galintshortrot)
    zp_galint_short_rot = [galint_short_rot[0] - botcut, galint_short_rot[1] - leftcut]
    # print(oop)

    print(galint_short_rot, botcut, leftcut)

    #plt.imshow(ih_cps_zoom, origin='lower', vmin=1., vmax=3.)
    #plt.plot(x_galint_short, y_galint_short, 'r*')
    #plt.show()

    #print(ih_rot.shape, len(xax_galintshortrot), len(yax_galintshortrot), 'hi')  # (46, 46) 46 46 hi # SOLVED IT!
    #plt.imshow(ih_rot, origin='lower', vmin=1., vmax=3., extent=ext_galintshortrot)
    #plt.plot(0.,0., 'r*')
    #plt.show()
    #print(oops)

    #print(ih_rot.shape, ih_rot.shape[0], len(ih_rot), img1_rot.shape)
    #print(oop)

    # GENERATE ALMA CO EMISSION
    colset1 = {'cmap': 'Blues', 'contourcolor': 'k', 'beamcol': 'b', 'alpha': 1.}
    colset2 = {'cmap': 'viridis', 'contourcolor': 'w', 'beamcol': 'c', 'alpha': 0.7}
    colset = colset1
    alph = colset['alpha']
    bc = colset['beamcol']
    cmap = colset['cmap']  # 'viridis'
    concol = colset['contourcolor']  # 'k'  # 'w'
    d0, min0, max0, extent_a, cmap_0, ebeam = dust_and_co(parfile, beamloc=bl, incl_beam=incl_beam, bcolor='b',
                                                          xmatch=xmatch, ymatch=ymatch)
    # xmatch, ymatch: 111, 163 ;; 116, 160 ;; 115, 158 ;; 84, 210
    #plt.imshow(d0, origin='lower', vmax=max0, vmin=min0, extent=extent_a, cmap='Blues')  # use ALMA extent!
    #plt.plot(0,0, 'r*')
    #plt.show()
    #print(oop)

    # START SETTING UP FIGURE!
    fig = plt.figure(figsize=(14, 7))  # , constrained_layout=True)  # 2.316 (16.212, 7)
    # gs = gridspec.GridSpec(6, 3)  # 6cols, 3rows
    # gs = gridspec.GridSpec(1231, 2851)  # 2851=1231+1620, 1231/2=615.5, 1620+615.5=2235.5
    # 2001rows, 1728cols, 1728/2=864. 2001, 1728*2=3456
    nrow = 2001
    ncol = int(1828+2001)  # ((1628+2001)/2001 = 1.8135932 ~ 9:5 + room for labels ~10:5 = 2:1)
    gs = gridspec.GridSpec(nrows=nrow, ncols=ncol)

    # DEFINE AXES: ax0 = large left panel, ax1 = top center, ax2 = top right, ax3 = bottom right, ax4 = bottom center
    ax0 = plt.subplot(gs[0:nrow, 0:int(ncol/2)])  # 1620/1231 = 1.316
    ax1 = plt.subplot(gs[0:int(nrow/2), int(ncol/2):int(3*ncol/4)])
    ax2 = plt.subplot(gs[0:int(nrow/2), int(3*ncol/4):ncol])
    ax3 = plt.subplot(gs[int(nrow/2):nrow, int(3*ncol/4):ncol])
    ax4 = plt.subplot(gs[int(nrow/2):nrow, int(ncol/2):int(3*ncol/4)])

    # SET UP ax0 (large, left-hand panel)
    mge_lab = r'\textbf{Dust-masked MGE}'  # r'\textbf{Model}'
    back = np.zeros(shape=img1_rot.shape) + 100.  # Set image background = "0" (100 bc in mag units)
    im0 = ax0.imshow(back, origin='lower', vmax=vmax, vmin=vmin, extent=fullext_rot, cmap='Greys_r')  # set background
    im0d = ax0.contour(img1_rot, levels, colors='k', linestyles='solid', extent=fullext_rot)  # plot data contours
    im0m = ax0.contour(mod_img_rot, levels, colors='r', linestyles='solid', extent=fullext_rot)  # plot model contours
    ax0.set_aspect('equal', adjustable='datalim')  # set axes aspect equal
    h1, _ = im0d.legend_elements()
    h2, _ = im0m.legend_elements()
    ax0.legend([h1[0], h2[0]], [r'\textbf{\textit{HST H}}', mge_lab], loc='upper left')  # set legend!

    # PLOT COMPASS IN ax0 (North is up, East is to the left)
    x0 = -70.  # 55.  # x origin of compass
    y0 = 50.  # y origin of compass
    ax0.plot([x0, x0], [y0, y0+15.], 'k-', linewidth=3)  # [x1,x2], [y1,y2]
    ax0.plot([x0, x0+15.], [y0, y0], 'k-', linewidth=3)  # [x1,x2], [y1,y2]
    ax0.text(x0+3, y0+17, r'\textbf{N}', color='k')
    ax0.text(x0+22, y0-3, r'\textbf{E}', color='k')

    # PLOT ZOOM-IN H, I images
    im1 = ax1.imshow(magh_rot, origin='lower', vmax=np.amax(magh_rot), vmin=np.amin(magh_rot), cmap='Greys_r',
                     extent=shortext_rot)
    im2 = ax2.imshow(magi_rot, origin='lower', vmax=np.amax(magi_rot), vmin=np.amin(magi_rot), cmap='Greys_r',
                     extent=shortext_rot)

    # PLOT IMAGE SCALEBAR ON ax2 (I-band image, rop R panel)
    ax2.plot([-0.8, -1.93327289], [-2.1, -2.1], 'k-')  # 500 pc / 441.2 pc_per_arcsec = 1.13327289 arcsec
    ax2.text(-0.85, -2., r'500 pc')
    # dsk = 1.6  # disk diameter, arcsec
    # ax2.plot([.66, .66-dsk*np.cos(np.deg2rad(18.6))], [-0.3, -0.3+dsk*np.sin(np.deg2rad(18.6))], 'k-')  # [x1, x2], [y1, y2]
    # PA = 18.6 deg, x2 = x1-1.6cos(18.6deg)


    #print(rah_rot[1120, 992], dech_rot[1120, 992])  # 50.51284123408508 40.8639595629505 [CORRECT ROTATED GALAXY CENTER]
    #print(rah[y_galint, x_galint], dech[y_galint, x_galint])  # 50.51284020887928 40.86395031448354 (shift: 0.004, 0.03)
    #print(ras[126], decs[15])  # 50.51204305553889 40.863144444444444
    #print(ras[0], ras[299])  # 50.512743055538884 50.51108194442777
    #print(oop)
    # rah_rot[1120, 992] = 50.51284123408508 ;; dech_rot[1120, 992] = 40.8639595629505
    # xmatch, ymatch should be ALMA full-cube pixel values corresponding to the RA, DEC used as 0pt for the HST extent


    #d0, min0, max0, extent_a, cmap_0, ebeam = dust_and_co(parfile, beamloc=(1.85,-2.), incl_beam=incl_beam, bcolor=bc,
    #                                                      xmatch=111, ymatch=163)  # 127, 151
    # 158, 134
    # plt.gca().set_aspect('equal', adjustable='box')
    #print(galint_rot)
    #print(rah_rot[1120, 992], dech_rot[1120, 992])  # 50.51284123408508 40.8639595629505
    #print(ras[111], decs[163])                      # 50.51212638887222 40.86396666666667

    #print(oop)

    # CRVAL1 = 5.051190416665E+01 (RA; value at CRPIX1, CUNIT1 = deg)
    # CRPIX1 = 1.510000000000E+02 (index runs from 1 to len(axis)) == pixel 150 in python
    # CRVAL2 = 4.086390000000E+01 (DEC; value at CRPIX2, CUNIT2 = deg)
    # CRPIX2 = 1.510000000000E+02 (index runs from 1 to len(axis)) == pixel 150 in python

    # PLOT ZOOM-IN DUST CONTOURS OVERLAID ON ALMA CO EMISSION
    if incl_beam:  # plot ALMA beam
        ax3.add_patch(ebeam)

    # ALMA SUBCUBE is (x,y) = (84:168, 118:182), so the FULL CUBE midpoint (150, 150) is at subcube coordinates (66, 32)
    # The SUBCUBE midpoint is at full-cube pixels: (126, 150), i.e. subcube pixels: (42, 32)

    fullarr = np.zeros(shape=ih_rot.shape)  # fill the background of ax3 (fill with zeros)
    #ax3.imshow(fullarr, origin='lower', extent=shortext_rot, cmap=cmap)  # plot background

    # PLOT CO EMISSION (moment 0 map)
    im3 = ax3.imshow(d0, origin='lower', vmax=max0, vmin=min0, extent=extent_a, cmap=cmap)  # use ALMA extent!

    # PLOT DUST CONTOURS (I - H)
    # lvs = np.linspace(0.15, 2.5, 25)  # dust contour levels
    # lvs = np.linspace(1.7, 2.5, 10)  # dust contour levels
    lvs = np.linspace(1.87, 3., 10)  # dust contour levels
    ax3.contour(ih_rot, levels=lvs, colors=concol, linewidths=.75, alpha=alph, extent=shortext_rot)  #
    # PLOT ZOOM-IN HST H-BAND IMAGE CONTOURS
    ax4.axis('equal')  # set axis equal
    # ax4.set_xlabel(xylabel)
    # ax4.set_ylabel(xylabel)
    # NOT USING PLACEHOLDER NOW?!
    placehold = np.zeros(shape=img1zoom_rot.shape) + 100  # set image background = "0" (100 bc in mag units)
    im1 = ax4.imshow(placehold, origin='lower', vmax=np.amax(magh), vmin=np.amin(magh), cmap='Greys',
                     extent=shortext_rot)
    # plot zoom-in data & model contours
    ctd = ax4.contour(img1zoom_rot, levels, colors='k', linestyles='solid', extent=shortext_rot)
    ctm = ax4.contour(modzoom_rot, levels, colors='r', linestyles='solid', extent=shortext_rot)
    h1c, _ = ctd.legend_elements()  # add same style legend as in ax0
    h2c, _ = ctm.legend_elements()  # add same style legend as in ax0
    ax4.legend([h1c[0], h2c[0]], [r'\textbf{\textit{HST H}}', mge_lab], loc='upper left')  # set legend!

    # AXIS LABELS
    ax0.set_xlabel(r'$\Delta$ RA [arcsec]')
    ax3.set_xlabel(r'$\Delta$ RA [arcsec]')
    ax4.set_xlabel(r'$\Delta$ RA [arcsec]')
    ax0.set_ylabel(r'$\Delta$ DEC [arcsec]')
    ax2.set_ylabel(r'$\Delta$ DEC [arcsec]')
    ax3.set_ylabel(r'$\Delta$ DEC [arcsec]')
    #ax3.minorticks_on()
    #ax4.minorticks_on()
    ax3.set_xticks([-2, -1, 0, 1, 2])
    ax3.set_xlim([shortext_rot[0], shortext_rot[1]])
    #print(extent_galshortrot)  # [2.365810689185513, -2.1341893113475647, -2.133785856482058, 2.3662141440510194]
    #print(xax_galshortrot)
    #print(oop)
    ax4.set_xticks([-2, -1, 0, 1, 2])
    ax4.set_xlim([shortext_rot[0], shortext_rot[1]])

    # IMAGE LABEL TEXT
    ax1.text(hx, hy, r'\textbf{\textit{HST H}}', color='k')
    ax2.text(hx, hy, r'\textbf{\textit{HST I}}', color='k')
    ax3.text(hx, hy, r'\textbf{ALMA CO($2 - 1$)}', color=bc)  # 'c'
    ax3.text(-0.4, ihy, r'\textbf{\textit{HST I -- H}}', color=concol, alpha=alph)
    # ax4.text(1.5, -1.95, r'\textbf{\textit{HST H}}', color='k')
    # ax4.text(2., 1.8, r'\textbf{\textit{HST H}}', color='k')

    ax1.xaxis.set_visible(False)  # ax1 no axes visible
    ax1.yaxis.set_visible(False)  # ax1 no axes visible
    ax2.xaxis.set_visible(False)  # ax2 y visible, x not visible
    ax4.yaxis.set_visible(False)  # ax4 x visible, y not visible
    ax2.yaxis.set_label_position("right")  # put yaxis label on right-hand side of panel
    ax3.yaxis.set_label_position("right")  # put yaxis label on right-hand side of panel
    ax2.yaxis.tick_right()  # put yaxis ticks on right-hand side of panel
    ax3.yaxis.tick_right()  # put yaxis ticks on right-hand side of panel

    #ax1.plot(0,0, 'r*')
    #ax2.plot(0,0, 'r*')
    #ax3.plot(0,0, 'r*')
    #ax4.plot(0,0, 'r*')

    plt.subplots_adjust(hspace=0.0)
    plt.subplots_adjust(wspace=0.0)

    plt.gca().set_aspect('equal', adjustable='box')

    plt.show()


if __name__ == '__main__':

    direc = '/Users/jonathancohn/Documents/dyn_mod/'
    gf = direc + 'galfit_u2698/'
    u2698 = direc + 'ugc_2698/'

    make_fig1(galfit_out=gf + 'galfit.121', parfile=u2698 + 'ugc_2698_finaltests_fiducial_out.txt')
