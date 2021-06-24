from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from scipy.ndimage import interpolation
import scipy.optimize as opt
from photutils.centroids import centroid_com, centroid_2dg  # centroid_quadratic,

def rot(image, x0, y0, angle=0., reshape=True):
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


# 2D GAUSSIAN!
def twoD_Gaussian(coords, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    x, y = coords
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)))

    return g.ravel()


def calc_noise(wht_data, sci_data, pxfs, x0, y0, xlen, ylen, nrow, ncol, amp=None, xstar=None, ystar=None, vmax=700.):
    import scipy.optimize as opt

    normwt = []
    for wd in range(len(wht_data)):
        unc_reg = wht_data[wd][y0-ylen:y0+ylen, x0-xlen:x0+xlen]
        noise = np.std(unc_reg, ddof=1)
        print(noise, noise / np.median(unc_reg))
        normwt.append(noise / np.median(unc_reg))

    #plt.imshow(wht_data, origin='lower')
    #plt.colorbar()
    #plt.plot(x0, y0, 'r*')
    #plt.show()

    if len(wht_data) == 1:
        plt.imshow(sci_data[0][ystar-15:ystar+15, xstar-15:xstar+15], origin='lower', vmax=vmax, vmin=0.)  # vmax=20.
        # 1934, 1118 --> need y, x -> 1118, 1934
        # amp = 37 for PGC 11179, 700 for NGC 384
        initial_guess = (amp, 15, 15, 2, 2, 0., 0.)  # amplitude, x0, y0, x_fwhm, y_fwhm, theta in radians, offset

        x = np.linspace(0, 30, 30)
        y = np.linspace(0, 30, 30)
        x, y = np.meshgrid(x, y)

        popt, pcov = opt.curve_fit(twoD_Gaussian, (x, y), sci_data[0][ystar-15:ystar+15, xstar-15:xstar+15].ravel(),
                                   p0=initial_guess)
        fwhms = [popt[3], popt[4]]
        print(popt)
        datafit = twoD_Gaussian((x,y), *popt)
        plt.contour(x,y, datafit.reshape(30,30), 4, colors='w')

        plt.text(10, 25, 'pxf' + pxfs[0], color='w')
        plt.text(1, 20, 'rms_dev/median=' + str(round(normwt[0],4)), color='w')
        plt.text(1, 5, 'xfwhm=' + str(round(fwhms[0],3)) + ', yfwhm=' + str(round(fwhms[1],3)), color='w')
        plt.show()

    else:
        # amp = 37 for PGC 11179, 700 for NGC 384
        initial_guess = (amp, 15, 15, 2, 2, 0., 0.)  # amplitude, x0, y0, x_fwhm, y_fwhm, theta in radians, offset

        x = np.linspace(0, 30, 30)
        y = np.linspace(0, 30, 30)
        x, y = np.meshgrid(x, y)

        fig, ax = plt.subplots(nrow, ncol, sharex=True, sharey=True, figsize=(8,8.25))
        plt.subplots_adjust(hspace=0., wspace=0.)
        for sd in range(len(sci_data)):
            if sd < ncol:
                row = 0
                col = sd
            else:
                row = 1
                col = sd - ncol
            ax[row, col].imshow(sci_data[sd][ystar - 15:ystar + 15, xstar - 15:xstar + 15], origin='lower', vmax=vmax,
                                vmin=0.)  # vmax = 20.
            # 1934, 1118 --> need y, x -> 1118, 1934
            # 1282:1312, 1153:1183

            popt, pcov = opt.curve_fit(twoD_Gaussian, (x, y),
                                       sci_data[sd][ystar-15:ystar+15, xstar-15:xstar+15].ravel(), p0=initial_guess)
            # 1282:1312, 1153:1183
            fwhms = [popt[3], popt[4]]
            print(popt)
            datafit = twoD_Gaussian((x,y), *popt)
            ax[row,col].contour(x,y, datafit.reshape(30,30), 4, colors='w')

            ax[row, col].text(10, 25, 'pxf' + pxfs[sd], color='w')
            ax[row, col].text(1, 20, 'rms_dev/median=' + str(round(normwt[sd],4)), color='w')
            ax[row, col].text(1, 5, 'xfwhm=' + str(round(fwhms[0],3)) + ', yfwhm=' + str(round(fwhms[1],3)), color='w')
        # plt.imshow(sci_data[1282:1312, 1153:1183], origin='lower', vmax=37.)
        #plt.colorbar()
        plt.show()


def find_center(idata, hdata):
    # '''  #
    '''  #
    # CENTROIDS: H-band:
    # h_data[np.abs(h_data) > 100] = 0.
    plt.imshow(h_data, origin='lower', vmax=4.5e5, vmin=0.)
    plt.colorbar()
    plt.plot(1390+76, 1325+73, 'r*')  # PGC 11179
    #plt.plot(1375+82, 1290+77, 'r*')  # from idx  # NGC 384
    #plt.plot(1375+82.034056318563543, 1290+76.753908662648286, 'c*')  # from centroid  # NGC 384
    plt.show()
    plt.imshow(i_data, origin='lower', vmax=100., vmin=0.)
    plt.colorbar()
    #plt.plot(1510+71, 1575+72, 'r*')  # from idx
    #plt.plot(1510+71.078015258901999, 1575+72.940969820368124, 'c*')  # from centroid
    plt.show()
    print(oop)
    #  '''
    # '''  #

    idata = np.nan_to_num(idata)

    #plt.imshow(idata, origin='lower', vmax=130)
    #plt.colorbar()
    #plt.plot(1927, 1922, 'w*')  # max
    #plt.plot(1875+52.13975768026607, 1875+47.180317370401404, 'c+')  # centroid matches max pretty well!!
    #plt.plot(1875+52.6664093, 1875+47.6567030, 'r*')  # 2D gaussian (not great)
    #plt.show()
    #print(oop)

    idata_small = idata[1875:1975, 1875:1975]

    #initial_guess = (59, 1582, 1648, 1.7, 2.1, 1.4, 5.)  # amplitude = 88.8/1.5, x0, y0 = 1582, 1648, x_fwhm~4 -> sigma~4/2.355=1.7,
    initial_guess = (130., 50, 50, 5., 5., 0.87, 0.3)  # amplitude = 88.8/1.5, x0, y0 = 1582, 1648, x_fwhm~4 -> sigma~4/2.355=1.7,
    # y_fwhm~5 -> sigma~5/2.355=2.1, theta in radians ~80 deg -> 1.4 rads; offset ~ 0?

    x = np.linspace(0, len(idata_small), len(idata_small))
    y = np.linspace(0, len(idata_small[0]), len(idata_small[0]))
    x, y = np.meshgrid(x, y)

    gg = twoD_Gaussian((x, y), 130., 50, 50, 5., 5., 0.87, 0.3)
    print(gg.shape)
    print(idata.shape)

    import scipy.optimize as opt
    popt, pcov = opt.curve_fit(twoD_Gaussian, (x, y), idata_small.ravel(), p0=initial_guess)
    print(popt)
    # [6.63812507e+01 5.26664093e+01 4.76567030e+01 6.05016694e+00 4.46070405e+00 6.05847758e+04 2.91658646e+00]
    print('above here is 2D gauss result!')
    #print(oop)

    # '''  #
    # THESE ARE FOR NGC 384
    hi,hf = 1375,1525 # 950,1800  # 900,1850
    hyi,hyf = 1290,1440  # hi,hf
    iy0,iy1 = 1875,1975  # pre-rot n384
    i0,i1 = 1875,1975  # pre-rot n384
    # hi,hf = 1390,1540  # pre-rot for p11179
    # hyi,hyf = 1325,1475  # pre-rot for p11179
    #hi, hf = 1405, 1555  # H-band rot p11179
    #hyi, hyf = 1625, 1775  # H-band rot p11179
    #i0, i1 = 1510, 1660  # 1000,2000  # 1200,2000
    #iy0, iy1 = 1575, 1725  # i0,i1
    hdata = hdata[hyi:hyf, hi:hf]
    idata = idata[iy0:iy1, i0:i1]
    hidx = np.unravel_index(np.argmax(hdata), hdata.shape)
    iidx = np.unravel_index(np.argmax(idata), idata.shape)
    print(hidx, iidx)  # ((78, 78), (72, 71))  # rotated H-band p11179!
    # pre-rot p11179 H-band: ((73, 76), (72, 71))
    # FOR NGC 384: (77, 82) 82 in x, 77 in y
    # plt.imshow(h_data, origin='lower', vmax=7e5, vmin=0.)
    # plt.colorbar()
    # plt.show()
    # print(oop)
    xh, yh = centroid_2dg(hdata, mask=np.isnan(hdata))
    print(xh, yh)  # Using 1290,1440 & 1375,1525: (82.034056318563543, 76.753908662648286)
    # using 900,1850: (556.0822735162559, 466.70973014659847)
    # Using 950,1800: (507.21197319028778, 416.78153137759034)  # better, still imperfect
    xi, yi = centroid_2dg(idata, mask=np.isnan(idata))
    print(xi, yi)  # Using 1575,1725 & 1510,1660: (72.940969820368124, 71.078015258901999)
    print(oop)
    # using 1200,2000: (383.27133931976698, 445.74008432184183)  # better, still imperfect
    # Using 1000,2000: (583.33511514826932, 645.7459270168938)
    # from scipy.ndimage import measurements
    # h_data[np.isnan(h_data)] = 0.
    # i_data[np.isnan(i_data)] = 0.
    # xh1,yh1 = measurements.center_of_mass(h_data)
    # print(xh1, yh1)  # (469.7267122214306, 550.71549453408932)  # worse, even accounting for flipping x,y!
    # xi1,yi1 = measurements.center_of_mass(i_data)
    # print(xi1, yi1)  # (420.9653334337014, 390.56088692909975)  # worse, even accounting for flipping x,y!
    plt.imshow(hdata, origin='lower', vmax=4.5e5, vmin=0.)
    plt.colorbar()
    plt.plot(xh, yh, 'r*')
    plt.plot(hidx[1], hidx[0], 'c*')
    # plt.plot(yh1, xh1, 'c*')
    plt.show()
    plt.imshow(idata, origin='lower', vmax=100., vmin=0.)
    plt.colorbar()
    plt.plot(xi, yi, 'r*')
    plt.plot(iidx[1], iidx[0], 'c*')
    # plt.plot(yi1, xi1, 'c*')
    plt.show()
    print(oop)


g2698 = '/Users/jonathancohn/Documents/dyn_mod/galfit/u2698/'
gf = '/Users/jonathancohn/Documents/dyn_mod/galfit/'
fj = '/Users/jonathancohn/Documents/dyn_mod/for_jonathan/'
hst_p = '/Users/jonathancohn/Documents/hst_pgc11179/'
hst_n = '/Users/jonathancohn/Documents/hst_ngc384/'
n384_h = fj + 'NGC0384_F160W_drz_sci.fits'
n384_adj_h = fj + 'NGC0384_F160W_drz_sci_adjusted.fits'
n384_adj_h_n7 = fj + 'NGC0384_F160W_drz_sci_adjusted_n7.fits'
n384_adj_h_inclsky = fj + 'NGC0384_F160W_drz_sci_adjusted_inclsky.fits'
n384_hmask = fj + 'NGC0384_F160W_drz_mask.fits'
n384_hmask_extended = fj + 'NGC0384_F160W_drz_mask_extended.fits'
n384_f814w_pixscale03 = fj + 'NGC0384_F814W_drc_sci.fits'
n384_f814w_pxf001_ps03_wht = hst_n + 'n384_f814w_drizflc_pxf001_003_wht.fits'
n384_f814w_pxf001_wht = hst_n + 'n384_f814w_drizflc_pxf001_006_wht.fits'
n384_f814w_pxf01_wht = hst_n + 'n384_f814w_drizflc_pxf01_006_wht.fits'
n384_f814w_pxf04_wht = hst_n + 'n384_f814w_drizflc_pxf04_006_wht.fits'
n384_f814w_pxf06_wht = hst_n + 'n384_f814w_drizflc_pxf06_006_wht.fits'
n384_f814w_pxf08_wht = hst_n + 'n384_f814w_drizflc_pxf08_006_wht.fits'
n384_f814w_pxf1_wht = hst_n + 'n384_f814w_drizflc_pxf1_006_wht.fits'
n384_f814w_pxf001_ps03 = hst_n + 'n384_f814w_drizflc_pxf001_003_sci.fits'
n384_f814w_pxf001 = hst_n + 'n384_f814w_drizflc_pxf001_006_sci.fits'
n384_f814w_pxf01 = hst_n + 'n384_f814w_drizflc_pxf01_006_sci.fits'
n384_f814w_pxf04 = hst_n + 'n384_f814w_drizflc_pxf04_006_sci.fits'
n384_f814w_pxf06 = hst_n + 'n384_f814w_drizflc_pxf06_006_sci.fits'
n384_f814w_pxf08 = hst_n + 'n384_f814w_drizflc_pxf08_006_sci.fits'
n384_f814w_pxf1 = hst_n + 'n384_f814w_drizflc_pxf1_006_sci.fits'
n384_psf_f814w = hst_n + 'n384_f814w00.fits'
p11179_h = fj + 'PGC11179_F160W_drz_sci.fits'
p11179_adj_h = fj + 'PGC11179_F160W_drz_sci_adjusted.fits'
p11179_adj_h_n7 = fj + 'PGC11179_F160W_drz_sci_adjusted_n7.fits'
p11179_adj_h_inclsky = fj + 'PGC11179_F160W_drz_sci_adjusted_inclsky.fits'
p11179_f814w_pxf001_wht = hst_p + 'p11179_f814w_drizflc_pxf001_006_wht.fits'
p11179_f814w_pxf01_wht = hst_p + 'p11179_f814w_drizflc_pxf01_006_wht.fits'
p11179_f814w_pxf02_wht = hst_p + 'p11179_f814w_drizflc_pxf02_006_wht.fits'
p11179_f814w_pxf04_wht = hst_p + 'p11179_f814w_drizflc_pxf04_006_wht.fits'
p11179_f814w_pxf06_wht = hst_p + 'p11179_f814w_drizflc_pxf06_006_wht.fits'
p11179_f814w_pxf08_wht = hst_p + 'p11179_f814w_drizflc_006_wht.fits'  # pxf08!
p11179_f814w_pxf09_wht = hst_p + 'p11179_f814w_drizflc_pxf09_006_wht.fits'
p11179_f814w_pxf1_wht = hst_p + 'p11179_f814w_drizflc_pxf1_006_wht.fits'
p11179_f814w_pxf001 = hst_p + 'p11179_f814w_drizflc_pxf001_006_sci.fits'
p11179_f814w_pxf01 = hst_p + 'p11179_f814w_drizflc_pxf01_006_sci.fits'
p11179_f814w_pxf02 = hst_p + 'p11179_f814w_drizflc_pxf02_006_sci.fits'
p11179_f814w_pxf04 = hst_p + 'p11179_f814w_drizflc_pxf04_006_sci.fits'
p11179_f814w_pxf06 = hst_p + 'p11179_f814w_drizflc_pxf06_006_sci.fits'
p11179_f814w_pxf08 = hst_p + 'p11179_f814w_drizflc_006_sci.fits'  # pxf08!
p11179_f814w_pxf09 = hst_p + 'p11179_f814w_drizflc_pxf09_006_sci.fits'
p11179_f814w_pxf1 = hst_p + 'p11179_f814w_drizflc_pxf1_006_sci.fits'
p11179_psf_f814w = hst_p + 'psf_f814w00.fits'
p11179_psf534_f814w = hst_p + 'psf_f814w_534diameter_00.fits'
# PSFs
p11179_f814w_psfm = hst_p + 'psf_ic0b14rmq_flc.fits'
p11179_f814w_psfx = hst_p + 'psf_ic0b14rxq_flc.fits'
p11179_f814w_psfz = hst_p + 'psf_ic0b14rzq_flc.fits'
n384_f814w_psf5 = hst_n + 'psf_ic0b09v5q_flc.fits'
n384_f814w_psfl = hst_n + 'psf_ic0b09vlq_flc.fits'
n384_f814w_psfo = hst_n + 'psf_ic0b09voq_flc.fits'

zp_H = 24.662  # https://www.stsci.edu/hst/instrumentation/wfc3/data-analysis/photometric-calibration/ir-photometric-calibration
zp_I = 24.684  # UVIS2 according to fits header; https://www.stsci.edu/hst/instrumentation/wfc3/data-analysis/photometric-calibration/uvis-photometric-calibration

with fits.open(n384_adj_h) as hdu:
    nh_hdr = hdu[0].header
    nh_data = hdu[0].data

with fits.open(n384_f814w_pixscale03) as hdu:
    ni_hdr = hdu[0].header
    ni_data = hdu[0].data

with fits.open(n384_f814w_pxf1_wht) as hdu:
    n_whdr_pxf1 = hdu[0].header
    n_wht_pxf1 = hdu[0].data

with fits.open(n384_f814w_pxf08_wht) as hdu:
    n_whdr_pxf8 = hdu[0].header
    n_wht_pxf8 = hdu[0].data

with fits.open(n384_f814w_pxf06_wht) as hdu:
    n_whdr_pxf6 = hdu[0].header
    n_wht_pxf6 = hdu[0].data

with fits.open(n384_f814w_pxf04_wht) as hdu:
    n_whdr_pxf4 = hdu[0].header
    n_wht_pxf4 = hdu[0].data

with fits.open(n384_f814w_pxf01_wht) as hdu:
    n_whdr_pxf01 = hdu[0].header
    n_wht_pxf01 = hdu[0].data

with fits.open(n384_f814w_pxf001_wht) as hdu:
    n_whdr_pxf001 = hdu[0].header
    n_wht_pxf001 = hdu[0].data

with fits.open(n384_f814w_pxf001_ps03_wht) as hdu:
    n_whdr_pxf001_ps03 = hdu[0].header
    n_wht_pxf001_ps03 = hdu[0].data

with fits.open(n384_f814w_pxf1) as hdu:
    ni_hdr_pxf1 = hdu[0].header
    ni_data_pxf1 = hdu[0].data

with fits.open(n384_f814w_pxf08) as hdu:
    ni_hdr_pxf08 = hdu[0].header
    ni_data_pxf08 = hdu[0].data

with fits.open(n384_f814w_pxf06) as hdu:
    ni_hdr_pxf06 = hdu[0].header
    ni_data_pxf06 = hdu[0].data

with fits.open(n384_f814w_pxf04) as hdu:
    ni_hdr_pxf04 = hdu[0].header
    ni_data_pxf04 = hdu[0].data

with fits.open(n384_f814w_pxf01) as hdu:
    ni_hdr_pxf01 = hdu[0].header
    ni_data_pxf01 = hdu[0].data

with fits.open(n384_f814w_pxf001) as hdu:
    ni_hdr_pxf001 = hdu[0].header
    ni_data_pxf001 = hdu[0].data

with fits.open(n384_f814w_pxf001_ps03) as hdu:
    ni_hdr_pxf001_ps03 = hdu[0].header
    ni_data_pxf001_ps03 = hdu[0].data

with fits.open(n384_psf_f814w) as hdu:
    nhdr_psfi = hdu[0].header
    ndat_psfi = hdu[0].data

with fits.open(p11179_h) as hdu:
    h_hdr = hdu[0].header
    h_data = hdu[0].data

with fits.open(p11179_f814w_pxf001) as hdu:
    i_hdr_pxf01 = hdu[0].header
    i_data_pxf01 = hdu[0].data

with fits.open(p11179_f814w_pxf01) as hdu:
    i_hdr_pxf1 = hdu[0].header
    i_data_pxf1 = hdu[0].data

with fits.open(p11179_f814w_pxf02) as hdu:
    i_hdr_pxf2 = hdu[0].header
    i_data_pxf2 = hdu[0].data

with fits.open(p11179_f814w_pxf04) as hdu:
    i_hdr_pxf4 = hdu[0].header
    i_data_pxf4 = hdu[0].data

with fits.open(p11179_f814w_pxf06) as hdu:
    i_hdr_pxf6 = hdu[0].header
    i_data_pxf6 = hdu[0].data

with fits.open(p11179_f814w_pxf08) as hdu:
    i_hdr_pxf8 = hdu[0].header
    i_data_pxf8 = hdu[0].data

with fits.open(p11179_f814w_pxf09) as hdu:
    i_hdr_pxf9 = hdu[0].header
    i_data_pxf9 = hdu[0].data

with fits.open(p11179_f814w_pxf1) as hdu:
    i_hdr_pxf10 = hdu[0].header
    i_data_pxf10 = hdu[0].data

with fits.open(p11179_f814w_pxf001_wht) as hdu:
    whdr_pxf01 = hdu[0].header
    wht_pxf01 = hdu[0].data

with fits.open(p11179_f814w_pxf01_wht) as hdu:
    whdr_pxf1 = hdu[0].header
    wht_pxf1 = hdu[0].data

with fits.open(p11179_f814w_pxf02_wht) as hdu:
    whdr_pxf2 = hdu[0].header
    wht_pxf2 = hdu[0].data

with fits.open(p11179_f814w_pxf04_wht) as hdu:
    whdr_pxf4 = hdu[0].header
    wht_pxf4 = hdu[0].data

with fits.open(p11179_f814w_pxf06_wht) as hdu:
    whdr_pxf6 = hdu[0].header
    wht_pxf6 = hdu[0].data

with fits.open(p11179_f814w_pxf08_wht) as hdu:
    whdr_pxf8 = hdu[0].header
    wht_pxf8 = hdu[0].data

with fits.open(p11179_f814w_pxf09_wht) as hdu:
    whdr_pxf9 = hdu[0].header
    wht_pxf9 = hdu[0].data

with fits.open(p11179_f814w_pxf1_wht) as hdu:
    whdr_pxf10 = hdu[0].header
    wht_pxf10 = hdu[0].data

with fits.open(p11179_psf_f814w) as hdu:
    hdr_psfi = hdu[0].header
    dat_psfi = hdu[0].data

with fits.open(p11179_psf534_f814w) as hdu:
    hdr_psfi534 = hdu[0].header
    dat_psfi534 = hdu[0].data

#find_center(ni_data_pxf08, nh_data)
#print(oop)
#nxctr = 1927
#nyctr = 1922
#plt.imshow(ndat_psfi, origin='lower')
#plt.colorbar()
#plt.show()
#print(oop)


psf_0 = np.zeros(shape=i_data_pxf8.shape)  # rmq -> pattstep = 0, no pattern, no dither. Just center it over galaxy!
psf_1 = np.zeros(shape=i_data_pxf8.shape)  # rxq -> pattstep = 1, pattern1=LINE (0,0) [must be 0,0, since step=2 exists]
psf_2 = np.zeros(shape=i_data_pxf8.shape)  # rzq -> pattstep = 2, pattern1=LINE (2.5, 2.5)
# IF it should be LINE-3PT: (0,0) , (2.33,2.33), (4.67,4.67)
# For xc0, yc0: POS TARG = -1.7952, -1.9130 [unit = arcsec]! (https://www.stsci.edu/itt/APT_help20/WFC3/appendixC3.html)
## For UVIS: POS TARG X ~ 0.0396 ["/pix] * x [pix]; POS TARG Y ~ 0.0027 ["/pix] * x [pix] + 0.0395 ["/pix] * y [pix]
## -> x [pix] = POS TARG X / 0.0396; y [pix] = (POS TARG Y - 0.0027 * x) / 0.0395
#  Also clarified here: http://guaix.fis.ucm.es/~agpaz/Instrumentacion_Espacio_2010/Espacio_Docs/HST/hst_c15_phaseII.pdf
# SAME FOR BOTH GALAXIES!! Just Change xctr, yctr
pxctr = 1581
pyctr = 1647
nxctr = 1927
nyctr = 1922
xctr,yctr = nxctr,nyctr  # pxctr,pyctr  # pxctr,pyctr ;; nxctr,nyctr
xc0 = xctr+(-1.7952 / 0.0396)  # = -45.3333333
xc1 = xctr+0
xc2 = xctr+2.5
yc0 = yctr+((-1.9130 - 0.0027 * (-1.7952 / 0.0396)) / 0.0395)  # = -45.3316456
yc1 = yctr+0
yc2 = yctr+2.5

xorig = np.zeros(shape=len(dat_psfi))
yorig = np.zeros(shape=len(dat_psfi[0]))
#origres = 0.02358  # arcsec/pix; from Tiny Tim: "Using undistorted critical sampling pixel size of 0.02358 arcsec"
print(dat_psfi.shape)  # 89, 89. I used 3.0 arcsec for pixel size in Tiny Tim! So: 0.0337078652 arcsec/pix
# WANT: 0.06"/pix. 89 pix * 0.06 arcsec/pix = 5.34 arcsec. Try in Tiny Tim!
print(dat_psfi534.shape)  # 159, 159. 5.34" size -> 0.0335849057 arcsec/pix. About the same as before!
origres = 0.0337078652
for i in range(len(xorig)):
    xorig[i] = origres * (i - len(dat_psfi) / 2.)
for i in range(len(yorig)):
    yorig[i] = origres * (i - len(dat_psfi[0]) / 2.)

x0 = np.zeros(shape=len(psf_0))
x1 = np.zeros(shape=len(psf_1))
x2 = np.zeros(shape=len(psf_2))
y0 = np.zeros(shape=len(psf_0[0]))
y1 = np.zeros(shape=len(psf_1[0]))
y2 = np.zeros(shape=len(psf_2[0]))
targetres = 0.06  # arcsec/pix; same as i_data res!

for i in range(len(x0)):
    x0[i] = targetres * (i - xc0)
    x1[i] = targetres * (i - xc1)
    x2[i] = targetres * (i - xc2)
for i in range(len(y0)):
    y0[i] = targetres * (i - yc0)
    y1[i] = targetres * (i - yc1)
    y2[i] = targetres * (i - yc2)

f_psf = interpolate.interp2d(xorig,yorig,dat_psfi)

psf_0 = f_psf(x0, y0)
psf_1 = f_psf(x1, y1)
psf_2 = f_psf(x2, y2)
#plt.imshow(psf_0, origin='lower')  # 1535, 1601
#plt.colorbar()
#plt.show()
psftest = f_psf(xorig,yorig)
print(xorig, yorig)
print(x0, y0)
#print(oop)

'''  # NGC 384 EXAMINE PSF DITHERS!
plt.imshow(psf_0[1805:1955, 1805:1955], origin='lower')  # center at 76,71 (lean toward lower)
plt.colorbar()
plt.show()
plt.imshow(psf_1[1805:1955, 1805:1955], origin='lower')  # center at 122,117 (lean toward lower)
plt.colorbar()
plt.show()
plt.imshow(psf_2[1805:1955, 1805:1955], origin='lower')  # center at 124,119 (lean toward higher)
plt.colorbar()
plt.show()
# '''  #

with fits.open(hst_n + 'ic0b09v5q_flc.fits', 'update') as hdu:
    hdr_v5q = hdu[0].header
    dat_v5q = hdu['SCI'].data
    hdr_v5q['history'] = 'Replaced ic0b09v5q_flc.fits science image with PSF data!'
#fits.writeto(n384_f814w_psf5, psf_0, hdr_v5q)
    # data_v5q = psf_0
    hdu['SCI',1].data = psf_0
    #hdu.flush()
    #fits.update(n384_f814w_psf5, psf_0, hdr_v5q, 'sci', 1)

    print(hdu.info())
print(oop)

with fits.open(hst_n + 'ic0b09vlq_flc.fits', 'update') as hdu:
    hdr_vlq = hdu[0].header
    dat_vlq = hdu[0].data

    hdr_vlq['history'] = 'Replaced ic0b09vlq_flc.fits science image with PSF data!'
#fits.writeto(n384_f814w_psfl, psf_1, hdr_vlq)
    #fits.update(n384_f814w_psfl, psf_1, hdr_vlq, 'sci')
    # dat_vlq = psf_1
    hdu['SCI',1].data = psf_1
    hdu.flush()

with fits.open(hst_n + 'ic0b09voq_flc.fits', 'update') as hdu:
    hdr_voq = hdu[0].header
    dat_voq = hdu[0].data

    hdr_voq['history'] = 'Replaced ic0b09voq_flc.fits science image with PSF data!'
#fits.writeto(n384_f814w_psfo, psf_2, hdr_voq)
    # fits.update(n384_f814w_psfo, psf_2, hdr_voq, 'sci')
    # dat_voq = psf_2
    hdu['SCI',1].data = psf_2
    hdu.flush()
print(oop)

# '''  # PGC 11179 EXAMINE PSF DITHERS
plt.imshow(psf_0[1575:1725, 1505:1655], origin='lower')  # center at 26,30 (lean toward lower)
plt.colorbar()
plt.show()
plt.imshow(psf_1[1575:1725, 1505:1655], origin='lower')  # center at 76,72 (lean toward lower)
plt.colorbar()
plt.show()
plt.imshow(psf_2[1575:1725, 1505:1655], origin='lower')  # center at 78,74 (lean toward higher)
plt.colorbar()
plt.show()

with fits.open(hst_p + 'ic0b14rmq_flc.fits', 'rb+') as hdu:
    hdr_rmq = hdu[0].header
    dat_rmq = hdu[0].data

hdr_rmq['history'] = 'Replaced ic0b14rmq_flc.fits science image with PSF data!'
#fits.writeto(p11179_f814w_psfm, psf_0, hdr_rmq)
fits.update(p11179_f814w_psfm, psf_0, hdr_rmq, 'sci')

with fits.open(hst_p + 'ic0b14rxq_flc.fits', 'rb+') as hdu:
    hdr_rxq = hdu[0].header
    dat_rxq = hdu[0].data

hdr_rxq['history'] = 'Replaced ic0b14rxq_flc.fits science image with PSF data!'
#fits.writeto(p11179_f814w_psfx, psf_1, hdr_rxq)
fits.update(p11179_f814w_psfx, psf_1, hdr_rxq, 'sci')

with fits.open(hst_p + 'ic0b14rzq_flc.fits', 'rb+') as hdu:
    hdr_rzq = hdu[0].header
    dat_rzq = hdu[0].data

hdr_rzq['history'] = 'Replaced ic0b14rzq_flc.fits science image with PSF data!'
# fits.writeto(p11179_f814w_psfz, psf_2, hdr_rzq)
fits.update(p11179_f814w_psfz, psf_2, hdr_rzq, 'sci')
print(oop)
# '''  #

#plt.imshow(ni_data_pxf08, origin='lower', vmax=10)  # N384: star at: x=1692, y=2538
#plt.imshow(ni_data_pxf001_ps03, origin='lower', vmax=10)  # this finally bad!
#plt.colorbar()
#plt.show()
#print(oop)

nweights = [n_wht_pxf001,n_wht_pxf01,n_wht_pxf4,n_wht_pxf6,n_wht_pxf8,n_wht_pxf1]
ndats = [ni_data_pxf001,ni_data_pxf01,ni_data_pxf04,ni_data_pxf06,ni_data_pxf08,ni_data_pxf1]
npxfs = ['0.01','0.1','0.4','0.6','0.8','1.0']
nnrow = 2
nncol = 3
# nguess = (130., 50, 50, 5., 5., 0.87, 0.3)
calc_noise(nweights, ndats, npxfs, nxctr, nyctr, 50, 50, nnrow, nncol, 700., 1692, 2538)
print(oop)

weights = [wht_pxf01,wht_pxf1,wht_pxf2,wht_pxf4,wht_pxf6,wht_pxf8,wht_pxf9,wht_pxf10]
dats = [i_data_pxf01,i_data_pxf1,i_data_pxf2,i_data_pxf4,i_data_pxf6,i_data_pxf8,i_data_pxf9,i_data_pxf10]
pxfs = ['0.01', '0.1', '0.2', '0.4', '0.6', '0.8', '0.9', '1.0']  # '0.01', '0.6', '0.9'
nrow = 2
ncol = 4
calc_noise(weights, dats, pxfs, 1581, 1647, 50, 50, nrow, ncol)  # 1934, 1118
# wht_pxf9 ; i_data_pxf9 ;; wht_pxf6 ; i_data_pxf6 ;; wht_pxf01 ; i_data_pxf01
print(oop)

i_data = i_data_pxf8
print(h_data.shape)  # (2637, 2710)
#print(h_hdr)
#print(oop)
#  I-band has been aligned with north, H-band has not!!!!!! So, ROTATE IT!
h_data, newpt = rot(h_data, h_hdr['CRPIX1'], h_hdr['CRPIX2'], angle=-102.7825141362319, reshape=True)
#plt.imshow(h_data, origin='lower', vmax=7.e5, vmin=0.)
#plt.colorbar()
#plt.show()
#print(oop)
#magi = zp_I - 2.5 * np.log10(i_data/1.5)
#plt.imshow(magi, origin='lower', vmax=28.)
#plt.colorbar()
#plt.show()
#print(oop)
print(h_data.shape)  # (3226, 3171)
print(i_data.shape)  # (3369, 3501)  # this one's bigger! (even after H-band rotation!)
# print(oop)

# '''  #
h_matchingi = np.zeros(shape=i_data.shape)  # H-band, matching the I-band shape!
# I want H[1367, 1457] to line up with I[1647, 1581]
# So: H extends from yy=1647-1367:yy+2710(?), and xx=1581-1466:xx+2637
#xx = 1647-(1325+73)  # pre-rotated values
#yy = 1581-(1390+76)  # pre-rotated values
xx = 1647-(1625+78)  # 78 from hidx[0]
yy = 1581-(1405+78)  # 78 from hidx[1]
if xx<0:
    print(h_data[-xx:,:].shape, 'hm', h_data.shape[0]-xx, h_data.shape[1]-yy)
    print(h_matchingi[:h_data.shape[0]+xx, yy:yy+h_data.shape[1]].shape)
    h_matchingi[:h_data.shape[0]+xx, yy:yy+h_data.shape[1]] = h_data[-xx:,:]
else:
    h_matchingi[xx:xx+h_data.shape[0], yy:yy+h_data.shape[1]] = h_data
# Gain = electrons/count
# convert i_data to counts, from counts/s
h_matchingi = h_matchingi / 1354.463046 / 2.5  #  H-band image units: e-, want counts/s: e- / s / (e-/count)
i_data /= 1.5  #  I-band units are electrons/s, want counts/s, GAIN here is 1.5, not 2.5!
# h_matchingi[xx:xx+2637, yy:yy+2710] = h_data / 1354.463046 / 2.5  # H-Band image units: e- / s / (e-/count)
#plt.imshow(i_data - h_matchingi, origin='lower')#, vmax=7e5, vmin=0.)
'''  # GOOD NOW!
plt.imshow(i_data, origin='lower', vmax=100, vmin=0.)
plt.plot(1510+71, 1575+72, 'r*')  # from idx
plt.plot()
plt.colorbar()
plt.show()
plt.imshow(h_matchingi, origin='lower', vmax=100, vmin=0.)
plt.plot(1510+71, 1575+72, 'r*')  # from idx
plt.colorbar()
plt.show()
# '''  #
mag_i = zp_I - 2.5 * np.log10(i_data)
mag_h = zp_H - 2.5 * np.log10(h_matchingi)
i_minus_h = mag_i - mag_h
i_minus_h[np.isnan(i_minus_h)] = 0.
vmax = np.amax(i_minus_h[1600:1700, 1525:1625])
i_minus_h = i_minus_h[1400:1900, 1325:1825]  # yrange, xrange
#xctr = 1581.05908-1325
xctr = 1510+71-1325
#yctr = 1645.05595-1400
yctr = 1575+72-1400
xax = np.zeros(shape=len(i_minus_h))
yax = np.zeros(shape=len(i_minus_h[0]))
for xi in range(len(i_minus_h)):
    xax[xi] = -0.06 * (xi - xctr)  # (arcsec/pix) * N_pix = arcsec
for yi in range(len(i_minus_h[0])):
    yax[yi] = 0.06 * (yi - yctr)  # (arcsec/pix) * N_pix = arcsec
extent = [xax[0], xax[-1], yax[0], yax[-1]]
plt.imshow(i_minus_h, origin='lower', vmin=0., vmax=vmax, extent=extent)#, vmax=3.)  #, vmax=12.)  # , vmax=0.1, vmin=-50 # , vmax=7e5, vmin=0.
plt.colorbar()
# plt.plot(1510+71, 1575+72, 'r*')  # from idx  # good when not using extent
plt.xlabel(r'$\Delta$ RA [arcsec]')
plt.ylabel(r'$\Delta$ Dec [arcsec]')
plt.title(r'PGC 11179 HST $I-H$')
plt.show()
print(oop)

# '''  #
'''  #
# CENTROIDS: H-band:
# h_data[np.abs(h_data) > 100] = 0.
plt.imshow(h_data, origin='lower', vmax=4.5e5, vmin=0.)
plt.colorbar()
plt.plot(1390+76, 1325+73, 'r*')  # PGC 11179
#plt.plot(1375+82, 1290+77, 'r*')  # from idx  # NGC 384
#plt.plot(1375+82.034056318563543, 1290+76.753908662648286, 'c*')  # from centroid  # NGC 384
plt.show()
plt.imshow(i_data, origin='lower', vmax=100., vmin=0.)
plt.colorbar()
#plt.plot(1510+71, 1575+72, 'r*')  # from idx
#plt.plot(1510+71.078015258901999, 1575+72.940969820368124, 'c*')  # from centroid
plt.show()
print(oop)
#  '''
# '''  #


i_data = np.nan_to_num(i_data)

initial_guess = (59,1582,1648,1.7,2.1,1.4,5.)  # amplitude = 88.8/1.5, x0, y0 = 1582, 1648, x_fwhm~4 -> sigma~4/2.355=1.7,
# y_fwhm~5 -> sigma~5/2.355=2.1, theta in radians ~80 deg -> 1.4 rads; offset ~ 0?

x = np.linspace(0, len(i_data), len(i_data))
y = np.linspace(0, len(i_data[0]), len(i_data[0]))
x, y = np.meshgrid(x, y)

gg = twoD_Gaussian((x,y), 59, 1582, 1648, 1.7, 2.1, 1.4, 5.)
print(gg.shape)
print(i_data.shape)
#print(i_data.ravel.shape)


popt, pcov = opt.curve_fit(twoD_Gaussian, (x, y), i_data.ravel(), p0=initial_guess)

print(popt)
# [ 3.29487947e-01  1.58105908e+03  1.64505595e+03 -1.07963065e+00 -1.29535590e+01  1.55820184e+00  4.78887205e-02]
## amplitude = 0.33, x0=1581.05908, y0=1645.05595, sigmax=-1.08, sigmay=-1.295, theta=1.55820184, offset=0.0478887205
print(oop)

# '''  #
#THESE ARE FOR NGC 384
#hi,hf = 1375,1525 # 950,1800  # 900,1850
#hyi,hyf = 1290,1440  # hi,hf
#hi,hf = 1390,1540  # pre-rot for p11179
#hyi,hyf = 1325,1475  # pre-rot for p11179
hi,hf = 1405,1555  # H-band rot p11179
hyi,hyf = 1625,1775  # H-band rot p11179
i0,i1 = 1510,1660  # 1000,2000  # 1200,2000
iy0,iy1 = 1575,1725  # i0,i1
h_data = h_data[hyi:hyf, hi:hf]
i_data = i_data[iy0:iy1, i0:i1]
hidx = np.unravel_index(np.argmax(h_data), h_data.shape)
iidx = np.unravel_index(np.argmax(i_data), i_data.shape)
print(hidx, iidx)  # ((78, 78), (72, 71))  # rotated H-band p11179!
# pre-rot p11179 H-band: ((73, 76), (72, 71))
# FOR NGC 384: (77, 82) 82 in x, 77 in y
#plt.imshow(h_data, origin='lower', vmax=7e5, vmin=0.)
#plt.colorbar()
#plt.show()
#print(oop)
xh, yh = centroid_2dg(h_data, mask=np.isnan(h_data))
print(xh, yh)  # Using 1290,1440 & 1375,1525: (82.034056318563543, 76.753908662648286)
# using 900,1850: (556.0822735162559, 466.70973014659847)
# Using 950,1800: (507.21197319028778, 416.78153137759034)  # better, still imperfect
xi, yi = centroid_2dg(i_data, mask=np.isnan(i_data))
print(xi, yi)  # Using 1575,1725 & 1510,1660: (72.940969820368124, 71.078015258901999)
# using 1200,2000: (383.27133931976698, 445.74008432184183)  # better, still imperfect
# Using 1000,2000: (583.33511514826932, 645.7459270168938)
# from scipy.ndimage import measurements
#h_data[np.isnan(h_data)] = 0.
#i_data[np.isnan(i_data)] = 0.
#xh1,yh1 = measurements.center_of_mass(h_data)
#print(xh1, yh1)  # (469.7267122214306, 550.71549453408932)  # worse, even accounting for flipping x,y!
#xi1,yi1 = measurements.center_of_mass(i_data)
#print(xi1, yi1)  # (420.9653334337014, 390.56088692909975)  # worse, even accounting for flipping x,y!
plt.imshow(h_data, origin='lower', vmax=4.5e5, vmin=0.)
plt.colorbar()
plt.plot(xh, yh, 'r*')
plt.plot(hidx[1], hidx[0], 'c*')
#plt.plot(yh1, xh1, 'c*')
plt.show()
plt.imshow(i_data, origin='lower', vmax=100., vmin=0.)
plt.colorbar()
plt.plot(xi, yi, 'r*')
plt.plot(iidx[1], iidx[0], 'c*')
#plt.plot(yi1, xi1, 'c*')
plt.show()
print(oop)
