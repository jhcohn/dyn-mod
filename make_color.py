from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import interpolation
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


def calc_noise(wht_data, sci_data, x0, y0, xlen, ylen):

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

    initial_guess = (37, 15, 15, 2, 2, 0., 0.)  # amplitude, x0, y0, x_fwhm, y_fwhm, theta in radians, offset

    x = np.linspace(0, 30, 30)
    y = np.linspace(0, 30, 30)
    x, y = np.meshgrid(x, y)

    import scipy.optimize as opt

    fig, ax = plt.subplots(2,2, sharex=True, sharey=True, figsize=(8,8.25))
    plt.subplots_adjust(hspace=0., wspace=0.)
    pxfs = ['0.4', '0.6', '0.8', '0.9']
    for sd in range(len(sci_data)):
        if sd < 2:
            row = 0
            col = sd
        else:
            row = 1
            col = sd - 2
        ax[row, col].imshow(sci_data[sd][1282:1312, 1153:1183], origin='lower', vmax=37., vmin=0.)

        popt, pcov = opt.curve_fit(twoD_Gaussian, (x, y), sci_data[sd][1282:1312, 1153:1183].ravel(), p0=initial_guess)
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



base = '/Users/jonathancohn/Documents/dyn_mod/galfit/u2698/'
gf = '/Users/jonathancohn/Documents/dyn_mod/galfit/'
fj = '/Users/jonathancohn/Documents/dyn_mod/for_jonathan/'
hst_p = '/Users/jonathancohn/Documents/hst_pgc11179/'
n384_h = fj + 'NGC0384_F160W_drz_sci.fits'
n384_adj_h = fj + 'NGC0384_F160W_drz_sci_adjusted.fits'
n384_adj_h_n7 = fj + 'NGC0384_F160W_drz_sci_adjusted_n7.fits'
n384_adj_h_inclsky = fj + 'NGC0384_F160W_drz_sci_adjusted_inclsky.fits'
n384_hmask = fj + 'NGC0384_F160W_drz_mask.fits'
n384_hmask_extended = fj + 'NGC0384_F160W_drz_mask_extended.fits'
p11179_h = fj + 'PGC11179_F160W_drz_sci.fits'
p11179_adj_h = fj + 'PGC11179_F160W_drz_sci_adjusted.fits'
p11179_adj_h_n7 = fj + 'PGC11179_F160W_drz_sci_adjusted_n7.fits'
p11179_adj_h_inclsky = fj + 'PGC11179_F160W_drz_sci_adjusted_inclsky.fits'
p11179_f814w_pxf04_wht = hst_p + 'p11179_f814w_drizflc_pxf04_006_wht.fits'
p11179_f814w_pxf06_wht = hst_p + 'p11179_f814w_drizflc_pxf06_006_wht.fits'
p11179_f814w_pxf08_wht = hst_p + 'p11179_f814w_drizflc_006_wht.fits'  # pxf08!
p11179_f814w_pxf09_wht = hst_p + 'p11179_f814w_drizflc_pxf09_006_wht.fits'
p11179_f814w_pxf04 = hst_p + 'p11179_f814w_drizflc_pxf04_006_sci.fits'
p11179_f814w_pxf06 = hst_p + 'p11179_f814w_drizflc_pxf06_006_sci.fits'
p11179_f814w_pxf08 = hst_p + 'p11179_f814w_drizflc_006_sci.fits'  # pxf08!
p11179_f814w_pxf09 = hst_p + 'p11179_f814w_drizflc_pxf09_006_sci.fits'
zp_H = 24.662  # https://www.stsci.edu/hst/instrumentation/wfc3/data-analysis/photometric-calibration/ir-photometric-calibration
zp_I = 24.684  # UVIS2 according to fits header; https://www.stsci.edu/hst/instrumentation/wfc3/data-analysis/photometric-calibration/uvis-photometric-calibration


with fits.open(p11179_h) as hdu:
    h_hdr = hdu[0].header
    h_data = hdu[0].data

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


calc_noise([wht_pxf4,wht_pxf6,wht_pxf8,wht_pxf9], [i_data_pxf4,i_data_pxf6,i_data_pxf8,i_data_pxf9], 1581, 1647, 50, 50)
print(oop)

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


import scipy.optimize as opt
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
