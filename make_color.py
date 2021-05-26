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
p11179_f814w = hst_p + 'p11179_f814w_drizflc_006_sci.fits'


with fits.open(p11179_f814w) as hdu:
    i_hdr = hdu[0].header
    i_data = hdu[0].data

with fits.open(p11179_h) as hdu:
    h_hdr = hdu[0].header
    h_data = hdu[0].data

print(h_data.shape)  # (2637, 2710)
#print(h_hdr)
#print(oop)
#  I-band has been aligned with north, H-band has not!!!!!! So, ROTATE IT!
h_data, newpt = rot(h_data, h_hdr['CRPIX1'], h_hdr['CRPIX2'], angle=-102.7825141362319, reshape=True)
#plt.imshow(h_data, origin='lower', vmax=7.e5, vmin=0.)
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
# i_minus_h = i_data - h_matchingi  # dividing by ncombine?
zp_H = 24.662  # https://www.stsci.edu/hst/instrumentation/wfc3/data-analysis/photometric-calibration/ir-photometric-calibration
zp_I = 24.684  # UVIS2 according to fits header; https://www.stsci.edu/hst/instrumentation/wfc3/data-analysis/photometric-calibration/uvis-photometric-calibration
mag_i = zp_I - 2.5 * np.log10(i_data)
mag_h = zp_H - 2.5 * np.log10(h_matchingi)
i_minus_h = mag_i - mag_h
i_minus_h[np.isnan(i_minus_h)] = 0.
vmax = np.amax(i_minus_h[1600:1700, 1525:1625])
i_minus_h = i_minus_h[1400:1900, 1325:1825]  # yrange, xrange
xctr = 1510+71-1325
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
