# Python 3 compatability

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import astropy.io.fits as fits
from matplotlib import rc

# re-defining plotting defaults
'''
from matplotlib import rcParams
rcParams.update({'xtick.major.pad': '7.0'})
rcParams.update({'xtick.major.size': '7.5'})
rcParams.update({'xtick.major.width': '1.5'})
rcParams.update({'xtick.minor.pad': '7.0'})
rcParams.update({'xtick.minor.size': '3.5'})
rcParams.update({'xtick.minor.width': '1.0'})
rcParams.update({'ytick.major.pad': '7.0'})
rcParams.update({'ytick.major.size': '7.5'})
rcParams.update({'ytick.major.width': '1.5'})
rcParams.update({'ytick.minor.pad': '7.0'})
rcParams.update({'ytick.minor.size': '3.5'})
rcParams.update({'ytick.minor.width': '1.0'})
rcParams.update({'font.size': 15})
'''

hduI = fits.open('galfit_u2698/ugc2698_f814w_pxfr075_pxs010_drc_align_sci.fits')
idat = hduI[0].data
hduI.close()
hduH = fits.open('galfit_u2698/ugc2698_f160w_pxfr075_pxs010_drz_rapidnuc_sci.fits')
hdat = hduH[0].data
hduH.close()

zp_I = 24.684  # 24.712 (UVIS1)  # 24.684 (UVIS2; according to header it's UVIS2!) # https://www.stsci.edu/hst/instrumentation/wfc3/data-analysis/photometric-calibration/uvis-photometric-calibration
magi = zp_I - 2.5 * np.log10(idat * 805.)  # image in counts/sec; texp = 805s, mag = zp - 2.5log10(total_counts)
zp_H = 24.6949  # https://www.stsci.edu/hst/instrumentation/wfc3/data-analysis/photometric-calibration/ir-photometric-calibration#section-cc19dbfc-8f60-4870-8765-43810de39924
magh = zp_H - 2.5 * np.log10(hdat * 898.467164)  # image in counts/sec; texp = 898.467164s, mag = zp - 2.5log10(tot)

# Central Cut
xi = 830
xf = 933
yi = 440
yf = 543
yctr = 491.0699 - yi
xctr = 880.8322 - xi
x_rad = np.zeros(shape=xf - xi)
for i in range(len(x_rad)):
    x_rad[i] = 0.1 * (i - xctr)  # (arcsec/pix) * N_pix = arcsec
y_rad = np.zeros(shape=yf - yi)
for i in range(len(y_rad)):
    y_rad[i] = 0.1 * (i - yctr)  # (arcsec/pix) * N_pix = arcsec
extent = [x_rad[0], x_rad[-1], y_rad[0], y_rad[-1]]
fullx = np.zeros(shape=len(hdat))
fully = np.zeros(shape=len(hdat[0]))
for i in range(len(fullx)):
    fullx[i] = 0.1 * (i - 491.0699)
for i in range(len(fully)):
    fully[i] = 0.1 * (i - 880.8322)
fullext = [fully[0], fully[-1], fullx[0], fullx[-1]]
print(extent)
magi = magi[yi:yf, xi:xf]
magh = magh[yi:yf, xi:xf]
vmax = 17.
vmin = 9.  # -40
from matplotlib import gridspec

fig = plt.figure(figsize=(16, 8))  # , constrained_layout=True)
gs = gridspec.GridSpec(6, 3)  # 3rows, 3cols
ax0 = plt.subplot(gs[0:6, 0:2])
magh_full = zp_H - 2.5 * np.log10(hdat * 898.467164)
im0 = ax0.imshow(magh_full, origin='lower', vmax=vmax, vmin=vmin, extent=fullext, cmap='Greys_r')
ax1 = plt.subplot(gs[0:2, 2])
im1 = ax1.imshow(magh, origin='lower', vmax=np.amax(magh), vmin=np.amin(magh), extent=extent, cmap='Greys_r')
ax2 = plt.subplot(gs[2:4, 2])
im2 = ax2.imshow(magi, origin='lower', vmax=np.amax(magi), vmin=np.amin(magi), extent=extent, cmap='Greys_r')
ax3 = plt.subplot(gs[4:6, 2])
ih = magi - magh
im3 = ax3.imshow(ih, origin='lower', vmax=np.amax(ih), vmin=np.amin(ih), extent=extent, cmap='Greys_r')  # 3, 1.5
ax0.set_xlabel('arcsec')
ax3.set_xlabel('arcsec')
ax0.set_ylabel('arcsec')
ax1.set_yticks([-4, -2, 0, 2, 4])
ax2.set_yticks([-4, -2, 0, 2, 4])
ax3.set_yticks([-4, -2, 0, 2, 4])
ax3.set_xticks([-4, -2, 0, 2, 4])
ax0.text(-80, 65, r'HST $H$')
ax1.text(-4, 4, r'HST $H$')
ax2.text(-4, 4, r'HST $I$')
ax3.text(-4, 4, r'HST $I - H$', color='w')

# plt.tight_layout()
plt.subplots_adjust(hspace=0.0)
plt.subplots_adjust(wspace=0.02)

# fig, ax = plt.subplots(1,3, figsize=(16,4))
# im0 = ax[0].imshow(magi, origin='lower', vmax=vmax, vmin=vmin, extent=extent)
# cbar0 = fig.colorbar(im0, ax=ax[0], pad=0.02)
# im1 = ax[1].imshow(magh, origin='lower', vmax=vmax, vmin=vmin, extent=extent)
# cbar1 = fig.colorbar(im1, ax=ax[1], pad=0.02)
# im2 = ax[2].imshow(magi - magh, origin='lower', vmax=3., vmin=1.5, extent=extent)
# cbar2 = fig.colorbar(im2, ax=ax[2], pad=0.02)
plt.show()
print(oop)