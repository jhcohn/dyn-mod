from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np

psf = 'ugc2698_f160w_pxfr075_pxs010_rapid_psf_drz_sci.fits'
hband = 'ugc2698_f160w_pxfr075_pxs010_drz_rapidnuc_sci.fits'
new_psf = 'ugc2698_f160w_pxfr075_pxs010_rapid_psf_drz_sci_no0.fits'
newclipped_psf = 'ugc2698_f160w_pxfr075_pxs010_rapid_psf_drz_sci_clippedno0.fits'  # no0, clipped [200:698,600:1160]
newclipped_psf2 = 'ugc2698_f160w_pxfr075_pxs010_rapid_psf_drz_sci_clipped2no0.fits'  # no0, clipped [100:798,500:1260]
new_hband = 'ugc2698_f160w_pxfr075_pxs010_drz_rapidnuc_sci_no0.fits'  # orig H-band; NaNs->0
hband_cps = 'ugc2698_f160w_pxfr075_pxs010_drz_rapidnuc_sci_nonan_cps.fits'  # orig H-band; NaNs->0, texp=1
hband_counts = 'ugc2698_f160w_pxfr075_pxs010_drz_rapidnuc_sci_nonan_counts.fits'  # orig H-band; NaNs->0, data*texp
hband_e = 'ugc2698_f160w_pxfr075_pxs010_drz_rapidnuc_sci_nonan_e.fits'  # orig H-band; NaNs->0, data*texp, gain->1, ncombine->1
re_skysub = 'ugc2698_f160w_pxfr075_pxs010_drz_rapidnuc_sci_nonan_e_skysub.fits'  # orig H-band; NaNs->0, data*texp, gain->1, ncombine->1
hband_e_akint = 'ugc2698_f160w_pxfr075_pxs010_drz_rapidnuc_sci_nonan_e_akintexp.fits'  # hband_e with texp = 1354.46
hband_e_akint2 = 'ugc2698_f160w_pxfr075_pxs010_drz_rapidnuc_sci_nonan_e_akintexp2.fits'  # hband_e_akint2 with TEXPTIME = 1354.46
re_akin_skysub = 'ugc2698_f160w_pxfr075_pxs010_drz_rapidnuc_sci_nonan_e_akintexp_skysub.fits'  # hband_e with texp = 1354.46
masked_e = 'ugc2698_f160w_pxfr075_pxs010_drz_rapidnuc_sci_nonan_e_cm.fits'  # orig H-band; NaNs->0, data*texp, gain->1, ncombine->1, multiply by combined mask
masked_rre = 'ugc2698_f160w_pxfr075_pxs010_drz_rapidnuc_sci_nonan_e_mem.fits'  # orig H-band; NaNs->0, data*texp, gain->1, ncombine->1, multiply by maskedgemask
new_hbandmult = 'ugc2698_f160w_pxfr075_pxs010_drz_rapidnuc_sci_multexptime.fits'
skysub_regH = 'ugc2698_f160w_pxfr075_pxs010_drz_rapidnuc_sci_no0_skysub.fits'  # orig H-band, nan->0, subtract 0.37
masked_skysub_rH = 'ugc2698_f160w_pxfr075_pxs010_drz_rapidnuc_sci_no0_skysub_combinedmask.fits'  # orig H-band, nan->0, subtract 0.37, multiply by combinedmask
dustmask = 'f160w_dust_mask_px010.fits'
edgemask = 'f160w_edgemask_px010.fits'
mask = 'f160w_mask_px010.fits'
cmask = 'f160w_combinedmask_px010.fits'  # edge, regular, and dust mask
mask2 = 'f160w_maskedgemask_px010.fits'  # edge, regular mask
hband_dustcorr = 'ugc2698_f160w_pxfr075_pxs010_ahcorr_rapidnuc_sci.fits'  # dust-corrected UGC 2698 H-band, from Ben
new_h_dustcorr = 'ugc2698_f160w_pxfr075_pxs010_ahcorr_rapidnuc_sci_nonan.fits'  # dust-corrected Hband, nan->0
skysub_ahcorr = 'ugc2698_f160w_pxfr075_pxs010_ahcorr_rapidnuc_sci_nonan_skysub.fits'  # ahcorr, nan->0, subtract 0.37
masked_skysub_ah = 'ugc2698_f160w_pxfr075_pxs010_ahcorr_rapidnuc_sci_nonan_skysub_maskedgemask.fits'  # ahcorr, nan->0, subtract 0.37, multiply by maskedgemask
hband_dustcorr_cps = 'ugc2698_f160w_pxfr075_pxs010_ahcorr_rapidnuc_sci_nonan_cps.fits'  # ahcorr, nan->0, texp=1
hband_dustcorr_counts = 'ugc2698_f160w_pxfr075_pxs010_ahcorr_rapidnuc_sci_nonan_counts.fits'  # ahcorr, nan->0, data*texp
hband_ahe = 'ugc2698_f160w_pxfr075_pxs010_ahcorr_rapidnuc_sci_nonan_e.fits'  # ahcorr, nan->0, data*texp, gain->1, ncombine->1
ahe_skysub = 'ugc2698_f160w_pxfr075_pxs010_ahcorr_rapidnuc_sci_nonan_e_skysub.fits'  # ahcorr, nan->0, data*texp, gain->1, ncombine->1
masked_ahe = 'ugc2698_f160w_pxfr075_pxs010_ahcorr_rapidnuc_sci_nonan_e_mem.fits'  # ahcorr, nan->0, data*texp, gain->1, ncombine->1, multiply by maskedgemask
masked_dustcorr_cps = 'ugc2698_f160w_pxfr075_pxs010_ahcorr_rapidnuc_sci_nonan_cps_maskedgemask.fits'  # ahcorr, nan->0, texp=1, multiply by maskedgemask
masked_dustcorr_counts = 'ugc2698_f160w_pxfr075_pxs010_ahcorr_rapidnuc_sci_nonan_counts_maskedgemask.fits'  # ahcorr, nan->0, data*texp, multiply by maskedgemask
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


# BEFORE running an image in GALFIT, check the units: if the units are in electrons, ensure the gain=1, NCOMBINE=1
# Elif the units are in counts, gain=2.5 and NCOMBINE=the number of combined images
# Note: if units are in counts/sec or electrons/sec, multiply the image by its exposure time, or set EXPTIME=1.0


with fits.open(n384_hmask_extended) as nm:
    hdr = nm[0].header
    data = nm[0].data

with fits.open(n384_adj_h) as nm:
    hdrh = nm[0].header
    datah = nm[0].data

# swap 0,1 in mask
data[data==0] = 2.  # choice of 2 here is arbitrary
data[data==1] = 0.
data[data==2] = 1.

fig, ax = plt.subplots(1,2, figsize=(10,5), sharey=True)
fig.subplots_adjust(hspace=0.01)

im = ax[0].imshow(data*datah, origin='lower', vmax=500, vmin=0)
#plt.colorbar(im)
im2 = ax[1].imshow(datah, origin='lower', vmax=500, vmin=0)
#plt.colorbar(im2)
plt.show()
print(oop)


with fits.open(n384_hmask) as nm:  # n384_hmask # n384_adj_h
    hdr = nm[0].header
    data = nm[0].data

# adjust mask based on neighbor galaxy position in n384_adj_h
# data[0:1200, 2000:] = 0.  # for the H-band image
data[0:1200, 2000:] = 1  # for the mask!

#plt.imshow(data, origin='lower', vmax=500, vmin=0)  # view n384_adj_h
plt.imshow(data, origin='lower')
plt.colorbar()
plt.show()

hdr['history'] = 'added large masking region to block out more of the neighbor galaxy'
fits.writeto(n384_hmask_extended, data, hdr)  # N384 mask, with the mask region around the neighboring galaxy extended
print(oop)


with fits.open(p11179_adj_h) as ph:
    print(ph.info())
    hdr = ph[0].header
    data = ph[0].data

data += 5.
hdr['history'] = 'added 5 to the data as artificial sky'
fits.writeto(p11179_adj_h_inclsky, data, hdr)  # regH, correcting header info for GALFIT input!
#print(oop)

with fits.open(n384_adj_h) as nh:
    print(nh.info())
    hdr = nh[0].header
    data = nh[0].data

data += 10.
hdr['history'] = 'added 10 to the data as artificial sky'
fits.writeto(n384_adj_h_inclsky, data, hdr)  # regH, correcting header info for GALFIT input!
print(oop)


with fits.open(p11179_h) as ph:
    print(ph.info())
    hdr = ph[0].header
    data = ph[0].data

hdr['CCDGAIN'] = 1.0
hdr['history'] = 'changed gain to 1, because image is actually in e, NOT counts'
hdr['history'] = 'keeping full exposure time'
fits.writeto(p11179_adj_h_n7, data, hdr)  # regH, correcting header info for GALFIT input!
#print(oop)

with fits.open(n384_h) as nh:
    print(nh.info())
    hdr = nh[0].header
    data = nh[0].data

hdr['CCDGAIN'] = 1.0
hdr['history'] = 'changed gain to 1, because image is actually in e, NOT counts'
hdr['history'] = 'keeping full exposure time'
fits.writeto(n384_adj_h_n7, data, hdr)  # regH, correcting header info for GALFIT input!
print(oop)


with fits.open(p11179_h) as ph:
    print(ph.info())
    hdr = ph[0].header
    data = ph[0].data

hdr['CCDGAIN'] = 1.0
hdr['NCOMBINE'] = 1.0
hdr['history'] = 'changed gain to 1, because image is actually in e, NOT counts'
hdr['history'] = 'changed NCOMBINE to 1, keeping full exposure time'
fits.writeto(p11179_adj_h, data, hdr)  # regH, correcting header info for GALFIT input!
#print(oop)

with fits.open(n384_h) as nh:
    print(nh.info())
    hdr = nh[0].header
    data = nh[0].data

hdr['CCDGAIN'] = 1.0
hdr['NCOMBINE'] = 1.0
hdr['history'] = 'changed gain to 1, because image is actually in e, NOT counts'
hdr['history'] = 'changed NCOMBINE to 1, keeping full exposure time'
fits.writeto(n384_adj_h, data, hdr)  # regH, correcting header info for GALFIT input!
print(oop)

'''  #
with fits.open(fj+'PGC11179_F814W_drc_sci_copy.fits') as pgci:
    hdr = pgci[0].header
    data = pgci[0].data

plt.imshow(data, origin='lower', vmax=1.2e4, vmin=-1e3)
plt.colorbar()
plt.show()
print(oop)

hdr['BUNIT'] = 'cps'
hdr['history'] = 'changed BUNIT from ELECTRONS to cps for display purposes only'
fits.writeto(fj+'PGC11179_F814W_drc_sci_copy_cps.fits', data, hdr)  # regH, correcting header info for GALFIT input!
print(oop)
# '''  #

with fits.open(hband_e_akint) as he:
    hdrhe = he[0].header
    datahe = he[0].data
hdrhe['TEXPTIME'] = 1354.46

hdrhe['history'] = "changed TEXPTIME to Akin's texp"
fits.writeto(base + hband_e_akint2, datahe, hdrhe)  # put Akin's texp in this image (even tho scale and counts will now be wrong; those are not important for GALFIT just reconstructing the model; just the error/residual will be meaningless)
print(oop)


# skysub re, rhe, ahe, akin-re
with fits.open(hband_ahe) as hdun:
    print(hdun.info())
    # print(hdu_h[0].header)
    hdrn = hdun[0].header
    datan = hdun[0].data
with fits.open(hband_e) as hdurh:
    print(hdurh.info())
    # print(hdu_h[0].header)
    hdrrh = hdurh[0].header
    datarh = hdurh[0].data
with fits.open(hband_e_akint) as hdura:
    print(hdurh.info())
    # print(hdu_h[0].header)
    hdra = hdura[0].header
    dataa = hdura[0].data

datan -= 337.5
datarh -= 337.5
dataa -= 337.5
msg = 'subtracted 337.5 from data image to remove the sky'
hdrn['history'] = msg
hdrrh['history'] = msg
hdra['history'] = msg
fits.writeto(base + ahe_skysub, datan, hdrn)  # sky-subtracted ahe
fits.writeto(base + re_skysub, datarh, hdrrh)  # sky-subtracted re
fits.writeto(base + re_akin_skysub, dataa, hdra)  # sky-subtracted re with Akin's texp

print("don't go below here")
print(oop)


with fits.open(hband_e) as he:
    hdrhe = he[0].header
    datahe = he[0].data
hdrhe['EXPTIME'] = 1354.46

hdrhe['history'] = "changed EXPTIME to Akin's texp"
fits.writeto(base + hband_e_akint, datahe, hdrhe)  # put Akin's texp in this image (even tho scale and counts will now be wrong; those are not important for GALFIT just reconstructing the model; just the error/residual will be meaningless)
print(oop)


with fits.open(cmask) as cm:
    hdrcm = cm[0].header
    datacm = cm[0].data
plt.imshow(datacm, origin='lower')  # yay already good for iraf (0=good, nonzero=reject)
plt.colorbar()
plt.show()
print(oop)


# DETERMINE SKY BELOW!!
with fits.open(hband_counts) as hb:  # hband_dustcorr_counts
    print(hb.info())
    hdrh = hb[0].header
    datah = hb[0].data
regs = [[1160, 1185, 60, 85], [1157, 1182, 21, 46], [1175, 1200, 62, 87], [1111, 1136, 8, 33]]
# means by region: 338.811, 337.205, 337.125, 336.7 --> mean = 337.46 (337.5)
# rmss by region: 339.494, 338.011, 337.876, 337.446 --> mean = 338.2
# regs = [[90, 115, 70, 95], [20, 45, 190, 215],
#         [6, 31, 1380, 1405], [140, 165, 1560, 1585], [1116, 1141, 1577, 1602], [1180, 1205, 1465, 1490]]
# means by region: 338.833, 346.611, 374.474, 370.064, 356.582, 358.588
# rmss by region: 339.231, 347.376, 375.325, 370.825, 357.407, 359.243

ans = []
for reg in regs:
    print(reg)
    skyreg = datah[reg[0]:reg[1], reg[2]:reg[3]]  # 1160:1185,60:85
    n = np.shape(skyreg)[0] * np.shape(skyreg)[1]
    rms = np.sqrt((1. / n) * (np.sum(skyreg**2)))
    rms_deviation = np.sqrt((1. / n) * np.sum((np.mean(skyreg) - skyreg)**2))
    a = [np.median(skyreg), np.mean(skyreg), rms, rms_deviation]
    ans.append(a)
    print(np.median(skyreg), np.mean(skyreg), rms, rms_deviation)

ans = np.asarray(ans)
for col in range(len(ans[0])):
    # print(ans[:, col])
    print(np.median(ans[:, col]))
    print(np.mean(ans[:, col]))
#                           median region | mean region  | rms          | rms_deviation
# electrons (median, mean): 336.3, 337.2  | 337.2, 337.5 | 337.9, 338.2 | 22.5, 22.4
# USE: sky = 337.5, sky rms_deviation = 22.4


# cps: median, mean = (0.37687877, 0.37709936); rms = 0.377858732438; rms_deviation = 0.0239433389818
# counts: median, mean = (338.61319, 338.8114); rms = 339.493665331; rms_deviation = 21.5123034564 (yay same for hband_counts and hband_dustcorr_counts)

# Run cps GALFIT with: galfit -skyped 0.38 -skyrms 0.02 galfit_params_u2698_ahcorr_n10_pf001_cps_zp24.txt
# Run counts GALFIT with: galfit -skyped 339.493665331 -skyrms 21.5123034564 galfit_params_u2698_ahcorr_n10_pf001_counts_zp24.txt
print(oop)


# SAVE MASKED IMAGE (rre)
with fits.open(hband_e) as hb:
    print(hb.info())
    hdrh = hb[0].header
    datah = hb[0].data

with fits.open(mask2) as hdu_regmask:
    print(hdu_regmask.info())
    hdrrm = hdu_regmask[0].header
    datarm = hdu_regmask[0].data

datah *= datarm
hdrh['history'] = 'multiplied by the regular mask (f160w_maskedgemask_px010.fits)'
fits.writeto(base + masked_rre, datah, hdrh)  # masked_ahe
print(oop)


# SAVE MASKED IMAGES (ahe, rhe)
with fits.open(hband_e) as hb:  # hband_ahe
    print(hb.info())
    hdrh = hb[0].header
    datah = hb[0].data

with fits.open(cmask) as hdu_regmask:  # mask2
    print(hdu_regmask.info())
    hdrrm = hdu_regmask[0].header
    datarm = hdu_regmask[0].data

datah *= datarm
# hdrh['history'] = 'multiplied by the regular mask (f160w_maskedgemask_px010.fits)'
hdrh['history'] = 'multiplied by the combined mask (f160w_combinedmask_px010.fits)'
fits.writeto(base + masked_e, datah, hdrh)  # masked_ahe
print(oop)


with fits.open(hband_dustcorr_counts) as hd:
    print(hd.info())
    hdrd = hd[0].header
    datad = hd[0].data

del hdrd['GAIN']  # remove incorrect gain entry
hdrd['CCDGAIN'] = 1.0
hdrd['NCOMBINE'] = 1.0
hdrd['history'] = 'changed gain to 1, because image is actually in e, NOT counts (was originally e/sec)'
hdrd['history'] = 'changed NCOMBINE to 1, keeping full exposure time'
fits.writeto(base + hband_ahe, datad, hdrd)  # ahcorr, correcting header info
print(oop)


with fits.open(hband_counts) as hd:
    print(hd.info())
    hdrd = hd[0].header
    datad = hd[0].data

hdrd['CCDGAIN'] = 1.0
hdrd['NCOMBINE'] = 1.0
hdrd['history'] = 'changed gain to 1, because image is actually in e, NOT counts (was originally e/sec)'
hdrd['history'] = 'changed NCOMBINE to 1, keeping full exposure time'
fits.writeto(base + hband_e, datad, hdrd)  # regH, correcting header info
print(oop)



with fits.open(new_hband) as hd:
    print(hd.info())
    hdrd = hd[0].header
    datad = hd[0].data

datad *= 898.467164
hdrd['history'] = 'multiplied image by exptime, so get it in units of counts (instead of counts/sec)'
fits.writeto(base + hband_counts, datad, hdrd)  # ahcorr texp=898.467164
print(oop)


with fits.open(hband_dustcorr_counts) as hb:
    print(hb.info())
    hdrh = hb[0].header
    datah = hb[0].data

with fits.open(mask2) as hdu_regmask:
    print(hdu_regmask.info())
    # print(hdu_h[0].header)
    hdrrm = hdu_regmask[0].header
    datarm = hdu_regmask[0].data

datah *= datarm
hdrh['history'] = 'multiplied by the regular mask (f160w_maskedgemask_px010.fits)'
fits.writeto(base + masked_dustcorr_counts, datah, hdrh)  # ahcorr data*texp maskedgemask
print(oop)


with fits.open(new_h_dustcorr) as hd:
    print(hd.info())
    hdrd = hd[0].header
    datad = hd[0].data

datad *= 898.467164
hdrd['EXPTIME'] = 898.467164
hdrd['GAIN'] = 2.5
hdrd['NCOMBINE'] = 2
hdrd['history'] = 'multiplied image by exptime, so get it in units of counts (instead of counts/sec)'
hdrd['history'] = 'set EXPTIME=898.467164'
hdrd['history'] = 'set GAIN=2.5'
hdrd['history'] = 'set NCOMBINE=2'
fits.writeto(base + hband_dustcorr_counts, datad, hdrd)  # ahcorr texp=898.467164
print(oop)


with fits.open(hband_dustcorr_cps) as hb:
    print(hb.info())
    hdrh = hb[0].header
    datah = hb[0].data

with fits.open(mask2) as hdu_regmask:
    print(hdu_regmask.info())
    # print(hdu_h[0].header)
    hdrrm = hdu_regmask[0].header
    datarm = hdu_regmask[0].data

datah *= datarm
hdrh['history'] = 'multiplied by the regular mask (f160w_maskedgemask_px010.fits)'
fits.writeto(base + masked_dustcorr_cps, datah, hdrh)  # ahcorr texp=1 maskedgemask
print(oop)


with fits.open(new_hband) as hb:
    print(hb.info())
    hdrh = hb[0].header
    datah = hb[0].data

# datah /= 898.467164
hdrh['EXPTIME'] = 1.0
hdrh['history'] = 'set new exptime to 1s, bc image in units of counts/sec'

with fits.open(new_h_dustcorr) as hd:
    print(hd.info())
    hdrd = hd[0].header
    datad = hd[0].data

# datad /= 898.467164
hdrd['EXPTIME'] = 1.0
hdrd['GAIN'] = 2.5
hdrd['NCOMBINE'] = 2
hdrd['history'] = 'set new exptime to 1s, bc image in units of counts/sec'
hdrd['history'] = 'set GAIN=2.5'
hdrd['history'] = 'set NCOMBINE=2'
fits.writeto(base + hband_cps, datah, hdrh)  # regH texp=1
fits.writeto(base + hband_dustcorr_cps, datad, hdrd)  # ahcorr texp=1
print(oop)

# no-nans (replaced with 0s), sky-subtracted images for ahcorr and regH
with fits.open(skysub_ahcorr) as hdun:
    print(hdun.info())
    # print(hdu_h[0].header)
    hdrn = hdun[0].header
    datan = hdun[0].data
with fits.open(skysub_regH) as hdurh:
    print(hdurh.info())
    # print(hdu_h[0].header)
    hdrrh = hdurh[0].header
    datarh = hdurh[0].data

# masked
with fits.open(mask2) as hdu_regmask:
    print(hdu_regmask.info())
    # print(hdu_h[0].header)
    hdrrm = hdu_regmask[0].header
    datarm = hdu_regmask[0].data
with fits.open(cmask) as hdu_cmask:
    print(hdu_cmask.info())
    # print(hdu_h[0].header)
    hdrcm = hdu_cmask[0].header
    datacm = hdu_cmask[0].data

datan *= datarm  # multiply ahcorr by regular mask
datarh *= datacm  # multiply regH by combined mask
hdrn['history'] = 'multiplied by the regular mask (f160w_maskedgemask_px010.fits)'
hdrrh['history'] = 'multiplied by the combined dust mask (f160w_combinedmask_px010.fits)'
fits.writeto(base + masked_skysub_ah, datan, hdrn)  # sky-subtracted ahcorr
fits.writeto(base + masked_skysub_rH, datarh, hdrrh)  # sky-subtracted regH

print("don't go below here")
print(oop)


# Save new sky-subtracted versions of ahcorr and regH
with fits.open(new_h_dustcorr) as hdun:  # ahcorr
    print(hdun.info())
    # print(hdu_h[0].header)
    hdrn = hdun[0].header
    datan = hdun[0].data
with fits.open(new_hband) as hdurh:  # regH
    print(hdurh.info())
    # print(hdu_h[0].header)
    hdrrh = hdurh[0].header
    datarh = hdurh[0].data

datan -= 0.37  # sky subtraction
datarh -= 0.37
datan[datan==-0.37] = 0.
datarh[datarh==-0.37] = 0.
hdrn['history'] = 'subtracted 0.37 from the image to remove sky'
hdrrh['history'] = 'subtracted 0.37 from the image to remove sky'
fits.writeto(base + skysub_ahcorr, datan, hdrn)  # sky-subtracted ahcorr
fits.writeto(base + skysub_regH, datarh, hdrrh)  # sky-subtracted regH

print("don't go below here")
print(oop)



with fits.open(newclipped_psf2) as hdun:
    print(hdun.info())
    # print(hdu_h[0].header)
    hdrn = hdun[0].header
    datan = hdun[0].data

plt.imshow(datan, origin='lower', vmin=0., vmax=1e-5)  # vmin=0., vmax=1e5) # vmin=np.amin(datan), vmax=1e-5
plt.colorbar()
plt.show()
print(oop)



print("don't go below here")
print(oop)


with fits.open(base + hband_dustcorr) as hdun:
    print(hdun.info())
    # print(hdu_h[0].header)
    hdrn = hdun[0].header
    datan = hdun[0].data

print(np.isnan(datan).any())
datan[np.isnan(datan)] = 0.
print(np.isnan(datan).any())
hdrn['history'] = 'replaced NaNs with 0s'
fits.writeto(base + new_h_dustcorr, datan, hdrn)  # new_psf
print(oop)


masks = [mask, edgemask]  # masks = [mask, dustmask, edgemask]
combined_mask = np.zeros(shape=(1231, 1620))

for m in masks:
    with fits.open(m) as hdun:
        print(hdun.info())
        # print(hdu_h[0].header)
        hdrn = hdun[0].header
        datan = hdun[0].data

        combined_mask += datan

plt.imshow(combined_mask, vmin=np.amin(combined_mask), vmax=np.amax(combined_mask), origin='lower')
plt.colorbar()
plt.show()

hdrn['history'] = 'added f160w_edgemask_px010, and f160w_mask_px010'  # [600:1160, 200:698]  # f160w_dust_mask_px010,
fits.writeto(mask2, combined_mask, hdrn)  # new_psf
print(oop)

hdrn['history'] = 'added f160w_dust_mask_px010, f160w_edgemask_px010, and f160w_mask_px010'  # [600:1160, 200:698]
fits.writeto(cmask, combined_mask, hdrn)  # new_psf

print(oop)


with fits.open(newclipped_psf2) as hdun:
    print(hdun.info())
    # print(hdu_h[0].header)
    hdrn = hdun[0].header
    datan = hdun[0].data

plt.imshow(datan, origin='lower', vmin=0., vmax=1e-5)  # vmin=0., vmax=1e5) # vmin=np.amin(datan), vmax=1e-5
plt.colorbar()
plt.show()
print(oop)

# print(hdrn)
print(datan.shape)
print(np.isnan(datan).any())
print("don't go below here")
# print(oop)


print(np.isnan(datan).any())
# datan = datan[200:698,600:1160]  # center at 880,449 [new_len=560,498], center at 280,249 = 560/2 +1, 498/2 +1
datan = datan[440:540, 830:930]  # center at 491,881 [newlen=100,698], newcenter 51,51 = (540-440)/2 +1, (930-830)/2 +1
print(np.isnan(datan).any())
hdrn['history'] = 'clipped to smaller region: from 1620,1231 -> [440:540, 830:930]'  # [600:1160, 200:698]
fits.writeto(newclipped_psf2, datan, hdrn)  # new_psf
print(oop)

print(np.isnan(datan).any())
datan[np.isnan(datan)] = 0.
print(np.isnan(datan).any())
hdrn['history'] = 'replaced NaNs with 0s'
fits.writeto(new_hband, datan, hdrn)  # new_psf
print(oop)


with fits.open(psf) as hdu_psf:
    print(hdu_psf.info())
    # print(hdu_psf[0].header)
    new_hdr = hdu_psf[0].header
    new_data = hdu_psf[0].data

with fits.open(hband) as hdu_h:
    print(hdu_h.info())
    # print(hdu_h[0].header)
    new_hdrh = hdu_h[0].header
    new_datah = hdu_h[0].data

print(np.isnan(new_datah).any())
print(np.isnan(new_data).any())
new_data[np.isnan(new_data)] = 0.
new_datah[np.isnan(new_datah)] = 0.
print(np.isnan(new_datah).any())
print(np.isnan(new_data).any())

# print(new_hdrh['EXPTIME'])
new_hdrh['history'] = 'image multiplied by EXPTIME, ADU now in counts instead of counts/s'
# print(new_hdrh['history'])

print(np.nanmedian(new_datah))
plt.imshow(new_datah)
plt.colorbar()
plt.show()
new_datah *= new_hdrh['EXPTIME']
print(np.nanmedian(new_datah))
plt.imshow(new_datah)
plt.colorbar()
plt.show()

fits.writeto(new_hband, new_datah, new_hdrh)

with fits.open(new_hband) as hdu:
    print(hdu.info())
print(oop)


# print(new_data.shape)  # (1231, 1620) # BOTH!
# BAH THEY'RE THE SAME SIZE: (1231, 1620)
new_data = new_data[443:, ]
print(len(new_data), len(new_data[0]))
new_hdr['NAXIS1'] = len(new_data)
new_hdr['NAXIS2'] = str(len(new_data[0]))
print(new_hdr)
print(oop)

fits.writeto(new_psf, new_data, new_hdr)

hdu1 = fits.PrimaryHDU(collapsed_mask)
hdul1 = fits.HDUList([hdu1])
hdul1.writeto(lucy_mask)  # write out to mask file

'''
SIMPLE  =                    T / Written by IDL:  Mon Oct 28 13:05:07 2019      
BITPIX  =                  -32 / array data type                                
NAXIS   =                    2 / number of array dimensions                     
NAXIS1  =                 1620                                                  
NAXIS2  =                 1231                                                  
EXTEND  =                    T                                                  
COMMENT   FITS (Flexible Image Transport System) format is defined in 'Astronomy
COMMENT   and Astrophysics', volume 376, page 359; bibcode: 2001A&A...376..359H 
FILENAME= 'ugc2698_f160w_pxfr075_pxs010_drz_sci.fits' / name of file            
FILETYPE= 'SCI      '          / type of data found in data file                
                                                                                
TELESCOP= 'HST'                / telescope used to acquire data                 
INSTRUME= 'WFC3  '             / identifier for instrument used to acquire data 
EQUINOX =               2000.0 / equinox of celestial coord. system             
                                                                                
              / DATA DESCRIPTION KEYWORDS                                       
                                                                                
ROOTNAME= 'ugc2698_f160w_pxfr075_pxs010' / rootname of the observation set      
IMAGETYP= 'EXT               ' / type of exposure identifier                    
PRIMESI = 'WFC3  '             / instrument designated as prime                 
                                                                                
              / TARGET INFORMATION                                              
                                                                                
TARGNAME= 'UGC2698                        ' / proposer's target name            
RA_TARG =   5.051200000000E+01 / right ascension of the target (deg) (J2000)    
DEC_TARG=   4.086388888889E+01 / declination of the target (deg) (J2000)        
                                                                                
              / PROPOSAL INFORMATION                                            
                                                                                
PROPOSID=                13416 / PEP proposal identifier                        
LINENUM = '01.001         '    / proposal logsheet line number                  
PR_INV_L= 'van den Bosch                 ' / last name of principal investigator
PR_INV_F= 'Remco               ' / first name of principal investigator         
PR_INV_M= '                    ' / middle name / initial of principal investigat
                                                                                
              / EXPOSURE INFORMATION                                            
                                                                                
SUNANGLE=            96.192055 / angle between sun and V1 axis                  
MOONANGL=            26.893627 / angle between moon and V1 axis                 
FGSLOCK = 'FINE              ' / commanded FGS lock (FINE,COARSE,GYROS,UNKNOWN) 
GYROMODE= 'T'                  / number of gyros scheduled, T=3+OBAD            
REFFRAME= 'ICRS    '           / guide star catalog version                     
MTFLAG  = ' '                  / moving target flag; T if it is a moving target 
                                                                                
DATE-OBS= '2013-08-28'         / UT date of start of observation (yyyy-mm-dd)   
TIME-OBS= '21:09:36'           / UT time of start of observation (hh:mm:ss)     
EXPSTART=   5.653288167814E+04 / exposure start time (Modified Julian Date)     
EXPEND  =       56532.89300036 / exposure end time (Modified Julian Date)       
EXPTIME =           898.467164 / exposure duration (seconds)--calculated        
TEXPTIME=           898.467164                                                  
EXPFLAG = 'NORMAL       '      / Exposure interruption indicator                
QUALCOM1= '                                                                    '
QUALCOM2= '                                                                    '
QUALCOM3= '                                                                    '
QUALITY = '                                                                    '
                                                                                
                                                                                
              / POINTING INFORMATION                                            
                                                                                
PA_V3   =            71.754967 / position angle of V3-axis of HST (deg)         
                                                                                
              / TARGET OFFSETS (POSTARGS)                                       
                                                                                
                                                                                
              / DIAGNOSTIC KEYWORDS                                             
                                                                                
OPUS_VER= 'HSTDP 2019_3      ' / data processing software system version        
CSYS_VER= 'hstdp-2019.2'       / Calibration software system version id         
CAL_VER = '3.5.0(Oct-09-2018)' / CALWF3 code version                            
PROCTIME=   5.861344687500E+04 / Pipeline processing time (MJD)                 
                                                                                
              / INSTRUMENT CONFIGURATION INFORMATION                            
                                                                                
OBSTYPE = 'IMAGING       '     / observation type - imaging or spectroscopic    
OBSMODE = 'MULTIACCUM'         / operating mode                                 
SCLAMP  = 'NONE          '     / lamp status, NONE or name of lamp which is on  
NRPTEXP =                    1 / number of repeat exposures in set: default 1   
SUBARRAY=                    F / data from a subarray (T) or full frame (F)     
SUBTYPE = 'FULLIMAG'           / Size/type of IR subarray                       
DETECTOR= 'IR  '               / detector in use: UVIS or IR                    
FILTER  = 'F160W  '            / element selected from filter wheel             
SAMP_SEQ= 'STEP50  '           / MultiAccum exposure time sequence name         
NSAMP   =                   15 / number of MULTIACCUM samples                   
SAMPZERO=             2.911756 / sample time of the zeroth read (sec)           
APERTURE= 'IR              '   / aperture name                                  
PROPAPER= '                '   / proposed aperture name                         
DIRIMAGE= 'NONE     '          / direct image for grism or prism exposure       
                                                                                
              / POST-SAA DARK KEYWORDS                                          
                                                                                
SAACRMAP= 'N/A               ' / SAA cosmic ray map file                        
                                                                                
              / SCAN KEYWORDS                                                   
                                                                                
SCAN_TYP= 'N                 ' / C:bostrophidon; D:C with dwell; N:N/A          
SCAN_WID=   0.000000000000E+00 / scan width (arcsec)                            
ANG_SIDE=   0.000000000000E+00 / angle between sides of parallelogram (deg)     
DWELL_LN=                    0 / dwell pts/line for scan pointing (1-99,0 if NA)
DWELL_TM=   0.000000000000E+00 / wait time (duration) at each dwell point (sec) 
SCAN_ANG=   0.000000000000E+00 / position angle of scan line (deg)              
SCAN_RAT=   0.000000000000E+00 / commanded rate of the line scan (arcsec/sec)   
NO_LINES=                    0 / number of lines per scan (1-99,0 if NA)        
SCAN_LEN=   0.000000000000E+00 / scan length (arcsec)                           
SCAN_COR= 'C                 ' / scan coordinate frame of ref: celestial,vehicle
CSMID   = 'IR     '            / Channel Select Mechanism ID                    
                                                                                
              / CALIBRATION SWITCHES: PERFORM, OMIT, COMPLETE, SKIPPED          
                                                                                
DQICORR = 'COMPLETE'           / data quality initialization                    
ZSIGCORR= 'COMPLETE'           / Zero read signal correction                    
ZOFFCORR= 'COMPLETE'           / subtract MULTIACCUM zero read                  
DARKCORR= 'COMPLETE'           / Subtract dark image                            
BLEVCORR= 'COMPLETE'           / subtract bias level computed from ref pixels   
NLINCORR= 'COMPLETE'           / correct for detector nonlinearities            
FLATCORR= 'COMPLETE'           / flat field data                                
CRCORR  = 'COMPLETE'           / identify cosmic ray hits                       
UNITCORR= 'COMPLETE'           / convert to count rates                         
PHOTCORR= 'COMPLETE'           / populate photometric header keywords           
RPTCORR = 'OMIT    '           / combine individual repeat observations         
DRIZCORR= 'COMPLETE'           / drizzle processing                             
                                                                                
              / CALIBRATION REFERENCE FILES                                     
                                                                                
BPIXTAB = 'iref$35620330i_bpx.fits' / bad pixel table                           
CCDTAB  = 'iref$t2c16200i_ccd.fits' / detector calibration parameters           
OSCNTAB = 'iref$q911321mi_osc.fits' / detector overscan table                   
CRREJTAB= 'iref$u6a1748ri_crr.fits' / cosmic ray rejection parameters           
DARKFILE= 'iref$3562021si_drk.fits' / dark image file name                      
NLINFILE= 'iref$u1k1727mi_lin.fits' / detector nonlinearities file              
PFLTFILE= 'iref$uc721145i_pfl.fits' / pixel to pixel flat field file name       
DFLTFILE= 'N/A                    ' / delta flat field file name                
LFLTFILE= 'N/A                    ' / low order flat                            
GRAPHTAB= 'N/A                    ' / the HST graph table                       
COMPTAB = 'N/A                    ' / the HST components table                  
IMPHTTAB= 'iref$wbj1825ri_imp.fits' / Image Photometry Table                    
IDCTAB  = 'iref$w3m18525i_idc.fits' / image distortion correction table         
DGEOFILE= 'N/A               ' / Distortion correction image                    
MDRIZTAB= 'iref$3562021pi_mdz.fits' / MultiDrizzle parameter table              
                                                                                
              / COSMIC RAY REJECTION ALGORITHM PARAMETERS                       
                                                                                
MEANEXP =                  0.0 / reference exposure time for parameters         
SCALENSE=                  0.0 / multiplicative scale factor applied to noise   
INITGUES= '   '                / initial guess method (MIN or MED)              
CRSIGMAS= '               '    / statistical rejection criteria                 
CRRADIUS=                  0.0 / rejection propagation radius (pixels)          
CRTHRESH=             0.000000 / rejection propagation threshold                
BADINPDQ=                    0 / data quality flag bits to reject               
CRMASK  =                    F / flag CR-rejected pixels in input files (T/F)   
                                                                                
              / PHOTOMETRY KEYWORDS                                             
                                                                                
PHOTMODE= 'WFC3 IR F160W'      / Obser                                          
PHOTFLAM=        1.9275602E-20 / Inverse sensitivity, ergs/cm2/A/e-             
PHOTFNU =        1.5187570E-07 / Inverse sensitivity, Jy*sec/e-                 
PHOTZPT =       -2.1100000E+01 / ST magnitude zero point                        
PHOTPLAM=        1.5369176E+04 / Pivot wavelength (Angstroms)                   
PHOTBW  =        8.2625085E+02 / RMS bandwidth of filter plus detector          
                                                                                
              / OTFR KEYWORDS                                                   
                                                                                
T_SGSTAR= '                  ' / OMS calculated guide star control              
                                                                                
              / PATTERN KEYWORDS                                                
                                                                                
PATTERN1= 'LINE                    ' / primary pattern type                     
P1_SHAPE= 'LINE              ' / primary pattern shape                          
P1_PURPS= 'DITHER    '         / primary pattern purpose                        
P1_NPTS =                    2 / number of points in primary pattern            
P1_PSPAC=                23.02 / point spacing for primary pattern (arc-sec)    
P1_LSPAC=             0.000000 / line spacing for primary pattern (arc-sec)     
P1_ANGLE=             0.000000 / angle between sides of parallelogram patt (deg)
P1_FRAME= 'POS-TARG '          / coordinate frame of primary pattern            
P1_ORINT=                0.713 / orientation of pattern to coordinate frame (deg
P1_CENTR= 'NO '                / center pattern relative to pointing (yes/no)   
PATTSTEP=                    1 / position number of this point in the pattern   
                                                                                
              / ENGINEERING PARAMETERS                                          
                                                                                
CCDAMP  = 'ABCD'               / CCD Amplifier Readout Configuration            
CCDGAIN =                  2.5 / commanded gain of CCD                          
                                                                                
              / CALIBRATED ENGINEERING PARAMETERS                               
                                                                                
ATODGNA =        2.3399999E+00 / calibrated gain for amplifier A                
ATODGNB =        2.3699999E+00 / calibrated gain for amplifier B                
ATODGNC =        2.3099999E+00 / calibrated gain for amplifier C                
ATODGND =        2.3800001E+00 / calibrated gain for amplifier D                
READNSEA=        2.0200001E+01 / calibrated read noise for amplifier A          
READNSEB=        1.9799999E+01 / calibrated read noise for amplifier B          
READNSEC=        1.9900000E+01 / calibrated read noise for amplifier C          
READNSED=        2.0100000E+01 / calibrated read noise for amplifier D          
BIASLEVA=             0.000000 / bias level for amplifier A                     
BIASLEVB=             0.000000 / bias level for amplifier B                     
BIASLEVC=             0.000000 / bias level for amplifier C                     
BIASLEVD=             0.000000 / bias level for amplifier D                     
                                                                                
              / ASSOCIATION KEYWORDS                                            
                                                                                
ASN_ID  = 'IC7Z01010 '         / unique identifier assigned to association      
ASN_TAB = 'ic7z01010_asn.fits     ' / name of the association table             
ASN_MTYP= 'PROD-DTH'           / Role of the Member in the Association          
CRDS_CTX= 'hst_0699.pmap'                                                       
CRDS_VER= '7.3.0, 7.3.0, a24ae72c'                                              
ATODTAB = 'N/A     '                                                            
BIACFILE= 'N/A     '                                                            
BIASFILE= 'N/A     '                                                            
D2IMFILE= 'N/A     '                                                            
DRKCFILE= 'N/A     '                                                            
FLSHFILE= 'N/A     '                                                            
NPOLFILE= 'N/A     '                                                            
PCTETAB = 'N/A     '                                                            
SNKCFILE= 'N/A     '                                                            
UPWCSVER= '1.4.0   '           / Version of STWCS used to updated the WCS       
PYWCSVER= '3.0.4   '           / Version of PYWCS used to updated the WCS       
DISTNAME= 'ic7z01r2q_w3m18525i-NOMODEL-NOMODEL'                                 
SIPNAME = 'ic7z01r2q_w3m18525i'                                                 
RULESVER=                  1.1 / Version ID for header kw rules file            
BLENDVER= '1.2.0   '           / Version of blendheader software used           
RULEFILE= '/Users/Boizelle/anaconda/envs/iraf27/lib/python2.7/site-packages/fi&'
CONTINUE  'tsblender/wfc3_header.rules&'                                        
CONTINUE  '' / Name of header kw rules file                                     
PROD_VER= 'DrizzlePac 2.1.21'                                                   
NDRIZIM =                    2 / Drizzle, No. images drizzled onto output       
D001OUDA= 'ugc2698_f160w_pxfr075_pxs010_drz.fits' / Drizzle, output data image  
D001VER = 'Callable C-based DRIZZLE Version 0.8 (20th M' / Drizzle, task version
D001SCAL=                  0.1 / Drizzle, pixel size (arcsec) of output image   
D001COEF= 'SIP     '           / Drizzle, source of coefficients                
D001OUWE= 'ugc2698_f160w_pxfr075_pxs010_drz_wht.fits' / Drizzle, output weightin
D001OUCO= 'ugc2698_f160w_pxfr075_pxs010_drz_ctx.fits' / Drizzle, output context 
D001WTSC=                    1 / Drizzle, weighting factor for input image      
D001MASK= 'ic7z01r2q_sci1_final_mask.fits' / Drizzle, input weighting image     
D001FVAL= 'INDEF   '           / Drizzle, fill value for zero weight output pix 
D001WKEY= '' / Input image WCS Version used                                     
D001OUUN= 'cps     '           / Drizzle, units of output image - counts or cps 
D001KERN= 'square  '           / Drizzle, form of weight distribution kernel    
D001GEOM= 'wcs     '           / Drizzle, source of geometric information       
D001ISCL=   0.1282500028610229 / Drizzle, default IDCTAB pixel size(arcsec)     
D001PIXF=                 0.75 / Drizzle, linear size of drop                   
D001DATA= 'ic7z01r2q_flt.fits[sci,1]' / Drizzle, input data image               
D001DEXP=           449.233582 / Drizzle, input image exposure time (s)         
D002OUDA= 'ugc2698_f160w_pxfr075_pxs010_drz.fits' / Drizzle, output data image  
D002VER = 'Callable C-based DRIZZLE Version 0.8 (20th M' / Drizzle, task version
D002SCAL=                  0.1 / Drizzle, pixel size (arcsec) of output image   
D002COEF= 'SIP     '           / Drizzle, source of coefficients                
D002OUWE= 'ugc2698_f160w_pxfr075_pxs010_drz_wht.fits' / Drizzle, output weightin
D002OUCO= 'ugc2698_f160w_pxfr075_pxs010_drz_ctx.fits' / Drizzle, output context 
D002WTSC=                    1 / Drizzle, weighting factor for input image      
D002MASK= 'ic7z01r3q_sci1_final_mask.fits' / Drizzle, input weighting image     
D002FVAL= 'INDEF   '           / Drizzle, fill value for zero weight output pix 
D002WKEY= '' / Input image WCS Version used                                     
D002OUUN= 'cps     '           / Drizzle, units of output image - counts or cps 
D002KERN= 'square  '           / Drizzle, form of weight distribution kernel    
D002GEOM= 'wcs     '           / Drizzle, source of geometric information       
D002ISCL=   0.1282500028610229 / Drizzle, default IDCTAB pixel size(arcsec)     
D002PIXF=                 0.75 / Drizzle, linear size of drop                   
D002DATA= 'ic7z01r3q_flt.fits[sci,1]' / Drizzle, input data image               
D002DEXP=           449.233582 / Drizzle, input image exposure time (s)         
INHERIT =                    T / inherit the primary header                     
EXPNAME = 'ic7z01r2q                ' / exposure identifier                     
BUNIT   = 'ELECTRONS/S'        / brightness units                               
WCSAXES =                    2 / number of World Coordinate System axes         
CRPIX1  =    809.6162680387491 / x-coordinate of reference pixel                
CRPIX2  =    615.3175137042995 / y-coordinate of reference pixel                
CRVAL1  =    50.51503112332516 / first axis value at reference pixel            
CRVAL2  =    40.86062514579236 / second axis value at reference pixel           
CTYPE1  = 'RA---TAN'           / the coordinate type for the first axis         
CTYPE2  = 'DEC--TAN'           / the coordinate type for the second axis        
ORIENTAT=    116.7549279549141 / position angle of image y axis (deg. e of n)   
VAFACTOR=                  1.0                                                  
CD1_1   = 1.25048678786937E-05 / partial of first axis coordinate w.r.t. x      
CD1_2   = 2.48038952103902E-05 / partial of first axis coordinate w.r.t. y      
CD2_1   = 2.48038952103902E-05 / partial of second axis coordinate w.r.t. x     
CD2_2   = -1.2504867878693E-05 / partial of second axis coordinate w.r.t. y     
LTV1    =        0.0000000E+00 / offset in X to subsection start                
LTV2    =        0.0000000E+00 / offset in Y to subsection start                
LTM1_1  =                  1.0 / reciprocal of sampling rate in X               
LTM2_2  =                  1.0 / reciprocal of sampling rate in Y               
PA_APER =              116.346 / Position Angle of reference aperture center (de
RA_APER =   5.051964506827E+01 / RA of aperture reference position              
DEC_APER=   4.086445761614E+01 / Declination of aperture reference position     
NCOMBINE=                    2 / number of image sets combined during CR rejecti
CENTERA1=                  513 / subarray axis1 center pt in unbinned dect. pix 
CENTERA2=                  513 / subarray axis2 center pt in unbinned dect. pix 
SIZAXIS1=                 1024 / subarray axis1 size in unbinned detector pixels
SIZAXIS2=                 1024 / subarray axis2 size in unbinned detector pixels
BINAXIS1=                    1 / axis1 data bin size in unbinned detector pixels
BINAXIS2=                    1 / axis2 data bin size in unbinned detector pixels
SAMPNUM =                   14 / MULTIACCUM sample number                       
SAMPTIME=           449.233582 / total integration time (sec)                   
DELTATIM=            50.000412 / integration time of this sample (sec)          
ROUTTIME=   5.653288687777E+04 / UT time of array readout (MJD)                 
TDFTRANS=                    0 / number of TDF transitions during current sample
PODPSFF =                    F / podps fill present (T/F)                       
STDCFFF =                    F / science telemetry fill data present (T=1/F=0)  
STDCFFP = '0x5569'             / science telemetry fill pattern (hex)           
SDQFLAGS=                31743 / serious data quality flags                     
SOFTERRS=                    0 / number of soft error pixels (DQF=1)            
RADESYS = 'ICRS    '                                                            
WCSNAME = 'DRZWCS  '                                                            
HISTORY CCD parameters table:                                                   
HISTORY   reference table iref$t2c16200i_ccd.fits                               
HISTORY     Ground                                                              
HISTORY     Reference data based on Thermal-Vac #3, gain=2.5 results for IR-4   
HISTORY     Readnoise,gain,saturation from TV3,MEB2 values. ISRs 2008-25,39,50  
HISTORY DQICORR complete ...                                                    
HISTORY   DQ array initialized ...                                              
HISTORY   reference table iref$35620330i_bpx.fits                               
HISTORY     INFLIGHT 01/11/2012 12/10/2013                                      
HISTORY     Bad Pixel Table generated using Cycle 20 Flats and Darks----------- 
HISTORY ZSIGCORR complete.                                                      
HISTORY BLEVCORR complete.                                                      
HISTORY   Overscan region table:                                                
HISTORY   reference table iref$q911321mi_osc.fits                               
HISTORY     GROUND                                                              
HISTORY     Initial values for ground test data processing                      
HISTORY ZOFFCORR complete.                                                      
HISTORY Uncertainty array initialized.                                          
HISTORY NLINCORR complete ...                                                   
HISTORY   reference image iref$u1k1727mi_lin.fits                               
HISTORY     INFLIGHT 09/02/2009 12/07/2009                                      
HISTORY     Non-linearity correction from WFC3 MEB1 TV3 and on-orbit frames---- 
HISTORY DARKCORR complete ...                                                   
HISTORY   reference image iref$3562021si_drk.fits                               
HISTORY     INFLIGHT 04/09/2009 17/11/2016                                      
HISTORY     Dark Created from 133 frames spanning cycles 17 to 24-------------- 
HISTORY PHOTCORR complete ...                                                   
HISTORY   reference table iref$wbj1825ri_imp.fits                               
HISTORY     INFLIGHT 08/05/2009 06/04/2012                                      
HISTORY     photometry keywords reference file--------------------------------- 
HISTORY UNITCORR complete.                                                      
HISTORY CRCORR complete.                                                        
HISTORY UNITCORR complete.                                                      
HISTORY FLATCORR complete ...                                                   
HISTORY   reference image iref$uc721145i_pfl.fits                               
HISTORY     INFLIGHT 01/08/2009 30/11/2010                                      
HISTORY     Flat-field created from inflight observations (Aug 2009- Nov 2010). 
HISTORY ============================================================            
HISTORY Header Generation rules:                                                
HISTORY     Rules used to combine headers of input files                        
HISTORY     Start of rules...                                                   
HISTORY ------------------------------------------------------------            
HISTORY !VERSION = 1.1                                                          
HISTORY !INSTRUMENT = WFC3                                                      
HISTORY ROOTNAME                                                                
HISTORY EXTNAME                                                                 
HISTORY EXTVER                                                                  
HISTORY A_0_2                                                                   
HISTORY A_0_3                                                                   
HISTORY A_0_4                                                                   
HISTORY A_1_1                                                                   
HISTORY A_1_2                                                                   
HISTORY A_1_3                                                                   
HISTORY A_2_0                                                                   
HISTORY A_2_1                                                                   
HISTORY A_2_2                                                                   
HISTORY A_3_0                                                                   
HISTORY A_3_1                                                                   
HISTORY A_4_0                                                                   
HISTORY A_ORDER                                                                 
HISTORY APERTURE                                                                
HISTORY ASN_ID                                                                  
HISTORY ASN_MTYP                                                                
HISTORY ASN_TAB                                                                 
HISTORY ATODCORR                                                                
HISTORY ATODGNA                                                                 
HISTORY ATODGNB                                                                 
HISTORY ATODGNC                                                                 
HISTORY ATODGND                                                                 
HISTORY ATODTAB                                                                 
HISTORY B_0_2                                                                   
HISTORY B_0_3                                                                   
HISTORY B_0_4                                                                   
HISTORY B_1_1                                                                   
HISTORY B_1_2                                                                   
HISTORY B_1_3                                                                   
HISTORY B_2_0                                                                   
HISTORY B_2_1                                                                   
HISTORY B_2_2                                                                   
HISTORY B_3_0                                                                   
HISTORY B_3_1                                                                   
HISTORY B_4_0                                                                   
HISTORY B_ORDER                                                                 
HISTORY BADINPDQ                                                                
HISTORY BIASCORR                                                                
HISTORY BIASFILE                                                                
HISTORY BIASLEVA                                                                
HISTORY BIASLEVB                                                                
HISTORY BIASLEVC                                                                
HISTORY BIASLEVD                                                                
HISTORY BINAXIS1                                                                
HISTORY BINAXIS2                                                                
HISTORY BLEVCORR                                                                
HISTORY BPIXTAB                                                                 
HISTORY BUNIT                                                                   
HISTORY CAL_VER                                                                 
HISTORY CCDAMP                                                                  
HISTORY CCDCHIP                                                                 
HISTORY CCDGAIN                                                                 
HISTORY CCDOFSAB                                                                
HISTORY CCDOFSCD                                                                
HISTORY CCDOFSTA                                                                
HISTORY CCDOFSTB                                                                
HISTORY CCDOFSTC                                                                
HISTORY CCDOFSTD                                                                
HISTORY CCDTAB                                                                  
HISTORY CD1_1                                                                   
HISTORY CD1_2                                                                   
HISTORY CD2_1                                                                   
HISTORY CD2_2                                                                   
HISTORY CENTERA1                                                                
HISTORY CENTERA2                                                                
HISTORY CHINJECT                                                                
HISTORY COMPTAB                                                                 
HISTORY CRCORR                                                                  
HISTORY CRMASK                                                                  
HISTORY CRPIX1                                                                  
HISTORY CRPIX2                                                                  
HISTORY CRRADIUS                                                                
HISTORY CRREJTAB                                                                
HISTORY CRSIGMAS                                                                
HISTORY CRSPLIT                                                                 
HISTORY CRTHRESH                                                                
HISTORY CRVAL1                                                                  
HISTORY CRVAL2                                                                  
HISTORY CTEDIR                                                                  
HISTORY CTEIMAGE                                                                
HISTORY CTYPE1                                                                  
HISTORY CTYPE2                                                                  
HISTORY DARKCORR                                                                
HISTORY DARKFILE                                                                
HISTORY DATAMAX                                                                 
HISTORY DATAMIN                                                                 
HISTORY DATE                                                                    
HISTORY DATE-OBS                                                                
HISTORY DEC_APER                                                                
HISTORY DEC_TARG                                                                
HISTORY DELTATIM                                                                
HISTORY DETECTOR                                                                
HISTORY DFLTFILE                                                                
HISTORY DGEOFILE                                                                
HISTORY DIRIMAGE                                                                
HISTORY DQICORR                                                                 
HISTORY DRIZCORR                                                                
HISTORY EQUINOX                                                                 
HISTORY ERRCNT                                                                  
HISTORY EXPEND                                                                  
HISTORY EXPFLAG                                                                 
HISTORY EXPNAME                                                                 
HISTORY EXPSCORR                                                                
HISTORY EXPSTART                                                                
HISTORY EXPTIME                                                                 
HISTORY FGSLOCK                                                                 
HISTORY FILENAME                                                                
HISTORY FILETYPE                                                                
HISTORY FILLCNT                                                                 
HISTORY FILTER                                                                  
HISTORY FLASHCUR                                                                
HISTORY FLASHDUR                                                                
HISTORY FLASHSTA                                                                
HISTORY FLATCORR                                                                
HISTORY FLSHCORR                                                                
HISTORY FLSHFILE                                                                
HISTORY GOODMAX                                                                 
HISTORY GOODMEAN                                                                
HISTORY GOODMIN                                                                 
HISTORY GRAPHTAB                                                                
HISTORY GYROMODE                                                                
HISTORY IDCSCALE                                                                
HISTORY IDCTAB                                                                  
HISTORY IDCTHETA                                                                
HISTORY IDCV2REF                                                                
HISTORY IDCV3REF                                                                
HISTORY IMAGETYP                                                                
HISTORY INHERIT                                                                 
HISTORY INITGUES                                                                
HISTORY INSTRUME                                                                
HISTORY IRAF-TLM                                                                
HISTORY LFLTFILE                                                                
HISTORY LINENUM                                                                 
HISTORY LTM1_1                                                                  
HISTORY LTM2_2                                                                  
HISTORY LTV1                                                                    
HISTORY LTV2                                                                    
HISTORY MDRIZSKY                                                                
HISTORY MDRIZTAB                                                                
HISTORY MEANBLEV                                                                
HISTORY MEANDARK                                                                
HISTORY MEANEXP                                                                 
HISTORY MEANFLSH                                                                
HISTORY MOONANGL                                                                
HISTORY MTFLAG                                                                  
HISTORY NAXIS1                                                                  
HISTORY NAXIS2                                                                  
HISTORY NCOMBINE                                                                
HISTORY NGOODPIX                                                                
HISTORY NLINCORR                                                                
HISTORY NLINFILE                                                                
HISTORY NRPTEXP                                                                 
HISTORY NSAMP                                                                   
HISTORY OBSMODE                                                                 
HISTORY OBSTYPE                                                                 
HISTORY OCD1_1                                                                  
HISTORY OCD1_2                                                                  
HISTORY OCD2_1                                                                  
HISTORY OCD2_2                                                                  
HISTORY OCRPIX1                                                                 
HISTORY OCRPIX2                                                                 
HISTORY OCRVAL1                                                                 
HISTORY OCRVAL2                                                                 
HISTORY OCTYPE1                                                                 
HISTORY OCTYPE2                                                                 
HISTORY OCX10                                                                   
HISTORY OCX11                                                                   
HISTORY OCY10                                                                   
HISTORY OCY11                                                                   
HISTORY ONAXIS1                                                                 
HISTORY ONAXIS2                                                                 
HISTORY OORIENTA                                                                
HISTORY OPUS_VER                                                                
HISTORY ORIENTAT                                                                
HISTORY ORIGIN                                                                  
HISTORY OSCNTAB                                                                 
HISTORY P1_ANGLE                                                                
HISTORY P1_CENTR                                                                
HISTORY P1_FRAME                                                                
HISTORY P1_LSPAC                                                                
HISTORY P1_NPTS                                                                 
HISTORY P1_ORINT                                                                
HISTORY P1_PSPAC                                                                
HISTORY P1_PURPS                                                                
HISTORY P1_SHAPE                                                                
HISTORY P2_ANGLE                                                                
HISTORY P2_CENTR                                                                
HISTORY P2_FRAME                                                                
HISTORY P2_LSPAC                                                                
HISTORY P2_NPTS                                                                 
HISTORY P2_ORINT                                                                
HISTORY P2_PSPAC                                                                
HISTORY P2_PURPS                                                                
HISTORY P2_SHAPE                                                                
HISTORY PA_APER                                                                 
HISTORY PA_V3                                                                   
HISTORY PATTERN1                                                                
HISTORY PATTERN2                                                                
HISTORY PATTSTEP                                                                
HISTORY PFLTFILE                                                                
HISTORY PHOTBW                                                                  
HISTORY PHOTCORR                                                                
HISTORY PHOTFLAM                                                                
HISTORY PHOTFNU                                                                 
HISTORY PHOTMODE                                                                
HISTORY PHOTPLAM                                                                
HISTORY PHOTZPT                                                                 
HISTORY PODPSFF                                                                 
HISTORY POSTARG1                                                                
HISTORY POSTARG2                                                                
HISTORY PR_INV_F                                                                
HISTORY PR_INV_L                                                                
HISTORY PR_INV_M                                                                
HISTORY PRIMESI                                                                 
HISTORY PROCTIME                                                                
HISTORY PROPAPER                                                                
HISTORY PROPOSID                                                                
HISTORY QUALCOM1                                                                
HISTORY QUALCOM2                                                                
HISTORY QUALCOM3                                                                
HISTORY QUALITY                                                                 
HISTORY RA_APER                                                                 
HISTORY RA_TARG                                                                 
HISTORY READNSEA                                                                
HISTORY READNSEB                                                                
HISTORY READNSEC                                                                
HISTORY READNSED                                                                
HISTORY REFFRAME                                                                
HISTORY REJ_RATE                                                                
HISTORY ROUTTIME                                                                
HISTORY RPTCORR                                                                 
HISTORY SAA_DARK                                                                
HISTORY SAA_EXIT                                                                
HISTORY SAA_TIME                                                                
HISTORY SAACRMAP                                                                
HISTORY SAMP_SEQ                                                                
HISTORY SAMPNUM                                                                 
HISTORY SAMPTIME                                                                
HISTORY SAMPZERO                                                                
HISTORY SCALENSE                                                                
HISTORY SCLAMP                                                                  
HISTORY SDQFLAGS                                                                
HISTORY SHADCORR                                                                
HISTORY SHADFILE                                                                
HISTORY SHUTRPOS                                                                
HISTORY SIMPLE                                                                  
HISTORY SIZAXIS1                                                                
HISTORY SIZAXIS2                                                                
HISTORY SKYSUB                                                                  
HISTORY SKYSUM                                                                  
HISTORY SNRMAX                                                                  
HISTORY SNRMEAN                                                                 
HISTORY SNRMIN                                                                  
HISTORY SOFTERRS                                                                
HISTORY STDCFFF                                                                 
HISTORY STDCFFP                                                                 
HISTORY SUBARRAY                                                                
HISTORY SUBTYPE                                                                 
HISTORY SUN_ALT                                                                 
HISTORY SUNANGLE                                                                
HISTORY T_SGSTAR                                                                
HISTORY TARGNAME                                                                
HISTORY TDFTRANS                                                                
HISTORY TELESCOP                                                                
HISTORY TIME-OBS                                                                
HISTORY UNITCORR                                                                
HISTORY VAFACTOR                                                                
HISTORY WCSAXES                                                                 
HISTORY WCSCDATE                                                                
HISTORY ZOFFCORR                                                                
HISTORY ZSIGCORR                                                                
HISTORY APERTURE    APERTURE    multi                                           
HISTORY ASN_ID    ASN_ID    first                                               
HISTORY ASN_MTYP    ASN_MTYP    multi                                           
HISTORY ASN_TAB    ASN_TAB    multi                                             
HISTORY ATODCORR    ATODCORR    multi                                           
HISTORY ATODGNA        ATODGNA        first                                     
HISTORY ATODGNB        ATODGNB        first                                     
HISTORY ATODGNC        ATODGNC        first                                     
HISTORY ATODGND        ATODGND        first                                     
HISTORY ATODTAB    ATODTAB    multi                                             
HISTORY BADINPDQ    BADINPDQ    sum                                             
HISTORY BIASCORR    BIASCORR    multi                                           
HISTORY BIASFILE    BIASFILE    multi                                           
HISTORY BIASLEVA    BIASLEVA    first                                           
HISTORY BIASLEVB    BIASLEVB    first                                           
HISTORY BIASLEVC    BIASLEVC    first                                           
HISTORY BIASLEVD    BIASLEVD    first                                           
HISTORY BINAXIS1    BINAXIS1    first                                           
HISTORY BINAXIS2    BINAXIS2    first                                           
HISTORY BLEVCORR    BLEVCORR    multi                                           
HISTORY BPIXTAB    BPIXTAB    multi                                             
HISTORY BUNIT        BUNIT        first                                         
HISTORY CAL_VER        CAL_VER        first                                     
HISTORY CCDAMP        CCDAMP        first                                       
HISTORY CCDCHIP    CCDCHIP    first                                             
HISTORY CCDGAIN        CCDGAIN        first                                     
HISTORY CCDOFSTA    CCDOFSTA    first                                           
HISTORY CCDOFSTB    CCDOFSTB    first                                           
HISTORY CCDOFSTC    CCDOFSTC    first                                           
HISTORY CCDOFSTD    CCDOFSTD    first                                           
HISTORY CCDTAB      CCDTAB      multi                                           
HISTORY CD1_1    CD1_1    first                                                 
HISTORY CD1_2    CD1_2    first                                                 
HISTORY CD2_1    CD2_1    first                                                 
HISTORY CD2_2    CD2_2    first                                                 
HISTORY CENTERA1    CENTERA1    first                                           
HISTORY CENTERA2    CENTERA2    first                                           
HISTORY CHINJECT    CHINJECT    multi                                           
HISTORY COMPTAB    COMPTAB    multi                                             
HISTORY CRCORR    CRCORR    multi                                               
HISTORY CRMASK    CRMASK    first                                               
HISTORY CRPIX1    CRPIX1    first                                               
HISTORY CRPIX2    CRPIX2    first                                               
HISTORY CRRADIUS    CRRADIUS    first                                           
HISTORY CRREJTAB    CRREJTAB    multi                                           
HISTORY CRSIGMAS    CRSIGMAS    multi                                           
HISTORY CRSPLIT    CRSPLIT    first                                             
HISTORY CRTHRESH    CRTHRESH    first                                           
HISTORY CTEDIR      CTEDIR      multi                                           
HISTORY CTEIMAGE    CTEIMAGE    first                                           
HISTORY CTYPE1    CTYPE1    multi                                               
HISTORY CTYPE2    CTYPE2    multi                                               
HISTORY CRVAL1    CRVAL1    first                                               
HISTORY CRVAL2    CRVAL2    first                                               
HISTORY DARKCORR    DARKCORR    multi                                           
HISTORY DARKFILE    DARKFILE    multi                                           
HISTORY DATE-OBS    DATE-OBS    first                                           
HISTORY DEC_APER    DEC_APER    first                                           
HISTORY DEC_TARG    DEC_TARG    first                                           
HISTORY DELTATIM    DELTATIM    first                                           
HISTORY DETECTOR    DETECTOR    first                                           
HISTORY DFLTFILE    DFLTFILE    multi                                           
HISTORY DGEOFILE    DGEOFILE    multi                                           
HISTORY DIRIMAGE    DIRIMAGE    multi                                           
HISTORY DQICORR    DQICORR    multi                                             
HISTORY DRIZCORR    DRIZCORR    multi                                           
HISTORY EQUINOX    EQUINOX    first                                             
HISTORY EXPEND    EXPEND    max                                                 
HISTORY EXPFLAG    EXPFLAG    multi                                             
HISTORY EXPNAME    EXPNAME    first                                             
HISTORY EXPSCORR    EXPSCORR    multi                                           
HISTORY EXPSTART    EXPSTART    min                                             
HISTORY EXPTIME   EXPTIME   sum                                                 
HISTORY EXPTIME      TEXPTIME    sum                                            
HISTORY EXTVER    EXTVER    first                                               
HISTORY FGSLOCK    FGSLOCK    multi                                             
HISTORY FILENAME    FILENAME    multi                                           
HISTORY FILETYPE    FILETYPE    multi                                           
HISTORY FILTER    FILTER    multi                                               
HISTORY FLASHCUR    FLASHCUR    multi                                           
HISTORY FLASHDUR    FLASHDUR    first                                           
HISTORY FLASHSTA    FLASHSTA    first                                           
HISTORY FLATCORR    FLATCORR    multi                                           
HISTORY FLSHCORR    FLSHCORR    multi                                           
HISTORY FLSHFILE    FLSHFILE    multi                                           
HISTORY GRAPHTAB    GRAPHTAB    multi                                           
HISTORY GYROMODE    GYROMODE    multi                                           
HISTORY IDCTAB    IDCTAB    multi                                               
HISTORY IMAGETYP    IMAGETYP    first                                           
HISTORY INHERIT    INHERIT    first # maintains IRAF compatibility              
HISTORY INITGUES    INITGUES    multi                                           
HISTORY INSTRUME    INSTRUME    first                                           
HISTORY LFLTFILE    LFLTFILE    multi                                           
HISTORY LINENUM    LINENUM    first                                             
HISTORY LTM1_1    LTM1_1    float_one                                           
HISTORY LTM2_2    LTM2_2    float_one                                           
HISTORY LTV1    LTV1    first                                                   
HISTORY LTV2    LTV2    first                                                   
HISTORY MDRIZTAB    MDRIZTAB    multi                                           
HISTORY MEANEXP    MEANEXP    first                                             
HISTORY MEANFLSH    MEANFLSH    first                                           
HISTORY MOONANGL  MOONANGL      first                                           
HISTORY MTFLAG      MTFLAG    first                                             
HISTORY NCOMBINE    NCOMBINE    sum                                             
HISTORY NLINCORR    NLINCORR    multi                                           
HISTORY NLINFILE    NLINFILE    multi                                           
HISTORY NRPTEXP    NRPTEXP    first                                             
HISTORY NSAMP    NSAMP    first                                                 
HISTORY OBSMODE    OBSMODE    multi                                             
HISTORY OBSTYPE       OBSTYPE    first                                          
HISTORY OPUS_VER    OPUS_VER    first                                           
HISTORY ORIENTAT    ORIENTAT    first                                           
HISTORY OSCNTAB    OSCNTAB    multi                                             
HISTORY P1_ANGLE    P1_ANGLE    first                                           
HISTORY P1_CENTR    P1_CENTR    multi                                           
HISTORY P1_FRAME    P1_FRAME    multi                                           
HISTORY P1_LSPAC    P1_LSPAC    first                                           
HISTORY P1_NPTS    P1_NPTS    first                                             
HISTORY P1_ORINT    P1_ORINT    first                                           
HISTORY P1_PSPAC    P1_PSPAC    first                                           
HISTORY P1_PURPS    P1_PURPS    multi                                           
HISTORY P1_SHAPE    P1_SHAPE    multi                                           
HISTORY P2_ANGLE    P2_ANGLE    first                                           
HISTORY P2_CENTR    P2_CENTR    multi                                           
HISTORY P2_FRAME    P2_FRAME    multi                                           
HISTORY P2_LSPAC    P2_LSPAC    first                                           
HISTORY P2_NPTS     P2_NPTS    first                                            
HISTORY P2_ORINT    P2_ORINT    first                                           
HISTORY P2_PSPAC    P2_PSPAC    first                                           
HISTORY P2_PURPS    P2_PURPS    multi                                           
HISTORY P2_SHAPE    P2_SHAPE    multi                                           
HISTORY PA_APER    PA_APER    first                                             
HISTORY PA_V3        PA_V3        first                                         
HISTORY PATTERN1    PATTERN1    multi                                           
HISTORY PATTERN2    PATTERN2    multi                                           
HISTORY PATTSTEP    PATTSTEP    first                                           
HISTORY PFLTFILE    PFLTFILE    multi                                           
HISTORY PHOTBW        PHOTBW        mean                                        
HISTORY PHOTCORR    PHOTCORR    multi                                           
HISTORY PHOTFLAM    PHOTFLAM    mean                                            
HISTORY PHOTFNU    PHOTFNU      mean                                            
HISTORY PHOTMODE    PHOTMODE    first                                           
HISTORY PHOTPLAM    PHOTPLAM    mean                                            
HISTORY PHOTZPT        PHOTZPT        mean                                      
HISTORY PODPSFF    PODPSFF    multi                                             
HISTORY PR_INV_F    PR_INV_F    first                                           
HISTORY PR_INV_L    PR_INV_L    first                                           
HISTORY PR_INV_M    PR_INV_M    first                                           
HISTORY PRIMESI        PRIMESI        multi                                     
HISTORY PROCTIME    PROCTIME    first                                           
HISTORY PROPAPER    PROPAPER    multi                                           
HISTORY PROPOSID    PROPOSID    first                                           
HISTORY QUALCOM1    QUALCOM1    multi                                           
HISTORY QUALCOM2    QUALCOM2    multi                                           
HISTORY QUALCOM3    QUALCOM3    multi                                           
HISTORY QUALITY    QUALITY    multi                                             
HISTORY RA_APER    RA_APER    first                                             
HISTORY RA_TARG        RA_TARG      first                                       
HISTORY READNSEA    READNSEA    first                                           
HISTORY READNSEB    READNSEB    first                                           
HISTORY READNSEC    READNSEC    first                                           
HISTORY READNSED    READNSED    first                                           
HISTORY REFFRAME    REFFRAME    multi                                           
HISTORY ROOTNAME    ROOTNAME    first                                           
HISTORY ROUTTIME    ROUTTIME    first                                           
HISTORY RPTCORR    RPTCORR    multi                                             
HISTORY SAACRMAP    SAACRMAP    multi                                           
HISTORY SAMP_SEQ    SAMP_SEQ    first                                           
HISTORY SAMPNUM    SAMPNUM    first                                             
HISTORY SAMPTIME    SAMPTIME    first                                           
HISTORY SAMPZERO    SAMPZERO    first                                           
HISTORY SCALENSE    SCALENSE    first                                           
HISTORY SCLAMP    SCLAMP    multi                                               
HISTORY SDQFLAGS    SDQFLAGS    first                                           
HISTORY SHADCORR    SHADCORR    multi                                           
HISTORY SHADFILE    SHADFILE    multi                                           
HISTORY SIZAXIS1    SIZAXIS1    first                                           
HISTORY SIZAXIS2    SIZAXIS2    first                                           
HISTORY SOFTERRS    SOFTERRS    sum                                             
HISTORY STDCFFF    STDCFFF    multi                                             
HISTORY STDCFFP    STDCFFP    multi                                             
HISTORY SUBARRAY    SUBARRAY    first                                           
HISTORY SUBTYPE    SUBTYPE    multi                                             
HISTORY SUNANGLE   SUNANGLE   first                                             
HISTORY T_SGSTAR    T_SGSTAR    multi                                           
HISTORY TARGNAME    TARGNAME    first                                           
HISTORY TDFTRANS    TDFTRANS    sum                                             
HISTORY TELESCOP    TELESCOP    first                                           
HISTORY TIME-OBS    TIME-OBS    first                                           
HISTORY UNITCORR    UNITCORR    multi                                           
HISTORY WCSAXES        WCSAXES        first                                     
HISTORY WCSCDATE    WCSCDATE    first                                           
HISTORY WCSNAME        WCSNAME        first                                     
HISTORY ZOFFCORR    ZOFFCORR    multi                                           
HISTORY ZSIGCORR    ZSIGCORR    multi                                           
HISTORY ------------------------------------------------------------            
HISTORY     End of rules...                                                     
HISTORY ============================================================            
HISTORY AstroDrizzle processing performed using:                                
HISTORY     AstroDrizzle Version 2.1.21                                         
HISTORY     Numpy Version 1.11.3                                                
HISTORY     PyFITS Version 2.0.3                                                
END                                                                             


'''