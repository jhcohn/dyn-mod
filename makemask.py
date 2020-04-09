import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
# https://stackoverflow.com/questions/36399381/whats-the-fastest-way-of-checking-if-a-point-is-inside-a-polygon-in-python
# http://xingxinghuang.blogspot.com/2014/05/imagebad-pixel-mask-using-ds9-region.html

point = Point(0.5, 0.5)
polygon = Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])
print(polygon.contains(point))
#from matplotlib.nxutils import points_inside_poly


ugc = '/Users/jonathancohn/Documents/dyn_mod/ugc_2698/'
slicefold = ugc + 'maskslices/'  # first attempt
slicefold2 = ugc + 'maskslices2/'  # baseline
slicefoldlax = ugc + 'masksliceslax/'
slicefoldstrict = ugc + 'maskslicesstrict/'


# ''' #
# PLOT FLUX MAP / SAVE TO FITS FILE
hdu = fits.open(ugc + 'UGC2698_C4_CO21_bri_20.3kms.pbcor_copy.fits')
data = hdu[0].data

fmask = ugc + 'UGC2698_C4_CO21_bri_20.3kms_jonathan_casaimviewhand_strictmask.fits'
fmask2 = ugc + 'UGC2698_C4_CO21_bri_20.3kms_jonathan_casaimviewhand_strictmask2.fits'
fmasklax = ugc + 'UGC2698_C4_CO21_bri_20.3kms_jonathan_casaimviewhand_strictmasklax.fits'
fmaskstrict = ugc + 'UGC2698_C4_CO21_bri_20.3kms_jonathan_casaimviewhand_strictmaskstrict.fits'

hdum = fits.open(fmask)
maskdat = hdum[0].data

hdum2 = fits.open(fmask2)
maskdat2 = hdum2[0].data

hduml = fits.open(fmasklax)
maskdatlax = hduml[0].data

hdums = fits.open(fmaskstrict)
maskdatstrict = hdums[0].data

freq1 = float(hdu[0].header['CRVAL3'])  # starting frequency in the data cube
f_step = float(hdu[0].header['CDELT3'])  # frequency step in the data cube  # note: fstep is negative for NGC_3258
f_0 = float(hdu[0].header['RESTFRQ'])
freq_axis = np.arange(freq1, freq1 + ((77-29) * f_step), f_step)  # [bluest, ..., reddest]
# NOTE: For NGC1332, this includes endpoint (arange shouldn't). However, if cut endpoint at fstep-1, arange doesn't
# include it...So, for 1332, include the extra point above, then cutting it off: freq_axis = freq_axis[:-1]
# NOTE: NGC_3258 DATA DOES *NOT* HAVE THIS ISSUE, SOOOOOOOOOO COMMENT OUT FOR NOW!

# Collapse the fluxes! Sum over all slices, multiplying each slice by the slice's mask and by the frequency step
dataclip = data[0]  # [29:78]
collapsed_fluxes = np.zeros(shape=(len(dataclip[0]), len(dataclip[0][0])))
collapsed_mask2 = np.zeros(shape=(len(dataclip[0]), len(dataclip[0][0])))
for zz in range(len(maskdat)):
    collapsed_mask2 += maskdat2[zz]  # just collapse mask
collapsed_mask2[collapsed_mask2 > 0] = 1.
for zz in range(len(maskdat)):
    collapsed_fluxes += dataclip[zz] * maskdat[zz] * abs(f_step)
collapsed_fluxes2 = np.zeros(shape=(len(dataclip[0]), len(dataclip[0][0])))
for zz in range(len(maskdat2)):
    collapsed_fluxes2 += dataclip[zz] * maskdat2[zz] * abs(f_step)
collapsed_fluxeslax = np.zeros(shape=(len(dataclip[0]), len(dataclip[0][0])))
for zz in range(len(maskdatlax)):
    collapsed_fluxeslax += dataclip[zz] * maskdatlax[zz] * abs(f_step)
collapsed_fluxesstrict = np.zeros(shape=(len(dataclip[0]), len(dataclip[0][0])))
for zz in range(len(maskdatstrict)):
    collapsed_fluxesstrict += dataclip[zz] * maskdatstrict[zz] * abs(f_step)

cf = ugc + 'ugc_2698_fluxmap_20.3kms_jonathan_casaimview_strictmask.fits'
cf2 = ugc + 'ugc_2698_fluxmap_20.3kms_jonathan_casaimview_strictmask2.fits'
cflax = ugc + 'ugc_2698_fluxmap_20.3kms_jonathan_casaimview_strictmasklax.fits'
cfstrict = ugc + 'ugc_2698_fluxmap_20.3kms_jonathan_casaimview_strictmaskstrict.fits'
#fits.writeto(cflax, collapsed_fluxeslax)
#fits.writeto(cfstrict, collapsed_fluxesstrict)
#fits.writeto(cf2, collapsed_fluxes2)
#fits.writeto(cf, collapsed_fluxes)
fits.writeto(ugc + 'ugc_2698_collapsemask_20.3kms_jonathan_casaimview_strictmask2.fits', collapsed_mask2)

old_fmap = ugc + 'ugc_2698_summed_model_fluxmap.fits'
hduof = fits.open(old_fmap)
oldf = hduof[0].data
print(np.shape(oldf), np.shape(collapsed_fluxes))

plot5 = True
if plot5:
    fig, axarr = plt.subplots(2, 3, sharex='col', sharey='row', gridspec_kw={'hspace': 0, 'wspace': 0}, figsize=(12,7))

    mn = np.amin([collapsed_fluxes, collapsed_fluxes2, collapsed_fluxeslax, collapsed_fluxesstrict, oldf])
    mx = np.amax([collapsed_fluxes, collapsed_fluxes2, collapsed_fluxeslax, collapsed_fluxesstrict, oldf])
    im = axarr[0,0].imshow(oldf, origin='lower', vmin=mn, vmax=mx, aspect='auto')
    axarr[0,0].text(50, 250, r'Old 20.1kms' +  '\n' + r'cube strictmask', color='w')
    im = axarr[0,1].imshow(collapsed_fluxes, origin='lower', vmin=mn, vmax=mx, aspect='auto')
    axarr[0,1].text(50, 250, r'First strictmask' +  '\n' + r'attempt', color='w')
    im = axarr[0,2].imshow(collapsed_fluxes2, origin='lower', vmin=mn, vmax=mx, aspect='auto')
    axarr[0,2].text(50, 250, r'New baseline' +  '\n' + r'strictmask', color='w')
    im = axarr[1,0].imshow(collapsed_fluxeslax, origin='lower', vmin=mn, vmax=mx, aspect='auto')
    axarr[1,0].text(50, 250, r'Lax strictmask', color='w')
    im = axarr[1,1].imshow(collapsed_fluxesstrict, origin='lower', vmin=mn, vmax=mx, aspect='auto')
    axarr[1,1].text(50, 250, r'Strict strictmask', color='w')
    #plt.subplots_adjust(wspace=0., hspace=0.)
    fig.colorbar(im, ax=axarr.ravel().tolist())
else:
    fig, axarr = plt.subplots(2, 2, sharex='col', sharey='row', gridspec_kw={'hspace': 0, 'wspace': 0}, figsize=(10, 7))

    diffs = (collapsed_fluxes - oldf)
    diffs2 = (collapsed_fluxes2 - oldf)
    diffslax = (collapsed_fluxeslax - oldf)
    diffsstrict = (collapsed_fluxesstrict - oldf)

    mn = np.amin([diffs, -diffs, diffs2, -diffs2, diffslax, -diffslax, diffsstrict, -diffsstrict])
    mx = np.amax([diffs, -diffs, diffs2, -diffs2, diffslax, -diffslax, diffsstrict, -diffsstrict])
    im = axarr[0, 0].imshow(diffs, origin='lower', vmin=mn, vmax=mx, aspect='auto', cmap='RdBu_r')
    axarr[0, 0].text(50, 250, r'First strictmask' + '\n' + r'attempt', color='k')
    im = axarr[0, 1].imshow(diffs2, origin='lower', vmin=mn, vmax=mx, aspect='auto', cmap='RdBu_r')
    axarr[0, 1].text(50, 250, r'New baseline' + '\n' + r'strictmask', color='k')
    im = axarr[1, 0].imshow(diffslax, origin='lower', vmin=mn, vmax=mx, aspect='auto', cmap='RdBu_r')
    axarr[1, 0].text(50, 250, r'Lax strictmask', color='k')
    im = axarr[1, 1].imshow(diffsstrict, origin='lower', vmin=mn, vmax=mx, aspect='auto', cmap='RdBu_r')
    axarr[1, 1].text(50, 250, r'Strict strictmask', color='k')
    fig.suptitle(r'Difference (mask - old 20.1kms cube strictmask)')
    fig.subplots_adjust(top=0.88)
    # plt.subplots_adjust(wspace=0., hspace=0.)
    fig.colorbar(im, ax=axarr.ravel().tolist())

# plt.imshow(collapsed_fluxeslax, origin='lower')
# plt.imshow(oldf, origin='lower')
# plt.colorbar()
plt.show()
print(oop)
# '''  #


# NEW 20.3kms cube is 300x300 pixels
nx = 300
ny = 300

x, y = np.meshgrid(np.arange(nx), np.arange(ny))
x, y = x.flatten(), y.flatten()
gridpoints = np.vstack((x,y)).T

hdu = fits.open(ugc + 'UGC2698_C4_CO21_bri_20.3kms.pbcor_copy.fits')
data = hdu[0].data[0]
print(np.shape(data))

mask = np.zeros(shape=(len(data), nx, ny))


for z in range(len(data)):  # 29, 77 [48] (23:71 [48])
    if 29 <= z <= 77:
        print(z)
        # zi = z - 29
        with open(slicefold + 'slice' + str(z) + 'casa.reg', 'r') as slice:

            for line in slice:
                if line.startswith('polygon'):
                    print(line)
                    line = line[8:-2]  # cut out "polygon(" and ")"
                    cols = [float(p)-1 for p in line.split(',')]  # turn into list of floats
                    corners = []  # pair up the floats as x,y vertices!
                    for i in range(len(cols)):
                        if i%2 == 0:
                            corners.append((cols[i+1], cols[i]))
                        else:
                            pass
                    print(corners)

                    polygon = Polygon(corners)  # create polygon

                    for x in range(len(mask[0])):
                        for y in range(len(mask[0][0])):
                            mask[z,x,y] = polygon.contains(Point(x,y))

                    #print(gridpoints)
                    #print(polygon.contains(gridpoints))


    # By inspection, data[0][i] == slice i on casa
    '''  #
    # Yay! currently good!
    plt.imshow(data[0][z] * 1e3, origin='lower')
    plt.colorbar()
    plt.show()
    plt.imshow(mask[:,:,zi] * data[0][z] * 1e3, origin='lower')
    plt.colorbar()
    plt.show()
    # '''  #

'''  #
for j in range(len(mask)):
    print(j)
    plt.imshow(mask[j,:,:], origin='lower')
    plt.pause(1)
# '''  #

fmask = ugc + 'UGC2698_C4_CO21_bri_20.3kms_jonathan_casaimviewhand_strictmask.fits'
fmask2 = ugc + 'UGC2698_C4_CO21_bri_20.3kms_jonathan_casaimviewhand_strictmask2.fits'
fmasklax = ugc + 'UGC2698_C4_CO21_bri_20.3kms_jonathan_casaimviewhand_strictmasklax.fits'
fmaskstrict = ugc + 'UGC2698_C4_CO21_bri_20.3kms_jonathan_casaimviewhand_strictmaskstrict.fits'
fits.writeto(fmask, mask)
