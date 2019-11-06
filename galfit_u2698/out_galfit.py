from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import AxesGrid

base = '/Users/jonathancohn/Documents/dyn_mod/galfit_u2698/'
out = base + 'imgblock_test43.fits'


with fits.open(out) as hdu:
    print(hdu.info())
    # print(hdu_h[0].header)
    hdr = hdu[0].header
    data = hdu[0].data
    data1 = hdu[1].data
    data2 = hdu[2].data
    data3 = hdu[3].data

data4 = data3 / data1
dataa = [data1, data2, data3, data4]
# 881, 491

'''
for dat in dataa:
    plt.imshow(dat)
    plt.colorbar()
    plt.show()
'''

fig = plt.figure()
grid = AxesGrid(fig, 111,
                nrows_ncols=(1, 3),
                axes_pad=0.01,
                cbar_mode='single',
                cbar_location='right',
                cbar_pad=0.1)

dataset = [data1, data2, data4]
labels = ['Input image', 'Model', 'Residual abs[(data - model) / data]']
i = 0
for ax in grid:
    if i == 0 or i == 1:
        im = ax.imshow(dataset[i], origin='lower', vmin=np.amin([np.nanmin(dataset[0]), np.nanmin(dataset[1])]),  # 0, 1
                       vmax=np.amax([np.nanmax(dataset[0]), np.nanmax(dataset[1])]))  # , cmap='RdBu_r'
    else:
        im = ax.imshow(np.abs(dataset[i]), origin='lower', vmin=np.nanmin(np.abs(dataset[2])), vmax=np.nanmax(np.abs(dataset[2])))
    ax.set_title(labels[i])
    i += 1
    ax.set_xlabel(r'x [pixels]', fontsize=20)  # x [arcsec]
    ax.set_ylabel(r'y [pixels]', fontsize=20)  # y [arcsec]
cbar = ax.cax.colorbar(im)
cbar = grid.cbar_axes[0].colorbar(im)
plt.show()
