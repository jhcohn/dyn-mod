import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

point = Point(0.5, 0.5)
polygon = Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])
print(polygon.contains(point))
#from matplotlib.nxutils import points_inside_poly


ugc = '/Users/jonathancohn/Documents/dyn_mod/ugc_2698/'
slicefold = ugc + 'maskslices/'


hdu = fits.open(ugc + 'UGC2698_C4_CO21_bri_20.3kms.pbcor_copy.fits')
data = hdu[0].data


def points_inside_poly(x,y,poly):
    '''
    Decide whether the point is inside the polygon.
    '''
    n = len(poly)
    inside = False

    p1x,p1y = poly[0]
    for i in range(n+1):
        p2x,p2y = poly[i % n]
        if y > min(p1y,p2y):
            if y <= max(p1y,p2y):
                if x <= max(p1x,p2x):
                    if p1y != p2y:
                        xints = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                    if p1x == p2x or x <= xints:
                        inside = not inside
        p1x,p1y = p2x,p2y

    return inside


#grid = points_inside_poly(points, cols)
#grid = grid.reshape((ny, nx))



# NEW 20.3kms cube is 300x300 pixels
nx = 300
ny = 300

x, y = np.meshgrid(np.arange(nx), np.arange(ny))
x, y = x.flatten(), y.flatten()
gridpoints = np.vstack((x,y)).T

mask = np.zeros(shape=(73-29, nx, ny))


for z in range(29, 73):  # 29, 73 [44] (23:71 [48])
    zi = z - 29
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
                        mask[zi,x,y] = polygon.contains(Point(x,y))

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

#'''  #
for j in range(len(mask)):
    print(j)
    plt.imshow(mask[j,:,:], origin='lower')
    plt.pause(1)
# '''  #

fmask = ugc + 'UGC2698_C4_CO21_bri_20.3kms_jonathan_casaimviewhand_strictmask.fits'
fits.writeto(fmask, mask)
