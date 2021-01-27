import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt


def mge_sum(oldmge, intensities, pc_per_ac):
    """
    Calculate total Luminosity of the MGE

    :param mge: MGE file, containing columns: j [component number], I [Lsol,H/pc^2], sigma[arcsec], q[unitless]
    :param pc_per_ac: parsec per arcsec scaling
    :return: the total luminosity from integrating the MGE
    """

    #intensities = []
    sigmas = []
    qs = []
    with open(oldmge, 'r') as mfile:
        for line in mfile:
            if not line.startswith('#'):
                cols = line.split()
                #intensities.append(float(cols[1]))  # cols[0] = component number, cols[1] = intensity[Lsol,H/pc^2]
                sigmas.append(float(cols[1]))  # cols[2] = sigma[arcsec]
                qs.append(float(cols[2]))  # cols[3] = qObs[unitless]

    sum = 0
    for i in range(len(intensities)):
        # volume under 2D gaussian function = 2 pi A sigma_x sigma_y
        sum += 2 * np.pi * intensities[i] * qs[i] * (pc_per_ac * sigmas[i]) ** 2  # convert sigma to pc
    # print(sum)  # Lsol,H

    return sum


def print_new_mge(oldmge, newmge, newintensities, msolold, msolnew):
    sigmas = []
    qs = []
    with open(oldmge, 'r') as mfile:
        for line in mfile:
            if not line.startswith('#'):
                cols = line.split()
                #intensities.append(float(cols[1]))  # cols[0] = component number, cols[1] = intensity[Lsol,H/pc^2]
                sigmas.append(float(cols[1]))  # cols[2] = sigma[arcsec]
                qs.append(float(cols[2]))  # cols[3] = qObs[unitless]

    with open(newmge, 'w+') as nfile:
        nfile.write('# calc from ' + oldmge + ' (msolold=' + str(msolold) + '), with msolnew=' + str(msolnew) + '\n')
        nfile.write('# I [Lsol/pc^2] sigma [arcsec] q\n')
        print(newmge)
        print('I [Lsol/pc^2] sigma [arcsec] q')
        for l in range(len(newintensities)):
            nfile.write(str(newintensities[l]) + ' ' + str(sigmas[l]) + ' ' + str(qs[l]) + '\n')
            print(str(newintensities[l]) + ' | ' + str(sigmas[l]) + ' | ' + str(qs[l]))


def just_print_mge(mgefile):
    with open(mgefile, 'r') as nfile:
        print('I [Lsol/pc^2] sigma [arcsec] q')
        for line in nfile:
            cols = line.split()
            print(cols[1] + ' | ' + cols[2] + ' | ' + cols[3])

# I = (64800/np.pi)**2 * 10 ** (0.4 * (msol - mu))
# mu  = msol - 2.5 * np.log10(I * (np.pi/64800)**2)

def i_to_mu_to_i2(msolnew, msolold, mgefile):

    mu1 = []
    with open(mgefile, 'r') as mge:
        for line in mge:
            if not line.startswith('#'):
                cols = line.split()
                mu1.append(msolold - 2.5 * np.log10(float(cols[0]) * (np.pi/64800)**2))

    i2 = (64800 / np.pi)**2 * 10**(0.4*(msolnew - np.asarray(mu1)))
    print(mgefile)
    print(i2)

    return i2

msol_h_old1216 = 3.32
msol_h_old1271 =  3.33  # ?
msol_v_old1277 = 4.83
# msol_k_old1216 = 3.28  # ?

msol_h_new = 3.37
msol_k_new = 3.27
msol_v_new = 4.87
hk = 0.2
vk = 3.1

ml_h_1216 = 1.3
ml_h_1271 = 1.40
ml_v_1277 = 9.3

pc_per_ac_1277 = 71e6 / 206265.
pc_per_ac_1271 = 80e6 / 206265.
pc_per_ac_1216 = 94e6 / 206265.
pc_per_ac_2698 = 91e6 / 206265.

pwd = '/Users/jonathancohn/Documents/dyn_mod/'

just_print_mge(pwd+'ugc_2698/ugc_2698_rhe_mge.txt')
print(oop)

i1277 = i_to_mu_to_i2(msol_v_new, msol_v_old1277, pwd+'ngc_1277_mge.txt')
i1271 = i_to_mu_to_i2(msol_h_new, msol_h_old1271, pwd+'ngc_1271_mge.txt')
i1216 = i_to_mu_to_i2(msol_h_new, msol_h_old1216, pwd+'mrk_1216_mge.txt')

lvtot1277 = mge_sum(pwd+'ngc_1277_mge.txt', i1277, pc_per_ac_1277) / 1e11
lhtot1271 = mge_sum(pwd+'ngc_1271_mge.txt', i1271, pc_per_ac_1271) / 1e11
lhtot1216 = mge_sum(pwd+'mrk_1216_mge.txt', i1216, pc_per_ac_1216) / 1e11

print('L_V 1277:', lvtot1277, 'e11', 'mass = ', lvtot1277 * ml_v_1277, 'e11')  # 18147530956.87589, 1.68772037899e11
print('L_H 1271:', lhtot1271, 'e11', 'mass = ', lhtot1271 * ml_h_1271, 'e11')  # 74762198796.07321, 1.04667078315e11
print('L_H 1216:', lhtot1216, 'e11', 'mass = ', lhtot1216 * ml_h_1216, 'e11')  # 122529215699.19711, 1.59287980409e11

print_new_mge(pwd+'ngc_1277_mge.txt', pwd+'new_ngc_1277_mge.txt', i1277, msol_v_old1277, msol_v_new)
print_new_mge(pwd+'ngc_1271_mge.txt', pwd+'new_ngc_1271_mge.txt', i1271, msol_h_old1271, msol_h_new)
print_new_mge(pwd+'mrk_1216_mge.txt', pwd+'new_mrk_1216_mge.txt', i1216, msol_h_old1216, msol_h_new)

#print(i1277)
#print(i1271)
#print(i1216)


# ltot_1277 =


'''
hdu = fits.open(pwd + 'ugc_2698/UGC2698_C4_CO21_bri_20.3kms.pbcor.fits')
data_2698 = hdu[0].data[0]  # The mask is stored in hdu_m[0].data, NOT hdu_m[0].data[0]
hdu.close()

for z in range(len(data_2698)):
    print(z)
    fig = plt.figure()
    plt.imshow(data_2698[z], origin='lower')
    cbar = plt.colorbar()
    cbar.set_label(r'Jy', rotation=270, labelpad=20.)
    plt.savefig(pwd + 'pics_ugc2698_channels/' + str(z))
    plt.clf()
print(oop)

'''