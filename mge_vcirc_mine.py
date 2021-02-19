#!/usr/bin/env python
"""
############################################################################

Copyright (C) 2003-2014, Michele Cappellari
E-mail: michele.cappellari_at_physics.ox.ac.uk

Updated versions of the software are available from my web page
http://purl.org/cappellari/software

If you have found this software useful for your research,
I would appreciate an acknowledgment to the use of the
"JAM modelling package of Cappellari (2008)"

This software is provided as is without any warranty whatsoever.
Permission to use, for non-commercial purposes is granted.
Permission to modify for personal or internal use is granted,
provided this copyright and disclaimer are included unchanged
at the beginning of the file. All other rights are reserved.

############################################################################

NAME:
  MGE_VCIRC

PURPOSE:
   This procedure calculates the circular velocity in the equatorial plane of
   an axisymmetric galaxy model described by a Multi-Gaussian Expansion
   parametrization. This implementation follows the approach described in
   Appendix A of Cappellari et al. (2002, ApJ, 578, 787), which allows for
   quick and accurate calculations also at very small and very large radii.

CALLING SEQUENCE:
   vcirc = mge_vcirc(surf_pot, sigma_pot, qObs_pot,
                     inc_deg, mbh, distance, rad, vcirc, soft=0)

INPUT PARAMETERS:
  SURF_POT: vector of length M containing the peak value of the MGE Gaussians
      describing the galaxy surface density in units of Msun/pc**2 (solar
      masses per parsec**2). This is the MGE model from which the model
      potential is computed.
  SIGMA_POT: vector of length M containing the dispersion in arcseconds of
      the MGE Gaussians describing the galaxy surface density.
  QOBS_POT: vector of length M containing the observed axial ratio of the MGE
      Gaussians describing the galaxy surface density.
  INC_DEG: inclination in degrees (90 being edge-on).
  MBH: Mass of a nuclear supermassive black hole in solar masses.
  DISTANCE: distance of the galaxy in Mpc.
  RAD: Vector of length P with the radius in arcseconds, measured from the
      galaxy centre, at which one wants to compute the model predictions.

KEYWORDS:
  SOFT: Softening length in arcsec for the Keplerian potential of the black
      hole. When this keyword is nonzero the black hole potential will be
      replaced by a Plummer potential with the given scale length.

OUTPUT PARAMETER:
  VCIRC: Vector of length P with the model predictions for the circular
      velocity at the given input radii RAD.

USAGE EXAMPLE:
   A simple usage example is given in the procedure
   TEST_MGE_CIRCULAR_VELOCITY at the end of this file.

REQUIRED ROUTINES:
      By M. Cappellari (included in the JAM distribution):
      - ANY
      - DIFF
      - QUADVA
      - RANGE

MODIFICATION HISTORY:
V1.0.0: Written and tested as part of the implementation of
    Schwarzschild's numerical orbit superposition method described in
    Cappellari et al. (2006). Michele Cappellari, Leiden, 3 February 2003
V3.0.0: This version retains only the few routines required for the computation
    of the circular velocity. All other unnecessary modelling routines have
    been removed. MC, Leiden, 22 November 2005
V3.0.1: Minor code polishing. MC, Oxford, 9 November 2006
V3.0.2: First released version. Included documentation. QUADVA integrator.
    MC, Windhoek, 1 October 2008
V4.0.0: Translated from IDL into Python. MC, Oxford, 10 April 2014
V4.0.1: Support both Python 2.6/2.7 and Python 3.x. MC, Oxford, 25 May 2014
 
"""

from __future__ import print_function

import numpy as np
import argparse
from cap_quadva import quadva  # this is from the jam package!
import matplotlib.pyplot as plt
from scipy import interpolate
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']  # for \text command

#
# The following set of routines computes the R acceleration
# for a density parametrized via the Multi-Gaussian Expansion method.
# The routines are designed to GUARANTEE a maximum relative error of
# 1e-4 in the case of positive Gaussians. This maximum error is reached
# only at the extremes of the usable radial range and only for a very
# flattened Gaussian (q=0.1). Inside the radial range normally adopted
# during orbit integration the error is instead <1e-6.
#
#----------------------------------------------------------------------

def _accelerationR_dRRcapitalh(u, r2, z2, e2, s2):
    #
    # Computes: -D[H[R,z,u],R]/R
    
    u2 = u**2
    p2 = 1. - e2*u2
    us2 = u2/s2
    return np.exp(-0.5*us2*(r2+z2/p2))*us2/np.sqrt(p2) # Cfr. equation (A3)

#----------------------------------------------------------------------

def _accR(R, z, dens, sigma, qintr, bhMass, soft):

    mgepot = np.empty_like(R)
    pot = np.empty_like(dens)
    e2 = 1. - qintr**2
    s2 = sigma**2
    r2 = R**2
    z2 = z**2
    d2 = r2 + z2
    
    for k in range(R.size):
        for j in range(dens.size):
            if (d2[k] < s2[j]/240.**2):
                e = np.sqrt(e2[j]) # pot is Integral in {u,0,1} of -D[H[R,z,u],R]/R at (R,z)=0
                pot[j] = (np.arcsin(e)/e - qintr[j])/(2*e2[j]*s2[j]) # Cfr. equation (A5)
            elif (d2[k] < s2[j]*245**2):
                pot[j] = quadva(_accelerationR_dRRcapitalh, [0.,1.], 
                                args=(r2[k], z2[k], e2[j], s2[j]))[0]
            else: # R acceleration in Keplerian limit (Cappellari et al. 2002)
               pot[j] = np.sqrt(np.pi/2)*sigma[j]/d2[k]**1.5 # Cfr. equation (A4)
        mgepot[k] = np.sum(s2*qintr*dens*pot)
    
    G = 0.00430237    # (km/s)**2 pc/Msun [6.674e-11 SI units (CODATA-14)]
    
    return -R*(4*np.pi*G*mgepot + G*bhMass/(d2 + soft**2)**1.5)

#----------------------------------------------------------------------

def mge_vcirc(surf_pc, sigma_arcsec, qobs, inc_deg, mbh, distance, rad, soft=0.):

    pc = distance*np.pi/0.648 # Constant factor to convert arcsec --> pc
    
    soft_pc = soft*pc           # Convert from arcsec to pc
    Rcirc = rad*pc              # Convert from arcsec to pc
    sigma = sigma_arcsec*pc     # Convert from arcsec to pc
    
    # Axisymmetric deprojection of total mass.
    # See equation (12)-(14) of Cappellari (2008)
    #
    inc = np.radians(inc_deg)      # Convert inclination to radians
    qintr = qobs**2 - np.cos(inc)**2
    if np.any(qintr <= 0.0):
        print(qobs, inc, np.cos(inc))
        raise ValueError('Inclination too low for deprojection')
    qintr = np.sqrt(qintr)/np.sin(inc)
    if np.any(qintr <= 0.05):
        print(qobs, inc, qintr)
        raise ValueError('q < 0.05 components')
    dens = surf_pc*qobs/(qintr*sigma*np.sqrt(2*np.pi))  # MGE deprojection
    
    # Equality of gravitational and centrifugal acceleration accR at z=0
    # R Vphi**2 == accR --> R (vcirc/R)**2 == accR
    #
    accR = _accR(Rcirc, Rcirc*0, dens, sigma, qintr.clip(0.001,0.999), mbh, soft_pc)
    vcirc = np.sqrt(Rcirc*np.abs(accR))  # circular velocity at rcirc
    
    return vcirc

#----------------------------------------------------------------------------

def test_mge_vcirc(surf=None, sigma=None, qObs=None, inc=60., mbh=1e6, distance=10., rad=np.logspace(-1,2,25), ml=5.0,
                   mge_lab=None):
    """
    Usage example for mge_vcirc()
    It takes a fraction of a second on a 2GHz computer
    
    """    
    import matplotlib.pyplot as plt
    from time import clock
    from scipy import interpolate

    '''
    # REPLACE THESE MOCK TEST THINGS WITH ACTUAL INPUT
    # Realistic MGE galaxy surface brightness
    # 
    surf = np.array([39483, 37158, 30646, 17759, 5955.1, 1203.5, 174.36, 21.105, 2.3599, 0.25493])
    sigma = np.array([0.153, 0.515, 1.58, 4.22, 10, 22.4, 48.8, 105, 227, 525])
    qObs = np.array([0.57, 0.57, 0.57, 0.57, 0.57, 0.57, 0.57, 0.57, 0.57, 0.57])
    
    inc = 60. # Inclination in degrees
    mbh = 1e6 # BH mass in solar masses
    distance = 10. # Mpc
    rad = np.logspace(-1,2,25) # Radii in arscec where Vcirc has to be computed
    ml = 5.0 # Adopted M/L ratio
    '''
    
    t = clock()
    if ml is None:
        vcirc = mge_vcirc(surf, sigma, qObs, inc, mbh, distance, rad)
        ylabel = r'$V_{circ}$ [km/s / M/L]'
    else:
        vcirc = mge_vcirc(surf*ml, sigma, qObs, inc, mbh, distance, rad)
        ylabel = r'$V_{circ}$ [km/s], assuming M/L=' + str(ml)

    print('Elapsed time:', clock()-t, ' seconds')

    plt.clf()
    '''  # 
    # COMPARE
    radii = []
    v_circ = []
    with open('/Users/jonathancohn/Documents/dyn_mod/ngc_3258/ngc3258_wfc3mge_vc2mlr.txt') as em:  # note: current file has units v_circ^2/(M/L) --> v_circ = np.sqrt(col * (M/L))
        for line in em:
            cols = line.split()  # note: currently using model "B1" = 2nd col in file (file has 4 cols of models)
            radii.append(206265. * np.arctan(float(cols[0]) / (distance*10**6)))  # file radii in pc, convert to arcsec
            # tan(theta) = pix*0.04*Mpc/arcsec / 31.3 Mpc --> theta = arcsec/rad * arctan(radius [Mpc] / dist)
            v_circ.append(float(cols[1]))  # v^2 / (M/L) --> units (km/s)^2 / (M_sol/L_sol)
    v_c_r = interpolate.interp1d(radii, v_circ, fill_value='extrapolate')  # create a function to interpolate v_circ

    rad2 = np.logspace(-1.7, 1., 20)
    # CALCULATE KEPLERIAN VELOCITY OF ANY POINT (x_disk, y_disk) IN THE DISK WITH RADIUS R (km/s)
    vel = np.sqrt(v_c_r(rad2) * ml)  # velocities sum in quadrature
    plt.plot(rad2, vel, 'k-', label=r'v(R) from numerical integration (B1)')
    # '''

    vc2 = interpolate.interp1d(rad, vcirc, fill_value='extrapolate')
    vcirc2 = vc2(np.logspace(-2, 2., 50))

    # plt.plot(np.logspace(-2., 2., 50), vcirc2, 'r--')
    # plt.plot(radii, np.sqrt(np.asarray(v_circ) * ml), 'k*'
    print(np.amax(vcirc))
    plt.plot(rad, vcirc, 'b-', label=r'MGE: ' + mge_lab)  #  's', markerfacecolor='none',
    plt.xlabel('R (arcsec)')
    plt.ylabel(ylabel)  # r'$V_{circ}$ (km/s)'
    # plt.xscale('log')
    # plt.yscale('log')
    #plt.ylim(1., 10**3.)
    #plt.xlim(np.min(rad), np.max(rad))
    plt.legend()#loc='upper left')  # lower right
    plt.show()


def rsoi(file, inc, mbh, distance, rad, ml, mge_lab, zoom=False):
    fig = plt.figure(figsize=(8,6))
    comp, surf, sigma, qObs = load_mge(file, logged=False)

    vcirc = mge_vcirc(surf * ml, sigma, qObs, inc, 0., distance, rad)  # replaced mbh with 0.
    ylabel = r'$V_{circ}$ [km/s], assuming M/L=' + str(ml)

    vc2 = interpolate.interp1d(rad, vcirc, fill_value='extrapolate')
    vcirc2 = vc2(np.logspace(-2, 2., 50))

    plt.plot(rad, vcirc, 'b--', label=r'MGE: ' + mge_lab)

    mbh_curve = np.sqrt(0.00429897278 * mbh / (rad * distance * 1e6 / 206265))
    idx = np.argwhere(np.diff(np.sign(vcirc - mbh_curve))).flatten()

    plt.plot(rad, mbh_curve, 'k-', label=r'Best-fit $M_{\mathrm{BH}}$')
    plt.xlabel('Radius [arcsec]')
    print(rad[idx])
    plt.axvline(x=rad[idx][0], color='r', label=r'r=' + str(round(rad[idx][0], 2)) + ' arcsec')
    # fiducial: 0.1745", ahe: 0.2117"
    plt.ylabel(ylabel)  # r'$V_{circ}$ (km/s)'
    # plt.xscale('log')
    # plt.yscale('log')
    if zoom:
        plt.xlim(0.04, 8.)
        plt.ylim(75., 750.)
    # plt.ylim(1., 10**3.)
    # plt.xlim(np.min(rad), np.max(rad))
    plt.legend()  # lower right  # loc='upper left'
    plt.show()


def multi_vcirc(files, inc=60., mbh=0., distance=91., rad=np.logspace(-1,2,25), ml=1.0, mge_labs=None, fmts=None,
                zoom=False):
    fig = plt.figure(figsize=(8,6))
    for f in range(len(files)):
        comp, surf, sigma, qObs = load_mge(files[f], logged=False)

        vcirc = mge_vcirc(surf * ml, sigma, qObs, inc[f], 0., distance, rad)  # replaced mbh with 0.
        vc2 = interpolate.interp1d(rad, vcirc, fill_value='extrapolate')
        vcirc2 = vc2(np.logspace(-2, 2., 50))

        plt.plot(rad, vcirc, fmts[f], markerfacecolor='none', label=r'MGE: ' + mge_labs[f])

    mbh_curve = np.sqrt(0.00429897278 * mbh / (rad * distance * 1e6 / 206265))
    idx = np.argwhere(np.diff(np.sign(vcirc - mbh_curve))).flatten()

    #if ml == 1:
    #    plt.ylabel(r'$v_{\text{c}}$ [km s$^{-1}$], assuming $(M/L)_H=M_\odot/L_\odot$')
    #else:
    import matplotlib as mpl
    mpl.rcParams['text.usetex'] = True
    mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']  # for \text command
    if ml == 1.0:
        ml = int(ml)
    plt.ylabel(r'$v_{c,\star}$ [km s$^{-1}$], assuming $(M/L)_H=' + str(ml) + 'M_\odot/L_\odot$')
    plt.xlabel(r'Radius [arcsec]')
    plt.xscale('log')
    # plt.yscale('log')
    if zoom:
        plt.xlim(0.04, 8.)
        plt.ylim(75., 750.)
    # plt.ylim(1., 10**3.)
    # plt.xlim(np.min(rad), np.max(rad))
    plt.legend()  # lower right  # loc='upper left'
    plt.show()

#----------------------------------------------------------------------

def par_dicts(parfile):
    """
    Return dictionaries that contain file names, parameter names, and initial guesses, for free and fixed parameters

    :param parfile: the parameter file
    :return: params (the free parameters), fixed_pars (fixed parameters), files (file names), and priors (prior
        boundaries as {'param_name': [min, max]}) dictionaries
    """

    params = {}
    fixed_pars = {}
    files = {}
    priors = {}

    # READ IN PARAMS FORM THE PARAMETER FILE
    with open(parfile, 'r') as pf:
        for line in pf:
            if line.startswith('Pa'):
                par_names = line.split()[1:]  # ignore the "Param" str in the first column
            elif line.startswith('Primax'):
                primax = line.split()[1:]
            elif line.startswith('Primin'):
                primin = line.split()[1:]
            elif line.startswith('V'):
                par_vals = line.split()[1:]
            elif line.startswith('Other_p'):
                fixed_names = line.split()[1:]
            elif line.startswith('Other_v'):
                fixed_vals = line.split()[1:]
            elif line.startswith('T'):
                file_types = line.split()[1:]
            elif line.startswith('F'):
                file_names = line.split()[1:]

    for n in range(len(par_names)):
        params[par_names[n]] = float(par_vals[n])
        priors[par_names[n]] = [float(primin[n]), float(primax[n])]

    for n in range(len(fixed_names)):
        if fixed_names[n] == 's_type' or fixed_names[n] == 'mtype':
            fixed_pars[fixed_names[n]] = fixed_vals[n]
        elif fixed_names[n] == 'gsize' or fixed_names[n] == 's':
            fixed_pars[fixed_names[n]] = int(fixed_vals[n])
        else:
            fixed_pars[fixed_names[n]] = float(fixed_vals[n])

    for n in range(len(file_types)):
        files[file_types[n]] = file_names[n]

    return params, fixed_pars, files, priors


def load_mge(filename, logged=False):

    comp = []
    surf_pots = []
    sigma_pots = []
    qobs = []
    with open(filename, 'r') as mge_f:
        for line in mge_f:
            if line.startswith('j'):
                logged = True
            if not line.startswith('#') and not line.startswith('j'):
                cols = line.split()
                comp.append(float(cols[0]))
                # if mge_params[1].startswith('log'):  # ie if file reports log10(L/pc^2) instead of L/pc^2
                if logged:  # ie if file reports log10(L/pc^2) instead of L/pc^2
                    surf_pots.append(10 ** float(cols[1]))  # units L_sol/pc^2 --> want M_sol/pc^2 --> multiply by ml
                else:  # if file reports L/pc^2
                    surf_pots.append(float(cols[1]))
                sigma_pots.append(float(cols[2]))
                qobs.append(float(cols[3]))

    return np.asarray(comp), np.asarray(surf_pots), np.asarray(sigma_pots), np.asarray(qobs)


if __name__ == '__main__':
    dm = '/Users/jonathancohn/Documents/dyn_mod/'
    ugc = dm + 'ugc_2698/'
    mdir = '/Users/jonathancohn/Documents/mge/'

    '''  # NGC 4111 STUFF HERE!
    mge_n4111 = dm + 'galfit_u2698/mge_parameters_atlas3d/mge_NGC4111.txt'
    comp, surf, sigma, qObs = load_mge(mge_n4111, logged=False)
    # d = 15.08, 14.6
    import dynamical_model as dm
    # pc_per_ac = 70.782731
    dm.mge_sum(mge_n4111, 14.6 * 1e6 / 206265)
    # print(oop)
    test_mge_vcirc(surf=surf, sigma=sigma, qObs=qObs, inc=84, mbh=0., distance=14.6, rad=np.linspace(0.01, 200, 25),
                   ml=4.487, mge_lab=r'NGC 4111')  # np.linspace(0, 200, 1000)  # np.logspace(-1, 2.3, 20)
    print(oop)
    # '''  #

    mge_rre = ugc + 'ugc_2698_rre_mge.txt'
    mge_rhe = ugc + 'ugc_2698_rhe_mge.txt'
    mge_ahe = ugc + 'ugc_2698_ahe_mge.txt'
    mge_akin = ugc + 'yildirim_table_2698.txt'
    mge_rrepsf = ugc + 'ugc_2698_rrepsf_mge.txt'
    mge_rhepsf = ugc + 'ugc_2698_rhepsf_mge.txt'
    mge_ahepsf = ugc + 'ugc_2698_ahepsf_mge.txt'

    all_mges = [mge_rre, mge_rrepsf, mge_rhe, mge_rhepsf, mge_ahe, mge_ahepsf, mge_akin]
    all_mge_labs = ['reg H, reg mask', 'reg H, dust mask, AGN', 'reg H, dust mask', 'reg H, reg mask, AGN',
                     'dust-corr H, reg mask', 'dust-corr H, reg mask, AGN', 'Akin']
    all_fmts = ['ok', '+k', 'om', '+m', 'ob', '+b', '*r']
    mges = [mge_rre, mge_rhe, mge_ahe, mge_akin]
    mge_labs = ['reg H, reg mask', 'reg H, dust mask', 'dust-corr H, reg mask', 'Akin']
    mges_psf = [mge_rrepsf, mge_rhepsf, mge_ahepsf]
    mge_psflabs = ['reg H, reg mask, AGN', 'reg H, dust mask, AGN', 'dust-corr H, reg mask, AGN']
    fmts = ['+k', 'om', 'sb', '*r']
    fmts_psf = ['+k', 'om', 'sb']

    my_mges = [mge_rre, mge_rhe, mge_ahe]
    my_incs = [67.59, 67.61, 67.56]
    mge_labs = [r'original $H$-band', r'dust-masked $H$-band', r'dust-corrected $H$-band']
    radcomp = np.logspace(-1., 0.5, 1000)  # np.logspace(-.7, -0.6, 1000)  # np.logspace(-.77, -0.75, 50)
#    multi_vcirc(my_mges, inc=67.61, mbh=2461189947.064265, distance=91., rad=radcomp, ml=1.70, mge_labs=mge_labs,
    #multi_vcirc(my_mges, inc=67.56, mbh=3154810151.4633145, distance=91., rad=radcomp, ml=1.58, mge_labs=[r'dust-corrected $H$-band'],
    #            fmts=fmts, zoom=False)  #
    #multi_vcirc(my_mges, inc=67.59, mbh=1708653896.954655, distance=91., rad=radcomp, ml=1.94, mge_labs=[r'original $H$-band'],
    #            fmts=fmts, zoom=False)  #
    multi_vcirc(my_mges, inc=my_incs, mbh=0., distance=91., rad=np.logspace(-1., 0.55, 25), ml=1., mge_labs=mge_labs,  # -1.7, 0.6, 25
                fmts=fmts, zoom=False)
    print(oop)
    multi_vcirc(all_mges, inc=67.6, mbh=0., distance=91., rad=np.logspace(-1., 0.5, 25), ml=1., mge_labs=all_mge_labs,  # -1.7, 0.6, 25
                fmts=all_fmts, zoom=False)
    # multi_vcirc(mges, inc=67.6, mbh=0., distance=91., rad=np.logspace(-1.7, 0.6, 25), ml=1., mge_labs=mge_labs, fmts=fmts,  # -1.7, 0.6, 25
    multi_vcirc(mges, inc=67.6, mbh=0., distance=91., rad=np.logspace(-1., 0.5, 25), ml=1., mge_labs=mge_labs, fmts=fmts,  # -1.7, 0.6, 25
                zoom=False)
    multi_vcirc(mges_psf, inc=67.6, mbh=0., distance=91., rad=np.logspace(-3, 2, 25), ml=1.67, mge_labs=mge_psflabs,
                fmts=fmts_psf, zoom=True)

    #for f in range(len(mges)):
    #    comp, surfpots, sigpots, qobs = load_mge(mges[f], logged=False)
    #
    #    test_mge_vcirc(surf=surfpots, sigma=sigpots, qObs=qobs, inc=68.1, mbh=0., distance=89.,
    #                   rad=np.logspace(-2, 3, 40), ml=None, mge_lab=mge_labs[f])

    '''
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument('--parfile')

    args = vars(parser.parse_args())
    print(args['parfile'])

    params, fixed_pars, files, priors = par_dicts(args['parfile'])

    comp, surf_pots, sigma_pots, qobs = load_mge(files['mge'])
    print(comp, surf_pots, sigma_pots, qobs)

    rad = np.logspace(-3, 1, 30)  # np.logspace(-2, 0.8, 20)
    test_mge_vcirc(surf=surf_pots, sigma=sigma_pots, qObs=qobs,
                   rad=rad, inc=fixed_pars['inc_star'], mbh=0., distance=fixed_pars['dist'],
                   ml=1.)  # params['ml_ratio']
    '''

    # SURF_POT is mge, mbh=0 bc stars only
    # maybe run this code each time for each model inclination angle (assume gas disk and stars have same inc)
    # maybe some diffs at smaller radii than whatever ben was using