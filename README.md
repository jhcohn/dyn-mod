# dyn-mod

Dynamical Modeling git repo

galfit_u2698/ - files with info on running galfit, and processing galfit output
 * out_galfit.py - file to display galfit output (superseded by mge_fit_mine.py)
 * psf_adj.py - file for creating many different versions of the H-band images. Also calculate sky region here (see under "DETERMINE SKY BELOW").

dyn_cluster/ - files I'm using on the cluster<br/>
  * dyn_qcvg.py - file to run emcee with q=True
  * dyn_qflat.py - dyn_qcvg.py but with flat sigma
  * dyn_105mod.py - dyn_model.py but with the option of uneven final pixel binning (e.g. 10x5)
  * dyn_u2698_105.py - dyn_qflat.py adjusted for u2698, and using dyn_105mod.py
  * dyndyn2.py - same as dyndyn2.py below but corrected
  * dyndyn3258.py - same as dyndyn2.py here, but specified for NGC 3258 instead of UGC 2698

param_files/ - parameter files to run with dyn_general.py<br/>
  * ngc1332_params.txt - defunct<br/>
  * ngc3258* - defunct<br/>
  * *params.txt - param files for input into dyn_model.py (to run 1 model) or dyn_emcee.py/etc. (to optimize galaxy parameters)<br/>
  * *mge.txt - files to input into mge_vcirc_mine.py, to use for calculating the stellar contribution in dyn_model.py<br/>
  * *bl_params.txt - param file using Ben's files for the B1 model from Boizelle+2019, using the B1 best-fit parameter input<br/>
  * *inex_params.txt - param file using my own fluxmap, lucy files, etc., also using an inexact "educated guess" input<br/>
  * *binex_params.txt - param file using Ben's files for the B1 model from Boizelle+2019, but using my "educated guess" for input<br/>
  * *out_params.txt - param file with input parameters as the best-fits from specific emcee runs<br/>

convert_mge.py - script to convert MGE units between GALFIT units, Cappellari's code units, and paper units, as well as for display on the wiki

dyn.py - defunct

dyn3.py - defunct

dyn_compare_vels.py - script to test stuff, including creating line profiles or comparing models

dyn_cvg.py - same as dyn_emcee.py, but with convergence test built-in; defunct

dyn_datamom.py - same as dynamical_model.py, but modified to create moment maps and PVDs only for the data (i.e. modified to not actually use the input model for anything)

dyn_emcee.py - script to use dyn_model with emcee to optimize parameters from an input parameter file; defunct

dyn_f.py - defunct

dyn_general.py - defunct

dyn_idl.py - same as dynamical_model.py, but for creating voronoi-binned moment maps fir input in the kinemetry code on Jonelle's computer.

dyn_model.py - script to run one model, given an input parameter file; defunct

dyn_moments.py - script to create moment maps from parameter files containing optimized parameters; merged into dynamical_model.py

dyn_oscompare.py - script to compare results for models with different oversampling factors

dyn_outputs.py - script to view outputs from emcee runs; defunct

dyn_par.py - updated version of dyn_writeparfile.py, WIP

dyn_talkfigs.py - same as dynamical_model.py, but modified to make figures used in my prelim talk

dyn_testpars.py - test different combinations of parameters in 2-peaked bimodal distribution from emcee

dyn_vormom.py - same as dynamical_model.py, but modified to calculate moment maps and then average them within voronoi bins; ended up deciding to go with the dyn_moments.py setup instead, where moment maps are averaged within voronoi bins directly, rather than voronoi-binning the cube and then calculating moment maps)

dyn_writeparfile.py - script to automatically generate parameter files (both for local laptop and cluster) and lsf files to run on the cluster

dynamical_model.py - updated script to use to run a single dynamical model

dynben.py - same as dynamical_model.py, but modified to test the chi^2 difference between my and Ben's models

dyndyn.py - script to use dyn_model with dynesty to optimize parameters from an input parameter file, for use on the cluster

dyndyn2.py - same as dyncluster.py, now with corrections; defunct

dyndyncluster.py - same as dyndyn.py, but modified for use on cluster; defunct

dynesty_demo.py - copy of the dynesty demo, in script form

dynesty_out.py - process my dynesty output (load pickle file, make corner/posterior plot, build parameter tables)

dynkin_out.py - script to produce output plots from kinemetry

dynxy.py - defunct

gas_mass.py - script used for working on the gas mass; has been merged into dynamical_model.py; defunct

grid.py - defunct

makemask.py - script used to construct strictmask cubes based on the hand-crafted strictmask region files I build in casa

mge_fit_mine.py. - modified version of Cappellari's MGE Fit Sectors code, adopted slightly for my use

mge_vcirc_mine.py - Cappellari's mge vcirc code, adopted slightly for my use

test_dyn_funcs.py - defunct
