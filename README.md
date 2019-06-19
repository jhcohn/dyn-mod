# dyn-mod

Dynamical Modeling git repo

dyn_cluster/ - files I'm using on the cluster<br/>
    dyn_qcvg.py - file to run emcee with q=True

param_files/ - parameter files to run with dyn_general.py<br/>
    ngc1332_params.txt - defunct<br/>
    ngc3258* - defunct<br/>
    *params.txt - param files for input into dyn_model.py (to run 1 model) or dyn_emcee.py/etc. (to optimize galaxy parameters)<br/>
    *mge.txt - files to input into mge_vcirc_mine.py, to use for calculating the stellar contribution in dyn_model.py<br/>
    *bl_params.txt - param file using Ben's files for the B1 model from Boizelle+2019, using the B1 best-fit parameter input<br/>
    *inex_params.txt - param file using my own fluxmap, lucy files, etc., also using an inexact "educated guess" input<br/>
    *binex_params.txt - param file using Ben's files for the B1 model from Boizelle+2019, but using my "educated guess" for input<br/>
    *out_params.txt - param file with input parameters as the best-fits from specific emcee runs<br/>

dyn.py - defunct

dyn3.py - defunct

dyn_cvg.py - same as dyn_emcee.py, but with convergence test built-in

dyn_emcee.py - code to use dyn_model with emcee to optimize parameters from an input parameter file

dyn_f.py - defunct

dyn_general.py - defunct

dyn_model.py - code to run one model, given an input parameter file

dynxy.py - defunct

grid.py - defunct

mge_vcirc_mine.py - Cappellari's mge vcirc code, adopted slightly for my use
