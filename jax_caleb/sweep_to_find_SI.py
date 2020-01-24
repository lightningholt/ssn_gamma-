import ssn_multi_probes
import jax.numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

from util import find_params_to_sigmoid
from util import sigmoid_params
import make_plot
import SSN_power_spec

from importlib import reload
reload(make_plot)


# Load in the .mat file with all the parameters
aa = sio.loadmat('../../../MATLAB/GammaSharedWithCaleb/ACISS/hand_selected_ConEffect_1_23.mat')
#extract them such that each is a one-dimensional array
Plocal = aa['Params']['Plocal'][0,0][0]
Jee = aa['Params']['Jee'][0, 0][:,0]
Jei = np.abs(aa['Params']['Jei'][0, 0][:,0]) #defined this as neg in MATLAB code
Jie = aa['Params']['Jie'][0, 0][:,0]
Jii = np.abs(aa['Params']['Jii'][0, 0][:,0]) #defined this as neg in MATLAB code
I2E = aa['Params']['I2E'][0,0][0]
sigEE = aa['Params']['sigEE'][0,0][:,0]
sigIE = aa['Params']['sigIE'][0,0][:,0]

#unchanging constant parameters, really just to get r_cent for plotting later
gridsizedeg = 2
gridperdeg = 5
gridsize = round(gridsizedeg*gridperdeg) + 1
magnFactor = 2 #mm/deg
#biological hyper_col length is ~750 um, magFactor is typically 2 mm/deg in macaque V1
# hyper_col = 0.8/magnFactor
hyper_col = 8/magnFactor * 10

#define stimulus conditions r_cent = Radius of the stim, contrasts = contrasts. 
dradius = gridsizedeg/8
r_cent = np.arange(dradius, round(gridsizedeg/2)+dradius, dradius)
# r_cent = np.array([0.3, 0.6, 0.9, 1.2, 1.5])
#r_cent = np.array([gridsizedeg/2])
# r_cent = np.array([0.9750])
contrasts = np.array([0, 25, 50, 100])

# didn't conisder pi * 0.774 in MATLAB code 
psi = 0.774 * np.pi
#trgt is defined for a 11x11 grid
trgt = (60, 181)

#SI = 1 - r_infty / max(r) 
SI_max = 0
ind_SI_max = 0
rad_inds = (3,4,5,6)

for ind in range(Jee.shape[0]):
    ind += 1

    # Jee Jei Jie Jii gE gI NMDAratio plocal sigR (or sigEE sigIE)
    params_init = np.array([Jee[ind]/psi, Jei[ind]/psi, Jie[ind]/psi, Jii[ind]/psi, 1, I2E[ind], 0.1, Plocal[ind], sigEE[ind],  sigIE[ind]])
    OLDSTYLE = False
    params_init = find_params_to_sigmoid(params_init, MULTI=True, OLDSTYLE = OLDSTYLE)

    spect, fs, f0, r_fp, CONVG = ssn_multi_probes.ssn_FP(params_init, OLDSTYLE= OLDSTYLE)
    
    if CONVG:
        r_targ = r_fp[trgt[0],rad_inds]/np.mean(r_fp[trgt[0], rad_inds])
        
        softmax_r = T * logsumexp( r_targ / T ) 
        suppression_index = 1 - (r_targ[-1]/softmax_r)
        
        if suppression_index > SI_max:
            SI_max = suppression_index
            ind_SI_max = ind
        
        ssn_multi_probes.save_results_make_plots(...)
    
    if np.mod(ind, 20) == 0: