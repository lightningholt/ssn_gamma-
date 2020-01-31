import ssn_multi_probes
from util import find_params_to_sigmoid

import jax.numpy as np
import scipy.io as sio

aa = sio.loadmat('hand_selected_ConEffect_1_23.mat')

# extract all parameters as 1-dim arrays
Plocal = aa['Params']['Plocal'][0,0][0]
Jee = aa['Params']['Jee'][0, 0][:,0]
Jei = np.abs(aa['Params']['Jei'][0, 0][:,0]) #MATLAB had these defined as negative
Jie = aa['Params']['Jie'][0, 0][:,0]
Jii = np.abs(aa['Params']['Jii'][0, 0][:,0]) #MATLAB had these defined as negative
I2E = aa['Params']['I2E'][0,0][0]
sigEE = aa['Params']['sigEE'][0,0][:,0]
sigIE = aa['Params']['sigIE'][0,0][:,0]

#hyper Params
diffPS = True
ground_truth = True
OLDSTYLE = False
lamSS = 10
SI = True
psi = 0.774 *np.pi

real_good_inds = np.array([123, 179, 222, 263, 287, 341, 385, 406, 451, 456])

min_loss = 100
min_loss_ind = 0

for rgi in real_good_inds:
    fname = 'matlab_'+str(rgi)+'_diffPS_'+str(1)+'_GT_'+str(1)+'_SI_'+str(1)+'_lamSS_'+str(lamSS)+'.pdf'
    hyper_params = {'diffPS':diffPS, 'ground_truth':ground_truth, 'OLDSTYLE':OLDSTYLE, 'SI':SI, 'fname':fname, 'lamSS':lamSS}

    params_init = np.array([Jee[rgi]/psi, Jei[rgi]/psi, Jie[rgi]/psi, Jii[rgi]/psi, 1, I2E[rgi], 0.1, Plocal[rgi], Plocal[rgi], sigEE[rgi], sigIE[rgi]])
    params_init = find_params_to_sigmoid(params_init, MULTI=True, OLDSTYLE=OLDSTYLE)

    _, _, _, loss_t = ssn_multi_probes.bfgs_multi_gamma(params_init, hyper_params)

    if np.minimum(loss_t) < min_loss:
        min_loss = np.minimum(loss_t)
        min_loss_ind = rgi

Results = {'min_loss':min_loss, 'min_loss_ind':min_loss_ind}
sio.savemat('labcompSweep.mat', Results)
