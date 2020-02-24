import ssn_multi_probes
from util import find_params_to_sigmoid

import jax.numpy as np
import numpy as onp
import scipy.io as sio
import time

aa = sio.loadmat('hand_selected_ConEffect_1_23.mat')

psi = 0.774 *np.pi
g0=0.44
Wconv = 1

W0 = 1.8
Jxe_min = W0*.5
Jxe_max = W0*1.5 
Jxi_min = W0*.5 * .5
Jxi_max = W0*1.5 * .5


# extract all parameters as 1-dim arrays
Plocal = aa['Params']['Plocal'][0,0][0]
Jee = aa['Params']['Jee'][0, 0][:,0]/psi
Jei = np.abs(aa['Params']['Jei'][0, 0][:,0])/psi #MATLAB had these defined as negative
Jie = aa['Params']['Jie'][0, 0][:,0]/psi
Jii = np.abs(aa['Params']['Jii'][0, 0][:,0])/psi #MATLAB had these defined as negative
I2E = aa['Params']['I2E'][0,0][0]
sigEE = aa['Params']['sigEE'][0,0][:,0]
sigIE = aa['Params']['sigIE'][0,0][:,0]

#convert old ranges to new
J_max = 3

Jee = onp.asarray(Jee)
Jei = onp.asarray(Jei)
Jie = onp.asarray(Jie)
Jii = onp.asarray(Jii)

Jee = onp.where(Jee > Jxe_max, 0.1 * Jxe_min + 0.9 * Jxe_max, Jee)
Jee = onp.where(Jee < Jxe_min, 0.9 * Jxe_min + 0.1 * Jxe_max, Jee)
Jei = onp.where(Jei > Jxi_max, 0.1 * Jxi_min + 0.9 * Jxi_max, Jei)
Jei = onp.where(Jei < Jxi_min, 0.9 * Jxi_min + 0.1 * Jxi_max, Jei)
Jie = onp.where(Jie > Jxe_max, 0.1 * Jxe_min + 0.9 * Jxe_max, Jie)
Jie = onp.where(Jie < Jxe_min, 0.9 * Jxe_min + 0.1 * Jxe_max, Jie)
Jii = onp.where(Jii > Jxi_max, 0.1 * Jxi_min + 0.9 * Jxi_max, Jii)
Jii = onp.where(Jii < Jxi_min, 0.9 * Jxi_min + 0.1 * Jxi_max, Jii)

#hyper Params
diffPS = True
ground_truth = False
OLDSTYLE = False
lamSS = 10
SI = True
psi = 0.774 *np.pi


if diffPS:
    dps = str(1)
else:
    dps = str(0)

if ground_truth:
    gt = str(1)
else:
    gt = str(0)

if SI:
    si = str(1)
else:
    si =str(0)

real_good_inds = np.array([123, 179, 222, 263, 287, 341, 385, 406, 451, 456])

min_loss = 100
min_loss_ind = 0

t0 = time.time()

for rgi in real_good_inds:
#     fname = 'newRange_'+str(rgi)+'_diffPS_'+dps+'_GT_'+gt+'_SI_'+si+'_lamSS_'+str(lamSS)+'.pdf'
    fname = 'sigRF_shrunk_'+str(rgi)+'_diffPS_'+dps+'_GT_'+gt+'_SI_'+si+'_lamSS_'+str(lamSS)+'.pdf'
    hyper_params = {'diffPS':diffPS, 'ground_truth':ground_truth, 'OLDSTYLE':OLDSTYLE, 'SI':SI, 'fname':fname, 'lamSS':lamSS}
    
    params_init = np.array([Jee[rgi], Jei[rgi], Jie[rgi], Jii[rgi], g0, g0*I2E[rgi], 0.4, Plocal[rgi], Plocal[rgi], sigEE[rgi], sigIE[rgi]])
    params_init = find_params_to_sigmoid(params_init, MULTI=True, OLDSTYLE=OLDSTYLE)

    _, _, _, loss_t = ssn_multi_probes.bfgs_multi_gamma(params_init, hyper_params)

    if np.min(loss_t) < min_loss:
        min_loss = np.min(loss_t)
        min_loss_ind = rgi

Results = {'min_loss':min_loss, 'min_loss_ind':min_loss_ind}
sio.savemat('labcompSweep.mat', Results)

tf = time.time - t0
print(tf)