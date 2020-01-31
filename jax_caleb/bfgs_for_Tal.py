import sys
import scipy.io as sio
import jax.numpy as np

import ssn_multi_probes
from util import find_params_to_sigmoid
from util import sigmoid_params

ii = int(sys.argv[1])
aa = sio.loadmat('small_sweep_for_Talapas.mat')
psi = np.pi * 0.774

params_init = np.array([aa['Jee'][0][ii]/psi, aa['Jei'][0][ii]/psi, aa['Jie'][0][ii]/psi, aa['Jii'][0][ii]/psi, 1, aa['I2E'][0][ii], 0.1, aa['Plocal'][0][ii], aa['Plocal'][0][ii], aa['sigEE'][0][ii], aa['sigEE'][0][ii]])

if aa['diffPS'][0][ii] > 0:
    diffPS = True
else:
    diffPS = False

if aa['ground_truth'][0][ii] > 0:
    ground_truth = True
else:
    ground_truth = False

if aa['SI'][0][ii] > 0:
    SI = True
else:
    SI = False

lamSS = aa['lamSS'][0][ii]

OLDSTYLE = False

params_init = find_params_to_sigmoid(params_init, MULTI=True, OLDSTYLE=OLDSTYLE)

fname='matlab_'+str(ii)+'_diffPS_'+str(aa['diffPS'][0][ii])+'_groundTruth_'+str(aa['ground_truth'][0][ii] )+'_SI_'+str(aa['SI'][0][ii])+'_lamSS_'+str(lamSS)+'.pdf'

hyper_params = {'diffPS':diffPS, 'ground_truth':ground_truth, 'OLDSTYLE':OLDSTYLE, 'SI':SI, 'fname':fname, 'lamSS':lamSS}

obs_spect, obs_rates, params, loss_t = ssn_multi_probes.bfgs_multi_gamma(params_init, hyper_params)