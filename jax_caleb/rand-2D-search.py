import jax.numpy as np
import numpy as onp
from jax import random

from bfgs_ssn_gamma_2D import ssn_PS
from util import find_params_to_sigmoid
from util import sigmoid_params
import SSN_power_spec
import make_plot
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from datetime import date
import os

#switch variable between targeted random search around ideal found parameters, and purely random parameters. 
IDEAL = False
print('Ideal =',IDEAL)

if IDEAL:
    dir_beg = 'Targ_2Neuron-search-'
else:
    dir_beg = 'Rand_2Neuron-search-'

dobj = date.today()
dir_ender = dobj.strftime("%y-%m-%d")

dirname = dir_beg+dir_ender
#os.mkdir(dirname)

# max and min values of parameters
if IDEAL: 
    # ideal modelo from previous run
    aa = sio.loadmat('Two_neuron_Results/Gaussian_Slower_learning_rate.mat')
    param_min = 0.9*aa['params'][0]
    param_max = 1.1*aa['params'][0] -  param_min
    
    if len(param_min) < 6:
        param_min = np.hstack((param_min[:4], 1, param_min[4:]))
        param_max = np.hstack((param_max[:4], 0, param_max[4:]))
    
#     Jee_min = param_min[0]
#     Jei_min = param_min[1]
#     Jie_min = param_min[2]
#     Jii_min = param_min[3]
    
#     Jee_max = param_max[0] - Jee_min
#     Jei_max = param_max[1] - Jei_min
#     Jie_max = param_max[2] - Jie_min
#     Jii_max = param_max[3] - Jii_min
        
#     if len(param_min) > 6:
#         gE_min = param_min[4]
#         gI_min = param_min[5]
#         NMDA_min = param_min[6]
        
#         gE_max = param_max[4] - gE_min
#         gI_max = param_max[5] - gI_min
#         NMDA_max = param_max[6] - NMDA_min   
#     else:
#         i2e_min = param_min[4]
#         gE_min = 1
#         gI_min = i2e_min
#         NMDA_min = param_min[5]
        
#         i2e_max = param_max[4] - i2e_min
#         gE_max = 1 - gE_min
#         gI_max = i2e_max - gI_min
#         NMDA_max = param_max[5] - NMDA_min

else:
    J_min = 0
#     Jee_min = J_min
#     Jei_min = J_min
#     Jie_min = J_min
#     Jii_min = J_min
    i2e_min = 0
    gE_min = 0
    gI_min = np.sqrt(0.1)
    NMDA_min = 0
    param_min = np.array([J_min, J_min, J_min, J_min, gE_min, gI_min, NMDA_min])
    
    J_max = 3
    Jee_max = J_max - J_min
    Jei_max = J_max - J_min
    Jie_max = J_max - J_min
    Jii_max = J_max - J_min
    i2e_max = 2 - i2e_min
    gE_max = np.sqrt(10) - gE_min
    gI_max = np.sqrt(10) - gI_min #because I do not want gI_min = 0, so I will offset the sigmoid
    NMDA_max = 1 - NMDA_min
    param_max = np.array([J_max, J_max, J_max, J_max, gE_max, gI_max, NMDA_max])

contrasts = np.array([0, 25, 50, 100])

num_perms = 1000

for nn in range(587, num_perms):
    key = random.PRNGKey(nn)
    
    params = param_min + param_max * random.uniform(key, shape=(7,))
    
    Jee = params[0]
    Jei = params[1]
    Jie = params[2]
    Jii = params[3]
    gE = params[4]
    gI = params[5]
    NMDAratio = params[6]
    
    key, jkey = random.split(key)
    # the next line corresponds  to 
    # Jee*Jii > Jie * Jei  <- not what we want
    while Jee*Jii > Jei*Jie:
        new_params = param_min[:4] + param_max[:4] * random.uniform(jkey, shape=(4,))
        _, jkey = random.split(jkey)
        
        Jee = new_params[0]
        Jei = new_params[1]
        Jie = new_params[2]
        Jii = new_params[3]
    
#     Jee = Jee_min + Jee_max*rand_params[0]
#     Jei = Jei_min + Jei_max*rand_params[1]
#     Jie = Jie_min + Jie_max*rand_params[2]
#     Jii = Jii_min + Jii_max*rand_params[3]
    
#     key, jkey = random.split(key)
#     while Jee*Jii > Jei*Jie:
#         _, jkey = random.split(jkey)
#         new_rand = random.uniform(jkey, shape=(4,))
#         Jee = Jee_min + Jee_max*new_rand[0]
#         Jei = Jei_min + Jei_max*new_rand[1]
#         Jie = Jie_min + Jie_max*new_rand[2]
#         Jii = Jii_min + Jii_max*new_rand[3]
#         #Jee, Jei, Jie, Jii = J_max*random.uniform(jkey, shape=(4,))
    
#     gE = gE_min + gE_max * rand_params[4]
#     gI = gI_min + gI_max * rand_params[5]
#     NMDAratio = NMDA_min + NMDA_max + rand_params[6]
    
    params = np.hstack((Jee, Jei, Jie, Jii, gE, gI, NMDAratio))
    params = find_params_to_sigmoid(params, MULTI=False)
    
#     params = np.array([ 3.07743762,  1.69581275,  5.69122332,  1.57580456, -1.43662069, -1.9727109 , -5.59829745])

    spect, fs, f0, r_fp, CONVG, Jacobian = ssn_PS(params, contrasts)
    spect = np.real(spect)/np.mean(np.real(spect))

    f0, hw, err = SSN_power_spec.infl_find_peak_freq(fs, spect)
    
    params = sigmoid_params(params,  MULTI=False)
    
    if IDEAL:
        fname = 'targeted_2neuron-'+str(nn)
    else:
        fname = 'rand_2neuron-'+str(nn)
    fsave = dirname+'/'+fname+'.mat'
    f_fig = dirname+'/'+fname+'.pdf'
    
    spect = onp.asarray(spect)
    r_fp = onp.asarray(r_fp)
    fs = onp.asarray(fs)
    f0 = onp.asarray(f0)
    hw = onp.asarray(hw)
    params = onp.asarray(params)
    
    
    Results = {'spect':spect,
               'obs_rates':r_fp,
               'CONVG':CONVG,
               'Jacobian':Jacobian,
               'fs':fs,
               'f0':f0,
               'hw':hw,
               'err':err,
               'params':params,
              }
    
    sio.savemat(fsave, Results)
    
    obs_fig = make_plot.PS_2D(fs, spect, contrasts, r_fp.T, f0)
    
    with PdfPages(f_fig) as pdf:
        pdf.savefig(obs_fig)
        plt.close(obs_fig)
    
