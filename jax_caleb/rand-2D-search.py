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


# max and min values of parameters
J_max = 3
i2e_max = 2
gE_max = np.sqrt(10)
gI_min = np.sqrt(0.1)
gI_max = np.sqrt(10) - gI_min #because I do not want gI_min = 0, so I will offset the sigmoid
NMDA_max = 1

contrasts = np.array([0, 25, 50, 100])

num_perms = 100

for nn in range(num_perms):
    key = random.PRNGKey(nn)
    jkey, gkey, nmdakey = random.split(key, 3)
    
    Jee, Jei, Jie, Jii = J_max*random.uniform(jkey, shape=(4,))
    while Jee*Jii > Jei*Jie:
        _, jkey = random.split(jkey)
        Jee, Jei, Jie, Jii = J_max*random.uniform(jkey, shape=(4,))
    
    gE, gI = gI_min + (gI_max- gI_min)*random.uniform(gkey, shape=(2,))
    NMDAratio = NMDA_max*random.uniform(nmdakey, shape=(1,))
    
    params = np.hstack((Jee, Jei, Jie, Jii, gE, gI, NMDAratio))
    params = find_params_to_sigmoid(params, MULTI=False)
    
#     params = np.array([ 3.07743762,  1.69581275,  5.69122332,  1.57580456, -1.43662069, -1.9727109 , -5.59829745])

    spect, fs, f0, r_fp, CONVG = ssn_PS(params, contrasts)
    spect = np.real(spect)/np.mean(np.real(spect))

    f0, hw = SSN_power_spec.infl_find_peak_freq(fs, spect)
    
    params = sigmoid_params(params)
    
    fname = 'rand_2neuron-'+str(nn)
    fsave = fname+'.mat'
    f_fig = fname+'.pdf'
    
    spect = onp.asarray(spect)
    r_fp = onp.asarray(r_fp)
    fs = onp.asarray(fs)
    f0 = onp.asarray(f0)
    hw = onp.asarray(hw)
    params = onp.asarray(params)
    
    
    Results = {'spect':spect,
               'obs_rates':r_fp,
               'CONVG':CONVG,
              'fs':fs,
              'f0':f0,
              'hw':hw,
              'params':params,
              }
    
    sio.savemat(fsave, Results)
    
    obs_fig = make_plot.PS_2D(fs, spect, contrasts, r_fp.T, f0)
    
    with PdfPages(f_fig) as pdf:
        pdf.savefig(obs_fig)
        plt.close(obs_fig)
    