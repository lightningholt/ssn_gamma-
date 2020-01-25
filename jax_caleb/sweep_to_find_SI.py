import ssn_multi_probes
import jax.numpy as np
from jax.scipy.special import logsumexp
import numpy as onp
import numpy as onp
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from util import find_params_to_sigmoid
from util import sigmoid_params
import make_plot
import SSN_power_spec

from importlib import reload
reload(make_plot)


# Load in the .mat file with all the parameters
aa = sio.loadmat('hand_selected_ConEffect_1_23.mat')
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
dx = 2*gridsizedeg/(gridsize -1) 

#define stimulus conditions r_cent = Radius of the stim, contrasts = contrasts. 
dradius = gridsizedeg/8
r_cent = np.arange(0, round(gridsizedeg/2)+dradius, dradius)
rad_inds = (0,3,4,5,6) #this makes the SS plot start at 0, and then go up from there (and hopefully back down)
contrasts = np.array([0, 25, 50, 100])

# didn't conisder pi * 0.774 in MATLAB code 
psi = 0.774 * np.pi
#trgt is defined for a 11x11 grid
Ndim = gridsizedeg * gridperdeg + 1
Ne = Ndim**2
trgt = (int(onp.floor(Ne/2)), int(onp.floor(Ne/2) + Ne))


#SI = 1 - r_infty / max(r) 
SI_max = 0
ind_SI_max = 0
T = 1e-2


for ind in range(1): #Jee.shape[0]
    
    # Jee Jei Jie Jii gE gI NMDAratio plocal sigR (or sigEE sigIE)
    params_init = np.array([Jee[ind]/psi, Jei[ind]/psi, Jie[ind]/psi, Jii[ind]/psi, 1, I2E[ind], 0.1, Plocal[ind], sigEE[ind],  sigIE[ind]])
    OLDSTYLE = False
    params_init = find_params_to_sigmoid(params_init, MULTI=True, OLDSTYLE = OLDSTYLE)

    spect, fs, f0, r_fp, CONVG = ssn_multi_probes.ssn_FP(params_init, OLDSTYLE= OLDSTYLE)
    
    spect = np.real(spect)/np.mean(np.real(spect))
    
    if CONVG:
#         r_targ = r_fp[trgt,rad_inds]/np.mean(r_fp[trgt, rad_inds], axis =1)
        
#         softmax_r = T * logsumexp( r_targ / T ) 
#         suppression_index = 1 - (r_targ[-1]/softmax_r)
        #find suppression index for both E/I cells
        r_targ = r_fp[trgt,:]
        r_targ = r_targ[:, rad_inds]/np.mean(r_targ[:, rad_inds], axis=1)[:,None]
        softmax_r = T * logsumexp( r_targ / T ) 
        suppression_index = 1 - (r_targ[:,-1]/softmax_r)
        
        if suppression_index[0] > SI_max:
            SI_max = suppression_index[0]
            ind_SI_max = ind
        
        f0 = SSN_power_spec.find_peak_freq(fs, spect, 9)

        params = sigmoid_params(params_init, MULTI=True, OLDSTYLE = OLDSTYLE)
        ff = make_plot.Maun_Con_SS(fs, spect, spect, r_fp[trgt, :].T, f0, contrasts, r_cent, params, rad_inds=rad_inds, SI= suppression_index, dx=dx)
        fi = make_plot.Maun_Con_SS(fs, spect, spect, r_fp[trgt, :].T, f0, contrasts, r_cent, params, rad_inds=rad_inds, SI= suppression_index, dx=dx, fignumber= 17)
        
        fname = 'matlab_'+str(ind)+'.pdf'
        
        with PdfPages(fname) as pdf:
            pdf.savefig(ff)
            pdf.savefig(fi)
            plt.close(ff)
            plt.close(fi)
        
        Results = {
            'obs_spect':spect,
            'obs_rates':r_fp,
            'obs_f0':f0,
            'CONVG':CONVG,
            'SI':suppression_index,
            'params':params,
                    }
        f_out = fname.split('.')[0]+'.mat'
#         f_out.append('.mat')
        sio.savemat(f_out, Results)