import jax.numpy as np
from jax import grad, value_and_grad

from scipy.optimize import minimize
import scipy.io as sio
import numpy as onp
import os
from sys import platform
import time

import SSN_classes
import SSN_power_spec
import gamma_SSN_losses as losses
import make_plot
from util import sigmoid_params

#the constant (non-optimized) parameters:

#fixed point algorithm:
dt = 1
xtol = 1e-6
Tmax = 500

#power spectrum resolution and range
fnums = 30
freq_range = [15,100]

#SSN parameters
n = 2
k = 0.04
tauE = 20 # in ms
tauI = 10 # in ms
psi = 0.774

t_scale = 1
tau_s = np.array([3, 5, 100])*t_scale #in ms, AMPA, GABA, NMDA current decay time constants

contrasts = np.array([0, 25, 50, 100])
cons = len(contrasts)
lower_bound_rates = -5 * np.ones([2, cons-1])
upper_bound_rates = np.vstack((70*np.ones(cons-1), 100*np.ones(cons-1)))

kink_control = 1 # how quickly log(1 + exp(x)) goes to ~x, where x = target_rates - found_rates    

def bfgs_gamma(params_init, fname='new_fig.pdf'):
    
    if platform == 'darwin':
        gd_iters = 10
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        gd_iters = 1000
        
    ## minimize instead
    dloss = grad(loss)
    # dloss = value_and_grad(loss)

    def jac_dloss(params):
        gradient = onp.asarray(dloss(params))
        norm_grad.append(onp.linalg.norm(gradient))
        return gradient

    def loss_hist(params):
        ll = onp.asarray(loss(params))
        loss_t.append(ll)
        return ll

    # gd_iters = 1000

    params = params_init
    loss_t = []
    norm_grad = []

    #def c_back(xk):
        #loss_t.append(xk.fun)
        #norm_grad.append(np.norm(xk.jac))


    t0 = time.time()

    res = minimize(loss_hist, params, method='BFGS', jac=jac_dloss, options={'disp':True})#, 'maxiter':gd_iters})

    params = res.x

    print("{} GD steps took {} seconds.".format(gd_iters, time.time()-t0))
    print("fit [Jee, Jei, Jie, Jii, i2e] = ", sigmoid_params(params, MULTI=False))
    
    init_spect, fs, _, init_r, _ = ssn_PS(params_init, contrasts)
    init_spect = np.real(init_spect/np.mean(np.real(init_spect)))
    
    target_PS = losses.get_target_spect(fs, ground_truth=True)
    target_rates = losses.get_target_rates()
    
    obs_spect, fs, obs_f0, obs_rates, _, _ = ssn_PS(params, contrasts)
    obs_spect = np.real(obs_spect/np.mean(np.real(obs_spect)))
    
    #fname = 'Lzian_Higher_Freqs_Wider_Peaks_GD.pdf'
    Results = {
        'obs_spect':obs_spect,
        'obs_rates':obs_rates,
        'obs_f0':obs_f0,
        'init_spect':init_spect,
        'init_rates':init_r,
        'target_spect':target_PS,
        'target_rates':target_rates,
        'lower_bound_rates':lower_bound_rates,
        'upper_bound_rates':upper_bound_rates,
        'kink_control':kink_control,
        'loss_t':loss_t,
        'params':params,
        'res':res,
    }
    if fname is not None:
        f_out = fname.split('.')[0]+'.mat'
#         f_out.append('.mat')
        sio.savemat(f_out, Results)
    
    make_plot.power_spect_rates_plot(fs, obs_spect, target_PS, contrasts, obs_rates.T, target_rates.T, init_spect, init_r.T, lower_bound_rates, upper_bound_rates, fname)
    
    return obs_spect, obs_rates, params, loss_t

def ssn_PS(pos_params, contrasts):
    params = sigmoid_params(pos_params, MULTI=False)
    
    #unpack parameters
    Jee = params[0] * np.pi * psi
    Jei = params[1] * np.pi * psi
    Jie = params[2] * np.pi * psi
    Jii = params[3] * np.pi * psi
    
    if len(params) < 6:
        i2e = params[4]
        gE = 1
        gI = 1
        NMDAratio = 0.4
    else:
        i2e = 1
        gE = params[4]
        gI = params[5]
        NMDAratio = params[6]
    
    
    cons = len(contrasts)

    #J2x2 = np.array([[Jee, -Jei], [Jie,  -Jii]]) * np.pi * psi #np.array([[2.5, -1.3], [2.4,  -1.0]]) * np.pi * psi
    #ssn = SSN_classes.SSN_2D_AMPAGABA(tau_s, NMDAratio, n,k,tauE,tauI, *np.abs(J2x2).ravel())

    ssn = SSN_classes.SSN_2D_AMPAGABA(tau_s, NMDAratio, n,k, tauE, tauI, Jee, Jei, Jie, Jii)
    
    r_init = np.zeros([ssn.N, len(contrasts)])
    inp_vec = np.array([[gE], [gI*i2e]]) * contrasts
    
    r_fp, CONVG = ssn.fixed_point_r(inp_vec, r_init=r_init, Tmax=Tmax, dt=dt, xtol=xtol)
    
    spect, fs, f0, Jacob = SSN_power_spec.linear_PS_sameTime(ssn, r_fp, SSN_power_spec.NoisePars(), freq_range, fnums, cons)
    
    return spect, fs, f0, r_fp, CONVG, Jacob

def loss(params):
    contrasts = np.array([0, 25, 50, 100])
    spect, fs, f0, r_fp, CONVG, _ = ssn_PS(params, contrasts)
    
    if CONVG:
    
        if np.max(np.abs(np.imag(spect))) > 0.01:
            print("Spectrum is dangerously imaginary")
            
        cons = len(contrasts)
        lower_bound_rates = -5 * np.ones([2, cons-1])
        upper_bound_rates = np.vstack((70*np.ones(cons-1), 100*np.ones(cons-1)))
        kink_control = 1 # how quickly log(1 + exp(x)) goes to ~x, where x = target_rates - found_rates    

        prefact_rates = 1
        prefact_params = 10

        fs_loss_inds = np.arange(0 , len(fs))
        fs_loss_inds = np.array([freq for freq in fs_loss_inds if fs[freq] >20])
        spect_loss = losses.loss_spect_nonzero_contrasts(fs[fs_loss_inds], spect[fs_loss_inds,:])

#         spect_loss = losses.loss_spect_contrasts(fs, np.real(spect))
        rates_loss = prefact_rates * losses.loss_rates_contrasts(r_fp[:,1:], lower_bound_rates, upper_bound_rates, kink_control) #fourth arg is slope which is set to 1 normally
        #param_loss = prefact_params * losses.loss_params(params)
        return spect_loss + rates_loss# + param_loss
    else:
        return np.inf