import os
from sys import platform
import time
import jax.numpy as np
from jax import grad, value_and_grad, jit, ops

import SSN_classes
import SSN_power_spec
import gamma_SSN_losses as losses
import make_plot

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
# NMDAratio = 0.4 #NMDA strength as a fraction of E synapse weight

contrasts = np.array([0, 25, 50, 100])
cons = len(contrasts)
lower_bound_rates = 10 * np.ones([2, cons-1])
upper_bound_rates = np.vstack((70*np.ones(cons-1), 100*np.ones(cons-1)))
kink_control = 1 # how quickly log(1 + exp(x)) goes to ~x, where x = target_rates - found_rates    

def full_gd_gamma(params_init, eta):
#     #the constant (non-optimized) parameters:

#     #fixed point algorithm:
#     dt = 1
#     xtol = 1e-6
#     Tmax = 500

#     #power spectrum resolution and range
#     fnums = 30
#     freq_range = [15,100]

#     #SSN parameters
#     n = 2
#     k = 0.04
#     tauE = 20 # in ms
#     tauI = 10 # in ms
#     psi = 0.774

#     t_scale = 1
#     tau_s = np.array([3, 5, 100])*t_scale #in ms, AMPA, GABA, NMDA current decay time constants
#     # NMDAratio = 0.4 #NMDA strength as a fraction of E synapse weight
    
    dloss = value_and_grad(loss)
    
    if platform == 'darwin':
        gd_iters = 10
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        gd_iters = 1000
    
    #eta = .001 #learning rate

    params = params_init
    loss_t = []
    t0 = time.time()
    for ii in range(gd_iters):
        if ii % 100 == 0:
            print("G.D. step ", ii+1)
        L, dL = dloss(params)
        params = params - eta * dL #dloss(params)
        loss_t.append(L)

    print("{} GD steps took {} seconds.".format(gd_iters, time.time()-t0))
    print("fit [Jee, Jei, Jie, Jii, i2e] = ", params)
    
    init_spect, fs, _, init_r = ssn_PS(params_init, contrasts)
    init_spect = np.real(init_spect/np.mean(np.real(init_spect)))
    
    target_PS = losses.get_target_spect(fs)
    target_rates = losses.get_target_rates()
    
    obs_spect, fs, f0, obs_rates = ssn_PS(params, contrasts)
    obs_spect = np.real(obs_spect/np.mean(np.real(obs_spect)))
    
    fname = 'Lzian_Higher_Freqs_GD.pdf'
    
    make_plot.power_spect_rates_plot(fs, obs_spect, target_PS, contrasts, obs_rates.T, target_rates.T, init_spect, init_r.T, lower_bound_rates, upper_bound_rates, fname)
    
    return obs_spect, obs_rates, params, loss_t
   

def ssn_PS(params, contrasts):
    #unpack parameters
    Jee = params[0]
    Jei = params[1]
    Jie = params[2]
    Jii = params[3]
    
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
    
    psi = 0.774
    J2x2 = np.array([[Jee, -Jei], [Jie,  -Jii]]) * np.pi * psi #np.array([[2.5, -1.3], [2.4,  -1.0]]) * np.pi * psi
    ssn = SSN_classes.SSN_2D_AMPAGABA(tau_s, NMDAratio, n,k,tauE,tauI, *np.abs(J2x2).ravel())
    
    r_init = np.zeros([ssn.N, len(contrasts)])
    inp_vec = np.array([[gE], [gI*i2e]]) * contrasts
    
    r_fp = ssn.fixed_point_r(inp_vec, r_init=r_init, Tmax=Tmax, dt=dt, xtol=xtol)
    
    spect, fs, f0, _ = SSN_power_spec.linear_PS_sameTime(ssn, r_fp, SSN_power_spec.NoisePars(), freq_range, fnums, cons)
    
    return spect, fs, f0, r_fp


#@jit
def loss(params):
    spect, fs, obs_f0, r_fp = ssn_PS(params, contrasts) 
    
    if np.max(np.abs(np.imag(spect))) > 0.01:
        print("Spectrum is dangerously imaginary")
    
    #half_width_rates = 20 # basin around acceptable rates 
    #lower_bound_rates = 0 # needs to be > 0, valley will start -lower_bound, 5 is a nice value with kink_control = 5
    #upper_bound_rates = 80 # valley ends at upper_bound, keeps rates from blowing up
    
    prefact_rates = 1
    prefact_params = 10
    
    #fs_loss_inds = np.arange(0 , len(fs))
    #fs_loss_inds = np.array([freq for freq in fs_loss_inds if fs[freq] >20])#np.where(fs > 0, fs_loss_inds, )
#     fs_loss = fs[np.where(fs > 20)]
    
    #spect_loss = losses.loss_spect_contrasts(fs[fs_loss_inds], np.real(spect[fs_loss_inds, :]))
    spect_loss = losses.loss_spect_nonzero_contrasts(fs, spect)
    rates_loss = prefact_rates * losses.loss_rates_contrasts(r_fp[:,1:], lower_bound_rates, upper_bound_rates, kink_control) #fourth arg is slope which is set to 1 normally
    param_loss = prefact_params * losses.loss_params(params)
#     peak_freq_loss = losses.loss_peak_freq(fs, obs_f0)
    
    if spect_loss/rates_loss < 1:
        print('rates loss is greater than spect loss')
#     print(spect_loss/rates_loss) 
    
    return spect_loss + param_loss + rates_loss # + peak_freq_loss #