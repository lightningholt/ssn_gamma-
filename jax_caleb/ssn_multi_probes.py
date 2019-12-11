import os
from sys import platform
import jax.numpy as np
import jax.random as random
from jax import grad, value_and_grad, jit, ops

import matplotlib.pyplot as plt 
import scipy.io as sio
import numpy as onp
import time
from scipy.optimize import minimize

import SSN_classes
import SSN_power_spec
import gamma_SSN_losses as losses
import make_plot
import MakeSSNconnectivity as make_conn
from util import sigmoid_params
 

#the constant (non-optimized) parameters:

#fixed point algorithm:
dt = 1
xtol = 1e-5
Tmax = 500

#power spectrum resolution and range
fnums = 30
freq_range = [15,100]

#SSN parameters
n = 2
k = 0.04
tauE = 30 # in ms
tauI = 10 # in ms
psi = 0.774

t_scale = 1
tau_s = np.array([3, 5, 100])*t_scale #in ms, AMPA, GABA, NMDA current decay time constants
# NMDAratio = 0.4 #NMDA strength as a fraction of E synapse weight

# define the network spatial parameters. Gridsizedeg is the key that determines everything. MagnFactor is biologically observed to be ~2mm/deg. Gridsizedeg = 2 and gridperdeg = 5 means that the network is 11 x 11 neurons (2*5 + 1 x 2*5 + 1)
gridsizedeg = 2
dradius = gridsizedeg/8
gridperdeg = 5
gridsize = round(gridsizedeg*gridperdeg) + 1
magnFactor = 2 #mm/deg
#biological hyper_col length is ~750 um, magFactor is typically 2 mm/deg in macaque V1
# hyper_col = 0.8/magnFactor
hyper_col = 8/magnFactor

#define stimulus conditions r_cent = Radius of the stim, contrasts = contrasts. 
# r_cent = np.array([0.3, 0.6, 0.9, 1.2, 1.5])
#r_cent = np.arange(dradius, round(gridsizedeg/2)+dradius, dradius)
r_cent = np.array([gridsizedeg/2])
# r_cent = np.array([0.9750])
#contrasts = np.array([0, 25, 50, 100])
contrasts = np.array([100])

X,Y, deltaD = make_conn.make_neur_distances(gridsizedeg, gridperdeg, hyper_col, PERIODIC = False)
OMap, _ = make_conn.make_orimap(hyper_col, X, Y, prngKey=22)
Inp, stimCon, _ = make_conn.makeInputs(OMap, r_cent, contrasts, X, Y, gridperdeg=gridperdeg, gridsizedeg=gridsizedeg)
Contrasts = stimCon[0,:]
Radii = stimCon[1,:]

probes = 5
LFPtarget = trgt + np.array( [ii * gridsize for ii in range(probes)])

Ne, Ni = deltaD.shape
tau_vec = np.hstack((tauE*np.ones(Ne), tauI*np.ones(Ni)))
N = Ne + Ni #number of neurons in the grid.
trgt = int(np.floor(Ne/2))

noise_pars = SSN_power_spec.NoisePars(corr_time=10)

gen_inds = np.arange(len(Contrasts))
con_max = (Contrasts == np.max(Contrasts))
rad_max = (Radii == np.max(Radii))

rad_inds = gen_inds[con_max] #I want to look across radii at max contrasts 
con_inds = gen_inds[rad_max] #I want to look across contrasts at max radii
gabor_inds = -1 #the last index is always the gabor at max contrast and max radius

cons = len(con_inds)

def bfgs_multi_gamma(params_init, fname='new_multi.pdf'):
    
    if platform == 'darwin':
        gd_iters = 10
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
        gd_iters = 10
    
    ## minimize instead
    dloss = grad(loss)
    # dloss = value_and_grad(loss)

    def jac_dloss(params):
        gradient = onp.asarray(dloss(params, probes))
        norm_grad.append(onp.linalg.norm(gradient))
        return gradient

    def loss_hist(params):
        ll = onp.asarray(loss(params, probes))
        loss_t.append(ll)
        return ll
    
    params = params_init
    loss_t = []
    norm_grad = []

    t0 = time.time()
    res = minimize(loss_hist, params, method='BFGS', jac=jac_dloss, options={'disp':True})#, 'maxiter':gd_iters})
    
    params = res.x

    print("{} GD steps took {} seconds.".format(gd_iters, time.time()-t0))
    print("fit [Jee, Jei, Jie, Jii, i2e] = ", sigmoid_params(params))
    
    obs_spect, obs_r, _ = save_results_make_plots(params_init, params, loss_t, Contrasts, Inp)
    
    return obs_spect, obs_rates, params, loss_t

def gd_multi_gamma(params_init, eta=0.001, fname='new_gd_multi.pdf'):
    
    dloss = value_and_grad(loss)
    
    if platform == 'darwin':
        gd_iters = 1
    else:
        #os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        gd_iters = 1
        #gd_iters = 3
        
    min_L = []
    min_params = []
    dd = 100 # time scale for scaling eta down.
    
    params = params_init
    loss_t = []
    t0 = time.time()
    
    for ii in range(gd_iters):
        print("G.D. step ", ii+1)
        L, dL = dloss(params, probes)
        params = params - eta * dL #dloss(params)
        loss_t.append(L)

    print("{} GD steps took {} seconds.".format(gd_iters, time.time()-t0))
    if len(params) < 8:
        print("fit [Jee, Jei, Jie, Jii, i2e, Plocal, sigR] = ", sigmoid_params(params, MULTI=True))
    else:
        print("fit [Jee, Jei, Jie, Jii, gE, gI, NMDAratio, Plocal, sigR] = ", sigmoid_params(params, MULTI=True))
        
    obs_spect, obs_r, _ = save_results_make_plots(params_init, params, loss_t, Contrasts, Inp)
    
    return obs_spect, obs_r, params, loss_t

        
def save_results_make_plots(params_init, params, loss_t, Contrasts, Inp, res=None)

    init_spect, _, _, init_r, init_CONVG = ssn_FP(params_init)
    
    #really just want to track the center neurons (E/I)
    init_r = init_r[(trgt, trgt+Ne),:]
    init_spect = init_spect/np.mean(init_spect)

    init_f0 = SSN_power_spec.find_peak_freq(fs, init_spect, len(Contrasts))
    
    target_PS = np.real(np.array(losses.get_multi_probe_spect(fs, fname ='test_spect.mat')))
    target_PS = target_PS/np.mean(target_PS)
    
#     ssn_obs, obs_r, CONVG = ssn_FP(params)
    obs_spect, fs, _, obs_r, CONVG = ssn_FP(params)
    
    obs_r = obs_r[(trgt, trgt+Ne), :]
    obs_spect = obs_spect/np.mean(obs_spect)
    
    obs_f0 = SSN_power_spec.find_peak_freq(fs, obs_spect, len(Contrasts))

    make_plot.Maun_Con_plots(fs, obs_spect, target_PS, Contrasts[con_inds],obs_r[:, con_inds].T, np.reshape(Inp[:,-1], (gridsize, gridsize)), obs_f0, initial_spect=init_spect, initial_rates=init_r[:, con_inds].T, initial_f0= init_f0, fname=fname)
    
    #save the results in dict for savemat
    Results = {
        'obs_spect':obs_spect,
        'obs_rates':obs_rates,
        'obs_f0':obs_f0,
        'CONVG':CONVG,
        'init_spect':init_spect,
        'init_rates':init_r,
        'init_CONVG':init_CONVG,
        'target_spect':target_PS,
        'loss_t':loss_t,
        'params':params,
        'params_init':params_init
        'res':res
    }
    
    
    if fname is not None:
        f_out = fname.split('.')[0]+'.mat'
#         f_out.append('.mat')
        sio.savemat(f_out, Results)
    
    return obs_spect, obs_rates, Results
    

    
    
def ssn_FP(pos_params):
    params = sigmoid_params(pos_params, MULTI=True)
    
    #unpack parameters
    Jee = params[0] * np.pi * psi
    Jei = params[1] * np.pi * psi
    Jie = params[2] * np.pi * psi
    Jii = params[3] * np.pi * psi
    
    if len(params) < 8:
        i2e = params[4]
        Plocal = params[-2]
        sigR = params[-1]
        gE = 1
        gI = 1 * i2e
        NMDAratio = 0.1
    else:
        i2e = 1
        gE = params[4]
        gI = params[5]
        NMDAratio = params[6]
        Plocal = params[-2]
        sigR = params[-1]
    
    W = make_conn.make_full_W(Plocal, Jee, Jei, Jie, Jii, sigR, deltaD, OMap)

    ssn = SSN_classes._SSN_AMPAGABA(tau_s, NMDAratio, n, k, Ne, Ni, tau_vec, W)
    ssn.topos_vec = np.ravel(OMap)
                        
    r_init = np.zeros([ssn.N, len(Contrasts)])
    inp_vec = np.vstack((gE*Inp, gI*Inp))
                        
    r_fp, CONVG = ssn.fixed_point_r(inp_vec, r_init=r_init, Tmax=Tmax, dt=dt, xtol=xtol)
    
    #calculate power spectrum
    if cons == 1:
        spect, fs, f0, _ = SSN_power_spec.linear_power_spect(ssn, r_fp, noise_pas, freq_range, fnums, cons, LFPrange=[LFPtarget[0]])
        
        if np.max(np.abs(np.imag(spect))) > 0.01:
            print("Spectrum is dangerously imaginary")
            
    else:
        spect, fs, f0, _ = SSN_power_spec.linear_PS_sameTime(ssn, r_fp[:, con_inds], noise_pars, freq_range, fnums, cons, LFPrange=[LFPtarget[0]])
        
        if np.max(np.abs(np.imag(spect))) > 0.01:
            print("Spectrum is dangerously imaginary")
        
        outer_spect = make_outer_spect(ssn, r_fp[:,gabor_inds], probes)
        spect = np.real(np.concatenate((spect, outer_spect), axis=1))
    
    return spect, fs, f0, r_fp, CONVG

def loss(params, probes):
    
#     ssn, r_fp, CONVG = ssn_FP(params)
    spect, fs, _, r_fp, CONVG = ssn_FP(params)
    
    if CONVG:
        spect = spect/np.mean(spect)
        
        fs_loss_inds = np.arange(0 , len(fs))
        fs_loss_inds = np.array([freq for freq in fs_loss_inds if fs[freq] >20])
        
        spect_loss = losses.loss_MaunCon_spect(fs[fs_loss_inds], spect[fs_loss_inds,:])
        return spect_loss
    else:
        return np.inf

def make_outer_spect(ssn, rs, probes):
    
    for pp in range(1, probes):
        
        if pp == 1:
            outer_spect, _, _ = SSN_power_spec.linear_power_spect(ssn, rs, noise_pars, freq_range, fnums, LFPrange=[LFPtarget[pp]])
        elif pp == 2:
            spect_2, _, _ = SSN_power_spec.linear_power_spect(ssn, rs, noise_pars, freq_range, fnums, LFPrange=[LFPtarget[pp]])
            outer_spect = np.concatenate((outer_spect[:, None], spect_2[:, None]), axis = 1)
        else:
            spect_2, _, _ = SSN_power_spec.linear_power_spect(ssn, rs, noise_pars, freq_range, fnums, LFPrange=[LFPtarget[pp]])
            outer_spect = np.concatenate((outer_spect, spect_2[:, None]), axis = 1)
        
    return outer_spect

if __name__ == "__main__":
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    from util import find_params_to_sigmoid
    factor_I_hate = 0.774 * np.pi
    #these are the modelo parameters that worked, but I didn't multiply J0 * 0.774 *pi in matlab
    #ideal parameters: [1.95, 1.25, 2.45, 1.5, 1, 1.25, 0.1, 1, any] = [Jee, Jei, Jie, Jii, gE, gI, NMDAratio, Plocal, sigR]
    params_init = np.array([1.95/factor_I_hate, 1.25/factor_I_hate, 2.45/factor_I_hate, 1.5/factor_I_hate, 1.25, 1, 1])
    params_init = find_params_to_sigmoid(params_init, MULTI=True)
    eta = 0.0001

    gd_multi_gamma(params_init, eta)
    # ssn_multi_probes.bfgs_multi_gamma(params_init)