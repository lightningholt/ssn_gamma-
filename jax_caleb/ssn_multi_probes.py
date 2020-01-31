import os
from sys import platform
import jax.numpy as np
import jax.random as random
from jax.scipy.special import logsumexp
from jax import grad, value_and_grad, jit, ops

import matplotlib.pyplot as plt 
import scipy.io as sio
import numpy as onp
import time
from scipy.optimize import minimize
from matplotlib.backends.backend_pdf import PdfPages

import SSN_classes
import SSN_power_spec
import gamma_SSN_losses as losses
import make_plot
import MakeSSNconnectivity as make_conn
from util import sigmoid_params


#the constant (non-optimized) parameters:
#defined here such that they're global (available to all function calls).

#fixed point algorithm:
dt = 1
xtol = 1e-4
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
gridperdeg = 5
gridsize = round(gridsizedeg*gridperdeg) + 1
magnFactor = 2 #mm/deg
dx = 2*gridsizedeg/(gridsize -1) 
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
#contrasts = np.array([100])

# Make spatial/topo network
#X, Y meshgrids of x (y)-position, deltaD = matrix of differencs between network neurons
X,Y, deltaD = make_conn.make_neur_distances(gridsizedeg, gridperdeg, hyper_col, PERIODIC = False)
# OMap = orientation map
OMap, _ = make_conn.make_orimap(hyper_col, X, Y, prngKey=4)

#number of neurons and time constants
Ne, Ni = deltaD.shape
tau_vec = np.hstack((tauE*np.ones(Ne), tauI*np.ones(Ni)))
N = Ne + Ni #number of neurons in the grid.
#trgt is the middle neuron
trgt = int(np.floor(Ne/2))

# decide on number of probes, and find locations of those cells
probes = 5
LFPtarget = trgt + np.array( [ii * gridsize for ii in range(probes)])

# Inp = N**2 x stimulus conditions (typically 8 - one for each contrasts and radius (shared at max con and max rad) which gives 7 conditions + gabor)
Inp, stimCon, _ = make_conn.makeInputs(OMap, r_cent, contrasts, X, Y, gridperdeg=gridperdeg, gridsizedeg=gridsizedeg)
# re-Define contrasts/radius to find indices for max contrast/ radius to make SS curves and such things later
Contrasts = stimCon[0,:]
Radii = stimCon[1,:]

# indices to find max 
gen_inds = np.arange(len(Contrasts))
con_max = (Contrasts == np.max(Contrasts))
rad_max = (Radii == np.max(Radii))

rad_inds = gen_inds[con_max] #I want to look across radii at max contrasts 
rad_inds = np.hstack((0, rad_inds)) #adding in the 0 contrast conditions
con_inds = gen_inds[rad_max] #I want to look across contrasts at max radii
gabor_inds = -1 #the last index is always the gabor at max contrast and max gabor sigma
cons = len(con_inds)

#define noise_parameters
noise_pars = SSN_power_spec.NoisePars(corr_time=10)

### FUNCTION DEF BELOW

def bfgs_multi_gamma(params_init, hyper_params):
    '''
    Fcn that does gradient descent using BFGS. 
    returns
    observed means found using SSN
    observed spectrum, observed rates, fit parameters, loss over time
    
    inputs
    params_init = initial parameters
    hyper_parameters = bunch of switches that could be changed-- see below
    '''
    
    diffPS = hyper_params['diffPS'] #fit using differences in PS (difffPS = True) or not (false)
    ground_truth = hyper_params['ground_truth'] # fit using groud_truth = True - PS found using MATLAB and SSN, or using idealized PS 
    OLDSTYLE = hyper_params['OLDSTYLE'] # OLDSTYLE = True - extended gE and gI ranges sqrt([0.1, 10]) , False - previous gE and gI ranges [0.5, 2]
    lamSS = hyper_params['lamSS'] # lamSS - weighting factor of Surround Suppression loss
    SI = hyper_params['SI'] # SI = true, SS loss fcn uses Suppression Index; False, SS loss fcn uses idealized rates --- True is better
    fname = hyper_params['fname'] # fname = file name to save .pdf and .mat files
    
    gd_iters = 10 #I think it needs this defined. 
    # if using my computer, limit gd_iters otherwise not.
#     if platform == 'darwin':
#         gd_iters = 10
#     else:
#         os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#         os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
#         gd_iters = 100
    
    ## minimize instead
    dloss = grad(loss) #jax grad to find jac for BFGS
    # dloss = value_and_grad(loss) #can't use this for jac for BFGS because it gives multiple outputs
    
    # define this jac_dloss to pass to BFGS and track the norm of the grad. -- how to minimize the obj
    def jac_dloss(params):
        gradient = onp.asarray(dloss(params, probes, lamSS = lamSS, SI = SI, ground_truth = ground_truth, diffPS = diffPS, OLDSTYLE = OLDSTYLE))
        norm_grad.append(onp.linalg.norm(gradient))
        return gradient
    
    #define this loss_hist to track the loss over time. -- obj to be minimized
    def loss_hist(params):
        ll = onp.asarray(loss(params, probes, lamSS = lamSS, SI = SI, ground_truth = ground_truth, diffPS = diffPS, OLDSTYLE = OLDSTYLE))
        loss_t.append(ll)
        return ll
    
    loss_t = []
    norm_grad = []

    t0 = time.time()
    res = minimize(loss_hist, params_init, method='BFGS', jac=jac_dloss, options={'disp':True})#, 'maxiter':gd_iters})
    
    params = res.x
    
    t_elapsed = time.time() - t0
    
#     print("{} GD steps took {} seconds.".format(gd_iters, time.time()-t0))
#     print("fit [Jee, Jei, Jie, Jii, gE, gI, NMDAratio, plocal, sigEE, sigIE] = ", sigmoid_params(params, MULTI=True, OLDSTYLE = OLDSTYLE))
    
    #fcn to save results and make plots... probably don't need this comment
    obs_spect, obs_r, _ = save_results_make_plots(params_init, params, loss_t, Contrasts, Inp, hyper_params, fname=fname, res=res, ground_truth=ground_truth, OLDSTYLE = OLDSTYLE, tf = t_elapsed)
    
    return obs_spect, obs_r, params, loss_t

def gd_multi_gamma(params_init, hyper_params):
    '''
    Fcn that does gradient descent using learning-rate annealing GD
    returns
    observed means found using SSN
    observed spectrum, observed rates, fit parameters, loss over time
    
    inputs
    params_init = initial parameters
    hyper_parameters = bunch of switches that could be changed-- see below
    '''
    
    diffPS = hyper_params['diffPS'] #fit using differences in PS (difffPS = True) or not (false)
    ground_truth = hyper_params['ground_truth'] # fit using groud_truth = True - PS found using MATLAB and SSN, or using idealized PS 
    OLDSTYLE = hyper_params['OLDSTYLE'] # OLDSTYLE = True - extended gE and gI ranges sqrt([0.1, 10]) , False - previous gE and gI ranges [0.5, 2]
    lamSS = hyper_params['lamSS'] # lamSS - weighting factor of Surround Suppression loss
    SI = hyper_params['SI'] # SI = true, SS loss fcn uses Suppression Index; False, SS loss fcn uses idealized rates --- True is better
    fname = hyper_params['fname'] # fname = file name to save .pdf and .mat files
    eta = hyper_params['eta'] # eta = learning rate 
    
    #find loss and gradient using jax
    dloss = value_and_grad(loss)
    
    #switch between my computer and lab comp -- easiest using platform
    if platform == 'darwin': #this is apple for some reason
        gd_iters = 1
    else:
        #os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        gd_iters = 1
        #gd_iters = 3
        
    #define lists to append loss and minimum parameters
    min_L = []
    min_params = []
    dd = 100 # time scale for scaling eta down. annealing time constant 
    loss_t = []
    
    #define params as initial parameters - will re-write later
    params = params_init
    
    t0 = time.time()
    
    #GD algorithm
    for ii in range(gd_iters):
        print("G.D. step ", ii+1)
        L, dL = dloss(params, probes,  lamSS = lamSS, SI = SI, ground_truth = ground_truth, diffPS = diffPS, OLDSTYLE = OLDSTYLE)
        params = params - eta * dL #dloss(params)
        loss_t.append(L)
    
    print("{} GD steps took {} seconds.".format(gd_iters, time.time()-t0))
    if len(params) < 8:
        print("fit [Jee, Jei, Jie, Jii, i2e, Plocal, sigR] = ", sigmoid_params(params, MULTI=True, OLDSTYLE=OLDSTYLE))
    else:
        print("fit [Jee, Jei, Jie, Jii, gE, gI, NMDAratio, Plocal, sigR] = ", sigmoid_params(params, MULTI=True, OLDSTYLE=OLDSTYLE))
        
    obs_spect, obs_r, _ = save_results_make_plots(params_init, params, loss_t, Contrasts, Inp,  hyper_params, fname=fname)
    
    return obs_spect, obs_r, params, loss_t

def loss(params, probes, lamSS = 2, SI = True, ground_truth = True, diffPS = False, OLDSTYLE = False):
    '''
    returns a single valued loss based on the surround suppression (reLU on SI - if SI = true, MSE between obs and idealized rates if SI = False) and power spectrum (MSE between obs and idealized PS)
    inputs
    params = parameters to ssn [Jee, Jei, Jie, Jii, gE, gI, NMDAratio, plocal, sigEE, sigIE ] 
    probes = 5 (typically), number of probes to calcule PS of Gabor input
    lamSS = weighting factor of SS vs PS loss
    SI = whether to use reLU on Suppression Index, or MSE on obs vs idealized rates
    ground_truth = whether to use PS found from MATLAB SSN or something similiar to Ray-Maunsell's plot
    OLDSTYLE = whether to use extended gE/gI range sqrt([0.1, 10]), or old limited one [0.5, 2]
    
    '''
    
    #ssn_FP finds fixed point and PS 
    spect, fs, _, r_fp, CONVG = ssn_FP(params, OLDSTYLE = OLDSTYLE)
    
    #this PS_inds is just in case I don't want to fit all PS curves. I could just fit the non-Maunsell curves for instance
    PS_inds = np.arange(spect.shape[1])
    #PS_inds = np.arange(len(con_inds)) #just Contrast effect
    
    suppression_index_loss = losses.loss_rates_SurrSupp(r_fp[trgt, rad_inds[:-1]],  SI = SI) # the -1 is to not include the gabor
        
    
    if CONVG:
        #spect = spect/np.mean(spect)
        
        #find spect_loss
        fs_loss_inds = np.arange(0 , len(fs))
        fs_loss_inds = np.array([freq for freq in fs_loss_inds if fs[freq] >20])
        
        spect_loss, _ = losses.loss_MaunCon_spect(fs[fs_loss_inds], spect[fs_loss_inds,:], PS_inds, ground_truth = ground_truth, diffPS= diffPS)
        
        #find rates bounding loss -- rates_loss
        #cause I don't fit the contrast = 0 case
        stimuli = len(Contrasts)-1
        lower_bound_rates = -5 * np.ones([N, stimuli])
        upper_bound_rates = np.vstack((70*np.ones((Ne, stimuli)), 100*np.ones((Ni, stimuli))))
        kink_control = 1 # how quickly log(1 + exp(x)) goes to ~x, where x = target_rates - found_rates    

        prefact_rates = 1
        #do not fit the contrast = 0 case -- which is 0 by definition
        rates_loss = prefact_rates * losses.loss_rates_contrasts(r_fp[:,1:], lower_bound_rates, upper_bound_rates, kink_control) #fourth arg is slope which is set to 1 normally
        
        return spect_loss + lamSS * suppression_index_loss + rates_loss
    else:
        return np.inf
    
# SSN_FP finds the PS and fixed point
def ssn_FP(pos_params, OLDSTYLE):
    ''' 
    Fcn that finds the fixed point and PS of the given SSN network. 
    returns spect, frequencies used to find that spect, peak frequencies (f0), fixed point rates (r_fp), and CONVG == 5 outputs
    
    inputs 
    pos_params are params that once sigmoided are always positive. 
    OLDSTYLE = old ranges for gE/gI in sigmoid params fcn
    '''
    
    params = sigmoid_params(pos_params, MULTI=True, OLDSTYLE = OLDSTYLE)
    
    #unpack parameters
    Jee = params[0] * np.pi * psi
    Jei = params[1] * np.pi * psi
    Jie = params[2] * np.pi * psi
    Jii = params[3] * np.pi * psi
    
    if len(params) < 8:
        i2e = params[4]
        Plocal = params[5]
        sigR = params[6]
        gE = 1
        gI = 1 * i2e
        NMDAratio = 0.1
    elif len(params) == 9:
        i2e = 1
        gE = params[4]
        gI = params[5]
        NMDAratio = params[6]
        Plocal = params[7]
        PlocalIE = Plocal
        sigR = params[8]
        sigEE = 0.35*np.sqrt(sigR)
        sigIE = 0.35/np.sqrt(sigR)
    elif len(params) == 10:
        i2e = 1
        gE = params[4]
        gI = params[5]
        NMDAratio = params[6]
        Plocal = params[7]
        sigEE = params[8]
        sigIE = params[9]
        PlocalIE = Plocal
    else: 
        i2e = 1
        gE = params[4]
        gI = params[5]
        NMDAratio = params[6]
        Plocal = params[7]
        PlocalIE = params[8]
        sigEE = params[9]
        sigIE = params[10]
    
    sigEE = sigEE / magnFactor # sigEE now in degress 
    sigIE = sigIE / magnFactor  # sigIE now in degrees
    
    W = make_conn.make_full_W(Plocal, Jee, Jei, Jie, Jii, sigEE, sigIE, deltaD, OMap, PlocalIE = PlocalIE)

    ssn = SSN_classes._SSN_AMPAGABA(tau_s, NMDAratio, n, k, Ne, Ni, tau_vec, W)
    ssn.topos_vec = np.ravel(OMap)
                        
    r_init = np.zeros([ssn.N, len(Contrasts)])
    #Inp_vec is Ne+Ni x stimulus conditions (typically 8) to find FP and such things 
    inp_vec = np.vstack((gE*Inp, gI*Inp))
    
                        
    r_fp, CONVG = ssn.fixed_point_r(inp_vec, r_init=r_init, Tmax=Tmax, dt=dt, xtol=xtol)
    
    #calculate power spectrum - find PS for each stimulus condition and concatenate them together 
    for cc in range(cons):
        if cc == 0:
            spect, fs, _ = SSN_power_spec.linear_power_spect(ssn, r_fp[:, cc], noise_pars, freq_range, fnums, LFPrange=[LFPtarget[0]])
        elif cc == 1:
            spect_2, _, _ = SSN_power_spec.linear_power_spect(ssn, r_fp[:, cc], noise_pars, freq_range, fnums, LFPrange=[LFPtarget[0]])
            spect = np.concatenate((spect[:, None], spect_2[:, None]), axis = 1)
        else:
            spect_2, _, _ = SSN_power_spec.linear_power_spect(ssn, r_fp[:, cc], noise_pars, freq_range, fnums, LFPrange=[LFPtarget[0]])
            spect = np.concatenate((spect, spect_2[:, None]), axis = 1)
        
    # My one-shot way of finding the PS. The above version works better 
#     if cons == 1:
#         spect, fs, f0, _ = SSN_power_spec.linear_power_spect(ssn, r_fp, noise_pars, freq_range, fnums, cons, LFPrange=[LFPtarget[0]])
        
#         if np.max(np.abs(np.imag(spect))) > 0.01:
#             print("Spectrum is dangerously imaginary")
            
#     else:
#         spect, fs, f0, _ = SSN_power_spec.linear_PS_sameTime(ssn, r_fp[:, con_inds], noise_pars, freq_range, fnums, cons, LFPrange=[LFPtarget[0]])
    
    #find PS at the outer neurons 
    outer_spect = make_outer_spect(ssn, r_fp[:,gabor_inds], probes)
    spect = np.concatenate((spect, outer_spect), axis=1)
    
    if np.max(np.abs(np.imag(spect))) > 0.01:
        print("Spectrum is dangerously imaginary")
    
    spect = np.real(spect)
    
    # I'm keeping f0 in the outputs just for congruency between the topo SSN and 2-D SSN code 
    f0 = 0
    #print(spect.shape)
    
    return spect, fs, f0, r_fp, CONVG

def make_outer_spect(ssn, rs, probes):
    '''
    Finds PS at outer (non-center) neurons 
    inputs:
    ssn = ssn obj that contains things like N and how to find the fixed point etc,.
    rs = fixed point rates 
    probes = # of locations that I wish to find the PS at. -- should be less than the width of network
    '''
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

def save_results_make_plots(params_init, params, loss_t, Contrasts, Inp,  hyper_params, fname=None, res=[], ground_truth = False, OLDSTYLE=False, tf = None, T=1e-2):
    '''
    function that saves resulst and make plots. 
    inputs
    params_init = initial paramaeters to find PS and rates initially
    params = fit parameters, to find observed PS and rates
    loss_t = loss over time
    Contrasts = stimulus strenghts presented
    Inp = Ne*Ni x stimulus condiditons of the input
    fname - finale name to save .mat and .pdf files 
    res = output of BFGS fcn
    ground_truth = False fit to idealized PS, True fit to MATLAB created SSN PS
    OLDSTYLE = False use extended gE/gI range sqrt([0.1, 10]), vs True use limited gE/gI range [0.5, 2]
    '''
    
    #find the initial power spectrum/rates
    init_spect, fs, _, init_r, init_CONVG = ssn_FP(params_init, OLDSTYLE)
    params_init = sigmoid_params(params_init, MULTI=True, OLDSTYLE = OLDSTYLE)
    
    #really just want to track the center neurons (E/I)
    init_r = init_r[(trgt, trgt+Ne),:]
    
    r_targ = init_r[:, rad_inds]/np.mean(init_r[:, rad_inds], axis=1)[:,None]
    softmax_r = T * logsumexp( r_targ / T ) 
    init_SI = 1 - (r_targ[:,-1]/softmax_r)
    
    init_spect = init_spect/np.mean(init_spect)
    
    #find the target PS
    target_PS = np.real(np.array(losses.get_multi_probe_spect(fs, fname ='test_spect.mat', ground_truth = ground_truth)))
    target_PS = target_PS/np.mean(target_PS)
    
    #find the observed PS and rates
    obs_spect, _, _, obs_r, CONVG = ssn_FP(params, OLDSTYLE)
    params = sigmoid_params(params, MULTI=True, OLDSTYLE = OLDSTYLE)
    
    obs_r = obs_r[(trgt, trgt+Ne), :]
    obs_spect = obs_spect/np.mean(obs_spect)
    
    r_targ = obs_r[:, rad_inds]/np.mean(obs_r[:, rad_inds], axis=1)[:,None]
    softmax_r = T * logsumexp( r_targ / T ) 
    obs_SI = 1 - (r_targ[:,-1]/softmax_r)
    
    obs_f0 = SSN_power_spec.find_peak_freq(fs, obs_spect, len(Contrasts))
    init_f0 = SSN_power_spec.find_peak_freq(fs, init_spect, len(Contrasts))

#     make_plot.Maun_Con_plots(fs, obs_spect, target_PS, Contrasts[con_inds],obs_r[:, con_inds].T, np.reshape(Inp[:,-1], (gridsize, gridsize)), obs_f0, initial_spect=init_spect, initial_rates=init_r[:, con_inds].T, initial_f0= init_f0, fname=fname)
    
    #save the results in dict for savemat
    Results = {
        'obs_spect':obs_spect,
        'obs_rates':obs_r,
        'obs_f0':obs_f0,
        'CONVG':CONVG,
        'init_spect':init_spect,
        'init_rates':init_r,
        'init_CONVG':init_CONVG,
        'target_spect':target_PS,
        'loss_t':loss_t,
        'params':params,
        'params_init':params_init,
        'res':res,
        'hyper_params':hyper_params,
        'time':tf
    }
    
    if fname is not None:
        f_out = fname.split('.')[0]+'.mat'
        print(f_out)
#         f_out.append('.mat')
        sio.savemat(f_out, Results)
    
    plot_rads = np.hstack((0, r_cent))
    
    obs_fig = make_plot.Maun_Con_SS(fs, obs_spect, target_PS, obs_r.T, obs_f0, contrasts, plot_rads, params, init_params = params_init, probes=probes, SI = obs_SI, dx = dx, fname=None, fignumber= 16)
    init_fig = make_plot.Maun_Con_SS(fs, init_spect, target_PS, init_r.T, init_f0, contrasts, plot_rads, params,  init_params = params_init, probes=probes, SI = init_SI, dx = dx, fname=None, fignumber= 17)
    
    #to save multiple page pdf document
    with PdfPages(fname) as pdf:
        pdf.savefig(obs_fig)
        plt.close(obs_fig)
        pdf.savefig(init_fig)
        plt.close(init_fig)
    
    return obs_spect, obs_r, Results


#way to run this as a .py file and not in ipynb. Thanks Taka! 
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