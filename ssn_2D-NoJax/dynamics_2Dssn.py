import numpy as np
import scipy.io as sio
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib import gridspec

import SSN_classes
import SSN_power_spec
import util
import make_plot

#the constant (non-optimized) parameters:

#fixed point algorithm:
dt = 1
xtol = 1e-6
Tmax = 50000
Nt = int(np.floor(Tmax/dt))
Tmax = Nt * dt

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
tau_s = np.array([5, 7, 100])*t_scale #in ms, AMPA, GABA, NMDA current decay time constants

contrasts = np.array([0, 25, 50, 100])
cons = len(contrasts)

def noise_rates(params):
    Jee, Jei, Jie, Jii, gE, gI, NMDAratio = params
    Jee = Jee * psi * np.pi
    Jei = Jei * psi * np.pi
    Jie = Jie * psi * np.pi
    Jii = Jii * psi * np.pi

    ssn = SSN_classes.SSN_2D_AMPAGABA(tau_s, NMDAratio, n,k, tauE, tauI, Jee, Jei, Jie, Jii)
    
    #find the fixed point of the rates, r_fp, and the analytic Power Spectrum, spect
    r_init = np.zeros([ssn.N, len(contrasts)])
    inp_vec = np.array([[gE], [gI]]) * contrasts
    r_fp, CONVG = ssn.fixed_point_r(inp_vec, r_init=r_init, Tmax=Tmax, dt=dt, xtol=xtol)
    
    #find the noisy voltages 
    v_init = np.zeros([ssn.N * ssn.num_rcpt, cons])
    v_dyn = np.zeros((Nt, ssn.N*ssn.num_rcpt, cons))
    
    inpv_vec = np.array([gE, gI, 0, 0, 0, 0])[:, None] *contrasts
    tt = np.arange(0, Nt, dt)
    noise_pars = SSN_power_spec.NoisePars()
    
    for cc in range(cons):
        noise_inp = np.array([noise_pars.stdevE, noise_pars.stdevI ])[:,None] * SSN_power_spec.makeInputNoise(ssn, SSN_power_spec.NoisePars(), dt, Nt)
        noise_inp = np.vstack(((1-noise_pars.NMDAratio)* noise_inp, noise_pars.NMDAratio * noise_inp, 0 * noise_inp ))
        AttNoiseInp = lambda t: util.MyLinInterp1(tt, noise_inp, t, 0, dt)
        
        #dynamicsAMPA_GABA defined below this fcn. Finds the firing rate r, and then the RHS of dvdt 
        dvdt = lambda t, v: dynamicsAMPA_GABA(ssn, v, inpv_vec[:, cc], AttNoiseInp(t))

        sol = solve_ivp(dvdt, (0, Tmax), v_init[:, cc], t_eval=tt)
        v_dyn[:,:, cc] = sol.y.T
    
    rr_t = np.zeros((Nt, ssn.N, cons))
    for t_ind in range(Nt):
        rr_t[t_ind, :, :] = ssn.powlaw(v_dyn[t_ind, :, :].reshape((ssn.num_rcpt, ssn.N, -1)).sum(axis=0))
    
    rates_color = ['red', 'blue']
#     fig_rates, ax = plt.subplots(2,2, figsize=(10,10))

    t_inds = np.arange(300,Nt)

#     for xx in range(2):
#         for yy in range(2):
#             r_ind = 2*xx+yy
#             print(r_ind)
#             ax[xx, yy].set_prop_cycle('color', rates_color)
#             ax[xx, yy].plot(tt[t_inds], rr_t[t_inds, :, r_ind], lw=2.25)
#             ax[xx, yy].plot(tt[t_inds], r_fp[:, r_ind]*np.ones(len(t_inds))[:,None], '--', lw=2.25)
#             tstr = 'C = '+str(contrasts[r_ind])
#             ax[xx,yy].set_title(tstr)
#             ax[xx,yy].set_xlabel('Time (ms)')
#             ax[xx,yy].set_ylabel('Firing rates (Hz)')
#             ax[xx,yy].legend(['E','I'])

    return r_fp, rr_t, v_dyn
    
    
def dynamicsAMPA_GABA(ssn, vv, inpDC, Noise):
    # 2 neuron firing rate,
    r = ssn.powlaw(vv.reshape(ssn.num_rcpt, ssn.N).sum(axis=0))
    dvdt = (-vv + ssn.Wrcpt @ r + inpDC + Noise)/ssn.tau_s_vec
    return dvdt
    
def simulatedPS(r_fp, rr_t, v_dyn, dt, freq_res, windowing='none'):
    
    TminPS = 20
    timerangefft = [np.max((200, TminPS)), Nt]
    #xtE1 = sum of input voltages to the E neuron. Will be fluctuations around the fixed point ish?
    # so I htink lines 154-160 of SimulatedPowerSpec_2D.m are just taking the voltages and finding the rates over a specific range. 
    # since I already have the rates at different times, just use them.
    #xtE1= rr_t[np.arange(timerangefft[0], timerangefft[1]), 0,:]
    xtE1 = np.sum(v_dyn[np.arange(timerangefft[0], timerangefft[1]), ::2, :], axis=1)
    x = xtE1 - np.mean(xtE1, axis=0)
    
    cons = x.shape[1]
    dt = dt/1000 #dt in secs
    
    window = 0
    
    N = int(np.diff(timerangefft)) #number of time steps in the whole sequence, so the 
    L = 1
    M = int(np.round(1/(freq_res * dt))) #time steps in each segment, since freq_res = 1/T_segment, then M = N_segments = T_segment/dt = (1/(freq_res * dtps))
    df = 1/(M*dt) # ~= freq_resolution. This is K*df0 where df0 = 1/(N*dt) is the full freq-resolution of the full segment (if we didn't chunk it.)
    K = int(np.floor(N/M)) #number of chunks

    Kmin = 4 #include this as a default variable to the fcn I'll eventually write

    if K < Kmin:
        # recalculate M and N
        K = Kmin
        M = int(np.floor(N/K))
        df = 1/(M*dt)
        print("Warning: frequency resolution increased to {:.0f} to allow for at least 4 disjoint segments in x".format(df))

    dM = N-K*M
    #suppose N = M*K; and M = 2*m+1 is odd, so that m = floor(M/2)
    m = int(np.floor(M/2))
    
    if windowing is not None: #Numerical Recipes recommends either Bartlett or Welch 
        if windowing == 'welch':
            window = 1-((np.arange(M)-m)/m)**2;
        elif windowing == 'bartlett':
            window = 1-np.abs(np.arange(M)-m)/m;
        elif windowing == 'hamming':
            window = 0.5*(1-np.cos(2*np.pi*np.arange(M)/(M-1)));
        elif windowing == 'hann':
            window = 0.5*(1-np.cos(2*pi*np.arange(M)/(M-1)));
        else:
            window = np.ones(M);
#             print(window)
    
    windowWeight = np.sum(window**2) #without windowing this is M

    fs = np.arange(m+1)*df
    sim_power = np.zeros((len(fs), cons))
    
    for cc in np.arange(cons):
        
        x1 = np.reshape(x[:K*M,cc],(M,K*L), 'F')
        half_overlap = np.reshape(x[m:(m+M*(K-1)),cc],(M, (K-1)*L), 'F')
        x1 = np.hstack((x1, half_overlap))
        if dM > m/2:
            x1 = np.hstack((x1, np.reshape(x[N-M:N,cc], (M,L))))
        x1 = window[:, None]*x1;

        xf = np.fft.fft(x1, axis=0)
        pf = np.mean(np.abs(xf)**2, axis=1)/M/windowWeight/df;

        power = np.zeros(m+1)
        power[0] = pf[0]

        if np.mod(M,2)==1: #if M is odd
            power[1:] = pf[1:(m+1)] + pf[M:(m):-1]
        else:
            power[1:m] = pf[1:m] + pf[M:m:-1]
            power[m] = pf[m]
        
        sim_power[:, cc] = power
    
    return sim_power, df, fs, M


def ssn_PS(params, contrasts):
    #params = sigmoid_params(pos_params, MULTI=False)
    
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
    spont_input= np.array([1,1]) * 2
    inp_vec = np.array([[gE],[gI]]) * contrasts + spont_input[:, None]
    
    r_fp, CONVG = ssn.fixed_point_r(inp_vec, r_init=r_init, Tmax=Tmax, dt=dt, xtol=xtol)
    
    spect, fs, f0, Jacob = SSN_power_spec.linear_PS_sameTime(ssn, r_fp, SSN_power_spec.NoisePars(), freq_range, fnums, cons)
    
    return spect, fs, f0, r_fp, 
