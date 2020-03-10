import numpy as np
#import jax.numpy as np

import SSN_classes
import SSN_power_spec # as SSN_power_spec

#fixed point algorithm:
dt = 1
xtol = 1e-5
Tmax = 1000

#power spectrum resolution and range
# fnums = 30
# freq_range = [15,100]
fnums = 300
freq_range = [0.1,150]

#SSN parameters
t_scale = 1
class ssn_pars1():
    n = 2
    k = 0.04
    tauE = 20 # in ms
    tauI = 10 # in ms
    psi = 0.774
    tau_s = np.array([4, 5, 100])*t_scale #in ms, AMPA, GABA, NMDA current decay time constants
    spont_input = np.array([1,1]) * 2 # * 0
    make_J2x2 = lambda Jee, Jei, Jie, Jii: np.array([[Jee, -Jei], [Jie,  -Jii]]) * np.pi * ssn_pars1.psi

t_scale = 1
class ssn_pars2():
    n = 2.2
    k = 5 # Hz
    tauE = 20 # in ms
    tauI = 10 # in ms
    psi = 10
    tau_s = np.array([4, 5, 100])*t_scale #in ms, AMPA, GABA, NMDA current decay time constants
    spont_input = np.array([1,1]) * 1 # * 0
    make_J2x2 = lambda Jee, Jei, Jie, Jii: np.array([[Jee, -Jei], [Jie,  -Jii]]) * ssn_pars2.psi

# ==============================================================================

rand_samp = lambda pmin, pmax: pmin + (pmax - pmin) * np.random.rand()



def ssn_PS(params, contrasts, ssn_pars=ssn_pars1):
    #unpack parameters
    Jee = params[0]   # 2.5
    Jei = params[1]   # 1.3
    Jie = params[2]   # 2.4
    Jii = params[3]   # 1.0
    gE = params[4]    # 1
    gI = params[5]    # 1
    NMDAratio = params[6] # 0.4

    n = ssn_pars.n
    k = ssn_pars.k
    tauE = ssn_pars.tauE
    tauI = ssn_pars.tauI
    psi = ssn_pars.psi
    tau_s = ssn_pars.tau_s
    spont_input = ssn_pars.spont_input
    
    J2x2 = ssn_pars.make_J2x2(*params[:4])

    ssn = SSN_classes.SSN_2D_AMPAGABA(n,k,tauE,tauI, *np.abs(J2x2).ravel(), tau_s=tau_s, NMDAratio=NMDAratio)

    spect = []
    lams = []
    rs = []
    r_init = np.zeros(ssn.N)
    for con in contrasts:
        inp_vec = np.array([gE,gI]) * con + spont_input
        r_fp, CONVG = ssn.fixed_point_r(inp_vec, r_init=r_init, Tmax=Tmax, dt=dt, xtol=xtol)
        r_init = r_fp
        if not CONVG:
            fs = np.linspace(*freq_range,fnums)
            break
        powspecE, fs, _, jacob_lams, _ = SSN_power_spec.linear_power_spect(ssn,
            r_fp, SSN_power_spec.NoisePars(), freq_range=freq_range, fnums=fnums, EIGS=True)

        rs.append(r_fp)
        spect.append(powspecE)
        lams.append(jacob_lams)

    rs = np.array(rs)    # shape = (#contrasts, 2)
    spect = np.array(spect) # shape = (#contrasts, #freqs)
    lams = np.array(lams)   # shape = (#contrasts, 4)

    return rs, spect, fs, lams


def sample_2D_SSN(Nsamps, contrasts, J_max, J_min, g_max, g_min, NMDA_min, NMDA_max, BALANCED=False, ssn_pars=ssn_pars1, J_I_min=None, J_I_max=None):
    J_I_min = J_I_min if J_I_min is not None else J_min
    J_I_max = J_I_max if J_I_max is not None else J_max

    smp = 0
    params_list = []
    rs_list = []
    spect_list = []
    lams_list = []
    for j in range(100*Nsamps):
        print(f"j = {j+1}, samples = {smp}")
        
        params = [1,1,1,1]
        # sample excitatory J's
        for i in [0,2]:
            params[i] = rand_samp(J_min, J_max) 
        # sample inhibitory J's
        for i in [1,3]:
            params[i] =  rand_samp(J_I_min, J_I_max) 
        # sample the g's
        for i in range(2):
            params.append( rand_samp(g_min, g_max) )
        # sample NMDAratio
        params.append( rand_samp(NMDA_min, NMDA_max) )
        
        Jee, Jei, Jie, Jii = params[:4]
        ge, gi = params[4:6]
        # guarantee positive det(J2x2):
        if not (Jee/Jie < Jei/Jii):
            continue
        # if BALANCED=True, guarantee positive Omega's
        if BALANCED and not (Jei/Jii < ge/gi):
            continue

        rs, spect, fs, lams = ssn_PS(params, contrasts, ssn_pars)
        # check if all contrasts had converged to f.p.
        if len(rs) < len(contrasts):
            continue

        smp += 1
        print(f"smp = {smp}")
        params_list.append(params)
        rs_list.append(rs)
        spect_list.append(spect)
        lams_list.append(lams)

        if smp == Nsamps:
            break

    return np.asarray(params_list), np.asarray(rs_list), np.asarray(spect_list), np.asarray(lams_list), j, fs
