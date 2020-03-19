import numpy as np

import SSN_classes 
import SSN_power_spec
import MakeSSNconnectivity as make_conn

dt = 1
xtol = 1e-5
Tmax = 100000

#power spectrum resolution and range
fnums = 35 #resolution
freq_range = [0.1, 100]

#SSN parameters
t_scale = 1 

class ssn_pars1():
    n = 2
    k = 0.04
    tauE = 30 * t_scale
    tauI = 10 * t_scale
    psi = 0.774 
    
    tau_s = np.array([5, 7, 100])*t_scale #in ms, AMPA, GABA, NMDA current decay time constants
    # NMDAratio = 0.4 #NMDA strength as a fraction of E synapse weight
    
    # define the network spatial parameters. Gridsizedeg is the key that determines everything. MagnFactor is biologically observed to be ~2mm/deg. Gridsizedeg = 2 and gridperdeg = 5 means that the network is 11 x 11 neurons (2*5 + 1 x 2*5 + 1)
    #gridsizedeg = 3.2
    gridsizedeg = 2
    gridperdeg = 5
    gridsize = round(gridsizedeg*gridperdeg) + 1
    magnFactor = 2 #mm/deg
    dx = 2*gridsizedeg/(gridsize -1) 
    #biological hyper_col length is ~750 um, magFactor is typically 2 mm/deg in macaque V1
    # hyper_col = 0.8/magnFactor
    hyper_col = 8/magnFactor * 10
    #define stimulus conditions r_cent = Radius of the stim, contrasts = contrasts. 
    #dradius = gridsizedeg/8
    #r_cent = np.arange(dradius, round(gridsizedeg/2)+dradius, dradius)
    dradius = 0.25 # in degrees
    r_cent = np.arange(dradius, 1+dradius, dradius)
    
    contrasts = np.array([0, 25, 50, 100])
    
    X,Y, deltaD = make_conn.make_neur_distances(gridsizedeg, gridperdeg, hyper_col, PERIODIC = False) 
    
    OMap, _ = make_conn.make_orimap(hyper_col, X, Y)
    
    Ne, Ni = deltaD.shape
    tau_vec = np.hstack((tauE*np.ones(Ne), tauI*np.ones(Ni)))
    N = Ne + Ni
    
    trgt = int(np.floor(Ne/2))
    probes = 5
    
    LFPtarget = trgt + np.array([ii for ii in range(probes)])*gridsize
    # LFPtarget = trgt + np.hstack((np.array([ii for ii in range(probes-2)])*gridsize, np.array([ii*gridsize - 1 for ii in range(1,3)])))
    #second choice will find diagonally separted ones. 
    
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
    rad_inds = np.hstack((0, rad_inds[:-1])) #adding in the 0 contrast conditions, removing the gabor
    con_inds = gen_inds[rad_max] #I want to look across contrasts at max radii
    gabor_inds = -1 #the last index is always the gabor at max contrast and max gabor sigma
    cons = len(con_inds)
    
    noise_pars = SSN_power_spec.NoisePars(corr_time= 10)
    
    spont_input = np.array([1,1]) * 2
    make_J2x2 = lambda Jee, Jei, Jie, Jii: np.array([[Jee, -Jei], [Jie, -Jii]]) * np.pi * ssn_pars.psi

class ssn_pars2D():
    n = 2
    k = 0.04
    tauE = 50 * t_scale
    tauI = 10 * t_scale
    psi = 0.774 
    
    tau_s = np.array([5, 7, 100])*t_scale #in ms, AMPA, GABA, NMDA current decay time constants
    contrasts = np.array([0, 25, 50, 100])
    spont_input = np.array([1,1]) * 2
    make_J2x2 = lambda Jee, Jei, Jie, Jii: np.array([[Jee, -Jei], [Jie, -Jii]]) * np.pi * ssn_pars.psi
    
### ==========================================================================================


def ssn_PS(params, ssn_pars=ssn_pars1):
    Jee = params[0] * np.pi * ssn_pars.psi
    Jei = params[1] * np.pi * ssn_pars.psi
    Jie = params[2] * np.pi * ssn_pars.psi
    Jii = params[3] * np.pi * ssn_pars.psi
    gE = params[4]
    gI = params[5]
    NMDAratio = params[6]
    Plocal = params[7]
    PlocalIE = params[8]
    sigEE = params[9]
    sigIE = params[10]
    
    sigEE = sigEE/ssn_pars.magnFactor
    sigIE = sigIE/ssn_pars.magnFactor
    
    n = ssn_pars.n
    k = ssn_pars.k
    tauE = ssn_pars.tauE
    tauI = ssn_pars.tauI
    psi = ssn_pars.psi
    tau_s = ssn_pars.tau_s
    spont_input = ssn_pars.spont_input
    
    print(ssn_pars.Ne, ssn_pars.Ni)
    
    W = make_conn.make_full_W(Plocal, Jee, Jei, Jie, Jii, sigEE, sigIE, ssn_pars.deltaD, ssn_pars.OMap, PlocalIE = PlocalIE)
    
    ssn = SSN_classes._SSN_AMPAGABA(tau_s, NMDAratio, n, k, ssn_pars.Ne, ssn_pars.Ni, ssn_pars.tau_vec, W)
    ssn.topos_vec = np.ravel(ssn_pars.OMap)
    
    rs = []
    spect = []
    
    r_init = np.zeros([ssn.N, len(ssn_pars.Contrasts)])
    inp_vec = np.vstack((gE*ssn_pars.Inp, gI * ssn_pars.Inp))
    
    r_fp, CONVG = ssn.fixed_point_r(inp_vec, r_init=r_init, Tmax=Tmax, dt = dt, xtol = xtol)
    
    if not CONVG:
        fs = np.nan * np.linspace(*freq_range, 2)
        f0 = np.nan * np.ones(8)
        return rs, spect, fs, f0
    
    rs.append(r_fp)
    
    for cc in range(ssn_pars.cons):
        powspecE, fs, _ = SSN_power_spec.linear_power_spect(ssn, r_fp[:, cc], ssn_pars.noise_pars, freq_range=freq_range, fnums=fnums, LFPrange=[ssn_pars.LFPtarget[0]])
        
        spect.append(powspecE)
    
    for pp in range(1, ssn_pars.probes):
        powspecE, fs, _ = SSN_power_spec.linear_power_spect(ssn, r_fp[:, -1], ssn_pars.noise_pars, freq_range= freq_range, fnums=fnums, LFPrange=[ssn_pars.LFPtarget[pp]])
        
        spect.append(powspecE)
    
    rs = np.array(rs)
    spect = np.array(spect)
    
    f0, _, _, _, _ = SSN_power_spec.infl_find_peak_freq(fs, spect.T)
    
    return rs, spect, fs, f0

def ssn_2D_PS(params, ssn_pars=ssn_pars2D):
    Jee = params[0] * np.pi * ssn_pars.psi
    Jei = params[1] * np.pi * ssn_pars.psi
    Jie = params[2] * np.pi * ssn_pars.psi
    Jii = params[3] * np.pi * ssn_pars.psi
    gE = params[4]
    gI = params[5]
    NMDAratio = params[6]
    
    n = ssn_pars.n
    k = ssn_pars.k
    tauE = ssn_pars.tauE
    tauI = ssn_pars.tauI
    psi = ssn_pars.psi
    tau_s = ssn_pars.tau_s
    spont_input = ssn_pars.spont_input
    
    ssn = SSN_classes.SSN_2D_AMPAGABA(tau_s, NMDAratio, n, k, tauE, tauI, Jee, Jei, Jie, Jii, )
    
    rs = []
    spect = []
    
    r_init = np.zeros([ssn.N, len(ssn_pars.contrasts)])
    inp_vec = np.array([[gE], [gI]])*ssn_pars2D.contrasts
    
    r_fp, CONVG = ssn.fixed_point_r(inp_vec, r_init=r_init, Tmax=Tmax, dt = dt, xtol = xtol)
    
    if not CONVG:
        dfdc = np.nan * np.ones(2)
        return dfdc
    
    rs.append(r_fp)
    
    for cc in range(len(ssn_pars.contrasts)):
        powspecE, fs, _ = SSN_power_spec.linear_power_spect(ssn, r_fp[:, cc], SSN_power_spec.NoisePars(corr_time=1), freq_range=freq_range, fnums=fnums)
        
        spect.append(powspecE)
    
    rs = np.array(rs)
    spect = np.array(spect)
    
    f0,_,_,_,_ = SSN_power_spec.infl_find_peak_freq(fs, spect.T)
    
    dfdc = np.diff(f0[1:])/np.diff(ssn_pars2D.contrasts[1:])
    print(dfdc)
    
    return dfdc


### ==========================================================================================


# makes a parameter a value between pmin and pmax uniformly distributed
rand_samp = lambda pmin, pmax: pmin + (pmax - pmin) * np.random.rand()


def sample_multi_SSN(Nsamps, contrasts, Jxe_max, Jxe_min, g_max, g_min, NMDA_min, NMDA_max, Plocal_min, Plocal_max, sig_min, sig_max, BALANCED = True, ssn_pars=ssn_pars1, Jxi_min = None, Jxi_max = None):
    Jxi_min = Jxi_min if Jxi_min is not None else Jxe_min
    Jxi_max = Jxi_max if Jxi_max is not None else Jxe_min
    
    smp = 0
    params_list = []
    rs_list = []
    spect_list = []
    f0_list = []
    
    for jj in range(Nsamps):
        print(f"j = {jj}, samples = {smp}")
        params = [1,1,1,1]

        for i in [0,2]:
            params[i] = rand_samp(Jxe_min, Jxe_max)

        for i in [1,3]:
            params[i] = rand_samp(Jxi_min, Jxi_max)

        for i in range(2):
            params.append( rand_samp(g_min, g_max) )

        params.append( rand_samp(NMDA_min, NMDA_max))
        
        for i in range(2):
            params.append( rand_samp(Plocal_min, Plocal_max))
        for i in range(2):
            params.append( rand_samp(sig_min, sig_max))
        
        Jee, Jei, Jie, Jii = params[:4]
        ge, gi = params[4:6]

        if not (Jee/Jie < Jei/Jii):
            continue
        if BALANCED and not (Jei/Jii < ge/gi):
            continue 

        rs, spect, fs, f0 = ssn_PS(params, ssn_pars= ssn_pars1)
        
        #if CONVG is false, returns nans for fs. 
        if np.isnan(fs).any():
            continue
        
        smp += 1
        print(f"smp={smp}")
        params_list.append(params)
        rs_list.append(rs)
        spect_list.append(spect)
        f0_list.append(f0)
        
        if smp == Nsamps:
            break
    
    return np.asarray(params_list), np.asarray(rs_list), np.asarray(spect_list), np.asarray(f0_list), fs, jj


def sample_target_SSN(Nsamps, contrasts, params_min, params_max, BALANCED = True, ssn_pars=ssn_pars1):
    
    Jee_min, Jei_min, Jie_min, Jii_min, gE_min, gI_min, NMDA_min, PlocalEE_min, PlocalIE_min, sigEE_min, sigIE_min = params_min
    Jee_max, Jei_max, Jie_max, Jii_max, gE_max, gI_max, NMDA_max, PlocalEE_max, PlocalIE_max, sigEE_max, sigIE_max = params_max
    
    
    smp = 0
    params_list = []
    rs_list = []
    spect_list = []
    f0_list = []
    
    for jj in range(Nsamps):
        print(f"j = {jj}, samples = {smp}")
        
        params = []
        params.append( rand_samp(Jee_min, Jee_max))
        params.append( rand_samp(Jei_min, Jei_max))
        params.append( rand_samp(Jie_min, Jie_max))
        params.append( rand_samp(Jii_min, Jii_max))
        
        params.append( rand_samp(gE_min, gE_max))
        params.append( rand_samp(gI_min, gI_max))
        
        params.append( rand_samp(NMDA_min, NMDA_max))
        params.append( rand_samp(PlocalEE_min, PlocalEE_max))
        params.append( rand_samp(PlocalIE_min, PlocalIE_max))
        
        params.append( rand_samp(sigEE_min, sigEE_max))
        params.append( rand_samp(sigIE_min, sigIE_max))

        Jee, Jei, Jie, Jii = params[:4]
        ge, gi = params[4:6]


        if not (Jee/Jie < Jei/Jii):
            continue
        if BALANCED and not (Jei/Jii < ge/gi):
            continue

        rs, spect, fs, f0 = ssn_PS(params, ssn_pars= ssn_pars1)
        
        #if CONVG is false, returns nans for fs. 
        if np.isnan(fs).any():
            continue
        
        smp += 1
        print(f"smp={smp}")
        params_list.append(params)
        rs_list.append(rs)
        spect_list.append(spect)
        f0_list.append(f0)
        
        if smp == Nsamps:
            break
    
    return np.asarray(params_list), np.asarray(rs_list), np.asarray(spect_list), np.asarray(f0_list), fs, jj


def multi_NonLocal_SSN(Nsamps, contrasts, Jxe_max, Jxe_min, g_max, g_min, NMDA_min, NMDA_max, Plocal_min, Plocal_max, sig_min, sig_max, BALANCED = True, ssn_pars=ssn_pars1, Jxi_min = None, Jxi_max = None, ARRAY = True):
    Jxi_min = Jxi_min if Jxi_min is not None else Jxe_min
    Jxi_max = Jxi_max if Jxi_max is not None else Jxe_min
    
    smp = 0
    rads_cons = len(ssn_pars.Contrasts)
    probes_cons = ssn_pars.probes + ssn_pars.cons-1
    params_list = np.empty((Nsamps, 11))
    rs_list = np.empty((Nsamps, ssn_pars.N, rads_cons))
    spect_list = np.empty((Nsamps, probes_cons, fnums))
    f0_list = np.empty((Nsamps, probes_cons))
    
    params_listNL = np.empty((Nsamps, 11))
    rs_listNL = np.empty((Nsamps, ssn_pars.N, rads_cons))
    spect_listNL = np.empty((Nsamps, probes_cons, fnums))
    f0_listNL = np.empty((Nsamps, probes_cons))
    interesting_inds = []
    
    for jj in range(Nsamps):
        print(f"j = {jj}, samples = {smp}")
        params = [1,1,1,1]

        for i in [0,2]:
            params[i] = rand_samp(Jxe_min, Jxe_max)

        for i in [1,3]:
            params[i] = rand_samp(Jxi_min, Jxi_max)

        for i in range(2):
            params.append( rand_samp(g_min, g_max) )

        params.append( rand_samp(NMDA_min, NMDA_max))
        
        Jee, Jei, Jie, Jii = params[:4]
        ge, gi = params[4:6]
        
        if not (Jee/Jie < Jei/Jii):
            continue
        if BALANCED and not (Jei/Jii < ge/gi):
            continue
        
        dfdc2d = ssn_2D_PS(params, ssn_pars=ssn_pars2D)
        
        if np.any(np.isnan(dfdc2d)):
            continue
            
        for i in range(2):
            params.append( rand_samp(Plocal_min, Plocal_max))
        for i in range(2):
            params.append( rand_samp(sig_min, sig_max))
        
        if params[-2] > params[-1]:
            sigTemp = params[-1]
            params[-1] = params[-2]
            params[-2] = sigTemp
        

        #rs, spect, fs, f0 = ssn_PS(params, ssn_pars= ssn_pars1)
        
        #if CONVG is false, returns nans for fs. 
#         if np.isnan(fs).any():
#             continue
        
        paramsNL = params[:7] + [0,0] + params[-2:]
        rsNL, spectNL, fs, f0NL = ssn_PS(paramsNL, ssn_pars = ssn_pars1)
        
        if np.isnan(fs).any():
            continue
        
        if np.all(~np.isnan(f0NL[1:])):
            interesting_inds.append(smp)
            
        
        print(f"smp={smp}")
        params_list[smp, :] = np.array(params)
#         rs_list[smp, :, :] = rs
#         spect_list[smp, :, :] = spect
#         f0_list[smp, :] = f0
        
        if np.isnan(f0NL[1:]).any():
            smp+=1
            continue
        
        params_listNL[smp,:] = np.array(paramsNL)
        rs_listNL[smp, :,:] = rsNL
        spect_listNL[smp, :,:] = spectNL
        f0_listNL[smp, :] = f0NL
        
        smp += 1
        if smp == Nsamps:
            break
            
    
    params_list = params_list[:smp, :]
#     rs_list = rs_list[:smp, :, :]
#     spect_list = spect_list[:smp, :, :] 
#     f0_list = f0_list[:smp, :]
        
    params_listNL = params_listNL[:smp,:]
    rs_listNL = rs_listNL[:smp, :,:]
    spect_listNL= spect_listNL[:smp, :,:]
    f0_listNL = f0_listNL[:smp, :]
    
    
    if ARRAY:
        return np.asarray(params_list), np.asarray(rs_list), np.asarray(spect_list), np.asarray(f0_list), np.asarray(params_listNL), np.asarray(rs_listNL), np.asarray(spect_listNL), np.asarray(f0_listNL), interesting_inds, fs, jj
    
    else:
        #return params_list.tolist(), rs_list.tolist(), spect_list.tolist(), f0_list.tolist(), params_listNL.tolist(), rs_listNL.tolist(), spect_listNL.tolist(), f0_listNL.tolist(), interesting_inds, fs.tolist(), jj
        return params_list.tolist(), params_listNL.tolist(), rs_listNL.tolist(), spect_listNL.tolist(), f0_listNL.tolist(), interesting_inds, fs.tolist(), jj