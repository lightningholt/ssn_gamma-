import jax.numpy as np

#from jax.ops import index_update

from util import toeplitz

#corr_time = 5 originally

class NoisePars(object):
	def __init__(self, stdevE=1.5, stdevI=1.5, corr_time=1, corr_length=0, NMDAratio=0):
        #NMDAratio is Inp.NoiseNMDAratio = 0 in the MATLAB code
        # in other words, its how much NMDA contributes to noise in network, not the actual amount of NMDA receptors in the network
		self.stdevE = stdevE
		self.stdevI = stdevI
		self.corr_time = corr_time
		self.corr_length = corr_length
		self.NMDAratio = NMDAratio

def make_eE_noiseCov(ssn, noise_pars, LFPrange):
	# setting up e_E and e_I: the projection/measurement vectors for
	# representing the "LFP" measurement (e_E good for LFP interpretation, but e_I ?)
	# eE = np.zeros(ssn.N)
	# # eE[LFPrange] =1/len(LFPrange)
	# index_update(eE, LFPrange, 1/len(LFPrange))
	# eI = np.zeros(ssn.N)
	# eI[ssn.Ne + LFPrange] =1/len(LFPrange)
    
	eE = np.hstack((
		np.array([i in LFPrange for i in range(ssn.Ne)], dtype=np.float32),
		np.zeros(ssn.Ni)))

	# the script assumes independent noise to E and I, and spatially uniform magnitude of noise
	noiseCov = np.hstack( (noise_pars.stdevE**2 * np.ones(ssn.Ne),
						   noise_pars.stdevI**2 * np.ones(ssn.Ni)) )

	OriVec = ssn.topos_vec
	if noise_pars.corr_length>0 and OriVec.size>1: #assumes one E and one I at every topos
	    dOri = np.abs(OriVec)
	    L = OriVec.size * np.diff(OriVec[:2])
	    dOri[dOri > L/2] = L-dOri[dOri > L/2] # distance on circle/periodic B.C.
	    SpatialFilt = toeplitz(np.exp(-(dOri**2)/(2*noise_pars.corr_length**2))/np.sqrt(2*pi)/noise_pars.corr_length*L/ssn.Ne)
	    sigTau1Sprd1 = 0.394 # roughly the std of spatially and temporally filtered noise when the white seed is randn(ssn.Nthetas,Nt)/sqrt(dt) and corr_time=corr_length = 1 (ms or angle, respectively)
	    SpatialFilt = SpatialFilt * np.sqrt(noise_pars.corr_length/2)/sigTau1Sprd1 # for the sake of output
	    SpatialFilt = np.kron(np.eye(2), SpatialFilt) # 2 for E/I
	else:
	    SpatialFilt = np.array(1)

	return eE, noiseCov, SpatialFilt # , eI

def linear_power_spect(ssn, rs, noise_pars, freq_range, fnums, LFPrange=None, GammaRange=[20,100]):
	"""
	LFPrange = noise_pars.Ori1 + (-10:10) = indices which contribute to the LFP: activity (or voltage
	is averaged over those dimensions and power-spec of that avg is calculated
	noise_pars.stdevE = 1.5; Std of E noise
	noise_pars.stdevI = 1.5; Std of E noise
	noise_pars.corr_time = 5; correlation time of noise in ms
	noise_pars.corr_length = 0.5; correlation length of noise in angles; 0 doesn't work well..: too small response

	Trgt = round(noise_pars.Ori1/OriVec(end)*ssn.Nthetas);
	LFPrange = Trgt+(-10:+10);

	example run:
	powspecE = linear_power_spect(ssn, r_fp, NoisePars(), freq_range=[10,100], fnums=50)

	by Yashar Ahmadian -- Nov 2015.
	"""
	if LFPrange is None:
		LFPrange = [0]
	eE, noiseCov, SpatialFilt = make_eE_noiseCov(ssn, noise_pars, LFPrange)
	tau_corr = noise_pars.corr_time

	#switch-block case 'ampa-gaba-nmda' in MATLAB code
	J = ssn.DCjacobian(rs)

	eE1 = np.kron(np.ones(ssn.num_rcpt), eE) # this tensor product by ones(...) is because of the unweighted sum of currents of different types inside the neuronal nonlinearity
	# eI1 = np.kron(np.ones(ssn.num_rcpt), eI) # this tensor product by ones(...) is because of the unweighted sum of currents of different types inside the neuronal nonlinearity

	# Jacob = ssn.jacobian(J) # np.kron(1/ssn.tau_s, np.ones(ssn.N))[:,None] * J  # equivalent to diag(tau_s) J (math)
	# JacobLams = np.linalg.eigvals(Jacob)

	maxF = freq_range[1]
	minF = freq_range[0]
	fs = np.linspace(minF,maxF,fnums) # grid of frequencies in Hz
	fs = fs/1000 # coverting from Hz to kHz = 1/ms

	#AnalPowSpecE = np.empty_like(fs)
	AnalPowSpecE = []
	for ii, ff in enumerate(fs):
		w = 2*np.pi * ff # omega
		vecE = np.linalg.solve(
				(-1j*w * np.diag(ssn.tau_s_vec)  - J).T.conj()  # ssn.inv_G(w,J).T.conj()
				, eE1)
		if ssn.num_rcpt<3: # i.e. if we only have AMPA and GABA (AMPA always first)
			#     vecE = SpatialFilt'*reshape(vecE,[ssn.N,ssn.num_rcpt]) # accounting for spatial correlations in noise input
			# !!HERE we ASSUME noise is coming only through the AMPA (m=1) channel... modify LATER for general
			# make sure it's also correct to just ignore the rest of vecE and only take
			# its 1st 1/ssn.num_rcpt (corresponding to AMPA)
			if SpatialFilt.size>1:
				vecE = SpatialFilt.T  @ vecE[:ssn.N]
			else:
				vecE = SpatialFilt * vecE[:ssn.N]
		elif ssn.num_rcpt==3: # i.e. if we have NMDA (which is always the last part)
			# assuming AMPA and NMDA channels (up to scaling) get the exact same realization of noise (i.e. noise cov is rank-deficient)
			if SpatialFilt.size>1:
				vecE = SpatialFilt.T @ ((1-noise_pars.NMDAratio) * vecE[:ssn.N]  + noise_pars.NMDAratio * vecE[-ssn.N:])
			else:
				vecE = SpatialFilt * ((1-noise_pars.NMDAratio) * vecE[:ssn.N]  + noise_pars.NMDAratio * vecE[-ssn.N:])

		#AnalPowSpecE[ii] = np.dot(vecE.conj(), noiseCov * vecE) * 2* tau_corr/np.abs(-1j*w * tau_corr + 1)**2 # the factor on the right is power-spec of pink noise with time-constant tau_corr and variance 1, which is 2*\tau /abs(-i\omega*\tau + 1)^2 ( FT of its time-domain cov)
		# index_update(AnalPowSpecE, ii,
		#	np.dot(vecE.conj(), noiseCov * vecE) * 2* tau_corr/np.abs(-1j*w * tau_corr + 1)**2 ) # the factor on the right is power-spec of pink noise with time-constant tau_corr and variance 1, which is 2*\tau /abs(-i\omega*\tau + 1)^2 ( FT of its time-domain cov)
		AnalPowSpecE.append( np.dot(vecE.conj(), noiseCov * vecE) * 2* tau_corr/np.abs(-1j*w * tau_corr + 1)**2 )
	#end of switch-block from MATLAB code

	AnalPowSpecE = np.real(np.asarray(AnalPowSpecE))
	fs = fs*1000 # coverting from kHz = 1/ms back to Hz

	# the above formulas give power separately for positive and negative freq's (which it is symmetric in):
	# to combine (as we have done in myPowerSpec.m, we multiply by 2:
	# also the above formulas give the power spectral DENSITY in (kHz)^{-1} (= ms) units, thus
	# we divide by 1000 to get it in per Hz
	AnalPowSpecE = AnalPowSpecE*2/1000

	df = fs[1]-fs[0]
	GammaPower = np.sum(AnalPowSpecE[(fs>GammaRange[0]) & (fs<GammaRange[1])]) *df # E gamma power

	return AnalPowSpecE, fs, GammaPower #, JacobLams, Jacob



def linear_PS_sameTime(ssn, rs, noise_pars, freq_range, fnums, cons, LFPrange=None, GammaRange=[20,100]):
    """
    Differs from linear_power_spect in that it solves for the power spectrum across frequencies at the same time. Avoids appends and over mutating operations. (for use in ML gamma)
    LFPrange = noise_pars.Ori1 + (-10:10) = indices which contribute to the LFP: activity (or voltage
    is averaged over those dimensions and power-spec of that avg is calculated
    noise_pars.stdevE = 1.5; Std of E noise
    noise_pars.stdevI = 1.5; Std of E noise
    noise_pars.corr_time = 5; correlation time of noise in ms
    noise_pars.corr_length = 0.5; correlation length of noise in angles; 0 doesn't work well..: too small response

    Trgt = round(noise_pars.Ori1/OriVec(end)*ssn.Nthetas);
    LFPrange = Trgt+(-10:+10);
    cons = number of contrasts 
    probes = number of probes 

    example run:
    powspecE = linear_power_spect(ssn, r_fp, NoisePars(), freq_range=[10,100], fnums=50)

    by Yashar Ahmadian -- Nov 2015. Modified by Caleb Holt -- Sep 2019
    """
    
    if LFPrange is None:
        LFPrange = [0]
    
    eE, noiseCov, SpatialFilt = make_eE_noiseCov(ssn, noise_pars, LFPrange)
        
    tau_corr = noise_pars.corr_time
    
    #switch-block case 'ampa-gaba-nmda' in MATLAB code
    J = ssn.DCjacobian(rs)
    
    eE1 = np.kron(np.ones(ssn.num_rcpt), eE) # this tensor product by ones(...) is because of the unweighted sum of currents of different types inside the neuronal nonlinearity
    # eI1 = np.kron(np.ones(ssn.num_rcpt), eI) # this tensor product by ones(...) is because of the unweighted sum of currents of different types inside the neuronal nonlinearity

    Jacob = ssn.jacobian(J) # np.kron(1/ssn.tau_s, np.ones(ssn.N))[:,None] * J  # equivalent to diag(tau_s) J (math)
    # JacobLams = np.linalg.eigvals(Jacob)

    maxF = freq_range[1]
    minF = freq_range[0]
    fs = np.linspace(minF,maxF,fnums) # grid of frequencies in Hz
    fs = fs/1000 # coverting from Hz to kHz = 1/ms
    cuE = np.array([eE1 for cc in range(cons) for ff in fs])
    #AnalPowSpecE = np.empty_like(fs)
    #AnalPowSpecE = []
    inv_G = np.array([-1j * 2 * np.pi * ff * np.diag(ssn.tau_s_vec) - J[cc] for cc in range(cons) for ff in fs])
    
    fscons = np.kron(np.ones([1, cons]), fs)
    noiseCov_fscons = np.kron(np.ones([1, fnums*cons]), noiseCov)
    
    
    vecE = np.linalg.solve(np.transpose(inv_G.conj(), [0,2,1]), cuE)
    
    if ssn.num_rcpt<3: # i.e. if we only have AMPA and GABA (AMPA always first)
    #     vecE = SpatialFilt'*reshape(vecE,[ssn.N,ssn.num_rcpt]) # accounting for spatial correlations in noise input
        # !!HERE we ASSUME noise is coming only through the AMPA (m=1) channel... modify LATER for general
        # make sure it's also correct to just ignore the rest of vecE and only take
        # its 1st 1/ssn.num_rcpt (corresponding to AMPA)
        if SpatialFilt.size>1:
            vecE = SpatialFilt.T  @ vecE[:,:ssn.N]
        else:
            vecE = SpatialFilt * vecE[:,:ssn.N]
    elif ssn.num_rcpt==3: # i.e. if we have NMDA (which is always the last part)
        # assuming AMPA and NMDA channels (up to scaling) get the exact same realization of noise (i.e. noise cov is rank-deficient)
        if SpatialFilt.size>1:
            vecE = SpatialFilt.T @ ((1-noise_pars.NMDAratio) * vecE[:,:ssn.N]  + noise_pars.NMDAratio * vecE[:,-ssn.N:])
        else:
            vecE = SpatialFilt * ((1-noise_pars.NMDAratio) * vecE[:,:ssn.N]  + noise_pars.NMDAratio * vecE[:,-ssn.N:])
    
#     vecE_conj = np.transpose(np.conj(vecE), [0,2,1])

#     y = (1-noise_pars.NMDAratio) * vecE[:, :ssn.N] + noise_pars.NMDAratio * vecE[:, -ssn.N:]
#     y_conj = np.transpose(np.conj(y), [0, 2, 1])

    # spect = np.einsum('ijk, imk -> ijm', y_conj, y)
    tapercons = 2 * tau_corr/np.abs(-1j * 2 * np.pi * fscons * tau_corr + 1)**2 #tau_corr is the pink noise time constant

    AnalPowSpecE = np.squeeze(np.einsum('ij, ij -> i',np.conj(vecE), noiseCov * vecE)) * np.squeeze(tapercons)
#     AnalPowSpecE = np.squeeze(np.dot(vecE.conj(), noiseCov * vecE.T)) * np.squeeze(tapercons)

    AnalPowSpecE = np.reshape(AnalPowSpecE*2/1000, [len(fs), cons], order='F')
#     AnalPowSpecE = AnalPowSpecE/np.mean(AnalPowSpecE)
    
    fs = 1000*fs
    df = fs[1]-fs[0]
    GammaPower = np.sum(AnalPowSpecE[(fs>GammaRange[0]) & (fs<GammaRange[1])]) *df # E gamma power
    
#     f0 = find_peak_freq(fs, AnalPowSpecE, 0)
    f0 = 0
    

    return AnalPowSpecE, fs, f0, Jacob #, GammaPower, JacobLams

def find_peak_freq(fs, spect, start_ind=4):
    '''
    Find's the peak frequency a la Ray and Maunsell. Subtracts the background spect (BS) 
    when contrast = 0, from each spect and 
    '''
    import numpy as onp
    spect = onp.asarray(spect)
    if onp.mean(onp.real(spect)) > 1:
        spect = onp.real(spect)/onp.mean(onp.real(spect))
    
    BS = spect[:, 0] #spectrum with no stimulus present; BS = Background Spectrum
    
    d_spect = spect[:, 1:] - BS[:, None] #difference between stimulus present and background spectrum
    
    return fs[onp.argmax(d_spect[start_ind:, :], axis=0) + start_ind]

def infl_find_peak_freq(fs, spect):
    '''
    function that finds the peak frequency using the inflectin point method. The peak frequency (f0) is assumed to be in the middle of two inflection pts.
    The half width is the distance in frequency between the two inflection points/2. 
    
    inputs
    fs = array of frequency
    spect = array of power spectra, dims are usually freq x contrasts(+probes)
    
    returns
    f0_out = greatest frequency with peak
    '''
    import numpy as onp
    
    nps = 6 # max number of inflection pts 
    nmaxpeaks = int(onp.floor(nps/2))
    cons = spect.shape[1]
    
    spect = onp.real(spect[:, :])
    Dspect = onp.diff(spect, axis=0, n=2)

    pos_curvature = onp.where(Dspect > 0, 1, 0)
    Inf_inds = onp.where(onp.diff(pos_curvature, axis=0) !=0, 1, 0)
    
    cc, ff = onp.nonzero(Inf_inds.T)
    # cc is a 1D array of contrast indices
    # ff is a 1D array of freq indices
    
    ff += 1 #because diff(spect) loses an index, i.e. if spect has length 30, dspect has length 28, so shift the freq indices up
    
    f0 = onp.empty((cons, nmaxpeaks))
    hw = onp.empty((cons, nmaxpeaks))
    #err = 1 means that power spectrum was too inflecty, i.e. jagged
    err = onp.zeros((cons))
    
    f0[:] = onp.nan
    hw[:] = onp.nan
    
    #jj indexes number of max peaks
    jj = 0
    
    #bunch of if statements to find the peak frequencies, f0, and  half width, hw
    for c in np.arange(len(cc)):
        #is there a peak before the end of frequencies
        if np.all(pos_curvature[ff[c]:, cc[c]] ==0):
            if cc[c] ==0:
                # report an error if I get a gamma bump at 0 contrast
                err[cc[c]] = 1
                
            #find f0 and hw when it doens't re-bend before fs = max(fs)
            #there still could be a bump, but not another inflection point cause I ran out of freqs
            if jj < nmaxpeaks:
                start_ind = ff[c]
                f0[cc[c], jj] = (fs[-1] + fs[start_ind])/2
                hw[cc[c], jj] = (fs[-1] - fs[start_ind])/2
            else:
                err[cc[c]] = 1
            #gotta reset jj
            jj = 0

        elif c == len(cc)-1:
            # if this is the last c, and it's negatively curved, just reset jj.
            jj = 0

        elif cc[c+1] == cc[c]: 
            if cc[c] == 0:
                # report an error if I get a gamma bump at 0 contrast
                err[cc[c]] = 1.4
            
            #find f0 and hw of gamma peak that re-bends before fs = max(fs)
            #if statement makes sure the contrasts doesn't change
            end_ind = ff[c+1]
            start_ind = ff[c]
            #make sure everything is negatively curved, i.e. not positively curved
            if np.all(pos_curvature[cc[c], start_ind:end_ind-1] == 0):
                if jj < nmaxpeaks:
                    f0[cc[c], jj] = (fs[end_ind] + fs[start_ind])/2
                    hw[cc[c], jj] = (fs[end_ind] - fs[start_ind])/2

                    jj+=1
                else:
                    err[cc[c]] = 1
        else:
            #catch all other cases
            jj = 0
        
    f0_out = onp.empty(cons)
    hw_out = onp.empty(cons)

    for c in onp.arange(cons):
        if onp.all(onp.isnan(f0[c])):
            f0_out[c] = onp.nan
            hw_out[c] = onp.nan
        else:
            f0_out[c] = onp.nanmax(f0[c])
            peak_ind = onp.nanargmax(f0[c])
            hw_out[c] = hw[c][peak_ind]
    
    return f0_out, hw_out, err
