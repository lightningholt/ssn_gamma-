import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np
from scipy.interpolate import interp1d
import matplotlib as mpl
from scipy.special import logsumexp

import SSN_power_spec

#removes bounding edges on the right and top of figures
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False

def RM_plot(params, rates, spect):
    '''
    This is a simple plot function that produces a figure showing the power-spectra
    (PS) for both contrast effect, and the locality of contrast dependence. It
    also shows the firing rates with contrast, and the suppression curve for the
    firing rates as the stimulus grows.

    Inputs:
    params - 1D array of parameters, assumed to be in [Jee, Jei, Jie, Jii, gE, gI, NMDAratio, PlocalEE, PlocalIE, sigEE, sigIE] order
    rates - 2D array of fixed point firing rates.  1st dimension is stimulus conditions, 2nd dimension is E and I neuorn at center of grid
    spect - 2D array of PS. 1st dimension is frequencies, 2nd dimension is contrasts + probes
    '''

    #My fixed parameters -- roughly in order they appear in plotting code below
    fnums = spect.shape[0]
    freq_range = [0.1, 100]
    fs = np.linspace(*freq_range, fnums)

    contrasts = np.array([0, 25, 50, 100])
    con_inds = (0, 1, 2, 6) #indices that correspond to max radius, and varying contrasts
    cons = len(contrasts)

    radii = np.arange(0, 1.25, 0.25) # in degrees -- gives [0, 0.25, 0.5, 0.75, 1]
    rad_inds = (0, 3, 4, 5, 6) #indices that correspond to max contrast, varying radius
    rads = len(radii)

    probes = 5
    Pdist = np.arange(probes) #Probe distance in number of neurons away
    gabor_inds = -1 #last index is typically Gabor -- for rates

    T = 1e-2 # to find softmax_r
    dx = 0.2 # degrees
    Gabor_sigma = 0.5 #degrees

    #Functions to find
    r_targ = rates[rad_inds, :]/np.mean(rates[rad_inds, :], axis=0)[None, :]
    softmax_r = T * logsumexp( r_targ / T )
    #for rates the last index is max radius and max contrast
    SI = 1 - (r_targ[-1, :]/softmax_r)

    #f0 is what I call peak freq, also outputs hw = halfwidth, and err = error.
    f0, _, _, infpt1, infpt2 = SSN_power_spec.infl_find_peak_freq(fs, spect)
    print(f0)

    probe_dist = dx * Pdist
    #GaborSigma = 0.3*np.max(radii)
    GaborSigma = 0.5
    Gabor_Cons = 100*np.exp(- probe_dist**2/2/GaborSigma**2); #for the linear interp1d

    #Figure Parameters
    fignumber = 29
    fig_RM = plt.figure(fignumber, figsize=(8,8), constrained_layout=True)
    con_color =  ['black', 'blue', 'green', 'red','gold']
    maun_color = ['gold', 'purple', 'green', 'maroon', 'xkcd:sky blue']
    rates_color = ['xkcd:orange', 'tab:cyan']

    gs = gridspec.GridSpec(5,3, figure=fig_RM)

    ## TIME to Plot

    ### contrast effect
    #plot stuff
    ax_spect_con = fig_RM.add_subplot(gs[0:2, 0:2])
    ax_spect_con.set_prop_cycle('color', con_color)
    ax_spect_con.plot(fs, spect[:, :cons])
    
    for cc in np.arange(1, cons):
        ax_spect_con.plot(infpt1[cc], spect[fs == infpt1[cc], cc], 'o', color=con_color[cc])
        ax_spect_con.plot(infpt2[cc], spect[ fs == infpt2[cc], cc], 'o', color = con_color[cc])
    
    # label stuff
    ax_spect_con.set_title('Contrast Effect')
    ax_spect_con.set_ylabel('Power-spectrum (a.u.)')
    ax_spect_con.set_xlabel('Frequency (Hz)')

    ### Maun effect
    #plot stuff
    ax_spect_maun = fig_RM.add_subplot(gs[2:4,0:2])
    ax_spect_maun.set_prop_cycle('color', maun_color)
    ax_spect_maun.plot(fs, spect[:, -probes:])
    #label stuff
    lstr = ['R = 0']
    for pp in range(1, probes):
        lstr.append('R = '+str(pp))
    ax_spect_maun.legend(lstr, loc='upper left', ncol=2)
    ax_spect_maun.set_title("Locality of contrast depenedence")
    ax_spect_maun.set_xlabel('Frequency (Hz)')
    ax_spect_maun.set_ylabel('Power-spectrum (a.u.)')

    #rates contrasts plots
    #Plot stuff
    ax_EI = fig_RM.add_subplot(gs[0, 2:])
    ax_EI.set_prop_cycle('color', rates_color)
    ax_EI.plot(contrasts[:cons], rates[con_inds, :])
    ax_EI.set_prop_cycle('color', rates_color)
    ax_EI.plot(contrasts[-1], rates[gabor_inds,0], '^')
    ax_EI.plot(contrasts[-1], rates[gabor_inds,1], '^')
    #label stuff
    ax_EI.set_xlabel('Contrast')
    ax_EI.set_ylabel('Firing rate (Hz)')
    ax_EI.set_xticks(contrasts)
    ax_EI.legend(['E', 'I'])

    #rates surround suppression (SS)
    ax_SS = fig_RM.add_subplot(gs[1, 2:])
    ax_SS.set_prop_cycle('color', rates_color)
    ax_SS.plot(radii, rates[rad_inds, :])
    #label stuff
    ax_SS.set_xlabel('Stimulus radius (degrees)')
    ax_SS.set_xticks(radii)
    ax_SS.set_ylabel('Firing rate (Hz)')
    tstr = 'SI = '+'({:.2f}, {:.2f})'.format(SI[0], SI[1]) #+', SI_I = '+'{:.2f}'.format(SI[1])
    ax_SS.set_title(tstr)
    ax_SS.set_ylim(bottom=0)

    #peak freq plots Contrast
    ax_con_f0 = fig_RM.add_subplot(gs[2, 2:])
    ax_con_f0.set_prop_cycle('color', con_color[1:])
    for cc in range(1, cons):
        ax_con_f0.plot(contrasts[cc], f0[cc],'o')
    #label stuff
    ax_con_f0.set_xlabel('Contrast')
    ax_con_f0.set_ylabel('Peak frequency (Hz)')


    #Peak freq Locality
    ax_maun_f0 = fig_RM.add_subplot(gs[3, 2:])
    ax_maun_f0.set_prop_cycle('color', maun_color)
    for pp in range(0, probes):
        ax_maun_f0.plot(Pdist[pp], f0[-probes+pp],'o')
    fit_f0 = interp1d(contrasts[1:], f0[1:cons], fill_value='extrapolate')
    ax_maun_f0.plot(Pdist, fit_f0(Gabor_Cons),'gray')
    #label stuff
    ax_maun_f0.set_xlabel('Probe location')
    ax_maun_f0.set_xticks(Pdist)
    ax_maun_f0.set_ylabel('Peak frequency (Hz)')
    
    ## MAKE parameter histogram
    W0 = 1.8
    g0 = 0.44

    #new ranges
    Jxe_max = W0*1.5
    Jxi_max = W0*1.5 * .5
    g_max = g0*1.5
    g_min = g0*.5
    NMDA_max = 1
    plocal_max = 1
    sigR_max = 1.5
    sigEE_max = 0.5
    sigIE_max = 0.5

    #Normalize Parameters
    params_max = np.array([Jxe_max, Jxi_max, Jxe_max, Jxi_max, g_max, g_max, NMDA_max, plocal_max, plocal_max, sigEE_max, sigIE_max])
    params = params/params_max

    #histogram parameters
    bar_pos = np.arange(len(params))
    width = 0.35
    label_params = ['Jee', 'Jei', 'Jie', 'Jii', 'gE', 'gI', 'NMDA','Plocal_EE', 'Plocal_IE', 'sigEE', 'sigIE']

    ax_params = fig_RM.add_subplot(gs[4,:])
    ax_params.bar(bar_pos, params, label='Normalized Parameters')
    #label stuff
    ax_params.set_ylabel('Normalized Parameters')
    ax_params.set_xticks(bar_pos)
    ax_params.set_xticklabels(label_params)
    ax_params.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', ncol=2, mode="expand", borderaxespad=0.)
    ax_params.set_ylim(top=1)

    return fig_RM


def fig3_RM(params, rates, fs, spect):
    '''
    This is a simple plot function that produces a figure showing the power-spectra
    (PS) for both contrast effect, and the locality of contrast dependence. It
    also shows the firing rates with contrast, and the suppression curve for the
    firing rates as the stimulus grows.

    Inputs:
    params - 1D array of parameters, assumed to be in [Jee, Jei, Jie, Jii, gE, gI, NMDAratio, PlocalEE, PlocalIE, sigEE, sigIE] order
    rates - 2D array of fixed point firing rates.  1st dimension is stimulus conditions, 2nd dimension is E and I neuorn at center of grid
    spect - 2D array of PS. 1st dimension is frequencies, 2nd dimension is contrasts + probes
    '''

    #My fixed parameters -- roughly in order they appear in plotting code below
    fnums = spect.shape[0]
    freq_important = [15, 100]
    new_fs = [ff for ff in fs if ff > freq_important[0] and ff < freq_important[1]]
    new_inds = [ii for ii in np.arange(fnums) if fs[ii] > freq_important[0] and fs[ii] < freq_important[1]]
    
    size_f = 12
    title_size = 15
    big_mark = 8
    thick_line = 2.5
    
    contrasts = np.array([0, 25, 50, 100])
    con_inds = (0, 1, 2, 6) #indices that correspond to max radius, and varying contrasts
    cons = len(contrasts)

    radii = np.arange(0, 1.25, 0.25) # in degrees -- gives [0, 0.25, 0.5, 0.75, 1]
    rad_inds = (0, 3, 4, 5, 6) #indices that correspond to max contrast, varying radius
    rads = len(radii)

    probes = 5
    Pdist = np.arange(probes) #Probe distance in number of neurons away
    gabor_inds = -1 #last index is typically Gabor -- for rates

    T = 1e-2 # to find softmax_r
    dx = 0.2 # degrees
    Gabor_sigma = 0.5 #degrees

    #Functions to find
    r_targ = rates[rad_inds, :]/np.mean(rates[rad_inds, :], axis=0)[None, :]
    softmax_r = T * logsumexp( r_targ / T )
    #for rates the last index is max radius and max contrast
    SI = 1 - (r_targ[-1, :]/softmax_r)

    #f0 is what I call peak freq, also outputs hw = halfwidth, and err = error.
    f0, _, _, infpt1, infpt2 = SSN_power_spec.infl_find_peak_freq(fs, spect)
    print(f0)

    probe_dist = dx * Pdist
    #GaborSigma = 0.3*np.max(radii)
    GaborSigma = 0.5
    Gabor_Cons = 100*np.exp(- probe_dist**2/2/GaborSigma**2); #for the linear interp1d

    #Figure Parameters
    fignumber = 29
    fig_RM = plt.figure(fignumber, figsize=(10, 6), constrained_layout=True)
    con_color =  ['black', 'blue', 'green', 'red','gold']
    maun_color = ['tab:orange', 'indigo', 'green', 'm', 'xkcd:sky blue']
    rates_color = ['xkcd:orange', 'tab:cyan']

    gs = gridspec.GridSpec(2,7, figure=fig_RM)

    ## TIME to Plot

    ### contrast effect
    #plot stuff
    ax_spect_con = fig_RM.add_subplot(gs[0, 0:3])
    ax_spect_con.set_prop_cycle('color', con_color)
    ax_spect_con.plot(new_fs, spect[new_inds, :cons], lw= thick_line)
    
#     for cc in np.arange(1, cons):
#         ax_spect_con.plot(infpt1[cc], spect[fs == infpt1[cc], cc], 'o', color=con_color[cc])
#         ax_spect_con.plot(infpt2[cc], spect[ fs == infpt2[cc], cc], 'o', color = con_color[cc])
    
    # label stuff
    #ax_spect_con.set_title('Contrast effect', fontsize=title_size)
    ax_spect_con.set_ylabel('LFP power (a.u.)', fontsize=size_f)
    ax_spect_con.set_xlabel('Frequency (Hz)', fontsize=size_f)
    ax_spect_con.legend([r'$c = 0 \%$', r'$c = 25\%$', r'$c = 50\%$', r'$c = 100\%$'], frameon=False, loc='upper right', ncol=2)

    ### Maun effect
    #plot stuff
    ax_spect_maun = fig_RM.add_subplot(gs[1,0:3])
    ax_spect_maun.set_prop_cycle('color', maun_color)
    ax_spect_maun.plot(new_fs, spect[new_inds, -probes:], lw= thick_line)
    #label stuff
    lstr = ['R = 0\xb0']
    for pp in range(1, probes):
        lstr.append('R = {dist:.2f}\xb0'.format(dist=pp*dx))
    ax_spect_maun.legend(lstr, loc='upper right', ncol=1, frameon=False)
    #ax_spect_maun.set_title("Locality of contrast dependence",fontsize=title_size)
    ax_spect_maun.set_xlabel('Frequency (Hz)', fontsize=size_f)
    ax_spect_maun.set_ylabel('LFP power (a.u.)', fontsize=size_f)

    #rates contrasts plots
    #Plot stuff
    ax_EI = fig_RM.add_subplot(gs[0, 5:])
    ax_EI.set_prop_cycle('color', rates_color)
    ax_EI.plot(contrasts[:cons], rates[con_inds, :], '-o', lw= thick_line)
    ax_EI.set_prop_cycle('color', rates_color)
    #ax_EI.plot(contrasts[-1], rates[gabor_inds,0], '^', ms=big_mark)
    #ax_EI.plot(contrasts[-1], rates[gabor_inds,1], '^', ms=big_mark)
    #label stuff
    ax_EI.set_xlabel('Contrast (%)', fontsize=size_f)
    ax_EI.set_ylabel('Firing rate (Hz)', fontsize=size_f)
    ax_EI.set_xticks(contrasts)
    ax_EI.legend(['Center E', 'Center I'], frameon=False)

    #rates surround suppression (SS)
    ax_SS = fig_RM.add_subplot(gs[1, 5:])
    ax_SS.set_prop_cycle('color', rates_color)
    ax_SS.plot(radii, rates[rad_inds, :], '-o', lw= thick_line)
    #label stuff
    ax_SS.set_xlabel('Stimulus radius (\xb0)', fontsize=size_f)
    ax_SS.set_xticks(radii)
    ax_SS.set_ylabel('Firing rate (Hz)', fontsize=size_f)
    #tstr = 'SI = '+'({:.2f}, {:.2f})'.format(SI[0], SI[1]) #+', SI_I = '+'{:.2f}'.format(SI[1])
    #ax_SS.set_title('Suppression curve', fontsize=title_size)
    ax_SS.set_ylim(bottom=0)

    #peak freq plots Contrast
    ax_con_f0 = fig_RM.add_subplot(gs[0, 3:5])
    fit_f0 = interp1d(contrasts[1:], f0[1:cons], fill_value='extrapolate')
    ax_con_f0.plot(contrasts[1:], fit_f0(contrasts[1:]), 'k', lw=thick_line)
    ax_con_f0.set_prop_cycle('color', con_color[1:])
    for cc in range(1, cons):
        ax_con_f0.plot(contrasts[cc], f0[cc],'o', ms=big_mark)
    #label stuff
    ax_con_f0.set_xlabel('Contrast (%)', fontsize=size_f)
    ax_con_f0.set_xticks(contrasts[1:])
    ax_con_f0.set_ylabel('Peak frequency (Hz)', fontsize=size_f)
    #ax_con_f0.set_ylim(bottom=np.min(f0[1:])-5)
    #ax_con_f0.set_xlim(left=np.min(contrasts[1:])-15)
    #ax_con_f0.set_title('Contrast effect', fontsize=title_size)


    #Peak freq Locality
    ax_maun_f0 = fig_RM.add_subplot(gs[1, 3:5])
    ax_maun_f0.plot(Pdist, fit_f0(Gabor_Cons), 'gray', label='Prediction', lw= thick_line)
    ax_maun_f0.set_prop_cycle('color', maun_color)
    for pp in range(0, probes):
        ax_maun_f0.plot(Pdist[pp], f0[-probes+pp],'o', label='_nolegend_', ms=big_mark)
    
    
    #label stuff
    ax_maun_f0.legend(frameon=False)
    ax_maun_f0.set_xlabel('Probe location (\xb0)', fontsize=size_f)
    ax_maun_f0.set_xticks(Pdist)
    labelstring = []
    for pp in Pdist*dx:
        labelstring.append('{dist:.2f}'.format(dist= pp))
    ax_maun_f0.set_xticklabels(labelstring)
    ax_maun_f0.set_ylabel('Peak frequency (Hz)', fontsize=size_f)
    #ax_maun_f0.set_title('Locality of contrast dependence', fontsize=title_size)
    
    return fig_RM

# ========================================================================

def hists_fig4(obs_rates, f0s, min_freq=10, dfdc=False):
    
    '''
    obs_rates dimensions are samples, 1 for some reason, neuron, stimConds
    f0s dimensions are sample, ff/hw/err/inflpt1/inflpt2, stimConds
    ff is peak frequency found using either infl or RM method. 
    gg is peak width
    '''
    
    fs = 18
    ls = 11
    ss = 15
    aa = 0.3 #alpha value
    
    contrasts = np.array([0, 25, 50, 100])
    cons = len(contrasts)
    probes = 5
    
    con_inds = np.arange(0, cons)
    probe_inds = np.arange(cons, f0s.shape[2])
    
    #max(axis=2) is because lams is a 6-dim vector for each contrast and instantiation
    ff_con = f0s[:, 0, con_inds]
    dff_con = np.diff(ff_con, axis=1)
    
    ff_probe = f0s[:, 0, probe_inds]
    dff_probe = np.diff(ff_probe, axis=1)
    
    hw_con = f0s[:, 1, con_inds]
    hw_probe = f0s[:, 1, probe_inds]
    
    #for rates 
    con_inds_rates = np.array([0,1,2,6])
    rad_inds = np.array([3,4,5,6])
    
    trgt = int(np.floor(obs_rates.shape[2]/4))
    trgt = (trgt, trgt + int(np.floor(obs_rates.shape[2]/2)))
    
    rs = obs_rates[:, 0, trgt, :]
    
    T = 1e-2
    r_targ = rs[:, :, rad_inds]/np.mean(rs[:, :, rad_inds], axis=2)[:, :, None]
    softmax_r = T * logsumexp( r_targ / T, axis=2 ) 
    obs_SI = 1 - (r_targ[:,:,-1]/softmax_r)
    
    dx = 0.2 # degrees
    Pdist = np.arange(probes)
    probe_dist = dx * Pdist
    #GaborSigma = 0.3*np.max(radii)
    GaborSigma = 0.5
    Gabor_Cons = 100*np.exp(- probe_dist**2/2/GaborSigma**2); #for the linear interp1d

    fit_f0 = interp1d(contrasts[1:], ff_con[:,1:])
    yfit = fit_f0(Gabor_Cons)

    SSE = np.sum( (ff_probe - yfit) **2, axis=1) #SSE = sum of squared error
    ff_probe_var = np.sum((ff_probe-np.nanmean(ff_probe, axis=1)[:, None])**2, axis=1)
    R2 = 1- SSE/ff_probe_var
    #R2 = np.array([rr for rr in R2 if rr is not np.isnan(rr)])
    R2 = np.array([rr for rr in R2 if rr is not np.isnan(rr) and not np.isinf(rr)])
    R2 = np.array([rr for rr in R2 if rr > -10])
    
    medR2 = np.median(R2)
    
    min_freq = 10 #Hz
    min_width = 0
    
    con_color = ['black', 'blue','green','red'] #colors for contrasts 0, 25, 50, 100,
    shift_colors = ['tab:cyan', 'gold'] #blue and green make cyan, red and green make yellow
    maun_color = ['tab:orange', 'indigo', 'green', 'm', 'xkcd:sky blue']
    maunshift = ['r', 'b', 'g', 'k']
    rates_color = ['xkcd:orange', 'tab:cyan']
    cons = len(con_color)
    contrasts = np.array([0, 25, 50, 100])
    histstyle_paper = {"histtype": "step", "linewidth": 2.25, "density": False}
    histstyle_nb = {"histtype": "bar", "linewidth":2, "alpha": aa, "density": False}

    histstyle = histstyle_paper
    
    nbins = 30

    ctypes = ["E", "I"]
    #fig, axs = plt.subplots(3,4, figsize=(16,6))
    
    fighists = plt.figure(27, constrained_layout=True, figsize=(14, 7))
    gs = gridspec.GridSpec(2,3, figure=fighists)
    
    axErates = fighists.add_subplot(gs[0,0])
    axIrates = fighists.add_subplot(gs[1,0])
    axf0 = fighists.add_subplot(gs[0,1])
    axf0shifts = fighists.add_subplot(gs[1,1])
#     axMaun = fighists.add_subplot(gs[0,2])
#     axMaunShifts = fighists.add_subplot(gs[1,])
    axSIrates = fighists.add_subplot(gs[0,2])
    axFit = fighists.add_subplot(gs[1,2])
    
    Emed = np.median(rs[:,0, :],axis=0)
    print(Emed)
    Imed = np.median(rs[:,:, 1],axis=0)
    print(Imed)
    
    lbound = 0.7
    rbound = 1500
    
    n_f0 = 1000*np.ones(nbins)
    n_hw = 1000*np.ones(nbins)
    
    #make the hists below here..............
    dcon = np.diff(contrasts[1:])
    
    # BINS
    bFreqs = np.arange(0,80, 3)
    bDFs = np.linspace(0,1,16)
    #_, bDFs = np.histogram(dff_con[~np.isnan(dff_con[:,0]),0]/dcon[0], nbins//2)
    _, bSI = np.histogram(obs_SI[~np.isnan(obs_SI[:,1]), 1], nbins)
    #bin width for SI = 0.02267407
    
    _, binsE = np.histogram(rs[~np.isnan(rs[:,0,1]), 0, 1], bins=nbins, density= True)
    _, binsI = np.histogram(rs[~np.isnan(rs[:,1,1]), 1, 1], bins=nbins, density= True)
    logbinsE = np.logspace(np.log10(binsE[0]),np.log10(binsE[-1]), len(binsE))
    logbinsI = np.logspace(np.log10(binsI[0]), np.log10(binsI[-1]), len(binsI))
    bluebinE = 2 * np.diff(np.log10(logbinsE))[0]
    bluebinI = 2 * np.diff(np.log10(logbinsI))[0]
    print(bluebinE)
    
    for c in np.arange(1, cons):
        
        axf0.hist(ff_con[:, c], bins=bFreqs, color=con_color[c], **histstyle_nb)
        axf0.hist(ff_con[:, c], bins=bFreqs, color=con_color[c], **histstyle)
        
        if c > 1:
            if c == 2:
                scale = 1
            else:
                scale = 1
            
            axf0shifts.hist(scale*dff_con[:,c-1]/dcon[c-2], bins=bDFs, color= shift_colors[c-2], **histstyle)
            axf0shifts.hist(scale*dff_con[:,c-1]/dcon[c-2], bins=bDFs, color= shift_colors[c-2], **histstyle_nb)
            #axhwshifts.hist(dhw, nbins, color= shift_colors[c-2], **histstyle)
            #axhwshifts.hist(dhw, nbins, color= shift_colors[c-2], **histstyle_nb)
            
        _, binsE = np.histogram(rs[:, 0, c], bins=nbins, density= True)
        _, binsI = np.histogram(rs[:, 1, c], bins=nbins, density= True)
        logbinsE = 10**(np.arange(np.log10(binsE[0]), np.log10(binsE[-1]) + bluebinE, bluebinE))
        logbinsI = 10**(np.arange(np.log10(binsI[0]), np.log10(binsI[-1]) + bluebinI, bluebinI))
    
        axErates.hist(rs[:, 0, c], bins=logbinsE, color=con_color[c], **histstyle)
        axErates.hist(rs[:, 0, c], bins=logbinsE, color=con_color[c], **histstyle_nb)
        axIrates.hist(rs[:, 1, c], bins=logbinsI, color=con_color[c], **histstyle)
        axIrates.hist(rs[:, 1, c], bins=logbinsI, color=con_color[c], **histstyle_nb)
        
#     for p in np.arange(probes):
        
#         axMaun.hist(ff_probe[:, p], nbins, color=maun_color[p], **histstyle_nb)
#         axMaun.hist(ff_probe[:, p], nbins, color=maun_color[p], **histstyle)
        
#         if p > 0:
#             axMaunShifts.hist(dff_probe[:, p-1], nbins, color= maunshift[p-1], **histstyle)
#             axMaunShifts.hist(dff_probe[:, p-1], nbins, color= maunshift[p-1], **histstyle_nb)
    
    for celltype in np.arange(len(ctypes)):
        axSIrates.hist(obs_SI[:, celltype], bins=bSI, color=rates_color[celltype], **histstyle)
        axSIrates.hist(obs_SI[:, celltype], bins=bSI, color=rates_color[celltype], **histstyle_nb)
    
    fit_bins = [0.8, 0.9, 1]
    R3 = np.maximum(R2, 0.9)
    goodR3 = np.sum(R3 > 0.9)
    badR3 = np.sum(R3==0.9)
    axFit.bar([0.9, 1], [badR3, goodR3], width=0.08, color=['r','forestgreen'], edgecolor='k', alpha=2*aa)
    #axFit.hist(R3[R3 > 0.9], bins = fit_bins, color='forestgreen', histtype='bar', alpha=aa, align='right')
    #axFit.hist(R3[R3 > 0.9], bins = fit_bins, color='forestgreen', histtype='step', lw=2*2, align='right')
    #axFit.hist(R3[R3 == 0.9], bins = fit_bins, color ='red', histtype='bar', alpha=aa, align='left' )
    #axFit.hist(R3[R3 == 0.9], bins = fit_bins, color ='red', histtype='step', lw=2*2, align='left' )
    #axFit.axvline(medR2, ls= '--',color='k')
    axInset = axFit.inset_axes([0.1, 0.6, 0.35, 0.35])
    axInset.hist(R2, nbins, color='forestgreen', **histstyle)
    axInset.hist(R2, nbins, color='forestgreen', **histstyle_nb)
    
    # =============
    # =============
    #cosmetic stuff below
    #shifts to the A,B,C, etc of the figure, where they appear relative to axes
    
    x_text = -0.25
    y_text = 1.05
    y_label= "Counts"
    
    axErates.set_xlim(left=lbound, right=rbound)
    axErates.set_xscale('log')
    #axErates.set_xlabel('$log_{10}$(Firing rates (Hz))', fontsize=fs)
    axErates.set_xlabel('Firing rate (Hz)', fontsize=ss)
    axErates.set_ylabel(y_label, fontsize=ss)
    #axErates.set_title('Excitatory cell', fontsize=fs)
    p25 = mpatches.Rectangle([1,1], 1, 1, facecolor=con_color[1], edgecolor=con_color[1], alpha=aa, lw=0.1, label=r'$c$ = 25%')
    p50 = mpatches.Rectangle([1,1], 1, 1, facecolor=con_color[2], edgecolor=con_color[2], alpha=aa, lw=0.1, label=r'$c$ = 50%')
    p100 = mpatches.Rectangle([1,1], 1, 1, facecolor=con_color[3], edgecolor=con_color[3], alpha=aa, lw=0.1, label=r'$c$ = 100%')
    axErates.legend(handles=[p25, p50, p100], frameon=False, ncol=1, fontsize=ls)
    axErates.text(x_text, y_text, 'A', transform=axErates.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')
    #axErates.legend(['C = 25', 'C = 50', 'C = 100'], frameon=False, loc='upper left', ncol=1, fontsize=ls)
    
    axIrates.set_xlim(left=lbound, right=rbound)
    axIrates.set_xscale('log')
    axIrates.set_xlabel('Firing rate (Hz)', fontsize=ss)
    axIrates.set_ylabel(y_label, fontsize=ss)
    #axIrates.set_title('Inhibitory cell', fontsize=fs)
    axIrates.text(x_text, y_text, 'D', transform=axIrates.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')
    
    axf0.set_xlabel('Peak frequency (Hz)', fontsize=ss)
    axf0.set_ylabel(y_label, fontsize=ss)
    #axf0.set_title('Gamma peak frequency', fontsize=fs)
    axf0.text(x_text, y_text,'B', transform=axf0.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')
    
    axf0shifts.set_xlabel(r'$\Delta$Peak freq. / $\Delta c$ (Hz / %)', fontsize=ss)
    axf0shifts.set_ylabel(y_label, fontsize=ss)
    pdelta50 = mpatches.Rectangle([1,1], 1, 1, facecolor=shift_colors[0], edgecolor=shift_colors[0], alpha=aa, lw=0.1, label='$\Delta c =$50%-25%')
    pdelta100 = mpatches.Rectangle([1,1], 1, 1, facecolor=shift_colors[1], edgecolor=shift_colors[1], alpha=aa, lw=0.1, label='$\Delta c =$100%-50%')
    axf0shifts.legend(handles=[pdelta50, pdelta100], fontsize=ls, frameon=False, )
    #axf0shifts.legend(['$\Delta C =$50-25', '$\Delta C =$100-50'], fontsize=ls, frameon=False)
    #axf0shifts.set_title('Shift in gamma peak frequency', fontsize=fs)
    axf0shifts.text(x_text, y_text, 'E', transform=axf0shifts.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')
    
    axSIrates.set_xlabel('Suppression index', fontsize=ss)
    axSIrates.set_ylabel(y_label, fontsize=ss)
    #axSIrates.set_title('E/I suppression index', fontsize =fs)
    axSIrates.text(x_text, y_text,'C', transform=axSIrates.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')
    pE = mpatches.Rectangle([1,1], 1, 1, facecolor=rates_color[0], edgecolor=rates_color[0], alpha=aa, lw=0.1, label='center E')
    pI = mpatches.Rectangle([1,1], 1, 1, facecolor=rates_color[1], edgecolor=rates_color[1], alpha=aa, lw=0.1, label='center I')
    axSIrates.legend(handles=[pE, pI], frameon=False, ncol=1, fontsize=ls)
    
    axFit.set_xlabel(r'$R^2$', fontsize=fs)
    axFit.set_xticks([0.9, 1])
    axFit.set_xticklabels(['<0.9', '[0.9,1]'], fontsize=ss)
    axFit.set_xlim([0.85,1.05])
    axFit.set_ylabel(y_label, fontsize=fs)
    #axFit.set_title(r'Linear fit $R^2$', fontsize=fs)
    axFit.text(x_text, y_text,'F', transform=axFit.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')
    #axhwshifts.legend(['$\Delta C =$50-25', '$\Delta C =$100-50'], fontsize=ls, frameon=False)
    axInset.set_yticks([])
    axInset.set_xticks([-2, -1, 0, 1])
    
    return fighists