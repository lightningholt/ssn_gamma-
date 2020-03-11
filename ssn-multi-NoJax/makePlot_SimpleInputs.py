import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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
    fig_RM = plt.figure(fignumber, figsize=(9, 6), constrained_layout=True)
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
    ax_spect_con.set_ylabel('Power-spectrum (a.u.)', fontsize=size_f)
    ax_spect_con.set_xlabel('Frequency (Hz)', fontsize=size_f)
    ax_spect_con.legend([r'$c = 0$', r'$c = 25$', r'$c = 50$', r'$c = 100$'], frameon=False, loc='upper right', ncol=2)

    ### Maun effect
    #plot stuff
    ax_spect_maun = fig_RM.add_subplot(gs[1,0:3])
    ax_spect_maun.set_prop_cycle('color', maun_color)
    ax_spect_maun.plot(new_fs, spect[new_inds, -probes:], lw= thick_line)
    #label stuff
    lstr = ['R = 0']
    for pp in range(1, probes):
        lstr.append('R = '+str(pp))
    ax_spect_maun.legend(lstr, loc='upper right', ncol=2, frameon=False)
    #ax_spect_maun.set_title("Locality of contrast dependence",fontsize=title_size)
    ax_spect_maun.set_xlabel('Frequency (Hz)', fontsize=size_f)
    ax_spect_maun.set_ylabel('Power-spectrum (a.u.)', fontsize=size_f)

    #rates contrasts plots
    #Plot stuff
    ax_EI = fig_RM.add_subplot(gs[0, 5:])
    ax_EI.set_prop_cycle('color', rates_color)
    ax_EI.plot(contrasts[:cons], rates[con_inds, :], '-o', lw= thick_line)
    ax_EI.set_prop_cycle('color', rates_color)
    #ax_EI.plot(contrasts[-1], rates[gabor_inds,0], '^', ms=big_mark)
    #ax_EI.plot(contrasts[-1], rates[gabor_inds,1], '^', ms=big_mark)
    #label stuff
    ax_EI.set_xlabel('Contrast', fontsize=size_f)
    ax_EI.set_ylabel('Firing rate (Hz)', fontsize=size_f)
    ax_EI.set_xticks(contrasts)
    ax_EI.legend(['Center E', 'Center I'], frameon=False)

    #rates surround suppression (SS)
    ax_SS = fig_RM.add_subplot(gs[1, 5:])
    ax_SS.set_prop_cycle('color', rates_color)
    ax_SS.plot(radii, rates[rad_inds, :], '-o', lw= thick_line)
    #label stuff
    ax_SS.set_xlabel('Stimulus radius (degrees)', fontsize=size_f)
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
    ax_con_f0.set_xlabel('Contrast', fontsize=size_f)
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
    ax_maun_f0.set_xlabel('Probe location', fontsize=size_f)
    ax_maun_f0.set_xticks(Pdist)
    ax_maun_f0.set_ylabel('Peak frequency (Hz)', fontsize=size_f)
    #ax_maun_f0.set_title('Locality of contrast dependence', fontsize=title_size)
    
    return fig_RM