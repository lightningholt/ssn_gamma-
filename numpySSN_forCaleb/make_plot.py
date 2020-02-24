import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np
from scipy.interpolate import interp1d


def power_spect_rates_plot(fs, obs_spect, target_spect, contrasts, obs_rates, target_rates, initial_spect=None, initial_rates= None, lower_bound = 0, upper_bound = 80, fname = None):
    
    cons = len(contrasts)
    fig_combined = plt.figure(13, constrained_layout=True)
#     if loss_t is None:
    if initial_spect is None:
        gs = gridspec.GridSpec(4,3, figure=fig_combined)
    else:
        gs = gridspec.GridSpec(8, 3, figure=fig_combined)
#     else:
#         if initial_spect is None:
#             gs = gridspec.GridSpec(6, 3, figure=fig_combined)
#         else:
#             gs = gridspec.GridSpec(10, 3, figure=fig_combined)

    ax_spect_targ = fig_combined.add_subplot(gs[0:4,0:2])
    ax_spect_targ.plot(fs, target_spect, "--")
    #colrs = ax.get_color_cycle()
    ax_spect_targ.set_prop_cycle(None)
    ax_spect_targ.plot(fs, obs_spect)
    
    ax_spect_targ.set_title('post gradient descent')
    ax_spect_targ.set_ylabel('power spectrum (a.u.)')
    ax_spect_targ.set_xlabel('frequency (Hz)')    
    
    if initial_spect is not(None):
        ax_spect_init = fig_combined.add_subplot(gs[4:8,0:2])
        ax_spect_init.plot(fs, target_spect, ":")
        ax_spect_init.set_prop_cycle(None)
        ax_spect_init.plot(fs, initial_spect)
        
        ax_spect_init.set_title('Pre Gradient Descent')
        ax_spect_init.set_ylabel('power spectrum (a.u.)')
        ax_spect_init.set_xlabel('frequency (Hz)')   
        
    if isinstance(lower_bound, int):
        lower_bound = lower_bound * ones_like(obs_rates.T)
    if isinstance(upper_bound, int):
        upper_bound = upper_bound * ones_like(obs_rates.T)    
    
    ax_E = fig_combined.add_subplot(gs[0:2, 2])
    ax_I = fig_combined.add_subplot(gs[2:4, 2])

    ax_E.plot(contrasts, target_rates[:, 0],  "r--")
    ax_E.set_prop_cycle(None)
    ax_E.plot(contrasts, obs_rates[:,0], 'r')
    ax_E.fill_between(contrasts[-np.max(lower_bound.shape):cons], lower_bound[0,:], upper_bound[0,:], color="r", alpha=0.3)
    ax_E.set_title('E Rates Post GD')
    # ax_E.set_xlabel('Contrasts')
    # ax_E.set_ylabel('Firing Rates (Hz)')

    ax_I.plot(contrasts, target_rates[:, 1],  "b--")
    ax_I.set_prop_cycle(None)
    ax_I.plot(contrasts, obs_rates[:,1], 'b')
    ax_I.fill_between(contrasts[-np.max(lower_bound.shape):cons], lower_bound[1,:], upper_bound[1,:], color="b", alpha=0.3)
    ax_I.set_title('I Rates Post GD')
    
    
    if initial_rates is not(None):
        ax_E_init = fig_combined.add_subplot(gs[4:6, 2])
        ax_I_init = fig_combined.add_subplot(gs[6:8, 2])
        
        ax_E_init.plot(contrasts, initial_rates[:,0], "r")
        ax_E_init.set_prop_cycle(None)
        ax_E_init.plot(contrasts, target_rates[:,0], "r:")
        ax_E_init.fill_between(contrasts[-np.max(lower_bound.shape):cons], lower_bound[0,:], upper_bound[0,:], color="r", alpha=0.3,)
        
        ax_I_init.plot(contrasts, initial_rates[:,1], "b")
        ax_I_init.set_prop_cycle(None)
        ax_I_init.plot(contrasts, target_rates[:,1], "b:")
        ax_I_init.fill_between(contrasts[-np.max(lower_bound.shape):cons], lower_bound[1,:], upper_bound[1,:], color="b", alpha=0.3)
        
        ax_E_init.set_title('E Rates Pre')
        ax_I_init.set_title('I Rates Pre')
        
#     if loss_t is not None:
#         ax_loss = fig_combined.add_subplot(gs[-2:, :])
#         ax_loss.plot(loss_t)
#         ax_loss.set_xlabel('loss')
    
    if fname is not None:
        plt.savefig(fname)
    
    # ax_I.set_xlabel('Contrasts')
    # ax_I.set_ylabel('Firing Rates (Hz)')


def PS_2D(fs, obs_spect, contrasts, obs_rates, obs_f0, fname = None):
    
    cons = len(contrasts)
    con_inds = np.array([0,1,2,3])
    gabor_inds = -1
    
    fig_combined = plt.figure(17, constrained_layout=True)
    con_color =  ['black', 'blue', 'green', 'red','gold']
    maun_color = ['gold', 'purple', 'green', 'maroon', 'xkcd:sky blue']
    rates_color = ['red', 'blue']
    
    gs = gridspec.GridSpec(2,3, figure=fig_combined)
    
    ax_spect = fig_combined.add_subplot(gs[0:2,0:2])
    ax_spect.set_prop_cycle('color', con_color)
    ax_spect.plot(fs, obs_spect)
    
    ax_spect.set_title('Constrast Effect')
    ax_spect.set_ylabel('Power spectrum (a.u.)')
    ax_spect.set_xlabel('Frequency (Hz)')
    lstr = []
    for pp in contrasts:
        lstr.append('C = '+str(pp))
    ax_spect.legend(lstr, loc='upper left', ncol=2)
    
    ax_EI = fig_combined.add_subplot(gs[0, 2:])
    ax_EI.set_prop_cycle('color', rates_color)
    ax_EI.plot(contrasts[:cons], obs_rates[con_inds, :])
    ax_EI.set_prop_cycle('color', rates_color)
    if cons > len(con_inds):
        ax_EI.plot(contrasts[-1], obs_rates[gabor_inds,0], '^')
        ax_EI.plot(contrasts[-1], obs_rates[gabor_inds,1], '^')
    ax_EI.set_xlabel('Contrast')
    ax_EI.set_ylabel('Firing rate (Hz)')
    ax_EI.set_xticks(contrasts)
#     ax_EI.set_title('Firing Rates')
    ax_EI.legend(['E', 'I'])
    
    #peak freq plots
    ax_con_f0 = fig_combined.add_subplot(gs[1, 2:])
    ax_con_f0.set_prop_cycle('color', con_color[1:])
    for cc in range(1, cons):
        ax_con_f0.plot(contrasts[cc], obs_f0[cc - 1],'o')
    ax_con_f0.set_xlabel('Contrast')
    ax_con_f0.set_xticks(contrasts)
    ax_con_f0.set_ylabel('Peak frequency (Hz)')
    
    if fname is not None:
        plt.savefig(fname)
    
    return fig_combined
    
    
def Maun_Con_plots(fs, obs_spect, target_spect, contrasts, obs_rates, stim, obs_f0, initial_spect=None, initial_rates= None, initial_f0 = None, probes=5, fname=None):
    
    cons = len(contrasts)
    fig_combined = plt.figure(15, constrained_layout=True)
    con_color =  ['black', 'blue', 'green', 'red','gold']
    maun_color = ['gold', 'purple', 'green', 'maroon', 'xkcd:sky blue']
#     if loss_t is None:
    if initial_spect is not None:
        gs = gridspec.GridSpec(8,6, figure=fig_combined)
    else:
        gs = gridspec.GridSpec(8,3, figure=fig_combined)
    
    ax_spect_con = fig_combined.add_subplot(gs[0:4,0:2])
    ax_spect_con.set_prop_cycle('color', con_color)
    ax_spect_con.plot(fs, target_spect[:, :cons-1], "--")
    ax_spect_con.plot(fs, target_spect[:, cons-1], '*')
    #colrs = ax.get_color_cycle()
    ax_spect_con.set_prop_cycle(None)
    ax_spect_con.set_prop_cycle('color', con_color)
    ax_spect_con.plot(fs, obs_spect[:, :cons-1])
    ax_spect_con.plot(fs, obs_spect[:, cons-1], 'o')
    
    ax_spect_con.set_title('Contrast Effect - Post GD')
    ax_spect_con.set_ylabel('Power Spectrum (a.u.)')
    ax_spect_con.set_xlabel('frequency (Hz)')  
    
    if initial_spect is not(None):
        ax_con_init = fig_combined.add_subplot(gs[0:4,3:5])
        ax_con_init.set_prop_cycle('color', con_color)
        ax_con_init.plot(fs, target_spect[:, :cons-1], ":")
        ax_con_init.plot(fs, target_spect[:, cons-1], "*")
        ax_con_init.set_prop_cycle(None)
        ax_con_init.set_prop_cycle('color', con_color)
        ax_con_init.plot(fs, initial_spect[:,:cons-1])
        ax_con_init.plot(fs, initial_spect[:,cons-1], 'o')
        
        ax_con_init.set_title('Contrast Effect - Pre GD')
        ax_con_init.set_ylabel('power spectrum (a.u.)')
        ax_con_init.set_xlabel('frequency (Hz)')
    
    ax_E = fig_combined.add_subplot(gs[0:2, 2])
    ax_I = fig_combined.add_subplot(gs[2:4, 2])
    
    ax_E.set_prop_cycle(None)
    ax_E.plot(contrasts[:cons-1], obs_rates[:cons-1,0], 'r')
    ax_E.plot(contrasts[cons-1], obs_rates[cons-1, 0], 'r^')
    ax_E.set_title('E Rates Post GD')
    
    ax_I.plot(contrasts[:cons-1], obs_rates[:cons-1,1], 'b')
    ax_I.plot(contrasts[cons-1], obs_rates[cons-1,1], 'b^')
    ax_I.set_title('I Rates Post GD')
    
    if initial_rates is not None:
        ax_E_init = fig_combined.add_subplot(gs[0:2, 5])
        ax_I_init = fig_combined.add_subplot(gs[2:4, 5])

        ax_E_init.set_prop_cycle(None)
        ax_E_init.plot(contrasts[:cons-1], initial_rates[:cons-1,0], 'r')
        ax_E_init.plot(contrasts[cons-1], initial_rates[cons-1, 0], 'r^')
        ax_E_init.set_title('E Rates Pre GD')

        ax_I_init.plot(contrasts[:cons-1], initial_rates[:cons-1,1], 'b')
        ax_I_init.plot(contrasts[cons-1], initial_rates[cons-1,1], 'b^')
        ax_I_init.set_title('I Rates Post GD')

    ax_spect_maun = fig_combined.add_subplot(gs[4:8,1:3])
    ax_spect_maun.set_prop_cycle('color', maun_color)
    ax_spect_maun.plot(fs, target_spect[:, -probes:], '--')
    ax_spect_maun.set_prop_cycle('color', maun_color)
    ax_spect_maun.plot(fs, obs_spect[:, -probes:])

    ax_spect_maun.set_title('Maun Effect - Post GD')
    ax_spect_maun.set_xlabel('frequency (Hz)')
    ax_spect_maun.set_ylabel('Power Spectrum (a.u.)')
    
    if initial_spect is not None:
        ax_init_maun = fig_combined.add_subplot(gs[4:8,4:6])
        ax_init_maun.set_prop_cycle('color', maun_color)
        ax_init_maun.plot(fs, target_spect[:, -probes:], ':')
        ax_init_maun.set_prop_cycle('color', maun_color)
        ax_init_maun.plot(fs, initial_spect[:, -probes:])

        ax_init_maun.set_title('Maun Effect - Pre GD')
        ax_init_maun.set_xlabel('frequency (Hz)')
        ax_init_maun.set_ylabel('Power Spectrum (a.u.)')
    
    ax_f0_maun = fig_combined.add_subplot(gs[4:6, 0])
    ax_f0_maun.set_prop_cycle('color', maun_color)
    for pp in range(probes):
        ax_f0_maun.plot(pp, obs_f0[pp], 'o')
    ax_f0_maun.set_title('Post GD')
    ax_f0_maun.set_xlabel('Probes')
    ax_f0_maun.set_ylabel('Peak frequency (Hz)')
    
    
    ax_stim = fig_combined.add_subplot(gs[6:8, 0])
    ax_stim.imshow(stim.T)
    gridsize = stim.shape
    midx = np.floor(gridsize[0]/2)
    midy = np.floor(gridsize[1]/2)
    ax_stim.set_prop_cycle('color', maun_color)
    
    for pp in range(probes):
        neur_inds_x = midx + pp
        neur_inds_y = midy
        ax_stim.scatter(neur_inds_x, neur_inds_y, 100)
    
    if initial_f0 is not None:
        ax_f0_init = fig_combined.add_subplot(gs[4:6, 3])
        ax_f0_init.set_prop_cycle('color', maun_color)
        for pp in range(probes):
            ax_f0_init.plot(pp, initial_f0[pp], 'o')
        ax_f0_init.set_title('Pre GD')
        ax_f0_init.set_xlabel('Probes')
        ax_f0_init.set_ylabel('Peak frequency (Hz)')
        
        
        ax_i_stim = fig_combined.add_subplot(gs[6:8, 3])
        ax_i_stim.imshow(stim.T)
        ax_i_stim.set_prop_cycle('color', maun_color)
        for pp in range(probes):
            neur_inds_x = midx + pp
            neur_inds_y = midy
            ax_i_stim.scatter(neur_inds_x, neur_inds_y, 100)
    
    if fname is not None:
        plt.savefig(fname)
        
def Maun_Con_SS(fs, obs_spect, target_spect, obs_rates, obs_f0, contrasts, radii, params, init_params = None, con_inds=(0,1,2,6), rad_inds=(3,4,5,6), probes=5, gabor_inds = -1, SI = None, dx = None, fname=None, fignumber = 16):
    
    cons = len(contrasts)
    fig_combined = plt.figure(fignumber, figsize=(8,8), constrained_layout=True)
    con_color =  ['black', 'blue', 'green', 'red','gold']
    maun_color = ['gold', 'purple', 'green', 'maroon', 'xkcd:sky blue']
    rates_color = ['red', 'blue']
#     if loss_t is None:
    
    gs = gridspec.GridSpec(5,3, figure=fig_combined)
    
    if fignumber == 16:
        titletag = '- Post GD'
    else:
        titletag = '- Pre GD'
    
    #spect plots - Contrast effect
    ax_spect_con = fig_combined.add_subplot(gs[0:2,0:2])
    ax_spect_con.set_prop_cycle('color', con_color)
    ax_spect_con.plot(fs, target_spect[:, :cons-1], "--")
    ax_spect_con.plot(fs, target_spect[:, cons-1], '--')
    #colrs = ax.get_color_cycle()
    ax_spect_con.set_prop_cycle(None)
    ax_spect_con.set_prop_cycle('color', con_color)
    ax_spect_con.plot(fs, obs_spect[:, :cons])
#     ax_spect_con.plot(fs, obs_spect[:, cons-1], 'o')
    
    contitle = 'Contrast Effect ' + titletag
    ax_spect_con.set_title(contitle)
    ax_spect_con.set_ylabel('Power spectrum (a.u.)')
    ax_spect_con.set_xlabel('Frequency (Hz)')
    
    #spect plots - Maun Effect
    ax_spect_maun = fig_combined.add_subplot(gs[2:4,0:2])
    ax_spect_maun.set_prop_cycle('color', maun_color)
    ax_spect_maun.plot(fs, target_spect[:, -probes:], '--', label='_nolegend_')
    ax_spect_maun.set_prop_cycle('color', maun_color)
    ax_spect_maun.plot(fs, obs_spect[:, -probes:])
    lstr = ['R = 0']
    for pp in range(1, probes):
        lstr.append('R = '+str(pp))
    ax_spect_maun.legend(lstr, loc='upper left', ncol=2)
    
    mauntitle = 'Ray & Maunsell Effect - ' + titletag
    ax_spect_maun.set_title(mauntitle)
    ax_spect_maun.set_xlabel('Frequency (Hz)')
    ax_spect_maun.set_ylabel('Power spectrum (a.u.)')
    
    #rates plots
    ax_EI = fig_combined.add_subplot(gs[0, 2:])
    ax_EI.set_prop_cycle('color', rates_color)
    ax_EI.plot(contrasts[:cons], obs_rates[con_inds, :])
    ax_EI.set_prop_cycle('color', rates_color)
    ax_EI.plot(contrasts[-1], obs_rates[gabor_inds,0], '^')
    ax_EI.plot(contrasts[-1], obs_rates[gabor_inds,1], '^')
    ax_EI.set_xlabel('Contrast')
    ax_EI.set_ylabel('Firing rate (Hz)')
    ax_EI.set_xticks(contrasts)
#     ax_EI.set_title('Firing Rates')
    ax_EI.legend(['E', 'I'])
    
    ax_SS = fig_combined.add_subplot(gs[1, 2:])
    ax_SS.set_prop_cycle('color', rates_color)
    ax_SS.plot(radii, obs_rates[rad_inds, :])
    ax_SS.set_xlabel('Stimulus Radius (degrees)')
    ax_SS.set_xticks(np.hstack((0,radii)))
    ax_SS.set_ylabel('Firing rate (Hz)')
    if SI is not None:
        tstr = 'SI = '+'({:.2f}, {:.2f})'.format(SI[0], SI[1]) #+', SI_I = '+'{:.2f}'.format(SI[1])
    else:
        tstr = 'Suppression Curve'
    ax_SS.set_title(tstr)
    ax_SS.set_ylim(bottom=0)
    
        
#     ax_SS.legend(['E', 'I'])
    
    #peak freq plots
    ax_con_f0 = fig_combined.add_subplot(gs[2, 2:])
    ax_con_f0.set_prop_cycle('color', con_color[1:])
    for cc in range(1, cons):
        ind = con_inds[cc] -1 #removed BS from f0 calculations
        ax_con_f0.plot(contrasts[cc], obs_f0[cc - 1],'o')
    ax_con_f0.set_xlabel('Contrast')
    ax_con_f0.set_ylabel('Peak frequency (Hz)')
    
    
    
    ax_maun_f0 = fig_combined.add_subplot(gs[3, 2:])
    RR = np.arange(probes)
    ax_maun_f0.set_prop_cycle('color', maun_color)
    for pp in range(0, probes):
        ax_maun_f0.plot(RR[pp], obs_f0[-probes+pp],'o')
    ax_maun_f0.set_xlabel('Probe location')
    ax_maun_f0.set_ylabel('Peak frequency (Hz)')
    
    if dx is not None:
        probe_dist = dx * np.arange(probes)
        

        #GaborSigma = 0.3*np.max(radii)
        GaborSigma = 0.5
        Gabor_Cons = 100*np.exp(- probe_dist**2/2/GaborSigma**2);
        #print(Gabor_Cons)

#         fit_f0 = np.interp(Gabor_Cons, contrasts[1:], obs_f0[:3]) #contrast[1:] means I'm not looking at 0 contrast, also why I stop at [:3] cause indices greater than 3 are for the Gabor and Maunsell effect.
#         ax_maun_f0.plot(RR, fit_f0,'gray')
        
        fit_f0 = interp1d(contrasts[1:], obs_f0[:3], fill_value='extrapolate')
        ax_maun_f0.plot(RR, fit_f0(Gabor_Cons),'gray')

    
    
    
    ## Params Bars
    J_max = 3
    i2e_max = 2
    gE_max = np.sqrt(10)
    gI_max = np.sqrt(10)
    NMDA_max = 1
    plocal_max = 1
    sigR_max = 1.5 #because I do not want sigR_min = 0, so I will offset the sigmoid
    sigEE_max = 0.5 # because I do not want sigEE_min = 0 so I will offset the sigmoid
    sigIE_max = 0.5 # because I do not want sigEE_min = 0 so I will offset the sigmoid
    
    if len(params) < 9:
        params_max = np.array([J_max, J_max, J_max, J_max, i2e_max, plocal_max, sigR_max])
        label_params = ['Jee', 'Jei', 'Jie', 'Jii', 'I/E', 'Plocal', 'sigR']
    elif len(params) == 9:
        params_max = np.array([J_max, J_max, J_max, J_max, gE_max, gI_max, NMDA_max, plocal_max, sigR_max])
        label_params = ['Jee', 'Jei', 'Jie', 'Jii', 'gE', 'gI', 'NMDA/AMPA','Plocal', 'sigR']
    elif len(params) == 10:
        params_max = np.array([J_max, J_max, J_max, J_max, gE_max, gI_max, NMDA_max, plocal_max, sigEE_max, sigIE_max])
        label_params = ['Jee', 'Jei', 'Jie', 'Jii', 'gE', 'gI', 'NMDA/AMPA','Plocal', 'sigEE', 'sigIE']
    else:
        params_max = np.array([J_max, J_max, J_max, J_max, gE_max, gI_max, NMDA_max, plocal_max, plocal_max, sigEE_max, sigIE_max])
        label_params = ['Jee', 'Jei', 'Jie', 'Jii', 'gE', 'gI', 'NMDA','Plocal_EE', 'Plocal_IE', 'sigEE', 'sigIE']
    
    #Normalize parameters by their max. 
    params = params/params_max
    if init_params is not None:
        init_params = init_params/params_max
    bar_pos = np.arange(len(params))
    width = 0.35
    
    ax_params = fig_combined.add_subplot(gs[4,:])
    if init_params is None:
        ax_params.bar(bar_pos, params, label='MATLAB Parameters')
    else:
        ax_params.bar(bar_pos-width/2, params, width, label='Found Parameters')
        ax_params.bar(bar_pos+width/2, init_params, width, label='Initial Parameters')
    ax_params.set_ylabel('Normalized Parameters')
    ax_params.set_xticks(bar_pos)
    ax_params.set_xticklabels(label_params)
    ax_params.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', ncol=2, mode="expand", borderaxespad=0.)
    ax_params.set_ylim(top=1)
    
    if fname is not None:
        plt.savefig(fname)
    
    return fig_combined

def peak_hists(df0, dhw, params=None):
    
    #axs[0,ctype].hist(np.log10(rs[:,c,ctype]), nbins, color=colors[c-1], alpha=.5)
    con_color =  ['blue', 'green', 'red']
    
    fig_combined = plt.figure(23, constrained_layout=True, figsize=(8, 8))
    
    gs = gridspec.GridSpec(3,2, figure=fig_combined)
    
    numperm = df0.shape[0]
    
    df0_mean = np.mean(df0, axis=1)
    df0 = np.hstack((df0, df0_mean[:,None]))
    
    dhw_mean = np.mean(dhw, axis=1)
    dhw = np.hstack((dhw, dhw_mean[:,None]))
    
    #conditions for hw and f0 to be nan are the same
    nums_25 = ~np.isnan(df0[:, 0])
    nums_100 = ~np.isnan(df0[:,1])
    nums_mean = ~np.isnan(df0[:,2])
    
    ax_df0 = fig_combined.add_subplot(gs[0,0])
    ax_df0.set_xlabel('Peak frequency shift (Hz)')
    ax_df0.set_ylabel('Counts')
    
    ax_dhw = fig_combined.add_subplot(gs[0, 1])
    ax_dhw.set_xlabel('Peak bump width shift (Hz)')
    ax_dhw.set_ylabel('Counts')
    
    cc = 0
    for c in con_color:
        ax_df0.hist(df0[:,cc], color=c, alpha=0.5)
        ax_dhw.hist(dhw[:,cc], color=c, alpha=0.5)
        cc +=1
    lstr = ['50-25', '100-50', 'Mean shift']
    ax_df0.legend(lstr)
    ax_dhw.legend(lstr)
    
    #corrcoef returns an 8x8 matrix of correlation coefficients, only care about how df0 correlates with all other params
    corr_f025 = np.corrcoef(df0[nums_25, 0], params[:, nums_25])[0, 1:] 
    corr_f0100 = np.corrcoef(df0[nums_100, 1], params[:, nums_100])[0, 1:]
    corr_f0mean= np.corrcoef(df0[nums_mean, 2], params[:, nums_mean])[0, 1:]
    corr_f0 =  np.vstack((corr_f025, corr_f0100, corr_f0mean))
    
    corr_hw25 = np.corrcoef(dhw[nums_25, 0], params[:, nums_25])[0, 1:]
    corr_hw100 = np.corrcoef(dhw[nums_100, 1], params[:, nums_100])[0, 1:]
    corr_hwmean= np.corrcoef(dhw[nums_mean, 2], params[:, nums_mean])[0, 1:]
    corr_hw =  np.vstack((corr_hw25, corr_hw100, corr_hwmean))
    
    print(corr_f0.shape)
    
    width = 0.3
    param_labels = ['Jee', 'Jei', 'Jie', 'Jii', 'gE', 'gI', 'NMDA']
    bar_pos = np.arange(len(param_labels))
    
    ax_corrf0 = fig_combined.add_subplot(gs[1,:])
    ax_corrhw = fig_combined.add_subplot(gs[2,:])
    cc = 0
    for c in con_color:
        ax_corrf0.bar(bar_pos -width + cc*width, corr_f0[cc, :], width=width, color=c)
        ax_corrhw.bar(bar_pos -width + cc*width, corr_hw[cc, :], width=width, color=c)
        cc +=1
        
    ax_corrf0.set_xticks(bar_pos)
    ax_corrf0.set_ylabel('Coefficient')
    ax_corrf0.set_xticklabels(param_labels)
    ax_corrf0.set_title('Correlation coefficient of changes in peak frequency with  parameters')
    
    
    ax_corrhw.set_xticks(bar_pos)
    ax_corrhw.set_ylabel('Coefficient')
    ax_corrhw.set_xticklabels(param_labels)
    ax_corrhw.set_title('Correlation coefficient of changes in bump width with  parameters')
        
    return fig_combined
    
def dyn_plots(t_range, v_dyn, sim_fs, sim_spect, anal_fs, spect, sim_f0, anal_f0):
    
    fs = 18
    ls = 11
    ss = 15
    
    contrasts = np.array([0, 25, 50, 100])
    cons = len(contrasts)
    
    tt = np.arange(t_range[0], t_range[1], t_range[2]) - t_range[0]
    t_inds = (int(t_range[0]*t_range[2]), int(t_range[1]*t_range[2]))
    
    fig_sim = plt.figure(8, figsize=(12,8), constrained_layout=True)
    con_color =  ['black', 'blue', 'green', 'red']
    maun_color = ['gold', 'purple', 'green', 'maroon', 'xkcd:sky blue']
    rates_color = ['xkcd:green', 'xkcd:maroon']
    
    rE_dyn = 0.04*np.max(np.sum(v_dyn[::2, :], axis=0), 0)**2
    rI_dyn = 0.04*np.max(np.sum(v_dyn[1::2, :], axis=0), 0)**2
    rE = np.mean(rE_dyn[0, -2000])
    
    gs = gridspec.GridSpec(3,2, figure=fig_sim)
    
    ax_blank = fig_sim.add_subplot(gs[0,0:1])
    ax_blank.set_xticks([])
    ax_blank.set_yticks([])
    
    vE = np.sum(v_dyn[t_inds[0]:t_inds[1],::2,:], axis=1)
    
    ax_volts = fig_sim.add_subplot(gs[0,1])
    ax_volts.set_prop_cycle('color', con_color)
    ax_volts.plot(tt, vE)
    ax_volts.set_xlabel('Time (ms)',fontsize=fs)
#     ax_volts.set_xticks(np.arange(t_range[1]-t_range[0]))
#     ax_volts.set_xticks(fontsize=ss)
#     ax_volts.set_yticks(fontsize=ss)
    ax_volts.set_ylabel('LFP (a.u.)', fontsize=fs)
    ax_volts.legend(['C = 0', 'C = 25', 'C = 50', 'C = 100'], frameon=False, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=4, fontsize=ls)
    ax_volts.set_ylim(top = 1.2*np.max(vE))
    
    ax_rates = fig_sim.add_subplot(gs[0, 2])
    
    f_inds = np.where(sim_fs > np.min(anal_fs), sim_fs, 0)
    f_inds = np.where(sim_fs< np.max(anal_fs), f_inds, 0)
    f_inds = np.nonzero(f_inds)
    
    new_fs = sim_fs[f_inds[0]]
    new_sim_spect = sim_spect[f_inds[0], :]/np.mean(sim_spect[f_inds[0], :])
    
    ax_comp = fig_sim.add_subplot(gs[1, 1:])
    ax_comp.set_prop_cycle('color', con_color)
    sim = ax_comp.plot(new_fs, new_sim_spect, 'o')
    anal = ax_comp.plot(anal_fs, spect)
    ax_comp.set_xlabel('Frequency (Hz)', fontsize=fs)
    ax_comp.set_ylabel('Power spectrum (a.u.)', fontsize=fs)
    ax_comp.set_ylim(top=1.5*np.max(new_sim_spect))
    
    sim = mlines.Line2D([], [], linestyle='none', color='gray', marker='o', label='Simulated')
    anal = mlines.Line2D([], [], color='gray', label='Linear approximation')
    ax_comp.legend(handles=[sim, anal], fontsize=ls, loc='upper left')
    
    ax_inset = ax_comp.inset_axes([0.7, 0.7, 0.25, 0.25])
    for cc in range(1, cons):
        ax_inset.plot(contrasts[cc], sim_f0[cc-1],'o', color=con_color[cc])
        ax_inset.plot(contrasts[cc], anal_f0[cc],'*', color=con_color[cc])
    ax_inset.set_xlabel('Contrasts')
    ax_inset.set_xticks(contrasts)
    ax_inset.spines['right'].set_visible(False)    
    ax_inset.spines['top'].set_visible(False)
        #ax_inset.set_ylabel('Peak frequency (Hz)')
    
    return fig_sim

def hists_fig2(rs, lams, min_freq=10, dfdc=False):
    
    fs = 18
    ls = 11
    ss = 15
    
    #max(axis=2) is because lams is a 6-dim vector for each contrast and instantiation
    max_real_lams = np.real(lams).max(axis=2) *1000/2/np.pi #convert to Hz ( * 1000/2/np.pi)
    max_imag_lams = np.imag(lams).max(axis=2) *1000/2/np.pi
    inds = np.imag(lams).argmax(axis=2) 
    gamma_HW = -np.real(lams)[np.arange(lams.shape[0])[:,None], np.arange(lams.shape[1])[None,:], inds] *1000/2/np.pi
    min_freq = 10 #Hz
    
    con_color = ['black', 'blue','green','red'] #colors for contrasts 0, 25, 50, 100,
    shift_colors = ['tab:cyan', 'gold'] #blue and green make cyan, red and green make yellow
    cons = len(con_color)
    contrasts = np.array([0, 25, 50, 100])
    histstyle_paper = {"histtype": "step", "linewidth": 2.25, "density": True,}
    histstyle_nb = {"histtype": "bar", "linewidth":2, "alpha": .3, "density": True,}

    histstyle = histstyle_paper
    
    nbins = 30

    ctypes = ["E", "I"]
    #fig, axs = plt.subplots(3,4, figsize=(16,6))
    
    fighists = plt.figure(23, constrained_layout=True, figsize=(12, 6))
    gs = gridspec.GridSpec(2,3, figure=fighists)
    
    axErates = fighists.add_subplot(gs[0,0])
    axIrates = fighists.add_subplot(gs[1,0])
    axf0 = fighists.add_subplot(gs[0,1])
    axf0shifts = fighists.add_subplot(gs[1,1])
    axhw = fighists.add_subplot(gs[0,2])
    axhwshifts = fighists.add_subplot(gs[1,2])
    
    Emed = np.median(rs[:,:,0],axis=0)
    print(Emed)
    Imed = np.median(rs[:,:, 1],axis=0)
    print(Imed)
    
    lbound = 0.7
    rbound = 1500
    
    for c in np.arange(1, cons):
#         axErates.hist(np.log10(rs[:, c, 0]), nbins, color=con_color[c], **histstyle)
#         axErates.hist(np.log10(rs[:, c, 0]), nbins, color=con_color[c], **histstyle_nb)
#         axIrates.hist(np.log10(rs[:, c, 1]), nbins, color=con_color[c], **histstyle)
#         axIrates.hist(np.log10(rs[:, c,1]), nbins, color=con_color[c], **histstyle_nb)
        
        f0 = max_imag_lams[max_imag_lams[:,c]>min_freq,c]
        hw = gamma_HW[max_imag_lams[:,c]>min_freq,c]
        axf0.hist(f0, nbins, color=con_color[c], **histstyle)
        axf0.hist(f0, nbins, color=con_color[c], **histstyle_nb)
        axhw.hist(hw, nbins, color=con_color[c], **histstyle)
        axhw.hist(hw, nbins, color=con_color[c], **histstyle_nb)
        
        _, binsE = np.histogram(rs[:, c, 0], bins=nbins)
        _, binsI = np.histogram(rs[:, c, 1], bins=nbins)
        
        logbinsE = np.logspace(np.log10(binsE[0]),np.log10(binsE[-1]), len(binsE))
        logbinsI = np.logspace(np.log10(binsI[0]), np.log10(binsI[-1]), len(binsI))
        
    
        axErates.hist(rs[:, c, 0], bins=logbinsE, color=con_color[c], **histstyle)
        axErates.hist(rs[:, c, 0], bins=logbinsE, color=con_color[c], **histstyle_nb)
        axIrates.hist(rs[:, c, 1], bins=logbinsI, color=con_color[c], **histstyle)
        axIrates.hist(rs[:, c,1], bins=logbinsI, color=con_color[c], **histstyle_nb)
        
        if c > 1:
            if not dfdc:
                dhw = np.diff(gamma_HW[max_imag_lams[:,c]>min_freq,:],axis=1)[:,c-1]
                df0 = np.diff(max_imag_lams[max_imag_lams[:,c]>min_freq,:],axis=1)[:,c-1]
            else:
                dhw = (np.diff(gamma_HW[max_imag_lams[:,c]>min_freq,:],axis=1)/np.diff(contrasts))[:,c-1]
                df0 = (np.diff(max_imag_lams[max_imag_lams[:,c]>min_freq,:],axis=1)/np.diff(contrasts))[:,c-1]
            axf0shifts.hist(df0, nbins, color= shift_colors[c-2], **histstyle)
            axf0shifts.hist(df0, nbins, color= shift_colors[c-2], **histstyle_nb)
            axhwshifts.hist(dhw, nbins, color= shift_colors[c-2], **histstyle)
            axhwshifts.hist(dhw, nbins, color= shift_colors[c-2], **histstyle_nb)
        
        
    axErates.set_xlim(left=lbound, right=rbound)
    axErates.set_xscale('log')
    #axErates.set_xlabel('$log_{10}$(Firing rates (Hz))', fontsize=fs)
    axErates.set_xlabel('Firing rate (Hz)', fontsize=ss)
    axErates.set_ylabel('Probability', fontsize=ss)
    axErates.set_title('Excitatory cell', fontsize=fs)
    p25 = mpatches.Rectangle([1,1], 1, 1, facecolor=con_color[1], edgecolor=con_color[1], alpha=0.3, lw=0.1, label='C = 25')
    p50 = mpatches.Rectangle([1,1], 1, 1, facecolor=con_color[2], edgecolor=con_color[2], alpha=0.3, lw=0.1, label='C = 50')
    p100 = mpatches.Rectangle([1,1], 1, 1, facecolor=con_color[3], edgecolor=con_color[3], alpha=0.3, lw=0.1, label='C = 100')
    c25 = mlines.Line2D([], [], color=con_color[1], label='C = 25')
    c50 = mlines.Line2D([], [], color=con_color[2], label='C = 50')
    c100 = mlines.Line2D([], [], color=con_color[3], label='C = 100')
    axErates.legend(handles=[p25, p50, p100], frameon=False, ncol=1, fontsize=ls)
    #axErates.legend(['C = 25', 'C = 50', 'C = 100'], frameon=False, loc='upper left', ncol=1, fontsize=ls)
    
    axIrates.set_xlim(left=lbound, right=rbound)
    axIrates.set_xscale('log')
    axIrates.set_xlabel('Firing rate (Hz)', fontsize=ss)
    axIrates.set_ylabel('Probability', fontsize=ss)
    axIrates.set_title('Inhibitory cell', fontsize=fs)
    
    axf0.set_xlabel('Peak frequency (Hz)', fontsize=ss)
    #axf0.set_ylabel('Probability', fontsize=fs)
    axf0.set_title('Gamma peak frequency', fontsize=fs)
    
    axhw.set_xlabel('Peak width (Hz)', fontsize=ss)
    #axhw.set_ylabel('Probability', fontsize=fs)
    axhw.set_title('Gamma peak width', fontsize=fs)
    
    axf0shifts.set_xlabel('$\Delta$Peak frequency (Hz)', fontsize=ss)
    #axf0shifts.set_ylabel('Probability', fontsize=fs)
    cdelta50 = mlines.Line2D([], [], color=shift_colors[0], label='$\Delta C =$50-25')
    cdelta100 = mlines.Line2D([], [], color=shift_colors[1], label='$\Delta C =$100-50')
    pdelta50 = mpatches.Rectangle([1,1], 1, 1, facecolor=shift_colors[0], edgecolor=shift_colors[0], alpha=0.3, lw=0.1, label='$\Delta C =$50-25')
    pdelta100 = mpatches.Rectangle([1,1], 1, 1, facecolor=shift_colors[1], edgecolor=shift_colors[1], alpha=0.3, lw=0.1, label='$\Delta C =$100-50')
    axf0shifts.legend(handles=[pdelta50, pdelta100], fontsize=ls, frameon=False)
    #axf0shifts.legend(['$\Delta C =$50-25', '$\Delta C =$100-50'], fontsize=ls, frameon=False)
    axf0shifts.set_title('Shift in gamma peak frequency', fontsize=fs)
    
    axhwshifts.set_xlabel('$\Delta$Peak width (Hz)', fontsize=ss)
    #axhwshifts.set_ylabel('Probability', fontsize=fs)
    axhwshifts.set_title('Shift in gamma peak width', fontsize=fs)
    #axhwshifts.legend(['$\Delta C =$50-25', '$\Delta C =$100-50'], fontsize=ls, frameon=False)
    
    return fighists