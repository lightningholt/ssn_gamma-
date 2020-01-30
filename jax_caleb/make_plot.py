import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

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
        

        GaborSigma = 0.3*np.max(radii)
        Gabor_Cons = 100*np.exp(- probe_dist**2/2/GaborSigma**2);
        print(Gabor_Cons)

        fit_f0 = np.interp(Gabor_Cons, contrasts[1:], obs_f0[:3]) #contrast[1:] means I'm not looking at 0 contrast, also why I stop at [:3] cause indices greater than 3 are for the Gabor and Maunsell effect.
        
        ax_maun_f0.plot(RR, fit_f0,'gray')
    
    
    
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