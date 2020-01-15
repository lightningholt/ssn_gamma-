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
        
def Maun_Con_SS(fs, obs_spect, target_spect, obs_rates, obs_f0, contrasts, radii, params, init_params, con_inds=(0,1,2,6), rad_inds=(3,4,5,6), probes=5, gabor_inds = -1, fname=None):
    
    cons = len(contrasts)
    fig_combined = plt.figure(16, constrained_layout=True)
    con_color =  ['black', 'blue', 'green', 'red','gold']
    maun_color = ['gold', 'purple', 'green', 'maroon', 'xkcd:sky blue']
    rates_color = ['red', 'blue']
#     if loss_t is None:
    
    gs = gridspec.GridSpec(3,4, figure=fig_combined)
    
    #spect plots - Contrast effect
    ax_spect_con = fig_combined.add_subplot(gs[0,0:2])
    ax_spect_con.set_prop_cycle('color', con_color)
    ax_spect_con.plot(fs, target_spect[:, :cons-1], "--")
    ax_spect_con.plot(fs, target_spect[:, cons-1], '--')
    #colrs = ax.get_color_cycle()
    ax_spect_con.set_prop_cycle(None)
    ax_spect_con.set_prop_cycle('color', con_color)
    ax_spect_con.plot(fs, obs_spect[:, :cons])
#     ax_spect_con.plot(fs, obs_spect[:, cons-1], 'o')
    
    ax_spect_con.set_title('Contrast Effect - Post GD')
    ax_spect_con.set_ylabel('Power Spectrum (a.u.)')
    ax_spect_con.set_xlabel('frequency (Hz)')
    
    #spect plots - Maun Effect
    ax_spect_maun = fig_combined.add_subplot(gs[1,0:2])
    ax_spect_maun.set_prop_cycle('color', maun_color)
    ax_spect_maun.plot(fs, target_spect[:, -probes:], '--', label='_nolegend_')
    ax_spect_maun.set_prop_cycle('color', maun_color)
    ax_spect_maun.plot(fs, obs_spect[:, -probes:])
    lstr = ['R = 0']
    for pp in range(1, probes):
        lstr.append('R = '+str(pp))
    ax_spect_maun.legend(lstr, loc='upper left', ncol=2)

    ax_spect_maun.set_title('Maun Effect - Post GD')
    ax_spect_maun.set_xlabel('frequency (Hz)')
    ax_spect_maun.set_ylabel('Power Spectrum (a.u.)')
    
    #rates plots
    ax_EI = fig_combined.add_subplot(gs[0, 2])
    ax_EI.set_prop_cycle('color', rates_color)
    ax_EI.plot(contrasts[:cons], obs_rates[con_inds, :])
    ax_EI.set_prop_cycle('color', rates_color)
    ax_EI.plot(contrasts[-1], obs_rates[gabor_inds,0], '^')
    ax_EI.plot(contrasts[-1], obs_rates[gabor_inds,1], '^')
    ax_EI.set_xlabel('Contrasts')
    ax_EI.set_ylabel('Rates (Hz)')
    ax_EI.set_title('Firing Rates')
    ax_EI.legend(['E', 'I'])
    
    ax_SS = fig_combined.add_subplot(gs[1, 2])
    ax_SS.plot(radii, obs_rates[rad_inds, 0])
    ax_SS.set_xlabel('Stim Radii')
    ax_SS.set_title('Suppression Curve')
    
    #peak freq plots
    ax_con_f0 = fig_combined.add_subplot(gs[0, -1])
    ax_con_f0.set_prop_cycle('color', con_color[1:])
    for cc in range(1, cons-1):
        ind = con_inds[cc] -1 #removed BS from f0 calculations
        ax_con_f0.plot(contrasts[cc], obs_f0[cc - 1],'o')
    ax_con_f0.set_xlabel('Contrasts')
    ax_con_f0.set_ylabel('Peak Frequency')
    
    ax_maun_f0 = fig_combined.add_subplot(gs[1, -1])
    RR = np.arange(probes)
    ax_maun_f0.set_prop_cycle('color', maun_color)
    for pp in range(0, probes):
        ax_maun_f0.plot(RR[pp], obs_f0[-probes+pp],'o')
    ax_maun_f0.set_xlabel('Probe Location')
    ax_maun_f0.set_ylabel('Peak Frequency')
    
    
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
    else:
        params_max = np.array([J_max, J_max, J_max, J_max, gE_max, gI_max, NMDA_max, plocal_max, sigEE_max, sigIE_max])
        label_params = ['Jee', 'Jei', 'Jie', 'Jii', 'gE', 'gI', 'NMDA/AMPA','Plocal', 'sigEE', 'sigIE']    
    
    #Normalize parameters by their max. 
    params = params/params_max
    init_params = init_params/params_max
    bar_pos = np.arange(len(params))
    width = 0.35
    
    ax_params = fig_combined.add_subplot(gs[2,:])
    ax_params.bar(bar_pos-width/2, params, width, label='Found Parameters')
    ax_params.bar(bar_pos+width/2, init_params, width, label='Initial Parameters')
    ax_params.set_ylabel('Normalized Parameters')
    ax_params.set_xticks(bar_pos)
    ax_params.set_xticklabels(label_params)
    ax_params.legend()
    
    if fname is not None:
        plt.savefig(fname)
    
    