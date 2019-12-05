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
        ax_init_maun.plot(fs, target_spect[:, -probes:])

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