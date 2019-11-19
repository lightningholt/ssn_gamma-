def make_loss_with_state_update():
    
    r_init = np.zeros([ssn.N, len(contrasts)]) #initial states in Taka's code
    prev_solutions = np.zeros([ssn.N, len(contrasts)])
    
    
    def loss(params):
        contrasts = np.array([0, 25, 50, 100])

        #unpack parameters
        params = sigmoid_jee(pos_params)

        psi = 0.774
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

        lower_bound_rates = -5 * np.ones([2, cons-1])
        upper_bound_rates = np.vstack((70*np.ones(cons-1), 100*np.ones(cons-1)))
        kink_control = 1 # how quickly log(1 + exp(x)) goes to ~x, where x = target_rates - found_rates    
        prefact_rates = 1
        prefact_params = 10
        fs_loss_inds = np.arange(0 , len(fs))
        fs_loss_inds = np.array([freq for freq in fs_loss_inds if fs[freq] >20])#np.where(fs > 0, fs_loss_inds, )

        cons = len(contrasts)

        ssn = SSN_classes.SSN_2D_AMPAGABA(tau_s, NMDAratio, n,k,tauE,tauI, Jee, Jei, Jie, Jii)

        inp_vec = np.array([[gE], [gI*i2e]]) * contrasts

        r_fp = ssn.fixed_point_r(inp_vec, r_init=r_init, Tmax=Tmax, dt=dt, xtol=xtol)
        @jax.custom_gradient
        def no_grad(f):
            return f(), lambda g: ()
        
        @no_grad
        def _():
            prev_solutions = r_fp #store the last
            
        spect, fs, f0, _ = SSN_power_spec.linear_PS_sameTime(ssn, r_fp, SSN_power_spec.NoisePars(), freq_range, fnums, cons)
        
        spect_loss = losses.loss_spect_contrasts(fs[fs_loss_inds], np.real(spect[fs_loss_inds, :]))
        rates_loss = prefact_rates * losses.loss_rates_contrasts(r_fp, lower_bound_rates, upper_bound_rates, kink_control) #fourth arg is slope which is set to 1 normally
        param_loss = prefact_params * losses.loss_params(params)
        
        if spect_loss/rates_loss < 1:
            print('rates loss is greater than spect loss')
        
        return spect_loss + param_loss + rates_loss
    
    return loss, r_init, prev_solutions

def make_callback(r_init, prev_solutions):
    def callback(_):
        r_init = prev_solutions

loss, r_init, prev_solutions = make_loss_with_state_update()
callback = make_callback(r_init, prev_solutions)

grad_loss = grad(loss)    