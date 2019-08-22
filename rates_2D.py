import jax.numpy as np

def rect_powerLaw(vv, kk, nn, N, cons):
    '''This gives the E/I rates given the synaptic inputs vv, and the nonlinear parameters kk, nn'''
    fv = kk*np.maximum(np.array([np.sum(vv[::2,:], axis=0), np.sum(vv[1::2,:],axis=0)]), np.zeros([N, cons]))**nn
    return fv

def dvdt(vv, kk, nn, N, rcpt_types, cons, Wtot, I_total, tauSvec, dt):
    '''Evaluates the RHS of the v-eqn'''
    delta_v = np.reshape(dt/tauSvec, [N*rcpt_types,1]) * (-vv + Wtot @ np.kron(np.ones([rcpt_types,1]), rect_powerLaw(vv, kk ,nn, N, cons)) + I_total)
    return delta_v


def loss(Jee, Jei, Jie, Jii, i2e, idealspect):
    N = 2
    rcpt_types = 3
    t = np.arange(0,5000.1, 0.1)
    fs = np.arange(0, 101, 1)
    fs = fs/1000 #convert from Hz to kHz
    c = np.array([0, 25, 50, 100])
    cons = len(c)
    
    J0 = np.array([[Jee, -Jei], [Jie, -Jii]])
    
    W = J0
#     print('Det(W) =', '%.3f' % np.linalg.det(W))

    #define nonlinearity parameters
    kk = 0.04
    nn = 2

    if rcpt_types > 1:
        g = np.array([1, i2e, 0, 0, 0, 0])
    else:
        g = np.array([1, i2e])

    tauE = 15
    tau_ratio = 1
    tauI = tauE/tau_ratio

    # tau = np.ones(N)
    # tau[:2:] = tauE
    # tau[1:2:] = tauI

    t_scale = 1
    tauNMDA = 100 * t_scale
    tauAMPA = 3 * t_scale
    tauGABA = 5 * t_scale
    nmdaRatio = 0.1 # sets the ratio of NMDA cells to AMPA cell 

    NoiseNMDAratio = 0
    NoiseTau = 1 * t_scale


    totalT = t[-1]
    dt = np.mean(np.diff(t))
    dt2 = np.sqrt(dt)
    
    if rcpt_types > 1:
        tauS = np.array([tauAMPA, tauNMDA, tauGABA])
        tauSvec = np.kron(tauS, np.ones(N))

        Wtot = np.array([[(1-nmdaRatio)*Jee, 0, 0, 0, 0, 0], [(1-nmdaRatio)* Jie, 0, 0, 0, 0, 0], [0, 0, nmdaRatio * Jee, 0, 0, 0], [0, 0, nmdaRatio * Jie, 0, 0, 0], [0, 0, 0, 0, 0, -Jei], [0, 0, 0, 0, 0, -Jii]])

    else:
        tauSvec = tau
        Wrcpt = W
        Wtot = W
    
    
    v1 = np.zeros([N*rcpt_types, cons])
    r_starcons = np.zeros([N, cons])
    
    I_total = np.kron( g.reshape(N*rcpt_types,1),  c.reshape(1,cons))
    #Conv = True
    indt = 0

    for tt in t:

        dv = dvdt(v1, kk, nn,  N, rcpt_types, cons, Wtot, I_total, tauSvec, dt)
        v1 = dv + v1
    #     vv_t[:,:, tt] = v1
        indt += 1

#         if tt >= totalT - 1000*dt:
#             itr = np.max(np.abs(dv))

#             if itr > 0.01:
#                 Conv = False
                
    r_starcons = rect_powerLaw(v1, kk, nn, N, cons)
    rs = nn*kk**(1/nn)*r_starcons**(1-1/nn)
    
    Phi = lambda rr: np.diag(rr)
    eE = np.array([[1], [0]])
    eE = np.kron(np.ones([rcpt_types,1]), eE)
    J = np.array([[Wtot @ np.kron(np.ones([rcpt_types, rcpt_types]), Phi(rs[:,cc])) -np.eye(N*rcpt_types)] for cc in range(cons)])
    Gf = np.array([-1j * 2 * np.pi * ff * np.diag(np.kron(tauS, np.ones(N))) - J[cc,1] for cc in range(cons) for ff in fs])

    cuE = np.array([eE for cc in range(cons) for ff in fs])
    fscons = np.kron(np.ones([1, cons]), fs)
    
    # iGf = np.linalg.inv(Gf)

    # x = np.einsum("ijk, ikm-> ijm", iGf, cuE)
#     x = np.matmul(iGf, cuE)
    x = np.linalg.solve(Gf, cuE)

    y = (1-NoiseNMDAratio) * x[:, :N] + NoiseNMDAratio * x[:, N:(N+N)]
    y_conj = np.transpose(np.conj(y), [0, 2, 1])

    # spect = np.einsum('ijk, imk -> ijm', y_conj, y)
    tapercons = 2 * NoiseTau/np.abs(-1j * 2 * np.pi * fscons * NoiseTau + 1)**2

    spect = np.squeeze(np.matmul(y_conj, y)) * np.squeeze(tapercons)

    spect = np.reshape(spect*2/1000, [len(fs), cons], order='F')
    spect = spect/np.mean(spect)
    
    ll = np.mean(np.abs(idealspect - spect)**2)
    
    return ll
