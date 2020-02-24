import numpy as np
#import jax.numpy as np

from util import Euler2fixedpt


# ============================  base classes ===================================

class _SSN_Base(object):
    def __init__(self, n, k, Ne, Ni, tau_vec=None, W=None):
        self.n = n
        self.k = k
        self.Ne = Ne
        self.Ni = Ni
        self.N = self.Ne + self.Ni
        if tau_vec is not None:
            self.tau_vec = tau_vec # rate time-consants of neurons. shape: (N,)
        else:
            self.tau_vec = np.random.rand(N) * 20 # in ms
        if W is not None:
            self.W = W # connectivity matrix. shape: (N, N)
        else:
            W = np.random.rand(N,N) / np.sqrt(self.N)
            sign_vec = np.hstack(np.ones(self.Ne), -np.ones(self.Ni))
            self.W = W * sign_vec[None, :] # to respect Dale

    @property
    def neuron_params(self):
        return dict(n=self.n, k=self.k)

    @property
    def dim(self):
        return self.N

    @property
    def tau_x_vec(self):
        """ time constants for the generalized state-vector, x """
        return self.tau_vec


    def powlaw(self, u):
        return  self.k * np.maximum(0,u)**self.n

    def drdt(self, r, inp_vec):
        return ( -r + self.powlaw(self.W @ r + inp_vec) ) / self.tau_vec

    def dxdt(self, x, inp_vec):
        """
        allowing for descendent SSN types whose state-vector, x, is different
        than the rate-vector, r.
        """
        return self.drdt(x, inp_vec)

    def gains_from_v(self, v):
        return self.n * self.k * np.maximum(0,v)**(self.n-1)

    def gains_from_r(self, r):
        return self.n * self.k**(1/self.n) * r**(1-1/self.n)

    def DCjacobian(self, r):
        """
        DC Jacobian (i.e. zero-frequency linear response) for
        linearization around rate vector r
        """
        Phi = self.gains_from_r(r)
        return -np.eye(ssn.N) + Phi[:, None] * self.W

    def jacobian(self, DCjacob, r=None):
        """
        dynamic Jacobian for linearization around rate vector r
        """
        if DCjacob is None:
            assert r is not None
            DCjacob = self.DCjacobian(r)
        return DCjacob / self.tau_x_vec[:, None] # equivalent to np.diag(tau_x_vec) * DCjacob

    def inv_G(self, omega, DCjacob, r=None):
        """
        inverse Green's function at angular frequency omega,
        for linearization around rate vector r
        """
        if DCjacob is None:
            assert r is not None
            DCjacob = self.DCjacobian(r)
        return -1j*omega * np.diag(self.tau_x_vec) - DCjacob

    def fixed_point_r(self, inp_vec, r_init=None, Tmax=500, dt=1, xtol=1e-5, PLOT=False):
        if r_init is None:
            r_init = np.zeros((self.N,))
        drdt = lambda r : self.drdt(r, inp_vec)
        r_fp, CONVG = Euler2fixedpt(drdt, r_init, Tmax, dt, xtol=xtol, PLOT=PLOT)
        if not CONVG:
            print('Did not reach fixed point.')
        #else:
        #    return r_fp
        return r_fp, CONVG

    def fixed_point(self, inp_vec, x_init=None, Tmax=500, dt=1, xtol=1e-5, PLOT=False):
        if x_init is None:
            x_init = np.zeros((self.dim,))
        dxdt = lambda x : self.dxdt(x, inp_vec)
        x_fp, CONVG = Euler2fixedpt(dxdt, x_init, Tmax, dt, xtol=xtol, PLOT=PLOT)
        if not CONVG:
            print('Did not reach fixed point.')
        #else:
        #    return x_fp
        return x_fp, CONVG


class _SSN_AMPAGABA_Base(_SSN_Base):
    """
    SSN with different synaptic receptor types.
    Dynamics of the model assumes the instantaneous neural I/O approximation
    suggested by Fourcaud and Brunel (2002).
    Convention for indexing of state-vector v (which is 2N or 3N dim)
    is according to kron(receptor_type_index, neural_index).
    """
    def __init__(self,*, tau_s=[4,5,100], NMDAratio=0.4, **kwargs):
        """
        tau_s = [tau_AMPA, tau_GABA, tau_NMDA] or [tau_AMPA, tau_GABA]
          decay time-consants for synaptic currents of different receptor types.
        NMDAratio: scalar
          ratio of E synaptic weights that are NMDA-type
          (model assumes this fraction is constant in all weights)
        Good values:
         tau_AMPA = 4, tau_GABA= 5  #in ms
         NMDAratio = 0.3-0.4
        """
        self.tau_s = np.squeeze(np.asarray(tau_s))
        self.tau_AMPA = tau_s[0]
        self.tau_GABA = tau_s[1]
        assert self.tau_s.size <= 3 and self.tau_s.ndim == 1
        if  self.tau_s.size == 3 and NMDAratio > 0:
            self.tau_NMDA = tau_s[2]
            self.NMDAratio = NMDAratio
        else:
            self.tau_s = self.tau_s[:2]
            self.NMDAratio = 0
        self.num_rcpt = self.tau_s.size

        super(_SSN_AMPAGABA_Base, self).__init__(**kwargs)

    @property
    def dim(self):
        return self.num_rcpt * self.N

    @property
    def Wrcpt(self):
        if not hasattr(self, '_Wrcpt'): # cache it in _Wrcpt once it's been created
            W_AMPA = (1-self.NMDAratio)* np.hstack((self.W[:,:self.Ne], np.zeros((self.N,self.Ni)) ))
            W_GABA = np.hstack((np.zeros((self.N,self.Ne)), self.W[:,self.Ne:]))
            Wrcpt = [W_AMPA, W_GABA]
            if self.NMDAratio > 0:
                W_NMDA = self.NMDAratio/(1-self.NMDAratio) * W_AMPA
                Wrcpt.append(W_NMDA)
            self._Wrcpt = np.vstack(Wrcpt) # shape = (self.num_rcpt*self.N, self.N)
        return self._Wrcpt

    @property
    def tau_s_vec(self):
        if not hasattr(self, '_tau_s_vec'): # cache it once it's been created
            self._tau_s_vec = np.kron(self.tau_s, np.ones(self.N))
        return self._tau_s_vec

    @property
    def tau_x_vec(self):
        """ time constants for the generalized state-vector, x """
        return self.tau_s_vec


    def dvdt(self, v, inp_vec):
        """
        Returns the AMPA/GABA/NMDA based dynamics, with the instantaneous
        neural I/O approximation suggested by Fourcaud and Brunel (2002).
        v and inp_vec are now of shape (self.num_rcpt * ssn.N,).
        """
        #total input to power law I/O is the sum of currents of different types:
        r = self.powlaw( v.reshape((self.num_rcpt, self.N)).sum(axis=0) )
        return ( -v + self.Wrcpt @ r + inp_vec ) / self.tau_s_vec

    def dxdt(self, x, inp_vec):
        return self.dvdt(x, inp_vec)

    def DCjacobian(self, r):
        """
        DC Jacobian (i.e. zero-frequency linear response) for
        linearization around state-vector v, leading to rate-vector r
        """
        Phi = self.gains_from_r(r)
        return ( -np.eye(self.num_rcpt * self.N) +
    			np.tile( self.Wrcpt * Phi[None,:] , (1, self.num_rcpt)) ) # broadcasting so that gain (Phi) varies by 2nd (presynaptic) neural index, and does not depend on receptor type or post-synaptic (1st) neural index

# ================ N neuron uniform all-2-all models ===========================

class SSNUniform(_SSN_Base):
    def __init__(self, n, k, tauE, tauI, Jee, Jei, Jie, Jii,
                                                Ne, Ni=None, **kwargs):
        Ni = Ni if Ni is not None else Ne
        tau_vec = np.hstack([tauE * np.ones(Ne), tauI * np.ones(Ni)])
        # W = np.block([[Jee/Ne * np.ones((Ne,Ne)), -Jei/Ni * np.ones((Ne,Ni))],
        #               [Jie/Ne * np.ones((Ni,Ne)), -Jii/Ni * np.ones((Ni,Ni))],])
        # since np.block not yet implemented in jax.numpy:
        W = np.vstack(
            [np.hstack([Jee/Ne * np.ones((Ne,Ne)), -Jei/Ni * np.ones((Ne,Ni))]),
             np.hstack([Jie/Ne * np.ones((Ni,Ne)), -Jii/Ni * np.ones((Ni,Ni))])])

        super(SSNUniform, self).__init__(n=n, k=k, Ne=Ne, Ni=Ni,
                                    tau_vec=tau_vec, W=W, **kwargs)

    @property
    def neuron_params(self):
        return dict(n=self.n, k=self.k,
                    tauE=self.tau_vec[0], tauI=self.tau_vec[self.Ne])

    @property
    def topos_vec(self):
        return np.zeros(self.N)

class SSNUniform_AMPAGABA(SSNUniform, _SSN_AMPAGABA_Base):
    pass

# ==========================  2 neuron models ==================================

class SSN_2D(SSNUniform):
    def __init__(self, n, k, tauE, tauI, Jee, Jei, Jie, Jii, **kwargs):
        super(SSN_2D, self).__init__(n, k, tauE, tauI, Jee, Jei, Jie, Jii,
                                        Ne=1, Ni=1, **kwargs)

class SSN_2D_AMPAGABA(SSN_2D, _SSN_AMPAGABA_Base):
    pass

# =============================== ring models ==================================

class SSNHomogRing(_SSN_Base):
    def __init__(self, n, k, tauE, tauI, J_2x2, s_conn,
                                            Ne, L=180, **kwargs): #, Ni=None,

        #Ni = Ni if Ni is not None else Ne
        Ni = Ne
        Ns = [Ne, Ni]
        tau_vec = np.hstack([tauE * np.ones(Ne), tauI * np.ones(Ni)])
        distsq = lambda x: np.minimum(np.abs(x), L-np.abs(x))**2
        from util import toeplitz
        # s_2x2 = np.array(s_conn) if not np.isscalar(s_conn) else s_conn * np.ones((2,2))
        # assert s_2x2.shape == (2,2)
        # blk = lambda i, j: toeplitz(np.exp(-distsq(topos_vec[i,j])/2/s_2x2[i,j]**2))
        topos_vec = np.linspace(0, L, Ne+1)[:-1]
        blk = toeplitz(np.exp(-distsq(topos_vec)/2/s_conn**2))
        W = np.vstack([
               # np.hstack([J_2x2[i,j]/Ns[i] * blk(i,j) for j in range(2)])
               np.hstack([J_2x2[i,j]/Ns[i] * blk for j in range(2)])
                                                    for i in range(2)])

        super(SSNHomogRing, self).__init__(n=n, k=k, Ne=Ne, Ni=Ni,
                                    tau_vec=tau_vec, W=W, **kwargs)

    @property
    def neuron_params(self):
        return dict(n=self.n, k=self.k,
                    tauE=self.tau_vec[0], tauI=self.tau_vec[self.Ne])

    @property
    def topos_vec(self):
        return np.zeros(self.N)

class SSNHomogRing_AMPAGABA(SSNHomogRing, _SSN_AMPAGABA_Base):
    pass


# ===================== non-period 1D topographic models =======================




# =========================== 2D topographic models ============================

class SSN2DTopoV1(_SSN_Base):
    def __init__(self, n, k, tauE, tauI, J_2x2, s_2x2, w_2x2, lam_2x2,
                                            Ne, L=180, **kwargs): #, Ni=None,

        #Ni = Ni if Ni is not None else Ne
        Ni = Ne
        Ns = [Ne, Ni]
        tau_vec = np.hstack([tauE * np.ones(Ne), tauI * np.ones(Ni)])
        distsq = lambda x: np.minimum(np.abs(x), L-np.abs(x))**2
        from util import toeplitz
        # s_2x2 = np.array(s_2x2) if not np.isscalar(s_2x2) else s_2x2 * np.ones((2,2))
        # assert s_2x2.shape == (2,2)
        # blk = lambda i, j: toeplitz(np.exp(-distsq(topos_vec[i,j])/2/s_2x2[i,j]**2))
        topos_vec = np.linspace(0, L, Ne+1)[:-1]
        blk = toeplitz(np.exp(-distsq(topos_vec)/2/s_2x2**2))
        W = np.vstack([
               # np.hstack([J_2x2[i,j]/Ns[i] * blk(i,j) for j in range(2)])
               np.hstack([J_2x2[i,j]/Ns[i] * blk for j in range(2)])
                                                    for i in range(2)])

        super(SSN2DTopoV1, self).__init__(n=n, k=k, Ne=Ne, Ni=Ni,
                                    tau_vec=tau_vec, W=W, **kwargs)

    @property
    def neuron_params(self):
        return dict(n=self.n, k=self.k,
                    tauE=self.tau_vec[0], tauI=self.tau_vec[self.Ne])

    @property
    def topos_vec(self):
        return np.zeros(self.N)

class SSN2DTopoV1_AMPAGABA(SSN2DTopoV1, _SSN_AMPAGABA_Base):
    pass
