#import numpy as np
import jax.numpy as np

from util import Euler2fixedpt


class _SSN_Base(object):
    def __init__(self, n, k, Ne, Ni, tau_vec, W): #, tau_vec=None, W=None):
        self.n = n
        self.k = k
        self.Ne = Ne
        self.Ni = Ni
        self.N = self.Ne + self.Ni
        self.tau_vec = tau_vec # rate time-consants of neurons. shape: (N,)
        self.W = W # connectivity matrix. shape: (N, N)

    @property
    def tau_x_vec(self):
        """ time constants for the generalized state-vector, x """
        return self.tau_vec

    @property
    def dim(self):
        return self.N

    @property
    def neuron_params(self):
        return dict(n=self.n, k=self.k)

    def powlaw(self, u):
        return  self.k * np.maximum(0,u)**self.n

    def drdt(self, r, inp_vec):
        return ( -r + self.powlaw(self.W @ r + inp_vec) ) / self.tau_vec[:, None]

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
        return r_fp

    def fixed_point(self, inp_vec, x_init=None, Tmax=500, dt=1, xtol=1e-5, PLOT=False):
        if x_init is None:
            x_init = np.zeros((self.dim,))
        dxdt = lambda x : self.dxdt(x, inp_vec)
        x_fp, CONVG = Euler2fixedpt(dxdt, x_init, Tmax, dt, xtol=xtol, PLOT=PLOT)
        if not CONVG:
            print('Did not reach fixed point.')
        #else:
        #    return x_fp
        return x_fp


class _SSN_AMPAGABA(_SSN_Base):
    """
    SSN with different synaptic receptor types.
    Dynamics of the model assumes the instantaneous neural I/O approximation
    suggested by Fourcaud and Brunel (2002).
    Convention for indexing of state-vector v (which is 2N or 3N dim)
    is according to kron(receptor_type_index, neural_index).
    """
    def __init__(self, tau_s=[4,5,100], NMDAratio=0.4):
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
        self.tau_s_vec = np.kron(self.tau_s, np.ones(self.N))

        W_AMPA = (1-self.NMDAratio)* np.hstack((self.W[:,:self.Ne], np.zeros((self.N,self.Ni)) ))
        W_GABA = np.hstack((np.zeros((self.N,self.Ne)), self.W[:,self.Ne:]))
        Wrcpt = [W_AMPA, W_GABA]
        if self.NMDAratio > 0:
            W_NMDA = self.NMDAratio/(1-self.NMDAratio) * W_AMPA
            Wrcpt.append(W_NMDA)
        self.Wrcpt = np.vstack(Wrcpt) # shape = (self.num_rcpt*self.N, self.N)

    @property
    def tau_x_vec(self):
        """ time constants for the generalized state-vector, x """
        return self.tau_s_vec

    @property
    def dim(self):
        return self.num_rcpt * self.N

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
        if len(r.shape) < 2:
            Phi = self.gains_from_r(r)
            return ( -np.eye(self.num_rcpt * self.N) +
    			np.tile( self.Wrcpt * Phi[None,:] , (1, self.num_rcpt)) ) # broadcasting so that gain (Phi) varies by 2nd (presynaptic) neural index, and does not depend on receptor type or post-synaptic (1st) neural index
        else:
            Phi = lambda rr: np.diag(self.gains_from_r(rr))
            return ( np.array([np.kron(np.ones((1, self.num_rcpt)), np.dot(self.Wrcpt,  Phi(r[:,cc]))) - np.eye(self.N*self.num_rcpt) for cc in range(r.shape[1])]) )


class SSN_2D(_SSN_Base):
    def __init__(self, n, k, tauE, tauI, Jee, Jei, Jie, Jii):
        Ne = 1
        Ni = 1
        tau_vec = np.asarray([tauE, tauI])
        W = np.asarray([[Jee, -Jei], [Jie, -Jii]])
        #super(SSN_2D, self).__init__(n, k, Ne, Ni, tau_vec, W)
        super(SSN_2D, SSN_2D).__init__(self, n, k, Ne, Ni, tau_vec, W)
        self.topos_vec = np.array([0])

    @property
    def neuron_params(self):
        return dict(n=self.n, k=self.k, tauE=self.tau_vec[0], tauI=self.tau_vec[1])

class SSN_2D_AMPAGABA(SSN_2D, _SSN_AMPAGABA):
    def __init__(self, tau_s, NMDAratio, *args):
        super(SSN_2D_AMPAGABA, self).__init__(*args)
        super(SSN_2D, self).__init__(tau_s, NMDAratio)
