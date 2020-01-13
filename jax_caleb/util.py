#import numpy as np
import jax.numpy as np
import numpy as onp

def Euler2fixedpt(dxdt, x_initial, Tmax, dt, xtol=1e-5, xmin=1e-0, PLOT=False, inds=None):
    """
    Finds the fixed point of the D-dim ODE set dx/dt = dxdt(x), using the
    Euler update with sufficiently large dt (to gain in computational time).
    Checks for convergence to stop the updates early.

    IN:
    dxdt = a function handle giving the right hand side function of dynamical system
    x_initial = initial condition for state variables (a column vector)
    Tmax = maximum time to which it would run the Euler (same units as dt, e.g. ms)
    dt = time step of Euler
    xtol = tolerance in relative change in x for determining convergence
    xmin = for x(i)<xmin, it checks convergenece based on absolute change, which must be smaller than xtol*xmin
        Note that one can effectively make the convergence-check purely based on absolute,
        as opposed to relative, change in x, by setting xmin to some very large
        value and inputting a value for 'xtol' equal to xtol_desired/xmin.
    PLOT: if True, plot the convergence of some component
    inds: indices of x (state-vector) to plot

    OUT:
    xvec = found fixed point solution
    CONVG = True if determined converged, False if not
    """

#     if PLOT:
#         if inds is None:
#             N = x_initial.size
#             inds = [int(N/4), int(3*N/4)]
#         xplot = x_initial[inds][:,None]

    Nmax = int(np.round(Tmax/dt))
    xvec = x_initial
    CONVG = False
    for n in range(Nmax):
        dx = dxdt(xvec) * dt
        xvec = xvec + dx
#         if PLOT:
#             #xplot = np.asarray([xplot, xvvec[inds]])
#             xplot = np.hstack((xplot,xvec[inds][:,None]))
        
        if n > 200:
            if np.abs( dx /np.maximum(xmin, np.abs(xvec)) ).max() < xtol:
                #print("      converged to fixed point at iter={},      as max(abs(dx./max(xvec,{}))) < {} ".format(n, xmin, xtol))
                CONVG = True
                break

    if not CONVG: # n == Nmax:
        print("\n Warning 1: reached Tmax={}, before convergence to fixed point.".format(Tmax))
        print("       max(abs(dx./max(abs(xvec), {}))) = {},   xtol={}.\n".format(xmin, np.abs( dx /np.maximum(xmin, np.abs(xvec)) ).max(), xtol))
        #mybeep(.2,350)
        #beep

#     if PLOT:
#         import matplotlib.pyplot as plt
#         plt.figure(244459)
#         plt.plot(np.arange(n+2)*dt, xplot.T, 'o-')

    return xvec, CONVG



# this is copied from scipy.linalg, to make compatible with jax.numpy
def toeplitz(c, r=None):
    """
    Construct a Toeplitz matrix.
    The Toeplitz matrix has constant diagonals, with c as its first column
    and r as its first row.  If r is not given, ``r == conjugate(c)`` is
    assumed.
    Parameters
    ----------
    c : array_like
        First column of the matrix.  Whatever the actual shape of `c`, it
        will be converted to a 1-D array.
    r : array_like
        First row of the matrix. If None, ``r = conjugate(c)`` is assumed;
        in this case, if c[0] is real, the result is a Hermitian matrix.
        r[0] is ignored; the first row of the returned matrix is
        ``[c[0], r[1:]]``.  Whatever the actual shape of `r`, it will be
        converted to a 1-D array.
    Returns
    -------
    A : (len(c), len(r)) ndarray
        The Toeplitz matrix. Dtype is the same as ``(c[0] + r[0]).dtype``.
    See also
    --------
    circulant : circulant matrix
    hankel : Hankel matrix
    Notes
    -----
    The behavior when `c` or `r` is a scalar, or when `c` is complex and
    `r` is None, was changed in version 0.8.0.  The behavior in previous
    versions was undocumented and is no longer supported.
    Examples
    --------
    >>> from scipy.linalg import toeplitz
    >>> toeplitz([1,2,3], [1,4,5,6])
    array([[1, 4, 5, 6],
           [2, 1, 4, 5],
           [3, 2, 1, 4]])
    >>> toeplitz([1.0, 2+3j, 4-1j])
    array([[ 1.+0.j,  2.-3.j,  4.+1.j],
           [ 2.+3.j,  1.+0.j,  2.-3.j],
           [ 4.-1.j,  2.+3.j,  1.+0.j]])
    """
    c = np.asarray(c).ravel()
    if r is None:
        r = c.conjugate()
    else:
        r = np.asarray(r).ravel()
    # Form a 1D array of values to be used in the matrix, containing a reversed
    # copy of r[1:], followed by c.
    vals = np.concatenate((r[-1:0:-1], c))
    a, b = np.ogrid[0:len(c), len(r) - 1:-1:-1]
    indx = a + b
    # `indx` is a 2D array of indices into the 1D array `vals`, arranged so
    # that `vals[indx]` is the Toeplitz matrix.
    return vals[indx]


def find_params_to_sigmoid(params, MULTI=True, OLDSTYLE = False):
    '''
    Takes the params I was using without sigmoids and finds what would produce the same values in the sigmoid 
    params = unsigmoided parameters
    the max inputs are the upper bounds for the sigmoid
    '''
    if OLDSTYLE:    
        J_max = 3
        i2e_max = 2
        gE_max = 2
        gI_max = 1.5 #because I do not want gI_min = 0, so I will offset the sigmoid
        gI_min = 0.5
        NMDA_max = 1
        plocal_max = 1
        sigR_max = 0.8 #because I do not want sigR_min = 0, so I will offset the sigmoid
        sigR_min = 0.7 # so the max of sigR = sigR_max + sigR_min = 1.5
        sigEE_max = 0.4 # because I do not want sigEE_min = 0 so I will offset the sigmoid
        sigEE_min = 0.1
        sigIE_max = 0.4 # because I do not want sigEE_min = 0 so I will offset the sigmoid
        sigIE_min = 0.1
    else:
        J_max = 3
        i2e_max = 2
        gE_max = onp.sqrt(10)
        #gI_max = 1.5 #because I do not want gI_min = 0, so I will offset the sigmoid
        #gI_min = 0.5
        gI_min = onp.sqrt(0.1)
        gI_max = onp.sqrt(10) - gI_min #because I do not want gI_min = 0, so I will offset the sigmoid
        NMDA_max = 1
        plocal_max = 1
        sigR_max = 0.8 #because I do not want sigR_min = 0, so I will offset the sigmoid
        sigR_min = 0.7 # so the max of sigR = sigR_max + sigR_min = 1.5
        sigEE_max = 0.4 # because I do not want sigEE_min = 0 so I will offset the sigmoid
        sigEE_min = 0.1
        sigIE_max = 0.4 # because I do not want sigEE_min = 0 so I will offset the sigmoid
        sigIE_min = 0.1
    
    #print('Jmax ', Jmax, ' i2e_max ', i2e_max, ' gE_max ', gE_max, ' gI_max ', gI_max, ' gI_min ', gI_min ' NMDA_max ', NMDA_max)
    
    sig_ready_J = -np.log(J_max/params[:4] - 1)
    if not MULTI:
        if len(params) < 6:
            sig_ready_i2e = -np.log(i2e_max/params[4] - 1)
            return np.hstack((sig_ready_J, sig_ready_i2e))
        else:
            sig_gE = -np.log(gE_max/params[4] - 1)
            sig_gI = -np.log(gI_max/(params[5] - gI_min) - 1)
            sig_NMDA = -np.log(NMDA_max/params[6] - 1)
            return np.hstack((sig_ready_J, sig_gE, sig_gI, sig_NMDA))
    else:
        if len(params) < 8:
            sig_ready_i2e = -np.log(i2e_max/params[4] - 1)
            sig_plocal = -np.log(plocal_max/params[5] - 1)
            sig_sigR = -np.log(sigR_max/(params[6]-sigR_min) -1)
            return np.hstack((sig_ready_J, sig_ready_i2e, sig_plocal, sig_sigR))
        elif len(params) == 9:
            sig_gE = -np.log(gE_max/params[4] - 1)
            sig_gI = -np.log(gI_max/(params[5] - gI_min) - 1)
            sig_NMDA = -np.log(NMDA_max/params[6] - 1)
            sig_plocal = -np.log(plocal_max/params[7] - 1)
            sig_sigR = -np.log(sigR_max/(params[8]-sigR_min) -1)
            return np.hstack((sig_ready_J, sig_gE, sig_gI, sig_NMDA, sig_plocal, sig_sigR))
        else:
            sig_gE = -np.log(gE_max/params[4] - 1)
            sig_gI = -np.log(gI_max/(params[5] - gI_min) - 1)
            sig_NMDA = -np.log(NMDA_max/params[6] - 1)
            sig_plocal = -np.log(plocal_max/params[7] - 1)
            sig_sigEE = -np.log(sigEE_max/(params[8]-sigEE_min) -1)
            sig_sigIE = -np.log(sigIE_max/(params[9]-sigIE_min) -1)
            return np.hstack((sig_ready_J, sig_gE, sig_gI, sig_NMDA, sig_plocal, sig_sigEE, sig_sigIE))
        
    
def sigmoid_params(pos_params, MULTI=True, OLDSTYLE=False):
    if OLDSTYLE:    
        J_max = 3
        i2e_max = 2
        gE_max = 2
        gI_max = 1.5 #because I do not want gI_min = 0, so I will offset the sigmoid
        gI_min = 0.5
        NMDA_max = 1
        plocal_max = 1
        sigR_max = 0.8 #because I do not want sigR_min = 0, so I will offset the sigmoid
        sigR_min = 0.7 # so the max of sigR = sigR_max + sigR_min = 1.5
        sigEE_max = 0.4 # because I do not want sigEE_min = 0 so I will offset the sigmoid
        sigEE_min = 0.1
        sigIE_max = 0.4 # because I do not want sigEE_min = 0 so I will offset the sigmoid
        sigIE_min = 0.1
    else:
        J_max = 3
        i2e_max = 2
        gE_max = onp.sqrt(10)
        #gI_max = 1.5 #because I do not want gI_min = 0, so I will offset the sigmoid
        #gI_min = 0.5
        gI_min = onp.sqrt(0.1)
        gI_max = onp.sqrt(10) - gI_min #because I do not want gI_min = 0, so I will offset the sigmoid
        NMDA_max = 1
        plocal_max = 1
        sigR_max = 0.8 #because I do not want sigR_min = 0, so I will offset the sigmoid
        sigR_min = 0.7 # so the max of sigR = sigR_max + sigR_min = 1.5
        sigEE_max = 0.4 # because I do not want sigEE_min = 0 so I will offset the sigmoid
        sigEE_min = 0.1
        sigIE_max = 0.4 # because I do not want sigEE_min = 0 so I will offset the sigmoid
        sigIE_min = 0.1
    
    Jee = J_max * logistic_sig(pos_params[0])
    Jei = J_max * logistic_sig(pos_params[1])
    Jie = J_max * logistic_sig(pos_params[2])
    Jii = J_max * logistic_sig(pos_params[3])
    
    if MULTI:
        if len(pos_params) < 8:
            i2e = i2e_max * logistic_sig(pos_params[4])
            plocal = plocal_max * logistic_sig(pos_params[-2])
            sigR = sigR_max * logistic_sig(pos_params[-1]) + sigR_min
            
            params = np.array([Jee, Jei, Jie, Jii, i2e, plocal, sigR])
        elif len(pos_params) == 9:
            gE = gE_max * logistic_sig(pos_params[4])
            gI = gI_max * logistic_sig(pos_params[5]) + gI_min
            NMDAratio = NMDA_max * logistic_sig(pos_params[6])
            
            plocal = plocal_max * logistic_sig(pos_params[7])
            sigR = sigR_max * logistic_sig(pos_params[8]) + sigR_min
            
            params = np.array([Jee, Jei, Jie, Jii, gE, gI, NMDAratio, plocal, sigR])
        else:
            gE = gE_max * logistic_sig(pos_params[4])
            gI = gI_max * logistic_sig(pos_params[5]) + gI_min
            NMDAratio = NMDA_max * logistic_sig(pos_params[6])
            
            plocal = plocal_max * logistic_sig(pos_params[7])
            sigEE = sigEE_max * logistic_sig(pos_params[8]) + sigEE_min
            sigIE = sigIE_max * logistic_sig(pos_params[9]) + sigIE_min
            
            params = np.array([Jee, Jei, Jie, Jii, gE, gI, NMDAratio, plocal, sigEE, sigIE])
    else:
        if len(pos_params) < 6:
            i2e = i2e_max * logistic_sig(pos_params[4])
            gE = 1
            gI = 1
            NMDAratio = 0.4

            params = np.array([Jee, Jei, Jie, Jii, i2e])

        else:
            i2e = 1
            gE = gE_max * logistic_sig(pos_params[4])
            gI = gI_max * logistic_sig(pos_params[5]) + gI_min
            NMDAratio = NMDA_max * logistic_sig(pos_params[6])

            params = np.array([Jee, Jei, Jie, Jii, gE, gI, NMDAratio])
    
    return params

def logistic_sig(x):
    return 1/(1 + np.exp(-x))