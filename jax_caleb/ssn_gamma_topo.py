import os
from sys import platform
import time
import jax.numpy as np
#mport numpy as np
from jax import grad, value_and_grad, jit, ops
import jax.random as random

import scipy.io as sio

import SSN_classes
import SSN_power_spec
import gamma_SSN_losses as losses
import make_plot
import MakeSSNconnectivity as make_conn
from util import sigmoid_params

if platform == 'darwin':
    gd_iters = 10
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    gd_iters = 1000

#the constant (non-optimized) parameters:

#fixed point algorithm:
dt = 1
xtol = 1e-6
Tmax = 500

#power spectrum resolution and range
fnums = 30
freq_range = [15,100]

#SSN parameters
n = 2
k = 0.04
tauE = 20 # in ms
tauI = 10 # in ms
psi = 0.774

t_scale = 1
tau_s = np.array([3, 5, 100])*t_scale #in ms, AMPA, GABA, NMDA current decay time constants

contrasts = np.array([0, 25, 50, 100])

gridsizedeg = 4
dradius = gridsizedeg/8
gridperdeg = 5
gridsize = round(gridsizedeg*gridperdeg) + 1
magnFactor = 2 #mm/deg
#biological hyper_col length is ~750 um, magFactor is typically 2 mm/deg in macaque V1
# hyper_col = 0.75/magnFactor
hyper_col = 8
Lx = gridsizedeg
Ly = gridsizedeg
# r_cent = np.array([0.3, 0.6, 0.9, 1.2, 1.5])
# r_cent = np.arange(dradius, round(gridsizedeg/2)+dradius, dradius)
r_cent = np.array([0.9750])

contrasts = np.array([0, 25, 50, 100])

X,Y, deltaD = make_conn.make_neur_distances(gridsizedeg, gridperdeg, hyper_col, Lx, Ly, PERIODIC = False)
OMap, _= make_conn.make_orimap(hyper_col, X, Y)
Inp, stimCon = make_conn.makeInputs(OMap, r_cent, contrasts, X, Y, gridperdeg=gridperdeg, gridsizedeg=gridsizedeg, Lx=Lx)
Contrasts = stimCon[0, :]

Ne, Ni = deltaD.shape
tau_vec = np.hstack((tauE*np.ones(Ne), tauI*np.ones(Ni)))
N = Ne + Ni #number of neurons in the grid.

params = [1.95, 1.25, 2.45, 1.5, 1.25] #Jee, Jei, Jie, Jii, i2e --- Modelo that WORKED! w/ locality = 1
Plocal = 1
sigR = 0.7

#unpack parameters
Jee = params[0]
Jei = params[1]
Jie = params[2]
Jii = params[3]

if len(params) < 6:
    i2e = params[4]
    gE = 1
    gI = 1 * i2e
    NMDAratio = 0.4
else:
    i2e = 1
    gE = params[4]
    gI = params[5]
    NMDAratio = params[6]
    
W = make_conn.make_full_W(Plocal, Jee, Jei, Jie, Jii, sigR, gridperdeg=gridperdeg, gridsizedeg= gridsizedeg)

ssn_Ampa = SSN_classes._SSN_AMPAGABA(tau_s, NMDAratio, n, k, Ne, Ni, tau_vec, W)
r_init = np.zeros([ssn_Ampa.N, len(Contrasts)])
inp_vec = np.vstack((gE*Inp, gI*Inp))

r_fp, CONVG = ssn_Ampa.fixed_point_r(inp_vec, r_init=r_init, Tmax=Tmax, dt=dt, xtol=xtol)

gen_inds = np.arange(len(Contrasts))
rad_inds = np.arange(len(contrasts)-1, len(r_cent)+len(contrasts)-1)#np.where(stimCon[0, :] == np.max(Contrasts), gen_inds, 0)
con_inds = np.hstack((np.arange(0, len(contrasts)-1), len(r_cent) + len(contrasts)-2))#np.where(stimCon[1, :] == np.max(stimCon[1,:]), gen_inds, 0)
gabor_inds = -1

trgt = np.floor(Ne/2)

con_inds = np.hstack((con_inds, gabor_inds))
cons = len(con_inds)
ssn_Ampa.topos_vec = np.ravel(OMap)

if ssn_Ampa.N > 2:
    LFPtarget = trgt + np.array( [ii * gridsize for ii in range(int(np.floor(gridsize/2)))])
else:
    LFPtarget = None
    
spect, fs, f0, _ = SSN_power_spec.linear_PS_sameTime(ssn_Ampa, r_fp[:, con_inds], SSN_power_spec.NoisePars(), freq_range, fnums, cons, LFPrange=[LFPtarget[0]])