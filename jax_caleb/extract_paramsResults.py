import scipy.io as sio
import numpy as np

def extract(dir_name, numperm=1000):
    bb = dir_name.split('_')
    
    if bb[0] = 'Rand':
        f_header = 'rand_2neuron-'
    elif bb[0] = 'Targ':
        f_header = 'targeted_2neuron-'
    
    f0 = np.empty(numperm, 3)
    hw = f0
    CONVG = np.zeros(numperm)
    Jee = np.empty(numperm)
    Jei = Jee
    Jie = Jee
    Jii = Jee
    gE = Jee
    gI = Jee
    NMDAratio = Jee
    
    for pp in range(numperm):
        fname = f_header+str(pp)+'.mat'
        
        aa = sio.loadmat(fname)
        
        