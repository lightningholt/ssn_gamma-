import jax.numpy as np
import jax.random as random
# import numpy as onp

# Python functions that recreates a MATLAB function of the same name, but hopefully in a clearer way. 

def make_neur_distances(gridsizedeg, gridperdeg, hyper_col, Lx, Ly, PERIODIC = True):
    '''
    Makes a matrix of distances between neurons in the network
    gridsizedeg = size of grid in degress
    gridperdeg = number of grid points per degree
    hyper_col = hypercolumn length of the network
    Lx = length of the x direction in degrees
    Ly = length of the y direction in degrees
    
    outputs:
    X = matrix of distances between neurons in the x direction
    Y = matrix of distances between neurons in the y direction
    deltaD = matrix of distances between neurons used to make W
    '''
    gridsize = 1 + round(gridperdeg *  gridsizedeg)
    
    dx = Lx/(gridsize - 1)
    dy = Ly/(gridsize - 1)
    
    [X, Y] = np.meshgrid(np.arange(0, Lx+dx, dx),np.arange(0, Ly+dy, dy))
    
    [indX, indY] = np.meshgrid(np.arange(gridsize), np.arange(gridsize))
    XDist = np.abs(np.ravel(indX, 'F') - np.ravel(indX, 'F')[:,None])
    YDist = np.abs(np.ravel(indY, 'F') - np.ravel(indY,'F')[:,None])
    
    if PERIODIC:
        XDist = np.where( XDist > gridsize/2, gridsize - XDist, XDist) 
        YDist = np.where( YDist > gridsize/2, gridsize - YDist, YDist)
    deltaD = np.sqrt(XDist**2 + YDist**2)*dx
    
    return X, Y, deltaD

def make_orimap(hyper_col, X, Y, nn=30, prngKey=0):
    '''
    Makes the orientation map for the grid
    hyper_col = hyper column length for the network in retinotopic degrees
    X = distances between neurons in retinotopic degrees
    Y = distances between neurons in retinotopic degrees
    
    Outputs
    OMap = orientation preference for each cell in the network
    Nthetas = 1/2 the number of cells in the network (or equivalent to number of E or I cells)
    '''
    kc = 2*np.pi/(hyper_col)

    z = np.zeros_like(X)
    key = random.PRNGKey(prngKey)
    subkey = key

    for j in range(nn):
        kj = kc * np.array([np.cos(j * np.pi/nn), np.sin(j * np.pi/nn)])
        
        sj = 2 * random.randint(subkey, shape=(), minval=1, maxval=3)-3 #random number that's either + or -1. 
        #randint inputs: PRNGkey, size tuple, minval (incl), maxval(excl)
        
        phij = random.uniform(subkey)*2*np.pi
        tmp = (X*kj[0] + Y*kj[1]) * sj + phij
        z = z + np.exp(1j * tmp)
        
        key, subkey = random.split(key)
    
    OMap = np.angle(z)
    OMap = (OMap - np.min(OMap)) * 180/(2*np.pi)
    Nthetas = len(OMap.ravel())
    
    return OMap, Nthetas

def make_Wxx_dist(dist, ori_dist, sigma, sigma_ori, from_neuron, MinSyn=1e-4, JNoise=0, JNoise_Normal=False, CellWiseNormalized=True, prngKey=0):
    '''
    Makes the connection from neuron y to neuron x  if that connection only depended on their spatial position and their preferred orientation difference.
    dist = distances between neurons (deltaD)
    ori_distance = differences between preferred orientations
    sigma = sets the length scales (should be sigEE, sigEI, sigIE, or sigII)
    sigma_ori = length scale of ori differences (MATLAB used 45)
    from_neuron = string denoting either E or I neurons 
    
    outpus 
    Wxy = connection strength between neuron y and neuron x. 
    '''
    
    key = random.PRNGKey(prngKey)
    
    if from_neuron == 'E':
        W = np.exp(-np.abs(dist)/sigma - ori_dist**2/(2*sigma_ori**2))
    elif from_neuron == 'I':
        W = np.exp(-dist**2/(2*sigma**2) - ori_dist**2/(2*sigma_ori**2))
    
    if JNoise_Normal:
        rand_dist = lambda x: random.normal(key, x.shape)
    else:
        rand_dist = lambda x: 2*random.uniform(key, x.shape) - 1
    
    W = (1 + (JNoise*rand_dist(W)))*W
    W = np.where(W < MinSyn, 0, W)
    tW = np.sum(W, axis=1)
    
    if CellWiseNormalized:
        W = W / tW
    else:
        W = W / np.mean(tW)
    
    return W

def make_full_W(Plocal, Jee, Jei, Jie, Jii, sigR, gridsizedeg = 4, gridperdeg = 5, hyper_col = 8, sigXI =0.02):
    '''
    Function that makes the full rank W = [[Wee, Wei], [Wie, Wii]] which obeys Dale's law (meaning Wxi is defined negative)
    
    Plocal = locality (how much of Wxe are delta functions vs not)
    All J's are defined positive if using sigmoid parameterization
    Jee = exc->exc connection strengths 
    Jei = inh->exc connection strengths 
    Jie = exc->inh connection strengths 
    Jii = inh->inh connection strengths
    
    Outputs:
    Currently not a sparse array. May want to use sparse arrays later 
    W = [[Wee, Wei], [Wie, Wii]]
    '''
    
    
    Lx = gridsizedeg
    Ly = gridsizedeg
    
    X, Y, deltaD = make_neur_distances(gridsizedeg, gridperdeg, hyper_col, Lx, Ly)
    OMap, Nthetas = make_orimap(hyper_col, X, Y)
    
    OriDist =  np.abs(np.ravel(OMap) - np.ravel(OMap)[:, None])
    OriDist = np.where( OriDist > 90, 180- OriDist, OriDist)
    sigOri = 45
    
    sigEE = 0.35*np.sqrt(sigR)
    sigIE = 0.35/np.sqrt(sigR)
    sigEI = sigXI
    sigII = sigXI
    
    # Wxe = Jxe * (lambda * identity + (1-lambda)*exp(distance)* gaussian(Ori))
    Wee = Jee * (Plocal*np.eye(deltaD.shape[0], deltaD.shape[1]) + (1 - Plocal) * make_Wxx_dist(deltaD, OriDist, sigEE, sigOri, 'E'))
    
    #Wxi = Jxi * gaussian(distance and Ori)
    Wei = -Jei * make_Wxx_dist(deltaD, OriDist, sigEI, sigOri, 'I')
    Wie = Jie * (Plocal*np.eye(deltaD.shape[0], deltaD.shape[1]) + (1 - Plocal) * make_Wxx_dist(deltaD, OriDist, sigIE, sigOri, 'E'))
    Wii = -Jii * make_Wxx_dist(deltaD, OriDist, sigII, sigOri, 'I')
    
    
    W = np.vstack((np.hstack((Wee, Wei)), np.hstack((Wie, Wii))))
    
    return W