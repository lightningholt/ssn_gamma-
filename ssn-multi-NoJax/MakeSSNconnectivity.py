# import jax.numpy as np
# import jax.random as random
import numpy as np

# Python functions that recreates a MATLAB function of the same name, but hopefully in a clearer way. 
# Also contains functions that make inputs to the network

def make_neur_distances(gridsizedeg, gridperdeg, hyper_col, PERIODIC = False):
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
    
    Lx = gridsizedeg
    Ly = gridsizedeg
    
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

def make_orimap(hyper_col, X, Y, nn=30, prngKey=4):
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
    

    for j in range(nn):
        kj = kc * np.array([np.cos(j * np.pi/nn), np.sin(j * np.pi/nn)])
        
        #sj = 2 * np.random.randint(subkey, shape=(), minval=1, maxval=3)-3 #random number that's either + or -1. 
        sj = 2 * np.random.randint(0,2) -1 #random +/- 1 
        #randint inputs: PRNGkey, size tuple, minval (incl), maxval(excl)
        
        phij = np.random.rand() * 2*np.pi
        tmp = (X*kj[0] + Y*kj[1]) * sj + phij
        z = z + np.exp(1j * tmp)
        
        #key, subkey = random.split(key)
    
    OMap = np.angle(z)
    OMap = (OMap - np.min(OMap)) * 180/(2*np.pi)
    Nthetas = len(OMap.ravel())
    
    return OMap, Nthetas

def make_Wxx_dist(dist, ori_dist, sigma, sigma_ori, from_neuron, MinSyn=1e-4, JNoise=0, JNoise_Normal=False, CellWiseNormalized=True, prngKey=0):
    '''
    Makes the connection from neuron y to neuron x  if that connection only depended on their spatial position and their preferred orientation difference.
    dist = distances between neurons (deltaD <- in degrees)
    ori_distance = differences between preferred orientations
    sigma = sets the length scales (should be sigEE, sigEI, sigIE, or sigII) <- degree units
    sigma_ori = length scale of ori differences (MATLAB used 45)
    from_neuron = string denoting either E or I neurons 
    
    outpus 
    Wxy = connection strength between neuron y and neuron x. 
    '''
    
    #key = random.PRNGKey(prngKey)
    
    if from_neuron == 'E':
        W = np.exp(-np.abs(dist)/sigma - ori_dist**2/(2*sigma_ori**2))
    elif from_neuron == 'I':
        W = np.exp(-dist**2/(2*sigma**2) - ori_dist**2/(2*sigma_ori**2))
    
    if JNoise_Normal:
        rand_dist = lambda x: np.random.normal(*x.shape)
    else:
        rand_dist = lambda x: 2*np.random.rand(*x.shape) - 1
    
    W = (1 + (JNoise*rand_dist(W)))*W
    W = np.where(W < MinSyn, 0, W)
    tW = np.sum(W, axis=1)
    
    if CellWiseNormalized:
        W = W / tW
    else:
        W = W / np.mean(tW)
    
    return W

def make_full_W(Plocal, Jee, Jei, Jie, Jii, sigEE, sigIE, deltaD, OMap, sigXI = 0.09, PlocalIE = None):
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
    
    if PlocalIE is None:
        PlocalIE = Plocal
    
    #X, Y, deltaD = make_neur_distances(gridsizedeg, gridperdeg, hyper_col)
    #OMap, Nthetas = make_orimap(hyper_col, X, Y)
    
    OriDist =  np.abs(np.ravel(OMap) - np.ravel(OMap)[:, None])
    OriDist = np.where( OriDist > 90, 180- OriDist, OriDist)
    sigOri = 45
    
#     sigEE = 0.35*np.sqrt(sigR)
#     sigIE = 0.35/np.sqrt(sigR)
    sigEI = sigXI
    sigII = sigXI
    
    # Wxe = Jxe * (lambda * identity + (1-lambda)*exp(distance)* gaussian(Ori))
    Wee = Jee * (Plocal*np.eye(deltaD.shape[0], deltaD.shape[1]) + (1 - Plocal) * make_Wxx_dist(deltaD, OriDist, sigEE, sigOri, 'E'))
    
    #Wxi = Jxi * gaussian(distance and Ori)
    Wei = -Jei * make_Wxx_dist(deltaD, OriDist, sigEI, sigOri, 'I')
    Wie = Jie * (PlocalIE*np.eye(deltaD.shape[0], deltaD.shape[1]) + (1 - PlocalIE) * make_Wxx_dist(deltaD, OriDist, sigIE, sigOri, 'E'))
    Wii = -Jii * make_Wxx_dist(deltaD, OriDist, sigII, sigOri, 'I')
    
    
    W = np.vstack((np.hstack((Wee, Wei)), np.hstack((Wie, Wii))))
    
    return W

def makeInputs(OMap, r_cent, contrasts, X, Y, gridsizedeg=4, gridperdeg=5, AngWidth=32):
    '''
    makes the input arrays for the various stimulus conditions
    all radii at the highest contrast - to test Surround Suppression
    all contrasts at the highest radius - to test contrast effect
    highest contrast and radius with a Gabor filter - to test Ray-Maunsell Effect
    
    OMap = orientation preference across the cortex
    r_cent = array of stimulus radii
    contrasts = array of stimulus contrasts
    various parameters of the network
    
    Outputs
    StimConds = array of dim Ne x stimCondition (the name is short for Stimulus Conditions)
    stimCondition = [max radii * varying contrasts, max contrast * vary radii, Gabor]
    
    '''
    rads = np.hstack((np.max(r_cent)*np.ones(len(contrasts)-1), r_cent)) # cause I don't want to double up the Contrast = 100 condition
    Contrasts = np.hstack((contrasts, np.ones(len(r_cent))*np.max(contrasts))) # need to add one for Gabor condition, but I would subtract one to not double up the C= 100 R= max condition
    
    gridsize = OMap.shape
    dx = gridsizedeg/gridsize[0]
    
    Mid1 = int(np.floor(gridsize[0]/2))
    Mid2 = int(np.floor(gridsize[1]/2))
    
    # Python does linear indexing weird, just going to use the found midpts
    # trgt = onp.ravel_multi_index((Mid1, Mid2), (Len[0], Len[1]))

    Orientation = OMap[Mid1, Mid2]

    dOri = np.abs(OMap - Orientation)
    dOri = np.where(dOri > 90, 180-dOri, dOri)
    In0 = np.ravel(np.exp(-dOri**2/(2*AngWidth**2)))
    
    RFdecay = 0.8/2 #biologic decay is 0.8 mm, magfactor =2 mm/deg 
    RFdecay = 0.1 * RFdecay
    RFdecay = 0.04
    #RFdecay = dx
    #GaborSigma = 0.3*np.max(r_cent) 
    GaborSigma = 0.5

    x0 = X[Mid1, Mid2]
    y0 = Y[Mid1, Mid2]

    x_space = X - x0
    y_space = Y - y0

    # find the distances across the cortex
    r_space = np.ravel(np.sqrt(x_space**2 + y_space**2))
    
    #find the spatial input for a constant grating
    InSr = (1 - (1/(1 + np.exp(-(r_space - rads[:, None])/RFdecay))))
    #find the spatial input for a Gabor 
    InGabor = np.exp(- r_space**2/2/GaborSigma**2);
    #include the contrasts with it
    if len(contrasts) > 1:
        StimConds = Contrasts[ :,None] * np.vstack((InSr, InGabor))
    else:
        StimConds = Contrasts[:,None] *InSr
    StimConds = StimConds * In0
    #include the relative drive between E and I cells  -- nixing this cause gE and gI are parametrs
    #InSpace = np.hstack( (StimConds, gI*StimConds)).T #.T makes it neurons by stimcond
    
    #array to reference to find max contrasts, etc
    stimulus_condition = np.vstack((Contrasts, np.hstack((rads, np.max(rads)))))
    
    return StimConds.T, stimulus_condition, InSr