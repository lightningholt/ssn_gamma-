import scipy.io as sio
import numpy as np


def extract(dir_name, numperm=1000):
    cons = 4
    bb = dir_name.split('_')
    
    if bb[0] == 'Rand':
        f_header = dir_name+'/rand_2neuron-'
    elif bb[0] == 'Targ':
        f_header = dir_name+'/targeted_2neuron-'
    
    f0 = np.empty((numperm, 3))
    hw = np.empty((numperm, 3))
    err = np.empty(numperm)
    CONVG = np.zeros(numperm)
    eigvals = np.empty((numperm, cons, 6))
    params = np.empty((numperm, 7))
#     Jee = np.empty(numperm)
#     Jei = Jee
#     Jie = Jee
#     Jii = Jee
#     gE = Jee
#     gI = Jee
#     NMDAratio = Jee
    
    for pp in range(numperm):
        fname = f_header+str(pp)+'.mat'
        aa = sio.loadmat(fname)
        params[pp, :] = aa['params'][0]
        
#         Jee[pp] = params[0]
#         Jei[pp] = params[1]
#         Jie[pp] = params[2]
#         Jii[pp] = params[3]
#         gE[pp] = params[4]
#         gI[pp] = params[5]
#         NMDAratio[pp] = params[6]
        
        CC = aa['CONVG']
        ee = aa['err'][0]
        CONVG[pp] = CC
        #err of 0 means no err, so if all ee are 0 then I want it to return 0
        err[pp] = 1 - np.all(ee == 0) 
        
        f0[pp, :] = aa['f0'][0][1:] 
        hw[pp, :] = aa['hw'][0][1:]
        
        if CC > 0 and np.all(ee == 0):
            #condition means no gamma peak at 0, and firing rates converged
            ev = np.linalg.eigvals(aa['Jacobian'])
            eigvals[pp, :,:] = ev
        else:
            eigvals[pp, :,:] = np.nan
    
    Results = {'f0':f0,
              'hw':hw,
              'CONVG':CONVG,
               'err':err,
              'eigvals':eigvals,
               'params':params}
#                'Jee':Jee,
#                'Jei':Jei,
#                'Jie':Jie,
#                'Jii':Jii,
#                'gE':gE,
#                'gI':gI,
#                'NMDAratio':NMDAratio
#               }
    
    fout = dir_name+'/extractedResults.mat'
    sio.savemat(fout, Results)
    
    return Results
        
def peak_histograms_correlations(Results):
    f0 = Results['f0']
    hw = Results['hw']
    params = Results['params']
    Jee = params[:, 0]
    Jei = params[:, 1]
    Jie = params[:, 2]
    Jii = params[:, 3]
    gE = params[:, 4]
    gI = params[:, 5]
    NMDAratio = params[:, 6]
    
    df0 = np.diff(f0, axis=1)
    dhw = np.diff(hw, axis=1)
    
    df0_50_25 = df0[:,0]
    df0_100_50 = df0[:,1]
    df0_mean = np.mean(df0, axis=1)
    
    dhw_50_25 = dhw[:,0]
    dhw_100_50 = dhw[:,1]
    dhw_mean = np.mean(dhw, axis=1)
    