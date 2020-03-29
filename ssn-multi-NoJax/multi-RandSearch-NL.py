import numpy as np
import time 
from datetime import date
import json

import SSN_classes
import SSN_power_spec
import gamma_SSN_losses as losses
import MakeSSNconnectivity as make_conn
import topoRandsearch as tRs

Nsamps = 100000
contrasts = np.asarray([0, 25, 50, 100])
W0 = 1.8
g0 = 0.44
Jxe_max = W0*1.5 
Jxe_min = W0*.5
Jxi_max = W0*1.5 * .5
Jxi_min = W0*.5 * .5
g_max = g0*1.5 #np.sqrt(10)
g_min = g0*.5 #np.sqrt(1/10)
NMDA_min = 0.0
NMDA_max = 0.5 
Plocal_min = 0
Plocal_max = 1
sig_min = 0.15 # in mm 
sig_max = 0.5 # in mm


BALANCED = True

t0 = time.time()
params, paramsNL, rsNL, spectNL, f0NL, interesting_inds, fs, j = tRs.multi_NonLocal_SSN(Nsamps, contrasts, Jxe_max, Jxe_min, g_max, g_min, NMDA_min, NMDA_max, Plocal_min, Plocal_max, sig_min, sig_max, Jxi_max=Jxi_max, Jxi_min=Jxi_min, ARRAY =False)
#params dimensions are smp x num_params
#rs dimensions are smp x 1 x 242 x stimcons
#spects dimensions are smp x (cons + probes) x fnums
# f0s dimensions are smp x f0/hw/err x stimcons

t0 = time.time() - t0
print('run time was', t0)

dobj = date.today()
f_ender = dobj.strftime("%y-%m-%d")
fname = 'NL-rand-CorrTimeRetino_10-'+f_ender+'.json'

#results = {'params':params, 'rates':rs, 'spects':spects, 'f0s':f0s, 'paramsNL':paramsNL, 'rsNL':rsNL, 'spectNL':spectNL, 'f0NL':f0NL, 'interesting_inds':interesting_inds}
results = {'params':params, 'paramsNL':paramsNL, 'rsNL':rsNL, 'spectNL':spectNL, 'f0NL':f0NL, 'interesting_inds':interesting_inds}

with open(fname, 'w') as json_file:
    json.dump(results, json_file)
