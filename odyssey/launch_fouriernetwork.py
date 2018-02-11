import numpy as np
from subprocess import call
import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-expt', type=int)

settings = parser.parse_args(); 

param_fn = "params_fouriernetwork_%d.txt" % settings.expt
fo = open(param_fn, "w")

os.makedirs("/n/home09/mnye/linear_fft/odyssey/results/fouriernetwork/expt%d/data" % settings.expt)
os.makedirs("/n/home09/mnye/linear_fft/odyssey/results/fouriernetwork/expt%d/logs" % settings.expt)


#depths

weightscales = [1.,1.25, 1.5, 1.75, 2., 2.25, 2.5, 2.75 , 3, 3.25, 3.5, 3.75, 4.]
rseed = 2
noffsets = 1
rseed_offsets = np.linspace(0,rseed*(noffsets-1),noffsets).astype(int)
expt = settings.expt

complexsizes = [32]
optimizer_params = [0.0001]
L1_betas = [0.00001] #[0.001, 0.0001, 0.00005]
boost_factors = [1.]
hidden_width_multipliers = [1.0]

i = 1
for n in complexsizes:
    for boost_factor in boost_factors:
        for hidden_width_multiplier in hidden_width_multipliers:
            for optimizer in optimizer_params:
                for beta in L1_betas:
                    for ws in weightscales:
                        for roff in rseed_offsets:                   
                            savefile = "/n/home09/mnye/linear_fft/odyssey/results/fouriernetwork/expt%d/data/res%d.npz" %(expt, i) 
                            fo.write("-rseed %d -rseed_offset %d -weightscale %g -complexsize %d -beta %g -optimizer %g -epochs 500000 -savefile %s -boost_factor %g -hidden_width_multiplier %g\n" % (rseed, roff, ws, n, beta, optimizer, savefile, boost_factor, hidden_width_multiplier))
                            i = i+1
                            #what is lr?
                            #epoch thing may need to be cut
fo.close()

call("python run_odyssey_array.py -cmd run_fouriernetwork_odyssey.py -expt %d -cores 8 -hours 25 -mem 24000 -partition serial_requeue -paramfile %s" % (expt,param_fn), shell=True)
