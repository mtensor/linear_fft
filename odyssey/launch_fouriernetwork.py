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

weightscales = [.1]
rseed = 2
noffsets = 4
rseed_offsets = np.linspace(0,rseed*(noffsets-1),noffsets).astype(int)
expt = settings.expt

complexsizes = [16, 32, 64, 128, 256]
optimizer_params = [0.001]
L1_betas = [0.01, 0.05, 0.005]

i = 1
for n in complexsizes:
    for optimizer in optimizer_params:
        for beta in L1_betas:
            for ws in weightscales:
                for roff in rseed_offsets:
                    fo.write("-rseed %d -rseed_offset %d -weightscale %g -complexsize %d -beta %g -optimizer %g -epochs 10000000 -runtoconv -savefile /n/home09/mnye/linear_fft/results/fouriernetwork/expt%d/data/res%d.npz\n" % (rseed, roff, ws, n, beta, optimizer, expt, i))
                    i = i+1
                    #what is lr?
                    #epoch thing may need to be cut
fo.close()

call("python run_odyssey_array.py -cmd run_fouriernetwork_odyssey.py -expt %d -cores 8 -hours 25 -mem 24000 -partition serial_requeue -paramfile %s" % (expt,param_fn), shell=True)
