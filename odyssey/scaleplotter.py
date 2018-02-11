#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 12:05:09 2017

@author: Maxwell

Run averaging code:
    
    Designed to work with the odessey code "launch_fouriernetwork."
    
    Assuming all the runs in an experiment have been done with the same hyperparameters, 
    this code will average the desired parameters
"""

import glob
import numpy as np
import argparse
from hand_code_real_fft_network_odyssey import hand_code_real_fft_network_fun
from hand_code_real_fft_network_odyssey import hand_code_fun_layer_less
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()

parser.add_argument('-expt', type=int)
parser.add_argument('-size', type=int)

settings = parser.parse_args(); 

experiment_num = settings.expt

weight_in = 1.
weight_out = 3.

complex_sizes = [16, 32, 64, 128]


def l0norm(W):
    norm0 = 0
    for i in range(len(W)):
        ones = np.ones(W[i].shape)
        norm0 = norm0 + np.sum(ones[W[i] != 0])
    return norm0

directory_path = "/n/home09/mnye/linear_fft/odyssey/results/fouriernetwork/expt%d/data/" % experiment_num

size_list_in = []
scaling_factor_list_in = []

size_list_out = []
scaling_factor_list_out = []


for res_num in glob.glob(directory_path + '*.npz'):
    try:
        variables = np.load(res_num)
        run_params = variables['params'][0]    
        
        if (run_params.complexsize in complex_sizes):

            if run_params.weightscale == weight_in:
                size_list_in.append(run_params.complexsize)
                scaling_factor_list_in.append(variables['scaling_factors'][-1])
            if run_params.weightscale == weight_out:
                size_list_out.append(run_params.complexsize)
                scaling_factor_list_out.append(variables['scaling_factors'][-1])         

    except IOError:
        print("there exists a trial which is not complete")
        #Whatever man    

assert len(size_list_out) == len(size_list_in)

true_scales = np.ones(len(size_list_in))


#sort array
order_in = np.argsort(size_list_in)
size_list_in = np.array(size_list_in)[order_in]
scaling_factor_list_in = np.array(scaling_factor_list_in)[order_in]

order_out = np.argsort(size_list_out)
size_list_out= np.array(size_list_out)[order_out]
scaling_factor_list_out = np.array(scaling_factor_list_out)[order_out]

assert (size_list_out == size_list_in).all()

print "in", scaling_factor_list_in
print "out", scaling_factor_list_out

print "sizelist", size_list_in
#sample for the plotting
fig = plt.figure()
fig, ax = plt.subplots()
ax.plot(size_list_in, scaling_factor_list_in,label='in')
ax.plot(size_list_out, scaling_factor_list_out, label='out')
ax.plot(size_list_in, true_scales, label='true scale')

ax.set(title='FFT scaling',
       xlabel='network size',
       ylabel='complexity scale factor')
ax.legend(loc='best') 
fig.savefig('FFTscaleexpt%d.png' %(experiment_num), dpi = 200)
