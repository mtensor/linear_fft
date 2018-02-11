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


parser = argparse.ArgumentParser()

parser.add_argument('-expt', type=int)
parser.add_argument('-size', type=int)

settings = parser.parse_args(); 

experiment_num = settings.expt
complex_size = settings.size

weight_in = 1.
weight_out = 4.

complex_sizes = [16, 32, 64, 128]


def l0norm(W):
    norm0 = 0
    for i in range(len(W)):
        ones = np.ones(W[i].shape)
        norm0 = norm0 + np.sum(ones[W[i] != 0])
    return norm0

directory_path = "/n/home09/mnye/linear_fft/odyssey/results/fouriernetwork/expt%d/data/" % experiment_num

size_list = []
scaling_factor_list = []

for res_num in glob.glob(directory_path + '*.npz'):
    try:
        variables = np.load(res_num)
        run_params = variables['params'][0]    
        
        if (run_params.complexsize in complex_sizes) and (variables['fnlossvec'][-1] < 16.):
            
            size_list.append(run_params.complexsize)
            scaling_factor_list.append(variables['scaling_factors'][-1])
    
    except IOError:
        print("there exists a trial which is not complete")
        #Whatever man    
assert (cutoff_list_list[0] == cutoff_list_list[i] for i in range(len(cutoff_list_list)))
cutoff_list = cutoff_list_list[0]
 
rect_errors_array = np.array(rect_errors_list)
l0_norms_array = np.array(l0_norms_list)
scaling_factors_array = np.array(scaling_factors_list)
    
av_rect_error = np.mean(rect_errors_array, axis=0) 
av_l0_norm = np.mean(l0_norms_array, axis = 0)
av_scaling_factor = np.mean(scaling_factors_array, axis=0)

av_fun_loss = np.mean(fun_loss_list)

optimal_L0 = l0norm(hand_code_fun_layer_less(complex_size,0))

logn = int(np.ceil(np.log2(complex_size)))
nlogn = float(complex_size * logn)
optimal_scale_factor = optimal_L0 / nlogn

print("Final average function error (unrectified): %g" %av_fun_loss)
for index in range(len(cutoff_list)):
    print("Cutoff factor: %g" %(cutoff_list[index]))
    print("\t Average function error of rectified network: %g" %(av_rect_error[index]))
    print("\t Average L_0 norm: %g (hand-coded value is %g) "%(av_l0_norm[index], optimal_L0))
    print("\t Average Complexity scaling factor: %g (hand-coded value is %g)" %(av_scaling_factor[index], optimal_scale_factor))



#sample for the plotting
x1 = np.arange(1,5,.1)
y1 = np.arange(1,5,.1)

x2 = np.arange(2,6, .1)
y2 = np.arange(3,7, .1)

fig = plt.figure()
fig, ax = plt.subplots()
ax.plot(x1, y1, label='Model length')
ax.plot(x2, y2, label='Data length')

ax.set(title='FFT convergence',
       xlabel='Weight initialization noise scale',
       ylabel='key cutoff value')
ax.legend(loc='best') 
fig.savefig('FFTexpt%dsize%d.png' %(experiment_num, complex_size), dpi = 200)
