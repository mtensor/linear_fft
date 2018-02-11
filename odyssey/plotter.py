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
complex_size = settings.size


def l0norm(W):
    norm0 = 0
    for i in range(len(W)):
        ones = np.ones(W[i].shape)
        norm0 = norm0 + np.sum(ones[W[i] != 0])
    return norm0

directory_path = "/n/home09/mnye/linear_fft/odyssey/results/fouriernetwork/expt%d/data/" % experiment_num

key_cutoff_list = []
weightscale_list = []

for res_num in glob.glob(directory_path + '*.npz'):
    #try:
    variables = np.load(res_num)
    run_params = variables['params'][0]    
        
    if run_params.complexsize == complex_size:
        key_cutoff_list.append(variables['key_cutoff'])
        weightscale_list.append(run_params.weightscale)
            #scaling_factors_list.append(variables['scaling_factors'])
    
            #fun_loss_list.append(variables['fnlossvec'][-1])
#except IOError:
#    print("there exists a trial which is not complete")
        #Whatever man    
 

#sort array
order = np.argsort(weightscale_list)
weightscale_list = np.array(weightscale_list)[order]
key_cutoff_list = np.array(key_cutoff_list)[order]

fig = plt.figure()
fig, ax = plt.subplots()
print "weightscale_list", weightscale_list
print "key key_cutoff list", key_cutoff_list
plt.plot(weightscale_list,key_cutoff_list)
ax.set(title='FFT convergence',
       xlabel='Weight initialization noise scale',
       ylabel='key cutoff value')

#fig.savefig(plotpath + "paritysize%d.png" % size, dpi=200)
fig.savefig('FFTexpt%dsize%d.png' %(experiment_num, complex_size), dpi = 200)
