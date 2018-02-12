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
normlist = []

for res_num in glob.glob(directory_path + '*.npz'):
    #try:
    variables = np.load(res_num)
    run_params = variables['params'][0]    
        
    if run_params.complexsize == complex_size:
        normlist.append(variables['l0_norms'][-1])
        weightscale_list.append(run_params.weightscale)
            #scaling_factors_list.append(variables['scaling_factors'])
    
            #fun_loss_list.append(variables['fnlossvec'][-1])
#except IOError:
#    print("there exists a trial which is not complete")
        #Whatever man    
 
Wopt = hand_code_fun_layer_less(complex_size,0)
l0opt = l0norm(Wopt)
l0optlist = np.ones(len(weightscale_list)) * l0opt 



#sort array
order = np.argsort(weightscale_list)
weightscale_list = np.array(weightscale_list)[order]
normlist = np.array(normlist)[order]

SMALL_SIZE = 8
MEDIUM_SIZE = 12
BIGGER_SIZE = 16

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
#plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
#plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
#plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title



fig = plt.figure()
fig, ax = plt.subplots()
print "weightscale_list", weightscale_list
print "normlist", normlist
ax.plot(weightscale_list,normlist, marker='o',markersize=10, linewidth=4.0, label='Network after training and optimal sparsification')
ax.plot(weigthscale_list,l0optlist, '--',label='Hand-coded FFT')
ax.set(title='Convergence',
       xlabel='Initialization noise scale',
       ylabel='L_0 norm')
ax.legend(loc='best')

#fig.savefig(plotpath + "paritysize%d.png" % size, dpi=200)
fig.savefig('FFTexpt%dsize%dl0.png' %(experiment_num, complex_size), dpi = 200)
