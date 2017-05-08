#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun May  7 17:35:13 2017

@author: Maxwell
"""

from IPython import get_ipython
get_ipython().magic('reset -sf')

import numpy as np

#first, code as if complex. And compare to actual np.fft.fft
n = 8 #must be power of 2
logn = int(np.log2(n))

W = [np.zeros([n,n],dtype = complex) for i in range(2*logn)]

#code for last laye
#Make big wrapper thinggy 
#fix so input isn't stupid - or don't 
def butterfly(W_in):
    n = W_in.shape[0]
    W = np.zeros([n,n],dtype = complex)
    for k in range(n/2):
        W[k,k] = 1
        W[k,k+n/2] = np.exp(-2*np.pi*1j*k/n)
        W[k+n/2,k+n/2] = - np.exp(-2*np.pi*1j*k/n)
        W[k+n/2,k] = 1
    return W
        
def rearrange(W_in):
    #I dont think we need to input an n so help me god
    n = W_in.shape[0]
    W = np.zeros([n,n],dtype = complex)
    for k in range(n/2):
        W[k,2*k] = 1
        W[k+n/2,2*k+1] = 1
    return W

def rearrange_rec(W_in,rec_depth):
    cutsize = 2**rec_depth
    n = W_in.shape[0]
    W = np.zeros([n,n],dtype = complex)
    for i in range(cutsize):
        W[i*n/cutsize:(i+1)*n/cutsize,i*n/cutsize:(i+1)*n/cutsize] = rearrange(
                W[i*n/cutsize:(i+1)*n/cutsize,i*n/cutsize:(i+1)*n/cutsize])
    return W
    
def butterfly_rec(W_in,rec_depth):
    cutsize = 2**rec_depth
    n = W_in.shape[0]
    W = np.zeros([n,n],dtype = complex)
    for i in range(cutsize):
        W[i*n/cutsize:(i+1)*n/cutsize,i*n/cutsize:(i+1)*n/cutsize] = butterfly(
                W[i*n/cutsize:(i+1)*n/cutsize,i*n/cutsize:(i+1)*n/cutsize])
    return W

for i in range(logn):
    W[i] = rearrange_rec(W[i],i)
    W[-1-i] = butterfly_rec(W[-1-i],i)
    
    #now i just have to write these functions 

W2 = [np.zeros([2*n,2*n],dtype = complex) for i in range(2*logn)]
for i in range(2*logn):
    W2[i][0:n,0:n] = W[i].real
    W2[i][n:2*n,n:2*n] = W[i].real
    W2[i][0:n,n:2*n] = -W[i].imag
    W2[i][n:2*n,0:n] = W[i].imag



comp_in = np.zeros([n,1],dtype = complex)
comp_in[0,0] = 2j
comp_out = comp_in
for i in range(2*logn):
    comp_out = np.matmul(W[i],comp_out)
    
real_in = np.zeros([2*n,1],dtype = np.float32)
real_in[8,0] = 2
real_out = real_in
for i in range(2*logn):
    real_out = np.matmul(W2[i],real_out) 
    
    
assert np.all(real_out[0:n] - comp_out.real.astype(complex) < 0.00000000001)
assert np.all(real_out[n:2*n] - comp_out.imag.astype(complex) < 0.00000000001)

    


    
     