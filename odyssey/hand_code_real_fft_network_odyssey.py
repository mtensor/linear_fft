#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun May  7 19:34:50 2017

@author: Maxwell
"""
import numpy as np

def hand_code_real_fft_network_fun(n,W_init_stddev):
    import numpy as np
    #must be power of 2
    logn = int(np.log2(n))
    
    W = [np.zeros([n,n],dtype = complex) for i in range(logn)]
    W_rearrange_list = [np.zeros([n,n],dtype = complex) for i in range(logn)]
    
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
    
    W_rearrange_compiled = np.identity(n,dtype = complex) 
    
    for i in range(logn):
        W_rearrange_list[i] = rearrange_rec(W[i],i)
        W[-1-i] = butterfly_rec(W[-1-i],i)
        
        #I'm flipping this now
        #W_rearrange_compiled = np.matmul(W_rearrange_compiled, W_rearrange_list[i])
        W_rearrange_compiled = np.matmul(W_rearrange_list[i], W_rearrange_compiled)

    W.insert(0, W_rearrange_compiled)
    
    assert len(W) == logn + 1
        
        #now i just have to write these functions 
    
    W2 = [np.zeros([2*n,2*n],dtype = np.float32) for i in range(logn + 1)]
    for i in range(logn + 1):
        
        W2[i][0:n,0:n] = W[i].real
        W2[i][n:2*n,n:2*n] = W[i].real
        W2[i][0:n,n:2*n] = -W[i].imag
        W2[i][n:2*n,0:n] = W[i].imag
        A = np.random.normal(scale=W_init_stddev,size=[2*n,2*n])
        W2[i] = W2[i] + A.astype(np.float32)      
    return W2

    
def hand_code_fun_layer_less(n,W_init_stddev):
    import numpy as np
    #grab a hand coded network with zero noise, combine first two layers
    W = hand_code_real_fft_network_fun(n,0)
    rearrange_layer = W.pop(0)
    W[0] = np.matmul(W[0],rearrange_layer)
    
    #add in the noise 
    for i in range(len(W)):
        A = np.random.normal(scale=W_init_stddev,size=[2*n,2*n])
        W[i] = W[i] + A.astype(np.float32)
        
    return W
    
    
     