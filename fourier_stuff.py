#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 12:01:51 2017

@author: Maxwell
"""

import numpy as np

def fourier_trans(x):
    if isinstance(x,np.ndarray):
        lenx = x.shape[1]
    elif isinstance(x,list):
        ft = []
        for i in range(len(x)):
            ft += [fourier_trans(x[i])]
        return ft
    else:
        raise TypeError("unsupported input type")
    
    if lenx % 2 != 0:
        raise ValueError("input to fourier_trans must have even length")

    collated_in = x[:,0:lenx/2] + 1j*x[:,lenx/2:lenx]
    raw_transform = np.fft.fft(collated_in,axis=1)
    split_ft = np.zeros(shape = x.shape)
    split_ft[:,0:lenx/2] = raw_transform.real
    split_ft[:,lenx/2:lenx] = raw_transform.imag
    assert not np.any(np.isnan(split_ft))
    return np.float32(split_ft)

#tests
n = 10;
testx = np.asarray([range(n)])
xout = fourier_trans(testx) 


test2 = np.zeros([2,10])
assert (test2 == fourier_trans(test2)).all()

test3 = np.vstack((testx,testx))
assert (fourier_trans(test3)[[0]] == xout).all()
assert (fourier_trans(test3)[[1]] == xout).all()




