#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 01:13:38 2017

@author: Maxwell
"""

import numpy as np
import tensorflow as tf

complex_n = 50
n = 2*complex_n
logn = int(np.ceil(np.log2(n)))
train_time = 10000
train_num = 1


#sanity checks
testw = tf.Variable(tf.zeros([n,n]))
y = l_0_norm(testw)

indices = tf.constant([[4], [3], [1], [7]])
updates = tf.constant([9, 10, 11, 12])
shape = tf.constant([8])
scatter = tf.scatter_nd(indices, updates, shape)
z = l_0_norm(tf.cast(scatter,tf.float32))


init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
curr_y = sess.run([y,scatter,z])
print(curr_y)


